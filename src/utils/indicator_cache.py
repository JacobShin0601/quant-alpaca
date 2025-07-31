#!/usr/bin/env python3
"""
Indicator Cache System for Performance Optimization
Caches calculated indicators to avoid redundant calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import hashlib
import pickle
import threading
import time


class IndicatorCache:
    """
    High-performance indicator caching system
    Stores calculated indicators and updates incrementally
    """
    
    def __init__(self, max_cache_size: int = 1000, cache_ttl_hours: int = 24):
        """
        Initialize indicator cache
        
        Args:
            max_cache_size: Maximum number of cache entries
            cache_ttl_hours: Cache time-to-live in hours
        """
        self.cache = {}  # {cache_key: CacheEntry}
        self.max_cache_size = max_cache_size
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.lock = threading.RLock()
        self.access_times = {}  # Track access times for LRU eviction
        
    def _generate_cache_key(self, market: str, strategy_name: str, 
                          parameters: Dict[str, Any], data_hash: str) -> str:
        """Generate unique cache key"""
        # Include strategy parameters in key
        param_str = str(sorted(parameters.items())) if parameters else ""
        
        # Create composite key
        key_components = f"{market}_{strategy_name}_{param_str}_{data_hash}"
        
        # Hash for consistent key length
        return hashlib.md5(key_components.encode()).hexdigest()
    
    def _hash_dataframe(self, df: pd.DataFrame) -> str:
        """Generate hash for DataFrame to detect changes"""
        try:
            # Use last 10 rows for hash to detect new data
            if len(df) > 10:
                hash_data = df.tail(10)
            else:
                hash_data = df
                
            # Create hash from essential columns
            key_columns = ['trade_price', 'high_price', 'low_price', 'candle_acc_trade_volume']
            available_columns = [col for col in key_columns if col in hash_data.columns]
            
            if not available_columns:
                # Fallback to all columns
                hash_str = str(hash_data.values.tobytes())
            else:
                hash_str = str(hash_data[available_columns].values.tobytes())
            
            return hashlib.md5(hash_str.encode()).hexdigest()[:16]
            
        except Exception:
            # Fallback: use timestamp and length
            return f"{len(df)}_{int(time.time())}"
    
    def get_cached_indicators(self, market: str, strategy_name: str, 
                            parameters: Dict[str, Any], df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Retrieve cached indicators if available and valid
        
        Returns:
            Cached DataFrame with indicators or None if not available
        """
        with self.lock:
            try:
                data_hash = self._hash_dataframe(df)
                cache_key = self._generate_cache_key(market, strategy_name, parameters, data_hash)
                
                if cache_key not in self.cache:
                    return None
                
                cache_entry = self.cache[cache_key]
                
                # Check if cache is still valid
                if datetime.now() - cache_entry['timestamp'] > self.cache_ttl:
                    del self.cache[cache_key]
                    if cache_key in self.access_times:
                        del self.access_times[cache_key]
                    return None
                
                # Update access time for LRU
                self.access_times[cache_key] = datetime.now()
                
                cached_df = cache_entry['data']
                
                # Verify cache integrity
                if len(cached_df) != len(df):
                    # Data length mismatch, invalidate cache
                    del self.cache[cache_key]
                    if cache_key in self.access_times:
                        del self.access_times[cache_key]
                    return None
                
                return cached_df.copy()
                
            except Exception as e:
                print(f"Error retrieving cached indicators: {e}")
                return None
    
    def cache_indicators(self, market: str, strategy_name: str, 
                        parameters: Dict[str, Any], df: pd.DataFrame, 
                        indicators_df: pd.DataFrame) -> None:
        """
        Cache calculated indicators
        
        Args:
            market: Market symbol
            strategy_name: Strategy name
            parameters: Strategy parameters
            df: Original price data
            indicators_df: DataFrame with calculated indicators
        """
        with self.lock:
            try:
                data_hash = self._hash_dataframe(df)
                cache_key = self._generate_cache_key(market, strategy_name, parameters, data_hash)
                
                # Store in cache
                self.cache[cache_key] = {
                    'data': indicators_df.copy(),
                    'timestamp': datetime.now(),
                    'market': market,
                    'strategy': strategy_name,
                    'data_length': len(df)
                }
                
                self.access_times[cache_key] = datetime.now()
                
                # Evict old entries if cache is full
                self._evict_if_needed()
                
            except Exception as e:
                print(f"Error caching indicators: {e}")
    
    def _evict_if_needed(self):
        """Evict least recently used entries if cache is full"""
        while len(self.cache) > self.max_cache_size:
            # Find least recently accessed entry
            if not self.access_times:
                # Fallback: clear all
                self.cache.clear()
                break
                
            oldest_key = min(self.access_times.keys(), 
                           key=lambda k: self.access_times[k])
            
            # Remove oldest entry
            if oldest_key in self.cache:
                del self.cache[oldest_key]
            if oldest_key in self.access_times:
                del self.access_times[oldest_key]
    
    def invalidate_market(self, market: str) -> None:
        """Invalidate all cache entries for a specific market"""
        with self.lock:
            keys_to_remove = []
            for key, entry in self.cache.items():
                if entry.get('market') == market:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                if key in self.cache:
                    del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
    
    def clear_expired_entries(self) -> int:
        """Clear expired cache entries and return count of removed entries"""
        with self.lock:
            current_time = datetime.now()
            expired_keys = []
            
            for key, entry in self.cache.items():
                if current_time - entry['timestamp'] > self.cache_ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                if key in self.cache:
                    del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
            
            return len(expired_keys)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                'total_entries': len(self.cache),
                'max_size': self.max_cache_size,
                'cache_utilization': len(self.cache) / self.max_cache_size,
                'markets': list(set(entry.get('market', 'unknown') for entry in self.cache.values())),
                'strategies': list(set(entry.get('strategy', 'unknown') for entry in self.cache.values()))
            }


class IncrementalIndicatorCalculator:
    """
    Optimized indicator calculator that updates indicators incrementally
    """
    
    def __init__(self, cache: IndicatorCache):
        self.cache = cache
        
    def calculate_indicators_optimized(self, strategy, market: str, 
                                     df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicators with caching optimization
        
        Args:
            strategy: Strategy instance with calculate_indicators method
            market: Market symbol
            df: Price data DataFrame
            
        Returns:
            DataFrame with indicators calculated
        """
        try:
            strategy_name = strategy.__class__.__name__
            parameters = getattr(strategy, 'parameters', {})
            
            # Try to get from cache first
            cached_result = self.cache.get_cached_indicators(
                market, strategy_name, parameters, df
            )
            
            if cached_result is not None:
                return cached_result
            
            # Calculate indicators using strategy
            result_df = strategy.calculate_indicators(df.copy())
            
            # Cache the result
            self.cache.cache_indicators(
                market, strategy_name, parameters, df, result_df
            )
            
            return result_df
            
        except Exception as e:
            print(f"Error in optimized indicator calculation: {e}")
            # Fallback to direct calculation
            return strategy.calculate_indicators(df.copy())
    
    def update_indicators_incremental(self, strategy, market: str, 
                                    existing_df: pd.DataFrame, 
                                    new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Update indicators incrementally by only recalculating necessary portions
        
        Args:
            strategy: Strategy instance
            market: Market symbol  
            existing_df: DataFrame with existing indicators
            new_data: New price data to append
            
        Returns:
            Updated DataFrame with indicators
        """
        try:
            if new_data.empty:
                return existing_df
            
            # Combine old and new data
            combined_df = pd.concat([existing_df, new_data], ignore_index=False)
            
            # Remove duplicates based on index
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df = combined_df.sort_index()
            
            # For most indicators, we need to recalculate the last portion
            # due to rolling window dependencies
            lookback_periods = max(
                getattr(strategy, 'parameters', {}).get('ma_period', 20),
                getattr(strategy, 'parameters', {}).get('rsi_period', 14),
                50  # Default safe lookback
            )
            
            # Only recalculate if we have enough data
            if len(existing_df) > lookback_periods:
                # Keep the stable part, recalculate the tail
                stable_part = existing_df.iloc[:-lookback_periods].copy()
                
                # Recalculate the tail portion with new data
                tail_data = combined_df.iloc[-(lookback_periods + len(new_data)):].copy()
                updated_tail = strategy.calculate_indicators(tail_data)
                
                # Combine stable part with updated tail
                result_df = pd.concat([stable_part, updated_tail.iloc[lookback_periods:]])
            else:
                # Not enough data for optimization, recalculate all
                result_df = strategy.calculate_indicators(combined_df)
            
            # Update cache
            strategy_name = strategy.__class__.__name__
            parameters = getattr(strategy, 'parameters', {})
            self.cache.cache_indicators(
                market, strategy_name, parameters, combined_df, result_df
            )
            
            return result_df
            
        except Exception as e:
            print(f"Error in incremental indicator update: {e}")
            # Fallback to full recalculation
            combined_df = pd.concat([existing_df, new_data], ignore_index=False)
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            return strategy.calculate_indicators(combined_df)


# Global cache instance
_global_indicator_cache = IndicatorCache(max_cache_size=500, cache_ttl_hours=12)
_global_calculator = IncrementalIndicatorCalculator(_global_indicator_cache)


def get_global_indicator_cache() -> IndicatorCache:
    """Get the global indicator cache instance"""
    return _global_indicator_cache


def get_global_calculator() -> IncrementalIndicatorCalculator:
    """Get the global incremental calculator instance"""
    return _global_calculator


def cleanup_cache_periodically():
    """Background task to clean up expired cache entries"""
    while True:
        try:
            expired_count = _global_indicator_cache.clear_expired_entries()
            if expired_count > 0:
                print(f"Cleaned up {expired_count} expired cache entries")
            time.sleep(3600)  # Run every hour
        except Exception as e:
            print(f"Error in cache cleanup: {e}")
            time.sleep(3600)


# Start background cleanup if this module is imported
import atexit
import threading

_cleanup_thread = threading.Thread(target=cleanup_cache_periodically, daemon=True)
_cleanup_thread.start()

def _cleanup_on_exit():
    """Cleanup function called on exit"""
    print("Indicator cache cleanup on exit")

atexit.register(_cleanup_on_exit)