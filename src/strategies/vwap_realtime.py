import pandas as pd
import numpy as np
from collections import deque
from typing import Dict, Any, Optional
from .vwap import VWAPStrategy


class RealtimeVWAPStrategy(VWAPStrategy):
    """VWAP strategy optimized for real-time trading with incremental updates"""
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(parameters)
        
        # Initialize circular buffers for efficient real-time calculation
        self.vwap_period = parameters['vwap_period']
        self.vwap_short_period = parameters.get('vwap_short_period', self.vwap_period // 2)
        self.volume_period = parameters.get('volume_period', 20)
        
        # Circular buffers for each market
        self.buffers = {}
        
    def initialize_market_buffer(self, market: str):
        """Initialize buffers for a specific market"""
        self.buffers[market] = {
            'price_volume': deque(maxlen=max(self.vwap_period, self.vwap_short_period)),
            'volume': deque(maxlen=max(self.vwap_period, self.vwap_short_period)),
            'typical_price': deque(maxlen=self.parameters.get('vwap_std_period', self.vwap_period)),
            'trade_price': deque(maxlen=self.parameters.get('momentum_period', 14)),
            'volume_for_ma': deque(maxlen=self.volume_period),
            # Pre-calculated sums for O(1) updates
            'price_volume_sum': 0.0,
            'volume_sum': 0.0,
            'price_volume_sum_short': 0.0,
            'volume_sum_short': 0.0,
            # Current indicators
            'vwap': None,
            'vwap_short': None,
            'vwap_deviation': None,
            'volume_ma': None,
            'volume_ratio': None
        }
    
    def update_indicators_incremental(self, market: str, new_candle: Dict[str, float]) -> Dict[str, float]:
        """Update indicators incrementally with new candle data"""
        
        # Initialize buffer if needed
        if market not in self.buffers:
            self.initialize_market_buffer(market)
        
        buffer = self.buffers[market]
        
        # Calculate new values
        typical_price = (new_candle['high_price'] + new_candle['low_price'] + new_candle['trade_price']) / 3
        price_volume = typical_price * new_candle['candle_acc_trade_volume']
        volume = new_candle['candle_acc_trade_volume']
        
        # Update circular buffers
        buffer['typical_price'].append(typical_price)
        buffer['trade_price'].append(new_candle['trade_price'])
        buffer['volume_for_ma'].append(volume)
        
        # Update VWAP calculation with sliding window
        if len(buffer['price_volume']) == buffer['price_volume'].maxlen:
            # Remove oldest value from sum
            buffer['price_volume_sum'] -= buffer['price_volume'][0]
            buffer['volume_sum'] -= buffer['volume'][0]
        
        buffer['price_volume'].append(price_volume)
        buffer['volume'].append(volume)
        buffer['price_volume_sum'] += price_volume
        buffer['volume_sum'] += volume
        
        # Calculate VWAP
        if buffer['volume_sum'] > 0:
            buffer['vwap'] = buffer['price_volume_sum'] / buffer['volume_sum']
        else:
            buffer['vwap'] = new_candle['trade_price']
        
        # Update short VWAP if different period
        if self.vwap_short_period != self.vwap_period:
            # Calculate short VWAP sum from the last N elements
            short_window = min(len(buffer['price_volume']), self.vwap_short_period)
            buffer['price_volume_sum_short'] = sum(list(buffer['price_volume'])[-short_window:])
            buffer['volume_sum_short'] = sum(list(buffer['volume'])[-short_window:])
            
            if buffer['volume_sum_short'] > 0:
                buffer['vwap_short'] = buffer['price_volume_sum_short'] / buffer['volume_sum_short']
            else:
                buffer['vwap_short'] = new_candle['trade_price']
        else:
            buffer['vwap_short'] = buffer['vwap']
        
        # Calculate deviations
        if buffer['vwap'] and buffer['vwap'] > 0:
            buffer['vwap_deviation'] = (new_candle['trade_price'] - buffer['vwap']) / buffer['vwap']
        else:
            buffer['vwap_deviation'] = 0
        
        # Calculate volume MA and ratio
        if len(buffer['volume_for_ma']) > 0:
            buffer['volume_ma'] = sum(buffer['volume_for_ma']) / len(buffer['volume_for_ma'])
            if buffer['volume_ma'] > 0:
                buffer['volume_ratio'] = volume / buffer['volume_ma']
            else:
                buffer['volume_ratio'] = 1.0
        else:
            buffer['volume_ma'] = volume
            buffer['volume_ratio'] = 1.0
        
        # Return current indicators
        return {
            'vwap': buffer['vwap'],
            'vwap_short': buffer['vwap_short'],
            'vwap_deviation': buffer['vwap_deviation'],
            'volume_ratio': buffer['volume_ratio'],
            'volume_ma': buffer['volume_ma']
        }
    
    def generate_signal_realtime(self, market: str, new_candle: Dict[str, float]) -> int:
        """Generate trading signal for new candle in real-time"""
        
        # Update indicators incrementally
        indicators = self.update_indicators_incremental(market, new_candle)
        
        # Check if we have enough data
        buffer = self.buffers[market]
        if len(buffer['price_volume']) < self.vwap_period:
            return 0  # Not enough data yet
        
        # Strategy parameters
        vwap_threshold = self.parameters.get('vwap_threshold', 0.005)
        volume_threshold = self.parameters.get('volume_threshold', 1.2)
        
        # Basic conditions
        price_below_vwap = indicators['vwap_deviation'] < -vwap_threshold
        price_above_vwap = indicators['vwap_deviation'] > vwap_threshold
        high_volume = indicators['volume_ratio'] > volume_threshold
        
        # Strategy variant logic
        strategy_variant = self.parameters.get('strategy_variant', 'mean_reversion')
        
        if strategy_variant == 'mean_reversion':
            if price_below_vwap and high_volume:
                return 1  # Buy signal
            elif price_above_vwap and high_volume:
                return -1  # Sell signal
                
        elif strategy_variant == 'trend_following':
            vwap_uptrend = indicators['vwap_short'] > indicators['vwap']
            vwap_downtrend = indicators['vwap_short'] < indicators['vwap']
            
            if vwap_uptrend and new_candle['trade_price'] > indicators['vwap'] and high_volume:
                return 1  # Buy signal
            elif vwap_downtrend and new_candle['trade_price'] < indicators['vwap'] and high_volume:
                return -1  # Sell signal
                
        elif strategy_variant == 'breakout':
            strong_breakout_up = indicators['vwap_deviation'] > (vwap_threshold * 2)
            strong_breakout_down = indicators['vwap_deviation'] < -(vwap_threshold * 2)
            
            if strong_breakout_up and high_volume:
                return 1  # Buy signal
            elif strong_breakout_down and high_volume:
                return -1  # Sell signal
        
        return 0  # Hold signal
    
    def get_current_indicators(self, market: str) -> Optional[Dict[str, float]]:
        """Get current indicator values for a market"""
        if market not in self.buffers:
            return None
            
        buffer = self.buffers[market]
        return {
            'vwap': buffer['vwap'],
            'vwap_short': buffer['vwap_short'],
            'vwap_deviation': buffer['vwap_deviation'],
            'volume_ratio': buffer['volume_ratio'],
            'volume_ma': buffer['volume_ma'],
            'data_points': len(buffer['price_volume'])
        }
    
    def reset_market_buffer(self, market: str):
        """Reset buffer for a specific market (e.g., at day start)"""
        if market in self.buffers:
            self.initialize_market_buffer(market)