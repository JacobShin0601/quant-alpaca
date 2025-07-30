"""
Adaptive Ensemble Strategy
Dynamically adjusts strategy weights based on recent performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import os
from datetime import datetime
from collections import deque

try:
    from .base import BaseStrategy
    from ..actions.market_regime import MarketRegimeDetector, MarketRegime
    from .registry import get_strategy
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from strategies.base import BaseStrategy
    from actions.market_regime import MarketRegimeDetector, MarketRegime
    from strategies.registry import get_strategy


class PerformanceTracker:
    """Tracks strategy performance for adaptive weighting"""
    
    def __init__(self, window: int = 100):
        self.window = window
        self.returns = {}  # strategy -> deque of returns
        self.trades = {}   # strategy -> deque of trade results
        self.positions = {}  # Track open positions per strategy
        
    def add_trade(self, strategy: str, entry_price: float, exit_price: float, 
                  signal: int, timestamp: pd.Timestamp):
        """Record a completed trade"""
        if strategy not in self.trades:
            self.trades[strategy] = deque(maxlen=self.window)
        
        return_pct = (exit_price - entry_price) / entry_price * signal
        
        self.trades[strategy].append({
            'timestamp': timestamp,
            'return': return_pct,
            'win': return_pct > 0
        })
        
    def add_return(self, strategy: str, return_pct: float):
        """Add a return observation"""
        if strategy not in self.returns:
            self.returns[strategy] = deque(maxlen=self.window)
        
        self.returns[strategy].append(return_pct)
    
    def get_performance_score(self, strategy: str) -> float:
        """Calculate performance score for a strategy"""
        if strategy not in self.returns or len(self.returns[strategy]) < 10:
            return 0.5  # Neutral score for new strategies
        
        returns = list(self.returns[strategy])
        
        # Calculate Sharpe-like ratio
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return > 0:
            sharpe = mean_return / std_return * np.sqrt(252)  # Annualized
        else:
            sharpe = mean_return * 100
        
        # Calculate win rate if we have trade data
        win_rate = 0.5
        if strategy in self.trades and len(self.trades[strategy]) >= 5:
            wins = sum(1 for t in self.trades[strategy] if t['win'])
            win_rate = wins / len(self.trades[strategy])
        
        # Combine metrics
        score = 0.6 * sharpe + 0.4 * (win_rate - 0.5) * 10
        
        # Normalize to [0, 1]
        return max(0, min(1, 0.5 + score / 10))


class AdaptiveEnsembleStrategy(BaseStrategy):
    """
    Adaptive ensemble that dynamically adjusts weights based on performance
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        """Initialize adaptive ensemble strategy"""
        super().__init__(parameters)
        
        # Load optimized parameters
        self.optimized_params = self._load_optimized_parameters()
        
        # Initialize components
        self.regime_detector = MarketRegimeDetector(parameters.get('regime_config'))
        self.performance_tracker = PerformanceTracker(
            window=parameters.get('performance_window', 100)
        )
        
        # Current market
        self.current_market = None
        
        # Strategy pool by regime
        self.strategy_pools = {
            MarketRegime.TRENDING_UP: ['supertrend', 'ichimoku', 'vwap', 'macd', 'aroon'],
            MarketRegime.TRENDING_DOWN: ['mean_reversion', 'bollinger_bands', 'keltner_channels', 
                                        'fibonacci_retracement'],
            MarketRegime.SIDEWAYS: ['mean_reversion', 'bollinger_bands', 'volume_profile', 
                                   'keltner_channels', 'stochastic'],
            MarketRegime.VOLATILE: ['atr_breakout', 'bollinger_bands', 'volume_profile', 
                                   'supertrend'],
            MarketRegime.UNKNOWN: ['vwap', 'bollinger_bands', 'mean_reversion']
        }
        
        # Adaptive parameters
        self.min_weight = parameters.get('min_weight', 0.05)
        self.max_weight = parameters.get('max_weight', 0.4)
        self.rebalance_frequency = parameters.get('rebalance_frequency', 50)
        self.min_strategies = parameters.get('min_strategies', 2)
        self.max_strategies = parameters.get('max_strategies', 4)
        
        # State
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_confidence = 0.0
        self.strategy_instances = {}
        self.active_strategies = {}
        self.current_weights = {}
        self.rebalance_counter = 0
        self.last_prices = {}
        
    def _load_optimized_parameters(self) -> Dict[str, Dict[str, Dict]]:
        """Load pre-optimized parameters"""
        optimized_params = {}
        optimization_dir = "results/optimization"
        
        if os.path.exists(optimization_dir):
            for strategy_dir in os.listdir(optimization_dir):
                strategy_path = os.path.join(optimization_dir, strategy_dir)
                
                if os.path.isdir(strategy_path):
                    strategy_name = strategy_dir
                    optimized_params[strategy_name] = {}
                    
                    for file in os.listdir(strategy_path):
                        if file.endswith("_optimized.json") and not file.startswith("all_markets"):
                            market = file.replace("_optimized.json", "")
                            
                            with open(os.path.join(strategy_path, file), 'r') as f:
                                data = json.load(f)
                                optimized_params[strategy_name][market] = data["optimized_params"]
        
        return optimized_params
    
    def _get_strategy_instance(self, strategy_name: str, market: str) -> Optional[BaseStrategy]:
        """Get or create strategy instance"""
        key = f"{strategy_name}_{market}"
        
        if key not in self.strategy_instances:
            # Get optimized parameters
            if (strategy_name in self.optimized_params and 
                market in self.optimized_params[strategy_name]):
                params = self.optimized_params[strategy_name][market]
            else:
                params = {}
            
            try:
                self.strategy_instances[key] = get_strategy(strategy_name, params)
            except Exception as e:
                self.logger.error(f"Failed to load strategy {strategy_name}: {e}")
                return None
        
        return self.strategy_instances[key]
    
    def _update_active_strategies(self, market: str):
        """Update active strategies based on performance"""
        # Get strategy pool for current regime
        pool = self.strategy_pools.get(self.current_regime, self.strategy_pools[MarketRegime.UNKNOWN])
        
        # Calculate performance scores
        scores = {}
        for strategy_name in pool:
            score = self.performance_tracker.get_performance_score(strategy_name)
            scores[strategy_name] = score
        
        # Select top performing strategies
        sorted_strategies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        n_strategies = min(self.max_strategies, max(self.min_strategies, len(sorted_strategies)))
        
        self.active_strategies = {}
        for strategy_name, score in sorted_strategies[:n_strategies]:
            strategy = self._get_strategy_instance(strategy_name, market)
            if strategy:
                self.active_strategies[strategy_name] = strategy
        
        # Update weights
        self._update_weights()
    
    def _update_weights(self):
        """Calculate adaptive weights based on performance"""
        if not self.active_strategies:
            return
        
        # Get performance scores
        scores = {}
        total_score = 0
        
        for strategy_name in self.active_strategies:
            score = self.performance_tracker.get_performance_score(strategy_name)
            scores[strategy_name] = max(0.1, score)  # Minimum score to avoid zero weights
            total_score += scores[strategy_name]
        
        # Calculate weights with constraints
        self.current_weights = {}
        
        for strategy_name, score in scores.items():
            # Raw weight based on performance
            raw_weight = score / total_score
            
            # Apply min/max constraints
            weight = max(self.min_weight, min(self.max_weight, raw_weight))
            self.current_weights[strategy_name] = weight
        
        # Normalize weights to sum to 1
        total_weight = sum(self.current_weights.values())
        if total_weight > 0:
            for strategy_name in self.current_weights:
                self.current_weights[strategy_name] /= total_weight
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators"""
        # Detect regime
        regime, indicators = self.regime_detector.detect_regime(df)
        
        df['regime'] = regime.value
        df['regime_confidence'] = max(indicators.regime_probability.values())
        
        self.current_regime = regime
        self.regime_confidence = max(indicators.regime_probability.values())
        
        # Update active strategies periodically
        if self.current_market:
            self.rebalance_counter += 1
            if (self.rebalance_counter >= self.rebalance_frequency or 
                not self.active_strategies):
                self._update_active_strategies(self.current_market)
                self.rebalance_counter = 0
            
            # Calculate indicators for active strategies
            for strategy_name, strategy in self.active_strategies.items():
                try:
                    strategy_df = strategy.calculate_indicators(df.copy())
                    
                    for col in strategy_df.columns:
                        if col not in ['open', 'high', 'low', 'close', 'volume', 'regime',
                                     'trade_price', 'high_price', 'low_price', 'opening_price',
                                     'candle_acc_trade_volume', 'candle_acc_trade_price']:
                            df[f"{strategy_name}_{col}"] = strategy_df[col]
                except Exception as e:
                    self.logger.error(f"Error calculating indicators for {strategy_name}: {e}")
        
        # Add regime indicators
        df['regime_adx'] = indicators.adx
        df['regime_volatility'] = indicators.volatility_ratio
        
        return df
    
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate signals with performance tracking"""
        self.current_market = market
        
        df['signal'] = 0
        df['signal_strength'] = 0.0
        df['signal_source'] = ''
        
        if not self.active_strategies or len(df) < 2:
            return df
        
        # Track performance
        if market in self.last_prices and len(df) > 1:
            current_price = df['trade_price'].iloc[-1] if 'trade_price' in df.columns else df['close'].iloc[-1]
            last_price = self.last_prices[market]
            
            return_pct = (current_price - last_price) / last_price
            
            # Update performance for all active strategies
            for strategy_name in self.active_strategies:
                self.performance_tracker.add_return(strategy_name, return_pct)
        
        # Collect signals
        strategy_signals = {}
        
        for strategy_name, strategy in self.active_strategies.items():
            try:
                strategy_df = strategy.generate_signals(df.copy(), market)
                
                if 'signal' in strategy_df and len(strategy_df) > 0:
                    strategy_signals[strategy_name] = strategy_df['signal'].iloc[-1]
                else:
                    strategy_signals[strategy_name] = 0
            except Exception as e:
                self.logger.error(f"Error generating signals for {strategy_name}: {e}")
                strategy_signals[strategy_name] = 0
        
        # Aggregate signals
        if strategy_signals and len(df) > 0:
            final_signal, signal_strength, signal_source = self._aggregate_signals(strategy_signals)
            
            df.iloc[-1, df.columns.get_loc('signal')] = final_signal
            df.iloc[-1, df.columns.get_loc('signal_strength')] = signal_strength
            df.iloc[-1, df.columns.get_loc('signal_source')] = signal_source
        
        # Update last price
        if len(df) > 0:
            self.last_prices[market] = df['trade_price'].iloc[-1] if 'trade_price' in df.columns else df['close'].iloc[-1]
        
        return df
    
    def _aggregate_signals(self, signals: Dict[str, int]) -> Tuple[int, float, str]:
        """Aggregate signals using adaptive weights"""
        if not signals or not self.current_weights:
            return 0, 0.0, "no_signal"
        
        weighted_sum = 0.0
        total_weight = 0.0
        signal_sources = []
        
        for strategy_name, signal in signals.items():
            if signal != 0 and strategy_name in self.current_weights:
                weight = self.current_weights[strategy_name]
                weighted_sum += signal * weight
                total_weight += weight
                
                # Track performance score for display
                score = self.performance_tracker.get_performance_score(strategy_name)
                signal_sources.append(f"{strategy_name}({weight:.2f},s:{score:.2f})")
        
        if total_weight > 0:
            normalized_signal = weighted_sum / total_weight
            signal_strength = abs(normalized_signal)
            
            # Adaptive threshold based on regime and performance
            base_threshold = 0.5
            if self.current_regime == MarketRegime.VOLATILE:
                base_threshold = 0.7
            elif self.current_regime == MarketRegime.TRENDING_UP:
                base_threshold = 0.4
            
            # Adjust threshold based on average performance
            avg_score = np.mean([self.performance_tracker.get_performance_score(s) 
                               for s in self.active_strategies])
            threshold = base_threshold * (1.5 - avg_score)
            
            if normalized_signal > threshold:
                final_signal = 1
            elif normalized_signal < -threshold:
                final_signal = -1
            else:
                final_signal = 0
            
            signal_source = ", ".join(signal_sources)
        else:
            final_signal = 0
            signal_strength = 0.0
            signal_source = "no_active_signals"
        
        return final_signal, signal_strength, signal_source
    
    def get_position_size(self, signal: int, current_price: float, 
                         portfolio_value: float) -> float:
        """Calculate position size with performance adjustment"""
        if signal == 0:
            return 0
        
        # Base size
        base_size = 0.25
        
        # Regime adjustment
        regime_multipliers = {
            MarketRegime.TRENDING_UP: 1.2,
            MarketRegime.TRENDING_DOWN: 0.8,
            MarketRegime.SIDEWAYS: 1.0,
            MarketRegime.VOLATILE: 0.5,
            MarketRegime.UNKNOWN: 0.6
        }
        
        regime_mult = regime_multipliers.get(self.current_regime, 0.8)
        
        # Performance adjustment
        if self.active_strategies:
            avg_performance = np.mean([
                self.performance_tracker.get_performance_score(s) 
                for s in self.active_strategies
            ])
            performance_mult = 0.5 + avg_performance  # 0.5 to 1.5
        else:
            performance_mult = 0.8
        
        # Calculate final size
        position_size = base_size * regime_mult * performance_mult
        position_size = min(0.4, position_size)  # Max 40% of portfolio
        
        position_value = portfolio_value * position_size
        
        return position_value / current_price if current_price > 0 else 0