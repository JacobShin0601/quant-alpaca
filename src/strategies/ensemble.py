"""
Ensemble Strategy Module
Dynamically selects and applies strategies based on market regime
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import json
import os
from datetime import datetime

try:
    from .base import BaseStrategy
    from ..actions.market_regime import MarketRegimeDetector, MarketRegime, RegimeIndicators
except ImportError:
    # Fallback for when module is run directly
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from strategies.base import BaseStrategy
    from actions.market_regime import MarketRegimeDetector, MarketRegime, RegimeIndicators


class RegimeStrategyConfig:
    """Configuration for regime-based strategy selection"""
    
    def __init__(self, config_path: str = "config/strategies/ensemble_config.json"):
        """Load ensemble configuration"""
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load configuration from file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default ensemble configuration"""
        return {
            "regime_strategies": {
                "trending_up": {
                    "primary": ["supertrend", "ichimoku", "donchian_channels", "aroon"],
                    "secondary": ["basic_momentum", "vwap", "fibonacci_retracement"],
                    "weights": {"primary": 0.7, "secondary": 0.3},
                    "risk_multiplier": 1.2
                },
                "trending_down": {
                    "primary": ["mean_reversion", "keltner_channels", "fibonacci_retracement"],
                    "secondary": ["bollinger_bands", "supertrend", "volume_profile"],
                    "weights": {"primary": 0.6, "secondary": 0.4},
                    "risk_multiplier": 0.8
                },
                "sideways": {
                    "primary": ["mean_reversion", "bollinger_bands", "keltner_channels", "volume_profile"],
                    "secondary": ["stochastic", "pairs", "ichimoku", "fibonacci_retracement"],
                    "weights": {"primary": 0.6, "secondary": 0.4},
                    "risk_multiplier": 1.0
                },
                "volatile": {
                    "primary": ["atr_breakout", "advanced_vwap", "volume_profile"],
                    "secondary": ["bollinger_bands", "keltner_channels", "aroon"],
                    "weights": {"primary": 0.5, "secondary": 0.5},
                    "risk_multiplier": 0.6
                }
            },
            "transition_rules": {
                "min_regime_duration": 10,  # Minimum candles before regime change
                "confidence_threshold": 0.6,  # Minimum confidence for regime
                "smooth_transition": True,  # Gradually shift between strategies
                "transition_periods": 5  # Periods for smooth transition
            },
            "risk_management": {
                "base_position_size": 0.25,  # 25% of capital per position
                "max_positions": 3,
                "stop_loss_multiplier": {
                    "trending_up": 1.5,
                    "trending_down": 1.0,
                    "sideways": 0.8,
                    "volatile": 0.6
                },
                "take_profit_multiplier": {
                    "trending_up": 2.0,
                    "trending_down": 1.2,
                    "sideways": 1.0,
                    "volatile": 0.8
                }
            },
            "performance_tracking": {
                "evaluate_period": 100,  # Candles
                "min_trades_for_evaluation": 10,
                "strategy_rotation": True,
                "underperform_threshold": -0.05  # -5% relative performance
            }
        }
    
    def get_regime_strategies(self, regime: MarketRegime) -> Dict[str, List[str]]:
        """Get strategies for a specific regime"""
        regime_key = regime.value
        if regime_key in self.config["regime_strategies"]:
            return self.config["regime_strategies"][regime_key]
        return {"primary": ["basic_momentum"], "secondary": [], "weights": {"primary": 1.0}}
    
    def get_strategy_config(self, strategy_name: str) -> Dict:
        """Load individual strategy configuration"""
        config_file = f"config/strategies/{strategy_name}.json"
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        
        # Default parameters for each strategy
        default_configs = {
            'ichimoku': {
                'tenkan_period': 9,
                'kijun_period': 26,
                'senkou_b_period': 52,
                'chikou_period': 26,
                'strategy_variant': 'classic',
                'use_volume': True,
                'volume_threshold': 1.2
            },
            'supertrend': {
                'atr_period': 10,
                'multiplier': 3.0,
                'strategy_variant': 'classic',
                'use_volume_analysis': True,
                'volume_threshold': 1.2,
                'use_confirmation': True
            },
            'atr_breakout': {
                'atr_period': 14,
                'atr_multiplier': 2.0,
                'lookback_period': 20,
                'strategy_variant': 'adaptive',
                'min_atr_pct': 0.5,
                'max_atr_pct': 5.0,
                'use_volume_confirmation': True,
                'volume_threshold': 1.5
            },
            'keltner_channels': {
                'ema_period': 20,
                'atr_period': 10,
                'multiplier': 2.0,
                'strategy_variant': 'mean_reversion',
                'squeeze_threshold': 0.015
            },
            'donchian_channels': {
                'upper_period': 20,
                'lower_period': 20,
                'middle_period': 10,
                'strategy_variant': 'breakout',
                'min_width_pct': 1.0,
                'use_volume': True,
                'volume_threshold': 1.2
            },
            'vwap': {
                'vwap_period': 20,
                'strategy_variant': 'mean_reversion',
                'vwap_threshold': 0.005,
                'volume_threshold': 1.2,
                'use_momentum': True
            },
            'advanced_vwap': {
                'vwap_period': 20,
                'adx_period': 14,
                'adx_threshold': 20,
                'profit_target_pct': 0.6,
                'stop_loss_pct': 0.3,
                'volatility_threshold': 0.15
            },
            'basic_momentum': {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'ma_short': 10,
                'ma_long': 30
            },
            'bollinger_bands': {
                'bb_period': 20,
                'bb_std_dev': 2.0,
                'lower_threshold': 0.1,
                'upper_threshold': 0.9,
                'use_rsi': True
            },
            'mean_reversion': {
                'bb_period': 20,
                'bb_std_dev': 2.0,
                'entry_zscore': 2.0,
                'exit_zscore': 0.5,
                'use_volume_filter': True
            },
            'macd': {
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9,
                'use_histogram_filter': True,
                'use_rsi_filter': True
            },
            'stochastic': {
                'k_period': 14,
                'd_period': 3,
                'smooth_k': 3,
                'oversold_level': 20,
                'overbought_level': 80,
                'use_volume_confirmation': True
            },
            'pairs': {
                'lookback_period': 60,
                'entry_threshold': 2.0,
                'exit_threshold': 0.5,
                'min_correlation': 0.7
            },
            'volume_profile': {
                'profile_period': 50,
                'num_bins': 20,
                'poc_threshold': 0.003,
                'min_volume_ratio': 1.2,
                'momentum_threshold': 0.002,
                'use_value_area': True
            },
            'fibonacci_retracement': {
                'swing_period': 20,
                'fib_proximity': 0.003,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'momentum_threshold': 0.001,
                'use_golden_ratio': True
            },
            'aroon': {
                'aroon_period': 25,
                'oscillator_threshold': 50,
                'momentum_threshold': 0.001,
                'use_volume_confirmation': True,
                'use_ma_confirmation': True,
                'use_trend_strength': True,
                'use_consolidation_breakout': True
            }
        }
        
        # Map strategy config names to base names
        base_name = self._get_base_strategy_name(strategy_name)
        return default_configs.get(base_name, {})


class StrategyPerformanceTracker:
    """Tracks performance of individual strategies within regimes"""
    
    def __init__(self, window: int = 100):
        """Initialize performance tracker"""
        self.window = window
        self.performance_history = {}
        self.trade_history = {}
        
    def update_performance(self, regime: str, strategy: str, 
                         trade_result: Dict[str, Any]):
        """Update strategy performance for a regime"""
        key = f"{regime}_{strategy}"
        
        if key not in self.performance_history:
            self.performance_history[key] = []
            self.trade_history[key] = []
        
        self.trade_history[key].append(trade_result)
        
        # Calculate recent performance
        recent_trades = self.trade_history[key][-self.window:]
        if recent_trades:
            win_rate = sum(1 for t in recent_trades if t['profit'] > 0) / len(recent_trades)
            avg_return = np.mean([t['return_pct'] for t in recent_trades])
            sharpe = self._calculate_sharpe(recent_trades)
            
            self.performance_history[key].append({
                'timestamp': trade_result['timestamp'],
                'win_rate': win_rate,
                'avg_return': avg_return,
                'sharpe': sharpe,
                'trade_count': len(recent_trades)
            })
    
    def get_strategy_performance(self, regime: str, strategy: str) -> Dict[str, float]:
        """Get current performance metrics for a strategy in a regime"""
        key = f"{regime}_{strategy}"
        
        if key not in self.performance_history or not self.performance_history[key]:
            return {'win_rate': 0.5, 'avg_return': 0, 'sharpe': 0, 'trade_count': 0}
        
        return self.performance_history[key][-1]
    
    def get_best_strategies(self, regime: str, top_n: int = 3) -> List[Tuple[str, float]]:
        """Get best performing strategies for a regime"""
        regime_strategies = {}
        
        for key in self.performance_history:
            if key.startswith(f"{regime}_"):
                strategy = key.replace(f"{regime}_", "")
                if self.performance_history[key]:
                    latest_perf = self.performance_history[key][-1]
                    # Score based on Sharpe ratio and win rate
                    score = latest_perf['sharpe'] * 0.6 + latest_perf['win_rate'] * 0.4
                    regime_strategies[strategy] = score
        
        # Sort by score
        sorted_strategies = sorted(regime_strategies.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        return sorted_strategies[:top_n]
    
    def _calculate_sharpe(self, trades: List[Dict]) -> float:
        """Calculate Sharpe ratio from trades"""
        if len(trades) < 2:
            return 0
        
        returns = [t['return_pct'] for t in trades]
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        # Annualized Sharpe (assuming minute data)
        return mean_return / std_return * np.sqrt(365 * 24 * 60)


class EnsembleStrategy(BaseStrategy):
    """
    Ensemble strategy that dynamically selects strategies based on market regime
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        """Initialize ensemble strategy"""
        super().__init__(parameters)
        
        # Store parameters for later use
        self.parameters = parameters
        
        # Initialize components
        self.regime_detector = MarketRegimeDetector(parameters.get('regime_config'))
        
        # Handle config_path which might be None during optimization
        config_path = parameters.get('config_path')
        if config_path is None:
            # Use default or create a dummy config
            self.config = RegimeStrategyConfig()
        else:
            self.config = RegimeStrategyConfig(config_path)
            
        self.performance_tracker = StrategyPerformanceTracker()
        
        # Check if we're in optimization mode (minimal parameters passed)
        self.optimization_mode = parameters.get('optimization_mode', False)
        # Don't override optimization_mode if it's already set to True
        if not self.optimization_mode:
            # Only check for required ensemble parameters
            required_params = ['confidence_threshold', 'min_regime_duration', 
                             'transition_periods', 'base_position_size']
            # Check if we have ONLY optimization parameters (and optimization_mode isn't explicitly set)
            has_only_optimization_params = all(p in parameters for p in required_params) and len(parameters) <= len(required_params) + 5
            if has_only_optimization_params:
                self.optimization_mode = True
            
        # Override config with optimization parameters if present
        if self.optimization_mode and hasattr(self.config, 'config'):
            # Update config with optimization parameters
            if 'confidence_threshold' in parameters:
                self.config.config['transition_rules']['confidence_threshold'] = parameters['confidence_threshold']
            if 'min_regime_duration' in parameters:
                self.config.config['transition_rules']['min_regime_duration'] = parameters['min_regime_duration']
            if 'transition_periods' in parameters:
                self.config.config['transition_rules']['transition_periods'] = parameters['transition_periods']
            if 'base_position_size' in parameters:
                self.config.config['risk_management']['base_position_size'] = parameters['base_position_size']
            if 'smooth_transition' in parameters:
                self.config.config['transition_rules']['smooth_transition'] = parameters['smooth_transition']
            if 'strategy_rotation' in parameters:
                self.config.config['performance_tracking']['strategy_rotation'] = parameters['strategy_rotation']
        
        # State management
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_confidence = 0.0
        self.regime_duration = 0
        self.active_strategies = {}
        self.strategy_instances = {}
        
        # Transition management
        self.in_transition = False
        self.transition_progress = 0
        self.previous_regime = None
        
        # Signal aggregation
        self.signal_history = []
        self.last_signal = 0
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators for all potential strategies"""
        try:
            # Detect current market regime
            regime, indicators = self.regime_detector.detect_regime(df)
            
            # Store regime information in dataframe
            df['regime'] = regime.value
            df['regime_confidence'] = max(indicators.regime_probability.values())
            
            # Update regime state
            self._update_regime_state(regime, indicators)
            
            # Load and initialize required strategies
            self._update_active_strategies(df)
            
            # In optimization mode, skip calculating sub-strategy indicators
            if self.optimization_mode:
                # Add only essential ensemble-specific indicators
                df['regime_adx'] = indicators.adx
                df['regime_volatility'] = indicators.volatility_ratio
                df['regime_trend_strength'] = indicators.trend_strength
                df['regime_choppiness'] = indicators.choppiness_index
                return df
            
            # Calculate indicators for all active strategies
            for strategy_name, strategy in self.active_strategies.items():
                try:
                    # Add strategy-specific prefix to avoid conflicts
                    strategy_df = strategy.calculate_indicators(df.copy())
                    
                    # Merge indicators with prefix
                    for col in strategy_df.columns:
                        if col not in ['open', 'high', 'low', 'close', 'volume', 'regime']:
                            df[f"{strategy_name}_{col}"] = strategy_df[col]
                except Exception as e:
                    # Log error but continue with other strategies
                    print(f"⚠️  Error calculating indicators for {strategy_name}: {e}")
                    continue
            
            # Add ensemble-specific indicators
            df['regime_adx'] = indicators.adx
            df['regime_volatility'] = indicators.volatility_ratio
            df['regime_trend_strength'] = indicators.trend_strength
            df['regime_choppiness'] = indicators.choppiness_index
            
            return df
        except Exception as e:
            # If any error occurs, return df with basic indicators
            print(f"⚠️  Error in ensemble calculate_indicators: {e}")
            # Add default values to prevent errors downstream
            df['regime'] = 'unknown'
            df['regime_confidence'] = 0.0
            df['regime_adx'] = 25.0
            df['regime_volatility'] = 1.0
            df['regime_trend_strength'] = 0.0
            df['regime_choppiness'] = 50.0
            return df
    
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate ensemble signals based on active strategies"""
        df['signal'] = 0
        df['signal_strength'] = 0.0
        df['signal_source'] = ''
        
        if self.current_regime == MarketRegime.UNKNOWN:
            return df
            
        # In optimization mode, generate simple signals based on regime
        if self.optimization_mode:
            return self._generate_optimization_signals(df, market)
        
        # Get signals from each active strategy
        strategy_signals = {}
        for strategy_name, strategy in self.active_strategies.items():
            strategy_df = strategy.generate_signals(df.copy(), market)
            if 'signal' in strategy_df and len(strategy_df) > 0:
                strategy_signals[strategy_name] = strategy_df['signal'].iloc[-1]
            else:
                strategy_signals[strategy_name] = 0
        
        # Aggregate signals based on regime configuration
        if len(df) > 0:
            final_signal, signal_strength, signal_source = self._aggregate_signals(
                strategy_signals, df.iloc[-1]
            )
        else:
            final_signal, signal_strength, signal_source = 0, 0.0, "no_data"
        
        # Apply risk management adjustments
        final_signal = self._apply_risk_management(final_signal, df)
        
        # Update dataframe
        if len(df) > 0:
            # Ensure columns exist before trying to access them
            if 'signal' not in df.columns:
                df['signal'] = 0
            if 'signal_strength' not in df.columns:
                df['signal_strength'] = 0.0
            if 'signal_source' not in df.columns:
                df['signal_source'] = ''
                
            df.iloc[-1, df.columns.get_loc('signal')] = final_signal
            df.iloc[-1, df.columns.get_loc('signal_strength')] = signal_strength
            df.iloc[-1, df.columns.get_loc('signal_source')] = signal_source
        
        # Store signal history
        if len(df) > 0:
            self.signal_history.append({
                'timestamp': df.index[-1],
                'regime': self.current_regime.value,
                'signal': final_signal,
                'strength': signal_strength,
                'source': signal_source,
                'strategies': strategy_signals
            })
        
        self.last_signal = final_signal
        
        return df
    
    def _update_regime_state(self, regime: MarketRegime, indicators: RegimeIndicators):
        """Update internal regime state"""
        if regime != self.current_regime:
            # Check if regime change is significant
            confidence = max(indicators.regime_probability.values())
            
            # Get config values safely
            confidence_threshold = self.config.config['transition_rules'].get('confidence_threshold', 0.6)
            
            if confidence >= confidence_threshold:
                # Check minimum duration
                min_duration = self.config.config['transition_rules'].get('min_regime_duration', 10)
                
                if self.regime_duration >= min_duration or self.current_regime == MarketRegime.UNKNOWN:
                    # Initiate regime change
                    self.previous_regime = self.current_regime
                    self.current_regime = regime
                    self.regime_duration = 0
                    self.in_transition = True
                    self.transition_progress = 0
                    
                    if not self.optimization_mode:
                        print(f"🔄 Regime change: {self.previous_regime.value} → {regime.value} "
                              f"(confidence: {confidence:.2%})")
        else:
            self.regime_duration += 1
            self.regime_confidence = max(indicators.regime_probability.values())
            
            # Update transition progress
            if self.in_transition:
                transition_periods = self.config.config['transition_rules'].get('transition_periods', 5)
                self.transition_progress = min(1.0, self.transition_progress + 1/transition_periods)
                
                if self.transition_progress >= 1.0:
                    self.in_transition = False
    
    def _update_active_strategies(self, df: pd.DataFrame):
        """Update active strategies based on current regime"""
        # Skip sub-strategy loading during optimization
        if self.optimization_mode:
            self.active_strategies = {}
            return
            
        regime_config = self.config.get_regime_strategies(self.current_regime)
        
        # Get all strategies for current regime
        all_strategies = regime_config['primary'] + regime_config.get('secondary', [])
        
        # Initialize new strategies
        for strategy_name in all_strategies:
            if strategy_name not in self.strategy_instances:
                # Load strategy configuration
                strategy_config = self.config.get_strategy_config(strategy_name)
                
                # Map config name to strategy class name
                base_strategy_name = self._get_base_strategy_name(strategy_name)
                
                # Initialize strategy
                try:
                    # Import get_strategy locally to avoid circular import
                    from .registry import get_strategy
                    
                    # Ensure strategy_config has all required parameters
                    # If config is empty, use a complete default config
                    if not strategy_config:
                        strategy_config = self._get_complete_default_config(base_strategy_name)
                    
                    self.strategy_instances[strategy_name] = get_strategy(
                        base_strategy_name, strategy_config
                    )
                except Exception as e:
                    print(f"⚠️  Failed to load strategy {strategy_name}: {e}")
                    continue
        
        # Update active strategies based on performance
        if self.config.config['performance_tracking']['strategy_rotation']:
            # Get best performing strategies for regime
            best_strategies = self.performance_tracker.get_best_strategies(
                self.current_regime.value, top_n=len(all_strategies)
            )
            
            # If we have performance data, use it
            if best_strategies and len(best_strategies) >= 2:
                self.active_strategies = {
                    strat: self.strategy_instances[strat] 
                    for strat, _ in best_strategies 
                    if strat in self.strategy_instances
                }
            else:
                # Use default configuration
                self.active_strategies = {
                    name: self.strategy_instances[name] 
                    for name in all_strategies 
                    if name in self.strategy_instances
                }
        else:
            # Use configuration without rotation
            self.active_strategies = {
                name: self.strategy_instances[name] 
                for name in all_strategies 
                if name in self.strategy_instances
            }
    
    def _get_base_strategy_name(self, config_name: str) -> str:
        """Map configuration name to base strategy class name"""
        # If config_name is already a base name, return it
        if config_name in ['basic_momentum', 'vwap', 'bollinger_bands', 'advanced_vwap',
                           'mean_reversion', 'macd', 'stochastic', 'pairs', 'ichimoku',
                           'supertrend', 'atr_breakout', 'keltner_channels', 'donchian_channels',
                           'volume_profile', 'fibonacci_retracement', 'aroon']:
            return config_name
            
        mapping = {
            'vwap_trend_following': 'vwap',
            'vwap_mean_reversion': 'vwap',
            'advanced_vwap_momentum': 'advanced_vwap',
            'advanced_vwap_bands': 'advanced_vwap',
            'bollinger_bands': 'bollinger_bands',
            'mean_reversion': 'mean_reversion',
            'basic_momentum': 'basic_momentum',
            'stochastic': 'stochastic',
            'macd': 'macd',
            'pairs': 'pairs',
            'ichimoku': 'ichimoku',
            'supertrend': 'supertrend',
            'atr_breakout': 'atr_breakout',
            'keltner_channels': 'keltner_channels',
            'donchian_channels': 'donchian_channels',
            'volume_profile': 'volume_profile',
            'fibonacci_retracement': 'fibonacci_retracement',
            'aroon': 'aroon'
        }
        
        return mapping.get(config_name, config_name)
    
    def _get_complete_default_config(self, strategy_name: str) -> Dict:
        """Get complete default configuration for any strategy"""
        # This ensures all strategies have required parameters
        default_configs = {
            'basic_momentum': {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'ma_short': 10,
                'ma_long': 30
            },
            'vwap': {
                'vwap_period': 20,
                'strategy_variant': 'mean_reversion',
                'vwap_threshold': 0.005,
                'volume_threshold': 1.2
            },
            'bollinger_bands': {
                'bb_period': 20,
                'bb_std_dev': 2.0,
                'lower_threshold': 0.1,
                'upper_threshold': 0.9
            },
            'advanced_vwap': {
                'vwap_period': 20,
                'adx_period': 14,
                'adx_threshold': 20,
                'profit_target_pct': 0.6,
                'stop_loss_pct': 0.3
            },
            'mean_reversion': {
                'bb_period': 20,
                'bb_std_dev': 2.0,
                'entry_zscore': 2.0,
                'exit_zscore': 0.5
            },
            'macd': {
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9
            },
            'stochastic': {
                'k_period': 14,
                'd_period': 3,
                'smooth_k': 3,
                'oversold_level': 20,
                'overbought_level': 80
            },
            'pairs': {
                'lookback_period': 60,
                'entry_threshold': 2.0,
                'exit_threshold': 0.5,
                'min_correlation': 0.7
            },
            'ichimoku': {
                'tenkan_period': 9,
                'kijun_period': 26,
                'senkou_b_period': 52,
                'chikou_period': 26
            },
            'supertrend': {
                'atr_period': 10,
                'multiplier': 3.0
            },
            'atr_breakout': {
                'atr_period': 14,
                'atr_multiplier': 2.0,
                'lookback_period': 20
            },
            'keltner_channels': {
                'ema_period': 20,
                'atr_period': 10,
                'multiplier': 2.0
            },
            'donchian_channels': {
                'upper_period': 20,
                'lower_period': 20,
                'middle_period': 10
            },
            'volume_profile': {
                'profile_period': 50,
                'num_bins': 20,
                'poc_threshold': 0.003
            },
            'fibonacci_retracement': {
                'swing_period': 20,
                'fib_proximity': 0.003,
                'rsi_oversold': 30,
                'rsi_overbought': 70
            },
            'aroon': {
                'aroon_period': 25,
                'oscillator_threshold': 50
            }
        }
        
        return default_configs.get(strategy_name, {})
    
    def _aggregate_signals(self, strategy_signals: Dict[str, int], 
                          current_data: pd.Series) -> Tuple[int, float, str]:
        """Aggregate signals from multiple strategies"""
        if not strategy_signals:
            return 0, 0.0, "no_signal"
        
        regime_config = self.config.get_regime_strategies(self.current_regime)
        weights = regime_config.get('weights', {'primary': 1.0})
        
        # Separate primary and secondary strategies
        primary_strategies = regime_config.get('primary', [])
        secondary_strategies = regime_config.get('secondary', [])
        
        # Calculate weighted signals
        weighted_sum = 0.0
        total_weight = 0.0
        signal_sources = []
        
        for strategy_name, signal in strategy_signals.items():
            if signal != 0:  # Only consider non-zero signals
                # Determine weight
                if strategy_name in primary_strategies:
                    weight = weights.get('primary', 0.7)
                elif strategy_name in secondary_strategies:
                    weight = weights.get('secondary', 0.3)
                else:
                    weight = 0.1  # Minimal weight for unknown strategies
                
                # Adjust weight based on transition
                if self.in_transition and self.previous_regime:
                    # Gradually shift weights during transition
                    prev_config = self.config.get_regime_strategies(self.previous_regime)
                    if strategy_name in prev_config.get('primary', []):
                        # Reduce weight of previous regime's strategies
                        weight *= (1 - self.transition_progress * 0.5)
                
                weighted_sum += signal * weight
                total_weight += weight
                signal_sources.append(strategy_name)
        
        # Calculate final signal
        if total_weight > 0:
            signal_strength = abs(weighted_sum / total_weight)
            
            # Determine final signal with threshold
            threshold = 0.5  # Minimum strength for signal
            if weighted_sum / total_weight > threshold:
                final_signal = 1
            elif weighted_sum / total_weight < -threshold:
                final_signal = -1
            else:
                final_signal = 0
            
            signal_source = ", ".join(signal_sources) if signal_sources else "no_consensus"
        else:
            final_signal = 0
            signal_strength = 0.0
            signal_source = "no_signal"
        
        return final_signal, signal_strength, signal_source
    
    def _apply_risk_management(self, signal: int, df: pd.DataFrame) -> int:
        """Apply regime-specific risk management to signals"""
        if signal == 0:
            return signal
        
        # Get regime-specific risk multiplier
        risk_config = self.config.config['risk_management']
        regime_risk = self.config.config['regime_strategies'][self.current_regime.value].get(
            'risk_multiplier', 1.0
        )
        
        # Check volatility conditions
        if len(df) > 0 and 'regime_volatility' in df.columns:
            current_volatility = df['regime_volatility'].iloc[-1]
        else:
            current_volatility = 1.0  # Default volatility
        
        # In high volatility, be more conservative
        if current_volatility > 2.0:
            # Reduce signal frequency
            if len(self.signal_history) > 0:
                last_signal_time = self.signal_history[-1]['timestamp']
                if len(df) > 0:
                    current_time = df.index[-1]
                else:
                    return 0
                
                # Minimum time between signals in high volatility
                min_signal_gap = 5  # 5 candles
                
                # Handle both datetime and int timestamps
                if isinstance(current_time, (int, float)):
                    current_ts = current_time
                else:
                    current_ts = current_time.timestamp() if hasattr(current_time, 'timestamp') else pd.Timestamp(current_time).timestamp()
                
                if isinstance(last_signal_time, (int, float)):
                    last_ts = last_signal_time
                else:
                    last_ts = last_signal_time.timestamp() if hasattr(last_signal_time, 'timestamp') else pd.Timestamp(last_signal_time).timestamp()
                
                if (current_ts - last_ts) / 60 < min_signal_gap:
                    return 0  # Skip signal
        
        # Check regime confidence
        if self.regime_confidence < 0.4:
            # Low confidence - only take high probability signals
            if len(df) > 0 and 'signal_strength' in df.columns:
                if df['signal_strength'].iloc[-1] < 0.7:
                    return 0
        
        # Apply regime-specific filters
        if self.current_regime == MarketRegime.TRENDING_DOWN:
            # In downtrend, be cautious with long positions
            if signal == 1:
                # Only take long if momentum is very strong
                if len(df) > 0 and 'regime_trend_strength' in df.columns:
                    if df['regime_trend_strength'].iloc[-1] < 0:
                        return 0
        
        elif self.current_regime == MarketRegime.VOLATILE:
            # In volatile regime, reduce position frequency
            if self.regime_duration < 5:
                # Wait for regime to stabilize
                return 0
        
        return signal
    
    def get_position_size(self, signal: int, current_price: float, 
                         portfolio_value: float) -> float:
        """Calculate position size based on regime and risk parameters"""
        if signal == 0:
            return 0
        
        risk_config = self.config.config.get('risk_management', {})
        base_size = risk_config.get('base_position_size', 0.25)
        
        # Adjust for regime
        if self.current_regime.value in self.config.config.get('regime_strategies', {}):
            regime_risk = self.config.config['regime_strategies'][self.current_regime.value].get(
                'risk_multiplier', 1.0
            )
        else:
            regime_risk = 1.0
        
        # Adjust for signal strength
        signal_strength = 1.0
        if self.signal_history and len(self.signal_history) > 0:
            signal_strength = self.signal_history[-1].get('strength', 1.0)
        
        # Calculate final position size
        position_size = base_size * regime_risk * signal_strength
        
        # Apply maximum position size
        max_position = 0.33  # Maximum 33% of portfolio
        position_size = min(position_size, max_position)
        
        # Convert to value
        position_value = portfolio_value * position_size
        
        return position_value / current_price if current_price > 0 else 0
    
    def get_stop_loss(self, entry_price: float, position_side: int) -> float:
        """Calculate stop loss based on regime"""
        risk_config = self.config.config.get('risk_management', {})
        sl_multipliers = risk_config.get('stop_loss_multiplier', {})
        
        # Get multiplier for current regime, default to 1.0
        if isinstance(sl_multipliers, dict):
            sl_multiplier = sl_multipliers.get(self.current_regime.value, 1.0)
        else:
            sl_multiplier = 1.0
        
        # Base stop loss (1.5% default)
        base_stop = 0.015
        
        # Adjust for regime
        stop_distance = base_stop * sl_multiplier
        
        if position_side == 1:  # Long position
            return entry_price * (1 - stop_distance)
        else:  # Short position (not used in spot trading)
            return entry_price * (1 + stop_distance)
    
    def get_take_profit(self, entry_price: float, position_side: int) -> float:
        """Calculate take profit based on regime"""
        risk_config = self.config.config.get('risk_management', {})
        tp_multipliers = risk_config.get('take_profit_multiplier', {})
        
        # Get multiplier for current regime, default to 1.5
        if isinstance(tp_multipliers, dict):
            tp_multiplier = tp_multipliers.get(self.current_regime.value, 1.5)
        else:
            tp_multiplier = 1.5
        
        # Base take profit (2% default)
        base_profit = 0.02
        
        # Adjust for regime
        profit_distance = base_profit * tp_multiplier
        
        if position_side == 1:  # Long position
            return entry_price * (1 + profit_distance)
        else:  # Short position (not used in spot trading)
            return entry_price * (1 - profit_distance)
    
    def _generate_optimization_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate simple signals for optimization mode"""
        # Use basic regime-based signals
        df['signal'] = 0
        
        if len(df) < 50:
            df['signal_strength'] = 0.0
            df['signal_source'] = 'insufficient_data'
            return df
            
        # Determine price column
        price_col = 'trade_price' if 'trade_price' in df.columns else 'close'
        
        # Calculate basic indicators
        df['ma20'] = df[price_col].rolling(20).mean()
        df['ma50'] = df[price_col].rolling(50).mean()
        df['returns'] = df[price_col].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Simple RSI calculation
        gains = df['returns'].where(df['returns'] > 0, 0)
        losses = -df['returns'].where(df['returns'] < 0, 0)
        avg_gains = gains.rolling(14).mean()
        avg_losses = losses.rolling(14).mean()
        rs = avg_gains / avg_losses
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Generate signals based on regime and basic indicators
        if self.current_regime == MarketRegime.TRENDING_UP:
            # Buy on dips in uptrend
            buy_condition = (df[price_col] < df['ma20'] * 0.99) & (df['rsi'] < 40)
            sell_condition = (df[price_col] > df['ma20'] * 1.03) | (df['rsi'] > 70)
            df.loc[buy_condition, 'signal'] = 1
            df.loc[sell_condition, 'signal'] = -1
            
        elif self.current_regime == MarketRegime.TRENDING_DOWN:
            # Be cautious in downtrend, look for oversold bounces
            buy_condition = (df['rsi'] < 20) & (df[price_col] < df['ma50'] * 0.95)
            sell_condition = (df[price_col] > df['ma20']) | (df['rsi'] > 60)
            df.loc[buy_condition, 'signal'] = 1
            df.loc[sell_condition, 'signal'] = -1
            
        elif self.current_regime == MarketRegime.SIDEWAYS:
            # Mean reversion in sideways market
            bb_std = df[price_col].rolling(20).std()
            bb_upper = df['ma20'] + 2 * bb_std
            bb_lower = df['ma20'] - 2 * bb_std
            
            buy_condition = (df[price_col] < bb_lower) & (df['rsi'] < 30)
            sell_condition = (df[price_col] > bb_upper) & (df['rsi'] > 70)
            df.loc[buy_condition, 'signal'] = 1
            df.loc[sell_condition, 'signal'] = -1
            
        else:  # VOLATILE or UNKNOWN
            # Use momentum in volatile regime
            momentum_period = 10
            momentum = df[price_col] / df[price_col].shift(momentum_period) - 1
            
            # Trade with the trend in volatile markets
            buy_condition = (momentum > 0.02) & (df['rsi'] < 60) & (df[price_col] > df['ma20'])
            sell_condition = (momentum < -0.01) | (df['rsi'] > 80) | (df[price_col] < df['ma20'] * 0.98)
            df.loc[buy_condition, 'signal'] = 1
            df.loc[sell_condition, 'signal'] = -1
        
        # Apply min_regime_duration filter if specified
        if 'min_regime_duration' in self.parameters and self.regime_duration < self.parameters['min_regime_duration']:
            # Reduce signal frequency but don't eliminate completely
            df['signal'] = df['signal'] * 0.5
            df.loc[df['signal'].abs() < 0.5, 'signal'] = 0
            df.loc[df['signal'] >= 0.5, 'signal'] = 1
            df.loc[df['signal'] <= -0.5, 'signal'] = -1
                
        # Set signal strength
        df['signal_strength'] = abs(df['signal']) * 0.7
        df['signal_source'] = 'optimization_mode'
        
        return df