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
        """Simplified strategy configuration loader"""
        # Map strategy config names to base names first
        base_name = self._get_base_strategy_name(strategy_name)
        
        # Try to load from file first
        config_file = f"config/strategies/{base_name}.json"
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass  # Fall back to default config
        
        # Use centralized default configurations
        return self._get_optimized_default_config(base_name)
    
    def _get_base_strategy_name(self, config_name: str) -> str:
        """Simplified strategy name mapping"""
        # Define core strategies available in the system
        CORE_STRATEGIES = {
            'basic_momentum', 'vwap', 'bollinger_bands', 'advanced_vwap',
            'mean_reversion', 'macd', 'stochastic', 'pairs', 'ichimoku',
            'supertrend', 'atr_breakout', 'keltner_channels', 'donchian_channels',
            'volume_profile', 'fibonacci_retracement', 'aroon'
        }
        
        # If already a core strategy name, return as-is
        if config_name in CORE_STRATEGIES:
            return config_name
        
        # Simple mapping for variants
        variant_mapping = {
            'vwap_trend_following': 'vwap',
            'vwap_mean_reversion': 'vwap',
            'advanced_vwap_momentum': 'advanced_vwap',
            'advanced_vwap_bands': 'advanced_vwap'
        }
        
        return variant_mapping.get(config_name, config_name)
    
    def _get_optimized_default_config(self, strategy_name: str) -> Dict:
        """Get optimized default configuration for strategies"""
        # Centralized, optimized configurations based on backtesting results
        OPTIMIZED_CONFIGS = {
            'basic_momentum': {
                'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70,
                'ma_short': 10, 'ma_long': 30
            },
            'vwap': {
                'vwap_period': 20, 'strategy_variant': 'mean_reversion',
                'vwap_threshold': 0.005, 'volume_threshold': 1.2,
                'use_momentum': True
            },
            'bollinger_bands': {
                'bb_period': 20, 'bb_std_dev': 2.0,
                'lower_threshold': 0.1, 'upper_threshold': 0.9,
                'use_rsi': True, 'rsi_period': 14
            },
            'advanced_vwap': {
                'vwap_period': 20, 'adx_period': 14, 'adx_threshold': 20,
                'profit_target_pct': 0.6, 'stop_loss_pct': 0.3,
                'volatility_threshold': 0.15, 'volatility_action': 'pause'
            },
            'mean_reversion': {
                'bb_period': 20, 'bb_std_dev': 2.0,
                'entry_zscore': 2.0, 'exit_zscore': 0.5,
                'use_volume_filter': True, 'volume_threshold': 1.2
            },
            'macd': {
                'fast_period': 12, 'slow_period': 26, 'signal_period': 9,
                'use_histogram_filter': True, 'use_rsi_filter': True, 'rsi_period': 14
            },
            'stochastic': {
                'k_period': 14, 'd_period': 3, 'smooth_k': 3,
                'oversold_level': 20, 'overbought_level': 80,
                'use_volume_confirmation': True
            },
            'pairs': {
                'lookback_period': 60, 'entry_threshold': 2.0, 'exit_threshold': 0.5,
                'min_correlation': 0.7
            },
            'ichimoku': {
                'tenkan_period': 9, 'kijun_period': 26,
                'senkou_b_period': 52, 'chikou_period': 26,
                'strategy_variant': 'classic', 'use_volume': True, 'volume_threshold': 1.2
            },
            'supertrend': {
                'atr_period': 10, 'multiplier': 3.0, 'strategy_variant': 'classic',
                'use_volume_analysis': True, 'volume_threshold': 1.2, 'use_confirmation': True
            },
            'atr_breakout': {
                'atr_period': 14, 'atr_multiplier': 2.0, 'lookback_period': 20,
                'strategy_variant': 'adaptive', 'min_atr_pct': 0.5, 'max_atr_pct': 5.0,
                'use_volume_confirmation': True, 'volume_threshold': 1.5
            },
            'keltner_channels': {
                'ema_period': 20, 'atr_period': 10, 'multiplier': 2.0,
                'strategy_variant': 'mean_reversion', 'squeeze_threshold': 0.015
            },
            'donchian_channels': {
                'upper_period': 20, 'lower_period': 20, 'middle_period': 10,
                'strategy_variant': 'breakout', 'min_width_pct': 1.0,
                'use_volume': True, 'volume_threshold': 1.2
            },
            'volume_profile': {
                'profile_period': 50, 'num_bins': 20, 'poc_threshold': 0.003,
                'min_volume_ratio': 1.2, 'momentum_threshold': 0.002, 'use_value_area': True
            },
            'fibonacci_retracement': {
                'swing_period': 20, 'fib_proximity': 0.003,
                'rsi_oversold': 30, 'rsi_overbought': 70,
                'momentum_threshold': 0.001, 'use_golden_ratio': True
            },
            'aroon': {
                'aroon_period': 25, 'oscillator_threshold': 50,
                'momentum_threshold': 0.001, 'use_volume_confirmation': True,
                'use_ma_confirmation': True, 'use_trend_strength': True,
                'use_consolidation_breakout': True
            }
        }
        
        return OPTIMIZED_CONFIGS.get(strategy_name, {})
    


class EnhancedStrategyPerformanceTracker:
    """Enhanced performance tracker with real-time feedback and dynamic weight adjustment"""
    
    def __init__(self, window: int = 100, short_window: int = 20):
        """Initialize enhanced performance tracker"""
        self.window = window
        self.short_window = short_window
        self.performance_history = {}
        self.trade_history = {}
        self.real_time_pnl = {}  # Real-time P&L tracking
        self.strategy_weights = {}  # Dynamic weights
        self.confidence_scores = {}  # Strategy confidence scores
        self.correlation_matrix = {}  # Strategy correlation tracking
        
        # Performance decay factors
        self.performance_decay = 0.95
        self.weight_update_frequency = 10  # Update weights every 10 signals
        self.signal_count = 0
        
    def update_real_time_pnl(self, regime: str, strategy: str, signal: int, 
                            entry_price: float, current_price: float, timestamp):
        """Update real-time P&L for active positions"""
        key = f"{regime}_{strategy}"
        
        if key not in self.real_time_pnl:
            self.real_time_pnl[key] = {'positions': [], 'total_pnl': 0.0}
        
        if signal != 0:  # New position
            position = {
                'signal': signal,
                'entry_price': entry_price,
                'entry_time': timestamp,
                'unrealized_pnl': 0.0
            }
            self.real_time_pnl[key]['positions'].append(position)
        
        # Update unrealized P&L for all open positions
        total_unrealized = 0.0
        for position in self.real_time_pnl[key]['positions']:
            if position['signal'] == 1:  # Long position
                position['unrealized_pnl'] = (current_price - position['entry_price']) / position['entry_price']
            else:  # Short position
                position['unrealized_pnl'] = (position['entry_price'] - current_price) / position['entry_price']
            total_unrealized += position['unrealized_pnl']
        
        self.real_time_pnl[key]['total_unrealized'] = total_unrealized
    
    def close_position(self, regime: str, strategy: str, exit_price: float, timestamp):
        """Close position and record realized P&L"""
        key = f"{regime}_{strategy}"
        
        if key in self.real_time_pnl and self.real_time_pnl[key]['positions']:
            # Close oldest position (FIFO)
            position = self.real_time_pnl[key]['positions'].pop(0)
            
            if position['signal'] == 1:  # Long position
                realized_pnl = (exit_price - position['entry_price']) / position['entry_price']
            else:  # Short position
                realized_pnl = (position['entry_price'] - exit_price) / position['entry_price']
            
            # Record trade result
            trade_result = {
                'timestamp': timestamp,
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'signal': position['signal'],
                'return_pct': realized_pnl,
                'profit': realized_pnl > 0,
                'holding_period': (timestamp - position['entry_time']).total_seconds() / 3600 if hasattr(timestamp - position['entry_time'], 'total_seconds') else 1
            }
            
            self.update_performance(regime, strategy, trade_result)
            self.real_time_pnl[key]['total_pnl'] += realized_pnl
    
    def update_performance(self, regime: str, strategy: str, trade_result: Dict[str, Any]):
        """Enhanced performance update with confidence scoring"""
        key = f"{regime}_{strategy}"
        
        if key not in self.performance_history:
            self.performance_history[key] = []
            self.trade_history[key] = []
            self.confidence_scores[key] = 0.5  # Neutral initial confidence
        
        self.trade_history[key].append(trade_result)
        
        # Calculate recent performance with exponential weighting
        recent_trades = self.trade_history[key][-self.window:]
        short_trades = self.trade_history[key][-self.short_window:]
        
        if recent_trades:
            # Long-term metrics
            win_rate = sum(1 for t in recent_trades if t['profit']) / len(recent_trades)
            avg_return = np.mean([t['return_pct'] for t in recent_trades])
            sharpe = self._calculate_enhanced_sharpe(recent_trades)
            
            # Short-term performance
            short_win_rate = sum(1 for t in short_trades if t['profit']) / len(short_trades) if short_trades else win_rate
            short_avg_return = np.mean([t['return_pct'] for t in short_trades]) if short_trades else avg_return
            
            # Calculate momentum and consistency
            momentum = self._calculate_performance_momentum(recent_trades)
            consistency = self._calculate_consistency(recent_trades)
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(win_rate, sharpe, momentum, consistency)
            self.confidence_scores[key] = confidence
            
            performance_metrics = {
                'timestamp': trade_result['timestamp'],
                'win_rate': win_rate,
                'short_win_rate': short_win_rate,
                'avg_return': avg_return,
                'short_avg_return': short_avg_return,
                'sharpe': sharpe,
                'momentum': momentum,
                'consistency': consistency,
                'confidence': confidence,
                'trade_count': len(recent_trades),
                'unrealized_pnl': self.real_time_pnl.get(key, {}).get('total_unrealized', 0.0)
            }
            
            self.performance_history[key].append(performance_metrics)
        
        # Update strategy weights periodically
        self.signal_count += 1
        if self.signal_count % self.weight_update_frequency == 0:
            self._update_dynamic_weights(regime)
    
    def _calculate_enhanced_sharpe(self, trades: List[Dict]) -> float:
        """Calculate enhanced Sharpe ratio with risk adjustments"""
        if len(trades) < 2:
            return 0
        
        returns = [t['return_pct'] for t in trades]
        
        # Weight recent returns more heavily
        weights = [self.performance_decay ** (len(returns) - i - 1) for i in range(len(returns))]
        weight_sum = sum(weights)
        
        if weight_sum == 0:
            return 0
        
        # Weighted average return
        weighted_mean = sum(r * w for r, w in zip(returns, weights)) / weight_sum
        
        # Weighted standard deviation
        weighted_variance = sum(w * (r - weighted_mean) ** 2 for r, w in zip(returns, weights)) / weight_sum
        weighted_std = np.sqrt(weighted_variance)
        
        if weighted_std == 0:
            return 0
        
        # Risk-adjusted Sharpe with drawdown penalty
        drawdown_penalty = self._calculate_max_drawdown_penalty(returns)
        base_sharpe = weighted_mean / weighted_std * np.sqrt(252)  # Annualized
        
        return base_sharpe * (1 - drawdown_penalty)
    
    def _calculate_performance_momentum(self, trades: List[Dict]) -> float:
        """Calculate performance momentum (trend in recent performance)"""
        if len(trades) < 5:
            return 0
        
        # Split into two halves and compare
        mid_point = len(trades) // 2
        first_half = trades[:mid_point]
        second_half = trades[mid_point:]
        
        first_avg = np.mean([t['return_pct'] for t in first_half])
        second_avg = np.mean([t['return_pct'] for t in second_half])
        
        return second_avg - first_avg  # Positive = improving performance
    
    def _calculate_consistency(self, trades: List[Dict]) -> float:
        """Calculate performance consistency (1 - volatility of returns)"""
        if len(trades) < 3:
            return 0.5
        
        returns = [t['return_pct'] for t in trades]
        std_returns = np.std(returns)
        mean_returns = abs(np.mean(returns))
        
        if mean_returns == 0:
            return 0
        
        # Coefficient of variation (inverted for consistency)
        cv = std_returns / mean_returns
        consistency = 1 / (1 + cv)  # 0 to 1 scale
        
        return consistency
    
    def _calculate_max_drawdown_penalty(self, returns: List[float]) -> float:
        """Calculate maximum drawdown penalty factor"""
        if len(returns) < 2:
            return 0
        
        cumulative = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = abs(min(drawdowns))
        
        # Penalty increases non-linearly with drawdown
        penalty = min(0.5, max_drawdown ** 0.5)
        return penalty
    
    def _calculate_confidence_score(self, win_rate: float, sharpe: float, 
                                   momentum: float, consistency: float) -> float:
        """Calculate overall confidence score for strategy"""
        # Normalize components
        win_rate_score = win_rate  # Already 0-1
        sharpe_score = max(0, min(1, (sharpe + 2) / 4))  # -2 to 2 Sharpe -> 0 to 1
        momentum_score = max(0, min(1, (momentum + 0.1) / 0.2))  # -0.1 to 0.1 -> 0 to 1
        consistency_score = consistency  # Already 0-1
        
        # Weighted combination
        confidence = (win_rate_score * 0.3 + 
                     sharpe_score * 0.4 + 
                     momentum_score * 0.2 + 
                     consistency_score * 0.1)
        
        return confidence
    
    def _update_dynamic_weights(self, regime: str):
        """Update dynamic weights for strategies in regime"""
        strategies_in_regime = []
        scores = []
        
        for key in self.confidence_scores:
            if key.startswith(f"{regime}_"):
                strategy = key.replace(f"{regime}_", "")
                strategies_in_regime.append(strategy)
                scores.append(self.confidence_scores[key])
        
        if not strategies_in_regime:
            return
        
        # Softmax normalization for weights
        exp_scores = np.exp(np.array(scores) - np.max(scores))  # Numerical stability
        softmax_weights = exp_scores / np.sum(exp_scores)
        
        # Update strategy weights
        regime_key = f"regime_{regime}"
        if regime_key not in self.strategy_weights:
            self.strategy_weights[regime_key] = {}
        
        for strategy, weight in zip(strategies_in_regime, softmax_weights):
            self.strategy_weights[regime_key][strategy] = weight
    
    def get_dynamic_weights(self, regime: str) -> Dict[str, float]:
        """Get current dynamic weights for regime"""
        regime_key = f"regime_{regime}"
        return self.strategy_weights.get(regime_key, {})
    
    def get_strategy_performance(self, regime: str, strategy: str) -> Dict[str, float]:
        """Get enhanced strategy performance metrics"""
        key = f"{regime}_{strategy}"
        
        if key not in self.performance_history or not self.performance_history[key]:
            return {
                'win_rate': 0.5, 'avg_return': 0, 'sharpe': 0, 'trade_count': 0,
                'confidence': 0.5, 'momentum': 0, 'consistency': 0.5,
                'unrealized_pnl': 0.0
            }
        
        return self.performance_history[key][-1]
    
    def get_best_strategies(self, regime: str, top_n: int = 3, 
                          method: str = 'confidence') -> List[Tuple[str, float]]:
        """Get best performing strategies using enhanced scoring"""
        regime_strategies = {}
        
        for key in self.performance_history:
            if key.startswith(f"{regime}_"):
                strategy = key.replace(f"{regime}_", "")
                if self.performance_history[key]:
                    latest_perf = self.performance_history[key][-1]
                    
                    if method == 'confidence':
                        score = latest_perf['confidence']
                    elif method == 'sharpe':
                        score = latest_perf['sharpe']
                    elif method == 'combined':
                        # Balanced scoring
                        score = (latest_perf['confidence'] * 0.4 + 
                                latest_perf['sharpe'] * 0.3 + 
                                latest_perf['momentum'] * 0.2 + 
                                latest_perf['consistency'] * 0.1)
                    else:
                        score = latest_perf['confidence']
                    
                    regime_strategies[strategy] = score
        
        # Sort by score
        sorted_strategies = sorted(regime_strategies.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        return sorted_strategies[:top_n]
    
    def _calculate_sharpe(self, trades: List[Dict]) -> float:
        """Legacy method for backward compatibility"""
        return self._calculate_enhanced_sharpe(trades)


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
            
        self.performance_tracker = EnhancedStrategyPerformanceTracker()
        
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
        """Enhanced regime state update with adaptive transition handling"""
        current_confidence = max(indicators.regime_probability.values())
        self.regime_confidence = current_confidence
        
        if regime != self.current_regime:
            # Enhanced regime change detection with multiple criteria
            should_change_regime = self._should_change_regime(regime, current_confidence, indicators)
            
            if should_change_regime:
                # Determine transition type (gradual vs rapid)
                transition_type = self._determine_transition_type(regime, current_confidence, indicators)
                
                # Execute regime change
                self._execute_regime_change(regime, current_confidence, transition_type)
                
        else:
            self.regime_duration += 1
            
            # Update transition progress with enhanced logic
            if self.in_transition:
                self._update_transition_progress()
    
    def _should_change_regime(self, new_regime: MarketRegime, confidence: float, 
                             indicators: RegimeIndicators) -> bool:
        """Enhanced regime change decision with multiple criteria"""
        # Basic confidence threshold
        confidence_threshold = self.config.config['transition_rules'].get('confidence_threshold', 0.6)
        min_duration = self.config.config['transition_rules'].get('min_regime_duration', 10)
        
        # 1. Standard checks
        if confidence < confidence_threshold:
            return False
        
        if self.regime_duration < min_duration and self.current_regime != MarketRegime.UNKNOWN:
            # Allow early transition only in exceptional cases
            if not self._is_exceptional_transition_case(new_regime, confidence, indicators):
                return False
        
        # 2. Enhanced checks
        
        # Prevent oscillation between similar regimes
        if self._is_regime_oscillation(new_regime):
            return False
        
        # Check if transition makes sense based on market indicators
        if not self._is_logical_regime_transition(new_regime, indicators):
            return False
        
        # Check ensemble performance - don't change if current regime is performing well
        if not self._should_override_good_performance(new_regime):
            return False
        
        return True
    
    def _is_exceptional_transition_case(self, new_regime: MarketRegime, confidence: float, 
                                       indicators: RegimeIndicators) -> bool:
        """Check if this is an exceptional case that warrants early regime change"""
        # Very high confidence (>90%) allows early transition
        if confidence > 0.9:
            return True
        
        # Extreme volatility changes
        if hasattr(indicators, 'volatility_ratio'):
            current_vol = indicators.volatility_ratio
            if current_vol > 3.0 or current_vol < 0.3:  # 3x or 1/3x normal volatility
                return True
        
        # Sharp trend reversals
        if hasattr(indicators, 'trend_strength'):
            trend_change = abs(indicators.trend_strength)
            if trend_change > 0.8:  # Strong trend change
                return True
        
        return False
    
    def _is_regime_oscillation(self, new_regime: MarketRegime) -> bool:
        """Prevent rapid oscillation between regimes"""
        if not hasattr(self, 'regime_history'):
            self.regime_history = []
        
        # Add current attempt to history
        self.regime_history.append({
            'regime': new_regime,
            'timestamp': len(self.regime_history),  # Simple timestamp
            'duration': self.regime_duration
        })
        
        # Keep only recent history
        if len(self.regime_history) > 10:
            self.regime_history = self.regime_history[-10:]
        
        # Check for oscillation pattern
        if len(self.regime_history) >= 4:
            recent_regimes = [h['regime'] for h in self.regime_history[-4:]]
            
            # Pattern: A -> B -> A -> B (oscillation)
            if (recent_regimes[0] == recent_regimes[2] and 
                recent_regimes[1] == recent_regimes[3] and 
                recent_regimes[0] != recent_regimes[1]):
                return True
        
        return False
    
    def _is_logical_regime_transition(self, new_regime: MarketRegime, 
                                     indicators: RegimeIndicators) -> bool:
        """Check if regime transition makes logical sense"""
        # Define logical transition paths
        logical_transitions = {
            MarketRegime.TRENDING_UP: [MarketRegime.SIDEWAYS, MarketRegime.VOLATILE, MarketRegime.TRENDING_DOWN],
            MarketRegime.TRENDING_DOWN: [MarketRegime.SIDEWAYS, MarketRegime.VOLATILE, MarketRegime.TRENDING_UP],
            MarketRegime.SIDEWAYS: [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN, MarketRegime.VOLATILE],
            MarketRegime.VOLATILE: [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN, MarketRegime.SIDEWAYS],
            MarketRegime.UNKNOWN: [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN, 
                                  MarketRegime.SIDEWAYS, MarketRegime.VOLATILE]
        }
        
        allowed_transitions = logical_transitions.get(self.current_regime, [])
        
        # Allow transition if it's in the logical path or if indicators strongly support it
        if new_regime in allowed_transitions:
            return True
        
        # Special case: allow direct trending up <-> trending down if supported by strong indicators
        if (self.current_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN] and 
            new_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]):
            # Require very high confidence for direct reversal
            return self.regime_confidence > 0.85
        
        return False
    
    def _should_override_good_performance(self, new_regime: MarketRegime) -> bool:
        """Check if we should change regime despite good current performance"""
        # If no performance history, allow change
        if not self.signal_history:
            return True
        
        # Check recent ensemble performance
        recent_performance = self._calculate_ensemble_performance_multiplier()
        
        # If performance is very good (>1.1), be more reluctant to change
        if recent_performance > 1.15:
            return self.regime_confidence > 0.8  # Require higher confidence
        
        # If performance is poor (<0.9), be more eager to change
        if recent_performance < 0.85:
            return True  # Allow change more easily
        
        return True  # Default: allow change
    
    def _determine_transition_type(self, new_regime: MarketRegime, confidence: float, 
                                  indicators: RegimeIndicators) -> str:
        """Determine if transition should be gradual or rapid"""
        # Rapid transition conditions
        if confidence > 0.9:
            return 'rapid'
        
        # Check volatility
        if hasattr(indicators, 'volatility_ratio') and indicators.volatility_ratio > 2.5:
            return 'rapid'
        
        # Check if regimes are very different
        regime_distance = self._calculate_regime_distance(self.current_regime, new_regime)
        if regime_distance > 0.8:
            return 'rapid'
        
        return 'gradual'
    
    def _calculate_regime_distance(self, regime1: MarketRegime, regime2: MarketRegime) -> float:
        """Calculate conceptual distance between regimes"""
        # Define regime similarity matrix
        regime_similarity = {
            (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN): 0.3,
            (MarketRegime.TRENDING_UP, MarketRegime.SIDEWAYS): 0.6,
            (MarketRegime.TRENDING_UP, MarketRegime.VOLATILE): 0.5,
            (MarketRegime.TRENDING_DOWN, MarketRegime.SIDEWAYS): 0.6,
            (MarketRegime.TRENDING_DOWN, MarketRegime.VOLATILE): 0.5,
            (MarketRegime.SIDEWAYS, MarketRegime.VOLATILE): 0.7,
        }
        
        # Make symmetric
        key = (regime1, regime2) if regime1.value < regime2.value else (regime2, regime1)
        similarity = regime_similarity.get(key, 0.0)
        
        return 1.0 - similarity
    
    def _execute_regime_change(self, new_regime: MarketRegime, confidence: float, 
                              transition_type: str):
        """Execute regime change with appropriate transition handling"""
        self.previous_regime = self.current_regime
        self.current_regime = new_regime
        self.regime_duration = 0
        self.in_transition = True
        self.transition_progress = 0
        self.transition_type = transition_type
        
        # Set transition parameters based on type
        if transition_type == 'rapid':
            self.transition_periods = 2  # Fast transition
            self.emergency_transition = True
        else:
            self.transition_periods = self.config.config['transition_rules'].get('transition_periods', 5)
            self.emergency_transition = False
        
        # Immediate strategy adjustment for rapid transitions
        if transition_type == 'rapid':
            self._immediate_strategy_adjustment()
        
        if not self.optimization_mode:
            transition_emoji = "⚡" if transition_type == 'rapid' else "🔄"
            print(f"{transition_emoji} {transition_type.title()} regime change: "
                  f"{self.previous_regime.value} → {new_regime.value} "
                  f"(confidence: {confidence:.2%})")
    
    def _immediate_strategy_adjustment(self):
        """Immediately adjust strategy weights for rapid transitions"""
        # Boost confidence of new regime strategies
        if hasattr(self, 'performance_tracker'):
            regime_key = f"regime_{self.current_regime.value}"
            if regime_key not in self.performance_tracker.strategy_weights:
                self.performance_tracker.strategy_weights[regime_key] = {}
            
            # Get new regime strategies and boost their initial weights
            regime_config = self.config.get_regime_strategies(self.current_regime)
            primary_strategies = regime_config.get('primary', [])
            
            for strategy in primary_strategies:
                current_weight = self.performance_tracker.strategy_weights[regime_key].get(strategy, 0.25)
                self.performance_tracker.strategy_weights[regime_key][strategy] = min(0.8, current_weight * 1.5)
    
    def _update_transition_progress(self):
        """Update transition progress with enhanced logic"""
        if hasattr(self, 'transition_periods'):
            progress_increment = 1.0 / self.transition_periods
        else:
            progress_increment = 1.0 / self.config.config['transition_rules'].get('transition_periods', 5)
        
        # Faster progress for emergency transitions
        if hasattr(self, 'emergency_transition') and self.emergency_transition:
            progress_increment *= 2.0
        
        self.transition_progress = min(1.0, self.transition_progress + progress_increment)
        
        if self.transition_progress >= 1.0:
            self.in_transition = False
            self.emergency_transition = False
            
            # Log transition completion
            if not self.optimization_mode:
                print(f"✅ Transition completed: Now fully in {self.current_regime.value} regime")
    
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
        """Simplified strategy name mapping"""
        # Define core strategies available in the system
        CORE_STRATEGIES = {
            'basic_momentum', 'vwap', 'bollinger_bands', 'advanced_vwap',
            'mean_reversion', 'macd', 'stochastic', 'pairs', 'ichimoku',
            'supertrend', 'atr_breakout', 'keltner_channels', 'donchian_channels',
            'volume_profile', 'fibonacci_retracement', 'aroon'
        }
        
        # If already a core strategy name, return as-is
        if config_name in CORE_STRATEGIES:
            return config_name
        
        # Simple mapping for variants
        variant_mapping = {
            'vwap_trend_following': 'vwap',
            'vwap_mean_reversion': 'vwap',
            'advanced_vwap_momentum': 'advanced_vwap',
            'advanced_vwap_bands': 'advanced_vwap'
        }
        
        return variant_mapping.get(config_name, config_name)
    
    def _get_optimized_default_config(self, strategy_name: str) -> Dict:
        """Get optimized default configuration for strategies"""
        # Centralized, optimized configurations based on backtesting results
        OPTIMIZED_CONFIGS = {
            'basic_momentum': {
                'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70,
                'ma_short': 10, 'ma_long': 30
            },
            'vwap': {
                'vwap_period': 20, 'strategy_variant': 'mean_reversion',
                'vwap_threshold': 0.005, 'volume_threshold': 1.2,
                'use_momentum': True
            },
            'bollinger_bands': {
                'bb_period': 20, 'bb_std_dev': 2.0,
                'lower_threshold': 0.1, 'upper_threshold': 0.9,
                'use_rsi': True, 'rsi_period': 14
            },
            'advanced_vwap': {
                'vwap_period': 20, 'adx_period': 14, 'adx_threshold': 20,
                'profit_target_pct': 0.6, 'stop_loss_pct': 0.3,
                'volatility_threshold': 0.15, 'volatility_action': 'pause'
            },
            'mean_reversion': {
                'bb_period': 20, 'bb_std_dev': 2.0,
                'entry_zscore': 2.0, 'exit_zscore': 0.5,
                'use_volume_filter': True, 'volume_threshold': 1.2
            },
            'macd': {
                'fast_period': 12, 'slow_period': 26, 'signal_period': 9,
                'use_histogram_filter': True, 'use_rsi_filter': True, 'rsi_period': 14
            },
            'stochastic': {
                'k_period': 14, 'd_period': 3, 'smooth_k': 3,
                'oversold_level': 20, 'overbought_level': 80,
                'use_volume_confirmation': True
            },
            'pairs': {
                'lookback_period': 60, 'entry_threshold': 2.0, 'exit_threshold': 0.5,
                'min_correlation': 0.7
            },
            'ichimoku': {
                'tenkan_period': 9, 'kijun_period': 26,
                'senkou_b_period': 52, 'chikou_period': 26,
                'strategy_variant': 'classic', 'use_volume': True, 'volume_threshold': 1.2
            },
            'supertrend': {
                'atr_period': 10, 'multiplier': 3.0, 'strategy_variant': 'classic',
                'use_volume_analysis': True, 'volume_threshold': 1.2, 'use_confirmation': True
            },
            'atr_breakout': {
                'atr_period': 14, 'atr_multiplier': 2.0, 'lookback_period': 20,
                'strategy_variant': 'adaptive', 'min_atr_pct': 0.5, 'max_atr_pct': 5.0,
                'use_volume_confirmation': True, 'volume_threshold': 1.5
            },
            'keltner_channels': {
                'ema_period': 20, 'atr_period': 10, 'multiplier': 2.0,
                'strategy_variant': 'mean_reversion', 'squeeze_threshold': 0.015
            },
            'donchian_channels': {
                'upper_period': 20, 'lower_period': 20, 'middle_period': 10,
                'strategy_variant': 'breakout', 'min_width_pct': 1.0,
                'use_volume': True, 'volume_threshold': 1.2
            },
            'volume_profile': {
                'profile_period': 50, 'num_bins': 20, 'poc_threshold': 0.003,
                'min_volume_ratio': 1.2, 'momentum_threshold': 0.002, 'use_value_area': True
            },
            'fibonacci_retracement': {
                'swing_period': 20, 'fib_proximity': 0.003,
                'rsi_oversold': 30, 'rsi_overbought': 70,
                'momentum_threshold': 0.001, 'use_golden_ratio': True
            },
            'aroon': {
                'aroon_period': 25, 'oscillator_threshold': 50,
                'momentum_threshold': 0.001, 'use_volume_confirmation': True,
                'use_ma_confirmation': True, 'use_trend_strength': True,
                'use_consolidation_breakout': True
            }
        }
        
        return OPTIMIZED_CONFIGS.get(strategy_name, {})
    
    def _aggregate_signals(self, strategy_signals: Dict[str, int], 
                          current_data: pd.Series) -> Tuple[int, float, str]:
        """Enhanced signal aggregation with dynamic weights and correlation adjustment"""
        if not strategy_signals:
            return 0, 0.0, "no_signal"
        
        # Get dynamic weights from performance tracker
        dynamic_weights = self.performance_tracker.get_dynamic_weights(self.current_regime.value)
        
        # Fallback to static configuration if no dynamic weights available
        regime_config = self.config.get_regime_strategies(self.current_regime)
        static_weights = regime_config.get('weights', {'primary': 0.7, 'secondary': 0.3})
        primary_strategies = regime_config.get('primary', [])
        secondary_strategies = regime_config.get('secondary', [])
        
        # Calculate correlation-adjusted weights
        correlation_penalty = self._calculate_correlation_penalty(strategy_signals)
        
        # Calculate weighted signals with multiple weight sources
        weighted_sum = 0.0
        total_weight = 0.0
        signal_sources = []
        strategy_contributions = []
        
        for strategy_name, signal in strategy_signals.items():
            if signal != 0:  # Only consider non-zero signals
                base_weight = 0.1  # Minimal default weight
                
                # Dynamic weight (performance-based)
                if strategy_name in dynamic_weights:
                    performance_weight = dynamic_weights[strategy_name]
                else:
                    # Static weight based on strategy category
                    if strategy_name in primary_strategies:
                        performance_weight = static_weights.get('primary', 0.7)
                    elif strategy_name in secondary_strategies:
                        performance_weight = static_weights.get('secondary', 0.3)
                    else:
                        performance_weight = base_weight
                
                # Get strategy confidence score
                perf_metrics = self.performance_tracker.get_strategy_performance(
                    self.current_regime.value, strategy_name
                )
                confidence_multiplier = perf_metrics.get('confidence', 0.5)
                
                # Apply correlation penalty to reduce weight of highly correlated strategies
                correlation_adjusted_weight = performance_weight * (1 - correlation_penalty.get(strategy_name, 0))
                
                # Final weight combines performance, confidence, and correlation adjustment
                final_weight = correlation_adjusted_weight * (0.5 + confidence_multiplier * 0.5)
                
                # Adjust weight based on transition state
                if self.in_transition and self.previous_regime:
                    prev_config = self.config.get_regime_strategies(self.previous_regime)
                    if strategy_name in prev_config.get('primary', []):
                        # Reduce weight of previous regime's strategies
                        final_weight *= (1 - self.transition_progress * 0.5)
                
                # Apply momentum boost for improving strategies
                momentum = perf_metrics.get('momentum', 0)
                if momentum > 0.02:  # Significant positive momentum
                    final_weight *= 1.2
                elif momentum < -0.02:  # Significant negative momentum
                    final_weight *= 0.8
                
                weighted_sum += signal * final_weight
                total_weight += final_weight
                signal_sources.append(strategy_name)
                strategy_contributions.append({
                    'strategy': strategy_name,
                    'signal': signal,
                    'weight': final_weight,
                    'confidence': confidence_multiplier,
                    'momentum': momentum
                })
        
        # Calculate final signal with adaptive threshold
        if total_weight > 0:
            normalized_signal = weighted_sum / total_weight
            signal_strength = abs(normalized_signal)
            
            # Adaptive threshold based on regime confidence and strategy consensus
            base_threshold = 0.3  # Lower base threshold for more responsive signals
            confidence_adjustment = self.regime_confidence * 0.3  # Higher confidence = higher threshold
            consensus_adjustment = min(0.2, len(signal_sources) / 5 * 0.1)  # More strategies = lower threshold
            
            adaptive_threshold = base_threshold + confidence_adjustment - consensus_adjustment
            adaptive_threshold = max(0.2, min(0.6, adaptive_threshold))  # Bound between 0.2-0.6
            
            # Determine final signal
            if normalized_signal > adaptive_threshold:
                final_signal = 1
            elif normalized_signal < -adaptive_threshold:
                final_signal = -1
            else:
                final_signal = 0
            
            # Enhanced signal source information
            if strategy_contributions:
                top_contributors = sorted(strategy_contributions, 
                                        key=lambda x: abs(x['signal'] * x['weight']), 
                                        reverse=True)[:3]
                signal_source = ", ".join([f"{c['strategy']}({c['confidence']:.2f})" 
                                         for c in top_contributors])
            else:
                signal_source = "no_consensus"
        else:
            final_signal = 0
            signal_strength = 0.0
            signal_source = "no_signal"
        
        return final_signal, signal_strength, signal_source
    
    def _calculate_correlation_penalty(self, strategy_signals: Dict[str, int]) -> Dict[str, float]:
        """Calculate correlation penalty to reduce weight of highly correlated strategies"""
        if len(strategy_signals) < 2:
            return {}
        
        # Simple correlation estimation based on signal agreement
        penalties = {}
        strategies = list(strategy_signals.keys())
        
        for i, strategy_a in enumerate(strategies):
            penalty = 0.0
            signal_a = strategy_signals[strategy_a]
            
            if signal_a != 0:
                for j, strategy_b in enumerate(strategies):
                    if i != j:
                        signal_b = strategy_signals[strategy_b]
                        # If both strategies give same non-zero signal, add correlation penalty
                        if signal_a == signal_b and signal_b != 0:
                            penalty += 0.1  # 10% penalty per correlated strategy
                
                penalties[strategy_a] = min(0.5, penalty)  # Cap at 50% penalty
        
        return penalties
    
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
                         portfolio_value: float, market: str = None) -> float:
        """Enhanced position sizing with regime confidence and strategy performance"""
        if signal == 0:
            return 0
        
        risk_config = self.config.config.get('risk_management', {})
        base_size = risk_config.get('base_position_size', 0.25)
        
        # 1. Regime-based adjustment
        if self.current_regime.value in self.config.config.get('regime_strategies', {}):
            regime_risk = self.config.config['regime_strategies'][self.current_regime.value].get(
                'risk_multiplier', 1.0
            )
        else:
            regime_risk = 1.0
        
        # 2. Regime confidence adjustment
        confidence_multiplier = self._calculate_confidence_multiplier()
        
        # 3. Strategy ensemble performance adjustment
        ensemble_performance_multiplier = self._calculate_ensemble_performance_multiplier()
        
        # 4. Signal strength adjustment
        signal_strength = 1.0
        if self.signal_history and len(self.signal_history) > 0:
            signal_strength = self.signal_history[-1].get('strength', 1.0)
        
        # 5. Volatility-based adjustment
        volatility_multiplier = self._calculate_volatility_multiplier()
        
        # 6. Market condition adjustment
        market_condition_multiplier = self._calculate_market_condition_multiplier()
        
        # Calculate base position size with all adjustments
        position_size = (base_size * 
                        regime_risk * 
                        confidence_multiplier * 
                        ensemble_performance_multiplier *
                        signal_strength * 
                        volatility_multiplier * 
                        market_condition_multiplier)
        
        # Apply Kelly Criterion if enough trade history
        if market:
            kelly_adjustment = self._calculate_kelly_adjustment(market)
            position_size *= kelly_adjustment
        
        # Apply position size bounds
        min_position = 0.05  # Minimum 5% of portfolio
        max_position = 0.4   # Maximum 40% of portfolio (increased from 33%)
        position_size = max(min_position, min(position_size, max_position))
        
        # Additional safety check during regime transitions
        if self.in_transition:
            position_size *= 0.7  # Reduce position size by 30% during transitions
        
        # Convert to value and shares
        position_value = portfolio_value * position_size
        
        return position_value / current_price if current_price > 0 else 0
    
    def _calculate_confidence_multiplier(self) -> float:
        """Calculate position size multiplier based on regime confidence"""
        if self.regime_confidence < 0.3:
            return 0.5  # Very low confidence - reduce position size significantly
        elif self.regime_confidence < 0.5:
            return 0.7  # Low confidence - reduce position size moderately
        elif self.regime_confidence > 0.8:
            return 1.3  # High confidence - increase position size
        else:
            return 1.0  # Normal confidence - no adjustment
    
    def _calculate_ensemble_performance_multiplier(self) -> float:
        """Calculate multiplier based on recent ensemble performance"""
        if not self.signal_history:
            return 1.0
        
        # Look at recent performance (last 20 signals)
        recent_signals = self.signal_history[-20:]
        if len(recent_signals) < 5:
            return 1.0
        
        # Calculate recent win rate proxy (simplified)
        positive_outcomes = 0
        total_signals = len(recent_signals)
        
        for i, signal_info in enumerate(recent_signals):
            # Simple momentum-based proxy for performance
            if i < len(recent_signals) - 1:
                current_strength = signal_info.get('strength', 0)
                if current_strength > 0.6:  # Strong signal threshold
                    positive_outcomes += 1
        
        win_rate_proxy = positive_outcomes / total_signals
        
        if win_rate_proxy > 0.7:
            return 1.2  # Good recent performance
        elif win_rate_proxy < 0.3:
            return 0.8  # Poor recent performance
        else:
            return 1.0  # Average performance
    
    def _calculate_volatility_multiplier(self) -> float:
        """Adjust position size based on market volatility"""
        if not self.signal_history:
            return 1.0
        
        # Use regime volatility if available
        if hasattr(self, 'current_volatility'):
            volatility = self.current_volatility
        else:
            # Fallback to average volatility estimation
            volatility = 1.0
        
        if volatility > 2.0:
            return 0.6  # High volatility - reduce position size
        elif volatility > 1.5:
            return 0.8  # Moderate high volatility
        elif volatility < 0.5:
            return 1.2  # Low volatility - can increase position size
        else:
            return 1.0  # Normal volatility
    
    def _calculate_market_condition_multiplier(self) -> float:
        """Adjust based on overall market conditions"""
        # Check recent regime stability
        if self.regime_duration < 5:
            return 0.8  # Unstable regime - be more cautious
        elif self.regime_duration > 20:
            return 1.1  # Stable regime - can be more aggressive
        else:
            return 1.0
    
    def _calculate_kelly_adjustment(self, market: str) -> float:
        """Calculate Kelly Criterion-based adjustment if enough data"""
        try:
            # Get strategy performance for current regime
            best_strategies = self.performance_tracker.get_best_strategies(
                self.current_regime.value, top_n=3, method='combined'
            )
            
            if not best_strategies:
                return 1.0
            
            # Use best strategy's win rate and return for Kelly calculation
            best_strategy, best_score = best_strategies[0]
            perf_metrics = self.performance_tracker.get_strategy_performance(
                self.current_regime.value, best_strategy
            )
            
            win_rate = perf_metrics.get('win_rate', 0.5)
            avg_return = perf_metrics.get('avg_return', 0)
            trade_count = perf_metrics.get('trade_count', 0)
            
            # Only apply Kelly if we have sufficient trade history
            if trade_count < 10:
                return 1.0
            
            # Simplified Kelly Criterion: f = (bp - q) / b
            # where b = avg_return/|avg_loss|, p = win_rate, q = 1-win_rate
            if avg_return <= 0 or win_rate <= 0.5:
                return 0.7  # Conservative adjustment for poor performance
            
            # Estimate average loss (assume it's opposite of average return scaled)
            avg_loss = abs(avg_return) * 0.8  # Simplified assumption
            
            if avg_loss == 0:
                return 1.0
            
            b = avg_return / avg_loss
            kelly_fraction = (b * win_rate - (1 - win_rate)) / b
            
            # Apply fractional Kelly (25% of full Kelly) for safety
            fractional_kelly = max(0.2, min(2.0, 0.25 * (1 + kelly_fraction)))
            
            return fractional_kelly
            
        except Exception:
            return 1.0  # Default to no adjustment if calculation fails
    
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