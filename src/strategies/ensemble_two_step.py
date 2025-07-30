"""
Two-Step Ensemble Strategy
1. Load pre-optimized individual strategies
2. Apply regime-based strategy selection with fixed weights
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import os
from datetime import datetime

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


class TwoStepEnsembleStrategy(BaseStrategy):
    """
    Two-step ensemble that uses pre-optimized strategies with regime-based selection
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        """Initialize two-step ensemble strategy"""
        super().__init__(parameters)
        
        # Load optimized parameters for all strategies
        self.optimized_params = self._load_optimized_parameters()
        
        # Initialize regime detector
        self.regime_detector = MarketRegimeDetector(parameters.get('regime_config'))
        
        # Current market being traded
        self.current_market = None
        
        # Regime-based strategy configuration
        self.regime_strategies = {
            MarketRegime.TRENDING_UP: {
                'strategies': ['supertrend', 'ichimoku', 'vwap'],
                'weights': [0.4, 0.35, 0.25]
            },
            MarketRegime.TRENDING_DOWN: {
                'strategies': ['mean_reversion', 'bollinger_bands', 'keltner_channels'],
                'weights': [0.4, 0.3, 0.3]
            },
            MarketRegime.SIDEWAYS: {
                'strategies': ['mean_reversion', 'bollinger_bands', 'volume_profile'],
                'weights': [0.35, 0.35, 0.3]
            },
            MarketRegime.VOLATILE: {
                'strategies': ['atr_breakout', 'bollinger_bands'],
                'weights': [0.6, 0.4]
            },
            MarketRegime.UNKNOWN: {
                'strategies': ['vwap', 'bollinger_bands'],
                'weights': [0.5, 0.5]
            }
        }
        
        # Initialize strategy instances
        self.strategy_instances = {}
        self.active_strategies = {}
        
        # State management
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_confidence = 0.0
        
    def _load_optimized_parameters(self) -> Dict[str, Dict[str, Dict]]:
        """Load pre-optimized parameters for all strategies and markets"""
        optimized_params = {}
        optimization_dir = "results/optimization"
        
        if not os.path.exists(optimization_dir):
            self.logger.warning(f"Optimization directory not found: {optimization_dir}")
            return {}
        
        # Load parameters for each strategy
        for strategy_dir in os.listdir(optimization_dir):
            strategy_path = os.path.join(optimization_dir, strategy_dir)
            
            if os.path.isdir(strategy_path):
                strategy_name = strategy_dir
                optimized_params[strategy_name] = {}
                
                # Load market-specific parameters
                for file in os.listdir(strategy_path):
                    if file.endswith("_optimized.json") and not file.startswith("all_markets"):
                        market = file.replace("_optimized.json", "")
                        
                        with open(os.path.join(strategy_path, file), 'r') as f:
                            data = json.load(f)
                            optimized_params[strategy_name][market] = data["optimized_params"]
        
        return optimized_params
    
    def _get_strategy_instance(self, strategy_name: str, market: str) -> Optional[BaseStrategy]:
        """Get or create strategy instance with optimized parameters"""
        key = f"{strategy_name}_{market}"
        
        if key not in self.strategy_instances:
            # Get optimized parameters for this strategy and market
            if (strategy_name in self.optimized_params and 
                market in self.optimized_params[strategy_name]):
                params = self.optimized_params[strategy_name][market]
            else:
                # Use default parameters if no optimization found
                self.logger.warning(f"No optimized parameters found for {strategy_name} on {market}")
                params = {}
            
            try:
                self.strategy_instances[key] = get_strategy(strategy_name, params)
            except Exception as e:
                self.logger.error(f"Failed to load strategy {strategy_name}: {e}")
                return None
        
        return self.strategy_instances[key]
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators for regime detection and active strategies"""
        # Detect current market regime
        regime, indicators = self.regime_detector.detect_regime(df)
        
        # Store regime information
        df['regime'] = regime.value
        df['regime_confidence'] = max(indicators.regime_probability.values())
        
        self.current_regime = regime
        self.regime_confidence = max(indicators.regime_probability.values())
        
        # Get active strategies for current regime
        if self.current_market:
            regime_config = self.regime_strategies.get(regime, self.regime_strategies[MarketRegime.UNKNOWN])
            
            # Load and calculate indicators for active strategies
            for strategy_name in regime_config['strategies']:
                strategy = self._get_strategy_instance(strategy_name, self.current_market)
                
                if strategy:
                    try:
                        # Calculate strategy indicators
                        strategy_df = strategy.calculate_indicators(df.copy())
                        
                        # Merge indicators with prefix to avoid conflicts
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
        df['regime_trend_strength'] = indicators.trend_strength
        
        return df
    
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate ensemble signals based on regime and active strategies"""
        # Set current market
        self.current_market = market
        
        df['signal'] = 0
        df['signal_strength'] = 0.0
        df['signal_source'] = ''
        
        if self.current_regime == MarketRegime.UNKNOWN and self.regime_confidence < 0.4:
            return df
        
        # Get regime configuration
        regime_config = self.regime_strategies.get(self.current_regime, 
                                                  self.regime_strategies[MarketRegime.UNKNOWN])
        
        # Collect signals from active strategies
        strategy_signals = {}
        
        for strategy_name in regime_config['strategies']:
            strategy = self._get_strategy_instance(strategy_name, market)
            
            if strategy:
                try:
                    # Generate signals for this strategy
                    strategy_df = strategy.generate_signals(df.copy(), market)
                    
                    if 'signal' in strategy_df and len(strategy_df) > 0:
                        strategy_signals[strategy_name] = strategy_df['signal'].iloc[-1]
                    else:
                        strategy_signals[strategy_name] = 0
                except Exception as e:
                    self.logger.error(f"Error generating signals for {strategy_name}: {e}")
                    strategy_signals[strategy_name] = 0
        
        # Aggregate signals with fixed weights
        if strategy_signals and len(df) > 0:
            final_signal, signal_strength, signal_source = self._aggregate_signals(
                strategy_signals, regime_config['weights']
            )
            
            df.iloc[-1, df.columns.get_loc('signal')] = final_signal
            df.iloc[-1, df.columns.get_loc('signal_strength')] = signal_strength
            df.iloc[-1, df.columns.get_loc('signal_source')] = signal_source
        
        return df
    
    def _aggregate_signals(self, signals: Dict[str, int], weights: List[float]) -> Tuple[int, float, str]:
        """Aggregate signals using fixed weights"""
        if not signals:
            return 0, 0.0, "no_signal"
        
        strategy_names = list(self.regime_strategies[self.current_regime]['strategies'])
        
        # Calculate weighted sum
        weighted_sum = 0.0
        total_weight = 0.0
        signal_sources = []
        
        for i, strategy_name in enumerate(strategy_names):
            if strategy_name in signals and signals[strategy_name] != 0:
                weight = weights[i] if i < len(weights) else 1.0 / len(strategy_names)
                weighted_sum += signals[strategy_name] * weight
                total_weight += weight
                signal_sources.append(f"{strategy_name}({weight:.2f})")
        
        # Determine final signal
        if total_weight > 0:
            normalized_signal = weighted_sum / total_weight
            signal_strength = abs(normalized_signal)
            
            # Threshold for signal generation
            threshold = 0.5 if self.current_regime != MarketRegime.VOLATILE else 0.7
            
            if normalized_signal > threshold:
                final_signal = 1
            elif normalized_signal < -threshold:
                final_signal = -1
            else:
                final_signal = 0
            
            signal_source = ", ".join(signal_sources) if signal_sources else "no_consensus"
        else:
            final_signal = 0
            signal_strength = 0.0
            signal_source = "no_active_strategies"
        
        return final_signal, signal_strength, signal_source
    
    def get_position_size(self, signal: int, current_price: float, 
                         portfolio_value: float) -> float:
        """Calculate position size based on regime"""
        if signal == 0:
            return 0
        
        # Base position size
        base_size = 0.25  # 25% of portfolio
        
        # Adjust for regime
        regime_multipliers = {
            MarketRegime.TRENDING_UP: 1.2,
            MarketRegime.TRENDING_DOWN: 0.8,
            MarketRegime.SIDEWAYS: 1.0,
            MarketRegime.VOLATILE: 0.6,
            MarketRegime.UNKNOWN: 0.5
        }
        
        multiplier = regime_multipliers.get(self.current_regime, 0.8)
        
        # Adjust for confidence
        confidence_factor = min(1.0, self.regime_confidence + 0.3)
        
        # Calculate final position size
        position_size = base_size * multiplier * confidence_factor
        position_value = portfolio_value * position_size
        
        return position_value / current_price if current_price > 0 else 0