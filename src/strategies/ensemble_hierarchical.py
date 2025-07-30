"""
Hierarchical Ensemble Strategy
Multi-level ensemble with strategy grouping and hierarchical aggregation
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


class HierarchicalEnsembleStrategy(BaseStrategy):
    """
    Hierarchical ensemble with three levels:
    Level 1: Individual strategies (optimized)
    Level 2: Strategy groups (momentum, mean-reversion, breakout)
    Level 3: Final ensemble aggregation
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        """Initialize hierarchical ensemble strategy"""
        super().__init__(parameters)
        
        # Load optimized parameters
        self.optimized_params = self._load_optimized_parameters()
        
        # Initialize components
        self.regime_detector = MarketRegimeDetector(parameters.get('regime_config'))
        
        # Current market
        self.current_market = None
        
        # Strategy groups
        self.strategy_groups = {
            'momentum': {
                'strategies': ['supertrend', 'ichimoku', 'macd', 'aroon'],
                'weight': 0.4,
                'regimes': [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]
            },
            'mean_reversion': {
                'strategies': ['mean_reversion', 'bollinger_bands', 'keltner_channels', 'stochastic'],
                'weight': 0.35,
                'regimes': [MarketRegime.SIDEWAYS, MarketRegime.TRENDING_DOWN]
            },
            'breakout': {
                'strategies': ['atr_breakout', 'donchian_channels', 'volume_profile'],
                'weight': 0.25,
                'regimes': [MarketRegime.VOLATILE, MarketRegime.TRENDING_UP]
            },
            'hybrid': {
                'strategies': ['vwap', 'fibonacci_retracement'],
                'weight': 0.2,
                'regimes': [MarketRegime.UNKNOWN, MarketRegime.SIDEWAYS]
            }
        }
        
        # Hierarchical weights
        self.intra_group_weights = parameters.get('intra_group_weights', 'equal')  # 'equal' or 'performance'
        self.inter_group_weights = parameters.get('inter_group_weights', 'regime')  # 'fixed' or 'regime'
        
        # State
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_confidence = 0.0
        self.strategy_instances = {}
        self.group_signals = {}
        self.strategy_performance = {}
        
        # Performance tracking for adaptive weights
        self.performance_window = parameters.get('performance_window', 50)
        self.performance_history = {}
        
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
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators for all levels"""
        # Detect regime
        regime, indicators = self.regime_detector.detect_regime(df)
        
        df['regime'] = regime.value
        df['regime_confidence'] = max(indicators.regime_probability.values())
        
        self.current_regime = regime
        self.regime_confidence = max(indicators.regime_probability.values())
        
        # Calculate indicators for all strategies in active groups
        if self.current_market:
            for group_name, group_config in self.strategy_groups.items():
                # Check if group is active in current regime
                if (self.inter_group_weights == 'fixed' or 
                    self.current_regime in group_config['regimes']):
                    
                    for strategy_name in group_config['strategies']:
                        strategy = self._get_strategy_instance(strategy_name, self.current_market)
                        
                        if strategy:
                            try:
                                strategy_df = strategy.calculate_indicators(df.copy())
                                
                                # Merge indicators
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
        """Generate hierarchical ensemble signals"""
        self.current_market = market
        
        df['signal'] = 0
        df['signal_strength'] = 0.0
        df['signal_source'] = ''
        
        if len(df) < 2:
            return df
        
        # Level 1: Collect individual strategy signals
        group_strategy_signals = {}
        
        for group_name, group_config in self.strategy_groups.items():
            # Check if group is active
            if (self.inter_group_weights == 'fixed' or 
                self.current_regime in group_config['regimes']):
                
                group_strategy_signals[group_name] = {}
                
                for strategy_name in group_config['strategies']:
                    strategy = self._get_strategy_instance(strategy_name, market)
                    
                    if strategy:
                        try:
                            strategy_df = strategy.generate_signals(df.copy(), market)
                            
                            if 'signal' in strategy_df and len(strategy_df) > 0:
                                signal = strategy_df['signal'].iloc[-1]
                                group_strategy_signals[group_name][strategy_name] = signal
                            else:
                                group_strategy_signals[group_name][strategy_name] = 0
                        except Exception as e:
                            self.logger.error(f"Error generating signals for {strategy_name}: {e}")
                            group_strategy_signals[group_name][strategy_name] = 0
        
        # Level 2: Aggregate within groups
        group_signals = {}
        group_strengths = {}
        
        for group_name, strategy_signals in group_strategy_signals.items():
            if strategy_signals:
                group_signal, group_strength = self._aggregate_group_signals(
                    group_name, strategy_signals
                )
                group_signals[group_name] = group_signal
                group_strengths[group_name] = group_strength
        
        # Level 3: Final aggregation across groups
        if group_signals and len(df) > 0:
            final_signal, signal_strength, signal_source = self._aggregate_final_signal(
                group_signals, group_strengths
            )
            
            df.iloc[-1, df.columns.get_loc('signal')] = final_signal
            df.iloc[-1, df.columns.get_loc('signal_strength')] = signal_strength
            df.iloc[-1, df.columns.get_loc('signal_source')] = signal_source
        
        return df
    
    def _aggregate_group_signals(self, group_name: str, 
                                signals: Dict[str, int]) -> Tuple[int, float]:
        """Aggregate signals within a strategy group"""
        if not signals:
            return 0, 0.0
        
        if self.intra_group_weights == 'equal':
            # Equal weighting within group
            weights = {s: 1.0/len(signals) for s in signals}
        else:
            # Performance-based weighting (would need performance tracking)
            # For now, use equal weights
            weights = {s: 1.0/len(signals) for s in signals}
        
        # Calculate weighted signal
        weighted_sum = sum(signals[s] * weights[s] for s in signals)
        total_weight = sum(weights.values())
        
        if total_weight > 0:
            normalized_signal = weighted_sum / total_weight
            
            # Group-specific thresholds
            thresholds = {
                'momentum': 0.5,
                'mean_reversion': 0.6,
                'breakout': 0.7,
                'hybrid': 0.5
            }
            
            threshold = thresholds.get(group_name, 0.5)
            
            if normalized_signal > threshold:
                return 1, abs(normalized_signal)
            elif normalized_signal < -threshold:
                return -1, abs(normalized_signal)
        
        return 0, 0.0
    
    def _aggregate_final_signal(self, group_signals: Dict[str, int], 
                               group_strengths: Dict[str, float]) -> Tuple[int, float, str]:
        """Final aggregation across strategy groups"""
        if not group_signals:
            return 0, 0.0, "no_signals"
        
        # Determine group weights
        if self.inter_group_weights == 'fixed':
            weights = {group: config['weight'] 
                      for group, config in self.strategy_groups.items()
                      if group in group_signals}
        else:
            # Regime-based weights
            weights = {}
            for group_name in group_signals:
                if self.current_regime in self.strategy_groups[group_name]['regimes']:
                    weights[group_name] = self.strategy_groups[group_name]['weight'] * 1.5
                else:
                    weights[group_name] = self.strategy_groups[group_name]['weight'] * 0.5
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {g: w/total_weight for g, w in weights.items()}
        
        # Calculate final signal
        weighted_sum = 0.0
        active_groups = []
        
        for group_name, signal in group_signals.items():
            if signal != 0 and group_name in weights:
                weight = weights[group_name]
                strength = group_strengths.get(group_name, 1.0)
                weighted_sum += signal * weight * strength
                active_groups.append(f"{group_name}({weight:.2f})")
        
        # Determine final signal
        signal_strength = abs(weighted_sum)
        
        # Hierarchical threshold based on number of agreeing groups
        agreement = sum(1 for s in group_signals.values() if s == np.sign(weighted_sum))
        
        if agreement >= 3:  # Strong agreement
            threshold = 0.3
        elif agreement >= 2:  # Moderate agreement
            threshold = 0.5
        else:  # Weak agreement
            threshold = 0.7
        
        if weighted_sum > threshold:
            final_signal = 1
        elif weighted_sum < -threshold:
            final_signal = -1
        else:
            final_signal = 0
        
        signal_source = f"H[{', '.join(active_groups)}]"
        
        return final_signal, signal_strength, signal_source
    
    def get_position_size(self, signal: int, current_price: float, 
                         portfolio_value: float) -> float:
        """Calculate position size with hierarchical confidence"""
        if signal == 0:
            return 0
        
        # Base size
        base_size = 0.25
        
        # Count agreeing groups
        agreement_count = 0
        total_groups = 0
        
        if hasattr(self, 'group_signals'):
            for group_signal in self.group_signals.values():
                if group_signal != 0:
                    total_groups += 1
                    if group_signal == signal:
                        agreement_count += 1
        
        # Agreement multiplier
        if total_groups > 0:
            agreement_ratio = agreement_count / total_groups
            agreement_mult = 0.5 + 0.5 * agreement_ratio  # 0.5 to 1.0
        else:
            agreement_mult = 0.7
        
        # Regime multiplier
        regime_multipliers = {
            MarketRegime.TRENDING_UP: 1.2,
            MarketRegime.TRENDING_DOWN: 0.8,
            MarketRegime.SIDEWAYS: 1.0,
            MarketRegime.VOLATILE: 0.6,
            MarketRegime.UNKNOWN: 0.7
        }
        
        regime_mult = regime_multipliers.get(self.current_regime, 0.8)
        
        # Calculate final size
        position_size = base_size * agreement_mult * regime_mult
        position_size = min(0.35, position_size)  # Max 35% of portfolio
        
        position_value = portfolio_value * position_size
        
        return position_value / current_price if current_price > 0 else 0