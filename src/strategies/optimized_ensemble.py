"""
Optimized Ensemble Strategy
Uses pre-optimized individual strategies with adaptive weighting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import os
from datetime import datetime

try:
    from .ensemble import EnsembleStrategy
    from .base import BaseStrategy
    from ..actions.market_regime import MarketRegimeDetector, MarketRegime
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from strategies.ensemble import EnsembleStrategy
    from strategies.base import BaseStrategy
    from actions.market_regime import MarketRegimeDetector, MarketRegime


class AdaptiveWeightManager:
    """Manages adaptive weights for strategies based on recent performance"""
    
    def __init__(self, window: int = 100, min_weight: float = 0.1, max_weight: float = 0.5):
        self.window = window
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.performance_history = {}
        self.current_weights = {}
        
    def update_performance(self, strategy: str, regime: str, return_pct: float):
        """Update strategy performance"""
        key = f"{regime}_{strategy}"
        if key not in self.performance_history:
            self.performance_history[key] = []
        
        self.performance_history[key].append(return_pct)
        
        # Keep only recent history
        if len(self.performance_history[key]) > self.window:
            self.performance_history[key].pop(0)
    
    def calculate_weights(self, regime: str, strategies: List[str]) -> Dict[str, float]:
        """Calculate adaptive weights based on recent performance"""
        weights = {}
        performances = {}
        
        # Calculate performance scores
        for strategy in strategies:
            key = f"{regime}_{strategy}"
            if key in self.performance_history and len(self.performance_history[key]) >= 10:
                # Calculate risk-adjusted performance
                returns = self.performance_history[key]
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                
                # Sharpe-like score
                if std_return > 0:
                    score = avg_return / std_return
                else:
                    score = avg_return
                
                performances[strategy] = max(0, score)  # No negative weights
            else:
                # Default performance for new strategies
                performances[strategy] = 1.0
        
        # Convert to weights
        total_performance = sum(performances.values())
        
        if total_performance > 0:
            for strategy, performance in performances.items():
                weight = performance / total_performance
                # Apply min/max constraints
                weight = max(self.min_weight, min(self.max_weight, weight))
                weights[strategy] = weight
        else:
            # Equal weights if no performance data
            equal_weight = 1.0 / len(strategies)
            for strategy in strategies:
                weights[strategy] = equal_weight
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            for strategy in weights:
                weights[strategy] /= total_weight
        
        self.current_weights[regime] = weights
        return weights


class OptimizedEnsembleStrategy(EnsembleStrategy):
    """
    Enhanced ensemble strategy that uses pre-optimized individual strategies
    with adaptive weighting based on recent performance
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        """Initialize optimized ensemble strategy"""
        # Load optimized strategy parameters
        self.optimized_params = self._load_optimized_parameters()
        
        # Initialize base ensemble
        super().__init__(parameters)
        
        # Initialize adaptive weight manager
        self.weight_manager = AdaptiveWeightManager(
            window=parameters.get('performance_window', 100),
            min_weight=parameters.get('min_strategy_weight', 0.1),
            max_weight=parameters.get('max_strategy_weight', 0.5)
        )
        
        # Track strategy performance
        self.strategy_returns = {}
        self.last_prices = {}
        
        # Ensemble configuration
        self.use_adaptive_weights = parameters.get('adaptive_weights', True)
        self.rebalance_frequency = parameters.get('rebalance_frequency', 50)
        self.rebalance_counter = 0
        
    def _load_optimized_parameters(self) -> Dict[str, Dict]:
        """Load pre-optimized strategy parameters"""
        optimized_params = {}
        optimization_dir = "results/optimization/strategies"
        
        if os.path.exists(optimization_dir):
            for file in os.listdir(optimization_dir):
                if file.endswith("_optimized.json"):
                    strategy = file.replace("_optimized.json", "")
                    with open(os.path.join(optimization_dir, file), 'r') as f:
                        data = json.load(f)
                        optimized_params[strategy] = data["optimized_params"]
        
        # Also check for ensemble optimization results
        ensemble_file = "results/optimization/ensemble_optimized.json"
        if os.path.exists(ensemble_file):
            with open(ensemble_file, 'r') as f:
                ensemble_data = json.load(f)
                if "strategy_parameters" in ensemble_data:
                    optimized_params.update(ensemble_data["strategy_parameters"])
        
        return optimized_params
    
    def _get_strategy_config(self, strategy_name: str) -> Dict:
        """Get configuration for a strategy, preferring optimized parameters"""
        # Check if we have optimized parameters
        if strategy_name in self.optimized_params:
            return self.optimized_params[strategy_name]
        
        # Otherwise use default configuration
        return super().get_strategy_config(strategy_name)
    
    def _update_active_strategies(self, df: pd.DataFrame):
        """Update active strategies with optimized parameters"""
        # Skip if in optimization mode
        if self.optimization_mode:
            self.active_strategies = {}
            return
        
        regime_config = self.config.get_regime_strategies(self.current_regime)
        all_strategies = regime_config['primary'] + regime_config.get('secondary', [])
        
        # Initialize strategies with optimized parameters
        for strategy_name in all_strategies:
            if strategy_name not in self.strategy_instances:
                # Get optimized or default config
                strategy_config = self._get_strategy_config(strategy_name)
                base_strategy_name = self._get_base_strategy_name(strategy_name)
                
                try:
                    from .registry import get_strategy
                    self.strategy_instances[strategy_name] = get_strategy(
                        base_strategy_name, strategy_config
                    )
                except Exception as e:
                    print(f"⚠️  Failed to load strategy {strategy_name}: {e}")
                    continue
        
        # Select active strategies based on performance
        if self.use_adaptive_weights and self.weight_manager.current_weights.get(self.current_regime.value):
            # Use adaptive selection based on weights
            weights = self.weight_manager.current_weights[self.current_regime.value]
            
            # Select strategies with highest weights
            sorted_strategies = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            active_count = min(len(sorted_strategies), 
                             len(regime_config['primary']) + len(regime_config.get('secondary', [])) // 2)
            
            self.active_strategies = {
                name: self.strategy_instances[name]
                for name, _ in sorted_strategies[:active_count]
                if name in self.strategy_instances
            }
        else:
            # Default selection
            self.active_strategies = {
                name: self.strategy_instances[name]
                for name in all_strategies
                if name in self.strategy_instances
            }
    
    def _aggregate_signals(self, strategy_signals: Dict[str, int], 
                          current_data: pd.Series) -> Tuple[int, float, str]:
        """Aggregate signals using adaptive weights"""
        if not strategy_signals:
            return 0, 0.0, "no_signal"
        
        # Get current regime
        regime_name = self.current_regime.value
        
        # Get weights
        if self.use_adaptive_weights:
            # Calculate adaptive weights if needed
            self.rebalance_counter += 1
            if (self.rebalance_counter >= self.rebalance_frequency or 
                regime_name not in self.weight_manager.current_weights):
                weights = self.weight_manager.calculate_weights(
                    regime_name, 
                    list(strategy_signals.keys())
                )
                self.rebalance_counter = 0
            else:
                weights = self.weight_manager.current_weights.get(
                    regime_name,
                    {s: 1.0/len(strategy_signals) for s in strategy_signals}
                )
        else:
            # Use static weights from configuration
            regime_config = self.config.get_regime_strategies(self.current_regime)
            primary_strategies = regime_config.get('primary', [])
            secondary_strategies = regime_config.get('secondary', [])
            config_weights = regime_config.get('weights', {'primary': 0.7, 'secondary': 0.3})
            
            weights = {}
            for strategy in strategy_signals:
                if strategy in primary_strategies:
                    weights[strategy] = config_weights.get('primary', 0.7) / len(primary_strategies)
                elif strategy in secondary_strategies:
                    weights[strategy] = config_weights.get('secondary', 0.3) / len(secondary_strategies)
                else:
                    weights[strategy] = 0.1
        
        # Calculate weighted signal
        weighted_sum = 0.0
        total_weight = 0.0
        signal_sources = []
        
        for strategy_name, signal in strategy_signals.items():
            if signal != 0 and strategy_name in weights:
                weight = weights[strategy_name]
                weighted_sum += signal * weight
                total_weight += weight
                signal_sources.append(f"{strategy_name}({weight:.2f})")
        
        # Determine final signal
        if total_weight > 0:
            signal_strength = abs(weighted_sum / total_weight)
            
            # Adaptive threshold based on regime
            thresholds = {
                'trending_up': 0.4,
                'trending_down': 0.6,
                'sideways': 0.5,
                'volatile': 0.7,
                'unknown': 0.6
            }
            threshold = thresholds.get(regime_name, 0.5)
            
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
    
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate signals and track performance"""
        # Generate signals using parent method
        df = super().generate_signals(df, market)
        
        # Track performance for adaptive weighting
        if self.use_adaptive_weights and len(df) > 1:
            current_price = df['trade_price'].iloc[-1] if 'trade_price' in df.columns else df['close'].iloc[-1]
            
            # Update strategy performance
            if market in self.last_prices:
                return_pct = (current_price - self.last_prices[market]) / self.last_prices[market]
                
                # Attribute returns to active strategies
                for strategy_name in self.active_strategies:
                    self.weight_manager.update_performance(
                        strategy_name,
                        self.current_regime.value,
                        return_pct
                    )
            
            self.last_prices[market] = current_price
        
        return df
    
    def get_position_size(self, signal: int, current_price: float, 
                         portfolio_value: float) -> float:
        """Calculate position size with regime and performance adjustments"""
        if signal == 0:
            return 0
        
        # Base position size from parent
        base_size = super().get_position_size(signal, current_price, portfolio_value)
        
        # Adjust based on strategy confidence
        if self.use_adaptive_weights and hasattr(self, 'weight_manager'):
            regime_name = self.current_regime.value
            if regime_name in self.weight_manager.current_weights:
                # Reduce size if weights are dispersed (low confidence)
                weights = list(self.weight_manager.current_weights[regime_name].values())
                weight_entropy = -sum(w * np.log(w + 1e-10) for w in weights if w > 0)
                max_entropy = -np.log(1.0 / len(weights))
                
                # Confidence factor: 0.5 to 1.0
                confidence = 0.5 + 0.5 * (1 - weight_entropy / max_entropy)
                base_size *= confidence
        
        return base_size