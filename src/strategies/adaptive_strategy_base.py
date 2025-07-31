"""
Adaptive Strategy Base Class
Provides regime-aware and rolling optimization capabilities for all strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod
try:
    from .base import BaseStrategy
    from ..actions.market_regime import MarketRegimeDetector, MarketRegime
except ImportError:
    # Handle when running from different contexts
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from strategies.base import BaseStrategy
    from actions.market_regime import MarketRegimeDetector, MarketRegime
import warnings
warnings.filterwarnings('ignore')


class AdaptiveStrategyMixin:
    """
    Mixin class that adds adaptive capabilities to any strategy
    Features:
    - Rolling window parameter optimization
    - Regime-based parameter adjustment  
    - Market microstructure regime detection
    - Dynamic parameter adaptation
    """
    
    def __init__(self, enable_adaptation: bool = True, **kwargs):
        super().__init__(**kwargs)
        
        self.enable_adaptation = enable_adaptation
        
        # Adaptation configuration
        self.adaptation_config = {
            'rolling_window': 1440,  # 24 hours for parameter reoptimization
            'reoptimize_frequency': 240,  # Every 4 hours
            'regime_lookback': 120,  # 2 hours for regime detection
            'parameter_memory': 5,  # Keep last 5 parameter sets
            'min_performance_samples': 60  # Minimum samples for performance evaluation
        }
        
        # State tracking
        self.parameter_history = []
        self.performance_history = []
        self.current_regime = MarketRegime.UNKNOWN
        self.last_optimization_idx = 0
        
        # Initialize regime detector
        self.regime_detector = MarketRegimeDetector()
        
        # Regime-specific parameters
        self.regime_parameters = {}
        self._initialize_regime_parameters()
    
    def _initialize_regime_parameters(self):
        """Initialize parameter sets for different market regimes"""
        base_params = getattr(self, 'parameters', {})
        
        # Create regime-specific parameter variations
        self.regime_parameters = {
            MarketRegime.TRENDING_UP: self._adjust_params_for_regime(base_params, 'trending_up'),
            MarketRegime.TRENDING_DOWN: self._adjust_params_for_regime(base_params, 'trending_down'),
            MarketRegime.SIDEWAYS: self._adjust_params_for_regime(base_params, 'sideways'),
            MarketRegime.VOLATILE: self._adjust_params_for_regime(base_params, 'volatile'),
            MarketRegime.UNKNOWN: base_params.copy()
        }
    
    def _adjust_params_for_regime(self, base_params: Dict, regime: str) -> Dict:
        """Adjust parameters based on market regime"""
        adjusted = base_params.copy()
        
        # Regime-specific adjustments
        if regime == 'trending_up' or regime == 'trending_down':
            # Trending markets: faster indicators, looser thresholds
            for param, value in adjusted.items():
                if 'period' in param and isinstance(value, int):
                    adjusted[param] = max(5, int(value * 0.8))  # 20% faster
                elif 'threshold' in param and isinstance(value, float):
                    adjusted[param] = value * 1.2  # 20% looser
                    
        elif regime == 'sideways':
            # Sideways markets: slower indicators, tighter thresholds
            for param, value in adjusted.items():
                if 'period' in param and isinstance(value, int):
                    adjusted[param] = int(value * 1.3)  # 30% slower
                elif 'threshold' in param and isinstance(value, float):
                    adjusted[param] = value * 0.8  # 20% tighter
                    
        elif regime == 'volatile':
            # Volatile markets: much slower indicators, much tighter thresholds
            for param, value in adjusted.items():
                if 'period' in param and isinstance(value, int):
                    adjusted[param] = int(value * 1.5)  # 50% slower
                elif 'threshold' in param and isinstance(value, float):
                    adjusted[param] = value * 0.6  # 40% tighter
        
        return adjusted
    
    def detect_market_regime(self, df: pd.DataFrame, current_idx: int) -> MarketRegime:
        """Detect current market regime"""
        if current_idx < self.adaptation_config['regime_lookback']:
            return MarketRegime.UNKNOWN
        
        # Get recent data for regime detection
        start_idx = max(0, current_idx - self.adaptation_config['regime_lookback'])
        regime_data = df.iloc[start_idx:current_idx+1].copy()
        
        # Use regime detector
        regime_result, regime_indicators = self.regime_detector.detect_regime(regime_data)
        return regime_result
    
    def should_reoptimize(self, current_idx: int) -> bool:
        """Check if parameters should be reoptimized"""
        if not self.enable_adaptation:
            return False
            
        # Reoptimize based on frequency
        idx_since_last = current_idx - self.last_optimization_idx
        return idx_since_last >= self.adaptation_config['reoptimize_frequency']
    
    def rolling_parameter_optimization(self, df: pd.DataFrame, current_idx: int) -> Dict[str, Any]:
        """Perform rolling window parameter optimization"""
        if current_idx < self.adaptation_config['rolling_window']:
            return self.parameters.copy()
        
        # Get optimization window
        start_idx = current_idx - self.adaptation_config['rolling_window']
        optimization_data = df.iloc[start_idx:current_idx].copy()
        
        # Get current regime
        current_regime = self.detect_market_regime(df, current_idx)
        
        # Start with regime-adjusted parameters
        base_regime_params = self.regime_parameters.get(current_regime, self.parameters)
        
        # Perform simple grid search optimization (simplified for performance)
        best_params = self._optimize_parameters_simple(optimization_data, base_regime_params)
        
        # Store optimization results
        self.parameter_history.append({
            'timestamp': df.index[current_idx] if current_idx < len(df) else None,
            'regime': current_regime,
            'parameters': best_params.copy(),
            'performance': self._evaluate_parameters(optimization_data, best_params)
        })
        
        # Keep only recent history
        if len(self.parameter_history) > self.adaptation_config['parameter_memory']:
            self.parameter_history = self.parameter_history[-self.adaptation_config['parameter_memory']:]
        
        self.last_optimization_idx = current_idx
        return best_params
    
    def _optimize_parameters_simple(self, data: pd.DataFrame, base_params: Dict) -> Dict:
        """Simple parameter optimization using grid search on key parameters"""
        best_params = base_params.copy()
        best_score = -np.inf
        
        # Define parameter variations to test
        param_variations = self._get_parameter_variations(base_params)
        
        for variation in param_variations:
            try:
                # Test parameters
                test_params = base_params.copy()
                test_params.update(variation)
                
                # Evaluate performance (simplified)
                score = self._evaluate_parameters(data, test_params)
                
                if score > best_score:
                    best_score = score
                    best_params = test_params.copy()
                    
            except Exception as e:
                continue  # Skip failed parameter combinations
        
        return best_params
    
    def _get_parameter_variations(self, base_params: Dict) -> List[Dict]:
        """Generate parameter variations for optimization"""
        variations = []
        
        # Test variations for common parameters
        for param_name, param_value in base_params.items():
            if isinstance(param_value, int) and 'period' in param_name:
                # Test ±20% for period parameters
                variations.extend([
                    {param_name: max(3, int(param_value * 0.8))},
                    {param_name: int(param_value * 1.2)}
                ])
            elif isinstance(param_value, float) and 'threshold' in param_name:
                # Test ±30% for threshold parameters
                variations.extend([
                    {param_name: param_value * 0.7},
                    {param_name: param_value * 1.3}
                ])
        
        # Add base parameters (no change)
        variations.append({})
        
        return variations[:10]  # Limit to 10 variations for performance
    
    def _evaluate_parameters(self, data: pd.DataFrame, test_params: Dict) -> float:
        """Evaluate parameter performance using Sharpe ratio"""
        try:
            # Create temporary strategy instance with test parameters
            if hasattr(self, 'calculate_indicators') and hasattr(self, 'generate_signals'):
                # Calculate indicators with test parameters
                temp_params = self.parameters
                self.parameters = test_params
                
                test_data = self.calculate_indicators(data.copy())
                test_data = self.generate_signals(test_data, 'test')
                
                # Restore original parameters
                self.parameters = temp_params
                
                # Calculate returns
                if 'signal' in test_data.columns and 'trade_price' in test_data.columns:
                    returns = np.log(test_data['trade_price'] / test_data['trade_price'].shift(1))
                    strategy_returns = returns * test_data['signal'].shift(1)
                    
                    # Calculate Sharpe ratio (annualized for 1-minute data)
                    if len(strategy_returns) > 10 and strategy_returns.std() > 0:
                        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(525600)
                        return sharpe
            
            return 0.0
            
        except Exception as e:
            return -999.0  # Very poor score for failed evaluations
    
    def get_adaptive_parameters(self, df: pd.DataFrame, current_idx: int) -> Dict[str, Any]:
        """Get current optimal parameters based on adaptation logic"""
        if not self.enable_adaptation:
            return self.parameters
        
        # Check if reoptimization is needed
        if self.should_reoptimize(current_idx):
            # Perform rolling optimization
            optimized_params = self.rolling_parameter_optimization(df, current_idx)
            self.parameters = optimized_params
            
        # Apply regime-based adjustments to current parameters
        current_regime = self.detect_market_regime(df, current_idx)
        if current_regime != self.current_regime:
            self.current_regime = current_regime
            
            # Blend optimized parameters with regime adjustments
            regime_params = self.regime_parameters.get(current_regime, self.parameters)
            
            # Weighted blend: 70% optimized, 30% regime-adjusted
            blended_params = {}
            for param_name in self.parameters.keys():
                if param_name in regime_params:
                    if isinstance(self.parameters[param_name], (int, float)):
                        blended_params[param_name] = (
                            self.parameters[param_name] * 0.7 + 
                            regime_params[param_name] * 0.3
                        )
                        if isinstance(self.parameters[param_name], int):
                            blended_params[param_name] = int(blended_params[param_name])
                    else:
                        blended_params[param_name] = self.parameters[param_name]
                else:
                    blended_params[param_name] = self.parameters[param_name]
            
            return blended_params
        
        return self.parameters
    
    def get_adaptation_status(self) -> Dict[str, Any]:
        """Get current adaptation status"""
        return {
            'adaptation_enabled': self.enable_adaptation,
            'current_regime': self.current_regime.value if self.current_regime else 'unknown',
            'last_optimization_idx': self.last_optimization_idx,
            'parameter_history_length': len(self.parameter_history),
            'current_parameters': self.parameters.copy()
        }


class AdaptiveStrategy(AdaptiveStrategyMixin, BaseStrategy):
    """
    Base class for adaptive strategies that combines BaseStrategy with AdaptiveStrategyMixin
    """
    
    def __init__(self, parameters: Dict[str, Any], enable_adaptation: bool = True):
        super().__init__(parameters=parameters, enable_adaptation=enable_adaptation)
    
    def generate_signals_adaptive(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate signals with adaptive parameter optimization"""
        df['signal'] = np.zeros(len(df), dtype=np.int8)
        
        # Process data with adaptive parameters
        for i in range(len(df)):
            if i < 100:  # Skip early periods
                continue
                
            # Get adaptive parameters for current time
            current_params = self.get_adaptive_parameters(df, i)
            
            # Temporarily update parameters
            original_params = self.parameters
            self.parameters = current_params
            
            # Generate signal for current row
            if hasattr(self, '_should_buy') and hasattr(self, '_should_sell'):
                current_row = df.iloc[i]
                
                if self._should_buy(current_row, df.iloc[:i+1]):
                    df.iloc[i, df.columns.get_loc('signal')] = 1
                elif self._should_sell(current_row, df.iloc[:i+1]):
                    df.iloc[i, df.columns.get_loc('signal')] = -1
            
            # Restore original parameters
            self.parameters = original_params
        
        return df


def create_adaptive_strategy(strategy_class, parameters: Dict[str, Any], enable_adaptation: bool = True):
    """
    Factory function to create an adaptive version of any strategy
    """
    class AdaptiveWrapper(AdaptiveStrategyMixin, strategy_class):
        def __init__(self, params, adaptation=True):
            super().__init__(parameters=params, enable_adaptation=adaptation)
        
        def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
            """Override to use adaptive signal generation"""
            if self.enable_adaptation:
                return self.generate_signals_adaptive(df, market)
            else:
                return super().generate_signals(df, market)
    
    return AdaptiveWrapper(parameters, enable_adaptation)


def main():
    """Example usage"""
    print("Adaptive Strategy Base loaded successfully")


if __name__ == "__main__":
    main()