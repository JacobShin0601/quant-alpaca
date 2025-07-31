"""
Adaptive Strategy Optimizer for Dynamic Parameter Optimization
Extends the base strategy optimizer to handle adaptive strategies with rolling optimization
"""

import optuna
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import json
import os
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

try:
    from .strategy_optimizer import StrategyOptimizer
    from ..backtesting.engine import BacktestEngine
    from ..strategies import get_strategy, STRATEGIES
    from ..strategies.adaptive_strategy_base import create_adaptive_strategy
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from optimization.strategy_optimizer import StrategyOptimizer
    from backtesting.engine import BacktestEngine
    from strategies import get_strategy, STRATEGIES
    from strategies.adaptive_strategy_base import create_adaptive_strategy


class AdaptiveStrategyOptimizer(StrategyOptimizer):
    """Enhanced optimizer for adaptive strategies with rolling optimization"""
    
    def __init__(self, config_path: str = "config/config_optimize.json"):
        """Initialize adaptive optimizer"""
        super().__init__(config_path)
        
        # Additional configuration for adaptive optimization
        self.adaptive_config = {
            'enable_rolling_optimization': True,
            'adaptation_window': 1440,  # 24 hours
            'reoptimization_frequency': 240,  # 4 hours
            'regime_detection': True,
            'walk_forward_validation': True,
            'adaptive_parameter_penalty': 0.1  # Penalty for parameter instability
        }
        
        # Update config with adaptive settings if present
        if 'adaptive_optimization' in self.config:
            self.adaptive_config.update(self.config['adaptive_optimization'])
        
        self.logger.info("Initialized Adaptive Strategy Optimizer")
    
    def optimize_adaptive_strategy(self, strategy_name: str, data: Dict[str, pd.DataFrame], 
                                 markets: List[str], train_ratio: float = 0.7) -> Dict[str, Any]:
        """Optimize adaptive strategy with rolling validation"""
        self.logger.info(f"Starting adaptive optimization for {strategy_name}")
        
        # Prepare data for adaptive optimization
        if len(markets) == 1:
            market = markets[0]
            market_data = data[market].copy()
        else:
            # For multi-market strategies, we'll use the first market
            market = markets[0]
            market_data = data[market].copy()
        
        # Check if strategy supports adaptation
        strategy_class = STRATEGIES.get(strategy_name)
        if not strategy_class:
            self.logger.error(f"Strategy {strategy_name} not found")
            return {}
        
        # Get parameter space
        param_space = self.config["strategy_params"].get(strategy_name, {})
        if not param_space:
            self.logger.warning(f"No parameter space defined for {strategy_name}")
            return {}
        
        # Create adaptive version if not already adaptive
        is_native_adaptive = hasattr(strategy_class, 'enable_adaptation')
        
        if self.adaptive_config['enable_rolling_optimization']:
            results = self._optimize_with_rolling_validation(
                strategy_name, market_data, market, param_space, train_ratio
            )
        else:
            # Fall back to standard optimization
            results = self.optimize_strategy(strategy_name, data, markets, train_ratio)
        
        # Add adaptive-specific metrics
        results['adaptive_optimization'] = {
            'enabled': True,
            'rolling_validation': self.adaptive_config['enable_rolling_optimization'],
            'regime_detection': self.adaptive_config['regime_detection'],
            'adaptation_window': self.adaptive_config['adaptation_window'],
            'reoptimization_frequency': self.adaptive_config['reoptimization_frequency']
        }
        
        return results
    
    def _optimize_with_rolling_validation(self, strategy_name: str, data: pd.DataFrame, 
                                        market: str, param_space: Dict, 
                                        train_ratio: float) -> Dict[str, Any]:
        """Optimize with rolling window validation for adaptive strategies"""
        self.logger.info(f"Running rolling validation optimization for {strategy_name}")
        
        # Calculate rolling windows
        total_length = len(data)
        initial_train_size = int(total_length * train_ratio)
        window_size = self.adaptive_config['adaptation_window']
        reopt_frequency = self.adaptive_config['reoptimization_frequency']
        
        # Results storage
        rolling_results = []
        parameter_stability_scores = []
        best_overall_params = {}
        best_overall_score = -np.inf
        
        # Initial optimization on first training window
        initial_data = data.iloc[:initial_train_size].copy()
        
        study = optuna.create_study(
            direction=self.config["optimization"]["direction"],
            sampler=self._get_sampler(),
            pruner=self._get_pruner()
        )
        
        # Initial optimization
        objective_func = lambda trial: self._adaptive_objective(
            trial, strategy_name, initial_data, market, param_space
        )
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study.optimize(
            objective_func,
            n_trials=max(20, self.config["optimization"]["n_trials"] // 4),  # Fewer trials per window
            timeout=self.config["optimization"]["timeout"] // 4
        )
        
        current_best_params = study.best_params.copy()
        best_overall_params = current_best_params.copy()
        best_overall_score = study.best_value
        
        # Rolling forward validation
        current_pos = initial_train_size
        
        while current_pos + window_size < total_length:
            # Define current training and validation windows
            train_start = max(0, current_pos - window_size)
            train_end = current_pos
            val_start = current_pos
            val_end = min(current_pos + reopt_frequency, total_length)
            
            train_window = data.iloc[train_start:train_end].copy()
            val_window = data.iloc[val_start:val_end].copy()
            
            # Reoptimize parameters on current training window
            if len(train_window) > 100:  # Minimum data requirement
                reopt_study = optuna.create_study(
                    direction=self.config["optimization"]["direction"],
                    sampler=self._get_sampler()
                )
                
                reopt_objective = lambda trial: self._adaptive_objective(
                    trial, strategy_name, train_window, market, param_space
                )
                
                reopt_study.optimize(
                    reopt_objective,
                    n_trials=max(10, self.config["optimization"]["n_trials"] // 8),
                    timeout=self.config["optimization"]["timeout"] // 8
                )
                
                new_params = reopt_study.best_params.copy()
                
                # Calculate parameter stability
                stability_score = self._calculate_parameter_stability(current_best_params, new_params)
                parameter_stability_scores.append(stability_score)
                
                # Apply parameter stability penalty
                adjusted_score = reopt_study.best_value - (self.adaptive_config['adaptive_parameter_penalty'] * (1 - stability_score))
                
                # Update best parameters if improved
                if adjusted_score > best_overall_score:
                    best_overall_params = new_params.copy()
                    best_overall_score = adjusted_score
                
                current_best_params = new_params
            
            # Validate on current validation window
            if len(val_window) > 20:
                val_result = self._validate_adaptive_params(
                    strategy_name, current_best_params, val_window, market
                )
                
                rolling_results.append({
                    'window_start': val_start,
                    'window_end': val_end,
                    'parameters': current_best_params.copy(),
                    'performance': val_result,
                    'stability_score': parameter_stability_scores[-1] if parameter_stability_scores else 1.0
                })
            
            current_pos += reopt_frequency
        
        # Calculate overall rolling validation metrics
        overall_metrics = self._calculate_rolling_metrics(rolling_results)
        
        # Final validation on remaining test data
        test_start = int(total_length * train_ratio)
        test_data = data.iloc[test_start:].copy()
        
        final_test_results = self._validate_adaptive_params(
            strategy_name, best_overall_params, test_data, market
        )
        
        return {
            'best_params': best_overall_params,
            'train_performance': best_overall_score,
            'test_performance': final_test_results,
            'rolling_validation': {
                'results': rolling_results,
                'overall_metrics': overall_metrics,
                'parameter_stability': np.mean(parameter_stability_scores) if parameter_stability_scores else 1.0,
                'stability_scores': parameter_stability_scores
            },
            'optimization_method': 'rolling_adaptive'
        }
    
    def _adaptive_objective(self, trial: optuna.Trial, strategy_name: str, 
                           data: pd.DataFrame, market: str, param_space: Dict) -> float:
        """Objective function for adaptive strategy optimization"""
        # Sample parameters
        params = {}
        for param_name, param_config in param_space.items():
            if param_config["type"] == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    step=param_config.get("step", 1)
                )
            elif param_config["type"] == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    step=param_config.get("step", None)
                )
            elif param_config["type"] == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config["choices"]
                )
        
        # Create adaptive strategy
        strategy_class = STRATEGIES[strategy_name]
        
        # Check if strategy is already adaptive
        if hasattr(strategy_class, 'enable_adaptation'):
            strategy = strategy_class(params)
            if hasattr(strategy, 'enable_adaptation'):
                strategy.enable_adaptation = True
        else:
            # Create adaptive wrapper
            strategy = create_adaptive_strategy(strategy_class, params, enable_adaptation=True)
        
        # Run backtest with adaptive strategy
        config = self.config.copy()
        config['strategy'] = {
            'name': strategy_name,
            'parameters': params
        }
        
        try:
            engine = BacktestEngine(config)
            # Override strategy with adaptive version
            engine.strategy = strategy
            engine.strategy_is_adaptive = True
            
            results = engine.run_backtest({market: data})
            
            # Get objective value
            objective_name = self.config["optimization"]["objective"]
            if objective_name in results:
                return results[objective_name]
            else:
                self.logger.warning(f"Objective {objective_name} not found in results")
                return -999.0
                
        except Exception as e:
            self.logger.warning(f"Backtest failed: {e}")
            return -999.0
    
    def _validate_adaptive_params(self, strategy_name: str, params: Dict, 
                                 data: pd.DataFrame, market: str) -> Dict:
        """Validate parameters with adaptive strategy"""
        try:
            strategy_class = STRATEGIES[strategy_name]
            
            # Create adaptive strategy
            if hasattr(strategy_class, 'enable_adaptation'):
                strategy = strategy_class(params)
                if hasattr(strategy, 'enable_adaptation'):
                    strategy.enable_adaptation = True
            else:
                strategy = create_adaptive_strategy(strategy_class, params, enable_adaptation=True)
            
            config = self.config.copy()
            config['strategy'] = {
                'name': strategy_name,
                'parameters': params
            }
            
            engine = BacktestEngine(config)
            engine.strategy = strategy
            engine.strategy_is_adaptive = True
            
            results = engine.run_backtest({market: data})
            
            # Extract clean results
            return {
                "total_return": results.get("total_return", -999),
                "sharpe_ratio": results.get("sharpe_ratio", -999),
                "sortino_ratio": results.get("sortino_ratio", -999),
                "max_drawdown": results.get("max_drawdown", -999),
                "total_trades": results.get("total_trades", 0),
                "calmar_ratio": results.get("calmar_ratio", 0),
                "volatility": results.get("volatility", 0)
            }
            
        except Exception as e:
            self.logger.warning(f"Validation failed: {e}")
            return {
                "total_return": -999,
                "sharpe_ratio": -999,
                "sortino_ratio": -999,
                "max_drawdown": -999,
                "total_trades": 0,
                "calmar_ratio": 0,
                "volatility": 0
            }
    
    def _calculate_parameter_stability(self, old_params: Dict, new_params: Dict) -> float:
        """Calculate parameter stability score between two parameter sets"""
        if not old_params or not new_params:
            return 1.0
        
        stability_scores = []
        
        for param_name in old_params.keys():
            if param_name in new_params:
                old_val = old_params[param_name]
                new_val = new_params[param_name]
                
                if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                    # Numerical parameter - calculate relative change
                    if old_val != 0:
                        relative_change = abs(new_val - old_val) / abs(old_val)
                        stability_score = max(0, 1 - relative_change)
                    else:
                        stability_score = 1.0 if new_val == 0 else 0.0
                else:
                    # Categorical parameter
                    stability_score = 1.0 if old_val == new_val else 0.0
                
                stability_scores.append(stability_score)
        
        return np.mean(stability_scores) if stability_scores else 1.0
    
    def _calculate_rolling_metrics(self, rolling_results: List[Dict]) -> Dict:
        """Calculate overall metrics from rolling validation results"""
        if not rolling_results:
            return {}
        
        # Extract performance metrics
        returns = [r['performance'].get('total_return', 0) for r in rolling_results]
        sharpe_ratios = [r['performance'].get('sharpe_ratio', 0) for r in rolling_results]
        max_drawdowns = [r['performance'].get('max_drawdown', 0) for r in rolling_results]
        stability_scores = [r.get('stability_score', 1.0) for r in rolling_results]
        
        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'mean_sharpe': np.mean(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'worst_max_drawdown': min(max_drawdowns) if max_drawdowns else 0,
            'mean_stability': np.mean(stability_scores),
            'min_stability': min(stability_scores) if stability_scores else 1.0,
            'total_windows': len(rolling_results)
        }


def main():
    """Example usage"""
    print("Adaptive Strategy Optimizer loaded successfully")


if __name__ == "__main__":
    main()