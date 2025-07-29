"""
Strategy Hyperparameter Optimizer using Optuna
"""

import optuna
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import json
import os
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

try:
    from ..backtesting.engine import BacktestEngine
    from ..strategies.registry import get_strategy, STRATEGIES
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from backtesting.engine import BacktestEngine
    from strategies.registry import get_strategy, STRATEGIES


class StrategyOptimizer:
    """Optimize strategy hyperparameters using Optuna"""
    
    def __init__(self, config_path: str = "config/config_optimize.json"):
        """Initialize optimizer with configuration"""
        self.config = self._load_config(config_path)
        self.results = {}
        self.best_params = {}
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load optimization configuration"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default optimization configuration"""
        return {
            "optimization": {
                "n_trials": 100,
                "n_jobs": -1,
                "timeout": 3600,  # 1 hour per strategy
                "objective": "sharpe_ratio",  # sharpe_ratio, sortino_ratio, total_return
                "direction": "maximize",
                "sampler": "TPE",  # TPE, CMA-ES, Random
                "pruner": "MedianPruner",
                "train_ratio": 0.7
            },
            "backtesting": {
                "initial_capital": 10000000,
                "commission_rate": 0.0005,
                "slippage_rate": 0.001,
                "min_data_points": 1000
            },
            "strategy_params": {
                "vwap": {
                    "vwap_period": {"type": "int", "low": 10, "high": 50, "step": 5},
                    "vwap_threshold": {"type": "float", "low": 0.001, "high": 0.02, "step": 0.001},
                    "volume_threshold": {"type": "float", "low": 1.0, "high": 2.5, "step": 0.1},
                    "strategy_variant": {"type": "categorical", "choices": ["mean_reversion", "trend_following", "breakout"]}
                },
                "bollinger_bands": {
                    "bb_period": {"type": "int", "low": 10, "high": 30, "step": 2},
                    "bb_std_dev": {"type": "float", "low": 1.5, "high": 3.0, "step": 0.25},
                    "lower_threshold": {"type": "float", "low": 0.05, "high": 0.25, "step": 0.05},
                    "upper_threshold": {"type": "float", "low": 0.75, "high": 0.95, "step": 0.05}
                },
                "macd": {
                    "fast_period": {"type": "int", "low": 8, "high": 15, "step": 1},
                    "slow_period": {"type": "int", "low": 20, "high": 30, "step": 2},
                    "signal_period": {"type": "int", "low": 5, "high": 12, "step": 1},
                    "use_histogram_filter": {"type": "categorical", "choices": [True, False]}
                },
                "stochastic": {
                    "k_period": {"type": "int", "low": 10, "high": 20, "step": 2},
                    "d_period": {"type": "int", "low": 3, "high": 5, "step": 1},
                    "oversold_level": {"type": "int", "low": 15, "high": 30, "step": 5},
                    "overbought_level": {"type": "int", "low": 70, "high": 85, "step": 5}
                },
                "ichimoku": {
                    "tenkan_period": {"type": "int", "low": 5, "high": 15, "step": 2},
                    "kijun_period": {"type": "int", "low": 20, "high": 35, "step": 3},
                    "senkou_b_period": {"type": "int", "low": 40, "high": 60, "step": 5},
                    "strategy_variant": {"type": "categorical", "choices": ["classic", "trend", "breakout"]}
                },
                "supertrend": {
                    "atr_period": {"type": "int", "low": 7, "high": 14, "step": 1},
                    "multiplier": {"type": "float", "low": 2.0, "high": 4.0, "step": 0.5},
                    "strategy_variant": {"type": "categorical", "choices": ["classic", "multi_timeframe"]}
                },
                "atr_breakout": {
                    "atr_period": {"type": "int", "low": 10, "high": 20, "step": 2},
                    "atr_multiplier": {"type": "float", "low": 1.5, "high": 3.0, "step": 0.25},
                    "lookback_period": {"type": "int", "low": 15, "high": 30, "step": 3}
                },
                "keltner_channels": {
                    "ema_period": {"type": "int", "low": 15, "high": 25, "step": 2},
                    "atr_period": {"type": "int", "low": 8, "high": 15, "step": 1},
                    "multiplier": {"type": "float", "low": 1.5, "high": 3.0, "step": 0.25}
                },
                "donchian_channels": {
                    "upper_period": {"type": "int", "low": 15, "high": 30, "step": 3},
                    "lower_period": {"type": "int", "low": 15, "high": 30, "step": 3},
                    "strategy_variant": {"type": "categorical", "choices": ["breakout", "reversal"]}
                },
                "volume_profile": {
                    "profile_period": {"type": "int", "low": 30, "high": 100, "step": 10},
                    "num_bins": {"type": "int", "low": 15, "high": 30, "step": 5},
                    "poc_threshold": {"type": "float", "low": 0.002, "high": 0.005, "step": 0.001}
                },
                "fibonacci_retracement": {
                    "swing_period": {"type": "int", "low": 15, "high": 30, "step": 3},
                    "fib_proximity": {"type": "float", "low": 0.002, "high": 0.005, "step": 0.001},
                    "rsi_oversold": {"type": "int", "low": 25, "high": 35, "step": 5},
                    "rsi_overbought": {"type": "int", "low": 65, "high": 75, "step": 5}
                },
                "aroon": {
                    "aroon_period": {"type": "int", "low": 20, "high": 30, "step": 2},
                    "oscillator_threshold": {"type": "int", "low": 40, "high": 60, "step": 5},
                    "use_trend_strength": {"type": "categorical", "choices": [True, False]}
                },
                "ensemble": {
                    "confidence_threshold": {"type": "float", "low": 0.4, "high": 0.8, "step": 0.1},
                    "min_regime_duration": {"type": "int", "low": 5, "high": 20, "step": 5},
                    "transition_periods": {"type": "int", "low": 3, "high": 10, "step": 1},
                    "base_position_size": {"type": "float", "low": 0.15, "high": 0.35, "step": 0.05}
                }
            }
        }
    
    def optimize_strategy(self, strategy_name: str, data: pd.DataFrame, 
                         markets: List[str], train_ratio: float = 0.7) -> Dict[str, Any]:
        """Optimize a single strategy"""
        self.logger.info(f"Optimizing strategy: {strategy_name}")
        
        # Split data into train and test
        split_idx = int(len(data) * train_ratio)
        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy()
        
        # Get parameter space for strategy
        param_space = self.config["strategy_params"].get(strategy_name, {})
        if not param_space:
            self.logger.warning(f"No parameter space defined for {strategy_name}")
            return {}
        
        # Create Optuna study
        study = optuna.create_study(
            direction=self.config["optimization"]["direction"],
            sampler=self._get_sampler(),
            pruner=self._get_pruner()
        )
        
        # Optimize
        objective_func = lambda trial: self._objective(
            trial, strategy_name, train_data, markets, param_space
        )
        
        study.optimize(
            objective_func,
            n_trials=self.config["optimization"]["n_trials"],
            timeout=self.config["optimization"]["timeout"],
            n_jobs=1  # Single process per strategy
        )
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        self.logger.info(f"Best parameters for {strategy_name}: {best_params}")
        self.logger.info(f"Best {self.config['optimization']['objective']}: {best_value:.4f}")
        
        # Validate on test set
        test_results = self._validate_params(strategy_name, best_params, test_data, markets)
        
        # Store results
        self.results[strategy_name] = {
            "best_params": best_params,
            "train_performance": best_value,
            "test_performance": test_results,
            "n_trials": len(study.trials),
            "optimization_history": [
                {
                    "params": trial.params,
                    "value": trial.value
                } for trial in study.trials
            ]
        }
        
        return best_params
    
    def optimize_all_strategies(self, data: pd.DataFrame, markets: List[str], 
                              exclude: List[str] = None) -> Dict[str, Dict]:
        """Optimize all strategies in parallel"""
        strategies = list(STRATEGIES.keys())
        
        # Exclude specified strategies
        if exclude:
            strategies = [s for s in strategies if s not in exclude]
        
        self.logger.info(f"Optimizing {len(strategies)} strategies")
        
        # Use multiprocessing for parallel optimization
        max_workers = self.config["optimization"].get("n_jobs", -1)
        if max_workers == -1:
            max_workers = os.cpu_count()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit optimization tasks
            future_to_strategy = {
                executor.submit(
                    self.optimize_strategy, 
                    strategy, 
                    data, 
                    markets,
                    self.config["optimization"]["train_ratio"]
                ): strategy 
                for strategy in strategies
            }
            
            # Collect results
            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    params = future.result()
                    self.best_params[strategy] = params
                    self.logger.info(f"Completed optimization for {strategy}")
                except Exception as e:
                    self.logger.error(f"Error optimizing {strategy}: {e}")
        
        return self.best_params
    
    def _objective(self, trial: optuna.Trial, strategy_name: str, 
                  data: pd.DataFrame, markets: List[str], 
                  param_space: Dict) -> float:
        """Objective function for optimization"""
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
        
        # Run backtest with sampled parameters
        try:
            performance = self._run_backtest(strategy_name, params, data, markets)
            
            # Get objective value
            objective_metric = self.config["optimization"]["objective"]
            if objective_metric == "sharpe_ratio":
                return performance.get("sharpe_ratio", -999)
            elif objective_metric == "sortino_ratio":
                return performance.get("sortino_ratio", -999)
            elif objective_metric == "total_return":
                return performance.get("total_return", -999)
            else:
                return performance.get("sharpe_ratio", -999)
                
        except Exception as e:
            self.logger.warning(f"Trial failed: {e}")
            return -999  # Return very bad value for failed trials
    
    def _run_backtest(self, strategy_name: str, params: Dict, 
                     data: pd.DataFrame, markets: List[str]) -> Dict:
        """Run backtest with given parameters"""
        # Initialize backtest engine
        engine = BacktestEngine({
            "initial_capital": self.config["backtesting"]["initial_capital"],
            "commission_rate": self.config["backtesting"]["commission_rate"],
            "slippage_rate": self.config["backtesting"]["slippage_rate"]
        })
        
        # Create strategy instance
        strategy = get_strategy(strategy_name, params)
        
        # Run backtest
        results = engine.run(data, strategy, markets)
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics(results)
        
        return performance
    
    def _calculate_performance_metrics(self, results: Dict) -> Dict:
        """Calculate performance metrics from backtest results"""
        equity_curve = results.get("equity_curve", [])
        returns = pd.Series(equity_curve).pct_change().dropna()
        
        if len(returns) < 2:
            return {"sharpe_ratio": -999, "sortino_ratio": -999, "total_return": -999}
        
        # Calculate metrics
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] if equity_curve[0] > 0 else -999
        
        # Annualized Sharpe ratio (assuming minute data)
        mean_return = returns.mean()
        std_return = returns.std()
        sharpe_ratio = mean_return / std_return * np.sqrt(365 * 24 * 60) if std_return > 0 else -999
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else std_return
        sortino_ratio = mean_return / downside_std * np.sqrt(365 * 24 * 60) if downside_std > 0 else -999
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": results.get("max_drawdown", 0),
            "win_rate": results.get("win_rate", 0),
            "profit_factor": results.get("profit_factor", 0)
        }
    
    def _validate_params(self, strategy_name: str, params: Dict, 
                        test_data: pd.DataFrame, markets: List[str]) -> Dict:
        """Validate optimized parameters on test set"""
        return self._run_backtest(strategy_name, params, test_data, markets)
    
    def _get_sampler(self) -> optuna.samplers.BaseSampler:
        """Get Optuna sampler based on configuration"""
        sampler_name = self.config["optimization"]["sampler"]
        
        if sampler_name == "TPE":
            return optuna.samplers.TPESampler()
        elif sampler_name == "CMA-ES":
            return optuna.samplers.CmaEsSampler()
        elif sampler_name == "Random":
            return optuna.samplers.RandomSampler()
        else:
            return optuna.samplers.TPESampler()
    
    def _get_pruner(self) -> optuna.pruners.BasePruner:
        """Get Optuna pruner based on configuration"""
        pruner_name = self.config["optimization"]["pruner"]
        
        if pruner_name == "MedianPruner":
            return optuna.pruners.MedianPruner()
        elif pruner_name == "SuccessiveHalving":
            return optuna.pruners.SuccessiveHalvingPruner()
        else:
            return optuna.pruners.MedianPruner()
    
    def save_results(self, output_dir: str = "results/optimization"):
        """Save optimization results to JSON files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save best parameters for each strategy
        for strategy_name, params in self.best_params.items():
            output_file = os.path.join(output_dir, f"{strategy_name}_optimized.json")
            
            with open(output_file, 'w') as f:
                json.dump({
                    "strategy": strategy_name,
                    "optimized_params": params,
                    "optimization_results": self.results.get(strategy_name, {}),
                    "timestamp": datetime.now().isoformat()
                }, f, indent=4)
            
            self.logger.info(f"Saved optimized parameters to {output_file}")
        
        # Save summary of all optimizations
        summary_file = os.path.join(output_dir, "optimization_summary.json")
        with open(summary_file, 'w') as f:
            json.dump({
                "optimization_config": self.config["optimization"],
                "strategies_optimized": list(self.best_params.keys()),
                "results": self.results,
                "timestamp": datetime.now().isoformat()
            }, f, indent=4)
        
        self.logger.info(f"Saved optimization summary to {summary_file}")