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
    from ..strategies import get_strategy, STRATEGIES
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from backtesting.engine import BacktestEngine
    from strategies import get_strategy, STRATEGIES


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
                    "base_position_size": {"type": "float", "low": 0.15, "high": 0.35, "step": 0.05},
                    "strategy_rotation": {"type": "categorical", "choices": [True, False]},
                    "smooth_transition": {"type": "categorical", "choices": [True, False]}
                }
            }
        }
    
    def optimize_strategy(self, strategy_name: str, data: pd.DataFrame, 
                         markets: List[str], train_ratio: float = 0.7) -> Dict[str, Any]:
        """Optimize a single strategy for each market separately"""
        self.logger.info(f"Optimizing strategy: {strategy_name}")
        
        # Optimize for each market separately
        market_results = {}
        
        # Use parallel optimization for multiple markets
        if len(markets) > 1:
            with ProcessPoolExecutor(max_workers=min(len(markets), os.cpu_count())) as executor:
                future_to_market = {}
                
                for market in markets:
                    future = executor.submit(
                        self._optimize_single_market,
                        strategy_name, data, market, train_ratio
                    )
                    future_to_market[future] = market
                
                for future in as_completed(future_to_market):
                    market = future_to_market[future]
                    try:
                        result = future.result()
                        market_results[market] = result
                        self.logger.info(f"Completed optimization for {strategy_name} on {market}")
                    except Exception as e:
                        self.logger.error(f"Error optimizing {strategy_name} on {market}: {e}")
        else:
            # Single market, no need for parallel processing
            for market in markets:
                result = self._optimize_single_market(strategy_name, data, market, train_ratio)
                market_results[market] = result
        
        # Store combined results
        self.results[strategy_name] = market_results
        self.best_params[strategy_name] = {
            market: results["best_params"] 
            for market, results in market_results.items()
        }
        
        return self.best_params[strategy_name]
    
    def _optimize_single_market(self, strategy_name: str, data: pd.DataFrame, 
                               market: str, train_ratio: float) -> Dict[str, Any]:
        """Optimize a strategy for a single market"""
        self.logger.info(f"Optimizing {strategy_name} for market: {market}")
        
        # Filter data for this market
        if 'market' in data.columns:
            market_data = data[data['market'] == market].copy()
        else:
            market_data = data.copy()
        
        # Split data into train and test
        split_idx = int(len(market_data) * train_ratio)
        train_data = market_data.iloc[:split_idx].copy()
        test_data = market_data.iloc[split_idx:].copy()
        
        # Get parameter space for strategy
        param_space = self.config["strategy_params"].get(strategy_name, {})
        if not param_space:
            self.logger.warning(f"No parameter space defined for {strategy_name}")
            return {}
            
        # Create Optuna study for this market
        study = optuna.create_study(
            direction=self.config["optimization"]["direction"],
            sampler=self._get_sampler(),
            pruner=self._get_pruner()
        )
        
        # Optimize for this specific market
        objective_func = lambda trial: self._objective(
            trial, strategy_name, train_data, [market], param_space
        )
        
        # Set Optuna logging to WARNING to reduce noise
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study.optimize(
            objective_func,
            n_trials=self.config["optimization"]["n_trials"],
            timeout=self.config["optimization"]["timeout"],
            n_jobs=1
        )
            
        # Get best parameters for this market
        best_params = study.best_params
        best_value = study.best_value
        
        # Count trials with zero trades (score of -100.0)
        zero_trade_trials = sum(1 for trial in study.trials if trial.value == -100.0)
        successful_trials = len(study.trials) - zero_trade_trials
        
        self.logger.info(f"Best parameters for {strategy_name} on {market}: {best_params}")
        self.logger.info(f"Best {self.config['optimization']['objective']}: {best_value:.4f}")
        self.logger.info(f"Completed {len(study.trials)} trials ({successful_trials} with trades, {zero_trade_trials} with zero trades)")
        
        if zero_trade_trials > len(study.trials) * 0.8:
            self.logger.warning(f"⚠️  {zero_trade_trials}/{len(study.trials)} trials had zero trades - consider relaxing strategy parameters!")
        
        # Validate on test set
        test_results = self._validate_params(strategy_name, best_params, test_data, [market])
        
        # Store market-specific results - only store essential data during optimization
        # Clean test_results to remove non-serializable data
        clean_test_results = {}
        if isinstance(test_results, dict):
            # Extract only the essential metrics, excluding history data with timestamps
            clean_test_results = {
                "total_return": test_results.get("total_return", -999),
                "sharpe_ratio": test_results.get("sharpe_ratio", -999),
                "sortino_ratio": test_results.get("sortino_ratio", -999),
                "max_drawdown": test_results.get("max_drawdown", -999),
                "total_trades": test_results.get("total_trades", 0),
                "calmar_ratio": test_results.get("calmar_ratio", 0),
                "volatility": test_results.get("volatility", 0),
                "buy_hold_benchmark": test_results.get("buy_hold_benchmark", {})
            }
        
        optimization_results = {
            "best_params": best_params,
            "train_performance": best_value,
            "test_performance": clean_test_results,
            "n_trials": len(study.trials),
            # Only store top 10 trials to reduce memory usage
            "optimization_history": [
                {
                    "params": trial.params,
                    "value": trial.value
                } for trial in sorted(study.trials, key=lambda t: t.value, reverse=True)[:10]
            ]
        }
        
        # Store for access from backtest_market.py
        self.best_train_score = best_value
        self.best_test_performance = clean_test_results
        self.total_trials = len(study.trials)
        self.optimization_history = optimization_results["optimization_history"]
        
        return optimization_results
    
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
        # Log trial number
        self.logger.info(f"\n🔄 Running optimization trial #{trial.number + 1} for {strategy_name}")
        
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
            # Debug logging - more detailed for ensemble
            if strategy_name == "ensemble":
                self.logger.info(f"Starting ensemble backtest trial with params: {params}")
            else:
                self.logger.debug(f"Running backtest for {strategy_name} with params: {params}")
            
            results = self._run_backtest(strategy_name, params, data, markets)
            
            if strategy_name == "ensemble":
                self.logger.info(f"Ensemble backtest completed, calculating metrics...")
                
            performance = self._calculate_performance_metrics(results)
            
            # Check for zero trades and log detailed information
            total_trades = performance.get('total_trades', 0)
            if total_trades == 0:
                self.logger.warning(f"🚨 Trial #{trial.number + 1} for {strategy_name} generated NO TRADES!")
                self.logger.warning(f"   Parameters: {params}")
                
                # Log additional debug info for mt_bollinger strategy
                if strategy_name == "mt_bollinger":
                    self.logger.warning(f"   Thresholds: lower={params.get('lower_threshold', 'N/A')}, upper={params.get('upper_threshold', 'N/A')}")
                    self.logger.warning(f"   Signal strength requirement: {params.get('min_signal_strength', 'N/A')}")
                    self.logger.warning(f"   Volume confirmation: {params.get('min_volume_confirmation', 'N/A')}")
                    self.logger.warning(f"   Alignment required: {params.get('require_timeframe_alignment', 'N/A')}")
                
                # Return a strong penalty for no trades
                return -100.0
            
            # Get objective value
            objective_metric = self.config["optimization"]["objective"]
            
            # Use regime-adjusted metrics if regime analysis is enabled
            if "regime_adjusted_sharpe" in performance:
                if objective_metric == "sharpe_ratio":
                    # Prefer regime-adjusted sharpe for more robust optimization
                    return performance.get("regime_adjusted_sharpe", -999)
            
            # Standard metrics
            if objective_metric == "sharpe_ratio":
                return performance.get("sharpe_ratio", -999)
            elif objective_metric == "sortino_ratio":
                return performance.get("sortino_ratio", -999)
            elif objective_metric == "total_return":
                return performance.get("total_return", -999)
            elif objective_metric == "regime_weighted_sharpe":
                return performance.get("regime_weighted_sharpe", performance.get("sharpe_ratio", -999))
            else:
                return performance.get("sharpe_ratio", -999)
                
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return -999  # Return very bad value for failed trials
    
    def _run_backtest(self, strategy_name: str, params: Dict, 
                     data: pd.DataFrame, markets: List[str]) -> Dict:
        """Run backtest with given parameters"""
        # Get commission rate from config
        commission_rate = self.config["backtesting"]["commission_rate"]
        slippage_rate = self.config["backtesting"]["slippage_rate"]
        
        # Create proper config structure for BacktestEngine
        backtest_config = {
            "backtesting": {
                "initial_balance": self.config["backtesting"]["initial_capital"],
                "fees": {
                    "limit_order": {
                        "krw_market": commission_rate,
                        "btc_market": commission_rate,
                        "usdt_market": commission_rate
                    },
                    "market_order": {
                        "krw_market": commission_rate,
                        "btc_market": commission_rate,
                        "usdt_market": commission_rate
                    }
                },
                "slippage": {
                    "limit_order": slippage_rate,
                    "market_order": slippage_rate
                },
                "max_positions": self.config["backtesting"].get("max_positions", 3),
                "order_type": self.config["backtesting"].get("order_type", "limit")
            },
            "position_sizing": self.config["backtesting"].get("position_sizing", {
                "max_position_pct": 0.2,
                "use_dynamic_sizing": False,
                "use_garch_sizing": False,
                "garch_config": {}
            }),
            "risk_management": self._get_risk_management_config(),
            "var_risk_management": {
                "enabled": False  # Disable VaR during optimization for speed
            },
            "regime_config": self.config["backtesting"].get("regime_analysis", {}).get("regime_config", {
                "lookback_period": 20,
                "volatility_threshold": 0.02,
                "trend_threshold": 0.0001,
                "volume_ma_period": 20,
                "atr_period": 14,
                "adx_period": 14,
                "adx_threshold": 25,
                "ma_periods": [20, 50],
                "bb_period": 20,
                "bb_std": 2.0,
                "volume_period": 20,  
                "choppiness_period": 14
            }),
            "analyze_regime_performance": False,  # Disable regime analysis during optimization
            "execution": {
                "log_level": "CRITICAL",  # Only log critical errors during optimization
                "max_log_trades": 0,  # Don't log trades during optimization
                "save_detailed_results": False,
                "warmup_period_minutes": 0,  # No warmup during optimization
                "max_stop_limit_warnings": 0  # No stop limit warnings during optimization
            },
            "strategy": {
                "name": strategy_name,
                "parameters": {**params, "optimization_mode": True} if strategy_name == "ensemble" else params
            }
        }
        
        # Initialize backtest engine with proper config
        engine = BacktestEngine(backtest_config)
        
        # Convert data format for backtest engine
        market_data = {}
        for market in markets:
            # Assuming data has multi-index or market column
            if 'market' in data.columns:
                market_df = data[data['market'] == market].copy()
            else:
                # If data is for single market
                market_df = data.copy()
            
            # Ensure required columns exist
            required_cols = ['trade_price', 'high_price', 'low_price', 'opening_price', 
                           'candle_acc_trade_volume', 'candle_acc_trade_price']
            if all(col in market_df.columns for col in required_cols):
                market_data[market] = market_df
        
        if not market_data:
            self.logger.error("No valid market data for backtest")
            return {"sharpe_ratio": -999, "sortino_ratio": -999, "total_return": -999}
        
        # Run backtest
        try:
            results = engine.run_backtest(market_data)
            return results
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            return {"sharpe_ratio": -999, "sortino_ratio": -999, "total_return": -999}
    
    def _calculate_performance_metrics(self, results: Dict) -> Dict:
        """Calculate performance metrics from backtest results"""
        # BacktestEngine already calculates these metrics
        if isinstance(results, dict):
            metrics = {
                "total_return": results.get("total_return", -999),
                "sharpe_ratio": results.get("sharpe_ratio", -999),
                "sortino_ratio": results.get("sortino_ratio", -999),
                "max_drawdown": results.get("max_drawdown", -999),
                "total_trades": results.get("total_trades", 0)
            }
            
            # Add regime-based performance if available
            if "regime_analysis" in results and results["regime_analysis"]:
                regime_metrics = results["regime_analysis"].get("regime_metrics", {})
                
                # Calculate weighted sharpe ratio across regimes
                weighted_sharpe = 0
                total_weight = 0
                
                for regime_name, regime_data in regime_metrics.items():
                    if regime_data and hasattr(regime_data, 'sharpe_ratio'):
                        weight = regime_data.regime_percentage / 100.0
                        weighted_sharpe += regime_data.sharpe_ratio * weight
                        total_weight += weight
                
                if total_weight > 0:
                    metrics["regime_weighted_sharpe"] = weighted_sharpe / total_weight
                    
                    # Penalty for high variance across regimes
                    sharpe_values = [regime_data.sharpe_ratio for regime_data in regime_metrics.values() 
                                   if regime_data and hasattr(regime_data, 'sharpe_ratio')]
                    if len(sharpe_values) > 1:
                        sharpe_std = np.std(sharpe_values)
                        # Adjust final sharpe by consistency across regimes
                        metrics["regime_adjusted_sharpe"] = metrics["sharpe_ratio"] - sharpe_std * 0.5
                    else:
                        metrics["regime_adjusted_sharpe"] = metrics["sharpe_ratio"]
            
            return metrics
        else:
            return {"sharpe_ratio": -999, "sortino_ratio": -999, "total_return": -999}
    
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
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        import pandas as pd
        from datetime import datetime
        
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, pd.Series):
            return self._convert_numpy_types(obj.to_dict())
        elif isinstance(obj, pd.DataFrame):
            return self._convert_numpy_types(obj.to_dict())
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def save_results(self, output_dir: str = "results/optimization"):
        """Save optimization results to JSON files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save best parameters for each strategy and market
        for strategy_name, market_params in self.best_params.items():
            # Save strategy-level summary
            strategy_dir = os.path.join(output_dir, strategy_name)
            os.makedirs(strategy_dir, exist_ok=True)
            
            # Save market-specific parameters
            for market, params in market_params.items():
                market_file = os.path.join(strategy_dir, f"{market}_optimized.json")
                
                # Get market-specific results
                market_results = self.results.get(strategy_name, {}).get(market, {})
                
                data_to_save = {
                    "strategy": strategy_name,
                    "market": market,
                    "optimized_params": self._convert_numpy_types(params),
                    "train_performance": self._convert_numpy_types(market_results.get("train_performance", {})),
                    "test_performance": self._convert_numpy_types(market_results.get("test_performance", {})),
                    "n_trials": market_results.get("n_trials", 0),
                    "timestamp": datetime.now().isoformat()
                }
                
                with open(market_file, 'w') as f:
                    json.dump(data_to_save, f, indent=4)
                
                self.logger.info(f"Saved {strategy_name} parameters for {market} to {market_file}")
            
            # Save combined strategy file
            combined_file = os.path.join(strategy_dir, "all_markets_optimized.json")
            
            combined_data = {
                "strategy": strategy_name,
                "markets": self._convert_numpy_types(market_params),
                "results": self._convert_numpy_types(self.results.get(strategy_name, {})),
                "timestamp": datetime.now().isoformat()
            }
            
            with open(combined_file, 'w') as f:
                json.dump(combined_data, f, indent=4)
        
        # Save summary of all optimizations
        summary_file = os.path.join(output_dir, "optimization_summary.json")
        
        # Convert numpy types in results
        summary_data = {
            "optimization_config": self._convert_numpy_types(self.config["optimization"]),
            "strategies_optimized": list(self.best_params.keys()),
            "market_specific_results": self._convert_numpy_types(self.best_params),
            "detailed_results": self._convert_numpy_types(self.results),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=4)
        
        self.logger.info(f"Saved optimization summary to {summary_file}")
        
        # Display summary table after saving
        self._display_optimization_summary()
    
    def _display_optimization_summary(self):
        """Display a summary table of optimization results"""
        if not self.results:
            return
            
        print("\n" + "="*80)
        print("📊 OPTIMIZATION RESULTS SUMMARY")
        print("="*80)
        
        # Create summary table
        summary_data = []
        
        for strategy_name, market_results in self.results.items():
            for market, results in market_results.items():
                if isinstance(results, dict):
                    train_perf = results.get("train_performance", -999)
                    test_perf = results.get("test_performance", {})
                    
                    # Extract performance metrics
                    if isinstance(test_perf, dict):
                        test_sharpe = test_perf.get("sharpe_ratio", -999)
                        test_return = test_perf.get("total_return", -999)
                        test_drawdown = test_perf.get("max_drawdown", -999)
                        test_trades = test_perf.get("total_trades", 0)
                        
                        # Extract Buy & Hold benchmark
                        bh_data = test_perf.get("buy_hold_benchmark", {})
                        bh_return = bh_data.get("total_return", 0)
                        excess_return = test_return - bh_return if test_return != -999 and bh_return != 0 else 0
                    else:
                        test_sharpe = test_perf if isinstance(test_perf, (int, float)) else -999
                        test_return = -999
                        test_drawdown = -999
                        test_trades = 0
                        excess_return = 0
                    
                    summary_data.append({
                        "Strategy": strategy_name,
                        "Market": market,
                        "Trials": results.get("n_trials", 0),
                        "Train Sharpe": f"{train_perf:.4f}" if train_perf != -999 else "N/A",
                        "Test Sharpe": f"{test_sharpe:.4f}" if test_sharpe != -999 else "N/A",
                        "Test Return": f"{test_return:.2%}" if test_return != -999 else "N/A",
                        "B&H Return": f"{bh_return:.2%}" if bh_return != 0 else "N/A",
                        "Excess Return": f"{excess_return:.2%}" if excess_return != 0 else "N/A",
                        "Max Drawdown": f"{test_drawdown:.2%}" if test_drawdown != -999 else "N/A",
                        "Total Trades": test_trades
                    })
        
        if summary_data:
            # Print table header
            headers = list(summary_data[0].keys())
            col_widths = {header: max(len(str(header)), 
                                    max(len(str(row[header])) for row in summary_data)) 
                         for header in headers}
            
            # Print header row
            header_line = " | ".join(f"{header:{col_widths[header]}}" for header in headers)
            print(header_line)
            print("-" * len(header_line))
            
            # Print data rows
            for row in summary_data:
                print(" | ".join(f"{str(row[header]):{col_widths[header]}}" for header in headers))
            
            print("="*80)
            
            # Print best parameters for each strategy/market
            print("\n🎯 OPTIMIZED PARAMETERS")
            print("="*80)
            
            for strategy_name, market_params in self.best_params.items():
                print(f"\n{strategy_name.upper()}:")
                for market, params in market_params.items():
                    print(f"  {market}:")
                    for param_name, param_value in params.items():
                        print(f"    - {param_name}: {param_value}")
            
            print("\n" + "="*80)
    
    def _get_risk_management_config(self) -> Dict:
        """Get risk management configuration with proper regime parameters"""
        # During optimization, use simplified risk management
        return {
            "enabled": False,  # Disable risk management during optimization
            "regime_parameters": {
                "trending_up": {"stop_loss": {}, "take_profit": {}},
                "trending_down": {"stop_loss": {}, "take_profit": {}},
                "sideways": {"stop_loss": {}, "take_profit": {}},
                "volatile": {"stop_loss": {}, "take_profit": {}},
                "unknown": {"stop_loss": {}, "take_profit": {}}
            }
        }