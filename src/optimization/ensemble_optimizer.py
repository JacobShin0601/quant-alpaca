"""
Two-Stage Ensemble Optimization System
Stage 1: Individual strategy optimization
Stage 2: Ensemble meta-optimization with optimized strategies
"""

import json
import os
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from .strategy_optimizer import StrategyOptimizer
    from ..strategies import STRATEGIES
    from ..backtesting.engine import BacktestEngine
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from optimization.strategy_optimizer import StrategyOptimizer
    from strategies import STRATEGIES
    from backtesting.engine import BacktestEngine


class EnsembleOptimizer:
    """Two-stage optimization for ensemble strategies"""
    
    def __init__(self, config_path: str = "config/config_ensemble_optimize.json"):
        """Initialize ensemble optimizer"""
        self.config = self._load_config(config_path)
        self.optimized_strategies = {}
        self.ensemble_params = {}
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default configuration for ensemble optimization"""
        return {
            "stage1": {
                "strategies_to_optimize": ["vwap", "bollinger_bands", "macd", "supertrend", 
                                         "mean_reversion", "atr_breakout", "ichimoku"],
                "optimization_config": "config/config_optimize.json",
                "parallel_workers": 4
            },
            "stage2": {
                "ensemble_structure": {
                    "trending_up": {
                        "primary_pool": ["supertrend", "ichimoku", "vwap"],
                        "secondary_pool": ["macd", "atr_breakout"],
                        "max_strategies": 3
                    },
                    "trending_down": {
                        "primary_pool": ["mean_reversion", "bollinger_bands"],
                        "secondary_pool": ["supertrend", "vwap"],
                        "max_strategies": 2
                    },
                    "sideways": {
                        "primary_pool": ["mean_reversion", "bollinger_bands"],
                        "secondary_pool": ["vwap", "macd"],
                        "max_strategies": 3
                    },
                    "volatile": {
                        "primary_pool": ["atr_breakout", "bollinger_bands"],
                        "secondary_pool": ["mean_reversion"],
                        "max_strategies": 2
                    }
                },
                "meta_parameters": {
                    "weight_optimization": True,
                    "adaptive_weights": True,
                    "performance_window": 100,
                    "rebalance_frequency": 50,
                    "min_strategy_weight": 0.1,
                    "max_strategy_weight": 0.5
                }
            },
            "performance_metrics": {
                "optimization_objective": "risk_adjusted_return",
                "risk_penalty": 0.5,
                "consistency_bonus": 0.2,
                "regime_adaptability_bonus": 0.3
            }
        }
    
    def optimize_individual_strategies(self, data: pd.DataFrame, markets: List[str]) -> Dict[str, Dict]:
        """Stage 1: Optimize individual strategies"""
        self.logger.info("Starting Stage 1: Individual Strategy Optimization")
        
        strategies = self.config["stage1"]["strategies_to_optimize"]
        optimizer = StrategyOptimizer(self.config["stage1"]["optimization_config"])
        
        # Parallel optimization of strategies
        with ProcessPoolExecutor(max_workers=self.config["stage1"]["parallel_workers"]) as executor:
            future_to_strategy = {
                executor.submit(
                    optimizer.optimize_strategy,
                    strategy,
                    data,
                    markets,
                    self.config.get("train_ratio", 0.7)
                ): strategy
                for strategy in strategies
            }
            
            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    params = future.result()
                    self.optimized_strategies[strategy] = params
                    self.logger.info(f"Optimized {strategy}: {params}")
                except Exception as e:
                    self.logger.error(f"Failed to optimize {strategy}: {e}")
        
        # Save optimized parameters
        self._save_optimized_strategies()
        
        return self.optimized_strategies
    
    def optimize_ensemble_meta_parameters(self, data: pd.DataFrame, markets: List[str]) -> Dict:
        """Stage 2: Optimize ensemble meta-parameters"""
        self.logger.info("Starting Stage 2: Ensemble Meta-Optimization")
        
        # Load optimized strategies if not already loaded
        if not self.optimized_strategies:
            self._load_optimized_strategies()
        
        # Split data for meta-optimization
        train_ratio = self.config.get("train_ratio", 0.7)
        split_idx = int(len(data) * train_ratio)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        # Optimize ensemble parameters
        best_ensemble_config = self._optimize_ensemble_structure(train_data, markets)
        
        # Validate on test data
        test_performance = self._validate_ensemble(best_ensemble_config, test_data, markets)
        
        self.ensemble_params = {
            "optimized_structure": best_ensemble_config,
            "train_performance": best_ensemble_config.get("performance", {}),
            "test_performance": test_performance,
            "strategy_parameters": self.optimized_strategies,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save ensemble configuration
        self._save_ensemble_config()
        
        return self.ensemble_params
    
    def _optimize_ensemble_structure(self, data: pd.DataFrame, markets: List[str]) -> Dict:
        """Optimize the ensemble structure and weights"""
        best_config = None
        best_score = -float('inf')
        
        # Grid search over ensemble configurations
        for weight_config in self._generate_weight_configurations():
            for strategy_selection in self._generate_strategy_selections():
                config = {
                    "weights": weight_config,
                    "strategy_selection": strategy_selection,
                    "adaptive_weights": self.config["stage2"]["meta_parameters"]["adaptive_weights"]
                }
                
                # Run backtest with this configuration
                performance = self._evaluate_ensemble_config(config, data, markets)
                score = self._calculate_ensemble_score(performance)
                
                if score > best_score:
                    best_score = score
                    best_config = config
                    best_config["performance"] = performance
                    best_config["score"] = score
                    
                    self.logger.info(f"New best ensemble config found: score={score:.4f}")
        
        return best_config
    
    def _generate_weight_configurations(self) -> List[Dict]:
        """Generate different weight configurations to test"""
        configs = []
        
        # Generate different weight distributions
        for primary_weight in [0.5, 0.6, 0.7, 0.8]:
            secondary_weight = 1.0 - primary_weight
            configs.append({
                "primary": primary_weight,
                "secondary": secondary_weight
            })
        
        return configs
    
    def _generate_strategy_selections(self) -> List[Dict]:
        """Generate different strategy selection configurations"""
        selections = []
        ensemble_structure = self.config["stage2"]["ensemble_structure"]
        
        # For each regime, try different combinations
        for regime, config in ensemble_structure.items():
            primary_pool = config["primary_pool"]
            secondary_pool = config["secondary_pool"]
            max_strategies = config["max_strategies"]
            
            # Try different numbers of strategies
            for n_strategies in range(2, max_strategies + 1):
                # Allocate between primary and secondary
                n_primary = max(1, n_strategies * 2 // 3)
                n_secondary = n_strategies - n_primary
                
                selections.append({
                    regime: {
                        "primary": primary_pool[:n_primary],
                        "secondary": secondary_pool[:n_secondary]
                    }
                })
        
        return selections
    
    def _evaluate_ensemble_config(self, config: Dict, data: pd.DataFrame, 
                                 markets: List[str]) -> Dict:
        """Evaluate an ensemble configuration"""
        # Create ensemble strategy with optimized sub-strategies
        ensemble_params = {
            "ensemble_config": config,
            "strategy_parameters": self.optimized_strategies,
            "optimization_mode": True
        }
        
        # Run backtest
        backtest_config = {
            "strategy": {
                "name": "optimized_ensemble",
                "parameters": ensemble_params
            },
            "backtesting": {
                "initial_balance": 10000000,
                "fees": {"limit_order": {"krw_market": 0.0005}},
                "slippage": {"limit_order": 0.001}
            }
        }
        
        engine = BacktestEngine(backtest_config)
        results = engine.run_backtest({market: data for market in markets})
        
        return results
    
    def _calculate_ensemble_score(self, performance: Dict) -> float:
        """Calculate ensemble score based on multiple metrics"""
        metrics = self.config["performance_metrics"]
        
        # Base return
        total_return = performance.get("total_return", 0)
        
        # Risk adjustment
        sharpe = performance.get("sharpe_ratio", 0)
        sortino = performance.get("sortino_ratio", 0)
        max_dd = abs(performance.get("max_drawdown", 0))
        
        # Consistency
        win_rate = performance.get("win_rate", 0)
        
        # Calculate composite score
        score = total_return
        score += sharpe * metrics["risk_penalty"]
        score += sortino * metrics["risk_penalty"] * 0.5
        score -= max_dd * metrics["risk_penalty"]
        score += win_rate * metrics["consistency_bonus"]
        
        # Regime adaptability bonus
        if "regime_performance" in performance:
            regime_scores = []
            for regime_perf in performance["regime_performance"].values():
                if regime_perf and "sharpe_ratio" in regime_perf:
                    regime_scores.append(regime_perf["sharpe_ratio"])
            
            if regime_scores:
                # Bonus for consistent performance across regimes
                regime_consistency = 1 - np.std(regime_scores) / (np.mean(regime_scores) + 1e-6)
                score += regime_consistency * metrics["regime_adaptability_bonus"]
        
        return score
    
    def _validate_ensemble(self, config: Dict, data: pd.DataFrame, 
                          markets: List[str]) -> Dict:
        """Validate ensemble on test data"""
        return self._evaluate_ensemble_config(config, data, markets)
    
    def _save_optimized_strategies(self):
        """Save optimized strategy parameters"""
        output_dir = "results/optimization/strategies"
        os.makedirs(output_dir, exist_ok=True)
        
        for strategy, params in self.optimized_strategies.items():
            output_file = os.path.join(output_dir, f"{strategy}_optimized.json")
            with open(output_file, 'w') as f:
                json.dump({
                    "strategy": strategy,
                    "optimized_params": params,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=4)
    
    def _load_optimized_strategies(self):
        """Load previously optimized strategy parameters"""
        input_dir = "results/optimization/strategies"
        
        if os.path.exists(input_dir):
            for file in os.listdir(input_dir):
                if file.endswith("_optimized.json"):
                    strategy = file.replace("_optimized.json", "")
                    with open(os.path.join(input_dir, file), 'r') as f:
                        data = json.load(f)
                        self.optimized_strategies[strategy] = data["optimized_params"]
    
    def _save_ensemble_config(self):
        """Save optimized ensemble configuration"""
        output_file = "results/optimization/ensemble_optimized.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.ensemble_params, f, indent=4)
        
        self.logger.info(f"Saved ensemble configuration to {output_file}")
    
    def create_production_config(self) -> Dict:
        """Create production-ready configuration with optimized parameters"""
        return {
            "strategy": {
                "name": "optimized_ensemble",
                "parameters": self.ensemble_params["optimized_structure"]
            },
            "optimized_strategies": self.optimized_strategies,
            "meta_parameters": self.config["stage2"]["meta_parameters"],
            "created_at": datetime.now().isoformat()
        }