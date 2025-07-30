#!/usr/bin/env python3
"""
Run two-stage ensemble optimization
Stage 1: Optimize individual strategies
Stage 2: Optimize ensemble meta-parameters with optimized strategies
"""

import argparse
import logging
import json
import os
import sys
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from optimization.ensemble_optimizer import EnsembleOptimizer
from data.collector import DataCollector


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/ensemble_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_data(markets, start_date, end_date, use_cached=True):
    """Load market data for optimization"""
    logger = logging.getLogger(__name__)
    
    if use_cached and os.path.exists("data/optimization_data.pkl"):
        logger.info("Loading cached data...")
        return pd.read_pickle("data/optimization_data.pkl")
    
    logger.info(f"Collecting data for markets: {markets}")
    collector = DataCollector()
    
    all_data = []
    for market in markets:
        df = collector.collect_data(market, start_date, end_date)
        df['market'] = market
        all_data.append(df)
    
    combined_data = pd.concat(all_data, axis=0)
    
    # Save for future use
    combined_data.to_pickle("data/optimization_data.pkl")
    
    return combined_data


def main():
    parser = argparse.ArgumentParser(description='Run ensemble optimization')
    parser.add_argument('--stage', choices=['1', '2', 'both'], default='both',
                       help='Which optimization stage to run')
    parser.add_argument('--config', default='config/config_ensemble_optimize.json',
                       help='Configuration file path')
    parser.add_argument('--markets', nargs='+', 
                       default=['KRW-BTC', 'KRW-ETH', 'KRW-XRP'],
                       help='Markets to optimize')
    parser.add_argument('--start-date', default='2023-01-01',
                       help='Start date for data collection')
    parser.add_argument('--end-date', default='2024-01-01',
                       help='End date for data collection')
    parser.add_argument('--use-cached-data', action='store_true',
                       help='Use cached data if available')
    parser.add_argument('--skip-stage1', action='store_true',
                       help='Skip stage 1 if strategies already optimized')
    
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    logger = setup_logging()
    
    logger.info("Starting Ensemble Optimization")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Markets: {args.markets}")
    logger.info(f"Stage: {args.stage}")
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Load data
    data = load_data(args.markets, args.start_date, args.end_date, args.use_cached_data)
    logger.info(f"Loaded {len(data)} data points")
    
    # Initialize optimizer
    optimizer = EnsembleOptimizer(args.config)
    
    # Run optimization stages
    if args.stage in ['1', 'both'] and not args.skip_stage1:
        logger.info("=" * 50)
        logger.info("STAGE 1: Individual Strategy Optimization")
        logger.info("=" * 50)
        
        optimized_strategies = optimizer.optimize_individual_strategies(data, args.markets)
        
        logger.info(f"Optimized {len(optimized_strategies)} strategies")
        for strategy, params in optimized_strategies.items():
            logger.info(f"{strategy}: {params}")
    
    if args.stage in ['2', 'both']:
        logger.info("=" * 50)
        logger.info("STAGE 2: Ensemble Meta-Optimization")
        logger.info("=" * 50)
        
        ensemble_params = optimizer.optimize_ensemble_meta_parameters(data, args.markets)
        
        logger.info("Ensemble optimization completed")
        logger.info(f"Best configuration score: {ensemble_params['optimized_structure']['score']:.4f}")
        
        # Print summary
        logger.info("\nOptimized Ensemble Structure:")
        logger.info(json.dumps(ensemble_params['optimized_structure'], indent=2))
        
        # Create production configuration
        production_config = optimizer.create_production_config()
        
        # Save production config
        production_config_file = "config/config_production_ensemble.json"
        with open(production_config_file, 'w') as f:
            json.dump(production_config, f, indent=4)
        
        logger.info(f"\nProduction configuration saved to: {production_config_file}")
        
        # Print performance comparison
        if 'train_performance' in ensemble_params and 'test_performance' in ensemble_params:
            logger.info("\nPerformance Summary:")
            logger.info(f"Train Sharpe Ratio: {ensemble_params['train_performance'].get('sharpe_ratio', 0):.4f}")
            logger.info(f"Test Sharpe Ratio: {ensemble_params['test_performance'].get('sharpe_ratio', 0):.4f}")
            logger.info(f"Train Total Return: {ensemble_params['train_performance'].get('total_return', 0):.2%}")
            logger.info(f"Test Total Return: {ensemble_params['test_performance'].get('total_return', 0):.2%}")
    
    logger.info("\nOptimization completed successfully!")


if __name__ == "__main__":
    main()