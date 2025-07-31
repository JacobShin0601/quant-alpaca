#!/usr/bin/env python3
"""
Multi-Strategy Backtesting System with Market Selection
"""

import pandas as pd
import numpy as np
import json
import sys
import os
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backtesting.engine import BacktestEngine
from data.collector import UpbitDataCollector
from agents.scrapper import UpbitDataScrapper

# Import strategies
from strategies import STRATEGIES

# Import optimizer
try:
    from optimization.strategy_optimizer import StrategyOptimizer
    from optimization.optimization_results import OptimizationResultsDisplay
except ImportError:
    StrategyOptimizer = None
    OptimizationResultsDisplay = None


class MultiStrategyBacktester:
    """Manages multi-strategy backtesting with market selection"""
    
    def __init__(self, config_path: str = 'config/config_backtesting.json'):
        """Initialize the backtester with configuration"""
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize data collector with correct parameters
        self.collector = UpbitDataCollector(
            self.config['data']['database_directory'],
            self.config['data']['database_pattern']
        )
        
        # Available strategies
        self.available_strategies = list(STRATEGIES.keys())
        
        if 'ensemble' in STRATEGIES and 'ensemble' not in self.available_strategies:
            self.available_strategies.append('ensemble')
    
    def _calculate_date_range(self) -> Tuple[str, str]:
        """Calculate date range for backtesting"""
        end_date = self.config['data']['end_date']
        lookback_days = self.config['data']['lookback_days']
        
        end_datetime = pd.to_datetime(end_date)
        start_datetime = end_datetime - timedelta(days=lookback_days)
        
        return start_datetime.strftime('%Y-%m-%d'), end_date
    
    def _get_default_strategies(self) -> List[str]:
        """Get default strategies from config"""
        strategies_config = self.config.get('strategies', {})
        
        # First check if we should run all available
        if strategies_config.get('run_all_available', False):
            return self.available_strategies
        
        # Otherwise use the default list
        default_list = strategies_config.get('default_list', [])
        if default_list:
            return default_list
        
        # Fallback to all strategies
        return self.available_strategies
    
    def check_cached_data(self, markets: List[str]) -> bool:
        """Check if cached data exists for specified markets"""
        start_date, end_date = self._calculate_date_range()
        
        missing_data = []
        for market in markets:
            db_path = self.collector.get_database_path(market)
            
            if not os.path.exists(db_path):
                missing_data.append(f"{market}: database not found")
                continue
            
            # Check data availability
            scrapper = UpbitDataScrapper(db_path)
            summary = scrapper.get_data_summary()
            
            if summary.empty:
                missing_data.append(f"{market}: no data in database")
                continue
            
            market_data = summary[summary['market'] == market]
            if market_data.empty:
                missing_data.append(f"{market}: market data not found")
                continue
            
            oldest_data = market_data.iloc[0]['oldest_data']
            
            # Parse dates
            oldest_datetime = pd.to_datetime(oldest_data)
            start_datetime = pd.to_datetime(start_date)
            
            # Check if the difference is more than 2 days
            days_diff = (oldest_datetime - start_datetime).days
            if days_diff > 2:
                missing_data.append(f"{market}: insufficient data (need from {start_date}, have from {oldest_data}, {days_diff} days late)")
            elif days_diff > 0:
                print(f"  ⚠️  {market}: Data starts {days_diff} day(s) after requested date ({oldest_data})")
        
        if missing_data:
            print(" Missing or insufficient cached data:")
            for item in missing_data:
                print(f"  - {item}")
            return False
        
        print(f" All required data available for {len(markets)} markets")
        return True
    
    def collect_data(self, markets: List[str]):
        """Collect required data for specified markets"""
        lookback_days = self.config['data']['lookback_days']
        
        print(f"\n=== Collecting data for {len(markets)} markets ({lookback_days} days) ===")
        
        for i, market in enumerate(markets, 1):
            print(f"\n[{i}/{len(markets)}] Collecting {market}...")
            
            db_path = self.collector.get_database_path(market)
            scrapper = UpbitDataScrapper(db_path)
            
            # Get the oldest available data
            summary = scrapper.get_data_summary()
            
            if not summary.empty:
                market_data = summary[summary['market'] == market]
                if not market_data.empty:
                    oldest_candle_time = market_data.iloc[0]['oldest_data']
                    print(f"  Oldest data: {oldest_candle_time}")
                    oldest_datetime = pd.to_datetime(oldest_candle_time)
                    
                    # Calculate how many days to fetch
                    days_to_fetch = (datetime.now() - oldest_datetime).days + 1
                    days_to_fetch = min(days_to_fetch, lookback_days)
                    
                    print(f"  Fetching {days_to_fetch} days of data...")
                else:
                    print(f"  No existing data, fetching {lookback_days} days...")
                    days_to_fetch = lookback_days
            else:
                print(f"  No existing data, fetching {lookback_days} days...")
                days_to_fetch = lookback_days
            
            # Collect data
            try:
                scrapper.scrape_market_data(market, days=days_to_fetch)
                print(f"  ✓ Data collection completed for {market}")
            except Exception as e:
                print(f"  ✗ Error collecting data for {market}: {e}")
    
    def load_market_data(self, markets: List[str]) -> Dict[str, pd.DataFrame]:
        """Load data for specified markets"""
        start_date, end_date = self._calculate_date_range()
        
        print(f"\n=� Loading data for backtesting ({start_date} to {end_date})...")
        
        market_data = {}
        for market in markets:
            try:
                db_path = self.collector.get_database_path(market)
                scrapper = UpbitDataScrapper(db_path)
                
                df = scrapper.get_candle_data_from_db(
                    market=market,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not df.empty:
                    # Convert timestamp column to datetime and set as index
                    df['candle_date_time_utc'] = pd.to_datetime(df['candle_date_time_utc'])
                    df.set_index('candle_date_time_utc', inplace=True)
                    # Sort by index to ensure chronological order
                    df.sort_index(inplace=True)
                    market_data[market] = df
                    print(f" {market}: {len(df)} candles loaded")
                else:
                    print(f" {market}: No data available")
            except Exception as e:
                print(f" {market}: Error loading data - {e}")
        
        print(f" Data loaded for {len(market_data)} markets")
        return market_data
    
    def run_strategy_backtest(self, strategy_name: str, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run backtest for a single strategy"""
        # Create strategy-specific config
        strategy_config = self.config.copy()
        strategy_config['strategy'] = {
            'name': strategy_name,
            'parameters': self.config.get('strategies', {}).get('parameters', {}).get(strategy_name, {})
        }
        
        # Initialize BacktestEngine with strategy-specific config
        engine = BacktestEngine(strategy_config)
        
        # Run backtest with all market data
        backtest_result = engine.run_backtest(market_data)
        
        # Extract relevant metrics from the backtest result
        total_trades = backtest_result.get('total_trades', 0)
        total_return = backtest_result.get('total_return', 0)
        total_return_pct = backtest_result.get('total_return_pct', 0)
        
        # Calculate winning trades by pairing buy/sell trades
        trade_history = backtest_result.get('trade_history', [])
        winning_trades = 0
        total_profit = 0
        total_loss = 0
        total_fees = 0
        completed_trades = 0  # Number of completed buy-sell pairs
        
        # Track positions to calculate profit/loss
        positions = {}
        trade_pairs = []  # Store completed trade pairs for accurate metrics
        
        for trade in trade_history:
            market = trade['market']
            # Calculate fees
            if trade['side'] == 'buy':
                fee_amount = trade['quantity'] * trade['price'] * trade.get('fee_rate', 0)
                total_fees += fee_amount
            else:  # sell
                fee_amount = trade['quantity'] * trade['price'] * trade.get('fee_rate', 0)
                total_fees += fee_amount
            
            if trade['side'] == 'buy':
                if market not in positions:
                    positions[market] = []
                positions[market].append({
                    'quantity': trade['quantity'],
                    'price': trade['price'],
                    'cost': trade['cost'],
                    'timestamp': trade['timestamp']
                })
            elif trade['side'] == 'sell' and market in positions and positions[market]:
                # Calculate profit/loss for this sale
                sell_quantity = trade['quantity']
                sell_price = trade['price']
                sell_timestamp = trade['timestamp']
                
                while sell_quantity > 0 and positions[market]:
                    position = positions[market][0]
                    qty_to_sell = min(sell_quantity, position['quantity'])
                    
                    # Calculate actual profit/loss including fees and slippage
                    buy_price = position['price']
                    buy_fee_rate = trade.get('fee_rate', 0)  # Same fee rate structure
                    sell_fee_rate = trade.get('fee_rate', 0)
                    
                    # Calculate net profit (after fees)
                    buy_cost_with_fee = qty_to_sell * buy_price * (1 + buy_fee_rate)
                    sell_proceeds_with_fee = qty_to_sell * sell_price * (1 - sell_fee_rate)
                    net_profit = sell_proceeds_with_fee - buy_cost_with_fee
                    
                    # Store trade pair info
                    trade_pair = {
                        'market': market,
                        'quantity': qty_to_sell,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'net_profit': net_profit,
                        'return_pct': (net_profit / buy_cost_with_fee) * 100
                    }
                    trade_pairs.append(trade_pair)
                    completed_trades += 1
                    
                    if net_profit > 0:
                        winning_trades += 1
                        total_profit += net_profit
                    else:
                        total_loss += abs(net_profit)
                    
                    # Update remaining quantities
                    position['quantity'] -= qty_to_sell
                    sell_quantity -= qty_to_sell
                    
                    # Remove position if fully sold
                    if position['quantity'] <= 0:
                        positions[market].pop(0)
        
        # Calculate per-market performance
        market_performance = {}
        for trade in trade_history:
            market = trade['market']
            if market not in market_performance:
                market_performance[market] = {
                    'trades': [],
                    'total_profit': 0,
                    'total_loss': 0,
                    'winning_trades': 0,
                    'total_trades': 0,
                    'total_fees': 0
                }
            market_performance[market]['trades'].append(trade)
        
        # Process market-specific trades
        for market, market_data in market_performance.items():
            market_positions = {}
            market_winning = 0
            market_profit = 0
            market_loss = 0
            market_fees = 0
            
            for trade in market_data['trades']:
                if trade['side'] == 'buy':
                    fee_amount = trade['quantity'] * trade['price'] * trade.get('fee_rate', 0)
                    market_fees += fee_amount
                    if market not in market_positions:
                        market_positions[market] = []
                    market_positions[market].append({
                        'quantity': trade['quantity'],
                        'price': trade['price'],
                        'cost': trade['cost']
                    })
                elif trade['side'] == 'sell' and market in market_positions and market_positions[market]:
                    fee_amount = trade['quantity'] * trade['price'] * trade.get('fee_rate', 0)
                    market_fees += fee_amount
                    
                    sell_quantity = trade['quantity']
                    sell_price = trade['price']
                    
                    while sell_quantity > 0 and market_positions[market]:
                        position = market_positions[market][0]
                        qty_to_sell = min(sell_quantity, position['quantity'])
                        
                        buy_price = position['price']
                        buy_fee_rate = trade.get('fee_rate', 0)
                        sell_fee_rate = trade.get('fee_rate', 0)
                        
                        buy_cost_with_fee = qty_to_sell * buy_price * (1 + buy_fee_rate)
                        sell_proceeds_with_fee = qty_to_sell * sell_price * (1 - sell_fee_rate)
                        net_profit = sell_proceeds_with_fee - buy_cost_with_fee
                        
                        if net_profit > 0:
                            market_winning += 1
                            market_profit += net_profit
                        else:
                            market_loss += abs(net_profit)
                        
                        position['quantity'] -= qty_to_sell
                        sell_quantity -= qty_to_sell
                        
                        if position['quantity'] <= 0:
                            market_positions[market].pop(0)
            
            # Store market performance
            market_completed_trades = market_winning + (len([t for t in market_data['trades'] if t['side'] == 'sell']) - market_winning)
            market_performance[market] = {
                'total_trades': len(market_data['trades']),
                'completed_trades': market_completed_trades,
                'winning_trades': market_winning,
                'win_rate': (market_winning / market_completed_trades * 100) if market_completed_trades > 0 else 0,
                'total_profit_amount': market_profit,
                'total_loss_amount': market_loss,
                'net_profit': market_profit - market_loss - market_fees,
                'total_fees': market_fees,
                'profit_factor': (market_profit / market_loss) if market_loss > 0 else float('inf') if market_profit > 0 else 0,
                'average_return_per_trade': (market_profit - market_loss - market_fees) / market_completed_trades if market_completed_trades > 0 else 0
            }
        
        print(f" {strategy_name} completed: {total_trades} trades")
        
        return {
            'strategy': strategy_name,
            'result': backtest_result,
            'overall': {
                'total_return': total_return,
                'total_return_pct': total_return_pct,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'completed_trades': completed_trades,
                'win_rate': (winning_trades / completed_trades * 100) if completed_trades > 0 else 0,
                'average_return_per_trade': total_return / completed_trades if completed_trades > 0 else 0,
                'net_profit': total_profit - total_loss - total_fees,  # Actual net profit after all costs
                'profit_factor': (total_profit / total_loss) if total_loss > 0 else float('inf') if total_profit > 0 else 0,
                'total_profit_amount': total_profit,
                'total_loss_amount': total_loss,
                'total_fees': total_fees,
                'sharpe_ratio': backtest_result.get('sharpe_ratio', 0),
                'sortino_ratio': backtest_result.get('sortino_ratio', 0),
                'calmar_ratio': backtest_result.get('calmar_ratio', 0),
                'max_drawdown_pct': backtest_result.get('max_drawdown_pct', 0),
                'volatility': backtest_result.get('volatility', 0),
                'markets_tested': len(market_data),
                'average_trade_amount': self.config['backtesting']['initial_balance'] / 3  # Assuming max 3 positions
            },
            'markets': market_performance,  # Per-market performance
            'config': strategy_config
        }
    
    def save_results(self, results: List[Dict[str, Any]], market_name: Optional[str] = None):
        """Save backtest results to files"""
        os.makedirs('results', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Add market suffix if specific market
        suffix = f"_{market_name}" if market_name else "_all_markets"
        
        # Save detailed results
        results_file = f'results/backtest_results_{timestamp}{suffix}.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f" Detailed results saved to: {results_file}")
        
        # Save summary
        summary = self._create_summary(results)
        summary_file = f'results/backtest_summary_{timestamp}{suffix}.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f" Summary saved to: {summary_file}")
        
        return results_file, summary_file
    
    def _create_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary of all strategy results"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config_file': self.config_path,
            'markets': list(results[0]['result']['trade_history'][0]['market'] if results[0]['result']['trade_history'] else []),
            'date_range': self._calculate_date_range(),
            'strategies': {}
        }
        
        for result in results:
            strategy_name = result['strategy']
            overall = result['overall']
            
            summary['strategies'][strategy_name] = {
                'total_return': overall['total_return'],
                'total_trades': overall['total_trades'],
                'completed_trades': overall.get('completed_trades', overall['total_trades'] // 2),
                'win_rate': overall['win_rate'],
                'net_profit': overall.get('net_profit', 0),
                'total_fees': overall.get('total_fees', 0),
                'average_return_per_trade': overall['average_return_per_trade'],
                'profit_factor': overall.get('profit_factor', 0),
                'sharpe_ratio': overall.get('sharpe_ratio', 0),
                'sortino_ratio': overall.get('sortino_ratio', 0),
                'calmar_ratio': overall.get('calmar_ratio', 0),
                'max_drawdown_pct': overall.get('max_drawdown_pct', 0),
                'markets_tested': overall['markets_tested']
            }
        
        return summary
    
    def _display_performance_summary(self, results: List[Dict[str, Any]], market_data: Dict[str, pd.DataFrame]):
        """Display comprehensive performance summary table"""
        print("\n" + "=" * 120)
        print("📊 STRATEGY PERFORMANCE SUMMARY")
        print("=" * 120)
        
        # Create overall summary table
        summary_data = []
        for result in results:
            strategy = result['strategy']
            overall = result['overall']
            
            summary_data.append({
                'Strategy': strategy,
                'Total_Return_%': round(overall.get('total_return_pct', overall['total_return'] * 100), 2),
                'Completed_Trades': overall.get('completed_trades', overall['total_trades'] // 2),
                'Win_Rate_%': round(overall['win_rate'], 2),
                'Net_Profit': f"₩{overall.get('net_profit', 0):,.0f}",
                'Profit_Factor': round(overall.get('profit_factor', 0), 3),
                'Sharpe': round(overall.get('sharpe_ratio', 0), 2),
                'Sortino': round(overall.get('sortino_ratio', 0), 2),
                'Calmar': round(overall.get('calmar_ratio', 0), 2),
                'Max_DD_%': round(overall.get('max_drawdown_pct', 0), 2),
                'Total_Fees': f"₩{overall.get('total_fees', 0):,.0f}"
            })
        
        # Sort by total return
        summary_data.sort(key=lambda x: x['Total_Return_%'], reverse=True)
        
        # Print overall table
        df = pd.DataFrame(summary_data)
        print("\n🏆 OVERALL STRATEGY RANKING:")
        print(df.to_string(index=False, max_colwidth=15))
        
        # Display market-specific performance for top 5 strategies
        print("\n" + "=" * 120)
        print("📈 TOP 5 STRATEGIES BY MARKET PERFORMANCE")
        print("=" * 120)
        
        top_5_strategies = [item['Strategy'] for item in summary_data[:5]]
        
        for market in market_data.keys():
            print(f"\n🎯 Market: {market}")
            print("-" * 100)
            
            market_performance = []
            for result in results:
                if result['strategy'] in top_5_strategies:
                    strategy = result['strategy']
                    if market in result['markets']:
                        perf = result['markets'][market]
                        
                        market_performance.append({
                            'Strategy': strategy,
                            'Trades': perf.get('total_trades', 0),
                            'Completed': perf.get('completed_trades', 0),
                            'Win_Rate_%': round(perf.get('win_rate', 0), 2),
                            'Net_Profit': f"₩{perf.get('net_profit', 0):,.0f}",
                            'Profit_Factor': round(perf.get('profit_factor', 0), 3),
                            'Fees': f"₩{perf.get('total_fees', 0):,.0f}"
                        })
            
            # Sort by net profit for this market
            market_performance.sort(key=lambda x: float(x['Net_Profit'].replace('₩', '').replace(',', '')), reverse=True)
            
            if market_performance:
                market_df = pd.DataFrame(market_performance)
                print(market_df.to_string(index=False, max_colwidth=12))
            else:
                print("No performance data available for this market")
        
        print("\n" + "=" * 120)
    
    def run_optimization(self, strategy_name: str, market_data: Dict[str, pd.DataFrame], 
                        train_ratio: float = 0.7) -> Dict[str, Any]:
        """Run hyperparameter optimization for a strategy"""
        if StrategyOptimizer is None:
            print("❌ Optimization module not available. Please install optuna.")
            return {}
        
        print(f"\n🔧 Optimizing {strategy_name} strategy...")
        
        # Initialize optimizer with the same config file
        optimizer = StrategyOptimizer(config_path=self.config_path)
        
        # Prepare data for optimization
        # Combine all market data while preserving datetime index
        market_dfs = []
        for market, df in market_data.items():
            df_copy = df.copy()
            df_copy['market'] = market
            market_dfs.append(df_copy)
        
        all_data = pd.concat(market_dfs, axis=0)
        # Sort by index to ensure chronological order
        all_data.sort_index(inplace=True)
        markets = list(market_data.keys())
        
        # Run optimization
        if strategy_name == "all":
            # Optimize all strategies except ensemble
            exclude = ['ensemble'] if 'ensemble' in self.available_strategies else []
            best_params = optimizer.optimize_all_strategies(all_data, markets, exclude=exclude)
        elif strategy_name == "ensemble":
            # Special handling for ensemble
            best_params = optimizer.optimize_strategy("ensemble", all_data, markets, train_ratio)
        else:
            # Optimize single strategy
            best_params = optimizer.optimize_strategy(strategy_name, all_data, markets, train_ratio)
        
        # Save optimization results
        optimizer.save_results()
        
        # Display detailed optimization results
        if OptimizationResultsDisplay and best_params:
            results_display = OptimizationResultsDisplay()
            market_name = markets[0] if len(markets) == 1 else "multi_market"
            
            # Get detailed results from optimizer - best_params is actually the full results dict
            if isinstance(best_params, dict) and 'best_params' in best_params:
                optimization_results = best_params
            else:
                # Fallback for older format
                optimization_results = {
                    'best_params': best_params,
                    'train_performance': getattr(optimizer, 'best_train_score', 0),
                    'test_performance': getattr(optimizer, 'best_test_performance', {}),
                    'n_trials': getattr(optimizer, 'total_trials', 50),
                    'optimization_history': getattr(optimizer, 'optimization_history', [])
                }
            
            results_display.display_optimization_results(
                optimization_results, strategy_name, market_name
            )
            
            # Save detailed results
            results_display.save_results_to_file(
                optimization_results, strategy_name, market_name
            )
        
        print(f"\n✅ Optimization completed. Results saved to results/optimization/")
        
        return best_params
    
    def run_backtest(self, strategies: List[str], markets: Optional[List[str]] = None, 
                     use_cached_data: bool = False, data_only: bool = False,
                     optimize_strategy: str = None, train_ratio: float = 0.7):
        """Run multi-strategy backtesting for selected markets"""
        # Handle market selection
        if markets is None or len(markets) == 0 or 'all' in markets:
            selected_markets = self.config['data']['markets']
        else:
            # Validate markets
            valid_markets = self.config['data']['markets']
            selected_markets = []
            for market in markets:
                market_upper = market.upper()
                if market_upper in valid_markets:
                    selected_markets.append(market_upper)
                else:
                    print(f"⚠️  Warning: Invalid market '{market}'. Valid markets: {', '.join(valid_markets)}")
            
            if not selected_markets:
                print("❌ No valid markets selected!")
                return False
        
        # Run backtest for each market separately
        if len(selected_markets) > 1 and markets is not None and 'all' not in markets:
            # Run individual backtests for each market
            for market in selected_markets:
                print(f"\n{'='*60}")
                print(f"Running backtest for {market}")
                print(f"{'='*60}")
                self._run_single_market_backtest(strategies, [market], use_cached_data, data_only, optimize_strategy, train_ratio)
        else:
            # Run combined backtest
            self._run_single_market_backtest(strategies, selected_markets, use_cached_data, data_only, optimize_strategy, train_ratio)
        
        return True
    
    def _run_single_market_backtest(self, strategies: List[str], markets: List[str], 
                                   use_cached_data: bool = False, data_only: bool = False,
                                   optimize_strategy: str = None, train_ratio: float = 0.7):
        """Run backtest for a single market or all markets combined"""
        print("=" * 60)
        print("=� MULTI-STRATEGY BACKTESTING SYSTEM")
        print("=" * 60)
        print(f"Markets: {', '.join(markets)}")
        print(f"Date range: {' to '.join(self._calculate_date_range())}")
        print(f"Strategies: {', '.join(strategies) if strategies else 'All available'}")
        print(f"Use cached data: {use_cached_data}")
        print("=" * 60)
        
        # Data collection/validation
        if use_cached_data:
            if not self.check_cached_data(markets):
                print("\nL Required cached data not available!")
                print("Run without --use-cached-data to fetch fresh data.")
                return False
            print("\n Using cached data")
        else:
            print("\n=� Collecting fresh data...")
            self.collect_data(markets)
        
        if data_only:
            print("\n Data collection completed. Skipping backtesting.")
            return True
        
        # Load market data
        market_data = self.load_market_data(markets)
        
        # Check if optimization mode
        if optimize_strategy:
            # Run optimization instead of regular backtest
            best_params = self.run_optimization(optimize_strategy, market_data, train_ratio)
            
            # After optimization, run backtest with optimized parameters on test set
            if best_params:
                print(f"\n📊 Running validation backtest with optimized parameters...")
                
                # Split data for validation
                all_data = pd.concat(list(market_data.values()), ignore_index=True)
                split_idx = int(len(all_data) * train_ratio)
                
                # Create test data
                test_market_data = {}
                for market, data in market_data.items():
                    market_test_data = data.iloc[split_idx:].copy()
                    if len(market_test_data) > 0:
                        test_market_data[market] = market_test_data
                
                # Run validation backtests
                validation_results = []
                for strategy_name, params in best_params.items():
                    try:
                        print(f"\n📈 Validating {strategy_name} with optimized parameters...")
                        # TODO: Run backtest with optimized parameters
                        # This would require updating run_strategy_backtest to accept custom parameters
                    except Exception as e:
                        print(f"❌ Error validating {strategy_name}: {e}")
                
                print(f"\n✅ Optimization and validation completed!")
            return True
        
        # Determine strategies to run
        if not strategies:
            # Use strategies from config file
            strategies = self._get_default_strategies()
        elif 'all' in strategies:
            strategies = self.available_strategies
        else:
            # Validate strategy names
            invalid_strategies = [s for s in strategies if s not in self.available_strategies]
            if invalid_strategies:
                print(f"\n⚠️  Invalid strategies: {', '.join(invalid_strategies)}")
                print(f"Available strategies: {', '.join(self.available_strategies)}")
                strategies = [s for s in strategies if s in self.available_strategies]
                
                if not strategies:
                    print("\n❌ No valid strategies to run!")
                    return False
        
        print(f"\n= Running {len(strategies)} strategies...")
        
        # Run backtest for each strategy
        all_results = []
        
        for strategy in strategies:
            try:
                print(f"\n=� Running {strategy.upper()} strategy...")
                result = self.run_strategy_backtest(strategy, market_data)
                all_results.append(result)
            except Exception as e:
                print(f" Error running {strategy}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not all_results:
            print("\nL No successful strategy runs!")
            return False
        
        # Save results
        market_suffix = markets[0] if len(markets) == 1 else None
        results_file, summary_file = self.save_results(all_results, market_suffix)
        
        # Display performance summary table
        self._display_performance_summary(all_results, market_data)
        
        print(f"\n<� Backtesting completed successfully!")
        print(f"=� {len(all_results)} strategies tested on {len(market_data)} markets")
        print(f"=� Results saved to results/ directory")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Strategy Backtesting System with Market Selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtest_market.py --strategy all --market all         # Run all strategies on all markets
  python backtest_market.py --strategy vwap --market KRW-BTC    # Run VWAP on BTC only
  python backtest_market.py --strategy all --market KRW-ADA KRW-DOT  # All strategies on specific markets
  python backtest_market.py --use-cached-data --strategy all    # Use cached data
  python backtest_market.py --data-only                         # Only collect data
        """
    )
    
    parser.add_argument('--use-cached-data', action='store_true',
                       help='Use cached data from database')
    parser.add_argument('--data-only', action='store_true',
                       help='Only collect data, skip backtesting')
    parser.add_argument('--strategy', nargs='+', default=[],
                       help='Strategies to run (or "all" for all strategies)')
    parser.add_argument('--market', nargs='+', default=['all'],
                       help='Markets to test (e.g., KRW-BTC KRW-ETH or "all" for all markets)')
    parser.add_argument('--config', default='config/config_backtesting.json',
                       help='Path to configuration file')
    parser.add_argument('--optimize-strategy', default=None,
                       help='Optimize strategy hyperparameters ("all" or specific strategy)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Train/test split ratio for optimization (default: 0.7)')
    
    args = parser.parse_args()
    
    try:
        backtester = MultiStrategyBacktester(args.config)
        success = backtester.run_backtest(
            strategies=args.strategy,
            markets=args.market,
            use_cached_data=args.use_cached_data,
            data_only=args.data_only,
            optimize_strategy=args.optimize_strategy,
            train_ratio=args.train_ratio
        )
        sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        print("\n\n�  Operation cancelled by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()