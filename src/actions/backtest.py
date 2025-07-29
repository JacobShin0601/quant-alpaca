#!/usr/bin/env python3
"""
Multi-Strategy Backtesting System
Usage:
  python backtest.py [--use-cached-data] [--data-only] [--strategy STRATEGIES...] [--config CONFIG_FILE]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.scrapper import UpbitDataScrapper
from data.collector import UpbitDataCollector
from backtesting.engine import BacktestEngine
from strategies import STRATEGIES


class MultiStrategyBacktester:
    def __init__(self, config_path: str = "config/config_backtesting.json"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize data collector
        self.collector = UpbitDataCollector(
            self.config['data']['database_directory'],
            self.config['data']['database_pattern']
        )
        
        # Backtesting engine will be initialized per strategy
        
        # Available strategies
        self.available_strategies = list(STRATEGIES.keys())
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f" Config loaded from: {self.config_path}")
        return config
    
    def _calculate_date_range(self) -> tuple:
        """Calculate start and end dates for backtesting"""
        end_date = datetime.strptime(self.config['data']['end_date'], '%Y-%m-%d')
        lookback_days = self.config['data']['lookback_days']
        start_date = end_date - timedelta(days=lookback_days)
        
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    
    def _get_default_strategies(self) -> List[str]:
        """Get default strategies from config file"""
        if 'strategies' in self.config:
            strategies_config = self.config['strategies']
            
            # Check if run_all_available is set to True
            if strategies_config.get('run_all_available', False):
                return self.available_strategies
            
            # Use default_list from config
            default_strategies = strategies_config.get('default_list', [])
            
            # Validate that all default strategies are available
            valid_strategies = [s for s in default_strategies if s in self.available_strategies]
            invalid_strategies = [s for s in default_strategies if s not in self.available_strategies]
            
            if invalid_strategies:
                print(f"⚠️  Warning: Invalid default strategies in config: {', '.join(invalid_strategies)}")
            
            if valid_strategies:
                print(f"📋 Using default strategies from config: {', '.join(valid_strategies)}")
                return valid_strategies
        
        # Fallback to all available strategies
        print(f"📋 No valid default strategies in config, using all available strategies")
        return self.available_strategies
    
    def check_cached_data(self) -> bool:
        """Check if required cached data exists"""
        markets = self.config['data']['markets']
        start_date, _ = self._calculate_date_range()
        
        missing_data = []
        
        for market in markets:
            db_path = self.collector.get_database_path(market)
            
            if not os.path.exists(db_path):
                missing_data.append(f"{market}: database not found")
                continue
            
            # Check data completeness using scrapper
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
            candle_count = market_data.iloc[0]['candle_count']
            
            # Allow for a small grace period (2 days) for data availability
            # Convert string dates to datetime for comparison
            oldest_datetime = pd.to_datetime(oldest_data)
            start_datetime = pd.to_datetime(start_date)
            
            # Check if the difference is more than 2 days
            days_diff = (oldest_datetime - start_datetime).days
            if days_diff > 2:
                missing_data.append(f"{market}: insufficient data (need from {start_date}, have from {oldest_data}, {days_diff} days late)")
            elif days_diff > 0:
                print(f"  ⚠️  {market}: Data starts {days_diff} day(s) after requested date ({oldest_data})")
        
        if missing_data:
            print(" Missing or insufficient cached data:")
            for item in missing_data:
                print(f"  - {item}")
            return False
        
        print(f" All required data available for {len(markets)} markets")
        return True
    
    def collect_data(self):
        """Collect required data for all markets"""
        markets = self.config['data']['markets']
        lookback_days = self.config['data']['lookback_days']
        
        print(f"\n=== Collecting data for {len(markets)} markets ({lookback_days} days) ===")
        
        for i, market in enumerate(markets, 1):
            print(f"\n[{i}/{len(markets)}] Collecting {market}...")
            
            db_path = self.collector.get_database_path(market)
            scrapper = UpbitDataScrapper(db_path)
            scrapper.scrape_market_data(market, days=lookback_days)
        
        print("\n Data collection completed")
    
    def load_market_data(self) -> Dict[str, pd.DataFrame]:
        """Load market data for backtesting"""
        markets = self.config['data']['markets']
        start_date, end_date = self._calculate_date_range()
        
        market_data = {}
        
        print(f"\n=� Loading data for backtesting ({start_date} to {end_date})...")
        
        for market in markets:
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
                print(f" {market}: {len(df)} candles loaded")
            else:
                print(f" {market}: No data available")
        
        if not market_data:
            raise ValueError("No market data loaded!")
        
        print(f" Data loaded for {len(market_data)} markets")
        return market_data
    
    def run_strategy_backtest(self, strategy_name: str, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run backtest for a specific strategy"""
        if strategy_name not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        print(f"\n=� Running {strategy_name.upper()} strategy...")
        
        # Create a config with the specific strategy
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
                'average_return_per_trade': (market_profit - market_loss - market_fees) / completed_trades if completed_trades > 0 else 0
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
    
    def _calculate_overall_performance(self, market_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate overall performance across all markets"""
        total_return = 0
        total_trades = 0
        total_wins = 0
        all_returns = []
        
        for market, result in market_results.items():
            if 'performance' in result:
                perf = result['performance']
                total_return += perf.get('total_return', 0)
                total_trades += perf.get('total_trades', 0)
                total_wins += perf.get('winning_trades', 0)
                
                if 'trade_returns' in perf:
                    all_returns.extend(perf['trade_returns'])
        
        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        avg_return = sum(all_returns) / len(all_returns) if all_returns else 0
        
        return {
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': total_wins,
            'win_rate': win_rate,
            'average_return_per_trade': avg_return,
            'markets_tested': len(market_results)
        }
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save backtest results to files"""
        os.makedirs('results', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        results_file = f'results/backtest_results_{timestamp}.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f" Detailed results saved to: {results_file}")
        
        # Save summary
        summary = self._create_summary(results)
        summary_file = f'results/backtest_summary_{timestamp}.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f" Summary saved to: {summary_file}")
        
        return results_file, summary_file
    
    def _create_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary of all strategy results"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config_file': self.config_path,
            'markets': self.config['data']['markets'],
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
    
    def run_backtest(self, strategies: List[str], use_cached_data: bool = False, data_only: bool = False):
        """Run multi-strategy backtesting"""
        print("=" * 60)
        print("=� MULTI-STRATEGY BACKTESTING SYSTEM")
        print("=" * 60)
        print(f"Markets: {', '.join(self.config['data']['markets'])}")
        print(f"Date range: {' to '.join(self._calculate_date_range())}")
        print(f"Strategies: {', '.join(strategies) if strategies else 'All available'}")
        print(f"Use cached data: {use_cached_data}")
        print("=" * 60)
        
        # Data collection/validation
        if use_cached_data:
            if not self.check_cached_data():
                print("\nL Required cached data not available!")
                print("Run without --use-cached-data to fetch fresh data.")
                return False
            print("\n Using cached data")
        else:
            print("\n=� Collecting fresh data...")
            self.collect_data()
        
        if data_only:
            print("\n Data collection completed. Skipping backtesting.")
            return True
        
        # Load market data
        market_data = self.load_market_data()
        
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
                print(f"\nL Invalid strategies: {', '.join(invalid_strategies)}")
                print(f"Available strategies: {', '.join(self.available_strategies)}")
                return False
        
        print(f"\n= Running {len(strategies)} strategies...")
        
        # Run backtest for each strategy
        all_results = []
        
        for strategy in strategies:
            try:
                result = self.run_strategy_backtest(strategy, market_data)
                all_results.append(result)
            except Exception as e:
                print(f" Error running {strategy}: {e}")
                continue
        
        if not all_results:
            print("\nL No successful strategy runs!")
            return False
        
        # Save results
        results_file, summary_file = self.save_results(all_results)
        
        # Display performance summary table
        self._display_performance_summary(all_results, market_data)
        
        print(f"\n<� Backtesting completed successfully!")
        print(f"=� {len(all_results)} strategies tested on {len(market_data)} markets")
        print(f"=� Results saved to results/ directory")
        
        return True
    
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


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Strategy Backtesting System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtest.py --strategy all                      # Run all strategies on all markets
  python backtest.py --strategy vwap --market KRW-BTC    # Run VWAP on BTC only
  python backtest.py --strategy all --market KRW-ADA KRW-DOT  # All strategies on specific markets
  python backtest.py --use-cached-data --strategy all   # Use cached data
  python backtest.py --data-only                        # Only collect data
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
    
    args = parser.parse_args()
    
    try:
        backtester = MultiStrategyBacktester(args.config)
        success = backtester.run_backtest(
            strategies=args.strategy,
            use_cached_data=args.use_cached_data,
            data_only=args.data_only
        )
        sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        print("\n\n�  Operation cancelled by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\nL Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()