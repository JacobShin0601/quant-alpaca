#!/usr/bin/env python3
"""
Per-Market Strategy Backtesting System
Runs each strategy separately for each market to get detailed performance metrics
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.scrapper import UpbitDataScrapper
from data.collector import UpbitDataCollector
from backtesting.engine import BacktestEngine
from actions.strategies import STRATEGIES


class PerMarketBacktester:
    def __init__(self, config_path: str = "config/config_backtesting.json"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize data collector
        self.collector = UpbitDataCollector(
            self.config['data']['database_directory'],
            self.config['data']['database_pattern']
        )
        
        self.available_strategies = list(STRATEGIES.keys())
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"✓ Config loaded from: {self.config_path}")
        return config
    
    def _calculate_date_range(self) -> tuple:
        """Calculate start and end dates for backtesting"""
        end_date = datetime.strptime(self.config['data']['end_date'], '%Y-%m-%d')
        lookback_days = self.config['data']['lookback_days']
        start_date = end_date - timedelta(days=lookback_days)
        
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    
    def load_market_data(self) -> Dict[str, pd.DataFrame]:
        """Load market data for backtesting"""
        markets = self.config['data']['markets']
        start_date, end_date = self._calculate_date_range()
        
        market_data = {}
        
        print(f"\n=== Loading data for backtesting ({start_date} to {end_date})...")
        
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
                print(f"  ✓ {market}: {len(df)} candles loaded")
            else:
                print(f"  ✗ {market}: No data available")
        
        if not market_data:
            raise ValueError("No market data loaded!")
        
        print(f"✓ Data loaded for {len(market_data)} markets")
        return market_data
    
    def run_single_market_backtest(self, strategy_name: str, market: str, market_df: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest for a single strategy on a single market"""
        # Create a config with the specific strategy
        strategy_config = self.config.copy()
        strategy_config['strategy'] = {
            'name': strategy_name,
            'parameters': self.config.get('strategies', {}).get('parameters', {}).get(strategy_name, {})
        }
        
        # Initialize BacktestEngine with strategy-specific config
        engine = BacktestEngine(strategy_config)
        
        # Run backtest with single market data
        backtest_result = engine.run_backtest({market: market_df})
        
        # Extract relevant metrics
        total_trades = backtest_result.get('total_trades', 0)
        total_return = backtest_result.get('total_return', 0)
        total_return_pct = backtest_result.get('total_return_pct', 0)
        
        # Calculate winning trades and profit/loss
        trade_history = backtest_result.get('trade_history', [])
        positions = []
        completed_trades = []
        
        for trade in trade_history:
            if trade['side'] == 'buy':
                positions.append({
                    'quantity': trade['quantity'],
                    'price': trade['price'],
                    'cost': trade['cost'],
                    'timestamp': trade['timestamp']
                })
            elif trade['side'] == 'sell' and positions:
                # Match with FIFO
                sell_quantity = trade['quantity']
                sell_proceeds = trade.get('proceeds', trade['quantity'] * trade['price'])
                
                while sell_quantity > 0 and positions:
                    position = positions[0]
                    qty_to_match = min(sell_quantity, position['quantity'])
                    
                    # Calculate profit/loss for this portion
                    buy_cost = (qty_to_match / position['quantity']) * position['cost']
                    sell_revenue = (qty_to_match / trade['quantity']) * sell_proceeds
                    profit = sell_revenue - buy_cost
                    
                    completed_trades.append({
                        'buy_time': position['timestamp'],
                        'sell_time': trade['timestamp'],
                        'quantity': qty_to_match,
                        'buy_price': position['price'],
                        'sell_price': trade['price'],
                        'profit': profit,
                        'return_pct': (profit / buy_cost) * 100
                    })
                    
                    # Update remaining quantities
                    position['quantity'] -= qty_to_match
                    sell_quantity -= qty_to_match
                    
                    # Remove position if fully sold
                    if position['quantity'] <= 0:
                        positions.pop(0)
        
        # Calculate statistics
        winning_trades = sum(1 for t in completed_trades if t['profit'] > 0)
        total_profit = sum(t['profit'] for t in completed_trades if t['profit'] > 0)
        total_loss = sum(abs(t['profit']) for t in completed_trades if t['profit'] < 0)
        
        return {
            'market': market,
            'strategy': strategy_name,
            'total_return_pct': total_return_pct,
            'total_trades': total_trades,
            'completed_trades': len(completed_trades),
            'winning_trades': winning_trades,
            'win_rate': (winning_trades / len(completed_trades) * 100) if completed_trades else 0,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'profit_factor': (total_profit / total_loss) if total_loss > 0 else float('inf') if total_profit > 0 else 0,
            'sharpe_ratio': backtest_result.get('sharpe_ratio', 0),
            'max_drawdown_pct': backtest_result.get('max_drawdown_pct', 0),
            'avg_trade_return': np.mean([t['return_pct'] for t in completed_trades]) if completed_trades else 0,
            'best_trade': max([t['return_pct'] for t in completed_trades]) if completed_trades else 0,
            'worst_trade': min([t['return_pct'] for t in completed_trades]) if completed_trades else 0,
            'trade_details': completed_trades
        }
    
    def run_all_combinations(self, strategies: List[str] = None, markets: List[str] = None):
        """Run all strategy-market combinations"""
        # Load market data
        all_market_data = self.load_market_data()
        
        # Determine which strategies and markets to use
        if not strategies:
            strategies = self.available_strategies
        if not markets:
            markets = list(all_market_data.keys())
        
        print(f"\n=== Running {len(strategies)} strategies on {len(markets)} markets...")
        print(f"Total combinations: {len(strategies) * len(markets)}")
        
        results = []
        
        for i, strategy in enumerate(strategies, 1):
            print(f"\n[{i}/{len(strategies)}] Testing {strategy} strategy...")
            
            for j, market in enumerate(markets, 1):
                if market not in all_market_data:
                    print(f"  [{j}/{len(markets)}] {market}: No data available")
                    continue
                
                print(f"  [{j}/{len(markets)}] {market}...", end='', flush=True)
                
                try:
                    result = self.run_single_market_backtest(
                        strategy, 
                        market, 
                        all_market_data[market]
                    )
                    results.append(result)
                    print(f" ✓ ({result['completed_trades']} trades, {result['total_return_pct']:.2f}% return)")
                except Exception as e:
                    print(f" ✗ Error: {e}")
                    continue
        
        return results
    
    def display_top_results(self, results: List[Dict[str, Any]], top_n: int = 10):
        """Display top N results by return"""
        print("\n" + "=" * 120)
        print("📊 TOP {} MARKET-STRATEGY PERFORMANCE RANKING".format(top_n))
        print("=" * 120)
        
        # Sort by total return
        sorted_results = sorted(results, key=lambda x: x['total_return_pct'], reverse=True)
        
        # Prepare data for display
        display_data = []
        for i, result in enumerate(sorted_results[:top_n], 1):
            display_data.append({
                'Rank': i,
                'Market': result['market'],
                'Strategy': result['strategy'],
                'Return_%': f"{result['total_return_pct']:.3f}",
                'Trades': result['completed_trades'],
                'Win_Rate_%': f"{result['win_rate']:.1f}",
                'Profit_Factor': f"{result['profit_factor']:.2f}" if result['profit_factor'] != float('inf') else "∞",
                'Sharpe': f"{result['sharpe_ratio']:.2f}",
                'Max_DD_%': f"{result['max_drawdown_pct']:.2f}",
                'Best_Trade_%': f"{result['best_trade']:.2f}",
                'Worst_Trade_%': f"{result['worst_trade']:.2f}"
            })
        
        df = pd.DataFrame(display_data)
        print("\n" + df.to_string(index=False))
        
        # Show summary statistics
        print("\n" + "-" * 120)
        print("📈 SUMMARY STATISTICS")
        print("-" * 120)
        
        total_positive = sum(1 for r in results if r['total_return_pct'] > 0)
        total_negative = sum(1 for r in results if r['total_return_pct'] < 0)
        avg_return = np.mean([r['total_return_pct'] for r in results])
        
        print(f"Total combinations tested: {len(results)}")
        print(f"Profitable combinations: {total_positive} ({total_positive/len(results)*100:.1f}%)")
        print(f"Loss-making combinations: {total_negative} ({total_negative/len(results)*100:.1f}%)")
        print(f"Average return: {avg_return:.3f}%")
        
        # Best performing strategy and market
        best_strategy_returns = {}
        best_market_returns = {}
        
        for r in results:
            strategy = r['strategy']
            market = r['market']
            ret = r['total_return_pct']
            
            if strategy not in best_strategy_returns:
                best_strategy_returns[strategy] = []
            best_strategy_returns[strategy].append(ret)
            
            if market not in best_market_returns:
                best_market_returns[market] = []
            best_market_returns[market].append(ret)
        
        # Calculate average returns
        avg_strategy_returns = {s: np.mean(returns) for s, returns in best_strategy_returns.items()}
        avg_market_returns = {m: np.mean(returns) for m, returns in best_market_returns.items()}
        
        best_strategy = max(avg_strategy_returns, key=avg_strategy_returns.get)
        best_market = max(avg_market_returns, key=avg_market_returns.get)
        
        print(f"\nBest performing strategy (average): {best_strategy} ({avg_strategy_returns[best_strategy]:.3f}%)")
        print(f"Best performing market (average): {best_market} ({avg_market_returns[best_market]:.3f}%)")
        
        print("\n" + "=" * 120)
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save detailed results to file"""
        os.makedirs('results', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results (without trade details for file size)
        results_for_save = []
        for r in results:
            r_copy = r.copy()
            r_copy.pop('trade_details', None)  # Remove detailed trade info
            results_for_save.append(r_copy)
        
        results_file = f'results/per_market_backtest_{timestamp}.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'config_file': self.config_path,
                'results': results_for_save
            }, f, indent=2, default=str)
        
        print(f"\n✓ Results saved to: {results_file}")
        
        # Also save as CSV for easy analysis
        df = pd.DataFrame(results_for_save)
        csv_file = f'results/per_market_backtest_{timestamp}.csv'
        df.to_csv(csv_file, index=False)
        print(f"✓ CSV saved to: {csv_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Per-Market Strategy Backtesting System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtest_per_market.py                     # Run all strategies on all markets
  python backtest_per_market.py --strategy vwap macd  # Run specific strategies
  python backtest_per_market.py --markets KRW-BTC KRW-ETH  # Run on specific markets
  python backtest_per_market.py --top 20            # Show top 20 results
        """
    )
    
    parser.add_argument('--strategy', nargs='+', default=[],
                       help='Strategies to run (default: all)')
    parser.add_argument('--markets', nargs='+', default=[],
                       help='Markets to test (default: all from config)')
    parser.add_argument('--top', type=int, default=10,
                       help='Number of top results to display (default: 10)')
    parser.add_argument('--config', default='config/config_backtesting.json',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        backtester = PerMarketBacktester(args.config)
        
        # Run all combinations
        results = backtester.run_all_combinations(
            strategies=args.strategy if args.strategy else None,
            markets=args.markets if args.markets else None
        )
        
        if results:
            # Display top results
            backtester.display_top_results(results, top_n=args.top)
            
            # Save results
            backtester.save_results(results)
            
            print("\n✓ Backtesting completed successfully!")
        else:
            print("\n✗ No results generated!")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Operation cancelled by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()