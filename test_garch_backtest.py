#!/usr/bin/env python3
"""
Test GARCH-based position sizing with backtesting
"""

import os
import sys
import json
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(__file__))
from src.backtesting.engine import BacktestEngine

def load_market_data(market: str, lookback_days: int = 60) -> pd.DataFrame:
    """Load market data from SQLite database"""
    # Convert market format from KRW-BTC to KRW_BTC
    market_file = market.replace('-', '_')
    db_path = f"data/candles/{market_file}_candles.db"
    
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return pd.DataFrame()
    
    conn = sqlite3.connect(db_path)
    
    # Calculate start date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    # Get timestamp range from database first
    timestamp_range = pd.read_sql_query("SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts FROM candles", conn)
    min_timestamp = timestamp_range.iloc[0]['min_ts']
    max_timestamp = timestamp_range.iloc[0]['max_ts']
    
    # Calculate timestamp for lookback
    # The timestamps appear to be in milliseconds
    lookback_ms = lookback_days * 24 * 60 * 60 * 1000
    start_timestamp = max_timestamp - lookback_ms
    
    # Make sure we don't go before the earliest data
    start_timestamp = max(start_timestamp, min_timestamp)
    
    query = f"""
    SELECT market, candle_date_time_kst, opening_price, high_price, low_price, trade_price, 
           timestamp, candle_acc_trade_price, candle_acc_trade_volume
    FROM candles 
    WHERE timestamp >= {start_timestamp}
    ORDER BY timestamp
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Convert timestamp to datetime (from milliseconds)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df

def run_comparison_test():
    """Run backtests with and without GARCH sizing for comparison"""
    
    # Load config
    with open('config/config_backtesting.json', 'r') as f:
        config = json.load(f)
    
    # Test with a single strategy for cleaner comparison
    config['strategy'] = {
        'name': 'basic_momentum',
        'parameters': config['strategies']['parameters']['basic_momentum']
    }
    
    # Load data
    print("Loading market data...")
    markets = ["KRW-BTC", "KRW-ETH", "KRW-XRP"]
    data = {}
    
    for market in markets:
        df = load_market_data(market, lookback_days=90)  # More data for GARCH
        if not df.empty:
            data[market] = df
            print(f"Loaded {len(df)} candles for {market}")
    
    if not data:
        print("No data loaded!")
        return
    
    # Test 1: Without GARCH sizing
    print("\n" + "="*50)
    print("Test 1: Traditional Position Sizing (Signal Strength)")
    print("="*50)
    
    config['position_sizing']['use_garch_sizing'] = False
    engine1 = BacktestEngine(config)
    results1 = engine1.run_backtest(data)
    
    print(f"\nResults without GARCH:")
    print(f"Final Value: ₩{results1['final_value']:,.0f}")
    print(f"Total Return: {results1['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {results1['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {results1['max_drawdown_pct']:.2f}%")
    print(f"Total Trades: {results1['total_trades']}")
    
    # Test 2: With GARCH sizing
    print("\n" + "="*50)
    print("Test 2: GARCH-based Position Sizing")
    print("="*50)
    
    config['position_sizing']['use_garch_sizing'] = True
    engine2 = BacktestEngine(config)
    results2 = engine2.run_backtest(data)
    
    print(f"\nResults with GARCH:")
    print(f"Final Value: ₩{results2['final_value']:,.0f}")
    print(f"Total Return: {results2['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {results2['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {results2['max_drawdown_pct']:.2f}%")
    print(f"Total Trades: {results2['total_trades']}")
    
    # Comparison
    print("\n" + "="*50)
    print("Performance Comparison")
    print("="*50)
    
    return_diff = results2['total_return_pct'] - results1['total_return_pct']
    sharpe_diff = results2['sharpe_ratio'] - results1['sharpe_ratio']
    dd_diff = results2['max_drawdown_pct'] - results1['max_drawdown_pct']
    
    print(f"Return Difference: {return_diff:+.2f}%")
    print(f"Sharpe Difference: {sharpe_diff:+.3f}")
    print(f"Drawdown Difference: {dd_diff:+.2f}%")
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save comparison report
    comparison = {
        'test_timestamp': timestamp,
        'markets': markets,
        'strategy': 'advanced_vwap',
        'results_without_garch': {
            'final_value': results1['final_value'],
            'total_return_pct': results1['total_return_pct'],
            'sharpe_ratio': results1['sharpe_ratio'],
            'max_drawdown_pct': results1['max_drawdown_pct'],
            'total_trades': results1['total_trades']
        },
        'results_with_garch': {
            'final_value': results2['final_value'],
            'total_return_pct': results2['total_return_pct'],
            'sharpe_ratio': results2['sharpe_ratio'],
            'max_drawdown_pct': results2['max_drawdown_pct'],
            'total_trades': results2['total_trades']
        },
        'improvements': {
            'return_difference': return_diff,
            'sharpe_difference': sharpe_diff,
            'drawdown_difference': dd_diff
        }
    }
    
    os.makedirs('results/garch_comparison', exist_ok=True)
    with open(f'results/garch_comparison/comparison_{timestamp}.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nDetailed results saved to: results/garch_comparison/comparison_{timestamp}.json")

if __name__ == "__main__":
    # Check if arch package is installed
    try:
        import arch
    except ImportError:
        print("Error: arch package not installed.")
        print("Please install it with: pip install arch")
        sys.exit(1)
    
    run_comparison_test()