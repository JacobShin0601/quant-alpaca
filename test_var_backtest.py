#!/usr/bin/env python3
"""
Test VaR/CVaR risk management with backtesting
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, Any

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
    
    # Get timestamp range from database first
    timestamp_range = pd.read_sql_query("SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts FROM candles", conn)
    min_timestamp = timestamp_range.iloc[0]['min_ts']
    max_timestamp = timestamp_range.iloc[0]['max_ts']
    
    # Calculate timestamp for lookback
    lookback_ms = lookback_days * 24 * 60 * 60 * 1000
    start_timestamp = max_timestamp - lookback_ms
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

def plot_var_analysis(results: Dict, save_path: str = None):
    """Plot VaR analysis results"""
    if 'var_metrics' not in results:
        print("No VaR metrics to plot")
        return
    
    var_history = results['var_metrics']['var_history']
    if not var_history:
        print("No VaR history to plot")
        return
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data
    timestamps = [m['timestamp'] for m in var_history]
    var_amounts = [m['var_amount'] for m in var_history]
    cvar_amounts = [m['cvar_amount'] for m in var_history]
    daily_pnls = [m['daily_pnl'] for m in var_history]
    var_utilizations = [m['var_utilization'] * 100 for m in var_history]
    
    # Plot 1: VaR and CVaR amounts over time
    ax1.plot(timestamps, var_amounts, label='VaR (95%)', color='blue', alpha=0.7)
    ax1.plot(timestamps, cvar_amounts, label='CVaR (95%)', color='red', alpha=0.7)
    ax1.set_title('VaR and CVaR Amounts Over Time')
    ax1.set_ylabel('Amount (KRW)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Daily P&L vs VaR limits
    ax2.scatter(timestamps, daily_pnls, alpha=0.5, s=10, label='Daily P&L')
    ax2.plot(timestamps, [-v for v in var_amounts], 'r--', label='VaR Limit', alpha=0.7)
    ax2.plot(timestamps, [-c for c in cvar_amounts], 'orange', linestyle='--', label='CVaR Limit', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_title('Daily P&L vs Risk Limits')
    ax2.set_ylabel('P&L (KRW)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: VaR utilization
    colors = ['red' if u > 100 else 'orange' if u > 80 else 'green' for u in var_utilizations]
    ax3.scatter(timestamps, var_utilizations, c=colors, alpha=0.6, s=10)
    ax3.axhline(y=100, color='red', linestyle='--', label='Breach Level')
    ax3.axhline(y=80, color='orange', linestyle='--', label='Warning Level')
    ax3.set_title('VaR Utilization (%)')
    ax3.set_ylabel('Utilization (%)')
    ax3.set_ylim(0, max(150, max(var_utilizations) * 1.1))
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: VaR breach analysis
    breach_data = []
    window_size = 20  # 20-period rolling window
    
    for i in range(window_size, len(daily_pnls)):
        window_pnls = daily_pnls[i-window_size:i]
        window_vars = var_amounts[i-window_size:i]
        breaches = sum(1 for pnl, var in zip(window_pnls, window_vars) if pnl < -var)
        breach_rate = breaches / window_size
        breach_data.append(breach_rate * 100)
    
    if breach_data:
        ax4.plot(timestamps[window_size:], breach_data, label='Rolling Breach Rate', color='purple')
        ax4.axhline(y=5, color='red', linestyle='--', label='Expected (5%)')
        ax4.set_title(f'Rolling {window_size}-Period VaR Breach Rate')
        ax4.set_ylabel('Breach Rate (%)')
        ax4.set_ylim(0, max(20, max(breach_data) * 1.1))
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"VaR analysis plot saved to: {save_path}")
    else:
        plt.show()

def run_var_comparison_test():
    """Run backtests with and without VaR limits for comparison"""
    
    # Load config
    with open('config/config_backtesting.json', 'r') as f:
        config = json.load(f)
    
    # Test with a single strategy
    config['strategy'] = {
        'name': 'basic_momentum',
        'parameters': config['strategies']['parameters']['basic_momentum']
    }
    
    # Disable GARCH for cleaner comparison
    config['position_sizing']['use_garch_sizing'] = False
    
    # Load data
    print("Loading market data...")
    markets = ["KRW-BTC", "KRW-ETH", "KRW-XRP"]
    data = {}
    
    for market in markets:
        df = load_market_data(market, lookback_days=60)
        if not df.empty:
            data[market] = df
            print(f"Loaded {len(df)} candles for {market}")
    
    if not data:
        print("No data loaded!")
        return
    
    # Test 1: Without VaR limits
    print("\n" + "="*50)
    print("Test 1: Without VaR Risk Limits")
    print("="*50)
    
    config['var_risk_management']['enabled'] = False
    engine1 = BacktestEngine(config)
    results1 = engine1.run_backtest(data)
    
    print(f"\nResults without VaR limits:")
    print(f"Final Value: ₩{results1['final_value']:,.0f}")
    print(f"Total Return: {results1['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {results1['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {results1['max_drawdown_pct']:.2f}%")
    print(f"Total Trades: {results1['total_trades']}")
    
    # Test 2: With VaR limits
    print("\n" + "="*50)
    print("Test 2: With VaR Risk Limits")
    print("="*50)
    
    config['var_risk_management']['enabled'] = True
    engine2 = BacktestEngine(config)
    results2 = engine2.run_backtest(data)
    
    print(f"\nResults with VaR limits:")
    print(f"Final Value: ₩{results2['final_value']:,.0f}")
    print(f"Total Return: {results2['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {results2['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {results2['max_drawdown_pct']:.2f}%")
    print(f"Total Trades: {results2['total_trades']}")
    
    if 'var_metrics' in results2:
        var_metrics = results2['var_metrics']
        print(f"\nVaR Risk Metrics:")
        print(f"Average VaR (1-day): {var_metrics['average_var_1d']:.2%}")
        print(f"Average CVaR (1-day): {var_metrics['average_cvar_1d']:.2%}")
        print(f"VaR Breaches: {var_metrics['var_breach_count']} ({var_metrics['var_breach_rate']:.1%})")
        print(f"CVaR Breaches: {var_metrics['cvar_breach_count']} ({var_metrics['cvar_breach_rate']:.1%})")
        print(f"Max VaR Utilization: {var_metrics['max_var_utilization']:.1%}")
        print(f"Days with Limit Breach: {var_metrics['limit_breach_days']}")
    
    # Comparison
    print("\n" + "="*50)
    print("Performance Comparison")
    print("="*50)
    
    return_diff = results2['total_return_pct'] - results1['total_return_pct']
    sharpe_diff = results2['sharpe_ratio'] - results1['sharpe_ratio']
    dd_diff = results2['max_drawdown_pct'] - results1['max_drawdown_pct']
    trade_diff = results2['total_trades'] - results1['total_trades']
    
    print(f"Return Difference: {return_diff:+.2f}%")
    print(f"Sharpe Difference: {sharpe_diff:+.3f}")
    print(f"Drawdown Difference: {dd_diff:+.2f}%")
    print(f"Trade Count Difference: {trade_diff:+d}")
    
    # Generate plots
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('results/var_analysis', exist_ok=True)
    
    if 'var_metrics' in results2 and results2['var_metrics'].get('var_history'):
        plot_path = f'results/var_analysis/var_plot_{timestamp}.png'
        plot_var_analysis(results2, plot_path)
    
    # Save comparison report
    comparison = {
        'test_timestamp': timestamp,
        'markets': markets,
        'strategy': 'basic_momentum',
        'results_without_var': {
            'final_value': results1['final_value'],
            'total_return_pct': results1['total_return_pct'],
            'sharpe_ratio': results1['sharpe_ratio'],
            'max_drawdown_pct': results1['max_drawdown_pct'],
            'total_trades': results1['total_trades']
        },
        'results_with_var': {
            'final_value': results2['final_value'],
            'total_return_pct': results2['total_return_pct'],
            'sharpe_ratio': results2['sharpe_ratio'],
            'max_drawdown_pct': results2['max_drawdown_pct'],
            'total_trades': results2['total_trades'],
            'var_metrics': results2.get('var_metrics', {})
        },
        'improvements': {
            'return_difference': return_diff,
            'sharpe_difference': sharpe_diff,
            'drawdown_difference': dd_diff,
            'trade_difference': trade_diff,
            'risk_controlled': 'Better' if dd_diff > 0 else 'Worse'
        }
    }
    
    with open(f'results/var_analysis/comparison_{timestamp}.json', 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: results/var_analysis/")

if __name__ == "__main__":
    run_var_comparison_test()