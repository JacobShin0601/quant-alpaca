#!/usr/bin/env python3
"""
Test regime performance analysis with backtesting
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

def run_regime_performance_test():
    """Run backtest with regime performance analysis"""
    
    # Load config
    with open('config/config_backtesting.json', 'r') as f:
        config = json.load(f)
    
    # Configure for a single strategy first
    config['strategy'] = {
        'name': 'basic_momentum',
        'parameters': config['strategies']['parameters']['basic_momentum']
    }
    
    # Enable regime analysis
    config['analyze_regime_performance'] = True
    config['regime_analysis'] = {
        "regime_config": {
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
            "volume_period": 20
        },
        "analysis": {
            "min_trades_per_regime": 5,
            "transition_lookback": 5,
            "performance_metrics": [
                "return", "sharpe", "win_rate", "drawdown", "holding_period"
            ]
        }
    }
    
    # Disable VaR for cleaner results
    config['var_risk_management']['enabled'] = False
    
    # Disable GARCH sizing for faster test
    config['position_sizing']['use_garch_sizing'] = False
    
    # Load data
    print("Loading market data...")
    markets = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-ADA", "KRW-DOT"]
    data = {}
    
    for market in markets:
        df = load_market_data(market, lookback_days=90)  # More data for better regime analysis
        if not df.empty:
            data[market] = df
            print(f"Loaded {len(df)} candles for {market}")
    
    if not data:
        print("No data loaded!")
        return
    
    print("\n" + "="*50)
    print("Running Ensemble Strategy with Regime Analysis")
    print("="*50)
    
    # Run backtest
    engine = BacktestEngine(config)
    results = engine.run_backtest(data)
    
    # Print overall results
    print(f"\nOverall Results:")
    print(f"Final Value: ₩{results['final_value']:,.0f}")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
    print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    
    # Check if regime analysis was performed
    if 'regime_analysis' in results:
        regime_data = results['regime_analysis']
        summary = regime_data['summary']
        
        print("\n" + "="*50)
        print("REGIME ANALYSIS SUMMARY")
        print("="*50)
        
        # Regime distribution
        print("\nRegime Distribution:")
        for regime, dist in summary['regime_distribution'].items():
            print(f"  {regime}: {dist['percentage']:.1f}% ({dist['count']} periods)")
        
        # Best/worst regimes
        print(f"\nBest Performing Regime (Sharpe): {summary.get('best_performing_regime', 'N/A')}")
        print(f"Most Profitable Regime: {summary.get('most_profitable_regime', 'N/A')}")
        print(f"Highest Win Rate Regime: {summary.get('highest_win_rate_regime', 'N/A')}")
        
        # Transition summary
        if 'transition_summary' in summary and summary['transition_summary']:
            trans_sum = summary['transition_summary']
            print(f"\nTotal Regime Transitions: {trans_sum.get('total_transitions', 0)}")
            
            if 'most_common' in trans_sum and trans_sum['most_common']:
                most_common = trans_sum['most_common']
                print(f"Most Common Transition: {most_common.from_regime.value} → {most_common.to_regime.value} ({most_common.transition_count} times)")
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs('results/regime_analysis', exist_ok=True)
        
        # Export regime analysis
        if hasattr(engine.regime_analyzer, 'export_regime_analysis'):
            export_path = f'results/regime_analysis/regime_analysis_{timestamp}.json'
            engine.regime_analyzer.export_regime_analysis(
                {
                    'regime_metrics': regime_data.get('regime_metrics', {}),
                    'transition_metrics': [],
                    'summary': summary
                },
                export_path
            )
            print(f"\nDetailed regime analysis saved to: {export_path}")
    
    # Test different strategies in different regimes
    print("\n" + "="*50)
    print("Testing Individual Strategies by Regime")
    print("="*50)
    
    strategies_to_test = ['basic_momentum', 'vwap', 'macd', 'mean_reversion']
    regime_results = {}
    
    for strategy_name in strategies_to_test:
        print(f"\nTesting {strategy_name}...")
        
        # Configure for single strategy
        config['strategy'] = {
            'name': strategy_name,
            'parameters': config['strategies']['parameters'].get(strategy_name, {})
        }
        
        # Run backtest
        engine = BacktestEngine(config)
        results = engine.run_backtest(data)
        
        if 'regime_analysis' in results:
            regime_results[strategy_name] = results['regime_analysis']
            
    # Compare strategies across regimes
    if regime_results:
        print("\n" + "="*80)
        print("STRATEGY COMPARISON BY REGIME")
        print("="*80)
        
        # Create comparison table
        comparison_data = []
        
        for strategy_name, regime_data in regime_results.items():
            if 'regime_metrics' in regime_data:
                for regime_name, metrics in regime_data['regime_metrics'].items():
                    if metrics:
                        comparison_data.append({
                            'Strategy': strategy_name,
                            'Regime': regime_name,
                            'Win Rate': f"{metrics['win_rate']:.1%}",
                            'Total Return': f"{metrics['total_return']:.2%}",
                            'Sharpe': f"{metrics['sharpe_ratio']:.2f}",
                            'Trades': metrics['total_trades']
                        })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.pivot_table(
                index='Strategy',
                columns='Regime',
                values=['Win Rate', 'Total Return', 'Sharpe', 'Trades'],
                aggfunc='first'
            )
            
            print("\nWin Rate by Strategy and Regime:")
            print(comparison_df['Win Rate'].to_string())
            
            print("\nTotal Return by Strategy and Regime:")
            print(comparison_df['Total Return'].to_string())
            
            print("\nSharpe Ratio by Strategy and Regime:")
            print(comparison_df['Sharpe'].to_string())

if __name__ == "__main__":
    run_regime_performance_test()