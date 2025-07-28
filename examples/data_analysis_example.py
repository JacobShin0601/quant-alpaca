#!/usr/bin/env python3

"""
Data Analysis Example for Quant-Alpaca

This example shows how to load and analyze data from the SQLite databases.
"""

import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.collector import UpbitDataCollector


def load_config():
    """Load configuration"""
    config_path = "config/config_backtesting.json"
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    # Load configuration
    config = load_config()
    
    # Initialize data collector
    collector = UpbitDataCollector(
        database_directory=config['data']['database_directory'],
        database_pattern=config['data']['database_pattern']
    )
    
    # Example 1: List available data
    print("=== Available Data ===")
    available_data = collector.list_available_data()
    for data_info in available_data:
        print(f"{data_info['market']}: {data_info['total_candles']:,} candles")
    
    # Example 2: Load single market data
    print("\n=== Loading BTC Data ===")
    btc_data = collector.get_single_market_data('KRW-BTC')
    if not btc_data.empty:
        print(f"Loaded {len(btc_data)} BTC candles")
        print(f"Price range: ₩{btc_data['trade_price'].min():,.0f} - ₩{btc_data['trade_price'].max():,.0f}")
        print(f"Date range: {btc_data.index.min()} to {btc_data.index.max()}")
    
    # Example 3: Get data summary
    print("\n=== BTC Summary ===")
    btc_summary = collector.get_data_summary('KRW-BTC')
    if btc_summary:
        print(f"Total candles: {btc_summary['total_candles']:,}")
        print(f"Price change: {btc_summary['price_stats']['price_change']:+.2f}%")
    
    # Example 4: Load data for specific date range
    print("\n=== Loading Recent Data ===")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)  # Last 7 days
    
    recent_data = collector.get_single_market_data(
        'KRW-BTC',
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat()
    )
    
    if not recent_data.empty:
        print(f"Loaded {len(recent_data)} recent BTC candles")
        print(f"Recent price change: {((recent_data['trade_price'].iloc[-1] - recent_data['trade_price'].iloc[0]) / recent_data['trade_price'].iloc[0] * 100):+.2f}%")
    
    # Example 5: Basic analysis
    if not btc_data.empty:
        print("\n=== Basic Analysis ===")
        
        # Calculate simple moving averages
        btc_data['ma_20'] = btc_data['trade_price'].rolling(window=20).mean()
        btc_data['ma_50'] = btc_data['trade_price'].rolling(window=50).mean()
        
        # Calculate RSI
        delta = btc_data['trade_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        btc_data['rsi'] = 100 - (100 / (1 + rs))
        
        print(f"Current RSI: {btc_data['rsi'].iloc[-1]:.2f}")
        print(f"Current MA20: ₩{btc_data['ma_20'].iloc[-1]:,.0f}")
        print(f"Current MA50: ₩{btc_data['ma_50'].iloc[-1]:,.0f}")
        
        # Find trend
        if btc_data['ma_20'].iloc[-1] > btc_data['ma_50'].iloc[-1]:
            trend = "Uptrend (MA20 > MA50)"
        else:
            trend = "Downtrend (MA20 < MA50)"
        print(f"Trend: {trend}")
    
    # Example 6: Export to CSV
    print("\n=== Exporting Data ===")
    try:
        collector.export_to_csv('KRW-BTC', 'exports/btc_data.csv')
        print("BTC data exported to exports/btc_data.csv")
    except Exception as e:
        print(f"Export failed: {e}")


if __name__ == "__main__":
    main()