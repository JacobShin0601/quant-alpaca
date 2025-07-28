#!/usr/bin/env python3

"""
Data Reader Utility for Quant-Alpaca

This script provides easy ways to read and export data from the SQLite databases.

Usage examples:
    python src/utils/data_reader.py --list                           # List all available data
    python src/utils/data_reader.py --market KRW-BTC --summary      # Get summary for BTC
    python src/utils/data_reader.py --market KRW-BTC --export-csv   # Export BTC to CSV
    python src/utils/data_reader.py --export-all-csv                # Export all to CSV
"""

import sys
import os
import argparse
import json
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.collector import UpbitDataCollector


def main():
    parser = argparse.ArgumentParser(description='Data Reader Utility for Quant-Alpaca')
    
    # Data source options
    parser.add_argument('--config', default='config/config_backtesting.json',
                        help='Configuration file path')
    
    # Actions
    parser.add_argument('--list', action='store_true',
                        help='List all available data files')
    parser.add_argument('--summary', action='store_true',
                        help='Show summary statistics for a market')
    parser.add_argument('--export-csv', action='store_true',
                        help='Export market data to CSV')
    parser.add_argument('--export-all-csv', action='store_true',
                        help='Export all markets to CSV files')
    
    # Market selection
    parser.add_argument('--market', type=str,
                        help='Market symbol (e.g., KRW-BTC)')
    
    # Date filtering
    parser.add_argument('--start-date', type=str,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                        help='End date (YYYY-MM-DD)')
    
    # Output options
    parser.add_argument('--output-dir', default='exports',
                        help='Output directory for exports')
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found at {args.config}")
        sys.exit(1)
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Initialize data collector
    collector = UpbitDataCollector(
        database_directory=config['data']['database_directory'],
        database_pattern=config['data']['database_pattern']
    )
    
    # Execute requested action
    if args.list:
        list_available_data(collector)
    
    elif args.summary:
        if not args.market:
            print("Error: --market is required for summary")
            sys.exit(1)
        show_market_summary(collector, args.market)
    
    elif args.export_csv:
        if not args.market:
            print("Error: --market is required for CSV export")
            sys.exit(1)
        export_market_csv(collector, args.market, args.output_dir, args.start_date, args.end_date)
    
    elif args.export_all_csv:
        export_all_csv(collector, config['data']['markets'], args.output_dir, args.start_date, args.end_date)
    
    else:
        parser.print_help()


def list_available_data(collector):
    """List all available data files"""
    available_data = collector.list_available_data()
    
    if not available_data:
        print("No data files found.")
        return
    
    print("Available Data Files:")
    print("=" * 80)
    
    for data_info in available_data:
        print(f"\nMarket: {data_info['market']}")
        print(f"  File: {data_info['filename']} ({data_info['file_size_mb']} MB)")
        
        if 'collection_period' in data_info:
            print(f"  Period: {data_info['collection_period']['start']} to {data_info['collection_period']['end']}")
            print(f"  Candles: {data_info['total_candles']:,}")
            print(f"  Lookback Days: {data_info['lookback_days']}")
        
        print(f"  Last Modified: {data_info['last_modified']}")
    
    print("\n" + "=" * 80)


def show_market_summary(collector, market):
    """Show summary statistics for a market"""
    summary = collector.get_data_summary(market)
    
    if not summary:
        print(f"No data found for market: {market}")
        return
    
    print(f"Summary for {market}:")
    print("=" * 50)
    print(f"Total Candles: {summary['total_candles']:,}")
    print(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"Price Range: ₩{summary['price_stats']['min_price']:,.0f} - ₩{summary['price_stats']['max_price']:,.0f}")
    print(f"Average Price: ₩{summary['price_stats']['avg_price']:,.0f}")
    print(f"Price Change: {summary['price_stats']['price_change']:+.2f}%")
    print(f"Total Volume: {summary['volume_stats']['total_volume']:,.2f}")
    print(f"Average Volume: {summary['volume_stats']['avg_volume']:,.2f}")
    print("=" * 50)


def export_market_csv(collector, market, output_dir, start_date, end_date):
    """Export single market to CSV"""
    safe_market = market.replace('-', '_')
    filename = f"{safe_market}_data.csv"
    output_path = os.path.join(output_dir, filename)
    
    collector.export_to_csv(market, output_path, start_date, end_date)


def export_all_csv(collector, markets, output_dir, start_date, end_date):
    """Export all markets to CSV"""
    print(f"Exporting {len(markets)} markets to CSV...")
    collector.export_all_to_csv(markets, output_dir, start_date, end_date)
    print(f"All exports completed. Check {output_dir}/ directory.")


if __name__ == "__main__":
    main()