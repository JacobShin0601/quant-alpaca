#!/usr/bin/env python3
"""
Test script to run improved backtesting with fixed issues
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from actions.backtest import MultiStrategyBacktester

def main():
    print("Testing improved backtesting engine...")
    print("=" * 60)
    print("Key improvements:")
    print("1. Fixed position sizing - now uses 15% of initial balance per position")
    print("2. Fixed Sharpe ratio calculation for minute-level data")
    print("3. Added minimum 30-minute interval between trades")
    print("4. Removed incorrect 'markets' parameter from run_backtest")
    print("=" * 60)
    
    # Create backtester
    backtester = MultiStrategyBacktester('config/config_backtesting.json')
    
    # Run backtest with a single strategy first to test
    success = backtester.run_backtest(
        strategies=['vwap'],  # Test with VWAP strategy
        use_cached_data=True,  # Use cached data
        data_only=False
    )
    
    if success:
        print("\n✓ Test backtest completed successfully!")
        print("Check the results/ directory for output files")
    else:
        print("\n✗ Test backtest failed")
        sys.exit(1)

if __name__ == "__main__":
    main()