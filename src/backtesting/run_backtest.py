#!/usr/bin/env python3
"""
Legacy backtest runner - for backward compatibility
Redirects to the new multi-strategy backtesting system
"""

import sys
import os
import json
import argparse
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import new multi-strategy backtester
from actions.backtest import MultiStrategyBacktester


def main():
    """Legacy main function - redirects to new multi-strategy system"""
    print("=" * 60)
    print("⚠️  LEGACY BACKTEST RUNNER")
    print("=" * 60)
    print("This script is deprecated. Please use the new multi-strategy system:")
    print("  python src/actions/backtest.py [OPTIONS]")
    print("  ./run_backtesting.sh [OPTIONS]")
    print("")
    print("For compatibility, redirecting to new system...")
    print("=" * 60)
    
    # Parse legacy arguments
    parser = argparse.ArgumentParser(description='Legacy backtest runner (deprecated)')
    parser.add_argument('--use-cached-data', action='store_true', 
                        help='Skip data collection, use existing cached data')
    parser.add_argument('--data-only', action='store_true',
                        help='Only collect data, skip backtesting')
    parser.add_argument('--config', default='config/config_backtesting.json',
                        help='Configuration file path')
    args = parser.parse_args()
    
    try:
        # Use new multi-strategy backtester
        backtester = MultiStrategyBacktester(args.config)
        
        # Run with default strategy (basic_momentum for compatibility)
        success = backtester.run_backtest(
            strategies=['basic_momentum'],  # Use first available strategy
            use_cached_data=args.use_cached_data,
            data_only=args.data_only
        )
        
        if success:
            print("\n✓ Legacy backtest completed successfully")
            print("💡 Consider migrating to the new multi-strategy system for more features")
        else:
            print("\n❌ Legacy backtest failed")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error running legacy backtest: {e}")
        print("Please use the new system: python src/actions/backtest.py")
        sys.exit(1)


if __name__ == "__main__":
    main()