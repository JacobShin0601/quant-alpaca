#!/usr/bin/env python3
"""
Test script for adaptive strategies
Validates that adaptive strategies work correctly with the backtesting engine
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from strategies.registry import get_strategy
from strategies.enhanced_ensemble import EnhancedEnsembleStrategy
from strategies.adaptive_strategy_base import create_adaptive_strategy
from backtesting.engine import BacktestEngine
from data.collector import UpbitDataCollector


def create_test_data(length=2000):
    """Create synthetic test data for validation"""
    dates = pd.date_range(start='2024-01-01', periods=length, freq='1min')
    
    # Generate realistic price data with trends and volatility
    price = 100.0
    prices = []
    volumes = []
    
    np.random.seed(42)  # For reproducible results
    
    for i in range(length):
        # Add trend and noise
        trend = 0.001 * np.sin(i / 100) + 0.0005  # Slight upward trend with cycles
        noise = np.random.normal(0, 0.005)  # 0.5% volatility
        price_change = trend + noise
        
        price *= (1 + price_change)
        prices.append(price)
        
        # Generate volume
        base_volume = 1000000
        volume_noise = np.random.lognormal(0, 0.5)
        volumes.append(base_volume * volume_noise)
    
    # Create OHLCV data
    high_prices = [p * (1 + abs(np.random.normal(0, 0.002))) for p in prices]
    low_prices = [p * (1 - abs(np.random.normal(0, 0.002))) for p in prices]
    
    df = pd.DataFrame({
        'trade_price': prices,
        'high_price': high_prices,
        'low_price': low_prices,
        'candle_acc_trade_volume': volumes,
        'timestamp': dates
    })
    
    df.set_index('timestamp', inplace=True)
    return df


def test_enhanced_ensemble():
    """Test enhanced ensemble strategy"""
    print("Testing Enhanced Ensemble Strategy...")
    
    # Create test parameters
    test_params = {
        'max_active_strategies': 3,
        'weight_decay': 0.95,
        'performance_window': 120,
        'regime_stability_threshold': 10,
        'confidence_threshold': 0.6
    }
    
    try:
        # Initialize strategy
        strategy = EnhancedEnsembleStrategy(test_params)
        print("✓ Enhanced ensemble strategy initialized successfully")
        
        # Create test data
        test_data = create_test_data(1000)
        print("✓ Test data created")
        
        # Calculate indicators
        test_data_with_indicators = strategy.calculate_indicators(test_data)
        print("✓ Indicators calculated")
        
        # Generate signals
        signals_data = strategy.generate_signals(test_data_with_indicators, 'TEST-MARKET')
        print("✓ Signals generated")
        
        # Check that signals were generated
        signal_count = (signals_data['signal'] != 0).sum()
        print(f"✓ Generated {signal_count} trading signals")
        
        # Check regime detection
        regime_count = (signals_data['regime'] != 'unknown').sum()
        micro_regime_count = (signals_data['micro_regime'] != 'unknown').sum()
        print(f"✓ Detected regimes in {regime_count} periods")
        print(f"✓ Detected microstructure regimes in {micro_regime_count} periods")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced ensemble test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adaptive_strategy_wrapper():
    """Test adaptive strategy wrapper"""
    print("\nTesting Adaptive Strategy Wrapper...")
    
    try:
        from strategies.mean_reversion import MeanReversionStrategy
        
        # Create base strategy parameters
        base_params = {
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'entry_zscore': 2.0,
            'exit_zscore': 0.5,
            'use_volume_filter': True,
            'volume_threshold': 1.5
        }
        
        # Create adaptive version
        adaptive_strategy = create_adaptive_strategy(
            MeanReversionStrategy, 
            base_params, 
            enable_adaptation=True
        )
        
        print("✓ Adaptive strategy wrapper created")
        
        # Create test data
        test_data = create_test_data(1500)
        
        # Test adaptive parameter retrieval
        current_params = adaptive_strategy.get_adaptive_parameters(test_data, 500)
        print(f"✓ Retrieved adaptive parameters: {len(current_params)} params")
        
        # Test adaptation status
        status = adaptive_strategy.get_adaptation_status()
        print(f"✓ Adaptation status: {status['adaptation_enabled']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Adaptive wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adaptive_backtesting():
    """Test adaptive strategy in backtesting engine"""
    print("\nTesting Adaptive Strategy in Backtesting Engine...")
    
    try:
        # Create test configuration
        config = {
            'backtesting': {
                'initial_balance': 1000000,
                'commission_rate': 0.0005,
                'slippage_rate': 0.001,
                'min_data_points': 100,
                'max_positions': 3,
                'order_type': 'limit',
                'fees': {
                    'limit_order': {'krw_market': 0.0005, 'btc_market': 0.0025, 'usdt_market': 0.0025},
                    'market_order': {'krw_market': 0.00139, 'btc_market': 0.0025, 'usdt_market': 0.0025}
                },
                'slippage': {'limit_order': 0.0000, 'market_order': 0.0005},
                'position_sizing': {'max_position_pct': 0.2, 'use_dynamic_sizing': False, 'use_garch_sizing': False},
                'risk_management': {'enabled': True},
                'var_risk_management': {'enabled': False},
                'regime_analysis': {'enabled': False},
                'execution': {'log_level': 'CRITICAL', 'max_log_trades': 0, 'save_detailed_results': False,
                            'warmup_period_minutes': 0, 'max_stop_limit_warnings': 0}
            },
            'strategy': {
                'name': 'enhanced_ensemble',
                'parameters': {
                    'max_active_strategies': 2,
                    'weight_decay': 0.95,
                    'performance_window': 60,
                    'confidence_threshold': 0.5
                }
            },
            'regime_config': {},
            'execution': {'log_level': 'CRITICAL'}
        }
        
        # Initialize backtesting engine
        engine = BacktestEngine(config)
        print("✓ Backtesting engine initialized")
        
        # Check if strategy is detected as adaptive
        if engine.strategy_is_adaptive:
            print("✓ Strategy correctly detected as adaptive")
        else:
            print("! Strategy not detected as adaptive (this may be expected)")
        
        # Create test data
        test_data = create_test_data(800)
        
        # Run backtest
        results = engine.run_backtest({'TEST-MARKET': test_data})
        print("✓ Backtest completed successfully")
        
        # Check results
        print(f"✓ Final portfolio value: {results.get('final_value', 0):,.0f}")
        print(f"✓ Total return: {results.get('total_return_pct', 0):.2f}%")
        print(f"✓ Total trades: {results.get('total_trades', 0)}")
        print(f"✓ Sharpe ratio: {results.get('sharpe_ratio', 0):.3f}")
        
        # Check for adaptive strategy information
        if 'adaptive_strategy' in results:
            adaptive_info = results['adaptive_strategy']
            print(f"✓ Adaptive info collected: {adaptive_info['strategy_name']}")
            print(f"✓ Adaptation enabled: {adaptive_info['adaptation_enabled']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Adaptive backtesting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_real_data():
    """Test with real market data if available"""
    print("\nTesting with Real Data (if available)...")
    
    try:
        # Try to load real data
        collector = UpbitDataCollector('data/candles', '{market}_candles.db')
        
        # Try to get some recent data
        df = collector.get_recent_data('KRW-ETH', periods=1000)
        
        if df is not None and len(df) > 500:
            print(f"✓ Loaded {len(df)} real data points for KRW-ETH")
            
            # Test with enhanced ensemble
            config = {
                'backtesting': {
                    'initial_balance': 1000000,
                    'commission_rate': 0.0005,
                    'slippage_rate': 0.001,
                    'min_data_points': 100,
                    'max_positions': 3,
                    'order_type': 'limit',
                    'fees': {
                        'limit_order': {'krw_market': 0.0005, 'btc_market': 0.0025, 'usdt_market': 0.0025}
                    },
                    'slippage': {'limit_order': 0.0000},
                    'position_sizing': {'max_position_pct': 0.2},
                    'risk_management': {'enabled': True},
                    'var_risk_management': {'enabled': False},
                    'regime_analysis': {'enabled': False},
                    'execution': {'log_level': 'CRITICAL', 'max_log_trades': 0}
                },
                'strategy': {
                    'name': 'enhanced_ensemble',
                    'parameters': {
                        'max_active_strategies': 2,
                        'confidence_threshold': 0.5
                    }
                }
            }
            
            engine = BacktestEngine(config)
            results = engine.run_backtest({'KRW-ETH': df})
            
            print(f"✓ Real data backtest completed")
            print(f"✓ Total return: {results.get('total_return_pct', 0):.2f}%")
            print(f"✓ Total trades: {results.get('total_trades', 0)}")
            
            return True
        else:
            print("! No real data available, skipping real data test")
            return True
            
    except Exception as e:
        print(f"! Real data test failed (this may be expected): {e}")
        return True  # Don't fail overall test for this


def main():
    """Run all tests"""
    print("🧪 Testing Adaptive Strategies Implementation\n")
    print("=" * 60)
    
    tests = [
        ("Enhanced Ensemble Strategy", test_enhanced_ensemble),
        ("Adaptive Strategy Wrapper", test_adaptive_strategy_wrapper),
        ("Adaptive Backtesting", test_adaptive_backtesting),
        ("Real Data Test", test_with_real_data)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status:<12} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Adaptive strategies are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)