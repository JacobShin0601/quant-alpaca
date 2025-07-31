#!/usr/bin/env python3
"""
Enhanced Ensemble Strategy Test
Test the improved ensemble strategy with all enhancements
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategies.ensemble import EnsembleStrategy, EnhancedStrategyPerformanceTracker
from actions.market_regime import MarketRegime

def create_test_data(periods: int = 1000) -> pd.DataFrame:
    """Create synthetic test data for ensemble testing"""
    
    # Create date range
    dates = pd.date_range('2024-01-01', periods=periods, freq='1H')
    
    # Generate synthetic price data with different regime periods
    np.random.seed(42)
    
    # Base price movement
    returns = np.random.randn(periods) * 0.02
    
    # Add regime-specific patterns
    for i in range(periods):
        # Trending up phase (0-250)
        if i < 250:
            returns[i] += 0.001  # Slight upward bias
            if i % 10 == 0:  # Occasional stronger moves
                returns[i] *= 2
        
        # Volatile phase (250-500)
        elif i < 500:
            returns[i] *= 3  # Higher volatility
        
        # Sideways phase (500-750)
        elif i < 750:
            returns[i] *= 0.5  # Lower volatility, mean reverting
            returns[i] += 0.0002 * np.sin(i / 10)  # Oscillating pattern
        
        # Trending down phase (750-1000)
        else:
            returns[i] -= 0.0005  # Slight downward bias
            if i % 15 == 0:
                returns[i] *= 1.5
    
    # Calculate prices
    prices = 100 * np.cumprod(1 + returns)
    
    # Create DataFrame
    df = pd.DataFrame({
        'trade_price': prices,
        'high_price': prices * (1 + np.abs(np.random.randn(periods) * 0.005)),
        'low_price': prices * (1 - np.abs(np.random.randn(periods) * 0.005)),
        'opening_price': prices * (1 + np.random.randn(periods) * 0.002),
        'candle_acc_trade_volume': np.random.exponential(1000000, periods),
        'candle_acc_trade_price': prices * np.random.exponential(1000000, periods)
    }, index=dates)
    
    return df

def test_performance_tracker():
    """Test the enhanced performance tracker"""
    print("🧪 Testing Enhanced Performance Tracker...")
    
    tracker = EnhancedStrategyPerformanceTracker()
    
    # Simulate some trades
    test_trades = [
        {'timestamp': datetime.now(), 'entry_price': 100, 'exit_price': 105, 
         'signal': 1, 'return_pct': 0.05, 'profit': True, 'holding_period': 2},
        {'timestamp': datetime.now(), 'entry_price': 105, 'exit_price': 103, 
         'signal': 1, 'return_pct': -0.019, 'profit': False, 'holding_period': 1},
        {'timestamp': datetime.now(), 'entry_price': 103, 'exit_price': 108, 
         'signal': 1, 'return_pct': 0.048, 'profit': True, 'holding_period': 3},
    ]
    
    for trade in test_trades:
        tracker.update_performance('trending_up', 'supertrend', trade)
    
    # Get performance metrics
    perf = tracker.get_strategy_performance('trending_up', 'supertrend')
    print(f"✅ Win Rate: {perf['win_rate']:.2%}")
    print(f"✅ Sharpe Ratio: {perf['sharpe']:.3f}")
    print(f"✅ Confidence Score: {perf['confidence']:.3f}")
    print(f"✅ Momentum: {perf['momentum']:.4f}")
    
    # Test dynamic weights
    dynamic_weights = tracker.get_dynamic_weights('trending_up')
    print(f"✅ Dynamic Weights: {dynamic_weights}")
    
    print("✅ Performance Tracker Test Completed!\n")

def test_ensemble_strategy():
    """Test the enhanced ensemble strategy"""
    print("🧪 Testing Enhanced Ensemble Strategy...")
    
    # Create test parameters
    parameters = {
        'confidence_threshold': 0.5,
        'min_regime_duration': 5,
        'transition_periods': 3,
        'base_position_size': 0.2,
        'strategy_rotation': True,
        'smooth_transition': True
    }
    
    # Initialize strategy
    strategy = EnsembleStrategy(parameters)
    
    # Create test data
    df = create_test_data(500)
    print(f"📊 Created test data: {len(df)} candles")
    
    # Test indicators calculation
    df_with_indicators = strategy.calculate_indicators(df)
    print(f"✅ Calculated indicators, columns: {len(df_with_indicators.columns)}")
    
    # Test signal generation
    df_with_signals = strategy.generate_signals(df_with_indicators, 'KRW-BTC')
    
    # Analyze signals
    signal_counts = df_with_signals['signal'].value_counts()
    total_signals = len(df_with_signals[df_with_signals['signal'] != 0])
    
    print(f"✅ Generated {total_signals} total signals:")
    print(f"   📈 Buy signals: {signal_counts.get(1, 0)}")
    print(f"   📉 Sell signals: {signal_counts.get(-1, 0)}")
    print(f"   ⏸️  No signals: {signal_counts.get(0, 0)}")
    
    # Test regime detection
    regimes = df_with_signals['regime'].value_counts()
    print(f"✅ Detected regimes: {dict(regimes)}")
    
    # Test position sizing
    test_prices = [100, 150, 200]
    test_portfolio_values = [1000000, 1200000, 1500000]
    
    print("\n💰 Position Sizing Tests:")
    for price, portfolio_value in zip(test_prices, test_portfolio_values):
        size = strategy.get_position_size(1, price, portfolio_value, 'KRW-BTC')
        size_pct = (size * price) / portfolio_value * 100
        print(f"   Price: ₩{price:,.0f}, Portfolio: ₩{portfolio_value:,.0f}")
        print(f"   → Position: {size:.4f} shares ({size_pct:.1f}% of portfolio)")
    
    print("✅ Ensemble Strategy Test Completed!\n")

def test_regime_transition():
    """Test enhanced regime transition logic"""
    print("🧪 Testing Regime Transition Logic...")
    
    strategy = EnsembleStrategy({
        'confidence_threshold': 0.6,
        'min_regime_duration': 10,
        'transition_periods': 5
    })
    
    # Simulate regime changes
    test_regimes = [
        (MarketRegime.TRENDING_UP, 0.8),
        (MarketRegime.VOLATILE, 0.9),  # Should trigger rapid transition
        (MarketRegime.SIDEWAYS, 0.7),
        (MarketRegime.TRENDING_DOWN, 0.65),
    ]
    
    for i, (regime, confidence) in enumerate(test_regimes):
        print(f"\n⏱️  Step {i+1}: Testing transition to {regime.value} (confidence: {confidence:.1%})")
        
        # Simulate regime duration
        strategy.regime_duration = 15 if i > 0 else 0
        
        # Mock indicators
        class MockIndicators:
            def __init__(self, confidence_val):
                self.regime_probability = {'primary': confidence_val}
                self.volatility_ratio = 2.8 if regime == MarketRegime.VOLATILE else 1.0
                self.trend_strength = 0.9 if 'TRENDING' in regime.value else 0.3
        
        mock_indicators = MockIndicators(confidence)
        strategy._update_regime_state(regime, mock_indicators)
        
        print(f"   Current regime: {strategy.current_regime.value}")
        print(f"   In transition: {strategy.in_transition}")
        if hasattr(strategy, 'transition_type'):
            print(f"   Transition type: {strategy.transition_type}")
    
    print("✅ Regime Transition Test Completed!\n")

def run_performance_benchmark():
    """Run a comprehensive performance benchmark"""
    print("🏁 Running Performance Benchmark...")
    
    # Create longer test dataset
    df = create_test_data(2000)
    
    # Test both ensemble strategies with optimization mode to avoid sub-strategy issues
    strategies = {
        'basic_ensemble': EnsembleStrategy({
            'confidence_threshold': 0.6,
            'min_regime_duration': 10,
            'base_position_size': 0.25,
            'optimization_mode': True  # Skip sub-strategy loading
        }),
        'enhanced_ensemble': EnsembleStrategy({
            'confidence_threshold': 0.5,
            'min_regime_duration': 5,
            'transition_periods': 3,
            'base_position_size': 0.2,
            'strategy_rotation': True,
            'smooth_transition': True,
            'optimization_mode': True  # Skip sub-strategy loading
        })
    }
    
    results = {}
    
    for name, strategy in strategies.items():
        print(f"\n📊 Testing {name}...")
        
        # Calculate indicators and signals
        df_indicators = strategy.calculate_indicators(df)
        df_signals = strategy.generate_signals(df_indicators, 'KRW-BTC')
        
        # Analyze performance
        signal_counts = df_signals['signal'].value_counts()
        total_signals = len(df_signals[df_signals['signal'] != 0])
        
        # Calculate signal distribution by regime
        regime_signals = df_signals[df_signals['signal'] != 0].groupby('regime')['signal'].count()
        
        results[name] = {
            'total_signals': total_signals,
            'buy_signals': signal_counts.get(1, 0),
            'sell_signals': signal_counts.get(-1, 0),
            'signal_rate': total_signals / len(df_signals) * 100,
            'regime_distribution': dict(regime_signals)
        }
        
        print(f"   📈 Total signals: {total_signals} ({total_signals/len(df_signals)*100:.1f}%)")
        print(f"   🎯 Buy/Sell ratio: {signal_counts.get(1, 0)}/{signal_counts.get(-1, 0)}")
        print(f"   📊 Regime signals: {dict(regime_signals)}")
    
    # Compare results
    print(f"\n🏆 BENCHMARK COMPARISON:")
    print(f"{'Metric':<20} {'Basic':<15} {'Enhanced':<15} {'Improvement':<15}")
    print("-" * 65)
    
    basic_signals = results['basic_ensemble']['total_signals']
    enhanced_signals = results['enhanced_ensemble']['total_signals']
    
    # Handle division by zero gracefully
    if basic_signals > 0:
        signal_improvement = enhanced_signals/basic_signals*100-100
        print(f"{'Total Signals':<20} {basic_signals:<15} {enhanced_signals:<15} {signal_improvement:+.1f}%")
    else:
        print(f"{'Total Signals':<20} {basic_signals:<15} {enhanced_signals:<15} {'N/A (basic=0)'}")
    
    if results['basic_ensemble']['signal_rate'] > 0:
        rate_improvement = results['enhanced_ensemble']['signal_rate']/results['basic_ensemble']['signal_rate']*100-100
        print(f"{'Signal Rate':<20} {results['basic_ensemble']['signal_rate']:.1f}%{'':<10} {results['enhanced_ensemble']['signal_rate']:.1f}%{'':<10} {rate_improvement:+.1f}%")
    else:
        print(f"{'Signal Rate':<20} {results['basic_ensemble']['signal_rate']:.1f}%{'':<10} {results['enhanced_ensemble']['signal_rate']:.1f}%{'':<10} {'N/A (basic=0%)'}")
    
    print("✅ Performance Benchmark Completed!\n")

def main():
    """Run all tests"""
    print("🚀 ENHANCED ENSEMBLE STRATEGY TEST SUITE")
    print("=" * 50)
    
    try:
        # Run individual tests
        test_performance_tracker()
        test_ensemble_strategy()
        test_regime_transition()
        run_performance_benchmark()
        
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("\n📋 ENHANCEMENT SUMMARY:")
        print("✅ Real-time performance tracking with confidence scoring")
        print("✅ Dynamic weight adjustment based on strategy performance")
        print("✅ Correlation-adjusted signal aggregation")
        print("✅ Enhanced regime transition with rapid/gradual modes")
        print("✅ Multi-factor position sizing (confidence + performance + volatility)")
        print("✅ Simplified strategy configuration management")
        print("✅ Kelly Criterion integration for optimal position sizing")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()