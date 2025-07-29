#!/usr/bin/env python3
"""Test script for ensemble strategy"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import required modules
from actions.market_regime import MarketRegimeDetector, MarketRegime
from actions.ensemble_strategy import EnsembleStrategy

def create_test_data():
    """Create sample market data for testing"""
    dates = pd.date_range(start='2025-07-01', end='2025-07-27', freq='T')
    
    # Create synthetic price data with different regimes
    n = len(dates)
    
    # Base price
    base_price = 150000000  # 150M KRW (BTC-like)
    
    # Create different market regimes
    prices = []
    volumes = []
    
    for i in range(n):
        # First week - uptrend
        if i < n // 4:
            trend = 0.00001 * i
            volatility = 0.001
        # Second week - downtrend  
        elif i < n // 2:
            trend = -0.00001 * (i - n//4)
            volatility = 0.0015
        # Third week - sideways
        elif i < 3 * n // 4:
            trend = 0
            volatility = 0.0008
        # Fourth week - volatile
        else:
            trend = 0
            volatility = 0.003
        
        # Generate price with trend and noise
        noise = np.random.normal(0, volatility)
        price = base_price * (1 + trend + noise)
        prices.append(price)
        
        # Generate volume
        base_volume = 100
        volume = base_volume * (1 + abs(noise) * 10) * np.random.uniform(0.8, 1.2)
        volumes.append(volume)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.001 for p in prices],
        'low': [p * 0.999 for p in prices],
        'trade_price': prices,
        'candle_acc_trade_volume': volumes,
        'high_price': [p * 1.001 for p in prices],
        'low_price': [p * 0.999 for p in prices],
        'opening_price': prices
    })
    
    df.set_index('timestamp', inplace=True)
    
    return df

def test_regime_detection():
    """Test market regime detection"""
    print("=== Testing Market Regime Detection ===\n")
    
    # Create test data
    df = create_test_data()
    
    # Initialize regime detector
    detector = MarketRegimeDetector()
    
    # Test regime detection at different points
    test_points = [len(df)//8, len(df)//4, len(df)//2, 3*len(df)//4, len(df)-1]
    
    for point in test_points:
        regime, indicators = detector.detect_regime(df.iloc[:point])
        
        print(f"At index {point} ({df.index[point-1]}):")
        print(f"  Regime: {regime.value}")
        print(f"  ADX: {indicators.adx:.2f}")
        print(f"  MA Alignment: {indicators.ma_alignment_score:.2f}")
        print(f"  Volatility Ratio: {indicators.volatility_ratio:.2f}")
        print(f"  Choppiness: {indicators.choppiness_index:.2f}")
        print(f"  Regime Probabilities: {indicators.regime_probability}")
        print()

def test_ensemble_strategy():
    """Test ensemble strategy"""
    print("\n=== Testing Ensemble Strategy ===\n")
    
    # Create test data
    df = create_test_data()
    
    # Initialize ensemble strategy
    config = {
        'config_path': 'config/strategies/ensemble_config.json',
        'regime_config': {
            'adx_period': 14,
            'ma_periods': [10, 20, 50],
            'bb_period': 20,
            'bb_std': 2,
            'atr_period': 14,
            'volume_period': 20,
            'choppiness_period': 14,
            'lookback_period': 100,
            'adx_trend_threshold': 25,
            'adx_strong_threshold': 40,
            'volatility_high_threshold': 2.0,
            'volatility_low_threshold': 0.5,
            'volume_spike_threshold': 2.0,
            'choppiness_sideways_threshold': 61.8
        }
    }
    
    strategy = EnsembleStrategy(config)
    
    # Test on different segments
    segments = [
        ("Uptrend", 0, len(df)//4),
        ("Downtrend", len(df)//4, len(df)//2),
        ("Sideways", len(df)//2, 3*len(df)//4),
        ("Volatile", 3*len(df)//4, len(df))
    ]
    
    for name, start, end in segments:
        segment_df = df.iloc[max(0, start-100):end].copy()  # Include lookback
        
        # Calculate indicators
        segment_df = strategy.calculate_indicators(segment_df)
        
        # Generate signals
        segment_df = strategy.generate_signals(segment_df, 'KRW-BTC')
        
        # Get last few rows
        last_rows = segment_df.tail(5)
        
        print(f"\n{name} Market Segment:")
        print(f"  Current Regime: {strategy.current_regime.value}")
        print(f"  Regime Confidence: {strategy.regime_confidence:.2%}")
        print(f"  Active Strategies: {list(strategy.active_strategies.keys())}")
        print(f"  Last 5 signals: {last_rows['signal'].tolist()}")
        
        if strategy.signal_history:
            last_signal = strategy.signal_history[-1]
            print(f"  Last Signal Details:")
            print(f"    - Strength: {last_signal['strength']:.2f}")
            print(f"    - Source: {last_signal['source']}")
            print(f"    - Strategies: {last_signal['strategies']}")

def main():
    """Run all tests"""
    print("Testing Ensemble Strategy System")
    print("="*50)
    
    test_regime_detection()
    test_ensemble_strategy()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main()