#!/usr/bin/env python3
"""
Simple test for regime performance analysis
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(__file__))

def test_regime_analysis():
    """Test regime analysis with sample data"""
    
    # Create sample results for testing
    from src.actions.regime_performance_analyzer import RegimePerformanceAnalyzer, RegimePerformanceMetrics, MarketRegime
    from datetime import timedelta
    
    # Create analyzer
    analyzer = RegimePerformanceAnalyzer()
    
    # Create sample regime metrics
    sample_metrics = {
        'trending_up': RegimePerformanceMetrics(
            regime=MarketRegime.TRENDING_UP,
            total_trades=150,
            winning_trades=90,
            losing_trades=60,
            win_rate=0.60,
            total_return=0.25,
            average_return=0.00167,
            total_pnl=750000,
            average_pnl=5000,
            max_win=50000,
            max_loss=-20000,
            sharpe_ratio=1.85,
            sortino_ratio=2.20,
            max_drawdown=-0.08,
            avg_holding_period=timedelta(hours=2, minutes=30),
            total_duration=timedelta(days=15),
            regime_percentage=35.5
        ),
        'trending_down': RegimePerformanceMetrics(
            regime=MarketRegime.TRENDING_DOWN,
            total_trades=80,
            winning_trades=35,
            losing_trades=45,
            win_rate=0.4375,
            total_return=-0.05,
            average_return=-0.000625,
            total_pnl=-150000,
            average_pnl=-1875,
            max_win=25000,
            max_loss=-35000,
            sharpe_ratio=-0.45,
            sortino_ratio=-0.30,
            max_drawdown=-0.15,
            avg_holding_period=timedelta(hours=1, minutes=45),
            total_duration=timedelta(days=8),
            regime_percentage=18.2
        ),
        'sideways': RegimePerformanceMetrics(
            regime=MarketRegime.SIDEWAYS,
            total_trades=120,
            winning_trades=65,
            losing_trades=55,
            win_rate=0.542,
            total_return=0.12,
            average_return=0.001,
            total_pnl=360000,
            average_pnl=3000,
            max_win=30000,
            max_loss=-15000,
            sharpe_ratio=1.20,
            sortino_ratio=1.55,
            max_drawdown=-0.06,
            avg_holding_period=timedelta(hours=3, minutes=15),
            total_duration=timedelta(days=12),
            regime_percentage=28.3
        ),
        'volatile': RegimePerformanceMetrics(
            regime=MarketRegime.VOLATILE,
            total_trades=50,
            winning_trades=20,
            losing_trades=30,
            win_rate=0.40,
            total_return=-0.08,
            average_return=-0.0016,
            total_pnl=-240000,
            average_pnl=-4800,
            max_win=40000,
            max_loss=-45000,
            sharpe_ratio=-0.80,
            sortino_ratio=-0.60,
            max_drawdown=-0.20,
            avg_holding_period=timedelta(hours=1, minutes=30),
            total_duration=timedelta(days=6),
            regime_percentage=18.0
        )
    }
    
    # Create analysis results
    analysis_results = {
        'regime_metrics': sample_metrics,
        'transition_metrics': [],
        'summary': {
            'total_regimes_analyzed': 4,
            'regime_distribution': {
                'trending_up': {'count': 1550, 'percentage': 35.5},
                'trending_down': {'count': 795, 'percentage': 18.2},
                'sideways': {'count': 1235, 'percentage': 28.3},
                'volatile': {'count': 786, 'percentage': 18.0}
            },
            'best_performing_regime': 'trending_up',
            'worst_performing_regime': 'volatile',
            'most_profitable_regime': 'trending_up',
            'highest_win_rate_regime': 'trending_up'
        }
    }
    
    # Create performance table
    table = analyzer.create_regime_performance_table(analysis_results)
    
    print("="*80)
    print("PERFORMANCE BY MARKET REGIME")
    print("="*80)
    print(table.to_string(index=False))
    print("="*80)
    
    # Print summary statistics
    print("\nREGIME SUMMARY:")
    print("-"*40)
    for regime_name, metrics in sample_metrics.items():
        print(f"\n{regime_name.upper()}:")
        print(f"  Time in Regime: {metrics.regime_percentage:.1f}%")
        print(f"  Win Rate: {metrics.win_rate:.1%}")
        print(f"  Total Return: {metrics.total_return:.2%}")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('results/regime_analysis', exist_ok=True)
    
    # Export
    analyzer.export_regime_analysis(
        analysis_results,
        f'results/regime_analysis/sample_regime_{timestamp}.json'
    )
    
    print(f"\nResults saved to: results/regime_analysis/sample_regime_{timestamp}.json")

if __name__ == "__main__":
    test_regime_analysis()