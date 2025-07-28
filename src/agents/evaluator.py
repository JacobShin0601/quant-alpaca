#!/usr/bin/env python3
"""
Strategy Evaluation and Comparison System
Evaluates and compares backtesting results from multiple strategies
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from actions.backtest import MultiStrategyBacktester


class StrategyEvaluator:
    """Evaluate and compare strategy performance"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def load_backtest_results(self, results_file: str) -> List[Dict[str, Any]]:
        """Load backtest results from JSON file"""
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f" Loaded results from: {results_file}")
        return results
    
    def create_performance_summary(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create performance summary DataFrame"""
        summary_data = []
        
        for result in results:
            strategy = result['strategy']
            overall = result['overall']
            
            summary_data.append({
                'Strategy': strategy,
                'Total_Return_%': round(overall['total_return'] * 100, 2),
                'Total_Trades': overall['total_trades'],
                'Winning_Trades': overall['winning_trades'],
                'Win_Rate_%': round(overall['win_rate'], 2),
                'Avg_Return_Per_Trade_%': round(overall['average_return_per_trade'] * 100, 4),
                'Markets_Tested': overall['markets_tested']
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('Total_Return_%', ascending=False)
        
        return df
    
    def create_market_performance_matrix(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create market-wise performance matrix"""
        strategies = [r['strategy'] for r in results]
        all_markets = set()
        
        # Get all markets
        for result in results:
            all_markets.update(result['markets'].keys())
        
        all_markets = sorted(list(all_markets))
        
        # Create matrix
        performance_matrix = []
        
        for market in all_markets:
            row = {'Market': market}
            for result in results:
                strategy = result['strategy']
                if market in result['markets']:
                    market_perf = result['markets'][market].get('performance', {})
                    total_return = market_perf.get('total_return', 0) * 100
                    row[strategy] = round(total_return, 2)
                else:
                    row[strategy] = 0
            performance_matrix.append(row)
        
        return pd.DataFrame(performance_matrix)
    
    def create_detailed_market_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """Create detailed analysis for each market with top 5 strategies"""
        # Get all markets
        all_markets = set()
        for result in results:
            all_markets.update(result['markets'].keys())
        
        market_analysis = {}
        
        for market in sorted(all_markets):
            market_data = []
            
            for result in results:
                strategy = result['strategy']
                if market in result['markets'] and 'performance' in result['markets'][market]:
                    perf = result['markets'][market]['performance']
                    
                    # Calculate comprehensive metrics
                    total_trades = perf.get('total_trades', 0)
                    winning_trades = perf.get('winning_trades', 0)
                    losing_trades = perf.get('losing_trades', 0)
                    total_return = perf.get('total_return', 0)
                    total_fees = perf.get('total_fees', 0)
                    total_profit = perf.get('total_profit_amount', 0)
                    total_loss = perf.get('total_loss_amount', 0)
                    
                    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                    profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
                    avg_return = perf.get('average_return_per_trade', 0)
                    
                    market_data.append({
                        'Strategy': strategy,
                        'Total_Return_%': round(total_return * 100, 4),
                        'Total_Trades': total_trades,
                        'Winning_Trades': winning_trades,
                        'Losing_Trades': losing_trades,
                        'Win_Rate_%': round(win_rate, 2),
                        'Profit_Amount_KRW': f"₩{total_profit:,.0f}",
                        'Loss_Amount_KRW': f"₩{abs(total_loss):,.0f}",
                        'Net_Profit_KRW': f"₩{total_profit + total_loss:,.0f}",
                        'Total_Fees_KRW': f"₩{total_fees:,.0f}",
                        'Profit_Factor': round(profit_factor, 3),
                        'Avg_Return_Per_Trade_%': round(avg_return * 100, 4),
                        'Return_After_Fees_%': round((total_return - (total_fees / 1000000)) * 100, 4)  # Assuming base amount
                    })
            
            # Sort by total return and take top 5
            market_data.sort(key=lambda x: x['Total_Return_%'], reverse=True)
            top_5_data = market_data[:5]
            
            market_analysis[market] = pd.DataFrame(top_5_data)
        
        return market_analysis
    
    def calculate_risk_metrics(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Calculate risk metrics for each strategy"""
        risk_data = []
        
        for result in results:
            strategy = result['strategy']
            
            # Collect all trade returns across markets
            all_returns = []
            for market_result in result['markets'].values():
                if 'performance' in market_result and 'trade_returns' in market_result['performance']:
                    all_returns.extend(market_result['performance']['trade_returns'])
            
            if not all_returns:
                continue
            
            returns_array = np.array(all_returns)
            
            # Calculate risk metrics
            volatility = np.std(returns_array) * 100
            max_drawdown = self._calculate_max_drawdown(returns_array) * 100
            sharpe_ratio = np.mean(returns_array) / np.std(returns_array) if np.std(returns_array) > 0 else 0
            var_95 = np.percentile(returns_array, 5) * 100  # Value at Risk (95%)
            
            risk_data.append({
                'Strategy': strategy,
                'Volatility_%': round(volatility, 2),
                'Max_Drawdown_%': round(max_drawdown, 2),
                'Sharpe_Ratio': round(sharpe_ratio, 3),
                'VaR_95_%': round(var_95, 2),
                'Total_Trades': len(returns_array)
            })
        
        return pd.DataFrame(risk_data)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
    
    def create_ranking_table(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create comprehensive ranking table"""
        performance_df = self.create_performance_summary(results)
        risk_df = self.calculate_risk_metrics(results)
        
        # Merge dataframes
        ranking_df = performance_df.merge(risk_df, on='Strategy', how='outer')
        
        # Calculate composite score
        # Normalize metrics (higher is better for returns, sharpe; lower is better for volatility, drawdown)
        ranking_df['Return_Score'] = self._normalize_score(ranking_df['Total_Return_%'], higher_better=True)
        ranking_df['Sharpe_Score'] = self._normalize_score(ranking_df['Sharpe_Ratio'], higher_better=True)
        ranking_df['Vol_Score'] = self._normalize_score(ranking_df['Volatility_%'], higher_better=False)
        ranking_df['DD_Score'] = self._normalize_score(ranking_df['Max_Drawdown_%'], higher_better=False)
        
        # Composite score (weighted average)
        ranking_df['Composite_Score'] = (
            0.4 * ranking_df['Return_Score'] +
            0.3 * ranking_df['Sharpe_Score'] +
            0.2 * ranking_df['Vol_Score'] +
            0.1 * ranking_df['DD_Score']
        )
        
        # Sort by composite score
        ranking_df = ranking_df.sort_values('Composite_Score', ascending=False)
        ranking_df['Rank'] = range(1, len(ranking_df) + 1)
        
        # Clean up temporary columns
        score_cols = ['Return_Score', 'Sharpe_Score', 'Vol_Score', 'DD_Score']
        ranking_df = ranking_df.drop(columns=score_cols)
        
        return ranking_df
    
    def _normalize_score(self, series: pd.Series, higher_better: bool = True) -> pd.Series:
        """Normalize series to 0-1 scale"""
        min_val = series.min()
        max_val = series.max()
        
        if max_val == min_val:
            return pd.Series([0.5] * len(series), index=series.index)
        
        normalized = (series - min_val) / (max_val - min_val)
        
        if not higher_better:
            normalized = 1 - normalized
        
        return normalized
    
    def generate_evaluation_report(self, results: List[Dict[str, Any]], output_file: Optional[str] = None) -> str:
        """Generate comprehensive evaluation report"""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.results_dir / f"evaluation_report_{timestamp}.json"
        
        # Create all analysis components
        performance_summary = self.create_performance_summary(results)
        market_matrix = self.create_market_performance_matrix(results)
        risk_metrics = self.calculate_risk_metrics(results)
        ranking_table = self.create_ranking_table(results)
        
        # Create report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_strategies_evaluated': len(results),
            'evaluation_summary': {
                'best_overall_strategy': ranking_table.iloc[0]['Strategy'],
                'highest_return_strategy': performance_summary.iloc[0]['Strategy'],
                'best_sharpe_strategy': risk_metrics.loc[risk_metrics['Sharpe_Ratio'].idxmax(), 'Strategy'],
                'lowest_risk_strategy': risk_metrics.loc[risk_metrics['Volatility_%'].idxmin(), 'Strategy']
            },
            'performance_summary': performance_summary.to_dict('records'),
            'market_performance_matrix': market_matrix.to_dict('records'),
            'risk_metrics': risk_metrics.to_dict('records'),
            'comprehensive_ranking': ranking_table.to_dict('records')
        }
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f" Evaluation report saved to: {output_file}")
        
        # Print summary to console
        self._print_summary(ranking_table, performance_summary)
        
        # Print detailed market analysis
        self.print_detailed_market_analysis(results)
        
        return str(output_file)
    
    def _print_summary(self, ranking_df: pd.DataFrame, performance_df: pd.DataFrame):
        """Print evaluation summary to console"""
        print("\n" + "=" * 60)
        print("=� STRATEGY EVALUATION SUMMARY")
        print("=" * 60)
        
        print("\n<� TOP 3 STRATEGIES (by composite score):")
        for _, row in ranking_df.head(5).iterrows():
            print(f"  {row['Rank']}. {row['Strategy']}: {row['Total_Return_%']}% return, "
                  f"{row['Win_Rate_%']}% win rate, Sharpe: {row['Sharpe_Ratio']}")
        
        print(f"\n=� PERFORMANCE OVERVIEW:")
        print(f"  Best Return: {performance_df.iloc[0]['Strategy']} ({performance_df.iloc[0]['Total_Return_%']}%)")
        print(f"  Total Strategies: {len(ranking_df)}")
        print(f"  Average Return: {performance_df['Total_Return_%'].mean():.2f}%")
        print(f"  Average Win Rate: {performance_df['Win_Rate_%'].mean():.2f}%")
        
        print("\n=� Full ranking and detailed metrics saved to results directory")
    
    def run_evaluation(self, config_path: str = "config/config_backtesting.json", 
                      strategies: List[str] = None, use_cached_data: bool = True) -> str:
        """Run complete evaluation process"""
        print("=, Starting strategy evaluation process...")
        
        # Run backtesting if needed
        backtester = MultiStrategyBacktester(config_path)
        
        if strategies is None:
            strategies = list(backtester.available_strategies)
        
        # Run backtesting
        success = backtester.run_backtest(
            strategies=strategies,
            use_cached_data=use_cached_data,
            data_only=False
        )
        
        if not success:
            raise RuntimeError("Backtesting failed")
        
        # Find the most recent results file
        results_files = list(self.results_dir.glob("backtest_results_*.json"))
        if not results_files:
            raise FileNotFoundError("No backtest results found")
        
        latest_results = max(results_files, key=os.path.getctime)
        
        # Load and evaluate results
        results = self.load_backtest_results(latest_results)
        report_file = self.generate_evaluation_report(results)
        
        return report_file
    
    def print_detailed_market_analysis(self, results: List[Dict[str, Any]]):
        """Print detailed market analysis with top 5 strategies per market"""
        market_analysis = self.create_detailed_market_analysis(results)
        
        print("\n" + "=" * 140)
        print("🎯 DETAILED MARKET ANALYSIS - TOP 5 STRATEGIES PER MARKET")
        print("=" * 140)
        
        for market, market_df in market_analysis.items():
            if not market_df.empty:
                print(f"\n📈 Market: {market}")
                print("-" * 120)
                print(market_df.to_string(index=False, max_colwidth=12))
            else:
                print(f"\n📈 Market: {market}")
                print("-" * 120)
                print("No performance data available")
        
        print("\n" + "=" * 140)


def main():
    """Main function for running evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Strategy Evaluation System")
    parser.add_argument('--results-file', help="Specific results file to evaluate")
    parser.add_argument('--config', default='config/config_backtesting.json',
                       help='Configuration file for new backtesting')
    parser.add_argument('--strategies', nargs='+', help="Strategies to evaluate")
    parser.add_argument('--use-cached-data', action='store_true',
                       help='Use cached data for backtesting')
    
    args = parser.parse_args()
    
    try:
        evaluator = StrategyEvaluator()
        
        if args.results_file:
            # Evaluate existing results
            results = evaluator.load_backtest_results(args.results_file)
            report_file = evaluator.generate_evaluation_report(results)
        else:
            # Run new evaluation
            report_file = evaluator.run_evaluation(
                config_path=args.config,
                strategies=args.strategies,
                use_cached_data=args.use_cached_data
            )
        
        print(f"\n<� Evaluation completed! Report: {report_file}")
    
    except Exception as e:
        print(f"L Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()