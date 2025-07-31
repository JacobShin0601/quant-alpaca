"""
Optimization Results Display Module
Enhanced display of optimization results with detailed train/test metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')


class OptimizationResultsDisplay:
    """Enhanced display of optimization results"""
    
    def __init__(self):
        self.results = {}
        
    def format_currency(self, value: float, initial_capital: float = 1000000) -> str:
        """Format currency values"""
        if abs(value) >= 1000000:
            return f"₩{value/1000000:.1f}M"
        elif abs(value) >= 1000:
            return f"₩{value/1000:.0f}K"
        else:
            return f"₩{value:.0f}"
    
    def format_percentage(self, value: float, decimals: int = 2) -> str:
        """Format percentage values"""
        return f"{value * 100:.{decimals}f}%"
    
    def format_ratio(self, value: float, decimals: int = 3) -> str:
        """Format ratio values"""
        if pd.isna(value) or np.isinf(value) or value > 999:
            return "N/A"
        return f"{value:.{decimals}f}"
    
    def calculate_log_returns(self, initial_capital: float, final_value: float) -> float:
        """Calculate log returns"""
        if initial_capital <= 0 or final_value <= 0:
            return 0.0
        return np.log(final_value / initial_capital)
    
    def calculate_risk_adjusted_return(self, total_return: float, volatility: float, 
                                     risk_free_rate: float = 0.02) -> float:
        """Calculate risk adjusted return"""
        if volatility == 0:
            return 0.0
        return (total_return - risk_free_rate) / volatility
    
    def create_detailed_results_table(self, optimization_results: Dict, 
                                    strategy_name: str, market: str) -> Dict[str, pd.DataFrame]:
        """Create detailed train/test results tables"""
        
        if not optimization_results or 'best_params' not in optimization_results:
            return {"train": pd.DataFrame(), "test": pd.DataFrame()}
        
        # Extract results
        train_performance = optimization_results.get('train_performance', 0)
        test_performance = optimization_results.get('test_performance', {})
        best_params = optimization_results.get('best_params', {})
        n_trials = optimization_results.get('n_trials', 0)
        
        # Get initial capital from config or default
        initial_capital = 3000000  # Default from config
        
        # Prepare train results
        train_data = {
            'Metric': ['Training Score'],
            'Value': [self.format_ratio(train_performance)]
        }
        
        # Prepare test results with detailed metrics
        test_metrics = []
        test_values = []
        
        if isinstance(test_performance, dict):
            # Total Return Amount
            total_return = test_performance.get('total_return', 0)
            final_value = initial_capital * (1 + total_return)
            profit_loss = final_value - initial_capital
            
            test_metrics.extend([
                '누적수익금액 (P&L)',
                '최종자산가치',
                '누적수익률',
                '누적로그수익률',
                '연환산수익률',
                '리스크조정수익',
                'Sharpe Ratio',
                'Sortino Ratio', 
                'Calmar Ratio',
                'MDD (Max Drawdown)',
                '변동성 (Volatility)',
                'Total Trades',
                '승률 (추정)',
                '평균거래당 수익',
                '',  # Separator
                'Buy & Hold 누적수익률',
                'Buy & Hold Sharpe Ratio',
                'Buy & Hold MDD',
                '전략 vs B&H 초과수익률'
            ])
            
            # Calculate additional metrics
            log_returns = self.calculate_log_returns(initial_capital, final_value)
            annualized_return = ((1 + total_return) ** (365/60) - 1) if total_return > -0.99 else -1.0
            volatility = test_performance.get('volatility', 0)
            risk_adjusted_return = self.calculate_risk_adjusted_return(total_return, volatility)
            
            sharpe_ratio = test_performance.get('sharpe_ratio', 0)
            sortino_ratio = test_performance.get('sortino_ratio', 0)
            calmar_ratio = test_performance.get('calmar_ratio', 0)
            max_drawdown = test_performance.get('max_drawdown', 0)
            total_trades = test_performance.get('total_trades', 0)
            
            # Estimate win rate (simplified)
            estimated_win_rate = max(0.3, min(0.7, 0.5 + sharpe_ratio * 0.1)) if sharpe_ratio != 0 else 0.5
            avg_profit_per_trade = profit_loss / max(1, total_trades)
            
            # Extract Buy & Hold benchmark data if available
            buy_hold_data = test_performance.get('buy_hold_benchmark', {})
            bh_total_return = buy_hold_data.get('total_return', 0)
            bh_sharpe_ratio = buy_hold_data.get('sharpe_ratio', 0)
            bh_max_drawdown = buy_hold_data.get('max_drawdown', 0)
            excess_return = total_return - bh_total_return
            
            test_values.extend([
                self.format_currency(profit_loss),
                self.format_currency(final_value),
                self.format_percentage(total_return),
                self.format_ratio(log_returns),
                self.format_percentage(annualized_return),
                self.format_ratio(risk_adjusted_return),
                self.format_ratio(sharpe_ratio),
                self.format_ratio(sortino_ratio),
                self.format_ratio(calmar_ratio),
                self.format_percentage(max_drawdown),
                self.format_percentage(volatility),
                f"{total_trades}",
                self.format_percentage(estimated_win_rate),
                self.format_currency(avg_profit_per_trade),
                '',  # Separator
                self.format_percentage(bh_total_return),
                self.format_ratio(bh_sharpe_ratio),
                self.format_percentage(bh_max_drawdown),
                self.format_percentage(excess_return)
            ])
        else:
            test_metrics = ['Test Score']
            test_values = [self.format_ratio(test_performance) if test_performance else 'N/A']
        
        # Create DataFrames - proper row x column format
        train_df = pd.DataFrame({
            'Metric': train_data['Metric'],
            'Value': train_data['Value']
        })
        
        test_df = pd.DataFrame({
            'Metric': test_metrics,
            'Value': test_values
        })
        
        return {"train": train_df, "test": test_df}
    
    def create_parameter_summary_table(self, optimization_results: Dict) -> pd.DataFrame:
        """Create parameter summary table"""
        if not optimization_results or 'best_params' not in optimization_results:
            return pd.DataFrame()
        
        best_params = optimization_results.get('best_params', {})
        n_trials = optimization_results.get('n_trials', 0)
        
        param_data = []
        for param_name, param_value in best_params.items():
            if isinstance(param_value, float):
                formatted_value = f"{param_value:.4f}"
            elif isinstance(param_value, bool):
                formatted_value = str(param_value)
            else:
                formatted_value = str(param_value)
            
            param_data.append({
                'Parameter': param_name,
                'Optimized Value': formatted_value
            })
        
        param_df = pd.DataFrame(param_data)
        
        # Add optimization info
        optimization_info = pd.DataFrame({
            'Parameter': ['Optimization Trials', 'Optimization Method'],
            'Optimized Value': [str(n_trials), 'Bayesian Optimization (TPE)']
        })
        
        return pd.concat([param_df, optimization_info], ignore_index=True)
    
    def display_compact_summary_table(self, optimization_results: Dict, 
                                     strategy_name: str, market: str):
        """Display compact horizontal summary table"""
        
        print("\n" + "="*120)
        print("📊 OPTIMIZATION RESULTS SUMMARY")
        print("="*120)
        
        if not optimization_results or 'test_performance' not in optimization_results:
            print("❌ No optimization results available")
            return
            
        test_perf = optimization_results.get('test_performance', {})
        train_perf = optimization_results.get('train_performance', 0)
        n_trials = optimization_results.get('n_trials', 0)
        
        # Calculate excess return vs Buy & Hold
        buy_hold_benchmark = test_perf.get('buy_hold_benchmark', {})
        buy_hold_return = 0
        if isinstance(buy_hold_benchmark, dict):
            buy_hold_return = buy_hold_benchmark.get('total_return', 0)
        
        excess_return = test_perf.get('total_return', 0) - buy_hold_return
        
        # Create horizontal summary table
        summary_data = {
            'Strategy': [strategy_name.upper()],
            'Market': [market],
            'Trials': [n_trials],
            'Train Sharpe': [f"{train_perf:.4f}"],
            'Test Sharpe': [f"{test_perf.get('sharpe_ratio', 0):.4f}"],
            'Test Return': [f"{test_perf.get('total_return', 0)*100:.2f}%"],
            'B&H Return': [f"{buy_hold_return*100:.2f}%"],
            'Excess Return': [f"{excess_return*100:.2f}%"],
            'Max Drawdown': [f"{test_perf.get('max_drawdown', 0)*100:.2f}%"],
            'Total Trades': [test_perf.get('total_trades', 0)]
        }
        
        df = pd.DataFrame(summary_data)
        
        # Format table with wide layout
        table_str = tabulate(df, headers='keys', tablefmt='grid', showindex=False)
        print(table_str)
        
        # Display optimized parameters in compact format
        print("\n🎯 OPTIMIZED PARAMETERS")
        print("="*120)
        print()
        best_params = optimization_results.get('best_params', {})
        
        if isinstance(best_params, dict) and market in best_params:
            params = best_params[market]
            print(f"{strategy_name.upper()}:")
            print(f"  {market}:")
            for param, value in params.items():
                print(f"    - {param}: {value}")
        else:
            print("No optimized parameters available")
        print()
        
        # Performance rating
        total_return = test_perf.get('total_return', 0)
        sharpe_ratio = test_perf.get('sharpe_ratio', 0)
        max_drawdown = test_perf.get('max_drawdown', 0)
        total_trades = test_perf.get('total_trades', 0)
        
        score = 0
        if total_return > 0.05: score += 1  # 5%+ return
        if sharpe_ratio > 1.0: score += 1   # Sharpe > 1
        if max_drawdown > -0.1: score += 1  # MDD < 10%
        if total_trades >= 10: score += 1   # Sufficient trades
        if excess_return > 0.02: score += 1  # 2%+ excess return over Buy & Hold
        
        rating_map = {0: "⭐ Poor", 1: "⭐⭐ Below Average", 2: "⭐⭐⭐ Average", 
                     3: "⭐⭐⭐⭐ Good", 4: "⭐⭐⭐⭐⭐ Excellent", 5: "⭐⭐⭐⭐⭐ Exceptional"}
        
        print(f"🏆 Performance Rating: {rating_map.get(score, '⭐ Poor')}")
        
        # Key insights
        insights = []
        if excess_return > 0.05:
            insights.append("✅ Significantly outperformed Buy & Hold")
        elif excess_return > 0.02:
            insights.append("✅ Moderately outperformed Buy & Hold") 
        elif excess_return < -0.02:
            insights.append("⚠️ Underperformed Buy & Hold")
        
        if sharpe_ratio > 1.5:
            insights.append("✅ Excellent risk-adjusted returns")
        elif sharpe_ratio < 0.5:
            insights.append("⚠️ Poor risk-adjusted returns")
            
        if max_drawdown < -0.2:
            insights.append("⚠️ High maximum drawdown")
        elif max_drawdown > -0.05:
            insights.append("✅ Low drawdown risk")
            
        if insights:
            print(f"💡 Key Insights: {' | '.join(insights)}")
        
        print("="*120)

    def display_optimization_results(self, optimization_results: Dict, 
                                   strategy_name: str, market: str, 
                                   show_details: bool = True):
        """Display comprehensive optimization results"""
        
        # Show compact summary first
        self.display_compact_summary_table(optimization_results, strategy_name, market)
        
        if not show_details:
            return
        
        print("\n" + "="*80)
        print(f"STRATEGY OPTIMIZATION RESULTS")
        print("="*80)
        print(f"Strategy: {strategy_name}")
        print(f"Market: {market}")
        print(f"Optimization Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        if not optimization_results or 'best_params' not in optimization_results:
            print("❌ No optimization results available")
            return
        
        # Create detailed results tables
        results_tables = self.create_detailed_results_table(optimization_results, strategy_name, market)
        param_table = self.create_parameter_summary_table(optimization_results)
        
        # Display parameter summary
        if not param_table.empty:
            print("\n📊 OPTIMIZED PARAMETERS")
            print("-" * 50)
            print(tabulate(param_table, headers='keys', tablefmt='grid', showindex=False))
        
        # Display training results
        if not results_tables["train"].empty:
            print("\n🎯 TRAINING PERFORMANCE")
            print("-" * 50)
            print(tabulate(results_tables["train"], headers='keys', tablefmt='grid', showindex=False))
        
        # Display test results  
        if not results_tables["test"].empty:
            print("\n📈 TEST PERFORMANCE (Out-of-Sample)")
            print("-" * 50)
            print(tabulate(results_tables["test"], headers='keys', tablefmt='grid', showindex=False))
        
        # Performance summary
        test_perf = optimization_results.get('test_performance', {})
        if isinstance(test_perf, dict):
            total_return = test_perf.get('total_return', 0)
            sharpe_ratio = test_perf.get('sharpe_ratio', 0)
            max_drawdown = test_perf.get('max_drawdown', 0)
            total_trades = test_perf.get('total_trades', 0)
            
            # Calculate excess return vs Buy & Hold
            buy_hold_benchmark = test_perf.get('buy_hold_benchmark', {})
            buy_hold_return = 0
            if isinstance(buy_hold_benchmark, dict):
                buy_hold_return = buy_hold_benchmark.get('total_return', 0)
            excess_return = total_return - buy_hold_return
            
            print("\n🎯 PERFORMANCE SUMMARY")
            print("-" * 50)
            
            # Performance rating (now includes Buy & Hold comparison)
            score = 0
            if total_return > 0.05: score += 1  # 5%+ return
            if sharpe_ratio > 1.0: score += 1   # Sharpe > 1
            if max_drawdown > -0.1: score += 1  # MDD < 10%
            if total_trades >= 10: score += 1   # Sufficient trades
            if excess_return > 0.02: score += 1  # 2%+ excess return over Buy & Hold
            
            rating_map = {0: "⭐ Poor", 1: "⭐⭐ Below Average", 2: "⭐⭐⭐ Average", 
                         3: "⭐⭐⭐⭐ Good", 4: "⭐⭐⭐⭐⭐ Excellent", 5: "⭐⭐⭐⭐⭐ Exceptional"}
            
            print(f"Performance Rating: {rating_map.get(score, '⭐ Poor')}")
            
            # Key insights (now includes Buy & Hold comparisons)
            insights = []
            if total_return > 0.1:
                insights.append("✅ Strong returns generated")
            elif total_return < -0.05:
                insights.append("⚠️ Negative returns - consider parameter adjustment")
            
            if excess_return > 0.05:
                insights.append("✅ Significantly outperformed Buy & Hold")
            elif excess_return > 0.02:
                insights.append("✅ Moderately outperformed Buy & Hold") 
            elif excess_return < -0.02:
                insights.append("⚠️ Underperformed Buy & Hold - strategy may not add value")
            
            if sharpe_ratio > 1.5:
                insights.append("✅ Excellent risk-adjusted returns")
            elif sharpe_ratio < 0.5:
                insights.append("⚠️ Poor risk-adjusted performance")
            
            if max_drawdown < -0.2:
                insights.append("⚠️ High drawdown - risk management needed")
            
            if total_trades < 5:
                insights.append("⚠️ Very few trades - may need parameter adjustment")
            elif total_trades > 100:
                insights.append("⚠️ Very frequent trading - check transaction costs")
            
            if insights:
                print("\n💡 Key Insights:")
                for insight in insights:
                    print(f"  {insight}")
        
        # Show optimization history if available
        if show_details and 'optimization_history' in optimization_results:
            history = optimization_results['optimization_history']
            if history and len(history) > 0:
                print(f"\n📊 OPTIMIZATION PROGRESS (Top {len(history)} trials)")
                print("-" * 50)
                
                history_data = []
                for i, trial in enumerate(history[:5]):  # Show top 5
                    history_data.append({
                        'Rank': i + 1,
                        'Score': self.format_ratio(trial.get('value', 0)),
                        'Key Parameters': str(trial.get('params', {}))[:60] + "..." if len(str(trial.get('params', {}))) > 60 else str(trial.get('params', {}))
                    })
                
                if history_data:
                    history_df = pd.DataFrame(history_data)
                    print(tabulate(history_df, headers='keys', tablefmt='grid', showindex=False))
        
        print("\n" + "="*80)
    
    def save_results_to_file(self, optimization_results: Dict, strategy_name: str, 
                           market: str, output_dir: str = "results/optimization"):
        """Save optimization results to JSON file"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{strategy_name}_{market}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Prepare data for saving
        save_data = {
            'strategy': strategy_name,
            'market': market,
            'optimization_date': datetime.now().isoformat(),
            'results': optimization_results
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {filepath}")


def main():
    """Example usage"""
    print("Optimization Results Display Module loaded successfully")


if __name__ == "__main__":
    main()