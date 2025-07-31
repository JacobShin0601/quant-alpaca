"""
Enhanced Trading Cost Modeling for High-Frequency Cryptocurrency Trading
Realistic cost modeling including slippage, market impact, and latency effects
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class EnhancedTradingCostModel:
    """Advanced trading cost model for cryptocurrency markets"""
    
    def __init__(self):
        # Default fee structures for major exchanges
        self.exchange_fees = {
            'upbit': {
                'maker': {'KRW': 0.0005, 'BTC': 0.0025, 'USDT': 0.0025},
                'taker': {'KRW': 0.0005, 'BTC': 0.0025, 'USDT': 0.0025}
            },
            'binance': {
                'maker': {'USDT': 0.001, 'BTC': 0.001},
                'taker': {'USDT': 0.001, 'BTC': 0.001}
            },
            'coinbase': {
                'maker': {'USD': 0.005, 'BTC': 0.005},
                'taker': {'USD': 0.005, 'BTC': 0.005}
            }
        }
        
        # Default slippage parameters
        self.slippage_params = {
            'base_slippage': 0.0001,  # 0.01% base slippage
            'impact_coefficient': 0.5,  # Square root impact
            'liquidity_adjustment': 1.0
        }
        
        # Latency parameters
        self.latency_params = {
            'signal_to_order': 0.1,  # 100ms signal processing
            'order_to_fill': 0.5,    # 500ms order execution
            'price_drift_factor': 0.02  # Price drift during latency
        }
    
    def calculate_dynamic_fees(self, market: str, volume_30d: float = 0, 
                             order_type: str = 'taker') -> float:
        """Calculate dynamic fees based on trading volume and market"""
        # Extract base currency from market (e.g., KRW-BTC -> KRW)
        base_currency = market.split('-')[0] if '-' in market else 'KRW'
        
        # Default to Upbit fees
        base_fee = self.exchange_fees['upbit'][order_type].get(base_currency, 0.0005)
        
        # Volume-based fee reduction (VIP levels)
        if volume_30d > 1000000000:  # 1B KRW
            fee_multiplier = 0.8
        elif volume_30d > 500000000:  # 500M KRW
            fee_multiplier = 0.85
        elif volume_30d > 100000000:  # 100M KRW  
            fee_multiplier = 0.9
        else:
            fee_multiplier = 1.0
            
        return base_fee * fee_multiplier
    
    def calculate_market_impact_slippage(self, order_size: float, avg_volume: float, 
                                       volatility: float, spread: float) -> float:
        """Calculate market impact slippage based on order characteristics"""
        # Volume impact (square root law)
        volume_ratio = order_size / (avg_volume + 1e-8)
        volume_impact = self.slippage_params['impact_coefficient'] * np.sqrt(volume_ratio)
        
        # Volatility adjustment
        vol_adjustment = volatility * 0.5
        
        # Spread component
        spread_impact = spread * 0.5
        
        # Combined slippage
        total_slippage = (self.slippage_params['base_slippage'] + 
                         volume_impact + vol_adjustment + spread_impact)
        
        return min(total_slippage, 0.01)  # Cap at 1%
    
    def calculate_timing_slippage(self, df: pd.DataFrame, entry_time: int, 
                                exit_time: int) -> Tuple[float, float]:
        """Calculate slippage due to execution timing and latency"""
        if entry_time >= len(df) or exit_time >= len(df):
            return 0.0, 0.0
            
        # Entry slippage due to latency
        signal_price = df['trade_price'].iloc[entry_time]
        
        # Simulate price movement during order processing
        if entry_time + 1 < len(df):
            execution_price = df['trade_price'].iloc[entry_time + 1]
            entry_slippage = abs(execution_price - signal_price) / signal_price
        else:
            entry_slippage = 0.0
            
        # Exit slippage
        if exit_time > 0:
            exit_signal_price = df['trade_price'].iloc[exit_time]
            if exit_time + 1 < len(df):
                exit_execution_price = df['trade_price'].iloc[exit_time + 1]
                exit_slippage = abs(exit_execution_price - exit_signal_price) / exit_signal_price
            else:
                exit_slippage = 0.0
        else:
            exit_slippage = 0.0
            
        return entry_slippage, exit_slippage
    
    def calculate_partial_fill_costs(self, order_size: float, fill_rate: float = 1.0,
                                   time_to_fill: int = 1) -> Dict[str, float]:
        """Calculate costs associated with partial fills"""
        if fill_rate >= 1.0:
            return {'additional_fees': 0.0, 'opportunity_cost': 0.0}
            
        # Additional fees for multiple fills
        num_fills = int(1 / fill_rate) if fill_rate > 0 else 1
        additional_fees = (num_fills - 1) * 0.0001  # 0.01% per additional fill
        
        # Opportunity cost from delayed execution
        opportunity_cost = time_to_fill * 0.00005  # 0.005% per minute delay
        
        return {
            'additional_fees': additional_fees,
            'opportunity_cost': opportunity_cost,
            'total_partial_cost': additional_fees + opportunity_cost
        }
    
    def estimate_liquidity_impact(self, df: pd.DataFrame, window: int = 60) -> pd.Series:
        """Estimate liquidity impact based on volume and volatility patterns"""
        # Volume relative to recent average
        volume_ma = df['candle_acc_trade_volume'].rolling(window=window).mean()
        volume_ratio = df['candle_acc_trade_volume'] / volume_ma
        
        # Volatility impact
        returns = df['trade_price'].pct_change()
        volatility = returns.rolling(window=window).std()
        vol_impact = volatility * 10  # Scale volatility to impact
        
        # Spread estimate
        spread_est = (df['high_price'] - df['low_price']) / df['trade_price']
        
        # Combined liquidity score (lower is better)
        liquidity_score = (1 / volume_ratio) * (1 + vol_impact) * (1 + spread_est)
        
        return liquidity_score
    
    def calculate_total_trading_costs(self, df: pd.DataFrame, trades: pd.DataFrame,
                                    position_sizes: pd.Series, market: str = 'KRW-BTC') -> pd.DataFrame:
        """Calculate comprehensive trading costs for a series of trades"""
        cost_breakdown = []
        
        for idx, trade in trades.iterrows():
            entry_time = trade.get('entry_time', 0)
            exit_time = trade.get('exit_time', len(df)-1)
            position_size = position_sizes.iloc[entry_time] if entry_time < len(position_sizes) else 1.0
            
            # Basic fees
            entry_fee = self.calculate_dynamic_fees(market, order_type='taker')
            exit_fee = self.calculate_dynamic_fees(market, order_type='taker')
            
            # Market impact slippage
            if entry_time < len(df):
                avg_volume = df['candle_acc_trade_volume'].iloc[max(0, entry_time-60):entry_time+1].mean()
                volatility = df['trade_price'].pct_change().iloc[max(0, entry_time-60):entry_time+1].std()
                spread = ((df['high_price'].iloc[entry_time] - df['low_price'].iloc[entry_time]) / 
                         df['trade_price'].iloc[entry_time])
                
                entry_slippage = self.calculate_market_impact_slippage(
                    abs(position_size), avg_volume, volatility, spread
                )
                exit_slippage = self.calculate_market_impact_slippage(
                    abs(position_size), avg_volume, volatility, spread
                )
            else:
                entry_slippage = exit_slippage = 0.0
            
            # Timing costs
            timing_entry, timing_exit = self.calculate_timing_slippage(df, entry_time, exit_time)
            
            # Partial fill costs
            fill_rate = trade.get('fill_rate', 1.0)
            partial_costs = self.calculate_partial_fill_costs(abs(position_size), fill_rate)
            
            # Total costs
            total_cost = (entry_fee + exit_fee + entry_slippage + exit_slippage + 
                         timing_entry + timing_exit + partial_costs['total_partial_cost'])
            
            cost_detail = {
                'trade_id': idx,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'position_size': position_size,
                'entry_fee': entry_fee,
                'exit_fee': exit_fee,
                'entry_slippage': entry_slippage,
                'exit_slippage': exit_slippage,
                'timing_entry': timing_entry,
                'timing_exit': timing_exit,
                'partial_fill_cost': partial_costs['total_partial_cost'],
                'total_cost': total_cost,
                'cost_bps': total_cost * 10000  # Cost in basis points
            }
            
            cost_breakdown.append(cost_detail)
        
        return pd.DataFrame(cost_breakdown)
    
    def optimize_execution_timing(self, df: pd.DataFrame, signal_times: List[int],
                                liquidity_threshold: float = 0.5) -> pd.DataFrame:
        """Optimize trade execution timing based on liquidity conditions"""
        liquidity_scores = self.estimate_liquidity_impact(df)
        
        optimized_executions = []
        
        for signal_time in signal_times:
            if signal_time >= len(df) - 10:
                continue
                
            # Look for best execution window within next 10 minutes
            window_start = signal_time
            window_end = min(signal_time + 10, len(df))
            
            window_liquidity = liquidity_scores.iloc[window_start:window_end]
            
            # Find optimal execution time (lowest liquidity score)
            if len(window_liquidity) > 0:
                optimal_offset = window_liquidity.idxmin() - signal_time
                optimal_time = signal_time + optimal_offset
                
                # Calculate cost improvement
                signal_cost = liquidity_scores.iloc[signal_time] if signal_time < len(liquidity_scores) else 1.0
                optimal_cost = liquidity_scores.iloc[optimal_time] if optimal_time < len(liquidity_scores) else 1.0
                cost_improvement = (signal_cost - optimal_cost) / signal_cost
                
                execution_info = {
                    'signal_time': signal_time,
                    'optimal_time': optimal_time,
                    'delay_minutes': optimal_offset,
                    'signal_liquidity_score': signal_cost,
                    'optimal_liquidity_score': optimal_cost,
                    'cost_improvement_pct': cost_improvement * 100,
                    'recommended': optimal_cost < liquidity_threshold
                }
                
                optimized_executions.append(execution_info)
        
        return pd.DataFrame(optimized_executions)
    
    def simulate_realistic_execution(self, df: pd.DataFrame, signals: pd.Series,
                                   position_sizes: pd.Series, 
                                   execution_params: Dict = None) -> pd.DataFrame:
        """Simulate realistic trade execution with all cost factors"""
        if execution_params is None:
            execution_params = {
                'max_position_pct': 0.2,
                'min_liquidity_score': 0.8,
                'max_slippage_tolerance': 0.005,
                'partial_fill_threshold': 0.8
            }
        
        liquidity_scores = self.estimate_liquidity_impact(df)
        executed_trades = []
        
        current_position = 0.0
        
        for i, signal in enumerate(signals):
            if signal == 0 or i >= len(df) - 1:
                continue
                
            desired_position = position_sizes.iloc[i] if i < len(position_sizes) else 0.1
            position_change = desired_position - current_position
            
            if abs(position_change) < 0.01:  # Minimum trade size
                continue
            
            # Check liquidity conditions
            current_liquidity = liquidity_scores.iloc[i] if i < len(liquidity_scores) else 1.0
            
            if current_liquidity > execution_params['min_liquidity_score']:
                # Poor liquidity - reduce position size or skip
                position_change *= 0.5
                
            # Calculate expected costs
            avg_volume = df['candle_acc_trade_volume'].iloc[max(0, i-60):i+1].mean()
            volatility = df['trade_price'].pct_change().iloc[max(0, i-60):i+1].std()
            spread = ((df['high_price'].iloc[i] - df['low_price'].iloc[i]) / 
                     df['trade_price'].iloc[i])
            
            expected_slippage = self.calculate_market_impact_slippage(
                abs(position_change), avg_volume, volatility, spread
            )
            
            # Skip trade if slippage too high
            if expected_slippage > execution_params['max_slippage_tolerance']:
                continue
            
            # Simulate partial fills
            fill_rate = min(1.0, np.random.normal(0.9, 0.1))  # Random fill rate
            fill_rate = max(0.5, fill_rate)  # Minimum 50% fill
            
            actual_position_change = position_change * fill_rate
            
            # Execute trade
            trade_info = {
                'timestamp': df.index[i],
                'signal': signal,
                'desired_position_change': position_change,
                'actual_position_change': actual_position_change,
                'fill_rate': fill_rate,
                'execution_price': df['trade_price'].iloc[i],
                'liquidity_score': current_liquidity,
                'expected_slippage': expected_slippage,
                'spread': spread,
                'volume_ratio': df['candle_acc_trade_volume'].iloc[i] / avg_volume
            }
            
            executed_trades.append(trade_info)
            current_position += actual_position_change
        
        return pd.DataFrame(executed_trades)


class HighFrequencyPerformanceMetrics:
    """High-frequency specific performance metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_hf_metrics(self, returns: pd.Series, trades: pd.DataFrame = None,
                           cost_breakdown: pd.DataFrame = None) -> Dict[str, float]:
        """Calculate comprehensive high-frequency performance metrics"""
        metrics = {}
        
        # Basic metrics (annualized for 1-minute data)
        annual_factor = np.sqrt(525600)  # Minutes in a year
        
        metrics['annualized_return'] = returns.mean() * 525600
        metrics['annualized_volatility'] = returns.std() * annual_factor
        metrics['sharpe_ratio'] = (returns.mean() / returns.std() * annual_factor) if returns.std() > 0 else 0
        
        # Sortino ratio (downside volatility)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_vol = downside_returns.std() * annual_factor
            metrics['sortino_ratio'] = (returns.mean() * 525600) / downside_vol if downside_vol > 0 else 0
        else:
            metrics['sortino_ratio'] = np.inf
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        metrics['max_drawdown'] = drawdown.min()
        
        # Calmar ratio
        metrics['calmar_ratio'] = abs(metrics['annualized_return'] / metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
        
        # High-frequency specific metrics
        if len(returns) > 1:
            # Hit rate
            positive_returns = (returns > 0).sum()
            metrics['hit_rate'] = positive_returns / len(returns)
            
            # Profit factor
            gross_profit = returns[returns > 0].sum()
            gross_loss = abs(returns[returns < 0].sum())
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else np.inf
            
            # Average win/loss
            winning_trades = returns[returns > 0]
            losing_trades = returns[returns < 0]
            
            metrics['avg_win'] = winning_trades.mean() if len(winning_trades) > 0 else 0
            metrics['avg_loss'] = losing_trades.mean() if len(losing_trades) > 0 else 0
            metrics['win_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0
            
            # Trade frequency metrics
            non_zero_returns = returns[returns != 0]
            metrics['trade_frequency_per_day'] = len(non_zero_returns) / (len(returns) / 1440) if len(returns) > 1440 else 0
            
            # Return consistency (lower is better)
            metrics['return_consistency'] = returns.std() / abs(returns.mean()) if returns.mean() != 0 else np.inf
        
        # Cost-adjusted metrics
        if cost_breakdown is not None and len(cost_breakdown) > 0:
            total_costs = cost_breakdown['total_cost'].sum()
            metrics['total_trading_costs_pct'] = total_costs * 100
            metrics['avg_cost_per_trade_bps'] = cost_breakdown['cost_bps'].mean()
            
            # Net returns after costs
            net_returns = returns - (cost_breakdown['total_cost'].sum() / len(returns))
            metrics['net_sharpe_ratio'] = (net_returns.mean() / net_returns.std() * annual_factor) if net_returns.std() > 0 else 0
        
        # Trade-specific metrics
        if trades is not None and len(trades) > 0:
            metrics['total_trades'] = len(trades)
            metrics['avg_trade_duration'] = trades['exit_time'].sub(trades['entry_time']).mean() if 'exit_time' in trades.columns else 0
            
            if 'fill_rate' in trades.columns:
                metrics['avg_fill_rate'] = trades['fill_rate'].mean()
                metrics['partial_fill_rate'] = (trades['fill_rate'] < 1.0).sum() / len(trades)
        
        return metrics
    
    def calculate_regime_specific_metrics(self, returns: pd.Series, regime_labels: pd.Series) -> Dict[str, Dict]:
        """Calculate performance metrics by market regime"""
        regime_metrics = {}
        
        unique_regimes = regime_labels.unique()
        
        for regime in unique_regimes:
            if pd.isna(regime):
                continue
                
            regime_mask = regime_labels == regime
            regime_returns = returns[regime_mask]
            
            if len(regime_returns) > 10:  # Minimum observations
                regime_metrics[f'regime_{regime}'] = self.calculate_hf_metrics(regime_returns)
                regime_metrics[f'regime_{regime}']['regime_percentage'] = (regime_mask.sum() / len(regime_labels)) * 100
        
        return regime_metrics
    
    def generate_performance_report(self, returns: pd.Series, trades: pd.DataFrame = None,
                                  cost_breakdown: pd.DataFrame = None,
                                  regime_labels: pd.Series = None) -> str:
        """Generate comprehensive performance report"""
        overall_metrics = self.calculate_hf_metrics(returns, trades, cost_breakdown)
        
        report = ["="*60]
        report.append("HIGH-FREQUENCY TRADING PERFORMANCE REPORT")
        report.append("="*60)
        
        # Overall performance
        report.append("\n📊 OVERALL PERFORMANCE")
        report.append("-" * 30)
        report.append(f"Annualized Return:      {overall_metrics['annualized_return']:.2%}")
        report.append(f"Annualized Volatility:  {overall_metrics['annualized_volatility']:.2%}")
        report.append(f"Sharpe Ratio:           {overall_metrics['sharpe_ratio']:.3f}")
        report.append(f"Sortino Ratio:          {overall_metrics['sortino_ratio']:.3f}")
        report.append(f"Maximum Drawdown:       {overall_metrics['max_drawdown']:.2%}")
        report.append(f"Calmar Ratio:           {overall_metrics['calmar_ratio']:.3f}")
        
        # Trading metrics
        report.append("\n📈 TRADING STATISTICS")
        report.append("-" * 30)
        report.append(f"Hit Rate:               {overall_metrics.get('hit_rate', 0):.2%}")
        report.append(f"Profit Factor:          {overall_metrics.get('profit_factor', 0):.2f}")
        report.append(f"Win/Loss Ratio:         {overall_metrics.get('win_loss_ratio', 0):.2f}")
        report.append(f"Average Win:            {overall_metrics.get('avg_win', 0):.4f}")
        report.append(f"Average Loss:           {overall_metrics.get('avg_loss', 0):.4f}")
        report.append(f"Trade Frequency/Day:    {overall_metrics.get('trade_frequency_per_day', 0):.1f}")
        
        # Cost analysis
        if cost_breakdown is not None:
            report.append("\n💰 COST ANALYSIS")
            report.append("-" * 30)
            report.append(f"Total Trading Costs:    {overall_metrics.get('total_trading_costs_pct', 0):.3f}%")
            report.append(f"Avg Cost per Trade:     {overall_metrics.get('avg_cost_per_trade_bps', 0):.1f} bps")
            report.append(f"Net Sharpe Ratio:       {overall_metrics.get('net_sharpe_ratio', 0):.3f}")
        
        # Regime analysis
        if regime_labels is not None:
            regime_metrics = self.calculate_regime_specific_metrics(returns, regime_labels)
            if regime_metrics:
                report.append("\n🌊 REGIME ANALYSIS")
                report.append("-" * 30)
                for regime_name, metrics in regime_metrics.items():
                    report.append(f"{regime_name}: Sharpe={metrics.get('sharpe_ratio', 0):.3f}, "
                                f"Coverage={metrics.get('regime_percentage', 0):.1f}%")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)


def main():
    """Example usage"""
    print("Enhanced Trading Cost Modeling module loaded successfully")


if __name__ == "__main__":
    main()