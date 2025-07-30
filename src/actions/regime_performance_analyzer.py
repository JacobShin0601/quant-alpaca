"""
Market Regime Performance Analyzer
Analyzes trading performance by market regime for better strategy insights
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import json

try:
    from .market_regime import MarketRegime, MarketRegimeDetector
except ImportError:
    from market_regime import MarketRegime, MarketRegimeDetector


@dataclass
class RegimePerformanceMetrics:
    """Performance metrics for a specific market regime"""
    regime: MarketRegime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    average_return: float
    total_pnl: float
    average_pnl: float
    max_win: float
    max_loss: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    avg_holding_period: timedelta
    total_duration: timedelta
    regime_percentage: float  # Percentage of time in this regime
    

@dataclass
class RegimeTransitionMetrics:
    """Metrics for regime transitions"""
    from_regime: MarketRegime
    to_regime: MarketRegime
    transition_count: int
    avg_return_before: float  # Average return before transition
    avg_return_after: float   # Average return after transition
    success_rate: float       # % of profitable transitions


class RegimePerformanceAnalyzer:
    """
    Analyzes trading performance across different market regimes
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize regime performance analyzer"""
        self.config = config or self._get_default_config()
        # Ensure regime config has all required fields
        regime_config = self.config.get('regime_config', {})
        if 'ma_periods' not in regime_config:
            regime_config['ma_periods'] = [20, 50]
        if 'bb_period' not in regime_config:
            regime_config['bb_period'] = 20
        if 'bb_std' not in regime_config:
            regime_config['bb_std'] = 2.0
        if 'volume_period' not in regime_config:
            regime_config['volume_period'] = 20
        
        self.regime_detector = MarketRegimeDetector(regime_config)
        
    def _get_default_config(self) -> Dict:
        """Default configuration for regime analysis"""
        return {
            "regime_config": {
                "lookback_period": 20,
                "volatility_threshold": 0.02,
                "trend_threshold": 0.0001,
                "volume_ma_period": 20,
                "atr_period": 14,
                "adx_period": 14,
                "adx_threshold": 25
            },
            "analysis": {
                "min_trades_per_regime": 5,  # Minimum trades for meaningful stats
                "transition_lookback": 5,     # Bars to analyze before/after transition
                "performance_metrics": [
                    "return", "sharpe", "win_rate", "drawdown", "holding_period"
                ]
            }
        }
    
    def analyze_regime_performance(self,
                                 portfolio_history: List[Dict],
                                 trade_history: List[Dict],
                                 market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze performance across different market regimes"""
        
        # First, detect regimes for all timestamps
        regime_history = self._detect_regimes_for_history(portfolio_history, market_data)
        
        # Group trades by regime
        trades_by_regime = self._group_trades_by_regime(trade_history, regime_history)
        
        # Calculate performance metrics for each regime
        regime_metrics = {}
        for regime in MarketRegime:
            if regime in trades_by_regime and len(trades_by_regime[regime]) >= self.config["analysis"]["min_trades_per_regime"]:
                metrics = self._calculate_regime_metrics(
                    trades_by_regime[regime],
                    portfolio_history,
                    regime_history,
                    regime
                )
                regime_metrics[regime.value] = metrics
        
        # Analyze regime transitions
        transition_metrics = self._analyze_regime_transitions(
            regime_history,
            portfolio_history,
            trade_history
        )
        
        # Create summary statistics
        summary = self._create_summary_statistics(
            regime_metrics,
            transition_metrics,
            regime_history
        )
        
        return {
            "regime_metrics": regime_metrics,
            "transition_metrics": transition_metrics,
            "summary": summary,
            "regime_history": regime_history
        }
    
    def _detect_regimes_for_history(self,
                                   portfolio_history: List[Dict],
                                   market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Detect market regime for each timestamp in history"""
        
        regime_history = []
        
        for i, point in enumerate(portfolio_history):
            timestamp = point['timestamp']
            
            # Get market data up to this timestamp
            regime_scores = {}
            detected_regimes = {}
            
            for market, df in market_data.items():
                if timestamp in df.index:
                    # Get data up to current timestamp
                    historical_data = df.loc[:timestamp]
                    
                    if len(historical_data) >= self.config["regime_config"]["lookback_period"]:
                        regime, confidence = self.regime_detector.detect_regime(historical_data)
                        detected_regimes[market] = regime
                        regime_scores[market] = confidence
            
            # Determine dominant regime (weighted by confidence)
            if detected_regimes:
                # Simple majority vote for now
                regime_counts = defaultdict(int)
                for regime in detected_regimes.values():
                    regime_counts[regime] += 1
                
                dominant_regime = max(regime_counts.items(), key=lambda x: x[1])[0]
            else:
                dominant_regime = MarketRegime.UNKNOWN
            
            regime_history.append({
                'timestamp': timestamp,
                'regime': dominant_regime,
                'portfolio_value': point['portfolio_value'],
                'regimes_by_market': detected_regimes
            })
        
        return regime_history
    
    def _group_trades_by_regime(self,
                               trade_history: List[Dict],
                               regime_history: List[Dict]) -> Dict[MarketRegime, List[Dict]]:
        """Group trades by the regime they occurred in"""
        
        trades_by_regime = defaultdict(list)
        
        # Create timestamp to regime mapping
        regime_map = {rh['timestamp']: rh['regime'] for rh in regime_history}
        
        for trade in trade_history:
            timestamp = trade['timestamp']
            
            # Find the regime at trade time
            regime = regime_map.get(timestamp, MarketRegime.UNKNOWN)
            
            # Add trade to appropriate regime bucket
            trades_by_regime[regime].append(trade)
        
        return dict(trades_by_regime)
    
    def _calculate_regime_metrics(self,
                                regime_trades: List[Dict],
                                portfolio_history: List[Dict],
                                regime_history: List[Dict],
                                regime: MarketRegime,
                                group_by: str = None) -> RegimePerformanceMetrics:
        """Calculate performance metrics for a specific regime"""
        
        # Filter portfolio history for this regime
        regime_portfolio = [
            ph for ph, rh in zip(portfolio_history, regime_history)
            if rh['regime'] == regime
        ]
        
        if not regime_portfolio or not regime_trades:
            return None
        
        # Calculate trade statistics
        buy_trades = [t for t in regime_trades if t['side'] == 'buy']
        # Include both regular sells and stop order sells
        sell_trades = [t for t in regime_trades if t['side'] == 'sell' or (t.get('action', '').endswith('_sell'))]
        
        # Match buy/sell pairs to calculate P&L
        trade_pnls = []
        holding_periods = []
        
        # Simple FIFO matching
        positions = defaultdict(list)  # market -> list of buy trades
        
        for trade in sorted(regime_trades, key=lambda x: x['timestamp']):
            market = trade['market']
            
            if trade['side'] == 'buy':
                positions[market].append(trade)
            elif (trade['side'] == 'sell' or trade.get('action', '').endswith('_sell')) and positions[market]:
                # Match with oldest buy
                buy_trade = positions[market].pop(0)
                
                # Calculate P&L
                buy_cost = buy_trade['quantity'] * buy_trade['price'] * (1 + buy_trade.get('fee_rate', 0))
                sell_proceeds = trade['quantity'] * trade['price'] * (1 - trade.get('fee_rate', 0))
                pnl = sell_proceeds - buy_cost
                pnl_pct = pnl / buy_cost
                
                # Handle timestamp differences for holding period
                buy_ts = buy_trade['timestamp']
                sell_ts = trade['timestamp']
                
                # Convert to timestamps if needed
                if isinstance(buy_ts, (int, float)):
                    buy_time = pd.to_datetime(buy_ts, unit='s')
                else:
                    buy_time = pd.to_datetime(buy_ts)
                    
                if isinstance(sell_ts, (int, float)):
                    sell_time = pd.to_datetime(sell_ts, unit='s')
                else:
                    sell_time = pd.to_datetime(sell_ts)
                
                holding_period = sell_time - buy_time
                
                trade_pnls.append({
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'holding_period': holding_period
                })
                
                holding_periods.append(holding_period)
        
        # Calculate metrics
        winning_trades = sum(1 for tp in trade_pnls if tp['pnl'] > 0)
        losing_trades = sum(1 for tp in trade_pnls if tp['pnl'] <= 0)
        total_trades = len(trade_pnls)
        
        if total_trades == 0:
            return None
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Returns and P&L
        returns = [tp['pnl_pct'] for tp in trade_pnls]
        pnls = [tp['pnl'] for tp in trade_pnls]
        
        # Calculate compounded total return
        total_return = 1.0
        for ret in returns:
            total_return *= (1 + ret)
        total_return = (total_return - 1) * 100  # Convert to percentage
        
        average_return = np.mean(returns) * 100 if returns else 0  # Convert to percentage
        total_pnl = sum(pnls)
        average_pnl = np.mean(pnls) if pnls else 0
        
        max_win = max(pnls) if pnls else 0
        max_loss = min(pnls) if pnls else 0
        
        # Risk metrics
        returns_series = pd.Series(returns)
        sharpe_ratio = self._calculate_sharpe_ratio(returns_series)
        sortino_ratio = self._calculate_sortino_ratio(returns_series)
        
        # Portfolio metrics for this regime
        portfolio_values = [p['portfolio_value'] for p in regime_portfolio]
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        # Time metrics
        avg_holding_period = pd.Timedelta(np.mean(holding_periods)) if holding_periods else timedelta(0)
        
        # Calculate regime duration
        regime_timestamps = [rh['timestamp'] for rh in regime_history if rh['regime'] == regime]
        total_duration = timedelta(0)
        if len(regime_timestamps) > 1:
            total_duration = regime_timestamps[-1] - regime_timestamps[0]
        
        # Regime percentage
        total_periods = len(regime_history)
        regime_periods = len([rh for rh in regime_history if rh['regime'] == regime])
        regime_percentage = (regime_periods / total_periods * 100) if total_periods > 0 else 0
        
        return RegimePerformanceMetrics(
            regime=regime,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_return=total_return,
            average_return=average_return,
            total_pnl=total_pnl,
            average_pnl=average_pnl,
            max_win=max_win,
            max_loss=max_loss,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            avg_holding_period=avg_holding_period,
            total_duration=total_duration,
            regime_percentage=regime_percentage
        )
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio for returns series"""
        if len(returns) < 2:
            return 0.0
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 0.0
        
        # Simple Sharpe ratio without annualization since these are per-trade returns
        # not time-based returns
        sharpe = mean_return / std_return
        return sharpe
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio for returns series"""
        if len(returns) < 2:
            return 0.0
        
        mean_return = returns.mean()
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if mean_return > 0 else 0.0
        
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return 0.0
        
        # Simple Sortino ratio without annualization for per-trade returns
        sortino = mean_return / downside_std
        return sortino
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown from a list of values"""
        if len(values) < 2:
            return 0.0
        
        running_max = values[0]
        max_drawdown = 0.0
        
        for value in values[1:]:
            running_max = max(running_max, value)
            drawdown = (value - running_max) / running_max if running_max > 0 else 0
            max_drawdown = min(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _analyze_regime_transitions(self,
                                   regime_history: List[Dict],
                                   portfolio_history: List[Dict],
                                   trade_history: List[Dict]) -> List[RegimeTransitionMetrics]:
        """Analyze performance around regime transitions"""
        
        transitions = []
        transition_stats = defaultdict(lambda: {
            'count': 0,
            'returns_before': [],
            'returns_after': [],
            'successful': 0
        })
        
        # Find all regime transitions
        for i in range(1, len(regime_history)):
            prev_regime = regime_history[i-1]['regime']
            curr_regime = regime_history[i]['regime']
            
            if prev_regime != curr_regime and prev_regime != MarketRegime.UNKNOWN and curr_regime != MarketRegime.UNKNOWN:
                # Found a transition
                transition_key = (prev_regime, curr_regime)
                stats = transition_stats[transition_key]
                stats['count'] += 1
                
                # Analyze performance around transition
                lookback = self.config["analysis"]["transition_lookback"]
                
                # Get returns before transition
                before_start = max(0, i - lookback)
                before_returns = []
                for j in range(before_start, i):
                    if j > 0:
                        ret = (portfolio_history[j]['portfolio_value'] - 
                              portfolio_history[j-1]['portfolio_value']) / portfolio_history[j-1]['portfolio_value']
                        before_returns.append(ret)
                
                # Get returns after transition
                after_end = min(len(portfolio_history), i + lookback)
                after_returns = []
                for j in range(i, after_end - 1):
                    ret = (portfolio_history[j+1]['portfolio_value'] - 
                          portfolio_history[j]['portfolio_value']) / portfolio_history[j]['portfolio_value']
                    after_returns.append(ret)
                
                if before_returns:
                    stats['returns_before'].extend(before_returns)
                if after_returns:
                    stats['returns_after'].extend(after_returns)
                
                # Check if transition was successful (positive return after)
                if after_returns and sum(after_returns) > 0:
                    stats['successful'] += 1
        
        # Create transition metrics
        for (from_regime, to_regime), stats in transition_stats.items():
            if stats['count'] > 0:
                avg_before = np.mean(stats['returns_before']) if stats['returns_before'] else 0
                avg_after = np.mean(stats['returns_after']) if stats['returns_after'] else 0
                success_rate = stats['successful'] / stats['count']
                
                transitions.append(RegimeTransitionMetrics(
                    from_regime=from_regime,
                    to_regime=to_regime,
                    transition_count=stats['count'],
                    avg_return_before=avg_before,
                    avg_return_after=avg_after,
                    success_rate=success_rate
                ))
        
        return transitions
    
    def _create_summary_statistics(self,
                                 regime_metrics: Dict[str, RegimePerformanceMetrics],
                                 transition_metrics: List[RegimeTransitionMetrics],
                                 regime_history: List[Dict]) -> Dict[str, Any]:
        """Create summary statistics across all regimes"""
        
        summary = {
            "total_regimes_analyzed": len(regime_metrics),
            "regime_distribution": {},
            "best_performing_regime": None,
            "worst_performing_regime": None,
            "most_profitable_regime": None,
            "highest_win_rate_regime": None,
            "regime_comparison": {},
            "transition_summary": {}
        }
        
        # Calculate regime distribution
        total_periods = len(regime_history)
        for regime in MarketRegime:
            count = sum(1 for rh in regime_history if rh['regime'] == regime)
            summary["regime_distribution"][regime.value] = {
                "count": count,
                "percentage": (count / total_periods * 100) if total_periods > 0 else 0
            }
        
        # Find best/worst regimes
        if regime_metrics:
            # By Sharpe ratio
            best_sharpe = max(regime_metrics.items(), 
                            key=lambda x: x[1].sharpe_ratio if x[1] else float('-inf'))
            worst_sharpe = min(regime_metrics.items(), 
                             key=lambda x: x[1].sharpe_ratio if x[1] else float('inf'))
            
            summary["best_performing_regime"] = best_sharpe[0]
            summary["worst_performing_regime"] = worst_sharpe[0]
            
            # By total P&L
            most_profitable = max(regime_metrics.items(),
                                key=lambda x: x[1].total_pnl if x[1] else float('-inf'))
            summary["most_profitable_regime"] = most_profitable[0]
            
            # By win rate
            highest_win_rate = max(regime_metrics.items(),
                                 key=lambda x: x[1].win_rate if x[1] else float('-inf'))
            summary["highest_win_rate_regime"] = highest_win_rate[0]
            
            # Create comparison table
            for regime_name, metrics in regime_metrics.items():
                if metrics:
                    summary["regime_comparison"][regime_name] = {
                        "total_return": f"{metrics.total_return:.2%}",
                        "win_rate": f"{metrics.win_rate:.2%}",
                        "sharpe_ratio": f"{metrics.sharpe_ratio:.2f}",
                        "max_drawdown": f"{metrics.max_drawdown:.2%}",
                        "avg_holding_period": str(metrics.avg_holding_period),
                        "regime_percentage": f"{metrics.regime_percentage:.1f}%"
                    }
        
        # Transition summary
        if transition_metrics:
            summary["transition_summary"]["total_transitions"] = sum(t.transition_count for t in transition_metrics)
            summary["transition_summary"]["most_common"] = max(
                transition_metrics, 
                key=lambda x: x.transition_count
            )
            summary["transition_summary"]["most_profitable"] = max(
                transition_metrics,
                key=lambda x: x.avg_return_after
            )
        
        return summary
    
    def create_detailed_regime_performance_table(self,
                                               analysis_results: Dict[str, Any],
                                               trade_history: List[Dict]) -> pd.DataFrame:
        """Create a detailed performance table by regime, strategy, and market"""
        
        regime_history = analysis_results.get("regime_history", [])
        
        if not regime_history or not trade_history:
            return pd.DataFrame()
        
        # Create timestamp to regime mapping
        regime_map = {rh['timestamp']: rh['regime'] for rh in regime_history}
        
        # Group trades by regime, strategy, and market
        grouped_trades = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        for trade in trade_history:
            timestamp = trade['timestamp']
            regime = regime_map.get(timestamp, MarketRegime.UNKNOWN)
            strategy = trade.get('strategy', 'unknown')
            market = trade['market']
            
            grouped_trades[regime.value][strategy][market].append(trade)
        
        # Calculate metrics for each group
        table_data = []
        
        for regime_name, strategy_data in grouped_trades.items():
            for strategy_name, market_data in strategy_data.items():
                for market_name, trades in market_data.items():
                    # Calculate metrics for this specific group
                    metrics = self._calculate_group_metrics(trades, regime_name)
                    
                    if metrics:
                        table_data.append({
                            "Regime": regime_name,
                            "Strategy": strategy_name,
                            "Market": market_name,
                            "Trades": metrics['total_trades'],
                            "Win Rate": f"{metrics['win_rate']*100:.1f}%",
                            "Total Return": f"{metrics['total_return']:.2f}%",
                            "Avg Return": f"{metrics['avg_return']:.2f}%",
                            "Total P&L": f"₩{metrics['total_pnl']:,.0f}",
                            "Sharpe": f"{metrics['sharpe']:.2f}",
                            "Max DD": f"{metrics['max_dd']:.2f}%",
                            "Avg Hold": metrics['avg_hold']
                        })
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        if not df.empty:
            # Sort by regime, then strategy, then market
            df = df.sort_values(['Regime', 'Strategy', 'Market'])
        
        return df
    
    def _calculate_group_metrics(self, trades: List[Dict], regime_name: str) -> Dict:
        """Calculate metrics for a specific group of trades"""
        
        # Separate buy and sell trades
        buy_trades = [t for t in trades if t['side'] == 'buy']
        sell_trades = [t for t in trades if t['side'] == 'sell' or t.get('action', '').endswith('_sell')]
        
        # Match trades to calculate P&L
        trade_pnls = []
        holding_periods = []
        positions = []
        
        for trade in sorted(trades, key=lambda x: x['timestamp']):
            if trade['side'] == 'buy':
                positions.append(trade)
            elif (trade['side'] == 'sell' or trade.get('action', '').endswith('_sell')) and positions:
                buy_trade = positions.pop(0)
                
                # Calculate P&L
                buy_cost = buy_trade['quantity'] * buy_trade['price'] * (1 + buy_trade.get('fee_rate', 0))
                sell_proceeds = trade['quantity'] * trade['price'] * (1 - trade.get('fee_rate', 0))
                pnl = sell_proceeds - buy_cost
                pnl_pct = pnl / buy_cost
                
                # Calculate holding period
                buy_ts = buy_trade['timestamp']
                sell_ts = trade['timestamp']
                
                if isinstance(buy_ts, (int, float)):
                    buy_time = pd.to_datetime(buy_ts, unit='s')
                else:
                    buy_time = pd.to_datetime(buy_ts)
                    
                if isinstance(sell_ts, (int, float)):
                    sell_time = pd.to_datetime(sell_ts, unit='s')
                else:
                    sell_time = pd.to_datetime(sell_ts)
                
                holding_period = sell_time - buy_time
                
                trade_pnls.append({
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'holding_period': holding_period
                })
                
                holding_periods.append(holding_period)
        
        if not trade_pnls:
            return None
        
        # Calculate metrics
        winning_trades = sum(1 for tp in trade_pnls if tp['pnl'] > 0)
        total_trades = len(trade_pnls)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        returns = [tp['pnl_pct'] for tp in trade_pnls]
        pnls = [tp['pnl'] for tp in trade_pnls]
        
        # Compound returns
        total_return = 1.0
        for ret in returns:
            total_return *= (1 + ret)
        total_return = (total_return - 1) * 100
        
        avg_return = np.mean(returns) * 100 if returns else 0
        total_pnl = sum(pnls)
        
        # Sharpe ratio
        sharpe = 0
        if len(returns) > 1:
            returns_series = pd.Series(returns)
            std_return = returns_series.std()
            if std_return > 0:
                sharpe = returns_series.mean() / std_return
        
        # Max drawdown (simplified)
        cumulative = 1.0
        peak = 1.0
        max_dd = 0.0
        
        for ret in returns:
            cumulative *= (1 + ret)
            peak = max(peak, cumulative)
            dd = (cumulative - peak) / peak
            max_dd = min(max_dd, dd)
        
        # Average holding period
        avg_hold = pd.Timedelta(np.mean(holding_periods)) if holding_periods else timedelta(0)
        avg_hold_str = str(avg_hold).split('.')[0] if avg_hold != timedelta(0) else "N/A"
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_return': avg_return,
            'total_pnl': total_pnl,
            'sharpe': sharpe,
            'max_dd': max_dd * 100,
            'avg_hold': avg_hold_str
        }
    
    def create_regime_performance_table(self, 
                                      analysis_results: Dict[str, Any]) -> pd.DataFrame:
        """Create a formatted performance table by regime"""
        
        regime_metrics = analysis_results.get("regime_metrics", {})
        
        if not regime_metrics:
            return pd.DataFrame()
        
        # Create data for table
        table_data = []
        
        for regime_name, metrics in regime_metrics.items():
            if metrics:
                row = {
                    "Regime": regime_name,
                    "Time %": f"{metrics.regime_percentage:.1f}%",
                    "Trades": metrics.total_trades,
                    "Win Rate": f"{metrics.win_rate*100:.1f}%",
                    "Total Return": f"{metrics.total_return:.2f}%",
                    "Avg Return": f"{metrics.average_return:.2f}%",
                    "Total P&L": f"₩{metrics.total_pnl:,.0f}",
                    "Sharpe": f"{metrics.sharpe_ratio:.2f}",
                    "Sortino": f"{metrics.sortino_ratio:.2f}" if not np.isnan(metrics.sortino_ratio) and not np.isinf(metrics.sortino_ratio) else "N/A",
                    "Max DD": f"{metrics.max_drawdown*100:.2f}%",
                    "Avg Hold": str(metrics.avg_holding_period).split('.')[0] if metrics.avg_holding_period != timedelta(0) else "N/A"
                }
                table_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Sort by Sharpe ratio
        df = df.sort_values("Sharpe", ascending=False)
        
        return df
    
    def export_regime_analysis(self,
                             analysis_results: Dict[str, Any],
                             filepath: str):
        """Export regime analysis results to file"""
        
        # Convert complex objects to serializable format
        export_data = {
            "summary": analysis_results["summary"],
            "regime_metrics": {},
            "transition_metrics": []
        }
        
        # Convert regime metrics
        for regime_name, metrics in analysis_results["regime_metrics"].items():
            if metrics:
                export_data["regime_metrics"][regime_name] = {
                    "regime": metrics.regime.value,
                    "total_trades": metrics.total_trades,
                    "winning_trades": metrics.winning_trades,
                    "losing_trades": metrics.losing_trades,
                    "win_rate": metrics.win_rate,
                    "total_return": metrics.total_return,
                    "average_return": metrics.average_return,
                    "total_pnl": metrics.total_pnl,
                    "average_pnl": metrics.average_pnl,
                    "max_win": metrics.max_win,
                    "max_loss": metrics.max_loss,
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "sortino_ratio": metrics.sortino_ratio,
                    "max_drawdown": metrics.max_drawdown,
                    "avg_holding_period": str(metrics.avg_holding_period),
                    "total_duration": str(metrics.total_duration),
                    "regime_percentage": metrics.regime_percentage
                }
        
        # Convert transition metrics
        for transition in analysis_results["transition_metrics"]:
            export_data["transition_metrics"].append({
                "from_regime": transition.from_regime.value,
                "to_regime": transition.to_regime.value,
                "transition_count": transition.transition_count,
                "avg_return_before": transition.avg_return_before,
                "avg_return_after": transition.avg_return_after,
                "success_rate": transition.success_rate
            })
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        # Also save the performance table as CSV
        table = self.create_regime_performance_table(analysis_results)
        if not table.empty:
            csv_filepath = filepath.replace('.json', '_table.csv')
            table.to_csv(csv_filepath, index=False)
            
            # Also save as formatted text
            txt_filepath = filepath.replace('.json', '_table.txt')
            with open(txt_filepath, 'w') as f:
                f.write("="*80 + "\n")
                f.write("PERFORMANCE BY MARKET REGIME\n")
                f.write("="*80 + "\n\n")
                f.write(table.to_string(index=False))
                f.write("\n\n" + "="*80 + "\n")