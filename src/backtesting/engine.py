import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import logging
import sys
import os

# Add actions directory to path for strategy imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'actions'))
from strategies import get_strategy


class BacktestEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.initial_balance = config['backtesting']['initial_balance']
        self.fees = config['backtesting']['fees']
        self.slippage = config['backtesting']['slippage']
        self.max_positions = config['backtesting']['max_positions']
        self.order_type = config['backtesting']['order_type']
        
        # Portfolio state
        self.cash = self.initial_balance
        self.positions = {}  # {market: quantity}
        self.portfolio_value_history = []
        self.trade_history = []
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config['execution']['log_level']))
        self.logger = logging.getLogger(__name__)
        
        # Initialize strategy
        strategy_config = config['strategy']
        self.strategy = get_strategy(strategy_config['name'], strategy_config['parameters'])
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using strategy"""
        return self.strategy.calculate_indicators(df)
    
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate buy/sell signals using strategy"""
        return self.strategy.generate_signals(df, market)
    
    def get_market_type(self, market: str) -> str:
        """Determine market type (krw, btc, usdt) from market string"""
        if market.startswith('KRW-'):
            return 'krw_market'
        elif market.startswith('BTC-'):
            return 'btc_market'
        elif market.startswith('USDT-'):
            return 'usdt_market'
        else:
            return 'krw_market'  # Default to KRW market
    
    def get_trading_costs(self, market: str) -> tuple:
        """Get fee rate and slippage for the market and order type"""
        market_type = self.get_market_type(market)
        
        if self.order_type == 'limit':
            fee_rate = self.fees['limit_order'][market_type]
            slippage = self.slippage['limit_order']
        else:  # market order
            fee_rate = self.fees['market_order'][market_type]
            slippage = self.slippage['market_order']
        
        return fee_rate, slippage
    
    def execute_trade(self, market: str, signal: int, price: float, timestamp: datetime):
        """Execute a trade based on signal"""
        fee_rate, slippage_rate = self.get_trading_costs(market)
        
        if signal == 1:  # Buy
            if len(self.positions) < self.max_positions:
                # Calculate position size (equal weight)
                position_value = self.cash * 0.9 / self.max_positions  # Leave 10% cash buffer
                
                # Apply slippage and fees
                execution_price = price * (1 + slippage_rate)
                quantity = position_value / execution_price
                cost = quantity * execution_price * (1 + fee_rate)
                
                if cost <= self.cash:
                    self.cash -= cost
                    self.positions[market] = self.positions.get(market, 0) + quantity
                    
                    self.trade_history.append({
                        'timestamp': timestamp,
                        'market': market,
                        'side': 'buy',
                        'quantity': quantity,
                        'price': execution_price,
                        'cost': cost,
                        'cash_after': self.cash,
                        'fee_rate': fee_rate,
                        'slippage': slippage_rate,
                        'order_type': self.order_type
                    })
                    
                    self.logger.info(f"BUY {market}: {quantity:.6f} at {execution_price:.2f} (fee: {fee_rate*100:.3f}%)")
        
        elif signal == -1 and market in self.positions and self.positions[market] > 0:  # Sell
            quantity = self.positions[market]
            
            # Apply slippage and fees
            execution_price = price * (1 - slippage_rate)
            proceeds = quantity * execution_price * (1 - fee_rate)
            
            self.cash += proceeds
            self.positions[market] = 0
            
            self.trade_history.append({
                'timestamp': timestamp,
                'market': market,
                'side': 'sell',
                'quantity': quantity,
                'price': execution_price,
                'proceeds': proceeds,
                'cash_after': self.cash,
                'fee_rate': fee_rate,
                'slippage': slippage_rate,
                'order_type': self.order_type
            })
            
            self.logger.info(f"SELL {market}: {quantity:.6f} at {execution_price:.2f} (fee: {fee_rate*100:.3f}%)")
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        portfolio_value = self.cash
        
        for market, quantity in self.positions.items():
            if quantity > 0 and market in current_prices:
                portfolio_value += quantity * current_prices[market]
        
        return portfolio_value
    
    def run_backtest(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run the backtest on provided data"""
        self.logger.info("Starting backtest...")
        
        # Prepare data with indicators
        prepared_data = {}
        for market, df in data.items():
            df_copy = df.copy()
            df_copy = self.calculate_indicators(df_copy)
            df_copy = self.generate_signals(df_copy, market)
            prepared_data[market] = df_copy
        
        # Get all timestamps and sort them
        all_timestamps = set()
        for df in prepared_data.values():
            all_timestamps.update(df.index)
        
        sorted_timestamps = sorted(list(all_timestamps))
        
        # Run backtest
        for timestamp in sorted_timestamps:
            current_prices = {}
            
            # Process each market at this timestamp
            for market, df in prepared_data.items():
                if timestamp in df.index:
                    row = df.loc[timestamp]
                    current_prices[market] = row['trade_price']
                    
                    # Execute trade if signal exists
                    if not pd.isna(row['signal']) and row['signal'] != 0:
                        self.execute_trade(market, int(row['signal']), row['trade_price'], timestamp)
            
            # Record portfolio value
            portfolio_value = self.calculate_portfolio_value(current_prices)
            self.portfolio_value_history.append({
                'timestamp': timestamp,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'positions': dict(self.positions)
            })
        
        # Calculate results
        results = self.calculate_results()
        self.logger.info("Backtest completed!")
        
        return results
    
    def calculate_results(self) -> Dict[str, Any]:
        """Calculate backtest performance metrics"""
        if not self.portfolio_value_history:
            return {}
        
        portfolio_values = [pv['portfolio_value'] for pv in self.portfolio_value_history]
        
        final_value = portfolio_values[-1]
        total_return = (final_value - self.initial_balance) / self.initial_balance
        
        # Calculate daily returns
        daily_returns = pd.Series(portfolio_values).pct_change().dropna()
        
        # Risk metrics
        volatility = daily_returns.std() * np.sqrt(365 * 24 * 60)  # Annualized for minute data
        sharpe_ratio = (daily_returns.mean() * 365 * 24 * 60) / volatility if volatility > 0 else 0
        
        # Drawdown
        portfolio_series = pd.Series(portfolio_values)
        running_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - running_max) / running_max
        max_drawdown = drawdown.min()
        
        results = {
            'initial_balance': self.initial_balance,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'total_trades': len(self.trade_history),
            'portfolio_history': self.portfolio_value_history,
            'trade_history': self.trade_history
        }
        
        return results