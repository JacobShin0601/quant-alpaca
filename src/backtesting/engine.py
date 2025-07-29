import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from strategies import get_strategy
from actions.dynamic_risk_management import DynamicRiskManager, OrderType, RiskParameters
from actions.market_regime import MarketRegimeDetector, MarketRegime
from actions.signal_strength import SignalStrengthCalculator
from actions.garch_position_sizing import GARCHPositionSizer, PositionSizeResult


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
        self.positions = {}  # {market: {'quantity': float, 'entry_price': float, 'entry_time': datetime}}
        self.portfolio_value_history = []
        self.trade_history = []
        self.stop_orders = {}  # {market: {'stop_loss': float, 'take_profit': [float], 'trailing_stop': float}}
        self.position_risk_params = {}  # {market: RiskParameters}
        self.last_trade_time = {}  # {market: datetime} - Track last trade time per market
        self.min_trade_interval = timedelta(minutes=30)  # Minimum 30 minutes between trades
        
        # Position sizing
        position_sizing_config = config.get('position_sizing', {})
        self.max_position_pct = position_sizing_config.get('max_position_pct', 0.2)  # Max 20% per position
        self.use_dynamic_sizing = position_sizing_config.get('use_dynamic_sizing', True)
        self.signal_strength_calc = SignalStrengthCalculator(position_sizing_config.get('signal_strength', {}))
        
        # GARCH position sizing
        self.use_garch_sizing = position_sizing_config.get('use_garch_sizing', False)
        self.garch_sizer = GARCHPositionSizer(position_sizing_config.get('garch_config', {}))
        
        # Risk management
        self.risk_manager = DynamicRiskManager(config.get('risk_management'))
        self.regime_detector = MarketRegimeDetector(config.get('regime_config'))
        
        # Logging control
        self.trade_count = 0
        self.max_log_trades = 10  # Only log first 10 trades
        self.total_timestamps = 0
        self.processed_timestamps = 0
        
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
    
    def execute_trade(self, market: str, signal: int, price: float, timestamp: datetime, 
                     df: pd.DataFrame = None, regime: MarketRegime = None):
        """Execute a trade based on signal with dynamic risk management"""
        fee_rate, slippage_rate = self.get_trading_costs(market)
        
        # Check stop orders first for existing positions
        if market in self.positions and market in self.stop_orders:
            self._check_stop_orders(market, price, timestamp)
        
        # Check if enough time has passed since last trade
        if market in self.last_trade_time:
            time_since_last_trade = timestamp - self.last_trade_time[market]
            if time_since_last_trade < self.min_trade_interval:
                return  # Skip trade if too soon
        
        if signal == 1:  # Buy
            if len(self.positions) < self.max_positions:
                # Calculate base position size (max 20% of initial balance)
                base_position_value = self.initial_balance * self.max_position_pct
                
                # Use GARCH position sizing if enabled
                if self.use_garch_sizing and df is not None:
                    # Get returns for GARCH model
                    returns = df['trade_price'].pct_change().dropna()
                    
                    # Get market data for adjustments
                    # Get current prices from positions and current market
                    current_prices = {market: price}
                    for pos_market in self.positions:
                        current_prices[pos_market] = self.positions[pos_market].get('entry_price', 0)
                    
                    market_data = {
                        'market_correlation': self._calculate_market_correlation(market, current_prices),
                        'current_drawdown': self._calculate_current_drawdown(market)
                    }
                    
                    # Calculate GARCH-based position size
                    garch_result = self.garch_sizer.calculate_position_size(
                        market=market,
                        returns=returns,
                        current_price=price,
                        market_data=market_data,
                        base_position=base_position_value
                    )
                    
                    position_value = garch_result.final_position
                    signal_strength = 0.6 + (garch_result.kelly_fraction * 0.8)  # Convert Kelly to signal strength
                    
                    # Log GARCH sizing details for first few trades
                    if self.trade_count < 5:
                        self.logger.info(f"GARCH Position Sizing for {market}:")
                        self.logger.info(f"  Predicted Volatility: {garch_result.predicted_volatility:.2%}")
                        self.logger.info(f"  Kelly Fraction: {garch_result.kelly_fraction:.2%}")
                        self.logger.info(f"  Final Position: {position_value:,.0f} ({garch_result.position_adjustment_reason})")
                
                # Calculate signal strength if dynamic sizing is enabled (non-GARCH)
                elif self.use_dynamic_sizing and df is not None:
                    signal_strength = self.signal_strength_calc.calculate_signal_strength(df, 'buy')
                    position_multiplier = self.signal_strength_calc.get_position_size_multiplier(signal_strength)
                    position_value = base_position_value * position_multiplier
                else:
                    position_value = base_position_value
                    signal_strength = 0.6  # Default neutral strength
                
                # Additional check: don't use more than 80% of available cash
                max_cash_use = self.cash * 0.8
                position_value = min(position_value, max_cash_use)
                
                if position_value < self.initial_balance * 0.01:
                    return  # Position too small
                
                # Apply slippage and fees
                execution_price = price * (1 + slippage_rate)
                quantity = position_value / execution_price
                cost = quantity * execution_price * (1 + fee_rate)
                
                if cost <= self.cash and cost > self.initial_balance * 0.001:  # Min position size 0.1% of initial
                    self.cash -= cost
                    
                    # Update position with entry details
                    self.positions[market] = {
                        'quantity': self.positions.get(market, {}).get('quantity', 0) + quantity,
                        'entry_price': execution_price,
                        'entry_time': timestamp,
                        'signal_strength': signal_strength if self.use_dynamic_sizing else 1.0
                    }
                    
                    # Update last trade time
                    self.last_trade_time[market] = timestamp
                    
                    # Calculate and set risk parameters
                    if regime is None and df is not None:
                        regime, _ = self.regime_detector.detect_regime(df)
                    elif regime is None:
                        regime = MarketRegime.UNKNOWN
                    
                    volatility_data = self._calculate_volatility_data(df, market) if df is not None else {}
                    
                    risk_params = self.risk_manager.calculate_position_risk_parameters(
                        entry_price=execution_price,
                        position_size=quantity,
                        regime=regime,
                        volatility_data=volatility_data
                    )
                    
                    # Store risk parameters
                    self.position_risk_params[market] = risk_params
                    self.risk_manager.add_position(market, risk_params)
                    
                    # Set initial stop orders
                    self._set_stop_orders(market, risk_params)
                    
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
                        'order_type': self.order_type,
                        'signal_strength': signal_strength if self.use_dynamic_sizing else 1.0,
                        'position_pct': (position_value / self.initial_balance) * 100
                    })
                    
                    # Convert timestamp to string format
                    if hasattr(timestamp, 'strftime'):
                        ts_str = timestamp.strftime('%Y-%m-%d %H:%M')
                    else:
                        ts_str = pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M')
                    self.trade_count += 1
                    if self.trade_count <= self.max_log_trades:
                        self.logger.info(f"[{ts_str}] BUY {market}: {quantity:.6f} at {execution_price:.2f} (fee: {fee_rate*100:.3f}%)")
                    
                    # Update last trade time
                    self.last_trade_time[market] = timestamp
        
        elif signal == -1 and market in self.positions:  # Sell
            # Check if position exists and has quantity
            if isinstance(self.positions[market], dict):
                has_quantity = self.positions[market]['quantity'] > 0
            else:
                has_quantity = self.positions[market] > 0
                
            if has_quantity:
                # Get quantity from position
                if isinstance(self.positions[market], dict):
                    quantity = self.positions[market]['quantity']
                else:
                    quantity = self.positions[market]
                
                # Apply slippage and fees
                execution_price = price * (1 - slippage_rate)
                proceeds = quantity * execution_price * (1 - fee_rate)
                
                self.cash += proceeds
                del self.positions[market]  # Remove position completely
                
                # Update last trade time
                self.last_trade_time[market] = timestamp
                
                # Remove stop orders
                if market in self.stop_orders:
                    del self.stop_orders[market]
                if market in self.position_risk_params:
                    del self.position_risk_params[market]
                    self.risk_manager.remove_position(market)
                
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
                
                # Convert timestamp to string format
                if hasattr(timestamp, 'strftime'):
                    ts_str = timestamp.strftime('%Y-%m-%d %H:%M')
                else:
                    ts_str = pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M')
                self.trade_count += 1
                if self.trade_count <= self.max_log_trades:
                    self.logger.info(f"[{ts_str}] SELL {market}: {quantity:.6f} at {execution_price:.2f} (fee: {fee_rate*100:.3f}%)")
                
                # Update last trade time
                self.last_trade_time[market] = timestamp
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        portfolio_value = self.cash
        
        for market, position in self.positions.items():
            if isinstance(position, dict):
                quantity = position['quantity']
            else:
                quantity = position
                
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
            # Generate initial signals
            df_copy = self.generate_signals(df_copy, market)
            prepared_data[market] = df_copy
        
        # Get all timestamps and sort them
        all_timestamps = set()
        for df in prepared_data.values():
            all_timestamps.update(df.index)
        
        sorted_timestamps = sorted(list(all_timestamps))
        self.total_timestamps = len(sorted_timestamps)
        
        # Run backtest
        last_progress = -1
        for i, timestamp in enumerate(sorted_timestamps):
            self.processed_timestamps = i + 1
            
            # Show progress every 10%
            progress = int((i / self.total_timestamps) * 100)
            if progress % 10 == 0 and progress != last_progress:
                if self.trade_count > self.max_log_trades:
                    # Count active positions
                    active_positions = 0
                    for position in self.positions.values():
                        if isinstance(position, dict):
                            if position['quantity'] > 0:
                                active_positions += 1
                        else:
                            if position > 0:
                                active_positions += 1
                    self.logger.info(f"Progress: {progress}% - Trades: {self.trade_count}, Cash: ₩{self.cash:,.0f}, Positions: {active_positions}")
                last_progress = progress
            
            current_prices = {}
            
            # Process each market at this timestamp
            for market, df in prepared_data.items():
                if timestamp in df.index:
                    row = df.loc[timestamp]
                    current_prices[market] = row['trade_price']
                    
                    # Check stop orders for existing positions first
                    if market in self.positions and market in self.stop_orders:
                        self._check_stop_orders(market, row['trade_price'], timestamp)
                    
                    # Check position status
                    has_position = market in self.positions
                    if has_position:
                        if isinstance(self.positions[market], dict):
                            has_position = self.positions[market]['quantity'] > 0
                        else:
                            has_position = self.positions[market] > 0
                    
                    # Get signal from pre-calculated signals
                    signal = row['signal']
                    
                    # Apply position-aware filtering
                    if signal == 1 and has_position:
                        # Already have position, skip buy signal
                        signal = 0
                    elif signal == -1 and not has_position:
                        # No position to sell, skip sell signal  
                        signal = 0
                    
                    # Execute trade if valid signal
                    if signal != 0:
                        # Get current dataframe for regime detection
                        current_df = prepared_data[market].loc[:timestamp]
                        self.execute_trade(market, int(signal), row['trade_price'], timestamp, current_df)
            
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
        
        # Final summary
        if self.trade_count > self.max_log_trades:
            self.logger.info(f"... ({self.trade_count - self.max_log_trades} more trades not shown)")
        
        self.logger.info(f"Backtest completed! Total trades: {self.trade_count}")
        
        return results
    
    def calculate_results(self) -> Dict[str, Any]:
        """Calculate backtest performance metrics"""
        if not self.portfolio_value_history:
            return {}
        
        portfolio_values = [pv['portfolio_value'] for pv in self.portfolio_value_history]
        
        final_value = portfolio_values[-1]
        total_return = (final_value - self.initial_balance) / self.initial_balance
        
        # Calculate returns per timestamp (minute-level)
        minute_returns = pd.Series(portfolio_values).pct_change().dropna()
        
        # Risk metrics - properly annualize minute-level data
        # Assume 365 days * 24 hours * 60 minutes = 525,600 minutes per year
        minutes_per_year = 365 * 24 * 60
        
        # Annualized volatility
        volatility = minute_returns.std() * np.sqrt(minutes_per_year)
        
        # Annualized return
        mean_minute_return = minute_returns.mean()
        annualized_return_from_minutes = (1 + mean_minute_return) ** minutes_per_year - 1
        
        # Sharpe ratio
        sharpe_ratio = annualized_return_from_minutes / volatility if volatility > 0 else 0
        
        # Sortino Ratio (downside deviation)
        downside_returns = minute_returns[minute_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(minutes_per_year) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return_from_minutes / downside_deviation if downside_deviation > 0 else 0
        
        # Drawdown
        portfolio_series = pd.Series(portfolio_values)
        running_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar Ratio (annualized return / max drawdown)
        # Use the already calculated annualized return
        calmar_ratio = abs(annualized_return_from_minutes / max_drawdown) if max_drawdown < 0 else 0
        
        results = {
            'initial_balance': self.initial_balance,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'total_trades': len(self.trade_history),
            'portfolio_history': self.portfolio_value_history,
            'trade_history': self.trade_history
        }
        
        return results
    
    def _calculate_market_correlation(self, market: str, current_prices: Dict[str, float]) -> float:
        """Calculate correlation of market with overall portfolio"""
        # Simple implementation - can be enhanced
        if len(self.positions) <= 1:
            return 0.0
        
        # For now, return a default moderate correlation
        # In production, would calculate actual rolling correlation
        return 0.5
    
    def _calculate_current_drawdown(self, market: str) -> float:
        """Calculate current drawdown for the market"""
        if not self.portfolio_value_history:
            return 0.0
        
        # Calculate portfolio drawdown
        recent_values = [pv['portfolio_value'] for pv in self.portfolio_value_history[-50:]]
        if len(recent_values) < 2:
            return 0.0
        
        max_value = max(recent_values)
        current_value = recent_values[-1]
        drawdown = (current_value - max_value) / max_value if max_value > 0 else 0
        
        return drawdown
    
    def _check_stop_orders(self, market: str, current_price: float, timestamp: datetime):
        """Check and execute stop orders"""
        if market not in self.stop_orders or market not in self.positions:
            return
        
        position = self.positions.get(market)
        if not position:
            return
        if isinstance(position, dict):
            quantity = position['quantity']
            entry_price = position['entry_price']
        else:
            # Legacy format
            quantity = position
            entry_price = None
        
        if quantity <= 0:
            return
        
        orders = self.stop_orders[market]
        risk_params = self.position_risk_params.get(market)
        
        # Update trailing stop if applicable
        if risk_params:
            market_data = {
                'position_age_minutes': (timestamp - position.get('entry_time', timestamp)).total_seconds() / 60
            }
            updates = self.risk_manager.update_position_stops(market, current_price, market_data)
            
            if updates.get('trailing_stop_updated'):
                orders['trailing_stop'] = updates['new_trailing_stop']
            if updates.get('breakeven_stop_set') or updates.get('time_based_tightening'):
                orders['stop_loss'] = updates['new_stop_loss']
        
        # Check stop loss (including trailing stop)
        effective_stop = orders.get('trailing_stop', 0)
        if orders.get('stop_loss', 0) > effective_stop:
            effective_stop = orders['stop_loss']
        
        if effective_stop > 0 and current_price <= effective_stop:
            # Execute stop loss
            self._execute_stop_order(market, quantity, current_price, timestamp, 'stop_loss')
            return
        
        # Check take profit levels
        if 'take_profit' in orders:
            for tp_level in orders['take_profit']:
                if current_price >= tp_level['price'] and tp_level['remaining_size'] > 0:
                    # Execute partial take profit
                    exit_quantity = min(tp_level['size'], quantity)
                    self._execute_stop_order(market, exit_quantity, current_price, timestamp, 'take_profit')
                    tp_level['remaining_size'] -= exit_quantity
                    
                    # Update position quantity
                    if isinstance(self.positions[market], dict):
                        self.positions[market]['quantity'] -= exit_quantity
                    else:
                        self.positions[market] -= exit_quantity
                    
                    if self.positions[market].get('quantity', 0) <= 0:
                        break
    
    def _set_stop_orders(self, market: str, risk_params: RiskParameters):
        """Set initial stop orders for a position"""
        orders = {
            'stop_loss': risk_params.stop_loss_price,
            'take_profit': []
        }
        
        # Get partial exit configuration
        regime_config = self.risk_manager.config["regime_parameters"][risk_params.regime.value]
        tp_config = regime_config["take_profit"]
        
        if "partial_exits" in tp_config:
            for exit in tp_config["partial_exits"]:
                orders['take_profit'].append({
                    'price': risk_params.entry_price * (1 + exit["pct"]),
                    'size': risk_params.position_size * exit["size"],
                    'remaining_size': risk_params.position_size * exit["size"]
                })
        
        self.stop_orders[market] = orders
    
    def _execute_stop_order(self, market: str, quantity: float, price: float, 
                           timestamp: datetime, order_type: str):
        """Execute a stop order"""
        fee_rate, slippage_rate = self.get_trading_costs(market)
        
        # Adjust slippage for stop orders (usually worse execution)
        if order_type == 'stop_loss':
            slippage_rate *= 1.5  # 50% worse slippage for stops
        
        execution_price = price * (1 - slippage_rate)
        proceeds = quantity * execution_price * (1 - fee_rate)
        
        self.cash += proceeds
        
        # Update or remove position
        if isinstance(self.positions[market], dict):
            self.positions[market]['quantity'] -= quantity
            if self.positions[market]['quantity'] <= 0:
                del self.positions[market]
                del self.stop_orders[market]
                self.risk_manager.remove_position(market)
        else:
            # Legacy format
            self.positions[market] -= quantity
            if self.positions[market] <= 0:
                del self.positions[market]
                del self.stop_orders[market]
        
        # Log the stop order execution
        self.trade_history.append({
            'timestamp': timestamp,
            'market': market,
            'side': 'sell',
            'action': f'{order_type}_sell',
            'quantity': quantity,
            'price': execution_price,
            'proceeds': proceeds,
            'cash_after': self.cash,
            'fee_rate': fee_rate,
            'slippage': slippage_rate,
            'order_type': order_type
        })
        
        if self.trade_count <= self.max_log_trades:
            self.logger.info(f"[{timestamp}] {order_type.upper()} {market}: {quantity:.6f} at {execution_price:.2f}")
    
    def _calculate_volatility_data(self, df: pd.DataFrame, market: str) -> Dict:
        """Calculate volatility data for risk management"""
        if df is None or len(df) < 20:
            return {}
        
        # Calculate ATR
        high_low = df['high_price'] - df['low_price']
        high_close = abs(df['high_price'] - df['trade_price'].shift(1))
        low_close = abs(df['low_price'] - df['trade_price'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean().iloc[-1]
        
        # Calculate volatility ratio
        returns = df['trade_price'].pct_change()
        current_vol = returns.rolling(window=20).std().iloc[-1]
        avg_vol = returns.rolling(window=50).std().mean()
        volatility_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        
        # Get other indicators if available
        momentum = None
        rsi = None
        
        if 'momentum' in df.columns:
            momentum = df['momentum'].iloc[-1]
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
        
        return {
            'atr': atr,
            'volatility_ratio': volatility_ratio,
            'momentum': momentum,
            'rsi': rsi
        }