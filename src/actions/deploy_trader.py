#!/usr/bin/env python3
"""
Deploy Trader Module
Handles trade execution for the deployment system with simulation capabilities
"""

import json
import time
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from dataclasses import dataclass

try:
    from .upbit import UpbitAPI, UpbitTrader
    from .dynamic_risk_management import DynamicRiskManager
    from .var_risk_management import VaRRiskManager
    from .market_regime import MarketRegime
except ImportError:
    from upbit import UpbitAPI, UpbitTrader
    from dynamic_risk_management import DynamicRiskManager
    from var_risk_management import VaRRiskManager
    from market_regime import MarketRegime


@dataclass
class OrderResult:
    """Order execution result"""
    success: bool
    market: str
    side: str  # buy, sell
    order_type: str  # market, limit, stop_loss, take_profit
    amount: float
    executed_price: float
    order_id: str
    timestamp: datetime
    fee: float
    slippage: float
    message: str


class DeployTrader:
    """
    Handles trade execution for the deployment system
    Supports both simulation and real trading modes
    """
    
    def __init__(self, 
                 upbit_trader: Optional[UpbitTrader],
                 dynamic_risk_manager: DynamicRiskManager,
                 var_risk_manager: VaRRiskManager,
                 simulation_mode: bool = True,
                 config: Optional[Dict] = None):
        """Initialize the trader"""
        self.upbit_trader = upbit_trader
        self.dynamic_risk_manager = dynamic_risk_manager
        self.var_risk_manager = var_risk_manager
        self.simulation_mode = simulation_mode
        
        # Set default config if not provided
        self.config = config or {
            'real_trading': False,
            'update_interval_seconds': 60,
            'risk_check_interval_minutes': 15,
            'max_slippage_pct': 0.5,
            'max_trades_per_day': 10,
            'max_position_size_pct': 20.0,
            'max_total_risk_pct': 50.0,
            'simulation_initial_balance': 10000000,
            'fee_rate': 0.05,  # 0.05% fee rate
        }
        
        # Initialize logger
        self.logger = logging.getLogger('quant_alpaca_trader')
        
        # Initialize simulation balance
        self.simulation_balance = self.config.get('simulation_initial_balance', 10000000)
        self.simulation_positions = {}
        self.simulation_orders = []
        
        # Log initialization
        mode = "SIMULATION" if self.simulation_mode else "REAL TRADING"
        self.logger.info(f"DeployTrader initialized in {mode} mode")
    
    def execute_buy(self, 
                   market: str, 
                   price: float, 
                   position_size_pct: float,
                   regime: MarketRegime) -> Dict[str, Any]:
        """
        Execute a buy order
        
        Args:
            market: Market symbol (e.g., KRW-BTC)
            price: Current market price
            position_size_pct: Position size as a percentage of available balance
            regime: Current market regime
            
        Returns:
            Dictionary with order result
        """
        try:
            # Calculate order amount
            if self.simulation_mode:
                available_balance = self.simulation_balance
            else:
                if self.upbit_trader is None:
                    return self._create_error_result(market, "Buy failed: Upbit trader not initialized")
                available_balance = self.upbit_trader.get_balance('KRW')
            
            # Calculate position size in KRW
            position_size_krw = available_balance * (position_size_pct / 100.0)
            
            # Apply minimum order size (5000 KRW)
            if position_size_krw < 5000:
                return self._create_error_result(
                    market, 
                    f"Buy failed: Position size too small ({position_size_krw:,.2f} KRW < 5,000 KRW minimum)"
                )
            
            # Calculate asset amount
            fee_rate = self.config.get('fee_rate', 0.05) / 100.0  # Convert to decimal
            amount = (position_size_krw * (1 - fee_rate)) / price
            
            # Ensure sufficient balance
            if position_size_krw > available_balance:
                return self._create_error_result(
                    market,
                    f"Buy failed: Insufficient balance ({position_size_krw:,.2f} KRW > {available_balance:,.2f} KRW)"
                )
            
            # Execute order based on mode
            if self.simulation_mode:
                return self._execute_simulated_buy(
                    market=market,
                    price=price,
                    amount=amount,
                    position_size_krw=position_size_krw,
                    regime=regime
                )
            else:
                return self._execute_real_buy(
                    market=market,
                    price=price,
                    position_size_krw=position_size_krw,
                    regime=regime
                )
                
        except Exception as e:
            return self._create_error_result(market, f"Buy execution error: {str(e)}")
    
    def _execute_simulated_buy(self, 
                              market: str, 
                              price: float, 
                              amount: float,
                              position_size_krw: float,
                              regime: MarketRegime) -> Dict[str, Any]:
        """Execute a simulated buy order"""
        # Apply random slippage
        slippage_pct = self._generate_slippage(regime=regime, side="buy")
        executed_price = price * (1 + slippage_pct / 100.0)
        
        # Recalculate amount based on executed price
        fee_rate = self.config.get('fee_rate', 0.05) / 100.0
        amount = (position_size_krw * (1 - fee_rate)) / executed_price
        
        # Calculate fee
        fee_amount = position_size_krw * fee_rate
        
        # Update simulation balance
        self.simulation_balance -= (position_size_krw)
        
        # Record position
        self.simulation_positions[market] = {
            'amount': amount,
            'entry_price': executed_price,
            'entry_time': datetime.now()
        }
        
        # Generate order ID
        order_id = f"sim_buy_{market}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        # Record order
        order_record = {
            'order_id': order_id,
            'market': market,
            'side': 'buy',
            'price': executed_price,
            'amount': amount,
            'total': position_size_krw,
            'fee': fee_amount,
            'timestamp': datetime.now().isoformat()
        }
        self.simulation_orders.append(order_record)
        
        # Log the order
        self.logger.info(f"SIMULATION BUY: {market} - {amount:.8f} @ {executed_price:,.2f} KRW (Slippage: {slippage_pct:+.3f}%)")
        
        return {
            'success': True,
            'market': market,
            'order_id': order_id,
            'amount': amount,
            'executed_price': executed_price,
            'total_price': position_size_krw,
            'fee': fee_amount,
            'slippage': slippage_pct,
            'timestamp': datetime.now(),
            'message': f"Simulated buy executed: {amount:.8f} {market.split('-')[1]} @ {executed_price:,.2f} KRW"
        }
    
    def _execute_real_buy(self, 
                         market: str, 
                         price: float,
                         position_size_krw: float,
                         regime: MarketRegime) -> Dict[str, Any]:
        """Execute a real buy order through the Upbit API"""
        if self.upbit_trader is None:
            return self._create_error_result(market, "Buy failed: Upbit trader not initialized")
        
        try:
            # Execute market order
            # Convert position size to string
            position_size_str = str(position_size_krw)
            
            # Execute market buy
            order_result = self.upbit_trader.buy_market_order(
                market=market,
                price=position_size_str
            )
            
            # Check if order was successful
            if 'uuid' not in order_result:
                return self._create_error_result(
                    market,
                    f"Buy failed: {order_result.get('error', {}).get('message', 'Unknown error')}"
                )
            
            # Wait for order to complete
            order_details = self._wait_for_order_completion(order_result['uuid'], max_wait_seconds=10)
            
            if not order_details:
                return self._create_error_result(
                    market,
                    "Buy failed: Could not retrieve order details after execution"
                )
            
            # Extract execution details
            executed_price = float(order_details.get('price', price))
            amount = float(order_details.get('executed_volume', position_size_krw / price))
            fee = float(order_details.get('paid_fee', 0))
            
            # Calculate slippage
            slippage_pct = ((executed_price - price) / price) * 100 if price > 0 else 0
            
            # Log the order
            self.logger.info(f"REAL BUY: {market} - {amount:.8f} @ {executed_price:,.2f} KRW (Slippage: {slippage_pct:+.3f}%)")
            
            return {
                'success': True,
                'market': market,
                'order_id': order_result['uuid'],
                'amount': amount,
                'executed_price': executed_price,
                'total_price': amount * executed_price,
                'fee': fee,
                'slippage': slippage_pct,
                'timestamp': datetime.now(),
                'message': f"Buy executed: {amount:.8f} {market.split('-')[1]} @ {executed_price:,.2f} KRW"
            }
            
        except Exception as e:
            return self._create_error_result(market, f"Buy execution error: {str(e)}")
    
    def execute_sell(self, 
                    market: str, 
                    price: float, 
                    amount: float,
                    entry_price: float,
                    order_type: str = 'market') -> Dict[str, Any]:
        """
        Execute a sell order
        
        Args:
            market: Market symbol (e.g., KRW-BTC)
            price: Current market price
            amount: Amount of asset to sell
            entry_price: Original entry price (for calculating profit/loss)
            order_type: Type of sell order (market, stop_loss, take_profit, risk_limit)
            
        Returns:
            Dictionary with order result
        """
        try:
            # Execute order based on mode
            if self.simulation_mode:
                return self._execute_simulated_sell(
                    market=market,
                    price=price,
                    amount=amount,
                    entry_price=entry_price,
                    order_type=order_type
                )
            else:
                return self._execute_real_sell(
                    market=market,
                    price=price,
                    amount=amount,
                    entry_price=entry_price,
                    order_type=order_type
                )
                
        except Exception as e:
            return self._create_error_result(market, f"Sell execution error: {str(e)}")
    
    def _execute_simulated_sell(self, 
                               market: str, 
                               price: float, 
                               amount: float,
                               entry_price: float,
                               order_type: str = 'market') -> Dict[str, Any]:
        """Execute a simulated sell order"""
        # Check if position exists
        if market not in self.simulation_positions:
            return self._create_error_result(
                market, 
                f"Sell failed: No position in {market}"
            )
        
        # Get position details
        position = self.simulation_positions[market]
        position_amount = position['amount']
        
        # Ensure sufficient amount
        if amount > position_amount:
            amount = position_amount
        
        # Determine slippage based on order type
        if order_type == 'market':
            # Standard market order slippage
            slippage_pct = self._generate_slippage(regime=MarketRegime.UNKNOWN, side="sell")
        elif order_type == 'stop_loss':
            # Higher slippage for stop-loss
            slippage_pct = self._generate_slippage(
                regime=MarketRegime.VOLATILE, 
                side="sell",
                base_multiplier=1.5
            )
        elif order_type == 'take_profit':
            # Lower slippage for take-profit (limit order)
            slippage_pct = self._generate_slippage(
                regime=MarketRegime.SIDEWAYS, 
                side="sell", 
                base_multiplier=0.5
            )
        else:
            # Default slippage
            slippage_pct = self._generate_slippage(regime=MarketRegime.UNKNOWN, side="sell")
        
        # Apply slippage to execution price
        executed_price = price * (1 + slippage_pct / 100.0)
        
        # Calculate total value and fee
        total_value = executed_price * amount
        fee_rate = self.config.get('fee_rate', 0.05) / 100.0
        fee_amount = total_value * fee_rate
        net_value = total_value - fee_amount
        
        # Calculate profit/loss
        profit_loss = (executed_price - entry_price) * amount
        
        # Update simulation balance
        self.simulation_balance += net_value
        
        # Update position
        if amount >= position_amount:
            # Close position completely
            del self.simulation_positions[market]
        else:
            # Reduce position
            self.simulation_positions[market]['amount'] -= amount
        
        # Generate order ID
        order_id = f"sim_sell_{market}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        # Record order
        order_record = {
            'order_id': order_id,
            'market': market,
            'side': 'sell',
            'price': executed_price,
            'amount': amount,
            'total': total_value,
            'fee': fee_amount,
            'timestamp': datetime.now().isoformat(),
            'profit_loss': profit_loss,
            'order_type': order_type
        }
        self.simulation_orders.append(order_record)
        
        # Log the order
        self.logger.info(
            f"SIMULATION SELL ({order_type}): {market} - {amount:.8f} @ {executed_price:,.2f} KRW "
            f"(P&L: {profit_loss:+,.2f} KRW, Slippage: {slippage_pct:+.3f}%)"
        )
        
        return {
            'success': True,
            'market': market,
            'order_id': order_id,
            'amount': amount,
            'executed_price': executed_price,
            'total_price': total_value,
            'fee': fee_amount,
            'slippage': slippage_pct,
            'profit_loss': profit_loss,
            'profit_loss_pct': ((executed_price / entry_price) - 1) * 100,
            'timestamp': datetime.now(),
            'message': f"Simulated sell executed: {amount:.8f} {market.split('-')[1]} @ {executed_price:,.2f} KRW"
        }
    
    def _execute_real_sell(self, 
                          market: str, 
                          price: float,
                          amount: float,
                          entry_price: float,
                          order_type: str = 'market') -> Dict[str, Any]:
        """Execute a real sell order through the Upbit API"""
        if self.upbit_trader is None:
            return self._create_error_result(market, "Sell failed: Upbit trader not initialized")
        
        try:
            # Convert amount to string for API
            amount_str = str(amount)
            
            # Execute market sell
            order_result = self.upbit_trader.sell_market_order(
                market=market,
                volume=amount_str
            )
            
            # Check if order was successful
            if 'uuid' not in order_result:
                return self._create_error_result(
                    market,
                    f"Sell failed: {order_result.get('error', {}).get('message', 'Unknown error')}"
                )
            
            # Wait for order to complete
            order_details = self._wait_for_order_completion(order_result['uuid'], max_wait_seconds=10)
            
            if not order_details:
                return self._create_error_result(
                    market,
                    "Sell failed: Could not retrieve order details after execution"
                )
            
            # Extract execution details
            executed_price = float(order_details.get('price', price))
            executed_amount = float(order_details.get('executed_volume', amount))
            fee = float(order_details.get('paid_fee', 0))
            total_price = executed_price * executed_amount
            
            # Calculate slippage and profit/loss
            slippage_pct = ((executed_price - price) / price) * 100 if price > 0 else 0
            profit_loss = (executed_price - entry_price) * executed_amount
            
            # Log the order
            self.logger.info(
                f"REAL SELL ({order_type}): {market} - {executed_amount:.8f} @ {executed_price:,.2f} KRW "
                f"(P&L: {profit_loss:+,.2f} KRW, Slippage: {slippage_pct:+.3f}%)"
            )
            
            return {
                'success': True,
                'market': market,
                'order_id': order_result['uuid'],
                'amount': executed_amount,
                'executed_price': executed_price,
                'total_price': total_price,
                'fee': fee,
                'slippage': slippage_pct,
                'profit_loss': profit_loss,
                'profit_loss_pct': ((executed_price / entry_price) - 1) * 100,
                'timestamp': datetime.now(),
                'message': f"Sell executed: {executed_amount:.8f} {market.split('-')[1]} @ {executed_price:,.2f} KRW"
            }
            
        except Exception as e:
            return self._create_error_result(market, f"Sell execution error: {str(e)}")
    
    def _wait_for_order_completion(self, uuid: str, max_wait_seconds: int = 10) -> Dict[str, Any]:
        """Wait for an order to complete and return details"""
        if self.upbit_trader is None:
            return {}
        
        start_time = time.time()
        
        while (time.time() - start_time) < max_wait_seconds:
            try:
                # Get order details
                order_details = self.upbit_trader.api._make_request('GET', '/v1/order', {'uuid': uuid})
                
                # Check if order is completed
                if order_details.get('state') == 'done' or order_details.get('state') == 'cancel':
                    return order_details
                
                # Wait before checking again
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error checking order status: {e}")
                time.sleep(1)
        
        # Return last known details if timed out
        try:
            return self.upbit_trader.api._make_request('GET', '/v1/order', {'uuid': uuid})
        except:
            return {}
    
    def _generate_slippage(self, 
                         regime: MarketRegime, 
                         side: str = "buy",
                         base_multiplier: float = 1.0) -> float:
        """
        Generate realistic slippage based on market regime
        
        Args:
            regime: Market regime
            side: Order side (buy or sell)
            base_multiplier: Multiplier for base slippage
            
        Returns:
            Slippage percentage (positive for price increase, negative for price decrease)
        """
        # Base slippage parameters
        max_slippage = self.config.get('max_slippage_pct', 0.5) * base_multiplier
        
        # Adjust based on regime
        if regime == MarketRegime.VOLATILE:
            max_slippage *= 2.0
        elif regime == MarketRegime.TRENDING_UP:
            if side == "buy":
                max_slippage *= 1.5
            else:
                max_slippage *= 0.8
        elif regime == MarketRegime.TRENDING_DOWN:
            if side == "buy":
                max_slippage *= 0.8
            else:
                max_slippage *= 1.5
        elif regime == MarketRegime.SIDEWAYS:
            max_slippage *= 0.7
        
        # Generate random slippage within range
        # For buys: positive slippage means buying at a higher price (worse)
        # For sells: negative slippage means selling at a lower price (worse)
        if side == "buy":
            slippage = np.random.uniform(0, max_slippage)
        else:  # sell
            slippage = np.random.uniform(-max_slippage, 0)
        
        return slippage
    
    def _create_error_result(self, market: str, message: str) -> Dict[str, Any]:
        """Create an error result dictionary"""
        self.logger.error(f"ORDER ERROR: {market} - {message}")
        
        return {
            'success': False,
            'market': market,
            'message': message,
            'timestamp': datetime.now()
        }
    
    def get_simulation_balance(self) -> float:
        """Get current simulation balance"""
        return self.simulation_balance
    
    def get_simulation_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all simulation positions"""
        return self.simulation_positions
    
    def get_simulation_orders(self) -> List[Dict[str, Any]]:
        """Get all simulation orders"""
        return self.simulation_orders
    
    def get_simulation_pnl(self) -> Dict[str, float]:
        """Calculate overall profit/loss for simulation"""
        initial_balance = self.config.get('simulation_initial_balance', 10000000)
        current_balance = self.simulation_balance
        
        # Add value of open positions
        position_value = 0
        for market, position in self.simulation_positions.items():
            # Skip if not a valid position
            if 'amount' not in position or 'entry_price' not in position:
                continue
            position_value += position['amount'] * position['entry_price']
        
        total_value = current_balance + position_value
        
        return {
            'initial_balance': initial_balance,
            'current_balance': current_balance,
            'position_value': position_value,
            'total_value': total_value,
            'absolute_pnl': total_value - initial_balance,
            'percentage_pnl': ((total_value / initial_balance) - 1) * 100 if initial_balance > 0 else 0
        }