#!/usr/bin/env python3
"""
Order Management System for Quant-Alpaca
Comprehensive order lifecycle management with real-time monitoring
"""

import json
import time
import uuid
import threading
import queue
import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import traceback

try:
    from .upbit import UpbitAPI, UpbitTrader
    from .market_regime import MarketRegime
except ImportError:
    from upbit import UpbitAPI, UpbitTrader
    from market_regime import MarketRegime


class OrderState(Enum):
    """Order states following Upbit API specification"""
    PENDING = "wait"           # 대기 (미체결)
    WATCH = "watch"           # 예약주문 대기
    DONE = "done"             # 전체 체결완료
    CANCEL = "cancel"         # 주문 취소
    PARTIAL = "partial"       # 부분 체결
    ERROR = "error"           # 오류 상태


class OrderType(Enum):
    """Order types following Upbit API specification"""
    LIMIT = "limit"           # 지정가 주문
    PRICE = "price"           # 시장가 주문 (매수)
    MARKET = "market"         # 시장가 주문 (매도)
    BEST = "best"             # 최유리 주문
    LIMIT_IOC = "limit_ioc"   # 지정가 IOC 주문
    LIMIT_FOK = "limit_fok"   # 지정가 FOK 주문
    BEST_IOC = "best_ioc"     # 최유리 IOC 주문
    BEST_FOK = "best_fok"     # 최유리 FOK 주문


@dataclass
class OrderInfo:
    """Comprehensive order information"""
    order_id: str
    identifier: str  # User-defined unique identifier
    market: str
    side: str  # "bid" or "ask"
    ord_type: OrderType
    price: Optional[float]
    volume: Optional[float]
    executed_volume: float = 0.0
    remaining_volume: float = 0.0
    state: OrderState = OrderState.PENDING
    created_at: datetime = None
    updated_at: datetime = None
    trades: List[Dict] = None
    paid_fee: float = 0.0
    locked: float = 0.0
    reserved_fee: float = 0.0
    remaining_fee: float = 0.0
    avg_price: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.trades is None:
            self.trades = []


@dataclass
class OrderChance:
    """Order possibility information"""
    market: str
    bid_fee: float  # 매수 수수료율
    ask_fee: float  # 매도 수수료율
    maker_bid_fee: float
    maker_ask_fee: float
    market_state: str
    isTradingSuspended: bool
    max_total: str
    min_total: str
    bid_account: Dict
    ask_account: Dict
    supported_ord_types: List[str]


class OrderManager:
    """
    Comprehensive order management system
    Handles order lifecycle, monitoring, and real-time updates
    """
    
    def __init__(self, upbit_trader: UpbitTrader, config: Optional[Dict] = None):
        """Initialize order manager"""
        self.upbit_trader = upbit_trader
        self.config = config or {}
        
        # Order tracking
        self.active_orders: Dict[str, OrderInfo] = {}  # order_id -> OrderInfo
        self.order_history: List[OrderInfo] = []
        self.pending_orders: Dict[str, OrderInfo] = {}  # identifier -> OrderInfo
        
        # Threading and queues
        self.order_queue = queue.Queue()
        self.monitor_queue = queue.Queue()
        self.running = False
        self.monitor_thread = None
        self.processor_thread = None
        
        # Callbacks
        self.order_callbacks: Dict[str, List[Callable]] = {
            'on_order_created': [],
            'on_order_filled': [],
            'on_order_partially_filled': [],
            'on_order_cancelled': [],
            'on_order_failed': []
        }
        
        # Configuration
        self.monitor_interval = self.config.get('monitor_interval_seconds', 1)
        self.max_retries = self.config.get('max_order_retries', 3)
        self.retry_delay = self.config.get('retry_delay_seconds', 1)
        
        # Logger
        self.logger = logging.getLogger('order_manager')
        
        # Order possibility cache
        self.order_chances_cache: Dict[str, OrderChance] = {}
        self.cache_expiry = {}
        
    def start(self):
        """Start order management system"""
        if self.running:
            return
            
        self.running = True
        self.logger.info("Starting Order Management System")
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_orders,
            daemon=True
        )
        self.monitor_thread.start()
        
        # Start order processor thread
        self.processor_thread = threading.Thread(
            target=self._process_orders,
            daemon=True
        )
        self.processor_thread.start()
        
    def stop(self):
        """Stop order management system"""
        self.logger.info("Stopping Order Management System")
        self.running = False
        
        # Wait for threads to complete
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=5)
    
    def get_order_chance(self, market: str, force_refresh: bool = False) -> Optional[OrderChance]:
        """
        Get order possibility information for a market
        
        Args:
            market: Market symbol (e.g., KRW-BTC)
            force_refresh: Force refresh cached data
            
        Returns:
            OrderChance object or None if failed
        """
        try:
            # Check cache first
            if not force_refresh and market in self.order_chances_cache:
                cache_time = self.cache_expiry.get(market, datetime.min)
                if datetime.now() - cache_time < timedelta(minutes=5):
                    return self.order_chances_cache[market]
            
            # Fetch from API
            response = self.upbit_trader.api._make_request('GET', '/v1/orders/chance', {'market': market})
            
            # Parse response
            order_chance = OrderChance(
                market=response['market']['id'],
                bid_fee=float(response['bid_fee']),
                ask_fee=float(response['ask_fee']),
                maker_bid_fee=float(response['maker_bid_fee']),
                maker_ask_fee=float(response['maker_ask_fee']),
                market_state=response['market']['state'],
                isTradingSuspended=response['market']['trade_suspension'],
                max_total=response['market']['max_total'],
                min_total=response['market']['min_total'],
                bid_account=response['bid_account'],
                ask_account=response['ask_account'],
                supported_ord_types=response['market'].get('supported_ord_types', [])
            )
            
            # Cache the result
            self.order_chances_cache[market] = order_chance
            self.cache_expiry[market] = datetime.now()
            
            return order_chance
            
        except Exception as e:
            self.logger.error(f"Failed to get order chance for {market}: {e}")
            return None
    
    def validate_order(self, market: str, side: str, ord_type: OrderType, 
                      volume: Optional[float] = None, price: Optional[float] = None) -> Dict[str, Any]:
        """
        Validate order before placing
        
        Returns:
            Dictionary with 'valid' bool and 'message' string
        """
        try:
            # Get order chance
            chance = self.get_order_chance(market)
            if not chance:
                return {'valid': False, 'message': 'Unable to get market information'}
            
            # Check if trading is suspended
            if chance.isTradingSuspended:
                return {'valid': False, 'message': f'Trading suspended for {market}'}
            
            # Check order type support
            if ord_type.value not in chance.supported_ord_types:
                return {'valid': False, 'message': f'Order type {ord_type.value} not supported for {market}'}
            
            # Check minimum order amount
            min_total = float(chance.min_total)
            max_total = float(chance.max_total) if chance.max_total != 'null' else float('inf')
            
            if side == 'bid':  # Buy order
                if ord_type in [OrderType.PRICE] and price:
                    if price < min_total:
                        return {'valid': False, 'message': f'Order amount {price:,.0f} KRW below minimum {min_total:,.0f} KRW'}
                    if price > max_total:
                        return {'valid': False, 'message': f'Order amount {price:,.0f} KRW above maximum {max_total:,.0f} KRW'}
                
                # Check balance
                bid_balance = float(chance.bid_account['balance'])
                if price and price > bid_balance:
                    return {'valid': False, 'message': f'Insufficient KRW balance: {bid_balance:,.0f} < {price:,.0f}'}
                    
            else:  # Sell order
                if volume:
                    ask_balance = float(chance.ask_account['balance'])
                    if volume > ask_balance:
                        return {'valid': False, 'message': f'Insufficient {market.split("-")[1]} balance: {ask_balance:.8f} < {volume:.8f}'}
            
            return {'valid': True, 'message': 'Order validation passed'}
            
        except Exception as e:
            return {'valid': False, 'message': f'Validation error: {str(e)}'}
    
    def place_order(self, market: str, side: str, ord_type: OrderType,
                   volume: Optional[float] = None, price: Optional[float] = None,
                   identifier: Optional[str] = None) -> Dict[str, Any]:
        """
        Place an order with comprehensive validation and tracking
        
        Args:
            market: Market symbol
            side: "bid" or "ask"
            ord_type: Order type
            volume: Order volume (for limit/market sell orders)
            price: Order price (for limit/market buy orders)
            identifier: Custom identifier for tracking
            
        Returns:
            Dictionary with order result
        """
        try:
            # Generate identifier if not provided
            if not identifier:
                identifier = f"{market}_{side}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            
            # Validate order
            validation = self.validate_order(market, side, ord_type, volume, price)
            if not validation['valid']:
                return {
                    'success': False,
                    'message': validation['message'],
                    'identifier': identifier
                }
            
            # Prepare order parameters
            params = {
                'market': market,
                'side': side,
                'ord_type': ord_type.value,
                'identifier': identifier
            }
            
            if volume is not None:
                params['volume'] = str(volume)
            if price is not None:
                params['price'] = str(price)
            
            # Place order through API
            response = self.upbit_trader.api._make_request('POST', '/v1/orders', params)
            
            # Create order info
            order_info = OrderInfo(
                order_id=response['uuid'],
                identifier=identifier,
                market=market,
                side=side,
                ord_type=ord_type,
                price=float(response['price']) if response.get('price') else None,
                volume=float(response['volume']) if response.get('volume') else None,
                remaining_volume=float(response['remaining_volume']) if response.get('remaining_volume') else 0.0,
                state=OrderState(response['state']),
                locked=float(response.get('locked', 0)),
                reserved_fee=float(response.get('reserved_fee', 0)),
                remaining_fee=float(response.get('remaining_fee', 0))
            )
            
            # Track the order
            self.active_orders[order_info.order_id] = order_info
            self.pending_orders[identifier] = order_info
            
            # Trigger callback
            self._trigger_callback('on_order_created', order_info)
            
            self.logger.info(f"Order placed: {identifier} ({order_info.order_id}) - {market} {side} {ord_type.value}")
            
            return {
                'success': True,
                'order_id': order_info.order_id,
                'identifier': identifier,
                'message': 'Order placed successfully'
            }
            
        except Exception as e:
            error_msg = f"Failed to place order: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'identifier': identifier
            }
    
    def cancel_order(self, order_id: str = None, identifier: str = None) -> Dict[str, Any]:
        """
        Cancel an order by order_id or identifier
        
        Args:
            order_id: Upbit order UUID
            identifier: Custom identifier
            
        Returns:
            Dictionary with cancellation result
        """
        try:
            if not order_id and not identifier:
                return {'success': False, 'message': 'Either order_id or identifier must be provided'}
            
            # Find order info
            order_info = None
            if order_id and order_id in self.active_orders:
                order_info = self.active_orders[order_id]
            elif identifier and identifier in self.pending_orders:
                order_info = self.pending_orders[identifier]
                order_id = order_info.order_id
            
            if not order_info:
                return {'success': False, 'message': 'Order not found'}
            
            # Cancel through API
            params = {}
            if order_id:
                params['uuid'] = order_id
            elif identifier:
                params['identifier'] = identifier
            
            response = self.upbit_trader.api._make_request('DELETE', '/v1/order', params)
            
            # Update order state
            order_info.state = OrderState.CANCEL
            order_info.updated_at = datetime.now()
            
            # Move to history
            self._move_to_history(order_info)
            
            # Trigger callback
            self._trigger_callback('on_order_cancelled', order_info)
            
            self.logger.info(f"Order cancelled: {order_info.identifier} ({order_info.order_id})")
            
            return {
                'success': True,
                'order_id': response['uuid'],
                'message': 'Order cancelled successfully'
            }
            
        except Exception as e:
            error_msg = f"Failed to cancel order: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def get_order_status(self, order_id: str = None, identifier: str = None) -> Optional[OrderInfo]:
        """Get current order status"""
        try:
            # Check local tracking first
            if order_id and order_id in self.active_orders:
                return self.active_orders[order_id]
            elif identifier and identifier in self.pending_orders:
                return self.pending_orders[identifier]
            
            # Query API
            params = {}
            if order_id:
                params['uuid'] = order_id
            elif identifier:
                params['identifier'] = identifier
            else:
                return None
            
            response = self.upbit_trader.api._make_request('GET', '/v1/order', params)
            
            # Update order info
            order_info = self._parse_order_response(response)
            
            # Update tracking
            if order_info.order_id in self.active_orders:
                self.active_orders[order_info.order_id] = order_info
            if order_info.identifier in self.pending_orders:
                self.pending_orders[order_info.identifier] = order_info
            
            return order_info
            
        except Exception as e:
            self.logger.error(f"Failed to get order status: {e}")
            return None
    
    def get_open_orders(self, market: str = None, state: str = 'wait') -> List[OrderInfo]:
        """Get list of open orders"""
        try:
            params = {'state': state}
            if market:
                params['market'] = market
            
            response = self.upbit_trader.api._make_request('GET', '/v1/orders/open', params)
            
            orders = []
            for order_data in response:
                order_info = self._parse_order_response(order_data)
                orders.append(order_info)
                
                # Update tracking
                self.active_orders[order_info.order_id] = order_info
                if order_info.identifier:
                    self.pending_orders[order_info.identifier] = order_info
            
            return orders
            
        except Exception as e:
            self.logger.error(f"Failed to get open orders: {e}")
            return []
    
    def get_closed_orders(self, market: str = None, state: str = 'done', 
                         limit: int = 100, page: int = 1) -> List[OrderInfo]:
        """Get list of closed orders"""
        try:
            params = {
                'state': state,
                'limit': limit,
                'page': page
            }
            if market:
                params['market'] = market
            
            response = self.upbit_trader.api._make_request('GET', '/v1/orders/closed', params)
            
            orders = []
            for order_data in response:
                order_info = self._parse_order_response(order_data)
                orders.append(order_info)
            
            return orders
            
        except Exception as e:
            self.logger.error(f"Failed to get closed orders: {e}")
            return []
    
    def _parse_order_response(self, response: Dict) -> OrderInfo:
        """Parse API response into OrderInfo"""
        try:
            # Parse trades
            trades = []
            if 'trades' in response and response['trades']:
                trades = response['trades']
            
            order_info = OrderInfo(
                order_id=response['uuid'],
                identifier=response.get('identifier', ''),
                market=response['market'],
                side=response['side'],
                ord_type=OrderType(response['ord_type']),
                price=float(response['price']) if response.get('price') else None,
                volume=float(response['volume']) if response.get('volume') else None,
                executed_volume=float(response.get('executed_volume', 0)),
                remaining_volume=float(response.get('remaining_volume', 0)),
                state=OrderState(response['state']),
                trades=trades,
                paid_fee=float(response.get('paid_fee', 0)),
                locked=float(response.get('locked', 0)),
                reserved_fee=float(response.get('reserved_fee', 0)),
                remaining_fee=float(response.get('remaining_fee', 0))
            )
            
            # Calculate average price if trades exist
            if trades:
                total_volume = sum(float(trade['volume']) for trade in trades)
                total_funds = sum(float(trade['funds']) for trade in trades)
                if total_volume > 0:
                    order_info.avg_price = total_funds / total_volume
            
            return order_info
            
        except Exception as e:
            self.logger.error(f"Failed to parse order response: {e}")
            raise
    
    def _monitor_orders(self):
        """Background thread to monitor active orders"""
        self.logger.info("Order monitoring started")
        
        while self.running:
            try:
                # Get all active orders
                active_order_ids = list(self.active_orders.keys())
                
                for order_id in active_order_ids:
                    if not self.running:
                        break
                    
                    try:
                        # Get current status
                        order_info = self.get_order_status(order_id=order_id)
                        if not order_info:
                            continue
                        
                        # Check for state changes
                        old_order = self.active_orders.get(order_id)
                        if old_order and old_order.state != order_info.state:
                            self._handle_order_state_change(old_order, order_info)
                        
                    except Exception as e:
                        self.logger.error(f"Error monitoring order {order_id}: {e}")
                
                # Sleep between monitoring cycles
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Error in order monitoring: {e}")
                time.sleep(5)
        
        self.logger.info("Order monitoring stopped")
    
    def _handle_order_state_change(self, old_order: OrderInfo, new_order: OrderInfo):
        """Handle order state changes"""
        self.logger.info(f"Order state change: {old_order.identifier} {old_order.state.value} -> {new_order.state.value}")
        
        # Update tracking
        self.active_orders[new_order.order_id] = new_order
        if new_order.identifier in self.pending_orders:
            self.pending_orders[new_order.identifier] = new_order
        
        # Handle specific state changes
        if new_order.state == OrderState.DONE:
            self._trigger_callback('on_order_filled', new_order)
            self._move_to_history(new_order)
            
        elif new_order.state == OrderState.PARTIAL:
            self._trigger_callback('on_order_partially_filled', new_order)
            
        elif new_order.state == OrderState.CANCEL:
            self._trigger_callback('on_order_cancelled', new_order)
            self._move_to_history(new_order)
            
        elif new_order.state == OrderState.ERROR:
            self._trigger_callback('on_order_failed', new_order)
            self._move_to_history(new_order)
    
    def _move_to_history(self, order_info: OrderInfo):
        """Move order from active tracking to history"""
        # Remove from active tracking
        if order_info.order_id in self.active_orders:
            del self.active_orders[order_info.order_id]
        if order_info.identifier in self.pending_orders:
            del self.pending_orders[order_info.identifier]
        
        # Add to history
        self.order_history.append(order_info)
        
        # Limit history size
        max_history = self.config.get('max_order_history', 1000)
        if len(self.order_history) > max_history:
            self.order_history = self.order_history[-max_history:]
    
    def _process_orders(self):
        """Background thread to process order queue"""
        self.logger.info("Order processor started")
        
        while self.running:
            try:
                # Process queued orders
                try:
                    order_request = self.order_queue.get(timeout=1)
                    self._process_single_order(order_request)
                except queue.Empty:
                    continue
                    
            except Exception as e:
                self.logger.error(f"Error in order processor: {e}")
                time.sleep(1)
        
        self.logger.info("Order processor stopped")
    
    def _process_single_order(self, order_request: Dict):
        """Process a single order request with retry logic"""
        retries = 0
        max_retries = self.max_retries
        
        while retries < max_retries and self.running:
            try:
                result = self.place_order(**order_request)
                if result['success']:
                    break
                else:
                    self.logger.warning(f"Order failed (attempt {retries + 1}): {result['message']}")
                    
            except Exception as e:
                self.logger.error(f"Order error (attempt {retries + 1}): {e}")
            
            retries += 1
            if retries < max_retries:
                time.sleep(self.retry_delay * retries)  # Exponential backoff
    
    def _trigger_callback(self, event: str, order_info: OrderInfo):
        """Trigger callbacks for order events"""
        callbacks = self.order_callbacks.get(event, [])
        for callback in callbacks:
            try:
                callback(order_info)
            except Exception as e:
                self.logger.error(f"Callback error for {event}: {e}")
    
    def add_callback(self, event: str, callback: Callable):
        """Add callback for order events"""
        if event in self.order_callbacks:
            self.order_callbacks[event].append(callback)
    
    def remove_callback(self, event: str, callback: Callable):
        """Remove callback for order events"""
        if event in self.order_callbacks and callback in self.order_callbacks[event]:
            self.order_callbacks[event].remove(callback)
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary including active orders"""
        active_count = len(self.active_orders)
        total_locked = sum(order.locked for order in self.active_orders.values())
        
        return {
            'active_orders': active_count,
            'total_locked_krw': total_locked,
            'pending_orders_by_market': self._group_orders_by_market(),
            'order_history_count': len(self.order_history)
        }
    
    def _group_orders_by_market(self) -> Dict[str, int]:
        """Group pending orders by market"""
        market_counts = {}
        for order in self.active_orders.values():
            market = order.market
            market_counts[market] = market_counts.get(market, 0) + 1
        return market_counts
    
    def export_order_history(self, filename: str = None) -> str:
        """Export order history to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"order_history_{timestamp}.json"
        
        history_data = []
        for order in self.order_history:
            order_dict = asdict(order)
            # Convert datetime objects to ISO strings
            order_dict['created_at'] = order.created_at.isoformat() if order.created_at else None
            order_dict['updated_at'] = order.updated_at.isoformat() if order.updated_at else None
            history_data.append(order_dict)
        
        with open(filename, 'w') as f:
            json.dump(history_data, f, indent=4, default=str)
        
        self.logger.info(f"Order history exported to {filename}")
        return filename


# High-level convenience functions
def create_buy_order(order_manager: OrderManager, market: str, amount_krw: float, 
                    ord_type: OrderType = OrderType.PRICE) -> Dict[str, Any]:
    """Create a buy order with KRW amount"""
    return order_manager.place_order(
        market=market,
        side='bid',
        ord_type=ord_type,
        price=amount_krw
    )


def create_sell_order(order_manager: OrderManager, market: str, volume: float,
                     ord_type: OrderType = OrderType.MARKET) -> Dict[str, Any]:
    """Create a sell order with asset volume"""
    return order_manager.place_order(
        market=market,
        side='ask',
        ord_type=ord_type,
        volume=volume
    )


def create_limit_buy_order(order_manager: OrderManager, market: str, volume: float, 
                          price: float) -> Dict[str, Any]:
    """Create a limit buy order"""
    return order_manager.place_order(
        market=market,
        side='bid',
        ord_type=OrderType.LIMIT,
        volume=volume,
        price=price
    )


def create_limit_sell_order(order_manager: OrderManager, market: str, volume: float,
                           price: float) -> Dict[str, Any]:
    """Create a limit sell order"""
    return order_manager.place_order(
        market=market,
        side='ask',
        ord_type=OrderType.LIMIT,
        volume=volume,
        price=price
    )