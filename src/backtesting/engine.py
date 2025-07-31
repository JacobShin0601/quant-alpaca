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
from actions.var_risk_management import VaRRiskManager, VaRMethod, VaRResult, RiskLimitStatus
from actions.regime_performance_analyzer import RegimePerformanceAnalyzer


class BacktestEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.initial_balance = config['backtesting']['initial_balance']
        self.fees = config['backtesting']['fees']
        self.slippage = config['backtesting']['slippage']
        self.max_positions = config['backtesting']['max_positions']
        self.order_type = config['backtesting']['order_type']
        
        # Store strategy name for trade history
        self.strategy_name = config.get('strategy', {}).get('name', 'unknown')
        
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
        garch_config = position_sizing_config.get('garch_config', {})
        
        # Create GARCH position sizer - it will use default config if not provided
        # The sizer expects a complete config structure, so we let it use its defaults
        # and only override specific values from our config
        if self.use_garch_sizing:
            # Initialize with default config
            self.garch_sizer = GARCHPositionSizer()
            
            # Override specific GARCH parameters if provided
            if garch_config:
                # Update GARCH model parameters
                if 'lookback_period' in garch_config:
                    self.garch_sizer.config['garch']['min_observations'] = garch_config['lookback_period']
                if 'update_frequency' in garch_config:
                    self.garch_sizer.config['garch']['refit_frequency'] = garch_config['update_frequency']
                
                # Update position sizing parameters
                if 'vol_target' in garch_config:
                    self.garch_sizer.config['position_sizing']['target_volatility'] = garch_config['vol_target']
                if 'leverage_limit' in garch_config:
                    self.garch_sizer.config['position_sizing']['max_leverage'] = garch_config['leverage_limit']
        else:
            # Create a dummy sizer that won't be used
            self.garch_sizer = None
        
        # Risk management
        self.risk_manager = DynamicRiskManager(config.get('risk_management'))
        self.regime_detector = MarketRegimeDetector(config.get('regime_config'))
        
        # VaR risk management
        self.var_manager = VaRRiskManager(config.get('var_risk_management'))
        self.use_var_limits = config.get('var_risk_management', {}).get('enabled', False)
        self.var_check_frequency = timedelta(minutes=config.get('var_risk_management', {}).get('check_frequency_minutes', 60))
        self.last_var_check = None
        self.current_var_result = None
        self.daily_pnl = 0
        self.start_of_day_value = self.initial_balance
        self.var_limit_breached = False
        self.var_metrics_history = []
        
        # Regime performance analysis
        self.regime_analyzer = RegimePerformanceAnalyzer(config.get('regime_analysis'))
        self.analyze_regime_performance = config.get('analyze_regime_performance', True)
        
        # Logging control
        self.trade_count = 0
        self.max_log_trades = 10  # Only log first 10 trades
        self.total_timestamps = 0
        self.processed_timestamps = 0
        
        # Stop-limit warning control
        self.stop_limit_warning_count = {}  # Track warnings per market
        self.max_stop_limit_warnings = config.get('execution', {}).get('max_stop_limit_warnings', 3)  # Only log first N warnings per market
        
        # Warmup period
        self.warmup_period_minutes = config.get('execution', {}).get('warmup_period_minutes', 0)
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config['execution']['log_level']))
        self.logger = logging.getLogger(__name__)
        
        # Initialize strategy
        strategy_config = config['strategy']
        self.strategy = get_strategy(strategy_config['name'], strategy_config['parameters'])
        
        # Check if strategy has adaptive capabilities
        self.strategy_is_adaptive = hasattr(self.strategy, 'enable_adaptation') and getattr(self.strategy, 'enable_adaptation', False)
        if self.strategy_is_adaptive:
            self.logger.info(f"Using adaptive strategy: {strategy_config['name']}")
        
        # Track adaptive parameter changes
        self.adaptive_parameter_history = []
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using strategy"""
        return self.strategy.calculate_indicators(df)
    
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate buy/sell signals using strategy"""
        # For adaptive strategies, use adaptive signal generation if available
        if self.strategy_is_adaptive and hasattr(self.strategy, 'generate_signals_adaptive'):
            return self.strategy.generate_signals_adaptive(df, market)
        else:
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
        # Check if we're still in warmup period
        if self.warmup_period_minutes > 0:
            # Calculate minutes since start
            if isinstance(timestamp, (int, float)):
                minutes_since_start = timestamp  # Already in minutes
            else:
                # Convert to minutes from start
                if hasattr(self, 'backtest_start_date') and self.backtest_start_date is not None:
                    # Ensure both are datetime objects
                    if isinstance(timestamp, type(self.backtest_start_date)):
                        time_diff = timestamp - self.backtest_start_date
                        minutes_since_start = time_diff.total_seconds() / 60
                    else:
                        # Type mismatch - convert timestamp to same type as backtest_start_date
                        try:
                            timestamp_dt = pd.to_datetime(timestamp)
                            time_diff = timestamp_dt - self.backtest_start_date
                            minutes_since_start = time_diff.total_seconds() / 60
                        except:
                            minutes_since_start = float('inf')  # Can't determine, allow trading
                else:
                    minutes_since_start = float('inf')  # Can't determine, allow trading
            
            # Skip trading if still in warmup period
            if minutes_since_start < self.warmup_period_minutes:
                return
        
        fee_rate, slippage_rate = self.get_trading_costs(market)
        
        # Check stop orders first for existing positions
        if market in self.positions and market in self.stop_orders:
            self._check_stop_orders(market, price, timestamp)
        
        # Check if enough time has passed since last trade
        if market in self.last_trade_time:
            # Handle both datetime and int timestamps
            if isinstance(timestamp, (int, float)):
                current_ts = timestamp
            else:
                current_ts = timestamp.timestamp() if hasattr(timestamp, 'timestamp') else pd.Timestamp(timestamp).timestamp()
            
            last_ts = self.last_trade_time[market]
            if isinstance(last_ts, (int, float)):
                last_trade_ts = last_ts
            else:
                last_trade_ts = last_ts.timestamp() if hasattr(last_ts, 'timestamp') else pd.Timestamp(last_ts).timestamp()
            
            time_since_last_trade = current_ts - last_trade_ts
            min_interval_seconds = self.min_trade_interval.total_seconds()
            
            if time_since_last_trade < min_interval_seconds:
                return  # Skip trade if too soon
        
        # Check VaR limits if enabled
        if self.use_var_limits and self.var_limit_breached:
            # Only allow closing positions when VaR limit is breached
            if signal == 1:  # Trying to buy
                return  # Block new positions
        
        if signal == 1:  # Buy
            # Additional VaR check for new positions
            if self.use_var_limits and self.current_var_result:
                current_portfolio_value = self.calculate_portfolio_value({market: price})
                risk_status = self.var_manager.check_risk_limits(
                    current_portfolio_value,
                    self.daily_pnl,
                    self.positions,
                    self.current_var_result,
                    timestamp
                )
                
                if not risk_status.trading_allowed:
                    return  # Trading not allowed due to risk limits
            
            if len(self.positions) < self.max_positions:
                # Calculate base position size (max 20% of initial balance)
                base_position_value = self.initial_balance * self.max_position_pct
                
                # Use GARCH position sizing if enabled
                if self.use_garch_sizing and self.garch_sizer is not None and df is not None:
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
                    
                    # Calculate and set risk parameters if risk management is enabled
                    risk_management_enabled = self.config.get('risk_management', {}).get('enabled', True)
                    
                    if risk_management_enabled:
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
                    else:
                        # Set empty stop orders when risk management is disabled
                        self.stop_orders[market] = {
                            'stop_loss': 0,
                            'take_profit': []
                        }
                    
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
                        'position_pct': (position_value / self.initial_balance) * 100,
                        'strategy': self.strategy_name
                    })
                    
                    # Convert timestamp to string format
                    ts_str = self._format_timestamp(timestamp)
                    self.trade_count += 1
                    if self.max_log_trades > 0 and self.trade_count <= self.max_log_trades:
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
                    'order_type': self.order_type,
                    'strategy': self.strategy_name
                })
                
                # Convert timestamp to string format
                ts_str = self._format_timestamp(timestamp)
                self.trade_count += 1
                if self.max_log_trades > 0 and self.trade_count <= self.max_log_trades:
                    self.logger.info(f"[{ts_str}] SELL {market}: {quantity:.6f} at {execution_price:.2f} (fee: {fee_rate*100:.3f}%)")
                
                # Update last trade time
                self.last_trade_time[market] = timestamp
    
    def _format_timestamp(self, timestamp) -> str:
        """Convert timestamp to readable format, handling index-based timestamps"""
        if isinstance(timestamp, (int, float)):
            # This is an index-based timestamp (minutes since start)
            if hasattr(self, 'backtest_start_date') and self.backtest_start_date is not None:
                # Calculate actual datetime from start date + minutes
                actual_datetime = self.backtest_start_date + pd.Timedelta(minutes=int(timestamp))
                return actual_datetime.strftime('%Y-%m-%d %H:%M')
            else:
                # Fallback to Unix timestamp handling
                return pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d %H:%M')
        elif hasattr(timestamp, 'strftime'):
            return timestamp.strftime('%Y-%m-%d %H:%M')
        else:
            return pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M')
    
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
        if self.max_log_trades > 0:
            self.logger.info("Starting backtest...")
        
        # Store the start date for timestamp display
        self.backtest_start_date = None
        for df in data.values():
            if len(df) > 0:
                first_timestamp = df.index[0]
                if self.backtest_start_date is None:
                    self.backtest_start_date = first_timestamp
                else:
                    # Handle comparison between different timestamp types
                    if isinstance(first_timestamp, type(self.backtest_start_date)):
                        if first_timestamp < self.backtest_start_date:
                            self.backtest_start_date = first_timestamp
                    elif isinstance(first_timestamp, (int, float)) and not isinstance(self.backtest_start_date, (int, float)):
                        # Convert datetime to comparable format if needed
                        continue
                    elif not isinstance(first_timestamp, (int, float)) and isinstance(self.backtest_start_date, (int, float)):
                        # Prefer datetime over int
                        self.backtest_start_date = first_timestamp
        
        # Log warmup period if configured
        if self.warmup_period_minutes > 0:
            if isinstance(self.backtest_start_date, (int, float)):
                # If start date is stored as minutes index, calculate warmup end in minutes
                warmup_end_minutes = self.warmup_period_minutes
                warmup_end_str = f"minute {warmup_end_minutes}"
            else:
                # If start date is a datetime, calculate actual end time
                warmup_end = self.backtest_start_date + pd.Timedelta(minutes=self.warmup_period_minutes)
                warmup_end_str = warmup_end.strftime('%Y-%m-%d %H:%M')
            if self.max_log_trades > 0:
                self.logger.info(f"Warmup period: {self.warmup_period_minutes} minutes (no trading until {warmup_end_str})")
        
        # Prepare data with indicators
        prepared_data = {}
        for market, df in data.items():
            df_copy = df.copy()
            df_copy = self.calculate_indicators(df_copy)
            # Generate initial signals
            df_copy = self.generate_signals(df_copy, market)
            prepared_data[market] = df_copy
        
        # Store prepared data for regime analysis
        self.prepared_data = prepared_data
        
        # Calculate Buy & Hold benchmark
        self.buy_hold_data = self._calculate_buy_hold_benchmark(data)
        
        # Get all timestamps and sort them
        all_timestamps = set()
        for df in prepared_data.values():
            all_timestamps.update(df.index)
        
        sorted_timestamps = sorted(list(all_timestamps))
        self.total_timestamps = len(sorted_timestamps)
        
        # Initialize VaR tracking
        self.start_of_day_value = self.initial_balance
        last_day = None
        current_prices = {}  # Initialize current_prices
        
        # Run backtest
        last_progress = -1
        for i, timestamp in enumerate(sorted_timestamps):
            self.processed_timestamps = i + 1
            
            # Check if new day for VaR reset
            current_day = timestamp.date() if hasattr(timestamp, 'date') else pd.to_datetime(timestamp).date()
            if last_day is None or current_day > last_day:
                self.start_of_day_value = self.calculate_portfolio_value(current_prices) if current_prices else self.initial_balance
                last_day = current_day
            
            # Show progress every 10%
            progress = int((i / self.total_timestamps) * 100)
            if progress % 10 == 0 and progress != last_progress:
                # Only log progress if not in optimization mode (when max_log_trades > 0)
                if self.max_log_trades > 0:
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
            
            # Calculate daily P&L
            self.daily_pnl = portfolio_value - self.start_of_day_value
            
            # Check VaR limits periodically
            if self.use_var_limits:
                if self.last_var_check is None:
                    should_check_var = True
                else:
                    # Handle both datetime and int timestamps
                    if isinstance(timestamp, (int, float)):
                        current_ts = timestamp
                    else:
                        current_ts = timestamp.timestamp() if hasattr(timestamp, 'timestamp') else pd.Timestamp(timestamp).timestamp()
                    
                    if isinstance(self.last_var_check, (int, float)):
                        last_check_ts = self.last_var_check
                    else:
                        last_check_ts = self.last_var_check.timestamp() if hasattr(self.last_var_check, 'timestamp') else pd.Timestamp(self.last_var_check).timestamp()
                    
                    time_since_check = current_ts - last_check_ts
                    check_frequency_seconds = self.var_check_frequency.total_seconds()
                    should_check_var = time_since_check >= check_frequency_seconds
                
                if should_check_var and len(self.portfolio_value_history) > 20:
                    # Calculate portfolio returns
                    portfolio_values = [pv['portfolio_value'] for pv in self.portfolio_value_history[-252:]]
                    portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
                    
                    if len(portfolio_returns) > 20:
                        # Calculate VaR
                        self.current_var_result = self.var_manager.calculate_var_cvar(
                            portfolio_returns,
                            portfolio_value,
                            VaRMethod.HISTORICAL
                        )
                        
                        # Check risk limits
                        risk_status = self.var_manager.check_risk_limits(
                            portfolio_value,
                            self.daily_pnl,
                            self.positions,
                            self.current_var_result,
                            timestamp
                        )
                        
                        # Handle risk limit breaches
                        if not risk_status.trading_allowed:
                            self.var_limit_breached = True
                            
                            # Close positions if required
                            if risk_status.positions_to_close:
                                for pos_market in risk_status.positions_to_close:
                                    if pos_market in self.positions:
                                        # Force sell signal
                                        self.execute_trade(pos_market, -1, current_prices.get(pos_market, 0), timestamp)
                        else:
                            self.var_limit_breached = False
                        
                        # Update VaR history
                        last_return = 0
                        if len(portfolio_returns) > 0 and not portfolio_returns.empty:
                            last_return = portfolio_returns.iloc[-1] if not pd.isna(portfolio_returns.iloc[-1]) else 0
                        self.var_manager.update_history(
                            last_return,
                            self.current_var_result,
                            timestamp
                        )
                        
                        # Store VaR metrics
                        self.var_metrics_history.append({
                            'timestamp': timestamp,
                            'var_1d': self.current_var_result.var_1d,
                            'cvar_1d': self.current_var_result.cvar_1d,
                            'var_amount': self.current_var_result.var_amount,
                            'cvar_amount': self.current_var_result.cvar_amount,
                            'daily_pnl': self.daily_pnl,
                            'var_utilization': abs(self.daily_pnl) / self.current_var_result.var_amount if self.current_var_result.var_amount > 0 else 0,
                            'limit_breached': self.var_limit_breached
                        })
                        
                        self.last_var_check = timestamp
            
            self.portfolio_value_history.append({
                'timestamp': timestamp,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'positions': dict(self.positions),
                'daily_pnl': self.daily_pnl,
                'var_limit_breached': self.var_limit_breached
            })
        
        # Calculate results
        results = self.calculate_results()
        
        # Final summary
        if self.max_log_trades > 0:
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
        # Calculate actual trading days and minutes
        total_minutes = len(minute_returns)
        if total_minutes > 0:
            # Estimate trading days (assuming data doesn't include all 24/7)
            # Crypto markets trade 24/7, but we'll use a more conservative approach
            # Assume average 1440 minutes per day (24 hours)
            trading_days = total_minutes / 1440
            
            # For annualization, use 252 trading days per year for traditional markets
            # For crypto, use 365 days
            days_per_year = 365
            annualization_factor = days_per_year / max(trading_days, 1)
        else:
            annualization_factor = 1
        
        # Calculate annualized metrics more conservatively
        # Use simple annualized return based on total return and time period
        if trading_days > 0:
            annualized_return = (1 + total_return) ** (annualization_factor) - 1
        else:
            annualized_return = 0
        
        # Annualized volatility
        if len(minute_returns) > 1:
            # Convert minute volatility to daily, then annualize
            minutes_per_day = 1440
            daily_returns = minute_returns.groupby(minute_returns.index // minutes_per_day).sum()
            if len(daily_returns) > 1:
                daily_volatility = daily_returns.std()
                annualized_volatility = daily_volatility * np.sqrt(days_per_year)
            else:
                annualized_volatility = minute_returns.std() * np.sqrt(252 * 6.5 * 60)  # Fallback
        else:
            annualized_volatility = 0
        
        # Sharpe ratio (using risk-free rate of 0 for simplicity)
        risk_free_rate = 0
        if annualized_volatility > 0:
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        else:
            sharpe_ratio = 0 if annualized_return <= 0 else np.sign(annualized_return) * 999
        
        # Sortino Ratio (downside deviation)
        downside_returns = minute_returns[minute_returns < 0]
        if len(downside_returns) > 1:
            # Calculate downside deviation
            downside_daily_returns = downside_returns.groupby(downside_returns.index // 1440).sum()
            if len(downside_daily_returns) > 1:
                downside_deviation = downside_daily_returns.std() * np.sqrt(days_per_year)
            else:
                downside_deviation = downside_returns.std() * np.sqrt(252 * 6.5 * 60)
        else:
            downside_deviation = 0
            
        if downside_deviation > 0:
            sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation
        else:
            sortino_ratio = 0 if annualized_return <= 0 else np.sign(annualized_return) * 999
        
        # Cap extreme values to prevent display issues
        sharpe_ratio = np.clip(sharpe_ratio, -999, 999)
        sortino_ratio = np.clip(sortino_ratio, -999, 999)
        
        volatility = annualized_volatility
        
        # Drawdown
        portfolio_series = pd.Series(portfolio_values)
        running_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar Ratio (annualized return / max drawdown)
        # Use the already calculated annualized return
        if max_drawdown < 0 and abs(max_drawdown) > 0.01:  # Avoid division by very small numbers
            calmar_ratio = annualized_return / abs(max_drawdown)
        else:
            calmar_ratio = 0
        
        # Cap extreme values
        calmar_ratio = np.clip(calmar_ratio, -999, 999)
        
        # Calculate VaR metrics if enabled
        var_metrics = {}
        if self.use_var_limits and self.var_metrics_history:
            # Get VaR summary
            var_summary = self.var_manager.get_risk_metrics_summary()
            
            # Calculate additional VaR statistics
            var_breaches = sum(1 for m in self.var_metrics_history if m.get('daily_pnl', 0) < -m.get('var_amount', 0))
            cvar_breaches = sum(1 for m in self.var_metrics_history if m.get('daily_pnl', 0) < -m.get('cvar_amount', 0))
            
            var_metrics = {
                'var_metrics': {
                    'average_var_1d': var_summary.get('average_var', 0),
                    'average_cvar_1d': var_summary.get('average_cvar', 0),
                    'var_breach_count': var_breaches,
                    'cvar_breach_count': cvar_breaches,
                    'var_breach_rate': var_breaches / len(self.var_metrics_history) if self.var_metrics_history else 0,
                    'cvar_breach_rate': cvar_breaches / len(self.var_metrics_history) if self.var_metrics_history else 0,
                    'max_var_utilization': max([m.get('var_utilization', 0) for m in self.var_metrics_history]) if self.var_metrics_history else 0,
                    'limit_breach_days': sum(1 for m in self.var_metrics_history if m.get('limit_breached', False)),
                    'var_history': self.var_metrics_history
                }
            }
        
        # Analyze regime performance if enabled
        regime_analysis = {}
        if self.analyze_regime_performance and hasattr(self, 'prepared_data'):
            try:
                analysis_results = self.regime_analyzer.analyze_regime_performance(
                    self.portfolio_value_history,
                    self.trade_history,
                    self.prepared_data
                )
                
                # Create performance tables
                regime_table = self.regime_analyzer.create_regime_performance_table(analysis_results)
                detailed_table = self.regime_analyzer.create_detailed_regime_performance_table(
                    analysis_results, self.trade_history
                )
                
                regime_analysis = {
                    'regime_analysis': {
                        'summary': analysis_results['summary'],
                        'performance_table': regime_table.to_dict('records') if not regime_table.empty else [],
                        'detailed_performance_table': detailed_table.to_dict('records') if not detailed_table.empty else [],
                        'regime_metrics': {k: self._serialize_regime_metrics(v) for k, v in analysis_results['regime_metrics'].items()}
                    }
                }
                
                # Print detailed performance table if available
                if not detailed_table.empty:
                    print("\n" + "="*120)
                    print("DETAILED PERFORMANCE BY MARKET REGIME, STRATEGY & MARKET")
                    print("="*120)
                    print(detailed_table.to_string(index=False))
                    print("="*120 + "\n")
                elif not regime_table.empty:
                    # Fall back to simple regime table if detailed not available
                    print("\n" + "="*80)
                    print("PERFORMANCE BY MARKET REGIME")
                    print("="*80)
                    print(regime_table.to_string(index=False))
                    print("="*80 + "\n")
                    
            except Exception as e:
                self.logger.warning(f"Failed to analyze regime performance: {e}")
        
        # Collect adaptive strategy information
        adaptive_info = {}
        if self.strategy_is_adaptive:
            adaptive_info['adaptive_strategy'] = {
                'strategy_name': self.strategy.__class__.__name__,
                'adaptation_enabled': getattr(self.strategy, 'enable_adaptation', False),
                'parameter_history': getattr(self.strategy, 'parameter_history', []),
                'adaptation_status': self.strategy.get_adaptation_status() if hasattr(self.strategy, 'get_adaptation_status') else {},
                'parameter_changes': self.adaptive_parameter_history
            }
            
            # Add enhanced ensemble specific information
            if hasattr(self.strategy, 'strategy_performance'):
                adaptive_info['adaptive_strategy']['strategy_performance'] = {
                    k: v[-10:] if v else [] for k, v in self.strategy.strategy_performance.items()
                }
        
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
            'trade_history': self.trade_history,
            'buy_hold_benchmark': self.buy_hold_data if hasattr(self, 'buy_hold_data') else {},
            **var_metrics,  # Add VaR metrics if available
            **regime_analysis,  # Add regime analysis if available
            **adaptive_info  # Add adaptive strategy information if available
        }
        
        return results
    
    def _calculate_buy_hold_benchmark(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate Buy & Hold benchmark performance"""
        if not data:
            return {'total_return': 0, 'final_value': self.initial_balance}
        
        # For multi-market portfolios, allocate equally among all markets
        allocation_per_market = self.initial_balance / len(data)
        
        buy_hold_values = []
        buy_hold_returns = {}
        
        # Get all timestamps
        all_timestamps = set()
        for df in data.values():
            all_timestamps.update(df.index)
        sorted_timestamps = sorted(list(all_timestamps))
        
        # Initialize buy & hold positions at first timestamp
        initial_prices = {}
        initial_shares = {}
        
        for market, df in data.items():
            if len(df) > 0:
                first_price = df['trade_price'].iloc[0]
                initial_prices[market] = first_price
                # Calculate shares bought with equal allocation
                initial_shares[market] = allocation_per_market / first_price
                buy_hold_returns[market] = []
        
        # Calculate buy & hold value at each timestamp
        for timestamp in sorted_timestamps:
            total_value = 0
            
            for market, df in data.items():
                if timestamp in df.index and market in initial_shares:
                    current_price = df.loc[timestamp]['trade_price']
                    # Value = shares * current price
                    market_value = initial_shares[market] * current_price
                    total_value += market_value
                    
                    # Store individual market return
                    if market in initial_prices:
                        market_return = (current_price / initial_prices[market]) - 1
                        buy_hold_returns[market].append(market_return)
            
            buy_hold_values.append({
                'timestamp': timestamp,
                'portfolio_value': total_value
            })
        
        # Calculate final metrics
        if buy_hold_values:
            final_value = buy_hold_values[-1]['portfolio_value']
            total_return = (final_value - self.initial_balance) / self.initial_balance
            
            # Calculate volatility
            portfolio_values = [bh['portfolio_value'] for bh in buy_hold_values]
            returns_series = pd.Series(portfolio_values).pct_change().dropna()
            
            # Annualized volatility (similar to strategy calculation)
            if len(returns_series) > 1:
                trading_days = len(returns_series) / 1440  # Assuming minute data
                days_per_year = 365
                daily_returns = returns_series.groupby(returns_series.index // 1440).sum()
                if len(daily_returns) > 1:
                    daily_volatility = daily_returns.std()
                    volatility = daily_volatility * np.sqrt(days_per_year)
                else:
                    volatility = returns_series.std() * np.sqrt(252 * 6.5 * 60)
            else:
                volatility = 0
            
            # Calculate Sharpe ratio
            if trading_days > 0:
                annualized_return = (1 + total_return) ** (365 / max(trading_days, 1)) - 1
            else:
                annualized_return = 0
                
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Calculate max drawdown
            portfolio_series = pd.Series(portfolio_values)
            running_max = portfolio_series.expanding().max()
            drawdown = (portfolio_series - running_max) / running_max
            max_drawdown = drawdown.min()
            
        else:
            final_value = self.initial_balance
            total_return = 0
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
            annualized_return = 0
        
        return {
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'annualized_return': annualized_return,
            'annualized_return_pct': annualized_return * 100,
            'portfolio_history': buy_hold_values,
            'individual_returns': buy_hold_returns
        }
    
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
    
    def _serialize_regime_metrics(self, metrics):
        """Serialize regime metrics for JSON export"""
        if metrics is None:
            return None
        
        return {
            'regime': metrics.regime.value,
            'total_trades': metrics.total_trades,
            'winning_trades': metrics.winning_trades,
            'losing_trades': metrics.losing_trades,
            'win_rate': metrics.win_rate,
            'total_return': metrics.total_return,
            'average_return': metrics.average_return,
            'total_pnl': metrics.total_pnl,
            'average_pnl': metrics.average_pnl,
            'max_win': metrics.max_win,
            'max_loss': metrics.max_loss,
            'sharpe_ratio': metrics.sharpe_ratio,
            'sortino_ratio': metrics.sortino_ratio,
            'max_drawdown': metrics.max_drawdown,
            'avg_holding_period': str(metrics.avg_holding_period),
            'total_duration': str(metrics.total_duration),
            'regime_percentage': metrics.regime_percentage
        }
    
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
            # Handle both datetime and int timestamps
            current_ts = timestamp if isinstance(timestamp, (int, float)) else timestamp.timestamp()
            entry_ts = position.get('entry_time', current_ts)
            if isinstance(entry_ts, pd.Timestamp):
                entry_ts = entry_ts.timestamp()
            elif hasattr(entry_ts, 'timestamp'):
                entry_ts = entry_ts.timestamp()
            
            market_data = {
                'position_age_minutes': (current_ts - entry_ts) / 60
            }
            updates = self.risk_manager.update_position_stops(market, current_price, market_data)
            
            if updates.get('trailing_stop_updated'):
                orders['trailing_stop'] = updates['new_trailing_stop']
            if updates.get('breakeven_stop_set') or updates.get('time_based_tightening'):
                orders['stop_loss'] = updates['new_stop_loss']
        
        # Check stop loss (including trailing stop)
        effective_stop = orders.get('trailing_stop', 0)
        stop_loss_price = orders.get('stop_loss', 0)
        if stop_loss_price > effective_stop:
            effective_stop = stop_loss_price
        
        if effective_stop > 0 and current_price <= effective_stop:
            # Check stop order type
            stop_type = orders.get('stop_loss_type', 'stop_market')
            
            if stop_type == 'stop_limit':
                # For stop-limit orders, check if limit price would be filled
                limit_price = orders.get('stop_loss_limit_price', effective_stop * 0.995)
                
                # Simulate order book depth - assume limit order fills if price goes below limit
                # In reality, this depends on liquidity and order book depth
                if current_price <= limit_price:
                    # Limit order fills
                    self._execute_stop_order(market, quantity, limit_price, timestamp, 'stop_loss_limit')
                else:
                    # Limit order doesn't fill, increment attempts
                    orders['stop_loss_attempts'] = orders.get('stop_loss_attempts', 0) + 1
                    
                    # After 3 failed attempts (3 minutes), convert to market order
                    if orders['stop_loss_attempts'] >= 3:
                        # Only log warning for first few occurrences per market
                        if market not in self.stop_limit_warning_count:
                            self.stop_limit_warning_count[market] = 0
                        
                        if self.stop_limit_warning_count[market] < self.max_stop_limit_warnings:
                            self.logger.warning(f"Stop-limit order failed {orders['stop_loss_attempts']} times for {market}, converting to market order")
                            self.stop_limit_warning_count[market] += 1
                            
                            # Add summary warning on last allowed warning
                            if self.stop_limit_warning_count[market] == self.max_stop_limit_warnings:
                                self.logger.info(f"Further stop-limit warnings for {market} will be suppressed")
                        
                        orders['stop_loss_type'] = 'stop_market'
                        self._execute_stop_order(market, quantity, current_price, timestamp, 'stop_loss_market')
            else:
                # Execute market order immediately
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
        # Initialize with safe defaults
        orders = {
            'stop_loss': 0,  # Default to 0 (no stop loss)
            'stop_loss_type': 'stop_market',  # Default order type
            'stop_loss_limit_price': 0,  # For stop-limit orders
            'stop_loss_attempts': 0,  # Track failed attempts
            'take_profit': []
        }
        
        # Set stop loss if available
        if hasattr(risk_params, 'stop_loss_price') and risk_params.stop_loss_price:
            orders['stop_loss'] = risk_params.stop_loss_price
            
            # Set order type
            if hasattr(risk_params, 'stop_loss_order_type'):
                if risk_params.stop_loss_order_type == OrderType.STOP_LIMIT:
                    orders['stop_loss_type'] = 'stop_limit'
                    # Set limit price with offset (default 0.5%)
                    orders['stop_loss_limit_price'] = risk_params.stop_loss_price * 0.995
                else:
                    orders['stop_loss_type'] = 'stop_market'
        
        # Skip partial exit configuration if risk management is disabled or regime_parameters missing
        if (not hasattr(self.risk_manager, 'config') or 
            'regime_parameters' not in self.risk_manager.config or
            not hasattr(risk_params, 'regime')):
            self.stop_orders[market] = orders
            return
        
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
        
        # Adjust slippage based on order type
        if 'stop_loss' in order_type:
            if 'market' in order_type:
                # Market orders have worse slippage
                slippage_rate *= 2.0  # 100% worse slippage for market stops
            elif 'limit' in order_type:
                # Limit orders have better execution (if they fill)
                slippage_rate *= 0.5  # 50% better slippage for limit stops
            else:
                slippage_rate *= 1.5  # 50% worse slippage for regular stops
        
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
            'order_type': order_type,
            'strategy': self.strategy_name
        })
        
        if self.trade_count <= self.max_log_trades:
            # Convert timestamp to string format
            ts_str = self._format_timestamp(timestamp)
            self.logger.info(f"[{ts_str}] {order_type.upper()} {market}: {quantity:.6f} at {execution_price:.2f}")
    
    def _calculate_volatility_data(self, df: pd.DataFrame, market: str) -> Dict:
        """Calculate volatility data for risk management"""
        if df is None or len(df) < 50:  # Need at least 50 rows for proper calculations
            return {'atr': 0, 'volatility_ratio': 1.0, 'momentum': None, 'rsi': None}
        
        # Calculate ATR
        high_low = df['high_price'] - df['low_price']
        high_close = abs(df['high_price'] - df['trade_price'].shift(1))
        low_close = abs(df['low_price'] - df['trade_price'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_series = tr.rolling(window=14).mean()
        atr = atr_series.iloc[-1] if len(atr_series) > 0 and not pd.isna(atr_series.iloc[-1]) else 0
        
        # Calculate volatility ratio
        returns = df['trade_price'].pct_change()
        current_vol_series = returns.rolling(window=20).std()
        current_vol = current_vol_series.iloc[-1] if len(current_vol_series) > 0 and not pd.isna(current_vol_series.iloc[-1]) else 0
        avg_vol = returns.rolling(window=50).std().mean()
        volatility_ratio = current_vol / avg_vol if avg_vol > 0 and current_vol > 0 else 1.0
        
        # Get other indicators if available
        momentum = None
        rsi = None
        
        if 'momentum' in df.columns and len(df) > 0:
            momentum = df['momentum'].iloc[-1] if not pd.isna(df['momentum'].iloc[-1]) else None
        if 'rsi' in df.columns and len(df) > 0:
            rsi = df['rsi'].iloc[-1] if not pd.isna(df['rsi'].iloc[-1]) else None
        
        return {
            'atr': atr,
            'volatility_ratio': volatility_ratio,
            'momentum': momentum,
            'rsi': rsi
        }