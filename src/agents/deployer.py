#!/usr/bin/env python3
"""
Deployer Agent for Quant-Alpaca
Handles real-time trading strategy execution with simulation and live trading capabilities
"""

import sys
import os
import json
import time
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import threading
import queue
import traceback

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import strategy registry
from strategies.registry import STRATEGIES, get_strategy
from strategies.base import BaseStrategy

# Import actions
from actions.upbit import UpbitAPI, UpbitTrader
from actions.deploy_trader import DeployTrader
from actions.market_regime import MarketRegimeDetector, MarketRegime
from actions.dynamic_risk_management import DynamicRiskManager
from actions.var_risk_management import VaRRiskManager
from actions.signal_strength import SignalStrengthAnalyzer
from actions.garch_position_sizing import GARCHPositionSizer

# Import data handling
from data.collector import UpbitDataCollector
from agents.scrapper import UpbitDataScrapper


class DeploymentAgent:
    """
    Deployment agent for executing trading strategies in real-time
    Supports both simulation and live trading modes
    """
    
    def __init__(self, config_path: str = 'config/config_deploy.json'):
        """Initialize the deployment agent with configuration"""
        self.config_path = config_path
        self.load_configuration()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize API clients
        self.setup_api_clients()
        
        # Initialize market data handler
        self.data_collector = UpbitDataCollector(
            self.config['data']['database_directory'],
            self.config['data']['database_pattern']
        )
        
        # Initialize risk managers
        self.setup_risk_managers()
        
        # Initialize strategies
        self.strategies = {}
        self.load_strategies()
        
        # Initialize market data cache
        self.market_data = {}
        self.market_regimes = {}
        
        # Initialize position tracker
        self.positions = {}
        self.pending_orders = {}
        
        # Initialize execution engine
        self.deploy_trader = DeployTrader(
            self.upbit_trader,
            self.dynamic_risk_manager,
            self.var_risk_manager,
            simulation_mode=(not self.config['execution']['real_trading']),
            config=self.config['execution']
        )
        
        # Status tracking
        self.running = False
        self.last_data_update = {}
        self.last_risk_check = datetime.now()
        self.last_position_update = datetime.now()
        
        # Performance tracking
        self.performance_metrics = {
            'start_balance': 0,
            'current_balance': 0,
            'unrealized_pnl': 0,
            'realized_pnl': 0,
            'trades_executed': 0,
            'win_count': 0,
            'loss_count': 0,
            'fees_paid': 0
        }
        
        # Threading components
        self.data_update_thread = None
        self.signal_thread = None
        self.execution_thread = None
        self.signal_queue = queue.Queue()
        self.execution_queue = queue.Queue()
        
        self.logger.info(f"DeploymentAgent initialized with config: {config_path}")
        self.logger.info(f"Trading mode: {'SIMULATION' if not self.config['execution']['real_trading'] else 'REAL TRADING'}")

    def load_configuration(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
                
            # Set defaults if not specified
            if 'execution' not in self.config:
                self.config['execution'] = {
                    'real_trading': False,
                    'update_interval_seconds': 60,
                    'risk_check_interval_minutes': 15,
                    'max_slippage_pct': 0.5,
                    'max_trades_per_day': 10,
                    'max_position_size_pct': 20.0,
                    'max_total_risk_pct': 50.0
                }
                
            if 'data' not in self.config:
                self.config['data'] = {
                    'database_directory': 'data/candles',
                    'database_pattern': '{market}_candles.db',
                    'lookback_hours': 168,
                    'candle_interval': '1m',
                }
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            sys.exit(f"Error loading configuration: {e}")
    
    def setup_logging(self):
        """Configure logging system"""
        log_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamped log file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"deploy_{timestamp}.log")
        
        # Configure logger
        self.logger = logging.getLogger('quant_alpaca_deployer')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def setup_api_clients(self):
        """Initialize API clients"""
        # Check for API keys in environment or config
        access_key = os.environ.get('UPBIT_ACCESS_KEY') or self.config.get('api_keys', {}).get('access_key')
        secret_key = os.environ.get('UPBIT_SECRET_KEY') or self.config.get('api_keys', {}).get('secret_key')
        
        if not access_key or not secret_key:
            self.logger.warning("API keys not found - running in SIMULATION mode only!")
            # In simulation mode, create dummy API clients
            self.upbit_api = None
            self.upbit_trader = None
        else:
            self.upbit_api = UpbitAPI(access_key, secret_key)
            self.upbit_trader = UpbitTrader(access_key, secret_key)
    
    def setup_risk_managers(self):
        """Initialize risk management components"""
        # Dynamic risk manager (stop-loss, take-profit)
        self.dynamic_risk_manager = DynamicRiskManager(
            config=self.config.get('risk_management', {}).get('dynamic_risk', {})
        )
        
        # VaR risk manager (portfolio level risk)
        self.var_risk_manager = VaRRiskManager(
            config=self.config.get('risk_management', {}).get('var_risk', {})
        )
        
        # Signal strength analyzer
        self.signal_analyzer = SignalStrengthAnalyzer(
            config=self.config.get('risk_management', {}).get('signal_strength', {})
        )
        
        # Position sizer
        self.position_sizer = GARCHPositionSizer(
            config=self.config.get('risk_management', {}).get('position_sizing', {})
        )
        
        # Market regime detector
        self.regime_detector = MarketRegimeDetector(
            config=self.config.get('risk_management', {}).get('market_regime', {})
        )
    
    def load_strategies(self):
        """Load and initialize strategies from configuration"""
        strategy_configs = self.config.get('strategies', {})
        
        for market, market_config in strategy_configs.items():
            strategy_name = market_config.get('strategy')
            
            if not strategy_name:
                self.logger.warning(f"No strategy specified for {market} - skipping")
                continue
                
            # Load optimized parameters if specified
            params_file = market_config.get('parameters_file')
            if params_file:
                try:
                    with open(params_file, 'r') as f:
                        parameters = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    self.logger.error(f"Failed to load parameters for {market} from {params_file}: {e}")
                    # Use default parameters from config
                    parameters = market_config.get('parameters', {})
            else:
                # Use parameters directly from config
                parameters = market_config.get('parameters', {})
                
            # Initialize the strategy
            try:
                self.strategies[market] = get_strategy(strategy_name, parameters)
                self.logger.info(f"Loaded {strategy_name} strategy for {market}")
            except ValueError as e:
                self.logger.error(f"Failed to load strategy for {market}: {e}")
                continue
    
    def start(self):
        """Start the deployment agent"""
        if self.running:
            self.logger.warning("Deployment agent already running")
            return
            
        self.running = True
        self.logger.info("Starting deployment agent")
        
        # Fetch initial data and account info
        self.update_initial_state()
        
        # Start data update thread
        self.data_update_thread = threading.Thread(
            target=self._data_update_loop, 
            daemon=True
        )
        self.data_update_thread.start()
        
        # Start signal generation thread
        self.signal_thread = threading.Thread(
            target=self._signal_generation_loop,
            daemon=True
        )
        self.signal_thread.start()
        
        # Start execution thread
        self.execution_thread = threading.Thread(
            target=self._execution_loop,
            daemon=True
        )
        self.execution_thread.start()
        
        self.logger.info("Deployment agent started")
        
        try:
            # Main monitoring loop
            while self.running:
                # Periodically update performance metrics
                self.update_performance_metrics()
                self.log_status()
                
                # Check risk limits periodically
                if (datetime.now() - self.last_risk_check).total_seconds() > (
                    self.config['execution'].get('risk_check_interval_minutes', 15) * 60
                ):
                    self.check_risk_limits()
                    self.last_risk_check = datetime.now()
                
                # Sleep to avoid high CPU usage
                time.sleep(30)
                
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Gracefully shut down the deployment agent"""
        self.logger.info("Shutting down deployment agent...")
        self.running = False
        
        # Wait for threads to complete
        if self.data_update_thread and self.data_update_thread.is_alive():
            self.data_update_thread.join(timeout=5)
        if self.signal_thread and self.signal_thread.is_alive():
            self.signal_thread.join(timeout=5)
        if self.execution_thread and self.execution_thread.is_alive():
            self.execution_thread.join(timeout=5)
        
        # Save final state
        self.save_performance_report()
        self.logger.info("Deployment agent shutdown complete")
    
    def update_initial_state(self):
        """Fetch initial market data and account information"""
        self.logger.info("Fetching initial market data and account information")
        
        # Get list of markets to monitor
        self.markets = list(self.strategies.keys())
        self.logger.info(f"Markets to monitor: {self.markets}")
        
        # Fetch initial market data
        for market in self.markets:
            self._fetch_and_process_market_data(market)
            self.last_data_update[market] = datetime.now()
        
        # Fetch account information if in real trading mode
        if self.upbit_trader and self.config['execution']['real_trading']:
            try:
                accounts = self.upbit_trader.api.get_accounts()
                
                # Track KRW balance
                for account in accounts:
                    if account['currency'] == 'KRW':
                        balance = float(account['balance'])
                        self.performance_metrics['start_balance'] = balance
                        self.performance_metrics['current_balance'] = balance
                        self.logger.info(f"Initial KRW balance: {balance:,.2f}")
                        break
                
                # Track existing positions
                for account in accounts:
                    if account['currency'] != 'KRW' and float(account['balance']) > 0:
                        market = f"KRW-{account['currency']}"
                        if market in self.markets:
                            amount = float(account['balance'])
                            avg_price = float(account['avg_buy_price'])
                            
                            self.positions[market] = {
                                'amount': amount,
                                'entry_price': avg_price,
                                'entry_time': datetime.now() - timedelta(days=1),  # Assume 1 day old
                                'strategy': self.strategies[market].__class__.__name__,
                                'realized_pnl': 0
                            }
                            
                            self.logger.info(f"Existing position in {market}: {amount} @ {avg_price}")
            except Exception as e:
                self.logger.error(f"Failed to fetch account information: {e}")
                
        # In simulation mode, set initial balance from config
        else:
            initial_balance = self.config['execution'].get('simulation_initial_balance', 10000000)
            self.performance_metrics['start_balance'] = initial_balance
            self.performance_metrics['current_balance'] = initial_balance
            self.logger.info(f"Simulation initial balance: {initial_balance:,.2f} KRW")
    
    def _data_update_loop(self):
        """Background thread for updating market data"""
        self.logger.info("Data update thread started")
        
        while self.running:
            try:
                update_interval = self.config['execution'].get('update_interval_seconds', 60)
                
                # Fetch data for each market
                for market in self.markets:
                    # Check if it's time to update this market
                    if (datetime.now() - self.last_data_update.get(market, datetime.min)).total_seconds() >= update_interval:
                        self._fetch_and_process_market_data(market)
                        self.last_data_update[market] = datetime.now()
                
                # Sleep for a short time to avoid high CPU usage
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in data update loop: {e}")
                self.logger.error(traceback.format_exc())
                time.sleep(5)  # Sleep longer on error
    
    def _fetch_and_process_market_data(self, market: str):
        """Fetch and process market data for a specific market"""
        try:
            # Get database path
            db_path = self.data_collector.get_database_path(market)
            scrapper = UpbitDataScrapper(db_path)
            
            # Calculate lookback period
            lookback_hours = self.config['data'].get('lookback_hours', 168)  # 7 days default
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=lookback_hours)
            
            # Format dates
            start_date = start_time.strftime("%Y-%m-%d")
            end_date = end_time.strftime("%Y-%m-%d")
            
            # Get data from database
            df = scrapper.get_candle_data_from_db(
                market=market,
                start_date=start_date,
                end_date=end_date
            )
            
            if df.empty:
                self.logger.warning(f"No data available for {market}, attempting to collect")
                # Try to collect data
                scrapper.scrape_market_data(market, days=lookback_hours/24)
                df = scrapper.get_candle_data_from_db(
                    market=market,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if df.empty:
                    self.logger.error(f"Failed to collect data for {market}")
                    return
            
            # Process timestamps
            df['candle_date_time_utc'] = pd.to_datetime(df['candle_date_time_utc'])
            df.set_index('candle_date_time_utc', inplace=True)
            
            # Update market data cache
            self.market_data[market] = df
            
            # Detect market regime
            regime, indicators = self.regime_detector.detect_regime(df)
            self.market_regimes[market] = regime
            
            self.logger.debug(f"Updated data for {market}, regime: {regime.value}, candles: {len(df)}")
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {market}: {e}")
            self.logger.error(traceback.format_exc())
    
    def _signal_generation_loop(self):
        """Background thread for signal generation"""
        self.logger.info("Signal generation thread started")
        
        while self.running:
            try:
                # Generate signals for each market
                for market in self.markets:
                    if market not in self.market_data:
                        continue
                    
                    df = self.market_data[market]
                    if df is None or df.empty:
                        continue
                    
                    # Check if we have a position in this market
                    has_position = market in self.positions
                    
                    # Get strategy for this market
                    strategy = self.strategies.get(market)
                    if strategy is None:
                        continue
                    
                    # Generate signal
                    signal = strategy.generate_signal_for_timestamp(df, market, has_position)
                    
                    # Only process non-zero signals
                    if signal != 0:
                        # Get current price
                        current_price = df['trade_price'].iloc[-1]
                        
                        # Add market regime info
                        regime = self.market_regimes.get(market, MarketRegime.UNKNOWN)
                        
                        # Analyze signal strength
                        signal_strength = self.signal_analyzer.analyze_signal(
                            df=df,
                            signal=signal,
                            regime=regime,
                            strategy_name=strategy.__class__.__name__
                        )
                        
                        # Calculate position size
                        if signal == 1:  # Buy signal
                            position_size_pct = self.position_sizer.calculate_position_size(
                                df=df,
                                signal_strength=signal_strength,
                                market=market,
                                regime=regime
                            )
                            
                            # Limit by max position size
                            max_position_pct = self.config['execution'].get('max_position_size_pct', 20.0)
                            position_size_pct = min(position_size_pct, max_position_pct)
                            
                            # Add signal to queue
                            self.signal_queue.put({
                                'market': market,
                                'signal': signal,
                                'price': current_price,
                                'timestamp': datetime.now(),
                                'signal_strength': signal_strength,
                                'regime': regime.value,
                                'position_size_pct': position_size_pct
                            })
                            
                            self.logger.info(f"BUY signal for {market} at {current_price} (strength: {signal_strength:.2f}, size: {position_size_pct:.2f}%)")
                            
                        elif signal == -1 and has_position:  # Sell signal
                            # Add signal to queue
                            self.signal_queue.put({
                                'market': market,
                                'signal': signal,
                                'price': current_price,
                                'timestamp': datetime.now(),
                                'signal_strength': signal_strength,
                                'regime': regime.value,
                                'position_size_pct': 100.0  # Sell entire position
                            })
                            
                            self.logger.info(f"SELL signal for {market} at {current_price} (strength: {signal_strength:.2f})")
                
                # Sleep to avoid high CPU usage
                time.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error in signal generation loop: {e}")
                self.logger.error(traceback.format_exc())
                time.sleep(5)  # Sleep longer on error
    
    def _execution_loop(self):
        """Background thread for trade execution"""
        self.logger.info("Execution thread started")
        
        while self.running:
            try:
                # Check for signals in the queue
                try:
                    signal_data = self.signal_queue.get(timeout=1)
                    
                    # Process the signal
                    market = signal_data['market']
                    signal = signal_data['signal']
                    price = signal_data['price']
                    position_size_pct = signal_data['position_size_pct']
                    
                    # Buy signal
                    if signal == 1:
                        # Skip if we already have a position
                        if market in self.positions:
                            self.logger.info(f"Already have position in {market}, ignoring buy signal")
                            continue
                        
                        # Check if we've reached max trades per day
                        trades_today = self.performance_metrics.get('trades_executed', 0)
                        max_trades = self.config['execution'].get('max_trades_per_day', 10)
                        if trades_today >= max_trades:
                            self.logger.warning(f"Reached max trades per day ({max_trades}), ignoring buy signal")
                            continue
                        
                        # Check portfolio risk
                        total_risk_pct = self._calculate_total_risk_pct()
                        max_risk_pct = self.config['execution'].get('max_total_risk_pct', 50.0)
                        if total_risk_pct + position_size_pct > max_risk_pct:
                            self.logger.warning(f"Adding position would exceed max risk ({total_risk_pct:.2f}% + {position_size_pct:.2f}% > {max_risk_pct:.2f}%), ignoring buy signal")
                            continue
                        
                        # Execute buy order
                        order_result = self.deploy_trader.execute_buy(
                            market=market,
                            price=price,
                            position_size_pct=position_size_pct,
                            regime=MarketRegime(signal_data['regime'])
                        )
                        
                        if order_result.get('success', False):
                            # Track the new position
                            self.positions[market] = {
                                'amount': order_result.get('amount', 0),
                                'entry_price': order_result.get('executed_price', price),
                                'entry_time': datetime.now(),
                                'strategy': self.strategies[market].__class__.__name__,
                                'realized_pnl': 0
                            }
                            
                            # Update performance metrics
                            self.performance_metrics['trades_executed'] += 1
                            self.performance_metrics['fees_paid'] += order_result.get('fee', 0)
                            
                            self.logger.info(f"BUY executed for {market}: {order_result}")
                        else:
                            self.logger.error(f"Failed to execute buy for {market}: {order_result}")
                    
                    # Sell signal
                    elif signal == -1:
                        # Skip if we don't have a position
                        if market not in self.positions:
                            self.logger.info(f"No position in {market}, ignoring sell signal")
                            continue
                        
                        # Execute sell order
                        order_result = self.deploy_trader.execute_sell(
                            market=market,
                            price=price,
                            amount=self.positions[market]['amount'],
                            entry_price=self.positions[market]['entry_price']
                        )
                        
                        if order_result.get('success', False):
                            # Calculate PnL
                            entry_price = self.positions[market]['entry_price']
                            exit_price = order_result.get('executed_price', price)
                            amount = self.positions[market]['amount']
                            pnl = (exit_price - entry_price) * amount
                            
                            # Update performance metrics
                            self.performance_metrics['realized_pnl'] += pnl
                            self.performance_metrics['fees_paid'] += order_result.get('fee', 0)
                            self.performance_metrics['trades_executed'] += 1
                            
                            # Update win/loss count
                            if pnl > 0:
                                self.performance_metrics['win_count'] += 1
                            else:
                                self.performance_metrics['loss_count'] += 1
                            
                            # Remove the position
                            del self.positions[market]
                            
                            self.logger.info(f"SELL executed for {market}: {order_result}, PnL: {pnl:,.2f} KRW")
                        else:
                            self.logger.error(f"Failed to execute sell for {market}: {order_result}")
                
                except queue.Empty:
                    # No signals to process
                    pass
                
                # Update position status and check risk limits
                self._update_positions()
                
            except Exception as e:
                self.logger.error(f"Error in execution loop: {e}")
                self.logger.error(traceback.format_exc())
                time.sleep(5)  # Sleep longer on error
    
    def _update_positions(self):
        """Update position status and check for stop/take-profit triggers"""
        if not self.positions:
            return
            
        # Only update every minute
        if (datetime.now() - self.last_position_update).total_seconds() < 60:
            return
            
        self.last_position_update = datetime.now()
        
        # Calculate total unrealized PnL
        total_unrealized_pnl = 0
        
        # Update each position
        for market, position in list(self.positions.items()):
            if market not in self.market_data:
                continue
                
            df = self.market_data[market]
            if df is None or df.empty:
                continue
                
            # Get current price
            current_price = df['trade_price'].iloc[-1]
            entry_price = position['entry_price']
            amount = position['amount']
            
            # Calculate unrealized PnL
            unrealized_pnl = (current_price - entry_price) * amount
            total_unrealized_pnl += unrealized_pnl
            
            # Get market regime
            regime = self.market_regimes.get(market, MarketRegime.UNKNOWN)
            
            # Update position risk parameters
            position_id = f"{market}_{position['entry_time'].strftime('%Y%m%d%H%M')}"
            
            # Check if position is already being tracked by risk manager
            if position_id not in self.dynamic_risk_manager.positions:
                # Calculate volatility metrics
                atr = df['high_price'].rolling(14).max() - df['low_price'].rolling(14).min()
                atr = atr.iloc[-1] / current_price
                
                volatility_data = {
                    'atr': atr,
                    'volatility_ratio': 1.0
                }
                
                # Initialize risk parameters
                risk_params = self.dynamic_risk_manager.calculate_position_risk_parameters(
                    entry_price=entry_price,
                    position_size=amount,
                    regime=regime,
                    volatility_data=volatility_data
                )
                
                # Add position to risk manager
                self.dynamic_risk_manager.add_position(position_id, risk_params)
            
            # Check for stop-loss or take-profit triggers
            market_data = {
                'momentum': 0,
                'rsi': 50,
                'position_age_minutes': (datetime.now() - position['entry_time']).total_seconds() / 60
            }
            
            updates = self.dynamic_risk_manager.update_position_stops(
                position_id=position_id,
                current_price=current_price,
                market_data=market_data
            )
            
            risk_params = self.dynamic_risk_manager.positions.get(position_id)
            if risk_params:
                # Check stop-loss
                if (current_price <= risk_params.stop_loss_price) or (
                    risk_params.is_trailing_active and current_price <= risk_params.current_trailing_stop
                ):
                    self.logger.info(f"Stop-loss triggered for {market} at {current_price}")
                    
                    # Execute sell order
                    order_result = self.deploy_trader.execute_sell(
                        market=market,
                        price=current_price,
                        amount=amount,
                        entry_price=entry_price,
                        order_type='stop_loss'
                    )
                    
                    if order_result.get('success', False):
                        # Calculate PnL
                        exit_price = order_result.get('executed_price', current_price)
                        pnl = (exit_price - entry_price) * amount
                        
                        # Update performance metrics
                        self.performance_metrics['realized_pnl'] += pnl
                        self.performance_metrics['fees_paid'] += order_result.get('fee', 0)
                        self.performance_metrics['trades_executed'] += 1
                        self.performance_metrics['loss_count'] += 1
                        
                        # Remove the position
                        del self.positions[market]
                        self.dynamic_risk_manager.remove_position(position_id)
                        
                        self.logger.info(f"STOP LOSS executed for {market}: {order_result}, PnL: {pnl:,.2f} KRW")
                    else:
                        self.logger.error(f"Failed to execute stop loss for {market}: {order_result}")
                
                # Check take-profit
                elif current_price >= risk_params.take_profit_price:
                    self.logger.info(f"Take-profit triggered for {market} at {current_price}")
                    
                    # Execute sell order
                    order_result = self.deploy_trader.execute_sell(
                        market=market,
                        price=current_price,
                        amount=amount,
                        entry_price=entry_price,
                        order_type='take_profit'
                    )
                    
                    if order_result.get('success', False):
                        # Calculate PnL
                        exit_price = order_result.get('executed_price', current_price)
                        pnl = (exit_price - entry_price) * amount
                        
                        # Update performance metrics
                        self.performance_metrics['realized_pnl'] += pnl
                        self.performance_metrics['fees_paid'] += order_result.get('fee', 0)
                        self.performance_metrics['trades_executed'] += 1
                        self.performance_metrics['win_count'] += 1
                        
                        # Remove the position
                        del self.positions[market]
                        self.dynamic_risk_manager.remove_position(position_id)
                        
                        self.logger.info(f"TAKE PROFIT executed for {market}: {order_result}, PnL: {pnl:,.2f} KRW")
                    else:
                        self.logger.error(f"Failed to execute take profit for {market}: {order_result}")
        
        # Update unrealized PnL
        self.performance_metrics['unrealized_pnl'] = total_unrealized_pnl
    
    def _calculate_total_risk_pct(self) -> float:
        """Calculate total portfolio risk percentage"""
        if not self.positions:
            return 0.0
            
        current_balance = self.performance_metrics['current_balance']
        if current_balance <= 0:
            return 100.0
            
        total_position_value = 0.0
        
        for market, position in self.positions.items():
            if market not in self.market_data:
                continue
                
            df = self.market_data[market]
            if df is None or df.empty:
                continue
                
            # Get current price
            current_price = df['trade_price'].iloc[-1]
            amount = position['amount']
            
            # Calculate position value
            position_value = current_price * amount
            total_position_value += position_value
        
        return (total_position_value / current_balance) * 100.0
    
    def check_risk_limits(self):
        """Check portfolio-level risk limits"""
        if not self.upbit_trader:
            return
            
        try:
            # Get current portfolio value
            total_portfolio_value = self.performance_metrics['current_balance']
            total_portfolio_value += self.performance_metrics['unrealized_pnl']
            
            # Calculate daily P&L
            daily_pnl = self.performance_metrics['realized_pnl'] + self.performance_metrics['unrealized_pnl']
            
            # Prepare position data for VaR
            position_values = {}
            position_returns = {}
            
            for market, position in self.positions.items():
                if market not in self.market_data:
                    continue
                    
                df = self.market_data[market]
                if df is None or df.empty:
                    continue
                    
                # Get current price
                current_price = df['trade_price'].iloc[-1]
                amount = position['amount']
                
                # Calculate position value
                position_value = current_price * amount
                position_values[market] = position_value
                
                # Calculate returns series
                returns = df['trade_price'].pct_change().dropna()
                position_returns[market] = returns
            
            # Calculate VaR if we have positions
            if position_values:
                # Dummy returns for testing if we don't have real data
                if not position_returns:
                    for market in position_values:
                        position_returns[market] = pd.Series([0.001, -0.002, 0.003])
                
                var_result = self.var_risk_manager.calculate_var_cvar(
                    returns=pd.Series([0.001, -0.002, 0.003]),  # Dummy for now
                    portfolio_value=total_portfolio_value
                )
                
                # Check risk limits
                risk_status = self.var_risk_manager.check_risk_limits(
                    current_portfolio_value=total_portfolio_value,
                    daily_pnl=daily_pnl,
                    positions=position_values,
                    var_result=var_result,
                    current_time=datetime.now()
                )
                
                # Log risk status
                self.logger.info(f"Risk check: VaR={var_result.var_amount:,.2f}, CVaR={var_result.cvar_amount:,.2f}")
                self.logger.info(f"Risk utilization: {risk_status.var_utilization:.2%} of limit")
                
                # Handle risk limit breaches
                if not risk_status.trading_allowed:
                    self.logger.warning("Risk limits breached: New trades blocked")
                
                if risk_status.positions_to_close:
                    self.logger.warning(f"Risk breach: {len(risk_status.positions_to_close)} positions need to be closed")
                    
                    # Close positions if needed
                    for market in risk_status.positions_to_close:
                        if market in self.positions:
                            position = self.positions[market]
                            
                            self.logger.warning(f"Closing position in {market} due to risk limits")
                            
                            if market not in self.market_data:
                                continue
                                
                            df = self.market_data[market]
                            if df is None or df.empty:
                                continue
                                
                            # Get current price
                            current_price = df['trade_price'].iloc[-1]
                            
                            # Execute sell order
                            order_result = self.deploy_trader.execute_sell(
                                market=market,
                                price=current_price,
                                amount=position['amount'],
                                entry_price=position['entry_price'],
                                order_type='risk_limit'
                            )
                            
                            if order_result.get('success', False):
                                # Calculate PnL
                                exit_price = order_result.get('executed_price', current_price)
                                amount = position['amount']
                                entry_price = position['entry_price']
                                pnl = (exit_price - entry_price) * amount
                                
                                # Update performance metrics
                                self.performance_metrics['realized_pnl'] += pnl
                                self.performance_metrics['fees_paid'] += order_result.get('fee', 0)
                                self.performance_metrics['trades_executed'] += 1
                                
                                if pnl > 0:
                                    self.performance_metrics['win_count'] += 1
                                else:
                                    self.performance_metrics['loss_count'] += 1
                                
                                # Remove the position
                                del self.positions[market]
                                
                                self.logger.info(f"RISK LIMIT SELL executed for {market}: {order_result}, PnL: {pnl:,.2f} KRW")
                            else:
                                self.logger.error(f"Failed to execute risk limit sell for {market}: {order_result}")
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            self.logger.error(traceback.format_exc())
    
    def update_performance_metrics(self):
        """Update performance metrics with latest data"""
        if not self.upbit_trader:
            return
            
        try:
            # In real trading mode, fetch actual balance
            if self.config['execution']['real_trading']:
                try:
                    krw_balance = self.upbit_trader.get_balance('KRW')
                    self.performance_metrics['current_balance'] = krw_balance
                except Exception as e:
                    self.logger.error(f"Failed to fetch KRW balance: {e}")
            # In simulation mode, calculate based on starting balance and PnL
            else:
                self.performance_metrics['current_balance'] = (
                    self.performance_metrics['start_balance'] + 
                    self.performance_metrics['realized_pnl'] -
                    self.performance_metrics['fees_paid']
                )
                
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def log_status(self):
        """Log current status and performance metrics"""
        # Calculate win rate
        total_closed_trades = self.performance_metrics['win_count'] + self.performance_metrics['loss_count']
        win_rate = self.performance_metrics['win_count'] / total_closed_trades if total_closed_trades > 0 else 0
        
        # Log overall status
        self.logger.info("========== CURRENT STATUS ==========")
        self.logger.info(f"Active positions: {len(self.positions)}")
        self.logger.info(f"Balance: {self.performance_metrics['current_balance']:,.2f} KRW")
        self.logger.info(f"Realized P&L: {self.performance_metrics['realized_pnl']:,.2f} KRW")
        self.logger.info(f"Unrealized P&L: {self.performance_metrics['unrealized_pnl']:,.2f} KRW")
        self.logger.info(f"Trades executed: {self.performance_metrics['trades_executed']}")
        self.logger.info(f"Win/Loss: {self.performance_metrics['win_count']}/{self.performance_metrics['loss_count']} ({win_rate:.2%})")
        self.logger.info(f"Fees paid: {self.performance_metrics['fees_paid']:,.2f} KRW")
        
        # Log position details
        if self.positions:
            self.logger.info("--------- ACTIVE POSITIONS ---------")
            for market, position in self.positions.items():
                if market not in self.market_data:
                    continue
                    
                df = self.market_data[market]
                if df is None or df.empty:
                    continue
                    
                # Get current price
                current_price = df['trade_price'].iloc[-1]
                entry_price = position['entry_price']
                amount = position['amount']
                
                # Calculate unrealized PnL
                unrealized_pnl = (current_price - entry_price) * amount
                pnl_pct = ((current_price / entry_price) - 1) * 100
                
                # Get position age
                age_hours = (datetime.now() - position['entry_time']).total_seconds() / 3600
                
                self.logger.info(f"{market}: {amount:.8f} @ {entry_price:,.2f} ’ {current_price:,.2f} "
                               f"({pnl_pct:+.2f}%, {unrealized_pnl:+,.2f} KRW, {age_hours:.1f}h)")
    
    def save_performance_report(self):
        """Save performance report to file"""
        report_dir = os.path.join(os.getcwd(), 'results/deployment')
        os.makedirs(report_dir, exist_ok=True)
        
        # Create timestamped report file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(report_dir, f"deploy_report_{timestamp}.json")
        
        # Calculate additional metrics
        total_closed_trades = self.performance_metrics['win_count'] + self.performance_metrics['loss_count']
        win_rate = self.performance_metrics['win_count'] / total_closed_trades if total_closed_trades > 0 else 0
        
        # Create report data
        report_data = {
            'timestamp': timestamp,
            'mode': 'REAL' if self.config['execution']['real_trading'] else 'SIMULATION',
            'duration_hours': (datetime.now() - datetime.strptime(timestamp, "%Y%m%d_%H%M%S")).total_seconds() / 3600,
            'metrics': {
                'start_balance': self.performance_metrics['start_balance'],
                'final_balance': self.performance_metrics['current_balance'],
                'realized_pnl': self.performance_metrics['realized_pnl'],
                'unrealized_pnl': self.performance_metrics['unrealized_pnl'],
                'total_pnl': self.performance_metrics['realized_pnl'] + self.performance_metrics['unrealized_pnl'],
                'pnl_pct': ((self.performance_metrics['current_balance'] / self.performance_metrics['start_balance']) - 1) * 100 if self.performance_metrics['start_balance'] > 0 else 0,
                'trades_executed': self.performance_metrics['trades_executed'],
                'win_count': self.performance_metrics['win_count'],
                'loss_count': self.performance_metrics['loss_count'],
                'win_rate': win_rate * 100,
                'fees_paid': self.performance_metrics['fees_paid']
            },
            'active_positions': {},
            'market_regimes': {},
            'config': self.config
        }
        
        # Add position details
        for market, position in self.positions.items():
            report_data['active_positions'][market] = {
                'amount': position['amount'],
                'entry_price': position['entry_price'],
                'entry_time': position['entry_time'].isoformat(),
                'strategy': position['strategy']
            }
            
            # Add current price and PnL if available
            if market in self.market_data and self.market_data[market] is not None:
                df = self.market_data[market]
                if not df.empty:
                    current_price = df['trade_price'].iloc[-1]
                    unrealized_pnl = (current_price - position['entry_price']) * position['amount']
                    
                    report_data['active_positions'][market].update({
                        'current_price': current_price,
                        'unrealized_pnl': unrealized_pnl,
                        'pnl_pct': ((current_price / position['entry_price']) - 1) * 100
                    })
        
        # Add market regime data
        for market, regime in self.market_regimes.items():
            report_data['market_regimes'][market] = regime.value
        
        # Save report
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=4)
            
        self.logger.info(f"Performance report saved to {report_file}")
        
        return report_file


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Quant-Alpaca Deployment Agent')
    parser.add_argument('--config', type=str, default='config/config_deploy.json',
                      help='Configuration file path')
    parser.add_argument('--simulation', action='store_true',
                      help='Run in simulation mode (no real trading)')
    parser.add_argument('--real-time', action='store_true',
                      help='Run in real-time trading mode')
    
    args = parser.parse_args()
    
    # Load config
    config_path = args.config
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Override trading mode from command line
        if args.simulation:
            config['execution']['real_trading'] = False
        if args.real_time:
            config['execution']['real_trading'] = True
            
        # Save updated config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
            
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error with configuration file: {e}")
        sys.exit(1)
    
    # Create and start deployment agent
    agent = DeploymentAgent(config_path=config_path)
    agent.start()