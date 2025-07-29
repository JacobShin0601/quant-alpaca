import pandas as pd
import numpy as np
from typing import Dict, Any
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """Base class for trading strategies"""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
    
    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        pass
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate buy/sell signals"""
        pass
    
    def generate_signal_for_timestamp(self, df: pd.DataFrame, market: str, has_position: bool) -> int:
        """Generate signal for a specific timestamp considering current position"""
        # Default implementation - use the last row's signal
        if len(df) == 0:
            return 0
        
        # Get the last row
        last_row = df.iloc[-1]
        
        # Basic position-aware logic
        if has_position:
            # Only allow sell signals when we have a position
            return -1 if self._should_sell(last_row, df) else 0
        else:
            # Only allow buy signals when we don't have a position
            return 1 if self._should_buy(last_row, df) else 0
    
    def _should_buy(self, last_row, df) -> bool:
        """Override in subclass to implement buy logic"""
        return False
    
    def _should_sell(self, last_row, df) -> bool:
        """Override in subclass to implement sell logic"""
        return False


class BasicMomentumStrategy(BaseStrategy):
    """Basic momentum strategy using RSI and moving averages"""
    
    def _should_buy(self, last_row, df) -> bool:
        """Buy when RSI is oversold and short MA > long MA"""
        if pd.isna(last_row['rsi']) or pd.isna(last_row['ma_short']) or pd.isna(last_row['ma_long']):
            return False
        return (
            last_row['rsi'] < self.parameters['rsi_oversold'] and
            last_row['ma_short'] > last_row['ma_long']
        )
    
    def _should_sell(self, last_row, df) -> bool:
        """Sell when RSI is overbought or short MA < long MA"""
        if pd.isna(last_row['rsi']) or pd.isna(last_row['ma_short']) or pd.isna(last_row['ma_long']):
            return False
        return (
            last_row['rsi'] > self.parameters['rsi_overbought'] or
            last_row['ma_short'] < last_row['ma_long']
        )
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI and moving averages"""
        # RSI calculation
        rsi_period = self.parameters['rsi_period']
        delta = df['trade_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        ma_short = self.parameters['ma_short']
        ma_long = self.parameters['ma_long']
        df['ma_short'] = df['trade_price'].rolling(window=ma_short).mean()
        df['ma_long'] = df['trade_price'].rolling(window=ma_long).mean()
        
        return df
    
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate buy/sell signals based on RSI and MA crossover"""
        df['signal'] = 0  # 0: hold, 1: buy, -1: sell
        
        # Buy condition: RSI oversold and short MA above long MA
        buy_condition = (
            (df['rsi'] < self.parameters['rsi_oversold']) &
            (df['ma_short'] > df['ma_long'])
        )
        
        # Sell condition: RSI overbought or short MA below long MA
        sell_condition = (
            (df['rsi'] > self.parameters['rsi_overbought']) |
            (df['ma_short'] < df['ma_long'])
        )
        
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        return df


class VWAPStrategy(BaseStrategy):
    """VWAP (Volume Weighted Average Price) strategy with multiple timeframes"""
    
    def _should_buy(self, last_row, df) -> bool:
        """Buy when price is below VWAP with high volume"""
        if pd.isna(last_row.get('vwap_deviation', np.nan)) or pd.isna(last_row.get('volume_ratio', np.nan)):
            return False
        
        vwap_threshold = self.parameters.get('vwap_threshold', 0.005)
        volume_threshold = self.parameters.get('volume_threshold', 1.2)
        
        return (
            last_row['vwap_deviation'] < -vwap_threshold and
            last_row['volume_ratio'] > volume_threshold
        )
    
    def _should_sell(self, last_row, df) -> bool:
        """Sell when price is above VWAP with high volume"""
        if pd.isna(last_row.get('vwap_deviation', np.nan)) or pd.isna(last_row.get('volume_ratio', np.nan)):
            return False
        
        vwap_threshold = self.parameters.get('vwap_threshold', 0.005)
        volume_threshold = self.parameters.get('volume_threshold', 1.2)
        
        return (
            last_row['vwap_deviation'] > vwap_threshold and
            last_row['volume_ratio'] > volume_threshold
        )
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate VWAP and related indicators"""
        # Calculate VWAP for different periods
        vwap_period = self.parameters['vwap_period']
        vwap_short_period = self.parameters.get('vwap_short_period', vwap_period // 2)
        
        # VWAP calculation
        df['typical_price'] = (df['high_price'] + df['low_price'] + df['trade_price']) / 3
        df['price_volume'] = df['typical_price'] * df['candle_acc_trade_volume']
        
        # Rolling VWAP
        df['vwap'] = (
            df['price_volume'].rolling(window=vwap_period).sum() /
            df['candle_acc_trade_volume'].rolling(window=vwap_period).sum()
        )
        
        # Short-term VWAP for comparison
        df['vwap_short'] = (
            df['price_volume'].rolling(window=vwap_short_period).sum() /
            df['candle_acc_trade_volume'].rolling(window=vwap_short_period).sum()
        )
        
        # VWAP bands (standard deviation bands)
        if self.parameters.get('use_vwap_bands', True):
            vwap_std_period = self.parameters.get('vwap_std_period', vwap_period)
            vwap_std_multiplier = self.parameters.get('vwap_std_multiplier', 2.0)
            
            df['vwap_std'] = df['typical_price'].rolling(window=vwap_std_period).std()
            df['vwap_upper'] = df['vwap'] + (df['vwap_std'] * vwap_std_multiplier)
            df['vwap_lower'] = df['vwap'] - (df['vwap_std'] * vwap_std_multiplier)
        
        # Price relative to VWAP
        df['price_vwap_ratio'] = df['trade_price'] / df['vwap']
        df['vwap_deviation'] = (df['trade_price'] - df['vwap']) / df['vwap']
        
        # Volume indicators
        volume_period = self.parameters.get('volume_period', 20)
        df['volume_ma'] = df['candle_acc_trade_volume'].rolling(window=volume_period).mean()
        df['volume_ratio'] = df['candle_acc_trade_volume'] / df['volume_ma']
        
        # Additional momentum indicators if enabled
        if self.parameters.get('use_momentum', True):
            momentum_period = self.parameters.get('momentum_period', 14)
            df['momentum'] = df['trade_price'] / df['trade_price'].shift(momentum_period) - 1
        
        return df
    
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate buy/sell signals based on VWAP strategy"""
        df['signal'] = 0  # 0: hold, 1: buy, -1: sell
        
        # Strategy parameters
        vwap_threshold = self.parameters.get('vwap_threshold', 0.005)  # 0.5% threshold
        volume_threshold = self.parameters.get('volume_threshold', 1.2)  # 20% above average
        momentum_threshold = self.parameters.get('momentum_threshold', 0.02)  # 2% momentum
        
        # Basic VWAP conditions
        price_below_vwap = df['vwap_deviation'] < -vwap_threshold
        price_above_vwap = df['vwap_deviation'] > vwap_threshold
        
        # Volume confirmation
        high_volume = df['volume_ratio'] > volume_threshold
        
        # Additional conditions based on strategy variant
        strategy_variant = self.parameters.get('strategy_variant', 'mean_reversion')
        
        if strategy_variant == 'mean_reversion':
            # Mean reversion: buy when price is below VWAP, sell when above
            buy_condition = (
                price_below_vwap &
                high_volume
            )
            
            sell_condition = (
                price_above_vwap &
                high_volume
            )
            
            # Use VWAP bands if available
            if 'vwap_lower' in df.columns and 'vwap_upper' in df.columns:
                buy_condition = buy_condition | (
                    (df['trade_price'] < df['vwap_lower']) & high_volume
                )
                sell_condition = sell_condition | (
                    (df['trade_price'] > df['vwap_upper']) & high_volume
                )
        
        elif strategy_variant == 'trend_following':
            # Trend following: buy when price and short VWAP above long VWAP
            vwap_uptrend = df['vwap_short'] > df['vwap']
            vwap_downtrend = df['vwap_short'] < df['vwap']
            
            buy_condition = (
                vwap_uptrend &
                (df['trade_price'] > df['vwap']) &
                high_volume
            )
            
            sell_condition = (
                vwap_downtrend &
                (df['trade_price'] < df['vwap']) &
                high_volume
            )
        
        elif strategy_variant == 'breakout':
            # Breakout: buy on strong move above VWAP with volume
            strong_breakout_up = df['vwap_deviation'] > (vwap_threshold * 2)
            strong_breakout_down = df['vwap_deviation'] < -(vwap_threshold * 2)
            
            buy_condition = (
                strong_breakout_up &
                high_volume
            )
            
            sell_condition = (
                strong_breakout_down &
                high_volume
            )
        
        else:  # 'combined' strategy
            # Combined approach using multiple signals
            momentum_up = (df['momentum'] > momentum_threshold) if 'momentum' in df.columns else True
            momentum_down = (df['momentum'] < -momentum_threshold) if 'momentum' in df.columns else True
            
            buy_condition = (
                (
                    (price_below_vwap & high_volume) |  # Mean reversion
                    ((df['trade_price'] > df['vwap']) & (df['vwap_short'] > df['vwap']) & high_volume)  # Trend
                ) &
                momentum_up
            )
            
            sell_condition = (
                (
                    (price_above_vwap & high_volume) |  # Mean reversion
                    ((df['trade_price'] < df['vwap']) & (df['vwap_short'] < df['vwap']) & high_volume)  # Trend
                ) &
                momentum_down
            )
        
        # Apply additional filters
        if self.parameters.get('use_price_filter', True):
            min_price = self.parameters.get('min_price', 1000)  # Minimum price filter
            price_filter = df['trade_price'] > min_price
            buy_condition = buy_condition & price_filter
            sell_condition = sell_condition & price_filter
        
        # Apply signals
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        return df


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands strategy with configurable parameters"""
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands and related indicators"""
        # Bollinger Bands parameters
        bb_period = self.parameters['bb_period']
        bb_std_dev = self.parameters['bb_std_dev']
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['trade_price'].rolling(window=bb_period).mean()
        bb_std = df['trade_price'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * bb_std_dev)
        df['bb_lower'] = df['bb_middle'] - (bb_std * bb_std_dev)
        
        # Bollinger Band position
        df['bb_position'] = (df['trade_price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Bollinger Band width (volatility measure)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Additional indicators
        if self.parameters.get('use_rsi', True):
            rsi_period = self.parameters.get('rsi_period', 14)
            delta = df['trade_price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate signals based on Bollinger Bands"""
        df['signal'] = 0
        
        # Strategy parameters
        lower_threshold = self.parameters.get('lower_threshold', 0.1)  # Buy when BB position < 0.1
        upper_threshold = self.parameters.get('upper_threshold', 0.9)  # Sell when BB position > 0.9
        
        # Basic Bollinger Band signals
        buy_condition = df['bb_position'] < lower_threshold
        sell_condition = df['bb_position'] > upper_threshold
        
        # Additional RSI filter if enabled
        if self.parameters.get('use_rsi', True) and 'rsi' in df.columns:
            rsi_oversold = self.parameters.get('rsi_oversold', 30)
            rsi_overbought = self.parameters.get('rsi_overbought', 70)
            
            buy_condition = buy_condition & (df['rsi'] < rsi_oversold)
            sell_condition = sell_condition & (df['rsi'] > rsi_overbought)
        
        # Volatility filter
        if self.parameters.get('use_volatility_filter', True):
            min_bb_width = self.parameters.get('min_bb_width', 0.02)  # Minimum 2% width
            volatility_filter = df['bb_width'] > min_bb_width
            buy_condition = buy_condition & volatility_filter
            sell_condition = sell_condition & volatility_filter
        
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        return df


class AdvancedVWAPStrategy(BaseStrategy):
    """Advanced VWAP strategy with real-time bands, ADX filter, and risk management"""
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate VWAP bands, ADX, and volatility indicators"""
        # VWAP calculation
        vwap_period = self.parameters['vwap_period']
        df['typical_price'] = (df['high_price'] + df['low_price'] + df['trade_price']) / 3
        df['price_volume'] = df['typical_price'] * df['candle_acc_trade_volume']
        
        # Rolling VWAP
        df['vwap'] = (
            df['price_volume'].rolling(window=vwap_period).sum() /
            df['candle_acc_trade_volume'].rolling(window=vwap_period).sum()
        )
        
        # VWAP bands using standard deviation
        vwap_std_period = self.parameters.get('vwap_std_period', vwap_period)
        vwap_std_multiplier = self.parameters.get('vwap_std_multiplier', 1.5)
        
        df['vwap_std'] = df['typical_price'].rolling(window=vwap_std_period).std()
        df['vwap_upper'] = df['vwap'] + (df['vwap_std'] * vwap_std_multiplier)
        df['vwap_lower'] = df['vwap'] - (df['vwap_std'] * vwap_std_multiplier)
        
        # ADX calculation
        df = self._calculate_adx(df)
        
        # Volatility spike detection (5-second window approximation)
        volatility_window = self.parameters.get('volatility_window', 5)  # 5 candles for 5 seconds
        df['price_change_pct'] = df['trade_price'].pct_change()
        df['volatility_spike'] = (
            df['price_change_pct'].rolling(window=volatility_window).sum().abs() * 100
        )
        
        # Volume profile
        volume_period = self.parameters.get('volume_period', 20)
        df['volume_ma'] = df['candle_acc_trade_volume'].rolling(window=volume_period).mean()
        df['volume_ratio'] = df['candle_acc_trade_volume'] / df['volume_ma']
        
        # Price distance from VWAP
        df['vwap_distance_pct'] = ((df['trade_price'] - df['vwap']) / df['vwap']) * 100
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ADX (Average Directional Index)"""
        adx_period = self.parameters.get('adx_period', 14)
        
        # True Range calculation
        df['high_low'] = df['high_price'] - df['low_price']
        df['high_close'] = abs(df['high_price'] - df['trade_price'].shift(1))
        df['low_close'] = abs(df['low_price'] - df['trade_price'].shift(1))
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        
        # Directional Movement
        df['plus_dm'] = np.where(
            (df['high_price'] - df['high_price'].shift(1)) > (df['low_price'].shift(1) - df['low_price']),
            np.maximum(df['high_price'] - df['high_price'].shift(1), 0),
            0
        )
        df['minus_dm'] = np.where(
            (df['low_price'].shift(1) - df['low_price']) > (df['high_price'] - df['high_price'].shift(1)),
            np.maximum(df['low_price'].shift(1) - df['low_price'], 0),
            0
        )
        
        # Smoothed calculations
        df['atr'] = df['tr'].rolling(window=adx_period).mean()
        df['plus_di'] = (df['plus_dm'].rolling(window=adx_period).mean() / df['atr']) * 100
        df['minus_di'] = (df['minus_dm'].rolling(window=adx_period).mean() / df['atr']) * 100
        
        # ADX calculation
        df['dx'] = (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])) * 100
        df['adx'] = df['dx'].rolling(window=adx_period).mean()
        
        # Clean up temporary columns
        df.drop(['high_low', 'high_close', 'low_close', 'tr', 'plus_dm', 'minus_dm', 'atr', 'dx'], axis=1, inplace=True)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate advanced VWAP signals with risk management"""
        df['signal'] = 0  # 0: hold, 1: buy, -1: sell
        df['signal_type'] = 'none'  # Track signal type for debugging
        df['stop_loss'] = 0.0
        df['take_profit'] = 0.0
        df['exit_signal'] = 0  # Exit signals for risk management
        
        # Strategy parameters
        adx_threshold = self.parameters.get('adx_threshold', 20)
        profit_target_pct = self.parameters.get('profit_target_pct', 0.6)
        stop_loss_pct = self.parameters.get('stop_loss_pct', 0.3)
        volatility_threshold = self.parameters.get('volatility_threshold', 0.15)
        
        # Current position tracking (simplified for backtesting)
        current_position = 0  # 1 for long, -1 for short, 0 for flat
        entry_price = 0.0
        
        for i in range(len(df)):
            if i < max(self.parameters['vwap_period'], self.parameters.get('adx_period', 14)):
                continue  # Skip until indicators are available
            
            current_row = df.iloc[i]
            
            # Check for exit conditions first
            if current_position != 0:
                exit_condition = self._check_exit_conditions(
                    current_row, entry_price, current_position, 
                    profit_target_pct, stop_loss_pct
                )
                
                if exit_condition:
                    df.iloc[i, df.columns.get_loc('exit_signal')] = -current_position
                    df.iloc[i, df.columns.get_loc('signal_type')] = exit_condition
                    current_position = 0
                    entry_price = 0.0
                    continue
            
            # Entry conditions (only when flat)
            if current_position == 0:
                # ADX filter: only trade in low ADX (sideways market)
                if current_row['adx'] <= adx_threshold:
                    
                    # Check for volatility spike
                    if current_row['volatility_spike'] >= volatility_threshold:
                        # High volatility - apply defensive logic
                        signal_info = self._handle_volatility_spike(current_row)
                        if signal_info['signal'] != 0:
                            df.iloc[i, df.columns.get_loc('signal')] = signal_info['signal']
                            df.iloc[i, df.columns.get_loc('signal_type')] = signal_info['type']
                            current_position = signal_info['signal']
                            entry_price = current_row['trade_price']
                    else:
                        # Normal conditions - band trading
                        signal_info = self._generate_band_signals(current_row)
                        if signal_info['signal'] != 0:
                            df.iloc[i, df.columns.get_loc('signal')] = signal_info['signal']
                            df.iloc[i, df.columns.get_loc('signal_type')] = signal_info['type']
                            df.iloc[i, df.columns.get_loc('stop_loss')] = signal_info['stop_loss']
                            df.iloc[i, df.columns.get_loc('take_profit')] = signal_info['take_profit']
                            current_position = signal_info['signal']
                            entry_price = current_row['trade_price']
        
        return df
    
    def _generate_band_signals(self, row) -> Dict[str, Any]:
        """Generate signals based on VWAP bands"""
        signal_info = {'signal': 0, 'type': 'none', 'stop_loss': 0.0, 'take_profit': 0.0}
        
        current_price = row['trade_price']
        vwap_upper = row['vwap_upper']
        vwap_lower = row['vwap_lower']
        
        profit_target_pct = self.parameters.get('profit_target_pct', 0.6) / 100
        stop_loss_pct = self.parameters.get('stop_loss_pct', 0.3) / 100
        
        # Buy signal: price near lower band
        band_proximity = self.parameters.get('band_proximity', 0.002)  # 0.2% proximity
        if abs(current_price - vwap_lower) / vwap_lower <= band_proximity:
            signal_info.update({
                'signal': 1,
                'type': 'band_buy',
                'take_profit': current_price * (1 + profit_target_pct),
                'stop_loss': current_price * (1 - stop_loss_pct)
            })
        
        # Sell signal: price near upper band
        elif abs(current_price - vwap_upper) / vwap_upper <= band_proximity:
            signal_info.update({
                'signal': -1,
                'type': 'band_sell',
                'take_profit': current_price * (1 - profit_target_pct),
                'stop_loss': current_price * (1 + stop_loss_pct)
            })
        
        return signal_info
    
    def _handle_volatility_spike(self, row) -> Dict[str, Any]:
        """Handle high volatility conditions"""
        signal_info = {'signal': 0, 'type': 'none'}
        
        # Volatility spike logic
        volatility_action = self.parameters.get('volatility_action', 'pause')
        
        if volatility_action == 'pause':
            # Don't trade during high volatility
            signal_info['type'] = 'volatility_pause'
        
        elif volatility_action == 'contrarian':
            # Take contrarian position on extreme moves
            vwap_distance = abs(row['vwap_distance_pct'])
            extreme_threshold = self.parameters.get('extreme_threshold', 2.0)  # 2% from VWAP
            
            if vwap_distance >= extreme_threshold:
                if row['vwap_distance_pct'] > 0:  # Price way above VWAP
                    signal_info.update({'signal': -1, 'type': 'volatility_contrarian_sell'})
                else:  # Price way below VWAP
                    signal_info.update({'signal': 1, 'type': 'volatility_contrarian_buy'})
        
        elif volatility_action == 'momentum':
            # Follow the momentum on spikes
            if row['price_change_pct'] > 0:
                signal_info.update({'signal': 1, 'type': 'volatility_momentum_buy'})
            else:
                signal_info.update({'signal': -1, 'type': 'volatility_momentum_sell'})
        
        return signal_info
    
    def _check_exit_conditions(self, row, entry_price: float, position: int, 
                              profit_target_pct: float, stop_loss_pct: float) -> str:
        """Check various exit conditions"""
        current_price = row['trade_price']
        vwap = row['vwap']
        
        profit_target_pct /= 100
        stop_loss_pct /= 100
        
        if position == 1:  # Long position
            # Take profit
            if current_price >= entry_price * (1 + profit_target_pct):
                return 'take_profit'
            
            # Stop loss
            if current_price <= entry_price * (1 - stop_loss_pct):
                return 'stop_loss'
            
            # VWAP touch exit
            if current_price <= vwap:
                return 'vwap_exit'
        
        elif position == -1:  # Short position
            # Take profit
            if current_price <= entry_price * (1 - profit_target_pct):
                return 'take_profit'
            
            # Stop loss
            if current_price >= entry_price * (1 + stop_loss_pct):
                return 'stop_loss'
            
            # VWAP touch exit
            if current_price >= vwap:
                return 'vwap_exit'
        
        return ''


class MeanReversionStrategy(BaseStrategy):
    """Z-Score based Mean Reversion strategy using Bollinger Bands"""
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands and Z-Score"""
        # Bollinger Bands parameters
        bb_period = self.parameters['bb_period']
        bb_std_dev = self.parameters['bb_std_dev']
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['trade_price'].rolling(window=bb_period).mean()
        bb_std = df['trade_price'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * bb_std_dev)
        df['bb_lower'] = df['bb_middle'] - (bb_std * bb_std_dev)
        
        # Z-Score calculation
        df['zscore'] = (df['trade_price'] - df['bb_middle']) / bb_std
        
        # Additional indicators
        if self.parameters.get('use_volume_filter', True):
            volume_period = self.parameters.get('volume_period', 20)
            df['volume_ma'] = df['candle_acc_trade_volume'].rolling(window=volume_period).mean()
            df['volume_ratio'] = df['candle_acc_trade_volume'] / df['volume_ma']
        
        return df
    
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate mean reversion signals based on Z-Score"""
        df['signal'] = 0
        
        # Strategy parameters
        entry_zscore = self.parameters.get('entry_zscore', 2.0)
        exit_zscore = self.parameters.get('exit_zscore', 0.5)
        volume_threshold = self.parameters.get('volume_threshold', 1.0)
        
        # Entry signals: extreme Z-Score values
        oversold_condition = df['zscore'] < -entry_zscore
        overbought_condition = df['zscore'] > entry_zscore
        
        # Volume filter
        if self.parameters.get('use_volume_filter', True):
            volume_filter = df['volume_ratio'] > volume_threshold
            oversold_condition = oversold_condition & volume_filter
            overbought_condition = overbought_condition & volume_filter
        
        # Apply signals
        df.loc[oversold_condition, 'signal'] = 1  # Buy on oversold
        df.loc[overbought_condition, 'signal'] = -1  # Sell on overbought
        
        # Exit signals: return to mean
        current_position = 0
        for i in range(1, len(df)):
            if current_position == 0:
                current_position = df.iloc[i]['signal']
            elif current_position == 1:  # Long position
                if df.iloc[i]['zscore'] > -exit_zscore:
                    df.iloc[i, df.columns.get_loc('signal')] = 0
                    current_position = 0
            elif current_position == -1:  # Short position
                if df.iloc[i]['zscore'] < exit_zscore:
                    df.iloc[i, df.columns.get_loc('signal')] = 0
                    current_position = 0
        
        return df


class MACDStrategy(BaseStrategy):
    """MACD strategy with signal line crossover and histogram"""
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD, Signal Line, and Histogram"""
        # MACD parameters
        fast_period = self.parameters['fast_period']
        slow_period = self.parameters['slow_period']
        signal_period = self.parameters['signal_period']
        
        # Calculate EMAs
        df['ema_fast'] = df['trade_price'].ewm(span=fast_period).mean()
        df['ema_slow'] = df['trade_price'].ewm(span=slow_period).mean()
        
        # MACD line
        df['macd'] = df['ema_fast'] - df['ema_slow']
        
        # Signal line
        df['macd_signal'] = df['macd'].ewm(span=signal_period).mean()
        
        # Histogram
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Additional indicators
        if self.parameters.get('use_rsi_filter', True):
            rsi_period = self.parameters.get('rsi_period', 14)
            delta = df['trade_price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate MACD crossover signals"""
        df['signal'] = 0
        
        # Strategy parameters
        histogram_threshold = self.parameters.get('histogram_threshold', 0)
        
        # MACD crossover signals
        macd_bullish = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        macd_bearish = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        # Histogram filter
        if self.parameters.get('use_histogram_filter', True):
            histogram_bullish = df['macd_histogram'] > histogram_threshold
            histogram_bearish = df['macd_histogram'] < -histogram_threshold
            
            macd_bullish = macd_bullish & histogram_bullish
            macd_bearish = macd_bearish & histogram_bearish
        
        # RSI filter
        if self.parameters.get('use_rsi_filter', True):
            rsi_oversold = self.parameters.get('rsi_oversold', 30)
            rsi_overbought = self.parameters.get('rsi_overbought', 70)
            
            rsi_bullish = df['rsi'] < rsi_overbought
            rsi_bearish = df['rsi'] > rsi_oversold
            
            macd_bullish = macd_bullish & rsi_bullish
            macd_bearish = macd_bearish & rsi_bearish
        
        df.loc[macd_bullish, 'signal'] = 1
        df.loc[macd_bearish, 'signal'] = -1
        
        return df


class StochasticStrategy(BaseStrategy):
    """Stochastic Oscillator strategy with %K and %D lines"""
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        # Stochastic parameters
        k_period = self.parameters['k_period']
        d_period = self.parameters['d_period']
        smooth_k = self.parameters.get('smooth_k', 3)
        
        # Calculate %K
        lowest_low = df['low_price'].rolling(window=k_period).min()
        highest_high = df['high_price'].rolling(window=k_period).max()
        
        df['stoch_k_raw'] = ((df['trade_price'] - lowest_low) / (highest_high - lowest_low)) * 100
        df['stoch_k'] = df['stoch_k_raw'].rolling(window=smooth_k).mean()
        
        # Calculate %D
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        # Volume confirmation
        if self.parameters.get('use_volume_confirmation', True):
            volume_period = self.parameters.get('volume_period', 20)
            df['volume_ma'] = df['candle_acc_trade_volume'].rolling(window=volume_period).mean()
            df['volume_ratio'] = df['candle_acc_trade_volume'] / df['volume_ma']
        
        return df
    
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate Stochastic crossover signals"""
        df['signal'] = 0
        
        # Strategy parameters
        oversold_level = self.parameters.get('oversold_level', 20)
        overbought_level = self.parameters.get('overbought_level', 80)
        
        # Stochastic crossover signals
        stoch_bullish = (
            (df['stoch_k'] > df['stoch_d']) & 
            (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1)) &
            (df['stoch_k'] < oversold_level)
        )
        
        stoch_bearish = (
            (df['stoch_k'] < df['stoch_d']) & 
            (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1)) &
            (df['stoch_k'] > overbought_level)
        )
        
        # Volume confirmation
        if self.parameters.get('use_volume_confirmation', True):
            volume_threshold = self.parameters.get('volume_threshold', 1.2)
            volume_filter = df['volume_ratio'] > volume_threshold
            
            stoch_bullish = stoch_bullish & volume_filter
            stoch_bearish = stoch_bearish & volume_filter
        
        df.loc[stoch_bullish, 'signal'] = 1
        df.loc[stoch_bearish, 'signal'] = -1
        
        return df


class PairsStrategy(BaseStrategy):
    """Pairs trading strategy for correlated cryptocurrencies"""
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(parameters)
        self.price_data = {}  # Store price data for correlation calculation
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate spread and correlation indicators"""
        # This strategy requires multiple assets, simplified for single asset
        lookback_period = self.parameters.get('lookback_period', 60)
        
        # Price ratio (for pairs trading, this would be price_A / price_B)
        df['price_ma'] = df['trade_price'].rolling(window=lookback_period).mean()
        df['price_ratio'] = df['trade_price'] / df['price_ma']
        
        # Z-score of the ratio
        df['ratio_mean'] = df['price_ratio'].rolling(window=lookback_period).mean()
        df['ratio_std'] = df['price_ratio'].rolling(window=lookback_period).std()
        df['spread_zscore'] = (df['price_ratio'] - df['ratio_mean']) / df['ratio_std']
        
        # Correlation (simplified - in real pairs trading, calculate between two assets)
        df['correlation'] = df['trade_price'].rolling(window=lookback_period).corr(df['price_ma'])
        
        return df
    
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate pairs trading signals"""
        df['signal'] = 0
        
        # Strategy parameters
        entry_threshold = self.parameters.get('entry_threshold', 2.0)
        exit_threshold = self.parameters.get('exit_threshold', 0.5)
        min_correlation = self.parameters.get('min_correlation', 0.7)
        
        # Only trade when correlation is high
        high_correlation = df['correlation'].abs() > min_correlation
        
        # Entry signals
        long_signal = (df['spread_zscore'] < -entry_threshold) & high_correlation
        short_signal = (df['spread_zscore'] > entry_threshold) & high_correlation
        
        # Exit signals (mean reversion)
        current_position = 0
        for i in range(1, len(df)):
            if current_position == 0:
                if long_signal.iloc[i]:
                    df.iloc[i, df.columns.get_loc('signal')] = 1
                    current_position = 1
                elif short_signal.iloc[i]:
                    df.iloc[i, df.columns.get_loc('signal')] = -1
                    current_position = -1
            else:
                # Exit condition: spread returns to mean
                if abs(df.iloc[i]['spread_zscore']) < exit_threshold:
                    df.iloc[i, df.columns.get_loc('signal')] = 0
                    current_position = 0
        
        return df


# Import ensemble strategy if available
ensemble_available = False
try:
    # For when strategies.py is imported as a module
    from .ensemble_strategy import EnsembleStrategy
    ensemble_available = True
except ImportError:
    try:
        # For when running directly
        from ensemble_strategy import EnsembleStrategy
        ensemble_available = True
    except ImportError:
        try:
            # For when imported from backtesting
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from ensemble_strategy import EnsembleStrategy
            ensemble_available = True
        except ImportError:
            pass

# Strategy registry
STRATEGIES = {
    'basic_momentum': BasicMomentumStrategy,
    'vwap': VWAPStrategy,
    'bollinger_bands': BollingerBandsStrategy,
    'advanced_vwap': AdvancedVWAPStrategy,
    'mean_reversion': MeanReversionStrategy,
    'macd': MACDStrategy,
    'stochastic': StochasticStrategy,
    'pairs': PairsStrategy,
}

# Add ensemble strategy if available
if ensemble_available:
    STRATEGIES['ensemble'] = EnsembleStrategy


def get_strategy(strategy_name: str, parameters: Dict[str, Any]) -> BaseStrategy:
    """Factory function to get strategy instance"""
    if strategy_name not in STRATEGIES:
        available = ', '.join(STRATEGIES.keys())
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available strategies: {available}")
    
    strategy_class = STRATEGIES[strategy_name]
    return strategy_class(parameters)