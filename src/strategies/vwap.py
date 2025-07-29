import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import BaseStrategy


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