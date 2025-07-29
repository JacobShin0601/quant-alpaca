import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import BaseStrategy


class DonchianChannelsStrategy(BaseStrategy):
    """Donchian Channels (Price Channels) breakout strategy"""
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Donchian Channels"""
        # Parameters
        upper_period = self.parameters.get('upper_period', 20)
        lower_period = self.parameters.get('lower_period', 20)
        middle_period = self.parameters.get('middle_period', 10)
        
        # Calculate channels
        df['dc_upper'] = df['high_price'].rolling(window=upper_period).max()
        df['dc_lower'] = df['low_price'].rolling(window=lower_period).min()
        df['dc_middle'] = (df['dc_upper'] + df['dc_lower']) / 2
        
        # Alternative middle line using shorter period
        df['dc_middle_alt'] = (
            df['high_price'].rolling(window=middle_period).max() + 
            df['low_price'].rolling(window=middle_period).min()
        ) / 2
        
        # Channel width
        df['dc_width'] = df['dc_upper'] - df['dc_lower']
        df['dc_width_pct'] = (df['dc_width'] / df['dc_middle']) * 100
        
        # Position within channel
        df['dc_position'] = (df['trade_price'] - df['dc_lower']) / (df['dc_upper'] - df['dc_lower'])
        
        # Breakout detection
        df['dc_upper_break'] = df['trade_price'] > df['dc_upper'].shift(1)
        df['dc_lower_break'] = df['trade_price'] < df['dc_lower'].shift(1)
        
        # New highs/lows
        df['new_high'] = df['high_price'] == df['dc_upper']
        df['new_low'] = df['low_price'] == df['dc_lower']
        
        # Trend detection
        df['dc_trend'] = np.where(
            df['dc_middle'] > df['dc_middle'].shift(1), 1,
            np.where(df['dc_middle'] < df['dc_middle'].shift(1), -1, 0)
        )
        
        # Volume analysis
        if self.parameters.get('use_volume', True):
            volume_period = self.parameters.get('volume_period', 20)
            df['volume_ma'] = df['candle_acc_trade_volume'].rolling(window=volume_period).mean()
            df['volume_ratio'] = df['candle_acc_trade_volume'] / df['volume_ma']
        
        return df
    
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate Donchian Channels signals"""
        df['signal'] = 0
        
        # Strategy parameters
        min_width_pct = self.parameters.get('min_width_pct', 1.0)  # Minimum 1% channel width
        volume_threshold = self.parameters.get('volume_threshold', 1.2)
        
        # Strategy variant
        strategy_variant = self.parameters.get('strategy_variant', 'breakout')
        
        if strategy_variant == 'breakout':
            # Classic breakout strategy
            buy_signal = (
                df['dc_upper_break'] &
                (df['dc_width_pct'] > min_width_pct) &
                (df['dc_trend'] >= 0)
            )
            
            sell_signal = (
                df['dc_lower_break'] &
                (df['dc_width_pct'] > min_width_pct) &
                (df['dc_trend'] <= 0)
            )
            
            # Volume confirmation
            if self.parameters.get('use_volume', True):
                buy_signal = buy_signal & (df['volume_ratio'] > volume_threshold)
                sell_signal = sell_signal & (df['volume_ratio'] > volume_threshold)
                
        elif strategy_variant == 'reversal':
            # Trade reversals at channel extremes
            buy_signal = (
                (df['dc_position'] < 0.1) &  # Near lower channel
                ~df['new_low'] &  # Not making new lows
                (df['trade_price'] > df['trade_price'].shift(1))  # Price turning up
            )
            
            sell_signal = (
                (df['dc_position'] > 0.9) &  # Near upper channel
                ~df['new_high'] &  # Not making new highs
                (df['trade_price'] < df['trade_price'].shift(1))  # Price turning down
            )
            
        else:  # 'turtle'
            # Turtle trading system inspired
            # Entry on 20-day breakout, exit on 10-day opposite breakout
            exit_upper = df['high_price'].rolling(window=10).max()
            exit_lower = df['low_price'].rolling(window=10).min()
            
            # Long entry and exit
            long_entry = df['dc_upper_break'] & (df['dc_trend'] > 0)
            long_exit = df['trade_price'] < exit_lower
            
            # Short entry and exit
            short_entry = df['dc_lower_break'] & (df['dc_trend'] < 0)
            short_exit = df['trade_price'] > exit_upper
            
            # Track position
            position = 0
            for i in range(len(df)):
                if position == 0:
                    if long_entry.iloc[i]:
                        df.iloc[i, df.columns.get_loc('signal')] = 1
                        position = 1
                    elif short_entry.iloc[i]:
                        df.iloc[i, df.columns.get_loc('signal')] = -1
                        position = -1
                elif position == 1:
                    if long_exit.iloc[i]:
                        df.iloc[i, df.columns.get_loc('signal')] = 0
                        position = 0
                elif position == -1:
                    if short_exit.iloc[i]:
                        df.iloc[i, df.columns.get_loc('signal')] = 0
                        position = 0
            
            return df
        
        df.loc[buy_signal, 'signal'] = 1
        df.loc[sell_signal, 'signal'] = -1
        
        return df