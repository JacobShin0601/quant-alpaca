import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import BaseStrategy


class KeltnerChannelsStrategy(BaseStrategy):
    """Keltner Channels strategy using EMA and ATR-based channels"""
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Keltner Channels"""
        # Parameters
        ema_period = self.parameters.get('ema_period', 20)
        atr_period = self.parameters.get('atr_period', 10)
        multiplier = self.parameters.get('multiplier', 2.0)
        
        # Calculate EMA (middle line)
        df['kc_middle'] = df['trade_price'].ewm(span=ema_period, adjust=False).mean()
        
        # Calculate ATR
        df['high_low'] = df['high_price'] - df['low_price']
        df['high_close'] = abs(df['high_price'] - df['trade_price'].shift(1))
        df['low_close'] = abs(df['low_price'] - df['trade_price'].shift(1))
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['true_range'].rolling(window=atr_period).mean()
        
        # Calculate channels
        df['kc_upper'] = df['kc_middle'] + (multiplier * df['atr'])
        df['kc_lower'] = df['kc_middle'] - (multiplier * df['atr'])
        
        # Position within channels
        df['kc_position'] = (df['trade_price'] - df['kc_lower']) / (df['kc_upper'] - df['kc_lower'])
        
        # Channel width (volatility measure)
        df['kc_width'] = (df['kc_upper'] - df['kc_lower']) / df['kc_middle']
        
        # Breakout detection
        df['kc_breakout_up'] = df['trade_price'] > df['kc_upper']
        df['kc_breakout_down'] = df['trade_price'] < df['kc_lower']
        
        # Trend direction based on middle line
        df['kc_trend'] = np.where(df['kc_middle'] > df['kc_middle'].shift(1), 1, -1)
        
        # Clean up
        df.drop(['high_low', 'high_close', 'low_close'], axis=1, inplace=True)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate Keltner Channels signals"""
        df['signal'] = 0
        
        # Ensure we have enough data
        if len(df) < 50:
            return df
            
        # Strategy variant
        strategy_variant = self.parameters.get('strategy_variant', 'mean_reversion')
        
        # Fill NaN values to avoid issues
        df = df.ffill().fillna(0)
        
        if strategy_variant == 'mean_reversion':
            # Buy at lower channel, sell at upper channel
            buy_signal = (
                (df['kc_position'] < 0.3) &  # Near lower band (more liberal)
                (df['kc_width'] > 0.01) &  # Sufficient volatility (more liberal)
                (df['kc_position'].notna()) &  # Valid position
                (df['kc_width'].notna())  # Valid width
            )
            
            sell_signal = (
                (df['kc_position'] > 0.7) &  # Near upper band (more liberal)
                (df['kc_width'] > 0.01) &
                (df['kc_position'].notna()) &  # Valid position
                (df['kc_width'].notna())  # Valid width
            )
            
        elif strategy_variant == 'breakout':
            # Trade breakouts
            buy_signal = (
                df['kc_breakout_up'] & 
                (df['kc_breakout_up'].shift(1) == False) &  # New breakout (removed trend filter)
                (df['kc_breakout_up'].notna()) &  # Valid breakout data
                (df['kc_width'] > 0.005)  # Minimum channel width
            )
            
            sell_signal = (
                df['kc_breakout_down'] & 
                (df['kc_breakout_down'].shift(1) == False) &  # New breakout (removed trend filter)
                (df['kc_breakout_down'].notna()) &  # Valid breakout data
                (df['kc_width'] > 0.005)  # Minimum channel width
            )
            
        else:  # 'squeeze'
            # Trade when channels squeeze (low volatility)
            squeeze_threshold = self.parameters.get('squeeze_threshold', 0.015)
            squeeze = (df['kc_width'] < squeeze_threshold) & (df['kc_width'].notna())
            
            buy_signal = (
                squeeze & 
                (df['trade_price'] > df['kc_middle']) &
                (df['kc_middle'].notna())
            )
            
            sell_signal = (
                squeeze & 
                (df['trade_price'] < df['kc_middle']) &
                (df['kc_middle'].notna())
            )
        
        df.loc[buy_signal, 'signal'] = 1
        df.loc[sell_signal, 'signal'] = -1
        
        return df