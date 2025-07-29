import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import BaseStrategy


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