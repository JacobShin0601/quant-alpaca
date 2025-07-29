import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import BaseStrategy


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