import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import BaseStrategy


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