import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import BaseStrategy


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