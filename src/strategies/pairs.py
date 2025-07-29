import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import BaseStrategy


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