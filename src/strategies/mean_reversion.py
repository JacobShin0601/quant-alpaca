import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import BaseStrategy


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
        signal_col_idx = df.columns.get_loc('signal')
        
        for i in range(1, len(df)):
            if current_position == 0:
                current_position = df.iloc[i]['signal']
            elif current_position == 1:  # Long position
                if df.iloc[i]['zscore'] > -exit_zscore:
                    df.iat[i, signal_col_idx] = 0
                    current_position = 0
            elif current_position == -1:  # Short position
                if df.iloc[i]['zscore'] < exit_zscore:
                    df.iat[i, signal_col_idx] = 0
                    current_position = 0
        
        return df