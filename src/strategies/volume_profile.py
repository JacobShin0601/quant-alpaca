import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import BaseStrategy


class VolumeProfileStrategy(BaseStrategy):
    """Volume Profile strategy that identifies high volume areas as support/resistance"""
    
    def _should_buy(self, last_row, df) -> bool:
        """Buy when price touches high volume node from below"""
        if pd.isna(last_row.get('near_support_poc', np.nan)):
            return False
        
        return last_row['near_support_poc'] and last_row.get('volume_increasing', False)
    
    def _should_sell(self, last_row, df) -> bool:
        """Sell when price touches high volume node from above"""
        if pd.isna(last_row.get('near_resistance_poc', np.nan)):
            return False
        
        return last_row['near_resistance_poc'] and last_row.get('volume_increasing', False)
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume Profile indicators - optimized version"""
        # Parameters
        profile_period = self.parameters.get('profile_period', 50)
        num_bins = self.parameters.get('num_bins', 20)
        poc_threshold = self.parameters.get('poc_threshold', 0.003)  # 0.3% proximity
        
        # Volume analysis
        df['volume_ma'] = df['candle_acc_trade_volume'].rolling(window=20).mean()
        df['volume_increasing'] = df['candle_acc_trade_volume'] > df['volume_ma']
        
        # Initialize columns
        df['poc_price'] = np.nan
        df['value_area_high'] = np.nan
        df['value_area_low'] = np.nan
        df['near_poc'] = False
        df['in_value_area'] = False
        df['near_support_poc'] = False
        df['near_resistance_poc'] = False
        
        # Simplified volume profile using rolling quantiles
        # This is much faster than calculating full volume profile for each window
        
        # Use VWAP (Volume Weighted Average Price) as a proxy for POC
        df['vwap'] = (df['trade_price'] * df['candle_acc_trade_volume']).rolling(window=profile_period).sum() / df['candle_acc_trade_volume'].rolling(window=profile_period).sum()
        
        # Use price quantiles weighted by volume as value area
        # High volume price levels
        df['price_75'] = df['high_price'].rolling(window=profile_period).quantile(0.75)
        df['price_25'] = df['low_price'].rolling(window=profile_period).quantile(0.25)
        
        # Use VWAP as POC
        df['poc_price'] = df['vwap']
        df['value_area_high'] = df['price_75']
        df['value_area_low'] = df['price_25']
        
        # Check proximity to POC and value area
        df['near_poc'] = (abs(df['trade_price'] - df['poc_price']) / df['poc_price']) < poc_threshold
        df['in_value_area'] = (df['trade_price'] >= df['value_area_low']) & (df['trade_price'] <= df['value_area_high'])
        
        # Determine if POC acts as support or resistance
        df['near_support_poc'] = (
            (df['trade_price'] < df['poc_price']) & 
            (abs(df['trade_price'] - df['poc_price']) / df['poc_price'] < poc_threshold)
        )
        df['near_resistance_poc'] = (
            (df['trade_price'] > df['poc_price']) & 
            (abs(df['trade_price'] - df['poc_price']) / df['poc_price'] < poc_threshold)
        )
        
        # Additional indicators
        df['price_momentum'] = df['trade_price'].pct_change(periods=10)
        df['volume_momentum'] = df['candle_acc_trade_volume'].pct_change(periods=10)
        
        return df
    
    # Helper methods removed - using vectorized operations instead
    
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate buy/sell signals based on Volume Profile"""
        df['signal'] = 0  # 0: hold, 1: buy, -1: sell
        
        # Strategy parameters
        min_volume_ratio = self.parameters.get('min_volume_ratio', 1.2)
        momentum_threshold = self.parameters.get('momentum_threshold', 0.002)
        
        # Buy conditions
        buy_condition = (
            df['near_support_poc'] &  # Price near POC from below
            df['volume_increasing'] &  # Volume increasing
            (df['volume_momentum'] > 0) &  # Volume momentum positive
            (df['price_momentum'] > -momentum_threshold)  # Price not falling rapidly
        )
        
        # Sell conditions  
        sell_condition = (
            df['near_resistance_poc'] &  # Price near POC from above
            df['volume_increasing'] &  # Volume increasing
            (df['volume_momentum'] > 0) &  # Volume momentum positive
            (df['price_momentum'] < momentum_threshold)  # Price not rising rapidly
        )
        
        # Additional filters
        if self.parameters.get('use_value_area', True):
            # Buy when price breaks above value area low
            value_area_breakout_buy = (
                (df['trade_price'] > df['value_area_low']) &
                (df['trade_price'].shift(1) <= df['value_area_low'].shift(1)) &
                df['volume_increasing']
            )
            
            # Sell when price breaks below value area high
            value_area_breakout_sell = (
                (df['trade_price'] < df['value_area_high']) &
                (df['trade_price'].shift(1) >= df['value_area_high'].shift(1)) &
                df['volume_increasing']
            )
            
            buy_condition = buy_condition | value_area_breakout_buy
            sell_condition = sell_condition | value_area_breakout_sell
        
        # Apply signals
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        return df