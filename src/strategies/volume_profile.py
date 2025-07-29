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
        """Calculate Volume Profile indicators"""
        # Parameters
        profile_period = self.parameters.get('profile_period', 50)
        num_bins = self.parameters.get('num_bins', 20)
        poc_threshold = self.parameters.get('poc_threshold', 0.003)  # 0.3% proximity
        
        # Volume analysis
        df['volume_ma'] = df['candle_acc_trade_volume'].rolling(window=20).mean()
        df['volume_increasing'] = df['candle_acc_trade_volume'] > df['volume_ma']
        
        # Calculate rolling volume profile
        for i in range(profile_period, len(df)):
            window_df = df.iloc[i-profile_period:i]
            
            # Create price bins
            price_min = window_df['low_price'].min()
            price_max = window_df['high_price'].max()
            bins = np.linspace(price_min, price_max, num_bins)
            
            # Calculate volume at each price level
            volume_profile = self._calculate_volume_profile(window_df, bins)
            
            # Find Point of Control (POC) - price level with highest volume
            poc_idx = np.argmax(volume_profile)
            poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2 if poc_idx < len(bins) - 1 else bins[poc_idx]
            
            # Find Value Area (70% of volume)
            value_area_high, value_area_low = self._calculate_value_area(bins, volume_profile)
            
            # Store results
            df.loc[i, 'poc_price'] = poc_price
            df.loc[i, 'value_area_high'] = value_area_high
            df.loc[i, 'value_area_low'] = value_area_low
            
            # Check proximity to POC and value area
            current_price = df.loc[i, 'trade_price']
            df.loc[i, 'near_poc'] = abs(current_price - poc_price) / poc_price < poc_threshold
            df.loc[i, 'in_value_area'] = value_area_low <= current_price <= value_area_high
            
            # Determine if POC acts as support or resistance
            df.loc[i, 'near_support_poc'] = (
                current_price < poc_price and 
                abs(current_price - poc_price) / poc_price < poc_threshold
            )
            df.loc[i, 'near_resistance_poc'] = (
                current_price > poc_price and 
                abs(current_price - poc_price) / poc_price < poc_threshold
            )
        
        # Additional indicators
        df['price_momentum'] = df['trade_price'].pct_change(periods=10)
        df['volume_momentum'] = df['candle_acc_trade_volume'].pct_change(periods=10)
        
        return df
    
    def _calculate_volume_profile(self, window_df: pd.DataFrame, bins: np.ndarray) -> np.ndarray:
        """Calculate volume at each price level"""
        volume_profile = np.zeros(len(bins) - 1)
        
        for _, row in window_df.iterrows():
            # Distribute volume across price range of candle
            low_idx = np.searchsorted(bins, row['low_price'])
            high_idx = np.searchsorted(bins, row['high_price'])
            
            if low_idx == high_idx:
                # Entire candle in one bin
                if 0 <= low_idx - 1 < len(volume_profile):
                    volume_profile[low_idx - 1] += row['candle_acc_trade_volume']
            else:
                # Distribute volume proportionally
                for idx in range(max(0, low_idx - 1), min(high_idx, len(volume_profile))):
                    volume_profile[idx] += row['candle_acc_trade_volume'] / (high_idx - low_idx + 1)
        
        return volume_profile
    
    def _calculate_value_area(self, bins: np.ndarray, volume_profile: np.ndarray) -> tuple:
        """Calculate Value Area (70% of volume)"""
        total_volume = np.sum(volume_profile)
        value_area_volume = total_volume * 0.7
        
        # Start from POC and expand outward
        poc_idx = np.argmax(volume_profile)
        accumulated_volume = volume_profile[poc_idx]
        
        low_idx = poc_idx
        high_idx = poc_idx
        
        while accumulated_volume < value_area_volume:
            # Check which side to expand
            left_volume = volume_profile[low_idx - 1] if low_idx > 0 else 0
            right_volume = volume_profile[high_idx + 1] if high_idx < len(volume_profile) - 1 else 0
            
            if left_volume > right_volume and low_idx > 0:
                low_idx -= 1
                accumulated_volume += left_volume
            elif high_idx < len(volume_profile) - 1:
                high_idx += 1
                accumulated_volume += right_volume
            else:
                break
        
        value_area_low = bins[low_idx]
        value_area_high = bins[high_idx + 1] if high_idx < len(bins) - 1 else bins[high_idx]
        
        return value_area_high, value_area_low
    
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