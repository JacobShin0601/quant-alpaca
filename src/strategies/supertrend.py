import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import BaseStrategy


class SuperTrendStrategy(BaseStrategy):
    """SuperTrend strategy using ATR-based dynamic support and resistance levels"""
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate SuperTrend indicator"""
        # SuperTrend parameters
        atr_period = self.parameters.get('atr_period', 10)
        multiplier = self.parameters.get('multiplier', 3.0)
        
        # Calculate ATR (Average True Range)
        df['high_low'] = df['high_price'] - df['low_price']
        df['high_close'] = abs(df['high_price'] - df['trade_price'].shift(1))
        df['low_close'] = abs(df['low_price'] - df['trade_price'].shift(1))
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['true_range'].rolling(window=atr_period).mean()
        
        # Calculate basic bands
        hl2 = (df['high_price'] + df['low_price']) / 2
        df['upper_band'] = hl2 + (multiplier * df['atr'])
        df['lower_band'] = hl2 - (multiplier * df['atr'])
        
        # Initialize SuperTrend
        df['supertrend'] = 0.0
        df['supertrend_direction'] = 1  # 1 for uptrend, -1 for downtrend
        
        # Calculate SuperTrend values
        for i in range(atr_period, len(df)):
            # Current price
            curr_close = df['trade_price'].iloc[i]
            
            # Previous values
            prev_close = df['trade_price'].iloc[i-1] if i > 0 else curr_close
            prev_supertrend = df['supertrend'].iloc[i-1] if i > 0 else df['lower_band'].iloc[i]
            prev_direction = df['supertrend_direction'].iloc[i-1] if i > 0 else 1
            
            # Current bands
            curr_upper = df['upper_band'].iloc[i]
            curr_lower = df['lower_band'].iloc[i]
            
            # Adjust bands based on previous values
            if curr_lower > prev_supertrend or prev_close < prev_supertrend:
                final_lower = curr_lower
            else:
                final_lower = prev_supertrend if prev_direction > 0 else curr_lower
                
            if curr_upper < prev_supertrend or prev_close > prev_supertrend:
                final_upper = curr_upper
            else:
                final_upper = prev_supertrend if prev_direction < 0 else curr_upper
            
            # Determine current SuperTrend and direction
            if prev_direction > 0:  # Was in uptrend
                if curr_close <= final_lower:
                    df.iloc[i, df.columns.get_loc('supertrend')] = final_upper
                    df.iloc[i, df.columns.get_loc('supertrend_direction')] = -1
                else:
                    df.iloc[i, df.columns.get_loc('supertrend')] = final_lower
                    df.iloc[i, df.columns.get_loc('supertrend_direction')] = 1
            else:  # Was in downtrend
                if curr_close >= final_upper:
                    df.iloc[i, df.columns.get_loc('supertrend')] = final_lower
                    df.iloc[i, df.columns.get_loc('supertrend_direction')] = 1
                else:
                    df.iloc[i, df.columns.get_loc('supertrend')] = final_upper
                    df.iloc[i, df.columns.get_loc('supertrend_direction')] = -1
        
        # Calculate trend changes
        df['trend_change'] = df['supertrend_direction'].diff()
        df['bullish_signal'] = df['trend_change'] > 0  # Changed from bearish to bullish
        df['bearish_signal'] = df['trend_change'] < 0  # Changed from bullish to bearish
        
        # Additional indicators
        if self.parameters.get('use_confirmation', True):
            # Price position relative to SuperTrend
            df['price_above_st'] = df['trade_price'] > df['supertrend']
            df['distance_from_st'] = (df['trade_price'] - df['supertrend']) / df['supertrend']
            
            # Trend strength using consecutive candles in same direction
            df['trend_strength'] = 0
            consecutive_count = 0
            prev_direction = 0
            
            for i in range(len(df)):
                curr_direction = df['supertrend_direction'].iloc[i]
                if curr_direction == prev_direction:
                    consecutive_count += 1
                else:
                    consecutive_count = 1
                df.iloc[i, df.columns.get_loc('trend_strength')] = consecutive_count * curr_direction
                prev_direction = curr_direction
        
        # Volume analysis
        if self.parameters.get('use_volume_analysis', True):
            volume_period = self.parameters.get('volume_period', 20)
            df['volume_ma'] = df['candle_acc_trade_volume'].rolling(window=volume_period).mean()
            df['volume_ratio'] = df['candle_acc_trade_volume'] / df['volume_ma']
        
        # Clean up temporary columns
        df.drop(['high_low', 'high_close', 'low_close'], axis=1, inplace=True)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate SuperTrend trading signals"""
        df['signal'] = 0
        
        # Strategy parameters
        min_trend_strength = self.parameters.get('min_trend_strength', 3)
        volume_threshold = self.parameters.get('volume_threshold', 1.2)
        distance_threshold = self.parameters.get('distance_threshold', 0.02)  # 2% from SuperTrend
        
        # Strategy variants
        strategy_variant = self.parameters.get('strategy_variant', 'classic')
        
        if strategy_variant == 'classic':
            # Classic SuperTrend signals - trade on trend changes
            buy_signal = df['bullish_signal']
            sell_signal = df['bearish_signal']
            
            # Apply volume filter if enabled
            if self.parameters.get('use_volume_analysis', True):
                buy_signal = buy_signal & (df['volume_ratio'] > volume_threshold)
                sell_signal = sell_signal & (df['volume_ratio'] > volume_threshold)
            
            df.loc[buy_signal, 'signal'] = 1
            df.loc[sell_signal, 'signal'] = -1
            
        elif strategy_variant == 'pullback':
            # Trade on pullbacks to SuperTrend line
            # Buy when price touches SuperTrend in uptrend
            pullback_buy = (
                (df['supertrend_direction'] == 1) &
                (abs(df['distance_from_st']) < distance_threshold) &
                (df['trade_price'] > df['trade_price'].shift(1)) &  # Price starting to bounce
                (df['trend_strength'] >= min_trend_strength)
            )
            
            # Sell when price touches SuperTrend in downtrend
            pullback_sell = (
                (df['supertrend_direction'] == -1) &
                (abs(df['distance_from_st']) < distance_threshold) &
                (df['trade_price'] < df['trade_price'].shift(1)) &  # Price starting to drop
                (df['trend_strength'] <= -min_trend_strength)
            )
            
            df.loc[pullback_buy, 'signal'] = 1
            df.loc[pullback_sell, 'signal'] = -1
            
        elif strategy_variant == 'breakout':
            # Trade on strong moves away from SuperTrend
            # Strong bullish breakout
            strong_buy = (
                (df['supertrend_direction'] == 1) &
                (df['distance_from_st'] > distance_threshold) &
                (df['volume_ratio'] > volume_threshold) &
                (df['bullish_signal'] | (df['trend_strength'] == 1))  # New trend or just started
            )
            
            # Strong bearish breakout
            strong_sell = (
                (df['supertrend_direction'] == -1) &
                (df['distance_from_st'] < -distance_threshold) &
                (df['volume_ratio'] > volume_threshold) &
                (df['bearish_signal'] | (df['trend_strength'] == -1))  # New trend or just started
            )
            
            df.loc[strong_buy, 'signal'] = 1
            df.loc[strong_sell, 'signal'] = -1
            
        else:  # 'filtered' strategy
            # Use multiple confirmations
            # Buy conditions
            buy_setup = (
                df['bullish_signal'] |  # Trend change to bullish
                (
                    (df['supertrend_direction'] == 1) &  # In uptrend
                    (df['trend_strength'] >= min_trend_strength) &  # Strong trend
                    (df['trade_price'] > df['supertrend'].shift(1)) &  # Price above previous ST
                    (df['trade_price'].pct_change() > 0)  # Positive momentum
                )
            )
            
            # Sell conditions
            sell_setup = (
                df['bearish_signal'] |  # Trend change to bearish
                (
                    (df['supertrend_direction'] == -1) &  # In downtrend
                    (df['trend_strength'] <= -min_trend_strength) &  # Strong trend
                    (df['trade_price'] < df['supertrend'].shift(1)) &  # Price below previous ST
                    (df['trade_price'].pct_change() < 0)  # Negative momentum
                )
            )
            
            # Apply filters
            if self.parameters.get('use_volume_analysis', True):
                buy_setup = buy_setup & (df['volume_ratio'] > volume_threshold)
                sell_setup = sell_setup & (df['volume_ratio'] > volume_threshold)
            
            # ATR filter for volatility
            if self.parameters.get('use_atr_filter', True):
                atr_min = self.parameters.get('atr_min_threshold', 0.001)  # 0.1% minimum ATR
                atr_filter = (df['atr'] / df['trade_price']) > atr_min
                buy_setup = buy_setup & atr_filter
                sell_setup = sell_setup & atr_filter
            
            df.loc[buy_setup, 'signal'] = 1
            df.loc[sell_setup, 'signal'] = -1
        
        return df
    
    def _should_buy(self, last_row, df) -> bool:
        """Buy signal for real-time trading"""
        if pd.isna(last_row.get('supertrend_direction', 0)):
            return False
        
        # Basic buy condition: in uptrend and price above SuperTrend
        return (
            last_row.get('supertrend_direction', 0) == 1 and
            last_row.get('trade_price', 0) > last_row.get('supertrend', float('inf'))
        )
    
    def _should_sell(self, last_row, df) -> bool:
        """Sell signal for real-time trading"""
        if pd.isna(last_row.get('supertrend_direction', 0)):
            return False
        
        # Basic sell condition: in downtrend and price below SuperTrend
        return (
            last_row.get('supertrend_direction', 0) == -1 and
            last_row.get('trade_price', 0) < last_row.get('supertrend', 0)
        )