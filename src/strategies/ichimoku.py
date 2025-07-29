import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import BaseStrategy


class IchimokuCloudStrategy(BaseStrategy):
    """Ichimoku Cloud strategy for trend identification and trading signals"""
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku Cloud components"""
        # Ichimoku parameters
        tenkan_period = self.parameters.get('tenkan_period', 9)  # Conversion Line
        kijun_period = self.parameters.get('kijun_period', 26)  # Base Line
        senkou_b_period = self.parameters.get('senkou_b_period', 52)  # Leading Span B
        chikou_period = self.parameters.get('chikou_period', 26)  # Lagging Span
        
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        high_tenkan = df['high_price'].rolling(window=tenkan_period).max()
        low_tenkan = df['low_price'].rolling(window=tenkan_period).min()
        df['tenkan_sen'] = (high_tenkan + low_tenkan) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        high_kijun = df['high_price'].rolling(window=kijun_period).max()
        low_kijun = df['low_price'].rolling(window=kijun_period).min()
        df['kijun_sen'] = (high_kijun + low_kijun) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, plotted 26 periods ahead
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(kijun_period)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
        high_senkou = df['high_price'].rolling(window=senkou_b_period).max()
        low_senkou = df['low_price'].rolling(window=senkou_b_period).min()
        df['senkou_span_b'] = ((high_senkou + low_senkou) / 2).shift(kijun_period)
        
        # Chikou Span (Lagging Span): Close plotted 26 periods behind
        df['chikou_span'] = df['trade_price'].shift(-chikou_period)
        
        # Cloud thickness and color (for trend strength)
        df['cloud_thickness'] = abs(df['senkou_span_a'] - df['senkou_span_b'])
        df['cloud_green'] = df['senkou_span_a'] > df['senkou_span_b']  # Bullish cloud
        
        # Price position relative to cloud
        df['price_above_cloud'] = df['trade_price'] > df[['senkou_span_a', 'senkou_span_b']].max(axis=1)
        df['price_below_cloud'] = df['trade_price'] < df[['senkou_span_a', 'senkou_span_b']].min(axis=1)
        df['price_in_cloud'] = ~(df['price_above_cloud'] | df['price_below_cloud'])
        
        # Additional trend strength indicators
        if self.parameters.get('use_trend_strength', True):
            # TK Cross strength
            df['tk_diff'] = df['tenkan_sen'] - df['kijun_sen']
            df['tk_cross_up'] = (df['tk_diff'] > 0) & (df['tk_diff'].shift(1) <= 0)
            df['tk_cross_down'] = (df['tk_diff'] < 0) & (df['tk_diff'].shift(1) >= 0)
            
            # Distance from cloud
            cloud_top = df[['senkou_span_a', 'senkou_span_b']].max(axis=1)
            cloud_bottom = df[['senkou_span_a', 'senkou_span_b']].min(axis=1)
            df['distance_from_cloud'] = np.where(
                df['price_above_cloud'], 
                (df['trade_price'] - cloud_top) / cloud_top,
                np.where(
                    df['price_below_cloud'],
                    (df['trade_price'] - cloud_bottom) / cloud_bottom,
                    0
                )
            )
        
        return df
    
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate Ichimoku trading signals"""
        df['signal'] = 0
        
        # Strategy parameters
        min_cloud_thickness = self.parameters.get('min_cloud_thickness', 0.001)  # 0.1% minimum
        distance_threshold = self.parameters.get('distance_threshold', 0.02)  # 2% from cloud
        
        # Strategy variants
        strategy_variant = self.parameters.get('strategy_variant', 'classic')
        
        if strategy_variant == 'classic':
            # Classic Ichimoku signals
            # Strong buy: Price above cloud, Tenkan > Kijun, Chikou > Price 26 periods ago
            strong_buy = (
                df['price_above_cloud'] &
                (df['tenkan_sen'] > df['kijun_sen']) &
                df['cloud_green'] &
                (df['cloud_thickness'] > min_cloud_thickness)
            )
            
            # Strong sell: Price below cloud, Tenkan < Kijun, Chikou < Price 26 periods ago
            strong_sell = (
                df['price_below_cloud'] &
                (df['tenkan_sen'] < df['kijun_sen']) &
                ~df['cloud_green'] &
                (df['cloud_thickness'] > min_cloud_thickness)
            )
            
            df.loc[strong_buy, 'signal'] = 1
            df.loc[strong_sell, 'signal'] = -1
            
        elif strategy_variant == 'breakout':
            # Cloud breakout strategy
            # Buy on cloud breakout with momentum
            cloud_breakout_up = (
                df['price_above_cloud'] &
                df['price_in_cloud'].shift(1) &
                (df['tenkan_sen'] > df['kijun_sen']) &
                df['cloud_green']
            )
            
            # Sell on cloud breakdown
            cloud_breakout_down = (
                df['price_below_cloud'] &
                df['price_in_cloud'].shift(1) &
                (df['tenkan_sen'] < df['kijun_sen']) &
                ~df['cloud_green']
            )
            
            df.loc[cloud_breakout_up, 'signal'] = 1
            df.loc[cloud_breakout_down, 'signal'] = -1
            
        elif strategy_variant == 'tk_cross':
            # Tenkan-Kijun cross strategy
            # Buy on TK cross above cloud
            tk_buy = (
                df['tk_cross_up'] &
                df['price_above_cloud'] &
                df['cloud_green']
            )
            
            # Sell on TK cross below cloud
            tk_sell = (
                df['tk_cross_down'] &
                df['price_below_cloud'] &
                ~df['cloud_green']
            )
            
            df.loc[tk_buy, 'signal'] = 1
            df.loc[tk_sell, 'signal'] = -1
            
        else:  # 'hybrid' strategy
            # Combine multiple Ichimoku signals
            # Entry conditions
            bullish_setup = (
                (df['price_above_cloud'] | (df['price_in_cloud'] & df['cloud_green'])) &
                (df['tenkan_sen'] > df['kijun_sen']) &
                (df['distance_from_cloud'] < distance_threshold)
            )
            
            bearish_setup = (
                (df['price_below_cloud'] | (df['price_in_cloud'] & ~df['cloud_green'])) &
                (df['tenkan_sen'] < df['kijun_sen']) &
                (abs(df['distance_from_cloud']) < distance_threshold)
            )
            
            # Confirmation with TK cross or cloud color change
            bullish_confirmation = (
                df['tk_cross_up'] |
                (df['cloud_green'] & ~df['cloud_green'].shift(1))
            )
            
            bearish_confirmation = (
                df['tk_cross_down'] |
                (~df['cloud_green'] & df['cloud_green'].shift(1))
            )
            
            df.loc[bullish_setup & bullish_confirmation, 'signal'] = 1
            df.loc[bearish_setup & bearish_confirmation, 'signal'] = -1
        
        # Additional filters
        if self.parameters.get('use_volume_filter', True):
            volume_period = self.parameters.get('volume_period', 20)
            df['volume_ma'] = df['candle_acc_trade_volume'].rolling(window=volume_period).mean()
            df['volume_ratio'] = df['candle_acc_trade_volume'] / df['volume_ma']
            volume_threshold = self.parameters.get('volume_threshold', 1.2)
            
            # Only keep signals with high volume
            df.loc[(df['signal'] != 0) & (df['volume_ratio'] < volume_threshold), 'signal'] = 0
        
        return df
    
    def _should_buy(self, last_row, df) -> bool:
        """Buy signal for real-time trading"""
        if pd.isna(last_row.get('price_above_cloud', False)):
            return False
        
        # Basic buy condition
        return (
            last_row.get('price_above_cloud', False) and
            last_row.get('tenkan_sen', 0) > last_row.get('kijun_sen', 0) and
            last_row.get('cloud_green', False)
        )
    
    def _should_sell(self, last_row, df) -> bool:
        """Sell signal for real-time trading"""
        if pd.isna(last_row.get('price_below_cloud', False)):
            return False
        
        # Basic sell condition
        return (
            last_row.get('price_below_cloud', False) and
            last_row.get('tenkan_sen', 0) < last_row.get('kijun_sen', 0) and
            not last_row.get('cloud_green', True)
        )