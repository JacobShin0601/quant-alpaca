import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import BaseStrategy


class AroonStrategy(BaseStrategy):
    """Aroon Indicator strategy for identifying trend changes and strength"""
    
    def _should_buy(self, last_row, df) -> bool:
        """Buy when Aroon shows strong uptrend beginning"""
        if pd.isna(last_row.get('aroon_bullish_crossover', np.nan)):
            return False
        
        return (
            last_row['aroon_bullish_crossover'] or 
            (last_row.get('aroon_up', 0) > 70 and last_row.get('aroon_down', 100) < 30)
        )
    
    def _should_sell(self, last_row, df) -> bool:
        """Sell when Aroon shows strong downtrend beginning"""
        if pd.isna(last_row.get('aroon_bearish_crossover', np.nan)):
            return False
        
        return (
            last_row['aroon_bearish_crossover'] or 
            (last_row.get('aroon_down', 0) > 70 and last_row.get('aroon_up', 100) < 30)
        )
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Aroon Up, Aroon Down, and Aroon Oscillator"""
        # Parameters
        aroon_period = self.parameters.get('aroon_period', 25)
        
        # Initialize columns
        df['aroon_up'] = np.nan
        df['aroon_down'] = np.nan
        df['aroon_oscillator'] = np.nan
        df['aroon_bullish_crossover'] = False
        df['aroon_bearish_crossover'] = False
        df['trend_strength'] = 'neutral'
        
        # Calculate Aroon indicators
        for i in range(aroon_period, len(df)):
            # Get the window
            window_high = df['high_price'].iloc[i-aroon_period+1:i+1]
            window_low = df['low_price'].iloc[i-aroon_period+1:i+1]
            
            # Find days since highest high and lowest low
            days_since_high = aroon_period - window_high.argmax() - 1
            days_since_low = aroon_period - window_low.argmin() - 1
            
            # Calculate Aroon Up and Down
            aroon_up = ((aroon_period - days_since_high) / aroon_period) * 100
            aroon_down = ((aroon_period - days_since_low) / aroon_period) * 100
            
            df.loc[i, 'aroon_up'] = aroon_up
            df.loc[i, 'aroon_down'] = aroon_down
            df.loc[i, 'aroon_oscillator'] = aroon_up - aroon_down
            
            # Detect crossovers
            if i > aroon_period:
                prev_up = df.loc[i-1, 'aroon_up']
                prev_down = df.loc[i-1, 'aroon_down']
                
                # Bullish crossover: Aroon Up crosses above Aroon Down
                if prev_up <= prev_down and aroon_up > aroon_down:
                    df.loc[i, 'aroon_bullish_crossover'] = True
                
                # Bearish crossover: Aroon Down crosses above Aroon Up
                if prev_down <= prev_up and aroon_down > aroon_up:
                    df.loc[i, 'aroon_bearish_crossover'] = True
            
            # Determine trend strength
            if aroon_up > 70 and aroon_down < 30:
                df.loc[i, 'trend_strength'] = 'strong_up'
            elif aroon_down > 70 and aroon_up < 30:
                df.loc[i, 'trend_strength'] = 'strong_down'
            elif aroon_up > 50 and aroon_down < 50:
                df.loc[i, 'trend_strength'] = 'weak_up'
            elif aroon_down > 50 and aroon_up < 50:
                df.loc[i, 'trend_strength'] = 'weak_down'
            else:
                df.loc[i, 'trend_strength'] = 'neutral'
        
        # Additional trend indicators
        df['price_momentum'] = df['trade_price'].pct_change(periods=5)
        df['volume_ma'] = df['candle_acc_trade_volume'].rolling(window=20).mean()
        df['volume_increasing'] = df['candle_acc_trade_volume'] > df['volume_ma']
        
        # Moving averages for trend confirmation
        df['ma_short'] = df['trade_price'].rolling(window=10).mean()
        df['ma_long'] = df['trade_price'].rolling(window=30).mean()
        df['ma_trend_up'] = df['ma_short'] > df['ma_long']
        
        return df
    
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate buy/sell signals based on Aroon indicator"""
        df['signal'] = 0  # 0: hold, 1: buy, -1: sell
        
        # Strategy parameters
        oscillator_threshold = self.parameters.get('oscillator_threshold', 50)
        momentum_threshold = self.parameters.get('momentum_threshold', 0.001)
        use_volume_confirmation = self.parameters.get('use_volume_confirmation', True)
        use_ma_confirmation = self.parameters.get('use_ma_confirmation', True)
        
        # Basic buy conditions
        buy_condition = (
            # Aroon bullish crossover
            df['aroon_bullish_crossover'] |
            # Strong uptrend (Aroon Up > 70, Aroon Down < 30)
            ((df['aroon_up'] > 70) & (df['aroon_down'] < 30)) |
            # Oscillator strongly positive
            (df['aroon_oscillator'] > oscillator_threshold)
        )
        
        # Basic sell conditions
        sell_condition = (
            # Aroon bearish crossover
            df['aroon_bearish_crossover'] |
            # Strong downtrend (Aroon Down > 70, Aroon Up < 30)
            ((df['aroon_down'] > 70) & (df['aroon_up'] < 30)) |
            # Oscillator strongly negative
            (df['aroon_oscillator'] < -oscillator_threshold)
        )
        
        # Apply volume confirmation if enabled
        if use_volume_confirmation:
            buy_condition = buy_condition & df['volume_increasing']
            sell_condition = sell_condition & df['volume_increasing']
        
        # Apply MA confirmation if enabled
        if use_ma_confirmation:
            buy_condition = buy_condition & df['ma_trend_up']
            sell_condition = sell_condition & ~df['ma_trend_up']
        
        # Apply momentum filter
        buy_condition = buy_condition & (df['price_momentum'] > -momentum_threshold)
        sell_condition = sell_condition & (df['price_momentum'] < momentum_threshold)
        
        # Advanced signals based on trend strength
        if self.parameters.get('use_trend_strength', True):
            # Strong trend entry
            strong_uptrend_entry = (
                (df['trend_strength'] == 'strong_up') &
                (df['aroon_up'] > 90) &
                df['volume_increasing'] &
                (df['price_momentum'] > 0)
            )
            
            strong_downtrend_entry = (
                (df['trend_strength'] == 'strong_down') &
                (df['aroon_down'] > 90) &
                df['volume_increasing'] &
                (df['price_momentum'] < 0)
            )
            
            buy_condition = buy_condition | strong_uptrend_entry
            sell_condition = sell_condition | strong_downtrend_entry
        
        # Consolidation breakout signals
        if self.parameters.get('use_consolidation_breakout', True):
            # Detect consolidation (both Aroon indicators low)
            consolidation = (df['aroon_up'] < 50) & (df['aroon_down'] < 50)
            
            # Breakout from consolidation
            consolidation_breakout_up = (
                consolidation.shift(1) &  # Was in consolidation
                (df['aroon_up'] > 70) &  # Now strong up
                df['volume_increasing']
            )
            
            consolidation_breakout_down = (
                consolidation.shift(1) &  # Was in consolidation
                (df['aroon_down'] > 70) &  # Now strong down
                df['volume_increasing']
            )
            
            buy_condition = buy_condition | consolidation_breakout_up
            sell_condition = sell_condition | consolidation_breakout_down
        
        # Apply signals
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        return df