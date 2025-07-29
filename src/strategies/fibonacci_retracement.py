import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import BaseStrategy


class FibonacciRetracementStrategy(BaseStrategy):
    """Fibonacci Retracement strategy that identifies support/resistance levels"""
    
    def _should_buy(self, last_row, df) -> bool:
        """Buy when price bounces off Fibonacci support levels"""
        if pd.isna(last_row.get('near_fib_support', np.nan)):
            return False
        
        return (
            last_row['near_fib_support'] and 
            last_row.get('price_bouncing', False) and
            last_row.get('volume_confirmation', False)
        )
    
    def _should_sell(self, last_row, df) -> bool:
        """Sell when price rejects from Fibonacci resistance levels"""
        if pd.isna(last_row.get('near_fib_resistance', np.nan)):
            return False
        
        return (
            last_row['near_fib_resistance'] and 
            last_row.get('price_rejecting', False) and
            last_row.get('volume_confirmation', False)
        )
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Fibonacci levels and related indicators"""
        # Parameters
        swing_period = self.parameters.get('swing_period', 20)
        fib_proximity = self.parameters.get('fib_proximity', 0.003)  # 0.3% proximity
        
        # Fibonacci retracement levels
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        # Initialize columns
        for level in fib_levels:
            df[f'fib_{int(level*100)}'] = np.nan
        
        df['swing_high'] = np.nan
        df['swing_low'] = np.nan
        df['trend_direction'] = 0
        df['near_fib_support'] = False
        df['near_fib_resistance'] = False
        df['price_bouncing'] = False
        df['price_rejecting'] = False
        df['volume_confirmation'] = False
        
        # Calculate swing highs and lows
        for i in range(swing_period, len(df)):
            window = df.iloc[i-swing_period:i]
            
            # Identify swing high and low
            swing_high = window['high_price'].max()
            swing_low = window['low_price'].min()
            
            df.loc[i, 'swing_high'] = swing_high
            df.loc[i, 'swing_low'] = swing_low
            
            # Determine trend direction
            if df.loc[i, 'trade_price'] > window['trade_price'].mean():
                df.loc[i, 'trend_direction'] = 1  # Uptrend
            else:
                df.loc[i, 'trend_direction'] = -1  # Downtrend
            
            # Calculate Fibonacci levels
            price_range = swing_high - swing_low
            
            if price_range > 0:
                for level in fib_levels:
                    if df.loc[i, 'trend_direction'] == 1:  # Uptrend - calculate retracements from high
                        df.loc[i, f'fib_{int(level*100)}'] = swing_high - (price_range * level)
                    else:  # Downtrend - calculate retracements from low
                        df.loc[i, f'fib_{int(level*100)}'] = swing_low + (price_range * level)
                
                # Check proximity to Fibonacci levels
                current_price = df.loc[i, 'trade_price']
                
                for level in fib_levels:
                    fib_price = df.loc[i, f'fib_{int(level*100)}']
                    
                    if abs(current_price - fib_price) / fib_price < fib_proximity:
                        if df.loc[i, 'trend_direction'] == 1 and current_price > fib_price:
                            df.loc[i, 'near_fib_support'] = True
                        elif df.loc[i, 'trend_direction'] == -1 and current_price < fib_price:
                            df.loc[i, 'near_fib_resistance'] = True
                
                # Check for price action patterns
                if i >= 3:  # Need at least 3 candles for pattern
                    # Bouncing pattern (for support)
                    if (df.loc[i-2, 'trade_price'] > df.loc[i-1, 'trade_price'] and 
                        df.loc[i-1, 'trade_price'] < df.loc[i, 'trade_price']):
                        df.loc[i, 'price_bouncing'] = True
                    
                    # Rejecting pattern (for resistance) 
                    if (df.loc[i-2, 'trade_price'] < df.loc[i-1, 'trade_price'] and 
                        df.loc[i-1, 'trade_price'] > df.loc[i, 'trade_price']):
                        df.loc[i, 'price_rejecting'] = True
        
        # Volume confirmation
        volume_ma = df['candle_acc_trade_volume'].rolling(window=20).mean()
        df['volume_confirmation'] = df['candle_acc_trade_volume'] > volume_ma * 1.2
        
        # Additional indicators
        df['rsi'] = self._calculate_rsi(df['trade_price'])
        df['price_momentum'] = df['trade_price'].pct_change(periods=5)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate buy/sell signals based on Fibonacci retracements"""
        df['signal'] = 0  # 0: hold, 1: buy, -1: sell
        
        # Strategy parameters
        rsi_oversold = self.parameters.get('rsi_oversold', 30)
        rsi_overbought = self.parameters.get('rsi_overbought', 70)
        momentum_threshold = self.parameters.get('momentum_threshold', 0.001)
        
        # Buy conditions
        buy_condition = (
            df['near_fib_support'] &  # Price near Fibonacci support
            df['price_bouncing'] &  # Price showing bounce pattern
            df['volume_confirmation'] &  # Volume confirmation
            (df['rsi'] < 50) &  # RSI not overbought
            (df['price_momentum'] > -momentum_threshold)  # Momentum not too negative
        )
        
        # Enhanced buy on strong Fibonacci levels (38.2% and 61.8%)
        strong_fib_buy = (
            ((abs(df['trade_price'] - df['fib_38']) / df['fib_38'] < 0.003) |
             (abs(df['trade_price'] - df['fib_61']) / df['fib_61'] < 0.003)) &
            df['price_bouncing'] &
            df['volume_confirmation'] &
            (df['rsi'] < rsi_oversold)
        )
        
        # Sell conditions
        sell_condition = (
            df['near_fib_resistance'] &  # Price near Fibonacci resistance
            df['price_rejecting'] &  # Price showing rejection pattern
            df['volume_confirmation'] &  # Volume confirmation
            (df['rsi'] > 50) &  # RSI not oversold
            (df['price_momentum'] < momentum_threshold)  # Momentum not too positive
        )
        
        # Enhanced sell on strong Fibonacci levels
        strong_fib_sell = (
            ((abs(df['trade_price'] - df['fib_38']) / df['fib_38'] < 0.003) |
             (abs(df['trade_price'] - df['fib_61']) / df['fib_61'] < 0.003)) &
            df['price_rejecting'] &
            df['volume_confirmation'] &
            (df['rsi'] > rsi_overbought)
        )
        
        # Combine conditions
        buy_condition = buy_condition | strong_fib_buy
        sell_condition = sell_condition | strong_fib_sell
        
        # Additional filter: Golden ratio (61.8%) special handling
        if self.parameters.get('use_golden_ratio', True):
            golden_ratio_support = (
                (abs(df['trade_price'] - df['fib_61']) / df['fib_61'] < 0.002) &  # Very close to 61.8%
                (df['trend_direction'] == 1) &  # In uptrend
                df['volume_confirmation'] &
                (df['rsi'] < 40)
            )
            
            golden_ratio_resistance = (
                (abs(df['trade_price'] - df['fib_61']) / df['fib_61'] < 0.002) &  # Very close to 61.8%
                (df['trend_direction'] == -1) &  # In downtrend
                df['volume_confirmation'] &
                (df['rsi'] > 60)
            )
            
            buy_condition = buy_condition | golden_ratio_support
            sell_condition = sell_condition | golden_ratio_resistance
        
        # Apply signals
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        return df