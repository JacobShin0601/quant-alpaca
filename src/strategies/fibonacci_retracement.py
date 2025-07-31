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
        """Calculate Fibonacci levels and related indicators - optimized version"""
        # Parameters
        swing_period = self.parameters.get('swing_period', 20)
        fib_proximity = self.parameters.get('fib_proximity', 0.003)  # 0.3% proximity
        
        # Calculate rolling swing highs and lows
        df['swing_high'] = df['high_price'].rolling(window=swing_period).max()
        df['swing_low'] = df['low_price'].rolling(window=swing_period).min()
        
        # Calculate rolling mean for trend detection
        df['price_ma'] = df['trade_price'].rolling(window=swing_period).mean()
        df['trend_direction'] = np.where(df['trade_price'] > df['price_ma'], 1, -1)
        
        # Calculate price range
        df['price_range'] = df['swing_high'] - df['swing_low']
        
        # Fibonacci levels - using proper names
        fib_levels = {'fib_236': 0.236, 'fib_382': 0.382, 'fib_500': 0.5, 'fib_618': 0.618, 'fib_786': 0.786}
        
        for name, level in fib_levels.items():
            # In uptrend: retracements from high
            uptrend_fib = df['swing_high'] - (df['price_range'] * level)
            # In downtrend: retracements from low  
            downtrend_fib = df['swing_low'] + (df['price_range'] * level)
            # Select based on trend
            df[name] = np.where(df['trend_direction'] == 1, uptrend_fib, downtrend_fib)
        
        # Check proximity to any Fibonacci level
        df['near_fib_support'] = False
        df['near_fib_resistance'] = False
        
        for name in fib_levels.keys():
            # Support: price above fib level in uptrend
            support_condition = (
                (df['trend_direction'] == 1) &
                (df['trade_price'] > df[name]) &
                (abs(df['trade_price'] - df[name]) / df[name] < fib_proximity)
            )
            df['near_fib_support'] = df['near_fib_support'] | support_condition
            
            # Resistance: price below fib level in downtrend
            resistance_condition = (
                (df['trend_direction'] == -1) &
                (df['trade_price'] < df[name]) &
                (abs(df['trade_price'] - df[name]) / df[name] < fib_proximity)
            )
            df['near_fib_resistance'] = df['near_fib_resistance'] | resistance_condition
        
        # Price action patterns using vectorized operations
        price_prev1 = df['trade_price'].shift(1)
        price_prev2 = df['trade_price'].shift(2)
        
        # Bouncing pattern (V-shape)
        df['price_bouncing'] = (
            (price_prev2 > price_prev1) &
            (price_prev1 < df['trade_price'])
        )
        
        # Rejecting pattern (inverted V-shape)
        df['price_rejecting'] = (
            (price_prev2 < price_prev1) &
            (price_prev1 > df['trade_price'])
        )
        
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
            ((abs(df['trade_price'] - df['fib_382']) / df['fib_382'] < 0.003) |
             (abs(df['trade_price'] - df['fib_618']) / df['fib_618'] < 0.003)) &
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
            ((abs(df['trade_price'] - df['fib_382']) / df['fib_382'] < 0.003) |
             (abs(df['trade_price'] - df['fib_618']) / df['fib_618'] < 0.003)) &
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
                (abs(df['trade_price'] - df['fib_618']) / df['fib_618'] < 0.002) &  # Very close to 61.8%
                (df['trend_direction'] == 1) &  # In uptrend
                df['volume_confirmation'] &
                (df['rsi'] < 40)
            )
            
            golden_ratio_resistance = (
                (abs(df['trade_price'] - df['fib_618']) / df['fib_618'] < 0.002) &  # Very close to 61.8%
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