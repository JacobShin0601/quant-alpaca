import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import BaseStrategy


class ATRBreakoutStrategy(BaseStrategy):
    """ATR (Average True Range) based breakout strategy for volatility trading"""
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR and breakout levels"""
        # ATR parameters
        atr_period = self.parameters.get('atr_period', 14)
        atr_multiplier = self.parameters.get('atr_multiplier', 2.0)
        lookback_period = self.parameters.get('lookback_period', 20)
        
        # Calculate True Range
        df['high_low'] = df['high_price'] - df['low_price']
        df['high_close'] = abs(df['high_price'] - df['trade_price'].shift(1))
        df['low_close'] = abs(df['low_price'] - df['trade_price'].shift(1))
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        
        # Calculate ATR
        df['atr'] = df['true_range'].rolling(window=atr_period).mean()
        
        # Calculate recent high and low
        df['recent_high'] = df['high_price'].rolling(window=lookback_period).max()
        df['recent_low'] = df['low_price'].rolling(window=lookback_period).min()
        
        # Calculate breakout levels
        df['upper_breakout'] = df['recent_high'] + (df['atr'] * atr_multiplier)
        df['lower_breakout'] = df['recent_low'] - (df['atr'] * atr_multiplier)
        
        # Alternative breakout calculation using closing price
        df['close_high'] = df['trade_price'].rolling(window=lookback_period).max()
        df['close_low'] = df['trade_price'].rolling(window=lookback_period).min()
        df['close_upper_breakout'] = df['close_high'] + (df['atr'] * atr_multiplier)
        df['close_lower_breakout'] = df['close_low'] - (df['atr'] * atr_multiplier)
        
        # Volatility metrics
        df['atr_pct'] = (df['atr'] / df['trade_price']) * 100  # ATR as percentage
        df['range_pct'] = ((df['recent_high'] - df['recent_low']) / df['trade_price']) * 100
        
        # Breakout detection
        df['upper_breakout_signal'] = df['trade_price'] > df['upper_breakout'].shift(1)
        df['lower_breakout_signal'] = df['trade_price'] < df['lower_breakout'].shift(1)
        
        # Trend filters - Always calculate as used in multiple variants
        ma_period = self.parameters.get('ma_period', 50)
        df['ma'] = df['trade_price'].rolling(window=ma_period).mean()
        df['trend_up'] = df['trade_price'] > df['ma']
        df['trend_down'] = df['trade_price'] < df['ma']
        
        # Volume analysis - Always calculate volume_ratio as it's used in multiple strategy variants
        volume_period = self.parameters.get('volume_period', 20)
        df['volume_ma'] = df['candle_acc_trade_volume'].rolling(window=volume_period).mean()
        df['volume_ratio'] = df['candle_acc_trade_volume'] / df['volume_ma']
        
        # Volume expansion during breakout
        df['volume_breakout'] = df['volume_ratio'] > self.parameters.get('volume_threshold', 1.5)
        
        # Momentum indicators - Always calculate as used in multiple variants
        momentum_period = self.parameters.get('momentum_period', 10)
        df['momentum'] = df['trade_price'].pct_change(momentum_period)
        df['momentum_positive'] = df['momentum'] > 0
        df['momentum_negative'] = df['momentum'] < 0
        
        # Clean up temporary columns
        df.drop(['high_low', 'high_close', 'low_close'], axis=1, inplace=True)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate ATR breakout signals"""
        df['signal'] = 0
        
        # Strategy parameters
        min_atr_pct = self.parameters.get('min_atr_pct', 0.5)  # Minimum 0.5% ATR
        max_atr_pct = self.parameters.get('max_atr_pct', 5.0)  # Maximum 5% ATR
        
        # Strategy variants
        strategy_variant = self.parameters.get('strategy_variant', 'classic')
        
        if strategy_variant == 'classic':
            # Classic ATR breakout
            # Buy on upper breakout
            buy_signal = (
                df['upper_breakout_signal'] &
                (df['atr_pct'] >= min_atr_pct) &
                (df['atr_pct'] <= max_atr_pct)
            )
            
            # Sell on lower breakout
            sell_signal = (
                df['lower_breakout_signal'] &
                (df['atr_pct'] >= min_atr_pct) &
                (df['atr_pct'] <= max_atr_pct)
            )
            
            # Apply trend filter if enabled
            if self.parameters.get('use_trend_filter', True):
                buy_signal = buy_signal & df['trend_up']
                sell_signal = sell_signal & df['trend_down']
            
            # Apply volume confirmation if enabled
            if self.parameters.get('use_volume_confirmation', True):
                buy_signal = buy_signal & df['volume_breakout']
                sell_signal = sell_signal & df['volume_breakout']
            
            df.loc[buy_signal, 'signal'] = 1
            df.loc[sell_signal, 'signal'] = -1
            
        elif strategy_variant == 'momentum':
            # Momentum-based ATR breakout
            # Buy on upper breakout with positive momentum
            buy_signal = (
                df['upper_breakout_signal'] &
                df['momentum_positive'] &
                (df['atr_pct'] >= min_atr_pct) &
                (df['volume_ratio'] > 1.0)
            )
            
            # Sell on lower breakout with negative momentum
            sell_signal = (
                df['lower_breakout_signal'] &
                df['momentum_negative'] &
                (df['atr_pct'] >= min_atr_pct) &
                (df['volume_ratio'] > 1.0)
            )
            
            df.loc[buy_signal, 'signal'] = 1
            df.loc[sell_signal, 'signal'] = -1
            
        elif strategy_variant == 'reversal':
            # Reversal strategy - fade the breakout
            # Sell on upper breakout (expecting reversal)
            reversal_sell = (
                df['upper_breakout_signal'] &
                (df['atr_pct'] > max_atr_pct * 0.8) &  # High volatility
                (df['range_pct'] > self.parameters.get('range_threshold', 5.0))  # Extended range
            )
            
            # Buy on lower breakout (expecting reversal)
            reversal_buy = (
                df['lower_breakout_signal'] &
                (df['atr_pct'] > max_atr_pct * 0.8) &  # High volatility
                (df['range_pct'] > self.parameters.get('range_threshold', 5.0))  # Extended range
            )
            
            df.loc[reversal_buy, 'signal'] = 1
            df.loc[reversal_sell, 'signal'] = -1
            
        else:  # 'adaptive' strategy
            # Adaptive strategy based on market conditions
            # Define market conditions
            low_volatility = df['atr_pct'] < min_atr_pct * 2
            normal_volatility = (df['atr_pct'] >= min_atr_pct * 2) & (df['atr_pct'] <= max_atr_pct * 0.7)
            high_volatility = df['atr_pct'] > max_atr_pct * 0.7
            
            # Low volatility: trade breakouts expecting expansion
            low_vol_buy = (
                low_volatility &
                df['upper_breakout_signal'] &
                (df['volume_ratio'] > 1.2)
            )
            
            low_vol_sell = (
                low_volatility &
                df['lower_breakout_signal'] &
                (df['volume_ratio'] > 1.2)
            )
            
            # Normal volatility: standard breakout trading
            normal_buy = (
                normal_volatility &
                df['upper_breakout_signal'] &
                df['trend_up'] &
                df['volume_breakout']
            )
            
            normal_sell = (
                normal_volatility &
                df['lower_breakout_signal'] &
                df['trend_down'] &
                df['volume_breakout']
            )
            
            # High volatility: mean reversion
            high_vol_buy = (
                high_volatility &
                df['lower_breakout_signal'] &  # Buy at extreme lows
                (df['momentum'] < -0.05)  # Strong negative momentum
            )
            
            high_vol_sell = (
                high_volatility &
                df['upper_breakout_signal'] &  # Sell at extreme highs
                (df['momentum'] > 0.05)  # Strong positive momentum
            )
            
            # Combine all signals
            buy_signals = low_vol_buy | normal_buy | high_vol_buy
            sell_signals = low_vol_sell | normal_sell | high_vol_sell
            
            df.loc[buy_signals, 'signal'] = 1
            df.loc[sell_signals, 'signal'] = -1
        
        # Position sizing based on ATR
        if self.parameters.get('use_atr_position_sizing', True):
            # Normalize position size inversely to volatility
            df['position_size'] = np.where(
                df['signal'] != 0,
                1.0 / (1.0 + df['atr_pct'] / 2.0),  # Lower size for higher volatility
                0
            )
        
        return df
    
    def _should_buy(self, last_row, df) -> bool:
        """Buy signal for real-time trading"""
        if pd.isna(last_row.get('upper_breakout_signal', False)):
            return False
        
        min_atr_pct = self.parameters.get('min_atr_pct', 0.5)
        max_atr_pct = self.parameters.get('max_atr_pct', 5.0)
        
        # Basic buy condition
        return (
            last_row.get('upper_breakout_signal', False) and
            last_row.get('atr_pct', 0) >= min_atr_pct and
            last_row.get('atr_pct', float('inf')) <= max_atr_pct and
            last_row.get('volume_ratio', 0) > 1.2
        )
    
    def _should_sell(self, last_row, df) -> bool:
        """Sell signal for real-time trading"""
        if pd.isna(last_row.get('lower_breakout_signal', False)):
            return False
        
        min_atr_pct = self.parameters.get('min_atr_pct', 0.5)
        max_atr_pct = self.parameters.get('max_atr_pct', 5.0)
        
        # Basic sell condition
        return (
            last_row.get('lower_breakout_signal', False) and
            last_row.get('atr_pct', 0) >= min_atr_pct and
            last_row.get('atr_pct', float('inf')) <= max_atr_pct and
            last_row.get('volume_ratio', 0) > 1.2
        )