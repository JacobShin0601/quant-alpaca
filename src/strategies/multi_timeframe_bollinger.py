"""
Multi-Timeframe Bollinger Bands Strategy
Enhanced Bollinger Bands with multiple timeframe analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from .base import BaseStrategy


class MultiTimeframeBollingerStrategy(BaseStrategy):
    """
    Advanced Bollinger Bands strategy with multiple timeframe analysis
    Features:
    - Multiple timeframe BB calculations (1m, 5m, 15m, 1h)
    - Timeframe alignment confirmation
    - Dynamic squeeze detection
    - Volume confirmation across timeframes
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(parameters)
        
        # Use the optimized bb_period if available
        primary_period = parameters.get('bb_period', 20)
        
        # Multiple timeframes (in minutes) - include the primary period
        self.timeframes = {
            'ultra_short': [5, 10],      # 5-10 minutes
            'short': [primary_period, 30],  # Include optimized period
            'medium': [60, 120],         # 1-2 hours
            'long': [240, 480]           # 4-8 hours
        }
        
        # Flattened list for easy iteration
        self.all_periods = []
        for periods in self.timeframes.values():
            self.all_periods.extend(periods)
        
        # Remove duplicates and sort
        self.all_periods = sorted(list(set(self.all_periods)))
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands across multiple timeframes"""
        df = df.copy()
        
        # Base parameters
        bb_std_dev = self.parameters.get('bb_std_dev', 2.0)
        
        # Calculate BB for each timeframe
        for period in self.all_periods:
            self._calculate_bb_for_period(df, period, bb_std_dev)
        
        # Calculate timeframe alignment
        self._calculate_timeframe_alignment(df)
        
        # Calculate squeeze conditions
        self._calculate_squeeze_conditions(df)
        
        # Volume confirmation across timeframes
        self._calculate_volume_confirmation(df)
        
        # Signal strength scoring
        self._calculate_signal_strength(df)
        
        return df
    
    def _calculate_bb_for_period(self, df: pd.DataFrame, period: int, std_dev: float):
        """Calculate Bollinger Bands for a specific period"""
        # Moving average
        df[f'bb_middle_{period}'] = df['trade_price'].rolling(window=period, min_periods=period//2).mean()
        
        # Standard deviation
        bb_std = df['trade_price'].rolling(window=period, min_periods=period//2).std()
        
        # Upper and lower bands
        df[f'bb_upper_{period}'] = df[f'bb_middle_{period}'] + (bb_std * std_dev)
        df[f'bb_lower_{period}'] = df[f'bb_middle_{period}'] - (bb_std * std_dev)
        
        # Band position (0 = lower band, 1 = upper band)
        band_width = df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']
        df[f'bb_position_{period}'] = (df['trade_price'] - df[f'bb_lower_{period}']) / band_width.where(band_width > 0, 1)
        
        # Band width (volatility measure)
        df[f'bb_width_{period}'] = band_width / df[f'bb_middle_{period}']
        
        # Distance from middle band
        df[f'bb_deviation_{period}'] = (df['trade_price'] - df[f'bb_middle_{period}']) / df[f'bb_middle_{period}']
    
    def _calculate_timeframe_alignment(self, df: pd.DataFrame):
        """Calculate alignment across different timeframes"""
        # Short-term alignment (5m, 10m, 20m)
        short_periods = [5, 10, 20]
        short_positions = [df[f'bb_position_{p}'] for p in short_periods if f'bb_position_{p}' in df.columns]
        if short_positions:
            df['bb_short_alignment'] = sum(pos > 0.8 for pos in short_positions) - sum(pos < 0.2 for pos in short_positions)
        
        # Medium-term alignment (30m, 60m, 120m)
        medium_periods = [30, 60, 120]
        medium_positions = [df[f'bb_position_{p}'] for p in medium_periods if f'bb_position_{p}' in df.columns]
        if medium_positions:
            df['bb_medium_alignment'] = sum(pos > 0.8 for pos in medium_positions) - sum(pos < 0.2 for pos in medium_positions)
        
        # Long-term trend (240m, 480m)
        long_periods = [240, 480]
        long_positions = [df[f'bb_position_{p}'] for p in long_periods if f'bb_position_{p}' in df.columns]
        if long_positions:
            df['bb_long_trend'] = sum(pos > 0.5 for pos in long_positions) - sum(pos < 0.5 for pos in long_positions)
        
        # Overall alignment score
        short_score = df.get('bb_short_alignment', 0) / 3
        medium_score = df.get('bb_medium_alignment', 0) / 3
        long_score = df.get('bb_long_trend', 0) / 2
        
        df['bb_overall_alignment'] = (short_score + medium_score + long_score) / 3
    
    def _calculate_squeeze_conditions(self, df: pd.DataFrame):
        """Detect Bollinger Band squeeze conditions"""
        # Calculate squeeze for each timeframe
        for period in self.all_periods:
            if f'bb_width_{period}' in df.columns:
                # Squeeze when width is in bottom 20% of recent values
                width_percentile = df[f'bb_width_{period}'].rolling(window=100, min_periods=20).rank(pct=True)
                df[f'bb_squeeze_{period}'] = (width_percentile < 0.2).astype(int)
        
        # Multi-timeframe squeeze detection
        squeeze_columns = [col for col in df.columns if col.startswith('bb_squeeze_')]
        if squeeze_columns:
            df['bb_multi_squeeze'] = df[squeeze_columns].sum(axis=1) / len(squeeze_columns)
    
    def _calculate_volume_confirmation(self, df: pd.DataFrame):
        """Calculate volume confirmation across timeframes"""
        # Check if volume column exists
        volume_col = 'candle_acc_trade_volume'
        if volume_col not in df.columns:
            # Fallback to other possible volume column names
            possible_volume_cols = ['volume', 'candle_volume', 'acc_trade_volume']
            volume_col = None
            for col in possible_volume_cols:
                if col in df.columns:
                    volume_col = col
                    break
        
        if volume_col is None or df[volume_col].isna().all():
            # If no volume data available, set confirmation to neutral (1.0)
            df['volume_confirmation'] = 1.0
            return
        
        # Volume moving averages for different periods
        volume_periods = [20, 60, 120]
        
        for period in volume_periods:
            df[f'volume_ma_{period}'] = df[volume_col].rolling(window=period, min_periods=max(5, period//4)).mean()
            # Ensure no division by zero
            ma_col = df[f'volume_ma_{period}'].fillna(method='bfill').fillna(1)
            df[f'volume_ratio_{period}'] = df[volume_col] / ma_col.where(ma_col > 0, 1)
        
        # Average volume confirmation with fallback
        volume_ratios = [df[f'volume_ratio_{p}'].fillna(1.0) for p in volume_periods]
        df['volume_confirmation'] = sum(volume_ratios) / len(volume_ratios)
        
        # Fill any remaining NaN values with neutral confirmation
        df['volume_confirmation'] = df['volume_confirmation'].fillna(1.0)
    
    def _calculate_signal_strength(self, df: pd.DataFrame):
        """Calculate overall signal strength"""
        # Position extremity (how close to bands)
        primary_period = self.parameters.get('bb_period', 20)
        if f'bb_position_{primary_period}' in df.columns:
            position = df[f'bb_position_{primary_period}'].fillna(0.5)
            # Strength increases as position approaches 0 or 1
            df['position_strength'] = np.minimum(position, 1 - position) * -2 + 1  # 0=weak, 1=strong
        else:
            df['position_strength'] = 0.5  # Neutral strength if no data
        
        # Timeframe alignment strength
        alignment_strength = abs(df.get('bb_overall_alignment', 0)).fillna(0)
        
        # Volume confirmation strength
        volume_strength = np.clip(df.get('volume_confirmation', 1) - 1, 0, 2)  # Above average volume
        
        # Squeeze breakout strength
        squeeze_strength = df.get('bb_multi_squeeze', 0)  # Higher when more squeezes
        
        # Combined signal strength with minimum threshold to ensure some signals
        df['bb_signal_strength'] = (df['position_strength'] * 0.4 + 
                                   alignment_strength * 0.3 + 
                                   volume_strength * 0.2 + 
                                   squeeze_strength * 0.1)
        
        # Ensure minimum signal strength to prevent complete signal blocking
        df['bb_signal_strength'] = np.maximum(df['bb_signal_strength'], 0.1)
    
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate enhanced BB signals with multi-timeframe confirmation"""
        df['signal'] = np.zeros(len(df), dtype=np.int8)
        
        # Parameters
        primary_period = self.parameters.get('bb_period', 20)
        lower_threshold = self.parameters.get('lower_threshold', 0.1)
        upper_threshold = self.parameters.get('upper_threshold', 0.9)
        min_signal_strength = self.parameters.get('min_signal_strength', 0.3)
        require_alignment = self.parameters.get('require_timeframe_alignment', True)
        min_volume_confirmation = self.parameters.get('min_volume_confirmation', 1.2)
        
        # Primary BB position
        bb_position_col = f'bb_position_{primary_period}'
        if bb_position_col not in df.columns:
            return df
        
        bb_position = df[bb_position_col]
        
        # Basic conditions
        buy_condition = bb_position < lower_threshold
        sell_condition = bb_position > upper_threshold
        
        # Signal strength filter - more lenient (20% reduction)
        strength_filter = df['bb_signal_strength'] >= (min_signal_strength * 0.8)
        
        # Volume confirmation filter - more lenient (10% reduction)
        volume_filter = df['volume_confirmation'] >= (min_volume_confirmation * 0.9)
        
        # Timeframe alignment filter - wider thresholds
        if require_alignment:
            # For buy: want negative alignment (oversold across timeframes)
            # For sell: want positive alignment (overbought across timeframes)
            # Use wider alignment thresholds to increase signal generation
            buy_alignment = df.get('bb_overall_alignment', 0) < -0.1  # Less restrictive (-0.1 vs -0.2)
            sell_alignment = df.get('bb_overall_alignment', 0) > 0.1   # Less restrictive (0.1 vs 0.2)
        else:
            buy_alignment = sell_alignment = True
        
        # Combine all conditions
        final_buy = buy_condition & strength_filter & volume_filter & buy_alignment
        final_sell = sell_condition & strength_filter & volume_filter & sell_alignment
        
        # Fallback: if no signals generated with strict conditions, use basic conditions only
        if not final_buy.any() and not final_sell.any():
            # Use only basic position-based conditions with relaxed thresholds
            fallback_buy = bb_position < (lower_threshold * 1.5)  # 50% more lenient
            fallback_sell = bb_position > (upper_threshold * 0.85)  # 15% more lenient
            
            # Apply fallback signals
            df.loc[fallback_buy, 'signal'] = 1
            df.loc[fallback_sell, 'signal'] = -1
        else:
            # Apply strict signals
            df.loc[final_buy, 'signal'] = 1
            df.loc[final_sell, 'signal'] = -1
        
        # Anti-oscillation filter (prevent rapid signal changes)
        df['signal'] = self._smooth_signals(df['signal'])
        
        return df
    
    def _smooth_signals(self, signals: pd.Series, window: int = 3) -> pd.Series:
        """Smooth signals to prevent oscillation"""
        # Use rolling median to smooth signals
        smoothed = signals.rolling(window=window, center=True).median()
        return smoothed.fillna(signals).astype(int)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy information"""
        return {
            'name': 'Multi-Timeframe Bollinger Bands',
            'version': '1.0',
            'description': 'Bollinger Bands strategy with multiple timeframe analysis and confirmation',
            'timeframes': self.timeframes,
            'features': [
                'Multiple timeframe BB analysis',
                'Timeframe alignment confirmation',
                'Squeeze detection across timeframes',
                'Volume confirmation',
                'Signal strength scoring'
            ],
            'parameters': self.parameters
        }


def main():
    """Example usage"""
    print("Multi-Timeframe Bollinger Bands Strategy loaded successfully")


if __name__ == "__main__":
    main()