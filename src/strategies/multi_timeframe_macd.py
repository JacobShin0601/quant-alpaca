"""
Multi-Timeframe MACD Strategy
Enhanced MACD with multiple timeframe analysis and convergence detection
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from .base import BaseStrategy


class MultiTimeframeMACDStrategy(BaseStrategy):
    """
    Advanced MACD strategy with multiple timeframe analysis
    Features:
    - Multiple timeframe MACD calculations
    - MACD convergence/divergence detection
    - Histogram momentum analysis
    - Cross-timeframe signal confirmation
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(parameters)
        
        # MACD configurations for different timeframes
        # Format: (fast_period, slow_period, signal_period)
        self.macd_configs = {
            'scalp': (3, 8, 5),      # Very short-term (scalping)
            'short': (5, 13, 8),     # Short-term
            'medium': (8, 21, 13),   # Medium-term  
            'standard': (12, 26, 9), # Standard MACD
            'long': (16, 34, 12)     # Long-term
        }
        
        # Timeframe weights for signal scoring
        self.timeframe_weights = {
            'scalp': 0.15,
            'short': 0.25,
            'medium': 0.25,
            'standard': 0.25,
            'long': 0.10
        }
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicators across multiple timeframes"""
        df = df.copy()
        
        # Calculate MACD for each timeframe
        for tf_name, (fast, slow, signal) in self.macd_configs.items():
            self._calculate_macd_for_timeframe(df, tf_name, fast, slow, signal)
        
        # Calculate convergence/divergence patterns
        self._calculate_macd_convergence(df)
        
        # Calculate momentum strength
        self._calculate_momentum_strength(df)
        
        # Calculate cross-timeframe signals
        self._calculate_cross_timeframe_signals(df)
        
        # Calculate overall MACD score
        self._calculate_macd_score(df)
        
        return df
    
    def _calculate_macd_for_timeframe(self, df: pd.DataFrame, tf_name: str, fast: int, slow: int, signal: int):
        """Calculate MACD components for a specific timeframe"""
        # EMA calculation
        ema_fast = df['trade_price'].ewm(span=fast, min_periods=fast//2).mean()
        ema_slow = df['trade_price'].ewm(span=slow, min_periods=slow//2).mean()
        
        # MACD line
        macd_line = ema_fast - ema_slow
        df[f'macd_line_{tf_name}'] = macd_line
        
        # Signal line
        signal_line = macd_line.ewm(span=signal, min_periods=signal//2).mean()
        df[f'macd_signal_{tf_name}'] = signal_line
        
        # Histogram
        histogram = macd_line - signal_line
        df[f'macd_histogram_{tf_name}'] = histogram
        
        # MACD crossover signals
        df[f'macd_cross_{tf_name}'] = np.where(macd_line > signal_line, 1, -1)
        
        # Zero line crossovers
        df[f'macd_zero_cross_{tf_name}'] = np.where(macd_line > 0, 1, -1)
        
        # Histogram momentum (increasing/decreasing)
        df[f'macd_momentum_{tf_name}'] = np.sign(histogram.diff())
        
        # MACD strength (normalized)
        price_std = df['trade_price'].rolling(window=slow*2, min_periods=slow).std()
        df[f'macd_strength_{tf_name}'] = abs(macd_line) / price_std.where(price_std > 0, 1)
    
    def _calculate_macd_convergence(self, df: pd.DataFrame):
        """Detect convergence/divergence patterns across timeframes"""
        # Get MACD lines for analysis
        macd_lines = {}
        for tf_name in self.macd_configs.keys():
            macd_lines[tf_name] = df[f'macd_line_{tf_name}']
        
        # Calculate convergence score
        convergence_scores = []
        timeframe_pairs = [('short', 'medium'), ('medium', 'standard'), ('standard', 'long')]
        
        for tf1, tf2 in timeframe_pairs:
            if tf1 in macd_lines and tf2 in macd_lines:
                # Convergence when both moving in same direction
                direction1 = np.sign(macd_lines[tf1].diff())
                direction2 = np.sign(macd_lines[tf2].diff())
                convergence = (direction1 == direction2).astype(int)
                convergence_scores.append(convergence)
        
        if convergence_scores:
            df['macd_convergence'] = sum(convergence_scores) / len(convergence_scores)
        else:
            df['macd_convergence'] = 0.5
        
        # Divergence detection (price vs MACD)
        self._detect_price_macd_divergence(df)
    
    def _detect_price_macd_divergence(self, df: pd.DataFrame):
        """Detect bullish/bearish divergences between price and MACD"""
        # Use standard MACD for divergence analysis
        if 'macd_line_standard' not in df.columns:
            df['bullish_divergence'] = 0
            df['bearish_divergence'] = 0
            return
        
        price = df['trade_price']
        macd = df['macd_line_standard']
        
        # Look for divergences in recent periods
        lookback = 20
        
        divergence_bull = []
        divergence_bear = []
        
        for i in range(len(df)):
            if i < lookback:
                divergence_bull.append(0)
                divergence_bear.append(0)
                continue
            
            # Get recent data
            recent_price = price.iloc[i-lookback:i+1]
            recent_macd = macd.iloc[i-lookback:i+1]
            
            # Find local extremes
            price_min_idx = recent_price.idxmin()
            price_max_idx = recent_price.idxmax()
            macd_min_idx = recent_macd.idxmin()
            macd_max_idx = recent_macd.idxmax()
            
            # Bullish divergence: price makes lower low, MACD makes higher low
            if (recent_price.iloc[-1] < recent_price.iloc[0] and 
                recent_macd.iloc[-1] > recent_macd.iloc[0]):
                divergence_bull.append(1)
            else:
                divergence_bull.append(0)
            
            # Bearish divergence: price makes higher high, MACD makes lower high
            if (recent_price.iloc[-1] > recent_price.iloc[0] and 
                recent_macd.iloc[-1] < recent_macd.iloc[0]):
                divergence_bear.append(1)
            else:
                divergence_bear.append(0)
        
        df['bullish_divergence'] = divergence_bull
        df['bearish_divergence'] = divergence_bear
    
    def _calculate_momentum_strength(self, df: pd.DataFrame):
        """Calculate momentum strength across timeframes"""
        momentum_scores = []
        
        for tf_name in self.macd_configs.keys():
            histogram_col = f'macd_histogram_{tf_name}'
            if histogram_col in df.columns:
                # Histogram momentum strength
                histogram = df[histogram_col]
                momentum = abs(histogram) / (abs(histogram).rolling(window=20, min_periods=5).mean() + 1e-8)
                momentum_scores.append(momentum * self.timeframe_weights[tf_name])
        
        if momentum_scores:
            df['macd_momentum_strength'] = sum(momentum_scores)
        else:
            df['macd_momentum_strength'] = 0
    
    def _calculate_cross_timeframe_signals(self, df: pd.DataFrame):
        """Calculate signals that require confirmation across timeframes"""
        # Collect crossover signals
        crossover_signals = []
        zero_cross_signals = []
        
        for tf_name in self.macd_configs.keys():
            cross_col = f'macd_cross_{tf_name}'
            zero_col = f'macd_zero_cross_{tf_name}'
            
            if cross_col in df.columns:
                crossover_signals.append(df[cross_col] * self.timeframe_weights[tf_name])
            if zero_col in df.columns:
                zero_cross_signals.append(df[zero_col] * self.timeframe_weights[tf_name])
        
        # Weighted average of signals
        if crossover_signals:
            df['macd_cross_score'] = sum(crossover_signals)
        else:
            df['macd_cross_score'] = 0
        
        if zero_cross_signals:
            df['macd_zero_score'] = sum(zero_cross_signals)
        else:
            df['macd_zero_score'] = 0
        
        # Strong signals when multiple timeframes agree
        df['macd_strong_bull'] = (
            (df['macd_cross_score'] > 0.5) & 
            (df['macd_zero_score'] > 0.3) & 
            (df['macd_convergence'] > 0.6)
        ).astype(int)
        
        df['macd_strong_bear'] = (
            (df['macd_cross_score'] < -0.5) & 
            (df['macd_zero_score'] < -0.3) & 
            (df['macd_convergence'] > 0.6)
        ).astype(int)
    
    def _calculate_macd_score(self, df: pd.DataFrame):
        """Calculate overall MACD signal score"""
        # Components
        cross_score = df.get('macd_cross_score', 0)
        zero_score = df.get('macd_zero_score', 0)
        momentum_strength = df.get('macd_momentum_strength', 0)
        convergence = df.get('macd_convergence', 0.5)
        
        # Divergence boost
        divergence_boost = (df.get('bullish_divergence', 0) * 0.3 - 
                           df.get('bearish_divergence', 0) * 0.3)
        
        # Combined score
        df['macd_total_score'] = (
            cross_score * 0.4 +
            zero_score * 0.3 +
            momentum_strength * 0.2 +
            convergence * 0.1 +
            divergence_boost
        )
    
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate multi-timeframe MACD signals"""
        df['signal'] = np.zeros(len(df), dtype=np.int8)
        
        # Parameters
        score_threshold = self.parameters.get('score_threshold', 0.3)
        require_convergence = self.parameters.get('require_convergence', True)
        use_divergence = self.parameters.get('use_divergence', True)
        min_momentum = self.parameters.get('min_momentum', 0.1)
        
        # Basic score-based signals
        strong_bull = df['macd_total_score'] > score_threshold
        strong_bear = df['macd_total_score'] < -score_threshold
        
        # Additional filters
        if require_convergence:
            convergence_filter = df['macd_convergence'] > 0.6
            strong_bull = strong_bull & convergence_filter
            strong_bear = strong_bear & convergence_filter
        
        if min_momentum > 0:
            momentum_filter = df['macd_momentum_strength'] > min_momentum
            strong_bull = strong_bull & momentum_filter
            strong_bear = strong_bear & momentum_filter
        
        # Divergence signals (if enabled)
        if use_divergence:
            divergence_bull = df['bullish_divergence'] == 1
            divergence_bear = df['bearish_divergence'] == 1
            
            # Combine with score-based signals
            strong_bull = strong_bull | divergence_bull
            strong_bear = strong_bear | divergence_bear
        
        # Apply signals
        df.loc[strong_bull, 'signal'] = 1
        df.loc[strong_bear, 'signal'] = -1
        
        # Signal smoothing to prevent whipsaws
        df['signal'] = self._smooth_signals(df['signal'])
        
        return df
    
    def _smooth_signals(self, signals: pd.Series, window: int = 3) -> pd.Series:
        """Smooth signals using rolling mode"""
        # Calculate rolling mode (most frequent value)
        smoothed = signals.rolling(window=window, center=True).apply(
            lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else 0
        )
        return smoothed.fillna(signals).astype(int)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy information"""
        return {
            'name': 'Multi-Timeframe MACD',
            'version': '1.0',
            'description': 'MACD strategy with multiple timeframe analysis and convergence detection',
            'macd_configs': self.macd_configs,
            'features': [
                'Multiple timeframe MACD analysis',
                'Convergence/divergence detection',
                'Cross-timeframe signal confirmation',
                'Histogram momentum analysis',
                'Price-MACD divergence detection'
            ],
            'parameters': self.parameters
        }


def main():
    """Example usage"""
    print("Multi-Timeframe MACD Strategy loaded successfully")


if __name__ == "__main__":
    main()