import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from enum import Enum


class SignalStrength(Enum):
    VERY_WEAK = 0.2    # 20% of max position
    WEAK = 0.4         # 40% of max position
    NEUTRAL = 0.6      # 60% of max position
    STRONG = 0.8       # 80% of max position
    VERY_STRONG = 1.0  # 100% of max position


class SignalStrengthCalculator:
    """Calculate signal strength based on multiple technical indicators"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Default weights for different indicators
        self.indicator_weights = self.config.get('indicator_weights', {
            'rsi': 0.2,
            'macd': 0.2,
            'bollinger': 0.15,
            'volume': 0.15,
            'momentum': 0.15,
            'vwap': 0.15
        })
        
        # Thresholds for signal strength classification
        self.strength_thresholds = self.config.get('strength_thresholds', {
            'very_weak': 0.2,
            'weak': 0.4,
            'neutral': 0.6,
            'strong': 0.8,
            'very_strong': 1.0
        })
    
    def calculate_signal_strength(self, df: pd.DataFrame, signal_type: str) -> float:
        """
        Calculate signal strength based on multiple indicators
        
        Args:
            df: DataFrame with technical indicators
            signal_type: 'buy' or 'sell'
            
        Returns:
            Signal strength value between 0 and 1
        """
        if len(df) == 0:
            return 0.0
        
        current_row = df.iloc[-1]
        scores = {}
        
        # RSI Score
        if 'rsi' in df.columns and not pd.isna(current_row['rsi']):
            rsi_score = self._calculate_rsi_score(current_row['rsi'], signal_type)
            scores['rsi'] = rsi_score
        
        # MACD Score
        if all(col in df.columns for col in ['macd', 'macd_signal']):
            macd_score = self._calculate_macd_score(
                current_row.get('macd', 0),
                current_row.get('macd_signal', 0),
                signal_type
            )
            scores['macd'] = macd_score
        
        # Bollinger Bands Score
        if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'trade_price']):
            bb_score = self._calculate_bollinger_score(
                current_row['trade_price'],
                current_row.get('bb_upper', 0),
                current_row.get('bb_lower', 0),
                signal_type
            )
            scores['bollinger'] = bb_score
        
        # Volume Score
        if 'volume_ratio' in df.columns:
            volume_score = self._calculate_volume_score(current_row.get('volume_ratio', 1.0))
            scores['volume'] = volume_score
        
        # Momentum Score
        if 'momentum' in df.columns:
            momentum_score = self._calculate_momentum_score(
                current_row.get('momentum', 0),
                signal_type
            )
            scores['momentum'] = momentum_score
        
        # VWAP Score
        if all(col in df.columns for col in ['trade_price', 'vwap']):
            vwap_score = self._calculate_vwap_score(
                current_row['trade_price'],
                current_row.get('vwap', current_row['trade_price']),
                signal_type
            )
            scores['vwap'] = vwap_score
        
        # Calculate weighted average
        if not scores:
            return 0.5  # Default neutral strength
        
        total_weight = 0
        weighted_sum = 0
        
        for indicator, score in scores.items():
            weight = self.indicator_weights.get(indicator, 0.1)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight > 0:
            final_score = weighted_sum / total_weight
        else:
            final_score = 0.5
        
        # Clip to [0, 1] range
        return np.clip(final_score, 0.0, 1.0)
    
    def _calculate_rsi_score(self, rsi: float, signal_type: str) -> float:
        """Calculate RSI-based signal strength"""
        if signal_type == 'buy':
            if rsi < 20:
                return 1.0  # Very oversold
            elif rsi < 30:
                return 0.8  # Oversold
            elif rsi < 40:
                return 0.6  # Slightly oversold
            elif rsi < 60:
                return 0.4  # Neutral
            else:
                return 0.2  # Not good for buying
        else:  # sell
            if rsi > 80:
                return 1.0  # Very overbought
            elif rsi > 70:
                return 0.8  # Overbought
            elif rsi > 60:
                return 0.6  # Slightly overbought
            elif rsi > 40:
                return 0.4  # Neutral
            else:
                return 0.2  # Not good for selling
    
    def _calculate_macd_score(self, macd: float, signal: float, signal_type: str) -> float:
        """Calculate MACD-based signal strength"""
        histogram = macd - signal
        
        if signal_type == 'buy':
            if histogram > 0 and macd > 0:
                return min(1.0, abs(histogram) * 1000)  # Strong bullish
            elif histogram > 0:
                return 0.7  # Bullish crossover
            elif histogram > -0.0002:
                return 0.5  # Neutral
            else:
                return 0.3  # Bearish
        else:  # sell
            if histogram < 0 and macd < 0:
                return min(1.0, abs(histogram) * 1000)  # Strong bearish
            elif histogram < 0:
                return 0.7  # Bearish crossover
            elif histogram < 0.0002:
                return 0.5  # Neutral
            else:
                return 0.3  # Bullish
    
    def _calculate_bollinger_score(self, price: float, upper: float, lower: float, signal_type: str) -> float:
        """Calculate Bollinger Bands-based signal strength"""
        if upper == lower:
            return 0.5
        
        position = (price - lower) / (upper - lower)
        
        if signal_type == 'buy':
            if position < 0:
                return 1.0  # Below lower band
            elif position < 0.2:
                return 0.8  # Near lower band
            elif position < 0.5:
                return 0.6  # Lower half
            else:
                return 0.3  # Upper half
        else:  # sell
            if position > 1:
                return 1.0  # Above upper band
            elif position > 0.8:
                return 0.8  # Near upper band
            elif position > 0.5:
                return 0.6  # Upper half
            else:
                return 0.3  # Lower half
    
    def _calculate_volume_score(self, volume_ratio: float) -> float:
        """Calculate volume-based signal strength"""
        if volume_ratio > 2.0:
            return 1.0  # Very high volume
        elif volume_ratio > 1.5:
            return 0.8  # High volume
        elif volume_ratio > 1.0:
            return 0.6  # Above average
        elif volume_ratio > 0.8:
            return 0.4  # Below average
        else:
            return 0.2  # Low volume
    
    def _calculate_momentum_score(self, momentum: float, signal_type: str) -> float:
        """Calculate momentum-based signal strength"""
        if signal_type == 'buy':
            if momentum > 0.02:
                return 1.0  # Strong positive momentum
            elif momentum > 0.01:
                return 0.8
            elif momentum > 0:
                return 0.6
            elif momentum > -0.01:
                return 0.4
            else:
                return 0.2  # Strong negative momentum
        else:  # sell
            if momentum < -0.02:
                return 1.0  # Strong negative momentum
            elif momentum < -0.01:
                return 0.8
            elif momentum < 0:
                return 0.6
            elif momentum < 0.01:
                return 0.4
            else:
                return 0.2  # Strong positive momentum
    
    def _calculate_vwap_score(self, price: float, vwap: float, signal_type: str) -> float:
        """Calculate VWAP-based signal strength"""
        deviation = (price - vwap) / vwap
        
        if signal_type == 'buy':
            if deviation < -0.02:
                return 1.0  # Significantly below VWAP
            elif deviation < -0.01:
                return 0.8
            elif deviation < 0:
                return 0.6
            elif deviation < 0.01:
                return 0.4
            else:
                return 0.2  # Significantly above VWAP
        else:  # sell
            if deviation > 0.02:
                return 1.0  # Significantly above VWAP
            elif deviation > 0.01:
                return 0.8
            elif deviation > 0:
                return 0.6
            elif deviation > -0.01:
                return 0.4
            else:
                return 0.2  # Significantly below VWAP
    
    def get_position_size_multiplier(self, signal_strength: float) -> float:
        """
        Convert signal strength to position size multiplier
        
        Args:
            signal_strength: Signal strength value (0-1)
            
        Returns:
            Position size multiplier (0.2-1.0)
        """
        if signal_strength >= self.strength_thresholds['very_strong']:
            return SignalStrength.VERY_STRONG.value
        elif signal_strength >= self.strength_thresholds['strong']:
            return SignalStrength.STRONG.value
        elif signal_strength >= self.strength_thresholds['neutral']:
            return SignalStrength.NEUTRAL.value
        elif signal_strength >= self.strength_thresholds['weak']:
            return SignalStrength.WEAK.value
        else:
            return SignalStrength.VERY_WEAK.value