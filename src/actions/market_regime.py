"""
Market Regime Detection Module
Identifies market conditions: trending_up, trending_down, sideways, volatile
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class MarketRegime(Enum):
    """Market regime types"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass
class RegimeIndicators:
    """Container for regime detection indicators"""
    adx: float
    adx_plus_di: float
    adx_minus_di: float
    ma_alignment_score: float
    volatility_ratio: float
    bb_width_pct: float
    volume_ratio: float
    choppiness_index: float
    returns_skewness: float
    returns_kurtosis: float
    trend_strength: float
    regime_probability: Dict[str, float]


class MarketRegimeDetector:
    """
    Detects market regime using multiple technical and statistical indicators
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize regime detector with configuration"""
        self.config = config or self._get_default_config()
        self.regime_history = []
        self.indicators_history = []
        
    def _get_default_config(self) -> Dict:
        """Default configuration for regime detection"""
        return {
            'adx_period': 14,
            'ma_periods': [10, 20, 50, 100],
            'bb_period': 20,
            'bb_std': 2,
            'atr_period': 14,
            'volume_period': 20,
            'choppiness_period': 14,
            'lookback_period': 100,
            
            # Thresholds
            'adx_trend_threshold': 25,
            'adx_strong_threshold': 40,
            'volatility_high_threshold': 2.0,
            'volatility_low_threshold': 0.5,
            'volume_spike_threshold': 2.0,
            'choppiness_sideways_threshold': 61.8,
            
            # Weights for ensemble
            'weights': {
                'adx': 0.25,
                'ma_alignment': 0.20,
                'volatility': 0.20,
                'volume': 0.10,
                'choppiness': 0.15,
                'statistical': 0.10
            }
        }
    
    def detect_regime(self, df: pd.DataFrame, 
                     return_history: bool = False) -> Tuple[MarketRegime, RegimeIndicators]:
        """
        Detect current market regime
        
        Args:
            df: DataFrame with OHLCV data
            return_history: Whether to return historical regime data
            
        Returns:
            Tuple of (MarketRegime, RegimeIndicators)
        """
        # Calculate all indicators
        indicators = self._calculate_indicators(df)
        
        # Determine regime based on indicators
        regime = self._determine_regime(indicators)
        
        # Store in history
        self.regime_history.append(regime)
        self.indicators_history.append(indicators)
        
        if return_history:
            return regime, indicators, self.regime_history
        
        return regime, indicators
    
    def _calculate_indicators(self, df: pd.DataFrame) -> RegimeIndicators:
        """Calculate all regime detection indicators"""
        # ADX and directional indicators
        adx_data = self._calculate_adx(df)
        
        # Moving average alignment
        ma_alignment = self._calculate_ma_alignment(df)
        
        # Volatility metrics
        volatility_metrics = self._calculate_volatility_metrics(df)
        
        # Volume analysis
        volume_ratio = self._calculate_volume_ratio(df)
        
        # Choppiness Index
        choppiness = self._calculate_choppiness_index(df)
        
        # Statistical properties
        stats = self._calculate_statistical_properties(df)
        
        # Trend strength
        trend_strength = self._calculate_trend_strength(df)
        
        # Calculate regime probabilities
        regime_probs = self._calculate_regime_probabilities(
            adx_data, ma_alignment, volatility_metrics, 
            volume_ratio, choppiness, stats
        )
        
        return RegimeIndicators(
            adx=adx_data['adx'],
            adx_plus_di=adx_data['plus_di'],
            adx_minus_di=adx_data['minus_di'],
            ma_alignment_score=ma_alignment,
            volatility_ratio=volatility_metrics['volatility_ratio'],
            bb_width_pct=volatility_metrics['bb_width_pct'],
            volume_ratio=volume_ratio,
            choppiness_index=choppiness,
            returns_skewness=stats['skewness'],
            returns_kurtosis=stats['kurtosis'],
            trend_strength=trend_strength,
            regime_probability=regime_probs
        )
    
    def _calculate_adx(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate ADX and directional indicators"""
        period = self.config['adx_period']
        
        # True Range
        high_low = df['high_price'] - df['low_price']
        high_close = abs(df['high_price'] - df['trade_price'].shift(1))
        low_close = abs(df['low_price'] - df['trade_price'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Directional Movement
        up_move = df['high_price'] - df['high_price'].shift(1)
        down_move = df['low_price'].shift(1) - df['low_price']
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed indicators
        # Avoid division by zero in DI calculations
        plus_dm_smooth = pd.Series(plus_dm).rolling(window=period).mean()
        minus_dm_smooth = pd.Series(minus_dm).rolling(window=period).mean()
        
        plus_di = pd.Series(np.where(atr > 0, 100 * plus_dm_smooth / atr, 0))
        minus_di = pd.Series(np.where(atr > 0, 100 * minus_dm_smooth / atr, 0))
        
        # ADX
        # Avoid division by zero
        di_sum = plus_di + minus_di
        dx = pd.Series(np.where(di_sum > 0, 100 * abs(plus_di - minus_di) / di_sum, 0))
        adx = dx.rolling(window=period).mean()
        
        return {
            'adx': adx.iloc[-1] if not adx.empty and not pd.isna(adx.iloc[-1]) else 0,
            'plus_di': plus_di.iloc[-1] if not plus_di.empty and not pd.isna(plus_di.iloc[-1]) else 0,
            'minus_di': minus_di.iloc[-1] if not minus_di.empty and not pd.isna(minus_di.iloc[-1]) else 0
        }
    
    def _calculate_ma_alignment(self, df: pd.DataFrame) -> float:
        """
        Calculate moving average alignment score
        1.0 = perfect uptrend alignment, -1.0 = perfect downtrend alignment
        """
        periods = self.config['ma_periods']
        mas = {}
        
        for period in periods:
            mas[period] = df['trade_price'].rolling(window=period).mean().iloc[-1]
        
        # Check alignment
        sorted_periods = sorted(periods)
        alignment_score = 0
        
        for i in range(len(sorted_periods) - 1):
            short_period = sorted_periods[i]
            long_period = sorted_periods[i + 1]
            
            if mas[short_period] > mas[long_period]:
                alignment_score += 1
            else:
                alignment_score -= 1
        
        # Normalize to [-1, 1]
        max_score = len(sorted_periods) - 1
        return alignment_score / max_score if max_score > 0 else 0
    
    def _calculate_volatility_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility-related metrics"""
        period = self.config['bb_period']
        std_mult = self.config['bb_std']
        
        # Bollinger Bands
        sma = df['trade_price'].rolling(window=period).mean()
        std = df['trade_price'].rolling(window=period).std()
        
        upper_band = sma + (std * std_mult)
        lower_band = sma - (std * std_mult)
        
        bb_width = upper_band - lower_band
        bb_width_pct = (bb_width / sma * 100).iloc[-1] if not sma.empty else 0
        
        # ATR-based volatility
        atr_period = self.config['atr_period']
        
        high_low = df['high_price'] - df['low_price']
        high_close = abs(df['high_price'] - df['trade_price'].shift(1))
        low_close = abs(df['low_price'] - df['trade_price'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=atr_period).mean()
        
        # Volatility ratio (current vs average)
        current_volatility = std.iloc[-1] if not std.empty else 0
        avg_volatility = std.rolling(window=50).mean().iloc[-1] if len(std) > 50 else current_volatility
        
        volatility_ratio = current_volatility / avg_volatility if avg_volatility > 0 else 1
        
        return {
            'bb_width_pct': bb_width_pct,
            'volatility_ratio': volatility_ratio,
            'atr': atr.iloc[-1] if not atr.empty else 0
        }
    
    def _calculate_volume_ratio(self, df: pd.DataFrame) -> float:
        """Calculate volume ratio (current vs average)"""
        period = self.config['volume_period']
        
        volume_ma = df['candle_acc_trade_volume'].rolling(window=period).mean()
        current_volume = df['candle_acc_trade_volume'].iloc[-1]
        avg_volume = volume_ma.iloc[-1] if not volume_ma.empty else current_volume
        
        return current_volume / avg_volume if avg_volume > 0 else 1
    
    def _calculate_choppiness_index(self, df: pd.DataFrame) -> float:
        """
        Calculate Choppiness Index
        Values: 0-100, where > 61.8 indicates sideways market
        """
        period = self.config['choppiness_period']
        
        # True Range sum
        high_low = df['high_price'] - df['low_price']
        high_close = abs(df['high_price'] - df['trade_price'].shift(1))
        low_close = abs(df['low_price'] - df['trade_price'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_sum = tr.rolling(window=period).sum()
        
        # High-Low range
        high_max = df['high_price'].rolling(window=period).max()
        low_min = df['low_price'].rolling(window=period).min()
        range_hl = high_max - low_min
        
        # Choppiness Index
        choppiness = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
        
        return choppiness.iloc[-1] if not choppiness.empty else 50
    
    def _calculate_statistical_properties(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate statistical properties of returns"""
        returns = df['trade_price'].pct_change().dropna()
        
        if len(returns) < 20:
            return {'skewness': 0, 'kurtosis': 0}
        
        # Use recent data
        recent_returns = returns.tail(100)
        
        return {
            'skewness': recent_returns.skew(),
            'kurtosis': recent_returns.kurtosis()
        }
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength using linear regression"""
        lookback = min(self.config['lookback_period'], len(df))
        
        if lookback < 10:
            return 0
        
        prices = df['trade_price'].tail(lookback).values
        x = np.arange(len(prices))
        
        # Linear regression
        coeffs = np.polyfit(x, prices, 1)
        slope = coeffs[0]
        
        # Normalize by price level
        avg_price = np.mean(prices)
        normalized_slope = slope / avg_price if avg_price > 0 else 0
        
        # R-squared for trend quality
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((prices - y_pred) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Combine slope and R-squared
        trend_strength = normalized_slope * r_squared
        
        return trend_strength
    
    def _calculate_regime_probabilities(self, adx_data: Dict, ma_alignment: float,
                                      volatility: Dict, volume_ratio: float,
                                      choppiness: float, stats: Dict) -> Dict[str, float]:
        """Calculate probability scores for each regime"""
        probs = {
            MarketRegime.TRENDING_UP.value: 0,
            MarketRegime.TRENDING_DOWN.value: 0,
            MarketRegime.SIDEWAYS.value: 0,
            MarketRegime.VOLATILE.value: 0
        }
        
        # ADX-based scoring
        adx_value = adx_data['adx']
        if adx_value > self.config['adx_strong_threshold']:
            if adx_data['plus_di'] > adx_data['minus_di']:
                probs[MarketRegime.TRENDING_UP.value] += 0.4
            else:
                probs[MarketRegime.TRENDING_DOWN.value] += 0.4
        elif adx_value < self.config['adx_trend_threshold']:
            probs[MarketRegime.SIDEWAYS.value] += 0.3
        
        # MA alignment scoring
        if ma_alignment > 0.5:
            probs[MarketRegime.TRENDING_UP.value] += 0.3
        elif ma_alignment < -0.5:
            probs[MarketRegime.TRENDING_DOWN.value] += 0.3
        else:
            probs[MarketRegime.SIDEWAYS.value] += 0.2
        
        # Volatility scoring
        if volatility['volatility_ratio'] > self.config['volatility_high_threshold']:
            probs[MarketRegime.VOLATILE.value] += 0.4
        elif volatility['volatility_ratio'] < self.config['volatility_low_threshold']:
            probs[MarketRegime.SIDEWAYS.value] += 0.2
        
        # Choppiness scoring
        if choppiness > self.config['choppiness_sideways_threshold']:
            probs[MarketRegime.SIDEWAYS.value] += 0.3
        else:
            # Trending market
            if ma_alignment > 0:
                probs[MarketRegime.TRENDING_UP.value] += 0.2
            else:
                probs[MarketRegime.TRENDING_DOWN.value] += 0.2
        
        # Volume scoring
        if volume_ratio > self.config['volume_spike_threshold']:
            probs[MarketRegime.VOLATILE.value] += 0.2
        
        # Normalize probabilities
        total = sum(probs.values())
        if total > 0:
            probs = {k: v/total for k, v in probs.items()}
        else:
            # Equal probabilities if no signals
            probs = {k: 0.25 for k in probs.keys()}
        
        return probs
    
    def _determine_regime(self, indicators: RegimeIndicators) -> MarketRegime:
        """Determine final regime based on indicators"""
        # Get regime with highest probability
        regime_probs = indicators.regime_probability
        max_regime = max(regime_probs, key=regime_probs.get)
        max_prob = regime_probs[max_regime]
        
        # Require minimum confidence
        if max_prob < 0.3:
            return MarketRegime.UNKNOWN
        
        # Additional validation rules
        if max_regime == MarketRegime.TRENDING_UP.value:
            # Validate uptrend
            if indicators.ma_alignment_score < 0 or indicators.adx < 20:
                # Check if it's volatile instead
                if indicators.volatility_ratio > 1.5:
                    return MarketRegime.VOLATILE
                return MarketRegime.SIDEWAYS
        
        elif max_regime == MarketRegime.TRENDING_DOWN.value:
            # Validate downtrend
            if indicators.ma_alignment_score > 0 or indicators.adx < 20:
                # Check if it's volatile instead
                if indicators.volatility_ratio > 1.5:
                    return MarketRegime.VOLATILE
                return MarketRegime.SIDEWAYS
        
        elif max_regime == MarketRegime.VOLATILE.value:
            # Validate high volatility
            if indicators.volatility_ratio < 1.2 and indicators.bb_width_pct < 3:
                # Not really volatile
                if indicators.choppiness_index > 61.8:
                    return MarketRegime.SIDEWAYS
                elif indicators.ma_alignment_score > 0.3:
                    return MarketRegime.TRENDING_UP
                elif indicators.ma_alignment_score < -0.3:
                    return MarketRegime.TRENDING_DOWN
        
        return MarketRegime(max_regime)
    
    def get_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for ML-based regime detection"""
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['returns'] = df['trade_price'].pct_change()
        features['log_returns'] = np.log(df['trade_price'] / df['trade_price'].shift(1))
        
        # Volatility features
        for window in [5, 10, 20]:
            features[f'volatility_{window}'] = features['returns'].rolling(window).std()
            features[f'realized_vol_{window}'] = np.sqrt(
                (features['log_returns'] ** 2).rolling(window).sum()
            )
        
        # Volume features
        features['volume_ratio'] = df['candle_acc_trade_volume'] / \
                                  df['candle_acc_trade_volume'].rolling(20).mean()
        features['volume_trend'] = df['candle_acc_trade_volume'].rolling(10).mean() / \
                                  df['candle_acc_trade_volume'].rolling(50).mean()
        
        # Microstructure features
        features['high_low_ratio'] = df['high_price'] / df['low_price']
        features['close_to_high'] = (df['trade_price'] - df['low_price']) / \
                                   (df['high_price'] - df['low_price'])
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(df['trade_price'], 14)
        
        # Autocorrelation features
        for lag in [1, 5, 10]:
            features[f'autocorr_{lag}'] = features['returns'].rolling(20).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else 0
            )
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi