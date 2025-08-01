"""
High-Frequency VWAP Strategy optimized for 1-minute cryptocurrency trading
Incorporates microstructure features, dynamic parameters, and advanced filtering
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base import BaseStrategy
import warnings
warnings.filterwarnings('ignore')


class HighFrequencyVWAPStrategy(BaseStrategy):
    """
    Advanced VWAP strategy optimized for high-frequency 1-minute trading
    Features:
    - Multi-timeframe VWAP analysis
    - Dynamic parameter adjustment based on market regimes
    - Microstructure-aware filtering
    - Volume flow analysis
    - Liquidity-based execution timing
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(parameters)
        
        # High-frequency specific parameters
        self.ultra_short_periods = [1, 3, 5, 10]  # 1-10 minutes
        self.short_periods = [15, 30, 60]         # 15-60 minutes  
        self.medium_periods = [120, 240, 480]     # 2-8 hours
        
        # Microstructure parameters
        self.min_spread_threshold = self.parameters.get('min_spread_threshold', 0.001)
        self.min_volume_ratio = self.parameters.get('min_volume_ratio', 1.2)
        self.liquidity_filter = self.parameters.get('use_liquidity_filter', True)
        
        # Dynamic regime detection
        self.regime_detection = self.parameters.get('use_regime_detection', True)
        self.volatility_lookback = self.parameters.get('volatility_lookback', 60)
        
        # Order flow parameters
        self.order_flow_period = self.parameters.get('order_flow_period', 10)
        self.flow_threshold = self.parameters.get('flow_threshold', 0.6)
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive VWAP indicators for high-frequency trading"""
        df = df.copy()
        
        # Basic price and volume preprocessing
        df['returns'] = np.log(df['trade_price'] / df['trade_price'].shift(1))
        df['typical_price'] = (df['high_price'] + df['low_price'] + df['trade_price']) / 3
        df['volume_price'] = df['typical_price'] * df['candle_acc_trade_volume']
        
        # Multi-timeframe VWAP calculation
        self._calculate_multi_timeframe_vwap(df)
        
        # Microstructure features
        self._calculate_microstructure_features(df)
        
        # Volume flow analysis
        self._calculate_volume_flow(df)
        
        # Market regime detection
        if self.regime_detection:
            self._detect_market_regime(df)
        
        # Liquidity scoring
        if self.liquidity_filter:
            self._calculate_liquidity_score(df)
        else:
            # Set default liquidity values when filter is disabled
            df['liquidity_score'] = 1.0
            df['liquidity_percentile'] = 0.5
        
        # Signal strength calculation
        self._calculate_signal_strength(df)
        
        return df
    
    def _calculate_multi_timeframe_vwap(self, df: pd.DataFrame):
        """Calculate VWAP across multiple timeframes"""
        # Ultra-short term VWAPs (1-10 minutes)
        for period in self.ultra_short_periods:
            pv_sum = df['volume_price'].rolling(window=period, min_periods=1).sum()
            vol_sum = df['candle_acc_trade_volume'].rolling(window=period, min_periods=1).sum()
            df[f'vwap_{period}m'] = pv_sum / vol_sum.where(vol_sum > 0, 1)
            df[f'vwap_dev_{period}m'] = (df['trade_price'] - df[f'vwap_{period}m']) / df[f'vwap_{period}m']
        
        # Short-term VWAPs (15-60 minutes)  
        for period in self.short_periods:
            pv_sum = df['volume_price'].rolling(window=period, min_periods=1).sum()
            vol_sum = df['candle_acc_trade_volume'].rolling(window=period, min_periods=1).sum()
            df[f'vwap_{period}m'] = pv_sum / vol_sum.where(vol_sum > 0, 1)
            df[f'vwap_dev_{period}m'] = (df['trade_price'] - df[f'vwap_{period}m']) / df[f'vwap_{period}m']
        
        # Medium-term VWAPs (2-8 hours)
        for period in self.medium_periods:
            pv_sum = df['volume_price'].rolling(window=period, min_periods=1).sum()
            vol_sum = df['candle_acc_trade_volume'].rolling(window=period, min_periods=1).sum()
            df[f'vwap_{period}m'] = pv_sum / vol_sum.where(vol_sum > 0, 1)
            df[f'vwap_dev_{period}m'] = (df['trade_price'] - df[f'vwap_{period}m']) / df[f'vwap_{period}m']
        
        # VWAP trend alignment score
        vwap_deviations = [df[f'vwap_dev_{p}m'] for p in [5, 15, 60, 240]]
        df['vwap_alignment'] = sum(np.sign(deviation) for deviation in vwap_deviations) / len(vwap_deviations)
    
    def _calculate_microstructure_features(self, df: pd.DataFrame):
        """Calculate market microstructure features"""
        # Bid-ask spread approximation
        df['spread_est'] = (df['high_price'] - df['low_price']) / df['trade_price']
        df['spread_ma'] = df['spread_est'].rolling(window=20).mean()
        df['spread_normalized'] = df['spread_est'] / df['spread_ma'].where(df['spread_ma'] > 0, 1)
        
        # Volume normalization (24-hour basis)
        df['volume_ma_1d'] = df['candle_acc_trade_volume'].rolling(window=1440, min_periods=60).mean()
        df['volume_normalized'] = df['candle_acc_trade_volume'] / df['volume_ma_1d'].where(df['volume_ma_1d'] > 0, 1)
        
        # Price momentum across ultra-short timeframes
        for period in [1, 3, 5]:
            df[f'momentum_{period}m'] = (df['trade_price'] / df['trade_price'].shift(period) - 1) * 100
        
        # Realized volatility (1-hour rolling)
        df['realized_vol_1h'] = df['returns'].rolling(window=60).std() * np.sqrt(1440)
        df['vol_regime'] = np.where(df['realized_vol_1h'] > df['realized_vol_1h'].rolling(window=240).mean() * 1.5, 2,
                                   np.where(df['realized_vol_1h'] < df['realized_vol_1h'].rolling(window=240).mean() * 0.7, 0, 1))
    
    def _calculate_volume_flow(self, df: pd.DataFrame):
        """Calculate volume flow and buying/selling pressure"""
        # Approximate order flow
        df['price_change'] = df['trade_price'].diff()
        df['buy_volume'] = np.where(df['price_change'] > 0, df['candle_acc_trade_volume'], 0)
        df['sell_volume'] = np.where(df['price_change'] < 0, df['candle_acc_trade_volume'], 0)
        df['neutral_volume'] = np.where(df['price_change'] == 0, df['candle_acc_trade_volume'], 0)
        
        # Rolling volume flow ratios
        for period in [5, 10, 20]:
            buy_sum = df['buy_volume'].rolling(window=period).sum()
            sell_sum = df['sell_volume'].rolling(window=period).sum()
            total_volume = buy_sum + sell_sum
            
            df[f'buy_ratio_{period}m'] = buy_sum / total_volume.where(total_volume > 0, 1)
            df[f'net_flow_{period}m'] = (buy_sum - sell_sum) / total_volume.where(total_volume > 0, 1)
        
        # Volume acceleration
        df['volume_acceleration'] = df['candle_acc_trade_volume'].diff(2)
        df['volume_momentum'] = df['candle_acc_trade_volume'].pct_change(5)
    
    def _detect_market_regime(self, df: pd.DataFrame):
        """Detect market regime for dynamic parameter adjustment"""
        # Volatility-based regime
        vol_short = df['realized_vol_1h']
        vol_long = vol_short.rolling(window=240, min_periods=60).mean()
        
        df['vol_regime'] = np.where(vol_short > vol_long * 1.5, 2,  # High vol
                                   np.where(vol_short < vol_long * 0.7, 0, 1))  # Low/Normal vol
        
        # Trend regime using multiple VWAP alignment
        df['trend_regime'] = np.where(df['vwap_alignment'] > 0.5, 1,    # Uptrend
                                     np.where(df['vwap_alignment'] < -0.5, -1, 0))  # Downtrend/Sideways
        
        # Combined regime
        df['market_regime'] = df['vol_regime'] * 3 + df['trend_regime']
    
    def _calculate_liquidity_score(self, df: pd.DataFrame):
        """Calculate liquidity score for execution timing optimization"""
        # Volume-based liquidity
        volume_liquidity = df['volume_normalized']
        
        # Spread-based liquidity (inverse relationship)
        spread_liquidity = 1 / (df['spread_normalized'] + 0.1)
        
        # Volatility adjustment (higher vol = lower liquidity)
        vol_adjustment = 1 / (df['realized_vol_1h'] * 10 + 0.1)
        
        # Combined liquidity score (higher is better)
        df['liquidity_score'] = (volume_liquidity * spread_liquidity * vol_adjustment) ** (1/3)
        df['liquidity_percentile'] = df['liquidity_score'].rolling(window=1440, min_periods=60).rank(pct=True)
    
    def _calculate_signal_strength(self, df: pd.DataFrame):
        """Calculate signal strength based on multiple factors"""
        # VWAP deviation strength
        primary_vwap_period = self.parameters.get('vwap_period', 30)
        
        # Find closest available period
        all_periods = self.ultra_short_periods + self.short_periods + self.medium_periods
        closest_period = min(all_periods, key=lambda x: abs(x - primary_vwap_period))
        vwap_dev_col = f'vwap_dev_{closest_period}m'
        
        if vwap_dev_col in df.columns:
            df['vwap_signal_strength'] = abs(df[vwap_dev_col])
        else:
            df['vwap_signal_strength'] = 0
        
        # Volume confirmation strength
        df['volume_signal_strength'] = df['volume_normalized'] * df.get('buy_ratio_10m', 0.5)
        
        # Multi-timeframe confirmation
        df['timeframe_confirmation'] = abs(df['vwap_alignment'])
        
        # Combined signal strength
        df['signal_strength'] = (df['vwap_signal_strength'] * 
                               df['volume_signal_strength'] * 
                               df['timeframe_confirmation'])
    
    def _should_buy(self, last_row, df) -> bool:
        """Enhanced buy condition with high-frequency considerations"""
        # Get dynamic parameters based on regime
        vwap_threshold = self._get_dynamic_threshold(last_row, 'buy')
        volume_threshold = self.parameters.get('volume_threshold', 1.2)
        
        # Primary VWAP signal
        primary_period = self.parameters.get('vwap_period', 30)
        vwap_dev = last_row.get(f'vwap_dev_{primary_period}m', 0)
        
        # Basic conditions
        basic_signal = (vwap_dev < -vwap_threshold and 
                       last_row.get('volume_normalized', 0) > volume_threshold)
        
        if not basic_signal:
            return False
        
        # High-frequency filters
        if not self._passes_hf_filters(last_row, 'buy'):
            return False
        
        # Multi-timeframe confirmation
        if not self._check_timeframe_alignment(last_row, 'buy'):
            return False
        
        # Volume flow confirmation
        if not self._check_volume_flow(last_row, 'buy'):
            return False
        
        return True
    
    def _should_sell(self, last_row, df) -> bool:
        """Enhanced sell condition with high-frequency considerations"""
        # Get dynamic parameters based on regime
        vwap_threshold = self._get_dynamic_threshold(last_row, 'sell')
        volume_threshold = self.parameters.get('volume_threshold', 1.2)
        
        # Primary VWAP signal
        primary_period = self.parameters.get('vwap_period', 30)
        vwap_dev = last_row.get(f'vwap_dev_{primary_period}m', 0)
        
        # Basic conditions
        basic_signal = (vwap_dev > vwap_threshold and 
                       last_row.get('volume_normalized', 0) > volume_threshold)
        
        if not basic_signal:
            return False
        
        # High-frequency filters
        if not self._passes_hf_filters(last_row, 'sell'):
            return False
        
        # Multi-timeframe confirmation
        if not self._check_timeframe_alignment(last_row, 'sell'):
            return False
        
        # Volume flow confirmation
        if not self._check_volume_flow(last_row, 'sell'):
            return False
        
        return True
    
    def _get_dynamic_threshold(self, last_row, signal_type: str) -> float:
        """Get dynamic VWAP threshold based on market regime"""
        base_threshold = self.parameters.get('vwap_threshold', 0.002)
        
        # Adjust based on volatility regime
        vol_regime = last_row.get('vol_regime', 1)
        if vol_regime == 2:  # High volatility
            threshold_multiplier = 1.5
        elif vol_regime == 0:  # Low volatility
            threshold_multiplier = 0.7
        else:  # Normal volatility
            threshold_multiplier = 1.0
        
        # Adjust based on liquidity
        liquidity_percentile = last_row.get('liquidity_percentile', 0.5)
        if liquidity_percentile < 0.2:  # Low liquidity
            threshold_multiplier *= 1.3
        elif liquidity_percentile > 0.8:  # High liquidity
            threshold_multiplier *= 0.8
        
        return base_threshold * threshold_multiplier
    
    def _passes_hf_filters(self, last_row, signal_type: str) -> bool:
        """Check high-frequency specific filters"""
        # Spread filter
        spread_normalized = last_row.get('spread_normalized', 1.0)
        if spread_normalized > 2.0:  # Spread too wide
            return False
        
        # Liquidity filter
        if self.liquidity_filter:
            liquidity_percentile = last_row.get('liquidity_percentile', 0.5)
            if liquidity_percentile < 0.1:  # Too illiquid
                return False
        
        # Volume acceleration filter
        volume_acceleration = last_row.get('volume_acceleration', 0)
        if signal_type == 'buy' and volume_acceleration < -last_row.get('candle_acc_trade_volume', 0) * 0.5:
            return False  # Volume declining too fast
        
        return True
    
    def _check_timeframe_alignment(self, last_row, signal_type: str) -> bool:
        """Check multi-timeframe VWAP alignment"""
        alignment_required = self.parameters.get('require_timeframe_alignment', False)
        if not alignment_required:
            return True
        
        # Check alignment across different timeframes
        short_vwap_dev = last_row.get('vwap_dev_5m', 0)
        medium_vwap_dev = last_row.get('vwap_dev_30m', 0)
        long_vwap_dev = last_row.get('vwap_dev_240m', 0)
        
        if signal_type == 'buy':
            return (short_vwap_dev < 0 and medium_vwap_dev < 0)
        else:  # sell
            return (short_vwap_dev > 0 and medium_vwap_dev > 0)
    
    def _check_volume_flow(self, last_row, signal_type: str) -> bool:
        """Check volume flow confirmation"""
        flow_confirmation = self.parameters.get('require_flow_confirmation', True)
        if not flow_confirmation:
            return True
        
        # Check order flow direction
        buy_ratio_5m = last_row.get('buy_ratio_5m', 0.5)
        net_flow_10m = last_row.get('net_flow_10m', 0)
        
        if signal_type == 'buy':
            return (buy_ratio_5m > self.flow_threshold and net_flow_10m > 0)
        else:  # sell
            return (buy_ratio_5m < (1 - self.flow_threshold) and net_flow_10m < 0)
    
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate high-frequency optimized trading signals - vectorized version"""
        df['signal'] = 0
        
        # Only generate signals after sufficient warm-up period
        warmup_period = max(self.medium_periods) + 50
        
        # Get parameters
        vwap_threshold = self.parameters.get('vwap_threshold', 0.002)
        volume_threshold = self.parameters.get('volume_threshold', 1.2)
        primary_period = self.parameters.get('vwap_period', 30)
        
        # Primary VWAP deviation column - find closest available period
        all_periods = self.ultra_short_periods + self.short_periods + self.medium_periods
        closest_period = min(all_periods, key=lambda x: abs(x - primary_period))
        vwap_dev_col = f'vwap_dev_{closest_period}m'
        
        # Basic buy signals (vectorized)
        if vwap_dev_col in df.columns:
            buy_conditions = (
                (df.index >= df.index[warmup_period]) &  # After warmup
                (df[vwap_dev_col] < -vwap_threshold) &  # Below VWAP
                (df['volume_normalized'] > volume_threshold) &  # High volume
                (df['spread_normalized'] <= 2.0)  # Reasonable spread
            )
            
            # Basic sell signals (vectorized)
            sell_conditions = (
                (df.index >= df.index[warmup_period]) &  # After warmup
                (df[vwap_dev_col] > vwap_threshold) &  # Above VWAP
                (df['volume_normalized'] > volume_threshold) &  # High volume
                (df['spread_normalized'] <= 2.0)  # Reasonable spread
            )
        else:
            # Fallback if column doesn't exist
            buy_conditions = pd.Series(False, index=df.index)
            sell_conditions = pd.Series(False, index=df.index)
        
        # Additional filters if enabled
        if self.liquidity_filter and 'liquidity_percentile' in df.columns:
            buy_conditions = buy_conditions & (df['liquidity_percentile'] >= 0.1)
            sell_conditions = sell_conditions & (df['liquidity_percentile'] >= 0.1)
        
        # Volume flow confirmation if available
        if 'buy_ratio_5m' in df.columns and self.parameters.get('require_flow_confirmation', True):
            flow_threshold = self.parameters.get('flow_threshold', 0.6)
            buy_conditions = buy_conditions & (df['buy_ratio_5m'] > flow_threshold)
            sell_conditions = sell_conditions & (df['buy_ratio_5m'] < (1 - flow_threshold))
        
        # Multi-timeframe alignment if enabled
        if self.parameters.get('require_timeframe_alignment', True):
            # At least 2 out of 3 timeframes should show the same direction
            buy_alignment = (
                (df.get('vwap_dev_5m', 0) < 0).astype(int) +
                (df.get('vwap_dev_15m', 0) < 0).astype(int) +
                (df.get('vwap_dev_30m', 0) < 0).astype(int)
            ) >= 2
            
            sell_alignment = (
                (df.get('vwap_dev_5m', 0) > 0).astype(int) +
                (df.get('vwap_dev_15m', 0) > 0).astype(int) +
                (df.get('vwap_dev_30m', 0) > 0).astype(int)
            ) >= 2
            
            buy_conditions = buy_conditions & buy_alignment
            sell_conditions = sell_conditions & sell_alignment
        
        # Apply signals
        df.loc[buy_conditions, 'signal'] = 1
        df.loc[sell_conditions, 'signal'] = -1
        
        # Signal quality filtering
        df = self._filter_signals_by_quality(df)
        
        return df
    
    def _filter_signals_by_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter signals based on quality metrics"""
        min_signal_strength = self.parameters.get('min_signal_strength', 0.1)
        min_liquidity_percentile = self.parameters.get('min_liquidity_percentile', 0.2)
        
        # Filter out low-quality signals
        quality_mask = (df['signal_strength'] >= min_signal_strength)
        
        # Only apply liquidity filter if liquidity data is available
        if 'liquidity_percentile' in df.columns:
            quality_mask = quality_mask & (df['liquidity_percentile'] >= min_liquidity_percentile)
        
        df.loc[~quality_mask, 'signal'] = 0
        
        # Anti-oscillation filter (prevent rapid signal changes)
        df['signal_smoothed'] = df['signal'].rolling(window=3, center=True).median()
        df['signal'] = df['signal_smoothed'].fillna(0).astype(int)
        
        return df
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy information"""
        return {
            'name': 'High-Frequency VWAP',
            'version': '2.0',
            'description': 'Advanced VWAP strategy optimized for 1-minute cryptocurrency trading',
            'timeframe': '1m',
            'features': [
                'Multi-timeframe VWAP analysis',
                'Dynamic parameter adjustment',
                'Microstructure filtering',
                'Volume flow analysis',
                'Liquidity-based execution'
            ],
            'parameters': self.parameters
        }


class AdaptiveHighFrequencyVWAPStrategy(HighFrequencyVWAPStrategy):
    """
    Adaptive version that automatically adjusts parameters based on market conditions
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(parameters)
        self.adaptation_window = parameters.get('adaptation_window', 1440)  # 1 day
        self.performance_memory = []
        
    def adapt_parameters(self, df: pd.DataFrame, current_idx: int) -> Dict[str, Any]:
        """Adapt parameters based on recent performance"""
        if current_idx < self.adaptation_window:
            return self.parameters
        
        # Analyze recent performance
        recent_data = df.iloc[current_idx - self.adaptation_window:current_idx]
        
        # Calculate recent volatility regime
        recent_vol = recent_data['realized_vol_1h'].mean()
        long_term_vol = df.iloc[:current_idx]['realized_vol_1h'].mean()
        
        adapted_params = self.parameters.copy()
        
        # Adapt VWAP threshold based on volatility
        vol_ratio = recent_vol / long_term_vol if long_term_vol > 0 else 1.0
        adapted_params['vwap_threshold'] = self.parameters['vwap_threshold'] * vol_ratio
        
        # Adapt volume threshold based on recent volume patterns
        recent_volume_norm = recent_data['volume_normalized'].mean()
        if recent_volume_norm > 1.5:  # High volume period
            adapted_params['volume_threshold'] = self.parameters['volume_threshold'] * 0.8
        elif recent_volume_norm < 0.8:  # Low volume period
            adapted_params['volume_threshold'] = self.parameters['volume_threshold'] * 1.2
        
        return adapted_params


def main():
    """Example usage"""
    print("High-Frequency VWAP Strategy loaded successfully")


if __name__ == "__main__":
    main()