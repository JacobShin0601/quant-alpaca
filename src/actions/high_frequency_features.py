"""
High-Frequency Feature Engineering for 1-minute Cryptocurrency Trading
Specialized features for microstructure analysis and high-frequency trading
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class HighFrequencyFeatureEngineer:
    """Advanced feature engineering for high-frequency cryptocurrency trading"""
    
    def __init__(self):
        self.features = {}
        self.lookback_periods = {
            'ultra_short': [1, 3, 5],          # 1-5 minutes
            'short': [10, 15, 20],             # 10-20 minutes  
            'medium': [60, 120, 240],          # 1-4 hours
            'long': [1440, 2880, 4320]         # 1-3 days (in minutes)
        }
    
    def calculate_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market microstructure features for high-frequency trading"""
        df = df.copy()
        
        # Basic returns with different timeframes
        df['returns'] = np.log(df['trade_price'] / df['trade_price'].shift(1))
        df['returns_1m'] = df['returns']
        df['returns_5m'] = np.log(df['trade_price'] / df['trade_price'].shift(5))
        df['returns_15m'] = np.log(df['trade_price'] / df['trade_price'].shift(15))
        
        # Volatility-adjusted returns
        for period in [60, 120, 240]:  # 1-4 hours
            vol = df['returns'].rolling(window=period).std()
            df[f'vol_adjusted_returns_{period}'] = df['returns'] / vol
            
        # Bid-Ask Spread estimation (using High-Low)
        df['estimated_spread'] = (df['high_price'] - df['low_price']) / df['trade_price']
        df['spread_ma_20'] = df['estimated_spread'].rolling(window=20).mean()
        df['spread_normalized'] = df['estimated_spread'] / df['spread_ma_20']
        
        # Volume normalization (24-hour rolling average)
        df['volume_ma_1d'] = df['candle_acc_trade_volume'].rolling(window=1440).mean()
        df['volume_normalized'] = df['candle_acc_trade_volume'] / df['volume_ma_1d']
        
        # Price momentum across multiple timeframes
        for period in [1, 5, 15, 30, 60]:
            df[f'momentum_{period}m'] = (df['trade_price'] / df['trade_price'].shift(period) - 1) * 100
            
        # Realized volatility (different periods)
        for period in [15, 30, 60, 240]:
            rv = df['returns'].rolling(window=period).std() * np.sqrt(1440)  # Annualized
            df[f'realized_vol_{period}m'] = rv
            
        return df
    
    def calculate_volume_clustering_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume clustering and flow features"""
        df = df.copy()
        
        # Volume Rate of Change
        for period in [5, 10, 20]:
            df[f'volume_roc_{period}'] = df['candle_acc_trade_volume'].pct_change(periods=period) * 100
            
        # Volume clustering detection
        volume_std = df['candle_acc_trade_volume'].rolling(window=60).std()
        volume_mean = df['candle_acc_trade_volume'].rolling(window=60).mean()
        df['volume_zscore'] = (df['candle_acc_trade_volume'] - volume_mean) / volume_std
        df['high_volume_flag'] = (df['volume_zscore'] > 2).astype(int)
        df['low_volume_flag'] = (df['volume_zscore'] < -1).astype(int)
        
        # Volume-weighted features
        df['vwap_1h'] = self._calculate_rolling_vwap(df, 60)
        df['vwap_4h'] = self._calculate_rolling_vwap(df, 240)
        df['price_vwap_ratio_1h'] = df['trade_price'] / df['vwap_1h']
        df['price_vwap_ratio_4h'] = df['trade_price'] / df['vwap_4h']
        
        # Order flow approximation
        df['buy_pressure'] = np.where(df['trade_price'] > df['trade_price'].shift(1), 
                                     df['candle_acc_trade_volume'], 0)
        df['sell_pressure'] = np.where(df['trade_price'] < df['trade_price'].shift(1), 
                                      df['candle_acc_trade_volume'], 0)
        
        for period in [10, 30, 60]:
            df[f'net_flow_{period}'] = (df['buy_pressure'].rolling(period).sum() - 
                                       df['sell_pressure'].rolling(period).sum())
            df[f'flow_ratio_{period}'] = (df['buy_pressure'].rolling(period).sum() / 
                                         (df['buy_pressure'].rolling(period).sum() + 
                                          df['sell_pressure'].rolling(period).sum() + 1e-8))
        
        return df
    
    def calculate_regime_detection_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for market regime detection"""
        df = df.copy()
        
        # Volatility regime detection
        for period in [60, 240, 480]:  # 1h, 4h, 8h
            vol = df['returns'].rolling(window=period).std() * np.sqrt(1440)
            vol_ma = vol.rolling(window=period//2).mean()
            df[f'vol_regime_{period}'] = np.where(vol > vol_ma * 1.5, 2,  # High vol
                                                 np.where(vol < vol_ma * 0.7, 0, 1))  # Low/Normal vol
        
        # Trend strength using ADX-like calculation
        for period in [20, 60]:
            df[f'trend_strength_{period}'] = self._calculate_trend_strength(df, period)
            
        # Market efficiency measure (Hurst exponent approximation)
        for period in [60, 240]:
            df[f'hurst_{period}'] = self._calculate_hurst_approx(df, period)
            
        # Choppiness index
        for period in [30, 60, 120]:
            df[f'choppiness_{period}'] = self._calculate_choppiness_index(df, period)
            
        return df
    
    def calculate_multi_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate multi-timeframe technical indicators optimized for 1-minute data"""
        df = df.copy()
        
        # Short-term RSI (optimized for 1-min data)
        for period in [5, 10, 14, 20]:
            df[f'rsi_{period}'] = self._calculate_rsi(df, period)
            
        # Fast MACD configurations
        fast_periods = [(3, 8, 5), (5, 13, 8), (8, 21, 13)]
        for i, (fast, slow, signal) in enumerate(fast_periods):
            ema_fast = df['trade_price'].ewm(span=fast).mean()
            ema_slow = df['trade_price'].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            df[f'macd_{i}_line'] = macd
            df[f'macd_{i}_signal'] = macd_signal
            df[f'macd_{i}_histogram'] = macd - macd_signal
            df[f'macd_{i}_cross'] = (macd > macd_signal).astype(int)
            
        # Bollinger Bands with multiple periods
        for period in [10, 20, 40]:
            sma = df['trade_price'].rolling(window=period).mean()
            std = df['trade_price'].rolling(window=period).std()
            df[f'bb_upper_{period}'] = sma + (2 * std)
            df[f'bb_lower_{period}'] = sma - (2 * std)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
            df[f'bb_position_{period}'] = (df['trade_price'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
            
        # Multi-timeframe trend alignment
        sma_periods = [5, 10, 20, 50]
        trend_scores = []
        for i in range(len(df)):
            score = 0
            for period in sma_periods:
                if i >= period:
                    sma = df['trade_price'].iloc[max(0, i-period+1):i+1].mean()
                    if df['trade_price'].iloc[i] > sma:
                        score += 1
            trend_scores.append(score / len(sma_periods))
        df['trend_alignment'] = trend_scores
        
        return df
    
    def calculate_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price action and pattern features"""
        df = df.copy()
        
        # Candlestick patterns
        df['body_size'] = abs(df['trade_price'] - df['opening_price']) / df['trade_price']
        df['upper_shadow'] = (df['high_price'] - np.maximum(df['trade_price'], df['opening_price'])) / df['trade_price']
        df['lower_shadow'] = (np.minimum(df['trade_price'], df['opening_price']) - df['low_price']) / df['trade_price']
        df['total_range'] = (df['high_price'] - df['low_price']) / df['trade_price']
        
        # Doji detection
        df['is_doji'] = (df['body_size'] < 0.001).astype(int)
        
        # Gap detection
        df['gap_up'] = (df['opening_price'] > df['high_price'].shift(1)).astype(int)
        df['gap_down'] = (df['opening_price'] < df['low_price'].shift(1)).astype(int)
        
        # Support/Resistance levels (local extremes)
        for period in [10, 20, 50]:
            df[f'local_high_{period}'] = df['high_price'].rolling(window=period, center=True).max() == df['high_price']
            df[f'local_low_{period}'] = df['low_price'].rolling(window=period, center=True).min() == df['low_price']
            
        # Price velocity and acceleration
        df['price_velocity_5'] = df['trade_price'].diff(5) / 5  # 5-minute price velocity
        df['price_acceleration_5'] = df['price_velocity_5'].diff(5) / 5
        
        return df
    
    def calculate_liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate liquidity-related features"""
        df = df.copy()
        
        # Amihud illiquidity measure approximation
        for period in [30, 60, 120]:
            price_impact = abs(df['returns']) / (df['candle_acc_trade_volume'] + 1e-8)
            df[f'illiquidity_{period}'] = price_impact.rolling(window=period).mean()
            
        # Volume-price relationship
        for period in [20, 60]:
            corr = df['returns'].rolling(window=period).corr(df['candle_acc_trade_volume'])
            df[f'volume_price_corr_{period}'] = corr
            
        # Roll's spread estimator
        for period in [30, 60]:
            covariance = df['returns'].rolling(window=period).cov(df['returns'].shift(1))
            df[f'roll_spread_{period}'] = 2 * np.sqrt(np.maximum(-covariance, 0))
            
        return df
    
    def calculate_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced statistical features"""
        df = df.copy()
        
        # Higher moments
        for period in [30, 60, 120]:
            returns_window = df['returns'].rolling(window=period)
            df[f'skewness_{period}'] = returns_window.skew()
            df[f'kurtosis_{period}'] = returns_window.kurt()
            
        # Jarque-Bera normality test approximation
        for period in [60, 120]:
            returns_window = df['returns'].rolling(window=period)
            skew = returns_window.skew()
            kurt = returns_window.kurt()
            df[f'jb_stat_{period}'] = period/6 * (skew**2 + (kurt-3)**2/4)
            
        # Autocorrelation features
        for lag in [1, 5, 10]:
            df[f'autocorr_lag_{lag}'] = df['returns'].rolling(window=60).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
            )
            
        return df
    
    def _calculate_rolling_vwap(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate rolling VWAP over specified period"""
        typical_price = (df['high_price'] + df['low_price'] + df['trade_price']) / 3
        volume_price = typical_price * df['candle_acc_trade_volume']
        
        rolling_vp = volume_price.rolling(window=period).sum()
        rolling_volume = df['candle_acc_trade_volume'].rolling(window=period).sum()
        
        return rolling_vp / rolling_volume
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate RSI with specified period"""
        delta = df['trade_price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_trend_strength(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate trend strength using directional movement"""
        high_diff = df['high_price'].diff()
        low_diff = df['low_price'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
        
        tr = np.maximum(df['high_price'] - df['low_price'],
                      np.maximum(abs(df['high_price'] - df['trade_price'].shift(1)),
                               abs(df['low_price'] - df['trade_price'].shift(1))))
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / 
                        pd.Series(tr).rolling(window=period).mean())
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / 
                         pd.Series(tr).rolling(window=period).mean())
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def _calculate_hurst_approx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Hurst exponent approximation using R/S analysis"""
        def hurst_window(returns_window):
            if len(returns_window) < 10:
                return np.nan
            
            # Simplified Hurst calculation
            cumulative = np.cumsum(returns_window - returns_window.mean())
            R = np.max(cumulative) - np.min(cumulative)
            S = returns_window.std()
            
            if S == 0:
                return 0.5
            
            rs = R / S
            return np.log(rs) / np.log(len(returns_window))
        
        return df['returns'].rolling(window=period).apply(hurst_window)
    
    def _calculate_choppiness_index(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Choppiness Index"""
        tr = np.maximum(df['high_price'] - df['low_price'],
                      np.maximum(abs(df['high_price'] - df['trade_price'].shift(1)),
                               abs(df['low_price'] - df['trade_price'].shift(1))))
        
        atr_sum = tr.rolling(window=period).sum()
        high_low_range = (df['high_price'].rolling(window=period).max() - 
                         df['low_price'].rolling(window=period).min())
        
        chop = 100 * np.log10(atr_sum / high_low_range) / np.log10(period)
        return chop
    
    def engineer_all_hf_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all high-frequency feature engineering methods"""
        print("Engineering microstructure features...")
        df = self.calculate_microstructure_features(df)
        
        print("Engineering volume clustering features...")
        df = self.calculate_volume_clustering_features(df)
        
        print("Engineering regime detection features...")
        df = self.calculate_regime_detection_features(df)
        
        print("Engineering multi-timeframe features...")
        df = self.calculate_multi_timeframe_features(df)
        
        print("Engineering price action features...")
        df = self.calculate_price_action_features(df)
        
        print("Engineering liquidity features...")
        df = self.calculate_liquidity_features(df)
        
        print("Engineering statistical features...")
        df = self.calculate_statistical_features(df)
        
        print(f"High-frequency feature engineering completed. Total features: {len(df.columns)}")
        
        return df
    
    def get_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get summary of engineered features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        summary = pd.DataFrame({
            'feature': numeric_cols,
            'non_null_count': [df[col].count() for col in numeric_cols],
            'null_percentage': [(df[col].isnull().sum() / len(df)) * 100 for col in numeric_cols],
            'mean': [df[col].mean() for col in numeric_cols],
            'std': [df[col].std() for col in numeric_cols],
            'min': [df[col].min() for col in numeric_cols],
            'max': [df[col].max() for col in numeric_cols]
        })
        
        return summary.sort_values('null_percentage')


class DynamicParameterOptimizer:
    """Dynamic parameter optimization for high-frequency strategies"""
    
    def __init__(self, optimization_window: int = 1440):  # 1 day default
        self.optimization_window = optimization_window
        self.parameter_history = {}
    
    def rolling_optimization(self, data: pd.DataFrame, strategy_func, param_ranges: Dict, 
                           optimization_metric: str = 'sharpe_ratio') -> pd.DataFrame:
        """Perform rolling window optimization"""
        results = []
        
        for i in range(self.optimization_window, len(data)):
            # Get training window
            train_data = data.iloc[i-self.optimization_window:i]
            
            # Optimize parameters (simplified grid search)
            best_params = self._optimize_parameters(train_data, strategy_func, param_ranges, optimization_metric)
            
            # Store results
            results.append({
                'timestamp': data.index[i],
                'best_params': best_params,
                'window_start': train_data.index[0],
                'window_end': train_data.index[-1]
            })
        
        return pd.DataFrame(results)
    
    def _optimize_parameters(self, data: pd.DataFrame, strategy_func, param_ranges: Dict, 
                           metric: str) -> Dict:
        """Optimize parameters using grid search (simplified)"""
        best_score = -np.inf
        best_params = {}
        
        # Create parameter combinations (simplified for demonstration)
        import itertools
        
        param_names = list(param_ranges.keys())
        param_values = [param_ranges[name] for name in param_names]
        
        for param_combination in itertools.product(*param_values):
            params = dict(zip(param_names, param_combination))
            
            try:
                # Run strategy with parameters
                signals = strategy_func(data, **params)
                performance = self._calculate_performance(data, signals)
                
                score = performance.get(metric, -np.inf)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    
            except Exception as e:
                continue
        
        return best_params
    
    def _calculate_performance(self, data: pd.DataFrame, signals: pd.Series) -> Dict:
        """Calculate performance metrics"""
        returns = data['returns'] * signals.shift(1)  # Lag signals by 1
        
        metrics = {}
        metrics['total_return'] = returns.sum()
        metrics['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(1440) if returns.std() > 0 else 0
        metrics['max_drawdown'] = (returns.cumsum() - returns.cumsum().expanding().max()).min()
        
        return metrics


class WalkForwardAnalyzer:
    """Walk-forward analysis framework for high-frequency strategies"""
    
    def __init__(self, train_period: int = 4320, test_period: int = 1440):  # 3 days train, 1 day test
        self.train_period = train_period
        self.test_period = test_period
    
    def run_walk_forward_analysis(self, data: pd.DataFrame, strategy_func, 
                                 param_ranges: Dict) -> pd.DataFrame:
        """Run complete walk-forward analysis"""
        results = []
        
        start_idx = self.train_period
        
        while start_idx + self.test_period < len(data):
            # Define periods
            train_start = start_idx - self.train_period
            train_end = start_idx
            test_start = start_idx
            test_end = start_idx + self.test_period
            
            # Get data splits
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            # Optimize on training data
            optimizer = DynamicParameterOptimizer()
            best_params = optimizer._optimize_parameters(train_data, strategy_func, param_ranges, 'sharpe_ratio')
            
            # Test on out-of-sample data
            test_signals = strategy_func(test_data, **best_params)
            test_performance = optimizer._calculate_performance(test_data, test_signals)
            
            # Store results
            result = {
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1], 
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'best_params': best_params,
                **{f'test_{k}': v for k, v in test_performance.items()}
            }
            results.append(result)
            
            # Move window forward
            start_idx += self.test_period
            
            print(f"Completed walk-forward period: {test_data.index[0]} to {test_data.index[-1]}")
        
        return pd.DataFrame(results)


def main():
    """Example usage of high-frequency feature engineering"""
    print("High-Frequency Feature Engineering module loaded successfully")
    print("Use HighFrequencyFeatureEngineer class for 1-minute cryptocurrency trading features")


if __name__ == "__main__":
    main()