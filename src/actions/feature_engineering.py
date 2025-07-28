import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


class FeatureEngineer:
    """Feature engineering for cryptocurrency trading data"""
    
    def __init__(self):
        self.features = {}
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic technical indicators"""
        df = df.copy()
        
        # Price-based features
        df['returns'] = df['trade_price'].pct_change()
        df['log_returns'] = np.log(df['trade_price'] / df['trade_price'].shift(1))
        
        # Simple Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'ma_{period}'] = df['trade_price'].rolling(window=period).mean()
            df[f'price_ma_{period}_ratio'] = df['trade_price'] / df[f'ma_{period}']
        
        # Exponential Moving Averages (EMA)
        for period in [5, 10, 12, 20, 26, 50, 100]:
            df[f'ema_{period}'] = df['trade_price'].ewm(span=period).mean()
            df[f'price_ema_{period}_ratio'] = df['trade_price'] / df[f'ema_{period}']
            
        # EMA crossovers
        df['ema_12_26_diff'] = df['ema_12'] - df['ema_26']
        df['ema_5_20_diff'] = df['ema_5'] - df['ema_20']
        
        # Volatility measures
        for period in [10, 20, 50]:
            df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()
            df[f'realized_vol_{period}'] = df['log_returns'].rolling(window=period).std() * np.sqrt(1440)  # 1-day vol
        
        # Volume features
        df['volume_ma_20'] = df['candle_acc_trade_volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['candle_acc_trade_volume'] / df['volume_ma_20']
        df['price_volume'] = df['trade_price'] * df['candle_acc_trade_volume']
        
        # High-Low spread
        df['hl_pct'] = (df['high_price'] - df['low_price']) / df['trade_price'] * 100
        df['hl_ratio'] = df['high_price'] / df['low_price']
        
        return df
    
    def calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum-based features"""
        df = df.copy()
        
        # Price momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['trade_price'] / df['trade_price'].shift(period) - 1
            df[f'momentum_{period}_ma'] = df[f'momentum_{period}'].rolling(window=5).mean()
        
        # RSI
        for period in [7, 14, 21]:
            delta = df['trade_price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Stochastic
        for period in [14, 21]:
            lowest_low = df['low_price'].rolling(window=period).min()
            highest_high = df['high_price'].rolling(window=period).max()
            df[f'stoch_k_{period}'] = ((df['trade_price'] - lowest_low) / (highest_high - lowest_low)) * 100
            df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(window=3).mean()
        
        return df
    
    def calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-following indicators"""
        df = df.copy()
        
        # ADX (Average Directional Index)
        for period in [14, 21]:
            high_diff = df['high_price'].diff()
            low_diff = df['low_price'].diff()
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
            
            tr = np.maximum(df['high_price'] - df['low_price'],
                          np.maximum(abs(df['high_price'] - df['trade_price'].shift(1)),
                                   abs(df['low_price'] - df['trade_price'].shift(1))))
            
            plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period).mean() / 
                           pd.Series(tr).ewm(alpha=1/period).mean())
            minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period).mean() / 
                            pd.Series(tr).ewm(alpha=1/period).mean())
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            df[f'adx_{period}'] = dx.ewm(alpha=1/period).mean()
            df[f'plus_di_{period}'] = plus_di
            df[f'minus_di_{period}'] = minus_di
        
        # MACD
        ema_12 = df['trade_price'].ewm(span=12).mean()
        ema_26 = df['trade_price'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Parabolic SAR
        df['psar'] = self._calculate_parabolic_sar(df)
        df['psar_trend'] = np.where(df['trade_price'] > df['psar'], 1, -1)
        
        # Aroon Indicator
        for period in [14, 25]:
            aroon_up = []
            aroon_down = []
            for i in range(len(df)):
                if i < period:
                    aroon_up.append(np.nan)
                    aroon_down.append(np.nan)
                else:
                    high_idx = df['high_price'].iloc[i-period:i+1].idxmax()
                    low_idx = df['low_price'].iloc[i-period:i+1].idxmin()
                    aroon_up.append(((period - (i - df.index.get_loc(high_idx))) / period) * 100)
                    aroon_down.append(((period - (i - df.index.get_loc(low_idx))) / period) * 100)
            
            df[f'aroon_up_{period}'] = aroon_up
            df[f'aroon_down_{period}'] = aroon_down
            df[f'aroon_oscillator_{period}'] = df[f'aroon_up_{period}'] - df[f'aroon_down_{period}']
        
        return df
    
    def calculate_oscillators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate oscillator indicators"""
        df = df.copy()
        
        # Williams %R
        for period in [14, 21]:
            highest_high = df['high_price'].rolling(window=period).max()
            lowest_low = df['low_price'].rolling(window=period).min()
            df[f'williams_r_{period}'] = -100 * ((highest_high - df['trade_price']) / (highest_high - lowest_low))
        
        # Commodity Channel Index (CCI)
        for period in [14, 20]:
            typical_price = (df['high_price'] + df['low_price'] + df['trade_price']) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
            df[f'cci_{period}'] = (typical_price - sma_tp) / (0.015 * mad)
        
        # Ultimate Oscillator
        bp = df['trade_price'] - np.minimum(df['low_price'], df['trade_price'].shift(1))
        tr = np.maximum(df['high_price'], df['trade_price'].shift(1)) - np.minimum(df['low_price'], df['trade_price'].shift(1))
        
        avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
        avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
        avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
        
        df['ultimate_oscillator'] = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
        
        # Stochastic RSI
        for period in [14, 21]:
            rsi = df[f'rsi_{period}'] if f'rsi_{period}' in df.columns else self._calculate_rsi(df, period)
            rsi_low = rsi.rolling(window=period).min()
            rsi_high = rsi.rolling(window=period).max()
            df[f'stoch_rsi_{period}'] = (rsi - rsi_low) / (rsi_high - rsi_low) * 100
        
        return df
    
    def calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-based indicators"""
        df = df.copy()
        
        # Bollinger Bands
        for period in [20, 50]:
            ma = df['trade_price'].rolling(window=period).mean()
            std = df['trade_price'].rolling(window=period).std()
            
            df[f'bb_upper_{period}'] = ma + (2 * std)
            df[f'bb_lower_{period}'] = ma - (2 * std)
            df[f'bb_middle_{period}'] = ma
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / df[f'bb_middle_{period}']
            df[f'bb_position_{period}'] = (df['trade_price'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
        
        # Keltner Channels
        for period in [20, 50]:
            ema = df['trade_price'].ewm(span=period).mean()
            atr = self._calculate_atr(df, period)
            multiplier = 2
            
            df[f'keltner_upper_{period}'] = ema + (multiplier * atr)
            df[f'keltner_lower_{period}'] = ema - (multiplier * atr)
            df[f'keltner_middle_{period}'] = ema
        
        # Average True Range (ATR)
        for period in [7, 14, 21]:
            df[f'atr_{period}'] = self._calculate_atr(df, period)
            df[f'atr_ratio_{period}'] = df[f'atr_{period}'] / df['trade_price']
        
        # Historical Volatility
        for period in [10, 20, 30]:
            df[f'hv_{period}'] = df['log_returns'].rolling(window=period).std() * np.sqrt(252)
        
        return df
    
    def calculate_pattern_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pattern recognition indicators"""
        df = df.copy()
        
        # Pivot Points
        df['pivot'] = (df['high_price'] + df['low_price'] + df['trade_price']) / 3
        df['pivot_r1'] = 2 * df['pivot'] - df['low_price']
        df['pivot_s1'] = 2 * df['pivot'] - df['high_price']
        df['pivot_r2'] = df['pivot'] + (df['high_price'] - df['low_price'])
        df['pivot_s2'] = df['pivot'] - (df['high_price'] - df['low_price'])
        
        # Support/Resistance levels
        for period in [10, 20, 50]:
            df[f'resistance_{period}'] = df['high_price'].rolling(window=period).max()
            df[f'support_{period}'] = df['low_price'].rolling(window=period).min()
            df[f'support_resistance_ratio_{period}'] = df['trade_price'] / ((df[f'resistance_{period}'] + df[f'support_{period}']) / 2)
        
        # Price channels
        for period in [20, 50]:
            highest = df['high_price'].rolling(window=period).max()
            lowest = df['low_price'].rolling(window=period).min()
            df[f'channel_position_{period}'] = (df['trade_price'] - lowest) / (highest - lowest)
        
        # Fractal indicators
        df['fractal_high'] = self._identify_fractals(df['high_price'], high=True)
        df['fractal_low'] = self._identify_fractals(df['low_price'], high=False)
        
        return df
    
    def calculate_ichimoku_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku Cloud components"""
        df = df.copy()
        
        # Ichimoku components
        high_9 = df['high_price'].rolling(window=9).max()
        low_9 = df['low_price'].rolling(window=9).min()
        df['tenkan_sen'] = (high_9 + low_9) / 2
        
        high_26 = df['high_price'].rolling(window=26).max()
        low_26 = df['low_price'].rolling(window=26).min()
        df['kijun_sen'] = (high_26 + low_26) / 2
        
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        
        high_52 = df['high_price'].rolling(window=52).max()
        low_52 = df['low_price'].rolling(window=52).min()
        df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
        
        df['chikou_span'] = df['trade_price'].shift(-26)
        
        # Ichimoku signals
        df['ichimoku_cloud_top'] = np.maximum(df['senkou_span_a'], df['senkou_span_b'])
        df['ichimoku_cloud_bottom'] = np.minimum(df['senkou_span_a'], df['senkou_span_b'])
        df['price_above_cloud'] = (df['trade_price'] > df['ichimoku_cloud_top']).astype(int)
        df['price_below_cloud'] = (df['trade_price'] < df['ichimoku_cloud_bottom']).astype(int)
        df['price_in_cloud'] = ((df['trade_price'] >= df['ichimoku_cloud_bottom']) & 
                               (df['trade_price'] <= df['ichimoku_cloud_top'])).astype(int)
        
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Helper function to calculate RSI"""
        delta = df['trade_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Helper function to calculate Average True Range"""
        high_low = df['high_price'] - df['low_price']
        high_close = abs(df['high_price'] - df['trade_price'].shift(1))
        low_close = abs(df['low_price'] - df['trade_price'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    def _calculate_parabolic_sar(self, df: pd.DataFrame, af_start=0.02, af_increment=0.02, af_max=0.2) -> pd.Series:
        """Helper function to calculate Parabolic SAR"""
        high = df['high_price'].values
        low = df['low_price'].values
        close = df['trade_price'].values
        
        sar = np.zeros(len(close))
        trend = np.zeros(len(close))
        af = np.zeros(len(close))
        ep = np.zeros(len(close))
        
        sar[0] = low[0]
        trend[0] = 1
        af[0] = af_start
        ep[0] = high[0]
        
        for i in range(1, len(close)):
            if trend[i-1] == 1:  # Uptrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                if low[i] <= sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    af[i] = af_start
                    ep[i] = low[i]
                else:
                    trend[i] = 1
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + af_increment, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            else:  # Downtrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                if high[i] >= sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    af[i] = af_start
                    ep[i] = high[i]
                else:
                    trend[i] = -1
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + af_increment, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
        
        return pd.Series(sar, index=df.index)
    
    def _identify_fractals(self, series: pd.Series, high: bool = True, window: int = 2) -> pd.Series:
        """Helper function to identify fractal points"""
        fractals = pd.Series(0, index=series.index)
        
        for i in range(window, len(series) - window):
            if high:
                if all(series.iloc[i] > series.iloc[i-j] for j in range(1, window+1)) and \
                   all(series.iloc[i] > series.iloc[i+j] for j in range(1, window+1)):
                    fractals.iloc[i] = 1
            else:
                if all(series.iloc[i] < series.iloc[i-j] for j in range(1, window+1)) and \
                   all(series.iloc[i] < series.iloc[i+j] for j in range(1, window+1)):
                    fractals.iloc[i] = 1
        
        return fractals
    
    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features"""
        df = df.copy()
        
        # Volume moving averages
        for period in [5, 10, 20]:
            df[f'volume_ma_{period}'] = df['candle_acc_trade_volume'].rolling(window=period).mean()
            df[f'volume_ratio_{period}'] = df['candle_acc_trade_volume'] / df[f'volume_ma_{period}']
        
        # Volume trend
        df['volume_trend_5'] = df['candle_acc_trade_volume'].rolling(window=5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
        )
        
        # On-Balance Volume (OBV)
        df['obv'] = (df['candle_acc_trade_volume'] * np.sign(df['returns'])).cumsum()
        df['obv_ma_10'] = df['obv'].rolling(window=10).mean()
        
        # Volume-Price Trend (VPT)
        df['vpt'] = (df['candle_acc_trade_volume'] * df['returns']).cumsum()
        
        # Accumulation/Distribution Line (A/D Line)
        mfm = ((df['trade_price'] - df['low_price']) - (df['high_price'] - df['trade_price'])) / (df['high_price'] - df['low_price'])
        mfv = mfm * df['candle_acc_trade_volume']
        df['ad_line'] = mfv.cumsum()
        
        # Chaikin Money Flow (CMF)
        for period in [10, 20]:
            df[f'cmf_{period}'] = (mfv.rolling(window=period).sum() / 
                                 df['candle_acc_trade_volume'].rolling(window=period).sum())
        
        # Volume Weighted Average Price (VWAP)
        cumulative_volume = df['candle_acc_trade_volume'].cumsum()
        cumulative_volume_price = (df['trade_price'] * df['candle_acc_trade_volume']).cumsum()
        df['vwap'] = cumulative_volume_price / cumulative_volume
        
        # Price Volume Trend (PVT)
        df['pvt'] = ((df['trade_price'].pct_change() * df['candle_acc_trade_volume'])).cumsum()
        
        # Ease of Movement (EOM)
        distance_moved = ((df['high_price'] + df['low_price']) / 2) - ((df['high_price'].shift(1) + df['low_price'].shift(1)) / 2)
        box_height = df['candle_acc_trade_volume'] / (df['high_price'] - df['low_price'])
        df['eom'] = distance_moved / box_height
        df['eom_ma'] = df['eom'].rolling(window=14).mean()
        
        # Force Index
        df['force_index'] = df['candle_acc_trade_volume'] * (df['trade_price'] - df['trade_price'].shift(1))
        df['force_index_ma'] = df['force_index'].rolling(window=13).mean()
        
        return df
    
    def calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time-based features"""
        df = df.copy()
        
        # Convert datetime if not already
        if 'candle_date_time_utc' in df.columns:
            df['datetime'] = pd.to_datetime(df['candle_date_time_utc'])
        else:
            df['datetime'] = df.index
        
        # Time features
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_month'] = df['datetime'].dt.day
        
        # Market session features
        df['is_korean_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 23)).astype(int)
        df['is_us_hours'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)  # Considering UTC
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def calculate_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistical features"""
        df = df.copy()
        
        # Rolling statistics
        for period in [10, 20, 50]:
            df[f'skew_{period}'] = df['returns'].rolling(window=period).skew()
            df[f'kurtosis_{period}'] = df['returns'].rolling(window=period).kurt()
            df[f'var_{period}'] = df['returns'].rolling(window=period).var()
        
        # Quantile features
        for period in [20, 50]:
            df[f'q25_{period}'] = df['trade_price'].rolling(window=period).quantile(0.25)
            df[f'q75_{period}'] = df['trade_price'].rolling(window=period).quantile(0.75)
            df[f'iqr_{period}'] = df[f'q75_{period}'] - df[f'q25_{period}']
        
        # Z-score
        for period in [20, 50]:
            rolling_mean = df['trade_price'].rolling(window=period).mean()
            rolling_std = df['trade_price'].rolling(window=period).std()
            df[f'zscore_{period}'] = (df['trade_price'] - rolling_mean) / rolling_std
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering methods"""
        print("Engineering technical indicators...")
        df = self.calculate_technical_indicators(df)
        
        print("Engineering momentum features...")
        df = self.calculate_momentum_features(df)
        
        print("Engineering trend indicators...")
        df = self.calculate_trend_indicators(df)
        
        print("Engineering oscillators...")
        df = self.calculate_oscillators(df)
        
        print("Engineering volatility indicators...")
        df = self.calculate_volatility_indicators(df)
        
        print("Engineering pattern indicators...")
        df = self.calculate_pattern_indicators(df)
        
        print("Engineering Ichimoku indicators...")
        df = self.calculate_ichimoku_indicators(df)
        
        print("Engineering volume features...")
        df = self.calculate_volume_features(df)
        
        print("Engineering time features...")
        df = self.calculate_time_features(df)
        
        print("Engineering statistical features...")
        df = self.calculate_statistical_features(df)
        
        print(f"Feature engineering completed. Total features: {len(df.columns)}")
        
        return df
    
    def get_feature_importance(self, df: pd.DataFrame, target_col: str = 'returns') -> pd.DataFrame:
        """Calculate feature importance using correlation"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
        
        importance_df = pd.DataFrame({
            'feature': correlations.index,
            'importance': correlations.values
        })
        
        return importance_df.dropna()


def main():
    """Example usage of FeatureEngineer"""
    # This would typically be called with real data
    print("FeatureEngineer module loaded successfully")
    print("Use FeatureEngineer class to engineer features from your trading data")


if __name__ == "__main__":
    main()