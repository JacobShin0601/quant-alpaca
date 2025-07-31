"""
Enhanced Ensemble Strategy with Market Microstructure Regimes
Advanced ensemble combining regime detection, microstructure analysis, and adaptive optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
try:
    from .base import BaseStrategy
    from .adaptive_strategy_base import AdaptiveStrategyMixin
    from ..actions.market_regime import MarketRegimeDetector, MarketRegime
except ImportError:
    # Handle when running from different contexts
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from strategies.base import BaseStrategy
    from strategies.adaptive_strategy_base import AdaptiveStrategyMixin
    from actions.market_regime import MarketRegimeDetector, MarketRegime
import warnings
warnings.filterwarnings('ignore')


class MicrostructureRegimeDetector:
    """
    Detects market microstructure regimes based on:
    - Volume patterns (clustering, flow, acceleration)
    - Liquidity conditions (spread, depth, impact)
    - Price action (momentum, mean reversion, breakout)
    """
    
    def __init__(self):
        self.regime_history = []
        
    def detect_microstructure_regime(self, df: pd.DataFrame, lookback: int = 60) -> str:
        """Detect current microstructure regime"""
        if len(df) < lookback:
            return 'unknown'
        
        recent_data = df.tail(lookback).copy()
        
        # Calculate microstructure indicators
        ms_indicators = self._calculate_microstructure_indicators(recent_data)
        
        # Classify regime based on indicators
        regime = self._classify_microstructure_regime(ms_indicators)
        
        return regime
    
    def _calculate_microstructure_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate microstructure indicators"""
        indicators = {}
        
        # Volume clustering
        volume_std = df['candle_acc_trade_volume'].std()
        volume_mean = df['candle_acc_trade_volume'].mean()
        indicators['volume_clustering'] = volume_std / volume_mean if volume_mean > 0 else 0
        
        # Volume acceleration
        volume_diff = df['candle_acc_trade_volume'].diff()
        indicators['volume_acceleration'] = volume_diff.mean()
        
        # Price momentum persistence
        returns = np.log(df['trade_price'] / df['trade_price'].shift(1))
        momentum_autocorr = returns.autocorr(lag=1) if len(returns) > 1 else 0
        indicators['momentum_persistence'] = momentum_autocorr
        
        # Spread proxy (high-low relative to price)
        spread_proxy = (df['high_price'] - df['low_price']) / df['trade_price']
        indicators['avg_spread'] = spread_proxy.mean()
        indicators['spread_volatility'] = spread_proxy.std()
        
        # Price impact approximation
        price_changes = abs(returns)
        volume_normalized = df['candle_acc_trade_volume'] / df['candle_acc_trade_volume'].mean()
        indicators['price_impact'] = (price_changes / volume_normalized).mean()
        
        # Mean reversion tendency
        price_ma = df['trade_price'].rolling(window=20).mean()
        deviations = (df['trade_price'] - price_ma) / price_ma
        mean_reversion = -(deviations * deviations.shift(-1)).corr(deviations)
        indicators['mean_reversion_strength'] = mean_reversion if not pd.isna(mean_reversion) else 0
        
        return indicators
    
    def _classify_microstructure_regime(self, indicators: Dict[str, float]) -> str:
        """Classify microstructure regime based on indicators"""
        # Define thresholds (can be optimized)
        high_volume_clustering = indicators['volume_clustering'] > 1.5
        high_momentum_persistence = indicators['momentum_persistence'] > 0.3
        high_spread = indicators['avg_spread'] > 0.01
        high_impact = indicators['price_impact'] > 0.001
        strong_mean_reversion = indicators['mean_reversion_strength'] > 0.2
        
        # Regime classification logic
        if high_momentum_persistence and not high_spread:
            return 'trending_liquid'
        elif high_momentum_persistence and high_spread:
            return 'trending_illiquid'
        elif strong_mean_reversion and not high_spread:
            return 'mean_reverting_liquid'
        elif strong_mean_reversion and high_spread:
            return 'mean_reverting_illiquid'
        elif high_volume_clustering and high_impact:
            return 'volatile_thin'
        elif not high_volume_clustering and not high_impact:
            return 'stable_liquid'
        else:
            return 'mixed'


class EnhancedEnsembleStrategy(AdaptiveStrategyMixin, BaseStrategy):
    """
    Enhanced Ensemble Strategy with:
    - Market regime detection
    - Microstructure regime analysis
    - Dynamic strategy selection
    - Adaptive parameter optimization
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(parameters=parameters, enable_adaptation=True)
        
        # Initialize regime detectors
        self.macro_regime_detector = MarketRegimeDetector()
        self.micro_regime_detector = MicrostructureRegimeDetector()
        
        # Strategy pool with regime preferences
        self.strategy_pool = {
            'trending_up': ['momentum', 'breakout', 'trend_following'],
            'trending_down': ['momentum', 'breakout', 'trend_following'],
            'sideways': ['mean_reversion', 'range_trading', 'pairs'],
            'volatile': ['volatility_trading', 'adaptive_range']
        }
        
        # Microstructure strategy preferences
        self.microstructure_preferences = {
            'trending_liquid': ['momentum', 'trend_following'],
            'trending_illiquid': ['adaptive_momentum', 'small_position'],
            'mean_reverting_liquid': ['mean_reversion', 'pairs'],
            'mean_reverting_illiquid': ['conservative_mean_reversion'],
            'volatile_thin': ['volatility_harvesting', 'small_positions'],
            'stable_liquid': ['arbitrage', 'momentum'],
            'mixed': ['adaptive', 'diversified']
        }
        
        # Strategy weights and performance tracking
        self.strategy_weights = {}
        self.strategy_performance = {}
        
        # Ensemble configuration
        self.ensemble_config = {
            'max_active_strategies': parameters.get('max_active_strategies', 3),
            'weight_decay': parameters.get('weight_decay', 0.95),
            'performance_window': parameters.get('performance_window', 120),
            'regime_stability_threshold': parameters.get('regime_stability_threshold', 10),
            'confidence_threshold': parameters.get('confidence_threshold', 0.6)
        }
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive indicators for ensemble decision making"""
        df = df.copy()
        
        # Basic price and volume features
        df['returns'] = np.log(df['trade_price'] / df['trade_price'].shift(1))
        df['volume_ma'] = df['candle_acc_trade_volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['candle_acc_trade_volume'] / df['volume_ma']
        
        # Volatility measures
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        df['volatility_60'] = df['returns'].rolling(window=60).std()
        
        # Trend strength
        df['trend_strength'] = self._calculate_trend_strength(df)
        
        # Market regime indicators
        df['regime'] = None
        df['micro_regime'] = None
        df['regime_confidence'] = 0.0
        
        return df
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength using multiple indicators"""
        # ADX-like calculation
        high_diff = df['high_price'].diff()
        low_diff = df['low_price'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
        
        tr = np.maximum(df['high_price'] - df['low_price'],
                      np.maximum(abs(df['high_price'] - df['trade_price'].shift(1)),
                               abs(df['low_price'] - df['trade_price'].shift(1))))
        
        plus_di = pd.Series(plus_dm).rolling(window=14).mean() / pd.Series(tr).rolling(window=14).mean() * 100
        minus_di = pd.Series(minus_dm).rolling(window=14).mean() / pd.Series(tr).rolling(window=14).mean() * 100
        
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        trend_strength = dx.rolling(window=14).mean()
        
        return trend_strength
    
    def detect_combined_regime(self, df: pd.DataFrame, current_idx: int) -> Tuple[str, str, float]:
        """Detect both macro and micro regimes with confidence"""
        lookback = 60
        
        if current_idx < lookback:
            return 'unknown', 'unknown', 0.0
        
        # Get recent data
        recent_data = df.iloc[max(0, current_idx - lookback):current_idx + 1].copy()
        
        # Detect macro regime
        macro_regime_result, macro_indicators = self.macro_regime_detector.detect_regime(recent_data)
        macro_regime = macro_regime_result.value
        
        # Detect microstructure regime  
        micro_regime = self.micro_regime_detector.detect_microstructure_regime(recent_data, lookback)
        
        # Calculate regime confidence
        confidence = self._calculate_regime_confidence(recent_data, macro_regime, micro_regime)
        
        return macro_regime, micro_regime, confidence
    
    def _calculate_regime_confidence(self, df: pd.DataFrame, macro_regime: str, micro_regime: str) -> float:
        """Calculate confidence in regime detection"""
        # Simple confidence based on indicator consistency
        trend_strength = df['trend_strength'].tail(10).mean() if 'trend_strength' in df.columns else 0
        volatility_consistency = 1 - (df['volatility_20'].tail(10).std() / df['volatility_20'].tail(10).mean())
        
        base_confidence = (trend_strength / 100 + max(0, volatility_consistency)) / 2
        
        # Adjust based on regime clarity
        if macro_regime in ['trending_up', 'trending_down'] and 'trending' in micro_regime:
            return min(1.0, base_confidence + 0.2)
        elif macro_regime == 'sideways' and 'mean_reverting' in micro_regime:
            return min(1.0, base_confidence + 0.2)
        else:
            return base_confidence
    
    def select_active_strategies(self, macro_regime: str, micro_regime: str, confidence: float) -> List[str]:
        """Select active strategies based on detected regimes"""
        # Get strategies preferred for macro regime
        macro_strategies = self.strategy_pool.get(macro_regime, ['adaptive'])
        
        # Get strategies preferred for microstructure regime
        micro_strategies = self.microstructure_preferences.get(micro_regime, ['adaptive'])
        
        # Combine and rank strategies
        strategy_scores = {}
        
        for strategy in set(macro_strategies + micro_strategies):
            score = 0.0
            
            # Score based on macro regime match
            if strategy in macro_strategies:
                score += 0.6 * confidence
            
            # Score based on micro regime match
            if strategy in micro_strategies:
                score += 0.4 * confidence
            
            # Add historical performance (if available)
            if strategy in self.strategy_performance:
                recent_perf = self.strategy_performance[strategy][-10:]  # Last 10 periods
                if recent_perf:
                    avg_performance = np.mean(recent_perf)
                    score += 0.3 * avg_performance
            
            strategy_scores[strategy] = score
        
        # Select top strategies
        sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
        max_strategies = self.ensemble_config['max_active_strategies']
        
        selected = [strategy for strategy, score in sorted_strategies[:max_strategies] 
                   if score > 0.1]  # Minimum score threshold
        
        return selected if selected else ['adaptive']  # Fallback
    
    def calculate_strategy_weights(self, selected_strategies: List[str], confidence: float) -> Dict[str, float]:
        """Calculate weights for selected strategies"""
        if not selected_strategies:
            return {}
        
        weights = {}
        total_weight = 0.0
        
        for strategy in selected_strategies:
            # Base weight
            weight = 1.0 / len(selected_strategies)
            
            # Adjust based on recent performance
            if strategy in self.strategy_performance:
                recent_perf = self.strategy_performance[strategy][-5:]
                if recent_perf:
                    perf_multiplier = 1.0 + np.mean(recent_perf)
                    weight *= max(0.1, perf_multiplier)  # Minimum 10% of base weight
            
            # Adjust based on confidence
            weight *= (0.5 + 0.5 * confidence)  # Scale by confidence
            
            weights[strategy] = weight
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def generate_ensemble_signal(self, strategy_weights: Dict[str, float], 
                                df: pd.DataFrame, current_idx: int) -> int:
        """Generate ensemble signal from weighted strategy signals"""
        if not strategy_weights or current_idx >= len(df):
            return 0
        
        weighted_signal = 0.0
        
        # For now, use simplified signal generation
        # In practice, this would call actual strategy implementations
        current_row = df.iloc[current_idx]
        
        for strategy_name, weight in strategy_weights.items():
            # Simplified strategy signal generation
            strategy_signal = self._get_strategy_signal(strategy_name, current_row, df.iloc[:current_idx+1])
            weighted_signal += strategy_signal * weight
        
        # Convert to discrete signal
        threshold = self.ensemble_config['confidence_threshold']
        
        if weighted_signal > threshold:
            return 1
        elif weighted_signal < -threshold:
            return -1
        else:
            return 0
    
    def _get_strategy_signal(self, strategy_name: str, current_row: pd.Series, df: pd.DataFrame) -> float:
        """Get signal from individual strategy (simplified implementation)"""
        # This is a simplified version - in practice would use actual strategy implementations
        if len(df) < 20:
            return 0.0
        
        if strategy_name in ['momentum', 'trend_following']:
            # Momentum-based signal
            short_ma = df['trade_price'].tail(10).mean()
            long_ma = df['trade_price'].tail(20).mean()
            return 1.0 if short_ma > long_ma else -1.0
            
        elif strategy_name in ['mean_reversion']:
            # Mean reversion signal
            current_price = current_row['trade_price']
            ma = df['trade_price'].tail(20).mean()
            std = df['trade_price'].tail(20).std()
            
            if current_price < ma - std:
                return 1.0  # Buy oversold
            elif current_price > ma + std:
                return -1.0  # Sell overbought
            else:
                return 0.0
                
        elif strategy_name in ['breakout']:
            # Breakout signal
            recent_high = df['high_price'].tail(20).max()
            recent_low = df['low_price'].tail(20).min()
            current_price = current_row['trade_price']
            
            if current_price > recent_high * 1.001:  # 0.1% breakout
                return 1.0
            elif current_price < recent_low * 0.999:
                return -1.0
            else:
                return 0.0
        
        # Default adaptive signal
        return 0.0
    
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate enhanced ensemble signals"""
        df['signal'] = np.zeros(len(df), dtype=np.int8)
        df['regime'] = 'unknown'
        df['micro_regime'] = 'unknown'
        df['regime_confidence'] = 0.0
        df['active_strategies'] = ''
        
        # Process each time step
        for i in range(100, len(df)):  # Start after warmup period
            # Detect regimes
            macro_regime, micro_regime, confidence = self.detect_combined_regime(df, i)
            
            # Update regime information
            df.iloc[i, df.columns.get_loc('regime')] = macro_regime
            df.iloc[i, df.columns.get_loc('micro_regime')] = micro_regime
            df.iloc[i, df.columns.get_loc('regime_confidence')] = confidence
            
            # Select active strategies
            active_strategies = self.select_active_strategies(macro_regime, micro_regime, confidence)
            df.iloc[i, df.columns.get_loc('active_strategies')] = ','.join(active_strategies)
            
            # Calculate strategy weights
            strategy_weights = self.calculate_strategy_weights(active_strategies, confidence)
            
            # Generate ensemble signal
            ensemble_signal = self.generate_ensemble_signal(strategy_weights, df, i)
            df.iloc[i, df.columns.get_loc('signal')] = ensemble_signal
            
            # Update strategy performance tracking (simplified)
            if i > 100:
                self._update_strategy_performance(df, i, active_strategies)
        
        return df
    
    def _update_strategy_performance(self, df: pd.DataFrame, current_idx: int, active_strategies: List[str]):
        """Update strategy performance tracking"""
        if current_idx < 2:
            return
        
        # Calculate recent return
        recent_return = df['returns'].iloc[current_idx] if 'returns' in df.columns else 0
        
        # Update performance for active strategies
        for strategy in active_strategies:
            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = []
            
            # Simplified performance attribution
            self.strategy_performance[strategy].append(recent_return)
            
            # Keep only recent performance
            window = self.ensemble_config['performance_window']
            if len(self.strategy_performance[strategy]) > window:
                self.strategy_performance[strategy] = self.strategy_performance[strategy][-window:]
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return enhanced strategy information"""
        return {
            'name': 'Enhanced Ensemble with Microstructure Regimes',
            'version': '1.0',
            'description': 'Advanced ensemble combining regime detection, microstructure analysis, and adaptive optimization',
            'features': [
                'Market regime detection',
                'Microstructure regime analysis',
                'Dynamic strategy selection',
                'Adaptive parameter optimization',
                'Performance-based weighting'
            ],
            'parameters': self.parameters,
            'current_performance': {k: np.mean(v[-10:]) if v else 0 
                                  for k, v in self.strategy_performance.items()}
        }


def main():
    """Example usage"""
    print("Enhanced Ensemble Strategy loaded successfully")


if __name__ == "__main__":
    main()