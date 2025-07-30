"""
Strategy Registry Module
Central registry for all available trading strategies
"""

from typing import Dict, Any
from .base import BaseStrategy
from .momentum import BasicMomentumStrategy
from .vwap import VWAPStrategy, AdvancedVWAPStrategy
from .bollinger_bands import BollingerBandsStrategy
from .mean_reversion import MeanReversionStrategy
from .macd import MACDStrategy
from .stochastic import StochasticStrategy
from .pairs import PairsStrategy
from .ichimoku import IchimokuCloudStrategy
from .supertrend import SuperTrendStrategy
from .atr_breakout import ATRBreakoutStrategy
from .keltner_channels import KeltnerChannelsStrategy
from .donchian_channels import DonchianChannelsStrategy
from .volume_profile import VolumeProfileStrategy
from .fibonacci_retracement import FibonacciRetracementStrategy
from .aroon import AroonStrategy

# Import ensemble strategy if available
ensemble_available = False
try:
    from .ensemble import EnsembleStrategy
    ensemble_available = True
except ImportError as e:
    import sys
    print(f"Warning: Could not import ensemble strategy in registry: {e}", file=sys.stderr)
    pass

# Strategy registry
STRATEGIES = {
    'basic_momentum': BasicMomentumStrategy,
    'vwap': VWAPStrategy,
    'bollinger_bands': BollingerBandsStrategy,
    'advanced_vwap': AdvancedVWAPStrategy,
    'mean_reversion': MeanReversionStrategy,
    'macd': MACDStrategy,
    'stochastic': StochasticStrategy,
    'pairs': PairsStrategy,
    'ichimoku': IchimokuCloudStrategy,
    'supertrend': SuperTrendStrategy,
    'atr_breakout': ATRBreakoutStrategy,
    'keltner_channels': KeltnerChannelsStrategy,
    'donchian_channels': DonchianChannelsStrategy,
    'volume_profile': VolumeProfileStrategy,
    'fibonacci_retracement': FibonacciRetracementStrategy,
    'aroon': AroonStrategy,
}

# Add ensemble strategy if available
if ensemble_available:
    STRATEGIES['ensemble'] = EnsembleStrategy


def get_strategy(strategy_name: str, parameters: Dict[str, Any]) -> BaseStrategy:
    """Factory function to get strategy instance"""
    if strategy_name not in STRATEGIES:
        available = ', '.join(STRATEGIES.keys())
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available strategies: {available}")
    
    strategy_class = STRATEGIES[strategy_name]
    return strategy_class(parameters)