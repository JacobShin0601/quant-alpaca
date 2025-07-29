"""
Trading Strategies Module
"""

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
from .registry import get_strategy, STRATEGIES

# Import ensemble if available
try:
    from .ensemble import EnsembleStrategy
except ImportError:
    EnsembleStrategy = None

__all__ = [
    'BaseStrategy',
    'BasicMomentumStrategy',
    'VWAPStrategy',
    'AdvancedVWAPStrategy',
    'BollingerBandsStrategy',
    'MeanReversionStrategy',
    'MACDStrategy',
    'StochasticStrategy',
    'PairsStrategy',
    'IchimokuCloudStrategy',
    'SuperTrendStrategy',
    'ATRBreakoutStrategy',
    'KeltnerChannelsStrategy',
    'DonchianChannelsStrategy',
    'VolumeProfileStrategy',
    'FibonacciRetracementStrategy',
    'AroonStrategy',
    'EnsembleStrategy',
    'get_strategy',
    'STRATEGIES'
]