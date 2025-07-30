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

# Import ensemble if available and add to STRATEGIES
try:
    from .ensemble import EnsembleStrategy
    # Manually add ensemble to STRATEGIES if not already there
    if 'ensemble' not in STRATEGIES:
        STRATEGIES['ensemble'] = EnsembleStrategy
except ImportError:
    EnsembleStrategy = None

# Import optimized ensemble if available
try:
    from .optimized_ensemble import OptimizedEnsembleStrategy
    if 'optimized_ensemble' not in STRATEGIES:
        STRATEGIES['optimized_ensemble'] = OptimizedEnsembleStrategy
except ImportError:
    OptimizedEnsembleStrategy = None

# Import new ensemble strategies
try:
    from .ensemble_two_step import TwoStepEnsembleStrategy
    if 'ensemble_two_step' not in STRATEGIES:
        STRATEGIES['ensemble_two_step'] = TwoStepEnsembleStrategy
except ImportError:
    TwoStepEnsembleStrategy = None

try:
    from .ensemble_adaptive import AdaptiveEnsembleStrategy
    if 'ensemble_adaptive' not in STRATEGIES:
        STRATEGIES['ensemble_adaptive'] = AdaptiveEnsembleStrategy
except ImportError:
    AdaptiveEnsembleStrategy = None

try:
    from .ensemble_hierarchical import HierarchicalEnsembleStrategy
    if 'ensemble_hierarchical' not in STRATEGIES:
        STRATEGIES['ensemble_hierarchical'] = HierarchicalEnsembleStrategy
except ImportError:
    HierarchicalEnsembleStrategy = None

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
    'OptimizedEnsembleStrategy',
    'get_strategy',
    'STRATEGIES'
]