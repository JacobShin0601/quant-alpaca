import pandas as pd
import numpy as np
from typing import Dict, Any
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """Base class for trading strategies"""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
    
    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        pass
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Generate buy/sell signals"""
        pass
    
    def generate_signal_for_timestamp(self, df: pd.DataFrame, market: str, has_position: bool) -> int:
        """Generate signal for a specific timestamp considering current position"""
        # Default implementation - use the last row's signal
        if len(df) == 0:
            return 0
        
        # Get the last row
        last_row = df.iloc[-1]
        
        # Basic position-aware logic
        if has_position:
            # Only allow sell signals when we have a position
            return -1 if self._should_sell(last_row, df) else 0
        else:
            # Only allow buy signals when we don't have a position
            return 1 if self._should_buy(last_row, df) else 0
    
    def _should_buy(self, last_row, df) -> bool:
        """Override in subclass to implement buy logic"""
        return False
    
    def _should_sell(self, last_row, df) -> bool:
        """Override in subclass to implement sell logic"""
        return False