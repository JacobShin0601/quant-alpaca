"""
Dynamic Risk Management Module
Implements regime-based stop-loss, take-profit, and trailing stop logic
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json

try:
    from .market_regime import MarketRegime
except ImportError:
    from market_regime import MarketRegime


class OrderType(Enum):
    """Order execution types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"


@dataclass
class RiskParameters:
    """Risk management parameters for a position"""
    entry_price: float
    position_size: float
    regime: MarketRegime
    
    # Stop-loss parameters
    stop_loss_price: float
    stop_loss_pct: float
    stop_loss_order_type: OrderType
    
    # Take-profit parameters
    take_profit_price: float
    take_profit_pct: float
    take_profit_order_type: OrderType
    
    # Trailing stop parameters
    trailing_stop_enabled: bool
    trailing_stop_pct: float
    trailing_stop_activation_pct: float
    trailing_stop_order_type: OrderType
    
    # Dynamic adjustments
    high_water_mark: float
    current_trailing_stop: float
    is_trailing_active: bool
    
    # Volatility adjustments
    volatility_multiplier: float
    atr_value: float


class DynamicRiskManager:
    """
    Manages dynamic stop-loss, take-profit, and trailing stops based on market regime
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize risk manager with configuration"""
        self.config = config or self._get_default_config()
        self.positions = {}  # Track active positions
        
    def _get_default_config(self) -> Dict:
        """Default risk management configuration"""
        return {
            "regime_parameters": {
                "trending_up": {
                    "stop_loss": {
                        "base_pct": 0.02,  # 2%
                        "volatility_adjusted": True,
                        "atr_multiplier": 1.5,
                        "order_type": "stop_limit",
                        "limit_offset_pct": 0.001  # 0.1% below stop
                    },
                    "take_profit": {
                        "base_pct": 0.06,  # 6%
                        "volatility_adjusted": True,
                        "atr_multiplier": 3.0,
                        "order_type": "limit",
                        "partial_exits": [
                            {"pct": 0.03, "size": 0.3},  # 30% at 3%
                            {"pct": 0.05, "size": 0.3},  # 30% at 5%
                            {"pct": 0.08, "size": 0.4}   # 40% at 8%
                        ]
                    },
                    "trailing_stop": {
                        "enabled": True,
                        "activation_pct": 0.02,  # Activate after 2% profit
                        "trail_pct": 0.015,      # Trail by 1.5%
                        "order_type": "stop_market",
                        "tighten_on_momentum": True,
                        "momentum_trail_pct": 0.01  # Tighter 1% in strong momentum
                    }
                },
                "trending_down": {
                    "stop_loss": {
                        "base_pct": 0.015,  # 1.5% (tighter)
                        "volatility_adjusted": True,
                        "atr_multiplier": 1.0,
                        "order_type": "stop_market",  # Faster execution
                        "limit_offset_pct": 0.0
                    },
                    "take_profit": {
                        "base_pct": 0.03,  # 3% (conservative)
                        "volatility_adjusted": True,
                        "atr_multiplier": 1.5,
                        "order_type": "limit",
                        "partial_exits": [
                            {"pct": 0.015, "size": 0.5},  # 50% at 1.5%
                            {"pct": 0.025, "size": 0.3},  # 30% at 2.5%
                            {"pct": 0.04, "size": 0.2}    # 20% at 4%
                        ]
                    },
                    "trailing_stop": {
                        "enabled": True,
                        "activation_pct": 0.01,  # Quick activation at 1%
                        "trail_pct": 0.008,      # Tight 0.8% trail
                        "order_type": "stop_market",
                        "tighten_on_momentum": False
                    }
                },
                "sideways": {
                    "stop_loss": {
                        "base_pct": 0.01,  # 1% (tight for range trading)
                        "volatility_adjusted": True,
                        "atr_multiplier": 0.8,
                        "order_type": "stop_limit",
                        "limit_offset_pct": 0.0005
                    },
                    "take_profit": {
                        "base_pct": 0.02,  # 2% (range bound)
                        "volatility_adjusted": True,
                        "atr_multiplier": 1.2,
                        "order_type": "limit",
                        "partial_exits": [
                            {"pct": 0.01, "size": 0.5},   # 50% at 1%
                            {"pct": 0.015, "size": 0.3},  # 30% at 1.5%
                            {"pct": 0.025, "size": 0.2}   # 20% at 2.5%
                        ]
                    },
                    "trailing_stop": {
                        "enabled": False,  # No trailing in sideways
                        "activation_pct": 0.015,
                        "trail_pct": 0.005,
                        "order_type": "stop_limit"
                    }
                },
                "volatile": {
                    "stop_loss": {
                        "base_pct": 0.025,  # 2.5% (wider for volatility)
                        "volatility_adjusted": True,
                        "atr_multiplier": 2.0,
                        "order_type": "stop_market",  # Fast execution critical
                        "limit_offset_pct": 0.0
                    },
                    "take_profit": {
                        "base_pct": 0.04,  # 4%
                        "volatility_adjusted": True,
                        "atr_multiplier": 2.0,
                        "order_type": "limit",
                        "partial_exits": [
                            {"pct": 0.02, "size": 0.4},   # 40% at 2%
                            {"pct": 0.035, "size": 0.4},  # 40% at 3.5%
                            {"pct": 0.05, "size": 0.2}    # 20% at 5%
                        ]
                    },
                    "trailing_stop": {
                        "enabled": True,
                        "activation_pct": 0.015,
                        "trail_pct": 0.02,  # Wider trail for volatility
                        "order_type": "stop_market",
                        "tighten_on_momentum": False
                    }
                }
            },
            "order_execution": {
                "stop_loss": {
                    "market_conditions": {
                        "high_volatility": "market",     # Use market orders in high vol
                        "normal": "stop_limit",          # Use stop-limit normally
                        "low_liquidity": "stop_limit"    # Avoid market in low liquidity
                    },
                    "urgency_threshold": 0.005,  # Switch to market if price moves 0.5%
                    "retry_strategy": {
                        "max_retries": 3,
                        "retry_delay_ms": 100,
                        "price_adjustment_pct": 0.001  # Adjust by 0.1% each retry
                    }
                },
                "take_profit": {
                    "default": "limit",
                    "partial_fill_timeout": 5000,  # 5 seconds
                    "convert_to_market_after_timeout": False
                }
            },
            "advanced_features": {
                "breakeven_stop": {
                    "enabled": True,
                    "activation_pct": 0.01,  # Move stop to breakeven at 1% profit
                    "offset_pct": 0.001      # Small profit buffer
                },
                "time_based_stops": {
                    "enabled": True,
                    "tighten_after_minutes": 60,  # Tighten stops after 1 hour
                    "tightening_factor": 0.8      # Reduce stop distance by 20%
                },
                "correlation_adjustment": {
                    "enabled": True,
                    "high_correlation_threshold": 0.7,
                    "risk_reduction_factor": 0.8  # Reduce risk by 20% for correlated assets
                }
            }
        }
    
    def calculate_position_risk_parameters(self, 
                                         entry_price: float,
                                         position_size: float,
                                         regime: MarketRegime,
                                         volatility_data: Dict) -> RiskParameters:
        """Calculate risk parameters for a new position"""
        
        regime_config = self.config["regime_parameters"][regime.value]
        
        # Calculate ATR-based adjustments
        atr = volatility_data.get('atr', entry_price * 0.02)  # Default 2% if no ATR
        volatility_multiplier = volatility_data.get('volatility_ratio', 1.0)
        
        # Stop-loss calculation
        sl_config = regime_config["stop_loss"]
        sl_pct = sl_config["base_pct"]
        
        if sl_config["volatility_adjusted"]:
            sl_pct = max(sl_config["base_pct"], 
                        (atr / entry_price) * sl_config["atr_multiplier"])
        
        stop_loss_price = entry_price * (1 - sl_pct)
        
        # Take-profit calculation
        tp_config = regime_config["take_profit"]
        tp_pct = tp_config["base_pct"]
        
        if tp_config["volatility_adjusted"]:
            tp_pct = max(tp_config["base_pct"],
                        (atr / entry_price) * tp_config["atr_multiplier"])
        
        take_profit_price = entry_price * (1 + tp_pct)
        
        # Trailing stop parameters
        ts_config = regime_config["trailing_stop"]
        
        # Determine order types based on market conditions
        sl_order_type = self._determine_stop_order_type(
            sl_config["order_type"], 
            volatility_data
        )
        
        return RiskParameters(
            entry_price=entry_price,
            position_size=position_size,
            regime=regime,
            stop_loss_price=stop_loss_price,
            stop_loss_pct=sl_pct,
            stop_loss_order_type=OrderType(sl_order_type),
            take_profit_price=take_profit_price,
            take_profit_pct=tp_pct,
            take_profit_order_type=OrderType(tp_config["order_type"]),
            trailing_stop_enabled=ts_config["enabled"],
            trailing_stop_pct=ts_config["trail_pct"],
            trailing_stop_activation_pct=ts_config["activation_pct"],
            trailing_stop_order_type=OrderType(ts_config["order_type"]),
            high_water_mark=entry_price,
            current_trailing_stop=0.0,
            is_trailing_active=False,
            volatility_multiplier=volatility_multiplier,
            atr_value=atr
        )
    
    def update_position_stops(self, 
                            position_id: str,
                            current_price: float,
                            market_data: Dict) -> Dict[str, Any]:
        """Update stops for an existing position"""
        
        if position_id not in self.positions:
            return {"error": "Position not found"}
        
        risk_params = self.positions[position_id]
        updates = {}
        
        # Check if we should activate trailing stop
        if risk_params.trailing_stop_enabled and not risk_params.is_trailing_active:
            profit_pct = (current_price - risk_params.entry_price) / risk_params.entry_price
            
            if profit_pct >= risk_params.trailing_stop_activation_pct:
                risk_params.is_trailing_active = True
                risk_params.high_water_mark = current_price
                risk_params.current_trailing_stop = current_price * (1 - risk_params.trailing_stop_pct)
                
                updates["trailing_stop_activated"] = True
                updates["trailing_stop_price"] = risk_params.current_trailing_stop
        
        # Update trailing stop if active
        if risk_params.is_trailing_active:
            if current_price > risk_params.high_water_mark:
                risk_params.high_water_mark = current_price
                new_trailing_stop = current_price * (1 - risk_params.trailing_stop_pct)
                
                # Check for momentum-based tightening
                if self._should_tighten_trail(risk_params.regime, market_data):
                    regime_config = self.config["regime_parameters"][risk_params.regime.value]
                    if regime_config["trailing_stop"].get("tighten_on_momentum", False):
                        momentum_trail_pct = regime_config["trailing_stop"]["momentum_trail_pct"]
                        new_trailing_stop = current_price * (1 - momentum_trail_pct)
                
                if new_trailing_stop > risk_params.current_trailing_stop:
                    risk_params.current_trailing_stop = new_trailing_stop
                    updates["trailing_stop_updated"] = True
                    updates["new_trailing_stop"] = new_trailing_stop
        
        # Check for breakeven stop
        if self.config["advanced_features"]["breakeven_stop"]["enabled"]:
            profit_pct = (current_price - risk_params.entry_price) / risk_params.entry_price
            be_activation = self.config["advanced_features"]["breakeven_stop"]["activation_pct"]
            
            if profit_pct >= be_activation and risk_params.stop_loss_price < risk_params.entry_price:
                be_offset = self.config["advanced_features"]["breakeven_stop"]["offset_pct"]
                new_stop = risk_params.entry_price * (1 + be_offset)
                
                if new_stop > risk_params.stop_loss_price:
                    risk_params.stop_loss_price = new_stop
                    updates["breakeven_stop_set"] = True
                    updates["new_stop_loss"] = new_stop
        
        # Time-based stop tightening
        if self.config["advanced_features"]["time_based_stops"]["enabled"]:
            position_age_minutes = market_data.get("position_age_minutes", 0)
            tighten_after = self.config["advanced_features"]["time_based_stops"]["tighten_after_minutes"]
            
            if position_age_minutes > tighten_after:
                factor = self.config["advanced_features"]["time_based_stops"]["tightening_factor"]
                distance = risk_params.entry_price - risk_params.stop_loss_price
                new_distance = distance * factor
                new_stop = risk_params.entry_price - new_distance
                
                if new_stop > risk_params.stop_loss_price:
                    risk_params.stop_loss_price = new_stop
                    updates["time_based_tightening"] = True
                    updates["new_stop_loss"] = new_stop
        
        return updates
    
    def get_exit_orders(self, 
                       position_id: str,
                       current_price: float,
                       market_conditions: Dict) -> List[Dict]:
        """Generate exit orders for a position"""
        
        if position_id not in self.positions:
            return []
        
        risk_params = self.positions[position_id]
        orders = []
        
        # Determine effective stop price
        effective_stop = max(
            risk_params.stop_loss_price,
            risk_params.current_trailing_stop if risk_params.is_trailing_active else 0
        )
        
        # Stop-loss order
        if effective_stop > 0:
            sl_order = {
                "type": "stop_loss",
                "price": effective_stop,
                "size": risk_params.position_size,
                "order_type": risk_params.stop_loss_order_type.value,
                "urgency": self._calculate_stop_urgency(current_price, effective_stop)
            }
            
            # Add limit price for stop-limit orders
            if risk_params.stop_loss_order_type == OrderType.STOP_LIMIT:
                offset = self.config["regime_parameters"][risk_params.regime.value]["stop_loss"]["limit_offset_pct"]
                sl_order["limit_price"] = effective_stop * (1 - offset)
            
            orders.append(sl_order)
        
        # Take-profit orders (partial exits)
        tp_config = self.config["regime_parameters"][risk_params.regime.value]["take_profit"]
        
        if "partial_exits" in tp_config:
            remaining_size = risk_params.position_size
            
            for exit in tp_config["partial_exits"]:
                exit_price = risk_params.entry_price * (1 + exit["pct"])
                exit_size = risk_params.position_size * exit["size"]
                
                if exit_size <= remaining_size:
                    orders.append({
                        "type": "take_profit",
                        "price": exit_price,
                        "size": exit_size,
                        "order_type": risk_params.take_profit_order_type.value,
                        "partial_exit_level": exit["pct"]
                    })
                    
                    remaining_size -= exit_size
        
        return orders
    
    def _determine_stop_order_type(self, default_type: str, volatility_data: Dict) -> str:
        """Determine appropriate order type based on market conditions"""
        
        exec_config = self.config["order_execution"]["stop_loss"]
        
        # Check volatility conditions
        volatility_ratio = volatility_data.get("volatility_ratio", 1.0)
        
        if volatility_ratio > 2.0:  # High volatility
            return exec_config["market_conditions"]["high_volatility"]
        elif volatility_ratio < 0.5:  # Low volatility/liquidity concern
            return exec_config["market_conditions"]["low_liquidity"]
        else:
            return exec_config["market_conditions"]["normal"]
    
    def _should_tighten_trail(self, regime: MarketRegime, market_data: Dict) -> bool:
        """Determine if trailing stop should be tightened based on momentum"""
        
        # Check momentum indicators
        momentum = market_data.get("momentum", 0)
        rsi = market_data.get("rsi", 50)
        
        if regime == MarketRegime.TRENDING_UP:
            # Tighten on strong upward momentum
            return momentum > 0.03 and rsi > 65
        
        return False
    
    def _calculate_stop_urgency(self, current_price: float, stop_price: float) -> str:
        """Calculate urgency level for stop order execution"""
        
        distance_pct = abs(current_price - stop_price) / current_price
        urgency_threshold = self.config["order_execution"]["stop_loss"]["urgency_threshold"]
        
        if distance_pct < urgency_threshold:
            return "high"  # Switch to market order
        elif distance_pct < urgency_threshold * 2:
            return "medium"
        else:
            return "low"
    
    def add_position(self, position_id: str, risk_params: RiskParameters):
        """Add a new position to track"""
        self.positions[position_id] = risk_params
    
    def remove_position(self, position_id: str):
        """Remove a closed position"""
        if position_id in self.positions:
            del self.positions[position_id]
    
    def get_position_risk_summary(self, position_id: str) -> Dict:
        """Get current risk parameters for a position"""
        
        if position_id not in self.positions:
            return {}
        
        params = self.positions[position_id]
        
        return {
            "entry_price": params.entry_price,
            "stop_loss": params.stop_loss_price,
            "take_profit": params.take_profit_price,
            "trailing_stop": params.current_trailing_stop if params.is_trailing_active else None,
            "regime": params.regime.value,
            "is_trailing": params.is_trailing_active,
            "high_water_mark": params.high_water_mark if params.is_trailing_active else None
        }