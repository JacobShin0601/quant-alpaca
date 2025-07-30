"""
VaR (Value at Risk) and CVaR (Conditional Value at Risk) Risk Management Module
Implements portfolio-level risk limits and daily loss control
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import warnings
from scipy import stats
warnings.filterwarnings('ignore')


class VaRMethod(Enum):
    """VaR calculation methods"""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"
    CORNISH_FISHER = "cornish_fisher"


@dataclass
class VaRResult:
    """VaR calculation results"""
    var_1d: float  # 1-day VaR
    cvar_1d: float  # 1-day CVaR (Expected Shortfall)
    confidence_level: float
    method: VaRMethod
    portfolio_value: float
    var_amount: float  # VaR in currency terms
    cvar_amount: float  # CVaR in currency terms
    historical_breaches: int  # Number of VaR breaches in history
    breach_rate: float  # Historical breach rate


@dataclass
class RiskLimitStatus:
    """Current risk limit status"""
    current_daily_loss: float
    var_limit: float
    cvar_limit: float
    var_utilization: float  # Current loss / VaR limit
    cvar_utilization: float  # Current loss / CVaR limit
    trading_allowed: bool
    positions_to_close: List[str]
    risk_reduction_required: float  # Percentage to reduce positions
    time_to_reset: timedelta


class VaRRiskManager:
    """
    VaR and CVaR based risk management system
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize VaR risk manager"""
        # Start with default config
        self.config = self._get_default_config()
        
        # Merge with provided config if available
        if config:
            # Handle both nested and flat config structures
            if 'var_config' in config:
                # Config already has the expected structure
                self._merge_config(self.config, config)
            else:
                # Assume the config is meant for var_config section
                self._merge_config(self.config['var_config'], config)
                
        self.daily_pnl_history = []
        self.var_history = []
        self.daily_loss_tracker = {}  # Track intraday losses
        self.risk_limits_breached = False
        self.last_reset_time = None
    
    def _merge_config(self, base: Dict, override: Dict) -> None:
        """Recursively merge override config into base config"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
        
    def _get_default_config(self) -> Dict:
        """Default VaR risk management configuration"""
        return {
            "var_config": {
                "confidence_level": 0.95,  # 95% VaR
                "lookback_days": 252,  # 1 year of history
                "method": "historical",  # historical, parametric, monte_carlo
                "update_frequency": 60,  # Update VaR every 60 minutes
                "use_ewma": True,  # Use exponentially weighted returns
                "ewma_lambda": 0.94,  # RiskMetrics lambda
                "monte_carlo_simulations": 10000,
                "use_cornish_fisher": True  # Adjust for skewness/kurtosis
            },
            "risk_limits": {
                "daily_var_multiplier": 2.0,  # Daily loss limit = 2x VaR
                "daily_cvar_multiplier": 1.5,  # Daily loss limit = 1.5x CVaR
                "intraday_var_multiplier": 1.0,  # Intraday limit = 1x VaR
                "use_dynamic_limits": True,  # Adjust limits based on regime
                "min_var_limit_pct": 0.02,  # Minimum 2% of portfolio
                "max_var_limit_pct": 0.10,  # Maximum 10% of portfolio
                "consecutive_breach_limit": 3,  # Max consecutive VaR breaches
                "breach_cooldown_hours": 24  # Cooldown after breach
            },
            "position_limits": {
                "max_position_var_contribution": 0.3,  # Max 30% of total VaR
                "concentration_penalty": True,  # Penalize concentrated positions
                "correlation_adjustment": True,  # Consider correlations
                "stress_test_multiplier": 3.0  # Stress scenario multiplier
            },
            "actions": {
                "var_breach_action": "reduce_positions",  # reduce_positions, close_all, stop_trading
                "position_reduction_pct": 0.5,  # Reduce positions by 50%
                "stop_new_trades_at_pct": 0.8,  # Stop new trades at 80% of limit
                "force_close_at_pct": 1.2,  # Force close all at 120% of limit
                "gradual_reduction": True,  # Gradually reduce vs immediate
                "prioritize_losing_positions": True  # Close losers first
            },
            "regime_adjustments": {
                "volatile_market_multiplier": 0.7,  # Reduce limits by 30%
                "trending_market_multiplier": 1.2,  # Increase limits by 20%
                "sideways_market_multiplier": 1.0,  # No change
                "crisis_mode_multiplier": 0.5  # Crisis mode: 50% reduction
            }
        }
    
    def calculate_var_cvar(self, 
                          returns: pd.Series,
                          portfolio_value: float,
                          method: Optional[VaRMethod] = None) -> VaRResult:
        """Calculate VaR and CVaR for portfolio"""
        
        config = self.config["var_config"]
        confidence_level = config["confidence_level"]
        method = method or VaRMethod(config["method"])
        
        # Clean returns
        returns_clean = returns.dropna()
        if len(returns_clean) < 30:
            # Not enough data, return conservative estimate
            return VaRResult(
                var_1d=0.05,  # 5% VaR
                cvar_1d=0.075,  # 7.5% CVaR
                confidence_level=confidence_level,
                method=method,
                portfolio_value=portfolio_value,
                var_amount=portfolio_value * 0.05,
                cvar_amount=portfolio_value * 0.075,
                historical_breaches=0,
                breach_rate=0.0
            )
        
        # Apply EWMA if configured
        if config["use_ewma"]:
            returns_weighted = self._apply_ewma_weights(returns_clean, config["ewma_lambda"])
        else:
            returns_weighted = returns_clean
        
        # Calculate VaR based on method
        if method == VaRMethod.HISTORICAL:
            var_pct, cvar_pct = self._historical_var(returns_weighted, confidence_level)
        elif method == VaRMethod.PARAMETRIC:
            var_pct, cvar_pct = self._parametric_var(returns_weighted, confidence_level)
        elif method == VaRMethod.MONTE_CARLO:
            var_pct, cvar_pct = self._monte_carlo_var(
                returns_weighted, confidence_level, config["monte_carlo_simulations"]
            )
        else:  # CORNISH_FISHER
            var_pct, cvar_pct = self._cornish_fisher_var(returns_weighted, confidence_level)
        
        # Convert to positive values (VaR is typically reported as positive)
        var_1d = abs(var_pct)
        cvar_1d = abs(cvar_pct)
        
        # Calculate historical breaches
        breaches = (returns_clean < -var_1d).sum()
        breach_rate = breaches / len(returns_clean) if len(returns_clean) > 0 else 0
        
        return VaRResult(
            var_1d=var_1d,
            cvar_1d=cvar_1d,
            confidence_level=confidence_level,
            method=method,
            portfolio_value=portfolio_value,
            var_amount=portfolio_value * var_1d,
            cvar_amount=portfolio_value * cvar_1d,
            historical_breaches=breaches,
            breach_rate=breach_rate
        )
    
    def _historical_var(self, returns: pd.Series, confidence_level: float) -> Tuple[float, float]:
        """Historical VaR calculation"""
        # Calculate percentile for VaR
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(returns, var_percentile)
        
        # CVaR is the mean of returns below VaR
        cvar = returns[returns <= var].mean()
        
        return var, cvar
    
    def _parametric_var(self, returns: pd.Series, confidence_level: float) -> Tuple[float, float]:
        """Parametric (variance-covariance) VaR calculation"""
        mean = returns.mean()
        std = returns.std()
        
        # Calculate z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # VaR calculation
        var = mean + z_score * std
        
        # CVaR for normal distribution
        phi = stats.norm.pdf(z_score)
        cvar = mean - std * phi / (1 - confidence_level)
        
        return var, cvar
    
    def _monte_carlo_var(self, returns: pd.Series, confidence_level: float, 
                        num_simulations: int) -> Tuple[float, float]:
        """Monte Carlo VaR calculation"""
        mean = returns.mean()
        std = returns.std()
        
        # Generate simulations
        simulated_returns = np.random.normal(mean, std, num_simulations)
        
        # Calculate VaR and CVaR from simulations
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(simulated_returns, var_percentile)
        cvar = simulated_returns[simulated_returns <= var].mean()
        
        return var, cvar
    
    def _cornish_fisher_var(self, returns: pd.Series, confidence_level: float) -> Tuple[float, float]:
        """Cornish-Fisher VaR (adjusting for skewness and kurtosis)"""
        mean = returns.mean()
        std = returns.std()
        skew = returns.skew()
        kurt = returns.kurtosis()
        
        # Standard z-score
        z = stats.norm.ppf(1 - confidence_level)
        
        # Cornish-Fisher expansion
        z_cf = z + (z**2 - 1) * skew / 6 + (z**3 - 3*z) * kurt / 24 - (2*z**3 - 5*z) * skew**2 / 36
        
        # VaR calculation
        var = mean + z_cf * std
        
        # Approximate CVaR (simplified)
        cvar = var * 1.4  # Rough approximation
        
        return var, cvar
    
    def _apply_ewma_weights(self, returns: pd.Series, lambda_param: float) -> pd.Series:
        """Apply exponentially weighted moving average weights"""
        weights = np.array([(1 - lambda_param) * lambda_param ** i 
                           for i in range(len(returns))])
        weights = weights[::-1]  # Reverse to give more weight to recent
        weights = weights / weights.sum()  # Normalize
        
        # Create weighted returns
        weighted_returns = returns * weights
        return weighted_returns
    
    def check_risk_limits(self,
                         current_portfolio_value: float,
                         daily_pnl: float,
                         positions: Dict[str, float],
                         var_result: VaRResult,
                         current_time: datetime) -> RiskLimitStatus:
        """Check if risk limits are breached and determine actions"""
        
        limits_config = self.config["risk_limits"]
        actions_config = self.config["actions"]
        
        # Reset daily tracker if new day
        # Handle both datetime and int timestamps
        if isinstance(current_time, (int, float)):
            current_dt = pd.to_datetime(current_time, unit='s')
        else:
            current_dt = current_time
            
        should_reset = False
        if self.last_reset_time is None:
            should_reset = True
        else:
            if isinstance(self.last_reset_time, (int, float)):
                last_reset_dt = pd.to_datetime(self.last_reset_time, unit='s')
            else:
                last_reset_dt = self.last_reset_time
            
            should_reset = current_dt.date() > last_reset_dt.date()
            
        if should_reset:
            self.daily_loss_tracker = {
                'daily_loss': 0,
                'worst_loss': 0,
                'breach_count': 0,
                'trading_restricted': False,
                'start_time': current_time  # Store original format
            }
            self.last_reset_time = current_time
        
        # Update daily loss
        self.daily_loss_tracker['daily_loss'] = min(daily_pnl, 0)  # Track losses only
        self.daily_loss_tracker['worst_loss'] = min(
            self.daily_loss_tracker['worst_loss'], 
            self.daily_loss_tracker['daily_loss']
        )
        
        # Calculate limits
        var_limit = var_result.var_amount * limits_config["daily_var_multiplier"]
        cvar_limit = var_result.cvar_amount * limits_config["daily_cvar_multiplier"]
        
        # Adjust limits based on dynamic configuration
        if limits_config["use_dynamic_limits"]:
            # Apply min/max constraints
            min_limit = current_portfolio_value * limits_config["min_var_limit_pct"]
            max_limit = current_portfolio_value * limits_config["max_var_limit_pct"]
            var_limit = max(min_limit, min(var_limit, max_limit))
            cvar_limit = max(min_limit, min(cvar_limit, max_limit))
        
        # Calculate utilization
        current_loss = abs(self.daily_loss_tracker['daily_loss'])
        var_utilization = current_loss / var_limit if var_limit > 0 else 0
        cvar_utilization = current_loss / cvar_limit if cvar_limit > 0 else 0
        
        # Determine trading status and required actions
        trading_allowed = True
        positions_to_close = []
        risk_reduction_required = 0.0
        
        # Check if we should stop new trades
        if var_utilization >= actions_config["stop_new_trades_at_pct"]:
            trading_allowed = False
            self.daily_loss_tracker['trading_restricted'] = True
        
        # Check if we need to reduce positions
        if var_utilization >= 1.0 or cvar_utilization >= 1.0:
            self.daily_loss_tracker['breach_count'] += 1
            
            # Calculate reduction required
            if var_utilization >= actions_config["force_close_at_pct"]:
                # Force close all positions
                risk_reduction_required = 1.0
                positions_to_close = list(positions.keys())
            else:
                # Partial reduction
                risk_reduction_required = actions_config["position_reduction_pct"]
                
                # Prioritize positions to close
                if actions_config["prioritize_losing_positions"]:
                    # Sort positions by P&L (would need position P&L data)
                    positions_to_close = self._prioritize_positions_to_close(positions)
                else:
                    positions_to_close = list(positions.keys())
        
        # Calculate time to reset
        # Handle both datetime and int timestamps
        if isinstance(current_time, (int, float)):
            current_dt = pd.to_datetime(current_time, unit='s')
        else:
            current_dt = current_time
            
        start_time = self.daily_loss_tracker['start_time']
        if isinstance(start_time, (int, float)):
            start_dt = pd.to_datetime(start_time, unit='s')
        else:
            start_dt = start_time
            
        time_since_start = current_dt - start_dt
        time_to_reset = timedelta(days=1) - time_since_start
        
        return RiskLimitStatus(
            current_daily_loss=self.daily_loss_tracker['daily_loss'],
            var_limit=var_limit,
            cvar_limit=cvar_limit,
            var_utilization=var_utilization,
            cvar_utilization=cvar_utilization,
            trading_allowed=trading_allowed,
            positions_to_close=positions_to_close[:int(len(positions) * risk_reduction_required)],
            risk_reduction_required=risk_reduction_required,
            time_to_reset=time_to_reset
        )
    
    def calculate_position_var_contribution(self,
                                          position_returns: Dict[str, pd.Series],
                                          position_values: Dict[str, float],
                                          total_portfolio_value: float) -> Dict[str, float]:
        """Calculate each position's contribution to portfolio VaR"""
        
        if not position_returns:
            return {}
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(position_returns)
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Calculate individual VaRs
        position_vars = {}
        confidence_level = self.config["var_config"]["confidence_level"]
        
        for position, returns in position_returns.items():
            if position in position_values:
                var_result = self.calculate_var_cvar(
                    returns, 
                    position_values[position],
                    VaRMethod.HISTORICAL
                )
                position_vars[position] = var_result.var_amount
        
        # Calculate marginal VaR contribution considering correlations
        var_contributions = {}
        total_var = sum(position_vars.values())
        
        for position in position_vars:
            if total_var > 0:
                # Simple approximation - can be enhanced with proper marginal VaR
                base_contribution = position_vars[position] / total_var
                
                # Adjust for correlations
                if len(position_returns) > 1:
                    avg_correlation = corr_matrix[position].mean()
                    correlation_adjustment = 0.5 + 0.5 * avg_correlation
                    var_contributions[position] = base_contribution * correlation_adjustment
                else:
                    var_contributions[position] = base_contribution
        
        return var_contributions
    
    def get_regime_adjusted_limits(self, 
                                  base_var: float,
                                  market_regime: str) -> Tuple[float, float]:
        """Adjust VaR limits based on market regime"""
        
        regime_config = self.config["regime_adjustments"]
        
        # Get multiplier based on regime
        multiplier = regime_config.get(f"{market_regime}_market_multiplier", 1.0)
        
        # Check for crisis mode (can be triggered by external events)
        if self._is_crisis_mode():
            multiplier = regime_config["crisis_mode_multiplier"]
        
        adjusted_var = base_var * multiplier
        adjusted_cvar = base_var * 1.5 * multiplier  # CVaR typically 1.5x VaR
        
        return adjusted_var, adjusted_cvar
    
    def _is_crisis_mode(self) -> bool:
        """Check if market is in crisis mode"""
        # Simplified - in production would check various crisis indicators
        recent_breaches = sum(1 for result in self.var_history[-5:] 
                             if result.breach_rate > 0.1)
        return recent_breaches >= 3
    
    def _prioritize_positions_to_close(self, positions: Dict[str, float]) -> List[str]:
        """Prioritize which positions to close first"""
        # In production, would sort by:
        # 1. Losing positions first
        # 2. Highest VaR contribution
        # 3. Lowest liquidity
        # For now, return as-is
        return list(positions.keys())
    
    def update_history(self, 
                      daily_return: float,
                      var_result: VaRResult,
                      timestamp: datetime):
        """Update historical tracking"""
        self.daily_pnl_history.append({
            'timestamp': timestamp,
            'daily_return': daily_return,
            'var_breach': daily_return < -var_result.var_1d,
            'cvar_breach': daily_return < -var_result.cvar_1d
        })
        
        self.var_history.append(var_result)
        
        # Keep only recent history
        max_history = self.config["var_config"]["lookback_days"]
        if len(self.daily_pnl_history) > max_history:
            self.daily_pnl_history = self.daily_pnl_history[-max_history:]
        if len(self.var_history) > max_history:
            self.var_history = self.var_history[-max_history:]
    
    def get_risk_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of risk metrics"""
        if not self.var_history:
            return {}
        
        recent_var = self.var_history[-1]
        
        # Calculate breach statistics
        total_days = len(self.daily_pnl_history)
        var_breaches = sum(1 for day in self.daily_pnl_history if day['var_breach'])
        cvar_breaches = sum(1 for day in self.daily_pnl_history if day['cvar_breach'])
        
        # Calculate average VaR
        avg_var = np.mean([v.var_1d for v in self.var_history])
        avg_cvar = np.mean([v.cvar_1d for v in self.var_history])
        
        return {
            'current_var_1d': recent_var.var_1d,
            'current_cvar_1d': recent_var.cvar_1d,
            'current_var_amount': recent_var.var_amount,
            'current_cvar_amount': recent_var.cvar_amount,
            'average_var': avg_var,
            'average_cvar': avg_cvar,
            'total_var_breaches': var_breaches,
            'total_cvar_breaches': cvar_breaches,
            'var_breach_rate': var_breaches / total_days if total_days > 0 else 0,
            'cvar_breach_rate': cvar_breaches / total_days if total_days > 0 else 0,
            'expected_breach_rate': 1 - recent_var.confidence_level,
            'model_accuracy': 'Good' if abs((var_breaches / total_days) - (1 - recent_var.confidence_level)) < 0.02 else 'Poor'
        }