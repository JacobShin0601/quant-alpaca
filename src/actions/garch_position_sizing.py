"""
GARCH-based Position Sizing Module
Implements volatility prediction using GARCH model for dynamic position sizing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    from arch import arch_model
except ImportError:
    print("Warning: arch package not installed. Install with: pip install arch")
    arch_model = None


@dataclass
class GARCHParameters:
    """GARCH model parameters and results"""
    model_type: str = "GARCH"  # GARCH, EGARCH, TGARCH, etc.
    p: int = 1  # ARCH order
    q: int = 1  # GARCH order
    distribution: str = "normal"  # normal, t, skewt
    volatility_forecast: float = 0.0
    model_params: Dict = None
    convergence_status: bool = False
    aic: float = float('inf')
    bic: float = float('inf')


@dataclass
class PositionSizeResult:
    """Position sizing calculation results"""
    base_position: float
    volatility_adjusted_position: float
    kelly_position: float
    final_position: float
    predicted_volatility: float
    target_volatility: float
    volatility_ratio: float
    kelly_fraction: float
    position_adjustment_reason: str


class GARCHPositionSizer:
    """
    GARCH-based position sizing with inverse volatility weighting and Kelly criterion
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize GARCH position sizer"""
        self.config = config or self._get_default_config()
        self.models = {}  # Cache fitted models per market
        self.volatility_history = {}  # Track realized vs predicted volatility
        
    def _get_default_config(self) -> Dict:
        """Default GARCH and position sizing configuration"""
        return {
            "garch": {
                "model_type": "GARCH",  # GARCH, EGARCH, TGARCH, GJRGARCH
                "p": 1,  # ARCH order
                "q": 1,  # GARCH order
                "distribution": "normal",  # normal, t, skewt, ged
                "vol_targeting": False,  # Whether to use volatility targeting
                "refit_frequency": 100,  # Refit model every N periods
                "min_observations": 100,  # Minimum data points for fitting
                "forecast_horizon": 1,  # Periods ahead to forecast
                "rescale": True,  # Rescale returns
                "use_realized_vol": True,  # Incorporate realized volatility
                "realized_vol_window": 20  # Window for realized volatility
            },
            "position_sizing": {
                "base_position": 1.0,  # Base position size (can be capital fraction)
                "target_volatility": 0.02,  # 2% daily target volatility
                "max_leverage": 2.0,  # Maximum position multiplier
                "min_position": 0.1,  # Minimum position size
                "inverse_vol_weighting": {
                    "enabled": True,
                    "vol_lookback": 20,  # Historical volatility window
                    "vol_floor": 0.005,  # Minimum volatility (0.5%)
                    "vol_cap": 0.10,  # Maximum volatility (10%)
                    "smoothing_factor": 0.3  # EMA smoothing for position changes
                },
                "kelly_criterion": {
                    "enabled": True,
                    "kelly_fraction": 0.25,  # Fractional Kelly (25%)
                    "win_rate_window": 50,  # Window for calculating win rate
                    "profit_loss_ratio_window": 50,  # Window for P/L ratio
                    "min_trades": 20,  # Minimum trades for Kelly calculation
                    "max_kelly_position": 0.5,  # Maximum Kelly position
                    "use_garch_vol": True  # Incorporate GARCH volatility
                }
            },
            "risk_adjustments": {
                "volatility_regime_multipliers": {
                    "low": 1.2,  # Below 25th percentile
                    "normal": 1.0,  # 25th-75th percentile
                    "high": 0.7,  # 75th-90th percentile
                    "extreme": 0.4  # Above 90th percentile
                },
                "correlation_adjustment": {
                    "enabled": True,
                    "lookback": 60,  # Correlation calculation window
                    "high_correlation_threshold": 0.7,
                    "adjustment_factor": 0.8  # Reduce position by 20%
                },
                "drawdown_adjustment": {
                    "enabled": True,
                    "max_drawdown_threshold": 0.1,  # 10% drawdown
                    "position_reduction": 0.5  # Reduce position by 50%
                }
            }
        }
    
    def fit_garch_model(self, returns: pd.Series, market: str) -> GARCHParameters:
        """Fit GARCH model to return series"""
        
        if arch_model is None:
            raise ImportError("arch package required for GARCH modeling")
        
        config = self.config["garch"]
        
        # Prepare returns
        returns_clean = returns.dropna()
        if len(returns_clean) < config["min_observations"]:
            return GARCHParameters(convergence_status=False)
        
        # Scale returns to percentage
        if config["rescale"]:
            returns_scaled = returns_clean * 100
        else:
            returns_scaled = returns_clean
        
        # Select and fit model
        try:
            if config["model_type"] == "GARCH":
                model = arch_model(
                    returns_scaled,
                    vol='Garch',
                    p=config["p"],
                    q=config["q"],
                    dist=config["distribution"]
                )
            elif config["model_type"] == "EGARCH":
                model = arch_model(
                    returns_scaled,
                    vol='EGARCH',
                    p=config["p"],
                    q=config["q"],
                    dist=config["distribution"]
                )
            elif config["model_type"] == "TGARCH":
                model = arch_model(
                    returns_scaled,
                    vol='GARCH',
                    p=config["p"],
                    o=1,  # Threshold order
                    q=config["q"],
                    dist=config["distribution"]
                )
            else:
                model = arch_model(
                    returns_scaled,
                    vol='Garch',
                    p=1,
                    q=1,
                    dist='normal'
                )
            
            # Fit model
            result = model.fit(disp='off', show_warning=False)
            
            # Forecast volatility
            forecast = result.forecast(horizon=config["forecast_horizon"])
            predicted_vol = np.sqrt(forecast.variance.values[-1, 0])
            
            # Convert back to original scale
            if config["rescale"]:
                predicted_vol = predicted_vol / 100
            
            # Store model for this market
            self.models[market] = {
                'model': model,
                'result': result,
                'last_fit': len(returns_clean)
            }
            
            return GARCHParameters(
                model_type=config["model_type"],
                p=config["p"],
                q=config["q"],
                distribution=config["distribution"],
                volatility_forecast=predicted_vol,
                model_params=result.params.to_dict(),
                convergence_status=result.convergence_flag == 0,
                aic=result.aic,
                bic=result.bic
            )
            
        except Exception as e:
            print(f"GARCH fitting error for {market}: {e}")
            return GARCHParameters(convergence_status=False)
    
    def calculate_kelly_fraction(self, 
                               returns: pd.Series,
                               predicted_vol: float) -> Tuple[float, Dict]:
        """Calculate Kelly fraction with GARCH volatility adjustment"""
        
        kelly_config = self.config["position_sizing"]["kelly_criterion"]
        
        if not kelly_config["enabled"] or len(returns) < kelly_config["min_trades"]:
            return 0.0, {"reason": "Insufficient data or Kelly disabled"}
        
        # Calculate win rate
        win_rate_window = min(len(returns), kelly_config["win_rate_window"])
        recent_returns = returns.tail(win_rate_window)
        wins = (recent_returns > 0).sum()
        win_rate = wins / len(recent_returns)
        
        # Calculate average win/loss
        winning_returns = recent_returns[recent_returns > 0]
        losing_returns = recent_returns[recent_returns < 0]
        
        if len(winning_returns) == 0 or len(losing_returns) == 0:
            return 0.0, {"reason": "No wins or losses in window"}
        
        avg_win = winning_returns.mean()
        avg_loss = abs(losing_returns.mean())
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Standard Kelly formula: f* = (p*b - q) / b
        # where p = win rate, q = 1-p, b = win/loss ratio
        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Adjust for GARCH volatility if enabled
        if kelly_config["use_garch_vol"] and predicted_vol > 0:
            # Reduce Kelly fraction in high volatility environments
            vol_adjustment = min(1.0, self.config["position_sizing"]["target_volatility"] / predicted_vol)
            kelly_fraction *= vol_adjustment
        
        # Apply fractional Kelly
        kelly_fraction *= kelly_config["kelly_fraction"]
        
        # Cap at maximum
        kelly_fraction = min(kelly_fraction, kelly_config["max_kelly_position"])
        kelly_fraction = max(0, kelly_fraction)  # No negative positions
        
        return kelly_fraction, {
            "win_rate": win_rate,
            "win_loss_ratio": win_loss_ratio,
            "raw_kelly": kelly_fraction / kelly_config["kelly_fraction"],
            "volatility_adjustment": vol_adjustment if kelly_config["use_garch_vol"] else 1.0
        }
    
    def calculate_position_size(self,
                              market: str,
                              returns: pd.Series,
                              current_price: float,
                              market_data: Dict,
                              base_position: Optional[float] = None) -> PositionSizeResult:
        """Calculate position size using GARCH volatility and Kelly criterion"""
        
        base_pos = base_position or self.config["position_sizing"]["base_position"]
        
        # Check if we need to refit GARCH model
        if (market not in self.models or 
            len(returns) - self.models[market]['last_fit'] >= self.config["garch"]["refit_frequency"]):
            garch_params = self.fit_garch_model(returns, market)
        else:
            # Use existing model for forecast
            try:
                result = self.models[market]['result']
                forecast = result.forecast(horizon=self.config["garch"]["forecast_horizon"])
                predicted_vol = np.sqrt(forecast.variance.values[-1, 0])
                if self.config["garch"]["rescale"]:
                    predicted_vol = predicted_vol / 100
                garch_params = GARCHParameters(
                    volatility_forecast=predicted_vol,
                    convergence_status=True
                )
            except:
                garch_params = self.fit_garch_model(returns, market)
        
        # If GARCH failed, use historical volatility
        if not garch_params.convergence_status:
            lookback = self.config["position_sizing"]["inverse_vol_weighting"]["vol_lookback"]
            predicted_vol = returns.tail(lookback).std() * np.sqrt(252)
        else:
            predicted_vol = garch_params.volatility_forecast
        
        # Apply volatility floor and cap
        vol_config = self.config["position_sizing"]["inverse_vol_weighting"]
        predicted_vol = max(vol_config["vol_floor"], min(predicted_vol, vol_config["vol_cap"]))
        
        # 1. Inverse volatility weighting
        target_vol = self.config["position_sizing"]["target_volatility"]
        vol_adjusted_position = base_pos * (target_vol / predicted_vol) if predicted_vol > 0 else base_pos
        
        # 2. Kelly criterion position sizing
        kelly_fraction, kelly_info = self.calculate_kelly_fraction(returns, predicted_vol)
        kelly_position = base_pos * kelly_fraction
        
        # 3. Combine strategies (use maximum of the two as they're complementary)
        combined_position = max(vol_adjusted_position, kelly_position)
        
        # 4. Apply risk adjustments
        final_position = self._apply_risk_adjustments(
            combined_position, predicted_vol, market_data, returns
        )
        
        # 5. Apply position limits
        final_position = max(
            self.config["position_sizing"]["min_position"],
            min(final_position, base_pos * self.config["position_sizing"]["max_leverage"])
        )
        
        # 6. Apply smoothing if enabled
        if vol_config["enabled"] and vol_config["smoothing_factor"] > 0:
            if market in self.volatility_history and 'last_position' in self.volatility_history[market]:
                last_position = self.volatility_history[market]['last_position']
                smoothing = vol_config["smoothing_factor"]
                final_position = smoothing * last_position + (1 - smoothing) * final_position
        
        # Store position history
        if market not in self.volatility_history:
            self.volatility_history[market] = {}
        self.volatility_history[market]['last_position'] = final_position
        self.volatility_history[market]['predicted_vol'] = predicted_vol
        
        # Determine adjustment reason
        if final_position < combined_position * 0.9:
            reason = "Risk adjustments applied"
        elif kelly_position > vol_adjusted_position:
            reason = f"Kelly criterion ({kelly_fraction:.2%} fraction)"
        else:
            reason = f"Inverse volatility ({predicted_vol:.2%} vol)"
        
        return PositionSizeResult(
            base_position=base_pos,
            volatility_adjusted_position=vol_adjusted_position,
            kelly_position=kelly_position,
            final_position=final_position,
            predicted_volatility=predicted_vol,
            target_volatility=target_vol,
            volatility_ratio=predicted_vol / target_vol,
            kelly_fraction=kelly_fraction,
            position_adjustment_reason=reason
        )
    
    def _apply_risk_adjustments(self,
                               position: float,
                               predicted_vol: float,
                               market_data: Dict,
                               returns: pd.Series) -> float:
        """Apply various risk adjustments to position size"""
        
        adjusted_position = position
        adjustments = self.config["risk_adjustments"]
        
        # 1. Volatility regime adjustment
        vol_percentile = self._get_volatility_percentile(predicted_vol, returns)
        if vol_percentile < 0.25:
            multiplier = adjustments["volatility_regime_multipliers"]["low"]
        elif vol_percentile < 0.75:
            multiplier = adjustments["volatility_regime_multipliers"]["normal"]
        elif vol_percentile < 0.90:
            multiplier = adjustments["volatility_regime_multipliers"]["high"]
        else:
            multiplier = adjustments["volatility_regime_multipliers"]["extreme"]
        
        adjusted_position *= multiplier
        
        # 2. Correlation adjustment
        if adjustments["correlation_adjustment"]["enabled"]:
            correlation = market_data.get("market_correlation", 0)
            if abs(correlation) > adjustments["correlation_adjustment"]["high_correlation_threshold"]:
                adjusted_position *= adjustments["correlation_adjustment"]["adjustment_factor"]
        
        # 3. Drawdown adjustment
        if adjustments["drawdown_adjustment"]["enabled"]:
            current_drawdown = market_data.get("current_drawdown", 0)
            if abs(current_drawdown) > adjustments["drawdown_adjustment"]["max_drawdown_threshold"]:
                adjusted_position *= adjustments["drawdown_adjustment"]["position_reduction"]
        
        return adjusted_position
    
    def _get_volatility_percentile(self, current_vol: float, returns: pd.Series) -> float:
        """Calculate percentile of current volatility vs historical"""
        
        # Calculate rolling volatility
        window = self.config["garch"]["realized_vol_window"]
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()
        
        if len(rolling_vol) == 0:
            return 0.5  # Default to median
        
        # Calculate percentile
        return (rolling_vol < current_vol).sum() / len(rolling_vol)
    
    def get_portfolio_positions(self,
                               markets: List[str],
                               returns_dict: Dict[str, pd.Series],
                               market_data_dict: Dict[str, Dict],
                               total_capital: float) -> Dict[str, PositionSizeResult]:
        """Calculate positions for multiple markets with portfolio constraints"""
        
        positions = {}
        total_allocated = 0
        
        # Calculate raw positions
        for market in markets:
            if market in returns_dict:
                result = self.calculate_position_size(
                    market=market,
                    returns=returns_dict[market],
                    current_price=market_data_dict[market].get('price', 0),
                    market_data=market_data_dict[market],
                    base_position=total_capital / len(markets)  # Equal weight base
                )
                positions[market] = result
                total_allocated += result.final_position
        
        # Normalize if over-allocated
        if total_allocated > total_capital:
            scale_factor = total_capital / total_allocated
            for market in positions:
                positions[market].final_position *= scale_factor
        
        return positions
    
    def get_diagnostics(self, market: str) -> Dict:
        """Get GARCH model diagnostics and performance metrics"""
        
        if market not in self.models:
            return {"error": "No model fitted for this market"}
        
        model_info = self.models[market]
        result = model_info['result']
        
        diagnostics = {
            "model_type": self.config["garch"]["model_type"],
            "convergence": result.convergence_flag == 0,
            "aic": result.aic,
            "bic": result.bic,
            "log_likelihood": result.loglikelihood,
            "parameters": result.params.to_dict(),
            "volatility_persistence": result.params.get('alpha[1]', 0) + result.params.get('beta[1]', 0),
            "unconditional_volatility": np.sqrt(result.params.get('omega', 0) / 
                                               (1 - result.params.get('alpha[1]', 0) - result.params.get('beta[1]', 0)))
        }
        
        # Add forecast accuracy if we have history
        if market in self.volatility_history and 'realized_vol' in self.volatility_history[market]:
            realized = self.volatility_history[market]['realized_vol']
            predicted = self.volatility_history[market]['predicted_vol']
            diagnostics['forecast_rmse'] = np.sqrt(np.mean((realized - predicted) ** 2))
            diagnostics['forecast_mae'] = np.mean(np.abs(realized - predicted))
        
        return diagnostics