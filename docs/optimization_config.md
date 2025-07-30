# Optimization Configuration Guide

## Overview
The optimization configuration file (`config/config_optimize.json`) allows you to control all aspects of the strategy optimization process, including backtesting parameters, risk management settings, and execution options.

## Configuration Sections

### 1. Optimization Settings
```json
"optimization": {
    "n_trials": 5,              // Number of optimization trials per strategy
    "n_jobs": -1,               // Number of parallel workers (-1 = use all CPUs)
    "timeout": 3600,            // Maximum time (seconds) per strategy optimization
    "objective": "risk_adjusted_return", // Optimization objective
    "direction": "maximize",    // Optimization direction
    "sampler": "TPE",          // Optuna sampler (TPE, CMA-ES, Random)
    "pruner": "MedianPruner",  // Optuna pruner
    "train_ratio": 0.7         // Train/test data split ratio
}
```

### 2. Backtesting Configuration

#### Basic Settings
```json
"backtesting": {
    "initial_capital": 3000000,
    "commission_rate": 0.0005,
    "slippage_rate": 0.001,
    "min_data_points": 1000,
    "max_positions": 3,
    "order_type": "limit"
}
```

#### Position Sizing
```json
"position_sizing": {
    "max_position_pct": 0.2,      // Maximum position size as % of portfolio
    "use_dynamic_sizing": false,   // Enable dynamic position sizing
    "use_garch_sizing": false,     // Enable GARCH-based sizing
    "garch_config": {
        "lookback_period": 252,
        "update_frequency": 24,
        "vol_target": 0.15,
        "leverage_limit": 1.0
    }
}
```

#### Risk Management
```json
"risk_management": {
    "enabled": true,              // Enable/disable risk management
    "stop_loss": {
        "enabled": true,
        "initial_stop_pct": 0.02, // 2% initial stop loss
        "order_type": "stop_limit", // "stop_limit" or "stop_market"
        "limit_offset_pct": 0.005, // 0.5% below stop for limit orders
        "max_attempts_before_market": 3, // Convert to market after N failed attempts
        "trailing_stop": {
            "enabled": true,
            "activation_pct": 0.01, // Activate after 1% profit
            "trail_pct": 0.015,    // Trail by 1.5%
            "order_type": "stop_limit" // Order type for trailing stops
        }
    },
    "take_profit": {
        "enabled": true,
        "target_pct": 0.03,       // 3% take profit target
        "partial_exits": [         // Partial position exits
            {"pct": 0.015, "size": 0.5},  // Exit 50% at 1.5% profit
            {"pct": 0.025, "size": 0.3}   // Exit 30% at 2.5% profit
        ]
    }
}
```

#### VAR Risk Management
```json
"var_risk_management": {
    "enabled": false,             // Enable Value at Risk limits
    "confidence_level": 0.95,
    "lookback_period": 252,
    "max_var_pct": 0.05,
    "check_frequency_minutes": 60
}
```

#### Regime Analysis
```json
"regime_analysis": {
    "enabled": true,              // Enable regime-based analysis
    "analyze_performance": true,   // Analyze performance by regime
    "regime_config": {
        "lookback_period": 20,
        "volatility_threshold": 0.02,
        "trend_threshold": 0.0001,
        // ... other regime detection parameters
    }
}
```

## Usage Examples

### 1. Conservative Optimization (Risk Management Enabled)
```json
{
    "backtesting": {
        "risk_management": {
            "enabled": true,
            "stop_loss": {
                "enabled": true,
                "initial_stop_pct": 0.015
            }
        }
    }
}
```

### 2. Aggressive Optimization (No Risk Management)
```json
{
    "backtesting": {
        "risk_management": {
            "enabled": false
        },
        "position_sizing": {
            "max_position_pct": 0.33
        }
    }
}
```

### 3. GARCH-Based Position Sizing
```json
{
    "backtesting": {
        "position_sizing": {
            "use_garch_sizing": true,
            "garch_config": {
                "vol_target": 0.20
            }
        }
    }
}
```

## Stop Order Execution

The backtesting engine now simulates realistic stop order execution:

1. **Stop-Limit Orders**: 
   - Placed at the stop price with a limit offset
   - May not fill if price gaps through the limit
   - Converts to market order after `max_attempts_before_market` failed attempts

2. **Stop-Market Orders**:
   - Execute immediately when stop price is hit
   - Higher slippage (2x normal) to simulate market impact

3. **Order Type Selection**:
   - Use `stop_limit` for normal market conditions
   - Use `stop_market` for volatile conditions or trailing stops

## Tips
- Set `risk_management.enabled` to `false` for faster optimization
- Increase `n_trials` for more thorough optimization
- Adjust `commission_rate` based on your exchange fees
- Use `regime_analysis` to understand strategy performance in different market conditions
- Set `stop_loss.order_type` to `"stop_market"` for guaranteed execution
- Use `"stop_limit"` for better fill prices but risk of non-execution