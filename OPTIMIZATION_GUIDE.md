# 📊 Strategy Optimization Guide

## 🎯 Overview

The strategy optimization system provides comprehensive hyperparameter tuning for all trading strategies with detailed performance analysis and visualization.

## 🚀 Quick Start

### Basic Optimization
```bash
# Optimize a specific strategy for a market
./run_optimization.sh --strategy mean_reversion --market KRW-ETH

# With custom configuration
./run_optimization.sh --strategy vwap --market KRW-BTC --config config/config_optimize.json
```

### Available Strategies
- `basic_momentum` - Basic momentum strategy
- `vwap` - VWAP-based strategy  
- `advanced_vwap` - Advanced VWAP with ADX
- `bollinger_bands` - Bollinger Bands strategy
- `mean_reversion` - Mean reversion strategy
- `macd` - MACD strategy
- `stochastic` - Stochastic oscillator
- `pairs` - Pairs trading
- `ichimoku` - Ichimoku Cloud
- `supertrend` - SuperTrend indicator
- `atr_breakout` - ATR-based breakout
- `keltner_channels` - Keltner Channels
- `donchian_channels` - Donchian Channels
- `volume_profile` - Volume Profile
- `fibonacci_retracement` - Fibonacci retracement
- `aroon` - Aroon oscillator
- `ensemble` - Basic ensemble
- `enhanced_ensemble` - **NEW** Adaptive ensemble with microstructure regimes
- `hf_vwap` - **NEW** High-frequency VWAP
- `adaptive_hf_vwap` - **NEW** Adaptive high-frequency VWAP
- `mt_bollinger` - **NEW** Multi-timeframe Bollinger Bands
- `mt_macd` - **NEW** Multi-timeframe MACD

## 📋 Optimization Results

The optimization system provides detailed train/test performance analysis:

### Training Results
- Training Score: Optimization objective score on training data

### Test Results (Out-of-Sample)
- **누적수익금액 (P&L)**: Total profit/loss amount
- **최종자산가치**: Final portfolio value  
- **누적수익률**: Cumulative return percentage
- **누적로그수익률**: Cumulative log returns
- **연환산수익률**: Annualized return
- **리스크조정수익**: Risk-adjusted return
- **Sharpe Ratio**: Risk-adjusted performance measure
- **Sortino Ratio**: Downside risk-adjusted returns
- **Calmar Ratio**: Return/Max Drawdown ratio
- **MDD (Max Drawdown)**: Maximum drawdown percentage
- **변동성 (Volatility)**: Portfolio volatility
- **Total Trades**: Number of trades executed
- **승률 (추정)**: Estimated win rate
- **평균거래당 수익**: Average profit per trade

### Performance Rating System
- ⭐ Poor (0/4)
- ⭐⭐ Below Average (1/4)  
- ⭐⭐⭐ Average (2/4)
- ⭐⭐⭐⭐ Good (3/4)
- ⭐⭐⭐⭐⭐ Excellent (4/4)

Rating criteria:
- Total return > 5%
- Sharpe ratio > 1.0
- Max drawdown < 10%
- Sufficient trades (≥10)

## ⚙️ Configuration

Edit `config/config_optimize.json` to customize:

### Optimization Settings
```json
{
  "optimization": {
    "n_trials": 50,           // Number of optimization trials
    "n_jobs": -1,             // CPU cores (-1 = all)
    "timeout": 600,           // Timeout in seconds
    "objective": "sharpe_ratio", // Optimization objective
    "direction": "maximize",   // maximize or minimize
    "sampler": "TPE",         // TPE, CMA-ES, Random
    "pruner": "MedianPruner", // Early stopping
    "train_ratio": 0.5        // Train/test split ratio
  }
}
```

### Backtesting Settings
```json
{
  "backtesting": {
    "initial_capital": 3000000,
    "commission_rate": 0.0005,
    "slippage_rate": 0.001,
    "order_type": "limit"
  }
}
```

## 🧠 Adaptive Strategies

### Enhanced Ensemble Strategy
Advanced ensemble with market microstructure regime detection:

```json
{
  "enhanced_ensemble": {
    "max_active_strategies": 3,
    "weight_decay": 0.95,
    "performance_window": 120,
    "regime_stability_threshold": 10,
    "confidence_threshold": 0.6
  }
}
```

**Features:**
- **Market Regime Detection**: trending_up, trending_down, sideways, volatile
- **Microstructure Regimes**: trending_liquid, trending_illiquid, mean_reverting_liquid, etc.
- **Dynamic Strategy Selection**: Automatically selects optimal strategies for current market conditions
- **Performance-Based Weighting**: Weights strategies based on recent performance

### High-Frequency Strategies
Optimized for 1-minute data:

```json
{
  "hf_vwap": {
    "vwap_period": 30,
    "vwap_threshold": 0.003,
    "volume_threshold": 1.5,
    "use_liquidity_filter": true,
    "use_regime_detection": true
  }
}
```

### Multi-Timeframe Strategies
Analyze multiple timeframes simultaneously:

```json
{
  "mt_bollinger": {
    "bb_period": 20,
    "bb_std_dev": 2.0,
    "require_timeframe_alignment": true,
    "min_volume_confirmation": 1.2
  }
}
```

## 📊 Advanced Features

### Rolling Parameter Optimization
All strategies can be made adaptive with automatic parameter reoptimization:

- **Rolling Window**: 24-hour parameter optimization window
- **Reoptimization Frequency**: Every 4 hours
- **Parameter Stability**: Prevents excessive parameter changes
- **Regime-Based Adjustments**: Parameters adjust based on market regime

### Walk-Forward Analysis
```json
{
  "adaptive_optimization": {
    "enable_rolling_optimization": true,
    "adaptation_window": 1440,
    "reoptimization_frequency": 240,
    "regime_detection": true,
    "walk_forward_validation": true
  }
}
```

## 📈 Results Storage

Optimization results are saved to:
- `results/optimization/`: Detailed JSON results
- Individual files: `{strategy}_{market}_{timestamp}.json`

### Results Structure
```json
{
  "strategy": "mean_reversion",
  "market": "KRW-ETH", 
  "optimization_date": "2025-01-31T...",
  "results": {
    "best_params": {...},
    "train_performance": 1.23,
    "test_performance": {...},
    "n_trials": 50,
    "optimization_history": [...]
  }
}
```

## 🎯 Best Practices

### Strategy Selection
1. **Start Simple**: Begin with `mean_reversion` or `vwap`
2. **Market-Specific**: Different strategies work better on different markets
3. **Ensemble for Stability**: Use `enhanced_ensemble` for robust performance

### Parameter Tuning
1. **Sufficient Trials**: Use at least 50 trials for complex strategies
2. **Train/Test Split**: Use 0.7 train ratio for sufficient test data
3. **Multiple Markets**: Test on different markets for robustness

### Performance Analysis
1. **Look Beyond Returns**: Consider Sharpe ratio, max drawdown
2. **Trade Frequency**: Balance returns with trading costs
3. **Regime Analysis**: Understand when strategies work best

## 🔧 Troubleshooting

### Common Issues
1. **No trades generated**: Adjust thresholds or periods
2. **Poor Sharpe ratio**: Increase risk management parameters
3. **Overfitting**: Reduce parameter ranges or increase regularization

### Performance Tips
1. **Use cached data**: `--use-cached-data` for faster iterations
2. **Parallel optimization**: Set `n_jobs: -1` in config
3. **Early stopping**: Enable pruning for faster convergence

## 📚 Examples

### Basic Strategy Optimization
```bash
# Optimize mean reversion strategy
./run_optimization.sh --strategy mean_reversion --market KRW-ETH
```

### Advanced Ensemble
```bash
# Optimize enhanced ensemble with microstructure regimes
./run_optimization.sh --strategy enhanced_ensemble --market KRW-BTC
```

### High-Frequency Trading
```bash
# Optimize high-frequency VWAP
./run_optimization.sh --strategy hf_vwap --market KRW-ETH
```

### Multi-Timeframe Analysis
```bash
# Optimize multi-timeframe MACD
./run_optimization.sh --strategy mt_macd --market KRW-SOL
```

---

For detailed implementation information, see the source code in `src/optimization/` and `src/strategies/`.