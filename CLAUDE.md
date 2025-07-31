# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a mature quantitative trading project named "quant-alpaca" for cryptocurrency trading automation, specifically targeting the Upbit exchange. The project features comprehensive backtesting, strategy optimization, ensemble trading, and risk management capabilities.

## Project Structure

```
quant-alpaca/
├── src/
│   ├── actions/
│   │   ├── backtest_market.py         # Main backtesting orchestrator
│   │   ├── upbit.py                   # Upbit API integration
│   │   ├── feature_engineering.py     # Feature engineering
│   │   ├── dynamic_risk_management.py # Dynamic risk management
│   │   ├── market_regime.py           # Market regime detection
│   │   ├── signal_strength.py         # Signal strength analysis
│   │   ├── var_risk_management.py     # VaR risk management
│   │   └── garch_position_sizing.py   # GARCH-based position sizing
│   ├── agents/
│   │   ├── scrapper.py                # Data scraping functionality
│   │   ├── orchestrator.py            # Agent orchestration
│   │   └── evaluator.py               # Performance evaluation
│   ├── data/
│   │   └── collector.py               # Data collection utilities
│   ├── backtesting/
│   │   └── engine.py                  # Advanced backtesting engine
│   ├── optimization/
│   │   ├── strategy_optimizer.py      # Optuna-based strategy optimization
│   │   └── ensemble_optimizer.py      # Ensemble strategy optimization
│   └── strategies/
│       ├── base.py                    # Base strategy class
│       ├── registry.py                # Strategy registry
│       ├── momentum.py                # Basic momentum strategies
│       ├── vwap.py                    # VWAP strategies (basic & advanced)
│       ├── bollinger_bands.py         # Bollinger Bands strategy
│       ├── mean_reversion.py          # Mean reversion strategy
│       ├── macd.py                    # MACD strategy
│       ├── stochastic.py              # Stochastic oscillator strategy
│       ├── pairs.py                   # Pairs trading strategy
│       ├── ichimoku.py                # Ichimoku Cloud strategy
│       ├── supertrend.py              # SuperTrend strategy
│       ├── atr_breakout.py            # ATR breakout strategy
│       ├── keltner_channels.py        # Keltner Channels strategy
│       ├── donchian_channels.py       # Donchian Channels strategy
│       ├── volume_profile.py          # Volume Profile strategy
│       ├── fibonacci_retracement.py   # Fibonacci Retracement strategy
│       ├── aroon.py                   # Aroon strategy
│       └── ensemble.py                # Ensemble strategy system
├── config/
│   ├── config_backtesting.json        # Main backtesting configuration
│   ├── config_optimize.json           # Unified optimization configuration
│   └── strategies/                    # Individual strategy configurations
├── data/
│   └── candles/                       # SQLite databases with OHLCV data
├── results/
│   └── optimization/                  # Optimization results by strategy
├── docs/                              # Documentation
├── examples/                          # Usage examples
├── run_backtesting.sh                 # Main backtesting script (separate from optimization)
├── run_optimization.sh                # Unified strategy optimization script
├── OPTIMIZATION_GUIDE.md              # Comprehensive optimization guide
├── test_adaptive_strategies.py        # Adaptive strategies test suite
└── test_enhanced_ensemble.py          # Enhanced ensemble strategy test suite
```

## Available Strategies

The project includes 23 implemented trading strategies:

### Core Strategies
1. **basic_momentum** - RSI and moving average momentum strategy
2. **vwap** - Volume Weighted Average Price strategy
3. **advanced_vwap** - Advanced VWAP with ADX filtering
4. **bollinger_bands** - Bollinger Bands mean reversion
5. **mean_reversion** - Z-score based mean reversion
6. **macd** - MACD crossover strategy
7. **stochastic** - Stochastic oscillator strategy
8. **pairs** - Statistical arbitrage pairs trading
9. **ichimoku** - Ichimoku Cloud strategy
10. **supertrend** - SuperTrend indicator strategy
11. **atr_breakout** - ATR-based breakout strategy
12. **keltner_channels** - Keltner Channels strategy
13. **donchian_channels** - Donchian Channels breakout
14. **volume_profile** - Volume Profile analysis
15. **fibonacci_retracement** - Fibonacci retracement levels
16. **aroon** - Aroon oscillator strategy

### Ensemble Strategies
17. **ensemble** - **ENHANCED** Advanced ensemble with sophisticated regime detection and dynamic strategy allocation
18. **enhanced_ensemble** - **NEW** Advanced ensemble with microstructure regimes and dual-regime detection

### High-Frequency Strategies (1-minute optimized)
19. **hf_vwap** - High-frequency VWAP with microstructure features
20. **adaptive_hf_vwap** - Self-adapting high-frequency VWAP strategy

### Multi-Timeframe Strategies
21. **mt_bollinger** - **NEW** Multi-timeframe Bollinger Bands analysis
22. **mt_macd** - **NEW** Multi-timeframe MACD with convergence detection

### Adaptive Framework
23. **Any Strategy + Adaptive** - All strategies can be made adaptive with rolling parameter optimization

## Key Features

### Backtesting Engine
- Comprehensive backtesting with realistic slippage and fees
- Market regime detection and analysis
- Risk management with stop-loss and take-profit
- Position sizing with GARCH volatility modeling
- VaR-based risk management
- Performance metrics and regime-specific analysis

### Strategy Optimization
- Optuna-based hyperparameter optimization
- Parallel optimization across multiple markets
- Train/validation split for robust parameter selection
- Fast optimization mode for quick testing
- Walk-forward analysis for robust validation
- Dynamic parameter optimization
- Comprehensive optimization results storage

### Enhanced Ensemble Trading System
- **Sophisticated Market Regime Detection**: Multi-criteria regime detection with trending_up, trending_down, sideways, volatile classifications
- **Real-time Performance Tracking**: Enhanced strategy performance tracker with confidence scoring and momentum analysis
- **Dynamic Weight Adjustment**: Performance-based strategy weight allocation using exponential weighting and Sharpe ratio optimization
- **Correlation-Adjusted Signal Aggregation**: Reduces weight of highly correlated strategies to improve diversification
- **Enhanced Regime Transition Logic**: Supports both rapid and gradual transitions with oscillation prevention
- **Multi-factor Position Sizing**: Combines regime confidence, strategy performance, volatility, and Kelly Criterion for optimal sizing
- **Microstructure Regime Detection**: Advanced detection of trending_liquid, mean_reverting_illiquid, volatile_thin market conditions
- **Adaptive Strategy Selection**: Dual-regime system (macro + microstructure) for optimal strategy selection
- **Parameter Stability Control**: Prevents excessive parameter changes through stability scoring and performance validation

### High-Frequency Trading Features
- **Microstructure Analysis**: Bid-ask spread estimation, volume clustering, liquidity scoring
- **Multi-timeframe Analysis**: 1m, 5m, 15m, 1h, 4h, 8h timeframe coordination
- **Order Flow Analysis**: Buy/sell pressure estimation, volume flow ratios
- **Market Regime Detection**: Volatility and trend regime classification
- **Dynamic Parameters**: Real-time parameter adjustment based on market conditions
- **Enhanced Cost Modeling**: Realistic slippage, market impact, and execution costs
- **Liquidity-based Execution**: Optimal timing based on liquidity conditions

### Risk Management
- Dynamic risk management based on market regimes
- Stop-loss and trailing stop-loss mechanisms
- Take-profit with partial exit capabilities
- Position sizing based on volatility (GARCH)
- VaR-based portfolio risk monitoring
- High-frequency specific risk controls

### Market Regime Detection
- Automatic detection of trending/sideways/volatile markets
- Regime-specific strategy adaptation
- Performance analysis by market regime
- Regime transition smoothing
- Volatility regime classification for HF strategies

## Development Commands

### Basic Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Run backtesting with default configuration
./run_backtesting.sh

# Run specific strategy on specific market
./run_backtesting.sh --strategy vwap --market KRW-ETH

# Use cached data (skip data collection)
./run_backtesting.sh --use-cached-data

# Run all strategies on all markets
./run_backtesting.sh --strategy all --market all
```

### Strategy Optimization (Unified System)
```bash
# Standard optimization (50 trials, ~10 minutes)
./run_optimization.sh --strategy mean_reversion --market KRW-ETH

# High-frequency strategy optimization
./run_optimization.sh --strategy hf_vwap --market KRW-ETH
./run_optimization.sh --strategy adaptive_hf_vwap --market KRW-ETH

# Standard ensemble with sophisticated regime detection
./run_optimization.sh --strategy ensemble --market KRW-BTC

# Enhanced ensemble with microstructure regimes
./run_optimization.sh --strategy enhanced_ensemble --market KRW-BTC

# Multi-timeframe strategies
./run_optimization.sh --strategy mt_bollinger --market KRW-ETH
./run_optimization.sh --strategy mt_macd --market KRW-SOL

# Custom configuration
./run_optimization.sh --strategy vwap --market KRW-BTC --config config/config_optimize.json
```

### Ensemble Trading
```bash
# Run standard enhanced ensemble
./run_backtesting.sh --strategy ensemble --market KRW-BTC

# Run ensemble with microstructure regimes
./run_backtesting.sh --strategy enhanced_ensemble --market KRW-BTC

# Test ensemble on all markets
./run_backtesting.sh --strategy ensemble --market all
```

### Configuration Files
- `config/config_backtesting.json` - Main backtesting settings
- `config/config_optimize.json` - Unified optimization parameters (includes all strategies)

### Optimization vs Backtesting Separation
- **Optimization**: `./run_optimization.sh` - Hyperparameter tuning with detailed results
- **Backtesting**: `./run_backtesting.sh` - Strategy performance testing with pre-set parameters

## Data Management

### Data Sources
- **Upbit API**: Real-time and historical cryptocurrency data
- **Supported Markets**: KRW-BTC, KRW-ETH, KRW-XRP, KRW-ADA, KRW-DOT, etc.
- **Data Storage**: SQLite databases in `data/candles/` directory
- **Timeframes**: 1-minute, 5-minute, 1-hour, 1-day candles

### Data Collection
```bash
# Collect data only (no backtesting)
./run_backtesting.sh --data-only

# Update data for specific markets
./run_backtesting.sh --data-only --market KRW-BTC KRW-ETH
```

## Results and Analysis

### Optimization Results
- Stored in `results/optimization/` by strategy and market
- Includes best parameters, performance metrics, and optimization history
- JSON format for easy analysis and parameter reuse

### Enhanced Performance Metrics
**Training Results:**
- Training optimization score

**Test Results (Out-of-Sample):**
- 누적수익금액 (P&L): Total profit/loss amount
- 최종자산가치: Final portfolio value
- 누적수익률: Cumulative return percentage
- 누적로그수익률: Cumulative log returns
- 연환산수익률: Annualized return
- 리스크조정수익: Risk-adjusted return
- Sharpe Ratio: Risk-adjusted performance measure
- Sortino Ratio: Downside risk-adjusted returns  
- Calmar Ratio: Return/Max Drawdown ratio
- MDD (Max Drawdown): Maximum drawdown percentage
- 변동성 (Volatility): Portfolio volatility
- Total Trades: Number of trades executed
- 승률 (추정): Estimated win rate
- 평균거래당 수익: Average profit per trade

**Performance Rating System:**
- ⭐ Poor to ⭐⭐⭐⭐⭐ Excellent based on return, Sharpe ratio, drawdown, and trade frequency

## Ensemble Strategy Details

### Enhanced Ensemble Features (ensemble.py)
The enhanced ensemble strategy provides sophisticated regime-based strategy allocation with the following advanced capabilities:

**Real-time Performance Tracking:**
- Enhanced strategy performance tracker with exponential weighting
- Confidence scoring based on win rate, Sharpe ratio, momentum, and consistency
- Real-time P&L tracking for active positions
- Performance-based dynamic weight adjustment using softmax normalization

**Advanced Regime Detection:**
- Multi-criteria regime change detection with confidence thresholds
- Prevention of regime oscillation through history tracking
- Support for both gradual and rapid regime transitions
- Enhanced transition logic with emergency transition handling

**Sophisticated Signal Aggregation:**
- Correlation-adjusted signal aggregation to reduce strategy correlation
- Dynamic weight calculation combining performance, confidence, and momentum
- Adaptive signal thresholds based on regime confidence and strategy consensus
- Multi-factor position sizing with Kelly Criterion integration

**Risk Management:**
- Regime-specific risk multipliers and position sizing
- Volatility-based position adjustment
- Market condition and regime confidence-based sizing
- Comprehensive stop-loss and take-profit management

### Enhanced Ensemble with Microstructure (enhanced_ensemble.py)
Advanced ensemble combining macro and microstructure regime detection:

**Dual Regime Detection:**
- Macro regimes: trending_up, trending_down, sideways, volatile
- Microstructure regimes: trending_liquid, trending_illiquid, mean_reverting_liquid, mean_reverting_illiquid, volatile_thin, stable_liquid, mixed
- Combined regime analysis for optimal strategy selection

**Microstructure Analysis:**
- Volume clustering and acceleration analysis
- Price momentum persistence calculation
- Spread proxy estimation and volatility analysis
- Price impact approximation and mean reversion tendency

**Strategy Selection:**
- Dynamic strategy pool based on regime preferences
- Performance-based strategy ranking and selection
- Adaptive weight calculation with confidence adjustment
- Maximum active strategies limit with scoring thresholds

### Testing and Validation
- Comprehensive test suite: `test_enhanced_ensemble.py`
- Performance tracker validation with multiple trade scenarios
- Regime transition testing with different confidence levels
- Benchmark comparison between basic and enhanced implementations
- Signal generation and position sizing validation

## Key Considerations

### Security
- API keys must be configured in environment variables
- Never commit sensitive configuration to repository
- Use secure API key management practices

### Performance
- Optimization can be CPU-intensive (use parallel processing)
- Large datasets may require significant memory
- Consider using cached data for repeated testing

### Risk Management
- Always test strategies on historical data before live trading
- Implement proper position sizing and risk controls
- Monitor regime changes and strategy performance
- Regular reoptimization may be necessary

### Dependencies
- Python 3.8+
- pandas, numpy for data manipulation
- optuna for optimization
- requests for API communication
- See `requirements.txt` for complete list