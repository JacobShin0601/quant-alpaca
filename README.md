# Quant-Alpaca: Advanced Cryptocurrency Trading System

A sophisticated quantitative trading system for cryptocurrency automation, specifically designed for the Upbit exchange. Features comprehensive backtesting, strategy optimization, and advanced ensemble trading capabilities.

## 🚀 Key Features

### Enhanced Ensemble Trading System
- **Sophisticated Market Regime Detection**: Multi-criteria analysis for trending, sideways, and volatile markets
- **Real-time Performance Tracking**: Advanced strategy performance evaluation with confidence scoring
- **Dynamic Strategy Allocation**: Performance-based weight adjustment with correlation penalties
- **Multi-factor Position Sizing**: Combines regime confidence, volatility, and Kelly Criterion
- **Microstructure Regime Analysis**: Advanced market microstructure detection for optimal strategy selection

### Comprehensive Strategy Library
- **23+ Trading Strategies**: From basic momentum to advanced multi-timeframe strategies
- **High-Frequency Capabilities**: 1-minute optimized strategies with microstructure features  
- **Adaptive Framework**: All strategies support rolling parameter optimization
- **Multi-timeframe Analysis**: Coordinated analysis across multiple timeframes

### Advanced Backtesting & Optimization
- **Realistic Cost Modeling**: Includes slippage, fees, and market impact
- **Optuna-based Optimization**: Hyperparameter tuning with parallel processing
- **Regime-specific Analysis**: Performance evaluation by market conditions
- **Risk Management**: VaR-based controls with dynamic position sizing

## 📋 Available Strategies

### Core Strategies (16)
- `basic_momentum` - RSI and moving average momentum
- `vwap` / `advanced_vwap` - Volume Weighted Average Price strategies
- `bollinger_bands` - Bollinger Bands mean reversion
- `mean_reversion` - Z-score based mean reversion
- `macd` - MACD crossover strategy
- `stochastic` - Stochastic oscillator strategy
- `pairs` - Statistical arbitrage pairs trading
- `ichimoku` - Ichimoku Cloud strategy
- `supertrend` - SuperTrend indicator strategy
- `atr_breakout` - ATR-based breakout strategy
- `keltner_channels` - Keltner Channels strategy
- `donchian_channels` - Donchian Channels breakout
- `volume_profile` - Volume Profile analysis
- `fibonacci_retracement` - Fibonacci retracement levels
- `aroon` - Aroon oscillator strategy

### Ensemble Strategies (2)
- `ensemble` - **ENHANCED** Advanced ensemble with sophisticated regime detection and dynamic strategy allocation
- `enhanced_ensemble` - **NEW** Advanced ensemble with microstructure regimes and dual-regime detection

### Specialized Strategies (5)
- `hf_vwap` / `adaptive_hf_vwap` - High-frequency VWAP strategies
- `mt_bollinger` / `mt_macd` - Multi-timeframe strategies
- **Adaptive Framework** - Any strategy can be made adaptive

## 🛠️ Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Backtesting
```bash
# Run enhanced ensemble on Bitcoin
./run_backtesting.sh --strategy ensemble --market KRW-BTC

# Run all strategies on all markets
./run_backtesting.sh --strategy all --market all

# Use cached data for faster testing
./run_backtesting.sh --strategy ensemble --market KRW-ETH --use-cached-data
```

### Strategy Optimization
```bash
# Optimize ensemble strategy
./run_optimization.sh --strategy ensemble --market KRW-BTC

# Optimize with microstructure regimes
./run_optimization.sh --strategy enhanced_ensemble --market KRW-ETH

# High-frequency strategy optimization
./run_optimization.sh --strategy hf_vwap --market KRW-SOL
```

### Data Collection Only
```bash
# Collect data without backtesting
./run_backtesting.sh --data-only --market KRW-BTC KRW-ETH
```

## 📈 Enhanced Ensemble Details

### Standard Ensemble (`ensemble`)
The enhanced ensemble strategy provides sophisticated regime-based strategy allocation:

**Key Features:**
- Real-time performance tracking with confidence scoring
- Dynamic weight adjustment using exponential weighting and Sharpe ratio optimization
- Correlation-adjusted signal aggregation to improve diversification
- Multi-criteria regime change detection with oscillation prevention
- Multi-factor position sizing with Kelly Criterion integration
- Support for both gradual and rapid regime transitions

### Microstructure Ensemble (`enhanced_ensemble`)  
Advanced ensemble with dual-regime detection system:

**Macro Regimes:** trending_up, trending_down, sideways, volatile
**Microstructure Regimes:** trending_liquid, trending_illiquid, mean_reverting_liquid, mean_reverting_illiquid, volatile_thin, stable_liquid, mixed

**Advanced Features:**
- Volume clustering and acceleration analysis
- Price momentum persistence calculation
- Spread proxy estimation and volatility analysis
- Dynamic strategy selection based on dual-regime analysis

## 📊 Supported Markets

- KRW-BTC (Bitcoin)
- KRW-ETH (Ethereum)  
- KRW-XRP (Ripple)
- KRW-ADA (Cardano)
- KRW-DOT (Polkadot)
- KRW-SOL (Solana)
- And more...

## 🧪 Testing & Validation

### Enhanced Ensemble Test Suite
```bash
python3 test_enhanced_ensemble.py
```

**Test Coverage:**
- Performance tracker validation with multiple trade scenarios
- Regime transition testing with different confidence levels
- Benchmark comparison between basic and enhanced implementations
- Signal generation and position sizing validation
- Correlation penalty and dynamic weight testing

### Adaptive Strategies Test
```bash
python3 test_adaptive_strategies.py
```

## 📂 Project Structure

```
quant-alpaca/
├── src/
│   ├── strategies/           # Trading strategy implementations
│   │   ├── ensemble.py       # Enhanced ensemble strategy
│   │   └── enhanced_ensemble.py  # Microstructure ensemble
│   ├── actions/              # Core trading actions and utilities
│   ├── backtesting/          # Advanced backtesting engine
│   └── optimization/         # Strategy optimization system
├── config/                   # Configuration files
├── results/                  # Backtest and optimization results
├── test_enhanced_ensemble.py # Comprehensive test suite
└── run_*.sh                  # Execution scripts
```

## 📋 Performance Metrics

### Enhanced Metrics System
- **누적수익률** - Cumulative return percentage
- **Sharpe Ratio** - Risk-adjusted performance measure
- **Sortino Ratio** - Downside risk-adjusted returns
- **Calmar Ratio** - Return/Max Drawdown ratio
- **MDD (Max Drawdown)** - Maximum drawdown percentage
- **승률 (Win Rate)** - Percentage of profitable trades
- **평균거래당 수익** - Average profit per trade

### Performance Rating System
⭐ Poor to ⭐⭐⭐⭐⭐ Excellent based on return, Sharpe ratio, drawdown, and trade frequency

## ⚠️ Important Considerations

### Security
- Configure API keys in environment variables only
- Never commit sensitive configuration to repository
- Use secure API key management practices

### Performance
- Optimization can be CPU-intensive (recommended: parallel processing)
- Large datasets may require significant memory
- Consider using cached data for repeated testing

### Risk Management
- Always test strategies on historical data before live trading
- Implement proper position sizing and risk controls
- Monitor regime changes and strategy performance regularly
- Regular reoptimization may be necessary for optimal performance

## 🔧 Configuration Files

- `config/config_backtesting.json` - Main backtesting settings
- `config/config_optimize.json` - Unified optimization parameters
- `config/strategies/` - Individual strategy configurations

## 📖 Documentation

- `CLAUDE.md` - Comprehensive project documentation and guidance
- `OPTIMIZATION_GUIDE.md` - Detailed optimization instructions
- `docs/` - Additional technical documentation

## 🤝 Contributing

This is a mature quantitative trading system. When contributing:
1. Follow existing code patterns and conventions
2. Test all changes thoroughly with the provided test suites
3. Ensure security best practices are maintained
4. Update documentation as needed

## 📄 License

[Add your license information here]

---

**⚠️ Disclaimer:** This software is for educational and research purposes. Cryptocurrency trading involves substantial risk. Always conduct thorough testing and risk assessment before deploying any trading system with real capital.