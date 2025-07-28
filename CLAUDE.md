# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a quantitative trading project named "quant-alpaca" that appears to be in early development stages. The project is structured for cryptocurrency trading automation, specifically targeting the Upbit exchange, with data scraping capabilities.

## Project Structure

```
quant-alpaca/
├── src/
│   ├── actions/
│   │   ├── upbit.py            # Upbit API integration
│   │   ├── strategies.py       # Trading strategies
│   │   └── feature_engineering.py # Feature engineering
│   ├── agents/
│   │   └── scrapper.py         # Data scraping functionality
│   ├── data/
│   │   └── collector.py        # Data collection utilities
│   └── backtesting/
│       ├── engine.py           # Backtesting engine
│       └── run_backtest.py     # Backtesting runner
├── config/                     # Configuration files
├── data/                       # Data storage
├── results/                    # Backtesting results
└── examples/                   # Usage examples
```

## Architecture

The project follows a modular architecture with:
- **Actions**: Contains trading logic, API integrations, and feature engineering
- **Agents**: Contains data collection and scraping modules
- **Config**: Configuration management
- **Data**: Raw data storage
- **Models**: ML/trading model storage
- **Results**: Output and analysis results
- **Backup**: Data backup storage

## Development Status

This project includes fully implemented trading strategies, data collection, and backtesting systems. Key components:
- Data scraping from Upbit API
- Multiple trading strategies (VWAP, MACD, Bollinger Bands, etc.)
- Feature engineering capabilities
- Backtesting engine with risk management

## Development Commands

No specific build, test, or development commands are currently configured. This is a Python project that will likely require:
- Python environment setup
- Package dependency management (requirements.txt or pyproject.toml to be added)
- Testing framework setup
- Linting and formatting tools

## Key Considerations

- This project deals with cryptocurrency trading, which requires careful security practices
- API keys and sensitive configuration should never be committed to the repository
- Consider implementing proper logging and error handling for trading operations
- Data integrity and backup strategies are crucial for trading systems