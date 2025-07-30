#!/bin/bash

# Backtesting script for quant-alpaca
# This script runs the backtesting engine with strategy selection and market support
# Usage: ./run_backtesting.sh [OPTIONS] [--strategy STRATEGIES...] [--market MARKETS...]

set -e  # Exit on any error

# Parse command line options
USE_CACHED_DATA=false
DATA_ONLY=false
STRATEGIES=()
MARKETS=()
CONFIG_FILE=""
OPTIMIZE_STRATEGY=""
TRAIN_RATIO=0.7

while [[ $# -gt 0 ]]; do
    case $1 in
        --use-cached-data)
            USE_CACHED_DATA=true
            shift
            ;;
        --data-only)
            DATA_ONLY=true
            shift
            ;;
        --strategy)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                STRATEGIES+=("$1")
                shift
            done
            ;;
        --market)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                MARKETS+=("$1")
                shift
            done
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --optimize-strategy)
            OPTIMIZE_STRATEGY="$2"
            shift 2
            ;;
        --ensemble)
            ENSEMBLE_TYPE="$2"
            shift 2
            ;;
        --train-ratio)
            TRAIN_RATIO="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [--strategy STRATEGIES...] [--market MARKETS...]"
            echo ""
            echo "Options:"
            echo "  --use-cached-data      Skip data collection, use existing cached data"
            echo "  --data-only           Only collect data, skip backtesting"
            echo "  --config FILE         Configuration file (default: config/config_backtesting.json)"
            echo "  --strategy STRATEGIES  Run specific strategies or 'all' for all strategies"
            echo "  --market MARKETS      Run specific markets or 'all' for all markets"
            echo "  --optimize-strategy   Optimize strategy hyperparameters ('all' or specific strategy)"
            echo "  --ensemble TYPE       Run ensemble strategy: two-step, adaptive, or hierarchical"
            echo "  --train-ratio RATIO   Train/test split ratio for optimization (default: 0.7)"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Available strategies:"
            echo "  basic_momentum, vwap, bollinger_bands, advanced_vwap"
            echo "  mean_reversion, macd, stochastic, pairs, ichimoku"
            echo "  supertrend, atr_breakout, keltner_channels, donchian_channels"
            echo "  volume_profile, fibonacci_retracement, aroon, ensemble"
            echo "  ensemble_two_step, ensemble_adaptive, ensemble_hierarchical"
            echo ""
            echo "Available markets (examples):"
            echo "  KRW-BTC, KRW-ETH, KRW-XRP, KRW-ADA, KRW-SOL, KRW-DOT"
            echo ""
            echo "Examples:"
            echo "  $0                                          # Run default strategies from config on all markets"
            echo "  $0 --use-cached-data                       # Run default strategies with cached data"
            echo "  $0 --strategy all --market all             # Run all strategies on all markets"
            echo "  $0 --strategy vwap --market KRW-BTC        # Run VWAP strategy on BTC only"
            echo "  $0 --strategy all --market KRW-ADA KRW-DOT # Run all strategies on specific markets"
            echo "  $0 --config config/strategies/vwap.json    # Use specific config file"
            echo "  $0 --optimize-strategy all --train-ratio 0.7   # Optimize all strategies"
            echo "  $0 --optimize-strategy vwap --train-ratio 0.8  # Optimize VWAP strategy only"
            echo "  $0 --ensemble two-step --market all        # Run two-step ensemble"
            echo "  $0 --ensemble adaptive --use-cached-data   # Run adaptive ensemble"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Starting Quant-Alpaca Backtesting..."
echo "===================================="
echo "Options: USE_CACHED_DATA=$USE_CACHED_DATA, DATA_ONLY=$DATA_ONLY"

# Check if optimization mode
if [ ! -z "$OPTIMIZE_STRATEGY" ]; then
    echo "Mode: OPTIMIZATION"
    echo "Optimize Strategy: $OPTIMIZE_STRATEGY"
    echo "Train Ratio: $TRAIN_RATIO"
fi

# Check if ensemble mode
if [ ! -z "$ENSEMBLE_TYPE" ]; then
    echo "Mode: ENSEMBLE"
    echo "Ensemble Type: $ENSEMBLE_TYPE"
    echo "This will use pre-optimized strategies from results/optimization/"
fi

# Set default config file if not specified
if [ -z "$CONFIG_FILE" ]; then
    CONFIG_FILE="config/config_backtesting.json"
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found at $CONFIG_FILE"
    exit 1
fi

echo "Configuration file: $CONFIG_FILE"

# Display strategy selection
if [ ${#STRATEGIES[@]} -eq 0 ]; then
    echo "Strategies: Using default strategies from config file"
elif [[ " ${STRATEGIES[@]} " =~ " all " ]]; then
    echo "Strategies: ALL available strategies will be tested"
else
    echo "Strategies: ${STRATEGIES[*]}"
fi

# Display market selection
if [ ${#MARKETS[@]} -eq 0 ]; then
    echo "Markets: Using default markets from config file"
elif [[ " ${MARKETS[@]} " =~ " all " ]]; then
    echo "Markets: ALL markets from config file will be tested"
else
    echo "Markets: ${MARKETS[*]}"
fi

# Create necessary directories
mkdir -p data
mkdir -p results

# Install required packages if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing Python dependencies..."
    python3 -m pip install -r requirements.txt
fi

# Run the backtesting
echo "Running backtesting..."

# Build command arguments
CMD_ARGS=""
if [ "$USE_CACHED_DATA" = true ]; then
    CMD_ARGS="$CMD_ARGS --use-cached-data"
fi
if [ "$DATA_ONLY" = true ]; then
    CMD_ARGS="$CMD_ARGS --data-only"
fi

# Add strategy arguments
if [ ${#STRATEGIES[@]} -gt 0 ]; then
    CMD_ARGS="$CMD_ARGS --strategy ${STRATEGIES[*]}"
fi

# Add market arguments
if [ ${#MARKETS[@]} -gt 0 ]; then
    CMD_ARGS="$CMD_ARGS --market ${MARKETS[*]}"
fi

# Add config file
CMD_ARGS="$CMD_ARGS --config $CONFIG_FILE"

# Add optimization parameters if specified
if [ ! -z "$OPTIMIZE_STRATEGY" ]; then
    CMD_ARGS="$CMD_ARGS --optimize-strategy $OPTIMIZE_STRATEGY --train-ratio $TRAIN_RATIO"
fi

# Add ensemble type if specified
if [ ! -z "$ENSEMBLE_TYPE" ]; then
    # Set strategy to the specific ensemble type
    STRATEGIES=("ensemble_$ENSEMBLE_TYPE")
    CMD_ARGS="$CMD_ARGS --strategy ensemble_$ENSEMBLE_TYPE"
fi

# Execute the backtesting
python3 src/actions/backtest_market.py $CMD_ARGS

echo ""
if [ "$DATA_ONLY" = true ]; then
    echo "Data collection completed successfully!"
    echo "Check the data/candles/ directory for database files."
else
    echo "Backtesting completed successfully!"
    echo "Check the results/ directory for output files."
fi