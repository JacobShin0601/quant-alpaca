#!/bin/bash

# Backtesting script for quant-alpaca
# This script runs the backtesting engine with strategy selection support
# Usage: ./run_backtesting.sh [OPTIONS] [--strategy STRATEGIES...]

set -e  # Exit on any error

# Parse command line options
USE_CACHED_DATA=false
DATA_ONLY=false
STRATEGIES=()
CONFIG_FILE=""

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
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [--strategy STRATEGIES...]"
            echo ""
            echo "Options:"
            echo "  --use-cached-data      Skip data collection, use existing cached data"
            echo "  --data-only           Only collect data, skip backtesting"
            echo "  --config FILE         Configuration file (default: config/config_backtesting.json)"
            echo "  --strategy STRATEGIES  Run specific strategies or 'all' for all strategies"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Available strategies:"
            echo "  basic_momentum, vwap, bollinger_bands, advanced_vwap"
            echo "  mean_reversion, macd, stochastic, pairs"
            echo ""
            echo "Examples:"
            echo "  $0                                          # Run default strategies from config"
            echo "  $0 --use-cached-data                       # Run default strategies with cached data"
            echo "  $0 --strategy all                           # Run all available strategies"
            echo "  $0 --strategy vwap macd                     # Run VWAP and MACD strategies only"
            echo "  $0 --config config/strategies/vwap.json    # Use specific config file"
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

# Add config file
CMD_ARGS="$CMD_ARGS --config $CONFIG_FILE"

# Execute the backtesting
python3 src/actions/backtest.py $CMD_ARGS

echo ""
if [ "$DATA_ONLY" = true ]; then
    echo "Data collection completed successfully!"
    echo "Check the data/candles/ directory for database files."
else
    echo "Backtesting completed successfully!"
    echo "Check the results/ directory for output files."
fi