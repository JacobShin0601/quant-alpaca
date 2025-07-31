#!/bin/bash

# Strategy Optimization script for quant-alpaca
# This script runs strategy hyperparameter optimization with detailed results

set -e

# Parse command line options
STRATEGY=""
MARKET=""
CONFIG_FILE="config/config_optimize.json"
USE_CACHED_DATA=false
DATA_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --market)
            MARKET="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --use-cached-data)
            USE_CACHED_DATA=true
            shift
            ;;
        --data-only)
            DATA_ONLY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --strategy STRATEGY --market MARKET [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --strategy STRATEGY   Strategy to optimize (e.g., mean_reversion)"
            echo "  --market MARKET      Market to optimize for (e.g., KRW-ETH)"
            echo "  --config FILE        Configuration file (default: config/config_optimize.json)"
            echo "  --use-cached-data    Skip data collection, use existing cached data"
            echo "  --data-only          Only collect data, skip optimization"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Available markets:"
            echo "  KRW-BTC, KRW-ETH, KRW-SOL, KRW-ADA, KRW-DOT, KRW-XRP"
            echo ""
            echo "Examples:"
            echo "  $0 --strategy mean_reversion --market KRW-ETH"
            echo "  $0 --strategy enhanced_ensemble --market KRW-BTC --use-cached-data"
            echo "  $0 --strategy hf_vwap --market KRW-SOL --data-only"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$STRATEGY" ] || [ -z "$MARKET" ]; then
    echo "Error: Both --strategy and --market are required"
    exit 1
fi

echo "Strategy Optimization Mode"
echo "=========================="
echo "Strategy: $STRATEGY"
echo "Market: $MARKET"
echo "Config: $CONFIG_FILE"
echo "Use cached data: $USE_CACHED_DATA"
echo "Data only: $DATA_ONLY"
echo ""

# Set environment variables for better performance
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1  # Prevent thread oversubscription
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Convert hyphens to underscores in strategy name
STRATEGY_NORMALIZED=$(echo "$STRATEGY" | sed 's/-/_/g')

# Build command arguments
CMD_ARGS=""
if [ "$USE_CACHED_DATA" = true ]; then
    CMD_ARGS="$CMD_ARGS --use-cached-data"
fi
if [ "$DATA_ONLY" = true ]; then
    CMD_ARGS="$CMD_ARGS --data-only"
fi

# Run optimization
python3 src/actions/backtest_market.py \
    $CMD_ARGS \
    --optimize-strategy "$STRATEGY_NORMALIZED" \
    --market "$MARKET" \
    --config "$CONFIG_FILE"

echo ""
if [ "$DATA_ONLY" = true ]; then
    echo "Data collection completed!"
    echo "Check the data/candles/ directory for database files."
else
    echo "Optimization completed!"
    echo "Check results/optimization/ for optimized parameters and detailed results."
fi