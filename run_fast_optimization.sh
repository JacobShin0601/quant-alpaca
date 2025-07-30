#!/bin/bash

# Fast optimization script for quant-alpaca
# This script runs optimizations with performance improvements

set -e

# Parse command line options
STRATEGY=""
MARKET=""
CONFIG_FILE="config/config_optimize_fast.json"

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
        -h|--help)
            echo "Usage: $0 --strategy STRATEGY --market MARKET [--config CONFIG_FILE]"
            echo ""
            echo "Options:"
            echo "  --strategy STRATEGY   Strategy to optimize (e.g., mean_reversion)"
            echo "  --market MARKET      Market to optimize for (e.g., KRW-ETH)"
            echo "  --config FILE        Configuration file (default: config/config_optimize_fast.json)"
            echo "  -h, --help           Show this help message"
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

echo "Fast Optimization Mode"
echo "====================="
echo "Strategy: $STRATEGY"
echo "Market: $MARKET"
echo "Config: $CONFIG_FILE"
echo ""

# Set environment variables for better performance
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1  # Prevent thread oversubscription
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Run optimization with cached data
python3 src/actions/backtest_market.py \
    --use-cached-data \
    --optimize-strategy "$STRATEGY" \
    --market "$MARKET" \
    --config "$CONFIG_FILE"

echo ""
echo "Optimization completed!"
echo "Check results/optimization/ for optimized parameters"