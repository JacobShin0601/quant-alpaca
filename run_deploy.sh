#!/bin/bash

# Deployment script for quant-alpaca
# This script runs the deployment agent for live trading with simulation and real trading options
# Usage: ./run_deploy.sh [OPTIONS]

set -e  # Exit on any error

# Parse command line options
SIMULATION_MODE=true
REAL_TRADING=false
CONFIG_FILE="config/config_deploy.json"
DEPLOYMENT_TIME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --simulation)
            SIMULATION_MODE=true
            REAL_TRADING=false
            shift
            ;;
        --real-time)
            SIMULATION_MODE=false
            REAL_TRADING=true
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --time)
            DEPLOYMENT_TIME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --simulation          Run in simulation mode (no real trading) (default)"
            echo "  --real-time           Run in real-time trading mode"
            echo "  --config FILE         Configuration file (default: config/config_deploy.json)"
            echo "  --time HOURS          Run the deployment for a specific time in hours"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                             # Run in simulation mode with default config"
            echo "  $0 --simulation                # Explicitly run in simulation mode"
            echo "  $0 --real-time                 # Run in real trading mode (USE WITH CAUTION)"
            echo "  $0 --config my_deploy_config.json  # Use custom config file"
            echo "  $0 --time 24                   # Run deployment for 24 hours then exit"
            echo ""
            echo "WARNING: The --real-time option will execute ACTUAL TRADES with REAL MONEY."
            echo "Ensure your API keys are configured and you understand the risks before using this option."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Starting Quant-Alpaca Deployment..."
echo "===================================="

# Display trading mode
if [ "$REAL_TRADING" = true ]; then
    echo -e "\e[31m⚠️  WARNING: REAL TRADING MODE ENABLED ⚠️\e[0m"
    echo -e "\e[31mThis will execute ACTUAL TRADES with REAL MONEY\e[0m"
    echo -e "\e[31mPress Ctrl+C within 5 seconds to abort\e[0m"
    sleep 5
    echo "Continuing with REAL TRADING mode..."
else
    echo "Mode: SIMULATION (no real trading)"
fi

# Set default config file if not specified
if [ -z "$CONFIG_FILE" ]; then
    CONFIG_FILE="config/config_deploy.json"
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

# Create necessary directories
mkdir -p data
mkdir -p results
mkdir -p logs

# Install required packages if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing Python dependencies..."
    python3 -m pip install -r requirements.txt
fi

# Check API keys
if [ "$REAL_TRADING" = true ]; then
    if [ -z "$UPBIT_ACCESS_KEY" ] || [ -z "$UPBIT_SECRET_KEY" ]; then
        echo "Error: API keys not found. Set UPBIT_ACCESS_KEY and UPBIT_SECRET_KEY environment variables."
        echo "Example: export UPBIT_ACCESS_KEY=your_key_here"
        echo "         export UPBIT_SECRET_KEY=your_secret_here"
        exit 1
    else
        echo "API keys found in environment variables"
    fi
fi

# Build command arguments
CMD_ARGS="--config $CONFIG_FILE"

if [ "$SIMULATION_MODE" = true ]; then
    CMD_ARGS="$CMD_ARGS --simulation"
fi

if [ "$REAL_TRADING" = true ]; then
    CMD_ARGS="$CMD_ARGS --real-time"
fi

# Run the deployment with optional time limit
echo "Running deployment..."
echo "Command: python3 src/agents/deployer.py $CMD_ARGS"

if [ -z "$DEPLOYMENT_TIME" ]; then
    # Run indefinitely
    python3 src/agents/deployer.py $CMD_ARGS
else
    # Run for specified time
    echo "Deployment will run for $DEPLOYMENT_TIME hours"
    
    # Use timeout command to limit execution time
    # Convert hours to seconds
    SECONDS=$((DEPLOYMENT_TIME * 3600))
    timeout $SECONDS python3 src/agents/deployer.py $CMD_ARGS
    
    echo "Deployment completed after $DEPLOYMENT_TIME hours"
fi

echo ""
echo "Deployment exited"