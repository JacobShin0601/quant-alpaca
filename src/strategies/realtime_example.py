"""
Example of using real-time VWAP strategy
"""

from vwap_realtime import RealtimeVWAPStrategy
import time


def main():
    # Initialize strategy with parameters
    strategy_params = {
        'vwap_period': 20,
        'vwap_short_period': 10,
        'vwap_threshold': 0.005,
        'volume_threshold': 1.2,
        'strategy_variant': 'mean_reversion',
        'volume_period': 20
    }
    
    strategy = RealtimeVWAPStrategy(strategy_params)
    
    # Simulate real-time trading
    market = 'KRW-BTC'
    
    # In real implementation, this would be connected to exchange websocket
    while True:
        # Get new 1-minute candle from exchange
        new_candle = {
            'timestamp': time.time(),
            'high_price': 50000000,
            'low_price': 49900000,
            'trade_price': 49950000,
            'candle_acc_trade_volume': 1.5,
            'candle_acc_trade_price': 74925000
        }
        
        # Generate signal for new candle
        signal = strategy.generate_signal_realtime(market, new_candle)
        
        if signal == 1:
            print(f"BUY signal for {market}")
            # Execute buy order
        elif signal == -1:
            print(f"SELL signal for {market}")
            # Execute sell order
        else:
            print(f"HOLD for {market}")
        
        # Get current indicators
        indicators = strategy.get_current_indicators(market)
        if indicators:
            print(f"Current VWAP: {indicators['vwap']:.2f}")
            print(f"VWAP Deviation: {indicators['vwap_deviation']:.4f}")
            print(f"Volume Ratio: {indicators['volume_ratio']:.2f}")
            print(f"Data Points: {indicators['data_points']}")
        
        # Wait for next candle (in real implementation, this would be event-driven)
        time.sleep(60)


if __name__ == "__main__":
    main()