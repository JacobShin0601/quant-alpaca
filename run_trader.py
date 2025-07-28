#!/usr/bin/env python3
"""
Upbit Quantitative Trading System
Usage:
  python run_trader.py --back-testing [--use-cached-data]
  python run_trader.py --real-time (not implemented yet)
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from agents.scrapper import UpbitDataScrapper


class TradingSystem:
    def __init__(self, config_path: str = "config/config_backtesting.json"):
        self.config_path = config_path
        self.config = self._load_config()
        # Handle both old and new config formats
        if 'database_path' in self.config['data']:
            # Old format
            db_path = self.config['data']['database_path']
        else:
            # New format - use first market as example for collector initialization
            from data.collector import UpbitDataCollector
            collector = UpbitDataCollector(
                self.config['data']['database_directory'],
                self.config['data']['database_pattern']
            )
            db_path = collector.get_database_path(self.config['data']['markets'][0])
        
        self.scrapper = UpbitDataScrapper(db_path)
    
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"✓ Config loaded from: {self.config_path}")
        return config
    
    def _calculate_start_date(self) -> str:
        """설정 기반으로 시작 날짜 계산"""
        end_date = datetime.strptime(self.config['data']['end_date'], '%Y-%m-%d')
        lookback_days = self.config['data']['lookback_days']
        start_date = end_date - timedelta(days=lookback_days)
        
        return start_date.strftime('%Y-%m-%d')
    
    def check_cached_data(self) -> bool:
        """캐시된 데이터 존재 확인"""
        db_path = self.config['data']['database_path']
        
        if not os.path.exists(db_path):
            print(f"✗ Database not found: {db_path}")
            return False
        
        summary = self.scrapper.get_data_summary()
        markets = self.config['data']['markets']
        
        if summary.empty:
            print("✗ No data found in database")
            return False
        
        # 각 마켓의 데이터 확인
        missing_markets = []
        insufficient_data = []
        start_date = self._calculate_start_date()
        
        for market in markets:
            market_data = summary[summary['market'] == market]
            
            if market_data.empty:
                missing_markets.append(market)
                continue
            
            # 데이터 날짜 범위 확인
            oldest_data = market_data.iloc[0]['oldest_data']
            latest_data = market_data.iloc[0]['latest_data']
            candle_count = market_data.iloc[0]['candle_count']
            
            if oldest_data > start_date:
                insufficient_data.append({
                    'market': market,
                    'required': start_date,
                    'available': oldest_data,
                    'count': candle_count
                })
        
        if missing_markets:
            print(f"✗ Missing markets: {', '.join(missing_markets)}")
        
        if insufficient_data:
            print("✗ Insufficient data for markets:")
            for item in insufficient_data:
                print(f"  {item['market']}: need from {item['required']}, have from {item['available']} ({item['count']} candles)")
        
        if missing_markets or insufficient_data:
            return False
        
        print(f"✓ All required data available for {len(markets)} markets")
        return True
    
    def fetch_required_data(self):
        """필요한 데이터 수집"""
        markets = self.config['data']['markets']
        lookback_days = self.config['data']['lookback_days']
        
        print(f"\n=== Fetching data for {len(markets)} markets ({lookback_days} days) ===")
        
        for i, market in enumerate(markets, 1):
            print(f"\n[{i}/{len(markets)}] Fetching {market}...")
            self.scrapper.scrape_market_data(market, days=lookback_days)
        
        print("\n✓ Data collection completed")
    
    def run_backtesting(self, use_cached_data: bool = False):
        """백테스팅 실행"""
        print("="*60)
        print("🚀 UPBIT QUANTITATIVE TRADING SYSTEM")
        print("="*60)
        print(f"Mode: Back-testing")
        print(f"Markets: {', '.join(self.config['data']['markets'])}")
        print(f"Date range: {self._calculate_start_date()} to {self.config['data']['end_date']}")
        print(f"Lookback days: {self.config['data']['lookback_days']}")
        print(f"Use cached data: {use_cached_data}")
        print("="*60)
        
        if use_cached_data:
            if not self.check_cached_data():
                print("\n❌ Required cached data not available!")
                print("Run without --use-cached-data to fetch fresh data, or run data collection first.")
                return False
            print("\n✓ Using cached data")
        else:
            print("\n📥 Fetching fresh data...")
            self.fetch_required_data()
        
        # 데이터 로드 및 백테스팅 준비
        print("\n📊 Loading data for backtesting...")
        
        # 각 마켓의 데이터 로드
        market_data = {}
        start_date = self._calculate_start_date()
        end_date = self.config['data']['end_date']
        
        for market in self.config['data']['markets']:
            df = self.scrapper.get_candle_data_from_db(
                market=market,
                start_date=start_date,
                end_date=end_date
            )
            
            if not df.empty:
                market_data[market] = df
                print(f"✓ {market}: {len(df)} candles loaded")
            else:
                print(f"✗ {market}: No data available")
        
        if not market_data:
            print("❌ No market data loaded!")
            return False
        
        print(f"\n✓ Data loaded for {len(market_data)} markets")
        print("🔄 Starting backtesting simulation...")
        
        # TODO: 실제 백테스팅 로직 구현 (전략 적용, 포트폴리오 관리 등)
        print("⚠️  Backtesting engine not implemented yet")
        print("📈 Ready for strategy implementation!")
        
        return True
    
    def run_realtime(self):
        """실시간 트레이딩 실행 (미구현)"""
        print("="*60)
        print("🔥 UPBIT REAL-TIME TRADING")
        print("="*60)
        print("⚠️  Real-time trading not implemented yet")
        print("📡 This will use REST API for live 1-minute candle data")
        print("💰 This will execute actual trades on Upbit")
        print("="*60)
        
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upbit Quantitative Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_trader.py --back-testing                    # Fetch fresh data and run backtesting
  python run_trader.py --back-testing --use-cached-data  # Use existing data for backtesting
  python run_trader.py --real-time                       # Run real-time trading (not implemented)
        """
    )
    
    # 모드 선택
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--back-testing', action='store_true',
                           help='Run backtesting mode')
    mode_group.add_argument('--real-time', action='store_true',
                           help='Run real-time trading mode (not implemented)')
    
    # 백테스팅 옵션
    parser.add_argument('--use-cached-data', action='store_true',
                       help='Use cached data from database (only for back-testing)')
    
    # 설정 파일
    parser.add_argument('--config', default='config/config_backtesting.json',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        trading_system = TradingSystem(args.config)
        
        if args.back_testing:
            success = trading_system.run_backtesting(use_cached_data=args.use_cached_data)
            sys.exit(0 if success else 1)
        
        elif args.real_time:
            success = trading_system.run_realtime()
            sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Operation cancelled by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()