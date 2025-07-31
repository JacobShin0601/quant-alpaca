import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import sqlite3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from actions.upbit import UpbitAPI


class UpbitDataScrapper:
    def __init__(self, db_path: str = "data/upbit_candles.db"):
        self.api = UpbitAPI(access_key="", secret_key="")
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """데이터베이스 초기화"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS candles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market TEXT NOT NULL,
                candle_date_time_utc TEXT NOT NULL,
                candle_date_time_kst TEXT NOT NULL,
                opening_price REAL NOT NULL,
                high_price REAL NOT NULL,
                low_price REAL NOT NULL,
                trade_price REAL NOT NULL,
                timestamp INTEGER NOT NULL,
                candle_acc_trade_price REAL NOT NULL,
                candle_acc_trade_volume REAL NOT NULL,
                unit INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(market, candle_date_time_utc, unit)
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_market_datetime 
            ON candles(market, candle_date_time_utc)
        ''')
        
        conn.commit()
        conn.close()
    
    def get_available_markets(self) -> List[str]:
        """거래 가능한 KRW 마켓 조회"""
        markets = self.api.get_markets()
        krw_markets = [market['market'] for market in markets 
                      if market['market'].startswith('KRW-')]
        return krw_markets
    
    def fetch_candle_data(self, market: str, count: int = 200, 
                         to_datetime: str = None) -> List[Dict]:
        """1분봉 데이터 수집"""
        try:
            candles = self.api.get_candles_minutes(
                market=market, 
                unit=1, 
                count=count, 
                to=to_datetime
            )
            return candles
        except Exception as e:
            print(f"Error fetching candle data for {market}: {e}")
            return []
    
    def save_candles_to_db(self, candles: List[Dict], market: str):
        """캔들 데이터를 데이터베이스에 저장"""
        if not candles:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for candle in candles:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO candles 
                    (market, candle_date_time_utc, candle_date_time_kst, 
                     opening_price, high_price, low_price, trade_price, 
                     timestamp, candle_acc_trade_price, candle_acc_trade_volume, unit)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    market,
                    candle['candle_date_time_utc'],
                    candle['candle_date_time_kst'],
                    candle['opening_price'],
                    candle['high_price'],
                    candle['low_price'],
                    candle['trade_price'],
                    candle['timestamp'],
                    candle['candle_acc_trade_price'],
                    candle['candle_acc_trade_volume'],
                    1  # unit value (1-minute candles)
                ))
            except sqlite3.IntegrityError:
                pass
        
        conn.commit()
        conn.close()
    
    def scrape_market_data(self, market: str, days: int = 1, 
                          request_delay: float = 0.12) -> int:
        """특정 마켓의 과거 데이터 수집 (Rate limit: 10 req/sec 대응)"""
        total_collected = 0
        current_datetime = None
        target_datetime = datetime.utcnow() - timedelta(days=days)
        request_count = 0
        start_time = time.time()
        
        print(f"Scraping {market} data for {days} days")
        print(f"Target date: {target_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        while True:
            # Rate limiting: 8 requests per second (안전 마진)
            if request_count > 0 and request_count % 8 == 0:
                elapsed = time.time() - start_time
                if elapsed < 1.0:
                    sleep_time = 1.0 - elapsed + 0.1  # 버퍼 추가
                    print(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                start_time = time.time()
            
            candles = self.fetch_candle_data(
                market=market, 
                count=200, 
                to_datetime=current_datetime
            )
            request_count += 1
            
            if not candles:
                print(f"No more candles available for {market}")
                break
            
            self.save_candles_to_db(candles, market)
            total_collected += len(candles)
            
            oldest_candle = candles[-1]
            oldest_datetime = pd.to_datetime(oldest_candle['candle_date_time_utc'])
            
            # 목표 날짜에 도달했는지 확인
            if oldest_datetime <= target_datetime:
                print(f"Reached target date: {oldest_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                break
            
            current_datetime = oldest_candle['candle_date_time_utc']
            
            # 진행상황 출력 (1000개마다)
            if total_collected % 1000 == 0:
                print(f"Collected {total_collected} candles for {market}, "
                      f"oldest: {oldest_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 요청 간 딜레이
            time.sleep(request_delay)
        
        print(f"Completed scraping {market}: {total_collected} total candles")
        return total_collected
    
    def scrape_all_markets(self, days: int = 1, limit_markets: int = None,
                          request_delay: float = 0.12):
        """모든 KRW 마켓 데이터 수집 (대용량 데이터 수집 대응)"""
        markets = self.get_available_markets()
        
        if limit_markets:
            markets = markets[:limit_markets]
        
        print(f"Starting to scrape {len(markets)} markets for {days} days")
        print(f"Estimated time: {len(markets) * days * 0.3:.1f} minutes (approximate)")
        
        start_time = time.time()
        total_candles = 0
        
        for i, market in enumerate(markets, 1):
            print(f"\n[{i}/{len(markets)}] Processing {market}")
            candles_collected = self.scrape_market_data(market, days, request_delay)
            total_candles += candles_collected
            
            # 마켓 간 딜레이 (더 안전하게)
            time.sleep(0.5)
        
        elapsed_time = time.time() - start_time
        print(f"\nScraping completed!")
        print(f"Total candles collected: {total_candles:,}")
        print(f"Total time: {elapsed_time/60:.1f} minutes")
        print(f"Average per market: {total_candles/len(markets):.0f} candles")
    
    def get_data_summary(self) -> pd.DataFrame:
        """데이터베이스 내 데이터 요약 정보"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                market,
                COUNT(*) as candle_count,
                MIN(candle_date_time_utc) as oldest_data,
                MAX(candle_date_time_utc) as latest_data
            FROM candles 
            GROUP BY market 
            ORDER BY candle_count DESC
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def get_candle_data_from_db(self, market: str, 
                               start_date: str = None, 
                               end_date: str = None,
                               limit: int = None) -> pd.DataFrame:
        """데이터베이스에서 캔들 데이터 조회"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM candles WHERE market = ?"
        params = [market]
        
        if start_date:
            query += " AND candle_date_time_utc >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND candle_date_time_utc <= ?"
            params.append(end_date)
        
        query += " ORDER BY candle_date_time_utc DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def get_latest_candles(self, market: str, count: int = 100) -> pd.DataFrame:
        """최신 캔들 데이터 조회"""
        return self.get_candle_data_from_db(market, limit=count)
    
    def get_latest_timestamp(self, market: str) -> Optional[str]:
        """특정 마켓의 최신 데이터 타임스탬프 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT MAX(candle_date_time_utc) 
            FROM candles 
            WHERE market = ?
        ''', (market,))
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result and result[0] else None
    
    def fetch_incremental_data(self, market: str, from_datetime: str = None) -> List[Dict]:
        """증분 데이터 수집 - 특정 시점 이후의 새로운 데이터만 가져오기"""
        try:
            # 최신 타임스탬프부터 현재까지의 데이터만 수집
            if not from_datetime:
                from_datetime = self.get_latest_timestamp(market)
            
            if not from_datetime:
                # 최초 수집시 - 최근 1시간 데이터만
                candles = self.api.get_candles_minutes(
                    market=market, 
                    unit=1, 
                    count=60  # 1시간분만
                )
            else:
                # 증분 수집 - 최대 200개 (약 3시간)
                candles = self.api.get_candles_minutes(
                    market=market, 
                    unit=1, 
                    count=200,
                    to=None  # 현재 시점까지
                )
                
                # 기존 데이터보다 새로운 것만 필터링
                if candles:
                    from_timestamp = pd.to_datetime(from_datetime)
                    filtered_candles = []
                    
                    for candle in candles:
                        candle_time = pd.to_datetime(candle['candle_date_time_utc'])
                        if candle_time > from_timestamp:
                            filtered_candles.append(candle)
                    
                    candles = filtered_candles
            
            return candles
            
        except Exception as e:
            print(f"Error fetching incremental data for {market}: {e}")
            return []
    
    def update_market_data_incremental(self, market: str) -> int:
        """마켓 데이터 증분 업데이트 - 새로운 데이터만 추가"""
        try:
            # 최신 타임스탬프 조회
            latest_timestamp = self.get_latest_timestamp(market)
            
            # 증분 데이터 수집
            new_candles = self.fetch_incremental_data(market, latest_timestamp)
            
            if new_candles:
                # 새 데이터 저장
                self.save_candles_to_db(new_candles, market)
                print(f"Updated {market}: {len(new_candles)} new candles added")
                return len(new_candles)
            else:
                # 새 데이터 없음
                return 0
                
        except Exception as e:
            print(f"Error updating incremental data for {market}: {e}")
            return 0
    
    def get_recent_candles_optimized(self, market: str, hours: int = 168) -> pd.DataFrame:
        """최적화된 최근 캔들 데이터 조회 - 메모리 효율적"""
        conn = sqlite3.connect(self.db_path)
        
        # 시간 기반으로 필요한 데이터만 조회
        query = '''
            SELECT * FROM candles 
            WHERE market = ? 
            AND datetime(candle_date_time_utc) >= datetime('now', '-{} hours')
            ORDER BY candle_date_time_utc DESC
        '''.format(hours)
        
        try:
            df = pd.read_sql_query(query, conn, params=[market])
            conn.close()
            return df
        except Exception as e:
            conn.close()
            print(f"Error querying recent candles for {market}: {e}")
            return pd.DataFrame()


def main():
    """메인 실행 함수 - 2달치 데이터 수집 예제"""
    scrapper = UpbitDataScrapper()
    
    print("=== Upbit 1분봉 데이터 수집기 ===")
    print("Available markets:")
    markets = scrapper.get_available_markets()
    for market in markets[:10]:
        print(f"  {market}")
    
    print(f"\nTotal KRW markets: {len(markets)}")
    
    # 2달치 비트코인 데이터 수집 예제
    print("\n=== 2달치 BTC 데이터 수집 시작 ===")
    scrapper.scrape_market_data("KRW-BTC", days=60)
    
    # 데이터 요약 출력
    print("\n=== 데이터 요약 ===")
    summary = scrapper.get_data_summary()
    print(summary)
    
    # 최신 데이터 확인
    df = scrapper.get_latest_candles("KRW-BTC", count=10)
    if not df.empty:
        print("\nLatest BTC candles:")
        print(df[['candle_date_time_kst', 'trade_price', 'candle_acc_trade_volume']])


if __name__ == "__main__":
    main()