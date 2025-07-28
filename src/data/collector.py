import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
import requests
import time
import os


class UpbitDataCollector:
    def __init__(self, database_directory: str, database_pattern: str = "{market}_candles.db"):
        self.database_directory = database_directory
        self.database_pattern = database_pattern
        self.base_url = "https://api.upbit.com"
        os.makedirs(self.database_directory, exist_ok=True)
    
    def get_database_path(self, market: str) -> str:
        """Get database path for specific market"""
        # Replace special characters in market name for filename
        safe_market = market.replace('-', '_')
        filename = self.database_pattern.format(market=safe_market)
        return os.path.join(self.database_directory, filename)
    
    def _ensure_database(self, market: str):
        """Create database and tables if they don't exist for specific market"""
        database_path = self.get_database_path(market)
        
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        # Create candles table
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
                unit INTEGER NOT NULL,
                UNIQUE(market, candle_date_time_utc, unit)
            )
        ''')
        
        # Create metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market TEXT NOT NULL,
                collection_start_date TEXT NOT NULL,
                collection_end_date TEXT NOT NULL,
                lookback_days INTEGER NOT NULL,
                candle_unit INTEGER NOT NULL,
                total_candles INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(market, candle_unit)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def fetch_candles(self, market: str, unit: int = 1, count: int = 200, to: str = None) -> List[Dict]:
        """Fetch candle data from Upbit API"""
        url = f"{self.base_url}/v1/candles/minutes/{unit}"
        params = {
            'market': market,
            'count': count
        }
        if to:
            params['to'] = to
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {market}: {e}")
            return []
    
    def collect_historical_data(self, markets: List[str], lookback_days: int, candle_unit: int = 1):
        """Collect historical data for specified markets and days"""
        print(f"Starting data collection for {len(markets)} markets, {lookback_days} days...")
        
        for market in markets:
            print(f"Collecting data for {market}...")
            
            # Ensure database exists for this market
            self._ensure_database(market)
            database_path = self.get_database_path(market)
            conn = sqlite3.connect(database_path)
            
            # Calculate total candles needed (approximately)
            total_candles_needed = lookback_days * 24 * (60 // candle_unit)
            collected_candles = 0
            to_timestamp = None
            first_candle_date = None
            last_candle_date = None
            
            while collected_candles < total_candles_needed:
                # Upbit API limits to 200 candles per request
                count = min(200, total_candles_needed - collected_candles)
                
                candles = self.fetch_candles(market, candle_unit, count, to_timestamp)
                
                if not candles:
                    break
                
                # Track date range
                if first_candle_date is None:
                    first_candle_date = candles[0]['candle_date_time_utc']
                last_candle_date = candles[-1]['candle_date_time_utc']
                
                # Insert candles into database
                for candle in candles:
                    try:
                        conn.execute('''
                            INSERT OR REPLACE INTO candles (
                                market, candle_date_time_utc, candle_date_time_kst,
                                opening_price, high_price, low_price, trade_price,
                                timestamp, candle_acc_trade_price, candle_acc_trade_volume, unit
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            candle['market'],
                            candle['candle_date_time_utc'],
                            candle['candle_date_time_kst'],
                            candle['opening_price'],
                            candle['high_price'],
                            candle['low_price'],
                            candle['trade_price'],
                            candle['timestamp'],
                            candle['candle_acc_trade_price'],
                            candle['candle_acc_trade_volume'],
                            candle_unit
                        ))
                    except sqlite3.IntegrityError:
                        pass  # Skip duplicate entries
                
                collected_candles += len(candles)
                
                # Set the timestamp for the next batch (oldest candle's timestamp)
                if candles:
                    to_timestamp = candles[-1]['candle_date_time_utc']
                
                # Rate limiting
                time.sleep(0.1)
                
                print(f"  Collected {collected_candles}/{total_candles_needed} candles for {market}")
            
            # Store metadata
            now = datetime.now().isoformat()
            try:
                conn.execute('''
                    INSERT OR REPLACE INTO metadata (
                        market, collection_start_date, collection_end_date,
                        lookback_days, candle_unit, total_candles,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    market, last_candle_date, first_candle_date,
                    lookback_days, candle_unit, collected_candles,
                    now, now
                ))
            except sqlite3.IntegrityError:
                conn.execute('''
                    UPDATE metadata SET
                        collection_start_date = ?,
                        collection_end_date = ?,
                        lookback_days = ?,
                        total_candles = ?,
                        updated_at = ?
                    WHERE market = ? AND candle_unit = ?
                ''', (
                    last_candle_date, first_candle_date,
                    lookback_days, collected_candles, now,
                    market, candle_unit
                ))
            
            conn.commit()
            conn.close()
            
            print(f"  Completed data collection for {market}")
            print(f"    Period: {last_candle_date} to {first_candle_date}")
            print(f"    Candles: {collected_candles:,}")
            print(f"    Database: {database_path}")
        
        print("Data collection completed!")
    
    def get_data_for_backtesting(self, markets: List[str], start_date: str, end_date: str, candle_unit: int = 1) -> Dict[str, pd.DataFrame]:
        """Get data for backtesting from database"""
        data = {}
        
        for market in markets:
            database_path = self.get_database_path(market)
            
            # Check if database file exists
            if not os.path.exists(database_path):
                print(f"Warning: Database file not found for {market}: {database_path}")
                continue
            
            conn = sqlite3.connect(database_path)
            
            query = '''
                SELECT * FROM candles 
                WHERE market = ? AND unit = ? 
                AND candle_date_time_utc BETWEEN ? AND ?
                ORDER BY candle_date_time_utc ASC
            '''
            
            df = pd.read_sql_query(query, conn, params=(market, candle_unit, start_date, end_date))
            
            if not df.empty:
                df['candle_date_time_utc'] = pd.to_datetime(df['candle_date_time_utc'])
                df.set_index('candle_date_time_utc', inplace=True)
                data[market] = df
                print(f"  Loaded {len(df)} candles for {market} from {database_path}")
            else:
                print(f"  No data found for {market} in date range {start_date} to {end_date}")
            
            conn.close()
        
        return data
    
    def get_metadata(self, market: str, candle_unit: int = 1) -> Dict[str, Any]:
        """Get metadata for a specific market"""
        database_path = self.get_database_path(market)
        
        if not os.path.exists(database_path):
            return {}
        
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM metadata
            WHERE market = ? AND candle_unit = ?
            ORDER BY updated_at DESC
            LIMIT 1
        ''', (market, candle_unit))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        
        return {}
    
    def display_cached_data_info(self, markets: List[str], candle_unit: int = 1):
        """Display information about cached data"""
        print("Cached Data Information:")
        print("=" * 50)
        
        for market in markets:
            metadata = self.get_metadata(market, candle_unit)
            
            if metadata:
                print(f"\n{market}:")
                print(f"  Period: {metadata['collection_start_date']} to {metadata['collection_end_date']}")
                print(f"  Total Candles: {metadata['total_candles']:,}")
                print(f"  Lookback Days: {metadata['lookback_days']}")
                print(f"  Last Updated: {metadata['updated_at']}")
                print(f"  Database: {self.get_database_path(market)}")
            else:
                print(f"\n{market}: No cached data found")
        
        print("\n" + "=" * 50)
    
    def get_single_market_data(self, market: str, start_date: str = None, end_date: str = None, candle_unit: int = 1) -> pd.DataFrame:
        """Get data for a single market"""
        database_path = self.get_database_path(market)
        
        if not os.path.exists(database_path):
            print(f"Warning: Database file not found for {market}: {database_path}")
            return pd.DataFrame()
        
        conn = sqlite3.connect(database_path)
        
        if start_date and end_date:
            query = '''
                SELECT * FROM candles 
                WHERE market = ? AND unit = ? 
                AND candle_date_time_utc BETWEEN ? AND ?
                ORDER BY candle_date_time_utc ASC
            '''
            params = (market, candle_unit, start_date, end_date)
        else:
            query = '''
                SELECT * FROM candles 
                WHERE market = ? AND unit = ?
                ORDER BY candle_date_time_utc ASC
            '''
            params = (market, candle_unit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if not df.empty:
            df['candle_date_time_utc'] = pd.to_datetime(df['candle_date_time_utc'])
            df.set_index('candle_date_time_utc', inplace=True)
        
        return df
    
    def export_to_csv(self, market: str, output_path: str, start_date: str = None, end_date: str = None, candle_unit: int = 1):
        """Export market data to CSV file"""
        df = self.get_single_market_data(market, start_date, end_date, candle_unit)
        
        if df.empty:
            print(f"No data found for {market}")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path)
        print(f"Exported {len(df)} candles for {market} to {output_path}")
    
    def export_all_to_csv(self, markets: List[str], output_directory: str, start_date: str = None, end_date: str = None, candle_unit: int = 1):
        """Export all markets data to CSV files"""
        os.makedirs(output_directory, exist_ok=True)
        
        for market in markets:
            safe_market = market.replace('-', '_')
            filename = f"{safe_market}_data.csv"
            output_path = os.path.join(output_directory, filename)
            self.export_to_csv(market, output_path, start_date, end_date, candle_unit)
    
    def get_data_summary(self, market: str, candle_unit: int = 1) -> Dict[str, Any]:
        """Get summary statistics for market data"""
        df = self.get_single_market_data(market, candle_unit=candle_unit)
        
        if df.empty:
            return {}
        
        summary = {
            'market': market,
            'total_candles': len(df),
            'date_range': {
                'start': df.index.min().isoformat(),
                'end': df.index.max().isoformat()
            },
            'price_stats': {
                'min_price': df['trade_price'].min(),
                'max_price': df['trade_price'].max(),
                'avg_price': df['trade_price'].mean(),
                'price_change': ((df['trade_price'].iloc[-1] - df['trade_price'].iloc[0]) / df['trade_price'].iloc[0] * 100)
            },
            'volume_stats': {
                'total_volume': df['candle_acc_trade_volume'].sum(),
                'avg_volume': df['candle_acc_trade_volume'].mean()
            }
        }
        
        return summary
    
    def list_available_data(self) -> List[Dict[str, Any]]:
        """List all available data files with metadata"""
        available_data = []
        
        if not os.path.exists(self.database_directory):
            return available_data
        
        for filename in os.listdir(self.database_directory):
            if filename.endswith('.db'):
                # Extract market name from filename
                safe_market = filename.replace('_candles.db', '')
                market = safe_market.replace('_', '-')
                
                database_path = os.path.join(self.database_directory, filename)
                
                # Get basic file info
                file_stat = os.stat(database_path)
                
                # Try to get metadata
                metadata = self.get_metadata(market)
                
                data_info = {
                    'market': market,
                    'filename': filename,
                    'path': database_path,
                    'file_size_mb': round(file_stat.st_size / (1024*1024), 2),
                    'last_modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                }
                
                if metadata:
                    data_info.update({
                        'collection_period': {
                            'start': metadata['collection_start_date'],
                            'end': metadata['collection_end_date']
                        },
                        'total_candles': metadata['total_candles'],
                        'lookback_days': metadata['lookback_days'],
                        'candle_unit': metadata['candle_unit']
                    })
                
                available_data.append(data_info)
        
        return available_data