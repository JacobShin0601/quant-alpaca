import jwt
import uuid
import hashlib
import requests
import json
from urllib.parse import urlencode
from typing import Dict, List, Any, Optional
import time


class UpbitAPI:
    def __init__(self, access_key: str, secret_key: str):
        self.access_key = access_key
        self.secret_key = secret_key
        self.base_url = "https://api.upbit.com"
    
    def _create_jwt_token(self, query_string: str = None) -> str:
        """JWT 토큰 생성"""
        payload = {
            'access_key': self.access_key,
            'nonce': str(uuid.uuid4()),
        }
        
        if query_string:
            query_hash = hashlib.sha512(query_string.encode()).hexdigest()
            payload['query_hash'] = query_hash
            payload['query_hash_alg'] = 'SHA512'
        
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None) -> Dict:
        """API 요청 실행"""
        url = f"{self.base_url}{endpoint}"
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
        }
        
        query_string = None
        if params:
            query_string = urlencode(params, doseq=True)
        
        jwt_token = self._create_jwt_token(query_string)
        headers['Authorization'] = f'Bearer {jwt_token}'
        
        if method.upper() == 'GET':
            response = requests.get(url, params=params, headers=headers)
        elif method.upper() == 'POST':
            response = requests.post(url, json=params, headers=headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    def get_accounts(self) -> List[Dict]:
        """계좌 정보 조회"""
        return self._make_request('GET', '/v1/accounts')
    
    def get_markets(self) -> List[Dict]:
        """마켓 코드 조회 (공개 API)"""
        url = f"{self.base_url}/v1/market/all"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_candles_minutes(self, market: str, unit: int = 1, count: int = 200, 
                           to: str = None) -> List[Dict]:
        """분봉 차트 조회 (공개 API)"""
        url = f"{self.base_url}/v1/candles/minutes/{unit}"
        params = {
            'market': market,
            'count': count
        }
        if to:
            params['to'] = to
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_ticker(self, markets: List[str]) -> List[Dict]:
        """현재가 조회 (공개 API)"""
        url = f"{self.base_url}/v1/ticker"
        params = {'markets': markets}
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def place_order(self, market: str, side: str, volume: str = None, 
                   price: str = None, ord_type: str = 'limit') -> Dict:
        """주문하기"""
        params = {
            'market': market,
            'side': side,
            'ord_type': ord_type
        }
        
        if volume:
            params['volume'] = volume
        if price:
            params['price'] = price
            
        return self._make_request('POST', '/v1/orders', params)
    
    def get_orders(self, market: str = None, state: str = 'wait') -> List[Dict]:
        """주문 목록 조회"""
        params = {'state': state}
        if market:
            params['market'] = market
            
        return self._make_request('GET', '/v1/orders', params)
    
    def cancel_order(self, uuid: str) -> Dict:
        """주문 취소"""
        params = {'uuid': uuid}
        return self._make_request('DELETE', '/v1/order', params)


class UpbitTrader:
    def __init__(self, access_key: str, secret_key: str):
        self.api = UpbitAPI(access_key, secret_key)
        self.markets_cache = None
    
    def get_krw_markets(self) -> List[str]:
        """KRW 마켓 코드 조회"""
        if not self.markets_cache:
            self.markets_cache = self.api.get_markets()
        
        return [market['market'] for market in self.markets_cache 
                if market['market'].startswith('KRW-')]
    
    def get_balance(self, currency: str = 'KRW') -> float:
        """계좌 잔고 조회"""
        accounts = self.api.get_accounts()
        for account in accounts:
            if account['currency'] == currency:
                return float(account['balance'])
        return 0.0
    
    def buy_market_order(self, market: str, price: str) -> Dict:
        """시장가 매수"""
        return self.api.place_order(
            market=market, 
            side='bid', 
            price=price, 
            ord_type='price'
        )
    
    def sell_market_order(self, market: str, volume: str) -> Dict:
        """시장가 매도"""
        return self.api.place_order(
            market=market, 
            side='ask', 
            volume=volume, 
            ord_type='market'
        )
    
    def buy_limit_order(self, market: str, volume: str, price: str) -> Dict:
        """지정가 매수"""
        return self.api.place_order(
            market=market, 
            side='bid', 
            volume=volume, 
            price=price, 
            ord_type='limit'
        )
    
    def sell_limit_order(self, market: str, volume: str, price: str) -> Dict:
        """지정가 매도"""
        return self.api.place_order(
            market=market, 
            side='ask', 
            volume=volume, 
            price=price, 
            ord_type='limit'
        )