import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import os
import json
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings("ignore")

# calculate_index에서 import
from .calculate_index import StrategyParams, TechnicalIndicators

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """전략 기본 클래스"""

    def __init__(self, params: StrategyParams):
        self.params = params
        self.positions = []
        self.trades = []
        # 전략 타입 구분 (단일종목 vs 포트폴리오)
        self.strategy_type = "single_asset"  # 기본값: 단일종목 전략

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """매매 신호 생성 (하위 클래스에서 구현)"""
        pass

    def calculate_position_size(self, df: pd.DataFrame, signal: int) -> float:
        """포지션 사이즈 계산"""
        # 기본 포지션 사이즈 (변동성 조정)
        base_size = 1.0

        if "volatility" in df.columns and not df["volatility"].isna().all():
            current_vol = df["volatility"].iloc[-1]
            if current_vol > 0:
                # 변동성 타겟에 맞춰 포지션 사이즈 조정
                vol_adjustment = self.params.volatility_target / current_vol
                base_size *= min(vol_adjustment, 2.0)  # 최대 2배로 제한

        return base_size * signal

    def calculate_stop_loss(
        self, df: pd.DataFrame, entry_price: float, position: int
    ) -> float:
        """손절가 계산 (ATR 기반)"""
        if "atr" in df.columns and not df["atr"].isna().all():
            atr = df["atr"].iloc[-1]
            if position > 0:  # 롱 포지션
                return entry_price - (self.params.stop_loss_atr_multiplier * atr)
            else:  # 숏 포지션
                return entry_price + (self.params.stop_loss_atr_multiplier * atr)
        return None

    def calculate_take_profit(
        self, df: pd.DataFrame, entry_price: float, position: int
    ) -> float:
        """익절가 계산 (ATR 기반)"""
        if "atr" in df.columns and not df["atr"].isna().all():
            atr = df["atr"].iloc[-1]
            if position > 0:  # 롱 포지션
                return entry_price + (self.params.take_profit_atr_multiplier * atr)
            else:  # 숏 포지션
                return entry_price - (self.params.take_profit_atr_multiplier * atr)
        return None


# ============================================================================
# 모멘텀 전략 (Momentum Strategies)
# ============================================================================


# [스윙] 1~3주 보유
class DualMomentumStrategy(BaseStrategy):
    """추세⇄평균회귀 듀얼 모멘텀 전략"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        # Donchian Channel 하이퍼파라미터
        self.donchian_period = getattr(params, "donchian_period", 20)
        # RSI 하이퍼파라미터
        self.rsi_period = getattr(params, "rsi_period", 14)
        self.rsi_oversold = getattr(params, "rsi_oversold", 30)
        self.rsi_overbought = getattr(params, "rsi_overbought", 70)
        # 모멘텀 하이퍼파라미터
        self.momentum_period = getattr(params, "momentum_period", 10)
        self.momentum_threshold = getattr(params, "momentum_threshold", 0.02)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """듀얼 모멘텀 신호 생성 - 단순화된 버전"""
        df = df.copy()
        df["signal"] = 0

        # 1. Donchian Channel 계산
        df["donchian_upper"] = df["high"].rolling(window=self.donchian_period).max()
        df["donchian_lower"] = df["low"].rolling(window=self.donchian_period).min()

        # 2. RSI 계산
        df["rsi"] = TechnicalIndicators.calculate_rsi(df["close"], self.rsi_period)

        # 3. 모멘텀 계산
        df["momentum"] = df["close"].pct_change(self.momentum_period)

        # 4. 신호 생성 (조정된 버전)
        # Donchian Channel 돌파 + 모멘텀 필터 (적절히 조정)
        long_condition = (df["close"] > df["donchian_upper"]) & (
            df["momentum"] > self.momentum_threshold * 0.2  # 임계값을 적절히 조정
        )
        short_condition = (df["close"] < df["donchian_lower"]) & (
            df["momentum"] < -self.momentum_threshold * 0.2  # 임계값을 적절히 조정
        )

        df.loc[long_condition, "signal"] = 1
        df.loc[short_condition, "signal"] = -1

        # 5. RSI 기반 평균회귀 신호 (Donchian 내부에서만, 적절히 조정)
        channel_inside = (df["close"] <= df["donchian_upper"]) & (
            df["close"] >= df["donchian_lower"]
        )

        # RSI 임계값을 적절히 조정
        rsi_oversold_adjusted = min(self.rsi_oversold + 5, 45)  # 적절한 과매도
        rsi_overbought_adjusted = max(self.rsi_overbought - 5, 55)  # 적절한 과매수

        rsi_long = (
            channel_inside & (df["rsi"] < rsi_oversold_adjusted) & (df["signal"] == 0)
        )
        rsi_short = (
            channel_inside & (df["rsi"] > rsi_overbought_adjusted) & (df["signal"] == 0)
        )

        df.loc[rsi_long, "signal"] = 1
        df.loc[rsi_short, "signal"] = -1

        # 6. 최소 보유 기간 적용
        df = self._apply_min_holding_period(df)

        return df

    def _apply_min_holding_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """최소 보유 기간 적용 - 수정된 버전"""
        min_holding_days = 3  # 최소 3일 보유 (적절히 조정)

        # 신호 변화 추적
        signals = df["signal"].values
        new_signals = signals.copy()

        current_signal = 0
        signal_start_idx = -1

        for i in range(len(signals)):
            if signals[i] != 0 and signals[i] != current_signal:
                # 새로운 신호 시작
                current_signal = signals[i]
                signal_start_idx = i
            elif signals[i] == 0 and current_signal != 0:
                # 신호 종료 - 최소 보유 기간 확인
                if signal_start_idx != -1 and (i - signal_start_idx) < min_holding_days:
                    # 최소 보유 기간 미달 시 신호 제거
                    new_signals[signal_start_idx:i] = 0
                current_signal = 0
                signal_start_idx = -1

        # 마지막 신호 처리
        if current_signal != 0 and signal_start_idx != -1:
            if (len(signals) - signal_start_idx) < min_holding_days:
                new_signals[signal_start_idx:] = 0

        df["signal"] = new_signals
        return df


# [스윙] 1~3주 보유
class VolatilityAdjustedBreakoutStrategy(BaseStrategy):
    """변동성 조정 채널 브레이크아웃 전략"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        # Keltner Channel 하이퍼파라미터
        self.keltner_period = getattr(params, "keltner_period", 20)
        self.keltner_multiplier = getattr(params, "keltner_multiplier", 2.0)
        # 볼륨 필터 하이퍼파라미터
        self.volume_period = getattr(params, "volume_period", 20)
        self.volume_threshold = getattr(params, "volume_threshold", 0.8)
        # 변동성 필터 하이퍼파라미터
        self.volatility_period = getattr(params, "volatility_period", 20)
        self.volatility_threshold = getattr(params, "volatility_threshold", 0.5)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """변동성 조정 브레이크아웃 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # Keltner Channel 계산 (하이퍼파라미터 적용)
        if "keltner_upper" not in df.columns or "keltner_lower" not in df.columns:
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            atr = df["high"] - df["low"]  # 간단한 ATR 계산
            atr_ma = atr.rolling(window=self.keltner_period).mean()
            typical_price_ma = typical_price.rolling(window=self.keltner_period).mean()

            df["keltner_upper"] = typical_price_ma + (self.keltner_multiplier * atr_ma)
            df["keltner_lower"] = typical_price_ma - (self.keltner_multiplier * atr_ma)

        # Keltner Channel 돌파 신호
        df.loc[df["close"] > df["keltner_upper"], "signal"] = 1  # 롱 신호
        df.loc[df["close"] < df["keltner_lower"], "signal"] = -1  # 숏 신호

        # 볼륨 필터 (하이퍼파라미터 적용)
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=self.volume_period).mean()
            volume_filter = df["volume"] > df["volume_ma"] * self.volume_threshold
        else:
            volume_filter = pd.Series([True] * len(df), index=df.index)

        # 변동성 필터 (하이퍼파라미터 적용)
        if "atr" in df.columns:
            df["atr_ma"] = df["atr"].rolling(window=self.volatility_period).mean()
            volatility_filter = df["atr"] > df["atr_ma"] * self.volatility_threshold
        else:
            volatility_filter = pd.Series([True] * len(df), index=df.index)

        # 필터 적용
        valid_signals = volume_filter & volatility_filter
        df.loc[~valid_signals, "signal"] = 0

        # 변동성 조정 포지션 사이즈
        df["position_size"] = df.apply(
            lambda row: self.calculate_position_size(df.loc[: row.name], row["signal"]),
            axis=1,
        )

        return df


# ============================================================================
# 자산배분/리스크 관리 전략 (Asset Allocation & Risk Management)
# ============================================================================


# [장기] 3개월~1년 리밸런싱
class RiskParityLeverageStrategy(BaseStrategy):
    """실전형 리스크 패리티 + 레버리지 ETF 전략"""

    def __init__(
        self,
        params: StrategyParams,
        symbols: list = None,
        group_constraints: dict = None,
        max_weight: float = 0.4,
        cash_weight: float = 0.05,
        leverage: float = 1.0,
    ):
        super().__init__(params)
        # 실전형 ETF 리스트 기본값 - 설정에서 동적으로 가져오기
        if symbols is None:
            # 설정에서 심볼 목록을 가져오거나 기본값 사용
            try:
                import json
                import os

                config_path = os.path.join(os.getcwd(), "config", "config_swing.json")
                if os.path.exists(config_path):
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                    self.symbols = config.get("data", {}).get("symbols", ["SPY", "QQQ"])
                else:
                    self.symbols = ["SPY", "QQQ"]
            except Exception:
                self.symbols = ["SPY", "QQQ"]
        else:
            self.symbols = symbols

        # 자산군별 제약조건 - 동적으로 생성
        if group_constraints is None:
            # 주식 자산군 (SPY, QQQ 등)
            equity_symbols = [
                s
                for s in self.symbols
                if s in ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "META", "NFLX"]
            ]
            # 기타 자산군 (나머지)
            other_symbols = [s for s in self.symbols if s not in equity_symbols]

            self.group_constraints = {}
            if equity_symbols:
                self.group_constraints["equity"] = {
                    "assets": equity_symbols,
                    "min": 0.2,
                    "max": 0.8,
                }
            if other_symbols:
                self.group_constraints["other"] = {
                    "assets": other_symbols,
                    "min": 0.0,
                    "max": 0.6,
                }
        else:
            self.group_constraints = group_constraints
        self.max_weight = max_weight
        self.cash_weight = cash_weight
        self.leverage = leverage

    def generate_signals(
        self, data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """리스크 패리티 기반 가중치 계산 및 신호 생성"""
        signals = {}
        # 공통 날짜 추출
        common_dates = set.intersection(
            *[set(df["datetime"]) for df in data_dict.values()]
        )
        common_dates = sorted(list(common_dates))
        # (여기서는 단순히 고정 비중 예시, 실제 구현은 리스크 패리티 최적화 필요)
        n = len(self.symbols)
        weights = [self.leverage / n] * n
        for idx, symbol in enumerate(self.symbols):
            df = data_dict[symbol].copy()
            df = df[df["datetime"].isin(common_dates)].sort_values("datetime")
            df["signal"] = 1
            df["weight"] = min(weights[idx], self.max_weight)
            signals[symbol] = df
        return signals


# ============================================================================
# 추세추종 전략 (Trend Following Strategies)
# ============================================================================


# [스윙] 1~3주 보유
class SwingEMACrossoverStrategy(BaseStrategy):
    """중기 이동평균 돌파 스윙 트레이딩 전략"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.ema_short_period = getattr(params, "ema_short_period", 20)
        self.ema_long_period = getattr(params, "ema_long_period", 50)
        self.min_holding_days = getattr(params, "min_holding_days", 5)
        self.max_holding_days = getattr(params, "max_holding_days", 15)
        # 추가 하이퍼파라미터
        self.slope_period = getattr(params, "slope_period", 5)
        self.volume_threshold = getattr(params, "volume_threshold", 0.8)
        self.volatility_threshold = getattr(params, "volatility_threshold", 0.5)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """EMA 크로스오버 스윙 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # EMA 계산
        df["ema_short"] = df["close"].ewm(span=self.ema_short_period).mean()
        df["ema_long"] = df["close"].ewm(span=self.ema_long_period).mean()

        # 크로스오버 신호 생성
        df["ema_cross"] = 0
        df.loc[df["ema_short"] > df["ema_long"], "ema_cross"] = 1
        df.loc[df["ema_short"] < df["ema_long"], "ema_cross"] = -1

        # 신호 변화점 감지 (크로스오버 발생 시점)
        df["signal_change"] = df["ema_cross"].diff()

        # 매수 신호: 상향 크로스오버
        df.loc[df["signal_change"] == 2, "signal"] = 1
        # 매도 신호: 하향 크로스오버
        df.loc[df["signal_change"] == -2, "signal"] = -1

        # 스윙 트레이딩을 위한 추가 필터링
        # 1. 추세 강도 확인 (EMA 기울기) - 하이퍼파라미터 적용
        df["ema_short_slope"] = df["ema_short"].diff(self.slope_period)
        df["ema_long_slope"] = df["ema_long"].diff(self.slope_period)

        # 2. 볼륨 확인 (하이퍼파라미터 적용) - 조건 완화
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            volume_filter = df["volume"] > df["volume_ma"] * (
                self.volume_threshold * 0.5
            )  # 조건 완화
        else:
            volume_filter = pd.Series([True] * len(df), index=df.index)

        # 3. 변동성 필터 (ATR 기반) - 하이퍼파라미터 적용 - 조건 완화
        if "atr" in df.columns:
            df["atr_ma"] = df["atr"].rolling(window=20).mean()
            volatility_filter = df["atr"] > df["atr_ma"] * (
                self.volatility_threshold * 0.5
            )  # 조건 완화
        else:
            volatility_filter = pd.Series([True] * len(df), index=df.index)

        # 필터 적용 - OR 조건으로 변경 (둘 중 하나만 만족하면 됨)
        valid_signals = volume_filter | volatility_filter
        df.loc[~valid_signals, "signal"] = 0

        return df


# [스윙] 1~2주 보유
class SwingRSIReversalStrategy(BaseStrategy):
    """RSI 리버설 스윙 트레이딩 전략"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.rsi_period = getattr(params, "rsi_period", 14)
        self.rsi_oversold = getattr(params, "rsi_oversold", 30)
        self.rsi_overbought = getattr(params, "rsi_overbought", 70)
        self.min_holding_days = getattr(params, "min_holding_days", 5)
        self.max_holding_days = getattr(params, "max_holding_days", 10)
        # 추가 하이퍼파라미터
        self.rsi_momentum_period = getattr(params, "rsi_momentum_period", 3)
        self.price_momentum_period = getattr(params, "price_momentum_period", 5)
        self.volume_threshold = getattr(params, "volume_threshold", 0.7)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI 리버설 스윙 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # RSI 계산
        if "rsi" not in df.columns:
            df["rsi"] = TechnicalIndicators.calculate_rsi(df["close"], self.rsi_period)

        # RSI 과매도/과매수 신호 (기본 신호)
        df["rsi_signal"] = 0
        df.loc[df["rsi"] < self.rsi_oversold, "rsi_signal"] = 1  # 매수 신호
        df.loc[df["rsi"] > self.rsi_overbought, "rsi_signal"] = -1  # 매도 신호

        # 신호 변화점 감지 (기존 방식)
        df["rsi_signal_change"] = df["rsi_signal"].diff()

        # 매수 신호: RSI가 과매도에서 벗어날 때
        df.loc[df["rsi_signal_change"] == 1, "signal"] = 1
        # 매도 신호: RSI가 과매수에서 벗어날 때
        df.loc[df["rsi_signal_change"] == -1, "signal"] = -1

        # 추가 신호 생성 (RSI 반전 신호)
        # RSI가 중간값(50) 근처에서 방향 전환할 때
        df["rsi_above_50"] = df["rsi"] > 50
        df["rsi_cross_50"] = df["rsi_above_50"].diff()

        # RSI가 50을 상향 돌파할 때 매수 신호 추가
        df.loc[
            (df["rsi_cross_50"] == True) & (df["rsi"] > 45) & (df["rsi"] < 55), "signal"
        ] = 1
        # RSI가 50을 하향 돌파할 때 매도 신호 추가
        df.loc[
            (df["rsi_cross_50"] == False) & (df["rsi"] > 45) & (df["rsi"] < 55),
            "signal",
        ] = -1

        # RSI 모멘텀 기반 신호 추가
        df["rsi_momentum"] = df["rsi"].diff(self.rsi_momentum_period)
        df["rsi_momentum_prev"] = df["rsi_momentum"].shift(1)

        # RSI 모멘텀이 음수에서 양수로 전환할 때 매수 신호
        df.loc[
            (df["rsi_momentum_prev"] < 0) & (df["rsi_momentum"] > 0) & (df["rsi"] < 60),
            "signal",
        ] = 1
        # RSI 모멘텀이 양수에서 음수로 전환할 때 매도 신호
        df.loc[
            (df["rsi_momentum_prev"] > 0) & (df["rsi_momentum"] < 0) & (df["rsi"] > 40),
            "signal",
        ] = -1

        # 가격 모멘텀 확인 (하이퍼파라미터 적용)
        df["price_momentum"] = df["close"].pct_change(self.price_momentum_period)

        # 볼륨 확인 (하이퍼파라미터 적용) - 조건 완화
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            volume_filter = df["volume"] > df["volume_ma"] * self.volume_threshold
            # 볼륨 필터는 선택적으로 적용 (너무 엄격하지 않게)
            volume_strict_filter = df["volume"] > df["volume_ma"] * (
                self.volume_threshold * 0.5
            )  # 조건 완화
        else:
            volume_filter = pd.Series([True] * len(df), index=df.index)
            volume_strict_filter = pd.Series([True] * len(df), index=df.index)

        # 필터링 조건 완화
        # RSI 매수: RSI 모멘텀 양수이거나 가격 모멘텀 양수일 때 (OR 조건으로 변경)
        rsi_buy_filter = (df["rsi_momentum"] > 0) | (df["price_momentum"] > 0)
        # RSI 매도: RSI 모멘텀 음수이거나 가격 모멘텀 음수일 때 (OR 조건으로 변경)
        rsi_sell_filter = (df["rsi_momentum"] < 0) | (df["price_momentum"] < 0)

        # 필터 적용 (조건 완화)
        df.loc[(df["signal"] == 1) & ~rsi_buy_filter, "signal"] = 0
        df.loc[(df["signal"] == -1) & ~rsi_sell_filter, "signal"] = 0

        # 볼륨 필터 적용
        df.loc[~volume_strict_filter, "signal"] = 0

        # 중복 신호 제거 (연속된 같은 방향 신호는 첫 번째만 유지)
        df["signal_shift"] = df["signal"].shift(1)
        df.loc[(df["signal"] == df["signal_shift"]) & (df["signal"] != 0), "signal"] = 0

        return df


# [스윙] 1~3주 보유
class DonchianSwingBreakoutStrategy(BaseStrategy):
    """Donchian Channel 돌파 스윙 트레이딩 전략"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.donchian_period = getattr(params, "donchian_period", 20)
        self.min_holding_days = getattr(params, "min_holding_days", 7)
        self.max_holding_days = getattr(params, "max_holding_days", 15)
        # 추가 하이퍼파라미터
        self.volume_threshold = getattr(params, "volume_threshold", 1.2)
        self.volatility_threshold = getattr(params, "volatility_threshold", 0.8)
        self.breakout_strength_threshold = getattr(
            params, "breakout_strength_threshold", 0.005
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Donchian Channel 돌파 스윙 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # Donchian Channel 계산
        if "donchian_upper" not in df.columns or "donchian_lower" not in df.columns:
            df["donchian_upper"] = df["high"].rolling(window=self.donchian_period).max()
            df["donchian_lower"] = df["low"].rolling(window=self.donchian_period).min()

        # 돌파 신호 생성
        df["breakout_signal"] = 0
        # 상향 돌파: 현재가가 이전 고점을 돌파
        df.loc[df["close"] > df["donchian_upper"].shift(1), "breakout_signal"] = 1
        # 하향 돌파: 현재가가 이전 저점을 돌파
        df.loc[df["close"] < df["donchian_lower"].shift(1), "breakout_signal"] = -1

        # 신호 변화점 감지
        df["breakout_change"] = df["breakout_signal"].diff()

        # 매수 신호: 상향 돌파
        df.loc[df["breakout_change"] == 1, "signal"] = 1
        # 매도 신호: 하향 돌파
        df.loc[df["breakout_change"] == -1, "signal"] = -1

        # 추가 필터링
        # 1. 돌파 강도 확인 (하이퍼파라미터 적용)
        df["breakout_strength"] = 0.0
        # 상향 돌파 강도
        df.loc[df["signal"] == 1, "breakout_strength"] = (
            df.loc[df["signal"] == 1, "close"]
            - df.loc[df["signal"] == 1, "donchian_upper"].shift(1)
        ) / df.loc[df["signal"] == 1, "donchian_upper"].shift(1)
        # 하향 돌파 강도
        df.loc[df["signal"] == -1, "breakout_strength"] = (
            df.loc[df["signal"] == -1, "donchian_lower"].shift(1)
            - df.loc[df["signal"] == -1, "close"]
        ) / df.loc[df["signal"] == -1, "donchian_lower"].shift(1)

        # 2. 볼륨 확인 (하이퍼파라미터 적용)
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            volume_filter = (
                df["volume"] > df["volume_ma"] * self.volume_threshold
            )  # 돌파 시 볼륨 증가 필요
        else:
            volume_filter = pd.Series([True] * len(df), index=df.index)

        # 3. 변동성 확인 (하이퍼파라미터 적용)
        if "atr" in df.columns:
            df["atr_ma"] = df["atr"].rolling(window=20).mean()
            volatility_filter = df["atr"] > df["atr_ma"] * self.volatility_threshold
        else:
            volatility_filter = pd.Series([True] * len(df), index=df.index)

        # 필터 적용 (하이퍼파라미터 적용)
        strength_filter = (
            abs(df["breakout_strength"]) > self.breakout_strength_threshold
        )

        # 필터 적용
        df.loc[~strength_filter, "signal"] = 0
        df.loc[~volume_filter, "signal"] = 0
        df.loc[~volatility_filter, "signal"] = 0

        return df


class StochasticStrategy(BaseStrategy):
    """Stochastic Oscillator 기반 전략"""

    def __init__(
        self,
        params: StrategyParams,
        k_period: int = 14,
        d_period: int = 3,
        low_threshold: float = 20,
        high_threshold: float = 80,
    ):
        super().__init__(params)
        self.k_period = k_period
        self.d_period = d_period
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        # stoch_k, stoch_d가 없으면 계산
        if "stoch_k" not in df.columns or "stoch_d" not in df.columns:
            low_min = df["low"].rolling(window=self.k_period).min()
            high_max = df["high"].rolling(window=self.k_period).max()
            df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min)
            df["stoch_d"] = df["stoch_k"].rolling(window=self.d_period).mean()
        # 신호: %K가 low_threshold 아래에서 상향 돌파 → 매수, high_threshold 위에서 하향 돌파 → 매도
        buy = (
            (df["stoch_k"].shift(1) < self.low_threshold)
            & (df["stoch_k"] > self.low_threshold)
            & (df["stoch_k"] > df["stoch_d"])
            & (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1))
        )
        sell = (
            (df["stoch_k"].shift(1) > self.high_threshold)
            & (df["stoch_k"] < self.high_threshold)
            & (df["stoch_k"] < df["stoch_d"])
            & (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1))
        )
        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1
        return df


class WilliamsRStrategy(BaseStrategy):
    """Williams %R 기반 전략"""

    def __init__(
        self,
        params: StrategyParams,
        period: int = 14,
        low_threshold: float = -80,
        high_threshold: float = -20,
    ):
        super().__init__(params)
        self.period = period
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        # williams_r이 없으면 계산
        if "williams_r" not in df.columns:
            high_max = df["high"].rolling(window=self.period).max()
            low_min = df["low"].rolling(window=self.period).min()
            df["williams_r"] = -100 * (high_max - df["close"]) / (high_max - low_min)
        # 신호: low_threshold 아래에서 위로 돌파 → 매수, high_threshold 위에서 아래로 돌파 → 매도
        buy = (df["williams_r"].shift(1) < self.low_threshold) & (
            df["williams_r"] > self.low_threshold
        )
        sell = (df["williams_r"].shift(1) > self.high_threshold) & (
            df["williams_r"] < self.high_threshold
        )
        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1
        return df


class CCIStrategy(BaseStrategy):
    """CCI 기반 전략"""

    def __init__(
        self, params: StrategyParams, period: int = 20, threshold: float = 100
    ):
        super().__init__(params)
        self.period = period
        self.threshold = threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        # cci가 없으면 계산
        if "cci" not in df.columns:
            tp = (df["high"] + df["low"] + df["close"]) / 3
            ma = tp.rolling(window=self.period).mean()
            md = tp.rolling(window=self.period).apply(
                lambda x: np.mean(np.abs(x - np.mean(x)))
            )
            df["cci"] = (tp - ma) / (0.015 * md)
        # 신호: -threshold 아래에서 상향 돌파 → 매수, +threshold 위에서 하향 돌파 → 매도
        buy = (df["cci"].shift(1) < -self.threshold) & (df["cci"] > -self.threshold)
        sell = (df["cci"].shift(1) > self.threshold) & (df["cci"] < self.threshold)
        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1
        return df


class CCIBollingerStrategy(BaseStrategy):
    """CCI + Bollinger Band 결합 전략 - CCI 과매도 + 밴드 이탈 = 매수 신호"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        # CCI 파라미터
        self.cci_period = getattr(params, "cci_period", 20)
        self.cci_oversold = getattr(params, "cci_oversold", -100)
        self.cci_overbought = getattr(params, "cci_overbought", 100)

        # Bollinger Band 파라미터
        self.bb_period = getattr(params, "bb_period", 20)
        self.bb_std = getattr(params, "bb_std", 2.0)

        # 휩쏘 방지 파라미터
        self.volume_threshold = getattr(params, "volume_threshold", 1.2)
        self.atr_threshold = getattr(params, "atr_threshold", 0.8)
        self.min_holding_period = getattr(params, "min_holding_period", 3)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """CCI + Bollinger Band 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # 1. CCI 계산
        if "cci" not in df.columns:
            tp = (df["high"] + df["low"] + df["close"]) / 3
            ma = tp.rolling(window=self.cci_period).mean()
            md = tp.rolling(window=self.cci_period).apply(
                lambda x: np.mean(np.abs(x - np.mean(x)))
            )
            df["cci"] = (tp - ma) / (0.015 * md)

        # 2. Bollinger Band 계산
        df["bb_middle"] = df["close"].rolling(window=self.bb_period).mean()
        bb_std = df["close"].rolling(window=self.bb_period).std()
        df["bb_upper"] = df["bb_middle"] + (self.bb_std * bb_std)
        df["bb_lower"] = df["bb_middle"] - (self.bb_std * bb_std)

        # 3. ATR 계산
        if "atr" not in df.columns:
            df["atr"] = TechnicalIndicators.calculate_atr(df, 14)

        # 4. 신호 생성
        # 매수 신호: CCI 과매도 + 밴드 하단 이탈
        buy_signal = (
            (df["cci"] < self.cci_oversold)  # CCI 과매도
            & (df["close"] < df["bb_lower"])  # 밴드 하단 이탈
            & (df["close"] > df["close"].shift(1))  # 가격 반등 시작
        )

        # 매도 신호: CCI 과매수 + 밴드 상단 이탈
        sell_signal = (
            (df["cci"] > self.cci_overbought)  # CCI 과매수
            & (df["close"] > df["bb_upper"])  # 밴드 상단 이탈
            & (df["close"] < df["close"].shift(1))  # 가격 하락 시작
        )

        df.loc[buy_signal, "signal"] = 1
        df.loc[sell_signal, "signal"] = -1

        # 5. 필터링 조건
        # 거래량 필터
        volume_filter = pd.Series([True] * len(df), index=df.index)
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            volume_filter = df["volume"] > df["volume_ma"] * self.volume_threshold

        # 변동성 필터
        volatility_filter = pd.Series([True] * len(df), index=df.index)
        if "atr" in df.columns:
            df["atr_ma"] = df["atr"].rolling(window=20).mean()
            volatility_filter = df["atr"] > df["atr_ma"] * self.atr_threshold

        # 6. 최종 신호 생성
        final_filter = volume_filter & volatility_filter
        df.loc[~final_filter, "signal"] = 0

        # 7. 최소 홀딩 기간 적용
        df = self._apply_min_holding_period(df)

        return df

    def _apply_min_holding_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """최소 홀딩 기간 적용"""
        df = df.copy()
        current_signal = 0
        signal_start = 0

        for i in range(len(df)):
            if df["signal"].iloc[i] != 0 and df["signal"].iloc[i] != current_signal:
                if i - signal_start >= self.min_holding_period or current_signal == 0:
                    current_signal = df["signal"].iloc[i]
                    signal_start = i
                else:
                    df.loc[df.index[i], "signal"] = 0

        return df


class StochDonchianStrategy(BaseStrategy):
    """Stoch %K 교차 + Donchian 채널 전략 - 추세 전환 감지"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        # Stochastic 파라미터
        self.stoch_k_period = getattr(params, "stoch_k_period", 14)
        self.stoch_d_period = getattr(params, "stoch_d_period", 3)
        self.stoch_low_threshold = getattr(params, "stoch_low_threshold", 20)
        self.stoch_high_threshold = getattr(params, "stoch_high_threshold", 80)

        # Donchian Channel 파라미터
        self.donchian_period = getattr(params, "donchian_period", 20)

        # 휩쏘 방지 파라미터
        self.volume_threshold = getattr(params, "volume_threshold", 1.2)
        self.atr_threshold = getattr(params, "atr_threshold", 0.8)
        self.min_holding_period = getattr(params, "min_holding_period", 3)
        self.confirmation_period = getattr(params, "confirmation_period", 2)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stoch %K 교차 + Donchian 채널 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # 1. Stochastic 계산
        if "stoch_k" not in df.columns or "stoch_d" not in df.columns:
            low_min = df["low"].rolling(window=self.stoch_k_period).min()
            high_max = df["high"].rolling(window=self.stoch_k_period).max()
            df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min)
            df["stoch_d"] = df["stoch_k"].rolling(window=self.stoch_d_period).mean()

        # 2. Donchian Channel 계산
        if "donchian_upper" not in df.columns or "donchian_lower" not in df.columns:
            df["donchian_upper"] = df["high"].rolling(window=self.donchian_period).max()
            df["donchian_lower"] = df["low"].rolling(window=self.donchian_period).min()

        # 3. ATR 계산
        if "atr" not in df.columns:
            df["atr"] = TechnicalIndicators.calculate_atr(df, 14)

        # 4. Stochastic 교차 신호
        df["stoch_cross"] = 0
        # %K가 %D를 상향 돌파 (골든 크로스)
        golden_cross = (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1)) & (
            df["stoch_k"] > df["stoch_d"]
        )
        # %K가 %D를 하향 돌파 (데드 크로스)
        dead_cross = (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1)) & (
            df["stoch_k"] < df["stoch_d"]
        )

        df.loc[golden_cross, "stoch_cross"] = 1
        df.loc[dead_cross, "stoch_cross"] = -1

        # 5. Donchian Channel 신호
        df["donchian_signal"] = 0
        # 상향 돌파
        df.loc[df["close"] > df["donchian_upper"].shift(1), "donchian_signal"] = 1
        # 하향 돌파
        df.loc[df["close"] < df["donchian_lower"].shift(1), "donchian_signal"] = -1

        # 6. 신호 결합 및 확인
        df["combined_signal"] = 0

        # 매수 신호: Stochastic 골든 크로스 + Donchian 상향 돌파
        buy_condition = (
            (df["stoch_cross"] == 1)
            & (df["donchian_signal"] == 1)
            & (df["stoch_k"] > self.stoch_low_threshold)  # 과매도 구간 제외
        )

        # 매도 신호: Stochastic 데드 크로스 + Donchian 하향 돌파
        sell_condition = (
            (df["stoch_cross"] == -1)
            & (df["donchian_signal"] == -1)
            & (df["stoch_k"] < self.stoch_high_threshold)  # 과매수 구간 제외
        )

        df.loc[buy_condition, "combined_signal"] = 1
        df.loc[sell_condition, "combined_signal"] = -1

        # 7. 신호 확인 (휩쏘 방지)
        df["signal_confirmed"] = False
        for i in range(self.confirmation_period, len(df)):
            if df["combined_signal"].iloc[i] != 0:
                # 신호 후 일정 기간 동안 방향 유지 확인
                recent_signals = df["combined_signal"].iloc[
                    i - self.confirmation_period : i + 1
                ]
                if len(set(recent_signals)) == 1 and recent_signals.iloc[0] != 0:
                    df.loc[df.index[i], "signal_confirmed"] = True

        # 8. 필터링 조건
        # 거래량 필터
        volume_filter = pd.Series([True] * len(df), index=df.index)
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            volume_filter = df["volume"] > df["volume_ma"] * self.volume_threshold

        # 변동성 필터
        volatility_filter = pd.Series([True] * len(df), index=df.index)
        if "atr" in df.columns:
            df["atr_ma"] = df["atr"].rolling(window=20).mean()
            volatility_filter = df["atr"] > df["atr_ma"] * self.atr_threshold

        # 9. 최종 신호 생성
        final_filter = volume_filter & volatility_filter & df["signal_confirmed"]
        df.loc[final_filter & (df["combined_signal"] == 1), "signal"] = 1
        df.loc[final_filter & (df["combined_signal"] == -1), "signal"] = -1

        # 10. 최소 홀딩 기간 적용
        df = self._apply_min_holding_period(df)

        return df

    def _apply_min_holding_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """최소 홀딩 기간 적용"""
        df = df.copy()
        current_signal = 0
        signal_start = 0

        for i in range(len(df)):
            if df["signal"].iloc[i] != 0 and df["signal"].iloc[i] != current_signal:
                if i - signal_start >= self.min_holding_period or current_signal == 0:
                    current_signal = df["signal"].iloc[i]
                    signal_start = i
                else:
                    df.loc[df.index[i], "signal"] = 0

        return df


class WhipsawPreventionStrategy(BaseStrategy):
    """휩쏘 방지 전략 - 다중 필터링으로 거짓 신호 제거"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        # 기본 지표 파라미터
        self.ema_short = getattr(params, "ema_short", 10)
        self.ema_long = getattr(params, "ema_long", 20)
        self.rsi_period = getattr(params, "rsi_period", 14)
        self.atr_period = getattr(params, "atr_period", 14)

        # 휩쏘 방지 파라미터
        self.signal_confirmation_period = getattr(
            params, "signal_confirmation_period", 3
        )
        self.volume_threshold = getattr(params, "volume_threshold", 1.5)
        self.volatility_threshold = getattr(params, "volatility_threshold", 0.8)
        self.trend_strength_threshold = getattr(
            params, "trend_strength_threshold", 0.02
        )
        self.min_holding_period = getattr(params, "min_holding_period", 5)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """휩쏘 방지 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # 1. 기본 지표 계산
        df["ema_short"] = df["close"].ewm(span=self.ema_short).mean()
        df["ema_long"] = df["close"].ewm(span=self.ema_long).mean()

        if "rsi" not in df.columns:
            df["rsi"] = TechnicalIndicators.calculate_rsi(df["close"], self.rsi_period)

        if "atr" not in df.columns:
            df["atr"] = TechnicalIndicators.calculate_atr(df, self.atr_period)

        # 2. 기본 신호 생성 (EMA 크로스오버)
        df["ema_cross"] = 0
        df.loc[df["ema_short"] > df["ema_long"], "ema_cross"] = 1
        df.loc[df["ema_short"] < df["ema_long"], "ema_cross"] = -1

        # 3. 휩쏘 방지 필터링

        # A. 신호 지속성 확인
        df["signal_strength"] = 0
        for i in range(self.signal_confirmation_period, len(df)):
            recent_signals = df["ema_cross"].iloc[
                i - self.signal_confirmation_period : i + 1
            ]
            if len(set(recent_signals)) == 1 and recent_signals.iloc[0] != 0:
                df.loc[df.index[i], "signal_strength"] = recent_signals.iloc[0]

        # B. 거래량 확인
        volume_filter = pd.Series([True] * len(df), index=df.index)
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            volume_filter = df["volume"] > df["volume_ma"] * self.volume_threshold

        # C. 변동성 필터
        volatility_filter = pd.Series([True] * len(df), index=df.index)
        if "atr" in df.columns:
            df["atr_ma"] = df["atr"].rolling(window=20).mean()
            volatility_filter = df["atr"] > df["atr_ma"] * self.volatility_threshold

        # D. 추세 강도 확인
        df["trend_strength"] = abs(df["ema_short"] - df["ema_long"]) / df["close"]
        trend_filter = df["trend_strength"] > self.trend_strength_threshold

        # E. RSI 필터 (과매수/과매도 구간 제외)
        rsi_filter = (df["rsi"] > 20) & (df["rsi"] < 80)

        # 4. 최종 신호 생성
        final_filter = volume_filter & volatility_filter & trend_filter & rsi_filter
        df.loc[final_filter, "signal"] = df.loc[final_filter, "signal_strength"]

        # 5. 최소 홀딩 기간 적용
        df = self._apply_min_holding_period(df)

        return df

    def _apply_min_holding_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """최소 홀딩 기간 적용으로 과도한 신호 변경 방지"""
        df = df.copy()
        current_signal = 0
        signal_start = 0

        for i in range(len(df)):
            if df["signal"].iloc[i] != 0 and df["signal"].iloc[i] != current_signal:
                # 새로운 신호 시작
                if i - signal_start >= self.min_holding_period or current_signal == 0:
                    current_signal = df["signal"].iloc[i]
                    signal_start = i
                else:
                    # 최소 홀딩 기간 미달 시 신호 무시
                    df.loc[df.index[i], "signal"] = 0

        return df


class DonchianRSIWhipsawStrategy(BaseStrategy):
    """Donchian + RSI 듀얼 모멘텀 휩쏘 방지 전략"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        # Donchian Channel 파라미터
        self.donchian_period = getattr(params, "donchian_period", 20)

        # RSI 파라미터
        self.rsi_period = getattr(params, "rsi_period", 14)
        self.rsi_oversold = getattr(params, "rsi_oversold", 30)
        self.rsi_overbought = getattr(params, "rsi_overbought", 70)

        # 휩쏘 방지 파라미터
        self.breakout_confirmation_period = getattr(
            params, "breakout_confirmation_period", 2
        )
        self.volume_threshold = getattr(params, "volume_threshold", 1.2)
        self.atr_threshold = getattr(params, "atr_threshold", 0.8)
        self.min_holding_period = getattr(params, "min_holding_period", 3)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Donchian + RSI 듀얼 모멘텀 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # 1. Donchian Channel 계산
        df["donchian_upper"] = df["high"].rolling(window=self.donchian_period).max()
        df["donchian_lower"] = df["low"].rolling(window=self.donchian_period).min()

        # 2. RSI 계산
        if "rsi" not in df.columns:
            df["rsi"] = TechnicalIndicators.calculate_rsi(df["close"], self.rsi_period)

        # 3. ATR 계산
        if "atr" not in df.columns:
            df["atr"] = TechnicalIndicators.calculate_atr(df, 14)

        # 4. Donchian 돌파 신호
        df["donchian_breakout"] = 0
        df.loc[df["close"] > df["donchian_upper"].shift(1), "donchian_breakout"] = 1
        df.loc[df["close"] < df["donchian_lower"].shift(1), "donchian_breakout"] = -1

        # 5. RSI 신호
        df["rsi_signal"] = 0
        df.loc[df["rsi"] < self.rsi_oversold, "rsi_signal"] = 1
        df.loc[df["rsi"] > self.rsi_overbought, "rsi_signal"] = -1

        # 6. 돌파 확인 (휩쏘 방지)
        df["breakout_confirmed"] = False
        for i in range(self.breakout_confirmation_period, len(df)):
            if df["donchian_breakout"].iloc[i] != 0:
                # 돌파 후 일정 기간 동안 돌파 상태 유지 확인
                recent_breakout = df["donchian_breakout"].iloc[
                    i - self.breakout_confirmation_period : i + 1
                ]
                if len(set(recent_breakout)) == 1 and recent_breakout.iloc[0] != 0:
                    df.loc[df.index[i], "breakout_confirmed"] = True

        # 7. 필터링 조건
        # 거래량 필터
        volume_filter = pd.Series([True] * len(df), index=df.index)
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            volume_filter = df["volume"] > df["volume_ma"] * self.volume_threshold

        # 변동성 필터
        volatility_filter = pd.Series([True] * len(df), index=df.index)
        if "atr" in df.columns:
            df["atr_ma"] = df["atr"].rolling(window=20).mean()
            volatility_filter = df["atr"] > df["atr_ma"] * self.atr_threshold

        # 8. 최종 신호 생성
        # Donchian 돌파가 확인되고 필터를 통과한 경우
        donchian_signals = df["breakout_confirmed"] & volume_filter & volatility_filter
        df.loc[donchian_signals & (df["donchian_breakout"] == 1), "signal"] = 1
        df.loc[donchian_signals & (df["donchian_breakout"] == -1), "signal"] = -1

        # RSI 신호는 Donchian Channel 내부에서만 적용
        channel_inside = (df["close"] <= df["donchian_upper"]) & (
            df["close"] >= df["donchian_lower"]
        )
        rsi_signals = channel_inside & volume_filter & volatility_filter
        df.loc[rsi_signals & (df["rsi_signal"] == 1), "signal"] = 1
        df.loc[rsi_signals & (df["rsi_signal"] == -1), "signal"] = -1

        # 9. 최소 홀딩 기간 적용
        df = self._apply_min_holding_period(df)

        return df

    def _apply_min_holding_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """최소 홀딩 기간 적용"""
        df = df.copy()
        current_signal = 0
        signal_start = 0

        for i in range(len(df)):
            if df["signal"].iloc[i] != 0 and df["signal"].iloc[i] != current_signal:
                if i - signal_start >= self.min_holding_period or current_signal == 0:
                    current_signal = df["signal"].iloc[i]
                    signal_start = i
                else:
                    df.loc[df.index[i], "signal"] = 0

        return df


class VolatilityFilteredBreakoutStrategy(BaseStrategy):
    """변동성 필터링 브레이크아웃 휩쏘 방지 전략"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        # 브레이크아웃 파라미터
        self.breakout_period = getattr(params, "breakout_period", 20)
        self.breakout_threshold = getattr(
            params, "breakout_threshold", 0.01
        )  # 2% → 1%로 완화

        # 변동성 필터 파라미터
        self.atr_period = getattr(params, "atr_period", 14)
        self.volatility_lookback = getattr(params, "volatility_lookback", 50)
        self.volatility_quantile = getattr(
            params, "volatility_quantile", 0.5
        )  # 0.7 → 0.5로 완화

        # 거래량 필터 파라미터
        self.volume_period = getattr(params, "volume_period", 20)
        self.volume_threshold = getattr(
            params, "volume_threshold", 1.2
        )  # 1.5 → 1.2로 완화

        # 신호 확인 파라미터
        self.confirmation_period = getattr(
            params, "confirmation_period", 2
        )  # 3 → 2로 완화
        self.min_holding_period = getattr(
            params, "min_holding_period", 3
        )  # 5 → 3으로 완화

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """변동성 필터링 브레이크아웃 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # 1. 브레이크아웃 레벨 계산
        df["breakout_high"] = df["high"].rolling(window=self.breakout_period).max()
        df["breakout_low"] = df["low"].rolling(window=self.breakout_period).min()

        # 2. ATR 계산
        if "atr" not in df.columns:
            df["atr"] = TechnicalIndicators.calculate_atr(df, self.atr_period)

        # 3. 변동성 필터 계산
        df["volatility_rank"] = (
            df["atr"].rolling(window=self.volatility_lookback).rank(pct=True)
        )
        volatility_filter = df["volatility_rank"] > self.volatility_quantile

        # 4. 거래량 필터
        volume_filter = pd.Series([True] * len(df), index=df.index)
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=self.volume_period).mean()
            volume_filter = df["volume"] > df["volume_ma"] * self.volume_threshold

        # 5. 브레이크아웃 신호 생성
        df["breakout_signal"] = 0

        # 상향 브레이크아웃 (조건 완화)
        upward_breakout = (df["close"] > df["breakout_high"].shift(1)) | (  # 고점 돌파
            df["close"] > df["close"].shift(1) * (1 + self.breakout_threshold)
        )  # 가격 상승

        # 하향 브레이크아웃 (조건 완화)
        downward_breakout = (df["close"] < df["breakout_low"].shift(1)) | (  # 저점 돌파
            df["close"] < df["close"].shift(1) * (1 - self.breakout_threshold)
        )  # 가격 하락

        df.loc[upward_breakout, "breakout_signal"] = 1
        df.loc[downward_breakout, "breakout_signal"] = -1

        # 6. 브레이크아웃 확인 (휩쏘 방지)
        df["breakout_confirmed"] = False
        for i in range(self.confirmation_period, len(df)):
            if df["breakout_signal"].iloc[i] != 0:
                # 브레이크아웃 후 일정 기간 동안 방향 유지 확인
                recent_signals = df["breakout_signal"].iloc[
                    i - self.confirmation_period : i + 1
                ]
                if len(set(recent_signals)) == 1 and recent_signals.iloc[0] != 0:
                    df.loc[df.index[i], "breakout_confirmed"] = True

        # 7. 최종 신호 생성
        final_filter = volatility_filter & volume_filter & df["breakout_confirmed"]
        df.loc[final_filter & (df["breakout_signal"] == 1), "signal"] = 1
        df.loc[final_filter & (df["breakout_signal"] == -1), "signal"] = -1

        # 디버깅 정보 (선택적)
        if len(df) > 0:
            total_signals = len(df[df["breakout_signal"] != 0])
            confirmed_signals = len(df[df["breakout_confirmed"] == True])
            volatility_passed = len(df[volatility_filter])
            volume_passed = len(df[volume_filter])
            final_signals = len(df[df["signal"] != 0])

            # 로깅 (필요시 주석 해제)
            # print(f"VolatilityFilteredBreakout Debug:")
            # print(f"  Total breakout signals: {total_signals}")
            # print(f"  Confirmed signals: {confirmed_signals}")
            # print(f"  Volatility filter passed: {volatility_passed}")
            # print(f"  Volume filter passed: {volume_passed}")
            # print(f"  Final signals: {final_signals}")

        # 8. 최소 홀딩 기간 적용
        df = self._apply_min_holding_period(df)

        return df

    def _apply_min_holding_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """최소 홀딩 기간 적용"""
        df = df.copy()
        current_signal = 0
        signal_start = 0

        for i in range(len(df)):
            if df["signal"].iloc[i] != 0 and df["signal"].iloc[i] != current_signal:
                if i - signal_start >= self.min_holding_period or current_signal == 0:
                    current_signal = df["signal"].iloc[i]
                    signal_start = i
                else:
                    df.loc[df.index[i], "signal"] = 0

        return df


class MultiTimeframeWhipsawStrategy(BaseStrategy):
    """다중 시간 프레임 휩쏘 방지 전략"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        # 다중 시간 프레임 파라미터
        self.short_period = getattr(params, "short_period", 5)
        self.medium_period = getattr(params, "medium_period", 10)
        self.long_period = getattr(params, "long_period", 20)

        # RSI 파라미터
        self.rsi_period = getattr(params, "rsi_period", 14)
        self.rsi_oversold = getattr(params, "rsi_oversold", 30)
        self.rsi_overbought = getattr(params, "rsi_overbought", 70)

        # 필터링 파라미터
        self.volume_threshold = getattr(params, "volume_threshold", 1.3)
        self.atr_threshold = getattr(params, "atr_threshold", 0.8)
        self.min_holding_period = getattr(params, "min_holding_period", 5)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """다중 시간 프레임 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # 1. 다중 시간 프레임 EMA 계산
        df["ema_short"] = df["close"].ewm(span=self.short_period).mean()
        df["ema_medium"] = df["close"].ewm(span=self.medium_period).mean()
        df["ema_long"] = df["close"].ewm(span=self.long_period).mean()

        # 2. RSI 계산
        if "rsi" not in df.columns:
            df["rsi"] = TechnicalIndicators.calculate_rsi(df["close"], self.rsi_period)

        # 3. ATR 계산
        if "atr" not in df.columns:
            df["atr"] = TechnicalIndicators.calculate_atr(df, 14)

        # 4. 다중 시간 프레임 신호 생성
        df["short_trend"] = np.where(df["close"] > df["ema_short"], 1, -1)
        df["medium_trend"] = np.where(df["close"] > df["ema_medium"], 1, -1)
        df["long_trend"] = np.where(df["close"] > df["ema_long"], 1, -1)

        # 5. 트렌드 일치도 계산
        df["trend_alignment"] = (
            df["short_trend"] + df["medium_trend"] + df["long_trend"]
        )

        # 6. 신호 생성 (모든 시간 프레임이 일치할 때만)
        df["trend_signal"] = 0
        df.loc[df["trend_alignment"] == 3, "trend_signal"] = 1  # 모든 시간 프레임 상승
        df.loc[df["trend_alignment"] == -3, "trend_signal"] = (
            -1
        )  # 모든 시간 프레임 하락

        # 7. RSI 필터 (과매수/과매도 구간에서 반대 신호)
        df["rsi_signal"] = 0
        df.loc[df["rsi"] < self.rsi_oversold, "rsi_signal"] = 1
        df.loc[df["rsi"] > self.rsi_overbought, "rsi_signal"] = -1

        # 8. 필터링 조건
        # 거래량 필터
        volume_filter = pd.Series([True] * len(df), index=df.index)
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            volume_filter = df["volume"] > df["volume_ma"] * self.volume_threshold

        # 변동성 필터
        volatility_filter = pd.Series([True] * len(df), index=df.index)
        if "atr" in df.columns:
            df["atr_ma"] = df["atr"].rolling(window=20).mean()
            volatility_filter = df["atr"] > df["atr_ma"] * self.atr_threshold

        # 9. 최종 신호 생성
        # 트렌드 신호 우선, RSI 신호는 보조
        trend_signals = volume_filter & volatility_filter & (df["trend_signal"] != 0)
        df.loc[trend_signals, "signal"] = df.loc[trend_signals, "trend_signal"]

        # RSI 신호는 트렌드가 불분명할 때만 적용
        unclear_trend = (
            (df["trend_alignment"].abs() <= 1) & volume_filter & volatility_filter
        )
        df.loc[unclear_trend & (df["rsi_signal"] != 0), "signal"] = df.loc[
            unclear_trend & (df["rsi_signal"] != 0), "rsi_signal"
        ]

        # 10. 최소 홀딩 기간 적용
        df = self._apply_min_holding_period(df)

        return df

    def _apply_min_holding_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """최소 홀딩 기간 적용"""
        df = df.copy()
        current_signal = 0
        signal_start = 0

        for i in range(len(df)):
            if df["signal"].iloc[i] != 0 and df["signal"].iloc[i] != current_signal:
                if i - signal_start >= self.min_holding_period or current_signal == 0:
                    current_signal = df["signal"].iloc[i]
                    signal_start = i
                else:
                    df.loc[df.index[i], "signal"] = 0

        return df


class AdaptiveWhipsawStrategy(BaseStrategy):
    """적응형 휩쏘 방지 전략 - 시장 상황에 따라 동적 조정"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        # 기본 지표 파라미터
        self.ema_period = getattr(params, "ema_period", 20)
        self.rsi_period = getattr(params, "rsi_period", 14)
        self.atr_period = getattr(params, "atr_period", 14)

        # 적응형 파라미터
        self.volatility_lookback = getattr(params, "volatility_lookback", 50)
        self.trend_lookback = getattr(params, "trend_lookback", 20)
        self.volume_lookback = getattr(params, "volume_lookback", 20)

        # 동적 임계값
        self.base_volume_threshold = getattr(params, "base_volume_threshold", 1.2)
        self.base_volatility_threshold = getattr(
            params, "base_volatility_threshold", 0.8
        )
        self.base_holding_period = getattr(params, "base_holding_period", 5)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """적응형 휩쏘 방지 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # 1. 기본 지표 계산
        df["ema"] = df["close"].ewm(span=self.ema_period).mean()

        if "rsi" not in df.columns:
            df["rsi"] = TechnicalIndicators.calculate_rsi(df["close"], self.rsi_period)

        if "atr" not in df.columns:
            df["atr"] = TechnicalIndicators.calculate_atr(df, self.atr_period)

        # 2. 시장 상황 분석
        df = self._analyze_market_conditions(df)

        # 3. 동적 임계값 계산
        df = self._calculate_adaptive_thresholds(df)

        # 4. 기본 신호 생성
        df["ema_signal"] = np.where(df["close"] > df["ema"], 1, -1)

        # 5. 적응형 필터링
        df = self._apply_adaptive_filters(df)

        # 6. 최종 신호 생성
        df["signal"] = df["filtered_signal"]

        return df

    def _analyze_market_conditions(self, df: pd.DataFrame) -> pd.DataFrame:
        """시장 상황 분석"""
        # 변동성 레벨
        df["volatility_level"] = (
            df["atr"].rolling(window=self.volatility_lookback).rank(pct=True)
        )

        # 트렌드 강도
        df["trend_strength"] = abs(df["close"] - df["ema"]) / df["close"]
        df["trend_strength_rank"] = (
            df["trend_strength"].rolling(window=self.trend_lookback).rank(pct=True)
        )

        # 거래량 레벨
        if "volume" in df.columns:
            df["volume_level"] = (
                df["volume"].rolling(window=self.volume_lookback).rank(pct=True)
            )
        else:
            df["volume_level"] = 0.5

        # 시장 상태 분류
        df["market_state"] = "normal"
        df.loc[df["volatility_level"] > 0.8, "market_state"] = "high_volatility"
        df.loc[df["trend_strength_rank"] < 0.2, "market_state"] = "sideways"
        df.loc[
            (df["volatility_level"] > 0.8) & (df["trend_strength_rank"] < 0.2),
            "market_state",
        ] = "whipsaw"

        return df

    def _calculate_adaptive_thresholds(self, df: pd.DataFrame) -> pd.DataFrame:
        """동적 임계값 계산"""
        # 변동성에 따른 임계값 조정
        df["adaptive_volume_threshold"] = self.base_volume_threshold
        df["adaptive_volatility_threshold"] = self.base_volatility_threshold
        df["adaptive_holding_period"] = self.base_holding_period

        # 휩쏘 상황에서는 더 엄격한 필터 적용
        whipsaw_mask = df["market_state"] == "whipsaw"
        df.loc[whipsaw_mask, "adaptive_volume_threshold"] *= 1.5
        df.loc[whipsaw_mask, "adaptive_volatility_threshold"] *= 1.3
        df.loc[whipsaw_mask, "adaptive_holding_period"] *= 2

        # 고변동성 상황에서는 중간 정도 필터
        high_vol_mask = df["market_state"] == "high_volatility"
        df.loc[high_vol_mask, "adaptive_volume_threshold"] *= 1.2
        df.loc[high_vol_mask, "adaptive_volatility_threshold"] *= 1.1
        df.loc[high_vol_mask, "adaptive_holding_period"] *= 1.5

        return df

    def _apply_adaptive_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """적응형 필터 적용"""
        df["filtered_signal"] = 0

        # 거래량 필터
        volume_filter = pd.Series([True] * len(df), index=df.index)
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            volume_filter = (
                df["volume"] > df["volume_ma"] * df["adaptive_volume_threshold"]
            )

        # 변동성 필터
        volatility_filter = pd.Series([True] * len(df), index=df.index)
        if "atr" in df.columns:
            df["atr_ma"] = df["atr"].rolling(window=20).mean()
            volatility_filter = (
                df["atr"] > df["atr_ma"] * df["adaptive_volatility_threshold"]
            )

        # RSI 필터
        rsi_filter = (df["rsi"] > 20) & (df["rsi"] < 80)

        # 최종 필터 적용
        final_filter = volume_filter & volatility_filter & rsi_filter
        df.loc[final_filter, "filtered_signal"] = df.loc[final_filter, "ema_signal"]

        # 적응형 홀딩 기간 적용
        df = self._apply_adaptive_holding_period(df)

        return df

    def _apply_adaptive_holding_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """적응형 홀딩 기간 적용"""
        df = df.copy()
        current_signal = 0
        signal_start = 0

        for i in range(len(df)):
            if (
                df["filtered_signal"].iloc[i] != 0
                and df["filtered_signal"].iloc[i] != current_signal
            ):
                holding_period = int(df["adaptive_holding_period"].iloc[i])
                if i - signal_start >= holding_period or current_signal == 0:
                    current_signal = df["filtered_signal"].iloc[i]
                    signal_start = i
                else:
                    df.loc[df.index[i], "filtered_signal"] = 0

        return df


class VWAPMACDScalpingStrategy(BaseStrategy):
    """VWAP + MACD 스켈핑 전략 - 1분 차트 기반 빠른 진입/청산"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        # VWAP 파라미터
        self.vwap_period = getattr(params, "vwap_period", 20)

        # MACD 파라미터 (1분 스켈핑용)
        self.macd_fast = getattr(params, "macd_fast", 12)
        self.macd_slow = getattr(params, "macd_slow", 26)
        self.macd_signal = getattr(params, "macd_signal", 9)

        # 스켈핑 파라미터
        self.volume_threshold = getattr(params, "volume_threshold", 1.5)
        self.atr_multiplier = getattr(params, "atr_multiplier", 1.0)
        self.min_holding_period = getattr(params, "min_holding_period", 2)
        self.max_holding_period = getattr(params, "max_holding_period", 10)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """VWAP + MACD 스켈핑 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # 1. VWAP 계산
        df["vwap"] = self._calculate_vwap(df)

        # 2. MACD 계산
        df["macd"], df["macd_signal"], df["macd_histogram"] = self._calculate_macd(df)

        # 3. ATR 계산
        if "atr" not in df.columns:
            df["atr"] = TechnicalIndicators.calculate_atr(df, 14)

        # 4. 신호 생성
        # 매수 신호: VWAP 상향 돌파 + MACD 골든 크로스
        buy_signal = (
            (df["close"] > df["vwap"])  # VWAP 상향 돌파
            & (df["macd"] > df["macd_signal"])  # MACD 골든 크로스
            & (
                df["macd"].shift(1) <= df["macd_signal"].shift(1)
            )  # 이전 봉에서는 크로스하지 않음
            & (df["close"] > df["close"].shift(1))  # 가격 상승
        )

        # 매도 신호: VWAP 하향 돌파 + MACD 데드 크로스
        sell_signal = (
            (df["close"] < df["vwap"])  # VWAP 하향 돌파
            & (df["macd"] < df["macd_signal"])  # MACD 데드 크로스
            & (
                df["macd"].shift(1) >= df["macd_signal"].shift(1)
            )  # 이전 봉에서는 크로스하지 않음
            & (df["close"] < df["close"].shift(1))  # 가격 하락
        )

        df.loc[buy_signal, "signal"] = 1
        df.loc[sell_signal, "signal"] = -1

        # 5. 거래량 필터
        volume_filter = pd.Series([True] * len(df), index=df.index)
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            volume_filter = df["volume"] > df["volume_ma"] * self.volume_threshold

        # 6. 변동성 필터
        volatility_filter = pd.Series([True] * len(df), index=df.index)
        if "atr" in df.columns:
            df["atr_ma"] = df["atr"].rolling(window=20).mean()
            volatility_filter = df["atr"] > df["atr_ma"] * 0.8

        # 7. 최종 신호 생성
        final_filter = volume_filter & volatility_filter
        df.loc[~final_filter, "signal"] = 0

        # 8. 스켈핑 홀딩 기간 적용
        df = self._apply_scalping_holding_period(df)

        return df

    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """VWAP 계산"""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        volume = (
            df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)
        )

        vwap = (typical_price * volume).rolling(
            window=self.vwap_period
        ).sum() / volume.rolling(window=self.vwap_period).sum()
        return vwap

    def _calculate_macd(
        self, df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD 계산"""
        ema_fast = df["close"].ewm(span=self.macd_fast).mean()
        ema_slow = df["close"].ewm(span=self.macd_slow).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.macd_signal).mean()
        histogram = macd - signal
        return macd, signal, histogram

    def _apply_scalping_holding_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """스켈핑 홀딩 기간 적용"""
        df = df.copy()
        current_signal = 0
        signal_start = 0

        for i in range(len(df)):
            if df["signal"].iloc[i] != 0 and df["signal"].iloc[i] != current_signal:
                if i - signal_start >= self.min_holding_period or current_signal == 0:
                    current_signal = df["signal"].iloc[i]
                    signal_start = i
                else:
                    df.loc[df.index[i], "signal"] = 0

            # 최대 홀딩 기간 체크
            if current_signal != 0 and i - signal_start >= self.max_holding_period:
                df.loc[df.index[i], "signal"] = 0
                current_signal = 0

        return df


class KeltnerRSIScalpingStrategy(BaseStrategy):
    """Keltner Channels + RSI 스켈핑 전략 - 변동성 기반 빠른 진입"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        # Keltner Channel 파라미터
        self.keltner_period = getattr(params, "keltner_period", 20)
        self.keltner_multiplier = getattr(params, "keltner_multiplier", 2.0)

        # RSI 파라미터
        self.rsi_period = getattr(params, "rsi_period", 14)
        self.rsi_oversold = getattr(params, "rsi_oversold", 30)
        self.rsi_overbought = getattr(params, "rsi_overbought", 70)

        # 스켈핑 파라미터
        self.volume_threshold = getattr(params, "volume_threshold", 1.3)
        self.min_holding_period = getattr(params, "min_holding_period", 2)
        self.max_holding_period = getattr(params, "max_holding_period", 8)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keltner Channels + RSI 스켈핑 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # 1. Keltner Channels 계산
        df["keltner_middle"] = df["close"].ewm(span=self.keltner_period).mean()
        if "atr" not in df.columns:
            df["atr"] = TechnicalIndicators.calculate_atr(df, self.keltner_period)
        df["keltner_upper"] = df["keltner_middle"] + (
            self.keltner_multiplier * df["atr"]
        )
        df["keltner_lower"] = df["keltner_middle"] - (
            self.keltner_multiplier * df["atr"]
        )

        # 2. RSI 계산
        if "rsi" not in df.columns:
            df["rsi"] = TechnicalIndicators.calculate_rsi(df["close"], self.rsi_period)

        # 3. 신호 생성
        # 매수 신호: Keltner 하단 이탈 + RSI 과매도에서 반등
        buy_signal = (
            (df["close"] < df["keltner_lower"])  # Keltner 하단 이탈
            & (df["rsi"] < self.rsi_oversold)  # RSI 과매도
            & (df["rsi"] > df["rsi"].shift(1))  # RSI 반등 시작
            & (df["close"] > df["close"].shift(1))  # 가격 반등
        )

        # 매도 신호: Keltner 상단 이탈 + RSI 과매수에서 하락
        sell_signal = (
            (df["close"] > df["keltner_upper"])  # Keltner 상단 이탈
            & (df["rsi"] > self.rsi_overbought)  # RSI 과매수
            & (df["rsi"] < df["rsi"].shift(1))  # RSI 하락 시작
            & (df["close"] < df["close"].shift(1))  # 가격 하락
        )

        df.loc[buy_signal, "signal"] = 1
        df.loc[sell_signal, "signal"] = -1

        # 4. 거래량 필터
        volume_filter = pd.Series([True] * len(df), index=df.index)
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            volume_filter = df["volume"] > df["volume_ma"] * self.volume_threshold

        # 5. 최종 신호 생성
        df.loc[~volume_filter, "signal"] = 0

        # 6. 스켈핑 홀딩 기간 적용
        df = self._apply_scalping_holding_period(df)

        return df

    def _apply_scalping_holding_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """스켈핑 홀딩 기간 적용"""
        df = df.copy()
        current_signal = 0
        signal_start = 0

        for i in range(len(df)):
            if df["signal"].iloc[i] != 0 and df["signal"].iloc[i] != current_signal:
                if i - signal_start >= self.min_holding_period or current_signal == 0:
                    current_signal = df["signal"].iloc[i]
                    signal_start = i
                else:
                    df.loc[df.index[i], "signal"] = 0

            # 최대 홀딩 기간 체크
            if current_signal != 0 and i - signal_start >= self.max_holding_period:
                df.loc[df.index[i], "signal"] = 0
                current_signal = 0

        return df


class AbsorptionScalpingStrategy(BaseStrategy):
    """흡수(Absorption) 패턴 기반 스켈핑 전략 - 오더플로우 분석"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        # 흡수 패턴 파라미터
        self.absorption_lookback = getattr(params, "absorption_lookback", 5)
        self.volume_ratio_threshold = getattr(params, "volume_ratio_threshold", 2.0)
        self.price_rejection_threshold = getattr(
            params, "price_rejection_threshold", 0.005
        )

        # 확정 조건 파라미터
        self.fibonacci_levels = getattr(
            params, "fibonacci_levels", [0.236, 0.382, 0.618]
        )
        self.ema_short = getattr(params, "ema_short", 5)
        self.ema_long = getattr(params, "ema_long", 10)
        self.macd_fast = getattr(params, "macd_fast", 12)
        self.macd_slow = getattr(params, "macd_slow", 26)

        # 스켈핑 파라미터
        self.min_holding_period = getattr(params, "min_holding_period", 2)
        self.max_holding_period = getattr(params, "max_holding_period", 6)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """흡수 패턴 기반 스켈핑 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # 1. 흡수 패턴 감지
        df["absorption_pattern"] = self._detect_absorption_pattern(df)

        # 2. 확정 조건 계산
        df["fibonacci_confirmed"] = self._check_fibonacci_levels(df)
        df["ema_cross_confirmed"] = self._check_ema_crossing(df)
        df["macd_confirmed"] = self._check_macd_convergence(df)

        # 3. 신호 생성
        # 매수 신호: 흡수 패턴 + 확정 조건
        buy_signal = (
            (df["absorption_pattern"] == 1)  # 흡수 패턴 감지
            & (
                df["fibonacci_confirmed"]
                | df["ema_cross_confirmed"]
                | df["macd_confirmed"]
            )  # 확정 조건 중 하나
            & (df["close"] > df["close"].shift(1))  # 가격 상승
        )

        # 매도 신호: 반대 흡수 패턴
        sell_signal = (
            (df["absorption_pattern"] == -1)  # 반대 흡수 패턴
            & (
                df["fibonacci_confirmed"]
                | df["ema_cross_confirmed"]
                | df["macd_confirmed"]
            )  # 확정 조건 중 하나
            & (df["close"] < df["close"].shift(1))  # 가격 하락
        )

        df.loc[buy_signal, "signal"] = 1
        df.loc[sell_signal, "signal"] = -1

        # 4. 스켈핑 홀딩 기간 적용
        df = self._apply_scalping_holding_period(df)

        return df

    def _detect_absorption_pattern(self, df: pd.DataFrame) -> pd.Series:
        """흡수 패턴 감지"""
        absorption = pd.Series(0, index=df.index)

        for i in range(self.absorption_lookback, len(df)):
            # 최근 봉들의 패턴 분석
            recent_highs = df["high"].iloc[i - self.absorption_lookback : i + 1]
            recent_lows = df["low"].iloc[i - self.absorption_lookback : i + 1]
            recent_volumes = (
                df["volume"].iloc[i - self.absorption_lookback : i + 1]
                if "volume" in df.columns
                else pd.Series(1, index=recent_highs.index)
            )

            # 매도 압력 감지 (높은 고가, 높은 거래량)
            selling_pressure = (recent_highs.max() > recent_highs.mean() * 1.01) and (
                recent_volumes.mean()
                > recent_volumes.rolling(window=20).mean().iloc[-1]
                * self.volume_ratio_threshold
            )

            # 가격 거부 패턴 (고가에서 하락)
            price_rejection = (
                df["close"].iloc[i]
                < df["high"].iloc[i] * (1 - self.price_rejection_threshold)
            ) and (df["close"].iloc[i] > df["close"].iloc[i - 1])

            # 흡수 패턴: 매도 압력이 있지만 가격이 상승
            if selling_pressure and price_rejection:
                absorption.iloc[i] = 1

            # 반대 흡수 패턴: 매수 압력이 있지만 가격이 하락
            buying_pressure = (recent_lows.min() < recent_lows.mean() * 0.99) and (
                recent_volumes.mean()
                > recent_volumes.rolling(window=20).mean().iloc[-1]
                * self.volume_ratio_threshold
            )

            reverse_rejection = (
                df["close"].iloc[i]
                > df["low"].iloc[i] * (1 + self.price_rejection_threshold)
            ) and (df["close"].iloc[i] < df["close"].iloc[i - 1])

            if buying_pressure and reverse_rejection:
                absorption.iloc[i] = -1

        return absorption

    def _check_fibonacci_levels(self, df: pd.DataFrame) -> pd.Series:
        """피보나치 레벨 확인"""
        confirmed = pd.Series(False, index=df.index)

        for i in range(20, len(df)):
            # 최근 고점/저점 찾기
            recent_high = df["high"].iloc[i - 20 : i + 1].max()
            recent_low = df["low"].iloc[i - 20 : i + 1].min()
            current_price = df["close"].iloc[i]

            # 피보나치 레벨 계산
            price_range = recent_high - recent_low
            for level in self.fibonacci_levels:
                fib_level = recent_low + (price_range * level)
                if abs(current_price - fib_level) / current_price < 0.01:  # 1% 이내
                    confirmed.iloc[i] = True
                    break

        return confirmed

    def _check_ema_crossing(self, df: pd.DataFrame) -> pd.Series:
        """EMA 크로싱 확인"""
        df["ema_short"] = df["close"].ewm(span=self.ema_short).mean()
        df["ema_long"] = df["close"].ewm(span=self.ema_long).mean()

        # 골든 크로스 또는 데드 크로스
        golden_cross = (df["ema_short"] > df["ema_long"]) & (
            df["ema_short"].shift(1) <= df["ema_long"].shift(1)
        )
        dead_cross = (df["ema_short"] < df["ema_long"]) & (
            df["ema_short"].shift(1) >= df["ema_long"].shift(1)
        )

        return golden_cross | dead_cross

    def _check_macd_convergence(self, df: pd.DataFrame) -> pd.Series:
        """MACD 0 수렴 확인"""
        ema_fast = df["close"].ewm(span=self.macd_fast).mean()
        ema_slow = df["close"].ewm(span=self.macd_slow).mean()
        macd = ema_fast - ema_slow

        # MACD가 0에 수렴하는 패턴
        convergence = abs(macd) < abs(macd).rolling(window=10).mean() * 0.5
        return convergence

    def _apply_scalping_holding_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """스켈핑 홀딩 기간 적용"""
        df = df.copy()
        current_signal = 0
        signal_start = 0

        for i in range(len(df)):
            if df["signal"].iloc[i] != 0 and df["signal"].iloc[i] != current_signal:
                if i - signal_start >= self.min_holding_period or current_signal == 0:
                    current_signal = df["signal"].iloc[i]
                    signal_start = i
                else:
                    df.loc[df.index[i], "signal"] = 0

            # 최대 홀딩 기간 체크
            if current_signal != 0 and i - signal_start >= self.max_holding_period:
                df.loc[df.index[i], "signal"] = 0
                current_signal = 0

        return df


class RSIBollingerScalpingStrategy(BaseStrategy):
    """RSI + Bollinger Bands 스켈핑 전략 - 1분 차트 최적화"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        # RSI 파라미터 (1분 스켈핑용)
        self.rsi_period = getattr(params, "rsi_period", 4)  # 짧은 기간
        self.rsi_oversold = getattr(params, "rsi_oversold", 20)
        self.rsi_overbought = getattr(params, "rsi_overbought", 80)

        # Bollinger Bands 파라미터
        self.bb_period = getattr(params, "bb_period", 20)
        self.bb_std = getattr(params, "bb_std", 2.0)

        # 스켈핑 파라미터
        self.volume_threshold = getattr(params, "volume_threshold", 1.2)
        self.min_holding_period = getattr(params, "min_holding_period", 2)
        self.max_holding_period = getattr(params, "max_holding_period", 8)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI + Bollinger Bands 스켈핑 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # 1. RSI 계산 (짧은 기간)
        if "rsi" not in df.columns:
            df["rsi"] = TechnicalIndicators.calculate_rsi(df["close"], self.rsi_period)

        # 2. Bollinger Bands 계산
        df["bb_middle"] = df["close"].rolling(window=self.bb_period).mean()
        bb_std = df["close"].rolling(window=self.bb_period).std()
        df["bb_upper"] = df["bb_middle"] + (self.bb_std * bb_std)
        df["bb_lower"] = df["bb_middle"] - (self.bb_std * bb_std)

        # 3. 신호 생성
        # 매수 신호: RSI 과매도 이탈 + Bollinger 하단 터치
        buy_signal = (
            (df["rsi"].shift(1) < self.rsi_oversold)  # 이전에 과매도
            & (df["rsi"] > self.rsi_oversold)  # 현재 과매도 이탈
            & (df["close"] <= df["bb_lower"])  # Bollinger 하단 터치
            & (df["close"] > df["close"].shift(1))  # 가격 반등
        )

        # 매도 신호: RSI 과매수 이탈 + Bollinger 상단 터치
        sell_signal = (
            (df["rsi"].shift(1) > self.rsi_overbought)  # 이전에 과매수
            & (df["rsi"] < self.rsi_overbought)  # 현재 과매수 이탈
            & (df["close"] >= df["bb_upper"])  # Bollinger 상단 터치
            & (df["close"] < df["close"].shift(1))  # 가격 하락
        )

        df.loc[buy_signal, "signal"] = 1
        df.loc[sell_signal, "signal"] = -1

        # 4. 거래량 필터
        volume_filter = pd.Series([True] * len(df), index=df.index)
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            volume_filter = df["volume"] > df["volume_ma"] * self.volume_threshold

        # 5. 최종 신호 생성
        df.loc[~volume_filter, "signal"] = 0

        # 6. 스켈핑 홀딩 기간 적용
        df = self._apply_scalping_holding_period(df)

        return df

    def _apply_scalping_holding_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """스켈핑 홀딩 기간 적용"""
        df = df.copy()
        current_signal = 0
        signal_start = 0

        for i in range(len(df)):
            if df["signal"].iloc[i] != 0 and df["signal"].iloc[i] != current_signal:
                if i - signal_start >= self.min_holding_period or current_signal == 0:
                    current_signal = df["signal"].iloc[i]
                    signal_start = i
                else:
                    df.loc[df.index[i], "signal"] = 0

            # 최대 홀딩 기간 체크
            if current_signal != 0 and i - signal_start >= self.max_holding_period:
                df.loc[df.index[i], "signal"] = 0
                current_signal = 0

        return df


class MeanReversionStrategy(BaseStrategy):
    """Bollinger Band 기반 Mean Reversion(평균회귀) 전략"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.bb_period = getattr(params, "bb_period", 20)
        self.bb_std = getattr(params, "bb_std", 2.0)
        self.entry_zscore = getattr(
            params, "entry_zscore", 2.0
        )  # 진입 임계값 (표준편차)
        self.exit_zscore = getattr(params, "exit_zscore", 0.5)  # 청산 임계값 (표준편차)
        self.min_holding_period = getattr(params, "min_holding_period", 3)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0

        # 볼린저 밴드 계산
        df["bb_middle"] = df["close"].rolling(window=self.bb_period).mean()
        bb_std = df["close"].rolling(window=self.bb_period).std()
        df["bb_upper"] = df["bb_middle"] + (self.bb_std * bb_std)
        df["bb_lower"] = df["bb_middle"] - (self.bb_std * bb_std)

        # z-score 계산 (현재가와 밴드 중심의 표준편차 단위 거리)
        df["zscore"] = (df["close"] - df["bb_middle"]) / bb_std

        # 진입 신호: 하단 밴드 이탈(과매도) → 매수, 상단 밴드 이탈(과매수) → 매도
        df.loc[df["zscore"] < -self.entry_zscore, "signal"] = 1
        df.loc[df["zscore"] > self.entry_zscore, "signal"] = -1

        # 청산 신호: zscore가 0에 근접(평균 복귀)
        exit_long = (df["signal"].shift(1) == 1) & (df["zscore"] > -self.exit_zscore)
        exit_short = (df["signal"].shift(1) == -1) & (df["zscore"] < self.exit_zscore)
        df.loc[exit_long | exit_short, "signal"] = 0

        # 최소 홀딩 기간 적용
        df = self._apply_min_holding_period(df)
        return df

    def _apply_min_holding_period(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        current_signal = 0
        signal_start = 0
        for i in range(len(df)):
            if df["signal"].iloc[i] != 0 and df["signal"].iloc[i] != current_signal:
                if i - signal_start >= self.min_holding_period or current_signal == 0:
                    current_signal = df["signal"].iloc[i]
                    signal_start = i
                else:
                    df.loc[df.index[i], "signal"] = 0
            # 평균 복귀 후 신호 종료
            if current_signal != 0 and (
                (current_signal == 1 and df["zscore"].iloc[i] > -self.exit_zscore)
                or (current_signal == -1 and df["zscore"].iloc[i] < self.exit_zscore)
            ):
                df.loc[df.index[i], "signal"] = 0
                current_signal = 0
        return df


# [장기] 연 1회 리밸런싱
class FixedWeightRebalanceStrategy(BaseStrategy):
    """2-ETF 고정비중 연 1회 리밸런싱 전략"""

    def __init__(
        self,
        params: StrategyParams,
        symbols: list,
        weights: list,
        rebalance_month: int = 1,
    ):
        super().__init__(params)
        self.symbols = symbols
        # numpy 배열일 경우 리스트로 변환
        try:
            import numpy as np

            if isinstance(weights, np.ndarray):
                weights = weights.tolist()
        except ImportError:
            pass
        if not isinstance(weights, list):
            weights = list(weights)
        self.weights = weights
        self.rebalance_month = rebalance_month  # 1=1월, 7=7월 등

    def generate_signals(
        self, data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """각 종목별로 고정 비중, 연 1회 리밸런싱 신호 생성"""
        signals = {}
        import pandas as pd

        # 인자 타입 체크: DataFrame이 들어오면 dict로 변환
        if isinstance(data_dict, pd.DataFrame):
            df = data_dict
            # 'datetime' 컬럼이 없으면 인덱스에서 복원 시도
            if "datetime" not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.copy()
                    df["datetime"] = df.index
                else:
                    df = df.reset_index()
                    if "datetime" not in df.columns:
                        df["datetime"] = pd.to_datetime("today")
            data_dict = {"single_symbol": df}
            self.symbols = list(data_dict.keys())
        # 공통 날짜 추출
        common_dates = set.intersection(
            *[set(df["datetime"]) for df in data_dict.values()]
        )
        common_dates = sorted(list(common_dates))
        for idx, symbol in enumerate(self.symbols):
            df = data_dict[symbol].copy()
            # 'datetime' 컬럼이 없으면 인덱스에서 복원 시도
            if "datetime" not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex):
                    df["datetime"] = df.index
                else:
                    df = df.reset_index()
                    if "datetime" not in df.columns:
                        df["datetime"] = pd.to_datetime("today")
            df = df[df["datetime"].isin(common_dates)].sort_values("datetime")
            df["signal"] = 0
            # weights가 numpy array일 수 있으니 list로 변환
            weight = (
                self.weights[idx]
                if isinstance(self.weights, (list, tuple))
                else float(self.weights)
            )
            df["weight"] = weight
            # 연 1회 리밸런싱: 1월 첫 거래일에만 신호 1, 나머지는 0
            df["rebalance"] = df["datetime"].dt.month == self.rebalance_month
            if df["rebalance"].any():
                first_rebalance_day = df[df["rebalance"]]["datetime"].dt.day.min()
                df["rebalance"] = df["rebalance"] & (
                    df["datetime"].dt.day == first_rebalance_day
                )
            df.loc[df["rebalance"], "signal"] = 1
            df.loc[df["rebalance"], "weight"] = weight
            signals[symbol] = df
        return signals


# [장기] 20일 주기 리밸런싱
class ETFMomentumRotationStrategy(BaseStrategy):
    """ETF 모멘텀 로테이션 전략"""

    def __init__(
        self,
        params: StrategyParams,
        top_n: int = 2,
        lookback_period: int = 20,
        rebalance_period: int = 20,
    ):
        super().__init__(params)
        self.top_n = top_n  # 상위 N개 선택
        self.lookback_period = lookback_period  # 모멘텀 계산 기간
        self.rebalance_period = rebalance_period  # 리밸런싱 주기

    def generate_signals(
        self, data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        import pandas as pd

        # DataFrame이 들어오면 dict로 변환
        if isinstance(data_dict, pd.DataFrame):
            df = data_dict
            if "datetime" not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.copy()
                    df["datetime"] = df.index
                else:
                    df = df.reset_index()
                    if "datetime" not in df.columns:
                        df["datetime"] = pd.to_datetime("today")
            data_dict = {"single_symbol": df}
            symbols = list(data_dict.keys())
        else:
            symbols = list(data_dict.keys())
        # 공통 날짜 추출
        common_dates = set.intersection(
            *[set(df["datetime"]) for df in data_dict.values()]
        )
        common_dates = sorted(list(common_dates))
        signals = {}
        for symbol in symbols:
            df = data_dict[symbol].copy()
            if "datetime" not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex):
                    df["datetime"] = df.index
                else:
                    df = df.reset_index()
                    if "datetime" not in df.columns:
                        df["datetime"] = pd.to_datetime("today")
            df = df[df["datetime"].isin(common_dates)].sort_values("datetime")
            df["momentum"] = df["close"].pct_change(self.lookback_period)
            df["signal"] = 0
            df["weight"] = 0
            signals[symbol] = df
        for i in range(self.lookback_period, len(common_dates)):
            if i % self.rebalance_period == 0:
                current_date = common_dates[i]
                momentums = [
                    (symbol, signals[symbol].iloc[i]["momentum"]) for symbol in symbols
                ]
                top_symbols = [
                    symbol
                    for symbol, _ in sorted(
                        momentums, key=lambda x: x[1], reverse=True
                    )[: self.top_n]
                ]
                weight_per_symbol = 1.0 / len(top_symbols) if top_symbols else 0.0
                for symbol in symbols:
                    if i < len(signals[symbol]):
                        if symbol in top_symbols:
                            signals[symbol].iloc[
                                i, signals[symbol].columns.get_loc("signal")
                            ] = 1.0
                            signals[symbol].iloc[
                                i, signals[symbol].columns.get_loc("weight")
                            ] = weight_per_symbol
                        else:
                            signals[symbol].iloc[
                                i, signals[symbol].columns.get_loc("signal")
                            ] = 0.0
                            signals[symbol].iloc[
                                i, signals[symbol].columns.get_loc("weight")
                            ] = 0.0
            else:
                for symbol in symbols:
                    if i < len(signals[symbol]) and i > 0:
                        prev_signal = signals[symbol].iloc[i - 1]["signal"]
                        prev_weight = signals[symbol].iloc[i - 1]["weight"]
                        signals[symbol].iloc[
                            i, signals[symbol].columns.get_loc("signal")
                        ] = prev_signal
                        signals[symbol].iloc[
                            i, signals[symbol].columns.get_loc("weight")
                        ] = prev_weight
        return signals


# [장기] 장기 추세 기반
class TrendFollowingMA200Strategy(BaseStrategy):
    """200일 이동평균 돌파/이탈 스위칭 전략"""

    def __init__(self, params: StrategyParams, ma_period: int = 200):
        super().__init__(params)
        self.ma_period = ma_period

    def generate_signals(
        self, data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        import pandas as pd

        # DataFrame이 들어오면 dict로 변환
        if isinstance(data_dict, pd.DataFrame):
            df = data_dict
            if "datetime" not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.copy()
                    df["datetime"] = df.index
                else:
                    df = df.reset_index()
                    if "datetime" not in df.columns:
                        df["datetime"] = pd.to_datetime("today")
            data_dict = {"single_symbol": df}
            symbols = list(data_dict.keys())
        else:
            symbols = list(data_dict.keys())
        signals = {}
        for symbol in symbols:
            df = data_dict[symbol].copy()
            if "datetime" not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex):
                    df["datetime"] = df.index
                else:
                    df = df.reset_index()
                    if "datetime" not in df.columns:
                        df["datetime"] = pd.to_datetime("today")
            df["ma"] = df["close"].rolling(self.ma_period).mean()
            df["signal"] = 0
            df.loc[df["close"] > df["ma"], "signal"] = 1
            df.loc[df["close"] <= df["ma"], "signal"] = 0
            df["weight"] = df["signal"]
            signals[symbol] = df
        return signals


# [장기] 20일 주기 리밸런싱
class ReturnStackingStrategy(BaseStrategy):
    """Return Stacking 전략"""

    def __init__(
        self,
        params: StrategyParams,
        symbols: list,
        weights: list,
        rebalance_period: int = 20,
    ):
        super().__init__(params)
        self.symbols = symbols
        try:
            import numpy as np

            if isinstance(weights, np.ndarray):
                weights = weights.tolist()
        except ImportError:
            pass
        if not isinstance(weights, list):
            weights = list(weights)
        self.weights = weights
        self.rebalance_period = rebalance_period

    def generate_signals(
        self, data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        import pandas as pd

        # DataFrame이 들어오면 dict로 변환
        if isinstance(data_dict, pd.DataFrame):
            df = data_dict
            if "datetime" not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.copy()
                    df["datetime"] = df.index
                else:
                    df = df.reset_index()
                    if "datetime" not in df.columns:
                        df["datetime"] = pd.to_datetime("today")
            data_dict = {"single_symbol": df}
            self.symbols = list(data_dict.keys())
        # 공통 날짜 추출
        common_dates = set.intersection(
            *[set(df["datetime"]) for df in data_dict.values()]
        )
        common_dates = sorted(list(common_dates))
        signals = {}
        for idx, symbol in enumerate(self.symbols):
            df = data_dict[symbol].copy()
            if "datetime" not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex):
                    df["datetime"] = df.index
                else:
                    df = df.reset_index()
                    if "datetime" not in df.columns:
                        df["datetime"] = pd.to_datetime("today")
            df = df[df["datetime"].isin(common_dates)].sort_values("datetime")
            df["signal"] = 1
            df["weight"] = self.weights[idx]
            signals[symbol] = df
        return signals


class StrategyManager:
    """전략 관리 및 백테스팅 클래스"""

    def __init__(self):
        self.strategies = {}
        self.results = {}

    def add_strategy(self, name: str, strategy: BaseStrategy):
        """전략 추가"""
        self.strategies[name] = strategy

    def load_data(self, symbol: str, file_path: str) -> pd.DataFrame:
        """CSV 파일에서 데이터 로드"""
        try:
            df = pd.read_csv(file_path)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)
            return df
        except Exception as e:
            logger.error(f"데이터 로드 오류: {e}")
            raise

    def run_strategy(
        self, strategy_name: str, df: pd.DataFrame, params: StrategyParams
    ) -> Dict[str, Any]:
        """전략 실행"""
        if strategy_name not in self.strategies:
            raise ValueError(f"전략 '{strategy_name}'이 등록되지 않았습니다.")

        strategy = self.strategies[strategy_name]

        # 기술적 지표 계산
        df_with_indicators = TechnicalIndicators.calculate_all_indicators(df, params)

        # 신호 생성
        if strategy_name == "risk_parity_leverage":
            # 리스크 패리티 전략은 별도 처리
            signals = strategy.generate_signals({symbol: df_with_indicators})
        else:
            signals = strategy.generate_signals(df_with_indicators)

        return {
            "strategy_name": strategy_name,
            "data": df_with_indicators,
            "signals": signals,
            "params": params,
        }

    def optimize_parameters(
        self, strategy_name: str, df: pd.DataFrame, param_ranges: Dict[str, List]
    ) -> StrategyParams:
        """하이퍼파라미터 최적화"""
        # 간단한 그리드 서치 구현
        best_params = None
        best_score = float("-inf")

        # 파라미터 조합 생성
        import itertools

        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())

        for combination in itertools.product(*param_values):
            params_dict = dict(zip(param_names, combination))
            params = StrategyParams(**params_dict)

            try:
                result = self.run_strategy(strategy_name, df, params)
                # 간단한 성과 지표 (수익률 기준)
                if "signals" in result and isinstance(result["signals"], pd.DataFrame):
                    signals = result["signals"]
                    if "signal" in signals.columns:
                        # 신호 기반 수익률 계산
                        returns = df["close"].pct_change()
                        strategy_returns = signals["signal"].shift(1) * returns
                        total_return = strategy_returns.sum()

                        if total_return > best_score:
                            best_score = total_return
                            best_params = params
            except Exception as e:
                logger.warning(f"파라미터 조합 {combination} 실행 중 오류: {e}")
                continue

        return best_params if best_params else StrategyParams()


def main():
    """메인 실행 함수"""
    # 전략 매니저 초기화
    manager = StrategyManager()

    # 기본 파라미터
    params = StrategyParams()

    # 전략들 등록
    manager.add_strategy("dual_momentum", DualMomentumStrategy(params))
    manager.add_strategy(
        "volatility_breakout", VolatilityAdjustedBreakoutStrategy(params)
    )

    # 리스크 패리티 전략 (여러 종목 필요)
    risk_parity_params = StrategyParams()
    manager.add_strategy(
        "risk_parity_leverage",
        RiskParityLeverageStrategy(risk_parity_params, ["NVDL", "TSLL", "CONL"]),
    )

    # 추가 전략 등록
    manager.add_strategy("stochastic", StochasticStrategy(params))
    manager.add_strategy("williams_r", WilliamsRStrategy(params))
    manager.add_strategy("cci", CCIStrategy(params))
    manager.add_strategy("whipsaw_prevention", WhipsawPreventionStrategy(params))
    manager.add_strategy("donchian_rsi_whipsaw", DonchianRSIWhipsawStrategy(params))
    manager.add_strategy(
        "volatility_filtered_breakout", VolatilityFilteredBreakoutStrategy(params)
    )
    manager.add_strategy(
        "multi_timeframe_whipsaw", MultiTimeframeWhipsawStrategy(params)
    )
    manager.add_strategy("adaptive_whipsaw", AdaptiveWhipsawStrategy(params))
    manager.add_strategy("cci_bollinger", CCIBollingerStrategy(params))
    manager.add_strategy("stoch_donchian", StochDonchianStrategy(params))
    manager.add_strategy("vwap_macd", VWAPMACDScalpingStrategy(params))
    manager.add_strategy("keltner_rsi", KeltnerRSIScalpingStrategy(params))
    manager.add_strategy("absorption", AbsorptionScalpingStrategy(params))
    manager.add_strategy("rsi_bollinger", RSIBollingerScalpingStrategy(params))
    manager.add_strategy("mean_reversion", MeanReversionStrategy(params))
    manager.add_strategy("trend_following_ma200", TrendFollowingMA200Strategy(params))
    manager.add_strategy(
        "return_stacking",
        ReturnStackingStrategy(params, ["NVDL", "TSLL", "CONL"], [0.33, 0.33, 0.34]),
    )

    print("퀀트 전략 클래스가 성공적으로 초기화되었습니다.")
    print("사용 가능한 전략:")
    print("1. dual_momentum - 추세⇄평균회귀 듀얼 모멘텀")
    print("2. volatility_breakout - 변동성 조정 채널 브레이크아웃")
    print("3. risk_parity_leverage - 퀀트 리스크 패리티 + 레버리지 ETF")
    print("4. swing_ema - 중기 이동평균 돌파 스윙 트레이딩")
    print("5. swing_rsi - RSI 리버설 스윙 트레이딩")
    print("6. swing_donchian - Donchian Channel 돌파 스윙 트레이딩")
    print("7. stochastic - Stochastic Oscillator 기반 전략")
    print("8. williams_r - Williams %R 기반 전략")
    print("9. cci - CCI 기반 전략")
    print("10. whipsaw_prevention - 휩쏘 방지 전략")
    print("11. donchian_rsi_whipsaw - Donchian + RSI 듀얼 모멘텀 휩쏘 방지 전략")
    print(
        "12. volatility_filtered_breakout - 변동성 필터링 브레이크아웃 휩쏘 방지 전략"
    )
    print("13. multi_timeframe_whipsaw - 다중 시간 프레임 휩쏘 방지 전략")
    print("14. adaptive_whipsaw - 적응형 휩쏘 방지 전략")
    print("15. cci_bollinger - CCI + Bollinger Band 결합 전략")
    print("16. stoch_donchian - Stoch %K 교차 + Donchian 채널 전략")
    print("17. vwap_macd - VWAP + MACD 스켈핑 전략")
    print("18. keltner_rsi - Keltner Channels + RSI 스켈핑 전략")
    print("19. absorption - 흡수(Absorption) 패턴 기반 스켈핑 전략")
    print("20. rsi_bollinger - RSI + Bollinger Bands 스켈핑 전략")
    print("21. mean_reversion - Bollinger Band 기반 Mean Reversion(평균회귀) 전략")
    print("22. trend_following_ma200 - 200일 이동평균 돌파/이탈 스위칭 전략")
    print("23. return_stacking - Return Stacking 전략")


# ============================================================================
# 새로운 스윙 전략들 (New Swing Strategies)
# ============================================================================


# [스윙] 1~3주 보유 - 돌파 전략
class SwingBreakoutStrategy(BaseStrategy):
    """스윙 돌파 전략 - 저항선/지지선 돌파 시 매매"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.strategy_type = "single_asset"
        # 돌파 전략 파라미터
        self.resistance_period = getattr(params, "resistance_period", 20)
        self.support_period = getattr(params, "support_period", 20)
        self.breakout_threshold = getattr(params, "breakout_threshold", 0.02)  # 2% 돌파
        self.volume_confirmation = getattr(params, "volume_confirmation", True)
        self.volume_multiplier = getattr(params, "volume_multiplier", 1.5)
        self.confirmation_candles = getattr(params, "confirmation_candles", 2)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """돌파 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # 저항선/지지선 계산
        df["resistance"] = df["high"].rolling(window=self.resistance_period).max()
        df["support"] = df["low"].rolling(window=self.support_period).min()

        # 상향 돌파 신호 (저항선 돌파) - 조건 완화
        resistance_breakout = df["close"] > df["resistance"].shift(1)

        # 하향 돌파 신호 (지지선 돌파) - 조건 완화
        support_breakdown = df["close"] < df["support"].shift(1)

        # 볼륨 확인 - 조건 완화
        if self.volume_confirmation and "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            volume_confirmed = df["volume"] > df["volume_ma"] * 0.5  # 0.5배로 완화
        else:
            volume_confirmed = pd.Series([True] * len(df), index=df.index)

        # 돌파 확인 (연속 캔들 확인)
        confirmed_breakout = resistance_breakout & volume_confirmed
        confirmed_breakdown = support_breakdown & volume_confirmed

        # 신호 생성
        df.loc[confirmed_breakout, "signal"] = 1  # 상향 돌파 시 매수
        df.loc[confirmed_breakdown, "signal"] = -1  # 하향 돌파 시 매도

        return df


# [스윙] 1~3주 보유 - 조정 후 진입 전략
class SwingPullbackEntryStrategy(BaseStrategy):
    """스윙 조정 후 진입 전략 - 추세 중 일시 하락 시 매수"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.strategy_type = "single_asset"
        # 조정 후 진입 파라미터
        self.trend_period = getattr(params, "trend_period", 50)
        self.pullback_period = getattr(params, "pullback_period", 10)
        self.pullback_threshold = getattr(params, "pullback_threshold", 0.05)  # 5% 조정
        self.rsi_oversold = getattr(params, "rsi_oversold", 35)
        self.rsi_period = getattr(params, "rsi_period", 14)
        self.min_trend_strength = getattr(params, "min_trend_strength", 0.02)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """조정 후 진입 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # 추세 계산 (EMA 기반)
        df["ema_trend"] = df["close"].ewm(span=self.trend_period).mean()
        df["trend_slope"] = df["ema_trend"].diff(self.trend_period // 2)

        # 상승 추세 확인
        uptrend = df["trend_slope"] > self.min_trend_strength

        # 조정 확인 (단기 하락)
        df["short_ma"] = df["close"].rolling(window=self.pullback_period).mean()
        pullback = df["close"] < df["short_ma"] * (1 - self.pullback_threshold)

        # RSI 과매도 확인
        if "rsi" not in df.columns:
            df["rsi"] = TechnicalIndicators.calculate_rsi(df["close"], self.rsi_period)
        oversold = df["rsi"] < self.rsi_oversold

        # 조정 후 진입 신호 (상승 추세 + 조정 + 과매도)
        entry_signal = uptrend & pullback & oversold

        # 신호 생성
        df.loc[entry_signal, "signal"] = 1

        return df


# [스윙] 1~3주 보유 - 캔들 패턴 기반 전략
class SwingCandlePatternStrategy(BaseStrategy):
    """스윙 캔들 패턴 기반 전략 - 장악형, 십자형 등 활용"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.strategy_type = "single_asset"
        # 캔들 패턴 파라미터
        self.body_threshold = getattr(params, "body_threshold", 0.6)  # 몸통 비율
        self.shadow_threshold = getattr(params, "shadow_threshold", 0.3)  # 그림자 비율
        self.confirmation_period = getattr(params, "confirmation_period", 3)
        self.volume_confirmation = getattr(params, "volume_confirmation", True)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """캔들 패턴 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # 캔들 구성요소 계산
        df["body"] = abs(df["close"] - df["open"])
        df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["total_range"] = df["high"] - df["low"]

        # 몸통과 그림자 비율
        df["body_ratio"] = df["body"] / df["total_range"]
        df["upper_shadow_ratio"] = df["upper_shadow"] / df["total_range"]
        df["lower_shadow_ratio"] = df["lower_shadow"] / df["total_range"]

        # 캔들 패턴 감지
        # 1. 장악형 (Engulfing)
        bullish_engulfing = (
            (df["body_ratio"].shift(1) < self.body_threshold)  # 전일 작은 몸통
            & (df["body_ratio"] > self.body_threshold)  # 당일 큰 몸통
            & (df["close"] > df["open"])  # 당일 양봉
            & (df["close"] > df["high"].shift(1))  # 전일 고가 돌파
            & (df["open"] < df["low"].shift(1))  # 전일 저가 하향 돌파
        )

        bearish_engulfing = (
            (df["body_ratio"].shift(1) < self.body_threshold)
            & (df["body_ratio"] > self.body_threshold)
            & (df["close"] < df["open"])  # 당일 음봉
            & (df["close"] < df["low"].shift(1))
            & (df["open"] > df["high"].shift(1))
        )

        # 2. 십자형 (Doji)
        doji = df["body_ratio"] < 0.1  # 몸통이 매우 작음

        # 3. 망치형 (Hammer) / 교수형 (Hanging Man)
        hammer = (
            (df["lower_shadow_ratio"] > self.shadow_threshold)
            & (df["upper_shadow_ratio"] < 0.1)
            & (df["body_ratio"] < 0.3)
        )

        # 볼륨 확인
        if self.volume_confirmation and "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            volume_confirmed = df["volume"] > df["volume_ma"]
        else:
            volume_confirmed = pd.Series([True] * len(df), index=df.index)

        # 신호 생성
        df.loc[bullish_engulfing & volume_confirmed, "signal"] = 1  # 상승 신호
        df.loc[bearish_engulfing & volume_confirmed, "signal"] = -1  # 하락 신호

        # 십자형은 추세 방향에 따라 신호 생성
        uptrend = df["close"] > df["close"].rolling(window=20).mean()
        df.loc[doji & uptrend & volume_confirmed, "signal"] = 1
        df.loc[doji & ~uptrend & volume_confirmed, "signal"] = -1

        return df


# [스윙] 1~3주 보유 - 볼린저 밴드 스윙 전략
class SwingBollingerBandStrategy(BaseStrategy):
    """스윙 볼린저 밴드 전략 - 밴드 하단 터치 시 매수, 상단 터치 시 매도"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.strategy_type = "single_asset"
        # 볼린저 밴드 파라미터
        self.bb_period = getattr(params, "bb_period", 20)
        self.bb_std = getattr(params, "bb_std", 2.0)
        self.rsi_period = getattr(params, "rsi_period", 14)
        self.rsi_oversold = getattr(params, "rsi_oversold", 30)
        self.rsi_overbought = getattr(params, "rsi_overbought", 70)
        self.confirmation_candles = getattr(params, "confirmation_candles", 2)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """볼린저 밴드 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # 볼린저 밴드 계산
        df["bb_middle"] = df["close"].rolling(window=self.bb_period).mean()
        bb_std = df["close"].rolling(window=self.bb_period).std()
        df["bb_upper"] = df["bb_middle"] + (self.bb_std * bb_std)
        df["bb_lower"] = df["bb_middle"] - (self.bb_std * bb_std)

        # RSI 계산
        if "rsi" not in df.columns:
            df["rsi"] = TechnicalIndicators.calculate_rsi(df["close"], self.rsi_period)
        oversold = df["rsi"] < self.rsi_oversold
        overbought = df["rsi"] > self.rsi_overbought

        # 밴드 하단 터치 + RSI 과매도 = 매수 신호
        lower_band_touch = df["close"] <= df["bb_lower"]
        buy_signal = lower_band_touch & oversold

        # 밴드 상단 터치 + RSI 과매수 = 매도 신호
        upper_band_touch = df["close"] >= df["bb_upper"]
        sell_signal = upper_band_touch & overbought

        # 신호 생성
        df.loc[buy_signal, "signal"] = 1
        df.loc[sell_signal, "signal"] = -1

        return df


# [스윙] 1~3주 보유 - MACD 스윙 전략
class SwingMACDStrategy(BaseStrategy):
    """스윙 MACD 전략 - MACD 크로스오버 + 히스토그램 변화"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.strategy_type = "single_asset"
        # MACD 파라미터
        self.macd_fast = getattr(params, "macd_fast", 12)
        self.macd_slow = getattr(params, "macd_slow", 26)
        self.macd_signal = getattr(params, "macd_signal", 9)
        self.histogram_threshold = getattr(params, "histogram_threshold", 0.001)
        self.confirmation_period = getattr(params, "confirmation_period", 3)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD 신호 생성"""
        df = df.copy()
        df["signal"] = 0

        # MACD 계산
        ema_fast = df["close"].ewm(span=self.macd_fast).mean()
        ema_slow = df["close"].ewm(span=self.macd_slow).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=self.macd_signal).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # MACD 크로스오버
        macd_cross_up = (df["macd"] > df["macd_signal"]) & (
            df["macd"].shift(1) <= df["macd_signal"].shift(1)
        )
        macd_cross_down = (df["macd"] < df["macd_signal"]) & (
            df["macd"].shift(1) >= df["macd_signal"].shift(1)
        )

        # 히스토그램 변화 확인
        histogram_increasing = df["macd_histogram"] > df["macd_histogram"].shift(1)
        histogram_decreasing = df["macd_histogram"] < df["macd_histogram"].shift(1)

        # 신호 생성
        df.loc[macd_cross_up & histogram_increasing, "signal"] = 1  # 상승 신호
        df.loc[macd_cross_down & histogram_decreasing, "signal"] = -1  # 하락 신호

        return df


# ============================================================================
# 포트폴리오 전략들 (Portfolio Strategies)
# ============================================================================


class PortfolioStrategy(BaseStrategy):
    """포트폴리오 전략 기본 클래스"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.strategy_type = "portfolio"  # 포트폴리오 전략으로 설정

    def generate_signals(
        self, data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """포트폴리오 신호 생성 (하위 클래스에서 구현)"""
        pass


# [포트폴리오] 동적 자산배분 전략
class DynamicAssetAllocationStrategy(PortfolioStrategy):
    """동적 자산배분 전략 - 시장 상황에 따른 자산 비중 조정"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.strategy_type = "portfolio"
        # 동적 배분 파라미터
        self.volatility_lookback = getattr(params, "volatility_lookback", 60)
        self.momentum_lookback = getattr(params, "momentum_lookback", 20)
        self.risk_aversion = getattr(params, "risk_aversion", 0.5)
        self.max_weight = getattr(params, "max_weight", 0.4)
        self.min_weight = getattr(params, "min_weight", 0.05)

    def generate_signals(
        self, data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """동적 자산배분 신호 생성"""
        signals = {}

        # 모든 종목의 공통 날짜 추출
        common_dates = set.intersection(
            *[set(df["datetime"]) for df in data_dict.values()]
        )
        common_dates = sorted(list(common_dates))

        # 각 종목별 신호 생성
        for symbol, df in data_dict.items():
            df_signal = df[df["datetime"].isin(common_dates)].copy()
            df_signal["signal"] = 1  # 기본적으로 보유
            df_signal["weight"] = 1.0 / len(data_dict)  # 동일가중

            # 변동성 기반 가중치 조정
            if len(df_signal) > self.volatility_lookback:
                volatility = (
                    df_signal["close"]
                    .pct_change()
                    .rolling(self.volatility_lookback)
                    .std()
                )
                momentum = df_signal["close"].pct_change(self.momentum_lookback)

                # 변동성이 낮고 모멘텀이 양수인 종목에 더 높은 가중치
                risk_score = volatility * self.risk_aversion - momentum * (
                    1 - self.risk_aversion
                )
                weight_adjustment = 1 / (1 + risk_score)

                # 가중치 범위 제한
                weight_adjustment = np.clip(
                    weight_adjustment, self.min_weight, self.max_weight
                )
                df_signal["weight"] = weight_adjustment

            signals[symbol] = df_signal

        return signals


# [포트폴리오] 섹터 로테이션 전략
class SectorRotationStrategy(PortfolioStrategy):
    """섹터 로테이션 전략 - 섹터별 모멘텀 기반 로테이션"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.strategy_type = "portfolio"
        # 섹터 로테이션 파라미터
        self.momentum_period = getattr(params, "momentum_period", 60)
        self.top_sectors = getattr(params, "top_sectors", 3)
        self.rebalance_frequency = getattr(params, "rebalance_frequency", 20)
        self.momentum_threshold = getattr(params, "momentum_threshold", 0.02)

    def generate_signals(
        self, data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """섹터 로테이션 신호 생성"""
        signals = {}

        # 모든 종목의 공통 날짜 추출
        common_dates = set.intersection(
            *[set(df["datetime"]) for df in data_dict.values()]
        )
        common_dates = sorted(list(common_dates))

        # 각 종목별 모멘텀 계산
        momentum_scores = {}
        for symbol, df in data_dict.items():
            df_temp = df[df["datetime"].isin(common_dates)].copy()
            if len(df_temp) > self.momentum_period:
                momentum = (
                    df_temp["close"].iloc[-1]
                    / df_temp["close"].iloc[-self.momentum_period]
                ) - 1
                momentum_scores[symbol] = momentum

        # 모멘텀 기준 상위 종목 선택
        sorted_symbols = sorted(
            momentum_scores.items(), key=lambda x: x[1], reverse=True
        )
        top_symbols = [
            symbol
            for symbol, momentum in sorted_symbols[: self.top_sectors]
            if momentum > self.momentum_threshold
        ]

        # 신호 생성
        for symbol, df in data_dict.items():
            df_signal = df[df["datetime"].isin(common_dates)].copy()

            if symbol in top_symbols:
                df_signal["signal"] = 1
                df_signal["weight"] = 1.0 / len(top_symbols)
            else:
                df_signal["signal"] = 0
                df_signal["weight"] = 0.0

            signals[symbol] = df_signal

        return signals


class MomentumAccelerationStrategy(BaseStrategy):
    """모멘텀 가속도 전략 - 상승 추세에서 모멘텀 가속을 포착"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.momentum_period = params.get("momentum_period", 10)
        self.acceleration_threshold = params.get("acceleration_threshold", 0.02)
        self.volume_threshold = params.get("volume_threshold", 1.5)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 60)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """모멘텀 가속도 신호 생성"""
        df = df.copy()

        # 모멘텀 계산
        df["momentum"] = df["close"].pct_change(self.momentum_period)
        df["momentum_ma"] = df["momentum"].rolling(window=self.momentum_period).mean()
        df["momentum_acceleration"] = df["momentum"] - df["momentum_ma"]

        # 거래량 증가율
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 신호 생성
        df["signal"] = 0

        # 매수 신호: 모멘텀 가속 + 거래량 증가 + RSI 강세
        buy_condition = (
            (df["momentum_acceleration"] > self.acceleration_threshold)
            & (df["volume_ratio"] > self.volume_threshold)
            & (df["rsi"] > self.rsi_threshold)
            & (df["close"] > df["close"].rolling(window=20).mean())
        )

        # 매도 신호: 모멘텀 감속 또는 RSI 과매수
        sell_condition = (
            (df["momentum_acceleration"] < -self.acceleration_threshold)
            | (df["rsi"] > 80)
            | (df["close"] < df["close"].rolling(window=20).mean())
        )

        df.loc[buy_condition, "signal"] = 1
        df.loc[sell_condition, "signal"] = -1

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class VolumeBreakoutStrategy(BaseStrategy):
    """거래량 브레이크아웃 전략 - 상승 추세에서 거래량 증가와 함께하는 브레이크아웃 포착"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.volume_period = params.get("volume_period", 20)
        self.volume_threshold = params.get("volume_threshold", 2.0)
        self.price_period = params.get("price_period", 20)
        self.price_threshold = params.get("price_threshold", 0.02)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 50)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """거래량 브레이크아웃 신호 생성"""
        df = df.copy()

        # 거래량 분석
        df["volume_ma"] = df["volume"].rolling(window=self.volume_period).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]

        # 가격 분석
        df["price_ma"] = df["close"].rolling(window=self.price_period).mean()
        df["price_breakout"] = df["close"] > df["price_ma"] * (1 + self.price_threshold)

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 신호 생성
        df["signal"] = 0

        # 매수 신호: 거래량 급증 + 가격 브레이크아웃 + RSI 강세
        buy_condition = (
            (df["volume_ratio"] > self.volume_threshold)
            & (df["price_breakout"])
            & (df["rsi"] > self.rsi_threshold)
            & (df["close"] > df["close"].shift(1))
        )

        # 매도 신호: 거래량 감소 또는 가격 하락
        sell_condition = (
            (df["volume_ratio"] < 0.5)
            | (df["close"] < df["price_ma"])
            | (df["rsi"] > 80)
        )

        df.loc[buy_condition, "signal"] = 1
        df.loc[sell_condition, "signal"] = -1

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class TrendStrengthStrategy(BaseStrategy):
    """추세 강도 기반 전략 - 상승 추세의 강도를 측정하여 진입"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.short_period = params.get("short_period", 10)
        self.long_period = params.get("long_period", 50)
        self.adx_period = params.get("adx_period", 14)
        self.adx_threshold = params.get("adx_threshold", 25)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 60)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """추세 강도 신호 생성"""
        df = df.copy()

        # 이동평균 계산
        df["sma_short"] = df["close"].rolling(window=self.short_period).mean()
        df["sma_long"] = df["close"].rolling(window=self.long_period).mean()

        # ADX 계산 (추세 강도)
        df["adx"] = self._calculate_adx(df, self.adx_period)

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 신호 생성
        df["signal"] = 0

        # 매수 신호: 강한 상승 추세 + RSI 강세
        buy_condition = (
            (df["sma_short"] > df["sma_long"])
            & (df["adx"] > self.adx_threshold)
            & (df["rsi"] > self.rsi_threshold)
            & (df["close"] > df["sma_short"])
        )

        # 매도 신호: 추세 약화 또는 RSI 과매수
        sell_condition = (
            (df["sma_short"] < df["sma_long"])
            | (df["adx"] < self.adx_threshold)
            | (df["rsi"] > 80)
        )

        df.loc[buy_condition, "signal"] = 1
        df.loc[sell_condition, "signal"] = -1

        return df

    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """ADX 계산"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range 계산
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        dm_plus = high - high.shift(1)
        dm_minus = low.shift(1) - low
        dm_plus = dm_plus.where(dm_plus > dm_minus, 0)
        dm_minus = dm_minus.where(dm_minus > dm_plus, 0)

        # Smoothed values
        tr_smooth = tr.rolling(window=period).mean()
        dm_plus_smooth = dm_plus.rolling(window=period).mean()
        dm_minus_smooth = dm_minus.rolling(window=period).mean()

        # DI 계산
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth

        # DX 계산
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)

        # ADX 계산
        adx = dx.rolling(window=period).mean()

        return adx

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class DefensiveHedgingStrategy(BaseStrategy):
    """방어적 헤징 전략 - 하락 추세에서 손실 최소화"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.volatility_period = params.get("volatility_period", 20)
        self.volatility_threshold = params.get("volatility_threshold", 0.03)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_oversold = params.get("rsi_oversold", 30)
        self.hedge_ratio = params.get("hedge_ratio", 0.3)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """방어적 헤징 신호 생성"""
        df = df.copy()

        # 변동성 계산
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=self.volatility_period).std()

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 이동평균
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()

        # 신호 생성
        df["signal"] = 0

        # 헤징 신호: 높은 변동성 + 하락 추세 + RSI 과매도
        hedge_condition = (
            (df["volatility"] > self.volatility_threshold)
            & (df["sma_20"] < df["sma_50"])
            & (df["rsi"] < self.rsi_oversold)
            & (df["close"] < df["sma_20"])
        )

        # 헤징 해제: 변동성 감소 또는 반등 신호
        unhedge_condition = (
            (df["volatility"] < self.volatility_threshold * 0.5)
            | (df["rsi"] > 50)
            | (df["close"] > df["sma_20"])
        )

        df.loc[hedge_condition, "signal"] = -1  # 헤징 포지션
        df.loc[unhedge_condition, "signal"] = 1  # 헤징 해제

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class ShortMomentumStrategy(BaseStrategy):
    """숏 모멘텀 전략 - 하락 추세에서 숏 포지션"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.momentum_period = params.get("momentum_period", 10)
        self.momentum_threshold = params.get("momentum_threshold", -0.02)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 70)
        self.volume_threshold = params.get("volume_threshold", 1.5)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """숏 모멘텀 신호 생성"""
        df = df.copy()

        # 모멘텀 계산
        df["momentum"] = df["close"].pct_change(self.momentum_period)
        df["momentum_ma"] = df["momentum"].rolling(window=self.momentum_period).mean()

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 거래량 분석
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

        # 이동평균
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()

        # 신호 생성
        df["signal"] = 0

        # 숏 신호: 하락 모멘텀 + RSI 과매수 + 거래량 증가
        short_condition = (
            (df["momentum"] < self.momentum_threshold)
            & (df["rsi"] > self.rsi_threshold)
            & (df["volume_ratio"] > self.volume_threshold)
            & (df["sma_20"] < df["sma_50"])
        )

        # 숏 해제: 반등 신호 또는 RSI 과매도
        cover_condition = (
            (df["momentum"] > 0) | (df["rsi"] < 30) | (df["close"] > df["sma_20"])
        )

        df.loc[short_condition, "signal"] = -1  # 숏 포지션
        df.loc[cover_condition, "signal"] = 1  # 숏 해제

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class SafeHavenRotationStrategy(BaseStrategy):
    """안전자산 순환 전략 - 하락 시 안전자산으로 순환"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.volatility_period = params.get("volatility_period", 20)
        self.volatility_threshold = params.get("volatility_threshold", 0.03)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 30)
        self.rotation_threshold = params.get("rotation_threshold", -0.05)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """안전자산 순환 신호 생성"""
        df = df.copy()

        # 변동성 계산
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=self.volatility_period).std()

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 누적 수익률
        df["cumulative_return"] = (1 + df["returns"]).cumprod() - 1
        df["rolling_return"] = df["cumulative_return"] - df["cumulative_return"].shift(
            20
        )

        # 이동평균
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()

        # 신호 생성
        df["signal"] = 0

        # 안전자산 순환 신호: 높은 변동성 + 하락 + RSI 과매도
        safe_haven_condition = (
            (df["volatility"] > self.volatility_threshold)
            & (df["rolling_return"] < self.rotation_threshold)
            & (df["rsi"] < self.rsi_threshold)
            & (df["sma_20"] < df["sma_50"])
        )

        # 리스크 자산 복귀: 변동성 감소 + 반등 신호
        risk_return_condition = (
            (df["volatility"] < self.volatility_threshold * 0.5)
            & (df["rsi"] > 50)
            & (df["rolling_return"] > 0)
        )

        df.loc[safe_haven_condition, "signal"] = -1  # 안전자산 순환
        df.loc[risk_return_condition, "signal"] = 1  # 리스크 자산 복귀

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class VolatilityExpansionStrategy(BaseStrategy):
    """변동성 확장 전략 - 변동성 높은 시장에서 변동성 확장 포착"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.volatility_period = params.get("volatility_period", 20)
        self.expansion_threshold = params.get("expansion_threshold", 1.5)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 50)
        self.atr_period = params.get("atr_period", 14)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """변동성 확장 신호 생성"""
        df = df.copy()

        # ATR 계산
        df["atr"] = self._calculate_atr(df, self.atr_period)
        df["atr_ma"] = df["atr"].rolling(window=self.volatility_period).mean()
        df["volatility_expansion"] = df["atr"] / df["atr_ma"]

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 볼린저 밴드
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        df["bb_std"] = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + (df["bb_std"] * 2)
        df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * 2)

        # 신호 생성
        df["signal"] = 0

        # 변동성 확장 신호: ATR 확장 + 볼린저 밴드 터치
        expansion_condition = (
            (df["volatility_expansion"] > self.expansion_threshold)
            & ((df["close"] > df["bb_upper"]) | (df["close"] < df["bb_lower"]))
            & (df["rsi"] > self.rsi_threshold)
        )

        # 변동성 수축 신호: ATR 수축
        contraction_condition = (df["volatility_expansion"] < 0.5) & (
            df["close"].between(df["bb_lower"], df["bb_upper"])
        )

        df.loc[expansion_condition, "signal"] = 1  # 변동성 확장 포착
        df.loc[contraction_condition, "signal"] = -1  # 변동성 수축

        return df

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """ATR 계산"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        return atr

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class GapTradingStrategy(BaseStrategy):
    """갭 트레이딩 전략 - 변동성 높은 시장에서 갭 포착"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.gap_threshold = params.get("gap_threshold", 0.02)
        self.fill_threshold = params.get("fill_threshold", 0.5)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 50)
        self.volume_threshold = params.get("volume_threshold", 1.5)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """갭 트레이딩 신호 생성"""
        df = df.copy()

        # 갭 계산
        df["gap_up"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
        df["gap_down"] = (df["close"].shift(1) - df["open"]) / df["close"].shift(1)

        # 갭 채우기 계산
        df["gap_fill_up"] = (df["low"] - df["close"].shift(1)) / df["close"].shift(1)
        df["gap_fill_down"] = (df["close"].shift(1) - df["high"]) / df["close"].shift(1)

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 거래량 분석
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

        # 신호 생성
        df["signal"] = 0

        # 갭 업 신호: 상승 갭 + 거래량 증가
        gap_up_condition = (
            (df["gap_up"] > self.gap_threshold)
            & (df["volume_ratio"] > self.volume_threshold)
            & (df["rsi"] > self.rsi_threshold)
        )

        # 갭 다운 신호: 하락 갭 + 거래량 증가
        gap_down_condition = (
            (df["gap_down"] > self.gap_threshold)
            & (df["volume_ratio"] > self.volume_threshold)
            & (df["rsi"] < self.rsi_threshold)
        )

        # 갭 채우기 신호
        gap_fill_condition = (df["gap_fill_up"] > self.fill_threshold) | (
            df["gap_fill_down"] > self.fill_threshold
        )

        df.loc[gap_up_condition, "signal"] = 1  # 갭 업 포착
        df.loc[gap_down_condition, "signal"] = -1  # 갭 다운 포착
        df.loc[gap_fill_condition, "signal"] = 0  # 갭 채우기

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class NewsEventStrategy(BaseStrategy):
    """뉴스 이벤트 기반 전략 - 변동성 높은 시장에서 이벤트 포착"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.volatility_period = params.get("volatility_period", 20)
        self.volatility_threshold = params.get("volatility_threshold", 2.0)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 50)
        self.volume_threshold = params.get("volume_threshold", 2.0)
        self.price_threshold = params.get("price_threshold", 0.03)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """뉴스 이벤트 신호 생성"""
        df = df.copy()

        # 변동성 계산
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=self.volatility_period).std()
        df["volatility_ratio"] = (
            df["volatility"]
            / df["volatility"].rolling(window=self.volatility_period).mean()
        )

        # 거래량 분석
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

        # 가격 변동
        df["price_change"] = abs(df["close"].pct_change())

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 신호 생성
        df["signal"] = 0

        # 이벤트 신호: 높은 변동성 + 거래량 급증 + 큰 가격 변동
        event_condition = (
            (df["volatility_ratio"] > self.volatility_threshold)
            & (df["volume_ratio"] > self.volume_threshold)
            & (df["price_change"] > self.price_threshold)
        )

        # 이벤트 후 정리 신호: 변동성 감소
        cleanup_condition = (df["volatility_ratio"] < 0.5) & (df["volume_ratio"] < 1.0)

        df.loc[event_condition, "signal"] = 1  # 이벤트 포착
        df.loc[cleanup_condition, "signal"] = -1  # 이벤트 후 정리

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class RangeBreakoutStrategy(BaseStrategy):
    """범위 브레이크아웃 전략 - 횡보장에서 범위 돌파 포착"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.range_period = params.get("range_period", 20)
        self.breakout_threshold = params.get("breakout_threshold", 0.02)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 50)
        self.volume_threshold = params.get("volume_threshold", 1.5)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """범위 브레이크아웃 신호 생성"""
        df = df.copy()

        # 범위 계산
        df["range_high"] = df["high"].rolling(window=self.range_period).max()
        df["range_low"] = df["low"].rolling(window=self.range_period).min()
        df["range_mid"] = (df["range_high"] + df["range_low"]) / 2

        # 브레이크아웃 계산
        df["breakout_up"] = (df["close"] - df["range_high"]) / df["range_high"]
        df["breakout_down"] = (df["range_low"] - df["close"]) / df["range_low"]

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 거래량 분석
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

        # 신호 생성
        df["signal"] = 0

        # 상향 브레이크아웃: 상단 돌파 + 거래량 증가
        breakout_up_condition = (
            (df["breakout_up"] > self.breakout_threshold)
            & (df["volume_ratio"] > self.volume_threshold)
            & (df["rsi"] > self.rsi_threshold)
        )

        # 하향 브레이크아웃: 하단 돌파 + 거래량 증가
        breakout_down_condition = (
            (df["breakout_down"] > self.breakout_threshold)
            & (df["volume_ratio"] > self.volume_threshold)
            & (df["rsi"] < self.rsi_threshold)
        )

        # 범위 내 복귀
        range_return_condition = (
            df["close"].between(df["range_low"], df["range_high"])
        ) & (df["volume_ratio"] < 1.0)

        df.loc[breakout_up_condition, "signal"] = 1  # 상향 브레이크아웃
        df.loc[breakout_down_condition, "signal"] = -1  # 하향 브레이크아웃
        df.loc[range_return_condition, "signal"] = 0  # 범위 내 복귀

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class SupportResistanceStrategy(BaseStrategy):
    """지지/저항 기반 전략 - 횡보장에서 지지/저항선 활용"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.support_period = params.get("support_period", 20)
        self.resistance_period = params.get("resistance_period", 20)
        self.touch_threshold = params.get("touch_threshold", 0.01)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_oversold = params.get("rsi_oversold", 30)
        self.rsi_overbought = params.get("rsi_overbought", 70)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """지지/저항 신호 생성"""
        df = df.copy()

        # 지지/저항선 계산
        df["support"] = df["low"].rolling(window=self.support_period).min()
        df["resistance"] = df["high"].rolling(window=self.resistance_period).max()

        # 지지/저항 터치 계산
        df["support_touch"] = (df["low"] - df["support"]) / df["support"]
        df["resistance_touch"] = (df["resistance"] - df["high"]) / df["resistance"]

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 신호 생성
        df["signal"] = 0

        # 지지선 반등: 지지선 터치 + RSI 과매도
        support_bounce_condition = (
            (abs(df["support_touch"]) < self.touch_threshold)
            & (df["rsi"] < self.rsi_oversold)
            & (df["close"] > df["low"])
        )

        # 저항선 반락: 저항선 터치 + RSI 과매수
        resistance_rejection_condition = (
            (abs(df["resistance_touch"]) < self.touch_threshold)
            & (df["rsi"] > self.rsi_overbought)
            & (df["close"] < df["high"])
        )

        df.loc[support_bounce_condition, "signal"] = 1  # 지지선 반등
        df.loc[resistance_rejection_condition, "signal"] = -1  # 저항선 반락

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class OscillatorConvergenceStrategy(BaseStrategy):
    """오실레이터 수렴 전략 - 횡보장에서 여러 오실레이터의 수렴 포착"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.rsi_period = params.get("rsi_period", 14)
        self.stoch_period = params.get("stoch_period", 14)
        self.williams_period = params.get("williams_period", 14)
        self.convergence_threshold = params.get("convergence_threshold", 10)
        self.volume_threshold = params.get("volume_threshold", 1.2)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """오실레이터 수렴 신호 생성"""
        df = df.copy()

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 스토캐스틱 계산
        df["stoch_k"] = self._calculate_stochastic(df, self.stoch_period)
        df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

        # 윌리엄스 %R 계산
        df["williams_r"] = self._calculate_williams_r(df, self.williams_period)

        # 오실레이터 수렴 계산
        df["oscillator_avg"] = (
            df["rsi"] + df["stoch_k"] + (100 + df["williams_r"])
        ) / 3
        df["oscillator_std"] = df[["rsi", "stoch_k", "williams_r"]].std(axis=1)

        # 거래량 분석
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

        # 신호 생성
        df["signal"] = 0

        # 수렴 신호: 오실레이터 수렴 + 거래량 증가
        convergence_condition = (
            (df["oscillator_std"] < self.convergence_threshold)
            & (df["volume_ratio"] > self.volume_threshold)
            & (df["oscillator_avg"].between(30, 70))
        )

        # 발산 신호: 오실레이터 발산
        divergence_condition = (
            df["oscillator_std"] > self.convergence_threshold * 2
        ) & (df["volume_ratio"] < 0.8)

        df.loc[convergence_condition, "signal"] = 1  # 수렴 신호
        df.loc[divergence_condition, "signal"] = -1  # 발산 신호

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_stochastic(self, df: pd.DataFrame, period: int) -> pd.Series:
        """스토캐스틱 %K 계산"""
        low_min = df["low"].rolling(window=period).min()
        high_max = df["high"].rolling(window=period).max()
        stoch_k = 100 * (df["close"] - low_min) / (high_max - low_min)
        return stoch_k

    def _calculate_williams_r(self, df: pd.DataFrame, period: int) -> pd.Series:
        """윌리엄스 %R 계산"""
        low_min = df["low"].rolling(window=period).min()
        high_max = df["high"].rolling(window=period).max()
        williams_r = -100 * (high_max - df["close"]) / (high_max - low_min)
        return williams_r


class MomentumAccelerationStrategy(BaseStrategy):
    """모멘텀 가속도 전략 - 상승 추세에서 모멘텀 가속을 포착"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.momentum_period = params.get("momentum_period", 10)
        self.acceleration_threshold = params.get("acceleration_threshold", 0.02)
        self.volume_threshold = params.get("volume_threshold", 1.5)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 60)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """모멘텀 가속도 신호 생성"""
        df = df.copy()

        # 모멘텀 계산
        df["momentum"] = df["close"].pct_change(self.momentum_period)
        df["momentum_ma"] = df["momentum"].rolling(window=self.momentum_period).mean()
        df["momentum_acceleration"] = df["momentum"] - df["momentum_ma"]

        # 거래량 증가율
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 신호 생성
        df["signal"] = 0

        # 매수 신호: 모멘텀 가속 + 거래량 증가 + RSI 강세
        buy_condition = (
            (df["momentum_acceleration"] > self.acceleration_threshold)
            & (df["volume_ratio"] > self.volume_threshold)
            & (df["rsi"] > self.rsi_threshold)
            & (df["close"] > df["close"].rolling(window=20).mean())
        )

        # 매도 신호: 모멘텀 감속 또는 RSI 과매수
        sell_condition = (
            (df["momentum_acceleration"] < -self.acceleration_threshold)
            | (df["rsi"] > 80)
            | (df["close"] < df["close"].rolling(window=20).mean())
        )

        df.loc[buy_condition, "signal"] = 1
        df.loc[sell_condition, "signal"] = -1

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class VolumeBreakoutStrategy(BaseStrategy):
    """거래량 브레이크아웃 전략 - 상승 추세에서 거래량 증가와 함께하는 브레이크아웃 포착"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.volume_period = params.get("volume_period", 20)
        self.volume_threshold = params.get("volume_threshold", 2.0)
        self.price_period = params.get("price_period", 20)
        self.price_threshold = params.get("price_threshold", 0.02)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 50)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """거래량 브레이크아웃 신호 생성"""
        df = df.copy()

        # 거래량 분석
        df["volume_ma"] = df["volume"].rolling(window=self.volume_period).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]

        # 가격 분석
        df["price_ma"] = df["close"].rolling(window=self.price_period).mean()
        df["price_breakout"] = df["close"] > df["price_ma"] * (1 + self.price_threshold)

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 신호 생성
        df["signal"] = 0

        # 매수 신호: 거래량 급증 + 가격 브레이크아웃 + RSI 강세
        buy_condition = (
            (df["volume_ratio"] > self.volume_threshold)
            & (df["price_breakout"])
            & (df["rsi"] > self.rsi_threshold)
            & (df["close"] > df["close"].shift(1))
        )

        # 매도 신호: 거래량 감소 또는 가격 하락
        sell_condition = (
            (df["volume_ratio"] < 0.5)
            | (df["close"] < df["price_ma"])
            | (df["rsi"] > 80)
        )

        df.loc[buy_condition, "signal"] = 1
        df.loc[sell_condition, "signal"] = -1

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class TrendStrengthStrategy(BaseStrategy):
    """추세 강도 기반 전략 - 상승 추세의 강도를 측정하여 진입"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.short_period = params.get("short_period", 10)
        self.long_period = params.get("long_period", 50)
        self.adx_period = params.get("adx_period", 14)
        self.adx_threshold = params.get("adx_threshold", 25)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 60)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """추세 강도 신호 생성"""
        df = df.copy()

        # 이동평균 계산
        df["sma_short"] = df["close"].rolling(window=self.short_period).mean()
        df["sma_long"] = df["close"].rolling(window=self.long_period).mean()

        # ADX 계산 (추세 강도)
        df["adx"] = self._calculate_adx(df, self.adx_period)

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 신호 생성
        df["signal"] = 0

        # 매수 신호: 강한 상승 추세 + RSI 강세
        buy_condition = (
            (df["sma_short"] > df["sma_long"])
            & (df["adx"] > self.adx_threshold)
            & (df["rsi"] > self.rsi_threshold)
            & (df["close"] > df["sma_short"])
        )

        # 매도 신호: 추세 약화 또는 RSI 과매수
        sell_condition = (
            (df["sma_short"] < df["sma_long"])
            | (df["adx"] < self.adx_threshold)
            | (df["rsi"] > 80)
        )

        df.loc[buy_condition, "signal"] = 1
        df.loc[sell_condition, "signal"] = -1

        return df

    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """ADX 계산"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range 계산
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        dm_plus = high - high.shift(1)
        dm_minus = low.shift(1) - low
        dm_plus = dm_plus.where(dm_plus > dm_minus, 0)
        dm_minus = dm_minus.where(dm_minus > dm_plus, 0)

        # Smoothed values
        tr_smooth = tr.rolling(window=period).mean()
        dm_plus_smooth = dm_plus.rolling(window=period).mean()
        dm_minus_smooth = dm_minus.rolling(window=period).mean()

        # DI 계산
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth

        # DX 계산
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)

        # ADX 계산
        adx = dx.rolling(window=period).mean()

        return adx

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class DefensiveHedgingStrategy(BaseStrategy):
    """방어적 헤징 전략 - 하락 추세에서 손실 최소화"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.volatility_period = params.get("volatility_period", 20)
        self.volatility_threshold = params.get("volatility_threshold", 0.03)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_oversold = params.get("rsi_oversold", 30)
        self.hedge_ratio = params.get("hedge_ratio", 0.3)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """방어적 헤징 신호 생성"""
        df = df.copy()

        # 변동성 계산
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=self.volatility_period).std()

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 이동평균
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()

        # 신호 생성
        df["signal"] = 0

        # 헤징 신호: 높은 변동성 + 하락 추세 + RSI 과매도
        hedge_condition = (
            (df["volatility"] > self.volatility_threshold)
            & (df["sma_20"] < df["sma_50"])
            & (df["rsi"] < self.rsi_oversold)
            & (df["close"] < df["sma_20"])
        )

        # 헤징 해제: 변동성 감소 또는 반등 신호
        unhedge_condition = (
            (df["volatility"] < self.volatility_threshold * 0.5)
            | (df["rsi"] > 50)
            | (df["close"] > df["sma_20"])
        )

        df.loc[hedge_condition, "signal"] = -1  # 헤징 포지션
        df.loc[unhedge_condition, "signal"] = 1  # 헤징 해제

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class ShortMomentumStrategy(BaseStrategy):
    """숏 모멘텀 전략 - 하락 추세에서 숏 포지션"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.momentum_period = params.get("momentum_period", 10)
        self.momentum_threshold = params.get("momentum_threshold", -0.02)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 70)
        self.volume_threshold = params.get("volume_threshold", 1.5)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """숏 모멘텀 신호 생성"""
        df = df.copy()

        # 모멘텀 계산
        df["momentum"] = df["close"].pct_change(self.momentum_period)
        df["momentum_ma"] = df["momentum"].rolling(window=self.momentum_period).mean()

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 거래량 분석
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

        # 이동평균
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()

        # 신호 생성
        df["signal"] = 0

        # 숏 신호: 하락 모멘텀 + RSI 과매수 + 거래량 증가
        short_condition = (
            (df["momentum"] < self.momentum_threshold)
            & (df["rsi"] > self.rsi_threshold)
            & (df["volume_ratio"] > self.volume_threshold)
            & (df["sma_20"] < df["sma_50"])
        )

        # 숏 해제: 반등 신호 또는 RSI 과매도
        cover_condition = (
            (df["momentum"] > 0) | (df["rsi"] < 30) | (df["close"] > df["sma_20"])
        )

        df.loc[short_condition, "signal"] = -1  # 숏 포지션
        df.loc[cover_condition, "signal"] = 1  # 숏 해제

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class SafeHavenRotationStrategy(BaseStrategy):
    """안전자산 순환 전략 - 하락 시 안전자산으로 순환"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.volatility_period = params.get("volatility_period", 20)
        self.volatility_threshold = params.get("volatility_threshold", 0.03)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 30)
        self.rotation_threshold = params.get("rotation_threshold", -0.05)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """안전자산 순환 신호 생성"""
        df = df.copy()

        # 변동성 계산
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=self.volatility_period).std()

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 누적 수익률
        df["cumulative_return"] = (1 + df["returns"]).cumprod() - 1
        df["rolling_return"] = df["cumulative_return"] - df["cumulative_return"].shift(
            20
        )

        # 이동평균
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()

        # 신호 생성
        df["signal"] = 0

        # 안전자산 순환 신호: 높은 변동성 + 하락 + RSI 과매도
        safe_haven_condition = (
            (df["volatility"] > self.volatility_threshold)
            & (df["rolling_return"] < self.rotation_threshold)
            & (df["rsi"] < self.rsi_threshold)
            & (df["sma_20"] < df["sma_50"])
        )

        # 리스크 자산 복귀: 변동성 감소 + 반등 신호
        risk_return_condition = (
            (df["volatility"] < self.volatility_threshold * 0.5)
            & (df["rsi"] > 50)
            & (df["rolling_return"] > 0)
        )

        df.loc[safe_haven_condition, "signal"] = -1  # 안전자산 순환
        df.loc[risk_return_condition, "signal"] = 1  # 리스크 자산 복귀

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class VolatilityExpansionStrategy(BaseStrategy):
    """변동성 확장 전략 - 변동성 높은 시장에서 변동성 확장 포착"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.volatility_period = params.get("volatility_period", 20)
        self.expansion_threshold = params.get("expansion_threshold", 1.5)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 50)
        self.atr_period = params.get("atr_period", 14)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """변동성 확장 신호 생성"""
        df = df.copy()

        # ATR 계산
        df["atr"] = self._calculate_atr(df, self.atr_period)
        df["atr_ma"] = df["atr"].rolling(window=self.volatility_period).mean()
        df["volatility_expansion"] = df["atr"] / df["atr_ma"]

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 볼린저 밴드
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        df["bb_std"] = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + (df["bb_std"] * 2)
        df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * 2)

        # 신호 생성
        df["signal"] = 0

        # 변동성 확장 신호: ATR 확장 + 볼린저 밴드 터치
        expansion_condition = (
            (df["volatility_expansion"] > self.expansion_threshold)
            & ((df["close"] > df["bb_upper"]) | (df["close"] < df["bb_lower"]))
            & (df["rsi"] > self.rsi_threshold)
        )

        # 변동성 수축 신호: ATR 수축
        contraction_condition = (df["volatility_expansion"] < 0.5) & (
            df["close"].between(df["bb_lower"], df["bb_upper"])
        )

        df.loc[expansion_condition, "signal"] = 1  # 변동성 확장 포착
        df.loc[contraction_condition, "signal"] = -1  # 변동성 수축

        return df

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """ATR 계산"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        return atr

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class GapTradingStrategy(BaseStrategy):
    """갭 트레이딩 전략 - 변동성 높은 시장에서 갭 포착"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.gap_threshold = params.get("gap_threshold", 0.02)
        self.fill_threshold = params.get("fill_threshold", 0.5)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 50)
        self.volume_threshold = params.get("volume_threshold", 1.5)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """갭 트레이딩 신호 생성"""
        df = df.copy()

        # 갭 계산
        df["gap_up"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
        df["gap_down"] = (df["close"].shift(1) - df["open"]) / df["close"].shift(1)

        # 갭 채우기 계산
        df["gap_fill_up"] = (df["low"] - df["close"].shift(1)) / df["close"].shift(1)
        df["gap_fill_down"] = (df["close"].shift(1) - df["high"]) / df["close"].shift(1)

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 거래량 분석
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

        # 신호 생성
        df["signal"] = 0

        # 갭 업 신호: 상승 갭 + 거래량 증가
        gap_up_condition = (
            (df["gap_up"] > self.gap_threshold)
            & (df["volume_ratio"] > self.volume_threshold)
            & (df["rsi"] > self.rsi_threshold)
        )

        # 갭 다운 신호: 하락 갭 + 거래량 증가
        gap_down_condition = (
            (df["gap_down"] > self.gap_threshold)
            & (df["volume_ratio"] > self.volume_threshold)
            & (df["rsi"] < self.rsi_threshold)
        )

        # 갭 채우기 신호
        gap_fill_condition = (df["gap_fill_up"] > self.fill_threshold) | (
            df["gap_fill_down"] > self.fill_threshold
        )

        df.loc[gap_up_condition, "signal"] = 1  # 갭 업 포착
        df.loc[gap_down_condition, "signal"] = -1  # 갭 다운 포착
        df.loc[gap_fill_condition, "signal"] = 0  # 갭 채우기

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class NewsEventStrategy(BaseStrategy):
    """뉴스 이벤트 기반 전략 - 변동성 높은 시장에서 이벤트 포착"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.volatility_period = params.get("volatility_period", 20)
        self.volatility_threshold = params.get("volatility_threshold", 2.0)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 50)
        self.volume_threshold = params.get("volume_threshold", 2.0)
        self.price_threshold = params.get("price_threshold", 0.03)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """뉴스 이벤트 신호 생성"""
        df = df.copy()

        # 변동성 계산
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=self.volatility_period).std()
        df["volatility_ratio"] = (
            df["volatility"]
            / df["volatility"].rolling(window=self.volatility_period).mean()
        )

        # 거래량 분석
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

        # 가격 변동
        df["price_change"] = abs(df["close"].pct_change())

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 신호 생성
        df["signal"] = 0

        # 이벤트 신호: 높은 변동성 + 거래량 급증 + 큰 가격 변동
        event_condition = (
            (df["volatility_ratio"] > self.volatility_threshold)
            & (df["volume_ratio"] > self.volume_threshold)
            & (df["price_change"] > self.price_threshold)
        )

        # 이벤트 후 정리 신호: 변동성 감소
        cleanup_condition = (df["volatility_ratio"] < 0.5) & (df["volume_ratio"] < 1.0)

        df.loc[event_condition, "signal"] = 1  # 이벤트 포착
        df.loc[cleanup_condition, "signal"] = -1  # 이벤트 후 정리

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class RangeBreakoutStrategy(BaseStrategy):
    """범위 브레이크아웃 전략 - 횡보장에서 범위 돌파 포착"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.range_period = params.get("range_period", 20)
        self.breakout_threshold = params.get("breakout_threshold", 0.02)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 50)
        self.volume_threshold = params.get("volume_threshold", 1.5)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """범위 브레이크아웃 신호 생성"""
        df = df.copy()

        # 범위 계산
        df["range_high"] = df["high"].rolling(window=self.range_period).max()
        df["range_low"] = df["low"].rolling(window=self.range_period).min()
        df["range_mid"] = (df["range_high"] + df["range_low"]) / 2

        # 브레이크아웃 계산
        df["breakout_up"] = (df["close"] - df["range_high"]) / df["range_high"]
        df["breakout_down"] = (df["range_low"] - df["close"]) / df["range_low"]

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 거래량 분석
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

        # 신호 생성
        df["signal"] = 0

        # 상향 브레이크아웃: 상단 돌파 + 거래량 증가
        breakout_up_condition = (
            (df["breakout_up"] > self.breakout_threshold)
            & (df["volume_ratio"] > self.volume_threshold)
            & (df["rsi"] > self.rsi_threshold)
        )

        # 하향 브레이크아웃: 하단 돌파 + 거래량 증가
        breakout_down_condition = (
            (df["breakout_down"] > self.breakout_threshold)
            & (df["volume_ratio"] > self.volume_threshold)
            & (df["rsi"] < self.rsi_threshold)
        )

        # 범위 내 복귀
        range_return_condition = (
            df["close"].between(df["range_low"], df["range_high"])
        ) & (df["volume_ratio"] < 1.0)

        df.loc[breakout_up_condition, "signal"] = 1  # 상향 브레이크아웃
        df.loc[breakout_down_condition, "signal"] = -1  # 하향 브레이크아웃
        df.loc[range_return_condition, "signal"] = 0  # 범위 내 복귀

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class SupportResistanceStrategy(BaseStrategy):
    """지지/저항 기반 전략 - 횡보장에서 지지/저항선 활용"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.support_period = params.get("support_period", 20)
        self.resistance_period = params.get("resistance_period", 20)
        self.touch_threshold = params.get("touch_threshold", 0.01)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_oversold = params.get("rsi_oversold", 30)
        self.rsi_overbought = params.get("rsi_overbought", 70)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """지지/저항 신호 생성"""
        df = df.copy()

        # 지지/저항선 계산
        df["support"] = df["low"].rolling(window=self.support_period).min()
        df["resistance"] = df["high"].rolling(window=self.resistance_period).max()

        # 지지/저항 터치 계산
        df["support_touch"] = (df["low"] - df["support"]) / df["support"]
        df["resistance_touch"] = (df["resistance"] - df["high"]) / df["resistance"]

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 신호 생성
        df["signal"] = 0

        # 지지선 반등: 지지선 터치 + RSI 과매도
        support_bounce_condition = (
            (abs(df["support_touch"]) < self.touch_threshold)
            & (df["rsi"] < self.rsi_oversold)
            & (df["close"] > df["low"])
        )

        # 저항선 반락: 저항선 터치 + RSI 과매수
        resistance_rejection_condition = (
            (abs(df["resistance_touch"]) < self.touch_threshold)
            & (df["rsi"] > self.rsi_overbought)
            & (df["close"] < df["high"])
        )

        df.loc[support_bounce_condition, "signal"] = 1  # 지지선 반등
        df.loc[resistance_rejection_condition, "signal"] = -1  # 저항선 반락

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class OscillatorConvergenceStrategy(BaseStrategy):
    """오실레이터 수렴 전략 - 횡보장에서 여러 오실레이터의 수렴 포착"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.rsi_period = params.get("rsi_period", 14)
        self.stoch_period = params.get("stoch_period", 14)
        self.williams_period = params.get("williams_period", 14)
        self.convergence_threshold = params.get("convergence_threshold", 10)
        self.volume_threshold = params.get("volume_threshold", 1.2)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """오실레이터 수렴 신호 생성"""
        df = df.copy()

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 스토캐스틱 계산
        df["stoch_k"] = self._calculate_stochastic(df, self.stoch_period)
        df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

        # 윌리엄스 %R 계산
        df["williams_r"] = self._calculate_williams_r(df, self.williams_period)

        # 오실레이터 수렴 계산
        df["oscillator_avg"] = (
            df["rsi"] + df["stoch_k"] + (100 + df["williams_r"])
        ) / 3
        df["oscillator_std"] = df[["rsi", "stoch_k", "williams_r"]].std(axis=1)

        # 거래량 분석
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

        # 신호 생성
        df["signal"] = 0

        # 수렴 신호: 오실레이터 수렴 + 거래량 증가
        convergence_condition = (
            (df["oscillator_std"] < self.convergence_threshold)
            & (df["volume_ratio"] > self.volume_threshold)
            & (df["oscillator_avg"].between(30, 70))
        )

        # 발산 신호: 오실레이터 발산
        divergence_condition = (
            df["oscillator_std"] > self.convergence_threshold * 2
        ) & (df["volume_ratio"] < 0.8)

        df.loc[convergence_condition, "signal"] = 1  # 수렴 신호
        df.loc[divergence_condition, "signal"] = -1  # 발산 신호

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_stochastic(self, df: pd.DataFrame, period: int) -> pd.Series:
        """스토캐스틱 %K 계산"""
        low_min = df["low"].rolling(window=period).min()
        high_max = df["high"].rolling(window=period).max()
        stoch_k = 100 * (df["close"] - low_min) / (high_max - low_min)
        return stoch_k

    def _calculate_williams_r(self, df: pd.DataFrame, period: int) -> pd.Series:
        """윌리엄스 %R 계산"""
        low_min = df["low"].rolling(window=period).min()
        high_max = df["high"].rolling(window=period).max()
        williams_r = -100 * (high_max - df["close"]) / (high_max - low_min)
        return williams_r


class MomentumAccelerationStrategy(BaseStrategy):
    """모멘텀 가속도 전략 - 상승 추세에서 모멘텀 가속을 포착"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.momentum_period = params.get("momentum_period", 10)
        self.acceleration_threshold = params.get("acceleration_threshold", 0.02)
        self.volume_threshold = params.get("volume_threshold", 1.5)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 60)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """모멘텀 가속도 신호 생성"""
        df = df.copy()

        # 모멘텀 계산
        df["momentum"] = df["close"].pct_change(self.momentum_period)
        df["momentum_ma"] = df["momentum"].rolling(window=self.momentum_period).mean()
        df["momentum_acceleration"] = df["momentum"] - df["momentum_ma"]

        # 거래량 증가율
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 신호 생성
        df["signal"] = 0

        # 매수 신호: 모멘텀 가속 + 거래량 증가 + RSI 강세
        buy_condition = (
            (df["momentum_acceleration"] > self.acceleration_threshold)
            & (df["volume_ratio"] > self.volume_threshold)
            & (df["rsi"] > self.rsi_threshold)
            & (df["close"] > df["close"].rolling(window=20).mean())
        )

        # 매도 신호: 모멘텀 감속 또는 RSI 과매수
        sell_condition = (
            (df["momentum_acceleration"] < -self.acceleration_threshold)
            | (df["rsi"] > 80)
            | (df["close"] < df["close"].rolling(window=20).mean())
        )

        df.loc[buy_condition, "signal"] = 1
        df.loc[sell_condition, "signal"] = -1

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class VolumeBreakoutStrategy(BaseStrategy):
    """거래량 브레이크아웃 전략 - 상승 추세에서 거래량 증가와 함께하는 브레이크아웃 포착"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.volume_period = params.get("volume_period", 20)
        self.volume_threshold = params.get("volume_threshold", 2.0)
        self.price_period = params.get("price_period", 20)
        self.price_threshold = params.get("price_threshold", 0.02)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 50)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """거래량 브레이크아웃 신호 생성"""
        df = df.copy()

        # 거래량 분석
        df["volume_ma"] = df["volume"].rolling(window=self.volume_period).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]

        # 가격 분석
        df["price_ma"] = df["close"].rolling(window=self.price_period).mean()
        df["price_breakout"] = df["close"] > df["price_ma"] * (1 + self.price_threshold)

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 신호 생성
        df["signal"] = 0

        # 매수 신호: 거래량 급증 + 가격 브레이크아웃 + RSI 강세
        buy_condition = (
            (df["volume_ratio"] > self.volume_threshold)
            & (df["price_breakout"])
            & (df["rsi"] > self.rsi_threshold)
            & (df["close"] > df["close"].shift(1))
        )

        # 매도 신호: 거래량 감소 또는 가격 하락
        sell_condition = (
            (df["volume_ratio"] < 0.5)
            | (df["close"] < df["price_ma"])
            | (df["rsi"] > 80)
        )

        df.loc[buy_condition, "signal"] = 1
        df.loc[sell_condition, "signal"] = -1

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class TrendStrengthStrategy(BaseStrategy):
    """추세 강도 기반 전략 - 상승 추세의 강도를 측정하여 진입"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.short_period = params.get("short_period", 10)
        self.long_period = params.get("long_period", 50)
        self.adx_period = params.get("adx_period", 14)
        self.adx_threshold = params.get("adx_threshold", 25)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 60)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """추세 강도 신호 생성"""
        df = df.copy()

        # 이동평균 계산
        df["sma_short"] = df["close"].rolling(window=self.short_period).mean()
        df["sma_long"] = df["close"].rolling(window=self.long_period).mean()

        # ADX 계산 (추세 강도)
        df["adx"] = self._calculate_adx(df, self.adx_period)

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 신호 생성
        df["signal"] = 0

        # 매수 신호: 강한 상승 추세 + RSI 강세
        buy_condition = (
            (df["sma_short"] > df["sma_long"])
            & (df["adx"] > self.adx_threshold)
            & (df["rsi"] > self.rsi_threshold)
            & (df["close"] > df["sma_short"])
        )

        # 매도 신호: 추세 약화 또는 RSI 과매수
        sell_condition = (
            (df["sma_short"] < df["sma_long"])
            | (df["adx"] < self.adx_threshold)
            | (df["rsi"] > 80)
        )

        df.loc[buy_condition, "signal"] = 1
        df.loc[sell_condition, "signal"] = -1

        return df

    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """ADX 계산"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range 계산
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        dm_plus = high - high.shift(1)
        dm_minus = low.shift(1) - low
        dm_plus = dm_plus.where(dm_plus > dm_minus, 0)
        dm_minus = dm_minus.where(dm_minus > dm_plus, 0)

        # Smoothed values
        tr_smooth = tr.rolling(window=period).mean()
        dm_plus_smooth = dm_plus.rolling(window=period).mean()
        dm_minus_smooth = dm_minus.rolling(window=period).mean()

        # DI 계산
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth

        # DX 계산
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)

        # ADX 계산
        adx = dx.rolling(window=period).mean()

        return adx

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class DefensiveHedgingStrategy(BaseStrategy):
    """방어적 헤징 전략 - 하락 추세에서 손실 최소화"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.volatility_period = params.get("volatility_period", 20)
        self.volatility_threshold = params.get("volatility_threshold", 0.03)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_oversold = params.get("rsi_oversold", 30)
        self.hedge_ratio = params.get("hedge_ratio", 0.3)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """방어적 헤징 신호 생성"""
        df = df.copy()

        # 변동성 계산
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=self.volatility_period).std()

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 이동평균
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()

        # 신호 생성
        df["signal"] = 0

        # 헤징 신호: 높은 변동성 + 하락 추세 + RSI 과매도
        hedge_condition = (
            (df["volatility"] > self.volatility_threshold)
            & (df["sma_20"] < df["sma_50"])
            & (df["rsi"] < self.rsi_oversold)
            & (df["close"] < df["sma_20"])
        )

        # 헤징 해제: 변동성 감소 또는 반등 신호
        unhedge_condition = (
            (df["volatility"] < self.volatility_threshold * 0.5)
            | (df["rsi"] > 50)
            | (df["close"] > df["sma_20"])
        )

        df.loc[hedge_condition, "signal"] = -1  # 헤징 포지션
        df.loc[unhedge_condition, "signal"] = 1  # 헤징 해제

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class ShortMomentumStrategy(BaseStrategy):
    """숏 모멘텀 전략 - 하락 추세에서 숏 포지션"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.momentum_period = params.get("momentum_period", 10)
        self.momentum_threshold = params.get("momentum_threshold", -0.02)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 70)
        self.volume_threshold = params.get("volume_threshold", 1.5)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """숏 모멘텀 신호 생성"""
        df = df.copy()

        # 모멘텀 계산
        df["momentum"] = df["close"].pct_change(self.momentum_period)
        df["momentum_ma"] = df["momentum"].rolling(window=self.momentum_period).mean()

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 거래량 분석
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

        # 이동평균
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()

        # 신호 생성
        df["signal"] = 0

        # 숏 신호: 하락 모멘텀 + RSI 과매수 + 거래량 증가
        short_condition = (
            (df["momentum"] < self.momentum_threshold)
            & (df["rsi"] > self.rsi_threshold)
            & (df["volume_ratio"] > self.volume_threshold)
            & (df["sma_20"] < df["sma_50"])
        )

        # 숏 해제: 반등 신호 또는 RSI 과매도
        cover_condition = (
            (df["momentum"] > 0) | (df["rsi"] < 30) | (df["close"] > df["sma_20"])
        )

        df.loc[short_condition, "signal"] = -1  # 숏 포지션
        df.loc[cover_condition, "signal"] = 1  # 숏 해제

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class SafeHavenRotationStrategy(BaseStrategy):
    """안전자산 순환 전략 - 하락 시 안전자산으로 순환"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.volatility_period = params.get("volatility_period", 20)
        self.volatility_threshold = params.get("volatility_threshold", 0.03)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 30)
        self.rotation_threshold = params.get("rotation_threshold", -0.05)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """안전자산 순환 신호 생성"""
        df = df.copy()

        # 변동성 계산
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=self.volatility_period).std()

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 누적 수익률
        df["cumulative_return"] = (1 + df["returns"]).cumprod() - 1
        df["rolling_return"] = df["cumulative_return"] - df["cumulative_return"].shift(
            20
        )

        # 이동평균
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()

        # 신호 생성
        df["signal"] = 0

        # 안전자산 순환 신호: 높은 변동성 + 하락 + RSI 과매도
        safe_haven_condition = (
            (df["volatility"] > self.volatility_threshold)
            & (df["rolling_return"] < self.rotation_threshold)
            & (df["rsi"] < self.rsi_threshold)
            & (df["sma_20"] < df["sma_50"])
        )

        # 리스크 자산 복귀: 변동성 감소 + 반등 신호
        risk_return_condition = (
            (df["volatility"] < self.volatility_threshold * 0.5)
            & (df["rsi"] > 50)
            & (df["rolling_return"] > 0)
        )

        df.loc[safe_haven_condition, "signal"] = -1  # 안전자산 순환
        df.loc[risk_return_condition, "signal"] = 1  # 리스크 자산 복귀

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class VolatilityExpansionStrategy(BaseStrategy):
    """변동성 확장 전략 - 변동성 높은 시장에서 변동성 확장 포착"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.volatility_period = params.get("volatility_period", 20)
        self.expansion_threshold = params.get("expansion_threshold", 1.5)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 50)
        self.atr_period = params.get("atr_period", 14)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """변동성 확장 신호 생성"""
        df = df.copy()

        # ATR 계산
        df["atr"] = self._calculate_atr(df, self.atr_period)
        df["atr_ma"] = df["atr"].rolling(window=self.volatility_period).mean()
        df["volatility_expansion"] = df["atr"] / df["atr_ma"]

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 볼린저 밴드
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        df["bb_std"] = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + (df["bb_std"] * 2)
        df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * 2)

        # 신호 생성
        df["signal"] = 0

        # 변동성 확장 신호: ATR 확장 + 볼린저 밴드 터치
        expansion_condition = (
            (df["volatility_expansion"] > self.expansion_threshold)
            & ((df["close"] > df["bb_upper"]) | (df["close"] < df["bb_lower"]))
            & (df["rsi"] > self.rsi_threshold)
        )

        # 변동성 수축 신호: ATR 수축
        contraction_condition = (df["volatility_expansion"] < 0.5) & (
            df["close"].between(df["bb_lower"], df["bb_upper"])
        )

        df.loc[expansion_condition, "signal"] = 1  # 변동성 확장 포착
        df.loc[contraction_condition, "signal"] = -1  # 변동성 수축

        return df

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """ATR 계산"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        return atr

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class GapTradingStrategy(BaseStrategy):
    """갭 트레이딩 전략 - 변동성 높은 시장에서 갭 포착"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.gap_threshold = params.get("gap_threshold", 0.02)
        self.fill_threshold = params.get("fill_threshold", 0.5)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 50)
        self.volume_threshold = params.get("volume_threshold", 1.5)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """갭 트레이딩 신호 생성"""
        df = df.copy()

        # 갭 계산
        df["gap_up"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
        df["gap_down"] = (df["close"].shift(1) - df["open"]) / df["close"].shift(1)

        # 갭 채우기 계산
        df["gap_fill_up"] = (df["low"] - df["close"].shift(1)) / df["close"].shift(1)
        df["gap_fill_down"] = (df["close"].shift(1) - df["high"]) / df["close"].shift(1)

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 거래량 분석
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

        # 신호 생성
        df["signal"] = 0

        # 갭 업 신호: 상승 갭 + 거래량 증가
        gap_up_condition = (
            (df["gap_up"] > self.gap_threshold)
            & (df["volume_ratio"] > self.volume_threshold)
            & (df["rsi"] > self.rsi_threshold)
        )

        # 갭 다운 신호: 하락 갭 + 거래량 증가
        gap_down_condition = (
            (df["gap_down"] > self.gap_threshold)
            & (df["volume_ratio"] > self.volume_threshold)
            & (df["rsi"] < self.rsi_threshold)
        )

        # 갭 채우기 신호
        gap_fill_condition = (df["gap_fill_up"] > self.fill_threshold) | (
            df["gap_fill_down"] > self.fill_threshold
        )

        df.loc[gap_up_condition, "signal"] = 1  # 갭 업 포착
        df.loc[gap_down_condition, "signal"] = -1  # 갭 다운 포착
        df.loc[gap_fill_condition, "signal"] = 0  # 갭 채우기

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class NewsEventStrategy(BaseStrategy):
    """뉴스 이벤트 기반 전략 - 변동성 높은 시장에서 이벤트 포착"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.volatility_period = params.get("volatility_period", 20)
        self.volatility_threshold = params.get("volatility_threshold", 2.0)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 50)
        self.volume_threshold = params.get("volume_threshold", 2.0)
        self.price_threshold = params.get("price_threshold", 0.03)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """뉴스 이벤트 신호 생성"""
        df = df.copy()

        # 변동성 계산
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=self.volatility_period).std()
        df["volatility_ratio"] = (
            df["volatility"]
            / df["volatility"].rolling(window=self.volatility_period).mean()
        )

        # 거래량 분석
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

        # 가격 변동
        df["price_change"] = abs(df["close"].pct_change())

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 신호 생성
        df["signal"] = 0

        # 이벤트 신호: 높은 변동성 + 거래량 급증 + 큰 가격 변동
        event_condition = (
            (df["volatility_ratio"] > self.volatility_threshold)
            & (df["volume_ratio"] > self.volume_threshold)
            & (df["price_change"] > self.price_threshold)
        )

        # 이벤트 후 정리 신호: 변동성 감소
        cleanup_condition = (df["volatility_ratio"] < 0.5) & (df["volume_ratio"] < 1.0)

        df.loc[event_condition, "signal"] = 1  # 이벤트 포착
        df.loc[cleanup_condition, "signal"] = -1  # 이벤트 후 정리

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class RangeBreakoutStrategy(BaseStrategy):
    """범위 브레이크아웃 전략 - 횡보장에서 범위 돌파 포착"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.range_period = params.get("range_period", 20)
        self.breakout_threshold = params.get("breakout_threshold", 0.02)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 50)
        self.volume_threshold = params.get("volume_threshold", 1.5)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """범위 브레이크아웃 신호 생성"""
        df = df.copy()

        # 범위 계산
        df["range_high"] = df["high"].rolling(window=self.range_period).max()
        df["range_low"] = df["low"].rolling(window=self.range_period).min()
        df["range_mid"] = (df["range_high"] + df["range_low"]) / 2

        # 브레이크아웃 계산
        df["breakout_up"] = (df["close"] - df["range_high"]) / df["range_high"]
        df["breakout_down"] = (df["range_low"] - df["close"]) / df["range_low"]

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 거래량 분석
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

        # 신호 생성
        df["signal"] = 0

        # 상향 브레이크아웃: 상단 돌파 + 거래량 증가
        breakout_up_condition = (
            (df["breakout_up"] > self.breakout_threshold)
            & (df["volume_ratio"] > self.volume_threshold)
            & (df["rsi"] > self.rsi_threshold)
        )

        # 하향 브레이크아웃: 하단 돌파 + 거래량 증가
        breakout_down_condition = (
            (df["breakout_down"] > self.breakout_threshold)
            & (df["volume_ratio"] > self.volume_threshold)
            & (df["rsi"] < self.rsi_threshold)
        )

        # 범위 내 복귀
        range_return_condition = (
            df["close"].between(df["range_low"], df["range_high"])
        ) & (df["volume_ratio"] < 1.0)

        df.loc[breakout_up_condition, "signal"] = 1  # 상향 브레이크아웃
        df.loc[breakout_down_condition, "signal"] = -1  # 하향 브레이크아웃
        df.loc[range_return_condition, "signal"] = 0  # 범위 내 복귀

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class SupportResistanceStrategy(BaseStrategy):
    """지지/저항 기반 전략 - 횡보장에서 지지/저항선 활용"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.support_period = params.get("support_period", 20)
        self.resistance_period = params.get("resistance_period", 20)
        self.touch_threshold = params.get("touch_threshold", 0.01)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_oversold = params.get("rsi_oversold", 30)
        self.rsi_overbought = params.get("rsi_overbought", 70)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """지지/저항 신호 생성"""
        df = df.copy()

        # 지지/저항선 계산
        df["support"] = df["low"].rolling(window=self.support_period).min()
        df["resistance"] = df["high"].rolling(window=self.resistance_period).max()

        # 지지/저항 터치 계산
        df["support_touch"] = (df["low"] - df["support"]) / df["support"]
        df["resistance_touch"] = (df["resistance"] - df["high"]) / df["resistance"]

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 신호 생성
        df["signal"] = 0

        # 지지선 반등: 지지선 터치 + RSI 과매도
        support_bounce_condition = (
            (abs(df["support_touch"]) < self.touch_threshold)
            & (df["rsi"] < self.rsi_oversold)
            & (df["close"] > df["low"])
        )

        # 저항선 반락: 저항선 터치 + RSI 과매수
        resistance_rejection_condition = (
            (abs(df["resistance_touch"]) < self.touch_threshold)
            & (df["rsi"] > self.rsi_overbought)
            & (df["close"] < df["high"])
        )

        df.loc[support_bounce_condition, "signal"] = 1  # 지지선 반등
        df.loc[resistance_rejection_condition, "signal"] = -1  # 저항선 반락

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class OscillatorConvergenceStrategy(BaseStrategy):
    """오실레이터 수렴 전략 - 횡보장에서 여러 오실레이터의 수렴 포착"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.rsi_period = params.get("rsi_period", 14)
        self.stoch_period = params.get("stoch_period", 14)
        self.williams_period = params.get("williams_period", 14)
        self.convergence_threshold = params.get("convergence_threshold", 10)
        self.volume_threshold = params.get("volume_threshold", 1.2)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """오실레이터 수렴 신호 생성"""
        df = df.copy()

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 스토캐스틱 계산
        df["stoch_k"] = self._calculate_stochastic(df, self.stoch_period)
        df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

        # 윌리엄스 %R 계산
        df["williams_r"] = self._calculate_williams_r(df, self.williams_period)

        # 오실레이터 수렴 계산
        df["oscillator_avg"] = (
            df["rsi"] + df["stoch_k"] + (100 + df["williams_r"])
        ) / 3
        df["oscillator_std"] = df[["rsi", "stoch_k", "williams_r"]].std(axis=1)

        # 거래량 분석
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

        # 신호 생성
        df["signal"] = 0

        # 수렴 신호: 오실레이터 수렴 + 거래량 증가
        convergence_condition = (
            (df["oscillator_std"] < self.convergence_threshold)
            & (df["volume_ratio"] > self.volume_threshold)
            & (df["oscillator_avg"].between(30, 70))
        )

        # 발산 신호: 오실레이터 발산
        divergence_condition = (
            df["oscillator_std"] > self.convergence_threshold * 2
        ) & (df["volume_ratio"] < 0.8)

        df.loc[convergence_condition, "signal"] = 1  # 수렴 신호
        df.loc[divergence_condition, "signal"] = -1  # 발산 신호

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_stochastic(self, df: pd.DataFrame, period: int) -> pd.Series:
        """스토캐스틱 %K 계산"""
        low_min = df["low"].rolling(window=period).min()
        high_max = df["high"].rolling(window=period).max()
        stoch_k = 100 * (df["close"] - low_min) / (high_max - low_min)
        return stoch_k

    def _calculate_williams_r(self, df: pd.DataFrame, period: int) -> pd.Series:
        """윌리엄스 %R 계산"""
        low_min = df["low"].rolling(window=period).min()
        high_max = df["high"].rolling(window=period).max()
        williams_r = -100 * (high_max - df["close"]) / (high_max - low_min)
        return williams_r


class MomentumAccelerationStrategy(BaseStrategy):
    """모멘텀 가속도 전략 - 상승 추세에서 모멘텀 가속을 포착"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.momentum_period = params.get("momentum_period", 10)
        self.acceleration_threshold = params.get("acceleration_threshold", 0.02)
        self.volume_threshold = params.get("volume_threshold", 1.5)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 60)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """모멘텀 가속도 신호 생성"""
        df = df.copy()

        # 모멘텀 계산
        df["momentum"] = df["close"].pct_change(self.momentum_period)
        df["momentum_ma"] = df["momentum"].rolling(window=self.momentum_period).mean()
        df["momentum_acceleration"] = df["momentum"] - df["momentum_ma"]

        # 거래량 증가율
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 신호 생성
        df["signal"] = 0

        # 매수 신호: 모멘텀 가속 + 거래량 증가 + RSI 강세
        buy_condition = (
            (df["momentum_acceleration"] > self.acceleration_threshold)
            & (df["volume_ratio"] > self.volume_threshold)
            & (df["rsi"] > self.rsi_threshold)
            & (df["close"] > df["close"].rolling(window=20).mean())
        )

        # 매도 신호: 모멘텀 감속 또는 RSI 과매수
        sell_condition = (
            (df["momentum_acceleration"] < -self.acceleration_threshold)
            | (df["rsi"] > 80)
            | (df["close"] < df["close"].rolling(window=20).mean())
        )

        df.loc[buy_condition, "signal"] = 1
        df.loc[sell_condition, "signal"] = -1

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class VolumeBreakoutStrategy(BaseStrategy):
    """거래량 브레이크아웃 전략 - 상승 추세에서 거래량 증가와 함께하는 브레이크아웃 포착"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.volume_period = params.get("volume_period", 20)
        self.volume_threshold = params.get("volume_threshold", 2.0)
        self.price_period = params.get("price_period", 20)
        self.price_threshold = params.get("price_threshold", 0.02)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 50)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """거래량 브레이크아웃 신호 생성"""
        df = df.copy()

        # 거래량 분석
        df["volume_ma"] = df["volume"].rolling(window=self.volume_period).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]

        # 가격 분석
        df["price_ma"] = df["close"].rolling(window=self.price_period).mean()
        df["price_breakout"] = df["close"] > df["price_ma"] * (1 + self.price_threshold)

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 신호 생성
        df["signal"] = 0

        # 매수 신호: 거래량 급증 + 가격 브레이크아웃 + RSI 강세
        buy_condition = (
            (df["volume_ratio"] > self.volume_threshold)
            & (df["price_breakout"])
            & (df["rsi"] > self.rsi_threshold)
            & (df["close"] > df["close"].shift(1))
        )

        # 매도 신호: 거래량 감소 또는 가격 하락
        sell_condition = (
            (df["volume_ratio"] < 0.5)
            | (df["close"] < df["price_ma"])
            | (df["rsi"] > 80)
        )

        df.loc[buy_condition, "signal"] = 1
        df.loc[sell_condition, "signal"] = -1

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class TrendStrengthStrategy(BaseStrategy):
    """추세 강도 기반 전략 - 상승 추세의 강도를 측정하여 진입"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.short_period = params.get("short_period", 10)
        self.long_period = params.get("long_period", 50)
        self.adx_period = params.get("adx_period", 14)
        self.adx_threshold = params.get("adx_threshold", 25)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_threshold = params.get("rsi_threshold", 60)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """추세 강도 신호 생성"""
        df = df.copy()

        # 이동평균 계산
        df["sma_short"] = df["close"].rolling(window=self.short_period).mean()
        df["sma_long"] = df["close"].rolling(window=self.long_period).mean()

        # ADX 계산 (추세 강도)
        df["adx"] = self._calculate_adx(df, self.adx_period)

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 신호 생성
        df["signal"] = 0

        # 매수 신호: 강한 상승 추세 + RSI 강세
        buy_condition = (
            (df["sma_short"] > df["sma_long"])
            & (df["adx"] > self.adx_threshold)
            & (df["rsi"] > self.rsi_threshold)
            & (df["close"] > df["sma_short"])
        )

        # 매도 신호: 추세 약화 또는 RSI 과매수
        sell_condition = (
            (df["sma_short"] < df["sma_long"])
            | (df["adx"] < self.adx_threshold)
            | (df["rsi"] > 80)
        )

        df.loc[buy_condition, "signal"] = 1
        df.loc[sell_condition, "signal"] = -1

        return df

    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """ADX 계산"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range 계산
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        dm_plus = high - high.shift(1)
        dm_minus = low.shift(1) - low
        dm_plus = dm_plus.where(dm_plus > dm_minus, 0)
        dm_minus = dm_minus.where(dm_minus > dm_plus, 0)

        # Smoothed values
        tr_smooth = tr.rolling(window=period).mean()
        dm_plus_smooth = dm_plus.rolling(window=period).mean()
        dm_minus_smooth = dm_minus.rolling(window=period).mean()

        # DI 계산
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth

        # DX 계산
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)

        # ADX 계산
        adx = dx.rolling(window=period).mean()

        return adx

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class DefensiveHedgingStrategy(BaseStrategy):
    """방어적 헤징 전략 - 하락 추세에서 손실 최소화"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.volatility_period = params.get("volatility_period", 20)
        self.volatility_threshold = params.get("volatility_threshold", 0.03)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_oversold = params.get("rsi_oversold", 30)
        self.hedge_ratio = params.get("hedge_ratio", 0.3)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """방어적 헤징 신호 생성"""
        df = df.copy()

        # 변동성 계산
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=self.volatility_period).std()

        # RSI 계산
        df["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)

        # 이동평균


class DefensiveHedgingStrategy(BaseStrategy):
    """방어적 헤징 전략 - 하락 추세에서 손실 최소화"""

    def __init__(self, params: StrategyParams):
        super().__init__(params)
        self.volatility_period = params.get("volatility_period", 20)
        self.volatility_threshold = params.get("volatility_threshold", 0.03)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_oversold = params.get("rsi_oversold", 30)
        self.hedge_ratio = params.get("hedge_ratio", 0.3)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """방어적 헤징 신호 생성"""
        df = df.copy()
        df["signal"] = 0
        return df
