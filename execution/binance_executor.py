import ccxt
import time
import logging
import pandas as pd

logger = logging.getLogger("PhalanxExec")

class BinanceExecutor:
    """
    [Phalanx Execution Module]
    Executor Name: Binance Future Executor
    Description: CCXT Wrapper for Binance Futures
    Role: Hands & Feet (Execution Layer)
    
    Responsibilities:
    1. Connect to Binance API (Auth)
    2. Execute Orders (Create, Cancel)
    3. Fetch Data (OHLCV, Ticker, Balance, Positions)
    4. Handle API Exceptions (Retry/Fail-safe)
    """

    def __init__(self, config):
        self.cfg = config
        self.exchange = None
        self._initialize_exchange()

    def _initialize_exchange(self):
        """거래소 인스턴스 초기화"""
        try:
            self.exchange = ccxt.binance({
                'apiKey': self.cfg.get('api_key', ''),
                'secret': self.cfg.get('secret_key', ''),
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # 선물 거래 강제
                    'adjustForTimeDifference': True
                }
            })
            self.exchange.load_markets()
            logger.info("Binance Executor Connected.")
        except Exception as e:
            logger.critical(f"Exchange Init Failed: {e}")
            raise e

    # ==========================================
    # 1. Market Data (Input Layer Support)
    # ==========================================
    def fetch_ohlcv(self, symbol, timeframe='15m', limit=200):
        """
        OHLCV 데이터 조회
        Failure Resilience: API 오류 시 빈 리스트 반환하여 시스템 멈춤 방지
        """
        try:
            # CCXT return: [[timestamp, open, high, low, close, volume], ...]
            return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except Exception as e:
            logger.warning(f"Fetch OHLCV Error ({symbol}): {e}")
            return []

    def fetch_ticker(self, symbol):
        """현재가 조회"""
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"Fetch Ticker Error ({symbol}): {e}")
            return {'last': 0.0}

    def fetch_balance(self):
        """잔고 조회"""
        try:
            return self.exchange.fetch_balance()
        except Exception as e:
            logger.error(f"Fetch Balance Error: {e}")
            return {'USDT': {'free': 0.0, 'total': 0.0}}

    def fetch_positions(self):
        """
        보유 포지션 조회 (Sync 용도)
        """
        try:
            return self.exchange.fetch_positions()
        except Exception as e:
            logger.error(f"Fetch Positions Error: {e}")
            return []

    # ==========================================
    # 2. Order Execution (Action Layer)
    # ==========================================
    def create_order(self, symbol, side, amount, order_type='market', params={}):
        """
        주문 생성 (핵심 기능)
        :return: (order_object, fill_price)
        """
        try:
            # 안전장치: 수량이 0 이하면 주문 거부
            if amount <= 0:
                logger.warning(f"Order Rejected: Zero Amount ({symbol})")
                return None, 0.0

            order = self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                params=params
            )
            
            # 시장가의 경우 체결가 추정 (average or price)
            fill_price = order.get('average') or order.get('price')
            
            # 체결가가 없으면(시장가 즉시 체결 전) 현재가로 대체
            if fill_price is None or fill_price == 0:
                ticker = self.fetch_ticker(symbol)
                fill_price = ticker['last']

            logger.info(f"Order Executed: {symbol} {side} {amount} @ {fill_price}")
            return order, float(fill_price)

        except Exception as e:
            logger.error(f"Order Failed ({symbol} {side}): {e}")
            raise e # 주문 실패는 상위 레벨에서 처리하도록 전파

    # ==========================================
    # 3. Risk Management Support (Utils)
    # ==========================================
    def set_leverage(self, leverage, symbol):
        """레버리지 설정"""
        try:
            # 바이낸스 전용 로직
            self.exchange.set_leverage(leverage, symbol)
        except Exception as e:
            # 이미 설정되어 있거나 오류 발생 시 로그만 남김 (Blocking 방지)
            # logger.debug(f"Set Leverage Skip: {e}")
            pass

    def amount_to_precision(self, symbol, amount):
        """
        수량 정밀도 보정 (RiskControl에서 호출)
        Exchange Metadata를 사용하여 정확한 주문 가능 수량으로 변환
        """
        try:
            return self.exchange.amount_to_precision(symbol, amount)
        except Exception as e:
            logger.error(f"Precision Error ({symbol}): {e}")
            # 실패 시 소수점 4자리 버림으로 안전 처리
            return float(int(amount * 10000) / 10000)