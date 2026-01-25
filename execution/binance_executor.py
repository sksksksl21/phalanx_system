# execution/binance_executor.py
# =========================================================
# [Phalanx Execution Module] BinanceExecutor (Live)
# Role: Real Order Execution + Accounting Bridge
# Design Goal:
# - VirtualExecutor와 **동일한 인터페이스 / 상태 모델**
# - 차이는 단 하나: 체결이 "가상"이 아니라 "실제"
# =========================================================

import logging
import time
import ccxt
import numpy as np
import pandas as pd


logger = logging.getLogger("PhalanxBinance")

BASE_FEE = 0.0005  # 백테스트와 동일 (실제 수수료는 체결 응답 우선)


class BinanceExecutor:
    """
    [IMPORTANT CONTRACT]
    - positions 구조는 VirtualExecutor와 동일
    - LiveEngine은 Executor가 Virtual/Binance인지 구분하지 않는다
    """


    def __init__(self, config):
        self.cfg = config

        # 자본 상태
        self.cash = 0.0
        self.equity = 0.0
        self.positions = {}
        self.history = []
        self.equity_curve = []

        self.MAX_POSITIONS = self.cfg.get("risk_settings", {}).get("max_positions", 5)

        self.exchange = ccxt.binance({
            "apiKey": (self.cfg.get("apiKey") or self.cfg.get("api_key") or self.cfg.get("API_KEY")),
            "secret": (self.cfg.get("secret") or self.cfg.get("secret_key") or self.cfg.get("SECRET_KEY")),
            "enableRateLimit": True,
            "timeout": 20000,
            "options": {
                "defaultType": "future",
                "adjustForTimeDifference": True
            }
        })

        # 초기 잔고 동기화
        self._sync_balance()

    # ======================================================
    # Balance / Precision
    # ======================================================
    def _sync_balance(self):
        try:
            bal = self.exchange.fetch_balance()
            usdt = bal.get("USDT", {})
            self.cash = float(usdt.get("free", 0))
            self.equity = float(usdt.get("total", self.cash))
        except Exception as e:
            logger.error(f"Balance sync failed: {e}")

    def fetch_balance(self):
        self._sync_balance()
        return {"USDT": {"free": self.cash, "total": self.equity}}

    def amount_to_precision(self, symbol, amount):
        if amount is None or (isinstance(amount, float) and np.isnan(amount)):
            return 0.0
        return float(self.exchange.amount_to_precision(symbol, amount))

    def price_to_precision(self, symbol, price):
        if price is None or (isinstance(price, float) and np.isnan(price)):
            return 0.0
        return float(self.exchange.price_to_precision(symbol, price))

    # ======================================================
    # Market Universe
    # ======================================================
    def get_top_targets(self):
        try:
            self.exchange.load_markets()
            tickers = self.exchange.fetch_tickers()
        except Exception as e:
            logger.error(f"fetch_tickers failed: {e}")
            return []

        markets = getattr(self.exchange, "markets", {}) or {}
        targets = []

        for s, t in tickers.items():
            if "/USDT:USDT" not in s:
                continue

            m = markets.get(s)
            if isinstance(m, dict):
                if m.get("active") is False:
                    continue
                info = m.get("info")
                if isinstance(info, dict):
                    status = str(info.get("status", "")).upper()
                    if status and status not in ("TRADING", "1"):
                        continue

            vol = 0.0
            if isinstance(t, dict):
                vol = t.get("quoteVolume", 0) or 0

            try:
                vol = float(vol)
            except Exception:
                vol = 0.0

            if vol > 0:
                targets.append((s, vol))

        targets.sort(key=lambda x: x[1], reverse=True)
        return [t[0] for t in targets][:20]

    # ======================================================



    def prepare_data(self, symbols, days=30, timeframe="15m", limit=1000):
        if symbols is None:
            return {}

        try:
            # markets 로딩은 시도만 (실패해도 진행)
            try:
                self.exchange.load_markets()
            except Exception as e:
                logger.warning(f"load_markets failed in prepare_data (continue): {e}")

            since = self.exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
            data_map = {}

            for sym in symbols:
                all_ohlcv = []
                temp_since = since
                retries = 3

                while True:
                    try:
                        ohlcv = self.exchange.fetch_ohlcv(sym, timeframe, since=temp_since, limit=limit)
                    except (ccxt.RequestTimeout, ccxt.NetworkError) as e:
                        retries -= 1
                        logger.warning(f"Network/Timeout on {sym}: {e} (retries left={retries})")
                        if retries <= 0:
                            all_ohlcv = []
                            break
                        time.sleep(1.0)
                        continue
                    except Exception as e:
                        logger.error(f"Error fetching {sym}: {e}")
                        all_ohlcv = []
                        break

                    if not ohlcv:
                        break

                    # 중복/역행 방지
                    last_ts = ohlcv[-1][0]
                    if all_ohlcv and last_ts <= all_ohlcv[-1][0]:
                        break

                    all_ohlcv.extend(ohlcv)
                    temp_since = last_ts + 1

                    if len(ohlcv) < limit:
                        break

                    time.sleep(0.2)

                if all_ohlcv:
                    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                    data_map[sym] = df

                time.sleep(0.3)  # 심볼 간 쿨다운

            return data_map

        except Exception as e:
            logger.error(f"prepare_data failed: {e}")
            return {}



    # Equity (MTM)
    # ======================================================
    def update_equity(self, current_prices):
        try:
            bal = self.exchange.fetch_balance()
            usdt = bal.get("USDT", {})
            self.cash = float(usdt.get("free", self.cash))
            self.equity = float(usdt.get("total", self.equity))
        except Exception as e:
            logger.error(f"Equity update failed: {e}")

        return self.equity

    def get_available_equity(self):
        self._sync_balance()
        return self.cash

    # ======================================================
    # 🔒 LIVE SAFETY: FETCH REAL POSITIONS
    # ======================================================
    def fetch_positions(self):
        """
        실계좌 선물 포지션 조회 (one-way 기준)
        return:
        {
          'BTC/USDT:USDT': {'side': 'LONG', 'amount': 0.01},
          'ETH/USDT:USDT': {'side': 'SHORT', 'amount': 0.5}
        }
        """
        positions = {}

        try:
            account = self.exchange.fetch_balance()
            raw_positions = account.get("info", {}).get("positions", [])

            for p in raw_positions:
                try:
                    amt = float(p.get("positionAmt", 0))
                    if amt == 0:
                        continue

                    symbol = p.get("symbol")
                    if not symbol:
                        continue

                    # Binance FUTURES symbol normalize: BTCUSDT -> BTC/USDT:USDT
                    if symbol.endswith("USDT"):
                        sym = symbol.replace("USDT", "/USDT:USDT")
                    else:
                        continue

                    side = "LONG" if amt > 0 else "SHORT"

                    positions[sym] = {
                        "side": side,
                        "amount": abs(amt)
                    }
                except Exception:
                    continue

        except Exception as e:
            logger.error(f"fetch_positions failed: {e}")

        return positions

    # ======================================================
    # Order Execution
    # ======================================================
    def create_order(self, symbol, side, amount):
        try:
            side_ccxt = "buy" if side == "LONG" else "sell"
            amt = self.amount_to_precision(symbol, amount)

            if amt <= 0:
                return None

            order = self.exchange.create_market_order(
                symbol=symbol,
                side=side_ccxt,
                amount=amt
            )

            for _ in range(5):
                order = self.exchange.fetch_order(order["id"], symbol)
                if order.get("status") == "closed":
                    break
                time.sleep(0.2)

            filled = float(order.get("filled", 0))
            cost = float(order.get("cost", 0))
            if filled <= 0:
                return None

            filled_price = cost / filled if cost > 0 else float(order.get("average", 0))

            fee_cost = 0.0
            fee_info = order.get("fee")
            if isinstance(fee_info, dict):
                fee_cost = float(fee_info.get("cost", 0))
            elif isinstance(order.get("fees"), list) and order["fees"]:
                fee_cost = sum(float(f.get("cost", 0)) for f in order["fees"])
            else:
                fee_cost = filled_price * filled * BASE_FEE

            leverage = self.cfg.get("risk_settings", {}).get("leverage", 1)
            margin = (filled_price * filled) / leverage

            self._sync_balance()

            return {
                "filled_price": filled_price,
                "filled_qty": filled,
                "fee": fee_cost,
                "margin": margin
            }

        except Exception as e:
            logger.error(f"Order failed {symbol} {side}: {e}")
            return None

    def close_position(self, symbol, price, reason="EXIT"):
        pos = self.positions.get(symbol)
        if not pos:
            return

        side = "sell" if pos["side"] == "LONG" else "buy"
        amt = self.amount_to_precision(symbol, pos.get("amount", 0))

        try:
            self.exchange.create_market_order(symbol, side, amt)
        except Exception as e:
            logger.error(f"Close failed {symbol}: {e}")

        self.positions.pop(symbol, None)
        self._sync_balance()
