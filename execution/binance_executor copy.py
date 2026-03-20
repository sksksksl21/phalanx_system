# execution/binance_executor.py
# =========================================================
# [Phalanx Execution Module] BinanceExecutor (Live - Futures)
# =========================================================
# ✅ LIVE 안정성 보강
# - 잔고: futures(USDT-M) 우선 조회 + info fallback(availableBalance/walletBalance)
# - get_available_equity(): "가용 USDT" 반환 (sizing 기준)
# - equity: "총(지갑) USDT" 또는 marginBalance 계열을 최대한 반영
# - 나머지 계약(orders/positions)은 기존 유지
# =========================================================

import logging
import time
import ccxt
import numpy as np
import pandas as pd

logger = logging.getLogger("PhalanxBinance")

BASE_FEE = 0.0005


class BinanceExecutor:
    """
    [IMPORTANT CONTRACT]
    - positions 구조는 VirtualExecutor와 동일 (엔진이 상태 권위)
    - LiveEngine은 Executor가 Virtual/Binance인지 구분하지 않는다
    """

    def __init__(self, config):
        self.cfg = config

        # 자본 상태
        self.cash = 0.0     # ✅ "가용 USDT" (available / free)
        self.equity = 0.0   # ✅ "총 USDT" (wallet/total-ish)

        # ⚠️ 상태 단일 진실원: positions는 엔진이 저장/복구한다.
        self.positions = {}
        self.history = []
        self.equity_curve = []

        rs = self.cfg.get("risk_settings", {}) or {}
        mp = rs.get("max_positions", None)
        if mp is None:
            mp = rs.get("max_open_positions", 5)
        try:
            mp = int(mp)
            if mp <= 0:
                mp = 1
        except Exception:
            mp = 5
        self.MAX_POSITIONS = mp

        self.exchange = ccxt.binance({
            "apiKey": (self.cfg.get("apiKey") or self.cfg.get("api_key") or self.cfg.get("API_KEY")),
            "secret": (self.cfg.get("secret") or self.cfg.get("secret_key") or self.cfg.get("SECRET_KEY")),
            "enableRateLimit": True,
            "timeout": 20000,
            "options": {
                "defaultType": "future",  # ✅ USDT-M Futures
                "adjustForTimeDifference": True
            }
        })

        # ✅ 서버 시간 오프셋(밀리초) 캐시
        self._time_offset_ms = 0
        self._time_sync_at = 0.0

        # ✅ 부팅 시 1회 강제 타임싱크 (Binance 권위)
        self._sync_time_best_effort(force=True)

        # balance sync
        self._sync_balance()

    def _sync_time_best_effort(self, force=False, min_interval_sec=30.0):
        """
        ✅ Binance 서버시간 권위화 + CCXT timeDifference 갱신
        - 핵심: exchange.load_time_difference() 호출로 ccxt 내부 timeDifference를 실제 적용
        - 그래도 실패할 수 있으니 fetch_time 기반 offset 캐시는 유지(관측/디버그용)
        """
        now = time.time()
        if (not force) and (now - float(getattr(self, "_time_sync_at", 0.0))) < float(min_interval_sec):
            return

        # ✅ recvWindow를 options에도 박아두면 ccxt 내부 호출에 기본값으로 전파되는 케이스가 있음
        try:
            rw = self._signed_params({}).get("recvWindow", 10000)
            opts = (getattr(self.exchange, "options", None) or {})
            if isinstance(opts, dict):
                opts["recvWindow"] = int(rw)
                self.exchange.options = opts
        except Exception:
            pass

        # 1) ✅ CCXT 내부 timeDifference 갱신 (가장 중요)
        try:
            if hasattr(self.exchange, "load_time_difference"):
                self.exchange.load_time_difference()
        except Exception as e:
            logger.warning(f"Time sync load_time_difference failed: {e}")

        # 2) 관측/로깅용 offset 캐시(기존 유지)
        try:
            server_ms = self.exchange.fetch_time()
            local_ms = int(time.time() * 1000)
            offset = int(server_ms) - int(local_ms)
            self._time_offset_ms = int(offset)
            self._time_sync_at = float(now)
            logger.info(f"⏱️ Time synced (offset_ms={self._time_offset_ms})")
        except Exception as e:
            logger.warning(f"Time sync failed: {e}")


    def _signed_params(self, params: dict = None) -> dict:
        """
        ✅ Binance signed endpoints용 공통 파라미터
        - recvWindow를 강제해 -1021 민감도를 낮춘다.
        - 필요 시 type='future' 같은 기존 params도 보존한다.
        """
        out = {}
        try:
            if isinstance(params, dict):
                out.update(params)
        except Exception:
            pass

        try:
            ss = (self.cfg.get("system_settings", {}) or {})
            rw = int(ss.get("binance_recv_window_ms", 10000) or 10000)
            if rw < 5000:
                rw = 5000
            if rw > 60000:
                rw = 60000
        except Exception:
            rw = 10000

        # ✅ 항상 덮어쓰기(외부가 더 작은 값을 넣어도 방어)
        out["recvWindow"] = int(rw)
        return out
    # ======================================================
    # Balance / Precision
    # ======================================================
    def _fetch_balance_future_best_effort(self):
        """
        선물 잔고를 확실히 잡기 위한 best-effort.
        - 우선: params={"type":"future", "recvWindow":...} 로 fetch_balance
        - 실패: 기본 fetch_balance(params={"recvWindow":...})
        """
        try:
            return self.exchange.fetch_balance(params=self._signed_params({"type": "future"}))
        except Exception:
            return self.exchange.fetch_balance(params=self._signed_params({}))

    def _extract_usdt_balances(self, bal: dict):
        """
        다양한 반환 형태에서 USDT 가용/총을 최대한 안정적으로 추출.
        return: (free_usdt, total_usdt)
        """
        free_usdt = None
        total_usdt = None

        if not isinstance(bal, dict):
            return 0.0, 0.0

        # 1) 가장 일반적인 형태: bal["USDT"]["free"/"total"]
        try:
            usdt = bal.get("USDT", {}) or {}
            if isinstance(usdt, dict):
                if usdt.get("free") is not None:
                    free_usdt = float(usdt.get("free") or 0)
                if usdt.get("total") is not None:
                    total_usdt = float(usdt.get("total") or 0)
                # 어떤 경우에는 used + free 구조
                if total_usdt is None and (usdt.get("used") is not None) and (usdt.get("free") is not None):
                    total_usdt = float(usdt.get("used") or 0) + float(usdt.get("free") or 0)
        except Exception:
            pass

        # 2) ccxt unified: bal.get("free", {}).get("USDT") / bal.get("total", {}).get("USDT")
        try:
            if free_usdt is None:
                f = (bal.get("free", {}) or {})
                if isinstance(f, dict) and f.get("USDT") is not None:
                    free_usdt = float(f.get("USDT") or 0)
            if total_usdt is None:
                t = (bal.get("total", {}) or {})
                if isinstance(t, dict) and t.get("USDT") is not None:
                    total_usdt = float(t.get("USDT") or 0)
        except Exception:
            pass

        # 3) 바이낸스 선물 info fallback: availableBalance / walletBalance / marginBalance
        #    (binance futures account 포맷은 ccxt 버전에 따라 다를 수 있음)
        try:
            info = bal.get("info", {}) or {}
            # 케이스 A: info가 dict이고 assets 리스트가 있음
            assets = info.get("assets", None)
            if isinstance(assets, list):
                for a in assets:
                    if not isinstance(a, dict):
                        continue
                    if str(a.get("asset", "")).upper() != "USDT":
                        continue
                    if free_usdt is None and a.get("availableBalance") is not None:
                        free_usdt = float(a.get("availableBalance") or 0)
                    if total_usdt is None:
                        # walletBalance가 가장 “총”에 가깝고, marginBalance도 대안
                        if a.get("walletBalance") is not None:
                            total_usdt = float(a.get("walletBalance") or 0)
                        elif a.get("marginBalance") is not None:
                            total_usdt = float(a.get("marginBalance") or 0)
                    break

            # 케이스 B: info 내부에 totalWalletBalance/availableBalance류가 바로 있는 경우
            if free_usdt is None:
                for k in ("availableBalance", "available", "totalAvailableBalance"):
                    if info.get(k) is not None:
                        free_usdt = float(info.get(k) or 0)
                        break

            if total_usdt is None:
                for k in ("totalWalletBalance", "walletBalance", "totalMarginBalance", "marginBalance"):
                    if info.get(k) is not None:
                        total_usdt = float(info.get(k) or 0)
                        break
        except Exception:
            pass

        # 최종 보정
        try:
            free_usdt = float(free_usdt) if free_usdt is not None else 0.0
        except Exception:
            free_usdt = 0.0

        try:
            total_usdt = float(total_usdt) if total_usdt is not None else None
        except Exception:
            total_usdt = None

        if total_usdt is None:
            total_usdt = free_usdt

        # 음수/NaN 방지
        if not np.isfinite(free_usdt) or free_usdt < 0:
            free_usdt = 0.0
        if not np.isfinite(total_usdt) or total_usdt < 0:
            total_usdt = max(free_usdt, 0.0)

        return free_usdt, total_usdt

    def _sync_balance(self):
        """
        ✅ LIVE sizing의 기준이 되는 값들을 정확히 동기화
        - self.cash   : "가용 USDT" (available/free)
        - self.equity : "총 USDT" (wallet/total-ish)
        - ✅ -1021(서버시간 불일치) 발생 시: time sync -> 재시도 1회
        """
        def _is_ts_error(e: Exception) -> bool:
            s = str(e)
            return ("-1021" in s) or ("Timestamp for this request" in s)

        try:
            bal = self._fetch_balance_future_best_effort()
            free_usdt, total_usdt = self._extract_usdt_balances(bal)

            self.cash = float(free_usdt)
            self.equity = float(total_usdt)
            return

        except Exception as e:
            if _is_ts_error(e):
                logger.error(f"Balance sync failed (ts) -> time sync & retry: {e}")
                try:
                    self._sync_time_best_effort(force=True)
                except Exception:
                    pass

                try:
                    bal = self._fetch_balance_future_best_effort()
                    free_usdt, total_usdt = self._extract_usdt_balances(bal)
                    self.cash = float(free_usdt)
                    self.equity = float(total_usdt)
                    return
                except Exception as e2:
                    logger.error(f"Balance sync retry failed: {e2}")
                    return

            logger.error(f"Balance sync failed: {e}")
            return


    def fetch_balance(self):
        self._sync_balance()
        return {"USDT": {"free": self.cash, "total": self.equity}}

    # ======================================================
    # Ticker (Engine compatibility)
    # ======================================================
    def fetch_ticker(self, symbol: str) -> dict:
        """
        LiveEngine 호환용 wrapper.
        ccxt의 fetch_ticker를 그대로 노출한다.
        """
        try:
            return self.exchange.fetch_ticker(symbol) or {}
        except Exception as e:
            logger.warning(f"fetch_ticker failed {symbol}: {e}")
            return {}


    def amount_to_precision(self, symbol, amount):
        if amount is None or (isinstance(amount, float) and np.isnan(amount)):
            return 0.0
        try:
            return float(self.exchange.amount_to_precision(symbol, amount))
        except Exception:
            try:
                return float(amount)
            except Exception:
                return 0.0

    def price_to_precision(self, symbol, price):
        if price is None or (isinstance(price, float) and np.isnan(price)):
            return 0.0
        try:
            return float(self.exchange.price_to_precision(symbol, price))
        except Exception:
            try:
                return float(price)
            except Exception:
                return 0.0
    def _ceil_amount_step(self, symbol: str, amount: float) -> float:
        """
        amount_to_precision()은 보통 '버림'이라 minNotional 충족을 깨뜨릴 수 있음.
        그래서 '올림'으로 step/precision에 맞춘다.
        """
        try:
            # load_markets 안 된 상태면 market()가 실패할 수 있으니 방어
            try:
                self.exchange.load_markets()
            except Exception:
                pass

            m = self.exchange.market(symbol)
            prec = int((m.get("precision", {}) or {}).get("amount", 6))
            step = 10 ** (-prec)

            amt = float(amount)
            if amt <= 0:
                return 0.0

            units = int(np.ceil(amt / step))
            out = units * step
            out = float(f"{out:.{prec}f}")
            return out
        except Exception:
            try:
                amt = float(amount)
                if amt <= 0:
                    return 0.0
                step = 1e-6
                out = np.ceil(amt / step) * step
                return float(f"{out:.6f}")
            except Exception:
                return 0.0

    def _bump_to_min_notional(self, symbol: str, amount: float, min_notional: float, est_px: float) -> float:
        """
        est_px 기준으로 notional >= min_notional 되게 수량을 올림(ceil) 보정.
        """
        try:
            amt = float(amount)
            if amt <= 0:
                return 0.0

            px = float(est_px)
            mn = float(min_notional)

            # px/mn이 유효하지 않으면 precision 올림만
            if px <= 0 or mn <= 0:
                return self._ceil_amount_step(symbol, amt)

            notional = px * amt
            if notional + 1e-12 >= mn:
                return self._ceil_amount_step(symbol, amt)

            need = (mn / px) * 1.001  # 약간 여유
            need = self._ceil_amount_step(symbol, need)

            # 올림 후에도 혹시 부족하면 한 번 더 올림
            if px * need + 1e-12 < mn:
                need = self._ceil_amount_step(symbol, need * 1.001)

            return need
        except Exception:
            return self._ceil_amount_step(symbol, amount)

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

        for s, t in (tickers or {}).items():
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

            vol = float(t.get("quoteVolume", 0) or 0)
            if vol > 0:
                targets.append((s, vol))

        targets.sort(key=lambda x: x[1], reverse=True)
        return [t[0] for t in targets][:30]

    # ======================================================
    # Historical OHLCV (15m)
    # ======================================================
    def prepare_data(self, symbols, days=30, timeframe="15m", limit=1000):
        """
        ✅ 개선:
        - min_history_bars_15m(설정) 미만의 심볼은 universe 후보에서 제거하여
        common timeline(교집합)이 짧은 신규심볼 하나 때문에 전체가 눌리는 현상을 방지
        - 단, 보유 포지션 심볼(self.positions)은 관리 필요 -> 예외로 유지
        """
        if symbols is None:
            return {}

        # ---- min bars from config (default: 260) ----
        try:
            ss = (self.cfg.get("system_settings", {}) or {})
            min_bars = int(ss.get("min_history_bars_15m", 260) or 260)
            if min_bars < 200:
                min_bars = 200
        except Exception:
            min_bars = 260

        # ---- Binance max limit paging hint (15m은 보통 1500까지) ----
        try:
            # 너무 작은 limit이면 paging 회수가 증가하므로 상향
            limit = int(limit or 1000)
            if limit < 500:
                limit = 500
            if limit > 1500:
                limit = 1500
        except Exception:
            limit = 1000

        try:
            try:
                self.exchange.load_markets()
            except Exception as e:
                logger.warning(f"load_markets failed in prepare_data (continue): {e}")

            since = self.exchange.milliseconds() - (int(days) * 24 * 60 * 60 * 1000)
            data_map = {}

            # positions 예외 보존
            pos_syms = set((self.positions or {}).keys())

            for sym in symbols:
                all_ohlcv = []
                temp_since = since
                retries = 3

                while True:
                    try:
                        ohlcv = self.exchange.fetch_ohlcv(sym, timeframe, since=temp_since, limit=limit)
                    except (ccxt.RequestTimeout, ccxt.NetworkError) as e:
                        retries -= 1
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

                    last_ts = ohlcv[-1][0]
                    if all_ohlcv and last_ts <= all_ohlcv[-1][0]:
                        break

                    all_ohlcv.extend(ohlcv)
                    temp_since = last_ts + 1

                    # 페이지가 덜 찼으면 끝
                    if len(ohlcv) < limit:
                        break

                    time.sleep(0.2)

                if all_ohlcv:
                    df = pd.DataFrame(
                        all_ohlcv,
                        columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )

                    # ✅ 핵심: 히스토리 짧은 신규 심볼 제거 (단, 포지션 심볼은 유지)
                    try:
                        n = int(len(df))
                    except Exception:
                        n = 0

                    if n < min_bars and sym not in pos_syms:
                        logger.warning(
                            f"⚠️ DATA_DROP_SHORT | {sym} bars={n} < min_bars={min_bars} "
                            f"(likely new listing) -> drop from universe"
                        )
                    else:
                        data_map[sym] = df

                time.sleep(0.3)

            return data_map

        except Exception as e:
            logger.error(f"prepare_data failed: {e}")
            return {}

    # ======================================================
    # Equity (MTM)
    # ======================================================
    def update_equity(self, current_prices):
        # 실전에서는 잔고 동기화가 진실.
        self._sync_balance()
        return self.equity

    def get_available_equity(self):
        # ✅ sizing 기준: 가용 USDT
        self._sync_balance()
        return self.cash

    # ======================================================
    # 🔒 LIVE SAFETY: FETCH REAL POSITIONS
    # ======================================================
    def fetch_positions(self):
        """
        실제 선물 계좌 포지션 조회.
        return: {
        "BTC/USDT:USDT": {"side":"LONG","amount":0.01,"entry_price":..., "margin":...}
        }

        ✅ FIX
        - isolatedWallet(특히 cross 모드에서 0이 자주 나옴)을 '우선'으로 잡지 않는다.
        - margin 후보들을 모두 수집한 뒤:
            1) >0 인 값이 있으면 그 중 가장 의미있는(보통 initial/positionInitialMargin) 값을 사용
            2) 전부 0/None이면 0.0 또는 None 처리
        - entryPrice/positionAmt parsing 안정화
        - 심볼 매핑: XXXUSDT -> XXX/USDT:USDT 유지
        """
        positions = {}
        try:
            account = self._fetch_balance_future_best_effort()
            info = (account.get("info", {}) or {})

            # ccxt/버전별로 info 구조가 다를 수 있어 방어
            raw_positions = []
            if isinstance(info, dict):
                raw_positions = info.get("positions", []) or []
            elif isinstance(info, list):
                # 일부 케이스에서 info 자체가 positions list일 수 있음
                raw_positions = info
            else:
                raw_positions = []

            for p in raw_positions:
                if not isinstance(p, dict):
                    continue

                # --- amount ---
                try:
                    amt = float(p.get("positionAmt", 0) or 0)
                except Exception:
                    continue
                if abs(amt) <= 0:
                    continue

                # --- symbol id ---
                symbol = p.get("symbol")
                if not symbol:
                    continue
                symbol = str(symbol).upper()
                if not symbol.endswith("USDT"):
                    continue

                sym = symbol.replace("USDT", "/USDT:USDT")
                side = "LONG" if amt > 0 else "SHORT"

                # --- entry price ---
                entry_price = None
                try:
                    ep = p.get("entryPrice", None)
                    if ep is not None:
                        epf = float(ep)
                        if epf > 0:
                            entry_price = epf
                except Exception:
                    entry_price = None

                # --- margin (best-effort, robust) ---
                # Binance futures position dict may include:
                # initialMargin, positionInitialMargin, maintMargin, isolatedWallet, etc.
                margin_candidates = []

                def _push_margin(key: str):
                    try:
                        if key in p and p.get(key) is not None:
                            v = float(p.get(key))
                            # NaN/inf 방지
                            if np.isfinite(v):
                                margin_candidates.append((key, v))
                    except Exception:
                        pass

                # ✅ 우선순위: "실제 포지션에 걸린 마진" 계열을 먼저
                for k in ("positionInitialMargin", "initialMargin", "maintMargin", "isolatedWallet"):
                    _push_margin(k)

                margin = None
                if margin_candidates:
                    # 1) 양수 값이 하나라도 있으면 양수 중에서 우선순위 기반 선택
                    positives = [(k, v) for (k, v) in margin_candidates if v > 0]
                    if positives:
                        # 우선순위 가중
                        priority = {
                            "positionInitialMargin": 0,
                            "initialMargin": 1,
                            "maintMargin": 2,
                            "isolatedWallet": 3,
                        }
                        positives.sort(key=lambda kv: priority.get(kv[0], 99))
                        margin = float(positives[0][1])
                    else:
                        # 전부 0이면 0.0으로 두되, None이 낫다면 None으로 바꿔도 됨
                        margin = 0.0

                out = {
                    "side": side,
                    "amount": abs(float(amt)),
                }
                if entry_price is not None:
                    out["entry_price"] = float(entry_price)
                if margin is not None:
                    out["margin"] = float(margin)

                positions[sym] = out

        except Exception as e:
            logger.error(f"fetch_positions failed: {e}")

        return positions

    # ======================================================
    # Internal: set leverage (best-effort)
    # ======================================================
    def _set_leverage_best_effort(self, symbol, leverage):
        try:
            lev = float(leverage)
            if lev <= 0:
                return
        except Exception:
            return

        try:
            if hasattr(self.exchange, "set_leverage"):
                self.exchange.set_leverage(int(round(lev)), symbol)
                return
        except Exception as e:
            logger.warning(f"set_leverage failed (unified) {symbol} lev={leverage}: {e}")

        try:
            if hasattr(self.exchange, "fapiPrivate_post_leverage"):
                market = self.exchange.market(symbol)
                req = {"symbol": market["id"], "leverage": int(round(lev))}
                self.exchange.fapiPrivate_post_leverage(req)
        except Exception as e:
            logger.warning(f"set_leverage failed (raw) {symbol} lev={leverage}: {e}")

    # ======================================================
    # Order Execution
    # ======================================================
    def create_order(self, symbol, side, amount, poll_timeout_sec=5.0):
        """
        시장가 진입.
        return:
            {"filled_price":..., "filled_qty":..., "fee":..., "margin":...}
        실패 시 None

        ✅ -1021(서버시간 불일치) 발생 시: time sync -> 재시도 1~2회
        """
        def _is_ts_error(e: Exception) -> bool:
            s = str(e)
            return ("-1021" in s) or ("Timestamp for this request" in s)

        def _create_once():
            # ------------------------------
            # 1) side 정규화: LONG/SHORT/BUY/SELL 모두 허용
            # ------------------------------
            s = str(side or "").strip().upper()
            side_map = {
                "LONG": "buy",
                "BUY": "buy",
                "BULL": "buy",
                "SHORT": "sell",
                "SELL": "sell",
                "BEAR": "sell",
            }
            side_ccxt = side_map.get(s, None)
            if side_ccxt is None:
                logger.error(f"Order failed {symbol} {side}: invalid side")
                return None

            # ------------------------------
            # 2) leverage best-effort
            # ------------------------------
            rs = self.cfg.get("risk_settings", {}) or {}
            leverage = rs.get("leverage", 1)
            self._set_leverage_best_effort(symbol, leverage)

            # ------------------------------
            # 3) amount precision + sanity
            # ------------------------------
            amt = self.amount_to_precision(symbol, amount)
            try:
                amt = float(amt)
            except Exception:
                return None
            if amt <= 0:
                return None

            # ------------------------------
            # 4) ✅ Min Notional(기본 20 USDT) 보정
            # ------------------------------
            try:
                min_notional = float(rs.get("min_notional_usdt", 20) or 20)
                if min_notional < 0:
                    min_notional = 20.0
            except Exception:
                min_notional = 20.0

            est_px = None
            try:
                t = self.exchange.fetch_ticker(symbol)
                est_px = float(t.get("last", 0) or 0)
            except Exception:
                est_px = None

            if est_px is not None and est_px > 0 and min_notional > 0:
                notional = est_px * amt
                if notional + 1e-12 < min_notional:
                    need_amt = self._bump_to_min_notional(symbol, amt, min_notional, est_px)  # ✅ 올림 기반 보정
                    if need_amt > amt:
                        logger.info(
                            f"Order notional too small -> bump amount {amt} -> {need_amt} "
                            f"(px≈{est_px:.2f}, notional≈{est_px*amt:.2f} -> {est_px*need_amt:.2f})"
                        )
                        amt = need_amt

                    if need_amt > 0 and need_amt > amt:
                        try:
                            lev = float(rs.get("leverage", 1) or 1)
                            if lev <= 0:
                                lev = 1.0
                        except Exception:
                            lev = 1.0

                        need_notional = est_px * need_amt
                        need_margin = need_notional / lev

                        avail = float(getattr(self, "cash", 0.0) or 0.0)
                        if need_margin > avail + 1e-9:
                            logger.error(
                                f"Order blocked {symbol} {side} : "
                                f"min_notional={min_notional} requires amt={need_amt} "
                                f"(notional≈{need_notional:.2f}, margin≈{need_margin:.2f}) "
                                f"but avail_cash={avail:.2f}"
                            )
                            return None

                        logger.info(
                            f"Order notional too small -> bump amount {amt} -> {need_amt} "
                            f"(px≈{est_px:.2f}, notional≈{notional:.2f} -> {need_notional:.2f})"
                        )
                        amt = need_amt

            # ------------------------------
            # 5) create market order
            # ------------------------------
            order = self.exchange.create_market_order(symbol, side_ccxt, amt, params=self._signed_params({}))
            oid = order.get("id") if isinstance(order, dict) else None

            filled = float((order or {}).get("filled", 0) or 0)
            cost = float((order or {}).get("cost", 0) or 0)
            average = (order or {}).get("average", None)

            t0 = time.time()
            while time.time() - t0 < float(poll_timeout_sec):
                if not oid:
                    break
                try:
                    o2 = self.exchange.fetch_order(oid, symbol, params=self._signed_params({}))
                    if isinstance(o2, dict):
                        order = o2
                        filled = float(order.get("filled", filled) or filled)
                        cost = float(order.get("cost", cost) or cost)
                        average = order.get("average", average)

                        status = str(order.get("status", "")).lower()
                        if status == "closed":
                            break
                        time.sleep(0.15 if filled > 0 else 0.2)
                    else:
                        time.sleep(0.2)
                except Exception:
                    time.sleep(0.2)

            if filled <= 0:
                return None

            # ------------------------------
            # 6) filled_price 추정
            # ------------------------------
            filled_price = None
            try:
                if cost > 0:
                    filled_price = float(cost) / float(filled)
            except Exception:
                filled_price = None

            if filled_price is None or filled_price <= 0:
                try:
                    if average is not None and float(average) > 0:
                        filled_price = float(average)
                except Exception:
                    filled_price = None

            if filled_price is None or filled_price <= 0:
                try:
                    t = self.exchange.fetch_ticker(symbol)
                    filled_price = float(t.get("last", 0) or 0)
                except Exception:
                    filled_price = 0.0

            # ------------------------------
            # 7) fee / margin
            # ------------------------------
            fee_cost = None
            try:
                fee_info = (order or {}).get("fee")
                if isinstance(fee_info, dict):
                    fee_cost = float(fee_info.get("cost", 0) or 0)
            except Exception:
                fee_cost = None

            if fee_cost is None:
                fee_cost = float(filled_price) * float(filled) * BASE_FEE

            try:
                lev = float(rs.get("leverage", 1) or 1)
                if lev <= 0:
                    lev = 1.0
            except Exception:
                lev = 1.0

            margin = (float(filled_price) * float(filled)) / float(lev)

            self._sync_balance()

            return {
                "filled_price": float(filled_price),
                "filled_qty": float(filled),
                "requested_qty": float(amt),   # ✅ 최종 주문수량(보정 후)
                "fee": float(fee_cost),
                "margin": float(margin),
            }

        # ---- main with retries for -1021
        try:
            return _create_once()
        except Exception as e:
            if _is_ts_error(e):
                logger.error(f"Order failed (ts) {symbol} {side}: {e} -> time sync & retry")
                try:
                    self._sync_time_best_effort(force=True)
                except Exception:
                    pass

                # retry 1
                try:
                    return _create_once()
                except Exception as e2:
                    if _is_ts_error(e2):
                        logger.error(f"Order retry failed (ts) {symbol} {side}: {e2} -> time sync & retry2")
                        try:
                            self._sync_time_best_effort(force=True)
                        except Exception:
                            pass
                        try:
                            return _create_once()
                        except Exception as e3:
                            logger.error(f"Order final failed {symbol} {side}: {e3}")
                            return None

                    logger.error(f"Order retry failed {symbol} {side}: {e2}")
                    return None

            logger.error(f"Order failed {symbol} {side}: {e}")
            return None

    def create_limit_order(self, symbol, side, amount, price, time_in_force="GTC", poll_timeout_sec=3.0):
        """
        ✅ 리밋 진입(오픈오더) 생성
        return:
        {"order_id":..., "requested_qty":..., "requested_price":..., "status":..., "raw":...}
        실패 시 None

        ✅ -1021 발생 시: time sync -> 재시도 1~2회
        ✅ precision(amount/price) + minNotional(설정) 보정 포함
        """
        def _is_ts_error(e: Exception) -> bool:
            s = str(e)
            return ("-1021" in s) or ("Timestamp for this request" in s)

        def _create_once():
            # 1) side 정규화
            s = str(side or "").strip().upper()
            side_map = {
                "LONG": "buy", "BUY": "buy", "BULL": "buy",
                "SHORT": "sell", "SELL": "sell", "BEAR": "sell",
            }
            side_ccxt = side_map.get(s, None)
            if side_ccxt is None:
                logger.error(f"Limit order failed {symbol} {side}: invalid side")
                return None

            # 2) leverage best-effort (시장가와 동일하게 유지)
            rs = self.cfg.get("risk_settings", {}) or {}
            leverage = rs.get("leverage", 1)
            self._set_leverage_best_effort(symbol, leverage)

            # 3) price/amount precision + sanity
            px = self.price_to_precision(symbol, price)
            amt = self.amount_to_precision(symbol, amount)
            try:
                px = float(px)
                amt = float(amt)
            except Exception:
                return None
            if px <= 0 or amt <= 0:
                return None

            # 4) ✅ Min Notional 보정 (price 기준)
            try:
                min_notional = float(rs.get("min_notional_usdt", 20) or 20)
                if min_notional < 0:
                    min_notional = 20.0
            except Exception:
                min_notional = 20.0

            if min_notional > 0:
                notional = px * amt
                if notional + 1e-12 < min_notional:
                    need_amt = self._bump_to_min_notional(symbol, amt, min_notional, px)
                    if need_amt > amt:
                        logger.info(
                            f"Limit notional too small -> bump amount {amt} -> {need_amt} "
                            f"(px={px:.8g}, notional={px*amt:.6g} -> {px*need_amt:.6g})"
                        )
                        amt = float(need_amt)

            # 5) create limit order (signed params + TIF)
            params = self._signed_params({})
            try:
                if time_in_force:
                    params["timeInForce"] = str(time_in_force)
            except Exception:
                pass

            order = self.exchange.create_order(
                symbol=symbol,
                type="limit",
                side=side_ccxt,
                amount=float(amt),
                price=float(px),
                params=params,
            )

            oid = order.get("id") if isinstance(order, dict) else None
            status = str((order or {}).get("status", "")).lower()

            # 6) (optional) 짧게 한번만 조회해서 즉시 reject/closed 여부 반영
            if oid and poll_timeout_sec and poll_timeout_sec > 0:
                t0 = time.time()
                while time.time() - t0 < float(poll_timeout_sec):
                    try:
                        o2 = self.exchange.fetch_order(str(oid), symbol, params=self._signed_params({}))
                        if isinstance(o2, dict):
                            order = o2
                            status = str((order or {}).get("status", "")).lower()
                            # open/closed/canceled 중 하나면 탈출
                            if status in ("open", "closed", "canceled", "cancelled", "rejected"):
                                break
                    except Exception:
                        break
                    time.sleep(0.15)

            return {
                "order_id": str(oid) if oid is not None else None,
                "requested_qty": float(amt),
                "requested_price": float(px),
                "status": status,
                "raw": order,
            }

        # ---- main with retries for -1021
        try:
            return _create_once()
        except Exception as e:
            if _is_ts_error(e):
                logger.error(f"Limit order failed (ts) {symbol} {side}: {e} -> time sync & retry")
                try:
                    self._sync_time_best_effort(force=True)
                except Exception:
                    pass

                try:
                    return _create_once()
                except Exception as e2:
                    if _is_ts_error(e2):
                        logger.error(f"Limit order retry failed (ts) {symbol} {side}: {e2} -> time sync & retry2")
                        try:
                            self._sync_time_best_effort(force=True)
                        except Exception:
                            pass
                        try:
                            return _create_once()
                        except Exception as e3:
                            logger.error(f"Limit order final failed {symbol} {side}: {e3}")
                            return None
                    logger.error(f"Limit order retry failed {symbol} {side}: {e2}")
                    return None

            logger.error(f"Limit order failed {symbol} {side}: {e}")
            return None

    def cancel_order_safe(self, order_id: str, symbol: str) -> bool:
        """
        ✅ cancel_order wrapper (signed params + -1021 retry)
        """
        if not order_id or not symbol:
            return False

        def _is_ts_error(e: Exception) -> bool:
            s = str(e)
            return ("-1021" in s) or ("Timestamp for this request" in s)

        def _cancel_once():
            self.exchange.cancel_order(str(order_id), symbol, params=self._signed_params({}))
            return True

        try:
            return _cancel_once()
        except Exception as e:
            if _is_ts_error(e):
                try:
                    self._sync_time_best_effort(force=True)
                except Exception:
                    pass
                try:
                    return _cancel_once()
                except Exception:
                    return False
            return False

    def _cancel_conditional_order_safe(self, symbol: str, order_id: str, is_algo: bool = False) -> bool:
        """
        ✅ 일반 주문 / Algo 조건주문 공통 취소
        - Algo 주문은 /fapi/v1/algoOrder 로 취소
        """
        if not order_id or not symbol:
            return False

        def _is_ts_error(e: Exception) -> bool:
            s = str(e)
            return ("-1021" in s) or ("Timestamp for this request" in s)

        def _to_market_id(sym: str) -> str:
            try:
                self.exchange.load_markets()
            except Exception:
                pass
            try:
                return str((self.exchange.market(sym) or {}).get("id") or sym)
            except Exception:
                return str(sym).replace("/USDT:USDT", "USDT").replace("/", "")

        if not is_algo:
            return self.cancel_order_safe(str(order_id), symbol)

        def _cancel_algo_once():
            market_id = _to_market_id(symbol)
            req = self._signed_params({"symbol": market_id})
            try:
                req["algoId"] = int(order_id)
            except Exception:
                req["algoId"] = str(order_id)

            # 1) ccxt raw method 우선
            for mname in (
                "fapiPrivateDeleteAlgoOrder",
                "fapiPrivate_delete_algoorder",
                "fapiPrivate_delete_algoOrder",
                "fapiPrivateDeleteAlgoorder",
            ):
                fn = getattr(self.exchange, mname, None)
                if callable(fn):
                    resp = fn(req)
                    if isinstance(resp, dict):
                        code = str(resp.get("code", "") or "")
                        msg = str(resp.get("msg", "") or "").lower()
                        if code == "200" or "success" in msg or resp.get("algoId") is not None:
                            return True
                    return True

            # 2) raw method이 없으면 direct request fallback
            if hasattr(self.exchange, "request"):
                resp = self.exchange.request("algoOrder", "fapiPrivate", "DELETE", req)
                if isinstance(resp, dict):
                    code = str(resp.get("code", "") or "")
                    msg = str(resp.get("msg", "") or "").lower()
                    if code == "200" or "success" in msg or resp.get("algoId") is not None:
                        return True
                return True

            return False

        try:
            return _cancel_algo_once()
        except Exception as e:
            if _is_ts_error(e):
                try:
                    self._sync_time_best_effort(force=True)
                except Exception:
                    pass
                try:
                    return _cancel_algo_once()
                except Exception:
                    return False
            return False

    def ensure_position_sl_limit_mark(
        self,
        symbol: str,
        position_side: str,
        amount: float,
        stop_price: float,
        limit_price: float,
        prev_order_id: str = None,
        prev_is_algo: bool = False,
        trigger: str = "MARK_PRICE",
    ) -> dict:
        """
        ✅ 거래소 Position SL 미러링
        - Stop-Limit + Mark trigger
        - Binance USDⓈ-M 조건주문은 Algo endpoint(/fapi/v1/algoOrder)만 사용
        - 성공 시 {"order_id": str, "is_algo": bool}
        - 실패 시 {"order_id":"", "is_algo":False}
        """

        def _is_ts_error(e: Exception) -> bool:
            s = str(e)
            return ("-1021" in s) or ("Timestamp for this request" in s)

        def _to_market_id(sym: str) -> str:
            try:
                self.exchange.load_markets()
            except Exception:
                pass
            try:
                return str((self.exchange.market(sym) or {}).get("id") or sym)
            except Exception:
                return str(sym).replace("/USDT:USDT", "USDT").replace("/", "")

        def _extract_algo_id(resp) -> str:
            if not isinstance(resp, dict):
                return ""
            for key in ("algoId", "orderId"):
                if resp.get(key) is not None:
                    return str(resp[key])
            for box in ("result", "data"):
                sub = resp.get(box, {}) or {}
                if isinstance(sub, dict):
                    for key in ("algoId", "orderId"):
                        if sub.get(key) is not None:
                            return str(sub[key])
            return ""

        ps = str(position_side or "").upper().strip()
        if ps not in ("LONG", "SHORT"):
            return {"order_id": "", "is_algo": False}

        try:
            sp = float(self.price_to_precision(symbol, stop_price))
            lp = float(self.price_to_precision(symbol, limit_price))
            amt = float(self.amount_to_precision(symbol, amount))
        except Exception:
            return {"order_id": "", "is_algo": False}

        if (not np.isfinite(sp)) or (not np.isfinite(lp)) or (not np.isfinite(amt)):
            return {"order_id": "", "is_algo": False}
        if sp <= 0 or lp <= 0 or amt <= 0:
            return {"order_id": "", "is_algo": False}

        side_binance = "SELL" if ps == "LONG" else "BUY"

        # ✅ 기존 조건주문 선취소 (충돌 회피)
        if prev_order_id:
            try:
                self._cancel_conditional_order_safe(symbol, str(prev_order_id), bool(prev_is_algo))
            except Exception:
                pass

        def _create_algo_once() -> dict:
            market_id = _to_market_id(symbol)

            # Binance New Algo Order docs:
            # algoType=CONDITIONAL, type=STOP, triggerPrice, workingType, price, quantity
            req = self._signed_params({
                "algoType": "CONDITIONAL",
                "symbol": market_id,
                "side": side_binance,
                "type": "STOP",
                "quantity": float(amt),
                "price": float(lp),
                "triggerPrice": float(sp),
                "workingType": str(trigger or "MARK_PRICE"),
                "timeInForce": "GTC",
                "reduceOnly": "true",
                "newOrderRespType": "RESULT",
            })

            # 1) ccxt raw method 우선
            for mname in (
                "fapiPrivatePostAlgoOrder",
                "fapiPrivate_post_algoorder",
                "fapiPrivate_post_algoOrder",
                "fapiPrivatePostAlgoorder",
            ):
                fn = getattr(self.exchange, mname, None)
                if callable(fn):
                    resp = fn(req)
                    oid = _extract_algo_id(resp)
                    if oid:
                        return {"order_id": oid, "is_algo": True}

            # 2) raw method이 없거나 응답 파싱 실패면 direct request fallback
            if hasattr(self.exchange, "request"):
                resp = self.exchange.request("algoOrder", "fapiPrivate", "POST", req)
                oid = _extract_algo_id(resp)
                if oid:
                    return {"order_id": oid, "is_algo": True}

            return {"order_id": "", "is_algo": False}

        try:
            # ✅ 일반 STOP fallback 금지 (-4120 유발)
            out = _create_algo_once()
            if out.get("order_id"):
                return out

            logger.error(f"Position SL create failed {symbol}: algoOrder returned empty response")
            return {"order_id": "", "is_algo": False}

        except Exception as e:
            if _is_ts_error(e):
                try:
                    self._sync_time_best_effort(force=True)
                except Exception:
                    pass

                try:
                    out = _create_algo_once()
                    if out.get("order_id"):
                        return out
                    logger.error(f"Position SL create failed {symbol}: algoOrder returned empty response after retry")
                    return {"order_id": "", "is_algo": False}
                except Exception as e2:
                    logger.error(f"Position SL create retry failed {symbol}: {e2}")
                    return {"order_id": "", "is_algo": False}

            logger.error(f"Position SL create failed {symbol}: {e}")
            return {"order_id": "", "is_algo": False}


    def fetch_order_safe(self, order_id: str, symbol: str) -> dict:
        """
        ✅ fetch_order wrapper (signed params + -1021 retry)
        """
        if not order_id or not symbol:
            return {}

        def _is_ts_error(e: Exception) -> bool:
            s = str(e)
            return ("-1021" in s) or ("Timestamp for this request" in s)

        def _fetch_once():
            return self.exchange.fetch_order(str(order_id), symbol, params=self._signed_params({})) or {}

        try:
            return _fetch_once()
        except Exception as e:
            if _is_ts_error(e):
                try:
                    self._sync_time_best_effort(force=True)
                except Exception:
                    pass
                try:
                    return _fetch_once()
                except Exception:
                    return {}
            return {}

    def close_position(self, symbol, price=None, reason="EXIT", poll_timeout_sec=5.0):
        """
        시장가 청산.
        - reduceOnly 강제
        - 기존 거래소 SL 조건주문(best-effort) 선취소
        - 성공 여부(True/False) 반환
        - ⚠️ positions 수정 금지 (엔진이 상태 권위)
        """
        pos = (self.positions or {}).get(symbol)
        if not pos:
            return False

        def _is_ts_error(e: Exception) -> bool:
            s = str(e)
            return ("-1021" in s) or ("Timestamp for this request" in s)

        def _try_close_once():
            side_u = str(pos.get("side", "")).upper()
            side_ccxt = "sell" if side_u == "LONG" else "buy"

            amt = self.amount_to_precision(symbol, pos.get("amount", 0))
            try:
                amt = float(amt)
            except Exception:
                amt = 0.0
            if amt <= 0:
                return False

            # ✅ 거래소 상주 Position SL 선취소
            try:
                ex_id = str(pos.get("exchange_sl_order_id") or "")
                ex_algo = bool(pos.get("exchange_sl_is_algo", False))
                if ex_id:
                    self._cancel_conditional_order_safe(symbol, ex_id, ex_algo)
            except Exception:
                pass

            params_ro = self._signed_params({"reduceOnly": True})

            order = self.exchange.create_market_order(
                symbol,
                side_ccxt,
                amt,
                params=params_ro,
            )

            oid = None
            if isinstance(order, dict):
                oid = order.get("id")

            if oid:
                t0 = time.time()
                while time.time() - t0 < float(poll_timeout_sec):
                    try:
                        o2 = self.exchange.fetch_order(oid, symbol, params=self._signed_params({}))
                        status = str((o2 or {}).get("status", "")).lower()
                        if status == "closed":
                            break
                        time.sleep(0.2)
                    except Exception:
                        time.sleep(0.2)

            return True

        try:
            ok = _try_close_once()
            self._sync_balance()
            return bool(ok)

        except Exception as e:
            if _is_ts_error(e):
                logger.error(f"Close failed {symbol}: {e} -> time sync & retry")
                try:
                    self._sync_time_best_effort(force=True)
                except Exception:
                    pass

                try:
                    ok = _try_close_once()
                    self._sync_balance()
                    return bool(ok)
                except Exception as e2:
                    logger.error(f"Close retry failed {symbol}: {e2}")
                    return False

            logger.error(f"Close failed {symbol}: {e}")
            return False