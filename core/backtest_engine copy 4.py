import sys
import os
import json
import logging
import pandas as pd
import numpy as np
import math
import pandas_ta as ta

# 상위 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from strategy.position_monitor import PositionMonitor
from strategy.risk_control import RiskControl

# [설정] 백테스트 전용 상수
BASE_FEE = 0.0005          # 0.05% 수수료
SLIPPAGE_ATR_FACTOR = 0.05 # ATR 대비 슬리피지 비율

# 로깅 설정
LOG_FILE = os.path.join(root_dir, "backtest_history.csv")
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("PhalanxBacktest")

try:
    from strategy.titan_strategy import TitanStrategy
    from execution.virtual_executor import VirtualExecutor
except ImportError as e:
    logger.critical(f"❌ Module Import Error: {e}")
    sys.exit(1)


class BacktestEngine:
    """
    [Phalanx Core Module] Backtest Orchestrator
    Mode: Time-Sequential & Priority-Based (Deterministic)

    FIXES:
    - Equity 정의 통일(선물 마진 회계): equity = cash + Σ(margin + unrealized_pnl)
    - Event 로그에 Cash/Equity 명확히 기록하여 대시보드 튐 제거
    - ENTRY/EXIT 직후에도 전체 심볼 가격으로 MTM 평가(단일 심볼 update 제거)
    """

    def __init__(self, days=7):
        self.root_dir = root_dir
        self.config_path = os.path.join(root_dir, "config.json")
        self.cfg = self._load_config()
        self.raw_1m_map = {}    # sym -> 1m raw df
        self.data_1m_map = {}   # sym -> 1m df (필요시 정제)
        self.raw_daily_map = {} # sym -> 1d raw df (DAILY EMA 주입용)
        self._bar_td = self._infer_bar_timedelta(default_minutes=15)
        self.cooldowns = {}
        self.consecutive_losses = {}
        self.test_days = days

        # ✅ 동일 캔들 재진입 금지(최소 1캔들)용 바 길이
        self.bar_td = self._infer_bar_timedelta(default_minutes=15)

        # 1. 전략 및 실행기 초기화
        self.titan = TitanStrategy()
        self.executor = VirtualExecutor()

        # [Safety] 초기 자본금 강제 주입
        if not hasattr(self.executor, 'initial_balance'):
            self.executor.initial_balance = 2500.0
        if not hasattr(self.executor, 'cash'):
            self.executor.cash = float(self.executor.initial_balance)
        if not hasattr(self.executor, 'equity'):
            self.executor.equity = float(self.executor.initial_balance)
        if not hasattr(self.executor, 'positions'):
            self.executor.positions = {}
        if not hasattr(self.executor, 'history'):
            self.executor.history = []
        if not hasattr(self.executor, 'equity_curve'):
            self.executor.equity_curve = []

        self.risk_ctrl = RiskControl(self.executor, self.cfg)
        self.monitor = PositionMonitor()

        # RAW / PROCESSED
        self.raw_data_map = {}   # {sym: df_raw}
        self.data_map = {}       # {sym: df_with_indicators}

        self.symbols = []  # 외부 주입 심볼 저장소

        # 마지막 MTM 평가용 가격 스냅샷
        self.last_prices = {}  # {sym: close}

        # Config 강제 주입
        strat_settings = self.cfg.get('strategy_settings', {})
        if 'blacklist' in strat_settings:
            self.titan.blacklist = set(strat_settings['blacklist'])
            print(f"🚫 [System] Configured Blacklist: {self.titan.blacklist}")

        # 로그 파일 초기화 (Cash/Equity 분리 저장)
        self.log_file = os.path.join(root_dir, "backtest_history.csv")
        with open(self.log_file, 'w', encoding="utf-8") as f:
            f.write("Datetime,Symbol,Side,Type,Price,Amount,PnL,Cash,Equity,Reason\n")

        # MTM Equity Curve CSV 경로 (대시보드용)
        self.equity_curve_file = os.path.join(root_dir, "backtest_equity_curve.csv")



    def _load_config(self):
        if not os.path.exists(self.config_path):
            return {}
        with open(self.config_path, 'r', encoding="utf-8") as f:
            return json.load(f)
        
        # --- [UF] enable flag ---
    def _is_uf_enabled(self) -> bool:
        try:
            return bool(self.cfg.get("system_settings", {}).get("use_universe_filter", False))
        except Exception:
            return False

    # --- [UF] read universe file (JSON) ---
    def _get_universe_from_json(self, asof_ts=None) -> list:
        """
        UF는 엔진 밖에서 돌고, 엔진은 결과 JSON만 읽는다.

        지원 포맷:
        1) {"symbols":[...]}  (구버전/다른 포맷)
        2) {"universe":[...]} (현재 UF 출력)
        3) {"universe":{"symbols":[...]}} 같은 변형도 방어

        경로 규칙:
        - config: system_settings.universe_selected_path
        - 상대경로면 root_dir 기준으로 결합
        - 기본값은 <root_dir>/universe_selected.json
        """
        # 1) path 읽기
        try:
            path = (self.cfg.get("system_settings", {}) or {}).get("universe_selected_path", None)
        except Exception:
            path = None

        if not path:
            path = os.path.join(self.root_dir, "universe_selected.json")
        else:
            # 2) 상대경로면 root_dir 기준으로
            try:
                if not os.path.isabs(path):
                    path = os.path.join(self.root_dir, path)
            except Exception:
                pass

        # 3) 파일 없으면 빈 리스트(=UF 실패 폴백 트리거)
        if not os.path.exists(path):
            return []

        # 4) JSON 로드
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f) or {}
        except Exception:
            return []

        # 5) 심볼 리스트 추출 (symbols / universe 둘 다 지원)
        syms = None

        # (A) {"symbols":[...]}
        if isinstance(obj, dict) and isinstance(obj.get("symbols", None), list):
            syms = obj.get("symbols")

        # (B) {"universe":[...]}  ← UF 현재 출력
        elif isinstance(obj, dict) and isinstance(obj.get("universe", None), list):
            syms = obj.get("universe")

        # (C) {"universe":{"symbols":[...]}} 같은 변형 방어
        elif isinstance(obj, dict) and isinstance(obj.get("universe", None), dict):
            u = obj.get("universe", {})
            if isinstance(u.get("symbols", None), list):
                syms = u.get("symbols")

        if not isinstance(syms, list):
            return []

        # 6) 문자열 정리 + 중복 제거(순서 보존)
        out = []
        seen = set()
        for s in syms:
            try:
                ss = str(s).strip()
                if not ss:
                    continue
                if ss in seen:
                    continue
                seen.add(ss)
                out.append(ss)
            except Exception:
                continue

        return out


    def _infer_bar_timedelta(self, default_minutes=15):
        tf = None
        try:
            tf = (self.cfg.get("strategy_settings", {}) or {}).get("timeframe", None)
        except Exception:
            tf = None

        if not tf:
            return pd.Timedelta(minutes=default_minutes)

        try:
            s = str(tf).strip().lower()
            if s.endswith("m"):
                return pd.Timedelta(minutes=int(s[:-1]))
            if s.endswith("h"):
                return pd.Timedelta(hours=int(s[:-1]))
            if s.endswith("d"):
                return pd.Timedelta(days=int(s[:-1]))
        except Exception:
            pass

        return pd.Timedelta(minutes=default_minutes)


        # -----------------------------------------------------

    def _entry_alpha_atr(self) -> float:
        """
        ✅ LIVE와 동일: system_settings.entry_alpha_atr
        - alpha >= 0
        """
        try:
            ss = (self.cfg.get("system_settings", {}) or {})
            a = float(ss.get("entry_alpha_atr", 0.0) or 0.0)
            if (not math.isfinite(a)) or a < 0:
                a = 0.0
            return float(a)
        except Exception:
            return 0.0



    # [ADD] Titan v9 계약: 최소 길이
    # -----------------------------------------------------
    def _t9_min_len(self) -> int:
        """
        ✅ Titan v9 엔진 계약(고정):
        analyze 입력 df 길이 >= max(200, flip_window_bars + 2)
        """
        try:
            p = getattr(self.titan, "params", None)
            p = p if isinstance(p, dict) else {}
            N = int(p.get("flip_window_bars", 5) or 5)
        except Exception:
            N = 5
        return int(max(200, N + 2))

    # -----------------------------------------------------
    # [ADD] Titan v9 계약: asof(Time Authority) 결합 강제 슬라이스
    # -----------------------------------------------------
    def _slice_for_t9(self, df: pd.DataFrame, asof) -> pd.DataFrame:
        """
        ✅ Titan v9 봉인형:
        df_for_sig = df.loc[:asof].tail(_t9_min_len())

        - asof(=current_time)가 df.index에 반드시 존재해야 함
        - 폴백(df.tail 등) 금지. 위반 시 None 반환(=스킵)
        """
        if df is None or (not isinstance(df, pd.DataFrame)) or df.empty or asof is None:
            return None
        if asof not in df.index:
            return None

        want = int(self._t9_min_len())
        try:
            return df.loc[:asof].tail(want).copy()
        except Exception:
            return None


    # =========================================================
    # ✅ Equity 계산 공식 강제 통일 (선물 마진 회계)
    # =========================================================
    def _force_mark_to_market_equity(self, prices: dict) -> float:
        """
        강제 MTM:
        equity = cash + Σ(position.margin + unrealized_pnl)
        - margin은 '묶인 내 돈'이라 equity에 포함
        - unrealized는 side에 따라 계산
        """
        cash = float(getattr(self.executor, "cash", 0.0))
        positions = getattr(self.executor, "positions", {}) or {}

        total = cash
        for sym, pos in positions.items():
            try:
                if sym not in prices:
                    continue
                px = float(prices[sym])
                entry = float(pos.get("entry_price", 0.0))
                amt = float(pos.get("amount", 0.0))
                side = str(pos.get("side", "LONG")).upper()
                margin = float(pos.get("margin", 0.0))

                if side == "LONG":
                    upnl = (px - entry) * amt
                else:
                    upnl = (entry - px) * amt

                total += margin + upnl
            except Exception:
                continue

        self.executor.equity = float(total)
        return float(total)

    def _sync_equity(self, prices: dict):
        """
        - VirtualExecutor.update_equity()가 있으면 호출해도 되지만,
          최종 값은 엔진에서 강제 공식으로 덮어쓴다.
        """
        try:
            if hasattr(self.executor, "update_equity"):
                self.executor.update_equity(prices)
        except Exception:
            pass
        self._force_mark_to_market_equity(prices)

    def _sweep_pending_entry_orders(self, current_time, current_rows: dict, current_prices: dict):
        """
        ✅ LIVE와 동일한 정책:
        - t_close에서 리밋 오픈오더 1개 제출
        - 다음 봉(=현재 루프 1회) 동안 high/low로 체결 판정
        - 다음 봉 종료 시점(=다음 루프 진입 시점)에 미체결이면 취소
        """
        peo = getattr(self, "pending_entry_orders", None)
        if not isinstance(peo, dict) or not peo:
            return 0

        removed = 0
        step_i = int(getattr(self, "_step_i", 0))

        for sym in list(peo.keys()):
            od = peo.get(sym) or {}

            # 이미 포지션이면 pending 제거
            if sym in (self.executor.positions or {}):
                peo.pop(sym, None)
                removed += 1
                continue

            # created / expire (TTL)
            try:
                created_i = int(od.get("created_i", step_i))
            except Exception:
                created_i = step_i

            # ✅ expire_i 의미: "다음 봉 1개" 동안 체결 판정 후, 그 다음 봉 시작에 만료
            # - 과거 오프바이원 값(expire_i<=created_i+1)은 created_i+2로 보정
            expire_i_raw = od.get("expire_i", None)
            try:
                expire_i = int(expire_i_raw) if expire_i_raw is not None else (created_i + 2)
            except Exception:
                expire_i = created_i + 2
            if expire_i <= (created_i + 1):
                expire_i = created_i + 2

            # 만기 체크 (다음 봉이 끝난 뒤, 그 다음 봉 시작에 취소)
            if step_i >= expire_i:
                oid = od.get("order_id")
                if oid:
                    if hasattr(self.executor, "cancel_order_safe"):
                        self.executor.cancel_order_safe(str(oid), sym)

                # 로그
                try:
                    self._log_csv(
                        current_time, sym, str(od.get("signal_side", "")),
                        "ENTRY_SKIP",
                        float(od.get("limit_price", od.get("anchor_close", 0.0)) or 0.0),
                        float(od.get("amount", 0.0) or 0.0),
                        0.0,
                        f"UNFILLED_CANCELED_NEXT_CANDLE | anchor={od.get('anchor_close')} limit={od.get('limit_price')} alpha={od.get('alpha')} oid={oid}"
                    )
                except Exception:
                    pass

                peo.pop(sym, None)
                removed += 1
                continue

            # ✅ fill 판정은 "오직 다음 봉(=created_i+1)"에서만 수행 (정합/결정론)
            if step_i != (created_i + 1):
                continue

            # 이번 봉에서 fill 판정 (현재봉 high/low)
            row = current_rows.get(sym)
            if row is None:
                continue

            sideU = str(od.get("signal_side", "")).upper()
            lp = od.get("limit_price", None)
            if lp is None:
                continue
            try:
                lp = float(lp)
            except Exception:
                continue
            if lp <= 0:
                continue

            try:
                hi = float(row.get("high"))
                lo = float(row.get("low"))
            except Exception:
                continue

            filled = False
            if sideU == "LONG":
                filled = (lo <= lp)
            elif sideU == "SHORT":
                filled = (hi >= lp)
            else:
                filled = False

            if not filled:
                continue

            # ✅ 체결가: limit_price ± slippage(보수적으로 불리)
            try:
                atr = float(row.get("atr", float(row.get("close", 0.0)) * 0.01))
                if (not math.isfinite(atr)) or atr <= 0:
                    atr = float(row.get("close", 0.0)) * 0.01
            except Exception:
                atr = float(row.get("close", 0.0)) * 0.01

            slippage = atr * SLIPPAGE_ATR_FACTOR
            fill_price = lp + slippage if sideU == "LONG" else lp - slippage

            amount = float(od.get("amount", 0.0) or 0.0)
            if amount <= 0:
                peo.pop(sym, None)
                removed += 1
                continue

            # 포지션 생성 (기존 _process_entry의 체결 로직을 이쪽으로 이동)
            notional_value = amount * fill_price
            leverage = self.cfg.get('risk_settings', {}).get('leverage', 1)
            margin_required = notional_value / leverage
            fee = notional_value * BASE_FEE

            if self.executor.cash < margin_required + fee:
                # 돈 부족이면 pending 제거(=실전에서 reject와 동일)
                peo.pop(sym, None)
                removed += 1
                continue

            self.executor.cash -= (margin_required + fee)

            self.executor.positions[sym] = {
                "side": sideU,
                "amount": float(amount),
                "entry_price": float(fill_price),
                "leverage": float(leverage),
                "margin": float(margin_required),
                "sl": float(od.get("sl")) if od.get("sl") is not None else None,
                "next_sl": None,
                "trail_sl": None,
                "entry_time": od.get("signal_time", current_time),
                "entry_atr": float(od.get("entry_atr")) if od.get("entry_atr") is not None else None,
            }

            prices2 = dict(current_prices)
            prices2[sym] = float(fill_price)
            self._sync_equity(prices2)

            try:
                self._log_csv(
                    od.get("signal_time", current_time),
                    sym,
                    sideU,
                    "ENTRY",
                    float(fill_price),
                    float(amount),
                    0.0,
                    f"LIMIT_FILLED | anchor={od.get('anchor_close')} limit={lp} alpha={od.get('alpha')} oid={od.get('order_id')}"
                )
            except Exception:
                pass

            peo.pop(sym, None)
            removed += 1

        self.pending_entry_orders = peo
        return removed
    

    def _fetch_ohlcv_range(self, sym: str, timeframe: str, since_ms: int, until_ms: int, limit: int = 1000):
        """
        거래소에서 OHLCV를 구간 단위로 페이지네이션 다운로드.
        - pickle/cache 저장 금지 (메모리 적재만)
        - executor가 exchange(fetch_ohlcv)를 들고 있다고 가정(대부분 ccxt 래퍼)
        """
        out = []
        fetch = None

        # 1) executor가 직접 fetch_ohlcv 제공하는 경우
        if hasattr(self.executor, "fetch_ohlcv"):
            fetch = self.executor.fetch_ohlcv
        # 2) executor.exchange.fetch_ohlcv(ccxt) 제공하는 경우
        elif hasattr(self.executor, "exchange") and hasattr(self.executor.exchange, "fetch_ohlcv"):
            fetch = self.executor.exchange.fetch_ohlcv

        if fetch is None:
            logger.error("❌ Intrabar requires executor.fetch_ohlcv or executor.exchange.fetch_ohlcv")
            return out

        cur = int(since_ms)
        guard = 0
        while cur < int(until_ms) and guard < 2000:
            guard += 1
            try:
                batch = fetch(sym, timeframe=timeframe, since=cur, limit=limit) or []
            except TypeError:
                # some wrappers use positional args: fetch(sym, timeframe, since, limit)
                batch = fetch(sym, timeframe, cur, limit) or []

            if not batch:
                break

            out.extend(batch)

            last_ts = int(batch[-1][0])
            # 다음 호출 since는 마지막+1ms로 진행 (무한루프 방지)
            nxt = last_ts + 1
            if nxt <= cur:
                break
            cur = nxt

            # until_ms 넘어가면 stop
            if last_ts >= int(until_ms):
                break

        # until_ms 이전만 필터링
        out = [r for r in out if int(r[0]) < int(until_ms)]
        return out

    def _prepare_daily_context(self, symbols):
        """
        백테 평가구간(test_days)은 유지하되,
        DAILY EMA 계산용 1d OHLCV를 별도로 확보한다.

        규칙:
        - 최소 40일봉 보장
        - 전략 최소 요구(daily_ema + 5)도 함께 반영
        - 각 심볼의 15m raw 끝시점 기준으로 과거 1d 구간 fetch
        """
        self.raw_daily_map = {}

        if not symbols:
            logger.warning("[DailyContext] symbols empty.")
            return self.raw_daily_map

        try:
            p = getattr(self.titan, "params", None)
            p = p if isinstance(p, dict) else {}
        except Exception:
            p = {}

        try:
            daily_len = int(p.get("daily_ema", 25) or 25)
        except Exception:
            daily_len = 25

        need_days = int(max(40, daily_len + 5))
        loaded = 0

        stats = {
            "ok": 0,
            "skip_no_15m": 0,
            "skip_fetch_empty": 0,
            "skip_clean_empty": 0,
            "fail": 0,
        }

        for sym in symbols:
            try:
                df15 = self.raw_data_map.get(sym)
                rows_15m = int(len(df15)) if isinstance(df15, pd.DataFrame) else 0

                if df15 is None or df15.empty:
                    stats["skip_no_15m"] += 1
                    self._log_daily_context_status(
                        sym=sym,
                        status="SKIP_NO_15M",
                        rows_15m=rows_15m,
                        rows_1d=0,
                        need_days=need_days,
                        reason="raw_15m_empty",
                    )
                    continue

                end_dt = pd.Timestamp(df15.index.max()).normalize() + pd.Timedelta(days=1)
                since_dt = end_dt - pd.Timedelta(days=int(need_days + 5))

                since_ms = int(pd.Timestamp(since_dt).timestamp() * 1000)
                until_ms = int(pd.Timestamp(end_dt).timestamp() * 1000)

                rows = self._fetch_ohlcv_range(sym, "1d", since_ms, until_ms, limit=1000)
                if not rows:
                    stats["skip_fetch_empty"] += 1
                    self._log_daily_context_status(
                        sym=sym,
                        status="SKIP_FETCH_EMPTY",
                        rows_15m=rows_15m,
                        rows_1d=0,
                        need_days=need_days,
                        start_1d=since_dt,
                        end_1d=end_dt,
                        reason="fetch_ohlcv_range_empty",
                    )
                    continue

                dfd = pd.DataFrame(
                    rows,
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )

                dfd["timestamp"] = pd.to_datetime(
                    dfd["timestamp"], unit="ms", utc=True, errors="coerce"
                ).dt.tz_convert(None)
                dfd = dfd.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

                for c in ["open", "high", "low", "close", "volume"]:
                    dfd[c] = pd.to_numeric(dfd[c], errors="coerce")

                dfd = dfd.dropna(subset=["open", "high", "low", "close"])
                if dfd.empty:
                    stats["skip_clean_empty"] += 1
                    self._log_daily_context_status(
                        sym=sym,
                        status="SKIP_CLEAN_EMPTY",
                        rows_15m=rows_15m,
                        rows_1d=0,
                        need_days=need_days,
                        start_1d=since_dt,
                        end_1d=end_dt,
                        reason="daily_df_empty_after_clean",
                    )
                    continue

                self.raw_daily_map[sym] = dfd
                loaded += 1
                stats["ok"] += 1

                self._log_daily_context_status(
                    sym=sym,
                    status="OK",
                    rows_15m=rows_15m,
                    rows_1d=len(dfd),
                    need_days=need_days,
                    start_1d=dfd.index.min(),
                    end_1d=dfd.index.max(),
                    reason="loaded",
                )

            except Exception as e:
                stats["fail"] += 1
                logger.warning(f"[DailyContext] {sym} load failed: {e}")
                self._log_daily_context_status(
                    sym=sym,
                    status="FAIL",
                    rows_15m=0,
                    rows_1d=0,
                    need_days=need_days,
                    reason=str(e),
                )

        logger.info(
            f"✅ Daily Context Ready: {loaded}/{len(symbols)} symbols | "
            f"need_days>={need_days} stats={stats}"
        )
        return self.raw_daily_map


    def _log_daily_context_status(
        self,
        sym: str,
        status: str,
        rows_15m: int,
        rows_1d: int,
        need_days: int,
        start_1d=None,
        end_1d=None,
        reason: str = "",
    ):
        try:
            logger.info(
                f"🗓️ DAILY_CTX | {sym} "
                f"status={status} rows15m={int(rows_15m)} rows1d={int(rows_1d)} "
                f"need_days>={int(need_days)} range1d={start_1d}~{end_1d} reason={reason}"
            )
        except Exception:
            pass


    def _inject_daily_ema_from_daily_map(self, sym: str, ind: pd.DataFrame) -> pd.DataFrame:
        """
        엔진이 별도로 받은 1d OHLCV로 DAILY EMA를 계산해 15m indicator df에 주입한다.
        - lookahead 방지: 일봉 EMA를 1일 shift 후 15m index에 ffill
        - 충분한 일봉이 없으면 ema_daily_ok=0 유지
        """
        if ind is None or not isinstance(ind, pd.DataFrame) or ind.empty:
            return ind

        out = ind.copy()
        rows_15m = int(len(out))

        try:
            p = getattr(self.titan, "params", None)
            p = p if isinstance(p, dict) else {}
        except Exception:
            p = {}

        try:
            daily_len = int(p.get("daily_ema", 25) or 25)
        except Exception:
            daily_len = 25

        need_days = int(max(40, daily_len + 5))

        dfd = getattr(self, "raw_daily_map", {}).get(sym)
        if dfd is None or dfd.empty:
            out["ema_daily"] = 0.0
            out["ema_daily_ok"] = 0
            self._log_daily_injection_status(
                sym=sym,
                status="SKIP_NO_DAILY_CTX",
                rows_15m=rows_15m,
                rows_1d=0,
                need_days=need_days,
                reason="raw_daily_map_missing",
            )
            return out

        dfd = dfd.copy().sort_index()
        dfd = dfd.dropna(subset=["close"])
        rows_1d = int(len(dfd))

        if len(dfd) < need_days:
            out["ema_daily"] = 0.0
            out["ema_daily_ok"] = 0
            self._log_daily_injection_status(
                sym=sym,
                status="SKIP_SHORT_DAILY",
                rows_15m=rows_15m,
                rows_1d=rows_1d,
                need_days=need_days,
                reason="insufficient_daily_rows",
            )
            return out

        try:
            dfd["ema_daily"] = ta.ema(dfd["close"], length=daily_len)
            dfd["ema_daily"] = dfd["ema_daily"].shift(1)  # 확정된 전일 값만 사용

            ema_mapped = dfd["ema_daily"].reindex(out.index, method="ffill")
            mapped_series = pd.Series(ema_mapped, index=out.index).astype(float)

            mapped_mask = mapped_series.notna()
            mapped_non_na = int(mapped_mask.sum())
            first_valid_15m = mapped_series.index[mapped_mask.argmax()] if mapped_non_na > 0 else None
            last_ema = float(mapped_series[mapped_mask].iloc[-1]) if mapped_non_na > 0 else None

            out["ema_daily"] = mapped_series.fillna(0.0)
            out["ema_daily_ok"] = 1 if mapped_non_na > 0 else 0

            self._log_daily_injection_status(
                sym=sym,
                status="OK" if mapped_non_na > 0 else "FAIL_NO_MAPPED_VALUES",
                rows_15m=rows_15m,
                rows_1d=rows_1d,
                need_days=need_days,
                mapped_non_na=mapped_non_na,
                first_valid_15m=first_valid_15m,
                last_ema=last_ema,
                reason="engine_daily_injected" if mapped_non_na > 0 else "mapped_non_na_zero",
            )
            return out

        except Exception as e:
            logger.warning(f"[DailyContext] {sym} ema inject failed: {e}")
            out["ema_daily"] = 0.0
            out["ema_daily_ok"] = 0
            self._log_daily_injection_status(
                sym=sym,
                status="FAIL_EXCEPTION",
                rows_15m=rows_15m,
                rows_1d=rows_1d,
                need_days=need_days,
                reason=str(e),
            )
            return out



    def _log_daily_injection_status(
        self,
        sym: str,
        status: str,
        rows_15m: int,
        rows_1d: int,
        need_days: int,
        mapped_non_na: int = 0,
        first_valid_15m=None,
        last_ema=None,
        reason: str = "",
    ):
        try:
            logger.info(
                f"🧪 DAILY_INJECT | {sym} "
                f"status={status} rows15m={int(rows_15m)} rows1d={int(rows_1d)} "
                f"need_days>={int(need_days)} mapped_non_na={int(mapped_non_na)} "
                f"first_valid_15m={first_valid_15m} last_ema={last_ema} reason={reason}"
            )
        except Exception:
            pass


    def _load_intrabar_1m(self, targets):
        """
        15m 백테 구간에 대응하는 1m OHLCV를 심볼별로 다운로드해서 메모리에 적재.
        """
        if not self.raw_data_map:
            return

    


        # 15m 데이터에서 전체 기간 산출
        all_starts = []
        all_ends = []
        for sym in targets:
            df15 = self.raw_data_map.get(sym)
            if df15 is None or df15.empty:
                continue
            all_starts.append(pd.to_datetime(df15.index.min()))
            all_ends.append(pd.to_datetime(df15.index.max()))

        if not all_starts or not all_ends:
            return

        start_dt = min(all_starts)
        end_dt = max(all_ends) + self._bar_td

        # backtest는 UTC-naive를 UTC로 본다는 전제 유지
        since_ms = int(pd.Timestamp(start_dt).timestamp() * 1000)
        until_ms = int(pd.Timestamp(end_dt).timestamp() * 1000)

        logger.info(f"📥 [Intrabar] Downloading 1m OHLCV | range={start_dt} ~ {end_dt} | symbols={len(targets)}")

        out_map = {}
        for sym in targets:
            rows = self._fetch_ohlcv_range(sym, "1m", since_ms, until_ms, limit=1000)
            if not rows:
                continue

            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce").dt.tz_convert(None)
            df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

            # 표준 컬럼 강제 float
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["open", "high", "low", "close"])

            out_map[sym] = df

        self.raw_1m_map = out_map
        self.data_1m_map = out_map  # 현재는 지표 불필요. 그대로 사용.
        logger.info(f"✅ [Intrabar] 1m ready: {len(self.data_1m_map)} symbols")

    def _inject_intrabar(self, sym: str, curr_row, current_time: pd.Timestamp):
        """
        15m current_time 구간 [t, t+bar) 의 1m 데이터로 high/low/close를 보강.
        라이브의 1m boost 효과를 백테에 반영.
        """
        df1 = self.data_1m_map.get(sym)
        if df1 is None or df1.empty:
            return curr_row

        t0 = pd.to_datetime(current_time)
        t1 = t0 + self._bar_td

        seg = df1.loc[(df1.index >= t0) & (df1.index < t1)]
        if seg is None or seg.empty:
            return curr_row

        # curr_row는 pandas Series일 가능성이 높으니 copy 후 overwrite
        try:
            r = curr_row.copy()
        except Exception:
            return curr_row

        r["high"] = float(seg["high"].max())
        r["low"] = float(seg["low"].min())
        r["close"] = float(seg["close"].iloc[-1])

        # open은 15m open 유지(백테 구조 유지). 원하면 1m 첫 open으로 바꿀 수 있으나 여기선 불변.
        return r


    # =========================================================
    # 1. Data Preparation Layer
    # =========================================================
    def prepare_data(self, symbols=None):
            """
            - ✅ universe.json 우선 사용 (root_dir/universe.json)
            - ✅ 파일 없거나 비정상이면 executor.get_top_targets() 폴백
            - ✅ raw_data_map 캐시가 있어도 무조건 다시 다운로드(재로드)
            - ✅ intrabar(1m)도 같은 symbols로 즉시 로드 (pickle 금지)
            - ✅ DAILY EMA용 1d OHLCV는 별도로 확보 (평가구간과 분리)
            """
            logger.info("📥 [Data Loader] Fetching Historical Data... (FORCE REFRESH)")

            # ---------------------------------------------------------
            # 1) 타겟 심볼 결정 (외부 주입 > universe.json > 폴백)
            # ---------------------------------------------------------
            if symbols:
                targets = symbols
            else:
                targets = []

                # (A) universe.json 로드
                uni_path = os.path.join(self.root_dir, "universe.json")
                try:
                    if os.path.exists(uni_path):
                        with open(uni_path, "r", encoding="utf-8") as f:
                            obj = json.load(f) or {}

                        # 지원 포맷:
                        # 1) {"universe":[...]}  (너가 만든 포맷)
                        # 2) {"symbols":[...]}   (호환)
                        if isinstance(obj, dict) and isinstance(obj.get("universe", None), list):
                            targets = obj.get("universe", []) or []
                        elif isinstance(obj, dict) and isinstance(obj.get("symbols", None), list):
                            targets = obj.get("symbols", []) or []
                except Exception:
                    targets = []

                # (B) 폴백: 거래소/익스큐터 top targets
                if not targets:
                    targets = self.executor.get_top_targets()

            # ---------------------------------------------------------
            # 2) 블랙리스트 제거 (기존 유지)
            # ---------------------------------------------------------
            filtered_targets = []
            for sym in targets:
                clean_sym = sym.split(":")[0]
                if clean_sym in self.titan.blacklist or sym in self.titan.blacklist:
                    continue
                filtered_targets.append(sym)
            targets = filtered_targets

            # ---------------------------------------------------------
            # 3) 강제 재다운로드 (엔진 days 권위 반영)
            # ---------------------------------------------------------
            try:
                raw_data_map = self.executor.prepare_data(targets, days=int(self.test_days))
            except TypeError:
                raw_data_map = self.executor.prepare_data(targets)

            if not raw_data_map:
                logger.error("❌ No Data Loaded.")
                return {}

            sorted_symbols = sorted(list(raw_data_map.keys()))

            # ✅ 1) raw_data_map 먼저 확정
            self.raw_data_map = {sym: raw_data_map[sym] for sym in sorted_symbols}

            # ✅ 2) symbols는 "확정된 raw_data_map" 기준으로 갱신
            self.symbols = list(self.raw_data_map.keys())

            # ✅ 3) DAILY EMA용 1d 컨텍스트 별도 로드
            self._prepare_daily_context(self.symbols)

            # ✅ 4) intrabar도 같은 symbols로 로드
            self._load_intrabar_1m(self.symbols)

            logger.info(f"✅ Raw Data Ready: {len(self.raw_data_map)} symbols loaded.")
            return self.raw_data_map


    def rebuild_indicators(self):
        """
        - dropna()는 '필수 컬럼 subset' 기준으로만 적용한다.
        - 워밍업(warmup)은 각 필수 컬럼의 첫 유효시점 중 '가장 늦은 시점'을 사용한다.
        - 심볼별 warmup으로 슬라이스
        - DAILY EMA는 엔진이 별도 1d 데이터로 계산해 주입한다.
        """
        self.data_map = {}

        self.required_cols = [
            "open", "high", "low", "close", "volume",
            "atr", "vol_ma", "ema_intra", "rsi", "adx", "st_val", "st_dir"
        ]

        temp_map = {}
        warmup_map = {}

        indicator_stats = {
            "ok": 0,
            "fail_calc": 0,
            "missing_required": 0,
            "all_nan_required": 0,
            "bad_required_cols": 0,
            "daily_ready": 0,
            "daily_not_ready": 0,
            "slice_empty": 0,
            "drop_all_nan": 0,
            "slice_fail": 0,
        }

        for sym, df in self.raw_data_map.items():
            try:
                ind = self.titan.calculate_indicators(sym, df.copy())

                # ✅ 엔진 주도 DAILY EMA 주입 (전략 내부 15m-resample 결과를 override)
                ind = self._inject_daily_ema_from_daily_map(sym, ind)

                if "ema_daily" not in ind.columns or "ema_daily_ok" not in ind.columns:
                    logger.warning(f"[Indicator] {sym} daily columns missing after injection.")
                    indicator_stats["daily_not_ready"] += 1
                else:
                    try:
                        daily_ok_last = int(ind["ema_daily_ok"].iloc[-1])
                    except Exception:
                        daily_ok_last = 0

                    if daily_ok_last == 1:
                        indicator_stats["daily_ready"] += 1
                    else:
                        indicator_stats["daily_not_ready"] += 1
                        logger.warning(
                            f"[Indicator] {sym} daily not ready after injection | "
                            f"rows={len(ind)} last_ema_daily_ok={daily_ok_last}"
                        )

                missing = [c for c in self.required_cols if c not in ind.columns]
                if missing:
                    indicator_stats["missing_required"] += 1
                    logger.warning(f"[Indicator] {sym} missing required cols: {missing}")
                    continue

                if ind[self.required_cols].isna().all().all():
                    indicator_stats["all_nan_required"] += 1
                    logger.warning(f"[Indicator] {sym} required columns all-NaN (pre-slice).")
                    continue

                bad_cols = [c for c in self.required_cols if ind[c].isna().all()]
                if bad_cols:
                    indicator_stats["bad_required_cols"] += 1
                    logger.warning(f"[Indicator] {sym} has all-NaN required cols (pre-slice): {bad_cols}")
                    continue

                col_warmups = []
                for c in self.required_cols:
                    s = ind[c]
                    first_valid = s.first_valid_index()
                    if first_valid is None:
                        col_warmups.append(len(ind))
                    else:
                        try:
                            col_warmups.append(ind.index.get_loc(first_valid))
                        except Exception:
                            col_warmups.append(0)

                sym_warmup = int(max(col_warmups)) if col_warmups else 0
                temp_map[sym] = ind
                warmup_map[sym] = sym_warmup

            except Exception as e:
                indicator_stats["fail_calc"] += 1
                logger.warning(f"[Indicator] {sym} indicator failed: {e}")

        for sym, ind in temp_map.items():
            try:
                sym_warmup = int(warmup_map.get(sym, 0))
                sliced = ind.iloc[sym_warmup:].copy()

                if len(sliced) == 0:
                    indicator_stats["slice_empty"] += 1
                    logger.warning(f"[Indicator] {sym} empty after slice | warmup={sym_warmup}")
                    continue

                sliced = sliced.dropna(subset=self.required_cols)
                if len(sliced) == 0:
                    indicator_stats["drop_all_nan"] += 1
                    logger.warning(f"[Indicator] {sym} all NaN after drop (required subset)")
                    continue

                self.data_map[sym] = sliced
                indicator_stats["ok"] += 1

                try:
                    daily_ok_last = int(sliced["ema_daily_ok"].iloc[-1]) if "ema_daily_ok" in sliced.columns else -1
                    ema_daily_last = float(sliced["ema_daily"].iloc[-1]) if "ema_daily" in sliced.columns else None
                except Exception:
                    daily_ok_last = -1
                    ema_daily_last = None

                logger.info(
                    f"🧩 INDICATOR_READY | {sym} "
                    f"rows_in={len(ind)} warmup={sym_warmup} rows_out={len(sliced)} "
                    f"daily_ok_last={daily_ok_last} ema_daily_last={ema_daily_last}"
                )

            except Exception as e:
                indicator_stats["slice_fail"] += 1
                logger.warning(f"[Indicator] {sym} slice failed: {e}")

        logger.info(
            f"Indicators Ready: {len(self.data_map)} symbols processed. "
            f"stats={indicator_stats}"
        )


    def _log_csv(self, dt, sym, side, type_note, price, amt, pnl, reason):
        cash = float(getattr(self.executor, "cash", 0.0))
        eq = float(getattr(self.executor, "equity", cash))
        line = f"{dt},{sym},{side},{type_note},{price},{amt},{pnl:.4f},{cash:.2f},{eq:.2f},{reason}\n"
        with open(self.log_file, 'a', encoding="utf-8") as f:
            f.write(line)
    def _get_sl_apply_mode(self) -> str:
        """
        LIVE와 동일하게 config에서 sl 적용 모드 읽기
        - "next": next_sl 저장 후 다음 캔들에 sl 승계
        - "same": 같은 캔들(확정봉)에서 sl 즉시 반영 + 히트판정도 new_sl 기준(모니터에서 처리)
        """
        try:
            mode = str(self.cfg.get("system_settings", {}).get("sl_apply_mode", "next")).strip().lower()
        except Exception:
            mode = "next"
        return mode if mode in ("next", "same") else "next"


    def _get_sl_strategy(self) -> str:
        """
        PositionMonitor의 sl_strategy를 config에서 읽는다.
        허용: supertrend | atr_trail | profit_lock | hybrid | armor
        """
        try:
            strat = str(self.cfg.get("system_settings", {}).get("sl_strategy", "supertrend")).strip().lower()
        except Exception:
            strat = "supertrend"

        allowed = {"supertrend", "atr_trail", "profit_lock", "hybrid", "armor"}
        return strat if strat in allowed else "supertrend"

    def _get_sl_params(self) -> dict:
        """
        PositionMonitor의 sl_params를 config에서 읽는다.
        예)
        system_settings:
          sl_params:
            atr_mult: 3.0
            trigger_atr: 2.0
            lock_atr: 0.5
        """
        try:
            p = self.cfg.get("system_settings", {}).get("sl_params", {}) or {}
            return p if isinstance(p, dict) else {}
        except Exception:
            return {}

    def _backtest_debug_enabled(self) -> bool:
        """
        PositionMonitor 내부 디버그 이벤트를 backtest_history.csv에 남길지 여부
        기본값 True
        """
        try:
            return bool((self.cfg.get("system_settings", {}) or {}).get("backtest_debug_to_history", True))
        except Exception:
            return True

    def _make_backtest_debug_sink(self, sym: str, pos: dict, candle_t):
        """
        PositionMonitor -> BacktestEngine debug sink
        EM_* 이벤트를 backtest_history.csv로 흘려보낸다.
        """
        if not self._backtest_debug_enabled():
            return None

        side = str((pos or {}).get("side", "")).upper().strip()

        def _fmt(v):
            try:
                if v is None:
                    return "None"
                if isinstance(v, float):
                    if not math.isfinite(v):
                        return "None"
                    return f"{v:.10g}"
                return str(v)
            except Exception:
                return "None"

        def _compact(payload: dict) -> str:
            parts = []
            if isinstance(payload, dict):
                for k, v in payload.items():
                    parts.append(f"{k}={_fmt(v)}")
            return " | ".join(parts)[:800]

        def _sink(stage: str, payload: dict = None):
            try:
                self._log_csv(
                    candle_t,
                    sym,
                    side,
                    f"EM_{str(stage).upper()}",
                    0.0,
                    float((pos or {}).get("amount", 0.0) or 0.0),
                    0.0,
                    _compact(payload or {}),
                )
            except Exception:
                pass

        return _sink


    # =========================================================
    # 2. Simulation Loop
    # =========================================================
    def run(self, show_report=False):
        if not self.data_map:
            if not self.raw_data_map:
                self.prepare_data()
            self.rebuild_indicators()

        if not self.data_map:
            logger.error("❌ Cannot start backtest: No Data.")
            return

        # ✅ SL 적용 모드(전 구간 고정) — LIVE와 동일
        apply_mode = self._get_sl_apply_mode()

        fixed_symbols = sorted(list(self.data_map.keys()))

        # 1) 공통 교집합 시도
        start_times = [self.data_map[sym].index[0] for sym in fixed_symbols if not self.data_map[sym].empty]
        if not start_times:
            logger.error("❌ No valid start times.")
            return

        max_start = max(start_times)

        common_timeline = None
        for sym in fixed_symbols:
            df = self.data_map[sym]
            if df.empty:
                continue
            idx = df.index[df.index >= max_start]
            common_timeline = idx if common_timeline is None else common_timeline.intersection(idx)

        if common_timeline is None or len(common_timeline) < 200:
            logger.error("Not enough synchronized data (common timeline too short). Fallback to base timeline.")

            base_sym = None
            base_len = -1
            for s in fixed_symbols:
                df = self.data_map.get(s)
                if df is None or df.empty:
                    continue
                l = len(df.index[df.index >= max_start])
                if l > base_len or (l == base_len and (base_sym is None or s < base_sym)):
                    base_len = l
                    base_sym = s

            if base_sym is None or base_len < 250:
                logger.error("❌ Fallback timeline too short. Abort.")
                return

            candidate_timeline_full = self.data_map[base_sym].index
        else:
            candidate_timeline_full = common_timeline

        if candidate_timeline_full is None or len(candidate_timeline_full) < 250:
            logger.error("❌ Candidate timeline too short.")
            return

        aligned_full_map = {}
        usable_syms = []
        for sym in fixed_symbols:
            df = self.data_map.get(sym)
            if df is None or df.empty:
                continue

            aligned_full = df.reindex(candidate_timeline_full).ffill()
            aligned_full = aligned_full.loc[aligned_full.index >= max_start].copy()
            if aligned_full.empty:
                continue

            if aligned_full[self.required_cols].iloc[0].isna().any():
                logger.warning(f"[Align] {sym} still NaN at aligned start after ffill. Drop symbol.")
                continue

            aligned_full_map[sym] = aligned_full
            usable_syms.append(sym)

        if not aligned_full_map:
            logger.error("❌ Alignment failed: no usable symbols after reindex/ffill.")
            return

        common_idx = None
        for sym, df in aligned_full_map.items():
            common_idx = df.index if common_idx is None else common_idx.intersection(df.index)

        if common_idx is None or len(common_idx) < 250:
            logger.error("❌ Alignment failed: common index too short after intersection.")
            return

        final_map = {}
        for sym, df in aligned_full_map.items():
            final_df = df.reindex(common_idx).copy()
            if final_df.empty:
                continue
            if final_df[self.required_cols].iloc[0].isna().any():
                logger.warning(f"[Align] {sym} invalid after final common reindex. Drop symbol.")
                continue
            final_map[sym] = final_df

        if not final_map:
            logger.error("❌ Alignment failed: no usable symbols after final common reindex.")
            return

        self.data_map = final_map
        fixed_symbols = sorted(list(self.data_map.keys()))
        timeline = self.data_map[fixed_symbols[0]].index  # 모두 동일

        # [Reset]
        self.executor.history = []
        self.executor.cash = float(self.executor.initial_balance)
        self.executor.equity = float(self.executor.initial_balance)
        self.executor.positions = {}
        self.executor.equity_curve = []
        self.cooldowns = {}
        self.consecutive_losses = {}
        self.last_prices = {}

        sim_times = timeline[200:]
        if len(sim_times) < 200:
            logger.error(f"❌ Not enough simulation steps after warmup: {len(sim_times)}")
            return

        scan_stats = {
            "steps": 0,
            "skip_in_position": 0,
            "skip_cooldown": 0,
            "skip_blacklist": 0,
            "skip_no_slice": 0,
            "skip_short_slice": 0,
            "ctx_daily_not_ready": 0,
            "analyze_no_signal": 0,
            "signal_long": 0,
            "signal_short": 0,
            "candidate_ct": 0,
        }

        for current_time in sim_times:
            # ✅ [ADD] step counter (pending TTL용)
            self._step_i = int(getattr(self, "_step_i", 0)) + 1
            scan_stats["steps"] += 1

            current_rows = {}
            current_prices = {}

            for sym in fixed_symbols:
                row = self.data_map[sym].loc[current_time]
                row = self._inject_intrabar(sym, row, current_time)
                current_rows[sym] = row
                current_prices[sym] = float(row["close"])

            self.last_prices = dict(current_prices)

            # ✅ sizing 기준 통일: 먼저 equity 업데이트(강제 MTM)
            self._sync_equity(current_prices)

            # ✅ [ADD] pending entry sweep (이번 봉 high/low로 체결 판정 + TTL 만료 취소)
            self._sweep_pending_entry_orders(current_time, current_rows, current_prices)

            # Step 1: 포지션 관리
            for sym in fixed_symbols:
                if sym not in self.executor.positions:
                    continue
                curr_row = current_rows[sym]
                self._process_existing_position(sym, curr_row, None, apply_mode=apply_mode)

            # Step 2: 신규 진입 후보
            candidates = []
            for sym in fixed_symbols:
                if sym in self.executor.positions:
                    scan_stats["skip_in_position"] += 1
                    continue

                curr_row = current_rows[sym]
                df = self.data_map[sym]
                clean_sym = sym.split(':')[0]

                # 쿨다운
                if sym in self.cooldowns:
                    if current_time < self.cooldowns[sym]:
                        scan_stats["skip_cooldown"] += 1
                        continue
                    else:
                        del self.cooldowns[sym]

                if clean_sym in self.titan.blacklist:
                    scan_stats["skip_blacklist"] += 1
                    continue

                # 현재 봉 기준 daily 컨텍스트 준비 여부 관측만 한다. 동작은 바꾸지 않는다.
                try:
                    daily_ok_now = int(curr_row.get("ema_daily_ok", 0) or 0)
                    if daily_ok_now != 1:
                        scan_stats["ctx_daily_not_ready"] += 1
                except Exception:
                    scan_stats["ctx_daily_not_ready"] += 1

                # ✅ Titan9 봉인형 슬라이스(=LIVE와 동일)
                past_data = self._slice_for_t9(df, current_time)
                if past_data is None:
                    scan_stats["skip_no_slice"] += 1
                    continue

                # min_len 미만이면 스킵(계약 준수)
                if len(past_data) < int(self._t9_min_len()):
                    scan_stats["skip_short_slice"] += 1
                    continue

                signal, sl_price, _tp_price = self.titan.analyze(sym, past_data)

                if signal:
                    score = float(curr_row.get('adx', 0))
                    sig_norm = str(signal).upper()
                    if sig_norm == "LONG":
                        scan_stats["signal_long"] += 1
                    elif sig_norm == "SHORT":
                        scan_stats["signal_short"] += 1

                    scan_stats["candidate_ct"] += 1

                    candidates.append({
                        'score': score,
                        'sym': sym,
                        'signal': signal,
                        'sl': sl_price,
                        'row': curr_row,
                        'atr_ratio': float(curr_row.get("atr_ratio", 0.0) or 0.0),
                        'prices': current_prices
                    })
                else:
                    scan_stats["analyze_no_signal"] += 1

            candidates.sort(key=lambda x: x['score'], reverse=True)

            for cand in candidates:
                if len(self.executor.positions) >= self.executor.MAX_POSITIONS:
                    break
                self._process_entry(
                    cand['sym'],
                    cand['signal'],
                    cand['sl'],
                    cand['row'],
                    cand['prices']
                )

            # MTM 최종 업데이트 (강제 MTM)
            self._sync_equity(current_prices)
            self.executor.equity_curve.append({'dt': current_time, 'equity': float(self.executor.equity)})

        logger.info(f"🧪 BACKTEST_SCAN_STATS | {scan_stats}")

        # Equity curve save (이하 기존 그대로)
        try:
            if getattr(self.executor, "equity_curve", None):
                df_curve = pd.DataFrame(self.executor.equity_curve).copy()
                if "dt" in df_curve.columns:
                    df_curve.rename(columns={"dt": "Datetime"}, inplace=True)
                if "equity" in df_curve.columns:
                    df_curve.rename(columns={"equity": "Equity"}, inplace=True)

                df_curve["Datetime"] = pd.to_datetime(df_curve["Datetime"], errors="coerce")
                df_curve = df_curve.dropna(subset=["Datetime", "Equity"]).sort_values("Datetime")
                df_curve.to_csv(self.equity_curve_file, index=False, encoding="utf-8-sig")
                logger.info(f"📈 MTM Equity Curve Saved: {self.equity_curve_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save equity curve CSV: {e}")

        if show_report:
            try:
                self.executor.report()
            except Exception as e:
                logger.error(f"❌ Report Error: {e}")


    def _process_entry(self, sym, signal, sl, curr_row, current_prices: dict):
        # ✅ 0) signal 정규화
        sig = str(signal).strip().upper()
        alias = {
            "BUY": "LONG", "LONG": "LONG", "BULL": "LONG",
            "SELL": "SHORT", "SHORT": "SHORT", "BEAR": "SHORT",
        }
        sig = alias.get(sig, sig)
        if sig not in ("LONG", "SHORT"):
            return

        # 이미 포지션/이미 pending이면 스킵
        if sym in (self.executor.positions or {}):
            return
        peo = getattr(self, "pending_entry_orders", {}) or {}
        if sym in peo:
            return

        # ✅ anchor = signal 봉의 close (LIVE: 확정봉 close 기준)
        try:
            anchor = float(curr_row.get("close"))
            if (not math.isfinite(anchor)) or anchor <= 0:
                return
        except Exception:
            return

        # ✅ entry_atr = signal 봉의 atr
        entry_atr = None
        try:
            v = curr_row.get("atr", None)
            entry_atr = float(v) if v is not None else None
            if entry_atr is not None and ((not math.isfinite(entry_atr)) or entry_atr <= 0):
                entry_atr = None
        except Exception:
            entry_atr = None

        alpha = float(self._entry_alpha_atr())
        atr_for_offset = float(entry_atr or 0.0)
        if (not math.isfinite(atr_for_offset)) or atr_for_offset < 0:
            atr_for_offset = 0.0

        # ✅ limit_price = anchor ± alpha*ATR
        if sig == "LONG":
            limit_price = anchor - (alpha * atr_for_offset)
        else:
            limit_price = anchor + (alpha * atr_for_offset)

        if (not math.isfinite(limit_price)) or limit_price <= 0:
            limit_price = anchor

        # ✅ sizing은 LIVE와 동일하게 "주문가(limit_price) vs SL" 기준
        self._sync_equity(current_prices)
        current_equity = float(getattr(self.executor, "equity", self.executor.cash))
        amount = self.risk_ctrl.calculate_entry_size(sym, float(limit_price), current_equity, sl, sig)
        if amount <= 0:
            return

        # ✅ executor에 limit 오더 생성(등록)
        side_ccxt = "buy" if sig == "LONG" else "sell"
        res = None
        if hasattr(self.executor, "create_limit_order"):
            res = self.executor.create_limit_order(sym, side_ccxt, float(amount), float(limit_price))
        oid = (res or {}).get("order_id") if isinstance(res, dict) else None
        if not oid:
            return

        # ✅ pending에 등록: 다음 봉 1개 동안 체결 판정 후, 그 다음 봉 시작에 만료
        step_i = int(getattr(self, "_step_i", 0))
        peo[sym] = {
            "order_id": str(oid),
            "signal_side": sig,
            "amount": float(amount),
            "sl": float(sl) if sl is not None else None,
            "signal_time": curr_row.name,          # 라벨은 신호봉 시간
            "anchor_close": float(anchor),
            "alpha": float(alpha),
            "limit_price": float(limit_price),
            "entry_atr": float(entry_atr) if entry_atr is not None else None,
            "created_i": int(step_i),
            "expire_i": int(step_i) + 2,          # ✅ "다음 봉(1개)" 동안 체결 판정 후, 그 다음 봉 시작에 만료
        }
        self.pending_entry_orders = peo

        # 로그(주문 생성)
        try:
            self._log_csv(
                curr_row.name, sym, sig,
                "ENTRY_PENDING",
                float(limit_price),
                float(amount),
                0.0,
                f"LIMIT_PENDING | anchor={anchor} atr={atr_for_offset} alpha={alpha} limit={limit_price} oid={oid}"
            )
        except Exception:
            pass

    def _process_existing_position(self, sym, curr_row, new_signal, apply_mode: str = "next"):
        pos = self.executor.positions[sym]

        # -----------------------------
        # 0) normalize mode
        # -----------------------------
        try:
            mode = str(apply_mode or "next").strip().lower()
        except Exception:
            mode = "next"
        if mode not in ("next", "same"):
            mode = "next"

        # =========================================================
        # ✅ [ADD] LIVE와 동일: next_sl 승계/정리(모니터 호출 직전)
        # - next 모드: next_sl 있으면 sl로 승계하고 next_sl=None
        # - same 모드: next_sl 있으면 폐기(next_sl=None)
        # =========================================================
        if mode == "next":
            try:
                if ("next_sl" in pos) and (pos.get("next_sl") is not None):
                    nxt = float(pos.get("next_sl"))
                    cur = pos.get("sl", None)
                    cur_f = float(cur) if cur is not None else None
                    if (cur_f is None) or (nxt != cur_f):
                        pos["sl"] = float(nxt)
                    pos["next_sl"] = None
            except Exception:
                pass
        else:
            try:
                if pos.get("next_sl") is not None:
                    pos["next_sl"] = None
            except Exception:
                pass

        # -----------------------------
        # 1) config 기반 SL 전략/파라미터
        # -----------------------------
        sl_strategy = self._get_sl_strategy()
        sl_params = self._get_sl_params()

        # -----------------------------
        # 2) ARMOR용 히스토리 DF (현재 시점 포함)
        # -----------------------------
        hist_df = None
        try:
            df_full = self.data_map.get(sym, None)
            if isinstance(df_full, pd.DataFrame) and not df_full.empty:
                hist_df = df_full.loc[:curr_row.name]
                # 컬럼명 표준화(소문자)
                try:
                    rename_map = {}
                    for c in hist_df.columns:
                        lc = str(c).lower()
                        if lc in ("open", "high", "low", "close", "volume"):
                            rename_map[c] = lc
                    if rename_map:
                        hist_df = hist_df.rename(columns=rename_map)
                except Exception:
                    pass

                # lookback 제한
                try:
                    lb = int(sl_params.get("armor_lookback", 300))
                    if lb > 0 and len(hist_df) > lb:
                        hist_df = hist_df.tail(lb)
                except Exception:
                    pass
        except Exception:
            hist_df = None

        # -----------------------------
        # 3) market_data 구성 (NaN/inf 안전)
        # -----------------------------
        def _safe_float(x, default=0.0):
            try:
                v = float(x)
                if not math.isfinite(v):
                    return float(default)
                return v
            except Exception:
                return float(default)

        close = _safe_float(curr_row.get("close", 0.0), 0.0)
        high = _safe_float(curr_row.get("high", close), close)
        low  = _safe_float(curr_row.get("low", close), close)

        atr = curr_row.get("atr", None)
        atr = _safe_float(atr, close * 0.01)
        if atr <= 0:
            atr = close * 0.01

        st_val = curr_row.get("st_val", None)
        try:
            st_val = float(st_val) if st_val is not None else None
            if (st_val is not None) and (not math.isfinite(st_val)):
                st_val = None
        except Exception:
            st_val = None

        adx = curr_row.get("adx", 0.0)
        adx = _safe_float(adx, 0.0)

        # ✅ [ADD] Backtest debug sink
        debug_sink = self._make_backtest_debug_sink(
            sym=sym,
            pos=pos,
            candle_t=curr_row.name,
        )

        market_data = {
            "close": close,
            "high": high,
            "low": low,
            "atr": atr,
            "st_val": st_val,
            "adx": adx,
            "df": hist_df,
            "candle_time": str(curr_row.name),
            "debug_sink": debug_sink,
        }

        # -----------------------------
        # 4) PositionMonitor 호출
        # -----------------------------
        prev_breached = bool(pos.get("sl_breached", False))

        action, exec_price, reason, new_sl = self.monitor.check_conditions(
            sym,
            pos,
            market_data,
            sl_apply_mode=mode,
            sl_strategy=sl_strategy,
            sl_params=sl_params,
        )

        now_breached = bool(pos.get("sl_breached", False))
        latched_now = (mode == "next") and (not prev_breached) and now_breached
        if latched_now:
            try:
                self._log_csv(
                    curr_row.name, sym, str(pos.get("side", "")),
                    "SL_BREACH_LATCH", float(close), float(pos.get("amount", 0.0) or 0.0),
                    0.0,
                    f"next_mode sl_breached=1 breached_sl={pos.get('sl_breached_sl')} cur_sl={pos.get('sl')}"
                )
            except Exception:
                pass

        # -----------------------------
        # 5) SL 갱신 + trail_sl 저장/갱신 (핵심 수정)
        # -----------------------------
        if action == "UPDATE_SL":
            old_sl = pos.get("sl", None)
            old_next = pos.get("next_sl", None)

            try:
                if new_sl is not None:
                    new_sl_f = float(new_sl)
                    if (not math.isfinite(new_sl_f)) or (new_sl_f <= 0):
                        return

                    cur = pos.get("sl", None)
                    cur_f = float(cur) if cur is not None else None
                    nxt = pos.get("next_sl", None)
                    nxt_f = float(nxt) if nxt is not None else None

                    # ✅ LIVE와 동일: sl / next_sl만 반영 (trail_sl 엔진 개입 금지)
                    if mode == "next":
                        if (cur_f is None or new_sl_f != cur_f) and (nxt_f is None or new_sl_f != nxt_f):
                            pos["next_sl"] = new_sl_f
                    else:
                        if (cur_f is None) or (new_sl_f != cur_f):
                            pos["sl"] = new_sl_f
                        pos.pop("next_sl", None)

                    # (선택) 백테 로그 남기기
                    try:
                        self._log_csv(
                            curr_row.name, sym, str(pos.get("side", "")),
                            "UPDATE_SL", float(close), float(pos.get("amount", 0.0) or 0.0),
                            0.0,
                            f"{reason} | apply_mode={mode} | old_sl={old_sl} old_next={old_next} new_sl={new_sl_f} "
                            f"cur_sl={pos.get('sl', None)} next_sl={pos.get('next_sl', None)}"
                        )
                    except Exception:
                        pass

            except Exception:
                pass
            return

        elif action == "EXIT":
            self._execute_exit(sym, pos, exec_price, reason, curr_row.name)
            return

        # -----------------------------
        # 6) new_signal flip (기존 유지)
        # -----------------------------
        if new_signal:
            ns = str(new_signal).strip().upper()
            alias = {
                "BUY": "LONG", "LONG": "LONG", "BULL": "LONG",
                "SELL": "SHORT", "SHORT": "SHORT", "BEAR": "SHORT",
            }
            ns = alias.get(ns, ns)
        else:
            ns = None

        if ns and ns != str(pos.get("side", "")).upper():
            slippage = atr * 0.01
            base = close
            flip_price = base - slippage if pos.get("side", "").upper() == "LONG" else base + slippage
            self._execute_exit(sym, pos, flip_price, "SIGNAL_FLIP", curr_row.name)




    def _execute_exit(self, sym, pos, price, reason, dt):
        amount = float(pos['amount'])
        pnl = (price - pos['entry_price']) * amount if pos['side'] == 'LONG' else (pos['entry_price'] - price) * amount

        exit_value = price * amount
        fee = exit_value * BASE_FEE

        margin_locked = float(pos.get('margin', 0.0))
        self.executor.cash += margin_locked + pnl - fee

        net_pnl = pnl - fee

        # ✅ 동일 캔들 재진입 금지: 최소 1캔들 쿨다운
        min_next = dt + self.bar_td

        if net_pnl > 0:
            self.consecutive_losses[sym] = 0
            self.cooldowns[sym] = min_next
        else:
            current_streak = self.consecutive_losses.get(sym, 0) + 1
            self.consecutive_losses[sym] = current_streak

            if current_streak == 1:
                wait_hours = 8
            elif current_streak == 2:
                wait_hours = 24
            elif current_streak == 3:
                wait_hours = 48
            else:
                wait_hours = 96

            cd = dt + pd.Timedelta(hours=wait_hours)
            self.cooldowns[sym] = max(cd, min_next)

        # ✅ EXIT 직후에도 전체 가격으로 MTM 재평가
        prices2 = dict(self.last_prices) if self.last_prices else {}
        prices2[sym] = float(price)
        self._sync_equity(prices2)

        self._log_csv(dt, sym, pos['side'], 'EXIT', price, amount, net_pnl, reason)
        try:
            pos.pop("trail_sl", None)
        except Exception:
            pass
        del self.executor.positions[sym]
        self.executor.history.append({'dt': dt, 'sym': sym, 'type': 'EXIT', 'pnl': net_pnl, 'reason': reason})


    def _get_universe_export_path(self, out_path: str = None) -> str:
        """
        universe.json 저장 경로 결정
        우선순위:
        1) 인자 out_path
        2) config: system_settings.universe_export_path
        3) <root_dir>/universe.json
        """
        import os

        path = out_path
        if not path:
            try:
                path = (self.cfg.get("system_settings", {}) or {}).get("universe_export_path", None)
            except Exception:
                path = None

        if not path:
            path = os.path.join(self.root_dir, "universe2.json")

        try:
            if not os.path.isabs(path):
                path = os.path.join(self.root_dir, path)
        except Exception:
            pass

        return path


    def _collect_symbol_stats_from_backtest_log(self) -> dict:
        """
        backtest_history.csv(=self.log_file)에서 EXIT만 집계하여 심볼별 성과 산출
        - CSV가 reason 컬럼에 콤마가 섞여도 깨지지 않게 split(',', 9)로 파싱
        반환:
        {sym: {"trades":int,"wins":int,"losses":int,"pnl_total":float,"avg_pnl":float,"win_rate":float,"profit_factor":float}}
        """
        import os
        import math

        path = getattr(self, "log_file", None) or ""
        if (not path) or (not os.path.exists(path)):
            return {}

        raw = {}  # sym -> accumulator
        try:
            with open(path, "r", encoding="utf-8") as f:
                _ = next(f, None)  # header skip
                for line in f:
                    line = (line or "").strip()
                    if not line:
                        continue

                    parts = line.split(",", 9)  # 10 columns max
                    if len(parts) < 9:
                        continue

                    # Datetime,Symbol,Side,Type,Price,Amount,PnL,Cash,Equity,Reason
                    # reason는 parts[9]에 통째로 들어감(있으면)
                    try:
                        sym = str(parts[1]).strip()
                        typ = str(parts[3]).strip()
                    except Exception:
                        continue

                    if not sym or typ != "EXIT":
                        continue

                    try:
                        pnl = float(parts[6])
                        if not math.isfinite(pnl):
                            continue
                    except Exception:
                        continue

                    acc = raw.get(sym)
                    if not acc:
                        acc = {
                            "trades": 0,
                            "wins": 0,
                            "losses": 0,
                            "pnl_total": 0.0,
                            "pnl_win": 0.0,
                            "pnl_loss": 0.0,  # 음수 누적
                        }
                        raw[sym] = acc

                    acc["trades"] += 1
                    acc["pnl_total"] += pnl
                    if pnl > 0:
                        acc["wins"] += 1
                        acc["pnl_win"] += pnl
                    else:
                        acc["losses"] += 1
                        acc["pnl_loss"] += pnl
        except Exception:
            return {}

        out = {}
        for sym, acc in raw.items():
            t = int(acc["trades"] or 0)
            if t <= 0:
                continue

            wins = int(acc["wins"] or 0)
            losses = int(acc["losses"] or 0)
            pnl_total = float(acc["pnl_total"] or 0.0)
            avg_pnl = pnl_total / t if t > 0 else 0.0
            win_rate = wins / t if t > 0 else 0.0

            pnl_win = float(acc["pnl_win"] or 0.0)
            pnl_loss = float(acc["pnl_loss"] or 0.0)  # 음수 or 0
            if pnl_loss < 0:
                profit_factor = pnl_win / abs(pnl_loss) if abs(pnl_loss) > 0 else float("inf")
            else:
                profit_factor = float("inf") if pnl_win > 0 else 0.0

            out[sym] = {
                "trades": t,
                "wins": wins,
                "losses": losses,
                "pnl_total": float(pnl_total),
                "avg_pnl": float(avg_pnl),
                "win_rate": float(win_rate),
                "profit_factor": float(profit_factor),
            }

        return out


    def export_universe_json(self, top_n: int = None, min_trades: int = None, out_path: str = None) -> dict:
        """
        백테스트 결과(로그)로 universe.json 생성/저장

        기본 규칙(결정론):
        - EXIT 기준 심볼별 pnl_total 내림차순
        - 동률 타이브레이커: profit_factor, win_rate, trades, 심볼명 오름차순
        - min_trades 미만 심볼은 제외(부족하면 trades>=1에서 다시 채움)
        - 저장 포맷: {"universe":[...], "meta":{...}}
        (BacktestEngine._get_universe_from_json이 {"universe":[...]}를 바로 읽을 수 있음)
        """
        import os
        import json
        import math
        from datetime import datetime, timezone

        # defaults
        if top_n is None:
            try:
                top_n = int((self.cfg.get("system_settings", {}) or {}).get("universe_size", 24) or 24)
            except Exception:
                top_n = 24
        if top_n <= 0:
            top_n = 24

        if min_trades is None:
            try:
                min_trades = int((self.cfg.get("system_settings", {}) or {}).get("universe_min_trades", 3) or 3)
            except Exception:
                min_trades = 3
        if min_trades < 0:
            min_trades = 0

        stats = self._collect_symbol_stats_from_backtest_log()

        def _pf_key(x):
            # inf를 매우 큰 값으로 정렬키화
            try:
                v = float(x)
                if math.isfinite(v):
                    return v
                return 1e18
            except Exception:
                return 0.0

        # 1차 후보: min_trades 충족
        primary = [s for s, st in stats.items() if int(st.get("trades", 0)) >= int(min_trades)]
        # 성과 없는 심볼(거래 0)은 이미 없음
        def _sort_key(sym):
            st = stats.get(sym, {}) or {}
            return (
                -float(st.get("pnl_total", 0.0) or 0.0),
                -_pf_key(st.get("profit_factor", 0.0)),
                -float(st.get("win_rate", 0.0) or 0.0),
                -int(st.get("trades", 0) or 0),
                str(sym),
            )

        primary_sorted = sorted(primary, key=_sort_key)
        selected = primary_sorted[:top_n]

        # 부족하면 trades>=1 풀에서 채움
        if len(selected) < top_n:
            pool = [s for s, st in stats.items() if int(st.get("trades", 0)) >= 1 and s not in set(selected)]
            pool_sorted = sorted(pool, key=_sort_key)
            need = top_n - len(selected)
            selected.extend(pool_sorted[:need])

        # 저장
        path = self._get_universe_export_path(out_path)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except Exception:
            pass

        meta = {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "days": int(getattr(self, "test_days", 0) or 0),
            "top_n": int(top_n),
            "min_trades": int(min_trades),
            "source_log": os.path.basename(getattr(self, "log_file", "backtest_history.csv") or "backtest_history.csv"),
            "metric": "pnl_total(EXIT net_pnl)",
        }

        payload = {
            "universe": list(selected),
            "meta": meta,
            # 필요하면 아래 주석 해제: 선택된 심볼의 요약 통계도 같이 저장
            "stats": {s: stats.get(s, {}) for s in selected},
        }

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            # 실패해도 시스템이 죽지 않게 dict만 반환
            return payload

        return payload


    def run_and_export_universe(self, show_report: bool = False, universe_size: int = None, min_trades: int = None, out_path: str = None) -> dict:
        """
        기존 run()은 그대로 두고,
        백테스트 수행 후 universe.json을 반드시 생성하는 래퍼.
        """
        self.run(show_report=show_report)
        return self.export_universe_json(top_n=universe_size, min_trades=min_trades, out_path=out_path)



if __name__ == "__main__":
    engine = BacktestEngine(days=7)
    engine.prepare_data()
    engine.rebuild_indicators()
    engine.run_and_export_universe(show_report=True)