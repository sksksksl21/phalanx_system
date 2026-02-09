import sys
import os
import json
import logging
import pandas as pd
import numpy as np
import math

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

    def __init__(self, days=30):
        self.root_dir = root_dir
        self.config_path = os.path.join(root_dir, "config.json")
        self.cfg = self._load_config()

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
            self.executor.initial_balance = 10000.0
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

        # 로그 파일 초기화 (윈도우별로 교체 가능하도록 reset 함수로 통일)
        self.log_file = os.path.join(root_dir, "backtest_history.csv")
        self._reset_log_file(self.log_file)

        # MTM Equity Curve CSV 경로 (대시보드용)
        self.equity_curve_file = os.path.join(root_dir, "backtest_equity_curve.csv")

    def _load_config(self):
        if not os.path.exists(self.config_path):
            return {}
        with open(self.config_path, 'r', encoding="utf-8") as f:
            return json.load(f)
        

    def _get_raw_cache_path(self) -> str:
        """
        RAW 캐시 파일 경로를 config 또는 기본값으로 결정
        - system_settings.raw_cache_path 가 있으면 우선
        - 없으면 root_dir/market_data_cache_30d_uni.pkl
        """
        try:
            p = (self.cfg.get("system_settings", {}) or {}).get("raw_cache_path", None)
        except Exception:
            p = None

        if not p:
            p = os.path.join(self.root_dir, "market_data_cache_30d_uni.pkl")

        try:
            if not os.path.isabs(p):
                p = os.path.join(self.root_dir, p)
        except Exception:
            pass

        return os.path.abspath(p)


    def _load_raw_cache_pkl(self) -> dict:
        """
        ✅ VirtualExecutor 다운로드를 막기 위한 핵심:
        - RAW cache(pkl)를 로드해서 {sym: df} 반환
        - 실패 시 {} 반환
        """
        import pickle

        path = self._get_raw_cache_path()
        if not os.path.exists(path):
            logger.warning(f"⚠️ RAW_CACHE_NOT_FOUND | path={path}")
            return {}

        try:
            with open(path, "rb") as f:
                obj = pickle.load(f) or {}
            if not isinstance(obj, dict):
                logger.warning(f"⚠️ RAW_CACHE_INVALID_TYPE | type={type(obj)} path={path}")
                return {}
            # 최소 검증: df-like만
            out = {}
            for sym, df in obj.items():
                if df is None:
                    continue
                try:
                    # DataFrame인지 확인
                    if hasattr(df, "columns") and hasattr(df, "__len__"):
                        out[str(sym)] = df
                except Exception:
                    continue

            logger.info(f"💾 RAW_CACHE_LOADED | path={path} symbols={len(out)}")
            return out
        except Exception as e:
            logger.error(f"❌ RAW_CACHE_LOAD_FAIL | path={path} err={e}")
            return {}



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

    def _get_symbol_pool_from_cache_pkl(self, pkl_path: str = None) -> list:
        """
        market_data_cache_30d.pkl 같은 캐시에서 '심볼 풀'을 뽑는다.
        - pkl이 dict면 keys를 심볼로 간주
        - pkl이 {sym: df} 형태를 기대
        """
        if pkl_path is None:
            # 기본값: 프로젝트 루트에 둔 캐시
            pkl_path = os.path.join(self.root_dir, "market_data_cache_30d_uni.pkl")

        if not os.path.exists(pkl_path):
            return []

        try:
            import pickle
            with open(pkl_path, "rb") as f:
                obj = pickle.load(f)
        except Exception:
            return []

        if not isinstance(obj, dict):
            return []

        syms = []
        for k in obj.keys():
            try:
                s = str(k).strip()
                if not s:
                    continue
                syms.append(s)
            except Exception:
                continue

        # 중복 제거(순서 보존)
        out = []
        seen = set()
        for s in syms:
            if s in seen:
                continue
            seen.add(s)
            out.append(s)

        return out

    def _build_targets_top_n_without_executor(self, top_n: int) -> list:
        """
        executor.get_top_targets()를 쓰지 않고 top_n 타겟 생성.
        우선순위:
        1) system_settings.universe_pool_path (json) 있으면 그걸 사용
        2) market_data_cache_30d.pkl 키를 사용
        3) 마지막 폴백: executor.get_top_targets() (어차피 25)
        """
        try:
            n = int(top_n)
            if n <= 0:
                n = 50
        except Exception:
            n = 50

        # 1) JSON 심볼 풀(사용자가 직접 관리 가능)
        pool_path = None
        try:
            pool_path = (self.cfg.get("system_settings", {}) or {}).get("universe_pool_path", None)
        except Exception:
            pool_path = None

        pool = []
        if pool_path:
            try:
                if not os.path.isabs(pool_path):
                    pool_path = os.path.join(self.root_dir, pool_path)
                if os.path.exists(pool_path):
                    with open(pool_path, "r", encoding="utf-8") as f:
                        obj = json.load(f) or {}
                    if isinstance(obj, dict) and isinstance(obj.get("symbols", None), list):
                        pool = [str(x).strip() for x in obj.get("symbols", []) if str(x).strip()]
                    elif isinstance(obj, list):
                        pool = [str(x).strip() for x in obj if str(x).strip()]
            except Exception:
                pool = []

        # 2) PKL 캐시 키
        if not pool:
            pool = self._get_symbol_pool_from_cache_pkl()

        # 3) 폴백(25 고정)
        if not pool:
            try:
                pool = self.executor.get_top_targets() or []
            except Exception:
                pool = []

        # 블랙리스트 제거 + 앞에서 n개
        out = []
        for sym in pool:
            try:
                s = str(sym).strip()
                if not s:
                    continue
                clean = s.split(":")[0]
                if clean in self.titan.blacklist or s in self.titan.blacklist:
                    continue
                out.append(s)
            except Exception:
                continue

        # 중복 제거(순서 보존)
        final = []
        seen = set()
        for s in out:
            if s in seen:
                continue
            seen.add(s)
            final.append(s)

        return final[:n]


    def _get_top_targets_n(self, top_n: int) -> list:
        """
        VirtualExecutor.get_top_targets()가 N을 받으면 그대로 사용,
        아니면 반환 리스트를 슬라이스해서 top_n을 맞춘다.
        """
        try:
            n = int(top_n)
            if n <= 0:
                n = 50
        except Exception:
            n = 50

        try:
            fn = getattr(self.executor, "get_top_targets", None)
            if fn is None:
                return []

            # (A) get_top_targets(n) 지원 시
            try:
                out = fn(n)
                if isinstance(out, list) and out:
                    return out[:n]
            except TypeError:
                pass
            except Exception:
                pass

            # (B) fallback: get_top_targets() 후 슬라이스
            out = fn()
            if not isinstance(out, list):
                return []
            return out[:n]
        except Exception:
            return []


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

    # =========================================================
    # 1. Data Preparation Layer
    # =========================================================
    def prepare_data(self, symbols=None, top_n: int = None, days: int = None):
        """
        ✅ 유니버스 확정 백테 전용 동작:
        - RAW PKL 캐시가 있으면: 캐시 심볼이 권위 (VirtualExecutor.get_top_targets() 절대 사용 안 함)
        - top_n은 캐시 심볼에서 자른다.
        - executor는 건드리지 않는다.
        """
        logger.info("📥 [Data Loader] Loading Historical Data... (CACHE FIRST)")

        # ---- sanitize ----
        try:
            top_n = int(top_n) if top_n is not None else None
            if top_n is not None and top_n <= 0:
                top_n = None
        except Exception:
            top_n = None

        try:
            days = int(days) if days is not None else None
            if days is not None and days <= 0:
                days = None
        except Exception:
            days = None

        # ---------------------------------------------------------
        # 1) 캐시 먼저 로드
        # ---------------------------------------------------------
        cached = {}
        try:
            cached = self._load_raw_cache_pkl()
        except Exception:
            cached = {}

        # ---------------------------------------------------------
        # 2) ✅ 캐시가 있으면 "캐시 심볼"이 권위
        # ---------------------------------------------------------
        if cached:
            # (A) 외부 주입 symbols가 있으면 그 교집합
            if symbols:
                base = [s for s in list(symbols) if s in cached]
            else:
                # (B) ✅ symbols=None이면 executor.get_top_targets() 쓰지 말고 캐시 keys 사용
                base = list(cached.keys())

            # 블랙리스트 제거
            filtered = []
            for sym in base:
                clean = str(sym).split(":")[0]
                if clean in self.titan.blacklist or sym in self.titan.blacklist:
                    continue
                filtered.append(sym)

            # top_n 적용
            if top_n is not None:
                filtered = filtered[:top_n]

            raw_data_map = {sym: cached[sym] for sym in filtered if sym in cached}

            # 방어: filtered가 비었으면 raw_data_map도 비게 됨
            sorted_symbols = sorted(list(raw_data_map.keys()))
            self.raw_data_map = {sym: raw_data_map[sym] for sym in sorted_symbols}
            self.symbols = list(self.raw_data_map.keys())

            logger.info(f"✅ Raw Data Ready (CACHE): {len(self.raw_data_map)} symbols loaded.")
            return self.raw_data_map

        # ---------------------------------------------------------
        # 3) 캐시 없으면 fallback 다운로드(기존 동작)
        # ---------------------------------------------------------
        logger.warning("⚠️ RAW_CACHE_EMPTY -> FALLBACK_DOWNLOAD (VirtualExecutor will download)")

        if symbols:
            targets = list(symbols)
        else:
            targets = []
            try:
                if hasattr(self.executor, "get_top_targets"):
                    try:
                        targets = self.executor.get_top_targets(top_n=top_n) if top_n is not None else self.executor.get_top_targets()
                    except TypeError:
                        targets = self.executor.get_top_targets()
            except Exception:
                targets = []

        # blacklist
        filtered_targets = []
        for sym in targets:
            clean_sym = str(sym).split(':')[0]
            if clean_sym in self.titan.blacklist or sym in self.titan.blacklist:
                continue
            filtered_targets.append(sym)
        targets = filtered_targets

        try:
            if days is not None:
                raw_data_map = self.executor.prepare_data(targets, days=days)
            else:
                raw_data_map = self.executor.prepare_data(targets)
        except TypeError:
            raw_data_map = self.executor.prepare_data(targets)

        if not raw_data_map:
            logger.error("❌ No Data Loaded.")
            return {}

        sorted_symbols = sorted(list(raw_data_map.keys()))
        self.raw_data_map = {sym: raw_data_map[sym] for sym in sorted_symbols}
        self.symbols = list(self.raw_data_map.keys())

        logger.info(f"✅ Raw Data Ready (DOWNLOAD): {len(self.raw_data_map)} symbols loaded.")
        return self.raw_data_map



    def rebuild_indicators(self):
        """
        - dropna()는 '필수 컬럼 subset' 기준으로만 적용한다.
        - 워밍업(warmup)은 각 필수 컬럼의 첫 유효시점 중 '가장 늦은 시점'을 사용한다.
        - 심볼별 warmup으로 슬라이스
        """
        self.data_map = {}

        self.required_cols = [
            "open", "high", "low", "close", "volume",
            "atr", "vol_ma", "ema_intra", "rsi", "adx", "st_val", "st_dir"
        ]

        temp_map = {}
        warmup_map = {}

        for sym, df in self.raw_data_map.items():
            try:
                ind = self.titan.calculate_indicators(sym, df.copy())

                missing = [c for c in self.required_cols if c not in ind.columns]
                if missing:
                    logger.warning(f"[Indicator] {sym} missing required cols: {missing}")
                    continue

                if ind[self.required_cols].isna().all().all():
                    logger.warning(f"[Indicator] {sym} required columns all-NaN (pre-slice).")
                    continue

                bad_cols = [c for c in self.required_cols if ind[c].isna().all()]
                if bad_cols:
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
                logger.warning(f"[Indicator] {sym} indicator failed: {e}")

        for sym, ind in temp_map.items():
            try:
                sym_warmup = int(warmup_map.get(sym, 0))
                sliced = ind.iloc[sym_warmup:].copy()

                if len(sliced) == 0:
                    logger.warning(f"[Indicator] {sym} empty after slice")
                    continue

                sliced = sliced.dropna(subset=self.required_cols)
                if len(sliced) == 0:
                    logger.warning(f"[Indicator] {sym} all NaN after drop (required subset)")
                    continue

                self.data_map[sym] = sliced

            except Exception as e:
                logger.warning(f"[Indicator] {sym} slice failed: {e}")

        logger.info(f"Indicators Ready: {len(self.data_map)} symbols processed.")

    def _reset_log_file(self, path: str):
        """
        윈도우별 backtest_history.csv를 새로 쓰기 위해 헤더를 강제로 재작성.
        """
        try:
            if not os.path.isabs(path):
                path = os.path.join(self.root_dir, path)
        except Exception:
            path = os.path.join(self.root_dir, "backtest_history.csv")

        self.log_file = os.path.abspath(path)

        try:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        except Exception:
            pass

        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("Datetime,Symbol,Side,Type,Price,Amount,PnL,Cash,Equity,Reason\n")



    def _log_csv(self, dt, sym, side, type_note, price, amt, pnl, reason):
        cash = float(getattr(self.executor, "cash", 0.0))
        eq = float(getattr(self.executor, "equity", cash))
        line = f"{dt},{sym},{side},{type_note},{price},{amt},{pnl:.4f},{cash:.2f},{eq:.2f},{reason}\n"
        with open(self.log_file, 'a', encoding="utf-8") as f:
            f.write(line)

    def _build_symbol_performance_table_from_csv(self, csv_path: str = None) -> pd.DataFrame:
        """
        backtest_history.csv(엔진이 기록한 로그)에서 종목별 성과 요약 테이블 생성.
        - Type == 'EXIT' 만 집계 대상으로 사용
        """
        path = csv_path or self.log_file
        if (not os.path.exists(path)) or os.path.getsize(path) == 0:
            return pd.DataFrame()

        try:
            df = pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

        if df.empty or "Type" not in df.columns or "Symbol" not in df.columns:
            return pd.DataFrame()

        df = df.copy()
        df["Type"] = df["Type"].astype(str).str.upper().str.strip()
        exits = df[df["Type"] == "EXIT"].copy()
        if exits.empty:
            return pd.DataFrame()

        # PnL 숫자화
        try:
            exits["PnL"] = pd.to_numeric(exits["PnL"], errors="coerce").fillna(0.0)
        except Exception:
            exits["PnL"] = 0.0

        rows = []
        for sym, g in exits.groupby("Symbol"):
            pnls = g["PnL"].astype(float).values
            trades = int(len(pnls))
            wins = int(np.sum(pnls > 0))
            losses = int(np.sum(pnls <= 0))
            winrate = (wins / trades) if trades > 0 else 0.0

            gross_profit = float(np.sum(pnls[pnls > 0])) if trades > 0 else 0.0
            gross_loss = float(-np.sum(pnls[pnls < 0])) if trades > 0 else 0.0
            pf = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

            avg_win = float(np.mean(pnls[pnls > 0])) if wins > 0 else 0.0
            avg_loss = float(np.mean(pnls[pnls < 0])) if losses > 0 else 0.0

            eq = np.cumsum(pnls)
            peak = np.maximum.accumulate(eq) if len(eq) else np.array([0.0])
            dd = (peak - eq)
            mdd = float(np.max(dd)) if len(dd) else 0.0

            total_pnl = float(np.sum(pnls))

            rows.append({
                "symbol": sym,
                "trades": trades,
                "winrate": winrate,
                "pf": float(pf),
                "total_pnl": total_pnl,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "mdd_trade_based": mdd,
            })

        out = pd.DataFrame(rows)
        if out.empty:
            return out

        out = out.sort_values(["pf", "total_pnl"], ascending=[False, False]).reset_index(drop=True)
        return out

    def _select_universe_from_table(
        self,
        perf_df: pd.DataFrame,
        max_universe_size: int = 25,
        min_trades: int = 30,
        min_pf: float = 1.05,
        max_mdd_trade_based: float = None,
    ) -> list:
        """
        성과 테이블(perf_df)에서 universe 심볼 리스트를 선정한다.
        - trade-based MDD는 가격 기반 MDD가 아니지만, 최소한의 tail-risk 필터로 사용 가능
        """
        if perf_df is None or perf_df.empty:
            return []

        df = perf_df.copy()

        try:
            df["trades"] = pd.to_numeric(df["trades"], errors="coerce").fillna(0).astype(int)
            df["pf"] = pd.to_numeric(df["pf"], errors="coerce").fillna(0.0).astype(float)
            df["mdd_trade_based"] = pd.to_numeric(df["mdd_trade_based"], errors="coerce").fillna(0.0).astype(float)
        except Exception:
            pass

        # 1) 표본수 필터
        df = df[df["trades"] >= int(min_trades)]

        # 2) PF 필터
        df = df[df["pf"] >= float(min_pf)]

        # 3) trade-based MDD 필터(옵션)
        if max_mdd_trade_based is not None:
            df = df[df["mdd_trade_based"] <= float(max_mdd_trade_based)]

        if df.empty:
            return []

        # 4) 스코어링: pf*(1/(1+mdd)) * log(1+trades)
        try:
            score = (df["pf"] * (1.0 / (1.0 + df["mdd_trade_based"])) * np.log1p(df["trades"]))
            df = df.assign(score=score)
        except Exception:
            df = df.assign(score=df["pf"])

        df = df.sort_values(["score", "pf", "total_pnl"], ascending=[False, False, False])

        # 5) 상위 N개
        try:
            n = int(max_universe_size)
            if n <= 0:
                n = 25
        except Exception:
            n = 25

        syms = []
        for s in df["symbol"].tolist():
            try:
                ss = str(s).strip()
                if ss and ss not in syms:
                    syms.append(ss)
            except Exception:
                continue

        return syms[:n]

    def _write_universe_json(self, symbols: list, out_path: str = None, meta: dict = None) -> str:
        """
        universe.json 저장
        포맷: {"universe":[...], "meta": {...}}
        """
        if out_path is None:
            out_path = os.path.join(self.root_dir, "universe.json")
        else:
            try:
                if not os.path.isabs(out_path):
                    out_path = os.path.join(self.root_dir, out_path)
            except Exception:
                out_path = os.path.join(self.root_dir, "universe.json")

        payload = {
            "universe": list(symbols or []),
            "meta": meta or {},
        }

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ universe.json write failed: {e}")
            return ""

        return out_path


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


    def _get_common_data_end_time(self):
        """
        현재 data_map 기준으로 '모든 심볼이 커버하는 공통 종료시점'을 계산.
        슬라이딩 윈도우의 기준 end_time으로 사용.
        """
        if not self.data_map:
            return None

        ends = []
        for sym, df in self.data_map.items():
            try:
                if df is None or df.empty:
                    continue
                ends.append(df.index[-1])
            except Exception:
                continue

        if not ends:
            return None

        # 모든 심볼이 가지고 있는 공통 구간을 보수적으로 잡기 위해 '최소 end' 사용
        try:
            return min(ends)
        except Exception:
            return None


    # =========================================================
    # 2. Simulation Loop
    # =========================================================
    def run_window(self, start_time=None, end_time=None, show_report=False, warmup_bars: int = 200):
        """
        기존 run() 로직을 유지하되, 특정 기간(start_time~end_time)만 시뮬레이션한다.
        - start_time 이전 warmup_bars는 지표/정렬을 위해 필요 (시뮬레이션은 start_time부터)
        """
        if not self.data_map:
            if not self.raw_data_map:
                self.prepare_data()
            self.rebuild_indicators()

        if not self.data_map:
            logger.error("❌ Cannot start backtest: No Data.")
            return

        apply_mode = self._get_sl_apply_mode()

        fixed_symbols = sorted(list(self.data_map.keys()))

        # 1) 공통 교집합 시도 (기존과 동일)
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

        # ✅ 여기까지는 기존 run()과 동일
        self.data_map = final_map
        fixed_symbols = sorted(list(self.data_map.keys()))
        timeline_full = self.data_map[fixed_symbols[0]].index  # 모두 동일

        # ---------------------------------------------------------
        # ✅ (추가) 윈도우 기간 slice
        # ---------------------------------------------------------
        timeline = timeline_full
        if start_time is not None:
            try:
                st = pd.to_datetime(start_time, utc=True) if not hasattr(start_time, "tzinfo") else start_time
            except Exception:
                st = start_time

            # 지표/ffill 안정화를 위해 warmup_bars만큼 앞도 포함
            try:
                warmup_delta = self.bar_td * int(warmup_bars)
                st_warm = st - warmup_delta
            except Exception:
                st_warm = st

            timeline = timeline[timeline >= st_warm]

        if end_time is not None:
            try:
                et = pd.to_datetime(end_time, utc=True) if not hasattr(end_time, "tzinfo") else end_time
            except Exception:
                et = end_time
            timeline = timeline[timeline <= et]

        if timeline is None or len(timeline) < (warmup_bars + 50):
            logger.error(f"❌ Window timeline too short. len={0 if timeline is None else len(timeline)}")
            return

        # ---------------------------------------------------------
        # [Reset] (기존과 동일)
        # ---------------------------------------------------------
        self.executor.history = []
        self.executor.cash = float(self.executor.initial_balance)
        self.executor.equity = float(self.executor.initial_balance)
        self.executor.positions = {}
        self.executor.equity_curve = []
        self.cooldowns = {}
        self.consecutive_losses = {}
        self.last_prices = {}

        # warmup_bars 이후부터 진행하되, 실제 시뮬레이션은 start_time 이후로 제한
        sim_times_all = timeline[int(warmup_bars):]
        if len(sim_times_all) < 200:
            logger.error(f"❌ Not enough simulation steps after warmup: {len(sim_times_all)}")
            return

        if start_time is not None:
            try:
                st2 = pd.to_datetime(start_time, utc=True) if not hasattr(start_time, "tzinfo") else start_time
            except Exception:
                st2 = start_time
            sim_times = sim_times_all[sim_times_all >= st2]
        else:
            sim_times = sim_times_all

        if end_time is not None:
            try:
                et2 = pd.to_datetime(end_time, utc=True) if not hasattr(end_time, "tzinfo") else end_time
            except Exception:
                et2 = end_time
            sim_times = sim_times[sim_times <= et2]

        if sim_times is None or len(sim_times) < 50:
            logger.error(f"❌ Window sim_times too short: {0 if sim_times is None else len(sim_times)}")
            return

        # ---------------------------------------------------------
        # 메인 루프 (기존 run()과 동일, 단 timeline 대신 sim_times 사용)
        # ---------------------------------------------------------
        for current_time in sim_times:
            current_rows = {}
            current_prices = {}

            for sym in fixed_symbols:
                row = self.data_map[sym].loc[current_time]
                current_rows[sym] = row
                current_prices[sym] = float(row["close"])

            self.last_prices = dict(current_prices)

            self._sync_equity(current_prices)

            # Step 1: 포지션 관리
            for sym in fixed_symbols:
                if sym not in self.executor.positions:
                    continue

                curr_row = current_rows[sym]
                pos = self.executor.positions[sym]

                if apply_mode == "next":
                    if 'next_sl' in pos and pos['next_sl'] is not None:
                        try:
                            if float(pos['next_sl']) != float(pos.get('sl', 0)):
                                pos['sl'] = float(pos['next_sl'])
                        except Exception:
                            pass
                        pos.pop('next_sl', None)
                else:
                    if 'next_sl' in pos and pos['next_sl'] is not None:
                        pos.pop('next_sl', None)

                self._process_existing_position(sym, curr_row, None, apply_mode=apply_mode)

            # Step 2: 신규 진입 후보
            candidates = []
            for sym in fixed_symbols:
                if sym in self.executor.positions:
                    continue

                curr_row = current_rows[sym]
                df = self.data_map[sym]
                clean_sym = sym.split(':')[0]

                if sym in self.cooldowns:
                    if current_time < self.cooldowns[sym]:
                        continue
                    else:
                        del self.cooldowns[sym]

                if clean_sym in self.titan.blacklist:
                    continue

                curr_idx_int = df.index.get_loc(current_time)
                start_idx = max(0, curr_idx_int - 250)
                past_data = df.iloc[start_idx: curr_idx_int + 1]

                signal, sl_price, _tp_price = self.titan.analyze(sym, past_data)

                if signal:
                    score = float(curr_row.get('adx', 0))
                    candidates.append({
                        'score': score,
                        'sym': sym,
                        'signal': signal,
                        'sl': sl_price,
                        'row': curr_row,
                        'prices': current_prices
                    })

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

            self._sync_equity(current_prices)
            self.executor.equity_curve.append({'dt': current_time, 'equity': float(self.executor.equity)})

        # equity curve save는 기존 run() 로직 그대로 유지
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


    def run(self, show_report=False):
        # 기존 run()은 전체기간 실행 → run_window로 위임
        return self.run_window(start_time=None, end_time=None, show_report=show_report, warmup_bars=200)


    def run_universe_selection_backtest(
        self,
        top_n: int = 100,
        universe_size: int = 25,
        min_trades: int = 30,
        min_pf: float = 1.05,
        max_mdd_trade_based: float = None,
        perf_out_csv: str = "symbol_performance.csv",
        universe_out_json: str = "universe.json",
    ):
        """
        [유니버스 확정 전용 백테]
        1) top_n 심볼로 데이터 준비
        2) 파라미터 고정(=config/titan 그대로) 상태로 포트폴리오 백테 run
        3) backtest_history.csv 기반으로 종목별 성과 테이블 생성
        4) 룰 기반 universe 선정 후 universe.json 저장
        """
        # 1) 데이터
        self.prepare_data(symbols=None, top_n=int(top_n))
        self.rebuild_indicators()

        if not self.data_map:
            logger.error("❌ Universe selection aborted: No data_map.")
            return

        # 2) 백테 실행(파라미터 고정)
        self.run(show_report=False)

        # 3) 종목별 성과 테이블
        perf = self._build_symbol_performance_table_from_csv()
        if perf is None or perf.empty:
            logger.error("❌ No symbol performance data (no EXIT logs).")
            return

        # 저장
        try:
            out_csv_path = perf_out_csv
            if not os.path.isabs(out_csv_path):
                out_csv_path = os.path.join(self.root_dir, out_csv_path)
            perf.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
            logger.info(f"📊 Symbol performance saved: {out_csv_path}")
        except Exception as e:
            logger.error(f"❌ perf csv save failed: {e}")

        # 4) universe 선정
        universe_syms = self._select_universe_from_table(
            perf_df=perf,
            max_universe_size=int(universe_size),
            min_trades=int(min_trades),
            min_pf=float(min_pf),
            max_mdd_trade_based=max_mdd_trade_based,
        )

        meta = {
            "top_n": int(top_n),
            "universe_size": int(universe_size),
            "min_trades": int(min_trades),
            "min_pf": float(min_pf),
            "max_mdd_trade_based": max_mdd_trade_based,
            "sl_apply_mode": self._get_sl_apply_mode(),
            "sl_strategy": self._get_sl_strategy(),
        }

        out_json_path = self._write_universe_json(universe_syms, out_path=universe_out_json, meta=meta)
        if out_json_path:
            logger.info(f"✅ universe.json saved: {out_json_path} | symbols={len(universe_syms)}")

    def run_universe_selection_backtest_sliding(
        self,
        top_n: int = 100,
        universe_size: int = 25,
        window_days: int = 30,
        num_windows: int = 3,
        warmup_bars: int = 200,
        # 윈도우별 pass 기준 (너무 빡세게 잡지 말고, 합산에서 거른다)
        min_trades_per_window: int = 6,
        min_pf_per_window: float = 1.20,
        max_mdd_trade_based: float = None,
        pass_required: int = 2,   # 3개 중 최소 2개 통과
        out_dir: str = "runs/sliding_30d_x3",
        universe_out_json: str = "universe.json",
    ):
        """
        30일×3 슬라이딩 윈도우 유니버스 선정:
        - 데이터/지표는 1회
        - 백테스트는 3회(기간만 다르게)
        - 윈도우별 성과 테이블을 합산(pass_count)하여 최종 유니버스 선정
        """
        # 1) 데이터: 캐시에 최소 90일(+버퍼)이 있어야 함.
        # 여기서는 prepare_data에 days를 강제하지 않고, 캐시 로딩 결과를 그대로 사용.
        self.prepare_data(symbols=None, top_n=int(top_n))
        self.rebuild_indicators()

        if not self.data_map:
            logger.error("❌ Sliding universe selection aborted: No data_map.")
            return

        # 2) 공통 종료시점(end_time) 계산
        end_time = self._get_common_data_end_time()
        if end_time is None:
            logger.error("❌ Cannot infer common end_time.")
            return

        try:
            wd = int(window_days)
            nw = int(num_windows)
            if wd <= 0 or nw <= 0:
                logger.error("❌ Invalid window config.")
                return
        except Exception:
            logger.error("❌ Invalid window config.")
            return

        # 3) 윈도우 구간 계산 (가장 최근이 W1)
        #    W1: (end-30d, end], W2: (end-60d, end-30d], W3: (end-90d, end-60d]
        windows = []
        for i in range(nw):
            w_end = end_time - pd.Timedelta(days=wd * i)
            w_start = w_end - pd.Timedelta(days=wd)
            windows.append((w_start, w_end))
        # windows[0]이 가장 최근

        # 4) 윈도우별 실행 + 성과 추출
        try:
            if not os.path.isabs(out_dir):
                out_dir_abs = os.path.join(self.root_dir, out_dir)
            else:
                out_dir_abs = out_dir
            os.makedirs(out_dir_abs, exist_ok=True)
        except Exception:
            out_dir_abs = os.path.join(self.root_dir, "runs", "sliding_30d_x3")
            os.makedirs(out_dir_abs, exist_ok=True)

        # 누적 집계 구조
        agg = {}  # sym -> dict(pass_count, sum_trades, sum_pnl, pf_list, mdd_max)
        per_window_perf_paths = []

        for idx, (w_start, w_end) in enumerate(windows, start=1):
            w_tag = f"w{idx}"
            w_dir = os.path.join(out_dir_abs, w_tag)
            os.makedirs(w_dir, exist_ok=True)

            # 윈도우별 로그파일 초기화
            log_path = os.path.join(w_dir, "backtest_history.csv")
            self._reset_log_file(log_path)

            # 실행
            self.run_window(start_time=w_start, end_time=w_end, show_report=False, warmup_bars=warmup_bars)

            # 성과 테이블
            perf = self._build_symbol_performance_table_from_csv(csv_path=log_path)
            if perf is None or perf.empty:
                logger.warning(f"⚠️ {w_tag}: perf empty.")
                continue

            # 저장
            perf_path = os.path.join(w_dir, "symbol_performance.csv")
            try:
                perf.to_csv(perf_path, index=False, encoding="utf-8-sig")
                per_window_perf_paths.append(perf_path)
            except Exception:
                pass

            # 윈도우 pass 판정
            p = perf.copy()
            try:
                p["trades"] = pd.to_numeric(p["trades"], errors="coerce").fillna(0).astype(int)
                p["pf"] = pd.to_numeric(p["pf"], errors="coerce").fillna(0.0).astype(float)
                p["total_pnl"] = pd.to_numeric(p["total_pnl"], errors="coerce").fillna(0.0).astype(float)
                p["mdd_trade_based"] = pd.to_numeric(p["mdd_trade_based"], errors="coerce").fillna(0.0).astype(float)
            except Exception:
                pass

            passed = p[(p["trades"] >= int(min_trades_per_window)) & (p["pf"] >= float(min_pf_per_window))].copy()
            if max_mdd_trade_based is not None:
                passed = passed[passed["mdd_trade_based"] <= float(max_mdd_trade_based)]

            passed_syms = set([str(x).strip() for x in passed["symbol"].tolist() if str(x).strip()])

            # 누적 집계 갱신 (pass_count + 보조통계)
            for _, r in p.iterrows():
                sym = str(r.get("symbol", "")).strip()
                if not sym:
                    continue

                trades = int(r.get("trades", 0))
                pf = float(r.get("pf", 0.0))
                pnl = float(r.get("total_pnl", 0.0))
                mdd = float(r.get("mdd_trade_based", 0.0))

                st = agg.get(sym)
                if st is None:
                    st = {
                        "pass_count": 0,
                        "sum_trades": 0,
                        "sum_pnl": 0.0,
                        "pf_list": [],
                        "mdd_max": 0.0,
                    }

                if sym in passed_syms:
                    st["pass_count"] += 1

                st["sum_trades"] += trades
                st["sum_pnl"] += pnl
                st["pf_list"].append(pf)
                if mdd > st["mdd_max"]:
                    st["mdd_max"] = mdd

                agg[sym] = st

        if not agg:
            logger.error("❌ Sliding selection failed: agg empty.")
            return

        # 5) 최종 후보 선정: pass_count 우선 + 보조 스코어
        rows = []
        for sym, st in agg.items():
            pc = int(st.get("pass_count", 0))
            pf_list = st.get("pf_list", []) or []
            avg_pf = float(np.mean(pf_list)) if len(pf_list) else 0.0
            min_pf = float(np.min(pf_list)) if len(pf_list) else 0.0
            rows.append({
                "symbol": sym,
                "pass_count": pc,
                "avg_pf": avg_pf,
                "min_pf": min_pf,
                "sum_trades": int(st.get("sum_trades", 0)),
                "sum_pnl": float(st.get("sum_pnl", 0.0)),
                "mdd_max": float(st.get("mdd_max", 0.0)),
            })

        df_agg = pd.DataFrame(rows)
        if df_agg.empty:
            logger.error("❌ Sliding selection failed: df_agg empty.")
            return

        # pass_required 적용
        df_agg = df_agg[df_agg["pass_count"] >= int(pass_required)]
        if df_agg.empty:
            logger.error("❌ No symbols satisfy pass_required.")
            return

        # 스코어: pass_count 최우선 + (avg_pf / (1+mdd)) * log(1+sum_trades)
        try:
            df_agg["score"] = (
                df_agg["pass_count"] * 10.0
                + (df_agg["avg_pf"] * (1.0 / (1.0 + df_agg["mdd_max"])) * np.log1p(df_agg["sum_trades"]))
            )
        except Exception:
            df_agg["score"] = df_agg["pass_count"] * 10.0 + df_agg["avg_pf"]

        df_agg = df_agg.sort_values(["score", "pass_count", "avg_pf", "sum_pnl"], ascending=[False, False, False, False])

        # 상위 universe_size
        try:
            n = int(universe_size)
            if n <= 0:
                n = 25
        except Exception:
            n = 25

        selected = []
        for s in df_agg["symbol"].tolist():
            ss = str(s).strip()
            if ss and ss not in selected:
                selected.append(ss)
            if len(selected) >= n:
                break

        # 6) 메타 + 저장
        meta = {
            "mode": "sliding_30d_x3",
            "top_n": int(top_n),
            "universe_size": int(universe_size),
            "window_days": int(window_days),
            "num_windows": int(num_windows),
            "warmup_bars": int(warmup_bars),
            "min_trades_per_window": int(min_trades_per_window),
            "min_pf_per_window": float(min_pf_per_window),
            "max_mdd_trade_based": max_mdd_trade_based,
            "pass_required": int(pass_required),
            "end_time": str(end_time),
            "windows": [{"start": str(ws), "end": str(we)} for (ws, we) in windows],
            "sl_apply_mode": self._get_sl_apply_mode(),
            "sl_strategy": self._get_sl_strategy(),
        }

        # 집계표 저장 (디버그/감사용)
        try:
            agg_path = os.path.join(out_dir_abs, "aggregate_scores.csv")
            df_agg.to_csv(agg_path, index=False, encoding="utf-8-sig")
            logger.info(f"📌 Sliding aggregate saved: {agg_path}")
        except Exception:
            pass

        out_json_path = self._write_universe_json(selected, out_path=universe_out_json, meta=meta)
        if out_json_path:
            logger.info(f"✅ sliding universe.json saved: {out_json_path} | symbols={len(selected)}")


    def _process_entry(self, sym, signal, sl, curr_row, current_prices: dict):
        # ✅ 0) signal 정규화 (필수)
        sig = str(signal).strip().upper()
        alias = {
            "BUY": "LONG", "LONG": "LONG", "BULL": "LONG",
            "SELL": "SHORT", "SHORT": "SHORT", "BEAR": "SHORT",
        }
        sig = alias.get(sig, sig)
        if sig not in ("LONG", "SHORT"):
            return

        atr = curr_row.get('atr', curr_row['close'] * 0.01)
        slippage = atr * SLIPPAGE_ATR_FACTOR
        entry_price = curr_row['close'] + slippage if sig == 'LONG' else curr_row['close'] - slippage

        # ✅ sizing 기준: equity로 통일
        self._sync_equity(current_prices)
        current_equity = float(getattr(self.executor, "equity", self.executor.cash))

        amount = self.risk_ctrl.calculate_entry_size(sym, entry_price, current_equity, sl, sig)

        if amount > 0:
            notional_value = amount * entry_price
            leverage = self.cfg.get('risk_settings', {}).get('leverage', 1)
            margin_required = notional_value / leverage
            fee = notional_value * BASE_FEE

            if self.executor.cash >= margin_required + fee:
                self.executor.cash -= (margin_required + fee)
                self.executor.positions[sym] = {
                    'side': sig,
                    'amount': float(amount),
                    'entry_price': float(entry_price),
                    'leverage': float(leverage),
                    'margin': float(margin_required),
                    'sl': float(sl) if sl is not None else None,
                    'next_sl': None,
                    'trail_sl': None,
                    'entry_time': curr_row.name
                    
                }

                # ✅ ENTRY 직후에도 전체 가격으로 MTM 재평가
                prices2 = dict(current_prices)
                prices2[sym] = float(entry_price)
                self._sync_equity(prices2)

                self._log_csv(curr_row.name, sym, sig, 'ENTRY', entry_price, amount, 0.0, 'Signal Entry')

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

        market_data = {
            "close": close,
            "high": high,
            "low": low,
            "atr": atr,
            "st_val": st_val,
            "adx": adx,
            "df": hist_df,
        }

        # -----------------------------
        # 4) PositionMonitor 호출
        # -----------------------------
        action, exec_price, reason, new_sl = self.monitor.check_conditions(
            sym,
            pos,
            market_data,
            sl_apply_mode=mode,
            sl_strategy=sl_strategy,
            sl_params=sl_params,
        )

        # -----------------------------
        # 5) SL 갱신 + trail_sl 저장/갱신 (핵심 수정)
        # -----------------------------
        if action == "UPDATE_SL":
            try:
                if new_sl is not None:
                    new_sl_f = float(new_sl)
                    if (not math.isfinite(new_sl_f)) or (new_sl_f <= 0):
                        new_sl_f = None

                    if new_sl_f is not None:
                        # 기존 SL
                        cur_sl_raw = pos.get("sl", None)
                        try:
                            cur_sl_f = float(cur_sl_raw) if cur_sl_raw is not None else None
                            if cur_sl_f is not None and (not math.isfinite(cur_sl_f)):
                                cur_sl_f = None
                        except Exception:
                            cur_sl_f = None

                        # 기존 trail_sl
                        trail_raw = pos.get("trail_sl", None)
                        try:
                            trail_f = float(trail_raw) if trail_raw is not None else None
                            if trail_f is not None and (not math.isfinite(trail_f)):
                                trail_f = None
                        except Exception:
                            trail_f = None

                        # 5-1) mode별 SL 반영(기존 정책 유지)
                        if mode == "next":
                            # next: next_sl에 저장
                            if (cur_sl_f is None) or (new_sl_f != cur_sl_f):
                                pos["next_sl"] = new_sl_f
                        else:
                            # same: 즉시 sl 반영 + next_sl 제거
                            if (cur_sl_f is None) or (new_sl_f != cur_sl_f):
                                pos["sl"] = new_sl_f
                            if "next_sl" in pos:
                                pos.pop("next_sl", None)

                        # 5-2) trail_sl 저장/갱신 규칙
                        # - "전략이 계산한 신규 SL(new_sl_f)" 자체를 trail_sl로 기록 (관측/디버그/정합성)
                        # - tighten 방향만 반영 (LONG: 증가, SHORT: 감소)
                        side = str(pos.get("side", "")).upper().strip()
                        if side == "LONG":
                            if (trail_f is None) or (new_sl_f > trail_f):
                                pos["trail_sl"] = float(new_sl_f)
                        elif side == "SHORT":
                            if (trail_f is None) or (new_sl_f < trail_f):
                                pos["trail_sl"] = float(new_sl_f)
                        else:
                            # side 이상 시에도 기록은 하되 덮어쓰기 최소화
                            if trail_f is None:
                                pos["trail_sl"] = float(new_sl_f)

            except Exception:
                pass

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


if __name__ == "__main__":
    engine = BacktestEngine(days=30)

    engine.run_universe_selection_backtest_sliding(
        top_n=100,
        universe_size=25,
        window_days=30,
        num_windows=3,
        warmup_bars=200,
        min_trades_per_window=6,
        min_pf_per_window=1.20,
        max_mdd_trade_based=None,
        pass_required=2,
        out_dir="runs/sliding_30d_x3",
        universe_out_json="universe.json",
    )
