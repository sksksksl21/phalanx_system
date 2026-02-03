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
    def prepare_data(self, symbols=None):
        """
        - raw_data_map이 이미 있으면 재다운로드하지 않는다.
        - 다만, 지표(data_map)는 trial마다 바뀔 수 있으므로 rebuild_indicators()로 재생성한다.
        """
        if self.raw_data_map and len(self.raw_data_map) > 0:
            return self.raw_data_map

        logger.info("📥 [Data Loader] Fetching Historical Data...")

        # 외부 주입 심볼 우선순위 처리
        if symbols:
            targets = symbols

        # ✅ UF JSON 우선 (enabled일 때만)
        elif self._is_uf_enabled():
            targets = self._get_universe_from_json()

            # UF 실패/비어있으면 안전 폴백
            if not targets:
                targets = self.executor.get_top_targets()

        elif hasattr(self, 'symbols') and self.symbols:
            targets = self.symbols

        else:
            targets = self.executor.get_top_targets()

        # 블랙리스트 제거
        filtered_targets = []
        for sym in targets:
            clean_sym = sym.split(':')[0]
            if clean_sym in self.titan.blacklist or sym in self.titan.blacklist:
                continue
            filtered_targets.append(sym)

        targets = filtered_targets

        raw_data_map = self.executor.prepare_data(targets)

        if not raw_data_map:
            logger.error("❌ No Data Loaded.")
            return {}

        sorted_symbols = sorted(list(raw_data_map.keys()))
        self.raw_data_map = {sym: raw_data_map[sym] for sym in sorted_symbols}

        logger.info(f"✅ Raw Data Ready: {len(self.raw_data_map)} symbols loaded.")
        self.symbols = list(self.raw_data_map.keys())
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

        for current_time in sim_times:
            current_rows = {}
            current_prices = {}

            for sym in fixed_symbols:
                row = self.data_map[sym].loc[current_time]
                current_rows[sym] = row
                current_prices[sym] = float(row["close"])

            self.last_prices = dict(current_prices)

            # ✅ sizing 기준 통일: 먼저 equity 업데이트(강제 MTM)
            self._sync_equity(current_prices)

            # Step 1: 포지션 관리
            for sym in fixed_symbols:
                if sym not in self.executor.positions:
                    continue

                curr_row = current_rows[sym]
                pos = self.executor.positions[sym]

                # ✅ next_sl 승계는 "next 모드"에서만
                if apply_mode == "next":
                    if 'next_sl' in pos and pos['next_sl'] is not None:
                        try:
                            if float(pos['next_sl']) != float(pos.get('sl', 0)):
                                pos['sl'] = float(pos['next_sl'])
                        except Exception:
                            pass
                        pos.pop('next_sl', None)
                else:
                    # same 모드에서는 next_sl 혼입 방지
                    if 'next_sl' in pos and pos['next_sl'] is not None:
                        pos.pop('next_sl', None)

                # ✅ apply_mode 전달
                self._process_existing_position(sym, curr_row, None, apply_mode=apply_mode)

            # Step 2: 신규 진입 후보
            candidates = []
            for sym in fixed_symbols:
                if sym in self.executor.positions:
                    continue

                curr_row = current_rows[sym]
                df = self.data_map[sym]
                clean_sym = sym.split(':')[0]

                # 쿨다운
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

                signal, sl_price, _tp_price = self.titan.analyze(sym, past_data)  # tp는 무시

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

            # MTM 최종 업데이트 (강제 MTM)
            self._sync_equity(current_prices)
            self.executor.equity_curve.append({'dt': current_time, 'equity': float(self.executor.equity)})

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
    engine.prepare_data()
    engine.rebuild_indicators()
    engine.run(show_report=True)
