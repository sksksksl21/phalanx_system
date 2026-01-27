# core/backtest_engine_B.py

import sys
import os
import json
import logging
import pandas as pd
import numpy as np

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
    # ✅ B는 VirtualExecutorB를 써야 1m exit 계약이 성립한다.
    from execution.virtual_executor_B import VirtualExecutorB
except ImportError as e:
    logger.critical(f"❌ Module Import Error: {e}")
    sys.exit(1)


class BacktestEngine:
    """
    [Phalanx Core Module] Backtest Orchestrator (B)
    - 15m entry
    - 1m exit-only touch check (SL/TP)
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
        self.executor = VirtualExecutorB()

        # [Safety] 초기 자본금 강제 주입
        if not hasattr(self.executor, 'initial_balance'):
            self.executor.initial_balance = 10000.0
        # 초기 상태도 강제 정합
        self.executor.cash = float(getattr(self.executor, "cash", self.executor.initial_balance))
        self.executor.equity = float(getattr(self.executor, "equity", self.executor.initial_balance))

        self.risk_ctrl = RiskControl(self.executor, self.cfg)
        self.monitor = PositionMonitor()

        # RAW / PROCESSED
        self.raw_data_map = {}   # {sym: df_raw}
        self.data_map = {}       # {sym: df_with_indicators}

        self.symbols = []  # 외부 주입 심볼 저장소

        # Config 강제 주입
        strat_settings = self.cfg.get('strategy_settings', {})
        if 'blacklist' in strat_settings:
            self.titan.blacklist = set(strat_settings['blacklist'])
            print(f"🚫 [System] Configured Blacklist: {self.titan.blacklist}")

        # 로그 파일 초기화
        self.log_file = os.path.join(root_dir, "backtest_history.csv")
        with open(self.log_file, 'w', encoding="utf-8") as f:
            f.write("Datetime,Symbol,Side,Type,Price,Amount,PnL,Balance,Reason\n")

        # MTM Equity Curve CSV 경로 (대시보드용)
        self.equity_curve_file = os.path.join(root_dir, "backtest_equity_curve.csv")

        # (B) required cols
        self.required_cols = [
            "open", "high", "low", "close", "volume",
            "atr", "vol_ma", "ema_intra", "rsi", "adx", "st_val", "st_dir"
        ]

    def _load_config(self):
        if not os.path.exists(self.config_path):
            return {}
        with open(self.config_path, 'r', encoding="utf-8") as f:
            return json.load(f)

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
    # 1. Data Preparation Layer
    # =========================================================
    def prepare_data(self, symbols=None):
        """
        - raw_data_map이 이미 있으면 재다운로드하지 않는다.
        - B: 15m raw 로딩 후, 1m 데이터도 (exit-only) 준비한다.
        """
        if self.raw_data_map and len(self.raw_data_map) > 0:
            return self.raw_data_map

        logger.info("📥 [Data Loader] Fetching Historical Data...")

        # 외부 주입 심볼 우선순위 처리
        if symbols:
            targets = symbols
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

        raw_data_map = self.executor.prepare_data(targets, days=self.test_days)

        if not raw_data_map:
            logger.error("❌ No Data Loaded.")
            return {}

        sorted_symbols = sorted(list(raw_data_map.keys()))
        self.raw_data_map = {sym: raw_data_map[sym] for sym in sorted_symbols}

        logger.info(f"✅ Raw Data Ready: {len(self.raw_data_map)} symbols loaded.")
        self.symbols = list(self.raw_data_map.keys())

        # ✅ (B 핵심) 1m 데이터 준비 (EXIT ONLY)
        try:
            logger.info("📥 [Data Loader] Fetching 1m Data (Exit Only)...")
            self.executor.prepare_data_1m(self.symbols, days=self.test_days)
        except Exception as e:
            # 1m이 일부 실패해도 15m 시뮬은 진행 가능 (해당 심볼은 1m exit 미적용)
            logger.error(f"❌ Failed to prepare 1m data (continue): {e}")

        return self.raw_data_map

    def rebuild_indicators(self):
        """
        - dropna()는 '필수 컬럼 subset' 기준으로만 적용한다.
        - 워밍업(warmup)은 각 필수 컬럼의 첫 유효시점 중 '가장 늦은 시점'을 사용한다.
        - 심볼별 warmup으로 슬라이스
        """
        self.data_map = {}

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

    def _log_csv(self, dt, sym, side, type_note, price, amt, pnl, balance, reason):
        line = f"{dt},{sym},{side},{type_note},{price},{amt},{pnl:.4f},{balance:.2f},{reason}\n"
        with open(self.log_file, 'a', encoding="utf-8") as f:
            f.write(line)

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

            # fallback base 심볼 선택
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

        # candidate 전체 구간 reindex/ffill 후 max_start부터 사용
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
        self.executor.cash = self.executor.initial_balance
        self.executor.equity = self.executor.initial_balance
        self.executor.positions = {}
        self.executor.equity_curve = []
        self.cooldowns = {}
        self.consecutive_losses = {}

        sim_times = timeline[200:]
        if len(sim_times) < 200:
            logger.error(f"❌ Not enough simulation steps after warmup: {len(sim_times)}")
            return

        # ✅ (B 핵심) 1m EXIT 중복스캔 방지용: 포지션별 마지막 체크 구간 관리
        # - entry_time부터 매번 current_time까지 스캔하면, 1) 느려지고 2) 겹친 구간 재스캔으로 이상해질 수 있음
        # - 그래서 pos['last_1m_check']를 둔다.
        for current_time in sim_times:
            current_rows = {}
            current_prices = {}
            for sym in fixed_symbols:
                row = self.data_map[sym].loc[current_time]
                current_rows[sym] = row
                current_prices[sym] = float(row["close"])

            # sizing 기준 통일: 먼저 equity 업데이트
            self.executor.update_equity(current_prices)

            # =====================================================
            # (B) Step 0: 1m EXIT 우선 적용 (SL/TP 터치)
            # =====================================================
            # positions dict를 직접 도는 중에 삭제되므로 list()로 복사해서 순회
            for sym in list(self.executor.positions.keys()):
                pos = self.executor.positions.get(sym)
                if not pos:
                    continue

                # 1m 데이터가 없는 심볼은 패스 (이 심볼은 15m 관리만)
                if not hasattr(self.executor, "check_exit_1m"):
                    break

                start_dt = pos.get("last_1m_check", pos.get("entry_time", None))
                if start_dt is None:
                    start_dt = current_time - self.bar_td  # fallback

                end_dt = current_time

                try:
                    hit = self.executor.check_exit_1m(sym, start_dt, end_dt)
                except Exception:
                    hit = None

                if hit:
                    # hit 계약: ("EXIT", ts, price, reason)
                    try:
                        _, ts, price, reason = hit
                    except Exception:
                        # 계약 깨진 경우는 그냥 무시 (터지지 않게)
                        continue

                    # 내부 _execute_exit로 처리해야 cooldown/log가 A와 동일하게 적용됨
                    pos_now = self.executor.positions.get(sym)
                    if pos_now:
                        self._execute_exit(sym, pos_now, float(price), str(reason), ts)

                    # 다음 심볼로
                    continue

                # 이번 구간까지 체크 완료 표시(중복 스캔 방지)
                pos2 = self.executor.positions.get(sym)
                if pos2:
                    pos2["last_1m_check"] = end_dt

            # Step 1: 포지션 관리 (15m PositionMonitor)
            for sym in fixed_symbols:
                if sym not in self.executor.positions:
                    continue

                curr_row = current_rows[sym]

                # 다음 캔들부터 SL 적용
                pos = self.executor.positions[sym]
                if 'next_sl' in pos and pos['next_sl'] is not None:
                    try:
                        if float(pos['next_sl']) != float(pos.get('sl', 0)):
                            pos['sl'] = float(pos['next_sl'])
                    except Exception:
                        pass
                    pos.pop('next_sl', None)

                self._process_existing_position(sym, curr_row, None)

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

                signal, sl_price, tp_price = self.titan.analyze(sym, past_data)

                if signal:
                    score = float(curr_row.get('adx', 0))
                    candidates.append({
                        'score': score, 'sym': sym, 'signal': signal,
                        'sl': sl_price, 'tp': tp_price, 'row': curr_row
                    })

            candidates.sort(key=lambda x: x['score'], reverse=True)

            for cand in candidates:
                if len(self.executor.positions) >= self.executor.MAX_POSITIONS:
                    break
                self._process_entry(cand['sym'], cand['signal'], cand['sl'], cand['tp'], cand['row'])

            # MTM 최종 업데이트
            self.executor.update_equity(current_prices)
            self.executor.equity_curve.append({'dt': current_time, 'equity': self.executor.equity})

        # Equity curve save
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

    def _process_entry(self, sym, signal, sl, tp, curr_row):
        atr = curr_row.get('atr', curr_row['close'] * 0.01)
        slippage = atr * SLIPPAGE_ATR_FACTOR

        entry_price = curr_row['close'] + slippage if signal == 'LONG' else curr_row['close'] - slippage

        # sizing 기준: cash → equity 로 통일
        current_equity = float(getattr(self.executor, "equity", self.executor.cash))

        amount = self.risk_ctrl.calculate_entry_size(sym, entry_price, current_equity, sl, signal)

        if amount > 0:
            notional_value = amount * entry_price
            leverage = self.cfg.get('risk_settings', {}).get('leverage', 1)
            margin_required = notional_value / leverage
            fee = notional_value * BASE_FEE

            if self.executor.cash >= margin_required + fee:
                self.executor.cash -= (margin_required + fee)
                self.executor.positions[sym] = {
                    'side': signal,
                    'amount': amount,
                    'entry_price': entry_price,
                    'leverage': leverage,
                    'margin': margin_required,
                    'sl': sl,
                    'tp1': tp,
                    'next_sl': None,
                    'tp1_hit': False,
                    'entry_time': curr_row.name,
                    # ✅ (B) 1m exit 체크 시작점
                    'last_1m_check': curr_row.name
                }
                self._log_csv(curr_row.name, sym, signal, 'ENTRY', entry_price, amount, 0, self.executor.equity, 'Signal Entry')

    def _process_existing_position(self, sym, curr_row, new_signal):
        # 15m 모니터링(기존대로)
        if sym not in self.executor.positions:
            return

        pos = self.executor.positions[sym]
        market_data = {
            'close': curr_row['close'],
            'high': curr_row['high'],
            'low': curr_row['low'],
            'atr': curr_row.get('atr', curr_row['close'] * 0.01),
            'st_val': curr_row.get('st_val', 0)
        }

        action, exec_price, reason, new_sl = self.monitor.check_conditions(sym, pos, market_data)

        # SL 갱신은 다음 캔들에만
        if action == 'UPDATE_SL':
            try:
                if new_sl is not None and float(new_sl) != float(pos.get('sl', 0)):
                    pos['next_sl'] = float(new_sl)
            except Exception:
                pass

        if action == 'TP1':
            close_amt = pos['amount'] * 0.5
            pnl = (exec_price - pos['entry_price']) * close_amt if pos['side'] == 'LONG' else (pos['entry_price'] - exec_price) * close_amt
            fee = exec_price * close_amt * BASE_FEE

            margin_release = pos['margin'] * 0.5
            pos['margin'] -= margin_release

            self.executor.cash += margin_release + pnl - fee
            pos['amount'] -= close_amt
            pos['tp1_hit'] = True

            self.executor.history.append({
                'dt': curr_row.name, 'sym': sym, 'type': 'TP1',
                'pnl': pnl - fee, 'reason': reason
            })

        elif action == 'EXIT':
            self._execute_exit(sym, pos, exec_price, reason, curr_row.name)
            return

        if new_signal and new_signal != pos['side']:
            atr = market_data['atr']
            slippage = atr * 0.01
            base = curr_row['close']
            flip_price = base - slippage if pos['side'] == 'LONG' else base + slippage
            self._execute_exit(sym, pos, flip_price, 'SIGNAL_FLIP', curr_row.name)

    def _execute_exit(self, sym, pos, price, reason, dt):
        # 포지션이 이미 지워졌을 수도 있음(1m exit 이후 15m 루프에서 다시 들어올 수 있으니 방어)
        if sym not in self.executor.positions:
            return

        amount = pos['amount']
        pnl = (price - pos['entry_price']) * amount if pos['side'] == 'LONG' else (pos['entry_price'] - price) * amount

        exit_value = price * amount
        fee = exit_value * BASE_FEE

        margin_locked = pos['margin']
        self.executor.cash += margin_locked + pnl - fee

        net_pnl = pnl - fee

        # 동일 캔들 재진입 금지: 최소 1캔들 쿨다운
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

        # equity는 루프에서 MTM으로 갱신되지만, 로그 시점 일관성 위해 최소 반영
        try:
            self.executor.equity = float(self.executor.cash)
        except Exception:
            pass

        self._log_csv(dt, sym, pos['side'], 'EXIT', price, amount, net_pnl, self.executor.equity, reason)

        try:
            del self.executor.positions[sym]
        except Exception:
            pass

        self.executor.history.append({'dt': dt, 'sym': sym, 'type': 'EXIT', 'pnl': net_pnl, 'reason': reason})


if __name__ == "__main__":
    engine = BacktestEngine(days=30)
    engine.prepare_data()
    engine.rebuild_indicators()
    engine.run(show_report=True)
