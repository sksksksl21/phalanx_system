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
    from execution.virtual_executor import VirtualExecutor
except ImportError as e:
    logger.critical(f"❌ Module Import Error: {e}")
    sys.exit(1)


class BacktestEngine:
    """
    [Phalanx Core Module] Backtest Orchestrator
    Mode: Time-Sequential & Priority-Based (Deterministic)

    [IMPORTANT FOR OPTIMIZATION]
    - raw OHLCV는 한 번만 준비/캐시 (self.raw_data_map)
    - trial마다 params가 바뀌면 반드시 지표를 재주입해야 함 (rebuild_indicators)
    """

    # ✅ 기본 30일
    def __init__(self, days=30):
        self.root_dir = root_dir
        self.config_path = os.path.join(root_dir, "config.json")
        self.cfg = self._load_config()

        self.cooldowns = {}
        self.consecutive_losses = {}
        self.test_days = days

        # 1. 전략 및 실행기 초기화
        self.titan = TitanStrategy()
        self.executor = VirtualExecutor()

        # [Safety] 초기 자본금 강제 주입
        if not hasattr(self.executor, 'initial_balance'):
            self.executor.initial_balance = 10000.0
            self.executor.cash = 10000.0
            self.executor.equity = 10000.0

        self.risk_ctrl = RiskControl(self.executor, self.cfg)
        self.monitor = PositionMonitor()

        # ---------------------------------------------------------
        # [RAW / PROCESSED 분리]
        # - raw_data_map: 지표 없는 순수 OHLCV (캐시 대상)
        # - data_map: 지표 포함 (trial마다 재생성 대상)
        # ---------------------------------------------------------
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

    def _load_config(self):
        if not os.path.exists(self.config_path):
            return {}
        with open(self.config_path, 'r', encoding="utf-8") as f:
            return json.load(f)

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
        elif hasattr(self, 'symbols') and self.symbols:
            targets = self.symbols
        else:
            targets = self.executor.get_top_targets()

        # ---------------------------------------------------------
        # 다운로드 대상에서 블랙리스트 제거
        # ---------------------------------------------------------
        filtered_targets = []
        for sym in targets:
            clean_sym = sym.split(':')[0]  # 'ETH/USDT:USDT' -> 'ETH/USDT'
            if clean_sym in self.titan.blacklist or sym in self.titan.blacklist:
                continue
            filtered_targets.append(sym)

        targets = filtered_targets

        raw_data_map = self.executor.prepare_data(targets)

        if not raw_data_map:
            logger.error("❌ No Data Loaded.")
            return {}

        # 순서 보장을 위해 정렬된 키 사용 (Deterministic)
        sorted_symbols = sorted(list(raw_data_map.keys()))
        self.raw_data_map = {sym: raw_data_map[sym] for sym in sorted_symbols}

        logger.info(f"✅ Raw Data Ready: {len(self.raw_data_map)} symbols loaded.")
        self.symbols = list(self.raw_data_map.keys())
        return self.raw_data_map

    def rebuild_indicators(self):
        """
        [Fix 핵심]
        - dropna()는 '필수 컬럼 subset' 기준으로만 적용한다.
        - 워밍업(warmup)은 각 필수 컬럼의 첫 유효시점 중 '가장 늦은 시점'을 사용한다.
        - 심볼별 warmup으로 슬라이스 (global_warmup 제거)
        """
        self.data_map = {}

        # 엔진이 최소로 필요로 하는 "필수 지표" (일봉 ema_daily는 제외)
        required_cols = [
            "open", "high", "low", "close", "volume",
            "atr", "vol_ma", "ema_intra", "rsi", "adx", "st_val", "st_dir"
        ]

        temp_map = {}
        warmup_map = {}  # sym -> sym_warmup

        # 1) 심볼별 지표 계산 + 심볼별 warmup 계산
        for sym, df in self.raw_data_map.items():
            try:
                ind = self.titan.calculate_indicators(sym, df.copy())

                missing = [c for c in required_cols if c not in ind.columns]
                if missing:
                    logger.warning(f"[Indicator] {sym} missing required cols: {missing}")
                    continue

                if ind[required_cols].isna().all().all():
                    logger.warning(f"[Indicator] {sym} required columns all-NaN (pre-slice).")
                    continue

                bad_cols = [c for c in required_cols if ind[c].isna().all()]
                if bad_cols:
                    logger.warning(f"[Indicator] {sym} has all-NaN required cols (pre-slice): {bad_cols}")
                    continue

                col_warmups = []
                for c in required_cols:
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

        # 2) 심볼별 warmup 이후로 슬라이스 + required subset 기준 dropna
        for sym, ind in temp_map.items():
            try:
                sym_warmup = int(warmup_map.get(sym, 0))
                sliced = ind.iloc[sym_warmup:].copy()

                if len(sliced) == 0:
                    logger.warning(f"[Indicator] {sym} empty after slice")
                    continue

                # ✅ dropna는 required subset에만 적용 (옵션 컬럼 NaN은 허용)
                sliced = sliced.dropna(subset=required_cols)
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
    # 2. Simulation Loop (Priority Based)
    # =========================================================
    def run(self, show_report=False):
        """
        - optimize에서는 show_report=False (매 trial마다 출력 폭발 방지)
        - 단독 backtest 실행에서는 show_report=True 권장
        """
        if not self.data_map:
            if not self.raw_data_map:
                self.prepare_data()
            self.rebuild_indicators()

        if not self.data_map:
            logger.error("❌ Cannot start backtest: No Data.")
            return

        fixed_symbols = sorted(list(self.data_map.keys()))

        # Timeline 동기화 (intersection)
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
            if common_timeline is None:
                common_timeline = idx
            else:
                common_timeline = common_timeline.intersection(idx)

        if common_timeline is None or len(common_timeline) < 200:
            logger.error("Not enough synchronized data (common timeline too short). Fallback to base timeline.")

            # ✅ Fallback: max_start 이후 데이터가 가장 긴 심볼을 선택
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

            # ✅ 중요: fallback에서는 max_start로 또 자르면 타임라인이 과도하게 줄 수 있으므로 전체 index를 사용
            timeline = self.data_map[base_sym].index
        else:
            timeline = common_timeline

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

        for current_time in sim_times:
            # Step 1: 포지션 관리
            current_prices = {}
            for sym in fixed_symbols:
                df = self.data_map[sym]
                if current_time not in df.index:
                    continue

                curr_row = df.loc[current_time]
                current_prices[sym] = curr_row['close']

                if sym in self.executor.positions:
                    # ✅ PATCH: 다음 캔들부터 SL 적용 (캔들 내 순서 모순 제거)
                    pos = self.executor.positions[sym]
                    if 'next_sl' in pos and pos['next_sl'] is not None:
                        try:
                            if float(pos['next_sl']) != float(pos.get('sl', 0)):
                                pos['sl'] = float(pos['next_sl'])
                        except Exception:
                            # 숫자 변환 실패 시 안전하게 스킵
                            pass
                        # 적용 후 제거 (단일 진실 유지)
                        pos.pop('next_sl', None)

                    self._process_existing_position(sym, curr_row, None)

            # Step 2: 신규 진입 후보
            candidates = []
            for sym in fixed_symbols:
                if sym in self.executor.positions:
                    continue

                df = self.data_map[sym]
                if current_time not in df.index:
                    continue

                curr_row = df.loc[current_time]
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

                signal, sl_price, tp_price = self.titan.analyze(sym, past_data)

                if signal:
                    score = float(curr_row.get('adx', 0))
                    candidates.append({
                        'score': score, 'sym': sym, 'signal': signal,
                        'sl': sl_price, 'tp': tp_price, 'row': curr_row
                    })

            # Step 3: 정렬
            candidates.sort(key=lambda x: x['score'], reverse=True)

            # Step 4: 진입
            for cand in candidates:
                if len(self.executor.positions) >= self.executor.MAX_POSITIONS:
                    break
                self._process_entry(cand['sym'], cand['signal'], cand['sl'], cand['tp'], cand['row'])

            # MTM
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
        current_equity = self.executor.cash

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
                    'entry_time': curr_row.name
                }
                self._log_csv(curr_row.name, sym, signal, 'ENTRY', entry_price, amount, 0, self.executor.equity, 'Signal Entry')

    def _process_existing_position(self, sym, curr_row, new_signal):
        pos = self.executor.positions[sym]
        market_data = {
            'close': curr_row['close'],
            'high': curr_row['high'],
            'low': curr_row['low'],
            'atr': curr_row.get('atr', curr_row['close'] * 0.01),
            'st_val': curr_row.get('st_val', 0)
        }

        action, exec_price, reason, new_sl = self.monitor.check_conditions(sym, pos, market_data)

        # ✅ PATCH: SL 갱신은 "다음 캔들"에만 적용한다.
        # - PositionMonitor는 new_sl을 계산하지만, 이번 캔들 히트 판정은 current_sl로 이미 끝났다는 전제
        # - 따라서 new_sl은 next_sl로 예약만 한다.
        if action == 'UPDATE_SL':
            try:
                if new_sl is not None and float(new_sl) != float(pos.get('sl', 0)):
                    pos['next_sl'] = float(new_sl)
            except Exception:
                # 숫자 변환 실패 시 예약하지 않음
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
        amount = pos['amount']
        pnl = (price - pos['entry_price']) * amount if pos['side'] == 'LONG' else (pos['entry_price'] - price) * amount

        exit_value = price * amount
        fee = exit_value * BASE_FEE

        margin_locked = pos['margin']
        self.executor.cash += margin_locked + pnl - fee

        net_pnl = pnl - fee

        if net_pnl > 0:
            self.consecutive_losses[sym] = 0
            self.cooldowns[sym] = dt
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

            self.cooldowns[sym] = dt + pd.Timedelta(hours=wait_hours)

        self._log_csv(dt, sym, pos['side'], 'EXIT', price, amount, net_pnl, self.executor.equity, reason)
        del self.executor.positions[sym]
        self.executor.history.append({'dt': dt, 'sym': sym, 'type': 'EXIT', 'pnl': net_pnl, 'reason': reason})


if __name__ == "__main__":
    # ✅ 단독 실행도 30일 기본
    engine = BacktestEngine(days=30)
    engine.prepare_data()
    engine.rebuild_indicators()
    engine.run(show_report=True)
