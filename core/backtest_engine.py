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
    
    def __init__(self, days=180):
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
        # [CRITICAL FIX] 다운로드 대상에서 블랙리스트 제거
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
        trial마다 반드시 호출해야 하는 핵심 함수
        - 현재 titan.params 기준으로 모든 심볼 DF에 지표를 재주입하고 self.data_map을 재구성한다.
        """
        if not self.raw_data_map:
            self.prepare_data()

        if not self.raw_data_map:
            logger.error("❌ rebuild_indicators failed: raw_data_map is empty.")
            self.data_map = {}
            return {}

        logger.info("⚙️ [Engine] Rebuilding Indicators (Trial-Aware)...")
        rebuilt = {}

        for sym in sorted(self.raw_data_map.keys()):
            df_raw = self.raw_data_map[sym]
            try:
                processed_df = self.titan.calculate_indicators(sym, df_raw)

                # 지표 계산 결과 정리
                processed_df.dropna(inplace=True)

                if not processed_df.empty:
                    rebuilt[sym] = processed_df
            except Exception as e:
                logger.error(f"❌ Indicator Rebuild Error {sym}: {e}")

        self.data_map = rebuilt
        self.symbols = list(self.data_map.keys())
        logger.info(f"✅ Indicators Ready: {len(self.data_map)} symbols processed.")
        return self.data_map

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

        # ---------------------------
        # [Fix #2] Timeline 동기화 강화:
        # - 모든 심볼의 "공통 인덱스(intersection)"로 timeline 구성
        # ---------------------------
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
            logger.error("Not enough synchronized data (common timeline too short).")
            return

        timeline = common_timeline

        # [Reset] 계좌 및 상태 초기화
        self.executor.history = []
        self.executor.cash = self.executor.initial_balance
        self.executor.equity = self.executor.initial_balance
        self.executor.positions = {}
        self.executor.equity_curve = []
        self.cooldowns = {}
        self.consecutive_losses = {}

        sim_times = timeline[200:]

        # 🕒 Main Time Loop
        for current_time in sim_times:

            # [Step 1] 현재가 업데이트 및 기존 포지션 관리 (청산 우선)
            current_prices = {}
            for sym in fixed_symbols:
                df = self.data_map[sym]
                if current_time not in df.index:
                    continue

                curr_row = df.loc[current_time]
                current_prices[sym] = curr_row['close']

                # 보유 중인 포지션 관리
                if sym in self.executor.positions:
                    self._process_existing_position(sym, curr_row, None)

            # [Step 2] 신규 진입 후보군 탐색
            candidates = []
            for sym in fixed_symbols:
                if sym in self.executor.positions:
                    continue

                df = self.data_map[sym]
                if current_time not in df.index:
                    continue

                # ---------------------------
                # [Fix #1] curr_row 오염 제거:
                # Step2에서 심볼별로 curr_row를 반드시 다시 잡는다
                # ---------------------------
                curr_row = df.loc[current_time]

                clean_sym = sym.split(':')[0]

                # 쿨다운 & 블랙리스트 체크
                if sym in self.cooldowns:
                    if current_time < self.cooldowns[sym]:
                        continue
                    else:
                        del self.cooldowns[sym]

                if clean_sym in self.titan.blacklist:
                    continue

                # 전략 분석: 과거 250봉을 슬라이딩 윈도우로 사용
                curr_idx_int = df.index.get_loc(current_time)
                start_idx = max(0, curr_idx_int - 250)
                past_data = df.iloc[start_idx: curr_idx_int + 1]

                signal, sl_price, tp_price = self.titan.analyze(sym, past_data)

                if signal:
                    # [핵심] 우선순위 점수: "그 심볼의 그 시점 row"에서만 뽑는다
                    score = float(curr_row.get('adx', 0))
                    candidates.append({
                        'score': score, 'sym': sym, 'signal': signal,
                        'sl': sl_price, 'tp': tp_price, 'row': curr_row
                    })

            # [Step 3] 우선순위 정렬 (Ranking)
            candidates.sort(key=lambda x: x['score'], reverse=True)

            # [Step 4] 순차 진입 (Execution)
            for cand in candidates:
                if len(self.executor.positions) >= self.executor.MAX_POSITIONS:
                    break
                self._process_entry(cand['sym'], cand['signal'], cand['sl'], cand['tp'], cand['row'])

            # Equity Curve 업데이트 (MTM)
            self.executor.update_equity(current_prices)
            self.executor.equity_curve.append({'dt': current_time, 'equity': self.executor.equity})

        # MTM Equity Curve Save (for dashboard)
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

        # 리포트 출력(원하면)
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

        if new_sl != pos['sl']:
            pos['sl'] = new_sl

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
        if pos['side'] == 'LONG':
            pnl = (price - pos['entry_price']) * amount
        else:
            pnl = (pos['entry_price'] - price) * amount

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
    engine = BacktestEngine(days=90)

    # 단독 실행 시에는 리포트 출력이 기본적으로 필요함
    engine.prepare_data()
    engine.rebuild_indicators()
    engine.run(show_report=True)
