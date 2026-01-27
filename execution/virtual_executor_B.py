# execution/virtual_executor_B.py

import logging
import pandas as pd
import numpy as np
import os
import sys

# [PATH SETUP] HistoryManager 호출을 위한 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from utils.history_manager import HistoryManager

logger = logging.getLogger("PhalanxVirtual")

# [설정] 백테스트 환경 변수
INITIAL_EQUITY = 10000.0
BASE_FEE = 0.0005        # 0.05%
SLIPPAGE_ATR_FACTOR = 0.05


class VirtualExecutorB:
    """
    [Phalanx Execution Module - Virtual V3.0]
    Role: Accounting, Physics Simulation & Data Recording
    Features:
    - Full Integration with HistoryManager
    - Symbol-level Performance Tracking

    [B EXTENSION]
    - 1m data cache + 1m exit-only touch checker
    """

    def __init__(self):
        self.equity = INITIAL_EQUITY
        self.cash = INITIAL_EQUITY
        self.positions = {}
        self.history = []   # 인메모리 로그 (콘솔 리포트용)
        self.equity_curve = []
        self.MAX_POSITIONS = 5

        # [DATA DRIVEN] HistoryManager 초기화
        csv_path = os.path.join(root_dir, "backtest_history.csv")
        # 백테스트 시작 시 기존 기록 삭제 (Clean State)
        if os.path.exists(csv_path):
            os.remove(csv_path)
        self.hm = HistoryManager(csv_path)

        # ============================
        # [B EXTENSION] 1m cache
        # ============================
        self.data_1m = {}  # {symbol: df_1m}

    def fetch_balance(self):
        return {'USDT': {'free': self.cash, 'total': self.equity}}

    def amount_to_precision(self, symbol, amount):
        """거래소 수량 정밀도 모방"""
        if amount is None:
            return 0.0
        # [Patch] NaN 방어: 백테스트 중 계산에서 NaN이 흘러들면 포맷이 깨질 수 있음
        if isinstance(amount, float) and np.isnan(amount):
            return 0.0
        return float(f"{amount:.6f}")

    def price_to_precision(self, symbol, price):
        """거래소 가격 정밀도 모방"""
        if price is None:
            return 0.0
        # [Patch] NaN 방어
        if isinstance(price, float) and np.isnan(price):
            return 0.0
        return float(f"{price:.4f}")

    def update_equity(self, current_prices):
        """Mark-to-Market (시가 평가)"""
        # [Patch] current_prices가 None일 수 있는 상황 방어
        if current_prices is None:
            current_prices = {}

        temp_equity = self.cash

        for sym, pos in self.positions.items():
            curr_price = current_prices.get(sym, pos.get('entry_price', 0))

            # [Patch] 가격/수량 결측 방어 (실전/백테의 데이터 오염 대비)
            if curr_price is None or (isinstance(curr_price, float) and np.isnan(curr_price)):
                curr_price = pos.get('entry_price', 0)

            amt = pos.get('amount', 0)
            if amt is None or (isinstance(amt, float) and np.isnan(amt)):
                amt = 0

            entry = pos.get('entry_price', 0)
            if entry is None or (isinstance(entry, float) and np.isnan(entry)):
                entry = 0

            # 미실현 손익 계산
            if pos.get('side') == 'LONG':
                pnl = (curr_price - entry) * amt
            else:
                pnl = (entry - curr_price) * amt

            # 증거금 + 미실현 손익
            margin = pos.get('margin', 0)
            if margin is None or (isinstance(margin, float) and np.isnan(margin)):
                margin = 0

            temp_equity += (margin + pnl)

        self.equity = temp_equity
        return self.equity

    def get_top_targets(self):
        """
        거래대금 상위 종목을 뽑되,
        - 선물 USDT perpetual
        - markets에서 active 아닌 심볼은 제외(상폐/정지/비활성 방지)
        ※ 어떤 상황에서도 'None'을 반환하지 않고, 항상 list를 반환한다.
        """
        import ccxt

        exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 20000,  # ✅ 무한대기 방지
            'options': {'defaultType': 'future', 'adjustForTimeDifference': True}
        })

        # ✅ markets 로딩은 "시도"만 하고 실패해도 진행 (active filter는 가능한 범위에서만)
        try:
            exchange.load_markets()
        except Exception as e:
            print(f"    ⚠️ load_markets failed (continue without full market filter): {e}")

        # ✅ ticker 조회 실패 시도 대비: 무조건 list 반환
        try:
            tickers = exchange.fetch_tickers()
        except Exception as e:
            print(f"    ❌ fetch_tickers failed: {e}")
            return []  # <-- 핵심: None 반환 금지

        markets = getattr(exchange, "markets", None)
        if not isinstance(markets, dict):
            markets = {}

        targets = []
        for s, t in (tickers or {}).items():
            if '/USDT:USDT' not in s:
                continue

            # ✅ 상폐/정지/비활성 마켓 제외 (markets 정보가 있을 때만)
            m = markets.get(s)
            if isinstance(m, dict):
                if m.get('active') is False:
                    continue
                info = m.get('info')
                if isinstance(info, dict):
                    status = str(info.get('status', '')).upper()
                    # 바이낸스는 보통 TRADING이 정상
                    if status and status not in ('TRADING', '1'):
                        continue

            vol = 0.0
            if isinstance(t, dict):
                vol = t.get('quoteVolume', 0) or 0

            try:
                vol = float(vol)
            except Exception:
                vol = 0.0

            if vol > 0:
                targets.append((s, vol))

        targets.sort(key=lambda x: x[1], reverse=True)
        return [t[0] for t in targets][:20]  # <-- 항상 list

    def prepare_data(self, symbols, days=30):
        # (기존 데이터 다운로드 로직 유지)
        import ccxt
        import time
        import pandas as pd  # 데이터프레임 변환용 추가

        exchange = ccxt.binance({'options': {'defaultType': 'future'}})

        # ✅ markets 미로딩 상태면 active/status 필터가 아예 안 돌아가서 이상한 심볼로 터질 수 있음
        #    실패해도 진행 (이 함수도 절대 None을 반환하지 않음)
        try:
            exchange.load_markets()
        except Exception as e:
            print(f"    ⚠️ load_markets failed in prepare_data (continue): {e}")

        since = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
        data_map = {}
        min_required = 2000

        # ✅ symbols가 None이면 여기서 바로 죽는다 -> 방어
        if symbols is None:
            print("    ⚠️ symbols is None -> return empty data_map")
            return {}

        print(f"📥 [VirtualExecutor] Downloading data for {len(symbols)} symbols...")

        for sym in symbols:
            # ✅ 상폐/정지/비활성 마켓 제외 (단, markets가 없으면 필터 스킵)
            markets = getattr(exchange, "markets", None)
            if isinstance(markets, dict) and markets:
                m = markets.get(sym)
                if isinstance(m, dict):
                    if m.get('active') is False:
                        print(f"    ⚠️ Skipping inactive market: {sym}")
                        continue
                    info = m.get('info')
                    if isinstance(info, dict):
                        status = str(info.get('status', '')).upper()
                        if status and status not in ('TRADING', '1'):
                            print(f"    ⚠️ Skipping non-trading market ({status}): {sym}")
                            continue

            all_ohlcv = []
            temp_since = since

            try:
                print(f"  - Fetching {sym}...")  # 진행상황 표시

                max_retries = 3

                while True:
                    try:
                        # ✅ 요청이 오래 걸리면 timeout으로 튕기게 되고, 아래에서 재시도/스킵됨
                        ohlcv = exchange.fetch_ohlcv(sym, '15m', since=temp_since, limit=1000)
                    except (ccxt.RequestTimeout, ccxt.NetworkError) as e:
                        max_retries -= 1
                        print(f"    ⚠️ Network/Timeout on {sym}: {e} (retries left={max_retries})")
                        if max_retries <= 0:
                            print(f"    ❌ Giving up {sym} due to repeated timeouts.")
                            all_ohlcv = []
                            break
                        time.sleep(1.0)
                        continue
                    except Exception as e:
                        print(f"    ❌ Error fetching {sym}: {e}")
                        all_ohlcv = []
                        break

                    if not ohlcv:
                        break

                    last_ts = ohlcv[-1][0]
                    if all_ohlcv and last_ts <= all_ohlcv[-1][0]:
                        break

                    all_ohlcv.extend(ohlcv)
                    temp_since = last_ts + 1

                    if len(ohlcv) < 1000:
                        break

                    time.sleep(0.2)

                # 데이터프레임 변환 (필수)
                if len(all_ohlcv) >= min_required:
                    df = pd.DataFrame(
                        all_ohlcv,
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )

                    # [Patch] 타입 정리 (수치 계산 안정성)
                    for c in ['open', 'high', 'low', 'close', 'volume']:
                        df[c] = pd.to_numeric(df[c], errors='coerce')

                    # [Patch] timestamp 중복 제거 + 정렬
                    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

                    # 인덱스 설정
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('datetime', inplace=True)

                    # [Patch] 인덱스 단조증가 보장
                    if not df.index.is_monotonic_increasing:
                        df = df.sort_index()

                    # [Patch] NaN 라인 제거(OHLC 핵심 결측이면 사용 불가)
                    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

                    data_map[sym] = df
                else:
                    print(f"    ⚠️ Not enough data for {sym}: {len(all_ohlcv)} rows")

            except Exception as e:
                print(f"    ❌ Error fetching {sym}: {e}")
                time.sleep(5)  # 에러 나면 5초 쿨타임 (중요)

            # [수정 2] ★★★ 종목 바뀔 때 1초 휴식 (이게 제일 중요) ★★★
            time.sleep(1.0)

        return data_map

    # ==========================================
    # 4. Reporting (Data-Driven V3.0)
    # ==========================================
    def report(self):
        """
        [Phalanx Analytics] Detailed Performance Report
        """
        if not self.history:
            print("❌ No trades executed.")
            return

        df = pd.DataFrame(self.history)
        df_curve = pd.DataFrame(self.equity_curve)

        # --- 전체 지표 계산 ---
        total_trades = len(df)
        win_trades = df[df['pnl'] > 0] if 'pnl' in df.columns else pd.DataFrame()
        loss_trades = df[df['pnl'] <= 0] if 'pnl' in df.columns else pd.DataFrame()

        win_rate = (len(win_trades) / total_trades * 100) if total_trades > 0 else 0
        avg_win = win_trades['pnl'].mean() if not win_trades.empty else 0
        avg_loss = loss_trades['pnl'].mean() if not loss_trades.empty else 0

        # [Patch] Profit Factor 계산 안정화:
        if not win_trades.empty and not loss_trades.empty:
            loss_sum = loss_trades['pnl'].sum()
            win_sum = win_trades['pnl'].sum()
            profit_factor = abs(win_sum / loss_sum) if loss_sum != 0 else float('inf')
        else:
            profit_factor = 0.0 if (win_trades.empty and loss_trades.empty) else (float('inf') if loss_trades.empty else 0.0)

        mdd = 0
        total_return = 0
        if not df_curve.empty and 'equity' in df_curve.columns:
            df_curve['peak'] = df_curve['equity'].cummax()
            df_curve['dd'] = np.where(
                df_curve['peak'] > 0,
                (df_curve['equity'] - df_curve['peak']) / df_curve['peak'] * 100,
                0
            )
            mdd = float(df_curve['dd'].min())
            total_return = (self.equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100

        print("\n" + "=" * 65)
        print(f"📊 PHALANX SYSTEM BACKTEST REPORT (V3.0 Data-Driven)")
        print("=" * 65)
        print(f"💰 Final Equity  : ${self.equity:,.2f}")
        print(f"📈 Total Return  : {total_return:.2f}%")
        print(f"🌊 MDD           : {mdd:.2f}%")
        print(f"⚖️  Profit Factor : {profit_factor:.2f}")
        print(f"✅ Win Rate      : {win_rate:.2f}% ({len(win_trades)}W / {len(loss_trades)}L)")
        print(f"🟢 Avg Win       : ${avg_win:.2f}")
        print(f"🔴 Avg Loss      : ${avg_loss:.2f}")

        # [Patch] Risk:Reward 0-div 방어
        if avg_loss == 0:
            rr = float('inf') if avg_win != 0 else 0.0
        else:
            rr = abs(avg_win / avg_loss)

        print(f"⚖️  Risk:Reward   : 1 : {rr:.2f}")
        print("-" * 65)

        # --- 심볼별 성과 분석 (핵심 추가) ---
        print("🏆 SYMBOL PERFORMANCE (PnL Ranking)")
        if 'sym' in df.columns and 'pnl' in df.columns:
            sym_stats = df.groupby('sym').agg({
                'pnl': 'sum',
                'type': 'count'
            }).sort_values(by='pnl', ascending=False)

            print("\n[TOP 5 PERFORMERS]")
            print(sym_stats.head(5).to_string())

            print("\n[BOTTOM 5 PERFORMERS]")
            print(sym_stats.tail(5).to_string())
        else:
            print("No symbol data available.")

        print("-" * 65)
        print(f"📄 Detailed CSV Log Saved: {self.hm.file_path}")
        print("=" * 65)

    # =========================================================
    # [B EXTENSION] 1m EXIT ONLY
    # =========================================================
    def prepare_data_1m(self, symbols, days=30):
        """
        1분봉은 SL/TP 터치 판정(청산) 용도로만 사용
        - 어떤 상황에서도 None 반환 금지, 실패 시 해당 심볼 스킵
        """
        import ccxt
        import time
        import pandas as pd

        exchange = ccxt.binance({'options': {'defaultType': 'future'}})

        # markets 로딩은 시도만
        try:
            exchange.load_markets()
        except Exception as e:
            print(f"    ⚠️ load_markets failed in prepare_data_1m (continue): {e}")

        since = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
        min_required = 1000  # 1m는 1000개 이상이면 일단 저장

        if symbols is None:
            print("    ⚠️ symbols is None -> skip prepare_data_1m")
            return

        print(f"📥 [VirtualExecutorB] Downloading 1m data for {len(symbols)} symbols...")

        for sym in symbols:
            if sym in self.data_1m:
                continue

            all_ohlcv = []
            temp_since = since

            try:
                print(f"  - Fetching (1m) {sym}...")

                max_retries = 3

                while True:
                    try:
                        ohlcv = exchange.fetch_ohlcv(sym, '1m', since=temp_since, limit=1000)
                    except (ccxt.RequestTimeout, ccxt.NetworkError) as e:
                        max_retries -= 1
                        print(f"    ⚠️ Network/Timeout on (1m) {sym}: {e} (retries left={max_retries})")
                        if max_retries <= 0:
                            print(f"    ❌ Giving up (1m) {sym} due to repeated timeouts.")
                            all_ohlcv = []
                            break
                        time.sleep(1.0)
                        continue
                    except Exception as e:
                        print(f"    ❌ Error fetching (1m) {sym}: {e}")
                        all_ohlcv = []
                        break

                    if not ohlcv:
                        break

                    last_ts = ohlcv[-1][0]
                    if all_ohlcv and last_ts <= all_ohlcv[-1][0]:
                        break

                    all_ohlcv.extend(ohlcv)
                    temp_since = last_ts + 1

                    if len(ohlcv) < 1000:
                        break

                    time.sleep(0.2)

                if len(all_ohlcv) >= min_required:
                    df = pd.DataFrame(
                        all_ohlcv,
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )

                    for c in ['open', 'high', 'low', 'close', 'volume']:
                        df[c] = pd.to_numeric(df[c], errors='coerce')

                    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('datetime', inplace=True)

                    if not df.index.is_monotonic_increasing:
                        df = df.sort_index()

                    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

                    self.data_1m[sym] = df
                else:
                    print(f"    ⚠️ Not enough 1m data for {sym}: {len(all_ohlcv)} rows")

            except Exception as e:
                print(f"    ❌ Error fetching (1m) {sym}: {e}")
                time.sleep(5)

            time.sleep(1.0)

    def check_exit_1m(self, sym, start_dt, end_dt):
        """
        start_dt ~ end_dt 사이 1분봉을 순회하며 SL/TP 터치 여부 판단

        반환:
        ('EXIT', ts, price, reason) 또는 None
        """
        if sym not in self.positions:
            return None
        if sym not in self.data_1m:
            return None

        pos = self.positions[sym]
        df = self.data_1m[sym]

        try:
            sliced = df.loc[(df.index > start_dt) & (df.index <= end_dt)]
        except Exception:
            return None

        if sliced.empty:
            return None

        side = pos.get('side')
        sl = pos.get('sl', None)
        tp = pos.get('tp1', None)

        if sl is None:
            return None

        for ts, row in sliced.iterrows():
            high = row['high']
            low = row['low']

            if side == 'LONG':
                # 보수적: SL 우선
                if low <= sl:
                    return ("EXIT", ts, float(sl), "SL_1M_TOUCH")
                if tp is not None and high >= tp:
                    return ("EXIT", ts, float(tp), "TP_1M_TOUCH")
            else:
                # SHORT
                if high >= sl:
                    return ("EXIT", ts, float(sl), "SL_1M_TOUCH")
                if tp is not None and low <= tp:
                    return ("EXIT", ts, float(tp), "TP_1M_TOUCH")

        return None

    def execute_exit(self, sym, price, reason, dt):
        """
        backtest_engine_b.py 계약:
        self.executor.execute_exit(sym, price, reason, ts)
        - 포지션 정산 + 수수료 차감 + cash 반영
        - HistoryManager / in-memory history 기록
        """
        if sym not in self.positions:
            return

        pos = self.positions[sym]
        amt = pos.get('amount', 0) or 0
        entry = pos.get('entry_price', 0) or 0
        side = pos.get('side', 'LONG')
        margin = pos.get('margin', 0) or 0

        # PnL
        if side == 'LONG':
            pnl = (float(price) - float(entry)) * float(amt)
        else:
            pnl = (float(entry) - float(price)) * float(amt)

        exit_value = float(price) * float(amt)
        fee = exit_value * BASE_FEE
        net_pnl = pnl - fee

        # cash 정산 (증거금 반환 + 손익 - 수수료)
        self.cash += float(margin) + net_pnl

        # equity 즉시 반영 (MTM 단순화)
        self.equity = self.cash

        # CSV 로그 (A 스타일 유지: HistoryManager 사용)
        try:
            self.hm.write(
                dt=dt,
                symbol=sym,
                side=side,
                type_='EXIT',
                price=float(price),
                amount=float(amt),
                pnl=float(net_pnl),
                balance=float(self.equity),
                reason=str(reason)
            )
        except Exception:
            pass

        # 인메모리 히스토리
        try:
            self.history.append({
                'dt': dt,
                'sym': sym,
                'type': 'EXIT',
                'pnl': float(net_pnl),
                'reason': str(reason)
            })
        except Exception:
            pass

        # 포지션 제거
        try:
            del self.positions[sym]
        except Exception:
            pass
