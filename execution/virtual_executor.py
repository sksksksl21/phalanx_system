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


class VirtualExecutor:
    """
    [Phalanx Execution Module - Virtual V3.0]
    Role: Accounting, Physics Simulation & Data Recording
    Features:
    - Full Integration with HistoryManager
    - Symbol-level Performance Tracking
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
        # (기존 ccxt 로직 유지)
        import ccxt
        exchange = ccxt.binance({'options': {'defaultType': 'future'}})
        try:
            tickers = exchange.fetch_tickers()
            targets = []
            for s in tickers:
                if '/USDT:USDT' in s:
                    vol = tickers[s].get('quoteVolume', 0)
                    if vol > 0:
                        targets.append((s, vol))
            targets.sort(key=lambda x: x[1], reverse=True)
            return [t[0] for t in targets][:20]
        except:
            return []

    def prepare_data(self, symbols, days=180):
        # (기존 데이터 다운로드 로직 유지)
        import ccxt
        import time
        import pandas as pd  # 데이터프레임 변환용 추가

        exchange = ccxt.binance({'options': {'defaultType': 'future'}})
        since = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
        data_map = {}
        min_required = 2000

        print(f"📥 [VirtualExecutor] Downloading data for {len(symbols)} symbols...")

        for sym in symbols:
            all_ohlcv = []
            temp_since = since

            try:
                print(f"  - Fetching {sym}...")  # 진행상황 표시
                while True:
                    # 1000개씩 끊어서 가져오기
                    ohlcv = exchange.fetch_ohlcv(sym, '15m', since=temp_since, limit=1000)
                    if not ohlcv:
                        break

                    # [Patch] 중복/정체 방지:
                    # - 일부 환경에서 동일 타임스탬프가 반복될 수 있으므로,
                    #   마지막 타임스탬프가 업데이트되지 않으면 루프가 길어질 수 있음
                    last_ts = ohlcv[-1][0]
                    if all_ohlcv and last_ts <= all_ohlcv[-1][0]:
                        # 더 이상 전진이 안 되면 탈출 (무한루프 방지)
                        break

                    all_ohlcv.extend(ohlcv)
                    temp_since = last_ts + 1

                    # 기간 충족했으면 탈출
                    if len(ohlcv) < 1000:
                        break

                    # [수정 1] 페이지 넘길 때 휴식 (0.05 -> 0.2로 늘림)
                    time.sleep(0.2)

                # 데이터프레임 변환 (필수)
                if len(all_ohlcv) >= min_required:
                    df = pd.DataFrame(
                        all_ohlcv,
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )

                    # [Patch] 타입 정리 (수치 계산 안정성)
                    # - ccxt는 숫자를 float으로 주지만, 환경/파싱 이슈로 object가 끼는 경우가 있음
                    for c in ['open', 'high', 'low', 'close', 'volume']:
                        df[c] = pd.to_numeric(df[c], errors='coerce')

                    # [Patch] timestamp 중복 제거 + 정렬
                    # - 데이터가 겹치면 전략 지표가 깨지고, resample/rolling도 왜곡됨
                    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

                    # 인덱스 설정
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('datetime', inplace=True)

                    # [Patch] 인덱스 단조증가 보장 (strategy에서 resample/ffill 안정화)
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
            # 이걸 안 넣으면 아까처럼 'Skipping...' 뜨면서 다 막힙니다.
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
        # - 손실 합계가 0이면 inf
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
            # [Patch] peak가 0일 수 있는 비정상 케이스 방어
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
            # PnL 합계, 거래 횟수, 승률 계산
            sym_stats = df.groupby('sym').agg({
                'pnl': 'sum',
                'type': 'count'  # 거래 횟수
            }).sort_values(by='pnl', ascending=False)

            # 상위 5개 (효자)
            print("\n[TOP 5 PERFORMERS]")
            print(sym_stats.head(5).to_string())

            # 하위 5개 (역적)
            print("\n[BOTTOM 5 PERFORMERS]")
            print(sym_stats.tail(5).to_string())
        else:
            print("No symbol data available.")

        print("-" * 65)
        print(f"📄 Detailed CSV Log Saved: {self.hm.file_path}")
        print("=" * 65)
