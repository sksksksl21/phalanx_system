import sys
import os
import time
import json
import logging
import traceback
import pandas as pd
from datetime import datetime

# 상위 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# 로깅 설정
LOG_FILE = os.path.join(root_dir, "phalanx_live.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8', mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PhalanxCore")

# 모듈 임포트 (구조 보존)
try:
    from execution.binance_executor import BinanceExecutor
    from strategy.titan_strategy import TitanStrategy
    from strategy.position_monitor import PositionMonitor
    from strategy.risk_control import RiskControl
    from utils.telegram_bot import TelegramBot
    # HistoryManager는 로컬 파일 로깅용 (간소화 구현 가정)
    from utils.history_manager import HistoryManager 
except ImportError as e:
    logger.critical(f"❌ CRITICAL: 필수 모듈 로드 실패 - {e}")
    sys.exit(1)

class LiveEngine:
    """
    [Phalanx Core Module]
    Role: System Orchestrator (Heart)
    Responsibilities:
    1. Life Cycle Management (Start, Loop, Stop)
    2. Data Flow Control (Exchange -> Strategy -> Executor)
    3. State Integrity (Single Source of Truth)
    4. Exception Handling (Resilience)
    """

    def __init__(self):
        self.root_dir = root_dir
        self.config_path = os.path.join(root_dir, "config.json")
        self.state_path = os.path.join(root_dir, "phalanx_state.json")
        self.history_path = os.path.join(root_dir, "trade_history.csv")
        
        # 1. 설정 로드
        self.cfg = self._load_config()
        
        # 2. 하위 모듈 초기화
        try:
            self.telegram = TelegramBot(self.cfg['telegram_token'], self.cfg['telegram_chat_id'])
            self.executor = BinanceExecutor(self.cfg)
            self.risk_ctrl = RiskControl(self.executor, self.cfg)
            self.history_mgr = HistoryManager(self.history_path)
            self.titan = TitanStrategy()

            # PositionMonitor에 의존성 주입
            self.monitor = PositionMonitor()
            
            
            logger.info("✅ Phalanx Engine Modules Initialized.")
            
        except Exception as e:
            logger.critical(f"Initialization Failed: {e}")
            sys.exit(1)

        # 3. 상태 로드 (제6원칙)
        self.positions = {}
        self._load_state()

        # 4. 런타임 변수
        self.is_running = True
        self.scan_interval = self.cfg.get('scan_interval', 60) # 기본 60초
        self.max_positions = self.cfg.get('max_positions', 5)

    def _load_config(self):
        if not os.path.exists(self.config_path):
            logger.critical("Config file missing.")
            sys.exit(1)
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_state(self):
        """상태 파일 로드 (Single Source of Truth)"""
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.positions = data.get('positions', {})
                logger.info(f"State Loaded: {len(self.positions)} positions active.")
            except Exception as e:
                logger.error(f"State Load Error: {e}")
                # 파일 손상 시 백업 후 초기화 (Level 3 대응)
                os.rename(self.state_path, self.state_path + ".bak")
                self.positions = {}

    def _save_state(self):
        """상태 저장 (Atomic Write 권장)"""
        try:
            temp_path = self.state_path + ".tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump({"positions": self.positions, "updated": time.time()}, f, indent=4)
            os.replace(temp_path, self.state_path)
        except Exception as e:
            logger.critical(f"State Save Error: {e}")

    # =========================================================
    # 🔄 Synchronization Layer
    # =========================================================
    def sync_wallet_and_positions(self):
        """
        [Reality Check] 거래소 실제 상태와 로컬 상태 동기화
        Phalanx 제6원칙: 진실은 state.json이지만, 물리적 불일치는 복구해야 함.
        """
        try:
            # 1. 잔고 확인 (단순 로깅)
            balance = self.executor.fetch_balance()
            usdt_free = balance['USDT']['free']

            
            
            


            # 2. 거래소 포지션 확인
            real_positions = self.executor.fetch_positions()
            real_pos_map = {}
            
            # 거래소 -> 맵 변환
            for p in real_positions:
                amt = float(p.get('contracts', 0) or p.get('amount', 0))
                if amt > 0:
                    sym = p['symbol']
                    real_pos_map[sym] = {
                        'amount': amt,
                        'entry_price': float(p.get('entryPrice', 0)),
                        'side': p['side'].upper() # LONG/SHORT
                    }
            
            # A. 로컬에는 있는데 거래소에 없는 경우 (청산됨/수동종료)
            for sym in list(self.positions.keys()):
                if sym not in real_pos_map:
                    logger.warning(f"⚠️ [Sync] Ghost Position Detected: {sym}. Removing from state.")
                    # 로그 남기고 삭제
                    
                    self.history_mgr.log_trade(
                        symbol=sym,
                        side=self.positions[sym]['side'], 
                        type_note="EXIT",       # 강제 청산 처리
                        price=0,                # 체결가 불명
                        amount=0,               # 수량 불명
                        pnl=0,                  # PnL 불명
                        fee=0,                  # 수수료 불명
                        balance=0,              # 잔고 불명 (단순 기록용)
                        reason="SYNC_LOST"      # 사유 명시
                    )
                    #self.history_mgr.log_trade(sym, self.positions[sym]['side'], 0, 0, 0, 0, 0, "SYNC_LOST")
                    del self.positions[sym]
                    self._save_state()

            # B. 거래소에는 있는데 로컬에 없는 경우 (수동진입/오류복구)
            # -> Phalanx는 '통제되지 않은 포지션'을 관리하려 들지 않음 (Log Only)
            #    단, 치명적 충돌 방지를 위해 관리 목록에는 넣지 않음.
            for sym in real_pos_map:
                if sym not in self.positions:
                    logger.info(f"👀 [Sync] Unknown Position on Exchange: {sym}")

        except Exception as e:
            logger.error(f"Sync Failed: {e}")

    # =========================================================
    # ⚔️ Strategy Execution Layer
    # =========================================================
    def scan_and_trade(self):
        """시장 스캔 및 신규 진입 로직"""
        # 최대 포지션 도달 시 스캔 생략
        if len(self.positions) >= self.max_positions:
            return

        try:
            # 1. 대상 선정 (Top Volume) - Executor가 지원해야 함 (임시 로직 구현)
            # Executor에 fetch_tickers가 없으면 이 부분은 Executor 확장이 필요함.
            # 여기서는 binance_executor.fetch_ticker 루프 또는 별도 로직 가정.
            # 효율성을 위해 Executor에 get_top_volume_tickers 구현 권장.
            # *현재 구현된 Executor에는 fetch_tickers가 없으므로 fetch_ohlcv 테스트용*
            
            # V32 설정의 Major Coins 우선 스캔
            candidates = list(self.titan.major_coins)
            # (확장 시: Volume 상위 코인 추가 로직 필요)

            blacklist = self.titan.get_blacklist()

            for symbol in candidates:
                if symbol in self.positions: continue
                if symbol in blacklist: continue
                
                # 2. 데이터 수집
                ohlcv = self.executor.fetch_ohlcv(symbol, '15m', limit=250)
                if not ohlcv or len(ohlcv) < 200: continue

                # DataFrame 변환 (제4원칙)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                df = self.titan.calculate_indicators(symbol, df)
                
                # 3. 전략 분석 (Titan)
                signal, sl_price, tp_price = self.titan.analyze(symbol, df)

                if signal:
                    self._execute_entry(symbol, signal, sl_price, tp_price)
                    if len(self.positions) >= self.max_positions: break
                    
        except Exception as e:
            logger.error(f"Scan Error: {e}")

    def _execute_entry(self, symbol, signal, sl_price, tp_price):
        """진입 실행 (Risk Control -> Executor -> State Update)"""
        try:
            # 1. 자금 관리
            balance = self.executor.fetch_balance()
            usdt_free = balance['USDT']['free']
            usdt_total = balance['USDT']['total'] # [NEW] 로그 기록용 (총 자산)
            
            ticker = self.executor.fetch_ticker(symbol)
            current_price = ticker['last']
            
            amount = self.risk_ctrl.calculate_entry_size(symbol, current_price, usdt_free, sl_price)
            
            if amount <= 0:
                logger.warning(f"Entry Rejected (Risk): {symbol} Amt: {amount}")
                return

            # 2. 주문 집행
            side_exec = 'buy' if signal == 'LONG' else 'sell'
            order, fill_price = self.executor.create_order(symbol, side_exec, amount)
            
            if order:
                # 3. 상태 갱신
                self.positions[symbol] = {
                    'entry_price': fill_price,
                    'amount': amount,
                    'side': signal,
                    'sl': sl_price,
                    'tp1': tp_price,
                    'tp1_hit': False,
                    'entry_time': time.time()
                }
                self._save_state()

                fee_est = fill_price * amount * 0.0005
                
                self.history_mgr.log_trade(
                    symbol=symbol,
                    side=side_exec,      # 'buy' or 'sell'
                    type_note="ENTRY",   # 진입
                    price=fill_price,
                    amount=amount,
                    pnl=0,               # 진입 시 실현 손익은 0
                    fee=fee_est,         # 수수료 기록
                    balance=usdt_total,  # 현재 총 자산 기록
                    reason="Signal"      # 전략 신호 진입
                )
                
                # 4. 알림
                msg = f"⚔️ <b>[진입 성공]</b> {symbol}\nSide: {signal}\nPrice: {fill_price}\nSize: {amount}\nSL: {sl_price}"
                self.telegram.send_message(msg)
                logger.info(f"Entry Success: {symbol} {signal}")

        except Exception as e:
            logger.error(f"Entry Execution Failed ({symbol}): {e}")
            self.telegram.send_message(f"🚫 진입 실패: {symbol}\n{e}")

    # =========================================================
    # 🏃 Main Loop
    # =========================================================
    def run(self):
        logger.info("🚀 PHALANX ENGINE V3.0 STARTED")
        self.telegram.send_message("🚀 <b>PHALANX ENGINE V3.0</b> 가동 시작")
        
        last_scan_time = 0
        last_sync_time = 0
        sync_interval = 300

        while self.is_running:
            try:
                now = time.time()

                # ---------------------------------------------------------
                # 1. 포지션 모니터링 & 청산 (백테스트와 로직 100% 일치화)
                # ---------------------------------------------------------
                if self.positions:
                    # 복사본으로 루프 (딕셔너리 변경 방지)
                    active_symbols = list(self.positions.keys())
                    
                    for sym in active_symbols:
                        pos = self.positions[sym]
                        
                        # [Step A] 실시간 데이터 확보 (백테스트의 curr_row 역할)
                        # 단순 Ticker가 아니라 OHLCV를 가져와서 지표를 계산해야 함
                        ohlcv = self.executor.fetch_ohlcv(sym, '15m', limit=250)
                        
                        if ohlcv is None or len(ohlcv) < 200:
                            continue # 데이터 부족 시 스킵
                            
                        # DataFrame 변환
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        
                        # [Step B] 지표 계산 (Titan에게 위임)
                        # 여기서 'st_val'(SuperTrend)이 생성됨 -> 이것이 없어서 문제였음
                        df = self.titan.calculate_indicators(sym, df)
                        
                        # 가장 최근 확정봉 기준 (또는 실시간봉 사용 여부 결정)
                        # 실전에서는 변동성 때문에 '직전 확정봉(-2)'의 ST값과 '현재가(-1)'를 비교하는 것이 안전
                        curr_row = df.iloc[-1]
                        
                        # [Step C] 시장 데이터 패키징 (Monitor용)
                        market_data = {
                            'close': curr_row['close'],
                            'high': curr_row['high'],
                            'low': curr_row['low'],
                            'atr': curr_row.get('atr', curr_row['close']*0.01),
                            'st_val': curr_row.get('st_val', 0) # ★ 이제 SuperTrend 값이 들어감
                        }

                        # [Step D] 판단 (PositionMonitor)
                        action, exec_price, reason, new_sl = self.monitor.check_conditions(sym, pos, market_data)

                        # [Step E] 실행 (Execution)
                        
                        # 1. SL 업데이트 (Trailing)
                        if new_sl != pos['sl']:
                            old_sl = pos['sl']
                            pos['sl'] = new_sl
                            self._save_state()
                            logger.info(f"🔄 [Trailing] {sym} SL Updated: {old_sl} -> {new_sl}")
                            # 필요 시 텔레그램 알림 (너무 잦으면 생략)

                        # 2. TP1 부분 익절
                        if action == 'TP1':
                            self._execute_live_exit(sym, pos, ratio=0.5, reason='TP1')

                        # 3. 완전 청산 (SL Trigger)
                        elif action == 'EXIT':
                            self._execute_live_exit(sym, pos, ratio=1.0, reason=reason)
                
                # ---------------------------------------------------------

                # 2. 계좌 동기화
                if now - last_sync_time > sync_interval:
                    self.sync_wallet_and_positions()
                    last_sync_time = now

                # 3. 신규 진입 스캔
                if now - last_scan_time > self.scan_interval:
                    self.scan_and_trade()
                    last_scan_time = now

                time.sleep(2) 

            except KeyboardInterrupt:
                logger.info("Shutdown Signal Received.")
                self.stop()
            except Exception as e:
                logger.critical(f"Main Loop Crash: {e}\n{traceback.format_exc()}")
                self.telegram.send_message(f"⚠️ <b>[ENGINE CRITICAL]</b>\n{e}")
                time.sleep(10)

    # =========================================================
    # 🛠️ Live Execution Helper
    # =========================================================
    def _execute_live_exit(self, sym, pos, ratio, reason):
        
        
        
        
        """실전 청산 실행 함수"""
        try:
            amount = pos['amount'] * ratio
            side = 'sell' if pos['side'] == 'LONG' else 'buy' # 청산은 반대 매매
            
            # 바이낸스 주문 전송
            order, fill_price = self.executor.create_order(sym, side, amount)
            
            if order:
                pnl = (fill_price - pos['entry_price']) * amount if pos['side'] == 'LONG' else (pos['entry_price'] - fill_price) * amount
                
                balance_data = self.executor.fetch_balance()
                usdt_total = balance_data['USDT']['total']
                
                # 2. 수수료 추산
                fee_est = fill_price * amount * 0.0005
                
                # 3. 타입 구분 (TP1인지 완전 청산인지)
                type_note = "TP1" if reason == 'TP1' else "EXIT"
                
                # 4. 로그 기록
                self.history_mgr.log_trade(
                    symbol=sym,
                    side=side,           # 'sell' or 'buy' (청산 방향)
                    type_note=type_note, # EXIT or TP1
                    price=fill_price,
                    amount=amount,
                    pnl=pnl,             # 실현 손익
                    fee=fee_est,         # 수수료
                    balance=usdt_total,  # 갱신된 총 자산
                    reason=reason        # STOP_LOSS, SIGNAL_FLIP, TP1 등
                )
                
                log_msg = f"📉 <b>[청산: {reason}]</b> {sym}\nPnL: ${pnl:.2f}\nPrice: {fill_price}"
                self.telegram.send_message(log_msg)
                logger.info(f"Exit Executed: {sym} ({reason}) PnL: {pnl}")
                
                # 상태 업데이트
                if ratio >= 1.0:
                    del self.positions[sym]
                else:
                    pos['amount'] -= amount
                    if reason == 'TP1': pos['tp1_hit'] = True
                
                self._save_state()
                
        except Exception as e:
            logger.error(f"Exit Execution Failed ({sym}): {e}")
            self.telegram.send_message(f"🚫 청산 실패 ({reason}): {sym}\n{e}")


    def stop(self):
        self.is_running = False
        self._save_state()
        logger.info("System Shutdown Complete.")
        self.telegram.send_message("🛑 시스템 종료")

if __name__ == "__main__":
    engine = LiveEngine()
    engine.run()