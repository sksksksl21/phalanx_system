import sys
import os
import json
import time
import ccxt
import logging
import traceback
import pandas as pd
from datetime import datetime

# ==========================================
# 1. 설정 및 경로
# ==========================================
current_file_path = os.path.abspath(__file__)
core_dir = os.path.dirname(current_file_path)
root_dir = os.path.dirname(core_dir)
sys.path.append(root_dir)

LOG_FILE_PATH = os.path.join(root_dir, "phalanx_live.log")
STATE_FILE_PATH = os.path.join(root_dir, "data", "state", "phalanx_state.json")
CONFIG_FILE_PATH = os.path.join(root_dir, "config.json")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding='utf-8', mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PHALANX_TITAN")

try:
    from Phalanx_System.utils.telegram_bot import TelegramBot
    from execution.execute_order import OrderExecutor
    from Phalanx_System.strategy.risk_control import RiskControl
    from data.history_manager import HistoryManager
    # 순환 참조 방지를 위해 Monitor는 여기서 import 해도 됨
    from strategy.position_monitor import PositionMonitor
    from strategy.titan_strategy import TitanStrategy
except ImportError as e:
    logger.critical(f"기본 모듈 로드 실패: {e}")
    sys.exit(1)

# ==========================================
# 2. PHALANX ENGINE: V34 REALITY CHECK + SYNC
# ==========================================
class PhalanxEngine:
    def __init__(self):
        if not os.path.exists(CONFIG_FILE_PATH):
            logger.error("Config file not found.")
            sys.exit(1)
        with open(CONFIG_FILE_PATH, 'r') as f:
            self.cfg = json.load(f)

        self.exchange = ccxt.binance({
            'apiKey': self.cfg.get('api_key', ''),
            'secret': self.cfg.get('secret_key', ''),
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        try: self.exchange.load_markets()
        except: pass

        self.telegram = TelegramBot(self.cfg.get('telegram_token'), self.cfg.get('telegram_chat_id'))
        self.executor = OrderExecutor(self.exchange)
        self.risk_ctrl = RiskControl(self.exchange, self.cfg)
        self.history_manager = HistoryManager(os.path.join(root_dir, "data", "state", "trade_history.csv"))
        
        # PositionMonitor 생성 (필수 객체 전달)
        self.position_monitor = PositionMonitor(
            self.exchange, 
            self.executor,
            self.history_manager,
            self.telegram 
        )

        self.titan = TitanStrategy()
        self.positions = {}
        self.is_paused = False
        self.MAX_POSITIONS = self.cfg.get('max_positions', 5)
        self.MAX_INVEST_PER_TRADE = 1000.0 
        self.pre_alert_cooldowns = {}
        
        self.load_state()

    def load_state(self):
        if os.path.exists(STATE_FILE_PATH):
            try:
                with open(STATE_FILE_PATH, 'r') as f:
                    data = json.load(f)
                    self.positions = data.get('positions', {})
            except: pass

    def save_state(self):
        try:
            with open(STATE_FILE_PATH, 'w') as f:
                json.dump({"positions": self.positions}, f, indent=4)
        except: pass

    def process_telegram_commands(self):
        cmd = self.telegram.get_latest_command()
        if cmd:
            if cmd == '/stop': return True
            elif cmd == '/pause': 
                self.is_paused = True
                self.telegram.send_message("⏸ PAUSED")
            elif cmd == '/resume': 
                self.is_paused = False
                self.telegram.send_message("▶️ RESUMED")
            elif cmd == '/status': 
                pos_msg = "\n".join([f"- {s}: {p['side']}" for s, p in self.positions.items()])
                if not pos_msg: pos_msg = "보유 포지션 없음"
                self.telegram.send_message(f"⚔️ <b>[TITAN STATUS]</b>\nActive: {len(self.positions)}/{self.MAX_POSITIONS}\n{pos_msg}")
            elif cmd == '/sync':
                self.sync_account()
                self.telegram.send_message("🔄 계좌 동기화 완료")
        return False

    # =========================================================
    # 🔄 SYNC FUNCTION
    # =========================================================
    def sync_account(self):
        """바이낸스 실계좌와 봇 상태 동기화"""
        try:
            raw_positions = self.exchange.fetch_positions()
            real_positions_map = {}
            
            for p in raw_positions:
                amt = float(p.get('contracts', 0) or p.get('amount', 0))
                if amt != 0:
                    sym = p['symbol']
                    real_positions_map[sym] = {
                        'amount': abs(amt),
                        'side': p['side'].upper(),
                        'entry_price': float(p['entryPrice'])
                    }

            # 로컬엔 있는데 거래소에 없으면 삭제 (청산됨)
            for sym in list(self.positions.keys()):
                if sym not in real_positions_map:
                    logger.warning(f"⚠️ [Sync] 포지션 소멸 확인 (삭제): {sym}")
                    del self.positions[sym]

            # 거래소엔 있는데 로컬에 없으면 추가 (외부 진입)
            for sym, data in real_positions_map.items():
                if sym not in self.positions:
                    logger.warning(f"⚠️ [Sync] 외부 포지션 발견 (복구): {sym}")
                    clean_symbol = sym.split(':')[0]
                    is_major = clean_symbol in self.titan.major_coins
                    
                    self.positions[sym] = {
                        'entry_price': data['entry_price'],
                        'amount': data['amount'],
                        'side': data['side'],
                        'sl': 0, # 모니터가 자동 업데이트
                        'tp1': 0,
                        'tp1_hit': True, # 보수적 관리
                        'entry_time': time.time(),
                        'entry_type': 'RESTORED',
                        'highest_price': data['entry_price'],
                        'lowest_price': data['entry_price']
                    }
            self.save_state()
        except Exception as e:
            logger.error(f"Sync Error: {e}")

    # =========================================================
    # ⚔️ CORE LOGIC
    # =========================================================
    def find_and_kill(self):
        try:
            tickers = self.exchange.fetch_tickers()
            blacklist = self.titan.get_blacklist()
            
            targets = []
            for s in tickers:
                if '/USDT:USDT' in s:
                    clean_s = s.split(':')[0]
                    if clean_s not in blacklist:
                        if tickers[s].get('quoteVolume', 0) > 50000000:
                            targets.append(s)
            
            targets = sorted(targets, key=lambda x: tickers[x].get('quoteVolume', 0), reverse=True)[:30]
            
            for symbol in targets:
                if symbol in self.positions: continue
                if len(self.positions) >= self.MAX_POSITIONS: return

                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, '15m', limit=1000)
                    if not ohlcv: continue

                    signal, sl_price, tp_price = self.titan.analyze(symbol, ohlcv)
                    
                    if signal:
                        logger.info(f"⚔️ TITAN SIGNAL: {symbol} [{signal}]")
                        msg = (f"⚔️ <b>[TITAN 진입]</b>\nTarget: {symbol}\nSide: {signal}\nSL: {sl_price:.4f}")
                        self.telegram.send_message(msg)
                        self.execute_market_order(symbol, signal, sl_price, tp_price)
                        time.sleep(1)
                    else:
                        # Pre-alert
                        current_price = ohlcv[-1][4]
                        if current_price == 0: continue
                        dist = abs(current_price - sl_price) / current_price * 100
                        
                        if dist < 0.5:
                            last_alert = self.pre_alert_cooldowns.get(symbol, 0)
                            if time.time() - last_alert > 3600:
                                potential_side = "SHORT" if current_price > sl_price else "LONG"
                                msg = (f"🚨 <b>[진입 임박]</b> {symbol}\n현재가: {current_price}\n기준가: {sl_price:.4f} (거리 {dist:.2f}%)\n👉 {potential_side} 대기 중")
                                self.telegram.send_message(msg)
                                logger.info(f"Pre-alert sent: {symbol}")
                                self.pre_alert_cooldowns[symbol] = time.time()

                except Exception as e:
                    continue

        except Exception as e:
            logger.error(f"Global Scan Error: {e}")

    # ★ [중요] 이 함수는 클래스 안에 있어야 하며, 들여쓰기가 find_and_kill과 동일해야 합니다.
    def execute_market_order(self, symbol, side, stop_loss, tp_price):
        try:
            balance_data = self.exchange.fetch_balance()
            free_balance = balance_data['USDT']['free']
            invest_amount = min(free_balance, self.MAX_INVEST_PER_TRADE)
            
            if invest_amount < 10: 
                logger.warning(f"잔액 부족: {invest_amount}")
                return

            ticker = self.exchange.fetch_ticker(symbol)
            amount = self.risk_ctrl.calculate_entry_size(symbol, ticker['last'], invest_amount)

            if amount <= 0: return

            side_lower = 'buy' if side == 'LONG' else 'sell'
            order, filled_price = self.executor.create_order(symbol, side_lower, amount, order_type='market')
            
            if order:
                clean_symbol = symbol.split(':')[0]
                is_major = clean_symbol in self.titan.major_coins
                
                self.positions[symbol] = {
                    'entry_price': filled_price,
                    'amount': amount,
                    'side': side,
                    'sl': stop_loss,
                    'tp1': tp_price,
                    'tp1_hit': False,
                    'entry_time': time.time(),
                    'entry_type': 'MAJOR' if is_major else 'ALT',
                    'highest_price': filled_price if side == 'LONG' else 0,
                    'lowest_price': filled_price if side == 'SHORT' else 999999
                }
                self.save_state()
                logger.info(f"Order Filled: {symbol} @ {filled_price}")

        except Exception as e:
            logger.error(f"Execution Failed ({symbol}): {e}")
            self.telegram.send_message(f"🚫 주문 실패: {symbol}\n{e}")

    def run(self):
        self.telegram.send_message("⚔️ <b>PHALANX V34: SYNC & ALERT</b> 가동")
        logger.info("SYSTEM START")
        
        self.sync_account() # 시작 시 동기화
        scan_count = 0

        while True:
            try:
                # 텔레그램 명령 처리 주석 해제 가능
                # if self.process_telegram_commands(): break

                if self.is_paused:
                    time.sleep(10); continue
                
                # 5분마다 동기화
                if scan_count % 30 == 0: self.sync_account()

                if self.positions:
                    self.position_monitor.monitor_and_respond(self.positions, self.save_state)

                if len(self.positions) < self.MAX_POSITIONS:
                    self.find_and_kill()

                scan_count += 1
                if scan_count % 6 == 0:
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"👀 [{ts}] TITAN 감시 중... (보유: {len(self.positions)})")

                time.sleep(10)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Loop Error: {e}")
                time.sleep(10)

if __name__ == "__main__":
    bot = PhalanxEngine()
    bot.run()