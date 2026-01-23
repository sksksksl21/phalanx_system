import csv
import os
import logging
from datetime import datetime

logger = logging.getLogger("PhalanxHistory")

class HistoryManager:
    def __init__(self, file_path):
        """
        [PHALANX TITAN V32] Trade History Manager
        Role: Unified Ledger & Performance Journal
        Path: data/trade_history.csv
        """
        self.file_path = file_path
        self.init_file()

    def init_file(self):
        """
        파일 초기화 및 헤더 정의
        * Physics Data: Fee, Balance
        * Analysis Data: Strategy, PnL%, Reason
        """
        try:
            dir_name = os.path.dirname(self.file_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)

            # 파일이 없거나 비어있으면 헤더 작성
            if not os.path.exists(self.file_path) or os.path.getsize(self.file_path) == 0:
                with open(self.file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Datetime",     # YYYY-MM-DD HH:MM:SS
                        "Timestamp",    # Unix Timestamp
                        "Symbol", 
                        "Side", 
                        "Type",         # ENTRY, EXIT, TP1, SL
                        "Price", 
                        "Amount", 
                        "Value",        # Notional Value
                        "Fee",          # 수수료 (Physics)
                        "PnL",          # 실현 손익
                        "PnL(%)",       # 수익률
                        "Balance",      # 누적 잔고 (Physics)
                        "Strategy",     # Titan V32
                        "Reason"        # 상세 사유 (Signal Flip, Trailing 등)
                    ])
                logger.info(f"📄 거래 장부 초기화 완료: {self.file_path}")

        except Exception as e:
            logger.error(f"❌ History Init Failed: {e}")

    def log_trade(self, symbol, side, type_note, price, amount, pnl, fee, balance, reason, strategy="TitanV32"):
        """
        거래 기록 (Engine 호출 규격 준수)
        """
        try:
            now = datetime.now()
            dt_str = now.strftime('%Y-%m-%d %H:%M:%S')
            ts = int(now.timestamp())
            
            # Value 계산
            value = price * amount
            
            # PnL% 계산 (진입에는 0)
            pnl_pct = 0.0
            if type_note in ['EXIT', 'TP1'] and value > 0:
                # 대략적인 수익률 (정확한 진입가는 Engine에서 관리하므로 여기선 PnL/Value로 추산)
                # 주의: 정확한 ROE는 Entry Price가 있어야 하므로, 여기선 단순 비율만 기록
                pnl_pct = (pnl / (value - pnl)) * 100 if (value - pnl) != 0 else 0

            with open(self.file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    dt_str,
                    ts,
                    symbol,
                    side,
                    type_note,
                    f"{price:.4f}",
                    f"{amount:.4f}",
                    f"{value:.2f}",
                    f"{fee:.4f}",       # 핵심: 비용
                    f"{pnl:.4f}",       # 핵심: 손익
                    f"{pnl_pct:.2f}%",  # 분석용
                    f"{balance:.2f}",   # 핵심: 자산 추적
                    strategy,
                    reason
                ])
                
        except Exception as e:
            logger.error(f"❌ Log Trade Failed ({symbol}): {e}")