import ccxt
import pandas as pd
import pandas_ta as ta
import time
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# 설정
TOP_N = 30
MAJOR_COINS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 'AVAX/USDT']
BLACKLIST = [
    'TRADOOR/USDT', 'BARD/USDT', 'PENGU/USDT', 'WLFI/USDT',
    'FARTCOIN/USDT', 'USDC/USDT', 'FDUSD/USDT'
]

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_radar():
    exchange = ccxt.binance({'options': {'defaultType': 'future'}, 'enableRateLimit': True})
    
    print("🚀 [TITAN RADAR] 레이더 가동 시작...")
    
    while True:
        try:
            # 1. 대상 선정
            tickers = exchange.fetch_tickers()
            targets = []
            for s in tickers:
                if s.endswith('/USDT:USDT') and s.split(':')[0] not in BLACKLIST:
                    if tickers[s].get('quoteVolume', 0) > 50000000:
                        targets.append(s)
            
            # 거래대금 상위 30개
            targets = sorted(targets, key=lambda x: tickers[x]['quoteVolume'], reverse=True)[:TOP_N]
            
            radar_data = []

            # 2. 데이터 분석
            print(f"\r⏳ Scanning {len(targets)} assets... ", end="")
            
            for symbol in targets:
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, '15m', limit=1000) # 메인엔진과 동일 조건
                    if not ohlcv: continue
                    
                    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
                    
                    clean_symbol = symbol.split('/')[0]
                    is_major = clean_symbol in MAJOR_COINS
                    
                    # 지표 계산
                    st_len = 10 if is_major else 12
                    st = ta.supertrend(df['high'], df['low'], df['close'], length=st_len, multiplier=3.0)
                    
                    if st is None or st.empty: continue

                    st_line = st[st.columns[0]].iloc[-1]
                    st_dir = st[st.columns[1]].iloc[-1]
                    curr_price = df['close'].iloc[-1]
                    
                    # ★ 거리 계산 (얼마나 가까운가?)
                    # 가격과 SuperTrend 라인 사이의 거리 (%)
                    distance = abs(curr_price - st_line) / curr_price * 100
                    
                    trend_str = "🟢 UP" if st_dir == 1 else "🔴 DOWN"
                    
                    # 알트용 EMA 체크
                    ema200 = ta.ema(df['close'], length=200).iloc[-1]
                    ema_ok = True
                    if not is_major:
                        if st_dir == 1 and curr_price <= ema200: ema_ok = False
                        if st_dir == -1 and curr_price >= ema200: ema_ok = False
                    
                    radar_data.append({
                        'Symbol': clean_symbol,
                        'Price': curr_price,
                        'Trend': trend_str,
                        'ST_Line': st_line,
                        'Dist(%)': distance, # 작을수록 곧 뒤집힘
                        'EMA_Filter': "✅" if (is_major or ema_ok) else "❌",
                        'Type': 'MAJOR' if is_major else 'ALT'
                    })

                except: continue

            # 3. 화면 출력 (거리순 정렬: 곧 신호 뜰 놈이 위로)
            radar_data.sort(key=lambda x: x['Dist(%)'])
            
            clear_screen()
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"📡 [TITAN RADAR] {ts} | Scanning Top {TOP_N}")
            print("=" * 85)
            print(f"{'SYMBOL':<8} | {'TYPE':<5} | {'PRICE':<10} | {'TREND':<8} | {'DIST(%)':<8} | {'EMA':<4} | {'STATUS'}")
            print("-" * 85)
            
            for d in radar_data:
                # 거리가 1% 미만이면 HOT 표시
                status = ""
                if d['Dist(%)'] < 1.0: status = "🔥 SOON!"
                if d['Dist(%)'] < 0.3: status = "🚨 IMMINENT!"
                
                print(f"{d['Symbol']:<8} | {d['Type']:<5} | {d['Price']:<10.4f} | {d['Trend']:<8} | {d['Dist(%)']:>6.2f}% | {d['EMA_Filter']:<4} | {status}")
            
            print("=" * 85)
            print("💡 Dist(%)가 0.00%에 가까워질수록 추세 반전(진입)이 임박한 것입니다.")
            print("💡 EMA가 ❌면 추세가 뒤집혀도 진입하지 않습니다 (안전장치).")
            
            # 15초마다 갱신
            time.sleep(15)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    run_radar()