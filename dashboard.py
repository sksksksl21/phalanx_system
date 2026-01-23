import streamlit as st
import pandas as pd
import json
import time
import os
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
from datetime import timedelta, datetime
import warnings

# Plotly 경고 무시
warnings.filterwarnings("ignore", category=UserWarning, module="plotly")

# ==========================================
# 1. 설정 및 경로 (Phalanx Infrastructure)
# ==========================================
st.set_page_config(
    page_title="PHALANX Titan Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 경로 자동 인식 (root_dir: phalanx_system/)
current_file_path = os.path.abspath(__file__)
root_dir = os.path.dirname(current_file_path)

# 파일 경로 매핑
LOG_FILE = os.path.join(root_dir, "phalanx_live.log")
STATE_FILE = os.path.join(root_dir, "phalanx_state.json")
HISTORY_FILE = os.path.join(root_dir, "trade_history.csv")
CONFIG_FILE = os.path.join(root_dir, "config.json")

# ==========================================
# 2. 데이터 로드 함수 (Utils)
# ==========================================
def load_json(filepath):
    if not os.path.exists(filepath): return {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except: return {}

def load_state_data():
    """State Loading (Single Source of Truth)"""
    data = load_json(STATE_FILE)
    positions = data.get("positions", {})
    return positions

def load_history():
    if not os.path.exists(HISTORY_FILE) or os.path.getsize(HISTORY_FILE) == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(HISTORY_FILE, on_bad_lines='skip')
        # Timestamp 처리 (파일명 변경 가능성 대비)
        time_col = [c for c in df.columns if 'time' in c.lower()][0]
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        return df
    except: return pd.DataFrame()

def read_logs():
    if not os.path.exists(LOG_FILE): return []
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors='replace') as f:
            return f.readlines()
    except: return []

@st.cache_resource
def init_exchange():
    config = load_json(CONFIG_FILE)
    return ccxt.binance({
        'apiKey': config.get('api_key', ''),
        'secret': config.get('secret_key', ''),
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })

exchange = init_exchange()

# ==========================================
# 3. 계산 및 차트 로직 (Observability)
# ==========================================
def calculate_metrics(config, positions):
    """자산 현황 및 미실현 손익 계산"""
    try:
        balance = exchange.fetch_balance()
        total_equity = float(balance['USDT']['total'])
        free_money = float(balance['USDT']['free'])
    except:
        # API 실패 시 기본값 (Config 참조)
        total_equity = config.get('sim_balance', 0.0)
        free_money = 0.0
    
    unrealized_pnl = 0
    active_positions = []
    
    if positions:
        for symbol, pos in positions.items():
            try:
                # 현재가 조회
                try: 
                    ticker = exchange.fetch_ticker(symbol)
                    current_price = float(ticker['last'])
                except: 
                    current_price = float(pos['entry_price'])
                
                entry = float(pos['entry_price'])
                amt = float(pos['amount'])
                side = pos.get('side', 'LONG').upper()
                sl = float(pos.get('sl', 0))
                tp1 = float(pos.get('tp1', 0))
                tp1_hit = pos.get('tp1_hit', False)
                
                # PnL 계산
                if side in ['BUY', 'LONG']:
                    pnl = (current_price - entry) * amt
                    roe = ((current_price - entry) / entry) * 100 * config.get('leverage', 3)
                else:
                    pnl = (entry - current_price) * amt
                    roe = ((entry - current_price) / entry) * 100 * config.get('leverage', 3)
                
                unrealized_pnl += pnl
                
                active_positions.append({
                    "Symbol": symbol,
                    "Side": side,
                    "Entry": entry,
                    "Current": current_price,
                    "PnL($)": pnl,
                    "ROE(%)": roe,
                    "SL": sl,
                    "TP1": "Done" if tp1_hit else f"{tp1:.4f}"
                })
            except Exception: pass

    return total_equity, free_money, unrealized_pnl, pd.DataFrame(active_positions)

@st.cache_data(ttl=60)
def fetch_chart_data(symbol):
    try:
        clean = symbol.split(':')[0]
        # Titan Strategy에 맞춰 15분봉 조회
        ohlcv = exchange.fetch_ohlcv(clean, '15m', limit=200) 
        if not ohlcv: return None

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = (pd.to_datetime(df['timestamp'], unit='ms') + timedelta(hours=9)) # KST 보정
        df.set_index('timestamp', inplace=True)
        
        # Titan V32 핵심 지표: EMA 200 & SuperTrend
        df['ema200'] = ta.ema(df['close'], length=200)
        
        # SuperTrend (Major/Alt 자동 구분은 어려우므로 기본 10, 3.0 설정)
        st_out = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3.0)
        if st_out is not None:
            df['supertrend'] = st_out[st_out.columns[0]] # 가격선
            df['st_dir'] = st_out[st_out.columns[1]] # 방향

        return df
    except: return None

def plot_minichart(symbol, pos_info=None):
    df = fetch_chart_data(symbol)
    if df is None:
        st.error(f"차트 데이터 로드 실패: {symbol}")
        return
    
    plot_df = df.tail(60)
    
    fig = make_subplots(rows=1, cols=1)
    
    # 캔들차트
    fig.add_trace(go.Candlestick(
        x=plot_df.index, 
        open=plot_df['open'], high=plot_df['high'], low=plot_df['low'], close=plot_df['close'], 
        name='Price'
    ))
    
    # 지표 시각화 (EMA200, SuperTrend)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ema200'], line=dict(color='yellow', width=1), name='EMA 200'))
    
    if 'supertrend' in plot_df.columns:
        # SuperTrend 색상 (상승:초록, 하락:빨강)
        colors = ['green' if d == 1 else 'red' for d in plot_df['st_dir']]
        # Plotly Scatter는 단일 색상이므로 Marker로 표현하거나 구간별로 나눠야 함.
        # 여기서는 단순화를 위해 보라색 라인으로 표시
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df['supertrend'], 
            line=dict(color='purple', width=1.5, dash='dot'), 
            name='SuperTrend'
        ))

    # 포지션 라인 표시 (Entry, SL)
    if pos_info:
        entry = float(pos_info['Entry'])
        sl = float(pos_info['SL'])
        side = pos_info['Side']
        
        entry_color = 'blue' if side in ['BUY', 'LONG'] else 'orange'
        fig.add_hline(y=entry, line_dash="solid", line_color=entry_color, annotation_text="ENTRY")
        fig.add_hline(y=sl, line_dash="dot", line_color="red", annotation_text="SL")

    fig.update_layout(
        height=450, 
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="white"), 
        xaxis_rangeslider_visible=False, 
        showlegend=True,
        title=dict(text=f"{symbol} (15m) - Titan Logic", x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 4. UI Layout (Main)
# ==========================================
positions = load_state_data()
history_df = load_history()
logs = read_logs()
config = load_json(CONFIG_FILE)

total_equity, free_money, unreal_pnl, active_df = calculate_metrics(config, positions)

# [Sidebar]
with st.sidebar:
    st.title("🛡️ PHALANX V3.0")
    st.caption("Titan Strategy Engine")
    
    # 시스템 상태 확인 (로그 파일 갱신 시간 기준)
    if os.path.exists(LOG_FILE):
        last_mod = time.time() - os.path.getmtime(LOG_FILE)
        if last_mod < 180: # 3분 이내 갱신이면 온라인
            st.success("🟢 ENGINE ONLINE")
        else:
            st.error(f"🔴 OFFLINE ({int(last_mod)}s ago)")
    else:
        st.error("🔴 NO LOG FILE")

    st.divider()
    st.metric("Total Equity", f"${total_equity:,.0f}")
    st.metric("Unrealized PnL", f"${unreal_pnl:,.2f}", 
              delta=f"{unreal_pnl/total_equity*100:.2f}%" if total_equity else "0%")
    
    st.divider()
    if st.button("🔄 Force Refresh"):
        st.rerun()
    
    auto_refresh = st.checkbox("Auto Refresh (15s)", value=True)

# [Main Tabs]
tab1, tab2, tab3 = st.tabs(["📊 Live Status", "📜 Trade History", "💻 System Logs"])

# Tab 1: Live Status
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Active Positions", f"{len(positions)} / {config.get('max_positions', 5)}")
    
    # 최근 전적 요약
    if not history_df.empty:
        # PnL 컬럼명 찾기 (대소문자 무관)
        pnl_col = next((c for c in history_df.columns if 'pnl' in c.lower() and '%' not in c), None)
        
        if pnl_col:
            recent = history_df.tail(20) # 최근 20개
            wins = len(recent[recent[pnl_col] > 0])
            win_rate = (wins / len(recent)) * 100
            cum_pnl = recent[pnl_col].sum()
            
            c2.metric("Recent Win Rate (20)", f"{win_rate:.1f}%")
            c3.metric("Recent PnL (20)", f"${cum_pnl:,.2f}", delta_color="normal")
    
    st.subheader("⚔️ Position Details")
    if not active_df.empty:
        # DataFrame 스타일링
        st.dataframe(
            active_df.style.format({
                "Entry": "{:.4f}", "Current": "{:.4f}", "SL": "{:.4f}", 
                "PnL($)": "{:.2f}", "ROE(%)": "{:.2f}%"
            }).map(lambda x: "color: #ff4b4b" if "SHORT" in str(x) else "color: #4bceff", subset=['Side']),
            use_container_width=True,
            hide_index=True
        )
        
        # 차트 시각화
        st.markdown("---")
        col_chart_sel, _ = st.columns([1, 3])
        with col_chart_sel:
            sel_symbol = st.selectbox("Select Symbol for Chart", active_df['Symbol'].tolist())
        
        if sel_symbol:
            row = active_df[active_df['Symbol'] == sel_symbol].iloc[0].to_dict()
            plot_minichart(sel_symbol, pos_info=row)
            
    else:
        st.info("현재 보유 포지션이 없습니다. Titan 엔진이 기회를 탐색 중입니다... 🦅")

# Tab 2: History
with tab2:
    st.subheader("📜 Execution History")
    if not history_df.empty:
        st.dataframe(
            history_df.iloc[::-1].head(100), # 최신순 정렬
            use_container_width=True, 
            height=600,
            hide_index=True
        )
    else:
        st.warning("저장된 거래 기록이 없습니다.")

# Tab 3: Logs
with tab3:
    st.subheader("💻 Engine Logs (Last 100 lines)")
    log_lines = logs[-100:] if logs else ["No logs found."]
    log_text = "".join(log_lines)
    st.text_area("Console Output", log_text, height=600, disabled=True)

# Auto Refresh Logic
if auto_refresh:
    time.sleep(15)
    st.rerun()