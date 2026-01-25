# dashboard.py
# =========================================================
# PHALANX Titan Dashboard (LiveEngine-compatible)
#
# ✅ LiveEngine(v9 DRY_RUN Multi-Track) 호환:
# - DRY_RUN 트랙별 분리 파일
#   - phalanx_state_{TRACK}.json
#   - trade_history_{TRACK}.csv
# - LIVE 파일
#   - phalanx_state.json
#   - (LIVE history는 엔진이 쓰는 경우에만 표시)
#
# FIX (이번 에러 포함):
# - Streamlit: use_container_width -> width='stretch'
# - pandas Styler.format(None) 에러 방지: na_rep="" + numeric coercion
# - ccxt exchange 객체는 cache_data 인자로 넘기지 않음 (UnhashableParamError 방지)
# =========================================================

import os
import json
import time
import warnings

import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta

warnings.filterwarnings("ignore", category=UserWarning, module="plotly")

# =========================
# 1) Page / Paths
# =========================
st.set_page_config(
    page_title="PHALANX Titan Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

current_file_path = os.path.abspath(__file__)
root_dir = os.path.dirname(current_file_path)

LOG_FILE = os.path.join(root_dir, "phalanx_live.log")
CONFIG_FILE = os.path.join(root_dir, "config.json")

# LIVE single-truth
STATE_FILE_LIVE = os.path.join(root_dir, "phalanx_state.json")

KST_TZ = "Asia/Seoul"

# =========================
# 2) Utils
# =========================
def load_json(filepath):
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def read_logs_tail(filepath, n=250):
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return lines[-n:]
    except Exception:
        return []

def file_fresh_seconds(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        return time.time() - os.path.getmtime(filepath)
    except Exception:
        return None

@st.cache_resource
def init_exchange(config):
    # DRY_RUN에서도 티커/차트용으로 사용 가능(키 없어도 작동)
    return ccxt.binance({
        "apiKey": config.get("api_key", ""),
        "secret": config.get("secret_key", ""),
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "future", "adjustForTimeDifference": True},
    })

def is_dry_run(config):
    return bool((config.get("system_settings") or {}).get("dry_run", False))

def get_max_positions(config):
    rs = config.get("risk_settings", {}) or {}
    return int(rs.get("max_open_positions", rs.get("max_positions", 5)))

def get_leverage(config):
    rs = config.get("risk_settings", {}) or {}
    try:
        return float(rs.get("leverage", 1))
    except Exception:
        return 1.0

def get_online_sec(config):
    try:
        return int(((config or {}).get("system_settings") or {}).get("dashboard_online_sec", 180))
    except Exception:
        return 180

def parse_last_heartbeat(log_lines):
    for line in reversed(log_lines or []):
        if "HEARTBEAT" in line:
            return line.strip()
    return ""

def _pick_first_existing(cols, candidates):
    cols_set = set(cols)
    for c in candidates:
        if c in cols_set:
            return c
    return None

def discover_dry_run_tracks(config):
    """
    LiveEngine: cfg["dry_run_tracks"]를 기반으로 트랙 목록 추출
    - 없으면 기본 A 트랙
    """
    tracks_cfg = (config.get("dry_run_tracks") or {})
    if not isinstance(tracks_cfg, dict) or not tracks_cfg:
        tracks_cfg = {"A": {"name": "baseline_15m", "entry_timeframe": "15m", "management_timeframe": "15m"}}

    out = []
    for tid, tc in tracks_cfg.items():
        tc = tc or {}
        out.append({
            "id": str(tid),
            "name": str(tc.get("name", tid)),
            "entry_tf": str(tc.get("entry_timeframe", "15m")),
            "manage_tf": str(tc.get("management_timeframe", "15m")),
            "state_file": os.path.join(root_dir, f"phalanx_state_{str(tid)}.json"),
            "history_file": os.path.join(root_dir, f"trade_history_{str(tid)}.csv"),
        })
    out = sorted(out, key=lambda x: x["id"])
    return out

def infer_engine_online_multi(config, candidates_files):
    """
    ONLINE 판정:
    - 후보 파일들(LOG/state/history 등) 중 하나라도 최근 online_sec 이내면 ONLINE
    """
    online_sec = get_online_sec(config)
    ages = []
    for fp in candidates_files:
        a = file_fresh_seconds(fp)
        if a is not None:
            ages.append(a)
    if not ages:
        return False, None
    age = min(ages)
    return (age < online_sec), age

def safe_styler(df: pd.DataFrame, fmt_map: dict | None = None):
    """
    pandas Styler.format에서 None/NaN 때문에 터지는 케이스 방어.
    - na_rep=""로 표시
    - fmt_map은 필요한 컬럼만 넣고, 없는 컬럼은 무시
    """
    if df is None or df.empty:
        return df

    fmt_map = fmt_map or {}
    # 존재하는 컬럼만
    fmt_map2 = {k: v for k, v in fmt_map.items() if k in df.columns}

    try:
        return df.style.format(fmt_map2, na_rep="")
    except Exception:
        # 최후 방어: 스타일 없이 반환
        return df

# =========================
# 3) Trade History (Continuity)
# =========================
def load_trade_history(filepath):
    """
    LiveEngine 스키마 기준 '정규화'만 수행.
    - 최소: dt/event/mode/symbol/side/price/amount/pnl/reason/cash/equity 생성
    """
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return pd.DataFrame()

    try:
        df = pd.read_csv(filepath, on_bad_lines="skip")
        if df is None or df.empty:
            return pd.DataFrame()

        cols = list(df.columns)

        # dt
        dt_col = _pick_first_existing(cols, ["dt", "time", "timestamp", "datetime"])
        if dt_col:
            df["dt"] = pd.to_datetime(df[dt_col], errors="coerce")
        else:
            df["dt"] = pd.NaT

        # event
        ev_col = _pick_first_existing(cols, ["event", "action", "type"])
        df["event"] = df[ev_col].astype(str) if ev_col else ""

        # mode
        mode_col = _pick_first_existing(cols, ["mode", "dry_run"])
        df["mode"] = df[mode_col].astype(str) if mode_col else ""

        # symbol
        sym_col = _pick_first_existing(cols, ["symbol", "sym"])
        df["symbol"] = df[sym_col].astype(str) if sym_col else ""

        # side
        side_col = _pick_first_existing(cols, ["side", "direction"])
        df["side"] = df[side_col].astype(str) if side_col else ""

        # price/amount/pnl (numeric coercion)
        price_col = _pick_first_existing(cols, ["price", "exec_price", "filled_price"])
        df["price"] = pd.to_numeric(df[price_col], errors="coerce") if price_col else pd.NA

        amt_col = _pick_first_existing(cols, ["amount", "qty", "filled_qty"])
        df["amount"] = pd.to_numeric(df[amt_col], errors="coerce") if amt_col else pd.NA

        pnl_col = _pick_first_existing(cols, ["pnl", "PnL", "pnl_usdt"])
        df["pnl"] = pd.to_numeric(df[pnl_col], errors="coerce") if pnl_col else pd.NA

        # reason
        reason_col = _pick_first_existing(cols, ["reason", "msg", "note"])
        df["reason"] = df[reason_col].astype(str) if reason_col else ""

        # cash/equity
        cash_col = _pick_first_existing(cols, ["cash", "free"])
        df["cash"] = pd.to_numeric(df[cash_col], errors="coerce") if cash_col else pd.NA

        eq_col = _pick_first_existing(cols, ["equity", "eq", "total"])
        df["equity"] = pd.to_numeric(df[eq_col], errors="coerce") if eq_col else pd.NA

        df = df.sort_values("dt", ascending=True, na_position="last").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()

def filter_history_since_last_boot(history_df, mode="DRY_RUN"):
    """
    모드별 마지막 BOOT 이후만 남김.
    BOOT이 없으면: 전체(해당 모드) 반환.
    """
    if history_df is None or history_df.empty:
        return pd.DataFrame()

    df = history_df.copy()

    if "mode" in df.columns:
        df["mode"] = df["mode"].astype(str).str.upper()
        df = df[df["mode"] == str(mode).upper()]

    if df.empty:
        return pd.DataFrame()

    df["event"] = df["event"].astype(str).str.upper()
    boots = df[df["event"] == "BOOT"]

    if boots.empty:
        return df

    t0 = boots["dt"].max()
    return df[df["dt"] >= t0]

def summarize_history(df_hist: pd.DataFrame):
    """
    세션 요약:
    - EXIT pnl 기준 승률/누적
    """
    if df_hist is None or df_hist.empty:
        return {
            "events": 0,
            "exits": 0,
            "win_rate_50": None,
            "pnl_sum": 0.0,
            "pnl_sum_50": 0.0,
            "last_event": None,
        }

    d = df_hist.copy()
    d["event_u"] = d["event"].astype(str).str.upper()

    exits = d[d["event_u"].str.contains("EXIT", na=False)]
    total_exits = int(len(exits))

    pnl_sum = 0.0
    if total_exits > 0 and "pnl" in exits.columns:
        pnl_sum = float(pd.to_numeric(exits["pnl"], errors="coerce").fillna(0).sum())

    recent = exits.tail(50) if total_exits > 0 else exits
    if total_exits > 0 and "pnl" in recent.columns and len(recent) > 0:
        rpnl = pd.to_numeric(recent["pnl"], errors="coerce").fillna(0)
        wins = int((rpnl > 0).sum())
        win_rate_50 = (wins / len(recent)) * 100.0 if len(recent) else None
        pnl_sum_50 = float(rpnl.sum())
    else:
        win_rate_50 = None
        pnl_sum_50 = 0.0

    last_event = None
    try:
        last_event = d.sort_values("dt", ascending=True).tail(1).iloc[0].to_dict()
        last_event.pop("event_u", None)
    except Exception:
        last_event = None

    return {
        "events": int(len(d)),
        "exits": total_exits,
        "win_rate_50": win_rate_50,
        "pnl_sum": pnl_sum,
        "pnl_sum_50": pnl_sum_50,
        "last_event": last_event,
    }

# =========================
# 4) Chart (cached, no exchange param)
# =========================
@st.cache_data(ttl=60)
def fetch_chart_data(symbol, timeframe="15m", limit=240):
    """
    ccxt exchange는 세션에서 접근 (cache_data 인자로 넘기지 않음)
    """
    ex = st.session_state.get("_exchange")
    if ex is None:
        return None

    try:
        clean = str(symbol).split(":")[0]
        ohlcv = ex.fetch_ohlcv(clean, timeframe=timeframe, limit=limit)
        if not ohlcv:
            return None

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        dt = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(KST_TZ)
        df["datetime"] = dt.dt.tz_localize(None)
        df = df.set_index("datetime").drop(columns=["timestamp"])

        df["ema200"] = ta.ema(df["close"], length=200)
        st_out = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
        if st_out is not None and len(st_out.columns) >= 2:
            df["supertrend"] = st_out.iloc[:, 0]
            df["st_dir"] = st_out.iloc[:, 1]

        return df
    except Exception:
        return None

def plot_minichart(symbol, pos_info=None):
    df = fetch_chart_data(symbol, timeframe="15m", limit=300)
    if df is None or df.empty:
        st.error(f"차트 데이터 로드 실패: {symbol}")
        return

    plot_df = df.tail(120)
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df["open"],
        high=plot_df["high"],
        low=plot_df["low"],
        close=plot_df["close"],
        name="Price",
    ))

    if "ema200" in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df.index,
            y=plot_df["ema200"],
            name="EMA200",
            line=dict(width=1),
        ))

    if "supertrend" in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df.index,
            y=plot_df["supertrend"],
            name="SuperTrend",
            line=dict(width=1.5, dash="dot"),
        ))

    title = f"{symbol} (15m)"
    if pos_info:
        try:
            entry = float(pos_info.get("Entry", 0) or 0)
            sl = float(pos_info.get("SL", 0) or 0)
            side = str(pos_info.get("Side", "")).upper()
            last_px = float(pos_info.get("Current", 0) or 0)

            if entry > 0:
                fig.add_hline(y=entry, line_dash="solid", annotation_text="ENTRY")
            if sl > 0:
                fig.add_hline(y=sl, line_dash="dot", annotation_text="SL")
            if last_px > 0:
                fig.add_hline(y=last_px, line_dash="dash", annotation_text="NOW")

            title = f"{symbol} (15m) | {side}"
        except Exception:
            pass

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=450,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis_rangeslider_visible=False,
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# 5) State Loaders
# =========================
def load_state_live(config):
    state = load_json(STATE_FILE_LIVE)
    positions = (state.get("positions", {}) or {}) if isinstance(state, dict) else {}

    return {
        "positions": positions,
        "paper_equity0": None,
        "paper_cash": None,
        "paper_equity": None,
        "cooldowns": state.get("cooldowns", {}) if isinstance(state, dict) else {},
        "consecutive_losses": state.get("consecutive_losses", {}) if isinstance(state, dict) else {},
    }

def load_state_track(track_meta, config):
    fp = track_meta["state_file"]
    state = load_json(fp)
    if not isinstance(state, dict):
        state = {}

    positions = state.get("positions", {}) or {}

    cfg_p0 = float(((config.get("system_settings") or {}).get("paper_equity", 10000.0)) or 10000.0)
    paper_equity0 = float(state.get("paper_equity0", cfg_p0))
    paper_cash = float(state.get("paper_cash", paper_equity0))
    paper_equity = float(state.get("paper_equity", paper_equity0))

    cooldowns = state.get("cooldowns", {}) or {}
    consecutive_losses = state.get("consecutive_losses", {}) or {}

    return {
        "positions": positions,
        "paper_equity0": paper_equity0,
        "paper_cash": paper_cash,
        "paper_equity": paper_equity,
        "cooldowns": cooldowns,
        "consecutive_losses": consecutive_losses,
    }

# =========================
# 6) Metrics
# =========================
def calculate_metrics_live(config, state, exchange):
    lev = get_leverage(config)
    positions = state.get("positions", {}) or {}

    total_equity = 0.0
    free_money = 0.0
    try:
        bal = exchange.fetch_balance()
        usdt = bal.get("USDT", {}) or {}
        total_equity = float(usdt.get("total", 0))
        free_money = float(usdt.get("free", 0))
    except Exception:
        total_equity = 0.0
        free_money = 0.0

    unrealized_pnl = 0.0
    rows = []

    for symbol, pos in positions.items():
        try:
            side = str(pos.get("side", "LONG")).upper()
            entry = float(pos.get("entry_price", 0) or 0)
            amt = float(pos.get("amount", 0) or 0)
            sl = float(pos.get("sl", 0) or 0)
            tp1 = pos.get("tp1", 0)
            tp1_hit = bool(pos.get("tp1_hit", False))
            margin = float(pos.get("margin", 0) or 0)

            try:
                t = exchange.fetch_ticker(symbol)
                current_price = float(t.get("last") or t.get("close") or entry)
            except Exception:
                current_price = entry

            if amt <= 0 or entry <= 0:
                continue

            if side in ("BUY", "LONG"):
                pnl = (current_price - entry) * amt
                roe = ((current_price - entry) / entry) * 100.0 * lev
            else:
                pnl = (entry - current_price) * amt
                roe = ((entry - current_price) / entry) * 100.0 * lev

            unrealized_pnl += pnl

            rows.append({
                "Symbol": symbol,
                "Side": side,
                "Qty": amt,
                "Entry": entry,
                "Current": current_price,
                "PnL($)": pnl,
                "ROE(%)": roe,
                "Margin": margin,
                "SL": sl,
                "TP1": "Done" if tp1_hit else (f"{float(tp1):.6f}" if tp1 not in (None, "") else ""),
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="PnL($)", ascending=False).reset_index(drop=True)

    return total_equity, free_money, unrealized_pnl, df

def calculate_metrics_track(config, state, exchange):
    lev = get_leverage(config)
    positions = state.get("positions", {}) or {}

    total_equity = float(state.get("paper_equity", 10000.0))
    free_money = float(state.get("paper_cash", total_equity))

    unrealized_pnl = 0.0
    rows = []

    for symbol, pos in positions.items():
        try:
            side = str(pos.get("side", "LONG")).upper()
            entry = float(pos.get("entry_price", 0) or 0)
            amt = float(pos.get("amount", 0) or 0)
            sl = float(pos.get("sl", 0) or 0)
            tp1 = pos.get("tp1", 0)
            tp1_hit = bool(pos.get("tp1_hit", False))
            margin = float(pos.get("margin", 0) or 0)

            try:
                t = exchange.fetch_ticker(symbol)
                current_price = float(t.get("last") or t.get("close") or entry)
            except Exception:
                current_price = entry

            if amt <= 0 or entry <= 0:
                continue

            if side in ("BUY", "LONG"):
                pnl = (current_price - entry) * amt
                roe = ((current_price - entry) / entry) * 100.0 * lev
            else:
                pnl = (entry - current_price) * amt
                roe = ((entry - current_price) / entry) * 100.0 * lev

            unrealized_pnl += pnl

            rows.append({
                "Symbol": symbol,
                "Side": side,
                "Qty": amt,
                "Entry": entry,
                "Current": current_price,
                "PnL($)": pnl,
                "ROE(%)": roe,
                "Margin": margin,
                "SL": sl,
                "TP1": "Done" if tp1_hit else (f"{float(tp1):.6f}" if tp1 not in (None, "") else ""),
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="PnL($)", ascending=False).reset_index(drop=True)

    return total_equity, free_money, unrealized_pnl, df

# =========================
# 7) Load Data
# =========================
config = load_json(CONFIG_FILE)
exchange = init_exchange(config)
st.session_state["_exchange"] = exchange  # ✅ cache_data에서 접근

dry = is_dry_run(config)
max_pos = get_max_positions(config)
lev = get_leverage(config)

tracks = discover_dry_run_tracks(config) if dry else []

# ONLINE 판정 후보 파일 모음
online_files = [LOG_FILE, CONFIG_FILE]
if dry:
    for t in tracks:
        online_files.append(t["state_file"])
        online_files.append(t["history_file"])
else:
    online_files.append(STATE_FILE_LIVE)

engine_online, age_sec = infer_engine_online_multi(config, online_files)

log_lines = read_logs_tail(LOG_FILE, n=250)
heartbeat_line = parse_last_heartbeat(log_lines)

# =========================
# 8) Sidebar
# =========================
with st.sidebar:
    st.title("🛡️ PHALANX")

    mode_text = "DRY_RUN" if dry else "LIVE"
    st.caption(f"Mode: **{mode_text}** | Leverage: **{lev:g}x** | MaxPos: **{max_pos}**")

    if engine_online:
        st.success("🟢 ENGINE ONLINE")
    else:
        if age_sec is None:
            st.error("🔴 OFFLINE (no log/state/history)")
        else:
            st.error(f"🔴 OFFLINE ({int(age_sec)}s ago)")

    if heartbeat_line:
        st.caption("Last Heartbeat (log)")
        st.code(heartbeat_line, language="text")

    st.divider()

    selected_track_id = None
    selected_track_meta = None

    if dry:
        labels = []
        track_map = {}
        for t in tracks:
            lab = f"{t['id']} | {t['name']} | entry={t['entry_tf']} manage={t['manage_tf']}"
            labels.append(lab)
            track_map[lab] = t

        if labels:
            sel = st.selectbox("DRY_RUN Track", labels, index=0)
            selected_track_meta = track_map.get(sel)
            selected_track_id = selected_track_meta["id"] if selected_track_meta else None
        else:
            st.warning("dry_run_tracks가 없거나 파싱 실패")
            selected_track_id = None
            selected_track_meta = None

    st.divider()

    if st.button("🔄 Force Refresh"):
        st.cache_data.clear()
        st.rerun()

    auto_refresh = st.checkbox("Auto Refresh (15s)", value=True)

# =========================
# 9) State / History Load
# =========================
if dry:
    track_states = {}
    track_hist_all = {}
    track_hist_session = {}
    track_hist_summary = {}

    for t in tracks:
        stt = load_state_track(t, config)
        track_states[t["id"]] = stt

        h_all = load_trade_history(t["history_file"])
        track_hist_all[t["id"]] = h_all

        h_sess = filter_history_since_last_boot(h_all, mode="DRY_RUN")
        track_hist_session[t["id"]] = h_sess

        track_hist_summary[t["id"]] = summarize_history(h_sess)

    if selected_track_id is None and tracks:
        selected_track_id = tracks[0]["id"]
        selected_track_meta = tracks[0]

    sel_state = track_states.get(selected_track_id, {})
    sel_hist_all = track_hist_all.get(selected_track_id, pd.DataFrame())
    sel_hist = track_hist_session.get(selected_track_id, pd.DataFrame())
    sel_hist_summary = track_hist_summary.get(selected_track_id, summarize_history(pd.DataFrame()))

    total_equity, free_money, unreal_pnl, active_df = calculate_metrics_track(config, sel_state, exchange)

else:
    state = load_state_live(config)
    total_equity, free_money, unreal_pnl, active_df = calculate_metrics_live(config, state, exchange)

    legacy_history_file = os.path.join(root_dir, "trade_history.csv")
    hist_df_all = load_trade_history(legacy_history_file) if os.path.exists(legacy_history_file) else pd.DataFrame()
    hist_df = filter_history_since_last_boot(hist_df_all, mode="LIVE")
    hist_summary = summarize_history(hist_df)

# =========================
# 10) Header Metrics (Top)
# =========================
st.title("🛡️ PHALANX Titan Dashboard")

if dry:
    top_c1, top_c2, top_c3, top_c4, top_c5 = st.columns([1, 1, 1, 1, 2])
    top_c1.metric("Mode", "DRY_RUN")
    top_c2.metric("Track", f"{selected_track_id}")
    top_c3.metric("Paper Equity", f"${total_equity:,.2f}")
    top_c4.metric("Paper Cash", f"${free_money:,.2f}")
    top_c5.metric("Unrealized PnL", f"${unreal_pnl:,.2f}")
else:
    top_c1, top_c2, top_c3, top_c4, top_c5 = st.columns([1, 1, 1, 1, 2])
    top_c1.metric("Mode", "LIVE")
    top_c2.metric("Engine", "ONLINE" if engine_online else "OFFLINE")
    top_c3.metric("Total Equity (USDT)", f"${total_equity:,.2f}")
    top_c4.metric("Free (USDT)", f"${free_money:,.2f}")
    top_c5.metric("Unrealized PnL", f"${unreal_pnl:,.2f}")

st.divider()

# =========================
# 11) Tabs
# =========================
if dry:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Status", "🧪 Track Compare", "📜 Trade History", "💻 Logs"])
else:
    tab1, tab3, tab4 = st.tabs(["📊 Live Status", "📜 Trade History", "💻 Logs"])

# -------------------------
# Tab 1: Live Status
# -------------------------
with tab1:
    if dry:
        st.subheader(f"⚔️ Position Details (Track {selected_track_id})")

        s_positions = sel_state.get("positions", {}) or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active Positions", f"{len(s_positions)} / {max_pos}")
        c2.metric("History Events (session)", f"{sel_hist_summary['events']}")
        c3.metric("Exits (session)", f"{sel_hist_summary['exits']}")
        c4.metric("PnL Sum (session)", f"{sel_hist_summary['pnl_sum']:+.2f}")

        if sel_hist_summary.get("win_rate_50") is not None:
            st.caption(
                f"WinRate(Last 50 exits): {sel_hist_summary['win_rate_50']:.1f}% | "
                f"PnL(Last 50 exits): {sel_hist_summary['pnl_sum_50']:+.2f}"
            )

        if sel_hist_summary.get("last_event"):
            with st.expander("Last Event (session)", expanded=False):
                st.json(sel_hist_summary["last_event"])

        st.markdown("---")

        if active_df is not None and not active_df.empty:
            show_df = active_df.copy()

            st.dataframe(
                safe_styler(show_df, {
                    "Qty": "{:.6f}",
                    "Entry": "{:.6f}",
                    "Current": "{:.6f}",
                    "PnL($)": "{:.2f}",
                    "ROE(%)": "{:.2f}",
                    "Margin": "{:.2f}",
                    "SL": "{:.6f}",
                }),
                width="stretch",
                hide_index=True
            )

            st.markdown("---")
            col_sel, col_chart = st.columns([1, 3], gap="large")

            with col_sel:
                sel_symbol = st.selectbox("Select Symbol for Chart", show_df["Symbol"].tolist(), index=0)
                row = show_df[show_df["Symbol"] == sel_symbol].iloc[0].to_dict()
                st.write("Selected")
                st.json(row)

            with col_chart:
                plot_minichart(sel_symbol, pos_info=row)
        else:
            st.info("현재 보유 포지션이 없습니다.")

    else:
        st.subheader("⚔️ Position Details (LIVE State)")
        positions = state.get("positions", {}) or {}

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active Positions", f"{len(positions)} / {max_pos}")
        c2.metric("Mode", "LIVE")
        c3.metric("Engine", "ONLINE" if engine_online else "OFFLINE")
        c4.metric("Unrealized PnL", f"${unreal_pnl:,.2f}")

        st.markdown("---")

        if active_df is not None and not active_df.empty:
            show_df = active_df.copy()

            st.dataframe(
                safe_styler(show_df, {
                    "Qty": "{:.6f}",
                    "Entry": "{:.6f}",
                    "Current": "{:.6f}",
                    "PnL($)": "{:.2f}",
                    "ROE(%)": "{:.2f}",
                    "Margin": "{:.2f}",
                    "SL": "{:.6f}",
                }),
                width="stretch",
                hide_index=True
            )

            st.markdown("---")
            col_sel, col_chart = st.columns([1, 3], gap="large")

            with col_sel:
                sel_symbol = st.selectbox("Select Symbol for Chart", show_df["Symbol"].tolist(), index=0)
                row = show_df[show_df["Symbol"] == sel_symbol].iloc[0].to_dict()
                st.write("Selected")
                st.json(row)

            with col_chart:
                plot_minichart(sel_symbol, pos_info=row)
        else:
            st.info("현재 보유 포지션이 없습니다.")

# -------------------------
# Tab 2: Track Compare (DRY_RUN only)
# -------------------------
if dry:
    with tab2:
        st.subheader("🧪 DRY_RUN Track Compare")

        rows = []
        for t in tracks:
            tid = t["id"]
            stt = track_states.get(tid, {})
            summ = track_hist_summary.get(tid, {}) or {}

            # WinRate50는 None 가능 -> 그대로 두되 Styler에서 na_rep으로 출력
            rows.append({
                "Track": tid,
                "Name": t.get("name", tid),
                "EntryTF": t.get("entry_tf", ""),
                "ManageTF": t.get("manage_tf", ""),
                "Positions": int(len((stt.get("positions", {}) or {}))),
                "PaperCash": float(stt.get("paper_cash", 0.0) or 0.0),
                "PaperEquity": float(stt.get("paper_equity", 0.0) or 0.0),
                "SessionEvents": int(summ.get("events", 0) or 0),
                "SessionExits": int(summ.get("exits", 0) or 0),
                "SessionPnL": float(summ.get("pnl_sum", 0.0) or 0.0),
                "WinRate50": float(summ.get("win_rate_50")) if summ.get("win_rate_50") is not None else None,
                "PnL50": float(summ.get("pnl_sum_50", 0.0) or 0.0),
            })

        df_cmp = pd.DataFrame(rows)
        if df_cmp.empty:
            st.warning("트랙 데이터가 없습니다.")
        else:
            st.dataframe(
                safe_styler(df_cmp, {
                    "PaperCash": "{:,.2f}",
                    "PaperEquity": "{:,.2f}",
                    "SessionPnL": "{:+.2f}",
                    "WinRate50": "{:.1f}",
                    "PnL50": "{:+.2f}",
                }),
                width="stretch",
                hide_index=True
            )

            st.caption("※ Entry 로직은 동일(15m 1회 계산)이며, 차이는 '관리 주기(manage_tf)'로 발생합니다.")

# -------------------------
# Tab 3: Trade History
# -------------------------
with tab3:
    st.subheader("📜 Trade History (Continuity)")

    colA, colB, colC = st.columns([1, 1, 2])

    with colA:
        show_session_only = st.checkbox("Show session only (since last BOOT)", value=True)

    with colB:
        show_tail = st.number_input("Rows", min_value=50, max_value=5000, value=500, step=50)

    if dry:
        history_fp = selected_track_meta["history_file"] if selected_track_meta else None
        st.caption(f"Source: {os.path.basename(history_fp) if history_fp else 'N/A'}")

        view_df = sel_hist if show_session_only else sel_hist_all

        if view_df is not None and not view_df.empty:
            d = view_df.copy().sort_values("dt", ascending=False, na_position="last")
            st.dataframe(d.head(int(show_tail)), width="stretch", height=720, hide_index=True)
        else:
            st.error(
                "trade_history_{TRACK}.csv 가 없거나 비어 있습니다.\n\n"
                "✅ LiveEngine이 BOOT/ENTRY/EXIT/UPDATE_SL/HEARTBEAT 를 트랙별 history 파일에 append 해야 합니다."
            )
    else:
        legacy_fp = os.path.join(root_dir, "trade_history.csv")
        st.caption(f"Source: {os.path.basename(legacy_fp)} (있을 때만 표시)")

        view_df = hist_df if show_session_only else hist_df_all

        if view_df is not None and not view_df.empty:
            d = view_df.copy().sort_values("dt", ascending=False, na_position="last")
            st.dataframe(d.head(int(show_tail)), width="stretch", height=720, hide_index=True)
        else:
            st.info("LIVE 모드에서 history 파일이 없거나 비어 있습니다. (state 기반 Live Status는 정상 표시)")

# -------------------------
# Tab 4: Logs
# -------------------------
with tab4:
    st.subheader("💻 Engine Logs (tail)")
    if log_lines:
        st.text_area("Log Tail", "".join(log_lines), height=680, disabled=True)
    else:
        st.warning("phalanx_live.log 파일이 없습니다. (콘솔 로그를 파일로 리다이렉트하면 표시됩니다.)")

# =========================
# 12) Auto refresh
# =========================
if auto_refresh:
    time.sleep(15)
    st.rerun()
