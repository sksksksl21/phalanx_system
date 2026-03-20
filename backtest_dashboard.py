import streamlit as st
import pandas as pd
import plotly.express as px
import os
import plotly.graph_objects as go
import re
import pickle
import glob

st.set_page_config(page_title="Phalanx Backtest Dashboard", layout="wide")

DEFAULT_INITIAL_EQUITY = 10000.0

REQUIRED_TRADE_COLUMNS = ["Datetime", "Symbol", "Side", "Type", "Price", "Amount", "PnL", "Cash", "Equity", "Reason"]
REQUIRED_EQUITY_COLUMNS = ["Datetime", "Equity"]


def _abs_path(filename: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, filename)


@st.cache_data(show_spinner=False)
def load_trade_log(path: str):
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_TRADE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Trade CSV missing columns: {missing}")

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    for col in ["Price", "Amount", "PnL", "Cash", "Equity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Datetime", "Type", "Symbol"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    return df



@st.cache_data(show_spinner=False)
def load_mtm_curve(path: str):
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_EQUITY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Equity CSV missing columns: {missing}")

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df["Equity"] = pd.to_numeric(df["Equity"], errors="coerce")

    df = df.dropna(subset=["Datetime", "Equity"]).sort_values("Datetime").reset_index(drop=True)
    return df
def _sym_to_fname(sym: str) -> str:
    # "1000PEPE/USDT:USDT" -> "1000PEPE_USDT_USDT"
    return str(sym).replace("/", "_").replace(":", "_")

@st.cache_data(show_spinner=False)
def load_ohlcv_csv_best_effort(root_dir: str, symbol: str, timeframe: str):
    """
    CSV 탐색(기존 로직 유지 + glob 확장)
    return: (df, used_path) or (None, [tried_paths])
    """
    fname = _sym_to_fname(symbol)
    tried = [
        os.path.join(root_dir, f"ohlcv_{fname}_{timeframe}.csv"),
        os.path.join(root_dir, f"{fname}_{timeframe}.csv"),
        os.path.join(root_dir, "ohlcv", f"ohlcv_{fname}_{timeframe}.csv"),
        os.path.join(root_dir, "data", f"ohlcv_{fname}_{timeframe}.csv"),
    ]

    # glob로도 한번 더 (혹시 파일명이 조금 다를 때)
    glob_patterns = [
        os.path.join(root_dir, f"*{fname}*{timeframe}*.csv"),
        os.path.join(root_dir, "ohlcv", f"*{fname}*{timeframe}*.csv"),
        os.path.join(root_dir, "data", f"*{fname}*{timeframe}*.csv"),
    ]
    for gp in glob_patterns:
        tried.extend(glob.glob(gp))

    # 중복 제거
    tried_unique = []
    seen = set()
    for p in tried:
        if p and p not in seen:
            tried_unique.append(p)
            seen.add(p)

    for p in tried_unique:
        if os.path.exists(p):
            df = pd.read_csv(p)
            # 표준 컬럼 기대: timestamp/open/high/low/close/volume or Datetime/open/high/low/close/volume
            if "Datetime" in df.columns:
                df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
                df = df.dropna(subset=["Datetime"])
                df = df.sort_values("Datetime").reset_index(drop=True)
            elif "timestamp" in df.columns:
                # ms timestamp 가정
                df["Datetime"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
                df = df.dropna(subset=["Datetime"])
                df = df.sort_values("Datetime").reset_index(drop=True)
            else:
                # 최소한 Datetime 만들어보기(첫 컬럼이 timestamp일 수도)
                pass
            return df, p

    return None, tried_unique

@st.cache_data(show_spinner=False)
def load_ohlcv_from_cache_pkl(cache_path: str, symbol: str, timeframe: str):
    """
    market_data_cache_30d.pkl에서 OHLCV 꺼내기 (구조가 달라도 최대한 best-effort)
    return: (df, source_desc) or (None, reason)
    """
    if not os.path.exists(cache_path):
        return None, f"cache_not_found:{cache_path}"

    try:
        with open(cache_path, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        return None, f"cache_load_failed:{e}"

    # 1) 가장 흔한 형태: dict[symbol] = df
    if isinstance(obj, dict):
        # case A: dict[(symbol,timeframe)] = df
        key1 = (symbol, timeframe)
        if key1 in obj and isinstance(obj[key1], pd.DataFrame):
            df = obj[key1].copy()
            return _normalize_ohlcv_df(df), f"pkl_key:{key1}"

        # case B: dict[symbol] = df
        if symbol in obj and isinstance(obj[symbol], pd.DataFrame):
            df = obj[symbol].copy()
            return _normalize_ohlcv_df(df), f"pkl_key:{symbol}"

        # case C: dict[timeframe][symbol] = df  또는 dict[symbol][timeframe] = df
        if timeframe in obj and isinstance(obj[timeframe], dict):
            inner = obj[timeframe]
            if symbol in inner and isinstance(inner[symbol], pd.DataFrame):
                df = inner[symbol].copy()
                return _normalize_ohlcv_df(df), f"pkl_key:{timeframe}->{symbol}"

        if symbol in obj and isinstance(obj[symbol], dict):
            inner = obj[symbol]
            if timeframe in inner and isinstance(inner[timeframe], pd.DataFrame):
                df = inner[timeframe].copy()
                return _normalize_ohlcv_df(df), f"pkl_key:{symbol}->{timeframe}"

    return None, f"cache_structure_unknown:type={type(obj)}"

def _normalize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    캔들 차트용으로 Datetime/open/high/low/close/volume 컬럼을 최대한 맞춘다.
    """
    out = df.copy()

    # timestamp -> Datetime
    if "Datetime" not in out.columns:
        if "timestamp" in out.columns:
            # ms timestamp 가정
            out["Datetime"] = pd.to_datetime(out["timestamp"], unit="ms", errors="coerce")
        elif "date" in out.columns:
            out["Datetime"] = pd.to_datetime(out["date"], errors="coerce")
        elif "time" in out.columns:
            out["Datetime"] = pd.to_datetime(out["time"], errors="coerce")

    if "Datetime" in out.columns:
        out["Datetime"] = pd.to_datetime(out["Datetime"], errors="coerce")
        out = out.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)

    # 컬럼명 소문자 표준화 시도
    lower_map = {c: c.lower() for c in out.columns}
    out.rename(columns=lower_map, inplace=True)

    # close/open/high/low/volume 최소 확보(이미 있으면 그대로)
    # (여기서 없으면 캔들 탭에서 안내 띄우면 됨)
    return out

def extract_sl_from_reason(reason: str):
    if not isinstance(reason, str):
        return np.nan
    m = re.search(r"(TRAILING|UPDATE_SL_APPLY):([0-9]*\.?[0-9]+)", reason)
    if not m:
        return np.nan
    return float(m.group(2))

def compute_mdd_from_equity_series(df_equity: pd.DataFrame, equity_col: str):
    if df_equity is None or df_equity.empty:
        return None

    temp = df_equity[["Datetime", equity_col]].dropna().copy()
    if temp.empty:
        return None

    temp["peak"] = temp[equity_col].cummax()
    temp["dd"] = (temp[equity_col] - temp["peak"]) / temp["peak"] * 100.0
    return float(temp["dd"].min())


def compute_exit_kpis(trade_df: pd.DataFrame):
    if trade_df is None or trade_df.empty:
        return None

    exits = trade_df[trade_df["Type"] == "EXIT"].copy()
    if exits.empty:
        return None

    wins = exits[exits["PnL"] > 0]
    losses = exits[exits["PnL"] < 0]

    total_trades = len(exits)
    win_rate = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
    total_pnl = float(exits["PnL"].sum())

    pos_sum = float(wins["PnL"].sum()) if not wins.empty else 0.0
    neg_sum = float(losses["PnL"].sum()) if not losses.empty else 0.0

    if neg_sum == 0.0:
        profit_factor = float("inf") if pos_sum > 0 else 0.0
    else:
        profit_factor = abs(pos_sum / neg_sum)

    return {
        "total_trades": int(total_trades),
        "win_rate": float(win_rate),
        "total_pnl": float(total_pnl),
        "profit_factor": float(profit_factor),
        "win_count": int(len(wins)),
        "loss_count": int(len(losses)),
    }

@st.cache_data(show_spinner=False)
def load_market_cache_pkl(path: str):
    """
    market_data_cache_30d.pkl 로드.
    기대 형태(가장 흔함):
      - dict: { "BTC/USDT:USDT": df, ... }
      - 또는 dict 내부에 data_map 같은 키가 있음
    """
    if not os.path.exists(path):
        return None

    obj = pd.read_pickle(path)

    # case A: 바로 dict(symbol->df)
    if isinstance(obj, dict):
        return obj

    # case B: wrapper dict with key candidates
    if isinstance(obj, dict) is False and hasattr(obj, "get"):
        for k in ("data_map", "market_data", "data", "ohlcv"):
            try:
                v = obj.get(k, None)
                if isinstance(v, dict):
                    return v
            except Exception:
                pass

    # 못 찾으면 원본 그대로 반환(디버그용)
    return obj


def get_ohlcv_from_cache(cache_obj, symbol: str):
    """
    cache에서 symbol의 OHLCV df를 찾아 표준 컬럼으로 정리해서 반환.
    표준 컬럼: Datetime, open, high, low, close, volume
    """
    if cache_obj is None:
        return None

    # 1) 심볼 키 그대로
    df = None
    if isinstance(cache_obj, dict):
        df = cache_obj.get(symbol)

        # 2) 혹시 "BTC/USDT" 형태로 저장되어 있으면 변환해서 재시도
        if df is None:
            alt = symbol.replace(":USDT", "")
            df = cache_obj.get(alt)

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None

    out = df.copy()

    # timestamp / Datetime 처리
    if "Datetime" in out.columns:
        out["Datetime"] = pd.to_datetime(out["Datetime"], errors="coerce")
    elif "timestamp" in out.columns:
        # ms 가정(네 executor/엔진 구조상 대부분 ms)
        out["Datetime"] = pd.to_datetime(out["timestamp"], unit="ms", errors="coerce")
    else:
        # 인덱스가 datetime인 형태도 흔함
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={"index": "Datetime"})
        else:
            return None

    # 컬럼명 정규화(혹시 대문자/다른 명칭이면 여기서 맞춰줌)
    col_map = {}
    for c in out.columns:
        lc = str(c).lower()
        if lc in ("open", "o"):
            col_map[c] = "open"
        elif lc in ("high", "h"):
            col_map[c] = "high"
        elif lc in ("low", "l"):
            col_map[c] = "low"
        elif lc in ("close", "c"):
            col_map[c] = "close"
        elif lc in ("volume", "vol", "v"):
            col_map[c] = "volume"
    out = out.rename(columns=col_map)

    need = ["Datetime", "open", "high", "low", "close"]
    if any(n not in out.columns for n in need):
        return None

    # 숫자형 변환
    for col in ["open", "high", "low", "close", "volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["Datetime", "open", "high", "low", "close"])
    out = out.sort_values("Datetime").reset_index(drop=True)
    return out


def _symbol_to_fname(symbol: str) -> str:
    # 파일명에 못 쓰는 문자 제거/치환
    s = str(symbol)
    s = s.replace(":", "_")
    s = s.replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\.]", "_", s)
    return s

@st.cache_data(show_spinner=False)
def load_ohlcv_for_symbol(symbol: str, timeframe: str = "15m"):
    """
    로컬 OHLCV CSV를 읽어 캔들 데이터로 쓴다.
    우선순위로 여러 패턴을 탐색.
    - 같은 폴더 기준:
      1) ohlcv_{SYMBOL}_{TF}.csv
      2) {SYMBOL}_{TF}.csv
      3) ohlcv/{SYMBOL}_{TF}.csv
      4) data/ohlcv_{SYMBOL}_{TF}.csv
    CSV 컬럼 허용:
      A) Datetime, open, high, low, close, volume
      B) timestamp(ms), open, high, low, close, volume
    """
    base = os.path.dirname(os.path.abspath(__file__))
    sym = _symbol_to_fname(symbol)
    tf = str(timeframe)

    candidates = [
        os.path.join(base, f"ohlcv_{sym}_{tf}.csv"),
        os.path.join(base, f"{sym}_{tf}.csv"),
        os.path.join(base, "ohlcv", f"{sym}_{tf}.csv"),
        os.path.join(base, "data", f"ohlcv_{sym}_{tf}.csv"),
    ]

    path = None
    for p in candidates:
        if os.path.exists(p):
            path = p
            break

    if path is None:
        return None, candidates

    df = pd.read_csv(path)

    # timestamp 형태 대응
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    elif "timestamp" in df.columns:
        # ms 가정
        df["Datetime"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    else:
        raise ValueError(f"OHLCV CSV missing Datetime or timestamp: {path}")

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"OHLCV CSV missing column '{col}': {path}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Datetime", "open", "high", "low", "close"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    return df, [path]


def build_sl_segments_from_trades(df_sym: pd.DataFrame):
    """
    SL이 시간에 따라 바뀌는(ENTRY/UPDATE_SL) 것을 계단식 구간으로 만든다.
    return: list of (t0, t1, sl)
    """
    if df_sym is None or df_sym.empty:
        return []

    segs = []
    in_trade = False
    cur_sl = None
    last_t = None

    for _, r in df_sym.sort_values("Datetime").iterrows():
        typ = str(r.get("Type", "")).upper()
        t = r["Datetime"]

        sl = r.get("sl", None)
        try:
            sl = float(sl) if pd.notna(sl) else None
        except Exception:
            sl = None

        if typ == "ENTRY":
            in_trade = True
            cur_sl = sl
            last_t = t

        elif typ == "UPDATE_SL" and in_trade:
            if cur_sl is not None and last_t is not None and t > last_t:
                segs.append((last_t, t, cur_sl))
            if sl is not None:
                cur_sl = sl
            last_t = t

        elif typ == "EXIT" and in_trade:
            if cur_sl is not None and last_t is not None and t > last_t:
                segs.append((last_t, t, cur_sl))
            in_trade = False
            cur_sl = None
            last_t = None

    return segs


import plotly.graph_objects as go
import pandas as pd
import numpy as np

def make_candle_with_trades(ohlcv_view: pd.DataFrame, df_sym: pd.DataFrame, symbol: str):
    import re
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    def _extract_sl(reason: str):
        if not isinstance(reason, str):
            return np.nan
        # Prefer APPLY if present, otherwise TRAILING
        m = re.search(r"UPDATE_SL_APPLY:([0-9]*\.?[0-9]+)", reason)
        if m:
            return float(m.group(1))
        m = re.search(r"TRAILING:([0-9]*\.?[0-9]+)", reason)
        if m:
            return float(m.group(1))
        return np.nan

    # -----------------------
    # 1) OHLCV normalize
    # -----------------------
    df = ohlcv_view.copy()
    if "Datetime" not in df.columns:
        raise ValueError("ohlcv_view must contain 'Datetime' column")

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime")

    rename_map = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc == "open":  rename_map[c] = "Open"
        if lc == "high":  rename_map[c] = "High"
        if lc == "low":   rename_map[c] = "Low"
        if lc == "close": rename_map[c] = "Close"
    df.rename(columns=rename_map, inplace=True)

    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            raise ValueError(f"ohlcv_view missing OHLC column: {col}")

    # -----------------------
    # 2) Base figure
    # -----------------------
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["Datetime"],
            open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="Candles",
            showlegend=False,
            increasing_line_width=0.5,
            decreasing_line_width=0.5,
            increasing_fillcolor="rgba(0,255,140,0.9)",
            decreasing_fillcolor="rgba(255,90,90,0.9)"
        )
    )

    # -----------------------
    # 3) Events normalize
    # -----------------------
    ev = df_sym.copy()
    ev["Datetime"] = pd.to_datetime(ev["Datetime"], errors="coerce")
    ev = ev.dropna(subset=["Datetime", "Type"]).sort_values("Datetime")

    if "Side" in ev.columns:
        ev["Side"] = ev["Side"].astype(str).str.upper()
    else:
        ev["Side"] = "LONG"

    # 중복 로그 정리: 같은 (t,type,side)면 마지막만 남김
    ev = ev.drop_duplicates(subset=["Datetime", "Type", "Side"], keep="last")

    entries = ev[ev["Type"] == "ENTRY"].copy()
    exits   = ev[ev["Type"] == "EXIT"].copy()
    upd_sl  = ev[ev["Type"] == "UPDATE_SL"].copy()

    # -----------------------
    # 4) ENTRY / EXIT markers (깔끔하게)
    # -----------------------
    if not entries.empty:
        eL = entries[entries["Side"] == "LONG"]
        eS = entries[entries["Side"] == "SHORT"]

        if not eL.empty:
            fig.add_trace(go.Scatter(
                x=eL["Datetime"], y=eL["Price"],
                mode="markers",
                name="ENTRY(L)",
                marker=dict(symbol="triangle-up", size=14),
                hovertemplate="ENTRY LONG<br>t=%{x}<br>price=%{y}<extra></extra>",
                showlegend=True
            ))
        if not eS.empty:
            fig.add_trace(go.Scatter(
                x=eS["Datetime"], y=eS["Price"],
                mode="markers",
                name="ENTRY(S)",
                marker=dict(symbol="triangle-down", size=14),
                hovertemplate="ENTRY SHORT<br>t=%{x}<br>price=%{y}<extra></extra>",
                showlegend=True
            ))

    if not exits.empty:
        fig.add_trace(go.Scatter(
            x=exits["Datetime"], y=exits["Price"],
            mode="markers",
            name="EXIT",
            marker=dict(symbol="x", size=16),
            customdata=exits["Reason"] if "Reason" in exits.columns else None,
            hovertemplate="EXIT<br>t=%{x}<br>price=%{y}<br>reason=%{customdata}<extra></extra>",
            showlegend=True
        ))

    # -----------------------
    # 5) SL step line (Reason에서 복원)
    # -----------------------
    if not entries.empty and not upd_sl.empty:
        base_time = df[["Datetime"]].copy()

        # UPDATE_SL에서 SL 추출
        upd = upd_sl.copy()
        upd["SL"] = upd["Reason"].apply(_extract_sl) if "Reason" in upd.columns else np.nan
        upd = upd.dropna(subset=["SL"]).sort_values("Datetime")

        if not upd.empty:
            # 같은 timestamp에 TRAILING / APPLY 둘 다 있으면 APPLY 우선
            upd["is_apply"] = upd["Reason"].astype(str).str.startswith("UPDATE_SL_APPLY:") if "Reason" in upd.columns else False
            upd = upd.sort_values(["Datetime", "is_apply"]).drop_duplicates(subset=["Datetime"], keep="last")

            exits_sorted = exits.sort_values("Datetime")[["Datetime"]].copy() if not exits.empty else None

            # 포지션 단위로 분리해서 선 끊기
            for _, ent in entries.sort_values("Datetime").iterrows():
                t_entry = ent["Datetime"]
                side = str(ent.get("Side", "LONG")).upper()

                # entry 이후 첫 exit (동일 심볼은 이미 필터링되어 있다고 가정)
                if exits_sorted is None or exits_sorted.empty:
                    t_exit = base_time["Datetime"].max()
                else:
                    cand = exits_sorted[exits_sorted["Datetime"] > t_entry]
                    t_exit = cand.iloc[0]["Datetime"] if not cand.empty else base_time["Datetime"].max()

                seg = base_time[(base_time["Datetime"] >= t_entry) & (base_time["Datetime"] <= t_exit)].copy()
                if seg.empty:
                    continue

                upd_seg = upd[(upd["Datetime"] >= t_entry) & (upd["Datetime"] <= t_exit)][["Datetime", "SL"]].copy()
                if upd_seg.empty:
                    continue

                # step: 각 캔들에 직전 SL 적용
                merged = pd.merge_asof(seg, upd_seg.sort_values("Datetime"), on="Datetime", direction="backward")
                merged["SL"] = merged["SL"].ffill()
                merged = merged.dropna(subset=["SL"])
                if merged.empty:
                    continue

                # 첫 SL이 entry 이후 첫 업데이트 시점부터 나오도록 (원하면 여기서 entry 시점까지 수평 연장 가능)
                fig.add_trace(go.Scatter(
                    x=merged["Datetime"],
                    y=merged["SL"],
                    mode="lines",
                    name=f"SL({side})",
                    line=dict(width=2, shape="hv"),
                    hovertemplate="SL<br>t=%{x}<br>sl=%{y}<extra></extra>",
                    showlegend=True
                ))

    fig.update_layout(
        title=f"{symbol} | Candles + ENTRY/EXIT + SL",
        xaxis_title="Datetime",
        yaxis_title="Price",
        height=700,
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    fig.update_xaxes(rangeslider_visible=False)

    return fig


def main():
    st.title("🛡️ Phalanx Backtest Dashboard (Event + MTM)")

    trade_path = _abs_path("backtest_history.csv")
    mtm_path = _abs_path("backtest_equity_curve.csv")

    # Load
    try:
        df_trade = load_trade_log(trade_path)
    except Exception as e:
        st.error(f"❌ Trade log load/validation failed: {e}")
        df_trade = None

    try:
        df_mtm = load_mtm_curve(mtm_path)
    except Exception as e:
        st.warning(f"⚠️ MTM curve load failed (없으면 무시 가능): {e}")
        df_mtm = None

    # Existence checks
    if df_trade is None:
        st.error("❌ backtest_history.csv 가 없습니다/깨졌습니다. 먼저 백테스트를 실행하세요.")
        st.caption(f"Expected: {trade_path}")
        return

    if df_trade.empty:
        st.warning("Trade log 데이터가 비어있습니다.")
        return

    # Sidebar filters
    st.sidebar.header("Filters")
    symbol_list = ["All"] + sorted(df_trade["Symbol"].dropna().unique().tolist())
    selected_symbol = st.sidebar.selectbox("Select Symbol (Event KPI/Log)", symbol_list)

    df_view = df_trade.copy()
    if selected_symbol != "All":
        df_view = df_view[df_view["Symbol"] == selected_symbol].copy()

    # KPIs
    st.subheader("📌 Portfolio KPI (Overall)")

    # Portfolio final equity from trade log (event-balance)
    eq_series = df_trade["Equity"].dropna()
    start_equity = float(eq_series.iloc[0]) if not eq_series.empty else DEFAULT_INITIAL_EQUITY
    final_equity_event = float(eq_series.iloc[-1]) if not eq_series.empty else DEFAULT_INITIAL_EQUITY
    roi_event = (final_equity_event - start_equity) / start_equity * 100.0 if start_equity > 0 else 0.0

    # MTM final equity
    final_equity_mtm = None
    roi_mtm = None
    if df_mtm is not None and not df_mtm.empty:
        final_equity_mtm = float(df_mtm["Equity"].iloc[-1])
        roi_mtm = (final_equity_mtm - start_equity) / start_equity * 100.0 if start_equity > 0 else 0.0

    kpi_all = compute_exit_kpis(df_trade)
    mdd_event = compute_mdd_from_equity_series(df_trade[["Datetime", "Equity"]].copy(), "Equity")
    mdd_mtm = compute_mdd_from_equity_series(df_mtm, "Equity") if df_mtm is not None else None

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("💰 Final Equity (Event)", f"${final_equity_event:,.2f}", f"{roi_event:.2f}%")

    if final_equity_mtm is not None:
        c2.metric("💹 Final Equity (MTM)", f"${final_equity_mtm:,.2f}", f"{roi_mtm:.2f}%")
    else:
        c2.metric("💹 Final Equity (MTM)", "N/A")

    if mdd_event is not None:
        c3.metric("🌊 MDD (Event)", f"{mdd_event:.2f}%")
    else:
        c3.metric("🌊 MDD (Event)", "N/A")

    if mdd_mtm is not None:
        c4.metric("🌊 MDD (MTM)", f"{mdd_mtm:.2f}%")
    else:
        c4.metric("🌊 MDD (MTM)", "N/A")

    if kpi_all:
        pf = kpi_all["profit_factor"]
        c5.metric("⚖️ Profit Factor (EXIT)", "∞" if pf == float("inf") else f"{pf:.2f}")
        c6.metric("✅ Win Rate (EXIT)", f"{kpi_all['win_rate']:.1f}%  ({kpi_all['win_count']}W/{kpi_all['loss_count']}L)")
    else:
        c5.metric("⚖️ Profit Factor (EXIT)", "N/A")
        c6.metric("✅ Win Rate (EXIT)", "N/A")

    st.caption(
        "Event=거래 이벤트 기록(Balance), MTM=캔들마다 시가평가 Equity. "
        "두 곡선은 로그 방식이 달라서 MDD/ROI가 다르게 나오는 게 정상입니다."
    )

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Equity & Drawdown", "📊 PnL Analysis", "📝 Trade Log", "🕯️ Candles & Trades"])
    
    with tab1:
        st.subheader("Equity Curves (Overlay)")

        # Build overlay data
        # Event equity series
        eq_event = df_trade[["Datetime", "Equity"]].dropna().copy()
        eq_event["Series"] = "Event"

        overlay = eq_event

        # MTM series (if exists)
        if df_mtm is not None and not df_mtm.empty:
            eq_mtm = df_mtm[["Datetime", "Equity"]].copy()
            eq_mtm["Series"] = "MTM"
            overlay = pd.concat([eq_event, eq_mtm], ignore_index=True)

        fig_eq = px.line(
            overlay,
            x="Datetime",
            y="Equity",
            color="Series",
            title="Equity Curve (Event vs MTM)",
        )
        st.plotly_chart(fig_eq, width='stretch')

        # Drawdown overlay (computed per series)
        st.subheader("Drawdown Curves (Overlay)")
        dd_frames = []

        def _dd(df_in, label):
            temp = df_in[["Datetime", "Equity"]].dropna().copy()
            if temp.empty:
                return None
            temp = temp.sort_values("Datetime")
            temp["peak"] = temp["Equity"].cummax()
            temp["dd"] = (temp["Equity"] - temp["peak"]) / temp["peak"] * 100.0
            temp["Series"] = label
            return temp[["Datetime", "dd", "Series"]]

        dd_event = _dd(eq_event, "Event")
        if dd_event is not None:
            dd_frames.append(dd_event)

        if df_mtm is not None and not df_mtm.empty:
            dd_mtm = _dd(df_mtm.rename(columns={"Equity": "Equity"}), "MTM")
            if dd_mtm is not None:
                dd_frames.append(dd_mtm)

        if dd_frames:
            dd_all = pd.concat(dd_frames, ignore_index=True)
            fig_dd = px.line(dd_all, x="Datetime", y="dd", color="Series", title="Drawdown (%) (Event vs MTM)")
            st.plotly_chart(fig_dd, width='stretch')
        else:
            st.info("Drawdown 표시를 위한 데이터가 부족합니다.")

    with tab2:
        exits = df_trade[df_trade["Type"] == "EXIT"].copy()
        if exits.empty:
            st.info("EXIT 거래가 없습니다.")
        else:
            # PnL by Symbol
            sym_pnl = exits.groupby("Symbol")["PnL"].sum().sort_values()
            bar_df = sym_pnl.reset_index()
            bar_df.columns = ["Symbol", "PnL"]
            fig_bar = px.bar(bar_df, x="PnL", y="Symbol", orientation="h", title="PnL by Symbol (EXIT Sum)")
            st.plotly_chart(fig_bar, width='stretch')

            # Trade PnL scatter
            scat = exits.copy()
            scat["abs_pnl"] = scat["PnL"].abs()
            fig_scatter = px.scatter(
                scat,
                x="Datetime",
                y="PnL",
                color="Symbol",
                size="abs_pnl",
                title="Trade PnL Distribution (EXIT)",
            )
            st.plotly_chart(fig_scatter, width='stretch')

    with tab3:
        st.subheader("Trade Log (Filtered)")
        st.dataframe(df_view.sort_values(by="Datetime", ascending=False), width='stretch')

        st.caption(f"Trade CSV: {trade_path}")
        if os.path.exists(mtm_path):
            st.caption(f"MTM CSV: {mtm_path}")
        else:
            st.caption("MTM CSV: (not found) — backtest_engine.py에서 backtest_equity_curve.csv 저장 패치를 적용하세요.")

    with tab4:
        st.subheader("🕯️ Candles + ENTRY/EXIT + SL (Backtest)")
        st.caption("OHLCV CSV가 로컬에 있어야 캔들 표시가 됩니다. (없으면 PKL 캐시를 먼저 시도)")

        # 심볼 선택 (All이면 차트가 의미 없으니 강제 선택 형태)
        sym_for_chart = selected_symbol
        if sym_for_chart == "All":
            sym_for_chart = st.selectbox("Select Symbol (Candles)", sorted(df_trade["Symbol"].unique().tolist()))

        # timeframe은 현재 시스템이 15m 고정이라도, 탐색/필터에 필요
        tf = st.selectbox("Timeframe", ["15m"], index=0)

        # 해당 심볼 트레이드만
        df_sym = df_trade[df_trade["Symbol"] == sym_for_chart].copy()
        if df_sym.empty:
            st.info("선택한 심볼의 트레이드 이벤트가 없습니다.")
        else:
            # ------------------------------------------------------------
            # 1) PKL cache first
            # ------------------------------------------------------------
            pkl_path = _abs_path("market_data_cache_30d.pkl")
            cache_obj = None
            ohlcv = None
            tried_paths = []

            try:
                cache_obj = load_market_cache_pkl(pkl_path)  # ✅ 네가 이미 만든 함수
            except Exception as e:
                st.warning(f"PKL load failed (will fallback to CSV): {e}")
                cache_obj = None

            if cache_obj is not None:
                try:
                    ohlcv = get_ohlcv_from_cache(cache_obj, sym_for_chart)  # ✅ 네가 이미 만든 함수
                except Exception as e:
                    st.warning(f"get_ohlcv_from_cache failed (will fallback to CSV): {e}")
                    ohlcv = None

            # ------------------------------------------------------------
            # 2) CSV fallback only if PKL miss
            # ------------------------------------------------------------
            if ohlcv is None:
                try:
                    ohlcv, tried_paths = load_ohlcv_for_symbol(sym_for_chart, timeframe=tf)  # ✅ 네가 이미 만든 함수
                except Exception as e:
                    st.error(f"OHLCV load failed: {e}")
                    ohlcv, tried_paths = None, []

            # ------------------------------------------------------------
            # 3) Render
            # ------------------------------------------------------------
            if ohlcv is None:
                st.warning("OHLCV를 찾지 못했습니다. 아래 경로들을 탐색했습니다:")
                for p in tried_paths:
                    st.code(p)
                st.caption("해결: market_data_cache_30d.pkl 구조를 get_ohlcv_from_cache가 못 읽는 경우이거나, CSV가 없음.")
            else:
                # 보기 편하게 트레이드 구간 주변만 자르기
                t0 = df_sym["Datetime"].min() - pd.Timedelta(hours=6)
                t1 = df_sym["Datetime"].max() + pd.Timedelta(hours=6)

                # ohlcv Datetime 보정(혹시 없으면)
                if "Datetime" not in ohlcv.columns:
                    if "timestamp" in ohlcv.columns:
                        ohlcv = ohlcv.copy()
                        ohlcv["Datetime"] = pd.to_datetime(ohlcv["timestamp"], unit="ms", errors="coerce")
                    elif "date" in ohlcv.columns:
                        ohlcv = ohlcv.copy()
                        ohlcv["Datetime"] = pd.to_datetime(ohlcv["date"], errors="coerce")

                ohlcv = ohlcv.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)

                ohlcv_view = ohlcv[(ohlcv["Datetime"] >= t0) & (ohlcv["Datetime"] <= t1)].copy()
                if ohlcv_view.empty:
                    ohlcv_view = ohlcv.copy()

                fig = make_candle_with_trades(ohlcv_view, df_sym, sym_for_chart)
                st.plotly_chart(fig, width='stretch')

                st.caption("ENTRY/EXIT/UPDATE_SL 마커와 SL(계단식)을 이벤트 로그 기반으로 복원해서 표시합니다.")

if __name__ == "__main__":
    main()
