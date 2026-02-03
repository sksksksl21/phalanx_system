# dashboard.py
# =========================================================
# PHALANX Titan Dashboard (LIVE-ONLY, LiveEngine-A compatible)
#
# ✅ LiveEngine (LIVE A-mode: 15m entry / 15m manage) 호환:
# - 단일 상태 파일: phalanx_state.json
# - 단일 히스토리 파일: trade_history.csv
# - (옵션) 로그 파일: phalanx_live.log
#
# 주요 정합성 업데이트:
# - DRY_RUN / multi-track 로직 전부 제거
# - state에서 positions/last_processed_time/last_bucket/freeze 추정 표시
# - trade_history.csv 스키마(엔진의 history_columns) 기준 정규화
# - Streamlit dataframe: width='stretch' 사용
# - Styler.format 안전화(na_rep="")
# - ccxt exchange 객체는 cache_data 인자로 넘기지 않음
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
import hashlib
import pickle
from typing import Optional




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

STATE_FILE = os.path.join(root_dir, "phalanx_state.json")
HISTORY_FILE = os.path.join(root_dir, "trade_history.csv")
RUNTIME_CACHE_FILE = os.path.join(root_dir, ".runtime_files.pkl")
HISTORY_PAD_HOURS = 3          # Tab2 차트 앞뒤 패딩
HISTORY_TF = "15m"             # 히스토리 차트 타임프레임(엔진 15m)
LIVE_TF = "15m"                # 라이브 차트 타임프레임
LIVE_LIMIT = 300               # 라이브 차트 캔들 개수


# ✅ [NEW] pick the most recently updated runtime files (fix path mismatch)

KST_TZ = "Asia/Seoul"

# ---- [NEW] robust file resolver (pick the most recently updated file) ----
def _candidate_roots(*paths: str) -> list[str]:
    out = []
    for p in paths:
        if not p:
            continue
        ap = os.path.abspath(p)
        if ap not in out:
            out.append(ap)
    # parent dirs (1~2 depth)도 포함
    more = []
    for r in list(out):
        more.append(os.path.dirname(r))
        more.append(os.path.dirname(os.path.dirname(r)))
    for r in more:
        r = os.path.abspath(r)
        if r not in out:
            out.append(r)
    return out

def _walk_find_latest(filename: str, roots: list[str], max_depth: int = 3) -> str | None:
    best_fp = None
    best_mtime = -1.0

    for root in roots:
        if not root or not os.path.exists(root):
            continue

        # 1) root 바로 아래 우선
        direct = os.path.join(root, filename)
        if os.path.exists(direct):
            try:
                mt = os.path.getmtime(direct)
                if mt > best_mtime:
                    best_mtime = mt
                    best_fp = direct
            except Exception:
                pass

        # 2) 그 다음은 제한 depth로 walk
        root = os.path.abspath(root)
        base_depth = root.rstrip(os.sep).count(os.sep)

        for cur, dirs, files in os.walk(root):
            cur_depth = cur.rstrip(os.sep).count(os.sep) - base_depth
            if cur_depth > max_depth:
                dirs[:] = []
                continue

            if filename in files:
                fp = os.path.join(cur, filename)
                try:
                    if os.path.getsize(fp) <= 0:
                        continue
                    mt = os.path.getmtime(fp)
                    if mt > best_mtime:
                        best_mtime = mt
                        best_fp = fp
                except Exception:
                    continue

    return best_fp

def resolve_runtime_files(
    root_dir: str,
    default_state: str,
    default_hist: str,
    default_log: str,
) -> dict:
    roots = _candidate_roots(
        root_dir,
        os.getcwd(),
        os.path.dirname(root_dir),
        os.path.dirname(os.getcwd()),
    )

    state_fp = _walk_find_latest(os.path.basename(default_state), roots, max_depth=4) or default_state
    hist_fp  = _walk_find_latest(os.path.basename(default_hist), roots, max_depth=4) or default_hist
    log_fp   = _walk_find_latest(os.path.basename(default_log), roots, max_depth=4) or default_log

    return {"STATE_FILE": state_fp, "HISTORY_FILE": hist_fp, "LOG_FILE": log_fp}

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
    # 키 없어도 티커/차트는 동작하는 경우가 많음(거래/계좌는 제한)
    return ccxt.binance({
        "apiKey": config.get("api_key", ""),
        "secret": config.get("secret_key", ""),
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "future", "adjustForTimeDifference": True},
    })

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

def infer_engine_online(config, candidates_files):
    """
    ONLINE 판정:
    - 후보 파일들(LOG/state/history/config) 중 하나라도 최근 online_sec 이내면 ONLINE
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

def safe_styler(df: pd.DataFrame, fmt_map: dict | None = None):
    if df is None or df.empty:
        return df
    fmt_map = fmt_map or {}
    fmt_map2 = {k: v for k, v in fmt_map.items() if k in df.columns}
    try:
        return df.style.format(fmt_map2, na_rep="")
    except Exception:
        return df


CACHE_DIR = os.path.join(root_dir, "_cache_ohlcv")
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_key(symbol: str, timeframe: str, since_ms: int, until_ms: int) -> str:
    s = f"{symbol}|{timeframe}|{since_ms}|{until_ms}".encode("utf-8")
    return hashlib.md5(s).hexdigest()

def load_ohlcv_pkl(symbol: str, timeframe: str, since_ms: int, until_ms: int):
    key = _cache_key(symbol, timeframe, since_ms, until_ms)
    fp = os.path.join(CACHE_DIR, f"ohlcv_{key}.pkl")
    if not os.path.exists(fp):
        return None
    try:
        return pd.read_pickle(fp)
    except Exception:
        return None

def save_ohlcv_pkl(df: pd.DataFrame, symbol: str, timeframe: str, since_ms: int, until_ms: int):
    key = _cache_key(symbol, timeframe, since_ms, until_ms)
    fp = os.path.join(CACHE_DIR, f"ohlcv_{key}.pkl")
    try:
        df.to_pickle(fp)
        return fp
    except Exception:
        return None

def _to_utc_ms_kst_naive(dt_kst_naive) -> int:
    ts = pd.Timestamp(dt_kst_naive).tz_localize(KST_TZ).tz_convert("UTC")
    return int(ts.timestamp() * 1000)

if os.path.exists(RUNTIME_CACHE_FILE):
    try:
        with open(RUNTIME_CACHE_FILE, "rb") as f:
            _runtime = pickle.load(f)
    except Exception:
        _runtime = None
else:
    _runtime = None

if not _runtime:
    _runtime = resolve_runtime_files(
        root_dir=root_dir,
        default_state=STATE_FILE,
        default_hist=HISTORY_FILE,
        default_log=LOG_FILE,
    )
    try:
        with open(RUNTIME_CACHE_FILE, "wb") as f:
            pickle.dump(_runtime, f)
    except Exception:
        pass

STATE_FILE = _runtime["STATE_FILE"]
HISTORY_FILE = _runtime["HISTORY_FILE"]
LOG_FILE = _runtime["LOG_FILE"]



@st.cache_data(ttl=60)
def fetch_ohlcv_range_with_cache(symbol: str, timeframe: str, t0_kst_naive, t1_kst_naive, limit=1500):
    """
    [HISTORY CHART CORE]
    - (t0~t1) 범위 OHLCV를 가져오고 pkl로 저장/재사용
    - 반환 df index는 KST tz-naive
    """
    ex = st.session_state.get("_exchange")
    if ex is None:
        return None, None, "no_exchange"

    clean = str(symbol).split(":")[0]

    since_ms = _to_utc_ms_kst_naive(t0_kst_naive)
    until_ms = _to_utc_ms_kst_naive(t1_kst_naive)

    # 1) pkl 캐시 우선
    cached = load_ohlcv_pkl(clean, timeframe, since_ms, until_ms)
    if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
        return cached, None, "pkl_hit"

    # 2) ccxt fetch (단발 + 필요시 반복)
    try:
        out = []
        ms = since_ms
        max_rounds = 6  # 너무 오래 끌지 않게 제한
        for _ in range(max_rounds):
            ohlcv = ex.fetch_ohlcv(clean, timeframe=timeframe, since=ms, limit=int(limit))
            if not ohlcv:
                break
            out.extend(ohlcv)

            last_ts = int(ohlcv[-1][0])
            # 진행이 없으면 종료
            if last_ts <= ms:
                break
            ms = last_ts + 1

            # 충분히 until 넘어가면 종료
            if last_ts >= until_ms:
                break

        if not out:
            return None, None, "no_data"

        df = pd.DataFrame(out, columns=["timestamp", "open", "high", "low", "close", "volume"])
        dt = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(KST_TZ)
        df["datetime"] = dt.dt.tz_localize(None)
        df = df.drop_duplicates(subset=["datetime"]).set_index("datetime").sort_index()
        df = df.drop(columns=["timestamp"], errors="ignore")

        # indicator
        df["ema200"] = ta.ema(df["close"], length=200)
        st_out = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
        if st_out is not None and len(st_out.columns) >= 2:
            df["supertrend"] = st_out.iloc[:, 0]
            df["st_dir"] = st_out.iloc[:, 1]

        saved_fp = save_ohlcv_pkl(df, clean, timeframe, since_ms, until_ms)
        return df, saved_fp, "fetched"

    except Exception as e:
        return None, None, f"error:{e}"

def fetch_ohlcv_window_cached(symbol: str, timeframe: str, t0_kst_naive, t1_kst_naive):
    """
    윈도우 범위 OHLCV를 pkl로 저장/재사용하는 단일 진입점.
    plot에서는 이것만 호출.
    """
    df, saved_fp, src = fetch_ohlcv_range_with_cache(
        symbol=symbol,
        timeframe=timeframe,
        t0_kst_naive=t0_kst_naive,
        t1_kst_naive=t1_kst_naive,
        limit=1500,
    )
    return df, saved_fp, src

# =========================
# 3) Trade History (LiveEngine schema normalize)
# =========================

PKL_DIR = os.path.join(root_dir, "_pkl_cache")
os.makedirs(PKL_DIR, exist_ok=True)

def _safe_name(s: str) -> str:
    s = (s or "").replace("/", "_").replace(":", "_").replace(" ", "_")
    return "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-", "."))

def _bundle_key(symbol: str, timeframe: str, start_dt, end_dt, pad_hours: int = 6) -> str:
    # start/end는 KST tz-naive
    base = f"{symbol}|{timeframe}|{start_dt}|{end_dt}|pad={pad_hours}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()  # 짧고 고정

def get_bundle_pkl_path(symbol: str, timeframe: str, start_dt, end_dt, pad_hours: int = 6) -> str:
    sym = _safe_name(symbol.split(":")[0])
    key = _bundle_key(sym, timeframe, start_dt, end_dt, pad_hours)
    return os.path.join(PKL_DIR, f"bundle_{sym}_{timeframe}_{key}.pkl")

def save_bundle_pkl(path: str, bundle: dict):
    try:
        pd.to_pickle(bundle, path)
        return True
    except Exception:
        return False

def load_bundle_pkl(path: str):
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return pd.read_pickle(path)
    except Exception:
        pass
    return None



def load_trade_history(filepath: str) -> pd.DataFrame:
    """
    trade_history.csv 로드 (BOM/헤더/무헤더 + 'reason에 콤마' 깨짐 복구)
    기대 헤더:
    dt,event,mode,symbol,side,price,amount,fee,margin,pnl,roe_pct,sl,reason,pos_count,cash,equity

    ✅ 포맷이 깨진 행(=reason에 콤마가 있는데 따옴표가 없는 경우)을
    '오른쪽 3개 필드(pos_count,cash,equity)부터' 역파싱해서 복구한다.
    """

    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return pd.DataFrame()

    header_cols = [
        "dt","event","mode","symbol","side","price","amount","fee","margin",
        "pnl","roe_pct","sl","reason","pos_count","cash","equity"
    ]
    N = len(header_cols)

    def _norm_sym(x: str) -> str:
        x = (x or "").strip()
        return x.split(":")[0].strip() if x else ""

    def _to_float(x):
        try:
            if x in (None, "", "None", "nan", "NaN"):
                return None
            return float(x)
        except Exception:
            return None

    def _to_int(x):
        try:
            if x in (None, "", "None", "nan", "NaN"):
                return None
            return int(float(x))
        except Exception:
            return None

    def _robust_parse_lines(lines: list[str]) -> pd.DataFrame:
        rows = []
        for ln in lines:
            ln = (ln or "").rstrip("\n")
            if not ln.strip():
                continue

            # 헤더 스킵
            if ln.lower().startswith("dt,event,mode"):
                continue

            parts = ln.split(",")
            if len(parts) < 4:
                continue

            # 1) 정상 케이스: 컬럼 수 정확히 맞음
            if len(parts) == N:
                row = dict(zip(header_cols, parts))
                rows.append(row)
                continue

            # 2) 깨진 케이스: reason에 콤마가 섞여 컬럼이 늘어남
            #    => 오른쪽 3개 (pos_count,cash,equity)는 "마지막 3개의 콤마" 기준으로 역파싱
            #    형식: [앞 12개] + [reason(콤마 포함 가능)] + pos_count + cash + equity
            if len(parts) > N:
                # 오른쪽 3개를 확보 (부족하면 None)
                equity = parts[-1].strip() if len(parts) >= 1 else ""
                cash   = parts[-2].strip() if len(parts) >= 2 else ""
                posc   = parts[-3].strip() if len(parts) >= 3 else ""

                core = parts[:-3]  # reason 포함된 앞부분
                if len(core) < 12:
                    # 최소 앞 12개가 안 되면 복구 불가
                    continue

                first12 = core[:12]
                reason_parts = core[12:]
                reason = ",".join(reason_parts).strip()  # ✅ reason 복구

                row = dict(zip(header_cols[:12], first12))
                row["reason"] = reason
                row["pos_count"] = posc
                row["cash"] = cash
                row["equity"] = equity
                rows.append(row)
                continue

            # 3) 컬럼이 모자라는 경우(희박): 앞에서부터 채우고 나머지 None
            if len(parts) < N:
                row = {c: None for c in header_cols}
                for i, c in enumerate(header_cols[:len(parts)]):
                    row[c] = parts[i]
                rows.append(row)
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=header_cols)

        # 문자열 정리
        for c in ["event","mode","symbol","side","reason"]:
            if c in df.columns:
                df[c] = df[c].astype(str).fillna("").str.strip()

        # 숫자 변환
        for c in ["price","amount","fee","margin","pnl","roe_pct","sl","cash","equity"]:
            if c in df.columns:
                df[c] = df[c].apply(_to_float)
        if "pos_count" in df.columns:
            df["pos_count"] = df["pos_count"].apply(_to_int)

        # dt 파싱(UTC로 읽고 KST naive로)
        if "dt" not in df.columns:
            return pd.DataFrame()

        df["dt"] = pd.to_datetime(df["dt"], errors="coerce", utc=True)
        df = df.dropna(subset=["dt"])
        df["dt"] = df["dt"].dt.tz_convert(KST_TZ).dt.tz_localize(None)

        # derived
        df["event_u"] = df["event"].astype(str).str.upper().str.strip()
        df["side_u"]  = df["side"].astype(str).str.upper().str.strip()
        df["symbol_n"] = df["symbol"].astype(str).map(_norm_sym)

        return df.sort_values("dt", ascending=True).reset_index(drop=True)

    try:
        # ✅ 먼저 pandas로 “정상 CSV” 시도
        df_try = pd.read_csv(filepath, on_bad_lines="skip", encoding="utf-8-sig")
        if (
            df_try is not None
            and not df_try.empty
            and (
                df_try.get("event", "").astype(str).str.upper()
                .isin(["ENTRY", "EXIT", "UPDATE_SL"]).any()
            )
        ):
            cols_l = [str(c).strip().lower() for c in df_try.columns]
            if all(k in cols_l for k in ["dt","event","mode"]):
                # canonical rename
                rename = {}
                for c in df_try.columns:
                    cl = str(c).strip().lower()
                    if cl in header_cols:
                        rename[c] = cl
                df_try = df_try.rename(columns=rename)

                # dt 파싱
                df_try["dt"] = pd.to_datetime(df_try["dt"], errors="coerce", utc=True)
                df_try = df_try.dropna(subset=["dt"])
                if not df_try.empty:
                    df_try["dt"] = df_try["dt"].dt.tz_convert(KST_TZ).dt.tz_localize(None)

                    # derived + 정렬
                    df_try["event_u"] = df_try["event"].astype(str).str.upper().str.strip()
                    df_try["side_u"]  = df_try.get("side", "").astype(str).str.upper().str.strip()
                    df_try["symbol_n"] = df_try.get("symbol", "").astype(str).map(_norm_sym)

                    return df_try.sort_values("dt", ascending=True).reset_index(drop=True)

        # ✅ pandas 결과가 비정상(ENTRY/EXIT가 증발)일 때: 원시 라인 복구 파서 사용
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return _robust_parse_lines(lines)

    except Exception:
        # 마지막 안전망: 원시 라인 복구 파서
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            return _robust_parse_lines(lines)
        except Exception:
            return pd.DataFrame()



def filter_clean_trade_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    KEEP: ENTRY / EXIT / UPDATE_SL
    - event 값이 'UPDATE SL' 같이 들어와도 UPDATE_SL로 정규화해서 살림
    """
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    # event_u 강제 생성/정규화
    if "event_u" not in d.columns:
        d["event_u"] = d.get("event", "").fillna("").astype(str).str.upper().str.strip()
    d["event_u"] = d["event_u"].astype(str).str.upper().str.strip()
    d["event_u"] = d["event_u"].str.replace(" ", "_", regex=False)
    d["event_u"] = d["event_u"].str.replace("-", "_", regex=False)

    # symbol_n 강제 생성
    if "symbol_n" not in d.columns:
        d["symbol_n"] = d.get("symbol", "").fillna("").astype(str).str.split(":").str[0].str.strip()
    d["symbol_n"] = d["symbol_n"].fillna("").astype(str).str.strip()

    keep = {"ENTRY", "EXIT", "UPDATE_SL"}
    d = d[d["event_u"].isin(keep)].copy()
    d = d[d["symbol_n"].str.len() > 0].copy()

    # dt 확인
    if "dt" not in d.columns:
        return pd.DataFrame()
    d = d.dropna(subset=["dt"]).sort_values("dt", ascending=True).reset_index(drop=True)
    return d



def build_trade_windows_from_events(ev: pd.DataFrame) -> list[dict]:
    """
    ev: 특정 symbol의 clean 이벤트(ENTRY/EXIT/UPDATE_SL만)
    return: [{"t0","t1","side_u"}, ...] (t1 None이면 OPEN)
    """
    if ev is None or ev.empty:
        return []

    d = ev.sort_values("dt", ascending=True).reset_index(drop=True)
    if "event_u" not in d.columns:
        d["event_u"] = d.get("event", "").astype(str).str.upper().str.strip()
    if "side_u" not in d.columns:
        d["side_u"] = d.get("side", "").astype(str).str.upper().str.strip()

    wins = []
    cur = None
    for _, r in d.iterrows():
        eu = str(r.get("event_u", "")).upper().strip()
        if eu == "ENTRY":
            if cur is not None:
                wins.append(cur)
            cur = {"t0": r["dt"], "t1": None, "side_u": str(r.get("side_u", "")).upper().strip()}
        elif eu == "EXIT":
            if cur is not None:
                cur["t1"] = r["dt"]
                wins.append(cur)
                cur = None

    if cur is not None:
        wins.append(cur)

    return wins

def plot_history_trade_window(symbol: str, ev_sym: pd.DataFrame, w: dict, pad_hours: int = 3):
    """
    symbol: "XRP/USDT:USDT" 또는 "XRP/USDT" 모두 OK
    ev_sym: 해당 심볼의 clean events (ENTRY/EXIT/UPDATE_SL)
    w: {"t0","t1","side_u"}
    """
    if ev_sym is None or ev_sym.empty or not w:
        st.warning("이 심볼에 표시할 트레이드 이벤트가 없습니다.")
        return

    t0 = w["t0"]
    t1 = w.get("t1", None)

    # 범위: entry~exit (+/- pad)
    start_dt = (pd.Timestamp(t0) - pd.Timedelta(hours=int(pad_hours))).to_pydatetime()
    end_base = pd.Timestamp(t1) if t1 is not None else pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(KST_TZ).tz_localize(None)
    end_dt = (end_base + pd.Timedelta(hours=int(pad_hours))).to_pydatetime()

    ohlcv, saved_fp, status = fetch_ohlcv_range_with_cache(
        symbol=symbol,
        timeframe="15m",
        t0_kst_naive=start_dt,
        t1_kst_naive=end_dt,
        limit=1500
    )

    if ohlcv is None or not isinstance(ohlcv, pd.DataFrame) or ohlcv.empty:
        st.error(f"OHLCV fetch 실패: {status}")
        return

    # 윈도우 이벤트만(차트 범위 내부)
    d = ev_sym.copy()
    if "event_u" not in d.columns:
        d["event_u"] = d.get("event", "").astype(str).str.upper().str.strip()
    if "side_u" not in d.columns:
        d["side_u"] = d.get("side", "").astype(str).str.upper().str.strip()

    d["px"] = pd.to_numeric(d.get("price", pd.NA), errors="coerce")
    d["sl_price"] = pd.to_numeric(d.get("sl", pd.NA), errors="coerce")

    # 표시 구간 캡
    cap_end = ohlcv.index.max() if t1 is None else t1
    d = d[(d["dt"] >= ohlcv.index.min()) & (d["dt"] <= cap_end)].copy()

    # SL step
    wins = [w]
    sl_step = build_sl_step_for_window(ohlcv.index, d, wins[0])

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(
        x=ohlcv.index, open=ohlcv["open"], high=ohlcv["high"], low=ohlcv["low"], close=ohlcv["close"],
        name="Candles",
    ))

    if "ema200" in ohlcv.columns:
        fig.add_trace(go.Scatter(x=ohlcv.index, y=ohlcv["ema200"], name="EMA200", line=dict(width=1)))
    if "supertrend" in ohlcv.columns:
        fig.add_trace(go.Scatter(x=ohlcv.index, y=ohlcv["supertrend"], name="SuperTrend", line=dict(width=1.2, dash="dot")))

    e_in = d[d["event_u"] == "ENTRY"]
    if not e_in.empty:
        fig.add_trace(go.Scatter(
            x=e_in["dt"], y=e_in["px"], mode="markers", name="ENTRY",
            marker=dict(symbol="triangle-up", size=12),
        ))

    e_out = d[d["event_u"] == "EXIT"]
    if not e_out.empty:
        fig.add_trace(go.Scatter(
            x=e_out["dt"], y=e_out["px"], mode="markers", name="EXIT",
            marker=dict(symbol="x", size=12),
            text=e_out.get("reason", ""),
            hovertemplate="EXIT<br>%{x}<br>price=%{y}<br>%{text}<extra></extra>",
        ))

    if sl_step is not None and sl_step.notna().any():
        fig.add_trace(go.Scatter(x=ohlcv.index, y=sl_step, mode="lines", name="SL(step)", line=dict(width=2)))

    src = "pkl_hit" if status == "pkl_hit" else status
    fig.update_layout(
        title=dict(text=f"{symbol} | HISTORY (pad={pad_hours}h) | {w.get('t0')} -> {w.get('t1') if w.get('t1') is not None else 'OPEN'} | src={src}", x=0.5),
        height=560,
        margin=dict(l=0, r=0, t=50, b=0),
        xaxis_rangeslider_visible=False,
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)


HIST_KEEP_EVENTS = {"ENTRY", "EXIT", "UPDATE_SL"}

def hist_clean_for_chart(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tab2 전용 클린:
    - HEARTBEAT / ENTRY_FAIL / ENTRY_REJECT / BOOT / RECONCILE_* 등 제거
    - ENTRY/EXIT/UPDATE_SL 만 유지
    - symbol_n 빈 값 제거
    - px/sl_price 생성
    """
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()
    if "event_u" not in d.columns:
        d["event_u"] = d.get("event", "").astype(str).str.upper().str.strip()
    if "symbol_n" not in d.columns:
        d["symbol_n"] = d.get("symbol", "").astype(str).str.split(":").str[0].str.strip()

    d = d[d["event_u"].isin(HIST_KEEP_EVENTS)].copy()
    d = d[d["symbol_n"].astype(str).str.len() > 0].copy()

    # 표준 컬럼
    d["px"] = pd.to_numeric(d.get("price", pd.NA), errors="coerce")
    d["sl_price"] = pd.to_numeric(d.get("sl", pd.NA), errors="coerce")

    d = d.dropna(subset=["dt"]).sort_values("dt", ascending=True).reset_index(drop=True)
    return d




def hist_symbols(df_clean: pd.DataFrame) -> list[str]:
    if df_clean is None or df_clean.empty:
        return []
    return sorted(df_clean["symbol_n"].dropna().unique().tolist())


def extract_trade_windows(ev: pd.DataFrame) -> list[dict]:
    """
    ENTRY ~ EXIT 윈도우 목록.
    - EXIT 없는 경우 open 윈도우(t1=None)
    """
    if ev is None or ev.empty:
        return []

    d = ev.sort_values("dt", ascending=True).reset_index(drop=True)
    wins = []
    cur = None

    for _, r in d.iterrows():
        eu = str(r.get("event_u","")).upper().strip()
        if eu == "ENTRY":
            if cur is not None:
                wins.append(cur)
            cur = {
                "side_u": str(r.get("side_u","")).upper().strip(),
                "t0": r["dt"],
                "t1": None,
            }
        elif eu == "EXIT":
            if cur is not None:
                cur["t1"] = r["dt"]
                wins.append(cur)
                cur = None

    if cur is not None:
        wins.append(cur)

    return wins


def build_sl_step_for_window(candle_index: pd.Index, ev: pd.DataFrame, win: dict) -> Optional[pd.Series]:
    if candle_index is None or len(candle_index) == 0 or ev is None or ev.empty or not win:
        return None

    t0 = win["t0"]
    t1 = win.get("t1", None)
    idx = candle_index[candle_index >= t0] if t1 is None else candle_index[(candle_index >= t0) & (candle_index <= t1)]
    if len(idx) == 0:
        return None

    d = ev.sort_values("dt", ascending=True).reset_index(drop=True)

    # 윈도우 내부 UPDATE_SL
    u = d[(d["event_u"] == "UPDATE_SL") & (d["dt"] >= t0)].copy()
    if t1 is not None:
        u = u[u["dt"] <= t1]
    u = u.dropna(subset=["dt", "sl_price"])

    s = pd.Series(index=idx, dtype="float64")

    # 초기 SL: ENTRY의 sl_price 우선, 없으면 첫 UPDATE_SL
    sl0 = None
    ent = d[(d["event_u"] == "ENTRY") & (d["dt"] == t0)]
    if not ent.empty:
        v = ent.iloc[0].get("sl_price", None)
        if pd.notna(v):
            sl0 = float(v)
    if sl0 is None and not u.empty:
        sl0 = float(u.iloc[0]["sl_price"])

    if sl0 is not None:
        s.iloc[0] = sl0

    # UPDATE_SL 계단
    for _, r in u.iterrows():
        t = r["dt"]
        slv = float(r["sl_price"])
        k = idx.searchsorted(t, side="right") - 1
        if k >= 0:
            s.iloc[k] = slv

    s = s.ffill()

    full = pd.Series(index=candle_index, dtype="float64")
    full.loc[idx] = s
    return full

@st.cache_data(ttl=300)
def fetch_ohlcv_window(symbol: str, timeframe: str, center_t0, center_t1=None, pad_hours: int = 3, hard_limit: int = 4000):
    """
    center_t0~center_t1 (KST naive) 기준으로 앞뒤 pad_hours 붙여 OHLCV fetch
    반환 df.index = KST naive
    """
    ex = st.session_state.get("_exchange")
    if ex is None:
        return None

    clean = str(symbol).split(":")[0]
    pad = pd.Timedelta(hours=int(pad_hours))

    start_dt = pd.Timestamp(center_t0) - pad
    if center_t1 is None:
        end_dt = pd.Timestamp(center_t0) + pad
    else:
        end_dt = pd.Timestamp(center_t1) + pad

    try:
        start_utc = pd.Timestamp(start_dt).tz_localize(KST_TZ).tz_convert("UTC")
        end_utc = pd.Timestamp(end_dt).tz_localize(KST_TZ).tz_convert("UTC")
        since_ms = int(start_utc.timestamp() * 1000)
        end_ms = int(end_utc.timestamp() * 1000)

        all_rows = []
        step_limit = 1500
        loops = 0
        while True:
            loops += 1
            if loops > 20 or len(all_rows) >= hard_limit:
                break

            batch = ex.fetch_ohlcv(clean, timeframe=timeframe, since=since_ms, limit=step_limit)
            if not batch:
                break

            all_rows.extend(batch)
            last_ts = int(batch[-1][0])
            since_ms = last_ts + 1

            if last_ts >= end_ms:
                break
            if len(batch) < step_limit:
                break

        if not all_rows:
            return None

        df = pd.DataFrame(all_rows, columns=["timestamp","open","high","low","close","volume"])
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df = df[df["timestamp"] <= end_ms].copy()

        dt = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(KST_TZ)
        df["datetime"] = dt.dt.tz_localize(None)
        df = df.set_index("datetime").drop(columns=["timestamp"])
        return df

    except Exception:
        return None

def plot_history_trade_symbol(symbol_n: str, hist_clean_df: pd.DataFrame):
    """
    symbol_n(예: XRP/USDT) 기준:
    - 히스토리 이벤트(ENTRY/EXIT/UPDATE_SL)만
    - 트레이드 윈도우 선택
    - 앞뒤 3시간 OHLCV fetch
    - ENTRY/EXIT/UPDATE_SL 표시 + SL step 라인
    """
    if hist_clean_df is None or hist_clean_df.empty:
        st.warning("history clean df empty")
        return

    ev = hist_clean_df[hist_clean_df["symbol_n"] == symbol_n].copy()
    if ev.empty:
        st.warning("해당 심볼 이벤트 없음")
        return

    wins = extract_trade_windows(ev)
    if not wins:
        st.warning("ENTRY~EXIT 윈도우 없음")
        return

    # closed 먼저, open 나중
    closed = [w for w in wins if w.get("t1") is not None]
    openw = [w for w in wins if w.get("t1") is None]
    ordered = closed + openw

    labels = []
    for w in ordered:
        tag = "CLOSED" if w.get("t1") is not None else "OPEN"
        labels.append(f"{tag} | {w.get('side_u','')} | {w['t0']} -> {w.get('t1','NOW')}")

    sel = st.selectbox("Trade Window", labels, index=0, key=f"hw_{symbol_n}")
    wsel = ordered[labels.index(sel)]

    pad_h = st.number_input("Pad(hours)", min_value=1, max_value=24, value=int(HISTORY_PAD_HOURS), step=1, key=f"pad_{symbol_n}")

    ohlcv = fetch_ohlcv_window(
        symbol=symbol_n,
        timeframe=HISTORY_TF,
        center_t0=wsel["t0"],
        center_t1=wsel.get("t1", None),
        pad_hours=int(pad_h),
    )
    if ohlcv is None or ohlcv.empty:
        st.error("OHLCV fetch 실패")
        return

    # 윈도우 마커 범위
    t0 = wsel["t0"]
    t1 = wsel.get("t1", None)
    end_cap = ohlcv.index.max() if t1 is None else t1
    evw = ev[(ev["dt"] >= t0) & (ev["dt"] <= end_cap)].copy()

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(
        x=ohlcv.index,
        open=ohlcv["open"], high=ohlcv["high"], low=ohlcv["low"], close=ohlcv["close"],
        name="Candles",
    ))

    # ENTRY/EXIT 마커
    e_in = evw[evw["event_u"] == "ENTRY"]
    if not e_in.empty:
        fig.add_trace(go.Scatter(
            x=e_in["dt"], y=e_in["px"],
            mode="markers", name="ENTRY",
            marker=dict(symbol="triangle-up", size=12),
        ))

    e_out = evw[evw["event_u"] == "EXIT"]
    if not e_out.empty:
        fig.add_trace(go.Scatter(
            x=e_out["dt"], y=e_out["px"],
            mode="markers", name="EXIT",
            marker=dict(symbol="x", size=12),
        ))

    # SL step 라인
    sl = build_sl_step_for_window(ohlcv.index, ev, wsel)
    if sl is not None and sl.notna().any():
        fig.add_trace(go.Scatter(
            x=ohlcv.index, y=sl,
            mode="lines", name="SL(step)",
            line=dict(width=2),
        ))

    # UPDATE_SL 마커(선택)
    u = evw[evw["event_u"] == "UPDATE_SL"].dropna(subset=["sl_price"])
    if not u.empty:
        fig.add_trace(go.Scatter(
            x=u["dt"], y=u["sl_price"],
            mode="markers", name="UPDATE_SL",
            marker=dict(symbol="circle", size=7),
        ))

    fig.update_layout(
        title=dict(text=f"{symbol_n} | HISTORY (ENTRY/EXIT/UPDATE_SL only)", x=0.5),
        height=560,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis_rangeslider_visible=False,
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)



def filter_history_since_last_boot(history_df, mode="LIVE"):
    """
    마지막 BOOT 이후만 표시.
    """
    if history_df is None or history_df.empty:
        return pd.DataFrame()

    df = history_df.copy()
    if "mode" in df.columns:
        df["mode"] = df["mode"].astype(str).str.upper()
        df = df[df["mode"] == str(mode).upper()]

    if df.empty:
        return pd.DataFrame()

    df["event_u"] = df["event"].astype(str).str.upper()
    boots = df[df["event_u"] == "BOOT"]
    if boots.empty:
        df.drop(columns=["event_u"], errors="ignore", inplace=True)
        return df

    t0 = boots["dt"].max()
    df = df[df["dt"] >= t0].copy()
    df.drop(columns=["event_u"], errors="ignore", inplace=True)
    return df

def summarize_history(df_hist: pd.DataFrame):
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


def get_symbol_events(hist_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    HISTORY ONLY
    - symbol_n 기준 필터
    - ENTRY/EXIT/UPDATE_SL만 유지
    - px=price, sl_price=sl
    """
    if hist_df is None or hist_df.empty:
        return pd.DataFrame()

    def _norm_sym(x: str) -> str:
        x = (x or "").strip()
        if x.lower() == "nan":
            return ""
        return x.split(":")[0].strip() if x else ""

    target = _norm_sym(symbol)
    if not target:
        return pd.DataFrame()

    d = hist_df.copy()

    if "symbol_n" not in d.columns:
        d["symbol_n"] = d.get("symbol", "").fillna("").astype(str).map(_norm_sym)
    else:
        d["symbol_n"] = d["symbol_n"].fillna("").astype(str).map(_norm_sym)

    d = d[d["symbol_n"] == target].copy()
    if d.empty:
        return pd.DataFrame()

    if "event_u" not in d.columns:
        d["event_u"] = d.get("event", "").fillna("").astype(str).str.upper().str.strip()
    if "side_u" not in d.columns:
        d["side_u"] = d.get("side", "").fillna("").astype(str).str.upper().str.strip()

    d["px"] = pd.to_numeric(d.get("price", pd.NA), errors="coerce")
    d["sl_price"] = pd.to_numeric(d.get("sl", pd.NA), errors="coerce")

    keep = {"ENTRY", "EXIT", "UPDATE_SL"}
    d = d[d["event_u"].isin(keep)].copy()

    d = d.dropna(subset=["dt"]).sort_values("dt", ascending=True).reset_index(drop=True)
    return d



def extract_trade_windows(ev: pd.DataFrame) -> list[dict]:
    """
    ENTRY~EXIT 윈도우 추출 (결정적)
    return: [{"side_u","t0","t1"}, ...]  t1=None이면 미청산
    """
    if ev is None or ev.empty:
        return []

    d = ev.sort_values("dt", ascending=True).reset_index(drop=True)

    wins = []
    cur = None

    for _, r in d.iterrows():
        eu = str(r.get("event_u","")).upper().strip()
        if eu == "ENTRY":
            if cur is not None:
                wins.append(cur)
            cur = {
                "side_u": str(r.get("side_u","")).upper().strip(),
                "t0": r["dt"],
                "t1": None,
            }
        elif eu == "EXIT":
            if cur is not None:
                cur["t1"] = r["dt"]
                wins.append(cur)
                cur = None

    if cur is not None:
        wins.append(cur)

    return wins



def build_sl_step_for_window(candle_index: pd.Index, ev: pd.DataFrame, win: dict) -> pd.Series | None:
    if candle_index is None or len(candle_index) == 0 or ev is None or ev.empty or not win:
        return None

    t0 = win["t0"]
    t1 = win.get("t1", None)

    if t1 is None:
        idx = candle_index[candle_index >= t0]
    else:
        idx = candle_index[(candle_index >= t0) & (candle_index <= t1)]

    if len(idx) == 0:
        return None

    d = ev.sort_values("dt", ascending=True).reset_index(drop=True)

    # 윈도우 내부 UPDATE_SL
    u = d[(d["event_u"] == "UPDATE_SL") & (d["dt"] >= t0)].copy()
    if t1 is not None:
        u = u[u["dt"] <= t1]
    u = u.dropna(subset=["dt", "sl_price"])

    s = pd.Series(index=idx, dtype="float64")

    # 0) 초기 SL: ENTRY sl_price가 있으면 사용, 없으면 윈도우 내 첫 UPDATE_SL을 시작값으로 사용
    sl0 = None
    ent = d[(d["event_u"] == "ENTRY") & (d["dt"] == t0)]
    if not ent.empty:
        v = ent.iloc[0].get("sl_price", None)
        if pd.notna(v):
            sl0 = float(v)

    if sl0 is None and not u.empty:
        sl0 = float(u.iloc[0]["sl_price"])

    if sl0 is not None:
        s.iloc[0] = sl0

    # 1) UPDATE_SL 계단 반영
    for _, r in u.iterrows():
        t = r["dt"]
        slv = float(r["sl_price"])
        k = idx.searchsorted(t, side="right") - 1
        if k >= 0:
            s.iloc[k] = slv

    s = s.ffill()

    full = pd.Series(index=candle_index, dtype="float64")
    full.loc[idx] = s
    return full



def plot_trade_chart(symbol: str, hist_df: pd.DataFrame):
    """
    ✅ 목표:
    - 선택한 trade window(t0~t1 + pad) 범위 OHLCV를 pkl로 저장/재사용
    - bundle/get_or_build_trade_bundle 없이 단일 경로로 동작
    """
    ev = get_symbol_events(hist_df, symbol)
    if ev is None or ev.empty:
        st.warning("해당 심볼의 HISTORY 이벤트(ENTRY/EXIT/UPDATE_SL)가 없습니다.")
        return

    wins = extract_trade_windows(ev)
    if not wins:
        st.warning("해당 심볼의 트레이드 윈도우(ENTRY/EXIT)가 없습니다.")
        return

    closed = [w for w in wins if w.get("t1") is not None]
    openw = [w for w in wins if w.get("t1") is None]
    ordered = closed + openw

    labels = []
    for w in ordered:
        tag = "CLOSED" if w.get("t1") is not None else "OPEN"
        labels.append(f"{tag} | {w.get('side_u','')} | {w.get('t0')} -> {w.get('t1') if w.get('t1') is not None else 'NOW'}")

    sel = st.selectbox("Select Trade Window (history)", labels, index=0, key=f"tw_{symbol}")
    wsel = ordered[labels.index(sel)]

    cA, cB, cC = st.columns([1, 1, 2])
    with cA:
        pad_hours = st.number_input("Pad(hours)", min_value=1, max_value=48, value=6, step=1, key=f"pad_{symbol}")
    with cB:
        force_refresh = st.checkbox("Force refetch (ignore pkl)", value=False, key=f"force_{symbol}")
    with cC:
        st.caption("pkl cache: (t0~t1 + pad) 범위 OHLCV 저장/재사용")

    # --- 범위 계산 ---
    t0 = wsel["t0"]
    t1 = wsel.get("t1", None)
    pad = pd.Timedelta(hours=int(pad_hours))

    t_fetch0 = t0 - pad
    # 미청산이면 t0+24h를 기본 캡(원하면 늘려)
    t_fetch1 = (t1 + pad) if t1 is not None else (t0 + pd.Timedelta(hours=24))

    # --- pkl 캐시 (강제 무시 옵션) ---
    if force_refresh:
        # 캐시 무시: 키가 범위 기반이라 "다른 이름"으로 저장하는 게 아니라
        # st.cache_data를 무효화하는 방식이 현실적
        try:
            st.cache_data.clear()
        except Exception:
            pass

    df, saved_fp, src = fetch_ohlcv_window_cached(symbol, "15m", t_fetch0, t_fetch1)
    if df is None or df.empty:
        st.error("차트 캔들 로드 실패(거래소 fetch / pkl 모두 실패)")
        return

    plot_df = df[(df.index >= t_fetch0) & (df.index <= t_fetch1)].copy()
    if plot_df.empty:
        st.error("선택 범위(plot_df)가 비었습니다. (pad/기간 확인)")
        return

    # --- SL step (윈도우에 맞춰 생성) ---
    sl = build_sl_step_for_window(plot_df.index, ev, wsel)

    st.caption(f"source={src}" + (f" | pkl={os.path.basename(saved_fp)}" if saved_fp else ""))

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df["open"], high=plot_df["high"], low=plot_df["low"], close=plot_df["close"],
        name="Candles",
    ))

    if "ema200" in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["ema200"], name="EMA200", line=dict(width=1)))
    if "supertrend" in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["supertrend"], name="SuperTrend", line=dict(width=1.5, dash="dot")))

    # --- 마커(윈도우 내부 이벤트만) ---
    end_cap = t_fetch1 if t1 is not None else plot_df.index.max()
    evw = ev[(ev["dt"] >= t0) & (ev["dt"] <= end_cap)].copy()

    e_in = evw[evw["event_u"] == "ENTRY"]
    if not e_in.empty:
        fig.add_trace(go.Scatter(
            x=e_in["dt"], y=e_in["px"],
            mode="markers", name="ENTRY",
            marker=dict(symbol="triangle-up", size=12),
        ))

    e_out = evw[evw["event_u"] == "EXIT"]
    if not e_out.empty:
        fig.add_trace(go.Scatter(
            x=e_out["dt"], y=e_out["px"],
            mode="markers", name="EXIT",
            marker=dict(symbol="x", size=12),
            text=e_out.get("reason", ""),
            hovertemplate="EXIT<br>%{x}<br>price=%{y}<br>%{text}<extra></extra>",
        ))

    if sl is not None and getattr(sl, "notna", None) and sl.notna().any():
        fig.add_trace(go.Scatter(x=plot_df.index, y=sl, mode="lines", name="SL(step)", line=dict(width=2)))
    else:
        st.info("SL(step) 데이터가 없습니다. (ENTRY sl / UPDATE_SL / sl 컬럼 확인)")

    fig.update_layout(
        title=dict(text=f"{symbol} | History Trade Chart (pkl cached range)", x=0.5),
        height=520,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis_rangeslider_visible=False,
        showlegend=True,
    )
    st.plotly_chart(fig, width='stretch')


# =========================
# 5) State / Metrics (LIVE)
# =========================
def load_state_live():
    state = load_json(STATE_FILE)
    if not isinstance(state, dict):
        state = {}

    positions = (state.get("positions", {}) or {})
    last_processed_time = state.get("last_processed_time", None)
    last_bucket = state.get("last_bucket", None)

    # freeze_new_entries는 state에 반드시 저장되는 필드가 아니므로,
    # "최근 RECONCILE_MISMATCH 또는 ENTRY_FROZEN 이벤트가 있는지"로 추정도 가능.
    # 여기서는 일단 None로 두고, history에서 보강 표시.
    return {
        "positions": positions,
        "last_processed_time": last_processed_time,
        "last_bucket": last_bucket,
        "telegram_last_update_id": state.get("telegram_last_update_id", 0),
    }

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
            tp1 = pos.get("tp1", None)
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
# 6) Load Data
# =========================
config = load_json(CONFIG_FILE)
exchange = init_exchange(config)
st.session_state["_exchange"] = exchange  # ✅ cache_data에서 접근

max_pos = get_max_positions(config)
lev = get_leverage(config)

state = load_state_live()

# ONLINE 판정 후보 파일 모음
online_files = [CONFIG_FILE, STATE_FILE, HISTORY_FILE, LOG_FILE]
engine_online, age_sec = infer_engine_online(config, online_files)

log_lines = read_logs_tail(LOG_FILE, n=250)
heartbeat_line = parse_last_heartbeat(log_lines)

hist_df_all = load_trade_history(HISTORY_FILE) if os.path.exists(HISTORY_FILE) else pd.DataFrame()
hist_df_sess = filter_history_since_last_boot(hist_df_all, mode="LIVE")
hist_summary = summarize_history(hist_df_sess)

# freeze 추정(최근 세션에 RECONCILE_MISMATCH/ENTRY_FROZEN 있으면 True로 보임)
freeze_est = False
try:
    if hist_df_sess is not None and not hist_df_sess.empty:
        evu = hist_df_sess["event"].astype(str).str.upper()
        freeze_est = bool(evu.isin(["RECONCILE_MISMATCH", "ENTRY_FROZEN"]).any())
except Exception:
    freeze_est = False

total_equity, free_money, unreal_pnl, active_df = calculate_metrics_live(config, state, exchange)

# =========================
# 7) Sidebar
# =========================
with st.sidebar:
    st.title("🛡️ PHALANX")

    st.caption(f"Mode: **LIVE** | Leverage: **{lev:g}x** | MaxPos: **{max_pos}**")

    if engine_online:
        st.success("🟢 ENGINE ONLINE")
    else:
        if age_sec is None:
            st.error("🔴 OFFLINE (no files)")
        else:
            st.error(f"🔴 OFFLINE ({int(age_sec)}s ago)")

    if heartbeat_line:
        st.caption("Last Heartbeat (log)")
        st.code(heartbeat_line, language="text")


    st.caption("Resolved Files (actual paths)")
    st.code(f"STATE:   {STATE_FILE}\nHISTORY: {HISTORY_FILE}\nLOG:     {LOG_FILE}", language="text")

    st.divider()

    lp = state.get("last_processed_time", None)
    lb = state.get("last_bucket", None)
    st.caption("Engine State (from phalanx_state.json)")
    st.write(f"- last_processed_time: `{lp}`" if lp else "- last_processed_time: (none)")
    st.write(f"- last_bucket: `{lb}`" if lb is not None else "- last_bucket: (none)")
    st.write(f"- freeze_new_entries (estimated): `{int(freeze_est)}`")
    st.write("hist_df_all columns:", list(hist_df_all.columns))
    st.write(hist_df_all.tail(20))

    st.divider()

    if st.button("🔄 Force Refresh"):
        st.cache_data.clear()
        st.rerun()

    auto_refresh = st.checkbox("Auto Refresh (15s)", value=True)

# =========================
# 8) Header Metrics
# =========================
st.title("🛡️ PHALANX Titan Dashboard")

top_c1, top_c2, top_c3, top_c4, top_c5 = st.columns([1, 1, 1, 1, 2])
top_c1.metric("Mode", "LIVE")
top_c2.metric("Engine", "ONLINE" if engine_online else "OFFLINE")
top_c3.metric("Total Equity (USDT)", f"${total_equity:,.2f}")
top_c4.metric("Free (USDT)", f"${free_money:,.2f}")
top_c5.metric("Unrealized PnL", f"${unreal_pnl:,.2f}")

st.divider()


def render_tab_live_status(
    state, max_pos, freeze_est, hist_summary,
    active_df, hist_df_all, hist_df_sess
):
    st.subheader("⚔️ Live + History (Chart)")

    positions = state.get("positions", {}) or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Positions", f"{len(positions)} / {max_pos}")
    c2.metric("Freeze (est.)", f"{int(freeze_est)}")
    c3.metric("History Events (session)", f"{hist_summary['events']}")
    c4.metric("PnL Sum (session exits)", f"{hist_summary['pnl_sum']:+.2f}")

    st.markdown("---")
    st.caption("LIVE Positions (state 기반)")

    if active_df is not None and not active_df.empty:
        show_df = active_df.copy()
        st.dataframe(
            safe_styler(show_df, {
                "Qty": "{:.6f}", "Entry": "{:.6f}", "Current": "{:.6f}",
                "PnL($)": "{:.2f}", "ROE(%)": "{:.2f}", "Margin": "{:.2f}", "SL": "{:.6f}",
            }),
            width='stretch',
            hide_index=True
        )
    else:
        st.info("현재 보유 포지션이 없습니다.")

    st.markdown("---")
    st.caption("HISTORY Chart (trade_history.csv 기반)")

    # 심볼 리스트는 ENTRY/EXIT/UPDATE_SL 기준으로 생성
    if hist_df_all is None or hist_df_all.empty:
        st.warning("trade_history.csv 가 비어있습니다.")
        return

    d = hist_df_all.copy()
    if "event_u" not in d.columns:
        d["event_u"] = d.get("event", "").astype(str).str.upper().str.strip()
    if "symbol_n" not in d.columns:
        d["symbol_n"] = d.get("symbol", "").astype(str).str.split(":").str[0].str.strip()

    d2 = d[(d["event_u"].isin(["ENTRY","EXIT","UPDATE_SL"])) & (d["symbol_n"].astype(str).str.len() > 0)]
    hist_syms = sorted(d2["symbol_n"].dropna().unique().tolist())

    if not hist_syms:
        st.warning("HISTORY에 ENTRY/EXIT/UPDATE_SL 심볼이 없습니다.")
        return

    colL, colR = st.columns([1, 3], gap="large")
    with colL:
        sel_hist_symbol = st.selectbox("Select Symbol (History)", hist_syms, index=0, key="hist_symbol")
        use_session = st.checkbox("Use session only (since last BOOT) for chart", value=False, key="hist_session_only")
        src_df = hist_df_sess if use_session else hist_df_all
        st.caption(f"history_rows={0 if src_df is None else len(src_df)}")

    with colR:
        # ✅ pkl bundle + SL 포함 차트 (이 함수 하나로 통일 권장)
        plot_trade_chart(sel_hist_symbol, src_df)

@st.cache_data(ttl=30)
def fetch_live_chart(symbol: str, timeframe="15m", limit=300):
    ex = st.session_state.get("_exchange")
    if ex is None:
        return None
    clean = str(symbol).split(":")[0]
    try:
        ohlcv = ex.fetch_ohlcv(clean, timeframe=timeframe, limit=int(limit))
        if not ohlcv:
            return None
        df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
        dt = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(KST_TZ)
        df["datetime"] = dt.dt.tz_localize(None)
        df = df.set_index("datetime").drop(columns=["timestamp"])
        return df
    except Exception:
        return None


def plot_live_position_chart(symbol: str, pos: dict):
    df = fetch_live_chart(symbol, timeframe=LIVE_TF, limit=LIVE_LIMIT)
    if df is None or df.empty:
        st.error(f"live chart load fail: {symbol}")
        return

    entry = float(pos.get("entry_price", 0) or 0)
    sl = float(pos.get("sl", 0) or 0)
    side = str(pos.get("side", "")).upper()

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Candles"
    ))

    if entry > 0:
        fig.add_hline(y=entry, line_dash="solid", annotation_text="ENTRY")
    if sl > 0:
        fig.add_hline(y=sl, line_dash="dot", annotation_text="SL")

    fig.update_layout(
        title=dict(text=f"{symbol} | LIVE ({side})", x=0.5),
        height=520,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis_rangeslider_visible=False,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# Tab 1: Live Status
# -------------------------


tab1, tab2, tab3 = st.tabs(["📊 Live (Positions)", "📜 History (Trades)", "💻 Logs"])

with tab1:
    st.subheader("📊 LIVE MODE (현재 관리중 포지션)")
    positions = (state.get("positions", {}) or {})
    if not positions:
        st.info("현재 보유 포지션이 없습니다.")
    else:
        # 심볼 선택
        live_syms = sorted(list(positions.keys()))
        sel = st.selectbox("Select Active Position", live_syms, index=0, key="live_sym")
        plot_live_position_chart(sel, positions.get(sel, {}))

def _raw_event_counts(fp: str):
    if not os.path.exists(fp):
        return {"_err": "not exists"}
    try:
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            txt = f.read()
        return {
            "contains_ENTRY": txt.upper().count("ENTRY"),
            "contains_EXIT": txt.upper().count("EXIT"),
            "contains_UPDATE_SL": txt.upper().count("UPDATE_SL"),
            "bytes": os.path.getsize(fp),
        }
    except Exception as e:
        return {"_err": str(e)}

def _raw_tail(fp: str, n=80) -> str:
    if not os.path.exists(fp):
        return "(not exists)"
    try:
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except Exception as e:
        return f"(tail read error: {e})"


with tab2:
    st.subheader("📜 History (Trades only: ENTRY / EXIT / UPDATE_SL)")

    if hist_df_all is None or hist_df_all.empty:
        st.warning("trade_history.csv 가 없거나 비어 있습니다.")
        st.stop()

    show_session_only = st.checkbox("Use session only (since last BOOT)", value=False)
    src_df = hist_df_sess if show_session_only else hist_df_all

    clean = filter_clean_trade_events(src_df)

    with st.expander("TAB2 DEBUG (필수)", expanded=True):
        st.write("HISTORY_FILE:", HISTORY_FILE)
        st.write("rows(all):", 0 if hist_df_all is None else len(hist_df_all))
        st.write("rows(src_df):", 0 if src_df is None else len(src_df))
        st.write("RAW COUNTS:", _raw_event_counts(HISTORY_FILE))
        st.text_area("RAW TAIL (last 80 lines)", _raw_tail(HISTORY_FILE, 80), height=260)   
        if src_df is not None and not src_df.empty:
            # event_u 분포
            tmp = src_df.copy()
            if "event_u" not in tmp.columns:
                tmp["event_u"] = tmp.get("event", "").fillna("").astype(str).str.upper().str.strip()
                tmp["event_u"] = tmp["event_u"].str.replace(" ", "_", regex=False).str.replace("-", "_", regex=False)

            st.write("unique event_u (top 30):", tmp["event_u"].value_counts().head(30))
            st.write("sample rows (tail 30):")
            show_cols = [c for c in ["dt","event","event_u","mode","symbol","symbol_n","side","price","sl","reason"] if c in tmp.columns]
            st.dataframe(tmp[show_cols].tail(30), width="stretch", hide_index=True)

        st.write("rows(clean):", len(clean) if clean is not None else 0)


    if clean.empty:
        st.warning("ENTRY/EXIT/UPDATE_SL 이벤트가 없습니다. (HEARTBEAT/BOOT/RECONCILE/FAIL/REJECT 등은 제외)")
        st.stop()

    # 심볼 선택 (trade 이벤트가 있는 심볼만)
    syms = sorted(clean["symbol_n"].dropna().unique().tolist())
    colL, colR = st.columns([1, 3], gap="large")

    with colL:
        sel = st.selectbox("Symbol", syms, index=0, key="hist2_symbol")
        pad_hours = st.number_input("Pad (hours)", min_value=1, max_value=12, value=3, step=1, key="hist2_pad")
        st.caption("※ trade 이벤트만 사용 (HEARTBEAT/BOOT/RECONCILE/FAIL/REJECT 제거)")

    # 해당 심볼 이벤트 + 윈도우
    ev_sym = clean[clean["symbol_n"] == sel].copy()
    wins = build_trade_windows_from_events(ev_sym)

    if not wins:
        st.warning("이 심볼에 ENTRY→EXIT 윈도우가 없습니다.")
        st.stop()

    # 윈도우 선택
    win_labels = []
    for w in wins:
        tag = "CLOSED" if w.get("t1") is not None else "OPEN"
        win_labels.append(f"{tag} | {w.get('side_u','')} | {w.get('t0')} -> {w.get('t1') if w.get('t1') is not None else 'NOW'}")

    with colR:
        selw = st.selectbox("Trade Window", win_labels, index=max(0, len(win_labels)-1), key="hist2_window")
        wsel = wins[win_labels.index(selw)]

        # symbol은 원본 풀네임으로 넣어도 되지만, ccxt는 :USDT 없는 쪽이 안정적이라 clean 처리
        sym_for_fetch = sel  # sel은 symbol_n (":USDT" 제거 상태)
        plot_history_trade_window(sym_for_fetch, ev_sym, wsel, pad_hours=int(pad_hours))


with tab3:
    st.subheader("💻 LOGS (tail)")
    if log_lines:
        st.text_area("Log Tail", "".join(log_lines), height=680, disabled=True)
    else:
        st.info("phalanx_live.log 파일이 없습니다.")

# =========================
# 10) Auto refresh
# =========================
if auto_refresh:
    time.sleep(15)
    st.rerun()

