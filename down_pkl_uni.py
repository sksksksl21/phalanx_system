# down_pkl.py
import os
import json
import pickle
from core.backtest_engine import BacktestEngine

# ============================================================
# 설정
# ============================================================
DAYS = 105
MIN_15M_ROWS = 10080
CACHE_FILE = "market_data_cache_30d_uni.pkl"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")
def fetch_top_usdt_perp_symbols(top_n: int, blacklist: set):
    """
    ✅ executor.get_top_targets() (25 고정) 우회용.
    - CCXT로 Binance USDⓈ-M 선물 티커를 조회
    - quoteVolume 기준 상위 top_n개를 "XXX/USDT:USDT" 형태로 반환
    """
    try:
        import ccxt
    except Exception as e:
        raise RuntimeError(f"❌ ccxt import failed: {e}")

    try:
        n = int(top_n)
        if n <= 0:
            n = 100
    except Exception:
        n = 100

    bl = set(str(x) for x in (blacklist or set()) if x)

    ex = ccxt.binanceusdm({"enableRateLimit": True})

    # 1) 마켓 로드
    try:
        markets = ex.load_markets()
    except Exception as e:
        raise RuntimeError(f"❌ load_markets failed: {e}")

    # 2) USDT 선물(스왑) 심볼 풀 수집
    # ccxt에서 binanceusdm의 swap 심볼은 보통 "BTC/USDT" 형태
    pool = []
    for sym, m in (markets or {}).items():
        try:
            if not isinstance(m, dict):
                continue
            if not m.get("active", True):
                continue
            if not m.get("swap", False):
                continue
            if m.get("quote") != "USDT":
                continue

            # 레버리지 토큰/이상한 마켓 방어(원하면 더 강화 가능하지만 지금은 최소)
            if "UP/" in sym or "DOWN/" in sym or "BULL/" in sym or "BEAR/" in sym:
                continue

            # 블랙리스트 체크는 clean 기준으로도
            clean = sym.split(":")[0]
            if sym in bl or clean in bl:
                continue

            pool.append(sym)
        except Exception:
            continue

    if not pool:
        raise RuntimeError("❌ No USDT perpetual symbols found from ccxt.")

    # 3) 티커로 거래대금(quoteVolume) 상위 정렬
    # fetch_tickers는 심볼 리스트를 넣으면 되는데, 너무 많으면 에러 날 수 있어
    # 그래서 안전하게 chunk로 나눔
    volumes = {}  # sym -> quoteVolume
    chunk = 200

    for i in range(0, len(pool), chunk):
        batch = pool[i:i + chunk]
        try:
            ticks = ex.fetch_tickers(batch) or {}
        except Exception:
            ticks = {}
        for s in batch:
            t = ticks.get(s) or {}
            qv = t.get("quoteVolume", None)
            if qv is None:
                # 대체 필드 방어(거래소/ccxt 버전에 따라 다름)
                qv = t.get("quoteVolume", None) or t.get("baseVolume", None)
            try:
                qv = float(qv) if qv is not None else 0.0
            except Exception:
                qv = 0.0
            volumes[s] = qv

    ranked = sorted(volumes.items(), key=lambda kv: kv[1], reverse=True)
    ranked = [s for s, _ in ranked if s]

    # 4) 엔진 포맷 "XXX/USDT:USDT"로 변환
    out = []
    seen = set()
    for s in ranked:
        try:
            base = str(s).strip()
            if not base:
                continue
            # "BTC/USDT" -> "BTC/USDT:USDT"
            fmt = base if ":" in base else (base + ":USDT")
            clean = fmt.split(":")[0]

            if fmt in bl or clean in bl:
                continue
            if fmt in seen:
                continue
            seen.add(fmt)
            out.append(fmt)
            if len(out) >= n:
                break
        except Exception:
            continue

    if len(out) < n:
        print(f"⚠️ Only resolved {len(out)} symbols (requested {n}).")

    return out


def load_blacklist():
    if not os.path.exists(CONFIG_PATH):
        return set()

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load config.json: {e}")
        return set()

    # ✅ 실제 위치: strategy_settings.blacklist
    ss = cfg.get("strategy_settings", {})
    if not isinstance(ss, dict):
        ss = {}

    bl = ss.get("blacklist", [])
    if not isinstance(bl, (list, tuple, set)):
        return set()

    return set(str(x) for x in bl if x)

# ============================================================
# RAW CACHE BUILD
# ============================================================
if __name__ == "__main__":
    print("🛠️ Building RAW market data cache (30d, 15m)")

    blacklist = load_blacklist()
    print(f"🚫 Configured Blacklist: {blacklist}")

    engine = BacktestEngine(days=DAYS)

    # ✅ get_top_targets()는 25 고정이므로, down_pkl은 여기서 강제로 100개를 만든다.
    TOP_N = 100
    targets0 = fetch_top_usdt_perp_symbols(TOP_N, blacklist)
    print(f"📡 Targets resolved via ccxt: {len(targets0)} (top{TOP_N})")

    # ✅ 블랙리스트 제외(이중 안전)
    targets = []
    for s in targets0:
        if s in blacklist:
            continue
        clean = s.split(":")[0]
        if clean in blacklist:
            continue
        targets.append(s)

    print(f"🚫 After blacklist filter: {len(targets)} (removed={len(targets0)-len(targets)})")
    print(f"🧾 targets sample: {targets[:10]}")

    raw_map = engine.executor.prepare_data(targets, days=DAYS)
    if not raw_map:
        raise RuntimeError("❌ No raw data fetched")

    print(f"📥 Raw data fetched: {len(raw_map)} symbols")

    filtered = {}
    for sym, df in raw_map.items():
        rows = len(df) if df is not None else 0
        if rows >= MIN_15M_ROWS:
            filtered[sym] = df
        else:
            print(f"⚠️ DROP (new listing) | {sym} rows={rows}")

    if not filtered:
        raise RuntimeError("❌ No symbols survived new-listing filter")

    print(f"✅ Survived after filter: {len(filtered)} symbols")

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(filtered, f)

    print(f"💾 RAW CACHE SAVED: {CACHE_FILE}")
    print("🎯 Next step: run optimize.py")
