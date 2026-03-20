# down_pkl.py
import os
import json
import pickle
from core.backtest_engine import BacktestEngine

# ============================================================
# 설정
# ============================================================
DAYS = 40
MIN_15M_ROWS = 3840
CACHE_FILE = "market_data_cache_7d.pkl"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")
UNIVERSE_PATH = os.path.join(ROOT_DIR, "universe.json")


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


def load_targets(engine):
    """
    universe.json이 있으면 그 심볼만 사용
    없으면 기존처럼 executor.get_top_targets() 사용
    지원 포맷:
      1) {"universe": [...]} 
      2) ["BTC/USDT:USDT", ...]
    """
    if os.path.exists(UNIVERSE_PATH):
        try:
            with open(UNIVERSE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                symbols = data.get("universe", [])
            elif isinstance(data, list):
                symbols = data
            else:
                symbols = []

            if not isinstance(symbols, (list, tuple, set)):
                symbols = []

            symbols = [str(x).strip() for x in symbols if str(x).strip()]

            if symbols:
                print(f"📂 universe.json loaded: {len(symbols)} symbols")
                return symbols
            else:
                print("⚠️ universe.json exists but no valid symbols found. Fallback to top targets.")

        except Exception as e:
            print(f"⚠️ Failed to load universe.json: {e}")
            print("↩️ Fallback to top targets.")

    targets = engine.executor.get_top_targets()
    print(f"📡 Fallback top targets fetched: {len(targets)}")
    return targets

def get_daily_need_days(engine):
    """
    전략의 daily_ema 요구사항을 읽어
    최소 보장 일봉 수를 반환한다.
    규칙:
    - 최소 40일
    - 전략 최소 요구(daily_ema + 5) 반영
    """
    try:
        p = getattr(engine.titan, "params", None)
        p = p if isinstance(p, dict) else {}
    except Exception:
        p = {}

    try:
        daily_len = int(p.get("daily_ema", 25) or 25)
    except Exception:
        daily_len = 25

    return int(max(40, daily_len + 5))

def build_cache_payload(engine, raw_15m_map: dict):
    """
    15m raw + 1d raw를 함께 payload로 만든다.
    - 15m survivor만 대상으로 daily context를 다운로드
    - daily 요구량 미충족 심볼은 여기서 제거
    """
    if not raw_15m_map:
        return {
            "schema_version": 2,
            "meta": {
                "days_15m": int(DAYS),
                "daily_need_days": 40,
                "symbols": 0,
            },
            "raw_15m_map": {},
            "raw_daily_map": {},
        }

    need_days = int(get_daily_need_days(engine))
    syms = sorted(list(raw_15m_map.keys()))

    engine.raw_data_map = {sym: raw_15m_map[sym] for sym in syms}
    engine.symbols = list(syms)

    # 엔진 helper로 1d context 다운로드
    engine._prepare_daily_context(syms)

    kept_15m = {}
    kept_1d = {}
    dropped_daily = []

    for sym in syms:
        df15 = raw_15m_map.get(sym)
        dfd = getattr(engine, "raw_daily_map", {}).get(sym)

        if df15 is None or getattr(df15, "empty", True):
            continue
        if dfd is None or getattr(dfd, "empty", True):
            dropped_daily.append((sym, 0))
            continue
        if len(dfd) < need_days:
            dropped_daily.append((sym, int(len(dfd))))
            continue

        kept_15m[sym] = df15.copy(deep=True) if hasattr(df15, "copy") else df15
        kept_1d[sym] = dfd.copy(deep=True) if hasattr(dfd, "copy") else dfd

    if dropped_daily:
        print("⚠️ DROP (daily insufficient)")
        for sym, nrows in dropped_daily:
            print(f"   - {sym} rows1d={nrows} need>={need_days}")

    payload = {
        "schema_version": 2,
        "meta": {
            "days_15m": int(DAYS),
            "daily_need_days": int(need_days),
            "symbols": int(len(kept_15m)),
        },
        "raw_15m_map": kept_15m,
        "raw_daily_map": kept_1d,
    }
    return payload

# ============================================================
# RAW CACHE BUILD
# ============================================================
if __name__ == "__main__":
    print("🛠️ Building RAW market data cache (15m + daily context)")

    blacklist = load_blacklist()
    print(f"🚫 Configured Blacklist: {blacklist}")

    engine = BacktestEngine(days=DAYS)

    targets0 = load_targets(engine)
    print(f"📡 Initial targets resolved: {len(targets0)}")

    # ✅ 블랙리스트 제외
    targets = [s for s in targets0 if s not in blacklist]
    print(f"🚫 After blacklist filter: {len(targets)} (removed={len(targets0)-len(targets)})")

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

    print(f"✅ Survived after 15m filter: {len(filtered)} symbols")

    payload = build_cache_payload(engine, filtered)

    raw_15m_map = payload.get("raw_15m_map", {}) or {}
    raw_daily_map = payload.get("raw_daily_map", {}) or {}
    meta = payload.get("meta", {}) or {}

    if not raw_15m_map:
        raise RuntimeError("❌ No symbols survived daily-context filter")

    print(
        f"✅ Final cache ready | symbols={len(raw_15m_map)} "
        f"daily_need_days>={meta.get('daily_need_days', 'NA')}"
    )

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(payload, f)

    print(f"💾 RAW CACHE SAVED: {CACHE_FILE}")
    print(f"   - 15m symbols: {len(raw_15m_map)}")
    print(f"   - 1d symbols : {len(raw_daily_map)}")
    print("🎯 Next step: run optimize.py")