# down_pkl.py
import os
import json
import pickle
from core.backtest_engine import BacktestEngine

# ============================================================
# 설정
# ============================================================
DAYS = 30
MIN_15M_ROWS = 2800
CACHE_FILE = "market_data_cache_30d.pkl"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")

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

    targets0 = engine.executor.get_top_targets()
    print(f"📡 Initial targets fetched: {len(targets0)}")

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

    print(f"✅ Survived after filter: {len(filtered)} symbols")

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(filtered, f)

    print(f"💾 RAW CACHE SAVED: {CACHE_FILE}")
    print("🎯 Next step: run optimize.py")
