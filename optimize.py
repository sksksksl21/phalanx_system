import optuna
import pandas as pd
import numpy as np
import os
import time
import pickle
from core.backtest_engine import BacktestEngine

# ============================================================
# Optuna / Path
# ============================================================
optuna.logging.set_verbosity(optuna.logging.WARNING)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_FILE = os.path.join(CURRENT_DIR, "optimization_results.csv")
DATA_CACHE_FILE = os.path.join(CURRENT_DIR, "market_data_cache_30d.pkl")
UNIVERSE_SNAPSHOT_FILE = os.path.join(CURRENT_DIR, "universe_optimization.json")

INITIAL_BALANCE = 10000.0
MIN_15M_ROWS = 2880
FIXED_UNIVERSE_SIZE = 25

# ============================================================
# Metrics (RAW, 최소)
# ============================================================
def calculate_metrics(initial_balance, history):
    if not history:
        return None

    df = pd.DataFrame(history)
    if df.empty or "pnl" not in df.columns:
        return None

    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]

    final_equity = initial_balance + df["pnl"].sum()
    total_return = (final_equity - initial_balance) / initial_balance * 100.0

    df["equity"] = initial_balance + df["pnl"].cumsum()
    peak = df["equity"].cummax()
    mdd = ((df["equity"] - peak) / peak).min() * 100.0

    gross_profit = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    return {
        "Final Equity": float(final_equity),
        "Total Return": float(total_return),
        "MDD": float(mdd),
        "Profit Factor": float(profit_factor),
        "Win Rate": float(len(wins) / len(df) * 100.0 if len(df) > 0 else 0.0),
        "Avg Win": float(wins["pnl"].mean()) if len(wins) > 0 else 0.0,
        "Avg Loss": float(losses["pnl"].mean()) if len(losses) > 0 else 0.0,
        "Total Trades": int(len(df)),
    }

# ============================================================
# Objective (trial마다 엔진 새로 생성)
# ============================================================
def objective(trial, raw_data_map, universe):

    engine = BacktestEngine(days=30)

    # --- 유니버스 & RAW 고정 ---
    engine.symbols = universe
    engine.raw_data_map = {sym: raw_data_map[sym] for sym in universe}

    params = {
        # --- 기본 ---
        "atr_period": trial.suggest_int("atr_period", 14, 30),
        "atr_multiplier": trial.suggest_float("atr_multiplier", 2.0, 5.0, step=0.5),
        "adx_threshold": trial.suggest_int("adx_threshold", 0, 30),
        "rsi_upper": trial.suggest_int("rsi_upper", 60, 80),
        "rsi_lower": trial.suggest_int("rsi_lower", 20, 40),
        "vol_factor": trial.suggest_float("vol_factor", 0.8, 1.5, step=0.1),
        "ema_intraday": 200,
        "daily_ema": trial.suggest_int("daily_ema", 5, 20, step=5),

        # --- 구조 ---
        "swing_len": trial.suggest_int("swing_len", 3, 9, step=2),
        "context_lookback": trial.suggest_int("context_lookback", 60, 180, step=30),
        "retest_tolerance_atr": trial.suggest_float("retest_tolerance_atr", 0.15, 0.50, step=0.05),
    }

    engine.titan.set_params(params)

    try:
        engine.rebuild_indicators()
        engine.run(show_report=False)
    except Exception:
        return -1e9

    metrics = calculate_metrics(INITIAL_BALANCE, engine.executor.history)
    if metrics is None:
        return INITIAL_BALANCE

    record = {**params, **metrics, "trial_id": trial.number}
    df = pd.DataFrame([record])

    if not os.path.exists(RESULT_FILE):
        df.to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(RESULT_FILE, index=False, mode="a", header=False, encoding="utf-8-sig")

    return metrics["Final Equity"]

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    if os.path.exists(RESULT_FILE):
        os.remove(RESULT_FILE)

    print("🚀 Optimization Start (Fixed 20 Universe / RAW CSV)")

    # --------------------------------------------------------
    # RAW 데이터 로드 (1회)
    # --------------------------------------------------------
    if not os.path.exists(DATA_CACHE_FILE):
        raise RuntimeError("❌ RAW DATA CACHE NOT FOUND")

    with open(DATA_CACHE_FILE, "rb") as f:
        raw_cache = pickle.load(f) or {}

    # --------------------------------------------------------
    # 신규상장 제거 + 고정 20개 선정
    # --------------------------------------------------------
    filtered = []
    for sym, df in raw_cache.items():
        if df is not None and len(df) >= MIN_15M_ROWS:
            filtered.append(sym)

    filtered = sorted(filtered)
    if len(filtered) < FIXED_UNIVERSE_SIZE:
        raise RuntimeError(f"❌ Not enough symbols after filter: {len(filtered)}")

    universe = filtered[:FIXED_UNIVERSE_SIZE]

    # snapshot 저장
    with open(UNIVERSE_SNAPSHOT_FILE, "w", encoding="utf-8") as f:
        pd.DataFrame({"symbol": universe}).to_json(f, orient="records", indent=2)

    print(f"🧭 Fixed Universe ({len(universe)}): {universe}")
    print(f"📄 CSV Output: {RESULT_FILE}")

    # --------------------------------------------------------
    # Optuna
    # --------------------------------------------------------
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: objective(t, raw_cache, universe),
        n_trials=100,
        n_jobs=1,
    )

    print("✅ Optimization Finished")
    print(f"🏆 Best Equity: {study.best_value:,.2f}")
