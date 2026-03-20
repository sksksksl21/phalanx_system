import optuna
import pandas as pd
import numpy as np
import os
import pickle
import json
import random
from datetime import datetime
from core.backtest_engine import BacktestEngine

# ============================================================
# [Optuna Optimization] TitanStrategy 20260307 aligned
# - optimize_20260307의 frozen snapshot 구조 유지
# - titan_strategy_20260307.py의 실제 params 키만 최적화
# - 산출물은 optuna_runs/<run_id>/ 아래로 전부 격리
#
# 실행 전 필요 파일:
# - market_data_cache_7d.pkl (CURRENT_DIR 기준)
# ============================================================

# ------------------------------------------------------------
# Determinism
# ------------------------------------------------------------
SEED = 42


def set_global_determinism(seed: int = 42):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)


set_global_determinism(SEED)

# ------------------------------------------------------------
# Path / Config
# ------------------------------------------------------------
optuna.logging.set_verbosity(optuna.logging.WARNING)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_CACHE_FILE = os.path.join(CURRENT_DIR, "market_data_cache_7d.pkl")

INITIAL_BALANCE = 2500.0
MIN_15M_ROWS = 600
MIN_DAILY_ROWS = 40
FIXED_UNIVERSE_SIZE = 29

# Optuna settings
OPT_DAYS = 7
N_TRIALS = 300
N_JOBS = 1
MIN_TRADES_FOR_VALID = 25  # objective gate (too-low trades -> reject)

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def snapshot_universe(path: str, universe: list[str]):
    """
    backtest_engine이 바로 읽을 수 있는 포맷으로 저장
    """
    ensure_dir(os.path.dirname(path))
    payload = {
        "universe": [str(s).strip() for s in universe if str(s).strip()]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_metrics_row(csv_path: str, row: dict):
    df = pd.DataFrame([row])
    ensure_dir(os.path.dirname(csv_path))
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(csv_path, index=False, mode="a", header=False, encoding="utf-8-sig")


def load_cache_payload(cache_file: str):
    """
    pkl payload 로더
    지원:
    1) 신규 포맷:
       {
         "schema_version": 2,
         "meta": {...},
         "raw_15m_map": {...},
         "raw_daily_map": {...}
       }
    2) 구버전 포맷:
       {sym: df15m, ...}
    """
    with open(cache_file, "rb") as f:
        obj = pickle.load(f) or {}

    # 신규 payload
    if isinstance(obj, dict) and ("raw_15m_map" in obj or "raw_daily_map" in obj):
        raw_15m_map = obj.get("raw_15m_map", {}) or {}
        raw_daily_map = obj.get("raw_daily_map", {}) or {}
        meta = obj.get("meta", {}) or {}
        return raw_15m_map, raw_daily_map, meta

    # 구버전 fallback
    if isinstance(obj, dict):
        return obj, {}, {"schema_version": 1, "legacy_raw_only": True}

    raise RuntimeError("❌ Invalid cache payload format")


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def calculate_metrics(initial_balance: float, engine) -> dict | None:
    """
    BacktestEngine의 MTM equity 기준 성과 계산
    - Final Equity: engine.executor.equity
    - MDD: engine.executor.equity_curve
    - PF/WinRate 등은 executor.history(Exit net_pnl) 기반
    """
    if engine is None or not hasattr(engine, "executor") or engine.executor is None:
        return None

    ex = engine.executor

    # --- Final Equity / Total Return ---
    final_equity = float(getattr(ex, "equity", initial_balance))
    total_return = float((final_equity - initial_balance) / initial_balance * 100.0)

    # --- MDD from equity_curve ---
    curve = getattr(ex, "equity_curve", None) or []
    mdd = 0.0
    try:
        dfc = pd.DataFrame(curve)
        if not dfc.empty and "equity" in dfc.columns:
            eq = pd.to_numeric(dfc["equity"], errors="coerce").dropna()
            if len(eq):
                peak = eq.cummax()
                dd = (eq / peak.replace(0, np.nan)) - 1.0
                mdd = float(dd.min() * 100.0)
    except Exception:
        mdd = 0.0

    # --- Trade stats from history ---
    hist = getattr(ex, "history", None) or []
    df = pd.DataFrame(hist)
    if df.empty or "pnl" not in df.columns:
        total_trades = 0
        profit_factor = 0.0
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        risk_reward = 0.0
    else:
        df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0).astype(float)
        total_trades = int(len(df))

        wins = df[df["pnl"] > 0.0]
        losses = df[df["pnl"] <= 0.0]

        win_rate = float(len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
        sum_win = float(wins["pnl"].sum()) if not wins.empty else 0.0
        sum_loss = float(losses["pnl"].sum()) if not losses.empty else 0.0

        profit_factor = (
            float(sum_win / abs(sum_loss))
            if sum_loss < 0
            else float("inf")
            if sum_win > 0
            else 0.0
        )
        avg_win = float(wins["pnl"].mean()) if not wins.empty else 0.0
        avg_loss = float(losses["pnl"].mean()) if not losses.empty else 0.0
        risk_reward = float(abs(avg_win / avg_loss)) if avg_loss < 0 and avg_win > 0 else 0.0

    return {
        "Final Equity": final_equity,
        "Total Return": total_return,
        "MDD": mdd,
        "Profit Factor": profit_factor,
        "Win Rate": win_rate,
        "Avg Win": avg_win,
        "Avg Loss": avg_loss,
        "Risk:Reward": risk_reward,
        "Total Trades": total_trades,
    }


def _objective_score_from_metrics(metrics: dict) -> float:
    """
    Best 값은 Final Equity로 두되,
    참고용 Objective Score도 같이 저장
    """
    final_equity = float(metrics.get("Final Equity", 0.0) or 0.0)
    profit_factor = float(metrics.get("Profit Factor", 0.0) or 0.0)
    risk_reward = float(metrics.get("Risk:Reward", 0.0) or 0.0)
    win_rate = float(metrics.get("Win Rate", 0.0) or 0.0)
    mdd_abs = abs(float(metrics.get("MDD", 0.0) or 0.0))

    score = final_equity
    score += min(profit_factor, 4.0) * 250.0
    score += min(risk_reward, 2.0) * 120.0
    score += min(win_rate, 75.0) * 2.0
    score -= mdd_abs * 45.0

    return float(score)


def pick_top5_by_views(results_csv: str, min_trades: int = 30, topk: int = 5):
    df = pd.read_csv(results_csv, encoding="utf-8-sig")
    if df.empty:
        return {}

    for c in ["Final Equity", "Total Return", "MDD", "Profit Factor", "Total Trades"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df_f = df[df["Total Trades"] >= min_trades].copy() if "Total Trades" in df.columns else df.copy()
    if df_f.empty:
        df_f = df.copy()

    if "Total Return" in df_f.columns and "MDD" in df_f.columns:
        df_f["MDD_abs"] = df_f["MDD"].abs()
        df_f["Return_over_MDD"] = df_f["Total Return"] / df_f["MDD_abs"].replace(0, np.nan)
    else:
        df_f["MDD_abs"] = np.nan
        df_f["Return_over_MDD"] = np.nan

    views = {}
    if "Final Equity" in df_f.columns:
        views["equity_top"] = df_f.sort_values("Final Equity", ascending=False).head(topk)
    if "Profit Factor" in df_f.columns:
        views["pf_top"] = df_f.sort_values("Profit Factor", ascending=False).head(topk)
    if "MDD_abs" in df_f.columns:
        views["mdd_low"] = df_f.sort_values("MDD_abs", ascending=True).head(topk)
    if "Return_over_MDD" in df_f.columns:
        views["efficiency_top"] = df_f.sort_values("Return_over_MDD", ascending=False).head(topk)

    return views


def redirect_trial_outputs(engine, result_csv_path: str):
    """
    trial 중간 산출물을 run_dir 하위 trash로 몰아 루트 오염 방지
    """
    null_dir = ensure_dir(os.path.join(os.path.dirname(result_csv_path), "_trash_trial_outputs"))
    hist_path = os.path.join(null_dir, "backtest_history.csv")
    curve_path = os.path.join(null_dir, "backtest_equity_curve.csv")

    if hasattr(engine, "log_file"):
        engine.log_file = hist_path
    if hasattr(engine, "equity_curve_file"):
        engine.equity_curve_file = curve_path

    ex = getattr(engine, "executor", None)
    if ex is not None:
        if hasattr(ex, "log_file"):
            ex.log_file = hist_path
        if hasattr(ex, "equity_curve_file"):
            ex.equity_curve_file = curve_path


# ------------------------------------------------------------
# Strategy params: 20260307 aligned
# ------------------------------------------------------------
def _fixed_strategy_params() -> dict:
    """
    Continuation Pullback 전략은 이미 파라미터가 최소화되어 있으므로
    별도의 legacy fixed toggle을 주입하지 않는다.
    """
    return {}



def _build_params_for_trial(trial) -> dict:
    """
    Continuation Pullback 전략의 실제 사용 파라미터만 최적화
    """
    params = {
        # --- Core / Exit ---
        "atr_period": trial.suggest_int("atr_period", 12, 24, step=2),
        "atr_multiplier": trial.suggest_float("atr_multiplier", 1.50, 3.25, step=0.25),

        # --- Trend filters ---
        "adx_threshold": trial.suggest_int("adx_threshold", 10, 28, step=2),
        "daily_ema": trial.suggest_int("daily_ema", 15, 40, step=5),
        "ema_intraday": trial.suggest_int("ema_intraday", 100, 250, step=25),

        # --- Pullback continuation geometry ---
        "pullback_lookback": trial.suggest_int("pullback_lookback", 2, 6, step=1),
        "pullback_tolerance_atr": trial.suggest_float("pullback_tolerance_atr", 0.30, 1.00, step=0.05),
        "breakout_buffer_atr": trial.suggest_float("breakout_buffer_atr", 0.00, 0.20, step=0.02),
    }

    params.update(_fixed_strategy_params())
    return params




def _validate_param_keys(engine, params: dict):
    titan = getattr(engine, "titan", None)
    if titan is None or not hasattr(titan, "params"):
        return
    allowed = set(getattr(titan, "params", {}).keys())
    unknown = [k for k in params.keys() if k not in allowed]
    if unknown:
        raise KeyError(f"Unknown params for TitanStrategy: {unknown}")


# ------------------------------------------------------------
# Frozen snapshot build / clone
# ------------------------------------------------------------
def build_frozen_backtest_snapshot(universe: list[str], days: int) -> dict:
    """
    백테와 동일한 prepare_data 경로를 1회만 실행해서
    15m / daily / 1m 데이터를 모두 freeze 한다.
    """
    seed_engine = BacktestEngine(days=days)
    seed_engine.prepare_data(symbols=list(universe))

    loaded_universe = list(getattr(seed_engine, "symbols", []) or [])
    if not loaded_universe:
        raise RuntimeError("❌ Frozen snapshot build failed: no symbols loaded from prepare_data().")

    snapshot = {
        "universe": list(loaded_universe),
        "raw_15m_map": {},
        "raw_daily_map": {},
        "raw_1m_map": {},
        "data_1m_map": {},
    }

    for sym in loaded_universe:
        df15 = seed_engine.raw_data_map.get(sym)
        if df15 is None or getattr(df15, "empty", True):
            raise RuntimeError(f"❌ Frozen snapshot missing 15m data: {sym}")

        dfd = seed_engine.raw_daily_map.get(sym)
        if dfd is None or getattr(dfd, "empty", True):
            raise RuntimeError(f"❌ Frozen snapshot missing daily context: {sym}")

        snapshot["raw_15m_map"][sym] = df15.copy(deep=True) if hasattr(df15, "copy") else df15
        snapshot["raw_daily_map"][sym] = dfd.copy(deep=True) if hasattr(dfd, "copy") else dfd

        df1 = seed_engine.raw_1m_map.get(sym)
        if df1 is not None and not getattr(df1, "empty", True):
            copied_1m = df1.copy(deep=True) if hasattr(df1, "copy") else df1
            snapshot["raw_1m_map"][sym] = copied_1m
            snapshot["data_1m_map"][sym] = copied_1m.copy(deep=True) if hasattr(copied_1m, "copy") else copied_1m

    return snapshot


def _clone_cache_into_engine(engine, frozen_snapshot: dict):
    """
    frozen snapshot을 trial/export용 엔진에 깊은 복사로 주입
    """
    universe = list(frozen_snapshot.get("universe", []) or [])
    raw_15m_map = frozen_snapshot.get("raw_15m_map", {}) or {}
    raw_daily_map = frozen_snapshot.get("raw_daily_map", {}) or {}
    raw_1m_map = frozen_snapshot.get("raw_1m_map", {}) or {}
    data_1m_map = frozen_snapshot.get("data_1m_map", {}) or {}

    engine.symbols = list(universe)
    engine.raw_data_map = {}
    engine.raw_daily_map = {}
    engine.raw_1m_map = {}
    engine.data_1m_map = {}
    engine.data_map = {}
    engine.last_prices = {}

    for sym in universe:
        if sym not in raw_15m_map:
            raise KeyError(f"Missing frozen 15m data for symbol: {sym}")
        if sym not in raw_daily_map:
            raise KeyError(f"Missing frozen daily data for symbol: {sym}")

        df15 = raw_15m_map[sym]
        dfd = raw_daily_map[sym]

        engine.raw_data_map[sym] = df15.copy(deep=True) if hasattr(df15, "copy") else df15
        engine.raw_daily_map[sym] = dfd.copy(deep=True) if hasattr(dfd, "copy") else dfd

        if sym in raw_1m_map:
            df1_raw = raw_1m_map[sym]
            engine.raw_1m_map[sym] = df1_raw.copy(deep=True) if hasattr(df1_raw, "copy") else df1_raw

        if sym in data_1m_map:
            df1_data = data_1m_map[sym]
            engine.data_1m_map[sym] = df1_data.copy(deep=True) if hasattr(df1_data, "copy") else df1_data


# ------------------------------------------------------------
# Objective
# ------------------------------------------------------------
def objective(trial, frozen_snapshot, result_csv_path: str):
    """
    최적화 타깃은 Final Equity 고정
    """
    engine = BacktestEngine(days=OPT_DAYS)
    _clone_cache_into_engine(engine, frozen_snapshot)

    params = _build_params_for_trial(trial)
    _validate_param_keys(engine, params)

    engine.titan.set_params(params)
    redirect_trial_outputs(engine, result_csv_path)

    try:
        engine.rebuild_indicators()
        engine.run(show_report=False)
    except Exception:
        return -1e9

    metrics = calculate_metrics(INITIAL_BALANCE, engine)
    if metrics is None:
        return -1e9

    total_trades = int(metrics.get("Total Trades", 0) or 0)
    final_equity = float(metrics.get("Final Equity", 0.0) or 0.0)
    objective_score = _objective_score_from_metrics(metrics)

    if total_trades < MIN_TRADES_FOR_VALID:
        metrics["Rejected"] = 1
        metrics["Objective Score"] = float(objective_score)
        metrics["Optimize Target"] = "Final Equity"
        record = {**params, **metrics, "trial_id": trial.number}
        write_metrics_row(result_csv_path, record)
        return -1e9

    metrics["Rejected"] = 0
    metrics["Objective Score"] = float(objective_score)
    metrics["Optimize Target"] = "Final Equity"

    record = {**params, **metrics, "trial_id": trial.number}
    write_metrics_row(result_csv_path, record)

    return float(final_equity)


# ------------------------------------------------------------
# Export helper
# ------------------------------------------------------------
def run_one_and_export(frozen_snapshot, params: dict, out_dir: str, tag: str, days: int = 30):
    ensure_dir(out_dir)

    hist_path = os.path.join(out_dir, f"{tag}_backtest_history.csv")
    curve_path = os.path.join(out_dir, f"{tag}_equity_curve.csv")
    param_path = os.path.join(out_dir, f"{tag}_params.json")
    metric_path = os.path.join(out_dir, f"{tag}_metrics.json")

    engine = BacktestEngine(days=days)
    _clone_cache_into_engine(engine, frozen_snapshot)

    _validate_param_keys(engine, params)
    engine.titan.set_params(params)

    if hasattr(engine, "log_file"):
        engine.log_file = hist_path
        with open(engine.log_file, "w", encoding="utf-8") as f:
            f.write("Datetime,Symbol,Side,Type,Price,Amount,PnL,Cash,Equity,Reason\n")

    if hasattr(engine, "equity_curve_file"):
        engine.equity_curve_file = curve_path
        with open(engine.equity_curve_file, "w", encoding="utf-8") as f:
            f.write("Datetime,Equity\n")

    ex = getattr(engine, "executor", None)
    if ex is not None:
        if hasattr(ex, "log_file"):
            ex.log_file = hist_path
        if hasattr(ex, "equity_curve_file"):
            ex.equity_curve_file = curve_path

    engine.rebuild_indicators()
    engine.run(show_report=False)

    metrics = calculate_metrics(INITIAL_BALANCE, engine) or {}

    with open(param_path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def _extract_params_from_row(row: pd.Series) -> dict:
    def _to_int(x, default=0):
        if pd.isna(x):
            return int(default)
        return int(float(x))

    def _to_float(x, default=0.0):
        if pd.isna(x):
            return float(default)
        return float(x)

    params = _fixed_strategy_params()

    params.update({
        "atr_period": _to_int(row.get("atr_period"), 18),
        "atr_multiplier": _to_float(row.get("atr_multiplier"), 2.25),
        "adx_threshold": _to_int(row.get("adx_threshold"), 18),
        "daily_ema": _to_int(row.get("daily_ema"), 25),
        "ema_intraday": _to_int(row.get("ema_intraday"), 200),
        "pullback_lookback": _to_int(row.get("pullback_lookback"), 3),
        "pullback_tolerance_atr": _to_float(row.get("pullback_tolerance_atr"), 0.60),
        "breakout_buffer_atr": _to_float(row.get("breakout_buffer_atr"), 0.05),
    })

    return params




# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("🚀 Optuna Optimization Start (Backtest-Aligned Frozen Snapshot / Strategy_20260307)")

    if not os.path.exists(DATA_CACHE_FILE):
        raise RuntimeError(f"❌ RAW DATA CACHE NOT FOUND: {DATA_CACHE_FILE}")

    # cache는 universe 후보 선정용
    raw_15m_map, raw_daily_map, cache_meta = load_cache_payload(DATA_CACHE_FILE)

    if not raw_15m_map:
        raise RuntimeError("❌ 15m cache empty")

    if not raw_daily_map:
        raise RuntimeError("❌ Daily cache missing. Rebuild cache with down_pkl.py first.")

    daily_need_days = int(cache_meta.get("daily_need_days", MIN_DAILY_ROWS) or MIN_DAILY_ROWS)

    # --------------------------------------------------------
    # Run directory
    # --------------------------------------------------------
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(os.path.join(CURRENT_DIR, "optuna_runs", run_id))

    results_csv = os.path.join(run_dir, "optimization_results.csv")
    universe_json = os.path.join(run_dir, "universe_optimization.json")

    print(f"📦 Run Dir: {run_dir}")
    print(f"📄 Results CSV: {results_csv}")
    print(
        f"📚 Cache Loaded | 15m={len(raw_15m_map)} 1d={len(raw_daily_map)} "
        f"daily_need_days>={daily_need_days}"
    )

    # --------------------------------------------------------
    # Build fixed universe (candidate selection only)
    # --------------------------------------------------------
    filtered = []
    common_syms = sorted(set(raw_15m_map.keys()) & set(raw_daily_map.keys()))

    for sym in common_syms:
        try:
            df15 = raw_15m_map.get(sym)
            dfd = raw_daily_map.get(sym)

            if df15 is None or dfd is None:
                continue
            if len(df15) < MIN_15M_ROWS:
                continue
            if len(dfd) < daily_need_days:
                continue

            filtered.append(sym)
        except Exception:
            continue

    filtered = sorted(filtered)
    if len(filtered) < FIXED_UNIVERSE_SIZE:
        raise RuntimeError(f"❌ Not enough symbols after filter: {len(filtered)}")

    universe = filtered[:FIXED_UNIVERSE_SIZE]
    print(f"🧭 Candidate Universe ({len(universe)}): {universe}")

    # --------------------------------------------------------
    # Build backtest-aligned frozen snapshot
    # --------------------------------------------------------
    frozen_snapshot = build_frozen_backtest_snapshot(universe, days=OPT_DAYS)
    universe = list(frozen_snapshot.get("universe", []) or [])

    if not universe:
        raise RuntimeError("❌ Frozen snapshot universe empty")

    snapshot_universe(universe_json, universe)

    print(
        f"🧊 Frozen Snapshot Ready | "
        f"symbols={len(universe)} "
        f"15m={len(frozen_snapshot.get('raw_15m_map', {}))} "
        f"1d={len(frozen_snapshot.get('raw_daily_map', {}))} "
        f"1m={len(frozen_snapshot.get('raw_1m_map', {}))}"
    )

    # --------------------------------------------------------
    # Optuna
    # --------------------------------------------------------
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(
        lambda t: objective(t, frozen_snapshot, results_csv),
        n_trials=N_TRIALS,
        n_jobs=N_JOBS,
    )

    print("✅ Optimization Finished")
    print(f"🏆 Best Final Equity: {study.best_value:,.2f}")

    # --------------------------------------------------------
    # Reports: top5 by multiple views + best vs baseline
    # --------------------------------------------------------
    reports_dir = ensure_dir(os.path.join(run_dir, "reports_top5"))

    views = pick_top5_by_views(results_csv, min_trades=max(1, MIN_TRADES_FOR_VALID), topk=5)
    priority = ["equity_top", "efficiency_top", "pf_top", "mdd_low"]

    picked = []
    seen = set()
    for v in priority:
        if v not in views:
            continue
        dfv = views[v]
        for _, row in dfv.iterrows():
            p = _extract_params_from_row(row)
            sig = tuple(sorted(p.items()))
            if sig in seen:
                continue
            seen.add(sig)
            picked.append((v, int(row.get("trial_id", -1)), p))
            if len(picked) >= 5:
                break
        if len(picked) >= 5:
            break

    with open(os.path.join(reports_dir, "universe.json"), "w", encoding="utf-8") as f:
        json.dump({"universe": universe}, f, ensure_ascii=False, indent=2)

    for i, (view_name, trial_id, params) in enumerate(picked, start=1):
        view_dir = ensure_dir(os.path.join(reports_dir, view_name))
        tag = f"rank_{i:02d}_trial{trial_id}"
        print(f"📌 Exporting {view_name}/{tag}")
        run_one_and_export(frozen_snapshot, params, view_dir, tag=tag, days=OPT_DAYS)

    for v, dfv in views.items():
        dfv.to_csv(os.path.join(reports_dir, f"top5_{v}.csv"), index=False, encoding="utf-8-sig")

    # Best / baseline
    best_params = _fixed_strategy_params()
    best_params.update(dict(study.best_trial.params))
    best_dir = ensure_dir(os.path.join(reports_dir, "best"))

    base_engine = BacktestEngine(days=OPT_DAYS)
    baseline_params = dict(getattr(base_engine.titan, "params", {}))
    baseline_params.update(_fixed_strategy_params())

    keep_keys = set(_extract_params_from_row(pd.Series({})).keys())
    baseline_params = {k: baseline_params[k] for k in keep_keys if k in baseline_params}

    baseline_dir = ensure_dir(os.path.join(reports_dir, "baseline"))

    print("📌 Exporting BEST report.")
    run_one_and_export(frozen_snapshot, best_params, best_dir, tag="best", days=OPT_DAYS)

    print("📌 Exporting BASELINE report.")
    run_one_and_export(frozen_snapshot, baseline_params, baseline_dir, tag="baseline", days=OPT_DAYS)

    with open(os.path.join(reports_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    with open(os.path.join(reports_dir, "baseline_params.json"), "w", encoding="utf-8") as f:
        json.dump(baseline_params, f, ensure_ascii=False, indent=2)

    print(f"✅ Reports saved under: {reports_dir}")


if __name__ == "__main__":
    main()