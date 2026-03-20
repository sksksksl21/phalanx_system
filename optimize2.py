import optuna
import pandas as pd
import numpy as np
import os
import time
import pickle
from core.backtest_engine import BacktestEngine
import json
from datetime import datetime


# ============================================================
# Optuna / Path
# ============================================================
optuna.logging.set_verbosity(optuna.logging.WARNING)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_FILE = os.path.join(CURRENT_DIR, "optimization_results.csv")
DATA_CACHE_FILE = os.path.join(CURRENT_DIR, "market_data_cache_7d.pkl")
UNIVERSE_SNAPSHOT_FILE = os.path.join(CURRENT_DIR, "universe_optimization.json")

INITIAL_BALANCE = 10000.0
MIN_15M_ROWS = 600
FIXED_UNIVERSE_SIZE = 24


def _ts_run_id():
    # 폴더명 안전하게
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path





def snapshot_universe(path: str, universe: list):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{"symbol": s} for s in universe], f, ensure_ascii=False, indent=2)

def write_metrics_row(csv_path: str, row: dict):
    df = pd.DataFrame([row])
    ensure_dir(os.path.dirname(csv_path))
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(csv_path, index=False, mode="a", header=False, encoding="utf-8-sig")
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


def pick_top5_by_views(results_csv: str, min_trades: int = 30, topk: int = 5):
    df = pd.read_csv(results_csv, encoding="utf-8-sig")
    if df.empty:
        return {}

    # 숫자 컬럼 변환 (안전)
    for c in ["Final Equity", "Total Return", "MDD", "Profit Factor", "Total Trades"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 기본 필터
    if "Total Trades" in df.columns:
        df_f = df[df["Total Trades"] >= min_trades].copy()
    else:
        df_f = df.copy()

    if df_f.empty:
        df_f = df.copy()

    # 파생 지표
    if "Total Return" in df_f.columns and "MDD" in df_f.columns:
        # MDD는 음수(예: -12.3)일 수 있어서 abs로
        df_f["MDD_abs"] = df_f["MDD"].abs()
        df_f["Return_over_MDD"] = df_f["Total Return"] / df_f["MDD_abs"].replace(0, np.nan)
    else:
        df_f["Return_over_MDD"] = np.nan
        df_f["MDD_abs"] = np.nan

    views = {}

    # A) Final Equity 상위
    if "Final Equity" in df_f.columns:
        views["equity_top"] = df_f.sort_values("Final Equity", ascending=False).head(topk)

    # B) Profit Factor 상위
    if "Profit Factor" in df_f.columns:
        views["pf_top"] = df_f.sort_values("Profit Factor", ascending=False).head(topk)

    # C) MDD 최소(방어형) -> abs 작은 순
    if "MDD_abs" in df_f.columns:
        views["mdd_low"] = df_f.sort_values("MDD_abs", ascending=True).head(topk)

    # D) Return/MDD 효율 상위
    if "Return_over_MDD" in df_f.columns:
        views["efficiency_top"] = df_f.sort_values("Return_over_MDD", ascending=False).head(topk)

    return views

def redirect_trial_outputs(engine, result_csv_path: str):
    """
    Optuna trial 중 발생하는 backtest_history / equity_curve 출력을
    run_dir/_trash_trial_outputs 로 강제 리다이렉트한다.
    같은 두 파일만 계속 덮어써서 루트 오염을 차단한다.
    """
    null_dir = ensure_dir(
        os.path.join(os.path.dirname(result_csv_path), "_trash_trial_outputs")
    )

    hist_path = os.path.join(null_dir, "backtest_history.csv")
    curve_path = os.path.join(null_dir, "backtest_equity_curve.csv")

    # engine 레벨
    if hasattr(engine, "log_file"):
        engine.log_file = hist_path
    if hasattr(engine, "equity_curve_file"):
        engine.equity_curve_file = curve_path

    # executor 레벨
    ex = getattr(engine, "executor", None)
    if ex is not None:
        if hasattr(ex, "log_file"):
            ex.log_file = hist_path
        if hasattr(ex, "equity_curve_file"):
            ex.equity_curve_file = curve_path

# ============================================================
# Objective (trial마다 엔진 새로 생성)
# ============================================================
def objective(trial, raw_data_map, universe, result_csv_path: str):

    engine = BacktestEngine(days=7)

    # --- 유니버스 & RAW 고정 ---
    engine.symbols = universe
    engine.raw_data_map = {sym: raw_data_map[sym] for sym in universe}

    params = {
        # --- 기존 튜닝 대상(유지) ---
        "atr_period": trial.suggest_int("atr_period", 10, 30),
        "atr_multiplier": trial.suggest_float("atr_multiplier", 1.5, 5.0, step=0.25),
        "adx_threshold": trial.suggest_int("adx_threshold", 0, 30),
        "rsi_upper": trial.suggest_int("rsi_upper", 60, 85),
        "rsi_lower": trial.suggest_int("rsi_lower", 15, 45),
        "vol_factor": trial.suggest_float("vol_factor", 0.6, 1.6, step=0.1),

        "ema_intraday": 200,  # 고정 유지(전략 설계상)
        "daily_ema": trial.suggest_int("daily_ema", 5, 30, step=5),

        "swing_len": trial.suggest_int("swing_len", 3, 9, step=2),
        "context_lookback": trial.suggest_int("context_lookback", 60, 210, step=30),
        "retest_tolerance_atr": trial.suggest_float("retest_tolerance_atr", 0.15, 0.60, step=0.05),

        # --- NEW: Gate / Confirmation 파라미터(추가) ---
        "use_daily_filter": trial.suggest_int("use_daily_filter", 0, 1),
        "use_vol_filter": trial.suggest_int("use_vol_filter", 0, 1),
        "use_st_dir_filter": trial.suggest_int("use_st_dir_filter", 0, 1),

        "use_structure_confirm": trial.suggest_int("use_structure_confirm", 0, 1),
        "structure_min_pivots": trial.suggest_int("structure_min_pivots", 2, 4),

        "use_vol_regime_gate": trial.suggest_int("use_vol_regime_gate", 0, 1),
        "atr_regime_len": trial.suggest_int("atr_regime_len", 30, 120, step=10),
        "atr_regime_factor": trial.suggest_float("atr_regime_factor", 1.00, 1.30, step=0.02),
        "atr_slope_gate": trial.suggest_int("atr_slope_gate", 0, 1),
    }

    engine.titan.set_params(params)
    redirect_trial_outputs(engine, result_csv_path)

    try:
        engine.rebuild_indicators()
        engine.run(show_report=False)
    except Exception:
        return -1e9

    metrics = calculate_metrics(INITIAL_BALANCE, engine.executor.history)
    if metrics is None:
        return INITIAL_BALANCE

    record = {**params, **metrics, "trial_id": trial.number}
    write_metrics_row(result_csv_path, record)

    return float(metrics["Final Equity"])



def run_one_and_export(raw_cache, universe, params: dict, out_dir: str, tag: str, days: int = 30):
    ensure_dir(out_dir)

    hist_path = os.path.join(out_dir, f"{tag}_backtest_history.csv")
    curve_path = os.path.join(out_dir, f"{tag}_equity_curve.csv")
    param_path = os.path.join(out_dir, f"{tag}_params.json")
    metric_path = os.path.join(out_dir, f"{tag}_metrics.json")

    engine = BacktestEngine(days=days)

    engine.symbols = universe
    engine.raw_data_map = {sym: raw_cache[sym] for sym in universe}

    engine.titan.set_params(params)

    # ✅ 루트 오염 차단: 엔진 + executor 모두 경로 강제
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

    metrics = calculate_metrics(INITIAL_BALANCE, engine.executor.history) or {}

    with open(param_path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    print("🚀 Optimization Start (Fixed Universe / RAW Cache)")

    if not os.path.exists(DATA_CACHE_FILE):
        raise RuntimeError("❌ RAW DATA CACHE NOT FOUND")

    with open(DATA_CACHE_FILE, "rb") as f:
        raw_cache = pickle.load(f) or {}

    # --------------------------------------------------------
    # run 폴더 생성 (이번 실행 산출물 전부 격리)
    # --------------------------------------------------------
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(os.path.join(CURRENT_DIR, "optuna_runs", run_id))

    RESULT_FILE = os.path.join(run_dir, "optimization_results.csv")
    UNIVERSE_SNAPSHOT_FILE = os.path.join(run_dir, "universe_optimization.json")

    print(f"📦 Run Dir: {run_dir}")
    print(f"📄 Results CSV: {RESULT_FILE}")

    # --------------------------------------------------------
    # 신규상장 제거 + 고정 유니버스 선정
    # --------------------------------------------------------
    filtered = []
    for sym, df in raw_cache.items():
        if df is not None and len(df) >= MIN_15M_ROWS:
            filtered.append(sym)

    filtered = sorted(filtered)
    if len(filtered) < FIXED_UNIVERSE_SIZE:
        raise RuntimeError(f"❌ Not enough symbols after filter: {len(filtered)}")

    universe = filtered[:FIXED_UNIVERSE_SIZE]
    snapshot_universe(UNIVERSE_SNAPSHOT_FILE, universe)

    print(f"🧭 Fixed Universe ({len(universe)}): {universe}")

    # --------------------------------------------------------
    # Optuna
    # --------------------------------------------------------
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: objective(t, raw_cache, universe, RESULT_FILE),
        n_trials=300,
        n_jobs=1,
    )

    print("✅ Optimization Finished")
    print(f"🏆 Best Equity: {study.best_value:,.2f}")

    # --------------------------------------------------------
    # 관점별 Top5 추출 + 최종 백테 5개만 저장
    # --------------------------------------------------------
    reports_dir = os.path.join(run_dir, "reports_top5")
    ensure_dir(reports_dir)

    views = pick_top5_by_views(RESULT_FILE, min_trades=30, topk=5)

    # 중복 제거하면서 최대 5개만 최종 선정 (우선순위: equity_top -> efficiency_top -> pf_top -> mdd_low)
    priority = ["equity_top", "efficiency_top", "pf_top", "mdd_low"]
    picked = []
    seen = set()

    for v in priority:
        if v not in views:
            continue
        dfv = views[v]
        for _, row in dfv.iterrows():
            # 파라미터 키들만 뽑기: metrics 컬럼 제외
            # (너 코드 기준 파라미터 키 리스트를 고정으로 가져감)
            p = {
                "atr_period": int(row["atr_period"]),
                "atr_multiplier": float(row["atr_multiplier"]),
                "adx_threshold": int(row["adx_threshold"]),
                "rsi_upper": int(row["rsi_upper"]),
                "rsi_lower": int(row["rsi_lower"]),
                "vol_factor": float(row["vol_factor"]),
                "ema_intraday": 200,
                "daily_ema": int(row["daily_ema"]),
                "swing_len": int(row["swing_len"]),
                "context_lookback": int(row["context_lookback"]),
                "retest_tolerance_atr": float(row["retest_tolerance_atr"]),

                # --- NEW 추가 ---
                "use_daily_filter": int(row.get("use_daily_filter", 1)),
                "use_vol_filter": int(row.get("use_vol_filter", 1)),
                "use_st_dir_filter": int(row.get("use_st_dir_filter", 1)),
                "use_structure_confirm": int(row.get("use_structure_confirm", 1)),
                "structure_min_pivots": int(row.get("structure_min_pivots", 2)),
                "use_vol_regime_gate": int(row.get("use_vol_regime_gate", 1)),
                "atr_regime_len": int(row.get("atr_regime_len", 50)),
                "atr_regime_factor": float(row.get("atr_regime_factor", 1.05)),
                "atr_slope_gate": int(row.get("atr_slope_gate", 1)),
            }

            sig = tuple(sorted(p.items()))
            if sig in seen:
                continue
            seen.add(sig)
            picked.append((v, int(row.get("trial_id", -1)), p))
            if len(picked) >= 5:
                break
        if len(picked) >= 5:
            break

    # 유니버스 스냅샷도 같이 저장
    with open(os.path.join(reports_dir, "universe.json"), "w", encoding="utf-8") as f:
        json.dump([{"symbol": s} for s in universe], f, ensure_ascii=False, indent=2)

    # 5개만 재실행 저장
    for i, (view_name, trial_id, params) in enumerate(picked, start=1):
        view_dir = os.path.join(reports_dir, view_name)
        ensure_dir(view_dir)

        tag = f"rank_{i:02d}_trial{trial_id}"
        print(f"📌 Exporting {view_name}/{tag}")
        run_one_and_export(raw_cache, universe, params, view_dir, tag=tag, days=7)

    # 관점별 top5 테이블도 저장(재현용)
    for v, dfv in views.items():
        dfv.to_csv(os.path.join(reports_dir, f"top5_{v}.csv"), index=False, encoding="utf-8-sig")

    print(f"✅ Top5 reports saved: {reports_dir}")

    # --------------------------------------------------------
    # 비교용 리포트 생성: BEST vs BASELINE
    # --------------------------------------------------------
    best_params = dict(study.best_trial.params)

    best_out = os.path.join(reports_dir, "best")
    base_out = os.path.join(reports_dir, "baseline")
    ensure_dir(best_out)
    ensure_dir(base_out)


    # baseline 정의: “옵티 범위와 동일한 키”를 중간값/기본값으로 맞춰둔 기준 세트
    baseline_params = {
        "atr_period": 15,
        "atr_multiplier": 2.0,
        "adx_threshold": 16,
        "rsi_upper": 67,
        "rsi_lower": 30,
        "vol_factor": 0.8,
        "ema_intraday": 200,
        "daily_ema": 15,
        "swing_len": 3,
        "context_lookback": 150,
        "retest_tolerance_atr": 0.40,

        # --- NEW (전략 기본값과 일치) ---
        "use_daily_filter": 1,
        "use_vol_filter": 1,
        "use_st_dir_filter": 1,
        "use_structure_confirm": 1,
        "structure_min_pivots": 2,
        "use_vol_regime_gate": 1,
        "atr_regime_len": 50,
        "atr_regime_factor": 1.05,
        "atr_slope_gate": 1,
    }

    print("📌 Exporting BEST report...")
    run_one_and_export(raw_cache, universe, best_params, best_out, tag="best", days=7)

    print("📌 Exporting BASELINE report...")
    run_one_and_export(raw_cache, universe, baseline_params, base_out, tag="baseline", days=7)

    # best 파라미터도 같이 저장
    with open(os.path.join(reports_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    with open(os.path.join(reports_dir, "baseline_params.json"), "w", encoding="utf-8") as f:
        json.dump(baseline_params, f, ensure_ascii=False, indent=2)

    print(f"✅ Reports saved under: {reports_dir}")
