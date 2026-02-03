# optimize_sl.py
# =========================================================
# SL-only Optimizer (Optuna) - PKL 기반 (NO FETCH) / config 전략 기반
#
# ✅ 핵심 동작
# - config.json(=BacktestEngine().cfg)에서 sl_strategy 읽음 (supertrend/atr_trail/profit_lock/hybrid/armor)
# - 그 전략에 "실제로 쓰이는 sl_params 키만" 최적화
# - RAW cache (down_pkl.py로 만든 market_data_cache_30d.pkl)만 사용 (절대 fetch 안 함)
# - trial 결과는 좋든 나쁘든 전부 CSV 기록 (passed/fail_reason 포함)
# - BacktestEngine cfg in-memory 주입 (config.json write 없음)
# - cfg 덮은 뒤 RiskControl 재생성 (risk_settings 참조 일치)
# =========================================================

import os
import sys
import json
import copy
import argparse
import pickle
import pandas as pd


# ---- PATH SETUP (same style as your project) ----
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# ---- Import BacktestEngine + RiskControl ----
try:
    from core.backtest_engine import BacktestEngine
except Exception:
    try:
        from backtest_engine import BacktestEngine
    except Exception as e:
        raise ImportError(
            f"❌ Cannot import BacktestEngine. Fix import path in optimize_sl.py. Original error: {e}"
        )

try:
    from strategy.risk_control import RiskControl
except Exception as e:
    raise ImportError(
        f"❌ Cannot import RiskControl. Fix import path. Original error: {e}"
    )

RESULT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sl_optimization_results.csv")


# =========================================================
# 0) helpers: cfg에서 sl 전략/모드/기존 params 읽기
# =========================================================
def _get_cfg_sl_strategy(cfg: dict) -> str:
    try:
        strat = str((cfg.get("system_settings", {}) or {}).get("sl_strategy", "supertrend")).strip().lower()
    except Exception:
        strat = "supertrend"
    allowed = {"supertrend", "atr_trail", "profit_lock", "hybrid", "armor"}
    return strat if strat in allowed else "supertrend"


def _get_cfg_sl_apply_mode(cfg: dict) -> str:
    try:
        mode = str((cfg.get("system_settings", {}) or {}).get("sl_apply_mode", "next")).strip().lower()
    except Exception:
        mode = "next"
    return mode if mode in ("next", "same") else "next"


def _get_cfg_sl_params(cfg: dict) -> dict:
    try:
        p = (cfg.get("system_settings", {}) or {}).get("sl_params", {}) or {}
        return p if isinstance(p, dict) else {}
    except Exception:
        return {}


# =========================================================
# 1) Search Space (전략별로 "쓰는 키만")
# =========================================================
def suggest_sl_params_by_strategy(trial, sl_strategy: str, base_sl_params: dict):
    """
    sl_strategy에 따라 '실제로 쓰는' 키만 suggest.
    base_sl_params는 config에 있는 기존 값(있으면)로 fallback.
    """
    strat = str(sl_strategy or "supertrend").strip().lower()
    base = base_sl_params or {}
    p = {}

    if strat == "supertrend":
        # supertrend는 sl_params를 쓰지 않음 (PositionMonitor 기준)
        return {}

    if strat == "atr_trail":
        # uses: atr_mult
        default = float(base.get("atr_mult", 3.0) or 3.0)
        p["atr_mult"] = trial.suggest_float("atr_mult", 1.0, 8.0, step=0.25)
        return p

    if strat == "profit_lock":
        # uses: trigger_atr, lock_atr
        p["trigger_atr"] = trial.suggest_float("trigger_atr", 0.5, 4.0, step=0.1)
        p["lock_atr"] = trial.suggest_float("lock_atr", 0.0, 2.0, step=0.05)
        return p

    if strat == "hybrid":
        # hybrid = supertrend + atr_trail -> params는 사실상 atr_trail 쪽(atr_mult)만
        p["atr_mult"] = trial.suggest_float("atr_mult", 1.0, 8.0, step=0.25)
        return p

    if strat == "armor":
        # armor 키 풀셋 (네 PositionMonitor._candidate_armor 기준)
        p["adx_period"] = trial.suggest_int("adx_period", 10, 20)
        p["adx_trend"] = trial.suggest_int("adx_trend", 16, 32)

        p["atr_period"] = trial.suggest_int("atr_period", 10, 30)

        p["atr_mult_trend"] = trial.suggest_float("atr_mult_trend", 2.5, 6.0, step=0.25)
        p["atr_mult_chop"] = trial.suggest_float("atr_mult_chop", 1.0, 4.0, step=0.25)

        p["swing_len"] = trial.suggest_int("swing_len", 3, 12)
        p["structure_buffer_atr"] = trial.suggest_float("structure_buffer_atr", 0.0, 0.8, step=0.05)

        p["profit_trigger_atr"] = trial.suggest_float("profit_trigger_atr", 0.6, 2.5, step=0.1)
        p["profit_lock_atr"] = trial.suggest_float("profit_lock_atr", 0.0, 1.0, step=0.05)
        p["fee_buffer_bps"] = trial.suggest_int("fee_buffer_bps", 0, 20)

        p["min_move_atr"] = trial.suggest_float("min_move_atr", 0.0, 0.6, step=0.05)
        p["max_step_atr"] = trial.suggest_float("max_step_atr", 0.5, 4.0, step=0.25)

        p["armor_lookback"] = trial.suggest_int("armor_lookback", 150, 800, step=50)
        return p

    # unknown -> 안전하게 supertrend 취급
    return {}


# =========================================================
# 2) Cache Loader (NO FETCH)
# =========================================================
def load_raw_cache(cache_file: str):
    if not os.path.exists(cache_file):
        raise RuntimeError(f"❌ RAW CACHE NOT FOUND: {cache_file}")

    with open(cache_file, "rb") as f:
        raw_cache = pickle.load(f) or {}

    if not isinstance(raw_cache, dict) or not raw_cache:
        raise RuntimeError("❌ RAW CACHE is empty or invalid dict")

    return raw_cache


def pick_fixed_universe(raw_cache: dict, fixed_size: int, min_rows: int):
    filtered = []
    for sym, df in raw_cache.items():
        try:
            rows = len(df) if df is not None else 0
        except Exception:
            rows = 0
        if rows >= int(min_rows):
            filtered.append(sym)

    filtered = sorted(filtered)
    if len(filtered) < int(fixed_size):
        raise RuntimeError(f"❌ Not enough symbols after filter: {len(filtered)} < {fixed_size}")

    return filtered[:int(fixed_size)]


# =========================================================
# 3) Runner: single backtest with SL-only cfg injection + RAW cache injection
# =========================================================
def run_backtest_sl_only(
    base_cfg: dict,
    raw_cache: dict,
    universe: list,
    days: int = 30,
    sl_strategy: str = "supertrend",
    sl_apply_mode: str = "next",
    sl_params: dict | None = None,
):
    """
    - base_cfg 복사 후 system_settings.{sl_strategy, sl_apply_mode, sl_params}만 덮음
    - BacktestEngine 생성 -> cfg in-memory 주입
    - cfg 주입 후 RiskControl 재생성
    - raw_data_map 주입 (NO FETCH)
    """
    cfg = copy.deepcopy(base_cfg) if isinstance(base_cfg, dict) else {}
    cfg.setdefault("system_settings", {})
    cfg["system_settings"]["sl_strategy"] = str(sl_strategy)
    cfg["system_settings"]["sl_apply_mode"] = str(sl_apply_mode)
    cfg["system_settings"]["sl_params"] = dict(sl_params or {})

    engine = BacktestEngine(days=int(days))

    # 1) cfg 덮기
    engine.cfg = cfg

    # 2) cfg 덮었으면 RiskControl도 재생성 (config 참조 일치)
    engine.risk_ctrl = RiskControl(engine.executor, engine.cfg)

    # 3) RAW cache 주입 (NO FETCH)
    engine.symbols = list(universe)
    engine.raw_data_map = {sym: raw_cache[sym] for sym in universe}

    # raw_data_map이 이미 있으니 prepare_data()는 다운로드 안 함
    engine.prepare_data()
    engine.rebuild_indicators()
    engine.run(show_report=False)

    final_equity = float(getattr(engine.executor, "equity", 0.0))

    equity_curve = getattr(engine.executor, "equity_curve", []) or []
    hist = getattr(engine.executor, "history", []) or []

    pnls = []
    for h in hist:
        if isinstance(h, dict) and h.get("type") == "EXIT":
            try:
                pnls.append(float(h.get("pnl", 0.0)))
            except Exception:
                pass

    trades = len(pnls)
    wins = sum(1 for x in pnls if x > 0)
    winrate = (wins / trades) if trades > 0 else 0.0

    gross_win = sum(x for x in pnls if x > 0)
    gross_loss = -sum(x for x in pnls if x <= 0)
    pf = (gross_win / gross_loss) if gross_loss > 0 else (gross_win if gross_win > 0 else 0.0)

    # MDD (0~1)
    mdd = 0.0
    if equity_curve:
        eqs = []
        for p in equity_curve:
            try:
                eqs.append(float(p.get("equity", 0.0)))
            except Exception:
                pass

        peak = -1e18
        dd_max = 0.0
        for e in eqs:
            if e > peak:
                peak = e
            if peak > 0:
                dd = (peak - e) / peak
                if dd > dd_max:
                    dd_max = dd
        mdd = float(dd_max)

    return {
        "final_equity": final_equity,
        "mdd": mdd,
        "trades": trades,
        "winrate": winrate,
        "pf": float(pf),
    }


# =========================================================
# 4) Objective: 기록은 무조건, score는 필터 반영
# =========================================================
def objective_sl_only(
    trial,
    base_cfg,
    raw_cache,
    universe,
    sl_strategy: str,
    sl_apply_mode: str,
    base_sl_params: dict,
    days=30,
    mdd_penalty=2.0,
    min_trades=30,
    min_pf=1.05,
):
    # 전략별 suggest
    sl_params = suggest_sl_params_by_strategy(trial, sl_strategy=sl_strategy, base_sl_params=base_sl_params)

    r = run_backtest_sl_only(
        base_cfg=base_cfg,
        raw_cache=raw_cache,
        universe=universe,
        days=days,
        sl_strategy=sl_strategy,
        sl_apply_mode=sl_apply_mode,
        sl_params=sl_params,
    )

    final_equity = float(r["final_equity"])
    mdd = float(r["mdd"])
    trades = int(r["trades"])
    pf = float(r["pf"])
    winrate = float(r["winrate"])

    # ---- hard filters ----
    passed = True
    fail_reasons = []

    if trades < int(min_trades):
        passed = False
        fail_reasons.append("MIN_TRADES")
    if pf < float(min_pf):
        passed = False
        fail_reasons.append("MIN_PF")

    fail_reason = "|".join(fail_reasons) if fail_reasons else ""

    # score
    score = final_equity - (float(mdd_penalty) * mdd * 10000.0)
    if not passed:
        score = -1e9

    # ✅ CSV는 무조건 기록
    record = {
        "trial_id": trial.number,
        "sl_strategy": str(sl_strategy),
        "sl_apply_mode": str(sl_apply_mode),
        "score": float(score),
        "passed": bool(passed),
        "fail_reason": str(fail_reason),
        "final_equity": float(final_equity),
        "mdd": float(mdd),
        "pf": float(pf),
        "winrate": float(winrate),
        "trades": int(trades),
        **(sl_params or {}),
    }

    df = pd.DataFrame([record])
    if not os.path.exists(RESULT_FILE):
        df.to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(RESULT_FILE, index=False, mode="a", header=False, encoding="utf-8-sig")

    return float(score)


# =========================================================
# 5) Helpers
# =========================================================
def load_base_cfg_from_engine(days=30):
    engine0 = BacktestEngine(days=int(days))
    base_cfg = engine0.cfg if isinstance(engine0.cfg, dict) else {}
    return base_cfg


# =========================================================
# 6) Main
# =========================================================
def main():
    try:
        import optuna
    except Exception as e:
        raise ImportError(
            "❌ optuna not installed. Install: pip install optuna\n"
            f"Original error: {e}"
        )

    # 항상 새로
    if os.path.exists(RESULT_FILE):
        os.remove(RESULT_FILE)

    ap = argparse.ArgumentParser(description="SL-only optimizer using RAW cache (NO FETCH) + config sl_strategy")
    ap.add_argument("--trials", type=int, default=150, help="number of optuna trials")
    ap.add_argument("--days", type=int, default=30, help="backtest days (must match cache)")
    ap.add_argument("--cache", type=str, default="market_data_cache_30d.pkl", help="pkl cache file from down_pkl.py")
    ap.add_argument("--min_rows", type=int, default=2800, help="min 15m rows for symbol filter")
    ap.add_argument("--universe_size", type=int, default=25, help="fixed universe size")
    ap.add_argument("--seed", type=int, default=42, help="sampler seed")
    ap.add_argument("--mdd_penalty", type=float, default=2.0, help="penalty multiplier for MDD")
    ap.add_argument("--min_trades", type=int, default=30, help="minimum EXIT trades")
    ap.add_argument("--min_pf", type=float, default=1.05, help="minimum profit factor filter")
    ap.add_argument("--study_name", type=str, default="SL_ONLY", help="optuna study name")
    ap.add_argument("--storage", type=str, default=None, help="optuna storage url, e.g. sqlite:///sl_opt.db")
    ap.add_argument("--universe_snapshot", type=str, default="universe_sl_opt.json", help="save fixed universe list")
    args = ap.parse_args()

    raw_cache = load_raw_cache(args.cache)
    universe = pick_fixed_universe(raw_cache, fixed_size=args.universe_size, min_rows=args.min_rows)

    # snapshot 저장
    try:
        with open(args.universe_snapshot, "w", encoding="utf-8") as f:
            json.dump([{"symbol": s} for s in universe], f, ensure_ascii=False, indent=2)
        print(f"🧭 Fixed Universe ({len(universe)}): saved to {args.universe_snapshot}")
    except Exception as e:
        print(f"⚠️ Failed to save universe snapshot: {e}")

    base_cfg = load_base_cfg_from_engine(days=args.days)

    # ✅ 여기서 config 기반으로 전략/모드 결정
    sl_strategy = _get_cfg_sl_strategy(base_cfg)
    sl_apply_mode = _get_cfg_sl_apply_mode(base_cfg)
    base_sl_params = _get_cfg_sl_params(base_cfg)

    print(f"🧩 Using SL strategy from config: {sl_strategy} (apply_mode={sl_apply_mode})")
    if sl_strategy == "supertrend":
        print("ℹ️ supertrend는 sl_params를 쓰지 않음 -> trials를 돌려도 결과는 동일해야 정상")

    sampler = optuna.samplers.TPESampler(seed=int(args.seed))

    if args.storage:
        study = optuna.create_study(
            study_name=str(args.study_name),
            direction="maximize",
            sampler=sampler,
            storage=str(args.storage),
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            study_name=str(args.study_name),
            direction="maximize",
            sampler=sampler,
        )

    def _obj(trial):
        return objective_sl_only(
            trial,
            base_cfg=base_cfg,
            raw_cache=raw_cache,
            universe=universe,
            sl_strategy=sl_strategy,
            sl_apply_mode=sl_apply_mode,
            base_sl_params=base_sl_params,
            days=int(args.days),
            mdd_penalty=float(args.mdd_penalty),
            min_trades=int(args.min_trades),
            min_pf=float(args.min_pf),
        )

    study.optimize(_obj, n_trials=int(args.trials), n_jobs=1)

    best = study.best_trial
    print("\n==================== BEST RESULT ====================")
    print("BEST SCORE:", float(best.value))
    print("BEST PARAMS:", json.dumps(best.params, indent=2, ensure_ascii=False))

    # best metrics 재확인 (동일 runner)
    best_sl_params = dict(best.params) if isinstance(best.params, dict) else {}
    r = run_backtest_sl_only(
        base_cfg=base_cfg,
        raw_cache=raw_cache,
        universe=universe,
        days=int(args.days),
        sl_strategy=sl_strategy,
        sl_apply_mode=sl_apply_mode,
        sl_params=best_sl_params,
    )
    print("\n==================== BEST METRICS ====================")
    print(json.dumps(r, indent=2, ensure_ascii=False))

    cfg_snip = {
        "system_settings": {
            "sl_apply_mode": sl_apply_mode,
            "sl_strategy": sl_strategy,
            "sl_params": best_sl_params,
        }
    }
    print("\n==================== CONFIG SNIPPET ====================")
    print(json.dumps(cfg_snip, indent=2, ensure_ascii=False))
    print(f"\n📄 CSV saved: {RESULT_FILE}")


if __name__ == "__main__":
    main()
