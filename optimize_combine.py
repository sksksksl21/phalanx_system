# optimize_entry_sl.py
# =========================================================
# Entry + SL Joint Optimizer (Optuna) - PKL 기반 (NO FETCH)
# - Optimizes:
#    (A) TitanStrategy entry params (cfg["strategy_settings"])
#    (B) SL params (cfg["system_settings"]["sl_params"])
# - Uses RAW cache built by down_pkl.py: market_data_cache_30d.pkl
# - Writes ALL trials (pass/fail) to CSV
# =========================================================

import os
import sys
import json
import copy
import argparse
import pickle
import math
import pandas as pd

# ---- PATH SETUP ----
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# ---- Import project modules ----
try:
    from core.backtest_engine import BacktestEngine
except Exception:
    try:
        from backtest_engine import BacktestEngine
    except Exception as e:
        raise ImportError(f"❌ Cannot import BacktestEngine: {e}")

try:
    from strategy.risk_control import RiskControl
except Exception as e:
    raise ImportError(f"❌ Cannot import RiskControl: {e}")

try:
    from strategy.titan_strategy import TitanStrategy
except Exception:
    TitanStrategy = None  # optional (engine already creates it)

RESULT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "joint_optimization_results.csv")


# =========================================================
# 1) Search Space
# =========================================================
def suggest_entry_params(trial):
    """
    Titan entry params 후보.
    ⚠️ 네 TitanStrategy가 실제로 읽는 키만 남기면 됨.
    (모르면 일단 이대로 두고, unused는 영향 없게 cfg에만 들어감)
    """
    p = {}

    # --- 기본 ---
    p["atr_period"] = trial.suggest_int("atr_period", 14, 30)
    p["atr_multiplier"] = trial.suggest_float("atr_multiplier", 2.0, 5.0, step=0.5)
    p["adx_threshold"] = trial.suggest_int("adx_threshold", 0, 30)
    p["rsi_upper"] = trial.suggest_int("rsi_upper", 60, 80)
    p["rsi_lower"] = trial.suggest_int("rsi_lower", 20, 40)
    p["vol_factor"] = trial.suggest_float("vol_factor", 0.8, 1.5, step=0.1)

    # --- 구조/컨텍스트 ---
    p["swing_len"] = trial.suggest_int("swing_len", 3, 12)
    p["context_lookback"] = trial.suggest_int("context_lookback", 60, 180, step=30)
    p["retest_tolerance_atr"] = trial.suggest_float("retest_tolerance_atr", 0.0, 1.0, step=0.05)

    return p


def suggest_armor_sl_params(trial):
    """
    ARMOR SL params (PositionMonitor._candidate_armor 기준)
    ✅ Optuna param name 충돌 방지 위해 전부 sl_ prefix 사용
    """
    p = {}
    p["sl_adx_period"] = trial.suggest_int("sl_adx_period", 10, 20)
    p["sl_adx_trend"] = trial.suggest_int("sl_adx_trend", 16, 32)
    p["sl_atr_period"] = trial.suggest_int("sl_atr_period", 10, 30)

    p["sl_atr_mult_trend"] = trial.suggest_float("sl_atr_mult_trend", 2.5, 6.0, step=0.25)
    p["sl_atr_mult_chop"] = trial.suggest_float("sl_atr_mult_chop", 1.0, 4.0, step=0.25)

    p["sl_swing_len"] = trial.suggest_int("sl_swing_len", 3, 12)
    p["sl_structure_buffer_atr"] = trial.suggest_float("sl_structure_buffer_atr", 0.0, 0.8, step=0.05)

    p["sl_profit_trigger_atr"] = trial.suggest_float("sl_profit_trigger_atr", 0.6, 2.5, step=0.1)
    p["sl_profit_lock_atr"] = trial.suggest_float("sl_profit_lock_atr", 0.0, 1.0, step=0.05)
    p["sl_fee_buffer_bps"] = trial.suggest_int("sl_fee_buffer_bps", 0, 20)

    p["sl_min_move_atr"] = trial.suggest_float("sl_min_move_atr", 0.0, 0.6, step=0.05)
    p["sl_max_step_atr"] = trial.suggest_float("sl_max_step_atr", 0.5, 4.0, step=0.25)

    p["sl_armor_lookback"] = trial.suggest_int("sl_armor_lookback", 150, 800, step=50)
    return p


# =========================================================
# 2) Cache / Universe
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
# 3) Runner (single trial)
# =========================================================
def run_backtest_joint(
    base_cfg: dict,
    raw_cache: dict,
    universe: list,
    days: int,
    sl_strategy: str,
    sl_apply_mode: str,
    entry_params: dict,
    sl_params: dict,
):
    """
    trial마다:
    - cfg 구성 (entry + sl)
    - engine 생성
    - cfg 주입 후 titan/risk_ctrl 재생성(권장)
    - raw_cache 주입
    - run
    - metrics 추출
    """
    cfg = copy.deepcopy(base_cfg) if isinstance(base_cfg, dict) else {}

    # ---- strategy_settings (ENTRY) ----
    cfg.setdefault("strategy_settings", {})
    if not isinstance(cfg["strategy_settings"], dict):
        cfg["strategy_settings"] = {}
    cfg["strategy_settings"].update(dict(entry_params or {}))

    # ---- system_settings (SL) ----
    cfg.setdefault("system_settings", {})
    if not isinstance(cfg["system_settings"], dict):
        cfg["system_settings"] = {}
    cfg["system_settings"]["sl_strategy"] = str(sl_strategy)
    cfg["system_settings"]["sl_apply_mode"] = str(sl_apply_mode)
    cfg["system_settings"]["sl_params"] = dict(sl_params or {})

    engine = BacktestEngine(days=int(days))

    # 1) cfg 덮기
    engine.cfg = cfg

    # 2) cfg 기반 컴포넌트 재생성(중요)
    engine.risk_ctrl = RiskControl(engine.executor, engine.cfg)

    # TitanStrategy가 cfg를 내부에서 참조/초기화한다면 재생성이 안전
    if TitanStrategy is not None:
        try:
            engine.titan = TitanStrategy()
            # blacklist는 BacktestEngine.__init__에서만 주입되므로, cfg에서 다시 반영
            bl = (engine.cfg.get("strategy_settings", {}) or {}).get("blacklist", [])
            if isinstance(bl, (list, tuple, set)):
                engine.titan.blacklist = set(bl)
        except Exception:
            pass

    # 3) RAW cache 주입 (NO FETCH)
    engine.symbols = list(universe)
    engine.raw_data_map = {sym: raw_cache[sym] for sym in universe}

    # run
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

    trades = int(len(pnls))
    wins = int(sum(1 for x in pnls if x > 0))
    winrate = (wins / trades) if trades > 0 else 0.0

    gross_win = float(sum(x for x in pnls if x > 0))
    gross_loss = float(-sum(x for x in pnls if x <= 0))
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
        "pf": float(pf),
        "winrate": float(winrate),
        "trades": int(trades),
    }


# =========================================================
# 4) Objective + CSV (ALL trials 기록)
# =========================================================
def objective_joint(
    trial,
    base_cfg,
    raw_cache,
    universe,
    days,
    sl_strategy,
    sl_apply_mode,
    mdd_penalty,
    min_trades,
    min_pf,
):
    def _sf(x, default=0.0):
        try:
            if x is None:
                return float(default)
            v = float(x)
            if not math.isfinite(v):
                return float(default)
            return v
        except Exception:
            return float(default)

    entry_params = suggest_entry_params(trial)

    if str(sl_strategy).lower() == "armor":
        sl_trial = suggest_armor_sl_params(trial)
        sl_params = {
            "atr_period": sl_trial["sl_atr_period"],
            "adx_period": sl_trial["sl_adx_period"],
            "adx_trend": sl_trial["sl_adx_trend"],
            "atr_mult_trend": sl_trial["sl_atr_mult_trend"],
            "atr_mult_chop": sl_trial["sl_atr_mult_chop"],
            "swing_len": sl_trial["sl_swing_len"],
            "structure_buffer_atr": sl_trial["sl_structure_buffer_atr"],
            "profit_trigger_atr": sl_trial["sl_profit_trigger_atr"],
            "profit_lock_atr": sl_trial["sl_profit_lock_atr"],
            "fee_buffer_bps": sl_trial["sl_fee_buffer_bps"],
            "min_move_atr": sl_trial["sl_min_move_atr"],
            "max_step_atr": sl_trial["sl_max_step_atr"],
            "armor_lookback": sl_trial["sl_armor_lookback"],
        }
    else:
        sl_params = {}

    r = run_backtest_joint(
        base_cfg=base_cfg,
        raw_cache=raw_cache,
        universe=universe,
        days=int(days),
        sl_strategy=sl_strategy,
        sl_apply_mode=sl_apply_mode,
        entry_params=entry_params,
        sl_params=sl_params,
    )

    final_equity = _sf(r.get("final_equity", 0.0), 0.0)
    mdd = _sf(r.get("mdd", 0.0), 0.0)
    trades = int(r.get("trades", 0) or 0)
    pf = _sf(r.get("pf", 0.0), 0.0)
    winrate = _sf(r.get("winrate", 0.0), 0.0)

    passed = True
    fail_reason = []
    if trades < int(min_trades):
        passed = False
        fail_reason.append("MIN_TRADES")
    if pf < float(min_pf):
        passed = False
        fail_reason.append("MIN_PF")

    score = final_equity - (float(mdd_penalty) * mdd * 10000.0)
    if not passed:
        score = -1e9

    rec = {
        "trial_id": int(trial.number),
        "sl_strategy": str(sl_strategy),
        "sl_apply_mode": str(sl_apply_mode),
        "score": _sf(score, -1e9),
        "passed": bool(passed),
        "fail_reason": "|".join(fail_reason),
        "final_equity": final_equity,
        "mdd": mdd,
        "pf": pf,
        "winrate": winrate,
        "trades": int(trades),
        **{f"en_{k}": v for k, v in (entry_params or {}).items()},
        **{f"sl_{k}": v for k, v in (sl_params or {}).items()},
    }

    df = pd.DataFrame([rec])
    if not os.path.exists(RESULT_FILE):
        df.to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(RESULT_FILE, index=False, mode="a", header=False, encoding="utf-8-sig")

    return float(_sf(score, -1e9))


def load_base_cfg_from_engine(days=30):
    e0 = BacktestEngine(days=int(days))
    return e0.cfg if isinstance(e0.cfg, dict) else {}


# =========================================================
# 5) Main
# =========================================================
def main():
    try:
        import optuna
    except Exception as e:
        raise ImportError(f"❌ optuna not installed: {e}")

    ap = argparse.ArgumentParser(description="Joint optimizer (Entry + SL) using RAW cache (NO FETCH)")
    ap.add_argument("--trials", type=int, default=200, help="number of optuna trials")
    ap.add_argument("--days", type=int, default=30, help="backtest days (must match cache)")
    ap.add_argument("--cache", type=str, default="market_data_cache_30d.pkl", help="pkl cache file from down_pkl.py")
    ap.add_argument("--min_rows", type=int, default=2800, help="min 15m rows for symbol filter")
    ap.add_argument("--universe_size", type=int, default=25, help="fixed universe size")
    ap.add_argument("--seed", type=int, default=42, help="sampler seed")
    ap.add_argument("--mdd_penalty", type=float, default=2.0, help="penalty multiplier for MDD")
    ap.add_argument("--min_trades", type=int, default=30, help="minimum EXIT trades")
    ap.add_argument("--min_pf", type=float, default=1.05, help="minimum profit factor filter")
    ap.add_argument("--sl_strategy", type=str, default="armor", help="armor|supertrend|atr_trail|profit_lock|hybrid")
    ap.add_argument("--sl_apply_mode", type=str, default="next", help="next|same")
    ap.add_argument("--study_name", type=str, default="JOINT_ENTRY_SL", help="optuna study name")
    ap.add_argument("--storage", type=str, default=None, help="optuna storage url, e.g. sqlite:///joint_opt.db")
    ap.add_argument("--universe_snapshot", type=str, default="universe_joint_opt.json", help="save fixed universe list")
    ap.add_argument("--reset_csv", action="store_true", help="delete result csv at start")
    args = ap.parse_args()

    if args.reset_csv and os.path.exists(RESULT_FILE):
        os.remove(RESULT_FILE)

    raw_cache = load_raw_cache(args.cache)
    universe = pick_fixed_universe(raw_cache, fixed_size=args.universe_size, min_rows=args.min_rows)

    # universe snapshot 저장
    try:
        with open(args.universe_snapshot, "w", encoding="utf-8") as f:
            json.dump([{"symbol": s} for s in universe], f, ensure_ascii=False, indent=2)
        print(f"🧭 Fixed Universe ({len(universe)}): saved to {args.universe_snapshot}")
    except Exception as e:
        print(f"⚠️ Failed to save universe snapshot: {e}")

    base_cfg = load_base_cfg_from_engine(days=args.days)

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
        return objective_joint(
            trial=trial,
            base_cfg=base_cfg,
            raw_cache=raw_cache,
            universe=universe,
            days=int(args.days),
            sl_strategy=str(args.sl_strategy),
            sl_apply_mode=str(args.sl_apply_mode),
            mdd_penalty=float(args.mdd_penalty),
            min_trades=int(args.min_trades),
            min_pf=float(args.min_pf),
        )

    study.optimize(_obj, n_trials=int(args.trials), n_jobs=1)

    best = study.best_trial
    print("\n==================== BEST RESULT ====================")
    print("BEST SCORE:", float(best.value))
    print("BEST PARAMS:", json.dumps(best.params, indent=2, ensure_ascii=False))
    print(f"\nCSV SAVED: {RESULT_FILE}")

    # config snippet (best params 그대로)
    # entry/sl 파라미터 분리해서 보여줌
    best_params = dict(best.params)
    en = {k: v for k, v in best_params.items() if not k.startswith("sl_")}

    # armor일 때만 sl_params가 의미 있음 (이 스크립트에선 trial param 네이밍이 겹치므로 그냥 전부 출력)
    cfg_snip = {
        "strategy_settings": en,
        "system_settings": {
            "sl_apply_mode": str(args.sl_apply_mode),
            "sl_strategy": str(args.sl_strategy),
        },
    }
    print("\n==================== CONFIG SNIPPET (RAW) ====================")
    print(json.dumps(cfg_snip, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
