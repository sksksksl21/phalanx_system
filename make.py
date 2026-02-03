from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path("universe_filter")

UTC_NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _backup_if_needed(path: Path, new_text: str) -> None:
    if not path.exists():
        return
    old = path.read_text(encoding="utf-8", errors="replace")
    if _sha256(old) == _sha256(new_text):
        return
    bak = path.with_suffix(path.suffix + ".bak")
    bak.write_text(old, encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _backup_if_needed(path, text)
    path.write_text(text, encoding="utf-8")


def _json_dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"


def main() -> None:
    # --- dirs ---
    for d in [
        ROOT / "input" / "ohlcv",
        ROOT / "output" / "cache",
        ROOT / "config",
        ROOT / "steps",
        ROOT / "lib",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # --- config defaults (can edit later) ---
    cfg_path = ROOT / "config" / "uf_config.json"
    if not cfg_path.exists():
        _write(
            cfg_path,
            _json_dump(
                {
                    "asof_timezone": "UTC",
                    "rebalance_rule": "WEEKLY",
                    "windows_days": [7, 30, 90],
                    "train_lookback_days": 90,
                    "horizon_days": 7,
                    "n_select": 25,
                    "min_history_days": 90,
                    "exclude_if_trade_count_7d_zero": True,
                    "feature_caps": {
                        "max_features_total": 20,
                        "max_outcome_features": 3,
                        "corr_cutoff": 0.80,
                        "importance_prune_frac": 0.20,
                        "importance_prune_loops": 3
                    },
                    "score": {
                        "lambda_scale_top30": 1.2,
                        "lambda_scale_mid": 1.0,
                        "lambda_scale_bottom30": 0.8
                    },
                    "model": {
                        "name": "RandomForestClassifier",
                        "n_estimators": 400,
                        "max_depth": 5,
                        "min_samples_leaf": 10,
                        "random_state": 42
                    },
                    "major_force_include": [],
                    "io": {
                        "input_csv": "universe_filter/input/backtest_history.csv",
                        "ohlcv_dir": "universe_filter/input/ohlcv",
                        "trades_parquet": "universe_filter/output/cache/trades.parquet",
                        "symbol_window_metrics_parquet": "universe_filter/output/cache/symbol_window_metrics.parquet",
                        "uf_dataset_parquet": "universe_filter/output/cache/uf_dataset.parquet",
                        "model_pkl": "universe_filter/output/cache/model.pkl",
                        "selected_symbols_json": "universe_filter/output/cache/selected_symbols.json",
                        "decision_table_csv": "universe_filter/output/decision_table.csv",
                        "run_report_json": "universe_filter/output/run_report.json",
                        "universe_json": "universe_filter/output/universe.json"
                    }
                }
            ),
        )

    # --- placeholders for outputs (kept valid JSON) ---
    for p, comment in [
        (ROOT / "output" / "run_report.json", "Run report (steps append here)."),
        (ROOT / "output" / "universe.json", "Universe output (written ONLY by Step05)."),
    ]:
        if not p.exists():
            _write(p, _json_dump({"_comment": comment}))
    if not (ROOT / "output" / "decision_table.csv").exists():
        _write(ROOT / "output" / "decision_table.csv", "")

    # --- write code files ---
    _write(ROOT / "README.md", README_MD)
    _write(ROOT / "run_pipeline.py", RUN_PIPELINE_PY)
    for name, text in LIB_FILES.items():
        _write(ROOT / "lib" / name, text)
    for name, text in STEP_FILES.items():
        _write(ROOT / "steps" / name, text)

    print("✅ Generated full UF code under universe_filter/")
    print("   - Existing files backed up as *.bak (if changed)")


README_MD = """# Universe Filter (UF) v1.0

_Generated: {UTC_NOW}_

## Hard Rules
- Strategy logic (entry/positioning/exit) MUST NOT be modified.
- UF outputs ONLY `output/universe.json` for engine consumption.
- All files live under `universe_filter/`.
- Pipeline is step-by-step: `steps/step_00` ~ `steps/step_05`.
- Only Step 05 writes `output/universe.json`.

## Entry point
- `python universe_filter/run_pipeline.py`

## Inputs
- Required: `input/backtest_history.csv`
- Optional: `input/ohlcv/*` (if missing, OHLCV features are skipped safely)

## Outputs
- Required: `output/universe.json` (Step05 only)
- Recommended: `output/decision_table.csv`, `output/run_report.json`
"""

RUN_PIPELINE_PY = r'''# =========================================================
# Universe Filter - Pipeline Runner (ENTRYPOINT)
# ---------------------------------------------------------
# Generated: {UTC_NOW}
#
# HARD RULES (v1.0):
# - Strategy logic (entry/positioning/exit) MUST NOT be modified.
# - UF MUST output ONLY universe.json for engine consumption.
# - All files MUST live under universe_filter/.
# - Pipeline is step-by-step; each step is a single runnable module.
# - Only Step 05 writes output/universe.json.
# =========================================================

from __future__ import annotations

import sys
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print(f"▶ {' '.join(cmd)}")
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main() -> None:
    root = Path(__file__).resolve().parent
    steps = [
        root / "steps" / "step_00_ingest.py",
        root / "steps" / "step_01_persistence.py",
        root / "steps" / "step_02_build_dataset.py",
        root / "steps" / "step_03_train_validate.py",
        root / "steps" / "step_04_select_universe.py",
        root / "steps" / "step_05_export_universe.py",
    ]
    for s in steps:
        _run([sys.executable, str(s)])
    print("✅ Pipeline completed.")


if __name__ == "__main__":
    main()
'''

LIB_FILES = {
"schema.py": r'''# =========================================================
# Universe Filter - lib/schema.py
# ---------------------------------------------------------
# Generated: {UTC_NOW}
# =========================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import List

REQUIRED_BACKTEST_HISTORY_COLS: List[str] = [
    "Datetime", "Symbol", "Side", "Type", "Price", "Amount", "PnL", "Cash", "Equity", "Reason"
]

REQUIRED_TYPES = {"ENTRY", "EXIT"}


@dataclass(frozen=True)
class UFPaths:
    input_csv: str
    ohlcv_dir: str
    trades_parquet: str
    symbol_window_metrics_parquet: str
    uf_dataset_parquet: str
    model_pkl: str
    selected_symbols_json: str
    decision_table_csv: str
    run_report_json: str
    universe_json: str
''',

"io.py": r'''# =========================================================
# Universe Filter - lib/io.py
# ---------------------------------------------------------
# Generated: {UTC_NOW}
# =========================================================
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def atomic_write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    ensure_parent(p)
    fd, tmp = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, p)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_csv(path: str | Path, df: pd.DataFrame) -> None:
    ensure_parent(path)
    df.to_csv(path, index=False)


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_parquet(path: str | Path, df: pd.DataFrame) -> None:
    ensure_parent(path)
    df.to_parquet(path, index=False)
''',

"trades.py": r'''# =========================================================
# Universe Filter - lib/trades.py
# ---------------------------------------------------------
# Generated: {UTC_NOW}
# =========================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class TradeMatchReport:
    n_rows: int
    n_entry: int
    n_exit: int
    n_trades: int
    missing_exit: int
    orphan_exit: int


def normalize_backtest_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    # numeric
    for c in ["Price", "Amount", "PnL", "Cash", "Equity"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Type"] = df["Type"].astype(str).str.upper()
    df["Side"] = df["Side"].astype(str).str.upper()
    df["Symbol"] = df["Symbol"].astype(str)

    return df


def match_trades_fifo(df: pd.DataFrame) -> Tuple[pd.DataFrame, TradeMatchReport]:
    """
    FIFO matching per (symbol, side):
      ENTRY opens a trade, EXIT closes the oldest open trade.

    Output trade ledger schema (one row per completed trade):
      trade_id, symbol, side, entry_ts, exit_ts, entry_price, exit_price, amount,
      pnl, entry_equity, exit_equity, holding_minutes
    """
    open_q: Dict[Tuple[str, str], List[dict]] = {}

    trades: List[dict] = []
    missing_exit = 0
    orphan_exit = 0

    n_entry = int((df["Type"] == "ENTRY").sum())
    n_exit = int((df["Type"] == "EXIT").sum())

    tid = 0
    for _, r in df.iterrows():
        key = (r["Symbol"], r["Side"])
        t = r["Type"]

        if t == "ENTRY":
            open_q.setdefault(key, []).append(
                {
                    "entry_ts": r["Datetime"],
                    "entry_price": r["Price"],
                    "amount": r["Amount"],
                    "entry_equity": r["Equity"],
                    "reason_entry": r.get("Reason", None),
                }
            )
        elif t == "EXIT":
            q = open_q.get(key, [])
            if not q:
                orphan_exit += 1
                continue
            o = q.pop(0)
            tid += 1
            entry_ts = o["entry_ts"]
            exit_ts = r["Datetime"]
            holding_minutes = (exit_ts - entry_ts).total_seconds() / 60.0
            trades.append(
                {
                    "trade_id": tid,
                    "symbol": key[0],
                    "side": key[1],
                    "entry_ts": entry_ts,
                    "exit_ts": exit_ts,
                    "entry_price": o["entry_price"],
                    "exit_price": r["Price"],
                    "amount": o["amount"],
                    "pnl": r["PnL"],
                    "entry_equity": o["entry_equity"],
                    "exit_equity": r["Equity"],
                    "holding_minutes": holding_minutes,
                    "reason_entry": o.get("reason_entry", None),
                    "reason_exit": r.get("Reason", None),
                }
            )
        else:
            # ignore other rows
            pass

    # remaining open trades => missing exit
    for q in open_q.values():
        missing_exit += len(q)

    out = pd.DataFrame(trades)
    if not out.empty:
        out["entry_ts"] = pd.to_datetime(out["entry_ts"], utc=True)
        out["exit_ts"] = pd.to_datetime(out["exit_ts"], utc=True)

    rep = TradeMatchReport(
        n_rows=len(df),
        n_entry=n_entry,
        n_exit=n_exit,
        n_trades=len(out),
        missing_exit=missing_exit,
        orphan_exit=orphan_exit,
    )
    return out, rep
''',

"metrics.py": r'''# =========================================================
# Universe Filter - lib/metrics.py
# ---------------------------------------------------------
# Generated: {UTC_NOW}
# =========================================================
from __future__ import annotations

import numpy as np
import pandas as pd


def equity_mdd(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    x = equity.astype(float).values
    peak = np.maximum.accumulate(x)
    dd = (x - peak) / np.where(peak == 0, 1.0, peak)
    return float(dd.min())  # negative


def profit_factor(pnl: pd.Series) -> float:
    wins = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses <= 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def expectancy(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    return float(pnl.mean())


def holding_iqr(x: pd.Series) -> float:
    if x.empty:
        return 0.0
    q75, q25 = np.percentile(x.astype(float).values, [75, 25])
    return float(q75 - q25)


def trade_count_std(trades: pd.DataFrame, freq: str = "D") -> float:
    if trades.empty:
        return 0.0
    # use exit_ts as realization time
    s = trades.set_index("exit_ts").groupby(pd.Grouper(freq=freq))["trade_id"].count()
    if len(s) <= 1:
        return 0.0
    return float(s.std(ddof=0))


def compute_symbol_window_metrics(trades: pd.DataFrame, asof: pd.Timestamp, window_days: int) -> pd.DataFrame:
    """
    trades: completed trades ledger with exit_ts, pnl, holding_minutes, exit_equity
    returns: per-symbol metrics for (asof, window_days)
    """
    start = asof - pd.Timedelta(days=window_days)
    w = trades[(trades["exit_ts"] > start) & (trades["exit_ts"] <= asof)].copy()
    if w.empty:
        return pd.DataFrame(columns=[
            "symbol","window_days","net_profit","trade_count","winrate",
            "avg_win","avg_loss","profit_factor","expectancy",
            "equity_mdd","holding_median","holding_iqr","trade_count_std_daily"
        ])

    rows = []
    for sym, g in w.groupby("symbol"):
        pnl = g["pnl"].astype(float)
        net = float(pnl.sum())
        tc = int(len(g))
        winrate = float((pnl > 0).mean()) if tc > 0 else 0.0
        avg_win = float(pnl[pnl > 0].mean()) if (pnl > 0).any() else 0.0
        avg_loss = float(pnl[pnl < 0].mean()) if (pnl < 0).any() else 0.0
        pf = profit_factor(pnl)
        exp = expectancy(pnl)

        # equity MDD uses exit_equity series within window per symbol (proxy)
        eq = g.sort_values("exit_ts")["exit_equity"].astype(float)
        mdd = equity_mdd(eq)

        hold = g["holding_minutes"].astype(float)
        hold_med = float(np.median(hold.values)) if len(hold) else 0.0
        hold_iqr = holding_iqr(hold)

        tc_std = trade_count_std(g, freq="D")

        rows.append({
            "symbol": sym,
            "window_days": int(window_days),
            "net_profit": net,
            "trade_count": tc,
            "winrate": winrate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": pf if np.isfinite(pf) else 9999.0,
            "expectancy": exp,
            "equity_mdd": mdd,  # negative or 0
            "holding_median": hold_med,
            "holding_iqr": hold_iqr,
            "trade_count_std_daily": tc_std,
        })
    return pd.DataFrame(rows)
''',

"features.py": r'''# =========================================================
# Universe Filter - lib/features.py
# ---------------------------------------------------------
# Generated: {UTC_NOW}
# =========================================================
from __future__ import annotations

import numpy as np
import pandas as pd


OUTCOME_FEATURES = {
    "expectancy_90d",
    "return_over_mdd_90d",
    "trade_count_stability_7d",
}


def corr_cut(df: pd.DataFrame, cutoff: float) -> list[str]:
    if df.shape[1] <= 1:
        return list(df.columns)
    c = df.corr().abs()
    keep = []
    for col in c.columns:
        ok = True
        for k in keep:
            if c.loc[col, k] > cutoff:
                ok = False
                break
        if ok:
            keep.append(col)
    return keep


def build_feature_frame(panel: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    panel columns include:
      symbol, window_days + metric cols
    Returns wide feature frame: one row per symbol with suffixes per window.
    """
    feats = []
    for w in windows:
        p = panel[panel["window_days"] == w].copy()
        if p.empty:
            continue
        p = p.set_index("symbol")
        cols = [c for c in p.columns if c not in ["window_days"]]
        p = p[cols]
        p = p.add_suffix(f"_{w}d")
        feats.append(p)
    if not feats:
        return pd.DataFrame()
    out = pd.concat(feats, axis=1).reset_index().rename(columns={"index": "symbol"})
    return out


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Example derived:
    if "net_profit_90d" in df.columns and "equity_mdd_90d" in df.columns:
        mdd = df["equity_mdd_90d"].abs() + 1e-9
        df["return_over_mdd_90d"] = df["net_profit_90d"] / mdd

    if "trade_count_std_daily_7d" in df.columns:
        df["trade_count_stability_7d"] = 1.0 / (1.0 + df["trade_count_std_daily_7d"].clip(lower=0.0))

    # rr ratio
    if "avg_win_90d" in df.columns and "avg_loss_90d" in df.columns:
        denom = df["avg_loss_90d"].abs().replace(0, np.nan)
        df["rr_ratio_90d"] = (df["avg_win_90d"] / denom).fillna(0.0)

    return df


def enforce_feature_policy(df: pd.DataFrame, max_total: int, max_outcome: int, corr_cutoff: float) -> pd.DataFrame:
    """
    - Corr cut
    - Outcome feature cap
    - Total feature cap
    Keeps 'symbol' and 'asof' always.
    """
    df = df.copy()
    id_cols = [c for c in ["symbol", "asof"] if c in df.columns]
    feat_cols = [c for c in df.columns if c not in id_cols]

    # corr-cut requires numeric only
    num = df[feat_cols].select_dtypes(include=[np.number]).copy()
    keep_num = corr_cut(num, corr_cutoff)

    # drop non-numeric features for v1 baseline (keeps stable)
    keep = keep_num

    # outcome cap
    outcome = [c for c in keep if c in OUTCOME_FEATURES]
    non_outcome = [c for c in keep if c not in OUTCOME_FEATURES]
    if len(outcome) > max_outcome:
        outcome = outcome[:max_outcome]
    keep = non_outcome + outcome

    # total cap
    if len(keep) > max_total:
        keep = keep[:max_total]

    return df[id_cols + keep]
''',

"labeling.py": r'''# =========================================================
# Universe Filter - lib/labeling.py
# ---------------------------------------------------------
# Generated: {UTC_NOW}
# =========================================================
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_lambda(mdd_series: pd.Series, cfg_score: dict) -> float:
    """
    λ = base * scale, where:
      base = median(MDD_all_symbols_in_window)
      scale = 1.2 (top30 volatility) / 1.0 (mid) / 0.8 (bottom30)
    Here we approximate regime by mdd distribution itself.
    """
    if mdd_series.empty:
        return 1e-6
    base = float(np.median(mdd_series.abs().values))
    q70 = float(np.quantile(mdd_series.abs().values, 0.70))
    q30 = float(np.quantile(mdd_series.abs().values, 0.30))
    cur = float(np.median(mdd_series.abs().values))
    if cur >= q70:
        scale = float(cfg_score.get("lambda_scale_top30", 1.2))
    elif cur <= q30:
        scale = float(cfg_score.get("lambda_scale_bottom30", 0.8))
    else:
        scale = float(cfg_score.get("lambda_scale_mid", 1.0))
    lam = base * scale
    return lam if lam > 1e-9 else 1e-9


def score_from_profit_mdd(net_profit: pd.Series, mdd: pd.Series, lam: float) -> pd.Series:
    denom = (mdd.abs() + lam)
    return net_profit / denom.replace(0, 1e-9)


def make_labels(score: pd.Series) -> pd.Series:
    """
    top 30% => 1
    bottom 50% => 0
    middle 20% => NaN (excluded)
    """
    if score.empty:
        return score.copy()
    q70 = score.quantile(0.70)
    q50 = score.quantile(0.50)
    y = pd.Series(np.nan, index=score.index)
    y[score >= q70] = 1.0
    y[score <= q50] = 0.0
    return y
''',

"model.py": r'''# =========================================================
# Universe Filter - lib/model.py
# ---------------------------------------------------------
# Generated: {UTC_NOW}
# =========================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
except Exception as e:  # pragma: no cover
    RandomForestClassifier = None
    permutation_importance = None
    _SKLEARN_ERR = e
else:
    _SKLEARN_ERR = None


@dataclass
class TrainedModel:
    model: object
    feature_cols: List[str]


def require_sklearn() -> None:
    if RandomForestClassifier is None or permutation_importance is None:
        raise ImportError(f"scikit-learn is required for Step03. Import error: {_SKLEARN_ERR}")


def train_rf_with_pruning(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    cfg_model: dict,
    prune_frac: float,
    prune_loops: int,
    random_state: int = 42,
) -> TrainedModel:
    require_sklearn()

    X = df[feature_cols].astype(float).values
    y = df[label_col].astype(int).values

    cols = feature_cols[:]
    model = RandomForestClassifier(
        n_estimators=int(cfg_model.get("n_estimators", 400)),
        max_depth=int(cfg_model.get("max_depth", 5)),
        min_samples_leaf=int(cfg_model.get("min_samples_leaf", 10)),
        random_state=int(cfg_model.get("random_state", random_state)),
        n_jobs=-1,
    )

    for _ in range(max(1, int(prune_loops))):
        model.fit(X, y)
        r = permutation_importance(model, X, y, n_repeats=5, random_state=random_state, n_jobs=-1)
        imp = pd.Series(r.importances_mean, index=cols).sort_values(ascending=False)
        n_drop = max(1, int(len(cols) * prune_frac))
        drop = list(imp.tail(n_drop).index)

        # keep at least 3 features
        if len(cols) - len(drop) < 3:
            break

        cols = [c for c in cols if c not in drop]
        X = df[cols].astype(float).values

    model.fit(X, y)
    return TrainedModel(model=model, feature_cols=cols)


def predict_proba(model: TrainedModel, df: pd.DataFrame) -> pd.Series:
    X = df[model.feature_cols].astype(float).values
    p = model.model.predict_proba(X)[:, 1]
    return pd.Series(p, index=df.index)
''',

"validation.py": r'''# =========================================================
# Universe Filter - lib/validation.py
# ---------------------------------------------------------
# Generated: {UTC_NOW}
# =========================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class WFWindow:
    train_end: pd.Timestamp
    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def weekly_asof_points(trades: pd.DataFrame, horizon_days: int) -> List[pd.Timestamp]:
    if trades.empty:
        return []
    t0 = trades["exit_ts"].min().floor("D")
    t1 = trades["exit_ts"].max().floor("D")
    # weekly points
    asofs = pd.date_range(start=t0, end=t1 - pd.Timedelta(days=horizon_days), freq="W-MON", tz="UTC")
    return list(asofs)


def build_walkforward_windows(trades: pd.DataFrame, lookback_days: int, horizon_days: int) -> List[WFWindow]:
    asofs = weekly_asof_points(trades, horizon_days=horizon_days)
    out: List[WFWindow] = []
    for t in asofs:
        train_end = t
        train_start = t - pd.Timedelta(days=lookback_days)
        test_start = t
        test_end = t + pd.Timedelta(days=horizon_days)
        out.append(WFWindow(train_end=train_end, train_start=train_start, test_start=test_start, test_end=test_end))
    return out


def slice_trades(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return trades[(trades["exit_ts"] > start) & (trades["exit_ts"] <= end)].copy()


def objective_vector_from_trades(trades: pd.DataFrame) -> Dict[str, float]:
    """
    v1 baseline objectives computed from realized trades in period:
      - mdd_proxy: min of cumulative pnl curve (as fraction of abs peak) on exit_equity proxy is not stable,
        so we use cumulative pnl drawdown on trades as proxy.
      - trade_count_std_daily: std of daily trade counts
      - holding_iqr: IQR of holding_minutes
    """
    if trades.empty:
        return {"mdd_proxy": 0.0, "trade_count_std_daily": 0.0, "holding_iqr": 0.0}

    g = trades.sort_values("exit_ts").copy()
    pnl = g["pnl"].astype(float).values
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    mdd_proxy = float(dd.min())  # negative

    daily = g.set_index("exit_ts").groupby(pd.Grouper(freq="D"))["trade_id"].count()
    tc_std = float(daily.std(ddof=0)) if len(daily) > 1 else 0.0

    hold = g["holding_minutes"].astype(float).values
    q75, q25 = np.percentile(hold, [75, 25])
    iqr = float(q75 - q25)

    return {"mdd_proxy": mdd_proxy, "trade_count_std_daily": tc_std, "holding_iqr": iqr}
''',

"selection.py": r'''# =========================================================
# Universe Filter - lib/selection.py
# ---------------------------------------------------------
# Generated: {UTC_NOW}
# =========================================================
from __future__ import annotations

from typing import List, Tuple

import pandas as pd


def apply_safety_gates(df_latest: pd.DataFrame, min_history_days: int, exclude_trade_count_7d_zero: bool) -> pd.DataFrame:
    df = df_latest.copy()
    if "history_days" in df.columns:
        df = df[df["history_days"] >= min_history_days]
    if exclude_trade_count_7d_zero and "trade_count_7d" in df.columns:
        df = df[df["trade_count_7d"] > 0]
    return df


def select_top_n(df_latest: pd.DataFrame, score_col: str, n: int, majors: List[str] | None = None) -> Tuple[List[str], pd.DataFrame]:
    majors = majors or []
    df = df_latest.copy()
    df = df.sort_values(score_col, ascending=False)
    selected = []
    # force majors first if present in df
    for m in majors:
        if m in set(df["symbol"].tolist()) and m not in selected:
            selected.append(m)
    for sym in df["symbol"].tolist():
        if sym not in selected:
            selected.append(sym)
        if len(selected) >= n:
            break
    return selected, df
''',

"reporting.py": r'''# =========================================================
# Universe Filter - lib/reporting.py
# ---------------------------------------------------------
# Generated: {UTC_NOW}
# =========================================================
from __future__ import annotations

from typing import Any, Dict

from .io import read_json, write_json


def update_run_report(path: str, patch: Dict[str, Any]) -> None:
    rep = read_json(path)
    rep = rep if isinstance(rep, dict) else {}
    rep.update(patch)
    write_json(path, rep)
''',
}

STEP_FILES = {
"step_00_ingest.py": r'''# =========================================================
# Step 00 - Ingest & Normalize
# ---------------------------------------------------------
# Generated: {UTC_NOW}
# =========================================================
from __future__ import annotations

import argparse
import pandas as pd

from universe_filter.lib.io import read_csv, write_parquet
from universe_filter.lib.schema import REQUIRED_BACKTEST_HISTORY_COLS
from universe_filter.lib.trades import normalize_backtest_history, match_trades_fifo
from universe_filter.lib.reporting import update_run_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="universe_filter/input/backtest_history.csv")
    ap.add_argument("--out", default="universe_filter/output/cache/trades.parquet")
    ap.add_argument("--report", default="universe_filter/output/run_report.json")
    args = ap.parse_args()

    df = read_csv(args.input)
    missing = [c for c in REQUIRED_BACKTEST_HISTORY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"backtest_history.csv missing columns: {missing}")

    df = normalize_backtest_history(df)
    trades, rep = match_trades_fifo(df)

    write_parquet(args.out, trades)
    update_run_report(args.report, {"step_00": rep.__dict__})

    print(f"✅ Step00 done. trades={len(trades)} missing_exit={rep.missing_exit} orphan_exit={rep.orphan_exit}")


if __name__ == "__main__":
    main()
''',

"step_01_persistence.py": r'''# =========================================================
# Step 01 - Persistence Check (EDA Gate)
# ---------------------------------------------------------
# Generated: {UTC_NOW}
# =========================================================
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from universe_filter.lib.io import read_parquet, write_csv
from universe_filter.lib.reporting import update_run_report
from universe_filter.lib.validation import build_walkforward_windows, slice_trades


def _sym_period_score(trades: pd.DataFrame) -> pd.Series:
    # simple: sum pnl per symbol
    if trades.empty:
        return pd.Series(dtype=float)
    return trades.groupby("symbol")["pnl"].sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default="universe_filter/output/cache/trades.parquet")
    ap.add_argument("--report", default="universe_filter/output/run_report.json")
    ap.add_argument("--out_csv", default="universe_filter/output/persistence_table.csv")
    ap.add_argument("--lookback_days", type=int, default=90)
    ap.add_argument("--horizon_days", type=int, default=7)
    args = ap.parse_args()

    trades = read_parquet(args.trades)
    if trades.empty:
        update_run_report(args.report, {"step_01": {"persistence": None, "note": "no trades"}})
        print("⚠️ Step01 skipped (no trades).")
        return

    trades["exit_ts"] = pd.to_datetime(trades["exit_ts"], utc=True)

    windows = build_walkforward_windows(trades, lookback_days=args.lookback_days, horizon_days=args.horizon_days)
    rows = []
    for w in windows:
        tr = slice_trades(trades, w.train_start, w.train_end)
        te = slice_trades(trades, w.test_start, w.test_end)
        s_tr = _sym_period_score(tr)
        s_te = _sym_period_score(te)
        common = s_tr.index.intersection(s_te.index)
        if len(common) < 5:
            continue
        a = s_tr.loc[common].rank()
        b = s_te.loc[common].rank()
        spearman = float(a.corr(b, method="pearson"))
        pearson = float(s_tr.loc[common].corr(s_te.loc[common], method="pearson"))
        rows.append({"asof": w.train_end.isoformat(), "n": int(len(common)), "spearman_rank": spearman, "pearson": pearson})

    out = pd.DataFrame(rows)
    if not out.empty:
        write_csv(args.out_csv, out)

        # policy hint
        med_s = float(out["spearman_rank"].median())
        hint = "tighten_outcome_features" if med_s < 0.05 else "normal"
        update_run_report(args.report, {"step_01": {"median_spearman_rank": med_s, "policy_hint": hint}})
        print(f"✅ Step01 done. median_spearman_rank={med_s:.4f} hint={hint}")
    else:
        update_run_report(args.report, {"step_01": {"median_spearman_rank": None, "policy_hint": "unknown"}})
        print("⚠️ Step01 produced no valid windows.")


if __name__ == "__main__":
    main()
''',

"step_02_build_dataset.py": r'''# =========================================================
# Step 02 - Build Symbol×Window Metrics & UF Dataset
# ---------------------------------------------------------
# Generated: {UTC_NOW}
# =========================================================
from __future__ import annotations

import argparse
import pandas as pd

from universe_filter.lib.io import read_json, read_parquet, write_parquet
from universe_filter.lib.metrics import compute_symbol_window_metrics
from universe_filter.lib.features import build_feature_frame, add_derived_features, enforce_feature_policy
from universe_filter.lib.reporting import update_run_report


def _asof_points(trades: pd.DataFrame) -> list[pd.Timestamp]:
    if trades.empty:
        return []
    t0 = trades["exit_ts"].min().floor("D")
    t1 = trades["exit_ts"].max().floor("D")
    return list(pd.date_range(t0, t1, freq="W-MON", tz="UTC"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="universe_filter/config/uf_config.json")
    ap.add_argument("--trades", default="universe_filter/output/cache/trades.parquet")
    ap.add_argument("--out_panel", default="universe_filter/output/cache/symbol_window_metrics.parquet")
    ap.add_argument("--out_ds", default="universe_filter/output/cache/uf_dataset.parquet")
    ap.add_argument("--report", default="universe_filter/output/run_report.json")
    args = ap.parse_args()

    cfg = read_json(args.config)
    windows = list(cfg.get("windows_days", [7, 30, 90]))
    caps = cfg.get("feature_caps", {})
    max_total = int(caps.get("max_features_total", 20))
    max_outcome = int(caps.get("max_outcome_features", 3))
    corr_cutoff = float(caps.get("corr_cutoff", 0.80))

    trades = read_parquet(args.trades)
    if trades.empty:
        write_parquet(args.out_panel, trades)
        write_parquet(args.out_ds, trades)
        update_run_report(args.report, {"step_02": {"note": "no trades"}})
        print("⚠️ Step02 skipped (no trades).")
        return

    trades["exit_ts"] = pd.to_datetime(trades["exit_ts"], utc=True)
    asofs = _asof_points(trades)

    panel_rows = []
    ds_rows = []
    for asof in asofs:
        # panel for each window
        panel_list = []
        for w in windows:
            m = compute_symbol_window_metrics(trades, asof=asof, window_days=int(w))
            if not m.empty:
                panel_list.append(m)
        if not panel_list:
            continue
        panel = pd.concat(panel_list, ignore_index=True)
        panel["asof"] = asof
        panel_rows.append(panel)

        feats = build_feature_frame(panel.drop(columns=["asof"]), windows=windows)
        if feats.empty:
            continue
        feats["asof"] = asof
        feats = add_derived_features(feats)

        # history_days proxy (using earliest exit in symbol)
        first_exit = trades.groupby("symbol")["exit_ts"].min()
        feats["history_days"] = (asof - feats["symbol"].map(first_exit)).dt.total_seconds() / 86400.0

        feats = enforce_feature_policy(feats, max_total=max_total, max_outcome=max_outcome, corr_cutoff=corr_cutoff)
        ds_rows.append(feats)

    panel_all = pd.concat(panel_rows, ignore_index=True) if panel_rows else pd.DataFrame()
    ds_all = pd.concat(ds_rows, ignore_index=True) if ds_rows else pd.DataFrame()

    write_parquet(args.out_panel, panel_all)
    write_parquet(args.out_ds, ds_all)

    update_run_report(args.report, {"step_02": {"asof_points": len(asofs), "rows_dataset": int(len(ds_all))}})
    print(f"✅ Step02 done. uf_dataset_rows={len(ds_all)} panel_rows={len(panel_all)}")


if __name__ == "__main__":
    main()
''',

"step_03_train_validate.py": r'''# =========================================================
# Step 03 - Train & Walk-forward Validate
# ---------------------------------------------------------
# Generated: {UTC_NOW}
# =========================================================
from __future__ import annotations

import argparse
import pickle
import pandas as pd
import numpy as np

from universe_filter.lib.io import read_json, read_parquet, write_csv, ensure_parent
from universe_filter.lib.labeling import compute_lambda, score_from_profit_mdd, make_labels
from universe_filter.lib.model import train_rf_with_pruning, predict_proba
from universe_filter.lib.reporting import update_run_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="universe_filter/config/uf_config.json")
    ap.add_argument("--ds", default="universe_filter/output/cache/uf_dataset.parquet")
    ap.add_argument("--panel", default="universe_filter/output/cache/symbol_window_metrics.parquet")
    ap.add_argument("--model_out", default="universe_filter/output/cache/model.pkl")
    ap.add_argument("--decision_out", default="universe_filter/output/decision_table.csv")
    ap.add_argument("--report", default="universe_filter/output/run_report.json")
    args = ap.parse_args()

    cfg = read_json(args.config)
    caps = cfg.get("feature_caps", {})
    prune_frac = float(caps.get("importance_prune_frac", 0.20))
    prune_loops = int(caps.get("importance_prune_loops", 3))
    cfg_model = cfg.get("model", {})
    score_cfg = cfg.get("score", {})

    ds = read_parquet(args.ds)
    panel = read_parquet(args.panel)
    if ds.empty or panel.empty:
        update_run_report(args.report, {"step_03": {"note": "missing dataset/panel"}})
        print("⚠️ Step03 skipped (dataset/panel empty).")
        return

    # Build training label at each asof using 90d metrics if available
    panel90 = panel[panel["window_days"] == 90].copy()
    if panel90.empty:
        update_run_report(args.report, {"step_03": {"note": "no 90d panel"}})
        print("⚠️ Step03 skipped (no 90d metrics).")
        return

    # join ds with 90d metrics to compute score/label per (symbol, asof)
    panel90["asof"] = pd.to_datetime(panel90["asof"], utc=True)
    ds["asof"] = pd.to_datetime(ds["asof"], utc=True)

    j = ds.merge(panel90[["symbol","asof","net_profit","equity_mdd","trade_count"]], on=["symbol","asof"], how="left")
    j = j.dropna(subset=["net_profit","equity_mdd"])

    lam = compute_lambda(j["equity_mdd"], score_cfg)
    j["uf_score"] = score_from_profit_mdd(j["net_profit"], j["equity_mdd"], lam)
    j["uf_label"] = make_labels(j["uf_score"])

    train = j.dropna(subset=["uf_label"]).copy()
    if len(train) < 30:
        update_run_report(args.report, {"step_03": {"note": "not enough labeled rows", "labeled": int(len(train))}})
        print(f"⚠️ Step03 skipped (not enough labeled rows: {len(train)}).")
        return

    feature_cols = [c for c in ds.columns if c not in ["symbol","asof","history_days"]]
    # keep numeric only
    feature_cols = [c for c in feature_cols if np.issubdtype(train[c].dtype, np.number)]

    model = train_rf_with_pruning(
        train,
        feature_cols=feature_cols,
        label_col="uf_label",
        cfg_model=cfg_model,
        prune_frac=prune_frac,
        prune_loops=prune_loops,
        random_state=int(cfg_model.get("random_state", 42))
    )

    # save model
    ensure_parent(args.model_out)
    with open(args.model_out, "wb") as f:
        pickle.dump(model, f)

    # decision table for all rows
    j["pred_proba"] = predict_proba(model, j)
    out = j[["asof","symbol","uf_score","uf_label","pred_proba","trade_count"]].copy()
    write_csv(args.decision_out, out.sort_values(["asof","pred_proba"], ascending=[True, False]))

    update_run_report(args.report, {
        "step_03": {
            "lambda": lam,
            "features_used": model.feature_cols,
            "labeled_rows": int(len(train)),
            "decision_rows": int(len(out))
        }
    })
    print(f"✅ Step03 done. features={len(model.feature_cols)} labeled={len(train)}")


if __name__ == "__main__":
    main()
''',

"step_04_select_universe.py": r'''# =========================================================
# Step 04 - Select Universe
# ---------------------------------------------------------
# Generated: {UTC_NOW}
# =========================================================
from __future__ import annotations

import argparse
import json
import pickle
import pandas as pd

from universe_filter.lib.io import read_json, read_parquet, write_json, write_csv
from universe_filter.lib.selection import apply_safety_gates, select_top_n
from universe_filter.lib.model import predict_proba
from universe_filter.lib.reporting import update_run_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="universe_filter/config/uf_config.json")
    ap.add_argument("--ds", default="universe_filter/output/cache/uf_dataset.parquet")
    ap.add_argument("--model", default="universe_filter/output/cache/model.pkl")
    ap.add_argument("--selected_out", default="universe_filter/output/cache/selected_symbols.json")
    ap.add_argument("--decision_out", default="universe_filter/output/decision_table.csv")
    ap.add_argument("--report", default="universe_filter/output/run_report.json")
    args = ap.parse_args()

    cfg = read_json(args.config)
    n_select = int(cfg.get("n_select", 25))
    min_hist = int(cfg.get("min_history_days", 90))
    excl0 = bool(cfg.get("exclude_if_trade_count_7d_zero", True))
    majors = list(cfg.get("major_force_include", []))

    ds = read_parquet(args.ds)
    if ds.empty:
        update_run_report(args.report, {"step_04": {"note": "dataset empty"}})
        print("⚠️ Step04 skipped (dataset empty).")
        return

    ds["asof"] = pd.to_datetime(ds["asof"], utc=True)
    latest_asof = ds["asof"].max()
    latest = ds[ds["asof"] == latest_asof].copy()

    # load model
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    # predict
    latest["pred_proba"] = predict_proba(model, latest)

    # safety gates need trade_count_7d (may be absent depending on caps)
    # If missing, we won't exclude by that gate.
    gated = apply_safety_gates(latest, min_history_days=min_hist, exclude_trade_count_7d_zero=excl0)

    selected, ranked = select_top_n(gated, score_col="pred_proba", n=n_select, majors=majors)

    write_json(args.selected_out, {"asof": latest_asof.isoformat(), "selected": selected})

    # update decision table (latest snapshot)
    dec = ranked[["asof","symbol","pred_proba"]].copy().sort_values("pred_proba", ascending=False)
    write_csv(args.decision_out, dec)

    update_run_report(args.report, {"step_04": {"latest_asof": latest_asof.isoformat(), "n_selected": len(selected)}})
    print(f"✅ Step04 done. latest_asof={latest_asof.isoformat()} selected={len(selected)}")


if __name__ == "__main__":
    main()
''',

"step_05_export_universe.py": r'''# =========================================================
# Step 05 - Export universe.json (Single Writer)
# ---------------------------------------------------------
# Generated: {UTC_NOW}
# =========================================================
from __future__ import annotations

import argparse
import pandas as pd

from universe_filter.lib.io import read_json, read_parquet, write_json
from universe_filter.lib.reporting import update_run_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="universe_filter/config/uf_config.json")
    ap.add_argument("--selected", default="universe_filter/output/cache/selected_symbols.json")
    ap.add_argument("--trades", default="universe_filter/output/cache/trades.parquet")
    ap.add_argument("--out", default="universe_filter/output/universe.json")
    ap.add_argument("--report", default="universe_filter/output/run_report.json")
    args = ap.parse_args()

    cfg = read_json(args.config)
    sel = read_json(args.selected)
    selected = sel.get("selected", [])
    asof = sel.get("asof", None)

    # zombie_N (median holding over last 90d) from trades
    trades = read_parquet(args.trades)
    zombie_N = None
    if not trades.empty:
        trades["exit_ts"] = pd.to_datetime(trades["exit_ts"], utc=True)
        tmax = trades["exit_ts"].max()
        w = trades[(trades["exit_ts"] > tmax - pd.Timedelta(days=90)) & (trades["exit_ts"] <= tmax)]
        if not w.empty:
            med_min = float(w["holding_minutes"].median())
            zombie_N = max(1, int(round(med_min / (60 * 24))))  # days

    out = {
        "asof": asof,
        "rebalance_rule": cfg.get("rebalance_rule", "WEEKLY"),
        "universe": selected,
        "meta": {
            "n_selected": len(selected),
            "model": cfg.get("model", {}).get("name", "RandomForestClassifier"),
            "windows_days": cfg.get("windows_days", [7, 30, 90]),
            "objective": ["mdd_down", "trade_count_std_down", "holding_iqr_down"],
            "zombie_N_days": zombie_N,
        }
    }

    # ✅ single-writer rule: only this step writes universe.json
    write_json(args.out, out)
    update_run_report(args.report, {"step_05": {"universe_written": True, "n": len(selected), "zombie_N_days": zombie_N}})
    print(f"✅ Step05 done. universe.json written. n={len(selected)}")


if __name__ == "__main__":
    main()
''',
}

    # NOTE: Step02/03 rely on 'trade_count_7d' only if it's kept in caps; v1 baseline keeps robust even if missing.
    #       If you want trade_count_7d always present, adjust feature policy caps accordingly.

    # Ensure Step05 is the only writer of universe.json by convention; generator does not add writers elsewhere.

if __name__ == "__main__":
    main()
