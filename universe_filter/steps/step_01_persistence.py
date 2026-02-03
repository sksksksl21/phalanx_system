# =========================================================
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
