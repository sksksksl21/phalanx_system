# =========================================================
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
    n_select = int(cfg.get("n_select", 18))
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
    if gated.empty:
        gated = latest
    selected, ranked = select_top_n(gated, score_col="pred_proba", n=n_select, majors=majors)

    write_json(args.selected_out, {"asof": latest_asof.isoformat(), "selected": selected})

    # update decision table (latest snapshot)
    dec = ranked[["asof","symbol","pred_proba"]].copy().sort_values("pred_proba", ascending=False)
    write_csv(args.decision_out, dec)

    update_run_report(args.report, {"step_04": {"latest_asof": latest_asof.isoformat(), "n_selected": len(selected)}})
    print(f"✅ Step04 done. latest_asof={latest_asof.isoformat()} selected={len(selected)}")


if __name__ == "__main__":
    main()
