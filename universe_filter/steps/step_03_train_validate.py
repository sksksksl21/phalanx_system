# =========================================================
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

    train[feature_cols] = train[feature_cols].replace([np.inf, -np.inf], np.nan)
    train[feature_cols] = train[feature_cols].fillna(0.0)


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
    
    j[model.feature_cols] = j[model.feature_cols].replace([np.inf, -np.inf], np.nan)
    j[model.feature_cols] = j[model.feature_cols].fillna(0.0)
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
