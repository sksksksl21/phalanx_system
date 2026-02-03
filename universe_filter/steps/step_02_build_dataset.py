# =========================================================
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
