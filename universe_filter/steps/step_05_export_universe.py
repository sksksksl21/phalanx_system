# =========================================================
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
