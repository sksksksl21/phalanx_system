# =========================================================
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
