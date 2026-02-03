from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

ROOT = Path("universe_filter")

# 빈 파일일 때만 헤더를 써넣는 정책(안전)
WRITE_ONLY_IF_EMPTY = True

UTC_NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_empty(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        return path.stat().st_size == 0
    except Exception:
        return False


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if WRITE_ONLY_IF_EMPTY and (not _is_empty(path)):
        return
    path.write_text(content, encoding="utf-8")


def _py_header(title: str, role: str, inputs: list[str], outputs: list[str], notes: list[str]) -> str:
    lines: list[str] = []
    lines.append("# =========================================================")
    lines.append(f"# {title}")
    lines.append("# ---------------------------------------------------------")
    lines.append(f"# Generated: {UTC_NOW}")
    lines.append("#")
    lines.append("# HARD RULES (v1.0):")
    lines.append("# - Strategy logic (entry/positioning/exit) MUST NOT be modified.")
    lines.append("# - UF MUST output ONLY universe.json for engine consumption.")
    lines.append("# - All files MUST live under universe_filter/.")
    lines.append("# - Pipeline is step-by-step; each step is a single runnable module.")
    lines.append("# - Only Step 05 writes output/universe.json.")
    lines.append("# =========================================================")
    lines.append("")
    lines.append('"""')
    lines.append("Role")
    lines.append("----")
    lines.append(role.strip())
    lines.append("")
    lines.append("Inputs")
    lines.append("------")
    for x in inputs:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("Outputs")
    lines.append("-------")
    for x in outputs:
        lines.append(f"- {x}")
    if notes:
        lines.append("")
        lines.append("Notes")
        lines.append("-----")
        for n in notes:
            lines.append(f"- {n}")
    lines.append('"""')
    lines.append("")
    return "\n".join(lines)


def _md_header(title: str, bullets: list[str]) -> str:
    lines = [f"# {title}", "", f"_Generated: {UTC_NOW}_", ""]
    lines += [f"- {b}" for b in bullets]
    lines.append("")
    return "\n".join(lines)


def _json_placeholder(comment: str) -> str:
    # JSON은 주석을 허용하지 않으니 최소 스켈레톤만
    return "{\n" + f'  "_comment": "{comment}"\n' + "}\n"


def main() -> None:
    # run_pipeline.py (엔트리포인트: 코드 로직은 나중에 채움)
    _write(
        ROOT / "run_pipeline.py",
        _py_header(
            title="Universe Filter - Pipeline Runner (ENTRYPOINT)",
            role=(
                "Sequentially runs Step 00 ~ Step 05.\n"
                "This file is the ONLY pipeline entrypoint.\n"
                "Implementation will be added later; header only for now."
            ),
            inputs=[
                "universe_filter/input/backtest_history.csv",
                "(optional) universe_filter/input/ohlcv/*",
                "universe_filter/config/uf_config.json",
            ],
            outputs=[
                "universe_filter/output/universe.json (written by Step 05 only)",
                "universe_filter/output/run_report.json",
                "universe_filter/output/decision_table.csv",
                "universe_filter/output/cache/*",
            ],
            notes=[
                "Do NOT implement strategy logic here.",
                "Orchestrates steps only; each step is a standalone runnable module.",
            ],
        ),
    )

    # steps headers
    _write(
        ROOT / "steps" / "step_00_ingest.py",
        _py_header(
            title="Step 00 - Ingest & Normalize",
            role="Converts input/backtest_history.csv into normalized Trade Ledger with trade_id and holding period.",
            inputs=["universe_filter/input/backtest_history.csv"],
            outputs=["universe_filter/output/cache/trades.parquet", "(report additions) universe_filter/output/run_report.json"],
            notes=[
                "Match ENTRY~EXIT into trade_id per (symbol, side).",
                "Record edge cases (missing EXIT, duplicates) into run_report (do not stop pipeline).",
            ],
        ),
    )

    _write(
        ROOT / "steps" / "step_01_persistence.py",
        _py_header(
            title="Step 01 - Persistence Check (EDA Gate)",
            role="Quickly checks whether past performance relates to next-period performance; influences feature policy only.",
            inputs=["universe_filter/output/cache/trades.parquet"],
            outputs=["universe_filter/output/run_report.json", "(optional) universe_filter/output/persistence_table.csv"],
            notes=[
                "Pipeline must NOT stop here; it only writes guidance flags.",
                "Compute Pearson/Spearman correlations on walk-forward style slices.",
            ],
        ),
    )

    _write(
        ROOT / "steps" / "step_02_build_dataset.py",
        _py_header(
            title="Step 02 - Build Symbol×Window Metrics & UF Dataset",
            role="Builds windowed metrics (7/30/90) and creates uf_dataset for modeling; enforces feature caps and corr-cut.",
            inputs=[
                "universe_filter/output/cache/trades.parquet",
                "universe_filter/config/uf_config.json",
                "(optional) universe_filter/input/ohlcv/*",
            ],
            outputs=[
                "universe_filter/output/cache/symbol_window_metrics.parquet",
                "universe_filter/output/cache/uf_dataset.parquet",
                "(report additions) universe_filter/output/run_report.json",
            ],
            notes=[
                "Feature count <= 20; outcome-based features <= 3.",
                "If OHLCV missing, OHLCV-based features are skipped WITHOUT failing pipeline.",
                "Apply |corr| > 0.80 cut (drop one of pair).",
            ],
        ),
    )

    _write(
        ROOT / "steps" / "step_03_train_validate.py",
        _py_header(
            title="Step 03 - Train & Walk-forward Validate",
            role="Trains RandomForest with constraints and validates via walk-forward; computes stability objectives (MDD/std/IQR).",
            inputs=["universe_filter/output/cache/uf_dataset.parquet", "universe_filter/config/uf_config.json"],
            outputs=[
                "(optional) universe_filter/output/cache/model.pkl",
                "universe_filter/output/decision_table.csv",
                "universe_filter/output/run_report.json",
            ],
            notes=[
                "RandomForest constraints: max_depth <= 5, min_samples_leaf >= 10.",
                "Permutation importance pruning: drop bottom 20% features (1~3 loops).",
                "Objective vector: MDD down, trade_count_std down, holding_iqr down.",
            ],
        ),
    )

    _write(
        ROOT / "steps" / "step_04_select_universe.py",
        _py_header(
            title="Step 04 - Select Universe",
            role="Selects N symbols for next rebalance using model scores + safety gates; writes selected_symbols.json (cache).",
            inputs=[
                "universe_filter/output/cache/uf_dataset.parquet (latest asof snapshot)",
                "(optional) universe_filter/output/cache/model.pkl",
                "universe_filter/config/uf_config.json",
            ],
            outputs=[
                "universe_filter/output/cache/selected_symbols.json",
                "universe_filter/output/decision_table.csv (final asof update)",
                "(report additions) universe_filter/output/run_report.json",
            ],
            notes=[
                "Default: pick top-N by model score.",
                "Safety: exclude trade_count_7d=0; exclude insufficient history (<90d) by default.",
                "Optional config: force-include majors (still does not touch strategy).",
            ],
        ),
    )

    _write(
        ROOT / "steps" / "step_05_export_universe.py",
        _py_header(
            title="Step 05 - Export universe.json (Single Writer)",
            role="Writes output/universe.json in the fixed schema using selected_symbols.json; this is the ONLY writer.",
            inputs=["universe_filter/output/cache/selected_symbols.json", "universe_filter/config/uf_config.json"],
            outputs=["universe_filter/output/universe.json"],
            notes=[
                "MUST be the single writer of universe.json.",
                "Include meta: asof, rebalance_rule, windows_days, objective, n_selected, (optional) zombie_N.",
            ],
        ),
    )

    # lib headers (shared library; not directly executed as steps)
    lib_files = {
        "io.py": "I/O helpers: read/write CSV/JSON/Parquet; path utilities; safe writes.",
        "schema.py": "Schema contracts & validators for inputs/outputs/intermediate artifacts.",
        "trades.py": "Trade ledger normalization, ENTRY~EXIT matching, trade_id assignment, holding calculations.",
        "metrics.py": "Metrics calculators: net_profit, winrate, PF, expectancy, equity MDD, trade_count std, holding IQR, etc.",
        "features.py": "Feature builder enforcing caps (<=20, outcome-based <=3), OHLCV-optional feature gating, corr-cut.",
        "labeling.py": "Score/label definitions: Score=NetProfit/(MDD+lambda), lambda scaling, label buckets (30/50/20).",
        "model.py": "Model factory/training for constrained RandomForest; permutation importance & feature pruning loop.",
        "validation.py": "Walk-forward splitter and evaluation for objective vector; pre/post comparisons.",
        "selection.py": "Universe selection logic & safety gates; force-include majors; explain drop reasons.",
        "reporting.py": "run_report.json and decision_table.csv generation/updating utilities.",
    }
    for fname, role in lib_files.items():
        _write(
            ROOT / "lib" / fname,
            _py_header(
                title=f"Universe Filter - lib/{fname}",
                role=role,
                inputs=["(varies)"],
                outputs=["(varies)"],
                notes=[
                    "This is a shared library module (NOT a runnable step).",
                    "Must not touch strategy logic; only UF data/modeling/selection/reporting.",
                ],
            ),
        )

    # README.md
    _write(
        ROOT / "README.md",
        _md_header(
            "Universe Filter (UF) - v1.0 Scaffold",
            [
                "UF runs BEFORE strategy; it selects the tradable symbol list only.",
                "Strategy logic (entry/positioning/exit) must NOT be modified.",
                "Pipeline is step-by-step: steps/step_00 ~ step_05.",
                "Only steps/step_05_export_universe.py writes output/universe.json.",
                "Primary input: input/backtest_history.csv",
                "Optional input: input/ohlcv/* (if absent, OHLCV features are skipped safely).",
                "Entry point: run_pipeline.py (will orchestrate steps).",
            ],
        ),
    )

    # uf_config.json placeholder (valid JSON)
    _write(
        ROOT / "config" / "uf_config.json",
        _json_placeholder("UF config placeholder. Will be replaced by final uf_config schema document."),
    )

    # output placeholders (valid JSON or empty text)
    _write(
        ROOT / "output" / "run_report.json",
        _json_placeholder("Run report placeholder. Steps append execution metadata and validation results here."),
    )
    _write(
        ROOT / "output" / "universe.json",
        _json_placeholder("Universe placeholder. Must be written ONLY by Step 05."),
    )
    # CSVs can be empty
    _write(ROOT / "output" / "decision_table.csv", "")

    print("✅ UF headers stamped (empty files only).")
    print(f"📁 root: {ROOT.resolve()}")


if __name__ == "__main__":
    main()
