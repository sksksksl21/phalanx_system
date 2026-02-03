# =========================================================
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
        _run([sys.executable, "-m", "universe_filter.steps." + s.stem])
    print("✅ Pipeline completed.")


if __name__ == "__main__":
    main()
