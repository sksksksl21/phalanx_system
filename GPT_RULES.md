# Phalanx System – GPT Working Constitution

This repository is the SINGLE SOURCE OF TRUTH for all code, structure, and logic.
ChatGPT must always read and respect the current state of this repository before proposing any change.

────────────────────────────────────────
I. SOURCE OF TRUTH
────────────────────────────────────────
1. The GitHub main branch is the only authoritative codebase.
2. ChatGPT must never assume, invent, or rely on memory from previous chats.
3. All reasoning, analysis, and changes must be based on files read from this repository.

────────────────────────────────────────
II. STRUCTURE IMMUTABILITY
────────────────────────────────────────
1. Folder and file structure is CONSTITUTIONAL.
2. No folder or file may be moved, renamed, deleted, or merged unless explicitly ordered by the user.
3. No new architecture, abstraction layer, or framework may be introduced without approval.

────────────────────────────────────────
III. TWO-STEP CHANGE PROCESS
────────────────────────────────────────
All changes must follow this strict protocol:

STEP 1 – DESIGN ONLY
ChatGPT must first provide:
- What will be changed
- Why it is necessary
- Which files will be touched
- What side effects are possible
- How it will be verified

No code is allowed in Step 1.

STEP 2 – PATCH
Only after user approval, ChatGPT may output:
- A git-diff style patch
- List of modified files
- Commands to test or verify the change

────────────────────────────────────────
IV. CHANGE SCOPE CONTROL
────────────────────────────────────────
1. All changes must be minimal and local.
2. Existing logic, behavior, and interfaces must be preserved unless explicitly approved.
3. Refactoring is forbidden unless requested.
4. Optimization is forbidden unless requested.

────────────────────────────────────────
V. BACKTEST & LIVE INTEGRITY
────────────────────────────────────────
1. Strategy logic inside `strategy/` must behave identically in:
   - optimize.py
   - run_param_backtest.py
   - backtest_engine
   - live_engine
2. No logic may diverge between testing and live execution.
3. Time handling, candle indexing, and lookahead prevention are sacred.

────────────────────────────────────────
VI. FORBIDDEN FILES
────────────────────────────────────────
These files are NOT part of the system logic and must never be used for reasoning:
- market_data_cache.pkl
- phalanx_state.json
- Any *.csv, *.log, *.pkl
These are runtime artifacts, not source code.

────────────────────────────────────────
VII. OUTPUT FORMAT
────────────────────────────────────────
All responses must be structured:

DESIGN MODE:
- Objective
- Files involved
- Logic changes
- Risks
- Verification plan

PATCH MODE:
- Modified files list
- git diff
- Test / run commands

────────────────────────────────────────
VIII. FAILURE MODE
────────────────────────────────────────
If ChatGPT is missing files, permissions, or clarity,
it must stop and request access rather than guessing.

Guessing is forbidden.
