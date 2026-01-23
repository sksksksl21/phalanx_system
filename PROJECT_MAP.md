# Phalanx System – Project Map

This document defines the functional architecture of the Phalanx trading system.
ChatGPT must use this map to understand responsibilities, data flow, and invariants.

────────────────────────────────────────
I. SYSTEM PURPOSE
────────────────────────────────────────
Phalanx is a quantitative crypto trading system designed for:
- Parameter optimization
- Backtest verification
- Forward testing
- Live trading
All four must operate on identical strategy logic.

The goal is not backtest profit.
The goal is survival and profitability under real market evolution.

────────────────────────────────────────
II. CORE EXECUTION FLOW
────────────────────────────────────────

Market Data
   ↓
strategy/titan_strategy.py
   ↓
strategy/risk_control.py
   ↓
strategy/position_monitor.py
   ↓
execution/(virtual_executor | binance_executor)
   ↓
core/(backtest_engine | live_engine)
   ↓
State & Logs

This flow must never be broken.

────────────────────────────────────────
III. MAJOR MODULES
────────────────────────────────────────

/strategy/
- titan_strategy.py
  Entry logic, regime filters, signal generation.
  No execution or money management here.

- risk_control.py
  Position sizing, risk per trade, margin control.

- position_monitor.py
  Exit logic, trailing stops, profit protection.

These three define the "brain" of Phalanx.

----------------------------------------

/core/
- backtest_engine.py
  Time-synced simulation of multiple symbols.

- live_engine.py
  Real exchange execution.

- main_engine.py
  Deprecated. Exists for legacy reasons only.
  Must never be referenced by any new logic.

----------------------------------------

/execution/
- virtual_executor.py
  Simulated fills for backtesting and forward testing.

- binance_executor.py
  Real trading interface.

Both must behave identically except for connectivity.

----------------------------------------

Top-level scripts:

- optimize.py
  Searches for optimal parameters.
  Does NOT validate realism.

- run_param_backtest.py
  Replays optimize results under realistic constraints.

- backtest_dashboard.py / dashboard.py
  Visualization only. Must not affect logic.

----------------------------------------

/utils/
- data_loader.py
  Historical data fetch & caching.

- history_manager.py
  Trade history & performance tracking.

- telegram_bot.py
  Notification only. No trading authority.

────────────────────────────────────────
IV. EXECUTION MODES
────────────────────────────────────────

Phalanx operates in three time modes:

1. Optimization Mode
   optimize.py
   → finds statistically strong parameter sets.

2. Validation Mode
   run_param_backtest.py
   → tests if optimize results survive realism.

3. Forward Test / Live Mode
   backtest_engine.py (forward) or live_engine.py
   → tests survival in new market regimes.

These modes must never share logic forks.

────────────────────────────────────────
V. INVARIANTS (MUST NEVER BE VIOLATED)
────────────────────────────────────────
- Strategy logic is single-source (strategy/ folder).
- No lookahead or future data leakage.
- Same code must drive backtest and live.
- Risk control is enforced centrally.
- Execution must be time-consistent.

────────────────────────────────────────
VI. FORBIDDEN ASSUMPTIONS
────────────────────────────────────────
ChatGPT must never:
- Assume the market is stationary
- Optimize for win rate
- Optimize for backtest return only
- Remove safety systems for profit

The system is designed to survive, not to look good.
