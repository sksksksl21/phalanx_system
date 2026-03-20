import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- load ---
df = pd.read_csv("ohlcv_15m.csv")  # 네 파일명으로 변경
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp").sort_index()

p = {
    "atr_period": 17,
    "swing_len": 5,
    "context_lookback": 120,
    "retest_tolerance_atr": 0.4,
    "atr_regime_len": 60,
    "atr_regime_factor": 1.06,
    "atr_slope_gate": True,
}

def atr_simple(df, period):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def compute_pivots_confirmed(df, swing_len):
    n = int(max(1, swing_len))
    w = 2*n + 1
    roll_high = df["high"].rolling(window=w, min_periods=w).max()
    roll_low  = df["low"].rolling(window=w, min_periods=w).min()
    pivot_high = (df["high"].shift(n) == roll_high).astype(int).fillna(0).astype(int)
    pivot_low  = (df["low"].shift(n) == roll_low).astype(int).fillna(0).astype(int)
    return pivot_high, pivot_low

def age_from_triggers(trigger):
    t = trigger.fillna(0).astype(int).to_numpy()
    idxn = np.arange(len(t), dtype=float)
    last = np.where(t == 1, idxn, np.nan)
    last = pd.Series(last, index=trigger.index).ffill()
    age = pd.Series(np.arange(len(t), dtype=float), index=trigger.index) - last
    return age.where(last.notna(), np.nan)

# --- core columns ---
out = df.copy()
out["atr"] = atr_simple(out, p["atr_period"]).bfill()
out["atr_ma"] = out["atr"].rolling(window=p["atr_regime_len"],
                                   min_periods=max(2, p["atr_regime_len"]//3)).mean().bfill()
out["atr_ratio"] = (out["atr"] / out["atr_ma"].replace(0, np.nan)).fillna(0.0)
out["atr_up"] = (out["atr"] > out["atr"].shift(1)).astype(int).fillna(0).astype(int)

out["pivot_high"], out["pivot_low"] = compute_pivots_confirmed(out, p["swing_len"])
nshift = int(max(1, p["swing_len"]))
out["pivot_high_price"] = np.where(out["pivot_high"] == 1, out["high"].shift(nshift), np.nan)
out["pivot_low_price"]  = np.where(out["pivot_low"] == 1, out["low"].shift(nshift), np.nan)
out["last_pivot_high"] = pd.Series(out["pivot_high_price"], index=out.index).ffill()
out["last_pivot_low"]  = pd.Series(out["pivot_low_price"], index=out.index).ffill()

lph = out["last_pivot_high"].astype(float)
lpl = out["last_pivot_low"].astype(float)

out["sweep_low"] = ((out["low"] < lpl) & (out["close"] > lpl) & lpl.notna()).astype(int)
out["recent_sweep_low"] = out["sweep_low"].rolling(p["context_lookback"], min_periods=1).max().fillna(0).astype(int)

out["choch_up_trigger"] = (
    (out["recent_sweep_low"] == 1) & lph.notna()
    & (out["close"] > lph)
    & (out["close"].shift(1) <= lph.shift(1))
).fillna(False).astype(int)

out["choch_level_up_raw"] = np.where(out["choch_up_trigger"] == 1, lph, np.nan)
out["choch_age_up"] = age_from_triggers(out["choch_up_trigger"])
out["choch_active_up"] = ((out["choch_age_up"].notna()) & (out["choch_age_up"] <= p["context_lookback"])).astype(int)

# (간단화) bos_ok_long=1로 두고, 리테스트+레짐만 먼저 시각화
reset_up = (out["choch_up_trigger"] == 1)
seg_up = reset_up.astype(int).cumsum()
out["choch_level_up"] = pd.Series(out["choch_level_up_raw"], index=out.index).groupby(seg_up).ffill()

lvl_up = out["choch_level_up"]
tol = out["atr"] * float(p["retest_tolerance_atr"])

# "현재 titan_strategy.py 최종값"과 동일한 형태(같은 봉도 가능)
out["retest_long"] = ((out["choch_active_up"] == 1) & lvl_up.notna()
                      & (out["low"] <= (lvl_up + tol))
                      & (out["close"] >= lvl_up)
                      & (out["close"] > out["open"])).astype(int)

# vol gate
out["vol_gate_ok"] = (out["atr_ratio"] > p["atr_regime_factor"]).astype(int)
if p["atr_slope_gate"]:
    out["vol_gate_ok"] = (out["vol_gate_ok"] & (out["atr_up"] == 1)).astype(int)

out["signal_long"] = ((out["retest_long"] == 1) & (out["vol_gate_ok"] == 1)).astype(int)
out["entry_next_open"] = out["signal_long"].shift(1).fillna(0).astype(int)

# --- plot ---
plt.figure(figsize=(12,4))
plt.plot(out.index, out["close"].values, label="close")
m = out["choch_level_up"].notna()
plt.plot(out.index[m], out.loc[m, "choch_level_up"].values, label="CHOCH level")

def mark(col, marker, label):
    mm = out[col].astype(int) == 1
    if mm.any():
        plt.scatter(out.index[mm], out.loc[mm, "close"].values, marker=marker, label=label)

mark("sweep_low", "v", "sweep_low")
mark("choch_up_trigger", "s", "choch_up_trigger")
mark("retest_long", "*", "retest_long")
mark("signal_long", "^", "signal_long")
mark("entry_next_open", "D", "entry(next_open)")

plt.legend(loc="best")
plt.tight_layout()
plt.show()