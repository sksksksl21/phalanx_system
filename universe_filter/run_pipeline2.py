# universe_filter/step_02.py
# =========================================================
# Step 02 (UF 2.0 - Simplified Ranker)
# ---------------------------------------------------------
# Purpose (your exact intent):
#   - Download TopN (default 100) symbols from Binance USDT Perp (by quoteVolume)
#   - Compute UF 2.0 chart features (Titan-aligned + SuperTrend-aligned)
#   - Rank by uf_score and output:
#       (1) decision_table.csv      (full ranked table for inspection)
#       (2) universe_step2.json     (TopK symbols only; YOU pick & use later)
#
# IMPORTANT:
#   - This step DOES NOT write universe.json (outside Step1~3 authority per your rule)
#   - No trade-log/panel/dataset building here. Pure screening + ranking generator.
# =========================================================

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================
# Utility
# =============================
def _ensure_dir(p: str) -> None:
    if not p:
        return
    os.makedirs(p, exist_ok=True)


def _log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except Exception:
        pass


def _read_json(path: str) -> dict:
    if not path or (not os.path.exists(path)):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _write_json(path: str, obj: dict) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =============================
# Indicator helpers (fallback)
# =============================
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    rs = up.ewm(alpha=1 / period, adjust=False).mean() / (dn.ewm(alpha=1 / period, adjust=False).mean() + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    # Wilder ADX (EMA approximation)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = _true_range(high, low, close).replace(0, np.nan)
    atrv = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100.0 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / (atrv + 1e-12)
    minus_di = 100.0 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / (atrv + 1e-12)
    dx = (100.0 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12)).fillna(0.0)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx


def _supertrend(df: pd.DataFrame, atr_period: int = 10, atr_mult: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """
    Returns:
      st_value: supertrend line
      st_dir:   +1 bullish, -1 bearish
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    atrv = _atr(high, low, close, period=atr_period)
    hl2 = (high + low) / 2.0
    upper = hl2 + atr_mult * atrv
    lower = hl2 - atr_mult * atrv

    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)

    st.iloc[0] = float(upper.iloc[0])
    direction.iloc[0] = -1.0

    for i in range(1, len(df)):
        prev_dir = float(direction.iloc[i - 1])
        cur_close = float(close.iloc[i])

        cur_upper = float(upper.iloc[i])
        cur_lower = float(lower.iloc[i])

        # band tightening rules
        prev_upper = float(upper.iloc[i - 1])
        prev_lower = float(lower.iloc[i - 1])
        prev_close = float(close.iloc[i - 1])

        adj_upper = cur_upper if (cur_upper < prev_upper or prev_close > prev_upper) else prev_upper
        adj_lower = cur_lower if (cur_lower > prev_lower or prev_close < prev_lower) else prev_lower

        # direction switch
        if prev_dir < 0 and cur_close > adj_upper:
            cur_dir = 1.0
        elif prev_dir > 0 and cur_close < adj_lower:
            cur_dir = -1.0
        else:
            cur_dir = prev_dir

        cur_st = adj_lower if cur_dir > 0 else adj_upper
        st.iloc[i] = float(cur_st)
        direction.iloc[i] = float(cur_dir)

    return st, direction


# =============================
# UF 2.0 chart feature spec
# =============================
@dataclass
class TitanParams:
    atr_period: int = 25
    atr_multiplier: float = 4.5
    adx_threshold: int = 17
    rsi_upper: int = 73
    rsi_lower: int = 28
    vol_factor: float = 0.9
    ema_intraday: int = 200
    daily_ema: int = 5
    swing_len: int = 3
    context_lookback: int = 120
    retest_tolerance_atr: float = 0.25
    use_daily_filter: bool = True
    use_vol_filter: bool = True


def _load_titan_params(cfg: dict) -> TitanParams:
    p = (cfg or {}).get("strategy_settings", {}) or {}
    uf_p = (cfg or {}).get("titan_params", {}) or {}
    z = {**p, **uf_p}

    def _b(k: str, default: bool) -> bool:
        try:
            return bool(z.get(k, default))
        except Exception:
            return default

    def _i(k: str, default: int) -> int:
        try:
            return int(z.get(k, default))
        except Exception:
            return default

    def _f(k: str, default: float) -> float:
        try:
            return float(z.get(k, default))
        except Exception:
            return default

    return TitanParams(
        atr_period=_i("atr_period", 25),
        atr_multiplier=_f("atr_multiplier", 4.5),
        adx_threshold=_i("adx_threshold", 17),
        rsi_upper=_i("rsi_upper", 73),
        rsi_lower=_i("rsi_lower", 28),
        vol_factor=_f("vol_factor", 0.9),
        ema_intraday=_i("ema_intraday", 200),
        daily_ema=_i("daily_ema", 5),
        swing_len=_i("swing_len", 3),
        context_lookback=_i("context_lookback", 120),
        retest_tolerance_atr=_f("retest_tolerance_atr", 0.25),
        use_daily_filter=_b("use_daily_filter", True),
        use_vol_filter=_b("use_vol_filter", True),
    )


def _compute_pivots(df: pd.DataFrame, swing_len: int) -> Tuple[pd.Series, pd.Series]:
    s = int(max(1, swing_len))
    win = 2 * s + 1
    h = df["high"]
    l = df["low"]
    roll_max = h.rolling(win, center=True).max()
    roll_min = l.rolling(win, center=True).min()
    pivot_high = (h == roll_max).fillna(False) & roll_max.notna()
    pivot_low = (l == roll_min).fillna(False) & roll_min.notna()
    return pivot_high.astype(bool), pivot_low.astype(bool)


def _retest_and_sweep_metrics(
    df: pd.DataFrame,
    atrv: pd.Series,
    pivot_high: pd.Series,
    pivot_low: pd.Series,
    tol_atr: float,
    ctx: int,
    follow_n: int = 8,
    follow_atr: float = 1.0,
) -> Dict[str, float]:
    close = df["close"]
    high = df["high"]
    low = df["low"]

    n = len(df)

    def _safe_rate(num: int, den: int) -> float:
        return float(num) / float(den) if den > 0 else 0.0

    total_retests = 0
    ok_retests = 0
    total_sweeps = 0
    ok_sweeps = 0

    piv_hi_idx = np.where(pivot_high.values)[0].tolist()
    piv_lo_idx = np.where(pivot_low.values)[0].tolist()

    # pivot lows
    for pi in piv_lo_idx:
        level = float(low.iloc[pi])
        j_end = min(n - 1, pi + int(ctx))
        retest_j = None
        for j in range(pi + 1, j_end + 1):
            atr_j = float(atrv.iloc[j]) if math.isfinite(float(atrv.iloc[j])) else 0.0
            tol = float(tol_atr) * atr_j
            if tol <= 0:
                continue
            if abs(float(close.iloc[j]) - level) <= tol:
                retest_j = j
                break
        if retest_j is None:
            continue

        total_retests += 1
        atr_r = float(atrv.iloc[retest_j]) if math.isfinite(float(atrv.iloc[retest_j])) else 0.0
        if atr_r > 0:
            k_end = min(n - 1, retest_j + int(follow_n))
            if (float(close.iloc[retest_j : k_end + 1].max()) - level) >= (follow_atr * atr_r):
                ok_retests += 1

        # sweep: low breaks below, close back above level
        tol = float(tol_atr) * atr_r
        if atr_r > 0 and tol > 0:
            if (float(low.iloc[retest_j]) < (level - tol)) and (float(close.iloc[retest_j]) > level):
                total_sweeps += 1
                k_end = min(n - 1, retest_j + int(follow_n))
                if (float(close.iloc[retest_j : k_end + 1].max()) - float(close.iloc[retest_j])) >= (follow_atr * atr_r):
                    ok_sweeps += 1

    # pivot highs
    for pi in piv_hi_idx:
        level = float(high.iloc[pi])
        j_end = min(n - 1, pi + int(ctx))
        retest_j = None
        for j in range(pi + 1, j_end + 1):
            atr_j = float(atrv.iloc[j]) if math.isfinite(float(atrv.iloc[j])) else 0.0
            tol = float(tol_atr) * atr_j
            if tol <= 0:
                continue
            if abs(float(close.iloc[j]) - level) <= tol:
                retest_j = j
                break
        if retest_j is None:
            continue

        total_retests += 1
        atr_r = float(atrv.iloc[retest_j]) if math.isfinite(float(atrv.iloc[retest_j])) else 0.0
        if atr_r > 0:
            k_end = min(n - 1, retest_j + int(follow_n))
            if (level - float(close.iloc[retest_j : k_end + 1].min())) >= (follow_atr * atr_r):
                ok_retests += 1

        # sweep: high breaks above, close back below level
        tol = float(tol_atr) * atr_r
        if atr_r > 0 and tol > 0:
            if (float(high.iloc[retest_j]) > (level + tol)) and (float(close.iloc[retest_j]) < level):
                total_sweeps += 1
                k_end = min(n - 1, retest_j + int(follow_n))
                if (float(close.iloc[retest_j]) - float(close.iloc[retest_j : k_end + 1].min())) >= (follow_atr * atr_r):
                    ok_sweeps += 1

    return {
        "retest_success_rate": _safe_rate(ok_retests, total_retests),
        "sweep_to_followthrough": _safe_rate(ok_sweeps, total_sweeps),
        "pivot_density": _safe_rate(len(piv_hi_idx) + len(piv_lo_idx), max(1, n)),
    }


def _wick_body_ratio(df: pd.DataFrame) -> float:
    o = df["open"].astype(float)
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)

    body = (c - o).abs()
    rng = (h - l).abs()
    wick = (rng - body).clip(lower=0.0)

    denom = body.replace(0.0, np.nan)
    r = (wick / denom).replace([np.inf, -np.inf], np.nan).dropna()
    return float(r.mean()) if not r.empty else 0.0


def _range_to_atr_ratio(df: pd.DataFrame, atrv: pd.Series) -> float:
    rng = (df["high"].astype(float) - df["low"].astype(float)).abs()
    denom = atrv.replace(0.0, np.nan)
    r = (rng / denom).replace([np.inf, -np.inf], np.nan).dropna()
    return float(r.mean()) if not r.empty else 0.0


def _volume_stability(df: pd.DataFrame) -> float:
    v = df["volume"].astype(float)
    m = float(v.mean()) if len(v) else 0.0
    s = float(v.std()) if len(v) else 0.0
    if m <= 0 or (not math.isfinite(m)) or (not math.isfinite(s)):
        return 0.0
    return float(s / m)


def _titan_candidate_mask(
    df: pd.DataFrame,
    params: TitanParams,
    atrv: pd.Series,
    adxv: pd.Series,
    rsiv: pd.Series,
    ema200: pd.Series,
    vol_ma: pd.Series,
    daily_ema_15m: Optional[pd.Series],
) -> Tuple[pd.Series, pd.Series]:
    close = df["close"].astype(float)

    vol_ok = pd.Series(True, index=df.index)
    if params.use_vol_filter:
        vol_ok = df["volume"].astype(float) >= (vol_ma.astype(float) * float(params.vol_factor))

    adx_ok = adxv.astype(float) >= float(params.adx_threshold)
    rsi_ok = (rsiv.astype(float) <= float(params.rsi_upper)) & (rsiv.astype(float) >= float(params.rsi_lower))

    long_intra = close > ema200
    short_intra = close < ema200

    daily_ok = pd.Series(True, index=df.index)
    if params.use_daily_filter and daily_ema_15m is not None:
        long_daily = close > daily_ema_15m
        short_daily = close < daily_ema_15m
        daily_ok = (long_daily & long_intra) | (short_daily & short_intra)

    cand = vol_ok & adx_ok & rsi_ok & (long_intra | short_intra) & daily_ok
    return cand.fillna(False), daily_ok.fillna(True)


def _compute_chart_features(
    ohlcv: pd.DataFrame,
    asof: pd.Timestamp,
    lookback_days: int,
    params: TitanParams,
    st_atr_period: Optional[int] = None,
    st_atr_mult: Optional[float] = None,
) -> Dict[str, float]:
    if ohlcv is None or ohlcv.empty:
        return {}

    t1 = pd.to_datetime(asof, utc=True)
    t0 = t1 - pd.Timedelta(days=int(lookback_days))
    df = ohlcv.loc[(ohlcv.index >= t0) & (ohlcv.index <= t1)].copy()
    if df.empty or len(df) < 300:
        return {}

    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            return {}
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    if len(df) < 300:
        return {}

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    atrv = _atr(high, low, close, period=int(params.atr_period))
    adxv = _adx(high, low, close, period=max(5, int(params.atr_period // 2)))
    rsiv = _rsi(close, period=14)
    ema200 = _ema(close, span=int(params.ema_intraday))
    vol_ma = df["volume"].astype(float).rolling(96, min_periods=20).mean()  # ~1 day of 15m bars

    # daily EMA mapped to 15m
    daily_ema_15m = None
    try:
        d = df[["close"]].resample("1D").last().dropna()
        if len(d) >= int(params.daily_ema) + 5:
            d["daily_ema"] = _ema(d["close"], span=int(params.daily_ema))
            daily_ema_15m = d["daily_ema"].reindex(df.index, method="ffill")
    except Exception:
        daily_ema_15m = None

    # supertrend (SL-aligned)
    st_p = int(st_atr_period) if st_atr_period is not None else max(10, int(params.atr_period // 2))
    st_m = float(st_atr_mult) if st_atr_mult is not None else max(2.0, float(params.atr_multiplier) * 0.6)
    st_val, st_dir = _supertrend(df, atr_period=st_p, atr_mult=st_m)

    # Titan candidate density
    cand_mask, daily_ok_mask = _titan_candidate_mask(df, params, atrv, adxv, rsiv, ema200, vol_ma, daily_ema_15m)
    titan_signal_rate = float(cand_mask.mean()) if len(cand_mask) else 0.0
    cand_int = cand_mask.astype(int)
    titan_signal_count = int(((cand_int.diff() == 1) & (cand_int == 1)).sum())

    denom = int(cand_mask.sum())
    daily_filter_pass_rate = float((daily_ok_mask[cand_mask]).mean()) if denom > 0 else 0.0

    # pivots / retest / sweeps
    pivot_high, pivot_low = _compute_pivots(df, swing_len=int(params.swing_len))
    ms = _retest_and_sweep_metrics(
        df=df,
        atrv=atrv.ffill().bfill().fillna(0.0),
        pivot_high=pivot_high,
        pivot_low=pivot_low,
        tol_atr=float(params.retest_tolerance_atr),
        ctx=int(params.context_lookback),
        follow_n=8,
        follow_atr=1.0,
    )

    # trend efficiency
    te = (df["close"].astype(float) - df["open"].astype(float)).abs()
    te_denom = atrv.replace(0.0, np.nan)
    trend_efficiency_atr = float((te / te_denom).replace([np.inf, -np.inf], np.nan).median(skipna=True))
    if not math.isfinite(trend_efficiency_atr):
        trend_efficiency_atr = 0.0

    adx_above_threshold_ratio = float((adxv >= float(params.adx_threshold)).mean()) if len(adxv) else 0.0

    # directional persistence (EMA200 regime run length mean)
    above = (close > ema200).astype(int)
    run_lengths = []
    if len(above) > 5:
        cur = int(above.iloc[0])
        r = 1
        for v in above.iloc[1:]:
            vv = int(v)
            if vv == cur:
                r += 1
            else:
                run_lengths.append(r)
                cur = vv
                r = 1
        run_lengths.append(r)
    directional_persistence = float(np.mean(run_lengths)) if run_lengths else 0.0

    # supertrend stability
    st_flip_count = int((st_dir.diff().fillna(0) != 0).sum())
    supertrend_flip_rate = float(st_flip_count) / float(max(1, len(st_dir)))

    st_run_lengths = []
    if len(st_dir) > 5:
        cur = float(st_dir.iloc[0])
        r = 1
        for v in st_dir.iloc[1:]:
            vv = float(v)
            if vv == cur:
                r += 1
            else:
                st_run_lengths.append(r)
                cur = vv
                r = 1
        st_run_lengths.append(r)
    st_hold_duration_mean = float(np.mean(st_run_lengths)) if st_run_lengths else 0.0

    # SL hit after tighten (heuristic proxy)
    tighten_events = 0
    hit_after = 0
    N = 8
    stv = st_val.astype(float)
    std = st_dir.astype(float)
    for i in range(2, len(df) - N):
        if std.iloc[i] > 0:
            tightened = stv.iloc[i] > stv.iloc[i - 1]
            if tightened:
                tighten_events += 1
                future_close = close.iloc[i + 1 : i + 1 + N]
                future_st = stv.iloc[i + 1 : i + 1 + N]
                if (future_close < future_st).any():
                    hit_after += 1
        else:
            tightened = stv.iloc[i] < stv.iloc[i - 1]
            if tightened:
                tighten_events += 1
                future_close = close.iloc[i + 1 : i + 1 + N]
                future_st = stv.iloc[i + 1 : i + 1 + N]
                if (future_close > future_st).any():
                    hit_after += 1
    sl_hit_after_tighten_rate = float(hit_after) / float(tighten_events) if tighten_events > 0 else 0.0

    # noise/liquidity proxies
    wick_to_body_ratio = _wick_body_ratio(df)
    range_to_atr_ratio = _range_to_atr_ratio(df, atrv)
    volume_stability = _volume_stability(df)

    # UF score (rule-based initial)
    uf_score = (
        0.30 * titan_signal_rate
        + 0.20 * float(ms.get("retest_success_rate", 0.0))
        + 0.15 * float(trend_efficiency_atr)
        + 0.15 * float(directional_persistence / 100.0)
        - 0.20 * float(supertrend_flip_rate)
        - 0.15 * float(wick_to_body_ratio)
    )
    if not math.isfinite(uf_score):
        uf_score = 0.0

    return {
        # Titan entry 친화도
        "titan_signal_rate": float(titan_signal_rate),
        "titan_signal_count": float(titan_signal_count),
        "daily_filter_pass_rate": float(daily_filter_pass_rate),
        # 구조 이벤트 품질
        "pivot_density": float(ms.get("pivot_density", 0.0)),
        "retest_success_rate": float(ms.get("retest_success_rate", 0.0)),
        "sweep_to_followthrough": float(ms.get("sweep_to_followthrough", 0.0)),
        # 추세 효율
        "adx_above_threshold_ratio": float(adx_above_threshold_ratio),
        "trend_efficiency_atr": float(trend_efficiency_atr),
        "directional_persistence": float(directional_persistence),
        # SuperTrend SL 안정성
        "supertrend_flip_rate": float(supertrend_flip_rate),
        "st_hold_duration_mean": float(st_hold_duration_mean),
        "sl_hit_after_tighten_rate": float(sl_hit_after_tighten_rate),
        # 노이즈/실행 가능성
        "wick_to_body_ratio": float(wick_to_body_ratio),
        "range_to_atr_ratio": float(range_to_atr_ratio),
        "volume_stability": float(volume_stability),
        # 종합
        "uf_score": float(uf_score),
    }


# =============================
# OHLCV Loader (ccxt) + cache
# =============================
def _make_exchange(cfg: dict):
    import ccxt  # local import

    ex_cfg = (cfg or {}).get("ohlcv_exchange", {}) or {}
    default_type = ex_cfg.get("defaultType", "future")
    timeout = int(ex_cfg.get("timeout", 20000))
    enable_rl = bool(ex_cfg.get("enableRateLimit", True))

    exchange = ccxt.binance(
        {
            "enableRateLimit": enable_rl,
            "timeout": timeout,
            "options": {"defaultType": default_type, "adjustForTimeDifference": True},
        }
    )
    try:
        exchange.load_markets()
    except Exception:
        pass
    return exchange


def _get_top_targets_from_exchange(exchange, top_n: int = 100) -> List[str]:
    """
    Binance futures USDT perpetual 기준 quoteVolume 상위 top_n 심볼 리스트 반환.
    """
    try:
        tickers = exchange.fetch_tickers()
    except Exception:
        return []

    markets = getattr(exchange, "markets", None)
    if not isinstance(markets, dict):
        markets = {}

    pairs = []
    for sym, t in (tickers or {}).items():
        if "/USDT:USDT" not in sym:
            continue

        m = markets.get(sym)
        if isinstance(m, dict):
            if m.get("active") is False:
                continue
            info = m.get("info")
            if isinstance(info, dict):
                status = str(info.get("status", "")).upper()
                if status and status not in ("TRADING", "1"):
                    continue

        vol = 0.0
        if isinstance(t, dict):
            vol = t.get("quoteVolume", 0) or 0
        try:
            vol = float(vol)
        except Exception:
            vol = 0.0

        if vol > 0:
            pairs.append((sym, vol))

    pairs.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in pairs[: int(top_n)]]

def _collect_blacklist(cfg: dict, uf_config_path: str) -> set[str]:
    """
    UF config + (optional) phalanx root config.json blacklist를 합쳐서 반환
    - uf_config.json: cfg["strategy_settings"]["blacklist"]
    - root config.json: (cfg["root_config_path"] or "config.json") 의 strategy_settings.blacklist
    """
    bl = set()

    def _norm(x: str) -> str:
        return str(x).strip().upper()

    # 1) uf_config.json 내부 blacklist
    try:
        uf_bl = (((cfg or {}).get("strategy_settings", {}) or {}).get("blacklist", []) or [])
        bl |= {_norm(x) for x in uf_bl if str(x).strip()}
    except Exception:
        pass

    # 2) root config.json blacklist (기본값: 프로젝트 루트의 config.json)
    try:
        # uf_config.json 기준 상대경로 허용
        uf_dir = os.path.dirname(os.path.abspath(uf_config_path))
        root_cfg_path = (cfg or {}).get("root_config_path", "config.json")
        if not os.path.isabs(root_cfg_path):
            # uf_config 기준이 아니라 "프로젝트 루트" 기준으로 두고 싶으면 아래처럼 변경 가능:
            # project_root = os.path.abspath(os.path.join(uf_dir, "..", ".."))
            # root_cfg_path = os.path.join(project_root, root_cfg_path)
            root_cfg_path = os.path.abspath(os.path.join(uf_dir, "..", "..", root_cfg_path))

        if os.path.exists(root_cfg_path):
            with open(root_cfg_path, "r", encoding="utf-8") as f:
                root_cfg = json.load(f) or {}
            root_bl = (((root_cfg or {}).get("strategy_settings", {}) or {}).get("blacklist", []) or [])
            bl |= {_norm(x) for x in root_bl if str(x).strip()}
    except Exception:
        pass

    return bl


def _cache_path(cfg: dict, symbol: str, timeframe: str) -> str:
    cache_dir = (cfg or {}).get("ohlcv_cache_dir", "universe_filter/output/cache/ohlcv")
    _ensure_dir(cache_dir)
    safe = symbol.replace("/", "_").replace(":", "_")
    return os.path.join(cache_dir, f"{safe}_{timeframe}.parquet")


def _load_cached_ohlcv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        if df is None or df.empty:
            return None
        if "timestamp" in df.columns and "datetime" not in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
        for c in ["open", "high", "low", "close", "volume"]:
            if c not in df.columns:
                return None
        return df
    except Exception:
        return None


def _save_cached_ohlcv(path: str, df: pd.DataFrame) -> None:
    try:
        out = df.copy()

        if "datetime" in out.columns:
            out["datetime"] = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
            out = out.dropna(subset=["datetime"]).sort_values("datetime")
        else:
            idx_name = out.index.name or "datetime"
            out = out.reset_index().rename(columns={idx_name: "datetime"})
            out["datetime"] = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
            out = out.dropna(subset=["datetime"]).sort_values("datetime")

        req = ["open", "high", "low", "close", "volume"]
        for c in req:
            if c not in out.columns:
                return

        out["timestamp"] = (out["datetime"].astype("int64") // 10**6).astype("int64")
        out = out.drop_duplicates(subset=["datetime"], keep="last").sort_values("datetime")

        _ensure_dir(os.path.dirname(path) or ".")
        out.to_parquet(path, index=False)
    except Exception:
        pass


def _download_ohlcv(
    exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    limit: int = 1000,
    max_batches: int = 50,
) -> pd.DataFrame:
    all_rows = []
    cur = since_ms
    batches = 0
    _log(f"    ⬇️  fetch_ohlcv start | {symbol} tf={timeframe} since={pd.to_datetime(since_ms, unit='ms', utc=True)}")
    while batches < max_batches:
        batches += 1
        try:
            rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cur, limit=limit)
        except Exception as e:
            _log(f"    ❌  fetch_ohlcv error | {symbol} batch={batches} error={e}")
            break
        if not rows:
            _log(f"    ⚠️  fetch_ohlcv empty | {symbol} batch={batches}")
            break
        all_rows.extend(rows)
        last_ts = rows[-1][0]
        _log(f"    ✅ batch ok | {symbol} batch={batches} rows={len(rows)} last={pd.to_datetime(last_ts, unit='ms', utc=True)}")
        if last_ts <= cur:
            _log(f"    ⚠️  stop (non-increasing ts) | {symbol} last_ts<=cur")
            break
        cur = last_ts + 1
        if len(rows) < limit:
            _log(f"    ⏹️  stop (final batch < limit) | {symbol} rows={len(rows)}")
            break

    if not all_rows:
        _log(f"    🚫 no rows | {symbol}")
        return pd.DataFrame()

    _log(f"    📦 fetched total | {symbol} total_rows={len(all_rows)}")
    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df[~df.index.duplicated(keep="last")]
    return df


def _get_ohlcv_for_symbol(
    exchange,
    cfg: dict,
    symbol: str,
    timeframe: str,
    asof: pd.Timestamp,
    lookback_days: int,
) -> pd.DataFrame:
    path = _cache_path(cfg, symbol, timeframe)
    cached = _load_cached_ohlcv(path)

    t1 = pd.to_datetime(asof, utc=True)
    t0 = t1 - pd.Timedelta(days=int(lookback_days) + 5)
    since_ms = int(t0.timestamp() * 1000)

    if cached is not None and not cached.empty:
        _log(f"  ♻️ cache hit | {symbol} tf={timeframe} rows={len(cached)} max={cached.index.max()}")
        if cached.index.max() >= t1 - pd.Timedelta(minutes=15):
            _log(f"  ✅ cache covers asof | {symbol}")
            return cached.loc[cached.index <= t1]

        _log(f"  ➕ cache extend | {symbol} from={cached.index.max()}")
        try:
            last_ms = int((cached.index.max().timestamp() * 1000) + 1)
        except Exception:
            last_ms = since_ms

        new_df = _download_ohlcv(exchange, symbol, timeframe, since_ms=last_ms)
        if not new_df.empty:
            merged = pd.concat([cached, new_df], axis=0)
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            _save_cached_ohlcv(path, merged)
            _log(f"  ✅ cache extended | {symbol} merged_rows={len(merged)}")
            return merged.loc[merged.index <= t1]

        _log(f"  ⚠️ extend got empty | {symbol} -> using cache only")
        return cached.loc[cached.index <= t1]

    _log(f"  🆕 cache miss | {symbol} tf={timeframe} -> download")
    df = _download_ohlcv(exchange, symbol, timeframe, since_ms=since_ms)
    if not df.empty:
        _save_cached_ohlcv(path, df)
        _log(f"  ✅ cached saved | {symbol} rows={len(df)}")
    return df.loc[df.index <= t1] if not df.empty else df


# =============================
# Ranking
# =============================
def _rank_symbols_by_chart_features(
    exchange,
    cfg: dict,
    symbols: List[str],
    asof: pd.Timestamp,
    timeframe: str,
    lookback_days: int,
    titan_params: TitanParams,
    st_atr_period: Optional[int],
    st_atr_mult: Optional[float],
    score_field: str = "uf_score",
) -> pd.DataFrame:
    rows = []
    for i, sym in enumerate(symbols or [], start=1):
        try:
            _log(f"🔧 [{i}/{len(symbols)}] {sym}")
            ohlcv = _get_ohlcv_for_symbol(
                exchange=exchange,
                cfg=cfg,
                symbol=sym,
                timeframe=timeframe,
                asof=asof,
                lookback_days=lookback_days,
            )
            cf = _compute_chart_features(
                ohlcv=ohlcv,
                asof=asof,
                lookback_days=lookback_days,
                params=titan_params,
                st_atr_period=st_atr_period,
                st_atr_mult=st_atr_mult,
            )
            if not cf:
                continue
            cf["symbol"] = sym
            rows.append(cf)
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if score_field not in df.columns:
        df[score_field] = np.nan

    df[score_field] = pd.to_numeric(df[score_field], errors="coerce")
    df = df.dropna(subset=[score_field])
    df = df.sort_values(score_field, ascending=False).reset_index(drop=True)
    return df


# =============================
# Main
# =============================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="universe_filter/config/uf_config.json")

    ap.add_argument("--top_n", type=int, default=100)
    ap.add_argument("--top_k", type=int, default=30)

    ap.add_argument("--timeframe", default=None)       # override config ohlcv_timeframe
    ap.add_argument("--lookback_days", type=int, default=None)  # override config ohlcv_lookback_days

    ap.add_argument("--asof", default=None)  # ISO string; default now(UTC)

    ap.add_argument("--decision_out", default="universe_filter/output/decision_table.csv")
    ap.add_argument("--universe_step2_out", default="universe_filter/output/universe_step2.json")

    ap.add_argument("--score_field", default="uf_score")

    args = ap.parse_args()

    cfg = _read_json(args.config)

    timeframe = str(args.timeframe or (cfg.get("ohlcv_timeframe", "15m")))
    lookback_days = int(args.lookback_days if args.lookback_days is not None else int(cfg.get("ohlcv_lookback_days", 90)))

    # optional: supertrend override
    st_cfg = (cfg or {}).get("supertrend_params", {}) or {}
    st_atr_period = st_cfg.get("atr_period", None)
    st_atr_mult = st_cfg.get("atr_mult", None)

    titan_params = _load_titan_params(cfg)

    # asof
    if args.asof:
        asof = pd.to_datetime(args.asof, utc=True, errors="coerce")
        if pd.isna(asof):
            asof = pd.Timestamp.now(tz="UTC")
    else:
        asof = pd.Timestamp.now(tz="UTC")

    # init exchange
    try:
        exchange = _make_exchange(cfg)
    except Exception as e:
        print(f"❌ Step02 failed: cannot init exchange: {e}")
        return

    # get TopN symbols
    base_syms = _get_top_targets_from_exchange(exchange, top_n=int(args.top_n))
    if not base_syms:
        print("❌ Step02 failed: no symbols fetched from exchange.")
        return

    # -----------------------------
    # APPLY BLACKLIST (UF + ROOT CONFIG)
    # -----------------------------
    bl = _collect_blacklist(cfg, args.config)

    def _is_blacklisted(sym: str) -> bool:
        s = sym.upper().strip()
        base = s.split(":")[0]              # BTC/USDT:USDT -> BTC/USDT
        return (s in bl) or (base in bl)

    before_n = len(base_syms)
    base_syms = [s for s in base_syms if not _is_blacklisted(s)]
    after_n = len(base_syms)

    print(f"🚫 blacklist applied | before={before_n} after={after_n} removed={before_n-after_n}")

    # rank
    ranked = _rank_symbols_by_chart_features(
        exchange=exchange,
        cfg=cfg,
        symbols=base_syms,
        asof=asof,
        timeframe=timeframe,
        lookback_days=lookback_days,
        titan_params=titan_params,
        st_atr_period=st_atr_period,
        st_atr_mult=st_atr_mult,
        score_field=str(args.score_field),
    )

    if ranked.empty:
        print("❌ Step02 failed: ranking empty (no symbols produced chart features).")
        return

    selected = ranked["symbol"].head(int(args.top_k)).astype(str).tolist()

    # outputs
    _ensure_dir(os.path.dirname(args.decision_out) or ".")
    ranked.to_csv(args.decision_out, index=False, encoding="utf-8-sig")

    step2_obj = {
        "asof": str(pd.to_datetime(asof, utc=True).isoformat()),
        "rebalance_rule": "N/A (Step2 ranking only)",
        "top_n": int(args.top_n),
        "top_k": int(args.top_k),
        "timeframe": str(timeframe),
        "lookback_days": int(lookback_days),
        "score_field": str(args.score_field),
        "symbols": selected,
        "meta": {
            "notes": "Step02 UF 2.0 simplified: TopN(volume) -> chart_features -> rank -> TopK",
            "titan_params": {
                "atr_period": titan_params.atr_period,
                "atr_multiplier": titan_params.atr_multiplier,
                "adx_threshold": titan_params.adx_threshold,
                "rsi_upper": titan_params.rsi_upper,
                "rsi_lower": titan_params.rsi_lower,
                "vol_factor": titan_params.vol_factor,
                "ema_intraday": titan_params.ema_intraday,
                "daily_ema": titan_params.daily_ema,
                "swing_len": titan_params.swing_len,
                "context_lookback": titan_params.context_lookback,
                "retest_tolerance_atr": titan_params.retest_tolerance_atr,
                "use_daily_filter": titan_params.use_daily_filter,
                "use_vol_filter": titan_params.use_vol_filter,
            },
            "supertrend_params": {"atr_period": st_atr_period, "atr_mult": st_atr_mult},
        },
    }
    _write_json(args.universe_step2_out, step2_obj)

    print(f"✅ Step02 done.")
    print(f"  ranked_rows={len(ranked)}")
    print(f"  decision_table={args.decision_out}")
    print(f"  universe_step2={args.universe_step2_out}")
    print(f"  selected_topk={len(selected)}")


if __name__ == "__main__":
    main()
