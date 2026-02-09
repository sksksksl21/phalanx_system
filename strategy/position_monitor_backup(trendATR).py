import logging
import math
from typing import Optional, Tuple, Any

import pandas as pd

logger = logging.getLogger("PhalanxMonitor")


class PositionMonitor:
    """
    [Phalanx Strategy Module]
    Role: Exit Logic Authority (Shared by Live & Backtest)

    SL 전략을 config로 선택 가능하게 확장:
      - supertrend : 외부 st_val 기반 SL (방향성 필터 포함)
      - atr_trail  : ATR 기반 트레일링 (가격 기준)
      - profit_lock: 일정 이익(ATR) 도달 후 SL을 이익구간으로 끌어올려 잠금
      - armor      : Structure + VolTrail(stateful) + OutcomeInsurance (Regime via ADX optional)
      - hybrid     : supertrend + atr_trail 중 "더 타이트" 선택

    반환 규격(기존 유지):
      (action, exec_price, reason, new_sl)
        - action: "UPDATE_SL" | "EXIT" | None
        - exec_price: EXIT일 때 청산가(=SL)
        - reason: "STOP_LOSS" | "TRAILING" | "TRAILING_ATR" | "PROFIT_LOCK" | "ARMOR" | "HYBRID" | None
        - new_sl: 계산된 신규 SL (None 가능)

    NOTE (중요):
      - Armor는 market_data["df"] 또는 market_data["history"](DataFrame)가 없으면 동작하지 않는다(기본).
      - Armor의 stateful trail은 position["trail_sl"]에 저장된다.
        (단일 진실원 저장/복원은 엔진/상태 계층에서 책임져야 한다)
    """

    def __init__(self):
        pass

    # -----------------------------
    # Public API
    # -----------------------------
    def check_conditions(
        self,
        symbol: str,
        position: dict,
        market_data: dict,
        sl_apply_mode: str = "next",
        sl_strategy: str = "supertrend",
        sl_params: Optional[dict] = None,
    ) -> Tuple[Optional[str], float, Optional[str], Optional[float]]:
        """
        Returns: (action, exec_price, reason, new_sl)
        """
        params = sl_params or {}

        # -----------------------------
        # 0) normalize inputs
        # -----------------------------
        side = str(position.get("side", "")).upper().strip()
        if side not in ("LONG", "SHORT"):
            return None, 0.0, None, _safe_float(position.get("sl", None))

        curr_price = _safe_float(market_data.get("close", 0.0), default=0.0) or 0.0
        high_price = _safe_float(market_data.get("high", curr_price), default=curr_price) or curr_price
        low_price = _safe_float(market_data.get("low", curr_price), default=curr_price) or curr_price

        mode = str(sl_apply_mode or "next").strip().lower()
        if mode not in ("next", "same"):
            mode = "next"

        strat = str(sl_strategy or "supertrend").strip().lower()

        current_sl = _safe_float(position.get("sl", None))
        entry_price = _safe_float(position.get("entry_price", None))

        st_val = _safe_float(market_data.get("st_val", None))
        if st_val is not None and st_val <= 0:
            st_val = None

        atr = _safe_float(market_data.get("atr", None))
        if atr is not None and atr <= 0:
            atr = None

        action: Optional[str] = None
        exec_price: float = 0.0
        reason: Optional[str] = None
        new_sl: Optional[float] = current_sl

        # -----------------------------
        # 1) Helpers
        # -----------------------------
        def _tighten_sl(candidate: Optional[float], cur: Optional[float]) -> Optional[float]:
            """
            SL은 손실을 키우는 방향으로 이동하면 안 됨(=tighten만 허용)
              - LONG : candidate > cur 일 때만 허용
              - SHORT: candidate < cur 일 때만 허용
            """
            if candidate is None:
                return None
            c = _safe_float(candidate)
            if c is None:
                return None

            if cur is None:
                return c

            cur_f = _safe_float(cur)
            if cur_f is None:
                return c

            if side == "LONG":
                return c if c > cur_f else None
            else:
                return c if c < cur_f else None

        def _apply_step_constraints(tightened_sl: Optional[float], cur_sl: Optional[float]) -> Optional[float]:
            """
            tighten 통과 값에 대해서만 min_move_atr / max_step_atr 적용.
            """
            if tightened_sl is None:
                return None

            # ATR 없으면 제약 계산 불가 -> tighten 결과 그대로
            if atr is None or atr <= 0:
                return _safe_float(tightened_sl)

            t = _safe_float(tightened_sl)
            if t is None:
                return None

            # 최초 세팅이면 제약 없이 허용
            if cur_sl is None:
                return t

            cur = _safe_float(cur_sl)
            if cur is None:
                return t

            min_move = _safe_float(params.get("min_move_atr", 0.0), default=0.0) or 0.0
            max_step = _safe_float(params.get("max_step_atr", 0.0), default=0.0) or 0.0
            if min_move < 0:
                min_move = 0.0
            if max_step < 0:
                max_step = 0.0

            if side == "LONG":
                desired = max(cur, t)
                if max_step > 0:
                    desired = min(desired, cur + (max_step * atr))
                if min_move > 0 and (desired - cur) < (min_move * atr):
                    return cur
                return desired
            else:
                desired = min(cur, t)
                if max_step > 0:
                    desired = max(desired, cur - (max_step * atr))
                if min_move > 0 and (cur - desired) < (min_move * atr):
                    return cur
                return desired

        def _maybe_apply_constraints(cand_tight: Optional[float], cur_sl: Optional[float]) -> Optional[float]:
            """
            기본은 Armor만 step constraint 적용.
            params["apply_step_all"]=True면 다른 전략도 동일 적용 가능.
            """
            apply_all = bool(params.get("apply_step_all", False))
            if apply_all:
                return _apply_step_constraints(cand_tight, cur_sl)
            return cand_tight

        # -----------------------------
        # 2) Candidate builders
        # -----------------------------
        def _candidate_supertrend() -> Optional[float]:
            """
            방향성 필터 포함:
              - LONG  : st_val < price 인 경우만 후보
              - SHORT : st_val > price 인 경우만 후보
            """
            if st_val is None:
                return None
            if side == "LONG":
                return st_val if st_val < curr_price else None
            else:
                return st_val if st_val > curr_price else None

        def _candidate_atr_trail() -> Optional[float]:
            if atr is None:
                return None
            m = _safe_float(params.get("atr_mult", 3.0), default=3.0) or 3.0
            if m <= 0:
                m = 3.0
            return (curr_price - (m * atr)) if side == "LONG" else (curr_price + (m * atr))

        def _candidate_profit_lock() -> Optional[float]:
            """
            이익이 trigger_atr*ATR 이상이면 SL을 entry +/- lock_atr*ATR로 당겨 잠금
            """
            if atr is None or entry_price is None or entry_price <= 0:
                return None

            trigger = _safe_float(params.get("trigger_atr", 2.0), default=2.0) or 2.0
            lock = _safe_float(params.get("lock_atr", 0.5), default=0.5) or 0.5
            if trigger <= 0:
                trigger = 2.0
            if lock < 0:
                lock = 0.0

            if side == "LONG":
                if curr_price >= entry_price + (trigger * atr):
                    return entry_price + (lock * atr)
                return None
            else:
                if curr_price <= entry_price - (trigger * atr):
                    return entry_price - (lock * atr)
                return None

        def _get_history_df() -> Optional[pd.DataFrame]:
            """
            ARMOR 계산용 과거 OHLCV DataFrame을 market_data에서 꺼낸다.
            허용 키 예: market_data["df"] (권장), market_data["history"]
            최소 컬럼: high, low, close
            """
            df = market_data.get("df", None)
            if df is None:
                df = market_data.get("history", None)

            if df is None or not isinstance(df, pd.DataFrame):
                return None

            for c in ("high", "low", "close"):
                if c not in df.columns:
                    return None

            try:
                dfx = df.copy()
                # float dtype 강제 (pd.NA/object 혼입 방지)
                dfx["high"] = pd.to_numeric(dfx["high"], errors="coerce")
                dfx["low"] = pd.to_numeric(dfx["low"], errors="coerce")
                dfx["close"] = pd.to_numeric(dfx["close"], errors="coerce")
                dfx = dfx.dropna(subset=["high", "low", "close"])
                if len(dfx) < 5:
                    return None
                return dfx
            except Exception:
                return None

        def _wilder_atr_from_df(df: pd.DataFrame, period: int) -> Optional[float]:
            try:
                period = int(period)
                if period <= 1:
                    period = 14

                high = df["high"].astype(float)
                low = df["low"].astype(float)
                close = df["close"].astype(float)

                prev_close = close.shift(1)
                tr1 = (high - low).abs()
                tr2 = (high - prev_close).abs()
                tr3 = (low - prev_close).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

                atr_s = tr.ewm(alpha=1.0 / float(period), adjust=False).mean()
                v = _safe_float(atr_s.iloc[-1])
                return v if (v is not None and v > 0) else None
            except Exception:
                return None

        def _adx_from_df(df: pd.DataFrame, period: int) -> Optional[float]:
            """
            간단 ADX (Wilder smoothing 기반). 내부에서 ATR 스칼라 대신 TR 시계열을 쓰지 않고
            "스칼라 ATR"로 나누는 단순화 버전. (레짐 판정용으로 충분)
            """
            try:
                period = int(period)
                if period <= 1:
                    period = 14

                high = df["high"].astype(float)
                low = df["low"].astype(float)

                up_move = high.diff()
                down_move = -low.diff()

                plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
                minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

                atr_v = _wilder_atr_from_df(df, period)
                if atr_v is None or atr_v <= 0:
                    return None

                plus_di = 100.0 * (plus_dm.ewm(alpha=1.0 / float(period), adjust=False).mean() / atr_v)
                minus_di = 100.0 * (minus_dm.ewm(alpha=1.0 / float(period), adjust=False).mean() / atr_v)

                denom = (plus_di + minus_di).abs().replace(0.0, float("nan"))
                dx = (100.0 * (plus_di - minus_di).abs() / denom).replace(
                    [float("inf"), -float("inf")], float("nan")
                )
                adx_s = dx.ewm(alpha=1.0 / float(period), adjust=False).mean()

                v = _safe_float(adx_s.iloc[-1])
                return v if (v is not None and math.isfinite(v)) else None
            except Exception:
                return None

        def _recent_swing_levels(df: pd.DataFrame, swing_len: int) -> Tuple[Optional[float], Optional[float]]:
            n = int(max(2, swing_len))
            w = df.tail(n)
            try:
                swing_low = _safe_float(w["low"].min())
                swing_high = _safe_float(w["high"].max())
                return swing_low, swing_high
            except Exception:
                return None, None

        def _candidate_armor() -> Optional[float]:
            """
            ARMOR SL = max/min(Structure, VolTrail, OutcomeInsurance)

            - L2 VolTrail은 position['trail_sl']로 stateful 유지
            - step/min_move 제약은 여기서 적용하지 않음 (tighten 후 1회)
            """
            df = _get_history_df()
            if df is None:
                # 기본은 "조용히 미동작" (정합성/계약상 df 없는 Armor는 가짜로 돌리면 위험)
                # 필요하면 params["armor_fallback"]="atr_trail"로 완화 가능
                fallback = str(params.get("armor_fallback", "none")).strip().lower()
                if fallback == "atr_trail":
                    return _candidate_atr_trail()
                return None

            swing_len = int(params.get("swing_len", 5))
            adx_period = int(params.get("adx_period", 14))
            adx_trend = _safe_float(params.get("adx_trend", 22.0), default=22.0) or 22.0

            atr_mult_trend = _safe_float(params.get("atr_mult_trend", 4.0), default=4.0) or 4.0
            atr_mult_chop = _safe_float(params.get("atr_mult_chop", 2.0), default=2.0) or 2.0

            profit_trigger = _safe_float(params.get("profit_trigger_atr", 1.2), default=1.2) or 1.2
            profit_lock = _safe_float(params.get("profit_lock_atr", 0.2), default=0.2) or 0.2

            structure_buffer_atr = _safe_float(params.get("structure_buffer_atr", 0.3), default=0.3) or 0.3
            fee_buffer_bps = _safe_float(params.get("fee_buffer_bps", 0.0), default=0.0) or 0.0

            local_atr = atr
            if local_atr is None:
                local_atr = _wilder_atr_from_df(df, int(params.get("atr_period", 14)))
            if local_atr is None or local_atr <= 0:
                return None

            if entry_price is None or entry_price <= 0:
                return None

            # Regime via ADX (market_data 우선, 없으면 df 계산)
            adx_v = _safe_float(market_data.get("adx", None))
            if adx_v is None:
                adx_v = _adx_from_df(df, adx_period)

            regime = "TREND" if (adx_v is not None and adx_v >= adx_trend) else "CHOP"
            m = atr_mult_trend if regime == "TREND" else atr_mult_chop
            if m is None or m <= 0:
                m = 2.0

            # L1 Structure
            swing_low, swing_high = _recent_swing_levels(df, swing_len)
            if swing_low is None or swing_high is None:
                return None

            if side == "LONG":
                structure_sl = float(swing_low) - (structure_buffer_atr * local_atr)
            else:
                structure_sl = float(swing_high) + (structure_buffer_atr * local_atr)

            # L2 VolTrail (STATEFUL)
            trail_sl = _safe_float(position.get("trail_sl", None))
            if side == "LONG":
                vol_candidate = curr_price - (m * local_atr)
                new_trail = vol_candidate if trail_sl is None else max(trail_sl, vol_candidate)
            else:
                vol_candidate = curr_price + (m * local_atr)
                new_trail = vol_candidate if trail_sl is None else min(trail_sl, vol_candidate)

            # state update (엔진이 position dict를 상태에 저장/복원해야 완전 정합)
            position["trail_sl"] = float(new_trail)
            vol_sl = float(new_trail)

            # L3 OutcomeInsurance
            fee_pad = float(entry_price) * (fee_buffer_bps * 0.0001)

            if side == "LONG":
                if curr_price >= entry_price + (profit_trigger * local_atr):
                    outcome_sl = float(entry_price) + fee_pad + (profit_lock * local_atr)
                else:
                    outcome_sl = None
            else:
                if curr_price <= entry_price - (profit_trigger * local_atr):
                    outcome_sl = float(entry_price) - fee_pad - (profit_lock * local_atr)
                else:
                    outcome_sl = None

            # Final (tightest)
            if side == "LONG":
                raw = max(
                    float(structure_sl),
                    float(vol_sl),
                    float(outcome_sl) if outcome_sl is not None else -float("inf"),
                )
            else:
                raw = min(
                    float(structure_sl),
                    float(vol_sl),
                    float(outcome_sl) if outcome_sl is not None else float("inf"),
                )

            return float(raw)

        # -----------------------------
        # 3) Choose strategy + compute new_sl
        # -----------------------------
        cand: Optional[float] = None

        if strat == "supertrend":
            cand = _tighten_sl(_candidate_supertrend(), current_sl)
            cand = _maybe_apply_constraints(cand, current_sl)
            if cand is not None:
                new_sl = float(cand)
                action = "UPDATE_SL"
                reason = "TRAILING"

        elif strat == "atr_trail":
            cand = _tighten_sl(_candidate_atr_trail(), current_sl)
            cand = _maybe_apply_constraints(cand, current_sl)
            if cand is not None:
                new_sl = float(cand)
                action = "UPDATE_SL"
                reason = "TRAILING_ATR"

        elif strat == "profit_lock":
            cand = _tighten_sl(_candidate_profit_lock(), current_sl)
            cand = _maybe_apply_constraints(cand, current_sl)
            if cand is not None:
                new_sl = float(cand)
                action = "UPDATE_SL"
                reason = "PROFIT_LOCK"

        elif strat == "armor":
            raw = _candidate_armor()
            tight = _tighten_sl(raw, current_sl)
            cand = _apply_step_constraints(tight, current_sl)  # armor는 기본으로 step 적용
            if cand is not None and (current_sl is None or _safe_float(cand) != _safe_float(current_sl)):
                new_sl = float(cand)
                action = "UPDATE_SL"
                reason = "ARMOR"

        elif strat == "hybrid":
            # 둘 다 있으면 LONG은 더 큰 SL(더 타이트), SHORT는 더 작은 SL(더 타이트) 선택
            t1 = _tighten_sl(_candidate_supertrend(), current_sl)
            t2 = _tighten_sl(_candidate_atr_trail(), current_sl)

            chosen = None
            if t1 is not None and t2 is not None:
                chosen = max(t1, t2) if side == "LONG" else min(t1, t2)
            elif t1 is not None:
                chosen = t1
            elif t2 is not None:
                chosen = t2

            chosen = _maybe_apply_constraints(chosen, current_sl)
            if chosen is not None:
                new_sl = float(chosen)
                action = "UPDATE_SL"
                reason = "HYBRID"

        else:
            # 알 수 없는 전략이면 안전하게 업데이트 안 함
            pass

        # -----------------------------
        # 4) StopLoss Hit Check (same/next)
        # -----------------------------
        effective_sl_for_hit = current_sl
        if mode == "same" and action == "UPDATE_SL":
            effective_sl_for_hit = new_sl

        sl_hit = False
        if effective_sl_for_hit is not None:
            slv = float(effective_sl_for_hit)
            if side == "LONG" and low_price <= slv:
                sl_hit = True
            elif side == "SHORT" and high_price >= slv:
                sl_hit = True

        if sl_hit:
            return "EXIT", float(effective_sl_for_hit), "STOP_LOSS", new_sl

        # -----------------------------
        # 5) Return
        # -----------------------------
        if action == "UPDATE_SL":
            return "UPDATE_SL", 0.0, reason, new_sl

        return None, 0.0, None, current_sl


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    """
    Robust float casting. Returns default on failure.
    """
    if x is None:
        return default
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default
