import logging
import math
import pandas as pd

logger = logging.getLogger("PhalanxMonitor")


class PositionMonitor:
    """
    [Phalanx Strategy Module]
    Role: Exit Logic Authority (Shared by Live & Backtest)

    SL 전략을 config로 선택 가능하게 확장:
      - supertrend : 기존 로직 그대로
      - atr_trail  : ATR 기반 트레일링 (가격 기준)
      - profit_lock: 일정 이익(ATR) 도달 후 SL을 이익구간으로 끌어올려 잠금
      - hybrid     : supertrend + atr_trail 중 "더 보수적(더 타이트)"한 SL 선택

    반환 규격(기존 유지):
      (action, exec_price, reason, new_sl)
        - action: "UPDATE_SL" | "EXIT" | None
        - exec_price: EXIT일 때 청산가(=SL)
        - reason: "STOP_LOSS" | "TRAILING" | "PROFIT_LOCK" | "HYBRID" | None
        - new_sl: 계산된 신규 SL (None 가능)
    """

    def __init__(self):
        pass

    def check_conditions(
        self,
        symbol,
        position,
        market_data,
        sl_apply_mode: str = "next",
        sl_strategy: str = "supertrend",
        sl_params: dict | None = None,
    ):
        # -----------------------------
        # 0) normalize inputs
        # -----------------------------
        params = sl_params or {}
        #logger.info(f"[SL_DEBUG] strat={sl_strategy} params_keys={list(params.keys())}")

        side = str(position.get("side", "")).upper().strip()
        if side not in ("LONG", "SHORT"):
            return None, 0.0, None, position.get("sl", None)

        # prices
        curr_price = float(market_data.get("close", 0) or 0)
        high_price = float(market_data.get("high", curr_price) or curr_price)
        low_price  = float(market_data.get("low", curr_price) or curr_price)

        # mode normalize
        try:
            mode = str(sl_apply_mode or "next").strip().lower()
        except Exception:
            mode = "next"
        if mode not in ("next", "same"):
            mode = "next"

        # strategy normalize
        try:
            strat = str(sl_strategy or "supertrend").strip().lower()
        except Exception:
            strat = "supertrend"

        # current_sl normalize (None-safe)
        cur_sl_raw = position.get("sl", None)
        try:
            current_sl = float(cur_sl_raw) if cur_sl_raw is not None else None
        except Exception:
            current_sl = None

        # entry_price (profit_lock에 필요)
        entry_raw = position.get("entry_price", None)
        try:
            entry_price = float(entry_raw) if entry_raw is not None else None
        except Exception:
            entry_price = None

        # st_val normalize
        st_val_raw = market_data.get("st_val", None)
        try:
            st_val = float(st_val_raw) if st_val_raw is not None else None
        except Exception:
            st_val = None
        if st_val is not None and st_val <= 0:
            st_val = None

        # atr normalize
        atr_raw = market_data.get("atr", None)
        try:
            atr = float(atr_raw) if atr_raw is not None else None
        except Exception:
            atr = None
        if atr is not None and atr <= 0:
            atr = None

        # 결과 변수
        action = None
        exec_price = 0.0
        reason = None
        new_sl = current_sl  # None 가능

        # -----------------------------
        # 1) SL 후보 계산(전략별)
        # -----------------------------
        def _get_history_df():
            """
            ARMOR 계산용 과거 OHLCV DataFrame을 market_data에서 꺼낸다.
            허용 키 예:
              - market_data["df"]  (권장)
              - market_data["history"]
            컬럼 표준: ['timestamp','open','high','low','close','volume']
            """
            df = market_data.get("df", None)
            if df is None:
                df = market_data.get("history", None)

            if df is None:
                return None

            if not isinstance(df, pd.DataFrame):
                return None

            # 필요한 컬럼 체크
            req = ("high", "low", "close")
            for c in req:
                if c not in df.columns:
                    return None

            # 숫자형 변환 시도(실패해도 None 리턴하지 않고 최대한 진행)
            try:
                dfx = df.copy()
                dfx["high"] = pd.to_numeric(dfx["high"], errors="coerce")
                dfx["low"] = pd.to_numeric(dfx["low"], errors="coerce")
                dfx["close"] = pd.to_numeric(dfx["close"], errors="coerce")
                dfx = dfx.dropna(subset=["high", "low", "close"])
                if len(dfx) < 5:
                    return None
                return dfx
            except Exception:
                return None
        
        def _wilder_atr_from_df(df: pd.DataFrame, period: int) -> float | None:
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

                atr = tr.ewm(alpha=1.0 / float(period), adjust=False).mean()
                v = float(atr.iloc[-1])
                if v > 0:
                    return v
                return None
            except Exception:
                return None

        def _adx_from_df(df: pd.DataFrame, period: int) -> float | None:
            """
            간단 ADX (Wilder smoothing 기반)
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

                atr = _wilder_atr_from_df(df, period)
                if atr is None or atr <= 0:
                    return None

                # Wilder smoothing
                plus_di = 100.0 * (plus_dm.ewm(alpha=1.0 / float(period), adjust=False).mean() / atr)
                minus_di = 100.0 * (minus_dm.ewm(alpha=1.0 / float(period), adjust=False).mean() / atr)

                denom = (plus_di + minus_di).abs().replace(0.0, float("nan"))
                dx = (100.0 * (plus_di - minus_di).abs() / denom).replace([float("inf"), -float("inf")], float("nan"))
                adx = dx.ewm(alpha=1.0 / float(period), adjust=False).mean()

                v = float(adx.iloc[-1])
                if math.isfinite(v):
                    return v
                return None
            except Exception:
                return None

        def _recent_swing_levels(df: pd.DataFrame, swing_len: int):
            n = int(max(2, swing_len))
            w = df.tail(n)
            try:
                swing_low = float(w["low"].min())
                swing_high = float(w["high"].max())
                return swing_low, swing_high
            except Exception:
                return None, None

        def _apply_step_constraints(tightened_sl: float | None, cur_sl: float | None):
            """
            tighten(방향성: LONG 위로만 / SHORT 아래로만) 을 통과한 값에 대해서만
            min_move_atr / max_step_atr 제약을 적용한다.

            즉, 호출 순서는 반드시:
              candidate -> _tighten_sl(...) -> _apply_step_constraints(...)
            """
            if tightened_sl is None:
                return None

            # atr 없으면 제약 계산 불가 -> tighten 결과 그대로
            if atr is None or atr <= 0:
                try:
                    return float(tightened_sl)
                except Exception:
                    return None

            try:
                t = float(tightened_sl)
            except Exception:
                return None

            # 최초 세팅이면 제약 없이 허용(=tighten 통과 값)
            if cur_sl is None:
                return t

            try:
                cur = float(cur_sl)
            except Exception:
                return t

            min_move = float(params.get("min_move_atr", 0.0))
            max_step = float(params.get("max_step_atr", 0.0))
            if min_move < 0:
                min_move = 0.0
            if max_step < 0:
                max_step = 0.0

            if side == "LONG":
                # tighten이 이미 통과했으니 t >= cur 가정(그래도 안전하게 max)
                desired = max(cur, t)

                if max_step > 0:
                    desired = min(desired, cur + (max_step * atr))

                if min_move > 0 and (desired - cur) < (min_move * atr):
                    return cur

                return desired
            else:
                # tighten이 이미 통과했으니 t <= cur 가정(그래도 안전하게 min)
                desired = min(cur, t)

                if max_step > 0:
                    desired = max(desired, cur - (max_step * atr))

                if min_move > 0 and (cur - desired) < (min_move * atr):
                    return cur

                return desired


        def _candidate_supertrend():
            if st_val is None:
                return None
            if side == "LONG":
                if st_val < curr_price:
                    return st_val
                return None
            else:  # SHORT
                if st_val > curr_price:
                    return st_val
                return None

        def _candidate_atr_trail():
            # 가격 기준 ATR 트레일: LONG=close - m*ATR, SHORT=close + m*ATR
            if atr is None:
                return None
            m = float(params.get("atr_mult", 3.0))
            if m <= 0:
                m = 3.0
            if side == "LONG":
                return curr_price - (m * atr)
            else:
                return curr_price + (m * atr)

        def _candidate_profit_lock():
            """
            이익이 trigger_atr*ATR 이상 나면 SL을 lock_atr*ATR 만큼 이익구간으로 끌어올림
              - LONG: if close >= entry + trigger*ATR  -> sl = max(cur_sl, entry + lock*ATR)
              - SHORT: if close <= entry - trigger*ATR -> sl = min(cur_sl, entry - lock*ATR)
            """
            if atr is None or entry_price is None:
                return None
            trigger = float(params.get("trigger_atr", 2.0))
            lock = float(params.get("lock_atr", 0.5))
            if trigger <= 0:
                trigger = 2.0
            # lock은 0 이상만 허용
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

        def _candidate_armor():
            """
            ARMOR SL = max/min(Structure, VolTrail, OutcomeInsurance)

            핵심 수정:
            - L2 VolTrail을 position['trail_sl']에 저장/갱신하여 stateful 유지
            - _apply_step_constraints()는 여기서 하지 않는다 (tighten 후 1회만)
            """
            df = _get_history_df()
            if df is None:
                return None

            # --- params ---
            swing_len = int(params.get("swing_len", 5))
            adx_period = int(params.get("adx_period", 14))
            adx_trend = float(params.get("adx_trend", 22))

            atr_mult_trend = float(params.get("atr_mult_trend", 4.0))
            atr_mult_chop = float(params.get("atr_mult_chop", 2.0))

            profit_trigger = float(params.get("profit_trigger_atr", 1.2))
            profit_lock = float(params.get("profit_lock_atr", 0.2))

            structure_buffer_atr = float(params.get("structure_buffer_atr", 0.3))
            fee_buffer_bps = float(params.get("fee_buffer_bps", 0.0))

            # --- need ATR / entry ---
            local_atr = atr
            if local_atr is None:
                local_atr = _wilder_atr_from_df(df, int(params.get("atr_period", 14)))
            if local_atr is None or local_atr <= 0:
                return None

            if entry_price is None or entry_price <= 0:
                return None

            # --- regime detection (ADX) ---
            adx_v = market_data.get("adx", None)
            try:
                adx_v = float(adx_v) if adx_v is not None else None
            except Exception:
                adx_v = None

            if adx_v is None:
                adx_v = _adx_from_df(df, adx_period)

            regime = "TREND" if (adx_v is not None and adx_v >= adx_trend) else "CHOP"
            m = atr_mult_trend if regime == "TREND" else atr_mult_chop
            if m <= 0:
                m = 2.0

            # -----------------------------
            # L1 Structure
            # -----------------------------
            swing_low, swing_high = _recent_swing_levels(df, swing_len)
            if swing_low is None or swing_high is None:
                return None

            if side == "LONG":
                structure_sl = float(swing_low) - (structure_buffer_atr * local_atr)
            else:
                structure_sl = float(swing_high) + (structure_buffer_atr * local_atr)

            # -----------------------------
            # L2 Vol Trail (STATEFUL)
            # -----------------------------
            trail_raw = position.get("trail_sl", None)
            try:
                trail_sl = float(trail_raw) if trail_raw is not None else None
            except Exception:
                trail_sl = None

            if side == "LONG":
                vol_candidate = curr_price - (m * local_atr)
                new_trail = vol_candidate if trail_sl is None else max(trail_sl, vol_candidate)
            else:
                vol_candidate = curr_price + (m * local_atr)
                new_trail = vol_candidate if trail_sl is None else min(trail_sl, vol_candidate)

            # ✅ trail state 갱신(엔진이 같은 객체를 들고 있으면 즉시 반영)
            position["trail_sl"] = float(new_trail)
            vol_sl = float(new_trail)

            # -----------------------------
            # L3 Outcome Insurance
            # -----------------------------
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

            # -----------------------------
            # Final (tightest)
            # -----------------------------
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



        def _tighten_sl(candidate, cur):
            """
            SL은 '손실을 키우는 방향'으로 이동하면 안 됨(=tighten만 허용)
              - LONG: SL은 올라가야 함 (candidate > cur)
              - SHORT: SL은 내려가야 함 (candidate < cur)
            """
            if candidate is None:
                return None
            try:
                c = float(candidate)
            except Exception:
                return None

            if cur is None:
                return c

            try:
                cur_f = float(cur)
            except Exception:
                return c

            if side == "LONG":
                return c if c > cur_f else None
            else:
                return c if c < cur_f else None

        cand = None
        if strat == "supertrend":
            cand = _tighten_sl(_candidate_supertrend(), current_sl)
            if cand is not None:
                new_sl = cand
                action = "UPDATE_SL"
                reason = "TRAILING"

        elif strat == "atr_trail":
            cand = _tighten_sl(_candidate_atr_trail(), current_sl)
            if cand is not None:
                new_sl = cand
                action = "UPDATE_SL"
                reason = "TRAILING_ATR"

        elif strat == "profit_lock":
            cand = _tighten_sl(_candidate_profit_lock(), current_sl)
            if cand is not None:
                new_sl = cand
                action = "UPDATE_SL"
                reason = "PROFIT_LOCK"

        elif strat == "armor":
            # 1) 후보 계산 (raw)
            raw = _candidate_armor()

            # 2) tighten (방향성 보장)
            tight = _tighten_sl(raw, current_sl)

            # 3) step/min_move 제약 (tighten 결과에만 1회 적용)
            cand = _apply_step_constraints(tight, current_sl)

            if cand is not None and (current_sl is None or float(cand) != float(current_sl)):
                new_sl = float(cand)
                action = "UPDATE_SL"
                reason = "ARMOR"


        elif strat == "hybrid":
            # 둘 다 있으면 LONG은 더 큰 SL(더 타이트), SHORT는 더 작은 SL(더 타이트) 선택
            c1 = _candidate_supertrend()
            c2 = _candidate_atr_trail()

            # 후보가 유효한지 + tighten 조건 통과시키기 위해 각각 tighten 적용
            t1 = _tighten_sl(c1, current_sl)
            t2 = _tighten_sl(c2, current_sl)

            chosen = None
            if t1 is not None and t2 is not None:
                chosen = max(t1, t2) if side == "LONG" else min(t1, t2)
            elif t1 is not None:
                chosen = t1
            elif t2 is not None:
                chosen = t2

            if chosen is not None:
                new_sl = float(chosen)
                action = "UPDATE_SL"
                reason = "HYBRID"

        else:
            # 알 수 없는 전략이면 기존 동작(안전): 업데이트 안 함
            cand = None

        # -----------------------------
        # 2) StopLoss Hit Check (same/next)
        # -----------------------------
        # current_sl이 None이면 히트 판정 자체를 하지 않는다.
        effective_sl_for_hit = current_sl
        if mode == "same" and action == "UPDATE_SL":
            effective_sl_for_hit = new_sl

        sl_hit = False
        if effective_sl_for_hit is not None:
            if side == "LONG" and low_price <= effective_sl_for_hit:
                sl_hit = True
            elif side == "SHORT" and high_price >= effective_sl_for_hit:
                sl_hit = True

        if sl_hit:
            return "EXIT", float(effective_sl_for_hit), "STOP_LOSS", new_sl

        # -----------------------------
        # 3) Return
        # -----------------------------
        if action == "UPDATE_SL":
            return "UPDATE_SL", 0.0, reason, new_sl

        return None, 0.0, None, current_sl
