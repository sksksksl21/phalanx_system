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
        - reason: "STOP_LOSS" | "TRAILING" | "TRAILING_ATR" | "PROFIT_LOCK" | "ARMOR" | "HYBRID" | "EMERGENCY" | None
        - new_sl: 계산된 신규 SL (None 가능)

    NOTE (중요):
      - Armor는 market_data["df"] 또는 market_data["history"](DataFrame)가 없으면 동작하지 않는다(기본).
      - Armor의 stateful trail은 position["trail_sl"]에 저장된다.
        (단일 진실원 저장/복원은 엔진/상태 계층에서 책임져야 한다)
    """

    def __init__(self):
        pass


    def _emit_em_debug(self, market_data: dict, stage: str, payload: dict):
        """
        Engine이 넘겨준 debug_sink로 emergency 진단 payload를 전달한다.
        PositionMonitor는 파일 I/O를 하지 않고, sink가 있으면 그것만 호출한다.
        """
        try:
            sink = (market_data or {}).get("debug_sink", None)
            if callable(sink):
                sink(stage, payload or {})
        except Exception:
            pass
        
    # -----------------------------
    # Public API
    # -----------------------------
    # =========================
    # [PATCH 1] check_conditions() 내부 수정
    # - armor 분기: 함수명 불일치 방지 + phased armor 후보 함수로 고정
    # - (선택) profit_lock 전략도 P1 통합을 원하면 _candidate_profit_lock_phased로 교체 가능(아래 PATCH 2)
    # =========================

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
        market_data = market_data or {}
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

        candle_time = market_data.get("candle_time", None)
        try:
            candle_time_s = str(pd.to_datetime(candle_time)) if candle_time is not None else None
        except Exception:
            candle_time_s = str(candle_time) if candle_time is not None else None

        # -----------------------------
        # [ADD] 0.5) next-mode SL breach latch -> exit on next candle
        # -----------------------------
        if mode == "next" and bool(position.get("sl_breached", False)):
            breached_at = position.get("sl_breached_at", None)
            breached_at_s = None
            try:
                breached_at_s = str(pd.to_datetime(breached_at)) if breached_at is not None else None
            except Exception:
                breached_at_s = str(breached_at) if breached_at is not None else None

            if (candle_time_s is None) or (breached_at_s is None) or (breached_at_s != candle_time_s):
                exec_sl = _safe_float(position.get("sl_breached_sl", None))
                if exec_sl is None:
                    exec_sl = _safe_float(position.get("sl", None))
                if exec_sl is not None:
                    exit_reason = str(position.get("sl_breached_reason", "STOP_LOSS") or "STOP_LOSS")

                    self._emit_em_debug(market_data, "NEXT_EXIT", {
                        "symbol": symbol,
                        "side": side,
                        "mode": mode,
                        "candle_time": candle_time_s,
                        "exec_sl": exec_sl,
                        "exit_reason": exit_reason,
                        "breached_at": breached_at_s,
                        "current_sl": current_sl,
                        "emergency_tag": position.get("emergency_tag"),
                        "defense_mode": int(bool(position.get("defense_mode", False))),
                    })

                    position.pop("sl_breached", None)
                    position.pop("sl_breached_sl", None)
                    position.pop("sl_breached_at", None)
                    position.pop("sl_breached_reason", None)
                    position.pop("sl_breached_tag", None)
                    position.pop("defense_mode", None)
                    position.pop("emergency_tag", None)

                    return "EXIT", float(exec_sl), exit_reason, _safe_float(position.get("sl", None))

        # -----------------------------
        # 1) Phase state (entry_atr fixed + MFE)
        # -----------------------------
        entry_atr_fixed: Optional[float] = None
        mfe_atr: float = 0.0
        phase: int = 1

        def _ensure_entry_atr() -> Optional[float]:
            ea = _safe_float(position.get("entry_atr", None))
            if ea is not None and ea > 0:
                return ea
            if atr is not None and atr > 0:
                position["entry_atr"] = float(atr)
                return float(atr)
            return None

        def _update_mfe_and_phase(entry_atr_v: float) -> Tuple[float, int]:
            if entry_price is None or entry_price <= 0:
                return 0.0, 1

            if side == "LONG":
                prev = _safe_float(position.get("peak_high", None))
                peak = high_price if prev is None else max(prev, high_price)
                position["peak_high"] = float(peak)
                mfe = max(0.0, (float(peak) - float(entry_price)) / float(entry_atr_v))
            else:
                prev = _safe_float(position.get("trough_low", None))
                trough = low_price if prev is None else min(prev, low_price)
                position["trough_low"] = float(trough)
                mfe = max(0.0, (float(entry_price) - float(trough)) / float(entry_atr_v))

            p1 = _safe_float(params.get("phase_p1_atr", 2.5), default=2.5) or 2.5
            p2 = _safe_float(params.get("phase_p2_atr", 5.5), default=5.5) or 5.5
            if mfe < p1:
                ph = 1
            elif mfe < p2:
                ph = 2
            else:
                ph = 3
            return float(mfe), int(ph)

        if entry_price is not None and entry_price > 0:
            entry_atr_fixed = _ensure_entry_atr()
            if entry_atr_fixed is not None and entry_atr_fixed > 0:
                mfe_atr, phase = _update_mfe_and_phase(entry_atr_fixed)

        self._emit_em_debug(market_data, "STATE", {
            "symbol": symbol,
            "side": side,
            "mode": mode,
            "strategy": strat,
            "candle_time": candle_time_s,
            "close": curr_price,
            "high": high_price,
            "low": low_price,
            "atr": atr,
            "current_sl": current_sl,
            "entry_price": entry_price,
            "entry_atr_fixed": entry_atr_fixed,
            "peak_high": _safe_float(position.get("peak_high", None)),
            "trough_low": _safe_float(position.get("trough_low", None)),
            "mfe_atr": mfe_atr,
            "phase": phase,
            "warn": int(bool(position.get("emergency_warn", False))),
            "defense_mode": int(bool(position.get("defense_mode", False))),
            "emergency_tag": position.get("emergency_tag"),
        })

        # -----------------------------
        # 2) Helpers
        # -----------------------------
        def _tighten_sl(candidate: Optional[float], cur: Optional[float]) -> Optional[float]:
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
            return c if c < cur_f else None

        def _apply_step_constraints(tightened_sl: Optional[float], cur_sl: Optional[float]) -> Optional[float]:
            if tightened_sl is None:
                return None
            if atr is None or atr <= 0:
                return _safe_float(tightened_sl)

            t = _safe_float(tightened_sl)
            if t is None:
                return None

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
            apply_all = bool(params.get("apply_step_all", False))
            return _apply_step_constraints(cand_tight, cur_sl) if apply_all else cand_tight

        def _apply_constraints_emergency(cand_tight: Optional[float], cur_sl: Optional[float]) -> Optional[float]:
            return _apply_step_constraints(cand_tight, cur_sl)

        # -----------------------------
        # 3) Candidate builders
        # -----------------------------
        def _candidate_supertrend() -> Optional[float]:
            if st_val is None:
                return None
            if side == "LONG":
                return st_val if st_val < curr_price else None
            return st_val if st_val > curr_price else None

        def _candidate_atr_trail() -> Optional[float]:
            if atr is None:
                return None
            m = _safe_float(params.get("atr_mult", 3.0), default=3.0) or 3.0
            if m <= 0:
                m = 3.0
            return (curr_price - (m * atr)) if side == "LONG" else (curr_price + (m * atr))

        def _get_history_df() -> Optional[pd.DataFrame]:
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
                dfx["high"] = pd.to_numeric(dfx["high"], errors="coerce")
                dfx["low"] = pd.to_numeric(dfx["low"], errors="coerce")
                dfx["close"] = pd.to_numeric(dfx["close"], errors="coerce")
                dfx = dfx.dropna(subset=["high", "low", "close"])
                if len(dfx) < 5:
                    return None
                return dfx
            except Exception:
                return None

        def _recent_swing_levels(df: pd.DataFrame, swing_len: int) -> Tuple[Optional[float], Optional[float]]:
            n = int(max(2, swing_len))
            try:
                w = df.tail(n + 1).iloc[:-1]
                if len(w) < 2:
                    w = df.tail(n)
                return _safe_float(w["low"].min()), _safe_float(w["high"].max())
            except Exception:
                return None, None

        def _candidate_emergency_defense() -> Tuple[Optional[float], Optional[str]]:
            if entry_price is None or entry_price <= 0:
                self._emit_em_debug(market_data, "PRECHECK", {
                    "symbol": symbol,
                    "side": side,
                    "pass": 0,
                    "reason": "entry_price_invalid",
                    "entry_price": entry_price,
                    "entry_atr_fixed": entry_atr_fixed,
                })
                return None, None

            if entry_atr_fixed is None or entry_atr_fixed <= 0:
                self._emit_em_debug(market_data, "PRECHECK", {
                    "symbol": symbol,
                    "side": side,
                    "pass": 0,
                    "reason": "entry_atr_invalid",
                    "entry_price": entry_price,
                    "entry_atr_fixed": entry_atr_fixed,
                })
                return None, None

            em_mfe_min = _safe_float(params.get("emergency_mfe_min_atr", 1.5), default=1.5) or 1.5
            em_giveback = _safe_float(params.get("emergency_giveback_atr", 1.0), default=1.0) or 1.0
            em_giveback_hard = _safe_float(params.get("emergency_giveback_hard_atr", 1.7), default=1.7) or 1.7

            em_swing_len = int(params.get("emergency_swing_len", params.get("swing_len", 5)))
            em_struct_buf = _safe_float(
                params.get("emergency_structure_buffer_atr", params.get("structure_buffer_atr", 0.3)),
                default=0.3
            ) or 0.3
            em_struct_confirm_atr = _safe_float(params.get("emergency_structure_confirm_atr", 0.0), default=0.0) or 0.0

            em_profit_lock = _safe_float(
                params.get("emergency_profit_lock_atr", params.get("profit_lock_atr", 0.2)),
                default=0.2
            ) or 0.2
            fee_buffer_bps = _safe_float(params.get("fee_buffer_bps", 0.0), default=0.0) or 0.0

            gate_pass = bool(mfe_atr >= em_mfe_min)
            self._emit_em_debug(market_data, "GATE", {
                "symbol": symbol,
                "side": side,
                "gate_pass": int(gate_pass),
                "mfe_atr": mfe_atr,
                "em_mfe_min": em_mfe_min,
                "entry_atr_fixed": entry_atr_fixed,
                "peak_high": _safe_float(position.get("peak_high", None)),
                "trough_low": _safe_float(position.get("trough_low", None)),
            })

            if not gate_pass:
                position.pop("emergency_warn", None)
                position.pop("emergency_warn_at", None)
                return None, None

            tags: list[str] = []
            candidates: list[float] = []

            if side == "LONG":
                peak = _safe_float(position.get("peak_high", None))
                if peak is None:
                    peak = high_price

                prev_warn = bool(position.get("emergency_warn", False))
                giveback_atr = (float(peak) - float(curr_price)) / float(entry_atr_fixed)

                if giveback_atr >= em_giveback:
                    position["emergency_warn"] = True
                    if candle_time_s is not None:
                        position["emergency_warn_at"] = candle_time_s
                else:
                    position.pop("emergency_warn", None)
                    position.pop("emergency_warn_at", None)

                warn_after = bool(position.get("emergency_warn", False))
                hard = giveback_atr >= em_giveback_hard
                consecutive = prev_warn and (giveback_atr >= em_giveback)

                self._emit_em_debug(market_data, "GIVEBACK", {
                    "symbol": symbol,
                    "side": side,
                    "peak": peak,
                    "curr_price": curr_price,
                    "giveback_atr": giveback_atr,
                    "warn_before": int(prev_warn),
                    "warn_after": int(warn_after),
                    "hard": int(hard),
                    "consecutive": int(consecutive),
                    "em_giveback": em_giveback,
                    "em_giveback_hard": em_giveback_hard,
                })

                if hard or consecutive:
                    tags.append("GIVEBACK")
                    candidates.append(float(peak) - (em_giveback * float(entry_atr_fixed)))

            else:
                trough = _safe_float(position.get("trough_low", None))
                if trough is None:
                    trough = low_price

                prev_warn = bool(position.get("emergency_warn", False))
                giveback_atr = (float(curr_price) - float(trough)) / float(entry_atr_fixed)

                if giveback_atr >= em_giveback:
                    position["emergency_warn"] = True
                    if candle_time_s is not None:
                        position["emergency_warn_at"] = candle_time_s
                else:
                    position.pop("emergency_warn", None)
                    position.pop("emergency_warn_at", None)

                warn_after = bool(position.get("emergency_warn", False))
                hard = giveback_atr >= em_giveback_hard
                consecutive = prev_warn and (giveback_atr >= em_giveback)

                self._emit_em_debug(market_data, "GIVEBACK", {
                    "symbol": symbol,
                    "side": side,
                    "trough": trough,
                    "curr_price": curr_price,
                    "giveback_atr": giveback_atr,
                    "warn_before": int(prev_warn),
                    "warn_after": int(warn_after),
                    "hard": int(hard),
                    "consecutive": int(consecutive),
                    "em_giveback": em_giveback,
                    "em_giveback_hard": em_giveback_hard,
                })

                if hard or consecutive:
                    tags.append("GIVEBACK")
                    candidates.append(float(trough) + (em_giveback * float(entry_atr_fixed)))

            df = _get_history_df()
            if df is not None:
                swing_low, swing_high = _recent_swing_levels(df, em_swing_len)
                if swing_low is not None and swing_high is not None:
                    buf_px = em_struct_buf * float(entry_atr_fixed)
                    conf_px = em_struct_confirm_atr * float(entry_atr_fixed)

                    if side == "LONG":
                        breached = (float(low_price) <= (float(swing_low) - buf_px))
                        confirmed = (float(curr_price) <= (float(swing_low) - conf_px))
                        self._emit_em_debug(market_data, "STRUCTURE", {
                            "symbol": symbol,
                            "side": side,
                            "df_ready": 1,
                            "swing_low": swing_low,
                            "buf_px": buf_px,
                            "conf_px": conf_px,
                            "low_price": low_price,
                            "curr_price": curr_price,
                            "breached": int(breached),
                            "confirmed": int(confirmed),
                        })
                        if breached and confirmed:
                            tags.append("STRUCTURE")
                            candidates.append(float(swing_low) - buf_px)
                    else:
                        breached = (float(high_price) >= (float(swing_high) + buf_px))
                        confirmed = (float(curr_price) >= (float(swing_high) + conf_px))
                        self._emit_em_debug(market_data, "STRUCTURE", {
                            "symbol": symbol,
                            "side": side,
                            "df_ready": 1,
                            "swing_high": swing_high,
                            "buf_px": buf_px,
                            "conf_px": conf_px,
                            "high_price": high_price,
                            "curr_price": curr_price,
                            "breached": int(breached),
                            "confirmed": int(confirmed),
                        })
                        if breached and confirmed:
                            tags.append("STRUCTURE")
                            candidates.append(float(swing_high) + buf_px)
                else:
                    self._emit_em_debug(market_data, "STRUCTURE", {
                        "symbol": symbol,
                        "side": side,
                        "df_ready": 1,
                        "swing_ready": 0,
                    })
            else:
                self._emit_em_debug(market_data, "STRUCTURE", {
                    "symbol": symbol,
                    "side": side,
                    "df_ready": 0,
                    "swing_ready": 0,
                })

            fee_pad = float(entry_price) * (fee_buffer_bps * 0.0001)
            if side == "LONG":
                lock_candidate = float(entry_price) + fee_pad + (em_profit_lock * float(entry_atr_fixed))
            else:
                lock_candidate = float(entry_price) - fee_pad - (em_profit_lock * float(entry_atr_fixed))
            candidates.append(lock_candidate)

            if not candidates:
                self._emit_em_debug(market_data, "RAW", {
                    "symbol": symbol,
                    "side": side,
                    "candidate_ct": 0,
                    "lock_candidate": lock_candidate,
                })
                return None, None

            raw = max(candidates) if side == "LONG" else min(candidates)
            tag = "EMERGENCY:" + ("+".join(tags) if tags else "LOCK")

            self._emit_em_debug(market_data, "RAW", {
                "symbol": symbol,
                "side": side,
                "candidate_ct": len(candidates),
                "tags": "+".join(tags) if tags else "LOCK",
                "lock_candidate": lock_candidate,
                "em_raw": raw,
                "em_tag": tag,
            })

            return float(raw), tag

        def _candidate_armor_phased() -> Optional[float]:
            if entry_price is None or entry_price <= 0:
                return None

            if entry_atr_fixed is None or entry_atr_fixed <= 0:
                logger.warning(f"[ARMOR_PHASED] missing entry_atr | symbol={symbol}")
                fallback = str(params.get("armor_fallback", "atr_trail")).strip().lower()
                return _candidate_atr_trail() if fallback == "atr_trail" else None

            swing_len = int(params.get("swing_len", 5))
            adx_trend = _safe_float(params.get("adx_trend", 22.0), default=22.0) or 22.0

            atr_mult_trend = _safe_float(params.get("atr_mult_trend", 4.0), default=4.0) or 4.0
            atr_mult_chop = _safe_float(params.get("atr_mult_chop", 2.0), default=2.0) or 2.0

            profit_lock = _safe_float(params.get("profit_lock_atr", 0.2), default=0.2) or 0.2
            structure_buffer_atr = _safe_float(params.get("structure_buffer_atr", 0.3), default=0.3) or 0.3
            fee_buffer_bps = _safe_float(params.get("fee_buffer_bps", 0.0), default=0.0) or 0.0

            adx_v = _safe_float(market_data.get("adx", None))
            in_trend = (adx_v is not None and adx_v >= adx_trend)

            if phase >= 3:
                mult = atr_mult_chop
            else:
                mult = atr_mult_trend if in_trend else atr_mult_chop

            candidates: list[float] = []

            df = _get_history_df()
            if df is not None:
                swing_low, swing_high = _recent_swing_levels(df, swing_len)
                if swing_low is not None and swing_high is not None:
                    if side == "LONG":
                        structure_sl = float(swing_low) - (structure_buffer_atr * entry_atr_fixed)
                    else:
                        structure_sl = float(swing_high) + (structure_buffer_atr * entry_atr_fixed)
                    candidates.append(float(structure_sl))
                else:
                    logger.warning(f"[ARMOR_PHASED] swing levels missing | symbol={symbol}")
            else:
                logger.warning(f"[ARMOR_PHASED] df missing -> STRUCTURE disabled | symbol={symbol}")

            trail_sl = _safe_float(position.get("trail_sl", None))
            if side == "LONG":
                vol_candidate = curr_price - (mult * entry_atr_fixed)
                new_trail = vol_candidate if trail_sl is None else max(trail_sl, vol_candidate)
            else:
                vol_candidate = curr_price + (mult * entry_atr_fixed)
                new_trail = vol_candidate if trail_sl is None else min(trail_sl, vol_candidate)

            position["trail_sl"] = float(new_trail)
            candidates.append(float(new_trail))

            if phase >= 2:
                fee_pad = float(entry_price) * (fee_buffer_bps * 0.0001)
                if side == "LONG":
                    outcome_sl = float(entry_price) + fee_pad + (profit_lock * entry_atr_fixed)
                else:
                    outcome_sl = float(entry_price) - fee_pad - (profit_lock * entry_atr_fixed)
                candidates.append(float(outcome_sl))

            if not candidates:
                logger.warning(f"[ARMOR_PHASED] no candidates | symbol={symbol} phase={phase}")
                return None

            raw = max(candidates) if side == "LONG" else min(candidates)
            return float(raw)

        # -----------------------------
        # 4) Choose strategy + compute new_sl
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
            _pl_fn = locals().get("_candidate_profit_lock", None)
            if callable(_pl_fn):
                pl_raw = _pl_fn()
                cand = _tighten_sl(pl_raw, current_sl)
                cand = _maybe_apply_constraints(cand, current_sl)
                if cand is not None:
                    new_sl = float(cand)
                    action = "UPDATE_SL"
                    reason = "PROFIT_LOCK"

        elif strat == "armor":
            raw = _candidate_armor_phased()
            tight = _tighten_sl(raw, current_sl)
            cand = _apply_step_constraints(tight, current_sl)
            if cand is not None and (current_sl is None or _safe_float(cand) != _safe_float(current_sl)):
                new_sl = float(cand)
                action = "UPDATE_SL"
                reason = "ARMOR"

        elif strat == "hybrid":
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

        # -----------------------------
        # 4.5) Emergency Defense Override
        # -----------------------------
        base_action = action
        base_reason = reason
        base_new_sl = new_sl

        em_raw, em_tag = _candidate_emergency_defense()
        em_tight = _tighten_sl(em_raw, current_sl)
        em_tight = _apply_constraints_emergency(em_tight, current_sl)

        take = False
        if em_tight is not None:
            if new_sl is None:
                take = True
            else:
                if side == "LONG":
                    take = float(em_tight) > float(new_sl)
                else:
                    take = float(em_tight) < float(new_sl)

            if take:
                new_sl = float(em_tight)
                action = "UPDATE_SL"
                reason = "EMERGENCY"
                position["emergency_tag"] = em_tag
                position["defense_mode"] = True

        self._emit_em_debug(market_data, "DECISION", {
            "symbol": symbol,
            "side": side,
            "base_action": base_action,
            "base_reason": base_reason,
            "base_new_sl": base_new_sl,
            "current_sl": current_sl,
            "em_raw": em_raw,
            "em_tag": em_tag,
            "em_tight": em_tight,
            "take": int(take),
            "final_action": action,
            "final_reason": reason,
            "final_new_sl": new_sl,
            "defense_mode": int(bool(position.get("defense_mode", False))),
        })

        # -----------------------------
        # 5) StopLoss Hit Check (same/next)
        # -----------------------------
        effective_sl_for_hit = current_sl
        if mode == "same" and action == "UPDATE_SL":
            effective_sl_for_hit = new_sl

        entry_time = position.get("entry_time", None)
        try:
            entry_time_s = str(pd.to_datetime(entry_time)) if entry_time is not None else None
        except Exception:
            entry_time_s = str(entry_time) if entry_time is not None else None

        is_entry_candle = (
            (candle_time_s is not None)
            and (entry_time_s is not None)
            and (entry_time_s == candle_time_s)
        )

        sl_hit = False
        if (not is_entry_candle) and (effective_sl_for_hit is not None):
            slv = float(effective_sl_for_hit)
            if side == "LONG" and low_price <= slv:
                sl_hit = True
            elif side == "SHORT" and high_price >= slv:
                sl_hit = True

        if sl_hit:
            if mode == "next":
                position["sl_breached"] = True
                position["sl_breached_sl"] = float(effective_sl_for_hit)
                if candle_time_s is not None:
                    position["sl_breached_at"] = candle_time_s

                is_emergency_context = (
                    (action == "UPDATE_SL" and reason == "EMERGENCY")
                    or bool(position.get("defense_mode", False))
                )
                position["sl_breached_reason"] = "EMERGENCY_STOP" if is_emergency_context else "STOP_LOSS"

                if position.get("emergency_tag", None) is not None:
                    position["sl_breached_tag"] = position.get("emergency_tag", None)

                self._emit_em_debug(market_data, "HIT", {
                    "symbol": symbol,
                    "side": side,
                    "mode": mode,
                    "effective_sl_for_hit": effective_sl_for_hit,
                    "curr_price": curr_price,
                    "high_price": high_price,
                    "low_price": low_price,
                    "is_entry_candle": int(is_entry_candle),
                    "is_emergency_context": int(is_emergency_context),
                    "sl_breached_reason": position.get("sl_breached_reason"),
                    "final_action_before_return": action,
                    "final_reason_before_return": reason,
                    "final_new_sl_before_return": new_sl,
                })

                if action == "UPDATE_SL":
                    return "UPDATE_SL", 0.0, reason, new_sl
                return None, 0.0, None, current_sl

            exit_reason = "EMERGENCY_STOP" if (reason == "EMERGENCY" or bool(position.get("defense_mode", False))) else "STOP_LOSS"

            self._emit_em_debug(market_data, "HIT", {
                "symbol": symbol,
                "side": side,
                "mode": mode,
                "effective_sl_for_hit": effective_sl_for_hit,
                "curr_price": curr_price,
                "high_price": high_price,
                "low_price": low_price,
                "is_entry_candle": int(is_entry_candle),
                "exit_reason": exit_reason,
                "final_action_before_return": action,
                "final_reason_before_return": reason,
                "final_new_sl_before_return": new_sl,
            })

            return "EXIT", float(effective_sl_for_hit), exit_reason, new_sl

        return action, float(exec_price), reason, new_sl

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
