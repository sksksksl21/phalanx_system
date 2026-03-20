import numpy as np
import pandas as pd
import pandas_ta as ta


class TitanStrategy:
    def __init__(self):
        # =====================================================================
        # [Universe / Symbol Policy]
        # =====================================================================
        self.major_coins = {
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "SOL/USDT",
            "XRP/USDT",
            "ADA/USDT",
            "AVAX/USDT",
        }

        # =====================================================================
        # [Blacklist]
        # =====================================================================
        self.blacklist = set()

        # =====================================================================
        # [Versioning]
        # =====================================================================
        self.__version__ = "9.0.0-ContinuationPullback"

        # =====================================================================
        # [Params] 최소화
        # - 추세 필터
        # - pullback 허용 범위
        # - continuation 확인 폭
        # =====================================================================
        self.params = {
            "atr_period": 18,
            "atr_multiplier": 2.25,
            "adx_threshold": 18,
            "daily_ema": 25,
            "ema_intraday": 200,
            "pullback_lookback": 3,
            "pullback_tolerance_atr": 0.6,
            "breakout_buffer_atr": 0.05,
        }


    def get_blacklist(self):
        return list(self.blacklist)

    def set_params(self, params):
        self.params.update(params)

    # =========================================================
    # Utils
    # =========================================================
    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if isinstance(df.index, pd.DatetimeIndex):
            if not df.index.is_monotonic_increasing:
                df = df.sort_index()
            return df

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
            df = df.set_index("timestamp")
            df = df[~df.index.isna()]
            df = df.sort_index()
            return df

        try:
            df.index = pd.to_datetime(df.index, errors="coerce", utc=False)
            df = df[~df.index.isna()]
            df = df.sort_index()
        except Exception:
            pass

        return df

    def _safe_pick_adx(self, adx_df: pd.DataFrame):
        if adx_df is None or not isinstance(adx_df, pd.DataFrame) or adx_df.empty:
            return None
        adx_cols = [c for c in adx_df.columns if str(c).upper().startswith("ADX_")]
        if adx_cols:
            return adx_df[adx_cols[0]]
        return adx_df[adx_df.columns[0]]

    def _safe_pick_supertrend(self, st_df: pd.DataFrame):
        if st_df is None or not isinstance(st_df, pd.DataFrame) or st_df.empty:
            return None, None

        cols = [str(c) for c in st_df.columns]

        dir_col = None
        for c in cols:
            if c.upper().startswith("SUPERTD_"):
                dir_col = c
                break

        val_col = None
        for c in cols:
            if c.upper().startswith("SUPERT_"):
                val_col = c
                break

        if val_col is None:
            for c in cols:
                if c.upper().startswith("SUPERTL_"):
                    val_col = c
                    break
        if val_col is None:
            for c in cols:
                if c.upper().startswith("SUPERTS_"):
                    val_col = c
                    break

        def _match(df_cols, target_str):
            for real in df_cols:
                if str(real) == target_str:
                    return real
            return None

        val_real = _match(st_df.columns, val_col) if val_col else None
        dir_real = _match(st_df.columns, dir_col) if dir_col else None

        st_val_ser = st_df[val_real] if val_real is not None else None
        st_dir_ser = st_df[dir_real] if dir_real is not None else None

        return st_val_ser, st_dir_ser

    def _safe_ema_daily_map(self, df_15m: pd.DataFrame, daily_len: int):
        try:
            daily_df = df_15m.resample("D").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last"}
            )
            daily_df = daily_df.dropna(subset=["open", "high", "low", "close"])
            if daily_df.empty:
                return None, False

            if len(daily_df) < (daily_len + 5):
                return None, False

            daily_df["ema_daily"] = ta.ema(daily_df["close"], length=daily_len)

            # ✅ 확정된 "어제" 값만 사용 (lookahead 방지)
            daily_df = daily_df.shift(1)

            ema_mapped = daily_df["ema_daily"].reindex(df_15m.index, method="ffill")
            return ema_mapped, True

        except Exception:
            return None, False

    def _compute_pullback_continuation_flags(self, df: pd.DataFrame, p: dict) -> pd.DataFrame:
        """
        순수 continuation pullback 구조:
        1) trend_up / trend_down
        2) 최근 N봉 안에 EMA 근처 pullback 발생
        3) 현재봉이 직전봉 돌파/이탈로 continuation 확인
        """
        df = df.copy()

        lookback = int(max(1, p.get("pullback_lookback", 3) or 3))
        tol_atr = float(p.get("pullback_tolerance_atr", 0.6) or 0.6)
        brk_atr = float(p.get("breakout_buffer_atr", 0.05) or 0.05)

        atr = pd.to_numeric(df["atr"], errors="coerce").fillna(0.0)
        ema = pd.to_numeric(df["ema_intra"], errors="coerce")
        close = pd.to_numeric(df["close"], errors="coerce")
        open_ = pd.to_numeric(df["open"], errors="coerce")
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        st_val = pd.to_numeric(df["st_val"], errors="coerce")

        df["ema_slope_up"] = (ema > ema.shift(1)).fillna(False).astype("int")
        df["ema_slope_down"] = (ema < ema.shift(1)).fillna(False).astype("int")

        trend_up = (
            ema.notna()
            & (close > ema)
            & (df["ema_slope_up"] == 1)
            & (df["st_dir"] > 0)
            & (st_val.isna() | (close > st_val))
        )

        trend_down = (
            ema.notna()
            & (close < ema)
            & (df["ema_slope_down"] == 1)
            & (df["st_dir"] < 0)
            & (st_val.isna() | (close < st_val))
        )

        df["trend_up"] = trend_up.fillna(False).astype("int")
        df["trend_down"] = trend_down.fillna(False).astype("int")

        tol_px = atr * tol_atr

        # pullback bar 자체는 과거봉에서 발생해야 하므로 recent는 뒤에서 shift(1) 처리
        df["pullback_long_touch"] = (
            trend_up
            & (low <= (ema + tol_px))
            & (close >= ema)
        ).fillna(False).astype("int")

        df["pullback_short_touch"] = (
            trend_down
            & (high >= (ema - tol_px))
            & (close <= ema)
        ).fillna(False).astype("int")

        df["pullback_long_recent"] = (
            df["pullback_long_touch"]
            .rolling(window=lookback, min_periods=1)
            .max()
            .shift(1)
            .fillna(0)
            .astype("int")
        )

        df["pullback_short_recent"] = (
            df["pullback_short_touch"]
            .rolling(window=lookback, min_periods=1)
            .max()
            .shift(1)
            .fillna(0)
            .astype("int")
        )

        df["breakout_up_level"] = (high.shift(1) + (atr * brk_atr)).astype(float)
        df["breakout_dn_level"] = (low.shift(1) - (atr * brk_atr)).astype(float)

        df["continuation_long"] = (
            trend_up
            & df["breakout_up_level"].notna()
            & (close > df["breakout_up_level"])
            & (close > open_)
        ).fillna(False).astype("int")

        df["continuation_short"] = (
            trend_down
            & df["breakout_dn_level"].notna()
            & (close < df["breakout_dn_level"])
            & (close < open_)
        ).fillna(False).astype("int")

        return df

    # =========================================================
    # Volume = Score only (NO GATE)
    # =========================================================
    def _calc_vol_score(self, curr, prev, p):
        use_vol = bool(p.get("use_vol_filter", True))
        if not use_vol:
            return True, 1.0, 0.0

        try:
            prev_vol = float(prev.get("volume", 0) or 0)
            prev_vol_ma = float(prev.get("vol_ma", 0) or 0)
            if prev_vol_ma <= 0:
                return True, 1.0, 0.0

            vol_ratio = prev_vol / prev_vol_ma
            thr = float(p.get("vol_factor", 1.0) or 1.0)

            vol_ok_like = bool(prev_vol > (prev_vol_ma * thr))

            denom = max(thr, 1e-12)
            raw = (vol_ratio / denom) - 1.0
            vol_score = float(np.clip(raw, -1.0, 1.0))

            return vol_ok_like, float(vol_ratio), float(vol_score)
        except Exception:
            return True, 1.0, 0.0

    # =========================================================
    # Market Structure Helpers (Pure, no I/O)
    # =========================================================
    def _compute_pivots_confirmed(self, df: pd.DataFrame, swing_len: int):
        n = int(max(1, swing_len))
        w = 2 * n + 1

        roll_high = df["high"].rolling(window=w, min_periods=w).max()
        roll_low = df["low"].rolling(window=w, min_periods=w).min()

        pivot_high_conf = (df["high"].shift(n) == roll_high).astype("int")
        pivot_low_conf = (df["low"].shift(n) == roll_low).astype("int")

        pivot_high_conf = pivot_high_conf.fillna(0).astype("int")
        pivot_low_conf = pivot_low_conf.fillna(0).astype("int")

        return pivot_high_conf, pivot_low_conf

    @staticmethod
    def _age_from_triggers(trigger: pd.Series) -> pd.Series:
        t = trigger.fillna(0).astype(int).to_numpy()
        idx = np.arange(len(t), dtype=float)
        last = np.where(t == 1, idx, np.nan)
        last = pd.Series(last, index=trigger.index).ffill()
        age = pd.Series(np.arange(len(t), dtype=float), index=trigger.index) - last
        age = age.where(last.notna(), np.nan)
        return age

    def _compute_trend_state_from_pivots(self, df: pd.DataFrame, p):
        """
        ✅ 확정 피봇 기반 추세 상태(up/down/range) 추정.
        - 최근 2개의 pivot_high_price, pivot_low_price를 이용
        - lookahead 없음(confirmed pivot만 사용)
        반환 컬럼:
          - ph_last, ph_prev, pl_last, pl_prev
          - trend_state: +1 uptrend, -1 downtrend, 0 range/unknown
        """
        need = int(p.get("structure_min_pivots", 2))

        ph = df["pivot_high_price"].copy()
        pl = df["pivot_low_price"].copy()

        # 각 시점에서 "최근 2개 피봇 가격"을 추출
        # 방법: 유효 값만 누적 리스트처럼 만들기 어렵기 때문에, forward-fill로 last를 만들고,
        # prev는 last가 갱신된 시점 기준으로 한 칸 전 last를 기억하는 방식 사용
        ph_last = ph.ffill()
        pl_last = pl.ffill()

        # prev: last가 갱신될 때만 이전 last를 스냅샷해 유지
        ph_change = ph.notna().astype(int)
        pl_change = pl.notna().astype(int)

        ph_seg = ph_change.cumsum()
        pl_seg = pl_change.cumsum()

        # 그룹별 첫 값(갱신 직후의 last)에서 그 직전 last를 당겨오도록 shift
        ph_prev = ph_last.groupby(ph_seg).transform("first").shift(1).ffill()
        pl_prev = pl_last.groupby(pl_seg).transform("first").shift(1).ffill()

        # 피봇이 충분치 않으면 NaN으로 정리
        ph_cnt = ph_change.cumsum()
        pl_cnt = pl_change.cumsum()

        ph_last = ph_last.where(ph_cnt >= 1, np.nan)
        pl_last = pl_last.where(pl_cnt >= 1, np.nan)
        ph_prev = ph_prev.where(ph_cnt >= need, np.nan)
        pl_prev = pl_prev.where(pl_cnt >= need, np.nan)

        # trend_state 판정
        trend_state = pd.Series(0, index=df.index).astype(int)

        cond_up = ph_last.notna() & ph_prev.notna() & pl_last.notna() & pl_prev.notna() & (ph_last > ph_prev) & (pl_last > pl_prev)
        cond_dn = ph_last.notna() & ph_prev.notna() & pl_last.notna() & pl_prev.notna() & (ph_last < ph_prev) & (pl_last < pl_prev)

        trend_state = trend_state.mask(cond_up, 1)
        trend_state = trend_state.mask(cond_dn, -1)

        return ph_last.astype(float), ph_prev.astype(float), pl_last.astype(float), pl_prev.astype(float), trend_state

    def _calc_entry_score(self, curr, p, side: str) -> float:
        side = str(side).upper()
        score = 0.0

        vol_score = float(np.clip(float(curr.get("vol_score_prev", 0.0) or 0.0), -1.0, 1.0))
        atr_ratio = float(curr.get("atr_ratio", 0.0) or 0.0)
        atr_up = int(curr.get("atr_up", 0) or 0)

        body_ratio = float(np.clip(float(curr.get("body_ratio", 0.0) or 0.0), 0.0, 1.0))
        close_pos = float(np.clip(float(curr.get("close_pos", 0.5) or 0.5), 0.0, 1.0))
        lower_wick_ratio = float(np.clip(float(curr.get("lower_wick_ratio", 0.0) or 0.0), 0.0, 1.0))
        upper_wick_ratio = float(np.clip(float(curr.get("upper_wick_ratio", 0.0) or 0.0), 0.0, 1.0))

        atr_need = float(p.get("atr_regime_factor", 1.0) or 1.0)
        reclaim_min = float(p.get("retest_reclaim_min", 0.55) or 0.55)
        body_min = float(p.get("retest_body_min", 0.20) or 0.20)

        if side == "LONG":
            age = curr.get("choch_age_up", np.nan)

            if int(curr.get("retest_long", 0)) == 1:
                score += 1.20

            if pd.notna(age):
                age = float(age)
                if age <= float(p.get("fresh_retest_bars_long", 4) or 4):
                    score += 0.90
                elif age <= float(p.get("retest_max_bars_long", 8) or 8):
                    score += 0.45
                else:
                    score -= 1.00

            if atr_ratio >= atr_need:
                score += 0.45
            elif atr_ratio >= 1.0:
                score += 0.20
            else:
                score -= 0.15

            if atr_up == 1:
                score += 0.20

            score += float(np.clip(vol_score * 0.35, -0.35, 0.35))

            if lower_wick_ratio >= upper_wick_ratio:
                score += 0.15
            if close_pos >= reclaim_min:
                score += 0.25
            if body_ratio >= body_min:
                score += 0.15
            if int(curr.get("bos_ok_long", 0)) == 1:
                score += 0.30

        else:
            age = curr.get("choch_age_down", np.nan)

            if int(curr.get("retest_short", 0)) == 1:
                score += 1.20

            if pd.notna(age):
                age = float(age)
                if age <= float(p.get("fresh_retest_bars_short", 3) or 3):
                    score += 0.90
                elif age <= float(p.get("retest_max_bars_short", 6) or 6):
                    score += 0.45
                else:
                    score -= 1.00

            if atr_ratio >= atr_need:
                score += 0.45
            elif atr_ratio >= 1.0:
                score += 0.20
            else:
                score -= 0.15

            if atr_up == 1:
                score += 0.20

            score += float(np.clip(vol_score * 0.35, -0.35, 0.35))

            if upper_wick_ratio >= lower_wick_ratio:
                score += 0.15
            if close_pos <= (1.0 - reclaim_min):
                score += 0.25
            if body_ratio >= body_min:
                score += 0.15
            if int(curr.get("bos_ok_short", 0)) == 1:
                score += 0.30

        return float(score)

    # =========================================================
    # Indicators
    # =========================================================
    def calculate_indicators(self, symbol, df):
        df = self._ensure_datetime_index(df)
        df = df.copy()
        p = self.params

        # =========================================================
        # 1) Intraday Indicators (15m)
        # =========================================================
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=int(p["atr_period"]))
        df["vol_ma"] = df["volume"].rolling(window=20, min_periods=1).mean()
        df["ema_intra"] = ta.ema(df["close"], length=int(p["ema_intraday"]))
        df["rsi"] = ta.rsi(df["close"], length=14)

        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        adx_series = self._safe_pick_adx(adx_df)
        df["adx"] = adx_series if adx_series is not None else 0.0

        st = ta.supertrend(df["high"], df["low"], df["close"], length=12, multiplier=3.0)
        st_val_ser, st_dir_ser = self._safe_pick_supertrend(st)
        df["st_val"] = st_val_ser if st_val_ser is not None else np.nan
        if st_dir_ser is not None:
            df["st_dir"] = st_dir_ser.astype("float").round(0).astype("int")
        else:
            df["st_dir"] = 0

        # =========================================================
        # 2) Daily EMA (lookahead 방지: shift(1))
        # =========================================================
        ema_daily_mapped, ok = self._safe_ema_daily_map(df, int(p.get("daily_ema", 25)))
        if not ok or ema_daily_mapped is None:
            df["ema_daily"] = 0.0
            df["ema_daily_ok"] = 0
        else:
            df["ema_daily"] = pd.Series(ema_daily_mapped, index=df.index).astype(float).fillna(0.0)
            df["ema_daily_ok"] = 1

        # =========================================================
        # 3) Continuation Pullback Flags
        # =========================================================
        df = self._compute_pullback_continuation_flags(df, p)

        # =========================================================
        # 4) 타입 정리
        # =========================================================
        int_cols = [
            "st_dir",
            "ema_daily_ok",
            "ema_slope_up",
            "ema_slope_down",
            "trend_up",
            "trend_down",
            "pullback_long_touch",
            "pullback_short_touch",
            "pullback_long_recent",
            "pullback_short_recent",
            "continuation_long",
            "continuation_short",
        ]

        float_cols = [
            "atr",
            "vol_ma",
            "ema_intra",
            "rsi",
            "adx",
            "st_val",
            "ema_daily",
            "breakout_up_level",
            "breakout_dn_level",
        ]

        for c in int_cols:
            if c in df.columns:
                df[c] = df[c].fillna(0).astype("int")

        for c in float_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

        return df



    # =========================================================
    # Analyze (Entry Strategy)
    # =========================================================
    def analyze(self, symbol, df):
        """
        Continuation Pullback Entry
        Returns:
        - signal: "LONG" | "SHORT" | None
        - sl_price: float
        - tp_price: float
        """
        if df is None or len(df) < 5:
            return None, 0.0, 0.0

        p = self.params

        try:
            curr = df.iloc[-1]
        except Exception:
            return None, 0.0, 0.0

        signal = None

        close = float(curr.get("close", 0.0) or 0.0)
        atr = float(curr.get("atr", 0.0) or 0.0)
        adx_val = float(curr.get("adx", 0.0) or 0.0)

        # ---------------------------------------------------------
        # 1) Daily filter (mandatory)
        # ---------------------------------------------------------
        daily_ok = int(curr.get("ema_daily_ok", 0) or 0) == 1
        ema_daily = curr.get("ema_daily", np.nan)

        is_daily_uptrend = False
        is_daily_downtrend = False
        if daily_ok and (ema_daily is not None) and (not pd.isna(ema_daily)):
            ema_daily = float(ema_daily)
            is_daily_uptrend = close > ema_daily
            is_daily_downtrend = close < ema_daily

        # ---------------------------------------------------------
        # 2) ADX filter (mandatory)
        # ---------------------------------------------------------
        adx_ok = adx_val > float(p.get("adx_threshold", 0) or 0)

        # ---------------------------------------------------------
        # 3) Continuation Pullback Entry
        # ---------------------------------------------------------
        long_ok = (
            int(curr.get("trend_up", 0) or 0) == 1
            and int(curr.get("pullback_long_recent", 0) or 0) == 1
            and int(curr.get("continuation_long", 0) or 0) == 1
            and is_daily_uptrend
            and adx_ok
        )

        short_ok = (
            int(curr.get("trend_down", 0) or 0) == 1
            and int(curr.get("pullback_short_recent", 0) or 0) == 1
            and int(curr.get("continuation_short", 0) or 0) == 1
            and is_daily_downtrend
            and adx_ok
        )

        if long_ok:
            signal = "LONG"
        elif short_ok:
            signal = "SHORT"

        # ---------------------------------------------------------
        # 4) Safety: ST 방향/가격 위치 최종 확인
        # ---------------------------------------------------------
        if signal is not None:
            st_dir = int(curr.get("st_dir", 0) or 0)
            st_val = curr.get("st_val", np.nan)

            if signal == "LONG":
                if (st_dir <= 0) or (pd.notna(st_val) and float(st_val) >= close):
                    signal = None
            elif signal == "SHORT":
                if (st_dir >= 0) or (pd.notna(st_val) and float(st_val) <= close):
                    signal = None

        # ---------------------------------------------------------
        # 5) Exit: 기존 유지 (SuperTrend SL + ATR TP)
        # ---------------------------------------------------------
        sl_val = curr.get("st_val", np.nan)
        sl_price = float(sl_val) if (sl_val is not None and not pd.isna(sl_val)) else 0.0

        tp_price = 0.0
        if signal == "LONG":
            tp_price = close + (atr * float(p["atr_multiplier"]))
        elif signal == "SHORT":
            tp_price = close - (atr * float(p["atr_multiplier"]))

        return signal, float(sl_price), float(tp_price)