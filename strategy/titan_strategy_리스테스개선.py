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
        self.__version__ = "8.4.0-LiquidityCHOCHBOSPullback-EntryQuality"

        # =====================================================================
        # [Params]18,2.25,18,25,3,150,0.5,3
        # =====================================================================
        self.params = {
            # --- Core ---
            "atr_period": 18,
            "atr_multiplier": 2.25,
            "adx_threshold": 18,
            "daily_ema": 25,
            "atr_regime_factor": 1.0,

            # --- Structure / Retest geometry ---
            "swing_len": 3,
            "context_lookback": 150,
            "retest_tolerance_atr": 0.6,
            "structure_min_pivots": 3,

            # --- Entry freshness ---
            "fresh_retest_bars_long": 4,
            "fresh_retest_bars_short": 4,
            "retest_max_bars_long": 4,
            "retest_max_bars_short": 4,

            # --- Entry score thresholds ---
            "entry_score_long_min": 2,
            "entry_score_short_min": 2,

            # --- Quality filters ---
            "sweep_max_body_ratio": 0.5,
            "sweep_reclaim_min": 0.5,
            "retest_body_min": 0.35,
            "retest_reclaim_min": 0.35,

            # --- Fixed toggles ---
            "use_daily_filter": True,
            "use_vol_filter": True,
            "use_st_dir_filter": True,
            "use_structure_confirm": True,
            "use_vol_regime_gate": True,
            "atr_slope_gate": True,
            "use_entry_scoring": True,

            # --- Fixed non-search params ---
            "rsi_upper": 67,
            "rsi_lower": 30,
            "vol_factor": 0.8,
            "atr_regime_len": 50,
            "ema_intraday": 200,
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
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=p["atr_period"])
        df["vol_ma"] = df["volume"].rolling(window=20, min_periods=1).mean()
        df["ema_intra"] = ta.ema(df["close"], length=p["ema_intraday"])
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
        # 2.5) Volume score columns (NO GATE) - prev 스냅샷
        # =========================================================
        prev_vol_ma = df["vol_ma"].shift(1)
        prev_vol = df["volume"].shift(1)
        thr = float(p.get("vol_factor", 1.0) or 1.0)

        df["vol_ratio_prev"] = (prev_vol / prev_vol_ma.replace(0, np.nan)).astype(float)
        df["vol_ok_like_prev"] = ((prev_vol_ma > 0) & (prev_vol > (prev_vol_ma * thr))).astype("int")

        denom = max(thr, 1e-12)
        raw = (df["vol_ratio_prev"] / denom) - 1.0
        df["vol_score_prev"] = raw.clip(-1.0, 1.0).fillna(0.0).astype(float)

        # =========================================================
        # 2.6) Volatility regime columns (ATR 기반)
        # =========================================================
        atr_reg_len = int(p.get("atr_regime_len", 50))
        df["atr_ma"] = df["atr"].rolling(window=atr_reg_len, min_periods=max(2, atr_reg_len // 3)).mean()
        df["atr_ratio"] = (df["atr"] / df["atr_ma"].replace(0, np.nan)).astype(float).fillna(0.0)
        df["atr_up"] = (df["atr"] > df["atr"].shift(1)).astype("int")

        # =========================================================
        # 3) Market Structure / Liquidity (15m)
        # =========================================================
        swing_len = int(p.get("swing_len", 5))
        lookback = int(p.get("context_lookback", 120))
        tol_atr = float(p.get("retest_tolerance_atr", 0.25))

        # 3-1) pivots (확정 지연 강제)
        ph_c, pl_c = self._compute_pivots_confirmed(df, swing_len=swing_len)
        df["pivot_high"] = ph_c
        df["pivot_low"] = pl_c

        n = int(max(1, swing_len))
        df["pivot_high_price"] = np.where(df["pivot_high"] == 1, df["high"].shift(n).astype(float), np.nan)
        df["pivot_low_price"] = np.where(df["pivot_low"] == 1, df["low"].shift(n).astype(float), np.nan)

        df["last_pivot_high"] = pd.Series(df["pivot_high_price"], index=df.index).ffill()
        df["last_pivot_low"] = pd.Series(df["pivot_low_price"], index=df.index).ffill()

        lph = df["last_pivot_high"].astype(float)
        lpl = df["last_pivot_low"].astype(float)

        # 3-1.5) trend_state (피봇 2개 기반)
        ph_last, ph_prev, pl_last, pl_prev, trend_state = self._compute_trend_state_from_pivots(df, p)
        df["ph_last"] = ph_last
        df["ph_prev"] = ph_prev
        df["pl_last"] = pl_last
        df["pl_prev"] = pl_prev
        df["trend_state"] = trend_state.astype(int)

        # 3-2) Candle anatomy for quality filters
        rng = (df["high"] - df["low"]).replace(0, np.nan)
        real_body = (df["close"] - df["open"]).abs()
        upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
        lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

        df["body_ratio"] = (real_body / rng).clip(lower=0.0, upper=1.0).fillna(0.0).astype(float)
        df["upper_wick_ratio"] = (upper_wick / rng).clip(lower=0.0, upper=1.0).fillna(0.0).astype(float)
        df["lower_wick_ratio"] = (lower_wick / rng).clip(lower=0.0, upper=1.0).fillna(0.0).astype(float)
        df["close_pos"] = ((df["close"] - df["low"]) / rng).clip(lower=0.0, upper=1.0).fillna(0.5).astype(float)

        # 3-3) Liquidity sweep (Rejection sweep ONLY)
        df["sweep_high"] = ((df["high"] > lph) & (df["close"] < lph) & lph.notna()).astype("int")
        df["sweep_low"] = ((df["low"] < lpl) & (df["close"] > lpl) & lpl.notna()).astype("int")

        sweep_body_max = float(p.get("sweep_max_body_ratio", 0.45) or 0.45)
        sweep_reclaim_min = float(p.get("sweep_reclaim_min", 0.55) or 0.55)

        df["sweep_high_q"] = (
            (df["sweep_high"] == 1)
            & (df["body_ratio"] <= sweep_body_max)
            & (df["close_pos"] <= (1.0 - sweep_reclaim_min))
        ).astype("int")

        df["sweep_low_q"] = (
            (df["sweep_low"] == 1)
            & (df["body_ratio"] <= sweep_body_max)
            & (df["close_pos"] >= sweep_reclaim_min)
        ).astype("int")

        # 3-4) 최근 sweep context (quality sweep 기준)
        df["recent_sweep_high"] = df["sweep_high_q"].rolling(lookback, min_periods=1).max().fillna(0).astype("int")
        df["recent_sweep_low"] = df["sweep_low_q"].rolling(lookback, min_periods=1).max().fillna(0).astype("int")

        # 3-5) CHOCH trigger
        choch_up_trigger = (
            (df["recent_sweep_low"] == 1)
            & lph.notna()
            & (df["close"] > lph)
            & (df["close"].shift(1) <= lph.shift(1).where(lph.shift(1).notna(), lph.shift(1)))
        ).fillna(False).astype("int")

        choch_down_trigger = (
            (df["recent_sweep_high"] == 1)
            & lpl.notna()
            & (df["close"] < lpl)
            & (df["close"].shift(1) >= lpl.shift(1).where(lpl.shift(1).notna(), lpl.shift(1)))
        ).fillna(False).astype("int")

        df["choch_up_trigger"] = choch_up_trigger
        df["choch_down_trigger"] = choch_down_trigger

        df["choch_level_up_raw"] = np.where(df["choch_up_trigger"] == 1, lph.astype(float), np.nan)
        df["choch_level_down_raw"] = np.where(df["choch_down_trigger"] == 1, lpl.astype(float), np.nan)

        df["choch_age_up"] = self._age_from_triggers(df["choch_up_trigger"])
        df["choch_age_down"] = self._age_from_triggers(df["choch_down_trigger"])

        df["choch_active_up"] = ((df["choch_age_up"].notna()) & (df["choch_age_up"] <= lookback)).astype("int")
        df["choch_active_down"] = ((df["choch_age_down"].notna()) & (df["choch_age_down"] <= lookback)).astype("int")

        # 3-6) BOS confirmation proxy
        df["bos_ok_long"] = ((df["trend_state"] <= 0) & (df["choch_active_up"] == 1)).astype("int")
        df["bos_ok_short"] = ((df["trend_state"] >= 0) & (df["choch_active_down"] == 1)).astype("int")

        # 3-7) Retest (Pullback to CHOCH level)
        atr = df["atr"].astype(float).fillna(0.0)
        tol = atr * tol_atr

        df["choch_level_up"] = pd.Series(df["choch_level_up_raw"], index=df.index).ffill().astype(float)
        df["choch_level_down"] = pd.Series(df["choch_level_down_raw"], index=df.index).ffill().astype(float)

        lvl_up = df["choch_level_up"]
        lvl_dn = df["choch_level_down"]

        max_long = int(p.get("retest_max_bars_long", 8))
        max_short = int(p.get("retest_max_bars_short", 6))
        body_min = float(p.get("retest_body_min", 0.20) or 0.20)
        reclaim_min = float(p.get("retest_reclaim_min", 0.55) or 0.55)

        long_touch_ok = lvl_up.notna() & (df["low"] >= (lvl_up - tol)) & (df["low"] <= (lvl_up + tol))
        short_touch_ok = lvl_dn.notna() & (df["high"] <= (lvl_dn + tol)) & (df["high"] >= (lvl_dn - tol))

        bull_reclaim_ok = (
            (df["close"] >= lvl_up)
            & (df["close"] > df["open"])
            & (df["body_ratio"] >= body_min)
            & (df["close_pos"] >= reclaim_min)
        )

        bear_reclaim_ok = (
            (df["close"] <= lvl_dn)
            & (df["close"] < df["open"])
            & (df["body_ratio"] >= body_min)
            & (df["close_pos"] <= (1.0 - reclaim_min))
        )

        df["retest_long"] = (
            (df["choch_active_up"] == 1)
            & df["choch_age_up"].notna()
            & (df["choch_age_up"] >= 1)
            & (df["choch_age_up"] <= max_long)
            & long_touch_ok
            & bull_reclaim_ok
        ).fillna(False).astype("int")

        df["retest_short"] = (
            (df["choch_active_down"] == 1)
            & df["choch_age_down"].notna()
            & (df["choch_age_down"] >= 1)
            & (df["choch_age_down"] <= max_short)
            & short_touch_ok
            & bear_reclaim_ok
        ).fillna(False).astype("int")

        # 3-8) Entry score
        df["entry_score_long"] = df.apply(lambda row: self._calc_entry_score(row, p, "LONG"), axis=1).astype(float)
        df["entry_score_short"] = df.apply(lambda row: self._calc_entry_score(row, p, "SHORT"), axis=1).astype(float)

        # =========================================================
        # Dropna 방지 / 타입 정리
        # =========================================================
        int_cols = [
            "pivot_high", "pivot_low",
            "sweep_high", "sweep_low",
            "sweep_high_q", "sweep_low_q",
            "recent_sweep_high", "recent_sweep_low",
            "choch_up_trigger", "choch_down_trigger",
            "choch_active_up", "choch_active_down",
            "bos_ok_long", "bos_ok_short",
            "retest_long", "retest_short",
            "vol_ok_like_prev",
            "atr_up",
            "trend_state",
        ]

        float_cols = [
            "pivot_high_price", "pivot_low_price",
            "last_pivot_high", "last_pivot_low",
            "choch_level_up", "choch_level_down",
            "st_val", "vol_ratio_prev", "vol_score_prev",
            "atr_ma", "atr_ratio",
            "ph_last", "ph_prev", "pl_last", "pl_prev",
            "choch_age_up", "choch_age_down",
            "ema_daily", "adx", "atr", "ema_intra", "rsi",
            "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "close_pos",
            "entry_score_long", "entry_score_short",
        ]

        for c in int_cols:
            if c in df.columns:
                df[c] = df[c].fillna(0).astype("int")

        for c in float_cols:
            if c in df.columns:
                df[c] = df[c].astype(float)

        if "ema_daily_ok" in df.columns:
            df["ema_daily_ok"] = df["ema_daily_ok"].fillna(0).astype("int")

        if "st_dir" in df.columns:
            df["st_dir"] = df["st_dir"].fillna(0).astype("int")

        return df
    # =========================================================
    # Analyze (Entry Strategy)
    # =========================================================
    def analyze(self, symbol, df):
        """
        TitanStrategy core signal generator.
        Returns:
        - signal: "LONG" | "SHORT" | None
        - sl_price: float
        - tp_price: float
        """
        if df is None or len(df) < 5:
            return None, 0.0, 0.0

        p = self.params

        # -------------------------
        # Slice / Current / Prev
        # -------------------------
        try:
            curr = df.iloc[-1]
            prev = df.iloc[-2]
        except Exception:
            return None, 0.0, 0.0

        signal = None

        # ---------------------------------------------------------
        # 1) Daily Trend Filter (EMA Daily)
        # ---------------------------------------------------------
        is_daily_uptrend = True
        is_daily_downtrend = True

        if bool(p.get("use_daily_filter", True)):
            daily_ok = int(curr.get("ema_daily_ok", 0) or 0) == 1
            ema_daily = curr.get("ema_daily", np.nan)

            if daily_ok and (ema_daily is not None) and (not pd.isna(ema_daily)):
                close = float(curr["close"])
                ema_daily = float(ema_daily)
                is_daily_uptrend = close > ema_daily
                is_daily_downtrend = close < ema_daily

        # ---------------------------------------------------------
        # 2) ADX (optional filter)
        # ---------------------------------------------------------
        adx_val = curr.get("adx", 0.0)
        if pd.isna(adx_val):
            adx_val = 0.0

        is_trend_alive = float(adx_val) > float(p.get("adx_threshold", 0))
        adx_filter_on = float(p.get("adx_threshold", 0)) > 0

        # ---------------------------------------------------------
        # 3) Volume (score-only)
        # ---------------------------------------------------------
        _vol_ok_like, _vol_ratio, _vol_score = self._calc_vol_score(curr, prev, p)

        # ---------------------------------------------------------
        # 4) Volatility Regime Gate (ATR 기반)
        # ---------------------------------------------------------
        vol_gate_ok = True
        if bool(p.get("use_vol_regime_gate", True)):
            atr_ratio = float(curr.get("atr_ratio", 0.0) or 0.0)
            need = float(p.get("atr_regime_factor", 1.0) or 1.0)
            vol_gate_ok = atr_ratio > need

            if bool(p.get("atr_slope_gate", True)):
                vol_gate_ok = vol_gate_ok and (int(curr.get("atr_up", 0)) == 1)

        # ---------------------------------------------------------
        # 5) Entry Logic (LS + CHOCH/BOS + Pullback + Quality)
        # ---------------------------------------------------------
        retest_long = int(curr.get("retest_long", 0)) == 1
        retest_short = int(curr.get("retest_short", 0)) == 1

        lvl_up = curr.get("choch_level_up", np.nan)
        lvl_dn = curr.get("choch_level_down", np.nan)
        level_ok_long = pd.notna(lvl_up) and float(lvl_up) > 0
        level_ok_short = pd.notna(lvl_dn) and float(lvl_dn) > 0

        use_structure = bool(p.get("use_structure_confirm", True))
        bos_ok_long = (int(curr.get("bos_ok_long", 0)) == 1) if use_structure else True
        bos_ok_short = (int(curr.get("bos_ok_short", 0)) == 1) if use_structure else True

        entry_score_long = curr.get("entry_score_long", np.nan)
        entry_score_short = curr.get("entry_score_short", np.nan)

        if pd.isna(entry_score_long):
            entry_score_long = self._calc_entry_score(curr, p, "LONG")
        else:
            entry_score_long = float(entry_score_long)

        if pd.isna(entry_score_short):
            entry_score_short = self._calc_entry_score(curr, p, "SHORT")
        else:
            entry_score_short = float(entry_score_short)

        long_score_ok = (not bool(p.get("use_entry_scoring", True))) or (
            entry_score_long >= float(p.get("entry_score_long_min", 2.15) or 2.15)
        )
        short_score_ok = (not bool(p.get("use_entry_scoring", True))) or (
            entry_score_short >= float(p.get("entry_score_short_min", 2.30) or 2.30)
        )

        # NOTE: vol_gate_ok 계산은 유지하되, 신호 자체는 entry_score에 반영되므로 하드 차단엔 쓰지 않는다.
        _ = vol_gate_ok

        if level_ok_long and retest_long and is_daily_uptrend and bos_ok_long and long_score_ok:
            if (not adx_filter_on) or is_trend_alive:
                signal = "LONG"
        elif level_ok_short and retest_short and is_daily_downtrend and bos_ok_short and short_score_ok:
            if (not adx_filter_on) or is_trend_alive:
                signal = "SHORT"

        # ---------------------------------------------------------
        # 6) Safety: SuperTrend 방향/가격 위치 불일치 시 차단
        # ---------------------------------------------------------
        if signal is not None and bool(p.get("use_st_dir_filter", True)):
            st_dir = int(curr.get("st_dir", 0))
            st_val = curr.get("st_val", np.nan)
            close = float(curr["close"])

            if signal == "LONG":
                if (st_dir <= 0) or (pd.notna(st_val) and float(st_val) >= close):
                    signal = None
            elif signal == "SHORT":
                if (st_dir >= 0) or (pd.notna(st_val) and float(st_val) <= close):
                    signal = None

        # ---------------------------------------------------------
        # 7) Exit (기존 유지: SuperTrend SL + ATR TP)
        # ---------------------------------------------------------
        sl_val = curr.get("st_val", np.nan)
        sl_price = float(sl_val) if (sl_val is not None and not pd.isna(sl_val)) else 0.0

        atr = curr.get("atr", 0.0)
        atr = float(atr) if not pd.isna(atr) else 0.0

        tp_price = 0.0
        if signal == "LONG":
            tp_price = float(curr["close"]) + (atr * float(p["atr_multiplier"]))
        elif signal == "SHORT":
            tp_price = float(curr["close"]) - (atr * float(p["atr_multiplier"]))

        return signal, float(sl_price), float(tp_price)