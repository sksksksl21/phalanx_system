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
        self.__version__ = "8.2.1-LiquidityMSSPullback-VolScoreOnly"

        # =====================================================================
        # [Hyperopt 대상 파라미터]16	4.5	8	70	38	0.9	200	15	5	60	0.2




        # =====================================================================
        self.params = {
            # --- 15분봉(Intraday) 설정 ---
            "atr_period": 16,
            "atr_multiplier": 4.5,
            "adx_threshold": 8,  # (옵션) 필터로만 사용
            "rsi_upper": 70,      # (옵션) 필터로만 사용
            "rsi_lower": 38,      # (옵션) 필터로만 사용
            "vol_factor": 0.9,    # 볼륨 신뢰도 점수 기준(게이트 아님)
            "ema_intraday": 200,

            # --- 일봉(Daily) 설정 ---
            "daily_ema": 15,

            # --- Market Structure / Liquidity 설정 ---
            "swing_len": 5,               # pivot 길이(프랙탈)
            "context_lookback": 60,      # MSS 유효기간/스윕 컨텍스트
            "retest_tolerance_atr": 0.2, # retest 레벨 허용오차(ATR 비율)
            "use_daily_filter": True,     # 일봉 EMA 필터 사용 여부
            "use_vol_filter": True,       # ✅ Gate 제거. 신뢰도 점수 계산만.

            # --- Safety (정합성/운영 안정) ---
            "use_st_dir_filter": True,    # 진입 방향과 ST 방향 불일치 시 신호 차단
        }

    def get_blacklist(self):
        return list(self.blacklist)

    def set_params(self, params):
        self.params.update(params)

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
        """
        pandas_ta.supertrend는 컬럼명이 버전에 따라 달라질 수 있어 방어적으로 pick한다.
        반환:
          - st_val: SuperTrend 값(선)
          - st_dir: 방향(+1 up, -1 down) 계열(없으면 None)
        """
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
        """
        Daily EMA는 '계산 가능할 때만' 사용한다.
        - Lookahead 방지: daily_df.shift(1)
        - 계산 불가면 (None, False) 반환
        """
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

            # 확정된 "어제" 값만 사용
            daily_df = daily_df.shift(1)

            ema_mapped = daily_df["ema_daily"].reindex(df_15m.index, method="ffill")
            return ema_mapped, True

        except Exception:
            return None, False

    # =========================================================
    # Volume = Score only (NO GATE)
    # =========================================================
    def _calc_vol_score(self, curr, prev, p):
        """
        볼륨은 Gate가 아니라 '신뢰도 점수'로만 반영한다.

        반환:
          (vol_ok_like_bool, vol_ratio, vol_score)
            - vol_ok_like_bool: 기존 기준(> vol_ma*vol_factor) 통과 여부(참고용)
            - vol_ratio: prev.volume / prev.vol_ma
            - vol_score: -1.0 ~ +1.0 (신뢰도/우선순위)
        """
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
        """
        ✅ Non-lookahead / Non-repainting(확정 지연 강제) 프랙탈 피봇 계산

        정의:
          - pivot_high_confirmed[t] = 1  <=>  (t - n)봉의 high가 [t-2n .. t] 구간에서 최댓값
          - pivot_low_confirmed[t]  = 1  <=>  (t - n)봉의 low 가 [t-2n .. t] 구간에서 최솟값

        즉, 피봇은 실제 발생 시점보다 n봉 뒤에 확정된다(실전/백테 동일).
        """
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
        """
        trigger(0/1)로부터 최근 트리거 이후 경과 봉 수(age)를 만든다.
        - 트리거가 한 번도 없으면 age=NaN
        - 트리거 봉에서 age=0, 이후 +1
        """
        t = trigger.fillna(0).astype(int).to_numpy()
        idx = np.arange(len(t), dtype=float)
        last = np.where(t == 1, idx, np.nan)
        last = pd.Series(last, index=trigger.index).ffill()
        age = pd.Series(np.arange(len(t), dtype=float), index=trigger.index) - last
        age = age.where(last.notna(), np.nan)
        return age

    def calculate_indicators(self, symbol, df):
        """
        [Indicator Engine]
        - 15분봉 지표 + 일봉(리샘플) EMA 필터
        - Lookahead Bias 방지: daily_df.shift(1)

        구조:
        - Rejection Sweep(유지) -> MSS(레벨 스냅샷) -> Retest(Pullback)
        - Pivot: 확정 지연 강제(Non-lookahead)
        - MSS: 레벨 스냅샷 고정 + 유효기간(age) 적용
        """
        df = self._ensure_datetime_index(df)
        df = df.copy()
        p = self.params

        # =========================================================
        # 1. Intraday Indicators (15분봉 기준)
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
        # 2. Daily Indicators (일봉 재가공 및 병합)
        # =========================================================
        ema_daily_mapped, ok = self._safe_ema_daily_map(df, int(p.get("daily_ema", 25)))
        if not ok or ema_daily_mapped is None:
            df["ema_daily"] = 0.0
            df["ema_daily_ok"] = 0
        else:
            df["ema_daily"] = pd.Series(ema_daily_mapped, index=df.index).astype(float).fillna(0.0)
            df["ema_daily_ok"] = 1

        # =========================================================
        # 2.5 Volume score columns (NO GATE)
        #   - prev 기준으로 스냅샷(로그와 동일하게 맞추기 위해 shift(1) 사용)
        # =========================================================
        prev_vol_ma = df["vol_ma"].shift(1)
        prev_vol = df["volume"].shift(1)
        thr = float(p.get("vol_factor", 1.0) or 1.0)

        # ratio / ok_like / score
        df["vol_ratio_prev"] = (prev_vol / prev_vol_ma.replace(0, np.nan)).astype(float)
        df["vol_ok_like_prev"] = ((prev_vol_ma > 0) & (prev_vol > (prev_vol_ma * thr))).astype("int")

        denom = max(thr, 1e-12)
        raw = (df["vol_ratio_prev"] / denom) - 1.0
        df["vol_score_prev"] = raw.clip(-1.0, 1.0).fillna(0.0).astype(float)

        # =========================================================
        # 3. Market Structure / Liquidity (15m)
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

        # 3-2) Liquidity sweep (Rejection sweep ONLY)
        df["sweep_high"] = ((df["high"] > lph) & (df["close"] < lph) & lph.notna()).astype("int")
        df["sweep_low"] = ((df["low"] < lpl) & (df["close"] > lpl) & lpl.notna()).astype("int")

        # 3-3) 최근 스윕 플래그(컨텍스트)
        df["recent_sweep_high"] = df["sweep_high"].rolling(lookback, min_periods=1).max().fillna(0).astype("int")
        df["recent_sweep_low"] = df["sweep_low"].rolling(lookback, min_periods=1).max().fillna(0).astype("int")

        # 3-4) MSS 트리거 + 레벨 스냅샷
        mss_up_trigger = (
            (df["recent_sweep_low"] == 1)
            & lph.notna()
            & (df["close"] > lph)
            & (df["close"].shift(1) <= lph.shift(1).where(lph.shift(1).notna(), lph.shift(1)))
        ).fillna(False).astype("int")

        mss_down_trigger = (
            (df["recent_sweep_high"] == 1)
            & lpl.notna()
            & (df["close"] < lpl)
            & (df["close"].shift(1) >= lpl.shift(1).where(lpl.shift(1).notna(), lpl.shift(1)))
        ).fillna(False).astype("int")

        df["mss_up_trigger"] = mss_up_trigger
        df["mss_down_trigger"] = mss_down_trigger

        df["mss_level_up_raw"] = np.where(df["mss_up_trigger"] == 1, lph.astype(float), np.nan)
        df["mss_level_down_raw"] = np.where(df["mss_down_trigger"] == 1, lpl.astype(float), np.nan)

        df["mss_age_up"] = self._age_from_triggers(df["mss_up_trigger"])
        df["mss_age_down"] = self._age_from_triggers(df["mss_down_trigger"])

        df["mss_active_up"] = ((df["mss_age_up"].notna()) & (df["mss_age_up"] <= lookback)).astype("int")
        df["mss_active_down"] = ((df["mss_age_down"].notna()) & (df["mss_age_down"] <= lookback)).astype("int")

        # 3-5) Retest
        atr = df["atr"].astype(float).fillna(0.0)
        tol = atr * tol_atr

        level_up_ffill = pd.Series(df["mss_level_up_raw"], index=df.index).ffill()
        level_down_ffill = pd.Series(df["mss_level_down_raw"], index=df.index).ffill()

        retest_long_raw = (
            (df["mss_active_up"] == 1)
            & pd.Series(level_up_ffill, index=df.index).notna()
            & (df["low"] <= (level_up_ffill + tol))
            & (df["close"] >= level_up_ffill)
            & (df["close"] > df["open"])
        ).fillna(False).astype("int")

        retest_short_raw = (
            (df["mss_active_down"] == 1)
            & pd.Series(level_down_ffill, index=df.index).notna()
            & (df["high"] >= (level_down_ffill - tol))
            & (df["close"] <= level_down_ffill)
            & (df["close"] < df["open"])
        ).fillna(False).astype("int")

        reset_up = (df["mss_up_trigger"] == 1) | (retest_long_raw == 1)
        reset_down = (df["mss_down_trigger"] == 1) | (retest_short_raw == 1)

        seg_up = (reset_up.shift(1) == True).astype(int).cumsum()
        seg_down = (reset_down.shift(1) == True).astype(int).cumsum()

        df["mss_level_up"] = df["mss_level_up_raw"].groupby(seg_up).ffill().astype(float)
        df["mss_level_down"] = df["mss_level_down_raw"].groupby(seg_down).ffill().astype(float)

        lvl_up = df["mss_level_up"]
        lvl_dn = df["mss_level_down"]

        df["retest_long"] = (
            (df["mss_active_up"] == 1)
            & lvl_up.notna()
            & (df["low"] <= (lvl_up + tol))
            & (df["close"] >= lvl_up)
            & (df["close"] > df["open"])
        ).fillna(False).astype("int")

        df["retest_short"] = (
            (df["mss_active_down"] == 1)
            & lvl_dn.notna()
            & (df["high"] >= (lvl_dn - tol))
            & (df["close"] <= lvl_dn)
            & (df["close"] < df["open"])
        ).fillna(False).astype("int")

        # dropna 방지(구조 컬럼)
        for c in [
            "pivot_high", "pivot_low",
            "pivot_high_price", "pivot_low_price",
            "last_pivot_high", "last_pivot_low",
            "sweep_high", "sweep_low",
            "recent_sweep_high", "recent_sweep_low",
            "mss_up_trigger", "mss_down_trigger",
            "mss_age_up", "mss_age_down",
            "mss_active_up", "mss_active_down",
            "mss_level_up", "mss_level_down",
            "retest_long", "retest_short",
            "vol_ok_like_prev",
        ]:
            if c in df.columns:
                if c in [
                    "pivot_high_price", "pivot_low_price",
                    "last_pivot_high", "last_pivot_low",
                    "mss_level_up", "mss_level_down",
                    "st_val", "vol_ratio_prev", "vol_score_prev",
                ]:
                    df[c] = df[c].astype(float)
                elif c in ["mss_age_up", "mss_age_down"]:
                    df[c] = df[c].astype(float)
                else:
                    df[c] = df[c].fillna(0).astype("int")

        return df

    def analyze(self, symbol, df):
        """
        ✅ Gate에서 volume 제거.
        - 신호는 구조(MSS+retest) + (daily/ADX/ST)만으로 결정
        - volume은 신뢰도(score)로만 남김(반환 스펙 유지)
        """
        if len(df) < 200:
            return None, 0.0, 0.0

        p = self.params
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        signal = None

        # ---------------------------------------------------------
        # 1) Multi-Timeframe Filter (옵션화)
        # ---------------------------------------------------------
        daily_ok = int(curr.get("ema_daily_ok", 0)) == 1
        daily_ema_val = float(curr.get("ema_daily", 0.0)) if daily_ok else 0.0

        use_daily = bool(p.get("use_daily_filter", True))

        is_daily_uptrend = (float(curr["close"]) > daily_ema_val) if (use_daily and daily_ok) else True
        is_daily_downtrend = (float(curr["close"]) < daily_ema_val) if (use_daily and daily_ok) else True

        # (옵션) ADX를 필터로만 유지
        adx_val = curr.get("adx", 0.0)
        if pd.isna(adx_val):
            adx_val = 0.0
        is_trend_alive = float(adx_val) > float(p.get("adx_threshold", 0))
        adx_filter_on = float(p.get("adx_threshold", 0)) > 0

        # ✅ Volume: 게이트 금지. (로그/스냅샷용)
        vol_ok_like, vol_ratio, vol_score = self._calc_vol_score(curr, prev, p)
        # 필요 시 외부에서 curr.get("vol_score_prev")로 신뢰도 활용

        # ---------------------------------------------------------
        # 2) Entry Logic (Structure Only)
        # ---------------------------------------------------------
        retest_long = int(curr.get("retest_long", 0)) == 1
        retest_short = int(curr.get("retest_short", 0)) == 1

        lvl_up = curr.get("mss_level_up", np.nan)
        lvl_dn = curr.get("mss_level_down", np.nan)
        level_ok = (pd.notna(lvl_up) and float(lvl_up) > 0) or (pd.notna(lvl_dn) and float(lvl_dn) > 0)

        if level_ok:
            if retest_long and is_daily_uptrend:
                if (not adx_filter_on) or is_trend_alive:
                    signal = "LONG"
            elif retest_short and is_daily_downtrend:
                if (not adx_filter_on) or is_trend_alive:
                    signal = "SHORT"

        # ---------------------------------------------------------
        # 2.5) Safety: ST 방향/가격 위치 불일치 시 신호 차단
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
        # 3) Exit (기존 유지: SuperTrend SL + ATR TP)
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
