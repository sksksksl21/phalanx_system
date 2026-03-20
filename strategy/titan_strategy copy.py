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
        self.__version__ = "8.3.0-LiquidityCHOCHBOSPullback-VolRegimeGate"

        # =====================================================================
        # [Params]
        # =====================================================================
        self.params = {
            # --- 15분봉(Intraday) 설정 ---
            "atr_period": 11,
            "atr_multiplier": 2.5,
            "adx_threshold": 1,   # (옵션) 필터로만 사용
            "rsi_upper": 80,      # (옵션) 필터로만 사용  (row에 없어서 기존값 유지)
            "rsi_lower": 33,      # (옵션) 필터로만 사용  (row에 없어서 기존값 유지)
            "vol_factor": 0.9,    # 볼륨 신뢰도 점수 기준(게이트 아님) (row에 없어서 기존값 유지)
            "ema_intraday": 200,

            # --- 일봉(Daily) 설정 ---
            "daily_ema": 30,

            # --- Market Structure / Liquidity 설정 ---
            "swing_len": 7,                 # pivot 길이(프랙탈)
            "context_lookback": 210,        # sweep/trigger 유효기간
            "retest_tolerance_atr": 0.5,    # retest 레벨 허용오차(ATR 비율)
            "use_daily_filter": False,      # 0 -> False
            "use_vol_filter": True,         # ✅ Gate 제거. 신뢰도 점수 계산만. (row에 없어서 기존값 유지)

            # --- Safety / Direction ---
            "use_st_dir_filter": False,     # 0 -> False

            # --- NEW: Structure Confirmation (CHOCH/BOS) ---
            "use_structure_confirm": False, # 0 -> False
            "structure_min_pivots": 3,

            # --- NEW: Volatility Regime Gate (no volume gate) ---
            "use_vol_regime_gate": False,   # 0 -> False
            "atr_regime_len": 110,
            "atr_regime_factor": 1.3,
            "atr_slope_gate": False,        # 0 -> False
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


    def _resolve_sl_price(self, curr, signal: str, p: dict) -> float:
        """
        SL은 반드시 포지션 방향과 일치해야 한다.
        - LONG: SL < close 인 값만 채택 (가장 타이트하게 = close에 가장 가까운 아래쪽 값 = max)
        - SHORT: SL > close 인 값만 채택 (가장 타이트하게 = close에 가장 가까운 위쪽 값 = min)

        우선순위(후보):
        1) SuperTrend(st_val) (방향 일치할 때만)
        2) CHOCH 레벨(방향 일치할 때만)  (LONG=level_down, SHORT=level_up)
        3) ATR 기반 fallback (항상 방향 일치)
        """
        try:
            close = float(curr.get("close", 0.0) or 0.0)
            if close <= 0:
                return 0.0

            # base metrics
            atr = curr.get("atr", 0.0)
            atr = float(atr) if (atr is not None and not pd.isna(atr)) else 0.0

            st_val = curr.get("st_val", np.nan)
            st = float(st_val) if (st_val is not None and not pd.isna(st_val)) else np.nan

            lvl_up = curr.get("choch_level_up", np.nan)
            lvl_dn = curr.get("choch_level_down", np.nan)
            lvl_up = float(lvl_up) if (lvl_up is not None and not pd.isna(lvl_up)) else np.nan
            lvl_dn = float(lvl_dn) if (lvl_dn is not None and not pd.isna(lvl_dn)) else np.nan

            # fallback multiplier (기본은 atr_multiplier 재사용)
            sl_atr_mult = float(p.get("sl_atr_multiplier", p.get("atr_multiplier", 1.0)) or 1.0)

            if signal == "LONG":
                cands = []

                # 1) ST (only if below close)
                if not pd.isna(st) and st < close:
                    cands.append(st)

                # 2) CHOCH down level (only if below close)
                if not pd.isna(lvl_dn) and lvl_dn > 0 and lvl_dn < close:
                    cands.append(lvl_dn)

                # 3) ATR fallback (always below close if atr>=0)
                fb = close - (atr * sl_atr_mult)
                if fb > 0 and fb < close:
                    cands.append(fb)

                if not cands:
                    return 0.0

                # LONG: close 아래에서 가장 타이트한(가장 큰) 값
                return float(max(cands))

            elif signal == "SHORT":
                cands = []

                # 1) ST (only if above close)
                if not pd.isna(st) and st > close:
                    cands.append(st)

                # 2) CHOCH up level (only if above close)
                if not pd.isna(lvl_up) and lvl_up > close:
                    cands.append(lvl_up)

                # 3) ATR fallback (always above close if atr>=0)
                fb = close + (atr * sl_atr_mult)
                if fb > close:
                    cands.append(fb)

                if not cands:
                    return 0.0

                # SHORT: close 위에서 가장 타이트한(가장 작은) 값
                return float(min(cands))

            return 0.0
        except Exception:
            return 0.0
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
        # 2.6) NEW: Volatility regime columns (ATR 기반)
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

        # 3-1.5) NEW: trend_state (피봇 2개 기반)
        ph_last, ph_prev, pl_last, pl_prev, trend_state = self._compute_trend_state_from_pivots(df, p)
        df["ph_last"] = ph_last
        df["ph_prev"] = ph_prev
        df["pl_last"] = pl_last
        df["pl_prev"] = pl_prev
        df["trend_state"] = trend_state.astype(int)

        # 3-2) Liquidity sweep (Rejection sweep ONLY)
        df["sweep_high"] = ((df["high"] > lph) & (df["close"] < lph) & lph.notna()).astype("int")
        df["sweep_low"] = ((df["low"] < lpl) & (df["close"] > lpl) & lpl.notna()).astype("int")

        # 3-3) 최근 sweep context
        df["recent_sweep_high"] = df["sweep_high"].rolling(lookback, min_periods=1).max().fillna(0).astype("int")
        df["recent_sweep_low"] = df["sweep_low"].rolling(lookback, min_periods=1).max().fillna(0).astype("int")

        # 3-4) CHOCH trigger (기존 MSS를 'CHOCH 레벨 회복'으로 승격)
        # - sweep_low 이후: close가 lph를 상향 돌파하면 CHOCH_UP
        # - sweep_high 이후: close가 lpl을 하향 돌파하면 CHOCH_DN
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

        # CHOCH 레벨 스냅샷(=돌파한 구조 레벨)
        df["choch_level_up_raw"] = np.where(df["choch_up_trigger"] == 1, lph.astype(float), np.nan)
        df["choch_level_down_raw"] = np.where(df["choch_down_trigger"] == 1, lpl.astype(float), np.nan)

        df["choch_age_up"] = self._age_from_triggers(df["choch_up_trigger"])
        df["choch_age_down"] = self._age_from_triggers(df["choch_down_trigger"])

        df["choch_active_up"] = ((df["choch_age_up"].notna()) & (df["choch_age_up"] <= lookback)).astype("int")
        df["choch_active_down"] = ((df["choch_age_down"].notna()) & (df["choch_age_down"] <= lookback)).astype("int")

        # 3-4.5) NEW: BOS confirmation proxy
        # 여기서 BOS를 "추세 상태 + CHOCH" 결합으로 강제한다.
        # - LONG: downtrend(-1) 또는 range(0)에서 CHOCH_UP 발생해야 의미가 큼
        # - SHORT: uptrend(+1) 또는 range(0)에서 CHOCH_DN 발생해야 의미가 큼
        df["bos_ok_long"] = ((df["trend_state"] <= 0) & (df["choch_active_up"] == 1)).astype("int")
        df["bos_ok_short"] = ((df["trend_state"] >= 0) & (df["choch_active_down"] == 1)).astype("int")

        # 3-5) Retest (Pullback to CHOCH level)
        atr = df["atr"].astype(float).fillna(0.0)
        tol = atr * tol_atr

        level_up_ffill = pd.Series(df["choch_level_up_raw"], index=df.index).ffill()
        level_down_ffill = pd.Series(df["choch_level_down_raw"], index=df.index).ffill()

        retest_long_raw = (
            (df["choch_active_up"] == 1)
            & pd.Series(level_up_ffill, index=df.index).notna()
            & (df["low"] <= (level_up_ffill + tol))
            & (df["close"] >= level_up_ffill)
            & (df["close"] > df["open"])
        ).fillna(False).astype("int")

        retest_short_raw = (
            (df["choch_active_down"] == 1)
            & pd.Series(level_down_ffill, index=df.index).notna()
            & (df["high"] >= (level_down_ffill - tol))
            & (df["close"] <= level_down_ffill)
            & (df["close"] < df["open"])
        ).fillna(False).astype("int")

        reset_up = (df["choch_up_trigger"] == 1) | (retest_long_raw == 1)
        reset_down = (df["choch_down_trigger"] == 1) | (retest_short_raw == 1)

        seg_up = reset_up.astype(int).cumsum()
        seg_down = reset_down.astype(int).cumsum()

        df["choch_level_up"] = df["choch_level_up_raw"].groupby(seg_up).ffill().astype(float)
        df["choch_level_down"] = df["choch_level_down_raw"].groupby(seg_down).ffill().astype(float)

        lvl_up = df["choch_level_up"]
        lvl_dn = df["choch_level_down"]

        df["retest_long"] = (
            (df["choch_active_up"] == 1)
            & lvl_up.notna()
            & (df["low"] <= (lvl_up + tol))
            & (df["close"] >= lvl_up)
            & (df["close"] > df["open"])
        ).fillna(False).astype("int")

        df["retest_short"] = (
            (df["choch_active_down"] == 1)
            & lvl_dn.notna()
            & (df["high"] >= (lvl_dn - tol))
            & (df["close"] <= lvl_dn)
            & (df["close"] < df["open"])
        ).fillna(False).astype("int")

        # =========================================================
        # Dropna 방지 / 타입 정리
        # =========================================================
        int_cols = [
            "pivot_high", "pivot_low",
            "sweep_high", "sweep_low",
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
        ✅ 네 유기적 연결(스윕→트리거→레벨스냅샷→리테스트)을 유지하면서,
        진입을 "LS + CHOCH/BOS + Volatility Regime Gate"로 강화.
        - Volume: score-only (NO GATE)
        - 구조: CHOCH 트리거 + BOS(추세상태 결합) + retest
        - 레짐: ATR 레짐(게이트)
        """
        if len(df) < 200:
            return None, 0.0, 0.0

        p = self.params
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        signal = None

        # ---------------------------------------------------------
        # 1) MTF Filter (Daily)
        # ---------------------------------------------------------
        daily_ok = int(curr.get("ema_daily_ok", 0)) == 1
        daily_ema_val = float(curr.get("ema_daily", 0.0)) if daily_ok else 0.0

        use_daily = bool(p.get("use_daily_filter", True))

        is_daily_uptrend = (float(curr["close"]) > daily_ema_val) if (use_daily and daily_ok) else True
        is_daily_downtrend = (float(curr["close"]) < daily_ema_val) if (use_daily and daily_ok) else True

        # ---------------------------------------------------------
        # 2) ADX filter (optional)
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
        # 5) Entry Logic (LS + CHOCH/BOS + Pullback)
        # ---------------------------------------------------------
        retest_long = int(curr.get("retest_long", 0)) == 1
        retest_short = int(curr.get("retest_short", 0)) == 1

        lvl_up = curr.get("choch_level_up", np.nan)
        lvl_dn = curr.get("choch_level_down", np.nan)
        level_ok = (pd.notna(lvl_up) and float(lvl_up) > 0) or (pd.notna(lvl_dn) and float(lvl_dn) > 0)

        # 구조 확인(옵션)
        use_structure = bool(p.get("use_structure_confirm", True))
        bos_ok_long = (int(curr.get("bos_ok_long", 0)) == 1) if use_structure else True
        bos_ok_short = (int(curr.get("bos_ok_short", 0)) == 1) if use_structure else True

        if level_ok and vol_gate_ok:
            if retest_long and is_daily_uptrend and bos_ok_long:
                if (not adx_filter_on) or is_trend_alive:
                    signal = "LONG"
            elif retest_short and is_daily_downtrend and bos_ok_short:
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
        # 7) Exit (FIX): 방향 일치 SL + ATR TP
        # ---------------------------------------------------------
        sl_price = 0.0
        if signal is not None:
            sl_price = float(self._resolve_sl_price(curr, signal, p) or 0.0)

        atr = curr.get("atr", 0.0)
        atr = float(atr) if (atr is not None and not pd.isna(atr)) else 0.0

        tp_price = 0.0
        if signal == "LONG":
            tp_price = float(curr["close"]) + (atr * float(p["atr_multiplier"]))
        elif signal == "SHORT":
            tp_price = float(curr["close"]) - (atr * float(p["atr_multiplier"]))

        return signal, float(sl_price), float(tp_price)