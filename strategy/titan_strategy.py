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
        self.__version__ = "8.1.0-LiquidityMSSPullback"

        # =====================================================================
        # [Hyperopt 대상 파라미터]
        # =====================================================================
        self.params = {
            # --- 15분봉(Intraday) 설정 ---
            "atr_period": 25,
            "atr_multiplier": 2.5,
            "adx_threshold": 16,  # (옵션) 필터로만 사용
            "rsi_upper": 71,      # (옵션) 필터로만 사용
            "rsi_lower": 36,      # (옵션) 필터로만 사용
            "vol_factor": 1.0,
            "ema_intraday": 200,
            # --- 일봉(Daily) 설정 ---
            "daily_ema": 25,

            # --- Market Structure / Liquidity 설정 ---
            "swing_len": 5,               # pivot 길이 (프랙탈)
            "context_lookback": 120,      # 최근 N봉에서 스윕/구조 이벤트 추적
            "retest_tolerance_atr": 0.25, # retest 레벨 허용오차(ATR 비율)
            "use_daily_filter": True,     # 일봉 EMA 필터 사용 여부
            "use_vol_filter": True,       # 거래량 필터 사용 여부
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
    # Market Structure Helpers (Pure, no I/O)
    # =========================================================
    def _compute_pivots(self, df: pd.DataFrame, swing_len: int):
        """
        프랙탈 피봇(high/low) 계산.
        - pivot_high: 해당 봉의 high가 좌우 swing_len 범위에서 최댓값이면 1
        - pivot_low : 해당 봉의 low 가 좌우 swing_len 범위에서 최솟값이면 1
        주의: 중앙봉 기준이라 오른쪽 미래 데이터가 필요 -> 실전/백테 공통으로
             "지표 계산 단계"에서만 만들고, 엔진이 캔들을 흘릴 때는
             이미 과거 봉의 pivot이 확정된 것으로 동작(실시간에서는 swing_len 지연 생김).
        """
        n = int(max(1, swing_len))
        w = 2 * n + 1

        # rolling center는 미래를 쓰는 "정의"지만, pivot은 원래 확정이 지연되는 개념이라
        # 이 형태가 오히려 현실적(확정까지 지연)이다.
        rh = df["high"].rolling(window=w, center=True).max()
        rl = df["low"].rolling(window=w, center=True).min()

        pivot_high = (df["high"] == rh).astype("int")
        pivot_low = (df["low"] == rl).astype("int")

        # 가장자리 NaN은 0 처리
        pivot_high = pivot_high.fillna(0).astype("int")
        pivot_low = pivot_low.fillna(0).astype("int")

        return pivot_high, pivot_low

    def calculate_indicators(self, symbol, df):
        """
        [Indicator Engine]
        - 15분봉 지표 + 일봉(리샘플) EMA 필터
        - Lookahead Bias 방지: daily_df.shift(1)

        추가:
        - Liquidity Sweep + MSS + Retest(Pullback) 구조형 진입 지표
        """
        df = self._ensure_datetime_index(df)
        df = df.copy()
        p = self.params

        # =========================================================
        # 1. Intraday Indicators (15분봉 기준)
        # =========================================================
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=p["atr_period"])
        df["vol_ma"] = df["volume"].rolling(window=20).mean()
        df["ema_intra"] = ta.ema(df["close"], length=p["ema_intraday"])
        df["rsi"] = ta.rsi(df["close"], length=14)

        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        adx_series = self._safe_pick_adx(adx_df)
        df["adx"] = adx_series if adx_series is not None else 0.0

        st = ta.supertrend(df["high"], df["low"], df["close"], length=12, multiplier=3.0)
        st_val_ser, st_dir_ser = self._safe_pick_supertrend(st)

        df["st_val"] = st_val_ser if st_val_ser is not None else 0.0
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
            df["ema_daily"] = pd.Series(ema_daily_mapped, index=df.index).fillna(0.0)
            df["ema_daily_ok"] = 1

        # =========================================================
        # 3. Market Structure / Liquidity (15m)
        # =========================================================
        swing_len = int(p.get("swing_len", 5))
        lookback = int(p.get("context_lookback", 120))
        tol_atr = float(p.get("retest_tolerance_atr", 0.25))

        # 3-1) pivots
        ph, pl = self._compute_pivots(df, swing_len=swing_len)
        df["pivot_high"] = ph
        df["pivot_low"] = pl

        # pivot price를 값으로 저장 후 ffill
        df["pivot_high_price"] = df["high"].where(df["pivot_high"] == 1, other=pd.NA)
        df["pivot_low_price"] = df["low"].where(df["pivot_low"] == 1, other=pd.NA)

        # 최근 pivot 레벨을 계속 들고감
        df["last_pivot_high"] = df["pivot_high_price"].ffill()
        df["last_pivot_low"] = df["pivot_low_price"].ffill()

        # 3-2) Liquidity sweep
        # - high sweep: high가 last_pivot_high를 뚫고 종가가 다시 아래로
        # - low sweep : low 가 last_pivot_low 를 깨고 종가가 다시 위로
        lph = df["last_pivot_high"]
        lpl = df["last_pivot_low"]

        # NaN 보호
        df["sweep_high"] = ((df["high"] > lph) & (df["close"] < lph) & lph.notna()).astype("int")
        df["sweep_low"] = ((df["low"] < lpl) & (df["close"] > lpl) & lpl.notna()).astype("int")

        # 3-3) 최근 스윕 시점(윈도우 내)
        # lookback 창에서 최근 1이 있는지 -> 더 강한 MSS 연결용
        df["recent_sweep_high"] = df["sweep_high"].rolling(lookback, min_periods=1).max().fillna(0).astype("int")
        df["recent_sweep_low"] = df["sweep_low"].rolling(lookback, min_periods=1).max().fillna(0).astype("int")

        # 3-4) MSS (Market Structure Shift)
        # bullish MSS: 최근 low sweep가 있었고, 종가가 last_pivot_high를 상향 돌파
        # bearish MSS: 최근 high sweep가 있었고, 종가가 last_pivot_low 를 하향 돌파
        df["mss_up"] = ((df["recent_sweep_low"] == 1) & (df["close"] > lph) & lph.notna()).astype("int")
        df["mss_down"] = ((df["recent_sweep_high"] == 1) & (df["close"] < lpl) & lpl.notna()).astype("int")

        # 3-5) Retest (Pullback) 트리거
        # MSS 돌파 레벨을 "가까이 되돌려 테스트" + 거부(rejection) 캔들
        # - LONG: 돌파 레벨(lph) 근처까지 내려왔다가(허용오차), 종가가 다시 위로(양봉/회복)
        # - SHORT: 돌파 레벨(lpl) 근처까지 올라갔다가, 종가가 다시 아래로
        atr = df["atr"].fillna(0.0)
        tol = atr * tol_atr

        # 거부 캔들(간단 버전): close가 open보다 유리 방향 / wick 존재는 선택
        # LONG rejection: 저점이 레벨 아래로 살짝 찍혀도 되고(close가 다시 위)
        df["retest_long"] = (
            (df["mss_up"].rolling(lookback, min_periods=1).max() == 1) &
            (df["low"] <= (lph + tol)) &
            (df["close"] >= lph) &
            (df["close"] > df["open"])
        ).fillna(False).astype("int")

        df["retest_short"] = (
            (df["mss_down"].rolling(lookback, min_periods=1).max() == 1) &
            (df["high"] >= (lpl - tol)) &
            (df["close"] <= lpl) &
            (df["close"] < df["open"])
        ).fillna(False).astype("int")

        # 엔진 dropna 방지: 구조 컬럼들도 NaN 없게
        for c in [
            "pivot_high", "pivot_low",
            "last_pivot_high", "last_pivot_low",
            "sweep_high", "sweep_low",
            "recent_sweep_high", "recent_sweep_low",
            "mss_up", "mss_down",
            "retest_long", "retest_short",
        ]:
            if c in df.columns:
                df[c] = df[c].fillna(0)

        return df

    def analyze(self, symbol, df):
        if len(df) < 200:
            return None, 0.0, 0.0

        p = self.params
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        signal = None

        # ---------------------------------------------------------
        # 1. Multi-Timeframe Filter (옵션화)
        # ---------------------------------------------------------
        daily_ok = int(curr.get("ema_daily_ok", 0)) == 1
        daily_ema_val = float(curr.get("ema_daily", 0.0)) if daily_ok else 0.0

        use_daily = bool(p.get("use_daily_filter", True))
        use_vol = bool(p.get("use_vol_filter", True))

        is_daily_uptrend = (curr["close"] > daily_ema_val) if (use_daily and daily_ok) else True
        is_daily_downtrend = (curr["close"] < daily_ema_val) if (use_daily and daily_ok) else True

        vol_ma = curr.get("vol_ma", 0)
        if pd.isna(vol_ma) or vol_ma <= 0:
            vol_ma = 0.0

        is_vol = (prev["volume"] > (vol_ma * p["vol_factor"])) if (use_vol and vol_ma > 0) else True

        # (옵션) ADX를 필터로만 유지 (원하면 off 가능)
        adx_val = curr.get("adx", 0)
        if pd.isna(adx_val):
            adx_val = 0.0
        is_trend_alive = float(adx_val) > float(p.get("adx_threshold", 0))

        # ---------------------------------------------------------
        # 2. Entry Logic (Liquidity Sweep -> MSS -> Retest Pullback)
        # ---------------------------------------------------------
        # 구조형 진입은 "retest_long/short"에서만 트리거
        retest_long = int(curr.get("retest_long", 0)) == 1
        retest_short = int(curr.get("retest_short", 0)) == 1

        # 안전장치: 구조 레벨이 유효할 때만
        last_high = curr.get("last_pivot_high", 0.0)
        last_low = curr.get("last_pivot_low", 0.0)
        level_ok = (pd.notna(last_high) and float(last_high) > 0) or (pd.notna(last_low) and float(last_low) > 0)

        # 필터 결합
        # - 기본: daily + volume 적용
        # - adx는 "옵션": 너무 타이트하면 꺼라(파라미터 adx_threshold=0으로도 가능)
        adx_filter_on = float(p.get("adx_threshold", 0)) > 0

        if level_ok:
            if retest_long and is_vol and is_daily_uptrend:
                if (not adx_filter_on) or is_trend_alive:
                    signal = "LONG"
            elif retest_short and is_vol and is_daily_downtrend:
                if (not adx_filter_on) or is_trend_alive:
                    signal = "SHORT"

        # ---------------------------------------------------------
        # 3. Exit (기존 유지: SuperTrend SL + ATR TP)
        # ---------------------------------------------------------
        sl_val = curr.get("st_val", 0.0)
        if pd.isna(sl_val):
            sl_val = 0.0
        sl_price = float(sl_val)

        atr = curr.get("atr", 0.0)
        if pd.isna(atr):
            atr = 0.0
        atr = float(atr)

        tp_price = 0.0
        if signal == "LONG":
            tp_price = float(curr["close"]) + (atr * float(p["atr_multiplier"]))
        elif signal == "SHORT":
            tp_price = float(curr["close"]) - (atr * float(p["atr_multiplier"]))

        return signal, float(sl_price), float(tp_price)
