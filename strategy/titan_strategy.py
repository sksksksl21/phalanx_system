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
        self.__version__ = "8.0.0-MultiTimeframeOpt"

        # =====================================================================
        # [Hyperopt 대상 파라미터]
        # =====================================================================
        self.params = {
            # --- 15분봉(Intraday) 설정 ---
            "atr_period": 25,
            "atr_multiplier": 2.5,
            "adx_threshold": 16,
            "rsi_upper": 71,
            "rsi_lower": 36,
            "vol_factor": 1.0,
            "ema_intraday": 200,
            # --- 일봉(Daily) 설정 ---
            "daily_ema": 25,
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

            # 너무 짧으면 일봉 EMA 신뢰 불가 → 비활성
            # (daily_len + 5) 정도 버퍼를 둬서 초기 NaN 구간 회피
            if len(daily_df) < (daily_len + 5):
                return None, False

            daily_df["ema_daily"] = ta.ema(daily_df["close"], length=daily_len)

            # 확정된 "어제" 값만 사용
            daily_df = daily_df.shift(1)

            ema_mapped = daily_df["ema_daily"].reindex(df_15m.index, method="ffill")
            return ema_mapped, True

        except Exception:
            return None, False

    def calculate_indicators(self, symbol, df):
        """
        [Indicator Engine]
        - 15분봉 지표 + 일봉(리샘플) EMA 필터
        - Lookahead Bias 방지: daily_df.shift(1)

        중요:
        - ema_daily를 NaN으로 두면 엔진의 dropna()에서 심볼이 통째로 죽을 수 있어,
          "ema_daily_ok" 플래그로 필터 활성/비활성을 분리한다.
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

        # ✅ 엔진 dropna()에 심볼이 죽지 않도록 "NaN을 만들지 않는다"
        # ✅ 대신 ema_daily_ok=0이면 analyze에서 일봉 필터를 '비활성'로 처리
        if not ok or ema_daily_mapped is None:
            df["ema_daily"] = 0.0
            df["ema_daily_ok"] = 0
        else:
            # 혹시 mapping 과정에서 일부 NaN이 남아도 엔진 dropna() 피해가도록 0으로 보정
            df["ema_daily"] = pd.Series(ema_daily_mapped, index=df.index).fillna(0.0)
            df["ema_daily_ok"] = 1

        return df

    def analyze(self, symbol, df):
        if len(df) < 200:
            return None, 0.0, 0.0

        p = self.params
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        signal = None

        # ---------------------------------------------------------
        # 1. Multi-Timeframe Filter
        # ---------------------------------------------------------
        daily_ok = int(curr.get("ema_daily_ok", 0)) == 1
        daily_ema_val = float(curr.get("ema_daily", 0.0)) if daily_ok else 0.0

        # 일봉 EMA가 유효할 때만 필터 적용
        is_daily_uptrend = (curr["close"] > daily_ema_val) if daily_ok else True
        is_daily_downtrend = (curr["close"] < daily_ema_val) if daily_ok else True

        vol_ma = curr.get("vol_ma", 0)
        if pd.isna(vol_ma) or vol_ma <= 0:
            vol_ma = 0.0

        is_vol = (prev["volume"] > (vol_ma * p["vol_factor"])) if vol_ma > 0 else False

        adx_val = curr.get("adx", 0)
        if pd.isna(adx_val):
            adx_val = 0.0
        is_trend_alive = float(adx_val) > float(p["adx_threshold"])

        # ---------------------------------------------------------
        # 2. Entry Logic
        # ---------------------------------------------------------
        prev_dir = prev.get("st_dir", 0)
        curr_dir = curr.get("st_dir", 0)
        if pd.isna(prev_dir):
            prev_dir = 0
        if pd.isna(curr_dir):
            curr_dir = 0

        st_flip_up = (prev_dir == -1) and (curr_dir == 1)
        st_flip_down = (prev_dir == 1) and (curr_dir == -1)

        if is_trend_alive:
            if st_flip_up and is_vol and is_daily_uptrend:
                if float(prev.get("rsi", 0) or 0) < float(p["rsi_upper"]):
                    signal = "LONG"
            elif st_flip_down and is_vol and is_daily_downtrend:
                if float(prev.get("rsi", 0) or 0) > float(p["rsi_lower"]):
                    signal = "SHORT"

        # ---------------------------------------------------------
        # 3. Exit (ATR Based)
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
