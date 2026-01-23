import pandas as pd
import pandas_ta as ta

class TitanStrategy:
    def __init__(self):
        # =====================================================================
        # [Universe / Symbol Policy]
        # - 메이저 코인 집합: 시스템이 "체급"을 구분하거나, 특정 룰을 다르게 적용할 때 기준으로 쓰기 좋음
        # - 현재 코드에서는 이 major_coins를 직접 사용하진 않지만,
        #   (1) 향후 메이저/알트 분리 로직 (2) 리스크/레버리지 차등 (3) 필터링 등에 쓰는 기반 데이터
        # =====================================================================
        self.major_coins = {
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT',
            'XRP/USDT', 'ADA/USDT', 'AVAX/USDT'
        }

        # =====================================================================
        # [Blacklist]
        # - 외부(엔진/운영자/리스크 모듈)에서 주입되는 "거래 금지 심볼" 저장소
        # - 전략은 판단만 해야 하므로, blacklist 자체를 "읽기/노출"하는 역할까지만 수행
        # =====================================================================
        self.blacklist = set()

        # =====================================================================
        # [Versioning]
        # - 전략 진화 추적용 버전 문자열
        # - 로그/백테스트 결과에 version이 박히면, 같은 조건에서 재현성(Deterministic)을 확보하기 쉬움
        # =====================================================================
        self.__version__ = "8.0.0-MultiTimeframeOpt"

        # =====================================================================
        # [Hyperopt 대상 파라미터]
        # - 핵심: 15분봉(진입 타이밍) + 일봉(대세 필터)을 동시에 최적화하기 위한 파라미터 묶음
        # - 실전/백테가 동일 strategy 파일을 참조한다는 전제에서, "튜닝=파라미터만" 바뀌도록 구성
        # =====================================================================
        self.params = {
            # --- 15분봉(Intraday) 설정 ---
            'atr_period': 25,        # ATR 계산 기간(변동성 추정의 스케일)
            'atr_multiplier': 2.5,   # TP 거리 = ATR * multiplier (변동성 기반 목표가)
            'adx_threshold': 16,     # 추세 강도 하한선(너무 횡보면 진입 억제)
            'rsi_upper': 71,         # LONG 진입 전 RSI 상단 제한(과열 진입 방지 성격)
            'rsi_lower': 36,         # SHORT 진입 전 RSI 하단 제한(과매도 진입 방지 성격)
            'vol_factor': 1.0,       # 거래량 필터 배수(평균 대비 n배 이상일 때만)
            'ema_intraday': 200,     # 15분봉 추세선(현재는 계산만 하고 Entry에서 직접 쓰진 않음)

            # --- 일봉(Daily) 설정 (NEW) ---
            'daily_ema': 25          # 일봉 EMA 기간(대세 필터의 민감도)
        }

    def get_blacklist(self):
        # =====================================================================
        # - 엔진/외부가 blacklist 상태를 조회할 수 있도록 리스트로 반환
        # - set은 JSON 직렬화/표준 출력에 불편하므로 list로 변환
        # =====================================================================
        return list(self.blacklist)

    def set_params(self, params):
        # =====================================================================
        # - 하이퍼옵트/운영 파라미터 주입 포인트
        # - self.params.update(...) 형태라, 일부 키만 덮어써도 동작
        # =====================================================================
        self.params.update(params)

    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [Safety Guard]
        - df.resample('D')를 사용하기 위해 DatetimeIndex 보장
        - 엔진이 이미 DatetimeIndex로 넘겨주는 것이 베스트지만,
          실전에서 누락되면 전략이 터지므로 최소 방어를 둔다.
        """
        df = df.copy()

        # (A) 이미 DatetimeIndex면 그대로 사용
        if isinstance(df.index, pd.DatetimeIndex):
            # 혹시 정렬이 깨져있으면 정렬 (resample/ffill 안정성)
            if not df.index.is_monotonic_increasing:
                df = df.sort_index()
            return df

        # (B) timestamp 컬럼이 있으면 인덱스로 승격
        if 'timestamp' in df.columns:
            # Data Naming Standardization: 시스템 표준 컬럼명에 timestamp가 포함될 수 있음
            # (밀리초 epoch이든, 문자열이든 pandas가 최대한 해석)
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=False)
            df = df.set_index('timestamp')
            df = df[~df.index.isna()]  # 변환 실패 제거
            df = df.sort_index()
            return df

        # (C) 최후의 방어: 인덱스를 datetime으로 변환 시도
        try:
            df.index = pd.to_datetime(df.index, errors='coerce', utc=False)
            df = df[~df.index.isna()]
            df = df.sort_index()
        except Exception:
            # 변환 불가면 resample을 포기해야 하므로,
            # calculate_indicators에서 일봉 지표를 안전하게 무시하도록 처리할 수밖에 없음
            pass

        return df

    def _safe_pick_adx(self, adx_df: pd.DataFrame):
        """
        [Indicator Robustness]
        - pandas_ta.adx() 반환 컬럼 순서가 환경에 따라 달라질 수 있으므로,
          'ADX_'로 시작하는 컬럼을 우선 선택한다.
        """
        if adx_df is None or not isinstance(adx_df, pd.DataFrame) or adx_df.empty:
            return None

        # pandas_ta 관례: 'ADX_14', 'DMP_14', 'DMN_14'
        adx_cols = [c for c in adx_df.columns if str(c).upper().startswith('ADX_')]
        if adx_cols:
            return adx_df[adx_cols[0]]

        # 그래도 없으면 기존 로직처럼 "첫 컬럼" fallback (최후 방어)
        return adx_df[adx_df.columns[0]]

    def _safe_pick_supertrend(self, st_df: pd.DataFrame):
        """
        [Indicator Robustness]
        - pandas_ta.supertrend() 반환 컬럼 순서/구성이 환경에 따라 달라질 수 있으므로,
          iloc 기반이 아니라 'SUPERT' 계열 컬럼명 기반으로 선택한다.
        - 일반적으로 다음이 존재:
          * SUPERT_...   : SuperTrend 라인(가격 레벨)
          * SUPERTd_...  : 방향(+1/-1)
          * SUPERTl_... / SUPERTs_... : 롱/숏 라인(옵션에 따라)
        """
        if st_df is None or not isinstance(st_df, pd.DataFrame) or st_df.empty:
            return None, None

        cols = [str(c) for c in st_df.columns]

        # 방향 컬럼 우선 선택: SUPERTd_
        dir_col = None
        for c in cols:
            if c.upper().startswith('SUPERTD_'):
                dir_col = c
                break

        # 값(라인) 컬럼 선택: SUPERT_ 우선
        val_col = None
        for c in cols:
            if c.upper().startswith('SUPERT_'):
                val_col = c
                break

        # 만약 SUPERT_가 없고, 롱/숏 라인만 있다면 롱/숏 둘 중 하나를 대표값으로 사용 (최후 방어)
        if val_col is None:
            for c in cols:
                if c.upper().startswith('SUPERTL_'):
                    val_col = c
                    break
        if val_col is None:
            for c in cols:
                if c.upper().startswith('SUPERTS_'):
                    val_col = c
                    break

        # 실제 컬럼명이 str로 바뀌었을 수 있으니, st_df.columns에서 원본 매칭
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

    def calculate_indicators(self, symbol, df):
        """
        [Indicator Engine]
        - 15분봉 지표 + (15분봉을 일봉으로 재구성한) 일봉 지표를 동시에 계산
        - 핵심 포인트:
          (1) Intraday(15m)에서 진입 타이밍을 잡고
          (2) Daily(1D)로 대세 방향을 필터링
          (3) Lookahead Bias 방지를 위해 daily_df.shift(1)로 "확정된 어제 지표"만 사용
        """
        # ---------------------------------------------------------
        # [Patch] DatetimeIndex 보장 (resample 안정화)
        # - df.index가 datetime이 아니면 resample이 즉시 에러
        # - 엔진이 보장하면 가장 좋지만, 실전 방어를 위해 전략 내부에서도 최소 보강
        # ---------------------------------------------------------
        df = self._ensure_datetime_index(df)
        df = df.copy()
        p = self.params

        # =========================================================
        # 1. Intraday Indicators (15분봉 기준)
        # =========================================================

        # [ATR] 변동성 기반 거리 추정
        # - TP 계산에 사용됨: tp = close ± ATR*multiplier
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=p['atr_period'])

        # [Volume MA] 거래량 평균 (20개 15분봉 = 5시간)
        # - "최근 평균 거래량 대비 폭발했는가"를 보기 위한 기준선
        df['vol_ma'] = df['volume'].rolling(window=20).mean()

        # [EMA(15m)] 장기 추세선(15분봉 200개 = 50시간)
        # - 현재 엔트리 조건에는 직접 쓰지 않지만,
        #   추후 "가격이 EMA 위/아래" 같은 추가 필터로 확장 가능
        df['ema_intra'] = ta.ema(df['close'], length=p['ema_intraday'])

        # [RSI] 과열/과매도 상태(추세 진입 전 과도한 위치에서 들어가는 것 방지용)
        df['rsi'] = ta.rsi(df['close'], length=14)

        # [ADX] 추세 강도(횡보/노이즈 구간 진입 억제용)
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        # pandas_ta는 컬럼명이 ADX_14 같은 형태로 나오므로 첫 컬럼을 사용
        # [Patch] 컬럼 순서 의존(iloc) 제거 -> ADX_ 컬럼 우선 선택
        adx_series = self._safe_pick_adx(adx_df)
        df['adx'] = adx_series if adx_series is not None else 0

        # [SuperTrend] 방향(st_dir) + 라인(st_val)
        # - 방향 전환(st flip)을 "진입 트리거"로 사용
        # - st_val은 "손절선(SL)"로 사용 (추세 따라 이동하는 동적 SL)
        st = ta.supertrend(df['high'], df['low'], df['close'], length=12, multiplier=3.0)

        # [Patch] iloc 기반(컬럼 순서 의존) 제거 -> 컬럼명 기반 선택
        st_val_ser, st_dir_ser = self._safe_pick_supertrend(st)

        if st_val_ser is not None:
            df['st_val'] = st_val_ser  # SuperTrend 라인 값(가격 레벨)
        else:
            # 지표 계산 실패 시 안전장치: 값 0으로 세팅
            df['st_val'] = 0

        if st_dir_ser is not None:
            # 방향은 +1/-1 형태가 일반적이나 float로 들어올 수 있어 정규화
            # (예: 1.0, -1.0) -> (1, -1)
            df['st_dir'] = st_dir_ser.astype('float').round(0).astype('int')
        else:
            df['st_dir'] = 0

        # =========================================================
        # 2. Daily Indicators (일봉 재가공 및 병합)
        # =========================================================

        # (1) 15분봉 데이터를 일봉으로 합침 (Resampling)
        # - 하루 단위로 O/H/L/C를 만들고, 그 위에서 EMA를 계산
        # - 주의: df.index가 DatetimeIndex여야 resample('D')가 정상 동작
        #
        # [Patch] df.index가 datetime 변환에 실패한 경우(resample 불가)를 대비해 예외 방어.
        try:
            daily_df = df.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            })

            # (2) 일봉상 EMA 계산 (대세 필터)
            daily_df['ema_daily'] = ta.ema(daily_df['close'], length=p['daily_ema'])

            # (3) [핵심] Lookahead Bias 방지
            # - 오늘 하루가 끝나기 전에는 "오늘 일봉 EMA"가 확정되지 않음
            # - shift(1)을 해서 "어제 일봉 EMA"만 오늘 00:00부터 사용하도록 강제
            daily_df = daily_df.shift(1)

            # (4) 15분봉 인덱스에 맞춰 일봉 EMA를 forward-fill로 매핑
            # - 하루 동안은 같은 '어제 EMA'가 유지됨
            df['ema_daily'] = daily_df['ema_daily'].reindex(df.index, method='ffill')

        except Exception:
            # resample 실패 시:
            # - 일봉 필터가 깨지면 전략이 중단되는 것보단,
            #   "일봉 필터를 일시적으로 무시"하는 현재 설계 철학(거래 0 방지)을 따른다.
            df['ema_daily'] = float('nan')

        return df

    def analyze(self, symbol, df):
        # =====================================================================
        # [Entry/Exit Decision]
        # - 입력 df는 이미 calculate_indicators로 지표 계산이 끝난 상태라고 가정
        # - 최소 길이 200 제한:
        #   * ema_intra(200) 같은 장기 지표 워밍업 보장 목적
        # =====================================================================
        if len(df) < 200:
            return None, 0.0, 0.0

        p = self.params

        # [현재 봉 / 이전 봉] 비교 기반 트리거(방향 전환/거래량 조건)를 위해 -1/-2 사용
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        signal = None

        # ---------------------------------------------------------
        # 1. Multi-Timeframe Filter
        # ---------------------------------------------------------

        # (1) 일봉 필터: 가격이 일봉 EMA 위/아래인지로 대세 방향을 판별
        # - ema_daily가 NaN이면(초기 구간/결측) 필터를 무시하도록 설계
        daily_ema_val = curr.get('ema_daily', 0)
        if pd.isna(daily_ema_val):
            daily_ema_val = 0

        # - ema_daily 값이 존재하면:
        #   LONG는 close > ema_daily 일 때만,
        #   SHORT는 close < ema_daily 일 때만 허용
        # - ema_daily 값이 없으면(True)로 통과시켜 "데이터 부족 때문에 거래가 0"이 되는 상황을 완화
        #
        # [주의] 이 설계는 "필터 결측 시 대세 필터가 무력화"될 수 있음.
        #       다만 사용자의 의도(거래 0 회피)를 유지하기 위해 그대로 둔다.
        is_daily_uptrend = curr['close'] > daily_ema_val if daily_ema_val > 0 else True
        is_daily_downtrend = curr['close'] < daily_ema_val if daily_ema_val > 0 else True

        # (2) 15분봉 필터: 거래량 & ADX(추세 강도)

        # - vol_ma가 0/NaN일 수 있으니 get으로 방어
        vol_ma = curr.get('vol_ma', 0)
        if pd.isna(vol_ma) or vol_ma <= 0:
            vol_ma = 0

        # - 거래량 폭발 조건:
        #   * 여기서는 "이전 봉(prev)의 volume"을 사용
        #   * 해석: "방향 전환이 확정된 시점" 직전 캔들의 거래량이 평균 대비 컸는지 확인
        #
        # [Patch] vol_ma가 0이면 비교가 무의미해지므로, 이 경우 거래량 필터를 완화(True)하지 않고
        #        "폭발 판정 불가"로 보고 False 처리하여 이상 진입을 줄인다.
        if vol_ma > 0:
            is_vol = prev['volume'] > (vol_ma * p['vol_factor'])
        else:
            is_vol = False

        # - 추세가 살아 있는가:
        #   * ADX가 threshold보다 높아야만 진입 로직이 활성화됨
        adx_val = curr.get('adx', 0)
        if pd.isna(adx_val):
            adx_val = 0
        is_trend_alive = adx_val > p['adx_threshold']

        # ---------------------------------------------------------
        # 2. Entry Logic
        # ---------------------------------------------------------

        # SuperTrend 방향 전환 확인 (flip)
        # - 이전 봉 방향이 -1이고 현재 +1이면: 상승 전환 트리거
        # - 반대면: 하락 전환 트리거
        #
        # [Patch] st_dir 결측/0일 경우 flip이 오작동하지 않도록 int화되어 있다고 가정(위에서 정규화)
        prev_dir = prev.get('st_dir', 0)
        curr_dir = curr.get('st_dir', 0)
        if pd.isna(prev_dir): prev_dir = 0
        if pd.isna(curr_dir): curr_dir = 0

        st_flip_up = (prev_dir == -1) and (curr_dir == 1)
        st_flip_down = (prev_dir == 1) and (curr_dir == -1)

        if is_trend_alive:
            # [LONG] 조건:
            # 1) 15m SuperTrend가 상승으로 전환
            # 2) 거래량 폭발
            # 3) 일봉 기준 상승 대세(일봉 EMA 위)
            # 4) 직전 RSI가 rsi_upper 미만 (과열 진입 제한)
            if st_flip_up and is_vol and is_daily_uptrend:
                if prev.get('rsi', 0) < p['rsi_upper']:
                    signal = 'LONG'

            # [SHORT] 조건:
            # 1) 15m SuperTrend가 하락으로 전환
            # 2) 거래량 폭발
            # 3) 일봉 기준 하락 대세(일봉 EMA 아래)
            # 4) 직전 RSI가 rsi_lower 초과 (과매도 진입 제한)
            elif st_flip_down and is_vol and is_daily_downtrend:
                if prev.get('rsi', 0) > p['rsi_lower']:
                    signal = 'SHORT'

        # ---------------------------------------------------------
        # 3. Exit (ATR Based)
        # ---------------------------------------------------------

        tp_price = 0.0

        # [SL] 손절선 = SuperTrend 라인
        # - 장점: 추세가 이어지면 SL이 따라 올라가며(혹은 내려가며) 자연스러운 트레일링 구조
        # - 주의: st_val이 0이거나 지표가 깨지면 SL이 비정상일 수 있으니,
        #   엔진/모니터 쪽에서 sanity check가 있으면 더 안전해짐
        sl_val = curr.get('st_val', 0)
        if pd.isna(sl_val):
            sl_val = 0
        sl_price = sl_val

        # [ATR] 결측 방어
        atr = curr.get('atr', 0.0)
        if pd.isna(atr):
            atr = 0.0

        # [TP] 목표가 = 현재가 ± ATR*multiplier
        # - 변동성이 크면 TP도 멀어지고, 변동성이 작으면 TP도 가까워지는 구조
        if signal == 'LONG':
            tp_price = curr['close'] + (atr * p['atr_multiplier'])
        elif signal == 'SHORT':
            tp_price = curr['close'] - (atr * p['atr_multiplier'])

        # 반환:
        # - signal: 'LONG'/'SHORT'/None
        # - sl_price: 손절선(슈퍼트렌드)
        # - tp_price: 목표가(ATR 기반)
        return signal, float(sl_price), float(tp_price)
