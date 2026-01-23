import pandas as pd
import logging

# 로깅 설정
logger = logging.getLogger("PhalanxData")

class DataLoader:
    """
    [Phalanx Utility Module]
    Role: Data Gatekeeper (Integrity Validator)
    
    Principles:
    1. Standardization: Columns must be ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    2. Monotonicity: Time must flow forward.
    3. Logical Validity: High >= Max(Open, Close), Low <= Min(Open, Close)
    4. Non-negativity: Volume >= 0
    """

    REQUIRED_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    @staticmethod
    def validate_and_format(data_input):
        """
        데이터 표준화 및 무결성 검증 (Gatekeeper)
        :param data_input: List of lists OR DataFrame
        :return: Validated DataFrame OR None (if invalid)
        """
        try:
            # 1. 표준 포맷 변환 (DataFrame)
            if isinstance(data_input, pd.DataFrame):
                df = data_input.copy()
            elif isinstance(data_input, list):
                if not data_input: 
                    return None
                df = pd.DataFrame(data_input, columns=DataLoader.REQUIRED_COLUMNS)
            else:
                logger.error("❌ Data Error: Unsupported input type.")
                return None

            # 컬럼명 소문자 통일 및 공백 제거
            df.columns = [c.lower().strip() for c in df.columns]

            # 필수 컬럼 존재 확인
            if not all(col in df.columns for col in DataLoader.REQUIRED_COLUMNS):
                logger.error(f"❌ Data Error: Missing columns. Required: {DataLoader.REQUIRED_COLUMNS}")
                return None

            # [Patch] 불필요 컬럼이 있어도 REQUIRED_COLUMNS만 사용하도록 정규화
            # - 데이터 소스에 따라 extra columns가 붙는 경우가 많아, 전략/엔진이 오염되는 것을 방지
            df = df[DataLoader.REQUIRED_COLUMNS].copy()

            # 2. 데이터 타입 강제 변환 (Numeric)
            # [Patch] timestamp는 숫자형(밀리초 epoch)로 유지하되, 원본이 datetime이면 변환 가능하게 방어
            # - CSV에서 timestamp가 문자열 날짜로 들어오는 경우가 있어 to_numeric이 NaN으로 만들 수 있음
            # - 먼저 datetime 파싱을 시도하고, 실패하면 to_numeric으로 간다.
            if df['timestamp'].dtype == object:
                # 문자열/혼합이면 datetime 파싱 시도
                ts_try = pd.to_datetime(df['timestamp'], errors='coerce', utc=False)
                # 파싱 성공한 값이 꽤 있으면 epoch(ms)로 변환
                if ts_try.notna().sum() > 0:
                    # ns -> ms
                    df['timestamp'] = (ts_try.view('int64') // 1_000_000).astype('float')

            # 나머지 컬럼 숫자 변환
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # timestamp도 숫자화 (위에서 datetime->ms 변환이 안 됐을 수도 있으니 최종적으로 한 번 더)
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

            # NaN 데이터 제거 (Level 1: Recoverable)
            if df.isnull().values.any():
                # [Patch] 경고 메시지의 row count가 "컬럼별 NaN 최대치"라 오해 소지 → 실제 drop rows 수 계산
                before = len(df)
                df.dropna(inplace=True)
                dropped = before - len(df)
                logger.warning(f"⚠️ Data Warning: NaN detected. Dropped {dropped} rows.")

            if df.empty:
                logger.error("❌ Data Error: Empty DataFrame after cleanup.")
                return None

            # =========================================================
            # [Patch] Timestamp 단위/형태 안정화
            # - timestamp가 초(s) 단위로 들어오면 엔진/리샘플 기준이 붕괴할 수 있음
            # - 경험적으로:
            #   * ms epoch는 보통 1e12 이상
            #   * s epoch는 보통 1e9~1e10
            # - 여기서는 "너무 작으면 초 단위"로 보고 ms로 보정
            # =========================================================
            ts_max = df['timestamp'].max()
            if ts_max < 1e11:  # 초 단위일 가능성 큼
                df['timestamp'] = df['timestamp'] * 1000.0

            # 3. Timestamp 검증 (단조 증가)
            # 타임스탬프가 역행하거나 중복되면 안 됨
            if not df['timestamp'].is_monotonic_increasing:
                logger.error("❌ Integrity Error: Timestamp is not monotonic increasing.")
                # 정렬 시도
                df.sort_values('timestamp', inplace=True)
                # 중복 제거
                df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)

            # [Patch] 단조 증가 재확인 (정렬/중복제거 후에도 깨질 수 있는 극단 케이스 방어)
            if not df['timestamp'].is_monotonic_increasing:
                logger.critical("❌ Fatal Integrity Error: Timestamp still not monotonic after cleanup.")
                return None

            # =========================================================
            # [Patch] 인덱스는 엔진/전략에서 DatetimeIndex가 필요할 수 있으므로 미리 준비
            # - 기존 계약(컬럼명 표준)을 깨지 않기 위해 'timestamp' 컬럼은 유지
            # - 추가로 '_datetime' 인덱스를 쓰고 싶을 때 활용 가능하도록 준비만 한다.
            #   (전략에서 df.set_index를 강제하지 않는 형태)
            # =========================================================
            # NOTE: 여기서 set_index를 강제하면 기존 엔진 흐름과 충돌할 수 있어 "생성만" 한다.
            # df['_datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

            # 4. 가격 논리적 타당성 검증 (High/Low Logic)
            # High는 Open, Close, Low보다 크거나 같아야 함
            # Low는 Open, Close, High보다 작거나 같아야 함
            # 벡터화 연산으로 고속 검사
            invalid_high = df[
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['high'] < df['low'])
            ]

            invalid_low = df[
                (df['low'] > df['open']) |
                (df['low'] > df['close']) |
                (df['low'] > df['high'])
            ]

            if not invalid_high.empty or not invalid_low.empty:
                logger.error(
                    f"❌ Integrity Error: Logical Price Violation (H/L). Invalid rows: {len(invalid_high) + len(invalid_low)}"
                )
                # 잘못된 캔들 제거 (전염 방지)
                df = df.drop(invalid_high.index).drop(invalid_low.index)

            # [Patch] 가격 논리 위반 제거 후 빈 DF 방어
            if df.empty:
                logger.error("❌ Data Error: Empty DataFrame after H/L integrity cleanup.")
                return None

            # 5. 거래량 비음수 검증
            invalid_vol = df[df['volume'] < 0]
            if not invalid_vol.empty:
                logger.error(f"❌ Integrity Error: Negative Volume detected. Rows: {len(invalid_vol)}")
                df = df[df['volume'] >= 0]

            # [Patch] volume이 모두 0인 데이터는 시장 데이터로서 의미가 약하므로 경고만 남김(차단하진 않음)
            if (df['volume'] == 0).all():
                logger.warning("⚠️ Data Warning: Volume is zero for all rows (possible data quality issue).")

            # 최종 데이터 길이 재확인
            if len(df) < 50:  # 최소 분석 가능 길이 미달
                logger.warning("⚠️ Data Warning: Insufficient data length after validation.")
                return None

            # [Patch] 최종 정렬 보장 (후속 rolling/resample 안정성)
            df.sort_values('timestamp', inplace=True)

            # [Patch] 인덱스 reset 보장 (드랍/필터 후 인덱스가 끊겨도 논리엔 문제 없지만 안정성↑)
            df.reset_index(drop=True, inplace=True)

            return df

        except Exception as e:
            logger.critical(f"❌ Critical Data Validation Error: {e}")
            return None

    @staticmethod
    def load_csv(file_path):
        """CSV 파일 로드 및 검증 (백테스트용)"""
        if not file_path:
            return None
        try:
            df = pd.read_csv(file_path)
            return DataLoader.validate_and_format(df)
        except Exception as e:
            logger.error(f"CSV Load Error: {e}")
            return None
