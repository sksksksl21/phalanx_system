import logging

# 로깅 설정
logger = logging.getLogger("PhalanxMonitor")

class PositionMonitor:
    """
    [Phalanx Strategy Module]
    Role: Exit Logic Authority (Shared by Live & Backtest)
    
    Strategy: "The Unlimited Runner"
    - TP1: ATR 3.0 도달 시 50% 익절 (현금 확보)
    - Runner: SuperTrend 라인을 SL로 사용하여 끝까지 추세 추종
    - Break-Even: 제거됨 (SuperTrend가 자연스럽게 따라 올라감)
    """
    
    def __init__(self):
        pass

    def check_conditions(self, symbol, position, market_data, sl_apply_mode: str = "next"):
        """
        청산 및 관리 조건 판단 (Pure Function)

        :param symbol: 코인 심볼
        :param position: 포지션 상태 딕셔너리
        :param market_data: 시장 데이터 {'close', 'high', 'low', 'atr', 'st_val'}
        :param sl_apply_mode: "next" (기존) | "same" (확정봉 즉시 반영)
        :return: (action, exec_price, reason, new_sl)
        """
        # 1. 포지션 정보 언패킹
        side = position['side']
        current_sl = position.get('sl', 0)

        # 2. 시장 데이터 언패킹
        curr_price = market_data['close']
        high_price = market_data['high']
        low_price = market_data['low']
        st_val = market_data.get('st_val', 0)  # SuperTrend 값

        # 결과 변수 초기화
        action = None       # 'TP1', 'EXIT', 'UPDATE_SL', None
        exec_price = 0.0
        reason = None
        new_sl = current_sl

        # normalize mode
        try:
            mode = str(sl_apply_mode or "next").strip().lower()
        except Exception:
            mode = "next"
        if mode not in ("next", "same"):
            mode = "next"

        # =========================================================
        # [Logic A] SuperTrend Trailing (The Runner Logic)
        # =========================================================
        if st_val > 0:  # 유효한 SuperTrend 값이 있을 때만
            if side == 'LONG':
                if st_val > current_sl and st_val < curr_price:
                    new_sl = st_val
                    action = 'UPDATE_SL'

            elif side == 'SHORT':
                if st_val < current_sl and st_val > curr_price:
                    new_sl = st_val
                    action = 'UPDATE_SL'

        # =========================================================
        # [Logic B] Stop Loss & Trend Reversal Check
        # =========================================================
        # ✅ 비교 실험 스위치:
        # - next 모드: 히트 판정은 current_sl (기존)
        # - same 모드: UPDATE_SL이 나온 캔들에서는 new_sl을 즉시 히트 판정에 사용
        effective_sl_for_hit = current_sl
        if mode == "same" and action == "UPDATE_SL":
            effective_sl_for_hit = new_sl

        sl_hit = False
        if side == 'LONG' and low_price <= effective_sl_for_hit:
            sl_hit = True
        elif side == 'SHORT' and high_price >= effective_sl_for_hit:
            sl_hit = True

        if sl_hit:
            return 'EXIT', effective_sl_for_hit, 'STOP_LOSS', new_sl

        # =========================================================
        # [Logic D] Return Status
        # =========================================================
        if action == 'UPDATE_SL':
            return 'UPDATE_SL', 0.0, 'TRAILING', new_sl

        return None, 0.0, None, current_sl
