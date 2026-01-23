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

    def check_conditions(self, symbol, position, market_data):
        """
        청산 및 관리 조건 판단 (Pure Function)
        
        :param symbol: 코인 심볼
        :param position: 포지션 상태 딕셔너리
        :param market_data: 시장 데이터 {'close', 'high', 'low', 'atr', 'st_val'}
        :return: (action, exec_price, reason, new_sl)
        """
        # 1. 포지션 정보 언패킹
        side = position['side']
        current_sl = position.get('sl', 0)
        
        # 2. 시장 데이터 언패킹
        curr_price = market_data['close']
        high_price = market_data['high']
        low_price = market_data['low']
        st_val = market_data.get('st_val', 0) # SuperTrend 값
        
        # 결과 변수 초기화
        action = None       # 'TP1', 'EXIT', 'UPDATE_SL', None
        exec_price = 0.0
        reason = None
        new_sl = current_sl

        # =========================================================
        # [Logic A] SuperTrend Trailing (The Runner Logic)
        # =========================================================
        # 설명: SL을 가격 기준이 아니라 '추세선(SuperTrend)' 기준으로 동기화
        #       추세가 이어지는 한 SL은 계속 이익 방향으로 전진함.
        #
        # [개선 핵심] "캔들 내 순서 문제" 제거:
        # - 같은 캔들에서 ST가 갱신되더라도,
        #   그 갱신 SL은 "다음 캔들부터" 적용한다.
        # - 따라서 이번 캔들의 SL 히트 판정에는 current_sl을 사용한다.
        
        if st_val > 0: # 유효한 SuperTrend 값이 있을 때만
            if side == 'LONG':
                # Long: ST가 현재 SL보다 위에 있으면 올린다. (내리진 않음)
                # 단, 현재가가 ST보다 위에 있어야 함 (이미 뚫린 상태면 업데이트 금지)
                if st_val > current_sl and st_val < curr_price:
                    new_sl = st_val
                    action = 'UPDATE_SL'
            
            elif side == 'SHORT':
                # Short: ST가 현재 SL보다 아래에 있으면 내린다.
                if st_val < current_sl and st_val > curr_price:
                    new_sl = st_val
                    action = 'UPDATE_SL'

        # =========================================================
        # [Logic B] Stop Loss & Trend Reversal Check
        # =========================================================
        # SuperTrend 라인(effective_sl)을 건드리면 즉시 청산
        #
        # [개선 핵심] 이번 캔들 SL 히트 판정은 "기존 SL(current_sl)"로 한다.
        # - SL 업데이트(new_sl)는 다음 캔들부터 유효하므로,
        #   이번 캔들 내 체결 순서 모순을 제거한다.
        effective_sl_for_hit = current_sl

        sl_hit = False
        if side == 'LONG' and low_price <= effective_sl_for_hit: 
            sl_hit = True
        elif side == 'SHORT' and high_price >= effective_sl_for_hit: 
            sl_hit = True

        if sl_hit:
            # 로그에는 STOP_LOSS로 찍히지만, 실제로는 '익절'일 확률이 높음 (Trailing Profit)
            return 'EXIT', effective_sl_for_hit, 'STOP_LOSS', new_sl

        # =========================================================
        # [Logic C] TP1 Check (Partial Take Profit)
        # =========================================================
        #if not position.get('tp1_hit', False):
        #    tp1_price = position['tp1']
        #    tp_hit = False
        #    
        #    if side == 'LONG' and high_price >= tp1_price: tp_hit = True
        ##    elif side == 'SHORT' and low_price <= tp1_price: tp_hit = True
         #   
        #    if tp_hit:
                # TP1 달성! (SL은 위에서 계산된 Trailing SL 유지 -> Break-Even 강제 이동 없음)
         #       return 'TP1', tp1_price, 'TP', new_sl

        # =========================================================
        # [Logic D] Return Status
        # =========================================================
        if action == 'UPDATE_SL':
            # 청산은 없지만 SL이 갱신됨 (다음 캔들부터 유효)
            return 'UPDATE_SL', 0.0, 'TRAILING', new_sl
            
        # 아무 일도 없음
        return None, 0.0, None, current_sl
