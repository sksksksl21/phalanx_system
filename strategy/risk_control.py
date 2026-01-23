import logging
import math

logger = logging.getLogger("PhalanxRisk")

class RiskControl:
    """
    [Phalanx AOS Module]
    Role: Financial Fortress Guard (Survival Mode)
    
    Principles:
    1. Free Margin Protection (Hard Cut < 50%)
    2. Dual-Cap Sizing (Risk Limit vs Margin Limit)
    3. Correlation Exposure Control
    """
    
    def __init__(self, executor, config):
        self.executor = executor
        self.risk_cfg = config.get('risk_settings', {}) if config.get('risk_settings') else config
        
        # 1. 기본 리스크 설정
        self.risk_per_trade = 0.01  # 1% Risk (고정)
        self.leverage = self.risk_cfg.get('leverage', 3)
        self.min_notional = 6.0
        
        # 2. [AOS] 생존 모드 설정
        self.margin_limit_per_pos = 0.06  # 포지션당 최대 증거금 점유율 10% (Dual-Cap)
        self.hard_cut_margin = 0.50       # 가용 자금 50% 미만 시 진입 금지
        self.max_side_exposure = 2        # 같은 방향 포지션 최대 2개 (Correlation Guard)

        # [Patch] 설정값 sanity check (결정적 재현성 + 런타임 안전)
        # - 잘못된 설정이 들어오면 "무조건 거래 0" 혹은 과도한 레버리지/노출이 날 수 있음
        if not isinstance(self.leverage, (int, float)) or self.leverage <= 0:
            self.leverage = 3
        if not isinstance(self.risk_per_trade, (int, float)) or self.risk_per_trade <= 0:
            self.risk_per_trade = 0.01
        if not isinstance(self.margin_limit_per_pos, (int, float)) or self.margin_limit_per_pos <= 0:
            self.margin_limit_per_pos = 0.06
        if not isinstance(self.hard_cut_margin, (int, float)) or self.hard_cut_margin <= 0:
            self.hard_cut_margin = 0.50
        if not isinstance(self.max_side_exposure, int) or self.max_side_exposure <= 0:
            self.max_side_exposure = 2
        if not isinstance(self.min_notional, (int, float)) or self.min_notional <= 0:
            self.min_notional = 6.0

    def check_account_health(self):
        """
        [Gate 1] 계좌 생존 여부 확인 (Free Margin Check)
        """
        # 현재 증거금 사용량 계산
        used_margin = sum(p.get('margin', 0) for p in self.executor.positions.values())

        # [Patch] NaN/None 방어 (백테스트 시 데이터 오염/초기화 누락 대비)
        if used_margin is None:
            used_margin = 0
        try:
            used_margin = float(used_margin)
        except Exception:
            used_margin = 0.0

        equity = self.executor.equity
        # [Patch] equity 결측 방어
        if equity is None:
            return False
        try:
            equity = float(equity)
        except Exception:
            return False
        
        if equity <= 0:
            return False
        
        free_margin_ratio = (equity - used_margin) / equity
        
        # [Patch] 비정상 free_margin_ratio 방어 (used_margin이 음수/NaN 등)
        if not math.isfinite(free_margin_ratio):
            return False
        if free_margin_ratio < 0:
            # used_margin이 equity보다 커진 비정상 상태 (시뮬레이션 오염 가능)
            return False

        # 가용 자금이 50% 미만이면 신규 진입 전면 차단
        if free_margin_ratio < self.hard_cut_margin:
            # logger.warning(f"🛡️ [AOS BLOCK] Low Oxygen: Free Margin {free_margin_ratio*100:.1f}% < 50%")
            return False
            
        return True

    def check_correlation(self, new_side):
        """
        [Gate 2] 상관관계 노출 확인 (Correlation Guard)
        """
        # [Patch] new_side 유효성 검사 (None/오타 방어)
        if new_side not in ('LONG', 'SHORT'):
            return False

        same_side_count = 0
        for pos in self.executor.positions.values():
            # [Patch] 포지션 dict 결측 방어
            if not isinstance(pos, dict):
                continue
            if pos.get('side') == new_side:
                same_side_count += 1
        
        # 같은 방향이 이미 N개 이상이면 차단
        if same_side_count >= self.max_side_exposure:
            # logger.warning(f"🛡️ [AOS BLOCK] Correlation Limit: {same_side_count} {new_side} positions already open.")
            return False
            
        return True

    def calculate_entry_size(self, symbol, entry_price, equity, sl_price, signal_side):
        """
        [Gate 3] 이중 캡 사이징 (Dual-Cap Sizing)
        """
        try:
            # 1. 기초 검문
            if not self.check_account_health():
                return 0.0
            if not self.check_correlation(signal_side):
                return 0.0

            # [Patch] 입력값 타입/NaN 방어
            if entry_price is None or sl_price is None or equity is None:
                return 0.0
            try:
                entry_price = float(entry_price)
                sl_price = float(sl_price)
                equity = float(equity)
            except Exception:
                return 0.0
            if not (math.isfinite(entry_price) and math.isfinite(sl_price) and math.isfinite(equity)):
                return 0.0

            if entry_price <= 0 or sl_price <= 0 or equity <= 0:
                return 0.0

            # 2. [Cap A] 손실 리스크 기반 수량 (Risk Based)
            price_diff = abs(entry_price - sl_price)

            # [Patch] 너무 작은 손절폭(거의 0) 방어:
            # - 분모가 극단적으로 작으면 qty_by_risk가 폭발 -> 비현실 포지션 크기
            # - 실전에서는 최소 tick/스프레드/슬리피지 때문에 의미 없는 값
            if price_diff <= 0:
                return 0.0

            risk_money = equity * self.risk_per_trade  # 1%
            if risk_money <= 0:
                return 0.0

            qty_by_risk = risk_money / price_diff

            # 3. [Cap B] 증거금 한도 기반 수량 (Margin Based)
            # 목표: 이 포지션의 증거금이 전체 Equity의 10%를 넘지 않게 하라.
            # Margin = (Qty * Price) / Leverage
            # Target Margin = Equity * 0.10
            # Qty = (Target Margin * Leverage) / Price
            target_margin = equity * self.margin_limit_per_pos

            # [Patch] target_margin이 음수/0이면 사이징 불가
            if target_margin <= 0:
                return 0.0

            qty_by_margin = (target_margin * self.leverage) / entry_price

            # [Patch] qty 음수/NaN 방어
            if not (math.isfinite(qty_by_risk) and math.isfinite(qty_by_margin)):
                return 0.0
            if qty_by_risk <= 0 or qty_by_margin <= 0:
                return 0.0

            # 4. [Dual-Cap] 더 작은 쪽 선택 (보수적 접근)
            raw_amount = min(qty_by_risk, qty_by_margin)

            # (디버그용: 어떤 캡이 적용되었는지 확인)
            # if qty_by_margin < qty_by_risk:
            #     print(f"🔒 [Cap Active] Margin Limit Applied on {symbol}")

            # 5. 최소 주문 금액 체크 (Min Notional Strict)
            notional_value = raw_amount * entry_price

            # [Patch] notional_value NaN/음수 방어
            if not math.isfinite(notional_value) or notional_value <= 0:
                return 0.0

            if notional_value < self.min_notional:
                return 0.0

            # 6. 정밀도 변환
            final_amount = self.executor.amount_to_precision(symbol, raw_amount)

            # [Patch] precision 결과가 0/NaN이면 진입 금지
            if final_amount is None:
                return 0.0
            try:
                final_amount = float(final_amount)
            except Exception:
                return 0.0
            if not math.isfinite(final_amount) or final_amount <= 0:
                return 0.0

            return float(final_amount)

        except Exception as e:
            logger.error(f"❌ [RISK ERROR] {symbol}: {e}")
            return 0.0
