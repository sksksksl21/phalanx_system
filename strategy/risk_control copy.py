import logging
import math

logger = logging.getLogger("PhalanxRisk")


class RiskControl:
    """
    [Phalanx AOS Module]
    Role: Financial Fortress Guard (Survival Mode)

    FIX 핵심:
    - LIVE의 '진짜 계좌 equity'는 executor.equity 같은 캐시값에 의존하면 안 됨.
    - calculate_entry_size()로 전달되는 equity(=실계좌 기준)를 단일 기준으로 사용.
    """

    def __init__(self, executor, config: dict):
        self.executor = executor
        cfg = config or {}
        self.risk_cfg = cfg.get("risk_settings", {}) if isinstance(cfg.get("risk_settings", {}), dict) else {}

        # ✅ config.json을 신뢰: risk_per_trade는 config 우선
        self.risk_per_trade = float(self.risk_cfg.get("risk_per_trade", 0.01))
        self.leverage = float(self.risk_cfg.get("leverage", 3))
        self.min_notional = float(self.risk_cfg.get("min_notional", 6.0))

        # ✅ 생존모드 파라미터도 config로 오버라이드 가능하게(없으면 기본값)
        self.margin_limit_per_pos = float(self.risk_cfg.get("margin_limit_per_pos", 0.06))  # equity 대비 pos margin 상한
        self.hard_cut_margin = float(self.risk_cfg.get("hard_cut_margin", 0.50))            # free margin ratio 하드컷
        self.max_side_exposure = int(self.risk_cfg.get("max_side_exposure", 2))             # 같은 방향 최대 개수

        # sanity
        if not (isinstance(self.leverage, (int, float)) and self.leverage > 0):
            self.leverage = 3.0
        if not (isinstance(self.risk_per_trade, (int, float)) and self.risk_per_trade > 0):
            self.risk_per_trade = 0.01
        if not (isinstance(self.margin_limit_per_pos, (int, float)) and self.margin_limit_per_pos > 0):
            self.margin_limit_per_pos = 0.06
        if not (isinstance(self.hard_cut_margin, (int, float)) and 0 < self.hard_cut_margin < 1.0):
            self.hard_cut_margin = 0.50
        if not (isinstance(self.max_side_exposure, int) and self.max_side_exposure > 0):
            self.max_side_exposure = 2
        if not (isinstance(self.min_notional, (int, float)) and self.min_notional > 0):
            self.min_notional = 6.0

    # -------------------------
    # Gates
    # -------------------------
    def _used_margin(self) -> float:
        used = 0.0
        try:
            for p in (self.executor.positions or {}).values():
                if not isinstance(p, dict):
                    continue
                m = p.get("margin", 0) or 0
                try:
                    used += float(m)
                except Exception:
                    pass
        except Exception:
            used = 0.0
        if not math.isfinite(used) or used < 0:
            used = 0.0
        return float(used)

    def check_account_health(self, equity: float) -> bool:
        """
        [Gate 1] Free Margin Protection
        - equity는 반드시 "실계좌 기준"을 넣어야 함 (LIVE: fetch_balance 기반)
        """
        try:
            equity = float(equity)
        except Exception:
            return False

        if not math.isfinite(equity) or equity <= 0:
            return False

        used_margin = self._used_margin()
        free_margin_ratio = (equity - used_margin) / equity

        if not math.isfinite(free_margin_ratio) or free_margin_ratio < 0:
            return False

        if free_margin_ratio < self.hard_cut_margin:
            return False

        return True

    def check_correlation(self, new_side: str) -> bool:
        if bool(self.risk_cfg.get("disable_side_exposure_limit", False)):
            return True

        s = str(new_side).upper()
        if s not in ("LONG", "SHORT"):
            return False

        same_side = 0
        for pos in (self.executor.positions or {}).values():
            if not isinstance(pos, dict):
                continue
            if str(pos.get("side", "")).upper() == s:
                same_side += 1

        return same_side < self.max_side_exposure

    # -------------------------
    # Sizing
    # -------------------------
    def calculate_entry_size(self, symbol, entry_price, equity, sl_price, signal_side):
        """
        [Gate 3] Dual-Cap Sizing (Risk vs Margin)
        """
        try:
            # inputs
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

            side = str(signal_side).upper()

            # Gate 1: account health (✅ equity는 "실계좌"를 넣어야 통과)
            if not self.check_account_health(equity):
                return 0.0

            # Gate 2: correlation
            if not self.check_correlation(side):
                return 0.0

            # Cap A: risk-based
            price_diff = abs(entry_price - sl_price)
            if price_diff <= 0:
                return 0.0

            risk_money = equity * float(self.risk_per_trade)
            if risk_money <= 0:
                return 0.0

            qty_by_risk = risk_money / price_diff

            # Cap B: margin-based (pos margin <= equity * margin_limit_per_pos)
            target_margin = equity * float(self.margin_limit_per_pos)
            if target_margin <= 0:
                return 0.0

            qty_by_margin = (target_margin * float(self.leverage)) / entry_price

            if not (math.isfinite(qty_by_risk) and math.isfinite(qty_by_margin)):
                return 0.0
            if qty_by_risk <= 0 or qty_by_margin <= 0:
                return 0.0

            raw_amount = min(qty_by_risk, qty_by_margin)

            # Min notional
            notional = raw_amount * entry_price
            if (not math.isfinite(notional)) or notional <= 0:
                return 0.0
            if notional < float(self.min_notional):
                return 0.0

            # precision
            try:
                final_amount = self.executor.amount_to_precision(symbol, raw_amount)
            except Exception:
                final_amount = raw_amount

            try:
                final_amount = float(final_amount)
            except Exception:
                return 0.0

            if not math.isfinite(final_amount) or final_amount <= 0:
                return 0.0

            # after precision, ensure min_notional again (precision으로 깎여서 below 가능)
            if final_amount * entry_price < float(self.min_notional):
                return 0.0

            return float(final_amount)

        except Exception as e:
            logger.error(f"❌ [RISK ERROR] {symbol}: {e}")
            return 0.0
