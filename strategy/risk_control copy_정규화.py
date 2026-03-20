import logging
import math

logger = logging.getLogger("PhalanxRisk")


class RiskControl:
    """
    [Phalanx AOS Module]
    Role: Financial Fortress Guard (Survival Mode)

    핵심 계약(단일 진실원):
    - calculate_entry_size(...)에 전달되는 equity를 "권위 있는 값"으로 사용한다.
      (LIVE: fetch_balance/available 등 엔진이 확정한 값 / BACKTEST: executor.equity)
    - check_account_health(equity)도 동일 equity 기준으로만 판단한다.
    - executor 내부 캐시(executor.equity)에 의존하지 않는다. (단, 백테스트에서 equity를 넘기지 않으면 fallback 가능)
    """

    def __init__(self, executor, config: dict):
        self.executor = executor
        cfg = config or {}
        self.risk_cfg = cfg.get("risk_settings", {}) if isinstance(cfg.get("risk_settings", {}), dict) else {}

        # config 우선
        self.risk_per_trade = float(self.risk_cfg.get("risk_per_trade", 0.01))
        self.leverage = float(self.risk_cfg.get("leverage", 3))
        self.min_notional = float(self.risk_cfg.get("min_notional", 6.0))

        # 생존모드 파라미터
        self.margin_limit_per_pos = float(self.risk_cfg.get("margin_limit_per_pos", 0.06))  # equity 대비 pos margin 상한
        self.hard_cut_margin = float(self.risk_cfg.get("hard_cut_margin", 0.50))            # free margin ratio 하드컷
        self.max_side_exposure = int(self.risk_cfg.get("max_side_exposure", 2))             # 같은 방향 최대 개수

        # NEW: 추가 안전장치/현실성
        self.max_notional_per_pos = float(self.risk_cfg.get("max_notional_per_pos", 0.0))  # 0이면 off
        self.max_total_margin_ratio = float(self.risk_cfg.get("max_total_margin_ratio", 0.60))  # used_margin/equity 상한(0~1)
        self.min_sl_distance_atr = float(self.risk_cfg.get("min_sl_distance_atr", 0.0))    # 0이면 off(전략이 SL 너무 가까운 경우 차단)
        self.min_sl_distance_pct = float(self.risk_cfg.get("min_sl_distance_pct", 0.0))    # 0이면 off

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

        if not (isinstance(self.max_total_margin_ratio, (int, float)) and 0 < self.max_total_margin_ratio < 1.0):
            self.max_total_margin_ratio = 0.60
        if not (isinstance(self.max_notional_per_pos, (int, float)) and self.max_notional_per_pos >= 0):
            self.max_notional_per_pos = 0.0
        if not (isinstance(self.min_sl_distance_atr, (int, float)) and self.min_sl_distance_atr >= 0):
            self.min_sl_distance_atr = 0.0
        if not (isinstance(self.min_sl_distance_pct, (int, float)) and self.min_sl_distance_pct >= 0):
            self.min_sl_distance_pct = 0.0

    # -------------------------
    # Helpers
    # -------------------------
    def _safe_float(self, x, default=0.0) -> float:
        try:
            v = float(x)
            if not math.isfinite(v):
                return float(default)
            return v
        except Exception:
            return float(default)

    def _used_margin(self) -> float:
        used = 0.0
        try:
            for p in (self.executor.positions or {}).values():
                if not isinstance(p, dict):
                    continue
                m = p.get("margin", 0) or 0
                used += self._safe_float(m, 0.0)
        except Exception:
            used = 0.0

        if not math.isfinite(used) or used < 0:
            used = 0.0
        return float(used)

    def _equity_fallback(self, equity: float) -> float:
        """
        equity가 비정상이면 fallback:
        - executor.equity가 있으면 사용(백테스트 편의)
        - 그래도 안 되면 0
        """
        eq = self._safe_float(equity, 0.0)
        if eq > 0:
            return eq

        ex_eq = getattr(self.executor, "equity", None)
        ex_eq = self._safe_float(ex_eq, 0.0)
        return ex_eq

    def _clamp(self, x: float, lo: float, hi: float) -> float:
        x = float(x)
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def _vol_weight(self, vol_pct: float) -> float:
        """
        변동성(%) 기반 비용 정규화 가중치.
        - vol_pct: 예) SL% (= |entry-sl|/entry) 또는 ATR% 등
        - vol_ref: 기준 변동성(%) (예: 0.03 = 3%)
        - k: 강도 (0이면 off)
        - clamp로 과도한 확대/축소 방지
        """
        vol_ref = float(self.risk_cfg.get("vol_ref_pct", 0.03))      # 3%
        k = float(self.risk_cfg.get("vol_weight_k", 0.5))            # 강도
        w_min = float(self.risk_cfg.get("vol_weight_min", 0.5))
        w_max = float(self.risk_cfg.get("vol_weight_max", 1.5))

        if k <= 0:
            return 1.0
        if vol_pct <= 0 or (not math.isfinite(vol_pct)):
            return 1.0

        w = (vol_ref / float(vol_pct)) ** k
        return self._clamp(w, w_min, w_max)

    def _liquidity_cap_qty(self, entry_price: float, vol_ma: float) -> float:
        """
        거래량(유동성) 기반 최대 수량 캡.
        - vol_ma: TitanStrategy가 만든 vol_ma(rolling mean volume) :contentReference[oaicite:3]{index=3}
        - dollar_vol(프록시) = entry_price * vol_ma
        - max_notional = dollar_vol * adv_cap_ratio
        - qty_cap = max_notional / entry_price
        """
        try:
            ep = self._safe_float(entry_price, 0.0)
            vma = self._safe_float(vol_ma, 0.0)
            if ep <= 0 or vma <= 0:
                return 0.0

            adv_cap_ratio = float((self.risk_cfg or {}).get("adv_cap_ratio", 0.0) or 0.0)  # 0이면 off
            if adv_cap_ratio <= 0:
                return 0.0

            dollar_vol = ep * vma
            if (not math.isfinite(dollar_vol)) or dollar_vol <= 0:
                return 0.0

            adv_min_dollar_vol = float((self.risk_cfg or {}).get("adv_min_dollar_vol", 0.0) or 0.0)  # 0이면 off
            if adv_min_dollar_vol > 0 and dollar_vol < adv_min_dollar_vol:
                return 0.0

            max_notional = dollar_vol * adv_cap_ratio
            if (not math.isfinite(max_notional)) or max_notional <= 0:
                return 0.0

            qty_cap = max_notional / ep
            if (not math.isfinite(qty_cap)) or qty_cap <= 0:
                return 0.0

            return float(qty_cap)
        except Exception:
            return 0.0

    # -------------------------
    # Gates
    # -------------------------
    def check_account_health(self, equity: float) -> bool:
        """
        [Gate 1] Free Margin Protection
        - equity는 반드시 엔진이 확정한 단일 기준값을 사용
        """
        eq = self._equity_fallback(equity)
        if eq <= 0:
            return False

        used_margin = self._used_margin()

        # NEW: used_margin 자체가 equity 대비 너무 크면 강제 차단(상태 오염/과다진입 방지)
        used_ratio = used_margin / eq if eq > 0 else 1.0
        if (not math.isfinite(used_ratio)) or used_ratio < 0:
            return False
        if used_ratio >= float(self.max_total_margin_ratio):
            return False

        free_margin_ratio = (eq - used_margin) / eq
        if not math.isfinite(free_margin_ratio) or free_margin_ratio < 0:
            return False

        if free_margin_ratio < float(self.hard_cut_margin):
            return False

        return True

    def check_correlation(self, new_side: str) -> bool:
        """
        [Gate 2] Correlation Guard (동방향 과다 노출 제한)
        """
        s = str(new_side).upper()
        if s not in ("LONG", "SHORT"):
            return False

        same_side = 0
        for pos in (self.executor.positions or {}).values():
            if not isinstance(pos, dict):
                continue
            if str(pos.get("side", "")).upper() == s:
                same_side += 1

        return same_side < int(self.max_side_exposure)

    # -------------------------
    # Sizing
    # -------------------------
    def calculate_entry_size(
        self,
        symbol,
        entry_price,
        equity,
        sl_price,
        signal_side,
        atr: float = 0.0,
        vol_ma: float = 0.0,   # ✅ NEW
    ):
        """
        [Gate 3] Dual-Cap Sizing (Risk vs Margin)

        - 기본: SL% 기반(손절 손실 동일화)
        - 추가: 변동성 가중치로 비용(슬리피지/스프레드/임팩트) 정규화 
        - ✅ 추가2(NEW): 거래량(유동성) 기반 notional cap으로 비용을 더 직접적으로 정규화
        """
        try:
            entry_price = self._safe_float(entry_price, 0.0)
            sl_price = self._safe_float(sl_price, 0.0)
            eq = self._equity_fallback(equity)
            atrv = self._safe_float(atr, 0.0)
            vma = self._safe_float(vol_ma, 0.0)

            if entry_price <= 0 or sl_price <= 0 or eq <= 0:
                return 0.0

            side = str(signal_side).upper()
            if side not in ("LONG", "SHORT"):
                return 0.0

            # Gate 1: account health
            if not self.check_account_health(eq):
                return 0.0

            # Gate 2: correlation
            if not self.check_correlation(side):
                return 0.0

            # SL distance sanity
            price_diff = abs(entry_price - sl_price)
            if price_diff <= 0 or (not math.isfinite(price_diff)):
                return 0.0

            if float(self.min_sl_distance_pct) > 0:
                if (price_diff / entry_price) < float(self.min_sl_distance_pct):
                    return 0.0

            if float(self.min_sl_distance_atr) > 0 and atrv > 0:
                if price_diff < (atrv * float(self.min_sl_distance_atr)):
                    return 0.0

            # 변동성(%) 계산: SL% 기본
            sl_pct = price_diff / entry_price
            vol_pct = sl_pct
            if atrv > 0 and math.isfinite(atrv):
                atr_pct = atrv / entry_price
                if math.isfinite(atr_pct) and atr_pct > 0:
                    vol_pct = max(vol_pct, atr_pct)

            # 변동성 가중치
            vol_w = self._vol_weight(vol_pct)

            # Cap A: risk-based
            risk_money = eq * float(self.risk_per_trade) * float(vol_w)
            if risk_money <= 0 or (not math.isfinite(risk_money)):
                return 0.0
            qty_by_risk = risk_money / price_diff

            # Cap B: margin-based
            target_margin = eq * float(self.margin_limit_per_pos)
            if target_margin <= 0 or (not math.isfinite(target_margin)):
                return 0.0
            qty_by_margin = (target_margin * float(self.leverage)) / entry_price

            if (not math.isfinite(qty_by_risk)) or (not math.isfinite(qty_by_margin)):
                return 0.0
            if qty_by_risk <= 0 or qty_by_margin <= 0:
                return 0.0

            raw_amount = min(qty_by_risk, qty_by_margin)

            # ✅ NEW: 거래량(유동성) cap
            qty_cap = self._liquidity_cap_qty(entry_price, vma)
            if qty_cap > 0:
                raw_amount = min(raw_amount, qty_cap)

            # notional hard cap (옵션)
            if float(self.max_notional_per_pos) > 0:
                max_amt_by_notional = float(self.max_notional_per_pos) / entry_price
                if math.isfinite(max_amt_by_notional) and max_amt_by_notional > 0:
                    raw_amount = min(raw_amount, max_amt_by_notional)

            # Min notional
            notional = raw_amount * entry_price
            if (not math.isfinite(notional)) or notional <= 0:
                return 0.0
            if notional < float(self.min_notional):
                return 0.0

            return float(raw_amount)

        except Exception:
            return 0.0