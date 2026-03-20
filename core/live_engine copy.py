# core/live_engine.py
# =========================================================
# [Phalanx Core Module] Live Engine (LIVE ONLY - A Mode)
# Mode: Time-Sequential & Priority-Based (Backtest-Identical)
#
# ✅ LIVE 전용 (A모드)
# - entry=15m / manage=15m
# - DRY_RUN / Track A,B 관련 코드 전부 제거
#
# 핵심 보강(감사 반영):
# 1) Time Authority: 15m 경계(00/15/30/45) 직후로 루프 정렬 (drift 제거)
# 2) Stale Blocking: now_utc 비교 금지. 심볼 간 상대 lag(ref=max(last_ts))로 stale 필터
# 3) Data Refresh: 새 캔들(15m)이 바뀌는 순간에만 prepare_data() 재호출
# 4) trade_history.csv 자동 생성/append (ENTRY/EXIT/UPDATE_SL/REJECT/RECONCILE/HEARTBEAT(optional))
# 5) 상태 단일 진실원: phalanx_state.json (positions + last_processed_time + last_bucket)

#
# ✅ Telegram Alert:
# - BOOT / RECONCILE_OK / RECONCILE_MISMATCH
# - ENTRY / ENTRY_REJECT / ENTRY_FAIL
# - UPDATE_SL (옵션) / EXIT
# =========================================================

from operator import pos
import sys
import os
import json
import time
import math
import logging
import pandas as pd
import urllib.parse
import urllib.request

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "phalanx_live.log")
LOG_PATH = os.path.abspath(LOG_PATH)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.StreamHandler(),              # 콘솔
        logging.FileHandler(LOG_PATH, encoding="utf-8"),  # 파일
    ],
)

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from strategy.position_monitor import PositionMonitor
from strategy.risk_control import RiskControl
from strategy.titan_strategy import TitanStrategy
from execution.binance_executor import BinanceExecutor
from utils.data_loader import DataLoader

BASE_FEE = 0.0005

logger = logging.getLogger("PhalanxLive")


class TelegramNotifier:
    """
    Minimal Telegram sender (no external dependency).
    config keys:
      - telegram_token / telegram_chat_id
      - system_settings:
          telegram_enabled (optional)
          telegram_send_update_sl (default False)
          telegram_send_heartbeat (default False)
          telegram_tag (default 'PHALANX')
          telegram_dedup_window_sec (default 15)
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        self.token = self.cfg.get("telegram_token") or self.cfg.get("telegramToken") or ""
        self.chat_id = self.cfg.get("telegram_chat_id") or self.cfg.get("telegramChatId") or ""
        ss = (self.cfg.get("system_settings") or {})

        if "telegram_enabled" in ss:
            self.enabled = bool(ss.get("telegram_enabled"))
        else:
            self.enabled = bool(self.token and self.chat_id)

        self.send_update_sl = bool(ss.get("telegram_send_update_sl", True))
        self.send_heartbeat = bool(ss.get("telegram_send_heartbeat", False))
        self.tag = str(ss.get("telegram_tag", "PHALANX"))

        self._last_text = None
        self._last_ts = 0.0
        self._dedup_window_sec = float(ss.get("telegram_dedup_window_sec", 15.0))
        self._freeze_reason = None
        self._freeze_since_utc = None

    
    def _post(self, text: str) -> bool:
        if not self.enabled or not self.token or not self.chat_id:
            return False
        try:
            now = time.time()
            if self._last_text == text and (now - self._last_ts) < self._dedup_window_sec:
                return False
            self._last_text = text
            self._last_ts = now

            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "disable_web_page_preview": True,
            }
            data = urllib.parse.urlencode(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                _ = resp.read()
            return True
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    def send(self, title: str, lines: list):
        if not self.enabled:
            return
        try:
            msg = f"[{self.tag}] {title}\n" + "\n".join([str(x) for x in (lines or [])])
            self._post(msg)
        except Exception:
            return


class LiveEngine:
    # -----------------------------------------------------
    # Config
    # -----------------------------------------------------
    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(self.config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise RuntimeError("config.json must be a JSON object")
        return cfg

    def __init__(self):
        # paths
        self.root_dir = root_dir
        self.config_path = os.path.join(root_dir, "config.json")

        # ✅ ledger split
        self.history_path = os.path.join(root_dir, "trade_history.csv")   # trades-only
        self.ops_history_path = os.path.join(root_dir, "ops_history.csv") # ops/monitoring

        self.state_path = os.path.join(root_dir, "phalanx_state.json")

        # config
        self.cfg = self._load_config()

        # time authority
        self.tf15_sec = 15 * 60
        self.loop_buffer_sec = float(self.cfg.get("system_settings", {}).get("loop_buffer_sec", 3.0))

        # core components
        self.titan = TitanStrategy()
        self.executor = BinanceExecutor(self.cfg)

        # ✅ [BALANCE SYNC @ BOOT] executor.cash / executor.equity 권위 최신화
        self._last_cash = None
        self._last_equity = None
        try:
            bal = self.executor.fetch_balance() or {}
            usdt = bal.get("USDT", {}) if isinstance(bal, dict) else {}
            self._last_cash = float(usdt.get("free", 0) or 0)
            self._last_equity = float(usdt.get("total", 0) or 0)
        except Exception as e:
            logger.error(f"[BOOT] fetch_balance failed: {e}")
            self._last_cash = float(getattr(self.executor, "cash", 0) or 0)
            self._last_equity = float(getattr(self.executor, "equity", 0) or 0)

        # RiskControl.check_account_health()가 executor.equity를 보므로 같이 맞춰줌
        try:
            if hasattr(self.executor, "cash"):
                self.executor.cash = float(self._last_cash or 0)
            if hasattr(self.executor, "equity"):
                self.executor.equity = float(self._last_equity or 0)
        except Exception:
            pass

        self.risk_ctrl = RiskControl(self.executor, self.cfg)
        self.monitor = PositionMonitor()
        self.data_loader = DataLoader()
        self.notifier = TelegramNotifier(self.cfg)

        # blacklist
        strat_settings = self.cfg.get("strategy_settings", {})
        if "blacklist" in strat_settings:
            self.titan.blacklist = set(strat_settings["blacklist"])
        logger.info(f"🚫 Blacklist loaded: {self.titan.blacklist}")

        # data caches (15m)
        self.raw_data_map = {}
        self.data_map = {}
        self.symbols = []

        # LIVE safety
        self.freeze_new_entries = False

        # state
        self.last_processed_time = None
        self.last_bucket = None

        # ✅ history/state init (trade + ops)
        self._ensure_history_file()

        # ✅ cooldown state init (before load_state: 구조 보장)
        self._init_cooldown_state()

        self._load_state()


        # reconcile at boot (LIVE only)
        self.reconcile_positions()

        # BOOT log
        boot_dt = pd.Timestamp.utcnow()

        # ✅ BOOT telegram
        self.notifier.send(
            title="BOOT",
            lines=[
                f"t_utc={boot_dt}",
                "mode=LIVE_A(15m entry/15m manage)",
                f"freeze_new_entries={int(self.freeze_new_entries)}",
                f"pos_count={int(len(self.executor.positions or {}))}",
                f"cash={float(self._last_cash or 0)}",
                f"equity={float(self._last_equity or 0)}",
            ],
        )

        logger.info("🟩 MODE: LIVE (A-mode 15m entry/15m manage)")

    # -----------------------------------------------------
    # History helpers
    # -----------------------------------------------------
    def _history_columns(self):
        # ✅ trades-only columns
        return [
            "dt", "event", "mode",
            "symbol", "side",
            "price", "amount",
            "fee", "margin", "pnl", "roe_pct",
            "sl",
            "reason",
            "pos_count",
            "cash", "equity",
        ]

    def _ops_columns(self):
        # ✅ ops/monitoring columns (원장과 분리)
        return [
            "dt", "event", "mode",
            "severity",
            "symbol", "side",
            "reason",
            "pos_count",
            "cash", "equity",
            "t_ref",           # time authority ref (optional)
            "active",          # active_symbols count (optional)
            "manage",          # manage_cnt (optional)
            "cand", "entry", "exit", "updSL",  # heartbeat counters (optional)
            "freeze",          # freeze meta (optional)
        ]



    def _ensure_history_file(self):
        try:
            # ---- trade ledger
            if (not os.path.exists(self.history_path)) or os.path.getsize(self.history_path) == 0:
                pd.DataFrame(columns=self._history_columns()).to_csv(self.history_path, index=False)

            # ---- ops ledger
            if (not os.path.exists(self.ops_history_path)) or os.path.getsize(self.ops_history_path) == 0:
                pd.DataFrame(columns=self._ops_columns()).to_csv(self.ops_history_path, index=False)

        except Exception as e:
            logger.error(f"History init failed: {e}")
    
    def _append_ops(self, row: dict):
        """
        ✅ ops_history.csv 기록 (모니터링/감사 원장)
        - RECONCILE / REJECT / FAIL / FREEZE / HEARTBEAT / GUARD / VERIFY_* 등
        """
        try:
            cols = self._ops_columns()
            base = {c: None for c in cols}
            base.update(row or {})

            # ✅ 최소 필드 강제 (감사 원장 안정화)
            if base.get("dt") is None:
                base["dt"] = str(pd.Timestamp.utcnow())
            if base.get("event") is None:
                base["event"] = "OPS"
            if base.get("mode") is None:
                base["mode"] = "LIVE"

            # ✅ 문자열 안정성
            if base.get("reason") is not None:
                r = str(base["reason"])
                r = r.replace("\n", " ").replace("\r", " ").replace(",", ";")
                base["reason"] = r

            if base.get("severity") is None:
                base["severity"] = "INFO"

            df = pd.DataFrame([base], columns=cols)
            df.to_csv(self.ops_history_path, mode="a", header=False, index=False)

        except Exception as e:
            logger.error(f"Ops history write failed: {e}")


    def _append_history(self, row: dict):
        """
        ✅ 단일 호출 인터페이스 유지하면서, event에 따라 자동 분기 기록
        - trade_history.csv : ENTRY / ENTRY_PENDING / ENTRY_SKIP / UPDATE_SL / EXIT
        (거래 흐름 추적에 필요한 이벤트는 trade 원장에 남긴다)
        - ops_history.csv   : 그 외 운영 이벤트 전부
        """
        try:
            ev = str((row or {}).get("event", "")).upper().strip()

            # ---- trade ledger (확장)
            if ev in {"ENTRY", "ENTRY_PENDING", "ENTRY_SKIP", "UPDATE_SL", "EXIT"}:
                cols = self._history_columns()
                base = {c: None for c in cols}
                base.update(row or {})
                # ✅ 최소 필드 강제 (감사 원장 안정화)
                if base.get("dt") is None:
                    base["dt"] = str(pd.Timestamp.utcnow())
                if base.get("event") is None:
                    base["event"] = "OPS"
                if base.get("mode") is None:
                    base["mode"] = "LIVE"



                # CSV 안정성
                if base.get("reason") is not None:
                    r = str(base["reason"])
                    r = r.replace("\n", " ").replace("\r", " ").replace(",", ";")
                    base["reason"] = r

                df = pd.DataFrame([base], columns=cols)
                df.to_csv(self.history_path, mode="a", header=False, index=False)
                return

            # ---- ops ledger (default)
            sev = (row or {}).get("severity", None)
            if sev is None:
                if any(k in ev for k in ["FAIL", "ERROR", "MISMATCH", "CRITICAL", "GUARD", "VERIFY"]):
                    sev = "WARN"
                else:
                    sev = "INFO"

            ops_row = dict(row or {})
            ops_row["severity"] = sev
            self._append_ops(ops_row)

        except Exception as e:
            logger.error(f"History write failed: {e}")

    # -----------------------------------------------------
    # LIVE State I/O
    # -----------------------------------------------------
    def _load_state(self):
        if not os.path.exists(self.state_path):
            self.executor.positions = {}
            self.last_processed_time = None
            self.last_bucket = None
            # ✅ pending limit entry orders (stateful)
            self.pending_entry_orders = {}
            return

        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            self.executor.positions = state.get("positions", {}) or {}
            self.last_processed_time = state.get("last_processed_time", None)
            self.last_bucket = state.get("last_bucket", None)

            # ✅ cooldowns / consecutive_losses (Backtest 동일)
            cds = state.get("cooldowns", {}) or {}
            cls = state.get("consecutive_losses", {}) or {}

            # cds: {sym: iso_str} -> {sym: Timestamp}
            out_cd = {}
            if isinstance(cds, dict):
                for k, v in cds.items():
                    try:
                        if v is None:
                            continue
                        out_cd[str(k)] = pd.to_datetime(v)
                    except Exception:
                        continue
            self.cooldowns = out_cd

            out_cl = {}
            if isinstance(cls, dict):
                for k, v in cls.items():
                    try:
                        out_cl[str(k)] = int(v)
                    except Exception:
                        continue
            self.consecutive_losses = out_cl




            # ✅ pending limit entry orders (stateful)
            peo = state.get("pending_entry_orders", {}) or {}
            self.pending_entry_orders = peo if isinstance(peo, dict) else {}

        except Exception as e:
            logger.error(f"State load failed: {e}")
            self.executor.positions = {}
            self.last_processed_time = None
            self.last_bucket = None
            self.pending_entry_orders = {}




    def _save_state(self):
        # cooldowns는 Timestamp -> iso string으로 저장
        cd_dump = {}
        try:
            if isinstance(getattr(self, "cooldowns", None), dict):
                for k, v in self.cooldowns.items():
                    try:
                        cd_dump[str(k)] = str(pd.to_datetime(v))
                    except Exception:
                        continue
        except Exception:
            cd_dump = {}

        cl_dump = {}
        try:
            if isinstance(getattr(self, "consecutive_losses", None), dict):
                for k, v in self.consecutive_losses.items():
                    try:
                        cl_dump[str(k)] = int(v)
                    except Exception:
                        continue
        except Exception:
            cl_dump = {}

        state = {
            "positions": self.executor.positions,
            "last_processed_time": self.last_processed_time,
            "last_bucket": self.last_bucket,
            "pending_entry_orders": getattr(self, "pending_entry_orders", {}) or {},
            # ✅ Backtest 동일: 쿨다운/연패 저장
            "cooldowns": cd_dump,
            "consecutive_losses": cl_dump,
        }

        try:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"State save failed: {e}")

    def _init_cooldown_state(self):
        """
        ✅ Backtest와 동일 쿨다운 상태 초기화 (LIVE)
        - self.cooldowns: {sym: pd.Timestamp}
        - self.consecutive_losses: {sym: int}
        - self.bar_td: 최소 1캔들 재진입 금지용 (15m)
        """
        try:
            self.bar_td = pd.Timedelta(minutes=15)
        except Exception:
            self.bar_td = pd.Timedelta(seconds=int(self.tf15_sec))

        if not hasattr(self, "cooldowns") or not isinstance(getattr(self, "cooldowns", None), dict):
            self.cooldowns = {}
        if not hasattr(self, "consecutive_losses") or not isinstance(getattr(self, "consecutive_losses", None), dict):
            self.consecutive_losses = {}

    def _is_in_cooldown(self, sym: str, now_t) -> bool:
        """
        ✅ now_t 기준 쿨다운 여부
        - cooldowns[sym] 가 now_t 보다 미래면 차단
        - LIVE 심볼키 혼용(full/base) 방어: variants 모두 검사
        """
        try:
            if not sym or not isinstance(getattr(self, "cooldowns", None), dict):
                return False

            now_ts = pd.to_datetime(now_t)

            try:
                keys = self._sym_variants(sym)
            except Exception:
                keys = [str(sym)]

            for k in keys:
                if k not in self.cooldowns:
                    continue

                until = self.cooldowns.get(k)
                if until is None:
                    try:
                        self.cooldowns.pop(k, None)
                    except Exception:
                        pass
                    continue

                until_ts = pd.to_datetime(until)

                if now_ts < until_ts:
                    return True

                # 만료면 정리
                try:
                    self.cooldowns.pop(k, None)
                except Exception:
                    pass

            return False

        except Exception:
            return False
        
    def _apply_cooldown_after_exit(self, sym: str, candle_t, exit_price: float, pos: dict):
        """
        ✅ Backtest와 동일한 cooldown 로직을 LIVE에 적용 (EXIT 직후 호출)

        규칙 (backtest_engine._execute_exit 동일):
        - net_pnl = pnl - fee
        pnl = (exit-entry)*amt (LONG) / (entry-exit)*amt (SHORT)
        fee = exit_notional * BASE_FEE
        - 최소 1캔들 재진입 금지: min_next = dt + self.bar_td
        - net_pnl > 0  -> consecutive_losses=0, cooldown=min_next
        - net_pnl <= 0 -> 연패에 따라 8/24/48/96h, cooldown=max(dt+wait, min_next)
        """
        try:
            sym_key = str(sym or "").strip()
            if not sym_key:
                return

            dt = pd.to_datetime(candle_t)

            px = float(exit_price or 0.0)
            if (not math.isfinite(px)) or (px <= 0):
                return

            p = pos if isinstance(pos, dict) else {}
            side = str(p.get("side", "")).upper()
            amt = float(p.get("amount", 0) or 0.0)
            entry = float(p.get("entry_price", 0) or 0.0)

            if side not in ("LONG", "SHORT"):
                return
            if amt <= 0 or entry <= 0:
                return

            # ✅ backtest 동일 pnl/fee 모델
            if side == "LONG":
                pnl = (px - entry) * amt
            else:
                pnl = (entry - px) * amt

            exit_value = abs(px * amt)
            fee = float(exit_value) * float(BASE_FEE)
            net_pnl = float(pnl - fee)

            # ✅ 동일 캔들 재진입 금지: 최소 1캔들
            min_next = dt + self.bar_td

            # ✅ 승/패에 따른 cooldown
            if net_pnl > 0:
                self.consecutive_losses[sym_key] = 0
                self.cooldowns[sym_key] = min_next
            else:
                streak = int(self.consecutive_losses.get(sym_key, 0)) + 1
                self.consecutive_losses[sym_key] = int(streak)

                if streak == 1:
                    wait_hours = 8
                elif streak == 2:
                    wait_hours = 24
                elif streak == 3:
                    wait_hours = 48
                else:
                    wait_hours = 96

                cd = dt + pd.Timedelta(hours=int(wait_hours))
                self.cooldowns[sym_key] = max(cd, min_next)

        except Exception:
            return

    def _clear_entry_freeze(self, reason: str = "cleared"):
        self.freeze_new_entries = False
        try:
            self.notifier._freeze_reason = str(reason)
            self.notifier._freeze_since_utc = None
        except Exception:
            pass

    def _freeze_meta(self) -> str:
        # heartbeat/telegram에 넣기 좋은 한 줄 요약
        try:
            r = getattr(self.notifier, "_freeze_reason", None)
            s = getattr(self.notifier, "_freeze_since_utc", None)
            if self.freeze_new_entries:
                return f"freeze=1 reason={r} since_utc={s}"
            return "freeze=0"
        except Exception:
            return f"freeze={int(self.freeze_new_entries)}"

    # -----------------------------------------------------
    # Restart Safety: account vs local state
    # -----------------------------------------------------
    def reconcile_positions(self):
        """
        ✅ 실계좌 포지션을 권위로 하여 로컬 포지션 상태를 동기화한다.
        - base_sym 기준으로 매칭
        - 거래소 권위 필드만 덮고(수량/사이드/진입가/마진), 관리 필드(sl/next_sl 등)는 유지

        ✅ 유령 포지션 고착 방지(핵심):
        - local_pos>0인데 fetch_positions가 "진짜로 0"을 지속 반환하는 경우,
        무한 가드로 고착되지 않도록 3회 연속 시 강제 정리한다.
        """
        if not hasattr(self.executor, "fetch_positions"):
            self.freeze_new_entries = False
            return

        if not hasattr(self, "_reconcile_real_zero_streak"):
            self._reconcile_real_zero_streak = 0

        local = self.executor.positions or {}
        local_count = int(len(local))

        try:
            real = self.executor.fetch_positions()
        except Exception as e:
            logger.error(f"fetch_positions failed: {e}")
            real = None

        if not isinstance(real, dict):
            logger.warning("⚠️ RECONCILE_GUARD | fetch_positions non-dict -> skip reconcile (no destructive changes)")
            self._append_ops({
                "dt": str(pd.Timestamp.utcnow()),
                "event": "RECONCILE_GUARD",
                "mode": "LIVE",
                "reason": "fetch_positions non-dict",
                "pos_count": local_count,
            })
            return

        # ---- real_open: amount>0 only, but INDEXED BY BASE KEY
        def _build_real_open_by_base(real_dict: dict):
            out = {}  # base -> (real_key, rp)
            for real_key, rp in (real_dict or {}).items():
                try:
                    amt = float((rp or {}).get("amount", 0) or 0)
                except Exception:
                    amt = 0.0
                if amt <= 0:
                    continue
                b = self._base_sym(real_key)
                if not b:
                    continue
                if b not in out:
                    out[b] = (real_key, rp)
            return out

        real_open_by_base = _build_real_open_by_base(real)
        real_count = int(len(real_open_by_base))

        # ✅ 핵심: local>0 & real_open=0 -> 1회 재조회 + 스트릭 + 3회 연속이면 강제 정리
        if local_count > 0 and real_count == 0:
            # 1) confirmation fetch
            try:
                real2 = self.executor.fetch_positions() or {}
            except Exception as e:
                real2 = None
                logger.error(f"fetch_positions confirm failed: {e}")

            if isinstance(real2, dict):
                real_open_by_base2 = _build_real_open_by_base(real2)
                if len(real_open_by_base2) > 0:
                    # confirmation에서 회복되면 정상 진행
                    real_open_by_base = real_open_by_base2
                    real_count = int(len(real_open_by_base2))

            if real_count == 0:
                self._reconcile_real_zero_streak += 1
                self.freeze_new_entries = True

                msg = f"local_pos={local_count} but real_open=0 (streak={self._reconcile_real_zero_streak})"
                logger.warning(f"⚠️ RECONCILE_GUARD_EMPTY | {msg}")

                self._append_ops({
                    "dt": str(pd.Timestamp.utcnow()),
                    "event": "RECONCILE_GUARD_EMPTY",
                    "mode": "LIVE",
                    "reason": msg,
                    "pos_count": local_count,
                })

                # 3회 연속이면: 유령 고착 방지 위해 로컬 강제 정리
                if self._reconcile_real_zero_streak >= 3:
                    cleared = list((self.executor.positions or {}).keys())
                    self.executor.positions = {}

                    self._append_ops({
                        "dt": str(pd.Timestamp.utcnow()),
                        "event": "RECONCILE_FORCE_CLEAR",
                        "mode": "LIVE",
                        "reason": f"forced clear local positions (count={len(cleared)}) after real_open=0 streak>=3",
                        "pos_count": 0,
                    })

                    try:
                        self.notifier.send(
                            title="RECONCILE_FORCE_CLEAR",
                            lines=[
                                f"cleared={len(cleared)}",
                                "reason=real_open=0 streak>=3",
                            ],
                        )
                    except Exception:
                        pass

                    # 강제정리 후에는 신규진입 동결 해제(단, 다음 루프에서 다시 reconcile로 검증됨)
                    self._clear_entry_freeze(reason="reconcile_force_clear")
                    self._reconcile_real_zero_streak = 0

                return

        # real_count 정상 -> 스트릭 리셋, freeze 해제(원인이 reconcile 계열이면)
        if real_count > 0:
            self._reconcile_real_zero_streak = 0
            if self.freeze_new_entries:
                self._clear_entry_freeze(reason="reconcile_ok")

        # (기존 로직 유지)
        if local_count >= 3 and real_count == 1:
            logger.warning(
                f"⚠️ RECONCILE_GUARD_PARTIAL | local_pos={local_count} real_open={real_count} -> skip reconcile (no removals)"
            )
            self._append_ops({
                "dt": str(pd.Timestamp.utcnow()),
                "event": "RECONCILE_GUARD_PARTIAL",
                "mode": "LIVE",
                "reason": f"local_pos={local_count} real_open={real_count}",
                "pos_count": local_count,
            })
            return

        MGMT_KEYS = {
            "side", "amount", "margin",
            "sl", "next_sl", "next_sl_bucket", "trail_sl",
            "entry_time", "entry_atr",
            "sl_breached", "sl_breached_sl",
            "sl_breached_at", "sl_breached_reason",
            "sl_breached_time",
            # ✅ 거래소 Position SL 메타 보존
            "exchange_sl_order_id", "exchange_sl_is_algo",
            "exchange_sl_stop_price", "exchange_sl_limit_price",
        }

        created_syms = []
        removed_syms = []
        updated_syms = []

        def _u(x): return str(x or "").upper()

        def _safe_float(x, default=None):
            try:
                v = float(x)
                if not math.isfinite(v):
                    return default
                return v
            except Exception:
                return default

        # ---- local base map (base -> local_key)
        local_by_base = {}
        for lk in (local or {}).keys():
            b = self._base_sym(lk)
            if b and (b not in local_by_base):
                local_by_base[b] = lk

        # ---- 제거 판단: base 기준
        for b, lk in list(local_by_base.items()):
            if b not in real_open_by_base:
                removed_syms.append(lk)

        # ---- 머지: base 기준으로 real 권위 필드만 반영, key는 로컬 우선
        merged = {}

        for b, (real_key, rp) in (real_open_by_base or {}).items():
            local_key = local_by_base.get(b)
            use_key = local_key if local_key else real_key

            lp = local.get(use_key, {}) if isinstance(local.get(use_key, {}), dict) else {}
            if (not lp) and local_key and (local_key in local):
                lp = local.get(local_key, {}) if isinstance(local.get(local_key, {}), dict) else {}

            keep_mgmt = {k: lp.get(k) for k in MGMT_KEYS if k in lp}

            mp = {}

            mp_side = _u((rp or {}).get("side")) or _u(lp.get("side"))
            mp_amt  = _safe_float((rp or {}).get("amount", None), _safe_float(lp.get("amount", 0), 0.0)) or 0.0
            mp["side"] = mp_side
            mp["amount"] = float(mp_amt)

            ep = (rp or {}).get("entry_price", None)
            if ep is not None:
                epf = _safe_float(ep, None)
                if epf is not None:
                    mp["entry_price"] = float(epf)
            elif lp.get("entry_price") is not None:
                mp["entry_price"] = lp.get("entry_price")

            mg = (rp or {}).get("margin", None)
            if mg is not None:
                mgf = _safe_float(mg, None)
                if mgf is not None:
                    mp["margin"] = float(mgf)
            elif lp.get("margin") is not None:
                mp["margin"] = lp.get("margin")

            mp.update(keep_mgmt)

            merged[use_key] = mp

            if use_key not in local:
                created_syms.append(use_key)
            else:
                if mp != local.get(use_key, {}):
                    updated_syms.append(use_key)

        # ---- 제거 적용
        for lk in (removed_syms or []):
            try:
                self.executor.positions.pop(lk, None)
            except Exception:
                pass

        # ---- 머지 적용
        self.executor.positions = merged

        # ---- ops log
        try:
            self._append_ops({
                "dt": str(pd.Timestamp.utcnow()),
                "event": "RECONCILE",
                "mode": "LIVE",
                "reason": f"created={len(created_syms)} removed={len(removed_syms)} updated={len(updated_syms)}",
                "pos_count": int(len(self.executor.positions or {})),
            })
        except Exception:
            pass


  

 


    def _sync_external_closes(self, authority_time) -> int:
        """
        ✅ 외부(수동) 청산 감지: 실계좌에 없는 로컬 포지션 제거

        ✅ 유령 고착 방지(핵심):
        - local_pos>0인데 real_open=0이 3회 연속이면, 외부청산으로 간주하고 로컬을 강제 정리한다.
        """
        local_positions = self.executor.positions or {}
        local_count = int(len(local_positions))
        if local_count == 0:
            return 0

        if not hasattr(self.executor, "fetch_positions"):
            return 0

        if not hasattr(self, "_external_real_zero_streak"):
            self._external_real_zero_streak = 0

        try:
            real = self.executor.fetch_positions() or {}
        except Exception as e:
            logger.error(f"[SYNC_EXTERNAL_CLOSE] fetch_positions failed: {e}")
            return 0

        if not isinstance(real, dict):
            logger.error("[SYNC_EXTERNAL_CLOSE] fetch_positions returned non-dict -> skip removals")
            self._append_ops({
                "dt": str(pd.Timestamp.utcnow()),
                "event": "EXTERNAL_CLOSE_SYNC_GUARD",
                "mode": "LIVE",
                "reason": "fetch_positions non-dict",
                "pos_count": local_count,
            })
            return 0

        def _build_real_open_set(real_dict: dict) -> set:
            out = set()
            for rsym, rp in (real_dict or {}).items():
                try:
                    amt = float((rp or {}).get("amount", 0) or 0)
                except Exception:
                    amt = 0.0
                if amt <= 0:
                    continue
                key = self._base_sym(rsym)
                if key:
                    out.add(key)
            return out

        real_open = _build_real_open_set(real)

        # ✅ local>0 & real_open=0 -> confirmation + streak + 3회 연속이면 강제 정리
        if local_count > 0 and len(real_open) == 0:
            # confirmation fetch
            try:
                real2 = self.executor.fetch_positions() or {}
            except Exception as e:
                real2 = None
                logger.error(f"[SYNC_EXTERNAL_CLOSE] fetch_positions confirm failed: {e}")

            if isinstance(real2, dict):
                real_open2 = _build_real_open_set(real2)
                if len(real_open2) > 0:
                    real_open = real_open2

            if len(real_open) == 0:
                self._external_real_zero_streak += 1
                self.freeze_new_entries = True

                msg = f"local_pos={local_count} but real_open=0 (streak={self._external_real_zero_streak})"
                logger.warning(f"⚠️ EXTERNAL_CLOSE_SYNC_GUARD | {msg}")

                self._append_ops({
                    "dt": str(pd.Timestamp.utcnow()),
                    "event": "EXTERNAL_CLOSE_SYNC_GUARD",
                    "mode": "LIVE",
                    "reason": msg,
                    "pos_count": local_count,
                })

                if self._external_real_zero_streak >= 3:
                    cleared = list((self.executor.positions or {}).keys())
                    self.executor.positions = {}

                    self._append_ops({
                        "dt": str(pd.Timestamp.utcnow()),
                        "event": "EXTERNAL_CLOSE_FORCE_CLEAR",
                        "mode": "LIVE",
                        "reason": f"forced clear local positions (count={len(cleared)}) after real_open=0 streak>=3",
                        "pos_count": 0,
                    })

                    try:
                        self.notifier.send(
                            title="EXTERNAL_CLOSE_FORCE_CLEAR",
                            lines=[
                                f"t={pd.to_datetime(authority_time)}",
                                f"cleared={len(cleared)}",
                                "reason=real_open=0 streak>=3",
                            ],
                        )
                    except Exception:
                        pass

                    self._clear_entry_freeze(reason="external_close_force_clear")
                    self._external_real_zero_streak = 0

                return 0

        # 정상 응답이면 스트릭 리셋 + freeze 해제
        if len(real_open) > 0:
            self._external_real_zero_streak = 0
            if self.freeze_new_entries:
                self._clear_entry_freeze(reason="external_sync_ok")

        # (기존 로직 유지: 실계좌에 없는 로컬 포지션 제거)
        if local_count >= 3 and len(real_open) == 1:
            logger.warning(
                f"⚠️ EXTERNAL_CLOSE_SYNC_GUARD_PARTIAL | local_pos={local_count} real_open={len(real_open)} -> skip removals"
            )
            self._append_ops({
                "dt": str(pd.Timestamp.utcnow()),
                "event": "EXTERNAL_CLOSE_SYNC_GUARD_PARTIAL",
                "mode": "LIVE",
                "reason": f"local_pos={local_count} real_open={len(real_open)}",
                "pos_count": local_count,
            })
            return 0

        removed = 0
        for lsym in list(local_positions.keys()):
            lkey = self._base_sym(lsym)
            if not lkey:
                continue
            if lkey in real_open:
                continue

            pos = local_positions.get(lsym) or {}
            removed += 1
            logger.warning(f"🟠 EXTERNAL_CLOSE_SYNC | {lsym} -> local pop")

            try:
                self.executor.positions.pop(lsym, None)
            except Exception:
                pass

            try:
                self.notifier.send(
                    title="EXTERNAL_CLOSE_SYNC",
                    lines=[
                        f"t={pd.to_datetime(authority_time)}",
                        f"symbol={lsym}",
                        f"side={str(pos.get('side','')).upper()}",
                        f"amount={pos.get('amount', None)}",
                    ],
                )
            except Exception:
                pass

        if removed > 0:
            self._append_ops({
                "dt": str(pd.to_datetime(authority_time)),
                "event": "EXTERNAL_CLOSE_SYNC",
                "mode": "LIVE",
                "reason": f"removed={removed}",
                "pos_count": int(len(self.executor.positions or {})),
            })

        return removed


    # -----------------------------------------------------
    # Indicators (engine responsibility) - 15m only
    # -----------------------------------------------------
    def rebuild_indicators(self):
        self.data_map = {}

        # ✅ Backtest와 동일: required_cols를 엔진 속성으로 고정
        self.required_cols = [
            "open", "high", "low", "close", "volume",
            "atr", "vol_ma", "ema_intra", "rsi", "adx", "st_val", "st_dir"
        ]

        temp_map = {}
        warmup_map = {}

        for sym, df in self.raw_data_map.items():
            try:
                ind = self.titan.calculate_indicators(sym, df.copy())

                missing = [c for c in self.required_cols if c not in ind.columns]
                if missing:
                    logger.warning(f"[Indicator] {sym} missing required cols: {missing}")
                    continue

                if ind[self.required_cols].isna().all().all():
                    logger.warning(f"[Indicator] {sym} required columns all-NaN (pre-slice).")
                    continue

                bad_cols = [c for c in self.required_cols if ind[c].isna().all()]
                if bad_cols:
                    logger.warning(f"[Indicator] {sym} has all-NaN required cols (pre-slice): {bad_cols}")
                    continue

                # ✅ Backtest와 동일: first_valid_index 기반 warmup 산정
                col_warmups = []
                for c in self.required_cols:
                    first_valid = ind[c].first_valid_index()
                    if first_valid is None:
                        col_warmups.append(len(ind))
                    else:
                        try:
                            col_warmups.append(ind.index.get_loc(first_valid))
                        except Exception:
                            col_warmups.append(0)

                sym_warmup = int(max(col_warmups)) if col_warmups else 0
                temp_map[sym] = ind
                warmup_map[sym] = sym_warmup

            except Exception as e:
                logger.warning(f"[Indicator] {sym} indicator failed: {e}")

        for sym, ind in temp_map.items():
            try:
                sym_warmup = int(warmup_map.get(sym, 0))
                sliced = ind.iloc[sym_warmup:].copy()

                if len(sliced) == 0:
                    logger.warning(f"[Indicator] {sym} empty after slice")
                    continue

                # ✅ Backtest와 동일: required subset dropna
                sliced = sliced.dropna(subset=self.required_cols)
                if len(sliced) == 0:
                    logger.warning(f"[Indicator] {sym} all NaN after drop (required subset)")
                    continue

                self.data_map[sym] = sliced

            except Exception as e:
                logger.warning(f"[Indicator] {sym} slice failed: {e}")

        logger.info(f"Indicators Ready: {len(self.data_map)} symbols processed.")




    def _align_data_map_common_timeline(self, min_need: int = 200):
        """
        ✅ Backtest 정합: (intersection + reindex + ffill) 강제 정렬
        - 교집합(common)이 짧으면 base_index(가장 긴 심볼)로 fallback 후 reindex+ffill
        - 마지막에 common_idx 교집합을 다시 만들고 충분 길이 확보되면 final reindex
        """
        try:
            if not isinstance(self.data_map, dict) or not self.data_map:
                return

            # 0) base symbol (longest)
            base_sym = None
            base_len = -1
            for s, df in self.data_map.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    if len(df) > base_len:
                        base_len = len(df)
                        base_sym = s

            if base_sym is None or base_len <= 0:
                logger.warning("⚠️ ALIGN_FAIL | no base symbol")
                return

            base_df = self.data_map[base_sym]
            base_index = base_df.index

            # ✅ required columns (백테와 동일)
            req = [
                "open", "high", "low", "close", "volume",
                "atr", "vol_ma", "ema_intra", "rsi", "adx", "st_val", "st_dir"
            ]

            # 1) common timeline (intersection)
            common_idx = None
            for sym, df in self.data_map.items():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                common_idx = df.index if common_idx is None else common_idx.intersection(df.index)

            # ✅ Backtest fallback: common이 짧으면 base_index로 fallback
            if common_idx is None:
                common_idx = base_index
            if len(common_idx) < (min_need + 5):
                logger.warning(
                    f"⚠️ ALIGN_FALLBACK | common_timeline too short -> use base_timeline "
                    f"(need>={min_need}) common_len={len(common_idx)} base_len={len(base_index)} base_sym={base_sym}"
                )
                common_idx = base_index

            # base 자체가 너무 짧으면 여기서 끝 (prepare_data에서 더 끌어오는 쪽으로)
            if len(common_idx) < (min_need + 5):
                logger.warning(
                    f"⚠️ ALIGN_ABORT | base_timeline too short (need>={min_need}) base_len={len(common_idx)} base_sym={base_sym}"
                )
                return

            # 2) reindex + ffill (1차)
            aligned_full_map = {}
            for sym, df in self.data_map.items():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                try:
                    aligned_full = df.reindex(common_idx).copy()
                    aligned_full = aligned_full.ffill()
                except Exception:
                    continue

                if aligned_full.empty:
                    continue

                # 시작점 필수 컬럼 유효성
                try:
                    if any(c not in aligned_full.columns for c in req):
                        logger.warning(f"⚠️ ALIGN_DROP | {sym} missing required cols")
                        continue
                    if aligned_full[req].iloc[0].isna().any():
                        logger.warning(f"⚠️ ALIGN_DROP | {sym} NaN at aligned start after ffill")
                        continue
                except Exception:
                    logger.warning(f"⚠️ ALIGN_DROP | {sym} missing required cols at aligned start")
                    continue

                aligned_full_map[sym] = aligned_full

            if not aligned_full_map:
                logger.warning("⚠️ ALIGN_FAIL | no usable symbols after reindex/ffill")
                return

            # 3) 최종 common_idx (교집합 재계산)
            final_common = None
            for sym, df in aligned_full_map.items():
                try:
                    final_common = df.index if final_common is None else final_common.intersection(df.index)
                except Exception:
                    continue

            # 교집합이 충분히 길면 final_common으로 2차 정렬, 아니면 1차 common_idx 유지(백테 fallback 성격)
            if final_common is not None and len(final_common) >= (min_need + 5):
                use_idx = final_common
            else:
                use_idx = common_idx
                logger.warning(
                    f"⚠️ ALIGN_FINAL_FALLBACK | intersection too short after reindex -> keep base_timeline "
                    f"len={len(use_idx)}"
                )

            # 4) final reindex
            final_map = {}
            for sym, df in aligned_full_map.items():
                try:
                    final_df = df.reindex(use_idx).copy()
                except Exception:
                    continue
                if final_df.empty:
                    continue
                try:
                    if final_df[req].iloc[0].isna().any():
                        continue
                except Exception:
                    continue
                final_map[sym] = final_df

            if not final_map:
                logger.warning("⚠️ ALIGN_FAIL | final_map empty")
                return

            self.data_map = final_map
            self.common_time_index = use_idx
            self.symbols = sorted(self.data_map.keys())

            logger.info(
                f"✅ ALIGN_OK | base_sym={base_sym} base_len={base_len} "
                f"aligned_syms={len(self.symbols)} common_len={len(use_idx)}"
            )

        except Exception as e:
            logger.error(f"ALIGN_CRASH | {e}")
            return



    



    # -----------------------------------------------------
    # Data preparation (15m)
    # -----------------------------------------------------
    def prepare_data(self):
        # ----------------------------
        # ✅ Symbol Canon (state key)
        # ----------------------------
        def _canon(sym: str) -> str:
            """
            state 표준키: XXX/USDT:USDT
            - 이미 :USDT가 있으면 유지
            - XXX/USDT -> XXX/USDT:USDT
            - XXXUSDT  -> XXX/USDT:USDT (최소 휴리스틱)
            """
            try:
                s = str(sym or "").strip()
            except Exception:
                return str(sym)

            if not s:
                return s

            # 이미 선물 표준
            if ":USDT" in s:
                return s

            # 'BTC/USDT' 형태
            if "/" in s:
                if s.endswith("/USDT"):
                    return s + ":USDT"
                return s

            # 'BTCUSDT' 형태 최소 처리
            if s.endswith("USDT") and len(s) > 4:
                base = s[:-4]
                return f"{base}/USDT:USDT"

            return s

        # -------------------------------------------------
        # ✅ Universe Source Priority:
        # 1) root_dir/universe.json
        # 2) executor.get_top_targets()
        # -------------------------------------------------
        universe_path = os.path.join(self.root_dir, "universe.json")
        targets = []
        source = "TOP_TARGETS"

        if os.path.exists(universe_path):
            try:
                with open(universe_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)

                cand = []
                if isinstance(obj, list):
                    cand = obj
                elif isinstance(obj, dict):
                    for k in ("universe", "symbols", "targets"):
                        v = obj.get(k)
                        if isinstance(v, list):
                            cand = v
                            break

                # normalize + dedup (order-preserving)
                cand = [str(x).strip() for x in (cand or []) if str(x).strip()]
                cand = list(dict.fromkeys(cand))

                if cand:
                    targets = cand
                    source = "ROOT_UNIVERSE_JSON"
                else:
                    logger.warning(f"⚠️ UNIVERSE_JSON_EMPTY | path={universe_path}")

            except Exception as e:
                logger.warning(f"⚠️ UNIVERSE_JSON_READ_FAIL | path={universe_path} err={e}")

        if not targets:
            targets = self.executor.get_top_targets() or []
            source = "TOP_TARGETS"

        targets = [_canon(s) for s in targets]
        logger.info(f"📡 MARKET_SCAN | source={source} ct={len(targets)} targets={targets}")

        # ✅ positions 키도 state 표준키로 정규화 (중복/미스매치 방지)
        try:
            old_pos = self.executor.positions or {}
            if isinstance(old_pos, dict) and old_pos:
                new_pos = {}
                for k, v in old_pos.items():
                    nk = _canon(k)
                    # 충돌 시: 기존 값 우선(이미 state에서 온 키가 더 신뢰됨)
                    if nk in new_pos:
                        continue
                    new_pos[nk] = v
                self.executor.positions = new_pos
        except Exception:
            pass

        pos_syms = list((self.executor.positions or {}).keys())
        combined = list(dict.fromkeys(list(targets) + list(pos_syms)))

        # ✅ blacklist 체크도 canon 기준으로 수행
        filtered = []
        for sym in combined:
            sym_c = _canon(sym)
            clean = sym_c.split(":")[0]
            if clean in self.titan.blacklist or sym_c in self.titan.blacklist:
                continue
            filtered.append(sym_c)

        if not filtered:
            logger.error("❌ No targets after blacklist+positions filter.")
            self.raw_data_map = {}
            self.data_map = {}
            self.symbols = []
            return

        try:
            out_of_universe = [s for s in pos_syms if s not in set(targets)]
            if out_of_universe:
                logger.info(f"🧩 DATA_EXTEND | positions_outside_universe={out_of_universe[:10]} ct={len(out_of_universe)}")
        except Exception:
            pass

        raw_map = self.executor.prepare_data(filtered) or {}

        # ✅ raw_map 키도 canon으로 정규화해서 validated_map에 적재
        validated_map = {}
        for sym, df in raw_map.items():
            sym_c = _canon(sym)

            v = DataLoader.validate_and_format(df)
            if v is None:
                logger.warning(f"⚠️ DATA_DROP | {sym_c} validate failed")
                continue

            v = v.copy()
            v["datetime"] = pd.to_datetime(v["timestamp"], unit="ms", errors="coerce")
            v = v.dropna(subset=["datetime"]).set_index("datetime")
            if v.empty:
                continue

            # 충돌 시: 길이가 더 긴 데이터 우선
            if sym_c in validated_map:
                try:
                    if len(v) > len(validated_map[sym_c]):
                        validated_map[sym_c] = v
                except Exception:
                    validated_map[sym_c] = v
            else:
                validated_map[sym_c] = v

        # ✅ Titan9 최소 길이 보장(200+)을 위해 15m 히스토리 보강
        min_len = int(self._t9_min_len())
        validated_map = self._ensure_min_history_15m(validated_map, min_bars=min_len)

        self.raw_data_map = {k: validated_map[k] for k in sorted(validated_map.keys())}
        self.rebuild_indicators()

        # ✅ 공통 타임라인 정렬(백테와 동일)
        try:
            self._align_data_map_common_timeline(min_need=min_len)
        except Exception as e:
            logger.error(f"[prepare_data] align_data_map_common_timeline failed: {e}")

        # ✅ symbols는 data_map(=ready) 키 기준(표준키)
        self.symbols = sorted(self.data_map.keys())
        logger.info(f"📊 INDICATORS_READY | symbols={self.symbols}")
        logger.info(f"📥 Data prepared: raw={len(self.raw_data_map)} | ready={len(self.data_map)}")
    def _ensure_min_history_15m(self, validated_map: dict, min_bars: int) -> dict:
        """
        ✅ LIVE에서 Titan9 최소 길이(min_bars) 충족하도록 15m 히스토리를 보강 로드
        - executor.prepare_data()가 짧게 가져오는 경우가 있어 exchange.fetch_ohlcv로 추가 확보
        - 실패하면 기존 validated_map 유지

        정합 강화:
        - EMA200/ATR(SuperTrend) 웜업을 위해 최소 600봉을 강제 (BT 대비 지표/게이트 뒤집힘 방지)
        """
        out = dict(validated_map or {})
        try:
            ex = getattr(self.executor, "exchange", None)
            if ex is None or not hasattr(ex, "fetch_ohlcv"):
                return out

            # ✅ 최소 600봉 확보(웜업)
            want = int(min_bars)
            want = max(want, 600)
            want = want + 50  # 여유분(누락/정렬/지표계산)

            for sym, df in list(out.items()):
                try:
                    if not isinstance(df, pd.DataFrame) or df.empty:
                        continue
                    if len(df) >= want:
                        continue

                    ohlcv = ex.fetch_ohlcv(sym, timeframe="15m", limit=want)
                    if not ohlcv:
                        continue

                    tmp = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                    tmp = DataLoader.validate_and_format(tmp)
                    if tmp is None or tmp.empty:
                        continue

                    tmp = tmp.copy()
                    tmp["datetime"] = pd.to_datetime(tmp["timestamp"], unit="ms", errors="coerce")
                    tmp = tmp.dropna(subset=["datetime"]).set_index("datetime")
                    if tmp.empty:
                        continue

                    out[sym] = tmp
                except Exception:
                    continue

        except Exception:
            return out

        return out
    # -----------------------------------------------------
    # Time / Loop Authority
    # -----------------------------------------------------
    def _current_bucket_15m(self):
        """
        ✅ 서버시간 기반 15m bucket
        - prepare_data 트리거/expire_bucket 등 루프 전반에서 동일한 authority 사용
        """
        try:
            now_ms = int(self._server_now_ms())
            return int((now_ms // 1000) // self.tf15_sec)
        except Exception:
            return int(time.time() // self.tf15_sec)

    def _bucket_15m_from_candle(self, candle_t) -> int:
        """
        candle_t(캔들 timestamp) 기준의 15m bucket.
        - sl_apply_mode=next에서 same-candle 승계를 막기 위한 기준
        """
        try:
            t = pd.to_datetime(candle_t, utc=True)
        except Exception:
            t = pd.Timestamp.utcnow().tz_localize("UTC")

        try:
            # tz-naive면 UTC로 가정
            if getattr(t, "tzinfo", None) is None:
                t = t.tz_localize("UTC")
        except Exception:
            pass

        try:
            return int(int(t.timestamp()) // int(self.tf15_sec))
        except Exception:
            return int(time.time() // int(self.tf15_sec))


    def _server_now_ms(self) -> int:
        """
        ✅ LIVE 단일 시간 권위
        - 거래소 서버시간(exchange.milliseconds) 우선
        - 실패 시 로컬 time.time() fallback
        """
        try:
            ex = getattr(self.executor, "exchange", None)
            if ex is not None and hasattr(ex, "milliseconds"):
                return int(ex.milliseconds())
        except Exception:
            pass
        return int(time.time() * 1000)

    def _next_bucket_open_ms_15m(self, now_ms: int) -> int:
        """
        ✅ 서버시간 기준 다음 15m 경계(open) ms
        """
        tf_ms = int(self.tf15_sec * 1000)
        b_open = (int(now_ms) // tf_ms) * tf_ms
        return int(b_open + tf_ms)


    def _pick_current_time_15m(self, candle_ts_mode: str = None):
        """
        ✅ LIVE Time Authority (Backtest-Identical: data-index authority)
        반환: (ref_time: pd.Timestamp | None, active_symbols: list[str])

        ✅ 핵심 정책:
        - ref는 "최신 캔들"로 고정한다. (ref = max(last_ts))
        - 커버리지 때문에 ref를 과거로 당기지 않는다.
        - active_symbols는 ref 시각을 포함하고, ref 대비 lag가 stale_lag_sec 이내인 심볼만 통과.

        ✅ Time Authority Sealing:
        - 거래소가 '막 열린 15m 캔들(미확정)'을 마지막 row로 포함하는 경우가 있어
        ref가 닫히지 않은 캔들이면 직전 확정 캔들로 봉인한다.
        - 중요: "uniq last_ts가 1개뿐"이어도 df에는 직전 캔들이 존재하므로
        대표 df.index[-2]로 봉인해야 한다. (엔진 중단 금지)
        """

        # 0) mode (라벨만)
        try:
            if candle_ts_mode is not None:
                mode = str(candle_ts_mode).strip().lower()
            else:
                mode = str(self.cfg.get("system_settings", {}).get("candle_ts_mode", "open")).strip().lower()
        except Exception:
            mode = "open"
        if mode not in ("open", "close"):
            mode = "open"

        if not self.data_map or not self.symbols:
            return None, []

        stale_lag_sec = int(self.cfg.get("system_settings", {}).get("stale_lag_sec", 3600) or 3600)

        # 1) last_ts_map + 대표 df(봉인 fallback용)
        last_ts_map = {}
        rep_df = None
        rep_sym = None
        rep_len = -1

        for sym in self.symbols:
            df = self.data_map.get(sym)
            if df is None or df.empty:
                continue
            try:
                last_ts_map[sym] = df.index[-1]
                # 대표 df는 "가장 긴 df"로 선택 ([-2] 확보 확률 최대)
                if len(df) > rep_len:
                    rep_len = len(df)
                    rep_df = df
                    rep_sym = sym
            except Exception:
                continue

        if not last_ts_map:
            logger.warning("⚠️ TIME_AUTHORITY_EMPTY | last_ts_map=0")
            return None, []

        # 2) ref 고정 = 최신 캔들
        try:
            ref = max(last_ts_map.values())
        except Exception:
            logger.warning("⚠️ TIME_AUTHORITY_EMPTY | ref=None")
            return None, []

        # 2.5) ✅ 미확정 15m 캔들 봉인 (엔진 중단 금지)
        try:
            bar_sec = 15 * 60
            now_utc = pd.Timestamp.utcnow()
            if getattr(now_utc, "tzinfo", None) is not None:
                now_utc = now_utc.tz_convert(None)

            if now_utc < (ref + pd.Timedelta(seconds=bar_sec)):
                uniq = sorted(set(last_ts_map.values()))

                sealed = None

                # (A) 여러 타임스탬프가 있으면 그중 하나 전으로 봉인
                if len(uniq) >= 2:
                    sealed = uniq[-2]

                # (B) uniq가 1개뿐이어도, df 자체에는 [-2]가 있으므로 대표 df에서 봉인
                if sealed is None:
                    try:
                        if rep_df is not None and len(rep_df.index) >= 2:
                            sealed = rep_df.index[-2]
                    except Exception:
                        sealed = None

                # (C) 봉인 성공 시 ref 교체, 실패 시 ref 유지(중단 금지)
                if sealed is not None:
                    logger.info(
                        f"🧷 TIME_AUTH_SEAL | drop_unclosed_ref={ref} -> sealed_ref={sealed} "
                        f"now_utc={now_utc} rep_sym={rep_sym}"
                    )
                    ref = sealed
                else:
                    logger.warning(
                        f"⚠️ TIME_AUTH_SEAL_SKIP | cannot_seal (keep ref) ref={ref} now_utc={now_utc}"
                    )

        except Exception:
            # 봉인 로직 예외는 ref 유지
            pass

        # 3) active_symbols 선별: (A) ref 포함 (B) ref 대비 lag <= stale_lag_sec
        active = []
        for sym, last_ts in last_ts_map.items():
            try:
                df = self.data_map.get(sym)
                if df is None or df.empty:
                    continue
                if ref not in df.index:
                    continue

                lag = (ref - last_ts).total_seconds()
                if lag > stale_lag_sec:
                    continue

                active.append(sym)
            except Exception:
                continue

        if not active:
            logger.warning(
                f"⚠️ TIME_AUTHORITY_NONE | t_ref={ref} mode={mode} active=0 stale_lag_sec={stale_lag_sec}"
            )
            return None, []

        cov = len(active) / float(len(last_ts_map))
        if cov < 1.0:
            missing = [s for s in last_ts_map.keys() if s not in set(active)]
            logger.warning(
                f"⚠️ TIME_AUTHORITY_PARTIAL | t_ref={ref} mode={mode} coverage={cov:.2%} "
                f"active={len(active)}/{len(last_ts_map)} missing_sample={missing[:10]}"
            )
        else:
            logger.info(
                f"✅ TIME_AUTHORITY_OK | t_ref={ref} mode={mode} coverage=100% syms={len(active)}"
            )

        return ref, active


    def _authority_time_15m(self, candle_ts_mode: str = None):
        """
        ✅ 호환 계약:
        - 반환: (current_time, active_symbols)
        - 내부 구현은 _pick_current_time_15m()를 단일 진실원으로 사용
        """
        return self._pick_current_time_15m(candle_ts_mode=candle_ts_mode)


    def _sleep_until_next_15m(self, candle_ts_mode: str = None):
        """
        ✅ 루프 정렬은 항상 'next 15m open boundary' 기준
        - candle_ts_mode는 "ref(권위시각)" 표기만 다르게 한다.
        * mode=open  -> next_ref = next_boundary_open - tf
        * mode=close -> next_ref = next_boundary_open
        """
        # 0) mode 결정
        try:
            if candle_ts_mode is not None:
                mode = str(candle_ts_mode).strip().lower()
            else:
                mode = str(self.cfg.get("system_settings", {}).get("candle_ts_mode", "open")).strip().lower()
        except Exception:
            mode = "open"
        if mode not in ("open", "close"):
            mode = "open"

        tf_ms = int(self.tf15_sec * 1000)

        try:
            now_ms = int(self._server_now_ms())
        except Exception:
            now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)

        # ✅ 항상 next 15m "open boundary"
        next_bucket_open_ms = ((now_ms // tf_ms) + 1) * tf_ms
        wait_sec = max(0.0, (next_bucket_open_ms - now_ms) / 1000.0)

        # ref 표기(체감 혼동 제거용)
        next_boundary_open = pd.to_datetime(next_bucket_open_ms, unit="ms", utc=True).tz_convert(None)
        if mode == "open":
            next_ref = next_boundary_open - pd.Timedelta(seconds=int(self.tf15_sec))
        else:
            next_ref = next_boundary_open

        logger.info(
            f"⏳ WAIT_NEXT_15M | boundary=open({next_boundary_open}) | mode={mode} -> next_ref={next_ref} | sleep={wait_sec:.2f}s"
        )

        try:
            time.sleep(wait_sec)
        except Exception:
            pass


    def _get_sl_apply_mode(self) -> str:
        """
        system_settings.sl_apply_mode: same/next
        ✅ LIVE 기본값은 same
        """
        try:
            mode = str(self.cfg.get("system_settings", {}).get("sl_apply_mode", "same")).strip().lower()
        except Exception:
            mode = "same"
        return mode if mode in ("next", "same") else "same"
    

    def _get_sl_strategy(self) -> str:
        """
        system_settings.sl_strategy
        허용: supertrend | atr_trail | profit_lock | hybrid | armor
        (미설정 시 supertrend)
        """
        try:
            strat = str(self.cfg.get("system_settings", {}).get("sl_strategy", "supertrend")).strip().lower()
        except Exception:
            strat = "supertrend"

        allowed = {"supertrend", "atr_trail", "profit_lock", "hybrid", "armor"}
        return strat if strat in allowed else "supertrend"


    def _get_sl_params(self) -> dict:
        """
        system_settings.sl_params (dict)
        예)
        system_settings:
            sl_params:
            atr_mult: 3.0
            trigger_atr: 2.0
            lock_atr: 0.5
        """
        try:
            p = self.cfg.get("system_settings", {}).get("sl_params", {}) or {}
            return p if isinstance(p, dict) else {}
        except Exception:
            return {}

    def _exchange_sl_enabled(self) -> bool:
        try:
            return bool((self.cfg.get("system_settings", {}) or {}).get("exchange_position_sl_enabled", True))
        except Exception:
            return True

    def _exchange_sl_limit_gap_pct(self) -> float:
        """
        Stop-Limit 체결 보정폭
        - 기본 0.2%
        """
        try:
            v = float((self.cfg.get("system_settings", {}) or {}).get("exchange_position_sl_limit_gap_pct", 0.002) or 0.002)
        except Exception:
            v = 0.002
        if (not math.isfinite(v)) or v < 0:
            v = 0.002
        return min(max(v, 0.0001), 0.02)

    def _calc_exchange_sl_prices(self, pos: dict, active_sl: float):
        """
        active_sl 자체를 거래소 Position SL로 미러링
        - stopPrice = active_sl
        - limitPrice = stopPrice보다 아주 조금 더 불리하게
        """
        try:
            side = str((pos or {}).get("side", "")).upper().strip()
        except Exception:
            side = ""

        try:
            sl = float(active_sl or 0.0)
        except Exception:
            sl = 0.0

        if side not in ("LONG", "SHORT"):
            return 0.0, 0.0
        if sl <= 0 or (not math.isfinite(sl)):
            return 0.0, 0.0

        gap = self._exchange_sl_limit_gap_pct()

        if side == "LONG":
            stop_px = float(sl)
            limit_px = float(sl * (1.0 - gap))
        else:
            stop_px = float(sl)
            limit_px = float(sl * (1.0 + gap))

        if stop_px <= 0 or limit_px <= 0:
            return 0.0, 0.0

        return float(stop_px), float(limit_px)

    def _sync_exchange_position_sl(self, sym: str, pos: dict, active_sl: float, reason: str = "SYNC"):
        """
        ✅ active SL -> 거래소 Position SL(Stop-Limit + Mark trigger) 미러링
        """
        if not self._exchange_sl_enabled():
            return
        if not hasattr(self.executor, "ensure_position_sl_limit_mark"):
            return
        if not isinstance(pos, dict) or not sym:
            return

        try:
            amt = float(pos.get("amount", 0.0) or 0.0)
        except Exception:
            amt = 0.0
        if amt <= 0 or (not math.isfinite(amt)):
            return

        stop_px, limit_px = self._calc_exchange_sl_prices(pos, active_sl)
        prev_id = str(pos.get("exchange_sl_order_id") or "")
        prev_is_algo = bool(pos.get("exchange_sl_is_algo", False))

        if stop_px <= 0 or limit_px <= 0:
            if prev_id:
                try:
                    self.executor._cancel_conditional_order_safe(sym, prev_id, prev_is_algo)
                except Exception:
                    pass
            pos["exchange_sl_order_id"] = None
            pos["exchange_sl_is_algo"] = False
            pos["exchange_sl_stop_price"] = None
            pos["exchange_sl_limit_price"] = None
            try:
                self._save_state()
            except Exception:
                pass
            return

        try:
            prev_stop = float(pos.get("exchange_sl_stop_price") or 0.0)
        except Exception:
            prev_stop = 0.0
        try:
            prev_limit = float(pos.get("exchange_sl_limit_price") or 0.0)
        except Exception:
            prev_limit = 0.0

        try:
            same_stop = prev_stop > 0 and abs((stop_px / prev_stop) - 1.0) <= 0.0005
            same_limit = prev_limit > 0 and abs((limit_px / prev_limit) - 1.0) <= 0.0005
        except Exception:
            same_stop, same_limit = False, False

        if prev_id and same_stop and same_limit:
            return

        try:
            out = self.executor.ensure_position_sl_limit_mark(
                symbol=sym,
                position_side=str(pos.get("side") or "").upper(),
                amount=float(amt),
                stop_price=float(stop_px),
                limit_price=float(limit_px),
                prev_order_id=prev_id,
                prev_is_algo=prev_is_algo,
                trigger="MARK_PRICE",
            )
        except Exception as e:
            logger.error(f"EXCHANGE_SL sync failed {sym}: {e}")
            out = {"order_id": "", "is_algo": False}

        oid = str((out or {}).get("order_id") or "")
        if not oid:
            pos["exchange_sl_order_id"] = None
            pos["exchange_sl_is_algo"] = False
            pos["exchange_sl_stop_price"] = None
            pos["exchange_sl_limit_price"] = None
            try:
                self._save_state()
            except Exception:
                pass
            return

        pos["exchange_sl_order_id"] = str(oid)
        pos["exchange_sl_is_algo"] = bool((out or {}).get("is_algo", False))
        pos["exchange_sl_stop_price"] = float(stop_px)
        pos["exchange_sl_limit_price"] = float(limit_px)

        try:
            self._save_state()
        except Exception:
            pass



    # [MOD] def _fetch_realtime_snapshot(self, symbols: list) -> dict
    def _fetch_realtime_snapshot(self, symbols: list) -> dict:
        """
        ✅ Realtime Snapshot (루프 1회)
        - 반환: {sym: {"last":float|None, "high":float|None, "low":float|None, "ts":utc_str, "_raw":dict(optional)}}
        - bulk 메서드(fetch_tickers)가 있으면 우선 사용
        """
        out = {}
        syms = [s for s in (symbols or []) if s]
        if not syms:
            return out

        now_utc = str(pd.Timestamp.utcnow())

        def _sf(x):
            try:
                v = float(x)
                if not math.isfinite(v) or v <= 0:
                    return None
                return v
            except Exception:
                return None

        # ---- 1) bulk 지원 시
        if hasattr(self.executor, "fetch_tickers"):
            try:
                ticks = self.executor.fetch_tickers(syms) or {}
                if isinstance(ticks, dict):
                    for sym in syms:
                        t = ticks.get(sym) or {}
                        px = self._extract_px(t)  # ✅ 핵심: 키 다양성 흡수
                        hi = _sf((t or {}).get("high"))
                        lo = _sf((t or {}).get("low"))
                        out[sym] = {"last": px, "high": hi, "low": lo, "ts": now_utc, "_raw": t}
                    return out
            except Exception:
                pass

        # ---- 2) fallback: per-symbol fetch_ticker
        for sym in syms:
            try:
                t = self.executor.fetch_ticker(sym) or {}
            except Exception:
                t = {}

            px = self._extract_px(t)  # ✅ 핵심
            hi = _sf((t or {}).get("high"))
            lo = _sf((t or {}).get("low"))
            out[sym] = {"last": px, "high": hi, "low": lo, "ts": now_utc, "_raw": t}

        return out

 
    def _base_sym(self, sym: str) -> str:
        """
        ✅ 비교/매칭용 base 심볼
        - "BTC/USDT:USDT" -> "BTC/USDT"
        """
        try:
            s = str(sym or "").strip()
            if not s:
                return ""
            return s.split(":")[0].strip()
        except Exception:
            return str(sym or "")

    def _sym_variants(self, sym: str) -> list:
        """
        ✅ canon 강제 금지 버전:
        - 저장/표시는 원문 키 유지
        - 비교/탐색만 base 변형을 함께 본다
        """
        s = str(sym or "").strip()
        b = self._base_sym(s)
        out = []
        if s:
            out.append(s)
        if b and b not in out:
            out.append(b)
        return out

    def _find_key(self, mp: dict, sym: str) -> str:
        """
        mp(dict)에서 sym의 (full/base) 중 존재하는 실제 key를 반환
        """
        if not isinstance(mp, dict) or not mp:
            return None
        for k in self._sym_variants(sym):
            if k in mp:
                return k
        return None


    def _extract_px(self, obj):
        if obj is None:
            return None

        # 숫자 바로 들어온 경우
        if not isinstance(obj, dict):
            try:
                v = float(obj)
                return v if math.isfinite(v) and v > 0 else None
            except Exception:
                return None

        # 1️⃣ top-level 키
        keys = [
            "last", "close",
            "lastPrice", "markPrice", "indexPrice",
            "price", "mark",
            "bid", "ask",
        ]

        for k in keys:
            try:
                v = obj.get(k)
                if v is None:
                    continue
                v = float(v)
                if math.isfinite(v) and v > 0:
                    return v
            except Exception:
                pass

        # 2️⃣ info 중첩
        info = obj.get("info")
        if isinstance(info, dict):
            for k in ["lastPrice", "markPrice", "indexPrice", "price", "last"]:
                try:
                    v = info.get(k)
                    if v is None:
                        continue
                    v = float(v)
                    if math.isfinite(v) and v > 0:
                        return v
                except Exception:
                    pass

        return None


    # -----------------------------------------------------
    # Candle Picking (15m stale blocking fix)
    # -----------------------------------------------------
    

    def _get_manage_row(self, sym, authority_time, realtime_snapshot: dict = None):
        """
        ✅ Backtest-Identical Manage Row
        - SL/EXIT 판단은 '15m 확정봉(데이터맵)'만 사용한다.
        - intrabar(1m) boost 금지
        - realtime_snapshot의 high/low 주입 금지(24h high/low 혼입 위험)
        - authority_time 캔들이 없으면:
            (A) df.index에서 authority_time 이하의 가장 최근 캔들(asof) 사용
            (B) 그것도 없으면 None 반환(=관리 스킵)
        """
        df = self.data_map.get(sym)

        if not hasattr(self, "_manage_row_err_once"):
            self._manage_row_err_once = {}

        def _log_once(key: str, msg: str):
            try:
                ref_tag = str(pd.to_datetime(authority_time)) if authority_time is not None else "None"
            except Exception:
                ref_tag = str(authority_time)

            prev = self._manage_row_err_once.get(key)
            if prev == ref_tag:
                return
            self._manage_row_err_once[key] = ref_tag

            logger.warning(msg)
            try:
                self._append_ops({
                    "dt": str(pd.Timestamp.utcnow()),
                    "event": "MANAGE_ROW_WARN",
                    "mode": "LIVE",
                    "symbol": sym,
                    "reason": f"{msg}",
                    "t_ref": ref_tag,
                    "pos_count": int(len(self.executor.positions or {})),
                })
            except Exception:
                pass

        if df is None or df.empty or authority_time is None:
            _log_once(f"none:{sym}", f"⚠️ MANAGE_ROW_NONE | {sym} df_empty_or_authority_none")
            return None

        # ---- 1) exact match 우선
        try:
            if authority_time in df.index:
                s = df.loc[authority_time]
                if isinstance(s, pd.Series):
                    s = s.copy(deep=True)
                    s.name = authority_time
                    return s
        except Exception as e:
            _log_once(f"row:{sym}", f"⚠️ MANAGE_ROW select failed | {sym} | {type(e).__name__}: {e}")

        # ---- 2) asof(<= authority_time) fallback (Backtest에서 timeline mismatch 때의 유사 동작)
        try:
            idx = df.index
            # idx는 정렬되어 있다는 전제(데이터 무결성)
            prev_idx = idx[idx <= authority_time]
            if len(prev_idx) > 0:
                ref2 = prev_idx[-1]
                s = df.loc[ref2]
                if isinstance(s, pd.Series):
                    s = s.copy(deep=True)
                    # name을 authority_time으로 바꾸지 않는다: 실제 row timestamp를 유지(디버깅/정합성)
                    s.name = ref2
                    _log_once(
                        f"asof:{sym}",
                        f"⚠️ MANAGE_ROW_ASOF | {sym} authority={authority_time} -> asof={ref2}"
                    )
                    return s
        except Exception as e:
            _log_once(f"asof:{sym}", f"⚠️ MANAGE_ROW asof failed | {sym} | {type(e).__name__}: {e}")

        # ---- 3) 없음
        _log_once(f"miss:{sym}", f"⚠️ MANAGE_ROW_MISS | {sym} no_row_for_authority={authority_time}")
        return None

    # -----------------------------------------------------
    # Titan9 계약: analyze 입력 슬라이싱 고정
    # -----------------------------------------------------
    

    # =========================================================
    # [PATCH] Titan9 Slice Contract (DEF-level seal)
    # 적용 위치: core/live_engine.py / class LiveEngine 내부
    # - (1) def 2개 추가: _t9_min_len, _slice_for_t9
    # - (2) _compute_candidates_15m()의 past_data 구성부 교체
    # =========================================================

    # -----------------------------------------------------
    # [ADD] Titan9 계약: 최소 길이 (def-level)
    # -----------------------------------------------------
    def _t9_min_len(self) -> int:
        """
        ✅ Titan v9 엔진 계약(고정):
        analyze 입력 df 길이 >= max(200, flip_window_bars + 2)
        """
        try:
            p = getattr(self.titan, "params", None)
            p = p if isinstance(p, dict) else {}
            N = int(p.get("flip_window_bars", 5) or 5)
        except Exception:
            N = 5
        return int(max(200, N + 2))
    # -----------------------------------------------------
    # [ADD] Titan9 계약: asof(Time Authority) 결합 강제 슬라이스 (def-level seal)
    # -----------------------------------------------------


    def _slice_for_t9(self, df: pd.DataFrame, asof) -> pd.DataFrame:
        """
        ✅ Titan v9 엔진 계약(봉인형):
        df_for_sig = df.loc[:asof].tail(_t9_min_len())

        - asof(=authority_time)가 df.index에 "반드시" 존재해야 함
        - 위반 가능성이 있으면 None 반환 -> 상위에서 스킵
        - 폴백(df.tail 등) 금지

        ✅ Backtest 정합 추가:
        - backtest는 공통 timeline에서 timeline[200:]부터 시뮬레이션을 시작한다.
        (즉, asof는 최소 200봉 이후에만 신호 평가)
        - LIVE도 동일하게 'asof 위치가 want 이상'일 때만 평가한다.
        (want=_t9_min_len() 이고 기본 200) → 1~2봉 차이로 prev 기반 게이트가 깨지는 것을 방지.
        """
        if df is None or (not isinstance(df, pd.DataFrame)) or df.empty or asof is None:
            return None

        # ✅ Time Authority 봉인: asof 미존재 시 컷
        if asof not in df.index:
            return None

        want = int(self._t9_min_len())

        # ✅ Backtest 정합: 최소 warmup 위치 보장 (loc >= want)
        try:
            loc = df.index.get_loc(asof)
            # get_loc이 slice/mask를 반환하는 케이스 방어
            if isinstance(loc, slice):
                loc = int(loc.stop) - 1
            elif not isinstance(loc, int):
                loc = int(loc[-1]) if hasattr(loc, "__len__") else int(loc)
            if int(loc) < int(want):
                return None
        except Exception:
            return None

        try:
            return df.loc[:asof].tail(want).copy()
        except Exception:
            return None




    # -----------------------------------------------------
    # [ADD] Retest 0 원인 진단 (def-level)
    # - placement: right after def _slice_for_t9(...)
    # -----------------------------------------------------
    def _diagnose_retest(self, sym: str, past_data: pd.DataFrame) -> dict:
        """
        ✅ retest=0/1 원인 진단 (로그/디버그 전용)
        - Titan 8.3.0(CHOCH) 기준으로 lvl_up/lvl_dn를 추출
        - 레거시 호환: choch_level_* 없으면 mss_level_*로 fallback
        - heuristic distance는 retest 정의와 더 일치하게 계산
        LONG: |low - lvl_up|, SHORT: |high - lvl_dn|
        """
        out = {
            "sym": sym,
            "obs": {},
            "calc": {},
            "heuristic": {},
            "params": {},
            "inputs": {},
            "why0": {},
            "mismatch": {"LONG": 0, "SHORT": 0},
            "note": "calc mirrors obs (strategy-aligned, backtest-consistent)",
        }

        try:
            if past_data is None or len(past_data) < 2:
                return out

            curr = past_data.iloc[-1]

            def _sf(x, default=None):
                try:
                    v = float(x)
                    if not math.isfinite(v):
                        return default
                    return v
                except Exception:
                    return default

            def _si(x, default=0):
                try:
                    return int(float(x))
                except Exception:
                    return default

            # ---- 전략 관측값 (단일 진실원)
            obs_rl = _si(curr.get("retest_long", 0), 0)
            obs_rs = _si(curr.get("retest_short", 0), 0)

            out["obs"] = {"retest_long": obs_rl, "retest_short": obs_rs}
            out["calc"] = {"retest_long": obs_rl, "retest_short": obs_rs}  # ✅ 정합: calc=obs

            # ---- 참고 입력(로그용)
            atr = _sf(curr.get("atr", None), None)
            close = _sf(curr.get("close", None), None)
            high = _sf(curr.get("high", None), None)
            low = _sf(curr.get("low", None), None)

            # ✅ Titan 8.3.0: CHOCH 레벨 우선, 없으면 MSS 레벨 fallback
            lvl_up = curr.get("choch_level_up", None)
            lvl_dn = curr.get("choch_level_down", None)
            if lvl_up is None:
                lvl_up = curr.get("mss_level_up", None)
            if lvl_dn is None:
                lvl_dn = curr.get("mss_level_down", None)

            out["inputs"] = {
                "atr": atr,
                "close": close,
                "high": high,
                "low": low,
                "lvl_up": _sf(lvl_up, None),
                "lvl_dn": _sf(lvl_dn, None),
            }

            # ---- (참고용) heuristic 계산: mismatch 판단에 쓰지 않음
            try:
                p = getattr(self.titan, "params", {}) or {}
                tol_atr = float(p.get("retest_tolerance_atr", p.get("retest_tolerance", 0.4)) or 0.4)
                tol_px = None
                dist_long = None
                dist_short = None

                if atr is not None and atr > 0 and close is not None:
                    tol_px = float(atr * tol_atr)

                    # ✅ retest 정의에 더 맞는 거리
                    # LONG: low가 lvl_up 근처로 내려왔는지
                    if lvl_up is not None and low is not None:
                        dist_long = abs(float(low) - float(lvl_up))
                    # SHORT: high가 lvl_dn 근처로 올라왔는지
                    if lvl_dn is not None and high is not None:
                        dist_short = abs(float(high) - float(lvl_dn))

                out["params"] = {
                    "tol_atr": tol_atr,
                    "tol_px": tol_px,
                    "dist_long": dist_long,
                    "dist_short": dist_short,
                }

                h_rl = 1 if (tol_px is not None and dist_long is not None and dist_long <= tol_px) else 0
                h_rs = 1 if (tol_px is not None and dist_short is not None and dist_short <= tol_px) else 0
                out["heuristic"] = {"retest_long": h_rl, "retest_short": h_rs}

                why0 = {"LONG": [], "SHORT": []}
                if h_rl == 0 and (dist_long is not None and tol_px is not None):
                    why0["LONG"].append(f"dist_gt_tol(dist={dist_long:.6g} tol={tol_px:.6g})")
                if h_rs == 0 and (dist_short is not None and tol_px is not None):
                    why0["SHORT"].append(f"dist_gt_tol(dist={dist_short:.6g} tol={tol_px:.6g})")
                out["why0"] = why0

            except Exception:
                pass

            return out

        except Exception:
            return out

    # -----------------------------------------------------
    # [ADD] analyze() 최종 반환값 디버그 (def-level)
    # - placement: right after def _diagnose_retest(...)
    # -----------------------------------------------------
    def _debug_analyze_result(self, sym: str, past_data: pd.DataFrame,
                              signal, sl, tp,
                              max_samples: int = 2):
        """
        ✅ analyze() 최종 반환값을 구조적으로 출력
        - 1~2개 심볼만 샘플로 찍음 (과도한 로그 방지)
        - 왜 signal=None 인지 gate 관점 + retest 관점 함께 출력
        """

        try:
            dbg_cnt = getattr(self, "_analyze_dbg_count", 0)
        except Exception:
            dbg_cnt = 0

        if dbg_cnt >= max_samples:
            return

        try:
            curr = past_data.iloc[-1]
            prev = past_data.iloc[-2] if len(past_data) >= 2 else None
        except Exception:
            return

        def _f(x):
            try:
                v = float(x)
                return v if math.isfinite(v) else None
            except Exception:
                return None

        def _i(x):
            try:
                return int(float(x))
            except Exception:
                return None

        snapshot = {
            "close": _f(curr.get("close")),
            "high": _f(curr.get("high")),
            "low": _f(curr.get("low")),
            "atr": _f(curr.get("atr")),
            "adx": _f(curr.get("adx")),
            "st_dir": _i(curr.get("st_dir")),
            "st_val": _f(curr.get("st_val")),
            "lvl_up": _f(curr.get("mss_level_up")),
            "lvl_dn": _f(curr.get("mss_level_down")),
            "retest_long": _i(curr.get("retest_long")),
            "retest_short": _i(curr.get("retest_short")),
        }

        logger.info(
            f"🧠 ANALYZE_RESULT | {sym} "
            f"signal={signal} sl={sl} tp={tp} "
            f"snapshot={snapshot}"
        )

        try:
            self._analyze_dbg_count = dbg_cnt + 1
        except Exception:
            pass


    # =========================================================
    # [MOD] _compute_candidates_15m() 내부 교체 패치
    # - "idx/get_loc + iloc[:idx+1] + _slice_for_strategy" 제거
    # - "_slice_for_t9(df, current_time_15m)"로 고정
    # =========================================================
    def _compute_candidates_15m(self, current_time_15m, symbols=None):
        """
        ✅ candidates 계산도 '동일 캔들 active_symbols' 기준으로만 수행
        + debug_entry_trace 시, 지표/게이트 분해 로그를 남겨 '왜 no_signal인지' 즉시 판별
        + ✅ retest=0 원인(=retest 계산식 관점 실패 요인)까지 분해 로그 추가

        ✅ 심볼 정합성 강화:
        - symbols 입력을 state 표준키(XXX/USDT:USDT)로 강제 정규화
        - data_map 키와 안 맞아 후보가 조용히 탈락하는 케이스를 차단

        ✅ Titan 8.3.0 정합:
        - MSS 키(mss_*) 제거 → CHOCH 키(choch_*) 사용
        - Volume gate 제거(전략은 score-only)
        - ATR Volatility Regime Gate(atr_ratio/atr_up) 반영
        - BOS(=bos_ok_long/short) + Daily + ADX(optional) + ST filter 로직을 전략과 동일하게 trace
        """

        def _canon(sym: str) -> str:
            try:
                s = str(sym or "").strip()
            except Exception:
                return str(sym)
            if not s:
                return s
            if ":USDT" in s:
                return s
            if "/" in s:
                if s.endswith("/USDT"):
                    return s + ":USDT"
                return s
            if s.endswith("USDT") and len(s) > 4:
                base = s[:-4]
                return f"{base}/USDT:USDT"
            return s

        candidates = []
        if not self.data_map:
            return candidates

        syms_in = symbols if symbols is not None else self.symbols
        if not syms_in:
            return candidates

        # ✅ 입력 심볼 표준화 + 중복 제거 (순서 유지)
        syms = []
        seen = set()
        for s in syms_in:
            cs = _canon(s)
            if not cs or cs in seen:
                continue
            seen.add(cs)
            syms.append(cs)

        debug_entry = bool(self.cfg.get("system_settings", {}).get("debug_entry_trace", False))
        logger.info(f"🔍 ENTRY_SCAN_START | t={current_time_15m} | universe={len(syms)}")

        miss_time = 0
        too_short = 0
        analyze_fail = 0
        no_signal = 0
        had_signal = 0
        miss_df = 0

        def _f(x, default=None):
            try:
                if x is None:
                    return default
                v = float(x)
                if not math.isfinite(v):
                    return default
                return v
            except Exception:
                return default

        def _i(x, default=None):
            try:
                if x is None:
                    return default
                return int(float(x))
            except Exception:
                return default

        def _snap_safe(curr, prev=None):
            """
            ✅ Titan 8.3.0 컬럼 스냅샷(CHOCH/BOS/ATR Regime 포함)
            """
            try:
                return {
                    "close": _f(curr.get("close", None), None),
                    "open": _f(curr.get("open", None), None),
                    "high": _f(curr.get("high", None), None),
                    "low": _f(curr.get("low", None), None),
                    "volume": _f(curr.get("volume", None), None),

                    "adx": _f(curr.get("adx", None), None),
                    "rsi": _f(curr.get("rsi", None), None),
                    "atr": _f(curr.get("atr", None), None),

                    # volume score-only (전략은 gate 아님)
                    "vol_ma_prev": _f(prev.get("vol_ma", None), None) if isinstance(prev, pd.Series) else None,
                    "vol_ratio_prev": _f(curr.get("vol_ratio_prev", None), None),
                    "vol_score_prev": _f(curr.get("vol_score_prev", None), None),

                    "ema_intra": _f(curr.get("ema_intra", None), None),
                    "ema_daily": _f(curr.get("ema_daily", None), None),
                    "ema_daily_ok": _i(curr.get("ema_daily_ok", None), None),

                    "st_dir": _i(curr.get("st_dir", None), None),
                    "st_val": _f(curr.get("st_val", None), None),

                    # sweep context
                    "recent_sweep_high": _i(curr.get("recent_sweep_high", None), None),
                    "recent_sweep_low": _i(curr.get("recent_sweep_low", None), None),

                    # ✅ CHOCH (NOT MSS)
                    "choch_up_trigger": _i(curr.get("choch_up_trigger", None), None),
                    "choch_down_trigger": _i(curr.get("choch_down_trigger", None), None),
                    "choch_active_up": _i(curr.get("choch_active_up", None), None),
                    "choch_active_down": _i(curr.get("choch_active_down", None), None),
                    "choch_level_up": _f(curr.get("choch_level_up", None), None),
                    "choch_level_down": _f(curr.get("choch_level_down", None), None),

                    # ✅ BOS / Trend state
                    "trend_state": _i(curr.get("trend_state", None), None),
                    "bos_ok_long": _i(curr.get("bos_ok_long", None), None),
                    "bos_ok_short": _i(curr.get("bos_ok_short", None), None),

                    # ✅ Retest flags
                    "retest_long": _i(curr.get("retest_long", None), None),
                    "retest_short": _i(curr.get("retest_short", None), None),

                    # ✅ ATR Regime Gate inputs
                    "atr_ma": _f(curr.get("atr_ma", None), None),
                    "atr_ratio": _f(curr.get("atr_ratio", None), None),
                    "atr_up": _i(curr.get("atr_up", None), None),
                }
            except Exception:
                return {}

        def _gate_trace(curr, prev):
            """
            ✅ TitanStrategy.analyze()와 동일한 논리로 gate를 재계산해서 로그에 찍는다.
            (디버그용이며, 실제 후보 생성/신호는 titan.analyze() 리턴을 그대로 사용)
            """
            try:
                p = getattr(self.titan, "params", {}) or {}

                # --- strategy switches ---
                use_daily = bool(p.get("use_daily_filter", True))
                use_st = bool(p.get("use_st_dir_filter", True))
                use_structure = bool(p.get("use_structure_confirm", True))
                use_vol_regime = bool(p.get("use_vol_regime_gate", True))
                atr_slope_gate = bool(p.get("atr_slope_gate", True))

                adx_th = float(p.get("adx_threshold", 0) or 0)
                adx_filter_on = adx_th > 0

                close = float(curr.get("close", 0) or 0)
                open_ = float(curr.get("open", close) or close)

                # --- Daily filter ---
                daily_ok = int(curr.get("ema_daily_ok", 0) or 0) == 1
                ema_daily = float(curr.get("ema_daily", 0) or 0)
                is_daily_up = (close > ema_daily) if (use_daily and daily_ok) else True
                is_daily_dn = (close < ema_daily) if (use_daily and daily_ok) else True

                # --- ADX (optional gate) ---
                adx = float(curr.get("adx", 0) or 0)
                is_trend_alive = (adx > adx_th) if adx_filter_on else True

                # --- Volatility regime gate (ATR 기반) ---
                vol_gate_ok = True
                atr_ratio = float(curr.get("atr_ratio", 0.0) or 0.0)
                need = float(p.get("atr_regime_factor", 1.0) or 1.0)
                if use_vol_regime:
                    vol_gate_ok = (atr_ratio > need)
                    if atr_slope_gate:
                        vol_gate_ok = vol_gate_ok and (int(curr.get("atr_up", 0) or 0) == 1)

                # --- Level / Retest ---
                retest_long = int(curr.get("retest_long", 0) or 0) == 1
                retest_short = int(curr.get("retest_short", 0) or 0) == 1

                lvl_up = curr.get("choch_level_up", None)
                lvl_dn = curr.get("choch_level_down", None)
                level_ok = (
                    (lvl_up is not None and pd.notna(lvl_up) and float(lvl_up) > 0)
                    or (lvl_dn is not None and pd.notna(lvl_dn) and float(lvl_dn) > 0)
                )

                # --- BOS confirm (strategy와 동일) ---
                bos_ok_long = (int(curr.get("bos_ok_long", 0) or 0) == 1) if use_structure else True
                bos_ok_short = (int(curr.get("bos_ok_short", 0) or 0) == 1) if use_structure else True

                # --- Pre signal (strategy 동일) ---
                pre_sig = None
                if level_ok and vol_gate_ok:
                    if retest_long and is_daily_up and bos_ok_long:
                        if (not adx_filter_on) or is_trend_alive:
                            pre_sig = "LONG"
                    elif retest_short and is_daily_dn and bos_ok_short:
                        if (not adx_filter_on) or is_trend_alive:
                            pre_sig = "SHORT"

                # --- ST filter (strategy 동일) ---
                post_sig = pre_sig
                if pre_sig is not None and use_st:
                    st_dir = int(curr.get("st_dir", 0) or 0)
                    st_val = curr.get("st_val", None)
                    st_val_f = None
                    try:
                        st_val_f = float(st_val) if (st_val is not None and math.isfinite(float(st_val))) else None
                    except Exception:
                        st_val_f = None

                    if pre_sig == "LONG":
                        if (st_dir <= 0) or (st_val_f is not None and st_val_f >= close):
                            post_sig = None
                    else:  # SHORT
                        if (st_dir >= 0) or (st_val_f is not None and st_val_f <= close):
                            post_sig = None

                return {
                    "level_ok": int(bool(level_ok)),
                    "retestL": int(bool(retest_long)),
                    "retestS": int(bool(retest_short)),

                    "daily_ok": int(bool(daily_ok)) if use_daily else -1,
                    "daily_up": int(bool(is_daily_up)),
                    "daily_dn": int(bool(is_daily_dn)),

                    "adx_on": int(bool(adx_filter_on)),
                    "adx_ok": int(bool(is_trend_alive)),

                    # ✅ volatility regime gate 상태를 명시
                    "vol_reg_on": int(bool(use_vol_regime)),
                    "atr_ratio": float(atr_ratio),
                    "atr_need": float(need),
                    "atr_up": int(curr.get("atr_up", 0) or 0),
                    "vol_gate_ok": int(bool(vol_gate_ok)),

                    # ✅ BOS 상태를 명시
                    "bos_on": int(bool(use_structure)),
                    "bosL": int(bool(bos_ok_long)),
                    "bosS": int(bool(bos_ok_short)),

                    "pre_sig": pre_sig,
                    "st_filter_on": int(bool(use_st)),
                    "post_sig": post_sig,
                    "close>open": int(bool(close > open_)),
                }
            except Exception:
                return {}

        min_len = int(self._t9_min_len())

        for sym in syms:
            # ✅ cooldown gate (Backtest 동일) - 키 표준화 후 체크
            if self._is_in_cooldown(sym, current_time_15m):
                if debug_entry:
                    try:
                        until = self.cooldowns.get(sym)
                        logger.info(f"🧊 COOLDOWN_SKIP | {sym} now={current_time_15m} until={until}")
                    except Exception:
                        pass
                continue

            df = self.data_map.get(sym)
            if df is None or df.empty:
                miss_df += 1
                if debug_entry:
                    logger.info(f"⚠️ DATA_MISS | {sym} not in data_map (symbol mismatch)")
                continue

            past_data = self._slice_for_t9(df, current_time_15m)
            if past_data is None:
                miss_time += 1
                continue

            if len(past_data) < min_len:
                too_short += 1
                continue

            if debug_entry:
                try:
                    curr_dbg = past_data.iloc[-1]
                    prev_dbg = past_data.iloc[-2]
                    snap0 = _snap_safe(curr_dbg, prev_dbg)
                    gate0 = _gate_trace(curr_dbg, prev_dbg)

                    diag = self._diagnose_retest(sym, past_data)
                    rl0 = (int(diag.get("obs", {}).get("retest_long", 0)) == 0)
                    rs0 = (int(diag.get("obs", {}).get("retest_short", 0)) == 0)

                    if rl0 or rs0:
                        logger.info(
                            f"🧪 PRE_ANALYZE | {sym} t={current_time_15m} rows={len(past_data)} "
                            f"snap={snap0} gate={gate0} "
                            f"retest_diag={{"
                            f"'obs':{diag.get('obs')},"
                            f"'calc':{diag.get('calc')},"
                            f"'params':{diag.get('params')},"
                            f"'inputs':{diag.get('inputs')},"
                            f"'why0':{diag.get('why0')},"
                            f"'mismatch':{diag.get('mismatch')}"
                            f"}}"
                        )
                    else:
                        logger.info(f"🧪 PRE_ANALYZE | {sym} t={current_time_15m} rows={len(past_data)} snap={snap0} gate={gate0}")
                except Exception:
                    pass

            try:
                signal, sl, tp = self.titan.analyze(sym, past_data)
                if debug_entry:
                    try:
                        self._debug_analyze_result(sym, past_data, signal, sl, tp)
                    except Exception:
                        pass
            except Exception as e:
                analyze_fail += 1
                logger.warning(f"⚠️ ANALYZE_FAIL | {sym} err={e}")
                continue

            if not signal:
                no_signal += 1
                continue

            had_signal += 1

            # ✅ loc는 KeyError 가능 -> 심볼/시간 불일치시 candidates에 넣지 않음
            try:
                curr_row = df.loc[current_time_15m]
            except Exception:
                miss_time += 1
                continue

            try:
                score = curr_row.get("adx", 0)
                score = float(score) if score is not None else 0.0
            except Exception:
                score = 0.0

            candidates.append({
                "score": float(score),
                "sym": sym,               # ✅ 표준키만 들어감
                "signal": signal,         # ✅ Titan analyze 리턴 그대로 사용
                "sl": sl,
                "tp": tp,
                "row": curr_row,
            })

            try:
                logger.info(f"🟢 ENTRY_CANDIDATE | {sym} signal={signal} adx={score:.2f} sl={sl} tp={tp}")
            except Exception:
                logger.info(f"🟢 ENTRY_CANDIDATE | {sym} signal={signal} sl={sl} tp={tp}")

        candidates.sort(key=lambda x: x["score"], reverse=True)

        if debug_entry:
            logger.info(
                f"🔎 ENTRY_SCAN_STATS | t={current_time_15m} "
                f"universe={len(syms)} miss_df={miss_df} miss_time={miss_time} too_short={too_short} "
                f"analyze_fail={analyze_fail} no_signal={no_signal} had_signal={had_signal} cand={len(candidates)}"
            )

        return candidates


    # -----------------------------------------------------
    # LIVE entry/exit helpers
    # -----------------------------------------------------
    def _max_positions_live(self):
        mp = getattr(self.executor, "MAX_POSITIONS", None)
        if mp is None:
            mp = self.cfg.get("risk_settings", {}).get("max_open_positions", 5)
        try:
            mp = int(mp)
            if mp <= 0:
                mp = 1
        except Exception:
            mp = 5
        return mp

    def _get_next_open_anchor(self, sym: str, candle_t):
        """
        ✅ LIVE 정합 앵커:
        - Backtest의 next-open 앵커는 LIVE 데이터에 next_t가 없어서 상시 None이 된다.
        - LIVE에서는 확정봉(candle_t)의 'close'를 앵커로 사용한다.
        => "봉 종가에 limit(혹은 market)로 진입" 모델과 정합.
        """
        try:
            df = self.data_map.get(sym)
            if df is None or df.empty:
                return None
            t = pd.to_datetime(candle_t)
            if t not in df.index:
                return None
            v = df.loc[t].get("close", None)
            v = float(v) if v is not None else None
            if v is None or (not math.isfinite(v)) or v <= 0:
                return None
            return float(v)
        except Exception:
            return None

    def _entry_alpha_atr(self) -> float:
        """
        ✅ entry limit price offset (ATR multiple)
        - system_settings.entry_alpha_atr (default 0.0)
        - alpha >= 0
        """
        try:
            ss = (self.cfg or {}).get("system_settings", {}) or {}
            a = float(ss.get("entry_alpha_atr", 0.0) or 0.0)
            if (not math.isfinite(a)) or a < 0:
                a = 0.0
            return float(a)
        except Exception:
            return 0.0

    def _place_limit_entry_order(self, sym: str, side: str, amount: float, limit_price: float):
        """
        ✅ Limit entry order submit (executor authority)
        - 반환: order_id(str) or None
        """
        try:
            # 1) executor에 구현된 안정 호출 우선
            if hasattr(self.executor, "create_limit_order"):
                res = self.executor.create_limit_order(sym, side, float(amount), float(limit_price))
                if isinstance(res, dict):
                    oid = res.get("order_id")
                    if oid:
                        return str(oid)
                return None

            # 2) fallback (기존 호환): exchange 직접 호출 (권장하지 않음)
            ex = getattr(self.executor, "exchange", None)
            if ex is None:
                return None

            order = ex.create_order(sym, "limit", str(side).lower(), float(amount), float(limit_price), params={})
            if isinstance(order, dict) and order.get("id"):
                return str(order["id"])
        except Exception as e:
            logger.error(f"limit entry place failed: {sym} err={e}")
        return None


    # [MOD] def _sweep_pending_entry_orders(self, current_bucket: int)
    def _sweep_pending_entry_orders(self, current_bucket: int):
        """
        ✅ 다음 15m 봉까지 미체결이면 cancel + ENTRY_SKIP 처리
        ✅ 그 사이에 체결되면:
        - positions에 생긴 것을 감지 (full/base 혼재 방어)
        - pending 메타(sl/entry_time/entry_atr)를 주입
        - ENTRY 로그 남기고 pending 제거
        - ✅ entry 직후 거래소 Position SL 미러링
        """
        peo = getattr(self, "pending_entry_orders", None)
        if not isinstance(peo, dict) or not peo:
            return 0

        removed = 0
        changed = False

        for key_sym in list(peo.keys()):
            od = peo.get(key_sym) or {}
            try:
                expire_bucket = int(od.get("expire_bucket"))
            except Exception:
                expire_bucket = None

            order_sym = str(od.get("order_sym") or key_sym)
            order_base = self._base_sym(order_sym)
            key_base = self._base_sym(key_sym)

            # 1) fill 감지
            pos_map = (self.executor.positions or {})
            filled_sym = None
            if order_sym in pos_map:
                filled_sym = order_sym
            elif order_base and (order_base in pos_map):
                filled_sym = order_base
            elif key_sym in pos_map:
                filled_sym = key_sym
            elif key_base and (key_base in pos_map):
                filled_sym = key_base

            if filled_sym is not None:
                pos = pos_map.get(filled_sym) or {}

                try:
                    if pos.get("sl") is None and od.get("sl") is not None:
                        pos["sl"] = float(od["sl"])
                except Exception:
                    pass

                try:
                    if pos.get("entry_time") in (None, "", "None"):
                        pos["entry_time"] = str(od.get("candle_t") or pd.Timestamp.utcnow())
                except Exception:
                    pass

                try:
                    if pos.get("entry_atr") is None and od.get("entry_atr") is not None:
                        pos["entry_atr"] = float(od["entry_atr"])
                except Exception:
                    pass

                pos.setdefault("next_sl", None)
                pos.setdefault("trail_sl", None)

                # ✅ 포지션 체결 직후 현재 SL을 거래소 Position SL로 미러링
                try:
                    self._sync_exchange_position_sl(
                        sym=filled_sym,
                        pos=pos,
                        active_sl=pos.get("sl"),
                        reason="ENTRY_FILL",
                    )
                except Exception:
                    pass

                try:
                    self._append_history({
                        "dt": str(od.get("candle_t") or pd.Timestamp.utcnow()),
                        "event": "ENTRY",
                        "mode": "LIVE",
                        "symbol": filled_sym,
                        "side": str(pos.get("side", od.get("signal_side", ""))).upper(),
                        "price": float(pos.get("entry_price")) if pos.get("entry_price") is not None else None,
                        "amount": float(pos.get("amount", 0) or 0),
                        "fee": None,
                        "margin": float(pos.get("margin", 0) or 0),
                        "pnl": 0.0,
                        "sl": float(pos.get("sl")) if pos.get("sl") is not None else None,
                        "reason": (f"LIMIT_FILLED | anchor={od.get('anchor_open')} | limit={od.get('limit_price')} | alpha={od.get('alpha')} | order_id={od.get('order_id')}")[:800],
                        "pos_count": int(len(self.executor.positions or {})),
                        "cash": float(self._last_cash or 0),
                        "equity": float(self._last_equity or 0),
                    })
                except Exception:
                    pass

                peo.pop(key_sym, None)
                removed += 1
                changed = True
                continue

            # 2) 만기 도달 -> cancel + ENTRY_SKIP
            if expire_bucket is not None and int(current_bucket) >= int(expire_bucket):
                oid = od.get("order_id")
                if oid:
                    try:
                        if hasattr(self.executor, "cancel_order_safe"):
                            self.executor.cancel_order_safe(str(oid), order_sym)
                        else:
                            ex = getattr(self.executor, "exchange", None)
                            if ex is not None:
                                ex.cancel_order(str(oid), order_sym, params={})
                    except Exception:
                        pass

                try:
                    self._append_history({
                        "dt": str(pd.Timestamp.utcnow()),
                        "event": "ENTRY_SKIP",
                        "mode": "LIVE",
                        "symbol": order_sym,
                        "side": str(od.get("signal_side", "")).upper(),
                        "price": float(od.get("limit_price")) if od.get("limit_price") is not None else float(od.get("anchor_open")) if od.get("anchor_open") is not None else None,
                        "amount": float(od.get("amount", 0) or 0),
                        "sl": float(od.get("sl")) if od.get("sl") is not None else None,
                        "reason": (f"UNFILLED_CANCELED_NEXT_CANDLE | anchor={od.get('anchor_open')} | limit={od.get('limit_price')} | alpha={od.get('alpha')} | order_id={oid}")[:800],
                        "pos_count": int(len(self.executor.positions or {})),
                        "cash": float(self._last_cash or 0),
                        "equity": float(self._last_equity or 0),
                    })
                except Exception:
                    pass

                peo.pop(key_sym, None)
                removed += 1
                changed = True

        if changed:
            try:
                self.pending_entry_orders = peo
                self._save_state()
            except Exception:
                pass

        return removed

    def _calc_entry_amount_live(self, sym, entry_price, sl_price, signal_side, row=None) -> float:
        """
        ✅ LIVE sizing adapter:
        - RiskControl.calculate_entry_size 시그니처에 맞춰 인자를 채워서 호출한다.
        - 실패(인자/값/예외) 시 0.0 반환 → 상위에서 ENTRY_REJECT 처리
        """
        # equity(total) sizing 기준 유지
        try:
            sizing_equity = float(self._last_equity or 0.0)
        except Exception:
            sizing_equity = 0.0

        # atr (옵션) : row(Series)에서 읽되 실패하면 0
        atr = 0.0
        try:
            if row is not None:
                # pandas Series는 get 지원
                atr = float(row.get("atr", 0.0) or 0.0)
        except Exception:
            atr = 0.0

        # 입력 정규화
        try:
            ep = float(entry_price or 0.0)
        except Exception:
            ep = 0.0
        try:
            sp = float(sl_price or 0.0)
        except Exception:
            sp = 0.0

        side = str(signal_side or "").upper()

        # RiskControl이 요구하는 최소 조건을 여기서 선제 체크
        if ep <= 0 or sp <= 0 or sizing_equity <= 0:
            return 0.0
        if side not in ("LONG", "SHORT"):
            return 0.0

        # ✅ 여기서 시그니처에 맞춰 호출 (핵심 수정)
        try:
            amt = float(
                self.risk_ctrl.calculate_entry_size(
                    sym,          # symbol
                    ep,           # entry_price
                    sizing_equity,# equity
                    sp,           # sl_price
                    side,         # signal_side
                    atr,          # atr (optional)
                )
            )
            return amt if (amt > 0) else 0.0
        except Exception:
            return 0.0


    def _process_entry_live(self, cand):
        """
        ✅ LIVE ONLY Entry (Revised)
        - anchor = 확정봉 close (기존 _get_next_open_anchor 그대로 사용)
        - limit_price = anchor ± alpha*ATR
        - 항상 리밋 오픈오더 1개 제출
        - 다음 봉까지 미체결이면 sweep에서 cancel
        - 시장가 전환 없음
        """
        sym = cand.get("sym")
        row = cand.get("row")
        raw_sig = str(cand.get("signal", "") or "")
        sl = cand.get("sl", None)
        tp = cand.get("tp", None)
        rt_snapshot = cand.get("rt_snapshot", None) if isinstance(cand, dict) else None

        if not sym or row is None:
            return False

        sym_full = str(sym or "").strip()
        sym_base = self._base_sym(sym_full)
        candle_t = pd.to_datetime(row.name)

        try:
            logger.info(f"🚀 ENTRY_ATTEMPT | {sym_full} base={sym_base} t={candle_t} raw_sig={raw_sig} sl={sl} tp={tp}")
        except Exception:
            pass

        # cooldown gate
        if self._is_in_cooldown(sym_full, candle_t):
            try:
                until = self.cooldowns.get(sym_full) or self.cooldowns.get(sym_base)
                logger.info(f"🧊 ENTRY_REJECT | {sym_full} reason=cooldown now={candle_t} until={until}")
                self._append_history({
                    "dt": str(candle_t),
                    "event": "ENTRY_REJECT",
                    "mode": "LIVE",
                    "symbol": sym_full,
                    "side": "",
                    "price": None,
                    "amount": 0.0,
                    "sl": float(sl) if sl is not None else None,
                    "reason": f"cooldown (until={until})",
                    "pos_count": int(len(self.executor.positions or {})),
                    "cash": float(self._last_cash or 0),
                    "equity": float(self._last_equity or 0),
                })
            except Exception:
                pass
            return False

        # signal normalize
        sig = raw_sig.strip().upper()
        alias = {"BUY": "LONG", "LONG": "LONG", "BULL": "LONG",
                "SELL": "SHORT", "SHORT": "SHORT", "BEAR": "SHORT"}
        sideU = alias.get(sig, None)
        if sideU not in ("LONG", "SHORT"):
            logger.info(f"⚪ ENTRY_REJECT | {sym_full} signal_invalid raw={raw_sig}")
            return False

        # already in position / pending
        pos_map = (self.executor.positions or {})
        peo_map = (getattr(self, "pending_entry_orders", {}) or {})
        if (sym_full in pos_map) or (sym_base in pos_map):
            try:
                logger.info(f"⏭️ ENTRY_SKIP | {sym_full} reason=already_in_position")
            except Exception:
                pass
            return False
        if (sym_full in peo_map) or (sym_base in peo_map):
            try:
                od = peo_map.get(sym_full) or peo_map.get(sym_base) or {}
                logger.info(f"⏭️ ENTRY_SKIP | {sym_full} reason=pending_entry_exists order_id={od.get('order_id')}")
            except Exception:
                pass
            return False

        # freeze / max pos
        if getattr(self, "freeze_new_entries", False):
            try:
                logger.info(f"⏭️ ENTRY_SKIP | {sym_full} reason=freeze_new_entries")
            except Exception:
                pass
            return False

        try:
            max_pos = int(self._max_positions_live())
            cur_pos = int(len(self.executor.positions or {}))
            if cur_pos >= max_pos:
                try:
                    logger.info(f"⏭️ ENTRY_SKIP | {sym_full} reason=max_positions_reached cur={cur_pos} max={max_pos}")
                except Exception:
                    pass
                return False
        except Exception:
            pass

        # anchor (확정봉 close)
        anchor = self._get_next_open_anchor(sym_full, candle_t)
        if anchor is None and sym_base != sym_full:
            anchor = self._get_next_open_anchor(sym_base, candle_t)

        if anchor is None:
            logger.info(f"⚪ ENTRY_REJECT | {sym_full} {sideU} reason=no_anchor")
            self._append_history({
                "dt": str(candle_t),
                "event": "ENTRY_REJECT",
                "mode": "LIVE",
                "symbol": sym_full,
                "side": sideU,
                "price": None,
                "amount": 0.0,
                "sl": float(sl) if sl is not None else None,
                "reason": "no_anchor",
                "pos_count": int(len(self.executor.positions or {})),
                "cash": float(self._last_cash or 0),
                "equity": float(self._last_equity or 0),
            })
            return False

        # sizing
        try:
            sl_for_sizing = float(sl) if sl is not None else 0.0
        except Exception:
            sl_for_sizing = 0.0

        amount = float(self._calc_entry_amount_live(sym_full, float(anchor), sl_for_sizing, sideU, row=row))
        if amount <= 0 and sym_base != sym_full:
            amount = float(self._calc_entry_amount_live(sym_base, float(anchor), sl_for_sizing, sideU, row=row))

        if amount <= 0:
            logger.info(f"⚪ ENTRY_REJECT | {sym_full} {sideU} amount<=0 sl={sl_for_sizing}")
            self._append_history({
                "dt": str(candle_t),
                "event": "ENTRY_REJECT",
                "mode": "LIVE",
                "symbol": sym_full,
                "side": sideU,
                "price": float(anchor),
                "amount": float(amount),
                "sl": float(sl) if sl is not None else None,
                "reason": "amount<=0 (sizing)",
                "pos_count": int(len(self.executor.positions or {})),
                "cash": float(self._last_cash or 0),
                "equity": float(self._last_equity or 0),
            })
            return False

        # entry_atr
        entry_atr = None
        try:
            v = row.get("atr", None)
            entry_atr = float(v) if v is not None else None
            if entry_atr is not None and ((not math.isfinite(entry_atr)) or entry_atr <= 0):
                entry_atr = None
        except Exception:
            entry_atr = None

        # alpha + limit_price
        alpha = float(self._entry_alpha_atr())
        atr_for_offset = float(entry_atr or 0.0)
        if (not math.isfinite(atr_for_offset)) or atr_for_offset < 0:
            atr_for_offset = 0.0

        if sideU == "LONG":
            limit_price = float(anchor) - (alpha * atr_for_offset)
        else:
            limit_price = float(anchor) + (alpha * atr_for_offset)

        # 안전 보정
        if (not math.isfinite(limit_price)) or limit_price <= 0:
            limit_price = float(anchor)

        try:
            logger.info(
                f"🟨 ENTRY_SUBMIT | {sym_full} side={sideU} type=LIMIT "
                f"amount={float(amount)} anchor={float(anchor)} atr={atr_for_offset} alpha={alpha} limit={limit_price}"
            )
        except Exception:
            pass

        limit_side = "buy" if sideU == "LONG" else "sell"

        oid = self._place_limit_entry_order(sym_full, limit_side, float(amount), float(limit_price))
        used_sym = sym_full
        if (not oid) and (sym_base != sym_full):
            oid = self._place_limit_entry_order(sym_base, limit_side, float(amount), float(limit_price))
            used_sym = sym_base

        if not oid:
            try:
                logger.info(
                    f"🟥 ENTRY_REJECT | {sym_full} side={sideU} reason=limit_order_submit_failed "
                    f"amount={float(amount)} limit={float(limit_price)}"
                )
            except Exception:
                pass
            return False

        try:
            logger.info(f"✅ ENTRY_PENDING | {used_sym} side={sideU} order_id={oid} limit={float(limit_price)} amount={float(amount)}")

            # ✅ [ADD] 오픈오더(ENTRY_PENDING) 텔레그램 전송
            try:
                if getattr(self, "notifier", None):
                    self.notifier.send(
                        title="ENTRY_PENDING",
                        lines=[
                            f"t={candle_t}",
                            f"symbol={used_sym}",
                            f"side={sideU}",
                            f"order_id={oid}",
                            f"anchor={float(anchor)}",
                            f"atr={float(atr_for_offset)}",
                            f"alpha={float(alpha)}",
                            f"limit={float(limit_price)}",
                            f"amount={float(amount)}",
                            f"sl={float(sl) if sl is not None else None}",
                        ],
                    )
            except Exception as e:
                logger.error(f"ENTRY_PENDING telegram failed | {used_sym} | {e}")

        except Exception:
            pass

        b = self._current_bucket_15m()
        peo = getattr(self, "pending_entry_orders", {}) or {}
        peo[used_sym] = {
            "order_id": str(oid),
            "order_sym": used_sym,
            "signal_side": sideU,
            "amount": float(amount),
            "sl": float(sl) if sl is not None else None,
            "tp": float(tp) if tp is not None else None,

            # ✅ 호환: 기존 필드명 유지(anchor_open) = anchor(close)
            "anchor_open": float(anchor),

            # ✅ 신규: 리트레이스/오프셋 정보
            "alpha": float(alpha),
            "limit_price": float(limit_price),

            "candle_t": str(candle_t),
            "entry_atr": float(entry_atr) if entry_atr is not None else None,
            "created_bucket": int(b),
            "expire_bucket": int(b) + 1,
        }
        self.pending_entry_orders = peo
        self._save_state()

        self._append_history({
            "dt": str(candle_t),
            "event": "ENTRY_PENDING",
            "mode": "LIVE",
            "symbol": used_sym,
            "side": sideU,
            "price": float(limit_price),
            "amount": float(amount),
            "sl": float(sl) if sl is not None else None,
            "reason": f"LIMIT_PENDING | anchor={anchor} atr={atr_for_offset} alpha={alpha} limit={limit_price} order_id={oid}",
            "pos_count": int(len(self.executor.positions or {})),
            "cash": float(self._last_cash or 0),
            "equity": float(self._last_equity or 0),
        })
        return False


    def _process_existing_position_live(self, sym, curr_row):
        """
        ✅ LIVE에서도 Backtest와 동일한 PositionMonitor 입력을 제공:
        - market_data: close/high/low/atr/st_val/adx + df(hist_df)
        - sl_strategy: armor 포함
        """
        pos = self.executor.positions.get(sym)
        if not pos:
            return "NONE"

        # -------------------------------------------------
        # ✅ curr_row normalize
        # -------------------------------------------------
        if isinstance(curr_row, pd.Series):
            row_series = curr_row
            candle_t = pd.to_datetime(getattr(curr_row, "name", None) or pd.Timestamp.utcnow())
        elif isinstance(curr_row, dict):
            candle_t = pd.Timestamp.utcnow()
            row_series = pd.Series(curr_row, name=candle_t)
        else:
            return "HOLD"

        # ---- apply mode ----
        apply_mode = self._get_sl_apply_mode()

        # -------------------------------------------------
        # ✅ next_sl 승계는 "next 모드"에서만 (기존 로직 유지)
        # -------------------------------------------------
        candle_bucket = self._bucket_15m_from_candle(candle_t)

        if apply_mode == "next":
            try:
                if ("next_sl" in pos) and (pos.get("next_sl") is not None):
                    promote_now = False
                    nb = pos.get("next_sl_bucket", None)

                    if nb is None:
                        promote_now = True
                    else:
                        try:
                            nb_i = int(nb)
                            promote_now = int(candle_bucket) > nb_i
                        except Exception:
                            promote_now = True

                    if promote_now:
                        nxt = float(pos.get("next_sl"))
                        cur = pos.get("sl", None)
                        cur_f = float(cur) if cur is not None else None
                        if (cur_f is None) or (nxt != cur_f):
                            pos["sl"] = float(nxt)
                        pos["next_sl"] = None
                        pos["next_sl_bucket"] = None
                        self._save_state()
            except Exception:
                pass
        else:
            try:
                if pos.get("next_sl") is not None or pos.get("next_sl_bucket") is not None:
                    pos["next_sl"] = None
                    pos["next_sl_bucket"] = None
                    self._save_state()
            except Exception:
                pass

        sl_strategy = self._get_sl_strategy()
        sl_params = self._get_sl_params()

        # -------------------------------------------------
        # ✅ hist_df
        # -------------------------------------------------
        hist_df = None
        try:
            df_full = self.data_map.get(sym)
            if isinstance(df_full, pd.DataFrame) and (not df_full.empty):
                asof = getattr(row_series, "name", None)
                hist_df = df_full.loc[:asof].copy() if asof is not None else df_full.copy()

                try:
                    rename_map = {}
                    for c in hist_df.columns:
                        lc = str(c).lower()
                        if lc in ("open", "high", "low", "close", "volume"):
                            rename_map[c] = lc
                    if rename_map:
                        hist_df = hist_df.rename(columns=rename_map)
                except Exception:
                    pass

                try:
                    lb = int(sl_params.get("armor_lookback", 300))
                    if lb > 0 and len(hist_df) > lb:
                        hist_df = hist_df.tail(lb)
                except Exception:
                    pass
        except Exception:
            hist_df = None

        # -------------------------------------------------
        # ✅ market_data
        # -------------------------------------------------
        def _safe_float(x, default=0.0):
            try:
                v = float(x)
                if not math.isfinite(v):
                    return float(default)
                return v
            except Exception:
                return float(default)

        close = _safe_float(row_series.get("close", 0.0), 0.0)
        high = _safe_float(row_series.get("high", close), close)
        low = _safe_float(row_series.get("low", close), close)

        atr = row_series.get("atr", None)
        atr = _safe_float(atr, close * 0.01)
        if atr <= 0:
            atr = close * 0.01

        st_val = row_series.get("st_val", None)
        try:
            st_val = float(st_val) if st_val is not None else None
            if (st_val is not None) and (not math.isfinite(st_val)):
                st_val = None
        except Exception:
            st_val = None

        adx = _safe_float(row_series.get("adx", 0.0), 0.0)

        market_data = {
            "close": close,
            "high": high,
            "low": low,
            "atr": atr,
            "st_val": st_val,
            "adx": adx,
            "df": hist_df,
            "candle_time": str(candle_t) if candle_t is not None else None,
        }

        # ✅ 재시작/누락 복구용: 현재 active SL이 있으면 거래소 Position SL 재무장
        try:
            self._sync_exchange_position_sl(
                sym=sym,
                pos=pos,
                active_sl=pos.get("sl"),
                reason="MANAGE_LOOP",
            )
        except Exception:
            pass

        # -------------------------------------------------
        # ✅ PositionMonitor
        # -------------------------------------------------
        prev_breached = bool(pos.get("sl_breached", False))
        action, exec_price, reason, new_sl = self.monitor.check_conditions(
            sym,
            pos,
            market_data,
            sl_apply_mode=apply_mode,
            sl_strategy=sl_strategy,
            sl_params=sl_params,
        )
        now_breached = bool(pos.get("sl_breached", False))

        latched_now = (apply_mode == "next") and (not prev_breached) and now_breached
        if latched_now:
            try:
                self._append_history({
                    "dt": market_data.get("candle_time"),
                    "event": "SL_BREACH_LATCH",
                    "symbol": sym,
                    "side": pos.get("side"),
                    "reason": (
                        f"next_mode sl_breached=1 "
                        f"breached_sl={pos.get('sl_breached_sl')} "
                        f"cur_sl={pos.get('sl')}"
                    )[:800],
                })
            except Exception:
                pass

            try:
                self._save_state()
            except Exception:
                pass

        # -------------------------------------------------
        # ✅ UPDATE_SL (기존 로직 유지)
        # -------------------------------------------------
        if action == "UPDATE_SL":
            applied_kind = "NONE"
            old_sl = pos.get("sl", None)
            old_next = pos.get("next_sl", None)

            try:
                if new_sl is not None:
                    new_sl_f = float(new_sl)
                    if (not math.isfinite(new_sl_f)) or (new_sl_f <= 0):
                        return "HOLD"

                    cur = pos.get("sl", None)
                    cur_f = float(cur) if cur is not None else None
                    nxt = pos.get("next_sl", None)
                    nxt_f = float(nxt) if nxt is not None else None

                    if apply_mode == "next":
                        if (cur_f is None or new_sl_f != cur_f) and (nxt_f is None or new_sl_f != nxt_f):
                            pos["next_sl"] = new_sl_f
                            try:
                                pos["next_sl_bucket"] = int(candle_bucket)
                            except Exception:
                                pos["next_sl_bucket"] = None
                            applied_kind = "NEXT"
                            self._save_state()
                    else:
                        if (cur_f is None) or (new_sl_f != cur_f):
                            pos["sl"] = new_sl_f
                            pos["next_sl"] = None
                            pos["next_sl_bucket"] = None
                            applied_kind = "SAME"
                            self._save_state()

                            try:
                                self._sync_exchange_position_sl(
                                    sym=sym,
                                    pos=pos,
                                    active_sl=pos.get("sl"),
                                    reason="UPDATE_SL",
                                )
                            except Exception:
                                pass
            except Exception:
                pass

            try:
                self._append_history({
                    "dt": market_data.get("candle_time"),
                    "event": "UPDATE_SL",
                    "mode": "LIVE",
                    "symbol": sym,
                    "side": pos.get("side"),
                    "price": float(market_data.get("close") or 0),
                    "amount": float(pos.get("amount") or 0),
                    "sl": float(pos.get("sl")) if pos.get("sl") is not None else None,
                    "reason": (reason or "TRAILING")[:800],
                    "pos_count": int(len(self.executor.positions or {})),
                    "cash": float(self._last_cash or 0),
                    "equity": float(self._last_equity or 0),
                })
            except Exception:
                pass

            try:
                if getattr(self, "notifier", None):
                    self.notifier.send(
                        title="UPDATE_SL",
                        lines=[
                            f"t={market_data.get('candle_time')}",
                            f"symbol={sym}",
                            f"side={pos.get('side')}",
                            f"new_sl={pos.get('sl')}",
                            f"apply_mode={apply_mode}",
                        ],
                    )
            except Exception as e:
                logger.error(f"UPDATE_SL telegram failed | {sym} | {e}")

            return "UPDATE_SL"

        # -------------------------------------------------
        # ✅ EXIT (여기만 cooldown 로직 추가)
        # -------------------------------------------------
        if action == "EXIT":
            # cooldown 계산에 쓸 exit_px를 먼저 확정 (backtest 동일 모델용)
            try:
                exit_px = float(exec_price) if exec_price is not None else float(market_data.get("close") or 0.0)
            except Exception:
                exit_px = float(market_data.get("close") or 0.0)

            try:
                ok = self.executor.close_position(sym, price=exec_price, reason=reason)
            except Exception:
                ok = False

            try:
                self._append_history({
                    "dt": market_data.get("candle_time"),
                    "event": "EXIT",
                    "mode": "LIVE",
                    "symbol": sym,
                    "side": pos.get("side"),
                    "price": float(exit_px) if exit_px is not None else float(market_data.get("close") or 0),
                    "amount": float(pos.get("amount") or 0),
                    "sl": float(pos.get("sl")) if pos.get("sl") is not None else None,
                    "reason": (reason or "EXIT")[:800],
                    "pos_count": int(len(self.executor.positions or {})),
                    "cash": float(self._last_cash or 0),
                    "equity": float(self._last_equity or 0),
                })
            except Exception:
                pass

            if ok:
                # ✅ [ADD] Backtest 동일 cooldown 갱신
                try:
                    self._apply_cooldown_after_exit(sym=sym, candle_t=candle_t, exit_price=float(exit_px), pos=pos)
                except Exception:
                    pass

                try:
                    self.executor.positions.pop(sym, None)
                except Exception:
                    pass
                try:
                    self._save_state()
                except Exception:
                    pass

                try:
                    if getattr(self, "notifier", None):
                        self.notifier.send(
                            title="EXIT",
                            lines=[
                                f"t={market_data.get('candle_time')}",
                                f"symbol={sym}",
                                f"side={pos.get('side')}",
                                f"exit_price={exit_px}",
                                f"reason={reason}",
                            ],
                        )
                except Exception as e:
                    logger.error(f"EXIT telegram failed | {sym} | {e}")

                return "EXIT"

            return "HOLD"

        return "HOLD"

    def _manage_positions_live(self, authority_time, active_symbols, realtime_snapshot: dict = None):
        """
        ✅ Backtest-Identical Manage Loop
        - SL/EXIT/UPDATE_SL 판단은 _get_manage_row()가 준 '확정봉 row'만 사용한다.
        - row가 없으면 해당 심볼은 관리 스킵(=백테 기준 정합)
        - 단, MTM equity 계산용 current_prices는:
            row가 있으면 row['close']
            row가 없고 포지션이 있으면 ticker last로만 보강 (SL 판단에는 사용 금지)
        반환:
        current_prices(dict), exit_ct(int), upd_ct(int), manage_cnt(int)
        """
        pos_keys = (self.executor.positions or {}).keys()
        manage_syms = set(active_symbols or []) | set(pos_keys)
        manage_cnt = int(len(manage_syms))

        current_prices = {}
        exit_ct = 0
        upd_ct = 0

        def _sf(x, default=None):
            try:
                v = float(x)
                if not math.isfinite(v):
                    return default
                return v
            except Exception:
                return default

        for sym in manage_syms:
            row = self._get_manage_row(sym, authority_time, realtime_snapshot=None)

            # ---- 1) row 기반 가격(확정봉 close)
            if row is not None:
                try:
                    px = _sf(row.get("close", None), None)
                    if px is not None:
                        current_prices[sym] = float(px)
                except Exception:
                    pass

                # ✅ SL/EXIT/UPDATE_SL 판단은 row 있을 때만 수행
                if sym in (self.executor.positions or {}):
                    act = self._process_existing_position_live(sym, row)
                    if act == "EXIT":
                        exit_ct += 1
                    elif act == "UPDATE_SL":
                        upd_ct += 1
                continue

            # ---- 2) row가 없으면: SL 판단은 스킵 (Backtest 기준)
            # 포지션이 있는 경우에만 MTM 가격을 ticker로 보강
            if sym in (self.executor.positions or {}):
                try:
                    if hasattr(self.executor, "fetch_ticker"):
                        t = self.executor.fetch_ticker(sym) or {}
                        last = _sf((t or {}).get("last", None), None)
                        if last is not None:
                            current_prices[sym] = float(last)
                except Exception:
                    pass

        return current_prices, exit_ct, upd_ct, manage_cnt


    def _force_mark_to_market_equity_live(self, prices: dict) -> float:
        """
        백테와 동일한 강제 MTM:
        equity = cash_free + Σ(margin + unrealized_pnl)
        - cash는 fetch_balance().USDT.free(가용) 기준을 유지
        - margin은 pos.margin(잠긴 돈)으로 더해서 total-equity를 재구성
        """
        cash = float(self._last_cash or 0.0)
        positions = (self.executor.positions or {})

        total = cash
        for sym, pos in positions.items():
            try:
                if sym not in prices:
                    continue
                px = float(prices[sym])
                entry = float(pos.get("entry_price", 0.0) or 0.0)
                amt = float(pos.get("amount", 0.0) or 0.0)
                side = str(pos.get("side", "LONG")).upper()
                margin = float(pos.get("margin", 0.0) or 0.0)

                if amt <= 0:
                    continue

                if side == "LONG":
                    upnl = (px - entry) * amt
                else:
                    upnl = (entry - px) * amt

                total += margin + upnl
            except Exception:
                continue

        # 단일 진실원: executor.equity도 이 값으로 덮는다
        try:
            self.executor.equity = float(total)
        except Exception:
            pass

        return float(total)

    def _sync_balance_authority(self):
        """
        ✅ Balance Authority (LIVE)
        - cash_free 권위: fetch_balance().USDT.free
        - equity_total 권위: 엔진 MTM 재구성 값(=cash_free + Σ(margin + uPnL))
        (run_once_live에서 _force_mark_to_market_equity_live()로 최종 확정)

        ✅ Swap/오염 방지:
        - executor.update_equity() 등 외부 메서드가 cash/equity 의미를 바꿔도
        self._last_cash/_last_equity 권위를 덮어쓰지 않는다.
        """
        try:
            bal = self.executor.fetch_balance() or {}
            usdt = bal.get("USDT", {}) if isinstance(bal, dict) else {}
            cash_free = float(usdt.get("free", 0) or 0)
            eq_total = float(usdt.get("total", 0) or 0)

            # 기본 관계 체크: total >= free 가 일반적.
            # 깨지면 exchange/ccxt 응답 이상 or 심볼/계정 유형 차이일 수 있으니 "free"만 신뢰.
            if eq_total > 0 and cash_free > eq_total:
                # total이 더 작게 오는 케이스 방어
                eq_total = cash_free

            self._last_cash = float(cash_free)
            # ⚠️ equity_total은 여기서 확정하지 않는다 (MTM으로 확정)
            self._last_equity = float(eq_total) if (self._last_equity is None) else float(self._last_equity)

            # RiskControl이 executor.equity를 보므로 cash는 free로, equity는 "현재 last_equity"로 미러
            try:
                if hasattr(self.executor, "cash"):
                    self.executor.cash = float(self._last_cash or 0)
                if hasattr(self.executor, "equity"):
                    # 여기선 임시값(이후 MTM으로 덮음)
                    self.executor.equity = float(self._last_equity or 0)
            except Exception:
                pass

        except Exception as e:
            logger.error(f"[BAL_AUTH] fetch_balance failed: {e}")
            # 실패 시: 기존 값 유지 + executor 값으로도 덮지 않음
            if self._last_cash is None:
                self._last_cash = float(getattr(self.executor, "cash", 0) or 0)
            if self._last_equity is None:
                self._last_equity = float(getattr(self.executor, "equity", 0) or 0)
    

    # -----------------------------------------------------
    # LIVE main loop (15m)
    # -----------------------------------------------------
    def run_once_live(self):
        def _canon(sym: str) -> str:
            try:
                s = str(sym or "").strip()
            except Exception:
                return str(sym)
            if not s:
                return s
            if ":USDT" in s:
                return s
            if "/" in s:
                if s.endswith("/USDT"):
                    return s + ":USDT"
                return s
            if s.endswith("USDT") and len(s) > 4:
                base = s[:-4]
                return f"{base}/USDT:USDT"
            return s

        b = self._current_bucket_15m()

        if self.last_bucket is None or b != self.last_bucket:
            self.prepare_data()
            self.last_bucket = b

        if not self.data_map:
            return

        # ✅ 이 루프의 권위 잔고(cash_free) 동기화 (swap 방지)
        self._sync_balance_authority()
        logger.info(f"💰 BAL_SYNC | cash_free={float(self._last_cash or 0):.2f} equity_total(raw)={float(self._last_equity or 0):.2f}")

        if self.freeze_new_entries:
            try:
                self.reconcile_positions()
            except Exception as e:
                logger.error(f"[RUN] reconcile_positions(freeze) failed: {e}")

        candle_ts_mode = self.cfg.get("system_settings", {}).get("candle_ts_mode", "open")
        if candle_ts_mode not in ("open", "close"):
            candle_ts_mode = "open"

        current_time, active_symbols = self._authority_time_15m(candle_ts_mode=candle_ts_mode)
        if current_time is None or not active_symbols:
            logger.warning("⚠️ TIME_AUTHORITY_NONE | skip loop")
            return

        active_symbols = [_canon(s) for s in (active_symbols or []) if _canon(s)]
        active_symbols = list(dict.fromkeys(active_symbols))

        try:
            self.reconcile_positions()
        except Exception as e:
            logger.error(f"[RUN] reconcile_positions failed: {e}")
            self._append_ops({
                "dt": str(pd.to_datetime(current_time)),
                "event": "RECONCILE_FAIL",
                "mode": "LIVE",
                "reason": f"{type(e).__name__}: {e}",
                "pos_count": int(len(self.executor.positions or {})),
                "t_ref": str(pd.to_datetime(current_time)),
            })

        try:
            self._sweep_pending_entry_orders(b)
        except Exception as e:
            logger.error(f"[RUN] sweep_pending_entry_orders failed: {e}")
            self._append_ops({
                "dt": str(pd.to_datetime(current_time)),
                "event": "PENDING_SWEEP_FAIL",
                "mode": "LIVE",
                "reason": f"{type(e).__name__}: {e}",
                "pos_count": int(len(self.executor.positions or {})),
                "t_ref": str(pd.to_datetime(current_time)),
            })

        try:
            self._sync_external_closes(current_time)
        except Exception as e:
            logger.error(f"[RUN] sync_external_closes failed: {e}")
            self._append_ops({
                "dt": str(pd.to_datetime(current_time)),
                "event": "EXTERNAL_SYNC_FAIL",
                "mode": "LIVE",
                "reason": f"{type(e).__name__}: {e}",
                "pos_count": int(len(self.executor.positions or {})),
                "t_ref": str(pd.to_datetime(current_time)),
            })

        # positions 키 표준화(런타임 오염 방어)
        try:
            old_pos = self.executor.positions or {}
            if isinstance(old_pos, dict) and old_pos:
                new_pos = {}
                for k, v in old_pos.items():
                    nk = _canon(k)
                    if nk in new_pos:
                        continue
                    new_pos[nk] = v
                self.executor.positions = new_pos
        except Exception:
            pass

        pos_keys = list((self.executor.positions or {}).keys())
        manage_syms = sorted(set(active_symbols or []) | set(pos_keys))
        rt_snapshot = self._fetch_realtime_snapshot(manage_syms)

        # ✅ manage first
        current_prices, exit_ct, upd_ct, manage_cnt = self._manage_positions_live(
            authority_time=current_time,
            active_symbols=active_symbols,
            realtime_snapshot=rt_snapshot,
        )

        # ✅ equity_total 권위 확정: MTM 강제 재구성
        mtm_equity = self._force_mark_to_market_equity_live(current_prices)
        self._last_equity = float(mtm_equity)
        try:
            self.executor.equity = float(mtm_equity)
        except Exception as e:
            logger.error(f"[RUN] executor.equity set failed: {e}")

        # ✅ entry
        entry_ct = 0
        cand_ct = 0

        if not self.freeze_new_entries:
            candidates = self._compute_candidates_15m(current_time, symbols=active_symbols)
            cand_ct = int(len(candidates))

            for c in candidates:
                try:
                    c["rt_snapshot"] = rt_snapshot
                except Exception:
                    pass

            max_pos = self._max_positions_live()
            pos_set = set((self.executor.positions or {}).keys())

            for cand in candidates:
                if len(self.executor.positions or {}) >= max_pos:
                    break

                try:
                    cand["sym"] = _canon(cand.get("sym"))
                except Exception:
                    pass
                csym = cand.get("sym")
                if not csym:
                    continue
                if csym in pos_set:
                    continue

                ok = self._process_entry_live(cand)
                if ok:
                    entry_ct += 1
                    pos_set.add(csym)
        else:
            logger.warning("🧊 ENTRY_FROZEN | reconcile mismatch or safety condition - no new entries")
            self._append_history({
                "dt": str(pd.to_datetime(current_time)),
                "event": "ENTRY_FROZEN",
                "mode": "LIVE",
                "symbol": "",
                "side": "",
                "reason": "freeze_new_entries=1",
                "pos_count": int(len(self.executor.positions or {})),
                "cash": float(self._last_cash or 0),
                "equity": float(self._last_equity or 0),
            })
            self.notifier.send(
                title="ENTRY_FROZEN",
                lines=[
                    f"t={pd.to_datetime(current_time)}",
                    "freeze_new_entries=1",
                    f"pos_count={int(len(self.executor.positions or {}))}",
                    "reason=reconcile/safety",
                ],
            )

        # ✅ update_equity 호출은 유지하되, 결과로 cash/equity를 다시 덮지 않는다 (swap 차단)
        try:
            self.executor.update_equity(current_prices)
        except Exception as e:
            logger.error(f"[RUN] update_equity failed: {e}")

        # ✅ 최종 equity_total은 다시 MTM으로 확정(단일 진실원)
        try:
            mtm_equity2 = self._force_mark_to_market_equity_live(current_prices)
            self._last_equity = float(mtm_equity2)
            try:
                self.executor.equity = float(mtm_equity2)
            except Exception:
                pass
        except Exception as e:
            logger.error(f"[RUN] mtm failed: {e}")

        # ✅ cash_free는 루프 시작에 동기화한 값 유지 (executor.cash 재흡수 금지)
        # (필요 시 다음 루프에서 _sync_balance_authority로 갱신)

        self.last_processed_time = str(pd.to_datetime(current_time))
        self._save_state()

        hb_time = pd.to_datetime(current_time)
        hb_reason = (
            f"universe(active)={len(active_symbols)} manage={int(manage_cnt)} "
            f"cand={cand_ct} entry={entry_ct} exit={int(exit_ct)} updSL={int(upd_ct)} "
            f"{self._freeze_meta()} eq={float(self._last_equity or 0):.2f}"
        )

        logger.info(
            f"💓 HEARTBEAT | t={hb_time} | "
            f"universe(active)={len(active_symbols)} manage={int(manage_cnt)} cand={cand_ct} "
            f"entry={entry_ct} exit={int(exit_ct)} updSL={int(upd_ct)} | "
            f"pos={len(self.executor.positions or {})} | {self._freeze_meta()} | "
            f"eq={float(self._last_equity or 0):.2f} cash={float(self._last_cash or 0):.2f}"
        )

        if bool(self.cfg.get("system_settings", {}).get("log_heartbeat_to_csv", False)):
            self._append_history({
                "dt": str(hb_time),
                "event": "HEARTBEAT",
                "mode": "LIVE",
                "symbol": "",
                "side": "",
                "reason": hb_reason,
                "pos_count": int(len(self.executor.positions or {})),
                "cash": float(self._last_cash or 0),
                "equity": float(self._last_equity or 0),
            })

        if bool(self.notifier.send_heartbeat):
            self.notifier.send(
                title="HEARTBEAT",
                lines=[
                    f"t={hb_time}",
                    f"universe(active)={len(active_symbols)}",
                    f"manage={int(manage_cnt)}",
                    f"cand={cand_ct} entry={entry_ct} exit={int(exit_ct)} updSL={int(upd_ct)}",
                    f"pos={len(self.executor.positions or {})}",
                    f"freeze={int(self.freeze_new_entries)}",
                    f"equity_total={float(self._last_equity or 0)}",
                    f"cash_free={float(self._last_cash or 0)}",
                ],
            )







if __name__ == "__main__":
    engine = LiveEngine()

    # 시작 시 한 번 미리 준비
    try:
        engine.prepare_data()
    except Exception as e:
        logger.error(f"Startup prepare_data failed: {e}")

    # 15m 경계로 정렬 후 루프
    engine._sleep_until_next_15m()
    while True:
        try:
            engine.run_once_live()
            engine._sleep_until_next_15m()
        except KeyboardInterrupt:
            logger.info("🛑 Manual stop")
            break
        except Exception as e:
            logger.error(f"Live Engine Error: {e}")
            time.sleep(10)
