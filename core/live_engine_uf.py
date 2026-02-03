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
# 6) Restart Safety: reconcile_positions mismatch 시 ENTRY FREEZE (청산/관리만 허용)
#
# ✅ Telegram Alert:
# - BOOT / RECONCILE_OK / RECONCILE_MISMATCH
# - ENTRY / ENTRY_REJECT / ENTRY_FAIL
# - UPDATE_SL (옵션) / EXIT
# =========================================================

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

        self.send_update_sl = bool(ss.get("telegram_send_update_sl", False))
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

        # --- [UF] enable flag ---
    def _is_uf_enabled(self) -> bool:
        try:
            return bool(self.cfg.get("system_settings", {}).get("use_universe_filter", False))
        except Exception:
            return False

    # --- [UF] read universe file (JSON) ---
    def _get_universe_from_json(self) -> list:
        """
        UF는 엔진 밖에서 돌고, LIVE는 결과 JSON만 읽는다.
        기본 경로:
          - config: system_settings.universe_selected_path
          - default: <root_dir>/universe_selected.json
        JSON 예:
          {"asof":"2026-02-03","rebalance":"daily","symbols":[...], "top_n":25}
        """
        try:
            path = self.cfg.get("system_settings", {}).get("universe_selected_path", None)
        except Exception:
            path = None

        if not path:
            path = os.path.join(self.root_dir, "universe_selected.json")

        if not os.path.exists(path):
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f) or {}
        except Exception:
            return []

        syms = obj.get("symbols", []) or []
        if not isinstance(syms, list):
            return []

        out = []
        for s in syms:
            try:
                ss = str(s).strip()
                if ss:
                    out.append(ss)
            except Exception:
                continue
        return out




    def __init__(self):
        # paths
        self.root_dir = root_dir
        self.config_path = os.path.join(root_dir, "config.json")
        self.history_path = os.path.join(root_dir, "trade_history.csv")
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

        # history/state init
        self._ensure_history_file()
        self._load_state()

        # reconcile at boot (LIVE only)
        self.reconcile_positions()

        # BOOT log
        boot_dt = pd.Timestamp.utcnow()
        

        # ✅ BOOT telegram
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
        # ✅ trades-only 유지(필터는 _append_history에서), 컬럼은 넓게(깨짐 방지)
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



    def _ensure_history_file(self):
        try:
            if (not os.path.exists(self.history_path)) or os.path.getsize(self.history_path) == 0:
                pd.DataFrame(columns=self._history_columns()).to_csv(self.history_path, index=False)
        except Exception as e:
            logger.error(f"History init failed: {e}")

    def _append_history(self, row: dict):
        try:
            ev = str((row or {}).get("event", "")).upper().strip()
            if ev not in {"ENTRY", "UPDATE_SL", "EXIT"}:
                return  # ✅ trades-only

            cols = self._history_columns()
            base = {c: None for c in cols}
            base.update(row or {})

            # ✅ CSV 안정성: 콤마/개행 제거 (reason 때문에 깨지는거 방지)
            if base.get("reason") is not None:
                r = str(base["reason"])
                r = r.replace("\n", " ").replace("\r", " ").replace(",", ";")
                base["reason"] = r

            df = pd.DataFrame([base], columns=cols)
            df.to_csv(self.history_path, mode="a", header=False, index=False)
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
            return

        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            self.executor.positions = state.get("positions", {}) or {}
            self.last_processed_time = state.get("last_processed_time", None)
            self.last_bucket = state.get("last_bucket", None)

        except Exception as e:
            logger.error(f"State load failed: {e}")
            self.executor.positions = {}
            self.last_processed_time = None
            self.last_bucket = None

    def _save_state(self):
        state = {
            "positions": self.executor.positions,
            "last_processed_time": self.last_processed_time,
            "last_bucket": self.last_bucket,
        }
        try:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"State save failed: {e}")



    def _set_entry_freeze(self, reason: str):
        self.freeze_new_entries = True
        try:
            self.notifier._freeze_reason = str(reason)
            self.notifier._freeze_since_utc = str(pd.Timestamp.utcnow())
        except Exception:
            pass

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
    def reconcile_positions(self, amount_tol=0.02):
        """
        ✅ 실계좌(바이낸스) 권위화는 하되,
        - side/amount/entry_price/margin 만 바이낸스를 권위로 갱신
        - sl/tp1/next_sl/tp1_hit/entry_time 등 '관리필드'는 로컬을 유지
        - 로컬에 없고 실계좌에만 있으면: 로컬 포지션을 생성(관리필드 빈칸 가능)
        - 실계좌에 없고 로컬에만 있으면: 로컬 포지션 제거 (유령 포지션 제거)
        """
        if not hasattr(self.executor, "fetch_positions"):
            self.freeze_new_entries = False
            return

        try:
            real = self.executor.fetch_positions() or {}
        except Exception as e:
            logger.error(f"fetch_positions failed: {e}")
            real = {}

        local = self.executor.positions or {}

        # ---- helper: 관리필드 목록(로컬에서 유지할 것들)
        MGMT_KEYS = {
            "sl", "next_sl", "entry_time",
            # 필요하면 여기에 추가 가능
        }

        # ---- 1) 실계좌에 있는 포지션을 로컬에 merge (바이낸스 권위 필드만 덮음)
        merged = {}

        for sym, rp in real.items():
            lp = local.get(sym, {}) if isinstance(local.get(sym, {}), dict) else {}

            # 로컬 관리필드 유지
            keep_mgmt = {k: lp.get(k) for k in MGMT_KEYS if k in lp}

            # 바이낸스 권위 필드
            mp = {}
            try:
                mp["side"] = str(rp.get("side", "")).upper()
            except Exception:
                mp["side"] = lp.get("side", "")
            try:
                mp["amount"] = float(rp.get("amount", 0) or 0)
            except Exception:
                mp["amount"] = float(lp.get("amount", 0) or 0)

            # entry_price/margin best-effort
            if rp.get("entry_price") is not None:
                try:
                    mp["entry_price"] = float(rp.get("entry_price"))
                except Exception:
                    pass
            else:
                if lp.get("entry_price") is not None:
                    mp["entry_price"] = lp.get("entry_price")

            if rp.get("margin") is not None:
                try:
                    mp["margin"] = float(rp.get("margin"))
                except Exception:
                    pass
            else:
                if lp.get("margin") is not None:
                    mp["margin"] = lp.get("margin")

            # merge
            mp.update(keep_mgmt)

            # entry_time이 없으면 보정(재시작/리컨실로 새로 붙는 포지션)
            if mp.get("entry_time") in (None, "", "None"):
                mp["entry_time"] = str(pd.Timestamp.utcnow())

            merged[sym] = mp

        # ---- 2) 실계좌에 없는 로컬 포지션 제거 (유령 제거)
        # merged만 남김 = real이 진실이므로 local-only는 제거

        # ---- 3) mismatch 판단 (진입 freeze는 mismatch일 때만)
        mismatch = False
        reasons = []

        def _u(x): 
            return str(x or "").upper()

        # real vs merged(=반영후 로컬) 비교는 의미 없고,
        # real vs local(기존) 불일치가 있었는지 기록만 남긴다.
        for sym, rp in real.items():
            if sym not in local:
                mismatch = True
                reasons.append(f"real_has_local_missing:{sym}")
                continue
            lp = local.get(sym, {}) or {}
            if _u(rp.get("side")) != _u(lp.get("side")):
                mismatch = True
                reasons.append(f"side_mismatch:{sym}")
                continue
            try:
                ra = float(rp.get("amount", 0) or 0)
                la = float(lp.get("amount", 0) or 0)
                if ra > 0 and la > 0:
                    if abs(ra - la) / max(ra, 1e-12) > amount_tol:
                        mismatch = True
                        reasons.append(f"amount_mismatch:{sym}")
            except Exception:
                mismatch = True
                reasons.append(f"amount_parse_fail:{sym}")

        for sym in (local or {}).keys():
            if sym not in real:
                mismatch = True
                reasons.append(f"local_has_real_missing:{sym}")
                break

        # ---- 4) 적용 + 저장
        self.executor.positions = merged
        self._save_state()  # ✅ 리컨실 결과 즉시 영구화

        now = pd.Timestamp.utcnow()

        if mismatch:
            # mismatch가 있었더라도 이제 merged로 정합성 맞췄으니
            # "신규진입 freeze"는 운영 철학에 따라 선택인데,
            # 너는 자동관리 최우선이라 freeze는 최소화가 맞다.
            # -> 여기서는 진입 freeze를 걸지 않고, 대신 로그/텔레그램만 남긴다.
            self._clear_entry_freeze("reconcile_mismatch_merged")
           


            msg = " | ".join(reasons)[:800]
            logger.warning("⚠️ RECONCILE_MISMATCH detected but merged -> keep managing existing positions")

            self._append_history({
                "dt": str(now),
                "event": "RECONCILE_MISMATCH",
                "mode": "LIVE",
                "symbol": "",
                "side": "",
                "reason": msg,
                "pos_count": int(len(self.executor.positions or {})),
            })

            self.notifier.send(
                title="RECONCILE_MISMATCH (MERGED)",
                lines=[
                    f"t_utc={now}",
                    "freeze_new_entries=0",
                    f"pos_count={int(len(self.executor.positions or {}))}",
                    f"reason={msg}",
                ],
            )
        else:
            self._clear_entry_freeze("reconcile_ok")
            logger.info("✅ RECONCILE_OK (binance authoritative merge ok)")

            self._append_history({
                "dt": str(now),
                "event": "RECONCILE_OK",
                "mode": "LIVE",
                "symbol": "",
                "side": "",
                "reason": "binance_authoritative_merge_ok",
                "pos_count": int(len(self.executor.positions or {})),
            })

            self.notifier.send(
                title="RECONCILE_OK",
                lines=[
                    f"t_utc={now}",
                    "freeze_new_entries=0",
                    f"pos_count={int(len(self.executor.positions or {}))}",
                ],
            )

    def _verify_position_state(self, sym: str):
        """
        ✅ Binance Authority Verify
        - 반환: (is_open, real_pos_dict_or_None)
        - is_open=False: 바이낸스에 포지션이 없음(청산 확정)
        - is_open=True : 바이낸스에 포지션이 있음(진입/잔존)
        - fetch 실패 시: (None, None)  # 상위 로직에서 안전조치
        """
        if not hasattr(self.executor, "fetch_positions"):
            return None, None

        try:
            real = self.executor.fetch_positions() or {}
        except Exception as e:
            logger.error(f"verify fetch_positions failed: {e}")
            return None, None

        rp = real.get(sym)
        if not rp:
            return False, None
        try:
            amt = float(rp.get("amount", 0) or 0)
            if amt <= 0:
                return False, None
        except Exception:
            return True, rp

        return True, rp



    # -----------------------------------------------------
    # Indicators (engine responsibility) - 15m only
    # -----------------------------------------------------
    def rebuild_indicators(self):
        self.data_map = {}

        required_cols = [
            "open", "high", "low", "close", "volume",
            "atr", "vol_ma", "ema_intra", "rsi", "adx", "st_val", "st_dir"
        ]

        temp_map = {}
        warmup_map = {}

        for sym, df in self.raw_data_map.items():
            try:
                ind = self.titan.calculate_indicators(sym, df.copy())

                missing = [c for c in required_cols if c not in ind.columns]
                if missing:
                    logger.warning(f"[Indicator] {sym} missing required cols: {missing}")
                    continue

                if ind[required_cols].isna().all().all():
                    logger.warning(f"[Indicator] {sym} required columns all-NaN (pre-slice).")
                    continue

                bad_cols = [c for c in required_cols if ind[c].isna().all()]
                if bad_cols:
                    logger.warning(f"[Indicator] {sym} has all-NaN required cols (pre-slice): {bad_cols}")
                    continue

                col_warmups = []
                for c in required_cols:
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

                sliced = sliced.dropna(subset=required_cols)
                if len(sliced) == 0:
                    logger.warning(f"[Indicator] {sym} all NaN after drop (required subset)")
                    continue

                self.data_map[sym] = sliced

            except Exception as e:
                logger.warning(f"[Indicator] {sym} slice failed: {e}")

        logger.info(f"Indicators Ready: {len(self.data_map)} symbols processed.")

    # -----------------------------------------------------
    # Data preparation (15m)
    # -----------------------------------------------------
    def prepare_data(self):
                # ✅ UF JSON 우선 (enabled일 때만)
        if self._is_uf_enabled():
            targets = self._get_universe_from_json() or []
            if not targets:
                targets = self.executor.get_top_targets() or []
                logger.info(f"📡 MARKET_SCAN | UF_EMPTY -> fallback top_targets={targets[:10]} (top10)")
            else:
                logger.info(f"📡 MARKET_SCAN | UF_JSON targets={targets[:10]} (top10)")
        else:
            targets = self.executor.get_top_targets() or []
            logger.info(f"📡 MARKET_SCAN | targets={targets[:10]} (top10)")


        filtered = []
        for sym in targets:
            clean = sym.split(":")[0]
            if clean in self.titan.blacklist or sym in self.titan.blacklist:
                continue
            filtered.append(sym)

        if not filtered:
            logger.error("❌ No targets after blacklist filter.")
            self.raw_data_map = {}
            self.data_map = {}
            self.symbols = []
            return

        raw_map = self.executor.prepare_data(filtered) or {}

        validated_map = {}
        for sym, df in raw_map.items():
            v = DataLoader.validate_and_format(df)
            if v is None:
                logger.warning(f"⚠️ DATA_DROP | {sym} validate failed")
                continue

            v = v.copy()
            v["datetime"] = pd.to_datetime(v["timestamp"], unit="ms", errors="coerce")
            v = v.dropna(subset=["datetime"]).set_index("datetime")
            if v.empty:
                continue

            validated_map[sym] = v

        self.raw_data_map = {k: validated_map[k] for k in sorted(validated_map.keys())}
        self.rebuild_indicators()
        self.symbols = sorted(self.data_map.keys())

        logger.info(f"📊 INDICATORS_READY | symbols={self.symbols}")
        logger.info(f"📥 Data prepared: raw={len(self.raw_data_map)} | ready={len(self.data_map)}")

    # -----------------------------------------------------
    # Time / Loop Authority
    # -----------------------------------------------------
    def _current_bucket_15m(self):
        return int(time.time() // self.tf15_sec)

    def _sleep_until_next_15m(self):
        now = time.time()
        next_tick = (math.floor(now / self.tf15_sec) * self.tf15_sec) + self.tf15_sec
        sleep_sec = (next_tick - now) + self.loop_buffer_sec
        if sleep_sec < 1:
            sleep_sec += self.tf15_sec
        logger.info(f"⏳ Waiting {sleep_sec:.2f}s for next 15m close...")
        time.sleep(sleep_sec)

    def _get_sl_apply_mode(self) -> str:
        """
        system_settings.sl_apply_mode: same/next
        """
        try:
            mode = str(self.cfg.get("system_settings", {}).get("sl_apply_mode", "next")).strip().lower()
        except Exception:
            mode = "next"
        return mode if mode in ("next", "same") else "next"

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

    # -----------------------------------------------------
    # Candle Picking (15m stale blocking fix)
    # -----------------------------------------------------
    def _pick_current_time_15m(self, stale_lag_sec=3600, min_coverage=0.90, max_back_steps=6):
        """
        ✅ LIVE Time Authority (15m) — '진짜 최신 캔들(ref)' 강제
        - ref = max(last_ts) (stale 제외 후)
        - active_symbols = ref 캔들을 실제로 가진 심볼만
        - 커버리지로 ref를 과거로 "당기지 않는다" (드리프트 차단)
        """
        if not self.data_map or not self.symbols:
            return None, []

        # 1) 심볼별 last_ts 수집
        last_times = {}
        for sym in self.symbols:
            df = self.data_map.get(sym)
            if df is None or df.empty or len(df) < 2:
                continue
            last_times[sym] = df.index[-1]

        if not last_times:
            return None, []

        # 2) ref = 진짜 최신
        ref = max(last_times.values())

        # 3) stale 제거
        good_syms = []
        for sym, t in last_times.items():
            try:
                lag = (ref - t).total_seconds()
            except Exception:
                continue
            if lag > stale_lag_sec:
                logger.warning(f"⚠️ Stale symbol filtered: {sym} last={t} lag={lag:.0f}s")
                continue
            good_syms.append(sym)

        if not good_syms:
            return None, []

        # 4) ref 캔들을 실제로 가진 심볼만 active
        active = []
        for sym in good_syms:
            df = self.data_map.get(sym)
            if df is None or df.empty:
                continue
            if ref in df.index:
                active.append(sym)

        if not active:
            return None, []

        cov = len(active) / float(len(good_syms)) if good_syms else 0.0
        if cov < 1.0:
            missing = [s for s in good_syms if s not in set(active)]
            logger.warning(
                f"⚠️ TIME_AUTHORITY_PARTIAL | t_ref={ref} coverage={cov:.2%} "
                f"active={len(active)}/{len(good_syms)} missing_sample={missing[:10]}"
            )
        else:
            logger.info(f"✅ TIME_AUTHORITY_OK | t_ref={ref} coverage=100% syms={len(active)}")

        return ref, active

    def _get_manage_row(self, sym, authority_time):
        """
        보유 포지션 관리용 캔들/가격 확보
        우선순위:
        1) data_map[sym][authority_time]
        2) data_map[sym] 마지막 캔들
        3) ticker fallback
        반환: dict or pd.Series or None
        """
        df = self.data_map.get(sym)

        if isinstance(df, pd.DataFrame) and not df.empty:
            if authority_time in df.index:
                return df.loc[authority_time]
            return df.iloc[-1]

        # ticker fallback
        try:
            t = self.executor.fetch_ticker(sym)
            px = float(t.get("last") or t.get("close") or 0)
            if px > 0:
                return {
                    "close": px,
                    "high": px,
                    "low": px,
                    "atr": px * 0.01,
                    "st_val": 0,
                }
        except Exception:
            pass

        return None


    def _compute_candidates_15m(self, current_time_15m, symbols=None):
        """
        ✅ candidates 계산도 '동일 캔들 active_symbols' 기준으로만 수행
        """
        candidates = []
        if not self.data_map:
            return candidates

        syms = symbols if symbols is not None else self.symbols
        if not syms:
            return candidates

        debug_entry = bool(self.cfg.get("system_settings", {}).get("debug_entry_trace", False))
        logger.info(f"🔍 ENTRY_SCAN_START | t={current_time_15m} | universe={len(syms)}")

        miss_time = 0
        too_short = 0
        analyze_fail = 0
        no_signal = 0
        had_signal = 0

        def _snap_safe(series):
            try:
                return {
                    "adx": series.get("adx", None),
                    "st_dir": series.get("st_dir", None),
                    "st_val": series.get("st_val", None),
                    "ema_daily": series.get("ema_daily", None),
                    "ema_intra": series.get("ema_intra", None),
                    "vol_ma": series.get("vol_ma", None),
                    "rsi": series.get("rsi", None),
                    "close": series.get("close", None),
                    "volume": series.get("volume", None),
                    "retest_long": series.get("retest_long", None),
                    "retest_short": series.get("retest_short", None),
                    "mss_up": series.get("mss_up", None),
                    "mss_down": series.get("mss_down", None),
                }
            except Exception:
                return {}

        for sym in syms:
            df = self.data_map.get(sym)
            if df is None or df.empty:
                continue
            if current_time_15m not in df.index:
                miss_time += 1
                continue

            try:
                idx = df.index.get_loc(current_time_15m)
                start = max(0, idx - 250)
                past_data = df.iloc[start: idx + 1]
            except Exception:
                miss_time += 1
                continue

            if len(past_data) < 2:
                too_short += 1
                continue

            try:
                signal, sl, tp = self.titan.analyze(sym, past_data)
            except Exception as e:
                analyze_fail += 1
                logger.warning(f"⚠️ ANALYZE_FAIL | {sym} err={e}")
                continue

            if debug_entry:
                try:
                    curr = past_data.iloc[-1]
                    logger.info(f"🧪 PRE_ANALYZE | {sym} t={current_time_15m} rows={len(past_data)} snap={_snap_safe(curr)}")
                except Exception:
                    pass

            if not signal:
                no_signal += 1
                continue

            had_signal += 1
            curr_row = df.loc[current_time_15m]

            try:
                score = curr_row.get("adx", 0)
                score = float(score) if score is not None else 0.0
            except Exception:
                score = 0.0

            candidates.append({
                "score": float(score),
                "sym": sym,
                "signal": signal,
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
                f"universe={len(syms)} miss_time={miss_time} too_short={too_short} "
                f"analyze_fail={analyze_fail} no_signal={no_signal} had_signal={had_signal} cand={len(candidates)}"
            )

        return candidates



    # -----------------------------------------------------
    # LIVE entry/exit helpers
    # -----------------------------------------------------
    def _max_positions_live(self):
        mp = getattr(self.executor, "MAX_POSITIONS", None)
        if mp is None:
            mp = self.cfg.get("risk_settings", {}).get("max_open_positions", 3)
        try:
            mp = int(mp)
            if mp <= 0:
                mp = 1
        except Exception:
            mp = 3
        return mp

    def _process_entry_live(self, cand):
        """
        ✅ 백테 정합성: sizing은 'equity(total)' 기준으로 통일
        - cash/free 기준 sizing 금지 (마진/포지션수/청산시점 연쇄 드리프트 원인)
        - 그래도 현금부족(가용)으로 주문이 거절될 수 있으니 cash는 로그/리젝트 원인용으로만 남김
        """
        sym = cand["sym"]
        row = cand["row"]
        signal = cand.get("signal", "")
        sl = cand.get("sl", None)
        tp = cand.get("tp", None)

        candle_t = pd.to_datetime(row.name)
        price = float(row["close"])

        raw_sig = str(signal)
        sig = raw_sig.strip().upper()
        alias = {
            "BUY": "LONG",
            "LONG": "LONG",
            "BULL": "LONG",
            "SELL": "SHORT",
            "SHORT": "SHORT",
            "BEAR": "SHORT",
        }
        signal_side = alias.get(sig, None)

        if signal_side is None:
            logger.info(f"⚪ ENTRY_REJECT | {sym} signal_invalid raw={raw_sig}")
            self._append_history({
                "dt": str(candle_t),
                "event": "ENTRY_REJECT",
                "mode": "LIVE",
                "symbol": sym,
                "side": str(sig),
                "price": float(price),
                "amount": 0.0,
                "sl": float(sl) if sl is not None else None,
                "reason": f"signal_invalid:{raw_sig}",
                "pos_count": int(len(self.executor.positions or {})),
                "cash": float(self._last_cash or 0),
                "equity": float(self._last_equity or 0),
            })
            self.notifier.send(
                title="ENTRY_REJECT",
                lines=[
                    f"t={candle_t}",
                    f"symbol={sym}",
                    f"side={sig}",
                    f"price={price}",
                    f"sl={sl}",
                    f"tp={tp}",
                    f"cash={float(self._last_cash or 0)}",
                    f"equity={float(self._last_equity or 0)}",
                    f"reason=signal_invalid:{raw_sig}",
                ],
            )
            return False

        # --- 현재 루프의 권위 잔고(이미 run_once_live에서 sync했지만, 0이면 한번 더 보정) ---
        cash_now = float(self._last_cash or 0)
        equity_now = float(self._last_equity or 0)

        if equity_now <= 0:  # ✅ sizing 베이스는 equity이므로 equity가 0/None이면 보정
            try:
                bal = self.executor.fetch_balance() or {}
                usdt = bal.get("USDT", {}) if isinstance(bal, dict) else {}
                cash_now = float(usdt.get("free", 0) or 0)
                equity_now = float(usdt.get("total", 0) or 0)
                self._last_cash, self._last_equity = cash_now, equity_now
                if hasattr(self.executor, "cash"):
                    self.executor.cash = cash_now
                if hasattr(self.executor, "equity"):
                    self.executor.equity = equity_now
            except Exception:
                pass

        # ✅ 핵심 변경: cash/free -> equity(total) 기반으로 sizing
        sizing_base = float(equity_now or 0)

        try:
            amount = self.risk_ctrl.calculate_entry_size(sym, price, sizing_base, sl, signal_side)
        except Exception as e:
            logger.error(f"ENTRY sizing failed: {e}")
            amount = 0.0

        if amount <= 0:
            logger.info(
                f"⚪ ENTRY_REJECT | {sym} {signal_side} price={price:.6f} sl={sl} "
                f"sizing_equity={sizing_base:.2f} cash_free={cash_now:.2f} -> amount={amount}"
            )
            self._append_history({
                "dt": str(candle_t),
                "event": "ENTRY_REJECT",
                "mode": "LIVE",
                "symbol": sym,
                "side": str(signal_side).upper(),
                "price": float(price),
                "amount": float(amount),
                "sl": float(sl) if sl is not None else None,
                "reason": "amount<=0 (sizing=equity)",
                "pos_count": int(len(self.executor.positions or {})),
                "cash": float(cash_now),
                "equity": float(sizing_base),
            })
            self.notifier.send(
                title="ENTRY_REJECT",
                lines=[
                    f"t={candle_t}",
                    f"symbol={sym}",
                    f"side={str(signal_side).upper()}",
                    f"price={price}",
                    f"amount={amount}",
                    f"sl={sl}",
                    f"tp={tp}",
                    f"sizing_equity={sizing_base}",
                    f"cash_free={cash_now}",
                    "reason=amount<=0 (sizing=equity)",
                ],
            )
            return False

        # --- 주문 ---
        result = self.executor.create_order(sym, signal_side, amount)
        if not result:
            logger.error(f"❌ ENTRY_FAIL | {sym} {signal_side} amt={amount}")
            self._append_history({
                "dt": str(candle_t),
                "event": "ENTRY_FAIL",
                "mode": "LIVE",
                "symbol": sym,
                "side": str(signal_side).upper(),
                "price": float(price),
                "amount": float(amount),
                "sl": float(sl) if sl is not None else None,
                "reason": "executor_create_order_failed",
                "pos_count": int(len(self.executor.positions or {})),
                "cash": float(cash_now),
                "equity": float(sizing_base),
            })
            self.notifier.send(
                title="ENTRY_FAIL",
                lines=[
                    f"t={candle_t}",
                    f"symbol={sym}",
                    f"side={str(signal_side).upper()}",
                    f"price={price}",
                    f"amount={amount}",
                    f"sl={sl}",
                    f"tp={tp}",
                    f"sizing_equity={sizing_base}",
                    f"cash_free={cash_now}",
                    "reason=executor_create_order_failed",
                ],
            )
            return False

        entry_price = float(result.get("filled_price", price) or price)
        filled_qty = float(result.get("filled_qty", amount) or amount)
        margin = float(result.get("margin", 0) or 0)
        fee = result.get("fee", None)

        # local cache (임시) 후 verify
        self.executor.positions[sym] = {
            "side": str(signal_side).upper(),
            "amount": float(filled_qty),
            "entry_price": float(entry_price),
            "sl": float(sl) if sl is not None else None,
            "next_sl": None,
            "entry_time": str(candle_t),
            "margin": float(margin),
        }
        try:
            self._save_state()
        except Exception:
            pass

        is_open, rp = self._verify_position_state(sym)
        if is_open is None:
            logger.critical(f"🚨 ENTRY_VERIFY_FETCH_FAIL -> ENTRY FREEZE | {sym}")
            self._set_entry_freeze("entry_verify_fetch_fail")
            self.reconcile_positions()  # reconcile_ok면 내부에서 clear됨
        elif is_open is False:
            logger.critical(f"🚨 ENTRY_VERIFY_FAIL (no position on binance) -> ENTRY FREEZE | {sym}")
            self._set_entry_freeze("entry_verify_no_position")
            self.reconcile_positions()  # reconcile_ok면 내부에서 clear됨
        else:
            try:
                if isinstance(rp, dict):
                    if rp.get("amount") is not None:
                        self.executor.positions[sym]["amount"] = float(rp.get("amount") or self.executor.positions[sym]["amount"])
                    if rp.get("entry_price") is not None:
                        self.executor.positions[sym]["entry_price"] = float(rp.get("entry_price") or self.executor.positions[sym]["entry_price"])
                    if rp.get("margin") is not None:
                        self.executor.positions[sym]["margin"] = float(rp.get("margin") or self.executor.positions[sym]["margin"])
            except Exception:
                pass
        self._clear_entry_freeze("entry_verify_ok")
        try:
            self._save_state()
        except Exception:
            pass

        logger.info(
            f"🟢 ENTRY | {sym} {signal_side} qty={self.executor.positions[sym]['amount']} "
            f"px={self.executor.positions[sym]['entry_price']} sl={sl} tp={tp} sizing_equity={sizing_base:.2f}"
        )

        self._append_history({
            "dt": str(candle_t),
            "event": "ENTRY",
            "mode": "LIVE",
            "symbol": sym,
            "side": str(signal_side).upper(),
            "price": float(self.executor.positions[sym]["entry_price"]),
            "amount": float(self.executor.positions[sym]["amount"]),
            "fee": float(fee) if fee is not None else None,
            "margin": float(self.executor.positions[sym].get("margin", margin) or 0),
            "pnl": 0.0,
            "sl": float(sl) if sl is not None else None,
            "reason": "signal_entry (sizing=equity)",
            "pos_count": int(len(self.executor.positions or {})),
            "cash": float(cash_now),
            "equity": float(sizing_base),
        })

        self.notifier.send(
            title="ENTRY",
            lines=[
                f"t={candle_t}",
                f"symbol={sym}",
                f"side={str(signal_side).upper()}",
                f"entry_price={self.executor.positions[sym]['entry_price']}",
                f"qty={self.executor.positions[sym]['amount']}",
                f"margin={self.executor.positions[sym].get('margin', margin)}",
                f"fee={fee}",
                f"sl={sl}",
                f"tp={tp}",
                f"sizing_equity={sizing_base}",
                f"cash_free={cash_now}",
                f"pos_count={int(len(self.executor.positions or {}))}",
                f"freeze_new_entries={int(self.freeze_new_entries)}",
            ],
        )

        return True




    def _process_existing_position_live(self, sym, curr_row):
        """
        ✅ LIVE에서도 Backtest와 동일한 PositionMonitor 입력을 제공:
        - market_data: close/high/low/atr/st_val/adx + df(hist_df)
        - sl_strategy: armor 포함
        """
        pos = self.executor.positions.get(sym)
        if not pos:
            return "NONE"

        candle_t = pd.to_datetime(curr_row.name)

        # ---- apply mode (single source: config) ----
        try:
            apply_mode = str(self.cfg.get("system_settings", {}).get("sl_apply_mode", "next")).strip().lower()
        except Exception:
            apply_mode = "next"
        if apply_mode not in ("next", "same"):
            apply_mode = "next"

        # -------------------------------------------------
        # ✅ next_sl 승계는 "next 모드"에서만 수행
        # -------------------------------------------------
        if apply_mode == "next":
            try:
                if ("next_sl" in pos) and (pos.get("next_sl") is not None):
                    nxt = float(pos.get("next_sl"))
                    cur = pos.get("sl", None)
                    cur_f = float(cur) if cur is not None else None

                    if (cur_f is None) or (nxt != cur_f):
                        pos["sl"] = float(nxt)

                    pos["next_sl"] = None
                    self._save_state()
            except Exception:
                pass
        else:
            try:
                if pos.get("next_sl") is not None:
                    pos["next_sl"] = None
                    self._save_state()
            except Exception:
                pass

        # ✅ config 기반 SL 전략/파라미터
        sl_strategy = self._get_sl_strategy()
        sl_params = self._get_sl_params()

        # -------------------------------------------------
        # ✅ hist_df 구성 (Backtest와 동일한 목적: armor/컨텍스트 전략 입력)
        # - self.data_map[sym]에서 현재 시점까지 slice
        # - 컬럼명 표준화(open/high/low/close/volume)
        # - sl_params["armor_lookback"] 있으면 제한 (기본 300)
        # -------------------------------------------------
        hist_df = None
        try:
            df_full = self.data_map.get(sym)
            if isinstance(df_full, pd.DataFrame) and (not df_full.empty):
                # 현재 시점까지 포함
                hist_df = df_full.loc[:curr_row.name].copy()

                # 컬럼명 표준화(소문자)
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

                # lookback 제한
                try:
                    lb = int(sl_params.get("armor_lookback", 300))
                    if lb > 0 and len(hist_df) > lb:
                        hist_df = hist_df.tail(lb)
                except Exception:
                    pass
        except Exception:
            hist_df = None

        # -------------------------------------------------
        # ✅ market_data 안전 변환 (NaN/inf/0 방어)
        # -------------------------------------------------
        def _safe_float(x, default=0.0):
            try:
                v = float(x)
                if not math.isfinite(v):
                    return float(default)
                return v
            except Exception:
                return float(default)

        close = _safe_float(curr_row.get("close", 0.0), 0.0)
        high  = _safe_float(curr_row.get("high", close), close)
        low   = _safe_float(curr_row.get("low", close), close)

        atr = curr_row.get("atr", None)
        atr = _safe_float(atr, close * 0.01)
        if atr <= 0:
            atr = close * 0.01

        st_val = curr_row.get("st_val", None)
        try:
            st_val = float(st_val) if st_val is not None else None
            if (st_val is not None) and (not math.isfinite(st_val)):
                st_val = None
        except Exception:
            st_val = None

        adx = _safe_float(curr_row.get("adx", 0.0), 0.0)

        market_data = {
            "close": close,
            "high": high,
            "low": low,
            "atr": atr,
            "st_val": st_val,
            "adx": adx,
            "df": hist_df,
        }

        # -------------------------------------------------
        # ✅ PositionMonitor 호출
        # -------------------------------------------------
        action, exec_price, reason, new_sl = self.monitor.check_conditions(
            sym,
            pos,
            market_data,
            sl_apply_mode=apply_mode,
            sl_strategy=sl_strategy,
            sl_params=sl_params,
        )

        # -------------------------------------------------
        # ✅ UPDATE_SL
        # -------------------------------------------------
        if action == "UPDATE_SL":
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
                            self._save_state()
                    else:
                        if (cur_f is None) or (new_sl_f != cur_f):
                            pos["sl"] = new_sl_f
                            pos["next_sl"] = None
                            self._save_state()
            except Exception:
                pass

            return "UPDATE_SL"

        if action != "EXIT":
            return "HOLD"

        # -------------------------------------------------
        # ✅ EXIT (이하 기존 로직 그대로)
        # -------------------------------------------------
        is_open_pre, _rp_pre = self._verify_position_state(sym)
        if is_open_pre is False:
            logger.warning(f"🟠 MANUAL_EXIT_DETECTED | {sym} -> local pop + clear freeze")

            self.executor.positions.pop(sym, None)
            self._save_state()
            self._clear_entry_freeze("manual_exit_detected")

            try:
                self._append_history({
                    "dt": str(candle_t),
                    "event": "EXIT",
                    "mode": "LIVE",
                    "symbol": sym,
                    "side": str(pos.get("side", "")).upper(),
                    "price": None,
                    "amount": float(pos.get("amount", 0)),
                    "margin": float(pos.get("margin", 0) or 0),
                    "pnl": None,
                    "roe_pct": None,
                    "sl": float(pos.get("sl", 0)) if pos.get("sl") is not None else None,
                    "reason": "MANUAL_EXIT_DETECTED",
                    "pos_count": int(len(self.executor.positions or {})),
                })
            except Exception:
                pass

            try:
                self.notifier.send(
                    title="EXIT",
                    lines=[
                        f"t={candle_t}",
                        f"symbol={sym}",
                        f"side={str(pos.get('side','')).upper()}",
                        "exec_price=None",
                        f"amount={float(pos.get('amount',0))}",
                        "pnl_est=None",
                        "roe_est_pct=None",
                        f"sl={pos.get('sl', None)}",
                        "reason=MANUAL_EXIT_DETECTED",
                        "close_call_ok=0",
                        "closed_ok=1",
                        f"{self._freeze_meta()}",
                    ],
                )
            except Exception:
                pass

            return "EXIT"

        # ---- 기존 EXIT 로직 계속 ----
        logger.info(f"🔴 EXIT | {sym} px={exec_price} reason={reason}")

        ok_close = False
        try:
            ok_close = bool(self.executor.close_position(sym, exec_price, reason))
        except Exception:
            ok_close = False

        is_open, _rp = self._verify_position_state(sym)

        if is_open is None:
            logger.critical(f"🚨 EXIT_VERIFY_FETCH_FAIL -> ENTRY FREEZE | {sym}")
            self._set_entry_freeze("exit_verify_fetch_fail")
            self.reconcile_positions()
            closed_ok = 0
        elif is_open is True:
            logger.critical(f"🚨 EXIT_VERIFY_FAIL (still open) -> ENTRY FREEZE | {sym}")
            self._set_entry_freeze("exit_verify_still_open")
            self.reconcile_positions()
            closed_ok = 0
        else:
            closed_ok = 1
            self._clear_entry_freeze("exit_verify_ok")

        pnl = None
        roe = None
        fee = None
        margin = float(pos.get("margin", 0) or 0)
        try:
            px = float(exec_price)
            entry = float(pos.get("entry_price", 0))
            amt = float(pos.get("amount", 0))
            side = str(pos.get("side", "")).upper()
            pnl = (px - entry) * amt if side == "LONG" else (entry - px) * amt
            fee = px * amt * BASE_FEE
            if margin > 0:
                roe = (float(pnl) / float(margin)) * 100.0
        except Exception:
            pass

        try:
            self._append_history({
                "dt": str(candle_t),
                "event": "EXIT",
                "mode": "LIVE",
                "symbol": sym,
                "side": str(pos.get("side", "")).upper(),
                "price": float(exec_price) if exec_price is not None else None,
                "amount": float(pos.get("amount", 0)),
                "fee": float(fee) if fee is not None else None,
                "margin": float(margin),
                "pnl": float(pnl) if pnl is not None else None,
                "roe_pct": float(roe) if roe is not None else None,
                "sl": float(pos.get("sl", 0)) if pos.get("sl") is not None else None,
                "reason": f"{reason} | close_call={int(ok_close)} | closed_ok={int(closed_ok)} | strat={sl_strategy}",
                "pos_count": int(max(len(self.executor.positions or {}) - (1 if closed_ok else 0), 0)),
            })
        except Exception:
            pass

        try:
            self.notifier.send(
                title="EXIT",
                lines=[
                    f"t={candle_t}",
                    f"symbol={sym}",
                    f"side={str(pos.get('side','')).upper()}",
                    f"exec_price={exec_price}",
                    f"amount={float(pos.get('amount',0))}",
                    f"pnl_est={pnl}",
                    f"roe_est_pct={roe}",
                    f"sl={pos.get('sl', None)}",
                    f"reason={reason}",
                    f"close_call_ok={int(ok_close)}",
                    f"closed_ok={int(closed_ok)}",
                    f"{self._freeze_meta()}",
                    f"apply_mode={apply_mode}",
                    f"sl_strategy={sl_strategy}",
                ],
            )
        except Exception:
            pass

        if closed_ok == 1:
            self.executor.positions.pop(sym, None)
            self._save_state()
            return "EXIT"

        return "HOLD"
    

    def _manage_positions_live(self, authority_time, active_symbols):
        """
        ✅ 관리 단일 권위 def
        - 관리 대상 = active_symbols ∪ 보유 포지션
        - authority_time 캔들 없으면 ticker fallback(또는 마지막 캔들)로 관리
        - equity 업데이트용 current_prices를 함께 반환
        반환:
        current_prices(dict), exit_ct(int), upd_ct(int), manage_cnt(int)
        """
        pos_keys = (self.executor.positions or {}).keys()
        manage_syms = set(active_symbols or []) | set(pos_keys)
        manage_cnt = int(len(manage_syms))

        current_prices = {}
        exit_ct = 0
        upd_ct = 0

        for sym in manage_syms:
            row = self._get_manage_row(sym, authority_time)
            if row is None:
                continue

            try:
                px = float(row["close"])
                current_prices[sym] = px
            except Exception:
                continue

            if sym in (self.executor.positions or {}):
                act = self._process_existing_position_live(sym, row)
                if act == "EXIT":
                    exit_ct += 1
                elif act == "UPDATE_SL":
                    upd_ct += 1

        return current_prices, exit_ct, upd_ct, manage_cnt

    # -----------------------------------------------------
    # LIVE main loop (15m)
    # -----------------------------------------------------
    def run_once_live(self):
        b = self._current_bucket_15m()

        if self.last_bucket is None or b != self.last_bucket:
            self.prepare_data()
            self.last_bucket = b

        if not self.data_map:
            return

        # ✅ 이 루프의 권위 잔고를 먼저 고정(=백테에서의 equity 입력과 동일한 권위)
        try:
            bal = self.executor.fetch_balance() or {}
            usdt = bal.get("USDT", {}) if isinstance(bal, dict) else {}
            self._last_cash = float(usdt.get("free", 0) or 0)    # 참고/로그/주문거절 원인용
            self._last_equity = float(usdt.get("total", 0) or 0) # ✅ sizing 권위
        except Exception as e:
            logger.error(f"[RUN] fetch_balance failed: {e}")
            self._last_cash = float(getattr(self.executor, "cash", self._last_cash or 0) or 0)
            self._last_equity = float(getattr(self.executor, "equity", self._last_equity or 0) or 0)

        # RiskControl이 executor.equity를 볼 수 있으니 같이 맞춤(=단일 진실원)
        try:
            if hasattr(self.executor, "cash"):
                self.executor.cash = float(self._last_cash or 0)
            if hasattr(self.executor, "equity"):
                self.executor.equity = float(self._last_equity or 0)
        except Exception:
            pass

        logger.info(f"💰 BAL_SYNC | cash_free={self._last_cash:.2f} equity_total={self._last_equity:.2f}")
        
        # ✅ freeze 상태면, entry 스킵 전에 reconcile로 자동 해제 시도
        if self.freeze_new_entries:
            try:
                self.reconcile_positions()
            except Exception:
                pass

        # ✅ Time Authority: 진짜 최신(ref) 강제
        current_time, active_symbols = self._pick_current_time_15m()
        if current_time is None or not active_symbols:
            logger.warning("⚠️ TIME_AUTHORITY_NONE | skip loop")
            return

        if self.last_processed_time is not None:
            try:
                if pd.to_datetime(current_time) <= pd.to_datetime(self.last_processed_time):
                    return
            except Exception:
                pass

        # -----------------------------------------------------
        # ✅ manage: active_symbols만 X (active ∪ 보유포지션)
        # -----------------------------------------------------
        current_prices, exit_ct, upd_ct, manage_cnt = self._manage_positions_live(
            authority_time=current_time,
            active_symbols=active_symbols,
        )

        # -----------------------------------------------------
        # entry: active_symbols 기준 유지(“동일 캔들” 정합성)
        # -----------------------------------------------------
        entry_ct = 0
        cand_ct = 0

        if not self.freeze_new_entries:
            candidates = self._compute_candidates_15m(current_time, symbols=active_symbols)
            cand_ct = int(len(candidates))

            max_pos = self._max_positions_live()
            for cand in candidates:
                if len(self.executor.positions or {}) >= max_pos:
                    break
                if cand["sym"] in (self.executor.positions or {}):
                    continue
                ok = self._process_entry_live(cand)  # ✅ 내부에서 sizing=equity
                if ok:
                    entry_ct += 1
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

        # -----------------------------------------------------
        # ✅ equity update: 보유 포지션 심볼 가격을 반드시 포함한 current_prices로 갱신
        # -----------------------------------------------------
        try:
            self.executor.update_equity(current_prices)
        except Exception:
            pass

        # ✅ [FIX] update_equity() 결과를 _last_equity(및 executor.*)에 재동기화
        # - heartbeat/CSV/telegram의 eq 값이 stale(=fetch_balance 시점)로 남는 문제 해결
        try:
            if hasattr(self.executor, "equity"):
                self._last_equity = float(getattr(self.executor, "equity", self._last_equity or 0) or 0)
            if hasattr(self.executor, "cash"):
                # cash는 실계좌 free와 다를 수 있으니, executor가 유지한다면 함께 동기화(로그 일관성)
                self._last_cash = float(getattr(self.executor, "cash", self._last_cash or 0) or 0)
        except Exception:
            pass

        self.last_processed_time = str(pd.to_datetime(current_time))
        self._save_state()

        # -----------------------------------------------------
        # ✅ HEARTBEAT (logger / csv / telegram 동일 포맷 정합)
        # -----------------------------------------------------
        hb_time = pd.to_datetime(current_time)
        hb_reason = (
            f"universe(active)={len(active_symbols)} manage={int(manage_cnt)} "
            f"cand={cand_ct} entry={entry_ct} exit={exit_ct} updSL={upd_ct} "
            f"{self._freeze_meta()} eq={float(self._last_equity or 0):.2f}"
        )

        logger.info(
            f"💓 HEARTBEAT | t={hb_time} | "
            f"universe(active)={len(active_symbols)} manage={int(manage_cnt)} cand={cand_ct} "
            f"entry={entry_ct} exit={exit_ct} updSL={upd_ct} | "
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
                    f"cand={cand_ct} entry={entry_ct} exit={exit_ct} updSL={upd_ct}",
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
