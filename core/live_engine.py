# core/live_engine.py
# =========================================================
# [Phalanx Core Module] Live Engine
# Mode: Time-Sequential & Priority-Based (Backtest-Identical)
# DRY_RUN:
# - 주문 전송 X, 체결 시뮬레이션 O
# - 잔고/에퀴티는 실계좌 대신 가상자본(기본 10,000) 사용
#
# 핵심 보강(감사 반영):
# 1) Time Authority: 15m 경계(00/15/30/45) 직후로 루프 정렬 (drift 제거)
# 2) Stale Blocking: now_utc 비교 금지. 심볼 간 상대 lag(ref=max(last_ts))로 stale 필터
# 3) Data Refresh: 새 캔들(15m)이 바뀌는 순간에만 prepare_data() 재호출
#
# [추가 - 연속성]
# 4) trade_history.csv 자동 생성/append (ENTRY/EXIT/UPDATE_SL/REJECT/HEARTBEAT(optional))
# 5) last_processed_time / last_bucket state 저장/복구
# 6) DRY_RUN 재시작 시 state가 없거나 초기화된 경우 trade_history.csv에서 paper_cash/equity 복구
#
# [DRY_RUN 투트랙]
# - Track A: entry=15m / manage=15m
# - Track B: entry=15m / manage=1m
# - ENTRY 로직은 절대 분기하지 않음 (15m 확정 시점에 1회 계산 후 A/B 복제)
# - 갈라지는 건 포지션 관리 루프만
# =========================================================

import sys
import os
import json
import time
import math
import logging
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from strategy.position_monitor import PositionMonitor
from strategy.risk_control import RiskControl
from strategy.titan_strategy import TitanStrategy
from execution.binance_executor import BinanceExecutor
from utils.data_loader import DataLoader

BASE_FEE = 0.0005

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("PhalanxLive")


class DryRunTrack:
    """
    DRY_RUN Track (paper trading)
    - trade_history_{track_id}.csv append
    - phalanx_state_{track_id}.json save/restore
    - cooldown / consecutive_losses
    - DEBUG: 실제 바인딩된 메서드 라인 출력
    """

    def __init__(self, track_id: str, entry_tf: str, manage_tf: str, cfg: dict, executor: BinanceExecutor):
        self.track_id = str(track_id)
        self.entry_tf = str(entry_tf)
        self.manage_tf = str(manage_tf)

        self.cfg = cfg or {}
        self.executor = executor

        self.state_path = os.path.join(root_dir, f"phalanx_state_{self.track_id}.json")
        self.history_path = os.path.join(root_dir, f"trade_history_{self.track_id}.csv")

        self.paper_equity0 = float(self.cfg.get("system_settings", {}).get("paper_equity", 10000.0))
        self.paper_cash = float(self.paper_equity0)
        self.paper_equity = float(self.paper_equity0)

        self.positions = {}             # {sym: {side, amount, entry_price, sl, tp1, ...}}
        self.cooldowns = {}             # {sym: timestamp}
        self.consecutive_losses = {}    # {sym: int}

        self.last_entry_bucket = None
        self.last_manage_bucket = None
        self.last_processed_entry_time = None
        self.last_processed_manage_time = None

        # 1) history/state 준비
        self._ensure_history_file()
        self._load_state()
        self._restore_paper_from_history_if_needed()

        # 2) BOOT 로그 + 바인딩 디버그
        boot_time = pd.Timestamp.utcnow()
        self._append_history({
            "dt": str(boot_time),
            "event": "BOOT",
            "mode": "DRY_RUN",
            "symbol": "",
            "side": "",
            "reason": f"engine_start track={self.track_id} entry={self.entry_tf} manage={self.manage_tf}",
            "pos_count": int(len(self.positions)),
            "cash": float(self.paper_cash),
            "equity": float(self.paper_equity),
        })

        self._debug_method_binding()

        # 3) 저장
        self._save_state()

    # -------------------------
    # Debug: which method version is actually bound
    # -------------------------
    def _debug_method_binding(self):
        try:
            import inspect

            def loc(fn):
                try:
                    file = inspect.getsourcefile(fn) or "?"
                    line = fn.__code__.co_firstlineno
                    return f"{os.path.basename(file)}:{line}"
                except Exception:
                    return "?:?"

            logger.info(
                f"🧩 DRYRUN_BINDING | track={self.track_id} "
                f"open_position@{loc(self.open_position)} "
                f"handle_exit@{loc(self.handle_exit)} "
                f"_save_state@{loc(self._save_state)} "
                f"_load_state@{loc(self._load_state)} "
                f"_append_history@{loc(self._append_history)}"
            )
        except Exception as e:
            logger.warning(f"[Track {self.track_id}] binding debug failed: {e}")

    # -------------------------
    # History helpers
    # -------------------------
    def _history_columns(self):
        return [
            "dt", "event", "mode", "symbol", "side",
            "price", "amount", "fee", "margin",
            "pnl", "roe_pct",
            "sl", "tp1",
            "reason", "pos_count", "cash", "equity",
        ]

    def _ensure_history_file(self):
        try:
            if (not os.path.exists(self.history_path)) or os.path.getsize(self.history_path) == 0:
                pd.DataFrame(columns=self._history_columns()).to_csv(self.history_path, index=False)
        except Exception as e:
            logger.error(f"[Track {self.track_id}] History init failed: {e}")

    def _append_history(self, row: dict):
        try:
            cols = self._history_columns()
            base = {c: None for c in cols}
            base.update(row or {})
            df = pd.DataFrame([base], columns=cols)
            df.to_csv(self.history_path, mode="a", header=False, index=False)
        except Exception as e:
            logger.error(f"[Track {self.track_id}] History write failed: {e}")

    def _restore_paper_from_history_if_needed(self):
        """
        state의 paper_cash/equity가 '초기값 수준'이면
        history 마지막 cash/equity로 복구.
        """
        try:
            if not os.path.exists(self.history_path) or os.path.getsize(self.history_path) == 0:
                return

            df = pd.read_csv(self.history_path, on_bad_lines="skip")
            if df.empty:
                return

            # DRY_RUN만
            if "mode" in df.columns:
                df = df[df["mode"] == "DRY_RUN"]
                if df.empty:
                    return

            # cash/equity 유효한 마지막
            if ("cash" not in df.columns) or ("equity" not in df.columns):
                return

            df2 = df.dropna(subset=["cash", "equity"], how="any")
            if df2.empty:
                return

            last = df2.iloc[-1]
            last_cash = float(last["cash"])
            last_eq = float(last["equity"])

            # state가 초기값(=paper_equity0)과 같으면 history 값으로 복구
            if abs(float(self.paper_cash) - float(self.paper_equity0)) < 1e-6:
                self.paper_cash = float(last_cash)
            if abs(float(self.paper_equity) - float(self.paper_equity0)) < 1e-6:
                self.paper_equity = float(last_eq)

        except Exception as e:
            logger.error(f"[Track {self.track_id}] Restore paper from history failed: {e}")

    # -------------------------
    # State
    # -------------------------
    def _load_state(self):
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            self.positions = state.get("positions", {}) or {}
            self.consecutive_losses = state.get("consecutive_losses", {}) or {}

            # cooldowns는 timestamp string -> Timestamp로 복구
            cd = state.get("cooldowns", {}) or {}
            restored = {}
            for k, v in cd.items():
                try:
                    restored[k] = pd.to_datetime(v)
                except Exception:
                    restored[k] = v
            self.cooldowns = restored

            pc = state.get("paper_cash", None)
            pe = state.get("paper_equity", None)
            p0 = state.get("paper_equity0", None)

            if p0 is not None:
                try:
                    self.paper_equity0 = float(p0)
                except Exception:
                    pass
            if pc is not None:
                try:
                    self.paper_cash = float(pc)
                except Exception:
                    pass
            if pe is not None:
                try:
                    self.paper_equity = float(pe)
                except Exception:
                    pass

            self.last_entry_bucket = state.get("last_entry_bucket", None)
            self.last_manage_bucket = state.get("last_manage_bucket", None)
            self.last_processed_entry_time = state.get("last_processed_entry_time", None)
            self.last_processed_manage_time = state.get("last_processed_manage_time", None)

        except Exception as e:
            logger.error(f"[Track {self.track_id}] State load failed: {e}")

    def _save_state(self):
        cooldown_dump = {}
        for k, v in (self.cooldowns or {}).items():
            try:
                cooldown_dump[k] = str(pd.to_datetime(v))
            except Exception:
                cooldown_dump[k] = str(v)

        state = {
            "positions": self.positions,
            "cooldowns": cooldown_dump,
            "consecutive_losses": self.consecutive_losses,
            "paper_cash": float(self.paper_cash),
            "paper_equity": float(self.paper_equity),
            "paper_equity0": float(self.paper_equity0),
            "last_entry_bucket": self.last_entry_bucket,
            "last_manage_bucket": self.last_manage_bucket,
            "last_processed_entry_time": self.last_processed_entry_time,
            "last_processed_manage_time": self.last_processed_manage_time,
        }
        try:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[Track {self.track_id}] State save failed: {e}")

    # -------------------------
    # Accounting
    # -------------------------
    def mtm_update_equity(self, current_prices: dict):
        """
        equity = cash + (모든 포지션의 margin + pnl)
        current_prices: {sym: last_price}
        """
        eq = float(self.paper_cash)

        for sym, pos in (self.positions or {}).items():
            try:
                # ❗ 가격이 없으면 MTM 금지 (shadow smoothing 방지)
                if sym not in current_prices:
                    eq += float(pos.get("margin", 0))
                    continue

                px = float(current_prices[sym])
                entry = float(pos.get("entry_price", 0))
                amt = float(pos.get("amount", 0))
                side = str(pos.get("side", "")).upper()
                margin = float(pos.get("margin", 0))

                pnl = (px - entry) * amt if side == "LONG" else (entry - px) * amt
                eq += (margin + pnl)

            except Exception:
                # 계산 실패 시에도 margin은 보존
                try:
                    eq += float(pos.get("margin", 0))
                except Exception:
                    pass

        self.paper_equity = eq
        return eq


    # -------------------------
    # Cooldown
    # -------------------------
    def update_cooldown(self, sym, exit_price, pos):
        try:
            price = float(exit_price)
            entry = float(pos.get("entry_price", 0))
            amt = float(pos.get("amount", 0))
            side = str(pos.get("side", "")).upper()
            pnl = (price - entry) * amt if side == "LONG" else (entry - price) * amt
        except Exception:
            pnl = 0.0

        now = pd.Timestamp.utcnow()

        if pnl > 0:
            self.consecutive_losses[sym] = 0
            self.cooldowns[sym] = now
        else:
            streak = int(self.consecutive_losses.get(sym, 0)) + 1
            self.consecutive_losses[sym] = streak

            if streak == 1:
                wait = 8
            elif streak == 2:
                wait = 24
            elif streak == 3:
                wait = 48
            else:
                wait = 96

            self.cooldowns[sym] = now + pd.Timedelta(hours=wait)

        logger.info(
            f"🧊 COOLDOWN | track={self.track_id} {sym} pnl={pnl:.4f} streak={self.consecutive_losses.get(sym,0)} "
            f"until={self.cooldowns.get(sym)}"
        )

    def is_in_cooldown(self, sym, candle_time):
        cd = self.cooldowns.get(sym)
        if cd is None:
            return False
        try:
            cd_t = pd.to_datetime(cd)
            return pd.to_datetime(candle_time) < cd_t
        except Exception:
            return False

    # -------------------------
    # Trade ops (DRY_RUN)
    # -------------------------
    def can_open_more(self):
        max_pos = int(self.cfg.get("risk_settings", {}).get("max_open_positions", 3))
        return len(self.positions) < max_pos

    def open_position(self, sym, signal, fill_px, amount_raw, sl, tp, entry_time):
        """
        엔진에서 동일 entry 신호/가격/수량이 들어옴(복제).
        트랙은 자기 cash/cooldown/poslimit만 보고 수락/거절.
        """
        # already has
        if sym in self.positions:
            return False

        # pos limit
        if not self.can_open_more():
            self._append_history({
                "dt": str(pd.to_datetime(entry_time)),
                "event": "ENTRY_REJECT",
                "mode": "DRY_RUN",
                "symbol": sym,
                "side": signal,
                "price": float(fill_px),
                "amount": float(amount_raw),
                "reason": f"max_positions track={self.track_id}",
                "pos_count": int(len(self.positions)),
                "cash": float(self.paper_cash),
                "equity": float(self.paper_equity),
                "sl": float(sl) if sl is not None else None,
                "tp1": float(tp) if tp is not None else None,
            })
            return False

        # cooldown
        if self.is_in_cooldown(sym, entry_time):
            self._append_history({
                "dt": str(pd.to_datetime(entry_time)),
                "event": "ENTRY_REJECT",
                "mode": "DRY_RUN",
                "symbol": sym,
                "side": signal,
                "price": float(fill_px),
                "amount": float(amount_raw),
                "reason": f"cooldown track={self.track_id}",
                "pos_count": int(len(self.positions)),
                "cash": float(self.paper_cash),
                "equity": float(self.paper_equity),
                "sl": float(sl) if sl is not None else None,
                "tp1": float(tp) if tp is not None else None,
            })
            return False

        # qty precision
        try:
            filled_qty = float(self.executor.amount_to_precision(sym, float(amount_raw)))
        except Exception:
            filled_qty = float(amount_raw)

        if filled_qty <= 0:
            self._append_history({
                "dt": str(pd.to_datetime(entry_time)),
                "event": "ENTRY_REJECT",
                "mode": "DRY_RUN",
                "symbol": sym,
                "side": signal,
                "price": float(fill_px),
                "amount": float(filled_qty),
                "reason": f"amount<=0 track={self.track_id}",
                "pos_count": int(len(self.positions)),
                "cash": float(self.paper_cash),
                "equity": float(self.paper_equity),
                "sl": float(sl) if sl is not None else None,
                "tp1": float(tp) if tp is not None else None,
            })
            return False

        # fee/margin
        fee = float(fill_px) * float(filled_qty) * BASE_FEE
        leverage = self.cfg.get("risk_settings", {}).get("leverage", 1)
        try:
            leverage = float(leverage)
            if leverage <= 0:
                leverage = 1.0
        except Exception:
            leverage = 1.0

        margin = (float(fill_px) * float(filled_qty)) / float(leverage)
        required = margin + fee

        # cash check
        if float(self.paper_cash) < float(required):
            self._append_history({
                "dt": str(pd.to_datetime(entry_time)),
                "event": "ENTRY_REJECT",
                "mode": "DRY_RUN",
                "symbol": sym,
                "side": signal,
                "price": float(fill_px),
                "amount": float(filled_qty),
                "fee": float(fee),
                "margin": float(margin),
                "reason": f"cash<{required:.2f} track={self.track_id}",
                "pos_count": int(len(self.positions)),
                "cash": float(self.paper_cash),
                "equity": float(self.paper_equity),
                "sl": float(sl) if sl is not None else None,
                "tp1": float(tp) if tp is not None else None,
            })
            return False

        # accept
        self.paper_cash -= float(required)

        self.positions[sym] = {
            "side": str(signal).upper(),
            "amount": float(filled_qty),
            "entry_price": float(fill_px),
            "sl": float(sl) if sl is not None else None,
            "tp1": float(tp) if tp is not None else None,
            "next_sl": None,
            "tp1_hit": False,
            "entry_time": str(entry_time),
            "margin": float(margin),
        }

        logger.info(
            f"🟦 DRY_RUN ENTRY | track={self.track_id} {sym} {signal} qty={filled_qty} px={fill_px} "
            f"sl={sl} tp={tp} fee={fee:.4f} margin={margin:.2f} cash={self.paper_cash:.2f}"
        )

        self._append_history({
            "dt": str(pd.to_datetime(entry_time)),
            "event": "ENTRY",
            "mode": "DRY_RUN",
            "symbol": sym,
            "side": str(signal).upper(),
            "price": float(fill_px),
            "amount": float(filled_qty),
            "fee": float(fee),
            "margin": float(margin),
            "pnl": 0.0,
            "pos_count": int(len(self.positions)),
            "cash": float(self.paper_cash),
            "equity": float(self.paper_equity),
            "sl": float(sl) if sl is not None else None,
            "tp1": float(tp) if tp is not None else None,
            "reason": f"track={self.track_id}",
        })

        self._save_state()
        return True

    def apply_next_sl_if_any(self, sym):
        pos = self.positions.get(sym)
        if not pos:
            return
        if pos.get("next_sl", None) is not None:
            try:
                if float(pos["next_sl"]) != float(pos.get("sl", 0) or 0):
                    pos["sl"] = float(pos["next_sl"])
            except Exception:
                pass
            pos["next_sl"] = None

    def handle_update_sl(self, sym, candle_time, close_px, reason, new_sl):
        try:
            pos = self.positions.get(sym, {})
            if not pos:
                return
            if new_sl is None:
                return
            if float(new_sl) == float(pos.get("sl", 0) or 0):
                return

            pos["next_sl"] = float(new_sl)

            logger.info(f"🟡 UPDATE_SL | track={self.track_id} {sym} next_sl={pos['next_sl']} reason={reason}")

            self._append_history({
                "dt": str(pd.to_datetime(candle_time)),
                "event": "UPDATE_SL",
                "mode": "DRY_RUN",
                "symbol": sym,
                "side": str(pos.get("side", "")),
                "price": float(close_px),
                "amount": float(pos.get("amount", 0)),
                "margin": float(pos.get("margin", 0)),
                "sl": float(pos.get("next_sl", 0)),
                "tp1": float(pos.get("tp1", 0)) if pos.get("tp1") is not None else None,
                "reason": f"{reason} track={self.track_id}",
                "pos_count": int(len(self.positions)),
                "cash": float(self.paper_cash),
                "equity": float(self.paper_equity),
            })
        except Exception:
            return

    def handle_exit(self, sym, candle_time, exec_px, reason):
        pos = self.positions.get(sym)
        if not pos:
            return

        pnl = None
        fee = None
        margin = float(pos.get("margin", 0) or 0)

        try:
            px = float(exec_px)
            entry = float(pos.get("entry_price", 0))
            amt = float(pos.get("amount", 0))
            side = str(pos.get("side", "")).upper()

            pnl = (px - entry) * amt if side == "LONG" else (entry - px) * amt
            fee = px * amt * BASE_FEE

            self.paper_cash += (margin + pnl - fee)
        except Exception:
            pass

        logger.info(f"🟦 DRY_RUN EXIT | track={self.track_id} {sym} px={exec_px} reason={reason} cash={self.paper_cash:.2f}")

        roe = None
        try:
            if margin > 0 and pnl is not None:
                roe = (float(pnl) / float(margin)) * 100.0
        except Exception:
            roe = None

        self._append_history({
            "dt": str(pd.to_datetime(candle_time)),
            "event": "EXIT",
            "mode": "DRY_RUN",
            "symbol": sym,
            "side": str(pos.get("side", "")),
            "price": float(exec_px) if exec_px is not None else None,
            "amount": float(pos.get("amount", 0)),
            "fee": float(fee) if fee is not None else None,
            "margin": float(margin),
            "pnl": float(pnl) if pnl is not None else None,
            "roe_pct": float(roe) if roe is not None else None,
            "sl": float(pos.get("sl", 0)) if pos.get("sl") is not None else None,
            "tp1": float(pos.get("tp1", 0)) if pos.get("tp1") is not None else None,
            "reason": f"{reason} track={self.track_id}",
            "pos_count": int(max(len(self.positions) - 1, 0)),
            "cash": float(self.paper_cash),
            "equity": float(self.paper_equity),
        })

        # cooldown update
        self.update_cooldown(sym, exec_px, pos)

        # remove pos
        self.positions.pop(sym, None)
        self._save_state()

    def heartbeat(self, candle_time, candidates, entry, exit_, updsl):
        logger.info(
            f"💓 HEARTBEAT | track={self.track_id} | t={pd.to_datetime(candle_time)} | "
            f"cand={int(candidates)} entry={int(entry)} exit={int(exit_)} updSL={int(updsl)} | "
            f"pos={len(self.positions)} | cash={self.paper_cash:.2f} eq={self.paper_equity:.2f} | "
            f"manage_tf={self.manage_tf}"
        )

        log_hb = bool(self.cfg.get("system_settings", {}).get("log_heartbeat_to_csv", False))
        if log_hb:
            self._append_history({
                "dt": str(pd.to_datetime(candle_time)),
                "event": "HEARTBEAT",
                "mode": "DRY_RUN",
                "symbol": "",
                "side": "",
                "reason": f"cand={int(candidates)} entry={int(entry)} exit={int(exit_)} updSL={int(updsl)} track={self.track_id}",
                "pos_count": int(len(self.positions)),
                "cash": float(self.paper_cash),
                "equity": float(self.paper_equity),
            })


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

        # config
        self.cfg = self._load_config()

        # time authority
        self.tf15_sec = 15 * 60
        self.tf1_sec = 60
        self.loop_buffer_sec = float(self.cfg.get("system_settings", {}).get("loop_buffer_sec", 3.0))

        # DRY_RUN flag
        self.dry_run = bool(self.cfg.get("system_settings", {}).get("dry_run", False))

        # core components
        self.titan = TitanStrategy()
        self.executor = BinanceExecutor(self.cfg)  # 시장데이터/정밀도 그대로 사용
        self.risk_ctrl = RiskControl(self.executor, self.cfg)
        self.monitor = PositionMonitor()
        self.data_loader = DataLoader()

        # blacklist
        strat_settings = self.cfg.get("strategy_settings", {})
        if "blacklist" in strat_settings:
            self.titan.blacklist = set(strat_settings["blacklist"])
        logger.info(f"🚫 Blacklist loaded: {self.titan.blacklist}")

        # data caches (15m)
        self.raw_data_map = {}
        self.data_map = {}
        self.symbols = []

        # LIVE only
        self.freeze_new_entries = False
        self.last_processed_time = None
        self.last_bucket = None

        # DRY_RUN tracks
        self.tracks = {}
        if self.dry_run:
            self._init_dry_run_tracks()
            logger.info("🟦 DRY_RUN enabled: reconcile_positions skipped (no entry freeze).")
        else:
            self.reconcile_positions()

        logger.info(f"🟦 MODE: {'DRY_RUN' if self.dry_run else 'LIVE'}")

    def _init_dry_run_tracks(self):
        tracks_cfg = self.cfg.get("dry_run_tracks", {}) or {}
        if not tracks_cfg:
            # fallback: single track as A (15m/15m)
            tracks_cfg = {
                "A": {"name": "baseline_15m", "entry_timeframe": "15m", "management_timeframe": "15m"}
            }

        for tid, tc in tracks_cfg.items():
            entry_tf = tc.get("entry_timeframe", "15m")
            manage_tf = tc.get("management_timeframe", "15m")
            self.tracks[str(tid)] = DryRunTrack(
                track_id=str(tid),
                entry_tf=str(entry_tf),
                manage_tf=str(manage_tf),
                cfg=self.cfg,
                executor=self.executor
            )

        logger.info("🧪 DRY_RUN tracks loaded: " + ", ".join(
            [f"{t.track_id}(entry={t.entry_tf},manage={t.manage_tf})" for t in self.tracks.values()]
        ))

    # -----------------------------------------------------
    # LIVE State I/O (unchanged behavior)
    # -----------------------------------------------------
    def _live_state_file(self):
        return os.path.join(root_dir, "phalanx_state.json")

    def _load_state(self):
        STATE_FILE = self._live_state_file()
        if not os.path.exists(STATE_FILE):
            self.executor.positions = {}
            self.last_processed_time = None
            self.last_bucket = None
            return

        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
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
        STATE_FILE = self._live_state_file()
        state = {
            "positions": self.executor.positions,
            "last_processed_time": self.last_processed_time,
            "last_bucket": self.last_bucket,
        }
        try:
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"State save failed: {e}")

    # -----------------------------------------------------
    # Restart Safety: account vs local state (LIVE only)
    # -----------------------------------------------------
    def reconcile_positions(self, amount_tol=0.02):
        if not hasattr(self.executor, "fetch_positions"):
            self.freeze_new_entries = False
            return

        try:
            real = self.executor.fetch_positions() or {}
        except Exception as e:
            logger.error(f"fetch_positions failed: {e}")
            real = {}

        local = self.executor.positions or {}

        def _u(x):
            return str(x).upper()

        mismatch = False

        for sym, rp in real.items():
            if sym not in local:
                mismatch = True
                logger.error(f"🚨 RECONCILE mismatch: real has {sym} but local missing")
                continue

            lp = local.get(sym, {})
            if _u(rp.get("side")) != _u(lp.get("side")):
                mismatch = True
                logger.error(f"🚨 RECONCILE mismatch: {sym} side real={rp.get('side')} local={lp.get('side')}")
                continue

            try:
                ra = float(rp.get("amount", 0))
                la = float(lp.get("amount", 0))
                if ra <= 0 or la <= 0:
                    mismatch = True
                    logger.error(f"🚨 RECONCILE mismatch: {sym} amount invalid real={ra} local={la}")
                else:
                    if abs(ra - la) / max(ra, 1e-12) > amount_tol:
                        mismatch = True
                        logger.error(f"🚨 RECONCILE mismatch: {sym} amount real={ra} local={la}")
            except Exception:
                mismatch = True
                logger.error(f"🚨 RECONCILE mismatch: {sym} amount parse failed")

        for sym in local.keys():
            if sym not in real:
                mismatch = True
                logger.error(f"🚨 RECONCILE mismatch: local has {sym} but real missing")
                break

        if mismatch:
            self.freeze_new_entries = True
            logger.critical("🚨 POSITION MISMATCH DETECTED -> ENTRY FREEZE (no new entries)")
        else:
            self.freeze_new_entries = False
            logger.info("✅ Position reconcile OK (state matches account)")

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

    def _current_bucket_1m(self):
        return int(time.time() // self.tf1_sec)

    def _sleep_until_next_15m(self):
        now = time.time()
        next_tick = (math.floor(now / self.tf15_sec) * self.tf15_sec) + self.tf15_sec
        sleep_sec = (next_tick - now) + self.loop_buffer_sec
        if sleep_sec < 1:
            sleep_sec += self.tf15_sec
        logger.info(f"⏳ Waiting {sleep_sec:.2f}s for next 15m close...")
        time.sleep(sleep_sec)

    def _sleep_until_next_1m(self):
        now = time.time()
        next_tick = (math.floor(now / self.tf1_sec) * self.tf1_sec) + self.tf1_sec
        sleep_sec = (next_tick - now) + max(0.2, min(self.loop_buffer_sec, 2.0))
        if sleep_sec < 0.2:
            sleep_sec += self.tf1_sec
        time.sleep(sleep_sec)

    # -----------------------------------------------------
    # Candle Picking (15m stale blocking fix)
    # -----------------------------------------------------
    def _pick_current_time_15m(self, stale_lag_sec=3600):
        last_times = {}
        for sym in self.symbols:
            df = self.data_map.get(sym)
            if df is None or df.empty:
                continue
            if len(df) < 2:
                continue
            last_times[sym] = df.index[-1]

        if not last_times:
            return None

        ref = max(last_times.values())
        good_times = []
        for sym, t in last_times.items():
            lag = (ref - t).total_seconds()
            if lag > stale_lag_sec:
                logger.warning(f"⚠️ Stale symbol filtered: {sym} last={t} lag={lag:.0f}s")
                continue
            good_times.append(t)

        if not good_times:
            return None

        return ref

    # -----------------------------------------------------
    # 1m fetch helper (B manage)
    # -----------------------------------------------------
    def _fetch_latest_1m_rows(self, symbols):
        """
        symbols의 최신 1m 캔들(1개)을 가져옴.
        ⚠️ 관리용 스냅샷이므로 validate_and_format 사용 금지
        결과: {sym: pd.Series(row)} where row.name = datetime
        """
        out = {}
        for sym in symbols:
            try:
                ohlcv = self.executor.exchange.fetch_ohlcv(
                    sym, timeframe="1m", limit=2
                )
                if not ohlcv:
                    continue

                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )

                # 최소 안전성 체크만 수행
                if df.empty or len(df) < 1:
                    continue

                if df[["open", "high", "low", "close"]].isna().any().any():
                    continue

                # 시간 인덱스 구성
                df["datetime"] = pd.to_datetime(
                    df["timestamp"], unit="ms", errors="coerce"
                )
                df = df.dropna(subset=["datetime"]).set_index("datetime")
                if df.empty:
                    continue

                # 가장 최근 캔들
                row = df.iloc[-1]
                row.name = df.index[-1]
                out[sym] = row

            except Exception:
                continue

        return out

    def _nearest_15m_indicators(self, sym, t):
        """
        1m 관리에서 ATR/ST를 15m 기준으로 가져옴(가장 최근 <= t).
        """
        df15 = self.data_map.get(sym)
        if df15 is None or df15.empty:
            return None
        try:
            # pick last index <= t
            idx = df15.index.searchsorted(pd.to_datetime(t), side="right") - 1
            if idx < 0:
                return None
            return df15.iloc[int(idx)]
        except Exception:
            return None

    # -----------------------------------------------------
    # DRY_RUN: manage loop (common)
    # -----------------------------------------------------
    def _track_manage_with_row(self, track: DryRunTrack, sym: str, candle_row: pd.Series, candle_time, ind15_row=None):
        """
        candle_row: (1m or 15m) with open/high/low/close/volume
        ind15_row: optional 15m indicator row for atr/st_val
        """
        if sym not in track.positions:
            return "NONE"

        track.apply_next_sl_if_any(sym)

        close_px = float(candle_row["close"])
        high_px = float(candle_row["high"])
        low_px = float(candle_row["low"])

        atr = None
        st_val = None
        try:
            if ind15_row is not None:
                atr = float(ind15_row.get("atr", close_px * 0.01))
                st_val = float(ind15_row.get("st_val", 0))
            else:
                atr = float(candle_row.get("atr", close_px * 0.01))
                st_val = float(candle_row.get("st_val", 0))
        except Exception:
            atr = close_px * 0.01
            st_val = 0.0

        market_data = {
            "close": float(close_px),
            "high": float(high_px),
            "low": float(low_px),
            "atr": float(atr) if atr is not None else float(close_px * 0.01),
            "st_val": float(st_val) if st_val is not None else 0.0,
        }

        pos = track.positions[sym]
        action, exec_price, reason, new_sl = self.monitor.check_conditions(sym, pos, market_data)

        if action == "UPDATE_SL":
            track.handle_update_sl(sym, candle_time, close_px, reason, new_sl)
            return "UPDATE_SL"

        if action == "EXIT":
            track.handle_exit(sym, candle_time, exec_price, reason)
            return "EXIT"

        return "HOLD"

    # -----------------------------------------------------
    # DRY_RUN: entry computation (15m only, single compute)
    # -----------------------------------------------------
    def _compute_candidates_15m(self, current_time_15m):
        """
        ENTRY 로직은 절대 분기하지 않음.
        - 여기서 signal/sl/tp/score를 1회만 계산해서 리턴
        """
        logger.info(f"🔍 ENTRY_SCAN_START | t={current_time_15m}")

        candidates = []

        if not self.data_map:
            return candidates

        for sym in self.symbols:
            df = self.data_map.get(sym)
            if df is None or df.empty:
                continue
            if current_time_15m not in df.index:
                continue

            curr_row = df.loc[current_time_15m]
            idx = df.index.get_loc(current_time_15m)
            start = max(0, idx - 250)
            past_data = df.iloc[start: idx + 1]



            # DEBUG: 입력 시점/지표 상태 확인
            last_ts = past_data.index[-1]
            req = ["adx", "st_dir", "st_val", "ema_daily", "ema_intra", "vol_ma", "rsi"]
            snap = {c: (None if c not in past_data.columns else float(past_data[c].iloc[-1])) for c in req}
            logger.info(f"🧪 PRE_ANALYZE | {sym} t={current_time_15m} last={last_ts} rows={len(past_data)} snap={snap}")



            signal, sl, tp = self.titan.analyze(sym, past_data)
            # ================= DEBUG: ENTRY DECISION TRACE =================
            curr = past_data.iloc[-1]
            prev = past_data.iloc[-2] if len(past_data) >= 2 else curr

            debug_struct = {
                "retest_long": int(curr.get("retest_long", 0)),
                "retest_short": int(curr.get("retest_short", 0)),
                "mss_up": int(curr.get("mss_up", 0)),
                "mss_down": int(curr.get("mss_down", 0)),
                "recent_sweep_low": int(curr.get("recent_sweep_low", 0)),
                "recent_sweep_high": int(curr.get("recent_sweep_high", 0)),
                "last_pivot_high": float(curr.get("last_pivot_high", 0) or 0),
                "last_pivot_low": float(curr.get("last_pivot_low", 0) or 0),
            }

            debug_filters = {
                "ema_daily_ok": int(curr.get("ema_daily_ok", 0)),
                "ema_daily": float(curr.get("ema_daily", 0) or 0),
                "vol": float(prev.get("volume", 0) or 0),
                "vol_ma": float(curr.get("vol_ma", 0) or 0),
                "vol_factor": float(self.titan.params.get("vol_factor", 1.0)),
                "adx": float(curr.get("adx", 0) or 0),
                "adx_th": float(self.titan.params.get("adx_threshold", 0)),
            }

            logger.info(
                f"🧪 ENTRY_DEBUG | {sym} | "
                f"signal={signal} | "
                f"struct={debug_struct} | "
                f"filters={debug_filters}"
            )
            # ===============================================================



            if not signal:
                logger.info(f"⚪ NO_ENTRY | {sym}")
            if signal:
                score = float(curr_row.get("adx", 0))
                candidates.append({
                    "score": score,
                    "sym": sym,
                    "signal": signal,
                    "sl": sl,
                    "tp": tp,
                    "row": curr_row,  # 15m row
                })
                logger.info(
                f"🟢 ENTRY_CANDIDATE | {sym} signal={signal} adx={score:.2f} sl={sl} tp={tp}"
                )

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates

    # -----------------------------------------------------
    # DRY_RUN main tick (1m loop)
    # -----------------------------------------------------
    def run_tick_dry_run(self):
        """
        1분마다 호출.
        - B(1m manage) 관리
        - 15m boundary면: prepare_data + A(15m manage) + ENTRY(1회 계산 후 A/B 복제)
        """
        # -------------------------
        # Step 0) B manage (1m)
        # -------------------------
        # B 트랙(=manage_tf == "1m")의 포지션 심볼만 1m 캔들 가져옴
        b_tracks = [t for t in self.tracks.values() if str(t.manage_tf) == "1m"]
        b_syms = set()
        for t in b_tracks:
            for s in (t.positions or {}).keys():
                b_syms.add(s)

        exit_count_1m = {t.track_id: 0 for t in b_tracks}
        updsl_count_1m = {t.track_id: 0 for t in b_tracks}

        if b_tracks and b_syms:
            rows_1m = self._fetch_latest_1m_rows(sorted(b_syms))
            # stale blocking (relative lag)
            if rows_1m:
                last_times = {s: rows_1m[s].name for s in rows_1m.keys()}
                ref = max(last_times.values())
                # allow 10 minutes lag by default for 1m
                good_syms = []
                for s, ts in last_times.items():
                    lag = (pd.to_datetime(ref) - pd.to_datetime(ts)).total_seconds()
                    if lag > 600:
                        logger.warning(f"⚠️ Stale 1m filtered: {s} last={ts} lag={lag:.0f}s")
                        continue
                    good_syms.append(s)

                for t in b_tracks:
                    for sym in good_syms:
                        if sym not in t.positions:
                            continue
                        row1m = rows_1m.get(sym)
                        if row1m is None:
                            continue
                        ind15 = self._nearest_15m_indicators(sym, row1m.name)
                        act = self._track_manage_with_row(
                            track=t,
                            sym=sym,
                            candle_row=row1m,
                            candle_time=row1m.name,
                            ind15_row=ind15
                        )
                        if act == "EXIT":
                            exit_count_1m[t.track_id] += 1
                        elif act == "UPDATE_SL":
                            updsl_count_1m[t.track_id] += 1

                # update equity for those tracks using 1m close prices
                px_map = {}
                for s in good_syms:
                    try:
                        px_map[s] = float(rows_1m[s]["close"])
                    except Exception:
                        pass
                for t in b_tracks:
                    t.mtm_update_equity(px_map)
                    t.last_processed_manage_time = str(pd.to_datetime(ref))
                    t._save_state()

        # -------------------------
        # Step 1) 15m boundary work
        # -------------------------
        entry_bucket = self._current_bucket_15m()
        any_track = next(iter(self.tracks.values()), None)
        if any_track is None:
            return

        if any_track.last_entry_bucket is None or int(entry_bucket) != int(any_track.last_entry_bucket):
            # 15m 새 캔들: 데이터 갱신(15m) + A manage + ENTRY
            try:
                self.prepare_data()
            except Exception as e:
                logger.error(f"prepare_data failed: {e}")
                # 실패 시 bucket 갱신하지 않음(재시도)
                return

            current_time_15m = self._pick_current_time_15m()
            if current_time_15m is None:
                return

            # A manage (15m)
            a_tracks = [t for t in self.tracks.values() if str(t.manage_tf) == "15m"]
            exit_count_15m = {t.track_id: 0 for t in a_tracks}
            updsl_count_15m = {t.track_id: 0 for t in a_tracks}

            # 공통 current_prices (15m close)
            current_prices_15m = {}
            for sym in self.symbols:
                df = self.data_map.get(sym)
                if df is None or df.empty:
                    continue
                if current_time_15m not in df.index:
                    continue
                row15 = df.loc[current_time_15m]
                current_prices_15m[sym] = float(row15["close"])

            for t in a_tracks:
                for sym in list(t.positions.keys()):
                    df = self.data_map.get(sym)
                    if df is None or df.empty:
                        continue
                    if current_time_15m not in df.index:
                        continue
                    row15 = df.loc[current_time_15m]
                    act = self._track_manage_with_row(
                        track=t,
                        sym=sym,
                        candle_row=row15,
                        candle_time=current_time_15m,
                        ind15_row=row15
                    )
                    if act == "EXIT":
                        exit_count_15m[t.track_id] += 1
                    elif act == "UPDATE_SL":
                        updsl_count_15m[t.track_id] += 1

                t.mtm_update_equity(current_prices_15m)
                t.last_processed_entry_time = str(pd.to_datetime(current_time_15m))
                t._save_state()

            # ENTRY (single compute) -> replicate to all tracks
            candidates = self._compute_candidates_15m(current_time_15m)

            entry_count = {t.track_id: 0 for t in self.tracks.values()}

            # fill_px 기준 통일: 15m close 사용(복제 정합성 최우선)
            for cand in candidates:
                sym = cand["sym"]
                signal = cand["signal"]
                sl = cand["sl"]
                tp = cand["tp"]
                row15 = cand["row"]
                fill_px = float(row15["close"])

                # 수량도 1회만 계산(복제 정합성)
                # 기준 equity는 "진입 시점 현금"이 트랙별로 다르므로,
                # amount 계산은 'paper_equity0' 기반이면 트랙 간 동일하되,
                # cash 부족은 각 트랙에서 reject로 남김.
                # -> 리스크 고정(실험 통제) 목적.
                base_equity = float(self.cfg.get("system_settings", {}).get("paper_equity", 10000.0))
                try:
                    amount = self.risk_ctrl.calculate_entry_size(sym, float(fill_px), float(base_equity), sl, signal)
                except Exception:
                    amount = 0.0

                if amount <= 0:
                    for t in self.tracks.values():
                        t._append_history({
                            "dt": str(pd.to_datetime(current_time_15m)),
                            "event": "ENTRY_REJECT",
                            "mode": "DRY_RUN",
                            "symbol": sym,
                            "side": signal,
                            "price": float(fill_px),
                            "amount": float(amount),
                            "reason": f"amount<=0 track={t.track_id}",
                            "pos_count": int(len(t.positions)),
                            "cash": float(t.paper_cash),
                            "equity": float(t.paper_equity),
                            "sl": float(sl) if sl is not None else None,
                            "tp1": float(tp) if tp is not None else None,
                        })
                    continue

                # replicate open to each track
                for t in self.tracks.values():
                    ok = t.open_position(
                        
                        sym=sym,
                        signal=signal,
                        fill_px=float(fill_px),
                        amount_raw=float(amount),
                        sl=sl,
                        tp=tp,
                        entry_time=current_time_15m
                    )
                    if ok:
                        entry_count[t.track_id] += 1
                        logger.info(f"✅ ENTRY_ACCEPT | track={t.track_id} {sym}")
                    else:
                        logger.info(f"❌ ENTRY_REJECT | track={t.track_id} {sym}")


            # finalize: set entry bucket processed for all tracks
            for t in self.tracks.values():
                t.last_entry_bucket = int(entry_bucket)
                t._save_state()

            # HEARTBEAT (15m event summary per track)
            for t in self.tracks.values():
                # counts: prefer 15m counts for 15m-manage tracks, 1m counts for 1m-manage tracks
                if str(t.manage_tf) == "15m":
                    ex = exit_count_15m.get(t.track_id, 0)
                    up = updsl_count_15m.get(t.track_id, 0)
                else:
                    ex = exit_count_1m.get(t.track_id, 0)
                    up = updsl_count_1m.get(t.track_id, 0)
                t.heartbeat(
                    candle_time=current_time_15m,
                    candidates=len(candidates),
                    entry=entry_count.get(t.track_id, 0),
                    exit_=ex,
                    updsl=up
                )

        else:
            # non-15m minute: optional heartbeat for 1m tracks only (off by default)
            pass

    # -----------------------------------------------------
    # LIVE main loop (15m) - keep original semantics
    # -----------------------------------------------------
    def run_once_live(self):
        # state load once at start if needed
        if self.last_bucket is None and self.last_processed_time is None:
            self._load_state()

        b = self._current_bucket_15m()
        if self.last_bucket is None or b != self.last_bucket:
            self.prepare_data()
            self.last_bucket = b

        if not self.data_map:
            return

        current_time = self._pick_current_time_15m()
        if current_time is None:
            return

        if self.last_processed_time is not None:
            try:
                if pd.to_datetime(current_time) <= pd.to_datetime(self.last_processed_time):
                    return
            except Exception:
                pass

        # manage
        current_prices = {}
        for sym in self.symbols:
            df = self.data_map[sym]
            if current_time not in df.index:
                continue
            curr_row = df.loc[current_time]
            current_prices[sym] = float(curr_row["close"])

            if sym in self.executor.positions:
                pos = self.executor.positions[sym]
                if "next_sl" in pos and pos["next_sl"] is not None:
                    try:
                        if float(pos["next_sl"]) != float(pos.get("sl", 0)):
                            pos["sl"] = float(pos["next_sl"])
                    except Exception:
                        pass
                    pos.pop("next_sl", None)

                self._process_existing_position_live(sym, curr_row)

        # entry
        if not self.freeze_new_entries:
            candidates = self._compute_candidates_15m(current_time)
            for cand in candidates:
                if len(self.executor.positions) >= self.executor.MAX_POSITIONS:
                    break
                self._process_entry_live(cand)

        # equity update
        self.executor.update_equity(current_prices)

        self.last_processed_time = str(pd.to_datetime(current_time))
        self._save_state()

    # -----------------------------------------------------
    # LIVE entry/exit helpers (minimal, reuse old logic shape)
    # -----------------------------------------------------
    def _process_entry_live(self, cand):
        sym = cand["sym"]
        row = cand["row"]
        signal = cand["signal"]
        sl = cand["sl"]
        tp = cand["tp"]

        price = float(row["close"])
        equity = float(self.executor.get_available_equity())

        amount = self.risk_ctrl.calculate_entry_size(sym, price, equity, sl, signal)
        if amount <= 0:
            logger.info(f"⚪ ENTRY_REJECT | {sym} {signal} price={price:.6f} sl={sl} equity={equity:.2f} -> amount={amount}")
            return False

        result = self.executor.create_order(sym, signal, amount)
        if not result:
            logger.error(f"❌ ENTRY FAIL | {sym} {signal} amt={amount}")
            return False

        entry_price = float(result["filled_price"])
        filled_qty = float(result.get("filled_qty", amount))
        margin = float(result.get("margin", 0))

        self.executor.positions[sym] = {
            "side": signal,
            "amount": filled_qty,
            "entry_price": entry_price,
            "sl": sl,
            "tp1": tp,
            "next_sl": None,
            "tp1_hit": False,
            "entry_time": str(row.name),
            "margin": margin,
        }

        logger.info(
            f"🟢 ENTRY | {sym} {signal} qty={filled_qty} px={entry_price} "
            f"sl={sl} tp={tp}"
        )
        return True

    def _process_existing_position_live(self, sym, curr_row):
        pos = self.executor.positions[sym]

        market_data = {
            "close": float(curr_row["close"]),
            "high": float(curr_row["high"]),
            "low": float(curr_row["low"]),
            "atr": float(curr_row.get("atr", curr_row["close"] * 0.01)),
            "st_val": float(curr_row.get("st_val", 0)),
        }

        action, exec_price, reason, new_sl = self.monitor.check_conditions(sym, pos, market_data)

        if action == "UPDATE_SL":
            try:
                if new_sl is not None and float(new_sl) != float(pos.get("sl", 0)):
                    pos["next_sl"] = float(new_sl)
                    logger.info(f"🟡 UPDATE_SL | {sym} next_sl={pos['next_sl']} reason={reason}")
            except Exception:
                pass
            return "UPDATE_SL"

        if action == "EXIT":
            logger.info(f"🔴 EXIT | {sym} px={exec_price} reason={reason}")
            self.executor.close_position(sym, exec_price, reason)
            self.executor.positions.pop(sym, None)
            return "EXIT"

        return "HOLD"


if __name__ == "__main__":
    engine = LiveEngine()

    # 첫 준비 (DRY_RUN은 15m boundary에서 prepare_data 호출하지만, 시작 시 한 번 미리 준비해도 안전)
    try:
        engine.prepare_data()
    except Exception as e:
        logger.error(f"Startup prepare_data failed: {e}")

    if engine.dry_run:
        # 1분 루프 (B manage 가능)
        # 시작 즉시 다음 1분 경계로 정렬
        engine._sleep_until_next_1m()
        while True:
            try:
                engine.run_tick_dry_run()
                engine._sleep_until_next_1m()
            except KeyboardInterrupt:
                logger.info("🛑 Manual stop")
                break
            except Exception as e:
                logger.error(f"Live Engine Error: {e}")
                time.sleep(3)
    else:
        # LIVE는 기존대로 15m 루프
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
