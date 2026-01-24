import optuna
import pandas as pd
import numpy as np
import os
import time
import pickle
import traceback
from core.backtest_engine import BacktestEngine

# ✅ Universe 단일화
from utils.universe import get_universe, save_universe_snapshot

# 1. Optuna 로그 레벨 조정
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 2. 파일명 및 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_FILE = os.path.join(CURRENT_DIR, "optimization_results.csv")

# ✅ 30일 운영 최적화 전용 캐시로 분리 (180d 캐시 오염 방지)
DATA_CACHE_FILE = os.path.join(CURRENT_DIR, "market_data_cache_30d.pkl")

UNIVERSE_SNAPSHOT_FILE = os.path.join(CURRENT_DIR, "universe_optimization.json")


# ==============================================================================
# [Core] 결과 계산
# ==============================================================================
def calculate_metrics(initial_balance: float, history):
    if not history:
        return None

    df = pd.DataFrame(history)
    if df.empty or "pnl" not in df.columns:
        return None

    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]

    cnt_win = len(wins)
    cnt_loss = len(losses)
    total_trades = cnt_win + cnt_loss
    if total_trades == 0:
        return None

    total_pnl = float(df["pnl"].sum())
    final_equity = float(initial_balance + total_pnl)
    return_pct = (final_equity - initial_balance) / initial_balance * 100.0

    avg_win = float(wins["pnl"].mean()) if cnt_win > 0 else 0.0
    avg_loss = float(losses["pnl"].mean()) if cnt_loss > 0 else 0.0
    win_rate = (cnt_win / total_trades) * 100.0

    df["cumulative_pnl"] = df["pnl"].cumsum()
    df["equity_curve"] = float(initial_balance) + df["cumulative_pnl"]
    peak = df["equity_curve"].cummax()
    drawdown = (df["equity_curve"] - peak) / peak * 100.0
    mdd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    gross_profit = float(wins["pnl"].sum()) if cnt_win > 0 else 0.0
    gross_loss = float(abs(losses["pnl"].sum())) if cnt_loss > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

    return {
        "Final Equity": final_equity,
        "Total Return": float(return_pct),
        "MDD": float(mdd),
        "Profit Factor": float(profit_factor),
        "Win Rate": float(win_rate),
        "Win Count": int(cnt_win),
        "Loss Count": int(cnt_loss),
        "Avg Win": float(avg_win),
        "Avg Loss": float(avg_loss),
        "Total Trades": int(total_trades),
    }


# ==============================================================================
# [Objective] 최적화 목표 함수
# ==============================================================================
def objective(trial, engine: BacktestEngine):
    params = {
        "atr_period": trial.suggest_int("atr_period", 10, 30),
        "atr_multiplier": trial.suggest_float("atr_multiplier", 2.0, 6.0, step=0.5),
        "adx_threshold": trial.suggest_int("adx_threshold", 15, 35),
        "rsi_upper": trial.suggest_int("rsi_upper", 60, 80),
        "rsi_lower": trial.suggest_int("rsi_lower", 20, 40),
        "vol_factor": trial.suggest_float("vol_factor", 0.8, 2.0, step=0.1),
        "ema_intraday": 200,
        # ✅ 30일 운영 최적화: daily_ema는 반드시 짧게
        "daily_ema": trial.suggest_int("daily_ema", 5, 20, step=5),
    }

    # 1) params 주입
    engine.titan.set_params(params)

    # 2) trial마다 지표 재생성 → 백테스트 실행
    try:
        engine.rebuild_indicators()
        engine.run(show_report=False)
    except Exception:
        # 디버그에 도움되게(옵션): 필요하면 출력
        # print(traceback.format_exc())
        return -99999

    initial_balance = getattr(engine.executor, "initial_balance", 10000.0)
    history = getattr(engine.executor, "history", [])
    metrics = calculate_metrics(initial_balance, history)

    if metrics:
        record = {**params, **metrics}
        record["trial_id"] = trial.number
        df_record = pd.DataFrame([record])

        if not os.path.exists(RESULT_FILE):
            df_record.to_csv(RESULT_FILE, index=False, mode="w", encoding="utf-8-sig")
        else:
            df_record.to_csv(
                RESULT_FILE,
                index=False,
                mode="a",
                header=False,
                encoding="utf-8-sig",
            )

        # Final Equity 최대화
        return metrics["Final Equity"]
    else:
        # 거래가 없거나 계산 불가 → 원금 반환(패널티 성격)
        return float(initial_balance)


# ==============================================================================
# [Main]
# ==============================================================================
if __name__ == "__main__":
    # 결과 파일 초기화
    if os.path.exists(RESULT_FILE):
        try:
            os.remove(RESULT_FILE)
        except:
            pass

    print("🚀 Hyperparameter Optimization Start (Mode: Safe/Serial)...")
    print(f"📂 작업 경로: {CURRENT_DIR}")
    print(f"📂 RAW 캐시 저장 예정 경로: {DATA_CACHE_FILE}")

    # 1) 엔진 생성 (✅ 30일 기준)
    engine = BacktestEngine(days=30)

    # 2) 캐시 로드(있으면)
    cached_raw = {}
    if os.path.exists(DATA_CACHE_FILE):
        print(f"✅ 기존 RAW 데이터 캐시 발견: {DATA_CACHE_FILE}")
        print("   파일을 로드합니다...")
        try:
            with open(DATA_CACHE_FILE, "rb") as f:
                cached_raw = pickle.load(f) or {}
            print(f"   캐시 로드 완료: {len(cached_raw)}개 종목 (RAW)")
        except Exception as e:
            print(f"❌ 캐시 로드 실패: {e}")
            cached_raw = {}

    # 3) ✅ Universe 선정: 14개(코어8 + 위성6)
    print(f"🚫 [System] Configured Blacklist: {engine.titan.blacklist}")

    universe_symbols = get_universe(
        executor=engine.executor,
        top_n=14,
        # ✅ 30일 운영형 최소 데이터 확보 (30일*96=2880 → 여유 포함 3000)
        validate_ohlcv=True,
        timeframe="15m",
        min_ohlcv_rows=3000,
        # ✅ 너무 촘촘하게 잡히면 위성이 부족할 수 있으니 기본은 50M 유지 (필요하면 20M으로 낮춰)
        min_quote_volume=50_000_000.0,
    )

    print(f"🧭 [Universe] Selected ({len(universe_symbols)}): {universe_symbols}")

    # 스냅샷 저장(Optimize → Backtest 동일 Universe 재현용)
    try:
        save_universe_snapshot(
            UNIVERSE_SNAPSHOT_FILE,
            universe_symbols,
            meta={
                "policy": "core8+sat6",
                "days": engine.test_days,
                "ts": int(time.time()),
            },
        )
        print(f"✅ Universe Snapshot 저장 완료: {UNIVERSE_SNAPSHOT_FILE}")
    except Exception as e:
        print(f"⚠️ Universe Snapshot 저장 실패: {e}")

    # 4) 캐시 merge + 부족분 다운로드
    engine.symbols = universe_symbols
    engine.raw_data_map = {}

    missing = []
    for sym in universe_symbols:
        if sym in cached_raw:
            engine.raw_data_map[sym] = cached_raw[sym]
        else:
            missing.append(sym)

    if missing:
        print(f"⏳ 캐시에 없는 심볼 {len(missing)}개 추가 다운로드 시작...")
        try:
            new_map = engine.executor.prepare_data(missing)  # VirtualExecutor.prepare_data(list)
            if new_map:
                engine.raw_data_map.update(new_map)
        except Exception as e:
            print(f"❌ 추가 다운로드 실패: {e}")

    print(
        f"📊 최종 RAW 데이터: {len(engine.raw_data_map)}개 종목 (Universe 목표: {len(universe_symbols)}개)"
    )

    # 5) 캐시 저장(갱신)
    if len(engine.raw_data_map) > 0:
        try:
            with open(DATA_CACHE_FILE, "wb") as f:
                pickle.dump(engine.raw_data_map, f)
            print(f"✅ RAW 데이터 캐시 저장 완료: {DATA_CACHE_FILE}")
        except Exception as e:
            print(f"❌ RAW 데이터 저장 실패: {e}")

    # 6) 최적화 시작
    print(f"✅ 최적화 루프 시작 (Universe Symbols: {len(universe_symbols)})")
    print(f"📄 결과 저장 경로: {RESULT_FILE}\n")

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, engine), n_trials=2000, n_jobs=1)

    print("\n" + "=" * 50)
    print(f"🎉 Optimization Finished! (Total Trials: {len(study.trials)})")
    print("=" * 50)
    try:
        print(f"🏆 Best Equity : ${study.best_value:,.2f}")
        print("-" * 50)
        print("🌟 Best Parameters:")
        for key, value in study.best_params.items():
            print(f"   - {key:<15}: {value}")
    except:
        print("⚠️ 유효한 결과가 없습니다.")
    print("=" * 50)
