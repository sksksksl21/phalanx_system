import optuna
import pandas as pd
import numpy as np
import os
import time
import ccxt
import json
import pickle
from core.backtest_engine import BacktestEngine
import logging

# 1. Optuna 로그 레벨 조정
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 2. 파일명 및 경로 설정 (절대 경로로 강제 지정)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_FILE = os.path.join(CURRENT_DIR, "optimization_results.csv")
DATA_CACHE_FILE = os.path.join(CURRENT_DIR, "market_data_cache.pkl")  # [CHANGED] raw_data_map 캐시

# ==============================================================================
# [Helper] 바이낸스 선물 거래량 상위 종목 긁어오기
# ==============================================================================
def fetch_target_symbols(limit=30):
    print(f"\n🔍 [Market Scan] 바이낸스 선물 거래량 상위 {limit}개 종목 스캔 중...")

    blacklist = set()
    try:
        config_path = os.path.join(CURRENT_DIR, "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            blacklist_list = cfg.get('strategy_settings', {}).get('blacklist', [])
            blacklist = set(blacklist_list)
    except Exception as e:
        print(f"⚠️ Config 로드 실패: {e}")

    print(f"🚫 [Config] Blacklist Loaded ({len(blacklist)}): {blacklist}")

    try:
        binance = ccxt.binance({'options': {'defaultType': 'future'}})
        tickers = binance.fetch_tickers()

        valid_tickers = []
        for symbol, data in tickers.items():
            if '/USDT' in symbol and data.get('quoteVolume'):
                valid_tickers.append((symbol, float(data['quoteVolume'])))

        sorted_tickers = sorted(valid_tickers, key=lambda x: x[1], reverse=True)

        final_symbols = []
        for sym, vol in sorted_tickers:
            clean_sym = sym.split(':')[0]
            if clean_sym not in blacklist and sym not in blacklist:
                final_symbols.append(sym)
                if len(final_symbols) >= limit:
                    break

        print(f"📊 Top Target Symbols ({len(final_symbols)}): {final_symbols}")
        return final_symbols

    except Exception as e:
        print(f"❌ [Scan Error] 종목 스캔 실패: {e}")
        return ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'SOL/USDT', 'BNB/USDT']

# ==============================================================================
# [Core] 결과 계산
# ==============================================================================
def calculate_metrics(initial_balance, history):
    if not history:
        return None

    df = pd.DataFrame(history)
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] <= 0]

    cnt_win = len(wins)
    cnt_loss = len(losses)
    total_trades = cnt_win + cnt_loss
    if total_trades == 0:
        return None

    total_pnl = df['pnl'].sum()
    final_equity = initial_balance + total_pnl
    return_pct = (final_equity - initial_balance) / initial_balance * 100

    avg_win = wins['pnl'].mean() if cnt_win > 0 else 0
    avg_loss = losses['pnl'].mean() if cnt_loss > 0 else 0
    win_rate = (cnt_win / total_trades) * 100

    df['cumulative_pnl'] = df['pnl'].cumsum()
    df['equity_curve'] = initial_balance + df['cumulative_pnl']
    peak = df['equity_curve'].cummax()
    drawdown = (df['equity_curve'] - peak) / peak * 100
    mdd = drawdown.min()

    return {
        "Final Equity": final_equity,
        "Total Return": return_pct,
        "MDD": mdd,
        "Profit Factor": (wins['pnl'].sum() / abs(losses['pnl'].sum())) if len(losses) > 0 and losses['pnl'].sum() != 0 else 0,
        "Win Rate": win_rate,
        "Win Count": cnt_win,
        "Loss Count": cnt_loss,
        "Avg Win": avg_win,
        "Avg Loss": avg_loss,
        "Total Trades": total_trades,
    }

# ==============================================================================
# [Objective] 최적화 목표 함수
# ==============================================================================
def objective(trial, engine: BacktestEngine):
    params = {
        'atr_period': trial.suggest_int('atr_period', 10, 30),
        'atr_multiplier': trial.suggest_float('atr_multiplier', 2.0, 6.0, step=0.5),
        'adx_threshold': trial.suggest_int('adx_threshold', 15, 35),
        'rsi_upper': trial.suggest_int('rsi_upper', 60, 80),
        'rsi_lower': trial.suggest_int('rsi_lower', 20, 40),
        'vol_factor': trial.suggest_float('vol_factor', 0.8, 2.0, step=0.1),
        'ema_intraday': 200,
        'daily_ema': trial.suggest_int('daily_ema', 5, 200, step=5),
    }

    # 1) params 주입
    engine.titan.set_params(params)

    # 2) [CRITICAL] trial마다 지표 재주입 (raw -> indicators)
    try:
        engine.rebuild_indicators()
        engine.run(show_report=False)
    except Exception:
        return -99999

    metrics = calculate_metrics(10000, engine.executor.history)

    if metrics:
        record = {**params, **metrics}
        record['trial_id'] = trial.number
        df_record = pd.DataFrame([record])

        if not os.path.exists(RESULT_FILE):
            df_record.to_csv(RESULT_FILE, index=False, mode='w', encoding='utf-8-sig')
        else:
            df_record.to_csv(RESULT_FILE, index=False, mode='a', header=False, encoding='utf-8-sig')

        return metrics['Final Equity']
    else:
        return 10000

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

    # 1. 엔진 생성
    engine = BacktestEngine(days=180)

    # 2. RAW 데이터 로드 or 다운로드
    if os.path.exists(DATA_CACHE_FILE):
        print(f"✅ 기존 RAW 데이터 캐시 발견: {DATA_CACHE_FILE}")
        print("   파일을 로드합니다... (다운로드 생략)")
        try:
            with open(DATA_CACHE_FILE, "rb") as f:
                engine.raw_data_map = pickle.load(f)
                engine.symbols = list(engine.raw_data_map.keys())
                print(f"   로드 완료: {len(engine.symbols)}개 종목 (RAW)")
        except Exception as e:
            print(f"❌ 캐시 로드 실패 (파일 깨짐 등): {e}")
            print("   새로 다운로드를 시작합니다...")
            if os.path.exists(DATA_CACHE_FILE):
                os.remove(DATA_CACHE_FILE)
            engine.raw_data_map = {}

    if not engine.raw_data_map:
        print("⏳ 데이터 다운로드 시작...")
        target_symbols = fetch_target_symbols(limit=30)

        engine.symbols = target_symbols
        engine.prepare_data()  # RAW 다운로드 실행

        raw_count = len(engine.raw_data_map)
        print(f"📊 메모리에 로드된 RAW 데이터: {raw_count}개 종목")

        if raw_count > 0:
            print(f"💾 RAW 데이터 저장 시도 중... -> {DATA_CACHE_FILE}")
            try:
                with open(DATA_CACHE_FILE, "wb") as f:
                    pickle.dump(engine.raw_data_map, f)
                print(f"✅ RAW 데이터 저장 성공! (용량: {os.path.getsize(DATA_CACHE_FILE) / 1024 / 1024:.2f} MB)")
            except Exception as e:
                print(f"❌ RAW 데이터 저장 실패: {e}")
        else:
            print("⚠️ 경고: 다운로드된 데이터가 없습니다. (저장할 내용 없음)")

    print(f"✅ 최적화 루프 시작 (Total Symbols: {len(engine.symbols)})")
    print(f"📄 결과 저장 경로: {RESULT_FILE}\n")

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, engine), n_trials=2000, n_jobs=1)

    print("\n" + "="*50)
    print(f"🎉 Optimization Finished! (Total Trials: {len(study.trials)})")
    print("="*50)
    try:
        print(f"🏆 Best Equity : ${study.best_value:,.2f}")
        print("-" * 50)
        print("🌟 Best Parameters:")
        for key, value in study.best_params.items():
            print(f"   - {key:<15}: {value}")
    except:
        print("⚠️ 유효한 결과가 없습니다.")
    print("="*50)
