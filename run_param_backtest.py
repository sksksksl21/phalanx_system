import os
import pickle
from core.backtest_engine import BacktestEngine

# Trial 0 params (사용자가 준 값)
PARAMS = {
    "atr_period": 26,
    "atr_multiplier": 2.5,
    "adx_threshold": 16,
    "rsi_upper": 71,
    "rsi_lower": 36,
    "vol_factor": 1.0,
    "ema_intraday": 200,
    "daily_ema": 25,
}

# optimize에서 쓰던 RAW 캐시 파일(있으면 재사용 → 다운로드 생략)
RAW_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "market_data_cache.pkl")

if __name__ == "__main__":
    engine = BacktestEngine(days=180)

    # 1) RAW 캐시가 있으면 로드 (다운로드 생략)
    if os.path.exists(RAW_CACHE_FILE):
        try:
            with open(RAW_CACHE_FILE, "rb") as f:
                # 여기엔 raw_data_map이 들어있어야 함 (우리가 수정한 optimize.py 기준)
                engine.raw_data_map = pickle.load(f)
                engine.symbols = list(engine.raw_data_map.keys())
            print(f"✅ RAW cache loaded: {len(engine.symbols)} symbols")
        except Exception as e:
            print(f"⚠️ RAW cache load failed: {e}")
            engine.raw_data_map = {}

    # 2) RAW 없으면 다운로드
    if not engine.raw_data_map:
        engine.prepare_data()

    # 3) 파라미터 주입
    engine.titan.set_params(PARAMS)

    # 4) 현재 params로 지표 재주입
    engine.rebuild_indicators()

    # 5) 실행 + 리포트 출력 + MTM equity curve 저장
    engine.run(show_report=True)

    print("\n✅ Backtest completed.")
    print("   - backtest_history.csv updated")
    print("   - backtest_equity_curve.csv updated (if patch exists)")
