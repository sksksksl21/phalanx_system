import logging
from typing import Iterable, List, Optional, Set, Tuple, Dict, Any

logger = logging.getLogger("PhalanxUniverse")


def _safe_set(x: Optional[Iterable[str]]) -> Set[str]:
    if x is None:
        return set()
    return set(x)


def _get_exchange_from_executor(executor) -> Any:
    """
    Executor가 이미 보유한 ccxt exchange가 있으면 재사용하고,
    없으면 public 모드로 Binance Futures exchange를 생성한다.
    """
    # BinanceExecutor: self.exchange
    ex = getattr(executor, "exchange", None)
    if ex is not None:
        return ex

    # VirtualExecutor 등: exchange가 없을 수 있음 → public 생성
    import ccxt
    ex = ccxt.binance({
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",
            "adjustForTimeDifference": True
        }
    })
    try:
        ex.load_markets()
    except Exception as e:
        # load_markets 실패해도 fetch_tickers는 동작할 때가 있어 방어
        logger.warning(f"[Universe] load_markets failed: {e}")
    return ex


def _is_usdt_perp_symbol(symbol: str) -> bool:
    """
    Phalanx 기준: USDT-M Perp 형태를 우선한다.
    Binance futures에서 흔히 'BTC/USDT:USDT' 형태가 퍼페추얼 심볼.
    """
    if not isinstance(symbol, str):
        return False
    # 가장 안전한 필터: :USDT 포함 (swap/perp)
    if "/USDT:USDT" in symbol:
        return True
    # 일부 환경에서 ':USDT' 없이 'BTC/USDT'로만 보일 수 있어 보조 허용
    if symbol.endswith("/USDT"):
        return True
    return False


def get_universe(
    executor,
    top_n: int = 30,
    blacklist: Optional[Iterable[str]] = None,
    metric: str = "quoteVolume",
    oversample: int = 200,
    min_metric: float = 0.0
) -> List[str]:
    """
    [Single Source of Truth] Universe Selection

    목적:
    - 실행 시점에 시장에서 유동성 상위 종목을 선정한다.
    - 블랙리스트를 '선정 이전'에 적용하여, 결과 리스트가 (가능한 한) top_n을 채우도록 한다.

    규칙:
    1) USDT 선물(우선: '/USDT:USDT') 필터
    2) metric 기준 내림차순 정렬
    3) blacklist 제외 후 top_n개 채움 (oversample 범위 내)

    반환:
    - 표준화된 심볼 리스트 (가능한 한 top_n개)
    """
    bl = _safe_set(blacklist)
    ex = _get_exchange_from_executor(executor)

    try:
        tickers: Dict[str, Dict[str, Any]] = ex.fetch_tickers()
    except Exception as e:
        logger.error(f"[Universe] fetch_tickers failed: {e}")
        return []

    ranked: List[Tuple[str, float]] = []
    for sym, t in tickers.items():
        if not _is_usdt_perp_symbol(sym):
            continue
        try:
            v = t.get(metric, 0) if isinstance(t, dict) else 0
            v = float(v) if v is not None else 0.0
        except Exception:
            v = 0.0
        if v <= min_metric:
            continue
        ranked.append((sym, v))

    ranked.sort(key=lambda x: x[1], reverse=True)

    # oversample 만큼만 먼저 본다 (API/연산 보호)
    if oversample and oversample > 0:
        ranked = ranked[:max(oversample, top_n)]

    selected: List[str] = []
    removed_bl = 0
    for sym, _v in ranked:
        if sym in bl:
            removed_bl += 1
            continue
        selected.append(sym)
        if len(selected) >= top_n:
            break

    if len(selected) < top_n:
        logger.warning(
            f"[Universe] Not enough symbols to fill top_n={top_n}. "
            f"selected={len(selected)}, blacklist_removed={removed_bl}, scanned={len(ranked)}"
        )
    else:
        logger.info(
            f"[Universe] Selected {len(selected)} symbols. "
            f"blacklist_removed={removed_bl}, scanned={len(ranked)}"
        )

    return selected


def save_universe_snapshot(path: str, symbols: List[str], meta: Optional[Dict[str, Any]] = None) -> None:
    """
    Optimize → Backtest 재현을 위해 universe 스냅샷 저장.
    (Backtest에서 동일 종목군으로 '새 데이터'를 검증 가능)
    """
    import json
    payload = {
        "symbols": symbols,
        "meta": meta or {}
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_universe_snapshot(path: str) -> List[str]:
    import json
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    symbols = payload.get("symbols", [])
    if not isinstance(symbols, list):
        return []
    return [s for s in symbols if isinstance(s, str)]
