import logging
import re
from typing import Iterable, List, Optional, Set, Tuple, Dict, Any

logger = logging.getLogger("PhalanxUniverse")

# -----------------------------
# 정책: 코어 8 + 위성 6 = 14
# -----------------------------
DEFAULT_CORE = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "LINK"]
DEFAULT_TOP_N = 14
DEFAULT_CORE_N = 8
DEFAULT_SATELLITE_N = 6

# 유니버스에서 원천 배제하고 싶은 기초자산(현물/스테이블/금은/지표성 등)
# - 네 의도: "썩은 코인" + "이상한 코인" + "코모디티/스테이블" 최대 배제
DEFAULT_EXCLUDE_BASE_ASSETS = {
    "USDC",
    "FDUSD",
    "TUSD",
    "BUSD",
    "USDP",
    "DAI",
    "PAXG",  # 금
    "XAU",   # 금 계열
    "XAG",   # 은 계열
}

# 심볼 유효성: 영문/숫자만 허용
# 예: BTC/USDT:USDT, 1000SHIB/USDT:USDT OK
# 예: 我踏马来了/USDT:USDT 같은 것 차단
_VALID_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,20}/USDT(?::USDT)?$")


def _safe_set(x: Optional[Iterable[str]]) -> Set[str]:
    return set(x) if x else set()


def _normalize_blacklist(blacklist: Optional[Iterable[str]]) -> Set[str]:
    """
    블랙리스트가 'BTC/USDT' 또는 'BTC/USDT:USDT' 어느 쪽으로 와도 둘 다 막히게 확장.
    """
    bl = _safe_set(blacklist)
    out = set()
    for s in bl:
        if not isinstance(s, str):
            continue
        out.add(s)
        out.add(s.split(":")[0])
        if s.endswith("/USDT") and ":USDT" not in s:
            out.add(s + ":USDT")
    return out


def _get_exchange_from_executor(executor) -> Any:
    """
    executor.exchange가 있으면 재사용.
    없으면 public 모드 ccxt.binance futures 생성.
    """
    ex = getattr(executor, "exchange", None) if executor is not None else None
    if ex is not None:
        return ex

    import ccxt
    ex = ccxt.binance(
        {
            "enableRateLimit": True,
            "options": {"defaultType": "future", "adjustForTimeDifference": True},
        }
    )
    try:
        ex.load_markets()
    except Exception as e:
        logger.warning(f"[Universe] load_markets failed: {e}")
    return ex


def _is_usdt_perp_symbol(symbol: str) -> bool:
    if not isinstance(symbol, str):
        return False
    if "/USDT:USDT" in symbol:
        return True
    if symbol.endswith("/USDT"):
        return True
    return False


def _is_clean_symbol(symbol: str) -> bool:
    if not isinstance(symbol, str):
        return False
    return _VALID_SYMBOL_RE.match(symbol) is not None


def _base_asset(symbol: str) -> str:
    """
    "BTC/USDT:USDT" -> "BTC"
    "1000PEPE/USDT:USDT" -> "1000PEPE"
    """
    if not isinstance(symbol, str) or "/" not in symbol:
        return ""
    return symbol.split("/")[0].strip().upper()


def _is_excluded_base(symbol: str, exclude_bases: Set[str]) -> bool:
    b = _base_asset(symbol)
    return (b in exclude_bases) if b else False


def _choose_core_symbols(tickers: Dict[str, Dict[str, Any]], core_assets: List[str]) -> List[str]:
    """
    core_assets: ["BTC","ETH",...]
    tickers keys에서 실제 존재하는 심볼로 매핑.
    우선순위: "BTC/USDT:USDT" > "BTC/USDT"
    """
    out = []
    for a in core_assets:
        cand1 = f"{a}/USDT:USDT"
        cand2 = f"{a}/USDT"
        if cand1 in tickers:
            out.append(cand1)
        elif cand2 in tickers:
            out.append(cand2)
        else:
            continue
    return out


def _has_enough_ohlcv(ex, symbol: str, timeframe: str, min_rows: int) -> bool:
    """
    현실적 데이터 필터:
    - 여기서의 min_rows는 "최근 N개 캔들"이 돌아오는지로 검증
    - 핵심: 검증 실패가 많을 때(30일 모드에서 특히) '심볼 형식 차이'를 함께 시도한다.
      (ex: 'AAA/USDT:USDT' vs 'AAA/USDT')
    """
    def _try(sym: str) -> Optional[int]:
        try:
            limit = min(1500, max(200, int(min_rows)))
            ohlcv = ex.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
            return len(ohlcv) if ohlcv is not None else 0
        except Exception:
            return None

    # 1) 원본 심볼로 시도
    n = _try(symbol)
    if n is not None and n >= min_rows:
        return True

    # 2) 심볼 형식 바꿔서 재시도 (환경/마켓 매핑 차이 방어)
    if isinstance(symbol, str):
        if symbol.endswith(":USDT"):
            alt = symbol.split(":")[0]  # "AAA/USDT:USDT" -> "AAA/USDT"
        else:
            alt = symbol + ":USDT"      # "AAA/USDT" -> "AAA/USDT:USDT"
        n2 = _try(alt)
        if n2 is not None and n2 >= min_rows:
            return True

    return False


def get_universe(
    executor,
    top_n: int = DEFAULT_TOP_N,
    core_assets: Optional[List[str]] = None,
    core_n: int = DEFAULT_CORE_N,
    satellite_n: int = DEFAULT_SATELLITE_N,
    blacklist: Optional[Iterable[str]] = None,
    metric: str = "quoteVolume",
    oversample: int = 300,
    min_quote_volume: float = 50_000_000.0,  # 24h 거래대금 최소 기준 (USDT)
    validate_ohlcv: bool = True,
    timeframe: str = "15m",
    min_ohlcv_rows: int = 800,
    exclude_base_assets: Optional[Set[str]] = None,
    # ✅ 핵심: 30일 모드에서 위성이 0개로 붕괴하는 것을 방지하는 "점진 완화" 옵션
    relax_ohlcv_if_insufficient: bool = True,
) -> List[str]:
    """
    [Phalanx Universe Policy]
    - Universe size = 14 (default)
    - Core 8: BTC, ETH, SOL, BNB, XRP, ADA, DOGE, LINK
    - Satellite 6: 필터 통과한 거래대금 상위

    위성 확보 정책(의도대로 동작하게 만드는 핵심):
    1) 가능한 한 validate_ohlcv + min_ohlcv_rows로 엄격하게 채움
    2) 그래도 위성이 부족하면(30일 모드에서 흔함):
       - min_ohlcv_rows를 800 -> 600 -> 400 으로 점진 완화 (validate 유지)
    3) 그래도 부족하면 마지막으로 validate_ohlcv를 끄고(거래대금+클린심볼+배제목록은 유지)
       위성 슬롯을 "반드시" 채운다.
    """
    if top_n != (core_n + satellite_n):
        top_n = core_n + satellite_n

    core_assets = core_assets or DEFAULT_CORE
    bl = _normalize_blacklist(blacklist)
    ex = _get_exchange_from_executor(executor)

    exclude_bases = set(exclude_base_assets) if exclude_base_assets is not None else set(DEFAULT_EXCLUDE_BASE_ASSETS)

    try:
        tickers: Dict[str, Dict[str, Any]] = ex.fetch_tickers()
    except Exception as e:
        logger.error(f"[Universe] fetch_tickers failed: {e}")
        return []

    # -----------------------------
    # 1) 코어 먼저 확정
    # -----------------------------
    core_symbols_all = _choose_core_symbols(tickers, core_assets)

    core_symbols: List[str] = []
    removed_core_bl = 0
    for s in core_symbols_all:
        if s in bl or s.split(":")[0] in bl:
            removed_core_bl += 1
            continue
        if not _is_usdt_perp_symbol(s) or not _is_clean_symbol(s):
            continue
        if _is_excluded_base(s, exclude_bases):
            continue
        core_symbols.append(s)
        if len(core_symbols) >= core_n:
            break

    # -----------------------------
    # 2) 위성 후보 풀 구성
    # -----------------------------
    ranked: List[Tuple[str, float]] = []
    for sym, t in tickers.items():
        if not _is_usdt_perp_symbol(sym):
            continue
        if not _is_clean_symbol(sym):
            continue
        if sym in bl or sym.split(":")[0] in bl:
            continue
        if sym in core_symbols:
            continue
        if _is_excluded_base(sym, exclude_bases):
            continue

        try:
            v = t.get(metric, 0) if isinstance(t, dict) else 0
            v = float(v) if v is not None else 0.0
        except Exception:
            v = 0.0

        if v < float(min_quote_volume):
            continue

        ranked.append((sym, v))

    ranked.sort(key=lambda x: x[1], reverse=True)
    if oversample and oversample > 0:
        ranked = ranked[: max(int(oversample), satellite_n * 20)]

    # -----------------------------
    # 3) 위성 선정 (의도대로 6개 "채우기" 로직)
    # -----------------------------
    satellites: List[str] = []
    skipped_ohlcv = 0

    def _fill_satellites_with_rows_threshold(rows_threshold: int, do_validate: bool) -> Tuple[List[str], int]:
        nonlocal ranked
        picked: List[str] = []
        skipped = 0
        for sym, _v in ranked:
            if sym in core_symbols or sym in picked:
                continue
            if do_validate:
                if not _has_enough_ohlcv(ex, sym, timeframe=timeframe, min_rows=rows_threshold):
                    skipped += 1
                    continue
            picked.append(sym)
            if len(picked) >= satellite_n:
                break
        return picked, skipped

    if validate_ohlcv:
        if relax_ohlcv_if_insufficient:
            # ✅ 점진 완화: 800 -> 600 -> 400
            for thr in (int(min_ohlcv_rows), 600, 400):
                satellites, skipped_ohlcv = _fill_satellites_with_rows_threshold(thr, do_validate=True)
                if len(satellites) >= satellite_n:
                    break

            # ✅ 그래도 부족하면 "마지막 보루": validate 끄고 채움 (거래대금/클린/배제/블랙리스트는 그대로)
            if len(satellites) < satellite_n:
                more, _sk = _fill_satellites_with_rows_threshold(rows_threshold=0, do_validate=False)
                satellites = more[:satellite_n]
        else:
            satellites, skipped_ohlcv = _fill_satellites_with_rows_threshold(int(min_ohlcv_rows), do_validate=True)
            if len(satellites) < satellite_n:
                more, _sk = _fill_satellites_with_rows_threshold(rows_threshold=0, do_validate=False)
                satellites = more[:satellite_n]
    else:
        satellites, skipped_ohlcv = _fill_satellites_with_rows_threshold(rows_threshold=0, do_validate=False)

    universe = (core_symbols + satellites)[:top_n]

    logger.info(
        f"[Universe] Policy core={len(core_symbols)}/{core_n}, sat={len(satellites)}/{satellite_n}, "
        f"total={len(universe)}/{top_n}, core_bl_removed={removed_core_bl}, "
        f"sat_ohlcv_skipped={skipped_ohlcv}, scanned={len(ranked)}"
    )

    if len(universe) < top_n:
        logger.warning(
            f"[Universe] Not enough symbols to fill target size. total={len(universe)}/{top_n}. "
            f"Try lowering min_quote_volume or min_ohlcv_rows, or disable validate_ohlcv."
        )

    return universe


def save_universe_snapshot(path: str, symbols: List[str], meta: Optional[Dict[str, Any]] = None) -> None:
    import json
    payload = {"symbols": symbols, "meta": meta or {}}
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
