import glob
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests

from extreme_price_movements.utils import tprint

BINANCE_API = "https://api.binance.com"
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_MAX_RETRIES = 3
HARDCODED_EXCLUDED_SYMBOLS = frozenset(
    {
        "CHESS/USDT",
        "CHESS/USDC",
        "DATA/USDT",
        "DATA/USDC",
        "DF/USDT",
        "DF/USDC",
        "ESP/USDT",
        "ESP/USDC",
        "FOGO/USDT",
        "FOGO/USDC",
        "FRAX/USDT",
        "FRAX/USDC",
        "MANTRA/USDT",
        "MANTRA/USDC",
    }
)
DEDUP_QUOTES = ("USDC",)
DEDUP_QUOTE_PRIORITY = {quote: rank for rank, quote in enumerate(DEDUP_QUOTES)}
SUPPORTED_TRAINING_QUOTES = frozenset(DEDUP_QUOTES)
SPOT_QUOTE_SUFFIXES = (
    "USDT",
    "USDC",
    "BUSD",
    "USD1",
    "FDUSD",
    "EUR",
    "BTC",
    "ETH",
)
PERP_INFO_PATH = "/fapi/v1/exchangeInfo"
PERP_QUOTES = frozenset({"USDT", "USDC"})

_PERP_SYMBOL_CACHE: Optional[set[str]] = None


def _request_error_category(exc: Exception) -> str:
    text = f"{exc.__class__.__name__} {exc}".lower()
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code == 429 or "429" in text or "too many requests" in text:
        return "rate_limited"
    if status_code in {418, 403, 401} or "forbidden" in text or "unauthorized" in text:
        return "auth_or_blocked"
    if status_code is not None and 500 <= int(status_code) < 600:
        return "server_error"
    if "timeout" in text:
        return "timeout"
    if "connection" in text or "network" in text:
        return "network"
    return "request_error"


def _request_retry_after(exc: Exception) -> Optional[float]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("Retry-After") or headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _binance_get_json(path: str) -> object:
    """GET a public Binance REST path with retry-after/backoff logging."""
    url = f"{BINANCE_API}{path}"
    last_exc: Optional[Exception] = None
    for attempt in range(1, REQUEST_MAX_RETRIES + 1):
        try:
            tprint(
                f"Binance public GET {path}: attempt={attempt}/{REQUEST_MAX_RETRIES}"
            )
            response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_exc = exc
            category = _request_error_category(exc)
            tprint(
                f"Binance public GET failed {path}: category={category} "
                f"attempt={attempt}/{REQUEST_MAX_RETRIES}: {exc}"
            )
            if attempt < REQUEST_MAX_RETRIES:
                retry_after = _request_retry_after(exc)
                if retry_after is not None:
                    tprint(
                        f"Binance Retry-After for {path}: sleeping {retry_after:.2f}s"
                    )
                    time.sleep(retry_after)
                else:
                    time.sleep(
                        min(8.0, 2.0 ** (attempt - 1)) + random.uniform(0.0, 0.25)
                    )
    assert last_exc is not None
    raise last_exc


def _normalize_symbol(symbol: str) -> str:
    norm = str(symbol or "").upper().strip().replace("_", "/")
    if not norm:
        return norm
    if "/" in norm:
        return norm
    for quote in SPOT_QUOTE_SUFFIXES:
        if norm.endswith(quote) and len(norm) > len(quote):
            return f"{norm[:-len(quote)]}/{quote}"
    return norm


def _is_supported_training_symbol(symbol: str) -> bool:
    norm = _normalize_symbol(symbol)
    if not norm or "/" not in norm:
        return False
    _base, quote = norm.rsplit("/", 1)
    return quote in SUPPORTED_TRAINING_QUOTES


def _is_valid_spot_symbol_format(symbol: str) -> bool:
    norm = _normalize_symbol(symbol)
    return bool(re.fullmatch(r"[A-Z0-9]+/[A-Z0-9]+", norm or ""))


def deduplicate_symbols_by_base(symbols: list[str]) -> list[str]:
    """Return at most one symbol per base asset for USDT/USDC/BUSD quote variants."""
    original_count = len(symbols)

    best_by_base: dict[str, tuple[int, str]] = {}
    passthrough: set[str] = set()

    for raw_symbol in symbols:
        symbol = _normalize_symbol(raw_symbol)
        if not symbol:
            continue
        if "/" not in symbol:
            passthrough.add(symbol)
            continue
        base, quote = symbol.rsplit("/", 1)
        if not base or quote not in DEDUP_QUOTE_PRIORITY:
            passthrough.add(symbol)
            continue
        rank = DEDUP_QUOTE_PRIORITY[quote]
        current = best_by_base.get(base)
        if (
            current is None
            or rank < current[0]
            or (rank == current[0] and symbol < current[1])
        ):
            best_by_base[base] = (rank, symbol)

    deduped = set(passthrough)
    deduped.update(symbol for _, symbol in best_by_base.values())
    result = sorted(deduped)

    removed = original_count - len(result)
    if removed > 0:
        tprint(
            f"Quote deduplication removed {removed} duplicate symbols: {original_count} → {len(result)}"
        )
    return result


def apply_hardcoded_universe_exclusions(symbols: list[str]) -> list[str]:
    cleaned = []
    removed = 0
    unsupported = 0
    for sym in symbols:
        norm = _normalize_symbol(sym)
        if norm in HARDCODED_EXCLUDED_SYMBOLS:
            removed += 1
            continue
        if not _is_valid_spot_symbol_format(norm):
            unsupported += 1
            continue
        if not _is_supported_training_symbol(norm):
            unsupported += 1
            continue
        cleaned.append(norm)
    if removed:
        tprint(
            f"Hardcoded universe exclusions removed {removed} symbols: {sorted(HARDCODED_EXCLUDED_SYMBOLS)}"
        )
    if unsupported:
        tprint(
            f"Unsupported quote filtering removed {unsupported} symbols "
            f"(allowed quotes: {sorted(SUPPORTED_TRAINING_QUOTES)})"
        )
    return sorted(set(cleaned))


def fetch_binance_cross_margin_pairs():
    tprint("Entering function: fetch_binance_cross_margin_pairs in universe.py")
    cache_file = os.path.join(os.path.dirname(__file__), ".margin_universe_cache.json")

    # Check disk cache first
    try:
        if os.path.exists(cache_file):
            mtime = os.path.getmtime(cache_file)
            # Use cache if it's less than 24h old
            if (time.time() - mtime) < 86400:
                import json

                with open(cache_file, "r") as f:
                    margin_pairs = json.load(f)
                tprint(
                    f"Loaded {len(margin_pairs)} cross margin pairs from disk cache."
                )
                return margin_pairs
    except Exception as e:
        tprint(f"Warning: Failed to read margin universe cache: {e}")

    data = _binance_get_json("/api/v3/exchangeInfo")

    # Filter for margin permitted symbols
    margin_pairs = []
    for s in data.get("symbols", []):
        if s.get("isMarginTradingAllowed", False):
            margin_pairs.append(s)

    # Update disk cache
    try:
        import json

        with open(cache_file, "w") as f:
            json.dump(margin_pairs, f)
    except Exception as e:
        tprint(f"Warning: Failed to write margin universe cache: {e}")

    tprint(f"Fetched {len(margin_pairs)} cross margin pairs from exchangeInfo.")
    return margin_pairs


def margin_pairs_to_spot_symbols(margin_pairs_json, quotes=("USDC",)):
    tprint(f"Entering function: margin_pairs_to_spot_symbols in universe.py")

    # Backward compatibility for single quote string
    if isinstance(quotes, str):
        quotes = [quotes]

    tprint(f"Processing {len(margin_pairs_json)} margin pairs for quotes {quotes}...")
    out = set()
    breakdown = {q: 0 for q in quotes}

    for row in margin_pairs_json:
        s = row.get("symbol", "")
        for q in quotes:
            if s.endswith(q):
                base = s[: -len(q)]
                if base:
                    out.add(f"{base}/{q}")
                    breakdown[q] += 1
                break  # Matched one quote

    res = sorted(out)

    breakdown_str = ", ".join([f"{q}: {count}" for q, count in breakdown.items()])
    tprint(f"Found {len(res)} spot symbols. Breakdown: {breakdown_str}")
    return res


def fetch_binance_perp_spot_symbols() -> set[str]:
    tprint(f"Entering function: fetch_binance_perp_spot_symbols in universe.py")
    raw = _binance_get_json(PERP_INFO_PATH)
    spot_symbols = set()
    for row in raw.get("symbols", []):
        if not isinstance(row, dict):
            continue
        if row.get("contractType") != "PERPETUAL":
            continue
        if row.get("status", "").upper() != "TRADING":
            continue
        quote = str(row.get("quoteAsset", "")).upper()
        if quote not in PERP_QUOTES:
            continue
        base = str(row.get("baseAsset", "")).upper()
        if not base:
            continue
        spot_symbols.add(f"{base}/{quote}")

    tprint(
        f"Fetched {len(spot_symbols)} perpetual perp symbols across {sorted(PERP_QUOTES)}."
    )
    return spot_symbols


def get_available_perp_spot_symbols(force_refresh: bool = False) -> set[str]:
    global _PERP_SYMBOL_CACHE
    if force_refresh or _PERP_SYMBOL_CACHE is None:
        _PERP_SYMBOL_CACHE = fetch_binance_perp_spot_symbols()
    return set(_PERP_SYMBOL_CACHE)


def filter_symbols_without_perp_support(symbols: list[str]) -> list[str]:
    """Drop symbols without a USDT or USDC perpetual available."""
    if not symbols:
        return []
    try:
        perp_symbols = get_available_perp_spot_symbols()
    except Exception as exc:
        tprint(
            "Warning: failed to refresh perp universe; skipping perp-based filtering: "
            f"{_request_error_category(exc)}: {exc}"
        )
        return sorted(set(symbols))

    perp_bases = {sym.split("/", 1)[0] for sym in perp_symbols if "/" in sym}
    out = sorted({sym for sym in symbols if sym.split("/", 1)[0] in perp_bases})

    removed = len(set(symbols)) - len(out)
    if removed > 0:
        tprint(
            "Filtered out symbols without USDT/USDC perps from universe selection: "
            f"{removed} removed"
        )
    return out


def fetch_24h_tickers():
    tprint("Entering function: fetch_24h_tickers in universe.py")
    data = _binance_get_json("/api/v3/ticker/24hr")
    tprint(f"Fetched {len(data)} 24h tickers.")
    return data


@dataclass
class MarginUniverseCache:
    symbols: list[str]
    asof_day: pd.Timestamp


def refresh_margin_universe_daily(
    cache: Optional[MarginUniverseCache], quotes=("USDC",)
) -> MarginUniverseCache:
    tprint(f"Entering function: refresh_margin_universe_daily in universe.py")
    today = pd.Timestamp.utcnow().floor("D")
    if cache is not None and cache.asof_day == today:
        tprint("Using cached margin universe for today.")
        return cache

    tprint("Refreshing margin universe...")
    pairs = fetch_binance_cross_margin_pairs()
    syms = filter_symbols_without_perp_support(
        apply_hardcoded_universe_exclusions(
            margin_pairs_to_spot_symbols(pairs, quotes=quotes)
        )
    )
    tprint(f"Refreshed margin universe: {len(syms)} symbols.")
    return MarginUniverseCache(symbols=syms, asof_day=today)


def build_fetch_universe(margin_symbols: list[str], market_basket: list[str], M: int):
    """
    Selects all post-exclusion margin symbols.
    Always includes market_basket.
    """
    tprint(f"Entering function: build_fetch_universe in universe.py")
    margin_symbols = filter_symbols_without_perp_support(
        apply_hardcoded_universe_exclusions(list(margin_symbols))
    )
    market_basket = filter_symbols_without_perp_support(
        apply_hardcoded_universe_exclusions(list(market_basket))
    )
    # Deduplicate quote variants (preferred quote first in DEDUP_QUOTES).
    margin_symbols = deduplicate_symbols_by_base(margin_symbols)
    market_basket = deduplicate_symbols_by_base(market_basket)
    tprint(
        f"Building universe from {len(margin_symbols)} margin symbols + "
        f"{len(market_basket)} basket symbols (no volume-based filtering)."
    )
    final = apply_hardcoded_universe_exclusions(
        list(set(margin_symbols).union(set(market_basket)))
    )
    tprint(f"Universe selected: {len(final)} symbols (no volume-based filtering)")
    return final


def get_training_universe(margin_symbols, cfg, store, ts_sig=None):
    """
    Standardized training universe selection:
    1. Fetch Universe (All except bottom 30 by volume)
    2. Union with Market Basket
    """

    def _local_store_symbols(_store):
        out = []
        ohlcv_dir = getattr(_store, "ohlcv_dir", None)
        if not ohlcv_dir:
            return out
        for path in glob.glob(os.path.join(ohlcv_dir, "symbol=*")):
            base = os.path.basename(path)
            if not base.startswith("symbol="):
                continue
            raw = base.replace("symbol=", "")
            out.append(raw.replace("_", "/", 1))
        return apply_hardcoded_universe_exclusions(out)

    offline_universe = bool(cfg.get("offline_backtest_skip_universe_refresh", False))
    if offline_universe:
        tprint(
            "Training universe: offline mode enabled, skipping margin refresh and live ticker ranking."
        )
        local_syms = _local_store_symbols(store)
        base_syms = margin_symbols if margin_symbols is not None else local_syms
        if not base_syms:
            base_syms = list(cfg.get("market_basket", []))
        # In offline mode, use all available symbols (no volume data to remove bottom 30)
        train_syms = deduplicate_symbols_by_base(list(set(base_syms)))
        train_syms = filter_symbols_without_perp_support(
            apply_hardcoded_universe_exclusions(train_syms)
        )
        M = int(cfg.get("fetch_symbols_M", 9999))
        if len(train_syms) > M:
            tprint(
                f"Offline mode: limiting universe from {len(train_syms)} to top {M} (alphabetical fallback)"
            )
            train_syms = train_syms[:M]
        return train_syms

    if margin_symbols is None:
        # Fallback if not provided.
        # Prefer live refresh; if unavailable (e.g., offline/DNS issues), use local store symbols.
        try:
            mu = refresh_margin_universe_daily(None, quotes=("USDC",))
            margin_symbols = mu.symbols
        except Exception as e:
            tprint(
                "Warning: margin universe refresh failed "
                f"({_request_error_category(e)}: {e}); "
                "falling back to local store symbols."
            )
            margin_symbols = _local_store_symbols(store)
            if not margin_symbols:
                tprint(
                    "Warning: local store symbol fallback empty; using market basket only."
                )
                margin_symbols = list(cfg.get("market_basket", []))

    syms_all = build_fetch_universe(
        margin_symbols, cfg["market_basket"], cfg["fetch_symbols_M"]
    )
    train_syms = deduplicate_symbols_by_base(list(set(syms_all)))
    train_syms = apply_hardcoded_universe_exclusions(train_syms)
    return train_syms


def select_live_candidates(
    margin_symbols: list[str], market_basket: list[str], pct: float = 0.05
):
    """
    Selects candidates for 1h analysis.
    Returns list of symbols to fetch (margin symbols + market basket).
    Price change filtering removed per user request.
    """
    tprint(f"Entering function: select_live_candidates in universe.py")
    margin_symbols = filter_symbols_without_perp_support(
        apply_hardcoded_universe_exclusions(list(margin_symbols))
    )
    market_basket = filter_symbols_without_perp_support(
        apply_hardcoded_universe_exclusions(list(market_basket))
    )
    tprint(
        f"Selecting live candidates from {len(margin_symbols)} margin symbols + {len(market_basket)} basket"
    )

    # Simply return margin symbols + market basket (no price change filtering)
    candidates = set(margin_symbols + market_basket)
    tprint(f"Live Candidates: {len(candidates)} (All margin symbols + Basket)")
    return sorted(list(candidates))
