import requests
import pandas as pd
import time
from dataclasses import dataclass
import os
import glob
from extreme_price_movements.utils import tprint

BINANCE_API = "https://api.binance.com"
HARDCODED_EXCLUDED_SYMBOLS = frozenset({
    "CHESS/USDT",
    "DATA/USDT",
    "DF/USDT",
})
DEDUP_QUOTES = ("USDT", "USDC", "BUSD")
DEDUP_QUOTE_PRIORITY = {quote: rank for rank, quote in enumerate(DEDUP_QUOTES)}


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").upper().replace("_", "/").strip()


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
        if current is None or rank < current[0] or (rank == current[0] and symbol < current[1]):
            best_by_base[base] = (rank, symbol)

    deduped = set(passthrough)
    deduped.update(symbol for _, symbol in best_by_base.values())
    result = sorted(deduped)

    removed = original_count - len(result)
    if removed > 0:
        tprint(f"Quote deduplication removed {removed} duplicate symbols: {original_count} → {len(result)}")
    return result


def apply_hardcoded_universe_exclusions(symbols: list[str]) -> list[str]:
    cleaned = []
    removed = 0
    for sym in symbols:
        norm = _normalize_symbol(sym)
        if norm in HARDCODED_EXCLUDED_SYMBOLS:
            removed += 1
            continue
        cleaned.append(norm)
    if removed:
        tprint(f"Hardcoded universe exclusions removed {removed} symbols: {sorted(HARDCODED_EXCLUDED_SYMBOLS)}")
    return sorted(set(cleaned))

def fetch_binance_cross_margin_pairs():
    tprint(f"Entering function: fetch_binance_cross_margin_pairs in universe.py")
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
                tprint(f"Loaded {len(margin_pairs)} cross margin pairs from disk cache.")
                return margin_pairs
    except Exception as e:
        tprint(f"Warning: Failed to read margin universe cache: {e}")

    # Use public exchangeInfo endpoint instead of SAPI
    r = requests.get(f"{BINANCE_API}/api/v3/exchangeInfo", timeout=30)
    r.raise_for_status()
    data = r.json()
    
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

def margin_pairs_to_spot_symbols(margin_pairs_json, quotes=("USDT", "USDC", "BUSD", "EUR")):
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
                base = s[:-len(q)]
                if base:
                    out.add(f"{base}/{q}")
                    breakdown[q] += 1
                break # Matched one quote

    res = sorted(out)

    breakdown_str = ", ".join([f"{q}: {count}" for q, count in breakdown.items()])
    tprint(f"Found {len(res)} spot symbols. Breakdown: {breakdown_str}")
    return res

def fetch_24h_tickers():
    tprint(f"Entering function: fetch_24h_tickers in universe.py")
    r = requests.get(f"{BINANCE_API}/api/v3/ticker/24hr", timeout=30)
    r.raise_for_status()
    data = r.json()
    tprint(f"Fetched {len(data)} 24h tickers.")
    return data

from typing import Optional

@dataclass
class MarginUniverseCache:
    symbols: list[str]
    asof_day: pd.Timestamp

def refresh_margin_universe_daily(cache: Optional[MarginUniverseCache], quotes=("USDT", "USDC", "BUSD", "EUR")) -> MarginUniverseCache:
    tprint(f"Entering function: refresh_margin_universe_daily in universe.py")
    today = pd.Timestamp.utcnow().floor("D")
    if cache is not None and cache.asof_day == today:
        tprint("Using cached margin universe for today.")
        return cache

    tprint("Refreshing margin universe...")
    pairs = fetch_binance_cross_margin_pairs()
    syms = apply_hardcoded_universe_exclusions(margin_pairs_to_spot_symbols(pairs, quotes=quotes))
    tprint(f"Refreshed margin universe: {len(syms)} symbols.")
    return MarginUniverseCache(symbols=syms, asof_day=today)

def build_fetch_universe(margin_symbols: list[str], market_basket: list[str], M: int):
    """
    Selects all margin symbols except bottom 30 by 24h volume.
    Always includes market_basket.
    """
    tprint(f"Entering function: build_fetch_universe in universe.py")
    margin_symbols = apply_hardcoded_universe_exclusions(list(margin_symbols))
    market_basket = apply_hardcoded_universe_exclusions(list(market_basket))
    # Deduplicate quote variants (USDT > USDC > BUSD) before volume ranking
    margin_symbols = deduplicate_symbols_by_base(margin_symbols)
    market_basket = deduplicate_symbols_by_base(market_basket)
    tprint(f"Building universe from {len(margin_symbols)} margin symbols + {len(market_basket)} basket symbols (Remove bottom 30 by volume).")
    try:
        tickers = fetch_24h_tickers()
        vol_map = {}
        for t in tickers:
            s = t["symbol"]
            v = float(t["quoteVolume"])
            vol_map[s] = v
        tprint(f"Volume map built for {len(vol_map)} tickers.")

        scored = []
        for s in margin_symbols:
            api_s = s.replace("/", "")
            vol = vol_map.get(api_s, 0.0)
            scored.append((vol, s))
        tprint(f"Scored {len(scored)} margin symbols.")

        scored.sort(key=lambda x: x[0], reverse=True)

        # Remove bottom 5 by volume
        bottom_n = 5
        if len(scored) > bottom_n:
            top_m = [x[1] for x in scored[:-bottom_n]]
        else:
            top_m = [x[1] for x in scored]
        tprint(f"Selected {len(top_m)} symbols by volume (removed bottom {bottom_n}).")

        final = apply_hardcoded_universe_exclusions(list(set(top_m).union(set(market_basket))))
        tprint(f"Universe selected: {len(final)} symbols (All except bottom {bottom_n} by vol + basket)")
        return final

    except Exception as e:
        tprint(f"Error fetching tickers for universe selection: {e}. Fallback to alphabet.")
        return apply_hardcoded_universe_exclusions(list(set(margin_symbols).union(set(market_basket))))

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
        tprint("Training universe: offline mode enabled, skipping margin refresh and live ticker ranking.")
        local_syms = _local_store_symbols(store)
        base_syms = margin_symbols if margin_symbols is not None else local_syms
        if not base_syms:
            base_syms = list(cfg.get("market_basket", []))
        # In offline mode, use all available symbols (no volume data to remove bottom 30)
        train_syms = deduplicate_symbols_by_base(
            list(set(base_syms).union(set(cfg["market_basket"])))
        )
        train_syms = apply_hardcoded_universe_exclusions(train_syms)
        return train_syms

    if margin_symbols is None:
        # Fallback if not provided.
        # Prefer live refresh; if unavailable (e.g., offline/DNS issues), use local store symbols.
        try:
            mu = refresh_margin_universe_daily(None, quotes=("USDT", "USDC", "BUSD", "EUR"))
            margin_symbols = mu.symbols
        except Exception as e:
            tprint(f"Warning: margin universe refresh failed ({e}); falling back to local store symbols.")
            margin_symbols = _local_store_symbols(store)
            if not margin_symbols:
                tprint("Warning: local store symbol fallback empty; using market basket only.")
                margin_symbols = list(cfg.get("market_basket", []))

    syms_all = build_fetch_universe(margin_symbols, cfg["market_basket"], cfg["fetch_symbols_M"])
    train_syms = deduplicate_symbols_by_base(
        list(set(syms_all).union(set(cfg["market_basket"])))
    )
    train_syms = apply_hardcoded_universe_exclusions(train_syms)
    return train_syms

def select_live_candidates(margin_symbols: list[str], market_basket: list[str], pct: float = 0.05):
    """
    Selects candidates based on 24h price change (Top Gainers/Losers).
    Returns list of symbols to fetch for 1h analysis.
    """
    tprint(f"Entering function: select_live_candidates in universe.py")
    margin_symbols = apply_hardcoded_universe_exclusions(list(margin_symbols))
    market_basket = apply_hardcoded_universe_exclusions(list(market_basket))
    tprint(f"Selecting live candidates from {len(margin_symbols)} margin symbols + {len(market_basket)} basket (pct={pct})")
    try:
        tickers = fetch_24h_tickers()
        # Map symbol -> priceChangePercent
        change_map = {}
        for t in tickers:
            s = t["symbol"]
            chg = float(t["priceChangePercent"])
            change_map[s] = chg
        tprint(f"Change map built for {len(change_map)} tickers.")

        # Filter for margin symbols
        valid = []
        for s in margin_symbols:
            api_s = s.replace("/", "")
            if api_s in change_map:
                valid.append((change_map[api_s], s))
        tprint(f"Found {len(valid)} valid margin symbols in ticker data.")

        if not valid:
            tprint("No valid margin symbols found. Returning market basket.")
            return market_basket

        # Sort by change
        valid.sort(key=lambda x: x[0]) # Ascending

        n = len(valid)
        k = max(5, int(n * pct))

        top_losers = [x[1] for x in valid[:k]]
        top_gainers = [x[1] for x in valid[-k:]]

        tprint(f"Selected {len(top_losers)} top losers (worst: {valid[0][0]}%) and {len(top_gainers)} top gainers (best: {valid[-1][0]}%)")

        candidates = set(top_losers + top_gainers + market_basket)
        tprint(f"Live Candidates: {len(candidates)} (Top/Bot {k} + Basket)")
        return sorted(list(candidates))

    except Exception as e:
        tprint(f"Error selecting live candidates: {e}. Fallback to basket.")
        return market_basket
    variance_stride = int(cfg.get("variance_filter_stride", 1) or 1)
