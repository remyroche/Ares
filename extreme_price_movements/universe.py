import requests
import pandas as pd
from dataclasses import dataclass
from extreme_price_movements.utils import tprint

BINANCE_API = "https://api.binance.com"

def fetch_binance_cross_margin_pairs():
    tprint(f"Entering function: fetch_binance_cross_margin_pairs in universe.py")
    r = requests.get(f"{BINANCE_API}/sapi/v1/margin/allPairs", timeout=30)
    r.raise_for_status()
    return r.json()

def margin_pairs_to_spot_symbols(margin_pairs_json, quote="USDT"):
    tprint(f"Entering function: margin_pairs_to_spot_symbols in universe.py")
    out = set()
    for row in margin_pairs_json:
        s = row.get("symbol", "")
        if not s.endswith(quote):
            continue
        base = s[:-len(quote)]
        if base:
            out.add(f"{base}/{quote}")
    return sorted(out)

def fetch_24h_tickers():
    tprint(f"Entering function: fetch_24h_tickers in universe.py")
    r = requests.get(f"{BINANCE_API}/api/v3/ticker/24hr", timeout=30)
    r.raise_for_status()
    return r.json()

@dataclass
class MarginUniverseCache:
    symbols: list[str]
    asof_day: pd.Timestamp

def refresh_margin_universe_daily(cache: MarginUniverseCache | None, quote="USDT") -> MarginUniverseCache:
    tprint(f"Entering function: refresh_margin_universe_daily in universe.py")
    today = pd.Timestamp.utcnow().tz_localize("UTC").floor("D")
    if cache is not None and cache.asof_day == today:
        return cache

    tprint("Refreshing margin universe...")
    pairs = fetch_binance_cross_margin_pairs()
    syms = margin_pairs_to_spot_symbols(pairs, quote=quote)
    return MarginUniverseCache(symbols=syms, asof_day=today)

def build_fetch_universe(margin_symbols: list[str], market_basket: list[str], M: int):
    """
    Selects top M symbols by 24h volume from margin_symbols.
    Always includes market_basket.
    """
    tprint(f"Entering function: build_fetch_universe in universe.py")
    try:
        tickers = fetch_24h_tickers()
        vol_map = {}
        for t in tickers:
            s = t["symbol"]
            v = float(t["quoteVolume"])
            vol_map[s] = v

        scored = []
        for s in margin_symbols:
            api_s = s.replace("/", "")
            vol = vol_map.get(api_s, 0.0)
            scored.append((vol, s))

        scored.sort(key=lambda x: x[0], reverse=True)

        top_m = [x[1] for x in scored[:M]]

        final = sorted(set(top_m).union(set(market_basket)))
        tprint(f"Universe selected: {len(final)} symbols (Top {M} by vol + basket)")
        return final

    except Exception as e:
        tprint(f"Error fetching tickers for universe selection: {e}. Fallback to alphabet.")
        return sorted(list(set(margin_symbols[:M]).union(set(market_basket))))

def select_live_candidates(margin_symbols: list[str], market_basket: list[str], pct: float = 0.05):
    """
    Selects candidates based on 24h price change (Top Gainers/Losers).
    Returns list of symbols to fetch for 1h analysis.
    """
    tprint(f"Entering function: select_live_candidates in universe.py")
    try:
        tickers = fetch_24h_tickers()
        # Map symbol -> priceChangePercent
        change_map = {}
        for t in tickers:
            s = t["symbol"]
            chg = float(t["priceChangePercent"])
            change_map[s] = chg

        # Filter for margin symbols
        valid = []
        for s in margin_symbols:
            api_s = s.replace("/", "")
            if api_s in change_map:
                valid.append((change_map[api_s], s))

        if not valid:
            return market_basket

        # Sort by change
        valid.sort(key=lambda x: x[0]) # Ascending

        n = len(valid)
        k = max(5, int(n * pct))

        top_losers = [x[1] for x in valid[:k]]
        top_gainers = [x[1] for x in valid[-k:]]

        candidates = set(top_losers + top_gainers + market_basket)
        tprint(f"Live Candidates: {len(candidates)} (Top/Bot {k} + Basket)")
        return sorted(list(candidates))

    except Exception as e:
        tprint(f"Error selecting live candidates: {e}. Fallback to basket.")
        return market_basket
