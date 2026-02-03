import requests
import pandas as pd
from dataclasses import dataclass
from extreme_price_movements.utils import tprint

BINANCE_API = "https://api.binance.com"

def fetch_binance_cross_margin_pairs():
    r = requests.get(f"{BINANCE_API}/sapi/v1/margin/allPairs", timeout=30)
    r.raise_for_status()
    return r.json()

def margin_pairs_to_spot_symbols(margin_pairs_json, quote="USDT"):
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
    r = requests.get(f"{BINANCE_API}/api/v3/ticker/24hr", timeout=30)
    r.raise_for_status()
    return r.json()

@dataclass
class MarginUniverseCache:
    symbols: list[str]
    asof_day: pd.Timestamp

def refresh_margin_universe_daily(cache: MarginUniverseCache | None, quote="USDT") -> MarginUniverseCache:
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
    try:
        tickers = fetch_24h_tickers()
        # Map symbol "BTCUSDT" -> volume
        # We need to match "BTC/USDT" to "BTCUSDT"
        vol_map = {}
        for t in tickers:
            s = t["symbol"]
            v = float(t["quoteVolume"]) # Use quote volume (USDT)
            vol_map[s] = v

        # Filter margin symbols and sort
        scored = []
        for s in margin_symbols:
            # s is "BTC/USDT"
            # api s is "BTCUSDT"
            api_s = s.replace("/", "")
            vol = vol_map.get(api_s, 0.0)
            scored.append((vol, s))

        scored.sort(key=lambda x: x[0], reverse=True)

        top_m = [x[1] for x in scored[:M]]

        # Ensure basket is present
        final = sorted(set(top_m).union(set(market_basket)))
        tprint(f"Universe selected: {len(final)} symbols (Top {M} by vol + basket)")
        return final

    except Exception as e:
        tprint(f"Error fetching tickers for universe selection: {e}. Fallback to alphabet.")
        return sorted(list(set(margin_symbols[:M]).union(set(market_basket))))
