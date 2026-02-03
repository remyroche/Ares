import requests
import pandas as pd
from dataclasses import dataclass

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

@dataclass
class MarginUniverseCache:
    symbols: list[str]
    asof_day: pd.Timestamp

def refresh_margin_universe_daily(cache: MarginUniverseCache | None, quote="USDT") -> MarginUniverseCache:
    today = pd.Timestamp.utcnow().tz_localize("UTC").floor("D")
    if cache is not None and cache.asof_day == today:
        return cache
    pairs = fetch_binance_cross_margin_pairs()
    syms = margin_pairs_to_spot_symbols(pairs, quote=quote)
    return MarginUniverseCache(symbols=syms, asof_day=today)

def build_fetch_universe(margin_symbols: list[str], market_basket: list[str], M: int):
    base = [s for s in margin_symbols if s.endswith("/USDT")]
    chosen = base[:M]
    return sorted(set(chosen).union(set(market_basket)))
