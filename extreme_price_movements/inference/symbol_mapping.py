"""Symbol mapping helpers for training/live quote transitions."""

from __future__ import annotations

from typing import Iterable, Set

KNOWN_QUOTES = ("USDT", "USDC", "BUSD", "USD1", "FDUSD", "TUSD", "BTC", "ETH")


def normalise_symbol(symbol: str) -> str:
    """Return a ccxt-style uppercase symbol, e.g. ``BTCUSDC`` -> ``BTC/USDC``."""
    raw = str(symbol or "").strip().upper().replace("_", "/")
    if "/" in raw:
        base, quote = raw.split("/", 1)
        return f"{base}/{quote}"
    for quote in KNOWN_QUOTES:
        if raw.endswith(quote) and len(raw) > len(quote):
            return f"{raw[:-len(quote)]}/{quote}"
    return raw


def symbol_base(symbol: str) -> str:
    """Return the base asset for a normalized symbol."""
    norm = normalise_symbol(symbol)
    return norm.split("/", 1)[0] if "/" in norm else norm


def symbol_quote(symbol: str) -> str:
    """Return the quote asset for a normalized symbol, or an empty string."""
    norm = normalise_symbol(symbol)
    return norm.split("/", 1)[1] if "/" in norm else ""


def convert_symbol_quote(symbol: str, quote: str) -> str:
    """Convert ``BTC/USDT`` to ``BTC/<quote>`` while preserving the base asset."""
    quote_norm = str(quote or "").strip().upper()
    if not quote_norm:
        return normalise_symbol(symbol)
    return f"{symbol_base(symbol)}/{quote_norm}"


def symbol_bases(symbols: Iterable[str]) -> Set[str]:
    """Return base assets from a symbol iterable."""
    return {symbol_base(sym) for sym in symbols if symbol_base(sym)}
