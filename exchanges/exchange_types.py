"""
Exchange Types

Common enumerations used across exchange modules to avoid circular imports.
"""

from enum import Enum

class TradingMode(Enum):
    """Trading modes."""
    TRADE = "trade"
    SPOT = "spot"
    FUTURES = "futures"
    MARGIN = "margin"
    PAPER = "paper"
    SIMULATION = "simulation"

class ExchangeType(Enum):
    """Exchange types."""
    BINANCE = "binance"
    BINGX = "bingx"
    BYBIT = "bybit"
    OKX = "okx"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    KUCOIN = "kucoin"
    GATEIO = "gateio"
    HUOBI = "huobi"
    MEXC = "mexc"
    PHEMEX = "phemex"
    UNKNOWN = "unknown"
