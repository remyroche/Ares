"""
Binance Exchange Package

This package contains the Binance exchange implementation with shared processing logic.
"""

# Note: BinanceExchange is imported from the parent-level exchanges.binance module
# To avoid circular imports, we don't re-export it here
from .klines_adapter import BinanceKlinesAdapter, create_binance_klines_adapter

__all__ = ['BinanceKlinesAdapter', 'create_binance_klines_adapter']