"""
Binance Exchange Package

This package contains the Binance exchange implementation with shared processing logic.
"""

from .klines_adapter import BinanceKlinesAdapter, create_binance_klines_adapter

__all__ = ['BinanceKlinesAdapter', 'create_binance_klines_adapter']