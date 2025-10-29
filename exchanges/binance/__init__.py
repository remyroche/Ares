"""
Binance Exchange Package

This package contains the Binance exchange implementation with shared processing logic.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util
spec = importlib.util.spec_from_file_location("binance_module", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "binance.py"))
binance_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(binance_module)
BinanceExchange = binance_module.BinanceExchange
create_binance_exchange = binance_module.create_binance_exchange
from .klines_adapter import BinanceKlinesAdapter, create_binance_klines_adapter

__all__ = ['BinanceExchange', 'create_binance_exchange', 'BinanceKlinesAdapter', 'create_binance_klines_adapter']