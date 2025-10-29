"""
BingX Exchange Package

This package contains BingX exchange implementation and klines downloading scripts.
"""

from .klines_adapter import (
    BingXKlinesAdapter,
    create_bingx_klines_adapter
)

# Import from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import directly from the bingx.py file to avoid circular imports
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("bingx", os.path.join(os.path.dirname(os.path.dirname(__file__)), "bingx.py"))
    bingx_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bingx_module)
    BingXExchange = bingx_module.BingXExchange
    create_bingx_exchange = bingx_module.create_bingx_exchange
    print(f"Importlib successful: BingXExchange={BingXExchange}, create_bingx_exchange={create_bingx_exchange}")
except Exception as e:
    print(f"Importlib failed: {e}")
    BingXExchange = None
    create_bingx_exchange = None

__all__ = [
    "BingXKlinesAdapter",
    "create_bingx_klines_adapter",
    "BingXExchange",
    "create_bingx_exchange"
]