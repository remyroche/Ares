"""
MEXC Exchange Package

This package contains MEXC exchange implementation and klines downloading scripts.
"""

from .mexc import MexcExchange
from .klines_adapter import (
    MexcKlinesAdapter,
    create_mexc_klines_adapter
)

__all__ = [
    "MexcExchange",
    "MexcKlinesAdapter",
    "create_mexc_klines_adapter"
]