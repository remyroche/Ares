"""
GateIO Exchange Package

This package contains the GateIO exchange implementation with shared processing logic.
"""

from .klines_adapter import GateioKlinesAdapter, create_gateio_klines_adapter

__all__ = ['GateioKlinesAdapter', 'create_gateio_klines_adapter']