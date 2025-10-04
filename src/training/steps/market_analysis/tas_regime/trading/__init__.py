"""Trading integration helpers for TAS.

Only the signal generation primitives live in this package; the trading
engine expects external components for position sizing, risk management and
performance tracking to be supplied by higher level orchestration code.
"""

from .trading_engine import TradingEngine, TradingConfig, TradingResult
from .signal_generator import TradingSignalGenerator, SignalConfig

__all__ = [
    'TradingEngine', 'TradingConfig', 'TradingResult',
    'TradingSignalGenerator', 'SignalConfig',
]
