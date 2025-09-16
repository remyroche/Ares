"""
Signal Generation Module

Analyst and tactician signal integration for trading decisions.
Combines signals from multiple sources with regime-aware weighting.
"""

from .analyst_signals import AnalystSignalGenerator
from .tactician_signals import TacticianSignalGenerator
from .signal_combiner import SignalCombiner
from .signal_validator import SignalValidator

__all__ = [
    "AnalystSignalGenerator",
    "TacticianSignalGenerator",
    "SignalCombiner",
    "SignalValidator"
]