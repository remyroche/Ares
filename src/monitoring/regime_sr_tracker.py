#!/usr/bin/env python3
"""
Regime and Support/Resistance Tracker (minimal scaffold)

Scaffolding for regime detection and S/R tracking.
"""


from enum import Enum



class RegimeType(Enum):
    BULL_TREND , "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


