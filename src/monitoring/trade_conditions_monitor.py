#!/usr/bin/env python3
"""
Trade Conditions Monitor (minimal scaffold)

Scaffolding for monitoring trade conditions and decisions.
"""


from enum import Enum



class TradeAction(Enum):
    ENTER_LONG , "enter_long"
ENTER_SHORT = "enter_short"
EXIT_LONG = "exit_long"
EXIT_SHORT = "exit_short"
HOLD = "hold"
CANCEL_ORDER = "cancel_order"


