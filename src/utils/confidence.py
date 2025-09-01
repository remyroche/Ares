# src/utils/confidence.py

import numpy as np

# Empirically derived baseline and range for dual confidence normalization
DUAL_CONF_BASELINE = 0.216
DUAL_CONF_RANGE = 0.784



from collections.abc import Iterable
from typing import Any


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else min(value, 1.0)


def direction_to_sign(direction: str) -> int:
    """Map a textual direction to a signed integer.

    LONG/BUY/UP/BULL(ISH) -> +1
    SHORT/SELL/DOWN/BEAR(ISH) -> -1
    others (e.g., HOLD/UNKNOWN) -> 0
    """
    if not isinstance(direction, str):
        return 0
    d = direction.strip().upper()
    if d in {"LONG", "BUY", "UP", "BULL", "BULLISH"}:
        return 1
    if d in {"SHORT", "SELL", "DOWN", "BEAR", "BEARISH"}:
        return -1
    return 0




