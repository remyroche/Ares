# src/analyst/decision_aggregator.py

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np

from src.analyst.regime_runtime import get_current_regime_info
from src.utils.logger import system_logger


def _safe_get(d: dict, k: Any, default: float = 0.0) -> float:
    try:
        v = d.get(k, default)
        return float(v)
    except Exception:
        return float(default)


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    vals = np.array([max(0.0, float(v)) for v in weights.values()], dtype=float)
    s = float(vals.sum())
    if s <= 0:
        return {k: 0.0 for k in weights}
    return {k: float(v) / s for k, v in zip(weights.keys(), vals, strict=False)}


def aggregate_weights(
    exchange: str,
    symbol: str,
    timeframe: str,
    specialized_candidates: dict[int, dict[str, float]] | None = None,
    generalist_score: float | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raise NotImplementedError("Removed unused function: aggregate_weights")
