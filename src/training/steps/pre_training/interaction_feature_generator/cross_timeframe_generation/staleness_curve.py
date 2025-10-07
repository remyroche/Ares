"""Shared staleness curve utilities.

Provides a single source of truth for how staleness curves are parameterised
across the cross timeframe pipeline.  The calculator caches generated curves so
that Phase-1 scoring, EHU/RIH assignment and any other component can request
consistent staleness metrics without diverging implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
import logging

import numpy as np

# Import tprint for enhanced logging
try:
    from src.utils.tprint import tprint, tprint_debug, tprint_success
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)


@dataclass(frozen=True)
class StalenessSummary:
    """Summary metrics for a staleness curve.

    Attributes
    ----------
    average:
        Average staleness over the lookback horizon.
    minimum:
        Minimum staleness observed across the curve.
    at_base:
        Staleness measured at the base timeframe increment.
    maximum:
        Maximum staleness observed across the curve.
    metadata:
        Additional metadata describing the curve generation parameters.
    """

    average: float
    minimum: float
    at_base: float
    maximum: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StalenessCurve:
    """Container for curve parameters and derived metrics."""

    feature_type: str
    family: str
    curve_params: Dict[str, Any]
    staleness_values: Dict[int, float]
    summary: StalenessSummary


class StalenessCurveCalculator:
    """Generates and caches staleness curves for HTF features."""

    def __init__(self, default_base_timeframe: int = 5):
        self.logger = logging.getLogger(__name__)
        self.default_base_timeframe = default_base_timeframe
        self._cache: Dict[Tuple[str, str, int, int], StalenessCurve] = {}

    def calculate_staleness_curve(
        self,
        feature_name: str,
        family: str,
        lookback: int,
        base_timeframe: Optional[int] = None,
    ) -> StalenessCurve:
        """Calculate (or retrieve) the staleness curve for a feature."""

        base_tf = base_timeframe or self.default_base_timeframe
        cache_key = (feature_name, family, lookback, base_tf)

        if cache_key in self._cache:
            return self._cache[cache_key]

        curve_params = self._get_curve_params(family, lookback)

        staleness_values: Dict[int, float] = {}
        for lag_minutes in range(0, lookback + 1, base_tf):
            staleness = self._calculate_staleness_at_lag(
                lag_minutes, curve_params, base_tf
            )
            staleness_values[lag_minutes] = staleness

        summary = self._summarise_curve(staleness_values, base_tf, family, lookback)

        curve = StalenessCurve(
            feature_type=feature_name,
            family=family,
            curve_params=curve_params,
            staleness_values=staleness_values,
            summary=summary,
        )

        self._cache[cache_key] = curve
        return curve

    def get_summary(
        self,
        feature_name: str,
        family: str,
        lookback: int,
        base_timeframe: Optional[int] = None,
    ) -> StalenessSummary:
        """Convenience accessor for just the summary metrics."""

        curve = self.calculate_staleness_curve(
            feature_name, family, lookback, base_timeframe
        )
        return curve.summary

    def _get_curve_params(self, family: str, lookback: int) -> Dict[str, Any]:
        """Determine the curve parameterisation for a feature family."""

        if family in ["trend_level_vol"]:
            tau = lookback / 2
            return {"tau": tau, "type": "exponential"}

        if family == "anchors":
            return {"type": "step", "session_reset": True}

        if family == "oscillators":
            tau = lookback / 1.5
            return {"tau": tau, "type": "exponential"}

        if family == "liquidity_micro":
            tau = lookback / 2.5
            return {"tau": tau, "type": "exponential"}

        tau = lookback / 2
        return {"tau": tau, "type": "exponential"}

    def _calculate_staleness_at_lag(
        self, lag_minutes: int, curve_params: Dict[str, Any], base_timeframe: int
    ) -> float:
        """Calculate staleness at a specific time lag."""

        curve_type = curve_params.get("type", "exponential")

        if curve_type == "exponential":
            tau = curve_params["tau"]
            delta_t = lag_minutes
            return float(1 - np.exp(-delta_t / tau))

        if curve_type == "step":
            session_reset = curve_params.get("session_reset", False)
            if session_reset:
                if lag_minutes <= 30:
                    return 0.0
                return float(min(1.0, (lag_minutes - 30) / 60))
            return float(min(1.0, lag_minutes / max(base_timeframe * 24, 1)))

        return float(min(1.0, lag_minutes / max(base_timeframe * 24, 1)))

    def _summarise_curve(
        self,
        staleness_values: Dict[int, float],
        base_timeframe: int,
        family: str,
        lookback: int,
    ) -> StalenessSummary:
        """Create summary metrics for a generated curve."""

        if staleness_values:
            values = np.array(list(staleness_values.values()), dtype=float)
            average = float(np.mean(values))
            minimum = float(np.min(values))
            maximum = float(np.max(values))
        else:
            average = minimum = maximum = 0.0

        at_base = float(staleness_values.get(base_timeframe, minimum))

        metadata = {
            "base_timeframe": base_timeframe,
            "family": family,
            "lookback": lookback,
        }

        return StalenessSummary(
            average=average,
            minimum=minimum,
            at_base=at_base,
            maximum=maximum,
            metadata=metadata,
        )

