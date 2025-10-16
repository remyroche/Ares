"""Central registry for NAS/TAS clustering calibration statistics and thresholds.

This module stores rolling calibration statistics so that multiple components can
resolve consistent threshold values without relying on hard-coded defaults.
It also provides utilities for resetting and updating the calibration state for
tests and runtime calibration passes.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

_DEFAULT_CALIBRATION: Dict[str, Any] = {
    "history": {
        "persistence": [],
        "noise_ratio": [],
        "temporal_stability": [],
        "confidence": [],
        "silhouette": [],
        "davies_bouldin": [],
        "cv_score": [],
    },
    "statistics": {
        "quantiles": {},
        "means": {},
    },
    "quality_thresholds": {
        "min_regime_persistence": 0.7,
        "max_feature_noise_ratio": 0.3,
        "min_temporal_stability": 0.6,
    },
    "confidence_levels": {
        "high": 0.8,
        "medium": 0.6,
        "low": 0.4,
    },
    "metric_thresholds": {
        "silhouette": {
            "excellent": 0.7,
            "good": 0.5,
            "fair": 0.3,
        },
        "davies_bouldin": {
            "excellent": 0.5,
            "good": 1.0,
            "fair": 2.0,
        },
        "cv_score": {
            "excellent": 0.8,
            "good": 0.6,
            "fair": 0.4,
        },
    },
}

_CALIBRATION_STATE: Dict[str, Any] = deepcopy(_DEFAULT_CALIBRATION)

def _merge_nested_dict(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Safely merge nested dictionaries while keeping copies."""

    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged

def reset_quality_calibration() -> None:
    """Reset calibration state to defaults."""

    global _CALIBRATION_STATE
    _CALIBRATION_STATE = deepcopy(_DEFAULT_CALIBRATION)

def update_quality_calibration(calibration: Dict[str, Any] | None) -> None:
    """Update the global calibration registry with new values.

    Args:
        calibration: Calibration payload captured in execution metadata. If
            ``None`` the registry resets to defaults.
    """

    global _CALIBRATION_STATE

    if calibration is None:
        reset_quality_calibration()
        return

    merged = _merge_nested_dict(_DEFAULT_CALIBRATION, calibration)
    _CALIBRATION_STATE = merged

def get_current_calibration() -> Dict[str, Any]:
    """Return a deepcopy of the current calibration state."""

    return deepcopy(_CALIBRATION_STATE)

def get_quality_thresholds() -> Dict[str, float]:
    """Get calibrated feature-quality thresholds with defaults applied."""

    return deepcopy(_CALIBRATION_STATE.get("quality_thresholds", {}))

def get_confidence_levels() -> Dict[str, float]:
    """Return calibrated confidence levels for reporting."""

    return deepcopy(_CALIBRATION_STATE.get("confidence_levels", {}))

def get_metric_thresholds(metric: str) -> Dict[str, float]:
    """Resolve label thresholds for a metric such as silhouette or CV."""

    metric_thresholds = _CALIBRATION_STATE.get("metric_thresholds", {})
    return deepcopy(metric_thresholds.get(metric, {}))

__all__ = [
    "get_confidence_levels",
    "get_current_calibration",
    "get_metric_thresholds",
    "get_quality_thresholds",
    "reset_quality_calibration",
    "update_quality_calibration",
]
