"""Utilities for loading regime feature quality thresholds from configuration."""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore

_DEFAULTS: Dict[str, Any] = {
    "quality_thresholds": {
        "min_regime_persistence": 0.2,
        "max_feature_noise_ratio": 1.2,
        "min_temporal_stability": 0.1,
    },
    "filter_thresholds": {
        "do_not_drop_patterns": [
            "regime_leaf_interaction__*",
            "regime_leaf_interaction_transition*__*",
        ],
        "variance": {"min_variance": 1.0e-8},
        "winsorization": {"lower_quantile": 0.01, "upper_quantile": 0.99},
        "correlation": {"threshold": 0.95},
        "quality": {
            "min_persistence": 0.2,
            "max_noise_ratio": 1.2,
            "min_stability": 0.1,
        },
    },
}

def _config_path(override_path: Optional[Path] = None) -> Path:
    """Return the path to the regime feature configuration file."""
    if override_path is not None:
        return override_path
    root = Path(__file__).resolve().parents[2]
    preferred = root / "config" / "features" / "regime_features.yaml"
    if preferred.is_file():
        return preferred
    return root / "config" / "regime_features.yaml"

def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``updates`` into ``base`` and return the result."""
    for key, value in updates.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base

@lru_cache(maxsize=None)
def get_regime_feature_thresholds(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load regime feature thresholds from YAML configuration.

    Parameters
    ----------
    config_path:
        Optional override path for the configuration file.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing quality and filtering thresholds with sensible
        defaults when the configuration file cannot be loaded.
    """

    path = _config_path(Path(config_path) if config_path else None)
    data: Dict[str, Any] = copy.deepcopy(_DEFAULTS)

    if path.is_file() and yaml is not None:
        try:
            with path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            data = _deep_update(data, loaded)
        except Exception:
            # Fall back to defaults if the file cannot be parsed. Errors are
            # intentionally swallowed to keep callers resilient to configuration
            # issues during experimentation.
            pass

    return data

__all__ = ["get_regime_feature_thresholds"]
