from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


LABEL_CONFIG_VERSION = "1.0"


def build_label_config(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
    params: Dict[str, Any],
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a canonical label configuration dictionary.

    The goal is to have a stable, JSON-serializable representation of the
    labeling setup that can be hashed into a compact identifier and stored
    alongside artifacts.
    """

    config: Dict[str, Any] = {
        "version": LABEL_CONFIG_VERSION,
        "symbol": str(symbol),
        "exchange": str(exchange),
        "timeframe": str(timeframe),
        "direction": str(direction),
        "params": {},
    }

    # Normalize parameter values into simple JSON-serializable types
    normalized_params: Dict[str, Any] = {}
    for key in sorted(params.keys()):
        value = params[key]
        if isinstance(value, (int, float, str, bool)) or value is None:
            normalized_params[key] = value
        else:
            # Fallback to string representation for exotic types
            normalized_params[key] = str(value)

    config["params"] = normalized_params

    if extra:
        config["extra"] = extra

    return config


def compute_label_config_id(config: Dict[str, Any]) -> str:
    """Compute a stable identifier for a label configuration.

    Uses MD5 over the JSON representation with sorted keys so that the same
    logical configuration always yields the same identifier.
    """

    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()
