# src/utils/confidence.py


# Empirically derived baseline and range for dual confidence normalization
DUAL_CONF_BASELINE = 0.216
DUAL_CONF_RANGE = 0.784


def normalize_dual_confidence(
    analyst_confidence: float,
    tactician_confidence: float,
    logger=None,
) -> tuple[float, float]:
    """Compute dual and normalized confidence in [0,1].

    Returns (dual_confidence, normalized_confidence).
    """
    dual = analyst_confidence * (tactician_confidence**2)
    normalized = max(0.0, min(1.0, (dual - DUAL_CONF_BASELINE) / DUAL_CONF_RANGE))
    try:
        if logger is not None:
            logger.info(
                {
                    "msg": "dual_confidence_compute",
                    "analyst": float(analyst_confidence),
                    "tactician": float(tactician_confidence),
                    "dual": float(dual),
                    "normalized": float(normalized),
                }
            )
    except Exception:
        pass
    return dual, normalized

from typing import Any, Iterable


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


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


def aggregate_directional_confidences(
    models: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate confidences across multiple models with direction-awareness.

    Logic:
    - If models point in the same direction, confidences are added then averaged
    - If models point in opposite directions, confidences are subtracted then averaged
    - For N models, compute the signed average: sum(sign_i * conf_i * w_i) / sum(w_i)
    - Result direction sign determines LONG/SHORT; magnitude in [0,1] is confidence

    Args:
        models: Iterable of {"direction": str, "confidence": float, "weight"?: float}

    Returns:
        dict with {"direction": "LONG"|"SHORT"|"HOLD", "confidence": float, "signed_value": float, "count": int}
    """
    signed_sum: float = 0.0
    total_weight: float = 0.0
    count_active: int = 0
    for m in models:
        if not isinstance(m, dict):
            continue
        conf = float(m.get("confidence", 0.0))
        conf = _clamp01(conf)
        sign = direction_to_sign(m.get("direction", "HOLD"))
        if sign == 0:
            # Ignore non-directional inputs for aggregation
            continue
        weight = float(m.get("weight", 1.0))
        if weight <= 0.0:
            continue
        signed_sum += sign * conf * weight
        total_weight += weight
        count_active += 1

    if count_active == 0 or total_weight == 0.0:
        return {"direction": "HOLD", "confidence": 0.0, "signed_value": 0.0, "count": 0}

    # Weighted average by total weight (per review suggestion)
    signed_avg = signed_sum / total_weight
    final_direction = "LONG" if signed_avg > 0 else ("SHORT" if signed_avg < 0 else "HOLD")
    final_confidence = _clamp01(abs(signed_avg))

    return {
        "direction": final_direction,
        "confidence": final_confidence,
        "signed_value": signed_avg,
        "count": count_active,
    }
