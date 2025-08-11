# src/utils/confidence.py


# Empirically derived baseline and range for dual confidence normalization
DUAL_CONF_BASELINE = 0.216
DUAL_CONF_RANGE = 0.784


def normalize_dual_confidence(
    analyst_confidence: float,
    tactician_confidence: float,
) -> tuple[float, float]:
    """Compute dual and normalized confidence in [0,1].

    Returns (dual_confidence, normalized_confidence).
    """
    dual = analyst_confidence * (tactician_confidence**2)
    normalized = max(0.0, min(1.0, (dual - DUAL_CONF_BASELINE) / DUAL_CONF_RANGE))
    return dual, normalized
