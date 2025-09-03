from __future__ import annotations

import numpy as np

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
                },
            )
    except Exception:
        pass
    return dual, normalized


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
    final_direction = (
        "LONG" if signed_avg > 0 else ("SHORT" if signed_avg < 0 else "HOLD")
    )
    final_confidence = _clamp01(abs(signed_avg))

    return {
        "direction": final_direction,
        "confidence": final_confidence,
        "signed_value": signed_avg,
        "count": count_active,
    }


def calculate_multi_output_confidence(
    direction_probability: float,
    direction_prediction: int,
    profit_prediction: float,
    current_price: float,
    predicted_price: float,
    direction_threshold: float = 0.6,
    profit_threshold: float = 0.001,
    price_threshold: float = 0.005,
    min_ensemble_confidence: float = 0.7,
) -> dict[str, Any]:
    """Calculate simplified confidence score for multi-output predictions.

    Since all predictions come from the same model, uses simple average
    instead of arbitrary weighting. No risk adjustments applied.

    Args:
        direction_probability: Model probability for direction (0-1)
        direction_prediction: Binary direction prediction (0/1)
        profit_prediction: Predicted profit percentage
        current_price: Current price level
        predicted_price: Predicted price level
        direction_threshold: Minimum direction confidence
        profit_threshold: Minimum expected profit
        price_threshold: Minimum price movement
        min_ensemble_confidence: Minimum ensemble confidence

    Returns:
        Dictionary containing confidence scores and predictions
    """
    # 1. Direction confidence
    base_direction_confidence = (
        abs(direction_probability - 0.5) * 2
    )  # Convert to 0-1 scale
    threshold_mask = base_direction_confidence >= direction_threshold
    direction_confidence = base_direction_confidence * threshold_mask

    # Add uncertainty penalty for predictions near 0.5
    uncertainty_penalty = np.exp(-10 * abs(direction_probability - 0.5))
    direction_confidence *= uncertainty_penalty

    # 2. Profit confidence
    profit_abs = abs(profit_prediction)
    base_profit_confidence = np.tanh(profit_abs * 100)  # Sigmoid-like function
    profit_threshold_mask = profit_abs >= profit_threshold
    profit_confidence = base_profit_confidence * profit_threshold_mask

    # 3. Price confidence
    price_movement = abs(predicted_price - current_price) / current_price
    base_price_confidence = np.tanh(price_movement * 50)  # Sigmoid-like function
    price_threshold_mask = price_movement >= price_threshold
    price_confidence = base_price_confidence * price_threshold_mask

    # 4. Simple average (no weighting since all from same model)
    simple_confidence = (
        direction_confidence + profit_confidence + price_confidence
    ) / 3.0

    # 5. Apply minimum ensemble confidence threshold
    final_confidence = (
        simple_confidence if simple_confidence >= min_ensemble_confidence else 0.0
    )
    final_confidence = _clamp01(final_confidence)

    return {
        "direction_confidence": _clamp01(direction_confidence),
        "profit_confidence": _clamp01(profit_confidence),
        "price_confidence": _clamp01(price_confidence),
        "simple_confidence": _clamp01(simple_confidence),
        "final_confidence": final_confidence,
        "direction_prediction": direction_prediction,
        "profit_prediction": profit_prediction,
        "predicted_price": predicted_price,
    }


def calculate_multi_output_confidence_batch(
    direction_probabilities: np.ndarray,
    direction_predictions: np.ndarray,
    profit_predictions: np.ndarray,
    current_prices: np.ndarray,
    predicted_prices: np.ndarray,
    direction_threshold: float = 0.6,
    profit_threshold: float = 0.001,
    price_threshold: float = 0.005,
    min_ensemble_confidence: float = 0.7,
) -> dict[str, np.ndarray]:
    """Calculate simplified confidence scores for batch of multi-output predictions.

    Vectorized version of calculate_multi_output_confidence for numpy arrays.

    Args:
        direction_probabilities: Model probabilities for direction (0-1)
        direction_predictions: Binary direction predictions (0/1)
        profit_predictions: Predicted profit percentages
        current_prices: Current price levels
        predicted_prices: Predicted price levels
        direction_threshold: Minimum direction confidence
        profit_threshold: Minimum expected profit
        price_threshold: Minimum price movement
        min_ensemble_confidence: Minimum ensemble confidence

    Returns:
        Dictionary containing confidence scores and predictions arrays
    """
    # 1. Direction confidence
    base_direction_confidence = np.abs(direction_probabilities - 0.5) * 2
    threshold_mask = base_direction_confidence >= direction_threshold
    direction_confidence = base_direction_confidence * threshold_mask

    # Add uncertainty penalty
    uncertainty_penalty = np.exp(-10 * np.abs(direction_probabilities - 0.5))
    direction_confidence *= uncertainty_penalty

    # 2. Profit confidence
    profit_abs = np.abs(profit_predictions)
    base_profit_confidence = np.tanh(profit_abs * 100)
    profit_threshold_mask = profit_abs >= profit_threshold
    profit_confidence = base_profit_confidence * profit_threshold_mask

    # 3. Price confidence
    price_movement = np.abs(predicted_prices - current_prices) / current_prices
    base_price_confidence = np.tanh(price_movement * 50)
    price_threshold_mask = price_movement >= price_threshold
    price_confidence = base_price_confidence * price_threshold_mask

    # 4. Simple average
    simple_confidence = (
        direction_confidence + profit_confidence + price_confidence
    ) / 3.0

    # 5. Apply minimum ensemble confidence threshold
    final_confidence = np.where(
        simple_confidence >= min_ensemble_confidence, simple_confidence, 0.0
    )
    final_confidence = np.clip(final_confidence, 0, 1)

    return {
        "direction_confidence": np.clip(direction_confidence, 0, 1),
        "profit_confidence": np.clip(profit_confidence, 0, 1),
        "price_confidence": np.clip(price_confidence, 0, 1),
        "simple_confidence": np.clip(simple_confidence, 0, 1),
        "final_confidence": final_confidence,
        "direction_prediction": direction_predictions,
        "profit_prediction": profit_predictions,
        "predicted_price": predicted_prices,
    }


def get_confidence_threshold_signals(
    confidence_scores: dict[str, np.ndarray], threshold: float = 0.7
) -> np.ndarray:
    """Get trading signals based on confidence threshold.

    Args:
        confidence_scores: Dictionary of confidence scores
        threshold: Minimum confidence threshold

    Returns:
        Binary trading signals (1 for long, -1 for short, 0 for no trade)
    """
    final_confidence = confidence_scores["final_confidence"]
    direction_prediction = confidence_scores["direction_prediction"]

    # Generate signals based on confidence threshold
    return np.where(
        (final_confidence >= threshold) & (direction_prediction == 1),
        1,  # Long signal
        np.where(
            (final_confidence >= threshold) & (direction_prediction == 0), -1, 0
        ),  # Short signal  # No signal
    )
