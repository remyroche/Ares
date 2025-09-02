# src/utils/confidence.py

import numpy as np
from collections.abc import Iterable
from typing import Any, Dict, List, Tuple, Union
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Empirically derived baseline and range for dual confidence normalization
DUAL_CONF_BASELINE = 0.216
DUAL_CONF_RANGE = 0.784


def normalize_dual_confidence(
    analyst_confidence: float, 
    tactician_confidence: float
) -> Tuple[float, float]:
    """
    Normalize dual confidence scores using empirically derived parameters.
    
    Args:
        analyst_confidence: Analyst's confidence score [0, 1]
        tactician_confidence: Tactician's confidence score [0, 1]
        
    Returns:
        Tuple of (dual_confidence, normalized_confidence)
    """
    try:
        # Clamp inputs to valid range
        analyst_confidence = _clamp01(analyst_confidence)
        tactician_confidence = _clamp01(tactician_confidence)
        
        # Calculate dual confidence using quadratic weighting
        dual = analyst_confidence * (tactician_confidence ** 2)
        
        # Normalize using empirical baseline and range
        normalized = max(0.0, min(1.0, (dual - DUAL_CONF_BASELINE) / DUAL_CONF_RANGE))
        
        logger.info(
            "dual_confidence_compute",
            extra={
                "analyst": float(analyst_confidence),
                "tactician": float(tactician_confidence),
                "dual": float(dual),
                "normalized": float(normalized),
            }
        )
        
        return dual, normalized
        
    except Exception as e:
        logger.error(f"Error in normalize_dual_confidence: {e}")
        return 0.0, 0.0


def _clamp01(value: float) -> float:
    """Clamp value to [0, 1] range."""
    return 0.0 if value < 0.0 else min(value, 1.0)


def direction_to_sign(direction: str) -> int:
    """
    Convert direction string to numerical sign.
    
    Args:
        direction: Direction string (e.g., "LONG", "SHORT", "BUY", "SELL")
        
    Returns:
        1 for bullish, -1 for bearish, 0 for neutral/hold
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
    models: Iterable[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Aggregate confidences across multiple models with direction awareness.
    
    Logic:
    - If models point in the same direction, confidences are added then averaged
    - If models point in opposite directions, confidences are subtracted then averaged
    - For N models, compute the signed average: sum(sign_i * conf_i * w_i) / sum(w_i)
    - Result direction sign determines LONG/SHORT; magnitude in [0, 1] is confidence
    
    Args:
        models: Iterable of {"direction": str, "confidence": float, "weight"?: float}
        
    Returns:
        dict with {"direction": "LONG"|"SHORT"|"HOLD", "confidence": float, 
                  "signed_value": float, "count": int}
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
        return {
            "direction": "HOLD", 
            "confidence": 0.0, 
            "signed_value": 0.0, 
            "count": 0
        }
    
    # Weighted average by total weight
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
    profit_prediction: float,
    predicted_price: float,
    current_price: float,
    direction_threshold: float = 0.1,
    profit_threshold: float = 0.001,
    price_threshold: float = 0.01,
    min_ensemble_confidence: float = 0.3
) -> Dict[str, Any]:
    """
    Calculate confidence scores for multi-output model predictions.
    
    Args:
        direction_probability: Probability of upward movement [0, 1]
        profit_prediction: Predicted profit/loss
        predicted_price: Predicted future price
        current_price: Current market price
        direction_threshold: Minimum direction confidence threshold
        profit_threshold: Minimum profit confidence threshold
        price_threshold: Minimum price movement threshold
        min_ensemble_confidence: Minimum ensemble confidence threshold
        
    Returns:
        Dictionary with confidence scores and predictions
    """
    try:
        # 1. Direction confidence
        base_direction_confidence = abs(direction_probability - 0.5) * 2  # Convert to 0-1 scale
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
        simple_confidence = (direction_confidence + profit_confidence + price_confidence) / 3.0
        
        # 5. Apply minimum ensemble confidence threshold
        final_confidence = simple_confidence if simple_confidence >= min_ensemble_confidence else 0.0
        final_confidence = _clamp01(final_confidence)
        
        return {
            'direction_confidence': _clamp01(direction_confidence),
            'profit_confidence': _clamp01(profit_confidence),
            'price_confidence': _clamp01(price_confidence),
            'simple_confidence': _clamp01(simple_confidence),
            'final_confidence': final_confidence,
            'direction_prediction': direction_probability,
            'profit_prediction': profit_prediction,
            'predicted_price': predicted_price
        }
        
    except Exception as e:
        logger.error(f"Error in calculate_multi_output_confidence: {e}")
        return {
            'direction_confidence': 0.0,
            'profit_confidence': 0.0,
            'price_confidence': 0.0,
            'simple_confidence': 0.0,
            'final_confidence': 0.0,
            'direction_prediction': 0.5,
            'profit_prediction': 0.0,
            'predicted_price': current_price
        }


def calculate_multi_output_confidence_batch(
    direction_probabilities: np.ndarray,
    profit_predictions: np.ndarray,
    predicted_prices: np.ndarray,
    current_prices: np.ndarray,
    direction_threshold: float = 0.1,
    profit_threshold: float = 0.001,
    price_threshold: float = 0.01,
    min_ensemble_confidence: float = 0.3
) -> Dict[str, np.ndarray]:
    """
    Calculate confidence scores for batch of multi-output model predictions.
    
    Args:
        direction_probabilities: Array of direction probabilities [0, 1]
        profit_predictions: Array of predicted profits/losses
        predicted_prices: Array of predicted future prices
        current_prices: Array of current market prices
        direction_threshold: Minimum direction confidence threshold
        profit_threshold: Minimum profit confidence threshold
        price_threshold: Minimum price movement threshold
        min_ensemble_confidence: Minimum ensemble confidence threshold
        
    Returns:
        Dictionary with confidence score arrays and predictions
    """
    try:
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
        simple_confidence = (direction_confidence + profit_confidence + price_confidence) / 3.0
        
        # 5. Apply minimum ensemble confidence threshold
        final_confidence = np.where(
            simple_confidence >= min_ensemble_confidence,
            simple_confidence,
            0.0
        )
        final_confidence = np.clip(final_confidence, 0, 1)
        
        return {
            'direction_confidence': np.clip(direction_confidence, 0, 1),
            'profit_confidence': np.clip(profit_confidence, 0, 1),
            'price_confidence': np.clip(price_confidence, 0, 1),
            'simple_confidence': np.clip(simple_confidence, 0, 1),
            'final_confidence': final_confidence,
            'direction_prediction': direction_probabilities,
            'profit_prediction': profit_predictions,
            'predicted_price': predicted_prices
        }
        
    except Exception as e:
        logger.error(f"Error in calculate_multi_output_confidence_batch: {e}")
        # Return zero arrays with same shape as inputs
        shape = direction_probabilities.shape
        return {
            'direction_confidence': np.zeros(shape),
            'profit_confidence': np.zeros(shape),
            'price_confidence': np.zeros(shape),
            'simple_confidence': np.zeros(shape),
            'final_confidence': np.zeros(shape),
            'direction_prediction': np.full(shape, 0.5),
            'profit_prediction': np.zeros(shape),
            'predicted_price': current_prices
        }


def get_confidence_threshold_signals(
    confidence_scores: Dict[str, Union[float, np.ndarray]],
    threshold: float = 0.5
) -> np.ndarray:
    """
    Generate trading signals based on confidence threshold.
    
    Args:
        confidence_scores: Dictionary with confidence scores and predictions
        threshold: Minimum confidence threshold for signal generation
        
    Returns:
        Array of signals: 1 for long, -1 for short, 0 for no signal
    """
    try:
        final_confidence = confidence_scores['final_confidence']
        direction_prediction = confidence_scores['direction_prediction']
        
        # Generate signals based on confidence threshold
        signals = np.where(
            (final_confidence >= threshold) & (direction_prediction == 1),
            1,  # Long signal
            np.where(
                (final_confidence >= threshold) & (direction_prediction == 0),
                -1,  # Short signal
                0  # No signal
            )
        )
        
        return signals
        
    except Exception as e:
        logger.error(f"Error in get_confidence_threshold_signals: {e}")
        # Return no signals array
        if isinstance(confidence_scores.get('final_confidence'), np.ndarray):
            return np.zeros_like(confidence_scores['final_confidence'])
        return np.array([0])
