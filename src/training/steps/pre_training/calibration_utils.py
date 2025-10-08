"""Calibration utilities shared by the pre-training pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import brier_score_loss

# Import common utilities for enhanced math operations and validation
from src.utils.common_operations import safe_divide, safe_mean, safe_std, validate_finite, validate_positive, safe_correlation
from src.utils.tprint import tprint, tprint_debug, tprint_warning, tprint_error


def _bin_edges(n_bins: int) -> np.ndarray:
    n_bins = max(1, int(n_bins))
    return np.linspace(0.0, 1.0, n_bins + 1)


def compute_classification_calibration(
    y_true: Sequence[Any],
    y_pred_proba: np.ndarray,
    *,
    classes: Optional[Sequence[Any]] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute Brier score, reliability diagram and ECE for classification outputs."""

    tprint_debug(f"📊 Computing classification calibration with {n_bins} bins")

    probabilities = np.asarray(y_pred_proba, dtype=float)
    if probabilities.ndim == 1:
        probabilities = np.column_stack([1.0 - probabilities, probabilities])

    n_samples, n_classes = probabilities.shape
    if n_samples == 0:
        return {
            'brier_score': 0.0,
            'brier_per_class': {},
            'expected_calibration_error': 0.0,
            'reliability_diagram': {},
            'n_bins': n_bins,
        }

    if classes is None:
        classes_array = np.arange(n_classes)
    else:
        if len(classes) != n_classes:
            raise ValueError("Number of classes must match probability columns")
        classes_array = np.asarray(list(classes))

    y_array = np.asarray(list(y_true))
    class_to_index = {cls: idx for idx, cls in enumerate(classes_array)}

    per_class_brier: Dict[str, float] = {}
    reliability_diagram: Dict[str, List[Dict[str, Optional[float]]]] = {}
    edges = _bin_edges(n_bins)

    for cls, class_index in class_to_index.items():
        true_binary = (y_array == cls).astype(int)
        per_class_brier[str(cls)] = float(brier_score_loss(true_binary, probabilities[:, class_index]))

        bin_records: List[Dict[str, Optional[float]]] = []
        for lower, upper in zip(edges[:-1], edges[1:]):
            if upper == 1.0:
                mask = (probabilities[:, class_index] >= lower) & (probabilities[:, class_index] <= upper)
            else:
                mask = (probabilities[:, class_index] >= lower) & (probabilities[:, class_index] < upper)
            count = int(mask.sum())
            if count > 0:
                mean_pred = float(probabilities[mask, class_index].mean())
                observed = float(true_binary[mask].mean())
            else:
                mean_pred = None
                observed = None
            bin_records.append(
                {
                    'bin_lower': float(lower),
                    'bin_upper': float(upper),
                    'count': count,
                    'mean_predicted': mean_pred,
                    'observed_frequency': observed,
                }
            )
        reliability_diagram[str(cls)] = bin_records

    # Use safe mean calculation from common utilities
    overall_brier = safe_mean(list(per_class_brier.values())) if per_class_brier else 0.0

    predicted_class_index = np.argmax(probabilities, axis=1)
    predicted_confidence = probabilities[np.arange(n_samples), predicted_class_index]
    predicted_labels = classes_array[predicted_class_index]
    correct_predictions = (predicted_labels == y_array)

    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            mask = (predicted_confidence >= lower) & (predicted_confidence <= upper)
        else:
            mask = (predicted_confidence >= lower) & (predicted_confidence < upper)
        count = mask.sum()
        if count == 0:
            continue
        accuracy = correct_predictions[mask].mean()
        avg_confidence = predicted_confidence[mask].mean()
        ece += abs(avg_confidence - accuracy) * (count / n_samples)

    # Enhanced logging of calibration results
    tprint_info(f"✅ Classification calibration computed: Brier={overall_brier:.4f}, ECE={ece:.4f}")
    tprint_debug(f"📈 Per-class Brier scores: {per_class_brier}")
    tprint_debug(f"🔢 Processed {n_samples} samples across {n_classes} classes")

    return {
        'brier_score': overall_brier,
        'brier_per_class': per_class_brier,
        'expected_calibration_error': float(ece),
        'reliability_diagram': reliability_diagram,
        'n_bins': n_bins,
    }


def evaluate_conformal_interval(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    coverage_target: float,
) -> Dict[str, Any]:
    """Compute conformal interval statistics from residuals."""

    if not (0.0 < coverage_target < 1.0):
        raise ValueError("coverage_target must be between 0 and 1")

    y_array = np.asarray(y_true, dtype=float)
    predictions = np.asarray(y_pred, dtype=float)
    if y_array.shape != predictions.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    residuals = np.abs(y_array - predictions)
    quantile = float(np.quantile(residuals, coverage_target))

    lower = predictions - quantile
    upper = predictions + quantile
    coverage = float(np.mean((y_array >= lower) & (y_array <= upper)))

    return {
        'coverage': coverage,
        'coverage_target': float(coverage_target),
        'coverage_met': bool(coverage >= coverage_target),
        'interval_width': float(np.mean(upper - lower)),
        'residual_quantile': quantile,
    }
