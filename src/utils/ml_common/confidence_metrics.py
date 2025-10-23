from src.utils.tprint import tprint, tprint_data_format, LogLevel

"""Prediction confidence and calibration metrics for ML models."""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from scipy.optimize import minimize_scalar
from scipy.stats import beta
import logging
import time
from datetime import datetime

# Enhanced dependency management with fast fail
try:
    from ..logger import get_logger
    _LOGGER = get_logger("MLCommon.ConfidenceMetrics")
    tprint("✅ Custom logger available for MLCommon.ConfidenceMetrics")
except Exception as e:
    tprint(f"⚠️ Custom logger not available: {e}. Using standard logging.")
    _LOGGER = logging.getLogger("MLCommon.ConfidenceMetrics")
    _LOGGER.setLevel(logging.INFO)

logger = _LOGGER

def _ensure_probability_matrix(prob_matrix: np.ndarray) -> np.ndarray:
    """Return a 2D probability matrix, coercing binary vectors into two-column form."""
    tprint(f"🔄 Ensuring probability matrix shape for input with shape: {np.asarray(prob_matrix).shape}")
    matrix = np.asarray(prob_matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = np.column_stack([1.0 - matrix, matrix])
        tprint("📊 Converted 1D probability array to 2D binary form")
    elif matrix.ndim != 2:
        tprint(f"❌ Invalid probability matrix dimensions: {matrix.ndim}")
        raise ValueError("Probability data must be 1D or 2D to form a matrix.")
    result = np.clip(matrix, 0.0, 1.0)
    tprint(f"✅ Probability matrix ensured with final shape: {result.shape}")
    return result

def _coerce_probability_array(probabilities: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
    """Normalise probability inputs into a numpy array."""
    tprint(f"🔄 Coercing probability array of type: {type(probabilities)}")
    if isinstance(probabilities, (list, tuple)):
        tprint(f"📊 Processing list/tuple with {len(probabilities)} elements")
        if len(probabilities) == 0:
            tprint("❌ Empty probability list provided")
            raise ValueError("No probability arrays supplied.")
        matrices = [_ensure_probability_matrix(prob) for prob in probabilities]
        try:
            result = np.stack(matrices, axis=1)  # (n_samples, n_outputs, n_classes)
            tprint(f"✅ Stacked multi-output probabilities with shape: {result.shape}")
            return result
        except ValueError as exc:
            tprint(f"❌ Failed to stack probability matrices: {exc}")
            raise ValueError("Inconsistent probability array shapes for multi-output model.") from exc

    array = np.asarray(probabilities)
    tprint(f"📊 Converted to numpy array with shape: {array.shape}, ndim: {array.ndim}")
    if array.ndim == 1:
        tprint("🔄 Processing 1D array")
        return _ensure_probability_matrix(array)
    if array.ndim in (2, 3):
        tprint(f"✅ Returning {array.ndim}D array as float")
        return array.astype(float)
    tprint(f"❌ Unsupported array dimensionality: {array.ndim}")
    raise ValueError("Probability array must be 1D, 2D or 3D.")

def _flatten_probabilities(prob_array: np.ndarray) -> np.ndarray:
    """Flatten probability arrays to 2D for aggregate statistics."""
    tprint(f"🔄 Flattening probability array with shape: {prob_array.shape}")
    if prob_array.ndim == 1:
        tprint("❌ Cannot flatten 1D array - must be at least 2D")
        raise ValueError("Probability array must be at least 2D after coercion.")
    if prob_array.ndim == 2:
        tprint("✅ Array already 2D, returning as-is")
        return prob_array
    if prob_array.ndim == 3:
        n_samples = prob_array.shape[0]
        result = prob_array.reshape(n_samples, -1)
        tprint(f"✅ Flattened 3D array to 2D with shape: {result.shape}")
        return result
    tprint(f"❌ Unsupported array dimensionality: {prob_array.ndim}")
    raise ValueError("Unsupported probability array dimensionality.")

def calculate_confidence_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                is_multi_output: bool = False,
                                output_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive prediction confidence and calibration metrics.

    Args:
        y_true: True labels
        y_pred_proba: Prediction probabilities (shape: n_samples, n_classes)
        is_multi_output: Whether this is a multi-output model
        output_names: List of output names for multi-output models

    Returns:
        Dictionary containing confidence and calibration metrics
    """
    start_time = time.time()
    tprint("🎯 Starting comprehensive confidence metrics calculation...")
    _LOGGER.info("🎯 Starting confidence metrics calculation...")

    # Input validation with detailed logging
    tprint(f"📊 Input validation - y_true shape: {np.asarray(y_true).shape}, is_multi_output: {is_multi_output}")
    if output_names:
        tprint(f"📊 Output names provided: {output_names}")

    if y_pred_proba is None or len(y_pred_proba) == 0:
        tprint("❌ No prediction probabilities available for confidence calculation")
        _LOGGER.error("❌ No prediction probabilities available for confidence calculation")
        return {'error': 'No prediction probabilities available'}

    try:
        tprint("🔄 Processing probability arrays...")
        probability_array = _coerce_probability_array(y_pred_proba)
        flattened_probs = _flatten_probabilities(probability_array)
        tprint(f"📊 Processing {flattened_probs.shape[0]} samples with {flattened_probs.shape[1]} probability columns")
        _LOGGER.debug(
            "📊 Processing %d samples with %d probability columns",
            flattened_probs.shape[0],
            flattened_probs.shape[1],
        )

        # Calculate prediction confidence statistics
        tprint("🔄 Calculating prediction confidence statistics...")
        max_probs = np.max(flattened_probs, axis=1)
        tprint(f"📊 Max probabilities calculated - mean: {np.mean(max_probs):.3f}, std: {np.std(max_probs):.3f}")

        confidence_metrics = {
            'mean_confidence': float(np.mean(max_probs)),
            'std_confidence': float(np.std(max_probs)),
            'min_confidence': float(np.min(max_probs)),
            'max_confidence': float(np.max(max_probs)),
            'median_confidence': float(np.median(max_probs)),
            'high_confidence_pct': float(np.mean(max_probs > 0.8) * 100),
            'low_confidence_pct': float(np.mean(max_probs < 0.6) * 100),
            'medium_confidence_pct': float(np.mean((max_probs >= 0.6) & (max_probs <= 0.8)) * 100)
        }
        tprint(f"✅ Confidence metrics calculated - mean: {confidence_metrics['mean_confidence']:.3f}, high conf: {confidence_metrics['high_confidence_pct']:.1f}%")

        _LOGGER.info(f"📈 Confidence stats - Mean: {confidence_metrics['mean_confidence']:.3f}, "
                    f"Std: {confidence_metrics['std_confidence']:.3f}, "
                    f"High conf: {confidence_metrics['high_confidence_pct']:.1f}%")

        # Calculate calibration metrics
        tprint("🔄 Calculating calibration metrics...")
        _LOGGER.debug("🔄 Calculating calibration metrics...")
        multi_output = is_multi_output or probability_array.ndim == 3
        tprint(f"📊 Multi-output calibration: {multi_output}")
        if multi_output:
            tprint("🔄 Processing multi-output calibration...")
            calibration_metrics = calculate_multi_output_calibration_metrics(
                y_true,
                probability_array,
                output_names,
            )
        else:
            tprint("🔄 Processing single-output calibration...")
            calibration_metrics = calculate_calibration_metrics(
                np.asarray(y_true).ravel(),
                np.asarray(probability_array, dtype=float)
            )
        confidence_metrics.update(calibration_metrics)
        tprint("✅ Calibration metrics calculated")

        # Calculate prediction distribution statistics
        tprint("🔄 Calculating prediction distribution metrics...")
        _LOGGER.debug("📊 Calculating prediction distribution metrics...")
        distribution_metrics = calculate_prediction_distribution(flattened_probs)
        confidence_metrics.update(distribution_metrics)
        tprint("✅ Distribution metrics calculated")

        execution_time = time.time() - start_time
        tprint(f"✅ Confidence metrics calculation completed successfully in {execution_time:.3f}s")
        _LOGGER.info(f"✅ Confidence metrics calculated successfully in {execution_time:.3f}s")

        return confidence_metrics

    except Exception as e:
        execution_time = time.time() - start_time
        tprint(f"❌ Failed to calculate confidence metrics after {execution_time:.3f}s: {e}")
        _LOGGER.error(f"❌ Failed to calculate confidence metrics after {execution_time:.3f}s: {e}")
        try:
            y_true_shape = np.asarray(y_true).shape
            tprint(f"📊 y_true shape: {y_true_shape}")
        except Exception:
            y_true_shape = 'unknown'
            tprint("❌ Could not determine y_true shape")
        try:
            proba_shape = np.asarray(y_pred_proba, dtype=object).shape
            tprint(f"📊 y_pred_proba shape: {proba_shape}")
        except Exception:
            proba_shape = 'unknown'
            tprint("❌ Could not determine y_pred_proba shape")
        _LOGGER.error(f"Input shapes - y_true: {y_true_shape}, y_pred_proba: {proba_shape}")
        return {'error': f'Confidence metrics calculation failed: {e}'}

def calculate_multi_output_calibration_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                             output_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Calculate calibration metrics for multi-output models.

    Args:
        y_true: True labels (shape: n_samples, n_outputs)
        y_pred_proba: Prediction probabilities (shape: n_samples, n_outputs)
        output_names: List of output names

    Returns:
        Dictionary containing multi-output calibration metrics
    """
    tprint("🔄 Starting multi-output calibration metrics calculation...")
    _LOGGER.debug("🔄 Starting multi-output calibration metrics calculation...")

    try:
        y_true_array = np.asarray(y_true)
        tprint(f"📊 y_true_array shape: {y_true_array.shape}")
        if y_true_array.ndim == 1:
            y_true_array = y_true_array.reshape(-1, 1)
            tprint("📊 Reshaped 1D y_true to 2D")
        if y_true_array.ndim != 2:
            tprint(f"❌ Invalid y_true dimensions: {y_true_array.ndim}")
            _LOGGER.error("❌ Multi-output calibration requires 2D target array")
            return {'error': 'Invalid target shape for multi-output calibration'}

        proba_array = np.asarray(y_pred_proba, dtype=float)
        tprint(f"📊 proba_array shape: {proba_array.shape}, ndim: {proba_array.ndim}")
        if proba_array.ndim == 2:
            if proba_array.shape[1] == y_true_array.shape[1]:
                tprint("📊 Converting 2D probabilities to 3D binary form")
                positive = np.clip(proba_array, 0.0, 1.0)
                negative = 1.0 - positive
                proba_array = np.stack((negative, positive), axis=-1)
            else:
                tprint("📊 Reshaping 2D probabilities to 3D")
                proba_array = proba_array.reshape(proba_array.shape[0], 1, proba_array.shape[1])
        elif proba_array.ndim != 3:
            tprint(f"❌ Invalid probability dimensions: {proba_array.ndim}")
            _LOGGER.error("❌ Multi-output calibration requires 2D or 3D probability array")
            return {'error': 'Invalid probability shape for multi-output calibration'}

        n_outputs = proba_array.shape[1]
        tprint(f"📊 Processing {n_outputs} outputs")
        if output_names is None:
            output_names = [f"output_{i+1}" for i in range(n_outputs)]
            tprint(f"📊 Generated default output names: {output_names}")
        if len(output_names) != n_outputs:
            tprint(f"⚠️ Output names count mismatch: {len(output_names)} vs {n_outputs}")
            _LOGGER.warning(
                "⚠️ Output names count (%d) does not match number of outputs (%d); trimming to match.",
                len(output_names),
                n_outputs,
            )
            output_names = output_names[:n_outputs]

        if y_true_array.shape[1] != n_outputs:
            tprint(f"❌ Mismatch between target outputs ({y_true_array.shape[1]}) and probability outputs ({n_outputs})")
            _LOGGER.error(
                "❌ Mismatch between target outputs (%d) and probability outputs (%d)",
                y_true_array.shape[1],
                n_outputs,
            )
            return {'error': 'Mismatch between target outputs and probability outputs'}

        per_output_calibration: Dict[str, Dict[str, Any]] = {}
        overall_metrics: Dict[str, Any] = {}

        tprint(f"🔄 Processing calibration for {n_outputs} outputs...")
        for i in range(n_outputs):
            tprint(f"🔄 Processing output {i+1}/{n_outputs}: {output_names[i]}")
            y_true_output = y_true_array[:, i].reshape(-1)
            y_pred_proba_output = proba_array[:, i, :]
            tprint(f"📊 Output {i+1} shapes - y_true: {y_true_output.shape}, y_pred_proba: {y_pred_proba_output.shape}")

            output_calibration = calculate_calibration_metrics(
                y_true_output,
                y_pred_proba_output
            )

            per_output_calibration[output_names[i]] = output_calibration
            tprint(f"✅ Output {i+1} calibration completed")

            if 'brier_score' in output_calibration:
                overall_metrics[f'{output_names[i]}_brier_score'] = output_calibration['brier_score']
                tprint(f"📊 Added Brier score for {output_names[i]}: {output_calibration['brier_score']:.4f}")
            if 'expected_calibration_error' in output_calibration:
                overall_metrics[f'{output_names[i]}_ece'] = output_calibration['expected_calibration_error']
                tprint(f"📊 Added ECE for {output_names[i]}: {output_calibration['expected_calibration_error']:.4f}")

        tprint("🔄 Aggregating overall metrics...")
        brier_scores = [
            cal['brier_score'] for cal in per_output_calibration.values()
            if 'brier_score' in cal and cal['brier_score'] is not None
        ]
        ece_scores = [
            cal['expected_calibration_error'] for cal in per_output_calibration.values()
            if 'expected_calibration_error' in cal and cal['expected_calibration_error'] is not None
        ]
        tprint(f"📊 Found {len(brier_scores)} Brier scores and {len(ece_scores)} ECE scores")

        overall_metrics['overall_brier_score'] = float(np.mean(brier_scores)) if brier_scores else None
        overall_metrics['overall_ece'] = float(np.mean(ece_scores)) if ece_scores else None

        brier_display = (
            f"{overall_metrics['overall_brier_score']:.4f}"
            if overall_metrics['overall_brier_score'] is not None else "n/a"
        )
        ece_display = (
            f"{overall_metrics['overall_ece']:.4f}"
            if overall_metrics['overall_ece'] is not None else "n/a"
        )
        tprint(f"📈 Multi-output calibration - Overall Brier: {brier_display}, ECE: {ece_display}")
        _LOGGER.info(f"📈 Multi-output calibration - Overall Brier: {brier_display}, ECE: {ece_display}")

        result = {
            'per_output_calibration': per_output_calibration,
            'overall_metrics': overall_metrics,
            'n_outputs': n_outputs,
            'output_names': output_names
        }
        tprint("✅ Multi-output calibration metrics calculation completed")
        return result

    except Exception as e:
        tprint(f"❌ Multi-output calibration metrics calculation failed: {e}")
        _LOGGER.error(f"❌ Multi-output calibration metrics calculation failed: {e}")
        return {'error': str(e)}

def calculate_calibration_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
    """
    Calculate calibration metrics including Brier score and calibration curve.

    Args:
        y_true: True labels
        y_pred_proba: Prediction probabilities

    Returns:
        Dictionary containing calibration metrics
    """
    tprint("🔄 Starting single-output calibration metrics calculation...")
    _LOGGER.debug("🔄 Starting calibration metrics calculation...")

    try:
        y_true_array = np.asarray(y_true).ravel()
        prob_matrix = np.asarray(y_pred_proba, dtype=float)
        tprint(f"📊 Input shapes - y_true: {y_true_array.shape}, y_pred_proba: {prob_matrix.shape}")
        if prob_matrix.ndim != 2:
            tprint(f"❌ Invalid probability matrix dimensions: {prob_matrix.ndim}")
            raise ValueError("Probability matrix must be 2-dimensional.")

        if prob_matrix.shape[1] == 2:
            # Binary classification
            tprint("📊 Processing binary classification calibration...")
            _LOGGER.debug("📊 Processing binary classification calibration...")
            brier_score = brier_score_loss(y_true_array, prob_matrix[:, 1])
            tprint(f"📊 Binary Brier score: {brier_score:.4f}")

            # Calibration curve
            tprint("🔄 Calculating calibration curve...")
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true_array, prob_matrix[:, 1], n_bins=10, strategy='uniform'
            )
            tprint(f"📊 Calibration curve calculated with {len(fraction_of_positives)} bins")

            # Calculate calibration error (ECE - Expected Calibration Error)
            tprint("🔄 Calculating Expected Calibration Error...")
            ece = calculate_expected_calibration_error(y_true_array, prob_matrix[:, 1], n_bins=10)
            tprint(f"📊 ECE: {ece:.4f}")

            calibration_quality = _assess_calibration_quality(brier_score, ece)
            tprint(f"📊 Calibration quality: {calibration_quality}")

            _LOGGER.info(f"📈 Binary calibration - Brier: {brier_score:.4f}, ECE: {ece:.4f}, Quality: {calibration_quality}")

            result = {
                'brier_score': float(brier_score),
                'expected_calibration_error': float(ece),
                'calibration_curve': {
                    'fraction_of_positives': fraction_of_positives.tolist(),
                    'mean_predicted_value': mean_predicted_value.tolist()
                },
                'calibration_quality': calibration_quality
            }
            tprint("✅ Binary calibration metrics completed")
            return result
        else:
            # Multiclass classification - use macro average
            tprint(f"📊 Processing multiclass calibration for {prob_matrix.shape[1]} classes...")
            _LOGGER.debug(f"📊 Processing multiclass calibration for {prob_matrix.shape[1]} classes...")
            brier_scores = []
            for i in range(prob_matrix.shape[1]):
                tprint(f"🔄 Processing class {i+1}/{prob_matrix.shape[1]}")
                class_mask = (y_true_array == i)
                if np.sum(class_mask) > 0:
                    class_brier = brier_score_loss(class_mask, prob_matrix[:, i])
                    brier_scores.append(class_brier)
                    tprint(f"📊 Class {i+1} Brier score: {class_brier:.4f}")
                else:
                    tprint(f"⚠️ Class {i+1} has no samples, skipping")

            mean_brier_score = np.mean(brier_scores) if brier_scores else 0.0
            calibration_quality = _assess_calibration_quality(mean_brier_score, None)
            tprint(f"📊 Multiclass mean Brier score: {mean_brier_score:.4f}, Quality: {calibration_quality}")

            _LOGGER.info(f"📈 Multiclass calibration - Mean Brier: {mean_brier_score:.4f}, Quality: {calibration_quality}")

            result = {
                'brier_score': float(mean_brier_score),
                'brier_scores_per_class': [float(score) for score in brier_scores],
                'calibration_quality': calibration_quality
            }
            tprint("✅ Multiclass calibration metrics completed")
            return result

    except Exception as e:
        tprint(f"❌ Failed to calculate calibration metrics: {e}")
        _LOGGER.error(f"❌ Failed to calculate calibration metrics: {e}")
        return {
            'brier_score': None,
            'expected_calibration_error': None,
            'calibration_quality': 'unknown'
        }

def calculate_expected_calibration_error(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for positive class
        n_bins: Number of bins for calibration curve

    Returns:
        Expected Calibration Error
    """
    tprint(f"🔄 Calculating ECE with {n_bins} bins...")
    _LOGGER.debug(f"🔄 Calculating ECE with {n_bins} bins...")

    try:
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        tprint(f"📊 Bin boundaries: {bin_boundaries}")

        ece = 0
        for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                bin_ece = np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                ece += bin_ece
                tprint(f"📊 Bin {i+1}: [{bin_lower:.2f}, {bin_upper:.2f}] - samples: {np.sum(in_bin)}, prop: {prop_in_bin:.3f}, acc: {accuracy_in_bin:.3f}, conf: {avg_confidence_in_bin:.3f}, ece: {bin_ece:.4f}")
            else:
                tprint(f"📊 Bin {i+1}: [{bin_lower:.2f}, {bin_upper:.2f}] - no samples")

        tprint(f"📊 ECE calculated: {ece:.4f}")
        _LOGGER.debug(f"📊 ECE calculated: {ece:.4f}")
        return float(ece)
    except Exception as e:
        tprint(f"❌ Failed to calculate ECE: {e}")
        _LOGGER.error(f"❌ Failed to calculate ECE: {e}")
        return 0.0

def calculate_prediction_distribution(y_pred_proba: np.ndarray) -> Dict[str, Any]:
    """
    Calculate prediction distribution statistics.

    Args:
        y_pred_proba: Prediction probabilities

    Returns:
        Dictionary containing distribution metrics
    """
    tprint("🔄 Calculating prediction distribution statistics...")
    try:
        prob_matrix = np.asarray(y_pred_proba, dtype=float)
        tprint(f"📊 Input probability matrix shape: {prob_matrix.shape}")
        if prob_matrix.ndim > 2:
            tprint("🔄 Flattening 3D+ probability matrix...")
            prob_matrix = _flatten_probabilities(prob_matrix)
        elif prob_matrix.ndim == 1:
            tprint("🔄 Converting 1D probability array to 2D...")
            prob_matrix = _ensure_probability_matrix(prob_matrix)

        # Calculate entropy for each prediction
        tprint("🔄 Calculating prediction entropy...")
        epsilon = 1e-10  # Small value to avoid log(0)
        prob_safe = np.clip(prob_matrix, epsilon, 1 - epsilon)
        entropy = -np.sum(prob_safe * np.log2(prob_safe), axis=1)
        tprint(f"📊 Entropy calculated - mean: {np.mean(entropy):.3f}, std: {np.std(entropy):.3f}")

        # Calculate prediction diversity
        tprint("🔄 Calculating prediction diversity...")
        max_probs = np.max(prob_matrix, axis=1)
        min_probs = np.min(prob_matrix, axis=1)
        prob_spread = max_probs - min_probs
        tprint(f"📊 Probability spread calculated - mean: {np.mean(prob_spread):.3f}, std: {np.std(prob_spread):.3f}")

        # Calculate entropy thresholds
        max_entropy = np.log2(prob_matrix.shape[1])
        high_entropy_threshold = max_entropy * 0.8
        low_entropy_threshold = max_entropy * 0.2
        high_entropy_pct = float(np.mean(entropy > high_entropy_threshold) * 100)
        low_entropy_pct = float(np.mean(entropy < low_entropy_threshold) * 100)
        tprint(f"📊 Entropy distribution - high: {high_entropy_pct:.1f}%, low: {low_entropy_pct:.1f}%")

        result = {
            'mean_entropy': float(np.mean(entropy)),
            'std_entropy': float(np.std(entropy)),
            'mean_prob_spread': float(np.mean(prob_spread)),
            'std_prob_spread': float(np.std(prob_spread)),
            'high_entropy_pct': high_entropy_pct,
            'low_entropy_pct': low_entropy_pct
        }
        tprint("✅ Prediction distribution statistics calculated")
        return result
    except Exception as e:
        tprint(f"❌ Failed to calculate prediction distribution: {e}")
        logger.warning(f"Failed to calculate prediction distribution: {e}")
        return {
            'mean_entropy': None,
            'std_entropy': None,
            'mean_prob_spread': None,
            'std_prob_spread': None,
            'high_entropy_pct': None,
            'low_entropy_pct': None
        }

def _assess_calibration_quality(brier_score: Optional[float], ece: Optional[float]) -> str:
    """
    Assess calibration quality based on Brier score and ECE.

    Args:
        brier_score: Brier score (lower is better)
        ece: Expected Calibration Error (lower is better)

    Returns:
        Quality assessment string
    """
    tprint(f"🔄 Assessing calibration quality - Brier: {brier_score}, ECE: {ece}")
    if brier_score is None:
        tprint("❌ No Brier score available for quality assessment")
        return 'unknown'

    if brier_score < 0.25:
        if ece is not None and ece < 0.05:
            tprint("✅ Excellent calibration quality (Brier < 0.25, ECE < 0.05)")
            return 'excellent'
        tprint("✅ Good calibration quality (Brier < 0.25)")
        return 'good'
    elif brier_score < 0.5:
        if ece is not None and ece < 0.1:
            tprint("⚠️ Fair calibration quality (Brier < 0.5, ECE < 0.1)")
            return 'fair'
        tprint("⚠️ Poor calibration quality (Brier < 0.5)")
        return 'poor'
    else:
        tprint("❌ Very poor calibration quality (Brier >= 0.5)")
        return 'very_poor'

def log_confidence_metrics(metrics: Dict[str, Any], model_name: str, logger: logging.Logger) -> None:
    """
    Log confidence metrics in a formatted way.

    Args:
        metrics: Confidence metrics dictionary
        model_name: Name of the model
        logger: Logger instance
    """
    if 'error' in metrics:
        logger.warning(f'⚠️ {model_name} confidence metrics: {metrics["error"]}')
        return

    try:
        # Basic confidence statistics
        logger.info(f'🎯 {model_name} Confidence: '
                   f'Mean={metrics.get("mean_confidence", 0):.3f}, '
                   f'High={metrics.get("high_confidence_pct", 0):.1f}%, '
                   f'Low={metrics.get("low_confidence_pct", 0):.1f}%')

        # Calibration metrics
        if metrics.get('brier_score') is not None:
            brier_score = metrics['brier_score']
            quality = metrics.get('calibration_quality', 'unknown')
            logger.info(f'📊 {model_name} Calibration: '
                       f'Brier={brier_score:.4f} ({quality}), '
                       f'ECE={metrics.get("expected_calibration_error", 0):.4f}')

        # Prediction distribution
        if metrics.get('mean_entropy') is not None:
            logger.info(f'🔀 {model_name} Distribution: '
                       f'Entropy={metrics.get("mean_entropy", 0):.3f}, '
                       f'Spread={metrics.get("mean_prob_spread", 0):.3f}')

    except Exception as e:
        logger.warning(f'Failed to log confidence metrics for {model_name}: {e}')

def get_confidence_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a summary of confidence metrics for reporting.

    Args:
        metrics: Confidence metrics dictionary

    Returns:
        Summary dictionary
    """
    if 'error' in metrics:
        return {'status': 'error', 'message': metrics['error']}

    try:
        return {
            'status': 'success',
            'confidence_level': _get_confidence_level(metrics.get('mean_confidence', 0)),
            'calibration_quality': metrics.get('calibration_quality', 'unknown'),
            'high_confidence_pct': metrics.get('high_confidence_pct', 0),
            'brier_score': metrics.get('brier_score', 0),
            'expected_calibration_error': metrics.get('expected_calibration_error', 0)
        }
    except Exception as e:
        return {'status': 'error', 'message': f'Summary generation failed: {e}'}

def _get_confidence_level(mean_confidence: float) -> str:
    """Get confidence level description based on mean confidence."""
    if mean_confidence >= 0.9:
        return 'very_high'
    elif mean_confidence >= 0.8:
        return 'high'
    elif mean_confidence >= 0.7:
        return 'medium'
    elif mean_confidence >= 0.6:
        return 'low'
    else:
        return 'very_low'

# Advanced Calibration Methods
class ModelConfidenceCalibration:
    """
    Advanced model confidence calibration using multiple methods.

    This class provides comprehensive confidence calibration including:
    - Platt scaling
    - Isotonic regression
    - Temperature scaling
    - Histogram binning
    - Bayesian calibration
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the confidence calibration system."""
        self.config = config or {}
        self.logger = logger.getChild('ModelConfidenceCalibration')

        # Calibration methods configuration
        self.calibration_methods = {
            'platt_scaling': self.config.get('enable_platt_scaling', True),
            'isotonic_regression': self.config.get('enable_isotonic_regression', True),
            'temperature_scaling': self.config.get('enable_temperature_scaling', True),
            'histogram_binning': self.config.get('enable_histogram_binning', True),
            'bayesian_calibration': self.config.get('enable_bayesian_calibration', True)
        }

        # Calibration parameters
        self.calibration_params = {
            'platt_scaling': {
                'learning_rate': 0.01,
                'max_iterations': 1000,
                'convergence_threshold': 1e-6
            },
            'isotonic_regression': {
                'out_of_bounds': 'clip',
                'increasing': True
            },
            'temperature_scaling': {
                'temperature_range': [0.1, 10.0],
                'optimization_method': 'lbfgs'
            },
            'histogram_binning': {
                'bin_count': 10,
                'bin_strategy': 'uniform'
            },
            'bayesian_calibration': {
                'prior_strength': 1.0,
                'mcmc_samples': 1000,
                'burn_in_samples': 100
            }
        }

        # Update with user configuration
        if 'calibration_parameters' in self.config:
            self.calibration_params.update(self.config['calibration_parameters'])

    async def calibrate_model_confidence(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                       method: str = 'all') -> Dict[str, Any]:
        """
        Calibrate model confidence using specified method(s).

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            method: Calibration method ('all', 'platt_scaling', 'isotonic_regression', etc.)

        Returns:
            Dictionary containing calibration results
        """
        try:
            self.logger.info(f"Starting confidence calibration using method: {method}")

            if method == 'all':
                return await self._calibrate_all_methods(y_true, y_pred_proba)
            elif method in self.calibration_methods:
                return await self._calibrate_single_method(y_true, y_pred_proba, method)
            else:
                raise ValueError(f"Unknown calibration method: {method}")

        except Exception as e:
            self.logger.exception(f"Error in confidence calibration: {e}")
            return {'error': str(e)}

    async def _calibrate_all_methods(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calibrate using all enabled methods."""
        results = {}

        for method_name, enabled in self.calibration_methods.items():
            if enabled:
                try:
                    method_result = await self._calibrate_single_method(y_true, y_pred_proba, method_name)
                    if method_result and 'error' not in method_result:
                        results[method_name] = method_result
                except Exception as e:
                    self.logger.warning(f"Failed to calibrate using {method_name}: {e}")

        # Calculate best calibration method
        if results:
            best_method = self._select_best_calibration_method(results)
            results['best_method'] = best_method
            results['calibrated_probabilities'] = results[best_method]['calibrated_probabilities']

        return results

    async def _calibrate_single_method(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                     method: str) -> Dict[str, Any]:
        """Calibrate using a single method."""
        if method == 'platt_scaling':
            return await self._apply_platt_scaling(y_true, y_pred_proba)
        elif method == 'isotonic_regression':
            return await self._apply_isotonic_regression(y_true, y_pred_proba)
        elif method == 'temperature_scaling':
            return await self._apply_temperature_scaling(y_true, y_pred_proba)
        elif method == 'histogram_binning':
            return await self._apply_histogram_binning(y_true, y_pred_proba)
        elif method == 'bayesian_calibration':
            return await self._apply_bayesian_calibration(y_true, y_pred_proba)
        else:
            raise ValueError(f"Unknown calibration method: {method}")

    async def _apply_platt_scaling(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Apply Platt scaling calibration."""
        try:
            if y_pred_proba.shape[1] == 2:
                # Binary classification
                prob_pos = y_pred_proba[:, 1]
            else:
                # Multiclass - use max probability
                prob_pos = np.max(y_pred_proba, axis=1)

            # Reshape for sklearn
            prob_pos = prob_pos.reshape(-1, 1)

            # Create and fit Platt scaling calibrator
            base_classifier = LogisticRegression(
                max_iter=self.calibration_params['platt_scaling']['max_iterations'],
                random_state=42
            )

            calibrator = CalibratedClassifierCV(
                estimator=base_classifier,
                method='sigmoid',
                cv='prefit'
            )

            # Fit calibrator
            calibrator.fit(prob_pos, y_true)

            # Calculate calibrated probabilities
            calibrated_prob = calibrator.predict_proba(prob_pos)[:, 1]

            # Calculate metrics
            brier_before = brier_score_loss(y_true, prob_pos[:, 0])
            brier_after = brier_score_loss(y_true, calibrated_prob)
            ece_before = calculate_expected_calibration_error(y_true, prob_pos[:, 0])
            ece_after = calculate_expected_calibration_error(y_true, calibrated_prob)

            # Get calibration coefficients
            calibrated_clf = calibrator.calibrated_classifiers_[0]
            A = calibrated_clf.calibrators_[0].coef_[0][0] if hasattr(calibrated_clf.calibrators_[0], 'coef_') else 1.0
            B = calibrated_clf.calibrators_[0].intercept_[0] if hasattr(calibrated_clf.calibrators_[0], 'intercept_') else 0.0

            return {
                'method': 'platt_scaling',
                'calibrated_probabilities': calibrated_prob,
                'calibration_coefficients': {'A': float(A), 'B': float(B)},
                'metrics': {
                    'brier_score_before': float(brier_before),
                    'brier_score_after': float(brier_after),
                    'ece_before': float(ece_before),
                    'ece_after': float(ece_after),
                    'improvement': float(brier_before - brier_after)
                },
                'calibration_quality': {
                    'convergence_achieved': True,
                    'final_loss': float(brier_after)
                }
            }

        except Exception as e:
            self.logger.error(f"Error in Platt scaling: {e}")
            return {'error': str(e)}

    async def _apply_isotonic_regression(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Apply isotonic regression calibration."""
        try:
            if y_pred_proba.shape[1] == 2:
                prob_pos = y_pred_proba[:, 1]
            else:
                prob_pos = np.max(y_pred_proba, axis=1)

            # Create and fit isotonic regression
            isotonic_reg = IsotonicRegression(
                out_of_bounds=self.calibration_params['isotonic_regression']['out_of_bounds'],
                increasing=self.calibration_params['isotonic_regression']['increasing']
            )

            isotonic_reg.fit(prob_pos, y_true)

            # Calculate calibrated probabilities
            calibrated_prob = isotonic_reg.predict(prob_pos)
            calibrated_prob = np.clip(calibrated_prob, 0.0, 1.0)

            # Calculate metrics
            brier_before = brier_score_loss(y_true, prob_pos)
            brier_after = brier_score_loss(y_true, calibrated_prob)
            ece_before = calculate_expected_calibration_error(y_true, prob_pos)
            ece_after = calculate_expected_calibration_error(y_true, calibrated_prob)

            # Calculate monotonicity improvement
            monotonicity_before = self._calculate_monotonicity_score(prob_pos, y_true)
            monotonicity_after = self._calculate_monotonicity_score(calibrated_prob, y_true)

            return {
                'method': 'isotonic_regression',
                'calibrated_probabilities': calibrated_prob,
                'calibration_function': {
                    'monotonic': self.calibration_params['isotonic_regression']['increasing'],
                    'piecewise_linear': True
                },
                'metrics': {
                    'brier_score_before': float(brier_before),
                    'brier_score_after': float(brier_after),
                    'ece_before': float(ece_before),
                    'ece_after': float(ece_after),
                    'monotonicity_improvement': float(monotonicity_after - monotonicity_before)
                },
                'calibration_quality': {
                    'monotonicity_achieved': self.calibration_params['isotonic_regression']['increasing'],
                    'smoothness_score': float(self._calculate_smoothness_score(calibrated_prob))
                }
            }

        except Exception as e:
            self.logger.error(f"Error in isotonic regression: {e}")
            return {'error': str(e)}

    async def _apply_temperature_scaling(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Apply temperature scaling calibration."""
        try:
            if y_pred_proba.shape[1] == 2:
                prob_pos = y_pred_proba[:, 1]
            else:
                prob_pos = np.max(y_pred_proba, axis=1)

            # Get temperature range
            temp_range = self.calibration_params['temperature_scaling']['temperature_range']

            # Optimize temperature parameter
            def temperature_loss(temperature):
                if temperature <= 0:
                    return float('inf')

                # Apply temperature scaling
                scaled_prob = 1.0 / (1.0 + np.exp(-(np.log(prob_pos / (1 - prob_pos)) / temperature)))

                # Calculate negative log-likelihood loss
                eps = 1e-15
                scaled_prob = np.clip(scaled_prob, eps, 1 - eps)
                nll = -np.mean(y_true * np.log(scaled_prob) + (1 - y_true) * np.log(1 - scaled_prob))

                return nll

            # Optimize temperature
            result = minimize_scalar(
                temperature_loss,
                bounds=temp_range,
                method='bounded'
            )

            optimal_temperature = result.x
            optimization_converged = result.success

            # Apply optimal temperature scaling
            scaled_prob = 1.0 / (1.0 + np.exp(-(np.log(prob_pos / (1 - prob_pos)) / optimal_temperature)))
            scaled_prob = np.clip(scaled_prob, 0.0, 1.0)

            # Calculate metrics
            brier_before = brier_score_loss(y_true, prob_pos)
            brier_after = brier_score_loss(y_true, scaled_prob)
            ece_before = calculate_expected_calibration_error(y_true, prob_pos)
            ece_after = calculate_expected_calibration_error(y_true, scaled_prob)

            return {
                'method': 'temperature_scaling',
                'calibrated_probabilities': scaled_prob,
                'calibration_coefficients': {
                    'temperature': float(optimal_temperature),
                    'bias': 0.0
                },
                'metrics': {
                    'brier_score_before': float(brier_before),
                    'brier_score_after': float(brier_after),
                    'ece_before': float(ece_before),
                    'ece_after': float(ece_after),
                    'temperature_effectiveness': float(brier_before - brier_after)
                },
                'calibration_quality': {
                    'optimization_converged': bool(optimization_converged),
                    'final_temperature': float(optimal_temperature)
                }
            }

        except Exception as e:
            self.logger.error(f"Error in temperature scaling: {e}")
            return {'error': str(e)}

    async def _apply_histogram_binning(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Apply histogram binning calibration."""
        try:
            if y_pred_proba.shape[1] == 2:
                prob_pos = y_pred_proba[:, 1]
            else:
                prob_pos = np.max(y_pred_proba, axis=1)

            # Get bin parameters
            bin_count = self.calibration_params['histogram_binning']['bin_count']
            bin_strategy = self.calibration_params['histogram_binning']['bin_strategy']

            # Create bins
            if bin_strategy == 'uniform':
                bins = np.linspace(0, 1, bin_count + 1)
            else:
                # Quantile-based bins
                bins = np.quantile(prob_pos, np.linspace(0, 1, bin_count + 1))
                bins[0] = 0.0
                bins[-1] = 1.0

            # Digitize probabilities into bins
            bin_indices = np.digitize(prob_pos, bins) - 1
            bin_indices = np.clip(bin_indices, 0, bin_count - 1)

            # Calculate bin statistics
            bin_counts = []
            bin_accuracies = []
            bin_avg_prob = []

            for bin_idx in range(bin_count):
                bin_mask = bin_indices == bin_idx
                bin_count_val = np.sum(bin_mask)
                bin_counts.append(int(bin_count_val))

                if bin_count_val > 0:
                    bin_accuracy = np.mean(y_true[bin_mask])
                    bin_avg_pred_prob = np.mean(prob_pos[bin_mask])
                else:
                    bin_accuracy = 0.5
                    bin_avg_pred_prob = (bins[bin_idx] + bins[bin_idx + 1]) / 2

                bin_accuracies.append(float(bin_accuracy))
                bin_avg_prob.append(float(bin_avg_pred_prob))

            # Create calibrated probabilities using bin averages
            calibrated_prob = np.zeros_like(prob_pos)
            for bin_idx in range(bin_count):
                bin_mask = bin_indices == bin_idx
                calibrated_prob[bin_mask] = bin_accuracies[bin_idx]

            # Calculate metrics
            brier_before = brier_score_loss(y_true, prob_pos)
            brier_after = brier_score_loss(y_true, calibrated_prob)
            ece_before = calculate_expected_calibration_error(y_true, prob_pos)
            ece_after = calculate_expected_calibration_error(y_true, calibrated_prob)

            return {
                'method': 'histogram_binning',
                'calibrated_probabilities': calibrated_prob,
                'calibration_bins': {
                    'bin_edges': bins.tolist(),
                    'bin_counts': bin_counts,
                    'bin_accuracies': bin_accuracies,
                    'bin_avg_probabilities': bin_avg_prob
                },
                'metrics': {
                    'brier_score_before': float(brier_before),
                    'brier_score_after': float(brier_after),
                    'ece_before': float(ece_before),
                    'ece_after': float(ece_after),
                    'binning_effectiveness': float(brier_before - brier_after)
                },
                'calibration_quality': {
                    'binning_quality': float(sum(1 for count in bin_counts if count > 0) / bin_count),
                    'bin_distribution': bin_strategy
                }
            }

        except Exception as e:
            self.logger.error(f"Error in histogram binning: {e}")
            return {'error': str(e)}

    async def _apply_bayesian_calibration(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Apply Bayesian calibration."""
        try:
            if y_pred_proba.shape[1] == 2:
                prob_pos = y_pred_proba[:, 1]
            else:
                prob_pos = np.max(y_pred_proba, axis=1)

            # Bayesian calibration using beta distribution
            prior_strength = self.calibration_params['bayesian_calibration']['prior_strength']

            # Group predictions into bins for Bayesian estimation
            n_bins = 10
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(prob_pos, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

            # Estimate beta parameters for each bin
            alpha_estimates = []
            beta_estimates = []

            for bin_idx in range(n_bins):
                bin_mask = bin_indices == bin_idx
                bin_labels = y_true[bin_mask]

                if len(bin_labels) == 0:
                    # Use prior for empty bins
                    alpha_estimates.append(prior_strength)
                    beta_estimates.append(prior_strength)
                else:
                    # Calculate posterior parameters
                    successes = np.sum(bin_labels)
                    failures = len(bin_labels) - successes

                    alpha_post = prior_strength + successes
                    beta_post = prior_strength + failures

                    alpha_estimates.append(alpha_post)
                    beta_estimates.append(beta_post)

            # Create calibrated probabilities using beta means
            calibrated_prob = np.zeros_like(prob_pos)
            for bin_idx in range(n_bins):
                bin_mask = bin_indices == bin_idx
                if np.any(bin_mask):
                    # Use beta mean for calibration
                    calibrated_prob[bin_mask] = alpha_estimates[bin_idx] / (alpha_estimates[bin_idx] + beta_estimates[bin_idx])

            # Calculate metrics
            brier_before = brier_score_loss(y_true, prob_pos)
            brier_after = brier_score_loss(y_true, calibrated_prob)
            ece_before = calculate_expected_calibration_error(y_true, prob_pos)
            ece_after = calculate_expected_calibration_error(y_true, calibrated_prob)

            return {
                'method': 'bayesian_calibration',
                'calibrated_probabilities': calibrated_prob,
                'calibration_posterior': {
                    'alpha_parameters': alpha_estimates,
                    'beta_parameters': beta_estimates,
                    'mean_parameters': [a / (a + b) for a, b in zip(alpha_estimates, beta_estimates)]
                },
                'metrics': {
                    'brier_score_before': float(brier_before),
                    'brier_score_after': float(brier_after),
                    'ece_before': float(ece_before),
                    'ece_after': float(ece_after),
                    'bayesian_improvement': float(brier_before - brier_after)
                },
                'calibration_quality': {
                    'prior_strength': prior_strength,
                    'effective_sample_size': len(prob_pos)
                }
            }

        except Exception as e:
            self.logger.error(f"Error in Bayesian calibration: {e}")
            return {'error': str(e)}

    def _select_best_calibration_method(self, results: Dict[str, Any]) -> str:
        """Select the best calibration method based on improvement."""
        best_method = None
        best_improvement = -float('inf')

        for method_name, method_result in results.items():
            if 'metrics' in method_result and 'improvement' in method_result['metrics']:
                improvement = method_result['metrics']['improvement']
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_method = method_name

        return best_method or list(results.keys())[0]

    def _calculate_monotonicity_score(self, probabilities: np.ndarray, labels: np.ndarray) -> float:
        """Calculate monotonicity score for calibration quality assessment."""
        try:
            if len(probabilities) <= 1:
                return 1.0

            # Sort by probability
            sorted_indices = np.argsort(probabilities)
            sorted_prob = probabilities[sorted_indices]
            sorted_labels = labels[sorted_indices]

            # Calculate correlation between probability and label
            correlation = np.corrcoef(sorted_prob, sorted_labels)[0, 1]
            return float(abs(correlation)) if not np.isnan(correlation) else 0.0

        except Exception as e:
            self.logger.warning(f"Error calculating monotonicity score: {e}")
            return 0.0

    def _calculate_smoothness_score(self, probabilities: np.ndarray) -> float:
        """Calculate smoothness score for calibration quality assessment."""
        try:
            if len(probabilities) <= 2:
                return 1.0

            # Calculate second derivative (smoothness)
            first_diff = np.diff(probabilities)
            second_diff = np.diff(first_diff)

            # Smoothness is inverse of average absolute second derivative
            smoothness = 1.0 / (1.0 + np.mean(np.abs(second_diff)))
            return float(smoothness)

        except Exception as e:
            self.logger.warning(f"Error calculating smoothness score: {e}")
            return 0.0

def log_confidence_metrics(confidence_metrics: Dict[str, Any], model_name: str = "Model", logger_instance: Optional[logging.Logger] = None) -> None:
    """
    Log comprehensive confidence metrics in a structured format.

    Args:
        confidence_metrics: Dictionary containing confidence metrics
        model_name: Name of the model for logging context
        logger_instance: Logger instance to use (defaults to module logger)
    """
    if logger_instance is None:
        logger_instance = _LOGGER

    try:
        if 'error' in confidence_metrics:
            logger_instance.error(f"❌ {model_name} confidence metrics error: {confidence_metrics['error']}")
            return

        # Log basic confidence statistics
        if 'mean_confidence' in confidence_metrics:
            logger_instance.info(f"🎯 {model_name} Confidence Stats:")
            logger_instance.info(f"  📊 Mean: {confidence_metrics['mean_confidence']:.3f}")
            logger_instance.info(f"  📊 Std: {confidence_metrics['std_confidence']:.3f}")
            logger_instance.info(f"  📊 Range: [{confidence_metrics['min_confidence']:.3f}, {confidence_metrics['max_confidence']:.3f}]")
            logger_instance.info(f"  📊 High conf (>0.8): {confidence_metrics['high_confidence_pct']:.1f}%")
            logger_instance.info(f"  📊 Low conf (<0.6): {confidence_metrics['low_confidence_pct']:.1f}%")

        # Log calibration metrics
        if 'brier_score' in confidence_metrics and confidence_metrics['brier_score'] is not None:
            logger_instance.info(f"🎯 {model_name} Calibration:")
            logger_instance.info(f"  📈 Brier Score: {confidence_metrics['brier_score']:.4f}")
            if 'expected_calibration_error' in confidence_metrics and confidence_metrics['expected_calibration_error'] is not None:
                logger_instance.info(f"  📈 ECE: {confidence_metrics['expected_calibration_error']:.4f}")
            if 'calibration_quality' in confidence_metrics:
                quality_emoji = "🟢" if confidence_metrics['calibration_quality'] == 'excellent' else "🟡" if confidence_metrics['calibration_quality'] == 'good' else "🔴"
                logger_instance.info(f"  {quality_emoji} Quality: {confidence_metrics['calibration_quality']}")

        # Log distribution metrics
        if 'mean_entropy' in confidence_metrics:
            logger_instance.info(f"🎯 {model_name} Distribution:")
            logger_instance.info(f"  📊 Mean Entropy: {confidence_metrics['mean_entropy']:.3f}")
            logger_instance.info(f"  📊 Entropy Std: {confidence_metrics['std_entropy']:.3f}")
            if 'uncertainty_pct' in confidence_metrics:
                logger_instance.info(f"  📊 High Uncertainty: {confidence_metrics['uncertainty_pct']:.1f}%")

        # Log prediction diversity
        if 'prediction_diversity' in confidence_metrics:
            logger_instance.info(f"🎯 {model_name} Diversity: {confidence_metrics['prediction_diversity']:.3f}")

    except Exception as e:
        logger_instance.error(f"❌ Failed to log confidence metrics for {model_name}: {e}")

# Convenience function for easy integration
async def calibrate_model_confidence(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   method: str = 'all',
                                   config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to calibrate model confidence.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        method: Calibration method ('all', 'platt_scaling', 'isotonic_regression', etc.)
        config: Configuration dictionary

    Returns:
        Calibration results
    """
    calibrator = ModelConfidenceCalibration(config)
    return await calibrator.calibrate_model_confidence(y_true, y_pred_proba, method)
