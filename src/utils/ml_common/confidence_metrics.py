"""Prediction confidence and calibration metrics for ML models."""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import logging

logger = logging.getLogger(__name__)


def calculate_confidence_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
    """
    Calculate comprehensive prediction confidence and calibration metrics.
    
    Args:
        y_true: True labels
        y_pred_proba: Prediction probabilities (shape: n_samples, n_classes)
        
    Returns:
        Dictionary containing confidence and calibration metrics
    """
    if y_pred_proba is None or len(y_pred_proba) == 0:
        return {'error': 'No prediction probabilities available'}
    
    try:
        # Calculate prediction confidence statistics
        max_probs = np.max(y_pred_proba, axis=1)
        
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
        
        # Calculate calibration metrics
        calibration_metrics = calculate_calibration_metrics(y_true, y_pred_proba)
        confidence_metrics.update(calibration_metrics)
        
        # Calculate prediction distribution statistics
        distribution_metrics = calculate_prediction_distribution(y_pred_proba)
        confidence_metrics.update(distribution_metrics)
        
        return confidence_metrics
        
    except Exception as e:
        logger.warning(f"Failed to calculate confidence metrics: {e}")
        return {'error': f'Confidence metrics calculation failed: {e}'}


def calculate_calibration_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
    """
    Calculate calibration metrics including Brier score and calibration curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Prediction probabilities
        
    Returns:
        Dictionary containing calibration metrics
    """
    try:
        if y_pred_proba.shape[1] == 2:
            # Binary classification
            brier_score = brier_score_loss(y_true, y_pred_proba[:, 1])
            
            # Calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba[:, 1], n_bins=10, strategy='uniform'
            )
            
            # Calculate calibration error (ECE - Expected Calibration Error)
            ece = calculate_expected_calibration_error(y_true, y_pred_proba[:, 1], n_bins=10)
            
            return {
                'brier_score': float(brier_score),
                'expected_calibration_error': float(ece),
                'calibration_curve': {
                    'fraction_of_positives': fraction_of_positives.tolist(),
                    'mean_predicted_value': mean_predicted_value.tolist()
                },
                'calibration_quality': _assess_calibration_quality(brier_score, ece)
            }
        else:
            # Multiclass classification - use macro average
            brier_scores = []
            for i in range(y_pred_proba.shape[1]):
                class_mask = (y_true == i)
                if np.sum(class_mask) > 0:
                    brier_scores.append(brier_score_loss(class_mask, y_pred_proba[:, i]))
            
            mean_brier_score = np.mean(brier_scores) if brier_scores else 0.0
            
            return {
                'brier_score': float(mean_brier_score),
                'brier_scores_per_class': [float(score) for score in brier_scores],
                'calibration_quality': _assess_calibration_quality(mean_brier_score, None)
            }
            
    except Exception as e:
        logger.warning(f"Failed to calculate calibration metrics: {e}")
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
    try:
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return float(ece)
    except Exception:
        return 0.0


def calculate_prediction_distribution(y_pred_proba: np.ndarray) -> Dict[str, Any]:
    """
    Calculate prediction distribution statistics.
    
    Args:
        y_pred_proba: Prediction probabilities
        
    Returns:
        Dictionary containing distribution metrics
    """
    try:
        # Calculate entropy for each prediction
        epsilon = 1e-10  # Small value to avoid log(0)
        y_pred_proba_safe = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        entropy = -np.sum(y_pred_proba_safe * np.log2(y_pred_proba_safe), axis=1)
        
        # Calculate prediction diversity
        max_probs = np.max(y_pred_proba, axis=1)
        min_probs = np.min(y_pred_proba, axis=1)
        prob_spread = max_probs - min_probs
        
        return {
            'mean_entropy': float(np.mean(entropy)),
            'std_entropy': float(np.std(entropy)),
            'mean_prob_spread': float(np.mean(prob_spread)),
            'std_prob_spread': float(np.std(prob_spread)),
            'high_entropy_pct': float(np.mean(entropy > np.log2(y_pred_proba.shape[1]) * 0.8) * 100),
            'low_entropy_pct': float(np.mean(entropy < np.log2(y_pred_proba.shape[1]) * 0.2) * 100)
        }
    except Exception as e:
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
    if brier_score is None:
        return 'unknown'
    
    if brier_score < 0.25:
        if ece is not None and ece < 0.05:
            return 'excellent'
        return 'good'
    elif brier_score < 0.5:
        if ece is not None and ece < 0.1:
            return 'fair'
        return 'poor'
    else:
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
