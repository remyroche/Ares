"""
Probability calibration utilities for meta-labeling.

Provides isotonic and Platt scaling with calibration plots and diagnostics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Any
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from scipy.stats import spearmanr
import warnings

from src.utils.tprint import tprint_info, tprint_success, tprint_warning


def fit_isotonic_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    out_of_bounds: str = 'clip'
) -> Tuple[IsotonicRegression, np.ndarray]:
    """
    Fit isotonic regression for probability calibration.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        sample_weights: Optional sample weights
        out_of_bounds: How to handle out-of-bounds values
        
    Returns:
        Tuple of (fitted calibrator, calibrated probabilities)
    """
    # Filter valid samples
    valid_mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true_clean = y_true[valid_mask]
    y_prob_clean = y_prob[valid_mask]
    
    if sample_weights is not None:
        sample_weights_clean = sample_weights[valid_mask]
    else:
        sample_weights_clean = None
    
    # Fit isotonic regression
    iso_reg = IsotonicRegression(out_of_bounds=out_of_bounds)
    
    if sample_weights_clean is not None:
        iso_reg.fit(y_prob_clean, y_true_clean, sample_weight=sample_weights_clean)
    else:
        iso_reg.fit(y_prob_clean, y_true_clean)
    
    # Apply calibration
    y_prob_calibrated = iso_reg.transform(y_prob)
    
    return iso_reg, y_prob_calibrated


def fit_platt_scaling(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weights: Optional[np.ndarray] = None
) -> Tuple[LogisticRegression, np.ndarray]:
    """
    Fit Platt scaling (logistic regression) for probability calibration.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        sample_weights: Optional sample weights
        
    Returns:
        Tuple of (fitted calibrator, calibrated probabilities)
    """
    # Filter valid samples
    valid_mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true_clean = y_true[valid_mask]
    y_prob_clean = y_prob[valid_mask]
    
    if sample_weights is not None:
        sample_weights_clean = sample_weights[valid_mask]
    else:
        sample_weights_clean = None
    
    # Fit logistic regression on log-odds
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Use log-odds as features for Platt scaling
        log_odds = np.log(y_prob_clean / (1 - y_prob_clean + 1e-15))
        log_odds = np.clip(log_odds, -10, 10)  # Clip extreme values
        
        platt_model = LogisticRegression(
            penalty='none',
            solver='lbfgs',
            max_iter=1000,
            fit_intercept=True
        )
        
        if sample_weights_clean is not None:
            platt_model.fit(log_odds.reshape(-1, 1), y_true_clean, sample_weight=sample_weights_clean)
        else:
            platt_model.fit(log_odds.reshape(-1, 1), y_true_clean)
    
    # Apply calibration
    log_odds_all = np.log(y_prob / (1 - y_prob + 1e-15))
    log_odds_all = np.clip(log_odds_all, -10, 10)
    y_prob_calibrated = platt_model.predict_proba(log_odds_all.reshape(-1, 1))[:, 1]
    
    return platt_model, y_prob_calibrated


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Compute comprehensive calibration metrics.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of calibration bins
        
    Returns:
        Dict with calibration metrics
    """
    # Filter valid samples
    valid_mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true_clean = y_true[valid_mask]
    y_prob_clean = y_prob[valid_mask]
    
    if len(y_true_clean) < 50:
        return {"ece": np.nan, "mce": np.nan, "brier": np.nan, "log_loss": np.nan}
    
    # Clip probabilities
    y_prob_clipped = np.clip(y_prob_clean, 1e-6, 1 - 1e-6)
    
    # Brier score and log loss
    brier = brier_score_loss(y_true_clean, y_prob_clipped)
    try:
        ll = log_loss(y_true_clean, y_prob_clipped)
    except Exception:
        ll = np.nan
    
    # Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    try:
        prob_true, prob_pred = calibration_curve(
            y_true_clean, y_prob_clipped, n_bins=n_bins, strategy='uniform'
        )
        
        # Bin weights
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob_clipped, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        bin_counts = np.bincount(bin_indices, minlength=n_bins)
        bin_weights = bin_counts / len(y_prob_clipped)
        
        # Calibration errors
        calibration_errors = np.abs(prob_true - prob_pred)
        ece = np.sum(bin_weights[:len(calibration_errors)] * calibration_errors)
        mce = np.max(calibration_errors) if len(calibration_errors) > 0 else np.nan
    except Exception:
        ece = np.nan
        mce = np.nan
    
    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier": float(brier),
        "log_loss": float(ll),
    }


def create_calibration_plot(
    y_true: np.ndarray,
    y_prob_original: np.ndarray,
    y_prob_calibrated: np.ndarray,
    method: str = "isotonic",
    n_bins: int = 10,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Create calibration plot comparing original vs calibrated probabilities.
    
    Args:
        y_true: True binary labels
        y_prob_original: Original predicted probabilities
        y_prob_calibrated: Calibrated probabilities
        method: Calibration method used
        n_bins: Number of bins for calibration curve
        save_path: Optional path to save plot
        
    Returns:
        Tuple of (figure, plot_data)
    """
    # Filter valid samples
    valid_mask = np.isfinite(y_true) & np.isfinite(y_prob_original) & np.isfinite(y_prob_calibrated)
    y_true_clean = y_true[valid_mask]
    y_prob_orig_clean = y_prob_original[valid_mask]
    y_prob_cal_clean = y_prob_calibrated[valid_mask]
    
    # Compute calibration curves
    prob_true_orig, prob_pred_orig = calibration_curve(
        y_true_clean, y_prob_orig_clean, n_bins=n_bins, strategy='uniform'
    )
    prob_true_cal, prob_pred_cal = calibration_curve(
        y_true_clean, y_prob_cal_clean, n_bins=n_bins, strategy='uniform'
    )
    
    # Compute metrics
    metrics_orig = compute_calibration_metrics(y_true_clean, y_prob_orig_clean, n_bins)
    metrics_cal = compute_calibration_metrics(y_true_clean, y_prob_cal_clean, n_bins)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Calibration curves
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.7)
    ax1.plot(prob_pred_orig, prob_true_orig, 'ro-', label=f'Original (ECE={metrics_orig["ece"]:.3f})', 
             markersize=4, alpha=0.7)
    ax1.plot(prob_pred_cal, prob_true_cal, 'bs-', label=f'{method.capitalize()} (ECE={metrics_cal["ece"]:.3f})', 
             markersize=4, alpha=0.7)
    
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title('Calibration Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogram of predictions
    bins_hist = np.linspace(0, 1, 21)
    ax2.hist(y_prob_orig_clean, bins=bins_hist, alpha=0.5, label='Original', density=True)
    ax2.hist(y_prob_cal_clean, bins=bins_hist, alpha=0.5, label=f'{method.capitalize()}', density=True)
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Density')
    ax2.set_title('Probability Distributions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        tprint_info(f"   Calibration plot saved to {save_path}")
    
    plot_data = {
        "prob_true_original": prob_true_orig,
        "prob_pred_original": prob_pred_orig,
        "prob_true_calibrated": prob_true_cal,
        "prob_pred_calibrated": prob_pred_cal,
        "metrics_original": metrics_orig,
        "metrics_calibrated": metrics_cal,
        "method": method
    }
    
    return fig, plot_data


def apply_temperature_scaling(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    init_temperature: float = 1.0,
    learning_rate: float = 0.01,
    max_iter: int = 1000
) -> Tuple[float, np.ndarray, Dict[str, Any]]:
    """
    Apply temperature scaling for probability calibration.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        sample_weights: Optional sample weights
        init_temperature: Initial temperature value
        learning_rate: Learning rate for optimization
        max_iter: Maximum iterations
        
    Returns:
        Tuple of (optimal_temperature, calibrated_probabilities, optimization_info)
    """
    # Filter valid samples
    valid_mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true_clean = y_true[valid_mask]
    y_prob_clean = y_prob[valid_mask]
    
    if sample_weights is not None:
        sample_weights_clean = sample_weights[valid_mask]
    else:
        sample_weights_clean = None
    
    def nll_loss(temperature):
        """Negative log likelihood with temperature scaling."""
        temp_probs = y_prob_clean ** temperature
        temp_probs = temp_probs / (temp_probs.sum(axis=1, keepdims=True) + 1e-15)
        
        # Clip for numerical stability
        temp_probs = np.clip(temp_probs, 1e-15, 1 - 1e-15)
        
        if sample_weights_clean is not None:
            nll = -np.sum(sample_weights_clean * (
                y_true_clean * np.log(temp_probs[:, 1] + 1e-15) +
                (1 - y_true_clean) * np.log(temp_probs[:, 0] + 1e-15)
            )) / np.sum(sample_weights_clean)
        else:
            nll = -np.mean(
                y_true_clean * np.log(temp_probs[:, 1] + 1e-15) +
                (1 - y_true_clean) * np.log(temp_probs[:, 0] + 1e-15)
            )
        return nll
    
    # Simple gradient descent for temperature optimization
    temperature = init_temperature
    best_temp = temperature
    best_loss = nll_loss(temperature)
    
    for i in range(max_iter):
        # Compute gradient numerically
        eps = 1e-6
        loss_plus = nll_loss(temperature + eps)
        loss_minus = nll_loss(temperature - eps)
        gradient = (loss_plus - loss_minus) / (2 * eps)
        
        # Update temperature
        temperature -= learning_rate * gradient
        temperature = max(0.1, min(10.0, temperature))  # Constrain temperature
        
        # Check for improvement
        current_loss = nll_loss(temperature)
        if current_loss < best_loss:
            best_loss = current_loss
            best_temp = temperature
    
    # Apply optimal temperature
    temp_probs_all = y_prob ** best_temp
    temp_probs_all = temp_probs_all / (temp_probs_all.sum(axis=1, keepdims=True) + 1e-15)
    calibrated_probs = temp_probs_all[:, 1] if temp_probs_all.shape[1] > 1 else temp_probs_all
    
    optimization_info = {
        "optimal_temperature": best_temp,
        "final_loss": best_loss,
        "iterations": max_iter,
        "converged": abs(temperature - best_temp) < 0.01
    }
    
    return best_temp, calibrated_probs, optimization_info


def compare_calibration_methods(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    create_plots: bool = True,
    save_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare multiple calibration methods.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        sample_weights: Optional sample weights
        create_plots: Whether to create calibration plots
        save_dir: Optional directory to save plots
        
    Returns:
        Dict with comparison results
    """
    tprint_info("   Comparing calibration methods...")
    
    results = {
        "original": {
            "probabilities": y_prob,
            "metrics": compute_calibration_metrics(y_true, y_prob)
        }
    }
    
    # Isotonic calibration
    try:
        iso_model, y_prob_iso = fit_isotonic_calibration(y_true, y_prob, sample_weights)
        results["isotonic"] = {
            "probabilities": y_prob_iso,
            "metrics": compute_calibration_metrics(y_true, y_prob_iso),
            "model": iso_model
        }
        tprint_info(f"   Isotonic: ECE={results['isotonic']['metrics']['ece']:.4f}")
    except Exception as e:
        tprint_warning(f"   Isotonic calibration failed: {e}")
    
    # Platt scaling
    try:
        platt_model, y_prob_platt = fit_platt_scaling(y_true, y_prob, sample_weights)
        results["platt"] = {
            "probabilities": y_prob_platt,
            "metrics": compute_calibration_metrics(y_true, y_prob_platt),
            "model": platt_model
        }
        tprint_info(f"   Platt: ECE={results['platt']['metrics']['ece']:.4f}")
    except Exception as e:
        tprint_warning(f"   Platt scaling failed: {e}")
    
    # Temperature scaling
    try:
        temp, y_prob_temp, temp_info = apply_temperature_scaling(y_true, y_prob, sample_weights)
        results["temperature"] = {
            "probabilities": y_prob_temp,
            "metrics": compute_calibration_metrics(y_true, y_prob_temp),
            "temperature": temp,
            "optimization_info": temp_info
        }
        tprint_info(f"   Temperature (T={temp:.3f}): ECE={results['temperature']['metrics']['ece']:.4f}")
    except Exception as e:
        tprint_warning(f"   Temperature scaling failed: {e}")
    
    # Create plots
    if create_plots:
        for method in ["isotonic", "platt", "temperature"]:
            if method in results:
                try:
                    save_path = f"{save_dir}/calibration_plot_{method}.png" if save_dir else None
                    fig, plot_data = create_calibration_plot(
                        y_true, y_prob, results[method]["probabilities"], method, save_path=save_path
                    )
                    results[method]["plot_data"] = plot_data
                    plt.close(fig)
                except Exception as e:
                    tprint_warning(f"   Failed to create {method} plot: {e}")
    
    # Find best method
    best_method = "original"
    best_ece = results["original"]["metrics"]["ece"]
    
    for method, result in results.items():
        if method != "original" and not np.isnan(result["metrics"]["ece"]):
            if result["metrics"]["ece"] < best_ece:
                best_ece = result["metrics"]["ece"]
                best_method = method
    
    results["best_method"] = best_method
    results["improvement"] = {
        "ece_reduction": results["original"]["metrics"]["ece"] - results[best_method]["metrics"]["ece"],
        "relative_improvement": (results["original"]["metrics"]["ece"] - results[best_method]["metrics"]["ece"]) / max(results["original"]["metrics"]["ece"], 1e-8) * 100
    }
    
    tprint_success(f"   Best method: {best_method} (ECE improvement: {results['improvement']['ece_reduction']:.4f})")
    
    return results
