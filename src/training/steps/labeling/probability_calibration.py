"""
Probability Calibration Module

This module provides isotonic and Platt calibration methods for meta_probability calibration.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
import warnings

from .calibration_utils import compare_calibration_methods

# Import tprint utilities
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
except ImportError:
    # Fallback implementation if tprint not available
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)


class ProbabilityCalibrator:
    """
    Probability calibrator supporting isotonic regression and Platt scaling.
    """
    
    def __init__(
        self,
        method: str = "isotonic",
        cv_folds: int = 5,
        random_state: int = 42,
        min_samples: int = 100,
        plot_calibration: bool = True,
        save_plots: bool = True,
        plot_dir: str = "./calibration_plots"
    ):
        """
        Initialize probability calibrator.
        
        Args:
            method: Calibration method ('isotonic', 'platt', or 'ensemble')
            cv_folds: Number of CV folds for out-of-sample calibration
            random_state: Random state for reproducibility
            min_samples: Minimum samples required for calibration
            plot_calibration: Whether to generate calibration plots
            save_plots: Whether to save plots to disk
            plot_dir: Directory to save calibration plots
        """
        self.method = method
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.min_samples = min_samples
        self.plot_calibration = plot_calibration
        self.save_plots = save_plots
        self.plot_dir = plot_dir
        
        # Calibration models
        self.isotonic_model = None
        self.platt_model = None
        self.calibration_data = {}
        
        # Create plot directory if needed
        if self.save_plots:
            import os
            os.makedirs(self.plot_dir, exist_ok=True)
    
    def fit(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Fit calibration models using cross-validation to avoid overfitting.
        
        Args:
            y_true: True binary labels (0 or 1)
            y_proba: Predicted probabilities
            sample_weights: Optional sample weights
            
        Returns:
            Dictionary with calibration results and metrics
        """
        if len(y_true) != len(y_proba):
            raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_proba={len(y_proba)}")
        
        if len(y_true) < self.min_samples:
            tprint_warning(f"Insufficient samples for calibration: {len(y_true)} < {self.min_samples}")
            return self._create_fallback_calibration(y_true, y_proba)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_proba))
        y_true_clean = y_true[valid_mask]
        y_proba_clean = y_proba[valid_mask]
        
        if sample_weights is not None:
            sample_weights = sample_weights[valid_mask]
        
        n_valid = len(y_true_clean)
        if n_valid < self.min_samples:
            tprint_warning(f"Insufficient valid samples for calibration: {n_valid} < {self.min_samples}")
            return self._create_fallback_calibration(y_true_clean, y_proba_clean)
        
        tprint_info(f"Fitting {self.method} calibration on {n_valid} samples...")
        
        # Generate out-of-fold predictions for calibration
        oof_predictions = np.zeros_like(y_proba_clean)
        skf = self._get_cv_splitter(y_true_clean)
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(y_true_clean, y_true_clean)):
            # Split data
            X_train, y_train = y_proba_clean[train_idx].reshape(-1, 1), y_true_clean[train_idx]
            X_val, y_val = y_proba_clean[val_idx].reshape(-1, 1), y_true_clean[val_idx]
            
            weights_train = None
            weights_val = None
            if sample_weights is not None:
                weights_train = sample_weights[train_idx]
                weights_val = sample_weights[val_idx]
            
            # Fit calibration model
            if self.method == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(X_train.ravel(), y_train, sample_weight=weights_train)
                
            elif self.method == "platt":
                calibrator = LogisticRegression(random_state=self.random_state, max_iter=1000)
                calibrator.fit(X_train, y_train, sample_weight=weights_train)
                
            elif self.method == "ensemble":
                # Fit both and average predictions
                iso_reg = IsotonicRegression(out_of_bounds='clip')
                platt_reg = LogisticRegression(random_state=self.random_state, max_iter=1000)
                
                iso_reg.fit(X_train.ravel(), y_train, sample_weight=weights_train)
                platt_reg.fit(X_train, y_train, sample_weight=weights_train)
                
                iso_pred = iso_reg.predict_proba(X_val)[:, 1] if hasattr(iso_reg, 'predict_proba') else iso_reg.predict(X_val.ravel())
                platt_pred = platt_reg.predict_proba(X_val)[:, 1]
                
                oof_predictions[val_idx] = 0.5 * iso_pred + 0.5 * platt_pred
                continue
            
            # Make predictions on validation set
            if self.method != "ensemble":
                if hasattr(calibrator, 'predict_proba'):
                    val_pred = calibrator.predict_proba(X_val)[:, 1]
                else:
                    val_pred = calibrator.predict(X_val.ravel())
                
                oof_predictions[val_idx] = val_pred
        
        # Fit final calibration model on all data
        if self.method == "isotonic":
            self.isotonic_model = IsotonicRegression(out_of_bounds='clip')
            self.isotonic_model.fit(y_proba_clean, y_true_clean, sample_weight=sample_weights)
            
        elif self.method == "platt":
            self.platt_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
            self.platt_model.fit(y_proba_clean.reshape(-1, 1), y_true_clean, sample_weight=sample_weights)
            
        elif self.method == "ensemble":
            self.isotonic_model = IsotonicRegression(out_of_bounds='clip')
            self.platt_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
            
            self.isotonic_model.fit(y_proba_clean, y_true_clean, sample_weight=sample_weights)
            self.platt_model.fit(y_proba_clean.reshape(-1, 1), y_true_clean, sample_weight=sample_weights)
        
        # Calculate calibration metrics
        calibration_metrics = self._calculate_calibration_metrics(
            y_true_clean, y_proba_clean, oof_predictions, sample_weights
        )
        
        # Store calibration data
        self.calibration_data = {
            "y_true": y_true_clean,
            "y_proba_original": y_proba_clean,
            "y_proba_calibrated": oof_predictions,
            "sample_weights": sample_weights,
            "metrics": calibration_metrics,
            "method": self.method,
            "n_samples": n_valid
        }
        
        # Generate calibration plots
        if self.plot_calibration:
            self._plot_calibration_curves(y_true_clean, y_proba_clean, oof_predictions)
        
        tprint_success(f"Calibration completed. Brier score: {calibration_metrics['brier_score']:.4f}")
        
        return {
            "calibrated_probabilities": oof_predictions,
            "metrics": calibration_metrics,
            "method": self.method,
            "n_samples": n_valid,
            "calibration_data": self.calibration_data
        }
    
    def predict(self, y_proba: np.ndarray) -> np.ndarray:
        """
        Apply calibration to new probabilities.
        
        Args:
            y_proba: Predicted probabilities to calibrate
            
        Returns:
            Calibrated probabilities
        """
        if self.method == "isotonic" and self.isotonic_model is not None:
            return self.isotonic_model.predict(y_proba)
        
        elif self.method == "platt" and self.platt_model is not None:
            return self.platt_model.predict_proba(y_proba.reshape(-1, 1))[:, 1]
        
        elif self.method == "ensemble" and self.isotonic_model is not None and self.platt_model is not None:
            iso_pred = self.isotonic_model.predict(y_proba)
            platt_pred = self.platt_model.predict_proba(y_proba.reshape(-1, 1))[:, 1]
            return 0.5 * iso_pred + 0.5 * platt_pred
        
        else:
            tprint_warning("Calibration model not fitted, returning original probabilities")
            return y_proba
    
    def _get_cv_splitter(self, y_true: np.ndarray):
        """Get appropriate CV splitter based on data size."""
        from sklearn.model_selection import StratifiedKFold
        
        if len(y_true) < self.cv_folds * 2:
            # Use fewer folds for small datasets
            folds = max(2, len(y_true) // 2)
            return StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)
        
        return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
    
    def _calculate_calibration_metrics(
        self,
        y_true: np.ndarray,
        y_proba_original: np.ndarray,
        y_proba_calibrated: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate calibration metrics."""
        metrics = {}
        
        # Brier score (lower is better)
        metrics['brier_score_original'] = brier_score_loss(y_true, y_proba_original, sample_weight=sample_weights)
        metrics['brier_score_calibrated'] = brier_score_loss(y_true, y_proba_calibrated, sample_weight=sample_weights)
        metrics['brier_improvement'] = metrics['brier_score_original'] - metrics['brier_score_calibrated']
        
        # Log loss
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics['log_loss_original'] = log_loss(y_true, y_proba_original, sample_weight=sample_weights)
            metrics['log_loss_calibrated'] = log_loss(y_true, y_proba_calibrated, sample_weight=sample_weights)
            metrics['log_loss_improvement'] = metrics['log_loss_original'] - metrics['log_loss_calibrated']
        
        # Expected Calibration Error (ECE)
        metrics['ece_original'] = self._calculate_ece(y_true, y_proba_original, sample_weights)
        metrics['ece_calibrated'] = self._calculate_ece(y_true, y_proba_calibrated, sample_weights)
        metrics['ece_improvement'] = metrics['ece_original'] - metrics['ece_calibrated']
        
        return metrics
    
    def _calculate_ece(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        n_bins: int = 10
    ) -> float:
        """Calculate Expected Calibration Error."""
        # Create bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba, bin_edges, right=True) - 1
        
        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue
            
            bin_confidence = y_proba[mask].mean()
            bin_accuracy = y_true[mask].mean()
            
            if sample_weights is not None:
                bin_weight = sample_weights[mask].sum() / sample_weights.sum()
            else:
                bin_weight = mask.sum() / len(y_true)
            
            ece += bin_weight * abs(bin_confidence - bin_accuracy)
        
        return ece
    
    def _plot_calibration_curves(
        self,
        y_true: np.ndarray,
        y_proba_original: np.ndarray,
        y_proba_calibrated: np.ndarray
    ) -> None:
        """Generate calibration plots."""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Original calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_proba_original, n_bins=10, strategy='quantile'
            )
            axes[0].plot(mean_predicted_value, fraction_of_positives, "s-", label="Original")
            axes[0].plot([0, 1], [0, 1], "k:", label="Perfect calibration")
            axes[0].set_xlabel("Mean predicted probability")
            axes[0].set_ylabel("Fraction of positives")
            axes[0].set_title("Original Calibration")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Calibrated calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_proba_calibrated, n_bins=10, strategy='quantile'
            )
            axes[1].plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibrated")
            axes[1].plot([0, 1], [0, 1], "k:", label="Perfect calibration")
            axes[1].set_xlabel("Mean predicted probability")
            axes[1].set_ylabel("Fraction of positives")
            axes[1].set_title(f"{self.method.title()} Calibration")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Histogram comparison
            axes[2].hist(y_proba_original, bins=20, alpha=0.5, label="Original", density=True)
            axes[2].hist(y_proba_calibrated, bins=20, alpha=0.5, label="Calibrated", density=True)
            axes[2].set_xlabel("Probability")
            axes[2].set_ylabel("Density")
            axes[2].set_title("Probability Distribution")
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if self.save_plots:
                plot_path = f"{self.plot_dir}/calibration_{self.method}_{len(y_true)}samples.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                tprint_info(f"Calibration plot saved: {plot_path}")
            
            plt.close()
            
        except Exception as e:
            tprint_warning(f"Failed to generate calibration plots: {e}")
    
    def _create_fallback_calibration(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Create fallback calibration when insufficient data."""
        tprint_warning("Using fallback calibration (identity function)")
        
        return {
            "calibrated_probabilities": y_proba.copy(),
            "metrics": {
                "brier_score_original": brier_score_loss(y_true, y_proba),
                "brier_score_calibrated": brier_score_loss(y_true, y_proba),
                "brier_improvement": 0.0,
                "log_loss_original": log_loss(y_true, y_proba),
                "log_loss_calibrated": log_loss(y_true, y_proba),
                "log_loss_improvement": 0.0,
                "ece_original": self._calculate_ece(y_true, y_proba),
                "ece_calibrated": self._calculate_ece(y_true, y_proba),
                "ece_improvement": 0.0,
            },
            "method": "identity",
            "n_samples": len(y_true),
            "calibration_data": None
        }


def calibrate_meta_probabilities(
    df: pd.DataFrame,
    y_true_col: str = "binary_label",
    y_proba_col: str = "meta_probability",
    method: str = "isotonic",
    cv_folds: int = 5,
    min_samples: int = 100,
    plot_dir: str = "./calibration_plots",
    verbose: bool = True,
    apply_temperature: bool = False,
    temperature: float = 1.0,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> Dict[str, Any]:
    """
    Calibrate meta_probabilities using isotonic regression or Platt scaling.
    
    Args:
        df: DataFrame with true labels and predicted probabilities
        y_true_col: Column name for true labels
        y_proba_col: Column name for predicted probabilities
        method: Calibration method ('isotonic', 'platt', 'ensemble')
        cv_folds: Number of CV folds for out-of-sample calibration
        min_samples: Minimum samples required for calibration
        plot_dir: Directory to save calibration plots
        verbose: Whether to print progress information
        apply_temperature: Whether to apply temperature scaling after calibration
        temperature: Temperature value (>0). Higher temp flattens probabilities.
        clip_min/clip_max: Optional probability clipping bounds
        
    Returns:
        Dictionary with calibrated probabilities and calibration metrics
    """
    if verbose:
        tprint_info(f"Calibrating {y_proba_col} using {method} method...")
    
    # Validate inputs
    if y_true_col not in df.columns or y_proba_col not in df.columns:
        missing_cols = [col for col in [y_true_col, y_proba_col] if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Get valid samples
    valid_mask = df[y_true_col].notna() & df[y_proba_col].notna()
    valid_df = df[valid_mask].copy()
    
    if len(valid_df) < min_samples:
        if verbose:
            tprint_warning(f"Insufficient valid samples: {len(valid_df)} < {min_samples}")
        return {
            "calibrated_probabilities": np.full(len(df), np.nan),
            "metrics": {},
            "method": method,
            "n_samples": 0,
            "applied": False
        }
    
    # Extract arrays
    y_true = valid_df[y_true_col].values
    y_proba = valid_df[y_proba_col].values
    sample_weights = valid_df["target_sample_weight"].values if "target_sample_weight" in valid_df.columns else None
    
    # Compare calibration methods (isotonic, platt, temperature) and select best ECE
    cal_results = compare_calibration_methods(
        y_true=y_true,
        y_prob=y_proba,
        sample_weights=sample_weights,
        create_plots=verbose,
        save_dir=plot_dir,
    )
    best_method = cal_results.get("best_method", "original")
    calibrated_selected = cal_results.get(best_method, {}).get("probabilities", y_proba)
    metrics_selected = cal_results.get(best_method, {}).get("metrics", {})
    
    # Optional temperature scaling after chosen method (if requested)
    if apply_temperature and best_method != "temperature":
        try:
            temp_probs = np.clip(calibrated_selected, 1e-6, 1 - 1e-6)
            temp_probs = temp_probs ** (1.0 / max(temperature, 1e-6))
            calibrated_selected = temp_probs
            metrics_selected["temperature_applied"] = temperature
        except Exception as e:
            tprint_warning(f"Temperature scaling failed: {e}")
    
    # Clip probabilities
    calibrated_selected = np.clip(calibrated_selected, clip_min, clip_max)
    
    # Apply back to full DataFrame
    calibrated_probs = np.full(len(df), np.nan)
    calibrated_probs[valid_mask] = calibrated_selected

    # Create result DataFrame
    calibrated_col = f"{y_proba_col}_calibrated_{best_method}"
    result_df = df.copy()
    result_df[calibrated_col] = calibrated_probs
    
    # Add calibration metrics to result
    if verbose:
        tprint_success(f"Calibration completed successfully")
        tprint_info(f"  Brier score improvement: {metrics_selected.get('brier_improvement', 0):.4f}")
        tprint_info(f"  ECE improvement: {metrics_selected.get('ece_improvement', 0):.4f}")
        tprint_info(f"  Calibrated column: {calibrated_col}")
    
    return {
        "calibrated_df": result_df,
        "calibrated_probabilities": calibrated_probs,
        "calibrated_column": calibrated_col,
        "metrics": metrics_selected,
        "method": best_method,
        "n_samples": len(y_true),
        "applied": True,
        "all_methods": cal_results,
    }


def select_brier_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    returns: np.ndarray,
    transaction_cost: float = 0.003,
    thresholds: Optional[List[float]] = None,
    min_trades_per_threshold: int = 10,
) -> Tuple[float, Dict[str, Any]]:
    """
    Select the probability threshold that maximizes expected net return.
    
    This function evaluates multiple thresholds and selects the one with
    the highest expected net return (mean return - transaction cost) while
    ensuring sufficient trade count for statistical validity.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities
        returns: Realized returns for each sample
        transaction_cost: Per-trade transaction cost (default 0.3%)
        thresholds: List of thresholds to evaluate (default: 0.5 to 0.8)
        min_trades_per_threshold: Minimum trades required to consider threshold
        
    Returns:
        Tuple of (optimal_threshold, metrics_dict) where metrics_dict contains
        per-threshold statistics and the selection rationale.
    """
    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    
    # Ensure arrays
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    returns = np.asarray(returns)
    
    # Handle NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_prob) | np.isnan(returns))
    y_true = y_true[valid_mask]
    y_prob = y_prob[valid_mask]
    returns = returns[valid_mask]
    
    if len(y_true) == 0:
        tprint_warning("No valid samples for threshold selection")
        return 0.5, {"error": "no_valid_samples", "thresholds": {}}
    
    results = {}
    best_threshold = 0.5
    best_net_return = -np.inf
    
    for thresh in thresholds:
        mask = y_prob >= thresh
        n_trades = mask.sum()
        
        if n_trades < min_trades_per_threshold:
            results[thresh] = {
                "n_trades": int(n_trades),
                "skipped": True,
                "reason": f"insufficient_trades (<{min_trades_per_threshold})"
            }
            continue
        
        # Calculate metrics for this threshold
        mean_return = returns[mask].mean()
        std_return = returns[mask].std()
        win_rate = y_true[mask].mean()
        net_return = mean_return - transaction_cost
        sharpe = net_return / std_return if std_return > 0 else 0.0
        
        results[thresh] = {
            "n_trades": int(n_trades),
            "mean_return": float(mean_return),
            "std_return": float(std_return),
            "net_return": float(net_return),
            "win_rate": float(win_rate),
            "sharpe": float(sharpe),
            "skipped": False,
        }
        
        # Update best if this threshold is better
        if net_return > best_net_return:
            best_net_return = net_return
            best_threshold = thresh
    
    tprint_info(f"Brier-optimal threshold: {best_threshold} (net return: {best_net_return:.4f})")
    
    return best_threshold, {
        "optimal_threshold": best_threshold,
        "optimal_net_return": float(best_net_return),
        "thresholds": results,
        "n_samples": len(y_true),
        "transaction_cost": transaction_cost,
    }


def validate_monotonicity(
    y_prob: np.ndarray,
    returns: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Validate that higher predicted probabilities correspond to higher returns.
    
    This is a critical check for calibration quality. A well-calibrated model
    should show monotonically increasing returns as probability increases.
    
    Args:
        y_prob: Predicted probabilities
        returns: Realized returns
        n_bins: Number of bins for analysis
        
    Returns:
        Dictionary with monotonicity analysis including violation count,
        per-bin statistics, and overall monotonicity score.
    """
    # Handle NaN values
    valid_mask = ~(np.isnan(y_prob) | np.isnan(returns))
    y_prob = np.asarray(y_prob)[valid_mask]
    returns = np.asarray(returns)[valid_mask]
    
    if len(y_prob) < n_bins * 2:
        return {
            "is_monotonic": False,
            "violations": n_bins - 1,
            "error": "insufficient_samples",
            "monotonicity_score": 0.0,
        }
    
    # Create bins based on probability quantiles
    try:
        bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
    except Exception:
        bin_edges = np.linspace(0, 1, n_bins + 1)
    
    bin_indices = np.digitize(y_prob, bin_edges[1:])
    
    bin_stats = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() < 2:
            bin_stats.append({
                "bin": i,
                "mean_prob": np.nan,
                "mean_return": np.nan,
                "count": int(mask.sum()),
            })
        else:
            bin_stats.append({
                "bin": i,
                "mean_prob": float(y_prob[mask].mean()),
                "mean_return": float(returns[mask].mean()),
                "count": int(mask.sum()),
            })
    
    # Count violations (adjacent bin with higher prob but lower return)
    violations = 0
    valid_bins = [s for s in bin_stats if not np.isnan(s["mean_return"])]
    
    for i in range(len(valid_bins) - 1):
        if valid_bins[i + 1]["mean_return"] < valid_bins[i]["mean_return"]:
            violations += 1
    
    total_comparisons = max(len(valid_bins) - 1, 1)
    monotonicity_score = 1.0 - (violations / total_comparisons)
    
    # Diagnostic output for low monotonicity
    if monotonicity_score < 0.5:
        try:
            from src.utils.tprint import tprint_warning
            bin_returns = [f"{s['mean_return']:.6f}" for s in valid_bins]
            bin_probs = [f"{s['mean_prob']:.3f}" for s in valid_bins]
            bin_counts = [str(s['count']) for s in valid_bins]
            tprint_warning(
                f"⚠️ Low monotonicity ({monotonicity_score:.2f}, {violations}/{total_comparisons} violations). "
                f"Bin returns: [{', '.join(bin_returns)}]"
            )
            tprint_warning(
                f"   Bin probs: [{', '.join(bin_probs)}] | Counts: [{', '.join(bin_counts)}]"
            )
            # Check for inverted relationship (all returns decreasing)
            if len(valid_bins) >= 3:
                returns_increasing = sum(
                    1 for i in range(len(valid_bins) - 1) 
                    if valid_bins[i + 1]["mean_return"] > valid_bins[i]["mean_return"]
                )
                if returns_increasing == 0:
                    tprint_warning(
                        "   🔴 CRITICAL: All bin returns are DECREASING with probability - model may be anti-predictive or labels inverted!"
                    )
                elif returns_increasing <= len(valid_bins) // 3:
                    tprint_warning(
                        "   🟡 WARNING: Most bin returns decrease with probability - check calibration or label construction"
                    )
        except ImportError:
            pass  # tprint not available
    
    # Calculate slope in high-probability region (top 3 bins)
    high_prob_bins = [s for s in bin_stats[-3:] if not np.isnan(s["mean_return"])]
    if len(high_prob_bins) >= 2:
        probs = [s["mean_prob"] for s in high_prob_bins]
        rets = [s["mean_return"] for s in high_prob_bins]
        slope = (rets[-1] - rets[0]) / (probs[-1] - probs[0] + 1e-9)
    else:
        slope = 0.0
    
    return {
        "is_monotonic": violations == 0,
        "violations": violations,
        "total_comparisons": total_comparisons,
        "monotonicity_score": float(monotonicity_score),
        "high_prob_slope": float(slope),
        "bin_stats": bin_stats,
    }


def calibration_quality_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    returns: np.ndarray,
    transaction_cost: float = 0.003,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Generate a comprehensive calibration quality report.
    
    This combines multiple quality metrics to assess whether the calibrated
    probabilities are suitable for trading decisions.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        returns: Realized returns
        transaction_cost: Per-trade cost
        n_bins: Number of bins for calibration analysis
        
    Returns:
        Dictionary with quality scores and recommendations.
    """
    # Handle NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_prob) | np.isnan(returns))
    y_true = np.asarray(y_true)[valid_mask]
    y_prob = np.asarray(y_prob)[valid_mask]
    returns = np.asarray(returns)[valid_mask]
    
    report = {
        "n_samples": len(y_true),
        "calibration_valid": len(y_true) >= 100,
    }
    
    if len(y_true) < 50:
        report["error"] = "insufficient_samples"
        report["quality_score"] = 0.0
        report["recommendation"] = "Collect more data before trading"
        return report
    
    # 1. Brier score
    brier = brier_score_loss(y_true, y_prob)
    report["brier_score"] = float(brier)
    
    # 2. ECE
    try:
        from sklearn.calibration import calibration_curve
        fraction_positive, mean_predicted = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='quantile'
        )
        ece = np.mean(np.abs(fraction_positive - mean_predicted))
    except Exception:
        ece = np.nan
    report["ece"] = float(ece) if not np.isnan(ece) else None
    
    # 3. Monotonicity
    monotonicity = validate_monotonicity(y_prob, returns, n_bins)
    report["monotonicity"] = monotonicity
    
    # 4. Optimal threshold
    optimal_thresh, thresh_info = select_brier_optimal_threshold(
        y_true, y_prob, returns, transaction_cost
    )
    report["optimal_threshold"] = optimal_thresh
    report["threshold_analysis"] = thresh_info
    
    # 5. Overall quality score (0-1)
    # Components:
    # - Brier: 0.25 weight (lower is better, normalize to 0-1)
    # - ECE: 0.25 weight (lower is better)
    # - Monotonicity: 0.30 weight
    # - Positive net return: 0.20 weight
    
    brier_score = max(0, 1 - brier / 0.25)  # 0.25 is considered very bad
    ece_score = max(0, 1 - (ece or 0.5) / 0.15) if ece else 0.5
    mono_score = monotonicity["monotonicity_score"]
    
    best_net_return = thresh_info.get("optimal_net_return", 0)
    return_score = 1.0 if best_net_return > 0 else 0.5 if best_net_return > -0.001 else 0.0
    
    quality_score = (
        0.25 * brier_score +
        0.25 * ece_score +
        0.30 * mono_score +
        0.20 * return_score
    )
    report["quality_score"] = float(quality_score)
    
    # Recommendation
    if quality_score >= 0.7:
        report["recommendation"] = "Calibration is good. Proceed with threshold-based trading."
        report["rating"] = "GOOD"
    elif quality_score >= 0.5:
        report["recommendation"] = "Calibration is acceptable. Consider recalibration or stricter thresholds."
        report["rating"] = "ACCEPTABLE"
    else:
        report["recommendation"] = "Calibration is poor. Recalibrate before trading or use higher thresholds."
        report["rating"] = "POOR"
    
    return report
