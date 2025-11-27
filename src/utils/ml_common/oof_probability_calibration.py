"""
OOF Probability Calibration for Specialist Models

This module provides lightweight probability calibration specifically designed
for Out-of-Fold (OOF) predictions from StandardizedXGBTrainer and similar
regime-detecting ML steps.

Design Philosophy:
- Train calibrators on OOF predictions (no data leakage)
- Simple calibration methods that work well with time-series data
- Support for both classification (probability) and regression (scalar) outputs
- Maintain 0-1 output range for downstream consumers

Calibration Methods:
1. Isotonic Regression - Non-parametric, flexible, best for OOF predictions
2. Platt Scaling - Parametric sigmoid, good for small datasets
3. Temperature Scaling - Simple single-parameter scaling
4. Beta Calibration - Flexible parametric method for probabilities

Usage with StandardizedXGBTrainer:
    ```python
    from src.utils.ml_common.oof_probability_calibration import OOFProbabilityCalibrator
    
    # After training with StandardizedXGBTrainer
    calibrator = OOFProbabilityCalibrator(method="isotonic")
    
    # Calibrate using OOF predictions
    calibrated_probs = calibrator.fit_transform(
        oof_predictions=results.oof_predictions,
        y_true=targets,
        data_index=data.index
    )
    
    # For live predictions, apply calibration
    live_calibrated = calibrator.transform(live_predictions)
    ```

Author: AI Assistant
Date: 2024-11-27
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OOFCalibrationConfig:
    """Configuration for OOF probability calibration."""
    
    # Calibration method
    method: str = "isotonic"  # "isotonic", "platt", "temperature", "beta"
    
    # Method-specific parameters
    isotonic_out_of_bounds: str = "clip"  # "clip" or "nan"
    platt_regularization: float = 1.0
    temperature_search_range: Tuple[float, float] = (0.1, 10.0)
    beta_prior_strength: float = 1.0
    
    # Validation
    min_samples_for_calibration: int = 100
    validation_split: float = 0.2  # For calibrator validation
    
    # Output format
    output_range: Tuple[float, float] = (0.0, 1.0)
    clip_to_range: bool = True
    
    # Persistence
    cache_dir: Optional[Path] = None
    model_id: Optional[str] = None


@dataclass
class OOFCalibrationResult:
    """Result from OOF probability calibration."""
    
    # Calibrated predictions
    calibrated_predictions: pd.Series
    
    # Calibration quality metrics
    brier_score_before: float
    brier_score_after: float
    ece_before: float  # Expected Calibration Error
    ece_after: float
    calibration_improvement: float
    
    # Calibrator metadata
    method_used: str
    n_samples_calibrated: int
    fit_timestamp: datetime
    
    # Method-specific parameters
    calibrator_params: Dict[str, Any] = field(default_factory=dict)


class OOFProbabilityCalibrator:
    """
    Probability calibration for OOF predictions.
    
    Designed to work with StandardizedXGBTrainer outputs where:
    - OOF predictions are available (no data leakage)
    - Output should be a calibrated 0-1 scalar
    - Downstream consumers expect calibrated probabilities
    
    Best Practices for OOF Calibration:
    1. Use isotonic regression for most cases (non-parametric, flexible)
    2. Use Platt scaling for small datasets (<500 OOF samples)
    3. Apply calibration after all OOF windows are complete
    4. Save calibrator for live inference
    """
    
    def __init__(self, config: Optional[OOFCalibrationConfig] = None):
        """
        Initialize OOF probability calibrator.
        
        Args:
            config: Calibration configuration (uses defaults if not provided)
        """
        self.config = config or OOFCalibrationConfig()
        self._calibrator = None
        self._is_fitted = False
        self._fit_result: Optional[OOFCalibrationResult] = None
        
        logger.info(f"Initialized OOFProbabilityCalibrator with method={self.config.method}")
    
    def fit(
        self,
        oof_predictions: Union[pd.Series, pd.DataFrame, np.ndarray],
        y_true: Union[pd.Series, np.ndarray],
        data_index: Optional[pd.Index] = None
    ) -> "OOFProbabilityCalibrator":
        """
        Fit calibrator on OOF predictions.
        
        Args:
            oof_predictions: OOF predictions from StandardizedXGBTrainer
                            Can be Series, DataFrame (with 'probability' column), or array
            y_true: True labels (0/1 for binary, continuous for regression)
            data_index: Optional datetime index for alignment
            
        Returns:
            self for method chaining
        """
        logger.info("Fitting OOF probability calibrator...")
        
        # Extract predictions
        pred_values = self._extract_predictions(oof_predictions, data_index)
        true_values = self._extract_true_values(y_true, pred_values.index)
        
        # Align on common index
        common_idx = pred_values.index.intersection(true_values.index)
        pred_aligned = pred_values.loc[common_idx].values
        true_aligned = true_values.loc[common_idx].values
        
        n_samples = len(pred_aligned)
        logger.info(f"Calibrating on {n_samples} OOF samples")
        
        if n_samples < self.config.min_samples_for_calibration:
            logger.warning(
                f"Insufficient samples for calibration ({n_samples} < {self.config.min_samples_for_calibration}). "
                "Using identity calibration."
            )
            self._calibrator = _IdentityCalibrator()
            self._is_fitted = True
            return self
        
        # Calculate pre-calibration metrics
        brier_before = self._brier_score(true_aligned, pred_aligned)
        ece_before = self._expected_calibration_error(true_aligned, pred_aligned)
        
        # Fit calibrator based on method
        if self.config.method == "isotonic":
            self._calibrator = self._fit_isotonic(pred_aligned, true_aligned)
        elif self.config.method == "platt":
            self._calibrator = self._fit_platt(pred_aligned, true_aligned)
        elif self.config.method == "temperature":
            self._calibrator = self._fit_temperature(pred_aligned, true_aligned)
        elif self.config.method == "beta":
            self._calibrator = self._fit_beta(pred_aligned, true_aligned)
        else:
            raise ValueError(f"Unknown calibration method: {self.config.method}")
        
        self._is_fitted = True
        
        # Calculate post-calibration metrics
        calibrated = self._calibrator.predict(pred_aligned)
        brier_after = self._brier_score(true_aligned, calibrated)
        ece_after = self._expected_calibration_error(true_aligned, calibrated)
        
        # Store result
        self._fit_result = OOFCalibrationResult(
            calibrated_predictions=pd.Series(calibrated, index=common_idx),
            brier_score_before=float(brier_before),
            brier_score_after=float(brier_after),
            ece_before=float(ece_before),
            ece_after=float(ece_after),
            calibration_improvement=float(brier_before - brier_after),
            method_used=self.config.method,
            n_samples_calibrated=n_samples,
            fit_timestamp=datetime.now(),
            calibrator_params=getattr(self._calibrator, 'params_', {})
        )
        
        logger.info(
            f"Calibration complete: Brier {brier_before:.4f} -> {brier_after:.4f} "
            f"(improvement: {brier_before - brier_after:.4f})"
        )
        
        return self
    
    def transform(
        self,
        predictions: Union[pd.Series, pd.DataFrame, np.ndarray]
    ) -> Union[pd.Series, np.ndarray]:
        """
        Apply calibration to new predictions.
        
        Args:
            predictions: Raw predictions to calibrate
            
        Returns:
            Calibrated predictions (same type as input)
        """
        if not self._is_fitted:
            raise RuntimeError("Calibrator must be fitted before transform")
        
        # Extract values
        if isinstance(predictions, pd.DataFrame):
            if 'probability' in predictions.columns:
                values = predictions['probability'].values
                index = predictions.index
            else:
                values = predictions.iloc[:, 0].values
                index = predictions.index
            return_series = True
        elif isinstance(predictions, pd.Series):
            values = predictions.values
            index = predictions.index
            return_series = True
        else:
            values = np.asarray(predictions)
            index = None
            return_series = False
        
        # Apply calibration
        calibrated = self._calibrator.predict(values)
        
        # Clip to output range
        if self.config.clip_to_range:
            calibrated = np.clip(
                calibrated,
                self.config.output_range[0],
                self.config.output_range[1]
            )
        
        if return_series:
            return pd.Series(calibrated, index=index, name='calibrated_probability')
        return calibrated
    
    def fit_transform(
        self,
        oof_predictions: Union[pd.Series, pd.DataFrame, np.ndarray],
        y_true: Union[pd.Series, np.ndarray],
        data_index: Optional[pd.Index] = None
    ) -> pd.Series:
        """
        Fit calibrator and return calibrated OOF predictions.
        
        Args:
            oof_predictions: OOF predictions from StandardizedXGBTrainer
            y_true: True labels
            data_index: Optional datetime index
            
        Returns:
            Calibrated predictions as Series
        """
        self.fit(oof_predictions, y_true, data_index)
        return self._fit_result.calibrated_predictions
    
    def get_calibration_metrics(self) -> Dict[str, Any]:
        """Get calibration quality metrics."""
        if self._fit_result is None:
            return {}
        
        return {
            'method': self._fit_result.method_used,
            'brier_before': self._fit_result.brier_score_before,
            'brier_after': self._fit_result.brier_score_after,
            'ece_before': self._fit_result.ece_before,
            'ece_after': self._fit_result.ece_after,
            'improvement': self._fit_result.calibration_improvement,
            'n_samples': self._fit_result.n_samples_calibrated,
        }
    
    # =========================================================================
    # Private Methods - Calibrator Fitting
    # =========================================================================
    
    def _fit_isotonic(self, pred: np.ndarray, true: np.ndarray) -> "_IsotonicCalibrator":
        """Fit isotonic regression calibrator."""
        from sklearn.isotonic import IsotonicRegression
        
        calibrator = _IsotonicCalibrator()
        calibrator.model_ = IsotonicRegression(
            out_of_bounds=self.config.isotonic_out_of_bounds
        )
        calibrator.model_.fit(pred, true)
        calibrator.params_ = {'out_of_bounds': self.config.isotonic_out_of_bounds}
        
        return calibrator
    
    def _fit_platt(self, pred: np.ndarray, true: np.ndarray) -> "_PlattCalibrator":
        """Fit Platt scaling (sigmoid) calibrator."""
        from sklearn.linear_model import LogisticRegression
        
        calibrator = _PlattCalibrator()
        calibrator.model_ = LogisticRegression(
            C=self.config.platt_regularization,
            solver='lbfgs',
            max_iter=1000
        )
        calibrator.model_.fit(pred.reshape(-1, 1), true)
        calibrator.params_ = {'regularization': self.config.platt_regularization}
        
        return calibrator
    
    def _fit_temperature(self, pred: np.ndarray, true: np.ndarray) -> "_TemperatureCalibrator":
        """Fit temperature scaling calibrator."""
        from scipy.optimize import minimize_scalar
        
        calibrator = _TemperatureCalibrator()
        
        # Optimize temperature to minimize NLL
        def nll_loss(temperature):
            if temperature <= 0:
                return float('inf')
            scaled = self._apply_temperature(pred, temperature)
            eps = 1e-15
            scaled = np.clip(scaled, eps, 1 - eps)
            nll = -np.mean(true * np.log(scaled) + (1 - true) * np.log(1 - scaled))
            return nll
        
        result = minimize_scalar(
            nll_loss,
            bounds=self.config.temperature_search_range,
            method='bounded'
        )
        
        calibrator.temperature_ = float(result.x)
        calibrator.params_ = {
            'temperature': calibrator.temperature_,
            'optimization_converged': result.success
        }
        
        return calibrator
    
    def _fit_beta(self, pred: np.ndarray, true: np.ndarray) -> "_BetaCalibrator":
        """Fit beta calibration."""
        from scipy.optimize import minimize
        from scipy.special import logit, expit
        
        calibrator = _BetaCalibrator()
        
        # Beta calibration: P_calibrated = sigmoid(a * logit(P_raw) + b + c * P_raw)
        def nll_loss(params):
            a, b, c = params
            eps = 1e-8
            pred_clipped = np.clip(pred, eps, 1 - eps)
            logit_pred = logit(pred_clipped)
            scaled = expit(a * logit_pred + b + c * pred_clipped)
            scaled = np.clip(scaled, eps, 1 - eps)
            nll = -np.mean(true * np.log(scaled) + (1 - true) * np.log(1 - scaled))
            return nll
        
        # Initial parameters
        x0 = [1.0, 0.0, 0.0]
        result = minimize(nll_loss, x0, method='L-BFGS-B')
        
        calibrator.a_ = float(result.x[0])
        calibrator.b_ = float(result.x[1])
        calibrator.c_ = float(result.x[2])
        calibrator.params_ = {
            'a': calibrator.a_,
            'b': calibrator.b_,
            'c': calibrator.c_,
            'optimization_converged': result.success
        }
        
        return calibrator
    
    # =========================================================================
    # Private Methods - Utilities
    # =========================================================================
    
    def _extract_predictions(
        self,
        predictions: Union[pd.Series, pd.DataFrame, np.ndarray],
        index: Optional[pd.Index]
    ) -> pd.Series:
        """Extract predictions as Series."""
        if isinstance(predictions, pd.DataFrame):
            if 'probability' in predictions.columns:
                return predictions['probability']
            elif 'prediction' in predictions.columns:
                return predictions['prediction']
            return predictions.iloc[:, 0]
        elif isinstance(predictions, pd.Series):
            return predictions
        else:
            idx = index if index is not None else pd.RangeIndex(len(predictions))
            return pd.Series(predictions, index=idx)
    
    def _extract_true_values(
        self,
        y_true: Union[pd.Series, np.ndarray],
        index: pd.Index
    ) -> pd.Series:
        """Extract true values as Series."""
        if isinstance(y_true, pd.Series):
            return y_true
        return pd.Series(y_true, index=index[:len(y_true)])
    
    def _brier_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Brier score."""
        return float(np.mean((y_true - y_pred) ** 2))
    
    def _expected_calibration_error(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_conf = np.mean(y_pred[mask])
                bin_acc = np.mean(y_true[mask])
                ece += mask.sum() * np.abs(bin_conf - bin_acc)
        
        return float(ece / len(y_true))
    
    @staticmethod
    def _apply_temperature(pred: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling."""
        eps = 1e-8
        pred_clipped = np.clip(pred, eps, 1 - eps)
        logit = np.log(pred_clipped / (1 - pred_clipped))
        scaled_logit = logit / temperature
        return 1 / (1 + np.exp(-scaled_logit))
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save calibrator to disk."""
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'calibrator': self._calibrator,
                'is_fitted': self._is_fitted,
                'fit_result': self._fit_result
            }, f)
        
        logger.info(f"Saved calibrator to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "OOFProbabilityCalibrator":
        """Load calibrator from disk."""
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        calibrator = cls(config=data['config'])
        calibrator._calibrator = data['calibrator']
        calibrator._is_fitted = data['is_fitted']
        calibrator._fit_result = data['fit_result']
        
        logger.info(f"Loaded calibrator from {filepath}")
        return calibrator


# =========================================================================
# Internal Calibrator Classes
# =========================================================================

class _IdentityCalibrator:
    """Identity calibrator (no-op)."""
    params_ = {}
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        return x


class _IsotonicCalibrator:
    """Wrapper for isotonic regression."""
    model_ = None
    params_ = {}
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model_.predict(x)


class _PlattCalibrator:
    """Wrapper for Platt scaling."""
    model_ = None
    params_ = {}
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(x.reshape(-1, 1))[:, 1]


class _TemperatureCalibrator:
    """Temperature scaling calibrator."""
    temperature_ = 1.0
    params_ = {}
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        eps = 1e-8
        x_clipped = np.clip(x, eps, 1 - eps)
        logit = np.log(x_clipped / (1 - x_clipped))
        scaled_logit = logit / self.temperature_
        return 1 / (1 + np.exp(-scaled_logit))


class _BetaCalibrator:
    """Beta calibration."""
    a_ = 1.0
    b_ = 0.0
    c_ = 0.0
    params_ = {}
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        from scipy.special import logit, expit
        
        eps = 1e-8
        x_clipped = np.clip(x, eps, 1 - eps)
        logit_x = logit(x_clipped)
        return expit(self.a_ * logit_x + self.b_ + self.c_ * x_clipped)


# =========================================================================
# Convenience Functions for Integration with StandardizedXGBTrainer
# =========================================================================

def calibrate_oof_predictions(
    oof_results,
    y_true: Union[pd.Series, np.ndarray],
    method: str = "isotonic"
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Convenience function to calibrate StandardizedXGBTrainer OOF results.
    
    Args:
        oof_results: XGBTrainingResults from StandardizedXGBTrainer
        y_true: True labels aligned with predictions
        method: Calibration method ("isotonic", "platt", "temperature", "beta")
        
    Returns:
        Tuple of (calibrated_predictions, calibration_metrics)
    
    Example:
        ```python
        results = trainer.train_and_predict(X, y, data_start, data_end)
        calibrated, metrics = calibrate_oof_predictions(results, y, method="isotonic")
        ```
    """
    config = OOFCalibrationConfig(method=method)
    calibrator = OOFProbabilityCalibrator(config)
    
    calibrated = calibrator.fit_transform(
        oof_predictions=oof_results.oof_predictions,
        y_true=y_true,
        data_index=oof_results.oof_predictions.index if hasattr(oof_results.oof_predictions, 'index') else None
    )
    
    return calibrated, calibrator.get_calibration_metrics()


def get_recommended_calibration_method(n_samples: int) -> str:
    """
    Get recommended calibration method based on sample size.
    
    Args:
        n_samples: Number of OOF samples
        
    Returns:
        Recommended method name
    """
    if n_samples < 200:
        return "platt"  # Parametric, works with small samples
    elif n_samples < 500:
        return "temperature"  # Simple, robust
    else:
        return "isotonic"  # Non-parametric, flexible, best with enough data
