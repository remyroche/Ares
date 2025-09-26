"""Shared calibration utilities for post-training model calibration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:  # Optional dependency - sklearn might not be installed in minimal environments
    from sklearn.exceptions import NotFittedError
except Exception:  # pragma: no cover - fallback when sklearn is unavailable
    NotFittedError = RuntimeError  # type: ignore

try:
    from src.utils.ml_common.validation.thresholding import calibrate_probabilities
    _CALIBRATION_AVAILABLE = True
except Exception:  # pragma: no cover - calibration utilities missing
    calibrate_probabilities = None  # type: ignore
    _CALIBRATION_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelCalibrationConfig:
    """Configuration for model calibration utilities."""

    method: str = "isotonic"
    cv: int = 3
    min_samples: int = 100
    validation_split: float = 0.2
    enforce_probabilistic: bool = True
    skip_models_without_proba: bool = True


CalibrationResult = Dict[str, Any]


class ModelCalibrationManager:
    """Utility class that performs post-training calibration for ML models."""

    def __init__(self, config: Optional[ModelCalibrationConfig] = None):
        self.config = config or ModelCalibrationConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def calibrate_model(
        self,
        model: Any,
        X_calibration: np.ndarray | None,
        y_calibration: np.ndarray | None,
    ) -> Tuple[Any, CalibrationResult]:
        """Calibrate a single trained model.

        Args:
            model: Trained model instance.
            X_calibration: Features used for calibration.
            y_calibration: Targets used for calibration.

        Returns:
            Tuple of calibrated model (or original if calibration failed)
            and calibration metadata.
        """

        metadata: CalibrationResult = {
            "calibrated": False,
            "samples_used": 0,
            "method": self.config.method,
            "cv": self.config.cv,
            "reason": None,
        }

        if model is None:
            metadata["reason"] = "model_missing"
            return model, metadata

        if not _CALIBRATION_AVAILABLE or calibrate_probabilities is None:
            metadata["reason"] = "calibration_utility_unavailable"
            return model, metadata

        if not self._can_calibrate_model(model):
            metadata["reason"] = "predict_proba_unavailable"
            return model, metadata

        if X_calibration is None or y_calibration is None:
            metadata["reason"] = "calibration_data_missing"
            return model, metadata

        X_calibration = self._to_numpy(X_calibration)
        y_calibration = self._to_numpy(y_calibration)
        if X_calibration.size == 0 or y_calibration.size == 0:
            metadata["reason"] = "calibration_data_empty"
            return model, metadata

        if len(X_calibration) < self.config.min_samples:
            metadata["reason"] = "insufficient_samples"
            metadata["samples_used"] = len(X_calibration)
            return model, metadata

        try:
            calibrated_model = calibrate_probabilities(
                estimator=model,
                X_train=X_calibration,
                y_train=y_calibration,
                method=self.config.method,
                cv=self.config.cv,
            )
        except NotFittedError as exc:  # pragma: no cover - depends on estimator state
            LOGGER.warning("Calibration failed because model is not fitted: %s", exc)
            metadata["reason"] = "model_not_fitted"
            return model, metadata
        except Exception as exc:  # pragma: no cover - runtime safeguard
            LOGGER.warning("Calibration failed: %s", exc)
            metadata["reason"] = f"exception:{exc}"
            return model, metadata

        metadata["samples_used"] = int(len(X_calibration))
        metadata["classes"] = int(len(np.unique(y_calibration)))
        metadata["reason"] = None
        metadata["calibrated"] = calibrated_model is not model

        if metadata["calibrated"]:
            setattr(calibrated_model, "_is_calibrated", True)
        else:
            # Even if calibration returned the original model we can still
            # annotate it so downstream consumers know calibration was attempted.
            setattr(model, "_is_calibrated", getattr(model, "_is_calibrated", False) or True)

        return calibrated_model, metadata

    def calibrate_models_dict(
        self,
        models: Dict[str, Any],
        X_calibration: np.ndarray | None,
        y_calibration: np.ndarray | None,
    ) -> Tuple[Dict[str, Any], Dict[str, CalibrationResult]]:
        """Calibrate multiple models provided in a dictionary."""

        calibrated_models: Dict[str, Any] = {}
        calibration_reports: Dict[str, CalibrationResult] = {}

        for name, model in models.items():
            calibrated_model, report = self.calibrate_model(model, X_calibration, y_calibration)
            calibrated_models[name] = calibrated_model
            calibration_reports[name] = report

        return calibrated_models, calibration_reports

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _can_calibrate_model(self, model: Any) -> bool:
        if model is None:
            return False
        if hasattr(model, "predict_proba"):
            return True
        if not self.config.enforce_probabilistic and hasattr(model, "decision_function"):
            return True
        return not self.config.skip_models_without_proba

    @staticmethod
    def _to_numpy(array_like: Any) -> np.ndarray:
        if array_like is None:
            return np.array([])
        if isinstance(array_like, np.ndarray):
            return array_like
        try:
            return np.asarray(array_like)
        except Exception:  # pragma: no cover - fallback when conversion fails
            return np.array([])
