"""Model evaluation helpers exposed via :mod:`src.utils.ml_common.models`.

Historically this package re-exported a ``ModelEvaluator`` class that did not
exist, causing imports to fail at runtime.  The implementation below provides a
lean yet practical evaluator that builds upon the shared metric utilities in
``ml_common.evaluation`` and optionally integrates with the model registry for
metadata persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..evaluation.unified_evaluator import (
    compute_classification_metrics,
    compute_regression_metrics,
)
from .model_registry import ModelRegistry
from ..logger import get_logger


@dataclass
class EvaluationResult:
    """Structured representation of evaluation metrics."""

    metrics: Dict[str, Any]
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    task_type: str = "classification"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelEvaluator:
    """Evaluate models with consistent metric computation and logging."""

    def __init__(
        self,
        model: Any,
        *,
        task_type: str = "auto",
        model_name: Optional[str] = None,
        registry: Optional[ModelRegistry] = None,
    ) -> None:
        self.model = model
        self.task_type = task_type
        self.model_name = model_name or getattr(model, "__class__", type(model)).__name__
        self.registry = registry
        self.logger = get_logger("ModelEvaluator").getChild(self.model_name)

    def evaluate(
        self,
        X: pd.DataFrame | np.ndarray,
        y_true: pd.Series | np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """Run model inference and compute metrics for ``y_true``."""

        y_true_array = np.asarray(y_true)
        predictions = self._predict(X)
        inferred_task = self._infer_task_type(y_true_array, predictions)
        probabilities = self._predict_proba(X) if inferred_task == "classification" else None

        metrics = self._compute_metrics(
            inferred_task,
            y_true_array,
            predictions,
            probabilities,
            sample_weight=sample_weight,
        )

        result_metadata = metadata.copy() if metadata else {}
        result_metadata.setdefault("model_name", self.model_name)
        result_metadata.setdefault("task_type", inferred_task)

        evaluation = EvaluationResult(
            metrics=metrics,
            predictions=predictions,
            probabilities=probabilities,
            task_type=inferred_task,
            metadata=result_metadata,
        )

        self._log_metrics(evaluation)
        self._persist_metadata(evaluation)
        return evaluation

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict"):
            return np.asarray(self.model.predict(X))
        raise AttributeError("Model does not implement a predict method")

    def _predict_proba(self, X: pd.DataFrame | np.ndarray) -> Optional[np.ndarray]:
        predict_proba = getattr(self.model, "predict_proba", None)
        if callable(predict_proba):
            try:
                return np.asarray(predict_proba(X))
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.debug("predict_proba failed: %s", exc)
        return None

    def _infer_task_type(self, y_true: np.ndarray, predictions: np.ndarray) -> str:
        if self.task_type != "auto":
            return self.task_type

        if np.issubdtype(y_true.dtype, np.floating) and np.issubdtype(predictions.dtype, np.floating):
            return "regression"

        unique_classes = np.unique(y_true)
        if unique_classes.size <= 20:
            return "classification"

        return "regression"

    def _compute_metrics(
        self,
        task_type: str,
        y_true: np.ndarray,
        predictions: np.ndarray,
        probabilities: Optional[np.ndarray],
        *,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        if task_type == "classification":
            metrics = compute_classification_metrics(y_true, predictions, probabilities)
        else:
            metrics = compute_regression_metrics(y_true, predictions)

        if sample_weight is not None:
            metrics["sample_weight_sum"] = float(np.sum(sample_weight))

        return metrics

    def _log_metrics(self, evaluation: EvaluationResult) -> None:
        summary = {
            key: value
            for key, value in evaluation.metrics.items()
            if isinstance(value, (int, float))
        }
        self.logger.info("Evaluation metrics: %s", summary)

    def _persist_metadata(self, evaluation: EvaluationResult) -> None:
        if not self.registry:
            return

        metadata = evaluation.metadata.copy()
        metadata.update({"metrics": evaluation.metrics})
        try:
            self.registry._update_registry_entry(self.model_name, "latest", metadata)
        except Exception as exc:  # pragma: no cover - registry is optional
            self.logger.warning("Failed to persist evaluation metadata: %s", exc)


__all__ = ["ModelEvaluator", "EvaluationResult"]

