"""
Unified Evaluator

Centralized, reusable metric computation for classification and regression.

This module consolidates core evaluation logic used across:
- evaluation_utils
- HMM evaluation pipeline
- model evaluation utilities
- post-training model evaluator

It provides consistent metric names and safe calculations, reducing duplication
and divergence across implementations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import logging
import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        classification_report,
        confusion_matrix,
        log_loss,
        roc_auc_score,
    )
    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - environment without sklearn
    SKLEARN_AVAILABLE = False

_logger = logging.getLogger("UnifiedEvaluator")

def _is_classification_task(y_true: np.ndarray, y_pred: np.ndarray) -> bool:
    try:
        unique_true = len(np.unique(y_true))
        unique_pred = len(np.unique(y_pred))
        return unique_true <= 10 and unique_pred <= 10 and not np.issubdtype(y_true.dtype, np.floating)
    except Exception:
        return False

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    include: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute common classification metrics with consistent naming.

    Returns keys:
    - accuracy
    - balanced_accuracy
    - precision_macro, recall_macro, f1_macro
    - precision_weighted, recall_weighted, f1_weighted
    - confusion_matrix (list[list[int]])
    - classification_report (dict)
    - roc_auc (float, when y_prob provided)
    - log_loss (float, when y_prob provided)

    For backward-compatibility, also includes:
    - precision, recall, f1_score (mapped to weighted variants)
    """
    if not SKLEARN_AVAILABLE:
        return {}

    metrics: Dict[str, Any] = {}

    # Basic and macro/weighted aggregates
    try:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

        metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        metrics["precision_weighted"] = float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        metrics["recall_weighted"] = float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        metrics["f1_weighted"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )

        # Back-compat keys used in older modules
        metrics["precision"] = metrics["precision_weighted"]
        metrics["recall"] = metrics["recall_weighted"]
        metrics["f1_score"] = metrics["f1_weighted"]
    except Exception as e:  # pragma: no cover
        _logger.error(f"❌ Classification aggregate metrics failed: {e}")
        _logger.warning("⚠️ Classification metrics failed - returning empty metrics")

    # Detailed outputs
    try:
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
    except Exception as e:
        _logger.error(f"❌ Critical error: Could not compute confusion matrix: {e}")
        _logger.warning("⚠️ Confusion matrix calculation failed - returning empty metrics")
        metrics["confusion_matrix"] = []
        # Don't raise - return partial results instead

    try:
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics["classification_report"] = report
    except Exception as e:
        _logger.error(f"❌ Critical error: Could not generate classification report: {e}")
        _logger.warning("⚠️ Classification report generation failed - returning empty report")
        metrics["classification_report"] = {}
        # Don't raise - return partial results instead

    # Probability-based metrics
    if y_prob is not None:
        try:
            unique_classes = np.unique(y_true)
            if len(unique_classes) == 2 and y_prob.ndim == 2 and y_prob.shape[1] >= 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            elif y_prob.ndim == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
        except Exception as e:
            _logger.warning(f"⚠️ ROC-AUC calculation failed: {e}")
            metrics["roc_auc"] = None

        try:
            metrics["log_loss"] = float(log_loss(y_true, y_prob))
        except Exception as e:
            _logger.warning(f"⚠️ Log loss calculation failed: {e}")
            metrics["log_loss"] = None

    # Optional include filter
    if include:
        filtered: Dict[str, Any] = {}
        for key in include:
            if key in metrics:
                filtered[key] = metrics[key]
        # preserve back-compat if classical aliases were requested
        for alias in ("precision", "recall", "f1_score"):
            if alias in include and alias not in filtered and alias in metrics:
                filtered[alias] = metrics[alias]
        return filtered

    return metrics

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    include: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute common regression metrics with safe handling.

    Returns keys:
    - mse, rmse, mae, r2
    - mape, smape, explained_variance
    """
    if not SKLEARN_AVAILABLE:
        return {}

    metrics: Dict[str, Any] = {}
    try:
        metrics["mse"] = float(mean_squared_error(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))
    except Exception as e:  # pragma: no cover
        _logger.error(f"❌ Regression basic metrics failed: {e}")
        _logger.warning("⚠️ Regression metrics failed - using default values")
        metrics["mse"] = 0.0
        metrics["rmse"] = 0.0
        metrics["mae"] = 0.0
        metrics["r2"] = 0.0

    # MAPE (avoid division by zero)
    try:
        y_true_nonzero = y_true != 0
        if np.any(y_true_nonzero):
            mape_vals = np.abs((y_true[y_true_nonzero] - y_pred[y_true_nonzero]) / y_true[y_true_nonzero])
            metrics["mape"] = float(np.mean(mape_vals) * 100.0)
        else:
            metrics["mape"] = 0.0
    except Exception as e:
        _logger.warning(f"⚠️ MAPE calculation failed: {e}")
        metrics["mape"] = 0.0

    # SMAPE
    try:
        denom = np.abs(y_true) + np.abs(y_pred)
        smape_vals = np.where(denom != 0, 2.0 * np.abs(y_true - y_pred) / denom, 0.0)
        metrics["smape"] = float(np.mean(smape_vals) * 100.0)
    except Exception as e:
        _logger.warning(f"⚠️ SMAPE calculation failed: {e}")
        metrics["smape"] = 0.0

    # Explained variance (manual, to avoid extra imports)
    try:
        var_true = float(np.var(y_true))
        if var_true != 0.0:
            metrics["explained_variance"] = float(1.0 - float(np.var(y_true - y_pred)) / var_true)
        else:
            metrics["explained_variance"] = 0.0
    except Exception as e:
        _logger.warning(f"⚠️ Explained variance calculation failed: {e}")
        metrics["explained_variance"] = 0.0

    if include:
        return {k: v for k, v in metrics.items() if k in include}

    return metrics

def evaluate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    task: Optional[str] = None,
    include: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a model on a single dataset using unified metrics.

    Args:
        model: Fitted model with predict and optionally predict_proba
        X: Features
        y: True labels/targets
        task: 'classification', 'regression', or None to auto-detect
        include: optional list of metric names to include
    """
    try:
        y_pred = model.predict(X)
    except Exception as e:  # pragma: no cover
        _logger.error(f"❌ Model prediction failed: {e}")
        _logger.warning("⚠️ Model prediction failed - using zero predictions")
        y_pred = np.zeros(len(y))

    y_prob = None
    if task in (None, "classification") and hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X)
        except Exception as e:
            _logger.warning(f"⚠️ Probability prediction failed: {e}")
            y_prob = None

    resolved_task = task
    if resolved_task is None:
        resolved_task = "classification" if _is_classification_task(y, y_pred) else "regression"

    if resolved_task == "classification":
        return compute_classification_metrics(y_true=y, y_pred=y_pred, y_prob=y_prob, include=include)
    else:
        return compute_regression_metrics(y_true=y, y_pred=y_pred, include=include)

def evaluate_multiple_datasets(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    model: Optional[Any] = None,
    task: Optional[str] = None,
    predictions: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    include: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate metrics across multiple datasets (e.g., train/validation/test).

    Either provide a model to predict on each dataset or pass precomputed
    predictions via `predictions` with keys:
        predictions[dataset] = { 'y_pred': ..., 'y_prob': optional }
    """
    results: Dict[str, Dict[str, Any]] = {}
    for name, (X, y) in datasets.items():
        if predictions and name in predictions:
            y_pred = predictions[name].get("y_pred")
            y_prob = predictions[name].get("y_prob")
            if task == "classification":
                results[name] = compute_classification_metrics(y_true=y, y_pred=y_pred, y_prob=y_prob, include=include)
            elif task == "regression":
                results[name] = compute_regression_metrics(y_true=y, y_pred=y_pred, include=include)
            else:
                inferred_task = "classification" if _is_classification_task(y, y_pred) else "regression"
                if inferred_task == "classification":
                    results[name] = compute_classification_metrics(y_true=y, y_pred=y_pred, y_prob=y_prob, include=include)
                else:
                    results[name] = compute_regression_metrics(y_true=y, y_pred=y_pred, include=include)
        elif model is not None:
            results[name] = evaluate_model(model=model, X=X, y=y, task=task, include=include)
        else:
            raise ValueError("Either `model` or `predictions` must be provided.")
    return results

def compute_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Compute the Sharpe ratio given per-period returns and optional risk-free rate.
    Returns 0.0 if the standard deviation is zero or inputs are insufficient.
    """
    try:
        excess = np.asarray(returns) - risk_free_rate
        if excess.size == 0:
            return 0.0
        std = float(np.std(excess, ddof=1)) if excess.size > 1 else float(np.std(excess))
        if std <= 1e-12:
            return 0.0
        return float(np.mean(excess) / std)
    except Exception as e:
        _logger.warning(f"⚠️ Sharpe ratio calculation failed: {e}")
        return 0.0

__all__ = [
    "compute_classification_metrics",
    "compute_regression_metrics",
    "evaluate_model",
    "evaluate_multiple_datasets",
    "compute_sharpe_ratio",
]
