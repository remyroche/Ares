"""Bootstrap-based overfitting checks for NAS/TAS evaluation pipelines.

This module intentionally builds on the existing NAS/TAS overfitting
infrastructure so that the new SPA / Reality Check guard rails do not drift
from the mature protections that already ship with the platform.  The helper
classes defined here orchestrate Hansen's SPA test together with the
``UniversalOverfittingDetector`` (and, when available, the learning-curve
enhanced detector) so consumers get a single, consistent entry point for
overfitting due diligence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional heavy dependency tree
    from ..advanced_validation import (
        OverfittingConfig,
        OverfittingReport,
        UniversalOverfittingDetector,
    )

    _HAS_UNIVERSAL_DETECTOR = True
except Exception as exc:  # pragma: no cover - fallback path
    logger.warning("Universal overfitting detector unavailable: %s", exc)
    UniversalOverfittingDetector = None  # type: ignore[assignment]
    OverfittingConfig = None  # type: ignore[assignment]
    OverfittingReport = None  # type: ignore[assignment]
    _HAS_UNIVERSAL_DETECTOR = False

try:  # pragma: no cover - optional dependency
    from ..advanced_overfitting_detection import (
        EnhancedOverfittingDetectorWithLearningCurves,
    )

    _HAS_ENHANCED_DETECTOR = True
except Exception as exc:  # pragma: no cover - fallback path
    logger.warning("Enhanced overfitting detector unavailable: %s", exc)
    EnhancedOverfittingDetectorWithLearningCurves = None  # type: ignore[assignment]
    _HAS_ENHANCED_DETECTOR = False


@dataclass
class SPAResult:
    p_value: float
    threshold: float
    passes: bool


@dataclass
class OverfittingProtectionResult:
    """Result bundle combining SPA and legacy overfitting defences."""

    spa: SPAResult
    universal_report: Optional["OverfittingReport"]
    enhanced_report: Optional["OverfittingReport"]
    passes: bool


def hansen_spa_test(
    scores: Iterable[float],
    null_hypothesis: float = 0.0,
    n_bootstrap: int = 500,
    transaction_cost_bp: float = 0.0,
    random_state: int = 42,
) -> SPAResult:
    """Run a simplified Hansen's SPA / White's Reality Check."""

    scores = np.asarray(list(scores), dtype=float)
    if scores.size == 0:
        return SPAResult(p_value=1.0, threshold=null_hypothesis, passes=False)

    adjusted_scores = scores - null_hypothesis - transaction_cost_bp / 10_000.0
    observed_stat = np.max(adjusted_scores)

    rng = np.random.default_rng(random_state)
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        resampled = rng.choice(adjusted_scores, size=adjusted_scores.size, replace=True)
        bootstrap_stats.append(np.max(resampled))

    bootstrap_stats = np.asarray(bootstrap_stats)
    p_value = float(np.mean(bootstrap_stats >= observed_stat))
    threshold = float(np.percentile(bootstrap_stats, 95))
    passes = bool(observed_stat > threshold)
    return SPAResult(p_value=p_value, threshold=threshold, passes=passes)


class OverfittingDefenseSuite:
    """Combine SPA testing with the platform's established detectors."""

    def __init__(
        self,
        *,
        spa_null_hypothesis: float = 0.0,
        spa_bootstrap_samples: int = 500,
        spa_transaction_cost_bp: float = 0.0,
        spa_random_state: int = 42,
        detector: Optional["UniversalOverfittingDetector"] = None,
        enhanced_detector: Optional["EnhancedOverfittingDetectorWithLearningCurves"] = None,
    ) -> None:
        self._spa_config = {
            "null_hypothesis": spa_null_hypothesis,
            "n_bootstrap": spa_bootstrap_samples,
            "transaction_cost_bp": spa_transaction_cost_bp,
            "random_state": spa_random_state,
        }
        self.detector = detector if detector is not None else self._build_detector()
        self.enhanced_detector = (
            enhanced_detector
            if enhanced_detector is not None
            else self._build_enhanced_detector()
        )

    def _build_detector(self) -> Optional["UniversalOverfittingDetector"]:
        if not _HAS_UNIVERSAL_DETECTOR:
            return None
        try:
            config = OverfittingConfig() if OverfittingConfig is not None else None
            return UniversalOverfittingDetector(config)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Failed to initialise universal overfitting detector: %s", exc)
            return None

    def _build_enhanced_detector(
        self,
    ) -> Optional["EnhancedOverfittingDetectorWithLearningCurves"]:
        if not _HAS_ENHANCED_DETECTOR:
            return None
        try:
            return EnhancedOverfittingDetectorWithLearningCurves()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Failed to initialise enhanced overfitting detector: %s", exc)
            return None

    def evaluate(
        self,
        *,
        sharpe_like_scores: Iterable[float],
        train_predictions: Optional[Iterable[Any]] = None,
        val_predictions: Optional[Iterable[Any]] = None,
        train_labels: Optional[Iterable[Any]] = None,
        val_labels: Optional[Iterable[Any]] = None,
        train_probabilities: Optional[Iterable[Any]] = None,
        val_probabilities: Optional[Iterable[Any]] = None,
        feature_importance: Optional[Iterable[float]] = None,
        model: Optional[Any] = None,
        X_train: Optional[Any] = None,
        X_val: Optional[Any] = None,
        y_train: Optional[Any] = None,
        y_val: Optional[Any] = None,
        X_test: Optional[Any] = None,
        y_test: Optional[Any] = None,
        model_name: str = "unknown",
        model_type: str = "unknown",
        fold_number: Optional[int] = None,
        spa_overrides: Optional[Dict[str, Any]] = None,
    ) -> OverfittingProtectionResult:
        """Run SPA alongside the legacy detectors and consolidate the verdict."""

        spa_kwargs = dict(self._spa_config)
        if spa_overrides:
            spa_kwargs.update(spa_overrides)
        spa_result = hansen_spa_test(sharpe_like_scores, **spa_kwargs)

        universal_report: Optional["OverfittingReport"] = None
        if (
            self.detector is not None
            and train_predictions is not None
            and val_predictions is not None
            and train_labels is not None
            and val_labels is not None
        ):
            try:
                universal_report = self.detector.detect_overfitting(
                    np.asarray(list(train_predictions)),
                    np.asarray(list(val_predictions)),
                    np.asarray(list(train_labels)),
                    np.asarray(list(val_labels)),
                    None if train_probabilities is None else np.asarray(list(train_probabilities)),
                    None if val_probabilities is None else np.asarray(list(val_probabilities)),
                    None if feature_importance is None else np.asarray(list(feature_importance)),
                    model_name=model_name,
                    model_type=model_type,
                    fold_number=fold_number,
                )
            except Exception as exc:  # pragma: no cover - logging only
                logger.warning("Universal overfitting detection failed: %s", exc)

        enhanced_report: Optional["OverfittingReport"] = None
        if (
            self.enhanced_detector is not None
            and model is not None
            and X_train is not None
            and X_val is not None
            and y_train is not None
            and y_val is not None
        ):
            try:
                enhanced_report = self.enhanced_detector.detect_overfitting_with_learning_curves(
                    model,
                    X_train,
                    X_val,
                    y_train,
                    y_val,
                    X_test=X_test,
                    y_test=y_test,
                    model_name=model_name,
                    model_type=model_type,
                    fold_number=fold_number,
                )
            except Exception as exc:  # pragma: no cover - logging only
                logger.warning("Enhanced overfitting detection failed: %s", exc)

        passes = spa_result.passes
        if universal_report is not None:
            passes = passes and not getattr(universal_report, "is_overfitting", False)
        if enhanced_report is not None:
            passes = passes and not getattr(enhanced_report, "is_overfitting", False)

        return OverfittingProtectionResult(
            spa=spa_result,
            universal_report=universal_report,
            enhanced_report=enhanced_report,
            passes=passes,
        )


__all__ = [
    "SPAResult",
    "OverfittingProtectionResult",
    "OverfittingDefenseSuite",
    "hansen_spa_test",
]
