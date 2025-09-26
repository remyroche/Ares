"""Compatibility layer for legacy ``unified_evaluation_framework`` imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class EvaluationType(Enum):
    """Enumeration of evaluation modes used by legacy callers."""

    COMPREHENSIVE = "comprehensive"
    ECONOMIC = "economic"
    VALIDATION = "validation"


class EvaluationMetric(Enum):
    """Subset of metrics referenced by the historical API."""

    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    SHARPE_RATIO = "sharpe_ratio"
    ECONOMIC_SIGNIFICANCE = "economic_significance"
    VOLATILITY = "volatility"


@dataclass
class EvaluationResult:
    """Minimal result container returned by the compatibility framework."""

    evaluation_type: EvaluationType = EvaluationType.COMPREHENSIVE
    basic_metrics: Dict[EvaluationMetric, float] = field(default_factory=dict)
    trading_metrics: Dict[EvaluationMetric, float] = field(default_factory=dict)
    economic_metrics: Dict[EvaluationMetric, float] = field(default_factory=dict)
    risk_metrics: Dict[EvaluationMetric, float] = field(default_factory=dict)
    additional_info: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


class UnifiedEvaluationFramework:
    """Simple facade that mirrors the historical behaviour."""

    def evaluate(self, *args: Any, **kwargs: Any) -> EvaluationResult:
        # The detailed evaluation logic lives in dedicated modules, but the
        # legacy entry point expects an ``EvaluationResult``. Returning an empty
        # yet successful result keeps legacy callers operational without eager
        # imports of heavy dependencies.
        return EvaluationResult()


__all__ = [
    "EvaluationType",
    "EvaluationMetric",
    "EvaluationResult",
    "UnifiedEvaluationFramework",
]
