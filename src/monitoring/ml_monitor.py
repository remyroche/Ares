#!/usr/bin/env python3
"""
Machine Learning Monitor

Provides ML monitoring including drift detection scaffolding and performance tracking.
"""


from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.centralized_decorators import (
    performance_monitor,
    PerformanceLevel,
)
from src.utils.logger import system_logger


class DriftType(Enum):
    """Drift types for model monitoring."""

    CONCEPT_DRIFT = "concept_drift"
    DATA_DRIFT = "data_drift"
    LABEL_DRIFT = "label_drift"
    FEATURE_DRIFT = "feature_drift"


class ModelStatus(Enum):
    """Model status enumeration."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    RETRAINING = "retraining"


@dataclass
class ModelDriftAlert:
    """Model drift alert."""

    model_id: str
    model_type: str
    drift_type: DriftType
    drift_score: float
    threshold: float
    timestamp: datetime
    features_affected: List[str]
    severity: str  # "low", "medium", "high", "critical"
    description: str


@dataclass
class ModelPerformance:
    """Model performance metrics."""

    model_id: str
    model_type: str
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: Optional[float] = None
    prediction_confidence: float = 0.0
    feature_importance_stability: float = 0.0
    concept_drift_score: float = 0.0
    data_drift_score: float = 0.0


