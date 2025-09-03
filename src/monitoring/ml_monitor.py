#!/usr/bin/env python3
"""
Machine Learning Monitor

Provides ML monitoring including drift detection scaffolding and performance tracking.
"""
from src.core.decorators import (
    handles_errors,
    log_execution_time
)

from src.core.domain import PerformanceLevel

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from src.utils.logger import system_logger
from datetime import datetime
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
    features_affected: list[str]
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
    auc_score: float | None = None
    prediction_confidence: float = 0.0
    feature_importance_stability: float = 0.0
    concept_drift_score: float = 0.0
    data_drift_score: float = 0.0

class MLMonitor:
    """
    ML Monitor with drift detection scaffolding and performance tracking.
    """
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("MLMonitor")

        self.ml_config = config.get("ml_monitor", {})
        self.enable_online_learning: bool = bool(self.ml_config.get("enable_online_learning", True))
        self.drift_detection_enabled: bool = bool(self.ml_config.get("drift_detection_enabled", True))
        self.feature_importance_tracking: bool = bool(
            self.ml_config.get("feature_importance_tracking", True),
        )
        self.auto_retraining_enabled: bool = bool(self.ml_config.get("auto_retraining_enabled", True))

        self.performances: list[ModelPerformance] = []
        self.alerts: list[ModelDriftAlert] = []

    @log_execution_time(level=PerformanceLevel.DETAILED)
    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid ML monitor configuration"),
            AttributeError: (False, "Missing ML monitor parameters"),
        },
        default_return=False,
        context="ml_monitor.initialize",
    )
    async def initialize(self) -> bool:
        self.logger.info("Initializing ML Monitor ...")
        self.performances.clear()
        self.alerts.clear()
        self.logger.info("✅ ML Monitor initialization completed")
        return True

    @handles_errors(default_return=None, context="ml_monitor.record_performance")
    async def record_performance(self, perf: ModelPerformance) -> None:
        self.performances.append(perf)

    def get_latest_performance(self, model_id: str) -> ModelPerformance | None:
        for p in reversed(self.performances):
            if p.model_id == model_id:
                return p
        return None

    def list_alerts(self) -> list[ModelDriftAlert]:
        return list(self.alerts)
