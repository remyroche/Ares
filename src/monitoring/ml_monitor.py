#!/usr/bin/env python3
"""
Machine Learning Monitor

This module provides comprehensive ML monitoring including online learning algorithms,
model drift detection, feature importance tracking, and automated retraining triggers.
"""

import asyncio
import contextlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    failed,
    initialization_error,
)


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


@dataclass
class FeatureImportance:
    """Feature importance tracking."""

    feature_name: str
    importance_score: float
    importance_rank: int
    stability_score: float
    drift_score: float
    timestamp: datetime


@dataclass
class OnlineLearningMetrics:
    """Online learning metrics."""

    model_id: str
    learning_rate: float
    adaptation_speed: float
    knowledge_retention: float
    forgetting_curve: list[float]
    performance_trend: list[float]
    last_update: datetime


class MLMonitor:
    """
    Machine Learning Monitor with online learning capabilities,
    drift detection, and automated retraining triggers.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize ML monitor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("MLMonitor")

        # ML monitor configuration
        self.ml_config = config.get("ml_monitor", {})
        self.enable_online_learning = self.ml_config.get("enable_online_learning", True)
        self.drift_detection_enabled = self.ml_config.get(
            "drift_detection_enabled",
            True,
        )
        self.feature_importance_tracking = self.ml_config.get(
            "feature_importance_tracking",
            True,
        )
        self.auto_retraining_enabled = self.ml_config.get(
            "auto_retraining_enabled",
            True,
        )
        self.drift_threshold = self.ml_config.get("drift_threshold", 0.1)

        # Monitoring intervals
        self.drift_check_interval = self.ml_config.get(
            "drift_check_interval",
            300,
        )  # 5 minutes
        self.performance_check_interval = self.ml_config.get(
            "performance_check_interval",
            60,
        )  # 1 minute
        self.feature_analysis_interval = self.ml_config.get(
            "feature_analysis_interval",
            600,
        )  # 10 minutes

        # Storage
        self.model_performance_history: dict[str, list[ModelPerformance]] = {}
        self.feature_importance_history: dict[str, list[FeatureImportance]] = {}
        self.drift_alerts: list[ModelDriftAlert] = []
        self.online_learning_metrics: dict[str, OnlineLearningMetrics] = {}

        # Reference data for drift detection
        self.reference_distributions: dict[str, dict[str, float]] = {}
        self.reference_performance: dict[str, float] = {}
        self.reference_feature_importance: dict[str, dict[str, float]] = {}

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_tasks: list[asyncio.Task] = []

        self.logger.info("🤖 ML Monitor initialized")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid ML monitor configuration"),
            AttributeError: (False, "Missing required ML monitor parameters"),
        },
        default_return=False,
        context="ML monitor initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the ML monitor."""
        try:
            self.logger.info("Initializing ML Monitor...")

            # Initialize reference data
            await self._load_reference_data()

            # Initialize drift detection
            if self.drift_detection_enabled:
                await self._initialize_drift_detection()

            # Initialize feature tracking
            if self.feature_importance_tracking:
                await self._initialize_feature_tracking()

            # Initialize online learning
            if self.enable_online_learning:
                await self._initialize_online_learning()

            self.logger.info("✅ ML Monitor initialization completed")
            return True

        except Exception:
            self.logger.exception(failed("❌ ML Monitor initialization failed: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="reference data loading",
    )
    async def _load_reference_data(self) -> None:
        """Load reference data for drift detection."""
        try:
            # This would load reference data from training or historical data
            # For now, create sample reference data
            self.reference_distributions = {
                "feature_1": {"mean": 0.0, "std": 1.0},
                "feature_2": {"mean": 0.0, "std": 1.0},
            }

            self.reference_performance = {
                "model_1": 0.85,
                "model_2": 0.82,
            }

            self.reference_feature_importance = {
                "model_1": {"feature_1": 0.6, "feature_2": 0.4},
                "model_2": {"feature_1": 0.5, "feature_2": 0.5},
            }

            self.logger.info("Reference data loaded")

        except Exception:
            self.logger.exception(error("Error loading reference data: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="drift detection initialization",
    )
    async def _initialize_drift_detection(self) -> None:
        """Initialize drift detection."""
        try:
            # Initialize drift detection structures
            self.drift_alerts.clear()

            self.logger.info("Drift detection initialized")

        except Exception:
            self.logger.exception(initialization_error("Error initializing drift detection: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="feature tracking initialization",
    )
    async def _initialize_feature_tracking(self) -> None:
        """Initialize feature importance tracking."""
        try:
            # Initialize feature tracking structures
            self.feature_importance_history.clear()

            self.logger.info("Feature importance tracking initialized")

        except Exception:
            self.logger.exception(initialization_error("Error initializing feature tracking: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="online learning initialization",
    )
    async def _initialize_online_learning(self) -> None:
        """Initialize online learning monitoring."""
        try:
            # Initialize online learning structures
            self.online_learning_metrics.clear()

            self.logger.info("Online learning monitoring initialized")

        except Exception:
            self.logger.exception(initialization_error("Error initializing online learning: {e}"))

    @handle_specific_errors(
        error_handlers={
            Exception: (False, "ML monitoring failed"),
        },
        default_return=False,
        context="ML monitoring",
    )
    async def start_monitoring(self) -> bool:
        """Start ML monitoring."""
        try:
            self.is_monitoring = True

            # Start monitoring tasks
            if self.drift_detection_enabled:
                drift_task = asyncio.create_task(self._drift_detection_loop())
                self.monitoring_tasks.append(drift_task)

            performance_task = asyncio.create_task(self._performance_monitoring_loop())
            self.monitoring_tasks.append(performance_task)

            if self.feature_importance_tracking:
                feature_task = asyncio.create_task(self._feature_analysis_loop())
                self.monitoring_tasks.append(feature_task)

            if self.enable_online_learning:
                online_task = asyncio.create_task(self._online_learning_loop())
                self.monitoring_tasks.append(online_task)

            self.logger.info("🚀 ML Monitor started")
            return True

        except Exception:
            self.logger.exception(error("Error starting ML monitoring: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="drift detection loop",
    )
    async def _drift_detection_loop(self) -> None:
        """Drift detection monitoring loop."""
        try:
            while self.is_monitoring:
                await self._perform_drift_detection()
                await asyncio.sleep(self.drift_check_interval)

        except Exception:
            self.logger.exception(error("Error in drift detection loop: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="performance monitoring loop",
    )
    async def _performance_monitoring_loop(self) -> None:
        """Performance monitoring loop."""
        try:
            while self.is_monitoring:
                await self._capture_performance_snapshots()
                await asyncio.sleep(self.performance_check_interval)

        except Exception:
            self.logger.exception(error("Error in performance monitoring loop: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="feature analysis loop",
    )
    async def _feature_analysis_loop(self) -> None:
        """Feature importance analysis loop."""
        try:
            while self.is_monitoring:
                await self._analyze_feature_importance()
                await asyncio.sleep(self.feature_analysis_interval)

        except Exception:
            self.logger.exception(error("Error in feature analysis loop: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="online learning loop",
    )
    async def _online_learning_loop(self) -> None:
        """Online learning monitoring loop."""
        try:
            while self.is_monitoring:
                await self._monitor_online_learning()
                await asyncio.sleep(self.performance_check_interval)

        except Exception:
            self.logger.exception(error("Error in online learning loop: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="drift detection",
    )
    async def _perform_drift_detection(self) -> None:
        """Perform drift detection for all models."""
        try:
            for model_id in self.model_performance_history:
                # Check concept drift
                concept_drift_score = self._calculate_concept_drift(model_id)
                if concept_drift_score > self.drift_threshold:
                    await self._create_drift_alert(
                        model_id,
                        "ensemble",
                        DriftType.CONCEPT_DRIFT,
                        concept_drift_score,
                        self.drift_threshold,
                    )

                # Check data drift
                data_drift_score = self._calculate_data_drift(model_id)
                if data_drift_score > self.drift_threshold:
                    await self._create_drift_alert(
                        model_id,
                        "ensemble",
                        DriftType.DATA_DRIFT,
                        data_drift_score,
                        self.drift_threshold,
                    )

                # Check feature drift
                feature_drift_score = self._calculate_feature_drift(model_id)
                if feature_drift_score > self.drift_threshold:
                    await self._create_drift_alert(
                        model_id,
                        "ensemble",
                        DriftType.FEATURE_DRIFT,
                        feature_drift_score,
                        self.drift_threshold,
                    )

        except Exception:
            self.logger.exception(error("Error performing drift detection: {e}"))

    def _calculate_concept_drift(self, model_id: str) -> float:
        """Calculate concept drift score for a model."""
        try:
            if model_id not in self.model_performance_history:
                return 0.0

            performance_history = self.model_performance_history[model_id]
            if len(performance_history) < 10:
                return 0.0

            # Calculate performance trend
            recent_performances = [p.accuracy for p in performance_history[-10:]]
            historical_performances = [p.accuracy for p in performance_history[:-10]]

            if len(historical_performances) < 10:
                return 0.0

            # Calculate drift using simple statistical comparison
            recent_mean = np.mean(recent_performances)
            historical_mean = np.mean(historical_performances)

            # Drift score based on mean difference
            drift_score = abs(recent_mean - historical_mean)

            return min(drift_score, 1.0)

        except Exception:
            self.logger.exception(error("Error calculating concept drift: {e}"))
            return 0.0

    def _calculate_data_drift(self, model_id: str) -> float:
        """Calculate data drift score for a model."""
        try:
            # This would compare current data distributions with reference distributions
            # For now, return a sample drift score
            return np.random.uniform(0.0, 0.2)

        except Exception:
            self.logger.exception(error("Error calculating data drift: {e}"))
            return 0.0

    def _calculate_feature_drift(self, model_id: str) -> float:
        """Calculate feature drift score for a model."""
        try:
            if model_id not in self.feature_importance_history:
                return 0.0

            feature_history = self.feature_importance_history[model_id]
            if len(feature_history) < 5:
                return 0.0

            # Calculate feature importance stability
            feature_names = {f.feature_name for f in feature_history}
            stability_scores = []

            for feature_name in feature_names:
                feature_scores = [
                    f.importance_score
                    for f in feature_history
                    if f.feature_name == feature_name
                ]

                if len(feature_scores) > 1:
                    # Calculate coefficient of variation
                    cv = np.std(feature_scores) / np.mean(feature_scores)
                    stability_scores.append(1.0 / (1.0 + cv))

            if stability_scores:
                return 1.0 - np.mean(stability_scores)
            return 0.0

        except Exception:
            self.logger.exception(error("Error calculating feature drift: {e}"))
            return 0.0

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="drift alert creation",
    )
    async def _create_drift_alert(
        self,
        model_id: str,
        model_type: str,
        drift_type: DriftType,
        drift_score: float,
        threshold: float,
    ) -> None:
        """Create a drift alert."""
        try:
            severity = self._determine_alert_severity(drift_score, threshold)

            alert = ModelDriftAlert(
                model_id=model_id,
                model_type=model_type,
                drift_type=drift_type,
                drift_score=drift_score,
                threshold=threshold,
                timestamp=datetime.now(),
                features_affected=[],  # Would be populated based on analysis
                severity=severity,
                description=f"{drift_type.value} detected for {model_id}",
            )

            self.drift_alerts.append(alert)

            # Trigger auto-retraining if enabled and severity is high
            if self.auto_retraining_enabled and severity in ["high", "critical"]:
                await self._trigger_auto_retraining(model_id, alert)

            self.logger.warning(
                f"Drift alert created: {model_id} - {drift_type.value} - {severity}",
            )

        except Exception:
            self.logger.exception(error("Error creating drift alert: {e}"))

    def _determine_alert_severity(self, drift_score: float, threshold: float) -> str:
        """Determine alert severity based on drift score."""
        if drift_score > threshold * 2:
            return "critical"
        if drift_score > threshold * 1.5:
            return "high"
        if drift_score > threshold:
            return "medium"
        return "low"

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="auto retraining trigger",
    )
    async def _trigger_auto_retraining(
        self,
        model_id: str,
        alert: ModelDriftAlert,
    ) -> None:
        """Trigger automatic model retraining."""
        try:
            self.logger.info(f"Triggering auto-retraining for model: {model_id}")

            # This would integrate with the training pipeline
            # For now, just log the action

        except Exception:
            self.logger.exception(error("Error triggering auto-retraining: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="performance snapshot capture",
    )
    async def _capture_performance_snapshots(self) -> None:
        """Capture performance snapshots for all models."""
        try:
            # This would integrate with the existing model monitoring
            # For now, create sample performance data
            for model_id in ["ensemble_1", "ensemble_2", "meta_learner"]:
                performance = ModelPerformance(
                    model_id=model_id,
                    model_type="ensemble",
                    timestamp=datetime.now(),
                    accuracy=np.random.uniform(0.7, 0.9),
                    precision=np.random.uniform(0.6, 0.8),
                    recall=np.random.uniform(0.6, 0.8),
                    f1_score=np.random.uniform(0.6, 0.8),
                    prediction_confidence=np.random.uniform(0.5, 0.9),
                )

                if model_id not in self.model_performance_history:
                    self.model_performance_history[model_id] = []

                self.model_performance_history[model_id].append(performance)

                # Limit history size
                if len(self.model_performance_history[model_id]) > 1000:
                    self.model_performance_history[model_id] = (
                        self.model_performance_history[model_id][-1000:]
                    )

        except Exception:
            self.logger.exception(error("Error capturing performance snapshots: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="feature importance analysis",
    )
    async def _analyze_feature_importance(self) -> None:
        """Analyze feature importance for all models."""
        try:
            for model_id in ["ensemble_1", "ensemble_2", "meta_learner"]:
                # This would integrate with actual model feature importance
                # For now, create sample feature importance data
                features = ["feature_1", "feature_2", "feature_3", "feature_4"]

                for i, feature_name in enumerate(features):
                    importance = FeatureImportance(
                        feature_name=feature_name,
                        importance_score=np.random.uniform(0.1, 0.9),
                        importance_rank=i + 1,
                        stability_score=np.random.uniform(0.7, 1.0),
                        drift_score=np.random.uniform(0.0, 0.3),
                        timestamp=datetime.now(),
                    )

                    if model_id not in self.feature_importance_history:
                        self.feature_importance_history[model_id] = []

                    self.feature_importance_history[model_id].append(importance)

                    # Limit history size
                    if len(self.feature_importance_history[model_id]) > 1000:
                        self.feature_importance_history[model_id] = (
                            self.feature_importance_history[model_id][-1000:]
                        )

        except Exception:
            self.logger.exception(error("Error analyzing feature importance: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="online learning monitoring",
    )
    async def _monitor_online_learning(self) -> None:
        """Monitor online learning metrics."""
        try:
            for model_id in ["ensemble_1", "ensemble_2", "meta_learner"]:
                # This would integrate with actual online learning algorithms
                # For now, create sample online learning metrics
                metrics = OnlineLearningMetrics(
                    model_id=model_id,
                    learning_rate=np.random.uniform(0.01, 0.1),
                    adaptation_speed=np.random.uniform(0.5, 1.0),
                    knowledge_retention=np.random.uniform(0.8, 1.0),
                    forgetting_curve=[np.random.uniform(0.7, 1.0) for _ in range(10)],
                    performance_trend=[np.random.uniform(0.7, 0.9) for _ in range(10)],
                    last_update=datetime.now(),
                )

                self.online_learning_metrics[model_id] = metrics

        except Exception:
            self.logger.exception(error("Error monitoring online learning: {e}"))

    def get_drift_alerts(self, severity: str | None = None) -> list[ModelDriftAlert]:
        """Get drift alerts, optionally filtered by severity."""
        if severity:
            return [alert for alert in self.drift_alerts if alert.severity == severity]
        return self.drift_alerts

    def get_model_performance_history(
        self,
        model_id: str,
        limit: int | None = None,
    ) -> list[ModelPerformance]:
        """Get performance history for a model."""
        history = self.model_performance_history.get(model_id, [])
        if limit:
            return history[-limit:]
        return history

    def get_feature_importance_history(
        self,
        model_id: str,
        limit: int | None = None,
    ) -> list[FeatureImportance]:
        """Get feature importance history for a model."""
        history = self.feature_importance_history.get(model_id, [])
        if limit:
            return history[-limit:]
        return history

    def get_online_learning_metrics(
        self,
        model_id: str,
    ) -> OnlineLearningMetrics | None:
        """Get online learning metrics for a model."""
        return self.online_learning_metrics.get(model_id)

    def get_ml_monitoring_summary(self) -> dict[str, Any]:
        """Get ML monitoring summary."""
        try:
            total_models = len(self.model_performance_history)
            total_alerts = len(self.drift_alerts)
            critical_alerts = len(
                [a for a in self.drift_alerts if a.severity == "critical"],
            )

            # Calculate average performance
            avg_performance = {}
            for model_id, history in self.model_performance_history.items():
                if history:
                    recent_performance = history[-1]
                    avg_performance[model_id] = {
                        "accuracy": recent_performance.accuracy,
                        "precision": recent_performance.precision,
                        "recall": recent_performance.recall,
                        "f1_score": recent_performance.f1_score,
                    }

            return {
                "total_models": total_models,
                "total_alerts": total_alerts,
                "critical_alerts": critical_alerts,
                "average_performance": avg_performance,
                "online_learning_enabled": self.enable_online_learning,
                "drift_detection_enabled": self.drift_detection_enabled,
                "auto_retraining_enabled": self.auto_retraining_enabled,
            }

        except Exception:
            self.logger.exception(error("Error getting ML monitoring summary: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="ML monitor stop",
    )
    async def stop_monitoring(self) -> None:
        """Stop ML monitoring."""
        try:
            self.is_monitoring = False

            # Cancel all monitoring tasks
            for task in self.monitoring_tasks:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            self.monitoring_tasks.clear()

            self.logger.info("🛑 ML Monitor stopped")

        except Exception:
            self.logger.exception(error("Error stopping ML monitoring: {e}"))


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="ML monitor setup",
)
async def setup_ml_monitor(config: dict[str, Any]) -> MLMonitor | None:
    """
    Setup and initialize ML monitor.

    Args:
        config: Configuration dictionary

    Returns:
        MLMonitor instance or None if setup failed
    """
    try:
        ml_monitor = MLMonitor(config)

        if await ml_monitor.initialize():
            return ml_monitor
        return None

    except Exception:
        system_logger.exception(error("Error setting up ML monitor: {e}"))
        return None
