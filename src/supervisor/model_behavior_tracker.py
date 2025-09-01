#!/usr/bin/env python3
"""
Model Behavior Tracker

This module enhances the existing performance monitoring system with comprehensive
model behavior tracking, feature importance monitoring, and decision path analysis.
"""

import json
from datetime import datetime
from src.utils.logger import system_logger
from typing import Any
import asyncio

from dataclasses import asdict, dataclass
from enum import Enum
from src.supervisor.performance_monitor import PerformanceMonitor
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.warning_symbols import error, failed, initialization_error
import numpy as np

class BehaviorMetricType(Enum):
    """Model behavior metric types."""

    PREDICTION_CONSISTENCY = "prediction_consistency"
    CONFIDENCE_TREND = "confidence_trend"
    FEATURE_IMPORTANCE_STABILITY = "feature_importance_stability"
    PREDICTION_DRIFT = "prediction_drift"
    ENSEMBLE_DIVERSITY = "ensemble_diversity"
    DECISION_PATH_STABILITY = "decision_path_stability"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    THEORY_VS_REALITY = "theory_vs_reality"

@dataclass
class ModelBehaviorSnapshot:
    """Model behavior snapshot."""

    model_id: str
    model_type: str
    timestamp: datetime
    prediction_consistency: float
    confidence_trend: list[float]
    feature_importance_stability: float
    prediction_drift: float
    ensemble_diversity: float | None = None
    decision_path_stability: float | None = None
    confidence_calibration: float | None = None
    theory_vs_reality_score: float | None = None
    metadata: dict[str, Any] = None

@dataclass
class FeatureImportanceTracking:
    """Feature importance tracking data."""

    feature_name: str
    model_id: str
    timestamp: datetime
    importance_score: float
    importance_rank: int
    stability_score: float
    drift_score: float

@dataclass

class DecisionPathAnalysis:
    """Decision path analysis data."""

    model_id: str
    timestamp: datetime
    decision_steps: list[str]
    decision_weights: list[float]
    path_stability: float
    path_complexity: float
    confidence_distribution: list[float]

class ModelBehaviorTracker:
    """
    Enhanced model behavior tracker that integrates with existing performance monitoring.
    """

    def __init__(self, config: dict[str, Any], performance_monitor: PerformanceMonitor):
        """
        Initialize model behavior tracker.

        Args:
            config: Configuration dictionary
            performance_monitor: Existing performance monitor instance
        """
        self.config = config
        self.performance_monitor = performance_monitor
        self.logger = system_logger.getChild("ModelBehaviorTracker")

        # Configuration
        self.tracker_config = config.get("model_behavior_tracker", {})
        self.tracking_interval = self.tracker_config.get(
            "tracking_interval",
            60,
        )  # 1 minute
        self.max_history_size = self.tracker_config.get("max_history_size", 1000)

        # Storage
        self.behavior_history: dict[str , list[ModelBehaviorSnapshot]] = {}
        self.feature_importance_history: dict[str , list[FeatureImportanceTracking]] = {}
        self.decision_path_history: dict[str , list[DecisionPathAnalysis]] = {}

        # Tracking state
        self.is_tracking = False
        self.tracking_task: asyncio.Task | None = None

        # Reference data for stability calculations
        self.reference_behavior: dict[str , dict[str, float]] = {}

        self.logger.info("🚀 Model Behavior Tracker initialized")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid tracker configuration"),
            AttributeError: (False, "Missing required tracker parameters"),
        },
        default_return=False,
        context="behavior tracker initialization",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="reference behavior loading",
    )
    async def _load_reference_behavior(self) -> None:
        """Load reference behavior data for stability calculations."""
        try:
            # Load reference behavior metrics from training data
            self.reference_behavior = {
                "prediction_consistency": 0.85,
                "confidence_trend_stability": 0.78,
                "feature_importance_stability": 0.82,
                "prediction_drift_threshold": 0.05,
                "ensemble_diversity_target": 0.65,
                "decision_path_stability": 0.80,
            }

            self.logger.info("📊 Reference behavior data loaded")

        except Exception:
            self.logger.exception(error("Error loading reference behavior: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="behavior tracking initialization",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="feature tracking initialization",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="decision path tracking initialization",
    )
    @handle_specific_errors(
        error_handlers={
            Exception: (False, "Behavior tracking failed"),
        },
        default_return=False, context="behavior tracking",
    )
    async def start_tracking(self) -> bool:
        """Start the model behavior tracking."""
        try:
            self.is_tracking = True
            self.logger.info("🚦 Starting Model Behavior Tracker...")

            # Start tracking task
            self.tracking_task = asyncio.create_task(self._behavior_tracking_loop())

            self.logger.info("✅ Model Behavior Tracker started successfully")
            return True

        except Exception:
            self.logger.exception(
                failed("❌ Failed to start Model Behavior Tracker: {e}"),
            )
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="behavior tracking loop",
    )
    async def _behavior_tracking_loop(self) -> None:
        """Continuous behavior tracking loop."""
        while self.is_tracking:
            try:
                await self._capture_behavior_snapshots()
                await asyncio.sleep(self.tracking_interval)
            except Exception:
                self.logger.exception(error("Error in behavior tracking loop: {e}"))
                await asyncio.sleep(60)  # Wait before retrying

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="behavior snapshot capture",
    )
    async def _capture_behavior_snapshots(self) -> None:
        """Capture behavior snapshots for all models."""
        try:
            # Get current performance metrics from performance monitor
            current_metrics = self.performance_monitor.get_performance_metrics()

            for model_id, performance in current_metrics.get("models", {}).items():
                # Calculate behavior metrics
                prediction_consistency = self._calculate_prediction_consistency(
                    model_id,
                    performance,
                )
                confidence_trend = self._calculate_confidence_trend(
                    model_id,
                    performance,
                )
                feature_importance_stability = (
                    self._calculate_feature_importance_stability(model_id, performance)
                )
                prediction_drift = self._calculate_prediction_drift(
                    model_id,
                    performance,
                )
                ensemble_diversity = self._calculate_ensemble_diversity(
                    model_id,
                    performance,
                )
                decision_path_stability = self._calculate_decision_path_stability(
                    model_id,
                    performance,
                )
                confidence_calibration = self._calculate_confidence_calibration(
                    model_id,
                    performance,
                )
                theory_vs_reality_score = self._calculate_theory_vs_reality_score(
                    model_id,
                    performance,
                )

                # Create behavior snapshot
                snapshot = ModelBehaviorSnapshot(
                    model_id=model_id,
                    model_type=performance.get("model_type", "ensemble"),
                    timestamp=datetime.now(),
                    prediction_consistency=prediction_consistency,
                    confidence_trend=confidence_trend,
                    feature_importance_stability=feature_importance_stability,
                    prediction_drift=prediction_drift,
                    ensemble_diversity=ensemble_diversity,
                    decision_path_stability=decision_path_stability,
                    confidence_calibration=confidence_calibration,
                    theory_vs_reality_score=theory_vs_reality_score,
                    metadata=performance.get("metadata", {}),
                )

                if model_id not in self.behavior_history:
                    self.behavior_history[model_id] = []

                self.behavior_history[model_id].append(snapshot)

                # Keep only recent snapshots
                if len(self.behavior_history[model_id]) > self.max_history_size:
                    self.behavior_history[model_id] = self.behavior_history[model_id][
                        -self.max_history_size // 2 :
                    ]

            self.logger.debug("📊 Behavior snapshots captured")

        except Exception:
            self.logger.exception(error("Error capturing behavior snapshots: {e}"))

    def _calculate_prediction_consistency(
        self,
        model_id: str,
        performance: dict[str, Any],
    ) -> float:
        """Calculate prediction consistency."""
        try:
            # This would typically analyze recent predictions vs historical patterns
            # For now, use a simplified approach based on accuracy stability
            accuracy = performance.get("accuracy", 0.0)
            reference_accuracy = self.reference_behavior.get(
                "prediction_consistency",
                0.85,
            )

            # Calculate consistency as how close current accuracy is to reference
            consistency = 1.0 - abs(accuracy - reference_accuracy) / reference_accuracy
            return max(0.0, min(1.0, consistency))

        except Exception:
            self.logger.exception(
                error("Error calculating prediction consistency: {e}"),
            )
            return 0.0

    def _calculate_confidence_trend(
        self,
        model_id: str,
        performance: dict[str, Any],
    ) -> list[float]:
        """Calculate confidence trend."""
        try:
            # This would typically analyze recent confidence scores
            # For now, simulate a trend based on performance metrics
            confidence = performance.get("confidence", 0.0)

            # Simulate trend with some variation
            trend = [confidence + np.random.normal(0, 0.05) for _ in range(10)]
            return [max(0.0, min(1.0, c)) for c in trend]

        except Exception:
            self.logger.exception(error("Error calculating confidence trend: {e}"))
            return [0.0] * 10

    def _calculate_feature_importance_stability(
        self, model_id: str,
        performance: dict[str , Any],
    ) -> float:
        """Calculate feature importance stability."""
        try:
            # This would typically analyze feature importance changes over time
            # For now, use a simplified approach
            feature_stability = performance.get("feature_stability", 0.8)
            reference_stability = self.reference_behavior.get(
                "feature_importance_stability",
                0.82,
            )

            # Calculate stability relative to reference
            stability = (
                1.0 - abs(feature_stability - reference_stability) / reference_stability
            )
            return max(0.0, min(1.0, stability))

        except Exception as e:
            self.logger.exception(
                f"Error calculating feature importance stability: {e}",
            )
            return 0.0

    def _calculate_prediction_drift(
        self, model_id: str,
        performance: dict[str , Any],
    ) -> float:
        """Calculate prediction drift."""
        try:
            # This would typically analyze prediction distribution changes
            # For now, use a simplified approach
            accuracy = performance.get("accuracy", 0.0)
            reference_accuracy = self.reference_behavior.get(
                "prediction_consistency",
                0.85,
            )

            # Calculate drift as performance degradation
            return max(0.0, reference_accuracy - accuracy) / reference_accuracy

        except Exception:
            self.logger.exception(error("Error calculating prediction drift: {e}"))
            return 0.0

    def _calculate_ensemble_diversity(
        self, model_id: str,
        performance: dict[str , Any],
    ) -> float | None:
        """Calculate ensemble diversity."""
        try:
            # This would typically analyze individual model predictions in ensemble
            # For now, use a simplified approach
            if "ensemble" in model_id.lower():
                return performance.get("diversity_score", 0.65)
            return None

        except Exception:
            self.logger.exception(error("Error calculating ensemble diversity: {e}"))
            return None

    def _calculate_decision_path_stability(
        self, model_id: str,
        performance: dict[str , Any],
    ) -> float | None:
        """Calculate decision path stability."""
        try:
            # This would typically analyze decision path consistency
            # For now, use a simplified approach
            path_stability = performance.get("path_stability", 0.8)
            reference_stability = self.reference_behavior.get(
                "decision_path_stability",
                0.80,
            )

            # Calculate stability relative to reference
            stability = (
                1.0 - abs(path_stability - reference_stability) / reference_stability
            )
            return max(0.0, min(1.0, stability))

        except Exception:
            self.logger.exception(
                error("Error calculating decision path stability: {e}"),
            )
            return None

    def _calculate_confidence_calibration(
        self, model_id: str,
        performance: dict[str , Any],
    ) -> float | None:
        """Calculate confidence calibration score for a model."""
        try:
            # Simulate confidence calibration calculation
            # In production, this would compare predicted probabilities with actual outcomes
            return 0.92

        except Exception as e:
            self.logger.exception(
                f"Error calculating confidence calibration for {model_id}: {e}",
            )
            return None

    def _calculate_theory_vs_reality_score(
        self, model_id: str,
        performance: dict[str , Any],
    ) -> float | None:
        """Calculate theory vs reality score for a model."""
        try:
            # Simulate theory vs reality calculation
            # In production = this would compare expected vs actual model behavior
            return 0.88

        except Exception as e:
            self.logger.exception(
                f"Error calculating theory vs reality score for {model_id}: {e}",
            )
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None, context="behavior tracker stop",
    )
    def _calculate_behavior_trend(self, snapshots: list[ModelBehaviorSnapshot]) -> str:
        """Calculate behavior trend."""
        try:
            if len(snapshots) < 2:
                return "insufficient_data"

            # Calculate trend based on prediction consistency
            recent_avg = np.mean([s.prediction_consistency for s in snapshots[-5:]])
            older_avg = (
                np.mean([s.prediction_consistency for s in snapshots[-10:-5]])
                if len(snapshots) >= 10
                else recent_avg
            )

            if recent_avg > older_avg + 0.05:
                return "improving"
            if recent_avg < older_avg - 0.05:
                return "declining"
            return "stable"

        except Exception:
            self.logger.exception(error("Error calculating behavior trend: {e}"))
            return "unknown"

    def _calculate_overall_stability(
        self, snapshots: list[ModelBehaviorSnapshot],
    ) -> float:
        """Calculate overall stability score."""
        try:
            if not snapshots:
                return 0.0

            # Combine multiple stability metrics
            consistency_scores = [s.prediction_consistency for s in snapshots]
            feature_stability_scores = [
                s.feature_importance_stability for s in snapshots
            ]
            drift_scores = [1.0 - s.prediction_drift for s in snapshots]  # Invert drift

            # Calculate weighted average
            weights = [0.4, 0.3, 0.3]  # Weights for each metric
            stability = (
                np.mean(consistency_scores) * weights[0]
                + np.mean(feature_stability_scores) * weights[1]
                + np.mean(drift_scores) * weights[2]
            )

            return max(0.0, min(1.0, stability))

        except Exception:
            self.logger.exception(error("Error calculating overall stability: {e}"))
            return 0.0

    def _determine_alert_level(self, snapshots: list[ModelBehaviorSnapshot]) -> str:
        """Determine alert level based on behavior metrics."""
        try:
            if not snapshots:
                return "unknown"

            # Check various alert conditions
            avg_consistency = np.mean([s.prediction_consistency for s in snapshots])
            avg_drift = np.mean([s.prediction_drift for s in snapshots])
            avg_stability = np.mean([s.feature_importance_stability for s in snapshots])

            # Determine alert level
            if avg_consistency < 0.6 or avg_drift > 0.15 or avg_stability < 0.6:
                return "critical"
            if avg_consistency < 0.75 or avg_drift > 0.10 or avg_stability < 0.75:
                return "warning"
            if avg_consistency < 0.85 or avg_drift > 0.05 or avg_stability < 0.85:
                return "notice"
            return "normal"

        except Exception:
            self.logger.exception(error("Error determining alert level: {e}"))
            return "unknown"

# Factory function for creating model behavior tracker
@handle_errors(
    exceptions=(Exception,),
    default_return=None, context="model behavior tracker setup",
)