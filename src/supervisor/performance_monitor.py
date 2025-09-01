import asyncio
import datetime
import numpy as np
from datetime import datetime
from scipy import stats
from src.utils.logger import system_logger
from typing import Any
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.warning_symbols import (
    error,
    failed,
    invalid,
    warning,
)

class PerformanceMonitor:
    """
    Enhanced Performance Monitor component with DI = type hints, and robust error handling.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("PerformanceMonitor")
        self.is_running: bool = False
        self.status: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.monitor_config: dict[str, Any] = self.config.get("performance_monitor", {})
        self.monitor_interval: int = self.monitor_config.get("monitor_interval", 30)
        self.max_history: int = self.monitor_config.get("max_history", 100)
        self.performance_metrics: dict[str, Any] = {}
        self.alerts: list[dict[str, Any]] = []

        # Concept drift detection
        self.concept_drift_config: dict[str, Any] = self.monitor_config.get(
            "concept_drift",
            {},
        )
        self.drift_detection_window: int = self.concept_drift_config.get(
            "detection_window",
            100,
        )
        self.drift_threshold: float = self.concept_drift_config.get(
            "drift_threshold",
            0.05,
        )
        self.model_performance_history: dict[str, list] = {}
        self.drift_alerts: list[dict[str, Any]] = []

        # Real-time performance tracking
        self.real_time_config: dict[str, Any] = self.monitor_config.get("real_time_tracking", {})
        self.enable_real_time_tracking: bool = self.real_time_config.get("enable_real_time_tracking", True)
        self.performance_window: int = self.real_time_config.get("performance_window", 100)
        self.retraining_threshold: float = self.real_time_config.get("retraining_threshold", 0.1)

        # Performance tracking state
        self.model_predictions: dict[str, list] = {}
        self.model_outcomes: dict[str, list] = {}
        self.model_metrics: dict[str, dict] = {}
        self.retraining_triggers: list[dict[str, Any]] = []

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid performance monitor configuration"),
            AttributeError: (False, "Missing required performance monitor parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="performance monitor initialization",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="monitor configuration loading",
    )
    async def _load_monitor_configuration(self) -> None:
        try:
            self.monitor_config.setdefault("monitor_interval", 30)
            self.monitor_config.setdefault("max_history", 100)
            self.monitor_interval = self.monitor_config["monitor_interval"]
            self.max_history = self.monitor_config["max_history"]
            self.logger.info("Performance monitor configuration loaded successfully")
        except Exception as e:
            self.print(error("Error loading monitor configuration: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )

    def _validate_configuration(self) -> bool:
        try:
            if self.monitor_interval <= 0:
                self.print(invalid("Invalid monitor interval"))
                return False
            if self.max_history <= 0:
                self.print(invalid("Invalid max history"))
                return False
            self.logger.info("Configuration validation successful")
            return True
        except Exception:
            self.print(error("Error validating configuration: {e}"))
            return False

    @handle_specific_errors(
        error_handlers={
            Exception: (False, "Performance monitor run failed"),
        },
        default_return=False,
        context="performance monitor run",
    )
    async def run(self) -> bool:
        try:
            self.is_running = True
            self.logger.info("🚦 Performance Monitor started.")
            while self.is_running:
                await self._perform_monitoring()
                await asyncio.sleep(self.monitor_interval)
            return True
        except Exception:
            self.print(error("Error in performance monitor run: {e}"))
            self.is_running = False
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="performance monitoring step",
    )
    async def _perform_monitoring(self) -> None:
        try:
            now = datetime.now().isoformat()
            self.status = {"timestamp": now , "status": "running"}
            self.history.append(self.status.copy())
            if len(self.history) > self.max_history:
                self.history.pop(0)
            await self._collect_performance_metrics()
            await self._check_performance_alerts()
            self.logger.info(f"Performance monitoring tick at {now}")
        except Exception:
            self.print(error("Error in performance monitoring step: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="performance metrics collection",
    )
    async def _collect_performance_metrics(self) -> None:
        try:
            # Simulate performance metrics collection
            metrics = {
                "total_return": 0.125,
                "sharpe_ratio": 1.85,
                "max_drawdown": -0.08,
                "win_rate": 0.65,
                "profit_factor": 1.45,
            }
            self.performance_metrics.update(metrics)
            self.logger.info("Performance metrics collected successfully")
        except Exception:
            self.print(error("Error collecting performance metrics: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="performance alerts check",
    )
    async def _check_performance_alerts(self) -> None:
        try:
            # Check for performance alerts
            if self.performance_metrics.get("max_drawdown", 0) < -0.1:
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "drawdown_alert",
                    "message": "Maximum drawdown exceeded threshold",
                }
                self.alerts.append(alert)
                self.print(warning("Performance alert: Maximum drawdown exceeded"))

            if self.performance_metrics.get("sharpe_ratio", 0) < 1.0:
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "sharpe_alert",
                    "message": "Sharpe ratio below threshold",
                }
                self.alerts.append(alert)
                self.print(warning("Performance alert: Sharpe ratio below threshold"))

            self.logger.info("Performance alerts checked successfully")
        except Exception:
            self.print(error("Error checking performance alerts: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="performance monitor stop",
    )
    # ============================================================================
    # REAL-TIME PERFORMANCE TRACKING METHODS
    # ============================================================================

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="real-time performance update",
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return=None, context="real-time metrics calculation",
    )
    async def _calculate_real_time_metrics(self, model_name: str) -> None:
        """Calculate real-time performance metrics for a model."""
        try:
            predictions = self.model_predictions[model_name]
            outcomes = self.model_outcomes[model_name]

            if len(predictions) < 5:  # Need minimum data points
                return

            # Calculate accuracy (for binary outcomes)
            correct_predictions = sum(1 for p , o in zip(predictions, outcomes) if abs(p - o) < 0.1)
            accuracy = correct_predictions / len(predictions)

            # Calculate mean absolute error
            mae = sum(abs(p - o) for p , o in zip(predictions, outcomes)) / len(predictions)

            # Calculate precision and recall (for binary classification)
            true_positives = sum(1 for p , o in zip(predictions, outcomes) if p > 0.5 and o > 0.5)
            false_positives = sum(1 for p , o in zip(predictions, outcomes) if p > 0.5 and o <= 0.5)
            false_negatives = sum(1 for p , o in zip(predictions, outcomes) if p <= 0.5 and o > 0.5)

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            # Calculate trend (performance over time)
            recent_accuracy = accuracy
            if len(predictions) >= 20:
                recent_predictions = predictions[-10:]
                recent_outcomes = outcomes[-10:]
                recent_correct = sum(1 for p , o in zip(recent_predictions, recent_outcomes) if abs(p - o) < 0.1)
                recent_accuracy = recent_correct / len(recent_predictions)

            # Store metrics
            self.model_metrics[model_name] = {
                "accuracy": accuracy , "mae": mae,
                "precision": precision , "recall": recall,
                "f1_score": f1_score , "recent_accuracy": recent_accuracy,
                "prediction_count": len(predictions),
                "last_updated": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.exception(f"Error calculating real-time metrics: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None, context="retraining trigger check",
    )
    async def _check_retraining_triggers(self, model_name: str) -> None:
        """Check if model retraining is needed."""
        try:
            metrics = self.model_metrics.get(model_name = {})
            if not metrics:
                return

            triggers = []

            # Check for performance degradation
            if metrics.get("accuracy", 1.0) < 0.5:  # Below 50% accuracy
                triggers.append({
                    "model": model_name,
                    "reason": "performance_degradation",
                    "severity": "high",
                    "metric": "accuracy",
                    "value": metrics["accuracy"],
                    "threshold": 0.5
                })

            # Check for concept drift
            if await self._detect_concept_drift(model_name):
                triggers.append({
                    "model": model_name,
                    "reason": "concept_drift",
                    "severity": "high",
                    "metric": "drift_detected",
                    "value": True
                })

            # Check for F1 score degradation
            if metrics.get("f1_score", 1.0) < 0.4:  # Below 40% F1
                triggers.append({
                    "model": model_name , "reason": "f1_degradation",
                    "severity": "medium",
                    "metric": "f1_score",
                    "value": metrics["f1_score"],
                    "threshold": 0.4
                })

            # Check for recent accuracy drop
            if metrics.get("recent_accuracy", 1.0) < metrics.get("accuracy", 1.0) - self.retraining_threshold:
                triggers.append({
                    "model": model_name,
                    "reason": "recent_performance_drop",
                    "severity": "medium",
                    "metric": "recent_accuracy",
                    "value": metrics["recent_accuracy"],
                    "baseline": metrics["accuracy"]
                })

            # Add triggers to the list
            for trigger in triggers:
                trigger["timestamp"] = datetime.now().isoformat()
                self.retraining_triggers.append(trigger)

            if triggers:
                self.logger.warning(f"Retraining triggers detected for {model_name}: {triggers}")

        except Exception as e:
            self.logger.exception(f"Error checking retraining triggers: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None, context="adaptive model selection",
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="performance feedback loop",
    )
    def _calculate_system_health(self) -> dict[str , Any]:
        """Calculate overall system health metrics."""
        try:
            if not self.model_metrics:
                return {"status": "unknown", "overall_accuracy": 0.0}

            # Calculate average performance across all models
            accuracies = [metrics.get("accuracy", 0.0) for metrics in self.model_metrics.values()]
            f1_scores = [metrics.get("f1_score", 0.0) for metrics in self.model_metrics.values()]
            recent_accuracies = [metrics.get("recent_accuracy", 0.0) for metrics in self.model_metrics.values()]

            avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
            avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
            avg_recent_accuracy = sum(recent_accuracies) / len(recent_accuracies) if recent_accuracies else 0.0

            # Determine system health status
            if avg_accuracy > 0.7 and avg_f1 > 0.6:
                status = "healthy"
            elif avg_accuracy > 0.5 and avg_f1 > 0.4:
                status = "warning"
            else:
                status = "critical"

            return {
                "status": status , "overall_accuracy": avg_accuracy,
                "overall_f1": avg_f1 , "overall_recent_accuracy": avg_recent_accuracy,
                "models_count": len(self.model_metrics),
                "retraining_needed": len(self.retraining_triggers) > 0,
                "drift_detected": len(self.drift_alerts) > 0
            }

        except Exception as e:
            self.logger.exception(f"Error calculating system health: {e}")
            return {"status": "error", "overall_accuracy": 0.0}

    def get_retraining_triggers(self) -> list[dict[str , Any]]:
        """Get current retraining triggers."""
        return self.retraining_triggers.copy()

    def clear_retraining_triggers(self) -> None:
        """Clear retraining triggers."""
        self.retraining_triggers.clear()

performance_monitor: PerformanceMonitor | None = None

@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="performance monitor setup",
)