"""Performance Monitor Module."

This module provides comprehensive performance monitoring for trading models,
including real-time tracking, drift detection, statistical analysis, and
performance metrics calculation. It integrates with the model behavior tracker
to provide holistic performance insights.
"""

import asyncio
import json
from datetime import datetime
from typing import Any

import numpy as np
import yaml
from scipy import stats

from src.core.decorators import handles_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
import copy
import os
    copy,
    error,
    failed,
    import,
    invalid,
    os.path,
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

    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid performance monitor configuration"),
            AttributeError: (False, "Missing required performance monitor parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="performance monitor initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the performance monitor with configuration validation."""
        try:
            self.logger.info("Initializing Performance Monitor...")
            await self._load_monitor_configuration()
            if not self._validate_configuration():
                self.print(invalid("Invalid configuration for performance monitor"))
                return False
            self.logger.info(
                "✅ Performance Monitor initialization completed successfully",
            )
            return True
        except (ValueError, AttributeError, KeyError) as e:
            self.print(failed(f"❌ Performance Monitor initialization failed: {type(e).__name__}: {e}"))
            return False
        except Exception as e:
            self.logger.exception("Unexpected error during initialization")
            self.print(failed(f"❌ Performance Monitor initialization failed with unexpected error: {e}"))
            return False

    @handles_errors(fallback=None)
    async def _load_monitor_configuration(self) -> None:
        try:
            self.monitor_config.setdefault("monitor_interval", 30)
            self.monitor_config.setdefault("max_history", 100)
            self.monitor_interval = self.monitor_config["monitor_interval"]
            self.max_history = self.monitor_config["max_history"]
            self.logger.info("Performance monitor configuration loaded successfully")
        except FileNotFoundError:
            self.logger.warning("Monitor configuration file not found, using defaults")
            self.monitor_config = {"monitor_interval": 60, "max_history": 100}
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            self.print(error(f"Error parsing monitor configuration: {e}"))
            self.monitor_config = {"monitor_interval": 60, "max_history": 100}
        except Exception as e:
            self.print(error(f"Unexpected error loading monitor configuration: {e}"))
            self.monitor_config = {"monitor_interval": 60, "max_history": 100}

    @handles_errors(fallback=False)
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

    @handles_errors(
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

    @handles_errors(fallback=None)
    async def _perform_monitoring(self) -> None:
        try:
            now = datetime.now().isoformat()
            self.status = {"timestamp": now, "status": "running"}
            self.history.append(self.status.copy())
            if len(self.history) > self.max_history:
                self.history.pop(0)
            await self._collect_performance_metrics()
            await self._check_performance_alerts()
            self.logger.info(f"Performance monitoring tick at {now}")
        except Exception:
            self.print(error("Error in performance monitoring step: {e}"))

    @handles_errors(fallback=None)
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

    @handles_errors(fallback=None)
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

    @handles_errors(fallback=None)
    async def stop(self) -> None:
        self.logger.info("🛑 Stopping Performance Monitor...")
        try:
            self.is_running = False
            self.status = {"timestamp": datetime.now().isoformat(), "status": "stopped"}
            self.logger.info("✅ Performance Monitor stopped successfully")
        except Exception:
            self.print(error("Error stopping performance monitor: {e}"))

    def get_status(self) -> dict[str, Any]:
        return self.status.copy()

    def get_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        history = self.history.copy()
        if limit:
            history = history[-limit:]
        return history

    def get_performance_metrics(self) -> dict[str, Any]:
        return self.performance_metrics.copy()

    def get_alerts(self) -> list[dict[str, Any]]:
        return self.alerts.copy()

    def get_drift_alerts(self) -> list[dict[str, Any]]:
        """Get concept drift alerts."""
        return self.drift_alerts.copy()

    def detect_concept_drift(self, model_name: str, current_performance: float) -> bool:
        """
        Detect concept drift for a specific model.

        Args:
            model_name: Name of the model to monitor
            current_performance: Current performance metric (e.g., accuracy = F1-score)

        Returns:
            bool: True if concept drift is detected
        """
        try:
            if model_name not in self.model_performance_history:
                self.model_performance_history[model_name] = []

            # Add current performance to history
            self.model_performance_history[model_name].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "performance": current_performance,
                },
            )

            # Keep only recent history
            if len(self.model_performance_history[model_name]) > self.drift_detection_window:
                self.model_performance_history[model_name] = self.model_performance_history[model_name][
                    -self.drift_detection_window:
                ]

            # Need enough data to detect drift
            if len(self.model_performance_history[model_name]) < 20:
                return False

            # Calculate performance statistics
            performances = [entry["performance"] for entry in self.model_performance_history[model_name]]
            recent_performances = performances[-10:]  # Last 10 predictions
            historical_performances = performances[:-10]  # Earlier predictions

            if len(historical_performances) < 10:
                return False

            # Calculate drift metrics
            recent_mean = np.mean(recent_performances)
            historical_mean = np.mean(historical_performances)
            recent_std = np.std(recent_performances)
            historical_std = np.std(historical_performances)

            # Drift detection using multiple methods
            drift_detected = False
            drift_reasons = []

            # Method 1: Mean shift detection
            mean_shift = abs(recent_mean - historical_mean)
            if mean_shift > self.drift_threshold:
                drift_detected = True
                drift_reasons.append(f"Mean shift: {mean_shift:.4f}")

            # Method 2: Variance shift detection
            variance_shift = abs(recent_std - historical_std)
            if variance_shift > self.drift_threshold:
                drift_detected = True
                drift_reasons.append(f"Variance shift: {variance_shift:.4f}")

            # Method 3: Performance degradation
            if recent_mean < historical_mean - self.drift_threshold:
                drift_detected = True
                drift_reasons.append(
                    f"Performance degradation: {recent_mean:.4f} vs {historical_mean:.4f}",
                )

            # Method 4: Kolmogorov-Smirnov test for distribution shift
            try:
                if len(recent_performances) >= 5 and len(historical_performances) >= 5:
                    ks_statistic, p_value = stats.ks_2samp(
                        recent_performances,
                        historical_performances,
                    )
                    if p_value < 0.05:  # Significant difference
                        drift_detected = True
                        drift_reasons.append(
                            f"Distribution shift (KS p-value: {p_value:.4f})",
                        )
            except ImportError:
                self.print(warning("scipy not available for KS test"))

            # Create drift alert if detected
            if drift_detected:
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "model_name": model_name,
                    "type": "concept_drift",
                    "severity": ("high" if mean_shift > self.drift_threshold * 2 else "medium"),
                    "message": f"Concept drift detected for {model_name}: {'; '.join(drift_reasons)}",
                    "metrics": {
                        "recent_mean": recent_mean,
                        "historical_mean": historical_mean,
                        "mean_shift": mean_shift,
                        "variance_shift": variance_shift,
                        "recent_std": recent_std,
                        "historical_std": historical_std,
                    },
                }
                self.drift_alerts.append(alert)
                self.logger.warning(
                    f"Concept drift detected for {model_name}: {drift_reasons}",
                )

            return drift_detected

        except Exception as e:
            self.logger.exception(
                f"Error detecting concept drift for {model_name}: {e}",
            )
            return False

    def get_model_performance_history(self, model_name: str) -> list[dict[str, Any]]:
        """Get performance history for a specific model."""
        return self.model_performance_history.get(model_name=[]).copy()

    def clear_drift_alerts(self) -> None:
        """Clear concept drift alerts."""
        self.drift_alerts.clear()

    # ============================================================================
    # REAL-TIME PERFORMANCE TRACKING METHODS
    # ============================================================================

    @handles_errors(fallback=None)
    async def update_model_performance(
        self,
        model_name: str,
        prediction: float,
        actual_outcome: float,
        timestamp: datetime = None,
    ) -> None:
        """Update real-time performance tracking for a model."""
        try:
            if not self.enable_real_time_tracking:
                return

            # Initialize tracking for new models
            if model_name not in self.model_predictions:
                self.model_predictions[model_name] = []
                self.model_outcomes[model_name] = []
                self.model_metrics[model_name] = {}

            # Store prediction and outcome
            self.model_predictions[model_name].append(prediction)
            self.model_outcomes[model_name].append(actual_outcome)

            # Maintain performance window
            if len(self.model_predictions[model_name]) > self.performance_window:
                self.model_predictions[model_name] = self.model_predictions[model_name][-self.performance_window:]
                self.model_outcomes[model_name] = self.model_outcomes[model_name][-self.performance_window:]

            # Calculate real-time metrics
            await self._calculate_real_time_metrics(model_name)

            # Check for retraining triggers
            await self._check_retraining_triggers(model_name)

            self.logger.info(f"Updated performance for {model_name}")

        except Exception as e:
            self.logger.exception(f"Error updating model performance: {e}")

    @handles_errors(fallback=None)
    async def _calculate_real_time_metrics(self, model_name: str) -> None:
        """Calculate real-time performance metrics for a model."""
        try:
            predictions = self.model_predictions[model_name]
            outcomes = self.model_outcomes[model_name]

            if len(predictions) < 5:  # Need minimum data points
                return

            # Calculate accuracy (for binary outcomes)
            correct_predictions = sum(1 for p, o in zip(predictions, outcomes) if abs(p - o) < 0.1)
            accuracy = correct_predictions / len(predictions)

            # Calculate mean absolute error
            mae = sum(abs(p - o) for p, o in zip(predictions, outcomes)) / len(predictions)

            # Calculate precision and recall (for binary classification)
            true_positives = sum(1 for p, o in zip(predictions, outcomes) if p > 0.5 and o > 0.5)
            false_positives = sum(1 for p, o in zip(predictions, outcomes) if p > 0.5 and o <= 0.5)
            false_negatives = sum(1 for p, o in zip(predictions, outcomes) if p <= 0.5 and o > 0.5)

            precision = (
                true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            )
            recall = (
                true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            )
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            # Calculate trend (performance over time)
            recent_accuracy = accuracy
            if len(predictions) >= 20:
                recent_predictions = predictions[-10:]
                recent_outcomes = outcomes[-10:]
                recent_correct = sum(1 for p, o in zip(recent_predictions, recent_outcomes) if abs(p - o) < 0.1)
                recent_accuracy = recent_correct / len(recent_predictions)

            # Store metrics
            self.model_metrics[model_name] = {
                "accuracy": accuracy,
                "mae": mae,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "recent_accuracy": recent_accuracy,
                "prediction_count": len(predictions),
                "last_updated": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.exception(f"Error calculating real-time metrics: {e}")

    @handles_errors(fallback=None)
    async def _check_retraining_triggers(self, model_name: str) -> None:
        """Check if model retraining is needed."""
        try:
            metrics = self.model_metrics.get(model_name={})
            if not metrics:
                return

            triggers = []

            # Check for performance degradation
            if metrics.get("accuracy", 1.0) < 0.5:  # Below 50% accuracy
                triggers.append(
                    {
                        "model": model_name,
                        "reason": "performance_degradation",
                        "severity": "high",
                        "metric": "accuracy",
                        "value": metrics["accuracy"],
                        "threshold": 0.5,
                    }
                )

            # Check for concept drift
            if await self._detect_concept_drift(model_name):
                triggers.append(
                    {
                        "model": model_name,
                        "reason": "concept_drift",
                        "severity": "high",
                        "metric": "drift_detected",
                        "value": True,
                    }
                )

            # Check for F1 score degradation
            if metrics.get("f1_score", 1.0) < 0.4:  # Below 40% F1
                triggers.append(
                    {
                        "model": model_name,
                        "reason": "f1_degradation",
                        "severity": "medium",
                        "metric": "f1_score",
                        "value": metrics["f1_score"],
                        "threshold": 0.4,
                    }
                )

            # Check for recent accuracy drop
            if metrics.get("recent_accuracy", 1.0) < metrics.get("accuracy", 1.0) - self.retraining_threshold:
                triggers.append(
                    {
                        "model": model_name,
                        "reason": "recent_performance_drop",
                        "severity": "medium",
                        "metric": "recent_accuracy",
                        "value": metrics["recent_accuracy"],
                        "baseline": metrics["accuracy"],
                    }
                )

            # Add triggers to the list
            for trigger in triggers:
                trigger["timestamp"] = datetime.now().isoformat()
                self.retraining_triggers.append(trigger)

            if triggers:
                self.logger.warning(f"Retraining triggers detected for {model_name}: {triggers}")

        except Exception as e:
            self.logger.exception(f"Error checking retraining triggers: {e}")

    @handles_errors(fallback=None)
    async def select_best_models(
        self,
        model_names: list[str],
        current_regime: str = None,
        required_count: int = 3,
    ) -> list[str]:
        """Select best performing models based on real-time metrics."""
        try:
            if not self.enable_real_time_tracking:
                return model_names[:required_count]

            # Calculate performance scores for each model
            model_scores = {}
            for model_name in model_names:
                metrics = self.model_metrics.get(model_name={})
                if not metrics:
                    model_scores[model_name] = 0.5  # Default score
                    continue

                # Calculate composite score
                accuracy_score = metrics.get("accuracy", 0.5)
                f1_score = metrics.get("f1_score", 0.5)
                recent_accuracy = metrics.get("recent_accuracy", 0.5)

                # Weight recent performance more heavily
                composite_score = (
                    0.3 * accuracy_score
                    + 0.3 * f1_score
                    + 0.4 * recent_accuracy  # Higher weight for recent performance
                )

                # Apply regime adjustment if available
                if current_regime:
                    regime_adjustment = self._get_regime_performance_adjustment(model_name=current_regime)
                    composite_score *= regime_adjustment

                model_scores[model_name] = composite_score

            # Sort models by score and return top performers
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            best_models = [model for model, score in sorted_models[:required_count]]

            self.logger.info(f"Selected best models: {best_models} with scores: {model_scores}")

            return best_models

        except Exception as e:
            self.logger.exception(f"Error selecting best models: {e}")
            return model_names[:required_count]

    def _get_regime_performance_adjustment(self, model_name: str, regime: str) -> float:
        """Get regime-specific performance adjustment for a model."""
        try:
            # Define regime-specific performance multipliers
            regime_multipliers = {
                "BULL": {
                    "tcn": 1.1,
                    "transformer": 1.0,
                    "lstm": 0.9,
                    "gru": 0.9,
                    "tabnet": 1.0,
                },
                "BEAR": {
                    "tcn": 0.9,
                    "transformer": 1.1,
                    "lstm": 1.0,
                    "gru": 1.0,
                    "tabnet": 0.9,
                },
                "SIDEWAYS": {
                    "tcn": 1.0,
                    "transformer": 1.0,
                    "lstm": 1.1,
                    "gru": 1.0,
                    "tabnet": 1.0,
                },
                "SR": {
                    "tcn": 1.2,
                    "transformer": 1.0,
                    "lstm": 0.8,
                    "gru": 0.8,
                    "tabnet": 1.1,
                },
                "CANDLE": {
                    "tcn": 0.9,
                    "transformer": 1.2,
                    "lstm": 1.0,
                    "gru": 1.0,
                    "tabnet": 0.9,
                },
            }

            # Extract model type from name (e.g., "tcn", "transformer", etc.)
            model_type = None
            for model_type_name in ["tcn", "transformer", "lstm", "gru", "tabnet"]:
                if model_type_name in model_name.lower():
                    model_type = model_type_name
                    break

            if not model_type:
                return 1.0  # Default multiplier

            regime_multiplier = regime_multipliers.get(regime={}).get(model_type, 1.0)
            return regime_multiplier

        except Exception as e:
            self.logger.exception(f"Error getting regime performance adjustment: {e}")
            return 1.0

    @handles_errors(fallback=None)
    async def get_performance_feedback(self) -> dict[str, Any]:
        """Get comprehensive performance feedback for the system."""
        try:
            feedback = {
                "timestamp": datetime.now().isoformat(),
                "real_time_tracking_enabled": self.enable_real_time_tracking,
                "models_tracked": list(self.model_metrics.keys()),
                "model_performances": self.model_metrics.copy(),
                "retraining_triggers": self.retraining_triggers.copy(),
                "drift_alerts": self.drift_alerts.copy(),
                "system_health": self._calculate_system_health(),
            }

            return feedback

        except Exception as e:
            self.logger.exception(f"Error getting performance feedback: {e}")
            return {}

    def _calculate_system_health(self) -> dict[str, Any]:
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
                "status": status,
                "overall_accuracy": avg_accuracy,
                "overall_f1": avg_f1,
                "overall_recent_accuracy": avg_recent_accuracy,
                "models_count": len(self.model_metrics),
                "retraining_needed": len(self.retraining_triggers) > 0,
                "drift_detected": len(self.drift_alerts) > 0,
            }

        except Exception as e:
            self.logger.exception(f"Error calculating system health: {e}")
            return {"status": "error", "overall_accuracy": 0.0}

    def get_retraining_triggers(self) -> list[dict[str, Any]]:
        """Get current retraining triggers."""
        return self.retraining_triggers.copy()

    def clear_retraining_triggers(self) -> None:
        """Clear retraining triggers."""
        self.retraining_triggers.clear()

performance_monitor: PerformanceMonitor | None = None

@handles_errors(fallback=None)
async def setup_performance_monitor(
    config: dict[str, Any] | None = None,
) -> PerformanceMonitor | None:
    try:
        global performance_monitor
        if config is None:
            config = {
                "performance_monitor": {"monitor_interval": 30, "max_history": 100},
            }
        performance_monitor = PerformanceMonitor(config)
        success = await performance_monitor.initialize()
        if success:
            return performance_monitor
        return None
    except Exception as e:
        print(f"Error setting up performance monitor: {e}")
        return None
