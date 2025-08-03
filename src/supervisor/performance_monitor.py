import asyncio
import datetime
from datetime import datetime
from typing import Any

import numpy as np

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class PerformanceMonitor:
    """
    Enhanced Performance Monitor component with DI, type hints, and robust error handling.
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

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid performance monitor configuration"),
            AttributeError: (False, "Missing required performance monitor parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="performance monitor initialization",
    )
    async def initialize(self) -> bool:
        try:
            self.logger.info("Initializing Performance Monitor...")
            await self._load_monitor_configuration()
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for performance monitor")
                return False
            self.logger.info(
                "✅ Performance Monitor initialization completed successfully",
            )
            return True
        except Exception as e:
            self.logger.error(f"❌ Performance Monitor initialization failed: {e}")
            return False

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
            self.logger.error(f"Error loading monitor configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        try:
            if self.monitor_interval <= 0:
                self.logger.error("Invalid monitor interval")
                return False
            if self.max_history <= 0:
                self.logger.error("Invalid max history")
                return False
            self.logger.info("Configuration validation successful")
            return True
        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
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
        except Exception as e:
            self.logger.error(f"Error in performance monitor run: {e}")
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
            self.status = {"timestamp": now, "status": "running"}
            self.history.append(self.status.copy())
            if len(self.history) > self.max_history:
                self.history.pop(0)
            await self._collect_performance_metrics()
            await self._check_performance_alerts()
            self.logger.info(f"Performance monitoring tick at {now}")
        except Exception as e:
            self.logger.error(f"Error in performance monitoring step: {e}")

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
        except Exception as e:
            self.logger.error(f"Error collecting performance metrics: {e}")

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
                self.logger.warning("Performance alert: Maximum drawdown exceeded")

            if self.performance_metrics.get("sharpe_ratio", 0) < 1.0:
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "sharpe_alert",
                    "message": "Sharpe ratio below threshold",
                }
                self.alerts.append(alert)
                self.logger.warning("Performance alert: Sharpe ratio below threshold")

            self.logger.info("Performance alerts checked successfully")
        except Exception as e:
            self.logger.error(f"Error checking performance alerts: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="performance monitor stop",
    )
    async def stop(self) -> None:
        self.logger.info("🛑 Stopping Performance Monitor...")
        try:
            self.is_running = False
            self.status = {"timestamp": datetime.now().isoformat(), "status": "stopped"}
            self.logger.info("✅ Performance Monitor stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping performance monitor: {e}")

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
            current_performance: Current performance metric (e.g., accuracy, F1-score)

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
            if (
                len(self.model_performance_history[model_name])
                > self.drift_detection_window
            ):
                self.model_performance_history[model_name] = (
                    self.model_performance_history[
                        model_name
                    ][-self.drift_detection_window :]
                )

            # Need enough data to detect drift
            if len(self.model_performance_history[model_name]) < 20:
                return False

            # Calculate performance statistics
            performances = [
                entry["performance"]
                for entry in self.model_performance_history[model_name]
            ]
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
                from scipy import stats

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
                self.logger.warning("scipy not available for KS test")

            # Create drift alert if detected
            if drift_detected:
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "model_name": model_name,
                    "type": "concept_drift",
                    "severity": "high"
                    if mean_shift > self.drift_threshold * 2
                    else "medium",
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
            self.logger.error(f"Error detecting concept drift for {model_name}: {e}")
            return False

    def get_model_performance_history(self, model_name: str) -> list[dict[str, Any]]:
        """Get performance history for a specific model."""
        return self.model_performance_history.get(model_name, []).copy()

    def clear_drift_alerts(self) -> None:
        """Clear concept drift alerts."""
        self.drift_alerts.clear()


performance_monitor: PerformanceMonitor | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="performance monitor setup",
)
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
