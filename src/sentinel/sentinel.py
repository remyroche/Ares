from collections.abc import Callable
from datetime import datetime
import asyncio
from typing import Any

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.trading_decorators import performance_monitor
from src.utils.warning_symbols import (
    error,
    failed,
    initialization_error,
    invalid,
    missing,
    warning,
)


class Sentinel:
    """Enhanced sentinel with monitoring and alerting helpers."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize sentinel with configuration."""
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("Sentinel")

        # Monitoring state
        self.is_monitoring: bool = False
        self.alerts: list[dict[str, Any]] = []
        self.monitoring_rules: dict[str, dict[str, Any]] = {}
        self.alert_callbacks: list[Callable[[dict[str, Any]], Any]] = []

        # Configuration
        self.sentinel_config: dict[str, Any] = self.config.get("sentinel", {})
        self.monitoring_interval: int = int(
            self.sentinel_config.get("monitoring_interval", 60),
        )
        self.alert_threshold: float = float(
            self.sentinel_config.get("alert_threshold", 0.8),
        )
        self.max_alerts: int = int(self.sentinel_config.get("max_alerts", 100))

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid sentinel configuration"),
            AttributeError: (False, "Missing required sentinel parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False, context="sentinel initialization",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None, context="sentinel configuration loading",
    )
    async def _load_sentinel_configuration(self) -> None:
        """Load and normalize sentinel configuration values."""
        self.sentinel_config.setdefault("monitoring_interval", 60)
        self.sentinel_config.setdefault("alert_threshold", 0.8)
        self.sentinel_config.setdefault("max_alerts", 100)
        self.sentinel_config.setdefault("enable_performance_monitoring", True)
        self.sentinel_config.setdefault("enable_error_monitoring", True)
        self.sentinel_config.setdefault("enable_system_monitoring", True)

        self.monitoring_interval = int(self.sentinel_config["monitoring_interval"])
        self.alert_threshold = float(self.sentinel_config["alert_threshold"])
        self.max_alerts = int(self.sentinel_config["max_alerts"])

        self.logger.info("Sentinel configuration loaded successfully")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False, context="configuration validation",
    )

    def _validate_configuration(self) -> bool:
        """Validate sentinel configuration values."""
        if self.monitoring_interval <= 0:
            self.logger.error(invalid("Invalid monitoring interval"))
            return False

        if not (0 <= self.alert_threshold <= 1):
            self.logger.error(invalid("Invalid alert threshold"))
            return False

        if self.max_alerts <= 0:
            self.logger.error(invalid("Invalid max alerts"))
            return False

        self.logger.info("Configuration validation successful")
        return True

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None, context="monitoring rules initialization",
    )
    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid monitoring parameters"),
            AttributeError: (False, "Missing monitoring components"),
            KeyError: (False, "Missing required monitoring data"),
        },
        default_return=False, context="monitoring start",
    )
    async def start_monitoring(self) -> bool:
        """Start the monitoring loop in the background."""
        if self.is_monitoring:
            self.logger.warning(warning("Monitoring already active"))
            return True

        self.is_monitoring = True
        self.logger.info("🔄 Starting Sentinel monitoring...")
        asyncio.create_task(self._monitoring_loop())
        self.logger.info("✅ Sentinel monitoring started successfully")
        return True

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="monitoring loop",
    )
    @performance_monitor
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            await self._perform_monitoring_checks()
            await asyncio.sleep(self.monitoring_interval)

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="monitoring checks",
    )
    @performance_monitor
    async def _perform_monitoring_checks(self) -> None:
        """Perform all monitoring checks configured."""
        if "performance" in self.monitoring_rules:
            await self._check_performance_metrics()

        if "errors" in self.monitoring_rules:
            await self._check_error_metrics()

        if "system" in self.monitoring_rules:
            await self._check_system_metrics()

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="performance monitoring",
    )
    @performance_monitor
    async def _check_performance_metrics(self) -> None:
        """Check performance metrics (simulated)."""
        cpu_usage = 0.6
        memory_usage = 0.7
        disk_usage = 0.8
        response_time = 500  # ms

        rules = self.monitoring_rules["performance"]

        if cpu_usage > rules["cpu_threshold"]:
            await self._create_alert("PERFORMANCE", "High CPU usage", cpu_usage)

        if memory_usage > rules["memory_threshold"]:
            await self._create_alert("PERFORMANCE", "High memory usage", memory_usage)

        if disk_usage > rules["disk_threshold"]:
            await self._create_alert("PERFORMANCE", "High disk usage", disk_usage)

        if response_time > rules["response_time_threshold"]:
            await self._create_alert(
                "PERFORMANCE",
                "High response time",
                float(response_time) / float(rules["response_time_threshold"]),
            )

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="error monitoring",
    )
    @performance_monitor
    async def _check_error_metrics(self) -> None:
        """Check error metrics (simulated)."""
        error_rate = 0.05
        consecutive_errors = 2
        critical_errors = 0

        rules = self.monitoring_rules["errors"]

        if error_rate > rules["error_rate_threshold"]:
            await self._create_alert("ERROR", "High error rate", error_rate)

        if consecutive_errors > rules["consecutive_errors_threshold"]:
            await self._create_alert(
                "ERROR",
                "High consecutive errors",
                float(consecutive_errors) / float(rules["consecutive_errors_threshold"]),
            )

        if critical_errors > rules["critical_error_threshold"]:
            await self._create_alert(
                "ERROR",
                "Critical errors detected",
                float(critical_errors),
            )

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="system monitoring",
    )
    @performance_monitor
    async def _check_system_metrics(self) -> None:
        """Check system metrics (simulated)."""
        uptime = 0.995
        connection_success_rate = 0.98
        data_quality = 0.95

        rules = self.monitoring_rules["system"]

        if uptime < rules["uptime_threshold"]:
            await self._create_alert("SYSTEM", "Low uptime", 1.0 - uptime)

        if connection_success_rate < rules["connection_threshold"]:
            await self._create_alert(
                "SYSTEM",
                "Low connection success rate",
                1.0 - connection_success_rate,
            )

        if data_quality < rules["data_quality_threshold"]:
            await self._create_alert("SYSTEM", "Low data quality", 1.0 - data_quality)

    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid alert parameters"),
            AttributeError: (None, "Missing alert components"),
            KeyError: (None, "Missing required alert data"),
        },
        default_return=None,
        context="alert creation",
    )
    async def _create_alert(self, alert_type: str, message: str, value: float) -> None:
        """
        Create an alert.

        Args:
            alert_type: Type of alert
            message: Alert message
            value: Alert value
        """
        # Below per-alert threshold, no alert needed
        if value < self.alert_threshold:
            return

        severity = (
            "HIGH" if value > 0.9 else "MEDIUM" if value > 0.8 else "LOW"
        )

        alert = {
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "message": message,
            "value": float(value),
            "severity": severity,
        }

        self.alerts.append(alert)
        if len(self.alerts) > self.max_alerts:
            self.alerts.pop(0)

        self.logger.warning(
            f"🚨 ALERT [{alert_type}]: {message} (Value: {value:.3f})",
        )

        await self._execute_alert_callbacks(alert)

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="alert callbacks execution",
    )
    async def _execute_alert_callbacks(self, alert: dict[str, Any]) -> None:
        """
        Execute alert callbacks.

        Args:
            alert: Alert information
        """
        if not self.alert_callbacks:
            return

        self.logger.info(
            f"Executing {len(self.alert_callbacks)} alert callbacks...",
        )

        for i, callback in enumerate(self.alert_callbacks, start=1):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
                self.logger.debug(f"Alert callback {i} executed successfully")
            except Exception as e:
                self.logger.warning(failed(f"Alert callback {i} failed: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="alert callback registration",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="alert callback removal",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=[],
        context="alerts getting",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="alerts clearing",
    )
    def clear_alerts(self) -> None:
        """Clear all alerts."""
        alert_count = len(self.alerts)
        self.alerts.clear()
        self.logger.info(f"Cleared {alert_count} alerts")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="sentinel cleanup",
    )
# Global sentinel instance
sentinel: Sentinel | None = None

@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="sentinel setup",
)