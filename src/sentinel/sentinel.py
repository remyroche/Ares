import asyncio
import datetime  # Added import for datetime
from collections.abc import Callable
from datetime import datetime
from typing import (
    Any,
)

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    failed,
    initialization_error,
    invalid,
    missing,
    warning,
)


class Sentinel:
    """
    Enhanced sentinel with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize sentinel with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("Sentinel")

        # Monitoring state
        self.is_monitoring: bool = False
        self.alerts: list[dict[str, Any]] = []
        self.monitoring_rules: dict[str, dict[str, Any]] = {}
        self.alert_callbacks: list[Callable] = []

        # Configuration
        self.sentinel_config: dict[str, Any] = self.config.get("sentinel", {})
        self.monitoring_interval: int = self.sentinel_config.get(
            "monitoring_interval",
            60,
        )
        self.alert_threshold: float = self.sentinel_config.get("alert_threshold", 0.8)
        self.max_alerts: int = self.sentinel_config.get("max_alerts", 100)

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid sentinel configuration"),
            AttributeError: (False, "Missing required sentinel parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="sentinel initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize sentinel with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Sentinel...")

            # Load sentinel configuration
            await self._load_sentinel_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.print(invalid("Invalid configuration for sentinel"))
                return False

            # Initialize monitoring rules
            await self._initialize_monitoring_rules()

            self.logger.info("✅ Sentinel initialization completed successfully")
            return True

        except Exception:
            self.print(failed("❌ Sentinel initialization failed: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="sentinel configuration loading",
    )
    async def _load_sentinel_configuration(self) -> None:
        """Load sentinel configuration."""
        try:
            # Set default sentinel parameters
            self.sentinel_config.setdefault("monitoring_interval", 60)
            self.sentinel_config.setdefault("alert_threshold", 0.8)
            self.sentinel_config.setdefault("max_alerts", 100)
            self.sentinel_config.setdefault("enable_performance_monitoring", True)
            self.sentinel_config.setdefault("enable_error_monitoring", True)
            self.sentinel_config.setdefault("enable_system_monitoring", True)

            # Update configuration
            self.monitoring_interval = self.sentinel_config["monitoring_interval"]
            self.alert_threshold = self.sentinel_config["alert_threshold"]
            self.max_alerts = self.sentinel_config["max_alerts"]

            self.logger.info("Sentinel configuration loaded successfully")

        except Exception:
            self.print(error("Error loading sentinel configuration: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate sentinel configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate monitoring interval
            if self.monitoring_interval <= 0:
                self.print(invalid("Invalid monitoring interval"))
                return False

            # Validate alert threshold
            if self.alert_threshold < 0 or self.alert_threshold > 1:
                self.print(invalid("Invalid alert threshold"))
                return False

            # Validate max alerts
            if self.max_alerts <= 0:
                self.print(invalid("Invalid max alerts"))
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception:
            self.print(error("Error validating configuration: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="monitoring rules initialization",
    )
    async def _initialize_monitoring_rules(self) -> None:
        """Initialize monitoring rules."""
        try:
            # Performance monitoring rules
            if self.sentinel_config.get("enable_performance_monitoring", True):
                self.monitoring_rules["performance"] = {
                    "cpu_threshold": 0.8,
                    "memory_threshold": 0.8,
                    "disk_threshold": 0.9,
                    "response_time_threshold": 1000,  # ms
                }

            # Error monitoring rules
            if self.sentinel_config.get("enable_error_monitoring", True):
                self.monitoring_rules["errors"] = {
                    "error_rate_threshold": 0.1,
                    "consecutive_errors_threshold": 5,
                    "critical_error_threshold": 1,
                }

            # System monitoring rules
            if self.sentinel_config.get("enable_system_monitoring", True):
                self.monitoring_rules["system"] = {
                    "uptime_threshold": 0.99,
                    "connection_threshold": 0.95,
                    "data_quality_threshold": 0.9,
                }

            self.logger.info(
                f"Initialized {len(self.monitoring_rules)} monitoring rule sets",
            )

        except Exception:
            self.print(initialization_error("Error initializing monitoring rules: {e}"))

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid monitoring parameters"),
            AttributeError: (False, "Missing monitoring components"),
            KeyError: (False, "Missing required monitoring data"),
        },
        default_return=False,
        context="monitoring start",
    )
    async def start_monitoring(self) -> bool:
        """
        Start monitoring with enhanced error handling.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.is_monitoring:
                self.print(warning("Monitoring already active"))
                return True

            self.is_monitoring = True
            self.logger.info("🔄 Starting Sentinel monitoring...")

            # Start monitoring loop
            asyncio.create_task(self._monitoring_loop())

            self.logger.info("✅ Sentinel monitoring started successfully")
            return True

        except Exception:
            self.print(error("Error starting monitoring: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="monitoring loop",
    )
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        try:
            while self.is_monitoring:
                # Perform monitoring checks
                await self._perform_monitoring_checks()

                # Wait for next interval
                await asyncio.sleep(self.monitoring_interval)

        except Exception:
            self.print(error("Error in monitoring loop: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="monitoring checks",
    )
    async def _perform_monitoring_checks(self) -> None:
        """Perform all monitoring checks."""
        try:
            # Performance monitoring
            if "performance" in self.monitoring_rules:
                await self._check_performance_metrics()

            # Error monitoring
            if "errors" in self.monitoring_rules:
                await self._check_error_metrics()

            # System monitoring
            if "system" in self.monitoring_rules:
                await self._check_system_metrics()

        except Exception:
            self.print(error("Error performing monitoring checks: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="performance monitoring",
    )
    async def _check_performance_metrics(self) -> None:
        """Check performance metrics."""
        try:
            # Simulate performance metrics collection
            cpu_usage = 0.6  # Simulated CPU usage
            memory_usage = 0.7  # Simulated memory usage
            disk_usage = 0.8  # Simulated disk usage
            response_time = 500  # Simulated response time in ms

            rules = self.monitoring_rules["performance"]

            # Check thresholds
            if cpu_usage > rules["cpu_threshold"]:
                await self._create_alert("PERFORMANCE", "High CPU usage", cpu_usage)

            if memory_usage > rules["memory_threshold"]:
                await self._create_alert(
                    "PERFORMANCE",
                    "High memory usage",
                    memory_usage,
                )

            if disk_usage > rules["disk_threshold"]:
                await self._create_alert("PERFORMANCE", "High disk usage", disk_usage)

            if response_time > rules["response_time_threshold"]:
                await self._create_alert(
                    "PERFORMANCE",
                    "High response time",
                    response_time,
                )

        except Exception:
            self.print(error("Error checking performance metrics: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="error monitoring",
    )
    async def _check_error_metrics(self) -> None:
        """Check error metrics."""
        try:
            # Simulate error metrics collection
            error_rate = 0.05  # Simulated error rate
            consecutive_errors = 2  # Simulated consecutive errors
            critical_errors = 0  # Simulated critical errors

            rules = self.monitoring_rules["errors"]

            # Check thresholds
            if error_rate > rules["error_rate_threshold"]:
                await self._create_alert("ERROR", "High error rate", error_rate)

            if consecutive_errors > rules["consecutive_errors_threshold"]:
                await self._create_alert(
                    "ERROR",
                    "High consecutive errors",
                    consecutive_errors,
                )

            if critical_errors > rules["critical_error_threshold"]:
                await self._create_alert(
                    "ERROR",
                    "Critical errors detected",
                    critical_errors,
                )

        except Exception:
            self.print(error("Error checking error metrics: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="system monitoring",
    )
    async def _check_system_metrics(self) -> None:
        """Check system metrics."""
        try:
            # Simulate system metrics collection
            uptime = 0.995  # Simulated uptime
            connection_success_rate = 0.98  # Simulated connection success rate
            data_quality = 0.95  # Simulated data quality

            rules = self.monitoring_rules["system"]

            # Check thresholds
            if uptime < rules["uptime_threshold"]:
                await self._create_alert("SYSTEM", "Low uptime", uptime)

            if connection_success_rate < rules["connection_threshold"]:
                await self._create_alert(
                    "SYSTEM",
                    "Low connection success rate",
                    connection_success_rate,
                )

            if data_quality < rules["data_quality_threshold"]:
                await self._create_alert("SYSTEM", "Low data quality", data_quality)

        except Exception:
            self.print(error("Error checking system metrics: {e}"))

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
        try:
            # Check if we should create an alert based on threshold
            if value < self.alert_threshold:
                return  # Below threshold, no alert needed

            # Create alert
            alert = {
                "timestamp": datetime.now().isoformat(),
                "type": alert_type,
                "message": message,
                "value": value,
                "severity": "HIGH"
                if value > 0.9
                else "MEDIUM"
                if value > 0.8
                else "LOW",
            }

            # Add to alerts list
            self.alerts.append(alert)

            # Limit alerts
            if len(self.alerts) > self.max_alerts:
                self.alerts.pop(0)  # Remove oldest alert

            # Log alert
            self.logger.warning(
                f"🚨 ALERT [{alert_type}]: {message} (Value: {value:.3f})",
            )

            # Execute alert callbacks
            await self._execute_alert_callbacks(alert)

        except Exception:
            self.print(error("Error creating alert: {e}"))

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
        try:
            if not self.alert_callbacks:
                return

            self.logger.info(
                f"Executing {len(self.alert_callbacks)} alert callbacks...",
            )

            for i, callback in enumerate(self.alert_callbacks):
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                    self.logger.debug(f"Alert callback {i+1} executed successfully")
                except Exception:
                    self.print(failed("Alert callback {i+1} failed: {e}"))

        except Exception:
            self.print(error("Error executing alert callbacks: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="alert callback registration",
    )
    def register_alert_callback(self, callback: Callable) -> None:
        """
        Register an alert callback.

        Args:
            callback: Callback function to execute when alerts are created
        """
        try:
            if callback not in self.alert_callbacks:
                self.alert_callbacks.append(callback)
                self.logger.info("Alert callback registered")
            else:
                self.print(warning("Alert callback already registered"))

        except Exception:
            self.print(error("Error registering alert callback: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="alert callback removal",
    )
    def unregister_alert_callback(self, callback: Callable) -> None:
        """
        Unregister an alert callback.

        Args:
            callback: Callback function to remove
        """
        try:
            if callback in self.alert_callbacks:
                self.alert_callbacks.remove(callback)
                self.logger.info("Alert callback unregistered")
            else:
                self.print(missing("Alert callback not found"))

        except Exception:
            self.print(error("Error unregistering alert callback: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="alerts getting",
    )
    def get_alerts(
        self,
        alert_type: str | None = None,
        severity: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get alerts with optional filtering.

        Args:
            alert_type: Optional alert type filter
            severity: Optional severity filter

        Returns:
            List[Dict[str, Any]]: Filtered alerts
        """
        try:
            filtered_alerts = self.alerts.copy()

            if alert_type:
                filtered_alerts = [
                    alert for alert in filtered_alerts if alert["type"] == alert_type
                ]

            if severity:
                filtered_alerts = [
                    alert for alert in filtered_alerts if alert["severity"] == severity
                ]

            return filtered_alerts

        except Exception:
            self.print(error("Error getting alerts: {e}"))
            return []

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="alerts clearing",
    )
    def clear_alerts(self) -> None:
        """Clear all alerts."""
        try:
            alert_count = len(self.alerts)
            self.alerts.clear()
            self.logger.info(f"Cleared {alert_count} alerts")

        except Exception:
            self.print(error("Error clearing alerts: {e}"))

    def get_sentinel_status(self) -> dict[str, Any]:
        """
        Get sentinel status information.

        Returns:
            Dict[str, Any]: Sentinel status
        """
        return {
            "is_monitoring": self.is_monitoring,
            "monitoring_interval": self.monitoring_interval,
            "alert_threshold": self.alert_threshold,
            "max_alerts": self.max_alerts,
            "current_alerts": len(self.alerts),
            "monitoring_rules_count": len(self.monitoring_rules),
            "alert_callbacks_count": len(self.alert_callbacks),
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="sentinel cleanup",
    )
    async def stop(self) -> None:
        """Stop the sentinel."""
        self.logger.info("🛑 Stopping Sentinel...")

        try:
            # Stop monitoring
            self.is_monitoring = False

            # Clear alerts
            self.clear_alerts()

            # Clear callbacks
            self.alert_callbacks.clear()

            self.logger.info("✅ Sentinel stopped successfully")

        except Exception:
            self.print(error("Error stopping sentinel: {e}"))


# Global sentinel instance
sentinel: Sentinel | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="sentinel setup",
)
async def setup_sentinel(config: dict[str, Any] | None = None) -> Sentinel | None:
    """
    Setup global sentinel.

    Args:
        config: Optional configuration dictionary

    Returns:
        Optional[Sentinel]: Global sentinel instance
    """
    try:
        global sentinel

        if config is None:
            config = {
                "sentinel": {
                    "monitoring_interval": 60,
                    "alert_threshold": 0.8,
                    "max_alerts": 100,
                    "enable_performance_monitoring": True,
                    "enable_error_monitoring": True,
                    "enable_system_monitoring": True,
                },
            }

        # Create sentinel
        sentinel = Sentinel(config)

        # Initialize sentinel
        success = await sentinel.initialize()
        if success:
            return sentinel
        return None

    except Exception as e:
        print(f"Error setting up sentinel: {e}")
        return None
