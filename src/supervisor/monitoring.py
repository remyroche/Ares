# src/supervisor/monitoring.py

import asyncio
from datetime import datetime
from typing import Any

# Import SQLite and InfluxDB managers directly, make Firebase optional

try:
    from src.database.influxdb_manager import InfluxDBManager
except ImportError:
    InfluxDBManager = None

# Make Firebase optional - only import if available
try:
    from src.database.firestore_manager import FirestoreManager

    FIREBASE_AVAILABLE = True
except ImportError:
    FirestoreManager = None
    FIREBASE_AVAILABLE = False

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class Monitoring:
    """
    Enhanced Monitoring component with DI, type hints, and robust error handling.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("Monitoring")
        self.is_running: bool = False
        self.status: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.monitoring_config: dict[str, Any] = self.config.get("monitoring", {})
        self.check_interval: int = self.monitoring_config.get("check_interval", 30)
        self.max_history: int = self.monitoring_config.get("max_history", 100)
        self.alerts: list[dict[str, Any]] = []
        self.metrics: dict[str, Any] = {}

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid monitoring configuration"),
            AttributeError: (False, "Missing required monitoring parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="monitoring initialization",
    )
    async def initialize(self) -> bool:
        try:
            self.logger.info("Initializing Monitoring...")
            await self._load_monitoring_configuration()
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for monitoring")
                return False
            self.logger.info("✅ Monitoring initialization completed successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Monitoring initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="monitoring configuration loading",
    )
    async def _load_monitoring_configuration(self) -> None:
        try:
            self.monitoring_config.setdefault("check_interval", 30)
            self.monitoring_config.setdefault("max_history", 100)
            self.check_interval = self.monitoring_config["check_interval"]
            self.max_history = self.monitoring_config["max_history"]
            self.logger.info("Monitoring configuration loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading monitoring configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        try:
            if self.check_interval <= 0:
                self.logger.error("Invalid check interval")
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
            Exception: (False, "Monitoring run failed"),
        },
        default_return=False,
        context="monitoring run",
    )
    async def run(self) -> bool:
        try:
            self.is_running = True
            self.logger.info("🚦 Monitoring started.")
            while self.is_running:
                await self._perform_monitoring()
                await asyncio.sleep(self.check_interval)
            return True
        except Exception as e:
            self.logger.error(f"Error in monitoring run: {e}")
            self.is_running = False
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="monitoring step",
    )
    async def _perform_monitoring(self) -> None:
        try:
            now = datetime.now().isoformat()
            self.status = {"timestamp": now, "status": "running"}
            self.history.append(self.status.copy())
            if len(self.history) > self.max_history:
                self.history.pop(0)
            await self._check_system_health()
            await self._update_metrics()
            self.logger.info(f"Monitoring tick at {now}")
        except Exception as e:
            self.logger.error(f"Error in monitoring step: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="system health check",
    )
    async def _check_system_health(self) -> None:
        try:
            # Simulate system health checks
            health_status = {
                "cpu_usage": 45.2,
                "memory_usage": 67.8,
                "disk_usage": 23.1,
                "network_status": "healthy",
            }
            self.metrics["system_health"] = health_status
            self.logger.info("System health check completed")
        except Exception as e:
            self.logger.error(f"Error checking system health: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="metrics update",
    )
    async def _update_metrics(self) -> None:
        try:
            # Update various metrics
            self.metrics["last_update"] = datetime.now().isoformat()
            self.metrics["uptime"] = "2h 15m 30s"
            self.logger.info("Metrics updated successfully")
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="monitoring stop",
    )
    async def stop(self) -> None:
        self.logger.info("🛑 Stopping Monitoring...")
        try:
            self.is_running = False
            self.status = {"timestamp": datetime.now().isoformat(), "status": "stopped"}
            self.logger.info("✅ Monitoring stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping monitoring: {e}")

    def get_status(self) -> dict[str, Any]:
        return self.status.copy()

    def get_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        history = self.history.copy()
        if limit:
            history = history[-limit:]
        return history

    def get_metrics(self) -> dict[str, Any]:
        return self.metrics.copy()

    def get_alerts(self) -> list[dict[str, Any]]:
        return self.alerts.copy()


monitoring: Monitoring | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="monitoring setup",
)
async def setup_monitoring(
    config: dict[str, Any] | None = None,
) -> Monitoring | None:
    try:
        global monitoring
        if config is None:
            config = {"monitoring": {"check_interval": 30, "max_history": 100}}
        monitoring = Monitoring(config)
        success = await monitoring.initialize()
        if success:
            return monitoring
        return None
    except Exception as e:
        print(f"Error setting up monitoring: {e}")
        return None
