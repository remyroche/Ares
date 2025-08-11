"""
Monitoring manager for pipeline metrics and performance tracking.

This module provides monitoring functionality for pipelines,
including metrics collection, performance tracking, and health monitoring.
"""

from datetime import datetime
from typing import Any

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    connection_error,
    error,
    failed,
    initialization_error,
    invalid,
)


class MonitoringManager:
    """
    Enhanced monitoring manager with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize monitoring manager with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("MonitoringManager")

        # Monitoring manager state
        self.is_monitoring: bool = False
        self.monitoring_results: dict[str, Any] = {}
        self.monitoring_history: list[dict[str, Any]] = []

        # Configuration
        self.monitoring_config: dict[str, Any] = self.config.get(
            "monitoring_manager",
            {},
        )
        self.monitoring_interval: int = self.monitoring_config.get(
            "monitoring_interval",
            60,
        )
        self.max_monitoring_history: int = self.monitoring_config.get(
            "max_monitoring_history",
            1000,
        )
        self.enable_performance_monitoring: bool = self.monitoring_config.get(
            "enable_performance_monitoring",
            True,
        )
        self.enable_health_monitoring: bool = self.monitoring_config.get(
            "enable_health_monitoring",
            True,
        )

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid monitoring manager configuration"),
            AttributeError: (False, "Missing required monitoring parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="monitoring manager initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize monitoring manager with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Monitoring Manager...")

            # Load monitoring configuration
            await self._load_monitoring_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.print(invalid("Invalid configuration for monitoring manager"))
                return False

            # Initialize monitoring modules
            await self._initialize_monitoring_modules()

            self.logger.info(
                "✅ Monitoring Manager initialization completed successfully",
            )
            return True

        except Exception:
            self.print(failed("❌ Monitoring Manager initialization failed: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="monitoring configuration loading",
    )
    async def _load_monitoring_configuration(self) -> None:
        """Load monitoring configuration."""
        try:
            # Set default monitoring parameters
            self.monitoring_config.setdefault("monitoring_interval", 60)
            self.monitoring_config.setdefault("max_monitoring_history", 1000)
            self.monitoring_config.setdefault("enable_performance_monitoring", True)
            self.monitoring_config.setdefault("enable_health_monitoring", True)
            self.monitoring_config.setdefault("enable_alerting", True)
            self.monitoring_config.setdefault("enable_metrics_collection", True)

            # Update configuration
            self.monitoring_interval = self.monitoring_config["monitoring_interval"]
            self.max_monitoring_history = self.monitoring_config[
                "max_monitoring_history"
            ]
            self.enable_performance_monitoring = self.monitoring_config[
                "enable_performance_monitoring"
            ]
            self.enable_health_monitoring = self.monitoring_config[
                "enable_health_monitoring"
            ]

            self.logger.info("Monitoring configuration loaded successfully")

        except Exception:
            self.print(error("Error loading monitoring configuration: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate monitoring configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate monitoring interval
            if self.monitoring_interval <= 0:
                self.print(invalid("Invalid monitoring interval"))
                return False

            # Validate max monitoring history
            if self.max_monitoring_history <= 0:
                self.print(invalid("Invalid max monitoring history"))
                return False

            # Validate that at least one monitoring type is enabled
            if not any(
                [
                    self.enable_performance_monitoring,
                    self.enable_health_monitoring,
                    self.monitoring_config.get("enable_alerting", True),
                    self.monitoring_config.get("enable_metrics_collection", True),
                ],
            ):
                self.print(error("At least one monitoring type must be enabled"))
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception:
            self.print(error("Error validating configuration: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="monitoring modules initialization",
    )
    async def _initialize_monitoring_modules(self) -> None:
        """Initialize monitoring modules."""
        try:
            # Initialize performance monitoring module
            if self.enable_performance_monitoring:
                await self._initialize_performance_monitoring()

            # Initialize health monitoring module
            if self.enable_health_monitoring:
                await self._initialize_health_monitoring()

            # Initialize alerting module
            if self.monitoring_config.get("enable_alerting", True):
                await self._initialize_alerting()

            # Initialize metrics collection module
            if self.monitoring_config.get("enable_metrics_collection", True):
                await self._initialize_metrics_collection()

            self.logger.info("Monitoring modules initialized successfully")

        except Exception:
            self.print(
                initialization_error("Error initializing monitoring modules: {e}"),
            )

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="performance monitoring initialization",
    )
    async def _initialize_performance_monitoring(self) -> None:
        """Initialize performance monitoring module."""
        try:
            # Initialize performance monitoring components
            self.performance_monitoring_components = {
                "cpu_monitoring": True,
                "memory_monitoring": True,
                "disk_monitoring": True,
                "network_monitoring": True,
            }

            self.logger.info("Performance monitoring module initialized")

        except Exception:
            self.print(
                initialization_error("Error initializing performance monitoring: {e}"),
            )

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="health monitoring initialization",
    )
    async def _initialize_health_monitoring(self) -> None:
        """Initialize health monitoring module."""
        try:
            # Initialize health monitoring components
            self.health_monitoring_components = {
                "system_health": True,
                "application_health": True,
                "service_health": True,
                "dependency_health": True,
            }

            self.logger.info("Health monitoring module initialized")

        except Exception:
            self.print(
                initialization_error("Error initializing health monitoring: {e}"),
            )

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="alerting initialization",
    )
    async def _initialize_alerting(self) -> None:
        """Initialize alerting module."""
        try:
            # Initialize alerting components
            self.alerting_components = {
                "alert_generation": True,
                "alert_routing": True,
                "alert_escalation": True,
                "alert_resolution": True,
            }

            self.logger.info("Alerting module initialized")

        except Exception:
            self.print(initialization_error("Error initializing alerting: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="metrics collection initialization",
    )
    async def _initialize_metrics_collection(self) -> None:
        """Initialize metrics collection module."""
        try:
            # Initialize metrics collection components
            self.metrics_collection_components = {
                "metrics_gathering": True,
                "metrics_processing": True,
                "metrics_storage": True,
                "metrics_analysis": True,
            }

            self.logger.info("Metrics collection module initialized")

        except Exception:
            self.print(
                initialization_error("Error initializing metrics collection: {e}"),
            )

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid monitoring parameters"),
            AttributeError: (False, "Missing monitoring components"),
            KeyError: (False, "Missing required monitoring data"),
        },
        default_return=False,
        context="monitoring execution",
    )
    async def execute_monitoring(self, monitoring_input: dict[str, Any]) -> bool:
        """
        Execute monitoring operations.

        Args:
            monitoring_input: Monitoring input dictionary

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self._validate_monitoring_inputs(monitoring_input):
                return False

            self.is_monitoring = True
            self.logger.info("🔄 Starting monitoring execution...")

            # Perform performance monitoring
            if self.enable_performance_monitoring:
                performance_results = await self._perform_performance_monitoring(
                    monitoring_input,
                )
                self.monitoring_results["performance_monitoring"] = performance_results

            # Perform health monitoring
            if self.enable_health_monitoring:
                health_results = await self._perform_health_monitoring(monitoring_input)
                self.monitoring_results["health_monitoring"] = health_results

            # Perform alerting
            if self.monitoring_config.get("enable_alerting", True):
                alerting_results = await self._perform_alerting(monitoring_input)
                self.monitoring_results["alerting"] = alerting_results

            # Perform metrics collection
            if self.monitoring_config.get("enable_metrics_collection", True):
                metrics_results = await self._perform_metrics_collection(
                    monitoring_input,
                )
                self.monitoring_results["metrics_collection"] = metrics_results

            # Store monitoring results
            await self._store_monitoring_results()

            self.is_monitoring = False
            self.logger.info("✅ Monitoring execution completed successfully")
            return True

        except Exception:
            self.print(error("Error executing monitoring: {e}"))
            self.is_monitoring = False
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="monitoring inputs validation",
    )
    def _validate_monitoring_inputs(self, monitoring_input: dict[str, Any]) -> bool:
        """
        Validate monitoring inputs.

        Args:
            monitoring_input: Monitoring input dictionary

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check required monitoring input fields
            required_fields = ["monitoring_type", "target_system", "timestamp"]
            for field in required_fields:
                if field not in monitoring_input:
                    self.logger.error(
                        f"Missing required monitoring input field: {field}",
                    )
                    return False

            # Validate data types
            if not isinstance(monitoring_input["monitoring_type"], str):
                self.print(invalid("Invalid monitoring type"))
                return False

            if not isinstance(monitoring_input["target_system"], str):
                self.print(invalid("Invalid target system"))
                return False

            return True

        except Exception:
            self.print(error("Error validating monitoring inputs: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="performance monitoring",
    )
    async def _perform_performance_monitoring(
        self,
        monitoring_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform performance monitoring.

        Args:
            monitoring_input: Monitoring input dictionary

        Returns:
            Dict[str, Any]: Performance monitoring results
        """
        try:
            results = {}

            # Perform CPU monitoring
            if self.performance_monitoring_components.get("cpu_monitoring", False):
                results["cpu_monitoring"] = self._perform_cpu_monitoring(
                    monitoring_input,
                )

            # Perform memory monitoring
            if self.performance_monitoring_components.get("memory_monitoring", False):
                results["memory_monitoring"] = self._perform_memory_monitoring(
                    monitoring_input,
                )

            # Perform disk monitoring
            if self.performance_monitoring_components.get("disk_monitoring", False):
                results["disk_monitoring"] = self._perform_disk_monitoring(
                    monitoring_input,
                )

            # Perform network monitoring
            if self.performance_monitoring_components.get("network_monitoring", False):
                results["network_monitoring"] = self._perform_network_monitoring(
                    monitoring_input,
                )

            self.logger.info("Performance monitoring completed")
            return results

        except Exception:
            self.print(error("Error performing performance monitoring: {e}"))
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="health monitoring",
    )
    async def _perform_health_monitoring(
        self,
        monitoring_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform health monitoring.

        Args:
            monitoring_input: Monitoring input dictionary

        Returns:
            Dict[str, Any]: Health monitoring results
        """
        try:
            results = {}

            # Perform system health monitoring
            if self.health_monitoring_components.get("system_health", False):
                results["system_health"] = self._perform_system_health_monitoring(
                    monitoring_input,
                )

            # Perform application health monitoring
            if self.health_monitoring_components.get("application_health", False):
                results["application_health"] = (
                    self._perform_application_health_monitoring(monitoring_input)
                )

            # Perform service health monitoring
            if self.health_monitoring_components.get("service_health", False):
                results["service_health"] = self._perform_service_health_monitoring(
                    monitoring_input,
                )

            # Perform dependency health monitoring
            if self.health_monitoring_components.get("dependency_health", False):
                results["dependency_health"] = (
                    self._perform_dependency_health_monitoring(monitoring_input)
                )

            self.logger.info("Health monitoring completed")
            return results

        except Exception:
            self.print(error("Error performing health monitoring: {e}"))
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="alerting",
    )
    async def _perform_alerting(
        self,
        monitoring_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform alerting.

        Args:
            monitoring_input: Monitoring input dictionary

        Returns:
            Dict[str, Any]: Alerting results
        """
        try:
            results = {}

            # Perform alert generation
            if self.alerting_components.get("alert_generation", False):
                results["alert_generation"] = self._perform_alert_generation(
                    monitoring_input,
                )

            # Perform alert routing
            if self.alerting_components.get("alert_routing", False):
                results["alert_routing"] = self._perform_alert_routing(monitoring_input)

            # Perform alert escalation
            if self.alerting_components.get("alert_escalation", False):
                results["alert_escalation"] = self._perform_alert_escalation(
                    monitoring_input,
                )

            # Perform alert resolution
            if self.alerting_components.get("alert_resolution", False):
                results["alert_resolution"] = self._perform_alert_resolution(
                    monitoring_input,
                )

            self.logger.info("Alerting completed")
            return results

        except Exception:
            self.print(error("Error performing alerting: {e}"))
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="metrics collection",
    )
    async def _perform_metrics_collection(
        self,
        monitoring_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform metrics collection.

        Args:
            monitoring_input: Monitoring input dictionary

        Returns:
            Dict[str, Any]: Metrics collection results
        """
        try:
            results = {}

            # Perform metrics gathering
            if self.metrics_collection_components.get("metrics_gathering", False):
                results["metrics_gathering"] = self._perform_metrics_gathering(
                    monitoring_input,
                )

            # Perform metrics processing
            if self.metrics_collection_components.get("metrics_processing", False):
                results["metrics_processing"] = self._perform_metrics_processing(
                    monitoring_input,
                )

            # Perform metrics storage
            if self.metrics_collection_components.get("metrics_storage", False):
                results["metrics_storage"] = self._perform_metrics_storage(
                    monitoring_input,
                )

            # Perform metrics analysis
            if self.metrics_collection_components.get("metrics_analysis", False):
                results["metrics_analysis"] = self._perform_metrics_analysis(
                    monitoring_input,
                )

            self.logger.info("Metrics collection completed")
            return results

        except Exception:
            self.print(error("Error performing metrics collection: {e}"))
            return {}

    # Performance monitoring methods
    def _perform_cpu_monitoring(
        self,
        monitoring_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform CPU monitoring."""
        try:
            # Simulate CPU monitoring
            return {
                "cpu_usage": 45.2,
                "cpu_load": 2.1,
                "cpu_temperature": 65.0,
                "monitoring_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing CPU monitoring: {e}"))
            return {}

    def _perform_memory_monitoring(
        self,
        monitoring_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform memory monitoring."""
        try:
            # Simulate memory monitoring
            return {
                "memory_usage": 78.5,
                "memory_available": 4.2,
                "memory_total": 16.0,
                "monitoring_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing memory monitoring: {e}"))
            return {}

    def _perform_disk_monitoring(
        self,
        monitoring_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform disk monitoring."""
        try:
            # Simulate disk monitoring
            return {
                "disk_usage": 62.3,
                "disk_free": 187.5,
                "disk_total": 500.0,
                "monitoring_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing disk monitoring: {e}"))
            return {}

    def _perform_network_monitoring(
        self,
        monitoring_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform network monitoring."""
        try:
            # Simulate network monitoring
            return {
                "network_latency": 15.2,
                "network_throughput": 125.8,
                "network_packet_loss": 0.1,
                "monitoring_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(connection_error("Error performing network monitoring: {e}"))
            return {}

    # Health monitoring methods
    def _perform_system_health_monitoring(
        self,
        monitoring_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform system health monitoring."""
        try:
            # Simulate system health monitoring
            return {
                "system_status": "healthy",
                "system_score": 0.95,
                "system_issues": 0,
                "monitoring_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing system health monitoring: {e}"))
            return {}

    def _perform_application_health_monitoring(
        self,
        monitoring_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform application health monitoring."""
        try:
            # Simulate application health monitoring
            return {
                "application_status": "healthy",
                "application_score": 0.92,
                "application_issues": 1,
                "monitoring_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.exception(
                f"Error performing application health monitoring: {e}",
            )
            return {}

    def _perform_service_health_monitoring(
        self,
        monitoring_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform service health monitoring."""
        try:
            # Simulate service health monitoring
            return {
                "service_status": "healthy",
                "service_score": 0.98,
                "service_issues": 0,
                "monitoring_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing service health monitoring: {e}"))
            return {}

    def _perform_dependency_health_monitoring(
        self,
        monitoring_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform dependency health monitoring."""
        try:
            # Simulate dependency health monitoring
            return {
                "dependency_status": "healthy",
                "dependency_score": 0.94,
                "dependency_issues": 2,
                "monitoring_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing dependency health monitoring: {e}"))
            return {}

    # Alerting methods
    def _perform_alert_generation(
        self,
        monitoring_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform alert generation."""
        try:
            # Simulate alert generation
            return {
                "alerts_generated": 0,
                "alert_level": "info",
                "alert_status": "normal",
                "generation_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing alert generation: {e}"))
            return {}

    def _perform_alert_routing(
        self,
        monitoring_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform alert routing."""
        try:
            # Simulate alert routing
            return {
                "alerts_routed": 0,
                "routing_status": "completed",
                "routing_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing alert routing: {e}"))
            return {}

    def _perform_alert_escalation(
        self,
        monitoring_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform alert escalation."""
        try:
            # Simulate alert escalation
            return {
                "alerts_escalated": 0,
                "escalation_status": "none",
                "escalation_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing alert escalation: {e}"))
            return {}

    def _perform_alert_resolution(
        self,
        monitoring_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform alert resolution."""
        try:
            # Simulate alert resolution
            return {
                "alerts_resolved": 0,
                "resolution_status": "none",
                "resolution_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing alert resolution: {e}"))
            return {}

    # Metrics collection methods
    def _perform_metrics_gathering(
        self,
        monitoring_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform metrics gathering."""
        try:
            # Simulate metrics gathering
            return {
                "metrics_gathered": 25,
                "gathering_status": "completed",
                "gathering_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing metrics gathering: {e}"))
            return {}

    def _perform_metrics_processing(
        self,
        monitoring_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform metrics processing."""
        try:
            # Simulate metrics processing
            return {
                "metrics_processed": 25,
                "processing_status": "completed",
                "processing_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing metrics processing: {e}"))
            return {}

    def _perform_metrics_storage(
        self,
        monitoring_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform metrics storage."""
        try:
            # Simulate metrics storage
            return {
                "metrics_stored": 25,
                "storage_status": "completed",
                "storage_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing metrics storage: {e}"))
            return {}

    def _perform_metrics_analysis(
        self,
        monitoring_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform metrics analysis."""
        try:
            # Simulate metrics analysis
            return {
                "metrics_analyzed": 25,
                "analysis_status": "completed",
                "analysis_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing metrics analysis: {e}"))
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="monitoring results storage",
    )
    async def _store_monitoring_results(self) -> None:
        """Store monitoring results."""
        try:
            # Add timestamp
            self.monitoring_results["timestamp"] = datetime.now().isoformat()

            # Add to history
            self.monitoring_history.append(self.monitoring_results.copy())

            # Limit history size
            if len(self.monitoring_history) > self.max_monitoring_history:
                self.monitoring_history.pop(0)

            self.logger.info("Monitoring results stored successfully")

        except Exception:
            self.print(error("Error storing monitoring results: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="monitoring results getting",
    )
    def get_monitoring_results(
        self,
        monitoring_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Get monitoring results.

        Args:
            monitoring_type: Optional monitoring type filter

        Returns:
            Dict[str, Any]: Monitoring results
        """
        try:
            if monitoring_type:
                return self.monitoring_results.get(monitoring_type, {})
            return self.monitoring_results.copy()

        except Exception:
            self.print(error("Error getting monitoring results: {e}"))
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="monitoring history getting",
    )
    def get_monitoring_history(
        self,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get monitoring history.

        Args:
            limit: Optional limit on number of records

        Returns:
            List[Dict[str, Any]]: Monitoring history
        """
        try:
            history = self.monitoring_history.copy()

            if limit:
                history = history[-limit:]

            return history

        except Exception:
            self.print(error("Error getting monitoring history: {e}"))
            return []

    def get_monitoring_status(self) -> dict[str, Any]:
        """
        Get monitoring status information.

        Returns:
            Dict[str, Any]: Monitoring status
        """
        return {
            "is_monitoring": self.is_monitoring,
            "monitoring_interval": self.monitoring_interval,
            "max_monitoring_history": self.max_monitoring_history,
            "enable_performance_monitoring": self.enable_performance_monitoring,
            "enable_health_monitoring": self.enable_health_monitoring,
            "enable_alerting": self.monitoring_config.get("enable_alerting", True),
            "enable_metrics_collection": self.monitoring_config.get(
                "enable_metrics_collection",
                True,
            ),
            "monitoring_history_count": len(self.monitoring_history),
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="monitoring manager cleanup",
    )
    async def stop(self) -> None:
        """Stop the monitoring manager."""
        self.logger.info("🛑 Stopping Monitoring Manager...")

        try:
            # Stop monitoring
            self.is_monitoring = False

            # Clear results
            self.monitoring_results.clear()

            # Clear history
            self.monitoring_history.clear()

            self.logger.info("✅ Monitoring Manager stopped successfully")

        except Exception:
            self.print(error("Error stopping monitoring manager: {e}"))


# Global monitoring manager instance
monitoring_manager: MonitoringManager | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="monitoring manager setup",
)
async def setup_monitoring_manager(
    config: dict[str, Any] | None = None,
) -> MonitoringManager | None:
    """
    Setup global monitoring manager.

    Args:
        config: Optional configuration dictionary

    Returns:
        Optional[MonitoringManager]: Global monitoring manager instance
    """
    try:
        global monitoring_manager

        if config is None:
            config = {
                "monitoring_manager": {
                    "monitoring_interval": 60,
                    "max_monitoring_history": 1000,
                    "enable_performance_monitoring": True,
                    "enable_health_monitoring": True,
                    "enable_alerting": True,
                    "enable_metrics_collection": True,
                },
            }

        # Create monitoring manager
        monitoring_manager = MonitoringManager(config)

        # Initialize monitoring manager
        success = await monitoring_manager.initialize()
        if success:
            return monitoring_manager
        return None

    except Exception as e:
        print(f"Error setting up monitoring manager: {e}")
        return None
