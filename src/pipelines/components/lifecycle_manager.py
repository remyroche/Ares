"""
Lifecycle manager for pipeline components.

This module provides lifecycle management functionality for pipeline
components, including initialization, execution, and cleanup phases.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any

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
    validation_error,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class LifecycleManager:
    """
    Enhanced lifecycle manager with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize lifecycle manager with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("LifecycleManager")

        # Lifecycle manager state
        self.is_managing: bool = False
        self.lifecycle_results: dict[str, Any] = {}
        self.lifecycle_history: list[dict[str, Any]] = []

        # Configuration
        self.lifecycle_config: dict[str, Any] = self.config.get("lifecycle_manager", {})
        self.lifecycle_interval: int = self.lifecycle_config.get(
            "lifecycle_interval",
            60,
        )
        self.max_lifecycle_history: int = self.lifecycle_config.get(
            "max_lifecycle_history",
            100,
        )
        self.enable_component_management: bool = self.lifecycle_config.get(
            "enable_component_management",
            True,
        )
        self.enable_dependency_management: bool = self.lifecycle_config.get(
            "enable_dependency_management",
            True,
        )

        # Component registry
        self.components: dict[str, Any] = {}
        self.dependencies: dict[str, list[str]] = {}
        self.initialization_callbacks: dict[str, Callable] = {}
        self.cleanup_callbacks: dict[str, Callable] = {}

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid lifecycle manager configuration"),
            AttributeError: (False, "Missing required lifecycle parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="lifecycle manager initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize lifecycle manager with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Lifecycle Manager...")

            # Load lifecycle configuration
            await self._load_lifecycle_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.print(invalid("Invalid configuration for lifecycle manager"))
                return False

            # Initialize lifecycle modules
            await self._initialize_lifecycle_modules()

            self.logger.info(
                "✅ Lifecycle Manager initialization completed successfully",
            )
            return True

        except Exception:
            self.print(failed("❌ Lifecycle Manager initialization failed: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="lifecycle configuration loading",
    )
    async def _load_lifecycle_configuration(self) -> None:
        """Load lifecycle configuration."""
        try:
            # Set default lifecycle parameters
            self.lifecycle_config.setdefault("lifecycle_interval", 60)
            self.lifecycle_config.setdefault("max_lifecycle_history", 100)
            self.lifecycle_config.setdefault("enable_component_management", True)
            self.lifecycle_config.setdefault("enable_dependency_management", True)
            self.lifecycle_config.setdefault("enable_health_monitoring", True)
            self.lifecycle_config.setdefault("enable_graceful_shutdown", True)

            # Update configuration
            self.lifecycle_interval = self.lifecycle_config["lifecycle_interval"]
            self.max_lifecycle_history = self.lifecycle_config["max_lifecycle_history"]
            self.enable_component_management = self.lifecycle_config[
                "enable_component_management"
            ]
            self.enable_dependency_management = self.lifecycle_config[
                "enable_dependency_management"
            ]

            self.logger.info("Lifecycle configuration loaded successfully")

        except Exception:
            self.print(error("Error loading lifecycle configuration: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate lifecycle configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate lifecycle interval
            if self.lifecycle_interval <= 0:
                self.print(invalid("Invalid lifecycle interval"))
                return False

            # Validate max lifecycle history
            if self.max_lifecycle_history <= 0:
                self.print(invalid("Invalid max lifecycle history"))
                return False

            # Validate that at least one lifecycle type is enabled
            if not any(
                [
                    self.enable_component_management,
                    self.enable_dependency_management,
                    self.lifecycle_config.get("enable_health_monitoring", True),
                    self.lifecycle_config.get("enable_graceful_shutdown", True),
                ],
            ):
                self.print(error("At least one lifecycle type must be enabled"))
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception:
            self.print(error("Error validating configuration: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="lifecycle modules initialization",
    )
    async def _initialize_lifecycle_modules(self) -> None:
        """Initialize lifecycle modules."""
        try:
            # Initialize component management module
            if self.enable_component_management:
                await self._initialize_component_management()

            # Initialize dependency management module
            if self.enable_dependency_management:
                await self._initialize_dependency_management()

            # Initialize health monitoring module
            if self.lifecycle_config.get("enable_health_monitoring", True):
                await self._initialize_health_monitoring()

            # Initialize graceful shutdown module
            if self.lifecycle_config.get("enable_graceful_shutdown", True):
                await self._initialize_graceful_shutdown()

            self.logger.info("Lifecycle modules initialized successfully")

        except Exception:
            self.print(
                initialization_error("Error initializing lifecycle modules: {e}"),
            )

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="component management initialization",
    )
    async def _initialize_component_management(self) -> None:
        """Initialize component management module."""
        try:
            # Initialize component management components
            self.component_management_components = {
                "component_registry": True,
                "component_initialization": True,
                "component_cleanup": True,
                "component_monitoring": True,
            }

            self.logger.info("Component management module initialized")

        except Exception:
            self.print(
                initialization_error("Error initializing component management: {e}"),
            )

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="dependency management initialization",
    )
    async def _initialize_dependency_management(self) -> None:
        """Initialize dependency management module."""
        try:
            # Initialize dependency management components
            self.dependency_management_components = {
                "dependency_resolution": True,
                "dependency_validation": True,
                "dependency_monitoring": True,
                "dependency_cleanup": True,
            }

            self.logger.info("Dependency management module initialized")

        except Exception:
            self.print(
                initialization_error("Error initializing dependency management: {e}"),
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
                "health_checks": True,
                "performance_monitoring": True,
                "error_tracking": True,
                "alerting": True,
            }

            self.logger.info("Health monitoring module initialized")

        except Exception:
            self.print(
                initialization_error("Error initializing health monitoring: {e}"),
            )

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="graceful shutdown initialization",
    )
    async def _initialize_graceful_shutdown(self) -> None:
        """Initialize graceful shutdown module."""
        try:
            # Initialize graceful shutdown components
            self.graceful_shutdown_components = {
                "shutdown_signals": True,
                "component_cleanup": True,
                "resource_release": True,
                "finalization": True,
            }

            self.logger.info("Graceful shutdown module initialized")

        except Exception:
            self.print(
                initialization_error("Error initializing graceful shutdown: {e}"),
            )

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid lifecycle parameters"),
            AttributeError: (False, "Missing lifecycle components"),
            KeyError: (False, "Missing required lifecycle data"),
        },
        default_return=False,
        context="lifecycle management",
    )
    async def manage_lifecycle(self, lifecycle_input: dict[str, Any]) -> bool:
        """
        Manage lifecycle operations.

        Args:
            lifecycle_input: Lifecycle input dictionary

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self._validate_lifecycle_inputs(lifecycle_input):
                return False

            self.is_managing = True
            self.logger.info("🔄 Starting lifecycle management...")

            # Perform component management
            if self.enable_component_management:
                component_results = await self._perform_component_management(
                    lifecycle_input,
                )
                self.lifecycle_results["component_management"] = component_results

            # Perform dependency management
            if self.enable_dependency_management:
                dependency_results = await self._perform_dependency_management(
                    lifecycle_input,
                )
                self.lifecycle_results["dependency_management"] = dependency_results

            # Perform health monitoring
            if self.lifecycle_config.get("enable_health_monitoring", True):
                health_results = await self._perform_health_monitoring(lifecycle_input)
                self.lifecycle_results["health_monitoring"] = health_results

            # Perform graceful shutdown
            if self.lifecycle_config.get("enable_graceful_shutdown", True):
                shutdown_results = await self._perform_graceful_shutdown(
                    lifecycle_input,
                )
                self.lifecycle_results["graceful_shutdown"] = shutdown_results

            # Store lifecycle results
            await self._store_lifecycle_results()

            self.is_managing = False
            self.logger.info("✅ Lifecycle management completed successfully")
            return True

        except Exception:
            self.print(error("Error managing lifecycle: {e}"))
            self.is_managing = False
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="lifecycle inputs validation",
    )
    def _validate_lifecycle_inputs(self, lifecycle_input: dict[str, Any]) -> bool:
        """
        Validate lifecycle inputs.

        Args:
            lifecycle_input: Lifecycle input dictionary

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check required lifecycle input fields
            required_fields = ["operation_type", "component_name", "timestamp"]
            for field in required_fields:
                if field not in lifecycle_input:
                    self.logger.error(
                        f"Missing required lifecycle input field: {field}",
                    )
                    return False

            # Validate data types
            if not isinstance(lifecycle_input["operation_type"], str):
                self.print(invalid("Invalid operation type"))
                return False

            if not isinstance(lifecycle_input["component_name"], str):
                self.print(invalid("Invalid component name"))
                return False

            return True

        except Exception:
            self.print(error("Error validating lifecycle inputs: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="component management",
    )
    async def _perform_component_management(
        self,
        lifecycle_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform component management.

        Args:
            lifecycle_input: Lifecycle input dictionary

        Returns:
            Dict[str, Any]: Component management results
        """
        try:
            results = {}

            # Perform component registry
            if self.component_management_components.get("component_registry", False):
                results["component_registry"] = self._perform_component_registry(
                    lifecycle_input,
                )

            # Perform component initialization
            if self.component_management_components.get(
                "component_initialization",
                False,
            ):
                results["component_initialization"] = (
                    self._perform_component_initialization(lifecycle_input)
                )

            # Perform component cleanup
            if self.component_management_components.get("component_cleanup", False):
                results["component_cleanup"] = self._perform_component_cleanup(
                    lifecycle_input,
                )

            # Perform component monitoring
            if self.component_management_components.get("component_monitoring", False):
                results["component_monitoring"] = self._perform_component_monitoring(
                    lifecycle_input,
                )

            self.logger.info("Component management completed")
            return results

        except Exception:
            self.print(error("Error performing component management: {e}"))
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="dependency management",
    )
    async def _perform_dependency_management(
        self,
        lifecycle_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform dependency management.

        Args:
            lifecycle_input: Lifecycle input dictionary

        Returns:
            Dict[str, Any]: Dependency management results
        """
        try:
            results = {}

            # Perform dependency resolution
            if self.dependency_management_components.get(
                "dependency_resolution",
                False,
            ):
                results["dependency_resolution"] = self._perform_dependency_resolution(
                    lifecycle_input,
                )

            # Perform dependency validation
            if self.dependency_management_components.get(
                "dependency_validation",
                False,
            ):
                results["dependency_validation"] = self._perform_dependency_validation(
                    lifecycle_input,
                )

            # Perform dependency monitoring
            if self.dependency_management_components.get(
                "dependency_monitoring",
                False,
            ):
                results["dependency_monitoring"] = self._perform_dependency_monitoring(
                    lifecycle_input,
                )

            # Perform dependency cleanup
            if self.dependency_management_components.get("dependency_cleanup", False):
                results["dependency_cleanup"] = self._perform_dependency_cleanup(
                    lifecycle_input,
                )

            self.logger.info("Dependency management completed")
            return results

        except Exception:
            self.print(error("Error performing dependency management: {e}"))
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="health monitoring",
    )
    async def _perform_health_monitoring(
        self,
        lifecycle_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform health monitoring.

        Args:
            lifecycle_input: Lifecycle input dictionary

        Returns:
            Dict[str, Any]: Health monitoring results
        """
        try:
            results = {}

            # Perform health checks
            if self.health_monitoring_components.get("health_checks", False):
                results["health_checks"] = self._perform_health_checks(lifecycle_input)

            # Perform performance monitoring
            if self.health_monitoring_components.get("performance_monitoring", False):
                results["performance_monitoring"] = (
                    self._perform_performance_monitoring(lifecycle_input)
                )

            # Perform error tracking
            if self.health_monitoring_components.get("error_tracking", False):
                results["error_tracking"] = self._perform_error_tracking(
                    lifecycle_input,
                )

            # Perform alerting
            if self.health_monitoring_components.get("alerting", False):
                results["alerting"] = self._perform_alerting(lifecycle_input)

            self.logger.info("Health monitoring completed")
            return results

        except Exception:
            self.print(error("Error performing health monitoring: {e}"))
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="graceful shutdown",
    )
    async def _perform_graceful_shutdown(
        self,
        lifecycle_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform graceful shutdown.

        Args:
            lifecycle_input: Lifecycle input dictionary

        Returns:
            Dict[str, Any]: Graceful shutdown results
        """
        try:
            results = {}

            # Perform shutdown signals
            if self.graceful_shutdown_components.get("shutdown_signals", False):
                results["shutdown_signals"] = self._perform_shutdown_signals(
                    lifecycle_input,
                )

            # Perform component cleanup
            if self.graceful_shutdown_components.get("component_cleanup", False):
                results["component_cleanup"] = self._perform_shutdown_component_cleanup(
                    lifecycle_input,
                )

            # Perform resource release
            if self.graceful_shutdown_components.get("resource_release", False):
                results["resource_release"] = self._perform_resource_release(
                    lifecycle_input,
                )

            # Perform finalization
            if self.graceful_shutdown_components.get("finalization", False):
                results["finalization"] = self._perform_finalization(lifecycle_input)

            self.logger.info("Graceful shutdown completed")
            return results

        except Exception:
            self.print(error("Error performing graceful shutdown: {e}"))
            return {}

    # Component management methods
    def _perform_component_registry(
        self,
        lifecycle_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform component registry."""
        try:
            # Simulate component registry
            component_name = lifecycle_input.get("component_name", "unknown")

            return {
                "component_name": component_name,
                "registry_status": "registered",
                "registry_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing component registry: {e}"))
            return {}

    def _perform_component_initialization(
        self,
        lifecycle_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform component initialization."""
        try:
            # Simulate component initialization
            component_name = lifecycle_input.get("component_name", "unknown")

            return {
                "component_name": component_name,
                "initialization_status": "initialized",
                "initialization_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(
                initialization_error("Error performing component initialization: {e}"),
            )
            return {}

    def _perform_component_cleanup(
        self,
        lifecycle_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform component cleanup."""
        try:
            # Simulate component cleanup
            component_name = lifecycle_input.get("component_name", "unknown")

            return {
                "component_name": component_name,
                "cleanup_status": "cleaned",
                "cleanup_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing component cleanup: {e}"))
            return {}

    def _perform_component_monitoring(
        self,
        lifecycle_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform component monitoring."""
        try:
            # Simulate component monitoring
            component_name = lifecycle_input.get("component_name", "unknown")

            return {
                "component_name": component_name,
                "monitoring_status": "active",
                "health_score": 0.95,
                "monitoring_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing component monitoring: {e}"))
            return {}

    # Dependency management methods
    def _perform_dependency_resolution(
        self,
        lifecycle_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform dependency resolution."""
        try:
            # Simulate dependency resolution
            return {
                "dependencies_resolved": 5,
                "resolution_status": "resolved",
                "resolution_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing dependency resolution: {e}"))
            return {}

    def _perform_dependency_validation(
        self,
        lifecycle_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform dependency validation."""
        try:
            # Simulate dependency validation
            return {
                "dependencies_validated": 5,
                "validation_status": "valid",
                "validation_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(validation_error("Error performing dependency validation: {e}"))
            return {}

    def _perform_dependency_monitoring(
        self,
        lifecycle_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform dependency monitoring."""
        try:
            # Simulate dependency monitoring
            return {
                "dependencies_monitored": 5,
                "monitoring_status": "active",
                "monitoring_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing dependency monitoring: {e}"))
            return {}

    def _perform_dependency_cleanup(
        self,
        lifecycle_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform dependency cleanup."""
        try:
            # Simulate dependency cleanup
            return {
                "dependencies_cleaned": 5,
                "cleanup_status": "cleaned",
                "cleanup_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing dependency cleanup: {e}"))
            return {}

    # Health monitoring methods
    def _perform_health_checks(self, lifecycle_input: dict[str, Any]) -> dict[str, Any]:
        """Perform health checks."""
        try:
            # Simulate health checks
            return {
                "health_checks_performed": 10,
                "health_status": "healthy",
                "health_score": 0.98,
                "check_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing health checks: {e}"))
            return {}

    def _perform_performance_monitoring(
        self,
        lifecycle_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform performance monitoring."""
        try:
            # Simulate performance monitoring
            return {
                "performance_metrics": ["cpu", "memory", "disk", "network"],
                "performance_status": "normal",
                "performance_score": 0.92,
                "monitoring_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing performance monitoring: {e}"))
            return {}

    def _perform_error_tracking(
        self,
        lifecycle_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform error tracking."""
        try:
            # Simulate error tracking
            return {
                "errors_tracked": 2,
                "error_rate": 0.02,
                "error_status": "low",
                "tracking_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing error tracking: {e}"))
            return {}

    def _perform_alerting(self, lifecycle_input: dict[str, Any]) -> dict[str, Any]:
        """Perform alerting."""
        try:
            # Simulate alerting
            return {
                "alerts_generated": 0,
                "alert_level": "info",
                "alert_status": "normal",
                "alerting_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing alerting: {e}"))
            return {}

    # Graceful shutdown methods
    def _perform_shutdown_signals(
        self,
        lifecycle_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform shutdown signals."""
        try:
            # Simulate shutdown signals
            return {
                "signals_sent": 5,
                "signal_status": "sent",
                "signal_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing shutdown signals: {e}"))
            return {}

    def _perform_shutdown_component_cleanup(
        self,
        lifecycle_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform shutdown component cleanup."""
        try:
            # Simulate shutdown component cleanup
            return {
                "components_cleaned": 5,
                "cleanup_status": "completed",
                "cleanup_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing shutdown component cleanup: {e}"))
            return {}

    def _perform_resource_release(
        self,
        lifecycle_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform resource release."""
        try:
            # Simulate resource release
            return {
                "resources_released": 10,
                "release_status": "completed",
                "release_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing resource release: {e}"))
            return {}

    def _perform_finalization(self, lifecycle_input: dict[str, Any]) -> dict[str, Any]:
        """Perform finalization."""
        try:
            # Simulate finalization
            return {
                "finalization_status": "completed",
                "finalization_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing finalization: {e}"))
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="lifecycle results storage",
    )
    async def _store_lifecycle_results(self) -> None:
        """Store lifecycle results."""
        try:
            # Add timestamp
            self.lifecycle_results["timestamp"] = datetime.now().isoformat()

            # Add to history
            self.lifecycle_history.append(self.lifecycle_results.copy())

            # Limit history size
            if len(self.lifecycle_history) > self.max_lifecycle_history:
                self.lifecycle_history.pop(0)

            self.logger.info("Lifecycle results stored successfully")

        except Exception:
            self.print(error("Error storing lifecycle results: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="lifecycle results getting",
    )
    def get_lifecycle_results(
        self,
        lifecycle_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Get lifecycle results.

        Args:
            lifecycle_type: Optional lifecycle type filter

        Returns:
            Dict[str, Any]: Lifecycle results
        """
        try:
            if lifecycle_type:
                return self.lifecycle_results.get(lifecycle_type, {})
            return self.lifecycle_results.copy()

        except Exception:
            self.print(error("Error getting lifecycle results: {e}"))
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="lifecycle history getting",
    )
    def get_lifecycle_history(
        self,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get lifecycle history.

        Args:
            limit: Optional limit on number of records

        Returns:
            List[Dict[str, Any]]: Lifecycle history
        """
        try:
            history = self.lifecycle_history.copy()

            if limit:
                history = history[-limit:]

            return history

        except Exception:
            self.print(error("Error getting lifecycle history: {e}"))
            return []

    def get_lifecycle_status(self) -> dict[str, Any]:
        """
        Get lifecycle status information.

        Returns:
            Dict[str, Any]: Lifecycle status
        """
        return {
            "is_managing": self.is_managing,
            "lifecycle_interval": self.lifecycle_interval,
            "max_lifecycle_history": self.max_lifecycle_history,
            "enable_component_management": self.enable_component_management,
            "enable_dependency_management": self.enable_dependency_management,
            "enable_health_monitoring": self.lifecycle_config.get(
                "enable_health_monitoring",
                True,
            ),
            "enable_graceful_shutdown": self.lifecycle_config.get(
                "enable_graceful_shutdown",
                True,
            ),
            "lifecycle_history_count": len(self.lifecycle_history),
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="lifecycle manager cleanup",
    )
    async def stop(self) -> None:
        """Stop the lifecycle manager."""
        self.logger.info("🛑 Stopping Lifecycle Manager...")

        try:
            # Stop lifecycle management
            self.is_managing = False

            # Clear results
            self.lifecycle_results.clear()

            # Clear history
            self.lifecycle_history.clear()

            # Clear components
            self.components.clear()

            # Clear dependencies
            self.dependencies.clear()

            self.logger.info("✅ Lifecycle Manager stopped successfully")

        except Exception:
            self.print(error("Error stopping lifecycle manager: {e}"))


# Global lifecycle manager instance
lifecycle_manager: LifecycleManager | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="lifecycle manager setup",
)
async def setup_lifecycle_manager(
    config: dict[str, Any] | None = None,
) -> LifecycleManager | None:
    """
    Setup global lifecycle manager.

    Args:
        config: Optional configuration dictionary

    Returns:
        Optional[LifecycleManager]: Global lifecycle manager instance
    """
    try:
        global lifecycle_manager

        if config is None:
            config = {
                "lifecycle_manager": {
                    "lifecycle_interval": 60,
                    "max_lifecycle_history": 100,
                    "enable_component_management": True,
                    "enable_dependency_management": True,
                    "enable_health_monitoring": True,
                    "enable_graceful_shutdown": True,
                },
            }

        # Create lifecycle manager
        lifecycle_manager = LifecycleManager(config)

        # Initialize lifecycle manager
        success = await lifecycle_manager.initialize()
        if success:
            return lifecycle_manager
        return None

    except Exception as e:
        print(f"Error setting up lifecycle manager: {e}")
        return None
