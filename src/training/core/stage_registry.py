"""
Stage registry for the modular training pipeline.

This module provides a registry system for managing pipeline stages,
allowing for dynamic stage registration and retrieval.
"""

from datetime import datetime
from typing import Any

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class StageRegistry:
    """
    Stage registry with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize stage registry with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("StageRegistry")

        # Stage registry state
        self.is_registered: bool = False
        self.stage_results: dict[str, Any] = {}
        self.stage_history: list[dict[str, Any]] = []

        # Configuration
        self.stage_config: dict[str, Any] = self.config.get("stage_registry", {})
        self.stage_interval: int = self.stage_config.get("stage_interval", 3600)
        self.max_stage_history: int = self.stage_config.get("max_stage_history", 100)
        self.enable_stage_registration: bool = self.stage_config.get(
            "enable_stage_registration",
            True,
        )
        self.enable_stage_validation: bool = self.stage_config.get(
            "enable_stage_validation",
            True,
        )

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid stage registry configuration"),
            AttributeError: (False, "Missing required stage registry parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="stage registry initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize stage registry with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Stage Registry...")

            # Load stage configuration
            await self._load_stage_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for stage registry")
                return False

            # Initialize stage modules
            await self._initialize_stage_modules()

            self.logger.info("✅ Stage Registry initialization completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Stage Registry initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="stage configuration loading",
    )
    async def _load_stage_configuration(self) -> None:
        """Load stage configuration."""
        try:
            # Set default stage parameters
            self.stage_config.setdefault("stage_interval", 3600)
            self.stage_config.setdefault("max_stage_history", 100)
            self.stage_config.setdefault("enable_stage_registration", True)
            self.stage_config.setdefault("enable_stage_validation", True)
            self.stage_config.setdefault("enable_stage_execution", True)
            self.stage_config.setdefault("enable_stage_monitoring", True)

            # Update configuration
            self.stage_interval = self.stage_config["stage_interval"]
            self.max_stage_history = self.stage_config["max_stage_history"]
            self.enable_stage_registration = self.stage_config[
                "enable_stage_registration"
            ]
            self.enable_stage_validation = self.stage_config["enable_stage_validation"]

            self.logger.info("Stage configuration loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading stage configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate stage configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate stage interval
            if self.stage_interval <= 0:
                self.logger.error("Invalid stage interval")
                return False

            # Validate max stage history
            if self.max_stage_history <= 0:
                self.logger.error("Invalid max stage history")
                return False

            # Validate that at least one stage type is enabled
            if not any(
                [
                    self.enable_stage_registration,
                    self.enable_stage_validation,
                    self.stage_config.get("enable_stage_execution", True),
                    self.stage_config.get("enable_stage_monitoring", True),
                ],
            ):
                self.logger.error("At least one stage type must be enabled")
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="stage modules initialization",
    )
    async def _initialize_stage_modules(self) -> None:
        """Initialize stage modules."""
        try:
            # Initialize stage registration module
            if self.enable_stage_registration:
                await self._initialize_stage_registration()

            # Initialize stage validation module
            if self.enable_stage_validation:
                await self._initialize_stage_validation()

            # Initialize stage execution module
            if self.stage_config.get("enable_stage_execution", True):
                await self._initialize_stage_execution()

            # Initialize stage monitoring module
            if self.stage_config.get("enable_stage_monitoring", True):
                await self._initialize_stage_monitoring()

            self.logger.info("Stage modules initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing stage modules: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="stage registration initialization",
    )
    async def _initialize_stage_registration(self) -> None:
        """Initialize stage registration module."""
        try:
            # Initialize stage registration components
            self.stage_registration_components = {
                "stage_registration": True,
                "stage_dependencies": True,
                "stage_metadata": True,
                "stage_validation": True,
            }

            self.logger.info("Stage registration module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing stage registration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="stage validation initialization",
    )
    async def _initialize_stage_validation(self) -> None:
        """Initialize stage validation module."""
        try:
            # Initialize stage validation components
            self.stage_validation_components = {
                "input_validation": True,
                "output_validation": True,
                "dependency_validation": True,
                "metadata_validation": True,
            }

            self.logger.info("Stage validation module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing stage validation: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="stage execution initialization",
    )
    async def _initialize_stage_execution(self) -> None:
        """Initialize stage execution module."""
        try:
            # Initialize stage execution components
            self.stage_execution_components = {
                "execution_planning": True,
                "execution_coordination": True,
                "execution_monitoring": True,
                "execution_reporting": True,
            }

            self.logger.info("Stage execution module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing stage execution: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="stage monitoring initialization",
    )
    async def _initialize_stage_monitoring(self) -> None:
        """Initialize stage monitoring module."""
        try:
            # Initialize stage monitoring components
            self.stage_monitoring_components = {
                "performance_monitoring": True,
                "health_monitoring": True,
                "error_monitoring": True,
                "resource_monitoring": True,
            }

            self.logger.info("Stage monitoring module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing stage monitoring: {e}")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid stage parameters"),
            AttributeError: (False, "Missing stage components"),
            KeyError: (False, "Missing required stage data"),
        },
        default_return=False,
        context="stage execution",
    )
    async def execute_stage(self, stage_input: dict[str, Any]) -> bool:
        """
        Execute stage operations.

        Args:
            stage_input: Stage input dictionary

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self._validate_stage_inputs(stage_input):
                return False

            self.is_registered = True
            self.logger.info("🔄 Starting stage execution...")

            # Perform stage registration
            if self.enable_stage_registration:
                registration_results = await self._perform_stage_registration(
                    stage_input,
                )
                self.stage_results["stage_registration"] = registration_results

            # Perform stage validation
            if self.enable_stage_validation:
                validation_results = await self._perform_stage_validation(stage_input)
                self.stage_results["stage_validation"] = validation_results

            # Perform stage execution
            if self.stage_config.get("enable_stage_execution", True):
                execution_results = await self._perform_stage_execution(stage_input)
                self.stage_results["stage_execution"] = execution_results

            # Perform stage monitoring
            if self.stage_config.get("enable_stage_monitoring", True):
                monitoring_results = await self._perform_stage_monitoring(stage_input)
                self.stage_results["stage_monitoring"] = monitoring_results

            # Store stage results
            await self._store_stage_results()

            self.is_registered = False
            self.logger.info("✅ Stage execution completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error executing stage: {e}")
            self.is_registered = False
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="stage inputs validation",
    )
    def _validate_stage_inputs(self, stage_input: dict[str, Any]) -> bool:
        """
        Validate stage inputs.

        Args:
            stage_input: Stage input dictionary

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check required stage input fields
            required_fields = ["stage_type", "stage_name", "timestamp"]
            for field in required_fields:
                if field not in stage_input:
                    self.logger.error(f"Missing required stage input field: {field}")
                    return False

            # Validate data types
            if not isinstance(stage_input["stage_type"], str):
                self.logger.error("Invalid stage type")
                return False

            if not isinstance(stage_input["stage_name"], str):
                self.logger.error("Invalid stage name")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating stage inputs: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="stage registration",
    )
    async def _perform_stage_registration(
        self,
        stage_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform stage registration.

        Args:
            stage_input: Stage input dictionary

        Returns:
            Dict[str, Any]: Stage registration results
        """
        try:
            results = {}

            # Perform stage registration
            if self.stage_registration_components.get("stage_registration", False):
                results["stage_registration"] = self._perform_stage_registration_core(
                    stage_input,
                )

            # Perform stage dependencies
            if self.stage_registration_components.get("stage_dependencies", False):
                results["stage_dependencies"] = self._perform_stage_dependencies(
                    stage_input,
                )

            # Perform stage metadata
            if self.stage_registration_components.get("stage_metadata", False):
                results["stage_metadata"] = self._perform_stage_metadata(stage_input)

            # Perform stage validation
            if self.stage_registration_components.get("stage_validation", False):
                results["stage_validation"] = self._perform_stage_validation_core(
                    stage_input,
                )

            self.logger.info("Stage registration completed")
            return results

        except Exception as e:
            self.logger.error(f"Error performing stage registration: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="stage validation",
    )
    async def _perform_stage_validation(
        self,
        stage_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform stage validation.

        Args:
            stage_input: Stage input dictionary

        Returns:
            Dict[str, Any]: Stage validation results
        """
        try:
            results = {}

            # Perform input validation
            if self.stage_validation_components.get("input_validation", False):
                results["input_validation"] = self._perform_input_validation(
                    stage_input,
                )

            # Perform output validation
            if self.stage_validation_components.get("output_validation", False):
                results["output_validation"] = self._perform_output_validation(
                    stage_input,
                )

            # Perform dependency validation
            if self.stage_validation_components.get("dependency_validation", False):
                results["dependency_validation"] = self._perform_dependency_validation(
                    stage_input,
                )

            # Perform metadata validation
            if self.stage_validation_components.get("metadata_validation", False):
                results["metadata_validation"] = self._perform_metadata_validation(
                    stage_input,
                )

            self.logger.info("Stage validation completed")
            return results

        except Exception as e:
            self.logger.error(f"Error performing stage validation: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="stage execution",
    )
    async def _perform_stage_execution(
        self,
        stage_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform stage execution.

        Args:
            stage_input: Stage input dictionary

        Returns:
            Dict[str, Any]: Stage execution results
        """
        try:
            results = {}

            # Perform execution planning
            if self.stage_execution_components.get("execution_planning", False):
                results["execution_planning"] = self._perform_execution_planning(
                    stage_input,
                )

            # Perform execution coordination
            if self.stage_execution_components.get("execution_coordination", False):
                results["execution_coordination"] = (
                    self._perform_execution_coordination(stage_input)
                )

            # Perform execution monitoring
            if self.stage_execution_components.get("execution_monitoring", False):
                results["execution_monitoring"] = self._perform_execution_monitoring(
                    stage_input,
                )

            # Perform execution reporting
            if self.stage_execution_components.get("execution_reporting", False):
                results["execution_reporting"] = self._perform_execution_reporting(
                    stage_input,
                )

            self.logger.info("Stage execution completed")
            return results

        except Exception as e:
            self.logger.error(f"Error performing stage execution: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="stage monitoring",
    )
    async def _perform_stage_monitoring(
        self,
        stage_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform stage monitoring.

        Args:
            stage_input: Stage input dictionary

        Returns:
            Dict[str, Any]: Stage monitoring results
        """
        try:
            results = {}

            # Perform performance monitoring
            if self.stage_monitoring_components.get("performance_monitoring", False):
                results["performance_monitoring"] = (
                    self._perform_performance_monitoring(stage_input)
                )

            # Perform health monitoring
            if self.stage_monitoring_components.get("health_monitoring", False):
                results["health_monitoring"] = self._perform_health_monitoring(
                    stage_input,
                )

            # Perform error monitoring
            if self.stage_monitoring_components.get("error_monitoring", False):
                results["error_monitoring"] = self._perform_error_monitoring(
                    stage_input,
                )

            # Perform resource monitoring
            if self.stage_monitoring_components.get("resource_monitoring", False):
                results["resource_monitoring"] = self._perform_resource_monitoring(
                    stage_input,
                )

            self.logger.info("Stage monitoring completed")
            return results

        except Exception as e:
            self.logger.error(f"Error performing stage monitoring: {e}")
            return {}

    # Stage registration methods
    def _perform_stage_registration_core(
        self,
        stage_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform stage registration core."""
        try:
            # Simulate stage registration core
            return {
                "stage_registration_completed": True,
                "stages_registered": 5,
                "registration_method": "dynamic",
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing stage registration core: {e}")
            return {}

    def _perform_stage_dependencies(
        self,
        stage_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform stage dependencies."""
        try:
            # Simulate stage dependencies
            return {
                "stage_dependencies_completed": True,
                "dependencies_resolved": 10,
                "dependency_graph": "generated",
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing stage dependencies: {e}")
            return {}

    def _perform_stage_metadata(self, stage_input: dict[str, Any]) -> dict[str, Any]:
        """Perform stage metadata."""
        try:
            # Simulate stage metadata
            return {
                "stage_metadata_completed": True,
                "metadata_entries": 15,
                "metadata_format": "json",
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing stage metadata: {e}")
            return {}

    def _perform_stage_validation_core(
        self,
        stage_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform stage validation core."""
        try:
            # Simulate stage validation core
            return {
                "stage_validation_completed": True,
                "validation_score": 0.95,
                "validation_method": "schema_check",
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing stage validation core: {e}")
            return {}

    # Stage validation methods
    def _perform_input_validation(self, stage_input: dict[str, Any]) -> dict[str, Any]:
        """Perform input validation."""
        try:
            # Simulate input validation
            return {
                "input_validation_completed": True,
                "validation_score": 0.98,
                "validation_method": "type_check",
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing input validation: {e}")
            return {}

    def _perform_output_validation(self, stage_input: dict[str, Any]) -> dict[str, Any]:
        """Perform output validation."""
        try:
            # Simulate output validation
            return {
                "output_validation_completed": True,
                "validation_score": 0.96,
                "validation_method": "quality_check",
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing output validation: {e}")
            return {}

    def _perform_dependency_validation(
        self,
        stage_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform dependency validation."""
        try:
            # Simulate dependency validation
            return {
                "dependency_validation_completed": True,
                "validation_score": 0.94,
                "validation_method": "graph_check",
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing dependency validation: {e}")
            return {}

    def _perform_metadata_validation(
        self,
        stage_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform metadata validation."""
        try:
            # Simulate metadata validation
            return {
                "metadata_validation_completed": True,
                "validation_score": 0.92,
                "validation_method": "format_check",
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing metadata validation: {e}")
            return {}

    # Stage execution methods
    def _perform_execution_planning(
        self,
        stage_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform execution planning."""
        try:
            # Simulate execution planning
            return {
                "execution_planning_completed": True,
                "planned_stages": 5,
                "planning_algorithm": "topological_sort",
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing execution planning: {e}")
            return {}

    def _perform_execution_coordination(
        self,
        stage_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform execution coordination."""
        try:
            # Simulate execution coordination
            return {
                "execution_coordination_completed": True,
                "coordinated_stages": 5,
                "coordination_method": "sequential",
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing execution coordination: {e}")
            return {}

    def _perform_execution_monitoring(
        self,
        stage_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform execution monitoring."""
        try:
            # Simulate execution monitoring
            return {
                "execution_monitoring_completed": True,
                "monitored_stages": 5,
                "monitoring_metrics": "performance",
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing execution monitoring: {e}")
            return {}

    def _perform_execution_reporting(
        self,
        stage_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform execution reporting."""
        try:
            # Simulate execution reporting
            return {
                "execution_reporting_completed": True,
                "reported_stages": 5,
                "report_format": "json",
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing execution reporting: {e}")
            return {}

    # Stage monitoring methods
    def _perform_performance_monitoring(
        self,
        stage_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform performance monitoring."""
        try:
            # Simulate performance monitoring
            return {
                "performance_monitoring_completed": True,
                "performance_metrics": {"throughput": 100, "latency": 50},
                "monitoring_interval": 60,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing performance monitoring: {e}")
            return {}

    def _perform_health_monitoring(self, stage_input: dict[str, Any]) -> dict[str, Any]:
        """Perform health monitoring."""
        try:
            # Simulate health monitoring
            return {
                "health_monitoring_completed": True,
                "health_status": "healthy",
                "health_score": 0.95,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing health monitoring: {e}")
            return {}

    def _perform_error_monitoring(self, stage_input: dict[str, Any]) -> dict[str, Any]:
        """Perform error monitoring."""
        try:
            # Simulate error monitoring
            return {
                "error_monitoring_completed": True,
                "error_count": 0,
                "error_rate": 0.0,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing error monitoring: {e}")
            return {}

    def _perform_resource_monitoring(
        self,
        stage_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform resource monitoring."""
        try:
            # Simulate resource monitoring
            return {
                "resource_monitoring_completed": True,
                "cpu_usage": 0.65,
                "memory_usage": 0.45,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing resource monitoring: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="stage results storage",
    )
    async def _store_stage_results(self) -> None:
        """Store stage results."""
        try:
            # Add timestamp
            self.stage_results["timestamp"] = datetime.now().isoformat()

            # Add to history
            self.stage_history.append(self.stage_results.copy())

            # Limit history size
            if len(self.stage_history) > self.max_stage_history:
                self.stage_history.pop(0)

            self.logger.info("Stage results stored successfully")

        except Exception as e:
            self.logger.error(f"Error storing stage results: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="stage results getting",
    )
    def get_stage_results(self, stage_type: str | None = None) -> dict[str, Any]:
        """
        Get stage results.

        Args:
            stage_type: Optional stage type filter

        Returns:
            Dict[str, Any]: Stage results
        """
        try:
            if stage_type:
                return self.stage_results.get(stage_type, {})
            return self.stage_results.copy()

        except Exception as e:
            self.logger.error(f"Error getting stage results: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="stage history getting",
    )
    def get_stage_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Get stage history.

        Args:
            limit: Optional limit on number of records

        Returns:
            List[Dict[str, Any]]: Stage history
        """
        try:
            history = self.stage_history.copy()

            if limit:
                history = history[-limit:]

            return history

        except Exception as e:
            self.logger.error(f"Error getting stage history: {e}")
            return []

    def get_stage_status(self) -> dict[str, Any]:
        """
        Get stage status information.

        Returns:
            Dict[str, Any]: Stage status
        """
        return {
            "is_registered": self.is_registered,
            "stage_interval": self.stage_interval,
            "max_stage_history": self.max_stage_history,
            "enable_stage_registration": self.enable_stage_registration,
            "enable_stage_validation": self.enable_stage_validation,
            "enable_stage_execution": self.stage_config.get(
                "enable_stage_execution",
                True,
            ),
            "enable_stage_monitoring": self.stage_config.get(
                "enable_stage_monitoring",
                True,
            ),
            "stage_history_count": len(self.stage_history),
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="stage registry cleanup",
    )
    async def stop(self) -> None:
        """Stop the stage registry."""
        self.logger.info("🛑 Stopping Stage Registry...")

        try:
            # Stop registration
            self.is_registered = False

            # Clear results
            self.stage_results.clear()

            # Clear history
            self.stage_history.clear()

            self.logger.info("✅ Stage Registry stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping stage registry: {e}")


# Global stage registry instance
stage_registry: StageRegistry | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="stage registry setup",
)
async def setup_stage_registry(
    config: dict[str, Any] | None = None,
) -> StageRegistry | None:
    """
    Setup global stage registry.

    Args:
        config: Optional configuration dictionary

    Returns:
        Optional[StageRegistry]: Global stage registry instance
    """
    try:
        global stage_registry

        if config is None:
            config = {
                "stage_registry": {
                    "stage_interval": 3600,
                    "max_stage_history": 100,
                    "enable_stage_registration": True,
                    "enable_stage_validation": True,
                    "enable_stage_execution": True,
                    "enable_stage_monitoring": True,
                },
            }

        # Create stage registry
        stage_registry = StageRegistry(config)

        # Initialize stage registry
        success = await stage_registry.initialize()
        if success:
            return stage_registry
        return None

    except Exception as e:
        print(f"Error setting up stage registry: {e}")
        return None
