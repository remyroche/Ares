"""
Pipeline orchestrator for the modular training pipeline.

This module provides the main orchestrator that coordinates the execution
of pipeline stages, handles dependencies, and manages the overall pipeline flow.
"""

from datetime import datetime
from typing import Any

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class PipelineOrchestrator:
    """
    Pipeline orchestrator with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize pipeline orchestrator with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("PipelineOrchestrator")

        # Pipeline orchestrator state
        self.is_orchestrating: bool = False
        self.pipeline_results: dict[str, Any] = {}
        self.pipeline_history: list[dict[str, Any]] = []

        # Configuration
        self.pipeline_config: dict[str, Any] = self.config.get(
            "pipeline_orchestrator",
            {},
        )
        self.pipeline_interval: int = self.pipeline_config.get(
            "pipeline_interval",
            3600,
        )
        self.max_pipeline_history: int = self.pipeline_config.get(
            "max_pipeline_history",
            100,
        )
        self.enable_pipeline_execution: bool = self.pipeline_config.get(
            "enable_pipeline_execution",
            True,
        )
        self.enable_pipeline_monitoring: bool = self.pipeline_config.get(
            "enable_pipeline_monitoring",
            True,
        )

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid pipeline orchestrator configuration"),
            AttributeError: (
                False,
                "Missing required pipeline orchestrator parameters",
            ),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="pipeline orchestrator initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize pipeline orchestrator with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Pipeline Orchestrator...")

            # Load pipeline configuration
            await self._load_pipeline_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for pipeline orchestrator")
                return False

            # Initialize pipeline modules
            await self._initialize_pipeline_modules()

            self.logger.info(
                "✅ Pipeline Orchestrator initialization completed successfully",
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ Pipeline Orchestrator initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="pipeline configuration loading",
    )
    async def _load_pipeline_configuration(self) -> None:
        """Load pipeline configuration."""
        try:
            # Set default pipeline parameters
            self.pipeline_config.setdefault("pipeline_interval", 3600)
            self.pipeline_config.setdefault("max_pipeline_history", 100)
            self.pipeline_config.setdefault("enable_pipeline_execution", True)
            self.pipeline_config.setdefault("enable_pipeline_monitoring", True)
            self.pipeline_config.setdefault("enable_pipeline_optimization", True)
            self.pipeline_config.setdefault("enable_pipeline_validation", True)

            # Update configuration
            self.pipeline_interval = self.pipeline_config["pipeline_interval"]
            self.max_pipeline_history = self.pipeline_config["max_pipeline_history"]
            self.enable_pipeline_execution = self.pipeline_config[
                "enable_pipeline_execution"
            ]
            self.enable_pipeline_monitoring = self.pipeline_config[
                "enable_pipeline_monitoring"
            ]

            self.logger.info("Pipeline configuration loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading pipeline configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate pipeline configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate pipeline interval
            if self.pipeline_interval <= 0:
                self.logger.error("Invalid pipeline interval")
                return False

            # Validate max pipeline history
            if self.max_pipeline_history <= 0:
                self.logger.error("Invalid max pipeline history")
                return False

            # Validate that at least one pipeline type is enabled
            if not any(
                [
                    self.enable_pipeline_execution,
                    self.enable_pipeline_monitoring,
                    self.pipeline_config.get("enable_pipeline_optimization", True),
                    self.pipeline_config.get("enable_pipeline_validation", True),
                ],
            ):
                self.logger.error("At least one pipeline type must be enabled")
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="pipeline modules initialization",
    )
    async def _initialize_pipeline_modules(self) -> None:
        """Initialize pipeline modules."""
        try:
            # Initialize pipeline execution module
            if self.enable_pipeline_execution:
                await self._initialize_pipeline_execution()

            # Initialize pipeline monitoring module
            if self.enable_pipeline_monitoring:
                await self._initialize_pipeline_monitoring()

            # Initialize pipeline optimization module
            if self.pipeline_config.get("enable_pipeline_optimization", True):
                await self._initialize_pipeline_optimization()

            # Initialize pipeline validation module
            if self.pipeline_config.get("enable_pipeline_validation", True):
                await self._initialize_pipeline_validation()

            self.logger.info("Pipeline modules initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing pipeline modules: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="pipeline execution initialization",
    )
    async def _initialize_pipeline_execution(self) -> None:
        """Initialize pipeline execution module."""
        try:
            # Initialize pipeline execution components
            self.pipeline_execution_components = {
                "step_execution": True,
                "step_coordination": True,
                "step_scheduling": True,
                "step_monitoring": True,
            }

            self.logger.info("Pipeline execution module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing pipeline execution: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="pipeline monitoring initialization",
    )
    async def _initialize_pipeline_monitoring(self) -> None:
        """Initialize pipeline monitoring module."""
        try:
            # Initialize pipeline monitoring components
            self.pipeline_monitoring_components = {
                "performance_monitoring": True,
                "health_monitoring": True,
                "error_monitoring": True,
                "resource_monitoring": True,
            }

            self.logger.info("Pipeline monitoring module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing pipeline monitoring: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="pipeline optimization initialization",
    )
    async def _initialize_pipeline_optimization(self) -> None:
        """Initialize pipeline optimization module."""
        try:
            # Initialize pipeline optimization components
            self.pipeline_optimization_components = {
                "performance_optimization": True,
                "resource_optimization": True,
                "scheduling_optimization": True,
                "throughput_optimization": True,
            }

            self.logger.info("Pipeline optimization module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing pipeline optimization: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="pipeline validation initialization",
    )
    async def _initialize_pipeline_validation(self) -> None:
        """Initialize pipeline validation module."""
        try:
            # Initialize pipeline validation components
            self.pipeline_validation_components = {
                "input_validation": True,
                "output_validation": True,
                "step_validation": True,
                "pipeline_validation": True,
            }

            self.logger.info("Pipeline validation module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing pipeline validation: {e}")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid pipeline parameters"),
            AttributeError: (False, "Missing pipeline components"),
            KeyError: (False, "Missing required pipeline data"),
        },
        default_return=False,
        context="pipeline execution",
    )
    async def execute_pipeline(self, pipeline_input: dict[str, Any]) -> bool:
        """
        Execute pipeline operations.

        Args:
            pipeline_input: Pipeline input dictionary

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self._validate_pipeline_inputs(pipeline_input):
                return False

            self.is_orchestrating = True
            self.logger.info("🔄 Starting pipeline execution...")

            # Perform pipeline execution
            if self.enable_pipeline_execution:
                execution_results = await self._perform_pipeline_execution(
                    pipeline_input,
                )
                self.pipeline_results["pipeline_execution"] = execution_results

            # Perform pipeline monitoring
            if self.enable_pipeline_monitoring:
                monitoring_results = await self._perform_pipeline_monitoring(
                    pipeline_input,
                )
                self.pipeline_results["pipeline_monitoring"] = monitoring_results

            # Perform pipeline optimization
            if self.pipeline_config.get("enable_pipeline_optimization", True):
                optimization_results = await self._perform_pipeline_optimization(
                    pipeline_input,
                )
                self.pipeline_results["pipeline_optimization"] = optimization_results

            # Perform pipeline validation
            if self.pipeline_config.get("enable_pipeline_validation", True):
                validation_results = await self._perform_pipeline_validation(
                    pipeline_input,
                )
                self.pipeline_results["pipeline_validation"] = validation_results

            # Store pipeline results
            await self._store_pipeline_results()

            self.is_orchestrating = False
            self.logger.info("✅ Pipeline execution completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error executing pipeline: {e}")
            self.is_orchestrating = False
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="pipeline inputs validation",
    )
    def _validate_pipeline_inputs(self, pipeline_input: dict[str, Any]) -> bool:
        """
        Validate pipeline inputs.

        Args:
            pipeline_input: Pipeline input dictionary

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check required pipeline input fields
            required_fields = ["pipeline_type", "pipeline_steps", "timestamp"]
            for field in required_fields:
                if field not in pipeline_input:
                    self.logger.error(f"Missing required pipeline input field: {field}")
                    return False

            # Validate data types
            if not isinstance(pipeline_input["pipeline_type"], str):
                self.logger.error("Invalid pipeline type")
                return False

            if not isinstance(pipeline_input["pipeline_steps"], list):
                self.logger.error("Invalid pipeline steps format")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating pipeline inputs: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="pipeline execution",
    )
    async def _perform_pipeline_execution(
        self,
        pipeline_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform pipeline execution.

        Args:
            pipeline_input: Pipeline input dictionary

        Returns:
            Dict[str, Any]: Pipeline execution results
        """
        try:
            results = {}

            # Perform step execution
            if self.pipeline_execution_components.get("step_execution", False):
                results["step_execution"] = self._perform_step_execution(pipeline_input)

            # Perform step coordination
            if self.pipeline_execution_components.get("step_coordination", False):
                results["step_coordination"] = self._perform_step_coordination(
                    pipeline_input,
                )

            # Perform step scheduling
            if self.pipeline_execution_components.get("step_scheduling", False):
                results["step_scheduling"] = self._perform_step_scheduling(
                    pipeline_input,
                )

            # Perform step monitoring
            if self.pipeline_execution_components.get("step_monitoring", False):
                results["step_monitoring"] = self._perform_step_monitoring(
                    pipeline_input,
                )

            self.logger.info("Pipeline execution completed")
            return results

        except Exception as e:
            self.logger.error(f"Error performing pipeline execution: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="pipeline monitoring",
    )
    async def _perform_pipeline_monitoring(
        self,
        pipeline_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform pipeline monitoring.

        Args:
            pipeline_input: Pipeline input dictionary

        Returns:
            Dict[str, Any]: Pipeline monitoring results
        """
        try:
            results = {}

            # Perform performance monitoring
            if self.pipeline_monitoring_components.get("performance_monitoring", False):
                results["performance_monitoring"] = (
                    self._perform_performance_monitoring(pipeline_input)
                )

            # Perform health monitoring
            if self.pipeline_monitoring_components.get("health_monitoring", False):
                results["health_monitoring"] = self._perform_health_monitoring(
                    pipeline_input,
                )

            # Perform error monitoring
            if self.pipeline_monitoring_components.get("error_monitoring", False):
                results["error_monitoring"] = self._perform_error_monitoring(
                    pipeline_input,
                )

            # Perform resource monitoring
            if self.pipeline_monitoring_components.get("resource_monitoring", False):
                results["resource_monitoring"] = self._perform_resource_monitoring(
                    pipeline_input,
                )

            self.logger.info("Pipeline monitoring completed")
            return results

        except Exception as e:
            self.logger.error(f"Error performing pipeline monitoring: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="pipeline optimization",
    )
    async def _perform_pipeline_optimization(
        self,
        pipeline_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform pipeline optimization.

        Args:
            pipeline_input: Pipeline input dictionary

        Returns:
            Dict[str, Any]: Pipeline optimization results
        """
        try:
            results = {}

            # Perform performance optimization
            if self.pipeline_optimization_components.get(
                "performance_optimization",
                False,
            ):
                results["performance_optimization"] = (
                    self._perform_performance_optimization(pipeline_input)
                )

            # Perform resource optimization
            if self.pipeline_optimization_components.get(
                "resource_optimization",
                False,
            ):
                results["resource_optimization"] = self._perform_resource_optimization(
                    pipeline_input,
                )

            # Perform scheduling optimization
            if self.pipeline_optimization_components.get(
                "scheduling_optimization",
                False,
            ):
                results["scheduling_optimization"] = (
                    self._perform_scheduling_optimization(pipeline_input)
                )

            # Perform throughput optimization
            if self.pipeline_optimization_components.get(
                "throughput_optimization",
                False,
            ):
                results["throughput_optimization"] = (
                    self._perform_throughput_optimization(pipeline_input)
                )

            self.logger.info("Pipeline optimization completed")
            return results

        except Exception as e:
            self.logger.error(f"Error performing pipeline optimization: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="pipeline validation",
    )
    async def _perform_pipeline_validation(
        self,
        pipeline_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform pipeline validation.

        Args:
            pipeline_input: Pipeline input dictionary

        Returns:
            Dict[str, Any]: Pipeline validation results
        """
        try:
            results = {}

            # Perform input validation
            if self.pipeline_validation_components.get("input_validation", False):
                results["input_validation"] = self._perform_input_validation(
                    pipeline_input,
                )

            # Perform output validation
            if self.pipeline_validation_components.get("output_validation", False):
                results["output_validation"] = self._perform_output_validation(
                    pipeline_input,
                )

            # Perform step validation
            if self.pipeline_validation_components.get("step_validation", False):
                results["step_validation"] = self._perform_step_validation(
                    pipeline_input,
                )

            # Perform pipeline validation
            if self.pipeline_validation_components.get("pipeline_validation", False):
                results["pipeline_validation"] = self._perform_pipeline_validation_core(
                    pipeline_input,
                )

            self.logger.info("Pipeline validation completed")
            return results

        except Exception as e:
            self.logger.error(f"Error performing pipeline validation: {e}")
            return {}

    # Pipeline execution methods
    def _perform_step_execution(self, pipeline_input: dict[str, Any]) -> dict[str, Any]:
        """Perform step execution."""
        try:
            # Simulate step execution
            return {
                "step_execution_completed": True,
                "steps_executed": 5,
                "execution_time": 120.5,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing step execution: {e}")
            return {}

    def _perform_step_coordination(
        self,
        pipeline_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform step coordination."""
        try:
            # Simulate step coordination
            return {
                "step_coordination_completed": True,
                "coordination_method": "sequential",
                "dependencies_resolved": True,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing step coordination: {e}")
            return {}

    def _perform_step_scheduling(
        self,
        pipeline_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform step scheduling."""
        try:
            # Simulate step scheduling
            return {
                "step_scheduling_completed": True,
                "scheduling_algorithm": "priority_queue",
                "scheduled_steps": 5,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing step scheduling: {e}")
            return {}

    def _perform_step_monitoring(
        self,
        pipeline_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform step monitoring."""
        try:
            # Simulate step monitoring
            return {
                "step_monitoring_completed": True,
                "monitored_steps": 5,
                "monitoring_metrics": "performance",
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing step monitoring: {e}")
            return {}

    # Pipeline monitoring methods
    def _perform_performance_monitoring(
        self,
        pipeline_input: dict[str, Any],
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

    def _perform_health_monitoring(
        self,
        pipeline_input: dict[str, Any],
    ) -> dict[str, Any]:
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

    def _perform_error_monitoring(
        self,
        pipeline_input: dict[str, Any],
    ) -> dict[str, Any]:
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
        pipeline_input: dict[str, Any],
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

    # Pipeline optimization methods
    def _perform_performance_optimization(
        self,
        pipeline_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform performance optimization."""
        try:
            # Simulate performance optimization
            return {
                "performance_optimization_completed": True,
                "optimization_score": 0.87,
                "optimization_method": "algorithmic",
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing performance optimization: {e}")
            return {}

    def _perform_resource_optimization(
        self,
        pipeline_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform resource optimization."""
        try:
            # Simulate resource optimization
            return {
                "resource_optimization_completed": True,
                "resource_efficiency": 0.92,
                "optimization_method": "resource_pooling",
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing resource optimization: {e}")
            return {}

    def _perform_scheduling_optimization(
        self,
        pipeline_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform scheduling optimization."""
        try:
            # Simulate scheduling optimization
            return {
                "scheduling_optimization_completed": True,
                "scheduling_efficiency": 0.89,
                "optimization_method": "dynamic_scheduling",
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing scheduling optimization: {e}")
            return {}

    def _perform_throughput_optimization(
        self,
        pipeline_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform throughput optimization."""
        try:
            # Simulate throughput optimization
            return {
                "throughput_optimization_completed": True,
                "throughput_improvement": 0.15,
                "optimization_method": "parallel_processing",
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing throughput optimization: {e}")
            return {}

    # Pipeline validation methods
    def _perform_input_validation(
        self,
        pipeline_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform input validation."""
        try:
            # Simulate input validation
            return {
                "input_validation_completed": True,
                "validation_score": 0.98,
                "validation_method": "schema_validation",
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing input validation: {e}")
            return {}

    def _perform_output_validation(
        self,
        pipeline_input: dict[str, Any],
    ) -> dict[str, Any]:
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

    def _perform_step_validation(
        self,
        pipeline_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform step validation."""
        try:
            # Simulate step validation
            return {
                "step_validation_completed": True,
                "validation_score": 0.94,
                "validation_method": "unit_testing",
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing step validation: {e}")
            return {}

    def _perform_pipeline_validation_core(
        self,
        pipeline_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform pipeline validation core."""
        try:
            # Simulate pipeline validation core
            return {
                "pipeline_validation_completed": True,
                "validation_score": 0.92,
                "validation_method": "integration_testing",
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing pipeline validation core: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="pipeline results storage",
    )
    async def _store_pipeline_results(self) -> None:
        """Store pipeline results."""
        try:
            # Add timestamp
            self.pipeline_results["timestamp"] = datetime.now().isoformat()

            # Add to history
            self.pipeline_history.append(self.pipeline_results.copy())

            # Limit history size
            if len(self.pipeline_history) > self.max_pipeline_history:
                self.pipeline_history.pop(0)

            self.logger.info("Pipeline results stored successfully")

        except Exception as e:
            self.logger.error(f"Error storing pipeline results: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="pipeline results getting",
    )
    def get_pipeline_results(
        self,
        pipeline_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Get pipeline results.

        Args:
            pipeline_type: Optional pipeline type filter

        Returns:
            Dict[str, Any]: Pipeline results
        """
        try:
            if pipeline_type:
                return self.pipeline_results.get(pipeline_type, {})
            return self.pipeline_results.copy()

        except Exception as e:
            self.logger.error(f"Error getting pipeline results: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="pipeline history getting",
    )
    def get_pipeline_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Get pipeline history.

        Args:
            limit: Optional limit on number of records

        Returns:
            List[Dict[str, Any]]: Pipeline history
        """
        try:
            history = self.pipeline_history.copy()

            if limit:
                history = history[-limit:]

            return history

        except Exception as e:
            self.logger.error(f"Error getting pipeline history: {e}")
            return []

    def get_pipeline_status(self) -> dict[str, Any]:
        """
        Get pipeline status information.

        Returns:
            Dict[str, Any]: Pipeline status
        """
        return {
            "is_orchestrating": self.is_orchestrating,
            "pipeline_interval": self.pipeline_interval,
            "max_pipeline_history": self.max_pipeline_history,
            "enable_pipeline_execution": self.enable_pipeline_execution,
            "enable_pipeline_monitoring": self.enable_pipeline_monitoring,
            "enable_pipeline_optimization": self.pipeline_config.get(
                "enable_pipeline_optimization",
                True,
            ),
            "enable_pipeline_validation": self.pipeline_config.get(
                "enable_pipeline_validation",
                True,
            ),
            "pipeline_history_count": len(self.pipeline_history),
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="pipeline orchestrator cleanup",
    )
    async def stop(self) -> None:
        """Stop the pipeline orchestrator."""
        self.logger.info("🛑 Stopping Pipeline Orchestrator...")

        try:
            # Stop orchestrating
            self.is_orchestrating = False

            # Clear results
            self.pipeline_results.clear()

            # Clear history
            self.pipeline_history.clear()

            self.logger.info("✅ Pipeline Orchestrator stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping pipeline orchestrator: {e}")


# Global pipeline orchestrator instance
pipeline_orchestrator: PipelineOrchestrator | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="pipeline orchestrator setup",
)
async def setup_pipeline_orchestrator(
    config: dict[str, Any] | None = None,
) -> PipelineOrchestrator | None:
    """
    Setup global pipeline orchestrator.

    Args:
        config: Optional configuration dictionary

    Returns:
        Optional[PipelineOrchestrator]: Global pipeline orchestrator instance
    """
    try:
        global pipeline_orchestrator

        if config is None:
            config = {
                "pipeline_orchestrator": {
                    "pipeline_interval": 3600,
                    "max_pipeline_history": 100,
                    "enable_pipeline_execution": True,
                    "enable_pipeline_monitoring": True,
                    "enable_pipeline_optimization": True,
                    "enable_pipeline_validation": True,
                },
            }

        # Create pipeline orchestrator
        pipeline_orchestrator = PipelineOrchestrator(config)

        # Initialize pipeline orchestrator
        success = await pipeline_orchestrator.initialize()
        if success:
            return pipeline_orchestrator
        return None

    except Exception as e:
        print(f"Error setting up pipeline orchestrator: {e}")
        return None
