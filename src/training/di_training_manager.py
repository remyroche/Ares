# src/training/di_training_manager.py

"""
Dependency injection-aware training manager.

This module provides a training manager that uses proper dependency injection
patterns for managing the training pipeline and its components.
"""

from typing import Any

from src.core.dependency_injection import DependencyContainer
from src.core.injectable_base import InjectableBase
from src.interfaces.base_interfaces import IExchangeClient, IStateManager
from src.utils.error_handler import handle_errors
from src.utils.warning_symbols import (
    failed,
    initialization_error,
    invalid,
    missing,
    warning,
)


class DITrainingManager(InjectableBase):
    """
    Dependency injection-aware training manager.

    This training manager uses proper dependency injection patterns
    for creating and managing training pipeline components.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        container: DependencyContainer | None = None,
        state_manager: IStateManager | None = None,
        exchange_client: IExchangeClient | None = None,
    ):
        super().__init__(config)

        self.container = container
        self.state_manager = state_manager
        self.exchange_client = exchange_client

        # Training configuration
        self.training_config = self.config.get("training", {})
        self.training_interval = self.training_config.get(
            "training_interval",
            86400,
        )  # 24 hours
        self.max_training_history = self.training_config.get(
            "max_training_history",
            1000,
        )
        self.enable_model_training = self.training_config.get(
            "enable_model_training",
            True,
        )
        self.enable_hyperparameter_optimization = self.training_config.get(
            "enable_hyperparameter_optimization",
            True,
        )

        # Training components (will be created via DI)
        self.training_steps: dict[str, Any] = {}
        self.training_pipeline: Any = None

        # Training state
        self.is_training = False
        self.training_history: list[dict[str, Any]] = []

    async def initialize(self) -> bool:
        """Initialize the training manager with dependency injection."""
        if not await super().initialize():
            return False

        try:
            # Create training pipeline and steps using DI
            await self._initialize_training_components()

            # Validate training configuration
            if not self._validate_training_configuration():
                return False

            self.logger.info("Training manager initialized successfully")
            return True

        except Exception as e:
            error_msg = f"Failed to initialize training manager: {e}"
            self.logger.exception(error_msg)
            self.print(failed(error_msg))
            return False

    async def _initialize_training_components(self) -> None:
        """Initialize training components using dependency injection."""
        try:
            # Create training pipeline
            if self.container:
                from src.training.core.pipeline_base import TrainingPipeline

                # Register training manager instance
                self.container.register_instance(DITrainingManager, self)

                # Create pipeline with DI
                self.training_pipeline = self.container.resolve(TrainingPipeline)
            else:
                # Fallback to manual creation
                from src.training.core.pipeline_base import TrainingPipeline

                self.training_pipeline = TrainingPipeline(self.training_config)

            # Initialize training steps
            await self._initialize_training_steps()

            self.logger.info("Training components initialized")

        except Exception as e:
            error_msg = f"Failed to initialize training components: {e}"
            self.logger.exception(error_msg)
            self.print(failed(error_msg))
            raise

    async def _initialize_training_steps(self) -> None:
        """Initialize training steps with dependency injection."""
        step_classes = [
            "step1_data_collection",
            "step2_data_validation",
            "step3_feature_engineering",
            "step4_data_preprocessing",
            "step5_model_training",
            "step6_model_validation",
            "step7_hyperparameter_optimization",
            "step8_ensemble_creation",
            "step9_model_evaluation",
            "step10_tactician_ensemble_creation",
            "step11_confidence_calibration",
            "step12_final_parameters_optimization",
        ]

        for step_name in step_classes:
            try:
                # Import step class dynamically
                module_path = f"src.training.steps.{step_name}"
                module = __import__(module_path, fromlist=[step_name])

                # Convert step name to class name (e.g., step1_data_collection -> Step1DataCollection)
                class_name = "".join(
                    [word.capitalize() for word in step_name.split("_")],
                )
                step_class = getattr(module, class_name)

                # Create step instance
                if self.container and self.container.is_registered(step_class):
                    step_instance = self.container.resolve(step_class)
                else:
                    # Create with configuration
                    step_instance = step_class(self.training_config)

                self.training_steps[step_name] = step_instance

            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize training step {step_name}: {e}",
                )

    def _validate_training_configuration(self) -> bool:
        """Validate training configuration."""
        try:
            # Validate training interval
            if self.training_interval <= 0:
                self.print(invalid("Invalid training interval"))
                return False

            # Validate max training history
            if self.max_training_history <= 0:
                self.print(invalid("Invalid max training history"))
                return False

            # Validate required directories exist
            required_dirs = ["models", "data", "checkpoints"]
            for dir_name in required_dirs:
                dir_path = self.training_config.get(f"{dir_name}_directory", dir_name)
                if not dir_path:
                    self.print(missing("Missing {dir_name} directory configuration"))
                    return False

            return True

        except Exception:
            self.print(failed("Configuration validation failed: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="training execution",
    )
    async def run_training_pipeline(
        self,
        symbol: str,
        exchange: str,
        training_type: str = "full",
    ) -> bool:
        """
        Run the complete training pipeline.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            training_type: Type of training (full, incremental, optimization)

        Returns:
            True if training completed successfully
        """
        if self.is_training:
            self.print(warning("Training already in progress"))
            return False

        try:
            self.is_training = True
            self.logger.info(
                f"Starting {training_type} training pipeline for {symbol} on {exchange}",
            )

            # Prepare training context
            training_context = {
                "symbol": symbol,
                "exchange": exchange,
                "training_type": training_type,
                "config": self.training_config,
                "state_manager": self.state_manager,
                "exchange_client": self.exchange_client,
            }

            # Execute training pipeline
            if training_type == "full":
                success = await self._run_full_training_pipeline(training_context)
            elif training_type == "incremental":
                success = await self._run_incremental_training(training_context)
            elif training_type == "optimization":
                success = await self._run_hyperparameter_optimization(training_context)
            else:
                msg = f"Unknown training type: {training_type}"
                raise ValueError(msg)

            # Record training result
            await self._record_training_result(training_context, success)

            self.logger.info(
                f"Training pipeline {'completed' if success else 'failed'}",
            )
            return success

        except Exception as e:
            error_msg = f"Training pipeline failed: {e}"
            self.logger.exception(error_msg)
            self.print(failed(error_msg))
            return False
        finally:
            self.is_training = False

    async def _run_full_training_pipeline(self, context: dict[str, Any]) -> bool:
        """Run the complete training pipeline."""
        try:
            if not self.training_pipeline:
                self.print(initialization_error("Training pipeline not initialized"))
                return False

            # Execute all training steps
            pipeline_steps = [
                "step1_data_collection",
                "step2_data_validation",
                "step3_feature_engineering",
                "step4_data_preprocessing",
                "step5_model_training",
                "step6_model_validation",
                "step7_hyperparameter_optimization",
                "step8_ensemble_creation",
                "step9_model_evaluation",
                "step10_tactician_ensemble_creation",
                "step11_confidence_calibration",
                "step12_final_parameters_optimization",
            ]

            for step_name in pipeline_steps:
                step = self.training_steps.get(step_name)
                if not step:
                    self.print(warning("Training step {step_name} not available"))
                    continue

                self.logger.info(f"Executing {step_name}")

                if hasattr(step, "execute"):
                    success = await step.execute(context)
                else:
                    success = await step.run(context)

                if not success:
                    self.print(failed("Training step {step_name} failed"))
                    return False

                self.logger.info(f"Completed {step_name}")

            return True

        except Exception as e:
            error_msg = f"Full training pipeline failed: {e}"
            self.logger.exception(error_msg)
            self.print(failed(error_msg))
            return False

    async def _run_incremental_training(self, context: dict[str, Any]) -> bool:
        """Run incremental training pipeline."""
        try:
            # Execute subset of steps for incremental training
            incremental_steps = [
                "step1_data_collection",
                "step3_feature_engineering",
                "step5_model_training",
                "step6_model_validation",
                "step9_model_evaluation",
            ]

            for step_name in incremental_steps:
                step = self.training_steps.get(step_name)
                if not step:
                    continue

                self.logger.info(f"Executing incremental {step_name}")

                # Set incremental mode in context
                context["incremental_mode"] = True

                if hasattr(step, "execute"):
                    success = await step.execute(context)
                else:
                    success = await step.run(context)

                if not success:
                    self.print(failed("Incremental step {step_name} failed"))
                    return False

            return True

        except Exception as e:
            error_msg = f"Incremental training failed: {e}"
            self.logger.exception(error_msg)
            self.print(failed(error_msg))
            return False

    async def _run_hyperparameter_optimization(self, context: dict[str, Any]) -> bool:
        """Run hyperparameter optimization."""
        try:
            if not self.enable_hyperparameter_optimization:
                self.logger.info("Hyperparameter optimization disabled")
                return True

            # Execute optimization steps
            optimization_steps = [
                "step7_hyperparameter_optimization",
                "step12_final_parameters_optimization",
            ]

            for step_name in optimization_steps:
                step = self.training_steps.get(step_name)
                if not step:
                    continue

                self.logger.info(f"Executing optimization {step_name}")

                if hasattr(step, "execute"):
                    success = await step.execute(context)
                else:
                    success = await step.run(context)

                if not success:
                    self.print(failed("Optimization step {step_name} failed"))
                    return False

            return True

        except Exception as e:
            error_msg = f"Hyperparameter optimization failed: {e}"
            self.logger.exception(error_msg)
            self.print(failed(error_msg))
            return False

    async def _record_training_result(
        self,
        context: dict[str, Any],
        success: bool,
    ) -> None:
        """Record training result in history."""
        try:
            result = {
                "timestamp": context.get("timestamp"),
                "symbol": context.get("symbol"),
                "exchange": context.get("exchange"),
                "training_type": context.get("training_type"),
                "success": success,
                "duration": context.get("duration", 0),
            }

            self.training_history.append(result)

            # Limit history size
            if len(self.training_history) > self.max_training_history:
                self.training_history = self.training_history[
                    -self.max_training_history :
                ]

            # Store in state manager
            if self.state_manager:
                self.state_manager.set_state("last_training_result", result)
                self.state_manager.set_state("training_history", self.training_history)

        except Exception as e:
            error_msg = f"Failed to record training result: {e}"
            self.logger.exception(error_msg)
            self.print(failed(error_msg))

    async def get_training_status(self) -> dict[str, Any]:
        """Get current training status."""
        return {
            "is_training": self.is_training,
            "is_initialized": self.is_initialized,
            "training_steps_available": list(self.training_steps.keys()),
            "last_training_result": (
                self.training_history[-1] if self.training_history else None
            ),
            "training_history_count": len(self.training_history),
            "configuration": {
                "training_interval": self.training_interval,
                "enable_model_training": self.enable_model_training,
                "enable_hyperparameter_optimization": self.enable_hyperparameter_optimization,
            },
        }

    async def stop_training(self) -> None:
        """Stop any running training operations."""
        if self.is_training:
            self.logger.info("Stopping training operations")
            # Implementation would depend on how training steps handle cancellation
            # For now, we just set the flag
            self.is_training = False

    async def shutdown(self) -> None:
        """Shutdown the training manager."""
        await self.stop_training()
        await super().shutdown()
