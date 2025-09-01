# src/training/training_orchestrator.py

from datetime import datetime
from typing import Any

from src.utils.error_handler import (
    handle_errors, handle_specific_errors)
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    failed,
    invalid, missing)


class TrainingOrchestrator:
    """Training orchestrator responsible for coordinating the overall training pipeline.
    This module handles the high-level coordination between different training components.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize training orchestrator.

        Args:
            config: Configuration dictionary

        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("TrainingOrchestrator")

        # Training state
        self.is_training: bool = False
        self.training_start_time: datetime | None = None
        self.training_results: dict[str, Any] = {}

        # Component managers (will be initialized)
        self.model_trainer = None
        self.optimization_manager = None
        self.ensemble_manager = None
        self.calibration_manager = None

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid training orchestrator configuration"),
            AttributeError: (False, "Missing required training components"),
            KeyError: (False, "Missing configuration keys")},
        default_return=False, context="training orchestrator initialization")
    async def initialize(self) -> bool:
        """Initialize training orchestrator and all component managers.

        Returns:
            bool: True if initialization successful, False otherwise

        """
        try:
            self.logger.info("Initializing Training Orchestrator...")

            # Initialize component managers
            await self._initialize_component_managers()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error(invalid("Invalid configuration for training orchestrator"))
                return False

            # Initialize validation framework
            await self._initialize_validation_framework()

            self.logger.info("✅ Training Orchestrator initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Training Orchestrator initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError), default_return=None,
        context="validation framework initialization")
    async def _initialize_validation_framework(self) -> None:
        """Initialize the validation framework components."""
        try:
            self.logger.info("Initializing validation framework...")

            # Initialize step dependency validator
            from src.utils.step_dependency_validator import StepDependencyValidator
            self.step_dependency_validator = StepDependencyValidator()

            # Initialize validator orchestrator
            from src.utils.validator_orchestrator import validator_orchestrator
            self.validator_orchestrator = validator_orchestrator

            # Initialize pipeline validator
            from src.utils.pipeline_validator import PipelineValidator
            self.pipeline_validator = PipelineValidator()

            self.logger.info("Validation framework initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing validation framework: {e}")
            raise

    @handle_errors(
        exceptions=(ValueError, AttributeError), default_return=None,
        context="component managers initialization")
    async def _initialize_component_managers(self) -> None:
        """Initialize all component managers."""
        try:
            self.logger.info("Initializing component managers...")

            # Initialize model trainer
            from src.training.model_trainer import ModelTrainer
            self.model_trainer = ModelTrainer(self.config)
            await self.model_trainer.initialize()

            # Initialize optimization manager
            from src.training.optimization_manager import OptimizationManager
            self.optimization_manager = OptimizationManager(self.config)
            await self.optimization_manager.initialize()

            # Initialize ensemble manager
            from src.training.ensemble_manager import EnsembleManager
            self.ensemble_manager = EnsembleManager(self.config)
            await self.ensemble_manager.initialize()

            # Initialize calibration manager
            from src.training.calibration_manager import CalibrationManager
            self.calibration_manager = CalibrationManager(self.config)
            await self.calibration_manager.initialize()

            self.logger.info("Component managers initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing component managers: {e}")
            raise

    def _validate_configuration(self) -> bool:
        """Validate training orchestrator configuration."""
        try:
            self.logger.info("Validating training orchestrator configuration...")

            # Check required configuration sections
            required_sections = ["training_orchestrator", "model_trainer", "optimization_manager"]
            missing_sections = [section for section in required_sections if section not in self.config]

            if missing_sections:
                self.logger.error(f"Missing required configuration sections: {missing_sections}")
                return False

            # Validate orchestrator configuration
            orchestrator_config = self.config.get("training_orchestrator", {})
            required_orchestrator_keys = ["max_concurrent_training", "training_timeout"]
            missing_keys = [key for key in required_orchestrator_keys if key not in orchestrator_config]

            if missing_keys:
                self.logger.error(f"Missing required orchestrator configuration keys: {missing_keys}")
                return False

            # Validate timeout values
            if orchestrator_config.get("training_timeout", 0) <= 0:
                self.logger.error("Training timeout must be positive")
                return False

            if orchestrator_config.get("max_concurrent_training", 0) <= 0:
                self.logger.error("Max concurrent training must be positive")
                return False

            self.logger.info("Training pipeline validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,), default_return=False,
        context="training pipeline execution")
    async def execute_training_pipeline(self, training_config: dict[str, Any]) -> bool:
        """Execute the complete training pipeline.

        Args:
            training_config: Training configuration

        Returns:
            bool: True if pipeline executed successfully, False otherwise

        """
        try:
            if self.is_training:
                self.logger.warning("Training pipeline is already running")
                return False

            self.logger.info("Starting training pipeline execution...")

            # Validate training configuration
            if not self._validate_training_config(training_config):
                self.logger.error("Invalid training configuration")
                return False

            # Set training state
            self.is_training = True
            self.training_start_time = datetime.now()

            # Execute pipeline steps
            pipeline_result = await self._execute_pipeline_steps(training_config)

            # Record results
            self.training_results = pipeline_result

            self.logger.info("Training pipeline execution completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error during training pipeline execution: {e}")
            self.is_training = False
            return False

    def _validate_training_config(self, training_config: dict[str, Any]) -> bool:
        """Validate training configuration."""
        try:
            self.logger.debug("Validating training configuration...")

            # Check required fields
            required_fields = ["model_type", "data_source", "training_params"]
            missing_fields = [field for field in required_fields if field not in training_config]

            if missing_fields:
                self.logger.error(f"Missing required training configuration fields: {missing_fields}")
                return False

            # Validate model type
            valid_model_types = ["regression", "classification", "ensemble"]
            if training_config.get("model_type") not in valid_model_types:
                self.logger.error(f"Invalid model type. Must be one of: {valid_model_types}")
                return False

            # Validate training parameters
            training_params = training_config.get("training_params", {})
            if not isinstance(training_params, dict):
                self.logger.error("Training parameters must be a dictionary")
                return False

            self.logger.debug("Training configuration validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Error validating training configuration: {e}")
            return False

    async def _execute_pipeline_steps(self, training_config: dict[str, Any]) -> dict[str, Any]:
        """Execute the pipeline steps."""
        try:
            self.logger.info("Executing pipeline steps...")

            pipeline_result = {
                "start_time": self.training_start_time.isoformat(),
                "end_time": None,
                "steps": {},
                "overall_success": True
            }

            # Step 1: Data preparation
            data_result = await self._prepare_training_data(training_config)
            pipeline_result["steps"]["data_preparation"] = data_result

            if not data_result.get("success", False):
                pipeline_result["overall_success"] = False
                self.logger.error("Data preparation failed")
                return pipeline_result

            # Step 2: Model training
            training_result = await self._train_model(training_config, data_result)
            pipeline_result["steps"]["model_training"] = training_result

            if not training_result.get("success", False):
                pipeline_result["overall_success"] = False
                self.logger.error("Model training failed")
                return pipeline_result

            # Step 3: Model evaluation
            evaluation_result = await self._evaluate_model(training_result)
            pipeline_result["steps"]["model_evaluation"] = evaluation_result

            # Step 4: Model optimization (if enabled)
            if training_config.get("enable_optimization", True):
                optimization_result = await self._optimize_model(training_result, evaluation_result)
                pipeline_result["steps"]["model_optimization"] = optimization_result

            # Step 5: Model calibration (if enabled)
            if training_config.get("enable_calibration", True):
                calibration_result = await self._calibrate_model(training_result)
                pipeline_result["steps"]["model_calibration"] = calibration_result

            # Step 6: Ensemble creation (if enabled)
            if training_config.get("enable_ensemble", False):
                ensemble_result = await self._create_ensemble(training_result, evaluation_result)
                pipeline_result["steps"]["ensemble_creation"] = ensemble_result

            pipeline_result["end_time"] = datetime.now().isoformat()
            self.logger.info("Pipeline steps execution completed")
            return pipeline_result

        except Exception as e:
            self.logger.error(f"Error executing pipeline steps: {e}")
            return {"error": str(e), "overall_success": False}

    async def _prepare_training_data(self, training_config: dict[str, Any]) -> dict[str, Any]:
        """Prepare training data."""
        try:
            self.logger.info("Preparing training data...")

            data_source = training_config.get("data_source")
            data_params = training_config.get("data_params", {})

            # Use data manager to prepare data
            from src.training.data_manager import DataManager
            data_manager = DataManager(self.config)
            await data_manager.initialize()

            data_result = await data_manager.prepare_training_data(data_source, data_params)
            await data_manager.cleanup()

            return {
                "success": True,
                "data_shape": data_result.get("shape", "unknown"),
                "features_count": data_result.get("features_count", 0),
                "samples_count": data_result.get("samples_count", 0)
            }

        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            return {"success": False, "error": str(e)}

    async def _train_model(self, training_config: dict[str, Any], data_result: dict[str, Any]) -> dict[str, Any]:
        """Train the model."""
        try:
            self.logger.info("Training model...")

            if not self.model_trainer:
                raise ValueError("Model trainer not initialized")

            training_params = training_config.get("training_params", {})
            model_type = training_config.get("model_type")

            training_result = await self.model_trainer.train_model(
                model_type=model_type,
                training_params=training_params,
                data_info=data_result
            )

            return {
                "success": True,
                "model_path": training_result.get("model_path"),
                "training_metrics": training_result.get("metrics", {}),
                "training_time": training_result.get("training_time")
            }

        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            return {"success": False, "error": str(e)}

    async def _evaluate_model(self, training_result: dict[str, Any]) -> dict[str, Any]:
        """Evaluate the trained model."""
        try:
            self.logger.info("Evaluating model...")

            if not self.model_trainer:
                raise ValueError("Model trainer not initialized")

            evaluation_result = await self.model_trainer.evaluate_model(
                model_path=training_result.get("model_path")
            )

            return {
                "success": True,
                "evaluation_metrics": evaluation_result.get("metrics", {}),
                "evaluation_time": evaluation_result.get("evaluation_time")
            }

        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            return {"success": False, "error": str(e)}

    async def _optimize_model(self, training_result: dict[str, Any], evaluation_result: dict[str, Any]) -> dict[str, Any]:
        """Optimize the model."""
        try:
            self.logger.info("Optimizing model...")

            if not self.optimization_manager:
                raise ValueError("Optimization manager not initialized")

            optimization_result = await self.optimization_manager.optimize_model(
                model_path=training_result.get("model_path"),
                current_metrics=evaluation_result.get("evaluation_metrics", {})
            )

            return {
                "success": True,
                "optimization_metrics": optimization_result.get("metrics", {}),
                "optimization_time": optimization_result.get("optimization_time")
            }

        except Exception as e:
            self.logger.error(f"Error optimizing model: {e}")
            return {"success": False, "error": str(e)}

    async def _calibrate_model(self, training_result: dict[str, Any]) -> dict[str, Any]:
        """Calibrate the model."""
        try:
            self.logger.info("Calibrating model...")

            if not self.calibration_manager:
                raise ValueError("Calibration manager not initialized")

            calibration_result = await self.calibration_manager.calibrate_model(
                model_path=training_result.get("model_path")
            )

            return {
                "success": True,
                "calibration_metrics": calibration_result.get("metrics", {}),
                "calibration_time": calibration_result.get("calibration_time")
            }

        except Exception as e:
            self.logger.error(f"Error calibrating model: {e}")
            return {"success": False, "error": str(e)}

    async def _create_ensemble(self, training_result: dict[str, Any], evaluation_result: dict[str, Any]) -> dict[str, Any]:
        """Create ensemble model."""
        try:
            self.logger.info("Creating ensemble...")

            if not self.ensemble_manager:
                raise ValueError("Ensemble manager not initialized")

            ensemble_result = await self.ensemble_manager.create_ensemble(
                base_model_path=training_result.get("model_path"),
                evaluation_metrics=evaluation_result.get("evaluation_metrics", {})
            )

            return {
                "success": True,
                "ensemble_path": ensemble_result.get("ensemble_path"),
                "ensemble_metrics": ensemble_result.get("metrics", {}),
                "ensemble_time": ensemble_result.get("ensemble_time")
            }

        except Exception as e:
            self.logger.error(f"Error creating ensemble: {e}")
            return {"success": False, "error": str(e)}

    @handle_errors(
        exceptions=(Exception,), default_return=False,
        context="training pipeline stop")
    async def stop_training_pipeline(self) -> bool:
        """Stop the training pipeline."""
        try:
            if not self.is_training:
                self.logger.warning("No training pipeline running to stop")
                return False

            self.logger.info("Stopping training pipeline...")

            # Stop component managers
            await self._stop_component_managers()

            # Update state
            self.is_training = False

            self.logger.info("Training pipeline stopped successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error stopping training pipeline: {e}")
            return False

    async def _stop_component_managers(self) -> None:
        """Stop all component managers."""
        try:
            self.logger.debug("Stopping component managers...")

            # Stop model trainer
            if self.model_trainer:
                await self.model_trainer.cleanup()

            # Stop optimization manager
            if self.optimization_manager:
                await self.optimization_manager.cleanup()

            # Stop ensemble manager
            if self.ensemble_manager:
                await self.ensemble_manager.cleanup()

            # Stop calibration manager
            if self.calibration_manager:
                await self.calibration_manager.cleanup()

            self.logger.debug("Component managers stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping component managers: {e}")

    @handle_errors(
        exceptions=(Exception,), default_return={},
        context="training results retrieval")
    def get_training_results(self) -> dict[str, Any]:
        """Get training results."""
        try:
            return self.training_results.copy()
        except Exception as e:
            self.logger.error(f"Error getting training results: {e}")
            return {}

    @handle_errors(
        exceptions=(Exception,), default_return=False,
        context="training status check")
    def is_pipeline_running(self) -> bool:
        """Check if training pipeline is running."""
        try:
            return self.is_training
        except Exception as e:
            self.logger.error(f"Error checking pipeline status: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,), default_return=None,
        context="training orchestrator cleanup")
    async def cleanup(self) -> None:
        """Cleanup training orchestrator resources."""
        try:
            self.logger.info("Cleaning up Training Orchestrator...")

            # Stop training if running
            if self.is_training:
                await self.stop_training_pipeline()

            # Cleanup component managers
            await self._stop_component_managers()

            # Clear results
            self.training_results.clear()

            self.logger.info("Training Orchestrator cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get training orchestrator status."""
        try:
            return {
                "is_training": self.is_training,
                "training_start_time": self.training_start_time.isoformat() if self.training_start_time else None,
                "component_managers": {
                    "model_trainer": self.model_trainer is not None,
                    "optimization_manager": self.optimization_manager is not None,
                    "ensemble_manager": self.ensemble_manager is not None,
                    "calibration_manager": self.calibration_manager is not None
                },
                "results_count": len(self.training_results)
            }
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            return {}
