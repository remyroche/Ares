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

    def __init__(...) -> ...:
                """..."""
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
            self.logger.info("Initializing Training Orchestrator...")

            # Initialize component managers
            await self._initialize_component_managers()

            # Validate configuration
            if not self._validate_configuration():
                return False

            # Initialize validation framework
            await self._initialize_validation_framework()

            self.logger.info("✅ Training Orchestrator initialized successfully")
            return True

        except Exception as e:
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

            self.is_training = True
            self.training_start_time = datetime.now()


            self.logger.info("Training pipeline execution completed successfully")
            return True

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

            # Update state
            self.is_training = False


        except Exception as e:
            self.logger.error(f"Error stopping training pipeline: {e}")
            return False

