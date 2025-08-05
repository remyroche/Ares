# src/training/training_orchestrator.py

import asyncio
from datetime import datetime
from typing import Any

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class TrainingOrchestrator:
    """
    Training orchestrator responsible for coordinating the overall training pipeline.
    This module handles the high-level coordination between different training components.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize training orchestrator.

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
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="training orchestrator initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize training orchestrator and all component managers.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Training Orchestrator...")
            
            # Initialize component managers
            await self._initialize_component_managers()
            
            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for training orchestrator")
                return False
                
            self.logger.info("✅ Training Orchestrator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Training Orchestrator initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="component managers initialization",
    )
    async def _initialize_component_managers(self) -> None:
        """Initialize all component managers."""
        try:
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
            
            self.logger.info("✅ All component managers initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize component managers: {e}")
            raise

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate training orchestrator configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate required configuration sections
            required_sections = ["training_orchestrator", "model_trainer", "optimization_manager"]
            
            for section in required_sections:
                if section not in self.config:
                    self.logger.error(f"Missing required configuration section: {section}")
                    return False
            
            # Validate training orchestrator specific settings
            orchestrator_config = self.config.get("training_orchestrator", {})
            
            if orchestrator_config.get("max_training_duration", 0) <= 0:
                self.logger.error("Invalid max_training_duration configuration")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid training parameters"),
            AttributeError: (False, "Missing training components"),
            KeyError: (False, "Missing required training data"),
        },
        default_return=False,
        context="training execution",
    )
    async def execute_training(
        self,
        training_input: dict[str, Any],
    ) -> bool:
        """
        Execute the complete training pipeline.

        Args:
            training_input: Training input parameters

        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            self.logger.info("🚀 Starting training pipeline execution...")
            self.training_start_time = datetime.now()
            self.is_training = True
            
            # Validate training input
            if not self._validate_training_input(training_input):
                return False
            
            # Execute training pipeline
            success = await self._execute_training_pipeline(training_input)
            
            if success:
                self.logger.info("✅ Training pipeline completed successfully")
                await self._store_training_results(training_input)
            else:
                self.logger.error("❌ Training pipeline failed")
            
            self.is_training = False
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Training execution failed: {e}")
            self.is_training = False
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="training input validation",
    )
    def _validate_training_input(self, training_input: dict[str, Any]) -> bool:
        """
        Validate training input parameters.

        Args:
            training_input: Training input parameters

        Returns:
            bool: True if input is valid, False otherwise
        """
        try:
            required_fields = ["symbol", "exchange", "timeframe", "lookback_days"]
            
            for field in required_fields:
                if field not in training_input:
                    self.logger.error(f"Missing required training input field: {field}")
                    return False
            
            # Validate specific field values
            if training_input.get("lookback_days", 0) <= 0:
                self.logger.error("Invalid lookback_days value")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Training input validation failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="training pipeline execution",
    )
    async def _execute_training_pipeline(
        self,
        training_input: dict[str, Any],
    ) -> bool:
        """
        Execute the main training pipeline.

        Args:
            training_input: Training input parameters

        Returns:
            bool: True if pipeline successful, False otherwise
        """
        try:
            self.logger.info("📊 Executing training pipeline...")
            
            # Step 1: Model Training
            self.logger.info("🔧 Step 1: Model Training")
            model_results = await self.model_trainer.train_models(training_input)
            if not model_results:
                self.logger.error("❌ Model training failed")
                return False
            
            # Step 2: Optimization
            self.logger.info("🔧 Step 2: Model Optimization")
            optimization_results = await self.optimization_manager.optimize_models(
                model_results, training_input
            )
            if not optimization_results:
                self.logger.error("❌ Model optimization failed")
                return False
            
            # Step 3: Ensemble Creation
            self.logger.info("🔧 Step 3: Ensemble Creation")
            ensemble_results = await self.ensemble_manager.create_ensembles(
                optimization_results, training_input
            )
            if not ensemble_results:
                self.logger.error("❌ Ensemble creation failed")
                return False
            
            # Step 4: Calibration
            self.logger.info("🔧 Step 4: Model Calibration")
            calibration_results = await self.calibration_manager.calibrate_models(
                ensemble_results, training_input
            )
            if not calibration_results:
                self.logger.error("❌ Model calibration failed")
                return False
            
            # Store final results
            self.training_results = {
                "model_results": model_results,
                "optimization_results": optimization_results,
                "ensemble_results": ensemble_results,
                "calibration_results": calibration_results,
                "training_input": training_input,
                "execution_time": datetime.now() - self.training_start_time,
            }
            
            self.logger.info("✅ Training pipeline completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Training pipeline execution failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="training results storage",
    )
    async def _store_training_results(self, training_input: dict[str, Any]) -> None:
        """
        Store training results for later retrieval.

        Args:
            training_input: Training input parameters
        """
        try:
            # Store results in a format that can be retrieved later
            results_key = f"{training_input['symbol']}_{training_input['exchange']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # This would typically store to database or file system
            self.logger.info(f"📁 Storing training results with key: {results_key}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store training results: {e}")

    def get_training_status(self) -> dict[str, Any]:
        """
        Get current training status.

        Returns:
            dict: Training status information
        """
        return {
            "is_training": self.is_training,
            "training_start_time": self.training_start_time,
            "training_duration": datetime.now() - self.training_start_time if self.training_start_time else None,
            "has_results": bool(self.training_results),
        }

    def get_training_results(self) -> dict[str, Any]:
        """
        Get the latest training results.

        Returns:
            dict: Training results
        """
        return self.training_results.copy()

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="training orchestrator cleanup",
    )
    async def stop(self) -> None:
        """Stop the training orchestrator and cleanup resources."""
        try:
            self.logger.info("🛑 Stopping Training Orchestrator...")
            
            # Stop component managers
            if self.model_trainer:
                await self.model_trainer.stop()
            if self.optimization_manager:
                await self.optimization_manager.stop()
            if self.ensemble_manager:
                await self.ensemble_manager.stop()
            if self.calibration_manager:
                await self.calibration_manager.stop()
            
            self.is_training = False
            self.logger.info("✅ Training Orchestrator stopped successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to stop Training Orchestrator: {e}")


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="training orchestrator setup",
)
async def setup_training_orchestrator(
    config: dict[str, Any] | None = None,
) -> TrainingOrchestrator | None:
    """
    Setup and return a configured TrainingOrchestrator instance.

    Args:
        config: Configuration dictionary

    Returns:
        TrainingOrchestrator: Configured training orchestrator instance
    """
    try:
        orchestrator = TrainingOrchestrator(config or {})
        if await orchestrator.initialize():
            return orchestrator
        return None
    except Exception as e:
        system_logger.error(f"Failed to setup training orchestrator: {e}")
        return None 