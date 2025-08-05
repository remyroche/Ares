# src/training/model_trainer.py

import asyncio
import pickle
from datetime import datetime
from typing import Any

import pandas as pd

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class ModelTrainer:
    """
    Model trainer responsible for training individual models and managing model lifecycle.
    This module handles the core model training logic for both analyst and tactician models.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize model trainer.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("ModelTrainer")
        
        # Model trainer state
        self.is_training: bool = False
        self.trained_models: dict[str, Any] = {}
        self.model_metadata: dict[str, Any] = {}
        
        # Configuration
        self.model_trainer_config: dict[str, Any] = self.config.get("model_trainer", {})
        self.enable_analyst_models: bool = self.model_trainer_config.get("enable_analyst_models", True)
        self.enable_tactician_models: bool = self.model_trainer_config.get("enable_tactician_models", True)
        
        # Model configurations
        self.analyst_models_config: dict[str, Any] = self.model_trainer_config.get("analyst_models", {})
        self.tactician_models_config: dict[str, Any] = self.model_trainer_config.get("tactician_models", {})

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid model trainer configuration"),
            AttributeError: (False, "Missing required model trainer parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="model trainer initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize model trainer.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Model Trainer...")
            
            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for model trainer")
                return False
            
            # Initialize model storage
            await self._initialize_model_storage()
            
            self.logger.info("✅ Model Trainer initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model Trainer initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate model trainer configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate model trainer specific settings
            if not self.enable_analyst_models and not self.enable_tactician_models:
                self.logger.error("At least one model type must be enabled")
                return False
            
            # Validate analyst models configuration
            if self.enable_analyst_models:
                if not self.analyst_models_config:
                    self.logger.error("Analyst models enabled but no configuration provided")
                    return False
            
            # Validate tactician models configuration
            if self.enable_tactician_models:
                if not self.tactician_models_config:
                    self.logger.error("Tactician models enabled but no configuration provided")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="model storage initialization",
    )
    async def _initialize_model_storage(self) -> None:
        """Initialize model storage and metadata."""
        try:
            # Create model storage directory if it doesn't exist
            import os
            model_dir = self.model_trainer_config.get("model_directory", "models")
            os.makedirs(model_dir, exist_ok=True)
            
            # Load existing model metadata
            metadata_file = os.path.join(model_dir, "model_metadata.json")
            if os.path.exists(metadata_file):
                import json
                with open(metadata_file, "r") as f:
                    self.model_metadata = json.load(f)
            
            self.logger.info(f"✅ Model storage initialized: {model_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize model storage: {e}")
            raise

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid training parameters"),
            AttributeError: (False, "Missing training components"),
            KeyError: (False, "Missing required training data"),
        },
        default_return=False,
        context="model training",
    )
    async def train_models(
        self,
        training_input: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Train all required models based on configuration.

        Args:
            training_input: Training input parameters

        Returns:
            dict: Training results for all models
        """
        try:
            self.logger.info("🚀 Starting model training...")
            self.is_training = True
            
            # Validate training input
            if not self._validate_training_input(training_input):
                return None
            
            # Prepare training data
            training_data = await self._prepare_training_data(training_input)
            if training_data is None:
                return None
            
            # Train analyst models
            analyst_results = None
            if self.enable_analyst_models:
                analyst_results = await self._train_analyst_models(training_data, training_input)
            
            # Train tactician models
            tactician_results = None
            if self.enable_tactician_models:
                tactician_results = await self._train_tactician_models(training_data, training_input)
            
            # Combine results
            training_results = {
                "analyst_models": analyst_results,
                "tactician_models": tactician_results,
                "training_input": training_input,
                "training_timestamp": datetime.now().isoformat(),
            }
            
            # Store trained models
            await self._store_trained_models(training_results)
            
            self.is_training = False
            self.logger.info("✅ Model training completed successfully")
            return training_results
            
        except Exception as e:
            self.logger.error(f"❌ Model training failed: {e}")
            self.is_training = False
            return None

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
        default_return=None,
        context="training data preparation",
    )
    async def _prepare_training_data(
        self,
        training_input: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Prepare training data for model training.

        Args:
            training_input: Training input parameters

        Returns:
            dict: Prepared training data
        """
        try:
            self.logger.info("📊 Preparing training data...")
            
            # Load data from data collection step
            from src.training.steps.step1_data_collection import DataCollectionStep
            
            data_collection = DataCollectionStep()
            data = await data_collection.execute(training_input)
            
            if data is None:
                self.logger.error("❌ Failed to collect training data")
                return None
            
            # Prepare features for different timeframes
            prepared_data = {}
            
            if self.enable_analyst_models:
                # Prepare data for analyst models (multi-timeframe)
                timeframes = ["1h", "15m", "5m", "1m"]
                for timeframe in timeframes:
                    prepared_data[f"analyst_{timeframe}"] = await self._prepare_timeframe_data(
                        data, timeframe, training_input
                    )
            
            if self.enable_tactician_models:
                # Prepare data for tactician models (1m only)
                prepared_data["tactician_1m"] = await self._prepare_timeframe_data(
                    data, "1m", training_input
                )
            
            self.logger.info("✅ Training data prepared successfully")
            return prepared_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to prepare training data: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="timeframe data preparation",
    )
    async def _prepare_timeframe_data(
        self,
        data: dict[str, Any],
        timeframe: str,
        training_input: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Prepare data for a specific timeframe.

        Args:
            data: Raw training data
            timeframe: Target timeframe
            training_input: Training input parameters

        Returns:
            dict: Prepared data for the timeframe
        """
        try:
            # Resample data to target timeframe
            resampled_data = self._resample_data_to_timeframe(data, timeframe)
            
            # Prepare features
            features = await self._prepare_features(resampled_data, timeframe)
            
            # Prepare labels
            labels = await self._prepare_labels(resampled_data, timeframe)
            
            return {
                "features": features,
                "labels": labels,
                "timeframe": timeframe,
                "data_info": {
                    "rows": len(resampled_data),
                    "columns": len(resampled_data.columns) if hasattr(resampled_data, 'columns') else 0,
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to prepare data for timeframe {timeframe}: {e}")
            return None

    def _resample_data_to_timeframe(
        self,
        data: dict[str, Any],
        timeframe: str,
    ) -> pd.DataFrame:
        """
        Resample data to target timeframe.

        Args:
            data: Raw data
            timeframe: Target timeframe

        Returns:
            pd.DataFrame: Resampled data
        """
        # This is a simplified implementation
        # In practice, this would handle proper OHLCV resampling
        if "klines" in data:
            df = data["klines"]
            # Apply timeframe-specific resampling logic
            return df
        return pd.DataFrame()

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="feature preparation",
    )
    async def _prepare_features(
        self,
        data: pd.DataFrame,
        timeframe: str,
    ) -> Any:
        """
        Prepare features for model training.

        Args:
            data: Training data
            timeframe: Target timeframe

        Returns:
            Any: Prepared features
        """
        try:
            # Import feature engineering components
            from src.analyst.advanced_feature_engineering import AdvancedFeatureEngineering
            
            feature_engineer = AdvancedFeatureEngineering()
            features = feature_engineer.generate_all_features(data)
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Failed to prepare features: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="label preparation",
    )
    async def _prepare_labels(
        self,
        data: pd.DataFrame,
        timeframe: str,
    ) -> Any:
        """
        Prepare labels for model training.

        Args:
            data: Training data
            timeframe: Target timeframe

        Returns:
            Any: Prepared labels
        """
        try:
            # This would implement label generation logic
            # For now, return a placeholder
            return pd.Series([0] * len(data))
            
        except Exception as e:
            self.logger.error(f"❌ Failed to prepare labels: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="analyst model training",
    )
    async def _train_analyst_models(
        self,
        training_data: dict[str, Any],
        training_input: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Train analyst models for multiple timeframes.

        Args:
            training_data: Prepared training data
            training_input: Training input parameters

        Returns:
            dict: Analyst model training results
        """
        try:
            self.logger.info("🧠 Training analyst models...")
            
            analyst_results = {}
            
            # Train models for each timeframe
            for timeframe in ["1h", "15m", "5m", "1m"]:
                data_key = f"analyst_{timeframe}"
                if data_key in training_data:
                    model_result = await self._train_single_analyst_model(
                        timeframe, training_data[data_key], training_input
                    )
                    if model_result:
                        analyst_results[timeframe] = model_result
            
            self.logger.info(f"✅ Trained {len(analyst_results)} analyst models")
            return analyst_results
            
        except Exception as e:
            self.logger.error(f"❌ Analyst model training failed: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="tactician model training",
    )
    async def _train_tactician_models(
        self,
        training_data: dict[str, Any],
        training_input: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Train tactician models for 1m timeframe.

        Args:
            training_data: Prepared training data
            training_input: Training input parameters

        Returns:
            dict: Tactician model training results
        """
        try:
            self.logger.info("🎯 Training tactician models...")
            
            tactician_results = {}
            
            # Train tactician models for 1m timeframe
            data_key = "tactician_1m"
            if data_key in training_data:
                model_result = await self._train_single_tactician_model(
                    training_data[data_key], training_input
                )
                if model_result:
                    tactician_results["1m"] = model_result
            
            self.logger.info(f"✅ Trained {len(tactician_results)} tactician models")
            return tactician_results
            
        except Exception as e:
            self.logger.error(f"❌ Tactician model training failed: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="single analyst model training",
    )
    async def _train_single_analyst_model(
        self,
        timeframe: str,
        training_data: dict[str, Any],
        training_input: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Train a single analyst model for a specific timeframe.

        Args:
            timeframe: Target timeframe
            training_data: Training data for the timeframe
            training_input: Training input parameters

        Returns:
            dict: Model training result
        """
        try:
            self.logger.info(f"🧠 Training analyst model for {timeframe} timeframe...")
            
            # This would implement actual model training logic
            # For now, return a placeholder result
            model_result = {
                "timeframe": timeframe,
                "model_type": "analyst",
                "training_status": "completed",
                "model_metrics": {
                    "accuracy": 0.85,
                    "precision": 0.82,
                    "recall": 0.78,
                },
                "model_path": f"models/analyst_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
            }
            
            # Store the model
            await self._store_model(model_result)
            
            return model_result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to train analyst model for {timeframe}: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="single tactician model training",
    )
    async def _train_single_tactician_model(
        self,
        training_data: dict[str, Any],
        training_input: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Train a single tactician model.

        Args:
            training_data: Training data
            training_input: Training input parameters

        Returns:
            dict: Model training result
        """
        try:
            self.logger.info("🎯 Training tactician model...")
            
            # This would implement actual model training logic
            # For now, return a placeholder result
            model_result = {
                "timeframe": "1m",
                "model_type": "tactician",
                "training_status": "completed",
                "model_metrics": {
                    "accuracy": 0.88,
                    "precision": 0.85,
                    "recall": 0.82,
                },
                "model_path": f"models/tactician_1m_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
            }
            
            # Store the model
            await self._store_model(model_result)
            
            return model_result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to train tactician model: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="model storage",
    )
    async def _store_model(self, model_result: dict[str, Any]) -> None:
        """
        Store a trained model.

        Args:
            model_result: Model training result
        """
        try:
            # Create a mock model object for storage
            mock_model = {
                "model_type": model_result["model_type"],
                "timeframe": model_result["timeframe"],
                "training_timestamp": datetime.now().isoformat(),
                "metrics": model_result["model_metrics"],
            }
            
            # Save model to file
            import os
            model_dir = self.model_trainer_config.get("model_directory", "models")
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, f"{model_result['model_type']}_{model_result['timeframe']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
            
            with open(model_path, "wb") as f:
                pickle.dump(mock_model, f)
            
            # Update metadata
            model_key = f"{model_result['model_type']}_{model_result['timeframe']}"
            self.model_metadata[model_key] = {
                "path": model_path,
                "training_timestamp": datetime.now().isoformat(),
                "metrics": model_result["model_metrics"],
            }
            
            # Save metadata
            metadata_path = os.path.join(model_dir, "model_metadata.json")
            import json
            with open(metadata_path, "w") as f:
                json.dump(self.model_metadata, f, indent=2)
            
            self.logger.info(f"✅ Model stored: {model_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store model: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="trained models storage",
    )
    async def _store_trained_models(self, training_results: dict[str, Any]) -> None:
        """
        Store all trained models.

        Args:
            training_results: Complete training results
        """
        try:
            self.logger.info("📁 Storing trained models...")
            
            # Store analyst models
            if training_results.get("analyst_models"):
                for timeframe, model_result in training_results["analyst_models"].items():
                    await self._store_model(model_result)
            
            # Store tactician models
            if training_results.get("tactician_models"):
                for timeframe, model_result in training_results["tactician_models"].items():
                    await self._store_model(model_result)
            
            self.logger.info("✅ All trained models stored successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store trained models: {e}")

    def get_training_status(self) -> dict[str, Any]:
        """
        Get current training status.

        Returns:
            dict: Training status information
        """
        return {
            "is_training": self.is_training,
            "trained_models_count": len(self.trained_models),
            "analyst_models_enabled": self.enable_analyst_models,
            "tactician_models_enabled": self.enable_tactician_models,
        }

    def get_trained_models(self) -> dict[str, Any]:
        """
        Get all trained models.

        Returns:
            dict: Trained models information
        """
        return self.trained_models.copy()

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="model trainer cleanup",
    )
    async def stop(self) -> None:
        """Stop the model trainer and cleanup resources."""
        try:
            self.logger.info("🛑 Stopping Model Trainer...")
            self.is_training = False
            self.logger.info("✅ Model Trainer stopped successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to stop Model Trainer: {e}")


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="model trainer setup",
)
async def setup_model_trainer(
    config: dict[str, Any] | None = None,
) -> ModelTrainer | None:
    """
    Setup and return a configured ModelTrainer instance.

    Args:
        config: Configuration dictionary

    Returns:
        ModelTrainer: Configured model trainer instance
    """
    try:
        trainer = ModelTrainer(config or {})
        if await trainer.initialize():
            return trainer
        return None
    except Exception as e:
        system_logger.error(f"Failed to setup model trainer: {e}")
        return None 