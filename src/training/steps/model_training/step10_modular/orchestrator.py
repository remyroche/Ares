from ..standardized_parquet_handler import standardized_parquet_handler
"""Step 10 Modular Orchestrator.

This is the main orchestrator for the unified regime intelligence system,
coordinating all modular components for clean, maintainable execution.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

from .config import Step10Config, create_step10_config
from .models import MultiTimeframeHMMEncoder
from .base.logger import setup_step10_logger
from .base.imports import validate_step10_imports
from .base.utils import ensure_directory
from .base.imports import safe_import_manager
from .features import FeatureEngineer
from .training import TrainingOrchestrator
from .prediction import PredictionManager
from .artifacts import ArtifactManager

from ...utils.logger import system_logger

logger = system_logger.getChild('Step10Orchestrator')

class UnifiedRegimeIntelligenceOrchestrator:
    """Main orchestrator for the unified regime intelligence system.

    This class coordinates all modular components to provide a clean,
    maintainable interface for the unified regime intelligence functionality.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Step 10 orchestrator.

        Args:
            config: Configuration dictionary (optional)
        """
        self.config = create_step10_config(config)
        self.logger = setup_step10_logger()

        # Validate imports
        if not validate_step10_imports():
            raise ImportError("Step 10 dependencies not satisfied")

        # Initialize components
        self._initialize_components()

        self.logger.info("✅ Step 10 Modular Orchestrator initialized")

    def _initialize_components(self) -> None:
        """Initialize all modular components."""
        try:
            # Core components
            self.model = MultiTimeframeHMMEncoder(self.config.get_model_config())
            self.feature_engineer = FeatureEngineer(self.config)
            self.training_orchestrator = TrainingOrchestrator(self.config)
            self.prediction_manager = PredictionManager(self.config)
            self.artifact_manager = ArtifactManager(self.config)

            # Setup directories
            self.artifacts_dir = ensure_directory(self.config.artifacts_dir)

            self.logger.info("✅ All components initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Component initialization failed: {e}")
            raise

    async def initialize(self) -> bool:
        """Initialize the system for execution.

        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("🚀 Initializing Step 10 Unified Regime Intelligence")

            # Validate configuration
            errors = self.config.validate()
            if errors:
                for error in errors:
                    self.logger.error(f"Configuration error: {error}")
                return False

            # Initialize components
            await self.training_orchestrator.initialize()
            await self.prediction_manager.initialize()
            await self.artifact_manager.initialize()

            self.logger.info("✅ Step 10 initialization completed")
            return True

        except Exception as e:
            self.logger.error(f"❌ Step 10 initialization failed: {e}")
            return False

    async def train(self, data: Dict[str, Any]) -> bool:
        """Train the unified regime intelligence model.

        Args:
            data: Training data dictionary

        Returns:
            True if training successful
        """
        try:
            self.logger.info("🚀 Starting Step 10 training")

            # Prepare features
            prepared_data = await self.feature_engineer.prepare_features(data)
            if not prepared_data:
                self.logger.error("❌ Feature preparation failed")
                return False

            # Train model
            training_result = await self.training_orchestrator.train(
                prepared_data, self.model
            )
            if not training_result:
                self.logger.error("❌ Model training failed")
                return False

            # Save artifacts
            await self.artifact_manager.save_training_artifacts(
                self.model, training_result
            )

            self.logger.info("✅ Step 10 training completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Step 10 training failed: {e}")
            return False

    async def predict(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make predictions using the trained model.

        Args:
            input_data: Input data for prediction

        Returns:
            Prediction results or None if failed
        """
        try:
            # Prepare features
            features = await self.feature_engineer.prepare_prediction_features(input_data)
            if not features:
                return None

            # Make predictions
            predictions = await self.prediction_manager.predict(self.model, features)

            # Add S/R integration if available
            if hasattr(self.prediction_manager, 'sr_predictor'):
                sr_results = await self.prediction_manager.enhance_with_sr_analysis(
                    predictions, input_data
                )
                predictions.update(sr_results)

            return predictions

        except Exception as e:
            self.logger.error(f"❌ Step 10 prediction failed: {e}")
            return None

    async def save_artifacts(self) -> bool:
        """Save all artifacts and model state.

        Returns:
            True if successful
        """
        try:
            await self.artifact_manager.save_model(self.model)
            await self.artifact_manager.save_metadata({
                "config": self.config.to_dict(),
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            })

            self.logger.info("✅ Artifacts saved successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to save artifacts: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get current system status.

        Returns:
            Status information dictionary
        """
        return {
            "initialized": hasattr(self, 'model'),
            "model_trained": self.training_orchestrator.is_trained if hasattr(self.training_orchestrator, 'is_trained') else False,
            "config_valid": len(self.config.validate()) == 0,
            "artifacts_dir": self.artifacts_dir,
            "components": {
                "model": self.model is not None,
                "feature_engineer": self.feature_engineer is not None,
                "training_orchestrator": self.training_orchestrator is not None,
                "prediction_manager": self.prediction_manager is not None,
                "artifact_manager": self.artifact_manager is not None,
            }
        }

    async def cleanup(self) -> None:
        """Cleanup resources and temporary files."""
        try:
            if hasattr(self.artifact_manager, 'cleanup'):
                await self.artifact_manager.cleanup()
            self.logger.info("✅ Step 10 cleanup completed")
        except Exception as e:
            self.logger.error(f"❌ Step 10 cleanup failed: {e}")

# Factory function for backward compatibility
async def run_step(symbol: str, exchange: str, timeframe: str = "1m",
                   data_dir: str = None, force_rerun: bool = False,
                   **kwargs) -> bool:
    """Run Step 10 with the modular architecture.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Data timeframe
        data_dir: Data directory
        force_rerun: Force rerun flag
        **kwargs: Additional configuration

    Returns:
        True if successful
    """
    try:
        # Create configuration
        config = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframes": [timeframe],
            "data_dir": data_dir or f"data/{exchange}/{symbol}",
            **kwargs
        }

        # Initialize orchestrator
        orchestrator = UnifiedRegimeIntelligenceOrchestrator(config)

        # Initialize system
        if not await orchestrator.initialize():
            return False

        # For now, just validate the setup
        status = orchestrator.get_status()
        if status["initialized"] and status["config_valid"]:
            logger.info("✅ Step 10 modular setup validated successfully")
            return True
        else:
            logger.error("❌ Step 10 modular setup validation failed")
            return False

    except Exception as e:
        logger.error(f"❌ Step 10 execution failed: {e}")
        return False
