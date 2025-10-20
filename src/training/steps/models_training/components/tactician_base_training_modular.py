"""
Tactician Base Training Modular Component.

This component handles training of individual Tactician base models for precise entry/exit timing.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import time

from .base_component import BaseModelsTrainingComponent
from src.training.steps.base_step import BaseStep
from ..core.model_trainer import ModelTrainer
from ..core.base_trainer import TrainingConfig, TrainingRole, ModelType


class TacticianModelType(Enum):
    """Types of Tactician models."""
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    NEURAL_NETWORK = "neural_network"


@dataclass
class TacticianBaseTrainingConfig:
    """Configuration for Tactician base training."""
    model_types: List[TacticianModelType]
    training_params: Dict[str, Any]
    validation_params: Dict[str, Any]
    timeframe: str = "15m"
    auto_save: bool = True


@dataclass
class TacticianBaseTrainingResult:
    """Result of Tactician base training."""
    success: bool
    models: Dict[str, Any]
    metrics: Dict[str, Any]
    training_time: float
    error_message: Optional[str] = None


class TacticianBaseTrainingModular(BaseModelsTrainingComponent, BaseStep):
    """
    ModularComponent implementation of Tactician Base Training.
    
    This component handles training of individual Tactician base models with comprehensive
    state management, performance monitoring, and error handling.
    """
    
    def __init__(
        self,
        name: str = "tactician_base_training",
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Tactician Base Training component.
        
        Args:
            name: Component name
            config: Configuration dictionary
            logger: Logger instance
        """
        # Set default configuration
        default_config = {
            'model': {
                'type': 'multi_model',
                'model_types': ['lightgbm', 'catboost', 'neural_network'],
            },
            'training': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'early_stopping_patience': 10,
                'checkpoint_frequency': 10
            },
            'validation': {
                'split': 0.2,
                'metrics': ['accuracy', 'precision', 'recall', 'f1_score']
            },
            'timeframe': '15m',
            'auto_save': True
        }
        
        if config:
            default_config.update(config)
        
        # Initialize both parent classes
        BaseModelsTrainingComponent.__init__(self, name, default_config, logger)
        BaseStep.__init__(self, name, default_config)
        
        # Tactician-specific configuration
        self.tactician_config = TacticianBaseTrainingConfig(
            model_types=[TacticianModelType(model) for model in self.model_config.get('model_types', [])],
            training_params=self.training_config,
            validation_params=self.validation_config,
            timeframe=self.model_config.get('timeframe', '15m'),
            auto_save=self.model_config.get('auto_save', True)
        )
        
        # Training state
        self._trained_models = {}
        self._training_results = {}
        self._model_trainer = None
        
        # Performance tracking
        self.training_time = 0.0
        self._performance_metrics = {}
    
    def _initialize_resources(self) -> bool:
        """Initialize training resources."""
        try:
            # Initialize model trainer
            training_config = TrainingConfig(
                role=TrainingRole.TACTICIAN,
                model_types=[ModelType(model.value) for model in self.tactician_config.model_types],
                timeframe=self.tactician_config.timeframe,
                symbol=self.config.get('symbol', 'ETHUSDT'),
                enable_ensemble=False,  # Individual models only
                custom_params=self.tactician_config.training_params
            )
            
            self._model_trainer = ModelTrainer(training_config, self.logger)
            
            self.logger.info("✅ Tactician base training resources initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize tactician base training resources: {e}")
            return False
    
    def _cleanup_resources(self) -> None:
        """Cleanup training resources."""
        try:
            if self._model_trainer:
                # Cleanup trainer if it has cleanup method
                if hasattr(self._model_trainer, 'cleanup'):
                    self._model_trainer.cleanup()
                self._model_trainer = None
            
            self.logger.info("✅ Tactician base training resources cleaned up")
            
        except Exception as e:
            self.logger.error(f"❌ Resource cleanup failed: {e}")
    
    def _process_data(self, data: Dict[str, Any], **kwargs) -> TacticianBaseTrainingResult:
        """
        Process training data and train tactician base models.
        
        Args:
            data: Dictionary containing 'X_train' and 'y_train'
            **kwargs: Additional arguments
            
        Returns:
            TacticianBaseTrainingResult
        """
        try:
            start_time = time.time()
            
            X_train = data.get('X_train')
            y_train = data.get('y_train')
            
            if X_train is None or y_train is None:
                return TacticianBaseTrainingResult(
                    success=False,
                    models={},
                    metrics={},
                    training_time=0.0,
                    error_message="Missing training data (X_train or y_train)"
                )
            
            # Initialize trainer
            if not self._model_trainer.initialize():
                return TacticianBaseTrainingResult(
                    success=False,
                    models={},
                    metrics={},
                    training_time=0.0,
                    error_message="Failed to initialize model trainer"
                )
            
            # Train models
            result = self._model_trainer.train(X_train, y_train)
            
            if result.success:
                # Store trained models
                self._trained_models = {model_type: result.model for model_type in self.tactician_config.model_types}
                self._training_results = result.metrics
                self._performance_metrics = result.metrics
                
                self.training_time = time.time() - start_time
                
                self.logger.info(f"✅ Tactician base training completed in {self.training_time:.2f}s")
                
                return TacticianBaseTrainingResult(
                    success=True,
                    models=self._trained_models,
                    metrics=result.metrics,
                    training_time=self.training_time
                )
            else:
                return TacticianBaseTrainingResult(
                    success=False,
                    models={},
                    metrics={},
                    training_time=time.time() - start_time,
                    error_message=result.error_message
                )
                
        except Exception as e:
            self.logger.error(f"❌ Tactician base training failed: {e}")
            return TacticianBaseTrainingResult(
                success=False,
                models={},
                metrics={},
                training_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _validate_training_data(self, data: Dict[str, Any]) -> bool:
        """Validate training data."""
        try:
            X_train = data.get('X_train')
            y_train = data.get('y_train')
            
            if X_train is None or y_train is None:
                self.logger.error("❌ Missing training data")
                return False
            
            if len(X_train) != len(y_train):
                self.logger.error("❌ Training data length mismatch")
                return False
            
            if len(X_train) == 0:
                self.logger.error("❌ Empty training data")
                return False
            
            self.logger.info(f"✅ Training data validated: {len(X_train)} samples")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data validation failed: {e}")
            return False
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        summary = super().get_training_summary()
        
        # Add tactician-specific information
        summary.update({
            'tactician_config': {
                'model_types': [mt.value for mt in self.tactician_config.model_types],
                'timeframe': self.tactician_config.timeframe
            },
            'trained_models': list(self._trained_models.keys()),
            'training_results': self._training_results,
        })
        
        return summary
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tactician base training step (BaseStep interface).
        
        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - direction: Trading direction ('longs', 'shorts', 'both')
                - execution_mode: Execution mode ('full', 'light', 'blank')
        
        Returns:
            Execution result dictionary
        """
        try:
            self.logger.info("🚀 Starting Tactician Base Training")
            
            # Set context for artifact management
            self._set_context(
                symbol=config.get('symbol'),
                exchange=config.get('exchange'),
                information='training',
                direction=config.get('direction', 'longs'),
                model='Tactician'
            )
            
            # Load training data
            training_data = self._load_dataframe('training_data')
            if training_data is None:
                training_data = self._load_dataframe('market_data')
                if training_data is None:
                    training_data = self._load_dataframe('processed_data')
            
            if training_data is None:
                return {
                    'success': False,
                    'error': 'No training data found. Please ensure data is available in artifacts.',
                    'artifacts': [],
                    'metrics': {}
                }
            
            # Load targets if available
            targets = self._load_dataframe('tactician_targets')
            if targets is None:
                target_columns = ['target', 'y', 'label', 'tactician_target', 'entry_target', 'exit_target']
                for col in target_columns:
                    if col in training_data.columns:
                        targets = training_data[col]
                        training_data = training_data.drop(columns=[col])
                        break
                
                if targets is None:
                    return {
                        'success': False,
                        'error': 'No target data found for tactician training',
                        'artifacts': [],
                        'metrics': {}
                    }
            
            # Load analyst predictions if available (for enhanced features)
            analyst_predictions = self._load_dataframe('analyst_predictions')
            if analyst_predictions is not None:
                # Add analyst predictions as features
                for col in analyst_predictions.columns:
                    training_data[f'analyst_{col}'] = analyst_predictions[col]
                self.logger.info(f"Enhanced training data with {len(analyst_predictions.columns)} analyst features")
            
            # Prepare data for component
            component_data = {
                'X_train': training_data,
                'y_train': targets
            }
            
            # Initialize component
            if not self.initialize():
                return {
                    'success': False,
                    'error': 'Failed to initialize tactician training component',
                    'artifacts': [],
                    'metrics': {}
                }
            
            # Process data with component
            result = self.process(component_data)
            
            if result.success:
                # Save trained models
                if result.models:
                    self._save_model(result.models, 'tactician_base_models')
                
                # Save metrics
                if result.metrics:
                    self._save_metadata(result.metrics, 'tactician_training_metrics')
                
                # Save training summary
                training_summary = self.get_training_summary()
                self._save_metadata(training_summary, 'tactician_training_summary')
                
                self.logger.info("✅ Tactician Base Training completed successfully")
                
                return {
                    'success': True,
                    'artifacts': [
                        'tactician_base_models',
                        'tactician_training_metrics',
                        'tactician_training_summary'
                    ],
                    'metrics': result.metrics,
                    'models_trained': len(result.models),
                    'training_time': result.training_time
                }
            else:
                return {
                    'success': False,
                    'error': f"Tactician training failed: {result.error_message}",
                    'artifacts': [],
                    'metrics': {}
                }
                
        except Exception as e:
            self.logger.error(f"❌ Tactician Base Training failed: {e}")
            return {
                'success': False,
                'error': f"Step execution failed: {str(e)}",
                'artifacts': [],
                'metrics': {}
            }
        finally:
            # Cleanup component
            if hasattr(self, 'cleanup'):
                self.cleanup()


def create_tactician_base_training(
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> TacticianBaseTrainingModular:
    """
    Create a Tactician Base Training component.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        TacticianBaseTrainingModular instance
    """
    return TacticianBaseTrainingModular(config=config, logger=logger)