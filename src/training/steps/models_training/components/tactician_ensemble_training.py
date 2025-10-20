"""
Tactician Ensemble Training Modular Component.

This component handles training of Tactician ensemble models that combine multiple base models
for enhanced entry/exit timing predictions.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import time

from .base_component import BaseModelsTrainingComponent
from src.training.steps.base_step import BaseStep
from ..core.ensemble_trainer import EnsembleTrainer
from ..core.base_trainer import TrainingConfig, TrainingRole, ModelType


class TacticianEnsembleMethod(Enum):
    """Tactician ensemble methods."""
    VOTING = "voting"
    AVERAGING = "averaging"
    STACKING = "stacking"
    BLENDING = "blending"


@dataclass
class TacticianEnsembleTrainingConfig:
    """Configuration for Tactician ensemble training."""
    base_models: List[str]
    ensemble_method: TacticianEnsembleMethod
    meta_learner_params: Dict[str, Any]
    timeframe: str = "15m"
    auto_save: bool = True


@dataclass
class TacticianEnsembleTrainingResult:
    """Result of Tactician ensemble training."""
    success: bool
    ensemble_model: Any
    individual_models: Dict[str, Any]
    ensemble_metrics: Dict[str, Any]
    training_time: float
    error_message: Optional[str] = None


class TacticianEnsembleTrainingModular(BaseModelsTrainingComponent, BaseStep):
    """
    ModularComponent implementation of Tactician Ensemble Training.
    
    This component handles training of Tactician ensemble models with comprehensive
    state management, performance monitoring, and error handling.
    """
    
    def __init__(
        self,
        name: str = "tactician_ensemble_training",
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Tactician Ensemble Training component.
        
        Args:
            name: Component name
            config: Configuration dictionary
            logger: Logger instance
        """
        # Set default configuration
        default_config = {
            'model': {
                'type': 'ensemble',
                'base_models': ['lightgbm', 'catboost', 'neural_network'],
                'ensemble_method': 'stacking',
                'meta_learner_type': 'lightgbm'
            },
            'training': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'early_stopping_patience': 10,
                'checkpoint_frequency': 10,
                'cv_folds': 5
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
        
        # Tactician ensemble-specific configuration
        self.ensemble_config = TacticianEnsembleTrainingConfig(
            base_models=self.model_config.get('base_models', []),
            ensemble_method=TacticianEnsembleMethod(self.model_config.get('ensemble_method', 'stacking')),
            meta_learner_params=self.model_config.get('meta_learner_params', {}),
            timeframe=self.model_config.get('timeframe', '15m'),
            auto_save=self.model_config.get('auto_save', True)
        )
        
        # Training state
        self._ensemble_model = None
        self._individual_models = {}
        self._training_results = {}
        self._ensemble_trainer = None
        
        # Performance tracking
        self.training_time = 0.0
        self._performance_metrics = {}
    
    def _initialize_resources(self) -> bool:
        """Initialize training resources."""
        try:
            # Initialize ensemble trainer
            training_config = TrainingConfig(
                role=TrainingRole.TACTICIAN,
                model_types=[ModelType(model) for model in self.ensemble_config.base_models],
                timeframe=self.ensemble_config.timeframe,
                symbol=self.config.get('symbol', 'ETHUSDT'),
                enable_ensemble=True,
                custom_params={
                    'ensemble_strategy': self.ensemble_config.ensemble_method.value,
                    'meta_learner_type': self.ensemble_config.meta_learner_params.get('type', 'lightgbm'),
                    'cv_folds': self.training_config.get('cv_folds', 5),
                    **self.ensemble_config.meta_learner_params
                }
            )
            
            self._ensemble_trainer = EnsembleTrainer(training_config, self.logger)
            
            self.logger.info("✅ Tactician ensemble training resources initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize tactician ensemble training resources: {e}")
            return False
    
    def _cleanup_resources(self) -> None:
        """Cleanup training resources."""
        try:
            if self._ensemble_trainer:
                # Cleanup trainer if it has cleanup method
                if hasattr(self._ensemble_trainer, 'cleanup'):
                    self._ensemble_trainer.cleanup()
                self._ensemble_trainer = None
            
            self.logger.info("✅ Tactician ensemble training resources cleaned up")
            
        except Exception as e:
            self.logger.error(f"❌ Resource cleanup failed: {e}")
    
    def _process_data(self, data: Dict[str, Any], **kwargs) -> TacticianEnsembleTrainingResult:
        """
        Process training data and train tactician ensemble models.
        
        Args:
            data: Dictionary containing 'X_train' and 'y_train'
            **kwargs: Additional arguments
            
        Returns:
            TacticianEnsembleTrainingResult
        """
        try:
            start_time = time.time()
            
            X_train = data.get('X_train')
            y_train = data.get('y_train')
            
            if X_train is None or y_train is None:
                return TacticianEnsembleTrainingResult(
                    success=False,
                    ensemble_model=None,
                    individual_models={},
                    ensemble_metrics={},
                    training_time=0.0,
                    error_message="Missing training data (X_train or y_train)"
                )
            
            # Initialize trainer
            if not self._ensemble_trainer.initialize():
                return TacticianEnsembleTrainingResult(
                    success=False,
                    ensemble_model=None,
                    individual_models={},
                    ensemble_metrics={},
                    training_time=0.0,
                    error_message="Failed to initialize ensemble trainer"
                )
            
            # Train ensemble
            result = self._ensemble_trainer.train(X_train, y_train)
            
            if result.success:
                # Store trained models
                self._ensemble_model = result.model
                self._individual_models = getattr(result, 'individual_models', {})
                self._training_results = result.metrics
                self._performance_metrics = result.metrics
                
                self.training_time = time.time() - start_time
                
                self.logger.info(f"✅ Tactician ensemble training completed in {self.training_time:.2f}s")
                
                return TacticianEnsembleTrainingResult(
                    success=True,
                    ensemble_model=self._ensemble_model,
                    individual_models=self._individual_models,
                    ensemble_metrics=result.metrics,
                    training_time=self.training_time
                )
            else:
                return TacticianEnsembleTrainingResult(
                    success=False,
                    ensemble_model=None,
                    individual_models={},
                    ensemble_metrics={},
                    training_time=time.time() - start_time,
                    error_message=result.error_message
                )
                
        except Exception as e:
            self.logger.error(f"❌ Tactician ensemble training failed: {e}")
            return TacticianEnsembleTrainingResult(
                success=False,
                ensemble_model=None,
                individual_models={},
                ensemble_metrics={},
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
        
        # Add tactician ensemble-specific information
        summary.update({
            'ensemble_config': {
                'base_models': self.ensemble_config.base_models,
                'ensemble_method': self.ensemble_config.ensemble_method.value,
                'timeframe': self.ensemble_config.timeframe
            },
            'ensemble_model': self._ensemble_model is not None,
            'individual_models': list(self._individual_models.keys()),
            'training_results': self._training_results,
        })
        
        return summary
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tactician ensemble training step (BaseStep interface).
        
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
            self.logger.info("🚀 Starting Tactician Ensemble Training")
            
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
                        'error': 'No target data found for tactician ensemble training',
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
            
            # Load tactician base model outputs if available
            base_model_outputs = self._load_dataframe('tactician_base_predictions')
            if base_model_outputs is not None:
                # Add base model outputs as features
                for col in base_model_outputs.columns:
                    training_data[f'base_{col}'] = base_model_outputs[col]
                self.logger.info(f"Enhanced training data with {len(base_model_outputs.columns)} base model features")
            
            # Prepare data for component
            component_data = {
                'X_train': training_data,
                'y_train': targets
            }
            
            # Initialize component
            if not self.initialize():
                return {
                    'success': False,
                    'error': 'Failed to initialize tactician ensemble training component',
                    'artifacts': [],
                    'metrics': {}
                }
            
            # Process data with component
            result = self.process(component_data)
            
            if result.success:
                # Save ensemble model
                if result.ensemble_model:
                    self._save_model(result.ensemble_model, 'tactician_ensemble_model')
                
                # Save individual models
                if result.individual_models:
                    for model_name, model in result.individual_models.items():
                        self._save_model(model, f'tactician_ensemble_{model_name}_model')
                
                # Save metrics
                if result.ensemble_metrics:
                    self._save_metadata(result.ensemble_metrics, 'tactician_ensemble_metrics')
                
                # Save training summary
                training_summary = self.get_training_summary()
                self._save_metadata(training_summary, 'tactician_ensemble_summary')
                
                self.logger.info("✅ Tactician Ensemble Training completed successfully")
                
                return {
                    'success': True,
                    'artifacts': [
                        'tactician_ensemble_model',
                        'tactician_ensemble_metrics',
                        'tactician_ensemble_summary'
                    ],
                    'metrics': result.ensemble_metrics,
                    'ensemble_method': self.ensemble_config.ensemble_method.value,
                    'base_models_count': len(result.individual_models),
                    'training_time': result.training_time
                }
            else:
                return {
                    'success': False,
                    'error': f"Tactician ensemble training failed: {result.error_message}",
                    'artifacts': [],
                    'metrics': {}
                }
                
        except Exception as e:
            self.logger.error(f"❌ Tactician Ensemble Training failed: {e}")
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


def create_tactician_ensemble_training(
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> TacticianEnsembleTrainingModular:
    """
    Create a Tactician Ensemble Training component.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        TacticianEnsembleTrainingModular instance
    """
    return TacticianEnsembleTrainingModular(config=config, logger=logger)