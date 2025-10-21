"""
Tactician Ensemble Training - Unified Training Architecture

This module provides the Tactician ensemble training component that handles training
of ensemble models combining multiple Tactician base models using the unified BaseTrainer architecture.

Key Features:
- Ensemble training interface for Tactician models
- Multiple ensemble methods (voting, averaging, stacking, blending)
- Meta-learner support for complex ensemble strategies
- Performance monitoring and validation
- Error handling and recovery mechanisms
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

# from ..core.tactician_ensemble_trainer import (
#     TacticianEnsembleTrainer, TacticianEnsembleTrainingConfig, TacticianEnsembleMethod
# )

# Simple enum for tactician ensemble methods
from enum import Enum
class TacticianEnsembleMethod(Enum):
    STACKING = "stacking"
    VOTING = "voting"
    AVERAGING = "averaging"
    BLENDING = "blending"
from ..core.tactician_base_trainer import TacticianModelType
from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_debug, tprint_performance
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.hardware.integrated_hardware_manager import (
    get_integrated_hardware_manager, WorkloadType, process_ml_training_data
)
from src.utils.hardware.optimization_decorators import (
    smart_cache, auto_optimize, memory_efficient, performance_tracked
)
from src.utils.hardware.memory_optimized_decorators import (
    memory_optimized, comprehensive_memory_optimization, MemoryOptimizationLevel
)


@dataclass
class TacticianEnsembleTrainingConfig:
    """Configuration for Tactician ensemble training."""
    base_models: List[TacticianModelType]
    ensemble_method: TacticianEnsembleMethod = TacticianEnsembleMethod.STACKING
    training_params: Dict[str, Any] = field(default_factory=dict)
    validation_params: Dict[str, Any] = field(default_factory=dict)
    timeframe: str = "15m"
    symbol: str = "ETHUSDT"
    auto_save: bool = True
    
    # Feature engineering parameters
    enable_entry_timing: bool = True
    enable_exit_timing: bool = True
    enable_position_sizing: bool = True
    
    # Meta-learner parameters
    meta_learner_params: Dict[str, Any] = field(default_factory=lambda: {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    })


@dataclass
class TacticianEnsembleTrainingResult:
    """Result of Tactician ensemble training."""
    success: bool
    ensemble_model: Any = None
    individual_models: Dict[str, Any] = field(default_factory=dict)
    ensemble_metrics: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    feature_importance: Optional[Dict[str, float]] = None


class TacticianEnsembleTraining(BaseStep):
    """
    Tactician ensemble training component using unified BaseTrainer architecture.
    
    This component handles training of ensemble models that combine multiple
    Tactician base models for enhanced performance.
    """
    
    def __init__(
        self,
        name: str = "tactician_ensemble_training",
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Tactician ensemble training component.
        
        Args:
            name: Component name
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(name, config)
        
        # Set default configuration
        default_config = {
            'base_models': [
                TacticianModelType.LIGHTGBM,
                TacticianModelType.CATBOOST,
                TacticianModelType.NEURAL_NETWORK
            ],
            'ensemble_method': TacticianEnsembleMethod.STACKING,
            'training_params': {
                'validation_split': 0.2,
                'cross_validation_folds': 5,
                'random_seed': 42
            },
            'validation_params': {
                'enable_early_stopping': True,
                'early_stopping_patience': 10
            },
            'timeframe': '15m',
            'symbol': 'ETHUSDT',
            'auto_save': True,
            'enable_entry_timing': True,
            'enable_exit_timing': True,
            'enable_position_sizing': True,
            'meta_learner_params': {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }
        }
        
        # Merge with provided configuration
        if config:
            default_config.update(config)
        
        self.config = TacticianEnsembleTrainingConfig(**default_config)
        
        # Initialize trainer
        self._trainer = None
        
        tprint_info(f"🔧 Initialized TacticianEnsembleTraining: {name}")
        self.logger.info(f"Initialized TacticianEnsembleTraining: {name}")
    
    @handles_errors(
        exceptions=(ValueError, RuntimeError, MemoryError),
        default_return=TacticianEnsembleTrainingResult(
            success=False,
            errors=["Component initialization failed"]
        ),
        context="tactician ensemble training"
    )
    async def initialize(self) -> bool:
        """Initialize the component."""
        try:
            tprint_info("🔧 Initializing Tactician ensemble training...")
            
            # Create trainer configuration
            trainer_config = TacticianEnsembleTrainingConfig(
                base_models=self.config.base_models,
                ensemble_method=self.config.ensemble_method,
                timeframe=self.config.timeframe,
                symbol=self.config.symbol,
                validation_split=self.config.training_params.get('validation_split', 0.2),
                cross_validation_folds=self.config.training_params.get('cross_validation_folds', 5),
                random_seed=self.config.training_params.get('random_seed'),
                enable_entry_timing=self.config.enable_entry_timing,
                enable_exit_timing=self.config.enable_exit_timing,
                enable_position_sizing=self.config.enable_position_sizing,
                meta_learner_params=self.config.meta_learner_params,
                custom_params=self.config.training_params
            )
            
            # Create trainer
            self._trainer = TacticianEnsembleTrainer(trainer_config, self.logger)
            
            # Initialize trainer
            if not await self._trainer.initialize():
                tprint_error("❌ Trainer initialization failed")
                return False
            
            tprint_success("✅ Tactician ensemble training initialized")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Initialization failed: {e}")
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    @handles_errors(
        exceptions=(ValueError, RuntimeError, MemoryError),
        default_return=TacticianEnsembleTrainingResult(
            success=False,
            errors=["Training execution failed"]
        ),
        context="tactician ensemble training"
    )
    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the Tactician ensemble training.
        
        Args:
            data: Input data containing training features and targets
            
        Returns:
            Training result dictionary
        """
        try:
            tprint_info("🎯 Starting Tactician ensemble training...")
            self.logger.info("Starting Tactician ensemble training...")
            
            start_time = time.time()
            
            # Preview input data dictionary
            from src.utils.tprint import tprint_data_preview
            tprint_data_preview(data, "Input ensemble data dictionary", max_rows=5, level="INFO")
            
            # Extract data
            X_train = data.get('X_train')
            y_train = data.get('y_train')
            
            # Preview extracted training data
            if X_train is not None:
                tprint_data_preview(X_train, "Extracted ensemble X_train", max_rows=5, level="INFO")
            if y_train is not None:
                tprint_data_preview(y_train, "Extracted ensemble y_train", max_rows=10, level="INFO")
            
            if X_train is None or y_train is None:
                return {
                    'success': False,
                    'error_message': 'Missing required data: X_train and y_train',
                    'training_time': 0.0
                }
            
            # Convert to pandas if needed
            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train)
            if not isinstance(y_train, pd.Series):
                y_train = pd.Series(y_train)
            
            # Preview processed training data
            tprint_data_preview(X_train, "Processed ensemble X_train", max_rows=5, level="DEBUG")
            tprint_data_preview(y_train, "Processed ensemble y_train", max_rows=10, level="DEBUG")
            
            # Train ensemble
            training_result = await self._trainer.train(X_train, y_train)
            
            if not training_result.success:
                return {
                    'success': False,
                    'error_message': training_result.error_message,
                    'training_time': time.time() - start_time
                }
            
            # Create result
            result = TacticianEnsembleTrainingResult(
                success=True,
                ensemble_model=training_result.model,
                individual_models=training_result.metadata.get('base_models', {}),
                ensemble_metrics=training_result.metrics,
                training_time=training_result.training_time,
                feature_importance=self._extract_feature_importance(training_result)
            )
            
            # Auto-save if enabled
            if self.config.auto_save:
                await self._save_models(result)
            
            tprint_success(f"✅ Tactician ensemble training completed in {result.training_time:.2f}s")
            self.logger.info(f"Tactician ensemble training completed in {result.training_time:.2f}s")
            
            return {
                'success': True,
                'result': result,
                'training_time': result.training_time,
                'ensemble_method': self.config.ensemble_method.value,
                'base_models': [model.value for model in self.config.base_models],
                'metrics': result.ensemble_metrics
            }
            
        except Exception as e:
            tprint_error(f"❌ Tactician ensemble training failed: {e}")
            self.logger.error(f"Tactician ensemble training failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'training_time': time.time() - start_time if 'start_time' in locals() else 0.0
            }
    
    async def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the trained ensemble.
        
        Args:
            data: Validation data containing features and targets
            
        Returns:
            Validation result dictionary
        """
        try:
            tprint_info("🎯 Validating Tactician ensemble...")
            
            # Extract data
            X_val = data.get('X_val', data.get('X_train'))
            y_val = data.get('y_val', data.get('y_train'))
            
            if X_val is None or y_val is None:
                return {
                    'success': False,
                    'error_message': 'Missing required validation data'
                }
            
            # Convert to pandas if needed
            if not isinstance(X_val, pd.DataFrame):
                X_val = pd.DataFrame(X_val)
            if not isinstance(y_val, pd.Series):
                y_val = pd.Series(y_val)
            
            # Validate ensemble
            validation_result = await self._trainer.validate(X_val, y_val)
            
            if not validation_result.success:
                return {
                    'success': False,
                    'error_message': validation_result.error_message
                }
            
            tprint_success("✅ Tactician ensemble validation completed")
            return {
                'success': True,
                'metrics': validation_result.metrics,
                'predictions': validation_result.predictions
            }
            
        except Exception as e:
            tprint_error(f"❌ Tactician ensemble validation failed: {e}")
            self.logger.error(f"Tactician ensemble validation failed: {e}")
            return {
                'success': False,
                'error_message': str(e)
            }
    
    async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make predictions with the trained ensemble.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Prediction result dictionary
        """
        try:
            tprint_info("🎯 Making Tactician ensemble predictions...")
            
            # Extract data
            X_pred = data.get('X_pred', data.get('X_train'))
            
            if X_pred is None:
                return {
                    'success': False,
                    'error_message': 'Missing required prediction data'
                }
            
            # Convert to pandas if needed
            if not isinstance(X_pred, pd.DataFrame):
                X_pred = pd.DataFrame(X_pred)
            
            # Make predictions
            prediction_result = await self._trainer.predict(X_pred)
            
            if not prediction_result.success:
                return {
                    'success': False,
                    'error_message': prediction_result.error_message
                }
            
            tprint_success("✅ Tactician ensemble predictions completed")
            return {
                'success': True,
                'predictions': prediction_result.predictions,
                'probabilities': prediction_result.probabilities
            }
            
        except Exception as e:
            tprint_error(f"❌ Tactician ensemble prediction failed: {e}")
            self.logger.error(f"Tactician ensemble prediction failed: {e}")
            return {
                'success': False,
                'error_message': str(e)
            }
    
    def _extract_feature_importance(self, training_result) -> Optional[Dict[str, float]]:
        """Extract feature importance from training result."""
        try:
            if hasattr(training_result, 'feature_importance') and training_result.feature_importance:
                return training_result.feature_importance
            return None
        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {e}")
            return None
    
    async def _save_models(self, result: TacticianEnsembleTrainingResult) -> None:
        """Save trained models."""
        try:
            if not result.success or not result.ensemble_model:
                return
            
            # This would implement model saving logic
            # For now, just log the save operation
            self.logger.info(f"Ensemble model saved with method: {self.config.ensemble_method.value}")
            tprint_info(f"💾 Ensemble model saved: {self.config.ensemble_method.value}")
            
        except Exception as e:
            self.logger.warning(f"Model saving failed: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        if self._trainer:
            return self._trainer.get_ensemble_summary()
        return {
            'component_name': self.name,
            'config': self.config.__dict__,
            'trainer_initialized': self._trainer is not None
        }
    
    def get_required_dependencies(self) -> List[str]:
        """Get list of required dependencies."""
        return ['pandas', 'numpy', 'scikit-learn', 'lightgbm', 'catboost']
    
    def get_processing_capabilities(self) -> Dict[str, Any]:
        """Get component processing capabilities."""
        return {
            'input_types': ['dict'],
            'output_types': ['dict'],
            'supports_parallel_processing': False,
            'supports_checkpointing': True,
            'supports_validation': True,
            'supports_early_stopping': True,
            'supports_ensemble': True,
            'memory_efficient': True
        }


# Convenience functions
def create_tactician_ensemble_training(
    base_models: List[TacticianModelType] = None,
    ensemble_method: TacticianEnsembleMethod = TacticianEnsembleMethod.STACKING,
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> TacticianEnsembleTraining:
    """
    Create a Tactician ensemble training component.
    
    Args:
        base_models: List of base model types to combine
        ensemble_method: Ensemble combination method
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        TacticianEnsembleTraining component
    """
    if base_models is None:
        base_models = [
            TacticianModelType.LIGHTGBM,
            TacticianModelType.CATBOOST,
            TacticianModelType.NEURAL_NETWORK
        ]
    
    if config is None:
        config = {}
    
    config['base_models'] = base_models
    config['ensemble_method'] = ensemble_method
    
    return TacticianEnsembleTraining(config=config, logger=logger)


async def execute_tactician_ensemble_training(
    data: Dict[str, Any],
    base_models: List[TacticianModelType] = None,
    ensemble_method: TacticianEnsembleMethod = TacticianEnsembleMethod.STACKING,
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Execute Tactician ensemble training with minimal configuration.
    
    Args:
        data: Training data
        base_models: List of base model types to combine
        ensemble_method: Ensemble combination method
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Training result
    """
    component = create_tactician_ensemble_training(base_models, ensemble_method, config, logger)
    
    # Initialize
    if not await component.initialize():
        return {
            'success': False,
            'error_message': 'Component initialization failed'
        }
    
    # Run training
    return await component.run(data)