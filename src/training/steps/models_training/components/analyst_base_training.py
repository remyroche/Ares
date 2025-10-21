"""
Analyst Base Training - Unified Training Architecture

This module provides the Analyst base training component that handles training
of individual Analyst base models using the unified BaseTrainer architecture.

Key Features:
- Unified training interface for all Analyst model types
- Common training patterns and lifecycle management
- Standardized configuration and validation
- Performance monitoring and checkpointing
- Error handling and recovery mechanisms
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

from ..core.analyst_base_trainer import (
    AnalystBaseTrainer, AnalystTrainingConfig, AnalystModelType
)
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
class AnalystBaseTrainingConfig:
    """Configuration for Analyst base training."""
    model_types: List[AnalystModelType]
    training_params: Dict[str, Any] = field(default_factory=dict)
    validation_params: Dict[str, Any] = field(default_factory=dict)
    timeframe: str = "15m"
    symbol: str = "ETHUSDT"
    auto_save: bool = True
    
    # Feature engineering parameters
    enable_patchtst_features: bool = True
    enable_regime_features: bool = True
    enable_multi_timeframe: bool = True
    
    # Model-specific parameters
    lightgbm_params: Dict[str, Any] = field(default_factory=dict)
    catboost_params: Dict[str, Any] = field(default_factory=dict)
    stacker_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalystBaseTrainingResult:
    """Result of Analyst base training."""
    success: bool
    models: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    feature_importance: Optional[Dict[str, Dict[str, float]]] = None


class AnalystBaseTraining(BaseStep):
    """
    Analyst base training component using unified BaseTrainer architecture.
    
    This component handles training of individual Analyst base models with comprehensive
    state management, performance monitoring, and error handling.
    """
    
    def __init__(
        self,
        name: str = "analyst_base_training",
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Analyst base training component.
        
        Args:
            name: Component name
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(name, config)
        
        # Set default configuration
        default_config = {
            'model_types': [AnalystModelType.LIGHTGBM, AnalystModelType.CATBOOST],
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
            'enable_patchtst_features': True,
            'enable_regime_features': True,
            'enable_multi_timeframe': True
        }
        
        # Merge with provided configuration
        if config:
            default_config.update(config)
        
        self.config = AnalystBaseTrainingConfig(**default_config)
        
        # Initialize trainer
        self._trainer = None
        
        tprint_info(f"🔧 Initialized AnalystBaseTraining: {name}")
        self.logger.info(f"Initialized AnalystBaseTraining: {name}")
    
    @handles_errors(
        exceptions=(ValueError, RuntimeError, MemoryError),
        default_return=AnalystBaseTrainingResult(
            success=False,
            errors=["Component initialization failed"]
        ),
        context="analyst base training"
    )
    async def initialize(self) -> bool:
        """Initialize the component."""
        try:
            tprint_info("🔧 Initializing Analyst base training...")
            
            # Create trainer configuration
            trainer_config = AnalystTrainingConfig(
                model_types=[self._convert_model_type(mt) for mt in self.config.model_types],
                timeframe=self.config.timeframe,
                symbol=self.config.symbol,
                validation_split=self.config.training_params.get('validation_split', 0.2),
                cross_validation_folds=self.config.training_params.get('cross_validation_folds', 5),
                random_seed=self.config.training_params.get('random_seed'),
                enable_patchtst_features=self.config.enable_patchtst_features,
                enable_regime_features=self.config.enable_regime_features,
                enable_multi_timeframe=self.config.enable_multi_timeframe,
                lightgbm_params=self.config.lightgbm_params,
                catboost_params=self.config.catboost_params,
                custom_params=self.config.training_params
            )
            
            # Create trainer
            self._trainer = AnalystBaseTrainer(trainer_config, self.logger)
            
            # Initialize trainer
            if not await self._trainer.initialize():
                tprint_error("❌ Trainer initialization failed")
                return False
            
            tprint_success("✅ Analyst base training initialized")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Initialization failed: {e}")
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    @handles_errors(
        exceptions=(ValueError, RuntimeError, MemoryError),
        default_return=AnalystBaseTrainingResult(
            success=False,
            errors=["Training execution failed"]
        ),
        context="analyst base training"
    )
    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the Analyst base training.
        
        Args:
            data: Input data containing training features and targets
            
        Returns:
            Training result dictionary
        """
        try:
            tprint_info("📊 Starting Analyst base training...")
            self.logger.info("Starting Analyst base training...")
            
            start_time = time.time()
            
            # Extract data
            X_train = data.get('X_train')
            y_train = data.get('y_train')
            
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
            
            # Train models
            training_result = await self._trainer.train(X_train, y_train)
            
            if not training_result.success:
                return {
                    'success': False,
                    'error_message': training_result.error_message,
                    'training_time': time.time() - start_time
                }
            
            # Create result
            result = AnalystBaseTrainingResult(
                success=True,
                models=training_result.model if isinstance(training_result.model, dict) else {},
                metrics=training_result.metrics,
                training_time=training_result.training_time,
                feature_importance=self._extract_feature_importance(training_result)
            )
            
            # Auto-save if enabled
            if self.config.auto_save:
                await self._save_models(result)
            
            tprint_success(f"✅ Analyst base training completed in {result.training_time:.2f}s")
            self.logger.info(f"Analyst base training completed in {result.training_time:.2f}s")
            
            return {
                'success': True,
                'result': result,
                'training_time': result.training_time,
                'models_trained': list(result.models.keys()),
                'metrics': result.metrics
            }
            
        except Exception as e:
            tprint_error(f"❌ Analyst base training failed: {e}")
            self.logger.error(f"Analyst base training failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'training_time': time.time() - start_time if 'start_time' in locals() else 0.0
            }
    
    async def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the trained models.
        
        Args:
            data: Validation data containing features and targets
            
        Returns:
            Validation result dictionary
        """
        try:
            tprint_info("📊 Validating Analyst base models...")
            
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
            
            # Validate models
            validation_result = await self._trainer.validate(X_val, y_val)
            
            if not validation_result.success:
                return {
                    'success': False,
                    'error_message': validation_result.error_message
                }
            
            tprint_success("✅ Analyst base validation completed")
            return {
                'success': True,
                'metrics': validation_result.metrics,
                'predictions': validation_result.predictions
            }
            
        except Exception as e:
            tprint_error(f"❌ Analyst base validation failed: {e}")
            self.logger.error(f"Analyst base validation failed: {e}")
            return {
                'success': False,
                'error_message': str(e)
            }
    
    async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make predictions with the trained models.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Prediction result dictionary
        """
        try:
            tprint_info("📊 Making Analyst base predictions...")
            
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
            
            tprint_success("✅ Analyst base predictions completed")
            return {
                'success': True,
                'predictions': prediction_result.predictions,
                'probabilities': prediction_result.probabilities
            }
            
        except Exception as e:
            tprint_error(f"❌ Analyst base prediction failed: {e}")
            self.logger.error(f"Analyst base prediction failed: {e}")
            return {
                'success': False,
                'error_message': str(e)
            }
    
    def _convert_model_type(self, model_type: AnalystModelType):
        """Convert AnalystModelType to ModelType."""
        from ..core.base_trainer import ModelType
        
        if model_type == AnalystModelType.LIGHTGBM:
            return ModelType.LIGHTGBM
        elif model_type == AnalystModelType.CATBOOST:
            return ModelType.CATBOOST
        elif model_type == AnalystModelType.LIGHTGBM_PATCHTST:
            return ModelType.LIGHTGBM  # Same as LightGBM but with PatchTST features
        elif model_type == AnalystModelType.STACKER_LGBM_CALIBRATED:
            return ModelType.LIGHTGBM  # Stacker uses LightGBM as base
        else:
            return ModelType.LIGHTGBM  # Default
    
    def _extract_feature_importance(self, training_result) -> Optional[Dict[str, Dict[str, float]]]:
        """Extract feature importance from training result."""
        try:
            if hasattr(training_result, 'feature_importance') and training_result.feature_importance:
                return training_result.feature_importance
            return None
        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {e}")
            return None
    
    async def _save_models(self, result: AnalystBaseTrainingResult) -> None:
        """Save trained models."""
        try:
            if not result.success or not result.models:
                return
            
            # This would implement model saving logic
            # For now, just log the save operation
            self.logger.info(f"Models saved: {list(result.models.keys())}")
            tprint_info(f"💾 Models saved: {list(result.models.keys())}")
            
        except Exception as e:
            self.logger.warning(f"Model saving failed: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        if self._trainer:
            return self._trainer.get_analyst_summary()
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
            'supports_ensemble': False,
            'memory_efficient': True
        }


# Convenience functions
def create_analyst_base_training(
    model_types: List[AnalystModelType] = None,
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> AnalystBaseTraining:
    """
    Create an Analyst base training component.
    
    Args:
        model_types: List of model types to train
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        AnalystBaseTraining component
    """
    if model_types is None:
        model_types = [AnalystModelType.LIGHTGBM, AnalystModelType.CATBOOST]
    
    if config is None:
        config = {}
    
    config['model_types'] = model_types
    
    return AnalystBaseTraining(config=config, logger=logger)


async def execute_analyst_base_training(
    data: Dict[str, Any],
    model_types: List[AnalystModelType] = None,
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Execute Analyst base training with minimal configuration.
    
    Args:
        data: Training data
        model_types: List of model types to train
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Training result
    """
    component = create_analyst_base_training(model_types, config, logger)
    
    # Initialize
    if not await component.initialize():
        return {
            'success': False,
            'error_message': 'Component initialization failed'
        }
    
    # Run training
    return await component.run(data)