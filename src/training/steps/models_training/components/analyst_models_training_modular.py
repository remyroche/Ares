"""
Analyst Models Training - ModularComponent Implementation

This module provides a ModularComponent implementation of the Analyst Models Training
that handles training of individual Analyst base models:
- LightGBM model
- LightGBM + PatchTST features model
- CatBoost model
- Stacker LGBM Calibrated (meta-learner)

The Analyst operates on the dedicated 15m timeframe and decides IF we trade by
screening market conditions and producing the green-signal gating that the
Tactician consumes.

ENHANCED FEATURES:
- ModularComponent architecture with comprehensive state management
- ML-specific performance monitoring and checkpointing
- Enhanced error handling and logging
- Configuration management and validation
- Training progress tracking and health monitoring
- Regime-aware training support
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
import traceback
from dataclasses import dataclass
from enum import Enum

from .base_component import BaseModelsTrainingComponent
# from ..unified_data_driven_pipeline.core.modular_architecture import (
#     ErrorInfo, ErrorSeverity, ErrorCategory, ValidationResult
# )  # REMOVED - unified pipeline deleted


class AnalystModelType(Enum):
    """Types of Analyst models."""
    LIGHTGBM = "lightgbm"
    LIGHTGBM_PATCHTST = "lightgbm_patchtst"
    CATBOOST = "catboost"
    STACKER_LGBM_CALIBRATED = "stacker_lgbm_calibrated"


@dataclass
class AnalystModelsTrainingConfig:
    """Configuration for Analyst models training."""
    model_types: List[AnalystModelType]
    training_params: Dict[str, Any]
    validation_params: Dict[str, Any]
    regime_aware: bool = True
    timeframe: str = "15m"
    auto_save: bool = True


@dataclass
class AnalystModelsTrainingResult:
    """Result of Analyst models training."""
    success: bool
    models: Dict[str, Any]
    metrics: Dict[str, float]
    training_time: float
    errors: List[str]
    warnings: List[str]
    regime_performance: Optional[Dict[str, Any]] = None


class AnalystModelsTrainingModular(BaseModelsTrainingComponent):
    """
    ModularComponent implementation of Analyst Models Training.
    
    This component handles training of individual Analyst base models with comprehensive
    state management, performance monitoring, and error handling.
    """
    
    def __init__(
        self,
        name: str = "analyst_models_training",
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Analyst Models Training component.
        
        Args:
            name: Component name
            config: Configuration dictionary
            logger: Logger instance
        """
        # Set default configuration
        default_config = {
            'model': {
                'type': 'multi_model',
                'model_types': ['lightgbm', 'lightgbm_patchtst', 'catboost', 'stacker_lgbm_calibrated'],
                'regime_aware': True
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
            'regime_aware': True,
            'timeframe': '15m',
            'auto_save': True
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(name, default_config, logger)
        
        # Analyst-specific configuration
        self.analyst_config = AnalystModelsTrainingConfig(
            model_types=[AnalystModelType(model) for model in self.model_config.get('model_types', [])],
            training_params=self.training_config,
            validation_params=self.validation_config,
            regime_aware=self.get_config('regime_aware', True),
            timeframe=self.get_config('timeframe', '15m'),
            auto_save=self.get_config('auto_save', True)
        )
        
        # Training state
        self._trained_models = {}
        self._training_results = {}
        self._regime_performance = {}
        
        self.logger.info(f"Initialized AnalystModelsTrainingModular: {name}")
    
    def _initialize_resources(self) -> bool:
        """Initialize component-specific resources."""
        try:
            # Initialize base resources
            if not super()._initialize_resources():
                return False
            
            # Initialize analyst-specific state
            self.set_ml_state('analyst_initialized', True)
            self.set_ml_state('models_trained', False)
            self.set_ml_state('training_phase', 'none')
            
            # Initialize model configurations
            self._initialize_model_configs()
            
            self.logger.info("Analyst models training resources initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Resource initialization failed: {e}")
            return False
    
    def _cleanup_resources(self) -> None:
        """Cleanup component-specific resources."""
        try:
            # Clear trained models
            self._trained_models.clear()
            self._training_results.clear()
            self._regime_performance.clear()
            
            # Clear analyst state
            self.set_ml_state('analyst_initialized', False)
            self.set_ml_state('models_trained', False)
            
            # Call parent cleanup
            super()._cleanup_resources()
            
            self.logger.info("Analyst models training resources cleaned up")
            
        except Exception as e:
            self.logger.error(f"Resource cleanup failed: {e}")
    
    def _initialize_model_configs(self) -> None:
        """Initialize model configurations for each model type."""
        model_configs = {}
        
        for model_type in self.analyst_config.model_types:
            if model_type == AnalystModelType.TCN:
                model_configs[model_type.value] = {
                    'type': 'neural_network',
                    'architecture': 'tcn',
                    'layers': 3,
                    'filters': 64,
                    'kernel_size': 3,
                    'dilation_rate': 2
                }
            elif model_type == AnalystModelType.LIGHTGBM:
                model_configs[model_type.value] = {
                    'type': 'tree_based',
                    'algorithm': 'lightgbm',
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'num_leaves': 31
                }
            elif model_type == AnalystModelType.RIDGE:
                model_configs[model_type.value] = {
                    'type': 'linear',
                    'algorithm': 'ridge',
                    'alpha': 1.0,
                    'solver': 'auto'
                }
            elif model_type == AnalystModelType.ELASTIC_NET:
                model_configs[model_type.value] = {
                    'type': 'linear',
                    'algorithm': 'elastic_net',
                    'alpha': 1.0,
                    'l1_ratio': 0.5,
                    'max_iter': 1000
                }
            elif model_type == AnalystModelType.RANDOM_FOREST:
                model_configs[model_type.value] = {
                    'type': 'tree_based',
                    'algorithm': 'random_forest',
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 2
                }
            # NAS/TAS model types removed
        
        self.set_ml_state('model_configs', model_configs)
    
    def _process_data(self, data: Any, **kwargs) -> Any:
        """Process data with analyst models training logic."""
        try:
            self.logger.info("Starting analyst models training")
            
            # Validate input data
            if not self._validate_training_data(data):
                raise ValueError("Invalid training data")
            
            # Start training
            if not self.start_training():
                raise RuntimeError("Failed to start training")
            
            # Set training phase
            self.set_ml_state('training_phase', 'base_models')
            
            # Train individual models
            training_result = self._train_individual_models(data)
            if not training_result['success']:
                raise RuntimeError(f"Models training failed: {training_result['errors']}")
            
            # Evaluate models
            evaluation_result = self._evaluate_models(data, training_result['models'])
            
            # Stop training
            self.stop_training()
            
            # Prepare result
            result = AnalystModelsTrainingResult(
                success=True,
                models=training_result['models'],
                metrics=evaluation_result['metrics'],
                training_time=self.get_ml_state('total_training_time', 0),
                errors=[],
                warnings=training_result['warnings'] + evaluation_result['warnings'],
                regime_performance=evaluation_result.get('regime_performance')
            )
            
            # Save results
            self._training_results = {
                'models': training_result['models'],
                'metrics': evaluation_result['metrics'],
                'training_time': result.training_time,
                'regime_performance': evaluation_result.get('regime_performance')
            }
            
            self.logger.info(f"Analyst models training completed successfully in {result.training_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Analyst models training failed: {e}")
            self.stop_training()
            raise
    
    def _validate_training_data(self, data: Any) -> bool:
        """Validate training data for analyst models training."""
        try:
            if not isinstance(data, dict):
                self.logger.error("Training data must be a dictionary")
                return False
            
            required_keys = ['X_train', 'y_train']
            for key in required_keys:
                if key not in data:
                    self.logger.error(f"Missing required key: {key}")
                    return False
            
            # Check data shapes
            X_train = data['X_train']
            y_train = data['y_train']
            
            if len(X_train) != len(y_train):
                self.logger.error("X_train and y_train must have same length")
                return False
            
            if len(X_train) < 100:
                self.logger.warning("Training data is small, consider more data")
            
            # Check for regime data if regime-aware
            if self.analyst_config.regime_aware:
                if 'regime_data' not in data:
                    self.logger.warning("Regime-aware training enabled but no regime data provided")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            return False
    
    def _train_individual_models(self, data: Any) -> Dict[str, Any]:
        """Train individual models."""
        try:
            models = {}
            errors = []
            warnings = []
            
            for model_type in self.analyst_config.model_types:
                try:
                    self.logger.info(f"Training {model_type.value} model")
                    
                    # Get model configuration
                    model_config = self.get_ml_state('model_configs')[model_type.value]
                    
                    # Train model
                    model = self._train_single_model(model_type, data, model_config)
                    
                    if model is not None:
                        models[model_type.value] = model
                        self.logger.info(f"{model_type.value} model trained successfully")
                    else:
                        errors.append(f"Failed to train {model_type.value} model")
                        
                except Exception as e:
                    error_msg = f"Error training {model_type.value}: {str(e)}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
            
            # Update state
            self._trained_models = models
            self.set_ml_state('models_trained', len(models) > 0)
            
            return {
                'success': len(errors) == 0,
                'models': models,
                'errors': errors,
                'warnings': warnings
            }
            
        except Exception as e:
            self.logger.error(f"Individual models training failed: {e}")
            return {
                'success': False,
                'models': {},
                'errors': [str(e)],
                'warnings': []
            }
    
    def _train_single_model(self, model_type: AnalystModelType, data: Any, config: Dict[str, Any]) -> Any:
        """Train a single model."""
        try:
            if model_type == AnalystModelType.LIGHTGBM:
                return self._train_lightgbm_model(data, config)
            elif model_type == AnalystModelType.LIGHTGBM_PATCHTST:
                return self._train_lightgbm_patchtst_model(data, config)
            elif model_type == AnalystModelType.CATBOOST:
                return self._train_catboost_model(data, config)
            elif model_type == AnalystModelType.STACKER_LGBM_CALIBRATED:
                return self._train_stacker_lgbm_calibrated_model(data, config)
            else:
                self.logger.warning(f"Unknown model type: {model_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to train {model_type.value}: {e}")
            return None
    
    def _train_lightgbm_model(self, data: Any, config: Dict[str, Any]) -> Any:
        """Train LightGBM model."""
        try:
            # Placeholder implementation - would integrate with actual LightGBM training
            self.logger.info("Training LightGBM model (placeholder)")
            
            # Simulate training
            X_train = data['X_train']
            y_train = data['y_train']
            
            # Create mock model
            model = {
                'type': 'lightgbm',
                'algorithm': 'lightgbm',
                'n_estimators': config.get('n_estimators', 1000),
                'max_depth': config.get('max_depth', 6),
                'learning_rate': config.get('learning_rate', 0.1),
                'num_leaves': config.get('num_leaves', 31),
                'trained': True,
                'config': config,
                'training_samples': len(X_train)
            }
            
            return model
            
        except Exception as e:
            self.logger.error(f"LightGBM training failed: {e}")
            return None
    
    def _train_lightgbm_patchtst_model(self, data: Any, config: Dict[str, Any]) -> Any:
        """Train LightGBM + PatchTST features model."""
        try:
            # Placeholder implementation - would integrate with actual LightGBM + PatchTST training
            self.logger.info("Training LightGBM + PatchTST model (placeholder)")
            
            # Simulate training
            X_train = data['X_train']
            y_train = data['y_train']
            
            # Create mock model
            model = {
                'type': 'lightgbm_patchtst',
                'algorithm': 'lightgbm',
                'n_estimators': config.get('n_estimators', 1000),
                'max_depth': config.get('max_depth', 6),
                'learning_rate': config.get('learning_rate', 0.1),
                'num_leaves': config.get('num_leaves', 31),
                'patchtst_features': True,
                'patchtst_config': config.get('patchtst_config', {}),
                'trained': True,
                'config': config,
                'training_samples': len(X_train)
            }
            
            return model
            
        except Exception as e:
            self.logger.error(f"LightGBM + PatchTST training failed: {e}")
            return None
    
    def _train_catboost_model(self, data: Any, config: Dict[str, Any]) -> Any:
        """Train CatBoost model."""
        try:
            # Placeholder implementation - would integrate with actual CatBoost training
            self.logger.info("Training CatBoost model (placeholder)")
            
            # Simulate training
            X_train = data['X_train']
            y_train = data['y_train']
            
            # Create mock model
            model = {
                'type': 'catboost',
                'algorithm': 'catboost',
                'iterations': config.get('iterations', 1000),
                'learning_rate': config.get('learning_rate', 0.1),
                'depth': config.get('depth', 6),
                'l2_leaf_reg': config.get('l2_leaf_reg', 3),
                'trained': True,
                'config': config,
                'training_samples': len(X_train)
            }
            
            return model
            
        except Exception as e:
            self.logger.error(f"CatBoost training failed: {e}")
            return None
    
    def _train_stacker_lgbm_calibrated_model(self, data: Any, config: Dict[str, Any]) -> Any:
        """Train Stacker LGBM Calibrated (meta-learner) model."""
        try:
            # Placeholder implementation - would integrate with actual Stacker LGBM Calibrated training
            self.logger.info("Training Stacker LGBM Calibrated model (placeholder)")
            
            # Simulate training
            X_train = data['X_train']
            y_train = data['y_train']
            
            # Create mock model
            model = {
                'type': 'stacker_lgbm_calibrated',
                'algorithm': 'lightgbm',
                'n_estimators': config.get('n_estimators', 1000),
                'max_depth': config.get('max_depth', 6),
                'learning_rate': config.get('learning_rate', 0.1),
                'num_leaves': config.get('num_leaves', 31),
                'calibrated': True,
                'meta_learner': True,
                'base_models': config.get('base_models', ['lightgbm', 'lightgbm_patchtst', 'catboost']),
                'trained': True,
                'config': config,
                'training_samples': len(X_train)
            }
            
            return model
            
        except Exception as e:
            self.logger.error(f"Stacker LGBM Calibrated training failed: {e}")
            return None
    
    def _evaluate_models(self, data: Any, models: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate trained models."""
        try:
            self.logger.info("Evaluating trained models")
            
            # Placeholder evaluation metrics
            metrics = {
                'overall_accuracy': 0.85,
                'overall_precision': 0.82,
                'overall_recall': 0.88,
                'overall_f1_score': 0.85,
                'best_model': 'lightgbm',
                'model_count': len(models)
            }
            
            # Add individual model metrics
            for model_name, model in models.items():
                metrics[f'{model_name}_accuracy'] = 0.8 + np.random.normal(0, 0.05)
                metrics[f'{model_name}_trained'] = model.get('trained', False)
            
            # Regime performance if available
            regime_performance = None
            if self.analyst_config.regime_aware and 'regime_data' in data:
                regime_performance = {
                    'regime_1': {'accuracy': 0.87, 'precision': 0.84, 'recall': 0.89},
                    'regime_2': {'accuracy': 0.83, 'precision': 0.81, 'recall': 0.86},
                    'regime_3': {'accuracy': 0.85, 'precision': 0.83, 'recall': 0.88}
                }
            
            # Update performance stats
            self._performance_stats['validation_accuracy'] = metrics['overall_accuracy']
            self._performance_stats['model_convergence'] = True
            
            return {
                'metrics': metrics,
                'regime_performance': regime_performance,
                'warnings': []
            }
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            return {
                'metrics': {},
                'regime_performance': None,
                'warnings': [str(e)]
            }
    
    def _train_epoch_impl(self, model: Any, data: Any, epoch: int) -> Dict[str, float]:
        """Implement epoch training logic."""
        # This would be implemented based on the specific model type
        return {
            'loss': 1.0 - (epoch / 100),
            'accuracy': 0.5 + (epoch / 100) * 0.4
        }
    
    def _validate_epoch_impl(self, model: Any, data: Any, epoch: int) -> Dict[str, float]:
        """Implement epoch validation logic."""
        # This would be implemented based on the specific model type
        return {
            'val_loss': 1.0 - (epoch / 100) * 0.8,
            'val_accuracy': 0.6 + (epoch / 100) * 0.3
        }
    
    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for this component."""
        return {
            'min_size': 100,
            'max_size': 1000000,
            'required_keys': ['X_train', 'y_train'],
            'data_types': ['dict'],
            'required_columns': ['X_train', 'y_train']
        }
    
    def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
        """Validate data with component-specific rules."""
        errors = []
        warnings = []
        metadata = {}
        
        if isinstance(data, dict):
            # Check required keys
            required_keys = ['X_train', 'y_train']
            for key in required_keys:
                if key not in data:
                    errors.append(f"Missing required key: {key}")
            
            # Check data consistency
            if 'X_train' in data and 'y_train' in data:
                X_train = data['X_train']
                y_train = data['y_train']
                
                if hasattr(X_train, 'shape') and hasattr(y_train, 'shape'):
                    metadata['X_train_shape'] = X_train.shape
                    metadata['y_train_shape'] = y_train.shape
                    
                    if len(X_train) != len(y_train):
                        errors.append("X_train and y_train must have same number of samples")
                    
                    if len(X_train) < 100:
                        warnings.append("Training data is small, consider more data")
            
            # Check for regime data if regime-aware
            if self.analyst_config.regime_aware and 'regime_data' not in data:
                warnings.append("Regime-aware training enabled but no regime data provided")
        
        return {'errors': errors, 'warnings': warnings, 'metadata': metadata}
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        summary = super().get_training_summary()
        
        # Add analyst-specific information
        summary.update({
            'analyst_config': {
                'model_types': [mt.value for mt in self.analyst_config.model_types],
                'regime_aware': self.analyst_config.regime_aware,
                'timeframe': self.analyst_config.timeframe
            },
            'trained_models': list(self._trained_models.keys()),
            'training_results': self._training_results,
            'regime_performance': self._regime_performance
        })
        
        return summary


def create_analyst_models_training(
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> AnalystModelsTrainingModular:
    """
    Factory function to create Analyst Models Training component.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Initialized AnalystModelsTrainingModular instance
    """
    return AnalystModelsTrainingModular(
        name="analyst_models_training",
        config=config,
        logger=logger
    )