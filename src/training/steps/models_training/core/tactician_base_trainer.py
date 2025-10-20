"""
Tactician Base Trainer - Unified Training Architecture

This module provides the base trainer class for all Tactician model training,
consolidating common functionality and providing a unified interface.

Key Features:
- Unified training interface for all Tactician model types
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

from .base_trainer import BaseTrainer, TrainingConfig, TrainingResult, ValidationResult, PredictionResult, TrainingRole, ModelType
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_debug, tprint_performance
from src.core.decorators import handles_errors, traced, log_execution_time


class TacticianModelType(Enum):
    """Types of Tactician models."""
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    NEURAL_NETWORK = "neural_network"
    LINEAR = "linear"


@dataclass
class TacticianTrainingConfig(TrainingConfig):
    """Tactician-specific training configuration."""
    # Tactician-specific parameters
    enable_entry_timing: bool = True
    enable_exit_timing: bool = True
    enable_position_sizing: bool = True
    
    # Model-specific parameters
    lightgbm_params: Dict[str, Any] = field(default_factory=dict)
    catboost_params: Dict[str, Any] = field(default_factory=dict)
    neural_network_params: Dict[str, Any] = field(default_factory=dict)
    linear_params: Dict[str, Any] = field(default_factory=dict)
    
    # Timing parameters
    entry_lookback: int = 10
    exit_lookback: int = 5
    position_sizing_lookback: int = 20
    
    # Validation parameters
    tactician_validation_split: float = 0.2
    tactician_cv_folds: int = 5


class TacticianBaseTrainer(BaseTrainer):
    """
    Base trainer for all Tactician model training.
    
    This class provides a unified interface for training different types of Tactician models
    while maintaining consistent patterns for configuration, validation, and error handling.
    """
    
    def __init__(self, config: TacticianTrainingConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the Tactician base trainer.
        
        Args:
            config: Tactician training configuration
            logger: Logger instance (optional)
        """
        # Set role to TACTICIAN
        config.role = TrainingRole.TACTICIAN
        
        super().__init__(config, logger)
        
        # Tactician-specific state
        self._tactician_state = {
            'entry_timing_features_created': False,
            'exit_timing_features_created': False,
            'position_sizing_features_created': False,
            'timing_features_completed': False
        }
        
        tprint_info(f"🔧 Initialized TacticianBaseTrainer for {config.timeframe}")
        self.logger.info(f"Initialized TacticianBaseTrainer for {config.timeframe}")
    
    @handles_errors(
        exceptions=(ValueError, RuntimeError, MemoryError),
        default_return=TrainingResult(
            success=False,
            error_message="Tactician training failed"
        ),
        context="tactician training"
    )
    @log_execution_time
    async def train(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> TrainingResult:
        """
        Train Tactician models with given data.
        
        Args:
            data: Training data
            targets: Target variables (optional, can be inferred from data)
            
        Returns:
            Training result with model and metrics
        """
        try:
            tprint_info("⚔️ Starting Tactician model training...")
            self.logger.info("Starting Tactician model training...")
            
            start_time = time.time()
            
            # Preprocess data
            processed_data, processed_targets = self._preprocess_data(data, targets)
            
            # Create features
            feature_data = await self._create_tactician_features(processed_data)
            
            # Train models
            training_results = {}
            for model_type in self.config.model_types:
                tprint_info(f"🔧 Training {model_type.value} model...")
                
                model_result = await self._train_single_model(
                    model_type, feature_data, processed_targets
                )
                training_results[model_type.value] = model_result
                
                if model_result.success:
                    tprint_success(f"✅ {model_type.value} model trained successfully")
                else:
                    tprint_error(f"❌ {model_type.value} model training failed: {model_result.error_message}")
            
            # Calculate overall metrics
            overall_metrics = self._calculate_overall_metrics(training_results)
            
            # Update training state
            self._training_state['training_completed'] = True
            self._training_state['training_started'] = True
            
            training_time = time.time() - start_time
            self._update_performance_metrics('training', training_time)
            
            # Create result
            result = TrainingResult(
                success=True,
                model=training_results,
                metrics=overall_metrics,
                training_time=training_time,
                metadata={
                    'tactician_state': self._tactician_state.copy(),
                    'models_trained': list(training_results.keys()),
                    'timing_features_completed': self._tactician_state['timing_features_completed']
                }
            )
            
            tprint_success(f"✅ Tactician training completed in {training_time:.2f}s")
            self.logger.info(f"Tactician training completed in {training_time:.2f}s")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Tactician training failed: {e}")
            self.logger.error(f"Tactician training failed: {e}")
            return TrainingResult(
                success=False,
                error_message=str(e),
                training_time=time.time() - start_time if 'start_time' in locals() else 0.0
            )
    
    async def validate(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> ValidationResult:
        """
        Validate the trained Tactician models.
        
        Args:
            data: Validation data
            targets: Target variables (optional)
            
        Returns:
            Validation result with metrics
        """
        try:
            tprint_info("⚔️ Validating Tactician models...")
            self.logger.info("Validating Tactician models...")
            
            start_time = time.time()
            
            # Preprocess data
            processed_data, processed_targets = self._preprocess_data(data, targets)
            
            # Create features
            feature_data = await self._create_tactician_features(processed_data)
            
            # Validate models
            validation_metrics = {}
            for model_type in self.config.model_types:
                model_key = f"{model_type.value}_model"
                if model_key in self._model_state:
                    model = self._model_state[model_key]
                    metrics = await self._validate_single_model(model, model_type, feature_data, processed_targets)
                    validation_metrics[model_type.value] = metrics
            
            validation_time = time.time() - start_time
            self._update_performance_metrics('validation', validation_time)
            
            result = ValidationResult(
                success=True,
                metrics=validation_metrics,
                metadata={
                    'validation_time': validation_time,
                    'models_validated': list(validation_metrics.keys())
                }
            )
            
            tprint_success(f"✅ Tactician validation completed in {validation_time:.2f}s")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Tactician validation failed: {e}")
            self.logger.error(f"Tactician validation failed: {e}")
            return ValidationResult(
                success=False,
                error_message=str(e)
            )
    
    async def predict(self, data: pd.DataFrame) -> PredictionResult:
        """
        Make predictions with the trained Tactician models.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Prediction result
        """
        try:
            tprint_info("⚔️ Making Tactician predictions...")
            self.logger.info("Making Tactician predictions...")
            
            start_time = time.time()
            
            # Preprocess data
            processed_data, _ = self._preprocess_data(data, None)
            
            # Create features
            feature_data = await self._create_tactician_features(processed_data)
            
            # Make predictions
            predictions = {}
            probabilities = {}
            
            for model_type in self.config.model_types:
                model_key = f"{model_type.value}_model"
                if model_key in self._model_state:
                    model = self._model_state[model_key]
                    pred, prob = await self._predict_single_model(model, model_type, feature_data)
                    predictions[model_type.value] = pred
                    if prob is not None:
                        probabilities[model_type.value] = prob
            
            prediction_time = time.time() - start_time
            self._update_performance_metrics('prediction', prediction_time)
            
            result = PredictionResult(
                success=True,
                predictions=predictions,
                probabilities=probabilities if probabilities else None,
                metadata={
                    'prediction_time': prediction_time,
                    'models_used': list(predictions.keys())
                }
            )
            
            tprint_success(f"✅ Tactician predictions completed in {prediction_time:.2f}s")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Tactician prediction failed: {e}")
            self.logger.error(f"Tactician prediction failed: {e}")
            return PredictionResult(
                success=False,
                error_message=str(e)
            )
    
    def _create_model(self, model_type: ModelType) -> Any:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model to create
            
        Returns:
            Model instance
        """
        try:
            if model_type == ModelType.LIGHTGBM:
                return self._create_lightgbm_model()
            elif model_type == ModelType.CATBOOST:
                return self._create_catboost_model()
            elif model_type == ModelType.NEURAL_NETWORK:
                return self._create_neural_network_model()
            elif model_type == ModelType.LINEAR:
                return self._create_linear_model()
            else:
                raise ValueError(f"Unsupported model type for Tactician: {model_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to create {model_type.value} model: {e}")
            return None
    
    def _get_feature_importance(self, model: Any) -> Optional[Dict[str, float]]:
        """
        Extract feature importance from model.
        
        Args:
            model: Trained model
            
        Returns:
            Feature importance dictionary
        """
        try:
            if hasattr(model, 'feature_importances_'):
                return dict(zip(model.feature_names_in_, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                return dict(zip(model.feature_names_in_, model.coef_))
            elif hasattr(model, 'get_feature_importance'):
                return model.get_feature_importance()
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {e}")
            return None
    
    async def _create_tactician_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create Tactician-specific features.
        
        Args:
            data: Input data
            
        Returns:
            Data with Tactician features
        """
        try:
            tprint_info("🔧 Creating Tactician features...")
            
            feature_data = data.copy()
            
            # Create entry timing features if enabled
            if self.config.enable_entry_timing:
                feature_data = await self._create_entry_timing_features(feature_data)
                self._tactician_state['entry_timing_features_created'] = True
            
            # Create exit timing features if enabled
            if self.config.enable_exit_timing:
                feature_data = await self._create_exit_timing_features(feature_data)
                self._tactician_state['exit_timing_features_created'] = True
            
            # Create position sizing features if enabled
            if self.config.enable_position_sizing:
                feature_data = await self._create_position_sizing_features(feature_data)
                self._tactician_state['position_sizing_features_created'] = True
            
            self._tactician_state['timing_features_completed'] = True
            tprint_success(f"✅ Created {feature_data.shape[1]} features")
            
            return feature_data
            
        except Exception as e:
            tprint_error(f"❌ Feature creation failed: {e}")
            self.logger.error(f"Feature creation failed: {e}")
            return data
    
    async def _create_entry_timing_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create entry timing features."""
        # Placeholder implementation
        # In a real implementation, this would create entry timing features
        return data
    
    async def _create_exit_timing_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create exit timing features."""
        # Placeholder implementation
        # In a real implementation, this would create exit timing features
        return data
    
    async def _create_position_sizing_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create position sizing features."""
        # Placeholder implementation
        # In a real implementation, this would create position sizing features
        return data
    
    def _create_lightgbm_model(self):
        """Create LightGBM model."""
        try:
            import lightgbm as lgb
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                **self.config.lightgbm_params
            }
            
            return lgb.LGBMRegressor(**params)
            
        except ImportError:
            self.logger.error("LightGBM not available")
            return None
    
    def _create_catboost_model(self):
        """Create CatBoost model."""
        try:
            import catboost as cb
            
            params = {
                'iterations': 1000,
                'learning_rate': 0.1,
                'depth': 6,
                'verbose': False,
                **self.config.catboost_params
            }
            
            return cb.CatBoostRegressor(**params)
            
        except ImportError:
            self.logger.error("CatBoost not available")
            return None
    
    def _create_neural_network_model(self):
        """Create neural network model."""
        # Placeholder implementation
        # In a real implementation, this would create a neural network
        return None
    
    def _create_linear_model(self):
        """Create linear model."""
        try:
            from sklearn.linear_model import LinearRegression
            
            params = {
                **self.config.linear_params
            }
            
            return LinearRegression(**params)
            
        except ImportError:
            self.logger.error("scikit-learn not available")
            return None
    
    async def _train_single_model(self, model_type: ModelType, data: pd.DataFrame, targets: pd.Series) -> TrainingResult:
        """Train a single model."""
        try:
            model = self._create_model(model_type)
            if model is None:
                return TrainingResult(
                    success=False,
                    error_message=f"Failed to create {model_type.value} model"
                )
            
            # Train the model
            model.fit(data, targets)
            
            # Store model
            model_key = f"{model_type.value}_model"
            self._model_state[model_key] = model
            
            # Get feature importance
            feature_importance = self._get_feature_importance(model)
            
            return TrainingResult(
                success=True,
                model=model,
                feature_importance=feature_importance,
                metadata={'model_type': model_type.value}
            )
            
        except Exception as e:
            return TrainingResult(
                success=False,
                error_message=str(e)
            )
    
    async def _validate_single_model(self, model: Any, model_type: ModelType, data: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        """Validate a single model."""
        try:
            predictions = model.predict(data)
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics = {
                'mse': mean_squared_error(targets, predictions),
                'mae': mean_absolute_error(targets, predictions),
                'r2': r2_score(targets, predictions),
                'rmse': np.sqrt(mean_squared_error(targets, predictions))
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Validation failed for {model_type.value}: {e}")
            return {}
    
    async def _predict_single_model(self, model: Any, model_type: ModelType, data: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions with a single model."""
        try:
            predictions = model.predict(data)
            
            # For regression models, we don't typically have probabilities
            # But we can calculate confidence intervals or uncertainty estimates
            probabilities = None
            
            return predictions, probabilities
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {model_type.value}: {e}")
            return np.array([]), None
    
    def _calculate_overall_metrics(self, training_results: Dict[str, TrainingResult]) -> Dict[str, float]:
        """Calculate overall training metrics."""
        successful_models = [r for r in training_results.values() if r.success]
        
        return {
            'total_models': len(training_results),
            'successful_models': len(successful_models),
            'success_rate': len(successful_models) / len(training_results) if training_results else 0.0
        }
    
    def get_tactician_summary(self) -> Dict[str, Any]:
        """Get Tactician-specific training summary."""
        base_summary = self.get_training_summary()
        base_summary.update({
            'tactician_state': self._tactician_state.copy(),
            'timing_features': {
                'entry_timing': self._tactician_state['entry_timing_features_created'],
                'exit_timing': self._tactician_state['exit_timing_features_created'],
                'position_sizing': self._tactician_state['position_sizing_features_created'],
                'completed': self._tactician_state['timing_features_completed']
            }
        })
        return base_summary