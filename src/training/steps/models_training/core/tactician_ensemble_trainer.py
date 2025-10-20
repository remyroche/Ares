"""
Tactician Ensemble Trainer - Unified Training Architecture

This module provides the ensemble trainer class for Tactician model training,
combining multiple base models for enhanced performance.

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

from .base_trainer import BaseTrainer, TrainingConfig, TrainingResult, ValidationResult, PredictionResult, TrainingRole, ModelType
from .tactician_base_trainer import TacticianBaseTrainer, TacticianTrainingConfig, TacticianModelType
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_debug, tprint_performance
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.hardware.integrated_hardware_manager import (
    get_integrated_hardware_manager, WorkloadType
)
from src.utils.hardware.optimization_decorators import (
    smart_cache, auto_optimize, memory_efficient, performance_tracked
)
from src.utils.hardware.memory_optimized_decorators import (
    memory_optimized, comprehensive_memory_optimization, MemoryOptimizationLevel
)


class TacticianEnsembleMethod(Enum):
    """Tactician ensemble combination methods."""
    VOTING = "voting"
    AVERAGING = "averaging"
    STACKING = "stacking"
    BLENDING = "blending"
    WEIGHTED = "weighted"


@dataclass
class TacticianEnsembleTrainingConfig(TacticianTrainingConfig):
    """Tactician ensemble training configuration."""
    # Ensemble-specific parameters
    ensemble_method: TacticianEnsembleMethod = TacticianEnsembleMethod.STACKING
    base_models: List[TacticianModelType] = field(default_factory=lambda: [
        TacticianModelType.LIGHTGBM,
        TacticianModelType.CATBOOST,
        TacticianModelType.NEURAL_NETWORK
    ])
    
                # Meta-learner parameters
                meta_learner_type: ModelType = ModelType.LIGHTGBM
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
    
    # Ensemble validation
    ensemble_validation_split: float = 0.2
    enable_cross_validation: bool = True
    cv_folds: int = 5


class TacticianEnsembleTrainer(BaseTrainer):
    """
    Ensemble trainer for Tactician models.
    
    This class provides ensemble training capabilities for combining multiple
    Tactician base models for enhanced performance.
    """
    
    def __init__(self, config: TacticianEnsembleTrainingConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the Tactician ensemble trainer.
        
        Args:
            config: Tactician ensemble training configuration
            logger: Logger instance (optional)
        """
        # Set role to ENSEMBLE
        config.role = TrainingRole.ENSEMBLE
        
        super().__init__(config, logger)
        
        # Ensemble-specific state
        self._ensemble_state = {
            'base_models_trained': False,
            'meta_learner_trained': False,
            'ensemble_created': False,
            'base_predictions_generated': False
        }
        
        # Base model trainers
        self._base_trainers = {}
        
        # Ensemble model
        self._ensemble_model = None
        
        tprint_info(f"🔧 Initialized TacticianEnsembleTrainer with {config.ensemble_method.value}")
        self.logger.info(f"Initialized TacticianEnsembleTrainer with {config.ensemble_method.value}")
    
    @handles_errors(
        exceptions=(ValueError, RuntimeError, MemoryError),
        default_return=TrainingResult(
            success=False,
            error_message="Tactician ensemble training failed"
        ),
        context="tactician ensemble training"
    )
    @log_execution_time
    async def train(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> TrainingResult:
        """
        Train Tactician ensemble models.
        
        Args:
            data: Training data
            targets: Target variables (optional, can be inferred from data)
            
        Returns:
            Training result with ensemble model and metrics
        """
        try:
            tprint_info("🎯 Starting Tactician ensemble training...")
            self.logger.info("Starting Tactician ensemble training...")
            
            start_time = time.time()
            
            # Preprocess data
            processed_data, processed_targets = self._preprocess_data(data, targets)
            
            # Train base models
            base_results = await self._train_base_models(processed_data, processed_targets)
            
            if not base_results['success']:
                return TrainingResult(
                    success=False,
                    error_message="Base model training failed",
                    training_time=time.time() - start_time
                )
            
            # Generate base predictions for ensemble training
            base_predictions = await self._generate_base_predictions(processed_data)
            
            # Train ensemble model
            ensemble_result = await self._train_ensemble_model(
                base_predictions, processed_targets
            )
            
            if not ensemble_result['success']:
                return TrainingResult(
                    success=False,
                    error_message="Ensemble model training failed",
                    training_time=time.time() - start_time
                )
            
            # Calculate ensemble metrics
            ensemble_metrics = await self._calculate_ensemble_metrics(
                processed_data, processed_targets
            )
            
            # Update training state
            self._training_state['training_completed'] = True
            self._training_state['training_started'] = True
            self._ensemble_state['ensemble_created'] = True
            
            training_time = time.time() - start_time
            self._update_performance_metrics('training', training_time)
            
            # Create result
            result = TrainingResult(
                success=True,
                model=self._ensemble_model,
                metrics=ensemble_metrics,
                training_time=training_time,
                metadata={
                    'ensemble_state': self._ensemble_state.copy(),
                    'base_models': list(self._base_trainers.keys()),
                    'ensemble_method': self.config.ensemble_method.value,
                    'base_results': base_results
                }
            )
            
            tprint_success(f"✅ Tactician ensemble training completed in {training_time:.2f}s")
            self.logger.info(f"Tactician ensemble training completed in {training_time:.2f}s")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Tactician ensemble training failed: {e}")
            self.logger.error(f"Tactician ensemble training failed: {e}")
            return TrainingResult(
                success=False,
                error_message=str(e),
                training_time=time.time() - start_time if 'start_time' in locals() else 0.0
            )
    
    async def validate(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> ValidationResult:
        """
        Validate the trained Tactician ensemble.
        
        Args:
            data: Validation data
            targets: Target variables (optional)
            
        Returns:
            Validation result with metrics
        """
        try:
            tprint_info("🎯 Validating Tactician ensemble...")
            self.logger.info("Validating Tactician ensemble...")
            
            start_time = time.time()
            
            # Preprocess data
            processed_data, processed_targets = self._preprocess_data(data, targets)
            
            # Generate base predictions
            base_predictions = await self._generate_base_predictions(processed_data)
            
            # Make ensemble predictions
            ensemble_predictions = await self._predict_ensemble(base_predictions)
            
            # Calculate validation metrics
            validation_metrics = await self._calculate_validation_metrics(
                ensemble_predictions, processed_targets
            )
            
            validation_time = time.time() - start_time
            self._update_performance_metrics('validation', validation_time)
            
            result = ValidationResult(
                success=True,
                metrics=validation_metrics,
                predictions=ensemble_predictions,
                metadata={
                    'validation_time': validation_time,
                    'ensemble_method': self.config.ensemble_method.value
                }
            )
            
            tprint_success(f"✅ Tactician ensemble validation completed in {validation_time:.2f}s")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Tactician ensemble validation failed: {e}")
            self.logger.error(f"Tactician ensemble validation failed: {e}")
            return ValidationResult(
                success=False,
                error_message=str(e)
            )
    
    async def predict(self, data: pd.DataFrame) -> PredictionResult:
        """
        Make predictions with the trained Tactician ensemble.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Prediction result
        """
        try:
            tprint_info("🎯 Making Tactician ensemble predictions...")
            self.logger.info("Making Tactician ensemble predictions...")
            
            start_time = time.time()
            
            # Preprocess data
            processed_data, _ = self._preprocess_data(data, None)
            
            # Generate base predictions
            base_predictions = await self._generate_base_predictions(processed_data)
            
            # Make ensemble predictions
            ensemble_predictions = await self._predict_ensemble(base_predictions)
            
            prediction_time = time.time() - start_time
            self._update_performance_metrics('prediction', prediction_time)
            
            result = PredictionResult(
                success=True,
                predictions=ensemble_predictions,
                metadata={
                    'prediction_time': prediction_time,
                    'ensemble_method': self.config.ensemble_method.value,
                    'base_models_used': list(self._base_trainers.keys())
                }
            )
            
            tprint_success(f"✅ Tactician ensemble predictions completed in {prediction_time:.2f}s")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Tactician ensemble prediction failed: {e}")
            self.logger.error(f"Tactician ensemble prediction failed: {e}")
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
        # For ensemble, we create the meta-learner
        return self._create_meta_learner()
    
    def _get_feature_importance(self, model: Any) -> Optional[Dict[str, float]]:
        """
        Extract feature importance from ensemble model.
        
        Args:
            model: Trained ensemble model
            
        Returns:
            Feature importance dictionary
        """
        try:
            if hasattr(model, 'feature_importances_'):
                return dict(zip(model.feature_names_in_, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                return dict(zip(model.feature_names_in_, model.coef_))
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {e}")
            return None
    
    async def _train_base_models(self, data: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """Train all base models."""
        try:
            tprint_info("🔧 Training base models...")
            
            base_results = {}
            successful_models = 0
            
            for model_type in self.config.base_models:
                tprint_info(f"🔧 Training {model_type.value} base model...")
                
                # Create base trainer
                base_config = TacticianTrainingConfig(
                    model_types=[model_type],
                    timeframe=self.config.timeframe,
                    symbol=self.config.symbol,
                    **self.config.custom_params
                )
                
                base_trainer = TacticianBaseTrainer(base_config, self.logger)
                
                # Train the model
                result = await base_trainer.train(data, targets)
                
                if result.success:
                    self._base_trainers[model_type.value] = base_trainer
                    base_results[model_type.value] = result
                    successful_models += 1
                    tprint_success(f"✅ {model_type.value} base model trained")
                else:
                    tprint_error(f"❌ {model_type.value} base model failed: {result.error_message}")
                    base_results[model_type.value] = result
            
            self._ensemble_state['base_models_trained'] = successful_models > 0
            
            return {
                'success': successful_models > 0,
                'successful_models': successful_models,
                'total_models': len(self.config.base_models),
                'results': base_results
            }
            
        except Exception as e:
            tprint_error(f"❌ Base model training failed: {e}")
            self.logger.error(f"Base model training failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _generate_base_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions from all base models."""
        try:
            tprint_info("🔧 Generating base predictions...")
            
            base_predictions = {}
            
            for model_name, trainer in self._base_trainers.items():
                # Get predictions from base trainer
                pred_result = await trainer.predict(data)
                
                if pred_result.success:
                    # Get the first (and only) model's predictions
                    model_predictions = list(pred_result.predictions.values())[0]
                    base_predictions[f"{model_name}_pred"] = model_predictions
                else:
                    tprint_warning(f"⚠️ Failed to get predictions from {model_name}")
            
            self._ensemble_state['base_predictions_generated'] = len(base_predictions) > 0
            
            return pd.DataFrame(base_predictions, index=data.index)
            
        except Exception as e:
            tprint_error(f"❌ Base prediction generation failed: {e}")
            self.logger.error(f"Base prediction generation failed: {e}")
            return pd.DataFrame()
    
    async def _train_ensemble_model(self, base_predictions: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """Train the ensemble meta-learner."""
        try:
            tprint_info("🔧 Training ensemble meta-learner...")
            
            if self.config.ensemble_method == TacticianEnsembleMethod.STACKING:
                # For stacking, we need a meta-learner
                self._ensemble_model = self._create_meta_learner()
                self._ensemble_model.fit(base_predictions, targets)
                
                        elif self.config.ensemble_method == TacticianEnsembleMethod.VOTING:
                            # For voting, we create a voting regressor
                            from sklearn.ensemble import VotingRegressor
                            
                            estimators = []
                            for model_name, trainer in self._base_trainers.items():
                                model = trainer._model_state.get(f"{model_name}_model")
                                if model is not None:
                                    estimators.append((model_name, model))
                            
                            self._ensemble_model = VotingRegressor(estimators)
                            self._ensemble_model.fit(base_predictions, targets)
                            
                        elif self.config.ensemble_method == TacticianEnsembleMethod.BLENDING:
                            # For blending, we use a weighted combination with learned weights
                            self._ensemble_model = self._create_blending_model()
                            self._ensemble_model.fit(base_predictions, targets)
                            
                        elif self.config.ensemble_method == TacticianEnsembleMethod.WEIGHTED:
                            # For weighted averaging, we learn optimal weights
                            self._ensemble_model = self._create_weighted_model()
                            self._ensemble_model.fit(base_predictions, targets)
                
            elif self.config.ensemble_method == TacticianEnsembleMethod.AVERAGING:
                # For averaging, we create a simple averaging model
                self._ensemble_model = self._create_averaging_model()
                
            else:
                raise ValueError(f"Unsupported ensemble method: {self.config.ensemble_method}")
            
            self._ensemble_state['meta_learner_trained'] = True
            
            return {'success': True, 'ensemble_method': self.config.ensemble_method.value}
            
        except Exception as e:
            tprint_error(f"❌ Ensemble model training failed: {e}")
            self.logger.error(f"Ensemble model training failed: {e}")
            return {'success': False, 'error': str(e)}
    
                def _create_meta_learner(self):
                    """Create meta-learner for stacking."""
                    try:
                        if self.config.meta_learner_type == ModelType.LIGHTGBM:
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
                                **self.config.meta_learner_params
                            }
                            return lgb.LGBMRegressor(**params)
                        else:
                            # Default to linear regression
                            from sklearn.linear_model import LinearRegression
                            return LinearRegression(**self.config.meta_learner_params)
                            
                    except ImportError:
                        self.logger.error("Required libraries not available")
                        return None
    
                def _create_averaging_model(self):
                    """Create averaging model."""
                    # Simple averaging model that doesn't need training
                    return None
                
                def _create_blending_model(self):
                    """Create blending model with learned weights."""
                    try:
                        from sklearn.linear_model import LinearRegression
                        return LinearRegression(**self.config.meta_learner_params)
                    except ImportError:
                        self.logger.error("scikit-learn not available")
                        return None
                
                def _create_weighted_model(self):
                    """Create weighted averaging model."""
                    # This is a simple wrapper that learns weights
                    class WeightedAveraging:
                        def __init__(self):
                            self.weights_ = None
                            
                        def fit(self, X, y):
                            from sklearn.linear_model import LinearRegression
                            # Use linear regression to learn weights
                            self.weights_ = LinearRegression(**self.config.meta_learner_params)
                            self.weights_.fit(X, y)
                            return self
                            
                        def predict(self, X):
                            if self.weights_ is not None:
                                return self.weights_.predict(X)
                            else:
                                return X.mean(axis=1).values
                    
                    return WeightedAveraging()
    
                async def _predict_ensemble(self, base_predictions: pd.DataFrame) -> np.ndarray:
                    """Make predictions with the ensemble model."""
                    try:
                        if self.config.ensemble_method == TacticianEnsembleMethod.AVERAGING:
                            # Simple averaging of base predictions
                            pred_columns = [col for col in base_predictions.columns if col.endswith('_pred')]
                            if pred_columns:
                                return base_predictions[pred_columns].mean(axis=1).values
                            else:
                                return np.zeros(len(base_predictions))
                        elif self.config.ensemble_method == TacticianEnsembleMethod.WEIGHTED:
                            # Weighted averaging (weights learned during training)
                            pred_columns = [col for col in base_predictions.columns if col.endswith('_pred')]
                            if pred_columns and hasattr(self, '_ensemble_weights'):
                                weights = self._ensemble_weights
                                if len(weights) == len(pred_columns):
                                    weighted_preds = base_predictions[pred_columns] * weights
                                    return weighted_preds.sum(axis=1).values
                                else:
                                    return base_predictions[pred_columns].mean(axis=1).values
                            else:
                                return base_predictions[pred_columns].mean(axis=1).values
                        else:
                            # Use trained ensemble model (stacking, voting, blending)
                            if self._ensemble_model is not None:
                                return self._ensemble_model.predict(base_predictions)
                            else:
                                return np.zeros(len(base_predictions))
                                
                    except Exception as e:
                        self.logger.error(f"Ensemble prediction failed: {e}")
                        return np.zeros(len(base_predictions))
    
    async def _calculate_ensemble_metrics(self, data: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        """Calculate ensemble-specific metrics."""
        try:
            # Generate base predictions
            base_predictions = await self._generate_base_predictions(data)
            
            # Make ensemble predictions
            ensemble_predictions = await self._predict_ensemble(base_predictions)
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics = {
                'mse': mean_squared_error(targets, ensemble_predictions),
                'mae': mean_absolute_error(targets, ensemble_predictions),
                'r2': r2_score(targets, ensemble_predictions),
                'rmse': np.sqrt(mean_squared_error(targets, ensemble_predictions))
            }
            
            # Add ensemble-specific metrics
            metrics.update({
                'base_models_count': len(self._base_trainers),
                'ensemble_method': self.config.ensemble_method.value
            })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Ensemble metrics calculation failed: {e}")
            return {}
    
    async def _calculate_validation_metrics(self, predictions: np.ndarray, targets: pd.Series) -> Dict[str, float]:
        """Calculate validation metrics."""
        try:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics = {
                'mse': mean_squared_error(targets, predictions),
                'mae': mean_absolute_error(targets, predictions),
                'r2': r2_score(targets, predictions),
                'rmse': np.sqrt(mean_squared_error(targets, predictions))
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Validation metrics calculation failed: {e}")
            return {}
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get ensemble-specific training summary."""
        base_summary = self.get_training_summary()
        base_summary.update({
            'ensemble_state': self._ensemble_state.copy(),
            'ensemble_method': self.config.ensemble_method.value,
            'base_models': list(self._base_trainers.keys()),
            'meta_learner_type': self.config.meta_learner_type.value
        })
        return base_summary