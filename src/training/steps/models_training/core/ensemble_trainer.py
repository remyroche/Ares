"""
Ensemble Trainer - Ensemble Model Training Implementation

This module provides ensemble training capabilities that combine multiple
individual models to create more robust and accurate predictions.

Key Features:
- Multi-model ensemble training
- Role-specific ensemble strategies (Analyst vs Tactician)
- Advanced ensemble methods (Stacking, Blending, Voting)
- Performance optimization and model selection
- Cross-validation and out-of-fold predictions
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_debug, tprint_performance
from src.utils.common_operations import (
    safe_divide, safe_correlation, safe_mean, safe_std, safe_float, safe_int,
    get_memory_usage, optimize_dataframe_memory, memory_checkpoint
)
from src.utils.common_utilities import calculate_data_quality_metrics, get_dataframe_info
from src.utils.math_validation import validate_finite, validate_positive, validate_range
from src.utils.hardware.integrated_hardware_manager import (
    get_integrated_hardware_manager, WorkloadType, process_ml_training_data
)
from src.utils.hardware.optimization_decorators import (
    smart_cache, auto_optimize, memory_efficient, performance_tracked
)
from src.utils.hardware.memory_optimized_decorators import (
    memory_optimized, comprehensive_memory_optimization, MemoryOptimizationLevel
)
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
from src.utils.kline_parquet import KlinesParquetManager
from src.core.decorators import handles_errors, traced, log_execution_time

from .base_trainer import (
    BaseTrainer, TrainingConfig, TrainingResult, ValidationResult, 
    PredictionResult, TrainingRole, ModelType
)
from .model_trainer import ModelTrainer

# Import SHAP explainability utilities
try:
    from src.utils.ml_common.explainability import (
        SHAPLIMEExplainer, ExplanationConfig, create_explainer, explain_stacking_ensemble
    )
    SHAP_AVAILABLE = True
except ImportError as e:
    SHAP_AVAILABLE = False
    tprint(f"⚠️ [ENSEMBLE_TRAINER] SHAP explainability utilities not available: {e}", color="yellow")


class EnsembleStrategy:
    """Ensemble strategy types."""
    STACKING = "stacking"
    BLENDING = "blending"
    VOTING = "voting"
    BAGGING = "bagging"


class EnsembleTrainer(BaseTrainer):
    """
    Ensemble trainer implementation.
    
    This class provides ensemble training capabilities that combine multiple
    individual models to create more robust and accurate predictions.
    """
    
    def __init__(self, config: TrainingConfig, logger: Optional[logging.Logger] = None):
        """Initialize the ensemble trainer."""
        super().__init__(config, logger)
        
        # Ensemble-specific configuration
        self.ensemble_strategy = config.custom_params.get('ensemble_strategy', EnsembleStrategy.STACKING)
        self.meta_learner_type = config.custom_params.get('meta_learner_type', 'lightgbm')
        self.cv_folds = config.custom_params.get('cv_folds', 5)
        
        # Individual trainers
        self._individual_trainers = {}
        self._meta_learner = None
        self._oof_predictions = {}
        self._ensemble_weights = {}
    
    @handles_errors(
        exceptions=(ValueError, RuntimeError, MemoryError),
        default_return=TrainingResult(success=False, error_message="Ensemble training failed"),
        context="ensemble training"
    )
    async def train(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> TrainingResult:
        """
        Train ensemble models with multiple strategies.
        
        Args:
            data: Training data
            targets: Target variables
            
        Returns:
            Training result with ensemble model and metrics
        """
        try:
            self.logger.info(f"🚀 Starting {self.config.role.value} ensemble training...")
            start_time = time.time()
            
            # Preprocess data
            processed_data, processed_targets = self._preprocess_data(data, targets)
            
            # Train individual models
            individual_results = await self._train_individual_models(processed_data, processed_targets)
            
            if not individual_results:
                return TrainingResult(success=False, error_message="No individual models trained successfully")
            
            # Generate out-of-fold predictions
            oof_predictions = await self._generate_oof_predictions(processed_data, processed_targets)
            
            # Train meta-learner
            meta_result = await self._train_meta_learner(oof_predictions, processed_targets)
            
            if not meta_result.success:
                return TrainingResult(success=False, error_message="Meta-learner training failed")
            
            # Calculate ensemble metrics
            ensemble_metrics = await self._calculate_ensemble_metrics(
                individual_results, meta_result, processed_data, processed_targets
            )
            
            # Generate SHAP explanations for complete ensemble
            ensemble_shap_explanations = None
            if SHAP_AVAILABLE:
                try:
                    tprint("🔍 [ENSEMBLE_TRAINER] Generating SHAP explanations for complete ensemble", color="cyan")
                    
                    # Prepare base models for ensemble explanation
                    base_models = {}
                    for model_name, result in individual_results.items():
                        if result.success and result.model is not None:
                            base_models[model_name] = result.model
                    
                    if base_models and self._meta_learner is not None:
                        # Create feature names
                        feature_names = list(processed_data.columns)
                        output_names = ["prediction"] if self.config.role == TrainingRole.TACTICIAN else ["class_0", "class_1"]
                        
                        # Use the stacking ensemble explanation function
                        ensemble_explanations = explain_stacking_ensemble(
                            base_models=base_models,
                            meta_model=self._meta_learner,
                            X=processed_data.values,
                            output_names=output_names,
                            feature_names=feature_names
                        )
                        
                        ensemble_shap_explanations = {
                            'base_model_explanations': {name: {
                                'shap_values': result.shap_values,
                                'base_values': result.shap_base_values,
                                'feature_names': result.shap_feature_names,
                                'explanation_time': result.explanation_time
                            } for name, result in ensemble_explanations.items() if name != 'meta_model'},
                            'meta_model_explanations': ensemble_explanations.get('meta_model', {}),
                            'total_explanation_time': sum(result.explanation_time for result in ensemble_explanations.values())
                        }
                        
                        tprint(f"✅ [ENSEMBLE_TRAINER] Complete ensemble SHAP explanations generated in {ensemble_shap_explanations['total_explanation_time']:.3f}s", color="green")
                    else:
                        tprint("⚠️ [ENSEMBLE_TRAINER] Cannot generate ensemble SHAP explanations - missing base models or meta-learner", color="yellow")
                        
                except Exception as e:
                    tprint(f"⚠️ [ENSEMBLE_TRAINER] Complete ensemble SHAP explanation failed: {e}", color="yellow")
                    ensemble_shap_explanations = None
            
            training_time = time.time() - start_time
            
            # Update state
            self._training_state['training_completed'] = True
            self._training_state['training_started'] = True
            self._update_performance_metrics('training', training_time)
            
            result = TrainingResult(
                success=True,
                model=self._meta_learner,
                metrics=ensemble_metrics,
                training_time=training_time,
                shap_explanations=ensemble_shap_explanations,
                metadata={
                    'ensemble_strategy': self.ensemble_strategy,
                    'individual_models': len(individual_results),
                    'meta_learner_type': self.meta_learner_type,
                    'individual_results': individual_results,
                    'oof_predictions_shape': oof_predictions.shape if oof_predictions is not None else None
                }
            )
            
            self.logger.info(f"✅ Ensemble training completed successfully in {training_time:.2f}s")
            tprint_success(f"Trained ensemble with {len(individual_results)} base models")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ensemble training failed: {e}")
            return TrainingResult(
                success=False,
                error_message=str(e),
                training_time=time.time() - start_time
            )
    
    async def _train_individual_models(
        self, 
        data: pd.DataFrame, 
        targets: pd.Series
    ) -> Dict[str, TrainingResult]:
        """Train individual models for ensemble."""
        try:
            individual_results = {}
            
            for model_type in self.config.model_types:
                self.logger.info(f"📊 Training {model_type.value} for ensemble...")
                
                # Create individual trainer
                individual_config = TrainingConfig(
                    role=self.config.role,
                    model_types=[model_type],
                    timeframe=self.config.timeframe,
                    symbol=self.config.symbol,
                    validation_split=self.config.validation_split,
                    cross_validation_folds=self.config.cross_validation_folds,
                    random_seed=self.config.random_seed,
                    enable_hyperparameter_optimization=self.config.enable_hyperparameter_optimization,
                    enable_ensemble=False,  # Individual models, not ensemble
                    custom_params=self.config.custom_params.copy()
                )
                
                trainer = ModelTrainer(individual_config, self.logger)
                
                # Train individual model
                result = await trainer.train(data, targets)
                
                if result.success:
                    individual_results[model_type.value] = result
                    self._individual_trainers[model_type.value] = trainer
                    self.logger.info(f"✅ {model_type.value} trained successfully")
                else:
                    self.logger.error(f"❌ {model_type.value} training failed: {result.error_message}")
            
            return individual_results
            
        except Exception as e:
            self.logger.error(f"Individual model training failed: {e}")
            return {}
    
    async def _generate_oof_predictions(
        self, 
        data: pd.DataFrame, 
        targets: pd.Series
    ) -> Optional[np.ndarray]:
        """Generate out-of-fold predictions for meta-learner training."""
        try:
            from sklearn.model_selection import KFold
            
            self.logger.info("🔄 Generating out-of-fold predictions...")
            
            # Initialize OOF predictions array
            oof_predictions = np.zeros((len(data), len(self._individual_trainers)))
            
            # Cross-validation for OOF predictions
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.config.random_seed)
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
                self.logger.info(f"📊 Processing fold {fold + 1}/{self.cv_folds}")
                
                # Split data
                train_data = data.iloc[train_idx]
                train_targets = targets.iloc[train_idx]
                val_data = data.iloc[val_idx]
                
                # Train models on fold
                for i, (model_name, trainer) in enumerate(self._individual_trainers.items()):
                    # Create fold-specific trainer
                    fold_config = TrainingConfig(
                        role=self.config.role,
                        model_types=[ModelType(model_name)],
                        timeframe=self.config.timeframe,
                        symbol=self.config.symbol,
                        validation_split=0.0,  # No validation split for OOF
                        cross_validation_folds=1,
                        random_seed=self.config.random_seed,
                        enable_hyperparameter_optimization=False,
                        enable_ensemble=False,
                        custom_params=self.config.custom_params.copy()
                    )
                    
                    fold_trainer = ModelTrainer(fold_config, self.logger)
                    
                    # Train on fold
                    fold_result = await fold_trainer.train(train_data, train_targets)
                    
                    if fold_result.success:
                        # Predict on validation set
                        val_predictions = await fold_trainer.predict(val_data)
                        if val_predictions.success:
                            oof_predictions[val_idx, i] = val_predictions.predictions
                        else:
                            self.logger.warning(f"Prediction failed for {model_name} in fold {fold}")
                    else:
                        self.logger.warning(f"Training failed for {model_name} in fold {fold}")
            
            # Store OOF predictions
            self._oof_predictions = oof_predictions
            
            self.logger.info(f"✅ Generated OOF predictions: {oof_predictions.shape}")
            return oof_predictions
            
        except Exception as e:
            self.logger.error(f"OOF prediction generation failed: {e}")
            return None
    
    async def _train_meta_learner(
        self, 
        oof_predictions: np.ndarray, 
        targets: pd.Series
    ) -> TrainingResult:
        """Train meta-learner on out-of-fold predictions."""
        try:
            self.logger.info(f"🧠 Training meta-learner ({self.meta_learner_type})...")
            
            # Create meta-learner
            if self.meta_learner_type == 'lightgbm':
                import lightgbm as lgb
                
                if self.config.role == TrainingRole.ANALYST:
                    meta_model = lgb.LGBMClassifier(
                        objective='binary',
                        metric='binary_logloss',
                        num_leaves=31,
                        learning_rate=0.05,
                        feature_fraction=0.9,
                        bagging_fraction=0.8,
                        bagging_freq=5,
                        verbose=-1
                    )
                else:  # Tactician
                    meta_model = lgb.LGBMRegressor(
                        objective='regression',
                        metric='rmse',
                        num_leaves=31,
                        learning_rate=0.05,
                        feature_fraction=0.9,
                        bagging_fraction=0.8,
                        bagging_freq=5,
                        verbose=-1
                    )
            
            elif self.meta_learner_type == 'catboost':
                from catboost import CatBoostClassifier, CatBoostRegressor
                
                if self.config.role == TrainingRole.ANALYST:
                    meta_model = CatBoostClassifier(
                        iterations=1000,
                        learning_rate=0.05,
                        depth=6,
                        loss_function='Logloss',
                        eval_metric='AUC',
                        verbose=False
                    )
                else:  # Tactician
                    meta_model = CatBoostRegressor(
                        iterations=1000,
                        learning_rate=0.05,
                        depth=6,
                        loss_function='RMSE',
                        eval_metric='RMSE',
                        verbose=False
                    )
            
            else:
                raise ValueError(f"Unsupported meta-learner type: {self.meta_learner_type}")
            
            # Train meta-learner
            meta_model.fit(oof_predictions, targets)
            
            # Calculate meta-learner metrics
            meta_predictions = meta_model.predict(oof_predictions)
            
            if self.config.role == TrainingRole.ANALYST:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                binary_predictions = (meta_predictions > 0.5).astype(int)
                metrics = {
                    'meta_accuracy': accuracy_score(targets, binary_predictions),
                    'meta_precision': precision_score(targets, binary_predictions),
                    'meta_recall': recall_score(targets, binary_predictions),
                    'meta_f1_score': f1_score(targets, binary_predictions)
                }
            else:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                metrics = {
                    'meta_mse': mean_squared_error(targets, meta_predictions),
                    'meta_mae': mean_absolute_error(targets, meta_predictions),
                    'meta_r2': r2_score(targets, meta_predictions),
                    'meta_rmse': np.sqrt(mean_squared_error(targets, meta_predictions))
                }
            
            # Store meta-learner
            self._meta_learner = meta_model
            
            # Generate SHAP explanations for meta-learner
            shap_explanations = None
            if SHAP_AVAILABLE:
                try:
                    tprint("🔍 [ENSEMBLE_TRAINER] Generating SHAP explanations for meta-learner", color="cyan")
                    shap_config = ExplanationConfig(
                        enable_shap=True,
                        enable_lime=False,
                        shap_sample_size=min(100, len(oof_predictions)),
                        shap_max_features=min(50, oof_predictions.shape[1])
                    )
                    explainer = create_explainer(shap_config)
                    
                    # Create feature names for meta-learner (base model predictions)
                    meta_feature_names = [f"base_model_{i}_pred" for i in range(oof_predictions.shape[1])]
                    output_names = ["prediction"] if self.config.role == TrainingRole.TACTICIAN else ["class_0", "class_1"]
                    
                    shap_result = explainer.explain_model(
                        model=meta_model,
                        X=oof_predictions,
                        model_name="Meta-learner",
                        output_names=output_names,
                        feature_names=meta_feature_names
                    )
                    
                    shap_explanations = {
                        'shap_values': shap_result.shap_values,
                        'base_values': shap_result.shap_base_values,
                        'feature_names': shap_result.shap_feature_names,
                        'explanation_time': shap_result.explanation_time
                    }
                    
                    tprint(f"✅ [ENSEMBLE_TRAINER] Meta-learner SHAP explanations generated in {shap_result.explanation_time:.3f}s", color="green")
                except Exception as e:
                    tprint(f"⚠️ [ENSEMBLE_TRAINER] Meta-learner SHAP explanation failed: {e}", color="yellow")
                    shap_explanations = None
            
            return TrainingResult(
                success=True,
                model=meta_model,
                metrics=metrics,
                shap_explanations=shap_explanations
            )
            
        except Exception as e:
            self.logger.error(f"Meta-learner training failed: {e}")
            return TrainingResult(success=False, error_message=str(e))
    
    async def _calculate_ensemble_metrics(
        self,
        individual_results: Dict[str, TrainingResult],
        meta_result: TrainingResult,
        data: pd.DataFrame,
        targets: pd.Series
    ) -> Dict[str, float]:
        """Calculate comprehensive ensemble metrics."""
        try:
            ensemble_metrics = {}
            
            # Individual model metrics
            for model_name, result in individual_results.items():
                if result.success and result.metrics:
                    for metric, value in result.metrics.items():
                        ensemble_metrics[f'{model_name}_{metric}'] = value
            
            # Meta-learner metrics
            if meta_result.success and meta_result.metrics:
                ensemble_metrics.update(meta_result.metrics)
            
            # Ensemble performance metrics
            ensemble_predictions = await self._get_ensemble_predictions(data)
            
            if ensemble_predictions is not None and targets is not None:
                if self.config.role == TrainingRole.ANALYST:
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    binary_predictions = (ensemble_predictions > 0.5).astype(int)
                    ensemble_metrics.update({
                        'ensemble_accuracy': accuracy_score(targets, binary_predictions),
                        'ensemble_precision': precision_score(targets, binary_predictions),
                        'ensemble_recall': recall_score(targets, binary_predictions),
                        'ensemble_f1_score': f1_score(targets, binary_predictions)
                    })
                else:
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    ensemble_metrics.update({
                        'ensemble_mse': mean_squared_error(targets, ensemble_predictions),
                        'ensemble_mae': mean_absolute_error(targets, ensemble_predictions),
                        'ensemble_r2': r2_score(targets, ensemble_predictions),
                        'ensemble_rmse': np.sqrt(mean_squared_error(targets, ensemble_predictions))
                    })
            
            # Diversity metrics
            diversity_metrics = self._calculate_diversity_metrics(individual_results)
            ensemble_metrics.update(diversity_metrics)
            
            return ensemble_metrics
            
        except Exception as e:
            self.logger.error(f"Ensemble metrics calculation failed: {e}")
            return {}
    
    def _calculate_diversity_metrics(self, individual_results: Dict[str, TrainingResult]) -> Dict[str, float]:
        """Calculate diversity metrics for ensemble."""
        try:
            diversity_metrics = {}
            
            # Get predictions from all models
            all_predictions = []
            for result in individual_results.values():
                if result.success and hasattr(result, 'predictions'):
                    all_predictions.append(result.predictions)
            
            if len(all_predictions) < 2:
                return diversity_metrics
            
            # Calculate pairwise correlations
            correlations = []
            for i in range(len(all_predictions)):
                for j in range(i + 1, len(all_predictions)):
                    corr = np.corrcoef(all_predictions[i], all_predictions[j])[0, 1]
                    correlations.append(corr)
            
            if correlations:
                diversity_metrics['avg_correlation'] = np.mean(correlations)
                diversity_metrics['min_correlation'] = np.min(correlations)
                diversity_metrics['max_correlation'] = np.max(correlations)
                diversity_metrics['correlation_std'] = np.std(correlations)
            
            return diversity_metrics
            
        except Exception as e:
            self.logger.warning(f"Diversity metrics calculation failed: {e}")
            return {}
    
    async def _get_ensemble_predictions(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Get ensemble predictions by combining individual model predictions."""
        try:
            if not self._individual_trainers or self._meta_learner is None:
                return None
            
            # Get predictions from all individual models
            individual_predictions = []
            for model_name, trainer in self._individual_trainers.items():
                prediction_result = await trainer.predict(data)
                if prediction_result.success:
                    individual_predictions.append(prediction_result.predictions)
                else:
                    self.logger.warning(f"Prediction failed for {model_name}")
                    return None
            
            # Stack predictions
            stacked_predictions = np.column_stack(individual_predictions)
            
            # Get meta-learner predictions
            ensemble_predictions = self._meta_learner.predict(stacked_predictions)
            
            return ensemble_predictions
            
        except Exception as e:
            self.logger.error(f"Ensemble prediction failed: {e}")
            return None
    
    def _create_model(self, model_type: ModelType) -> Any:
        """Create model instance (not used in ensemble trainer)."""
        return None
    
    def _get_feature_importance(self, model: Any) -> Optional[Dict[str, float]]:
        """Extract feature importance from ensemble model."""
        try:
            if self._meta_learner is None:
                return None
            
            if hasattr(self._meta_learner, 'feature_importance'):
                # Get feature names from individual models
                feature_names = list(self._individual_trainers.keys())
                return dict(zip(feature_names, self._meta_learner.feature_importance()))
            elif hasattr(self._meta_learner, 'get_feature_importance'):
                feature_names = list(self._individual_trainers.keys())
                return dict(zip(feature_names, self._meta_learner.get_feature_importance()))
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"Failed to extract ensemble feature importance: {e}")
            return None
    
    @handles_errors(
        exceptions=(ValueError, RuntimeError),
        default_return=ValidationResult(success=False, error_message="Ensemble validation failed"),
        context="ensemble validation"
    )
    async def validate(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> ValidationResult:
        """Validate ensemble model."""
        try:
            if self._meta_learner is None:
                return ValidationResult(success=False, error_message="No ensemble model available")
            
            # Get ensemble predictions
            predictions = await self._get_ensemble_predictions(data)
            
            if predictions is None:
                return ValidationResult(success=False, error_message="Failed to generate ensemble predictions")
            
            # Calculate validation metrics
            metrics = {}
            if targets is not None:
                if self.config.role == TrainingRole.ANALYST:
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    binary_predictions = (predictions > 0.5).astype(int)
                    metrics = {
                        'accuracy': accuracy_score(targets, binary_predictions),
                        'precision': precision_score(targets, binary_predictions),
                        'recall': recall_score(targets, binary_predictions),
                        'f1_score': f1_score(targets, binary_predictions)
                    }
                else:
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    metrics = {
                        'mse': mean_squared_error(targets, predictions),
                        'mae': mean_absolute_error(targets, predictions),
                        'r2': r2_score(targets, predictions),
                        'rmse': np.sqrt(mean_squared_error(targets, predictions))
                    }
            
            return ValidationResult(
                success=True,
                metrics=metrics,
                predictions=predictions
            )
            
        except Exception as e:
            self.logger.error(f"Ensemble validation failed: {e}")
            return ValidationResult(success=False, error_message=str(e))
    
    @handles_errors(
        exceptions=(ValueError, RuntimeError),
        default_return=PredictionResult(success=False, error_message="Ensemble prediction failed"),
        context="ensemble prediction"
    )
    async def predict(self, data: pd.DataFrame) -> PredictionResult:
        """Make predictions with ensemble model."""
        try:
            if self._meta_learner is None:
                return PredictionResult(success=False, error_message="No ensemble model available")
            
            # Get ensemble predictions
            predictions = await self._get_ensemble_predictions(data)
            
            if predictions is None:
                return PredictionResult(success=False, error_message="Failed to generate ensemble predictions")
            
            # Get probabilities if available
            probabilities = None
            if hasattr(self._meta_learner, 'predict_proba'):
                # Get individual model probabilities
                individual_probabilities = []
                for trainer in self._individual_trainers.values():
                    pred_result = await trainer.predict(data)
                    if pred_result.success and pred_result.probabilities is not None:
                        individual_probabilities.append(pred_result.probabilities)
                
                if individual_probabilities:
                    # Average probabilities
                    probabilities = np.mean(individual_probabilities, axis=0)
            
            return PredictionResult(
                success=True,
                predictions=predictions,
                probabilities=probabilities
            )
            
        except Exception as e:
            self.logger.error(f"Ensemble prediction failed: {e}")
            return PredictionResult(success=False, error_message=str(e))
