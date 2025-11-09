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
from src.utils.hardware.m1_memory_optimizer import optimize_memory
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
from src.utils.kline_parquet import KlinesParquetManager
from src.core.decorators import handles_errors, traced, log_execution_time

from .base_trainer import (
    BaseTrainer, TrainingConfig, TrainingResult, ValidationResult, 
    PredictionResult, TrainingRole, ModelType
)
from .model_trainer import ModelTrainer
from .training_metrics_collector import TrainingMetricsCollector, ModelMetrics


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
        
        # Metrics collector
        self._metrics_collector = TrainingMetricsCollector(logger)
    
    @handles_errors(
        exceptions=(ValueError, RuntimeError, MemoryError),
        default_return=TrainingResult(success=False, error_message="Ensemble training failed"),
        context="ensemble training"
    )
    async def train(
        self, 
        data: pd.DataFrame, 
        targets: pd.Series,
        base_predictions: Optional[pd.DataFrame] = None
    ) -> TrainingResult:
        """
        Train ensemble model with comprehensive metrics collection.
        
        Args:
            data: Training data
            targets: Target values
            base_predictions: Pre-computed predictions from base models (REQUIRED for stacking)
            
        Returns:
            Training result with ensemble model and comprehensive metrics
        """
        try:
            self.logger.info(f"🚀 Starting {self.config.role.value} ensemble training with comprehensive metrics...")
            start_time = time.time()
            
            # CRITICAL: Base predictions are REQUIRED for proper stacking
            if base_predictions is None or base_predictions.empty:
                error_msg = "Base predictions are required for ensemble training. Cannot re-train on tiny subsets."
                self.logger.error(f"❌ {error_msg}")
                return TrainingResult(success=False, error_message=error_msg)
            
            tprint_info(f"✅ Using base model predictions: {base_predictions.shape}")
            
            # Start metrics collection session
            training_type = f"{self.config.role.value}_ensemble"
            self._metrics_collector.start_session(
                training_type=training_type,
                symbol=self.config.symbol,
                timeframe=self.config.timeframe
            )
            
            # Preprocess data
            processed_data, processed_targets = self._preprocess_data(data, targets)
            
            # Align base predictions with processed targets (remove duplicates first)
            if base_predictions.index.duplicated().any():
                tprint_warning(f"⚠️ Removing {base_predictions.index.duplicated().sum()} duplicate indices from base_predictions")
                base_predictions = base_predictions[~base_predictions.index.duplicated(keep='first')]
            
            if not base_predictions.index.equals(processed_targets.index):
                tprint_warning("⚠️ Aligning base predictions to processed targets index...")
                base_predictions = base_predictions.reindex(processed_targets.index)
            
            # Use base predictions as OOF predictions for meta-learner
            oof_predictions = base_predictions.values
            tprint_info(f"📊 Using {oof_predictions.shape[1]} base model predictions for meta-learning")
            
            # Phase 3: Collect pre-HPO metrics for meta-learner
            tprint_info("📈 Phase 3: Collecting pre-HPO metrics for meta-learner...")
            meta_model_metrics = self._metrics_collector.collect_pre_hpo_metrics(
                model_name=f"{self.config.role.value}_ensemble_meta",
                model_type=self.meta_learner_type,
                model=self._create_meta_learner_model(),
                X=pd.DataFrame(oof_predictions),
                y=processed_targets,
                n_folds=self.cv_folds
            )
            
            # Phase 4: Train meta-learner
            tprint_info("🧠 Phase 4: Training meta-learner...")
            meta_result = await self._train_meta_learner(oof_predictions, processed_targets)
            
            if not meta_result.success:
                return TrainingResult(success=False, error_message="Meta-learner training failed")
            
            # Phase 5: Collect post-HPO metrics
            tprint_info("📊 Phase 5: Collecting post-HPO metrics for meta-learner...")
            meta_model_metrics = self._metrics_collector.collect_post_hpo_metrics(
                model_metrics=meta_model_metrics,
                model=self._meta_learner,
                X=pd.DataFrame(oof_predictions),
                y=processed_targets,
                best_params=meta_result.metadata.get('best_params', {}),
                hpo_n_trials=0,  # Meta-learner typically doesn't use HPO
                hpo_time=0.0,
                n_folds=self.cv_folds
            )
            
            # Add metrics to session
            self._metrics_collector.add_model_metrics(meta_model_metrics)
            
            # Calculate ensemble metrics
            tprint_info("📈 Phase 6: Calculating ensemble metrics...")
            ensemble_metrics = await self._calculate_ensemble_metrics(
                base_predictions, meta_result, processed_data, processed_targets
            )
            
            training_time = time.time() - start_time
            
            # Finalize session and generate report
            tprint_info("📝 Generating comprehensive ensemble training report...")
            session = self._metrics_collector.finalize_session(
                total_training_time=training_time,
                data_quality_score=0.85,
                n_samples=len(processed_data),
                n_features=len(processed_data.columns)
            )
            
            # Generate and save report
            report_path = self._metrics_collector.save_report()
            
            # Update state
            self._training_state['training_completed'] = True
            self._training_state['training_started'] = True
            self._update_performance_metrics('training', training_time)
            
            # Generate final predictions using meta-learner
            final_predictions = None
            if self._meta_learner is not None and oof_predictions is not None:
                try:
                    final_predictions = self._meta_learner.predict(oof_predictions)
                except Exception as e:
                    self.logger.warning(f"Could not generate final predictions: {e}")
            
            result = TrainingResult(
                success=True,
                model=self._meta_learner,
                predictions=final_predictions,
                metrics=ensemble_metrics,
                training_time=training_time,
                metadata={
                    'ensemble_strategy': self.ensemble_strategy,
                    'base_models_count': base_predictions.shape[1] if base_predictions is not None else 0,
                    'meta_learner_type': self.meta_learner_type,
                    'oof_predictions_shape': oof_predictions.shape if oof_predictions is not None else None,
                    'comprehensive_metrics': meta_model_metrics,
                    'report_path': str(report_path)
                }
            )
            
            self.logger.info(f"✅ Ensemble training completed successfully in {training_time:.2f}s")
            tprint_success(f"✅ Trained ensemble with {base_predictions.shape[1]} base models and comprehensive metrics")
            tprint_success(f"📄 Report saved to: {report_path}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ensemble training failed: {e}")
            import traceback
            traceback.print_exc()
            return TrainingResult(
                success=False,
                error_message=str(e),
                training_time=time.time() - start_time
            )
    
    def _create_meta_learner_model(self) -> Any:
        """Create a meta-learner model for pre-HPO metrics."""
        try:
            if self.meta_learner_type == 'lightgbm':
                import lightgbm as lgb
                return lgb.LGBMRegressor(verbose=-1)
            elif self.meta_learner_type == 'catboost':
                from catboost import CatBoostRegressor
                return CatBoostRegressor(verbose=False)
            else:
                import lightgbm as lgb
                return lgb.LGBMRegressor(verbose=-1)
        except Exception as e:
            self.logger.warning(f"Failed to create meta-learner model: {e}")
            return None
    
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
                
                # Always use Regressor for trading models (predicting continuous values)
                # Both ANALYST and TACTICIAN predict continuous targets
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
                from catboost import CatBoostRegressor
                import os
                
                # Get GPU manager for hardware acceleration
                from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
                from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
                
                gpu_manager = get_m1_gpu_manager()
                cpu_optimizer = get_m1_cpu_optimizer()
                
                # Determine optimal thread count from hardware
                n_threads = cpu_optimizer.get_optimal_thread_count() if cpu_optimizer else (os.cpu_count() or 4)
                
                # Check GPU availability for CatBoost
                gpu_available = gpu_manager.is_m1 if gpu_manager else False
                
                # Performance optimizations
                performance_params = {
                    'thread_count': n_threads,
                    'bootstrap_type': 'Bernoulli',  # Changed from 'Bayesian' to support subsample parameter
                    'verbose': False,
                    'allow_writing_files': False,
                }
                
                # Add GPU acceleration if available
                if gpu_available:
                    performance_params['task_type'] = 'GPU'
                    performance_params['devices'] = '0'
                    tprint_info("🚀 CatBoost meta-learner using GPU acceleration")
                else:
                    performance_params['task_type'] = 'CPU'
                
                # Always use Regressor for trading models (predicting continuous values)
                # Both ANALYST and TACTICIAN predict continuous targets
                meta_model = CatBoostRegressor(
                    iterations=1000,
                    learning_rate=0.05,
                    depth=6,
                    loss_function='RMSE',
                    eval_metric='RMSE',
                    **performance_params
                )
            
            else:
                raise ValueError(f"Unsupported meta-learner type: {self.meta_learner_type}")
            
            # Align targets to match oof_predictions shape
            if len(targets) != len(oof_predictions):
                tprint_warning(f"⚠️ Aligning targets from {len(targets)} to {len(oof_predictions)} samples for meta-learner")
                # Get the index from oof_predictions if it's a DataFrame, otherwise use first N samples
                if isinstance(oof_predictions, pd.DataFrame):
                    targets = targets.loc[oof_predictions.index]
                else:
                    targets = targets.iloc[:len(oof_predictions)]
            
            # Train meta-learner
            meta_model.fit(oof_predictions, targets)
            
            # Calculate meta-learner metrics
            meta_predictions = meta_model.predict(oof_predictions)
            
            # Both ANALYST and TACTICIAN use regression metrics (both predict continuous targets)
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            metrics = {
                'meta_mse': mean_squared_error(targets, meta_predictions),
                'meta_mae': mean_absolute_error(targets, meta_predictions),
                'meta_r2': r2_score(targets, meta_predictions),
                'meta_rmse': np.sqrt(mean_squared_error(targets, meta_predictions))
            }
            
            # Also calculate comparison to simple average of base models
            if oof_predictions.shape[1] > 1:
                base_avg_predictions = oof_predictions.mean(axis=1)
                metrics['base_avg_r2'] = r2_score(targets, base_avg_predictions)
                metrics['base_avg_mse'] = mean_squared_error(targets, base_avg_predictions)
                metrics['improvement_over_avg'] = metrics['meta_r2'] - metrics['base_avg_r2']
                tprint_info(f"📊 Meta-learner R²: {metrics['meta_r2']:.4f} vs Base Average R²: {metrics['base_avg_r2']:.4f} (Δ: {metrics['improvement_over_avg']:+.4f})")
            
            # Store meta-learner
            self._meta_learner = meta_model
            
            return TrainingResult(
                success=True,
                model=meta_model,
                metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"Meta-learner training failed: {e}")
            return TrainingResult(success=False, error_message=str(e))
    
    async def _calculate_ensemble_metrics(
        self,
        base_predictions: pd.DataFrame,
        meta_result: TrainingResult,
        data: pd.DataFrame,
        targets: pd.Series
    ) -> Dict[str, float]:
        """Calculate comprehensive ensemble metrics."""
        try:
            ensemble_metrics = {}
            
            # Meta-learner metrics
            if meta_result.success and meta_result.metrics:
                ensemble_metrics.update(meta_result.metrics)
            
            # Ensemble performance metrics
            ensemble_predictions = await self._get_ensemble_predictions(data)
            
            if ensemble_predictions is not None and targets is not None:
                # Both ANALYST and TACTICIAN use regression metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                ensemble_metrics.update({
                    'ensemble_mse': mean_squared_error(targets, ensemble_predictions),
                    'ensemble_mae': mean_absolute_error(targets, ensemble_predictions),
                    'ensemble_r2': r2_score(targets, ensemble_predictions),
                    'ensemble_rmse': np.sqrt(mean_squared_error(targets, ensemble_predictions))
                })
            
            # Diversity metrics from base predictions
            diversity_metrics = self._calculate_diversity_metrics_from_predictions(base_predictions)
            ensemble_metrics.update(diversity_metrics)
            
            return ensemble_metrics
            
        except Exception as e:
            self.logger.error(f"Ensemble metrics calculation failed: {e}")
            return {}
    
    def _calculate_diversity_metrics_from_predictions(self, base_predictions: pd.DataFrame) -> Dict[str, float]:
        """Calculate diversity metrics from base model predictions DataFrame."""
        try:
            diversity_metrics = {}
            
            if base_predictions is None or base_predictions.empty or base_predictions.shape[1] < 2:
                return diversity_metrics
            
            # Calculate pairwise correlations between base model predictions
            correlations = []
            n_models = base_predictions.shape[1]
            
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    pred_i = base_predictions.iloc[:, i].values
                    pred_j = base_predictions.iloc[:, j].values
                    corr = np.corrcoef(pred_i, pred_j)[0, 1]
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
