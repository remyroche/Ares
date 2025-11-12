"""
Model Trainer - Individual Model Training Implementation

This module provides concrete implementations for training individual models
across different roles (Analyst, Tactician) with role-specific optimizations.

Key Features:
- Role-specific training logic (Analyst vs Tactician)
- Model-specific implementations (LightGBM, CatBoost, Neural Networks)
- Optimized training pipelines for different timeframes
- Enhanced feature engineering and selection
- Performance monitoring and optimization
"""

import logging
import os
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
from .training_metrics_collector import TrainingMetricsCollector, ModelMetrics

# Shared feature engineering
from src.feature_generation.shared.feature_engineer import (
    AnalystFeatureEngineer,
    TacticianFeatureEngineer
)


class ModelTrainer(BaseTrainer):
    """
    Individual model trainer implementation.
    
    This class provides concrete implementations for training individual models
    with role-specific optimizations and model-specific training logic.
    """
    
    def __init__(self, config: TrainingConfig, logger: Optional[logging.Logger] = None):
        """Initialize the model trainer."""
        super().__init__(config, logger)
        
        # Role-specific configuration
        self._setup_role_specific_config()
        
        # Model-specific state
        self._model_instances = {}
        self._training_histories = {}
        self._validation_histories = {}
        
        # Metrics collector
        self._metrics_collector = TrainingMetricsCollector(logger)
        
        # Shared feature engineers
        self._analyst_feature_engineer = AnalystFeatureEngineer(logger=logger)
        self._tactician_feature_engineer = TacticianFeatureEngineer(logger=logger)
    
    def _setup_role_specific_config(self):
        """Setup role-specific configuration."""
        if self.config.role == TrainingRole.ANALYST:
            # Analyst-specific optimizations
            self.config.custom_params.update({
                'enable_feature_interaction': True,
                'enable_regime_features': True,
                'confidence_threshold': 0.4,
                'timeframe_optimization': True
            })
        elif self.config.role == TrainingRole.TACTICIAN:
            # Tactician-specific optimizations
            self.config.custom_params.update({
                'enable_timing_features': True,
                'enable_analyst_signals': True,
                'enable_risk_features': True,
                'precision_optimization': True
            })
    
    @handles_errors(
        exceptions=(ValueError, RuntimeError, MemoryError),
        default_return=TrainingResult(success=False, error_message="Training failed"),
        context="model training"
    )
    async def train(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> TrainingResult:
        """
        Train individual models with role-specific optimizations and comprehensive metrics collection.
        
        Args:
            data: Training data
            targets: Target variables
            
        Returns:
            Training result with models and comprehensive metrics
        """
        try:
            self.logger.info(f"🚀 Starting {self.config.role.value} model training with comprehensive metrics...")
            start_time = time.time()
            
            # SINGLE SOURCE OF TRUTH: Check environment variable to override HPO
            # This takes precedence over all other config sources
            disable_hpo_env = os.getenv('DISABLE_HPO', 'false').lower() in ('true', '1', 'yes')
            if disable_hpo_env:
                tprint_warning("🚫 HPO DISABLED via DISABLE_HPO environment variable")
                tprint_info("   Using saved optimal parameters from config")
                self.config.enable_hyperparameter_optimization = False
            
            # Start metrics collection session
            training_type = f"{self.config.role.value}_base"
            self._metrics_collector.start_session(
                training_type=training_type,
                symbol=self.config.symbol,
                timeframe=self.config.timeframe
            )
            
            # Preprocess data
            tprint_info("=" * 80)
            tprint_info("📊 PREPROCESSING DATA FOR MODEL TRAINING")
            tprint_info("=" * 80)
            tprint_info(f"Input data shape: {data.shape}")
            tprint_info(f"Input columns (first 20): {list(data.columns[:20])}")
            
            processed_data, processed_targets = self._preprocess_data(data, targets)

            tprint_info(f"After preprocessing: {processed_data.shape}")
            tprint_info(f"Processed columns (first 20): {list(processed_data.columns[:20])}")
            tprint_info("=" * 80)

            # Validate dataset size before training
            n_samples = len(processed_data)
            n_features = len(processed_data.columns)
            samples_per_feature = n_samples / n_features if n_features > 0 else 0

            # Minimum recommended: 10-15 samples per feature for reliable training
            min_samples_recommended = n_features * 10
            min_samples_absolute = max(100, n_features * 3)  # Absolute minimum

            tprint_info(f"📊 Dataset validation:")
            tprint_info(f"   Samples: {n_samples}")
            tprint_info(f"   Features: {n_features}")
            tprint_info(f"   Samples/Feature ratio: {samples_per_feature:.2f}")
            tprint_info(f"   Recommended minimum: {min_samples_recommended} samples ({n_features} features × 10)")

            if n_samples < min_samples_absolute:
                error_msg = (
                    f"❌ CRITICAL: Dataset too small for reliable training!\n"
                    f"   Current: {n_samples} samples\n"
                    f"   Absolute minimum: {min_samples_absolute} samples\n"
                    f"   Recommended: {min_samples_recommended}+ samples\n"
                    f"   This will likely result in severe overfitting and poor generalization."
                )
                tprint_error(error_msg)
                self.logger.error(error_msg)
                # Don't fail immediately, but log prominent warning
                tprint_warning("⚠️  Proceeding with training despite insufficient data - expect overfitting!")
            elif n_samples < min_samples_recommended:
                warning_msg = (
                    f"⚠️  WARNING: Dataset is smaller than recommended!\n"
                    f"   Current: {n_samples} samples ({samples_per_feature:.1f} samples/feature)\n"
                    f"   Recommended: {min_samples_recommended}+ samples (10+ samples/feature)\n"
                    f"   Training may result in overfitting. Consider collecting more data."
                )
                tprint_warning(warning_msg)
                self.logger.warning(warning_msg)
            else:
                tprint_success(f"✅ Dataset size is adequate for training")

            # Train each model type with comprehensive metrics
            training_results = {}
            best_model = None
            best_metrics = {}
            all_model_metrics = []
            
            for model_type in self.config.model_types:
                self.logger.info(f"📊 Training {model_type.value} model with metrics collection...")
                
                # Create model for pre-HPO baseline
                model = self._create_model(model_type)
                # Allow None for models that are created during training (DEPTHWISE_CNN, Neural Networks)
                if model is None and model_type not in [ModelType.DEPTHWISE_CNN, ModelType.NEURAL_NETWORK]:
                    self.logger.error(f"Failed to create {model_type.value} model")
                    continue
                
                # Collect pre-HPO metrics
                tprint_info(f"📊 Phase 1: Collecting pre-HPO baseline metrics for {model_type.value}...")
                model_metrics = self._metrics_collector.collect_pre_hpo_metrics(
                    model_name=f"{self.config.role.value}_{model_type.value}",
                    model_type=model_type.value,
                    model=model,
                    X=processed_data,
                    y=processed_targets,
                    n_folds=self.config.cross_validation_folds
                )
                
                # Run hyperparameter optimization if enabled
                best_params = {}
                hpo_n_trials = 0
                hpo_time = 0.0
                
                if self.config.enable_hyperparameter_optimization:
                    tprint_info(f"🔧 Phase 2: Running hyperparameter optimization for {model_type.value}...")
                    hpo_start = time.time()
                    
                    model, best_params = await self._optimize_hyperparameters(
                        model, model_type, processed_data, processed_targets
                    )
                    
                    hpo_time = time.time() - hpo_start
                    hpo_n_trials = self.config.custom_params.get('hpo_n_trials', 50)
                    
                    tprint_success(f"✅ HPO completed in {hpo_time:.2f}s with {hpo_n_trials} trials")
                    
                    # CRITICAL FIX: Train final model with best params and get test metrics
                    # HPO only does CV, we need train/val/test split evaluation
                    tprint_info(f"🎯 Phase 3: Training final model with optimized parameters and evaluating on test set...")
                    model_result = await self._train_single_model(
                        model, model_type, processed_data, processed_targets, best_params
                    )
                else:
                    tprint_info(f"⏭️  Skipping HPO (disabled in config)")
                    
                    # Train model with default parameters
                    tprint_info(f"🎯 Phase 3: Training model with default parameters...")
                    model_result = await self._train_single_model(
                        model, model_type, processed_data, processed_targets
                    )
                
                if model_result.success:
                    # Collect post-HPO metrics using engineered data
                    tprint_info(f"📈 Phase 4: Collecting post-HPO metrics for {model_type.value}...")
                    # Use engineered data if available, otherwise use processed_data
                    data_for_metrics = model_result.metadata.get('engineered_data', processed_data)
                    model_metrics = self._metrics_collector.collect_post_hpo_metrics(
                        model_metrics=model_metrics,
                        model=model_result.model,
                        X=data_for_metrics,
                        y=processed_targets,
                        best_params=best_params,
                        hpo_n_trials=hpo_n_trials,
                        hpo_time=hpo_time,
                        n_folds=self.config.cross_validation_folds
                    )
                    
                    # Add to session
                    self._metrics_collector.add_model_metrics(model_metrics)
                    all_model_metrics.append(model_metrics)
                    
                    training_results[model_type.value] = model_result
                    self._model_instances[model_type.value] = model_result.model
                    
                    # Track best model
                    if not best_model or self._is_better_model(model_result.metrics, best_metrics):
                        best_model = model_result.model
                        best_metrics = model_result.metrics
                    
                    tprint_success(f"✅ {model_type.value} training completed with comprehensive metrics")
                    self.logger.info(f"✅ {model_type.value} training completed")
                else:
                    tprint_error(f"❌ {model_type.value} training failed: {model_result.error_message}")
                    self.logger.error(f"❌ {model_type.value} training failed: {model_result.error_message}")
            
            # Calculate overall metrics
            overall_metrics = self._calculate_overall_metrics(training_results)
            training_time = time.time() - start_time
            
            # Finalize session and generate report
            tprint_info("📝 Generating comprehensive training report...")
            session = self._metrics_collector.finalize_session(
                total_training_time=training_time,
                data_quality_score=0.85,  # TODO: Calculate actual quality score
                n_samples=len(processed_data),
                n_features=len(processed_data.columns)
            )
            
            # Generate and save report
            report_path = self._metrics_collector.save_report()
            
            # Update state
            self._training_state['training_completed'] = True
            self._training_state['training_started'] = True
            self._update_performance_metrics('training', training_time)
            
            # Extract trained models dict for reporting
            trained_models = {
                model_type.value: self._model_instances.get(model_type.value)
                for model_type in self.config.model_types
                if model_type.value in self._model_instances
            }
            
            result = TrainingResult(
                success=len(training_results) > 0,
                model=best_model,
                metrics=overall_metrics,
                training_time=training_time,
                metadata={
                    'models_trained': len(training_results),
                    'role': self.config.role.value,
                    'timeframe': self.config.timeframe,
                    'individual_results': training_results,
                    'comprehensive_metrics': all_model_metrics,
                    'report_path': str(report_path),
                    'trained_models': trained_models,  # Add models dict for reporting
                    'model_instances': self._model_instances,  # Store all model instances
                    'trained_feature_columns': list(processed_data.columns)  # CRITICAL: Store feature columns for prediction
                }
            )
            
            if result.success:
                self.logger.info(f"✅ Training completed successfully in {training_time:.2f}s")
                tprint_success(f"✅ Trained {len(training_results)} models with comprehensive metrics")
                tprint_success(f"📄 Report saved to: {report_path}")
            else:
                self.logger.error("❌ All model training failed")
                tprint_error("❌ Training failed for all models")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return TrainingResult(
                success=False,
                error_message=str(e),
                training_time=time.time() - start_time
            )
    
    async def _train_single_model(
        self, 
        model: Any, 
        model_type: ModelType, 
        data: pd.DataFrame, 
        targets: pd.Series,
        best_params: Optional[Dict[str, Any]] = None
    ) -> TrainingResult:
        """Train a single model with role-specific optimizations."""
        try:
            start_time = time.time()
            
            # Store best params for use in training functions
            if best_params:
                self._best_params = best_params
            else:
                self._best_params = {}
            
            # Role-specific feature engineering
            tprint_info(f"🔧 Training {model_type.value} - Feature engineering for {self.config.role.value}")
            tprint_info(f"   Input data: {data.shape}")
            
            engineered_data = data
            if self.config.role == TrainingRole.ANALYST:
                engineered_data = self._engineer_analyst_features(data, targets)
                tprint_info(f"   After ANALYST engineering: {engineered_data.shape}")
                tprint_info(f"   Engineered columns (first 20): {list(engineered_data.columns[:20])}")
            elif self.config.role == TrainingRole.TACTICIAN:
                engineered_data = self._engineer_tactician_features(data, targets)
                tprint_info(f"   After TACTICIAN engineering: {engineered_data.shape}")
            
            # CRITICAL: Align targets with engineered data if shape changed
            aligned_targets = targets
            if len(engineered_data) != len(targets):
                tprint_warning(f"⚠️ Feature engineering changed data shape: {len(data)} → {len(engineered_data)}")
                tprint_info(f"   Engineered data has {engineered_data.index.duplicated().sum()} duplicate indices")
                tprint_info(f"   Targets has {targets.index.duplicated().sum()} duplicate indices")
                
                # Remove duplicates from both engineered_data and targets
                if engineered_data.index.duplicated().any():
                    tprint_warning(f"   Removing {engineered_data.index.duplicated().sum()} duplicate indices from engineered_data")
                    engineered_data = engineered_data[~engineered_data.index.duplicated(keep='first')]
                
                if targets.index.duplicated().any():
                    tprint_warning(f"   Removing {targets.index.duplicated().sum()} duplicate indices from targets")
                    targets = targets[~targets.index.duplicated(keep='first')]
                
                # Align targets by index
                common_index = engineered_data.index.intersection(targets.index)
                if len(common_index) > 0:
                    aligned_targets = targets.loc[common_index]
                    engineered_data = engineered_data.loc[common_index]
                    tprint_success(f"✅ Aligned to {len(common_index)} common samples")
                    tprint_info(f"   After alignment: data shape={engineered_data.shape}, targets shape={aligned_targets.shape}")
                else:
                    raise ValueError("No common indices between engineered data and targets!")
            
            # Verify alignment before passing to model
            if len(engineered_data) != len(aligned_targets):
                raise ValueError(f"CRITICAL: Alignment failed! data={len(engineered_data)}, targets={len(aligned_targets)}")
            
            # Model-specific training
            result = None
            if model_type == ModelType.LIGHTGBM:
                result = await self._train_lightgbm_model(model, engineered_data, aligned_targets)
            elif model_type == ModelType.DEPTHWISE_CNN:
                result = await self._train_depthwise_cnn_model(model, engineered_data, aligned_targets)
            elif model_type == ModelType.CATBOOST:
                result = await self._train_catboost_model(model, engineered_data, aligned_targets)
            elif model_type == ModelType.NEURAL_NETWORK:
                result = await self._train_neural_network_model(model, engineered_data, aligned_targets)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Store engineered data in result metadata for post-HPO metrics
            if result and result.success:
                result.metadata['engineered_data'] = engineered_data
            
            return result
                
        except Exception as e:
            self.logger.error(f"Single model training failed: {e}")
            return TrainingResult(success=False, error_message=str(e))
    
    async def _optimize_hyperparameters(
        self,
        model: Any,
        model_type: ModelType,
        data: pd.DataFrame,
        targets: pd.Series
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Optimize hyperparameters using Bayesian TPE optimization.

        Args:
            model: Base model to optimize
            model_type: Type of model
            data: Training data
            targets: Training targets

        Returns:
            Tuple of (optimized_model, best_params)
        """
        try:
            # Get HPO config
            n_trials = self.config.custom_params.get('hpo_n_trials', 50)

            # Define search spaces based on model type
            if model_type == ModelType.LIGHTGBM:
                search_space = {
                    'num_leaves': ('int', 20, 100),
                    'learning_rate': ('float', 0.01, 0.1),
                    'feature_fraction': ('float', 0.6, 1.0),
                    'bagging_fraction': ('float', 0.6, 1.0),
                    'min_child_samples': ('int', 5, 50)
                }
            elif model_type == ModelType.CATBOOST:
                search_space = {
                    'depth': ('int', 4, 10),
                    'learning_rate': ('float', 0.01, 0.1),
                    'l2_leaf_reg': ('float', 1, 10),
                    'border_count': ('int', 32, 255)
                }
            elif model_type == ModelType.DEPTHWISE_CNN:
                # CNN HPO is handled inline during training due to model architecture
                # Return None to signal that HPO will happen during training
                tprint_info(f"ℹ️  HPO for {model_type.value} will be handled during model training (inline)")
                return model, {}
            else:
                # Return model as-is for types without HPO
                tprint_info(f"ℹ️  HPO not configured for {model_type.value}, using default parameters")
                return model, {}
            
            # Use BayesianTPEOptimizer
            optimizer = BayesianTPEOptimizer()
            
            # Validate dataset size for cross-validation
            min_samples_required = 10  # Minimum samples needed for meaningful CV
            if len(data) < min_samples_required:
                self.logger.warning(f"⚠️ Dataset too small for HPO ({len(data)} samples < {min_samples_required}), skipping optimization")
                return model, {}
            
            # Adjust CV folds based on dataset size
            cv_folds = min(3, max(2, len(data) // 5))  # At least 5 samples per fold
            
            # Define objective function
            def objective(params):
                try:
                    # Create model with params
                    if model_type == ModelType.LIGHTGBM:
                        import lightgbm as lgb
                        test_model = lgb.LGBMRegressor(**params, verbose=-1)
                    elif model_type == ModelType.CATBOOST:
                        from catboost import CatBoostRegressor
                        test_model = CatBoostRegressor(**params, verbose=False)
                    
                    # Cross-validate with adjusted folds
                    from sklearn.model_selection import cross_val_score, KFold
                    
                    # Use KFold with shuffle to avoid empty folds
                    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    
                    scores = cross_val_score(
                        test_model, data, targets, 
                        cv=kfold, scoring='r2', n_jobs=-1
                    )
                    
                    # Filter out any invalid scores
                    valid_scores = [s for s in scores if not np.isnan(s) and not np.isinf(s)]
                    if not valid_scores:
                        return -999999  # No valid scores
                    
                    return np.mean(valid_scores)
                except Exception as e:
                    # Log specific error types for debugging
                    if "0 sample" in str(e) or "empty" in str(e).lower():
                        self.logger.warning(f"⚠️ HPO trial failed due to empty fold: {e}")
                    else:
                        self.logger.warning(f"⚠️ HPO trial failed: {e}")
                    return -999999  # Very bad score
            
            # Run optimization
            best_params = {}
            best_score = -float('inf')
            
            for trial in range(n_trials):
                # Sample parameters
                trial_params = {}
                for param_name, param_spec in search_space.items():
                    if param_spec[0] == 'int':
                        trial_params[param_name] = np.random.randint(param_spec[1], param_spec[2] + 1)
                    elif param_spec[0] == 'float':
                        trial_params[param_name] = np.random.uniform(param_spec[1], param_spec[2])
                
                # Evaluate
                score = objective(trial_params)
                
                # Update best
                if score > best_score:
                    best_score = score
                    best_params = trial_params.copy()
            
            # Create optimized model
            if model_type == ModelType.LIGHTGBM:
                import lightgbm as lgb
                optimized_model = lgb.LGBMRegressor(**best_params, verbose=-1)
            elif model_type == ModelType.CATBOOST:
                from catboost import CatBoostRegressor
                optimized_model = CatBoostRegressor(**best_params, verbose=False)
            else:
                optimized_model = model
            
            self.logger.info(f"✅ HPO completed: best score = {best_score:.4f}")
            return optimized_model, best_params
            
        except Exception as e:
            self.logger.error(f"HPO failed: {e}, using default model")
            return model, {}
    
    def _engineer_analyst_features(self, data: pd.DataFrame, targets: pd.Series) -> pd.DataFrame:
        """Engineer features specific to Analyst role using shared module."""
        try:
            # Use shared feature engineer for consistency with inference
            engineered_data = self._analyst_feature_engineer.engineer_features(data)
            return engineered_data
            
        except Exception as e:
            self.logger.warning(f"Analyst feature engineering failed: {e}")
            return data
    
    def _engineer_tactician_features(self, data: pd.DataFrame, targets: pd.Series) -> pd.DataFrame:
        """Engineer features specific to Tactician role using shared module."""
        try:
            # Use shared feature engineer for consistency with inference
            # Extract analyst confidence from data if available
            analyst_confidence = None
            if 'analyst_confidence' in data.columns:
                analyst_confidence = data['analyst_confidence'].iloc[-1] if len(data) > 0 else None
            
            engineered_data = self._tactician_feature_engineer.engineer_features(
                data,
                analyst_confidence=analyst_confidence
            )
            return engineered_data
            
        except Exception as e:
            self.logger.warning(f"Tactician feature engineering failed: {e}")
            return data
    
    async def _train_lightgbm_model(self, model: Any, data: pd.DataFrame, targets: pd.Series) -> TrainingResult:
        """Train LightGBM model with role-specific parameters from YAML config."""
        try:
            # Import LightGBM
            import lightgbm as lgb
            from sklearn.model_selection import train_test_split
            import os
            from src.utils.tprint import tprint_data_preview, tprint_info

            # Log model-specific training start
            tprint_info("=" * 80)
            tprint_info("🌟 MODEL-SPECIFIC TRAINING: LightGBM")
            tprint_info("=" * 80)
            tprint_data_preview(
                data,
                name="LightGBM Training Data",
                max_rows=5,
                max_cols=10,
                show_dtypes=True,
                show_shape=True
            )
            tprint_data_preview(
                targets.to_frame() if isinstance(targets, pd.Series) else targets,
                name="LightGBM Target Data",
                max_rows=5,
                max_cols=10,
                show_dtypes=True,
                show_shape=True
            )

            # Get CPU optimizer for threading
            cpu_optimizer = get_m1_cpu_optimizer()
            n_threads = cpu_optimizer.get_optimal_thread_count() if cpu_optimizer else (os.cpu_count() or 4)

            tprint_info(f"🔧 LightGBM Configuration:")
            tprint_info(f"   CPU Threads: {n_threads}")
            tprint_info(f"   Training samples: {len(data)}")
            tprint_info(f"   Features: {len(data.columns)}")

            # CRITICAL FIX: Split data into train/val/test (70/15/15) for proper evaluation
            # This prevents overfitting detection issues and provides honest metrics
            
            # First split: separate test set (15%)
            X_temp, X_test, y_temp, y_test = train_test_split(
                data, targets, test_size=0.15, random_state=42, shuffle=False  # No shuffle for time series
            )
            
            # Second split: train (70%) and validation (15%) from remaining 85%
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.176, random_state=42, shuffle=False  # 0.176 * 0.85 ≈ 0.15
            )

            tprint_info(f"   📊 Data splits (temporal order preserved):")
            tprint_info(f"      Train: {len(X_train)} samples ({len(X_train)/len(data)*100:.1f}%)")
            tprint_info(f"      Val: {len(X_val)} samples ({len(X_val)/len(data)*100:.1f}%)")
            tprint_info(f"      Test: {len(X_test)} samples ({len(X_test)/len(data)*100:.1f}%)")
            tprint_info("=" * 80)
            
            # CRITICAL FIX: Load optimal params from YAML config when HPO is disabled
            if hasattr(self, '_best_params') and self._best_params:
                tprint_info(f"   Using HPO-optimized parameters from current session")
                model_params = self._best_params
            else:
                # Load optimal params from YAML config file
                tprint_info(f"   Loading optimal parameters from YAML config")
                import yaml
                from pathlib import Path
                config_path = Path('src/training/steps/model_training/analyst_base_config.yaml')
                if config_path.exists():
                    with open(config_path) as f:
                        yaml_config = yaml.safe_load(f)
                        lgbm_config = yaml_config.get('analyst_config', {}).get('base_models', {}).get('lgbm', {})
                        model_params = lgbm_config.get('params', {})
                        tprint_success(f"   ✅ Loaded {len(model_params)} parameters from YAML config")
                else:
                    tprint_warning(f"   ⚠️  YAML config not found, using defaults")
                    model_params = {}
            
            # Extract hyperparameters (from HPO, YAML, or defaults)
            n_estimators = model_params.get('n_estimators', 1000)
            learning_rate = model_params.get('learning_rate', 0.1)
            max_depth = model_params.get('max_depth', 8)
            num_leaves = model_params.get('num_leaves', 31)
            subsample = model_params.get('subsample', 0.8)
            colsample_bytree = model_params.get('colsample_bytree', 0.8)
            reg_alpha = model_params.get('reg_alpha', 0.0)
            reg_lambda = model_params.get('reg_lambda', 0.0)
            min_child_samples = model_params.get('min_child_samples', 20)
            
            # Adjust parameters for small datasets to avoid "no more leaves" warning
            n_samples = len(X_train)
            if n_samples < 100:
                # For very small datasets, use more conservative parameters
                min_child_samples = max(1, min(min_child_samples, n_samples // 10))
                num_leaves = min(num_leaves, max(7, n_samples // 3))
                max_depth = min(max_depth, 5)
                self.logger.info(f"📊 Adjusted LightGBM params for small dataset ({n_samples} samples): "
                               f"num_leaves={num_leaves}, max_depth={max_depth}, min_child_samples={min_child_samples}")
            
            # Build parameters dictionary
            params = {
                # Task configuration
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                
                # Hyperparameters (from YAML/HPO)
                'num_leaves': num_leaves,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'min_child_samples': min_child_samples,
                
                # Performance optimizations (NOT tuned by HPO)
                'n_jobs': n_threads,
                'bagging_freq': 5,
                'verbose': -1,
                'force_col_wise': True,  # Faster for many features
            }
            
            tprint_info(f"Training LightGBM: depth={max_depth}, leaves={num_leaves}, lr={learning_rate}")
            tprint_info(f"📊 LightGBM training data: {data.shape}")
            tprint_info(f"   Feature columns (first 20): {list(data.columns[:20])}")
            tprint_info(f"   Feature columns (last 10): {list(data.columns[-10:])}")
            
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train model WITHOUT early stopping for final training
            # Early stopping is only used during HPO
            tprint_info(f"   Training for full {n_estimators} iterations (no early stopping)")
            model = lgb.train(
                params,
                train_data,
                num_boost_round=n_estimators,
                valid_sets=[valid_data],
                callbacks=[]  # No early stopping for final training
            )
            
            # CRITICAL FIX: Evaluate on train/val/test splits separately
            # This provides honest metrics and detects overfitting
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            # Predictions on each split
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            
            # Calculate directional accuracy (for regression)
            # Accuracy = % of predictions within acceptable error threshold
            def calculate_accuracy(y_true, y_pred, threshold=0.1):
                """Calculate accuracy as % of predictions within threshold of true value"""
                errors = np.abs(y_true - y_pred)
                within_threshold = errors <= threshold
                return np.mean(within_threshold)
            
            # Metrics for each split
            metrics = {
                # Training set metrics
                'train_mse': mean_squared_error(y_train, train_pred),
                'train_mae': mean_absolute_error(y_train, train_pred),
                'train_r2': r2_score(y_train, train_pred),
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'train_accuracy': calculate_accuracy(y_train, train_pred),
                
                # Validation set metrics
                'val_mse': mean_squared_error(y_val, val_pred),
                'val_mae': mean_absolute_error(y_val, val_pred),
                'val_r2': r2_score(y_val, val_pred),
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                'val_accuracy': calculate_accuracy(y_val, val_pred),
                
                # Test set metrics (CRITICAL - unseen data)
                'test_mse': mean_squared_error(y_test, test_pred),
                'test_mae': mean_absolute_error(y_test, test_pred),
                'test_r2': r2_score(y_test, test_pred),
                'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'test_accuracy': calculate_accuracy(y_test, test_pred),
                
                # Overfitting analysis
                'train_test_r2_gap': r2_score(y_train, train_pred) - r2_score(y_test, test_pred),
                'overfitting_ratio': (r2_score(y_train, train_pred) - r2_score(y_test, test_pred)) / max(r2_score(y_train, train_pred), 0.01),
                'generalization_score': r2_score(y_test, test_pred) / max(r2_score(y_train, train_pred), 0.01),
                
                # Legacy metrics (for backward compatibility - use test metrics)
                'mse': mean_squared_error(y_test, test_pred),
                'mae': mean_absolute_error(y_test, test_pred),
                'r2': r2_score(y_test, test_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                
                # Model info
                'iterations_used': model.current_iteration(),
                'best_iteration': model.best_iteration
            }
            
            # Get feature importance
            feature_importance = dict(zip(data.columns, model.feature_importance()))
            
            # Log comprehensive results
            tprint_success(f"✅ LightGBM trained: {model.current_iteration()} iterations")
            tprint_info(f"   📊 Train R²: {metrics['train_r2']:.4f}, RMSE: {metrics['train_rmse']:.4f}, Accuracy: {metrics['train_accuracy']:.2%}")
            tprint_info(f"   📊 Val R²: {metrics['val_r2']:.4f}, RMSE: {metrics['val_rmse']:.4f}, Accuracy: {metrics['val_accuracy']:.2%}")
            tprint_info(f"   📊 Test R²: {metrics['test_r2']:.4f}, RMSE: {metrics['test_rmse']:.4f}, Accuracy: {metrics['test_accuracy']:.2%}")
            tprint_info(f"   ⚠️  Train-Test Gap: {metrics['train_test_r2_gap']:.4f} ({metrics['overfitting_ratio']*100:.1f}%)")
            
            # Overfitting warning
            if metrics['overfitting_ratio'] > 0.2:
                tprint_warning(f"   ⚠️  HIGH OVERFITTING detected! Model may not generalize well.")
            elif metrics['overfitting_ratio'] > 0.1:
                tprint_warning(f"   ⚠️  Moderate overfitting detected.")
            else:
                tprint_success(f"   ✅ Good generalization (overfitting ratio < 10%)")
            
            return TrainingResult(
                success=True,
                model=model,
                metrics=metrics,
                feature_importance=feature_importance
            )
            
        except Exception as e:
            self.logger.error(f"LightGBM training failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return TrainingResult(success=False, error_message=str(e))
    
    async def _train_catboost_model(self, model: Any, data: pd.DataFrame, targets: pd.Series) -> TrainingResult:
        """Train CatBoost model with role-specific parameters from YAML config."""
        try:
            import numpy as np
            from src.utils.tprint import tprint_data_preview, tprint_info, tprint_warning

            # Log model-specific training start
            tprint_info("=" * 80)
            tprint_info("🌟 MODEL-SPECIFIC TRAINING: CatBoost")
            tprint_info("=" * 80)
            tprint_data_preview(
                data,
                name="CatBoost Training Data (Before Filtering)",
                max_rows=5,
                max_cols=10,
                show_dtypes=True,
                show_shape=True
            )

            # Validate shape alignment (should already be aligned by _train_single_model)
            if len(data) != len(targets):
                raise ValueError(f"Shape mismatch: data={len(data)}, targets={len(targets)}. This should have been fixed upstream!")

            # CRITICAL: Keep ONLY numeric columns for CatBoost (drop datetime, string, categorical)
            original_cols = data.columns.tolist()
            data = data.select_dtypes(include=[np.number]).copy()

            if len(data.columns) < len(original_cols):
                dropped_cols = set(original_cols) - set(data.columns)
                tprint_warning(f"⚠️ Dropped {len(dropped_cols)} non-numeric columns: {dropped_cols}")
                tprint_info(f"✅ Using {len(data.columns)} numeric features: {list(data.columns)}")

                # Log after filtering
                tprint_data_preview(
                    data,
                    name="CatBoost Training Data (After Filtering)",
                    max_rows=5,
                    max_cols=10,
                    show_dtypes=True,
                    show_shape=True
                )

            # Import CatBoost
            from catboost import CatBoostRegressor
            from sklearn.model_selection import train_test_split
            import os

            # Get GPU manager for hardware acceleration
            gpu_manager = get_m1_gpu_manager()
            cpu_optimizer = get_m1_cpu_optimizer()

            # Determine optimal thread count from hardware
            n_threads = cpu_optimizer.get_optimal_thread_count() if cpu_optimizer else (os.cpu_count() or 4)

            # Check GPU availability for CatBoost
            gpu_available = gpu_manager.is_m1 if gpu_manager else False

            tprint_info(f"🔧 CatBoost Configuration:")
            tprint_info(f"   CPU Threads: {n_threads}")
            tprint_info(f"   GPU Available: {gpu_available}")
            tprint_info(f"   Training samples: {len(data)}")
            tprint_info(f"   Features: {len(data.columns)}")

            # CRITICAL FIX: Split data into train/val/test (70/15/15) for proper evaluation
            # First split: separate test set (15%)
            X_temp, X_test, y_temp, y_test = train_test_split(
                data, targets, test_size=0.15, random_state=42, shuffle=False
            )
            
            # Second split: train (70%) and validation (15%) from remaining 85%
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.176, random_state=42, shuffle=False
            )

            tprint_info(f"   📊 Data splits (temporal order preserved):")
            tprint_info(f"      Train: {len(X_train)} samples ({len(X_train)/len(data)*100:.1f}%)")
            tprint_info(f"      Val: {len(X_val)} samples ({len(X_val)/len(data)*100:.1f}%)")
            tprint_info(f"      Test: {len(X_test)} samples ({len(X_test)/len(data)*100:.1f}%)")
            tprint_info("=" * 80)
            
            # CRITICAL FIX: Load optimal params from YAML config when HPO is disabled
            if hasattr(self, '_best_params') and self._best_params:
                tprint_info(f"   Using HPO-optimized parameters from current session")
                model_params = self._best_params
            else:
                # Load optimal params from YAML config file
                tprint_info(f"   Loading optimal parameters from YAML config")
                import yaml
                from pathlib import Path
                config_path = Path('src/training/steps/model_training/analyst_base_config.yaml')
                if config_path.exists():
                    with open(config_path) as f:
                        yaml_config = yaml.safe_load(f)
                        catboost_config = yaml_config.get('analyst_config', {}).get('base_models', {}).get('catboost', {})
                        model_params = catboost_config.get('params', {})
                        tprint_success(f"   ✅ Loaded {len(model_params)} parameters from YAML config")
                else:
                    tprint_warning(f"   ⚠️  YAML config not found, using defaults")
                    model_params = {}
            
            # Extract hyperparameters (from HPO, YAML, or defaults)
            iterations = model_params.get('iterations', 500)
            learning_rate = model_params.get('learning_rate', 0.1)
            depth = model_params.get('depth', 6)
            l2_leaf_reg = model_params.get('l2_leaf_reg', 3.0)
            subsample = model_params.get('subsample', 0.8)
            colsample_bylevel = model_params.get('colsample_bylevel', 0.8)
            border_count = model_params.get('border_count', 128)
            max_ctr_complexity = model_params.get('max_ctr_complexity', 2)
            early_stopping_rounds = model_params.get('early_stopping_rounds', 50)
            random_seed = model_params.get('random_seed', 42)
            
            # Performance optimizations (NOT tuned by HPO)
            performance_params = {
                'thread_count': n_threads,  # Use all available CPU threads
                'bootstrap_type': 'Bernoulli',  # Changed from 'Bayesian' to support subsample parameter
                'verbose': False,
                'random_seed': random_seed,
                'allow_writing_files': False,  # Disable temp file writing
            }
            
            # CatBoost GPU support: Disable for now as it's not properly configured on M1/M2/M3 Macs
            # Can be enabled later when CatBoost properly supports Metal GPU
            performance_params['task_type'] = 'CPU'
            gpu_available = False  # Force CPU to enable colsample_bylevel
            tprint_info(f"🔧 CatBoost using CPU with {n_threads} threads")
            
            # Hyperparameters (tuned by HPO)
            hyperparameters = {
                'iterations': iterations,
                'learning_rate': learning_rate,
                'depth': depth,
                'l2_leaf_reg': l2_leaf_reg,
                'subsample': subsample,
                'border_count': border_count,
                'max_ctr_complexity': max_ctr_complexity,
            }
            
            # Only add colsample_bylevel if not using GPU (GPU doesn't support RSM in non-pairwise modes)
            if not gpu_available:
                hyperparameters['colsample_bylevel'] = colsample_bylevel
            
            # Combine all parameters
            all_params = {**performance_params, **hyperparameters}
            
            # Add loss function and eval metric based on role
            all_params['loss_function'] = 'RMSE'
            all_params['eval_metric'] = 'RMSE'
            
            # Create CatBoost model (always Regressor for continuous targets)
            model = CatBoostRegressor(**all_params)
            
            tprint_info(f"Training CatBoost: depth={depth}, iterations={iterations}, lr={learning_rate}")
            tprint_info(f"📊 CatBoost training data: {data.shape}")
            tprint_info(f"   Feature columns (first 20): {list(data.columns[:20])}")
            tprint_info(f"   Feature columns (last 10): {list(data.columns[-10:])}")
            
            # Train model WITHOUT early stopping for final training
            # Early stopping is only used during HPO
            tprint_info(f"   Training for full {iterations} iterations (no early stopping)")
            model.fit(
                X_train, y_train, 
                eval_set=(X_val, y_val), 
                early_stopping_rounds=None,  # No early stopping for final training
                verbose=False,
                use_best_model=False  # Train for full iterations
            )
            
            # CRITICAL FIX: Evaluate on train/val/test splits separately
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            # Predictions on each split
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            
            # Calculate directional accuracy (for regression)
            def calculate_accuracy(y_true, y_pred, threshold=0.1):
                """Calculate accuracy as % of predictions within threshold of true value"""
                errors = np.abs(y_true - y_pred)
                within_threshold = errors <= threshold
                return np.mean(within_threshold)
            
            # Metrics for each split
            metrics = {
                # Training set metrics
                'train_mse': mean_squared_error(y_train, train_pred),
                'train_mae': mean_absolute_error(y_train, train_pred),
                'train_r2': r2_score(y_train, train_pred),
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'train_accuracy': calculate_accuracy(y_train, train_pred),
                
                # Validation set metrics
                'val_mse': mean_squared_error(y_val, val_pred),
                'val_mae': mean_absolute_error(y_val, val_pred),
                'val_r2': r2_score(y_val, val_pred),
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                'val_accuracy': calculate_accuracy(y_val, val_pred),
                
                # Test set metrics (CRITICAL - unseen data)
                'test_mse': mean_squared_error(y_test, test_pred),
                'test_mae': mean_absolute_error(y_test, test_pred),
                'test_r2': r2_score(y_test, test_pred),
                'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'test_accuracy': calculate_accuracy(y_test, test_pred),
                
                # Overfitting analysis
                'train_test_r2_gap': r2_score(y_train, train_pred) - r2_score(y_test, test_pred),
                'overfitting_ratio': (r2_score(y_train, train_pred) - r2_score(y_test, test_pred)) / max(r2_score(y_train, train_pred), 0.01),
                'generalization_score': r2_score(y_test, test_pred) / max(r2_score(y_train, train_pred), 0.01),
                
                # Legacy metrics (for backward compatibility - use test metrics)
                'mse': mean_squared_error(y_test, test_pred),
                'mae': mean_absolute_error(y_test, test_pred),
                'r2': r2_score(y_test, test_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                
                # Model info
                'iterations_used': model.get_best_iteration() or model.tree_count_,
                'best_iteration': model.get_best_iteration()
            }
            
            # Get feature importance
            feature_importance = dict(zip(data.columns, model.get_feature_importance()))
            
            # Log comprehensive results
            tprint_success(f"✅ CatBoost trained: {metrics['iterations_used']} iterations")
            tprint_info(f"   📊 Train R²: {metrics['train_r2']:.4f}, RMSE: {metrics['train_rmse']:.4f}, Accuracy: {metrics['train_accuracy']:.2%}")
            tprint_info(f"   📊 Val R²: {metrics['val_r2']:.4f}, RMSE: {metrics['val_rmse']:.4f}, Accuracy: {metrics['val_accuracy']:.2%}")
            tprint_info(f"   📊 Test R²: {metrics['test_r2']:.4f}, RMSE: {metrics['test_rmse']:.4f}, Accuracy: {metrics['test_accuracy']:.2%}")
            tprint_info(f"   ⚠️  Train-Test Gap: {metrics['train_test_r2_gap']:.4f} ({metrics['overfitting_ratio']*100:.1f}%)")
            
            # Overfitting warning
            if metrics['overfitting_ratio'] > 0.2:
                tprint_warning(f"   ⚠️  HIGH OVERFITTING detected! Model may not generalize well.")
            elif metrics['overfitting_ratio'] > 0.1:
                tprint_warning(f"   ⚠️  Moderate overfitting detected.")
            else:
                tprint_success(f"   ✅ Good generalization (overfitting ratio < 10%)")
            
            return TrainingResult(
                success=True,
                model=model,
                metrics=metrics,
                feature_importance=feature_importance
            )
            
        except Exception as e:
            self.logger.error(f"CatBoost training failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return TrainingResult(success=False, error_message=str(e))
    
    async def _train_depthwise_cnn_model(self, model: Any, data: pd.DataFrame, targets: pd.Series) -> TrainingResult:
        """Train DepthwiseSeparableCNN model with role-specific parameters."""
        try:
            # Import DepthwiseCNN model
            from src.models.tcn_regressor import DepthwiseSeparableCNNRegressor
            import numpy as np
            from src.utils.tprint import tprint_data_preview, tprint_info

            # Log model-specific training start
            tprint_info("=" * 80)
            tprint_info("🌟 MODEL-SPECIFIC TRAINING: DepthwiseCNN")
            tprint_info("=" * 80)
            tprint_data_preview(
                data,
                name="DepthwiseCNN Training Data",
                max_rows=5,
                max_cols=10,
                show_dtypes=True,
                show_shape=True
            )
            tprint_data_preview(
                targets.to_frame() if isinstance(targets, pd.Series) else targets,
                name="DepthwiseCNN Target Data",
                max_rows=5,
                max_cols=10,
                show_dtypes=True,
                show_shape=True
            )

            self.logger.info("🔷 Training DepthwiseSeparableCNN model...")

            # Note: We don't need to split here - the model handles validation_split internally
            # Just use the full dataset for training

            # Get model parameters from config
            model_params = self.config.custom_params.get('depthwise_cnn', {})
            if isinstance(model_params, dict) and 'params' in model_params:
                model_params = model_params['params']

            # Extract hyperparameters (with defaults)
            filters = model_params.get('filters', 64)
            kernel_size = model_params.get('kernel_size', 3)
            dropout = model_params.get('dropout', 0.2)
            learning_rate = model_params.get('learning_rate', 0.001)
            batch_size = model_params.get('batch_size', 64)
            epochs = model_params.get('epochs', 50)
            validation_split = model_params.get('validation_split', 0.2)
            early_stopping_patience = model_params.get('early_stopping_patience', 7)
            reduce_lr_patience = model_params.get('reduce_lr_patience', 5)
            use_batch_norm = model_params.get('use_batch_norm', False)
            verbose = model_params.get('verbose', 0)

            tprint_info(f"🔧 DepthwiseCNN Configuration:")
            tprint_info(f"   Filters: {filters}")
            tprint_info(f"   Kernel Size: {kernel_size}")
            tprint_info(f"   Dropout: {dropout}")
            tprint_info(f"   Learning Rate: {learning_rate}")
            tprint_info(f"   Batch Size: {batch_size}")
            tprint_info(f"   Epochs: {epochs}")
            tprint_info(f"   Validation Split: {validation_split}")
            tprint_info(f"   Training samples: {len(data)}")
            tprint_info(f"   Features: {len(data.columns)}")
            tprint_info(f"   Feature columns (first 20): {list(data.columns[:20])}")
            tprint_info(f"   Feature columns (last 10): {list(data.columns[-10:])}")
            tprint_info("=" * 80)
            
            # Create model
            cnn_model = DepthwiseSeparableCNNRegressor(
                filters=filters,
                kernel_size=kernel_size,
                dropout=dropout,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split,
                early_stopping_patience=early_stopping_patience,
                reduce_lr_patience=reduce_lr_patience,
                use_batch_norm=use_batch_norm,
                verbose=verbose
            )
            
            # Train model
            start_time = time.time()
            cnn_model.fit(
                data.values if isinstance(data, pd.DataFrame) else data,
                targets.values if isinstance(targets, pd.Series) else targets
            )
            training_time = time.time() - start_time
            
            # Make predictions
            predictions = cnn_model.predict(
                data.values if isinstance(data, pd.DataFrame) else data
            )
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            mse = mean_squared_error(targets, predictions)
            mae = mean_absolute_error(targets, predictions)
            r2 = r2_score(targets, predictions)
            rmse = np.sqrt(mse)
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': rmse,
                'training_time': training_time
            }
            
            tprint_success(f"✅ DepthwiseCNN trained: R²={r2:.4f}, RMSE={rmse:.4f}, Time={training_time:.2f}s")
            
            return TrainingResult(
                success=True,
                model=cnn_model,
                predictions=predictions,
                metrics=metrics,
                training_time=training_time,
                metadata={
                    'model_type': 'DepthwiseSeparableCNNRegressor',
                    'filters': filters,
                    'kernel_size': kernel_size,
                    'dropout': dropout,
                    'epochs': epochs
                }
            )
            
        except Exception as e:
            self.logger.error(f"DepthwiseCNN training failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            traceback.print_exc()
            return TrainingResult(success=False, error_message=str(e))
    
    async def _train_tcn_model(self, model: Any, data: pd.DataFrame, targets: pd.Series) -> TrainingResult:
        """Train Temporal Convolutional Network model with role-specific parameters."""
        try:
            # Import TCN model
            from src.models.causal_dilated_tcn import CausalDilatedTCNModel, CausalTCNConfig
            from sklearn.model_selection import train_test_split
            
            # Split data for validation
            X_train, X_val, y_train, y_val = train_test_split(
                data, targets, test_size=0.2, random_state=42
            )
            
            # Get model parameters from YAML config
            model_params = {}
            hpo_enabled = False
            hpo_config = None
            if hasattr(model, 'get_params'):
                try:
                    model_params = model.get_params()
                    if 'hpo' in model_params and isinstance(model_params['hpo'], dict):
                        hpo_config = model_params['hpo']
                        hpo_enabled = hpo_config.get('enabled', False)
                except:
                    pass
            
            # Extract hyperparameters from YAML (or use defaults)
            num_filters = model_params.get('num_filters', 64)
            num_layers = model_params.get('num_layers', 4)
            kernel_size = model_params.get('kernel_size', 3)
            dilation_base = model_params.get('dilation_base', 2)
            dropout = model_params.get('dropout', 0.2 if self.config.role == TrainingRole.ANALYST else 0.1)
            learning_rate = model_params.get('learning_rate', 0.001)
            batch_size = model_params.get('batch_size', 64)
            epochs = model_params.get('epochs', 50)
            early_stopping_patience = model_params.get('early_stopping_patience', 7)
            use_autoencoder = model_params.get('use_autoencoder', True)
            latent_dim = model_params.get('latent_dim', 16)
            train_autoencoder_if_missing = model_params.get('train_autoencoder_if_missing', True)
            autoencoder_epochs = model_params.get('autoencoder_epochs', 25)
            
            # Determine autoencoder path based on role
            autoencoder_path = model_params.get(
                'autoencoder_path',
                "models/analyst_autoencoder_encoder.pth" if self.config.role == TrainingRole.ANALYST else "models/tactician_autoencoder_encoder.pth"
            )
            
            # Create TCN configuration from YAML parameters
            tcn_config = CausalTCNConfig(
                num_filters=num_filters,
                num_layers=num_layers,
                kernel_size=kernel_size,
                dilation_base=dilation_base,
                dropout=dropout,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs,
                early_stopping_patience=early_stopping_patience,
                use_autoencoder=use_autoencoder,
                autoencoder_path=autoencoder_path,
                latent_dim=latent_dim,
                train_autoencoder_if_missing=train_autoencoder_if_missing,
                autoencoder_epochs=autoencoder_epochs
            )
            
            tprint_info(f"Training TCN: layers={num_layers}, filters={num_filters}, latent_dim={latent_dim}")
            
            # Run HPO if enabled
            if hpo_enabled and hpo_config:
                self.logger.info("🎯 Hierarchical HPO enabled - optimizing hyperparameters...")
                try:
                    from src.training.steps.models_training.core.tcn_autoencoder_hpo import AutoencoderTCNHPO
                    from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import OptimizationStage
                    
                    # Map stage strings to enums
                    stage_map = {
                        'coarse_grid': OptimizationStage.COARSE_GRID,
                        'fine_grid': OptimizationStage.FINE_GRID,
                        'tpe': OptimizationStage.TPE,
                        'bohb': OptimizationStage.BOHB,
                        'random': OptimizationStage.RANDOM
                    }
                    stages = [stage_map[s] for s in hpo_config.get('stages', ['coarse_grid', 'fine_grid', 'tpe'])]
                    
                    # Create HPO instance
                    hpo = AutoencoderTCNHPO(
                        role=self.config.role.value if hasattr(self.config.role, 'value') else str(self.config.role),
                        metric=hpo_config.get('metric', 'accuracy'),
                        n_rounds=hpo_config.get('n_rounds', 2),
                        stages=stages,
                        enable_final_refinement=hpo_config.get('enable_final_refinement', True),
                        final_refinement_trials=hpo_config.get('final_refinement_trials', 50),
                        save_results=hpo_config.get('save_results', True),
                        results_dir=hpo_config.get('results_dir', 'artifacts/hpo/tcn'),
                        verbose=True
                    )
                    
                    # Run optimization
                    result = hpo.optimize(
                        X_train=X_train.values if isinstance(X_train, pd.DataFrame) else X_train,
                        y_train=y_train.values if isinstance(y_train, pd.Series) else y_train,
                        X_val=X_val.values if isinstance(X_val, pd.DataFrame) else X_val,
                        y_val=y_val.values if isinstance(y_val, pd.Series) else y_val
                    )
                    
                    # Update config with optimized parameters
                    best_params = result.best_params
                    tcn_config = CausalTCNConfig(
                        # Use optimized structure params
                        num_filters=best_params['num_filters'],
                        num_layers=best_params['num_layers'],
                        kernel_size=best_params['kernel_size'],
                        dilation_base=best_params['dilation_base'],
                        
                        # Use optimized training params
                        learning_rate=best_params['tcn_learning_rate'],
                        dropout=best_params['tcn_dropout'],
                        batch_size=best_params['batch_size'],
                        epochs=best_params['tcn_epochs'],
                        early_stopping_patience=best_params['early_stopping_patience'],
                        
                        # Use optimized autoencoder params
                        use_autoencoder=True,
                        latent_dim=best_params['latent_dim'],
                        autoencoder_epochs=best_params['ae_epochs'],
                        train_autoencoder_if_missing=True,
                        autoencoder_path="models/analyst_autoencoder_encoder.pth" if self.config.role == TrainingRole.ANALYST else "models/tactician_autoencoder_encoder.pth"
                    )
                    
                    self.logger.info(f"✅ HPO complete! Best {hpo_config.get('metric', 'accuracy')}: {result.best_score:.4f}")
                    self.logger.info(f"   Optimized latent_dim: {best_params['latent_dim']}")
                    self.logger.info(f"   Optimized TCN layers: {best_params['num_layers']}")
                    self.logger.info(f"   Optimized TCN filters: {best_params['num_filters']}")
                    
                except Exception as hpo_error:
                    self.logger.error(f"❌ HPO failed: {hpo_error}, using default config")
                    import traceback
                    traceback.print_exc()
                    # Fall back to default config (already defined above)
            
            # Create and train TCN model
            model = CausalDilatedTCNModel(config=tcn_config)
            model.fit(X_train.values if isinstance(X_train, pd.DataFrame) else X_train, 
                     y_train.values if isinstance(y_train, pd.Series) else y_train)
            
            # Get predictions on full data
            predictions = model.predict(data.values if isinstance(data, pd.DataFrame) else data)
            
            # Calculate regression metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            metrics = {
                'mse': mean_squared_error(targets, predictions),
                'mae': mean_absolute_error(targets, predictions),
                'r2': r2_score(targets, predictions),
                'rmse': np.sqrt(mean_squared_error(targets, predictions))
            }
            
            self.logger.info(f"✅ TCN model trained successfully - R2: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
            
            return TrainingResult(
                success=True,
                model=model,
                metrics=metrics,
                feature_importance=None  # TCN doesn't provide feature importance directly
            )
            
        except ImportError as e:
            self.logger.error(f"TCN training failed - missing dependencies: {e}")
            return TrainingResult(success=False, error_message=f"Missing dependencies: {e}")
        except Exception as e:
            self.logger.error(f"TCN training failed: {e}")
            import traceback
            traceback.print_exc()
            return TrainingResult(success=False, error_message=str(e))
    
    async def _train_neural_network_model(self, model: Any, data: pd.DataFrame, targets: pd.Series) -> TrainingResult:
        """Train neural network model with role-specific architecture."""
        try:
            # Import PyTorch
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(data.values)
            y_tensor = torch.FloatTensor(targets.values)
            
            # Create dataset and dataloader
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Role-specific architecture
            if self.config.role == TrainingRole.ANALYST:
                model = nn.Sequential(
                    nn.Linear(data.shape[1], 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
                criterion = nn.BCELoss()
            else:  # Tactician
                model = nn.Sequential(
                    nn.Linear(data.shape[1], 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
                criterion = nn.MSELoss()
            
            # Optimizer
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop
            model.train()
            for epoch in range(100):
                total_loss = 0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if epoch % 20 == 0:
                    self.logger.info(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
            
            # Get predictions and metrics
            model.eval()
            with torch.no_grad():
                predictions = model(X_tensor).squeeze().numpy()
            
            if self.config.role == TrainingRole.ANALYST:
                # Binary classification metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                binary_predictions = (predictions > 0.5).astype(int)
                metrics = {
                    'accuracy': accuracy_score(targets, binary_predictions),
                    'precision': precision_score(targets, binary_predictions),
                    'recall': recall_score(targets, binary_predictions),
                    'f1_score': f1_score(targets, binary_predictions)
                }
            else:
                # Regression metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                metrics = {
                    'mse': mean_squared_error(targets, predictions),
                    'mae': mean_absolute_error(targets, predictions),
                    'r2': r2_score(targets, predictions),
                    'rmse': np.sqrt(mean_squared_error(targets, predictions))
                }
            
            return TrainingResult(
                success=True,
                model=model,
                metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"Neural network training failed: {e}")
            return TrainingResult(success=False, error_message=str(e))
    
    def _create_model(self, model_type: ModelType) -> Any:
        """Create model instance based on type."""
        try:
            if model_type == ModelType.LIGHTGBM:
                import lightgbm as lgb
                # Always use Regressor for trading models (predicting continuous values like directional_confidence)
                return lgb.LGBMRegressor()
            elif model_type == ModelType.DEPTHWISE_CNN:
                # Return None, will be created in training method
                return None
            elif model_type == ModelType.CATBOOST:
                from catboost import CatBoostRegressor
                # Always use Regressor for trading models (predicting continuous values)
                return CatBoostRegressor()
            elif model_type == ModelType.NEURAL_NETWORK:
                # Return None, will be created in training method
                return None
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except ImportError as e:
            self.logger.error(f"Failed to import required library: {e}")
            return None
    
    def _get_feature_importance(self, model: Any) -> Optional[Dict[str, float]]:
        """Extract feature importance from model."""
        try:
            if hasattr(model, 'feature_importance'):
                return dict(zip(self._get_feature_names(), model.feature_importance()))
            elif hasattr(model, 'get_feature_importance'):
                return dict(zip(self._get_feature_names(), model.get_feature_importance()))
            else:
                return None
        except Exception as e:
            self.logger.warning(f"Failed to extract feature importance: {e}")
            return None
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for importance extraction."""
        # This would be set during preprocessing
        return getattr(self, '_feature_names', [])
    
    def _is_better_model(self, current_metrics: Dict[str, float], best_metrics: Dict[str, float]) -> bool:
        """Check if current model is better than best model."""
        if not best_metrics:
            return True
        
        # Use primary metric for comparison
        primary_metric = 'f1_score' if self.config.role == TrainingRole.ANALYST else 'r2'
        
        if primary_metric in current_metrics and primary_metric in best_metrics:
            return current_metrics[primary_metric] > best_metrics[primary_metric]
        
        return False
    
    def _calculate_overall_metrics(self, training_results: Dict[str, TrainingResult]) -> Dict[str, float]:
        """Calculate overall metrics from individual model results."""
        if not training_results:
            return {}
        
        # Average metrics across all successful models
        all_metrics = {}
        for result in training_results.values():
            if result.success and result.metrics:
                for metric, value in result.metrics.items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)
        
        # Calculate averages
        overall_metrics = {}
        for metric, values in all_metrics.items():
            overall_metrics[f'avg_{metric}'] = np.mean(values)
            overall_metrics[f'std_{metric}'] = np.std(values)
        
        return overall_metrics
    
    @handles_errors(
        exceptions=(ValueError, RuntimeError),
        default_return=ValidationResult(success=False, error_message="Validation failed"),
        context="model validation"
    )
    async def validate(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> ValidationResult:
        """Validate trained models."""
        try:
            if not self._model_instances:
                return ValidationResult(success=False, error_message="No trained models available")
            
            # Use best model for validation
            best_model = max(self._model_instances.values(), key=lambda m: getattr(m, 'score', 0))
            
            # Get predictions
            predictions = best_model.predict(data)
            
            # Calculate validation metrics
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
            else:
                metrics = {}
            
            return ValidationResult(
                success=True,
                metrics=metrics,
                predictions=predictions
            )
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return ValidationResult(success=False, error_message=str(e))
    
    @handles_errors(
        exceptions=(ValueError, RuntimeError),
        default_return=PredictionResult(success=False, error_message="Prediction failed"),
        context="model prediction"
    )
    async def predict(self, data: pd.DataFrame) -> PredictionResult:
        """Make predictions with trained models."""
        try:
            if not self._model_instances:
                return PredictionResult(success=False, error_message="No trained models available")
            
            # Use best model for prediction
            best_model = max(self._model_instances.values(), key=lambda m: getattr(m, 'score', 0))
            
            # Get predictions
            predictions = best_model.predict(data)
            
            # Get probabilities if available
            probabilities = None
            if hasattr(best_model, 'predict_proba'):
                probabilities = best_model.predict_proba(data)
            
            return PredictionResult(
                success=True,
                predictions=predictions,
                probabilities=probabilities
            )
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return PredictionResult(success=False, error_message=str(e))
