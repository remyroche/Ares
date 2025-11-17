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
        
        # HPO guard to ensure we only run optimization once per training session
        self._hpo_completed = False
        
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
                
                if self.config.enable_hyperparameter_optimization and not self._hpo_completed:
                    tprint_info(f"🔧 Phase 2: Running hyperparameter optimization for {model_type.value}...")
                    hpo_start = time.time()
                    
                    model, best_params = await self._optimize_hyperparameters(
                        model, model_type, processed_data, processed_targets
                    )
                    
                    hpo_time = time.time() - hpo_start
                    hpo_n_trials = int(self.config.custom_params.get('hpo_n_trials', 20))
                    self._hpo_completed = True
                    
                    tprint_success(f"✅ HPO completed in {hpo_time:.2f}s with {hpo_n_trials} trials")
                    
                    # CRITICAL FIX: Train final model with best params and get test metrics
                    # HPO only does CV, we need train/val/test split evaluation
                    tprint_info(f"🎯 Phase 3: Training final model with optimized parameters and evaluating on test set...")
                    model_result = await self._train_single_model(
                        model, model_type, processed_data, processed_targets, best_params
                    )
                elif self.config.enable_hyperparameter_optimization and self._hpo_completed:
                    tprint_info("⏭️  Skipping HPO (already completed this session)")
                    tprint_info(f"🎯 Phase 3: Training model with previously optimized parameters...")
                    model_result = await self._train_single_model(
                        model, model_type, processed_data, processed_targets, getattr(self, '_best_params', {})
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

            # Calculate comprehensive additional metrics
            comprehensive_metadata = self._calculate_comprehensive_metadata(
                training_results=training_results,
                processed_data=processed_data,
                processed_targets=processed_targets,
                all_model_metrics=all_model_metrics
            )

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
                    'trained_models': trained_models,
                    'model_instances': self._model_instances,
                    'trained_feature_columns': list(model_result.metadata.get('engineered_data', processed_data).columns),
                    'n_features': len(processed_data.columns),
                    'n_samples': len(processed_data),
                    **comprehensive_metadata  # Unpack comprehensive metrics
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
                engineered_data = self._engineer_analyst_features(data, targets, allow_uniform_defaults=False)
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
        """Disable legacy per-model HPO in favour of external hierarchical PnL-first HPO."""
        try:
            self.logger.info(
                "Per-model R²-based HPO is disabled; using hierarchical PnL-first HPO pipeline instead."
            )
            return model, {}
        except Exception as e:
            self.logger.error(f"HPO stub failed: {e}")
            return model, {}
    
    def _engineer_analyst_features(self, data: pd.DataFrame, targets: pd.Series, **kwargs) -> pd.DataFrame:
        """Engineer features specific to Analyst role using shared module."""
        try:
            # Use shared feature engineer for consistency with inference
            engineered_data = self._analyst_feature_engineer.engineer_features(data, **kwargs)
            # Remove uniform default regime columns if they were auto-inserted (e.g., all 0.25)
            try:
                regime_cols = [c for c in engineered_data.columns if c.startswith('regime_confidence_')]
                for c in regime_cols:
                    col = engineered_data[c]
                    if col.nunique(dropna=False) == 1 and float(col.iloc[0]) == 0.25:
                        engineered_data = engineered_data.drop(columns=[c])
            except Exception:
                pass
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
            n_estimators = model_params.get('n_estimators', 800)
            learning_rate = model_params.get('learning_rate', 0.05)
            max_depth = model_params.get('max_depth', 6)
            num_leaves = model_params.get('num_leaves', 31)
            subsample = model_params.get('subsample', 0.7)  # bagging_fraction
            colsample_bytree = model_params.get('colsample_bytree', 0.7)  # feature_fraction
            # Stronger regularization defaults when HPO is off
            reg_alpha = model_params.get('reg_alpha', 0.5)   # lambda_l1
            reg_lambda = model_params.get('reg_lambda', 1.0) # lambda_l2
            min_child_samples = model_params.get('min_child_samples', 128)
            
            # Early stopping controls
            es_rounds = model_params.get('early_stopping_rounds', 50)
            min_boost_rounds = model_params.get('min_boost_rounds', 100)
            num_boost_round = max(n_estimators, min_boost_rounds)
            
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
                'force_col_wise': True,
            }

            # Derive monotonic constraints aligned with data columns
            try:
                cols = list(data.columns)
                constraints = []
                for c in cols:
                    lc = c.lower()
                    v = 0
                    # Regime confidences: bullish +1, bearish -1
                    if 'regime_confidence_bull' in lc or 'regime_bull' in lc:
                        v = 1
                    elif 'regime_confidence_bear' in lc or 'regime_bear' in lc:
                        v = -1
                    # Volatility/Range proxies: assume higher vol tends to increase absolute move; conservatively +1
                    elif 'price_range' in lc or 'true_range' in lc or 'atr' in lc or 'volatility' in lc:
                        v = 1
                    # Volume and liquidity: conservatively +1
                    elif 'volume' in lc or 'quote_volume' in lc or 'trades' in lc:
                        v = 1
                    # Momentum: price change indicators often positively related to future return at short horizons
                    elif 'mom' in lc or 'momentum' in lc or 'roc' in lc:
                        v = 1
                    # Bearish-type momentum flags
                    elif 'drawdown' in lc or 'down' in lc:
                        v = -1
                    constraints.append(v)
                if constraints and any(v != 0 for v in constraints):
                    params['monotone_constraints'] = constraints
                    tprint_info(f"   ✅ Applied monotone constraints for {sum(1 for v in constraints if v!=0)}/{len(constraints)} features")
                else:
                    tprint_info("   ℹ️ No monotone constraints inferred (all zero)")
            except Exception as _e:
                tprint_warning(f"   ⚠️ Failed to set monotone constraints: {_e}")
            
            tprint_info(f"Training LightGBM: depth={max_depth}, leaves={num_leaves}, lr={learning_rate}")
            tprint_info(f"📊 LightGBM training data: {data.shape}")
            tprint_info(f"   Feature columns (first 20): {list(data.columns[:20])}")
            tprint_info(f"   Feature columns (last 10): {list(data.columns[-10:])}")
            
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train with early stopping on validation set, enforcing a minimum number of rounds
            tprint_info(f"   Training up to {num_boost_round} iterations with early stopping ({es_rounds} rounds)")
            model = lgb.train(
                params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(es_rounds, verbose=False)]
            )
            
            # CRITICAL FIX: Evaluate on train/val/test splits separately
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            # Use best_iteration if available for predictions
            best_iter = model.best_iteration if getattr(model, 'best_iteration', None) else model.current_iteration()
            train_pred = model.predict(X_train, num_iteration=best_iter)
            val_pred = model.predict(X_val, num_iteration=best_iter)
            test_pred = model.predict(X_test, num_iteration=best_iter)
            
            def calculate_accuracy(y_true, y_pred, rel=0.1, min_abs=0.01):
                tol = np.maximum(min_abs, rel * np.std(y_true))
                return float(np.mean(np.abs(y_true - y_pred) <= tol))
            
            metrics = {
                'train_mse': mean_squared_error(y_train, train_pred),
                'train_mae': mean_absolute_error(y_train, train_pred),
                'train_r2': r2_score(y_train, train_pred),
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'val_mse': mean_squared_error(y_val, val_pred),
                'val_mae': mean_absolute_error(y_val, val_pred),
                'val_r2': r2_score(y_val, val_pred),
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                'test_mse': mean_squared_error(y_test, test_pred),
                'test_mae': mean_absolute_error(y_test, test_pred),
                'test_r2': r2_score(y_test, test_pred),
                'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'train_test_r2_gap': r2_score(y_train, train_pred) - r2_score(y_test, test_pred),
                'overfitting_ratio': (r2_score(y_train, train_pred) - r2_score(y_test, test_pred)) / max(r2_score(y_train, train_pred), 0.01),
                'generalization_score': r2_score(y_test, test_pred) / max(r2_score(y_train, train_pred), 0.01),
                'mse': mean_squared_error(y_test, test_pred),
                'mae': mean_absolute_error(y_test, test_pred),
                'r2': r2_score(y_test, test_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'iterations_used': best_iter,
                'best_iteration': best_iter
            }
            
            feature_importance = dict(zip(data.columns, model.feature_importance()))
            
            # Top features and basic leakage flags
            top_features = sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)[:20]
            tprint_info("✨ Top 20 features by importance:")
            for name, imp in top_features:
                tprint_info(f"   {name:<40} {imp:>8.1f}")
            leakage_flags = []
            leakage_markers = ['target', 'label', 'future', 'lead', 't+', 'shift(-', 'ahead', 'lookahead', 'leak']
            for name, _ in top_features:
                lname = name.lower()
                if any(m in lname for m in leakage_markers):
                    leakage_flags.append(name)
            if leakage_flags:
                tprint_warning(f"⚠️ Potential leakage features among top importance: {leakage_flags}")
            
            tprint_success(f"✅ LightGBM trained: {best_iter} iterations (best)")
            tprint_info(f"   📊 Train R²: {metrics['train_r2']:.4f}, RMSE: {metrics['train_rmse']:.4f}")
            tprint_info(f"   📊 Val R²: {metrics['val_r2']:.4f}, RMSE: {metrics['val_rmse']:.4f}")
            tprint_info(f"   📊 Test R²: {metrics['test_r2']:.4f}, RMSE: {metrics['test_rmse']:.4f}")
            tprint_info(f"   ⚠️  Train-Test Gap: {metrics['train_test_r2_gap']:.4f} ({metrics['overfitting_ratio']*100:.1f}%)")
            
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
                feature_importance=feature_importance,
                metadata={
                    'n_features': len(data.columns),
                    'n_samples': len(data),
                    'test_predictions': test_pred.tolist() if hasattr(test_pred, 'tolist') else list(test_pred),
                    'train_predictions': train_pred.tolist() if hasattr(train_pred, 'tolist') else list(train_pred),
                    'val_predictions': val_pred.tolist() if hasattr(val_pred, 'tolist') else list(val_pred)
                }
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
            iterations = model_params.get('iterations', 600)
            learning_rate = model_params.get('learning_rate', 0.05)
            depth = model_params.get('depth', 5)
            # Stronger regularization defaults when HPO is off
            l2_leaf_reg = model_params.get('l2_leaf_reg', 12.0)
            subsample = model_params.get('subsample', 0.7)  # CPU only
            colsample_bylevel = model_params.get('colsample_bylevel', 0.7)
            border_count = model_params.get('border_count', 128)
            max_ctr_complexity = model_params.get('max_ctr_complexity', 2)
            early_stopping_rounds = model_params.get('early_stopping_rounds', 50)
            random_seed = model_params.get('random_seed', 42)
            
            # Performance optimizations (NOT tuned by HPO)
            performance_params = {
                'thread_count': n_threads,
                'bootstrap_type': 'Bernoulli',
                'verbose': False,
                'random_seed': random_seed,
                'allow_writing_files': False,
            }
            
            performance_params['task_type'] = 'CPU'
            gpu_available = False
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
            
            if not gpu_available:
                hyperparameters['colsample_bylevel'] = colsample_bylevel
            
            all_params = {**performance_params, **hyperparameters}
            all_params['loss_function'] = 'RMSE'
            all_params['eval_metric'] = 'RMSE'
            
            model = CatBoostRegressor(**all_params)
            
            tprint_info(f"Training CatBoost: depth={depth}, iterations={iterations}, lr={learning_rate}")
            tprint_info(f"📊 CatBoost training data: {data.shape}")
            tprint_info(f"   Feature columns (first 20): {list(data.columns[:20])}")
            tprint_info(f"   Feature columns (last 10): {list(data.columns[-10:])}")
            
            # Enable early stopping and use best model
            fit_params = {
                'X': X_train,
                'y': y_train,
                'eval_set': (X_val, y_val),
                'use_best_model': True,
                'verbose': False,
                'early_stopping_rounds': early_stopping_rounds,
            }
            
            # Fit model with validation for early stopping
            model.fit(**fit_params)
            
            # Evaluate on splits using best iteration implicitly
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            
            def calculate_accuracy(y_true, y_pred, rel=0.1, min_abs=0.01):
                tol = np.maximum(min_abs, rel * np.std(y_true))
                return float(np.mean(np.abs(y_true - y_pred) <= tol))
            
            metrics = {
                'train_mse': mean_squared_error(y_train, train_pred),
                'train_mae': mean_absolute_error(y_train, train_pred),
                'train_r2': r2_score(y_train, train_pred),
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'train_accuracy': calculate_accuracy(y_train, train_pred),
                'val_mse': mean_squared_error(y_val, val_pred),
                'val_mae': mean_absolute_error(y_val, val_pred),
                'val_r2': r2_score(y_val, val_pred),
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                'val_accuracy': calculate_accuracy(y_val, val_pred),
                'test_mse': mean_squared_error(y_test, test_pred),
                'test_mae': mean_absolute_error(y_test, test_pred),
                'test_r2': r2_score(y_test, test_pred),
                'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'test_accuracy': calculate_accuracy(y_test, test_pred),
            }
            
            tprint_success("✅ CatBoost trained with early stopping")
            
            
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
            
            # Metrics for each split (regression-only: R^2/RMSE/MAE/MSE)
            metrics = {
                # Training set metrics
                'train_mse': mean_squared_error(y_train, train_pred),
                'train_mae': mean_absolute_error(y_train, train_pred),
                'train_r2': r2_score(y_train, train_pred),
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                
                # Validation set metrics
                'val_mse': mean_squared_error(y_val, val_pred),
                'val_mae': mean_absolute_error(y_val, val_pred),
                'val_r2': r2_score(y_val, val_pred),
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                
                # Test set metrics (CRITICAL - unseen data)
                'test_mse': mean_squared_error(y_test, test_pred),
                'test_mae': mean_absolute_error(y_test, test_pred),
                'test_r2': r2_score(y_test, test_pred),
                'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                
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
            
            # Log comprehensive results (regression metrics only)
            tprint_success(f"✅ CatBoost trained: {metrics['iterations_used']} iterations")
            tprint_info(f"   📊 Train R²: {metrics['train_r2']:.4f}, RMSE: {metrics['train_rmse']:.4f}")
            tprint_info(f"   📊 Val R²: {metrics['val_r2']:.4f}, RMSE: {metrics['val_rmse']:.4f}")
            tprint_info(f"   📊 Test R²: {metrics['test_r2']:.4f}, RMSE: {metrics['test_rmse']:.4f}")
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
                feature_importance=feature_importance,
                metadata={
                    'n_features': len(data.columns),
                    'n_samples': len(data),
                    'test_predictions': test_pred.tolist() if hasattr(test_pred, 'tolist') else list(test_pred),
                    'train_predictions': train_pred.tolist() if hasattr(train_pred, 'tolist') else list(train_pred),
                    'val_predictions': val_pred.tolist() if hasattr(val_pred, 'tolist') else list(val_pred)
                }
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
        """
        Calculate overall metrics from individual model results.

        CRITICAL FIX: This method now properly organizes metrics by split type
        (train/val/test) so they can be extracted by the report generator.
        """
        if not training_results:
            return {}

        # Collect all metrics from successful models
        all_metrics = {}
        all_feature_importance = {}

        for model_name, result in training_results.items():
            if result.success and result.metrics:
                for metric, value in result.metrics.items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)

            # Collect feature importance if available
            if result.success and result.feature_importance:
                for feature, importance in result.feature_importance.items():
                    if feature not in all_feature_importance:
                        all_feature_importance[feature] = []
                    all_feature_importance[feature].append(importance)

        # Calculate averages and organize by split type
        overall_metrics = {}

        for metric, values in all_metrics.items():
            avg_value = np.mean(values)
            std_value = np.std(values)

            # Add to overall with avg/std prefix (backward compatibility)
            overall_metrics[f'avg_{metric}'] = avg_value
            overall_metrics[f'std_{metric}'] = std_value

            # CRITICAL FIX: Also add split-specific metrics WITHOUT prefix
            # This allows _extract_comprehensive_metrics to find them
            if metric.startswith('train_'):
                # Keep both prefixed and non-prefixed for compatibility
                overall_metrics[metric] = avg_value
            elif metric.startswith('val_'):
                overall_metrics[metric] = avg_value
            elif metric.startswith('test_'):
                overall_metrics[metric] = avg_value
            else:
                # Non-split metrics (like overfitting_ratio, generalization_score, etc.)
                overall_metrics[metric] = avg_value

        # Add aggregated feature importance (top 20 features by average importance)
        if all_feature_importance:
            feature_importance_avg = {
                feature: np.mean(importances)
                for feature, importances in all_feature_importance.items()
            }
            # Store as dict for report extraction
            overall_metrics['feature_importance'] = feature_importance_avg

        # Add data quality metrics
        if training_results:
            # Get first successful result to extract data info
            first_result = next((r for r in training_results.values() if r.success), None)
            if first_result and first_result.metadata:
                overall_metrics['feature_count'] = first_result.metadata.get('n_features', 0)
                overall_metrics['sample_count'] = first_result.metadata.get('n_samples', 0)

        return overall_metrics

    def _calculate_comprehensive_metadata(
        self,
        training_results: Dict[str, Any],
        processed_data: pd.DataFrame,
        processed_targets: pd.Series,
        all_model_metrics: List[Any]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive metadata including data quality, model complexity,
        prediction statistics, and error analysis.

        Args:
            training_results: Dictionary of training results by model
            processed_data: Processed training data
            processed_targets: Processed target values
            all_model_metrics: List of model metrics objects

        Returns:
            Dictionary with comprehensive metadata for reporting
        """
        metadata = {}

        # ===== DATA QUALITY METRICS =====
        data_quality = {}
        try:
            # Missing values analysis
            missing_counts = processed_data.isnull().sum()
            total_values = len(processed_data) * len(processed_data.columns)
            data_quality['missing_values_count'] = int(missing_counts.sum())
            data_quality['missing_values_pct'] = float(missing_counts.sum() / total_values * 100)

            # Feature statistics
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                data_quality['numeric_features_count'] = len(numeric_cols)
                data_quality['mean_feature_variance'] = float(processed_data[numeric_cols].var().mean())
                data_quality['mean_feature_std'] = float(processed_data[numeric_cols].std().mean())

            # Target statistics
            data_quality['target_mean'] = float(processed_targets.mean())
            data_quality['target_std'] = float(processed_targets.std())
            data_quality['target_min'] = float(processed_targets.min())
            data_quality['target_max'] = float(processed_targets.max())
            data_quality['target_range'] = float(processed_targets.max() - processed_targets.min())

            metadata['data_quality'] = data_quality
        except Exception as e:
            self.logger.warning(f"Failed to calculate data quality metrics: {e}")
            metadata['data_quality'] = {}

        # ===== MODEL COMPLEXITY METRICS =====
        model_complexity = {}
        try:
            for model_name, result in training_results.items():
                if result.success and result.model:
                    model = result.model
                    complexity_info = {}

                    # LightGBM complexity
                    if hasattr(model, 'num_trees'):
                        complexity_info['num_trees'] = model.num_trees()
                        complexity_info['num_leaves'] = model.params.get('num_leaves', 'N/A')
                        complexity_info['max_depth'] = model.params.get('max_depth', 'N/A')

                    # CatBoost complexity
                    elif hasattr(model, 'tree_count_'):
                        complexity_info['num_trees'] = model.tree_count_
                        complexity_info['depth'] = getattr(model, 'get_param', lambda x: 'N/A')('depth')

                    # Store model-specific complexity
                    if complexity_info:
                        model_complexity[model_name] = complexity_info

            metadata['model_complexity'] = model_complexity
        except Exception as e:
            self.logger.warning(f"Failed to calculate model complexity metrics: {e}")
            metadata['model_complexity'] = {}

        # ===== PREDICTION STATISTICS =====
        prediction_stats = {}
        try:
            # Collect predictions from all models
            all_predictions = []
            for result in training_results.values():
                if result.success and result.metadata and 'test_predictions' in result.metadata:
                    all_predictions.extend(result.metadata['test_predictions'])

            if all_predictions:
                predictions_array = np.array(all_predictions)
                prediction_stats['prediction_mean'] = float(np.mean(predictions_array))
                prediction_stats['prediction_std'] = float(np.std(predictions_array))
                prediction_stats['prediction_min'] = float(np.min(predictions_array))
                prediction_stats['prediction_max'] = float(np.max(predictions_array))
                prediction_stats['prediction_median'] = float(np.median(predictions_array))

                # Skewness and kurtosis
                from scipy import stats
                prediction_stats['prediction_skewness'] = float(stats.skew(predictions_array))
                prediction_stats['prediction_kurtosis'] = float(stats.kurtosis(predictions_array))

            metadata['prediction_statistics'] = prediction_stats
        except Exception as e:
            self.logger.warning(f"Failed to calculate prediction statistics: {e}")
            metadata['prediction_statistics'] = {}

        # ===== ERROR ANALYSIS =====
        error_analysis = {}
        try:
            # Calculate directional accuracy (sign agreement)
            for model_name, result in training_results.items():
                if result.success and result.metrics:
                    # Directional accuracy from residuals
                    if 'test_mae' in result.metrics and 'test_rmse' in result.metrics:
                        # MAE/RMSE ratio indicates error distribution
                        mae = result.metrics['test_mae']
                        rmse = result.metrics['test_rmse']
                        if rmse > 0:
                            error_analysis[f'{model_name}_mae_rmse_ratio'] = float(mae / rmse)

            # Overall error statistics
            if any(key.endswith('_mae_rmse_ratio') for key in error_analysis):
                ratios = [v for k, v in error_analysis.items() if k.endswith('_mae_rmse_ratio')]
                error_analysis['avg_mae_rmse_ratio'] = float(np.mean(ratios))

            metadata['error_analysis'] = error_analysis
        except Exception as e:
            self.logger.warning(f"Failed to calculate error analysis: {e}")
            metadata['error_analysis'] = {}

        return metadata

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
