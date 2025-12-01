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

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]

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

            # Target and feature diagnostics on the exact training subset
            if processed_targets is not None:
                try:
                    if isinstance(processed_targets, pd.Series):
                        y = processed_targets.astype(float)
                    else:
                        y = pd.Series(processed_targets).astype(float)
                    if y.nunique(dropna=True) <= 1:
                        tprint_warning(
                            "⚠️ Target appears constant or nearly constant after preprocessing; "
                            "model may not learn useful structure."
                        )
                    else:
                        desc = y.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
                        tprint_info("📊 Target statistics on training subset:")
                        tprint_info(
                            f"   count={desc['count']:.0f}, mean={desc['mean']:.6f}, std={desc['std']:.6f}, "
                            f"min={desc['min']:.6f}, p1={desc['1%']:.6f}, p5={desc['5%']:.6f}, "
                            f"median={desc['50%']:.6f}, p95={desc['95%']:.6f}, p99={desc['99%']:.6f}, max={desc['max']:.6f}"
                        )

                        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            corrs = processed_data[numeric_cols].corrwith(y).dropna()
                            if not corrs.empty:
                                corrs_sorted = corrs.reindex(corrs.abs().sort_values(ascending=False).index)
                                tprint_info("📈 Top feature–target correlations (by |corr|):")
                                for name, val in corrs_sorted.head(20).items():
                                    tprint_info(f"   {name}: corr={val:.4f}")

                                strong = corrs_sorted[corrs_sorted.abs() >= 0.95]
                                if not strong.empty:
                                    tprint_warning(
                                        f"⚠️ Extremely high feature–target correlations detected (|corr|≥0.95); "
                                        f"potential leakage: {list(strong.index)}"
                                    )
                except Exception as diag_exc:
                    tprint_warning(f"⚠️ Failed to compute target/feature diagnostics: {diag_exc}")

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
            
            # Generate and save report + consolidated CSV of model metrics
            report_path = self._metrics_collector.save_report()
            metrics_csv_path = self._metrics_collector.save_metrics_csv()
            
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
                    'metrics_csv_path': str(metrics_csv_path),
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
                engineered_data = self._engineer_analyst_features(data, targets)
                tprint_info(f"   After ANALYST engineering: {engineered_data.shape}")
                tprint_info(f"   Engineered columns (first 20): {list(engineered_data.columns[:20])}")
                try:
                    specialist_keys = [
                        'liquidity_regime',
                        'alpha_score',
                        'risk_regime',
                        'path_regime',
                        'mean_reversion',
                        'mr_regime',
                        'smc_regime',
                        'breakout',
                        'support_scalar',
                        'resistance_scalar',
                    ]
                    specialist_cols = [
                        c for c in engineered_data.columns
                        if any(key in c.lower() for key in specialist_keys)
                    ]
                    if specialist_cols:
                        tprint_info(
                            "   ✅ Detected "
                            f"{len(specialist_cols)} specialist/regime feature columns "
                            f"(first 20): {specialist_cols[:20]}"
                        )
                    else:
                        tprint_warning(
                            "   ⚠️ No specialist/regime feature columns detected after ANALYST engineering"
                        )
                except Exception as diag_exc:
                    tprint_warning(
                        f"   ⚠️ Failed to inspect specialist/regime feature columns: {diag_exc}"
                    )
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
            # TEMPORARY: For ANALYST debug runs, skip non-LightGBM models so we can
            # focus on LightGBM behaviour and data/window diagnostics.
            if self.config.role == TrainingRole.ANALYST and model_type in (
                ModelType.KNN,
                ModelType.NGBOOST,
                ModelType.BAYESIANRIDGE,
            ):
                tprint_info(
                    f"⏭️ Skipping {model_type.value} training for ANALYST role "
                    f"(temporary LightGBM-only debug run)"
                )
                return TrainingResult(
                    success=False,
                    error_message=f"{model_type.value} skipped for LightGBM-only debug run",
                    training_time=time.time() - start_time,
                )

            if model_type == ModelType.LIGHTGBM:
                result = await self._train_lightgbm_model(model, engineered_data, aligned_targets)
            elif model_type == ModelType.EXTRATREES:
                result = await self._train_extratrees_model(model, engineered_data, aligned_targets)
            elif model_type == ModelType.DEPTHWISE_CNN:
                result = await self._train_depthwise_cnn_model(model, engineered_data, aligned_targets)
            elif model_type == ModelType.CATBOOST:
                result = await self._train_catboost_model(model, engineered_data, aligned_targets)
            elif model_type == ModelType.NGBOOST:
                result = await self._train_ngboost_model(model, engineered_data, aligned_targets)
            elif model_type == ModelType.KNN:
                result = await self._train_knn_model(model, engineered_data, aligned_targets)
            elif model_type == ModelType.BAYESIANRIDGE:
                result = await self._train_bayesianridge_model(model, engineered_data, aligned_targets)
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

    async def _train_knn_model(self, model: Any, data: pd.DataFrame, targets: pd.Series) -> TrainingResult:
        """Train KNeighborsRegressor model using YAML-driven hyperparameters."""
        try:
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            import yaml
            from pathlib import Path
            from src.utils.tprint import tprint_data_preview, tprint_info, tprint_warning, tprint_success

            tprint_info("=" * 80)
            tprint_info("🌟 MODEL-SPECIFIC TRAINING: KNN")
            tprint_info("=" * 80)
            tprint_data_preview(
                data,
                name="KNN Training Data",
                max_rows=5,
                max_cols=10,
                show_dtypes=True,
                show_shape=True
            )

            # For analyst role, optionally restrict KNN to specialist regime
            # outputs when those columns are present. This ensures KNN uses
            # only compact regime/specialist signals rather than the full
            # feature set. Log columns before/after for diagnostics.
            if getattr(self.config, 'role', None) == TrainingRole.ANALYST:
                try:
                    tprint_info(
                        f"   🔍 KNN pre-restriction columns ({len(data.columns)}): "
                        f"{list(data.columns)[:20]}{'...' if len(data.columns) > 20 else ''}"
                    )
                    specialist_prefixes = (
                        'risk_',
                        'alpha_',
                        'macro_',
                        'liquidity_regime_',
                        'breakout_',
                        'path_',
                    )
                    specialist_exact = {
                        'risk_regime',
                        'path_regime',
                        'resistance_scalar',
                        'support_scalar',
                        'breakout_scalar_resistance',
                        'breakout_scalar_support',
                        'breakout_long_edge_score',
                        'breakout_short_edge_score',
                        'is_resistance',
                        'is_support',
                        'predicted',
                    }
                    # Include generic prob_/confidence_ columns (e.g., SMC)
                    def _is_specialist(col: str) -> bool:
                        return (
                            any(col.startswith(p) for p in specialist_prefixes)
                            or col in specialist_exact
                            or col.startswith('prob_')
                            or col.startswith('confidence_')
                        )

                    specialist_cols = [c for c in data.columns if _is_specialist(c)]
                    if specialist_cols:
                        tprint_info(
                            f"   🔧 KNN feature restriction: using {len(specialist_cols)} specialist features "
                            f"out of {len(data.columns)} total"
                        )
                        tprint_info(
                            f"   🔍 KNN specialist columns ({len(specialist_cols)}): "
                            f"{specialist_cols[:20]}{'...' if len(specialist_cols) > 20 else ''}"
                        )
                        data = data[specialist_cols].copy()
                    else:
                        # For Analyst KNN, specialist regime features are
                        # mandatory. If none are present, abort this model
                        # rather than silently training on the full feature
                        # set, which leads to unstable and hard-to-interpret
                        # behaviour.
                        tprint_error(
                            "   ❌ KNN feature restriction: no specialist regime columns found; "
                            "aborting KNN training for Analyst role."
                        )
                        raise ValueError(
                            "KNN requires specialist regime features (risk_*/alpha_*/macro_*/liquidity_regime_*/"
                            "breakout_*/path_ or prob_/confidence_ columns); none were found in the training data."
                        )
                except Exception as fe:
                    tprint_warning(
                        f"   ⚠️ KNN feature restriction failed, using full feature set: {fe}"
                    )

            # Temporal 70/15/15 split (no shuffle) to respect ordering
            X_temp, X_test, y_temp, y_test = train_test_split(
                data, targets, test_size=0.15, random_state=42, shuffle=False
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.176, random_state=42, shuffle=False
            )

            tprint_info("   📊 Data splits (temporal order preserved):")
            tprint_info(f"      Train: {len(X_train)} samples ({len(X_train)/len(data)*100:.1f}%)")
            tprint_info(f"      Val: {len(X_val)} samples ({len(X_val)/len(data)*100:.1f}%)")
            tprint_info(f"      Test: {len(X_test)} samples ({len(X_test)/len(data)*100:.1f}%)")
            tprint_info("=" * 80)

            # Clean NaN targets for NGBoost: it requires finite y values.
            try:
                def _drop_nan_targets_ngb(X: pd.DataFrame, y: pd.Series, label: str):
                    mask = y.notna()
                    dropped = int((~mask).sum())
                    if dropped > 0:
                        tprint_warning(
                            f"   ⚠️ Dropping {dropped} {label} samples with NaN target for NGBoost"
                        )
                    return X[mask], y[mask]

                X_train, y_train = _drop_nan_targets_ngb(X_train, y_train, "train")
                X_val, y_val = _drop_nan_targets_ngb(X_val, y_val, "val")
                X_test, y_test = _drop_nan_targets_ngb(X_test, y_test, "test")

                if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
                    raise ValueError("No valid samples remain after NaN target filtering for NGBoost")
            except Exception as nan_exc:
                tprint_warning(
                    f"   ⚠️ Failed to clean NaN targets for NGBoost; training may fail: {nan_exc}"
                )

            # Clean NaN targets to satisfy NGBoost's requirement of finite y values
            try:
                def _drop_nan_targets(X: pd.DataFrame, y: pd.Series, label: str):
                    mask = y.notna()
                    dropped = int((~mask).sum())
                    if dropped > 0:
                        tprint_warning(
                            f"   ⚠️ Dropping {dropped} {label} samples with NaN target for NGBoost"
                        )
                    return X[mask], y[mask]

                X_train, y_train = _drop_nan_targets(X_train, y_train, "train")
                X_val, y_val = _drop_nan_targets(X_val, y_val, "val")
                X_test, y_test = _drop_nan_targets(X_test, y_test, "test")

                if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
                    raise ValueError("No valid samples remain after NaN target filtering for NGBoost")
            except Exception as nan_exc:
                tprint_warning(
                    f"   ⚠️ Failed to clean NaN targets for NGBoost; training may fail: {nan_exc}"
                )

            # Load KNN params from analyst_base_config.yaml if available
            model_params: Dict[str, Any] = {}
            config_path = Path('src/training/steps/model_training/analyst_base_config.yaml')
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        yaml_config = yaml.safe_load(f)
                    knn_cfg = (
                        yaml_config.get('analyst_config', {})
                        .get('base_models', {})
                        .get('knn', {})
                    ) or {}
                    base_params = knn_cfg.get('params', {}) or {}
                    hpo_cfg = knn_cfg.get('hpo', {}) or {}
                    optimal_params = {}
                    if isinstance(hpo_cfg, dict):
                        optimal_params = hpo_cfg.get('optimal_params', {}) or {}
                    model_params = {**base_params, **optimal_params}
                    if optimal_params:
                        tprint_success(
                            f"   ✅ Loaded {len(base_params)} base KNN params and "
                            f"{len(optimal_params)} optimal params from YAML config"
                        )
                    elif base_params:
                        tprint_success(f"   ✅ Loaded {len(base_params)} KNN params from YAML config")
                    else:
                        tprint_warning("   ⚠️ No KNN params found in YAML, using defaults")
                except Exception as yaml_exc:
                    tprint_warning(f"   ⚠️ Failed to load KNN params from YAML: {yaml_exc}")

            n_neighbors = int(model_params.get('n_neighbors', 50))
            algorithm = model_params.get('algorithm', 'auto')
            leaf_size = int(model_params.get('leaf_size', 30))
            weights = model_params.get('weights', 'distance')
            p = int(model_params.get('p', 2))

            tprint_info(
                f"Training KNN: n_neighbors={n_neighbors}, algorithm={algorithm}, leaf_size={leaf_size}, "
                f"weights={weights}, p={p}"
            )

            knn = KNeighborsRegressor(
                n_neighbors=n_neighbors,
                algorithm=algorithm,
                leaf_size=leaf_size,
                weights=weights,
                p=p,
                n_jobs=-1,
            )

            knn.fit(X_train, y_train)

            # Evaluate on splits (mirror LightGBM/NGBoost schema)
            train_pred = knn.predict(X_train)
            val_pred = knn.predict(X_val)
            test_pred = knn.predict(X_test)

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
            }

            train_r2 = metrics['train_r2']
            test_r2 = metrics['test_r2']
            metrics['train_test_r2_gap'] = train_r2 - test_r2
            denom = max(abs(train_r2), 1e-3)
            metrics['overfitting_ratio'] = metrics['train_test_r2_gap'] / denom
            metrics['generalization_score'] = test_r2 / denom

            tprint_success("✅ KNN trained")
            tprint_info(
                f"   📊 Train R²: {metrics['train_r2']:.4f}, RMSE: {metrics['train_rmse']:.4f}"
            )
            tprint_info(
                f"   📊 Val R²: {metrics['val_r2']:.4f}, RMSE: {metrics['val_rmse']:.4f}"
            )
            tprint_info(
                f"   📊 Test R²: {metrics['test_r2']:.4f}, RMSE: {metrics['test_rmse']:.4f}"
            )

            if metrics['overfitting_ratio'] > 0.2:
                tprint_warning("   ⚠️ HIGH OVERFITTING detected for KNN")
            elif metrics['overfitting_ratio'] > 0.1:
                tprint_warning("   ⚠️ Moderate overfitting detected for KNN")
            else:
                tprint_success("   ✅ Good generalization (overfitting ratio < 10%)")

            return TrainingResult(
                success=True,
                model=knn,
                metrics=metrics,
                feature_importance=None,
                metadata={
                    'n_features': len(data.columns),
                    'n_samples': len(data),
                    'test_predictions': test_pred.tolist() if hasattr(test_pred, 'tolist') else list(test_pred),
                    'train_predictions': train_pred.tolist() if hasattr(train_pred, 'tolist') else list(train_pred),
                    'val_predictions': val_pred.tolist() if hasattr(val_pred, 'tolist') else list(val_pred),
                },
            )

        except Exception as e:
            self.logger.error(f"KNN training failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return TrainingResult(success=False, error_message=str(e))

    async def _train_bayesianridge_model(self, model: Any, data: pd.DataFrame, targets: pd.Series) -> TrainingResult:
        """Train BayesianRidge model using YAML-driven hyperparameters."""
        try:
            from sklearn.linear_model import BayesianRidge
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            import yaml
            from pathlib import Path
            from src.utils.tprint import tprint_data_preview, tprint_info, tprint_warning, tprint_success

            tprint_info("=" * 80)
            tprint_info("🌟 MODEL-SPECIFIC TRAINING: BayesianRidge")
            tprint_info("=" * 80)
            tprint_data_preview(
                data,
                name="BayesianRidge Training Data",
                max_rows=5,
                max_cols=10,
                show_dtypes=True,
                show_shape=True
            )

            # Temporal 70/15/15 split (no shuffle)
            X_temp, X_test, y_temp, y_test = train_test_split(
                data, targets, test_size=0.15, random_state=42, shuffle=False
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.176, random_state=42, shuffle=False
            )

            tprint_info("   📊 Data splits (temporal order preserved):")
            tprint_info(f"      Train: {len(X_train)} samples ({len(X_train)/len(data)*100:.1f}%)")
            tprint_info(f"      Val: {len(X_val)} samples ({len(X_val)/len(data)*100:.1f}%)")
            tprint_info(f"      Test: {len(X_test)} samples ({len(X_test)/len(data)*100:.1f}%)")
            tprint_info("=" * 80)

            # Load BayesianRidge params from YAML if available
            model_params: Dict[str, Any] = {}
            config_path = Path('src/training/steps/model_training/analyst_base_config.yaml')
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        yaml_config = yaml.safe_load(f)
                    bayes_cfg = (
                        yaml_config.get('analyst_config', {})
                        .get('base_models', {})
                        .get('bayesianridge', {})
                    ) or {}
                    base_params = bayes_cfg.get('params', {}) or {}
                    hpo_cfg = bayes_cfg.get('hpo', {}) or {}
                    optimal_params = {}
                    if isinstance(hpo_cfg, dict):
                        optimal_params = hpo_cfg.get('optimal_params', {}) or {}
                    model_params = {**base_params, **optimal_params}
                    if optimal_params:
                        tprint_success(
                            f"   ✅ Loaded {len(base_params)} base BayesianRidge params and "
                            f"{len(optimal_params)} optimal params from YAML config"
                        )
                    elif base_params:
                        tprint_success(f"   ✅ Loaded {len(base_params)} BayesianRidge params from YAML config")
                    else:
                        tprint_warning("   ⚠️ No BayesianRidge params found in YAML, using defaults")
                except Exception as yaml_exc:
                    tprint_warning(f"   ⚠️ Failed to load BayesianRidge params from YAML: {yaml_exc}")

            # Map legacy 'n_iter' key to sklearn's 'max_iter' if needed
            if 'n_iter' in model_params and 'max_iter' not in model_params:
                try:
                    mapped_iter = int(model_params.get('n_iter', 0))
                    if mapped_iter > 0:
                        tprint_info(
                            f"   ℹ️ Mapping legacy BayesianRidge param 'n_iter'={mapped_iter} "
                            f"to 'max_iter'."
                        )
                        model_params['max_iter'] = mapped_iter
                except Exception:
                    # If mapping fails, leave handling to sklearn's own validation
                    pass
                finally:
                    # Remove legacy key to avoid confusion
                    model_params.pop('n_iter', None)

            # Drop parameters that are not supported by sklearn.linear_model.BayesianRidge
            try:
                supported_params = set(BayesianRidge().get_params().keys())
                invalid_keys = [k for k in model_params.keys() if k not in supported_params]
                for key in invalid_keys:
                    tprint_warning(
                        f"   ⚠️ Dropping unsupported BayesianRidge param '{key}' (value={model_params[key]})"
                    )
                    model_params.pop(key, None)
            except Exception as param_exc:
                tprint_warning(
                    f"   ⚠️ Failed to validate BayesianRidge params against estimator, using all params as-is: {param_exc}"
                )

            bayes = BayesianRidge(**model_params)

            tprint_info(f"Training BayesianRidge with params: {model_params}")

            bayes.fit(X_train, y_train)

            # Evaluate on splits
            train_pred = bayes.predict(X_train)
            val_pred = bayes.predict(X_val)
            test_pred = bayes.predict(X_test)

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
            }

            train_r2 = metrics['train_r2']
            test_r2 = metrics['test_r2']
            metrics['train_test_r2_gap'] = train_r2 - test_r2
            denom = max(abs(train_r2), 1e-3)
            metrics['overfitting_ratio'] = metrics['train_test_r2_gap'] / denom
            metrics['generalization_score'] = test_r2 / denom

            tprint_success("✅ BayesianRidge trained")
            tprint_info(
                f"   📊 Train R²: {metrics['train_r2']:.4f}, RMSE: {metrics['train_rmse']:.4f}"
            )
            tprint_info(
                f"   📊 Val R²: {metrics['val_r2']:.4f}, RMSE: {metrics['val_rmse']:.4f}"
            )
            tprint_info(
                f"   📊 Test R²: {metrics['test_r2']:.4f}, RMSE: {metrics['test_rmse']:.4f}"
            )

            if train_r2 < 0.02:
                # Extremely low explanatory power: treat as low-signal rather than classical overfitting
                tprint_info(
                    "   ℹ️ BayesianRidge is in a low-signal regime (train R² < 0.02); "
                    "overfitting assessment is not strongly informative."
                )
            elif metrics['overfitting_ratio'] > 0.2:
                tprint_warning("   ⚠️ HIGH OVERFITTING detected for BayesianRidge")
            elif metrics['overfitting_ratio'] > 0.1:
                tprint_warning("   ⚠️ Moderate overfitting detected for BayesianRidge")
            else:
                tprint_success("   ✅ Good generalization (overfitting ratio < 10%)")

            return TrainingResult(
                success=True,
                model=bayes,
                metrics=metrics,
                feature_importance=None,
                metadata={
                    'n_features': len(data.columns),
                    'n_samples': len(data),
                    'test_predictions': test_pred.tolist() if hasattr(test_pred, 'tolist') else list(test_pred),
                    'train_predictions': train_pred.tolist() if hasattr(train_pred, 'tolist') else list(train_pred),
                    'val_predictions': val_pred.tolist() if hasattr(val_pred, 'tolist') else list(val_pred),
                },
            )

        except Exception as e:
            self.logger.error(f"BayesianRidge training failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
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
            engineered_data = self._analyst_feature_engineer.engineer_features(
                data,
                allow_uniform_defaults=False,
                **kwargs,
            )
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
    
    async def _train_ngboost_model(self, model: Any, data: pd.DataFrame, targets: pd.Series) -> TrainingResult:
        """Train NGBoost model with YAML-driven hyperparameters."""
        try:
            from ngboost import NGBRegressor
            from ngboost.distns import Normal
            from ngboost.learners import default_tree_learner
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            import yaml
            from pathlib import Path
            from src.utils.tprint import tprint_data_preview, tprint_info, tprint_warning, tprint_success

            tprint_info("=" * 80)
            tprint_info("🌟 MODEL-SPECIFIC TRAINING: NGBoost")
            tprint_info("=" * 80)
            tprint_data_preview(
                data,
                name="NGBoost Training Data",
                max_rows=5,
                max_cols=10,
                show_dtypes=True,
                show_shape=True
            )
            tprint_data_preview(
                targets.to_frame() if isinstance(targets, pd.Series) else targets,
                name="NGBoost Target Data",
                max_rows=5,
                max_cols=10,
                show_dtypes=True,
                show_shape=True
            )

            # Temporal 70/15/15 split (no shuffle) to respect ordering
            X_temp, X_test, y_temp, y_test = train_test_split(
                data, targets, test_size=0.15, random_state=42, shuffle=False
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.176, random_state=42, shuffle=False
            )

            tprint_info("   📊 Data splits (temporal order preserved):")
            tprint_info(f"      Train: {len(X_train)} samples ({len(X_train)/len(data)*100:.1f}%)")
            tprint_info(f"      Val: {len(X_val)} samples ({len(X_val)/len(data)*100:.1f}%)")
            tprint_info(f"      Test: {len(X_test)} samples ({len(X_test)/len(data)*100:.1f}%)")
            tprint_info("=" * 80)

            # Load NGBoost params from YAML (analyst_base_config.yaml) if available
            model_params: Dict[str, Any] = {}
            config_path = Path('src/training/steps/model_training/analyst_base_config.yaml')
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        yaml_config = yaml.safe_load(f)
                    ngb_cfg = (
                        yaml_config.get('analyst_config', {})
                        .get('base_models', {})
                        .get('ngboost', {})
                    ) or {}
                    base_params = ngb_cfg.get('params', {}) or {}
                    hpo_cfg = ngb_cfg.get('hpo', {}) or {}
                    optimal_params = {}
                    if isinstance(hpo_cfg, dict):
                        optimal_params = hpo_cfg.get('optimal_params', {}) or {}
                    # Start from base params and overlay optimal params
                    model_params = dict(base_params)
                    if optimal_params:
                        # Direct top-level overrides
                        for key in ['n_estimators', 'learning_rate', 'minibatch_frac', 'verbose']:
                            if key in optimal_params:
                                model_params[key] = optimal_params[key]
                        # Map base learner-specific keys into nested params
                        if (
                            'base_learner_max_depth' in optimal_params
                            or 'base_learner_min_samples_leaf' in optimal_params
                        ):
                            base_learner_params = model_params.get('base_learner_params', {}) or {}
                            if 'base_learner_max_depth' in optimal_params:
                                base_learner_params['max_depth'] = optimal_params['base_learner_max_depth']
                            if 'base_learner_min_samples_leaf' in optimal_params:
                                base_learner_params['min_samples_leaf'] = optimal_params['base_learner_min_samples_leaf']
                            model_params['base_learner_params'] = base_learner_params
                        tprint_success(
                            f"   ✅ Loaded {len(base_params)} base NGBoost params and "
                            f"{len(optimal_params)} optimal params from YAML config"
                        )
                    elif base_params:
                        tprint_success(f"   ✅ Loaded {len(base_params)} NGBoost params from YAML config")
                    else:
                        tprint_warning("   ⚠️ No NGBoost params found in YAML, using defaults")
                except Exception as yaml_exc:
                    tprint_warning(f"   ⚠️ Failed to load NGBoost params from YAML: {yaml_exc}")

            n_estimators = int(model_params.get('n_estimators', 500))
            learning_rate = float(model_params.get('learning_rate', 0.01))
            minibatch_frac = float(model_params.get('minibatch_frac', 1.0))
            verbose = bool(model_params.get('verbose', False))

            base_learner_params = model_params.get('base_learner_params', {}) or {}
            base_max_depth = int(base_learner_params.get('max_depth', 4))
            base_min_samples_leaf = int(base_learner_params.get('min_samples_leaf', 20))

            # NGBoost expects `Base` to be an instantiated sklearn regressor
            # (see ngboost.api.NGBRegressor). The library's own default is the
            # `default_tree_learner` instance, which is a pre-configured
            # DecisionTreeRegressor. To customise its depth/leaf settings while
            # preserving sane defaults, clone its parameters, override the
            # relevant keys, and instantiate a fresh tree of the same class.
            try:
                default_params = default_tree_learner.get_params(deep=True)
            except Exception:
                default_params = {}

            tree_params = dict(default_params)
            tree_params['max_depth'] = base_max_depth
            tree_params['min_samples_leaf'] = base_min_samples_leaf

            base_learner = default_tree_learner.__class__(**tree_params)

            ngb = NGBRegressor(
                Dist=Normal,
                Base=base_learner,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                minibatch_frac=minibatch_frac,
                verbose=verbose,
                random_state=42,
            )

            tprint_info(
                f"Training NGBoost: n_estimators={n_estimators}, lr={learning_rate}, "
                f"minibatch_frac={minibatch_frac}, max_depth={base_max_depth}, "
                f"min_samples_leaf={base_min_samples_leaf}"
            )

            ngb.fit(X_train, y_train)

            # Evaluate on splits
            train_pred = ngb.predict(X_train)
            val_pred = ngb.predict(X_val)
            test_pred = ngb.predict(X_test)

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
            }

            metrics['train_test_r2_gap'] = metrics['train_r2'] - metrics['test_r2']
            metrics['overfitting_ratio'] = (
                metrics['train_test_r2_gap'] / max(metrics['train_r2'], 0.01)
            )
            metrics['generalization_score'] = (
                metrics['test_r2'] / max(metrics['train_r2'], 0.01)
            )

            tprint_success("✅ NGBoost trained")
            tprint_info(
                f"   📊 Train R²: {metrics['train_r2']:.4f}, RMSE: {metrics['train_rmse']:.4f}"
            )
            tprint_info(
                f"   📊 Val R²: {metrics['val_r2']:.4f}, RMSE: {metrics['val_rmse']:.4f}"
            )
            tprint_info(
                f"   📊 Test R²: {metrics['test_r2']:.4f}, RMSE: {metrics['test_rmse']:.4f}"
            )
            if metrics['overfitting_ratio'] > 0.2:
                tprint_warning("   ⚠️ HIGH OVERFITTING detected for NGBoost")
            elif metrics['overfitting_ratio'] > 0.1:
                tprint_warning("   ⚠️ Moderate overfitting detected for NGBoost")
            else:
                tprint_success("   ✅ Good generalization (overfitting ratio < 10%)")

            return TrainingResult(
                success=True,
                model=ngb,
                metrics=metrics,
                feature_importance=None,
                metadata={
                    'n_features': len(data.columns),
                    'n_samples': len(data),
                    'test_predictions': test_pred.tolist() if hasattr(test_pred, 'tolist') else list(test_pred),
                    'train_predictions': train_pred.tolist() if hasattr(train_pred, 'tolist') else list(train_pred),
                    'val_predictions': val_pred.tolist() if hasattr(val_pred, 'tolist') else list(val_pred),
                },
            )

        except ImportError as e:
            self.logger.error(f"NGBoost training skipped (library not available): {e}")
            return TrainingResult(success=False, error_message=str(e))
        except Exception as e:
            self.logger.error(f"NGBoost training failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return TrainingResult(success=False, error_message=str(e))

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

            # Dataset diagnostics before temporal split
            try:
                x_len = len(data)
                y_len = len(targets)
                tprint_info("📊 LightGBM dataset before split:")
                tprint_info(f"   X.shape={getattr(data, 'shape', None)}, y.shape={getattr(targets, 'shape', (y_len,))}")

                # Index diagnostics for time-aware debugging
                if hasattr(data, "index"):
                    try:
                        x_index = data.index
                        tprint_info(f"   X index type: {type(x_index).__name__}")
                        if len(x_index) > 0:
                            tprint_info(
                                f"   X index range: {x_index.min()} → {x_index.max()}"
                            )
                    except Exception as idx_exc:
                        tprint_warning(
                            f"   ⚠️ Failed to log X index diagnostics: {idx_exc}"
                        )
                if hasattr(targets, "index"):
                    try:
                        y_index = targets.index
                        tprint_info(f"   y index type: {type(y_index).__name__}")
                        if len(y_index) > 0:
                            tprint_info(
                                f"   y index range: {y_index.min()} → {y_index.max()}"
                            )
                    except Exception as tidx_exc:
                        tprint_warning(
                            f"   ⚠️ Failed to log y index diagnostics: {tidx_exc}"
                        )

                # Heuristic feature block composition (base vs specialist-like)
                try:
                    cols = list(data.columns)
                    specialist_prefixes = (
                        "liquidity_",
                        "alpha_",
                        "macro_alpha_",
                        "risk_",
                        "smc_",
                        "breakout_",
                        "path_",
                        "regime_",
                        "resistance_",
                        "support_",
                    )
                    specialist_cols = [
                        c for c in cols
                        if any(p in c.lower() for p in specialist_prefixes)
                    ]
                    base_cols = [c for c in cols if c not in specialist_cols]
                    tprint_info(
                        "📊 LightGBM feature composition (heuristic): "
                        f"total={len(cols)}, base≈{len(base_cols)}, specialist≈{len(specialist_cols)}"
                    )
                    if specialist_cols:
                        tprint_info(
                            f"   Example specialist-like cols: {specialist_cols[:10]}"
                        )
                except Exception as feat_exc:
                    self.logger.debug(
                        f"Failed to log LightGBM feature composition: {feat_exc}"
                    )
            except Exception as ds_exc:
                self.logger.debug(
                    f"Failed to log LightGBM dataset diagnostics before split: {ds_exc}"
                )

            constant_target = False
            try:
                if isinstance(targets, pd.Series):
                    y_diag = targets.astype(float)
                else:
                    y_diag = pd.Series(targets).astype(float)

                unique_vals = pd.unique(y_diag.dropna())
                y_mean = float(y_diag.mean())
                y_std = float(y_diag.std())
                y_min = float(y_diag.min())
                y_max = float(y_diag.max())

                tprint_info(
                    f"📊 LightGBM target diagnostics: "
                    f"len={len(y_diag)}, n_unique={len(unique_vals)}, "
                    f"mean={y_mean:.6f}, std={y_std:.6f}, "
                    f"min={y_min:.6f}, max={y_max:.6f}"
                )

                target_range = float(y_max - y_min)
                target_std = float(y_std)
                constant_target = len(unique_vals) <= 1 or (abs(target_range) == 0.0 and target_std == 0.0)
                if constant_target:
                    tprint_warning(
                        "⚠️ LightGBM target is effectively constant on the full training subset; "
                        "model will be forced to a trivial solution. "
                        "Check labeling thresholds, temporal splits, or light-mode filtering."
                    )
            except Exception as diag_exc:
                tprint_warning(f"⚠️ Failed to compute LightGBM target diagnostics: {diag_exc}")

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
                    lgbm_config = (
                        yaml_config.get('analyst_config', {})
                        .get('base_models', {})
                        .get('lgbm', {})
                    ) or {}
                    base_params = lgbm_config.get('params', {}) or {}
                    hpo_cfg = lgbm_config.get('hpo', {}) or {}
                    optimal_params = {}
                    if isinstance(hpo_cfg, dict):
                        optimal_params = hpo_cfg.get('optimal_params', {}) or {}
                    model_params = {**base_params, **optimal_params}
                    if optimal_params:
                        tprint_success(
                            f"   ✅ Loaded {len(base_params)} base LightGBM params and "
                            f"{len(optimal_params)} optimal params from YAML config"
                        )
                    elif base_params:
                        tprint_success(f"   ✅ Loaded {len(base_params)} LightGBM params from YAML config")
                    else:
                        tprint_warning("   ⚠️ No LightGBM params found in YAML, using defaults")
                else:
                    tprint_warning(f"   ⚠️  YAML config not found, using defaults")
                    model_params = {}
            
            # Extract hyperparameters (from HPO, YAML, or defaults)
            # ENHANCED: Increased from conservative defaults to reduce underfitting
            n_estimators = model_params.get('n_estimators', 1000)  # Increased from 800
            learning_rate = model_params.get('learning_rate', 0.05)
            max_depth = model_params.get('max_depth', 12)  # Increased from 6 for deeper trees
            num_leaves = model_params.get('num_leaves', 128)  # Increased from 31 for more complexity
            subsample = model_params.get('subsample', 0.7)  # bagging_fraction
            colsample_bytree = model_params.get('colsample_bytree', 0.7)  # feature_fraction
            # Stronger regularization defaults when HPO is off
            reg_alpha = model_params.get('reg_alpha', 0.5)   # lambda_l1
            reg_lambda = model_params.get('reg_lambda', 1.0) # lambda_l2
            min_child_samples = model_params.get('min_child_samples', 10)  # Reduced from 128 to capture finer patterns
            
            # Early stopping controls with safety floors to avoid degenerate 1-iteration models
            es_rounds = model_params.get('early_stopping_rounds', 50)
            min_boost_rounds = model_params.get('min_boost_rounds', 100)
            # Enforce sensible lower bounds regardless of YAML/HPO values
            if n_estimators < 100:
                tprint_warning(
                    f"   n_estimators={n_estimators} too low, raising to 100 to avoid degenerate trees"
                )
                n_estimators = 100
            if min_boost_rounds < 100:
                tprint_warning(
                    f"   min_boost_rounds={min_boost_rounds} too low, raising to 100"
                )
                min_boost_rounds = 100
            if es_rounds < 10:
                tprint_warning(
                    f"   early_stopping_rounds={es_rounds} too low, raising to 10"
                )
                es_rounds = 10
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
            # Default to GOSS-style sampling unless explicitly overridden in model_params
            boosting_type_cfg = str(model_params.get('boosting_type', 'goss')).lower()
            top_rate = model_params.get('top_rate')
            other_rate = model_params.get('other_rate')
            data_sample_strategy_cfg = str(model_params.get('data_sample_strategy', '')).lower()

            use_goss = (boosting_type_cfg == 'goss') or (data_sample_strategy_cfg == 'goss')

            params = {
                # Task configuration
                'objective': 'regression',
                'metric': 'rmse',
                # Use gbdt booster and control sampling via data_sample_strategy to
                # avoid LightGBM's deprecation warnings about boosting='goss'.
                'boosting_type': 'gbdt',
                
                # Hyperparameters (from YAML/HPO)
                'num_leaves': num_leaves,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'colsample_bytree': colsample_bytree,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'min_child_samples': min_child_samples,
                
                # Performance optimizations (NOT tuned by HPO)
                'n_jobs': n_threads,
                'verbose': -1,
                'force_col_wise': True,
            }

            # Configure sampling/bagging depending on whether we want GOSS semantics.
            if use_goss:
                # GOSS is incompatible with standard bagging, so disable bagging
                # and rely on top_rate/other_rate via data_sample_strategy='goss'.
                params['data_sample_strategy'] = 'goss'
                params['subsample'] = 1.0
                params['bagging_freq'] = 0

                # If using GOSS and GOSS-specific rates weren't set in model_params,
                # apply robust defaults.
                if top_rate is None:
                    top_rate = 0.2
                if other_rate is None:
                    other_rate = 0.1
                params['top_rate'] = top_rate
                params['other_rate'] = other_rate
                tprint_info(
                    f"   🔧 LightGBM GOSS enabled: boosting_type='gbdt', data_sample_strategy='goss', "
                    f"top_rate={top_rate}, other_rate={other_rate}, bagging_freq=0, subsample=1.0"
                )
            else:
                # For non-GOSS strategies, keep standard bagging configuration.
                params['subsample'] = subsample
                params['bagging_freq'] = 5

                # If the caller explicitly requested a non-GOSS data_sample_strategy,
                # propagate it through.
                if data_sample_strategy_cfg and data_sample_strategy_cfg != 'goss':
                    params['data_sample_strategy'] = data_sample_strategy_cfg

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
            
            train_r2_value = r2_score(y_train, train_pred)
            test_r2_value = r2_score(y_test, test_pred)
            train_test_gap_value = train_r2_value - test_r2_value
            denom = max(abs(train_r2_value), 1e-3)
            overfitting_ratio_value = train_test_gap_value / denom
            generalization_score_value = test_r2_value / denom

            if constant_target:
                # Force metrics to zero for constant-target regimes
                train_r2_value = 0.0
                test_r2_value = 0.0
                train_test_gap_value = 0.0
                overfitting_ratio_value = 0.0
                generalization_score_value = 0.0

            metrics = {
                'train_mse': mean_squared_error(y_train, train_pred),
                'train_mae': mean_absolute_error(y_train, train_pred),
                'train_r2': train_r2_value,
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'val_mse': mean_squared_error(y_val, val_pred),
                'val_mae': mean_absolute_error(y_val, val_pred),
                'val_r2': r2_score(y_val, val_pred),
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                'test_mse': mean_squared_error(y_test, test_pred),
                'test_mae': mean_absolute_error(y_test, test_pred),
                'test_r2': test_r2_value,
                'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'train_test_r2_gap': train_test_gap_value,
                'overfitting_ratio': overfitting_ratio_value,
                'generalization_score': generalization_score_value,
                'mse': mean_squared_error(y_test, test_pred),
                'mae': mean_absolute_error(y_test, test_pred),
                'r2': test_r2_value,
                'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'iterations_used': best_iter,
                'best_iteration': best_iter
            }

            if constant_target:
                metrics['constant_target'] = True
            
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

            low_signal = constant_target or abs(metrics['train_r2']) < 0.02

            if best_iter is not None and best_iter <= 5:
                if low_signal:
                    tprint_info(
                        f"   ℹ️ LightGBM stopped after {best_iter} iterations in a low-signal regime; "
                        f"this often indicates the target contains little usable structure."
                    )
                else:
                    tprint_warning(
                        f"   ⚠️ LightGBM used only {best_iter} boosting iterations – model may be degenerate "
                        f"(check target variance and training subset size)."
                    )

            if low_signal:
                tprint_info(
                    "   ℹ️ LightGBM is in a low-signal regime (train R² < 0.02 or constant target); "
                    "overfitting assessment is not strongly informative."
                )
            elif metrics['overfitting_ratio'] > 0.2:
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

    async def _train_extratrees_model(self, model: Any, data: pd.DataFrame, targets: pd.Series) -> TrainingResult:
        """Train ExtraTreesRegressor model with role-specific parameters from YAML config."""
        try:
            from sklearn.ensemble import ExtraTreesRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            import os
            from src.utils.tprint import tprint_data_preview, tprint_info

            # Log model-specific training start
            tprint_info("=" * 80)
            tprint_info("🌟 MODEL-SPECIFIC TRAINING: ExtraTrees")
            tprint_info("=" * 80)
            tprint_data_preview(
                data,
                name="ExtraTrees Training Data",
                max_rows=5,
                max_cols=10,
                show_dtypes=True,
                show_shape=True
            )

            # Get CPU optimizer for threading
            cpu_optimizer = get_m1_cpu_optimizer()
            n_threads = cpu_optimizer.get_optimal_thread_count() if cpu_optimizer else (os.cpu_count() or 4)

            tprint_info(f"🔧 ExtraTrees Configuration:")
            tprint_info(f"   CPU Threads: {n_threads}")
            tprint_info(f"   Training samples: {len(data)}")
            tprint_info(f"   Features: {len(data.columns)}")

            # CRITICAL FIX: Split data into train/val/test (70/15/15) for proper evaluation
            X_temp, X_test, y_temp, y_test = train_test_split(
                data, targets, test_size=0.15, random_state=42, shuffle=False
            )
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
                        extratrees_config = yaml_config.get('analyst_config', {}).get('base_models', {}).get('extratrees', {})
                        model_params = extratrees_config.get('params', {})
                        if model_params:
                            tprint_success(f"   ✅ Loaded {len(model_params)} parameters from YAML config")
                        else:
                            tprint_warning(f"   ⚠️  No ExtraTrees config found in YAML, using defaults")
                else:
                    tprint_warning(f"   ⚠️  YAML config not found, using defaults")
                    model_params = {}

            # Extract hyperparameters (from HPO, YAML, or defaults)
            n_estimators = model_params.get('n_estimators', 200)
            max_depth = model_params.get('max_depth', 15)
            min_samples_split = model_params.get('min_samples_split', 10)
            min_samples_leaf = model_params.get('min_samples_leaf', 4)
            max_features = model_params.get('max_features', 'sqrt')  # CRITICAL: sqrt as specified
            bootstrap = model_params.get('bootstrap', True)  # CRITICAL: bootstrap=True as specified

            # Build parameters dictionary
            params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'max_features': max_features,
                'bootstrap': bootstrap,
                'random_state': 42,
                'n_jobs': n_threads,
                'verbose': 0
            }

            tprint_info(f"Training ExtraTrees: n_estimators={n_estimators}, max_depth={max_depth}, max_features={max_features}, bootstrap={bootstrap}")
            tprint_info(f"📊 ExtraTrees training data: {data.shape}")
            tprint_info(f"   Feature columns (first 20): {list(data.columns[:20])}")
            tprint_info(f"   Feature columns (last 10): {list(data.columns[-10:])}")

            # Create and train model
            model = ExtraTreesRegressor(**params)
            model.fit(X_train, y_train)

            # CRITICAL FIX: Evaluate on train/val/test splits separately
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)

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
                'n_estimators': n_estimators
            }

            feature_importance = dict(zip(data.columns, model.feature_importances_))

            # Top features and basic leakage flags
            top_features = sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)[:20]
            tprint_info("✨ Top 20 features by importance:")
            for name, imp in top_features:
                tprint_info(f"   {name:<40} {imp:>8.6f}")
            leakage_flags = []
            leakage_markers = ['target', 'label', 'future', 'lead', 't+', 'shift(-', 'ahead', 'lookahead', 'leak']
            for name, _ in top_features:
                lname = name.lower()
                if any(m in lname for m in leakage_markers):
                    leakage_flags.append(name)
            if leakage_flags:
                tprint_warning(f"⚠️ Potential leakage features among top importance: {leakage_flags}")

            tprint_success(f"✅ ExtraTrees trained with {n_estimators} estimators")
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
            self.logger.error(f"ExtraTrees training failed: {e}")
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
            
            # Create model (TabR-backed DepthwiseSeparableCNNRegressor)
            cnn_model = DepthwiseSeparableCNNRegressor(
                # Map legacy DepthwiseCNN-style config to TabR parameters
                k_neighbors=model_params.get('k_neighbors', 96),
                use_embeddings=model_params.get('use_embeddings', False),
                n_encoder_layers=model_params.get('n_encoder_layers', 0),
                n_predictor_layers=model_params.get('n_predictor_layers', 1),
                d_embedding=filters,
                learning_rate=learning_rate,
                weight_decay=model_params.get('weight_decay', 1e-6),
                batch_size=batch_size,
                max_epochs=epochs,
                early_stopping_patience=early_stopping_patience,
                lr_scheduler_patience=reduce_lr_patience,
                dropout=dropout,
                random_state=model_params.get('random_state', 42),
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
            if torch is None or DataLoader is None or TensorDataset is None:
                raise ImportError("PyTorch is not available but a neural network model was requested")

            # Resolve configuration-driven hyperparameters (with sensible defaults)
            nn_params = self.config.custom_params.get('neural_network', {})
            if isinstance(nn_params, dict) and 'params' in nn_params:
                nn_params = nn_params['params']

            batch_size = int(nn_params.get('batch_size', 64))
            epochs = int(nn_params.get('epochs', 100))
            learning_rate = float(nn_params.get('learning_rate', 0.001))

            # Fallback to global training config for patience / validation_split when not overridden
            early_stopping_patience = int(
                nn_params.get('early_stopping_patience', self.config.early_stopping_patience)
            )
            validation_split = float(
                nn_params.get('validation_split', self.config.validation_split)
            )
            reduce_lr_patience = int(
                nn_params.get('reduce_lr_patience', max(1, early_stopping_patience // 2))
            )

            # Convert to tensors
            X_tensor = torch.as_tensor(data.values, dtype=torch.float32)
            y_tensor = torch.as_tensor(targets.values, dtype=torch.float32).view(-1)

            # Create dataset and temporal train/validation split
            full_dataset = TensorDataset(X_tensor, y_tensor)
            n_samples = len(full_dataset)

            if n_samples <= 0:
                raise ValueError("Neural network training received empty dataset")

            val_size = int(n_samples * validation_split) if validation_split > 0 else 0
            val_size = max(0, min(n_samples - 1, val_size))  # Ensure at least 1 train sample
            train_size = n_samples - val_size

            if val_size > 0:
                # Use deterministic split for reproducibility; keep temporal ordering by slicing
                indices = list(range(n_samples))
                train_indices = indices[:train_size]
                val_indices = indices[train_size:]
                train_subset = TensorDataset(X_tensor[train_indices], y_tensor[train_indices])
                val_subset = TensorDataset(X_tensor[val_indices], y_tensor[val_indices])
            else:
                train_subset = full_dataset
                val_subset = None

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False) if val_subset is not None else None

            # Role-specific architecture
            input_dim = data.shape[1]
            if self.config.role == TrainingRole.ANALYST:
                model = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1),
                    nn.Sigmoid(),
                )
                criterion = nn.BCELoss()
            else:  # Tactician or other regression roles
                model = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                )
                criterion = nn.MSELoss()

            # Optimizer and LR scheduler
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=reduce_lr_patience,
                verbose=False,
            )

            # Training loop with early stopping based on validation loss
            model.train()
            best_state: Optional[Dict[str, Any]] = None
            best_val_loss = float("inf")
            epochs_no_improve = 0
            enable_early_stopping = bool(self.config.enable_early_stopping and val_loader is not None)

            for epoch in range(epochs):
                running_loss = 0.0
                n_train = 0

                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    batch_size_actual = batch_X.size(0)
                    running_loss += loss.item() * batch_size_actual
                    n_train += batch_size_actual

                train_loss = running_loss / max(1, n_train)

                # Validation pass if available
                val_loss = None
                if val_loader is not None:
                    model.eval()
                    val_running = 0.0
                    n_val = 0
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            outputs = model(batch_X).squeeze()
                            loss = criterion(outputs, batch_y)
                            batch_size_actual = batch_X.size(0)
                            val_running += loss.item() * batch_size_actual
                            n_val += batch_size_actual
                    val_loss = val_running / max(1, n_val)
                    scheduler.step(val_loss)
                    model.train()
                else:
                    scheduler.step(train_loss)

                if epoch % max(1, epochs // 10) == 0:
                    if val_loss is not None:
                        self.logger.info(
                            f"NN Epoch {epoch}/{epochs} - train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                        )
                    else:
                        self.logger.info(
                            f"NN Epoch {epoch}/{epochs} - train_loss={train_loss:.4f} (no validation set)"
                        )

                # Early stopping
                if enable_early_stopping and val_loss is not None:
                    if val_loss < best_val_loss - 1e-6:
                        best_val_loss = val_loss
                        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= early_stopping_patience:
                            self.logger.info(
                                f"Neural network early stopping triggered after {epoch + 1} epochs "
                                f"(patience={early_stopping_patience}, best_val_loss={best_val_loss:.4f})"
                            )
                            break

            # Restore best validation model if available
            if best_state is not None:
                model.load_state_dict(best_state)

            # Get predictions and metrics on full dataset
            model.eval()
            with torch.no_grad():
                predictions_tensor = model(X_tensor).squeeze()
            predictions = predictions_tensor.detach().cpu().numpy()

            y_true = targets.values if isinstance(targets, pd.Series) else np.asarray(targets)

            if self.config.role == TrainingRole.ANALYST:
                # Binary classification metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

                binary_predictions = (predictions > 0.5).astype(int)
                metrics = {
                    'accuracy': float(accuracy_score(y_true, binary_predictions)),
                    'precision': float(precision_score(y_true, binary_predictions, zero_division=0)),
                    'recall': float(recall_score(y_true, binary_predictions, zero_division=0)),
                    'f1_score': float(f1_score(y_true, binary_predictions, zero_division=0)),
                }
            else:
                # Regression metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

                mse = mean_squared_error(y_true, predictions)
                mae = mean_absolute_error(y_true, predictions)
                r2 = r2_score(y_true, predictions)
                rmse = float(np.sqrt(mse))
                metrics = {
                    'mse': float(mse),
                    'mae': float(mae),
                    'r2': float(r2),
                    'rmse': rmse,
                }

            return TrainingResult(
                success=True,
                model=model,
                metrics=metrics,
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
            elif model_type == ModelType.EXTRATREES:
                from sklearn.ensemble import ExtraTreesRegressor
                # ExtraTreesRegressor with bootstrap and sqrt max_features
                return ExtraTreesRegressor(
                    max_features='sqrt',
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type == ModelType.DEPTHWISE_CNN:
                # Return None, will be created in training method
                return None
            elif model_type == ModelType.CATBOOST:
                from catboost import CatBoostRegressor
                # Always use Regressor for trading models (predicting continuous values)
                return CatBoostRegressor()
            elif model_type == ModelType.NGBOOST:
                try:
                    from ngboost import NGBRegressor
                    from ngboost.distns import Normal
                    from ngboost.learners import default_tree_learner

                    # Pass the learner factory/class directly so NGBoost can
                    # construct fresh base learners internally.
                    return NGBRegressor(Dist=Normal, Base=default_tree_learner, random_state=42)
                except ImportError as e:
                    self.logger.error(f"Failed to import NGBoost: {e}")
                    return None
            elif model_type == ModelType.KNN:
                from sklearn.neighbors import KNeighborsRegressor
                # Use a reasonable default; YAML-driven params will be applied in the
                # dedicated training method.
                return KNeighborsRegressor()
            elif model_type == ModelType.BAYESIANRIDGE:
                from sklearn.linear_model import BayesianRidge
                return BayesianRidge()
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

                from scipy import stats
                skew_value = float(stats.skew(predictions_array))
                kurtosis_value = float(stats.kurtosis(predictions_array))
                if not np.isfinite(skew_value):
                    skew_value = 0.0
                if not np.isfinite(kurtosis_value):
                    kurtosis_value = 0.0
                prediction_stats['prediction_skewness'] = skew_value
                prediction_stats['prediction_kurtosis'] = kurtosis_value

            # Advanced prediction–target relationship metrics (IC, IR, hit ratios, H-test)
            try:
                # Reconstruct the common temporal 70/15/15 split used in training
                from sklearn.model_selection import train_test_split

                X_temp, X_test, y_temp, y_test = train_test_split(
                    processed_data,
                    processed_targets,
                    test_size=0.15,
                    random_state=42,
                    shuffle=False,
                )

                # Ensure float targets and preserve index for time-based IC metrics
                if isinstance(y_test, pd.Series):
                    y_test_series = y_test.astype(float)
                else:
                    y_test_series = pd.Series(y_test, index=X_test.index).astype(float)

                # Collect per-model test predictions that match the test split length
                per_model_preds = {}
                for model_name, result in training_results.items():
                    if not (result.success and result.metadata and 'test_predictions' in result.metadata):
                        continue
                    preds = np.array(result.metadata['test_predictions'], dtype=float)
                    if len(preds) != len(y_test_series):
                        continue
                    per_model_preds[model_name] = preds

                if per_model_preds:
                    # Build matrix [n_models, n_samples] and aggregated prediction for IC metrics
                    model_names = list(per_model_preds.keys())
                    pred_matrix = np.vstack([per_model_preds[m] for m in model_names])
                    avg_pred = pred_matrix.mean(axis=0)

                    # Spearman IC and IC-based metrics using existing utilities
                    try:
                        from src.utils.ml_common.optimization.ic_snr_objective import (
                            compute_spearman_ic,
                            compute_ic_metrics_purged,
                            ICSNRConfig,
                        )

                        # Overall IC on aggregated predictions
                        ic_value = compute_spearman_ic(
                            y_true=y_test_series.values.astype(float),
                            y_pred=avg_pred.astype(float),
                        )
                        prediction_stats['spearman_ic'] = float(ic_value)

                        # Purged IC metrics (provides SNR / information ratio and volatility)
                        y_true_ic = pd.Series(y_test_series.values.astype(float), index=y_test_series.index)
                        y_pred_ic = pd.Series(avg_pred.astype(float), index=y_test_series.index)
                        ic_config = ICSNRConfig()
                        ic_metrics = compute_ic_metrics_purged(
                            y_true=y_true_ic,
                            y_pred=y_pred_ic,
                            config=ic_config,
                            datetime_index=y_true_ic.index if isinstance(y_true_ic.index, pd.DatetimeIndex) else None,
                        )

                        prediction_stats['ic_mean'] = float(ic_metrics.mean_ic)
                        prediction_stats['ic_median'] = float(ic_metrics.median_ic)
                        prediction_stats['ic_std'] = float(ic_metrics.std_ic)
                        prediction_stats['ic_snr'] = float(ic_metrics.snr)
                        prediction_stats['ic_sharpe'] = float(ic_metrics.ic_sharpe)
                        prediction_stats['ic_volatility'] = float(ic_metrics.std_ic)
                        prediction_stats['ic_final_score'] = float(ic_metrics.final_score)
                        prediction_stats['ic_n_valid_folds'] = int(ic_metrics.n_valid_folds)

                        # Information Ratio of predictions (signal-to-noise of IC stream)
                        prediction_stats['information_ratio_predictions'] = float(ic_metrics.snr)
                        prediction_stats['rolling_ic_volatility'] = float(ic_metrics.std_ic)
                    except Exception as ic_exc:
                        self.logger.debug(f"IC metrics calculation failed: {ic_exc}")

                    # Fragmentation ratio of prediction signal (choppiness of sign)
                    try:
                        pred_signs = np.sign(avg_pred)
                        if len(pred_signs) > 1:
                            sign_changes = np.sum(pred_signs[1:] != pred_signs[:-1])
                            fragmentation_ratio = sign_changes / float(len(pred_signs) - 1)
                            prediction_stats['prediction_fragmentation_ratio'] = float(fragmentation_ratio)
                    except Exception as frag_exc:
                        self.logger.debug(f"Fragmentation ratio calculation failed: {frag_exc}")

                    # Prediction–return calibration via simple linear regression
                    try:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            avg_pred.astype(float),
                            y_test_series.values.astype(float),
                        )
                        prediction_stats['prediction_return_calibration_slope'] = float(slope)
                        prediction_stats['prediction_return_calibration_intercept'] = float(intercept)
                        prediction_stats['prediction_return_calibration_r'] = float(r_value)
                        prediction_stats['prediction_return_calibration_r2'] = float(r_value ** 2)
                        prediction_stats['prediction_return_calibration_pvalue'] = float(p_value)
                    except Exception as calib_exc:
                        self.logger.debug(f"Prediction–return calibration calculation failed: {calib_exc}")

                    # Hit ratio at top-K thresholds (fraction of positive returns in top-K predictions)
                    try:
                        y_true_arr = y_test_series.values.astype(float)
                        preds_arr = avg_pred.astype(float)
                        order = np.argsort(-preds_arr)
                        n_samples_test = len(preds_arr)
                        for k_frac in [0.01, 0.05, 0.10]:
                            k = max(1, int(n_samples_test * k_frac))
                            if k <= 0 or k > n_samples_test:
                                continue
                            idx = order[:k]
                            y_top = y_true_arr[idx]
                            hit_ratio = float(np.mean(y_top > 0)) if len(y_top) > 0 else 0.0
                            key = f"hit_ratio_top_{int(k_frac * 100)}pct"
                            prediction_stats[key] = hit_ratio
                    except Exception as hit_exc:
                        self.logger.debug(f"Top-K hit ratio calculation failed: {hit_exc}")

                    # H-test style residual serial-independence check via lag-1 autocorrelation
                    try:
                        residuals = y_test_series.values.astype(float) - avg_pred.astype(float)
                        if len(residuals) >= 2:
                            residual_series = pd.Series(residuals, index=y_test_series.index)
                            lag1_autocorr = residual_series.autocorr(lag=1)
                            prediction_stats['residual_lag1_autocorr'] = float(lag1_autocorr)

                            # Simple H-style statistic and normal-approximate p-value
                            h_stat = float(lag1_autocorr * np.sqrt(len(residual_series)))
                            try:
                                h_pvalue = float(2 * stats.norm.sf(abs(h_stat)))
                            except Exception:
                                h_pvalue = 1.0

                            prediction_stats['h_test_statistic'] = h_stat
                            prediction_stats['h_test_pvalue'] = h_pvalue

                            status = 'clean'
                            if abs(lag1_autocorr) > 0.20:
                                status = 'critical_dependence'
                            elif abs(lag1_autocorr) > 0.10:
                                status = 'moderate_dependence'
                            prediction_stats['residual_autocorr_status'] = status
                    except Exception as h_exc:
                        self.logger.debug(f"Residual H-test style check failed: {h_exc}")

            except Exception as adv_exc:
                self.logger.debug(f"Advanced prediction statistics calculation failed: {adv_exc}")

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

    def _calibrate_probabilities_isotonic(self,
                                         y_true: np.ndarray,
                                         y_proba: np.ndarray) -> Optional[Any]:
        """
        Fit isotonic regression calibrator on validation data.

        Maps model's raw output (0.0-1.0) to actual empirical probability of success.
        Addresses "Negative R²" problem where model probabilities are worse than
        guessing the average win rate.

        Args:
            y_true: True binary labels (0 or 1)
            y_proba: Raw probabilities from model (shape: n_samples or n_samples x 2)

        Returns:
            Fitted isotonic regression calibrator or None if fitting fails
        """
        try:
            from sklearn.isotonic import IsotonicRegression
            import warnings

            # Extract positive class probabilities if needed
            if y_proba.ndim == 2:
                y_proba_pos = y_proba[:, 1]
            else:
                y_proba_pos = y_proba

            # Remove NaN values
            valid_mask = ~(np.isnan(y_true) | np.isnan(y_proba_pos))
            if valid_mask.sum() < 10:
                self.logger.warning(f"Insufficient valid samples for calibration ({valid_mask.sum() < 10})")
                return None

            y_true_valid = y_true[valid_mask]
            y_proba_valid = y_proba_pos[valid_mask]

            # Fit isotonic regression
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                calibrator = IsotonicRegression(out_of_bounds='clip', n_subwindows=3)
                calibrator.fit(y_proba_valid, y_true_valid)

            self.logger.info(f"✅ Probability calibration fitted on {valid_mask.sum()} samples")
            return calibrator

        except Exception as e:
            self.logger.warning(f"Failed to fit isotonic calibrator: {e}")
            return None

    def calibrate_predictions(self,
                             probabilities: np.ndarray,
                             calibrator: Optional[Any] = None) -> np.ndarray:
        """
        Apply probability calibration to model predictions.

        Transforms model's raw probabilities to actual empirical probabilities
        using fitted calibrator (Isotonic Regression).

        Args:
            probabilities: Raw probabilities from model
            calibrator: Fitted isotonic regression calibrator

        Returns:
            Calibrated probabilities
        """
        if calibrator is None:
            return probabilities

        try:
            # Extract positive class probabilities if 2D
            if probabilities.ndim == 2:
                y_proba_pos = probabilities[:, 1]
                is_2d = True
            else:
                y_proba_pos = probabilities
                is_2d = False

            # Apply calibration
            calibrated = calibrator.predict(y_proba_pos)

            # Reconstruct 2D array if needed
            if is_2d:
                calibrated_proba = np.column_stack([1 - calibrated, calibrated])
                return calibrated_proba
            else:
                return calibrated

        except Exception as e:
            self.logger.warning(f"Calibration failed: {e}, returning original probabilities")
            return probabilities
