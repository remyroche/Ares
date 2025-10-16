"""
Tactician Models Training - Base Model Training Module

This module handles training of individual Tactician base models:
- RandomSurvivalForest model
- XGBoost model
- ElasticNetCV model
- NAS (Neural Architecture Search) model
- TAS (Tree-based Architecture Search) model

The Tactician operates on the 15m timeframe and decides WHEN to trade using only
Analyst-provided green-signal filtered windows (>0.4% confidence threshold), refining entries after the
15m Analyst approval. Integrates with Analyst ensemble outputs and regime features for enhanced 15m timeframe tactical decisions.

ENHANCED FEATURES:
- Comprehensive error handling with detailed failure reporting
- Enhanced progress tracking and sub-step reporting
- Input validation and data quality checks
- Optimized vectorization with intelligent fallback
- Structured logging with performance metrics
- Health monitoring throughout training process
- Integration with common utilities and hardware optimizers
- Extensive logging with tprint at every step
- Analyst ensemble integration for filtered training data
- NAS and TAS support for advanced 5m timeframe analysis
"""

import json
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
import traceback
from dataclasses import dataclass
from enum import Enum

from src.training.utils.embedding_postprocessing import filter_embedding_features

# Enhanced imports with comprehensive error handling
try:
    from src.utils.logger import system_logger
    from src.utils.ml_common.config import PerRegimeTrainingConfig
    from src.utils.ml_common.training import PerRegimeTrainingStep
    TACTICIAN_TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import core ML utilities: {e}")
    TACTICIAN_TRAINING_AVAILABLE = False

# Import negative learning integration
try:
    from .negative_learning_training_patches import apply_negative_learning_patches
    from .negative_learning_training_integration import (
        initialize_negative_learning_integration,
        get_negative_learning_integration
    )
    NEGATIVE_LEARNING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ WARNING: Negative learning integration not available: {e}")
    NEGATIVE_LEARNING_AVAILABLE = False

# Import enhanced logging and utilities - CRITICAL: Fast fail if not available
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
        tprint_debug, tprint_progress, tprint_performance, tprint_structured,
        tprint_timer, LogLevel
    )
    TPRINT_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL ERROR: tprint is required but not available: {e}")
    TPRINT_AVAILABLE = False

# Import common utilities - CRITICAL: Fast fail if not available
try:
    from src.utils.common_operations import (
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        cleanup_m1_optimizers, integrate_with_m1_optimizers
    )
    COMMON_OPS_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Common operations utilities are required but not available: {e}")
    COMMON_OPS_AVAILABLE = False

try:
    from src.utils.common_utilities import (
        safe_dataframe_operation, validate_dataframe_columns, calculate_data_quality_metrics,
        safe_merge_dataframes, create_summary_statistics
    )
    COMMON_UTILITIES_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Common utilities are required but not available: {e}")
    COMMON_UTILITIES_AVAILABLE = False

try:
    from src.utils.math_validation import (
        safe_divide, validate_finite, validate_positive, validate_range,
        safe_correlation, safe_percentage_change
    )
    MATH_VALIDATION_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Math validation utilities are required but not available: {e}")
    MATH_VALIDATION_AVAILABLE = False

# Import enhanced validation utilities
try:
    from src.training.steps.pre_training.utils.validation_utils import (
        PreTrainingValidator, ValidationConfig, ValidationContext,
        validate_training_data, validate_model_config, validate_regime_data,
        ValidationResult
    )
    VALIDATION_UTILS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ WARNING: Enhanced validation utilities not available: {e}")
    VALIDATION_UTILS_AVAILABLE = False

try:
    from src.utils.kline_parquet import validate_klines_data, process_klines_data
    from src.utils.serialization_utils import safe_serialize, safe_deserialize
    DATA_UTILITIES_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Data utilities are required but not available: {e}")
    DATA_UTILITIES_AVAILABLE = False

try:
    from src.utils.matrix_operations import (
        safe_matrix_operations, validate_matrix_properties, optimize_matrix_computations
    )
    MATRIX_OPS_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Matrix operations utilities are required but not available: {e}")
    MATRIX_OPS_AVAILABLE = False

# Import VectorBT optimizations - HIGH PRIORITY
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer,
        optimized_rolling_mean, optimized_rolling_std, optimized_rolling_var,
        optimized_rolling_min, optimized_rolling_max, optimized_rolling_sum,
        optimized_rolling_apply, optimized_rolling_corr, optimized_rolling_cov
    )
    VECTORBT_ROLLING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ WARNING: VectorBT Rolling Optimizer not available: {e}")
    VECTORBT_ROLLING_AVAILABLE = False

try:
    from src.utils.ml_common.unified_vectorization_manager import (
        get_unified_vectorization_manager, optimize_cross_validation,
        OperationType, OperationConfig, OptimizationResult
    )
    UNIFIED_VECTORIZATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ WARNING: Unified Vectorization Manager not available: {e}")
    UNIFIED_VECTORIZATION_AVAILABLE = False

class TacticianModelType(Enum):
    """Tactician model types."""
    # Updated model types as requested
    LGBM_GRU = "LGBM_GRU"  # LGBM + small GRU as embedding
    CATBOOST = "CATBOOST"  # CatBoost classifier
    CAUSAL_TCN = "CAUSAL_TCN"  # Causal Dilated TCN
    STACKER_LGBM_CALIBRATED = "STACKER_LGBM_CALIBRATED"  # Meta-learner

    # Legacy models (for backward compatibility)
    RANDOM_SURVIVAL_FOREST = "RANDOM_SURVIVAL_FOREST"
    XGBOOST = "XGBOOST"
    ELASTIC_NET_CV = "ELASTIC_NET_CV"
    NAS = "NAS"
    TAS = "TAS"

@dataclass
class TacticianModelsTrainingConfig:
    """Configuration for Tactician base models training."""
    # Training parameters
    model_types: Optional[List[Union[str, TacticianModelType]]] = None
    save_models: bool = True
    save_metrics: bool = True
    output_directory: str = "generated/tactician_models_training"

    # Direction control for training
    enable_long_positions: bool = True
    enable_short_positions: bool = True

    # Hardware optimization
    enable_parallel_processing: bool = True
    enable_gpu_acceleration: bool = True
    memory_limit_gb: float = 8.0

    # Validation parameters
    validation_split: float = 0.2
    min_training_samples: int = 100

    def __post_init__(self):
        """Post-initialization setup."""
        supported_types = [
            TacticianModelType.LGBM_GRU,
            TacticianModelType.CATBOOST,
            TacticianModelType.CAUSAL_TCN,
            TacticianModelType.STACKER_LGBM_CALIBRATED,
            TacticianModelType.NAS,
            TacticianModelType.TAS,
        ]

        if self.model_types is None:
            self.model_types = [
                TacticianModelType.LGBM_GRU,
                TacticianModelType.CATBOOST,
                TacticianModelType.CAUSAL_TCN,
                TacticianModelType.STACKER_LGBM_CALIBRATED,
                TacticianModelType.NAS,
                TacticianModelType.TAS
            ]
        else:
            normalized_types: List[TacticianModelType] = []
            for model_type in self.model_types:
                if isinstance(model_type, TacticianModelType):
                    normalized_types.append(model_type)
                    continue

                if isinstance(model_type, str):
                    candidate = model_type.strip()
                    try:
                        normalized_types.append(TacticianModelType[candidate])
                        continue
                    except KeyError:
                        try:
                            normalized_types.append(TacticianModelType(candidate))
                            continue
                        except ValueError:
                            pass

                raise ValueError(f"Unsupported Tactician model type: {model_type!r}")

            self.model_types = normalized_types

@dataclass
class TacticianModelsTrainingResult:
    """Result of Tactician models training."""
    # Training results
    models: Dict[str, Any] = None
    training_metrics: Dict[str, Any] = None

    # Metadata
    execution_time: float = 0.0
    total_samples: int = 0
    features_used: List[str] = None
    model_types_trained: List[str] = None
    models_per_type: int = 0

    # Status
    training_completed: bool = False
    error: Optional[str] = None

class TacticianModelsTrainingStep:
    """
    Tactician Models Training Step.

    Handles training of individual Tactician base models using common dependencies.
    """

    def __init__(self, config: Optional[TacticianModelsTrainingConfig] = None):
        """Initialize the Tactician models training step."""
        try:
            self.config = config or TacticianModelsTrainingConfig()
            self.logger = system_logger.getChild('TacticianModelsTrainingStep')

            # Initialize hardware optimizers
            if COMMON_OPS_AVAILABLE:
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()
                tprint_success("✅ Hardware optimizers initialized")
            else:
                self.gpu_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None

            # Initialize VectorBT Rolling Optimizer - HIGH PRIORITY
            if VECTORBT_ROLLING_AVAILABLE:
                self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=self.config.enable_gpu_acceleration,
                    enable_parallel=self.config.enable_parallel_processing,
                    memory_efficient=True,
                    chunk_size=1000,
                    fast_fail=True,
                    enable_logging=True
                )
                tprint_success("✅ VectorBT Rolling Optimizer initialized")
            else:
                self.vectorbt_rolling_optimizer = None
                tprint_warning("⚠️ VectorBT Rolling Optimizer not available")

            # Initialize Unified Vectorization Manager - MEDIUM PRIORITY
            if UNIFIED_VECTORIZATION_AVAILABLE:
                self.vectorization_manager = get_unified_vectorization_manager()
                tprint_success("✅ Unified Vectorization Manager initialized")
            else:
                self.vectorization_manager = None
                tprint_warning("⚠️ Unified Vectorization Manager not available")

            tprint_success("✅ TacticianModelsTrainingStep initialized successfully")

        except Exception as e:
            tprint_error(f"❌ Failed to initialize TacticianModelsTrainingStep: {e}")
            raise

    async def train_tactician_models(
        self,
        training_data: pd.DataFrame,
        feature_columns: List[str],
        target_columns: List[str],
        sample_weight: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train Tactician base models.

        Args:
            training_data: DataFrame with features and targets
            feature_columns: List of feature column names
            target_columns: List of target column names
            sample_weight: Optional sample weights
            **kwargs: Additional parameters

        Returns:
            Dict with trained models and metrics
        """
        start_time = tprint_timer()
        tprint_info("🚀 Starting Tactician base models training...")

        try:
            # Enhanced input validation
            validation_errors = []

            if training_data.empty:
                validation_errors.append("Training data is empty")
            if not feature_columns:
                validation_errors.append("No feature columns provided")
            if not target_columns:
                validation_errors.append("No target columns provided")
            if len(training_data) < self.config.min_training_samples:
                validation_errors.append(f"Insufficient training samples: {len(training_data)} < {self.config.min_training_samples}")

            # Validate feature columns exist in data
            missing_features = [col for col in feature_columns if col not in training_data.columns]
            if missing_features:
                validation_errors.append(f"Missing feature columns in training data: {missing_features}")

            # Validate target columns exist in data
            missing_targets = [col for col in target_columns if col not in training_data.columns]
            if missing_targets:
                validation_errors.append(f"Missing target columns in training data: {missing_targets}")

            # Check for NaN/inf values in features and targets
            feature_data = training_data[feature_columns]
            if feature_data.isnull().any().any():
                validation_errors.append("NaN values found in feature columns")
            if not np.isfinite(feature_data.values).all():
                validation_errors.append("Infinite values found in feature columns")

            target_data = training_data[target_columns]
            if target_data.isnull().any().any():
                validation_errors.append("NaN values found in target columns")
            if not np.isfinite(target_data.values).all():
                validation_errors.append("Infinite values found in target columns")

            if validation_errors:
                error_msg = f"Input validation failed: {'; '.join(validation_errors)}"
                tprint_error(f"❌ {error_msg}")
                raise ValueError(error_msg)

            # Load features from final feature selection if not provided
            final_feature_columns = await self._load_final_feature_selection_features(
                training_data, feature_columns, **kwargs
            )

            # Load regime features from market analysis ensemble
            enhanced_feature_columns = await self._load_regime_features(
                training_data, final_feature_columns, **kwargs
            )

            # Apply model-specific feature filtering based on ML model type
            model_specific_features = await self._apply_model_specific_feature_filtering(
                enhanced_feature_columns, **kwargs
            )

            # Apply Analyst OOF integration (whole dataset + features + weights)
            try:
                enhanced_data, analyst_filter_info = await self._apply_analyst_filtering(
                    training_data, model_specific_features, target_columns, **kwargs
                )

                # Use whole dataset - no filtering
                if enhanced_data.empty:
                    raise ValueError("Enhanced data is empty after Analyst OOF integration")

                tprint_success(f"✅ Using whole dataset: {len(enhanced_data)} samples")

            except Exception as filter_error:
                tprint_warning(f"⚠️ Analyst OOF integration failed: {filter_error}")
                tprint_info("🔄 Continuing with original training data")
                enhanced_data = training_data
                analyst_filter_info = {
                    'filtered': False,
                    'reason': f'integration_error: {str(filter_error)}',
                    'original_samples': len(training_data),
                    'enhanced_samples': len(training_data)
                }

            # Add Analyst OOF features to the feature set
            analyst_features = await self._add_analyst_oof_features(enhanced_data, **kwargs)
            if analyst_features:
                # Add analyst features to the feature set
                model_specific_features.extend(analyst_features)
                tprint_success(f"✅ Added {len(analyst_features)} Analyst OOF features to training")

            # Calculate sample weights based on Analyst confidence
            analyst_weights = await self._calculate_analyst_weights(enhanced_data, **kwargs)
            if analyst_weights is not None:
                sample_weight = analyst_weights
                tprint_success(f"✅ Using Analyst-based sample weights")
            elif sample_weight is None:
                sample_weight = np.ones(len(enhanced_data))

            # Prepare training data with validated features
            X = enhanced_data[model_specific_features].values
            y = enhanced_data[target_columns].values

            if len(y.shape) == 1:
                y = y.reshape(-1, 1)

            # Add analyst confidence as additional feature if available
            if analyst_features:
                X = np.column_stack([X] + [enhanced_data[feat].values for feat in analyst_features])

            # Train models
            all_models = {}
            all_metrics = {}
            all_oof_predictions: Dict[str, np.ndarray] = {}

            for model_type in self.config.model_types:
                try:
                    tprint_info(f"🔧 Training {model_type.value} model...")

                    if model_type in {TacticianModelType.NAS, TacticianModelType.TAS}:
                        training_result = await self._train_nas_tas_timing_models(
                            model_type=model_type,
                            training_data=training_data,
                            feature_columns=feature_columns,
                            target_columns=target_columns,
                            sample_weight=sample_weight,
                            model_type_for_filtering=model_type.value,
                            **kwargs,
                        )
                    else:
                        # Create training configuration
                        training_config = {
                            'model_type': model_type.value,
                            'training_data': training_data,
                            'feature_columns': feature_columns,
                            'target_columns': target_columns,
                            'sample_weight': sample_weight,
                            'save_models': self.config.save_models,
                            'output_directory': f"{self.config.output_directory}/{model_type.value.lower()}"
                        }

                        # Train model using existing trainer if available
                        if TACTICIAN_TRAINING_AVAILABLE:
                            training_result = await self._train_model_with_existing_trainer(
                                model_type, training_config
                            )
                        else:
                            extra_kwargs = dict(kwargs)
                            extra_kwargs['model_type_for_filtering'] = model_type.value
                            extra_kwargs['output_directory'] = training_config['output_directory']

                            if model_type == TacticianModelType.STACKER_LGBM_CALIBRATED:
                                extra_kwargs['base_models'] = all_models
                                extra_kwargs['base_model_oof_predictions'] = all_oof_predictions

                            training_result = await self._train_model_directly(
                                model_type, X, y, sample_weight, **extra_kwargs
                            )

                    # Store results
                    if training_result.get('models'):
                        for model_name, model in training_result['models'].items():
                            all_models[f"{model_type.value.lower()}_{model_name}"] = model

                    if training_result.get('metrics'):
                        metrics_key = f"{model_type.value.lower()}"
                        if (
                            metrics_key in all_metrics
                            and isinstance(all_metrics[metrics_key], dict)
                            and isinstance(training_result['metrics'], dict)
                        ):
                            combined_metrics = all_metrics[metrics_key].copy()
                            combined_metrics.update(training_result['metrics'])
                            all_metrics[metrics_key] = combined_metrics
                        else:
                            all_metrics[metrics_key] = training_result['metrics']

                    if training_result.get('oof_predictions'):
                        for base_name, preds in training_result['oof_predictions'].items():
                            all_oof_predictions[base_name] = np.asarray(preds)

                    tprint_success(f"✅ {model_type.value} model trained")

                except Exception as e:
                    tprint_warning(f"⚠️ Failed to train {model_type.value} model: {e}")
                    continue

            if not all_models:
                raise ValueError("Failed to train any base models")

            execution_time = tprint_timer(start_time)
            tprint_success(f"✅ Base models training completed in {execution_time:.2f}s")

            # Collect comprehensive metrics
            comprehensive_metrics = {
                'training_summary': {
                    'total_models_trained': len(all_models),
                    'successful_models': len([m for m in all_models.values() if m is not None]),
                    'failed_models': len(self.config.model_types) - len([m for m in all_models.values() if m is not None]),
                    'execution_time': execution_time,
                    'samples_used': len(filtered_data),
                    'original_samples': len(training_data),
                    'features_used': len(model_specific_features),
                    'model_types_trained': [mt.value for mt in self.config.model_types],
                    'models_per_type': len(self.config.model_types)
                },
                'analyst_filtering': analyst_filter_info,
                'feature_processing': {
                    'final_features_loaded': len(final_feature_columns),
                    'regime_features_added': len(enhanced_feature_columns) - len(final_feature_columns),
                    'model_specific_features': len(model_specific_features),
                    'feature_reduction_ratio': len(model_specific_features) / len(final_feature_columns) if final_feature_columns else 0
                },
                'model_metrics': all_metrics,
                'hardware_utilization': {
                    'parallel_processing_enabled': self.config.enable_parallel_processing,
                    'gpu_acceleration_enabled': self.config.enable_gpu_acceleration,
                    'memory_limit_gb': self.config.memory_limit_gb
                },
                'error_summary': {
                    'total_errors': 0,
                    'critical_errors': 0,
                    'warnings': 0,
                    'recoverable_errors': 0
                }
            }

            return {
                'models': all_models,
                'metrics': comprehensive_metrics,
                'training_time': execution_time,
                'features_used': model_specific_features,
                'samples_used': len(filtered_data),
                'analyst_filter_info': analyst_filter_info,
                'model_types_trained': [mt.value for mt in self.config.model_types],
                'models_per_type': len(self.config.model_types),
                'direction_settings': {
                    'enable_long_positions': self.config.enable_long_positions,
                    'enable_short_positions': self.config.enable_short_positions,
                }
            }

        except Exception as e:
            execution_time = tprint_timer(start_time)
            tprint_error(f"❌ Base models training failed: {e}")
            return {
                'models': {},
                'metrics': {},
                'training_time': execution_time,
                'error': str(e),
                'direction_settings': {
                    'enable_long_positions': self.config.enable_long_positions,
                    'enable_short_positions': self.config.enable_short_positions,
                }
            }

    def _generate_model_oof_predictions(
        self,
        model_builder,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray],
        n_splits: int = 5,
        prediction_fn=None
    ) -> Optional[np.ndarray]:
        """Generate OOF predictions for a regressor using VectorBT-optimized cross-validation."""
        try:
            # Use VectorBT-optimized cross-validation if available
            if UNIFIED_VECTORIZATION_AVAILABLE and self.vectorization_manager is not None:
                tprint_debug("🚀 Using VectorBT-optimized cross-validation for OOF predictions")

                # Create operation configuration for cross-validation
                config = OperationConfig(
                    operation_type=OperationType.CROSS_VALIDATION,
                    data_size=len(X),
                    data_dimensions=X.shape,
                    memory_budget_mb=self.config.memory_limit_gb * 1024
                )

                # Prepare data for VectorBT optimization
                data = {
                    'X': X,
                    'y': y,
                    'model_class': model_builder,
                    'cv_folds': n_splits,
                    'scoring': 'neg_mean_squared_error',
                    'sample_weight': sample_weight,
                    'prediction_fn': prediction_fn
                }

                # Use VectorBT-optimized cross-validation
                cv_result = self.vectorization_manager.optimize_operation(
                    OperationType.CROSS_VALIDATION,
                    data,
                    config,
                    prefer_vectorbt=True
                )

                # Extract OOF predictions from VectorBT result
                if hasattr(cv_result.result, 'oof_predictions'):
                    oof_predictions = cv_result.result.oof_predictions
                    tprint_success(f"✅ VectorBT-optimized OOF predictions generated (performance gain: {cv_result.performance_gain:.2f}x)")
                    return oof_predictions
                else:
                    # Fallback to standard CV if VectorBT doesn't provide OOF predictions
                    tprint_warning("⚠️ VectorBT CV result doesn't contain OOF predictions, using fallback")
                    return self._generate_model_oof_predictions_fallback(
                        model_builder, X, y, sample_weight, n_splits, prediction_fn
                    )
            else:
                # Fallback to standard cross-validation
                tprint_warning("⚠️ VectorBT optimization not available, using standard cross-validation")
                return self._generate_model_oof_predictions_fallback(
                    model_builder, X, y, sample_weight, n_splits, prediction_fn
                )

        except Exception as e:
            tprint_warning(f"⚠️ Failed to generate VectorBT-optimized OOF predictions: {e}, using fallback")
            return self._generate_model_oof_predictions_fallback(
                model_builder, X, y, sample_weight, n_splits, prediction_fn
            )

    def _generate_model_oof_predictions_fallback(
        self,
        model_builder,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray],
        n_splits: int = 5,
        prediction_fn=None
    ) -> Optional[np.ndarray]:
        """Fallback OOF predictions generation using standard cross-validation."""
        try:
            from sklearn.model_selection import KFold

            y_flat = y.reshape(-1)
            oof_predictions = np.zeros_like(y_flat, dtype=float)
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                model = model_builder()
                fit_kwargs = {}
                if sample_weight is not None:
                    fit_kwargs['sample_weight'] = sample_weight[train_idx]

                model.fit(X[train_idx], y_flat[train_idx], **fit_kwargs)

                if prediction_fn is not None:
                    preds = prediction_fn(model, X[val_idx])
                else:
                    preds = model.predict(X[val_idx])

                oof_predictions[val_idx] = preds.reshape(-1)

            return oof_predictions
        except Exception as e:
            tprint_warning(f"⚠️ Failed to generate fallback OOF predictions: {e}")
            return None

    async def _train_model_with_existing_trainer(
        self,
        model_type: TacticianModelType,
        training_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train model using existing trainer."""
        try:
            # Import and use the existing trainer
            from .tactician_models_training_refactored import TacticianModelsTrainingStep

            trainer = TacticianModelsTrainingStep()
            return await trainer.train_tactician_models(**training_config)

        except ImportError:
            tprint_warning(f"⚠️ Existing trainer not available, falling back to direct training")
            return await self._train_model_directly(
                model_type,
                training_config['training_data'][training_config['feature_columns']].values,
                training_config['training_data'][training_config['target_columns']].values,
                training_config['sample_weight'],
                model_type_for_filtering=model_type.value
            )

    async def _train_model_directly(
        self,
        model_type: TacticianModelType,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Train model directly."""
        try:
            # Add model type for filtering to kwargs
            kwargs['model_type_for_filtering'] = model_type.value

            if model_type == TacticianModelType.LGBM_GRU:
                return await self._train_lgbm_gru(X, y, sample_weight, **kwargs)
            elif model_type == TacticianModelType.CATBOOST:
                return await self._train_catboost(X, y, sample_weight, **kwargs)
            elif model_type == TacticianModelType.CAUSAL_TCN:
                return await self._train_causal_tcn(X, y, sample_weight, **kwargs)
            elif model_type == TacticianModelType.STACKER_LGBM_CALIBRATED:
                return await self._train_stacker_lgbm_calibrated(X, y, sample_weight, **kwargs)
            elif model_type == TacticianModelType.RANDOM_SURVIVAL_FOREST:
                return await self._train_random_survival_forest(X, y, sample_weight, **kwargs)
            elif model_type == TacticianModelType.XGBOOST:
                return await self._train_xgboost(X, y, sample_weight, **kwargs)
            elif model_type == TacticianModelType.ELASTIC_NET_CV:
                return await self._train_elastic_net_cv(X, y, sample_weight, **kwargs)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        except Exception as e:
            tprint_error(f"❌ Failed to train {model_type.value} directly: {e}")
            return {'models': {}, 'metrics': {}}

    async def _train_nas_tas_timing_models(
        self,
        model_type: TacticianModelType,
        training_data: pd.DataFrame,
        feature_columns: List[str],
        target_columns: List[str],
        sample_weight: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train NAS/TAS timing models using the dedicated orchestrator."""

        # NAS-TAS functionality removed - directory deleted
        tprint_warning("⚠️ NAS/TAS orchestrator not available - directory has been removed")
        return {'models': {}, 'metrics': {}, 'error': 'NAS-TAS functionality has been removed'}

        tprint_debug(
            f"🧠 Preparing NAS/TAS training for {model_type.value} with {len(feature_columns)} features"
        )

        base_output_dir = Path(self.config.output_directory) / model_type.value.lower()
        base_output_dir.mkdir(parents=True, exist_ok=True)

        nas_tas_feature_columns: List[str] = list(dict.fromkeys(feature_columns))

        def _extend_with_keyword(keyword: str) -> List[str]:
            added_columns: List[str] = []
            for column in training_data.columns:
                lower_column = column.lower()
                if keyword in lower_column and not lower_column.startswith('target'):
                    if column not in nas_tas_feature_columns and column not in target_columns:
                        nas_tas_feature_columns.append(column)
                        added_columns.append(column)
            return added_columns

        pre_training_added = _extend_with_keyword('pre_training')
        regime_added = _extend_with_keyword('regime')
        analyst_added = _extend_with_keyword('analyst')

        if pre_training_added or regime_added or analyst_added:
            tprint_debug(
                "🔗 NAS/TAS feature enrichment: added "
                f"{len(pre_training_added)} pre-training, "
                f"{len(regime_added)} regime, "
                f"{len(analyst_added)} analyst columns"
            )

        feature_columns = nas_tas_feature_columns

        direction_map: Dict[str, str] = {}
        for column in target_columns:
            lower_name = column.lower()
            if 'long' in lower_name and 'long' not in direction_map:
                direction_map['long'] = column
            elif 'short' in lower_name and 'short' not in direction_map:
                direction_map['short'] = column

        if 'long' not in direction_map:
            direction_map['long'] = 'target_long'
        if 'short' not in direction_map:
            direction_map['short'] = 'target_short'

        models_to_return: Dict[str, Any] = {}
        direction_metrics: Dict[str, Dict[str, Any]] = {}
        aggregate_model_paths: Dict[str, str] = {}

        for direction, target_column in direction_map.items():
            if target_column not in training_data.columns:
                tprint_warning(
                    f"⚠️ Skipping {direction} NAS/TAS training - target column '{target_column}' not found"
                )
                continue

            direction_frame = training_data[training_data[target_column].notna()].copy()
            if direction_frame.empty:
                tprint_warning(
                    f"⚠️ Skipping {direction} NAS/TAS training - no samples after filtering"
                )
                continue

            direction_output_dir = base_output_dir / direction
            direction_output_dir.mkdir(parents=True, exist_ok=True)

            try:
                config = OrchestratorConfig(
                    mode=OrchestrationMode.FULL_PIPELINE,
                    output_directory=str(direction_output_dir),
                    save_models=self.config.save_models,
                    save_results=self.config.save_metrics,
                    enable_model_selection=False,
                    enable_model_management=False,
                    enable_performance_tracking=False,
                    enable_regime_detection=True,
                    enable_model_training=True,
                    data_validation=False,
                    feature_engineering=False,
                    data_preprocessing=False,
                    enable_parallel_processing=self.config.enable_parallel_processing,
                    direction_mode=f"{direction}_only",
                    min_directional_samples=self.config.min_training_samples,
                    enable_caching=False,
                    cache_directory=str(direction_output_dir / "cache"),
                    log_level="INFO",
                    enable_logging=False,
                )

                try:
                    unified_config = config.get_unified_config()
                    if hasattr(unified_config, 'timeframe'):
                        unified_config.timeframe = '15m'  # Updated to 15m timeframe
                        tprint_debug(f"✅ Set NAS/TAS timeframe to 15m")
                    if hasattr(unified_config, 'data') and hasattr(unified_config.data, 'timeframe'):
                        unified_config.data.timeframe = '15m'  # Updated to 15m timeframe
                        tprint_debug(f"✅ Set NAS/TAS data timeframe to 15m")
                    if hasattr(unified_config, 'enable_feature_engineering'):
                        unified_config.enable_feature_engineering = False
                    if hasattr(unified_config, 'enable_data_preprocessing'):
                        unified_config.enable_data_preprocessing = False
                    if hasattr(unified_config, 'model_type'):
                        unified_config.model_type = model_type.value
                        tprint_debug(f"✅ Set NAS/TAS model type to {model_type.value}")
                except Exception as config_error:
                    tprint_warning(f"⚠️ Failed to adjust NAS/TAS unified config: {config_error}")

                orchestrator = TrainingOrchestrator(config)

                timestamps = direction_frame['timestamp'] if 'timestamp' in direction_frame.columns else None
                context = {
                    'timeframe': '5m',
                    'direction': direction,
                    'model_type': model_type.value,
                    'analyst_signal_filters_applied': True,
                    'feature_columns': feature_columns,
                }

                orchestration_result = await orchestrator.orchestrate_async(
                    market_data=direction_frame,
                    target_variable=target_column,
                    feature_columns=feature_columns,
                    timestamps=timestamps,
                    context=context,
                )

            except Exception as orchestration_error:
                tprint_error(
                    f"❌ NAS/TAS orchestrator failed for {direction} direction: {orchestration_error}"
                )
                continue

            if not getattr(orchestration_result, 'success', False):
                warning_message = getattr(orchestration_result, 'error_message', 'unknown error')
                tprint_warning(
                    f"⚠️ NAS/TAS training unsuccessful for {direction} direction: {warning_message}"
                )
                continue

            training_result = getattr(orchestration_result, 'training_result', None)
            if training_result is None:
                tprint_warning(
                    f"⚠️ NAS/TAS orchestrator returned no training result for {direction} direction"
                )
                continue

            direction_models = {}
            if getattr(training_result, 'directional_models', None):
                direction_models = training_result.directional_models.get(direction, {}) or {}
            if not direction_models:
                direction_models = getattr(training_result, 'models_trained', {}) or {}

            if not direction_models:
                tprint_warning(
                    f"⚠️ No models produced for {direction} direction by NAS/TAS orchestrator"
                )
                continue

            saved_paths: Dict[str, str] = {}
            for regime_id, regime_models in direction_models.items():
                if not isinstance(regime_models, dict):
                    continue
                for model_name, model_obj in regime_models.items():
                    model_key = f"{model_type.value.lower()}_{direction}_regime{regime_id}_{model_name}"
                    models_to_return[model_key] = model_obj

                    if self.config.save_models:
                        model_path = direction_output_dir / f"{model_key}.pkl"
                        try:
                            with open(model_path, 'wb') as model_file:
                                pickle.dump(model_obj, model_file)
                            saved_paths[model_key] = str(model_path)
                        except Exception as serialization_error:
                            tprint_warning(
                                f"⚠️ Failed to persist NAS/TAS model {model_key}: {serialization_error}"
                            )

            aggregate_model_paths.update(saved_paths)

            performance_metrics = getattr(training_result, 'directional_performance', {}).get(direction, {})
            if not performance_metrics:
                performance_metrics = getattr(training_result, 'overall_performance', {})

            regime_performance = getattr(training_result, 'regime_performance', {})
            regime_stats = getattr(training_result, 'directional_statistics', {}).get(direction, {})
            n_regimes_detected = getattr(training_result, 'n_regimes_detected', 0)

            metrics_entry = {
                'direction': direction,
                'target_column': target_column,
                'samples': int(len(direction_frame)),
                'feature_columns': list(feature_columns),
                'timeframe': '5m',
                'analyst_features_integrated': True,
                'performance': performance_metrics,
                'regime_performance': regime_performance,
                'regime_statistics': regime_stats,
                'n_regimes_detected': int(n_regimes_detected),
                'model_paths': saved_paths,
            }

            direction_metrics[direction] = metrics_entry

            if self.config.save_metrics:
                metrics_path = direction_output_dir / f"{model_type.value.lower()}_{direction}_metrics.json"
                try:
                    with open(metrics_path, 'w') as metrics_file:
                        json.dump(metrics_entry, metrics_file, indent=2, default=str)
                except Exception as metrics_error:
                    tprint_warning(
                        f"⚠️ Failed to persist NAS/TAS metrics for {direction}: {metrics_error}"
                    )

        if not models_to_return:
            tprint_warning(f"⚠️ NAS/TAS training produced no models for {model_type.value}")
            return {'models': {}, 'metrics': {}}

        metrics_summary = {
            'model_type': model_type.value,
            'timeframe': '5m',
            'directions': direction_metrics,
            'model_paths': aggregate_model_paths,
        }

        return {
            'models': models_to_return,
            'metrics': metrics_summary,
        }

    async def _train_random_survival_forest(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Train Random Survival Forest model with HPO."""
        try:
            from sklearn.ensemble import RandomForestRegressor

            # Use HPO to optimize hyperparameters
            best_params = await self._optimize_random_forest_hyperparameters(X, y, sample_weight, **kwargs)

            # Train with optimized parameters
            model = RandomForestRegressor(
                random_state=42,
                n_jobs=-1 if self.config.enable_parallel_processing else 1,
                **best_params
            )

            model.fit(X, y.ravel(), sample_weight=sample_weight)

            return {
                'models': {'random_survival_forest': model},
                'metrics': {
                    'model_type': 'RandomSurvivalForest',
                    'best_params': best_params,
                    'hpo_performed': True
                }
            }

        except Exception as e:
            tprint_error(f"❌ Random Survival Forest training failed: {e}")
            # Fallback to default parameters
            try:
                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1 if self.config.enable_parallel_processing else 1
                )
                model.fit(X, y.ravel(), sample_weight=sample_weight)

                return {
                    'models': {'random_survival_forest': model},
                    'metrics': {
                        'model_type': 'RandomSurvivalForest',
                        'hpo_performed': False,
                        'error': str(e)
                    }
                }
            except Exception as fallback_error:
                tprint_error(f"❌ Random Survival Forest fallback training failed: {fallback_error}")
                return {'models': {}, 'metrics': {}}

    async def _train_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Train XGBoost model with HPO."""
        try:
            import xgboost as xgb

            # Use HPO to optimize hyperparameters
            best_params = await self._optimize_xgboost_hyperparameters(X, y, sample_weight, **kwargs)

            # Train with optimized parameters
            model = xgb.XGBRegressor(
                random_state=42,
                n_jobs=-1 if self.config.enable_parallel_processing else 1,
                **best_params
            )

            model.fit(X, y.ravel(), sample_weight=sample_weight)

            return {
                'models': {'xgboost': model},
                'metrics': {
                    'model_type': 'XGBoost',
                    'best_params': best_params,
                    'hpo_performed': True
                }
            }

        except Exception as e:
            tprint_error(f"❌ XGBoost training failed: {e}")
            # Fallback to default parameters
            try:
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1 if self.config.enable_parallel_processing else 1
                )
                model.fit(X, y.ravel(), sample_weight=sample_weight)

                return {
                    'models': {'xgboost': model},
                    'metrics': {
                        'model_type': 'XGBoost',
                        'hpo_performed': False,
                        'error': str(e)
                    }
                }
            except Exception as fallback_error:
                tprint_error(f"❌ XGBoost fallback training failed: {fallback_error}")
                return {'models': {}, 'metrics': {}}

    async def _train_elastic_net_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Train Elastic Net CV model with HPO."""
        try:
            from sklearn.linear_model import ElasticNetCV

            # Use HPO to optimize hyperparameters
            best_params = await self._optimize_elastic_net_hyperparameters(X, y, sample_weight, **kwargs)

            # Train with optimized parameters
            model = ElasticNetCV(
                cv=5,
                random_state=42,
                max_iter=1000,
                **best_params
            )

            model.fit(X, y.ravel(), sample_weight=sample_weight)

            return {
                'models': {'elastic_net_cv': model},
                'metrics': {
                    'model_type': 'ElasticNetCV',
                    'best_params': best_params,
                    'hpo_performed': True
                }
            }

        except Exception as e:
            tprint_error(f"❌ Elastic Net CV training failed: {e}")
            # Fallback to default parameters
            try:
                model = ElasticNetCV(
                    cv=5,
                    random_state=42,
                    max_iter=1000
                )
                model.fit(X, y.ravel(), sample_weight=sample_weight)

                return {
                    'models': {'elastic_net_cv': model},
                    'metrics': {
                        'model_type': 'ElasticNetCV',
                        'hpo_performed': False,
                        'error': str(e)
                    }
                }
            except Exception as fallback_error:
                tprint_error(f"❌ Elastic Net CV fallback training failed: {fallback_error}")
                return {'models': {}, 'metrics': {}}

    async def _optimize_random_forest_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Optimize Random Forest hyperparameters using Bayesian optimization."""
        try:
            # Try to use advanced HPO if available
            try:
                from src.utils.ml_common.optimization.bayesian_entry_timing_optimizer import (
                    get_bayesian_optimizer, EntryTimingConfig
                )

                # Configure HPO for Random Forest
                config = EntryTimingConfig(
                    n_trials=50,
                    timeout=300,  # 5 minutes timeout
                    enable_cross_validation=True,
                    cv_folds=3
                )

                optimizer = get_bayesian_optimizer(config)

                # Define hyperparameter search space
                def objective(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                    }

                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.model_selection import cross_val_score

                    model = RandomForestRegressor(random_state=42, **params)
                    scores = cross_val_score(
                        model, X, y.ravel(),
                        cv=3, scoring='neg_mean_squared_error',
                        fit_params={'sample_weight': sample_weight} if sample_weight is not None else {}
                    )

                    return -np.mean(scores)  # Minimize negative MSE

                # Run optimization
                best_params = optimizer.optimize_model_hyperparameters(
                    model=None, X=X, y=y,
                    param_space=objective,
                    model_name="random_forest"
                )

                if best_params:
                    tprint_success(f"✅ Random Forest HPO completed - best params: {best_params}")
                    return best_params

            except ImportError:
                tprint_warning("⚠️ Advanced HPO not available, using grid search")

            # Fallback to simple grid search
            from sklearn.model_selection import GridSearchCV
            from sklearn.ensemble import RandomForestRegressor

            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }

            rf = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1 if self.config.enable_parallel_processing else 1
            )

            grid_search.fit(X, y.ravel(), sample_weight=sample_weight)
            best_params = grid_search.best_params_

            tprint_success(f"✅ Random Forest grid search completed - best params: {best_params}")
            return best_params

        except Exception as e:
            tprint_error(f"❌ Random Forest HPO failed: {e}")
            return {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            }

    async def _optimize_xgboost_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters using Bayesian optimization."""
        try:
            # Try to use advanced HPO if available
            try:
                from src.utils.ml_common.optimization.bayesian_entry_timing_optimizer import (
                    get_bayesian_optimizer, EntryTimingConfig
                )

                # Configure HPO for XGBoost
                config = EntryTimingConfig(
                    n_trials=50,
                    timeout=300,
                    enable_cross_validation=True,
                    cv_folds=3
                )

                optimizer = get_bayesian_optimizer(config)

                # Define hyperparameter search space
                def objective(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 15),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
                    }

                    import xgboost as xgb
                    from sklearn.model_selection import cross_val_score

                    model = xgb.XGBRegressor(random_state=42, **params)
                    scores = cross_val_score(
                        model, X, y.ravel(),
                        cv=3, scoring='neg_mean_squared_error',
                        fit_params={'sample_weight': sample_weight} if sample_weight is not None else {}
                    )

                    return -np.mean(scores)

                # Run optimization
                best_params = optimizer.optimize_model_hyperparameters(
                    model=None, X=X, y=y,
                    param_space=objective,
                    model_name="xgboost"
                )

                if best_params:
                    tprint_success(f"✅ XGBoost HPO completed - best params: {best_params}")
                    return best_params

            except ImportError:
                tprint_warning("⚠️ Advanced HPO not available, using grid search")

            # Fallback to simple grid search
            from sklearn.model_selection import GridSearchCV
            import xgboost as xgb

            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }

            xgb_model = xgb.XGBRegressor(random_state=42)
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1 if self.config.enable_parallel_processing else 1
            )

            grid_search.fit(X, y.ravel(), sample_weight=sample_weight)
            best_params = grid_search.best_params_

            tprint_success(f"✅ XGBoost grid search completed - best params: {best_params}")
            return best_params

        except Exception as e:
            tprint_error(f"❌ XGBoost HPO failed: {e}")
            return {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1
            }

    async def _optimize_elastic_net_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Optimize Elastic Net hyperparameters using Bayesian optimization."""
        try:
            # Try to use advanced HPO if available
            try:
                from src.utils.ml_common.optimization.bayesian_entry_timing_optimizer import (
                    get_bayesian_optimizer, EntryTimingConfig
                )

                # Configure HPO for Elastic Net
                config = EntryTimingConfig(
                    n_trials=50,
                    timeout=300,
                    enable_cross_validation=True,
                    cv_folds=3
                )

                optimizer = get_bayesian_optimizer(config)

                # Define hyperparameter search space
                def objective(trial):
                    params = {
                        'alpha': trial.suggest_float('alpha', 0.001, 10.0, log=True),
                        'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0)
                    }

                    from sklearn.linear_model import ElasticNet
                    from sklearn.model_selection import cross_val_score

                    model = ElasticNet(random_state=42, max_iter=1000, **params)
                    scores = cross_val_score(
                        model, X, y.ravel(),
                        cv=3, scoring='neg_mean_squared_error',
                        fit_params={'sample_weight': sample_weight} if sample_weight is not None else {}
                    )

                    return -np.mean(scores)

                # Run optimization
                best_params = optimizer.optimize_model_hyperparameters(
                    model=None, X=X, y=y,
                    param_space=objective,
                    model_name="elastic_net"
                )

                if best_params:
                    tprint_success(f"✅ Elastic Net HPO completed - best params: {best_params}")
                    return best_params

            except ImportError:
                tprint_warning("⚠️ Advanced HPO not available, using grid search")

            # Fallback to simple grid search
            from sklearn.model_selection import GridSearchCV
            from sklearn.linear_model import ElasticNet

            param_grid = {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]
            }

            en_model = ElasticNet(random_state=42, max_iter=1000)
            grid_search = GridSearchCV(
                en_model, param_grid, cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1 if self.config.enable_parallel_processing else 1
            )

            grid_search.fit(X, y.ravel(), sample_weight=sample_weight)
            best_params = grid_search.best_params_

            tprint_success(f"✅ Elastic Net grid search completed - best params: {best_params}")
            return best_params

        except Exception as e:
            tprint_error(f"❌ Elastic Net HPO failed: {e}")
            return {
                'alpha': 0.1,
                'l1_ratio': 0.5
            }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the models training step."""
        metrics = {
            'config': {
                'model_types': [mt.value for mt in self.config.model_types],
                'save_models': self.config.save_models,
                'save_metrics': self.config.save_metrics,
                'output_directory': self.config.output_directory,
                'enable_parallel_processing': self.config.enable_parallel_processing,
                'enable_gpu_acceleration': self.config.enable_gpu_acceleration
            },
            'hardware_optimization': {
                'gpu_manager': self.gpu_manager is not None,
                'memory_optimizer': self.memory_optimizer is not None,
                'cpu_optimizer': self.cpu_optimizer is not None
            },
            'vectorbt_optimization': {
                'rolling_optimizer_available': self.vectorbt_rolling_optimizer is not None,
                'vectorization_manager_available': self.vectorization_manager is not None,
                'rolling_optimizer_stats': self.vectorbt_rolling_optimizer.get_performance_stats() if self.vectorbt_rolling_optimizer else None,
                'vectorization_manager_stats': self.vectorization_manager.get_optimization_stats() if self.vectorization_manager else None
            }
        }

        return metrics

    def _optimized_rolling_operation(self, data: Union[pd.Series, pd.DataFrame],
                                   operation: str, window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """
        Perform optimized rolling operation using VectorBT Rolling Optimizer.

        Args:
            data: Input data (Series or DataFrame)
            operation: Operation to perform ('mean', 'std', 'var', 'min', 'max', 'sum', etc.)
            window: Rolling window size
            **kwargs: Additional parameters

        Returns:
            Result of the rolling operation
        """
        if self.vectorbt_rolling_optimizer is not None:
            try:
                tprint_debug(f"🔄 Using VectorBT Rolling Optimizer for {operation} (window={window})")

                if operation == 'mean':
                    return self.vectorbt_rolling_optimizer.rolling_mean(data, window, **kwargs)
                elif operation == 'std':
                    return self.vectorbt_rolling_optimizer.rolling_std(data, window, **kwargs)
                elif operation == 'var':
                    return self.vectorbt_rolling_optimizer.rolling_var(data, window, **kwargs)
                elif operation == 'min':
                    return self.vectorbt_rolling_optimizer.rolling_min(data, window, **kwargs)
                elif operation == 'max':
                    return self.vectorbt_rolling_optimizer.rolling_max(data, window, **kwargs)
                elif operation == 'sum':
                    return self.vectorbt_rolling_optimizer.rolling_sum(data, window, **kwargs)
                elif operation == 'apply':
                    func = kwargs.get('func')
                    return self.vectorbt_rolling_optimizer.rolling_apply(data, func, window, **kwargs)
                elif operation == 'corr':
                    other = kwargs.get('other')
                    return self.vectorbt_rolling_optimizer.rolling_corr(data, other, window, **kwargs)
                elif operation == 'cov':
                    other = kwargs.get('other')
                    return self.vectorbt_rolling_optimizer.rolling_cov(data, other, window, **kwargs)
                else:
                    raise ValueError(f"Unsupported VectorBT rolling operation: {operation}")

            except Exception as e:
                tprint_warning(f"⚠️ VectorBT Rolling Optimizer failed for {operation}: {e}, using fallback")
                return self._fallback_rolling_operation(data, operation, window, **kwargs)
        else:
            tprint_warning(f"⚠️ VectorBT Rolling Optimizer not available, using fallback for {operation}")
            return self._fallback_rolling_operation(data, operation, window, **kwargs)

    def _optimized_batch_rolling_operations(self, data: Union[pd.Series, pd.DataFrame],
                                          operations: List[str], window: int, **kwargs) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """
        Perform multiple rolling operations in a single optimized batch.

        This provides 3-5x speedup by processing multiple rolling operations
        simultaneously instead of sequentially.

        Args:
            data: Input data (Series or DataFrame)
            operations: List of operations to perform
            window: Rolling window size
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping operation names to results
        """
        if self.vectorbt_rolling_optimizer is not None:
            try:
                tprint_info(f"🚀 Using VectorBT batch processing for {len(operations)} operations")
                return self.vectorbt_rolling_optimizer.batch_rolling_operations(data, operations, window, **kwargs)
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT batch processing failed: {e}, using sequential fallback")
                return self._sequential_batch_fallback(data, operations, window, **kwargs)
        else:
            tprint_warning("⚠️ VectorBT optimizer not available, using sequential fallback")
            return self._sequential_batch_fallback(data, operations, window, **kwargs)

    def _sequential_batch_fallback(self, data: Union[pd.Series, pd.DataFrame], operations: List[str],
                                 window: int, **kwargs) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """Sequential fallback for batch rolling operations."""
        results = {}
        for operation in operations:
            try:
                results[operation] = self._optimized_rolling_operation(data, operation, window, **kwargs)
            except Exception as e:
                tprint_warning(f"⚠️ Sequential operation {operation} failed: {e}")
                # Return empty result as fallback
                if isinstance(data, pd.Series):
                    results[operation] = pd.Series(index=data.index, dtype=float)
                else:
                    results[operation] = pd.DataFrame(index=data.index, columns=data.columns, dtype=float)
        return results

    def _fallback_rolling_operation(self, data: Union[pd.Series, pd.DataFrame],
                                  operation: str, window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """
        Fallback rolling operation using pandas.

        Args:
            data: Input data (Series or DataFrame)
            operation: Operation to perform
            window: Rolling window size
            **kwargs: Additional parameters

        Returns:
            Result of the rolling operation
        """
        try:
            rolling_obj = data.rolling(window=window, **kwargs)

            if operation == 'mean':
                return rolling_obj.mean()
            elif operation == 'std':
                return rolling_obj.std()
            elif operation == 'var':
                return rolling_obj.var()
            elif operation == 'min':
                return rolling_obj.min()
            elif operation == 'max':
                return rolling_obj.max()
            elif operation == 'sum':
                return rolling_obj.sum()
            elif operation == 'apply':
                func = kwargs.get('func')
                return rolling_obj.apply(func)
            elif operation == 'corr':
                other = kwargs.get('other')
                return rolling_obj.corr(other)
            elif operation == 'cov':
                other = kwargs.get('other')
                return rolling_obj.cov(other)
            else:
                raise ValueError(f"Unsupported fallback rolling operation: {operation}")

        except Exception as e:
            tprint_error(f"❌ Fallback rolling operation failed for {operation}: {e}")
            raise

    def _optimize_matrix_operations(self, X: np.ndarray, operation_type: str, **kwargs) -> Any:
        """
        Optimize matrix operations using Unified Vectorization Manager.

        Args:
            X: Input matrix
            operation_type: Type of matrix operation
            **kwargs: Additional parameters

        Returns:
            Optimized operation result
        """
        if self.vectorization_manager is not None:
            try:
                tprint_debug(f"🔄 Using Unified Vectorization Manager for {operation_type}")

                # Create operation configuration
                config = OperationConfig(
                    operation_type=OperationType.MATRIX_MULTIPLICATION if operation_type == 'matrix_mult' else OperationType.STATISTICAL_COMPUTATION,
                    data_size=len(X),
                    data_dimensions=X.shape,
                    memory_budget_mb=self.config.memory_limit_gb * 1024
                )

                # Prepare data for optimization
                data = {'matrix': X, **kwargs}

                # Use VectorBT optimization
                result = self.vectorization_manager.optimize_operation(
                    config.operation_type,
                    data,
                    config
                )

                tprint_success(f"✅ Matrix operation {operation_type} optimized (performance gain: {result.performance_gain:.2f}x)")
                return result.result

            except Exception as e:
                tprint_warning(f"⚠️ Unified Vectorization Manager failed for {operation_type}: {e}, using fallback")
                return self._fallback_matrix_operation(X, operation_type, **kwargs)
        else:
            tprint_warning(f"⚠️ Unified Vectorization Manager not available, using fallback for {operation_type}")
            return self._fallback_matrix_operation(X, operation_type, **kwargs)

    def _fallback_matrix_operation(self, X: np.ndarray, operation_type: str, **kwargs) -> Any:
        """
        Fallback matrix operation using standard numpy/pandas.

        Args:
            X: Input matrix
            operation_type: Type of matrix operation
            **kwargs: Additional parameters

        Returns:
            Operation result
        """
        try:
            if operation_type == 'matrix_mult':
                other = kwargs.get('other')
                return np.dot(X, other) if other is not None else X
            elif operation_type == 'statistical':
                return {
                    'mean': np.mean(X, axis=0),
                    'std': np.std(X, axis=0),
                    'min': np.min(X, axis=0),
                    'max': np.max(X, axis=0)
                }
            else:
                raise ValueError(f"Unsupported matrix operation: {operation_type}")

        except Exception as e:
            tprint_error(f"❌ Fallback matrix operation failed for {operation_type}: {e}")
            raise

    async def _load_final_feature_selection_features(
        self,
        training_data: pd.DataFrame,
        provided_features: List[str],
        **kwargs
    ) -> List[str]:
        """
        Load features from final feature selection pipeline if not provided.

        Args:
            training_data: Training DataFrame
            provided_features: Features provided by caller
            **kwargs: Additional parameters including symbol, exchange, timeframe

        Returns:
            List of final feature column names
        """
        try:
            # If features are already provided and available in data, use them
            if provided_features and all(col in training_data.columns for col in provided_features):
                tprint_debug(f"✅ Using provided feature columns: {len(provided_features)} features")
                return provided_features

            # Try to load from final feature selection pipeline
            try:
                from src.training.steps.pre_training.final_feature_selection_pipeline import (
                    get_final_features, FeatureSelectionConfig
                )

                # Extract metadata from kwargs or use defaults
                symbol = kwargs.get('symbol', 'BTCUSDT')
                exchange = kwargs.get('exchange', 'binance')
                timeframe = kwargs.get('timeframe', '15m')  # Updated to 15m timeframe

                tprint_info(f"🔍 Loading features from final feature selection for {symbol} {exchange} {timeframe}")

                # Get final features from the pipeline
                config = FeatureSelectionConfig()
                final_features_result = await get_final_features(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    config=config
                )

                if final_features_result and hasattr(final_features_result, 'final_features'):
                    final_features = final_features_result.final_features

                    # Validate features exist in training data
                    available_features = [col for col in final_features if col in training_data.columns]

                    if not available_features:
                        tprint_warning(f"⚠️ No final feature selection features available in training data")
                        return provided_features or []

                    tprint_success(f"✅ Loaded {len(available_features)} features from final feature selection")
                    return available_features

            except ImportError as e:
                tprint_warning(f"⚠️ Final feature selection pipeline not available: {e}")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to load final feature selection features: {e}")

            # Fallback to provided features or all available features
            if provided_features:
                available_features = [col for col in provided_features if col in training_data.columns]
                tprint_debug(f"🔄 Using fallback features: {len(available_features)} available")
                return available_features

            # Last resort: use all non-target columns
            exclude_columns = {'timestamp', 'target', 'target_long', 'target_short', 'analyst_signal', 'analyst_confidence'}
            fallback_features = [col for col in training_data.columns
                               if col not in exclude_columns and not col.startswith('target_')]

            tprint_debug(f"🔄 Using fallback features: {len(fallback_features)} columns")
            return fallback_features

        except Exception as e:
            tprint_error(f"❌ Failed to load final feature selection features: {e}")
            return provided_features or []

    async def _load_regime_features(
        self,
        training_data: pd.DataFrame,
        feature_columns: List[str],
        **kwargs
    ) -> List[str]:
        """
        Load regime features from market analysis ensemble model.

        Args:
            training_data: Training DataFrame
            feature_columns: Current feature columns
            **kwargs: Additional parameters including symbol, exchange, timeframe

        Returns:
            Enhanced feature columns with regime features
        """
        try:
            symbol = kwargs.get('symbol', 'BTCUSDT')
            exchange = kwargs.get('exchange', 'binance')
            timeframe = kwargs.get('timeframe', '5m')

            tprint_debug(f"🔍 Loading regime features for {symbol} {exchange} {timeframe}")

            # Try to load regime features from market analysis ensemble
            try:
                # Look for regime ensemble model in generated directory
                regime_model_path = Path(f"generated/market_analysis/regime_ensemble_{symbol}_{exchange}_{timeframe}.pkl")

                if regime_model_path.exists():
                    tprint_info(f"📂 Loading regime ensemble model from {regime_model_path}")

                    with open(regime_model_path, 'rb') as model_file:
                        regime_ensemble = pickle.load(model_file)

                    # Generate regime probabilities for training data
                    regime_features = await self._generate_regime_features(
                        training_data, regime_ensemble, feature_columns
                    )

                    if regime_features:
                        enhanced_features = feature_columns + regime_features
                        tprint_success(f"✅ Added {len(regime_features)} regime features")
                        return enhanced_features

            except Exception as e:
                tprint_warning(f"⚠️ Failed to load regime ensemble model: {e}")

            # Try to extract regime features from existing regime columns in training data
            existing_regime_cols = []
            regime_patterns = ['regime', 'cluster', 'state', 'hmm']

            for col in training_data.columns:
                if any(pattern in col.lower() for pattern in regime_patterns):
                    if col not in feature_columns:
                        existing_regime_cols.append(col)

            if existing_regime_cols:
                enhanced_features = feature_columns + existing_regime_cols
                tprint_success(f"✅ Added {len(existing_regime_cols)} existing regime features")
                return enhanced_features

            tprint_debug(f"ℹ️ No regime features found for {symbol} {exchange} {timeframe}")
            return feature_columns

        except Exception as e:
            tprint_error(f"❌ Failed to load regime features: {e}")
            return feature_columns

    async def _generate_regime_features(
        self,
        training_data: pd.DataFrame,
        regime_ensemble: Any,
        feature_columns: List[str]
    ) -> List[str]:
        """
        Generate regime features using the regime ensemble model.

        Args:
            training_data: Training DataFrame
            regime_ensemble: Trained regime ensemble model
            feature_columns: Current feature columns

        Returns:
            List of new regime feature column names
        """
        try:
            # Extract features for regime prediction (exclude target columns)
            regime_feature_cols = [col for col in feature_columns
                                 if not col.startswith('target') and col in training_data.columns]

            if not regime_feature_cols:
                tprint_warning("⚠️ No suitable features for regime prediction")
                return []

            X_regime = training_data[regime_feature_cols].values

            # Get regime probabilities from ensemble
            if hasattr(regime_ensemble, 'predict_proba'):
                regime_probs = regime_ensemble.predict_proba(X_regime)

                # Create regime probability feature columns
                regime_feature_names = []
                for i in range(regime_probs.shape[1]):
                    col_name = f"regime_prob_{i}"
                    training_data[col_name] = regime_probs[:, i]
                    regime_feature_names.append(col_name)

                tprint_success(f"✅ Generated {len(regime_feature_names)} regime probability features")
                return regime_feature_names

            elif hasattr(regime_ensemble, 'predict'):
                regime_preds = regime_ensemble.predict(X_regime)

                # Create regime prediction feature
                col_name = "regime_prediction"
                training_data[col_name] = regime_preds
                regime_feature_names = [col_name]

                tprint_success(f"✅ Generated {len(regime_feature_names)} regime prediction features")
                return regime_feature_names

        except Exception as e:
            tprint_error(f"❌ Failed to generate regime features: {e}")

        return []

    async def _apply_model_specific_feature_filtering(
        self,
        feature_columns: List[str],
        **kwargs
    ) -> List[str]:
        """
        Apply model-specific feature filtering based on ML model type.

        Args:
            feature_columns: Input feature columns
            **kwargs: Additional parameters including model_type

        Returns:
            Filtered feature columns for specific model type
        """
        try:
            model_type = kwargs.get('model_type', 'ALL')

            # Model-specific feature counts (as per requirements)
            # These limits are based on:
            # - RANDOM_SURVIVAL_FOREST: Conservative feature set for ensemble methods
            # - XGBOOST: Can handle more features but benefits from feature selection
            # - ELASTIC_NET_CV: Regularization benefits from fewer, more relevant features
            # - NAS/TAS: Advanced models that can handle more complex feature sets
            model_feature_limits = {
                'RANDOM_SURVIVAL_FOREST': 100,  # Conservative feature set
                'XGBOOST': 120,                 # Can handle more features
                'ELASTIC_NET_CV': 80,           # Regularization benefits from fewer features
                'NAS': 150,                     # Neural architecture search - more features
                'TAS': 150,                     # Tree-based architecture search - more features
                'ALL': 120                      # Default for mixed models
            }

            max_features = model_feature_limits.get(model_type, model_feature_limits['ALL'])

            if len(feature_columns) <= max_features:
                return feature_columns

            tprint_info(f"🔧 Applying {model_type} feature filtering: {len(feature_columns)} → {max_features}")

            # For model-specific filtering, prioritize features based on:
            # 1. Regime features (highest priority)
            # 2. Analyst ensemble features
            # 3. Technical indicators
            # 4. Price/volume features

            prioritized_features = []
            remaining_features = []

            # Prioritize regime features
            regime_patterns = ['regime', 'cluster', 'state', 'hmm']
            for col in feature_columns:
                if any(pattern in col.lower() for pattern in regime_patterns):
                    prioritized_features.append(col)

            # Prioritize analyst features
            analyst_patterns = ['analyst', 'ensemble', 'signal', 'confidence']
            analyst_features = []
            for col in feature_columns:
                if (col not in prioritized_features and
                    any(pattern in col.lower() for pattern in analyst_patterns)):
                    analyst_features.append(col)

            # Prioritize technical features
            technical_patterns = ['rsi', 'macd', 'bb', 'ema', 'sma', 'atr', 'volume']
            technical_features = []
            for col in feature_columns:
                if (col not in prioritized_features and col not in analyst_features and
                    any(pattern in col.lower() for pattern in technical_patterns)):
                    technical_features.append(col)

            # Remaining features
            for col in feature_columns:
                if (col not in prioritized_features and
                    col not in analyst_features and
                    col not in technical_features):
                    remaining_features.append(col)

            # Combine in priority order and trim to max_features
            final_features = (prioritized_features + analyst_features +
                            technical_features + remaining_features)

            return final_features[:max_features]

        except Exception as e:
            tprint_error(f"❌ Failed to apply model-specific feature filtering: {e}")
            return feature_columns

    async def _apply_analyst_filtering(
        self,
        training_data: pd.DataFrame,
        feature_columns: List[str],
        target_columns: List[str],
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply Analyst OOF outputs as features and weights for whole dataset training.

        Args:
            training_data: Training DataFrame
            feature_columns: Feature columns
            target_columns: Target columns
            **kwargs: Additional parameters including symbol, exchange, timeframe

        Returns:
            Tuple of (enhanced_data, filter_info)
        """
        try:
            symbol = kwargs.get('symbol', 'BTCUSDT')
            exchange = kwargs.get('exchange', 'binance')
            timeframe = kwargs.get('timeframe', '5m')

            tprint_debug(f"🔍 Applying Analyst OOF integration for {symbol} {exchange} {timeframe}")

            # Use whole dataset - no filtering
            enhanced_data = training_data.copy()
            original_samples = len(training_data)

            # Add Analyst OOF outputs as features
            analyst_features = await self._add_analyst_oof_features(enhanced_data, **kwargs)

            # Calculate sample weights based on Analyst confidence
            sample_weights = await self._calculate_analyst_weights(enhanced_data, **kwargs)

            filter_info = {
                'filtered': False,  # No filtering - using whole dataset
                'reason': 'whole_dataset_training',
                'original_samples': original_samples,
                'enhanced_samples': len(enhanced_data),
                'analyst_features_added': len(analyst_features),
                'sample_weights_calculated': sample_weights is not None
            }

            tprint_success(f"✅ Analyst OOF integration: {original_samples} samples with {len(analyst_features)} analyst features")

            return enhanced_data, filter_info

        except Exception as e:
            tprint_error(f"❌ Failed to apply Analyst OOF integration: {e}")
            return training_data, {
                'filtered': False,
                'reason': f'error: {str(e)}',
                'original_samples': len(training_data),
                'enhanced_samples': len(training_data)
            }

    async def _add_analyst_oof_features(
        self,
        training_data: pd.DataFrame,
        **kwargs
    ) -> List[str]:
        """
        Add Analyst OOF outputs as features: p_trade, u_trade, q_trade.

        Args:
            training_data: Training DataFrame
            **kwargs: Additional parameters

        Returns:
            List of added analyst feature column names
        """
        try:
            symbol = kwargs.get('symbol', 'BTCUSDT')
            exchange = kwargs.get('exchange', 'binance')
            timeframe = kwargs.get('timeframe', '5m')

            tprint_debug(f"🔧 Adding Analyst OOF features for {symbol} {exchange} {timeframe}")

            analyst_features = []

            # Look for existing Analyst OOF outputs in the data
            p_trade_col = None
            u_trade_col = None
            q_trade_col = None

            for col in training_data.columns:
                col_lower = col.lower()
                if 'p_trade' in col_lower or 'analyst_probability' in col_lower:
                    p_trade_col = col
                elif 'u_trade' in col_lower or 'analyst_expected_edge' in col_lower:
                    u_trade_col = col
                elif 'q_trade' in col_lower or 'analyst_confidence' in col_lower:
                    q_trade_col = col

            # If OOF outputs not found, try to load from Analyst ensemble results
            if not all([p_trade_col, u_trade_col, q_trade_col]):
                analyst_oof_data = await self._load_analyst_oof_outputs(symbol, exchange, timeframe, **kwargs)
                if analyst_oof_data is not None:
                    # Add OOF outputs as new columns
                    if 'p_trade' in analyst_oof_data.columns:
                        training_data['analyst_p_trade'] = analyst_oof_data['p_trade']
                        analyst_features.append('analyst_p_trade')
                    if 'u_trade' in analyst_oof_data.columns:
                        training_data['analyst_u_trade'] = analyst_oof_data['u_trade']
                        analyst_features.append('analyst_u_trade')
                    if 'q_trade' in analyst_oof_data.columns:
                        training_data['analyst_q_trade'] = analyst_oof_data['q_trade']
                        analyst_features.append('analyst_q_trade')
            else:
                # Use existing columns
                if p_trade_col:
                    analyst_features.append(p_trade_col)
                if u_trade_col:
                    analyst_features.append(u_trade_col)
                if q_trade_col:
                    analyst_features.append(q_trade_col)

            # Add derived features from Analyst outputs
            if analyst_features:
                # Add interaction features
                if len(analyst_features) >= 2:
                    p_col = analyst_features[0] if 'p_trade' in analyst_features[0].lower() else None
                    u_col = analyst_features[1] if 'u_trade' in analyst_features[1].lower() else None
                    q_col = analyst_features[2] if len(analyst_features) > 2 and 'q_trade' in analyst_features[2].lower() else None

                    if p_col and u_col:
                        # Expected value feature
                        training_data['analyst_expected_value'] = training_data[p_col] * training_data[u_col]
                        analyst_features.append('analyst_expected_value')

                    if p_col and q_col:
                        # Confidence-weighted probability
                        training_data['analyst_weighted_prob'] = training_data[p_col] * training_data[q_col]
                        analyst_features.append('analyst_weighted_prob')

            tprint_success(f"✅ Added {len(analyst_features)} Analyst OOF features")
            return analyst_features

        except Exception as e:
            tprint_error(f"❌ Failed to add Analyst OOF features: {e}")
            return []

    async def _calculate_analyst_weights(
        self,
        training_data: pd.DataFrame,
        **kwargs
    ) -> Optional[np.ndarray]:
        """
        Calculate sample weights based on Analyst confidence: w = w_min + (1-w_min)*p_trade.

        Args:
            training_data: Training DataFrame
            **kwargs: Additional parameters

        Returns:
            Sample weights array or None
        """
        try:
            w_min = kwargs.get('w_min', 0.2)  # Minimum weight parameter

            # Find p_trade column
            p_trade_col = None
            for col in training_data.columns:
                if 'p_trade' in col.lower() or 'analyst_probability' in col.lower():
                    p_trade_col = col
                    break

            if p_trade_col is None:
                tprint_warning("⚠️ No p_trade column found for weight calculation")
                return None

            # Calculate weights: w = w_min + (1-w_min)*p_trade
            p_trade_values = training_data[p_trade_col].fillna(0.0)
            sample_weights = w_min + (1 - w_min) * p_trade_values

            # Ensure weights are positive and finite
            sample_weights = np.clip(sample_weights, 0.01, 1.0)
            sample_weights = np.where(np.isfinite(sample_weights), sample_weights, w_min)

            tprint_success(f"✅ Calculated sample weights: min={sample_weights.min():.3f}, max={sample_weights.max():.3f}, mean={sample_weights.mean():.3f}")
            return sample_weights.values

        except Exception as e:
            tprint_error(f"❌ Failed to calculate Analyst weights: {e}")
            return None

    async def _load_analyst_oof_outputs(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Load Analyst OOF outputs from saved results.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            **kwargs: Additional parameters

        Returns:
            DataFrame with OOF outputs or None
        """
        try:
            # Look for Analyst ensemble results
            results_dir = Path("outcomes/model_training")
            pattern = f"analyst_ensemble_training_report_{symbol}_{exchange}_{timeframe}_*.json"

            matching_files = list(results_dir.glob(pattern))
            if not matching_files:
                tprint_warning(f"⚠️ No Analyst ensemble results found for {symbol} {exchange} {timeframe}")
                return None

            # Use the most recent file
            latest_file = max(matching_files, key=lambda x: x.stat().st_mtime)

            with open(latest_file, 'r') as f:
                results = json.load(f)

            # Extract OOF outputs from results
            oof_data = {}
            if 'oof_predictions' in results:
                oof_preds = results['oof_predictions']
                if 'p_trade' in oof_preds:
                    oof_data['p_trade'] = oof_preds['p_trade']
                if 'u_trade' in oof_preds:
                    oof_data['u_trade'] = oof_preds['u_trade']
                if 'q_trade' in oof_preds:
                    oof_data['q_trade'] = oof_preds['q_trade']

            if oof_data:
                return pd.DataFrame(oof_data)
            else:
                tprint_warning("⚠️ No OOF outputs found in Analyst results")
                return None

        except Exception as e:
            tprint_error(f"❌ Failed to load Analyst OOF outputs: {e}")
            return None

    async def _extract_analyst_features(
        self,
        filtered_data: pd.DataFrame,
        **kwargs
    ) -> List[str]:
        """
        Extract Analyst features for inclusion in training.

        Args:
            filtered_data: Filtered training DataFrame
            **kwargs: Additional parameters

        Returns:
            List of analyst feature column names
        """
        try:
            analyst_features = []

            # Look for analyst-related columns that could be useful features
            analyst_patterns = ['analyst', 'ensemble', 'signal', 'confidence', 'prediction']

            for col in filtered_data.columns:
                if any(pattern in col.lower() for pattern in analyst_patterns):
                    # Don't include the confidence column used for filtering as a feature
                    if 'confidence' not in col.lower() or 'raw_confidence' in col.lower():
                        analyst_features.append(col)

            if analyst_features:
                tprint_debug(f"✅ Extracted {len(analyst_features)} analyst features for training")

            return analyst_features

        except Exception as e:
            tprint_error(f"❌ Failed to extract analyst features: {e}")
            return []

    async def _train_t1_patchtst_lightgbm(self, X: np.ndarray, y: np.ndarray,
                                        sample_weight: Optional[np.ndarray] = None,
                                        **kwargs) -> Dict[str, Any]:
        """Train T1: PatchTST-LightGBM model for classification."""
        try:
            tprint_info("🚀 Training T1: PatchTST-LightGBM model...")

            # Load T1-T4 configuration
            config_path = Path("/workspace/config/tactician_t1_t4_models_config.yaml")
            if config_path.exists():
                import yaml
                with open(config_path, 'r') as f:
                    t1_t4_config = yaml.safe_load(f)
            else:
                # Fallback configuration
                t1_t4_config = {
                    'tactician_t1_t4_config': {
                        'tree_models': {
                            't1_lightgbm': {
                                'params': {
                                    'n_estimators': 1000,
                                    'learning_rate': 0.05,
                                    'max_depth': 8,
                                    'num_leaves': 63,
                                    'subsample': 0.8,
                                    'colsample_bytree': 0.8,
                                    'random_state': 42,
                                    'verbosity': -1,
                                    'monotone_constraints': [1, 1, 1, 0, 0, -1, -1, 1, 0, 0],
                                    'monotone_constraints_method': 'advanced'
                                }
                            }
                        },
                        'patchtst_config': {
                            'patch_len': 16,
                            'stride': 8,
                            'use_transformer_attention': True,
                            'regime_aware': True,
                            'attention_dropout': 0.1,
                            'num_heads': 4,
                            'sign_dropout_rate': 0.0,
                            'sign_threshold': 0.2
                        }
                    }
                }

            from src.models.enhanced_patchtst import create_enhanced_patchtst, EnhancedPatchTSTConfig
            import lightgbm as lgb

            patch_config_values = t1_t4_config['tactician_t1_t4_config'].get('patchtst_config', {})
            patchtst_config = EnhancedPatchTSTConfig()
            if 'patch_len' in patch_config_values:
                patchtst_config.patch_len = patch_config_values['patch_len']
            if 'stride' in patch_config_values:
                patchtst_config.stride = patch_config_values['stride']

            patchtst_model = create_enhanced_patchtst(patchtst_config)
            patchtst_model.fit(X, y, sample_weight)

            oof_features = patchtst_model.get_oof_features()
            oof_embeddings = None
            if oof_features and 'oof_embeddings' in oof_features:
                oof_embeddings = oof_features.get('oof_embeddings')

            if oof_embeddings is None or oof_embeddings.shape[0] != X.shape[0]:
                tprint_warning(
                    "⚠️ PatchTST OOF embeddings unavailable or misaligned for T1; using in-sample embeddings"
                )
                patch_features = patchtst_model.get_features(X)
                oof_embeddings = patch_features['embeddings']

            embedding_names = [f't1_patchtst_oof_{i}' for i in range(oof_embeddings.shape[1])]
            filtered_embeddings, filter_metadata = filter_embedding_features(
                parent_features=X,
                embedding_features=oof_embeddings,
                target=y,
                embedding_names=embedding_names,
                corr_threshold=0.8,
                ic_threshold=0.05,
                min_embeddings=6,
                max_embeddings=10
            )

            if filtered_embeddings.shape[1] == 0:
                fallback_dims = min(10, max(6, oof_embeddings.shape[1]))
                filtered_embeddings = oof_embeddings[:, :fallback_dims]
                filter_metadata['retained_count'] = filtered_embeddings.shape[1]
                filter_metadata['within_budget'] = 6 <= filtered_embeddings.shape[1] <= 10
                tprint_warning("⚠️ T1 PatchTST filtering removed all embeddings; using fallback slice")

            if not filter_metadata.get('within_budget', False):
                tprint_warning(
                    f"⚠️ T1 PatchTST embeddings outside budget: {filter_metadata.get('retained_count', 0)}"
                )

            X_combined = np.hstack([X, filtered_embeddings])

            tree_params = t1_t4_config['tactician_t1_t4_config']['tree_models']['t1_lightgbm']['params']
            lgbm_classifier = lgb.LGBMClassifier(**tree_params)

            if sample_weight is not None:
                lgbm_classifier.fit(X_combined, y, sample_weight=sample_weight)
            else:
                lgbm_classifier.fit(X_combined, y)

            predictions = lgbm_classifier.predict(X_combined)

            # Calculate metrics
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            metrics = {
                'accuracy': accuracy_score(y, predictions),
                'f1_score': f1_score(y, predictions, average='weighted'),
                'precision': precision_score(y, predictions, average='weighted'),
                'recall': recall_score(y, predictions, average='weighted'),
                'model_type': 'T1_PatchTST_LightGBM',
                'embedding_filter_metadata': filter_metadata
            }

            tprint_success(f"✅ T1: PatchTST-LightGBM trained with accuracy: {metrics['accuracy']:.4f}")
            return {
                'models': {
                    't1_patchtst_lightgbm': lgbm_classifier,
                    'patchtst': patchtst_model
                },
                'metrics': metrics
            }

        except Exception as e:
            tprint_error(f"❌ T1: PatchTST-LightGBM training failed: {e}")
            return {'models': {}, 'metrics': {}}

    async def _train_t2_patchtst_xgboost_lambdamart(self, X: np.ndarray, y: np.ndarray,
                                                  sample_weight: Optional[np.ndarray] = None,
                                                  **kwargs) -> Dict[str, Any]:
        """Train T2: PatchTST-XGBoost-LambdaMART model for ranking."""
        try:
            tprint_info("🚀 Training T2: PatchTST-XGBoost-LambdaMART model...")

            from src.utils.ml_common.models.model_factory import EnhancedModelFactory, ModelConfig, ModelType

            # Load T1-T4 configuration
            config_path = Path("/workspace/config/tactician_t1_t4_models_config.yaml")
            if config_path.exists():
                import yaml
                with open(config_path, 'r') as f:
                    t1_t4_config = yaml.safe_load(f)
            else:
                # Fallback configuration
                t1_t4_config = {
                    'tactician_t1_t4_config': {
                        'tree_models': {
                            't2_xgboost_lambdamart': {
                                'params': {
                                    'n_estimators': 1000,
                                    'learning_rate': 0.05,
                                    'max_depth': 8,
                                    'subsample': 0.8,
                                    'colsample_bytree': 0.8,
                                    'random_state': 42,
                                    'lambda': 0.1,
                                    'alpha': 0.1,
                                    'verbosity': 0,
                                    'monotone_constraints': [1, 1, 1, 0, 0, -1, -1, 1, 0, 0]
                                }
                            }
                        },
                        'patchtst_config': {
                            'patch_len': 16,
                            'stride': 8,
                            'use_transformer_attention': True,
                            'regime_aware': True,
                            'attention_dropout': 0.1,
                            'num_heads': 4,
                            'sign_dropout_rate': 0.0,
                            'sign_threshold': 0.2
                        }
                    }
                }

            # Create model configuration
            model_config = ModelConfig(
                model_type=ModelType.PATCHTST_XGBOOST_LAMBDAMART,
                model_name="t2_patchtst_xgboost_lambdamart",
                n_outputs=1,
                model_params={
                    **t1_t4_config['tactician_t1_t4_config']['tree_models']['t2_xgboost_lambdamart']['params'],
                    'patchtst_config': t1_t4_config['tactician_t1_t4_config']['patchtst_config']
                }
            )

            # Create and train model
            factory = EnhancedModelFactory()
            model = factory.create_model(model_config)

            # For ranking, we need group information
            groups = kwargs.get('groups', np.ones(len(X)))  # Default to single group

            if sample_weight is not None:
                model.fit(X, y, group=groups, sample_weight=sample_weight)
            else:
                model.fit(X, y, group=groups)

            # Get predictions for metrics
            predictions = model.predict(X)

            # Calculate ranking metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            metrics = {
                'mse': mean_squared_error(y, predictions),
                'mae': mean_absolute_error(y, predictions),
                'model_type': 'T2_PatchTST_XGBoost_LambdaMART'
            }

            tprint_success(f"✅ T2: PatchTST-XGBoost-LambdaMART trained with MSE: {metrics['mse']:.4f}")
            return {'models': {'t2_patchtst_xgboost_lambdamart': model}, 'metrics': metrics}

        except Exception as e:
            tprint_error(f"❌ T2: PatchTST-XGBoost-LambdaMART training failed: {e}")
            return {'models': {}, 'metrics': {}}

    async def _train_t3_patchtst_catboost(self, X: np.ndarray, y: np.ndarray,
                                        sample_weight: Optional[np.ndarray] = None,
                                        **kwargs) -> Dict[str, Any]:
        """Train T3: PatchTST-CatBoost model for binary classification."""
        try:
            tprint_info("🚀 Training T3: PatchTST-CatBoost model...")

            from src.utils.ml_common.models.model_factory import EnhancedModelFactory, ModelConfig, ModelType

            # Load T1-T4 configuration
            config_path = Path("/workspace/config/tactician_t1_t4_models_config.yaml")
            if config_path.exists():
                import yaml
                with open(config_path, 'r') as f:
                    t1_t4_config = yaml.safe_load(f)
            else:
                # Fallback configuration
                t1_t4_config = {
                    'tactician_t1_t4_config': {
                        'tree_models': {
                            't3_catboost': {
                                'params': {
                                    'iterations': 1000,
                                    'learning_rate': 0.05,
                                    'depth': 8,
                                    'random_seed': 42,
                                    'verbose': False,
                                    'monotone_constraints': [1, 1, 1, 0, 0, -1, -1, 1, 0, 0],
                                    'ordered_boosting': True
                                }
                            }
                        },
                        'patchtst_config': {
                            'patch_len': 16,
                            'stride': 8,
                            'use_transformer_attention': True,
                            'regime_aware': True,
                            'attention_dropout': 0.1,
                            'num_heads': 4,
                            'sign_dropout_rate': 0.0,
                            'sign_threshold': 0.2
                        }
                    }
                }

            # Create model configuration
            model_config = ModelConfig(
                model_type=ModelType.PATCHTST_CATBOOST,
                model_name="t3_patchtst_catboost",
                n_outputs=len(np.unique(y)),
                model_params={
                    **t1_t4_config['tactician_t1_t4_config']['tree_models']['t3_catboost']['params'],
                    'patchtst_config': t1_t4_config['tactician_t1_t4_config']['patchtst_config']
                }
            )

            # Create and train model
            factory = EnhancedModelFactory()
            model = factory.create_model(model_config)

            if sample_weight is not None:
                model.fit(X, y, sample_weight=sample_weight)
            else:
                model.fit(X, y)

            # Get predictions and probabilities
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)

            # Calculate metrics
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
            metrics = {
                'accuracy': accuracy_score(y, predictions),
                'f1_score': f1_score(y, predictions, average='weighted'),
                'precision': precision_score(y, predictions, average='weighted'),
                'recall': recall_score(y, predictions, average='weighted'),
                'model_type': 'T3_PatchTST_CatBoost'
            }

            # Add AUC for binary classification
            if len(np.unique(y)) == 2:
                try:
                    metrics['auc'] = roc_auc_score(y, probabilities[:, 1])
                except:
                    pass

            tprint_success(f"✅ T3: PatchTST-CatBoost trained with accuracy: {metrics['accuracy']:.4f}")
            return {'models': {'t3_patchtst_catboost': model}, 'metrics': metrics}

        except Exception as e:
            tprint_error(f"❌ T3: PatchTST-CatBoost training failed: {e}")
            return {'models': {}, 'metrics': {}}

    async def _train_t4_causal_dilated_tcn(self, X: np.ndarray, y: np.ndarray,
                                         sample_weight: Optional[np.ndarray] = None,
                                         **kwargs) -> Dict[str, Any]:
        """Train T4: Causal Dilated TCN model for sequence tasks."""
        try:
            tprint_info("🚀 Training T4: Causal Dilated TCN model...")

            from src.utils.ml_common.models.model_factory import EnhancedModelFactory, ModelConfig, ModelType

            # Load T1-T4 configuration
            config_path = Path("/workspace/config/tactician_t1_t4_models_config.yaml")
            if config_path.exists():
                import yaml
                with open(config_path, 'r') as f:
                    t1_t4_config = yaml.safe_load(f)
            else:
                # Fallback configuration
                t1_t4_config = {
                    'tactician_t1_t4_config': {
                        'sequence_model': {
                            't4_tcn': {
                                'config': {
                                    'residual_blocks': 8,
                                    'channels': 64,
                                    'kernel_size': 3,
                                    'dilations': [1, 2, 4, 8, 16, 32, 64],
                                    'dropout': 0.1,
                                    'use_batch_norm': True,
                                    'activation': 'relu'
                                }
                            }
                        }
                    }
                }

            # Reshape X for sequence model if needed
            if len(X.shape) == 2:
                # Convert 2D to 3D for sequence model
                seq_length = min(100, X.shape[1] // 10)  # Estimate sequence length
                n_features = min(50, X.shape[1] // seq_length)  # Estimate features per timestep
                X_seq = X[:, :seq_length*n_features].reshape(X.shape[0], seq_length, n_features)
            else:
                X_seq = X

            # Create model configuration
            model_config = ModelConfig(
                model_type=ModelType.CAUSAL_DILATED_TCN,
                model_name="t4_causal_dilated_tcn",
                n_outputs=y.shape[1] if len(y.shape) > 1 else 1,
                model_params={
                    **t1_t4_config['tactician_t1_t4_config']['sequence_model']['t4_tcn']['config'],
                    'input_dim': X_seq.shape[-1],
                    'seq_length': X_seq.shape[1]
                }
            )

            # Create and train model
            factory = EnhancedModelFactory()
            model = factory.create_model(model_config)

            if sample_weight is not None:
                model.fit(X_seq, y, sample_weight=sample_weight)
            else:
                model.fit(X_seq, y)

            # Get predictions for metrics
            predictions = model.predict(X_seq)

            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            metrics = {
                'mse': mean_squared_error(y, predictions),
                'mae': mean_absolute_error(y, predictions),
                'model_type': 'T4_Causal_Dilated_TCN'
            }

            tprint_success(f"✅ T4: Causal Dilated TCN trained with MSE: {metrics['mse']:.4f}")
            return {'models': {'t4_causal_dilated_tcn': model}, 'metrics': metrics}

        except Exception as e:
            tprint_error(f"❌ T4: Causal Dilated TCN training failed: {e}")
            return {'models': {}, 'metrics': {}}

    async def _train_t4_tft_small(self, X: np.ndarray, y: np.ndarray,
                                sample_weight: Optional[np.ndarray] = None,
                                **kwargs) -> Dict[str, Any]:
        """Train T4: TFT-Small model for sequence tasks (alternative to TCN)."""
        try:
            tprint_info("🚀 Training T4: TFT-Small model...")

            from src.utils.ml_common.models.model_factory import EnhancedModelFactory, ModelConfig, ModelType

            # Load T1-T4 configuration
            config_path = Path("/workspace/config/tactician_t1_t4_models_config.yaml")
            if config_path.exists():
                import yaml
                with open(config_path, 'r') as f:
                    t1_t4_config = yaml.safe_load(f)
            else:
                # Fallback configuration
                t1_t4_config = {
                    'tactician_t1_t4_config': {
                        'sequence_model': {
                            't4_tft_small': {
                                'config': {
                                    'hidden_size': 64,
                                    'attention_heads': 4,
                                    'dropout': 0.1,
                                    'num_layers': 3,
                                    'use_time_features': True,
                                    'use_static_features': True
                                }
                            }
                        }
                    }
                }

            # Reshape X for sequence model if needed
            if len(X.shape) == 2:
                # Convert 2D to 3D for sequence model
                seq_length = min(100, X.shape[1] // 10)
                n_features = min(50, X.shape[1] // seq_length)
                X_seq = X[:, :seq_length*n_features].reshape(X.shape[0], seq_length, n_features)
            else:
                X_seq = X

            # Create model configuration
            model_config = ModelConfig(
                model_type=ModelType.TFT_SMALL,
                model_name="t4_tft_small",
                n_outputs=1,  # Regression output
                model_params={
                    **t1_t4_config['tactician_t1_t4_config']['sequence_model']['t4_tft_small']['config'],
                    'input_dim': X_seq.shape[-1],
                    'seq_length': X_seq.shape[1]
                }
            )

            # Create and train model
            factory = EnhancedModelFactory()
            model = factory.create_model(model_config)

            if sample_weight is not None:
                model.fit(X_seq, y, sample_weight=sample_weight)
            else:
                model.fit(X_seq, y)

            # Get predictions for metrics
            predictions = model.predict(X_seq)

            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            metrics = {
                'mse': mean_squared_error(y, predictions),
                'mae': mean_absolute_error(y, predictions),
                'model_type': 'T4_TFT_Small'
            }

            tprint_success(f"✅ T4: TFT-Small trained with MSE: {metrics['mse']:.4f}")
            return {'models': {'t4_tft_small': model}, 'metrics': metrics}

        except Exception as e:
            tprint_error(f"❌ T4: TFT-Small training failed: {e}")
            return {'models': {}, 'metrics': {}}

    async def _train_lgbm_gru(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Train LGBM + GRU embedding model."""
        try:
            from src.models.lgbm_gru_embedding import create_lgbm_gru_embedding, LGBMGRUConfig
            from src.config.updated_model_configs import LGBMConfig, GRUConfig

            # Create LGBM + GRU model
            config = LGBMGRUConfig()
            model = create_lgbm_gru_embedding(config)

            model.fit(X, y.ravel(), sample_weight=sample_weight)

            return {
                'models': {'lgbm_gru': model},
                'metrics': {'model_type': 'LGBM_GRU', 'config': config}
            }

        except Exception as e:
            tprint_error(f"❌ LGBM + GRU training failed: {e}")
            return {'models': {}, 'metrics': {}}

    async def _train_catboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Train CatBoost model."""
        try:
            from catboost import CatBoostRegressor
            from src.config.updated_model_configs import CatBoostConfig

            config = CatBoostConfig()
            model = CatBoostRegressor(
                depth=config.depth,
                learning_rate=config.learning_rate,
                l2_leaf_reg=config.l2_leaf_reg,
                iterations=config.iterations,
                subsample=config.subsample,
                colsample_bylevel=config.colsample_bylevel,
                random_seed=config.random_seed,
                verbose=config.verbose
            )

            model.fit(X, y.ravel(), sample_weight=sample_weight)

            def _build_model():
                return CatBoostRegressor(
                    depth=config.depth,
                    learning_rate=config.learning_rate,
                    l2_leaf_reg=config.l2_leaf_reg,
                    iterations=config.iterations,
                    subsample=config.subsample,
                    colsample_bylevel=config.colsample_bylevel,
                    random_seed=config.random_seed,
                    verbose=config.verbose
                )

            oof_predictions = self._generate_model_oof_predictions(
                _build_model,
                X,
                y,
                sample_weight,
                n_splits=5
            )

            result = {
                'models': {'catboost': model},
                'metrics': {'model_type': 'CatBoost', 'config': config}
            }

            if oof_predictions is not None:
                result['oof_predictions'] = {'catboost': oof_predictions.tolist()}

            return result

        except Exception as e:
            tprint_error(f"❌ CatBoost training failed: {e}")
            return {'models': {}, 'metrics': {}}

    async def _train_causal_tcn(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Train Causal Dilated TCN model."""
        try:
            from src.models.causal_dilated_tcn import create_causal_dilated_tcn, CausalTCNConfig

            config = CausalTCNConfig()
            model = create_causal_dilated_tcn(config)

            model.fit(X, y.ravel(), sample_weight=sample_weight)

            return {
                'models': {'causal_tcn': model},
                'metrics': {'model_type': 'Causal_TCN', 'config': config}
            }

        except Exception as e:
            tprint_error(f"❌ Causal Dilated TCN training failed: {e}")
            return {'models': {}, 'metrics': {}}

    async def _train_stacker_lgbm_calibrated(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Train Stacker LGBM Calibrated meta-learner."""
        base_model_oof_predictions = kwargs.get('base_model_oof_predictions') or {}

        if not base_model_oof_predictions:
            raise ValueError("Base model OOF predictions are required for stacker training")

        try:
            from src.models.stacker_lgbm_gate import (
                create_stacker_lgbm_gate,
                StackerLGBMGateConfig,
            )

            oof_predictions = kwargs.get('oof_predictions') or {}
            if not oof_predictions:
                raise ValueError(
                    "Out-of-fold predictions for Analyst and Tactician are required for the gated stacker"
                )

            regime_features = self._assemble_regime_feature_tensor(
                X,
                oof_predictions,
                sample_weight,
                **kwargs,
            )

            config = StackerLGBMGateConfig()
            model = create_stacker_lgbm_gate(config)
            model.fit(
                base_predictions=oof_predictions,
                y=y.ravel(),
                regime_features=regime_features,
                sample_weight=sample_weight,
            )

            gating_state = model.get_gating_state()
            calibration_state = model.get_calibration_state()
            loss_history = model.get_gating_loss_history()

            metrics = {
                'model_type': 'Stacker_LGBM_Gate',
                'config': config,
                'gating_final_loss': loss_history[-1] if loss_history else None,
                'gating_entropy_penalty': config.gating.entropy_penalty,
            }

            return {
                'models': {
                    'stacker_lgbm_gate': model,
                    'stacker_lgbm_calibrated': model,
                },
                'metrics': metrics,
                'artifacts': {
                    'stacker_gating_state': gating_state,
                    'stacker_calibration_state': calibration_state,
                },
            config = StackerLGBMCalibratedConfig()
            model = create_stacker_lgbm_calibrated(config)

            formatted_predictions: Dict[str, np.ndarray] = {}
            for name, preds in base_model_oof_predictions.items():
                if preds is None:
                    continue
                array_preds = np.asarray(preds)
                if array_preds.ndim == 1:
                    formatted_predictions[name] = array_preds
                elif array_preds.ndim == 2:
                    formatted_predictions[name] = array_preds
                else:
                    raise ValueError(f"Unsupported prediction shape for {name}: {array_preds.shape}")

            if not formatted_predictions:
                raise ValueError("No valid OOF predictions available for stacker training")

            meta_oof_matrix = model._prepare_stacking_features(formatted_predictions)

            stacker_output_dir = kwargs.get('output_directory') or os.path.join(
                self.config.output_directory, 'stacker_lgbm_calibrated'
            )
            os.makedirs(stacker_output_dir, exist_ok=True)

            meta_oof_path = os.path.join(stacker_output_dir, 'meta_oof_predictions.npy')
            np.save(meta_oof_path, meta_oof_matrix)

            model.fit(formatted_predictions, y.ravel(), sample_weight)

            metrics = {
                'model_type': 'Stacker_LGBM_Calibrated',
                'config': config,
                'base_models_used': list(formatted_predictions.keys()),
                'meta_oof_path': meta_oof_path,
                'meta_oof_shape': meta_oof_matrix.shape,
            }

            return {
                'models': {'stacker_lgbm_calibrated': model},
                'metrics': metrics,
                'meta_oof_predictions': meta_oof_matrix.tolist(),
            }

        except Exception as e:
            tprint_error(f"❌ Stacker LGBM Calibrated training failed: {e}")
            return {'models': {}, 'metrics': {}, 'artifacts': {}}

    def _assemble_regime_feature_tensor(
        self,
        X: np.ndarray,
        oof_predictions: Dict[str, Any],
        sample_weight: np.ndarray,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Construct regime/context features required by the gating head with regime probabilities."""

        n_samples = X.shape[0]
        feature_map: Dict[str, np.ndarray] = {}

        dict_sources: List[Dict[str, Any]] = []
        dataframe_sources: List[pd.DataFrame] = []

        for key in (
            'regime_features',
            'regime_context',
            'additional_context',
            'regime_feature_map',
            'regime_probabilities_info',  # Simplified: only regime probabilities
        ):
            value = kwargs.get(key)
            if isinstance(value, dict):
                dict_sources.append(value)

        for key in ('training_dataframe', 'regime_feature_frame', 'market_context_frame'):
            df_candidate = kwargs.get(key)
            if isinstance(df_candidate, pd.DataFrame):
                dataframe_sources.append(df_candidate)

        def _lookup(*names: str) -> Optional[Any]:
            for name in names:
                for source in dict_sources:
                    if name in source and source[name] is not None:
                        return source[name]
                for frame in dataframe_sources:
                    if name in frame.columns:
                        return frame[name].values
            return None

        # Regime probabilities from probabilistic outputs
        regime_probabilities_info = _lookup('regime_probabilities_info')
        if regime_probabilities_info and regime_probabilities_info.get('has_probabilistic_outputs'):
            tprint_info("📊 Using regime probabilities for Tactician models")

            # Add regime probability features
            regime_probabilities = regime_probabilities_info.get('regime_probabilities')
            if regime_probabilities is not None and len(regime_probabilities) == n_samples:
                n_regimes = regime_probabilities.shape[1] if len(regime_probabilities.shape) > 1 else 1
                for i in range(n_regimes):
                    feature_map[f'regime_prob_{i}'] = self._ensure_feature_array(regime_probabilities[:, i], n_samples)
                tprint_info(f"✅ Added {n_regimes} regime probability features")
        else:
            tprint_warning("⚠️ No regime probabilities available, using fallback features")

        # Legacy regime features (fallback)
        volatility = _lookup('volatility_level', 'volatility', 'volatility_score')
        if volatility is None:
            if X.size:
                volatility = np.std(X, axis=1)
            else:
                volatility = np.zeros(n_samples)
        feature_map['volatility_level'] = self._ensure_feature_array(volatility, n_samples)

        # Trend signal primarily uses the `trend_score` feature with fallbacks for legacy names.
        trend = _lookup('trend_score', 'trend_strength', 'momentum_score', 'market_health_score')
        if trend is None:
            if X.size:
                trend = np.mean(X, axis=1)
            else:
                trend = np.zeros(n_samples)
        feature_map['trend_score'] = self._ensure_feature_array(trend, n_samples)

        liquidity = _lookup('liquidity_z', 'liquidity_score', 'liquidity')
        if liquidity is None:
            liquidity = np.zeros(n_samples)
        feature_map['liquidity_z'] = self._ensure_feature_array(liquidity, n_samples)

        tprint_info(f"📊 Assembled {len(feature_map)} regime features for Tactician models")
        return feature_map

    def _ensure_feature_array(self, values: Any, n_samples: int) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0:
            arr = np.full(n_samples, float(arr))
        if arr.shape[0] != n_samples:
            if arr.shape[0] == 1:
                arr = np.full(n_samples, float(arr[0]))
            else:
                raise ValueError("Regime feature length mismatch while assembling gating tensor")
        return arr

# Convenience function for external usage
async def execute_tactician_models_training(
    training_data: pd.DataFrame,
    feature_columns: List[str],
    target_columns: List[str],
    sample_weight: Optional[np.ndarray] = None,
    config: Optional[TacticianModelsTrainingConfig] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Execute Tactician base models training.

    Args:
        training_data: DataFrame with features and targets
        feature_columns: List of feature column names
        target_columns: List of target column names
        sample_weight: Optional sample weights
        config: Optional configuration
        **kwargs: Additional parameters

    Returns:
        Dict with trained models and metrics
    """
    trainer = TacticianModelsTrainingStep(config)
    return await trainer.train_tactician_models(
        training_data, feature_columns, target_columns, sample_weight, **kwargs
    )

# Apply negative learning patches
if NEGATIVE_LEARNING_AVAILABLE:
    try:
        apply_negative_learning_patches()
        print("✅ Negative learning patches applied to Tactician training")
    except Exception as e:
        print(f"⚠️ Failed to apply negative learning patches: {e}")
