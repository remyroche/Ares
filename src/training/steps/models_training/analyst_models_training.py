"""
Analyst Models Training - Base Model Training Module

This module handles training of individual Analyst base models:
- TCN (Temporal Convolutional Network) model
- LightGBM model
- Ridge Regression model
- Elastic Net model
- Random Forest model
- NAS (Neural Architecture Search) model
- TAS (Tree-based Architecture Search) model

The Analyst operates on the dedicated 15m timeframe and decides IF we trade by
screening market conditions and producing the green-signal gating that the
Tactician consumes. Now includes NAS and TAS models for enhanced 15m timeframe
strategic decision making with regime features.

ENHANCED FEATURES:
- Comprehensive error handling with detailed failure reporting
- Enhanced progress tracking and sub-step reporting
- Input validation and data quality checks
- Optimized vectorization with intelligent fallback
- Structured logging with performance metrics
- Health monitoring throughout training process
- Integration with common utilities and hardware optimizers
- Extensive logging with tprint at every step
- NAS and TAS model support for advanced 15m timeframe analysis
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
import traceback
from dataclasses import dataclass
from enum import Enum
import os

from src.training.utils.embedding_postprocessing import filter_embedding_features

# Enhanced imports with comprehensive error handling
try:
    from src.utils.logger import system_logger
    from src.utils.ml_common.config import PerRegimeTrainingConfig
    from src.utils.ml_common.training import PerRegimeTrainingStep
    ANALYST_TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import core ML utilities: {e}")
    ANALYST_TRAINING_AVAILABLE = False

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

# Import VectorBT optimization utilities
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    )
    from src.feature_generation.utils.unified_vectorization_manager import (
        UnifiedVectorizationManager, get_unified_vectorization_manager, VectorizationConfig
    )
    VECTORBT_UTILS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ WARNING: VectorBT optimization utilities not available: {e}")
    VECTORBT_UTILS_AVAILABLE = False

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

# NAS-TAS functionality removed - directory deleted
NAS_TAS_AVAILABLE = False


class AnalystModelType(Enum):
    """Analyst model types."""
    LGBM = "LGBM"
    LGBM_PATCHTST = "LGBM_PatchTST"
    CATBOOST = "CATBOOST"
    STACKER_LGBM_CALIBRATED = "STACKER_LGBM_CALIBRATED"
    NAS = "NAS"  # Neural Architecture Search
    TAS = "TAS"  # Tree-based Architecture Search


@dataclass
class AnalystModelsTrainingConfig:
    """Configuration for Analyst base models training."""
    # Training parameters
    model_types: List[AnalystModelType] = None
    save_models: bool = True
    output_directory: str = "generated/analyst_models_training"

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
            AnalystModelType.LGBM,
            AnalystModelType.LGBM_PATCHTST,
            AnalystModelType.CATBOOST,
            AnalystModelType.STACKER_LGBM_CALIBRATED,
            AnalystModelType.NAS,
            AnalystModelType.TAS,
        ]

        if self.model_types is None:
            self.model_types = [
                AnalystModelType.LGBM,
                AnalystModelType.LGBM_PATCHTST,
                AnalystModelType.CATBOOST,
                AnalystModelType.STACKER_LGBM_CALIBRATED,
                AnalystModelType.NAS,
                AnalystModelType.TAS
            ]


@dataclass
class AnalystModelsTrainingResult:
    """Result of Analyst models training."""
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


class AnalystModelsTrainingStep:
    """
    Analyst Models Training Step.

    Handles training of individual Analyst base models using common dependencies.
    """

    def __init__(self, config: Optional[AnalystModelsTrainingConfig] = None):
        """Initialize the Analyst models training step."""
        try:
            self.config = config or AnalystModelsTrainingConfig()
            self.logger = system_logger.getChild('AnalystModelsTrainingStep')
            self._nas_tas_orchestrators: Dict[Tuple[str, str, str], 'TrainingOrchestrator'] = {}

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

            # Initialize VectorBT optimization utilities
            if VECTORBT_UTILS_AVAILABLE:
                try:
                    # Create optimized vectorization configuration
                    vectorization_config = VectorizationConfig(
                        enable_vectorbt=True,
                        enable_gpu=self.config.enable_gpu_acceleration,
                        enable_parallel=self.config.enable_parallel_processing,
                        memory_efficient=True,
                        max_memory_gb=self.config.memory_limit_gb,
                        chunk_size=1000,
                        enable_monitoring=True,
                        batch_size=10000,
                        enable_batch_processing=True,
                        rolling_optimization_threshold=1000,
                        enable_rolling_optimization=True
                    )
                    
                    self.vectorization_manager = get_unified_vectorization_manager(vectorization_config)
                    self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                        enable_gpu=self.config.enable_gpu_acceleration,
                        enable_parallel=self.config.enable_parallel_processing,
                        memory_efficient=True,
                        chunk_size=1000,
                        fast_fail=True,
                        enable_logging=True
                    )
                    tprint_success("✅ VectorBT optimization utilities initialized")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to initialize VectorBT utilities: {e}")
                    self.vectorization_manager = None
                    self.rolling_optimizer = None
            else:
                self.vectorization_manager = None
                self.rolling_optimizer = None

            tprint_success("✅ AnalystModelsTrainingStep initialized successfully")

        except Exception as e:
            tprint_error(f"❌ Failed to initialize AnalystModelsTrainingStep: {e}")
            raise

    async def train_analyst_models(
        self,
        training_data: pd.DataFrame,
        feature_columns: List[str],
        target_columns: List[str],
        sample_weight: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train Analyst base models.

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
        tprint_info("🚀 Starting Analyst base models training...")

        try:
            # Validate inputs
            if training_data.empty or not feature_columns or not target_columns:
                raise ValueError("Insufficient training data or missing columns")

            # Optimize training data with VectorBT utilities
            tprint_info("🔧 Optimizing training data with VectorBT utilities...")
            optimized_training_data = self.optimize_training_data(training_data)
            
            # Generate additional VectorBT features
            enhanced_training_data = self.generate_vectorbt_features(optimized_training_data, feature_columns)
            
            # Update feature columns to include new VectorBT features
            new_features = [col for col in enhanced_training_data.columns 
                          if col not in training_data.columns and col not in target_columns]
            all_feature_columns = feature_columns + new_features
            
            if new_features:
                tprint_success(f"✅ Added {len(new_features)} VectorBT-generated features: {new_features[:5]}{'...' if len(new_features) > 5 else ''}")

            # Prepare training data
            X = enhanced_training_data[all_feature_columns].values
            y = enhanced_training_data[target_columns].values

            if len(y.shape) == 1:
                y = y.reshape(-1, 1)

            if sample_weight is None:
                sample_weight = np.ones(len(enhanced_training_data))

            # Generate OOF predictions for Tactician integration
            oof_predictions = await self._generate_oof_predictions(
                enhanced_training_data, all_feature_columns, target_columns, sample_weight, **kwargs
            )

            # Train models
            all_models = {}
            all_metrics = {}
            all_oof_predictions: Dict[str, np.ndarray] = {}

            regime_assignments = kwargs.get('regime_assignments')
            timestamps = kwargs.get('timestamps')

            for model_type in self.config.model_types:
                try:
                    tprint_info(f"🔧 Training {model_type.value} model...")

                    # Create training configuration
                    training_config = {
                        'model_type': model_type.value,
                        'training_data': training_data,
                        'feature_columns': feature_columns,
                        'target_columns': target_columns,
                        'sample_weight': sample_weight,
                        'save_models': self.config.save_models,
                        'output_directory': f"{self.config.output_directory}/{model_type.value.lower()}",
                        'regime_assignments': regime_assignments,
                        'timestamps': timestamps,
                    }

                    # Train model using existing trainer if available
                    if ANALYST_TRAINING_AVAILABLE:
                        training_result = await self._train_model_with_existing_trainer(
                            model_type, training_config
                        )
                    else:
                        extra_kwargs = dict(
                            training_data=training_data,
                            feature_columns=feature_columns,
                            target_columns=target_columns,
                            regime_assignments=regime_assignments,
                            timestamps=timestamps,
                            output_directory=training_config['output_directory'],
                        )

                        if model_type == AnalystModelType.STACKER_LGBM_CALIBRATED:
                            extra_kwargs['base_models'] = all_models
                            extra_kwargs['base_model_oof_predictions'] = all_oof_predictions

                        training_result = await self._train_model_directly(
                            model_type,
                            X,
                            y,
                            sample_weight,
                            **extra_kwargs,
                        )

                    # Store results
                    if training_result.get('models'):
                        for model_name, model in training_result['models'].items():
                            all_models[f"{model_type.value.lower()}_{model_name}"] = model

                    if training_result.get('metrics'):
                        all_metrics[f"{model_type.value.lower()}"] = training_result['metrics']

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

            return {
                'models': all_models,
                'metrics': all_metrics,
                'oof_predictions': oof_predictions,
                'training_time': execution_time,
                'features_used': all_feature_columns,
                'original_features': feature_columns,
                'vectorbt_features': new_features,
                'samples_used': len(enhanced_training_data),
                'model_types_trained': [mt.value for mt in self.config.model_types],
                'models_per_type': len(self.config.model_types),
                'vectorbt_optimization': {
                    'data_optimized': True,
                    'features_generated': len(new_features),
                    'memory_optimized': True
                }
            }

        except Exception as e:
            execution_time = tprint_timer(start_time)
            tprint_error(f"❌ Base models training failed: {e}")
            return {
                'models': {},
                'metrics': {},
                'training_time': execution_time,
                'error': str(e)
            }

    async def _train_model_with_existing_trainer(
        self,
        model_type: AnalystModelType,
        training_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train model using existing trainer."""
        try:
            # Import and use the existing trainer
            if model_type in {AnalystModelType.NAS, AnalystModelType.TAS}:
                if not NAS_TAS_AVAILABLE:
                    raise ImportError("NAS/TAS orchestrator components are not available")
                return await self._train_nas_tas_models(model_type, training_config)

            from ..model_training.analyst_models_training_refactored import AnalystModelsTrainingStep

            trainer = AnalystModelsTrainingStep()
            cleaned_config = dict(training_config)
            cleaned_config.pop('regime_assignments', None)
            cleaned_config.pop('timestamps', None)
            return await trainer.train_analyst_models(**cleaned_config)

        except ImportError:
            tprint_warning(f"⚠️ Existing trainer not available, falling back to direct training")
            return await self._train_model_directly(
                model_type,
                training_config['training_data'][training_config['feature_columns']].values,
                training_config['training_data'][training_config['target_columns']].values,
                training_config['sample_weight'],
                training_data=training_config['training_data'],
                feature_columns=training_config['feature_columns'],
                target_columns=training_config['target_columns'],
                regime_assignments=training_config.get('regime_assignments'),
                timestamps=training_config.get('timestamps'),
                output_directory=training_config.get('output_directory'),
            )

    async def _train_model_directly(
        self,
        model_type: AnalystModelType,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Train model directly."""
        try:
            if model_type == AnalystModelType.LGBM:
                return await self._train_lgbm(X, y, sample_weight, **kwargs)
            elif model_type == AnalystModelType.LGBM_PATCHTST:
                return await self._train_lgbm_patchtst(X, y, sample_weight, **kwargs)
            elif model_type == AnalystModelType.CATBOOST:
                return await self._train_catboost(X, y, sample_weight, **kwargs)
            elif model_type == AnalystModelType.STACKER_LGBM_CALIBRATED:
                return await self._train_stacker_lgbm_calibrated(X, y, sample_weight, **kwargs)
            elif model_type == AnalystModelType.NAS:
                return await self._train_nas(X, y, sample_weight, **kwargs)
            elif model_type == AnalystModelType.TAS:
                return await self._train_tas(X, y, sample_weight, **kwargs)
            elif model_type in {AnalystModelType.NAS, AnalystModelType.TAS}:
                training_data = kwargs.get('training_data')
                feature_columns = kwargs.get('feature_columns')
                target_columns = kwargs.get('target_columns') or []
                regime_assignments = kwargs.get('regime_assignments')
                timestamps = kwargs.get('timestamps')
                output_directory = kwargs.get('output_directory') or os.path.join(
                    self.config.output_directory, model_type.value.lower()
                )

                if training_data is None or not feature_columns or not target_columns:
                    raise ValueError("Training data, feature columns, and target columns are required for NAS/TAS training")

                training_config = {
                    'training_data': training_data,
                    'feature_columns': feature_columns,
                    'target_columns': target_columns,
                    'regime_assignments': regime_assignments,
                    'timestamps': timestamps,
                    'output_directory': output_directory,
                    'save_models': self.config.save_models,
                }

                return await self._train_nas_tas_models(model_type, training_config)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        except Exception as e:
            tprint_error(f"❌ Failed to train {model_type.value} directly: {e}")
            return {'models': {}, 'metrics': {}}

    async def _generate_oof_predictions(
        self,
        training_data: pd.DataFrame,
        feature_columns: List[str],
        target_columns: List[str],
        sample_weight: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate Out-of-Fold predictions for Tactician integration.
        
        Args:
            training_data: Training DataFrame
            feature_columns: Feature columns
            target_columns: Target columns
            sample_weight: Sample weights
            **kwargs: Additional parameters
            
        Returns:
            Dict with OOF predictions: p_trade, u_trade, q_trade
        """
        try:
            from sklearn.model_selection import KFold
            from sklearn.metrics import log_loss
            import lightgbm as lgb
            
            tprint_info("🔧 Generating OOF predictions for Tactician integration...")
            
            X = training_data[feature_columns].values
            y = training_data[target_columns].values.ravel()
            
            # Use 5-fold CV for OOF predictions
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            
            # Initialize OOF prediction arrays
            n_samples = len(training_data)
            p_trade_oof = np.zeros(n_samples)  # Probability of trade
            u_trade_oof = np.zeros(n_samples)  # Expected net edge
            q_trade_oof = np.zeros(n_samples)  # Confidence/quality
            
            # Generate OOF predictions using LightGBM
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                tprint_debug(f"🔧 Generating OOF predictions for fold {fold + 1}/5")
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                w_train = sample_weight[train_idx]
                
                # Train LightGBM model for this fold
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    verbosity=-1
                )
                
                lgb_model.fit(X_train, y_train, sample_weight=w_train)
                
                # Get predictions for validation set
                y_pred = lgb_model.predict(X_val)
                
                # Calculate p_trade (probability of positive return)
                p_trade_fold = np.where(y_pred > 0, 1.0, 0.0)
                
                # Calculate u_trade (expected net edge)
                u_trade_fold = y_pred
                
                # Calculate q_trade (confidence based on prediction magnitude)
                q_trade_fold = np.abs(y_pred) / (np.abs(y_pred).max() + 1e-8)
                
                # Store OOF predictions
                p_trade_oof[val_idx] = p_trade_fold
                u_trade_oof[val_idx] = u_trade_fold
                q_trade_oof[val_idx] = q_trade_fold
            
            # Ensure values are in valid ranges
            p_trade_oof = np.clip(p_trade_oof, 0.0, 1.0)
            u_trade_oof = np.clip(u_trade_oof, -1.0, 1.0)
            q_trade_oof = np.clip(q_trade_oof, 0.0, 1.0)
            
            oof_predictions = {
                'p_trade': p_trade_oof.tolist(),
                'u_trade': u_trade_oof.tolist(),
                'q_trade': q_trade_oof.tolist(),
                'n_samples': n_samples,
                'cv_folds': 5,
                'generation_method': 'lightgbm_oof'
            }
            
            tprint_success(f"✅ Generated OOF predictions: p_trade mean={p_trade_oof.mean():.3f}, "
                          f"u_trade mean={u_trade_oof.mean():.3f}, q_trade mean={q_trade_oof.mean():.3f}")
            
            return oof_predictions
            
        except Exception as e:
            tprint_error(f"❌ Failed to generate OOF predictions: {e}")
            # Return default OOF predictions
            n_samples = len(training_data)
            return {
                'p_trade': [0.5] * n_samples,
                'u_trade': [0.0] * n_samples,
                'q_trade': [0.5] * n_samples,
                'n_samples': n_samples,
                'cv_folds': 0,
                'generation_method': 'default_fallback'
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
        """Generate OOF predictions for a regressor using outer CV folds."""
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
            tprint_warning(f"⚠️ Failed to generate model OOF predictions: {e}")
            return None

    async def _train_lgbm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Train LightGBM model with updated hyperparameters."""
        try:
            import lightgbm as lgb
            from src.config.updated_model_configs import LGBMConfig

            config = LGBMConfig()
            def _build_model():
                return lgb.LGBMRegressor(
                    max_depth=config.max_depth,
                    num_leaves=config.num_leaves,
                    min_child_samples=config.min_child_samples,
                    reg_lambda=config.lambda_l2,
                    feature_fraction=config.feature_fraction,
                    learning_rate=config.learning_rate,
                    n_estimators=config.n_estimators,
                    random_state=config.random_state,
                    n_jobs=config.n_jobs,
                    verbose=config.verbose
                )

            model = _build_model()
            model.fit(X, y.ravel(), sample_weight=sample_weight)

            oof_predictions = self._generate_model_oof_predictions(
                _build_model,
                X,
                y,
                sample_weight,
                n_splits=5
            )

            result = {
                'models': {'lgbm': model},
                'metrics': {'model_type': 'LGBM', 'config': config}
            }

            if oof_predictions is not None:
                result['oof_predictions'] = {'lgbm': oof_predictions.tolist()}

            return result

        except Exception as e:
            tprint_error(f"❌ LGBM training failed: {e}")
            return {'models': {}, 'metrics': {}}

    async def _train_lgbm_patchtst(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Train LGBM + PatchTST model."""
        try:
            from src.models.enhanced_patchtst import create_enhanced_patchtst, EnhancedPatchTSTConfig
            from src.config.updated_model_configs import LGBMConfig, PatchTSTConfig

            # Create PatchTST model
            patchtst_config = PatchTSTConfig()
            patchtst_model = create_enhanced_patchtst(patchtst_config)

            # Fit PatchTST to get features
            patchtst_model.fit(X, y, sample_weight)

            # Retrieve stored OOF embeddings
            oof_features = patchtst_model.get_oof_features()
            oof_embeddings = None
            if oof_features and 'oof_embeddings' in oof_features:
                oof_embeddings = oof_features.get('oof_embeddings')

            if oof_embeddings is None or oof_embeddings.shape[0] != X.shape[0]:
                tprint_warning(
                    "⚠️ PatchTST OOF embeddings unavailable or misaligned; falling back to in-sample features"
                )
                patchtst_features = patchtst_model.get_features(X)
                oof_embeddings = patchtst_features['embeddings']

            embedding_names = [f'patchtst_oof_{i}' for i in range(oof_embeddings.shape[1])]
            filtered_embeddings, filter_metadata = filter_embedding_features(
                parent_features=X,
                embedding_features=oof_embeddings,
                target=y,
                embedding_names=embedding_names,
                corr_threshold=patchtst_config.correlation_threshold if hasattr(patchtst_config, 'correlation_threshold') else 0.8,
                ic_threshold=patchtst_config.ic_threshold if hasattr(patchtst_config, 'ic_threshold') else 0.05,
                min_embeddings=patchtst_config.export_dims - 4 if patchtst_config.export_dims > 6 else 6,
                max_embeddings=patchtst_config.export_dims if patchtst_config.export_dims <= 10 else 10
            )

            if filtered_embeddings.shape[1] == 0:
                fallback_dims = min(10, max(6, oof_embeddings.shape[1]))
                filtered_embeddings = oof_embeddings[:, :fallback_dims]
                filter_metadata['retained_count'] = filtered_embeddings.shape[1]
                filter_metadata['within_budget'] = 6 <= filtered_embeddings.shape[1] <= 10
                tprint_warning("⚠️ No embeddings survived filtering; using fallback slice of OOF embeddings")

            if not filter_metadata.get('within_budget', False):
                tprint_warning(
                    f"⚠️ Filtered PatchTST embeddings outside budget: {filter_metadata.get('retained_count', 0)}"
                )

            tprint_info(
                f"📦 PatchTST OOF embedding dimensions retained: {filtered_embeddings.shape[1]}"
            )

            # Combine original features with filtered OOF embeddings
            X_combined = np.hstack([X, filtered_embeddings])

            # Train LGBM on combined features
            lgb_config = LGBMConfig()
            import lightgbm as lgb
            lgbm_model = lgb.LGBMRegressor(
                max_depth=lgb_config.max_depth,
                num_leaves=lgb_config.num_leaves,
                min_child_samples=lgb_config.min_child_samples,
                reg_lambda=lgb_config.lambda_l2,
                feature_fraction=lgb_config.feature_fraction,
                learning_rate=lgb_config.learning_rate,
                n_estimators=lgb_config.n_estimators,
                random_state=lgb_config.random_state,
                n_jobs=lgb_config.n_jobs,
                verbose=lgb_config.verbose
            )
            
            lgbm_model.fit(X_combined, y.ravel(), sample_weight=sample_weight)

            return {
                'models': {
                    'lgbm_patchtst': lgbm_model,
                    'patchtst': patchtst_model
                },
                'metrics': {
                    'model_type': 'LGBM_PatchTST',
                    'lgbm_config': lgb_config,
                    'patchtst_config': patchtst_config,
                    'embedding_filter_metadata': filter_metadata
                }
            }

        except Exception as e:
            tprint_error(f"❌ LGBM + PatchTST training failed: {e}")
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

            # Generate meta OOF matrix and persist before fitting
            meta_oof_matrix = model._prepare_stacking_features(formatted_predictions)

            stacker_output_dir = kwargs.get('output_directory') or os.path.join(
                self.config.output_directory, 'stacker_lgbm_calibrated'
            )
            os.makedirs(stacker_output_dir, exist_ok=True)

            meta_oof_path = os.path.join(stacker_output_dir, 'meta_oof_predictions.npy')
            np.save(meta_oof_path, meta_oof_matrix)

            # Fit final stacker on OOF-derived features
            model.fit(formatted_predictions, y.ravel(), sample_weight)

            # Update metrics with the final stacker information
            metrics = {
                'model_type': 'Stacker_LGBM_Calibrated',
                'config': config,
                'base_models_used': list(formatted_predictions.keys()),
                'meta_oof_path': meta_oof_path,
                'meta_oof_predictions': meta_oof_matrix.tolist(),
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
                'direction_settings': {
                    'enable_long_positions': config.enable_long_positions,
                    'enable_short_positions': config.enable_short_positions,
                }
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
            tprint_info("📊 Using regime probabilities for Analyst models")
            
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

        tprint_info(f"📊 Assembled {len(feature_map)} regime features for Analyst models")
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

    async def _train_nas(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Train NAS (Neural Architecture Search) model for 15m timeframe."""
        try:
            from .nas_tas.training_orchestrator import TrainingOrchestrator, OrchestratorConfig, OrchestrationMode

            # Create training configuration for NAS
            output_directory = kwargs.get('output_directory') or f"{self.config.output_directory}/nas"
            os.makedirs(output_directory, exist_ok=True)

            orchestrator_config = OrchestratorConfig(
                mode=OrchestrationMode.TRAINING_ONLY,
                enable_regime_detection=True,
                enable_model_training=True,
                enable_model_selection=True,
                enable_model_management=False,
                enable_performance_tracking=False,
                output_directory=output_directory,
                save_models=self.config.save_models,
                save_results=self.config.save_models,
                direction_mode='both',
                separate_directional_features=False,
            )

            # Set parallel processing
            orchestrator_config.enable_parallel_processing = self.config.enable_parallel_processing
            orchestrator_config.max_workers = max(1, os.cpu_count() or 1)

            # Initialize orchestrator
            orchestrator = TrainingOrchestrator(orchestrator_config)

            # Prepare data for training
            training_data = kwargs.get('training_data')
            feature_columns = kwargs.get('feature_columns')
            target_columns = kwargs.get('target_columns') or ['target']
            regime_assignments = kwargs.get('regime_assignments')
            timestamps = kwargs.get('timestamps')

            if training_data is None or not feature_columns:
                raise ValueError("Training data and feature columns required for NAS training")

            # Create dataset for orchestrator
            feature_set = list(dict.fromkeys(feature_columns))
            dataset = training_data[feature_set + target_columns].dropna(subset=target_columns).copy()

            if dataset.empty:
                raise ValueError("No valid data for NAS training")

            dataset = dataset.rename(columns={target_columns[0]: 'target'})

            # Determine timestamps to maintain temporal ordering
            timestamps_series = None
            if timestamps is not None:
                timestamps_series = timestamps
            elif 'timestamp' in training_data.columns:
                timestamps_series = training_data.loc[dataset.index, 'timestamp']

            # Set context for analyst training
            context = {
                'source': 'AnalystModelsTrainingStep',
                'model_type': 'NAS',
                'timeframe': '15m',
                'feature_count': len(feature_set),
            }

            if regime_assignments is not None:
                context['regime_assignment_shape'] = getattr(regime_assignments, 'shape', None)

            # Train NAS model
            orchestration_result = await orchestrator.orchestrate_async(
                market_data=dataset,
                target_variable='target',
                feature_columns=feature_set,
                timestamps=timestamps_series,
                context=context,
            )

            if not orchestration_result.success or not orchestration_result.training_result:
                raise RuntimeError(f"NAS orchestrator failed: {orchestration_result.error_message}")

            training_result = orchestration_result.training_result
            if not training_result.success:
                raise RuntimeError(f"NAS training unsuccessful: {training_result.error_message}")

            # Extract trained models
            models = {}
            metrics = {
                'model_type': 'NAS',
                'timeframe': '15m',
                'overall_performance': training_result.overall_performance,
                'regime_performance': training_result.regime_performance,
                'n_regimes': training_result.n_regimes_detected,
                'execution_time': orchestration_result.execution_time,
            }

            # Store regime-specific models
            for regime_id, regime_models in training_result.models_trained.items():
                for model_name, model_info in regime_models.items():
                    combined_key = f"regime_{regime_id}_{model_name}"
                    models[combined_key] = model_info.get('model')

                    # Add per-model metrics
                    if combined_key not in metrics:
                        metrics[combined_key] = {}
                    metrics[combined_key].update({
                        'regime': regime_id,
                        'train_metrics': model_info.get('train_metrics'),
                        'val_metrics': model_info.get('val_metrics'),
                        'test_metrics': model_info.get('test_metrics'),
                        'feature_importance': model_info.get('feature_importance'),
                        'hyperparameters': model_info.get('hyperparameters'),
                    })

            return {
                'models': models,
                'metrics': metrics
            }

        except Exception as e:
            tprint_error(f"❌ NAS training failed: {e}")
            return {'models': {}, 'metrics': {}}

    async def _train_tas(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Train TAS (Tree-based Architecture Search) model for 15m timeframe."""
        try:
            from .nas_tas.training_orchestrator import TrainingOrchestrator, OrchestratorConfig, OrchestrationMode

            # Create training configuration for TAS
            output_directory = kwargs.get('output_directory') or f"{self.config.output_directory}/tas"
            os.makedirs(output_directory, exist_ok=True)

            orchestrator_config = OrchestratorConfig(
                mode=OrchestrationMode.TRAINING_ONLY,
                enable_regime_detection=True,
                enable_model_training=True,
                enable_model_selection=True,
                enable_model_management=False,
                enable_performance_tracking=False,
                output_directory=output_directory,
                save_models=self.config.save_models,
                save_results=self.config.save_models,
                direction_mode='both',
                separate_directional_features=False,
            )

            # Set parallel processing
            orchestrator_config.enable_parallel_processing = self.config.enable_parallel_processing
            orchestrator_config.max_workers = max(1, os.cpu_count() or 1)

            # Initialize orchestrator
            orchestrator = TrainingOrchestrator(orchestrator_config)

            # Prepare data for training
            training_data = kwargs.get('training_data')
            feature_columns = kwargs.get('feature_columns')
            target_columns = kwargs.get('target_columns') or ['target']
            regime_assignments = kwargs.get('regime_assignments')
            timestamps = kwargs.get('timestamps')

            if training_data is None or not feature_columns:
                raise ValueError("Training data and feature columns required for TAS training")

            # Create dataset for orchestrator
            feature_set = list(dict.fromkeys(feature_columns))
            dataset = training_data[feature_set + target_columns].dropna(subset=target_columns).copy()

            if dataset.empty:
                raise ValueError("No valid data for TAS training")

            dataset = dataset.rename(columns={target_columns[0]: 'target'})

            # Determine timestamps to maintain temporal ordering
            timestamps_series = None
            if timestamps is not None:
                timestamps_series = timestamps
            elif 'timestamp' in training_data.columns:
                timestamps_series = training_data.loc[dataset.index, 'timestamp']

            # Set context for analyst training
            context = {
                'source': 'AnalystModelsTrainingStep',
                'model_type': 'TAS',
                'timeframe': '15m',
                'feature_count': len(feature_set),
            }

            if regime_assignments is not None:
                context['regime_assignment_shape'] = getattr(regime_assignments, 'shape', None)

            # Train TAS model
            orchestration_result = await orchestrator.orchestrate_async(
                market_data=dataset,
                target_variable='target',
                feature_columns=feature_set,
                timestamps=timestamps_series,
                context=context,
            )

            if not orchestration_result.success or not orchestration_result.training_result:
                raise RuntimeError(f"TAS orchestrator failed: {orchestration_result.error_message}")

            training_result = orchestration_result.training_result
            if not training_result.success:
                raise RuntimeError(f"TAS training unsuccessful: {training_result.error_message}")

            # Extract trained models
            models = {}
            metrics = {
                'model_type': 'TAS',
                'timeframe': '15m',
                'overall_performance': training_result.overall_performance,
                'regime_performance': training_result.regime_performance,
                'n_regimes': training_result.n_regimes_detected,
                'execution_time': orchestration_result.execution_time,
            }

            # Store regime-specific models
            for regime_id, regime_models in training_result.models_trained.items():
                for model_name, model_info in regime_models.items():
                    combined_key = f"regime_{regime_id}_{model_name}"
                    models[combined_key] = model_info.get('model')

                    # Add per-model metrics
                    if combined_key not in metrics:
                        metrics[combined_key] = {}
                    metrics[combined_key].update({
                        'regime': regime_id,
                        'train_metrics': model_info.get('train_metrics'),
                        'val_metrics': model_info.get('val_metrics'),
                        'test_metrics': model_info.get('test_metrics'),
                        'feature_importance': model_info.get('feature_importance'),
                        'hyperparameters': model_info.get('hyperparameters'),
                    })

            return {
                'models': models,
                'metrics': metrics
            }

        except Exception as e:
            tprint_error(f"❌ TAS training failed: {e}")
            return {'models': {}, 'metrics': {}}

    def optimize_training_data(self, training_data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize training data using VectorBT utilities for better performance.
        
        Args:
            training_data: Input training DataFrame
            
        Returns:
            Optimized training DataFrame
        """
        if not VECTORBT_UTILS_AVAILABLE or self.vectorization_manager is None:
            tprint_warning("⚠️ VectorBT utilities not available, returning original data")
            return training_data
        
        try:
            tprint_info("🔧 Optimizing training data with VectorBT utilities...")
            
            # Optimize DataFrame for memory efficiency and VectorBT processing
            optimized_data = self.vectorization_manager.optimize_dataframe(training_data)
            
            # Get performance statistics
            stats = self.vectorization_manager.get_performance_stats()
            memory_savings = stats.get('memory_savings', 0)
            
            if memory_savings > 0:
                tprint_success(f"✅ Data optimization completed: {memory_savings:.1f}% memory savings")
            else:
                tprint_success("✅ Data optimization completed")
            
            return optimized_data
            
        except Exception as e:
            tprint_warning(f"⚠️ Data optimization failed: {e}, returning original data")
            return training_data

    def generate_vectorbt_features(self, training_data: pd.DataFrame, 
                                 feature_columns: List[str]) -> pd.DataFrame:
        """
        Generate additional features using VectorBT rolling operations.
        
        Args:
            training_data: Input training DataFrame
            feature_columns: List of feature column names
            
        Returns:
            DataFrame with additional VectorBT-generated features
        """
        if not VECTORBT_UTILS_AVAILABLE or self.rolling_optimizer is None:
            tprint_warning("⚠️ VectorBT rolling optimizer not available, returning original data")
            return training_data
        
        try:
            tprint_info("🔧 Generating VectorBT features...")
            
            # Create a copy to avoid modifying original data
            enhanced_data = training_data.copy()
            
            # Generate rolling features for numeric columns
            numeric_columns = training_data[feature_columns].select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col in training_data.columns:
                    try:
                        # Generate rolling mean and std features
                        rolling_mean_20 = self.rolling_optimizer.rolling_mean(training_data[col], window=20)
                        rolling_std_20 = self.rolling_optimizer.rolling_std(training_data[col], window=20)
                        rolling_mean_50 = self.rolling_optimizer.rolling_mean(training_data[col], window=50)
                        
                        # Add new features
                        enhanced_data[f'{col}_rolling_mean_20'] = rolling_mean_20
                        enhanced_data[f'{col}_rolling_std_20'] = rolling_std_20
                        enhanced_data[f'{col}_rolling_mean_50'] = rolling_mean_50
                        
                        # Generate rolling correlation with price if it's a price column
                        if 'close' in col.lower() or 'price' in col.lower():
                            for other_col in numeric_columns:
                                if other_col != col and 'volume' in other_col.lower():
                                    try:
                                        rolling_corr = self.rolling_optimizer.rolling_corr(
                                            training_data[col], training_data[other_col], window=20
                                        )
                                        enhanced_data[f'{col}_{other_col}_rolling_corr_20'] = rolling_corr
                                    except Exception as e:
                                        tprint_debug(f"⚠️ Failed to generate correlation for {col}-{other_col}: {e}")
                        
                    except Exception as e:
                        tprint_debug(f"⚠️ Failed to generate rolling features for {col}: {e}")
                        continue
            
            # Get performance statistics
            stats = self.rolling_optimizer.get_performance_stats()
            vectorbt_ops = stats.get('vectorbt_operations', 0)
            
            tprint_success(f"✅ Generated VectorBT features: {vectorbt_ops} operations completed")
            
            return enhanced_data
            
        except Exception as e:
            tprint_warning(f"⚠️ VectorBT feature generation failed: {e}, returning original data")
            return training_data

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the models training step."""
        metrics = {
            'config': {
                'model_types': [mt.value for mt in self.config.model_types],
                'save_models': self.config.save_models,
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
                'vectorization_manager': self.vectorization_manager is not None,
                'rolling_optimizer': self.rolling_optimizer is not None,
                'vectorbt_utils_available': VECTORBT_UTILS_AVAILABLE
            }
        }

        # Add VectorBT performance stats if available
        if self.vectorization_manager is not None:
            try:
                vectorbt_stats = self.vectorization_manager.get_performance_stats()
                metrics['vectorbt_performance'] = {
                    'total_operations': vectorbt_stats.get('total_operations', 0),
                    'vectorbt_operations': vectorbt_stats.get('vectorbt_operations', 0),
                    'memory_optimizations': vectorbt_stats.get('memory_optimizations', 0),
                    'memory_savings': vectorbt_stats.get('memory_savings', 0),
                    'cache_hit_rate': vectorbt_stats.get('cache_hit_rate', 0)
                }
            except Exception as e:
                tprint_debug(f"⚠️ Failed to get VectorBT performance stats: {e}")

        return metrics

    def _infer_direction_from_target(self, target_column: str) -> str:
        """Infer trading direction from the target column name."""
        target_lower = target_column.lower()
        if 'long' in target_lower:
            return 'long'
        if 'short' in target_lower:
            return 'short'
        return 'combined'

    def _get_nas_tas_orchestrator(
        self,
        model_type: AnalystModelType,
        direction: str,
        base_output_directory: str,
    ) -> TrainingOrchestrator:
        """Get or create a NAS/TAS orchestrator for the specified direction."""
        if not NAS_TAS_AVAILABLE:
            raise RuntimeError("NAS/TAS orchestrator components are not available")

        direction_key = direction if direction in {'long', 'short'} else 'combined'
        cache_key = (model_type.value, direction_key, base_output_directory)

        if cache_key in self._nas_tas_orchestrators:
            return self._nas_tas_orchestrators[cache_key]

        output_directory = os.path.join(base_output_directory, direction_key)
        os.makedirs(output_directory, exist_ok=True)

        orchestrator_config = OrchestratorConfig(
            mode=OrchestrationMode.TRAINING_ONLY,
            enable_regime_detection=True,
            enable_model_training=True,
            enable_model_selection=True,
            enable_model_management=False,
            enable_performance_tracking=False,
            output_directory=output_directory,
            save_models=self.config.save_models,
            save_results=self.config.save_models,
            direction_mode=(
                'long_only' if direction_key == 'long'
                else 'short_only' if direction_key == 'short'
                else 'both'
            ),
            separate_directional_features=False,
        )
        orchestrator_config.enable_parallel_processing = self.config.enable_parallel_processing
        orchestrator_config.max_workers = max(1, os.cpu_count() or 1)

        orchestrator = TrainingOrchestrator(orchestrator_config)
        self._nas_tas_orchestrators[cache_key] = orchestrator
        return orchestrator

    async def _train_nas_tas_models(
        self,
        model_type: AnalystModelType,
        training_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train NAS/TAS models using the unified orchestrator."""
        if not NAS_TAS_AVAILABLE:
            raise RuntimeError("NAS/TAS orchestrator components are not available")

        training_data: pd.DataFrame = training_config['training_data']
        feature_columns: List[str] = training_config['feature_columns']
        target_columns: List[str] = training_config['target_columns']
        regime_assignments = training_config.get('regime_assignments')
        timestamps = training_config.get('timestamps')
        base_output_directory = training_config.get('output_directory', self.config.output_directory)

        if training_data is None or not feature_columns or not target_columns:
            raise ValueError("Training data, feature columns, and target columns must be provided for NAS/TAS training")

        models: Dict[str, Any] = {}
        metrics: Dict[str, Any] = {}

        for target_column in target_columns:
            if target_column not in training_data.columns:
                tprint_warning(f"⚠️ Target column '{target_column}' not found for {model_type.value}, skipping")
                continue

            direction = self._infer_direction_from_target(target_column)
            direction_key = direction if direction in {'long', 'short'} else 'combined'

            tprint_info(
                f"🤖 Delegating {model_type.value} ({direction_key}) training to NAS/TAS orchestrator"
            )

            # Prepare dataset for orchestrator
            feature_set = list(dict.fromkeys(feature_columns))
            dataset = training_data[feature_set + [target_column]].dropna(subset=[target_column]).copy()

            if dataset.empty:
                tprint_warning(f"⚠️ No data available for {model_type.value} ({direction_key}) after cleaning, skipping")
                continue

            dataset = dataset.rename(columns={target_column: 'target'})

            # Determine timestamps to maintain temporal ordering
            timestamps_series = None
            if timestamps is not None:
                timestamps_series = timestamps
            elif 'timestamp' in training_data.columns:
                timestamps_series = training_data.loc[dataset.index, 'timestamp']

            orchestrator = self._get_nas_tas_orchestrator(model_type, direction_key, base_output_directory)

            context = {
                'source': 'AnalystModelsTrainingStep',
                'model_type': model_type.value,
                'direction': direction_key,
                'feature_count': len(feature_set),
            }

            if regime_assignments is not None:
                context['regime_assignment_shape'] = getattr(regime_assignments, 'shape', None)

            orchestration_result = await orchestrator.orchestrate_async(
                market_data=dataset,
                target_variable='target',
                feature_columns=feature_set,
                timestamps=timestamps_series,
                context=context,
            )

            if not orchestration_result.success or not orchestration_result.training_result:
                raise RuntimeError(
                    f"NAS/TAS orchestrator failed for {model_type.value} ({direction_key}): "
                    f"{orchestration_result.error_message}"
                )

            training_result = orchestration_result.training_result
            if not training_result.success:
                raise RuntimeError(
                    f"NAS/TAS training unsuccessful for {model_type.value} ({direction_key}): "
                    f"{training_result.error_message}"
                )

            direction_metrics = {
                'direction': direction_key,
                'overall_performance': training_result.overall_performance,
                'regime_performance': training_result.regime_performance,
                'execution_time': orchestration_result.execution_time,
                'n_regimes': training_result.n_regimes_detected,
            }

            if training_result.directional_statistics:
                direction_metrics['directional_statistics'] = training_result.directional_statistics.get(direction_key)

            direction_models_metrics: Dict[str, Any] = {}

            for regime_id, regime_models in training_result.models_trained.items():
                for model_name, model_info in regime_models.items():
                    combined_key = f"{direction_key}_regime_{regime_id}_{model_name}"
                    models[combined_key] = model_info.get('model')
                    direction_models_metrics[combined_key] = {
                        'train_metrics': model_info.get('train_metrics'),
                        'val_metrics': model_info.get('val_metrics'),
                        'test_metrics': model_info.get('test_metrics'),
                        'feature_importance': model_info.get('feature_importance'),
                        'hyperparameters': model_info.get('hyperparameters'),
                    }

            if direction_models_metrics:
                direction_metrics['models'] = direction_models_metrics

            metrics[direction_key] = direction_metrics

        if not models:
            raise RuntimeError(f"NAS/TAS orchestrator did not return any trained models for {model_type.value}")

        return {
            'models': models,
            'metrics': metrics,
            'direction_settings': {
                'enable_long_positions': config.enable_long_positions,
                'enable_short_positions': config.enable_short_positions,
            }
        }


# Convenience function for external usage
async def execute_analyst_models_training(
    training_data: pd.DataFrame,
    feature_columns: List[str],
    target_columns: List[str],
    sample_weight: Optional[np.ndarray] = None,
    config: Optional[AnalystModelsTrainingConfig] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Execute Analyst base models training.

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
    trainer = AnalystModelsTrainingStep(config)
    return await trainer.train_analyst_models(
        training_data, feature_columns, target_columns, sample_weight, **kwargs
    )


# Apply negative learning patches
if NEGATIVE_LEARNING_AVAILABLE:
    try:
        apply_negative_learning_patches()
        print("✅ Negative learning patches applied to Analyst training")
    except Exception as e:
        print(f"⚠️ Failed to apply negative learning patches: {e}")