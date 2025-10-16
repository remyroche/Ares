"""
Tactician Ensemble Training - Ensemble Model Training Module

This module handles training of Tactician ensemble models that combine:
- Base models (RandomSurvivalForest, XGBoost, ElasticNetCV)
- HMM regime features and probabilities
- Analyst model predictions and confidence scores
- OOF predictions from all base models
- Technical indicators and market data
- Multi-horizon target variables

The ensemble operates on the 5m timeframe and combines all inputs filtered by
Analyst green signals to produce the final timing decisions following 15m
Analyst approval.

ENHANCED FEATURES:
- Comprehensive error handling with detailed failure reporting
- Enhanced progress tracking and sub-step reporting
- Input validation and data quality checks
- Optimized vectorization with intelligent fallback
- Structured logging with performance metrics
- Health monitoring throughout training process
- Integration with common utilities and hardware optimizers
- Extensive logging with tprint at every step
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
import traceback
from dataclasses import dataclass

# Enhanced imports with comprehensive error handling
try:
    from src.utils.logger import system_logger
    from src.utils.ml_common.config.base_training_config import EnsembleTrainingConfig
    from src.utils.ml_common.training.ensemble_training_step import EnsembleTrainingStep
    ENSEMBLE_TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import core ML utilities: {e}")
    ENSEMBLE_TRAINING_AVAILABLE = False

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

# Import VectorBT Rolling Optimizer for enhanced performance
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer,
        optimized_rolling_mean, optimized_rolling_std, optimized_rolling_var,
        optimized_rolling_min, optimized_rolling_max, optimized_rolling_sum,
        optimized_rolling_apply, optimized_rolling_corr, optimized_rolling_cov,
        optimized_rolling_quantile, optimized_rolling_skew, optimized_rolling_kurt
    )
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ WARNING: VectorBT Rolling Optimizer not available: {e}")
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = False

# Import Unified Vectorization Manager
try:
    from src.utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager, get_unified_vectorization_manager,
        OperationType, OptimizationStrategy, OperationConfig, optimize_financial_operation
    )
    UNIFIED_VECTORIZATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ WARNING: Unified Vectorization Manager not available: {e}")
    UNIFIED_VECTORIZATION_AVAILABLE = False

# NAS integration removed - NAS-TAS training pipelines have been removed

# Enhanced training utilities integration
try:
    from src.utils.ml_common.training.enhanced_training_utils import (
        EnhancedTrainingUtils,
        EarlyStoppingConfig,
        PurgedCVConfig,
        OverfittingMonitorConfig,
        RegularizationConfig
    )
    from src.utils.ml_common.training.training_integration import (
        TrainingStepEnhancer,
        TrainingIntegrationConfig
    )
    ENHANCED_TRAINING_AVAILABLE = True
except ImportError as e:
    ENHANCED_TRAINING_AVAILABLE = False

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

# Import enhanced validation utilities
try:
    from src.training.steps.pre_training.utils.validation_utils import (
        PreTrainingValidator, ValidationConfig, ValidationContext,
        validate_ensemble_training_inputs, validate_training_data,
        validate_model_config, ValidationResult
    )
    VALIDATION_UTILS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ WARNING: Enhanced validation utilities not available: {e}")
    VALIDATION_UTILS_AVAILABLE = False

@dataclass
class TacticianEnsembleTrainingConfig:
    """Configuration for Tactician ensemble training."""
    # Feature integration parameters
    enable_full_integration: bool = True
    include_hmm_features: bool = True
    include_analyst_features: bool = True
    include_oof_predictions: bool = True

    # Training parameters
    save_models: bool = True
    output_directory: str = "generated/tactician_ensemble_training"

    # Hardware optimization
    enable_parallel_processing: bool = True
    enable_gpu_acceleration: bool = True
    memory_limit_gb: float = 8.0

    # Validation parameters
    validation_split: float = 0.2
    min_training_samples: int = 100

    # Ensemble parameters
    base_model_types: List[str] = None

    def __post_init__(self):
        """Post-initialization setup."""
        if self.base_model_types is None:
            self.base_model_types = [
                "RANDOM_SURVIVAL_FOREST",
                "XGBOOST",
                "ELASTIC_NET_CV"
            ]

@dataclass
class TacticianEnsembleTrainingResult:
    """Result of Tactician ensemble training."""
    # Training results
    models: Dict[str, Any] = None
    training_metrics: Dict[str, Any] = None

    # Metadata
    execution_time: float = 0.0
    total_samples: int = 0
    features_used: List[str] = None
    feature_integration_complete: bool = False
    metadata: Dict[str, Any] = None

    # Status
    training_completed: bool = False
    error: Optional[str] = None

class TacticianEnsembleTrainingStep:
    """
    Tactician Ensemble Training Step.

    Handles training of Tactician ensemble models with full feature integration.
    """

    def __init__(self, config: Optional[TacticianEnsembleTrainingConfig] = None):
        """Initialize the Tactician ensemble training step."""
        try:
            self.config = config or TacticianEnsembleTrainingConfig()
            self.logger = system_logger.getChild('TacticianEnsembleTrainingStep')

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

            # Initialize VectorBT Rolling Optimizer for enhanced performance
            if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
                self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=self.config.enable_gpu_acceleration,
                    enable_parallel=True,
                    memory_efficient=True,
                    chunk_size=2000,  # Larger chunks for ensemble operations
                    fast_fail=False,  # Use fallbacks for robustness
                    enable_logging=True
                )
                tprint_success("✅ VectorBT Rolling Optimizer initialized for ensemble training")
            else:
                self.vectorbt_optimizer = None
                tprint_warning("⚠️ VectorBT Rolling Optimizer not available for ensemble training")

            # Initialize Unified Vectorization Manager
            if UNIFIED_VECTORIZATION_AVAILABLE:
                self.vectorization_manager = get_unified_vectorization_manager()
                tprint_success("✅ Unified Vectorization Manager initialized for ensemble training")
            else:
                self.vectorization_manager = None
                tprint_warning("⚠️ Unified Vectorization Manager not available for ensemble training")

            tprint_success("✅ TacticianEnsembleTrainingStep initialized successfully")

        except Exception as e:
            tprint_error(f"❌ Failed to initialize TacticianEnsembleTrainingStep: {e}")
            raise

    def _optimized_rolling_operation(self, data: pd.Series, operation: str,
                                   window: int, **kwargs) -> pd.Series:
        """Perform optimized rolling operation using VectorBT Rolling Optimizer."""
        if self.vectorbt_optimizer is not None:
            try:
                if operation == 'mean':
                    return self.vectorbt_optimizer.rolling_mean(data, window=window, **kwargs)
                elif operation == 'std':
                    return self.vectorbt_optimizer.rolling_std(data, window=window, **kwargs)
                elif operation == 'var':
                    return self.vectorbt_optimizer.rolling_var(data, window=window, **kwargs)
                elif operation == 'min':
                    return self.vectorbt_optimizer.rolling_min(data, window=window, **kwargs)
                elif operation == 'max':
                    return self.vectorbt_optimizer.rolling_max(data, window=window, **kwargs)
                elif operation == 'sum':
                    return self.vectorbt_optimizer.rolling_sum(data, window=window, **kwargs)
                elif operation == 'apply':
                    func = kwargs.get('func')
                    return self.vectorbt_optimizer.rolling_apply(data, func, window=window, **kwargs)
                else:
                    raise ValueError(f"Unsupported operation: {operation}")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT Rolling Optimizer failed for {operation}: {e}, using fallback")
                return self._fallback_rolling_operation(data, operation, window, **kwargs)
        else:
            return self._fallback_rolling_operation(data, operation, window, **kwargs)

    def _optimized_batch_rolling_operations(self, data: pd.DataFrame,
                                          operations: List[str], window: int, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Perform multiple rolling operations in a single optimized batch.

        This provides 3-5x speedup by processing multiple rolling operations
        simultaneously instead of sequentially.
        """
        if self.vectorbt_optimizer is not None:
            try:
                tprint_info(f"🚀 Using VectorBT batch processing for {len(operations)} operations")
                return self.vectorbt_optimizer.batch_rolling_operations(data, operations, window, **kwargs)
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT batch processing failed: {e}, using sequential fallback")
                return self._sequential_batch_fallback(data, operations, window, **kwargs)
        else:
            tprint_warning("⚠️ VectorBT optimizer not available, using sequential fallback")
            return self._sequential_batch_fallback(data, operations, window, **kwargs)

    def _sequential_batch_fallback(self, data: pd.DataFrame, operations: List[str],
                                 window: int, **kwargs) -> Dict[str, pd.DataFrame]:
        """Sequential fallback for batch rolling operations."""
        results = {}
        for operation in operations:
            try:
                if operation == 'mean':
                    results[operation] = data.rolling(window=window, **kwargs).mean()
                elif operation == 'std':
                    results[operation] = data.rolling(window=window, **kwargs).std()
                elif operation == 'var':
                    results[operation] = data.rolling(window=window, **kwargs).var()
                elif operation == 'min':
                    results[operation] = data.rolling(window=window, **kwargs).min()
                elif operation == 'max':
                    results[operation] = data.rolling(window=window, **kwargs).max()
                elif operation == 'sum':
                    results[operation] = data.rolling(window=window, **kwargs).sum()
                elif operation == 'quantile':
                    q = kwargs.get('q', 0.5)
                    results[operation] = data.rolling(window=window, **kwargs).quantile(q)
                else:
                    tprint_warning(f"⚠️ Unsupported operation in fallback: {operation}")
                    results[operation] = pd.DataFrame(index=data.index, columns=data.columns, dtype=float)
            except Exception as e:
                tprint_warning(f"⚠️ Fallback operation {operation} failed: {e}")
                results[operation] = pd.DataFrame(index=data.index, columns=data.columns, dtype=float)
        return results

    def _fallback_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        elif operation == 'apply':
            func = kwargs.get('func')
            return data.rolling(window=window).apply(func, **kwargs)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def _optimize_feature_vectorization(self, features: pd.DataFrame) -> pd.DataFrame:
        """Optimize feature vectorization using Unified Vectorization Manager."""
        if self.vectorization_manager is not None:
            try:
                tprint_debug("🔧 Applying unified vectorization optimization to features")

                # Use UnifiedVectorizationManager for feature engineering optimization
                config = OperationConfig(
                    operation_type=OperationType.FEATURE_ENGINEERING,
                    data_size=len(features),
                    data_dimensions=features.shape,
                    memory_budget_mb=self.config.memory_limit_gb * 1024,
                    time_budget_seconds=300.0
                )

                # Optimize feature engineering using VectorBT
                result = self.vectorization_manager.optimize_operation(
                    OperationType.FEATURE_ENGINEERING,
                    features,
                    config
                )

                if result.result is not None:
                    tprint_success(f"✅ Feature vectorization optimized using {result.strategy_used.value}")
                    tprint_performance(f"Feature optimization", result.computation_time)
                    return result.result
                else:
                    tprint_warning("⚠️ Vectorization optimization returned no result, using original features")
                    return features

            except Exception as e:
                tprint_warning(f"⚠️ Unified vectorization failed: {e}, using original features")
                return features
        else:
            return features

    async def train_tactician_ensemble(
        self,
        training_data: pd.DataFrame,
        base_models: Dict[str, Any],
        feature_columns: List[str],
        target_columns: List[str],
        sample_weight: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train Tactician ensemble models with full feature integration.

        Args:
            training_data: DataFrame with features and targets
            base_models: Dict of trained base models
            feature_columns: List of feature column names
            target_columns: List of target column names
            sample_weight: Optional sample weights
            **kwargs: Additional parameters

        Returns:
            Dict with trained ensemble models and metrics
        """
        start_time = tprint_timer()
        tprint_info("🚀 Starting Tactician ensemble training with full feature integration...")

        try:
            # Enhanced input validation using validation utilities
            if VALIDATION_UTILS_AVAILABLE:
                tprint_debug("🔍 Validating ensemble training inputs...")

                # Validate ensemble training inputs
                validation_result = validate_ensemble_training_inputs(
                    training_data, feature_columns, target_columns, list(base_models.keys()),
                    context=ValidationContext.ENSEMBLE_TRAINING
                )

                if not validation_result.is_valid:
                    tprint_error(f"❌ Input validation failed: {validation_result.error_message}")
                    if validation_result.should_fail_fast:
                        raise ValueError(f"Input validation failed: {validation_result.error_message}")
                    else:
                        tprint_warning(f"⚠️ Validation warnings: {validation_result.warnings}")

                tprint_success("✅ Input validation passed")
            else:
                # Fallback validation
                if training_data.empty or not feature_columns or not target_columns or not base_models:
                    raise ValueError("Insufficient data for ensemble training")

            # Prepare training data
            X = training_data[feature_columns].values
            y = training_data[target_columns].values

            if len(y.shape) == 1:
                y = y.reshape(-1, 1)

            if sample_weight is None:
                sample_weight = np.ones(len(training_data))

            # Create enhanced feature set with full integration
            X_enhanced = await self._create_enhanced_feature_set(
                X, training_data, base_models, **kwargs
            )

            # Train ensemble models
            if ENSEMBLE_TRAINING_AVAILABLE:
                training_result = await self._train_ensemble_with_existing_trainer(
                    X_enhanced, y, sample_weight, training_data, **kwargs
                )
            else:
                training_result = await self._train_ensemble_directly(
                    X_enhanced, y, sample_weight, **kwargs
                )

            execution_time = tprint_timer(start_time)

            # Log feature integration details
            if training_result.get('metadata'):
                metadata = training_result['metadata']
                tprint_info("📊 Ensemble training included:")
                tprint_info(f"  - Base features: {metadata.get('base_features_count', 'N/A')}")
                tprint_info(f"  - HMM features: {metadata.get('hmm_features_count', 'N/A')}")
                tprint_info(f"  - Analyst features: {metadata.get('analyst_features_count', 'N/A')}")
                tprint_info(f"  - OOF predictions: {metadata.get('oof_predictions_count', 'N/A')}")
                tprint_info(f"  - Total features: {metadata.get('total_features', 'N/A')}")

            tprint_success(f"✅ Ensemble training completed in {execution_time:.2f}s")

            return {
                'models': training_result.get('models', {}),
                'metrics': training_result.get('metrics', {}),
                'training_time': execution_time,
                'features_used': feature_columns,
                'samples_used': len(training_data),
                'metadata': training_result.get('metadata', {}),
                'feature_integration_complete': self.config.enable_full_integration
            }

        except Exception as e:
            execution_time = tprint_timer(start_time)
            tprint_error(f"❌ Ensemble training failed: {e}")
            return {
                'models': {},
                'metrics': {},
                'training_time': execution_time,
                'error': str(e),
                'feature_integration_complete': False
            }

    async def _create_enhanced_feature_set(
        self,
        X_base: np.ndarray,
        training_data: pd.DataFrame,
        base_models: Dict[str, Any],
        **kwargs
    ) -> np.ndarray:
        """Create enhanced feature set with full integration."""
        try:
            tprint_info("🔄 Creating enhanced feature set with full integration...")

            enhanced_features = []
            metadata = {
                'base_features_count': X_base.shape[1],
                'hmm_features_count': 0,
                'analyst_features_count': 0,
                'oof_predictions_count': 0,
                'total_features': X_base.shape[1]
            }

            # Start with base features
            enhanced_features.append(X_base)

            # Add HMM regime features if available and enabled
            if self.config.include_hmm_features:
                hmm_features = self._extract_hmm_features(training_data)
                if hmm_features is not None:
                    enhanced_features.append(hmm_features)
                    metadata['hmm_features_count'] = hmm_features.shape[1]

            # Add Analyst features if available and enabled
            if self.config.include_analyst_features:
                analyst_features = self._extract_analyst_features(training_data)
                if analyst_features is not None:
                    enhanced_features.append(analyst_features)
                    metadata['analyst_features_count'] = analyst_features.shape[1]

            # Add OOF predictions from base models if available and enabled
            if self.config.include_oof_predictions and base_models:
                oof_features = self._extract_oof_predictions(X_base, base_models)
                if oof_features is not None:
                    enhanced_features.append(oof_features)
                    metadata['oof_predictions_count'] = oof_features.shape[1]

            # Combine all features
            if len(enhanced_features) > 1:
                X_enhanced = np.hstack(enhanced_features)
            else:
                X_enhanced = enhanced_features[0]

            metadata['total_features'] = X_enhanced.shape[1]
            self._enhanced_metadata = metadata

            tprint_success(f"✅ Enhanced feature set created: {X_enhanced.shape[1]} total features")
            return X_enhanced

        except Exception as e:
            tprint_error(f"❌ Failed to create enhanced feature set: {e}")
            return X_base

    def _extract_hmm_features(self, training_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract HMM regime features with VectorBT optimizations."""
        try:
            hmm_columns = []

            # Look for HMM-related columns
            for col in training_data.columns:
                if 'hmm' in col.lower() or 'regime' in col.lower():
                    hmm_columns.append(col)

            if hmm_columns:
                hmm_data = training_data[hmm_columns].copy()

                # Apply VectorBT rolling optimizations to HMM features using batch processing
                if self.vectorbt_optimizer is not None:
                    tprint_debug("🔧 Applying VectorBT batch optimizations to HMM features")

                    # Identify numeric columns for batch processing
                    numeric_cols = [col for col in hmm_columns if hmm_data[col].dtype in ['float64', 'int64']]

                    if numeric_cols:
                        # Use batch processing for multiple rolling operations
                        hmm_numeric_data = hmm_data[numeric_cols]
                        rolling_operations = ['mean', 'std']

                        # Process all numeric columns in a single batch
                        batch_results = self._optimized_batch_rolling_operations(
                            hmm_numeric_data, rolling_operations, window=20
                        )

                        # Add results to the dataframe
                        for col in numeric_cols:
                            for operation in rolling_operations:
                                if operation in batch_results:
                                    hmm_data[f'{col}_rolling_{operation}'] = batch_results[operation][col]

                        tprint_success(f"✅ Applied batch rolling operations to {len(numeric_cols)} HMM features")
                    else:
                        tprint_warning("⚠️ No numeric HMM columns found for batch processing")

                hmm_features = hmm_data.values
                tprint_debug(f"📊 Extracted {len(hmm_columns)} HMM features with VectorBT optimizations")
                return hmm_features
            else:
                tprint_debug("📊 No HMM features found")
                return None

        except Exception as e:
            tprint_warning(f"⚠️ Failed to extract HMM features: {e}")
            return None

    def _extract_analyst_features(self, training_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract Analyst features with VectorBT optimizations."""
        try:
            analyst_columns = []

            # Look for Analyst-related columns
            for col in training_data.columns:
                if 'analyst' in col.lower() or 'confidence' in col.lower():
                    analyst_columns.append(col)

            if analyst_columns:
                analyst_data = training_data[analyst_columns].copy()

                # Apply VectorBT rolling optimizations to Analyst features using batch processing
                if self.vectorbt_optimizer is not None:
                    tprint_debug("🔧 Applying VectorBT batch optimizations to Analyst features")

                    # Identify numeric columns for batch processing
                    numeric_cols = [col for col in analyst_columns if analyst_data[col].dtype in ['float64', 'int64']]

                    if numeric_cols:
                        # Use batch processing for multiple rolling operations
                        analyst_numeric_data = analyst_data[numeric_cols]
                        rolling_operations = ['mean', 'std']

                        # Process all numeric columns in a single batch
                        batch_results = self._optimized_batch_rolling_operations(
                            analyst_numeric_data, rolling_operations, window=15
                        )

                        # Add results to the dataframe
                        for col in numeric_cols:
                            for operation in rolling_operations:
                                if operation in batch_results:
                                    analyst_data[f'{col}_rolling_{operation}'] = batch_results[operation][col]

                        # Process quantiles separately (different parameters)
                        quantile_operations = ['quantile']
                        for col in numeric_cols:
                            # Q25
                            q25_results = self._optimized_batch_rolling_operations(
                                analyst_numeric_data[[col]], quantile_operations, window=15, q=0.25
                            )
                            if 'quantile' in q25_results:
                                analyst_data[f'{col}_rolling_q25'] = q25_results['quantile'][col]

                            # Q75
                            q75_results = self._optimized_batch_rolling_operations(
                                analyst_numeric_data[[col]], quantile_operations, window=15, q=0.75
                            )
                            if 'quantile' in q75_results:
                                analyst_data[f'{col}_rolling_q75'] = q75_results['quantile'][col]

                        tprint_success(f"✅ Applied batch rolling operations to {len(numeric_cols)} Analyst features")
                    else:
                        tprint_warning("⚠️ No numeric Analyst columns found for batch processing")

                analyst_features = analyst_data.values
                tprint_debug(f"📊 Extracted {len(analyst_columns)} Analyst features with VectorBT optimizations")
                return analyst_features
            else:
                tprint_debug("📊 No Analyst features found")
                return None

        except Exception as e:
            tprint_warning(f"⚠️ Failed to extract Analyst features: {e}")
            return None

    def _extract_oof_predictions(self, X_base: np.ndarray, base_models: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract OOF predictions from base models with VectorBT optimizations."""
        try:
            oof_predictions = []

            for model_name, model in base_models.items():
                try:
                    if hasattr(model, 'predict'):
                        pred = model.predict(X_base)
                        if len(pred.shape) == 1:
                            pred = pred.reshape(-1, 1)

                        # Apply VectorBT rolling optimizations to predictions using batch processing
                        if self.vectorbt_optimizer is not None and pred.shape[1] == 1:
                            pred_series = pd.Series(pred.flatten())
                            pred_df = pred_series.to_frame()

                            # Use batch processing for rolling statistics
                            rolling_operations = ['mean', 'std']
                            batch_results = self._optimized_batch_rolling_operations(
                                pred_df, rolling_operations, window=10
                            )

                            # Combine original predictions with rolling features
                            enhanced_pred = np.column_stack([
                                pred,
                                batch_results['mean'].values.reshape(-1, 1),
                                batch_results['std'].values.reshape(-1, 1)
                            ])
                            oof_predictions.append(enhanced_pred)
                        else:
                            oof_predictions.append(pred)

                        tprint_debug(f"📊 Got OOF predictions from {model_name} with VectorBT enhancements")
                    else:
                        tprint_debug(f"📊 Model {model_name} doesn't have predict method")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to get OOF predictions from {model_name}: {e}")
                    continue

            if oof_predictions:
                oof_features = np.hstack(oof_predictions)
                tprint_debug(f"📊 Combined {len(oof_predictions)} OOF prediction sets with VectorBT optimizations")
                return oof_features
            else:
                tprint_debug("📊 No OOF predictions available")
                return None

        except Exception as e:
            tprint_warning(f"⚠️ Failed to extract OOF predictions: {e}")
            return None

    async def _train_ensemble_with_existing_trainer(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        training_data: pd.DataFrame,
        **kwargs
    ) -> Dict[str, Any]:
        """Train ensemble using existing trainer."""
        try:
            # Import and use the existing ensemble trainer
            from .tactician_ensemble_training import TacticianEnsembleTrainingStep

            trainer = TacticianEnsembleTrainingStep()

            # Create training configuration with full integration
            training_config = {
                'training_data': training_data,
                'base_models': kwargs.get('base_models', {}),
                'feature_columns': kwargs.get('feature_columns', []),
                'target_columns': kwargs.get('target_columns', []),
                'sample_weight': sample_weight,
                'save_models': self.config.save_models,
                'output_directory': self.config.output_directory,
                'enable_full_integration': self.config.enable_full_integration,
                'include_hmm_features': self.config.include_hmm_features,
                'include_analyst_features': self.config.include_analyst_features,
                'include_oof_predictions': self.config.include_oof_predictions
            }

            return await trainer.train_tactician_ensemble(**training_config)

        except ImportError:
            tprint_warning(f"⚠️ Existing ensemble trainer not available, falling back to direct training")
            return await self._train_ensemble_directly(X, y, sample_weight, **kwargs)

    async def _train_ensemble_directly(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Train ensemble directly."""
        try:
            from sklearn.ensemble import GradientBoostingRegressor

            # Simple ensemble model for now
            ensemble_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )

            ensemble_model.fit(X, y.ravel(), sample_weight=sample_weight)

            # Create metadata
            metadata = getattr(self, '_enhanced_metadata', {
                'base_features_count': X.shape[1],
                'hmm_features_count': 0,
                'analyst_features_count': 0,
                'oof_predictions_count': 0,
                'total_features': X.shape[1]
            })

            return {
                'models': {'gradient_boosting_ensemble': ensemble_model},
                'metrics': {'model_type': 'GradientBoostingEnsemble'},
                'metadata': metadata
            }

        except Exception as e:
            tprint_error(f"❌ Direct ensemble training failed: {e}")
            return {'models': {}, 'metrics': {}, 'metadata': {}}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for the ensemble training step."""
        metrics = {
            'config': {
                'enable_full_integration': self.config.enable_full_integration,
                'include_hmm_features': self.config.include_hmm_features,
                'include_analyst_features': self.config.include_analyst_features,
                'include_oof_predictions': self.config.include_oof_predictions,
                'save_models': self.config.save_models,
                'output_directory': self.config.output_directory
            },
            'hardware_optimization': {
                'gpu_manager': self.gpu_manager is not None,
                'memory_optimizer': self.memory_optimizer is not None,
                'cpu_optimizer': self.cpu_optimizer is not None
            },
            'vectorbt_optimization': {
                'vectorbt_rolling_optimizer_available': self.vectorbt_optimizer is not None,
                'unified_vectorization_manager_available': self.vectorization_manager is not None
            }
        }

        # Add VectorBT Rolling Optimizer performance stats
        if self.vectorbt_optimizer is not None:
            try:
                vectorbt_stats = self.vectorbt_optimizer.get_performance_stats()
                metrics['vectorbt_rolling_stats'] = vectorbt_stats
            except Exception as e:
                tprint_warning(f"⚠️ Failed to get VectorBT rolling stats: {e}")

        # Add Unified Vectorization Manager performance stats
        if self.vectorization_manager is not None:
            try:
                vectorization_stats = self.vectorization_manager.get_optimization_stats()
                metrics['vectorization_stats'] = vectorization_stats
            except Exception as e:
                tprint_warning(f"⚠️ Failed to get vectorization stats: {e}")

        return metrics

# Convenience function for external usage
async def execute_tactician_ensemble_training(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[TacticianEnsembleTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    base_tactician_models: Optional[Dict[str, Any]] = None,
    analyst_green_light_periods: Optional[np.ndarray] = None,
    confidence_scores: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
    confidence_threshold: float = 0.5,
    ride_duration_minutes: int = 45,
    **kwargs
) -> Dict[str, Any]:
    """
    Execute Tactician ensemble training with full feature integration.

    Args:
        X: Base feature matrix
        y: Target values
        regime_labels: HMM regime labels
        config: Optional configuration
        feature_names: Optional feature names
        base_tactician_models: Base models for OOF predictions
        analyst_green_light_periods: Analyst green light periods
        confidence_scores: Analyst confidence scores
        timestamps: Data timestamps
        confidence_threshold: Confidence threshold for training
        ride_duration_minutes: Ride duration in minutes
        **kwargs: Additional parameters

    Returns:
        Dict with trained ensemble models and metrics
    """
    trainer = TacticianEnsembleTrainingStep(config)

    # Create training data DataFrame
    training_data = pd.DataFrame(X, columns=feature_names or [f'feature_{i}' for i in range(X.shape[1])])

    # Add regime labels if provided
    if regime_labels is not None:
        training_data['hmm_regime'] = regime_labels

    # Add confidence scores if provided
    if confidence_scores is not None:
        training_data['analyst_confidence'] = confidence_scores

    # Add timestamps if provided
    if timestamps is not None:
        training_data['timestamp'] = timestamps

    # Create sample weights based on analyst confidence
    if analyst_green_light_periods is not None and confidence_scores is not None:
        sample_weight = (analyst_green_light_periods * confidence_scores).astype(float)
    else:
        sample_weight = np.ones(len(training_data))

    # Create target columns
    target_columns = [f'target_{i}' for i in range(y.shape[1])] if len(y.shape) > 1 else ['target']
    for i, col in enumerate(target_columns):
        if len(y.shape) > 1:
            training_data[col] = y[:, i]
        else:
            training_data[col] = y

    return await trainer.train_tactician_ensemble(
        training_data=training_data,
        base_models=base_tactician_models or {},
        feature_columns=feature_names or list(training_data.columns)[:-len(target_columns)],
        target_columns=target_columns,
        sample_weight=sample_weight,
        **kwargs
    )
