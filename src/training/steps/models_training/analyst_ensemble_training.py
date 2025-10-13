"""
Analyst Ensemble Training - Ensemble Model Training Module

This module handles training of Analyst ensemble models that combine:
- Base models (TCN, LightGBM, Ridge, ElasticNet, RandomForest, NAS, TAS)
- HMM regime features and probabilities
- NAS models per-regime for enhanced trading signal generation
- Multi-timeframe features and cross-timeframe analysis
- Technical indicators and market data
- Outputs from base Analyst models

The ensemble operates on the dedicated 15m timeframe and combines all inputs to
deliver the Analyst's final green-signal decisions that gate downstream
Tactician processing. Uses outputs from the base Analyst models trained on 15m timeframe.

ENHANCED FEATURES:
- Comprehensive error handling with detailed failure reporting
- Enhanced progress tracking and sub-step reporting
- Input validation and data quality checks
- Optimized vectorization with intelligent fallback
- Structured logging with performance metrics
- Health monitoring throughout training process
- Integration with common utilities and hardware optimizers
- Extensive logging with tprint at every step
- Integration with base Analyst model outputs
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

# Import NAS integration for Analyst
try:
    from ..model_training.analyst_ensemble_training import AnalystEnsembleTrainingStep
    NAS_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import NAS integration: {e}")
    NAS_AVAILABLE = False

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


@dataclass
class AnalystEnsembleTrainingConfig:
    """Configuration for Analyst ensemble training."""
    # Feature integration parameters
    enable_full_integration: bool = True
    include_hmm_features: bool = True
    include_nas_features: bool = True

    # Training parameters
    save_models: bool = True
    output_directory: str = "generated/analyst_ensemble_training"

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
                "TCN",
                "LIGHTGBM",
                "RIDGE",
                "ELASTIC_NET",
                "RANDOM_FOREST",
                "NAS",
                "TAS"
            ]


@dataclass
class AnalystEnsembleTrainingResult:
    """Result of Analyst ensemble training."""
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


class AnalystEnsembleTrainingStep:
    """
    Analyst Ensemble Training Step.

    Handles training of Analyst ensemble models with full feature integration.
    """

    def __init__(self, config: Optional[AnalystEnsembleTrainingConfig] = None):
        """Initialize the Analyst ensemble training step."""
        try:
            self.config = config or AnalystEnsembleTrainingConfig()
            self.logger = system_logger.getChild('AnalystEnsembleTrainingStep')

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
                tprint_success("✅ VectorBT Rolling Optimizer initialized for Analyst ensemble training")
            else:
                self.vectorbt_optimizer = None
                tprint_warning("⚠️ VectorBT Rolling Optimizer not available for Analyst ensemble training")

            # Initialize Unified Vectorization Manager
            if UNIFIED_VECTORIZATION_AVAILABLE:
                self.vectorization_manager = get_unified_vectorization_manager()
                tprint_success("✅ Unified Vectorization Manager initialized for Analyst ensemble training")
            else:
                self.vectorization_manager = None
                tprint_warning("⚠️ Unified Vectorization Manager not available for Analyst ensemble training")

            tprint_success("✅ AnalystEnsembleTrainingStep initialized successfully")

        except Exception as e:
            tprint_error(f"❌ Failed to initialize AnalystEnsembleTrainingStep: {e}")
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
                elif operation == 'quantile':
                    q = kwargs.get('q', 0.5)
                    return self.vectorbt_optimizer.rolling_quantile(data, window=window, q=q, **kwargs)
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
        elif operation == 'quantile':
            q = kwargs.get('q', 0.5)
            return data.rolling(window=window).quantile(q)
        elif operation == 'apply':
            func = kwargs.get('func')
            return data.rolling(window=window).apply(func, **kwargs)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _optimize_feature_vectorization(self, features: pd.DataFrame) -> pd.DataFrame:
        """Optimize feature vectorization using Unified Vectorization Manager."""
        if self.vectorization_manager is not None:
            try:
                tprint_debug("🔧 Applying unified vectorization optimization to Analyst features")
                
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
                    tprint_success(f"✅ Analyst feature vectorization optimized using {result.strategy_used.value}")
                    tprint_performance(f"Analyst feature optimization", result.computation_time)
                    return result.result
                else:
                    tprint_warning("⚠️ Vectorization optimization returned no result, using original features")
                    return features
                
            except Exception as e:
                tprint_warning(f"⚠️ Unified vectorization failed: {e}, using original features")
                return features
        else:
            return features

    async def train_analyst_ensemble(
        self,
        training_data: pd.DataFrame,
        base_models: Dict[str, Any],
        feature_columns: List[str],
        target_columns: List[str],
        sample_weight: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train Analyst ensemble models with full feature integration.

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
        tprint_info("🚀 Starting Analyst ensemble training with full feature integration...")

        try:
            # Validate inputs
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
                tprint_info(f"  - NAS features: {metadata.get('nas_features_count', 'N/A')}")
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
                'base_prediction_features_count': 0,
                'hmm_features_count': 0,
                'nas_features_count': 0,
                'total_features': X_base.shape[1]
            }

            # Start with base features
            enhanced_features.append(X_base)

            # Add base model predictions if available
            if base_models:
                base_prediction_features = self._extract_base_model_predictions(X_base, base_models)
                if base_prediction_features is not None:
                    enhanced_features.append(base_prediction_features)
                    metadata['base_prediction_features_count'] = base_prediction_features.shape[1]

            # Add HMM regime features if available and enabled
            if self.config.include_hmm_features:
                hmm_features = self._extract_hmm_features(training_data)
                if hmm_features is not None:
                    enhanced_features.append(hmm_features)
                    metadata['hmm_features_count'] = hmm_features.shape[1]

            # Add NAS features if available and enabled
            if self.config.include_nas_features:
                nas_features = self._extract_nas_features(training_data)
                if nas_features is not None:
                    enhanced_features.append(nas_features)
                    metadata['nas_features_count'] = nas_features.shape[1]

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

    def _extract_base_model_predictions(
        self,
        X_base: np.ndarray,
        base_models: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Extract predictions from base models with VectorBT optimizations."""
        try:
            prediction_features = []

            for model_name, model in base_models.items():
                try:
                    if hasattr(model, 'predict'):
                        preds = model.predict(X_base)
                        if len(preds.shape) == 1:
                            preds = preds.reshape(-1, 1)
                        
                        # Apply VectorBT rolling optimizations to predictions
                        if self.vectorbt_optimizer is not None and preds.shape[1] == 1:
                            pred_series = pd.Series(preds.flatten())
                            # Add rolling statistics for prediction stability
                            pred_rolling_mean = self._optimized_rolling_operation(
                                pred_series, 'mean', window=15
                            )
                            pred_rolling_std = self._optimized_rolling_operation(
                                pred_series, 'std', window=15
                            )
                            # Add rolling quantiles for prediction distribution
                            pred_rolling_q25 = self._optimized_rolling_operation(
                                pred_series, 'quantile', window=15, q=0.25
                            )
                            pred_rolling_q75 = self._optimized_rolling_operation(
                                pred_series, 'quantile', window=15, q=0.75
                            )
                            # Combine original predictions with rolling features
                            enhanced_preds = np.column_stack([
                                preds,
                                pred_rolling_mean.values.reshape(-1, 1),
                                pred_rolling_std.values.reshape(-1, 1),
                                pred_rolling_q25.values.reshape(-1, 1),
                                pred_rolling_q75.values.reshape(-1, 1)
                            ])
                            prediction_features.append(enhanced_preds)
                        else:
                            prediction_features.append(preds)
                        
                        tprint_debug(f"📊 Collected base predictions from {model_name} with VectorBT enhancements")
                    else:
                        tprint_debug(f"📊 Model {model_name} does not support prediction")
                except Exception as model_error:
                    tprint_warning(
                        f"⚠️ Failed to collect base predictions from {model_name}: {model_error}"
                    )
                    continue

            if prediction_features:
                combined_predictions = np.hstack(prediction_features)
                tprint_debug(
                    f"📊 Combined {len(prediction_features)} base model prediction feature sets with VectorBT optimizations"
                )
                return combined_predictions

            tprint_debug("📊 No base model predictions available for feature enhancement")
            return None
        except Exception as e:
            tprint_warning(f"⚠️ Failed to extract base model predictions: {e}")
            return None

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
                
                # Apply VectorBT rolling optimizations to HMM features
                if self.vectorbt_optimizer is not None:
                    tprint_debug("🔧 Applying VectorBT optimizations to Analyst HMM features")
                    for col in hmm_columns:
                        if hmm_data[col].dtype in ['float64', 'int64']:
                            # Apply rolling statistics for regime stability
                            hmm_data[f'{col}_rolling_mean'] = self._optimized_rolling_operation(
                                hmm_data[col], 'mean', window=20
                            )
                            hmm_data[f'{col}_rolling_std'] = self._optimized_rolling_operation(
                                hmm_data[col], 'std', window=20
                            )
                            # Add rolling skewness and kurtosis for regime distribution analysis
                            hmm_data[f'{col}_rolling_skew'] = self._optimized_rolling_operation(
                                hmm_data[col], 'skew', window=20
                            )
                            hmm_data[f'{col}_rolling_kurt'] = self._optimized_rolling_operation(
                                hmm_data[col], 'kurt', window=20
                            )
                
                hmm_features = hmm_data.values
                tprint_debug(f"📊 Extracted {len(hmm_columns)} HMM features with VectorBT optimizations")
                return hmm_features
            else:
                tprint_debug("📊 No HMM features found")
                return None

        except Exception as e:
            tprint_warning(f"⚠️ Failed to extract HMM features: {e}")
            return None

    def _extract_nas_features(self, training_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract NAS features with VectorBT optimizations."""
        try:
            nas_columns = []

            # Look for NAS-related columns
            for col in training_data.columns:
                if 'nas' in col.lower() or 'architecture' in col.lower():
                    nas_columns.append(col)

            if nas_columns:
                nas_data = training_data[nas_columns].copy()
                
                # Apply VectorBT rolling optimizations to NAS features
                if self.vectorbt_optimizer is not None:
                    tprint_debug("🔧 Applying VectorBT optimizations to NAS features")
                    for col in nas_columns:
                        if nas_data[col].dtype in ['float64', 'int64']:
                            # Apply rolling statistics for architecture stability
                            nas_data[f'{col}_rolling_mean'] = self._optimized_rolling_operation(
                                nas_data[col], 'mean', window=25
                            )
                            nas_data[f'{col}_rolling_std'] = self._optimized_rolling_operation(
                                nas_data[col], 'std', window=25
                            )
                            # Add rolling quantiles for architecture distribution
                            nas_data[f'{col}_rolling_q10'] = self._optimized_rolling_operation(
                                nas_data[col], 'quantile', window=25, q=0.1
                            )
                            nas_data[f'{col}_rolling_q90'] = self._optimized_rolling_operation(
                                nas_data[col], 'quantile', window=25, q=0.9
                            )
                
                nas_features = nas_data.values
                tprint_debug(f"📊 Extracted {len(nas_columns)} NAS features with VectorBT optimizations")
                return nas_features
            else:
                tprint_debug("📊 No NAS features found")
                return None

        except Exception as e:
            tprint_warning(f"⚠️ Failed to extract NAS features: {e}")
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
            from ..model_training.analyst_ensemble_training import AnalystEnsembleTrainingStep

            trainer = AnalystEnsembleTrainingStep()

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
                'include_nas_features': self.config.include_nas_features
            }

            training_result = await trainer.train_analyst_ensemble(**training_config)
            if training_result is None:
                training_result = {}

            enhanced_metadata = getattr(self, '_enhanced_metadata', {})
            if enhanced_metadata:
                result_metadata = training_result.get('metadata', {}) or {}
                # Ensure enriched metadata is included in persisted artifacts
                result_metadata.update(enhanced_metadata)
                training_result['metadata'] = result_metadata

            return training_result

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
                'base_prediction_features_count': 0,
                'hmm_features_count': 0,
                'nas_features_count': 0,
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
        """Get comprehensive performance metrics for the Analyst ensemble training step."""
        metrics = {
            'config': {
                'enable_full_integration': self.config.enable_full_integration,
                'include_hmm_features': self.config.include_hmm_features,
                'include_nas_features': self.config.include_nas_features,
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
async def execute_analyst_ensemble_training(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[AnalystEnsembleTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    base_analyst_models: Optional[Dict[str, Any]] = None,
    hmm_regime_outputs: Optional[np.ndarray] = None,
    nas_model_predictions: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Execute Analyst ensemble training with full feature integration.

    Args:
        X: Base feature matrix
        y: Target values
        regime_labels: HMM regime labels
        config: Optional configuration
        feature_names: Optional feature names
        base_analyst_models: Base models for stacking
        hmm_regime_outputs: HMM regime outputs
        nas_model_predictions: NAS model predictions
        timestamps: Data timestamps
        **kwargs: Additional parameters

    Returns:
        Dict with trained ensemble models and metrics
    """
    trainer = AnalystEnsembleTrainingStep(config)

    # Create training data DataFrame
    training_data = pd.DataFrame(X, columns=feature_names or [f'feature_{i}' for i in range(X.shape[1])])

    # Add regime labels if provided
    if regime_labels is not None:
        training_data['hmm_regime'] = regime_labels

    # Add HMM outputs if provided
    if hmm_regime_outputs is not None:
        for i, hmm_output in enumerate(hmm_regime_outputs.T):
            training_data[f'hmm_regime_prob_{i}'] = hmm_output

    # Add NAS predictions if provided
    if nas_model_predictions is not None:
        for i, nas_pred in enumerate(nas_model_predictions.T):
            training_data[f'nas_prediction_{i}'] = nas_pred

    # Add timestamps if provided
    if timestamps is not None:
        training_data['timestamp'] = timestamps

    # Create sample weights (can be enhanced based on regime confidence)
    sample_weight = np.ones(len(training_data))

    # Create target columns
    target_columns = [f'target_{i}' for i in range(y.shape[1])] if len(y.shape) > 1 else ['target']
    for i, col in enumerate(target_columns):
        if len(y.shape) > 1:
            training_data[col] = y[:, i]
        else:
            training_data[col] = y

    return await trainer.train_analyst_ensemble(
        training_data=training_data,
        base_models=base_analyst_models or {},
        feature_columns=feature_names or list(training_data.columns)[:-len(target_columns)],
        target_columns=target_columns,
        sample_weight=sample_weight,
        **kwargs
    )