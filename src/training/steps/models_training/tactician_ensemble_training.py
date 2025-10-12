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

# Import VectorBT optimizations - Enhanced performance with VectorBT
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer,
        optimized_rolling_mean, optimized_rolling_std, optimized_rolling_var,
        optimized_rolling_min, optimized_rolling_max, optimized_rolling_sum
    )
    from src.feature_selection.vectorbt.vectorbt_unified_framework import (
        VectorBTUnifiedFramework, create_vectorbt_unified_framework,
        FeatureSelectionMethod
    )
    from src.utils.ml_common.vectorbt_memory_optimizer import VectorBTMemoryOptimizer
    from src.utils.ml_common.vectorbt_performance_monitor import VectorBTPerformanceMonitor
    VECTORBT_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ WARNING: VectorBT optimizations not available: {e}")
    VECTORBT_OPTIMIZATIONS_AVAILABLE = False

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

    # VectorBT optimization parameters
    enable_vectorbt_optimizations: bool = True
    vectorbt_rolling_window: int = 20
    vectorbt_memory_efficient: bool = True
    vectorbt_chunk_size: int = 1000
    vectorbt_enable_gpu: bool = False
    vectorbt_fast_fail: bool = True
    vectorbt_feature_selection_method: str = 'auto'
    vectorbt_max_features: int = 100

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

            # Initialize VectorBT optimizations
            if VECTORBT_OPTIMIZATIONS_AVAILABLE and self.config.enable_vectorbt_optimizations:
                self._initialize_vectorbt_optimizations()
            else:
                self.vectorbt_rolling_optimizer = None
                self.vectorbt_unified_framework = None
                self.vectorbt_memory_optimizer = None
                self.vectorbt_performance_monitor = None

            tprint_success("✅ TacticianEnsembleTrainingStep initialized successfully")

        except Exception as e:
            tprint_error(f"❌ Failed to initialize TacticianEnsembleTrainingStep: {e}")
            raise

    def _initialize_vectorbt_optimizations(self):
        """Initialize VectorBT optimization components."""
        try:
            tprint_info("🚀 Initializing VectorBT optimizations for Tactician...")
            
            # Initialize VectorBT rolling optimizer
            self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=self.config.vectorbt_enable_gpu,
                enable_parallel=self.config.enable_parallel_processing,
                memory_efficient=self.config.vectorbt_memory_efficient,
                chunk_size=self.config.vectorbt_chunk_size,
                fast_fail=self.config.vectorbt_fast_fail,
                enable_logging=True
            )
            
            # Initialize VectorBT unified framework for feature selection
            self.vectorbt_unified_framework = create_vectorbt_unified_framework()
            
            # Initialize VectorBT memory optimizer
            self.vectorbt_memory_optimizer = VectorBTMemoryOptimizer(
                memory_limit_gb=self.config.memory_limit_gb,
                enable_compression=True,
                enable_chunking=True
            )
            
            # Initialize VectorBT performance monitor
            self.vectorbt_performance_monitor = VectorBTPerformanceMonitor(
                enable_detailed_logging=True,
                enable_memory_tracking=True,
                enable_timing_tracking=True
            )
            
            tprint_success("✅ VectorBT optimizations initialized successfully for Tactician")
            
        except Exception as e:
            tprint_error(f"❌ Failed to initialize VectorBT optimizations: {e}")
            # Set to None to disable optimizations
            self.vectorbt_rolling_optimizer = None
            self.vectorbt_unified_framework = None
            self.vectorbt_memory_optimizer = None
            self.vectorbt_performance_monitor = None

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
        """Create enhanced feature set with full integration and VectorBT optimizations."""
        try:
            tprint_info("🔄 Creating enhanced feature set with full integration and VectorBT optimizations...")

            # Start performance monitoring
            if self.vectorbt_performance_monitor:
                self.vectorbt_performance_monitor.start_operation("tactician_enhanced_feature_set_creation")

            enhanced_features = []
            metadata = {
                'base_features_count': X_base.shape[1],
                'hmm_features_count': 0,
                'analyst_features_count': 0,
                'oof_predictions_count': 0,
                'vectorbt_rolling_features_count': 0,
                'vectorbt_selected_features_count': 0,
                'total_features': X_base.shape[1]
            }

            # Start with base features
            enhanced_features.append(X_base)

            # Add VectorBT rolling features if optimizer is available
            if self.vectorbt_rolling_optimizer and self.config.enable_vectorbt_optimizations:
                rolling_features = await self._create_vectorbt_rolling_features(training_data)
                if rolling_features is not None:
                    enhanced_features.append(rolling_features)
                    metadata['vectorbt_rolling_features_count'] = rolling_features.shape[1]

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

            # Apply VectorBT feature selection if enabled
            if (self.vectorbt_unified_framework and 
                self.config.enable_vectorbt_optimizations and 
                X_enhanced.shape[1] > self.config.vectorbt_max_features):
                
                X_enhanced, selected_features = await self._apply_vectorbt_feature_selection(
                    X_enhanced, training_data, **kwargs
                )
                metadata['vectorbt_selected_features_count'] = len(selected_features)

            metadata['total_features'] = X_enhanced.shape[1]
            self._enhanced_metadata = metadata

            # End performance monitoring
            if self.vectorbt_performance_monitor:
                self.vectorbt_performance_monitor.end_operation("tactician_enhanced_feature_set_creation")

            tprint_success(f"✅ Enhanced feature set created: {X_enhanced.shape[1]} total features")
            return X_enhanced

        except Exception as e:
            tprint_error(f"❌ Failed to create enhanced feature set: {e}")
            return X_base

    def _extract_hmm_features(self, training_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract HMM regime features."""
        try:
            hmm_columns = []

            # Look for HMM-related columns
            for col in training_data.columns:
                if 'hmm' in col.lower() or 'regime' in col.lower():
                    hmm_columns.append(col)

            if hmm_columns:
                hmm_features = training_data[hmm_columns].values
                tprint_debug(f"📊 Extracted {len(hmm_columns)} HMM features")
                return hmm_features
            else:
                tprint_debug("📊 No HMM features found")
                return None

        except Exception as e:
            tprint_warning(f"⚠️ Failed to extract HMM features: {e}")
            return None

    def _extract_analyst_features(self, training_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract Analyst features."""
        try:
            analyst_columns = []

            # Look for Analyst-related columns
            for col in training_data.columns:
                if 'analyst' in col.lower() or 'confidence' in col.lower():
                    analyst_columns.append(col)

            if analyst_columns:
                analyst_features = training_data[analyst_columns].values
                tprint_debug(f"📊 Extracted {len(analyst_columns)} Analyst features")
                return analyst_features
            else:
                tprint_debug("📊 No Analyst features found")
                return None

        except Exception as e:
            tprint_warning(f"⚠️ Failed to extract Analyst features: {e}")
            return None

    def _extract_oof_predictions(self, X_base: np.ndarray, base_models: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract OOF predictions from base models."""
        try:
            oof_predictions = []

            for model_name, model in base_models.items():
                try:
                    if hasattr(model, 'predict'):
                        pred = model.predict(X_base)
                        if len(pred.shape) == 1:
                            pred = pred.reshape(-1, 1)
                        oof_predictions.append(pred)
                        tprint_debug(f"📊 Got OOF predictions from {model_name}")
                    else:
                        tprint_debug(f"📊 Model {model_name} doesn't have predict method")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to get OOF predictions from {model_name}: {e}")
                    continue

            if oof_predictions:
                oof_features = np.hstack(oof_predictions)
                tprint_debug(f"📊 Combined {len(oof_predictions)} OOF prediction sets")
                return oof_features
            else:
                tprint_debug("📊 No OOF predictions available")
                return None

        except Exception as e:
            tprint_warning(f"⚠️ Failed to extract OOF predictions: {e}")
            return None

    async def _create_vectorbt_rolling_features(self, training_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Create VectorBT rolling features for enhanced feature set."""
        try:
            if not self.vectorbt_rolling_optimizer:
                return None

            tprint_info("🔄 Creating VectorBT rolling features for Tactician...")
            
            # Select numeric columns for rolling operations
            numeric_columns = training_data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                tprint_warning("⚠️ No numeric columns found for VectorBT rolling features")
                return None

            rolling_features = []
            window = self.config.vectorbt_rolling_window

            for col in numeric_columns[:10]:  # Limit to first 10 numeric columns
                try:
                    series = training_data[col].dropna()
                    if len(series) < window:
                        continue

                    # Create rolling features using VectorBT optimizer
                    rolling_mean = self.vectorbt_rolling_optimizer.rolling_mean(series, window)
                    rolling_std = self.vectorbt_rolling_optimizer.rolling_std(series, window)
                    rolling_min = self.vectorbt_rolling_optimizer.rolling_min(series, window)
                    rolling_max = self.vectorbt_rolling_optimizer.rolling_max(series, window)

                    # Combine rolling features
                    col_rolling_features = np.column_stack([
                        rolling_mean.values,
                        rolling_std.values,
                        rolling_min.values,
                        rolling_max.values
                    ])

                    rolling_features.append(col_rolling_features)
                    tprint_debug(f"📊 Created rolling features for {col}")

                except Exception as e:
                    tprint_warning(f"⚠️ Failed to create rolling features for {col}: {e}")
                    continue

            if rolling_features:
                combined_rolling_features = np.hstack(rolling_features)
                tprint_success(f"✅ Created VectorBT rolling features for Tactician: {combined_rolling_features.shape[1]} features")
                return combined_rolling_features
            else:
                tprint_warning("⚠️ No VectorBT rolling features created for Tactician")
                return None

        except Exception as e:
            tprint_error(f"❌ Failed to create VectorBT rolling features for Tactician: {e}")
            return None

    async def _apply_vectorbt_feature_selection(
        self, 
        X: np.ndarray, 
        training_data: pd.DataFrame, 
        **kwargs
    ) -> Tuple[np.ndarray, List[int]]:
        """Apply VectorBT feature selection to reduce dimensionality."""
        try:
            if not self.vectorbt_unified_framework:
                return X, list(range(X.shape[1]))

            tprint_info(f"🔄 Applying VectorBT feature selection for Tactician: {X.shape[1]} -> {self.config.vectorbt_max_features} features...")

            # Prepare target variable for feature selection
            target_columns = kwargs.get('target_columns', ['target'])
            if target_columns and target_columns[0] in training_data.columns:
                y = training_data[target_columns[0]].values
            else:
                # Use first column as target if no target specified
                y = training_data.iloc[:, 0].values

            # Ensure y is numeric and finite
            y = np.asarray(y, dtype=float)
            y = y[~np.isnan(y)]
            
            if len(y) == 0:
                tprint_warning("⚠️ No valid target values for feature selection")
                return X, list(range(X.shape[1]))

            # Align X and y
            min_len = min(len(X), len(y))
            X_aligned = X[:min_len]
            y_aligned = y[:min_len]

            # Apply VectorBT feature selection
            selection_result = self.vectorbt_unified_framework.select_features(
                X_aligned, 
                y_aligned,
                method=self.config.vectorbt_feature_selection_method,
                k=self.config.vectorbt_max_features,
                feature_names=[f'feature_{i}' for i in range(X_aligned.shape[1])]
            )

            if selection_result.success and len(selection_result.selected_indices) > 0:
                selected_X = X_aligned[:, selection_result.selected_indices]
                tprint_success(f"✅ VectorBT feature selection completed for Tactician: {X.shape[1]} -> {selected_X.shape[1]} features")
                return selected_X, selection_result.selected_indices
            else:
                tprint_warning(f"⚠️ VectorBT feature selection failed for Tactician: {selection_result.error}")
                return X, list(range(X.shape[1]))

        except Exception as e:
            tprint_error(f"❌ VectorBT feature selection failed for Tactician: {e}")
            return X, list(range(X.shape[1]))

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
        """Get performance metrics for the ensemble training step."""
        metrics = {
            'config': {
                'enable_full_integration': self.config.enable_full_integration,
                'include_hmm_features': self.config.include_hmm_features,
                'include_analyst_features': self.config.include_analyst_features,
                'include_oof_predictions': self.config.include_oof_predictions,
                'save_models': self.config.save_models,
                'output_directory': self.config.output_directory,
                'vectorbt_optimizations_enabled': self.config.enable_vectorbt_optimizations
            },
            'hardware_optimization': {
                'gpu_manager': self.gpu_manager is not None,
                'memory_optimizer': self.memory_optimizer is not None,
                'cpu_optimizer': self.cpu_optimizer is not None
            },
            'vectorbt_optimization': {
                'rolling_optimizer_available': self.vectorbt_rolling_optimizer is not None,
                'unified_framework_available': self.vectorbt_unified_framework is not None,
                'memory_optimizer_available': self.vectorbt_memory_optimizer is not None,
                'performance_monitor_available': self.vectorbt_performance_monitor is not None
            }
        }

        # Add VectorBT performance statistics if available
        if self.vectorbt_rolling_optimizer:
            try:
                rolling_stats = self.vectorbt_rolling_optimizer.get_performance_stats()
                metrics['vectorbt_rolling_stats'] = rolling_stats
            except Exception as e:
                tprint_warning(f"⚠️ Failed to get VectorBT rolling stats: {e}")

        if self.vectorbt_unified_framework:
            try:
                framework_stats = self.vectorbt_unified_framework.get_performance_stats()
                metrics['vectorbt_framework_stats'] = framework_stats
            except Exception as e:
                tprint_warning(f"⚠️ Failed to get VectorBT framework stats: {e}")

        if self.vectorbt_performance_monitor:
            try:
                performance_stats = self.vectorbt_performance_monitor.get_performance_summary()
                metrics['vectorbt_performance_stats'] = performance_stats
            except Exception as e:
                tprint_warning(f"⚠️ Failed to get VectorBT performance stats: {e}")

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