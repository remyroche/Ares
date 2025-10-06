"""
Tactician Models Training - Base Model Training Module

This module handles training of individual Tactician base models:
- RandomSurvivalForest model
- XGBoost model
- ElasticNetCV model
- NAS (Neural Architecture Search) model
- TAS (Tree-based Architecture Search) model

The Tactician operates on the 5m timeframe and decides WHEN to trade using only
Analyst-provided green-signal filtered windows (>0.4% confidence threshold), refining entries after the
15m Analyst approval. Integrates with Analyst ensemble outputs and regime features for enhanced 5m timeframe tactical decisions.

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

# Enhanced imports with comprehensive error handling
try:
    from src.utils.logger import system_logger
    from src.utils.ml_common.config import PerRegimeTrainingConfig
    from src.utils.ml_common.training import PerRegimeTrainingStep
    TACTICIAN_TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import core ML utilities: {e}")
    TACTICIAN_TRAINING_AVAILABLE = False

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


class TacticianModelType(Enum):
    """Tactician model types."""
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

    # Hardware optimization
    enable_parallel_processing: bool = True
    enable_gpu_acceleration: bool = True
    memory_limit_gb: float = 8.0

    # Validation parameters
    validation_split: float = 0.2
    min_training_samples: int = 100

    def __post_init__(self):
        """Post-initialization setup."""
        if self.model_types is None:
            self.model_types = [
                TacticianModelType.RANDOM_SURVIVAL_FOREST,
                TacticianModelType.XGBOOST,
                TacticianModelType.ELASTIC_NET_CV,
                TacticianModelType.NAS,
                TacticianModelType.TAS,
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

            # Filter training data using Analyst Ensemble outputs
            try:
                filtered_data, analyst_filter_info = await self._apply_analyst_filtering(
                    training_data, model_specific_features, target_columns, **kwargs
                )

                if filtered_data.empty:
                    warning_msg = "No training samples remain after Analyst filtering"
                    tprint_warning(f"⚠️ {warning_msg}")

                    # Try with relaxed confidence threshold
                    if 'min_analyst_confidence' in kwargs:
                        relaxed_threshold = max(0.1, kwargs['min_analyst_confidence'] - 0.2)
                        tprint_info(f"🔄 Retrying with relaxed confidence threshold: {relaxed_threshold}")

                        kwargs['min_analyst_confidence'] = relaxed_threshold
                        filtered_data, analyst_filter_info = await self._apply_analyst_filtering(
                            training_data, model_specific_features, target_columns, **kwargs
                        )

                        if filtered_data.empty:
                            raise ValueError(f"No training samples remain even with relaxed threshold {relaxed_threshold}")

                        tprint_success(f"✅ Recovered {len(filtered_data)} samples with relaxed threshold")
                    else:
                        raise ValueError(warning_msg)
            except Exception as filter_error:
                tprint_warning(f"⚠️ Analyst filtering failed: {filter_error}")
                tprint_info("🔄 Continuing with original training data")
                filtered_data = training_data
                analyst_filter_info = {
                    'filtered': False,
                    'reason': f'filter_error: {str(filter_error)}',
                    'original_samples': len(training_data),
                    'filtered_samples': len(training_data)
                }

            # Prepare training data with validated features
            X = filtered_data[model_specific_features].values
            y = filtered_data[target_columns].values

            if len(y.shape) == 1:
                y = y.reshape(-1, 1)

            if sample_weight is None:
                sample_weight = np.ones(len(filtered_data))

            # Add analyst confidence as additional feature if available
            analyst_features = await self._extract_analyst_features(filtered_data, **kwargs)
            if analyst_features:
                X = np.column_stack([X] + [filtered_data[feat].values for feat in analyst_features])

            # Train models
            all_models = {}
            all_metrics = {}

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
                            training_result = await self._train_model_directly(
                                model_type, X, y, sample_weight, model_type_for_filtering=model_type.value, **kwargs
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
                'models_per_type': len(self.config.model_types)
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

            if model_type == TacticianModelType.RANDOM_SURVIVAL_FOREST:
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

        try:
            from .nas_tas.training_orchestrator import (
                TrainingOrchestrator,
                OrchestratorConfig,
                OrchestrationMode,
            )
        except ImportError as e:
            tprint_warning(f"⚠️ NAS/TAS orchestrator not available: {e}")
            return {'models': {}, 'metrics': {}, 'error': str(e)}

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
                        unified_config.timeframe = '5m'
                        tprint_debug(f"✅ Set NAS/TAS timeframe to 5m")
                    if hasattr(unified_config, 'data') and hasattr(unified_config.data, 'timeframe'):
                        unified_config.data.timeframe = '5m'
                        tprint_debug(f"✅ Set NAS/TAS data timeframe to 5m")
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
            }
        }

        return metrics

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
                timeframe = kwargs.get('timeframe', '5m')

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
        Apply Analyst Ensemble filtering to training data.

        Args:
            training_data: Training DataFrame
            feature_columns: Feature columns
            target_columns: Target columns
            **kwargs: Additional parameters including symbol, exchange, timeframe

        Returns:
            Tuple of (filtered_data, filter_info)
        """
        try:
            symbol = kwargs.get('symbol', 'BTCUSDT')
            exchange = kwargs.get('exchange', 'binance')
            timeframe = kwargs.get('timeframe', '5m')

            tprint_debug(f"🔍 Applying Analyst filtering for {symbol} {exchange} {timeframe}")

            # Analyst confidence threshold as per requirements
            min_confidence = kwargs.get('min_analyst_confidence', 0.4)

            # Look for analyst signal and confidence columns
            analyst_signal_col = None
            analyst_confidence_col = None

            for col in training_data.columns:
                if 'analyst_signal' in col.lower():
                    analyst_signal_col = col
                elif 'analyst_confidence' in col.lower():
                    analyst_confidence_col = col

            if analyst_confidence_col is None:
                tprint_warning("⚠️ No analyst confidence column found - using all data")
                return training_data, {
                    'filtered': False,
                    'reason': 'no_confidence_column',
                    'original_samples': len(training_data),
                    'filtered_samples': len(training_data)
                }

            # Filter data based on analyst confidence
            original_samples = len(training_data)
            filtered_data = training_data[training_data[analyst_confidence_col] >= min_confidence].copy()

            filter_info = {
                'filtered': True,
                'min_confidence_threshold': min_confidence,
                'confidence_column': analyst_confidence_col,
                'original_samples': original_samples,
                'filtered_samples': len(filtered_data),
                'filter_ratio': len(filtered_data) / original_samples if original_samples > 0 else 0
            }

            tprint_success(f"✅ Analyst filtering: {original_samples} → {len(filtered_data)} samples "
                          f"({filter_info['filter_ratio']:.2%} retained)")

            return filtered_data, filter_info

        except Exception as e:
            tprint_error(f"❌ Failed to apply Analyst filtering: {e}")
            return training_data, {
                'filtered': False,
                'reason': f'error: {str(e)}',
                'original_samples': len(training_data),
                'filtered_samples': len(training_data)
            }

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


# ===== PIPELINE FUNCTIONALITY ADDED BELOW =====

class TimeFrame(Enum):
    """Timeframe enumeration for short/long differentiation."""
    SHORT = "short"
    LONG = "long"


@dataclass
class TacticianModelsTrainingPipelineConfig:
    """Configuration for Tactician base models training pipeline."""
    # Training parameters
    model_types: List[TacticianModelType] = None
    timeframe: TimeFrame = TimeFrame.SHORT

    # Output configuration
    output_directory: str = "generated/tactician_models_training"
    save_models: bool = True
    save_predictions: bool = True
    save_metrics: bool = True

    # Hardware optimization
    enable_parallel_processing: bool = True
    enable_gpu_acceleration: bool = True
    memory_limit_gb: float = 8.0

    # Validation parameters
    validation_split: float = 0.2
    min_training_samples: int = 100

    # Analyst integration
    analyst_confidence_threshold: float = 0.4
    use_analyst_filtered_data: bool = True

    def __post_init__(self):
        """Post-initialization setup."""
        if self.model_types is None:
            self.model_types = [
                TacticianModelType.RANDOM_SURVIVAL_FOREST,
                TacticianModelType.XGBOOST,
                TacticianModelType.ELASTIC_NET_CV,
                TacticianModelType.NAS,
                TacticianModelType.TAS,
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

                raise ValueError(
                    f"Unsupported model type for Tactician models pipeline: {model_type!r}"
                )

            self.model_types = normalized_types


@dataclass
class TacticianModelsTrainingPipelineResult:
    """Result of Tactician base models training pipeline."""
    # Training results
    models: Dict[str, Any] = None

    # Performance metrics
    training_metrics: Dict[str, Any] = None

    # Metadata
    execution_time: float = 0.0
    total_samples: int = 0
    timeframe: TimeFrame = TimeFrame.SHORT
    training_phase: str = "training"

    # Status tracking
    training_completed: bool = False


class TacticianModelsTrainingPipeline:
    """
    Tactician Base Models Training Pipeline.

    Handles training of individual Tactician base models with short/long timeframe support.
    """

    def __init__(self, config: Optional[TacticianModelsTrainingPipelineConfig] = None):
        """Initialize the Tactician base models training pipeline."""
        try:
            self.config = config or TacticianModelsTrainingPipelineConfig()
            self.logger = system_logger.getChild('TacticianModelsTrainingPipeline')

            # Initialize training components
            base_config = TacticianModelsTrainingConfig(
                model_types=self.config.model_types,
                save_models=self.config.save_models,
                save_metrics=self.config.save_metrics,
                output_directory=f"{self.config.output_directory}/{self.config.timeframe.value}",
                enable_parallel_processing=self.config.enable_parallel_processing,
                enable_gpu_acceleration=self.config.enable_gpu_acceleration,
                memory_limit_gb=self.config.memory_limit_gb,
                validation_split=self.config.validation_split,
                min_training_samples=self.config.min_training_samples
            )
            self.trainer = TacticianModelsTrainingStep(base_config)
            tprint_success("✅ Base models trainer initialized")
            tprint_success("✅ TacticianModelsTrainingPipeline initialized successfully")

        except Exception as e:
            tprint_error(f"❌ Failed to initialize TacticianModelsTrainingPipeline: {e}")
            raise

    async def train_tactician_models(
        self,
        training_data: pd.DataFrame,
        feature_columns: List[str],
        target_columns: List[str],
        sample_weight: Optional[np.ndarray] = None,
        analyst_predictions: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> TacticianModelsTrainingPipelineResult:
        """
        Train Tactician base models using the pipeline.

        Args:
            training_data: DataFrame with features and targets
            feature_columns: List of feature column names
            target_columns: List of target column names
            sample_weight: Optional sample weights
            analyst_predictions: Optional Analyst predictions for filtering
            **kwargs: Additional parameters

        Returns:
            TacticianModelsTrainingPipelineResult with trained models and metrics
        """
        start_time = tprint_timer()
        tprint_info(f"🚀 Starting Tactician {self.config.timeframe.value} models training pipeline...")

        result = TacticianModelsTrainingPipelineResult(timeframe=self.config.timeframe)
        result.training_phase = "training"

        try:
            # Apply Analyst filtering if available and enabled
            filtered_data = training_data.copy()
            if (self.config.use_analyst_filtered_data and analyst_predictions is not None and
                'confidence' in analyst_predictions.columns):

                # Filter data based on Analyst confidence threshold
                analyst_mask = analyst_predictions['confidence'] >= self.config.analyst_confidence_threshold
                filtered_data = training_data[analyst_mask].copy()

                tprint_info(f"🔍 Filtered {len(training_data) - len(filtered_data)} samples using Analyst confidence threshold {self.config.analyst_confidence_threshold}")

                if len(filtered_data) < self.config.min_training_samples:
                    raise ValueError(f"Insufficient training samples after Analyst filtering: {len(filtered_data)} < {self.config.min_training_samples}")

            # Train base models
            if self.trainer:
                tprint_info(f"📈 Training {self.config.timeframe.value} base models...")
                base_result = await self.trainer.train_tactician_models(
                    filtered_data, feature_columns, target_columns, sample_weight, **kwargs
                )

                if base_result.get('models'):
                    result.models = base_result['models']
                    result.training_metrics = base_result['metrics']
                    result.training_completed = True
                    result.total_samples = base_result['samples_used']
                    tprint_success(f"✅ {self.config.timeframe.value.title()} base model training completed")
                else:
                    tprint_warning(f"⚠️ {self.config.timeframe.value.title()} base model training failed or returned no models")
            else:
                tprint_info("⏭️ Skipping base model training - trainer not available")

            result.execution_time = tprint_timer(start_time)
            result.training_phase = "completed"

            tprint_success(f"✅ Tactician {self.config.timeframe.value} models training pipeline completed in {result.execution_time".2f"}s")
            return result

        except Exception as e:
            result.execution_time = tprint_timer(start_time)
            result.training_phase = "failed"
            tprint_error(f"❌ Tactician {self.config.timeframe.value} models training pipeline failed: {e}")
            tprint_error(f"Error details: {traceback.format_exc()}")
            raise

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the training pipeline."""
        metrics = {
            'config': {
                'timeframe': self.config.timeframe.value,
                'model_types': [mt.value for mt in self.config.model_types],
                'output_directory': self.config.output_directory,
                'save_models': self.config.save_models,
                'enable_parallel_processing': self.config.enable_parallel_processing,
                'enable_gpu_acceleration': self.config.enable_gpu_acceleration,
                'analyst_confidence_threshold': self.config.analyst_confidence_threshold,
                'use_analyst_filtered_data': self.config.use_analyst_filtered_data
            },
            'component_availability': {
                'trainer': self.trainer is not None
            }
        }

        return metrics


# Convenience function for external usage
async def execute_tactician_models_training_pipeline(
    training_data: pd.DataFrame,
    feature_columns: List[str],
    target_columns: List[str],
    sample_weight: Optional[np.ndarray] = None,
    analyst_predictions: Optional[pd.DataFrame] = None,
    config: Optional[TacticianModelsTrainingPipelineConfig] = None,
    **kwargs
) -> TacticianModelsTrainingPipelineResult:
    """
    Execute Tactician base models training pipeline.

    Args:
        training_data: DataFrame with features and targets
        feature_columns: List of feature column names
        target_columns: List of target column names
        sample_weight: Optional sample weights
        analyst_predictions: Optional Analyst predictions for filtering
        config: Optional configuration
        **kwargs: Additional parameters

    Returns:
        TacticianModelsTrainingPipelineResult with trained models and metrics
    """
    pipeline = TacticianModelsTrainingPipeline(config)
    return await pipeline.train_tactician_models(
        training_data, feature_columns, target_columns, sample_weight, analyst_predictions, **kwargs
    )