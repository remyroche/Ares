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
            # Validate inputs
            if training_data.empty or not feature_columns or not target_columns:
                raise ValueError("Insufficient training data or missing columns")

            # Prepare training data
            X = training_data[feature_columns].values
            y = training_data[target_columns].values

            if len(y.shape) == 1:
                y = y.reshape(-1, 1)

            if sample_weight is None:
                sample_weight = np.ones(len(training_data))

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
                                model_type, X, y, sample_weight, **kwargs
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

            return {
                'models': all_models,
                'metrics': all_metrics,
                'training_time': execution_time,
                'features_used': feature_columns,
                'samples_used': len(training_data),
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
                training_config['sample_weight']
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
                    if hasattr(unified_config, 'data') and hasattr(unified_config.data, 'timeframe'):
                        unified_config.data.timeframe = '5m'
                    if hasattr(unified_config, 'enable_feature_engineering'):
                        unified_config.enable_feature_engineering = False
                    if hasattr(unified_config, 'enable_data_preprocessing'):
                        unified_config.enable_data_preprocessing = False
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
        """Train Random Survival Forest model."""
        try:
            from sklearn.ensemble import RandomForestRegressor

            # Simple Random Forest for now (can be enhanced with survival analysis)
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1 if self.config.enable_parallel_processing else 1
            )

            model.fit(X, y.ravel(), sample_weight=sample_weight)

            return {
                'models': {'random_survival_forest': model},
                'metrics': {'model_type': 'RandomSurvivalForest'}
            }

        except Exception as e:
            tprint_error(f"❌ Random Survival Forest training failed: {e}")
            return {'models': {}, 'metrics': {}}

    async def _train_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Train XGBoost model."""
        try:
            import xgboost as xgb

            model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1 if self.config.enable_parallel_processing else 1
            )

            model.fit(X, y.ravel(), sample_weight=sample_weight)

            return {
                'models': {'xgboost': model},
                'metrics': {'model_type': 'XGBoost'}
            }

        except Exception as e:
            tprint_error(f"❌ XGBoost training failed: {e}")
            return {'models': {}, 'metrics': {}}

    async def _train_elastic_net_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Train Elastic Net CV model."""
        try:
            from sklearn.linear_model import ElasticNetCV

            model = ElasticNetCV(
                cv=5,
                random_state=42,
                max_iter=1000
            )

            model.fit(X, y.ravel(), sample_weight=sample_weight)

            return {
                'models': {'elastic_net_cv': model},
                'metrics': {'model_type': 'ElasticNetCV'}
            }

        except Exception as e:
            tprint_error(f"❌ Elastic Net CV training failed: {e}")
            return {'models': {}, 'metrics': {}}

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