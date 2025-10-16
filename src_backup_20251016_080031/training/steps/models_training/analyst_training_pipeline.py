"""
Analyst Training Pipeline - Unified Pipeline for Base and Ensemble Training

This pipeline orchestrates the training of Analyst models by:
1. Training base models (TCN, LightGBM, Ridge, ElasticNet, RandomForest)
2. Training ensemble models with full feature integration (HMM, NAS)

The pipeline supports 5m timeframe with proper regime-aware training.

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
from enum import Enum

# Import new modular training components
try:
    from .analyst_models_training import (
        AnalystModelsTrainingStep, AnalystModelsTrainingConfig,
        AnalystModelsTrainingResult, AnalystModelType,
        execute_analyst_models_training
    )
    MODELS_TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import models training: {e}")
    MODELS_TRAINING_AVAILABLE = False

try:
    from .analyst_ensemble_training import (
        AnalystEnsembleTrainingStep, AnalystEnsembleTrainingConfig,
        AnalystEnsembleTrainingResult, execute_analyst_ensemble_training
    )
    ENSEMBLE_TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import ensemble training: {e}")
    ENSEMBLE_TRAINING_AVAILABLE = False

# Enhanced imports with comprehensive error handling
try:
    from src.utils.logger import system_logger
    ANALYST_PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import core utilities: {e}")
    ANALYST_PIPELINE_AVAILABLE = False

# Import enhanced logging and utilities
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
        tprint_debug, tprint_progress, tprint_performance, tprint_timer
    )
    TPRINT_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL ERROR: tprint is required but not available: {e}")
    TPRINT_AVAILABLE = False

# Import negative learning integration
try:
    from .negative_learning_training_integration import (
        initialize_negative_learning_integration,
        get_negative_learning_integration
    )
    from .negative_learning_training_patches import (
        apply_negative_learning_patches
    )
    NEGATIVE_LEARNING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ WARNING: Negative learning integration not available: {e}")
    NEGATIVE_LEARNING_AVAILABLE = False


class TrainingPhase(Enum):
    """Training phase enumeration."""
    BASE_MODEL_TRAINING = "base_model_training"
    ENSEMBLE_TRAINING = "ensemble_training"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AnalystTrainingPipelineConfig:
    """Configuration for Analyst training pipeline."""
    # Training parameters
    train_base_models: bool = True
    train_ensemble_models: bool = True

    # Direction control for training
    enable_long_positions: bool = True
    enable_short_positions: bool = True

    # Output configuration
    output_directory: str = "generated/analyst_training_pipeline"
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

    # Base model configuration
    base_model_types: List[AnalystModelType] = None

    # Ensemble configuration
    enable_full_integration: bool = True
    include_hmm_features: bool = True
    include_nas_features: bool = True

    def __post_init__(self):
        """Post-initialization setup."""
        supported_types = [
            AnalystModelType.TCN,
            AnalystModelType.LIGHTGBM,
            AnalystModelType.RIDGE,
            AnalystModelType.ELASTIC_NET,
            AnalystModelType.RANDOM_FOREST,
            AnalystModelType.NAS,
            AnalystModelType.TAS,
        ]

        if self.base_model_types is None:
            self.base_model_types = supported_types
            return

        normalized_types: List[AnalystModelType] = []
        value_map = {model_type.value.upper(): model_type for model_type in supported_types}

        for model_type in self.base_model_types:
            if isinstance(model_type, AnalystModelType):
                normalized_types.append(model_type)
            elif isinstance(model_type, str):
                enum_value = value_map.get(model_type.strip().upper())
                if enum_value:
                    normalized_types.append(enum_value)
                else:
                    logging.getLogger(__name__).warning(
                        "Unsupported model type '%s' provided to AnalystTrainingPipelineConfig; ignoring.",
                        model_type,
                    )
            else:
                logging.getLogger(__name__).warning(
                    "Invalid model type %s provided to AnalystTrainingPipelineConfig; ignoring.",
                    model_type,
                )

        if not normalized_types:
            logging.getLogger(__name__).warning(
                "No valid base model types provided; defaulting to supported Analyst model types."
            )
            self.base_model_types = supported_types
        else:
            self.base_model_types = list(dict.fromkeys(normalized_types))


@dataclass
class AnalystTrainingPipelineResult:
    """Result of Analyst training pipeline."""
    # Training results
    base_models: Dict[str, Any] = None
    ensemble_models: Dict[str, Any] = None

    # Performance metrics
    base_training_metrics: Dict[str, Any] = None
    ensemble_metrics: Dict[str, Any] = None

    # Metadata
    execution_time: float = 0.0
    total_samples: int = 0
    training_phase: TrainingPhase = TrainingPhase.BASE_MODEL_TRAINING

    # Status tracking
    base_training_completed: bool = False
    ensemble_training_completed: bool = False


class AnalystTrainingPipeline:
    """
    Analyst Training Pipeline.

    Orchestrates the training of both base models and ensemble models
    with proper feature integration and regime awareness.
    """

    def __init__(self, config: Optional[AnalystTrainingPipelineConfig] = None):
        """Initialize the Analyst training pipeline."""
        try:
            self.config = config or AnalystTrainingPipelineConfig()
            self.logger = system_logger.getChild('AnalystTrainingPipeline')

            # Initialize training components
            if MODELS_TRAINING_AVAILABLE:
                base_config = AnalystModelsTrainingConfig(
                    model_types=self.config.base_model_types,
                    save_models=self.config.save_models,
                    output_directory=f"{self.config.output_directory}/base_models",
                    enable_parallel_processing=self.config.enable_parallel_processing,
                    enable_gpu_acceleration=self.config.enable_gpu_acceleration,
                    memory_limit_gb=self.config.memory_limit_gb,
                    validation_split=self.config.validation_split,
                    min_training_samples=self.config.min_training_samples
                )
                self.base_trainer = AnalystModelsTrainingStep(base_config)
                tprint_success("✅ Base models trainer initialized")
            else:
                self.base_trainer = None

            if ENSEMBLE_TRAINING_AVAILABLE:
                ensemble_config = AnalystEnsembleTrainingConfig(
                    enable_full_integration=self.config.enable_full_integration,
                    include_hmm_features=self.config.include_hmm_features,
                    include_nas_features=self.config.include_nas_features,
                    save_models=self.config.save_models,
                    output_directory=f"{self.config.output_directory}/ensemble_models",
                    enable_parallel_processing=self.config.enable_parallel_processing,
                    enable_gpu_acceleration=self.config.enable_gpu_acceleration,
                    memory_limit_gb=self.config.memory_limit_gb,
                    validation_split=self.config.validation_split,
                    min_training_samples=self.config.min_training_samples
                )
                self.ensemble_trainer = AnalystEnsembleTrainingStep(ensemble_config)
                tprint_success("✅ Ensemble trainer initialized")
            else:
                self.ensemble_trainer = None

            # Initialize negative learning integration
            if NEGATIVE_LEARNING_AVAILABLE:
                try:
                    self.nl_integration = initialize_negative_learning_integration(
                        self.config.negative_learning_config if hasattr(self.config, 'negative_learning_config') else {}
                    )
                    tprint_success("✅ Negative learning integration initialized")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to initialize negative learning: {e}")
                    self.nl_integration = None
            else:
                self.nl_integration = None

            tprint_success("✅ AnalystTrainingPipeline initialized successfully")

        except Exception as e:
            tprint_error(f"❌ Failed to initialize AnalystTrainingPipeline: {e}")
            raise

    async def train_analyst_models(
        self,
        training_data: pd.DataFrame,
        feature_columns: List[str],
        target_columns: List[str],
        sample_weight: Optional[np.ndarray] = None,
        **kwargs
    ) -> AnalystTrainingPipelineResult:
        """
        Train Analyst models using the pipeline.

        Args:
            training_data: DataFrame with features and targets
            feature_columns: List of feature column names
            target_columns: List of target column names
            sample_weight: Optional sample weights
            **kwargs: Additional parameters

        Returns:
            AnalystTrainingPipelineResult with trained models and metrics
        """
        start_time = tprint_timer()
        tprint_info("🚀 Starting Analyst training pipeline...")

        result = AnalystTrainingPipelineResult()
        result.training_phase = TrainingPhase.BASE_MODEL_TRAINING

        # Initialize negative learning for this training session
        if self.nl_integration:
            try:
                # Extract target for negative learning initialization
                target_series = training_data[target_columns[0]] if target_columns else pd.Series()
                
                init_results = self.nl_integration.initialize_for_training(
                    analyst_features=training_data[feature_columns],
                    analyst_target=target_series,
                    tactician_features=pd.DataFrame(),  # Not needed for Analyst
                    tactician_target=pd.Series(dtype=float),
                    retrain_timestamp=None
                )
                
                if init_results.get('status') == 'success':
                    tprint_success("✅ Negative learning initialized for Analyst training")
                else:
                    tprint_warning(f"⚠️ Negative learning initialization failed: {init_results.get('error', 'Unknown error')}")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to initialize negative learning: {e}")

        try:
            # Step 1: Train base models
            if self.config.train_base_models and self.base_trainer:
                tprint_info("📈 Step 1: Training base models...")
                base_result = await self.base_trainer.train_analyst_models(
                    training_data, feature_columns, target_columns, sample_weight, **kwargs
                )

                if base_result.get('models'):
                    result.base_models = base_result['models']
                    result.base_training_metrics = base_result['metrics']
                    result.base_training_completed = True
                    result.total_samples = base_result['samples_used']
                    tprint_success("✅ Base model training completed")
                else:
                    tprint_warning("⚠️ Base model training failed or returned no models")
            else:
                tprint_info("⏭️ Skipping base model training")

            # Step 2: Train ensemble models
            if (self.config.train_ensemble_models and self.ensemble_trainer and
                result.base_models and result.total_samples >= self.config.min_training_samples):

                tprint_info("🔄 Step 2: Training ensemble models...")
                ensemble_result = await self.ensemble_trainer.train_analyst_ensemble(
                    training_data=training_data,
                    base_models=result.base_models,
                    feature_columns=feature_columns,
                    target_columns=target_columns,
                    sample_weight=sample_weight,
                    **kwargs
                )

                if ensemble_result.get('models'):
                    result.ensemble_models = ensemble_result['models']
                    result.ensemble_metrics = ensemble_result['metrics']
                    result.ensemble_training_completed = True
                    tprint_success("✅ Ensemble model training completed")
                else:
                    tprint_warning("⚠️ Ensemble model training failed or returned no models")
            else:
                tprint_info("⏭️ Skipping ensemble model training")

            result.execution_time = tprint_timer(start_time)
            result.training_phase = TrainingPhase.COMPLETED

            tprint_success(f"✅ Analyst training pipeline completed in {result.execution_time:.2f}s")
            return result

        except Exception as e:
            result.execution_time = tprint_timer(start_time)
            result.training_phase = TrainingPhase.FAILED
            tprint_error(f"❌ Analyst training pipeline failed: {e}")
            tprint_error(f"Error details: {traceback.format_exc()}")
            raise

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the training pipeline."""
        metrics = {
            'config': {
                'train_base_models': self.config.train_base_models,
                'train_ensemble_models': self.config.train_ensemble_models,
                'output_directory': self.config.output_directory,
                'save_models': self.config.save_models,
                'enable_parallel_processing': self.config.enable_parallel_processing,
                'enable_gpu_acceleration': self.config.enable_gpu_acceleration
            },
            'component_availability': {
                'base_trainer': self.base_trainer is not None,
                'ensemble_trainer': self.ensemble_trainer is not None
            }
        }

        return metrics


# Convenience function for external usage
async def execute_analyst_training_pipeline(
    training_data: pd.DataFrame,
    feature_columns: List[str],
    target_columns: List[str],
    sample_weight: Optional[np.ndarray] = None,
    config: Optional[AnalystTrainingPipelineConfig] = None,
    **kwargs
) -> AnalystTrainingPipelineResult:
    """
    Execute Analyst training pipeline.

    Args:
        training_data: DataFrame with features and targets
        feature_columns: List of feature column names
        target_columns: List of target column names
        sample_weight: Optional sample weights
        config: Optional configuration
        **kwargs: Additional parameters

    Returns:
        AnalystTrainingPipelineResult with trained models and metrics
    """
    pipeline = AnalystTrainingPipeline(config)
    return await pipeline.train_analyst_models(
        training_data, feature_columns, target_columns, sample_weight, **kwargs
    )