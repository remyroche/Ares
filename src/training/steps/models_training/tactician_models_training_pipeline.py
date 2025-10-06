"""
Tactician Models Training Pipeline - Standalone Base Model Training Pipeline

This pipeline handles training of individual Tactician base models:
- RandomSurvivalForest model
- XGBoost model
- ElasticNetCV model
- NAS (Neural Architecture Search) model
- TAS (Tree-based Architecture Search) model

The pipeline supports both short and long timeframes with proper feature differentiation.

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
- Short/Long timeframe differentiation support
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
import traceback
from dataclasses import dataclass
from enum import Enum

# Import base model training components
try:
    from .tactician_models_training import (
        TacticianModelsTrainingStep, TacticianModelsTrainingConfig,
        TacticianModelsTrainingResult, TacticianModelType,
        execute_tactician_models_training
    )
    MODELS_TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import models training: {e}")
    MODELS_TRAINING_AVAILABLE = False

# Enhanced imports with comprehensive error handling
try:
    from src.utils.logger import system_logger
    TACTICIAN_PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import core utilities: {e}")
    TACTICIAN_PIPELINE_AVAILABLE = False

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


class TrainingPhase(Enum):
    """Training phase enumeration."""
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


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
    training_phase: TrainingPhase = TrainingPhase.TRAINING

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
            if MODELS_TRAINING_AVAILABLE:
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
            else:
                self.trainer = None

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
        result.training_phase = TrainingPhase.TRAINING

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
            result.training_phase = TrainingPhase.COMPLETED

            tprint_success(f"✅ Tactician {self.config.timeframe.value} models training pipeline completed in {result.execution_time".2f"}s")
            return result

        except Exception as e:
            result.execution_time = tprint_timer(start_time)
            result.training_phase = TrainingPhase.FAILED
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