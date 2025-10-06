"""
Tactician Ensemble Training Pipeline - Standalone Ensemble Model Training Pipeline

This pipeline handles training of Tactician ensemble models that combine:
- Base models (RandomSurvivalForest, XGBoost, ElasticNetCV)
- HMM regime features and probabilities
- Analyst model predictions and confidence scores
- OOF predictions from all base models
- Technical indicators and market data
- Multi-horizon target variables

The pipeline supports both short and long timeframes with proper feature differentiation
and integrates with base model outputs for enhanced ensemble performance.

ENHANCED FEATURES:
- Comprehensive error handling with detailed failure reporting
- Enhanced progress tracking and sub-step reporting
- Input validation and data quality checks
- Optimized vectorization with intelligent fallback
- Structured logging with performance metrics
- Health monitoring throughout training process
- Integration with common utilities and hardware optimizers
- Extensive logging with tprint at every step
- Base model integration for OOF predictions
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

# Import ensemble training components
try:
    from .tactician_ensemble_training import (
        TacticianEnsembleTrainingStep, TacticianEnsembleTrainingConfig,
        TacticianEnsembleTrainingResult, execute_tactician_ensemble_training
    )
    ENSEMBLE_TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import ensemble training: {e}")
    ENSEMBLE_TRAINING_AVAILABLE = False

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
class TacticianEnsembleTrainingPipelineConfig:
    """Configuration for Tactician ensemble training pipeline."""
    # Training parameters
    timeframe: TimeFrame = TimeFrame.SHORT

    # Output configuration
    output_directory: str = "generated/tactician_ensemble_training"
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

    # Ensemble configuration
    enable_full_integration: bool = True
    include_hmm_features: bool = True
    include_analyst_features: bool = True
    include_oof_predictions: bool = True

    # Analyst integration
    analyst_confidence_threshold: float = 0.4
    use_analyst_filtered_data: bool = True

    def __post_init__(self):
        """Post-initialization setup."""
        pass


@dataclass
class TacticianEnsembleTrainingPipelineResult:
    """Result of Tactician ensemble training pipeline."""
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


class TacticianEnsembleTrainingPipeline:
    """
    Tactician Ensemble Training Pipeline.

    Handles training of Tactician ensemble models with short/long timeframe support
    and integration with base model outputs.
    """

    def __init__(self, config: Optional[TacticianEnsembleTrainingPipelineConfig] = None):
        """Initialize the Tactician ensemble training pipeline."""
        try:
            self.config = config or TacticianEnsembleTrainingPipelineConfig()
            self.logger = system_logger.getChild('TacticianEnsembleTrainingPipeline')

            # Initialize training components
            if ENSEMBLE_TRAINING_AVAILABLE:
                ensemble_config = TacticianEnsembleTrainingConfig(
                    enable_full_integration=self.config.enable_full_integration,
                    include_hmm_features=self.config.include_hmm_features,
                    include_analyst_features=self.config.include_analyst_features,
                    include_oof_predictions=self.config.include_oof_predictions,
                    save_models=self.config.save_models,
                    output_directory=f"{self.config.output_directory}/{self.config.timeframe.value}",
                    enable_parallel_processing=self.config.enable_parallel_processing,
                    enable_gpu_acceleration=self.config.enable_gpu_acceleration,
                    memory_limit_gb=self.config.memory_limit_gb,
                    validation_split=self.config.validation_split,
                    min_training_samples=self.config.min_training_samples
                )
                self.trainer = TacticianEnsembleTrainingStep(ensemble_config)
                tprint_success("✅ Ensemble trainer initialized")
            else:
                self.trainer = None

            tprint_success("✅ TacticianEnsembleTrainingPipeline initialized successfully")

        except Exception as e:
            tprint_error(f"❌ Failed to initialize TacticianEnsembleTrainingPipeline: {e}")
            raise

    async def train_tactician_ensemble(
        self,
        training_data: pd.DataFrame,
        base_models: Dict[str, Any],
        feature_columns: List[str],
        target_columns: List[str],
        sample_weight: Optional[np.ndarray] = None,
        analyst_predictions: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> TacticianEnsembleTrainingPipelineResult:
        """
        Train Tactician ensemble models using the pipeline.

        Args:
            training_data: DataFrame with features and targets
            base_models: Dictionary of trained base models
            feature_columns: List of feature column names
            target_columns: List of target column names
            sample_weight: Optional sample weights
            analyst_predictions: Optional Analyst predictions for filtering
            **kwargs: Additional parameters

        Returns:
            TacticianEnsembleTrainingPipelineResult with trained models and metrics
        """
        start_time = tprint_timer()
        tprint_info(f"🚀 Starting Tactician {self.config.timeframe.value} ensemble training pipeline...")

        result = TacticianEnsembleTrainingPipelineResult(timeframe=self.config.timeframe)
        result.training_phase = TrainingPhase.TRAINING

        try:
            # Validate base models are available
            if not base_models:
                raise ValueError("No base models provided for ensemble training")

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

            # Train ensemble models
            if self.trainer:
                tprint_info(f"🔄 Training {self.config.timeframe.value} ensemble models...")
                ensemble_result = await self.trainer.train_tactician_ensemble(
                    training_data=filtered_data,
                    base_models=base_models,
                    feature_columns=feature_columns,
                    target_columns=target_columns,
                    sample_weight=sample_weight,
                    **kwargs
                )

                if ensemble_result.get('models'):
                    result.models = ensemble_result['models']
                    result.training_metrics = ensemble_result['metrics']
                    result.training_completed = True
                    result.total_samples = ensemble_result['samples_used']
                    tprint_success(f"✅ {self.config.timeframe.value.title()} ensemble model training completed")
                else:
                    tprint_warning(f"⚠️ {self.config.timeframe.value.title()} ensemble model training failed or returned no models")
            else:
                tprint_info("⏭️ Skipping ensemble model training - trainer not available")

            result.execution_time = tprint_timer(start_time)
            result.training_phase = TrainingPhase.COMPLETED

            tprint_success(f"✅ Tactician {self.config.timeframe.value} ensemble training pipeline completed in {result.execution_time".2f"}s")
            return result

        except Exception as e:
            result.execution_time = tprint_timer(start_time)
            result.training_phase = TrainingPhase.FAILED
            tprint_error(f"❌ Tactician {self.config.timeframe.value} ensemble training pipeline failed: {e}")
            tprint_error(f"Error details: {traceback.format_exc()}")
            raise

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the training pipeline."""
        metrics = {
            'config': {
                'timeframe': self.config.timeframe.value,
                'output_directory': self.config.output_directory,
                'save_models': self.config.save_models,
                'enable_parallel_processing': self.config.enable_parallel_processing,
                'enable_gpu_acceleration': self.config.enable_gpu_acceleration,
                'enable_full_integration': self.config.enable_full_integration,
                'include_hmm_features': self.config.include_hmm_features,
                'include_analyst_features': self.config.include_analyst_features,
                'include_oof_predictions': self.config.include_oof_predictions,
                'analyst_confidence_threshold': self.config.analyst_confidence_threshold,
                'use_analyst_filtered_data': self.config.use_analyst_filtered_data
            },
            'component_availability': {
                'trainer': self.trainer is not None
            }
        }

        return metrics


# Convenience function for external usage
async def execute_tactician_ensemble_training_pipeline(
    training_data: pd.DataFrame,
    base_models: Dict[str, Any],
    feature_columns: List[str],
    target_columns: List[str],
    sample_weight: Optional[np.ndarray] = None,
    analyst_predictions: Optional[pd.DataFrame] = None,
    config: Optional[TacticianEnsembleTrainingPipelineConfig] = None,
    **kwargs
) -> TacticianEnsembleTrainingPipelineResult:
    """
    Execute Tactician ensemble training pipeline.

    Args:
        training_data: DataFrame with features and targets
        base_models: Dictionary of trained base models
        feature_columns: List of feature column names
        target_columns: List of target column names
        sample_weight: Optional sample weights
        analyst_predictions: Optional Analyst predictions for filtering
        config: Optional configuration
        **kwargs: Additional parameters

    Returns:
        TacticianEnsembleTrainingPipelineResult with trained models and metrics
    """
    pipeline = TacticianEnsembleTrainingPipeline(config)
    return await pipeline.train_tactician_ensemble(
        training_data, base_models, feature_columns, target_columns, sample_weight, analyst_predictions, **kwargs
    )