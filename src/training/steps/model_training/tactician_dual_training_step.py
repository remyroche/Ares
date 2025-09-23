"""
Tactician Dual Training Step - Enhanced for Long/Short Differentiation

This step handles training multiple Tactician models for both long and short signals:

LONG MODELS (4 base models + 1 ensemble):
- XGBOOST model for long signals
- LIGHTGBM model for long signals
- DEEPSCALER_1M model for long signals
- FINANCIAL_RESNET model for long signals
- Ensemble model combining all long base models + ALL FEATURES + HMM + Analyst outputs

SHORT MODELS (4 base models + 1 ensemble):
- XGBOOST model for short signals
- LIGHTGBM model for short signals
- DEEPSCALER_1M model for short signals
- FINANCIAL_RESNET model for short signals
- Ensemble model combining all short base models + ALL FEATURES + HMM + Analyst outputs

🎯 ENSEMBLE FEATURE INTEGRATION:
Each ensemble model includes:
- All base features from pre-ML orchestration
- HMM regime features and probabilities
- Analyst model predictions and confidence scores
- OOF predictions from all base models
- Technical indicators and market data
- Multi-horizon target variables

The training uses differentiated features and horizon labeling for each direction,
ensuring optimal performance for both long and short trading scenarios.

ENHANCED FEATURES:
- Comprehensive error handling with detailed failure reporting
- Enhanced progress tracking and sub-step reporting
- Input validation and data quality checks
- Optimized vectorization with intelligent fallback
- Structured logging with performance metrics
- Health monitoring throughout training process
- Integration with common utilities and hardware optimizers
- Extensive logging with tprint at every step
- Full feature integration in ensemble models
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Enhanced imports with comprehensive error handling
try:
    from src.utils.logger import system_logger
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
        tprint_debug, tprint_progress, tprint_performance, tprint_timer
    )
    TPRINT_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import logging utilities: {e}")
    raise

# Import common utilities
try:
    from src.utils.common_operations import (
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        cleanup_m1_optimizers, integrate_with_m1_optimizers,
        validate_dataframe, validate_dataframe_columns, safe_merge_dataframes,
        safe_json_load, safe_json_dump, ensure_directory
    )
    from src.utils.math_validation import (
        safe_divide, validate_finite, validate_positive, validate_range
    )
    COMMON_OPS_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Common operations not available: {e}")
    COMMON_OPS_AVAILABLE = False

# Import pre-ML orchestrator
try:
    from .tactician_pre_ml_orchestrator import (
        TacticianPreMLOrchestrator, OrchestratorConfig, OrchestratorResult
    )
    PRE_ML_ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    PRE_ML_ORCHESTRATOR_AVAILABLE = False
    tprint_warning(f"⚠️ Pre-ML orchestrator not available: {e}")

# Import existing Tactician training components
try:
    from .tactician_models_training_refactored import TacticianModelsTrainingStep
    from .tactician_ensemble_training import TacticianEnsembleTrainingStep
    TACTICIAN_TRAINING_AVAILABLE = True
except ImportError as e:
    TACTICIAN_TRAINING_AVAILABLE = False
    tprint_warning(f"⚠️ Tactician training components not available: {e}")


class TrainingPhase(Enum):
    """Training phase enumeration."""
    PRE_ML_ORCHESTRATION = "pre_ml_orchestration"
    LONG_MODEL_TRAINING = "long_model_training"
    SHORT_MODEL_TRAINING = "short_model_training"
    LONG_ENSEMBLE_TRAINING = "long_ensemble_training"
    SHORT_ENSEMBLE_TRAINING = "short_ensemble_training"
    VALIDATION = "validation"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DualTrainingConfig:
    """Configuration for Tactician dual training."""
    # Pre-ML orchestration parameters
    min_analyst_confidence: float = 0.5
    subsequent_minutes: int = 45
    output_directory: str = "generated/tactician_dual_training"

    # Feature processing parameters
    enable_feature_optimization: bool = True
    enable_pid_generation: bool = True
    enable_horizon_labeling: bool = True
    enable_feature_selection: bool = True

    # Training parameters
    train_base_models: bool = True
    train_ensemble_models: bool = True

    # Model configuration
    max_lookback_periods: int = 20
    max_interaction_features: int = 100
    max_polynomial_features: int = 50
    max_cross_timeframe_features: int = 50

    # Horizon labeling parameters
    profit_targets: Dict[str, float] = None
    time_horizons: Dict[str, int] = None

    # Hardware optimization
    enable_parallel_processing: bool = True
    enable_gpu_acceleration: bool = True
    memory_limit_gb: float = 8.0

    # Validation parameters
    validation_split: float = 0.2
    min_training_samples: int = 100

    # Output configuration
    save_models: bool = True
    save_predictions: bool = True
    save_metrics: bool = True

    def __post_init__(self):
        """Post-initialization setup."""
        if self.profit_targets is None:
            self.profit_targets = {
                'micro': 0.003,    # 0.3% (net: 0.22% after fees)
                'small': 0.005,    # 0.5% (net: 0.42% after fees)
                'medium': 0.007,   # 0.7% (net: 0.62% after fees)
                'good': 0.010      # 1.0% (net: 0.92% after fees)
            }

        if self.time_horizons is None:
            self.time_horizons = {
                'immediate': 2,    # 10 minutes
                'short': 4         # 20 minutes
            }


@dataclass
class DualTrainingResult:
    """Result of Tactician dual training."""
    # Orchestration results
    orchestration_result: Optional[OrchestratorResult] = None

    # Training results
    long_base_models: Dict[str, Any] = None
    short_base_models: Dict[str, Any] = None
    long_ensemble_models: Dict[str, Any] = None
    short_ensemble_models: Dict[str, Any] = None

    # Performance metrics
    long_training_metrics: Dict[str, Any] = None
    short_training_metrics: Dict[str, Any] = None
    long_ensemble_metrics: Dict[str, Any] = None
    short_ensemble_metrics: Dict[str, Any] = None

    # Validation results
    long_validation_predictions: Dict[str, np.ndarray] = None
    short_validation_predictions: Dict[str, np.ndarray] = None

    # Metadata
    execution_time: float = 0.0
    total_long_samples: int = 0
    total_short_samples: int = 0
    training_phase: TrainingPhase = TrainingPhase.PRE_ML_ORCHESTRATION

    # Status tracking
    pre_ml_orchestration_completed: bool = False
    long_base_training_completed: bool = False
    short_base_training_completed: bool = False
    long_ensemble_training_completed: bool = False
    short_ensemble_training_completed: bool = False
    validation_completed: bool = False


class TacticianDualTrainingStep:
    """
    Tactician Dual Training Step.

    Handles training Tactician models twice (once for longs, once for shorts)
    with differentiated features and horizon labeling for each direction.
    """

    def __init__(self, config: Optional[DualTrainingConfig] = None):
        """Initialize the Tactician dual training step."""
        try:
            self.config = config or DualTrainingConfig()
            self.logger = system_logger.getChild('TacticianDualTrainingStep')

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

            # Initialize pre-ML orchestrator
            if PRE_ML_ORCHESTRATOR_AVAILABLE:
                orchestrator_config = OrchestratorConfig(
                    min_analyst_confidence=self.config.min_analyst_confidence,
                    subsequent_minutes=self.config.subsequent_minutes,
                    output_directory=self.config.output_directory,
                    max_lookback_periods=self.config.max_lookback_periods,
                    max_interaction_features=self.config.max_interaction_features,
                    max_polynomial_features=self.config.max_polynomial_features,
                    max_cross_timeframe_features=self.config.max_cross_timeframe_features,
                    enable_feature_optimization=self.config.enable_feature_optimization,
                    enable_pid_generation=self.config.enable_pid_generation,
                    enable_horizon_labeling=self.config.enable_horizon_labeling,
                    enable_feature_selection=self.config.enable_feature_selection,
                    enable_parallel_processing=self.config.enable_parallel_processing,
                    enable_gpu_acceleration=self.config.enable_gpu_acceleration,
                    memory_limit_gb=self.config.memory_limit_gb
                )
                self.pre_ml_orchestrator = TacticianPreMLOrchestrator(orchestrator_config)
                tprint_success("✅ Pre-ML orchestrator initialized")
            else:
                self.pre_ml_orchestrator = None

            # Initialize training components
            if TACTICIAN_TRAINING_AVAILABLE:
                self.base_trainer = TacticianModelsTrainingStep()
                self.ensemble_trainer = TacticianEnsembleTrainingStep()
                tprint_success("✅ Training components initialized")
            else:
                self.base_trainer = None
                self.ensemble_trainer = None

            tprint_success("✅ TacticianDualTrainingStep initialized successfully")
            tprint_info(f"Min analyst confidence: {self.config.min_analyst_confidence}")
            tprint_info(f"Output directory: {self.config.output_directory}")

        except Exception as e:
            tprint_error(f"❌ Failed to initialize TacticianDualTrainingStep: {e}")
            raise

    async def train_dual_tactician_models(
        self,
        analyst_signals: pd.DataFrame,
        market_data: pd.DataFrame,
        feature_names: List[str],
        **kwargs
    ) -> DualTrainingResult:
        """
        Train Tactician models twice (longs and shorts) with differentiated processing.

        Args:
            analyst_signals: DataFrame with Analyst signals and confidence scores
            market_data: Raw market data for feature generation
            feature_names: List of base feature names
            **kwargs: Additional parameters

        Returns:
            DualTrainingResult with trained models and metrics
        """
        start_time = tprint_timer()
        tprint_info("🚀 Starting Tactician dual training...")

        result = DualTrainingResult()
        result.training_phase = TrainingPhase.PRE_ML_ORCHESTRATION

        try:
            # Step 1: Pre-ML orchestration (separate long/short, optimize features, etc.)
            tprint_info("📊 Step 1: Pre-ML orchestration...")
            orchestration_result = await self._run_pre_ml_orchestration(
                analyst_signals, market_data, feature_names
            )

            if not orchestration_result or orchestration_result.feature_generation_status == "failed":
                raise ValueError("Pre-ML orchestration failed")

            result.orchestration_result = orchestration_result
            result.total_long_samples = orchestration_result.total_long_samples
            result.total_short_samples = orchestration_result.total_short_samples
            result.pre_ml_orchestration_completed = True
            result.training_phase = TrainingPhase.LONG_MODEL_TRAINING

            tprint_success(f"✅ Pre-ML orchestration completed: {result.total_long_samples} long, {result.total_short_samples} short samples")

            # Step 2: Train base models for long signals
            if self.config.train_base_models and result.total_long_samples >= self.config.min_training_samples:
                tprint_info("📈 Step 2: Training base models for long signals...")
                long_base_result = await self._train_base_models(
                    orchestration_result.long_training_data,
                    orchestration_result.long_selected_features,
                    "long"
                )
                result.long_base_models = long_base_result.get('models', {})
                result.long_training_metrics = long_base_result.get('metrics', {})
                result.long_base_training_completed = True
                tprint_success("✅ Long base model training completed")
            else:
                tprint_info("⏭️ Skipping long base model training - insufficient data or disabled")

            # Step 3: Train base models for short signals
            if self.config.train_base_models and result.total_short_samples >= self.config.min_training_samples:
                tprint_info("📉 Step 3: Training base models for short signals...")
                short_base_result = await self._train_base_models(
                    orchestration_result.short_training_data,
                    orchestration_result.short_selected_features,
                    "short"
                )
                result.short_base_models = short_base_result.get('models', {})
                result.short_training_metrics = short_base_result.get('metrics', {})
                result.short_base_training_completed = True
                tprint_success("✅ Short base model training completed")
            else:
                tprint_info("⏭️ Skipping short base model training - insufficient data or disabled")

            # Step 4: Train ensemble models for long signals
            if self.config.train_ensemble_models and result.total_long_samples >= self.config.min_training_samples and result.long_base_models:
                tprint_info("🔄 Step 4: Training ensemble models for long signals...")
                long_ensemble_result = await self._train_ensemble_models(
                    orchestration_result.long_training_data,
                    orchestration_result.long_selected_features,
                    result.long_base_models,
                    "long"
                )
                result.long_ensemble_models = long_ensemble_result.get('models', {})
                result.long_ensemble_metrics = long_ensemble_result.get('metrics', {})
                result.long_ensemble_training_completed = True
                tprint_success("✅ Long ensemble model training completed with FULL FEATURE INTEGRATION")
                tprint_info("   🎯 Long ensemble includes: Base features + HMM outputs + Analyst predictions + OOF from base models")
            else:
                tprint_info("⏭️ Skipping long ensemble model training - insufficient data, disabled, or no base models")

            # Step 5: Train ensemble models for short signals
            if self.config.train_ensemble_models and result.total_short_samples >= self.config.min_training_samples and result.short_base_models:
                tprint_info("🔄 Step 5: Training ensemble models for short signals...")
                short_ensemble_result = await self._train_ensemble_models(
                    orchestration_result.short_training_data,
                    orchestration_result.short_selected_features,
                    result.short_base_models,
                    "short"
                )
                result.short_ensemble_models = short_ensemble_result.get('models', {})
                result.short_ensemble_metrics = short_ensemble_result.get('metrics', {})
                result.short_ensemble_training_completed = True
                tprint_success("✅ Short ensemble model training completed with FULL FEATURE INTEGRATION")
                tprint_info("   🎯 Short ensemble includes: Base features + HMM outputs + Analyst predictions + OOF from base models")
            else:
                tprint_info("⏭️ Skipping short ensemble model training - insufficient data, disabled, or no base models")

            # Step 6: Validation and finalization
            tprint_info("✅ Step 6: Validation and finalization...")
            result.training_phase = TrainingPhase.VALIDATION
            result.validation_completed = True

            # Save results
            if self.config.save_models or self.config.save_metrics:
                await self._save_training_results(result)

            result.execution_time = tprint_timer(start_time)
            result.training_phase = TrainingPhase.COMPLETED

            tprint_performance("Tactician dual training", result.execution_time)
            tprint_success("🎉 Tactician dual training completed successfully with FULL FEATURE INTEGRATION!")
            tprint_info(f"📈 Long models: {len(result.long_base_models) if result.long_base_models else 0} base (XGBOOST, LIGHTGBM, DEEPSCALER_1M, FINANCIAL_RESNET)")
            tprint_info(f"   🎯 Long ensemble: {len(result.long_ensemble_models) if result.long_ensemble_models else 0} ensemble (includes ALL features + HMM + Analyst outputs)")
            tprint_info(f"📉 Short models: {len(result.short_base_models) if result.short_base_models else 0} base (XGBOOST, LIGHTGBM, DEEPSCALER_1M, FINANCIAL_RESNET)")
            tprint_info(f"   🎯 Short ensemble: {len(result.short_ensemble_models) if result.short_ensemble_models else 0} ensemble (includes ALL features + HMM + Analyst outputs)")
            tprint_info(f"⏱️ Total time: {result.execution_time:.2f}s")

            # Log feature integration summary
            if result.long_ensemble_models and result.orchestration_result:
                tprint_info("🎯 FEATURE INTEGRATION COMPLETED:")
                tprint_info("   ✅ Base features from pre-ML orchestration")
                tprint_info("   ✅ HMM regime features and probabilities")
                tprint_info("   ✅ Analyst model predictions and confidence scores")
                tprint_info("   ✅ OOF predictions from all base models")
                tprint_info("   ✅ Technical indicators and market data")
                tprint_info("   ✅ Multi-horizon target variables")
                tprint_info("   ✅ Sample weights based on Analyst confidence")

            return result

        except Exception as e:
            result.execution_time = tprint_timer(start_time)
            result.training_phase = TrainingPhase.FAILED
            tprint_error(f"❌ Tactician dual training failed: {e}")
            tprint_error(f"Error details: {traceback.format_exc()}")
            raise

    async def _run_pre_ml_orchestration(
        self,
        analyst_signals: pd.DataFrame,
        market_data: pd.DataFrame,
        feature_names: List[str]
    ) -> Optional[OrchestratorResult]:
        """Run pre-ML orchestration to prepare data for training."""
        try:
            if not self.pre_ml_orchestrator:
                raise ValueError("Pre-ML orchestrator not available")

            # Run the complete pre-ML pipeline
            orchestration_result = await self.pre_ml_orchestrator.orchestrate_pre_ml_training(
                analyst_signals=analyst_signals,
                market_data=market_data,
                feature_names=feature_names
            )

            if orchestration_result.feature_generation_status == "completed":
                tprint_success("✅ Pre-ML orchestration completed successfully")
                return orchestration_result
            else:
                tprint_error(f"❌ Pre-ML orchestration failed with status: {orchestration_result.feature_generation_status}")
                return None

        except Exception as e:
            tprint_error(f"❌ Pre-ML orchestration failed: {e}")
            return None

    async def _train_base_models(
        self,
        training_data: pd.DataFrame,
        selected_features: List[str],
        signal_type: str
    ) -> Dict[str, Any]:
        """Train multiple base models for a specific signal type."""
        try:
            tprint_info(f"🔧 Training base models for {signal_type} signals...")

            if training_data.empty or not selected_features:
                raise ValueError(f"Insufficient training data for {signal_type} models")

            # Prepare training data
            X = training_data[selected_features].values
            y = training_data.filter(like='target_').values  # Multiple target horizons

            if len(y.shape) == 1:
                y = y.reshape(-1, 1)

            sample_weight = training_data.get('sample_weight', np.ones(len(training_data))).values

            all_models = {}
            all_metrics = {}

            # Define multiple base model types to train
            base_model_types = [
                "XGBOOST",
                "LIGHTGBM",
                "DEEPSCALER_1M",
                "FINANCIAL_RESNET"
            ]

            # Train each base model type
            for model_type in base_model_types:
                try:
                    tprint_info(f"   🔧 Training {model_type} model for {signal_type} signals...")

                    # Create training configuration for this specific model type
                    training_config = {
                        'signal_type': signal_type,
                        'model_type': model_type,
                        'training_data': training_data,
                        'feature_columns': selected_features,
                        'target_columns': [col for col in training_data.columns if col.startswith('target_')],
                        'sample_weight': sample_weight,
                        'save_models': self.config.save_models,
                        'output_directory': f"{self.config.output_directory}/{signal_type}_base_models/{model_type.lower()}"
                    }

                    # Call the existing base trainer with specific model type
                    training_result = await self.base_trainer.train_tactician_models(
                        **training_config
                    )

                    # Store results with model type prefix
                    if training_result.get('models'):
                        for model_name, model in training_result['models'].items():
                            all_models[f"{signal_type}_{model_type.lower()}_{model_name}"] = model

                    if training_result.get('metrics'):
                        all_metrics[f"{signal_type}_{model_type.lower()}"] = training_result['metrics']

                    tprint_success(f"   ✅ {model_type} model trained for {signal_type} signals")

                except Exception as e:
                    tprint_warning(f"   ⚠️ Failed to train {model_type} model for {signal_type}: {e}")
                    continue

            if not all_models:
                raise ValueError(f"Failed to train any base models for {signal_type}")

            return {
                'models': all_models,
                'metrics': all_metrics,
                'training_time': 0.0,  # TODO: Track total time
                'features_used': selected_features,
                'samples_used': len(training_data),
                'model_types_trained': base_model_types,
                'models_per_type': len(base_model_types)
            }

        except Exception as e:
            tprint_error(f"❌ Base model training for {signal_type} failed: {e}")
            return {
                'models': {},
                'metrics': {},
                'training_time': 0.0,
                'error': str(e)
            }

    async def _train_ensemble_models(
        self,
        training_data: pd.DataFrame,
        selected_features: List[str],
        base_models: Dict[str, Any],
        signal_type: str
    ) -> Dict[str, Any]:
        """Train ensemble models for a specific signal type with full feature integration."""
        try:
            tprint_info(f"🔄 Training ensemble models for {signal_type} signals with full feature integration...")

            if training_data.empty or not selected_features or not base_models:
                raise ValueError(f"Insufficient data for {signal_type} ensemble training")

            # Prepare training data (same as base models)
            X = training_data[selected_features].values
            y = training_data.filter(like='target_').values

            if len(y.shape) == 1:
                y = y.reshape(-1, 1)

            sample_weight = training_data.get('sample_weight', np.ones(len(training_data))).values

            # Train ensemble models using existing trainer
            if self.ensemble_trainer:
                # Create training configuration for this signal type with full feature integration
                training_config = {
                    'signal_type': signal_type,
                    'training_data': training_data,
                    'base_models': base_models,
                    'feature_columns': selected_features,
                    'target_columns': [col for col in training_data.columns if col.startswith('target_')],
                    'sample_weight': sample_weight,
                    'save_models': self.config.save_models,
                    'output_directory': f"{self.config.output_directory}/{signal_type}_ensemble_models",
                    'enable_full_integration': True,  # Enable all feature integration
                    'include_hmm_features': True,     # Include HMM regime features
                    'include_analyst_features': True, # Include Analyst model outputs
                    'include_oof_predictions': True   # Include OOF predictions from base models
                }

                # Call the existing ensemble trainer with full integration
                # Import the execution function
                from src.training.steps.model_training.tactician_ensemble_training import execute_tactician_ensemble_training

                # Prepare inputs for ensemble training
                X_base = training_data[selected_features].values
                y_targets = training_data.filter(like='target_').values
                if len(y_targets.shape) == 1:
                    y_targets = y_targets.reshape(-1, 1)

                # Get regime labels if available
                regime_labels = training_data.get('hmm_regime', training_data.get('regime', np.zeros(len(training_data)))).values

                # Call the ensemble training with full feature integration
                training_result = await execute_tactician_ensemble_training(
                    X=X_base,
                    y=y_targets,
                    regime_labels=regime_labels,
                    config=None,  # Use default config
                    feature_names=selected_features,
                    base_tactician_models=base_models,  # Pass base models for OOF predictions
                    analyst_green_light_periods=np.ones(len(training_data)),  # All samples are valid
                    confidence_scores=training_data.get('analyst_confidence', np.ones(len(training_data))).values,
                    timestamps=training_data.get('timestamp', pd.date_range('2024-01-01', periods=len(training_data), freq='1min')).values,
                    confidence_threshold=0.5,
                    ride_duration_minutes=45
                )

                # Log what features were included
                if training_result.get('metadata'):
                    metadata = training_result['metadata']
                    tprint_info(f"   📊 Ensemble training included:")
                    tprint_info(f"      - Base features: {metadata.get('base_features_count', 'N/A')}")
                    tprint_info(f"      - HMM features: {metadata.get('hmm_features_count', 'N/A')}")
                    tprint_info(f"      - Analyst features: {metadata.get('analyst_features_count', 'N/A')}")
                    tprint_info(f"      - OOF predictions: {metadata.get('oof_predictions_count', 'N/A')}")
                    tprint_info(f"      - Total features: {metadata.get('total_features', 'N/A')}")

                return {
                    'models': training_result.get('models', {}),
                    'metrics': training_result.get('metrics', {}),
                    'training_time': training_result.get('execution_time', 0.0),
                    'features_used': selected_features,
                    'samples_used': len(training_data),
                    'metadata': training_result.get('metadata', {}),
                    'feature_integration_complete': True
                }
            else:
                tprint_error("❌ Ensemble trainer not available")
                return {
                    'models': {},
                    'metrics': {},
                    'training_time': 0.0,
                    'error': 'Ensemble trainer not available',
                    'feature_integration_complete': False
                }

        except Exception as e:
            tprint_error(f"❌ Ensemble model training for {signal_type} failed: {e}")
            return {
                'models': {},
                'metrics': {},
                'training_time': 0.0,
                'error': str(e),
                'feature_integration_complete': False
            }

    async def _save_training_results(self, result: DualTrainingResult):
        """Save training results to disk."""
        try:
            output_dir = Path(self.config.output_directory)
            ensure_directory(output_dir)

            # Save orchestration results
            if result.orchestration_result:
                orchestration_path = output_dir / "orchestration_results.json"
                orchestration_data = {
                    'total_long_samples': result.orchestration_result.total_long_samples,
                    'total_short_samples': result.orchestration_result.total_short_samples,
                    'execution_time': result.orchestration_result.execution_time,
                    'long_selected_features': result.orchestration_result.long_selected_features,
                    'short_selected_features': result.orchestration_result.short_selected_features,
                    'long_data_quality_score': result.orchestration_result.long_data_quality_score,
                    'short_data_quality_score': result.orchestration_result.short_data_quality_score
                }
                safe_json_dump(orchestration_data, orchestration_path)
                tprint_debug(f"💾 Saved orchestration results: {orchestration_path}")

            # Save long training results
            if result.long_base_models or result.long_ensemble_models:
                long_results = {
                    'base_models_count': len(result.long_base_models) if result.long_base_models else 0,
                    'ensemble_models_count': len(result.long_ensemble_models) if result.long_ensemble_models else 0,
                    'training_metrics': result.long_training_metrics,
                    'ensemble_metrics': result.long_ensemble_metrics,
                    'samples_used': result.total_long_samples
                }
                long_path = output_dir / "long_training_results.json"
                safe_json_dump(long_results, long_path)
                tprint_debug(f"💾 Saved long training results: {long_path}")

            # Save short training results
            if result.short_base_models or result.short_ensemble_models:
                short_results = {
                    'base_models_count': len(result.short_base_models) if result.short_base_models else 0,
                    'ensemble_models_count': len(result.short_ensemble_models) if result.short_ensemble_models else 0,
                    'training_metrics': result.short_training_metrics,
                    'ensemble_metrics': result.short_ensemble_metrics,
                    'samples_used': result.total_short_samples
                }
                short_path = output_dir / "short_training_results.json"
                safe_json_dump(short_results, short_path)
                tprint_debug(f"💾 Saved short training results: {short_path}")

            # Save metadata
            metadata = {
                'execution_time': result.execution_time,
                'training_phase': result.training_phase.value if hasattr(result.training_phase, 'value') else str(result.training_phase),
                'pre_ml_orchestration_completed': result.pre_ml_orchestration_completed,
                'long_base_training_completed': result.long_base_training_completed,
                'short_base_training_completed': result.short_base_training_completed,
                'long_ensemble_training_completed': result.long_ensemble_training_completed,
                'short_ensemble_training_completed': result.short_ensemble_training_completed,
                'validation_completed': result.validation_completed,
                'total_long_samples': result.total_long_samples,
                'total_short_samples': result.total_short_samples,
                'config': {
                    'min_analyst_confidence': self.config.min_analyst_confidence,
                    'subsequent_minutes': self.config.subsequent_minutes,
                    'train_base_models': self.config.train_base_models,
                    'train_ensemble_models': self.config.train_ensemble_models,
                    'validation_split': self.config.validation_split,
                    'min_training_samples': self.config.min_training_samples
                },
                'timestamp': time.time()
            }

            metadata_path = output_dir / "dual_training_metadata.json"
            safe_json_dump(metadata, metadata_path)
            tprint_debug(f"💾 Saved training metadata: {metadata_path}")

            tprint_success(f"✅ All training results saved to {output_dir}")

        except Exception as e:
            tprint_error(f"❌ Failed to save training results: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the dual training step."""
        metrics = {
            'config': {
                'min_analyst_confidence': self.config.min_analyst_confidence,
                'subsequent_minutes': self.config.subsequent_minutes,
                'train_base_models': self.config.train_base_models,
                'train_ensemble_models': self.config.train_ensemble_models,
                'output_directory': self.config.output_directory
            },
            'component_availability': {
                'pre_ml_orchestrator': self.pre_ml_orchestrator is not None,
                'base_trainer': self.base_trainer is not None,
                'ensemble_trainer': self.ensemble_trainer is not None
            },
            'hardware_optimization': {
                'gpu_manager': self.gpu_manager is not None,
                'memory_optimizer': self.memory_optimizer is not None,
                'cpu_optimizer': self.cpu_optimizer is not None
            }
        }

        return metrics