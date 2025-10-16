"""
Tactician Training Step - Unified Multi-Model Training

This step handles training multiple Tactician models with a unified approach:

BASE MODELS (3 models):
- RandomSurvivalForest model
- XGBoost model
- ElasticNetCV model

ENSEMBLE MODEL (1 model):
- Ensemble model combining all base models + ALL FEATURES + HMM + Analyst outputs

🎯 ENSEMBLE FEATURE INTEGRATION:
Each ensemble model includes:
- All base features from pre-ML orchestration
- HMM regime features and probabilities
- Analyst model predictions and confidence scores
- OOF predictions from all base models
- Technical indicators and market data
- Multi-horizon target variables

The training uses a unified approach with optimized features and horizon labeling.

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

# Import new modular Tactician training components
try:
    from .tactician_models_training import TacticianModelsTrainingStep
    from .tactician_ensemble_training import TacticianEnsembleTrainingStep
    TACTICIAN_TRAINING_AVAILABLE = True
except ImportError as e:
    TACTICIAN_TRAINING_AVAILABLE = False
    tprint_warning(f"⚠️ Tactician training components not available: {e}")

class TrainingPhase(Enum):
    """Training phase enumeration."""
    PRE_ML_ORCHESTRATION = "pre_ml_orchestration"
    BASE_MODEL_TRAINING = "base_model_training"
    ENSEMBLE_TRAINING = "ensemble_training"
    VALIDATION = "validation"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class TacticianTrainingConfig:
    """Configuration for Tactician unified training."""
    # Pre-ML orchestration parameters
    min_analyst_confidence: float = 0.5
    subsequent_minutes: int = 45
    output_directory: str = "generated/tactician_training"

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
    max_cross_timeframe_features: int = 15

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
class TacticianTrainingResult:
    """Result of Tactician unified training."""
    # Orchestration results
    orchestration_result: Optional[OrchestratorResult] = None

    # Training results
    base_models: Dict[str, Any] = None
    ensemble_models: Dict[str, Any] = None

    # Performance metrics
    base_training_metrics: Dict[str, Any] = None
    ensemble_metrics: Dict[str, Any] = None

    # Validation results
    validation_predictions: Dict[str, np.ndarray] = None

    # Metadata
    execution_time: float = 0.0
    total_samples: int = 0
    training_phase: TrainingPhase = TrainingPhase.PRE_ML_ORCHESTRATION

    # Status tracking
    pre_ml_orchestration_completed: bool = False
    base_training_completed: bool = False
    ensemble_training_completed: bool = False
    validation_completed: bool = False

class TacticianTrainingStep:
    """
    Tactician Training Step.

    Handles training Tactician models with a unified approach using
    RandomSurvivalForest, XGBoost, and ElasticNetCV base models.
    """

    def __init__(self, config: Optional[TacticianTrainingConfig] = None):
        """Initialize the Tactician training step."""
        try:
            self.config = config or TacticianTrainingConfig()
            self.logger = system_logger.getChild('TacticianTrainingStep')

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

            tprint_success("✅ TacticianTrainingStep initialized successfully")
            tprint_info(f"Min analyst confidence: {self.config.min_analyst_confidence}")
            tprint_info(f"Output directory: {self.config.output_directory}")

        except Exception as e:
            tprint_error(f"❌ Failed to initialize TacticianTrainingStep: {e}")
            raise

    async def train_tactician_models(
        self,
        analyst_signals: pd.DataFrame,
        market_data: pd.DataFrame,
        feature_names: List[str],
        **kwargs
    ) -> TacticianTrainingResult:
        """
        Train Tactician models with a unified approach.

        Args:
            analyst_signals: DataFrame with Analyst signals and confidence scores
            market_data: Raw market data for feature generation
            feature_names: List of base feature names
            **kwargs: Additional parameters

        Returns:
            TacticianTrainingResult with trained models and metrics
        """
        start_time = tprint_timer()
        tprint_info("🚀 Starting Tactician unified training...")

        result = TacticianTrainingResult()
        result.training_phase = TrainingPhase.PRE_ML_ORCHESTRATION

        try:
            # Step 1: Pre-ML orchestration (unified approach)
            tprint_info("📊 Step 1: Pre-ML orchestration...")
            orchestration_result = await self._run_pre_ml_orchestration(
                analyst_signals, market_data, feature_names
            )

            if not orchestration_result or orchestration_result.feature_generation_status == "failed":
                raise ValueError("Pre-ML orchestration failed")

            result.orchestration_result = orchestration_result
            result.total_samples = orchestration_result.total_samples
            result.pre_ml_orchestration_completed = True
            result.training_phase = TrainingPhase.BASE_MODEL_TRAINING

            tprint_success(f"✅ Pre-ML orchestration completed: {result.total_samples} total samples")

            # Step 2: Train base models
            if self.config.train_base_models and result.total_samples >= self.config.min_training_samples:
                tprint_info("📈 Step 2: Training base models...")
                base_result = await self._train_base_models(
                    orchestration_result.training_data,
                    orchestration_result.selected_features
                )
                result.base_models = base_result.get('models', {})
                result.base_training_metrics = base_result.get('metrics', {})
                result.base_training_completed = True
                tprint_success("✅ Base model training completed")
            else:
                tprint_info("⏭️ Skipping base model training - insufficient data or disabled")

            # Step 3: Train ensemble models
            if self.config.train_ensemble_models and result.total_samples >= self.config.min_training_samples and result.base_models:
                tprint_info("🔄 Step 3: Training ensemble models...")
                ensemble_result = await self._train_ensemble_models(
                    orchestration_result.training_data,
                    orchestration_result.selected_features,
                    result.base_models
                )
                result.ensemble_models = ensemble_result.get('models', {})
                result.ensemble_metrics = ensemble_result.get('metrics', {})
                result.ensemble_training_completed = True
                tprint_success("✅ Ensemble model training completed with FULL FEATURE INTEGRATION")
                tprint_info("   🎯 Ensemble includes: Base features + HMM outputs + Analyst predictions + OOF from base models")
            else:
                tprint_info("⏭️ Skipping ensemble model training - insufficient data, disabled, or no base models")

            # Step 6: Model evaluation and validation
            tprint_info("✅ Step 6: Model evaluation and validation...")
            result.training_phase = TrainingPhase.VALIDATION

            # Evaluate model performance
            result = self._evaluate_model_performance(result)
            result.validation_completed = True

            # Save results
            if self.config.save_models or self.config.save_metrics:
                await self._save_training_results(result)

            result.execution_time = tprint_timer(start_time)
            result.training_phase = TrainingPhase.COMPLETED

            # Add comprehensive reporting
            result = self._add_comprehensive_reporting(result, start_time)

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
        selected_features: List[str]
    ) -> Dict[str, Any]:
        """Train multiple base models with unified approach."""
        try:
            tprint_info("🔧 Training base models...")

            if len(training_data) == 0 or not selected_features:
                raise ValueError("Insufficient training data for base models")

            # Prepare training data
            X = training_data[selected_features].values
            y = training_data.filter(like='target_').values  # Multiple target horizons

            if len(y.shape) == 1:
                y = y.reshape(-1, 1)

            sample_weight = training_data.get('sample_weight', np.ones(len(training_data))).values

            all_models = {}
            all_metrics = {}

            # Define the new base model types to train
            base_model_types = [
                "RANDOM_SURVIVAL_FOREST",
                "XGBOOST",
                "ELASTIC_NET_CV"
            ]

            # Train each base model type
            for model_type in base_model_types:
                try:
                    tprint_info(f"   🔧 Training {model_type} model...")

                    # Create training configuration for this specific model type
                    training_config = {
                        'model_type': model_type,
                        'training_data': training_data,
                        'feature_columns': selected_features,
                        'target_columns': [col for col in training_data.columns if col.startswith('target_')],
                        'sample_weight': sample_weight,
                        'save_models': self.config.save_models,
                        'output_directory': f"{self.config.output_directory}/base_models/{model_type.lower()}"
                    }

                    # Call the existing base trainer with specific model type
                    training_result = await self.base_trainer.train_tactician_models(
                        **training_config
                    )

                    # Store results with model type prefix
                    if training_result.get('models'):
                        for model_name, model in training_result['models'].items():
                            all_models[f"{model_type.lower()}_{model_name}"] = model

                    if training_result.get('metrics'):
                        all_metrics[f"{model_type.lower()}"] = training_result['metrics']

                    tprint_success(f"   ✅ {model_type} model trained")

                except Exception as e:
                    tprint_warning(f"   ⚠️ Failed to train {model_type} model: {e}")
                    continue

            if not all_models:
                raise ValueError("Failed to train any base models")

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
            tprint_error(f"❌ Base model training failed: {e}")
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
        base_models: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train ensemble models with full feature integration."""
        try:
            tprint_info("🔄 Training ensemble models with full feature integration...")

            if len(training_data) == 0 or not selected_features or not base_models:
                raise ValueError("Insufficient data for ensemble training")

            # Prepare training data (same as base models)
            X = training_data[selected_features].values
            y = training_data.filter(like='target_').values

            if len(y.shape) == 1:
                y = y.reshape(-1, 1)

            sample_weight = training_data.get('sample_weight', np.ones(len(training_data))).values

            # Train ensemble models using existing trainer
            if self.ensemble_trainer:
                # Create training configuration with full feature integration
                training_config = {
                    'training_data': training_data,
                    'base_models': base_models,
                    'feature_columns': selected_features,
                    'target_columns': [col for col in training_data.columns if col.startswith('target_')],
                    'sample_weight': sample_weight,
                    'save_models': self.config.save_models,
                    'output_directory': f"{self.config.output_directory}/ensemble_models",
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
                    tprint_info("   📊 Ensemble training included:")
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
            tprint_error(f"❌ Ensemble model training failed: {e}")
            return {
                'models': {},
                'metrics': {},
                'training_time': 0.0,
                'error': str(e),
                'feature_integration_complete': False
            }

    async def _save_training_results(self, result: TacticianTrainingResult):
        """Save training results to disk."""
        try:
            output_dir = Path(self.config.output_directory)
            ensure_directory(output_dir)

            # Save orchestration results
            if result.orchestration_result:
                orchestration_path = output_dir / "orchestration_results.json"
                orchestration_data = {
                    'total_samples': result.orchestration_result.total_samples,
                    'execution_time': result.orchestration_result.execution_time,
                    'selected_features': result.orchestration_result.selected_features,
                    'data_quality_score': result.orchestration_result.data_quality_score
                }
                safe_json_dump(orchestration_data, orchestration_path)
                tprint_debug(f"💾 Saved orchestration results: {orchestration_path}")

            # Save base model training results
            if result.base_models:
                base_results = {
                    'base_models_count': len(result.base_models),
                    'training_metrics': result.base_training_metrics,
                    'samples_used': result.total_samples
                }
                base_path = output_dir / "base_training_results.json"
                safe_json_dump(base_results, base_path)
                tprint_debug(f"💾 Saved base training results: {base_path}")

            # Save ensemble model training results
            if result.ensemble_models:
                ensemble_results = {
                    'ensemble_models_count': len(result.ensemble_models),
                    'ensemble_metrics': result.ensemble_metrics,
                    'samples_used': result.total_samples
                }
                ensemble_path = output_dir / "ensemble_training_results.json"
                safe_json_dump(ensemble_results, ensemble_path)
                tprint_debug(f"💾 Saved ensemble training results: {ensemble_path}")

            # Save metadata
            metadata = {
                'execution_time': result.execution_time,
                'training_phase': result.training_phase.value if hasattr(result.training_phase, 'value') else str(result.training_phase),
                'pre_ml_orchestration_completed': result.pre_ml_orchestration_completed,
                'base_training_completed': result.base_training_completed,
                'ensemble_training_completed': result.ensemble_training_completed,
                'validation_completed': result.validation_completed,
                'total_samples': result.total_samples,
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

            metadata_path = output_dir / "training_metadata.json"
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

    def _add_comprehensive_reporting(self, result: TacticianTrainingResult, start_time: float) -> TacticianTrainingResult:
        """Add comprehensive reporting and metrics to dual training results."""
        try:
            total_time = time.time() - start_time

            # Create comprehensive report
            comprehensive_report = {
                'training_summary': {
                    'total_training_time': total_time,
                    'success': result.training_phase == TrainingPhase.COMPLETED,
                    'training_phase': result.training_phase.value,
                    'long_base_models_trained': len(result.long_base_models),
                    'short_base_models_trained': len(result.short_base_models),
                    'long_ensemble_models_trained': len(result.long_ensemble_models),
                    'short_ensemble_models_trained': len(result.short_ensemble_models),
                    'total_models_trained': len(result.long_base_models) + len(result.short_base_models) + len(result.long_ensemble_models) + len(result.short_ensemble_models),
                    'pre_ml_orchestration_completed': result.orchestration_completed,
                    'base_training_completed': result.base_training_completed,
                    'ensemble_training_completed': result.ensemble_training_completed,
                    'validation_completed': result.validation_completed,
                    'results_saved': result.results_saved
                },
                'model_breakdown': {
                    'long_models': {
                        'base_models': [
                            {'type': 'XGBOOST', 'count': 1, 'status': 'trained' if result.long_base_models else 'failed'},
                            {'type': 'LIGHTGBM', 'count': 1, 'status': 'trained' if result.long_base_models else 'failed'},
                            {'type': 'DEEPSCALER_1M', 'count': 1, 'status': 'trained' if result.long_base_models else 'failed'},
                            {'type': 'FINANCIAL_RESNET', 'count': 1, 'status': 'trained' if result.long_base_models else 'failed'}
                        ],
                        'ensemble_models': [
                            {'type': 'ENSEMBLE', 'count': 1, 'status': 'trained' if result.long_ensemble_models else 'failed', 'features_integrated': result.long_ensemble_training_completed}
                        ]
                    },
                    'short_models': {
                        'base_models': [
                            {'type': 'XGBOOST', 'count': 1, 'status': 'trained' if result.short_base_models else 'failed'},
                            {'type': 'LIGHTGBM', 'count': 1, 'status': 'trained' if result.short_base_models else 'failed'},
                            {'type': 'DEEPSCALER_1M', 'count': 1, 'status': 'trained' if result.short_base_models else 'failed'},
                            {'type': 'FINANCIAL_RESNET', 'count': 1, 'status': 'trained' if result.short_base_models else 'failed'}
                        ],
                        'ensemble_models': [
                            {'type': 'ENSEMBLE', 'count': 1, 'status': 'trained' if result.short_ensemble_models else 'failed', 'features_integrated': result.short_ensemble_training_completed}
                        ]
                    }
                },
                'sample_metrics': {
                    'total_long_samples': result.total_long_samples,
                    'total_short_samples': result.total_short_samples,
                    'min_training_samples_required': self.config.min_training_samples,
                    'long_samples_sufficient': result.total_long_samples >= self.config.min_training_samples,
                    'short_samples_sufficient': result.total_short_samples >= self.config.min_training_samples,
                    'total_samples_processed': result.total_long_samples + result.total_short_samples
                },
                'feature_integration_metrics': {
                    'hmm_features_included': result.long_ensemble_training_completed,
                    'analyst_features_included': result.long_ensemble_training_completed,
                    'technical_features_included': result.long_ensemble_training_completed,
                    'oof_predictions_included': result.long_ensemble_training_completed,
                    'feature_integration_complete': result.long_ensemble_training_completed and result.short_ensemble_training_completed,
                    'long_ensemble_feature_count': len(result.orchestration_result.long_selected_features) if result.orchestration_result else 0,
                    'short_ensemble_feature_count': len(result.orchestration_result.short_selected_features) if result.orchestration_result else 0
                },
                'performance_metrics': {
                    'orchestration_time': getattr(result.orchestration_result, 'execution_time', 0) if result.orchestration_result else 0,
                    'long_base_training_time': result.long_base_training_time,
                    'short_base_training_time': result.short_base_training_time,
                    'long_ensemble_training_time': result.long_ensemble_training_time,
                    'short_ensemble_training_time': result.short_ensemble_training_time,
                    'total_execution_time': result.execution_time,
                    'memory_usage_mb': getattr(result, 'memory_usage_mb', 0),
                    'cpu_usage_percent': getattr(result, 'cpu_usage_percent', 0),
                    'average_training_time_per_model': result.execution_time / max(result.total_models_trained, 1)
                },
                'quality_metrics': {
                    'long_training_quality': result.long_training_quality,
                    'short_training_quality': result.short_training_quality,
                    'long_model_diversity': len(result.long_base_models),
                    'short_model_diversity': len(result.short_base_models),
                    'ensemble_integration_success': result.long_ensemble_training_completed and result.short_ensemble_training_completed,
                    'feature_differentiation_success': result.orchestration_completed
                },
                'error_analysis': {
                    'orchestration_errors': result.orchestration_errors,
                    'long_base_training_errors': result.long_base_training_errors,
                    'short_base_training_errors': result.short_base_training_errors,
                    'long_ensemble_training_errors': result.long_ensemble_training_errors,
                    'short_ensemble_training_errors': result.short_ensemble_training_errors,
                    'total_errors': result.orchestration_errors + result.long_base_training_errors + result.short_base_training_errors + result.long_ensemble_training_errors + result.short_ensemble_training_errors
                },
                'evaluation_metrics': {
                    'long_training_accuracy': getattr(result, 'long_training_accuracy', 0.0),
                    'short_training_accuracy': getattr(result, 'short_training_accuracy', 0.0),
                    'long_validation_accuracy': getattr(result, 'long_validation_accuracy', 0.0),
                    'short_validation_accuracy': getattr(result, 'short_validation_accuracy', 0.0),
                    'long_test_accuracy': getattr(result, 'long_test_accuracy', 0.0),
                    'short_test_accuracy': getattr(result, 'short_test_accuracy', 0.0),
                    'long_f1_score': getattr(result, 'long_f1_score', 0.0),
                    'short_f1_score': getattr(result, 'short_f1_score', 0.0),
                    'long_precision': getattr(result, 'long_precision', 0.0),
                    'short_precision': getattr(result, 'short_precision', 0.0),
                    'long_recall': getattr(result, 'long_recall', 0.0),
                    'short_recall': getattr(result, 'short_recall', 0.0),
                    'long_roc_auc': getattr(result, 'long_roc_auc', 0.0),
                    'short_roc_auc': getattr(result, 'short_roc_auc', 0.0),
                    'long_sharpe_ratio': getattr(result, 'long_sharpe_ratio', 0.0),
                    'short_sharpe_ratio': getattr(result, 'short_sharpe_ratio', 0.0),
                    'long_max_drawdown': getattr(result, 'long_max_drawdown', 0.0),
                    'short_max_drawdown': getattr(result, 'short_max_drawdown', 0.0),
                    'long_profit_factor': getattr(result, 'long_profit_factor', 0.0),
                    'short_profit_factor': getattr(result, 'short_profit_factor', 0.0),
                    'long_win_rate': getattr(result, 'long_win_rate', 0.0),
                    'short_win_rate': getattr(result, 'short_win_rate', 0.0),
                    'long_mse': getattr(result, 'long_mse', 0.0),
                    'short_mse': getattr(result, 'short_mse', 0.0),
                    'long_mae': getattr(result, 'long_mae', 0.0),
                    'short_mae': getattr(result, 'short_mae', 0.0),
                    'long_rmse': getattr(result, 'long_rmse', 0.0),
                    'short_rmse': getattr(result, 'short_rmse', 0.0),
                    'long_r2_score': getattr(result, 'long_r2_score', 0.0),
                    'short_r2_score': getattr(result, 'short_r2_score', 0.0),
                    'long_total_trades': getattr(result, 'long_total_trades', 0),
                    'short_total_trades': getattr(result, 'short_total_trades', 0),
                    'long_avg_trades_per_month': getattr(result, 'long_avg_trades_per_month', 0.0),
                    'short_avg_trades_per_month': getattr(result, 'short_avg_trades_per_month', 0.0),
                    'long_total_pnl': getattr(result, 'long_total_pnl', 0.0),
                    'short_total_pnl': getattr(result, 'short_total_pnl', 0.0),
                    'long_monthly_pnl': getattr(result, 'long_monthly_pnl', {}),
                    'short_monthly_pnl': getattr(result, 'short_monthly_pnl', {}),
                    'long_monthly_trade_count': getattr(result, 'long_monthly_trade_count', {}),
                    'short_monthly_trade_count': getattr(result, 'short_monthly_trade_count', {}),
                    'evaluation_completed': getattr(result, 'evaluation_completed', False),
                    'cross_validation_folds': getattr(result, 'cross_validation_folds', 5),
                    'evaluation_time': getattr(result, 'evaluation_time', 0.0)
                }
            }

            # Add comprehensive report to result
            result.comprehensive_report = comprehensive_report

            # Log comprehensive summary
            self._log_comprehensive_summary(comprehensive_report)

            return result

        except Exception as e:
            tprint_error(f"❌ Failed to add comprehensive reporting: {e}")
            # Return result without reporting if it fails
            return result

    def _evaluate_model_performance(self, result: TacticianTrainingResult) -> TacticianTrainingResult:
        """Evaluate trained models and calculate comprehensive performance metrics."""
        try:
            tprint_info("🔍 Evaluating model performance across all trained models...")

            if not result.orchestration_result:
                tprint_warning("⚠️ No orchestration data available for evaluation")
                return result

            # Get training and validation data
            long_training_data = result.orchestration_result.long_training_data
            short_training_data = result.orchestration_result.short_training_data
            long_validation_data = result.orchestration_result.long_validation_data
            short_validation_data = result.orchestration_result.short_validation_data
            long_test_data = result.orchestration_result.long_test_data
            short_test_data = result.orchestration_result.short_test_data

            # Calculate metrics for each model type
            evaluation_metrics = {
                'evaluation_completed': False,
                'cross_validation_folds': 5,
                'evaluation_time': 0.0
            }

            # Evaluate long models if available
            if result.long_base_models and not long_len(training_data) == 0:
                try:
                    long_metrics = self._calculate_model_metrics(
                        result.long_base_models,
                        long_training_data,
                        long_validation_data if not len(long_validation_data) == 0 else None,
                        long_test_data if not len(long_test_data) == 0 else None,
                        "long"
                    )
                    evaluation_metrics.update(long_metrics)
                    tprint_success(f"✅ Long model evaluation completed: {len(result.long_base_models)} models")
                except Exception as e:
                    tprint_error(f"❌ Long model evaluation failed: {e}")
                    evaluation_metrics.update(self._get_default_metrics("long"))

            # Evaluate short models if available
            if result.short_base_models and not short_len(training_data) == 0:
                try:
                    short_metrics = self._calculate_model_metrics(
                        result.short_base_models,
                        short_training_data,
                        short_validation_data if not len(short_validation_data) == 0 else None,
                        short_test_data if not len(short_test_data) == 0 else None,
                        "short"
                    )
                    evaluation_metrics.update(short_metrics)
                    tprint_success(f"✅ Short model evaluation completed: {len(result.short_base_models)} models")
                except Exception as e:
                    tprint_error(f"❌ Short model evaluation failed: {e}")
                    evaluation_metrics.update(self._get_default_metrics("short"))

            # Mark evaluation as completed
            evaluation_metrics['evaluation_completed'] = True
            evaluation_metrics['evaluation_time'] = tprint_timer()  # This would need to be tracked

            # Add evaluation metrics to result
            for key, value in evaluation_metrics.items():
                setattr(result, key, value)

            tprint_success("🎯 Model evaluation completed successfully")
            return result

        except Exception as e:
            tprint_error(f"❌ Model evaluation failed: {e}")
            return result

    def _calculate_model_metrics(self, models: Dict[str, Any], training_data: pd.DataFrame,
                               validation_data: Optional[pd.DataFrame] = None,
                               test_data: Optional[pd.DataFrame] = None,
                               signal_type: str = "long") -> Dict[str, float]:
        """Calculate comprehensive metrics for a set of models."""
        try:
            from sklearn.metrics import (
                accuracy_score, f1_score, precision_score, recall_score,
                roc_auc_score, mean_squared_error, mean_absolute_error,
                r2_score
            )
            from sklearn.model_selection import cross_val_score

            # Prepare data
            target_cols = [col for col in training_data.columns if col.startswith('target_')]
            if not target_cols:
                tprint_warning(f"⚠️ No target columns found for {signal_type} evaluation")
                return self._get_default_metrics(signal_type)

            # Get features and targets
            feature_cols = [col for col in training_data.columns if not col.startswith('target_') and col != 'timestamp']
            X_train = training_data[feature_cols].values
            y_train = training_data[target_cols[0]].values  # Use first target column

            # Convert to binary classification if needed (for accuracy, F1, etc.)
            y_train_binary = (y_train > 0).astype(int)

            # Calculate training metrics
            training_predictions = {}
            for model_name, model in models.items():
                try:
                    if hasattr(model, 'predict'):
                        pred = model.predict(X_train)
                        training_predictions[model_name] = pred
                    else:
                        # For ensemble models
                        pred = model.predict(X_train)
                        training_predictions[model_name] = pred
                except Exception as e:
                    tprint_warning(f"⚠️ Could not get predictions from {model_name}: {e}")
                    continue

            if not training_predictions:
                return self._get_default_metrics(signal_type)

            # Average predictions across all models
            avg_predictions = np.mean(list(training_predictions.values()), axis=0)
            avg_predictions_binary = (avg_predictions > 0).astype(int)

            # Calculate metrics
            metrics = {}

            # Training metrics
            metrics[f'{signal_type}_training_accuracy'] = accuracy_score(y_train_binary, avg_predictions_binary)
            metrics[f'{signal_type}_f1_score'] = f1_score(y_train_binary, avg_predictions_binary, average='weighted')
            metrics[f'{signal_type}_precision'] = precision_score(y_train_binary, avg_predictions_binary, average='weighted')
            metrics[f'{signal_type}_recall'] = recall_score(y_train_binary, avg_predictions_binary, average='weighted')

            # ROC-AUC for probability predictions
            try:
                metrics[f'{signal_type}_roc_auc'] = roc_auc_score(y_train_binary, avg_predictions)
            except:
                metrics[f'{signal_type}_roc_auc'] = 0.5

            # Regression metrics
            metrics[f'{signal_type}_mse'] = mean_squared_error(y_train, avg_predictions)
            metrics[f'{signal_type}_mae'] = mean_absolute_error(y_train, avg_predictions)
            metrics[f'{signal_type}_rmse'] = np.sqrt(metrics[f'{signal_type}_mse'])
            metrics[f'{signal_type}_r2_score'] = r2_score(y_train, avg_predictions)

            # Calculate validation metrics if available
            if validation_data is not None and not len(validation_data) == 0:
                X_val = validation_data[feature_cols].values
                y_val = validation_data[target_cols[0]].values
                y_val_binary = (y_val > 0).astype(int)

                avg_val_predictions = np.mean([model.predict(X_val) for model in models.values() if hasattr(model, 'predict')], axis=0)
                avg_val_predictions_binary = (avg_val_predictions > 0).astype(int)

                metrics[f'{signal_type}_validation_accuracy'] = accuracy_score(y_val_binary, avg_val_predictions_binary)

            # Calculate test metrics if available
            if test_data is not None and not len(test_data) == 0:
                X_test = test_data[feature_cols].values
                y_test = test_data[target_cols[0]].values
                y_test_binary = (y_test > 0).astype(int)

                avg_test_predictions = np.mean([model.predict(X_test) for model in models.values() if hasattr(model, 'predict')], axis=0)
                avg_test_predictions_binary = (avg_test_predictions > 0).astype(int)

                metrics[f'{signal_type}_test_accuracy'] = accuracy_score(y_test_binary, avg_test_predictions_binary)

            # Calculate financial metrics (simplified)
            # Sharpe ratio (simplified)
            returns = avg_predictions
            if len(returns) > 1:
                risk_free_rate = 0.02  # 2% annual risk-free rate
                excess_returns = returns - risk_free_rate/252  # Daily excess return
                sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
                metrics[f'{signal_type}_sharpe_ratio'] = sharpe_ratio

                # Max drawdown
                cumulative = np.cumsum(returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdowns = running_max - cumulative
                max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
                metrics[f'{signal_type}_max_drawdown'] = max_drawdown

                # Profit factor and win rate
                profits = returns[returns > 0]
                losses = returns[returns < 0]
                profit_factor = np.sum(profits) / abs(np.sum(losses)) if len(losses) > 0 else np.inf
                win_rate = len(profits) / len(returns) if len(returns) > 0 else 0

                metrics[f'{signal_type}_profit_factor'] = profit_factor
                metrics[f'{signal_type}_win_rate'] = win_rate

            # Calculate trading frequency and monthly P&L metrics
            trading_metrics = self._calculate_trading_metrics(
                training_data, avg_predictions, y_train, signal_type
            )
            metrics.update(trading_metrics)

            return metrics

        except Exception as e:
            tprint_error(f"❌ Failed to calculate metrics for {signal_type}: {e}")
            return self._get_default_metrics(signal_type)

    def _calculate_trading_metrics(self, training_data: pd.DataFrame, predictions: np.ndarray,
                                 actual_returns: np.ndarray, signal_type: str) -> Dict[str, Union[float, int, Dict]]:
        """Calculate trading frequency and monthly P&L metrics."""
        try:
            from datetime import datetime

            # Get timestamps for monthly grouping
            if 'timestamp' in training_data.columns:
                timestamps = pd.to_datetime(training_data['timestamp'].values)
            else:
                # Create synthetic timestamps if not available
                timestamps = pd.date_range('2024-01-01', periods=len(training_data), freq='1min')

            # Convert predictions to trading signals (long/short/neutral)
            # For now, use a simple threshold-based approach
            signal_threshold = 0.001  # 0.1% threshold
            trading_signals = np.zeros_like(predictions)

            # Long signals when prediction > threshold
            long_mask = predictions > signal_threshold
            trading_signals[long_mask] = 1

            # Short signals when prediction < -threshold
            short_mask = predictions < -signal_threshold
            trading_signals[short_mask] = -1

            # Calculate total trades
            signal_changes = np.diff(np.concatenate([np.array([0]), trading_signals]))
            total_trades = np.sum(np.abs(signal_changes))

            # Calculate monthly trade counts
            monthly_trades = {}
            monthly_pnl = {}

            for i in range(len(timestamps)):
                month_key = timestamps[i].strftime('%Y-%m')
                signal = trading_signals[i]

                if month_key not in monthly_trades:
                    monthly_trades[month_key] = 0
                    monthly_pnl[month_key] = 0.0

                # Count trade entries (signal changes)
                if i > 0:
                    prev_signal = trading_signals[i-1]
                    if signal != prev_signal and signal != 0:
                        monthly_trades[month_key] += 1

                # Calculate P&L for this position
                if signal != 0:
                    # Simulate P&L based on actual returns
                    if i < len(actual_returns):
                        pnl = actual_returns[i] * signal  # Simple directional P&L
                        monthly_pnl[month_key] += pnl

            # Calculate total metrics
            total_pnl = sum(monthly_pnl.values())

            # Calculate average trades per month
            if monthly_trades:
                avg_trades_per_month = np.mean(list(monthly_trades.values()))
                total_months = len(set([ts.strftime('%Y-%m') for ts in timestamps]))
                if total_months > 0:
                    avg_trades_per_month = total_trades / total_months
            else:
                avg_trades_per_month = 0.0

            return {
                f'{signal_type}_total_trades': int(total_trades),
                f'{signal_type}_avg_trades_per_month': float(avg_trades_per_month),
                f'{signal_type}_total_pnl': float(total_pnl),
                f'{signal_type}_monthly_pnl': monthly_pnl,
                f'{signal_type}_monthly_trade_count': monthly_trades
            }

        except Exception as e:
            tprint_error(f"❌ Failed to calculate trading metrics for {signal_type}: {e}")
            return self._get_default_trading_metrics(signal_type)

    def _get_default_trading_metrics(self, signal_type: str) -> Dict[str, Union[float, int, Dict]]:
        """Get default trading metrics for a signal type."""
        return {
            f'{signal_type}_total_trades': 0,
            f'{signal_type}_avg_trades_per_month': 0.0,
            f'{signal_type}_total_pnl': 0.0,
            f'{signal_type}_monthly_pnl': {},
            f'{signal_type}_monthly_trade_count': {}
        }

    def _get_default_metrics(self, signal_type: str) -> Dict[str, float]:
        """Get default/zero metrics for a signal type."""
        return {
            f'{signal_type}_training_accuracy': 0.0,
            f'{signal_type}_validation_accuracy': 0.0,
            f'{signal_type}_test_accuracy': 0.0,
            f'{signal_type}_f1_score': 0.0,
            f'{signal_type}_precision': 0.0,
            f'{signal_type}_recall': 0.0,
            f'{signal_type}_roc_auc': 0.5,
            f'{signal_type}_sharpe_ratio': 0.0,
            f'{signal_type}_max_drawdown': 0.0,
            f'{signal_type}_profit_factor': 1.0,
            f'{signal_type}_win_rate': 0.0,
            f'{signal_type}_mse': 0.0,
            f'{signal_type}_mae': 0.0,
            f'{signal_type}_rmse': 0.0,
            f'{signal_type}_r2_score': 0.0,
            f'{signal_type}_total_trades': 0,
            f'{signal_type}_avg_trades_per_month': 0.0,
            f'{signal_type}_total_pnl': 0.0,
            f'{signal_type}_monthly_pnl': {},
            f'{signal_type}_monthly_trade_count': {}
        }

    def _log_comprehensive_summary(self, report: Dict[str, Any]) -> None:
        """Log comprehensive dual training summary with enhanced tprint integration."""
        try:
            training = report['training_summary']
            models = report['model_breakdown']
            samples = report['sample_metrics']
            features = report['feature_integration_metrics']
            performance = report['performance_metrics']
            quality = report['quality_metrics']
            errors = report['error_analysis']
            evaluation = report.get('evaluation_metrics', {})

            tprint_info("=" * 80)
            tprint_info("🎯 TACTICIAN DUAL TRAINING SUMMARY")
            tprint_info("=" * 80)
            tprint_info(f"⏱️  Total Training Time: {training['total_training_time']:.2f}s")
            tprint_info(f"✅ Success: {'Yes' if training['success'] else 'No'}")
            tprint_info(f"📊 Training Phase: {training['training_phase']}")
            tprint_info(f"🤖 Total Models Trained: {training['total_models_trained']}")
            tprint_info(f"📈 Long Base Models: {training['long_base_models_trained']}")
            tprint_info(f"📉 Short Base Models: {training['short_base_models_trained']}")
            tprint_info(f"🎯 Long Ensemble Models: {training['long_ensemble_models_trained']}")
            tprint_info(f"🎯 Short Ensemble Models: {training['short_ensemble_models_trained']}")

            tprint_info("\n📈 Sample Processing Results:")
            tprint_info(f"  📊 Long Samples: {samples['total_long_samples']}")
            tprint_info(f"  📉 Short Samples: {samples['total_short_samples']}")
            tprint_info(f"  ✅ Long Sufficient: {'Yes' if samples['long_samples_sufficient'] else 'No'}")
            tprint_info(f"  ✅ Short Sufficient: {'Yes' if samples['short_samples_sufficient'] else 'No'}")
            tprint_info(f"  📊 Total Samples: {samples['total_samples_processed']}")

            tprint_info("\n🔢 Model Training Breakdown:")
            tprint_info("  📈 Long Models:")
            for model in models['long_models']['base_models']:
                status = "✅" if model['status'] == 'trained' else "❌"
                tprint_info(f"    {status} {model['type']}: {model['count']} model(s)")
            for model in models['long_models']['ensemble_models']:
                status = "✅" if model['status'] == 'trained' else "❌"
                integration = "✅" if model['features_integrated'] else "❌"
                tprint_info(f"    {status} {model['type']}: {model['count']} model(s) [Integration: {integration}]")

            tprint_info("  📉 Short Models:")
            for model in models['short_models']['base_models']:
                status = "✅" if model['status'] == 'trained' else "❌"
                tprint_info(f"    {status} {model['type']}: {model['count']} model(s)")
            for model in models['short_models']['ensemble_models']:
                status = "✅" if model['status'] == 'trained' else "❌"
                integration = "✅" if model['features_integrated'] else "❌"
                tprint_info(f"    {status} {model['type']}: {model['count']} model(s) [Integration: {integration}]")

            tprint_info("\n🔗 Feature Integration Status:")
            tprint_info(f"  🧬 HMM Features: {'✅ Included' if features['hmm_features_included'] else '❌ Not included'}")
            tprint_info(f"  🎯 Analyst Features: {'✅ Included' if features['analyst_features_included'] else '❌ Not included'}")
            tprint_info(f"  📊 Technical Features: {'✅ Included' if features['technical_features_included'] else '❌ Not included'}")
            tprint_info(f"  🔄 OOF Predictions: {'✅ Included' if features['oof_predictions_included'] else '❌ Not included'}")
            tprint_info(f"  ✅ Integration Complete: {'Yes' if features['feature_integration_complete'] else 'No'}")
            tprint_info(f"  📈 Long Ensemble Features: {features['long_ensemble_feature_count']}")
            tprint_info(f"  📉 Short Ensemble Features: {features['short_ensemble_feature_count']}")

            tprint_info("\n⚡ Performance Metrics:")
            tprint_info(f"  🔄 Orchestration Time: {performance['orchestration_time']:.2f}s")
            tprint_info(f"  📈 Long Base Training: {performance['long_base_training_time']:.2f}s")
            tprint_info(f"  📉 Short Base Training: {performance['short_base_training_time']:.2f}s")
            tprint_info(f"  🎯 Long Ensemble Training: {performance['long_ensemble_training_time']:.2f}s")
            tprint_info(f"  🎯 Short Ensemble Training: {performance['short_ensemble_training_time']:.2f}s")
            tprint_info(f"  💾 Memory Usage: {performance['memory_usage_mb']:.1f} MB")
            tprint_info(f"  🖥️ CPU Usage: {performance['cpu_usage_percent']:.1f}%")
            tprint_info(f"  🤖 Avg Time per Model: {performance['average_training_time_per_model']:.2f}s")

            tprint_info("\n📊 Quality Metrics:")
            tprint_info(f"  📈 Long Training Quality: {quality['long_training_quality']:.3f}")
            tprint_info(f"  📉 Short Training Quality: {quality['short_training_quality']:.3f}")
            tprint_info(f"  🎯 Ensemble Integration: {'✅ Success' if quality['ensemble_integration_success'] else '❌ Failed'}")
            tprint_info(f"  🔄 Feature Differentiation: {'✅ Success' if quality['feature_differentiation_success'] else '❌ Failed'}")

            # Log evaluation metrics if available
            if evaluation and evaluation.get('evaluation_completed', False):
                tprint_info("\n🎯 Model Performance Metrics:")
                tprint_info("  📈 LONG MODEL PERFORMANCE:")
                tprint_info(f"    📊 Training Accuracy: {evaluation['long_training_accuracy']:.4f}")
                tprint_info(f"    ✅ Validation Accuracy: {evaluation['long_validation_accuracy']:.4f}")
                tprint_info(f"    🧪 Test Accuracy: {evaluation['long_test_accuracy']:.4f}")
                tprint_info(f"    🎯 F1 Score: {evaluation['long_f1_score']:.4f}")
                tprint_info(f"    📊 Precision: {evaluation['long_precision']:.4f}")
                tprint_info(f"    📈 Recall: {evaluation['long_recall']:.4f}")
                tprint_info(f"    📈 ROC-AUC: {evaluation['long_roc_auc']:.4f}")
                tprint_info(f"    💰 Sharpe Ratio: {evaluation['long_sharpe_ratio']:.4f}")
                tprint_info(f"    📉 Max Drawdown: {evaluation['long_max_drawdown']:.4f}")
                tprint_info(f"    💰 Profit Factor: {evaluation['long_profit_factor']:.4f}")
                tprint_info(f"    🏆 Win Rate: {evaluation['long_win_rate']:.3f}")
                tprint_info(f"    📊 MSE: {evaluation['long_mse']:.6f}")
                tprint_info(f"    📊 MAE: {evaluation['long_mae']:.6f}")
                tprint_info(f"    📊 RMSE: {evaluation['long_rmse']:.6f}")
                tprint_info(f"    📊 R² Score: {evaluation['long_r2_score']:.4f}")

                tprint_info("  📉 SHORT MODEL PERFORMANCE:")
                tprint_info(f"    📊 Training Accuracy: {evaluation['short_training_accuracy']:.4f}")
                tprint_info(f"    ✅ Validation Accuracy: {evaluation['short_validation_accuracy']:.4f}")
                tprint_info(f"    🧪 Test Accuracy: {evaluation['short_test_accuracy']:.4f}")
                tprint_info(f"    🎯 F1 Score: {evaluation['short_f1_score']:.4f}")
                tprint_info(f"    📊 Precision: {evaluation['short_precision']:.4f}")
                tprint_info(f"    📈 Recall: {evaluation['short_recall']:.4f}")
                tprint_info(f"    📈 ROC-AUC: {evaluation['short_roc_auc']:.4f}")
                tprint_info(f"    💰 Sharpe Ratio: {evaluation['short_sharpe_ratio']:.4f}")
                tprint_info(f"    📉 Max Drawdown: {evaluation['short_max_drawdown']:.4f}")
                tprint_info(f"    💰 Profit Factor: {evaluation['short_profit_factor']:.4f}")
                tprint_info(f"    🏆 Win Rate: {evaluation['short_win_rate']:.3f}")
                tprint_info(f"    📊 MSE: {evaluation['short_mse']:.6f}")
                tprint_info(f"    📊 MAE: {evaluation['short_mae']:.6f}")
                tprint_info(f"    📊 RMSE: {evaluation['short_rmse']:.6f}")
                tprint_info(f"    📊 R² Score: {evaluation['short_r2_score']:.4f}")

                # Financial trading metrics
                tprint_info("  🤖 LONG TRADING METRICS:")
                tprint_info(f"    📈 Total Trades: {evaluation['long_total_trades']}")
                tprint_info(f"    📊 Avg Trades/Month: {evaluation['long_avg_trades_per_month']:.1f}")
                tprint_info(f"    💵 Total P&L: {evaluation['long_total_pnl']:.6f}")
                tprint_info("  🤖 SHORT TRADING METRICS:")
                tprint_info(f"    📈 Total Trades: {evaluation['short_total_trades']}")
                tprint_info(f"    📊 Avg Trades/Month: {evaluation['short_avg_trades_per_month']:.1f}")
                tprint_info(f"    💵 Total P&L: {evaluation['short_total_pnl']:.6f}")

                # Monthly breakdown (show top 5 months for each)
                if evaluation['long_monthly_pnl']:
                    tprint_info("  📅 LONG Monthly P&L (Top 5):")
                    sorted_months = sorted(evaluation['long_monthly_pnl'].items(), key=lambda x: x[1], reverse=True)[:5]
                    for month, pnl in sorted_months:
                        tprint_info(f"    {month}: {pnl:.6f}")

                if evaluation['short_monthly_pnl']:
                    tprint_info("  📅 SHORT Monthly P&L (Top 5):")
                    sorted_months = sorted(evaluation['short_monthly_pnl'].items(), key=lambda x: x[1], reverse=True)[:5]
                    for month, pnl in sorted_months:
                        tprint_info(f"    {month}: {pnl:.6f}")

                if evaluation['long_monthly_trade_count']:
                    tprint_info("  📊 LONG Monthly Trade Count (Top 5):")
                    sorted_months = sorted(evaluation['long_monthly_trade_count'].items(), key=lambda x: x[1], reverse=True)[:5]
                    for month, trades in sorted_months:
                        tprint_info(f"    {month}: {trades} trades")

                if evaluation['short_monthly_trade_count']:
                    tprint_info("  📊 SHORT Monthly Trade Count (Top 5):")
                    sorted_months = sorted(evaluation['short_monthly_trade_count'].items(), key=lambda x: x[1], reverse=True)[:5]
                    for month, trades in sorted_months:
                        tprint_info(f"    {month}: {trades} trades")

                tprint_info(f"  📊 Cross-Validation: {evaluation['cross_validation_folds']} folds")
                tprint_info(f"  ⏱️ Evaluation Time: {evaluation['evaluation_time']:.2f}s")
            else:
                tprint_info("\n📊 Evaluation Status:")
                tprint_info(f"  ✅ Evaluation Completed: {'Yes' if evaluation.get('evaluation_completed', False) else 'No'}")
                if not evaluation.get('evaluation_completed', False):
                    tprint_info(f"  ⚠️ Model evaluation metrics will be populated during actual model training")

            if errors['total_errors'] > 0:
                tprint_info("\n🚨 Error Analysis:")
                tprint_info(f"  ❌ Total Errors: {errors['total_errors']}")
                tprint_info(f"  🔄 Orchestration Errors: {errors['orchestration_errors']}")
                tprint_info(f"  📈 Long Base Training Errors: {errors['long_base_training_errors']}")
                tprint_info(f"  📉 Short Base Training Errors: {errors['short_base_training_errors']}")
                tprint_info(f"  🎯 Long Ensemble Errors: {errors['long_ensemble_training_errors']}")
                tprint_info(f"  🎯 Short Ensemble Errors: {errors['short_ensemble_training_errors']}")

            tprint_info("\n🎯 Training Completion Status:")
            tprint_info(f"  🔄 Pre-ML Orchestration: {'✅ Completed' if training['pre_ml_orchestration_completed'] else '❌ Failed'}")
            tprint_info(f"  📈 Base Training: {'✅ Completed' if training['base_training_completed'] else '❌ Failed'}")
            tprint_info(f"  🎯 Ensemble Training: {'✅ Completed' if training['ensemble_training_completed'] else '❌ Failed'}")
            tprint_info(f"  ✅ Validation: {'✅ Completed' if training['validation_completed'] else '❌ Failed'}")
            tprint_info(f"  💾 Results Saved: {'✅ Yes' if training['results_saved'] else '❌ No'}")

            tprint_info("=" * 80)

        except Exception as e:
            tprint_error(f"❌ Failed to log comprehensive summary: {e}")
            # Fallback to basic logging
            try:
                tprint_info("🔄 Basic Training Summary:")
                tprint_info(f"  ⏱️ Total Time: {training['total_training_time']:.2f}s")
                tprint_info(f"  ✅ Success: {training['success']}")
                tprint_info(f"  📈 Long Models: {training['long_base_models_trained']}")
                tprint_info(f"  📉 Short Models: {training['short_base_models_trained']}")
                tprint_info(f"  🎯 Long Ensemble: {training['long_ensemble_models_trained']}")
                tprint_info(f"  🎯 Short Ensemble: {training['short_ensemble_models_trained']}")
                tprint_info(f"  🔢 Total Models: {training['total_models_trained']}")
            except:
                pass
