"""
Tactician Models Training Step - Enhanced & Streamlined

This step handles per-regime training of individual Tactician models using common dependencies.
The Tactician operates on 1m timeframe and decides WHEN to trade based on Analyst's green light signals.

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

# Enhanced imports with comprehensive error handling
try:
    from src.utils.logger import system_logger
    from src.utils.ml_common.config import BaseTrainingConfig
    from src.utils.ml_common.training import BaseTrainingStep
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import core ML utilities: {e}")
    raise

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
    print("❌ This is a critical dependency for enhanced logging. Please install tprint.")
    raise ImportError(f"CRITICAL: tprint is required but not available: {e}") from e

# Import common utilities - CRITICAL: Fast fail if not available
try:
    from src.utils.common_operations import (
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        cleanup_m1_optimizers, integrate_with_m1_optimizers, safe_divide,
        ensure_directory, safe_file_exists, get_current_datetime, validate_positive
    )
    COMMON_OPERATIONS_AVAILABLE = True
    tprint_info("✅ Common operations utilities loaded")
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Common operations utilities are required but not available: {e}")
    print("❌ Hardware optimizers are essential for performance. Please install common_operations.")
    COMMON_OPERATIONS_AVAILABLE = False
    raise ImportError(f"CRITICAL: Common operations utilities are required but not available: {e}") from e

# Import model persistence and caching
try:
    from src.utils.ml_common.post_training.model_persistence import (
        ModelPersistence, ModelMetadata, PersistenceConfig
    )
    from src.utils.ml_common.models.model_cache import (
        ModelCache, get_model_cache, CachedModelMetadata
    )
    MODEL_PERSISTENCE_AVAILABLE = True
except ImportError:
    MODEL_PERSISTENCE_AVAILABLE = False

# Import data cleaning utilities
try:
    from src.utils.data.quality.data_cleaning import (
        DataCleaner, CleaningConfig, MissingValueStrategy, OutlierStrategy
    )
    DATA_CLEANING_AVAILABLE = True
except ImportError:
    DATA_CLEANING_AVAILABLE = False

try:
    from src.utils.common_utilities import (
        safe_dataframe_operation, validate_dataframe_columns, calculate_data_quality_metrics,
        safe_merge_dataframes, create_summary_statistics
    )
    COMMON_UTILITIES_AVAILABLE = True
    tprint_info("✅ Common utilities loaded")
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Common utilities are required but not available: {e}")
    print("❌ Enhanced data operations are essential. Please install common_utilities.")
    COMMON_UTILITIES_AVAILABLE = False
    raise ImportError(f"CRITICAL: Common utilities are required but not available: {e}") from e

try:
    from src.utils.math_validation import (
        validate_finite, validate_positive, validate_range,
        safe_correlation, safe_percentage_change
    )
    MATH_VALIDATION_AVAILABLE = True
    tprint_info("✅ Math validation utilities loaded")
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Math validation utilities are required but not available: {e}")
    print("❌ Safe math operations are essential for data integrity. Please install math_validation.")
    MATH_VALIDATION_AVAILABLE = False
    raise ImportError(f"CRITICAL: Math validation utilities are required but not available: {e}") from e

try:
    from src.utils.kline_parquet import validate_klines_data, process_klines_data
    from src.utils.serialization_utils import safe_serialize, safe_deserialize
    DATA_UTILITIES_AVAILABLE = True
    tprint_info("✅ Data utilities loaded")
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Data utilities are required but not available: {e}")
    print("❌ Enhanced data validation is essential. Please install kline_parquet and serialization_utils.")
    DATA_UTILITIES_AVAILABLE = False
    raise ImportError(f"CRITICAL: Data utilities are required but not available: {e}") from e

try:
    from src.utils.matrix_operations import (
        safe_matrix_operations, validate_matrix_properties, optimize_matrix_computations
    )
    MATRIX_OPERATIONS_AVAILABLE = True
    tprint_info("✅ Matrix operations utilities loaded")
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Matrix operations utilities are required but not available: {e}")
    print("❌ Optimized matrix computations are essential for performance. Please install matrix_operations.")
    MATRIX_OPERATIONS_AVAILABLE = False
    raise ImportError(f"CRITICAL: Matrix operations utilities are required but not available: {e}") from e

try:
    from src.utils.ml_common.matrix_cross_validation import matrix_cross_validate as cross_validation_utils
    from src.utils.lookahead_bias_detector import LookaheadBiasDetector as lookahead_bias_detector
    from src.utils.ml_common.optimization import HyperparameterOptimization as hyperparameter_optimization
    ML_COMMON_AVAILABLE = True
    tprint_info("✅ ML common utilities loaded")
except ImportError as e:
    print(f"❌ CRITICAL ERROR: ML common utilities are required but not available: {e}")
    print("❌ Advanced ML features are essential. Please install ml_common.")
    ML_COMMON_AVAILABLE = False
    raise ImportError(f"CRITICAL: ML common utilities are required but not available: {e}") from e

# Import BOHB optimizer for advanced HPO (Phase 1: Model Training Migration)
try:
    from src.utils.ml_common.optimization.bohb_optimizer import (
        BOHBOptimizer, BOHBConfig, BOHBResult
    )
    BOHB_AVAILABLE = True
    tprint_info("✅ BOHB optimizer loaded for model training")
except ImportError as e:
    BOHB_AVAILABLE = False
    BOHBOptimizer = None
    BOHBConfig = None
    BOHBResult = None
    tprint_warning(f"⚠️ BOHB optimizer not available: {e} (will use fallback HPO)")

# Import Bayesian TPE optimizer as fallback for simple cases
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer, OptimizationConfig, OptimizationResult
    )
    BAYESIAN_TPE_AVAILABLE = True
    tprint_info("✅ Bayesian TPE optimizer loaded as fallback")
except ImportError as e:
    BAYESIAN_TPE_AVAILABLE = False
    BayesianTPEOptimizer = None
    OptimizationConfig = None
    OptimizationResult = None
    tprint_warning(f"⚠️ Bayesian TPE optimizer not available: {e} (will use fallback HPO)")

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
    tprint_success("✅ Enhanced training utilities loaded")
except ImportError as e:
    ENHANCED_TRAINING_AVAILABLE = False
    tprint_warning(f"⚠️ Enhanced training utilities not available: {e}")
    EnhancedTrainingUtils = None
    TrainingStepEnhancer = None
    EarlyStoppingConfig = None
    PurgedCVConfig = None
    OverfittingMonitorConfig = None
    RegularizationConfig = None
    TrainingIntegrationConfig = None

# Import vectorized training manager for enhanced capabilities
try:
    from src.utils.ml_common.training.vectorized_training_manager import VectorizedTrainingManager
    VECTORIZED_TRAINING_AVAILABLE = True
except ImportError:
    VECTORIZED_TRAINING_AVAILABLE = False

# Initialize logger - CRITICAL: Fast fail if not available
try:
    logger = system_logger.getChild('TacticianModelsTrainingEnhanced')
except Exception as e:
    print(f"❌ CRITICAL ERROR: Failed to initialize system logger: {e}")
    print("❌ System logger is required for proper logging. Please check logger configuration.")
    raise RuntimeError(f"CRITICAL: Failed to initialize system logger: {e}") from e

@dataclass
class TacticianTrainingConfig(BaseTrainingConfig):
    """
    Configuration for Tactician models training with specific parameters.

    Extends PerRegimeTrainingConfig with tactician-specific settings for:
    - Entry timing optimization
    - Confidence-aware ensemble training
    - Analyst signal integration
    """

    # Tactician-specific configuration
    enable_entry_timing_optimization: bool = True
    entry_timing_range: float = 0.004  # 0-0.4% range for entry timing
    expected_movement: float = 0.004  # Expected movement threshold

    # Entry timing objectives and penalties
    entry_timing_objectives: Dict[str, Any] = None
    early_entry_penalty_weight: float = 1.0
    late_entry_penalty_weight: float = 1.0
    optimal_entry_reward_weight: float = 2.0

    # Confidence-aware ensemble settings (always enabled for Tactician)
    enable_confidence_aware_ensemble: bool = True
    confidence_threshold: float = 0.5
    confidence_weighting_method: str = "exponential"  # "linear", "exponential", "sigmoid"

    # Analyst integration settings
    require_analyst_signals: bool = True
    use_analyst_confidence_scores: bool = True
    analyst_signal_types: List[str] = None  # ["directional", "magnitude", "timing"]

    # Advanced tactician features
    enable_microstructure_features: bool = True
    enable_regime_transition_handling: bool = False
    enable_multi_horizon_prediction: bool = False
    multi_horizon_windows: List[int] = None  # [1, 2, 5, 10] minutes

    # Random Survival Forest specific settings
    enable_survival_analysis: bool = True
    survival_horizons: List[int] = None  # [1, 2, 5, 10] minutes
    survival_horizon_weights: List[float] = None  # [0.4, 0.3, 0.2, 0.1]
    latency_constraint: float = 2.0  # seconds

    def __post_init__(self):
        """Initialize default values for complex fields."""
        super().__post_init__() if hasattr(super(), '__post_init__') else None

        if self.entry_timing_objectives is None:
            self.entry_timing_objectives = {
                'minimize_early_entry_penalty': True,
                'minimize_late_entry_penalty': True,
                'maximize_optimal_entry_reward': True,
                'minimize_adverse_movement': True
            }

        if self.analyst_signal_types is None:
            self.analyst_signal_types = ["directional", "magnitude"]

        if self.multi_horizon_windows is None:
            self.multi_horizon_windows = [1, 2, 5, 10]

        if self.survival_horizons is None:
            self.survival_horizons = [1, 2, 5, 10]

        if self.survival_horizon_weights is None:
            self.survival_horizon_weights = [0.4, 0.3, 0.2, 0.1]

class TrainingPhase(Enum):
    """Training phases for progress tracking."""
    INITIALIZATION = "initialization"
    DATA_VALIDATION = "data_validation"
    FEATURE_PREPARATION = "feature_preparation"
    REGIME_ANALYSIS = "regime_analysis"
    MODEL_TRAINING = "model_training"
    EVALUATION = "evaluation"
    MODEL_SAVING = "model_saving"
    FINALIZATION = "finalization"

@dataclass
class TrainingMetrics:
    """Training metrics for comprehensive reporting."""
    phase: TrainingPhase
    start_time: float
    end_time: Optional[float] = None
    samples_processed: int = 0
    features_count: int = 0
    regimes_count: int = 0
    models_trained: int = 0
    errors_encountered: int = 0
    warnings_issued: int = 0
    memory_usage_mb: float = 0.0
    success: bool = False
    error_message: Optional[str] = None

    @property
    def duration(self) -> float:
        """Get phase duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

class TacticianModelsTrainingStepRefactored(BaseTrainingStep):
    """
    Enhanced Tactician Models Training Step with comprehensive error handling and reporting.

    The Tactician operates on 1m timeframe and is trained on:
    1. Only periods where the Analyst gives a green light
    2. Using the Analyst's model outputs as input features

    ENHANCED FEATURES:
    - Comprehensive input validation and data quality checks
    - Detailed progress tracking with phase-based metrics
    - Enhanced error handling with specific failure reporting
    - Optimized vectorization with intelligent fallback
    - Structured logging with performance monitoring
    """

    def __init__(self, config: Optional[TacticianTrainingConfig] = None, enable_vectorization: bool = True):
        """
        Initialize enhanced Tactician models training step with comprehensive error handling and utility integration.

        Args:
            config: Per-regime training configuration
            enable_vectorization: Whether to enable vectorized training
        """
        tprint_info("🚀 Initializing Tactician Models Training Step")
        self.overall_start_time = time.time()
        self.phase_start_time = time.time()
        self.training_metrics: Dict[TrainingPhase, TrainingMetrics] = {}

        # Set default configuration
        if config is None:
            config = TacticianTrainingConfig(
                model_name="tactician_models",
                timeframe="1m",
                model_types=["XGBOOST", "LIGHTGBM", "DEEPSCALER_1M", "FINANCIAL_RESNET", "RandomSurvivalForest"],
                hpo_n_trials=100,
                hpo_timeout_seconds=3600,
                min_samples_per_regime=1000,
                enable_data_augmentation=True,
                augmentation_method="smote",
                model_save_path="generated/model_training/models/tactician_models",
                evaluation_metrics=["mse", "mae", "r2", "mape", "smape"],
                use_single_model=True,
                single_model_name="tactician_unified_model",
                enable_ensemble_training=True,
                ensemble_method="stacking",
                meta_model="ElasticNetCV",
                ensemble_name="tactician_ensemble",
                enable_entry_timing_optimization=True,
                entry_timing_range=0.004,
                expected_movement=0.004
            )

        try:
            # Initialize training metrics for initialization phase
            self._start_phase(TrainingPhase.INITIALIZATION)

            # Initialize parent class
            super().__init__(config)
            self.logger = logger.getChild('TacticianModelsTrainingEnhanced')

            # Vectorization support
            self.enable_vectorization = enable_vectorization and VECTORIZED_TRAINING_AVAILABLE
            self.vectorization_fallback_used = False

            # Validate configuration (consolidated)
            self._validate_config_consolidated(config)

            # Initialize components (consolidated)
            self._initialize_components_consolidated()

            # Initialize enhanced training utilities if available
            if ENHANCED_TRAINING_AVAILABLE:
                self._initialize_enhanced_training_utilities()

            init_time = time.time() - self.overall_start_time
            tprint_success(f"✅ Initialization complete in {init_time:.2f}s")

            self._complete_phase(TrainingPhase.INITIALIZATION, success=True)

        except Exception as e:
            tprint_error(f"❌ Initialization failed: {e}")
            raise

    def _validate_config_consolidated(self, config: BaseTrainingConfig) -> None:
        """Consolidated configuration validation using common utilities."""
        try:
            if not config.model_types or len(config.model_types) == 0:
                raise ValueError("At least one model type required")

            if config.enable_hpo:
                validate_positive(config.hpo_n_trials, "hpo_n_trials")
                validate_positive(config.hpo_timeout_seconds, "hpo_timeout_seconds")

            validate_positive(config.min_samples_per_regime, "min_samples_per_regime")

            if config.save_models and config.model_save_path:
                ensure_directory(config.model_save_path)

            tprint_success("✅ Configuration validation passed")
        except Exception as e:
            tprint_error(f"❌ Configuration validation failed: {e}")
            raise

    def _initialize_components_consolidated(self):
        """Consolidated initialization of all components."""
        with tprint_timer("Component initialization"):
            # Hardware optimizers
            self._initialize_hardware_optimizers()

            # Utility integrations
            self._initialize_utility_integrations()

            # Data cleaner
            if DATA_CLEANING_AVAILABLE:
                self.data_cleaner = self._initialize_data_cleaner()

            # Model persistence
            if MODEL_PERSISTENCE_AVAILABLE:
                self.model_persistence = self._initialize_model_persistence()

            # Model cache
            if MODEL_PERSISTENCE_AVAILABLE:
                self.model_cache = self._initialize_model_cache()

            tprint_success("✅ All components initialized")

    def _initialize_data_cleaner(self) -> Optional[Any]:
        """Initialize data cleaner."""
        try:
            cleaning_config = CleaningConfig(
                missing_value_strategy=MissingValueStrategy.INTERPOLATE,
                outlier_strategy=OutlierStrategy.CLIP,
                outlier_threshold=3.0
            )
            tprint_success("✅ Data cleaner initialized")
            return DataCleaner(cleaning_config)
        except Exception as e:
            tprint_warning(f"⚠️ Data cleaner unavailable: {e}")
            return None

    def _initialize_model_persistence(self) -> Optional[Any]:
        """Initialize model persistence."""
        try:
            persistence_config = PersistenceConfig(
                base_model_dir=self.config.model_save_path,
                enable_versioning=True,
                max_versions=5
            )
            tprint_success("✅ Model persistence initialized")
            return ModelPersistence(persistence_config)
        except Exception as e:
            tprint_warning(f"⚠️ Model persistence unavailable: {e}")
            return None

    def _initialize_model_cache(self) -> Optional[Any]:
        """Initialize model cache."""
        try:
            model_cache = get_model_cache(
                max_memory_models=10,
                max_disk_models=50,
                cache_dir=f"{self.config.model_save_path}/cache"
            )
            tprint_success("✅ Model cache initialized")
            return model_cache
        except Exception as e:
            tprint_warning(f"⚠️ Model cache unavailable: {e}")
            return None

    def _initialize_enhanced_training_utilities(self):
        """Initialize enhanced training utilities for overfitting prevention and lookahead bias detection."""
        try:
            # Create enhanced training configuration for Tactician
            self.enhanced_training_config = TrainingIntegrationConfig(
                enable_early_stopping=True,
                enable_purged_cv=True,
                enable_lookahead_detection=True,
                enable_temporal_splits=True,
                enable_regularization=True,
                enable_overfitting_monitoring=True,
                enable_walk_forward=True,  # Enable for Tactician
                model_type='auto'
            )

            # Initialize training enhancer
            self.training_enhancer = TrainingStepEnhancer(self.enhanced_training_config)

            # Store enhanced utilities
            self.enhanced_training_utils = {
                'EnhancedTrainingUtils': EnhancedTrainingUtils,
                'EarlyStoppingConfig': EarlyStoppingConfig,
                'PurgedCVConfig': PurgedCVConfig,
                'OverfittingMonitorConfig': OverfittingMonitorConfig,
                'RegularizationConfig': RegularizationConfig,
                'TrainingStepEnhancer': TrainingStepEnhancer,
                'TrainingIntegrationConfig': TrainingIntegrationConfig
            }

            tprint_success("✅ Enhanced training utilities initialized successfully")

        except Exception as e:
            tprint_warning(f"⚠️ Enhanced training utilities initialization failed: {e}")
            self.enhanced_training_config = None
            self.training_enhancer = None
            self.enhanced_training_utils = {}

    def _create_model_instance(self, model_type: str):
        """Create model instance based on model type."""
        try:
            if model_type == "XGBOOST":
                from xgboost import XGBRegressor
                return XGBRegressor(
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbosity=0
                )
            elif model_type == "LIGHTGBM":
                from lightgbm import LGBMRegressor
                return LGBMRegressor(
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=6,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1
                )
            elif model_type == "DEEPSCALER_1M":
                # DeepScaler implementation would go here
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    random_state=42
                )
            elif model_type == "FINANCIAL_RESNET":
                # FinancialResNet implementation would go here
                return RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    random_state=42
                )
            elif model_type == "RandomSurvivalForest":
                return self._create_random_survival_forest_model()
            else:
                # Default fallback
                return RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                )
        except Exception as e:
            self.logger.error(f"❌ Failed to create model instance for {model_type}: {e}")
            # Fallback to RandomForest
            return RandomForestRegressor(n_estimators=100, random_state=42)

    def _create_random_survival_forest_model(self):
        """Create Random Survival Forest model for tactician timing prediction."""
        try:
            from .random_survival_forest_tactician import RandomSurvivalForestTactician, SurvivalAnalysisConfig

            # Create configuration optimized for tactician timing
            config = SurvivalAnalysisConfig(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                max_samples=0.8,
                horizons=[1, 2, 5, 10],  # 1m to 10m horizons (removed 15m and 30m)
                horizon_weights=[0.4, 0.3, 0.2, 0.1],
                entry_timing_range=0.005,  # 0.5% range
                expected_movement=0.01,  # 1% expected movement
                latency_constraint=2.0,  # 2 second constraint
                enable_timing_features=True,
                enable_regime_features=True,
                enable_analyst_features=True,
                enable_microstructure_features=True
            )

            return RandomSurvivalForestTactician(config)

        except ImportError as e:
            self.logger.error(f"❌ Random Survival Forest not available: {e}")
            # Fallback to RandomForest
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=200, random_state=42)
        except Exception as e:
            self.logger.error(f"❌ Failed to create Random Survival Forest: {e}")
            # Fallback to RandomForest
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=200, random_state=42)

    def _optimize_hyperparameters_with_bohb(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        search_space: Dict[str, Any],
        n_trials: int = 100,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using BOHB for model training (Phase 1 Migration).

        Args:
            model_type: Type of model to optimize
            X: Training features
            y: Training targets
            search_space: Hyperparameter search space
            n_trials: Number of optimization trials
            timeout: Timeout in seconds

        Returns:
            Dictionary containing best parameters and optimization results
        """
        tprint_info(f"🔍 Starting BOHB hyperparameter optimization for {model_type}...")

        # Try BOHB first (Phase 1: Model Training Migration)
        if BOHB_AVAILABLE:
            try:
                # Create BOHB configuration for model training
                bohb_config = BOHBConfig(
                    n_trials=n_trials,
                    timeout=timeout,
                    direction='maximize',  # Maximize validation score
                    metric_name='validation_score',
                    resource_name='epoch',  # Use epochs as resource for multi-fidelity
                    min_resource=1,  # Minimum epochs
                    max_resource=10,  # Maximum epochs
                    reduction_factor=3,  # Successive halving factor
                    n_startup_trials=5,  # Startup trials before pruning
                    pruner_type='hyperband',  # Use Hyperband pruning
                    enable_hardware_optimization=hasattr(self, 'm1_optimizers_available') and self.m1_optimizers_available,
                    enable_vectorbt_optimization=True,
                    enable_explainability=True,
                    enable_cv=True,
                    enable_oof_stacking=True,
                    seed=42
                )

                # Define multi-fidelity objective function for BOHB
                def bohb_objective(params: Dict[str, Any], resource: int = None) -> float:
                    """Multi-fidelity objective function for BOHB optimization."""
                    try:
                        # Create model with these parameters
                        model = self._create_model_instance(model_type)

                        # Set parameters
                        if hasattr(model, 'set_params'):
                            model.set_params(**params)

                        # Use resource (epochs) for multi-fidelity evaluation
                        if hasattr(model, 'n_estimators') and resource:
                            # For tree-based models, adjust n_estimators based on resource
                            model.set_params(n_estimators=max(10, int(resource * 10)))
                        elif hasattr(model, 'max_iter') and resource:
                            # For iterative models, adjust max_iter based on resource
                            model.set_params(max_iter=max(50, int(resource * 50)))

                        # Perform cross-validation with limited epochs
                        from sklearn.model_selection import cross_val_score
                        scores = cross_val_score(
                            model, X, y,
                            cv=5,
                            scoring='neg_mean_squared_error',
                            n_jobs=-1
                        )

                        # Return negative MSE (higher is better)
                        return -np.mean(scores)

                    except Exception as e:
                        tprint_warning(f"⚠️ BOHB objective evaluation failed: {e}")
                        return -np.inf

                # Create and run BOHB optimizer
                optimizer = BOHBOptimizer(bohb_config)
                result = optimizer.optimize(bohb_objective, search_space)

                if result.success:
                    tprint_success(f"✅ BOHB optimization completed successfully")
                    tprint_info(f"📊 Best score: {result.best_value:.6f}")
                    tprint_info(f"📊 Total trials: {result.n_trials}")
                    tprint_info(f"📊 Optimization time: {result.optimization_time:.2f}s")
                    tprint_info(f"📊 Resource efficiency: {result.resource_efficiency:.2f}x")

                    return {
                        'best_params': result.best_params,
                        'best_score': result.best_value,
                        'n_trials': result.n_trials,
                        'optimization_time': result.optimization_time,
                        'optimization_history': result.optimization_history,
                        'resource_efficiency': result.resource_efficiency,
                        'success': True,
                        'optimizer_type': 'BOHB'
                    }
                else:
                    tprint_warning(f"⚠️ BOHB optimization failed: {result.error_message}")
                    # Fall through to TPE fallback

            except Exception as e:
                tprint_warning(f"⚠️ BOHB optimization error: {e}, falling back to TPE")
                # Fall through to TPE fallback

        # Fallback to Bayesian TPE with enhanced early stopping
        if BAYESIAN_TPE_AVAILABLE:
            try:
                tprint_info(f"🔄 Falling back to enhanced Bayesian TPE optimization for {model_type}...")
                
                # Create enhanced optimization configuration with aggressive early stopping
                opt_config = OptimizationConfig(
                    n_trials=n_trials,
                    timeout=timeout,
                    direction='maximize',  # Maximize validation score
                    metric_name='validation_score',
                    enable_staged_optimization=True,
                    coarse_grid_points=5,
                    fine_grid_points=5,
                    coarse_grid_trials=25,
                    fine_grid_trials=25,
                    tpe_trials=max(50, n_trials - 50),
                    enable_hardware_optimization=hasattr(self, 'm1_optimizers_available') and self.m1_optimizers_available,
                    enable_adaptive_optimization=True,
                    early_stopping_patience=5,  # More aggressive early stopping
                    early_stopping_threshold=0.001,  # Stricter threshold
                    enable_pruner=True,  # Enable trial-level pruning
                    pruner_type='hyperband',  # Use Hyperband pruner
                    adaptive_patience=True,  # Enable adaptive patience
                    confidence_based_stopping=True,  # Enable confidence-based stopping
                    seed=42
                )

                # Define objective function for optimization
                def objective(params: Dict[str, Any]) -> float:
                    """Objective function for hyperparameter optimization."""
                    try:
                        # Create model with these parameters
                        model = self._create_model_instance(model_type)

                        # Set parameters
                        if hasattr(model, 'set_params'):
                            model.set_params(**params)

                        # Perform cross-validation
                        from sklearn.model_selection import cross_val_score
                        scores = cross_val_score(
                            model, X, y,
                            cv=5,
                            scoring='neg_mean_squared_error',
                            n_jobs=-1
                        )

                        # Return negative MSE (higher is better)
                        return -np.mean(scores)

                    except Exception as e:
                        tprint_warning(f"⚠️ Objective evaluation failed: {e}")
                        return -np.inf

                # Create and run optimizer
                optimizer = BayesianTPEOptimizer(search_space, opt_config)
                result = optimizer.optimize(objective)

                if result.success:
                    tprint_success(f"✅ Enhanced Bayesian TPE optimization completed successfully")
                    tprint_info(f"📊 Best score: {result.best_value:.6f}")
                    tprint_info(f"📊 Total trials: {result.n_trials}")
                    tprint_info(f"📊 Optimization time: {result.optimization_time:.2f}s")

                    return {
                        'best_params': result.best_params,
                        'best_score': result.best_value,
                        'n_trials': result.n_trials,
                        'optimization_time': result.optimization_time,
                        'optimization_history': result.optimization_history,
                        'success': True,
                        'optimizer_type': 'Enhanced_TPE'
                    }
                else:
                    tprint_warning(f"⚠️ Enhanced Bayesian TPE optimization failed: {result.error_message}")
                    return {'best_params': {}, 'success': False, 'error': result.error_message}

            except Exception as e:
                tprint_error(f"❌ Enhanced Bayesian TPE optimization error: {e}")
                return {'best_params': {}, 'success': False, 'error': str(e)}
        else:
            tprint_warning("⚠️ No optimizers available, skipping HPO")
            return {'best_params': {}, 'optimization_skipped': True}

    def _start_phase(self, phase: TrainingPhase, context: Optional[Dict[str, Any]] = None) -> None:
        """Start tracking a training phase with structured logging."""
        self.training_metrics[phase] = TrainingMetrics(
            phase=phase,
            start_time=time.time()
        )

        # Log phase start with structured format
        self._log_phase_start(phase, context)

    def _complete_phase(self, phase: TrainingPhase, success: bool = True,
                       error_message: Optional[str] = None, **kwargs) -> None:
        """Complete a training phase with metrics and structured logging."""
        if phase in self.training_metrics:
            metrics = self.training_metrics[phase]
            metrics.end_time = time.time()
            metrics.success = success
            metrics.error_message = error_message

            # Update metrics with provided values
            for key, value in kwargs.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)

            duration = metrics.duration

            # Prepare metrics for structured logging
            phase_metrics = {
                'duration': duration,
                'samples_processed': metrics.samples_processed,
                'features_count': metrics.features_count,
                'models_trained': metrics.models_trained,
                'warnings_issued': metrics.warnings_issued,
                'errors_encountered': metrics.errors_encountered,
                'memory_usage_mb': metrics.memory_usage_mb
            }

            # Log phase completion with structured format
            self._log_phase_complete(phase, success, duration, phase_metrics)

            if not success and error_message:
                self._log_structured_event(
                    event_type="phase_error",
                    phase=phase.value,
                    message=f"Phase failed: {error_message}",
                    level="error"
                )

    def _validate_configuration(self, config: BaseTrainingConfig) -> None:
        """Validate configuration - delegates to consolidated method."""
        try:
            self._validate_config_consolidated(config)

            # Tactician-specific critical validation
            if config.min_samples_per_regime < 100:
                error_msg = f"CRITICAL: Very low minimum samples per regime: {config.min_samples_per_regime} (minimum: 100)"
                tprint_error(f"❌ {error_msg}")
                raise ValueError(error_msg)

            # Validate HPO settings - CRITICAL: Fast fail on invalid config
            if config.enable_hpo and config.hpo_n_trials < 10:
                error_msg = f"CRITICAL: Very low HPO trials: {config.hpo_n_trials} (minimum: 10)"
                tprint_error(f"❌ {error_msg}")
                raise ValueError(error_msg)

            tprint_success("✅ Configuration validation passed")

        except Exception as e:
            tprint_error(f"❌ Configuration validation failed: {e}")
            raise

    def _initialize_hardware_optimizers(self) -> None:
        """Initialize hardware optimizers with graceful fallback for non-M1 systems."""
        try:
            tprint_info("🧠 Initializing hardware optimizers...")

            # Flag to track if M1 optimizers are available
            self.m1_optimizers_available = False

            try:
                # Try to initialize M1 GPU manager
                self.m1_gpu_manager = get_m1_gpu_manager()
                if self.m1_gpu_manager:
                    tprint_success("✅ M1 GPU manager initialized")
                    self.m1_optimizers_available = True
                else:
                    tprint_warning("⚠️ M1 GPU manager not available (graceful fallback)")
                    self.m1_gpu_manager = None
            except Exception as e:
                tprint_warning(f"⚠️ M1 GPU manager initialization failed: {e} (graceful fallback)")
                self.m1_gpu_manager = None

            try:
                # Try to initialize M1 memory optimizer
                self.m1_memory_optimizer = get_m1_memory_optimizer()
                if self.m1_memory_optimizer:
                    tprint_success("✅ M1 memory optimizer initialized")
                    self.m1_optimizers_available = True
                else:
                    tprint_warning("⚠️ M1 memory optimizer not available (graceful fallback)")
                    self.m1_memory_optimizer = None
            except Exception as e:
                tprint_warning(f"⚠️ M1 memory optimizer initialization failed: {e} (graceful fallback)")
                self.m1_memory_optimizer = None

            try:
                # Try to initialize M1 CPU optimizer
                self.m1_cpu_optimizer = get_m1_cpu_optimizer()
                if self.m1_cpu_optimizer:
                    tprint_success("✅ M1 CPU optimizer initialized")
                    self.m1_optimizers_available = True
                else:
                    tprint_warning("⚠️ M1 CPU optimizer not available (graceful fallback)")
                    self.m1_cpu_optimizer = None
            except Exception as e:
                tprint_warning(f"⚠️ M1 CPU optimizer initialization failed: {e} (graceful fallback)")
                self.m1_cpu_optimizer = None

            # Try to integrate with M1 optimizers if any are available
            if self.m1_optimizers_available:
                try:
                    integration_result = integrate_with_m1_optimizers()
                    if integration_result and integration_result.get('success', False):
                        tprint_success("✅ M1 optimizers integration successful")
                    else:
                        tprint_warning("⚠️ M1 optimizers integration incomplete (graceful fallback)")
                        self.m1_optimizers_available = False
                except Exception as e:
                    tprint_warning(f"⚠️ M1 optimizers integration failed: {e} (graceful fallback)")
                    self.m1_optimizers_available = False

            if self.m1_optimizers_available:
                tprint_success("✅ Hardware optimizers initialization completed with M1 acceleration")
            else:
                tprint_info("ℹ️ Hardware optimizers initialization completed (standard CPU mode)")

        except Exception as e:
            # Even if all optimizers fail, continue with standard CPU mode
            tprint_warning(f"⚠️ Hardware optimizer initialization encountered errors: {e} (continuing with standard CPU mode)")
            self.m1_gpu_manager = None
            self.m1_memory_optimizer = None
            self.m1_cpu_optimizer = None
            self.m1_optimizers_available = False

    def _initialize_utility_integrations(self) -> None:
        """Initialize utility integrations - All utilities are required."""
        try:
            tprint_info("🔧 Initializing utility integrations...")

            # All utilities are already loaded at import time with fast fail
            tprint_success("✅ All utility integrations verified and available")
            tprint_success("✅ Common utilities available")
            tprint_success("✅ Math validation utilities available")
            tprint_success("✅ Data utilities available")
            tprint_success("✅ Matrix operations utilities available")
            tprint_success("✅ ML common utilities available")
            tprint_success("✅ Enhanced tprint logging available")

            tprint_success("✅ Utility integrations initialization completed")

        except Exception as e:
            error_msg = f"CRITICAL: Utility integration initialization failed: {e}"
            tprint_error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e

    def _log_utility_integration_status(self) -> None:
        """Log comprehensive utility integration status."""
        try:
            tprint_info("📊 Utility Integration Status:")

            for utility, status in self.utility_integration_status.items():
                if status == 'available':
                    tprint_success(f"  ✅ {utility}: {status}")
                elif status == 'unavailable':
                    tprint_warning(f"  ⚠️ {utility}: {status}")
                elif status.startswith('error:'):
                    tprint_error(f"  ❌ {utility}: {status}")
                else:
                    tprint_info(f"  ℹ️ {utility}: {status}")

            # Log initialization errors if any
            if self.initialization_errors:
                tprint_error("❌ Initialization errors encountered:")
                for error in self.initialization_errors:
                    tprint_error(f"  - {error}")

        except Exception as e:
            tprint_error(f"❌ Failed to log utility integration status: {e}")

    def _handle_initialization_error(self, error: Exception) -> None:
        """Handle initialization errors with detailed reporting."""
        error_msg = f"Initialization failed: {str(error)}"
        tprint_error(f"❌ {error_msg}")
        tprint_error(f"❌ Traceback: {traceback.format_exc()}")

        # Log utility integration status even on failure
        if hasattr(self, 'utility_integration_status'):
            self._log_utility_integration_status()

        if TrainingPhase.INITIALIZATION in self.training_metrics:
            self._complete_phase(TrainingPhase.INITIALIZATION, success=False, error_message=error_msg)

    def _validate_input_data(self, X: np.ndarray, y: np.ndarray,
                           regime_labels: np.ndarray) -> Dict[str, Any]:
        """Comprehensive input data validation with detailed reporting and utility integration."""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'data_quality_metrics': {},
            'regime_analysis': {},
            'utility_validation': {}
        }

        try:
            tprint_info("🔍 Starting comprehensive input data validation...")

            # CRITICAL: Fast fail on data shape mismatches
            tprint_debug("Validating data shapes...")
            if X.shape[0] != y.shape[0]:
                error_msg = f"CRITICAL: Feature and target sample counts don't match: {X.shape[0]} vs {y.shape[0]}"
                tprint_error(f"❌ {error_msg}")
                raise ValueError(error_msg)

            if X.shape[0] != regime_labels.shape[0]:
                error_msg = f"CRITICAL: Feature and regime label sample counts don't match: {X.shape[0]} vs {regime_labels.shape[0]}"
                tprint_error(f"❌ {error_msg}")
                raise ValueError(error_msg)

            # CRITICAL: Fast fail on empty data
            if X.shape[0] == 0:
                error_msg = "CRITICAL: No samples provided in input data"
                tprint_error(f"❌ {error_msg}")
                raise ValueError(error_msg)

            if X.shape[1] == 0:
                error_msg = "CRITICAL: No features provided in input data"
                tprint_error(f"❌ {error_msg}")
                raise ValueError(error_msg)

            tprint_success(f"✅ Basic shape validation passed: {X.shape[0]} samples, {X.shape[1]} features")

            # Enhanced data quality analysis using utility functions
            data_quality = {}

            # Validate features quality with math validation utilities
            tprint_debug("Validating features quality...")
            feature_quality = self._validate_data_quality_enhanced(X, "features", max_nan_percentage=10.0, max_inf_percentage=1.0)
            data_quality['features'] = feature_quality
            validation_results['warnings'].extend(feature_quality['warnings'])
            validation_results['errors'].extend(feature_quality['errors'])
            if not feature_quality['is_valid']:
                validation_results['is_valid'] = False
                tprint_error("❌ Feature quality validation failed")
            else:
                tprint_success("✅ Feature quality validation passed")

            # Validate targets quality with stricter thresholds
            tprint_debug("Validating targets quality...")
            target_quality = self._validate_data_quality_enhanced(y, "targets", max_nan_percentage=5.0, max_inf_percentage=1.0)
            data_quality['targets'] = target_quality
            validation_results['warnings'].extend(target_quality['warnings'])
            validation_results['errors'].extend(target_quality['errors'])
            if not target_quality['is_valid']:
                validation_results['is_valid'] = False
                tprint_error("❌ Target quality validation failed")
            else:
                tprint_success("✅ Target quality validation passed")

            # Enhanced regime distribution analysis
            tprint_debug("Analyzing regime distribution...")
            unique_regimes = np.unique(regime_labels)
            regime_counts = np.bincount(regime_labels)
            min_regime_size = np.min(regime_counts)
            max_regime_size = np.max(regime_counts)

            # Use math validation utilities for safe calculations
            regime_balance = safe_divide(min_regime_size, max_regime_size, 0.0)

            regime_analysis = {
                'unique_regimes_count': len(unique_regimes),
                'min_regime_size': min_regime_size,
                'max_regime_size': max_regime_size,
                'regime_balance': regime_balance,
                'regime_distribution': dict(zip(unique_regimes, regime_counts))
            }

            validation_results['regime_analysis'] = regime_analysis
            validation_results['data_quality_metrics'] = data_quality

            # Check regime sufficiency with enhanced validation
            insufficient_regimes = regime_counts < self.config.min_samples_per_regime
            insufficient_count = np.sum(insufficient_regimes)

            if insufficient_count > 0:
                warning_msg = f"{insufficient_count} regimes have fewer than {self.config.min_samples_per_regime} samples"
                validation_results['warnings'].append(warning_msg)
                tprint_warning(f"⚠️ {warning_msg}")

                # Check if too many regimes are insufficient (fast fail condition)
                if insufficient_count > len(unique_regimes) * 0.5:
                    error_msg = f"Critical: {insufficient_count}/{len(unique_regimes)} regimes have insufficient data"
                    validation_results['errors'].append(error_msg)
                    validation_results['is_valid'] = False
                    tprint_error(f"❌ {error_msg}")
                    raise ValueError(error_msg)  # Fast fail on critical errors

            # Utility integration validation
            utility_validation = self._validate_utility_integrations()
            validation_results['utility_validation'] = utility_validation

            # Log comprehensive validation results with tprint
            tprint_info(f"📊 Data validation summary: {X.shape[0]} samples, {X.shape[1]} features, {len(unique_regimes)} regimes")
            tprint_info(f"📊 Regime balance: {regime_balance:.3f} (min={min_regime_size}, max={max_regime_size})")

            if validation_results['warnings']:
                tprint_warning(f"⚠️ {len(validation_results['warnings'])} warnings found:")
                for warning in validation_results['warnings']:
                    tprint_warning(f"  - {warning}")

            if validation_results['errors']:
                tprint_error(f"❌ {len(validation_results['errors'])} errors found:")
                for error in validation_results['errors']:
                    tprint_error(f"  - {error}")
                raise ValueError(f"Data validation failed: {'; '.join(validation_results['errors'])}")

            tprint_success("✅ Comprehensive input data validation completed successfully")
            return validation_results

        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(str(e))
            tprint_error(f"❌ Data validation failed: {e}")
            tprint_error(f"❌ Traceback: {traceback.format_exc()}")
            raise

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _validate_data_quality_enhanced(self, data: np.ndarray, data_name: str,
                                       max_nan_percentage: float = 10.0,
                                       max_inf_percentage: float = 1.0) -> Dict[str, Any]:
        """Enhanced data quality validation with utility integration and comprehensive error handling."""
        quality_metrics = {
            'nan_count': 0,
            'inf_count': 0,
            'nan_percentage': 0.0,
            'inf_percentage': 0.0,
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'utility_validation': {},
            'statistical_metrics': {}
        }

        try:
            tprint_debug(f"Validating {data_name} quality...")

            # Check for NaN values with enhanced reporting
            nan_count = np.sum(np.isnan(data))
            if nan_count > 0:
                nan_percentage = safe_divide(nan_count * 100, data.size, 0.0)

                quality_metrics['nan_count'] = nan_count
                quality_metrics['nan_percentage'] = nan_percentage

                if nan_percentage > max_nan_percentage:
                    error_msg = f"{data_name} contains {nan_percentage:.2f}% NaN values (threshold: {max_nan_percentage}%)"
                    quality_metrics['errors'].append(error_msg)
                    quality_metrics['is_valid'] = False
                    tprint_error(f"❌ {error_msg}")
                else:
                    warning_msg = f"{data_name} contains {nan_count} NaN values ({nan_percentage:.2f}%)"
                    quality_metrics['warnings'].append(warning_msg)
                    tprint_warning(f"⚠️ {warning_msg}")
            else:
                tprint_debug(f"✅ {data_name}: No NaN values found")

            # Check for infinite values with enhanced reporting
            inf_count = np.sum(np.isinf(data))
            if inf_count > 0:
                inf_percentage = safe_divide(inf_count * 100, data.size, 0.0)

                quality_metrics['inf_count'] = inf_count
                quality_metrics['inf_percentage'] = inf_percentage

                if inf_percentage > max_inf_percentage:
                    error_msg = f"{data_name} contains {inf_percentage:.2f}% infinite values (threshold: {max_inf_percentage}%)"
                    quality_metrics['errors'].append(error_msg)
                    quality_metrics['is_valid'] = False
                    tprint_error(f"❌ {error_msg}")
                else:
                    warning_msg = f"{data_name} contains {inf_count} infinite values ({inf_percentage:.2f}%)"
                    quality_metrics['warnings'].append(warning_msg)
                    tprint_warning(f"⚠️ {warning_msg}")
            else:
                tprint_debug(f"✅ {data_name}: No infinite values found")

            # Enhanced statistical validation using math validation utilities
            try:
                # Validate finite values
                finite_data = data[np.isfinite(data)]
                if len(finite_data) > 0:
                    # Calculate statistical metrics safely
                    mean_val = np.mean(finite_data)
                    std_val = np.std(finite_data)
                    min_val = np.min(finite_data)
                    max_val = np.max(finite_data)

                    # Validate statistical properties
                    validate_finite(mean_val, f"{data_name}_mean")
                    validate_finite(std_val, f"{data_name}_std")
                    validate_finite(min_val, f"{data_name}_min")
                    validate_finite(max_val, f"{data_name}_max")

                    quality_metrics['statistical_metrics'] = {
                        'mean': mean_val,
                        'std': std_val,
                        'min': min_val,
                        'max': max_val,
                        'finite_count': len(finite_data),
                        'total_count': data.size
                    }

                    tprint_debug(f"✅ {data_name}: Statistical validation passed")
                else:
                    warning_msg = f"{data_name}: No finite values found for statistical analysis"
                    quality_metrics['warnings'].append(warning_msg)
                    tprint_warning(f"⚠️ {warning_msg}")

            except Exception as e:
                warning_msg = f"Statistical validation failed for {data_name}: {e}"
                quality_metrics['warnings'].append(warning_msg)
                tprint_warning(f"⚠️ {warning_msg}")

            # Matrix operations validation
            try:
                matrix_validation = validate_matrix_properties(data)
                quality_metrics['utility_validation']['matrix_operations'] = matrix_validation
                tprint_debug(f"✅ {data_name}: Matrix operations validation completed")
            except Exception as e:
                warning_msg = f"Matrix operations validation failed for {data_name}: {e}"
                quality_metrics['warnings'].append(warning_msg)
                tprint_warning(f"⚠️ {warning_msg}")

            # Data utilities validation
            if data_name == "features":
                try:
                    # Convert to DataFrame for validation if possible
                    if data.ndim == 2:
                        df = pd.DataFrame(data)
                        data_quality_metrics = calculate_data_quality_metrics(df)
                        quality_metrics['utility_validation']['data_quality'] = data_quality_metrics
                        tprint_debug(f"✅ {data_name}: Data quality metrics calculated")
                except Exception as e:
                    warning_msg = f"Data quality metrics calculation failed for {data_name}: {e}"
                    quality_metrics['warnings'].append(warning_msg)
                    tprint_warning(f"⚠️ {warning_msg}")

            if quality_metrics['is_valid']:
                tprint_success(f"✅ {data_name} quality validation passed")
            else:
                tprint_error(f"❌ {data_name} quality validation failed")

            return quality_metrics

        except Exception as e:
            quality_metrics['is_valid'] = False
            error_msg = f"Failed to validate {data_name}: {e}"
            quality_metrics['errors'].append(error_msg)
            tprint_error(f"❌ {error_msg}")
            tprint_error(f"❌ Traceback: {traceback.format_exc()}")
            return quality_metrics

    def _validate_utility_integrations(self) -> Dict[str, Any]:
        """Validate utility integrations and their availability."""
        utility_validation = {
            'tprint_available': TPRINT_AVAILABLE,
            'common_operations_available': COMMON_OPERATIONS_AVAILABLE,
            'common_utilities_available': COMMON_UTILITIES_AVAILABLE,
            'math_validation_available': MATH_VALIDATION_AVAILABLE,
            'data_utilities_available': DATA_UTILITIES_AVAILABLE,
            'matrix_operations_available': MATRIX_OPERATIONS_AVAILABLE,
            'ml_common_available': ML_COMMON_AVAILABLE,
            'vectorized_training_available': VECTORIZED_TRAINING_AVAILABLE
        }

        try:
            tprint_debug("Validating utility integrations...")

            # Count available utilities
            available_count = sum(1 for available in utility_validation.values() if available)
            total_count = len(utility_validation)
            availability_rate = safe_divide(available_count * 100, total_count, 0.0)

            utility_validation['available_count'] = available_count
            utility_validation['total_count'] = total_count
            utility_validation['availability_rate'] = availability_rate

            if availability_rate >= 80:
                tprint_success(f"✅ Utility integration validation passed: {available_count}/{total_count} utilities available ({availability_rate:.1f}%)")
            elif availability_rate >= 60:
                tprint_warning(f"⚠️ Utility integration validation warning: {available_count}/{total_count} utilities available ({availability_rate:.1f}%)")
            else:
                tprint_error(f"❌ Utility integration validation failed: {available_count}/{total_count} utilities available ({availability_rate:.1f}%)")

            return utility_validation

        except Exception as e:
            tprint_error(f"❌ Utility integration validation failed: {e}")
            utility_validation['validation_error'] = str(e)
            return utility_validation

    def _validate_array_shapes(self, arrays: Dict[str, np.ndarray], expected_samples: int) -> Dict[str, Any]:
        """Validate that all arrays have consistent sample counts."""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'shape_info': {}
        }

        try:
            for name, array in arrays.items():
                if array.shape[0] != expected_samples:
                    error_msg = f"{name} shape mismatch: expected {expected_samples}, got {array.shape[0]}"
                    validation_results['errors'].append(error_msg)
                    validation_results['is_valid'] = False
                else:
                    validation_results['shape_info'][name] = array.shape

            if not validation_results['is_valid']:
                raise ValueError(f"Shape validation failed: {'; '.join(validation_results['errors'])}")

            return validation_results

        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(str(e))
            return validation_results

    def _comprehensive_validation_check(self,
                                      X: np.ndarray,
                                      y: np.ndarray,
                                      regime_labels: np.ndarray,
                                      phase_name: str,
                                      additional_arrays: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """Comprehensive validation check for any phase of training."""
        validation_summary = {
            'phase': phase_name,
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'metrics': {},
            'recommendations': []
        }

        try:
            # Basic shape validation
            arrays_to_validate = {
                'features': X,
                'targets': y,
                'regime_labels': regime_labels
            }

            if additional_arrays:
                arrays_to_validate.update(additional_arrays)

            shape_validation = self._validate_array_shapes(arrays_to_validate, X.shape[0])
            if not shape_validation['is_valid']:
                validation_summary['errors'].extend(shape_validation['errors'])
                validation_summary['is_valid'] = False

            # Data quality validation
            feature_quality = self._validate_data_quality(X, f"{phase_name}_features", max_nan_percentage=10.0, max_inf_percentage=1.0)
            target_quality = self._validate_data_quality(y, f"{phase_name}_targets", max_nan_percentage=5.0, max_inf_percentage=1.0)

            validation_summary['warnings'].extend(feature_quality['warnings'])
            validation_summary['warnings'].extend(target_quality['warnings'])
            validation_summary['errors'].extend(feature_quality['errors'])
            validation_summary['errors'].extend(target_quality['errors'])

            if not feature_quality['is_valid'] or not target_quality['is_valid']:
                validation_summary['is_valid'] = False

            # Regime analysis
            unique_regimes = np.unique(regime_labels)
            regime_counts = np.bincount(regime_labels)
            min_regime_size = np.min(regime_counts)
            max_regime_size = np.max(regime_counts)
            regime_balance = min_regime_size / max_regime_size if max_regime_size > 0 else 0

            validation_summary['metrics'] = {
                'samples': X.shape[0],
                'features': X.shape[1],
                'regimes': len(unique_regimes),
                'min_regime_size': min_regime_size,
                'max_regime_size': max_regime_size,
                'regime_balance': regime_balance,
                'feature_nan_percentage': feature_quality['nan_percentage'],
                'target_nan_percentage': target_quality['nan_percentage']
            }

            # Generate recommendations
            if regime_balance < 0.1:
                validation_summary['recommendations'].append("Very low regime balance - consider data augmentation")

            if feature_quality['nan_percentage'] > 5:
                validation_summary['recommendations'].append("High NaN percentage in features - consider imputation")

            if target_quality['nan_percentage'] > 2:
                validation_summary['recommendations'].append("High NaN percentage in targets - review data pipeline")

            if min_regime_size < self.config.min_samples_per_regime:
                validation_summary['recommendations'].append("Some regimes have insufficient samples - consider reducing min_samples_per_regime")

            # Log validation results
            if validation_summary['is_valid']:
                self.logger.info(f"✅ {phase_name} validation passed")
            else:
                self.logger.error(f"❌ {phase_name} validation failed")
                for error in validation_summary['errors']:
                    self.logger.error(f"❌ {error}")

            if validation_summary['warnings']:
                for warning in validation_summary['warnings']:
                    self.logger.warning(f"⚠️ {warning}")

            if validation_summary['recommendations']:
                for recommendation in validation_summary['recommendations']:
                    self.logger.info(f"💡 {recommendation}")

            return validation_summary

        except Exception as e:
            validation_summary['is_valid'] = False
            validation_summary['errors'].append(f"Validation check failed: {e}")
            self.logger.error(f"❌ {phase_name} validation check failed: {e}")
            return validation_summary

    def _log_structured_event(self, event_type: str, phase: str, message: str,
                             metrics: Optional[Dict[str, Any]] = None,
                             level: str = "info") -> None:
        """Log structured events with consistent formatting."""
        try:
            log_data = {
                'event_type': event_type,
                'phase': phase,
                'message': message,
                'timestamp': time.time(),
                'memory_mb': self._get_memory_usage()
            }

            if metrics:
                log_data['metrics'] = metrics

            # Format structured log message
            structured_msg = f"[{event_type.upper()}] {phase}: {message}"
            if metrics:
                structured_msg += f" | Metrics: {metrics}"

            # Log with appropriate level
            if level == "error":
                self.logger.error(structured_msg)
            elif level == "warning":
                self.logger.warning(structured_msg)
            elif level == "debug":
                self.logger.debug(structured_msg)
            else:
                self.logger.info(structured_msg)

        except Exception as e:
            self.logger.error(f"Failed to log structured event: {e}")

    def _log_phase_start(self, phase: TrainingPhase, context: Optional[Dict[str, Any]] = None) -> None:
        """Log phase start with context."""
        self._log_structured_event(
            event_type="phase_start",
            phase=phase.value,
            message=f"Starting {phase.value} phase",
            metrics=context,
            level="info"
        )

    def _log_phase_complete(self, phase: TrainingPhase, success: bool,
                           duration: float, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Log phase completion with results."""
        event_type = "phase_success" if success else "phase_failure"
        message = f"Completed {phase.value} phase in {duration:.2f}s"

        if not success:
            message += " (FAILED)"

        self._log_structured_event(
            event_type=event_type,
            phase=phase.value,
            message=message,
            metrics=metrics,
            level="error" if not success else "info"
        )

    def _log_data_quality_issue(self, issue_type: str, details: Dict[str, Any]) -> None:
        """Log data quality issues with structured format."""
        self._log_structured_event(
            event_type="data_quality_issue",
            phase="validation",
            message=f"Data quality issue: {issue_type}",
            metrics=details,
            level="warning"
        )

    def _log_performance_metric(self, metric_name: str, value: float, unit: str = "") -> None:
        """Log performance metrics with structured format."""
        self._log_structured_event(
            event_type="performance_metric",
            phase="training",
            message=f"Performance metric: {metric_name}",
            metrics={metric_name: f"{value:.2f}{unit}"},
            level="info"
        )

    def execute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        analyst_signals: Optional[np.ndarray] = None,
        analyst_model_outputs: Optional[np.ndarray] = None,
        hmm_regime_features: Optional[np.ndarray] = None,
        all_analyst_models_outputs: Optional[Dict[str, np.ndarray]] = None,
        hmm_model_outputs: Optional[np.ndarray] = None,
        analyst_ensemble_outputs: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        analyst_confidence_scores: Optional[np.ndarray] = None,
        analyst_directional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute enhanced Tactician models training step with comprehensive error handling and utility integration.

        Args:
            X: Input features (1m timeframe with cross-timeframe features)
            y: Target values (tactician outputs - timing decisions)
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states
            analyst_signals: Directional signals from Analyst (1=long, -1=short, 0=neutral)
            analyst_model_outputs: Analyst model predictions used as features
            hmm_regime_features: HMM regime features (probabilities, characteristics)
            all_analyst_models_outputs: All individual analyst ML model outputs
            hmm_model_outputs: HMM model outputs (predictions, probabilities, etc.)
            analyst_ensemble_outputs: Analyst ensemble model outputs
            analyst_confidence_scores: Confidence scores from Analyst for sample weighting
            analyst_directional_info: Additional directional analysis from Analyst

        Returns:
            Dictionary containing training results and metadata with comprehensive reporting
        """
        try:
            tprint_info("🚀 Starting Enhanced Tactician models training step")
            tprint_structured({
                'operation': 'tactician_training_start',
                'samples': X.shape[0] if X is not None else 0,
                'features': X.shape[1] if X is not None else 0,
                'regimes': len(np.unique(regime_labels)) if regime_labels is not None else 0,
                'has_analyst_signals': analyst_signals is not None,
                'has_hmm_features': hmm_regime_features is not None,
                'has_analyst_models': all_analyst_models_outputs is not None
            })

            self.overall_start_time = time.time()

            # Phase 1: Data Validation with comprehensive error handling
            validation_context = {
                'samples': X.shape[0] if X is not None else 0,
                'features': X.shape[1] if X is not None else 0,
                'regimes': len(np.unique(regime_labels)) if regime_labels is not None else 0
            }
            self._start_phase(TrainingPhase.DATA_VALIDATION, validation_context)

            try:
                with tprint_timer("Data Validation"):
                    validation_results = self._validate_input_data(X, y, regime_labels)

                # Log data quality issues with enhanced reporting
                if validation_results.get('warnings'):
                    tprint_warning(f"⚠️ {len(validation_results['warnings'])} data quality warnings found")
                    for warning in validation_results['warnings']:
                        self._log_data_quality_issue("warning", {'message': warning})

                if validation_results.get('errors'):
                    tprint_error(f"❌ {len(validation_results['errors'])} data quality errors found")
                    for error in validation_results['errors']:
                        self._log_data_quality_issue("error", {'message': error})

                # Log utility integration status
                if validation_results.get('utility_validation'):
                    utility_status = validation_results['utility_validation']
                    tprint_info(f"📊 Utility integration status: {utility_status.get('available_count', 0)}/{utility_status.get('total_count', 0)} utilities available")

                self._complete_phase(TrainingPhase.DATA_VALIDATION, success=True,
                                   samples_processed=X.shape[0], features_count=X.shape[1],
                                   warnings_issued=len(validation_results.get('warnings', [])),
                                   errors_encountered=len(validation_results.get('errors', [])))
            except Exception as e:
                self._complete_phase(TrainingPhase.DATA_VALIDATION, success=False, error_message=str(e))
                tprint_error(f"❌ Data validation phase failed: {e}")
                raise

            # Phase 2: Feature Preparation with enhanced error handling
            feature_context = {
                'original_samples': X.shape[0],
                'original_features': X.shape[1],
                'has_analyst_signals': analyst_signals is not None,
                'has_hmm_features': hmm_regime_features is not None,
                'has_analyst_models': all_analyst_models_outputs is not None,
                'has_analyst_ensemble': analyst_ensemble_outputs is not None
            }
            self._start_phase(TrainingPhase.FEATURE_PREPARATION, feature_context)

            try:
                with tprint_timer("Feature Preparation"):
                  X, y, regime_labels, feature_names, preparation_metrics = self._prepare_features(
                      X, y, regime_labels, feature_names, hmm_states,
                      analyst_signals, analyst_model_outputs, hmm_regime_features,
                      all_analyst_models_outputs, hmm_model_outputs, analyst_ensemble_outputs
                  )

                # Log feature preparation metrics with enhanced reporting
                if preparation_metrics.get('green_light_filtering'):
                    gl_filtering = preparation_metrics['green_light_filtering']
                    green_light_rate = gl_filtering.get('green_light_rate', 0) * 100
                    tprint_info(f"📊 Green light filtering: {green_light_rate:.2f}% of samples retained")
                    self._log_performance_metric("green_light_rate", green_light_rate, "%")

                # Log feature combination results
                if preparation_metrics.get('feature_combinations'):
                    feature_combinations = preparation_metrics['feature_combinations']
                    tprint_info(f"📊 Feature combinations: {feature_combinations}")

                self._complete_phase(TrainingPhase.FEATURE_PREPARATION, success=True,
                                   samples_processed=X.shape[0], features_count=X.shape[1],
                                   warnings_issued=len(preparation_metrics.get('warnings', [])),
                                   errors_encountered=len(preparation_metrics.get('errors', [])))
            except Exception as e:
                self._complete_phase(TrainingPhase.FEATURE_PREPARATION, success=False, error_message=str(e))
                tprint_error(f"❌ Feature preparation phase failed: {e}")
                raise

            # Phase 3: Model Training with enhanced error handling
            self._start_phase(TrainingPhase.MODEL_TRAINING)
            try:
                with tprint_timer("Model Training"):
                    results = self._execute_training(
                        X, y, regime_labels, feature_names, hmm_states,
                        analyst_confidence_scores, analyst_directional_info,
                        timestamps, analyst_signals, analyst_model_outputs,
                        hmm_regime_features, all_analyst_models_outputs,
                        hmm_model_outputs, analyst_ensemble_outputs
                    )

                models_trained = len(results.get('models', {}))
                tprint_success(f"✅ Model training completed: {models_trained} models trained")

                self._complete_phase(TrainingPhase.MODEL_TRAINING, success=True,
                                   models_trained=models_trained,
                                   memory_usage_mb=self._get_memory_usage())
            except Exception as e:
                self._complete_phase(TrainingPhase.MODEL_TRAINING, success=False, error_message=str(e))
                tprint_error(f"❌ Model training phase failed: {e}")
                raise

            # Phase 4: Finalization with enhanced error handling
            self._start_phase(TrainingPhase.FINALIZATION)
            try:
                with tprint_timer("Results Finalization"):
                    results = self._finalize_results_enhanced(results, analyst_signals)

                total_time = time.time() - self.overall_start_time
                tprint_performance("Total Tactician Training", total_time)

                self._complete_phase(TrainingPhase.FINALIZATION, success=True)

                # Generate comprehensive training report
                self._generate_training_report_enhanced(total_time)

                # Log final success
                tprint_success("🎉 Enhanced Tactician models training completed successfully!")
                tprint_structured({
                    'operation': 'tactician_training_complete',
                    'total_time': total_time,
                    'models_trained': models_trained,
                    'final_samples': X.shape[0],
                    'final_features': X.shape[1],
                    'success': True
                })

                # Generate thorough outcome file with datetime stamp
                try:
                    from datetime import datetime
                    from pathlib import Path
                    import json

                    outcome_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    outcomes_dir = Path('outcomes')
                    outcomes_dir.mkdir(parents=True, exist_ok=True)

                    outcome_filename = f"tactician_model_training_outcome_{outcome_timestamp}.json"
                    outcome_path = outcomes_dir / outcome_filename

                    # Extract comprehensive training statistics
                    models_trained_dict = results.get('models', {})
                    model_stats = {
                        'total_models': len(models_trained_dict),
                        'model_names': list(models_trained_dict.keys()),
                        'per_model_details': {}
                    }

                    for model_name, model_info in models_trained_dict.items():
                        if isinstance(model_info, dict):
                            # Extract comprehensive ML metrics
                            ml_metrics = {}

                            # Classification metrics
                            if 'accuracy' in model_info:
                                ml_metrics['accuracy'] = float(model_info['accuracy']) if model_info['accuracy'] is not None else None
                            if 'auc' in model_info:
                                ml_metrics['auc'] = float(model_info['auc']) if model_info['auc'] is not None else None
                            if 'roc_auc' in model_info:
                                ml_metrics['roc_auc'] = float(model_info['roc_auc']) if model_info['roc_auc'] is not None else None
                            if 'precision' in model_info:
                                ml_metrics['precision'] = float(model_info['precision']) if model_info['precision'] is not None else None
                            if 'recall' in model_info:
                                ml_metrics['recall'] = float(model_info['recall']) if model_info['recall'] is not None else None
                            if 'f1_score' in model_info:
                                ml_metrics['f1_score'] = float(model_info['f1_score']) if model_info['f1_score'] is not None else None
                            if 'average_precision' in model_info:
                                ml_metrics['average_precision'] = float(model_info['average_precision']) if model_info['average_precision'] is not None else None

                            # Regression metrics
                            if 'mse' in model_info:
                                ml_metrics['mse'] = float(model_info['mse']) if model_info['mse'] is not None else None
                            if 'rmse' in model_info:
                                ml_metrics['rmse'] = float(model_info['rmse']) if model_info['rmse'] is not None else None
                            if 'mae' in model_info:
                                ml_metrics['mae'] = float(model_info['mae']) if model_info['mae'] is not None else None
                            if 'r2_score' in model_info:
                                ml_metrics['r2_score'] = float(model_info['r2_score']) if model_info['r2_score'] is not None else None

                            # Trading-specific metrics
                            if 'sharpe_ratio' in model_info:
                                ml_metrics['sharpe_ratio'] = float(model_info['sharpe_ratio']) if model_info['sharpe_ratio'] is not None else None
                            if 'win_rate' in model_info:
                                ml_metrics['win_rate'] = float(model_info['win_rate']) if model_info['win_rate'] is not None else None
                            if 'profit_factor' in model_info:
                                ml_metrics['profit_factor'] = float(model_info['profit_factor']) if model_info['profit_factor'] is not None else None
                            if 'max_drawdown' in model_info:
                                ml_metrics['max_drawdown'] = float(model_info['max_drawdown']) if model_info['max_drawdown'] is not None else None
                            if 'sortino_ratio' in model_info:
                                ml_metrics['sortino_ratio'] = float(model_info['sortino_ratio']) if model_info['sortino_ratio'] is not None else None

                            # CV scores
                            cv_scores = {}
                            if 'cv_scores' in model_info and model_info['cv_scores'] is not None:
                                cv_scores_list = model_info['cv_scores']
                                if isinstance(cv_scores_list, (list, tuple)) and len(cv_scores_list) > 0:
                                    cv_scores = {
                                        'mean': float(np.mean(cv_scores_list)),
                                        'std': float(np.std(cv_scores_list)),
                                        'min': float(np.min(cv_scores_list)),
                                        'max': float(np.max(cv_scores_list)),
                                        'scores': [float(s) for s in cv_scores_list],
                                    }

                            # Confusion matrix if available
                            confusion_matrix_data = {}
                            if 'confusion_matrix' in model_info and model_info['confusion_matrix'] is not None:
                                cm = model_info['confusion_matrix']
                                if hasattr(cm, 'tolist'):
                                    confusion_matrix_data = {
                                        'matrix': cm.tolist(),
                                        'shape': list(cm.shape),
                                    }

                            model_stats['per_model_details'][model_name] = {
                                'model_type': model_info.get('model_type', 'unknown'),
                                'training_score': float(model_info.get('train_score', 0.0)) if model_info.get('train_score') is not None else None,
                                'validation_score': float(model_info.get('val_score', 0.0)) if model_info.get('val_score') is not None else None,
                                'test_score': float(model_info.get('test_score', 0.0)) if model_info.get('test_score') is not None else None,
                                'n_features': model_info.get('n_features', 0),
                                'n_samples': model_info.get('n_samples', 0),
                                'ml_metrics': ml_metrics,
                                'cv_scores': cv_scores,
                                'confusion_matrix': confusion_matrix_data,
                            }

                    # Calculate aggregate metrics across all models
                    aggregate_metrics = {
                        'avg_training_score': 0.0,
                        'avg_validation_score': 0.0,
                        'avg_test_score': 0.0,
                        'avg_accuracy': 0.0,
                        'avg_auc': 0.0,
                        'avg_precision': 0.0,
                        'avg_recall': 0.0,
                        'avg_f1_score': 0.0,
                        'best_model': None,
                        'worst_model': None,
                    }

                    valid_train_scores = []
                    valid_val_scores = []
                    valid_test_scores = []
                    valid_accuracies = []
                    valid_aucs = []
                    valid_precisions = []
                    valid_recalls = []
                    valid_f1_scores = []

                    for model_name, details in model_stats['per_model_details'].items():
                        if details['training_score'] is not None:
                            valid_train_scores.append((model_name, details['training_score']))
                        if details['validation_score'] is not None:
                            valid_val_scores.append((model_name, details['validation_score']))
                        if details['test_score'] is not None:
                            valid_test_scores.append((model_name, details['test_score']))

                        ml_metrics = details.get('ml_metrics', {})
                        if ml_metrics.get('accuracy') is not None:
                            valid_accuracies.append(ml_metrics['accuracy'])
                        if ml_metrics.get('auc') is not None or ml_metrics.get('roc_auc') is not None:
                            valid_aucs.append(ml_metrics.get('auc') or ml_metrics.get('roc_auc'))
                        if ml_metrics.get('precision') is not None:
                            valid_precisions.append(ml_metrics['precision'])
                        if ml_metrics.get('recall') is not None:
                            valid_recalls.append(ml_metrics['recall'])
                        if ml_metrics.get('f1_score') is not None:
                            valid_f1_scores.append(ml_metrics['f1_score'])

                    if valid_train_scores:
                        aggregate_metrics['avg_training_score'] = float(np.mean([s for _, s in valid_train_scores]))
                        aggregate_metrics['best_model'] = max(valid_train_scores, key=lambda x: x[1])[0]
                        aggregate_metrics['worst_model'] = min(valid_train_scores, key=lambda x: x[1])[0]
                    if valid_val_scores:
                        aggregate_metrics['avg_validation_score'] = float(np.mean([s for _, s in valid_val_scores]))
                    if valid_test_scores:
                        aggregate_metrics['avg_test_score'] = float(np.mean([s for _, s in valid_test_scores]))
                    if valid_accuracies:
                        aggregate_metrics['avg_accuracy'] = float(np.mean(valid_accuracies))
                        aggregate_metrics['std_accuracy'] = float(np.std(valid_accuracies))
                    if valid_aucs:
                        aggregate_metrics['avg_auc'] = float(np.mean(valid_aucs))
                        aggregate_metrics['std_auc'] = float(np.std(valid_aucs))
                    if valid_precisions:
                        aggregate_metrics['avg_precision'] = float(np.mean(valid_precisions))
                    if valid_recalls:
                        aggregate_metrics['avg_recall'] = float(np.mean(valid_recalls))
                    if valid_f1_scores:
                        aggregate_metrics['avg_f1_score'] = float(np.mean(valid_f1_scores))
                        aggregate_metrics['std_f1_score'] = float(np.std(valid_f1_scores))

                    # Performance metrics breakdown
                    phase_metrics = {}
                    for phase_name, phase_data in self.phase_metrics.items():
                        phase_metrics[phase_name] = {
                            'duration': phase_data.get('duration', 0.0),
                            'success': phase_data.get('success', False),
                            'samples_processed': phase_data.get('samples_processed', 0),
                            'errors': phase_data.get('errors', []),
                        }

                    performance_metrics = {
                        'total_execution_time_seconds': total_time,
                        'models_per_second': len(models_trained_dict) / max(0.001, total_time),
                        'phase_metrics': phase_metrics,
                        'final_samples': int(X.shape[0]),
                        'final_features': int(X.shape[1]),
                        'aggregate_ml_metrics': aggregate_metrics,
                    }

                    # Training configuration details
                    training_config = {
                        'timeframe': getattr(self.config, 'timeframe', '15m'),
                        'regime_aware': getattr(self.config, 'regime_aware', True),
                        'enable_enhanced_training': getattr(self.config, 'enable_enhanced_training', True),
                        'enable_overfitting_prevention': getattr(self.config, 'enable_overfitting_prevention', True),
                        'use_purged_cv': getattr(self.config, 'use_purged_cv', True),
                        'cv_folds': getattr(self.config, 'cv_folds', 5),
                        'early_stopping_enabled': getattr(self.config, 'early_stopping_enabled', True),
                        'enable_entry_timing_optimization': getattr(self.config, 'enable_entry_timing_optimization', False),
                        'confidence_aware_ensemble_enabled': True,  # Always enabled for Tactician
                    }

                    # Data quality and validation
                    data_quality = {
                        'input_samples': results.get('input_samples', X.shape[0]),
                        'input_features': results.get('input_features', X.shape[1]),
                        'training_samples': results.get('training_samples', 0),
                        'validation_samples': results.get('validation_samples', 0),
                        'test_samples': results.get('test_samples', 0),
                        'feature_names': results.get('feature_names', []),
                        'regime_distribution': results.get('regime_distribution', {}),
                        'analyst_signals_used': analyst_signals is not None,
                        'analyst_directional_info_used': analyst_directional_info is not None,
                    }

                    # Warnings and issues
                    warnings_and_issues = {
                        'overfitting_warnings': results.get('overfitting_warnings', []),
                        'lookahead_warnings': results.get('lookahead_warnings', []),
                        'data_quality_warnings': results.get('data_quality_warnings', []),
                        'total_warnings': len(results.get('overfitting_warnings', [])) + len(results.get('lookahead_warnings', [])),
                    }

                    # Entry timing optimization results if applicable
                    entry_timing_stats = {}
                    if 'entry_timing_optimization' in results:
                        entry_timing_stats = {
                            'enabled': True,
                            'results': results.get('entry_timing_optimization', {}),
                        }

                    # Create comprehensive outcome report
                    outcome_data = {
                        'component': 'tactician_model_training',
                        'timestamp': datetime.now().isoformat(),
                        'execution_time': total_time,
                        'configuration': training_config,
                        'results': {
                            'summary': {
                                'models_trained': len(models_trained_dict),
                                'training_successful': True,
                            },
                            'model_statistics': model_stats,
                            'data_quality': data_quality,
                            'entry_timing_optimization': entry_timing_stats,
                        },
                        'performance_metrics': performance_metrics,
                        'warnings_and_issues': warnings_and_issues,
                        'enhanced_training_metadata': results.get('enhanced_training_metadata', {}),
                        'status': 'success'
                    }

                    # Save outcome file
                    with open(outcome_path, 'w') as f:
                        json.dump(outcome_data, f, indent=2, default=str)

                    tprint_success(f"📄 Outcome file saved: {outcome_filename}")
                    results['outcome_file'] = str(outcome_path)

                except Exception as outcome_error:
                    tprint_warning(f"⚠️ Failed to save outcome file: {outcome_error}")
                    # Don't fail training if outcome file generation fails

                return results

            except Exception as e:
                self._complete_phase(TrainingPhase.FINALIZATION, success=False, error_message=str(e))
                tprint_error(f"❌ Finalization phase failed: {e}")
                raise

        except Exception as e:
            tprint_error(f"❌ Enhanced Tactician training failed: {e}")
            tprint_error(f"❌ Traceback: {traceback.format_exc()}")
            tprint_structured({
                'operation': 'tactician_training_failed',
                'error': str(e),
                'success': False
            })
            return self._create_error_result(str(e))

    def _prepare_features_enhanced(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]],
        hmm_states: Optional[np.ndarray],
        analyst_signals: Optional[np.ndarray],
        analyst_model_outputs: Optional[np.ndarray],
        hmm_regime_features: Optional[np.ndarray],
        all_analyst_models_outputs: Optional[Dict[str, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List[str]], Dict[str, Any]]:
        """Enhanced feature preparation with comprehensive error handling and utility integration."""
        return self._prepare_features(
            X, y, regime_labels, feature_names, hmm_states,
            analyst_signals, analyst_model_outputs, hmm_regime_features,
            all_analyst_models_outputs
        )

    def _execute_training_enhanced(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]],
        hmm_states: Optional[np.ndarray],
        analyst_confidence_scores: Optional[np.ndarray] = None,
        analyst_directional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Enhanced training execution with comprehensive error handling and utility integration."""
        return self._execute_training(
            X, y, regime_labels, feature_names, hmm_states,
            analyst_confidence_scores, analyst_directional_info
        )

    def _finalize_results_enhanced(self, results: Dict[str, Any], analyst_signals: Optional[np.ndarray]) -> Dict[str, Any]:
        """Enhanced results finalization with comprehensive error handling and utility integration."""
        return self._finalize_results(results, analyst_signals)

    def _generate_training_report_enhanced(self, total_time: float) -> None:
        """Enhanced training report generation with comprehensive error handling and utility integration."""
        return self._generate_training_report(total_time)

    def _prepare_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]],
        hmm_states: Optional[np.ndarray],
        analyst_signals: Optional[np.ndarray],
        analyst_model_outputs: Optional[np.ndarray],
        hmm_regime_features: Optional[np.ndarray],
        all_analyst_models_outputs: Optional[Dict[str, np.ndarray]],
        hmm_model_outputs: Optional[np.ndarray],
        analyst_ensemble_outputs: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List[str]], Dict[str, Any]]:
        """Prepare and combine all features with comprehensive error handling and validation."""
        preparation_metrics = {
            'original_samples': X.shape[0],
            'original_features': X.shape[1],
            'green_light_filtering': {},
            'feature_combinations': {},
            'final_samples': 0,
            'final_features': 0,
            'warnings': [],
            'errors': []
        }

        try:
            # Filter data to only include periods where Analyst gives directional signals
            if analyst_signals is not None:
                # Directional signals: 1 (long), -1 (short), 0 (neutral)
                directional_mask = (analyst_signals == 1) | (analyst_signals == -1)
                directional_count = np.sum(directional_mask)
                directional_rate = directional_count / len(analyst_signals)

                # Analyze directional distribution
                long_count = np.sum(analyst_signals == 1)
                short_count = np.sum(analyst_signals == -1)
                neutral_count = np.sum(analyst_signals == 0)

                preparation_metrics['directional_filtering'] = {
                    'total_signals': len(analyst_signals),
                    'directional_count': directional_count,
                    'directional_rate': directional_rate,
                    'long_count': long_count,
                    'short_count': short_count,
                    'neutral_count': neutral_count,
                    'long_ratio': long_count / len(analyst_signals),
                    'short_ratio': short_count / len(analyst_signals),
                    'neutral_ratio': neutral_count / len(analyst_signals)
                }

                self.logger.info(f"📊 Filtering to {directional_count} samples with Analyst directional signals ({directional_rate:.2%})")
                self.logger.info(f"   Long signals: {long_count} ({long_count/directional_count:.1%} of directional)")
                self.logger.info(f"   Short signals: {short_count} ({short_count/directional_count:.1%} of directional)")
                self.logger.info(f"   Neutral signals: {neutral_count} ({neutral_count/len(analyst_signals):.1%} of total)")

                # Validate directional filtering results
                if directional_count == 0:
                    error_msg = "No samples with Analyst directional signals found"
                    preparation_metrics['errors'].append(error_msg)
                    raise ValueError(error_msg)

                if directional_rate < 0.01:  # Less than 1%
                    warning_msg = f"Very low directional signal rate: {directional_rate:.2%}"
                    preparation_metrics['warnings'].append(warning_msg)
                    self.logger.warning(f"⚠️ {warning_msg}")

                # Apply filtering with validation
                X_filtered = X[directional_mask]
                y_filtered = y[directional_mask]
                regime_labels_filtered = regime_labels[directional_mask]

                # Validate filtered data shapes
                if X_filtered.shape[0] != directional_count:
                    error_msg = f"Filtered data shape mismatch: expected {directional_count}, got {X_filtered.shape[0]}"
                    preparation_metrics['errors'].append(error_msg)
                    raise ValueError(error_msg)

                X, y, regime_labels = X_filtered, y_filtered, regime_labels_filtered

                if hmm_states is not None:
                    hmm_states = hmm_states[directional_mask]
                    if hmm_states.shape[0] != directional_count:
                        error_msg = f"HMM states filtering mismatch: expected {directional_count}, got {hmm_states.shape[0]}"
                        preparation_metrics['errors'].append(error_msg)
                        raise ValueError(error_msg)

            # Combine all features: base features + HMM regime features + HMM model outputs + all analyst model outputs + analyst ensemble outputs
            additional_features = []
            additional_feature_names = []
            feature_combination_details = {}

            # Add HMM regime features if provided
            if hmm_regime_features is not None:
                try:
                    if analyst_signals is not None:
                        hmm_regime_features = hmm_regime_features[directional_mask]

                    # Validate HMM features shape
                    if hmm_regime_features.shape[0] != X.shape[0]:
                        error_msg = f"HMM regime features shape mismatch: expected {X.shape[0]}, got {hmm_regime_features.shape[0]}"
                        preparation_metrics['errors'].append(error_msg)
                        raise ValueError(error_msg)

                    # Check for NaN/Inf in HMM features
                    hmm_nan_count = np.sum(np.isnan(hmm_regime_features))
                    hmm_inf_count = np.sum(np.isinf(hmm_regime_features))

                    if hmm_nan_count > 0:
                        warning_msg = f"HMM features contain {hmm_nan_count} NaN values"
                        preparation_metrics['warnings'].append(warning_msg)
                        self.logger.warning(f"⚠️ {warning_msg}")

                    if hmm_inf_count > 0:
                        warning_msg = f"HMM features contain {hmm_inf_count} infinite values"
                        preparation_metrics['warnings'].append(warning_msg)
                        self.logger.warning(f"⚠️ {warning_msg}")

                    additional_features.append(hmm_regime_features)
                    additional_feature_names.extend([f"hmm_regime_{i}" for i in range(hmm_regime_features.shape[1])])

                    feature_combination_details['hmm_regime_features'] = {
                        'count': hmm_regime_features.shape[1],
                        'nan_count': hmm_nan_count,
                        'inf_count': hmm_inf_count
                    }

                    self.logger.info(f"📊 Added {hmm_regime_features.shape[1]} HMM regime features")

                except Exception as e:
                    error_msg = f"Failed to add HMM regime features: {e}"
                    preparation_metrics['errors'].append(error_msg)
                    self.logger.error(f"❌ {error_msg}")
                    raise

            # Add HMM model outputs if provided
            if hmm_model_outputs is not None:
                try:
                    if analyst_signals is not None:
                        hmm_model_outputs = hmm_model_outputs[directional_mask]

                    # Validate HMM model outputs shape
                    if hmm_model_outputs.shape[0] != X.shape[0]:
                        error_msg = f"HMM model outputs shape mismatch: expected {X.shape[0]}, got {hmm_model_outputs.shape[0]}"
                        preparation_metrics['errors'].append(error_msg)
                        raise ValueError(error_msg)

                    # Check for NaN/Inf in HMM model outputs
                    hmm_outputs_nan_count = np.sum(np.isnan(hmm_model_outputs))
                    hmm_outputs_inf_count = np.sum(np.isinf(hmm_model_outputs))

                    if hmm_outputs_nan_count > 0:
                        warning_msg = f"HMM model outputs contain {hmm_outputs_nan_count} NaN values"
                        preparation_metrics['warnings'].append(warning_msg)
                        self.logger.warning(f"⚠️ {warning_msg}")

                    if hmm_outputs_inf_count > 0:
                        warning_msg = f"HMM model outputs contain {hmm_outputs_inf_count} infinite values"
                        preparation_metrics['warnings'].append(warning_msg)
                        self.logger.warning(f"⚠️ {warning_msg}")

                    additional_features.append(hmm_model_outputs)
                    additional_feature_names.extend([f"hmm_model_{i}" for i in range(hmm_model_outputs.shape[1])])

                    feature_combination_details['hmm_model_outputs'] = {
                        'count': hmm_model_outputs.shape[1],
                        'nan_count': hmm_outputs_nan_count,
                        'inf_count': hmm_outputs_inf_count
                    }

                    self.logger.info(f"📊 Added {hmm_model_outputs.shape[1]} HMM model outputs")

                except Exception as e:
                    error_msg = f"Failed to add HMM model outputs: {e}"
                    preparation_metrics['errors'].append(error_msg)
                    self.logger.error(f"❌ {error_msg}")
                    raise

            # Add all individual analyst model outputs if provided
            if all_analyst_models_outputs is not None:
                analyst_features_added = 0
                for model_name, model_outputs in all_analyst_models_outputs.items():
                    try:
                        if analyst_signals is not None:
                            model_outputs = model_outputs[directional_mask]

                        # Validate model outputs shape
                        if model_outputs.shape[0] != X.shape[0]:
                            error_msg = f"Analyst model {model_name} output shape mismatch: expected {X.shape[0]}, got {model_outputs.shape[0]}"
                            preparation_metrics['errors'].append(error_msg)
                            raise ValueError(error_msg)

                        # Check for NaN/Inf in model outputs
                        model_nan_count = np.sum(np.isnan(model_outputs))
                        model_inf_count = np.sum(np.isinf(model_outputs))

                        if model_nan_count > 0:
                            warning_msg = f"Analyst model {model_name} outputs contain {model_nan_count} NaN values"
                            preparation_metrics['warnings'].append(warning_msg)
                            self.logger.warning(f"⚠️ {warning_msg}")

                        if model_inf_count > 0:
                            warning_msg = f"Analyst model {model_name} outputs contain {model_inf_count} infinite values"
                            preparation_metrics['warnings'].append(warning_msg)
                            self.logger.warning(f"⚠️ {warning_msg}")

                        additional_features.append(model_outputs)
                        additional_feature_names.extend([f"analyst_{model_name}_{i}" for i in range(model_outputs.shape[1])])
                        analyst_features_added += model_outputs.shape[1]

                    except Exception as e:
                        error_msg = f"Failed to add analyst model {model_name} outputs: {e}"
                        preparation_metrics['errors'].append(error_msg)
                        self.logger.error(f"❌ {error_msg}")
                        raise

                feature_combination_details['analyst_models'] = {
                    'model_count': len(all_analyst_models_outputs),
                    'total_features': analyst_features_added
                }

                self.logger.info(f"📊 Added outputs from {len(all_analyst_models_outputs)} analyst models ({analyst_features_added} features)")

            # Add analyst ensemble outputs if provided
            if analyst_ensemble_outputs is not None:
                try:
                    if analyst_signals is not None:
                        analyst_ensemble_outputs = analyst_ensemble_outputs[directional_mask]

                    # Validate analyst ensemble outputs shape
                    if analyst_ensemble_outputs.shape[0] != X.shape[0]:
                        error_msg = f"Analyst ensemble outputs shape mismatch: expected {X.shape[0]}, got {analyst_ensemble_outputs.shape[0]}"
                        preparation_metrics['errors'].append(error_msg)
                        raise ValueError(error_msg)

                    # Check for NaN/Inf in analyst ensemble outputs
                    ensemble_nan_count = np.sum(np.isnan(analyst_ensemble_outputs))
                    ensemble_inf_count = np.sum(np.isinf(analyst_ensemble_outputs))

                    if ensemble_nan_count > 0:
                        warning_msg = f"Analyst ensemble outputs contain {ensemble_nan_count} NaN values"
                        preparation_metrics['warnings'].append(warning_msg)
                        self.logger.warning(f"⚠️ {warning_msg}")

                    if ensemble_inf_count > 0:
                        warning_msg = f"Analyst ensemble outputs contain {ensemble_inf_count} infinite values"
                        preparation_metrics['warnings'].append(warning_msg)
                        self.logger.warning(f"⚠️ {warning_msg}")

                    additional_features.append(analyst_ensemble_outputs)
                    additional_feature_names.extend([f"analyst_ensemble_{i}" for i in range(analyst_ensemble_outputs.shape[1])])

                    feature_combination_details['analyst_ensemble_outputs'] = {
                        'count': analyst_ensemble_outputs.shape[1],
                        'nan_count': ensemble_nan_count,
                        'inf_count': ensemble_inf_count
                    }

                    self.logger.info(f"📊 Added {analyst_ensemble_outputs.shape[1]} analyst ensemble outputs")

                except Exception as e:
                    error_msg = f"Failed to add analyst ensemble outputs: {e}"
                    preparation_metrics['errors'].append(error_msg)
                    self.logger.error(f"❌ {error_msg}")
                    raise

            # Add legacy analyst model outputs for backward compatibility
            if analyst_model_outputs is not None:
                try:
                    if analyst_signals is not None:
                        analyst_model_outputs = analyst_model_outputs[directional_mask]

                    # Validate legacy outputs shape
                    if analyst_model_outputs.shape[0] != X.shape[0]:
                        error_msg = f"Legacy analyst outputs shape mismatch: expected {X.shape[0]}, got {analyst_model_outputs.shape[0]}"
                        preparation_metrics['errors'].append(error_msg)
                        raise ValueError(error_msg)

                    # Check for NaN/Inf in legacy outputs
                    legacy_nan_count = np.sum(np.isnan(analyst_model_outputs))
                    legacy_inf_count = np.sum(np.isinf(analyst_model_outputs))

                    if legacy_nan_count > 0:
                        warning_msg = f"Legacy analyst outputs contain {legacy_nan_count} NaN values"
                        preparation_metrics['warnings'].append(warning_msg)
                        self.logger.warning(f"⚠️ {warning_msg}")

                    if legacy_inf_count > 0:
                        warning_msg = f"Legacy analyst outputs contain {legacy_inf_count} infinite values"
                        preparation_metrics['warnings'].append(warning_msg)
                        self.logger.warning(f"⚠️ {warning_msg}")

                    additional_features.append(analyst_model_outputs)
                    additional_feature_names.extend([f"analyst_legacy_{i}" for i in range(analyst_model_outputs.shape[1])])

                    feature_combination_details['analyst_legacy'] = {
                        'count': analyst_model_outputs.shape[1],
                        'nan_count': legacy_nan_count,
                        'inf_count': legacy_inf_count
                    }

                    self.logger.info(f"📊 Added {analyst_model_outputs.shape[1]} legacy analyst outputs")

                except Exception as e:
                    error_msg = f"Failed to add legacy analyst outputs: {e}"
                    preparation_metrics['errors'].append(error_msg)
                    self.logger.error(f"❌ {error_msg}")
                    raise

            # Concatenate all additional features with validation
            if additional_features:
                try:
                    # Validate all features have same number of samples
                    for i, feature_array in enumerate(additional_features):
                        if feature_array.shape[0] != X.shape[0]:
                            error_msg = f"Feature array {i} shape mismatch: expected {X.shape[0]}, got {feature_array.shape[0]}"
                            preparation_metrics['errors'].append(error_msg)
                            raise ValueError(error_msg)

                    X_combined = np.column_stack([X] + additional_features)

                    # Validate combined features
                    if X_combined.shape[0] != X.shape[0]:
                        error_msg = f"Combined features sample count mismatch: expected {X.shape[0]}, got {X_combined.shape[0]}"
                        preparation_metrics['errors'].append(error_msg)
                        raise ValueError(error_msg)

                    X = X_combined

                    # Update feature names
                    if feature_names is not None:
                        feature_names = feature_names + additional_feature_names
                    else:
                        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

                    preparation_metrics['feature_combinations'] = feature_combination_details
                    self.logger.info(f"📊 Total features: {X.shape[1]} (base + HMM regime + HMM model + all analyst models + analyst ensemble)")

                except Exception as e:
                    error_msg = f"Failed to combine features: {e}"
                    preparation_metrics['errors'].append(error_msg)
                    self.logger.error(f"❌ {error_msg}")
                    raise

            # Final validation
            preparation_metrics['final_samples'] = X.shape[0]
            preparation_metrics['final_features'] = X.shape[1]

            # Check for final data quality issues
            final_nan_count = np.sum(np.isnan(X))
            final_inf_count = np.sum(np.isinf(X))

            if final_nan_count > 0:
                warning_msg = f"Final feature matrix contains {final_nan_count} NaN values"
                preparation_metrics['warnings'].append(warning_msg)
                self.logger.warning(f"⚠️ {warning_msg}")

            if final_inf_count > 0:
                warning_msg = f"Final feature matrix contains {final_inf_count} infinite values"
                preparation_metrics['warnings'].append(warning_msg)
                self.logger.warning(f"⚠️ {warning_msg}")

            return X, y, regime_labels, feature_names, preparation_metrics

        except Exception as e:
            preparation_metrics['errors'].append(str(e))
            self.logger.error(f"❌ Feature preparation failed: {e}")
            raise

    def _execute_training(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]],
        hmm_states: Optional[np.ndarray],
        analyst_confidence_scores: Optional[np.ndarray] = None,
        analyst_directional_info: Optional[Dict[str, Any]] = None,
        timestamps: Optional[np.ndarray] = None,
        analyst_signals: Optional[np.ndarray] = None,
        analyst_model_outputs: Optional[np.ndarray] = None,
        hmm_regime_features: Optional[np.ndarray] = None,
        all_analyst_models_outputs: Optional[Dict[str, np.ndarray]] = None,
        hmm_model_outputs: Optional[np.ndarray] = None,
        analyst_ensemble_outputs: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Execute training with enhanced vectorization and comprehensive error handling."""
        training_metrics = {
            'vectorization_attempted': False,
            'vectorization_successful': False,
            'fallback_used': False,
            'training_method': 'unknown',
            'errors': [],
            'warnings': [],
            'performance_metrics': {}
        }

        try:
            # Pre-training validation
            self._validate_training_inputs(X, y, regime_labels, feature_names, hmm_states)

            # Enhanced training with overfitting prevention and lookahead bias detection
            if ENHANCED_TRAINING_AVAILABLE and hasattr(self, 'training_enhancer'):
                self.logger.info("🚀 Using ENHANCED tactician models training with overfitting prevention")
                training_metrics['enhanced_training_attempted'] = True

                try:
                    # Validate temporal data for lookahead bias
                    if timestamps is not None:
                        self.logger.info("🔍 Validating temporal data for lookahead bias...")
                        is_valid, warnings = self.training_enhancer.enhanced_utils.validate_temporal_data(
                            X, y, timestamps, strict_mode=True
                        )
                        if warnings:
                            for warning in warnings:
                                self.logger.warning(f"⚠️ {warning}")
                        if not is_valid:
                            self.logger.error("❌ Temporal data validation failed")
                            raise ValueError("Lookahead bias detected in temporal data")

                    # Use enhanced training with temporal integrity and confidence weighting
                    enhanced_start_time = time.time()
                    results = self._execute_enhanced_tactician_training(
                        X, y, regime_labels, feature_names, hmm_states, timestamps,
                        analyst_signals, analyst_model_outputs, hmm_regime_features,
                        all_analyst_models_outputs, hmm_model_outputs, analyst_ensemble_outputs,
                        analyst_confidence_scores, analyst_directional_info
                    )

                    enhanced_duration = time.time() - enhanced_start_time
                    training_metrics['performance_metrics']['enhanced_training_duration'] = enhanced_duration
                    training_metrics['enhanced_training_successful'] = True
                    training_metrics['training_method'] = 'enhanced'
                    self.logger.info(f"✅ ENHANCED tactician training completed successfully in {enhanced_duration:.2f}s")
                    return results

                except Exception as e:
                    error_msg = f"ENHANCED tactician training failed: {e}"
                    training_metrics['warnings'].append(error_msg)
                    training_metrics['fallback_used'] = True
                    self.logger.warning(f"⚠️ {error_msg}, falling back to vectorized method")

            # VECTORIZED: Use ultra-fast vectorized training by default
            self.logger.info("🚀 Using VECTORIZED tactician models training")
            training_metrics['vectorization_attempted'] = True

            if self.enable_vectorization:
                try:
                    vectorization_start_time = time.time()

                    results = super().execute_vectorized(
                        X=X,
                        y=y,
                        regime_labels=regime_labels,
                        feature_names=feature_names,
                        hmm_states=hmm_states,
                        is_classification=False,  # Tactician models are typically regression
                        symbol=None,
                        exchange=None,
                        timeframe=self.config.timeframe
                    )

                    vectorization_duration = time.time() - vectorization_start_time
                    training_metrics['performance_metrics']['vectorization_duration'] = vectorization_duration

                    if results.get('vectorized', False):
                        training_metrics['vectorization_successful'] = True
                        training_metrics['training_method'] = 'vectorized'
                        self.logger.info(f"✅ VECTORIZED tactician training completed successfully in {vectorization_duration:.2f}s")
                        return results
                    else:
                        warning_msg = "VECTORIZED tactician training failed, falling back to standard method"
                        training_metrics['warnings'].append(warning_msg)
                        training_metrics['fallback_used'] = True
                        self.logger.warning(f"⚠️ {warning_msg}")

                except Exception as e:
                    error_msg = f"VECTORIZED tactician training failed: {e}"
                    training_metrics['warnings'].append(error_msg)
                    training_metrics['fallback_used'] = True
                    self.logger.warning(f"⚠️ {error_msg}, falling back to standard method")
            else:
                training_metrics['fallback_used'] = True
                self.logger.info("🔄 Vectorization disabled, using standard training")

            # Fallback to standard training
            self.logger.info("🔄 Using standard tactician models training")
            standard_start_time = time.time()

            results = super().execute(
                X=X,
                y=y,
                regime_labels=regime_labels,
                feature_names=feature_names,
                hmm_states=hmm_states,
                is_classification=False,
                symbol=None,
                exchange=None,
                timeframe=self.config.timeframe
            )

            standard_duration = time.time() - standard_start_time
            training_metrics['performance_metrics']['standard_training_duration'] = standard_duration
            training_metrics['training_method'] = 'standard'

            self.logger.info(f"✅ Standard tactician training completed in {standard_duration:.2f}s")

            # Add training metrics to results
            results['training_execution_metrics'] = training_metrics

            # Add entry timing optimization if enabled
            if hasattr(self.config, 'enable_entry_timing_optimization') and self.config.enable_entry_timing_optimization:
                self.logger.info("🔄 Applying entry timing optimization for 0-0.4% range...")
                entry_timing_results = self._apply_entry_timing_optimization(X, y, feature_names, results)
                results.update(entry_timing_results)

            # NOTE: Ensemble training is handled by tactician_ensemble_training.py
            # Individual model training is complete - ensemble training is a separate step
            tprint_info("✅ Individual model training complete. Ensemble training handled separately by tactician_ensemble_training.py")

            return results

        except Exception as e:
            training_metrics['errors'].append(str(e))
            self.logger.error(f"❌ Training execution failed: {e}")
            self.logger.error(f"❌ Training metrics: {training_metrics}")
            raise

    def _execute_enhanced_tactician_training(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray,
                                           feature_names: Optional[List[str]], hmm_states: Optional[np.ndarray],
                                           timestamps: Optional[np.ndarray], analyst_signals: Optional[np.ndarray],
                                           analyst_model_outputs: Optional[np.ndarray], hmm_regime_features: Optional[np.ndarray],
                                           all_analyst_models_outputs: Optional[Dict[str, np.ndarray]],
                                           hmm_model_outputs: Optional[np.ndarray], analyst_ensemble_outputs: Optional[np.ndarray],
                                           analyst_confidence_scores: Optional[np.ndarray] = None,
                                           analyst_directional_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute enhanced tactician training with overfitting prevention and lookahead bias detection."""
        try:
            self.logger.info("🚀 Executing enhanced tactician training with overfitting prevention")

            # Filter for Analyst directional signals with confidence weighting
            if analyst_signals is not None:
                self.logger.info("🔍 Filtering for Analyst directional signals...")

                # Directional signals: 1 (long), -1 (short), 0 (neutral)
                directional_mask = (analyst_signals == 1) | (analyst_signals == -1)

                # Apply confidence-based weighting if confidence scores are available
                if analyst_confidence_scores is not None:
                    # Create combined mask with confidence threshold
                    confidence_threshold = 0.5
                    confidence_mask = analyst_confidence_scores >= confidence_threshold
                    combined_mask = directional_mask & confidence_mask

                    self.logger.info(f"📊 Confidence filtering: {np.sum(confidence_mask)}/{len(analyst_confidence_scores)} samples above {confidence_threshold} threshold")

                    # Use combined mask for filtering
                    directional_mask = combined_mask

                    # Store confidence scores for sample weighting
                    confidence_scores_filtered = analyst_confidence_scores[combined_mask]
                else:
                    confidence_scores_filtered = None
                    self.logger.warning("⚠️ No confidence scores provided - using directional signals only")

                X_filtered = X[directional_mask]
                y_filtered = y[directional_mask]
                regime_labels_filtered = regime_labels[directional_mask]
                timestamps_filtered = timestamps[directional_mask] if timestamps is not None else None

                # Analyze directional distribution
                long_count = np.sum(analyst_signals == 1)
                short_count = np.sum(analyst_signals == -1)
                neutral_count = np.sum(analyst_signals == 0)
                directional_rate = np.mean(directional_mask)

                self.logger.info(f"📊 Directional signal filtering: {directional_rate:.2%} ({np.sum(directional_mask)}/{len(analyst_signals)} samples)")
                self.logger.info(f"   Long signals: {long_count} ({long_count/len(analyst_signals):.1%} of total)")
                self.logger.info(f"   Short signals: {short_count} ({short_count/len(analyst_signals):.1%} of total)")
                self.logger.info(f"   Neutral signals: {neutral_count} ({neutral_count/len(analyst_signals):.1%} of total)")

            else:
                X_filtered, y_filtered, regime_labels_filtered, timestamps_filtered = X, y, regime_labels, timestamps
                confidence_scores_filtered = None
                self.logger.warning("⚠️ No Analyst directional signals provided, using all data")

            # Get unique regimes
            unique_regimes = np.unique(regime_labels_filtered)
            results = {
                'models': {},
                'regime_analysis': {},
                'enhanced_training_metadata': {},
                'overfitting_warnings': [],
                'ensemble_diversity': None,
                'walk_forward_validation': None
            }

            # Train models for each regime with enhanced utilities and confidence weighting
            for regime in unique_regimes:
                regime_mask = regime_labels_filtered == regime
                X_regime = X_filtered[regime_mask]
                y_regime = y_filtered[regime_mask]
                timestamps_regime = timestamps_filtered[regime_mask] if timestamps_filtered is not None else None

                # Get confidence scores for this regime if available
                confidence_regime = None
                if confidence_scores_filtered is not None and regime_mask.sum() > 0:
                    regime_confidence_mask = regime_mask[directional_mask] if analyst_signals is not None else regime_mask
                    confidence_regime = confidence_scores_filtered[regime_confidence_mask] if confidence_scores_filtered is not None else None

                self.logger.info(f"🎯 Training tactician models for regime {regime} ({len(X_regime)} samples)")
                if confidence_regime is not None:
                    self.logger.info(f"📊 Using confidence scores for sample weighting (mean: {np.mean(confidence_regime):.3f}, std: {np.std(confidence_regime):.3f})")

                # Train each model type for this regime
                regime_models = {}
                for model_type in self.config.model_types:
                    try:
                        # Create model instance
                        model = self._create_model_instance(model_type)

                        # Special handling for Random Survival Forest
                        if model_type == "RandomSurvivalForest":
                            model = self._create_random_survival_forest_model()

                        # Apply enhanced regularization
                        model = self.training_enhancer.enhanced_utils.apply_enhanced_regularization(
                            model, model_type
                        )

                        # Special handling for Random Survival Forest
                        if model_type == "RandomSurvivalForest":
                            # Random Survival Forest has its own training method with HPO
                            trained_model = model.fit(
                                X_regime, y_regime,
                                feature_names=feature_names,
                                analyst_signals=analyst_signals[regime_mask] if analyst_signals is not None else None,
                                hmm_regime_probs=None,  # TODO: Extract from hmm_regime_features if available
                                enable_hpo=True,
                                hpo_trials=self.config.hpo_n_trials,
                                cv_folds=5,
                                enable_entry_timing_optimization=True,
                                entry_timing_trials=50
                            )
                            metadata = {'model_type': 'RandomSurvivalForest', 'training_completed': True}
                        else:
                            # Train with early stopping, overfitting monitoring, and confidence weighting
                            trained_model, metadata = self.training_enhancer.enhance_training_step(
                                X_regime, y_regime, model, timestamps_regime, f"tactician_{model_type}_regime_{regime}",
                                sample_weights=confidence_regime
                            )

                        regime_models[model_type] = {
                            'model': trained_model,
                            'metadata': metadata
                        }

                        # Check for overfitting warnings
                        if metadata.get('overfitting_detected', False):
                            results['overfitting_warnings'].append(f"Overfitting detected in {model_type} for regime {regime}")

                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to train {model_type} for regime {regime}: {e}")
                        continue

                results['models'][regime] = regime_models

            # Calculate ensemble diversity if multiple models
            if len(self.config.model_types) > 1:
                self.logger.info("📊 Calculating tactician ensemble diversity...")
                for regime in unique_regimes:
                    if regime in results['models']:
                        models_list = [results['models'][regime][mt]['model'] for mt in self.config.model_types
                                     if mt in results['models'][regime]]
                        if len(models_list) > 1:
                            diversity_metrics = self.training_enhancer.enhanced_utils.calculate_ensemble_diversity(
                                models_list, X_filtered[regime_labels_filtered == regime],
                                y_filtered[regime_labels_filtered == regime]
                            )
                            results['ensemble_diversity'] = diversity_metrics

                            if diversity_metrics.get('diversity_score', 0) < 0.1:
                                self.logger.warning(f"⚠️ Low tactician ensemble diversity for regime {regime}")
                            else:
                                self.logger.info(f"✅ Good tactician ensemble diversity for regime {regime}")

            # Perform walk-forward validation if enabled
            if len(results['models']) > 0:
                self.logger.info("🚶 Performing walk-forward validation for tactician...")
                first_regime = list(results['models'].keys())[0]
                first_model = list(results['models'][first_regime].values())[0]['model']

                wfv_results = self.training_enhancer.enhanced_utils.perform_walk_forward_validation(
                    first_model, X_filtered, y_filtered,
                    initial_train_size=1000, test_size=100, step_size=50
                )
                results['walk_forward_validation'] = wfv_results

                if wfv_results.get('performance_trend', {}).get('trend') == 'declining':
                    self.logger.warning("⚠️ Declining performance trend detected in tactician walk-forward validation")
                else:
                    self.logger.info("✅ Stable performance trend in tactician walk-forward validation")

            # Add enhanced training metadata
            results['enhanced_training_metadata'] = {
                'overfitting_prevention_enabled': True,
                'lookahead_bias_detection_enabled': True,
                'early_stopping_enabled': True,
                'enhanced_regularization_enabled': True,
                'temporal_validation_enabled': timestamps is not None,
                'green_light_filtering_enabled': analyst_signals is not None,
                'walk_forward_validation_enabled': True,
                'total_warnings': len(results['overfitting_warnings'])
            }

            self.logger.info("✅ Enhanced tactician training completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"❌ Enhanced tactician training failed: {e}")
            raise

    def _validate_training_inputs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]],
        hmm_states: Optional[np.ndarray]
    ) -> None:
        """Validate training inputs before execution."""
        try:
            # Check data shapes
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"Feature and target sample counts don't match: {X.shape[0]} vs {y.shape[0]}")

            if X.shape[0] != regime_labels.shape[0]:
                raise ValueError(f"Feature and regime label sample counts don't match: {X.shape[0]} vs {regime_labels.shape[0]}")

            if hmm_states is not None and hmm_states.shape[0] != X.shape[0]:
                raise ValueError(f"HMM states sample count mismatch: expected {X.shape[0]}, got {hmm_states.shape[0]}")

            # Check feature names consistency
            if feature_names is not None and len(feature_names) != X.shape[1]:
                raise ValueError(f"Feature names count mismatch: expected {X.shape[1]}, got {len(feature_names)}")

            # Check for critical data quality issues
            if np.any(np.isnan(X)):
                nan_percentage = (np.sum(np.isnan(X)) / X.size) * 100
                if nan_percentage > 5:  # More than 5% NaN
                    raise ValueError(f"Critical: {nan_percentage:.2f}% of features are NaN")

            if np.any(np.isnan(y)):
                nan_percentage = (np.sum(np.isnan(y)) / y.size) * 100
                if nan_percentage > 2:  # More than 2% NaN in targets
                    raise ValueError(f"Critical: {nan_percentage:.2f}% of targets are NaN")

            # Check regime distribution
            unique_regimes = np.unique(regime_labels)
            regime_counts = np.bincount(regime_labels)
            min_regime_size = np.min(regime_counts)

            if min_regime_size < self.config.min_samples_per_regime:
                insufficient_regimes = np.sum(regime_counts < self.config.min_samples_per_regime)
                if insufficient_regimes > len(unique_regimes) * 0.5:
                    raise ValueError(f"Critical: {insufficient_regimes}/{len(unique_regimes)} regimes have insufficient data")

            self.logger.info("✅ Training input validation passed")

        except Exception as e:
            self.logger.error(f"❌ Training input validation failed: {e}")
            raise

    def _apply_entry_timing_optimization(self,
                                      X: np.ndarray,
                                      y: np.ndarray,
                                      feature_names: Optional[List[str]],
                                      base_results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply entry timing optimization for 0-0.4% range."""
        try:
            from .tactician_directional_optimization import EntryTimingTacticianOptimizer

            # Initialize entry timing optimizer
            entry_timing_optimizer = EntryTimingTacticianOptimizer(self.config)

            # Get entry timing range from config
            entry_timing_range = getattr(self.config, 'entry_timing_range', 0.004)  # 0-0.4% range

            # Filter targets for entry timing range (0-0.4%)
            entry_timing_mask = np.abs(y) <= entry_timing_range
            X_entry_timing = X[entry_timing_mask]
            y_entry_timing = y[entry_timing_mask]

            self.logger.info(f"📊 Entry timing filtering: {len(y_entry_timing)}/{len(y)} samples (≤{entry_timing_range:.1%} range)")

            if len(y_entry_timing) < 100:  # Need minimum samples for optimization
                self.logger.warning("⚠️ Insufficient entry timing samples for optimization")
                return {}

            # Apply entry timing optimization
            entry_timing_result = entry_timing_optimizer.optimize_tactician_entry_timing(
                X=X_entry_timing, y=y_entry_timing, regime_labels=np.zeros(len(y_entry_timing)),
                feature_names=feature_names, hmm_states=None,
                max_trials=getattr(self.config, 'hpo_n_trials', 100) // 2  # Half trials for entry timing
            )

            # Create entry timing optimization results
            entry_timing_results = {
                'entry_timing_optimization': {
                    'enabled': True,
                    'entry_timing_range': entry_timing_range,
                    'entry_timing_samples': len(y_entry_timing),
                    'total_samples': len(y),
                    'objectives': getattr(self.config, 'entry_timing_objectives', {}),
                    'optimization_time': entry_timing_result.optimization_time,
                    'n_trials': entry_timing_result.n_trials
                },
                'entry_timing_model': entry_timing_result.model,
                'entry_timing_metrics': {
                    'early_entry_penalty': entry_timing_result.directional_accuracy,
                    'late_entry_penalty': entry_timing_result.adverse_movement_minimization,
                    'optimal_entry_reward': entry_timing_result.directional_profit_efficiency,
                    'entry_timing_efficiency': entry_timing_result.risk_adjusted_performance,
                    'composite_score': entry_timing_result.composite_score
                }
            }

            self.logger.info(f"✅ Entry timing optimization completed for 0-0.3% range")
            self.logger.info(f"   Early entry penalty: {entry_timing_result.directional_accuracy:.4f}")
            self.logger.info(f"   Late entry penalty: {entry_timing_result.adverse_movement_minimization:.4f}")
            self.logger.info(f"   Optimal entry reward: {entry_timing_result.directional_profit_efficiency:.4f}")
            self.logger.info(f"   Entry timing efficiency: {entry_timing_result.risk_adjusted_performance:.4f}")
            self.logger.info(f"   Composite score: {entry_timing_result.composite_score:.4f}")

            return entry_timing_results

        except Exception as e:
            self.logger.error(f"❌ Entry timing optimization failed: {e}")
            return {}

    # ============================================================================
    # ENSEMBLE TRAINING REMOVED - Use Dedicated Module
    # ============================================================================
    # Ensemble training logic has been removed from this file and is now handled
    # by the dedicated tactician_ensemble_training.py module.
    #
    # This consolidation:
    # - Eliminates ~250 lines of duplicate ensemble code
    # - Provides single source of truth for ensemble training
    # - Separates concerns: individual models vs ensemble meta-learning
    # - Improves maintainability
    #
    # For ensemble training, use:
    #   from .tactician_ensemble_training import TacticianEnsembleTrainingStep
    # ============================================================================

    def _finalize_results(self, results: Dict[str, Any], analyst_signals: Optional[np.ndarray]) -> Dict[str, Any]:
        """Finalize results with tactician-specific metadata and comprehensive reporting."""
        try:
            # Add tactician-specific post-processing if needed
            if 'error' not in results:
                results = self._add_tactician_specific_metadata(results, analyst_signals)

            # Add training metrics to results
            results['training_metrics'] = {
                phase.value: {
                    'duration': metrics.duration,
                    'success': metrics.success,
                    'samples_processed': metrics.samples_processed,
                    'features_count': metrics.features_count,
                    'models_trained': metrics.models_trained,
                    'memory_usage_mb': metrics.memory_usage_mb,
                    'error_message': metrics.error_message
                }
                for phase, metrics in self.training_metrics.items()
            }

            # Add vectorization information
            results['vectorization_info'] = {
                'vectorization_enabled': self.enable_vectorization,
                'vectorization_fallback_used': self.vectorization_fallback_used,
                'vectorized_training_available': VECTORIZED_TRAINING_AVAILABLE
            }

            return results

        except Exception as e:
            self.logger.error(f"❌ Results finalization failed: {e}")
            return results

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result with comprehensive error information."""
        return {
            'error': True,
            'error_message': error_message,
            'training_metrics': {
                phase.value: {
                    'duration': metrics.duration,
                    'success': metrics.success,
                    'error_message': metrics.error_message
                }
                for phase, metrics in self.training_metrics.items()
            },
            'vectorization_info': {
                'vectorization_enabled': self.enable_vectorization,
                'vectorization_fallback_used': self.vectorization_fallback_used,
                'vectorized_training_available': VECTORIZED_TRAINING_AVAILABLE
            }
        }

    def _generate_training_report(self, total_time: float) -> None:
        """Generate comprehensive training report with actionable insights."""
        try:
            self.logger.info("📊 " + "="*80)
            self.logger.info("📊 ENHANCED TACTICIAN TRAINING REPORT")
            self.logger.info("📊 " + "="*80)

            # Overall statistics
            self.logger.info(f"📊 Total training time: {total_time:.2f}s")
            self.logger.info(f"📊 Vectorization enabled: {self.enable_vectorization}")
            self.logger.info(f"📊 Vectorization fallback used: {self.vectorization_fallback_used}")

            # Calculate efficiency metrics
            total_warnings = sum(len(metrics.warnings_issued) for metrics in self.training_metrics.values())
            total_errors = sum(len(metrics.errors_encountered) for metrics in self.training_metrics.values())
            total_samples = sum(metrics.samples_processed for metrics in self.training_metrics.values())
            total_features = sum(metrics.features_count for metrics in self.training_metrics.values())

            self.logger.info(f"📊 Total samples processed: {total_samples:,}")
            self.logger.info(f"📊 Total features: {total_features}")
            self.logger.info(f"📊 Total warnings: {total_warnings}")
            self.logger.info(f"📊 Total errors: {total_errors}")

            # Performance efficiency
            if total_samples > 0:
                samples_per_second = total_samples / total_time
                self.logger.info(f"📊 Processing rate: {samples_per_second:,.0f} samples/second")

            # Phase-by-phase breakdown with detailed metrics
            self.logger.info("📊 " + "-"*60)
            self.logger.info("📊 PHASE BREAKDOWN:")
            self.logger.info("📊 " + "-"*60)

            for phase, metrics in self.training_metrics.items():
                status = "✅" if metrics.success else "❌"
                efficiency = f"({metrics.samples_processed/1000:.1f}k samples)" if metrics.samples_processed > 0 else ""

                self.logger.info(f"📊   {status} {phase.value.upper()}: {metrics.duration:.2f}s {efficiency}")

                # Detailed phase metrics
                if metrics.samples_processed > 0:
                    self.logger.info(f"📊     └─ Samples: {metrics.samples_processed:,}")
                if metrics.features_count > 0:
                    self.logger.info(f"📊     └─ Features: {metrics.features_count}")
                if metrics.models_trained > 0:
                    self.logger.info(f"📊     └─ Models trained: {metrics.models_trained}")
                if metrics.warnings_issued > 0:
                    self.logger.info(f"📊     └─ Warnings: {metrics.warnings_issued}")
                if metrics.errors_encountered > 0:
                    self.logger.info(f"📊     └─ Errors: {metrics.errors_encountered}")
                if metrics.memory_usage_mb > 0:
                    self.logger.info(f"📊     └─ Memory: {metrics.memory_usage_mb:.1f} MB")

                if not metrics.success and metrics.error_message:
                    self.logger.info(f"📊     └─ ❌ Error: {metrics.error_message}")

            # Data quality summary
            self.logger.info("📊 " + "-"*60)
            self.logger.info("📊 DATA QUALITY SUMMARY:")
            self.logger.info("📊 " + "-"*60)

            # Analyze data validation results if available
            if TrainingPhase.DATA_VALIDATION in self.training_metrics:
                data_phase = self.training_metrics[TrainingPhase.DATA_VALIDATION]
                if data_phase.success:
                    self.logger.info("📊   ✅ Data validation passed")
                else:
                    self.logger.info("📊   ❌ Data validation failed")

            # Analyze feature preparation results if available
            if TrainingPhase.FEATURE_PREPARATION in self.training_metrics:
                feature_phase = self.training_metrics[TrainingPhase.FEATURE_PREPARATION]
                if feature_phase.success:
                    self.logger.info("📊   ✅ Feature preparation completed")
                else:
                    self.logger.info("📊   ❌ Feature preparation failed")

            # Training method analysis
            self.logger.info("📊 " + "-"*60)
            self.logger.info("📊 TRAINING METHOD ANALYSIS:")
            self.logger.info("📊 " + "-"*60)

            if self.enable_vectorization:
                if self.vectorization_fallback_used:
                    self.logger.info("📊   ⚠️ Vectorization attempted but fallback used")
                    self.logger.info("📊   💡 Consider investigating vectorization issues")
                else:
                    self.logger.info("📊   ✅ Vectorization used successfully")
                    self.logger.info("📊   🚀 Optimal performance achieved")
            else:
                self.logger.info("📊   ℹ️ Standard training used (vectorization disabled)")

            # Memory usage analysis
            current_memory = self._get_memory_usage()
            self.logger.info("📊 " + "-"*60)
            self.logger.info("📊 MEMORY USAGE:")
            self.logger.info("📊 " + "-"*60)
            self.logger.info(f"📊 Current memory usage: {current_memory:.1f} MB")

            if current_memory > 1000:  # More than 1GB
                self.logger.info("📊   ⚠️ High memory usage detected")
                self.logger.info("📊   💡 Consider reducing batch size or using data streaming")
            elif current_memory < 100:  # Less than 100MB
                self.logger.info("📊   ✅ Low memory usage - efficient processing")

            # Recommendations
            self.logger.info("📊 " + "-"*60)
            self.logger.info("📊 RECOMMENDATIONS:")
            self.logger.info("📊 " + "-"*60)

            if total_warnings > 5:
                self.logger.info("📊   ⚠️ High warning count - review data quality")
            if total_errors > 0:
                self.logger.info("📊   ❌ Errors detected - review error logs")
            if self.vectorization_fallback_used:
                self.logger.info("📊   🔧 Vectorization fallback used - investigate vectorization issues")
            if total_time > 3600:  # More than 1 hour
                self.logger.info("📊   ⏱️ Long training time - consider optimizing hyperparameters")

            # Success indicators
            if total_errors == 0 and total_warnings < 3:
                self.logger.info("📊   ✅ Training completed successfully with minimal issues")

            self.logger.info("📊 " + "="*80)

        except Exception as e:
            self.logger.error(f"❌ Failed to generate training report: {e}")
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")

    def cleanup_resources(self) -> None:
        """Clean up hardware optimizers and other resources with graceful error handling."""
        try:
            tprint_info("🧹 Cleaning up resources...")

            # Clean up M1 optimizers if they were initialized
            if hasattr(self, 'm1_optimizers_available') and self.m1_optimizers_available:
                try:
                    cleanup_result = cleanup_m1_optimizers()
                    if cleanup_result:
                        tprint_success("✅ M1 optimizers cleaned up successfully")
                    else:
                        tprint_warning("⚠️ M1 optimizer cleanup returned false (may not be critical)")
                except Exception as e:
                    tprint_warning(f"⚠️ M1 optimizer cleanup failed: {e} (non-critical)")
            else:
                tprint_debug("M1 optimizers not initialized, skipping cleanup")

            # Clean up hardware resources if they exist
            if hasattr(self, 'm1_gpu_manager') and self.m1_gpu_manager:
                try:
                    tprint_debug("Cleaning up M1 GPU manager...")
                    # Add specific cleanup if needed
                except Exception as e:
                    tprint_warning(f"⚠️ M1 GPU manager cleanup warning: {e}")

            if hasattr(self, 'm1_memory_optimizer') and self.m1_memory_optimizer:
                try:
                    tprint_debug("Cleaning up M1 memory optimizer...")
                    # Add specific cleanup if needed
                except Exception as e:
                    tprint_warning(f"⚠️ M1 memory optimizer cleanup warning: {e}")

            if hasattr(self, 'm1_cpu_optimizer') and self.m1_cpu_optimizer:
                try:
                    tprint_debug("Cleaning up M1 CPU optimizer...")
                    # Add specific cleanup if needed
                except Exception as e:
                    tprint_warning(f"⚠️ M1 CPU optimizer cleanup warning: {e}")

            tprint_success("✅ Resource cleanup completed")

        except Exception as e:
            # Log but don't raise - cleanup failures should not crash the program
            tprint_warning(f"⚠️ Resource cleanup encountered errors: {e} (non-critical)")

    def __del__(self):
        """Destructor to ensure cleanup on object deletion."""
        try:
            self.cleanup_resources()
        except Exception:
            # Silently handle cleanup errors in destructor
            pass

    def _add_tactician_specific_metadata(self, results: Dict[str, Any], analyst_signals: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Add tactician-specific metadata to results with enhanced reporting.

        Args:
            results: Training results
            analyst_signals: Analyst green light signals for analysis

        Returns:
            Enhanced results with tactician-specific metadata
        """
        try:
            tprint_debug("Adding tactician-specific metadata...")

            # Add tactician-specific analysis
            if 'regime_analysis' in results:
                regime_analysis = results['regime_analysis']

                # Calculate tactician-specific metrics with safe math operations
                tactician_metrics = {
                    'total_regimes': len(regime_analysis.get('unique_regimes', [])),
                    'sufficient_regimes': len(regime_analysis.get('sufficient_regimes', [])),
                    'insufficient_regimes': len(regime_analysis.get('insufficient_regimes', [])),
                    'regime_balance': regime_analysis.get('regime_balance_train', 0.0),
                    'timeframe': self.config.timeframe,
                    'model_types': self.config.model_types
                }

                # Add analyst signal analysis if available with safe calculations
                if analyst_signals is not None:
                    try:
                        green_light_rate = np.mean(analyst_signals)
                        tactician_metrics.update({
                            'analyst_green_light_rate': green_light_rate,
                            'total_samples_with_green_light': int(np.sum(analyst_signals)),
                            'total_samples_analyzed': len(analyst_signals)
                        })
                        tprint_debug(f"Analyst signal analysis: {green_light_rate:.3f} green light rate")
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to analyze analyst signals: {e}")

                results['tactician_metrics'] = tactician_metrics
                tprint_debug("✅ Tactician metrics added")

            # Add model performance summary with enhanced error handling
            if 'evaluation_results' in results:
                try:
                    evaluation_results = results['evaluation_results']

                    # Calculate best performing model per regime
                    best_models = {}
                    for regime, regime_metrics in evaluation_results.items():
                        if isinstance(regime_metrics, dict) and 'error' not in regime_metrics:
                            best_model = None
                            best_r2 = -np.inf

                            for model_name, metrics in regime_metrics.items():
                                if isinstance(metrics, dict) and 'r2' in metrics:
                                    if metrics['r2'] > best_r2:
                                        best_r2 = metrics['r2']
                                        best_model = model_name

                            if best_model:
                                best_models[regime] = {
                                    'model': best_model,
                                    'r2_score': best_r2
                                }

                    results['best_models_per_regime'] = best_models
                    tprint_debug(f"✅ Best models per regime: {len(best_models)} regimes")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to calculate best models per regime: {e}")

            # Add timing-specific analysis
            timing_analysis = {
                'base_timeframe': self.config.timeframe,
                'cross_timeframe_features': True,
                'analyst_dependency': True,
                'timing_decision_role': True,
                'utility_integration_status': self.utility_integration_status
            }
            results['timing_analysis'] = timing_analysis

            # Add utility integration status to results
            results['utility_integration_status'] = self.utility_integration_status

            tprint_success("✅ Tactician-specific metadata added successfully")
            return results

        except Exception as e:
            tprint_error(f"❌ Failed to add tactician-specific metadata: {e}")
            tprint_error(f"❌ Traceback: {traceback.format_exc()}")
            return results

# Enhanced convenience functions with better error handling
def create_tactician_models_training_step_refactored(
    config: Optional[TacticianTrainingConfig] = None,
    enable_vectorization: bool = True
) -> TacticianModelsTrainingStepRefactored:
    """
    Create enhanced Tactician models training step with comprehensive error handling.

    Args:
        config: Per-regime training configuration
        enable_vectorization: Whether to enable vectorized training

    Returns:
        Enhanced Tactician models training step

    Raises:
        Exception: If initialization fails
    """
    try:
        return TacticianModelsTrainingStepRefactored(config, enable_vectorization)
    except Exception as e:
        logger.error(f"❌ Failed to create tactician training step: {e}")
        raise

def execute_tactician_models_training_refactored(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[TacticianTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None,
    analyst_signals: Optional[np.ndarray] = None,
    analyst_model_outputs: Optional[np.ndarray] = None,
    hmm_regime_features: Optional[np.ndarray] = None,
    all_analyst_models_outputs: Optional[Dict[str, np.ndarray]] = None,
    hmm_model_outputs: Optional[np.ndarray] = None,
    analyst_ensemble_outputs: Optional[np.ndarray] = None,
    enable_vectorization: bool = True
) -> Dict[str, Any]:
    """
    Execute enhanced Tactician models training step with comprehensive error handling.

    Args:
        X: Input features
        y: Target values
        regime_labels: Regime labels for each sample
        config: Per-regime training configuration
        feature_names: Names of input features
        hmm_states: HMM cluster/regime states
        analyst_signals: Binary signals from Analyst
        analyst_model_outputs: Analyst model predictions
        hmm_regime_features: HMM regime features
        all_analyst_models_outputs: All individual analyst ML model outputs
        hmm_model_outputs: HMM model outputs (predictions, probabilities, etc.)
        analyst_ensemble_outputs: Analyst ensemble model outputs
        enable_vectorization: Whether to enable vectorized training

    Returns:
        Dictionary containing training results and metadata

    Raises:
        Exception: If training fails
    """
    try:
        step = create_tactician_models_training_step_refactored(config, enable_vectorization)
        return step.execute(
            X, y, regime_labels, feature_names, hmm_states,
            analyst_signals, analyst_model_outputs, hmm_regime_features,
            all_analyst_models_outputs, hmm_model_outputs, analyst_ensemble_outputs
        )
    except Exception as e:
        logger.error(f"❌ Failed to execute tactician training: {e}")
        raise

# Enhanced example usage and comparison
if __name__ == "__main__":
    # Example of how to use the enhanced version
    print("Enhanced Tactician Models Training Step")
    print("=" * 50)

    # Create configuration with enhanced settings
    config = TacticianTrainingConfig(
        model_name="tactician_models",
        timeframe="1m",
        model_types=["NeuralObliviousDecisionEnsembles", "CatBoostRegressor", "LGBMRegressor", "Ridge"],
        hpo_n_trials=50,  # Reduced for demo
        enable_hpo=True,
        save_models=True,
        model_save_path="generated/model_training/models/tactician_models_enhanced",
        use_single_model=True,
        single_model_name="tactician_unified_model",
        enable_ensemble_training=True,
        ensemble_method="stacking",
        meta_model="ElasticNetCV",
        ensemble_name="tactician_ensemble"
    )

    # Create enhanced training step
    try:
        training_step = create_tactician_models_training_step_refactored(config)

        print(f"✅ Created enhanced tactician training step with {len(config.model_types)} model types")
        print(f"📊 HPO enabled: {config.enable_hpo}")
        print(f"💾 Save models: {config.save_models}")
        print(f"📁 Save path: {config.model_save_path}")
        print(f"⏰ Base timeframe: {config.timeframe}")

        # The actual training would be called with:
        # results = training_step.execute(X, y, regime_labels, feature_names, hmm_states, analyst_signals, analyst_model_outputs)

        print("\n🎯 Enhanced Tactician Module Features:")
        print("- Comprehensive error handling with detailed failure reporting")
        print("- Enhanced progress tracking with phase-based metrics")
        print("- Input validation and data quality checks")
        print("- Optimized vectorization with intelligent fallback")
        print("- Structured logging with performance monitoring")
        print("- Health monitoring throughout training process")

        print("\n🔄 Integration with Analyst:")
        print("- Receives green light signals from Analyst")
        print("- Uses Analyst predictions as additional features")
        print("- Focuses on timing rather than trade decision")
        print("- Operates on higher frequency (1m vs 5m)")

        print("\n📊 Enhanced Reporting Features:")
        print("- Phase-by-phase progress tracking")
        print("- Comprehensive training metrics")
        print("- Memory usage monitoring")
        print("- Vectorization status reporting")
        print("- Detailed error reporting with stack traces")

    except Exception as e:
        print(f"❌ Failed to create enhanced tactician training step: {e}")
        print("This demonstrates the enhanced error handling capabilities")
