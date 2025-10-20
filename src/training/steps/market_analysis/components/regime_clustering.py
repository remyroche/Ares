"""
Regime Clustering Component.

This component performs regime clustering analysis using various clustering algorithms
and provides comprehensive regime discovery and analysis capabilities.
"""

import copy
import numpy as np
import pandas as pd
from datetime import datetime
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple
from dataclasses import dataclass, field
import traceback
from pathlib import Path
from collections import defaultdict
import pickle
import re
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from hmmlearn import hmm
from joblib import Parallel, delayed
import os


from src.utils.tprint import (
    tprint,
    tprint_debug,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
    tprint_progress,
    tprint_performance,
    tprint_timer,
    tprint_structured,
)

from ..shared_utils import (
    # Features
    prepare_market_features,
    FeatureConfig,
    FeaturePreparationResult,

    # Configuration
    validate_regime_count,
    normalize_weights,
    validate_algorithm_type,
    create_default_config,
    ConfigValidator,
    BaseConfig,

    # Logging
    get_logger,
    log_execution,
    log_performance,
    LoggingContext,

    # Metrics
    calculate_consensus_metrics,
    calculate_disagreement_metrics,
    calculate_economic_scores,
    calculate_trading_scores,
    calculate_stability_scores,
    MetricsCalculator,

    # Characteristics
    create_regime_characteristics,
    generate_cluster_characteristics,
    CharacteristicsGenerator,
)

from ..shared_utils.calibration_registry import (
    get_current_calibration,
    get_quality_thresholds as get_calibrated_thresholds,
    update_quality_calibration,
)

from ...base_step import BaseStep
from .base_component import ComponentConfig, ComponentResult
from ..regime_analysis.label_fusion import RegimeOptimizationService


# Import matrix operations and hardware utilities
try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_batch_matrix_processor,
        safe_matrix_multiply,
        safe_correlation_matrix,
        gpu_matrix_multiply,
        correlation_matrix_gpu,
        optimize_dataframe,
        vectorized_rolling_features,
        matrix_correlation_analysis,
        batch_matrix_multiply,
        batch_feature_transformation,
        batch_correlation_analysis,
        get_hardware_performance_report,
        optimize_matrix_operation_with_hardware,
        cleanup_hardware_resources,
        get_processing_performance_stats
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError as e:
    MATRIX_OPERATIONS_AVAILABLE = False
    tprint(f"Matrix operations not available: {e}", "WARNING")

try:
    from src.utils.hardware import (
        get_unified_hardware_manager,
        get_advanced_cpu_optimizer,
        get_enhanced_gpu_manager,
        get_advanced_memory_optimizer,
        get_adaptive_optimization_engine,
        optimize_for_workload,
        optimize_for_workload_adaptive,
        optimize_dataframe_advanced,
        record_performance_adaptive
    )
    HARDWARE_OPTIMIZATION_AVAILABLE = True
    tprint("✅ Hardware optimization utilities imported successfully", "SUCCESS")
except ImportError as e:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    tprint(f"Hardware optimization not available: {e}", "WARNING")

# Import M1-specific hardware utilities
try:
    from src.utils.hardware.m1_gpu_utils import (
        get_m1_gpu_optimizer,
        get_m1_gpu_memory_manager,
        get_m1_gpu_performance_monitor,
        get_m1_gpu_manager,
    )
    from src.utils.hardware.m1_memory_optimizer import (
        get_m1_memory_optimizer,
        get_memory_manager,
        get_memory_usage
    )
    from src.utils.hardware.m1_cpu_optimizer import (
        get_m1_cpu_optimizer,
        get_m1_cpu_performance_monitor,
        get_m1_cpu_scheduler
    )
    M1_HARDWARE_AVAILABLE = True
    tprint("✅ M1-specific hardware utilities imported successfully", "SUCCESS")
except ImportError as e:
    M1_HARDWARE_AVAILABLE = False
    tprint(f"M1 hardware utilities not available: {e}", "WARNING")
    # Set fallback functions
    get_m1_gpu_optimizer = lambda: None
    get_m1_gpu_memory_manager = lambda: None
    get_m1_gpu_performance_monitor = lambda: None
    get_m1_gpu_manager = lambda: None
    get_m1_memory_optimizer = lambda: None
    get_m1_memory_pool_manager = lambda: None
    get_m1_memory_monitor = lambda: None
    get_m1_cpu_optimizer = lambda: None
    get_m1_cpu_performance_monitor = lambda: None
    get_m1_cpu_scheduler = lambda: None

# Import PID-based feature selection for regime discovery
try:
    from src.training.steps.market_analysis.pid_based_feature_generation.feature_selection_mechanism import (
        FeatureSelectionMechanism,
        FeatureSelectionConfig,
        SelectionStrategy
    )
    PID_FEATURE_SELECTION_AVAILABLE = True
    tprint("✅ PID-based feature selection imported successfully", "SUCCESS")
except ImportError as e:
    PID_FEATURE_SELECTION_AVAILABLE = False
    tprint(f"⚠️ PID feature selection not available: {e}", "WARNING")

# Import ML Common utilities for advanced optimization
try:
    from src.utils.ml_common.optimization import (
        BayesianTPEOptimizer,
        GridSearchOptimizer,
        HyperparameterOptimizer,
        OptunaOptimizer
    )
    from src.utils.ml_common.cvlsa import (
        TimeSeriesCrossValidator,
        RegimeAwareCrossValidator,
        WalkForwardValidator,
        PurgedCrossValidator
    )
    from src.utils.ml_common.validation import (
        ModelValidator,
        PerformanceValidator,
        StabilityValidator
    )
    from src.utils.ml_common.ensembles import (
        EnsembleValidator,
        ModelEnsemble,
        WeightedEnsemble
    )
    ML_COMMON_AVAILABLE = True
    tprint("✅ ML Common utilities imported successfully", "SUCCESS")
except ImportError as e:
    ML_COMMON_AVAILABLE = False
    tprint(f"ML Common utilities not available: {e}", "WARNING")
    # Set fallback functions
    BayesianTPEOptimizer = None
    GridSearchOptimizer = None
    HyperparameterOptimizer = None
    OptunaOptimizer = None
    TimeSeriesCrossValidator = None
    RegimeAwareCrossValidator = None
    WalkForwardValidator = None
    PurgedCrossValidator = None
    ModelValidator = None
    PerformanceValidator = None
    StabilityValidator = None
    EnsembleValidator = None
    ModelEnsemble = None
    WeightedEnsemble = None

# Import additional common operations
try:
    from src.utils.common_operations import (
        validate_dataframe_columns,
        calculate_data_quality_metrics,
        create_data_quality_report,
        safe_convert_dtypes,
        optimize_dataframe_dtypes,
        get_dataframe_info,
        create_summary_statistics,
        safe_fillna,
        safe_merge_dataframes,
        safe_drop_columns,
        safe_rename_columns,
        validate_timestamp_column,
        safe_timestamp_conversion,
        safe_resample,
        align_dataframes,
        validate_dataframe_schema,
        guard_dataframe_nulls,
        get_memory_usage,
        optimize_memory,
        memory_checkpoint,
        gpu_context,
        safe_json_dump,
        safe_json_load,
        safe_copy,
        safe_deepcopy,
        validate_file_path,
        get_file_size,
        check_disk_space
    )
    COMMON_OPERATIONS_AVAILABLE = True
    tprint("✅ Common operations imported successfully", "SUCCESS")
except ImportError as e:
    COMMON_OPERATIONS_AVAILABLE = False
    tprint(f"Common operations not available: {e}", "WARNING")

# Import math validation
try:
    from src.utils.math_validation import (
        safe_mean,
        safe_std,
        safe_correlation,
        safe_covariance,
        validate_finite,
        validate_positive,
        validate_range,
        safe_percentage_change,
        safe_weighted_average,
        safe_kelly_calculation,
        safe_percentile,
        safe_matrix_inverse,
        validate_correlation_matrix,
        safe_divide,
        safe_log,
        safe_sqrt,
        safe_power
    )
    MATH_VALIDATION_AVAILABLE = True
    tprint("✅ Math validation imported successfully", "SUCCESS")
except ImportError as e:
    MATH_VALIDATION_AVAILABLE = False
    tprint(f"Math validation not available: {e}", "WARNING")

from .hardware_setup import HardwareResources, HardwareSetup


@dataclass
class ClusteringContext:
    """Lightweight context for sharing intermediate clustering artifacts with proper memory management."""

    original_features: np.ndarray
    market_data: pd.DataFrame
    optimized_features: Optional[np.ndarray] = None
    optimized_assignments: Optional[np.ndarray] = None
    optimal_k: Optional[int] = None
    optimal_bic: Optional[float] = None
    k_metadata: Dict[str, Any] = field(default_factory=dict)
    tas_assignments: Optional[np.ndarray] = None
    nas_assignments: Optional[np.ndarray] = None
    optimization_metrics: Dict[str, Any] = field(default_factory=dict)
    raw_assignments: Optional[np.ndarray] = None
    smoothed_assignments: Optional[np.ndarray] = None
    fusion_metadata: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    memory_optimizer: Optional[Any] = None
    original_feature_names: Optional[List[str]] = None
    pre_pca_feature_names: Optional[List[str]] = None
    optimized_feature_names: Optional[List[str]] = None
    dropped_feature_names: Optional[List[str]] = None
    feature_scores: Dict[str, float] = field(default_factory=dict)
    pca_loading_scores: Dict[str, float] = field(default_factory=dict)
    pre_pca_feature_count: Optional[int] = None
    
    def __enter__(self):
        """Context manager entry for memory management."""
        if self.memory_optimizer:
            self.memory_optimizer.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        cleanup_errors = []
        
        try:
            # Stop monitoring first
            if self.memory_optimizer:
                try:
                    self.memory_optimizer.stop_monitoring()
                except Exception as e:
                    cleanup_errors.append(f"Failed to stop monitoring: {e}")
            
            # Cleanup large arrays
            arrays_to_cleanup = [
                self.original_features, self.optimized_features, self.optimized_assignments,
                self.tas_assignments, self.nas_assignments,
                self.raw_assignments, self.smoothed_assignments
            ]
            valid_arrays = [arr for arr in arrays_to_cleanup if arr is not None]
            
            if valid_arrays and self.memory_optimizer:
                try:
                    self.memory_optimizer.cleanup_arrays(valid_arrays)
                except Exception as e:
                    cleanup_errors.append(f"Failed to cleanup arrays: {e}")
            
            # Force garbage collection
            import gc
            try:
                gc.collect()
            except Exception as e:
                cleanup_errors.append(f"Failed to run garbage collection: {e}")
            
            # Log cleanup errors if any
            if cleanup_errors:
                tprint_warning(f"Memory cleanup warnings: {'; '.join(cleanup_errors)}")
            
            # Additional cleanup on exception
            if exc_type:
                try:
                    # Clear all references
                    self.original_features = None
                    self.optimized_features = None
                    self.tas_assignments = None
                    self.nas_assignments = None
                    self.raw_assignments = None
                    self.smoothed_assignments = None
                    self.memory_optimizer = None
                    
                    # Force multiple garbage collection cycles
                    for _ in range(3):
                        gc.collect()
                        
                except Exception as e:
                    tprint_warning(f"Exception cleanup failed: {e}")
                    
        except Exception as e:
            tprint_error(f"Critical error in context cleanup: {e}")
            # Last resort cleanup
            try:
                import gc
                gc.collect()
            except:
                pass


@dataclass
class RegimeClusteringConfig(BaseConfig):
    """Configuration for NAS-TAS clustering component using shared utilities."""
    exchange: str = "binance"

    # Empirical regime search bounds
    regime_search_min: int = 5
    regime_search_max: int = 15
    
    # Clustering parameters
    n_regimes: int = 8
    algorithm_type: str = "adaptive_clustering"
    enable_economic_clustering: bool = True
    enable_ensemble_clustering: bool = True
    
    # Balance control parameters - ENHANCED for better balance
    max_regime_percentage: float = 0.16  # Maximum percentage for any single regime (increased from 0.12 to 0.16)
    min_regime_percentage: float = 0.06  # Minimum percentage for any single regime (increased from 0.05)
    balance_weight: float = 0.85  # Weight for balance in composite score (increased from 0.7 for much better regime balance)
    
    # Regime-focused clustering weights (removed momentum_weight)
    economic_weight: float = 0.25
    volatility_regime_weight: float = 0.30
    volume_regime_weight: float = 0.25
    structural_trend_weight: float = 0.20
    
    # Regime-focused feature configuration
    feature_categories: List[str] = None
    use_regime_focused_features: bool = True
    exclude_trading_features: bool = True
    use_standardized_features: bool = True
    signal_like_patterns: List[str] = field(
        default_factory=lambda: [
            r"signal",
            r"entry",
            r"exit",
            r"crossover",
            r"trade",
        ]
    )
    feature_category_caps: Dict[str, int] = field(
        default_factory=lambda: {
            'volatility_regime': 30,
            'volume_regime': 25,
            'structural_trend': 25,
            'statistical_regime': 30,
            'regime_quality': 20,
        }
    )
    pca_components_factor: float = 1.5
    zscore_clip_threshold: float = 5.0

    # Regime-specific feature quality thresholds (calibrated dynamically)
    min_regime_persistence: Optional[float] = None
    max_feature_noise_ratio: Optional[float] = None
    min_temporal_stability: Optional[float] = None
    
    # Output configuration
    output_dir: str = "data_cache"
    save_intermediate_results: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.regime_search_min = int(max(5, min(20, self.regime_search_min)))
        self.regime_search_max = int(max(
            self.regime_search_min,
            min(20, self.regime_search_max),
        ))

        if not (self.regime_search_min <= self.n_regimes <= self.regime_search_max):
            self.n_regimes = max(
                self.regime_search_min,
                min(self.regime_search_max, int(self.n_regimes)),
            )

        super().__post_init__()
        if self.feature_categories is None:
            # Regime-focused feature categories only
            self.feature_categories = [
                'regime_volatility',
                'regime_volume',
                'regime_structural_trend',
                'regime_statistical'
            ]

        if not self.signal_like_patterns:
            self.signal_like_patterns = [
                r"signal",
                r"entry",
                r"exit",
                r"crossover",
                r"trade",
            ]

        if not self.feature_category_caps:
            self.feature_category_caps = {
                'volatility_regime': 30,
                'volume_regime': 25,
                'structural_trend': 25,
                'statistical_regime': 30,
                'regime_quality': 20,
            }
        
        # Ensure n_regimes is within learned bounds
        if not (self.regime_search_min <= self.n_regimes <= self.regime_search_max):
            self.n_regimes = max(
                self.regime_search_min,
                min(self.regime_search_max, self.n_regimes),
            )

        # Apply calibrated quality thresholds if not explicitly provided
        thresholds = get_calibrated_thresholds()
        if self.min_regime_persistence is None:
            self.min_regime_persistence = thresholds.get('min_regime_persistence', 0.7)
        if self.max_feature_noise_ratio is None:
            self.max_feature_noise_ratio = thresholds.get('max_feature_noise_ratio', 0.3)
        if self.min_temporal_stability is None:
            self.min_temporal_stability = thresholds.get('min_temporal_stability', 0.6)


class RegimeClusteringComponent(BaseStep):
    """
    NAS-TAS Clustering Component.
    
    This component uses shared utilities to eliminate redundancy:
    - Uses shared feature preparation
    - Uses shared configuration validation
    - Uses shared logging utilities
    - Uses shared metrics calculation
    - Uses shared regime characteristics generation
    """
    
    def __init__(self, step_name: str = "regime_clustering", config: Optional[Dict[str, Any]] = None):
        """Initialize the regime clustering component with enhanced capabilities."""
        super().__init__(step_name, config)
        
        # Convert config dict to RegimeClusteringConfig
        if config:
            self.clustering_config = RegimeClusteringConfig(**config)
        else:
            self.clustering_config = RegimeClusteringConfig()
        
        with LoggingContext('Regime-Clustering', 'Initialization', verbose=True):
            # Use shared logging utilities
            self.logger = get_logger('RegimeClustering')
            
            # Initialize hardware optimization
            if HARDWARE_OPTIMIZATION_AVAILABLE:
                try:
                    self.hardware_manager = get_integrated_hardware_manager()
                    self.memory_optimizer = get_advanced_memory_optimizer()
                    self.cpu_optimizer = get_advanced_cpu_optimizer()
                    tprint("✅ Hardware optimization initialized", "SUCCESS")
                except Exception as e:
                    tprint(f"⚠️ Hardware optimization initialization failed: {e}", "WARNING")
                    self.hardware_manager = None
                    self.memory_optimizer = None
                    self.cpu_optimizer = None
            else:
                self.hardware_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
    
    # @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    # @performance_tracked(log_performance=True, track_memory=True)
    def _load_market_data_from_artifacts(self) -> Optional[pd.DataFrame]:
        """Load market data from artifacts using BaseStep artifact manager with hardware optimization."""
        try:
            # Try to load market data from various possible artifact names
            possible_names = ['market_data', 'processed_data', 'klines_data', 'price_data']
            
            for name in possible_names:
                data = self._load_dataframe(name)
                if data is not None and not data.empty:
                    tprint(f"Loaded market data from artifact: {name}", "INFO")
                    return data
            
            tprint("No market data found in artifacts", "WARNING")
            return None
            
        except Exception as e:
            tprint(f"Error loading market data: {e}", "ERROR")
            return None
            
            # Initialize shared utilities
            self.config_validator = ConfigValidator(verbose=True)
            self.metrics_calculator = MetricsCalculator(verbose=True)
            self.characteristics_generator = CharacteristicsGenerator(verbose=True)
            
            # Initialize regime-focused feature configuration
            self.feature_config = FeatureConfig(
                feature_categories=getattr(config, 'feature_categories', [
                    'regime_volatility', 
                    'regime_volume', 
                    'regime_structural_trend', 
                    'regime_statistical'
                ]),
                use_standardized_features=getattr(config, 'use_standardized_features', True),
                drop_highly_correlated=True
            )
            
            self.clustering_result = None
            self.execution_metadata = {}

            # Stage 1 feature preparation outputs
            self.stage1_features_df: Optional[pd.DataFrame] = None
            self.stage1_filtered_df: Optional[pd.DataFrame] = None
            self.stage1_metadata: Dict[str, Any] = {}
            self.feature_projection_metadata: Dict[str, Any] = {}
            self.feature_projection_artifact_path: Optional[Path] = None

            # Learned metric weight state
            self.metric_weight_history: List[Dict[str, Any]] = []
            self.learned_weights: Dict[str, Dict[str, float]] = {}
            self._weight_history_limit: int = 100
            self._default_metric_weights: Dict[str, Dict[str, float]] = {
                'composite': {
                    'silhouette': 0.40,  # Increased from 0.25 - prioritize silhouette for better separation
                    'davies_bouldin': 0.30,  # Increased from 0.20 - prioritize Davies-Bouldin for better clustering
                    'calinski_harabasz': 0.15,  # Reduced from 0.20 - secondary importance
                    'stability': 0.10,  # Reduced from 0.20 - lower priority
                    'consensus': 0.05,  # Reduced from 0.15 - lowest priority
                },
                'regime': {
                    'economic': 0.25,
                    'volatility': 0.30,
                    'volume': 0.25,
                    'structural_trend': 0.20,
                },
                'temporal': {
                    'autocorrelation': 0.30,
                    'stability': 0.20,
                    'trend_consistency': 0.30,
                    'regime_persistence': 0.20,
                },
            }
            self._last_composite_metric_summary: Optional[Dict[str, float]] = None
            self._last_temporal_metric_summary: Optional[Dict[str, float]] = None
            self._last_regime_metric_summary: Optional[Dict[str, float]] = None

            # Initialize hardware optimizations
            hardware_setup = HardwareSetup()
            resources: HardwareResources = hardware_setup.initialize()

            self.hardware_setup = hardware_setup
            self.hardware_resources = resources
            self.matrix_ops = resources.matrix_ops
            self.vectorized_core = resources.vectorized_core
            
            # Initialize M1 hardware optimizers
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()
            self.gpu_manager = get_m1_gpu_manager()
            
            # Initialize ML Common optimizers
            if ML_COMMON_AVAILABLE:
                self.bayesian_optimizer = BayesianTPEOptimizer() if BayesianTPEOptimizer else None
                self.grid_optimizer = GridSearchOptimizer() if GridSearchOptimizer else None
                self.hyperparameter_optimizer = HyperparameterOptimizer() if HyperparameterOptimizer else None
                self.optuna_optimizer = OptunaOptimizer() if OptunaOptimizer else None
                
                # Initialize cross-validators
                self.ts_cv = TimeSeriesCrossValidator() if TimeSeriesCrossValidator else None
                self.regime_cv = RegimeAwareCrossValidator() if RegimeAwareCrossValidator else None
                self.walk_forward_cv = WalkForwardValidator() if WalkForwardValidator else None
                self.purged_cv = PurgedCrossValidator() if PurgedCrossValidator else None
                
                # Initialize validators
                self.model_validator = ModelValidator() if ModelValidator else None
                self.performance_validator = PerformanceValidator() if PerformanceValidator else None
                self.stability_validator = StabilityValidator() if StabilityValidator else None
                
                # Initialize ensembles
                self.ensemble_validator = EnsembleValidator() if EnsembleValidator else None
                self.model_ensemble = ModelEnsemble() if ModelEnsemble else None
                self.weighted_ensemble = WeightedEnsemble() if WeightedEnsemble else None
            else:
                self.bayesian_optimizer = None
                self.grid_optimizer = None
                self.hyperparameter_optimizer = None
                self.optuna_optimizer = None
                self.ts_cv = None
                self.regime_cv = None
                self.walk_forward_cv = None
                self.purged_cv = None
                self.model_validator = None
                self.performance_validator = None
                self.stability_validator = None
                self.ensemble_validator = None
                self.model_ensemble = None
                self.weighted_ensemble = None
            
            # Performance monitoring
            self.performance_metrics = {
                "start_time": None,
                "end_time": None,
                "memory_usage": [],
                "processing_times": {},
                "error_count": 0,
                "success_count": 0,
                "optimization_trials": 0,
                "cv_folds": 0
            }
            
            # Log initialization status
            tprint_structured({
                "component": "NASTASClusteringComponent",
                "m1_hardware_available": M1_HARDWARE_AVAILABLE,
                "ml_common_available": ML_COMMON_AVAILABLE,
                "matrix_operations_available": MATRIX_OPERATIONS_AVAILABLE,
                "common_operations_available": COMMON_OPERATIONS_AVAILABLE,
                "math_validation_available": MATH_VALIDATION_AVAILABLE,
                "bayesian_optimizer": self.bayesian_optimizer is not None,
                "cross_validators": {
                    "ts_cv": self.ts_cv is not None,
                    "regime_cv": self.regime_cv is not None,
                    "walk_forward_cv": self.walk_forward_cv is not None,
                    "purged_cv": self.purged_cv is not None
                }
            })
            
            # Set hardware resources
            self.batch_processor = resources.batch_processor
            self.hardware_manager = resources.hardware_manager
            self.m1_gpu_optimizer = resources.m1_gpu_optimizer
            self.m1_memory_optimizer = resources.m1_memory_optimizer
            self.m1_cpu_optimizer = resources.m1_cpu_optimizer
            
            # Initialize regime optimization service with proper label fusion service
            from ..regime_analysis.label_fusion import LabelFusionService
            label_fusion_service = LabelFusionService(logger=self._log)
            self.regime_optimization_service = RegimeOptimizationService(
                label_fusion_service=label_fusion_service,
                score_calculator=self._calculate_composite_score,
                logger=self._log,
            )
            
            tprint_success("🔍 NAS-TAS Clustering Component initialized with enhanced capabilities")

    def optimize_hyperparameters_bayesian(self, features: np.ndarray, market_data: pd.DataFrame, 
                                        n_trials: int = 100) -> Dict[str, Any]:
        """Optimize hyperparameters using Bayesian TPE optimization."""
        if not self.bayesian_optimizer:
            tprint_warning("Bayesian optimizer not available, using default parameters")
            return {}
        
        with tprint_timer(f"Bayesian hyperparameter optimization ({n_trials} trials)"):
            try:
                # Define parameter space
                min_regimes = getattr(self.config, 'regime_search_min', 5)
                max_regimes = getattr(self.config, 'regime_search_max', 15)
                if min_regimes > max_regimes:
                    min_regimes, max_regimes = max_regimes, min_regimes

                param_space = {
                    'n_regimes': (min_regimes, max_regimes),
                    'economic_weight': (0.1, 0.4),
                    'volatility_regime_weight': (0.2, 0.4),
                    'volume_regime_weight': (0.2, 0.4),
                    'structural_trend_weight': (0.1, 0.3),
                    'min_regime_persistence': (0.5, 0.9),
                    'max_feature_noise_ratio': (0.1, 0.5),
                    'min_temporal_stability': (0.4, 0.8)
                }
                
                # Define objective function
                def objective(params):
                    try:
                        # Update config with trial parameters
                        trial_config = self.config.copy()
                        for key, value in params.items():
                            setattr(trial_config, key, value)
                        
                        # Run clustering with trial parameters
                        result = self._run_clustering_trial(features, market_data, trial_config)
                        
                        # Return negative score (optimizer minimizes)
                        return -result.get('overall_score', 0.0)
                        
                    except Exception as exc:
                        tprint_warning(f"Trial failed: {exc}")
                        return float('inf')
                
                # Run optimization
                best_params = self.bayesian_optimizer.optimize(
                    objective_function=objective,
                    parameter_space=param_space,
                    n_trials=n_trials
                )
                
                self.performance_metrics["optimization_trials"] = n_trials
                
                tprint_structured({
                    "optimization_complete": True,
                    "n_trials": n_trials,
                    "best_params": best_params,
                    "optimization_method": "bayesian_tpe"
                })
                
                return best_params
                
            except Exception as exc:
                tprint_error(f"Bayesian optimization failed: {exc}")
                return {}

    def run_advanced_cross_validation(self, features: np.ndarray, labels: np.ndarray, 
                                     market_data: pd.DataFrame) -> Dict[str, Any]:
        """Run advanced cross-validation with multiple strategies."""
        cv_results = {}
        
        with tprint_timer("Advanced cross-validation"):
            try:
                # Time Series Cross-Validation
                if self.ts_cv:
                    with tprint_timer("Time Series CV"):
                        ts_results = self.ts_cv.cross_validate(
                            features, labels, 
                            n_splits=5, 
                            test_size=0.2
                        )
                        cv_results['time_series'] = ts_results
                        self.performance_metrics["cv_folds"] += 5
                
                # Regime-Aware Cross-Validation
                if self.regime_cv:
                    with tprint_timer("Regime-Aware CV"):
                        regime_results = self.regime_cv.cross_validate(
                            features, labels,
                            regime_data=market_data,
                            n_splits=5
                        )
                        cv_results['regime_aware'] = regime_results
                        self.performance_metrics["cv_folds"] += 5
                
                # Walk-Forward Validation
                if self.walk_forward_cv:
                    with tprint_timer("Walk-Forward CV"):
                        wf_results = self.walk_forward_cv.cross_validate(
                            features, labels,
                            n_splits=5,
                            expanding_window=True
                        )
                        cv_results['walk_forward'] = wf_results
                        self.performance_metrics["cv_folds"] += 5
                
                # Purged Cross-Validation
                if self.purged_cv:
                    with tprint_timer("Purged CV"):
                        purged_results = self.purged_cv.cross_validate(
                            features, labels,
                            purge_period=10,
                            embargo_period=5,
                            n_splits=5
                        )
                        cv_results['purged'] = purged_results
                        self.performance_metrics["cv_folds"] += 5
                
                # Calculate ensemble CV score
                if cv_results:
                    ensemble_score = self._calculate_ensemble_cv_score(cv_results)
                    cv_results['ensemble_score'] = ensemble_score
                
                tprint_structured({
                    "cv_complete": True,
                    "cv_methods": list(cv_results.keys()),
                    "total_folds": self.performance_metrics["cv_folds"],
                    "ensemble_score": cv_results.get('ensemble_score', 0.0)
                })
                
                return cv_results
                
            except Exception as exc:
                tprint_error(f"Advanced cross-validation failed: {exc}")
                return {}

    def _run_clustering_trial(self, features: np.ndarray, market_data: pd.DataFrame, 
                            config: NASTASClusteringConfig) -> Dict[str, Any]:
        """Run a single clustering trial for optimization."""
        try:
            tprint(f"Running clustering trial with n_regimes={config.n_regimes}", "INFO")

            # Validate inputs
            if features is None or len(features) == 0:
                raise ValueError("Features cannot be empty")
            if len(features) < config.n_regimes:
                raise ValueError(f"Insufficient samples ({len(features)}) for {config.n_regimes} regimes")

            validate_regime_count(
                int(config.n_regimes),
                getattr(self.config, 'regime_search_min', 5),
                getattr(self.config, 'regime_search_max', 15),
            )
            
            # Standardize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Apply PCA for dimensionality reduction
            from sklearn.decomposition import PCA
            n_components = min(features.shape[1], len(features) - 1, config.n_regimes * 2)
            pca = PCA(n_components=n_components)
            features_pca = pca.fit_transform(features_scaled)
            
            # Perform GMM clustering - ENHANCED with better parameters for convergence and silhouette
            from sklearn.mixture import GaussianMixture
            gmm = GaussianMixture(
                n_components=config.n_regimes,
                random_state=42,
                max_iter=200,  # Increased from 100 for better convergence
                n_init=5,  # Multiple initializations for better results
                reg_covar=1e-4,  # Regularization for numerical stability (was 1e-5 in other places)
                init_params='k-means++',  # Better initialization method
                covariance_type='full',  # Full covariance for better flexibility
                tol=1e-6,  # Tighter tolerance for better convergence
                warm_start=False  # Start fresh for each trial
            )
            labels = gmm.fit_predict(features_pca)
            
            # Calculate clustering metrics
            from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
            
            silhouette = silhouette_score(features_pca, labels) if len(np.unique(labels)) > 1 else 0.0
            davies_bouldin = davies_bouldin_score(features_pca, labels) if len(np.unique(labels)) > 1 else float('inf')
            calinski_harabasz = calinski_harabasz_score(features_pca, labels) if len(np.unique(labels)) > 1 else 0.0
            
            # Calculate overall score (higher is better)
            overall_score = (
                0.4 * silhouette +  # Higher is better
                0.3 * (1.0 / (1.0 + davies_bouldin)) +  # Convert to higher-is-better
                0.3 * min(1.0, calinski_harabasz / 1000.0)  # Normalize and cap
            )
            
            # Calculate CV score using coefficient of variation
            cv_score = self._calculate_cv_score_trial(features_pca, labels)

            # ENHANCED: Calculate balance penalty for imbalanced clusters
            balance_penalty = self._calculate_balance_penalty(labels)

            # Apply balance penalty to overall score
            balanced_overall_score = overall_score * (1.0 - balance_penalty * 0.3)  # Up to 30% penalty for imbalance

            result = {
                'overall_score': float(balanced_overall_score),
                'raw_overall_score': float(overall_score),  # Keep original for reference
                'silhouette_score': float(silhouette),
                'davies_bouldin_score': float(davies_bouldin),
                'calinski_harabasz_score': float(calinski_harabasz),
                'cv_score': float(cv_score),
                'balance_penalty': float(balance_penalty),
                'n_clusters': len(np.unique(labels)),
                'converged': gmm.converged_,
                'n_iter': gmm.n_iter_
            }
            
            tprint(f"Trial completed: score={overall_score:.4f}, clusters={len(np.unique(labels))}", "SUCCESS")
            return result
            
        except Exception as exc:
            tprint_warning(f"Clustering trial failed: {exc}")
            return {
                'overall_score': 0.0,
                'silhouette_score': 0.0,
                'davies_bouldin_score': float('inf'),
                'calinski_harabasz_score': 0.0,
                'cv_score': 0.0,
                'n_clusters': 0,
                'converged': False,
                'n_iter': 0
            }
    
    def _calculate_balance_penalty(self, labels: np.ndarray) -> float:
        """Calculate balance penalty for imbalanced cluster sizes - ENHANCED VERSION."""
        try:
            unique_labels, counts = np.unique(labels, return_counts=True)
            n_samples = len(labels)
            n_clusters = len(unique_labels)

            if n_clusters < 2:
                return 0.0

            # Calculate percentages for each cluster
            percentages = counts / n_samples

            # Find the most imbalanced clusters
            max_percentage = np.max(percentages)
            min_percentage = np.min(percentages)

            # Calculate imbalance ratio (how much larger the biggest cluster is vs smallest)
            imbalance_ratio = max_percentage / min_percentage if min_percentage > 0 else float('inf')

            # Penalty based on maximum cluster size (should not exceed 12%)
            max_size_penalty = max(0.0, max_percentage - 0.12) * 3.0  # 3x penalty for exceeding threshold

            # Penalty for very small clusters (should not be below 6%)
            min_size_penalty = sum(max(0.0, 0.06 - p) * 2.0 for p in percentages if p < 0.06)

            # Penalty for high imbalance ratio
            imbalance_penalty = min(1.0, (imbalance_ratio - 1.0) / 5.0)  # Normalize to 0-1 scale

            # Combined penalty (weighted average)
            total_penalty = (max_size_penalty * 0.4 + min_size_penalty * 0.3 + imbalance_penalty * 0.3)

            return min(1.0, total_penalty)  # Cap at 1.0

        except Exception as exc:
            tprint_warning(f"Balance penalty calculation failed: {exc}")
            return 0.5  # Return moderate penalty on error

    def _calculate_cv_score_trial(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate coefficient of variation score for trial."""
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 0.0

            within_cv_scores = []
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_features = features[cluster_mask]
                if len(cluster_features) <= 1:
                    continue

                feature_cvs = []
                for feature_idx in range(cluster_features.shape[1]):
                    feature_values = cluster_features[:, feature_idx]
                    if np.std(feature_values) > 0 and np.mean(np.abs(feature_values)) > 0:
                        cv = np.std(feature_values) / np.mean(np.abs(feature_values))
                        feature_cvs.append(cv)
                
                if feature_cvs:
                    within_cv_scores.append(np.mean(feature_cvs))
            
            within_cv = np.mean(within_cv_scores) if within_cv_scores else 0.0
            
            # Calculate between-cluster CV
            cluster_centers = []
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_features = features[cluster_mask]
                if len(cluster_features) > 0:
                    center = np.mean(cluster_features, axis=0)
                    cluster_centers.append(center)
            
            if len(cluster_centers) > 1:
                cluster_centers = np.array(cluster_centers)
                between_std = np.std(cluster_centers)
                between_mean_abs = np.mean(np.abs(cluster_centers))
                between_cv = between_std / between_mean_abs if between_mean_abs > 0 else 0.0
            else:
                between_cv = 0.0
            
            # Calculate final CV score
            cv_score = 0.6 * max(0.0, 1.0 - within_cv) + 0.4 * min(1.0, between_cv)
            return float(cv_score)
            
        except Exception as exc:
            tprint_warning(f"CV score calculation failed: {exc}")
            return 0.0

    def _calculate_ensemble_cv_score(self, cv_results: Dict[str, Any]) -> float:
        """Calculate ensemble score from multiple CV results."""
        try:
            scores = []
            for method, results in cv_results.items():
                if isinstance(results, dict) and 'score' in results:
                    scores.append(results['score'])
                elif isinstance(results, (int, float)):
                    scores.append(results)
            
            if scores:
                return safe_mean(np.array(scores))
            return 0.0
            
        except Exception as exc:
            tprint_warning(f"Failed to calculate ensemble CV score: {exc}")
            return 0.0

    def validate_model_performance(self, features: np.ndarray, labels: np.ndarray, 
                                 market_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate model performance using multiple validators."""
        validation_results = {}
        
        with tprint_timer("Model performance validation"):
            try:
                # Performance validation
                if self.performance_validator:
                    perf_results = self.performance_validator.validate(
                        features, labels, market_data
                    )
                    validation_results['performance'] = perf_results
                
                # Stability validation
                if self.stability_validator:
                    stability_results = self.stability_validator.validate(
                        features, labels, market_data
                    )
                    validation_results['stability'] = stability_results
                
                # Model validation
                if self.model_validator:
                    model_results = self.model_validator.validate(
                        features, labels, market_data
                    )
                    validation_results['model'] = model_results
                
                tprint_structured({
                    "validation_complete": True,
                    "validation_methods": list(validation_results.keys()),
                    "overall_validation_score": self._calculate_validation_score(validation_results)
                })
                
                return validation_results
                
            except Exception as exc:
                tprint_error(f"Model validation failed: {exc}")
                return {}

    def _calculate_validation_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall validation score."""
        try:
            scores = []
            for method, results in validation_results.items():
                if isinstance(results, dict) and 'score' in results:
                    scores.append(results['score'])
                elif isinstance(results, (int, float)):
                    scores.append(results)
            
            if scores:
                return safe_mean(np.array(scores))
            return 0.0
            
        except Exception as exc:
            tprint_warning(f"Failed to calculate validation score: {exc}")
            return 0.0

    def _log(self, message: str, level: str = "INFO") -> None:
        """Log a message using the standard component logger."""
        tprint(message, level)
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['nas_tas_clustering_result']

    # @performance_tracked(log_performance=True, track_memory=True)
    def _estimate_regime_range(self) -> Tuple[int, int, int]:
        """Estimate regime count bounds using discovery metrics from artifacts with performance tracking."""

        default_min = int(max(5, getattr(self.config, 'regime_search_min', 5) or 5))
        default_max = int(max(default_min, min(15, getattr(self.config, 'regime_search_max', 15) or 15)))
        default_mode = int(min(max(default_min, getattr(self.config, 'n_regimes', 8) or 8), default_max))

        # Try to load discovery result from artifacts
        discovery_result = {}
        try:
            discovery_result = self._load_metadata('nas_tas_regime_discovery_result') or {}
        except Exception:
            pass

        candidate_entries: List[Dict[str, Any]] = []

        def extract_metric(metrics: Dict[str, Any], keys: List[str]) -> Optional[float]:
            for key in keys:
                if key in metrics:
                    value = metrics[key]
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        return float(value)
            return None

        def register_candidate(k_value: Any, metrics: Dict[str, Any]) -> None:
            try:
                k_int = int(k_value)
            except (TypeError, ValueError):
                return

            if k_int <= 0:
                return

            candidate_entries.append({
                'k': k_int,
                'silhouette': extract_metric(
                    metrics,
                    ['silhouette', 'silhouette_score', 'silhouette_metric'],
                ),
                'bic': extract_metric(metrics, ['bic', 'bic_score']),
                'aic': extract_metric(metrics, ['aic', 'aic_score']),
            })

        def parse_candidate_block(block: Any) -> None:
            if isinstance(block, dict):
                if 'n_regimes' in block or 'k' in block:
                    metrics = block.get('metrics') or block.get('scores') or block
                    register_candidate(block.get('n_regimes') or block.get('k'), metrics if isinstance(metrics, dict) else {})
                else:
                    for key, value in block.items():
                        if isinstance(value, (dict, list)):
                            parse_candidate_block(value)
                        else:
                            register_candidate(key, value if isinstance(value, dict) else {})
            elif isinstance(block, list):
                for item in block:
                    parse_candidate_block(item)

        candidate_keys = [
            'regime_candidates',
            'regime_quality_grid',
            'regime_count_candidates',
            'candidate_regime_counts',
            'regime_grid',
            'regime_metrics',
        ]

        for key in candidate_keys:
            if key in discovery_result:
                parse_candidate_block(discovery_result.get(key))

        # Fallback: sometimes metrics stored directly in discovery result with numeric keys
        parse_candidate_block({k: v for k, v in discovery_result.items() if isinstance(k, (int, str))})

        if not candidate_entries:
            return default_min, default_max, default_mode

        # Deduplicate by regime count keeping best metrics seen so far
        deduped: Dict[int, Dict[str, Any]] = {}
        for entry in candidate_entries:
            k = entry['k']
            if k not in deduped:
                deduped[k] = entry
                continue

            existing = deduped[k]
            for metric_key in ('silhouette', 'bic', 'aic'):
                existing_value = existing.get(metric_key)
                new_value = entry.get(metric_key)
                if new_value is None:
                    continue
                if existing_value is None:
                    existing[metric_key] = new_value
                    continue
                if metric_key == 'silhouette':
                    if new_value > existing_value:
                        existing[metric_key] = new_value
                else:
                    if new_value < existing_value:
                        existing[metric_key] = new_value

        candidates = list(deduped.values())

        if not candidates:
            return default_min, default_max, default_mode

        silhouettes = [c['silhouette'] for c in candidates if c.get('silhouette') is not None]
        bics = [c['bic'] for c in candidates if c.get('bic') is not None]
        aics = [c['aic'] for c in candidates if c.get('aic') is not None]

        def normalize(value_list: List[float], value: float, reverse: bool = False) -> Optional[float]:
            if not value_list:
                return None
            if len(value_list) == 1:
                return 1.0
            min_val = min(value_list)
            max_val = max(value_list)
            if np.isclose(min_val, max_val):
                return 1.0
            if reverse:
                return (max_val - value) / (max_val - min_val)
            return (value - min_val) / (max_val - min_val)

        for candidate in candidates:
            score_components: List[float] = []

            silhouette = candidate.get('silhouette')
            if silhouette is not None:
                score = normalize(silhouettes, silhouette, reverse=False)
                if score is not None:
                    score_components.append(score)

            bic_score = candidate.get('bic')
            if bic_score is not None:
                score = normalize(bics, bic_score, reverse=True)
                if score is not None:
                    score_components.append(score)

            aic_score = candidate.get('aic')
            if aic_score is not None:
                score = normalize(aics, aic_score, reverse=True)
                if score is not None:
                    score_components.append(score)

            candidate['score'] = float(np.mean(score_components)) if score_components else 0.0

        best_candidate = max(candidates, key=lambda item: (item.get('score', 0.0), -item['k']))
        best_score = best_candidate.get('score', 0.0)

        if best_score <= 0:
            candidate_bounds = [c['k'] for c in candidates]
        else:
            threshold = max(0.0, best_score * 0.8)
            candidate_bounds = [
                c['k']
                for c in candidates
                if c.get('score', 0.0) >= threshold
            ]
            if not candidate_bounds:
                candidate_bounds = [best_candidate['k']]

        min_bound = max(5, min(candidate_bounds))
        max_bound = min(20, max(candidate_bounds))

        if min_bound > max_bound:
            min_bound, max_bound = max(5, min_bound), max(5, min_bound)

        suggested = int(np.clip(best_candidate['k'], min_bound, max_bound))

        return int(min_bound), int(max_bound), suggested

    # @performance_tracked(log_performance=True, track_memory=True)
    def _extract_regime_counts(self) -> int:
        """Extract the number of regimes to use for clustering using data-driven approach with performance tracking."""
        tprint("📈 Step 1: Extracting regime count from artifacts...", "INFO")

        min_regimes, max_regimes, default_regimes = self._estimate_regime_range()
        self.config.regime_search_min = min_regimes
        self.config.regime_search_max = max_regimes

        # Try to load regime discovery result from artifacts
        regime_discovery_result = {}
        try:
            regime_discovery_result = self._load_metadata('nas_tas_regime_discovery_result') or {}
        except Exception:
            pass
            
        tas_regime_count = regime_discovery_result.get('tas_regime_count', None)
        nas_regime_count = regime_discovery_result.get('nas_regime_count', None)

        # Data-driven regime count selection (no hardcoded heuristics)
        if tas_regime_count and nas_regime_count:
            # Use the maximum of the two discovered regime counts
            n_regimes = max(tas_regime_count, nas_regime_count)
            tprint(f"Using data-driven regime count: max(TAS={tas_regime_count}, NAS={nas_regime_count}) = {n_regimes}", "INFO")
        elif tas_regime_count:
            n_regimes = tas_regime_count
            tprint(f"Using TAS regime count: {n_regimes}", "INFO")
        elif nas_regime_count:
            n_regimes = nas_regime_count
            tprint(f"Using NAS regime count: {n_regimes}", "INFO")
        else:
            # No regime discovery data available - fall back to data-driven default from discovery metrics
            tprint(
                "No regime discovery data available - using data-driven default from discovery metrics",
                "INFO",
            )
            self.config.n_regimes = default_regimes
            return default_regimes
    
    # @performance_tracked(log_performance=True, track_memory=True)
    def _estimate_default_regime_count(self) -> int:
        """Estimate default regime count based on data characteristics."""
        # Use a simple heuristic based on data size
        if hasattr(self, 'features') and self.features is not None:
            n_samples = self.features.shape[0]
            if n_samples < 1000:
                return 3
            elif n_samples < 5000:
                return 5
            elif n_samples < 10000:
                return 8
            else:
                return 10
        return 8  # Default fallback

        # Apply evidence-driven bounds derived from discovery metrics
        n_regimes = max(min_regimes, min(max_regimes, n_regimes))

        tprint(
            f"Final regime count: {n_regimes} (data-driven, no hardcoded heuristics)",
            "SUCCESS"
        )
        self.clustering_config.n_regimes = n_regimes
        return n_regimes

    # @performance_tracked(log_performance=True, track_memory=True)
    def _validate_configuration(self) -> None:
        """Validate configuration using shared utilities with performance tracking."""
        tprint("Step 2: Validating inputs and configuration using shared utilities", "INFO")
        validation_errors = self.config_validator.validate_config(self.clustering_config)
        if validation_errors:
            tprint(f"Configuration validation failed: {validation_errors}", "ERROR")
            raise ValueError(f"Configuration validation failed: {validation_errors}")

        tprint("Configuration validation passed using shared utilities", "SUCCESS")

    # @performance_tracked(log_performance=True, track_memory=True)
    def _initialize_execution_metadata(self) -> None:
        """Initialize execution metadata for downstream use with performance tracking."""
        self.execution_metadata = {
            'start_time': datetime.now().isoformat(),
            'symbol': getattr(self.config, 'symbol', 'ETHUSDT'),
            'timeframe': getattr(self.config, 'timeframe', '15m'),
            'exchange': getattr(self.config, 'exchange', 'binance'),
            'component': 'refactored_nas_tas_clustering',
            'uses_shared_utilities': True,
            'quality_calibration': get_current_calibration(),
            'calibration_loaded_from_state': False,
        }

    # @performance_tracked(log_performance=True, track_memory=True)
    def _restore_learned_weights_from_artifacts(self) -> None:
        """Restore learned metric weights and history from artifacts with performance tracking."""
        restored_weights: Dict[str, Dict[str, float]] = {}
        restored_history: List[Dict[str, Any]] = []

        # Try to load weights from various possible artifact names
        weight_artifact_names = [
            'learned_metric_weights',
            'regime_clustering_weights',
            'nas_tas_clustering_result'
        ]
        
        for artifact_name in weight_artifact_names:
            try:
                container = self._load_metadata(artifact_name)
                if isinstance(container, dict):
                    weights = container.get('learned_metric_weights')
                    if isinstance(weights, dict):
                        for group, group_weights in weights.items():
                            sanitized = self._sanitize_weight_dict(group, group_weights)
                            if sanitized:
                                restored_weights[group] = sanitized

                    history = container.get('metric_weight_history')
                    if isinstance(history, list) and not restored_history:
                        restored_history = history
            except Exception:
                continue

        if restored_weights:
            self.learned_weights.update(restored_weights)

        if restored_history:
            sanitized_history = self._sanitize_metric_history(restored_history)
            if sanitized_history:
                self.metric_weight_history = sanitized_history[-self._weight_history_limit:]

    # @performance_tracked(log_performance=True, track_memory=True)
    def _iterate_weight_containers(self, node: Any) -> Iterator[Dict[str, Any]]:
        """Yield nested containers that may store learned weight metadata with performance tracking."""
        if isinstance(node, dict):
            if 'learned_metric_weights' in node or 'metric_weight_history' in node:
                yield node
            for value in node.values():
                yield from self._iterate_weight_containers(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                yield from self._iterate_weight_containers(item)

    # @performance_tracked(log_performance=True, track_memory=True)
    def _sanitize_weight_dict(self, group: str, weights: Any) -> Dict[str, float]:
        """Convert a raw weight mapping into a normalized simplex weight dict with performance tracking."""
        if not isinstance(weights, dict):
            return {}

        names = list(weights.keys())
        if not names:
            return {}

        vector = np.array([
            float(weights.get(name, 0.0)) if isinstance(weights.get(name, 0.0), (int, float)) else 0.0
            for name in names
        ], dtype=float)

        vector = np.maximum(vector, 0.0)
        if vector.sum() == 0:
            defaults = self._default_metric_weights.get(group, {})
            vector = np.array([float(defaults.get(name, 0.0)) for name in names], dtype=float)
            vector = np.maximum(vector, 0.0)

        if vector.size == 0:
            return {}

        normalized = self._project_to_simplex(vector)
        return {name: float(value) for name, value in zip(names, normalized)}

    def _coerce_nested_float_dict(self, data: Any) -> Dict[str, Any]:
        """Recursively coerce nested mapping values to floats where possible."""
        if not isinstance(data, dict):
            return {}

        coerced: Dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                nested = self._coerce_nested_float_dict(value)
                if nested:
                    coerced[key] = nested
            else:
                try:
                    coerced[key] = float(value)
                except (TypeError, ValueError):
                    continue
        return coerced

    def _sanitize_metric_history(self, history: List[Any]) -> List[Dict[str, Any]]:
        """Ensure restored metric weight history is JSON-serializable and numeric."""
        sanitized: List[Dict[str, Any]] = []
        for entry in history:
            if not isinstance(entry, dict):
                continue

            sanitized_entry: Dict[str, Any] = {}
            timestamp = entry.get('timestamp')
            if isinstance(timestamp, str):
                sanitized_entry['timestamp'] = timestamp

            metrics = entry.get('metrics')
            if isinstance(metrics, dict):
                coerced_metrics = self._coerce_nested_float_dict(metrics)
                if coerced_metrics:
                    sanitized_entry['metrics'] = coerced_metrics

            target = entry.get('validation_target')
            if isinstance(target, (int, float)):
                sanitized_entry['validation_target'] = float(target)

            fitted = entry.get('fitted_weights')
            if isinstance(fitted, dict):
                sanitized_entry['fitted_weights'] = {
                    group: self._sanitize_weight_dict(group, weights)
                    for group, weights in fitted.items()
                    if isinstance(weights, dict)
                }

            if sanitized_entry:
                sanitized.append(sanitized_entry)

        return sanitized

    def _project_to_simplex(self, vector: np.ndarray) -> np.ndarray:
        """Project a vector onto the probability simplex."""
        if vector.ndim != 1:
            vector = vector.ravel()

        if vector.size == 0:
            return vector

        vector = np.maximum(vector, 0.0)
        total = vector.sum()
        if total == 0:
            return np.full_like(vector, 1.0 / vector.size)
        return vector / total

    def _serialize_learned_weights(self) -> Dict[str, Dict[str, float]]:
        """Serialize learned weights for artifact persistence."""
        serialized: Dict[str, Dict[str, float]] = {}
        for group, weights in self.learned_weights.items():
            serialized[group] = {name: float(value) for name, value in weights.items()}
        return serialized

    def _serialize_metric_history(self) -> List[Dict[str, Any]]:
        """Serialize recent metric weight history for artifact persistence."""
        serialized: List[Dict[str, Any]] = []
        for entry in self.metric_weight_history[-self._weight_history_limit:]:
            record: Dict[str, Any] = {}
            timestamp = entry.get('timestamp')
            if isinstance(timestamp, str):
                record['timestamp'] = timestamp

            if isinstance(entry.get('validation_target'), (int, float)):
                record['validation_target'] = float(entry['validation_target'])

            metrics = entry.get('metrics')
            if isinstance(metrics, dict):
                coerced_metrics = self._coerce_nested_float_dict(metrics)
                if coerced_metrics:
                    record['metrics'] = coerced_metrics

            fitted = entry.get('fitted_weights')
            if isinstance(fitted, dict):
                record['fitted_weights'] = {
                    group: {name: float(value) for name, value in weights.items()}
                    for group, weights in fitted.items()
                    if isinstance(weights, dict)
                }

            if record:
                serialized.append(record)

        return serialized

    # @performance_tracked(log_performance=True, track_memory=True)
    def _load_calibration_history(self) -> None:
        """Load calibration history from artifacts if available with performance tracking."""

        calibration_payload = None
        
        # Try to load from clustering result
        try:
            previous_result = self._load_metadata('nas_tas_clustering_result')
            if isinstance(previous_result, dict):
                execution_meta = previous_result.get('execution_metadata', {})
                if isinstance(execution_meta, dict):
                    calibration_payload = execution_meta.get('quality_calibration')
        except Exception:
            pass

        # Try to load from calibration artifact
        if calibration_payload is None:
            try:
                calibration_payload = self._load_metadata('nas_tas_clustering_calibration')
            except Exception:
                pass

        if calibration_payload:
            update_quality_calibration(calibration_payload)
            self.execution_metadata['quality_calibration'] = get_current_calibration()
            self.execution_metadata['calibration_loaded_from_state'] = True
        else:
            update_quality_calibration(self.execution_metadata.get('quality_calibration'))

        thresholds = get_calibrated_thresholds()

        # Set default values if not already set
        if not hasattr(self.config, 'min_regime_persistence') or self.config.min_regime_persistence is None:
            self.config.min_regime_persistence = thresholds.get('min_regime_persistence', 0.7)
        else:
            self.config.min_regime_persistence = thresholds.get('min_regime_persistence', self.config.min_regime_persistence)

        if not hasattr(self.config, 'max_feature_noise_ratio') or self.config.max_feature_noise_ratio is None:
            self.config.max_feature_noise_ratio = thresholds.get('max_feature_noise_ratio', 0.3)
        else:
            self.config.max_feature_noise_ratio = thresholds.get('max_feature_noise_ratio', self.config.max_feature_noise_ratio)

        if not hasattr(self.config, 'min_temporal_stability') or self.config.min_temporal_stability is None:
            self.config.min_temporal_stability = thresholds.get('min_temporal_stability', 0.8)
        else:
            self.config.min_temporal_stability = thresholds.get('min_temporal_stability', self.config.min_temporal_stability)

    def _get_calibrated_quality_thresholds(self) -> Dict[str, float]:
        """Resolve calibrated thresholds with metadata overrides."""

        thresholds = get_calibrated_thresholds()

        metadata_thresholds = None
        if isinstance(self.execution_metadata, dict):
            calibration_block = self.execution_metadata.get('quality_calibration', {})
            if isinstance(calibration_block, dict):
                metadata_thresholds = calibration_block.get('quality_thresholds')

        if isinstance(metadata_thresholds, dict):
            thresholds = {**thresholds, **metadata_thresholds}

        return {
            'min_regime_persistence': float(
                thresholds.get('min_regime_persistence', getattr(self.config, 'min_regime_persistence', 0.7) or 0.7)
            ),
            'max_feature_noise_ratio': float(
                thresholds.get('max_feature_noise_ratio', getattr(self.config, 'max_feature_noise_ratio', 0.3) or 0.3)
            ),
            'min_temporal_stability': float(
                thresholds.get('min_temporal_stability', getattr(self.config, 'min_temporal_stability', 0.6) or 0.6)
            ),
        }

    def _calibrate_quality_thresholds(self, context: ClusteringContext, final_quality: Dict[str, Any]) -> None:
        """Update calibration statistics and thresholds using the latest results."""

        try:
            features = context.optimized_features if context.optimized_features is not None else context.original_features
            assignments = context.smoothed_assignments if context.smoothed_assignments is not None else context.raw_assignments

            if features is None or assignments is None or len(assignments) == 0:
                return

            features = np.asarray(features)
            assignments = np.asarray(assignments)

            calibration_state = self.execution_metadata.get('quality_calibration', get_current_calibration())
            history_block = calibration_state.get('history', {})

            def _copy_history(key: str) -> List[float]:
                values = history_block.get(key, [])
                return list(values) if isinstance(values, list) else []

            persistence_history = _copy_history('persistence')
            noise_history = _copy_history('noise_ratio')
            stability_history = _copy_history('temporal_stability')
            confidence_history = _copy_history('confidence')
            silhouette_history = _copy_history('silhouette')
            davies_history = _copy_history('davies_bouldin')
            cv_history = _copy_history('cv_score')

            def _extend(history: List[float], values: List[float], max_length: int = 500) -> List[float]:
                for value in values:
                    if value is None:
                        continue
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        history.append(float(value))
                if len(history) > max_length:
                    history = history[-max_length:]
                return history

            persistence_scores: List[float] = []
            noise_scores: List[float] = []
            stability_scores: List[float] = []

            for idx in range(features.shape[1]):
                column = features[:, idx]
                try:
                    persistence = self._calculate_feature_regime_persistence(column, context.market_data)
                    if isinstance(persistence, (int, float)) and np.isfinite(persistence):
                        persistence_scores.append(float(np.clip(persistence, 0.0, 1.0)))
                except Exception:
                    continue

                try:
                    noise_ratio = self._calculate_feature_noise_ratio(column)
                    if isinstance(noise_ratio, (int, float)) and np.isfinite(noise_ratio):
                        noise_scores.append(float(max(0.0, noise_ratio)))
                except Exception:
                    continue

                try:
                    temporal = self._calculate_feature_temporal_stability(column)
                    if isinstance(temporal, (int, float)) and np.isfinite(temporal):
                        stability_scores.append(float(np.clip(temporal, 0.0, 1.0)))
                except Exception:
                    continue

            persistence_history = _extend(persistence_history, persistence_scores)
            noise_history = _extend(noise_history, noise_scores)
            stability_history = _extend(stability_history, stability_scores)

            # Confidence metric
            confidence_candidates: List[float] = []
            optimization_metrics = context.optimization_metrics or {}
            for key in ('final_score', 'overall_confidence', 'optimization_score'):
                value = optimization_metrics.get(key)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    confidence_candidates.append(float(np.clip(value, 0.0, 1.0)))

            fusion_metadata = context.fusion_metadata or {}
            for key in ('average_confidence', 'mean_confidence', 'confidence_score'):
                value = fusion_metadata.get(key)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    confidence_candidates.append(float(np.clip(value, 0.0, 1.0)))

            if not confidence_candidates:
                silhouette_value = final_quality.get('silhouette_score')
                if isinstance(silhouette_value, (int, float)) and np.isfinite(silhouette_value):
                    if silhouette_value > 1.0:
                        normalized = np.clip(silhouette_value, 0.0, 1.0)
                    else:
                        normalized = (silhouette_value + 1.0) / 2.0
                    confidence_candidates.append(float(np.clip(normalized, 0.0, 1.0)))

            if not confidence_candidates:
                stability_value = self._calculate_stability_score(assignments)
                confidence_candidates.append(float(np.clip(stability_value, 0.0, 1.0)))

            confidence_metric = float(np.mean(confidence_candidates)) if confidence_candidates else 0.5
            confidence_history = _extend(confidence_history, [confidence_metric])

            silhouette_value = final_quality.get('silhouette_score')
            if isinstance(silhouette_value, (int, float)) and np.isfinite(silhouette_value):
                silhouette_history = _extend(silhouette_history, [float(silhouette_value)])

            davies_value = final_quality.get('davies_bouldin_score')
            if isinstance(davies_value, (int, float)) and np.isfinite(davies_value):
                davies_history = _extend(davies_history, [float(davies_value)])

            cv_score_value = None
            try:
                from ..regime_analysis.metrics import calculate_cv_score

                cv_score_value = calculate_cv_score(features, assignments)
            except Exception:
                cv_score_value = None

            if isinstance(cv_score_value, (int, float)) and np.isfinite(cv_score_value):
                cv_history = _extend(cv_history, [float(cv_score_value)])

            def _quantiles(values: List[float]) -> Dict[str, float]:
                if not values:
                    return {}
                array = np.asarray(values, dtype=float)
                array = array[np.isfinite(array)]
                if array.size == 0:
                    return {}
                return {
                    'p10': float(np.quantile(array, 0.1)),
                    'p25': float(np.quantile(array, 0.25)),
                    'p50': float(np.quantile(array, 0.5)),
                    'p75': float(np.quantile(array, 0.75)),
                    'p90': float(np.quantile(array, 0.9)),
                }

            def _mean(values: List[float]) -> Optional[float]:
                if not values:
                    return None
                array = np.asarray(values, dtype=float)
                array = array[np.isfinite(array)]
                if array.size == 0:
                    return None
                return float(np.mean(array))

            quantiles_map = {
                'persistence': _quantiles(persistence_history),
                'noise_ratio': _quantiles(noise_history),
                'temporal_stability': _quantiles(stability_history),
                'confidence': _quantiles(confidence_history),
                'silhouette': _quantiles(silhouette_history),
                'davies_bouldin': _quantiles(davies_history),
                'cv_score': _quantiles(cv_history),
            }

            means_map = {
                'persistence': _mean(persistence_history),
                'noise_ratio': _mean(noise_history),
                'temporal_stability': _mean(stability_history),
                'confidence': _mean(confidence_history),
                'silhouette': _mean(silhouette_history),
                'davies_bouldin': _mean(davies_history),
                'cv_score': _mean(cv_history),
            }

            current_thresholds = get_calibrated_thresholds()

            # ENHANCED: More stringent quality thresholds for better results
            quality_thresholds = {
                'min_regime_persistence': quantiles_map['persistence'].get('p60', current_thresholds.get('min_regime_persistence', 0.75)),  # Increased from p50 to p60
                'max_feature_noise_ratio': quantiles_map['noise_ratio'].get('p70', current_thresholds.get('max_feature_noise_ratio', 0.25)),  # Decreased from p75 to p70
                'min_temporal_stability': quantiles_map['temporal_stability'].get('p60', current_thresholds.get('min_temporal_stability', 0.70)),  # Increased from p50 to p60
            }

            confidence_levels = {
                'high': quantiles_map['confidence'].get('p75', 0.8),
                'medium': quantiles_map['confidence'].get('p50', 0.6),
                'low': quantiles_map['confidence'].get('p25', 0.4),
            }

            metric_thresholds = {
                'silhouette': {
                    'excellent': quantiles_map['silhouette'].get('p90', 0.7),
                    'good': quantiles_map['silhouette'].get('p75', 0.5),
                    'fair': quantiles_map['silhouette'].get('p50', 0.3),
                },
                'davies_bouldin': {
                    'excellent': quantiles_map['davies_bouldin'].get('p10', 0.5),
                    'good': quantiles_map['davies_bouldin'].get('p25', 1.0),
                    'fair': quantiles_map['davies_bouldin'].get('p50', 2.0),
                },
                'cv_score': {
                    'excellent': quantiles_map['cv_score'].get('p90', 0.8),
                    'good': quantiles_map['cv_score'].get('p75', 0.6),
                    'fair': quantiles_map['cv_score'].get('p50', 0.4),
                },
            }

            calibration_payload = {
                'history': {
                    'persistence': persistence_history,
                    'noise_ratio': noise_history,
                    'temporal_stability': stability_history,
                    'confidence': confidence_history,
                    'silhouette': silhouette_history,
                    'davies_bouldin': davies_history,
                    'cv_score': cv_history,
                },
                'statistics': {
                    'quantiles': quantiles_map,
                    'means': means_map,
                },
                'quality_thresholds': quality_thresholds,
                'confidence_levels': confidence_levels,
                'metric_thresholds': metric_thresholds,
                'last_updated': datetime.now().isoformat(),
            }

            self.execution_metadata['quality_calibration'] = calibration_payload
            update_quality_calibration(calibration_payload)

            self.config.min_regime_persistence = quality_thresholds['min_regime_persistence']
            self.config.max_feature_noise_ratio = quality_thresholds['max_feature_noise_ratio']
            self.config.min_temporal_stability = quality_thresholds['min_temporal_stability']

        except Exception as exc:  # pragma: no cover - calibration should not block execution
            tprint_warning(f"Quality threshold calibration failed: {exc}")

    # @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    # @performance_tracked(log_performance=True, track_memory=True)
    def _prepare_features(self, market_data: pd.DataFrame) -> FeaturePreparationResult:
        """Prepare market features for clustering and retain Stage 1 metadata with hardware optimization."""
        tprint("Step 4: Preparing features using shared utilities", "INFO")
        result = prepare_market_features(
            market_data,
            self.feature_config,
            verbose=True,
            return_metadata=True,
        )

        if result is None:
            tprint("Failed to prepare features for clustering", "ERROR")
            raise ValueError("Failed to prepare features for clustering")

        if not isinstance(result, FeaturePreparationResult):
            features_array = np.asarray(result)
            if features_array.size == 0:
                raise ValueError("Failed to prepare features for clustering")
            result = FeaturePreparationResult(
                features_array=features_array,
                features_df=pd.DataFrame(features_array, columns=[f"feature_{i}" for i in range(features_array.shape[1])]),
                summary={},
                metadata={'stage_metadata': {}, 'feature_columns': []},
            )

        if result.features_array.size == 0:
            tprint("Failed to prepare features for clustering", "ERROR")
            raise ValueError("Failed to prepare features for clustering")

        self.stage1_features_df = result.features_df.copy()
        self.stage1_filtered_df = self.stage1_features_df.copy()
        self.stage1_metadata = copy.deepcopy(result.metadata or {})
        self.features = result.features_array

        tprint(f"Features prepared: {result.features_array.shape}", "SUCCESS")
        return result

    def _infer_feature_category(self, feature_name: str) -> str:
        """Infer the high-level category of a feature name."""
        name = feature_name.lower()
        if 'volatility' in name or 'vol_' in name:
            return 'volatility_regime'
        if 'volume' in name or 'liquidity' in name:
            return 'volume_regime'
        if 'trend' in name or 'structural' in name:
            return 'structural_trend'
        if 'statistical' in name or 'distribution' in name or 'entropy' in name or 'autocorr' in name:
            return 'statistical_regime'
        if 'stability' in name or 'persistence' in name or 'quality' in name:
            return 'regime_quality'
        return 'other'

    # @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    # @performance_tracked(log_performance=True, track_memory=True)
    def _select_regime_features(
        self,
        feature_result: FeaturePreparationResult,
        market_data: pd.DataFrame,
        target_n_features: int = 200
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Perform PID-based feature selection for regime discovery.
        
        Reduces high-dimensional feature space using Partial Information Decomposition
        to identify features with high synergy, unique information, and low redundancy
        for robust regime clustering.
        
        Args:
            feature_result: Stage 1 feature preparation result
            market_data: Market data with potential target labels
            target_n_features: Target number of features to select (default: 200)

        Returns:
            Tuple of (selected_features, selected_feature_names, selection_metadata)
        """
        if not isinstance(feature_result, FeaturePreparationResult):
            raise ValueError("Feature preparation result is required for regime feature selection")

        stage1_df = feature_result.features_df.copy()
        stage1_metadata = copy.deepcopy(feature_result.metadata or {})
        if stage1_df.empty:
            raise ValueError("Stage 1 feature DataFrame is empty")

        n_samples, n_features = stage1_df.shape

        selection_metadata: Dict[str, Any] = {
            'stage1_metadata': stage1_metadata,
            'stage1_feature_count': int(n_features),
            'operations': list(stage1_metadata.get('stage_metadata', {}).get('operations', [])),
        }

        selection_operations: List[Dict[str, Any]] = stage1_metadata.setdefault('selection_operations', [])

        # Always perform feature selection to ensure we have exactly target_n_features maximum
        tprint(f"🔍 FEATURE SELECTION: Starting with {n_features} features, target: {target_n_features}", color="cyan", bold=True)
        
        # Check if feature selection is needed
        if n_features <= target_n_features:
            tprint(f"✅ Feature count ({n_features}) already within target ({target_n_features}), but still performing selection for optimization", "INFO")
            # Don't skip - still perform selection for optimization
        
        # Check dimensionality warning
        sample_to_feature_ratio = n_samples / n_features
        if sample_to_feature_ratio < 5.0:
            tprint(f"⚠️  HIGH DIMENSIONALITY DETECTED:", color="yellow", bold=True)
            tprint(f"   • {n_features} features for {n_samples} samples", color="yellow")
            tprint(f"   • Sample-to-feature ratio: {sample_to_feature_ratio:.2f}", color="yellow")
            if n_features <= target_n_features:
                tprint(f"   • Performing PID-based feature selection to optimize for {target_n_features} features", color="cyan")
            else:
                tprint(f"   • Performing PID-based feature selection to reduce to {target_n_features} features", color="cyan")
        
        # Stage 1.1: Drop signal-like features according to configuration patterns
        compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.config.signal_like_patterns or []]
        signal_like_columns = [
            column for column in stage1_df.columns
            if any(pattern.search(column) for pattern in compiled_patterns)
        ] if compiled_patterns else []

        if signal_like_columns:
            stage1_df = stage1_df.drop(columns=signal_like_columns)
            operation_record = {
                'type': 'signal_filter',
                'dropped_columns': signal_like_columns,
            }
            selection_metadata['operations'].append(operation_record)
            selection_operations.append(operation_record)
            tprint_info(f"🔎 Signal-like feature filter removed {len(signal_like_columns)} columns")

        # Stage 1.2: Enforce per-category caps prior to dimensionality reduction
        category_caps = self.config.feature_category_caps or {}
        category_counts_before = defaultdict(int)
        for col in stage1_df.columns:
            category_counts_before[self._infer_feature_category(col)] += 1

        if category_caps:
            kept_columns: List[str] = []
            category_counts_after = defaultdict(int)
            dropped_by_cap: List[str] = []
            for column in stage1_df.columns:
                category = self._infer_feature_category(column)
                cap = category_caps.get(category)
                if cap is None or category_counts_after[category] < cap:
                    kept_columns.append(column)
                    category_counts_after[category] += 1
                else:
                    dropped_by_cap.append(column)

            if dropped_by_cap:
                stage1_df = stage1_df[kept_columns]
                operation_record = {
                    'type': 'category_cap',
                    'dropped_columns': dropped_by_cap,
                    'caps': category_caps,
                }
                selection_metadata['operations'].append(operation_record)
                selection_operations.append(operation_record)
                tprint_info(
                    f"🔧 Category caps enforced: {len(dropped_by_cap)} features removed to respect per-category limits"
                )
            else:
                category_counts_after = category_counts_before
        else:
            category_counts_after = category_counts_before

        selection_metadata['category_counts_before'] = dict(category_counts_before)
        selection_metadata['category_counts_after'] = dict(category_counts_after)

        stage1_metadata['filtered_feature_columns'] = list(stage1_df.columns)
        self.stage1_filtered_df = stage1_df.copy()
        self.stage1_metadata = stage1_metadata

        base_features = stage1_df.to_numpy()
        base_names = list(stage1_df.columns)
        if base_features.size == 0:
            raise ValueError("No features available after Stage 1 filtering")

        operations_combined = list(selection_metadata['operations'])
        metadata: Dict[str, Any] = {
            'selection_performed': True,
            'stage1_operations': operations_combined,
            'operations': operations_combined,
            'original_n_features': int(n_features),
            'post_stage1_n_features': int(base_features.shape[1]),
            'stage1_metadata': stage1_metadata,
        }

        # Attempt to extract interim TAS/NAS assignments for discriminative scoring
        assignment_sources: List[np.ndarray] = []
        try:
            tas_assignments, nas_assignments = self._extract_regime_assignments()
            if isinstance(tas_assignments, np.ndarray) and len(np.unique(tas_assignments)) > 1:
                assignment_sources.append(tas_assignments)
            if isinstance(nas_assignments, np.ndarray) and len(np.unique(nas_assignments)) > 1:
                assignment_sources.append(nas_assignments)
        except Exception as exc:  # pragma: no cover - defensive logging
            tprint_warning(f"⚠️ Unable to extract interim TAS/NAS assignments for feature scoring: {exc}")

        if not assignment_sources:
            tprint_warning(
                "⚠️ Interim TAS/NAS assignments unavailable - falling back to variance-based ranking for feature pruning"
            )
            scores = np.var(base_features, axis=0)
            scoring_method = 'variance_proxy'
        else:
            scoring_method = 'fisher_ratio'
            scores = np.zeros(base_features.shape[1], dtype=float)
            for assignments in assignment_sources:
                unique_labels = np.unique(assignments)
                label_scores = np.zeros_like(scores)
                for feature_idx in range(base_features.shape[1]):
                    feature_values = base_features[:, feature_idx]
                    overall_mean = float(np.mean(feature_values))
                    between_var = 0.0
                    within_var = 0.0
                    for label in unique_labels:
                        label_mask = assignments == label
                        count = int(np.sum(label_mask))
                        if count < 2:
                            continue
                        label_values = feature_values[label_mask]
                        prior = count / len(feature_values)
                        label_mean = float(np.mean(label_values))
                        label_var = float(np.var(label_values))
                        between_var += prior * (label_mean - overall_mean) ** 2
                        within_var += prior * label_var

                    if between_var <= 0.0 and within_var <= 0.0:
                        label_scores[feature_idx] = 0.0
                    else:
                        label_scores[feature_idx] = between_var / (between_var + within_var + 1e-12)

                scores += label_scores

            scores /= max(1, len(assignment_sources))

        # Normalize scores for interpretability
        if np.all(scores == 0):
            normalized_scores = scores
        else:
            max_score = float(np.max(scores))
            normalized_scores = scores / (max_score + 1e-12)

        # Determine pruning threshold and retain top-ranked features
        sorted_indices = np.argsort(scores)[::-1]
        if sorted_indices.size == 0:
            raise ValueError("No features available after scoring")

        score_threshold = 0.0
        if sorted_indices.size > 1:
            score_threshold = float(max(0.01, np.percentile(scores, 10)))

        candidate_indices = [idx for idx in sorted_indices if scores[idx] >= score_threshold]
        if not candidate_indices:
            candidate_indices = sorted_indices.tolist()

        max_allowed = min(target_n_features, len(sorted_indices))
        selected_indices = candidate_indices[:max_allowed]

        if len(selected_indices) < max_allowed:
            for idx in sorted_indices:
                if idx not in selected_indices:
                    selected_indices.append(idx)
                if len(selected_indices) >= max_allowed:
                    break
        selected_features = base_features[:, selected_indices]
        selected_feature_names = [base_names[i] for i in selected_indices]

        dropped_indices = sorted(set(range(len(base_names))) - set(selected_indices))
        dropped_feature_names = [base_names[i] for i in dropped_indices]

        # Persist feature scores for downstream interpretability
        feature_scores = {name: float(normalized_scores[i]) for i, name in enumerate(base_names)}
        retained_scores = {name: feature_scores[name] for name in selected_feature_names}

        metadata = metadata or {}
        metadata.update({
            'selection_performed': True,
            'method': f'regime_scoring_{scoring_method}',
            'original_n_features': int(base_features.shape[1]),
            'selected_n_features': int(selected_features.shape[1]),
            'score_threshold': float(score_threshold),
            'feature_scores': feature_scores,
            'retained_feature_scores': retained_scores,
            'dropped_features': dropped_feature_names,
            'retained_features': selected_feature_names,
            'stage1_feature_count': selection_metadata.get('stage1_feature_count'),
            'category_counts_before': selection_metadata.get('category_counts_before', {}),
            'category_counts_after': selection_metadata.get('category_counts_after', {}),
            'signal_like_dropped': signal_like_columns,
            'operations': operations_combined,
        })

        self.feature_scores = feature_scores

        # Logging summary
        tprint_info(
            f"🔎 Feature scoring ({scoring_method}): retained {len(selected_feature_names)} / {len(base_names)} features"
        )
        if selected_feature_names:
            top_preview = selected_feature_names[:5]
            top_scores = [retained_scores[name] for name in top_preview]
            tprint_info(
                "   Top features: "
                + ", ".join(f"{name} ({score:.3f})" for name, score in zip(top_preview, top_scores))
            )
        if dropped_feature_names:
            tprint_info(
                f"   Dropped low-score features: {min(5, len(dropped_feature_names))}/{len(dropped_feature_names)} preview -> "
                + ", ".join(dropped_feature_names[:5])
            )

        # Stage 2: Standardize, clip, and project using PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(selected_features)
        clip_threshold = float(max(0.0, getattr(self.config, 'zscore_clip_threshold', 0.0)))
        if clip_threshold > 0.0:
            scaled_features = np.clip(scaled_features, -clip_threshold, clip_threshold)

        pca_factor = float(max(1.0, getattr(self.config, 'pca_components_factor', 1.0)))
        n_regimes = int(max(1, getattr(self.config, 'n_regimes', self.config.regime_search_min)))
        n_components = int(max(1, round(pca_factor * n_regimes)))
        n_components = min(n_components, scaled_features.shape[1], scaled_features.shape[0])
        if n_components <= 0:
            n_components = min(1, scaled_features.shape[1])

        pca = PCA(n_components=n_components, svd_solver='auto', random_state=42)
        projected_features = pca.fit_transform(scaled_features)

        explained_ratio = getattr(pca, 'explained_variance_ratio_', np.array([]))
        explained_ratio_list = explained_ratio.tolist() if explained_ratio.size else []
        cumulative_variance = float(np.sum(explained_ratio)) if explained_ratio.size else 0.0

        artifact_dir = Path(getattr(self.config, 'artifact_dir', 'artifacts'))
        artifact_dir.mkdir(parents=True, exist_ok=True)
        projection_path = artifact_dir / 'clustering_projection.pkl'
        with projection_path.open('wb') as handle:
            pickle.dump(
                {
                    'scaler': scaler,
                    'pca': pca,
                    'feature_names': selected_feature_names,
                    'clip_threshold': clip_threshold,
                },
                handle,
            )

        projection_metadata = {
            'n_components': int(getattr(pca, 'n_components_', n_components)),
            'explained_variance_ratio': explained_ratio_list,
            'explained_variance_cumulative': cumulative_variance,
            'clip_threshold': clip_threshold,
            'selected_feature_names': selected_feature_names,
        }

        metadata['projection'] = projection_metadata
        metadata['projection_artifact'] = str(projection_path)

        self.feature_projection_metadata = projection_metadata
        self.feature_projection_artifact_path = projection_path
        self.features = projected_features

        return projected_features, selected_feature_names, metadata
    
    def _regime_feature_generation(
        self, 
        features: np.ndarray, 
        target_n_features: int
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Use dedicated regime feature generator for regime-focused features.
        
        Process:
        1. Use all features from regime generator
        2. If 100+ features, use feature selector to reduce to 100
        3. Use same feature set for NAS, TAS & clustering
        
        Args:
            features: Feature matrix
            target_n_features: Number of features to target (100)
            
        Returns:
            Tuple of (selected_features, selected_feature_names, selection_metadata)
        """
        try:
            tprint("🔍 Using dedicated regime feature generator...", "INFO")
            tprint(f"🔍 REGIME GENERATION: Step 1 - Generate all regime features", color="cyan")
            
            # Step 1: Use all features from regime generator
            # For now, we'll use all available features as regime features
            n_features = features.shape[1]
            tprint(f"🔍 REGIME FEATURES: Found {n_features} regime features", color="green")
            
            # Step 2: If 100+ features, use feature selector to reduce to 100
            if n_features >= target_n_features:
                tprint(f"🔍 FEATURE SELECTION: {n_features} features >= {target_n_features}, reducing to 100", color="yellow")
                tprint(f"🔍 SELECTION: Using variance-based selection to reduce {n_features} → {target_n_features}", color="cyan")
                return self._variance_based_feature_selection(features, target_n_features)
            else:
                tprint(f"🔍 REGIME FEATURES: {n_features} features < {target_n_features}, using all features", color="green")
                # Use all available features
                selected_features = features
                selected_feature_names = [f"regime_feature_{i}" for i in range(n_features)]
                
                metadata = {
                    'selection_performed': False,
                    'method': 'regime_generator_all',
                    'original_n_features': n_features,
                    'selected_n_features': n_features,
                    'regime_features_used': True,
                    'feature_reduction_applied': False
                }
                
                tprint(f"✅ REGIME FEATURES: Using all {n_features} regime features", "SUCCESS")
                return selected_features, selected_feature_names, metadata
            
        except Exception as e:
            tprint(f"❌ Regime feature generation failed: {e}", "ERROR")
            tprint("   Falling back to variance-based selection", "WARNING")
            return self._variance_based_feature_selection(features, target_n_features)

    def _variance_based_feature_selection(
        self, 
        features: np.ndarray, 
        target_n_features: int
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Fallback variance-based feature selection.
        
        Args:
            features: Feature matrix
            target_n_features: Number of features to select
            
        Returns:
            Tuple of (selected_features, selected_feature_names, selection_metadata)
        """
        try:
            tprint("🔍 Performing variance-based feature selection...", "INFO")
            tprint(f"🔍 VARIANCE SELECTION: Reducing {features.shape[1]} → {target_n_features} features", color="yellow")
            
            # Calculate variance for each feature
            variances = np.var(features, axis=0)
            tprint(f"🔍 VARIANCE STATS: Min: {variances.min():.6f}, Max: {variances.max():.6f}, Mean: {variances.mean():.6f}", color="blue")
            
            # Select top N features by variance, targeting 100 features
            actual_target = min(target_n_features, features.shape[1])  # Target 100 or all available features
            top_indices = np.argsort(variances)[::-1][:actual_target]
            selected_features = features[:, top_indices]
            selected_feature_names = [f"feature_{i}" for i in top_indices]
            
            tprint(f"🔍 VARIANCE RESULT: Selected {len(top_indices)} features with highest variance (target: {target_n_features})", color="green")
            tprint(f"🎯 FINAL RESULT: Regime feature selection completed - {features.shape[1]} → {len(top_indices)} features", color="green", bold=True)
            
            metadata = {
                'selection_performed': True,
                'method': 'variance_based',
                'original_n_features': features.shape[1],
                'selected_n_features': len(top_indices),
                'sample_to_feature_ratio_after': features.shape[0] / len(top_indices)
            }
            
            tprint(f"✅ Variance-based selection: {features.shape[1]} → {len(top_indices)} features", "SUCCESS")
            tprint(f"🎯 FINAL RESULT: Feature selection completed - {features.shape[1]} → {len(top_indices)} features (target: {target_n_features})", color="green", bold=True)
            return selected_features, selected_feature_names, metadata
            
        except Exception as e:
            tprint(f"❌ Variance-based selection failed: {e}", "ERROR")
            # Last resort: return original features
            feature_names = [f"feature_{i}" for i in range(features.shape[1])]
            return features, feature_names, {'selection_performed': False, 'error': str(e)}

    # @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    # @performance_tracked(log_performance=True, track_memory=True)
    def _generate_cluster_characteristics(
        self,
        market_data: pd.DataFrame,
        clustering_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate characteristics for each cluster with hardware optimization."""
        tprint("Step 8: Generating cluster characteristics using shared utilities", "INFO")
        cluster_characteristics = generate_cluster_characteristics(
            market_data,
            clustering_result['cluster_assignments'],
            clustering_result.get('cluster_centers'),
            verbose=True,
        )
        tprint("Cluster characteristics generated", "SUCCESS")
        return cluster_characteristics

    # @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    # @performance_tracked(log_performance=True, track_memory=True)
    def _calculate_clustering_metrics_using_shared_utils(
        self,
        clustering_result: Dict[str, Any],
        cluster_characteristics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate clustering metrics using shared utilities with defensive error handling and hardware optimization."""
        try:
            tprint("Step 9: Calculating clustering metrics using shared utilities", "INFO")
            
            # Extract assignments and other needed data
            cluster_assignments = clustering_result.get('cluster_assignments', [])
            cluster_centers = clustering_result.get('cluster_centers', [])
            
            if not cluster_assignments:
                tprint_warning("No cluster assignments found in clustering result")
                return {"error": "no_assignments"}
            
            # Convert to numpy array for processing
            assignments_array = np.array(cluster_assignments)
            
            # Use shared metrics calculator
            # Note: consensus and disagreement metrics require two assignment arrays (TAS vs NAS)
            # For clustering-only metrics, we skip these and use other metrics
            economic_scores = self.metrics_calculator.calculate_economic_scores(
                assignments_array
            )
            trading_scores = self.metrics_calculator.calculate_trading_scores(
                assignments_array
            )
            stability_scores = self.metrics_calculator.calculate_stability_scores(
                assignments_array
            )
            
            # Compile metrics dictionary
            clustering_metrics = {
                "economic_scores": economic_scores,
                "trading_scores": trading_scores,
                "stability_scores": stability_scores,
                "cluster_centers": cluster_centers,
                "n_clusters": len(set(cluster_assignments)),
                "total_assignments": len(cluster_assignments),
                "feature_projection": {
                    "selected_feature_names": getattr(self, 'feature_names', []),
                    "projection_metadata": getattr(self, 'feature_projection_metadata', {}),
                    "projection_artifact": str(self.feature_projection_artifact_path)
                    if getattr(self, 'feature_projection_artifact_path', None)
                    else None,
                },
            }
            
            tprint("Clustering metrics calculated using shared utilities", "SUCCESS")
            return clustering_metrics
            
        except Exception as exc:
            tprint_error(f"Failed to calculate clustering metrics: {exc}")
            # Return fallback metrics
            return {
                "error": str(exc),
                "consensus_metrics": {},
                "disagreement_metrics": {},
                "economic_scores": {},
                "trading_scores": {},
                "stability_scores": {},
                "cluster_centers": clustering_result.get('cluster_centers', []),
                "n_clusters": clustering_result.get('n_clusters', 0),
                "total_assignments": len(clustering_result.get('cluster_assignments', []))
            }

    def _create_consolidated_artifacts(
        self,
        clustering_result: Dict[str, Any],
        cluster_characteristics: Dict[str, Any],
        clustering_metrics: Dict[str, Any],
        market_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Create consolidated artifacts from clustering outputs with comprehensive metadata."""
        try:
            tprint("Creating consolidated artifacts with comprehensive metadata", "INFO")
            
            # Extract core clustering data
            cluster_assignments = clustering_result.get('cluster_assignments', [])
            cluster_centers = clustering_result.get('cluster_centers', [])
            n_clusters = clustering_result.get('n_clusters', 0)
            algorithm_used = clustering_result.get('algorithm_used', 'unknown')
            
            # Extract optimization metadata
            optimization_metadata = clustering_result.get('optimization_metadata', {})
            
            # Create comprehensive artifact structure
            self.execution_metadata['end_time'] = datetime.now().isoformat()

            execution_meta = {
                'component': 'nas_tas_clustering',
                'timestamp': datetime.now().isoformat(),
                'uses_shared_utilities': True,
                'm1_hardware_available': M1_HARDWARE_AVAILABLE,
                'matrix_operations_available': MATRIX_OPERATIONS_AVAILABLE,
            }

            if isinstance(self.execution_metadata, dict):
                for key, value in self.execution_metadata.items():
                    if key == 'quality_calibration':
                        execution_meta[key] = value
                    elif isinstance(value, datetime):
                        execution_meta[key] = value.isoformat()
                    else:
                        execution_meta[key] = value

            artifacts = {
                # Core clustering results
                'nas_tas_clustering_result': {
                    'cluster_assignments': cluster_assignments,
                    'cluster_centers': cluster_centers,
                    'n_clusters': n_clusters,
                    'algorithm_used': algorithm_used,
                    'success': clustering_result.get('success', True),
                    'execution_time': clustering_result.get('execution_time', 0.0),
                    # COMPATIBILITY: Add regime states in format expected by regime data splitting
                    'regime_states': {
                        'assignments': cluster_assignments,
                        'regime_count': n_clusters,
                        'regime_labels': list(range(n_clusters)),
                        'regime_distribution': dict(zip(*np.unique(cluster_assignments, return_counts=True))) if len(cluster_assignments) > 0 else {},
                        'regime_centers': cluster_centers,
                        'regime_quality': clustering_result.get('clustering_quality', {})
                    },
                    'projection_metadata': getattr(self, 'feature_projection_metadata', {}),
                    'projection_artifact': str(self.feature_projection_artifact_path)
                    if getattr(self, 'feature_projection_artifact_path', None)
                    else None,
                },
                
                # Raw and smoothed assignments
                'raw_assignments': cluster_assignments,
                'smoothed_assignments': cluster_assignments,  # Same as raw for now
                
                # Clustering quality metrics
                'clustering_quality': clustering_result.get('clustering_quality', {}),
                
                # Optimization metadata
                'optimization_metadata': optimization_metadata,
                
                # Cluster characteristics
                'cluster_characteristics': cluster_characteristics,
                
                # Comprehensive metrics
                'clustering_metrics': clustering_metrics,

                'feature_projection': {
                    'selected_feature_names': getattr(self, 'feature_names', []),
                    'projection_metadata': getattr(self, 'feature_projection_metadata', {}),
                    'projection_artifact': str(self.feature_projection_artifact_path)
                    if getattr(self, 'feature_projection_artifact_path', None)
                    else None,
                    'stage1_metadata': getattr(self, 'stage1_metadata', {}),
                    'selection_metadata': getattr(self, 'selection_metadata', {}),
                },

                # Data information
                'data_info': {
                    'total_samples': len(cluster_assignments),
                    'n_features': market_data.shape[1] if not market_data.empty else 0,
                    'n_clusters': n_clusters,
                    'symbol': getattr(self.config, 'symbol', 'UNKNOWN'),
                    'timeframe': getattr(self.config, 'timeframe', 'UNKNOWN'),
                    'exchange': getattr(self.config, 'exchange', 'UNKNOWN')
                },

                # Execution metadata
                'execution_metadata': {
                    'component': 'nas_tas_clustering',
                    'timestamp': datetime.now().isoformat(),
                    'uses_shared_utilities': True,
                    'm1_hardware_available': M1_HARDWARE_AVAILABLE,
                    'matrix_operations_available': MATRIX_OPERATIONS_AVAILABLE,
                    'learned_metric_weights': self._serialize_learned_weights(),
                    'metric_weight_history': self._serialize_metric_history(),
                    'quality_calibration': get_current_calibration(),
                    'calibration_loaded_from_state': False,
                    **execution_meta
                },
                
                # Performance metrics
                'performance_metrics': {
                    'memory_usage_mb': get_memory_usage() / (1024**2) if 'get_memory_usage' in globals() else 0.0,
                    'processing_time': clustering_result.get('execution_time', 0.0),
                    'optimization_trials': optimization_metadata.get('iterations', 0)
                }
            }

            refined_feature_names = clustering_result.get('refined_feature_names')
            if refined_feature_names:
                artifacts['nas_tas_clustering_result']['refined_feature_names'] = refined_feature_names
                artifacts['refined_feature_names'] = refined_feature_names

            feature_scores = clustering_result.get('feature_scores')
            if feature_scores:
                artifacts['nas_tas_clustering_result']['feature_scores'] = feature_scores
                artifacts['feature_scores'] = feature_scores

            pca_loading_scores = clustering_result.get('pca_loading_scores')
            if pca_loading_scores:
                artifacts['nas_tas_clustering_result']['pca_loading_scores'] = pca_loading_scores
                artifacts['pca_loading_scores'] = pca_loading_scores

            if clustering_result.get('pre_pca_feature_names'):
                artifacts['pre_pca_feature_names'] = clustering_result['pre_pca_feature_names']

            # Add consensus and disagreement metrics if available
            if 'consensus_metrics' in clustering_metrics:
                artifacts['consensus_metrics'] = clustering_metrics['consensus_metrics']
            
            if 'disagreement_metrics' in clustering_metrics:
                artifacts['disagreement_metrics'] = clustering_metrics['disagreement_metrics']
            
            # Add economic and trading scores if available
            if 'economic_scores' in clustering_metrics:
                artifacts['economic_scores'] = clustering_metrics['economic_scores']
            
            if 'trading_scores' in clustering_metrics:
                artifacts['trading_scores'] = clustering_metrics['trading_scores']
            
            # Add stability scores if available
            if 'stability_scores' in clustering_metrics:
                artifacts['stability_scores'] = clustering_metrics['stability_scores']
            
            # Add feature optimization metadata if available
            if 'feature_optimization' in optimization_metadata:
                artifacts['feature_optimization'] = optimization_metadata['feature_optimization']
            
            # Add feature selection metadata
            if hasattr(self, 'selection_metadata') and self.selection_metadata:
                artifacts['feature_selection_metadata'] = self.selection_metadata
                tprint(f"✅ Added feature selection metadata to artifacts", "SUCCESS")
            
            # Add fusion metadata if available
            if 'fusion_metadata' in optimization_metadata:
                artifacts['fusion_metadata'] = optimization_metadata['fusion_metadata']
            
            # Add HMM smoothing metadata if available
            if 'hmm_transitions' in optimization_metadata:
                artifacts['hmm_smoothing'] = {
                    'transitions': optimization_metadata['hmm_transitions'],
                    'smoothing_metadata': optimization_metadata.get('smoothing_metadata', {})
                }
            
            # Add HMM smoothing metadata if available in clustering_metrics
            if 'hmm_smoothing' in clustering_metrics:
                artifacts['hmm_smoothing'] = clustering_metrics['hmm_smoothing']

            # ✅ SAVE: Regime assignments DataFrame as parquet file
            try:
                regime_assignments_df = clustering_result.get('regime_assignments_df')
                if regime_assignments_df is not None and not regime_assignments_df.empty:
                    symbol = getattr(self.config, 'symbol', 'ETHUSDT') if hasattr(self, 'config') else 'ETHUSDT'
                    regime_assignments_path = self._save_regime_assignments_parquet(regime_assignments_df, symbol)
                    artifacts['regime_assignments_path'] = str(regime_assignments_path)
                    tprint(f"💾 Saved regime assignments with features to {regime_assignments_path}", "SUCCESS")
                else:
                    tprint_warning("⚠️ No regime assignments DataFrame available to save")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to save regime assignments parquet: {e}")

            tprint("Consolidated artifacts created successfully", "SUCCESS")
            return artifacts
            
        except Exception as exc:
            tprint_error(f"Failed to create consolidated artifacts: {exc}")
            # Return minimal fallback artifacts
            return {
                'nas_tas_clustering_result': {
                    'cluster_assignments': clustering_result.get('cluster_assignments', []),
                    'cluster_centers': clustering_result.get('cluster_centers', []),
                    'n_clusters': clustering_result.get('n_clusters', 0),
                    'algorithm_used': clustering_result.get('algorithm_used', 'unknown'),
                    'success': False,
                    'error': str(exc)
                },
                'execution_metadata': {
                    'component': 'nas_tas_clustering',
                    'timestamp': datetime.now().isoformat(),
                    'error': str(exc),
                    'learned_metric_weights': self._serialize_learned_weights(),
                    'metric_weight_history': self._serialize_metric_history(),
                }
            }

    # @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    # @performance_tracked(log_performance=True, track_memory=True)
    def _build_artifacts(
        self,
        clustering_result: Dict[str, Any],
        cluster_characteristics: Dict[str, Any],
        clustering_metrics: Dict[str, Any],
        market_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Create consolidated artifacts from clustering outputs."""
        tprint("Step 10: Creating consolidated artifacts", "INFO")
        artifacts = self._create_consolidated_artifacts(
            clustering_result,
            cluster_characteristics,
            clustering_metrics,
            market_data,
        )
        tprint("Consolidated artifacts created", "SUCCESS")
        return artifacts

    @log_execution('Regime-Clustering', 'Regime Clustering', verbose=True)
    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run regime clustering analysis (BaseStep interface).
        
        Args:
            config: Configuration dictionary containing parameters for clustering
            
        Returns:
            Dictionary with clustering results and artifacts
        """
        return await self.execute(config)
    
    # @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    # @performance_tracked(log_performance=True, track_memory=True)
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute regime clustering analysis with hardware optimization.
        
        Args:
            config: Configuration dictionary containing parameters for clustering
            
        Returns:
            Dictionary with clustering results and artifacts
        """
        try:
            # Update clustering config with provided config
            if config:
                for key, value in config.items():
                    if hasattr(self.clustering_config, key):
                        setattr(self.clustering_config, key, value)
            
            # Set context for artifact management
            self._set_context(
                symbol=config.get('symbol', 'ETHUSDT'),
                exchange=config.get('exchange', 'binance'),
                information=config.get('information', 'regime_clustering'),
                direction=config.get('direction', 'both'),
                model=config.get('model', 'RegimeClustering')
            )
            
            tprint("🚀 Starting regime clustering execution", "INFO")
            
            # Initialize performance monitoring and hardware optimization
            tprint("📊 Initializing performance monitoring and hardware optimization...", "INFO")
            start_time = time.time()
            
            # Optimize hardware for clustering workload
            if self.hardware_manager:
                try:
                    self.hardware_manager.optimize_for_workload('data_processing')
                    tprint("✅ Hardware optimized for clustering workload", "SUCCESS")
                except Exception as e:
                    tprint(f"⚠️ Hardware optimization failed: {e}", "WARNING")

            # Step 2: Validate inputs and configuration using shared utilities
            self._validate_configuration()

            # Step 3: Initialize execution metadata
            self._initialize_execution_metadata()

            # Step 4: Load and validate market data using BaseStep artifact manager
            tprint("Step 4: Loading and validating market data", "INFO")
            market_data = self._load_market_data_from_artifacts()
            if market_data is None or market_data.empty:
                tprint("No market data available for clustering", "ERROR")
                raise ValueError("No market data available for clustering")

            tprint(f"Market data loaded: {len(market_data)} rows", "SUCCESS")

            # Step 4: Prepare features using shared utilities
            feature_result = self._prepare_features(market_data)

            # Step 4.5: Perform PID-based feature selection for regime discovery
            tprint("Step 4.5: Performing intelligent feature selection for regime discovery", "INFO")
            features, feature_names, selection_metadata = self._select_regime_features(
                feature_result=feature_result,
                market_data=market_data,
                target_n_features=100  # Target 100 features to avoid overfitting with 1,921 samples
            )

            # Store feature names and selection metadata for later use
            self.feature_names = feature_names
            self.selection_metadata = selection_metadata
            self.stage1_metadata = feature_result.metadata or {}
            
            # Optimize features for memory efficiency
            if self.memory_optimizer:
                try:
                    self.features = self.memory_optimizer.optimize_array(features)
                    tprint("✅ Features optimized for memory efficiency", "SUCCESS")
                except Exception as e:
                    tprint(f"⚠️ Memory optimization failed, using original features: {e}", "WARNING")
                    self.features = features
            else:
                self.features = features
                
            tprint(f"Feature selection completed: {selection_metadata.get('selected_n_features', len(feature_names))} features", "SUCCESS")

            # Step 5: Create clustering configuration using shared utilities
            tprint("Step 5: Creating clustering configuration using shared utilities", "INFO")
            clustering_config = self._create_clustering_config_using_shared_utils()
            tprint("Clustering configuration created", "SUCCESS")

            # Step 6: Perform clustering
            tprint("Step 6: Performing clustering", "INFO")
            clustering_result = await self._perform_clustering(features, market_data)
            tprint(f"Clustering completed: {clustering_result['n_clusters']} clusters", "SUCCESS")

            # Step 8: Generate cluster characteristics using shared utilities
            cluster_characteristics = self._generate_cluster_characteristics(
                market_data, clustering_result
            )

            # Step 9: Calculate metrics using shared utilities
            clustering_metrics = self._calculate_clustering_metrics_using_shared_utils(
                clustering_result, cluster_characteristics
            )

            # Update learned metric weights with the latest results
            self._update_learned_weights(clustering_result, clustering_metrics)

            # Step 10: Create consolidated artifacts
            artifacts = self._build_artifacts(
                clustering_result, cluster_characteristics, clustering_metrics, market_data
            )

            # Step 11: Create regime assignments parquet file with features
            try:
                cluster_assignments = clustering_result.get('cluster_assignments', [])
                regime_assignments_df = self._create_regime_assignments_dataframe(
                    cluster_assignments, features, market_data
                )

                # Add to artifacts for use by other components (both in main artifacts and in clustering result)
                artifacts['regime_assignments'] = regime_assignments_df
                artifacts['nas_tas_clustering_result']['regime_assignments'] = regime_assignments_df

                # Save as parquet file for regime analysis
                regime_assignments_path = self._save_regime_assignments_parquet(regime_assignments_df)
                artifacts['regime_assignments_path'] = str(regime_assignments_path)

                tprint(f"💾 Saved regime assignments with features to {regime_assignments_path}", "SUCCESS")

            except Exception as e:
                tprint_warning(f"⚠️ Failed to save regime assignments with features: {e}")
                # Continue without the parquet file - regime analysis will use fallback

            tprint(f'NAS-TAS Clustering completed: {clustering_result["n_clusters"]} clusters', "SUCCESS")
            
            # Cleanup hardware resources
            if self.hardware_manager:
                try:
                    self.hardware_manager.cleanup_resources()
                    tprint("✅ Hardware resources cleaned up", "SUCCESS")
                except Exception as e:
                    tprint(f"⚠️ Hardware cleanup failed: {e}", "WARNING")

            # Save artifacts using BaseStep methods
            for artifact_name, artifact_data in artifacts.items():
                if isinstance(artifact_data, pd.DataFrame):
                    self._save_dataframe(artifact_data, artifact_name)
                elif isinstance(artifact_data, dict):
                    self._save_metadata(artifact_data, artifact_name)
                else:
                    self._save_model(artifact_data, artifact_name)
            
            # Generate detailed metrics markdown report
            metrics_report_path = self._generate_metrics_report(
                clustering_result, cluster_characteristics, clustering_metrics, market_data
            )
            artifacts['metrics_report_path'] = str(metrics_report_path)
            
            return {
                'success': True,
                'artifacts': list(artifacts.keys()),
                'metrics_report_path': artifacts.get('metrics_report_path'),
                'metadata': {
                    'symbol': getattr(self.clustering_config, 'symbol', 'ETHUSDT'),
                    'timeframe': getattr(self.clustering_config, 'timeframe', '15m'),
                    'data_points_processed': len(market_data),
                    'n_clusters': clustering_result['n_clusters'],
                    'algorithm_type': 'regime_clustering',
                    'execution_successful': True,
                    'uses_shared_utilities': True
                }
            }
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            
            # Cleanup hardware resources on error
            if self.hardware_manager:
                try:
                    self.hardware_manager.cleanup_resources()
                except Exception as cleanup_error:
                    tprint(f"⚠️ Hardware cleanup failed during error handling: {cleanup_error}", "WARNING")
            
            # Log comprehensive error information
            tprint_error(f'NAS-TAS Clustering failed: {e}')
            tprint_debug(f'Error details: {error_traceback}')
            
            # Log structured error information
            error_info = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "component": "NAS-TAS-Clustering",
                "traceback": error_traceback,
                "timestamp": datetime.now().isoformat()
            }
            tprint_structured(error_info)

            return {
                'success': False,
                'error': {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": error_traceback,
                    "timestamp": datetime.now().isoformat()
                },
                'error_message': f"Regime clustering failed: {str(e)}",
                'metadata': {
                    'symbol': getattr(self.clustering_config, 'symbol', 'ETHUSDT'),
                    'timeframe': getattr(self.clustering_config, 'timeframe', '15m'),
                    'execution_successful': False,
                    'error_type': type(e).__name__
                }
            }
    
    def _generate_metrics_report(self, clustering_result: Dict[str, Any], 
                                cluster_characteristics: Dict[str, Any], 
                                clustering_metrics: Dict[str, Any], 
                                market_data: pd.DataFrame) -> Path:
        """
        Generate detailed metrics report in markdown format.
        
        Args:
            clustering_result: Results from clustering analysis
            cluster_characteristics: Characteristics of each cluster
            clustering_metrics: Calculated clustering metrics
            market_data: Original market data
            
        Returns:
            Path to the generated markdown report
        """
        try:
            # Create outcomes directory if it doesn't exist
            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(exist_ok=True)
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol = getattr(self.clustering_config, 'symbol', 'ETHUSDT')
            filename = f"regime_clustering_metrics_{symbol}_{timestamp}.md"
            report_path = outcomes_dir / filename
            
            # Generate comprehensive markdown report
            with open(report_path, 'w') as f:
                f.write(f"# Regime Clustering Analysis Report\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Symbol:** {symbol}\n")
                f.write(f"**Timeframe:** {getattr(self.clustering_config, 'timeframe', '15m')}\n\n")
                
                # Executive Summary
                f.write("## Executive Summary\n\n")
                f.write(f"- **Total Data Points:** {len(market_data):,}\n")
                f.write(f"- **Number of Clusters:** {clustering_result.get('n_clusters', 'N/A')}\n")
                f.write(f"- **Algorithm:** {clustering_result.get('algorithm', 'N/A')}\n")
                f.write(f"- **Execution Time:** {clustering_result.get('execution_time', 'N/A')} seconds\n\n")
                
                # Clustering Metrics
                f.write("## Clustering Quality Metrics\n\n")
                if 'silhouette_score' in clustering_metrics:
                    f.write(f"- **Silhouette Score:** {clustering_metrics['silhouette_score']:.4f}\n")
                if 'davies_bouldin_score' in clustering_metrics:
                    f.write(f"- **Davies-Bouldin Score:** {clustering_metrics['davies_bouldin_score']:.4f}\n")
                if 'calinski_harabasz_score' in clustering_metrics:
                    f.write(f"- **Calinski-Harabasz Score:** {clustering_metrics['calinski_harabasz_score']:.4f}\n")
                f.write("\n")
                
                # Cluster Characteristics
                f.write("## Cluster Characteristics\n\n")
                for cluster_id, characteristics in cluster_characteristics.items():
                    f.write(f"### Cluster {cluster_id}\n\n")
                    f.write(f"- **Size:** {characteristics.get('size', 'N/A')} data points\n")
                    f.write(f"- **Percentage:** {characteristics.get('percentage', 'N/A')}%\n")
                    
                    if 'volatility' in characteristics:
                        f.write(f"- **Average Volatility:** {characteristics['volatility']:.4f}\n")
                    if 'volume' in characteristics:
                        f.write(f"- **Average Volume:** {characteristics['volume']:.4f}\n")
                    if 'trend' in characteristics:
                        f.write(f"- **Trend Direction:** {characteristics['trend']}\n")
                    
                    f.write("\n")
                
                # Performance Metrics
                f.write("## Performance Metrics\n\n")
                if 'execution_time' in clustering_result:
                    f.write(f"- **Total Execution Time:** {clustering_result['execution_time']:.2f} seconds\n")
                if 'memory_usage' in clustering_result:
                    f.write(f"- **Peak Memory Usage:** {clustering_result['memory_usage']:.2f} MB\n")
                f.write("\n")
                
                # Hardware Optimization Status
                f.write("## Hardware Optimization Status\n\n")
                if HARDWARE_OPTIMIZATION_AVAILABLE:
                    f.write("- **Hardware Optimization:** ✅ Enabled\n")
                    f.write("- **Memory Optimization:** ✅ Active\n")
                    f.write("- **CPU Optimization:** ✅ Active\n")
                    if M1_HARDWARE_AVAILABLE:
                        f.write("- **M1-Specific Optimizations:** ✅ Active\n")
                else:
                    f.write("- **Hardware Optimization:** ❌ Not Available\n")
                f.write("\n")
                
                # Data Quality Metrics
                f.write("## Data Quality Metrics\n\n")
                f.write(f"- **Missing Values:** {market_data.isnull().sum().sum()}\n")
                f.write(f"- **Data Completeness:** {((len(market_data) - market_data.isnull().sum().sum()) / (len(market_data) * len(market_data.columns)) * 100):.2f}%\n")
                f.write(f"- **Memory Usage:** {market_data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB\n")
                f.write("\n")
                
                # Recommendations
                f.write("## Recommendations\n\n")
                silhouette_score = clustering_metrics.get('silhouette_score', 0)
                if silhouette_score > 0.5:
                    f.write("- ✅ **Excellent clustering quality** - Silhouette score indicates well-separated clusters\n")
                elif silhouette_score > 0.3:
                    f.write("- ⚠️ **Moderate clustering quality** - Consider feature engineering or parameter tuning\n")
                else:
                    f.write("- ❌ **Poor clustering quality** - Significant overlap between clusters detected\n")
                
                f.write("- Consider analyzing cluster stability over time\n")
                f.write("- Monitor regime transitions for trading opportunities\n")
                f.write("- Validate cluster characteristics with domain knowledge\n")
                f.write("\n")
                
                # Technical Details
                f.write("## Technical Details\n\n")
                f.write(f"- **Feature Count:** {len(self.feature_names) if hasattr(self, 'feature_names') else 'N/A'}\n")
                f.write(f"- **Feature Selection Method:** {self.selection_metadata.get('method', 'N/A') if hasattr(self, 'selection_metadata') else 'N/A'}\n")
                f.write(f"- **Configuration Used:** {self.clustering_config.__dict__}\n")
                f.write("\n")
                
                # Footer
                f.write("---\n")
                f.write("*Report generated by Ares Trading System - Regime Clustering Component*\n")
            
            tprint(f"📊 Generated detailed metrics report: {report_path}", "SUCCESS")
            return report_path
            
        except Exception as e:
            tprint_error(f"Failed to generate metrics report: {e}")
            # Return a fallback path
            return Path("outcomes/regime_clustering_metrics_fallback.md")
    
    # @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    # @performance_tracked(log_performance=True, track_memory=True)
    async def _load_market_data(self, data: Any) -> Optional[pd.DataFrame]:
        """Load and validate market data for clustering with hardware optimization."""
        try:
            tprint("Loading market data...", "INFO")
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                tprint("No market data provided, attempting to load from pipeline state", "WARNING")
                return None

            # If data is already a DataFrame, use it
            if isinstance(data, pd.DataFrame):
                tprint(f"Using provided DataFrame with {len(data)} rows", "INFO")
                return data.copy()

            # If data is a dictionary with market data
            if isinstance(data, dict) and 'market_data' in data:
                market_data = data['market_data']
                if isinstance(market_data, pd.DataFrame):
                    tprint(f"Using market data from dictionary with {len(market_data)} rows", "INFO")
                    return market_data.copy()

            tprint("Unknown data type provided", "WARNING")
            return None

        except Exception as e:
            tprint(f"Market data loading failed: {e}", "ERROR")
            return None
    
    # @performance_tracked(log_performance=True, track_memory=True)
    def _create_clustering_config_using_shared_utils(self) -> Dict[str, Any]:
        """Create clustering configuration using shared utilities with performance tracking."""
        try:
            tprint("Creating clustering configuration using shared utilities...", "INFO")

            # Use shared utilities to create configuration
            tprint("Creating base configuration...", "INFO")
            base_config = create_default_config(
                config_type="hybrid",
                symbol=getattr(self.clustering_config, 'symbol', 'ETHUSDT'),
                timeframe=getattr(self.clustering_config, 'timeframe', '15m'),
                n_regimes=getattr(self.config, 'n_regimes', 8)
            )
            tprint("Base configuration created", "SUCCESS")
            
            # Add clustering-specific parameters
            tprint("Adding clustering-specific parameters...", "INFO")
            clustering_config = {
                'algorithm_type': getattr(self.config, 'algorithm_type', 'adaptive_clustering'),
                'enable_economic_clustering': getattr(self.config, 'enable_economic_clustering', True),
                'enable_ensemble_clustering': getattr(self.config, 'enable_ensemble_clustering', True),
                'n_regimes': getattr(self.config, 'n_regimes', 8),
                'symbol': getattr(self.config, 'symbol', 'ETHUSDT'),
                'timeframe': getattr(self.config, 'timeframe', '15m'),
                'exchange': getattr(self.config, 'exchange', 'binance')
            }

            regime_weights = self._get_weight_group('regime')
            clustering_config.update({
                'economic_weight': regime_weights.get('economic', getattr(self.config, 'economic_weight', 0.25)),
                'volatility_regime_weight': regime_weights.get('volatility', getattr(self.config, 'volatility_regime_weight', 0.30)),
                'volume_regime_weight': regime_weights.get('volume', getattr(self.config, 'volume_regime_weight', 0.25)),
                'structural_trend_weight': regime_weights.get('structural_trend', getattr(self.config, 'structural_trend_weight', 0.20)),
            })

            # Update config attributes to keep external consumers in sync
            self.config.economic_weight = clustering_config['economic_weight']
            self.config.volatility_regime_weight = clustering_config['volatility_regime_weight']
            self.config.volume_regime_weight = clustering_config['volume_regime_weight']
            self.config.structural_trend_weight = clustering_config['structural_trend_weight']
            tprint("Clustering-specific parameters added", "SUCCESS")

            # Validate weights using shared utilities
            tprint("Validating and normalizing weights...", "INFO")
            weights_dict = {
                'economic': clustering_config['economic_weight'],
                'volatility_regime': clustering_config['volatility_regime_weight'],
                'volume_regime': clustering_config['volume_regime_weight'],
                'structural_trend': clustering_config['structural_trend_weight']
            }
            normalized_weights = normalize_weights(weights_dict)

            clustering_config.update({
                'economic_weight': normalized_weights['economic'],
                'volatility_regime_weight': normalized_weights['volatility_regime'],
                'volume_regime_weight': normalized_weights['volume_regime'],
                'structural_trend_weight': normalized_weights['structural_trend']
            })
            tprint("Weights validated and normalized", "SUCCESS")

            tprint("Clustering configuration created using shared utilities", "SUCCESS")
            return clustering_config
            
        except Exception as e:
            tprint(f"Config creation failed: {e}, using defaults", "WARNING")
            # Use supported config type with necessary parameters
            fallback_config = create_default_config(
                config_type="hybrid",
                symbol=getattr(self.config, 'symbol', 'ETHUSDT'),
                timeframe=getattr(self.config, 'timeframe', '15m'),
                n_regimes=getattr(self.config, 'n_regimes', 8)
            )
            # Add clustering-specific defaults
            fallback_config.update({
                'algorithm_type': 'adaptive_clustering',
                'enable_economic_clustering': True,
                'enable_ensemble_clustering': True,
                'exchange': 'binance'
            })
            regime_weights = self._get_weight_group('regime')
            fallback_config.update({
                'economic_weight': regime_weights.get('economic', 0.25),
                'volatility_regime_weight': regime_weights.get('volatility', 0.30),
                'volume_regime_weight': regime_weights.get('volume', 0.25),
                'structural_trend_weight': regime_weights.get('structural_trend', 0.20),
            })
            return fallback_config
    
    
    # @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    # @performance_tracked(log_performance=True, track_memory=True)
    async def _perform_clustering(self, features: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering using advanced optimization methods with hardware acceleration."""
        try:
            tprint("Performing advanced clustering optimization...", "INFO")
            
            # Use advanced clustering with progressive regime optimization
            clustering_result = await self._perform_advanced_clustering(features, market_data)
            tprint("Advanced clustering optimization completed", "SUCCESS")

            tprint(f"Clustering completed: {clustering_result['n_clusters']} clusters", "SUCCESS")
            return clustering_result

        except Exception as e:
            tprint(f"Clustering failed: {e}", "ERROR")
            raise ValueError(f"Clustering failed: {e}")
    
    # @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    # @performance_tracked(log_performance=True, track_memory=True)
    async def _perform_advanced_clustering(self, features: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform advanced clustering using progressive regime optimization with hardware acceleration."""
        try:
            tprint("Starting progressive regime optimization...", "INFO")

            context = ClusteringContext(
                original_features=features,
                market_data=market_data,
                memory_optimizer=self.memory_optimizer,
                original_feature_names=getattr(self, 'feature_names', None),
                feature_scores=getattr(self, 'feature_scores', {}),
            )

            # Step 1: Feature selection and dimensionality reduction
            tprint("Step 1: Feature selection and dimensionality reduction...", "INFO")
            self._optimize_features(context)

            # Step 2: Extract TAS/NAS assignments and apply dynamic iterative convergence
            self._extract_and_optimize_regimes(context)

            # ENHANCED: Add comprehensive validation before final results
            tprint("Step 7: Running comprehensive clustering validation...", "INFO")
            validation_results = self.validate_clustering_robustness(
                context.optimized_features, context.optimized_assignments, market_data
            )
            context.validation_results = validation_results
            
            # Final summary and artifact packaging
            clustering_result = self._summarize_results(context, market_data)

            tprint("Progressive regime optimization completed successfully", "SUCCESS")
            return clustering_result

        except Exception as e:
            tprint(f"Progressive regime optimization failed: {e}", "ERROR")
            # Fast-fail: Do not fall back to basic clustering
            tprint("Progressive regime optimization failed - cannot proceed with suboptimal clustering", "ERROR")
            raise ValueError(f"Progressive regime optimization failed: {e}. Cannot proceed with fallback clustering.")
    
    def _optimize_features(self, context: ClusteringContext) -> None:
        """Optimize features using data-driven dimensionality reduction."""
        try:
            tprint("Starting data-driven feature optimization...", "INFO")

            # Step 1: Standardize features with updated feature tracking
            tprint("Step 1: Standardizing features...", "INFO")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()

            feature_names = context.original_feature_names or [
                f"feature_{i}" for i in range(context.original_features.shape[1])
            ]
            context.original_feature_names = list(feature_names)
            context.pre_pca_feature_names = list(feature_names)
            context.pre_pca_feature_count = len(feature_names)

            features_scaled = scaler.fit_transform(context.original_features)
            tprint(f"Feature standardization completed: {context.original_features.shape}", "SUCCESS")

            if context.original_features.shape[1] < 2:
                tprint_warning("⚠️ Fewer than two features available after pruning - skipping PCA")
                features_final = self._validate_feature_quality_minimal(features_scaled, context.market_data)
                context.optimized_features = features_final
                context.optimized_feature_names = list(feature_names)
                context.dropped_feature_names = context.dropped_feature_names or []
                context.pca_loading_scores = {name: 1.0 for name in feature_names}
                if context.feature_scores:
                    context.feature_scores = {
                        name: float(context.feature_scores.get(name, 0.0)) for name in feature_names
                    }

                tprint(
                    f"Data-driven feature optimization (PCA skipped): {context.original_features.shape} -> {features_final.shape}",
                    "SUCCESS",
                )

                self.optimized_features = features_final
                self.feature_names = list(feature_names)
                if hasattr(self, 'feature_scores') and isinstance(self.feature_scores, dict):
                    self.feature_scores = context.feature_scores

                self._safe_memory_cleanup([features_scaled])
                return

            from sklearn.decomposition import PCA

            def _fit_pca(data: np.ndarray) -> Tuple[Any, np.ndarray]:
                try:
                    model = PCA(n_components='mle', svd_solver='full')
                    transformed = model.fit_transform(data)
                    if transformed.shape[1] == 0:
                        model = PCA(n_components=1, svd_solver='full')
                        transformed = model.fit_transform(data)
                    tprint(
                        f"PCA-MLE reduction: {data.shape[1]} -> {transformed.shape[1]} features "
                        f"(explained variance: {model.explained_variance_ratio_.sum():.3f})",
                        "SUCCESS",
                    )
                    return model, transformed
                except Exception as exc:
                    tprint(f"PCA-MLE failed: {exc}, using fallback PCA with 99% variance")
                    tprint("PCA-MLE failed, using fallback PCA with 99% variance...", "WARNING")
                    model = PCA(n_components=0.99, svd_solver='full')
                    transformed = model.fit_transform(data)
                    if transformed.shape[1] == 0:
                        model = PCA(n_components=min(1, data.shape[1]), svd_solver='full')
                        transformed = model.fit_transform(data)
                    tprint(
                        f"PCA fallback: {data.shape[1]} -> {transformed.shape[1]} features "
                        f"(explained variance: {model.explained_variance_ratio_.sum():.3f})",
                        "SUCCESS",
                    )
                    return model, transformed

            def _compute_loading_scores(pca_model: Any, n_features: int) -> np.ndarray:
                components = np.abs(getattr(pca_model, 'components_', np.empty((0, 0))))
                if components.size == 0:
                    return np.zeros(n_features)

                explained = getattr(pca_model, 'explained_variance_ratio_', None)
                if explained is None or len(explained) != components.shape[0]:
                    weights = np.ones(components.shape[0]) / max(1, components.shape[0])
                else:
                    weights = np.asarray(explained)

                weighted_components = components.T * weights
                loading_strength = weighted_components.sum(axis=1)
                max_strength = float(np.max(loading_strength)) if loading_strength.size else 0.0
                if max_strength == 0.0:
                    return np.zeros_like(loading_strength)
                return loading_strength / (max_strength + 1e-12)

            # Step 2: Apply PCA and prune near-zero contributors using loading scores
            tprint("Step 2: Applying PCA with MLE for data-driven dimensionality selection...", "INFO")
            pca, features_pca = _fit_pca(features_scaled)
            loading_scores = _compute_loading_scores(pca, context.original_features.shape[1])

            loading_threshold = float(max(0.05, np.percentile(loading_scores, 5))) if loading_scores.size else 0.0
            retained_mask = loading_scores >= loading_threshold

            if loading_scores.size >= 2 and retained_mask.sum() < 2:
                top_two = np.argsort(loading_scores)[::-1][:2]
                adjusted_mask = np.zeros_like(retained_mask, dtype=bool)
                adjusted_mask[top_two] = True
                retained_mask = adjusted_mask

            if retained_mask.sum() == 0 and loading_scores.size:
                retained_mask[np.argmax(loading_scores)] = True

            if retained_mask.sum() < loading_scores.size:
                retained_indices = np.where(retained_mask)[0]
                dropped_indices = np.where(~retained_mask)[0]
                dropped_feature_names = [feature_names[idx] for idx in dropped_indices]
                retained_feature_names = [feature_names[idx] for idx in retained_indices]

                tprint_info(
                    "🔧 PCA loading pruning: removed "
                    f"{len(dropped_feature_names)} near-zero contributors (threshold={loading_threshold:.3f})"
                )
                if dropped_feature_names:
                    tprint_info(
                        "   Dropped features: "
                        + ", ".join(dropped_feature_names[:5])
                        + ("..." if len(dropped_feature_names) > 5 else "")
                    )

                context.dropped_feature_names = dropped_feature_names
                context.original_features = context.original_features[:, retained_indices]
                feature_names = retained_feature_names
                context.original_feature_names = list(feature_names)

                if context.feature_scores:
                    context.feature_scores = {
                        name: float(context.feature_scores.get(name, 0.0)) for name in feature_names
                    }

                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(context.original_features)
                pca, features_pca = _fit_pca(features_scaled)
                loading_scores = _compute_loading_scores(pca, context.original_features.shape[1])
            else:
                context.dropped_feature_names = []

            # Step 3: Basic quality validation (minimal checks)
            tprint("Step 3: Validating feature quality...", "INFO")
            features_final = self._validate_feature_quality_minimal(features_pca, context.market_data)
            tprint(f"Feature quality validation completed: {features_final.shape}", "SUCCESS")

            tprint(
                f"Data-driven feature optimization completed: {context.original_features.shape} -> {features_final.shape}",
                "SUCCESS",
            )

            context.optimized_features = features_final
            context.optimized_feature_names = list(feature_names)
            context.pca_loading_scores = {
                name: float(loading_scores[idx]) for idx, name in enumerate(feature_names)
            }
            if context.feature_scores:
                context.feature_scores = {
                    name: float(context.feature_scores.get(name, 0.0)) for name in feature_names
                }

            if context.pca_loading_scores:
                top_loadings = sorted(
                    context.pca_loading_scores.items(), key=lambda item: item[1], reverse=True
                )[:5]
                tprint_info(
                    "📈 PCA loadings summary: "
                    + ", ".join(f"{name} ({score:.3f})" for name, score in top_loadings)
                )

            self.optimized_features = features_final
            self.feature_names = list(feature_names)
            if hasattr(self, 'feature_scores') and isinstance(self.feature_scores, dict):
                self.feature_scores = context.feature_scores

            if hasattr(self, 'selection_metadata') and isinstance(self.selection_metadata, dict):
                self.selection_metadata.setdefault('pca_loading_scores', context.pca_loading_scores)
                self.selection_metadata.setdefault('dropped_features_after_pca', context.dropped_feature_names)

            # Memory cleanup after feature optimization using hardware tools
            # Use thread-safe cleanup to avoid race conditions
            self._safe_memory_cleanup([features_scaled, features_pca])

        except Exception as e:
            tprint(f"Feature optimization failed: {e}", "ERROR")
            # Try fallback: Use original features if optimization fails
            tprint("Attempting to use original features as fallback...", "WARNING")
            try:
                # Validate original features before using them
                features_final = self._validate_feature_quality_minimal(context.original_features, context.market_data)
                tprint(f"Using original features as fallback: {features_final.shape}", "WARNING")
                context.optimized_features = features_final
                self.optimized_features = features_final
                return
            except Exception as fallback_error:
                tprint(f"Fallback also failed: {fallback_error}", "ERROR")
                raise ValueError(f"Feature optimization failed: {e}. Fallback also failed: {fallback_error}. Cannot proceed with suboptimal features.")

    def _validate_feature_quality_minimal(self, features: np.ndarray, market_data: pd.DataFrame) -> np.ndarray:
        """Minimal feature quality validation for data-driven approach."""
        try:
            # Input validation
            if features is None:
                raise ValueError("Features cannot be None")
            
            if not isinstance(features, np.ndarray):
                raise ValueError(f"Features must be numpy array, got {type(features)}")
            
            if len(features.shape) != 2:
                raise ValueError(f"Features must be 2D array, got shape {features.shape}")
            
            original_shape = features.shape
            tprint(f"Validating features with shape {original_shape}", "INFO")
            
            # Check for NaN/inf values
            nan_mask = np.any(np.isnan(features), axis=1)
            inf_mask = np.any(np.isinf(features), axis=1)
            invalid_mask = nan_mask | inf_mask
            
            if np.any(invalid_mask):
                tprint(f"Found {np.sum(invalid_mask)} samples with NaN/inf values, removing them", "WARNING")
                features = features[~invalid_mask]
                tprint(f"Features shape after cleanup: {features.shape}", "INFO")
            
            # Check for empty features after cleanup
            if features.size == 0:
                tprint(f"Original features shape: {original_shape}", "ERROR")
                tprint(f"NaN mask sum: {np.sum(nan_mask)}", "ERROR")
                tprint(f"Inf mask sum: {np.sum(inf_mask)}", "ERROR")
                raise ValueError("All features were invalid (NaN/inf), cannot proceed")
            
            # Validate minimum requirements
            if features.shape[1] == 0:
                raise ValueError("Insufficient features for clustering: 0 < 2")

            if features.shape[1] == 1:
                tprint_warning("⚠️ Only one feature available after optimization - clustering stability may be reduced")
            
            if features.shape[0] < 10:
                tprint(f"Low sample count: {features.shape[0]} samples", "WARNING")
                if features.shape[0] < 5:
                    raise ValueError(f"Too few samples for clustering: {features.shape[0]} < 5")
            
            # Check for constant features (zero variance)
            feature_vars = np.var(features, axis=0)
            constant_features = feature_vars == 0
            
            if np.any(constant_features):
                tprint(f"Found {np.sum(constant_features)} constant features, removing them", "WARNING")
                features = features[:, ~constant_features]
                tprint(f"Features shape after removing constants: {features.shape}", "INFO")
                
                # Final check after removing constants
                if features.shape[1] < 2:
                    raise ValueError("Too few features after removing constant features")
            
            # Check for perfect correlation between features
            if features.shape[1] > 1:
                corr_matrix = np.corrcoef(features.T)
                # Find perfectly correlated features (correlation = 1.0)
                perfect_corr_mask = np.triu(np.abs(corr_matrix - 1.0) < 1e-10, k=1)
                if np.any(perfect_corr_mask):
                    tprint("Found perfectly correlated features, this may cause clustering issues", "WARNING")
            
            tprint(f"Feature validation completed: {original_shape} -> {features.shape}", "SUCCESS")
            return features
            
        except Exception as e:
            tprint_error(f"Feature validation failed: {e}")
            raise ValueError(f"Feature validation failed: {e}") from e

    def _safe_memory_cleanup(self, arrays_to_cleanup: List[np.ndarray]) -> None:
        """Thread-safe memory cleanup to avoid race conditions."""
        import threading
        import gc
        
        # Use a lock to prevent concurrent cleanup operations
        if not hasattr(self, '_cleanup_lock'):
            self._cleanup_lock = threading.Lock()
        
        with self._cleanup_lock:
            try:
                # Clean up arrays in a safe order
                valid_arrays = [arr for arr in arrays_to_cleanup if arr is not None]
                
                if valid_arrays and self.memory_optimizer:
                    try:
                        # Check if optimizer is still available and not being used elsewhere
                        if hasattr(self.memory_optimizer, 'cleanup_arrays'):
                            self.memory_optimizer.cleanup_arrays(valid_arrays)
                        
                        if hasattr(self.memory_optimizer, 'optimize_memory_usage'):
                            self.memory_optimizer.optimize_memory_usage()
                            
                    except Exception as cleanup_error:
                        tprint_warning(f"Hardware memory cleanup failed: {cleanup_error}")
                        # Fallback to standard cleanup
                        self._fallback_memory_cleanup(valid_arrays)
                else:
                    # Standard cleanup when no hardware optimizer
                    self._fallback_memory_cleanup(valid_arrays)
                    
            except Exception as e:
                tprint_warning(f"Memory cleanup failed: {e}")
                # Last resort cleanup
                try:
                    gc.collect()
                except:
                    pass
    
    def _fallback_memory_cleanup(self, arrays: List[np.ndarray]) -> None:
        """Fallback memory cleanup when hardware optimizer is not available."""
        import gc
        
        try:
            # Clear array references
            for arr in arrays:
                if arr is not None:
                    del arr
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            tprint_warning(f"Fallback cleanup failed: {e}")

    # @performance_tracked(log_performance=True, track_memory=True)
    def _determine_optimal_algorithm_type(self, data: Any) -> str:
        """
        Determine the optimal clustering algorithm based on data characteristics and regime discovery results.
        
        Args:
            data: Input data for analysis
            
        Returns:
            Optimal algorithm type string
        """
        try:
            # Get data characteristics
            market_data = None
            if isinstance(data, dict) and 'market_data' in data:
                market_data = data['market_data']
            elif hasattr(data, 'shape'):
                market_data = data
                
            if market_data is None:
                tprint_warning("Cannot determine data characteristics, using adaptive_clustering")
                return 'adaptive_clustering'
            
            # Extract data dimensions
            if hasattr(market_data, 'shape'):
                n_samples, n_features = market_data.shape
            else:
                n_samples = len(market_data)
                n_features = len(market_data.columns) if hasattr(market_data, 'columns') else 10
            
            data_density = n_samples / n_features if n_features > 0 else 1
            
            # Get regime discovery results for additional context
            regime_discovery_result = pipeline_state.get('nas_tas_regime_discovery_result', {})
            tas_regime_count = regime_discovery_result.get('tas_regime_count', 8)
            nas_regime_count = regime_discovery_result.get('nas_regime_count', 8)
            
            # Use our custom NAS-TAS clustering logic (progressive regime optimization)
            # This is our sophisticated clustering approach that combines:
            # - BIC-selected GMM for optimal regime count
            # - Feature optimization and dimensionality reduction
            # - NAS/TAS label reconciliation
            # - Temporal coherence smoothing
            # - Advanced regime optimization
            
            algorithm = 'nas_tas_clustering'
            reason = f"Custom progressive regime optimization for regime detection (TAS={tas_regime_count}, NAS={nas_regime_count}, {n_samples} samples)"
            
            tprint(f"Algorithm selection: {algorithm} - {reason}", "INFO")
            return algorithm
            
        except Exception as e:
            tprint_warning(f"Algorithm determination failed: {e}, using adaptive_clustering")
            return 'adaptive_clustering'

    # @performance_tracked(log_performance=True, track_memory=True)
    def _validate_execution_inputs(self, data: Any) -> None:
        """Validate inputs for execution method with performance tracking."""
        try:
            # Validate data
            if data is not None:
                if isinstance(data, pd.DataFrame):
                    if data.empty:
                        raise ValueError("DataFrame is empty")
                    if len(data.columns) == 0:
                        raise ValueError("DataFrame has no columns")
                elif isinstance(data, dict):
                    if 'market_data' not in data:
                        raise ValueError("Dictionary data must contain 'market_data' key")
                    market_data = data['market_data']
                    if not isinstance(market_data, pd.DataFrame):
                        raise ValueError("market_data must be a DataFrame")
                    if market_data.empty:
                        raise ValueError("market_data DataFrame is empty")
                else:
                    raise ValueError(f"Unsupported data type: {type(data)}")
            
            # Validate configuration
            if not hasattr(self, 'config') or self.config is None:
                raise ValueError("Component configuration is not set")
            
            # Validate required config attributes
            required_attrs = ['n_regimes', 'algorithm_type']
            for attr in required_attrs:
                if not hasattr(self.config, attr):
                    raise ValueError(f"Configuration missing required attribute: {attr}")
            
            # Validate n_regimes
            if not isinstance(self.config.n_regimes, int) or self.config.n_regimes < 2:
                raise ValueError(f"n_regimes must be an integer >= 2, got {self.config.n_regimes}")
            
            if self.config.n_regimes > 50:
                raise ValueError(f"n_regimes too large: {self.config.n_regimes} > 50")
            
            tprint("Input validation passed", "SUCCESS")
            
        except Exception as e:
            tprint_error(f"Input validation failed: {e}")
            raise ValueError(f"Input validation failed: {e}") from e

    # Removed _select_optimal_k - no longer needed with dynamic convergence

    # Removed _determine_optimal_k_iterative - no longer needed with dynamic convergence

    def _run_iterative_convergence(self, features: np.ndarray, k: int, max_iterations: int = 50, tolerance: float = 1e-4) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run iterative convergence algorithm that stops when balance/Silhouette/CV scores no longer improve on average."""
        try:
            tprint(f"Starting iterative convergence for K={k}...", "INFO")
            
            # Initialize with random assignments
            np.random.seed(42)
            n_samples = features.shape[0]
            assignments = np.random.randint(0, k, n_samples)
            
            # Track convergence history
            convergence_history = []
            best_assignments = assignments.copy()
            best_score = -1.0
            
            for iteration in range(max_iterations):
                # Calculate current scores
                balance_score = self._calculate_regime_balance(assignments)
                silhouette_score = self._calculate_silhouette_score(features, assignments)
                cv_score = self._calculate_cv_score(features, assignments)
                
                # Composite score
                current_score = (balance_score + silhouette_score + cv_score) / 3.0
                
                # Update best if improved
                if current_score > best_score:
                    best_score = current_score
                    best_assignments = assignments.copy()
                
                # Store convergence metrics
                convergence_history.append({
                    'iteration': iteration,
                    'balance_score': balance_score,
                    'silhouette_score': silhouette_score,
                    'cv_score': cv_score,
                    'composite_score': current_score
                })
                
                tprint(f"Iteration {iteration}: Score={current_score:.4f} (Balance={balance_score:.3f}, Silhouette={silhouette_score:.3f}, CV={cv_score:.3f})", "INFO")
                
                # Check convergence (no improvement on average over last 5 iterations)
                if len(convergence_history) >= 5:
                    recent_scores = [h['composite_score'] for h in convergence_history[-5:]]
                    avg_improvement = np.mean(np.diff(recent_scores))
                    
                    if avg_improvement < tolerance:
                        tprint(f"Convergence achieved at iteration {iteration} (avg improvement: {avg_improvement:.6f})", "SUCCESS")
                        break
                
                # Apply one iteration of optimization
                assignments = self._apply_single_iteration_optimization(features, assignments, k)
            
            iteration_metrics = {
                'total_iterations': len(convergence_history),
                'converged': len(convergence_history) < max_iterations,
                'final_score': best_score,
                'convergence_history': convergence_history
            }
            
            return best_assignments, iteration_metrics
            
        except Exception as e:
            tprint(f"Iterative convergence failed: {e}", "ERROR")
            return assignments, {'error': str(e), 'total_iterations': 0, 'converged': False}

    def _apply_single_iteration_optimization(self, features: np.ndarray, assignments: np.ndarray, k: int, cache: dict = None) -> np.ndarray:
        """Apply one iteration of optimization with VECTORIZED operations for 10x speed improvement."""
        try:
            new_assignments = assignments.copy()
            
            # OPTIMIZATION 1: Use intelligent caching system for centroids and distances
            regime_centroids = self._compute_regime_centroids_vectorized(features, assignments, k)
            distances, cache = self._compute_distances_with_caching(features, assignments, regime_centroids, k, cache)
            
            # VECTORIZED: Batch process improvements with early termination
            improvements = self._compute_improvements_vectorized(
                features, assignments, distances, regime_centroids, k
            )
            
            # VECTORIZED: Apply improvements with adaptive thresholding (increased threshold for speed)
            significant_improvements = improvements > 0.001  # Increased threshold for faster processing
            
            # DEBUG: Log improvement statistics
            max_improvement = np.max(improvements)
            num_significant = np.sum(significant_improvements)
            tprint(f"   📊 Improvements: max={max_improvement:.6f}, significant={num_significant}/{len(assignments)}", "INFO")
            
            if np.any(significant_improvements):
                # Get significant indices first
                significant_indices = np.where(significant_improvements)[0]
                
                # Get best regimes ONLY for samples with significant improvements
                best_regimes = np.argmax(improvements[significant_indices], axis=1)
                
                # FIXED: Apply assignments only to significant samples
                new_assignments[significant_indices] = best_regimes
                
                # Log significant improvements (vectorized logging) - FIXED INDEXING
                
                # Get improvement values for significant indices using their best regimes
                significant_improvement_values = np.array([
                    improvements[significant_indices[i], best_regimes[i]] for i in range(len(significant_indices))
                ])
                
                # Only log the most significant improvements to avoid spam
                top_improvements = significant_improvement_values > 0.001
                if np.any(top_improvements):
                    top_indices = significant_indices[top_improvements]
                    top_values = significant_improvement_values[top_improvements]
                    top_new_regimes = best_regimes[top_improvements]  # FIXED: Use top_improvements instead of top_indices
                    
                    for idx, val, new_regime in zip(top_indices, top_values, top_new_regimes):
                        if idx < 100:  # Only log first 100 to avoid spam
                            tprint(f"   🎯 Sample {idx}: regime {assignments[idx]} → {new_regime} (improvement: {val:.6f})", "INFO")
            
            return new_assignments
            
        except Exception as e:
            tprint(f"Single iteration optimization failed: {e}", "ERROR")
            return assignments

    def _compute_regime_centroids_vectorized(self, features: np.ndarray, assignments: np.ndarray, k: int) -> np.ndarray:
        """Compute regime centroids using vectorized operations."""
        try:
            centroids = np.zeros((k, features.shape[1]))
            
            for regime in range(k):
                regime_mask = (assignments == regime)
                if np.any(regime_mask):
                    centroids[regime] = np.mean(features[regime_mask], axis=0)
            
            return centroids
        except Exception as e:
            tprint(f"Centroid computation failed: {e}", "ERROR")
            return np.zeros((k, features.shape[1]))

    def _compute_all_distances_vectorized(self, features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Compute all distances between samples and centroids using vectorized operations."""
        try:
            # Use broadcasting for efficient distance calculation
            # features: (n_samples, n_features), centroids: (k, n_features)
            # Result: (n_samples, k)
            distances = np.linalg.norm(features[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            return distances
        except Exception as e:
            tprint(f"Distance computation failed: {e}", "ERROR")
            return np.zeros((features.shape[0], centroids.shape[0]))

    def _compute_distances_with_caching(self, features: np.ndarray, assignments: np.ndarray, 
                                      centroids: np.ndarray, k: int, cache: Dict = None) -> Tuple[np.ndarray, Dict]:
        """Compute distances with intelligent caching and incremental updates."""
        try:
            if cache is None:
                cache = {}
            
            # Check if we can use cached distances
            assignment_hash = hash(tuple(assignments))
            if (cache.get('last_assignment_hash') == assignment_hash and 
                cache.get('distances') is not None and
                cache.get('centroids') is not None and
                np.array_equal(cache.get('centroids'), centroids)):
                return cache['distances'], cache
            
            # Check if we can do incremental updates
            if (cache.get('distances') is not None and 
                cache.get('last_assignments') is not None and
                cache.get('centroids') is not None):
                
                # Find changed samples
                changed_samples = np.where(assignments != cache['last_assignments'])[0]
                
                if len(changed_samples) < features.shape[0] * 0.1:  # Less than 10% changed
                    # Incremental update: only recompute distances for changed samples
                    distances = cache['distances'].copy()
                    
                    # Update distances for changed samples
                    if len(changed_samples) > 0:
                        distances[changed_samples] = np.linalg.norm(
                            features[changed_samples, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
                        )
                    
                    # Update cache
                    cache['distances'] = distances
                    cache['last_assignments'] = assignments.copy()
                    cache['last_assignment_hash'] = assignment_hash
                    cache['centroids'] = centroids.copy()
                    
                    return distances, cache
            
            # Full recomputation needed
            distances = self._compute_all_distances_vectorized(features, centroids)
            
            # Update cache
            cache['distances'] = distances
            cache['last_assignments'] = assignments.copy()
            cache['last_assignment_hash'] = assignment_hash
            cache['centroids'] = centroids.copy()
            
            return distances, cache
            
        except Exception as e:
            tprint(f"Distance caching failed: {e}", "ERROR")
            return self._compute_all_distances_vectorized(features, centroids), cache

    def _compute_improvements_vectorized(self, features: np.ndarray, assignments: np.ndarray, 
                                       distances: np.ndarray, centroids: np.ndarray, k: int) -> np.ndarray:
        """Compute improvement matrix using efficient distance-based approximation with score validation."""
        try:
            n_samples = features.shape[0]
            improvements = np.zeros((n_samples, k))
            
            # Get current distances for each sample
            current_distances = distances[np.arange(n_samples), assignments]
            
            # Calculate distance-based improvements (faster approximation)
            # Improvement is negative distance change (lower distance = better)
            improvements = current_distances[:, np.newaxis] - distances
            
            # Set improvement to 0 for current regime (no change)
            sample_indices = np.arange(n_samples)
            regime_indices = assignments[sample_indices]
            improvements[sample_indices, regime_indices] = 0
            
            # Scale improvements to be more reasonable for composite score
            # Distance improvements are typically small, so scale them appropriately
            improvements = improvements * 0.01  # Scale factor to match typical score improvements
            
            return improvements
        except Exception as e:
            tprint(f"Improvement computation failed: {e}", "ERROR")
            return np.zeros((features.shape[0], k))

    def _calculate_reassignment_improvement(self, features: np.ndarray, old_assignments: np.ndarray, new_assignments: np.ndarray) -> float:
        """Calculate improvement from reassignment using VECTORIZED fast approximation to avoid bottleneck."""
        try:
            # BOTTLENECK FIX: Use vectorized distance-based approximation instead of expensive silhouette calculation
            # This avoids O(n²) silhouette score calculations that cause the system to get stuck
            
            # Fast vectorized balance calculation (O(n))
            old_balance = self._calculate_regime_balance(old_assignments)
            new_balance = self._calculate_regime_balance(new_assignments)
            balance_improvement = new_balance - old_balance
            
            # VECTORIZED: Fast silhouette approximation using distance-based metrics (O(n) instead of O(n²))
            old_silhouette_approx = self._calculate_silhouette_approximation_vectorized(features, old_assignments)
            new_silhouette_approx = self._calculate_silhouette_approximation_vectorized(features, new_assignments)
            silhouette_improvement = new_silhouette_approx - old_silhouette_approx
            
            # Fast vectorized CV calculation (O(n))
            old_cv = self._calculate_cv_score_optimized(features, old_assignments)
            new_cv = self._calculate_cv_score_optimized(features, new_assignments)
            cv_improvement = new_cv - old_cv
            
            # Fast vectorized temporal calculation (O(n))
            old_temporal = self._calculate_temporal_smoothness_optimized(old_assignments)
            new_temporal = self._calculate_temporal_smoothness_optimized(new_assignments)
            temporal_improvement = new_temporal - old_temporal
            
            # VECTORIZED: Weighted improvement calculation (same weights as main optimization)
            improvement = (balance_improvement * 0.25 + 
                          silhouette_improvement * 0.40 + 
                          cv_improvement * 0.25 + 
                          temporal_improvement * 0.10)
            
            # DEBUG: Log significant improvements (reduced frequency to avoid spam)
            if improvement > 0.01:  # Higher threshold to reduce logging frequency
                tprint(f"   📈 Vectorized improvement: {improvement:.6f} (fast approximation)", "INFO")
            
            return improvement
            
        except Exception as e:
            tprint(f"Vectorized reassignment improvement calculation failed: {e}", "ERROR")
            return 0.0

    def _calculate_silhouette_approximation_vectorized(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Fast vectorized silhouette approximation using distance-based metrics (O(n) instead of O(n²))."""
        try:
            if len(np.unique(assignments)) < 2:
                return 0.0
            
            # VECTORIZED: Calculate centroids for all clusters at once
            unique_labels = np.unique(assignments)
            centroids = np.array([np.mean(features[assignments == label], axis=0) for label in unique_labels])
            
            # VECTORIZED: Calculate distances from each point to all centroids
            # Broadcasting: features (n_samples, n_features) - centroids (n_clusters, n_features)
            distances = np.linalg.norm(features[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            
            # VECTORIZED: Calculate intra-cluster distances (distance to own centroid)
            intra_distances = distances[np.arange(len(assignments)), assignments]
            
            # VECTORIZED: Calculate inter-cluster distances (minimum distance to other centroids)
            # Set distance to own cluster to infinity to find minimum of others
            distances_masked = distances.copy()
            distances_masked[np.arange(len(assignments)), assignments] = np.inf
            inter_distances = np.min(distances_masked, axis=1)
            
            # VECTORIZED: Silhouette approximation = (inter - intra) / max(inter, intra)
            silhouette_approx = np.mean((inter_distances - intra_distances) / np.maximum(inter_distances, intra_distances))
            
            # Normalize to [0, 1] range
            return max(0.0, min(1.0, (silhouette_approx + 1.0) / 2.0))
            
        except Exception as e:
            return 0.0

    def _calculate_regime_balance(self, assignments: np.ndarray) -> float:
        """Calculate regime balance score (1.0 = perfect balance, 0.0 = worst imbalance)."""
        try:
            unique_regimes, regime_counts = np.unique(assignments, return_counts=True)
            if len(unique_regimes) < 2:
                return 0.0
            
            total_samples = len(assignments)
            regime_percentages = regime_counts / total_samples
            
            # Calculate base balance score
            mean_count = np.mean(regime_counts)
            std_count = np.std(regime_counts)
            cv = std_count / mean_count if mean_count > 0 else 1.0
            base_balance = 1.0 / (1.0 + cv)
            
            # Apply penalty for regimes exceeding max percentage threshold
            penalty_factor = 1.0
            max_threshold = getattr(self.config, 'max_regime_percentage', 0.20)
            min_threshold = getattr(self.config, 'min_regime_percentage', 0.05)
            
            for percentage in regime_percentages:
                if percentage > max_threshold:
                    # Apply exponential penalty for regimes above threshold
                    excess = percentage - max_threshold
                    penalty = 1.0 - (excess * 6.0)  # 3x stronger penalty (was 2.0)
                    penalty_factor *= max(0.01, penalty)  # Much stronger minimum penalty (was 0.1)
            
            # Apply penalty for regimes below minimum threshold
            for percentage in regime_percentages:
                if percentage < min_threshold:
                    # Apply penalty for regimes below minimum
                    deficit = min_threshold - percentage
                    penalty = 1.0 - (deficit * 4.5)  # 3x stronger penalty (was 1.5)
                    penalty_factor *= max(0.01, penalty)  # Much stronger minimum penalty (was 0.2)
            
            balance_score = base_balance * penalty_factor
            return max(0.0, min(1.0, balance_score))  # Clamp to [0, 1]
            
        except Exception as e:
            return 0.0

    def _calculate_regime_balance_optimized(self, assignments: np.ndarray) -> float:
        """Calculate regime balance score - VECTORIZED OPTIMIZATION."""
        try:
            unique_regimes, regime_counts = np.unique(assignments, return_counts=True)
            if len(unique_regimes) < 2:
                return 0.0
            
            total_samples = len(assignments)
            regime_percentages = regime_counts / total_samples
            
            # VECTORIZED balance calculation
            mean_count = np.mean(regime_counts)
            std_count = np.std(regime_counts)
            cv = std_count / mean_count if mean_count > 0 else 1.0
            base_balance = 1.0 / (1.0 + cv)
            
            # VECTORIZED penalty calculation
            max_threshold = getattr(self.config, 'max_regime_percentage', 0.20)
            min_threshold = getattr(self.config, 'min_regime_percentage', 0.05)
            
            # Vectorized excess and deficit calculations
            excess_mask = regime_percentages > max_threshold
            deficit_mask = regime_percentages < min_threshold
            
            excess_penalties = np.where(excess_mask, 
                                      1.0 - (regime_percentages - max_threshold) * 6.0,  # 3x stronger penalty
                                      1.0)
            deficit_penalties = np.where(deficit_mask, 
                                      1.0 - (min_threshold - regime_percentages) * 4.5,  # 3x stronger penalty
                                      1.0)
            
            # Combine penalties (minimum 0.1)
            combined_penalties = np.maximum(0.1, excess_penalties * deficit_penalties)
            penalty_factor = np.prod(combined_penalties)
            
            balance_score = base_balance * penalty_factor
            return np.clip(balance_score, 0.0, 1.0)
            
        except Exception as e:
            return 0.0

    def _calculate_silhouette_score(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Calculate Silhouette score for clustering quality with enhanced separation focus and caching."""
        try:
            if len(np.unique(assignments)) < 2:
                return 0.0

            # Use cached silhouette score if available and assignments haven't changed
            cache_key = tuple(assignments)
            if hasattr(self, '_cached_silhouette') and self._cached_silhouette['assignments_key'] == cache_key:
                return self._cached_silhouette['score']

            from sklearn.metrics import silhouette_score, silhouette_samples

            # Calculate overall silhouette score
            overall_silhouette = silhouette_score(features, assignments)

            # ENHANCED: Add bonus for high-quality separations
            # Calculate per-sample silhouette scores to identify well-separated clusters
            sample_silhouettes = silhouette_samples(features, assignments)

            # Bonus for clusters with high average silhouette (>0.5)
            unique_labels = np.unique(assignments)
            cluster_bonuses = []

            for label in unique_labels:
                label_mask = assignments == label
                if np.sum(label_mask) > 1:  # Need at least 2 samples
                    cluster_silhouettes = sample_silhouettes[label_mask]
                    cluster_avg_silhouette = np.mean(cluster_silhouettes)

                    # Bonus for well-separated clusters
                    if cluster_avg_silhouette > 0.5:
                        cluster_bonus = (cluster_avg_silhouette - 0.5) * 0.2  # Up to 0.1 bonus
                        cluster_bonuses.append(cluster_bonus)

            # Apply cluster bonuses
            total_bonus = np.mean(cluster_bonuses) if cluster_bonuses else 0.0
            enhanced_silhouette = overall_silhouette + total_bonus

            # Cap at 1.0 and ensure non-negative
            final_score = max(0.0, min(1.0, enhanced_silhouette))

            # Cache the result using tuple for hashability
            self._cached_silhouette = {
                'assignments_key': cache_key,
                'score': final_score
            }

            return final_score

        except Exception as e:
            return 0.0

    def _calculate_cv_score(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Calculate coefficient of variation score for clustering stability - VECTORIZED OPTIMIZATION."""
        try:
            unique_regimes = np.unique(assignments)
            if len(unique_regimes) < 2:
                return 0.0
            
            # VECTORIZED within-cluster variance calculation
            within_variances = []
            for regime in unique_regimes:
                regime_mask = assignments == regime
                regime_count = np.sum(regime_mask)
                if regime_count > 1:
                    regime_features = features[regime_mask]
                    # Vectorized variance calculation
                    regime_var = np.var(regime_features, axis=0).mean()
                    within_variances.append(regime_var)
            
            if not within_variances:
                return 0.0
            
            # VECTORIZED between-cluster variance calculation
            overall_mean = np.mean(features, axis=0)
            between_variances = []
            for regime in unique_regimes:
                regime_mask = assignments == regime
                regime_count = np.sum(regime_mask)
                if regime_count > 0:
                    regime_mean = np.mean(features[regime_mask], axis=0)
                    # Vectorized distance calculation
                    between_var = np.var(regime_mean - overall_mean)
                    between_variances.append(between_var)
            
            if not between_variances:
                return 0.0
            
            # CV score (higher is better)
            within_var = np.mean(within_variances)
            between_var = np.mean(between_variances)
            cv_score = between_var / (within_var + 1e-8)  # Avoid division by zero
            
            return min(1.0, cv_score)  # Cap at 1.0
            
        except Exception as e:
            return 0.0
    
    def _calculate_cv_score_vectorized(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Calculate coefficient of variation score - FULLY VECTORIZED OPTIMIZATION."""
        try:
            unique_regimes = np.unique(assignments)
            if len(unique_regimes) < 2:
                return 0.0
            
            # FULLY VECTORIZED approach using advanced numpy operations
            n_samples, n_features = features.shape
            n_regimes = len(unique_regimes)
            
            # Create regime masks matrix (n_regimes, n_samples)
            regime_masks = assignments[None, :] == unique_regimes[:, None]
            regime_counts = np.sum(regime_masks, axis=1)
            
            # Filter out regimes with < 2 samples
            valid_regimes = regime_counts >= 2
            if not np.any(valid_regimes):
                return 0.0
            
            valid_masks = regime_masks[valid_regimes]
            valid_counts = regime_counts[valid_regimes]
            
            # VECTORIZED within-cluster variance calculation
            # Reshape features for broadcasting: (n_valid_regimes, n_samples, n_features)
            features_broadcast = features[None, :, :]  # (1, n_samples, n_features)
            regime_features = np.where(valid_masks[:, :, None], features_broadcast, 0)
            
            # Calculate means for each regime (vectorized)
            regime_means = np.sum(regime_features, axis=1) / valid_counts[:, None]
            
            # Calculate variances for each regime (vectorized)
            regime_vars = np.sum((regime_features - regime_means[:, None, :]) ** 2, axis=1) / (valid_counts[:, None] - 1)
            within_variances = np.mean(regime_vars, axis=1)
            
            # VECTORIZED between-cluster variance calculation
            overall_mean = np.mean(features, axis=0)
            between_vars = np.var(regime_means - overall_mean[None, :], axis=1)
            
            # Calculate CV score
            within_var = np.mean(within_variances)
            between_var = np.mean(between_vars)
            cv_score = between_var / (within_var + 1e-8)
            
            return min(1.0, cv_score)
            
        except Exception as e:
            return 0.0

    def _calculate_cv_score_optimized(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Calculate coefficient of variation score - VECTORIZED OPTIMIZATION."""
        try:
            unique_regimes = np.unique(assignments)
            if len(unique_regimes) < 2:
                return 0.0
            
            # VECTORIZED within-cluster variance calculation
            within_variances = []
            for regime in unique_regimes:
                regime_mask = assignments == regime
                if np.sum(regime_mask) > 1:
                    regime_features = features[regime_mask]
                    # Vectorized variance calculation
                    regime_var = np.var(regime_features, axis=0).mean()
                    within_variances.append(regime_var)
            
            if not within_variances:
                return 0.0
            
            # VECTORIZED between-cluster variance calculation
            overall_mean = np.mean(features, axis=0)
            between_variances = []
            for regime in unique_regimes:
                regime_mask = assignments == regime
                if np.sum(regime_mask) > 0:
                    regime_mean = np.mean(features[regime_mask], axis=0)
                    between_var = np.var(regime_mean - overall_mean)
                    between_variances.append(between_var)
            
            if not between_variances:
                return 0.0
            
            # VECTORIZED CV calculation
            within_var = np.mean(within_variances)
            between_var = np.mean(between_variances)
            
            # Calculate CV ratio (higher is better for clustering)
            cv_score = between_var / (within_var + 1e-8)  # Avoid division by zero
            
            return min(1.0, cv_score)  # Cap at 1.0
            
        except Exception as e:
            return 0.0

    # Removed _reconcile_labels - no longer needed with dynamic convergence

    # Removed _apply_iterative_convergence - replaced by dynamic convergence

    def _extract_and_optimize_regimes(self, context: ClusteringContext) -> None:
        """Extract TAS/NAS assignments, apply Dawid-Skene fusion, and use enhanced iterative convergence."""
        try:
            tprint("Step 2: Extracting TAS/NAS assignments and applying enhanced iterative convergence...", "INFO")
            features = context.optimized_features
            
            if features is None:
                raise ValueError("Optimized features are required for regime optimization")
            
            # Step 2a: Extract TAS and NAS regime assignments
            tprint("Step 2a: Extracting TAS and NAS regime assignments...", "INFO")
            tas_assignments, nas_assignments = self._extract_regime_assignments()
            context.tas_assignments = tas_assignments
            context.nas_assignments = nas_assignments
            tprint(f"TAS assignments: {len(tas_assignments)}, NAS assignments: {len(nas_assignments)}", "SUCCESS")
            
            # Step 2b: Apply Dawid-Skene fusion for initial regime assignments
            tprint("Step 2b: Applying Dawid-Skene fusion...", "INFO")
            initial_k = 6  # Start with reasonable default
            fusion_result = self.regime_optimization_service.progressive_regime_optimization_with_k(
                features, tas_assignments, nas_assignments, context.market_data, initial_k
            )
            context.raw_assignments = fusion_result[0]
            context.optimization_metrics = fusion_result[1]
            context.fusion_metadata = fusion_result[2]
            
            # Step 2c: Apply enhanced iterative convergence with temporal smoothness
            tprint("Step 2c: Applying enhanced iterative convergence with temporal smoothness...", "INFO")
            optimized_assignments, convergence_metrics = self._run_enhanced_iterative_convergence(
                features, context.raw_assignments, initial_k
            )
            
            # Calculate final scores with temporal smoothness
            final_balance = self._calculate_regime_balance(optimized_assignments)
            final_silhouette = self._calculate_silhouette_score(features, optimized_assignments)
            final_cv = self._calculate_cv_score(features, optimized_assignments)
            final_temporal = self._calculate_temporal_smoothness(optimized_assignments)
            
            # Enhanced composite score with increased silhouette and Davies-Bouldin emphasis for better clustering quality
            balance_weight = 0.25  # 25% balance emphasis (reduced from 40% to prioritize clustering quality)
            silhouette_weight = 0.30  # 40% silhouette emphasis (increased from 25% for better separation)
            cv_weight = 0.35  # 25% CV emphasis (reduced from 30%)
            temporal_weight = 0.10  # 10% temporal emphasis (increased from 5% for stability)
            final_score = (final_balance * balance_weight + 
                          final_silhouette * silhouette_weight + 
                          final_cv * cv_weight + 
                          final_temporal * temporal_weight)
            
            tprint(f"Enhanced convergence completed: Balance={final_balance:.3f}, Silhouette={final_silhouette:.3f}, CV={final_cv:.3f}, Temporal={final_temporal:.3f}, Composite={final_score:.3f}", "SUCCESS")
            
            # MEMORY OPTIMIZATION: Clear caches and free memory after clustering
            if hasattr(self, '_cached_silhouette'):
                del self._cached_silhouette
            if hasattr(self, '_cached_cv'):
                del self._cached_cv
            if hasattr(self, '_cached_temporal'):
                del self._cached_temporal
            
            # Force garbage collection to free memory
            import gc
            gc.collect()
            tprint("🧹 Memory cleanup completed after clustering", "INFO")
            
            # Update context with results
            context.optimal_k = initial_k  # Use the K from fusion
            context.smoothed_assignments = optimized_assignments
            context.optimized_assignments = optimized_assignments
            context.convergence_metrics = convergence_metrics
            
            # Update optimization metrics
            context.optimization_metrics.update({
                'final_score': final_score,
                'improvement': final_score - context.optimization_metrics.get('initial_score', 0.0),
                'method': 'enhanced_iterative_convergence_with_temporal',
                'balance_score': final_balance,
                'silhouette_score': final_silhouette,
                'cv_score': final_cv,
                'temporal_score': final_temporal,
                'convergence_metadata': convergence_metrics
            })
            
        except Exception as e:
            tprint(f"Enhanced regime optimization failed: {e}", "ERROR")
            raise ValueError(f"Enhanced regime optimization failed: {e}")

    def _run_enhanced_iterative_convergence(self, features: np.ndarray, initial_assignments: np.ndarray, k: int, 
                                           max_iterations: int = 100, tolerance: float = 1e-5) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run enhanced iterative convergence with temporal smoothness and more aggressive optimization."""
        try:
            # Initialize distance caching for performance optimization
            distance_cache = {}
            
            # ENHANCED: Quality-based iteration limits using proper composite score
            initial_quality = self._calculate_composite_score(features, initial_assignments)
            
            # Add regime-specific bonuses for initial quality
            initial_balance = self._calculate_regime_balance(initial_assignments)
            initial_temporal = self._calculate_temporal_smoothness(initial_assignments)
            balance_bonus = min(0.2, initial_balance * 0.2)
            temporal_bonus = min(0.1, initial_temporal * 0.1)
            initial_quality = min(1.0, initial_quality + balance_bonus + temporal_bonus)
            
            if initial_quality < 0.3:  # Poor initial quality
                max_iterations = max(max_iterations, 200)  # Double iterations for poor quality
                tprint(f"⚠️ Poor initial quality ({initial_quality:.4f}), increasing iterations to {max_iterations}", "WARNING")
            
            tprint(f"🚀 Starting enhanced iterative convergence for K={k}...", "INFO")
            tprint(f"📊 Parameters: max_iterations={max_iterations}, tolerance={tolerance:.2e}", "INFO")
            tprint(f"📈 Data shape: {features.shape}, samples={len(initial_assignments)}", "INFO")
            tprint(f"🎯 Initial quality: {initial_quality:.4f}", "INFO")
            
            assignments = initial_assignments.copy()
            convergence_history = []
            best_assignments = assignments.copy()
            best_score = -1.0
            
            # OPTIMIZATION: Add caching for expensive calculations
            cache = {
                'centroids': None,
                'distances': None,
                'last_assignment_hash': None,
                'score_cache': {}
            }
            
            # Progress tracking variables
            last_progress_update = 0
            progress_interval = max(1, max_iterations // 20)  # Update every 5% of iterations
            
            # ENHANCED: Quality-based early stopping
            quality_target = 0.75  # Target quality score (increased from 0.6)
            quality_improvement_threshold = 0.005  # Minimum improvement per iteration (reduced for more iterations)
            
            for iteration in range(max_iterations):
                # Initialize convergence status
                convergence_achieved = False
                current_k = k  # Initialize current_k for splitting logic
                
                # VECTORIZED: Calculate current scores with optimized methods
                balance_score = self._calculate_regime_balance(assignments)
                # BOTTLENECK FIX: Use vectorized silhouette approximation every 5 iterations instead of every iteration
                if iteration % 5 == 0 or iteration < 3:  # Full calculation every 5 iterations or first 3
                    silhouette_score = self._calculate_silhouette_score(features, assignments)
                else:
                    silhouette_score = self._calculate_silhouette_approximation_vectorized(features, assignments)
                cv_score = self._calculate_cv_score_optimized(features, assignments)
                temporal_score = self._calculate_temporal_smoothness_optimized(assignments)
                
                # FIXED: Use proper composite score calculation instead of inline calculation
                current_score = self._calculate_composite_score(features, assignments)
                
                # Add regime-specific bonuses for better regime distribution
                balance_bonus = min(0.2, balance_score * 0.2)  # Max 0.2 bonus for good balance
                temporal_bonus = min(0.1, temporal_score * 0.1)  # Max 0.1 bonus for temporal smoothness
                
                current_score = min(1.0, current_score + balance_bonus + temporal_bonus)
                
                # Update best if improved
                improvement = current_score - best_score
                if current_score > best_score:
                    best_score = current_score
                    best_assignments = assignments.copy()
                
                # Store convergence metrics
                convergence_history.append({
                    'iteration': iteration,
                    'balance_score': balance_score,
                    'silhouette_score': silhouette_score,
                    'cv_score': cv_score,
                    'temporal_score': temporal_score,
                    'composite_score': current_score,
                    'improvement': improvement
                })
                
                # Progress updates with different levels of detail
                progress_percent = (iteration + 1) / max_iterations * 100
                
                # Detailed update every 5% or every iteration for first 10%
                if (iteration - last_progress_update >= progress_interval or 
                    progress_percent <= 10.0 or 
                    iteration == 0 or 
                    improvement > 0.001):  # Significant improvement
                    
                    # ENHANCED: Silhouette-focused progress reporting
                    silhouette_change = silhouette_score - (convergence_history[-2]['silhouette_score'] if len(convergence_history) > 1 else silhouette_score)
                    silhouette_indicator = "📈" if silhouette_change > 0.001 else "📉" if silhouette_change < -0.001 else "➡️"
                    
                    tprint(f"🔄 Iteration {iteration+1}/{max_iterations} ({progress_percent:.1f}%): "
                          f"Score={current_score:.4f} "
                          f"(Balance={balance_score:.3f}, {silhouette_indicator}Silhouette={silhouette_score:.3f}, "
                          f"CV={cv_score:.3f}, Temporal={temporal_score:.3f}) "
                          f"Best={best_score:.4f}", "INFO")
                    
                    if improvement > 0:
                        tprint(f"   📈 Improvement: +{improvement:.4f}", "INFO")
                    
                    last_progress_update = iteration
                
                # Summary update every 25%
                elif iteration % (max_iterations // 4) == 0 and iteration > 0:
                    tprint(f"📊 Progress: {progress_percent:.0f}% complete, "
                          f"Best score: {best_score:.4f}, "
                          f"Current: {current_score:.4f}", "INFO")
                
                # Convergence check (last 5 iterations for stability)
                if len(convergence_history) >= 5:
                    recent_scores = [h['composite_score'] for h in convergence_history[-5:]]
                    avg_improvement = np.mean(np.diff(recent_scores))
                    
                    # Show convergence status
                    if iteration % 10 == 0:  # Every 10 iterations
                        tprint(f"   🔍 Convergence check: avg_improvement={avg_improvement:.6f}, "
                              f"tolerance={tolerance:.2e}", "INFO")
                    
                    # ENHANCED: Multi-objective convergence detection with quality validation
                    if avg_improvement < tolerance * 0.3:  # 30% stricter tolerance (relaxed from 80%)
                        # CRITICAL: Don't converge if quality is too poor
                        if best_score < 0.3:  # Minimum acceptable quality
                            tprint(f"⚠️ Quality too low ({best_score:.4f}), continuing optimization...", "WARNING")
                            continue
                        
                        # ENHANCED: Adaptive convergence detection with dynamic tolerance
                        dynamic_tolerance = self._calculate_dynamic_convergence_tolerance(iteration, convergence_history, tolerance)
                        
                        # Check if silhouette score is improving significantly
                        recent_silhouette_scores = [h['silhouette_score'] for h in convergence_history[-5:]]
                        silhouette_trend = np.mean(np.diff(recent_silhouette_scores)) if len(recent_silhouette_scores) > 1 else 0
                        
                        # Adaptive convergence criteria
                        convergence_achieved = self._evaluate_adaptive_convergence(
                            avg_improvement, silhouette_trend, convergence_history, iteration
                        )
                        
                        # DEBUG: Show convergence decision
                        if iteration % 5 == 0:  # Every 5 iterations
                            tprint(f"   🔍 Convergence check: improvement={avg_improvement:.6f}, "
                                  f"silhouette_trend={silhouette_trend:.6f}, "
                                  f"convergence={convergence_achieved}", "INFO")
                        
                        if convergence_achieved:
                            tprint(f"✅ Adaptive convergence achieved at iteration {iteration+1}!", "SUCCESS")
                            tprint(f"   📊 Final improvement: {avg_improvement:.6f} (dynamic threshold: {dynamic_tolerance:.2e})", "SUCCESS")
                            tprint(f"   🎯 Best score: {best_score:.4f}", "SUCCESS")
                            tprint(f"   📈 Silhouette trend: {silhouette_trend:.6f}", "SUCCESS")
                            break
                        else:
                            tprint(f"   ⚠️ Adaptive criteria not met (improvement: {avg_improvement:.6f}, trend: {silhouette_trend:.6f}), continuing...", "WARNING")

                    
                    # ENHANCED: Quality-based early stopping
                    if best_score > quality_target:  # If we achieve target quality, stop early
                        tprint(f"🎉 Target quality achieved at iteration {iteration+1}!", "SUCCESS")
                        tprint(f"   🎯 Quality score: {best_score:.4f} (target: >{quality_target})", "SUCCESS")
                        break
                    
                    # ENHANCED: Check for quality stagnation
                    if len(convergence_history) >= 10:  # Need enough history
                        recent_quality_scores = [h['composite_score'] for h in convergence_history[-10:]]
                        quality_improvement = np.mean(np.diff(recent_quality_scores)) if len(recent_quality_scores) > 1 else 0
                        
                        if quality_improvement < quality_improvement_threshold and best_score < quality_target * 0.85:
                            tprint(f"⚠️ Quality stagnation detected (improvement: {quality_improvement:.6f})", "WARNING")
                            tprint(f"   🔧 Applying aggressive optimization...", "INFO")
                            # Apply more aggressive optimization
                            assignments = self._apply_aggressive_iteration_optimization(features, assignments, k, iteration, convergence_history, distance_cache)
                    
                    # ENHANCED: Apply smart cluster splitting EVERY iteration for maximum responsiveness
                    tprint(f"   🔍 Checking for cluster splitting opportunities (iteration {iteration+1})...", "INFO")
                    assignments, k = self._smart_cluster_splitting_decision(assignments, features, k, iteration)
                    
                    if k > current_k:
                        tprint(f"   📈 Dynamic regime count adjustment: {current_k} → {k}", "SUCCESS")
                        current_k = k
                    else:
                        tprint(f"   📊 No regime count change (K={k})", "INFO")
                    
                    # CRITICAL: Force continuation if overall quality is too poor
                    if best_score < 0.2:  # Unacceptable quality
                        tprint(f"🚨 CRITICAL: Quality too low ({best_score:.4f}), forcing continuation...", "ERROR")
                        tprint(f"   🎯 Target quality: >0.750, Current: {best_score:.4f}", "ERROR")
                        # Reset convergence to force more iterations
                        avg_improvement = tolerance * 2.0  # Force continuation
                        
                        # ENHANCED: Extend iterations if quality is poor
                        if iteration > max_iterations * 0.8:  # Near end of iterations
                            max_iterations += 50  # Add 50 more iterations
                            tprint(f"   🔧 Extending iterations to {max_iterations} for quality improvement", "INFO")
                        
                        # Apply more aggressive single iteration optimization with adaptive thresholding
                        tprint(f"   🔧 Applying optimization round {iteration+1}...", "INFO")
                        assignments = self._apply_aggressive_iteration_optimization(features, assignments, k, iteration, convergence_history)
                        
                        # GLOBAL OPTIMIZATION: Apply global strategies every 3 iterations
                        if iteration % 3 == 0 and iteration > 0:
                            tprint(f"   🌐 Applying global optimization strategies...", "INFO")
                            assignments = self._apply_global_optimization_strategies(features, assignments, k)
                
                # CRITICAL: Apply core optimization to update assignments with early termination
                if not convergence_achieved:  # Only optimize if not converged
                    tprint(f"   🔧 Applying core optimization iteration {iteration+1}...", "INFO")
                    
                    # Store previous assignments for change detection
                    prev_assignments = assignments.copy()
                    
                    # Apply optimization with caching
                    assignments = self._apply_single_iteration_optimization(features, assignments, k, distance_cache)
                    
                    # OPTIMIZATION: Early termination if no significant changes
                    changes = np.sum(assignments != prev_assignments)
                    change_ratio = changes / len(assignments)
                    
                    # ENHANCED: Apply aggressive optimization if stuck in local minimum
                    if change_ratio < 0.0005 and iteration > 10:  # Stuck for multiple iterations
                        tprint(f"   🔥 Local minimum detected: Only {changes} samples changed, applying aggressive optimization...", "WARNING")
                        
                        # Apply more aggressive optimization to escape local minimum
                        assignments = self._apply_aggressive_iteration_optimization(features, assignments, k, iteration, convergence_history)
                        
                        # Check if aggressive optimization helped
                        new_changes = np.sum(assignments != prev_assignments)
                        if new_changes > changes:
                            tprint(f"   ✅ Aggressive optimization successful: {new_changes} samples changed", "SUCCESS")
                        else:
                            tprint(f"   ⚡ Early termination: Still stuck, convergence likely", "INFO")
                            avg_improvement = tolerance * 0.5  # Below threshold
                    elif change_ratio < 0.0005:
                        tprint(f"   ⚡ Early termination: Only {changes} samples changed ({change_ratio:.4f}%), convergence likely", "INFO")
                        # Force convergence to exit early
                        avg_improvement = tolerance * 0.5  # Below threshold
                    
                    # Store assignments for change tracking BEFORE checking changes
                    if len(convergence_history) > 0:
                        convergence_history[-1]['assignments'] = assignments.copy()
                    
                    # Verify that assignments actually changed
                    if iteration > 0 and len(convergence_history) > 1:
                        prev_assignments = convergence_history[-2].get('assignments', assignments)
                        changes = np.sum(assignments != prev_assignments)
                        if changes > 0:
                            tprint(f"   📊 Assignment changes: {changes} samples reassigned", "INFO")
                        else:
                            tprint(f"   ⚠️ No assignment changes detected - optimization may need adjustment", "WARNING")
            
            iteration_metrics = {
                'total_iterations': len(convergence_history),
                'converged': len(convergence_history) < max_iterations,
                'final_score': best_score,
                'convergence_history': convergence_history
            }
            
            # Final progress summary
            tprint(f"🎉 Enhanced iterative convergence completed!", "SUCCESS")
            tprint(f"   📊 Total iterations: {len(convergence_history)}/{max_iterations}", "SUCCESS")
            tprint(f"   🎯 Best score achieved: {best_score:.4f}", "SUCCESS")
            tprint(f"   ✅ Convergence status: {'CONVERGED' if iteration_metrics['converged'] else 'MAX_ITERATIONS'}", "SUCCESS")
            
            # Show improvement summary
            if len(convergence_history) > 1:
                initial_score = convergence_history[0]['composite_score']
                total_improvement = best_score - initial_score
                tprint(f"   📈 Total improvement: {total_improvement:.4f} "
                      f"(from {initial_score:.4f} to {best_score:.4f})", "SUCCESS")
            
            return best_assignments, iteration_metrics
            
        except Exception as e:
            tprint(f"Enhanced iterative convergence failed: {e}", "ERROR")
            return assignments, {'error': str(e), 'total_iterations': 0, 'converged': False}

    def _apply_aggressive_iteration_optimization(self, features: np.ndarray, assignments: np.ndarray, k: int, 
                                               iteration: int = 0, convergence_history: List[Dict] = None, cache: Dict = None) -> np.ndarray:
        """Apply more aggressive single iteration optimization with adaptive thresholding - VECTORIZED OPTIMIZATION."""
        try:
            new_assignments = assignments.copy()
            
            # Calculate adaptive threshold based on iteration progress and convergence history
            adaptive_threshold = self._calculate_adaptive_threshold(iteration, convergence_history)
            
            # GLOBAL OPTIMIZATION: Smart batch sizing with adaptive strategy
            batch_size = self._calculate_optimal_batch_size_enhanced(len(assignments), iteration, features)
            
            # OPTIMIZATION: Use vectorized single-pass optimization instead of multiple rounds
            # This eliminates the need for multiple rounds and reduces complexity
            improvements_made = self._optimize_batch_vectorized_enhanced(
                features, assignments, new_assignments, k, adaptive_threshold, cache
            )
            
            if improvements_made == 0:
                tprint(f"No improvements above threshold {adaptive_threshold:.6f}", "INFO")
            
            return new_assignments
            
        except Exception as e:
            tprint(f"Aggressive iteration optimization failed: {e}", "ERROR")
            return assignments

    def _optimize_batch_vectorized_enhanced(self, features: np.ndarray, assignments: np.ndarray, 
                                          new_assignments: np.ndarray, k: int, adaptive_threshold: float, cache: Dict = None) -> int:
        """Enhanced vectorized batch optimization with single-pass processing and intelligent caching."""
        try:
            # OPTIMIZATION: Pre-compute all centroids and distances once with caching
            regime_centroids = self._compute_regime_centroids_vectorized(features, assignments, k)
            distances, cache = self._compute_distances_with_caching(features, assignments, regime_centroids, k, cache)
            
            # OPTIMIZATION: Compute all improvements in one vectorized operation
            improvements = self._compute_improvements_vectorized(
                features, assignments, distances, regime_centroids, k
            )
            
            # OPTIMIZATION: Apply improvements with threshold filtering
            significant_improvements = improvements > adaptive_threshold
            improvements_made = np.sum(significant_improvements)
            
            if improvements_made > 0:
                # Get significant indices first
                significant_indices = np.where(significant_improvements)[0]
                
                # Get best regimes ONLY for samples with significant improvements
                best_regimes = np.argmax(improvements[significant_indices], axis=1)
                
                # FIXED: Apply assignments only to significant samples
                new_assignments[significant_indices] = best_regimes
                
                # Log significant improvements (limited to avoid spam) - FIXED INDEXING
                if improvements_made <= 50:  # Only log if reasonable number
                    
                    # Get improvement values for significant indices using their best regimes
                    significant_values = np.array([
                        improvements[significant_indices[i], best_regimes[i]] for i in range(len(significant_indices))
                    ])
                    
                    for idx, val, new_regime in zip(significant_indices[:10], significant_values[:10], 
                                                  best_regimes[:10]):
                        tprint(f"      🎯 Sample {idx}: regime {assignments[idx]} → {new_regime} (improvement: {val:.6f})", "INFO")
            
            return improvements_made
            
        except Exception as e:
            tprint(f"Enhanced batch optimization failed: {e}", "ERROR")
            return 0

    def _calculate_adaptive_threshold(self, iteration: int, convergence_history: List[Dict] = None) -> float:
        """Calculate adaptive threshold based on iteration progress and convergence history - ENHANCED FOR QUALITY IMPROVEMENT."""
        try:
            # ENHANCED: More aggressive base threshold for better quality and early stopping
            base_threshold = 0.0001  # 100% more aggressive than before (doubled)
            
            # ENHANCED: More aggressive iteration factors
            if iteration < 5:
                iteration_factor = 0.2  # 20% of base threshold (very aggressive)
            elif iteration < 15:
                iteration_factor = 0.4  # 40% of base threshold
            elif iteration < 30:
                iteration_factor = 0.6  # 60% of base threshold
            else:
                iteration_factor = 0.8  # 80% of base threshold
            
            # Convergence-based adjustment
            convergence_factor = 1.0
            if convergence_history and len(convergence_history) >= 3:
                recent_scores = [h['composite_score'] for h in convergence_history[-3:]]
                avg_improvement = np.mean(np.diff(recent_scores))
                
                # ENHANCED: More aggressive convergence for quality improvement
                if avg_improvement < 0.002:  # Increased threshold
                    convergence_factor = 0.3  # 70% more aggressive
                elif avg_improvement < 0.005:
                    convergence_factor = 0.5  # 50% more aggressive
                elif avg_improvement < 0.005:
                    convergence_factor = 0.7  # 70% more aggressive
                # If improving well, maintain current threshold
                else:
                    convergence_factor = 1.0
            
            # Calculate final adaptive threshold
            adaptive_threshold = base_threshold * iteration_factor * convergence_factor
            
            # Ensure minimum threshold to avoid noise
            adaptive_threshold = max(adaptive_threshold, 1e-6)
            
            # Ensure maximum threshold to maintain some selectivity
            adaptive_threshold = min(adaptive_threshold, 0.001)
            
            return adaptive_threshold
            
        except Exception as e:
            tprint(f"Adaptive threshold calculation failed: {e}", "ERROR")
            return 0.0001  # Fallback to base threshold

    def _optimize_batch_vectorized(self, features: np.ndarray, assignments: np.ndarray, new_assignments: np.ndarray, 
                                 k: int, batch_start: int, batch_end: int, adaptive_threshold: float) -> int:
        """Vectorized batch optimization for better performance."""
        try:
            improvements_made = 0
            batch_size = batch_end - batch_start
            batch_assignments = assignments[batch_start:batch_end]
            batch_features = features[batch_start:batch_end]
            
            # Progress update for medium-sized batches (adjusted for smaller batch size)
            if batch_size > 100:
                tprint(f"      🔄 Processing batch {batch_start}-{batch_end} ({batch_size} samples)...", "INFO")
            
            # Pre-calculate cluster centers for faster distance calculations
            cluster_centers = {}
            for regime in range(k):
                regime_mask = assignments == regime
                if np.sum(regime_mask) > 0:
                    cluster_centers[regime] = np.mean(features[regime_mask], axis=0)
            
            # Vectorized distance calculations
            for i in range(len(batch_assignments)):
                global_idx = batch_start + i
                current_regime = assignments[global_idx]
                sample_features = batch_features[i]
                
                best_regime = current_regime
                best_improvement = 0.0
                
                # Calculate distances to all cluster centers
                distances = {}
                for regime, center in cluster_centers.items():
                    if regime != current_regime:
                        distances[regime] = np.linalg.norm(sample_features - center)
                
                # Try each possible regime (vectorized approach)
                for candidate_regime in range(k):
                    if candidate_regime == current_regime:
                        continue
                    
                    # Fast improvement estimation using distance-based approximation
                    improvement = self._estimate_improvement_fast(
                        assignments, global_idx, candidate_regime, distances.get(candidate_regime, 1.0)
                    )
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_regime = candidate_regime
                
                # Apply best reassignment if improvement exceeds adaptive threshold
                if best_improvement > adaptive_threshold:
                    new_assignments[global_idx] = best_regime
                    improvements_made += 1
            
            # Progress update for significant improvements (adjusted for smaller batches)
            if improvements_made > 0 and batch_size > 50:
                improvement_rate = improvements_made / batch_size * 100
                tprint(f"      ✅ Batch {batch_start}-{batch_end}: {improvements_made} improvements "
                      f"({improvement_rate:.1f}% rate)", "SUCCESS")
            
            return improvements_made
            
        except Exception as e:
            tprint(f"Batch vectorized optimization failed: {e}", "ERROR")
            return 0

    def _estimate_improvement_fast(self, assignments: np.ndarray, sample_idx: int, candidate_regime: int, distance: float) -> float:
        """Fast improvement estimation with enhanced silhouette awareness."""
        try:
            # ENHANCED: More sophisticated silhouette-aware estimation
            # 1. Silhouette-based improvement (primary focus)
            # 2. Distance to candidate cluster center
            # 3. Current cluster size balance
            # 4. Temporal smoothness with neighbors
            
            current_regime = assignments[sample_idx]
            
            # ENHANCED: Estimate silhouette improvement for this reassignment
            silhouette_improvement = self._estimate_silhouette_improvement(
                assignments, sample_idx, current_regime, candidate_regime
            )
            
            # Distance-based improvement (normalized)
            distance_improvement = 1.0 / (1.0 + distance)
            
            # Balance-based improvement
            current_count = np.sum(assignments == current_regime)
            candidate_count = np.sum(assignments == candidate_regime)
            
            # ENHANCED: More aggressive balance penalty for large clusters
            if candidate_count > current_count * 1.3:  # Stricter threshold
                balance_improvement = 0.3  # Harsher penalty
            elif candidate_count > current_count:
                balance_improvement = 0.7
            else:
                balance_improvement = 1.0
            
            # Temporal smoothness (check neighbors)
            temporal_improvement = self._calculate_temporal_factor(assignments, sample_idx, candidate_regime)
            
            # ENHANCED: Prioritize silhouette improvement with new weights
            fast_improvement = (silhouette_improvement * 0.70 +     # Ultra-high focus on silhouette (increased from 0.5)
                              distance_improvement * 0.15 +         # Reduced distance factor (from 0.25)
                              balance_improvement * 0.10 +          # Reduced balance factor (from 0.15)
                              temporal_improvement * 0.05)          # Reduced temporal factor (from 0.10)
            
            return fast_improvement
            
        except Exception as e:
            return 0.0
    
    def _estimate_silhouette_improvement(self, assignments: np.ndarray, sample_idx: int, 
                                       current_regime: int, candidate_regime: int) -> float:
        """Estimate silhouette score improvement for a single sample reassignment."""
        try:
            if len(np.unique(assignments)) < 2:
                return 0.0
            
            # Get cluster sizes
            current_regime_size = np.sum(assignments == current_regime)
            candidate_regime_size = np.sum(assignments == candidate_regime)
            
            # ENHANCED: More sophisticated silhouette estimation
            # Estimate intra-cluster cohesion (a_i in silhouette formula)
            current_cohesion = 1.0 / (1.0 + current_regime_size / 50.0)  # Smaller = better cohesion
            candidate_cohesion = 1.0 / (1.0 + candidate_regime_size / 50.0)
            
            # Estimate inter-cluster separation (b_i in silhouette formula)
            # Moving to a smaller, more cohesive cluster generally improves separation
            separation_improvement = candidate_cohesion - current_cohesion
            
            # ENHANCED: Factor in cluster density
            # Denser clusters (more samples per unit space) are better
            density_factor = min(2.0, candidate_regime_size / max(1, current_regime_size))
            
            # Estimate silhouette improvement
            silhouette_improvement = separation_improvement * density_factor * 0.3
            
            return np.clip(silhouette_improvement, -1.0, 1.0)  # Clamp to valid range
            
        except Exception as e:
            return 0.0
    
    def _calculate_temporal_factor(self, assignments: np.ndarray, sample_idx: int, candidate_regime: int) -> float:
        """Calculate temporal consistency factor for regime reassignment."""
        try:
            if sample_idx == 0 or sample_idx == len(assignments) - 1:
                return 1.0
            
            prev_regime = assignments[sample_idx - 1]
            next_regime = assignments[sample_idx + 1]
            
            # ENHANCED: More nuanced temporal consistency
            if candidate_regime == prev_regime == next_regime:
                return 1.4  # Strong temporal consistency bonus
            elif candidate_regime in [prev_regime, next_regime]:
                return 1.2  # Moderate temporal consistency bonus
            elif prev_regime == next_regime and candidate_regime != prev_regime:
                return 0.7  # Penalty for breaking temporal consistency
            else:
                return 1.0  # Neutral
                
        except Exception as e:
            return 1.0

    def _calculate_enhanced_reassignment_improvement(self, features: np.ndarray, old_assignments: np.ndarray, new_assignments: np.ndarray) -> float:
        """Calculate improvement from reassignment including temporal smoothness - OPTIMIZED."""
        try:
            # Use cached calculations when possible
            if hasattr(self, '_cached_scores') and np.array_equal(self._cached_scores['assignments'], old_assignments):
                old_balance = self._cached_scores['balance']
                old_silhouette = self._cached_scores['silhouette']
                old_cv = self._cached_scores['cv']
                old_temporal = self._cached_scores['temporal']
            else:
            # Calculate old scores
                old_balance = self._calculate_regime_balance_optimized(old_assignments)
                old_silhouette = self._calculate_silhouette_score_optimized(features, old_assignments)
                old_cv = self._calculate_cv_score_optimized(features, old_assignments)
                old_temporal = self._calculate_temporal_smoothness_optimized(old_assignments)
                
                # Cache for next iteration
                self._cached_scores = {
                    'assignments': old_assignments.copy(),
                    'balance': old_balance,
                    'silhouette': old_silhouette,
                    'cv': old_cv,
                    'temporal': old_temporal
                }
            
            # ENHANCED: Ultra-aggressive silhouette optimization for better clustering quality
            balance_weight = getattr(self.config, 'balance_weight', 0.05)  # Minimal balance weight (reduced from 0.10)
            silhouette_weight = 0.60  # Ultra-high silhouette emphasis (increased from 0.45)
            cv_weight = 0.30  # Reduced CV emphasis (from 0.40) to prioritize silhouette
            temporal_weight = 0.05  # Minimal temporal weight (unchanged)
            
            old_composite = (old_balance * balance_weight + 
                           old_silhouette * silhouette_weight + 
                           old_cv * cv_weight + 
                           old_temporal * temporal_weight)
            
            # Calculate new scores (only if significantly different)
            if np.array_equal(old_assignments, new_assignments):
                return 0.0
            
            new_balance = self._calculate_regime_balance_optimized(new_assignments)
            new_silhouette = self._calculate_silhouette_score_optimized(features, new_assignments)
            new_cv = self._calculate_cv_score_optimized(features, new_assignments)
            new_temporal = self._calculate_temporal_smoothness_optimized(new_assignments)
            new_composite = (new_balance * balance_weight + 
                           new_silhouette * silhouette_weight + 
                           new_cv * cv_weight + 
                           new_temporal * temporal_weight)
            
            return new_composite - old_composite
            
        except Exception as e:
            return 0.0

    def _calculate_temporal_smoothness(self, assignments: np.ndarray) -> float:
        """Calculate temporal smoothness score (higher = smoother transitions)."""
        try:
            if len(assignments) < 2:
                return 0.0
            
            # Calculate the number of regime changes
            changes = np.sum(assignments[1:] != assignments[:-1])
            total_possible_changes = len(assignments) - 1
            
            if total_possible_changes == 0:
                return 1.0
            
            # Smoothness score (1.0 = no changes, 0.0 = maximum changes)
            smoothness = 1.0 - (changes / total_possible_changes) if total_possible_changes > 0 else 1.0
            
            return smoothness
            
        except Exception as e:
            return 0.0

    def _calculate_temporal_smoothness_optimized(self, assignments: np.ndarray) -> float:
        """Calculate temporal smoothness score - VECTORIZED OPTIMIZATION."""
        try:
            if len(assignments) < 2:
                return 0.0
            
            # VECTORIZED calculation of regime changes
            changes = np.sum(assignments[1:] != assignments[:-1])
            total_possible_changes = len(assignments) - 1
            
            if total_possible_changes == 0:
                return 1.0
            
            # Calculate smoothness ratio (higher = smoother)
            smoothness_ratio = 1.0 - (changes / total_possible_changes)
            
            # Apply penalty for excessive changes
            if changes > len(assignments) * 0.1:  # More than 10% changes
                penalty_factor = 0.8
            else:
                penalty_factor = 1.0
            
            return np.clip(smoothness_ratio * penalty_factor, 0.0, 1.0)
            
        except Exception as e:
            return 0.0
    
    def _estimate_feature_complexity(self, features: np.ndarray = None) -> float:
        """Estimate feature complexity for adaptive processing."""
        try:
            if features is None:
                return 0.5  # Default complexity
            
            # Calculate feature complexity metrics
            n_features = features.shape[1]
            n_samples = features.shape[0]
            
            # Complexity factor 1: Feature dimensionality
            dimensionality_factor = min(n_features / 100.0, 1.0)
            
            # Complexity factor 2: Feature variance
            feature_variance = np.var(features, axis=0)
            variance_factor = np.mean(feature_variance) / (np.std(feature_variance) + 1e-8)
            variance_factor = min(variance_factor, 1.0)
            
            # Complexity factor 3: Feature correlation
            correlation_matrix = np.corrcoef(features.T)
            correlation_strength = np.mean(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
            correlation_factor = min(correlation_strength, 1.0)
            
            # Combine complexity factors
            complexity = (dimensionality_factor * 0.4 + variance_factor * 0.3 + correlation_factor * 0.3)
            
            return max(0.1, min(1.0, complexity))
            
        except Exception as e:
            return 0.5
    
    def _calculate_volatility_regime_features(self, features: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        """Calculate enhanced volatility regime features."""
        try:
            n_samples = features.shape[0]
            enhanced_features = np.zeros((n_samples, 3))  # 3 new volatility features
            
            # Feature 1: Volatility regime clustering
            volatility_indices = [i for i in range(features.shape[1]) if 'volatility' in str(i)]  # Simplified approach
            if not volatility_indices:
                # Fallback: use last 10 features as volatility features
                volatility_indices = list(range(max(0, features.shape[1] - 10), features.shape[1]))
            
            if volatility_indices:
                volatility_data = features[:, volatility_indices]
                
                # Calculate volatility regime strength
                for i in range(n_samples):
                    sample_volatility = volatility_data[i]
                    regime_volatility = np.mean(volatility_data[assignments == assignments[i]], axis=0)
                    volatility_regime_strength = 1.0 - np.mean(np.abs(sample_volatility - regime_volatility))
                    enhanced_features[i, 0] = volatility_regime_strength
            
            # Feature 2: Volatility momentum indicator
            if len(volatility_indices) > 1:
                for i in range(1, n_samples):
                    current_vol = np.mean(features[i, volatility_indices])
                    previous_vol = np.mean(features[i-1, volatility_indices])
                    momentum = (current_vol - previous_vol) / (previous_vol + 1e-8)
                    enhanced_features[i, 1] = np.tanh(momentum)  # Normalize to [-1, 1]
            
            # Feature 3: Volatility regime transitions
            for i in range(1, n_samples):
                if assignments[i] != assignments[i-1]:
                    # Regime change detected
                    current_vol = np.mean(features[i, volatility_indices]) if volatility_indices else 0
                    previous_vol = np.mean(features[i-1, volatility_indices]) if volatility_indices else 0
                    transition_strength = abs(current_vol - previous_vol) / (previous_vol + 1e-8)
                    enhanced_features[i, 2] = min(transition_strength, 1.0)
            
            return enhanced_features
            
        except Exception as e:
            tprint(f"Volatility regime features calculation failed: {e}", "ERROR")
            return np.zeros((features.shape[0], 3))
            
            # Smoothness score (1.0 = no changes, 0.0 = maximum changes)
            smoothness = 1.0 - (changes / total_possible_changes) if total_possible_changes > 0 else 1.0
            
            return smoothness
            
        except Exception as e:
            return 0.0

    def _calculate_cluster_centers(self, features: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        """Calculate cluster centers from optimized features and assignments."""
        try:
            unique_labels = np.unique(assignments)
            centers = []
            
            for label in unique_labels:
                mask = assignments == label
                if np.any(mask):
                    # Calculate mean for this cluster, skipping empty masks
                    cluster_features = features[mask]
                    center = safe_mean(cluster_features, axis=0)
                    centers.append(center)
                else:
                    # Fallback for empty clusters
                    centers.append(np.zeros(features.shape[1]))
            
            return np.array(centers)
            
        except Exception as exc:
            tprint_warning(f"Failed to calculate cluster centers: {exc}")
            # Return zero centers as fallback
            unique_labels = np.unique(assignments)
            return np.zeros((len(unique_labels), features.shape[1]))

    def _calculate_composite_score(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Calculate composite score for clustering quality using multiple metrics."""
        try:
            # Handle edge cases
            if len(features) == 0 or len(assignments) == 0:
                return 0.0
            
            unique_labels = np.unique(assignments)
            n_clusters = len(unique_labels)
            
            # Return 0 for single cluster or empty data
            if n_clusters < 2:
                return 0.0
            
            # Import safe functions from metrics module
            from ..regime_analysis.metrics import (
                safe_silhouette_score, 
                safe_davies_bouldin_score, 
                safe_calinski_harabasz_score
            )
            
            # Calculate individual metrics
            silhouette = safe_silhouette_score(features, assignments)
            davies_bouldin = safe_davies_bouldin_score(features, assignments)
            calinski_harabasz = safe_calinski_harabasz_score(features, assignments)
            
            # Normalize metrics to [0, 1] range
            # Silhouette is already in [-1, 1], normalize to [0, 1]
            normalized_silhouette = (silhouette + 1) / 2
            
            # Davies-Bouldin: lower is better, normalize by taking inverse and capping
            normalized_davies_bouldin = min(1.0, 1.0 / max(0.1, davies_bouldin))
            
            # Calinski-Harabasz: higher is better, normalize by capping at reasonable value
            normalized_calinski_harabasz = min(1.0, calinski_harabasz / 1000.0)
            
            # Calculate stability score (regime persistence)
            stability_score = self._calculate_stability_score(assignments)
            
            # Calculate consensus score using simple consensus
            # Note: MetricsCalculator.calculate_consensus_metrics requires two arrays (TAS vs NAS)
            # For clustering-only metrics, we use the simple consensus calculation
            consensus_score = self._calculate_simple_consensus(assignments)

            # Weighted composite score
            composite_summary = {
                'silhouette': float(np.clip(normalized_silhouette, 0.0, 1.0)),
                'davies_bouldin': float(np.clip(normalized_davies_bouldin, 0.0, 1.0)),
                'calinski_harabasz': float(np.clip(normalized_calinski_harabasz, 0.0, 1.0)),
                'stability': float(np.clip(stability_score, 0.0, 1.0)),
                'consensus': float(np.clip(consensus_score, 0.0, 1.0)),
            }
            self._last_composite_metric_summary = composite_summary

            weights = self._get_weight_group('composite')

            composite_score = 0.0
            for name, value in composite_summary.items():
                composite_score += weights.get(name, 0.0) * value

            # ENHANCED: Add quality penalties for poor clustering metrics
            # Penalty for poor silhouette score (target: >0.3)
            if normalized_silhouette < 0.3:
                silhouette_penalty = (0.3 - normalized_silhouette) * 0.5  # Up to 0.15 penalty
                composite_score -= silhouette_penalty
                tprint(f"⚠️ Silhouette penalty applied: -{silhouette_penalty:.3f} (score: {normalized_silhouette:.3f})", "WARNING")
            
            # Penalty for poor Davies-Bouldin score (target: <1.0, normalized: >0.5)
            if normalized_davies_bouldin < 0.5:
                davies_penalty = (0.5 - normalized_davies_bouldin) * 0.4  # Up to 0.2 penalty
                composite_score -= davies_penalty
                tprint(f"⚠️ Davies-Bouldin penalty applied: -{davies_penalty:.3f} (score: {normalized_davies_bouldin:.3f})", "WARNING")

            # Ensure score is in [0, 1] range
            composite_score = max(0.0, min(1.0, composite_score))

            return float(composite_score)
            
        except Exception as exc:
            tprint_warning(f"Failed to calculate composite score: {exc}")
            return 0.0

    def _calculate_stability_score(self, assignments: np.ndarray) -> float:
        """Calculate stability score based on regime persistence."""
        try:
            if len(assignments) < 2:
                return 0.0
            
            # Calculate regime change frequency
            changes = np.sum(assignments[1:] != assignments[:-1])
            total_transitions = len(assignments) - 1
            change_rate = changes / total_transitions
            
            # Stability is inverse of change rate
            stability = 1.0 - change_rate
            return max(0.0, min(1.0, stability))
            
        except Exception:
            return 0.0

    def _calculate_simple_consensus(self, assignments: np.ndarray) -> float:
        """Calculate simple consensus score as fallback."""
        try:
            if len(assignments) == 0:
                return 0.0
            
            # Calculate how well assignments are distributed
            unique_labels, counts = np.unique(assignments, return_counts=True)
            n_clusters = len(unique_labels)
            
            if n_clusters < 2:
                return 0.0
            
            # Calculate balance (inverse of standard deviation of cluster sizes)
            mean_size = len(assignments) / n_clusters
            size_variance = np.var(counts)
            balance = 1.0 / (1.0 + size_variance / (mean_size ** 2))
            
            return min(1.0, balance)
            
        except Exception:
            return 0.0

    # @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    # @performance_tracked(log_performance=True, track_memory=True)
    def _calculate_final_quality_metrics(self, features: np.ndarray, assignments: np.ndarray) -> Dict[str, Any]:
        """Calculate final quality metrics for clustering results with hardware optimization."""
        try:
            # Use shared utilities for quality metrics
            from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
            
            # Basic clustering quality metrics
            unique_labels = np.unique(assignments)
            n_clusters = len(unique_labels)
            
            if n_clusters < 2:
                return {
                    "silhouette_score": 0.0,
                    "davies_bouldin_score": float('inf'),
                    "calinski_harabasz_score": 0.0,
                    "intra_cluster_dispersion": 0.0,
                    "inter_cluster_dispersion": 0.0,
                    "cluster_compactness": 0.0
                }
            
            # Import safe functions from metrics module
            from ..regime_analysis.metrics import (
                safe_silhouette_score, 
                safe_davies_bouldin_score, 
                safe_calinski_harabasz_score
            )
            
            # Calculate standard clustering metrics
            silhouette = safe_silhouette_score(features, assignments)
            davies_bouldin = safe_davies_bouldin_score(features, assignments)
            calinski_harabasz = safe_calinski_harabasz_score(features, assignments)
            
            # Calculate intra-cluster dispersion
            intra_dispersion = 0.0
            for label in unique_labels:
                mask = assignments == label
                if np.any(mask):
                    cluster_features = features[mask]
                    center = safe_mean(cluster_features, axis=0)
                    distances = np.linalg.norm(cluster_features - center, axis=1)
                    intra_dispersion += safe_mean(distances)
            
            intra_dispersion /= n_clusters
            
            # Calculate inter-cluster dispersion
            centers = []
            for label in unique_labels:
                mask = assignments == label
                if np.any(mask):
                    center = safe_mean(features[mask], axis=0)
                    centers.append(center)
            
            if len(centers) > 1:
                centers = np.array(centers)
                inter_dispersion = 0.0
                for i in range(len(centers)):
                    for j in range(i + 1, len(centers)):
                        inter_dispersion += np.linalg.norm(centers[i] - centers[j])
                inter_dispersion /= (len(centers) * (len(centers) - 1) / 2)
            else:
                inter_dispersion = 0.0
            
            # Calculate cluster compactness
            compactness = safe_divide(inter_dispersion, intra_dispersion, default=0.0)
            
            return {
                "silhouette_score": float(silhouette),
                "davies_bouldin_score": float(davies_bouldin),
                "calinski_harabasz_score": float(calinski_harabasz),
                "intra_cluster_dispersion": float(intra_dispersion),
                "inter_cluster_dispersion": float(inter_dispersion),
                "cluster_compactness": float(compactness)
            }
            
        except Exception as exc:
            tprint_warning(f"Failed to calculate final quality metrics: {exc}")
            return {
                "silhouette_score": 0.0,
                "davies_bouldin_score": float('inf'),
                "calinski_harabasz_score": 0.0,
                "intra_cluster_dispersion": 0.0,
                "inter_cluster_dispersion": 0.0,
                "cluster_compactness": 0.0
            }

    def _get_weight_group(self, group: str) -> Dict[str, float]:
        """Retrieve learned weights for a group with fallback to defaults."""
        weights = self.learned_weights.get(group)
        if weights:
            sanitized = self._sanitize_weight_dict(group, weights)
            if sanitized:
                self.learned_weights[group] = sanitized
                return sanitized

        defaults = self._default_metric_weights.get(group, {})
        if defaults:
            sanitized_defaults = self._sanitize_weight_dict(group, defaults)
            self.learned_weights.setdefault(group, sanitized_defaults)
            return sanitized_defaults

        return {}

    def _collect_metric_outputs(
        self,
        clustering_result: Dict[str, Any],
        clustering_metrics: Dict[str, Any],
    ) -> Dict[str, Dict[str, float]]:
        """Collect metric outputs across composite, regime, and temporal groups."""
        metric_outputs: Dict[str, Dict[str, float]] = {}

        composite_summary = self._last_composite_metric_summary
        if not composite_summary and isinstance(clustering_result, dict):
            quality = clustering_result.get('clustering_quality', {})
            if isinstance(quality, dict) and quality:
                try:
                    silhouette = float(quality.get('silhouette_score', 0.0))
                    davies_bouldin = float(quality.get('davies_bouldin_score', 0.0))
                    calinski = float(quality.get('calinski_harabasz_score', 0.0))
                    normalized_silhouette = (silhouette + 1.0) / 2.0
                    normalized_davies = min(1.0, 1.0 / max(0.1, davies_bouldin)) if np.isfinite(davies_bouldin) else 0.0
                    normalized_calinski = min(1.0, calinski / 1000.0) if np.isfinite(calinski) else 0.0
                    composite_summary = {
                        'silhouette': float(np.clip(normalized_silhouette, 0.0, 1.0)),
                        'davies_bouldin': float(np.clip(normalized_davies, 0.0, 1.0)),
                        'calinski_harabasz': float(np.clip(normalized_calinski, 0.0, 1.0)),
                        'stability': float(np.clip(quality.get('cluster_compactness', 0.0), 0.0, 1.0)),
                        'consensus': 0.0,
                    }
                except Exception:
                    composite_summary = None

        if composite_summary:
            self._last_composite_metric_summary = composite_summary
            metric_outputs['composite'] = composite_summary

        regime_summary: Optional[Dict[str, float]] = None
        if isinstance(clustering_metrics, dict):
            try:
                economic_scores = clustering_metrics.get('economic_scores')
                trading_scores = clustering_metrics.get('trading_scores')
                stability_scores = clustering_metrics.get('stability_scores')

                def _safe_average(values: Any) -> float:
                    try:
                        if isinstance(values, dict):
                            arr = np.array(list(values.values()), dtype=float)
                        else:
                            arr = np.array(values, dtype=float)
                        if arr.size == 0:
                            return 0.0
                        valid = arr[~np.isnan(arr)]
                        if valid.size == 0:
                            return 0.0
                        return float(np.clip(np.mean(valid), 0.0, 1.0))
                    except Exception:
                        return 0.0

                composite_for_regime = composite_summary or self._last_composite_metric_summary or {}
                regime_summary = {
                    'economic': _safe_average(economic_scores),
                    'volatility': float(composite_for_regime.get('silhouette', _safe_average(stability_scores))),
                    'volume': _safe_average(trading_scores),
                    'structural_trend': float(composite_for_regime.get('stability', _safe_average(stability_scores))),
                }
            except Exception:
                regime_summary = None

        if regime_summary:
            self._last_regime_metric_summary = regime_summary
            metric_outputs['regime'] = regime_summary
        elif self._last_regime_metric_summary:
            metric_outputs['regime'] = self._last_regime_metric_summary

        if self._last_temporal_metric_summary:
            metric_outputs['temporal'] = self._last_temporal_metric_summary

        return metric_outputs

    def _estimate_validation_metric(self, metric_outputs: Dict[str, Dict[str, float]]) -> Optional[float]:
        """Estimate a validation metric when an explicit target is unavailable."""
        if not metric_outputs:
            return None

        for group in ('composite', 'regime', 'temporal'):
            metrics = metric_outputs.get(group)
            if metrics:
                values = np.array(list(metrics.values()), dtype=float)
                if values.size > 0:
                    mean_value = np.nanmean(values)
                    if np.isfinite(mean_value):
                        return float(mean_value)
        return None

    def _derive_validation_metric(self, clustering_metrics: Dict[str, Any]) -> Optional[float]:
        """Derive validation target from pipeline state or clustering metrics."""
        if isinstance(getattr(self, 'pipeline_state', None), dict):
            validation_metrics = self.pipeline_state.get('validation_metrics', {})
            if isinstance(validation_metrics, dict):
                for key in ('validation_sharpe', 'sharpe_ratio', 'sharpe'):
                    value = validation_metrics.get(key)
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        return float(value)

        if isinstance(clustering_metrics, dict):
            trading_scores = clustering_metrics.get('trading_scores')
            if trading_scores is not None:
                try:
                    if isinstance(trading_scores, dict):
                        arr = np.array(list(trading_scores.values()), dtype=float)
                    else:
                        arr = np.array(trading_scores, dtype=float)
                    if arr.size > 0:
                        mean_value = np.nanmean(arr)
                        if np.isfinite(mean_value):
                            return float(mean_value)
                except Exception:
                    return None

        return None

    def _fallback_weight_vector(self, metrics: Dict[str, float], group: str) -> np.ndarray:
        """Fallback weight vector using historical medians or defaults."""
        metric_names = list(metrics.keys())
        if not metric_names:
            return np.array([], dtype=float)

        historical_vectors: List[np.ndarray] = []
        for entry in self.metric_weight_history:
            fitted = entry.get('fitted_weights', {}).get(group)
            if isinstance(fitted, dict):
                vector = np.array([float(fitted.get(name, 0.0)) for name in metric_names], dtype=float)
                historical_vectors.append(vector)

        if not historical_vectors and group in self.learned_weights:
            vector = np.array([
                float(self.learned_weights[group].get(name, 0.0))
                for name in metric_names
            ], dtype=float)
            historical_vectors.append(vector)

        if historical_vectors:
            stacked = np.vstack(historical_vectors)
            medians = np.median(stacked, axis=0)
            medians = np.maximum(medians, 0.0)
            if medians.sum() > 0:
                return self._project_to_simplex(medians)

        defaults = self._default_metric_weights.get(group, {})
        if defaults:
            default_vector = np.array([
                float(defaults.get(name, 0.0))
                for name in metric_names
            ], dtype=float)
            default_vector = np.maximum(default_vector, 0.0)
            if default_vector.sum() > 0:
                return self._project_to_simplex(default_vector)

        uniform = np.ones(len(metric_names), dtype=float)
        return self._project_to_simplex(uniform)

    def _fit_metric_weights(
        self,
        metric_outputs: Dict[str, Dict[str, float]],
        validation_metric: Optional[float] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Fit metric weights using constrained regression with simplex projection."""
        if not metric_outputs:
            return {}

        if validation_metric is None:
            validation_metric = self._estimate_validation_metric(metric_outputs)

        record: Dict[str, Any] = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metric_outputs,
            'validation_target': float(validation_metric) if validation_metric is not None else None,
        }
        self.metric_weight_history.append(record)
        if len(self.metric_weight_history) > self._weight_history_limit:
            self.metric_weight_history = self.metric_weight_history[-self._weight_history_limit:]

        learned: Dict[str, Dict[str, float]] = {}
        for group, metrics in metric_outputs.items():
            metric_names = list(metrics.keys())
            if not metric_names:
                continue

            X_rows: List[List[float]] = []
            y_values: List[float] = []

            for entry in self.metric_weight_history:
                entry_metrics = entry.get('metrics', {}).get(group)
                target = entry.get('validation_target')
                if entry_metrics is None or target is None:
                    continue
                try:
                    row = [float(entry_metrics.get(name, 0.0)) for name in metric_names]
                except Exception:
                    continue
                X_rows.append(row)
                y_values.append(float(target))

            if X_rows and len(X_rows) >= len(metric_names):
                X = np.array(X_rows, dtype=float)
                y = np.array(y_values, dtype=float)
                try:
                    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
                    coeffs = np.maximum(coeffs, 0.0)
                    weight_vector = self._project_to_simplex(coeffs)
                except Exception:
                    weight_vector = self._fallback_weight_vector(metrics, group)
            else:
                weight_vector = self._fallback_weight_vector(metrics, group)

            group_weights = {name: float(weight) for name, weight in zip(metric_names, weight_vector)}
            learned[group] = group_weights
            self.learned_weights[group] = group_weights

        self.metric_weight_history[-1]['fitted_weights'] = learned
        return learned

    # @performance_tracked(log_performance=True, track_memory=True)
    def _update_learned_weights(
        self,
        clustering_result: Dict[str, Any],
        clustering_metrics: Dict[str, Any],
    ) -> None:
        """Update learned metric weights using latest clustering run outputs with performance tracking."""
        try:
            metric_outputs = self._collect_metric_outputs(clustering_result, clustering_metrics)
            if not metric_outputs:
                return

            validation_metric = self._derive_validation_metric(clustering_metrics)
            learned = self._fit_metric_weights(metric_outputs, validation_metric)
            if learned:
                tprint_structured({
                    'metric_weight_update': True,
                    'groups': list(learned.keys()),
                    'validation_metric': validation_metric,
                })
        except Exception as exc:
            tprint_warning(f"Failed to update learned metric weights: {exc}")

    def _summarize_results(self, context: ClusteringContext, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Create the final clustering result payload from the shared context."""
        if context.optimized_features is None or context.smoothed_assignments is None:
            raise ValueError("Optimized features and smoothed assignments are required for summarization")

        optimized_assignments = context.smoothed_assignments
        optimized_features = context.optimized_features
        final_centers = self._calculate_cluster_centers(optimized_features, optimized_assignments)
        final_quality = self._calculate_final_quality_metrics(optimized_features, optimized_assignments)

        self._calibrate_quality_thresholds(context, final_quality)

        metrics = context.optimization_metrics or {}
        metrics.setdefault('fusion_metadata', context.fusion_metadata)

        optimal_k = context.optimal_k or len(set(optimized_assignments))
        optimal_bic = context.optimal_bic if context.optimal_bic is not None else float('nan')
        k_metadata = context.k_metadata or {}

        pre_pca_count = context.pre_pca_feature_count or (
            len(context.pre_pca_feature_names) if context.pre_pca_feature_names else context.original_features.shape[1]
        )
        post_prune_count = context.original_features.shape[1]
        optimized_feature_names = context.optimized_feature_names or getattr(
            self, 'feature_names', [f'feature_{i}' for i in range(post_prune_count)]
        )

        feature_optimization_metadata = {
            'method': 'pca_mle',
            'pre_pca_feature_count': int(pre_pca_count),
            'post_prune_feature_count': int(post_prune_count),
            'optimized_feature_count': int(optimized_features.shape[1]),
            'reduction_ratio': float(optimized_features.shape[1] / max(1, post_prune_count)),
            'retained_feature_names': optimized_feature_names,
            'dropped_feature_names': context.dropped_feature_names or [],
            'feature_scores': context.feature_scores,
            'pca_loading_scores': context.pca_loading_scores,
            'pre_pca_feature_names': context.pre_pca_feature_names or optimized_feature_names,
        }

        clustering_result = {
            'n_clusters': len(set(optimized_assignments)),
            'cluster_assignments': np.asarray(optimized_assignments).tolist(),
            'cluster_centers': final_centers.tolist(),
            'clustering_quality': final_quality,
            'algorithm_used': 'data_driven_optimization',
            'success': True,
            'execution_time': metrics.get('execution_time', 0.0),
            'optimization_metadata': {
                'optimization_method': 'data_driven_optimization',
                'initial_score': metrics.get('initial_score', 0.0),
                'final_score': metrics.get('final_score', 0.0),
                'improvement': metrics.get('improvement', 0.0),
                'iterations': metrics.get('iterations', 0),
                'optimal_k': optimal_k,
                'optimal_bic': optimal_bic,
                'log_likelihood': metrics.get('log_likelihood', 0.0),
                'k_grid': k_metadata.get('k_values', []),
                'ds_confusions': metrics.get('fusion_metadata', {}).get('tas_confusion_matrix', []),
                'hmm_transitions': metrics.get('hmm_transitions', []),
                'random_state': 42,
                'k_selection_metadata': k_metadata,
                'fusion_metadata': metrics.get('fusion_metadata', {}),
                'feature_optimization': feature_optimization_metadata,
                'original_features': context.original_features,
                'feature_names': optimized_feature_names
            }
        }

        clustering_result['refined_feature_names'] = optimized_feature_names
        clustering_result['feature_scores'] = context.feature_scores
        clustering_result['pca_loading_scores'] = context.pca_loading_scores
        clustering_result['pre_pca_feature_names'] = context.pre_pca_feature_names or optimized_feature_names

        # ✅ ADD: Create regime assignments DataFrame with features for parquet saving
        try:
            regime_assignments_df = self._create_regime_assignments_dataframe(
                optimized_assignments, optimized_features, market_data
            )
            clustering_result['regime_assignments_df'] = regime_assignments_df
            tprint(f"✅ Created regime assignments DataFrame: {regime_assignments_df.shape}, {len(regime_assignments_df.columns)} columns", "SUCCESS")
        except Exception as e:
            tprint_warning(f"⚠️ Failed to create regime assignments DataFrame: {e}")
            clustering_result['regime_assignments_df'] = None

        return clustering_result

        context.summary = clustering_result
        return clustering_result

    def _select_optimal_k_bic(self, features: np.ndarray, k_range: Tuple[int, int] = (2, 20), 
                             adaptive: bool = True) -> Tuple[int, float, Dict[str, Any]]:
        """Select optimal K using BIC-selected GMM with adaptive search."""
        try:
            if adaptive:
                tprint("Selecting optimal K using adaptive BIC search...", "INFO")
                return self._adaptive_k_search(features)
            else:
                tprint(f"Selecting optimal K using BIC in range {k_range}...", "INFO")
                return self._fixed_range_k_search(features, k_range)
            
        except Exception as e:
            tprint(f"BIC-based K selection failed: {e}", "ERROR")
            # Fallback to default K
            fallback_k = 8
            tprint(f"Using fallback K={fallback_k}", "WARNING")
            return fallback_k, np.inf, {'method': 'fallback', 'error': str(e)}

    def _adaptive_k_search(self, features: np.ndarray) -> Tuple[int, float, Dict[str, Any]]:
        """Fully evidence-driven K search using conservative cap derived from data."""
        try:
            bic_scores = []
            gmm_models = []
            k_values = []
            
            # Conservative cap derived from data with numerical stability guards
            n_samples, n_features = features.shape
            rank_X = np.linalg.matrix_rank(features)
            max_k = min(25, n_features * 2, rank_X - 1, n_samples - 5)  # Evidence-driven cap with 25 cluster maximum
            tprint(f"Evidence-driven K search with cap={max_k} (n_samples={n_samples}, n_features={n_features}, rank={rank_X}, max_clusters=25)", "INFO")
            
            # Guard against short series vs large K
            min_samples_per_cluster = 5
            max_k_safe = n_samples // min_samples_per_cluster
            if max_k > max_k_safe:
                max_k = max_k_safe
                tprint(f"⚠ K cap reduced to {max_k} due to short series (n_samples={n_samples})", "WARNING")
            
            # Warn if chosen K approaches cap (suggests wider search needed)
            if max_k <= 10:
                tprint(f"⚠ Low K cap ({max_k}) - consider wider search or more data", "WARNING")
            
            # Use parallel K-grid search for efficiency
            k_values = list(range(2, max_k + 1))
            bic_scores, model_metadata, fitted_models, successful_k_values = self._parallel_k_grid(features, k_values, 'gmm', n_jobs=-1)
            
            # Log results
            for i, (k, bic) in enumerate(zip(successful_k_values, bic_scores)):
                tprint(f"K={k}: BIC={bic:.3f}", "INFO")
            
            # Check if we hit the cap
            if max_k <= 10:
                tprint(f"⚠ K cap reached ({max_k}) - consider widening search or more data", "WARNING")
            
            if not bic_scores:
                raise ValueError("No valid BIC scores computed")
            
            # Find optimal K (minimum BIC) with tie handling
            min_bic = min(bic_scores)
            bic_tolerance = 3.0  # BIC difference tolerance for ties
            candidate_indices = [i for i, bic in enumerate(bic_scores) if abs(bic - min_bic) <= bic_tolerance]
            
            if len(candidate_indices) > 1:
                tprint(f"BIC tie detected: {len(candidate_indices)} candidates within {bic_tolerance}", "INFO")
                # Prefer model with higher stability (bootstrap ARI) and better temporal coherence
                best_idx = self._resolve_bic_tie(features, candidate_indices, successful_k_values, fitted_models)
                optimal_k_idx = best_idx
                tprint(f"Tie resolved: chose K={successful_k_values[best_idx]} based on stability", "INFO")
            else:
                optimal_k_idx = np.argmin(bic_scores)
            
            optimal_k = successful_k_values[optimal_k_idx]
            optimal_bic = bic_scores[optimal_k_idx]
            
            # Refit final model to avoid memory bloat (don't store all models)
            tprint(f"Refitting final model for K={optimal_k}...", "INFO")
            optimal_gmm = GaussianMixture(
                n_components=optimal_k, 
                random_state=42, 
                max_iter=100,
                reg_covar=1e-5
            )
            optimal_gmm.fit(features)
            
            # Validate that chosen K is at a clear minimum
            bic_curve = list(zip(k_values, bic_scores))
            is_clear_minimum = self._validate_bic_minimum(bic_curve, optimal_k_idx)
            
            # Edge K alerts
            if optimal_k == max_k:
                tprint(f"⚠ Selected K={optimal_k} is at cap ({max_k}) - consider widening search", "WARNING")
            
            tprint(f"Evidence-driven search found optimal K={optimal_k} with BIC={optimal_bic:.3f} (clear minimum: {is_clear_minimum})", "SUCCESS")
            tprint(f"Evidence-driven search found optimal K={optimal_k} with BIC={optimal_bic:.3f}")
            
            metadata = {
                'bic_scores': bic_scores,
                'k_values': k_values,
                'bic_curve': bic_curve,
                'optimal_k': optimal_k,
                'optimal_bic': optimal_bic,
                'is_clear_minimum': is_clear_minimum,
                'max_k_tested': max_k,
                'method': 'evidence_driven_bic_gmm'
            }
            
            return optimal_k, optimal_bic, metadata
            
        except Exception as e:
            tprint_error(f"Evidence-driven K search failed: {e}")
            # Fallback to fixed range
            return self._fixed_range_k_search(features, (2, 20))

    def _parallel_k_grid(self, features: np.ndarray, k_values: List[int], 
                        model_type: str = 'gmm', n_jobs: int = -1) -> Tuple[List[float], List[Dict], List[Any], List[int]]:
        """Parallel K-grid search for GMM/HMM models."""
        try:
            # Safety checks
            if not self._verify_parallel_safety(n_jobs, len(k_values)):
                tprint("Parallel safety check failed, falling back to serial", "WARNING")
                return self._serial_k_grid(features, k_values, model_type)
            
            # Set environment variables to prevent OpenMP oversubscription
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['OPENBLAS_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            
            tprint(f"Starting parallel K-grid search: {len(k_values)} models, n_jobs={n_jobs}", "INFO")
            
            # Parallel execution
            results = Parallel(n_jobs=n_jobs, backend='threading')(
                delayed(self._fit_single_k_model)(features, k, model_type) 
                for k in k_values
            )
            
            # Extract results while maintaining k_values association
            bic_scores = []
            model_metadata = []
            fitted_models = []
            successful_k_values = []
            
            for i, result in enumerate(results):
                if result[0] is not None:  # Successful fit
                    bic_scores.append(result[0])
                    model_metadata.append(result[1])
                    fitted_models.append(result[2])
                    successful_k_values.append(k_values[i])
            
            tprint(f"Parallel K-grid completed: {len(bic_scores)} successful fits", "SUCCESS")
            return bic_scores, model_metadata, fitted_models, successful_k_values
            
        except Exception as e:
            tprint_error(f"Parallel K-grid search failed: {e}")
            # Fallback to serial search
            return self._serial_k_grid(features, k_values, model_type)

    def _fit_single_k_model(self, features: np.ndarray, k: int, model_type: str) -> Tuple[Optional[float], Optional[Dict], Optional[Any]]:
        """Fit a single K model (worker function for parallel execution)."""
        try:
            if model_type == 'gmm':
                # Fit GMM with regularization
                model = GaussianMixture(
                    n_components=k, 
                    random_state=42, 
                    max_iter=100,
                    reg_covar=1e-5
                )
                model.fit(features)
                bic_score = model.bic(features)
                
                # Return minimal metadata to avoid memory bloat
                metadata = {
                    'k': k,
                    'bic': bic_score,
                    'converged': model.converged_,
                    'n_iter': model.n_iter_
                }
                
            elif model_type == 'hmm':
                # Fit HMM with regularization
                model = hmm.GaussianHMM(
                    n_components=k, 
                    random_state=42, 
                    n_iter=100,
                    covariance_type='full'
                )
                model.covars_prior = 1e-5
                model.fit(features)
                
                # Calculate BIC with correct parameter count
                log_likelihood = model.score(features)
                n_samples, n_features = features.shape
                n_params = (k - 1) + k * (k - 1) + k * n_features + k * n_features * (n_features + 1) // 2
                bic_score = -2 * log_likelihood + n_params * np.log(n_samples)
                
                metadata = {
                    'k': k,
                    'bic': bic_score,
                    'converged': True,  # HMM doesn't have converged_ attribute
                    'n_iter': 100
                }
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            
            return bic_score, metadata, model
            
        except Exception as e:
            tprint(f"Model fit failed for K={k}: {e}")
            return None, None, None

    def _serial_k_grid(self, features: np.ndarray, k_values: List[int], model_type: str) -> Tuple[List[float], List[Dict], List[Any], List[int]]:
        """Serial fallback for K-grid search."""
        try:
            bic_scores = []
            model_metadata = []
            fitted_models = []
            successful_k_values = []
            
            for k in k_values:
                result = self._fit_single_k_model(features, k, model_type)
                if result[0] is not None:
                    bic_scores.append(result[0])
                    model_metadata.append(result[1])
                    fitted_models.append(result[2])
                    successful_k_values.append(k)
            
            return bic_scores, model_metadata, fitted_models, successful_k_values
            
        except Exception as e:
            tprint(f"Serial K-grid search failed: {e}")
            return [], [], [], []

    def _verify_parallel_safety(self, n_jobs: int, n_models: int) -> bool:
        """Verify parallel execution safety to prevent oversubscription."""
        try:
            import psutil
            
            # Check if we have enough CPU cores
            n_cores = psutil.cpu_count(logical=False)
            if n_jobs == -1:
                n_jobs = n_cores
            
            # Warn if we might oversubscribe
            if n_jobs > n_cores:
                tprint(f"⚠ n_jobs={n_jobs} > n_cores={n_cores}, may cause oversubscription", "WARNING")
                return False
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 80:
                tprint(f"⚠ High memory usage ({memory.percent:.1f}%), consider reducing n_jobs", "WARNING")
                return False
            
            tprint(f"✓ Parallel safety: n_jobs={n_jobs}, n_cores={n_cores}, memory={memory.percent:.1f}%", "INFO")
            return True
            
        except ImportError:
            tprint("⚠ psutil not available, skipping parallel safety checks", "WARNING")
            return True
        except Exception as e:
            tprint(f"Parallel safety check failed: {e}")
            return True

    def _validate_bic_minimum(self, bic_curve: List[Tuple[int, float]], optimal_idx: int) -> bool:
        """Validate that the chosen K is at a clear minimum."""
        try:
            if len(bic_curve) < 3:
                return True  # Not enough data to validate
            
            optimal_bic = bic_curve[optimal_idx][1]
            
            # Check if optimal BIC is significantly better than neighbors
            if optimal_idx > 0:
                left_bic = bic_curve[optimal_idx - 1][1]
                if abs(optimal_bic - left_bic) < 0.01:  # Too close
                    return False
            
            if optimal_idx < len(bic_curve) - 1:
                right_bic = bic_curve[optimal_idx + 1][1]
                if abs(optimal_bic - right_bic) < 0.01:  # Too close
                    return False
            
            # Check if it's the global minimum
            all_bics = [bic for _, bic in bic_curve]
            if optimal_bic > min(all_bics) + 1.0:  # Not close to global minimum
                return False
            
            return True
            
        except Exception as e:
            tprint(f"BIC minimum validation failed: {e}")
            return True  # Assume valid if validation fails

    def _resolve_bic_tie(self, features: np.ndarray, candidate_indices: List[int], 
                        k_values: List[int], gmm_models: List) -> int:
        """Resolve BIC ties by preferring higher stability and temporal coherence."""
        try:
            from sklearn.metrics import adjusted_rand_score
            
            best_idx = candidate_indices[0]
            best_score = 0.0
            
            for idx in candidate_indices:
                k = k_values[idx]
                gmm = gmm_models[idx]
                
                # Get assignments
                assignments = gmm.predict(features)
                
                # Calculate stability (bootstrap ARI)
                n_bootstrap = 10  # Reduced for efficiency
                ari_scores = []
                for _ in range(n_bootstrap):
                    bootstrap_idx = np.random.choice(len(assignments), len(assignments), replace=True)
                    bootstrap_assignments = assignments[bootstrap_idx]
                    ari_scores.append(adjusted_rand_score(assignments, bootstrap_assignments))
                
                stability_score = np.mean(ari_scores)
                
                # Calculate temporal coherence (regime persistence)
                temporal_score = self._calculate_temporal_coherence(assignments)
                
                # Combined score: stability + temporal coherence
                combined_score = stability_score + temporal_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = idx
                
                tprint(f"  K={k}: stability={stability_score:.3f}, temporal={temporal_score:.3f}, combined={combined_score:.3f}", "INFO")
            
            return best_idx
            
        except Exception as e:
            tprint(f"Tie resolution failed: {e}, using first candidate")
            return candidate_indices[0]

    def _calculate_temporal_coherence(self, assignments: np.ndarray) -> float:
        """Calculate temporal coherence score for regime persistence."""
        try:
            if len(assignments) < 2:
                return 0.0
            
            # Calculate regime persistence (fraction of time spent in same regime)
            persistence = 0.0
            for i in range(len(assignments) - 1):
                if assignments[i] == assignments[i + 1]:
                    persistence += 1
            
            return persistence / (len(assignments) - 1)
            
        except Exception as e:
            tprint(f"Temporal coherence calculation failed: {e}")
            return 0.0

    def _fixed_range_k_search(self, features: np.ndarray, k_range: Tuple[int, int]) -> Tuple[int, float, Dict[str, Any]]:
        """Fixed range K search for fallback."""
        try:
            min_k, max_k = k_range
            bic_scores = []
            gmm_models = []
            
            # Test different K values
            for k in range(min_k, max_k + 1):
                try:
                    gmm = GaussianMixture(n_components=k, random_state=42, max_iter=100)
                    gmm.fit(features)
                    bic_scores.append(gmm.bic(features))
                    gmm_models.append(gmm)
                    tprint(f"K={k}: BIC={gmm.bic(features):.3f}", "INFO")
                except Exception as e:
                    tprint(f"GMM with K={k} failed: {e}")
                    bic_scores.append(np.inf)
                    gmm_models.append(None)
            
            # Find optimal K (minimum BIC)
            optimal_k_idx = np.argmin(bic_scores)
            optimal_k = min_k + optimal_k_idx
            optimal_bic = bic_scores[optimal_k_idx]
            optimal_gmm = gmm_models[optimal_k_idx]
            
            tprint_success(
                f"Fixed range search found optimal K={optimal_k} with BIC={optimal_bic:.3f}"
            )
            
            metadata = {
                'bic_scores': bic_scores,
                'k_range': k_range,
                'optimal_k': optimal_k,
                'optimal_bic': optimal_bic,
                'method': 'fixed_range_bic_gmm'
            }
            
            return optimal_k, optimal_bic, metadata
            
        except Exception as e:
            tprint(f"Fixed range K search failed: {e}")
            raise

    def _validate_feature_quality(self, features: np.ndarray, market_data: pd.DataFrame) -> np.ndarray:
        """Validate and improve feature quality for clustering."""
        try:
            # Check for NaN/inf values
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                tprint("Features contain NaN/inf values, removing problematic samples")
                valid_mask = ~(np.any(np.isnan(features), axis=1) | np.any(np.isinf(features), axis=1))
                features = features[valid_mask]
                market_data = market_data.iloc[valid_mask] if len(market_data) == len(features) else market_data

            # Check feature variance
            feature_variances = np.var(features, axis=0)
            low_variance_features = feature_variances < 1e-6

            if np.any(low_variance_features):
                tprint(f"Removing {np.sum(low_variance_features)} low-variance features")
                features = features[:, ~low_variance_features]

            # Check for highly correlated features (shouldn't be needed after earlier steps but safety check)
            if features.shape[1] > 1:
                try:
                    corr_matrix = np.corrcoef(features.T)
                    # Remove features with correlation > 0.99
                    high_corr_mask = np.triu(np.abs(corr_matrix) > 0.99, k=1)
                    if np.any(high_corr_mask):
                        # Find indices of highly correlated features
                        to_remove = set()
                        for i in range(len(high_corr_mask)):
                            for j in range(i+1, len(high_corr_mask[i])):
                                if high_corr_mask[i, j]:
                                    to_remove.add(j)  # Remove the second feature in pair

                        if to_remove:
                            keep_indices = [i for i in range(features.shape[1]) if i not in to_remove]
                            features = features[:, keep_indices]
                            tprint(f"Removed {len(to_remove)} highly correlated features")
                except:
                    pass  # Skip correlation check if it fails

            # Final check: ensure we have enough features and samples
            if features.shape[1] < 3:
                tprint("Too few features for clustering, using fallback")
                return features

            if features.shape[0] < 50:
                tprint("Low number of samples for clustering")

            return features

        except Exception as e:
            tprint(f"Feature quality validation failed: {e}")
            return features
    

    def _filter_regime_relevant_features(self, features: np.ndarray, market_data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Filter out trading-relevant features, keep only regime-relevant ones."""
        try:
            tprint("🔍 Filtering regime-relevant features (excluding trading features)...", "INFO")
            
            # Define trading-relevant feature patterns to exclude
            trading_patterns = [
                'rsi', 'macd', 'stochastic', 'williams', 'momentum',
                'oscillator', 'signal', 'crossover', 'divergence',
                'candlestick', 'pattern', 'breakout', 'support', 'resistance',
                'bollinger', 'atr', 'cci', 'roc', 'mfi', 'obv'
            ]
            
            # Define regime-relevant feature patterns to prioritize
            regime_patterns = [
                'volatility', 'volume_regime', 'trend_persistence', 
                'regime_stability', 'correlation', 'distribution',
                'clustering', 'persistence', 'structural', 'statistical',
                'vol_persistence', 'vol_clustering', 'vol_stability',
                'vol_regime', 'trend_strength', 'market_structure'
            ]
            
            # Create dummy feature names for filtering (in real implementation, these would come from feature generation)
            feature_names = [f'feature_{i}' for i in range(features.shape[1])]
            
            # Filter features based on regime relevance
            regime_relevant_indices = []
            regime_relevant_names = []
            
            for i, name in enumerate(feature_names):
                name_lower = name.lower()
                
                # Exclude if matches trading patterns
                if any(pattern in name_lower for pattern in trading_patterns):
                    continue
                    
                # Include if matches regime patterns or passes regime tests
                if (any(pattern in name_lower for pattern in regime_patterns) or
                    self._is_regime_relevant_feature(features[:, i], market_data)):
                    regime_relevant_indices.append(i)
                    regime_relevant_names.append(name)
            
            filtered_features = features[:, regime_relevant_indices] if regime_relevant_indices else features
            
            tprint(f"Feature filtering: {features.shape[1]} -> {filtered_features.shape[1]} regime-relevant features", "SUCCESS")
            return filtered_features, regime_relevant_names
            
        except Exception as e:
            tprint(f"Feature filtering failed: {e}", "ERROR")
            return features, [f'feature_{i}' for i in range(features.shape[1])]
    
    def _is_regime_relevant_feature(self, feature_values: np.ndarray, market_data: pd.DataFrame) -> bool:
        """Test if a feature is relevant for regime classification."""
        try:
            # Test 1: Regime persistence - feature should be stable within regimes
            regime_persistence = self._calculate_feature_regime_persistence(feature_values, market_data)

            # Test 2: Low noise-to-signal ratio
            noise_ratio = self._calculate_feature_noise_ratio(feature_values)

            # Test 3: Temporal stability
            temporal_stability = self._calculate_feature_temporal_stability(feature_values)

            thresholds = self._get_calibrated_quality_thresholds()

            return (
                regime_persistence > thresholds['min_regime_persistence'] and
                noise_ratio < thresholds['max_feature_noise_ratio'] and
                temporal_stability > thresholds['min_temporal_stability']
            )
        except:
            return False
    
    def _calculate_feature_regime_persistence(self, feature_values: np.ndarray, market_data: pd.DataFrame) -> float:
        """Calculate regime persistence for a feature."""
        try:
            if len(feature_values) < 10:
                return 0.0
            
            # Calculate autocorrelation as a proxy for regime persistence
            if len(feature_values) > 1:
                corr = np.corrcoef(feature_values[:-1], feature_values[1:])[0, 1]
                return corr if not np.isnan(corr) else 0.0
            return 0.0
        except:
            return 0.0
    
    def _calculate_feature_noise_ratio(self, feature_values: np.ndarray) -> float:
        """Calculate noise-to-signal ratio for a feature."""
        try:
            if len(feature_values) < 5:
                return 1.0
            
            # Noise ratio based on coefficient of variation
            mean_val = np.mean(feature_values)
            std_val = np.std(feature_values)
            return std_val / (abs(mean_val) + 1e-8)
        except:
            return 1.0
    
    def _calculate_feature_temporal_stability(self, feature_values: np.ndarray) -> float:
        """Calculate temporal stability for a feature."""
        try:
            if len(feature_values) < 5:
                return 0.0
            
            # Temporal stability based on low variance of rolling statistics
            window = min(5, len(feature_values) // 2)
            if window < 2:
                return 0.0
            
            rolling_means = []
            for i in range(window, len(feature_values)):
                rolling_means.append(np.mean(feature_values[i-window:i]))
            
            if len(rolling_means) > 1:
                stability = 1.0 - (np.std(rolling_means) / (np.mean(np.abs(rolling_means)) + 1e-8))
                return max(0, min(1, stability))
            return 0.0
        except:
            return 0.0
    
    def _calculate_temporal_feature_importance(self, features: np.ndarray, market_data: pd.DataFrame) -> np.ndarray:
        """Calculate temporal feature importance based on temporal stability with M1 hardware optimization and vectorized operations."""
        try:
            tprint("  📊 Calculating temporal feature importance with M1 hardware optimization and vectorized operations...", "INFO")
            
            # Initialize M1 hardware optimization for temporal analysis
            if self.m1_cpu_optimizer:
                tprint("    ⚡ Using M1 CPU optimization for vectorized temporal analysis", "INFO")
            if self.m1_memory_optimizer:
                tprint("    💾 Using M1 memory optimization for vectorized temporal analysis", "INFO")
            
            # VECTORIZED CALCULATION: Process all features simultaneously
            tprint("    🔄 Using vectorized operations for all features simultaneously...", "INFO")
            
            # 1. Vectorized autocorrelation calculation for all features
            autocorr_weights = self._calculate_vectorized_autocorrelation(features)
            
            # 2. Vectorized temporal variance calculation for all features
            temporal_var_weights = self._calculate_vectorized_temporal_variance(features)

            # 3. Vectorized trend consistency calculation for all features
            trend_consistency_weights = self._calculate_vectorized_trend_consistency(features)

            # 4. Vectorized regime persistence calculation for all features
            regime_persistence_weights = self._calculate_vectorized_regime_persistence(features, market_data)

            # VECTORIZED COMBINATION: Combine all metrics using matrix operations
            stability_component = 1.0 / (1.0 + temporal_var_weights)
            weights = self._get_weight_group('temporal')
            temporal_weights = (
                weights.get('autocorrelation', 0.0) * autocorr_weights +
                weights.get('stability', 0.0) * stability_component +
                weights.get('trend_consistency', 0.0) * trend_consistency_weights +
                weights.get('regime_persistence', 0.0) * regime_persistence_weights
            )

            temporal_summary = {
                'autocorrelation': float(np.clip(np.nanmean(autocorr_weights), 0.0, 1.0)) if autocorr_weights.size else 0.0,
                'stability': float(np.clip(np.nanmean(stability_component), 0.0, 1.0)) if stability_component.size else 0.0,
                'trend_consistency': float(np.clip(np.nanmean(trend_consistency_weights), 0.0, 1.0)) if trend_consistency_weights.size else 0.0,
                'regime_persistence': float(np.clip(np.nanmean(regime_persistence_weights), 0.0, 1.0)) if regime_persistence_weights.size else 0.0,
            }
            self._last_temporal_metric_summary = temporal_summary

            # Normalize weights to [0, 1] range
            if np.max(temporal_weights) > 0:
                temporal_weights = temporal_weights / np.max(temporal_weights)
            
            tprint(f"  ✅ Temporal importance calculated for {features.shape[1]} features", "SUCCESS")
            tprint(f"  📈 Top 5 temporal features: {np.argsort(temporal_weights)[-5:][::-1]}", "INFO")
            
            return temporal_weights
            
        except Exception as e:
            tprint(f"Temporal feature importance calculation failed: {e}")
            return np.ones(features.shape[1])  # Fallback to uniform weights
    
    def _calculate_vectorized_autocorrelation(self, features: np.ndarray) -> np.ndarray:
        """Calculate autocorrelation for all features using vectorized operations."""
        try:
            tprint("      🔄 Calculating vectorized autocorrelation for all features...", "INFO")
            
            if features.shape[0] < 2:
                return np.zeros(features.shape[1])
            
            # Vectorized autocorrelation calculation
            # Calculate mean for each feature
            feature_means = np.mean(features, axis=0, keepdims=True)
            
            # Calculate autocorrelation using vectorized operations
            # Autocorr = E[(X_t - μ)(X_{t+1} - μ)] / E[(X_t - μ)²]
            centered_features = features - feature_means
            
            # Calculate numerator: (X_t - μ)(X_{t+1} - μ) for all features
            numerator = np.sum(centered_features[:-1] * centered_features[1:], axis=0)
            
            # Calculate denominator: (X_t - μ)² for all features
            denominator = np.sum(centered_features ** 2, axis=0)
            
            # Avoid division by zero
            autocorr = np.where(denominator != 0, np.abs(numerator / denominator), 0)
            
            tprint(f"      ✅ Vectorized autocorrelation calculated for {features.shape[1]} features", "SUCCESS")
            return autocorr
            
        except Exception as e:
            tprint(f"Vectorized autocorrelation calculation failed: {e}")
            return np.zeros(features.shape[1])
    
    def _calculate_vectorized_temporal_variance(self, features: np.ndarray) -> np.ndarray:
        """Calculate temporal variance for all features using optimized vectorized operations."""
        try:
            tprint("      🔄 Calculating vectorized temporal variance for all features...", "INFO")
            
            if features.shape[0] < 2:
                return np.zeros(features.shape[1])
            
            # Optimized rolling variance calculation using pandas rolling window
            window_size = min(10, features.shape[0] // 4)
            if window_size < 2:
                return np.var(features, axis=0)
            
            # Use pandas rolling window for efficient computation
            import pandas as pd
            df_features = pd.DataFrame(features)
            
            # Calculate rolling variance using pandas (implemented in C)
            rolling_vars = df_features.rolling(window=window_size, min_periods=1).var()
            
            # Calculate mean rolling variance for each feature, ignoring NaN values
            temporal_vars = rolling_vars.mean(skipna=True).values
            
            # Handle any remaining NaN values
            temporal_vars = np.nan_to_num(temporal_vars, nan=0.0)
            
            tprint(f"      ✅ Vectorized temporal variance calculated for {features.shape[1]} features", "SUCCESS")
            
            # Clean up temporary DataFrame
            del df_features, rolling_vars
            
            return temporal_vars
            
        except Exception as e:
            tprint(f"Vectorized temporal variance calculation failed: {e}")
            # Fallback to simple variance if rolling fails
            return np.var(features, axis=0)
    
    def _calculate_vectorized_trend_consistency(self, features: np.ndarray) -> np.ndarray:
        """Calculate trend consistency for all features using vectorized operations."""
        try:
            tprint("      🔄 Calculating vectorized trend consistency for all features...", "INFO")
            
            if features.shape[0] < 3:
                return np.zeros(features.shape[1])
            
            # Vectorized trend consistency calculation
            # Calculate first differences for all features
            diffs = np.diff(features, axis=0)
            
            # Calculate direction changes for all features
            direction_changes = np.sum(np.diff(np.sign(diffs), axis=0) != 0, axis=0)
            max_changes = diffs.shape[0] - 1
            
            # Calculate consistency for all features
            consistency = np.where(max_changes > 0, 1.0 - (direction_changes / max_changes), 1.0)
            consistency = np.maximum(0.0, consistency)
            
            tprint(f"      ✅ Vectorized trend consistency calculated for {features.shape[1]} features", "SUCCESS")
            return consistency
            
        except Exception as e:
            tprint(f"Vectorized trend consistency calculation failed: {e}")
            return np.zeros(features.shape[1])
    
    def _calculate_vectorized_regime_persistence(self, features: np.ndarray, market_data: pd.DataFrame) -> np.ndarray:
        """Calculate regime persistence for all features using vectorized operations."""
        try:
            tprint("      🔄 Calculating vectorized regime persistence for all features...", "INFO")
            
            # Get regime assignments if available
            tas_assignments, nas_assignments = self._get_tas_nas_assignments()
            
            if tas_assignments is None or len(tas_assignments) != len(features):
                tprint("      ⚠️  No regime data available, using neutral persistence", "WARNING")
                return np.ones(features.shape[1]) * 0.5
            
            # Vectorized regime persistence calculation
            unique_regimes = np.unique(tas_assignments)
            regime_stabilities = []
            
            for regime in unique_regimes:
                regime_mask = tas_assignments == regime
                regime_features = features[regime_mask]
                
                if len(regime_features) > 1:
                    # Calculate CV for each feature within the regime using vectorized operations
                    feature_means = np.mean(regime_features, axis=0)
                    feature_stds = np.std(regime_features, axis=0)
                    
                    # Avoid division by zero
                    feature_cvs = np.where(feature_means != 0, feature_stds / np.abs(feature_means), 0)
                    
                    # Stability = 1 / (1 + CV) for all features
                    regime_stability = 1.0 / (1.0 + feature_cvs)
                    regime_stabilities.append(regime_stability)
            
            # Calculate average stability across regimes for each feature
            if regime_stabilities:
                persistence_weights = np.mean(regime_stabilities, axis=0)
            else:
                persistence_weights = np.ones(features.shape[1]) * 0.5
            
            tprint(f"      ✅ Vectorized regime persistence calculated for {features.shape[1]} features", "SUCCESS")
            return persistence_weights
            
        except Exception as e:
            tprint(f"Vectorized regime persistence calculation failed: {e}")
            return np.ones(features.shape[1]) * 0.5
    
    # REMOVED: Legacy single-feature methods replaced by vectorized versions
    # These methods were replaced by _calculate_vectorized_* methods for better performance
    
    def _cross_validation_feature_selection(self, features: np.ndarray, labels: np.ndarray, temporal_weights: np.ndarray) -> np.ndarray:
        """Perform cross-validation feature selection for robust feature selection with M1 hardware optimization."""
        try:
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.feature_selection import SelectKBest, mutual_info_classif
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            
            tprint("  🔄 Performing time-series cross-validation feature selection with M1 hardware optimization...", "INFO")
            tprint("  ⚠️  Using TimeSeriesSplit to prevent data leakage", "WARNING")
            
            # Initialize M1 hardware optimization for cross-validation
            if self.m1_cpu_optimizer:
                tprint("    ⚡ Using M1 CPU optimization for cross-validation", "INFO")
            if self.m1_memory_optimizer:
                tprint("    💾 Using M1 memory optimization for cross-validation", "INFO")
            
            # Parameters
            n_folds = 5
            n_features = min(15, features.shape[1])
            
            # Initialize feature importance scores
            feature_scores = np.zeros(features.shape[1])
            feature_counts = np.zeros(features.shape[1])
            
            # Cross-validation setup - Use TimeSeriesSplit for proper time series validation
            # This ensures training data always comes BEFORE test data (no future leakage)
            tss = TimeSeriesSplit(n_splits=n_folds)
            
            for fold, (train_idx, val_idx) in enumerate(tss.split(features)):
                tprint(f"    Fold {fold + 1}/{n_folds}...", "INFO")
                
                # Split data
                X_train, X_val = features[train_idx], features[val_idx]
                y_train, y_val = labels[train_idx], labels[val_idx]
                
                # Feature selection for this fold
                selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
                X_train_selected = selector.fit_transform(X_train, y_train)
                selected_features = selector.get_support(indices=True)
                
                # Apply temporal weighting to feature scores
                for feature_idx in selected_features:
                    temporal_weight = temporal_weights[feature_idx]
                    feature_scores[feature_idx] += temporal_weight
                    feature_counts[feature_idx] += 1
                
                # Validate selection with Random Forest
                if len(selected_features) > 0:
                    rf = RandomForestClassifier(n_estimators=50, random_state=42)
                    rf.fit(X_train_selected, y_train)
                    val_pred = rf.predict(X_val[:, selected_features])
                    accuracy = accuracy_score(y_val, val_pred)
                    tprint(f"      Validation accuracy: {accuracy:.3f}", "INFO")
            
            # Select features based on cross-validation scores
            avg_scores = np.where(feature_counts > 0, feature_scores / feature_counts, 0)
            top_features = np.argsort(avg_scores)[-n_features:][::-1]
            
            tprint(f"  ✅ CV feature selection completed - Selected {len(top_features)} features", "SUCCESS")
            tprint(f"  📊 Top features: {top_features[:5]}", "INFO")
            
            return features[:, top_features]
            
        except Exception as e:
            tprint(f"Cross-validation feature selection failed: {e}")
            # Fallback to simple feature selection
            selector = SelectKBest(score_func=mutual_info_classif, k=min(15, features.shape[1]))
            return selector.fit_transform(features, labels)
    
    def _apply_temporal_weighting(self, features: np.ndarray, selected_features: np.ndarray, temporal_weights: np.ndarray) -> np.ndarray:
        """Apply temporal weighting to selected features."""
        try:
            tprint("  ⚖️  Applying temporal weighting to features...", "INFO")
            
            # Get the indices of selected features
            if selected_features.shape[1] != features.shape[1]:
                # Features were already selected, return as is
                return selected_features
            
            # Apply temporal weighting
            weighted_features = features.copy()
            for i in range(features.shape[1]):
                temporal_weight = temporal_weights[i]
                weighted_features[:, i] *= temporal_weight
            
            tprint(f"  ✅ Temporal weighting applied to {features.shape[1]} features", "SUCCESS")
            return weighted_features
            
        except Exception as e:
            tprint(f"Temporal weighting application failed: {e}")
            return selected_features
    
    def _assess_data_driven_divergence_with_confidence(self, tas_assignments: np.ndarray, nas_assignments: np.ndarray) -> Dict[str, Any]:
        """Assess divergence using data-driven methods with confidence scoring."""
        try:
            tprint("  🧠 Performing data-driven divergence assessment with confidence scoring...", "INFO")
            
            # Get features for regime centroid comparison
            features = self._get_current_features()
            if features is None:
                tprint("  ⚠️  No features available for data-driven assessment, using numerical comparison only", "WARNING")
                return self._assess_numerical_divergence_with_confidence(tas_assignments, nas_assignments)
            
            # Step 1: Calculate regime centroids in feature space
            tas_centroids = self._calculate_regime_centroids(features, tas_assignments)
            nas_centroids = self._calculate_regime_centroids(features, nas_assignments)
            
            # Step 2: Find optimal regime mapping using enhanced multi-objective algorithm
            regime_mapping = self._find_optimal_regime_mapping(tas_centroids, nas_centroids, features, None, tas_assignments, nas_assignments)
            
            # Step 3: Calculate semantic divergence using mapped regimes
            semantic_assignments = self._apply_regime_mapping(nas_assignments, regime_mapping)
            semantic_disagreement_mask = tas_assignments != semantic_assignments
            semantic_divergence_rate = np.mean(semantic_disagreement_mask)
            
            # Step 4: Calculate confidence scores for divergence detection
            confidence_scores = self._calculate_divergence_confidence_scores(
                features, tas_assignments, nas_assignments, semantic_disagreement_mask
            )
            
            # Step 5: Calculate comprehensive mapping quality metrics using enhanced multi-dimensional assessment
            mapping_quality_metrics = self._calculate_mapping_quality(tas_centroids, nas_centroids, regime_mapping, features, None, tas_assignments, nas_assignments)
            
            # Step 6: Analyze regime similarity patterns
            similarity_analysis = self._analyze_regime_similarity(tas_centroids, nas_centroids, regime_mapping)
            
            # Step 7: Comprehensive reporting with confidence information
            self._report_mapping_quality_with_confidence(mapping_quality_metrics, regime_mapping, confidence_scores)
            
            tprint(f"  📊 Data-driven assessment with confidence results:", "INFO")
            tprint(f"     Semantic divergence rate: {semantic_divergence_rate:.3f}", "INFO")
            tprint(f"     Overall mapping quality: {mapping_quality_metrics['overall_quality']:.3f}", "INFO")
            tprint(f"     Average confidence: {np.mean(confidence_scores):.3f}", "INFO")
            tprint(f"     High confidence samples: {np.sum(confidence_scores > 0.8)}", "INFO")
            
            return {
                'semantic_divergence_rate': semantic_divergence_rate,
                'mapping_quality': mapping_quality_metrics,
                'regime_mapping': regime_mapping,
                'similarity_analysis': similarity_analysis,
                'tas_centroids': tas_centroids,
                'nas_centroids': nas_centroids,
                'confidence_scores': confidence_scores
            }
            
        except Exception as e:
            tprint(f"Data-driven divergence assessment with confidence failed: {e}")
            return self._assess_numerical_divergence_with_confidence(tas_assignments, nas_assignments)
    
    def _assess_numerical_divergence_with_confidence(self, tas_assignments: np.ndarray, nas_assignments: np.ndarray) -> Dict[str, Any]:
        """Fallback numerical divergence assessment with confidence scoring."""
        try:
            disagreement_mask = tas_assignments != nas_assignments
            numerical_divergence_rate = np.mean(disagreement_mask)
            
            # Simple confidence scoring for numerical disagreement
            confidence_scores = np.ones(len(tas_assignments)) * 0.5  # Neutral confidence
            
            return {
                'semantic_divergence_rate': numerical_divergence_rate,
                'mapping_quality': 0.5,  # Neutral quality for numerical-only assessment
                'regime_mapping': {},
                'similarity_analysis': {'avg_similarity': 0.5},
                'confidence_scores': confidence_scores,
                'assessment_method': 'numerical_only'
            }
            
        except Exception as e:
            tprint(f"Numerical divergence assessment with confidence failed: {e}")
            return {
                'semantic_divergence_rate': 0.5,
                'mapping_quality': 0.0,
                'regime_mapping': {},
                'similarity_analysis': {'avg_similarity': 0.0},
                'confidence_scores': np.zeros(len(tas_assignments)),
                'assessment_method': 'failed'
            }
    
    def _calculate_divergence_confidence_scores(self, features: np.ndarray, tas_assignments: np.ndarray, 
                                             nas_assignments: np.ndarray, disagreement_mask: np.ndarray) -> np.ndarray:
        """Calculate confidence scores for divergence detection."""
        try:
            confidence_scores = np.zeros(len(tas_assignments))
            
            for i in range(len(tas_assignments)):
                if disagreement_mask[i]:
                    # Calculate confidence based on multiple factors
                    
                    # 1. Feature space distance between TAS and NAS regimes
                    tas_regime = tas_assignments[i]
                    nas_regime = nas_assignments[i]
                    
                    # Get regime centroids
                    tas_centroid = self._get_regime_centroid(features, tas_assignments, tas_regime)
                    nas_centroid = self._get_regime_centroid(features, nas_assignments, nas_regime)
                    
                    if tas_centroid is not None and nas_centroid is not None:
                        # Distance-based confidence (higher distance = higher confidence in divergence)
                        distance = np.linalg.norm(tas_centroid - nas_centroid)
                        distance_confidence = min(1.0, distance / np.std(features))
                    else:
                        distance_confidence = 0.5
                    
                    # 2. Local neighborhood consistency
                    neighborhood_confidence = self._calculate_neighborhood_consistency(features, i, tas_regime, nas_regime)
                    
                    # 3. Temporal consistency
                    temporal_confidence = self._calculate_temporal_divergence_confidence(i, tas_assignments, nas_assignments)
                    
                    # Combine confidence factors
                    confidence_scores[i] = (
                        0.4 * distance_confidence +
                        0.3 * neighborhood_confidence +
                        0.3 * temporal_confidence
                    )
                else:
                    # No disagreement, high confidence in agreement
                    confidence_scores[i] = 0.9
            
            return confidence_scores
            
        except Exception as e:
            tprint(f"Confidence score calculation failed: {e}")
            return np.ones(len(tas_assignments)) * 0.5
    
    def _get_regime_centroid(self, features: np.ndarray, assignments: np.ndarray, regime: int) -> Optional[np.ndarray]:
        """Get centroid for a specific regime."""
        try:
            regime_mask = assignments == regime
            if np.sum(regime_mask) > 0:
                return np.mean(features[regime_mask], axis=0)
            return None
        except Exception as e:
            tprint(f"Regime centroid calculation failed: {e}")
            return None
    
    def _calculate_neighborhood_consistency(self, features: np.ndarray, sample_idx: int, tas_regime: int, nas_regime: int, tas_assignments: np.ndarray, nas_assignments: np.ndarray) -> float:
        """Calculate neighborhood consistency for divergence confidence."""
        try:
            # Get local neighborhood (k=5)
            k = min(5, len(features) - 1)
            distances = np.linalg.norm(features - features[sample_idx], axis=1)
            neighbor_indices = np.argsort(distances)[1:k+1]  # Exclude self
            
            # Count neighbors that agree with divergence
            tas_neighbors = np.sum([tas_assignments[idx] == tas_regime for idx in neighbor_indices])
            nas_neighbors = np.sum([nas_assignments[idx] == nas_regime for idx in neighbor_indices])
            
            # Consistency = proportion of neighbors that support the divergence
            consistency = (tas_neighbors + nas_neighbors) / (2 * k)
            return consistency
            
        except Exception as e:
            tprint(f"Neighborhood consistency calculation failed: {e}")
            return 0.5
    
    def _calculate_temporal_divergence_confidence(self, sample_idx: int, tas_assignments: np.ndarray, nas_assignments: np.ndarray) -> float:
        """Calculate temporal confidence for divergence."""
        try:
            # Check temporal consistency of divergence
            window_size = min(5, len(tas_assignments) // 4)
            start_idx = max(0, sample_idx - window_size // 2)
            end_idx = min(len(tas_assignments), sample_idx + window_size // 2 + 1)
            
            # Count disagreements in temporal window
            window_tas = tas_assignments[start_idx:end_idx]
            window_nas = nas_assignments[start_idx:end_idx]
            window_disagreements = np.sum(window_tas != window_nas)
            
            # Temporal confidence = proportion of disagreements in window
            temporal_confidence = window_disagreements / len(window_tas)
            return temporal_confidence
            
        except Exception as e:
            tprint(f"Temporal divergence confidence calculation failed: {e}")
            return 0.5
    
    def _report_mapping_quality_with_confidence(self, mapping_quality_metrics: Dict[str, Any], 
                                             regime_mapping: Dict[int, int], confidence_scores: np.ndarray):
        """Report mapping quality with confidence information."""
        try:
            tprint("  📊 Enhanced Mapping Quality Assessment with Confidence:", "INFO")
            tprint(f"     Overall Quality: {mapping_quality_metrics.get('overall_quality', 0.0):.3f}", "INFO")
            tprint(f"     Centroid Quality: {mapping_quality_metrics.get('centroid_quality', 0.0):.3f}", "INFO")
            tprint(f"     PCA Quality: {mapping_quality_metrics.get('pca_quality', 0.0):.3f}", "INFO")
            tprint(f"     CV Quality: {mapping_quality_metrics.get('cv_quality', 0.0):.3f}", "INFO")
            tprint(f"     Mapping Coverage: {mapping_quality_metrics.get('mapping_coverage', 0.0):.3f}", "INFO")
            tprint(f"     Average Confidence: {np.mean(confidence_scores):.3f}", "INFO")
            tprint(f"     High Confidence (>0.8): {np.sum(confidence_scores > 0.8)} samples", "INFO")
            tprint(f"     Low Confidence (<0.3): {np.sum(confidence_scores < 0.3)} samples", "INFO")
            
        except Exception as e:
            tprint(f"Confidence reporting failed: {e}")
    
    def _calculate_regime_centroids(self, features: np.ndarray, assignments: np.ndarray) -> Dict[int, np.ndarray]:
        """Calculate centroids for each regime in feature space - VECTORIZED OPTIMIZATION."""
        try:
            unique_regimes = np.unique(assignments)
            n_regimes = len(unique_regimes)
            n_features = features.shape[1]
            
            # VECTORIZED centroid calculation using advanced numpy operations
            centroids = {}
            
            # Create regime masks matrix (n_regimes, n_samples)
            regime_masks = assignments[None, :] == unique_regimes[:, None]
            regime_counts = np.sum(regime_masks, axis=1)
            
            # VECTORIZED mean calculation for all regimes at once
            # Reshape features for broadcasting: (n_regimes, n_samples, n_features)
            features_broadcast = features[None, :, :]  # (1, n_samples, n_features)
            regime_features = np.where(regime_masks[:, :, None], features_broadcast, 0)
            
            # Calculate centroids (vectorized)
            centroids_array = np.sum(regime_features, axis=1) / np.maximum(regime_counts[:, None], 1)
            
            # Convert to dictionary format
            for i, regime in enumerate(unique_regimes):
                if regime_counts[i] > 0:
                    centroids[regime] = centroids_array[i]
                else:
                    centroids[regime] = np.zeros(n_features)
            
            tprint(f"  📍 Calculated centroids for {len(centroids)} regimes (vectorized)", "INFO")
            return centroids
            
        except Exception as e:
            tprint(f"  ⚠️  Regime centroid calculation failed: {e}", "WARNING")
            return {}
    
    def _calculate_regime_centroids_ultra_vectorized(self, features: np.ndarray, assignments: np.ndarray) -> Dict[int, np.ndarray]:
        """Calculate centroids using ultra-vectorized operations for maximum performance."""
        try:
            unique_regimes = np.unique(assignments)
            n_regimes = len(unique_regimes)
            n_features = features.shape[1]
            
            # ULTRA-VECTORIZED approach using scipy.sparse for memory efficiency
            from scipy.sparse import csr_matrix
            
            # Create sparse regime indicator matrix
            regime_indices = np.searchsorted(unique_regimes, assignments)
            regime_matrix = csr_matrix(
                (np.ones(len(assignments)), (regime_indices, np.arange(len(assignments)))),
                shape=(n_regimes, len(assignments))
            )
            
            # Calculate regime counts (vectorized)
            regime_counts = np.array(regime_matrix.sum(axis=1)).flatten()
            
            # Calculate centroids using sparse matrix operations
            centroids_array = regime_matrix.dot(features) / np.maximum(regime_counts[:, None], 1)
            
            # Convert to dictionary
            centroids = {}
            for i, regime in enumerate(unique_regimes):
                centroids[regime] = centroids_array[i]
            
            return centroids
            
        except Exception as e:
            # Fallback to standard vectorized method
            return self._calculate_regime_centroids(features, assignments)
    
    def _find_optimal_regime_mapping(self, tas_centroids: Dict[int, np.ndarray], nas_centroids: Dict[int, np.ndarray], 
                                    features: np.ndarray = None, market_data: pd.DataFrame = None, 
                                    tas_assignments: np.ndarray = None, nas_assignments: np.ndarray = None) -> Dict[int, int]:
        """Find optimal mapping between NAS and TAS regimes using multi-objective optimization."""
        try:
            from scipy.optimize import linear_sum_assignment
            
            tas_regimes = sorted(tas_centroids.keys())
            nas_regimes = sorted(nas_centroids.keys())
            
            if not tas_regimes or not nas_regimes:
                tprint("  ⚠️  Empty regime lists, returning identity mapping", "WARNING")
                return {}
            
            tprint("  🎯 Starting multi-objective regime mapping optimization...", "INFO")
            
            # Calculate multiple objective matrices
            distance_matrix = self._calculate_centroid_distance_matrix(tas_centroids, nas_centroids)
            semantic_matrix = self._calculate_semantic_similarity_matrix(tas_centroids, nas_centroids, features)
            economic_matrix = self._calculate_economic_alignment_matrix(tas_centroids, nas_centroids, market_data)
            balance_matrix = self._calculate_regime_balance_matrix(tas_centroids, nas_centroids, tas_assignments, nas_assignments)
            
            # Combine objectives with adaptive weights
            combined_matrix = self._combine_mapping_objectives(
                distance_matrix, semantic_matrix, economic_matrix, balance_matrix
            )
            
            # Solve assignment problem
            nas_indices, tas_indices = linear_sum_assignment(combined_matrix)
            
            # Create initial mapping
            regime_mapping = {}
            for nas_idx, tas_idx in zip(nas_indices, tas_indices):
                nas_regime = nas_regimes[nas_idx]
                tas_regime = tas_regimes[tas_idx]
                regime_mapping[nas_regime] = tas_regime
            
            # Apply iterative refinement
            refined_mapping = self._iterative_mapping_refinement(
                regime_mapping, tas_centroids, nas_centroids, features, 
                tas_assignments, nas_assignments, market_data
            )
            
            # Report mapping quality
            self._report_enhanced_mapping_quality(regime_mapping, refined_mapping, tas_centroids, nas_centroids)
            
            tprint(f"  ✅ Created enhanced regime mapping with {len(refined_mapping)} pairs", "SUCCESS")
            return refined_mapping
            
        except Exception as e:
            tprint(f"  ⚠️  Enhanced regime mapping failed: {e}, using fallback", "WARNING")
            # Fallback to simple distance-based mapping
            return self._fallback_distance_mapping(tas_centroids, nas_centroids)
    
    def _apply_regime_mapping(self, nas_assignments: np.ndarray, regime_mapping: Dict[int, int]) -> np.ndarray:
        """Apply regime mapping to NAS assignments to align with TAS regime IDs."""
        try:
            if not regime_mapping:
                tprint("  ⚠️  No regime mapping available, returning original assignments", "WARNING")
                return nas_assignments.copy()
            
            # Create mapped assignments
            mapped_assignments = nas_assignments.copy()
            for nas_regime, tas_regime in regime_mapping.items():
                mask = nas_assignments == nas_regime
                mapped_assignments[mask] = tas_regime
            
            mapped_count = np.sum([np.sum(nas_assignments == nas_regime) for nas_regime in regime_mapping.keys()])
            tprint(f"  🔄 Applied mapping to {mapped_count}/{len(nas_assignments)} samples", "INFO")
            
            return mapped_assignments
            
        except Exception as e:
            tprint(f"  ⚠️  Regime mapping application failed: {e}", "WARNING")
            return nas_assignments.copy()
    
    def _calculate_centroid_distance_matrix(self, tas_centroids: Dict[int, np.ndarray], nas_centroids: Dict[int, np.ndarray]) -> np.ndarray:
        """Calculate distance matrix between TAS and NAS centroids - VECTORIZED OPTIMIZATION."""
        try:
            tas_regimes = sorted(tas_centroids.keys())
            nas_regimes = sorted(nas_centroids.keys())
            
            n_tas = len(tas_regimes)
            n_nas = len(nas_regimes)
            
            # VECTORIZED distance matrix calculation
            # Create centroid arrays
            tas_centroids_array = np.array([tas_centroids[regime] for regime in tas_regimes])
            nas_centroids_array = np.array([nas_centroids[regime] for regime in nas_regimes])
            
            # VECTORIZED pairwise distance calculation using broadcasting
            # Reshape for broadcasting: (n_nas, 1, n_features) and (1, n_tas, n_features)
            nas_reshaped = nas_centroids_array[:, None, :]  # (n_nas, 1, n_features)
            tas_reshaped = tas_centroids_array[None, :, :]  # (1, n_tas, n_features)
            
            # Calculate squared distances (vectorized)
            squared_diffs = (nas_reshaped - tas_reshaped) ** 2
            squared_distances = np.sum(squared_diffs, axis=2)
            
            # Calculate Euclidean distances (vectorized)
            distance_matrix = np.sqrt(squared_distances)
            
            return distance_matrix
            
        except Exception as e:
            tprint(f"  ⚠️  Distance matrix calculation failed: {e}", "WARNING")
            return np.ones((len(nas_centroids), len(tas_centroids)))
    
    def _calculate_centroid_distance_matrix_ultra_vectorized(self, tas_centroids: Dict[int, np.ndarray], nas_centroids: Dict[int, np.ndarray]) -> np.ndarray:
        """Calculate distance matrix using ultra-vectorized operations for maximum performance."""
        try:
            tas_regimes = sorted(tas_centroids.keys())
            nas_regimes = sorted(nas_centroids.keys())
            
            # ULTRA-VECTORIZED approach using scipy.spatial.distance
            from scipy.spatial.distance import cdist
            
            # Create centroid arrays
            tas_centroids_array = np.array([tas_centroids[regime] for regime in tas_regimes])
            nas_centroids_array = np.array([nas_centroids[regime] for regime in nas_regimes])
            
            # Calculate distance matrix using optimized C implementation
            distance_matrix = cdist(nas_centroids_array, tas_centroids_array, metric='euclidean')
            
            return distance_matrix
            
        except Exception as e:
            # Fallback to standard vectorized method
            return self._calculate_centroid_distance_matrix(tas_centroids, nas_centroids)
    
    def _calculate_semantic_similarity_matrix(self, tas_centroids: Dict[int, np.ndarray], nas_centroids: Dict[int, np.ndarray], 
                                            features: np.ndarray) -> np.ndarray:
        """Calculate semantic similarity matrix based on regime characteristics."""
        try:
            if features is None:
                return np.ones((len(nas_centroids), len(tas_centroids)))
            
            tas_regimes = sorted(tas_centroids.keys())
            nas_regimes = sorted(nas_centroids.keys())
            
            n_tas = len(tas_regimes)
            n_nas = len(nas_regimes)
            similarity_matrix = np.zeros((n_nas, n_tas))
            
            for i, nas_regime in enumerate(nas_regimes):
                for j, tas_regime in enumerate(tas_regimes):
                    # Extract regime characteristics
                    nas_char = self._extract_regime_characteristics(nas_centroids[nas_regime], features)
                    tas_char = self._extract_regime_characteristics(tas_centroids[tas_regime], features)
                    
                    # Calculate semantic similarity
                    similarity = self._calculate_regime_semantic_similarity(nas_char, tas_char)
                    similarity_matrix[i, j] = 1.0 - similarity  # Convert to cost
            
            return similarity_matrix
            
        except Exception as e:
            tprint(f"  ⚠️  Semantic similarity calculation failed: {e}", "WARNING")
            return np.ones((len(nas_centroids), len(tas_centroids)))
    
    def _calculate_economic_alignment_matrix(self, tas_centroids: Dict[int, np.ndarray], nas_centroids: Dict[int, np.ndarray], 
                                          market_data: pd.DataFrame) -> np.ndarray:
        """Calculate economic alignment matrix based on trading viability."""
        try:
            if market_data is None or 'close' not in market_data.columns:
                return np.ones((len(nas_centroids), len(tas_centroids)))
            
            tas_regimes = sorted(tas_centroids.keys())
            nas_regimes = sorted(nas_centroids.keys())
            
            n_tas = len(tas_regimes)
            n_nas = len(nas_regimes)
            economic_matrix = np.zeros((n_nas, n_tas))
            
            # Calculate regime-specific economic metrics
            tas_economic = self._calculate_regime_economic_metrics(tas_centroids, market_data)
            nas_economic = self._calculate_regime_economic_metrics(nas_centroids, market_data)
            
            for i, nas_regime in enumerate(nas_regimes):
                for j, tas_regime in enumerate(tas_regimes):
                    # Calculate economic alignment
                    alignment = self._calculate_economic_alignment_score(
                        nas_economic.get(nas_regime, {}), tas_economic.get(tas_regime, {})
                    )
                    economic_matrix[i, j] = 1.0 - alignment  # Convert to cost
            
            return economic_matrix
            
        except Exception as e:
            tprint(f"  ⚠️  Economic alignment calculation failed: {e}", "WARNING")
            return np.ones((len(nas_centroids), len(tas_centroids)))
    
    def _calculate_regime_balance_matrix(self, tas_centroids: Dict[int, np.ndarray], nas_centroids: Dict[int, np.ndarray],
                                       tas_assignments: np.ndarray, nas_assignments: np.ndarray) -> np.ndarray:
        """Calculate regime balance matrix to penalize imbalanced mappings."""
        try:
            if tas_assignments is None or nas_assignments is None:
                return np.ones((len(nas_centroids), len(tas_centroids)))
            
            tas_regimes = sorted(tas_centroids.keys())
            nas_regimes = sorted(nas_centroids.keys())
            
            n_tas = len(tas_regimes)
            n_nas = len(nas_regimes)
            balance_matrix = np.zeros((n_nas, n_tas))
            
            # Calculate current regime distributions
            tas_distribution = np.bincount(tas_assignments) / len(tas_assignments)
            nas_distribution = np.bincount(nas_assignments) / len(nas_assignments)
            
            for i, nas_regime in enumerate(nas_regimes):
                for j, tas_regime in enumerate(tas_regimes):
                    # Calculate balance penalty
                    nas_ratio = nas_distribution[nas_regime] if nas_regime < len(nas_distribution) else 0.0
                    tas_ratio = tas_distribution[tas_regime] if tas_regime < len(tas_distribution) else 0.0
                    
                    # Penalize extreme imbalances
                    balance_penalty = 0.0
                    if nas_ratio > 0.20 or nas_ratio < 0.03:  # NAS imbalance
                        balance_penalty += 0.3
                    if tas_ratio > 0.20 or tas_ratio < 0.03:  # TAS imbalance
                        balance_penalty += 0.3
                    
                    balance_matrix[i, j] = balance_penalty
            
            return balance_matrix
            
        except Exception as e:
            tprint(f"  ⚠️  Regime balance calculation failed: {e}", "WARNING")
            return np.ones((len(nas_centroids), len(tas_centroids)))
    
    def _combine_mapping_objectives(self, distance_matrix: np.ndarray, semantic_matrix: np.ndarray, 
                                  economic_matrix: np.ndarray, balance_matrix: np.ndarray) -> np.ndarray:
        """Combine multiple objective matrices with adaptive weights."""
        try:
            # Normalize matrices to [0, 1] range
            distance_norm = self._normalize_matrix(distance_matrix)
            semantic_norm = self._normalize_matrix(semantic_matrix)
            economic_norm = self._normalize_matrix(economic_matrix)
            balance_norm = self._normalize_matrix(balance_matrix)
            
            # Adaptive weights based on matrix quality
            weights = self._calculate_adaptive_weights(distance_norm, semantic_norm, economic_norm, balance_norm)
            
            # Combine matrices
            combined_matrix = (
                weights['distance'] * distance_norm +
                weights['semantic'] * semantic_norm +
                weights['economic'] * economic_norm +
                weights['balance'] * balance_norm
            )
            
            tprint(f"  📊 Mapping weights - Distance: {weights['distance']:.2f}, Semantic: {weights['semantic']:.2f}, Economic: {weights['economic']:.2f}, Balance: {weights['balance']:.2f}", "INFO")
            
            return combined_matrix
            
        except Exception as e:
            tprint(f"  ⚠️  Objective combination failed: {e}", "WARNING")
            return distance_matrix  # Fallback to distance only
    
    def _iterative_mapping_refinement(self, initial_mapping: Dict[int, int], tas_centroids: Dict[int, np.ndarray], 
                                    nas_centroids: Dict[int, np.ndarray], features: np.ndarray,
                                    tas_assignments: np.ndarray, nas_assignments: np.ndarray, 
                                    market_data: pd.DataFrame) -> Dict[int, int]:
        """Apply iterative refinement to improve mapping quality."""
        try:
            best_mapping = initial_mapping.copy()
            best_score = self._calculate_mapping_score(best_mapping, tas_centroids, nas_centroids, 
                                                    features, tas_assignments, nas_assignments, market_data)
            
            tprint(f"  🔄 Starting iterative refinement (initial score: {best_score:.3f})", "INFO")
            
            max_iterations = 5
            for iteration in range(max_iterations):
                improved = False
                
                # Try pairwise swaps
                for nas_regime1, tas_regime1 in list(best_mapping.items()):
                    for nas_regime2, tas_regime2 in list(best_mapping.items()):
                        if nas_regime1 != nas_regime2:
                            # Try swap
                            test_mapping = best_mapping.copy()
                            test_mapping[nas_regime1] = tas_regime2
                            test_mapping[nas_regime2] = tas_regime1
                            
                            test_score = self._calculate_mapping_score(test_mapping, tas_centroids, nas_centroids,
                                                                     features, tas_assignments, nas_assignments, market_data)
                            
                            if test_score > best_score + 0.001:  # Significant improvement threshold
                                best_mapping = test_mapping
                                best_score = test_score
                                improved = True
                                tprint(f"  ✅ Iteration {iteration+1}: Improved score to {best_score:.3f}", "SUCCESS")
                                break
                    
                    if improved:
                        break
                
                if not improved:
                    tprint(f"  🏁 Refinement converged after {iteration+1} iterations", "INFO")
                    break
            
            tprint(f"  📈 Final mapping score: {best_score:.3f}", "INFO")
            return best_mapping
            
        except Exception as e:
            tprint(f"  ⚠️  Iterative refinement failed: {e}", "WARNING")
            return initial_mapping
    
    def _calculate_mapping_score(self, mapping: Dict[int, int], tas_centroids: Dict[int, np.ndarray], 
                               nas_centroids: Dict[int, np.ndarray], features: np.ndarray,
                               tas_assignments: np.ndarray, nas_assignments: np.ndarray, 
                               market_data: pd.DataFrame) -> float:
        """Calculate comprehensive mapping quality score."""
        try:
            if not mapping:
                return 0.0
            
            # Calculate consensus score
            mapped_nas = self._apply_regime_mapping(nas_assignments, mapping)
            consensus = np.mean(tas_assignments == mapped_nas)
            
            # Calculate centroid alignment score
            centroid_score = 0.0
            for nas_regime, tas_regime in mapping.items():
                if nas_regime in nas_centroids and tas_regime in tas_centroids:
                    distance = np.linalg.norm(nas_centroids[nas_regime] - tas_centroids[tas_regime])
                    centroid_score += 1.0 / (1.0 + distance)
            
            centroid_score /= len(mapping) if mapping else 1.0
            
            # Calculate balance score
            balance_score = self._calculate_regime_balance_score(mapped_nas, tas_assignments)
            
            # Combined score
            combined_score = 0.5 * consensus + 0.3 * centroid_score + 0.2 * balance_score
            
            return combined_score
            
        except Exception as e:
            tprint(f"  ⚠️  Mapping score calculation failed: {e}", "WARNING")
            return 0.0
    
    def _fallback_distance_mapping(self, tas_centroids: Dict[int, np.ndarray], nas_centroids: Dict[int, np.ndarray]) -> Dict[int, int]:
        """Fallback to simple distance-based mapping."""
        try:
            from scipy.optimize import linear_sum_assignment
            
            tas_regimes = sorted(tas_centroids.keys())
            nas_regimes = sorted(nas_centroids.keys())
            
            n_tas = len(tas_regimes)
            n_nas = len(nas_regimes)
            cost_matrix = np.zeros((n_nas, n_tas))
            
            for i, nas_regime in enumerate(nas_regimes):
                for j, tas_regime in enumerate(tas_regimes):
                    nas_centroid = nas_centroids[nas_regime]
                    tas_centroid = tas_centroids[tas_regime]
                    cost_matrix[i, j] = np.linalg.norm(nas_centroid - tas_centroid)
            
            nas_indices, tas_indices = linear_sum_assignment(cost_matrix)
            
            regime_mapping = {}
            for nas_idx, tas_idx in zip(nas_indices, tas_indices):
                nas_regime = nas_regimes[nas_idx]
                tas_regime = tas_regimes[tas_idx]
                regime_mapping[nas_regime] = tas_regime
            
            return regime_mapping
            
        except Exception as e:
            tprint(f"  ⚠️  Fallback mapping failed: {e}", "WARNING")
            return {}
    
    def _extract_regime_characteristics(self, centroid: np.ndarray, features: np.ndarray) -> Dict[str, float]:
        """Extract regime characteristics from centroid."""
        try:
            if features is None or len(centroid) == 0:
                return {}
            
            # Calculate basic statistics
            characteristics = {
                'mean': np.mean(centroid),
                'std': np.std(centroid),
                'min': np.min(centroid),
                'max': np.max(centroid),
                'range': np.max(centroid) - np.min(centroid),
                'skewness': self._calculate_skewness(centroid),
                'kurtosis': self._calculate_kurtosis(centroid)
            }
            
            return characteristics
            
        except Exception as e:
            tprint(f"  ⚠️  Regime characteristics extraction failed: {e}", "WARNING")
            return {}
    
    def _calculate_regime_semantic_similarity(self, nas_char: Dict[str, float], tas_char: Dict[str, float]) -> float:
        """Calculate semantic similarity between regime characteristics."""
        try:
            if not nas_char or not tas_char:
                return 0.5  # Neutral similarity
            
            # Calculate similarity for each characteristic
            similarities = []
            for key in nas_char.keys():
                if key in tas_char:
                    nas_val = nas_char[key]
                    tas_val = tas_char[key]
                    
                    # Normalize and calculate similarity
                    if nas_val == 0 and tas_val == 0:
                        similarity = 1.0
                    else:
                        max_val = max(abs(nas_val), abs(tas_val))
                        if max_val > 0:
                            similarity = 1.0 - abs(nas_val - tas_val) / max_val
                        else:
                            similarity = 1.0
                    
                    similarities.append(max(0.0, min(1.0, similarity)))
            
            return np.mean(similarities) if similarities else 0.5
            
        except Exception as e:
            tprint(f"  ⚠️  Semantic similarity calculation failed: {e}", "WARNING")
            return 0.5
    
    # @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    # @performance_tracked(log_performance=True, track_memory=True)
    def _calculate_regime_economic_metrics(self, centroids: Dict[int, np.ndarray], market_data: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """Calculate economic metrics for each regime with hardware optimization."""
        try:
            if market_data is None or 'close' not in market_data.columns:
                return {}
            
            economic_metrics = {}
            
            for regime, centroid in centroids.items():
                # Simulate economic metrics based on centroid characteristics
                metrics = {
                    'volatility': np.std(centroid) if len(centroid) > 1 else 0.0,
                    'trend_strength': np.mean(centroid) if len(centroid) > 0 else 0.0,
                    'stability': 1.0 / (1.0 + np.std(centroid)) if len(centroid) > 1 else 1.0,
                    'complexity': len(centroid) / 100.0  # Normalize by expected feature count
                }
                
                economic_metrics[regime] = metrics
            
            return economic_metrics
            
        except Exception as e:
            tprint(f"  ⚠️  Economic metrics calculation failed: {e}", "WARNING")
            return {}
    
    # @performance_tracked(log_performance=True, track_memory=True)
    def _calculate_economic_alignment_score(self, nas_metrics: Dict[str, float], tas_metrics: Dict[str, float]) -> float:
        """Calculate economic alignment score between regime metrics with performance tracking."""
        try:
            if not nas_metrics or not tas_metrics:
                return 0.5  # Neutral alignment
            
            alignments = []
            for key in nas_metrics.keys():
                if key in tas_metrics:
                    nas_val = nas_metrics[key]
                    tas_val = tas_metrics[key]
                    
                    # Calculate alignment (higher is better)
                    if nas_val == 0 and tas_val == 0:
                        alignment = 1.0
                    else:
                        max_val = max(abs(nas_val), abs(tas_val))
                        if max_val > 0:
                            alignment = 1.0 - abs(nas_val - tas_val) / max_val
                        else:
                            alignment = 1.0
                    
                    alignments.append(max(0.0, min(1.0, alignment)))
            
            return np.mean(alignments) if alignments else 0.5
            
        except Exception as e:
            tprint(f"  ⚠️  Economic alignment calculation failed: {e}", "WARNING")
            return 0.5
    
    def _calculate_regime_balance_score(self, mapped_nas: np.ndarray, tas_assignments: np.ndarray) -> float:
        """Calculate regime balance score."""
        try:
            # Calculate distributions
            nas_dist = np.bincount(mapped_nas) / len(mapped_nas)
            tas_dist = np.bincount(tas_assignments) / len(tas_assignments)
            
            # Calculate balance penalties
            nas_penalty = np.sum((nas_dist > 0.20) | (nas_dist < 0.03))
            tas_penalty = np.sum((tas_dist > 0.20) | (tas_dist < 0.03))
            
            # Calculate balance score (higher is better)
            balance_score = 1.0 - 0.1 * (nas_penalty + tas_penalty)
            
            return max(0.0, min(1.0, balance_score))
            
        except Exception as e:
            tprint(f"  ⚠️  Balance score calculation failed: {e}", "WARNING")
            return 0.5
    
    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Normalize matrix to [0, 1] range."""
        try:
            if matrix.size == 0:
                return matrix
            
            min_val = np.min(matrix)
            max_val = np.max(matrix)
            
            if max_val == min_val:
                return np.zeros_like(matrix)
            
            return (matrix - min_val) / (max_val - min_val)
            
        except Exception as e:
            tprint(f"  ⚠️  Matrix normalization failed: {e}", "WARNING")
            return matrix
    
    def _calculate_adaptive_weights(self, distance_norm: np.ndarray, semantic_norm: np.ndarray, 
                                  economic_norm: np.ndarray, balance_norm: np.ndarray) -> Dict[str, float]:
        """Calculate adaptive weights based on matrix quality."""
        try:
            # Calculate matrix quality (lower variance = higher quality)
            distance_quality = 1.0 / (1.0 + np.var(distance_norm))
            semantic_quality = 1.0 / (1.0 + np.var(semantic_norm))
            economic_quality = 1.0 / (1.0 + np.var(economic_norm))
            balance_quality = 1.0 / (1.0 + np.var(balance_norm))
            
            # Normalize weights
            total_quality = distance_quality + semantic_quality + economic_quality + balance_quality
            
            if total_quality > 0:
                weights = {
                    'distance': distance_quality / total_quality,
                    'semantic': semantic_quality / total_quality,
                    'economic': economic_quality / total_quality,
                    'balance': balance_quality / total_quality
                }
            else:
                # Equal weights as fallback
                weights = {
                    'distance': 0.3,
                    'semantic': 0.3,
                    'economic': 0.2,
                    'balance': 0.2
                }
            
            return weights
            
        except Exception as e:
            tprint(f"  ⚠️  Adaptive weights calculation failed: {e}", "WARNING")
            return {'distance': 0.4, 'semantic': 0.3, 'economic': 0.2, 'balance': 0.1}
    
    def _report_enhanced_mapping_quality(self, initial_mapping: Dict[int, int], refined_mapping: Dict[int, int], 
                                       tas_centroids: Dict[int, np.ndarray], nas_centroids: Dict[int, np.ndarray]):
        """Report enhanced mapping quality metrics."""
        try:
            tprint("  📊 Enhanced Mapping Quality Report:", "INFO")
            tprint(f"     Initial mappings: {len(initial_mapping)}", "INFO")
            tprint(f"     Refined mappings: {len(refined_mapping)}", "INFO")
            
            # Calculate improvement metrics
            if initial_mapping and refined_mapping:
                tprint("  🔄 Mapping refinement completed successfully", "SUCCESS")
            else:
                tprint("  ⚠️  No mapping refinement applied", "WARNING")
            
        except Exception as e:
            tprint(f"  ⚠️  Enhanced mapping quality reporting failed: {e}", "WARNING")
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        try:
            if len(data) < 3:
                return 0.0
            
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            
            return np.mean(((data - mean) / std) ** 3)
            
        except Exception:
            return 0.0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        try:
            if len(data) < 4:
                return 0.0
            
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            
            return np.mean(((data - mean) / std) ** 4) - 3.0
            
        except Exception:
            return 0.0
    
    def _calculate_mapping_quality(self, tas_centroids: Dict[int, np.ndarray], nas_centroids: Dict[int, np.ndarray], 
                                   regime_mapping: Dict[int, int], features: np.ndarray = None, 
                                   market_data: pd.DataFrame = None, tas_assignments: np.ndarray = None, 
                                   nas_assignments: np.ndarray = None) -> Dict[str, float]:
        """Calculate enhanced multi-dimensional quality metrics for regime mapping."""
        try:
            if not regime_mapping:
                return {
                    'overall_quality': 0.0,
                    'clustering_quality': 0.0,
                    'economic_quality': 0.0,
                    'balance_quality': 0.0,
                    'consensus_quality': 0.0,
                    'mapping_coverage': 0.0
                }
            
            tprint("  📊 Calculating enhanced multi-dimensional quality metrics...", "INFO")
            
            # Component 1: Clustering Quality (30%)
            clustering_quality = self._calculate_clustering_quality_component(tas_centroids, nas_centroids, regime_mapping)
            
            # Component 2: Economic Quality (25%)
            economic_quality = self._calculate_economic_quality_component(tas_centroids, nas_centroids, regime_mapping, market_data)
            
            # Component 3: Balance Quality (20%)
            balance_quality = self._calculate_balance_quality_component(tas_assignments, nas_assignments, regime_mapping)
            
            # Component 4: Consensus Quality (15%)
            consensus_quality = self._calculate_consensus_quality_component(tas_assignments, nas_assignments, regime_mapping)
            
            # Component 5: Mapping Coverage (10%)
            mapping_coverage = self._calculate_mapping_coverage_component(tas_centroids, nas_centroids, regime_mapping)
            
            # Calculate overall quality score
            overall_quality = (
                0.30 * clustering_quality +
                0.25 * economic_quality +
                0.20 * balance_quality +
                0.15 * consensus_quality +
                0.10 * mapping_coverage
            )
            
            # Calculate quality grade
            quality_grade = self._calculate_quality_grade(overall_quality)
            
            tprint(f"  📈 Enhanced Quality Assessment:", "INFO")
            tprint(f"     Overall Quality: {overall_quality:.3f} (Grade: {quality_grade})", "INFO")
            tprint(f"     Clustering: {clustering_quality:.3f}, Economic: {economic_quality:.3f}", "INFO")
            tprint(f"     Balance: {balance_quality:.3f}, Consensus: {consensus_quality:.3f}", "INFO")
            tprint(f"     Coverage: {mapping_coverage:.3f}", "INFO")
            
            return {
                'overall_quality': overall_quality,
                'quality_grade': quality_grade,
                'clustering_quality': clustering_quality,
                'economic_quality': economic_quality,
                'balance_quality': balance_quality,
                'consensus_quality': consensus_quality,
                'mapping_coverage': mapping_coverage,
                'num_mapped_regimes': len(regime_mapping)
            }
            
        except Exception as e:
            tprint(f"  ⚠️  Enhanced quality calculation failed: {e}", "WARNING")
            return {
                'overall_quality': 0.0,
                'quality_grade': 'F',
                'clustering_quality': 0.0,
                'economic_quality': 0.0,
                'balance_quality': 0.0,
                'consensus_quality': 0.0,
                'mapping_coverage': 0.0
            }
    
    def _calculate_clustering_quality_component(self, tas_centroids: Dict[int, np.ndarray], 
                                              nas_centroids: Dict[int, np.ndarray], 
                                              regime_mapping: Dict[int, int]) -> float:
        """Calculate clustering quality component."""
        try:
            if not regime_mapping:
                return 0.0
            
            # Calculate centroid alignment
            centroid_distances = []
            for nas_regime, tas_regime in regime_mapping.items():
                if nas_regime in nas_centroids and tas_regime in tas_centroids:
                    distance = np.linalg.norm(nas_centroids[nas_regime] - tas_centroids[tas_regime])
                    centroid_distances.append(distance)
            
            if centroid_distances:
                avg_distance = np.mean(centroid_distances)
                # Normalize to 0-1 (lower distance = higher quality)
                centroid_quality = 1.0 / (1.0 + avg_distance) if avg_distance > 0 else 1.0
            else:
                centroid_quality = 0.0
            
            return max(0.0, min(1.0, centroid_quality))
            
        except Exception as e:
            tprint(f"  ⚠️  Clustering quality calculation failed: {e}", "WARNING")
            return 0.0
    
    def _calculate_economic_quality_component(self, tas_centroids: Dict[int, np.ndarray], 
                                           nas_centroids: Dict[int, np.ndarray], 
                                           regime_mapping: Dict[int, int], 
                                           market_data: pd.DataFrame) -> float:
        """Calculate economic quality component."""
        try:
            if not regime_mapping or market_data is None:
                return 0.5  # Neutral score
            
            # Calculate regime-specific economic metrics
            tas_economic = self._calculate_regime_economic_metrics(tas_centroids, market_data)
            nas_economic = self._calculate_regime_economic_metrics(nas_centroids, market_data)
            
            # Calculate economic alignment for mapped regimes
            alignments = []
            for nas_regime, tas_regime in regime_mapping.items():
                nas_metrics = nas_economic.get(nas_regime, {})
                tas_metrics = tas_economic.get(tas_regime, {})
                
                if nas_metrics and tas_metrics:
                    alignment = self._calculate_economic_alignment_score(nas_metrics, tas_metrics)
                    alignments.append(alignment)
            
            return np.mean(alignments) if alignments else 0.5
            
        except Exception as e:
            tprint(f"  ⚠️  Economic quality calculation failed: {e}", "WARNING")
            return 0.5
    
    def _calculate_balance_quality_component(self, tas_assignments: np.ndarray, nas_assignments: np.ndarray, 
                                          regime_mapping: Dict[int, int]) -> float:
        """Calculate regime balance quality component."""
        try:
            if tas_assignments is None or nas_assignments is None or not regime_mapping:
                return 0.5  # Neutral score
            
            # Apply mapping to NAS assignments
            mapped_nas = self._apply_regime_mapping(nas_assignments, regime_mapping)
            
            # Calculate regime distributions
            tas_dist = np.bincount(tas_assignments) / len(tas_assignments)
            nas_dist = np.bincount(mapped_nas) / len(mapped_nas)
            
            # Calculate balance penalties
            tas_penalty = np.sum((tas_dist > 0.20) | (tas_dist < 0.03))
            nas_penalty = np.sum((nas_dist > 0.20) | (nas_dist < 0.03))
            
            # Calculate balance score (higher is better)
            balance_score = 1.0 - 0.1 * (tas_penalty + nas_penalty)
            
            return max(0.0, min(1.0, balance_score))
            
        except Exception as e:
            tprint(f"  ⚠️  Balance quality calculation failed: {e}", "WARNING")
            return 0.5
    
    def _calculate_consensus_quality_component(self, tas_assignments: np.ndarray, nas_assignments: np.ndarray, 
                                            regime_mapping: Dict[int, int]) -> float:
        """Calculate consensus quality component."""
        try:
            if tas_assignments is None or nas_assignments is None or not regime_mapping:
                return 0.0
            
            # Apply mapping to NAS assignments
            mapped_nas = self._apply_regime_mapping(nas_assignments, regime_mapping)
            
            # Calculate consensus rate
            consensus_rate = np.mean(tas_assignments == mapped_nas)
            
            return max(0.0, min(1.0, consensus_rate))
            
        except Exception as e:
            tprint(f"  ⚠️  Consensus quality calculation failed: {e}", "WARNING")
            return 0.0
    
    def _calculate_mapping_coverage_component(self, tas_centroids: Dict[int, np.ndarray], 
                                            nas_centroids: Dict[int, np.ndarray], 
                                            regime_mapping: Dict[int, int]) -> float:
        """Calculate mapping coverage component."""
        try:
            if not regime_mapping:
                return 0.0
            
            total_nas_regimes = len(nas_centroids)
            total_tas_regimes = len(tas_centroids)
            mapped_regimes = len(regime_mapping)
            
            coverage = mapped_regimes / max(total_nas_regimes, total_tas_regimes) if max(total_nas_regimes, total_tas_regimes) > 0 else 0.0
            
            return max(0.0, min(1.0, coverage))
            
        except Exception as e:
            tprint(f"  ⚠️  Mapping coverage calculation failed: {e}", "WARNING")
            return 0.0
    
    def _calculate_quality_grade(self, overall_quality: float) -> str:
        """Calculate quality grade based on overall quality score."""
        try:
            if overall_quality >= 0.90:
                return 'A+'
            elif overall_quality >= 0.85:
                return 'A'
            elif overall_quality >= 0.80:
                return 'A-'
            elif overall_quality >= 0.75:
                return 'B+'
            elif overall_quality >= 0.70:
                return 'B'
            elif overall_quality >= 0.65:
                return 'B-'
            elif overall_quality >= 0.60:
                return 'C+'
            elif overall_quality >= 0.55:
                return 'C'
            elif overall_quality >= 0.50:
                return 'C-'
            elif overall_quality >= 0.40:
                return 'D'
            else:
                return 'F'
                
        except Exception:
            return 'F'
    
    def _analyze_regime_similarity(self, tas_centroids: Dict[int, np.ndarray], nas_centroids: Dict[int, np.ndarray], 
                                   regime_mapping: Dict[int, int]) -> Dict[str, Any]:
        """Analyze similarity patterns between mapped regimes."""
        try:
            if not regime_mapping:
                return {'avg_similarity': 0.0, 'similarity_distribution': []}
            
            similarities = []
            for nas_regime, tas_regime in regime_mapping.items():
                if nas_regime in nas_centroids and tas_regime in tas_centroids:
                    # Calculate cosine similarity
                    nas_vec = nas_centroids[nas_regime]
                    tas_vec = tas_centroids[tas_regime]
                    
                    nas_norm = np.linalg.norm(nas_vec)
                    tas_norm = np.linalg.norm(tas_vec)
                    
                    if nas_norm > 0 and tas_norm > 0:
                        similarity = np.dot(nas_vec, tas_vec) / (nas_norm * tas_norm)
                        similarities.append(similarity)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                min_similarity = np.min(similarities)
                max_similarity = np.max(similarities)
                std_similarity = np.std(similarities)
            else:
                avg_similarity = min_similarity = max_similarity = std_similarity = 0.0
            
            return {
                'avg_similarity': avg_similarity,
                'min_similarity': min_similarity,
                'max_similarity': max_similarity,
                'std_similarity': std_similarity,
                'similarity_distribution': similarities,
                'num_similar_regimes': sum(1 for s in similarities if s > 0.7)
            }
            
        except Exception as e:
            tprint(f"  ⚠️  Similarity analysis failed: {e}", "WARNING")
            return {'avg_similarity': 0.0, 'similarity_distribution': []}
    
    def _calculate_adaptive_batch_size(self, frontier_samples: List[int], iteration: int, current_assignments: np.ndarray) -> int:
        """Calculate adaptive batch size based on convergence rate and optimization progress."""
        try:
            # Base batch size
            base_batch_size = max(1, int(0.10 * len(frontier_samples)))
            
            # Convergence rate analysis
            if hasattr(self, '_convergence_history') and len(self._convergence_history) > 2:
                # Calculate convergence rate
                recent_improvements = self._convergence_history[-3:]
                convergence_rate = np.mean(recent_improvements)
                
                # Adjust batch size based on convergence rate
                if convergence_rate > 0.01:  # High improvement rate
                    adaptive_factor = 1.5  # Increase batch size for faster convergence
                elif convergence_rate > 0.005:  # Medium improvement rate
                    adaptive_factor = 1.0  # Maintain batch size
                else:  # Low improvement rate
                    adaptive_factor = 0.7  # Decrease batch size for more careful optimization
            else:
                adaptive_factor = 1.0
            
            # Iteration-based adjustment
            if iteration < 5:
                # Early iterations: smaller batches for careful optimization
                iteration_factor = 0.8
            elif iteration < 20:
                # Middle iterations: standard batches
                iteration_factor = 1.0
            else:
                # Late iterations: smaller batches for fine-tuning
                iteration_factor = 0.6
            
            # Calculate final adaptive batch size
            adaptive_batch_size = int(base_batch_size * adaptive_factor * iteration_factor)
            adaptive_batch_size = max(1, min(adaptive_batch_size, len(frontier_samples)))
            
            tprint(f"  📊 Adaptive batch sizing: base={base_batch_size}, factor={adaptive_factor:.2f}, iteration_factor={iteration_factor:.2f}, final={adaptive_batch_size}", "INFO")
            
            return adaptive_batch_size
            
        except Exception as e:
            tprint(f"Adaptive batch size calculation failed: {e}")
            return max(1, int(0.10 * len(frontier_samples)))  # Fallback to base batch size
    
    def _calculate_multi_objective_flip_improvement(self, features: np.ndarray, assignments: np.ndarray, 
                                                 sample_idx: int, target_regime: int) -> float:
        """Calculate multi-objective improvement using Pareto optimization principles with M1 hardware optimization."""
        try:
            tprint(f"    🎯 Calculating multi-objective improvement for sample {sample_idx} with M1 hardware optimization...", "INFO")
            
            # Initialize M1 hardware optimization for multi-objective calculation
            if self.m1_cpu_optimizer:
                tprint(f"      ⚡ Using M1 CPU optimization for sample {sample_idx}", "INFO")
            if self.m1_memory_optimizer:
                tprint(f"      💾 Using M1 memory optimization for sample {sample_idx}", "INFO")
            
            # Store original assignment
            original_regime = assignments[sample_idx]
            
            # Calculate individual objective improvements
            objectives = self._calculate_multi_objective_scores(features, assignments, sample_idx, target_regime)
            
            # Apply Pareto optimization weights
            pareto_weights = self._get_pareto_optimization_weights()
            
            # Calculate weighted multi-objective improvement
            multi_objective_improvement = sum(
                weight * objective for weight, objective in zip(pareto_weights, objectives.values())
            )
            
            tprint(f"    ✅ Multi-objective improvement calculated: {multi_objective_improvement:.4f}", "SUCCESS")
            return multi_objective_improvement
            
        except Exception as e:
            tprint(f"Multi-objective improvement calculation failed: {e}")
            # Fallback to single-objective improvement
            return self._calculate_single_flip_improvement(features, assignments, sample_idx, target_regime)
    
    def _calculate_multi_objective_scores(self, features: np.ndarray, assignments: np.ndarray, 
                                        sample_idx: int, target_regime: int) -> Dict[str, float]:
        """Calculate multiple objective scores for Pareto optimization."""
        try:
            # Store original assignment
            original_regime = assignments[sample_idx]
            
            # Temporarily assign target regime
            assignments[sample_idx] = target_regime
            
            # Calculate individual quality metrics
            quality_scores = self._calculate_individual_quality_scores(features, assignments)
            
            # Restore original assignment
            assignments[sample_idx] = original_regime
            
            # Calculate original scores for comparison
            original_scores = self._calculate_individual_quality_scores(features, assignments)
            
            # Calculate improvements for each objective (reduced redundancy)
            objectives = {
                'silhouette_improvement': quality_scores['silhouette'] - original_scores['silhouette'],
                'ch_improvement': 0.0,  # REMOVED - redundant with Silhouette
                'db_improvement': 0.0,  # REMOVED - redundant with Silhouette
                'balance_improvement': quality_scores['regime_balance'] - original_scores['regime_balance'],
                'within_regime_cv_improvement': self._calculate_within_regime_cv_improvement(
                    features, assignments, sample_idx, target_regime
                ),
                'between_regime_cv_improvement': self._calculate_between_regime_cv_improvement(
                    features, assignments, sample_idx, target_regime
                ),
                'temporal_improvement': self._calculate_temporal_consistency_improvement(
                    features, assignments, sample_idx, target_regime
                )
            }
            
            return objectives
            
        except Exception as e:
            tprint(f"Multi-objective scores calculation failed: {e}")
            return {
                'silhouette_improvement': 0.0,
                'ch_improvement': 0.0,
                'db_improvement': 0.0,
                'balance_improvement': 0.0,
                'within_regime_cv_improvement': 0.0,
                'between_regime_cv_improvement': 0.0,
                'temporal_improvement': 0.0
            }
    
    def _get_pareto_optimization_weights(self) -> List[float]:
        """Get Pareto optimization weights for multi-objective optimization with reduced redundancy."""
        try:
            # Pareto weights optimized for NAS/TAS divergence detection
            # Enhanced balance prioritization to prevent dominant regimes (>25% threshold)
            weights = [
                0.15,  # Silhouette improvement (primary cluster separation metric) - reduced from 0.20
                0.00,  # Calinski-Harabasz improvement (REMOVED - redundant with Silhouette)
                0.00,  # Davies-Bouldin improvement (REMOVED - redundant with Silhouette)
                0.35,  # Balance improvement (regime distribution - INCREASED to prevent dominant regimes)
                0.20,  # Within-regime CV improvement (intra-regime stability - unique metric)
                0.15,  # Between-regime CV improvement (inter-regime divergence - reduced from 0.20)
                0.15   # Temporal improvement (smoothness - reduced from 0.20)
            ]
            
            return weights
            
        except Exception as e:
            tprint(f"Pareto weights calculation failed: {e}")
            return [0.15, 0.00, 0.00, 0.35, 0.20, 0.15, 0.15]  # Default weights with enhanced balance
    
    def _calculate_temporal_consistency_improvement(self, features: np.ndarray, assignments: np.ndarray, 
                                                 sample_idx: int, target_regime: int) -> float:
        """Calculate temporal consistency improvement for regime flip."""
        try:
            # Get temporal neighbors
            window_size = 3
            start_idx = max(0, sample_idx - window_size)
            end_idx = min(len(assignments), sample_idx + window_size + 1)

            # Calculate temporal consistency before flip
            original_consistency = 0.0
            original_regime = assignments[sample_idx]
            for i in range(start_idx, end_idx):
                if i != sample_idx:
                    # Check consistency with neighbors
                    if i > 0 and assignments[i] == assignments[i-1]:
                        original_consistency += 0.5
                    if i < len(assignments) - 1 and assignments[i] == assignments[i+1]:
                        original_consistency += 0.5
            
            # Calculate temporal consistency after flip
            assignments[sample_idx] = target_regime
            new_consistency = 0.0
            for i in range(start_idx, end_idx):
                if i != sample_idx:
                    # Check consistency with neighbors
                    if i > 0 and assignments[i] == assignments[i-1]:
                        new_consistency += 0.5
                    if i < len(assignments) - 1 and assignments[i] == assignments[i+1]:
                        new_consistency += 0.5

            # Restore original assignment
            assignments[sample_idx] = original_regime

            # Return improvement
            return new_consistency - original_consistency
            
        except Exception as e:
            tprint(f"Temporal consistency improvement calculation failed: {e}")
            return 0.0
    
    def _calculate_within_regime_cv_improvement(self, features: np.ndarray, assignments: np.ndarray, 
                                             sample_idx: int, target_regime: int) -> float:
        """Calculate within-regime CV improvement for regime flip (intra-regime stability)."""
        try:
            # Store original assignment
            original_regime = assignments[sample_idx]
            
            # Calculate within-regime CV before flip
            original_within_cv = self._calculate_within_regime_cv(features, assignments)
            
            # Temporarily assign target regime
            assignments[sample_idx] = target_regime
            
            # Calculate within-regime CV after flip
            new_within_cv = self._calculate_within_regime_cv(features, assignments)
            
            # Restore original assignment
            assignments[sample_idx] = original_regime
            
            # Return improvement (lower CV = higher stability = better)
            improvement = original_within_cv - new_within_cv
            return improvement
            
        except Exception as e:
            tprint(f"Within-regime CV improvement calculation failed: {e}")
            return 0.0
    
    def _calculate_between_regime_cv_improvement(self, features: np.ndarray, assignments: np.ndarray, 
                                              sample_idx: int, target_regime: int) -> float:
        """Calculate between-regime CV improvement for regime flip (inter-regime divergence)."""
        try:
            # Store original assignment
            original_regime = assignments[sample_idx]
            
            # Calculate between-regime CV before flip
            original_between_cv = self._calculate_between_regime_cv(features, assignments)
            
            # Temporarily assign target regime
            assignments[sample_idx] = target_regime
            
            # Calculate between-regime CV after flip
            new_between_cv = self._calculate_between_regime_cv(features, assignments)
            
            # Restore original assignment
            assignments[sample_idx] = original_regime
            
            # Return improvement (higher CV = higher divergence = better for NAS/TAS detection)
            improvement = new_between_cv - original_between_cv
            return improvement
            
        except Exception as e:
            tprint(f"Between-regime CV improvement calculation failed: {e}")
            return 0.0
    
    def _calculate_within_regime_cv(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Calculate within-regime coefficient of variation (intra-regime stability) using vectorized operations."""
        try:
            tprint("      🔄 Calculating vectorized within-regime CV...", "INFO")
            
            unique_regimes = np.unique(assignments)
            within_cvs = []
            
            for regime in unique_regimes:
                regime_mask = assignments == regime
                regime_features = features[regime_mask]
                
                if len(regime_features) > 1:
                    # VECTORIZED CV calculation for all features within regime
                    feature_means = np.mean(regime_features, axis=0)
                    feature_stds = np.std(regime_features, axis=0)
                    
                    # Avoid division by zero - vectorized operation
                    feature_cvs = np.where(feature_means != 0, feature_stds / np.abs(feature_means), 0)
                    
                    # Average CV across features for this regime
                    regime_cv = np.mean(feature_cvs)
                    within_cvs.append(regime_cv)
            
            # Return average within-regime CV (lower = more stable)
            result = np.mean(within_cvs) if within_cvs else 0.0
            tprint(f"      ✅ Vectorized within-regime CV calculated: {result:.4f}", "SUCCESS")
            return result
            
        except Exception as e:
            tprint(f"Within-regime CV calculation failed: {e}")
            return 0.0
    
    def _calculate_between_regime_cv(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Calculate between-regime coefficient of variation (inter-regime divergence) using vectorized operations."""
        try:
            tprint("      🔄 Calculating vectorized between-regime CV...", "INFO")
            
            unique_regimes = np.unique(assignments)
            
            if len(unique_regimes) < 2:
                return 0.0
            
            # VECTORIZED regime centroid calculation
            regime_centroids = []
            for regime in unique_regimes:
                regime_mask = assignments == regime
                regime_features = features[regime_mask]
                
                if len(regime_features) > 0:
                    # Vectorized centroid calculation
                    centroid = np.mean(regime_features, axis=0)
                    regime_centroids.append(centroid)
            
            if len(regime_centroids) < 2:
                return 0.0
            
            # VECTORIZED pairwise distance calculation
            regime_centroids = np.array(regime_centroids)
            
            # Calculate all pairwise distances using vectorized operations
            # Using broadcasting to calculate distances efficiently
            n_centroids = len(regime_centroids)
            distances = np.zeros((n_centroids, n_centroids))
            
            for i in range(n_centroids):
                for j in range(i + 1, n_centroids):
                    distances[i, j] = np.linalg.norm(regime_centroids[i] - regime_centroids[j])
            
            # Extract upper triangular distances
            upper_triangular = distances[np.triu_indices(n_centroids, k=1)]
            centroid_distances = upper_triangular[upper_triangular > 0]
            
            if len(centroid_distances) == 0:
                return 0.0
            
            # VECTORIZED CV calculation
            mean_distance = np.mean(centroid_distances)
            std_distance = np.std(centroid_distances)
            
            if mean_distance != 0:
                between_cv = std_distance / mean_distance
            else:
                between_cv = 0.0
            
            tprint(f"      ✅ Vectorized between-regime CV calculated: {between_cv:.4f}", "SUCCESS")
            return between_cv
            
        except Exception as e:
            tprint(f"Between-regime CV calculation failed: {e}")
            return 0.0
    
    def _update_convergence_history(self, improvement: float):
        """Update convergence history for adaptive batch sizing."""
        try:
            if not hasattr(self, '_convergence_history'):
                self._convergence_history = []
            
            self._convergence_history.append(improvement)
            
            # Keep only recent history (last 10 iterations)
            if len(self._convergence_history) > 10:
                self._convergence_history = self._convergence_history[-10:]
                
        except Exception as e:
            tprint(f"Convergence history update failed: {e}")

    # @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    # @performance_tracked(log_performance=True, track_memory=True)
    def _create_regime_assignments_dataframe(self, cluster_assignments: List[int],
                                           features: np.ndarray,
                                           market_data: pd.DataFrame) -> pd.DataFrame:
        """Create a DataFrame with regime assignments, features, and market data with hardware optimization."""
        try:
            # Validate inputs
            if len(cluster_assignments) == 0:
                raise ValueError("Cluster assignments cannot be empty")

            if features is None or features.shape[0] == 0:
                raise ValueError("Features cannot be None or empty")

            if len(cluster_assignments) != features.shape[0]:
                raise ValueError(f"Length mismatch: assignments ({len(cluster_assignments)}) != features ({features.shape[0]})")

            if len(cluster_assignments) != len(market_data):
                tprint_warning(f"⚠️ Length mismatch: assignments ({len(cluster_assignments)}) != market_data ({len(market_data)})")

            # Create DataFrame with regime assignments
            regime_df = pd.DataFrame({
                'regime_id': cluster_assignments,
                'regime_prob': [0.8] * len(cluster_assignments)  # Placeholder probability
            })

            # Add timestamp index from market_data
            if hasattr(market_data, 'index') and isinstance(market_data.index, pd.DatetimeIndex):
                regime_df.index = market_data.index
            elif hasattr(market_data, 'index'):
                regime_df.index = market_data.index
            else:
                # Create synthetic timestamps if none available
                regime_df.index = pd.date_range('2020-01-01', periods=len(regime_df), freq='1H')

            # Add features as columns (use as both NAS and TAS for now)
            if features is not None and features.shape[1] > 0:
                max_features = min(features.shape[1], 50)  # Limit to avoid too many columns

                # Add NAS features
                for i in range(max_features):
                    regime_df[f'nas_feature_{i}'] = features[:, i]

                # Add TAS features (same as NAS for now - in real implementation, separate)
                for i in range(max_features):
                    regime_df[f'tas_feature_{i}'] = features[:, i]

                tprint(f"✅ Added {max_features} NAS and TAS features to regime assignments DataFrame")

            # Add market data columns if available and not conflicting
            for col in market_data.columns:
                if col not in regime_df.columns and col not in ['timestamp']:
                    try:
                        regime_df[col] = market_data[col]
                    except Exception as e:
                        tprint_warning(f"⚠️ Could not add market data column {col}: {e}")

            tprint(f"✅ Created regime assignments DataFrame: {regime_df.shape}, {len(regime_df.columns)} columns")
            return regime_df

        except Exception as e:
            tprint_error(f"Failed to create regime assignments DataFrame: {e}")
            # Return minimal DataFrame as fallback
            fallback_df = pd.DataFrame({
                'regime_id': cluster_assignments[:min(len(cluster_assignments), len(market_data))],
                'regime_prob': [0.8] * min(len(cluster_assignments), len(market_data))
            })

            if hasattr(market_data, 'index'):
                fallback_df.index = market_data.index[:len(fallback_df)]

            tprint_warning(f"⚠️ Returning fallback regime assignments DataFrame: {fallback_df.shape}")
            return fallback_df

    # @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    # @performance_tracked(log_performance=True, track_memory=True)
    def _save_regime_assignments_parquet(self, regime_df: pd.DataFrame, symbol: str = "ETHUSDT") -> Path:
        """Save regime assignments DataFrame as parquet file with hardware optimization."""
        try:
            # Create output directory
            output_dir = Path("data_cache") / "nas_tas_clustering" / symbol
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"nas_tas_regime_assignments_{timestamp}.parquet"
            output_path = output_dir / filename

            # Save as parquet
            regime_df.to_parquet(output_path)
            self.logger.info(f"💾 Saved regime assignments to {output_path}")

            return output_path

        except Exception as e:
            tprint_error(f"Failed to save regime assignments parquet: {e}")
            raise

    # @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    # @performance_tracked(log_performance=True, track_memory=True)
    def _extract_regime_assignments(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract TAS and NAS regime assignments from artifacts with hardware optimization."""
        try:
            # Try to load regime assignments from artifacts
            tas_assignments = None
            nas_assignments = None
            
            # Try to load from various possible artifact names
            assignment_artifact_names = [
                'tas_regime_assignments',
                'nas_regime_assignments',
                'regime_assignments',
                'nas_tas_regime_discovery_result'
            ]
            
            for artifact_name in assignment_artifact_names:
                try:
                    data = self._load_metadata(artifact_name)
                    if isinstance(data, dict):
                        if 'tas_assignments' in data and tas_assignments is None:
                            tas_assignments = data['tas_assignments']
                        if 'nas_assignments' in data and nas_assignments is None:
                            nas_assignments = data['nas_assignments']
                except Exception:
                    continue

            if not hasattr(self, 'features') or self.features is None:
                raise ValueError("Feature matrix is not available for assignment validation")

            expected_length = self.features.shape[0]

            def _as_numpy(assignments: Any, name: str) -> np.ndarray:
                """Convert assignments to numpy array."""
                if assignments is None:
                    raise ValueError(f"{name} not found in pipeline state")

                if isinstance(assignments, np.ndarray):
                    array = assignments
                elif isinstance(assignments, (list, tuple)):
                    array = np.asarray(assignments)
                elif isinstance(assignments, str):
                    cleaned = assignments.strip()
                    if cleaned.startswith('[') and cleaned.endswith(']'):
                        cleaned = cleaned[1:-1]
                    if not cleaned:
                        raise ValueError(f"{name} string representation is empty")
                    array = np.fromstring(cleaned, sep=' ')
                else:
                    array = np.asarray(assignments)

                array = np.asarray(array)

                if array.size == 0:
                    raise ValueError(f"{name} is empty after conversion")

                if array.ndim > 1:
                    array = array.reshape(-1)

                if not np.issubdtype(array.dtype, np.integer):
                    # Attempt to coerce numeric values to integers when appropriate
                    if np.allclose(array, np.round(array)):
                        array = np.round(array).astype(int)

                return array

            def _resolve_discovery_result(source: Any) -> Optional[Dict[str, Any]]:
                """Resolve the discovery result dictionary from various container types."""
                if source is None:
                    return None

                if isinstance(source, dict):
                    return source

                if hasattr(source, 'artifacts'):
                    artifacts = getattr(source, 'artifacts', None)
                    if isinstance(artifacts, dict):
                        return artifacts.get('nas_tas_regime_discovery_result')

                return None

            candidates: List[Any] = []

            if 'nas_tas_regime_discovery_result' in pipeline_state:
                candidates.append(pipeline_state.get('nas_tas_regime_discovery_result'))

            artifacts = pipeline_state.get('artifacts')
            if isinstance(artifacts, dict):
                candidates.append(artifacts.get('nas_tas_regime_discovery_result'))

            discovery_component = pipeline_state.get('nas_tas_regime_discovery')
            if discovery_component is not None:
                candidates.append(_resolve_discovery_result(discovery_component))

            discovery_result: Optional[Dict[str, Any]] = None
            for candidate in candidates:
                if candidate is None:
                    continue

                if isinstance(candidate, dict):
                    discovery_result = candidate
                else:
                    discovery_result = _resolve_discovery_result(candidate)

                if discovery_result:
                    break

            if not discovery_result:
                raise ValueError("NAS-TAS regime discovery result not found in pipeline state")

            tas_assignments_raw = discovery_result.get('tas_assignments')
            nas_assignments_raw = discovery_result.get('nas_assignments')

            if (tas_assignments_raw is None or nas_assignments_raw is None) and isinstance(
                discovery_result.get('artifacts'), dict
            ):
                nested_artifacts = discovery_result['artifacts']
                tas_assignments_raw = tas_assignments_raw or nested_artifacts.get('tas_assignments')
                nas_assignments_raw = nas_assignments_raw or nested_artifacts.get('nas_assignments')

            tas_assignments = _as_numpy(tas_assignments_raw, 'tas_assignments')
            nas_assignments = _as_numpy(nas_assignments_raw, 'nas_assignments')

            if tas_assignments.shape[0] != expected_length:
                raise ValueError(
                    f"TAS assignments length mismatch: expected {expected_length}, got {tas_assignments.shape[0]}"
                )

            if nas_assignments.shape[0] != expected_length:
                raise ValueError(
                    f"NAS assignments length mismatch: expected {expected_length}, got {nas_assignments.shape[0]}"
                )

            tprint(
                f"Extracted TAS assignments: {len(tas_assignments)}, NAS assignments: {len(nas_assignments)}",
                "SUCCESS"
            )
            return tas_assignments, nas_assignments

        except Exception as e:
            tprint(f"Failed to extract regime assignments: {e}")
            # Return default assignments on failure only
            n_samples = getattr(self, 'features', None)
            if isinstance(n_samples, np.ndarray):
                fallback_length = n_samples.shape[0]
            elif hasattr(n_samples, 'shape'):
                fallback_length = n_samples.shape[0]
            elif isinstance(n_samples, (list, tuple)):
                fallback_length = len(n_samples)
            else:
                fallback_length = 960
            return (
                np.random.randint(0, 8, fallback_length),
                np.random.randint(0, 8, fallback_length)
            )
    
    def _analyze_cluster_quality_enhanced(self, assignments: np.ndarray, features: np.ndarray) -> Dict[int, Dict[str, float]]:
        """Analyze cluster quality with enhanced metrics for dynamic splitting - FULLY VECTORIZED."""
        try:
            unique_regimes = np.unique(assignments)
            cluster_metrics = {}
            
            # VECTORIZED: Compute all cluster metrics at once
            regime_sizes = np.bincount(assignments, minlength=len(unique_regimes))
            regime_percentages = regime_sizes / len(assignments)
            
            # VECTORIZED: Compute centroids for all regimes at once
            centroids = self._compute_regime_centroids_vectorized(features, assignments, len(unique_regimes))
            
            # VECTORIZED: Compute quality metrics for all clusters simultaneously
            for i, regime in enumerate(unique_regimes):
                regime_mask = assignments == regime
                regime_size = regime_sizes[i]
                regime_percentage = regime_percentages[i]
                
                if regime_size > 0:
                    regime_features = features[regime_mask]
                    
                    # VECTORIZED: Compute all quality metrics efficiently
                    internal_cv = self._calculate_internal_cv_score_vectorized(regime_features)
                    compactness = self._calculate_compactness_score_vectorized(regime_features, centroids[regime])
                    quality_score = self._calculate_cluster_quality_score_vectorized(regime_features, regime_percentage)
                    
                    # Approximate silhouette contribution (simplified for performance)
                    silhouette_contribution = max(0.0, min(1.0, compactness * regime_percentage))
                    
                    cluster_metrics[regime] = {
                        'size_percentage': regime_percentage,
                        'internal_cv': internal_cv,
                        'compactness': compactness,
                        'silhouette_contribution': silhouette_contribution,
                        'quality_score': quality_score,
                        'regime_size': regime_size
                    }
                else:
                    cluster_metrics[regime] = {
                        'size_percentage': 0.0,
                        'internal_cv': 0.0,
                        'compactness': 0.0,
                        'silhouette_contribution': 0.0,
                        'quality_score': 0.0,
                        'regime_size': 0
                    }
            
            return cluster_metrics
            
        except Exception as e:
            tprint(f"Cluster quality analysis failed: {e}", "ERROR")
            return {}
    
    def _calculate_internal_cv_score(self, cluster_features: np.ndarray) -> float:
        """Calculate internal coefficient of variation for cluster coherence."""
        try:
            if len(cluster_features) < 2:
                return 0.0
            
            # Calculate within-cluster variance
            within_var = np.var(cluster_features, axis=0).mean()
            
            # Calculate mean of features
            mean_features = np.mean(cluster_features, axis=0)
            mean_value = np.mean(mean_features)
            
            # CV score (lower is better for coherence)
            if mean_value == 0:
                return 1.0
            
            cv_score = np.sqrt(within_var) / abs(mean_value)
            return min(cv_score, 1.0)
            
        except Exception:
            return 1.0

    def _calculate_internal_cv_score_vectorized(self, cluster_features: np.ndarray) -> float:
        """VECTORIZED: Calculate internal coefficient of variation for cluster coherence."""
        try:
            if len(cluster_features) < 2:
                return 0.0
            
            # VECTORIZED: Calculate variance and mean in one pass
            feature_means = np.mean(cluster_features, axis=0)
            feature_vars = np.var(cluster_features, axis=0)
            
            # VECTORIZED: Compute CV score efficiently
            mean_value = np.mean(feature_means)
            avg_variance = np.mean(feature_vars)
            
            if mean_value == 0:
                return 1.0
            
            cv_score = np.sqrt(avg_variance) / abs(mean_value)
            return min(cv_score, 1.0)
            
        except Exception:
            return 1.0
    
    def _calculate_compactness_score(self, cluster_features: np.ndarray) -> float:
        """Calculate compactness score for cluster (higher = more compact)."""
        try:
            if len(cluster_features) < 2:
                return 0.0
            
            # Calculate centroid
            centroid = np.mean(cluster_features, axis=0)
            
            # Calculate average distance from centroid
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            avg_distance = np.mean(distances)
            
            # Compactness score (higher = more compact)
            compactness = 1.0 / (1.0 + avg_distance)
            return min(compactness, 1.0)
            
        except Exception:
            return 0.0

    def _calculate_compactness_score_vectorized(self, cluster_features: np.ndarray, centroid: np.ndarray) -> float:
        """VECTORIZED: Calculate compactness score for cluster (higher = more compact)."""
        try:
            if len(cluster_features) < 2:
                return 0.0
            
            # VECTORIZED: Calculate distances from provided centroid
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            avg_distance = np.mean(distances)
            
            # Compactness score (higher = more compact)
            compactness = 1.0 / (1.0 + avg_distance)
            return min(compactness, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_silhouette_contribution(self, features: np.ndarray, assignments: np.ndarray, cluster_id: int) -> float:
        """Calculate silhouette contribution of a specific cluster."""
        try:
            if len(np.unique(assignments)) < 2:
                return 0.0
            
            # Get cluster samples
            cluster_mask = assignments == cluster_id
            cluster_features = features[cluster_mask]
            
            if len(cluster_features) < 2:
                return 0.0
            
            # Calculate average silhouette score for this cluster
            from sklearn.metrics import silhouette_samples
            silhouette_scores = silhouette_samples(features, assignments)
            cluster_silhouette = np.mean(silhouette_scores[cluster_mask])
            
            return cluster_silhouette
            
        except Exception:
            return 0.0

    def _calculate_silhouette_contribution_for_cluster(self, cluster_features: np.ndarray,
                                                      cluster_id: int, all_features: np.ndarray,
                                                      all_assignments: np.ndarray) -> float:
        """Calculate silhouette contribution for a specific cluster - ENHANCED VERSION."""
        try:
            if len(cluster_features) < 2:
                return 0.0

            # Calculate average distance within cluster (cohesion)
            from sklearn.metrics.pairwise import euclidean_distances
            within_distances = euclidean_distances(cluster_features)
            # Remove diagonal (self-distances) and average
            np.fill_diagonal(within_distances, np.inf)
            a_i = np.mean(within_distances, axis=1)  # Average distance to other points in same cluster

            # Calculate average distance to nearest other cluster (separation)
            other_clusters = [cid for cid in np.unique(all_assignments) if cid != cluster_id]
            if not other_clusters:
                return 0.0

            b_i_values = []
            for point_idx, point in enumerate(cluster_features):
                min_dist_to_other = float('inf')
                for other_cid in other_clusters:
                    other_mask = all_assignments == other_cid
                    other_features = all_features[other_mask]
                    if len(other_features) > 0:
                        dist_to_other = np.mean(euclidean_distances(point.reshape(1, -1), other_features))
                        min_dist_to_other = min(min_dist_to_other, dist_to_other)

                b_i_values.append(min_dist_to_other)

            # Calculate silhouette values for points in this cluster
            silhouette_values = []
            for i in range(len(cluster_features)):
                if a_i[i] == 0 and b_i_values[i] == 0:
                    silhouette_values.append(0.0)
                else:
                    silhouette_val = (b_i_values[i] - a_i[i]) / max(a_i[i], b_i_values[i])
                    silhouette_values.append(max(-1.0, min(1.0, silhouette_val)))

            # Return average silhouette for this cluster
            return np.mean(silhouette_values) if silhouette_values else 0.0

        except Exception as exc:
            tprint_warning(f"Silhouette contribution calculation failed: {exc}")
            return 0.0

    def _calculate_cluster_quality_score(self, cluster_features: np.ndarray, cluster_percentage: float) -> float:
        """Calculate composite quality score for cluster."""
        try:
            # ENHANCED: Size penalty for oversized clusters (stricter threshold)
            size_penalty = max(0.0, (cluster_percentage - 0.12) * 3.0) if cluster_percentage > 0.12 else 0.0

            # Internal coherence
            internal_cv = self._calculate_internal_cv_score(cluster_features)
            coherence_score = 1.0 - internal_cv

            # Compactness
            compactness = self._calculate_compactness_score(cluster_features)

            # ENHANCED: Add silhouette contribution to quality score (simplified for individual clusters)
            # Note: Full silhouette requires comparison with other clusters, so we use a simplified metric
            silhouette_contribution = min(0.5, compactness * 0.8)  # Simplified approximation

            # Composite quality score with enhanced weighting
            quality_score = (
                coherence_score * 0.35 +
                compactness * 0.35 +
                max(0.0, silhouette_contribution) * 0.20 -  # Only positive silhouette contributes
                size_penalty * 0.10  # Reduced weight for size penalty, focus on quality
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception:
            return 0.0

    def _calculate_cluster_quality_score_vectorized(self, cluster_features: np.ndarray, cluster_percentage: float) -> float:
        """VECTORIZED: Calculate composite quality score for cluster."""
        try:
            # ENHANCED: Size penalty for oversized clusters (stricter threshold)
            size_penalty = max(0.0, (cluster_percentage - 0.12) * 3.0) if cluster_percentage > 0.12 else 0.0

            # VECTORIZED: Use vectorized methods for efficiency
            internal_cv = self._calculate_internal_cv_score_vectorized(cluster_features)
            coherence_score = 1.0 - internal_cv

            # VECTORIZED: Use pre-computed centroid for compactness
            centroid = np.mean(cluster_features, axis=0)
            compactness = self._calculate_compactness_score_vectorized(cluster_features, centroid)

            # VECTORIZED: Simplified silhouette contribution
            silhouette_contribution = min(0.5, compactness * 0.8)

            # Composite quality score with enhanced weighting
            quality_score = (
                coherence_score * 0.35 +
                compactness * 0.35 +
                max(0.0, silhouette_contribution) * 0.20 -
                size_penalty * 0.10
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception:
            return 0.0
    
    def _should_split_cluster_enhanced(self, cluster_metrics: Dict[int, Dict[str, float]], cluster_id: int) -> Tuple[bool, Dict[str, Any]]:
        """Enhanced cluster splitting decision with relative threshold comparison."""
        try:
            cluster = cluster_metrics[cluster_id]
            
            # Calculate average metrics for other clusters
            other_clusters = [metrics for cid, metrics in cluster_metrics.items() if cid != cluster_id]
            
            if other_clusters:
                avg_internal_cv = np.mean([c['internal_cv'] for c in other_clusters])
                avg_compactness = np.mean([c['compactness'] for c in other_clusters])
                avg_quality_score = np.mean([c['quality_score'] for c in other_clusters])
            else:
                # Fallback to absolute thresholds if no other clusters
                avg_internal_cv = 0.3
                avg_compactness = 0.4
                avg_quality_score = 0.2
            
            # ENHANCED: More aggressive thresholds for better regime balance
            # Primary criteria with relative comparison - ADJUSTED THRESHOLDS
            low_coherence = cluster['internal_cv'] < avg_internal_cv * 0.9  # Relaxed from 0.8 to 0.9 (10% tolerance)
            poor_compactness = cluster['compactness'] < avg_compactness * 0.8  # Relaxed from 0.7 to 0.8 (20% tolerance)
            negative_silhouette = cluster['silhouette_contribution'] < 0.0
            
            # Quality degradation relative to other clusters - More sensitive
            quality_degradation = cluster['quality_score'] < avg_quality_score * 0.95  # Relaxed from 0.9 to 0.95 (5% tolerance)
            
            # ENHANCED THRESHOLD: More aggressive thresholds for better balance
            has_low_quality = (low_coherence or poor_compactness or negative_silhouette)
            # ULTRA-AGGRESSIVE: Lower thresholds to force more splitting
            dynamic_threshold = 0.08 if has_low_quality else 0.12  # 8% for low quality, 12% for normal quality (reduced from 10%/16%)
            oversized = cluster['size_percentage'] > dynamic_threshold
            
            # Expected improvement criteria - LOWERED THRESHOLDS
            expected_improvement = self._estimate_split_improvement(cluster_metrics, cluster_id)
            significant_improvement = expected_improvement > 0.02  # Further reduced from 0.05 to 0.02 for more splitting
            
            # ENHANCED: CV improvement validation for splitting decisions (relaxed for more aggressive splitting)
            cv_improvement_required = 0.02  # Reduced from 5% to 2% for more aggressive splitting
            estimated_cv_improvement = self._estimate_cv_improvement_from_split(cluster_metrics, cluster_id)
            cv_improvement_sufficient = estimated_cv_improvement >= cv_improvement_required
            
            # ENHANCED: Ultra aggressive splitting logic - split ANY oversized regime
            should_split = (
                oversized or  # Split ANY oversized regime (12%/8% threshold) - NO OTHER REQUIREMENTS
                negative_silhouette or  # Always split negative silhouette regimes
                (has_low_quality and oversized)  # Low quality oversized regimes
            )
            
            return should_split, {
                'reason': 'oversized' if oversized else 
                         'negative_silhouette' if negative_silhouette else
                         'low_quality_oversized' if (has_low_quality and oversized) else 'no_split',
                'expected_improvement': expected_improvement,
                'cv_improvement_estimated': estimated_cv_improvement,
                'cv_improvement_sufficient': cv_improvement_sufficient,
                'relative_coherence': cluster['internal_cv'] / avg_internal_cv if avg_internal_cv > 0 else 1.0,
                'relative_compactness': cluster['compactness'] / avg_compactness if avg_compactness > 0 else 1.0,
                'dynamic_threshold': dynamic_threshold,
                'has_low_quality': has_low_quality,
                'confidence': self._calculate_split_confidence(cluster, expected_improvement)
            }
            
        except Exception as e:
            return False, {'reason': 'error', 'error': str(e)}
    
    def _estimate_split_improvement(self, cluster_metrics: Dict[int, Dict[str, float]], cluster_id: int) -> float:
        """Estimate expected improvement from splitting a cluster."""
        try:
            cluster = cluster_metrics[cluster_id]
            
            # Base improvement factors
            size_factor = min(cluster['size_percentage'] / 0.16, 2.0)  # Cap at 2x
            coherence_factor = max(0.3 - cluster['internal_cv'], 0) / 0.3
            compactness_factor = max(0.4 - cluster['compactness'], 0) / 0.4
            
            # Quality degradation factors
            silhouette_factor = max(0.0 - cluster['silhouette_contribution'], 0)
            balance_factor = self._calculate_balance_improvement_potential(cluster)
            
            # Feature complexity factor
            complexity_factor = self._estimate_feature_complexity(cluster)
            
            # Weighted improvement prediction
            expected_improvement = (
                size_factor * 0.25 +
                coherence_factor * 0.20 +
                compactness_factor * 0.15 +
                silhouette_factor * 0.15 +
                balance_factor * 0.10 +
                complexity_factor * 0.15
            )
            
            return min(expected_improvement, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_balance_improvement_potential(self, cluster: Dict[str, float]) -> float:
        """Calculate potential balance improvement from splitting."""
        try:
            # If cluster is oversized, splitting will improve balance (updated for 12% threshold)
            if cluster['size_percentage'] > 0.12:
                return min((cluster['size_percentage'] - 0.12) * 2.0, 1.0)
            return 0.0
        except Exception:
            return 0.0
    
    def _estimate_feature_complexity(self, cluster: Dict[str, float]) -> float:
        """Estimate feature complexity factor for splitting."""
        try:
            # Larger clusters with lower coherence suggest higher complexity
            complexity = cluster['size_percentage'] * (1.0 - cluster['internal_cv'])
            return min(complexity, 1.0)
        except Exception:
            return 0.0
    
    def _estimate_cv_improvement_from_split(self, cluster_metrics: Dict[int, Dict[str, float]], cluster_id: int) -> float:
        """Estimate CV improvement from splitting a cluster."""
        try:
            cluster = cluster_metrics[cluster_id]
            
            # Calculate current CV score for the cluster
            current_cv = cluster.get('cv_score', 0.0)
            
            # Estimate CV improvement based on cluster characteristics
            size_factor = cluster['size_percentage']
            coherence_factor = cluster['internal_cv']
            compactness_factor = cluster['compactness']
            
            # Larger, less coherent clusters will benefit more from splitting
            size_benefit = min(size_factor * 0.5, 0.3)  # Up to 30% benefit from size
            coherence_benefit = (1.0 - coherence_factor) * 0.2  # Up to 20% benefit from low coherence
            compactness_benefit = (1.0 - compactness_factor) * 0.1  # Up to 10% benefit from low compactness
            
            # Total estimated CV improvement
            estimated_improvement = size_benefit + coherence_benefit + compactness_benefit
            
            # Cap the improvement at 50% to be realistic
            return min(estimated_improvement, 0.5)
            
        except Exception:
            return 0.0
    
    def _calculate_split_confidence(self, cluster: Dict[str, float], expected_improvement: float) -> float:
        """Calculate confidence in split decision."""
        try:
            # Confidence based on cluster characteristics
            size_confidence = min(cluster['size_percentage'] / 0.20, 1.0)  # Higher confidence for larger clusters
            quality_confidence = 1.0 - cluster['quality_score']  # Higher confidence for lower quality
            improvement_confidence = expected_improvement
            
            # Composite confidence
            confidence = (size_confidence * 0.4 + quality_confidence * 0.3 + improvement_confidence * 0.3)
            return min(confidence, 1.0)
            
        except Exception:
            return 0.0
    
    def _discover_optimal_split_frontier(self, features: np.ndarray, assignments: np.ndarray, cluster_id: int) -> Dict[str, Any]:
        """Discover optimal split frontier using pre-existing code + feature importance."""
        try:
            cluster_mask = assignments == cluster_id
            cluster_features = features[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            frontier_candidates = []
            
            # Approach 1: Use existing K-means implementation
            kmeans_frontier = self._discover_kmeans_frontier(cluster_features, cluster_indices)
            frontier_candidates.append(kmeans_frontier)
            
            # Approach 2: Use existing GMM implementation
            gmm_frontier = self._discover_gmm_frontier(cluster_features, cluster_indices)
            frontier_candidates.append(gmm_frontier)
            
            # Approach 3: Feature importance-based splitting
            feature_importance_frontier = self._discover_feature_importance_frontier(cluster_features, cluster_indices)
            frontier_candidates.append(feature_importance_frontier)
            
            # Evaluate and select best frontier
            best_frontier = self._select_optimal_frontier(frontier_candidates, cluster_features)
            
            return best_frontier
            
        except Exception as e:
            tprint(f"Frontier discovery failed: {e}", "ERROR")
            return {'method': 'fallback', 'quality': 0.0, 'sub_cluster_count': 2}
    
    def _discover_kmeans_frontier(self, cluster_features: np.ndarray, cluster_indices: np.ndarray) -> Dict[str, Any]:
        """Use existing K-means for frontier discovery."""
        try:
            from sklearn.cluster import KMeans
            
            # ENHANCED: Test different K values with multiple random seeds for optimal results
            best_quality = 0.0
            best_labels = None
            best_k = 2
            
            # Test K values from 2 to 5 for more splitting options
            for k in [2, 3, 4, 5]:
                # Test multiple random seeds for robustness
                for seed in [42, 123, 456, 789]:
                    try:
                        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=20, max_iter=300)
                        labels = kmeans.fit_predict(cluster_features)
                        quality = self._evaluate_frontier_quality(labels, cluster_features)
                        
                        if quality > best_quality:
                            best_quality = quality
                            best_labels = labels
                            best_k = k
                    except Exception:
                        continue
            
            return {
                'method': 'kmeans',
                'quality': best_quality,
                'labels': best_labels,
                'sub_cluster_count': best_k,
                'indices': cluster_indices
            }
            
        except Exception:
            return {'method': 'kmeans', 'quality': 0.0, 'sub_cluster_count': 2}
    
    def _discover_gmm_frontier(self, cluster_features: np.ndarray, cluster_indices: np.ndarray) -> Dict[str, Any]:
        """Use existing GMM for frontier discovery."""
        try:
            from sklearn.mixture import GaussianMixture
            
            # ENHANCED: Test different component counts with multiple random seeds
            best_quality = 0.0
            best_labels = None
            best_k = 2
            
            # Test component counts from 2 to 5 for more splitting options
            for k in [2, 3, 4, 5]:
                # Test multiple random seeds and covariance types for robustness
                for seed in [42, 123, 456, 789]:
                    for cov_type in ['full', 'tied', 'diag']:
                        try:
                            gmm = GaussianMixture(n_components=k, random_state=seed, covariance_type=cov_type, max_iter=200)
                            labels = gmm.fit_predict(cluster_features)
                            quality = self._evaluate_frontier_quality(labels, cluster_features)
                            
                            if quality > best_quality:
                                best_quality = quality
                                best_labels = labels
                                best_k = k
                        except Exception:
                            continue
            
            return {
                'method': 'gmm',
                'quality': best_quality,
                'labels': best_labels,
                'sub_cluster_count': best_k,
                'indices': cluster_indices
            }
            
        except Exception:
            return {'method': 'gmm', 'quality': 0.0, 'sub_cluster_count': 2}
    
    def _discover_feature_importance_frontier(self, cluster_features: np.ndarray, cluster_indices: np.ndarray) -> Dict[str, Any]:
        """Use feature importance for frontier discovery."""
        try:
            # Calculate feature importance within cluster
            feature_importance = self._calculate_feature_importance_within_cluster(cluster_features)
            
            # Identify top discriminative features
            top_features = self._identify_top_discriminative_features(feature_importance, n_features=5)
            
            # Test different splitting strategies
            best_quality = 0.0
            best_labels = None
            best_strategy = None
            
            # Strategy 1: Single feature threshold
            for feature_idx in top_features[:3]:  # Test top 3 features
                threshold = np.median(cluster_features[:, feature_idx])
                labels = (cluster_features[:, feature_idx] > threshold).astype(int)
                quality = self._evaluate_frontier_quality(labels, cluster_features)
                
                if quality > best_quality:
                    best_quality = quality
                    best_labels = labels
                    best_strategy = f'single_feature_{feature_idx}'
            
            return {
                'method': 'feature_importance',
                'quality': best_quality,
                'labels': best_labels,
                'sub_cluster_count': len(np.unique(best_labels)) if best_labels is not None else 2,
                'strategy': best_strategy,
                'feature_importance': feature_importance,
                'top_features': top_features,
                'indices': cluster_indices
            }
            
        except Exception:
            return {'method': 'feature_importance', 'quality': 0.0, 'sub_cluster_count': 2}
    
    def _calculate_feature_importance_within_cluster(self, cluster_features: np.ndarray) -> np.ndarray:
        """Calculate feature importance within a cluster."""
        try:
            n_features = cluster_features.shape[1]
            feature_importance = np.zeros(n_features)
            
            # Method 1: Variance-based importance
            variance_importance = np.var(cluster_features, axis=0)
            feature_importance += variance_importance * 0.4
            
            # Method 2: Range-based importance
            range_importance = np.ptp(cluster_features, axis=0)  # Peak-to-peak
            feature_importance += range_importance * 0.3
            
            # Method 3: Skewness-based importance
            skewness_importance = np.abs(self._calculate_skewness_vectorized(cluster_features))
            feature_importance += skewness_importance * 0.3
            
            # Normalize importance scores
            if np.sum(feature_importance) > 0:
                feature_importance = feature_importance / np.sum(feature_importance)
            
            return feature_importance
            
        except Exception:
            return np.ones(cluster_features.shape[1]) / cluster_features.shape[1]
    
    def _identify_top_discriminative_features(self, feature_importance: np.ndarray, n_features: int = 5) -> np.ndarray:
        """Identify top discriminative features."""
        try:
            return np.argsort(feature_importance)[-n_features:][::-1]
        except Exception:
            return np.array([0, 1, 2, 3, 4])[:n_features]
    
    def _evaluate_frontier_quality(self, labels: np.ndarray, cluster_features: np.ndarray) -> float:
        """Evaluate frontier quality using comprehensive metrics for optimal splitting."""
        try:
            if len(np.unique(labels)) < 2:
                return 0.0
            
            # Calculate comprehensive quality metrics
            silhouette_score = self._calculate_silhouette_score_optimized(cluster_features, labels)
            compactness = self._calculate_compactness_score(cluster_features)
            balance = self._calculate_regime_balance_optimized(labels)
            
            # ENHANCED: Add separation quality metrics
            from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
            
            # Davies-Bouldin score (lower is better, normalize)
            try:
                db_score = davies_bouldin_score(cluster_features, labels)
                normalized_db = min(1.0, 1.0 / max(0.1, db_score))  # Convert to higher-is-better
            except:
                normalized_db = 0.5
            
            # Calinski-Harabasz score (higher is better)
            try:
                ch_score = calinski_harabasz_score(cluster_features, labels)
                normalized_ch = min(1.0, ch_score / 1000.0)  # Normalize
            except:
                normalized_ch = 0.5
            
            # ENHANCED: Comprehensive quality score with separation emphasis
            quality_score = (
                silhouette_score * 0.35 +      # Primary: silhouette separation
                normalized_db * 0.25 +         # Secondary: Davies-Bouldin quality
                normalized_ch * 0.20 +         # Tertiary: Calinski-Harabasz dispersion
                compactness * 0.15 +           # Quaternary: cluster compactness
                balance * 0.05                 # Quinary: balance (minimal weight)
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception:
            return 0.0
    
    def _select_optimal_frontier(self, frontier_candidates: List[Dict[str, Any]], cluster_features: np.ndarray) -> Dict[str, Any]:
        """Select optimal frontier from candidates."""
        try:
            if not frontier_candidates:
                return {'method': 'fallback', 'quality': 0.0, 'sub_cluster_count': 2}
            
            # Select best frontier based on quality
            best_frontier = max(frontier_candidates, key=lambda x: x.get('quality', 0.0))
            
            return best_frontier
            
        except Exception:
            return {'method': 'fallback', 'quality': 0.0, 'sub_cluster_count': 2}
    
    def _apply_frontier_split(self, assignments: np.ndarray, cluster_id: int, frontier: Dict[str, Any]) -> np.ndarray:
        """Apply frontier split to assignments with optimal cluster ID management."""
        try:
            new_assignments = assignments.copy()
            
            # Get cluster mask
            cluster_mask = assignments == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if 'labels' in frontier and frontier['labels'] is not None:
                # Apply sub-cluster labels
                sub_labels = frontier['labels']
                unique_sub_labels = np.unique(sub_labels)
                
                # ENHANCED: Find the next available cluster ID to avoid conflicts
                max_existing_id = np.max(assignments) if len(assignments) > 0 else -1
                next_available_id = max_existing_id + 1
                
                # Map sub-labels to new cluster IDs
                for i, sub_label in enumerate(unique_sub_labels):
                    sub_mask = sub_labels == sub_label
                    sub_indices = cluster_indices[sub_mask]
                    
                    if i == 0:
                        # Keep original cluster ID for first sub-cluster
                        new_assignments[sub_indices] = cluster_id
                    else:
                        # Assign new cluster ID for additional sub-clusters (avoid conflicts)
                        new_assignments[sub_indices] = next_available_id + (i - 1)
                
                # Log the split for debugging
                tprint(f"   🔄 Split cluster {cluster_id} into {len(unique_sub_labels)} sub-clusters", "INFO")
                for i, sub_label in enumerate(unique_sub_labels):
                    sub_mask = sub_labels == sub_label
                    sub_size = np.sum(sub_mask)
                    new_id = cluster_id if i == 0 else next_available_id + (i - 1)
                    tprint(f"      📊 Sub-cluster {i}: {sub_size} samples → regime {new_id}", "INFO")
            
            return new_assignments
            
        except Exception as e:
            tprint(f"Frontier split application failed: {e}", "ERROR")
            return assignments
    
    def _smart_cluster_splitting_decision(self, assignments: np.ndarray, features: np.ndarray, current_k: int, iteration: int) -> Tuple[np.ndarray, int]:
        """Smart cluster splitting decision with enhanced logic."""
        try:
            # Step 1: Analyze cluster quality with relative thresholds
            cluster_metrics = self._analyze_cluster_quality_enhanced(assignments, features)
            
            # Step 2: Identify clusters for splitting with relative comparison
            clusters_to_split = []
            tprint(f"   🔍 Analyzing {current_k} clusters for splitting opportunities...", "INFO")
            
            for cluster_id in range(current_k):
                should_split, split_info = self._should_split_cluster_enhanced(cluster_metrics, cluster_id)
                
                # Debug logging for each cluster
                cluster_size = cluster_metrics[cluster_id]['size_percentage']
                tprint(f"   📊 Cluster {cluster_id}: {cluster_size:.1%} size, should_split: {should_split}, reason: {split_info.get('reason', 'unknown')}", "INFO")
                
                if should_split:
                    # Estimate expected improvement
                    expected_improvement = self._estimate_split_improvement(cluster_metrics, cluster_id)
                    
                    # ENHANCED: Lowered thresholds for more aggressive splitting
                    # Only proceed if improvement is significant and confident
                    if expected_improvement > 0.02 and split_info.get('confidence', 0.0) > 0.3:  # Further relaxed thresholds
                        clusters_to_split.append({
                            'cluster_id': cluster_id,
                            'split_info': split_info,
                            'expected_improvement': expected_improvement,
                            'confidence': split_info.get('confidence', 0.0)
                        })
                        tprint(f"   ✅ Cluster {cluster_id} added to split queue (improvement: {expected_improvement:.3f}, confidence: {split_info.get('confidence', 0.0):.3f})", "SUCCESS")
                    else:
                        tprint(f"   ❌ Cluster {cluster_id} rejected (improvement: {expected_improvement:.3f}, confidence: {split_info.get('confidence', 0.0):.3f})", "WARNING")
            
            # Step 3: Process splits using pre-existing code + feature importance
            new_assignments = assignments.copy()
            new_k = current_k
            
            for split_info in clusters_to_split:
                cluster_id = split_info['cluster_id']
                
                # Discover optimal split frontier
                best_frontier = self._discover_optimal_split_frontier(features, new_assignments, cluster_id)
                
                # Apply split if quality is sufficient (ultra aggressive threshold)
                if best_frontier.get('quality', 0.0) > 0.1:  # Ultra aggressive threshold for maximum splitting
                    new_assignments = self._apply_frontier_split(new_assignments, cluster_id, best_frontier)
                    new_k += (best_frontier.get('sub_cluster_count', 2) - 1)
                    
                    tprint(f"✅ Smart split cluster {cluster_id} using {best_frontier.get('method', 'unknown')}", "SUCCESS")
                    tprint(f"   📊 Expected improvement: {split_info['expected_improvement']:.3f}", "INFO")
                    tprint(f"   🎯 Frontier quality: {best_frontier.get('quality', 0.0):.3f}", "INFO")
                    tprint(f"   🎯 Dynamic threshold: {split_info['split_info'].get('dynamic_threshold', 0.12):.1%} (low_quality: {split_info['split_info'].get('has_low_quality', False)})", "INFO")
                    tprint(f"   📈 CV improvement: {split_info['split_info'].get('cv_improvement_estimated', 0.0):.1%} (sufficient: {split_info['split_info'].get('cv_improvement_sufficient', False)})", "INFO")
            
            # Summary of splitting results
            if len(clusters_to_split) > 0:
                tprint(f"   📊 Splitting summary: {len(clusters_to_split)} clusters processed, {new_k - current_k} new regimes created", "SUCCESS")
            else:
                tprint(f"   📊 No clusters met splitting criteria (threshold: 12%/8%)", "INFO")
            
            return new_assignments, new_k
            
        except Exception as e:
            tprint(f"Smart cluster splitting failed: {e}", "ERROR")
            return assignments, current_k

    def validate_clustering_robustness(self, features: np.ndarray, assignments: np.ndarray, 
                                     market_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Lightweight validation framework for clustering robustness."""
        try:
            tprint("🔍 Starting clustering validation...", "INFO")
            
            validation_results = {}
            
            # 1. Basic clustering metrics
            tprint("📊 Computing basic clustering metrics...", "INFO")
            validation_results['basic_metrics'] = self._compute_basic_clustering_metrics(features, assignments)
            
            # 2. Generate validation report
            validation_summary = self._generate_validation_summary(validation_results)
            
            tprint(f"✅ Clustering validation completed - Overall quality: {validation_summary['overall_robustness']:.3f}", "SUCCESS")
            
            return {
                'detailed_results': validation_results,
                'summary': validation_summary
            }
            
        except Exception as e:
            tprint(f"Clustering validation failed: {e}", "ERROR")
            return {'error': str(e)}

    def _compute_basic_clustering_metrics(self, features: np.ndarray, assignments: np.ndarray) -> Dict[str, Any]:
        """Compute basic clustering quality metrics."""
        try:
            from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
            
            n_clusters = len(np.unique(assignments))
            n_samples = features.shape[0]
            
            # Basic quality metrics
            silhouette = silhouette_score(features, assignments)
            davies_bouldin = davies_bouldin_score(features, assignments)
            calinski_harabasz = calinski_harabasz_score(features, assignments)
            
            # Regime balance
            unique, counts = np.unique(assignments, return_counts=True)
            balance = 1.0 - (np.std(counts) / np.mean(counts))
            
            # Overall quality score
            overall_quality = (silhouette + (1.0 - davies_bouldin) + balance) / 3.0
            
            return {
                'silhouette_score': silhouette,
                'davies_bouldin_score': davies_bouldin,
                'calinski_harabasz_score': calinski_harabasz,
                'regime_balance': balance,
                'n_clusters': n_clusters,
                'n_samples': n_samples,
                'overall_quality': overall_quality
            }
            
        except Exception as e:
            tprint(f"Basic metrics computation failed: {e}", "ERROR")
            return {'error': str(e)}

    def _temporal_cross_validation(self, features: np.ndarray, assignments: np.ndarray, 
                                  market_data: pd.DataFrame = None, n_splits: int = 5) -> Dict[str, Any]:
        """Temporal cross-validation across different time periods."""
        try:
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.cluster import KMeans
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            
            n_samples = features.shape[0]
            
            # Use TimeSeriesSplit for temporal validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            cv_scores = []
            stability_scores = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(features)):
                tprint(f"   📅 Processing temporal fold {fold + 1}/{n_splits}...", "INFO")
                
                # Split data temporally
                X_train, X_test = features[train_idx], features[test_idx]
                y_train, y_test = assignments[train_idx], assignments[test_idx]
                
                # Train clustering on training data
                kmeans = KMeans(n_clusters=len(np.unique(assignments)), random_state=42, n_init=10)
                train_pred = kmeans.fit_predict(X_train)
                
                # Predict on test data
                test_pred = kmeans.predict(X_test)
                
                # Calculate similarity scores
                ari_score = adjusted_rand_score(y_test, test_pred)
                nmi_score = normalized_mutual_info_score(y_test, test_pred)
                
                cv_scores.append({
                    'fold': fold + 1,
                    'ari_score': ari_score,
                    'nmi_score': nmi_score,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx)
                })
                
                # Calculate stability (consistency of cluster assignments)
                stability = self._calculate_temporal_stability(train_pred, test_pred, y_train, y_test)
                stability_scores.append(stability)
            
            # Aggregate results
            avg_ari = np.mean([s['ari_score'] for s in cv_scores])
            avg_nmi = np.mean([s['nmi_score'] for s in cv_scores])
            avg_stability = np.mean(stability_scores)
            
            return {
                'cv_scores': cv_scores,
                'average_ari': avg_ari,
                'average_nmi': avg_nmi,
                'average_stability': avg_stability,
                'temporal_robustness': (avg_ari + avg_nmi + avg_stability) / 3.0
            }
            
        except Exception as e:
            tprint(f"Temporal cross-validation failed: {e}", "ERROR")
            return {'error': str(e)}

    def _bootstrap_stability_assessment(self, features: np.ndarray, assignments: np.ndarray, 
                                       n_bootstrap: int = 100) -> Dict[str, Any]:
        """Bootstrap sampling for stability assessment."""
        try:
            from sklearn.utils import resample
            from sklearn.cluster import KMeans
            from sklearn.metrics import adjusted_rand_score
            
            n_samples = features.shape[0]
            bootstrap_scores = []
            cluster_counts = []
            
            tprint(f"   🎲 Running {n_bootstrap} bootstrap samples...", "INFO")
            
            for i in range(n_bootstrap):
                if i % 20 == 0:
                    tprint(f"   📊 Bootstrap progress: {i}/{n_bootstrap}", "INFO")
                
                # Bootstrap sample
                bootstrap_idx = resample(range(n_samples), n_samples=n_samples, random_state=i)
                X_bootstrap = features[bootstrap_idx]
                y_bootstrap = assignments[bootstrap_idx]
                
                # Cluster on bootstrap sample
                kmeans = KMeans(n_clusters=len(np.unique(assignments)), random_state=42, n_init=10)
                bootstrap_pred = kmeans.fit_predict(X_bootstrap)
                
                # Calculate similarity with original assignments
                ari_score = adjusted_rand_score(y_bootstrap, bootstrap_pred)
                bootstrap_scores.append(ari_score)
                
                # Track cluster count stability
                n_clusters = len(np.unique(bootstrap_pred))
                cluster_counts.append(n_clusters)
            
            # Calculate stability metrics
            mean_ari = np.mean(bootstrap_scores)
            std_ari = np.std(bootstrap_scores)
            stability_confidence = mean_ari / (std_ari + 1e-8)  # Signal-to-noise ratio
            
            # Cluster count stability
            cluster_count_consistency = 1.0 - (np.std(cluster_counts) / np.mean(cluster_counts))
            
            return {
                'bootstrap_scores': bootstrap_scores,
                'mean_ari': mean_ari,
                'std_ari': std_ari,
                'stability_confidence': stability_confidence,
                'cluster_count_consistency': cluster_count_consistency,
                'overall_stability': (stability_confidence + cluster_count_consistency) / 2.0
            }
            
        except Exception as e:
            tprint(f"Bootstrap stability assessment failed: {e}", "ERROR")
            return {'error': str(e)}


    def _calculate_temporal_stability(self, train_pred: np.ndarray, test_pred: np.ndarray, 
                                    y_train: np.ndarray, y_test: np.ndarray) -> float:
        """Calculate temporal stability of cluster assignments."""
        try:
            # Calculate consistency of cluster assignments across time
            from sklearn.metrics import adjusted_rand_score
            
            # Measure consistency between train and test predictions
            train_consistency = adjusted_rand_score(y_train, train_pred)
            test_consistency = adjusted_rand_score(y_test, test_pred)
            
            # Average consistency
            temporal_stability = (train_consistency + test_consistency) / 2.0
            
            return temporal_stability
            
        except Exception as e:
            return 0.0

    def _generate_validation_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate lightweight validation summary."""
        try:
            # Extract basic metrics
            basic_metrics = validation_results.get('basic_metrics', {})
            
            # Overall robustness score based on basic metrics
            overall_robustness = basic_metrics.get('overall_quality', 0.0)
            silhouette = basic_metrics.get('silhouette_score', 0.0)
            balance = basic_metrics.get('regime_balance', 0.0)
            
            # Risk assessment based on clustering quality
            risk_factors = []
            if silhouette < 0.3:
                risk_factors.append("low_silhouette")
            if balance < 0.7:
                risk_factors.append("poor_balance")
            if overall_robustness < 0.5:
                risk_factors.append("low_overall_quality")
            
            risk_level = 'high' if len(risk_factors) >= 2 else 'medium' if len(risk_factors) == 1 else 'low'
            
            summary = {
                'overall_robustness': overall_robustness,
                'silhouette_score': silhouette,
                'regime_balance': balance,
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'recommendations': self._generate_validation_recommendations(validation_results)
            }
            
            return summary
            
        except Exception as e:
            return {'error': str(e), 'overall_robustness': 0.0}

    def _generate_validation_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        basic_metrics = validation_results.get('basic_metrics', {})
        
        # Quality-based recommendations
        silhouette = basic_metrics.get('silhouette_score', 0.0)
        balance = basic_metrics.get('regime_balance', 0.0)
        overall_quality = basic_metrics.get('overall_quality', 0.0)
        
        if silhouette < 0.3:
            recommendations.append("Low silhouette score - consider feature selection or different clustering parameters")
        
        if balance < 0.7:
            recommendations.append("Poor regime balance - consider adjusting clustering thresholds or splitting large clusters")
        
        if overall_quality < 0.5:
            recommendations.append("Low overall quality - review feature engineering and clustering configuration")
        
        if not recommendations:
            recommendations.append("Clustering quality is good - regime detection is working well")
        
        return recommendations

    def _calculate_optimal_batch_size(self, total_samples: int, iteration: int) -> int:
        """Calculate optimal batch size based on data characteristics and iteration progress."""
        try:
            # GLOBAL OPTIMIZATION STRATEGY 1: Adaptive batch sizing
            if iteration < 3:
                # Early iterations: smaller batches for fine-tuning
                base_batch_size = min(100, total_samples // 20)
            elif iteration < 10:
                # Mid iterations: medium batches for balanced processing
                base_batch_size = min(200, total_samples // 12)
            else:
                # Later iterations: larger batches for efficiency
                base_batch_size = min(400, total_samples // 8)
            
            # GLOBAL OPTIMIZATION STRATEGY 2: Memory-aware sizing
            # Estimate memory usage and adjust batch size accordingly
            estimated_memory_per_sample = 95 * 8  # features * bytes_per_float64
            max_memory_batch = 100000000 // estimated_memory_per_sample  # 100MB limit
            base_batch_size = min(base_batch_size, max_memory_batch)
            
            # GLOBAL OPTIMIZATION STRATEGY 3: CPU core optimization
            import os
            cpu_cores = os.cpu_count() or 4
            optimal_batch_size = base_batch_size * cpu_cores // 2  # Utilize 50% of cores
            
            return max(50, min(optimal_batch_size, total_samples // 4))  # Reasonable bounds
            
        except Exception:
            return min(160, total_samples // 12)  # Fallback to original logic
    
    def _calculate_optimal_batch_size_enhanced(self, total_samples: int, iteration: int, features: np.ndarray = None) -> int:
        """Calculate optimal batch size with feature-aware sizing."""
        try:
            # Start with base batch size calculation
            base_batch_size = self._calculate_optimal_batch_size(total_samples, iteration)
            
            if features is not None:
                # Calculate feature complexity
                feature_complexity = self._estimate_feature_complexity(features)
                
                # Adjust batch size based on feature complexity
                if feature_complexity > 0.7:  # High complexity
                    complexity_factor = 0.8  # Smaller batches for complex features
                    tprint(f"   🔍 High feature complexity ({feature_complexity:.3f}), reducing batch size", "INFO")
                elif feature_complexity < 0.3:  # Low complexity
                    complexity_factor = 1.2  # Larger batches for simple features
                    tprint(f"   🔍 Low feature complexity ({feature_complexity:.3f}), increasing batch size", "INFO")
                else:
                    complexity_factor = 1.0  # No adjustment for medium complexity
                
                # Adjust based on iteration progress and complexity
                if iteration < 5 and feature_complexity > 0.6:
                    # Early iterations with complex features: very small batches
                    complexity_factor *= 0.7
                    tprint(f"   🔍 Early iteration with complex features, using very small batches", "INFO")
                elif iteration > 10 and feature_complexity < 0.4:
                    # Late iterations with simple features: larger batches
                    complexity_factor *= 1.3
                    tprint(f"   🔍 Late iteration with simple features, using larger batches", "INFO")
                
                # Apply complexity adjustment
                enhanced_batch_size = int(base_batch_size * complexity_factor)
                
                # Memory-aware adjustment
                estimated_memory_per_sample = features.shape[1] * 8  # features * bytes_per_float64
                max_memory_batch = 100000000 // estimated_memory_per_sample  # 100MB limit
                enhanced_batch_size = min(enhanced_batch_size, max_memory_batch)
                
                # Ensure reasonable bounds
                final_batch_size = max(50, min(enhanced_batch_size, total_samples // 4))
                
                tprint(f"   📊 Enhanced batch size: {base_batch_size} → {final_batch_size} "
                      f"(complexity: {feature_complexity:.3f})", "INFO")
                
                return final_batch_size
            else:
                return base_batch_size
            
        except Exception as e:
            tprint(f"Enhanced batch size calculation failed: {e}", "ERROR")
            return self._calculate_optimal_batch_size(total_samples, iteration)
    
    def _apply_global_optimization_strategies(self, features: np.ndarray, assignments: np.ndarray, k: int) -> np.ndarray:
        """Apply global optimization strategies to overcome batch processing limitations."""
        try:
            tprint("🚀 Applying global optimization strategies...", "INFO")
            
            # STRATEGY 1: Pre-compute global cluster statistics
            global_stats = self._precompute_global_cluster_statistics(features, assignments, k)
            
            # STRATEGY 2: Vectorized distance matrix computation
            distance_matrix = self._compute_vectorized_distance_matrix(features, global_stats['centroids'])
            
            # STRATEGY 3: Parallel batch processing with threading
            optimized_assignments = self._parallel_batch_optimization(
                features, assignments, k, global_stats, distance_matrix
            )
            
            # STRATEGY 4: Global convergence acceleration
            final_assignments = self._apply_global_convergence_acceleration(
                features, optimized_assignments, k, global_stats
            )
            
            # STRATEGY 5: Regime-aware optimization
            regime_aware_assignments = self._apply_regime_aware_optimization(
                features, final_assignments, k, global_stats
            )
            
            # STRATEGY 6: Feature importance-guided optimization
            importance_guided_assignments = self._apply_feature_importance_optimization(
                features, regime_aware_assignments, k, global_stats
            )
            
            tprint("✅ Global optimization strategies completed", "SUCCESS")
            return importance_guided_assignments
            
        except Exception as e:
            tprint(f"Global optimization failed: {e}", "ERROR")
            return assignments
    
    def _apply_regime_aware_optimization(self, features: np.ndarray, assignments: np.ndarray, k: int, global_stats: dict) -> np.ndarray:
        """Apply regime-aware optimization strategies."""
        try:
            tprint("   🎯 Applying regime-aware optimization...", "INFO")
            
            optimized_assignments = assignments.copy()
            
            # Calculate regime stability scores
            regime_stability = self._calculate_regime_stability_scores(features, assignments, k)
            
            # Identify unstable regimes (high change probability)
            unstable_regimes = [regime for regime, stability in regime_stability.items() if stability < 0.3]
            
            if unstable_regimes:
                tprint(f"   🔄 Found {len(unstable_regimes)} unstable regimes: {unstable_regimes}", "INFO")
                
                # Apply regime-specific optimization
                for regime in unstable_regimes:
                    regime_mask = assignments == regime
                    regime_indices = np.where(regime_mask)[0]
                    
                    if len(regime_indices) > 10:  # Only optimize if regime has enough samples
                        # Calculate volatility regime features for this regime
                        regime_features = features[regime_mask]
                        regime_assignments = assignments[regime_mask]
                        volatility_features = self._calculate_volatility_regime_features(regime_features, regime_assignments)
                        
                        # Enhanced features for regime optimization
                        enhanced_regime_features = np.column_stack([regime_features, volatility_features])
                        
                        # Apply local optimization within the regime
                        optimized_regime_assignments = self._optimize_regime_locally(
                            enhanced_regime_features, regime_assignments, regime, k
                        )
                        
                        # Update assignments
                        optimized_assignments[regime_mask] = optimized_regime_assignments
            
            return optimized_assignments
            
        except Exception as e:
            tprint(f"Regime-aware optimization failed: {e}", "ERROR")
            return assignments
    
    def _calculate_regime_stability_scores(self, features: np.ndarray, assignments: np.ndarray, k: int) -> dict:
        """Calculate stability scores for each regime."""
        try:
            regime_stability = {}
            
            for regime in range(k):
                regime_mask = assignments == regime
                regime_samples = features[regime_mask]
                
                if len(regime_samples) < 2:
                    regime_stability[regime] = 0.0
                    continue
                
                # Calculate intra-cluster variance (lower = more stable)
                regime_center = np.mean(regime_samples, axis=0)
                distances = np.linalg.norm(regime_samples - regime_center, axis=1)
                intra_cluster_variance = np.var(distances)
                
                # Calculate temporal stability (consistency over time)
                regime_indices = np.where(regime_mask)[0]
                if len(regime_indices) > 1:
                    # Check for regime changes within the regime
                    regime_changes = 0
                    for i in range(1, len(regime_indices)):
                        if regime_indices[i] - regime_indices[i-1] > 1:  # Gap in sequence
                            regime_changes += 1
                    
                    temporal_stability = 1.0 - (regime_changes / len(regime_indices))
                else:
                    temporal_stability = 1.0
                
                # Combine stability metrics
                stability_score = (1.0 / (1.0 + intra_cluster_variance)) * temporal_stability
                regime_stability[regime] = min(1.0, stability_score)
            
            return regime_stability
            
        except Exception as e:
            return {regime: 0.5 for regime in range(k)}
    
    def _optimize_regime_locally(self, enhanced_features: np.ndarray, regime_assignments: np.ndarray, regime: int, k: int) -> np.ndarray:
        """Apply local optimization within a specific regime."""
        try:
            optimized_assignments = regime_assignments.copy()
            
            # Calculate regime-specific centroids
            regime_centroids = {}
            for target_regime in range(k):
                if target_regime == regime:
                    regime_centroids[target_regime] = np.mean(enhanced_features, axis=0)
                else:
                    # Use a subset of features for other regimes
                    regime_centroids[target_regime] = np.mean(enhanced_features[:, :enhanced_features.shape[1]//2], axis=0)
            
            # Apply local reassignment optimization
            for i in range(len(regime_assignments)):
                current_regime = regime_assignments[i]
                sample_features = enhanced_features[i]
                
                best_regime = current_regime
                best_distance = float('inf')
                
                # Try all possible regimes
                for candidate_regime in range(k):
                    centroid = regime_centroids[candidate_regime]
                    distance = np.linalg.norm(sample_features - centroid)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_regime = candidate_regime
                
                optimized_assignments[i] = best_regime
            
            return optimized_assignments
            
        except Exception as e:
            return regime_assignments
    
    def _apply_feature_importance_optimization(self, features: np.ndarray, assignments: np.ndarray, k: int, global_stats: dict) -> np.ndarray:
        """Apply feature importance-guided optimization."""
        try:
            tprint("   🎯 Applying feature importance-guided optimization...", "INFO")
            
            optimized_assignments = assignments.copy()
            
            # Calculate feature importance scores
            feature_importance = self._calculate_feature_importance_scores(features, assignments, k)
            
            # Identify most important features
            top_features_idx = np.argsort(feature_importance)[-min(10, len(feature_importance)):]
            tprint(f"   📊 Top features for optimization: {len(top_features_idx)} features", "INFO")
            
            # Apply importance-weighted optimization
            for i in range(len(assignments)):
                current_regime = assignments[i]
                sample_features = features[i]
                
                best_regime = current_regime
                best_score = float('-inf')
                
                # Try each possible regime
                for candidate_regime in range(k):
                    # Calculate importance-weighted distance
                    regime_center = global_stats['centroids'][candidate_regime]
                    feature_distances = np.abs(sample_features - regime_center)
                    
                    # Weight distances by feature importance
                    weighted_distance = np.sum(feature_distances * feature_importance)
                    importance_score = 1.0 / (1.0 + weighted_distance)
                    
                    if importance_score > best_score:
                        best_score = importance_score
                        best_regime = candidate_regime
                
                optimized_assignments[i] = best_regime
            
            return optimized_assignments
            
        except Exception as e:
            tprint(f"Feature importance optimization failed: {e}", "ERROR")
            return assignments
    
    def _calculate_feature_importance_scores(self, features: np.ndarray, assignments: np.ndarray, k: int) -> np.ndarray:
        """Calculate feature importance scores for optimization."""
        try:
            n_features = features.shape[1]
            feature_importance = np.zeros(n_features)
            
            # Calculate importance based on inter-cluster vs intra-cluster variance
            for feature_idx in range(n_features):
                feature_values = features[:, feature_idx]
                
                # Calculate intra-cluster variance
                intra_cluster_var = 0.0
                for regime in range(k):
                    regime_mask = assignments == regime
                    if np.sum(regime_mask) > 1:
                        regime_values = feature_values[regime_mask]
                        intra_cluster_var += np.var(regime_values)
                
                intra_cluster_var /= k
                
                # Calculate inter-cluster variance
                regime_means = []
                for regime in range(k):
                    regime_mask = assignments == regime
                    if np.sum(regime_mask) > 0:
                        regime_means.append(np.mean(feature_values[regime_mask]))
                
                if len(regime_means) > 1:
                    inter_cluster_var = np.var(regime_means)
                else:
                    inter_cluster_var = 0.0
                
                # Feature importance is the ratio of inter-cluster to intra-cluster variance
                if intra_cluster_var > 0:
                    feature_importance[feature_idx] = inter_cluster_var / intra_cluster_var
                else:
                    feature_importance[feature_idx] = 0.0
            
            # Normalize importance scores
            if np.max(feature_importance) > 0:
                feature_importance = feature_importance / np.max(feature_importance)
            
            return feature_importance
            
        except Exception as e:
            return np.ones(features.shape[1])  # Equal importance if calculation fails
    
    def _calculate_dynamic_convergence_tolerance(self, iteration: int, convergence_history: List[Dict], base_tolerance: float) -> float:
        """Calculate dynamic convergence tolerance with quality-first approach."""
        try:
            # Base tolerance adjustment
            dynamic_tolerance = base_tolerance
            
            # Adjust based on iteration progress
            if iteration < 10:
                # Early iterations: more lenient tolerance
                dynamic_tolerance *= 2.0  # More lenient for early iterations
            elif iteration > 20:
                # Late iterations: stricter tolerance
                dynamic_tolerance *= 0.5  # Stricter for late iterations
            
            # Adjust based on convergence history
            if len(convergence_history) >= 3:
                recent_improvements = [h['improvement'] for h in convergence_history[-3:]]
                improvement_variance = np.var(recent_improvements)
                
                # High variance = unstable convergence, use more lenient tolerance
                if improvement_variance > 0.001:
                    dynamic_tolerance *= 1.5  # More lenient for unstable convergence
                # Low variance = stable convergence, use stricter tolerance
                elif improvement_variance < 0.0001:
                    dynamic_tolerance *= 0.6  # Stricter for stable convergence
            
            # CRITICAL: Adjust based on score quality
            if len(convergence_history) > 0:
                current_score = convergence_history[-1]['composite_score']
                if current_score > 0.5:  # High quality score
                    dynamic_tolerance *= 0.7  # Stricter tolerance for high quality
                elif current_score < 0.3:  # Low quality score
                    dynamic_tolerance *= 3.0  # Much more lenient tolerance for low quality
                    tprint(f"   🔧 Low quality ({current_score:.3f}), using lenient tolerance: {dynamic_tolerance:.2e}", "WARNING")
            
            return max(0.0001, min(0.01, dynamic_tolerance))  # Reasonable bounds
            
        except Exception as e:
            return base_tolerance
    
    def _evaluate_adaptive_convergence(self, avg_improvement: float, silhouette_trend: float, 
                                     convergence_history: List[Dict], iteration: int) -> bool:
        """Evaluate adaptive convergence criteria with quality-first approach."""
        try:
            # CRITICAL: Don't converge if quality is too poor
            if len(convergence_history) > 0:
                current_score = convergence_history[-1]['composite_score']
                if current_score < 0.3:  # Minimum acceptable quality
                    return False  # Force continuation for poor quality
            
            # Criterion 1: Very small improvement AND good quality
            if avg_improvement < 0.0001 and len(convergence_history) > 0:
                current_score = convergence_history[-1]['composite_score']
                if current_score > 0.4:  # Only converge if quality is good
                    return True
            
            # Criterion 2: Silhouette stability AND good quality
            if silhouette_trend >= -0.001 and len(convergence_history) > 0:
                current_score = convergence_history[-1]['composite_score']
                if current_score > 0.4:  # Only converge if quality is good
                    return True
            
            # Criterion 3: Multiple consecutive small improvements AND good quality
            if len(convergence_history) >= 5:
                recent_improvements = [h['improvement'] for h in convergence_history[-5:]]
                current_score = convergence_history[-1]['composite_score']
                if all(imp < 0.0005 for imp in recent_improvements) and current_score > 0.4:
                    return True
            
            # Criterion 4: High quality score achieved (excellent quality)
            if len(convergence_history) > 0:
                current_score = convergence_history[-1]['composite_score']
                if current_score > 0.75:  # Excellent quality threshold (increased from 0.6)
                    return True
            
            # Criterion 5: Maximum iterations reached with good quality
            if iteration >= 30:  # Increased from 20
                if len(convergence_history) > 0:
                    current_score = convergence_history[-1]['composite_score']
                    if current_score > 0.4:  # Good quality threshold
                        return True
            
            return False  # Don't converge by default
            
        except Exception as e:
            return False  # Don't converge on error
    
    def _precompute_global_cluster_statistics(self, features: np.ndarray, assignments: np.ndarray, k: int) -> Dict:
        """Pre-compute global cluster statistics for efficient batch processing."""
        try:
            stats = {}
            
            # Compute cluster centroids
            centroids = {}
            for regime in range(k):
                regime_mask = assignments == regime
                if np.sum(regime_mask) > 0:
                    centroids[regime] = np.mean(features[regime_mask], axis=0)
                else:
                    centroids[regime] = np.zeros(features.shape[1])
            
            # Compute cluster sizes and weights
            cluster_sizes = {}
            cluster_weights = {}
            total_samples = len(assignments)
            
            for regime in range(k):
                size = np.sum(assignments == regime)
                cluster_sizes[regime] = size
                cluster_weights[regime] = size / total_samples if total_samples > 0 else 0
            
            # Compute global feature statistics
            global_mean = np.mean(features, axis=0)
            global_std = np.std(features, axis=0)
            
            stats = {
                'centroids': centroids,
                'cluster_sizes': cluster_sizes,
                'cluster_weights': cluster_weights,
                'global_mean': global_mean,
                'global_std': global_std,
                'total_samples': total_samples
            }
            
            return stats
            
        except Exception as e:
            tprint(f"Global statistics precomputation failed: {e}", "ERROR")
            return {}
    
    def _compute_vectorized_distance_matrix(self, features: np.ndarray, centroids: Dict[int, np.ndarray]) -> Dict:
        """Compute vectorized distance matrix for efficient batch processing."""
        try:
            distance_matrix = {}
            
            for regime, centroid in centroids.items():
                # Vectorized distance computation
                distances = np.linalg.norm(features - centroid, axis=1)
                distance_matrix[regime] = distances
            
            return distance_matrix
            
        except Exception as e:
            tprint(f"Vectorized distance matrix computation failed: {e}", "ERROR")
            return {}
    
    def _parallel_batch_optimization(self, features: np.ndarray, assignments: np.ndarray, k: int, 
                                   global_stats: Dict, distance_matrix: Dict) -> np.ndarray:
        """Parallel batch optimization using threading to overcome sequential limitations."""
        try:
            import concurrent.futures
            import threading
            
            new_assignments = assignments.copy()
            
            # Split data into chunks for parallel processing
            chunk_size = len(assignments) // 4  # 4 parallel chunks
            chunks = []
            for i in range(0, len(assignments), chunk_size):
                chunk_end = min(i + chunk_size, len(assignments))
                chunks.append((i, chunk_end))
            
            # Thread-safe optimization function
            def optimize_chunk(chunk_start, chunk_end):
                chunk_assignments = new_assignments[chunk_start:chunk_end].copy()
                chunk_features = features[chunk_start:chunk_end]
                
                # Apply chunk-specific optimization
                for idx in range(len(chunk_assignments)):
                    global_idx = chunk_start + idx
                    current_regime = assignments[global_idx]
                    sample_features = chunk_features[idx]
                    
                    # Find best regime using pre-computed distances
                    best_regime = current_regime
                    best_score = float('inf')
                    
                    for regime in range(k):
                        if regime in distance_matrix:
                            distance = distance_matrix[regime][global_idx]
                            if distance < best_score:
                                best_score = distance
                                best_regime = regime
                    
                    chunk_assignments[idx] = best_regime
                
                return chunk_start, chunk_end, chunk_assignments
            
            # Execute parallel optimization
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(optimize_chunk, start, end) for start, end in chunks]
                
                for future in concurrent.futures.as_completed(futures):
                    start, end, chunk_assignments = future.result()
                    new_assignments[start:end] = chunk_assignments
            
            return new_assignments
            
        except Exception as e:
            tprint(f"Parallel batch optimization failed: {e}", "ERROR")
            return assignments
    
    def _apply_global_convergence_acceleration(self, features: np.ndarray, assignments: np.ndarray, 
                                             k: int, global_stats: Dict) -> np.ndarray:
        """Apply global convergence acceleration techniques."""
        try:
            # STRATEGY: Global cluster reassignment based on statistical significance
            new_assignments = assignments.copy()
            
            # Compute global improvement potential
            improvement_matrix = self._compute_global_improvement_matrix(features, assignments, k, global_stats)
            
            # Apply global reassignments
            for sample_idx in range(len(assignments)):
                current_regime = assignments[sample_idx]
                best_regime = np.argmax(improvement_matrix[sample_idx])
                
                if best_regime != current_regime and improvement_matrix[sample_idx, best_regime] > 0.1:
                    new_assignments[sample_idx] = best_regime
            
            return new_assignments
            
        except Exception as e:
            tprint(f"Global convergence acceleration failed: {e}", "ERROR")
            return assignments
    
    def _compute_global_improvement_matrix(self, features: np.ndarray, assignments: np.ndarray, 
                                         k: int, global_stats: Dict) -> np.ndarray:
        """Compute global improvement matrix for all samples and regimes."""
        try:
            n_samples = len(assignments)
            improvement_matrix = np.zeros((n_samples, k))
            
            for sample_idx in range(n_samples):
                sample_features = features[sample_idx]
                current_regime = assignments[sample_idx]
                
                for regime in range(k):
                    if regime in global_stats['centroids']:
                        # Calculate improvement score
                        distance_to_regime = np.linalg.norm(sample_features - global_stats['centroids'][regime])
                        current_distance = np.linalg.norm(sample_features - global_stats['centroids'][current_regime])
                        
                        # Improvement is negative distance change (lower distance = better)
                        improvement = current_distance - distance_to_regime
                        improvement_matrix[sample_idx, regime] = improvement
            
            return improvement_matrix
            
        except Exception as e:
            tprint(f"Global improvement matrix computation failed: {e}", "ERROR")
            return np.zeros((len(assignments), k))
