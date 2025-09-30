"""
NAS-TAS Clustering Component.

This component uses shared utilities to eliminate redundancy between NAS and TAS components.
It demonstrates how to use the shared_utils package for common functionality.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple
from dataclasses import dataclass, field
import traceback
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

from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult
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
        get_m1_memory_pool_manager,
        get_m1_memory_monitor
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
                self.original_features, self.optimized_features,
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
class NASTASClusteringConfig(BaseConfig):
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


class NASTASClusteringComponent(BaseMarketAnalysisComponent):
    """
    NAS-TAS Clustering Component.
    
    This component uses shared utilities to eliminate redundancy:
    - Uses shared feature preparation
    - Uses shared configuration validation
    - Uses shared logging utilities
    - Uses shared metrics calculation
    - Uses shared regime characteristics generation
    """
    
    def __init__(self, config: Optional[NASTASClusteringConfig] = None):
        """Initialize the NAS-TAS clustering component with enhanced capabilities."""
        with LoggingContext('NAS-TAS-Clustering', 'Initialization', verbose=True):
            super().__init__(config)

            # Use shared logging utilities
            self.logger = get_logger('NASTASClustering')
            
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

            # Learned metric weight state
            self.metric_weight_history: List[Dict[str, Any]] = []
            self.learned_weights: Dict[str, Dict[str, float]] = {}
            self._weight_history_limit: int = 100
            self._default_metric_weights: Dict[str, Dict[str, float]] = {
                'composite': {
                    'silhouette': 0.25,
                    'davies_bouldin': 0.20,
                    'calinski_harabasz': 0.20,
                    'stability': 0.20,
                    'consensus': 0.15,
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
            
            # Perform GMM clustering
            from sklearn.mixture import GaussianMixture
            gmm = GaussianMixture(
                n_components=config.n_regimes,
                random_state=42,
                max_iter=100
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
            
            result = {
                'overall_score': float(overall_score),
                'silhouette_score': float(silhouette),
                'davies_bouldin_score': float(davies_bouldin),
                'calinski_harabasz_score': float(calinski_harabasz),
                'cv_score': float(cv_score),
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

    def _estimate_regime_range(
        self,
        pipeline_state: Dict[str, Any],
    ) -> Tuple[int, int, int]:
        """Estimate regime count bounds using discovery metrics."""

        default_min = getattr(self.config, 'regime_search_min', 5)
        default_max = getattr(self.config, 'regime_search_max', 15)
        default_mode = getattr(self.config, 'n_regimes', 8)

        discovery_result = pipeline_state.get('nas_tas_regime_discovery_result', {}) or {}

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

    def _extract_regime_counts(self, pipeline_state: Dict[str, Any]) -> int:
        """Extract the number of regimes to use for clustering using data-driven approach."""
        tprint("📈 Step 1: Extracting regime count from previous step artifacts...", "INFO")

        min_regimes, max_regimes, default_regimes = self._estimate_regime_range(pipeline_state)
        self.config.regime_search_min = min_regimes
        self.config.regime_search_max = max_regimes

        regime_discovery_result = pipeline_state.get('nas_tas_regime_discovery_result', {})
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

        # Apply evidence-driven bounds derived from discovery metrics
        n_regimes = max(min_regimes, min(max_regimes, n_regimes))

        tprint(
            f"Final regime count: {n_regimes} (data-driven, no hardcoded heuristics)",
            "SUCCESS"
        )
        self.config.n_regimes = n_regimes
        return n_regimes

    def _validate_configuration(self) -> None:
        """Validate configuration using shared utilities."""
        tprint("Step 2: Validating inputs and configuration using shared utilities", "INFO")
        validation_errors = self.config_validator.validate_config(self.config)
        if validation_errors:
            tprint(f"Configuration validation failed: {validation_errors}", "ERROR")
            raise ValueError(f"Configuration validation failed: {validation_errors}")

        tprint("Configuration validation passed using shared utilities", "SUCCESS")

    def _initialize_execution_metadata(self) -> None:
        """Initialize execution metadata for downstream use."""
        self.execution_metadata = {
            'start_time': datetime.now().isoformat(),
            'symbol': getattr(self.config, 'symbol', 'BTCUSDT'),
            'timeframe': getattr(self.config, 'timeframe', '15m'),
            'exchange': getattr(self.config, 'exchange', 'binance'),
            'component': 'refactored_nas_tas_clustering',
            'uses_shared_utilities': True,
            'quality_calibration': get_current_calibration(),
            'calibration_loaded_from_state': False,
        }

    def _restore_learned_weights_from_state(self, pipeline_state: Dict[str, Any]) -> None:
        """Restore learned metric weights and history from prior pipeline state."""
        if not isinstance(pipeline_state, dict):
            return

        restored_weights: Dict[str, Dict[str, float]] = {}
        restored_history: List[Dict[str, Any]] = []

        for container in self._iterate_weight_containers(pipeline_state):
            weights = container.get('learned_metric_weights')
            if isinstance(weights, dict):
                for group, group_weights in weights.items():
                    sanitized = self._sanitize_weight_dict(group, group_weights)
                    if sanitized:
                        restored_weights[group] = sanitized

            history = container.get('metric_weight_history')
            if isinstance(history, list) and not restored_history:
                restored_history = history

        if restored_weights:
            self.learned_weights.update(restored_weights)

        if restored_history:
            sanitized_history = self._sanitize_metric_history(restored_history)
            if sanitized_history:
                self.metric_weight_history = sanitized_history[-self._weight_history_limit:]

    def _iterate_weight_containers(self, node: Any) -> Iterator[Dict[str, Any]]:
        """Yield nested containers that may store learned weight metadata."""
        if isinstance(node, dict):
            if 'learned_metric_weights' in node or 'metric_weight_history' in node:
                yield node
            for value in node.values():
                yield from self._iterate_weight_containers(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                yield from self._iterate_weight_containers(item)

    def _sanitize_weight_dict(self, group: str, weights: Any) -> Dict[str, float]:
        """Convert a raw weight mapping into a normalized simplex weight dict."""
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

    def _load_calibration_history(self, pipeline_state: Dict[str, Any]) -> None:
        """Load calibration history from the pipeline state if available."""

        calibration_payload = None
        if isinstance(pipeline_state, dict):
            previous_result = pipeline_state.get('nas_tas_clustering_result')
            if isinstance(previous_result, dict):
                execution_meta = previous_result.get('execution_metadata', {})
                if isinstance(execution_meta, dict):
                    calibration_payload = execution_meta.get('quality_calibration')

            if calibration_payload is None:
                calibration_payload = pipeline_state.get('nas_tas_clustering_calibration')

        if calibration_payload:
            update_quality_calibration(calibration_payload)
            self.execution_metadata['quality_calibration'] = get_current_calibration()
            self.execution_metadata['calibration_loaded_from_state'] = True
        else:
            update_quality_calibration(self.execution_metadata.get('quality_calibration'))

        thresholds = get_calibrated_thresholds()
        self.config.min_regime_persistence = thresholds.get('min_regime_persistence', self.config.min_regime_persistence)
        self.config.max_feature_noise_ratio = thresholds.get('max_feature_noise_ratio', self.config.max_feature_noise_ratio)
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

            quality_thresholds = {
                'min_regime_persistence': quantiles_map['persistence'].get('p50', current_thresholds.get('min_regime_persistence', 0.7)),
                'max_feature_noise_ratio': quantiles_map['noise_ratio'].get('p75', current_thresholds.get('max_feature_noise_ratio', 0.3)),
                'min_temporal_stability': quantiles_map['temporal_stability'].get('p50', current_thresholds.get('min_temporal_stability', 0.6)),
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

    def _prepare_features(self, market_data: pd.DataFrame) -> Any:
        """Prepare market features for clustering."""
        tprint("Step 4: Preparing features using shared utilities", "INFO")
        features = prepare_market_features(market_data, self.feature_config, verbose=True)
        if features is None:
            tprint("Failed to prepare features for clustering", "ERROR")
            raise ValueError("Failed to prepare features for clustering")

        self.features = features
        tprint(f"Features prepared: {features.shape}", "SUCCESS")
        return features

    def _generate_cluster_characteristics(
        self,
        market_data: pd.DataFrame,
        clustering_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate characteristics for each cluster."""
        tprint("Step 8: Generating cluster characteristics using shared utilities", "INFO")
        cluster_characteristics = generate_cluster_characteristics(
            market_data,
            clustering_result['cluster_assignments'],
            clustering_result.get('cluster_centers'),
            verbose=True,
        )
        tprint("Cluster characteristics generated", "SUCCESS")
        return cluster_characteristics

    def _calculate_clustering_metrics_using_shared_utils(
        self,
        clustering_result: Dict[str, Any],
        cluster_characteristics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate clustering metrics using shared utilities with defensive error handling."""
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
            consensus_metrics = self.metrics_calculator.calculate_consensus_metrics(
                assignments_array, verbose=True
            )
            disagreement_metrics = self.metrics_calculator.calculate_disagreement_metrics(
                assignments_array, verbose=True
            )
            economic_scores = self.metrics_calculator.calculate_economic_scores(
                assignments_array, verbose=True
            )
            trading_scores = self.metrics_calculator.calculate_trading_scores(
                assignments_array, verbose=True
            )
            stability_scores = self.metrics_calculator.calculate_stability_scores(
                assignments_array, verbose=True
            )
            
            # Compile metrics dictionary
            clustering_metrics = {
                "consensus_metrics": consensus_metrics,
                "disagreement_metrics": disagreement_metrics,
                "economic_scores": economic_scores,
                "trading_scores": trading_scores,
                "stability_scores": stability_scores,
                "cluster_centers": cluster_centers,
                "n_clusters": len(set(cluster_assignments)),
                "total_assignments": len(cluster_assignments)
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
                    'execution_time': clustering_result.get('execution_time', 0.0)
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

    @log_execution('NAS-TAS-Clustering', 'NAS-TAS Clustering', verbose=True)
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute NAS-TAS clustering using shared utilities.
        
        Args:
            data: Market data for clustering
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with clustering results
        """
        try:
            # Input validation
            self._validate_execution_inputs(data, pipeline_state)
            tprint("🚀 Starting NAS-TAS clustering execution with M1 hardware optimization", "INFO")
            
            # Store pipeline state as instance attribute for use in other methods
            self.pipeline_state = pipeline_state
            self._restore_learned_weights_from_state(pipeline_state)
            
            # Initialize performance monitoring
            tprint("📊 Initializing performance monitoring...", "INFO")
            start_time = time.time()
            
            # Step 1: Extract regime count from previous step artifacts
            n_regimes = self._extract_regime_counts(pipeline_state)
            self.config.n_regimes = n_regimes
            tprint(f"Using extracted regime count: {n_regimes}", "INFO")

            # Step 2: Validate inputs and configuration using shared utilities
            self._validate_configuration()

            # Step 3: Initialize execution metadata
            self._initialize_execution_metadata()
            self._load_calibration_history(pipeline_state)

            # Step 4: Load and validate market data
            tprint("Step 4: Loading and validating market data", "INFO")
            market_data = await self._load_market_data(data)
            if market_data is None or market_data.empty:
                tprint("No market data available for clustering", "ERROR")
                raise ValueError("No market data available for clustering")

            tprint(f"Market data loaded: {len(market_data)} rows", "SUCCESS")

            # Step 4: Prepare features using shared utilities
            features = self._prepare_features(market_data)

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

            tprint(f'NAS-TAS Clustering completed: {clustering_result["n_clusters"]} clusters', "SUCCESS")
            
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': getattr(self.config, 'symbol', 'BTCUSDT'),
                    'timeframe': getattr(self.config, 'timeframe', '15m'),
                    'data_points_processed': len(market_data),
                    'n_clusters': clustering_result['n_clusters'],
                    'algorithm_type': 'nas_tas_clustering',
                    'execution_successful': True,
                    'uses_shared_utilities': True
                }
            )
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            
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

            return ComponentResult(
                success=False,
                artifacts={
                    "error_details": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "traceback": error_traceback,
                        "timestamp": datetime.now().isoformat()
                    }
                },
                error_message=f"NAS-TAS clustering failed: {str(e)}",
                metadata={
                    'symbol': getattr(self.config, 'symbol', 'BTCUSDT'),
                    'timeframe': getattr(self.config, 'timeframe', '15m'),
                    'execution_successful': False,
                    'error_type': type(e).__name__
                }
            )
    
    async def _load_market_data(self, data: Any) -> Optional[pd.DataFrame]:
        """Load and validate market data for clustering."""
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
    
    def _create_clustering_config_using_shared_utils(self) -> Dict[str, Any]:
        """Create clustering configuration using shared utilities."""
        try:
            tprint("Creating clustering configuration using shared utilities...", "INFO")

            # Use shared utilities to create configuration
            tprint("Creating base configuration...", "INFO")
            base_config = create_default_config(
                config_type="hybrid",
                symbol=getattr(self.config, 'symbol', 'BTCUSDT'),
                timeframe=getattr(self.config, 'timeframe', '15m'),
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
                'symbol': getattr(self.config, 'symbol', 'BTCUSDT'),
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
                symbol=getattr(self.config, 'symbol', 'BTCUSDT'),
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
    
    
    async def _perform_clustering(self, features: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering using advanced optimization methods."""
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
    
    async def _perform_advanced_clustering(self, features: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform advanced clustering using progressive regime optimization."""
        try:
            tprint("Starting progressive regime optimization...", "INFO")

            context = ClusteringContext(
                original_features=features, 
                market_data=market_data,
                memory_optimizer=self.memory_optimizer
            )

            # Step 1: Feature selection and dimensionality reduction
            tprint("Step 1: Feature selection and dimensionality reduction...", "INFO")
            self._optimize_features(context)

            # Step 2: Select optimal K using BIC-selected GMM
            self._select_optimal_k(context)

            # Step 3/4: Reconcile NAS/TAS labels and prepare optimization metrics
            self._reconcile_labels(context)

            # Step 5: Smooth regime assignments for temporal coherence
            self._smooth_assignments(context)

            # Final summary and artifact packaging
            clustering_result = self._summarize_results(context)

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

            # Step 1: Standardize features (keep standardization as-is)
            tprint("Step 1: Standardizing features...", "INFO")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(context.original_features)
            tprint(f"Feature standardization completed: {context.original_features.shape}", "SUCCESS")

            # Step 2: Apply PCA with Minka's MLE for data-driven dimensionality selection
            tprint("Step 2: Applying PCA with MLE for data-driven dimensionality selection...", "INFO")
            from sklearn.decomposition import PCA

            # Use PCA with MLE to automatically select number of components
            # Add fallback for small samples or rank-deficient cases
            try:
                pca = PCA(n_components='mle', svd_solver='full')
                features_pca = pca.fit_transform(features_scaled)
                tprint(
                    f"PCA-MLE reduction: {context.original_features.shape[1]} -> {features_pca.shape[1]} features "
                    f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})",
                    "SUCCESS"
                )
            except Exception as e:
                tprint(f"PCA-MLE failed: {e}, using fallback PCA with 99% variance")
                tprint("PCA-MLE failed, using fallback PCA with 99% variance...", "WARNING")
                pca = PCA(n_components=0.99, svd_solver='full')
                features_pca = pca.fit_transform(features_scaled)
                tprint(
                    f"PCA fallback: {context.original_features.shape[1]} -> {features_pca.shape[1]} features "
                    f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})",
                    "SUCCESS"
                )

            # Step 3: Basic quality validation (minimal checks)
            tprint("Step 3: Validating feature quality...", "INFO")
            features_final = self._validate_feature_quality_minimal(features_pca, context.market_data)
            tprint(f"Feature quality validation completed: {features_final.shape}", "SUCCESS")

            tprint(
                f"Data-driven feature optimization completed: {context.original_features.shape} -> {features_final.shape}",
                "SUCCESS"
            )

            context.optimized_features = features_final
            self.optimized_features = features_final

            # Memory cleanup after feature optimization using hardware tools
            # Use thread-safe cleanup to avoid race conditions
            self._safe_memory_cleanup([features_scaled, features_pca])

        except Exception as e:
            tprint(f"Feature optimization failed: {e}", "ERROR")
            # Fast-fail: Do not return original features if optimization fails
            raise ValueError(f"Feature optimization failed: {e}. Cannot proceed with suboptimal features.")

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
                raise ValueError("All features were invalid (NaN/inf), cannot proceed")
            
            # Validate minimum requirements
            if features.shape[1] < 2:
                raise ValueError(f"Insufficient features for clustering: {features.shape[1]} < 2")
            
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

    def _validate_execution_inputs(self, data: Any, pipeline_state: Dict[str, Any]) -> None:
        """Validate inputs for execution method."""
        try:
            # Validate pipeline_state
            if not isinstance(pipeline_state, dict):
                raise ValueError(f"pipeline_state must be a dict, got {type(pipeline_state)}")
            
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

    def _select_optimal_k(self, context: ClusteringContext) -> None:
        """Select the optimal number of clusters using BIC scoring."""
        if context.optimized_features is None:
            raise ValueError("Optimized features are required before selecting optimal K")

        tprint("Step 2: Selecting optimal K using BIC-selected GMM...", "INFO")
        optimal_k, optimal_bic, k_metadata = self._select_optimal_k_bic(context.optimized_features)
        context.optimal_k = optimal_k
        context.optimal_bic = optimal_bic
        context.k_metadata = k_metadata
        tprint(f"BIC-selected K={optimal_k} with BIC={optimal_bic:.3f}", "SUCCESS")

    def _reconcile_labels(self, context: ClusteringContext) -> None:
        """Reconcile NAS/TAS regime assignments and prepare optimization metrics."""
        if context.optimal_k is None:
            raise ValueError("Optimal K must be selected before label reconciliation")

        tprint("Step 3: Extracting TAS and NAS regime assignments...", "INFO")
        tas_assignments, nas_assignments = self._extract_regime_assignments()
        context.tas_assignments = tas_assignments
        context.nas_assignments = nas_assignments
        tprint(
            f"TAS assignments: {len(tas_assignments)}, NAS assignments: {len(nas_assignments)}",
            "SUCCESS"
        )

        tprint("Step 4: Progressive regime optimization with BIC-selected K...", "INFO")
        optimized_assignments, optimization_metrics, fusion_metadata = self.regime_optimization_service.progressive_regime_optimization_with_k(
            context.optimized_features,
            tas_assignments,
            nas_assignments,
            context.market_data,
            context.optimal_k
        )

        context.raw_assignments = optimized_assignments
        context.optimization_metrics = optimization_metrics
        context.fusion_metadata = fusion_metadata or {}

        pre_smoothing_score = optimization_metrics.get('initial_score', 0.0)
        tprint(f"Progressive optimization completed - Score: {pre_smoothing_score:.3f} (pre-smoothing)", "SUCCESS")

    def _smooth_assignments(self, context: ClusteringContext) -> None:
        """Apply temporal smoothing to regime assignments when beneficial."""
        if context.raw_assignments is None:
            raise ValueError("Assignments must be generated before smoothing")

        tprint("Step 5: Applying HMM smoothing for temporal coherence...", "INFO")
        features = context.optimized_features
        base_assignments = context.raw_assignments
        metrics = context.optimization_metrics or {}

        if features is None:
            raise ValueError("Optimized features are required before smoothing")

        initial_score = metrics.get('initial_score')
        if initial_score is None and features is not None:
            initial_score = self._calculate_composite_score(features, base_assignments)
            metrics['initial_score'] = initial_score

        smoothed_assignments, smoothing_metadata = self.regime_optimization_service.apply_hmm_smoothing(
            features, base_assignments
        )
        final_score = self._calculate_composite_score(features, smoothed_assignments)

        if initial_score is None or final_score > initial_score:
            tprint(
                f"HMM smoothing improved score: {initial_score:.3f} → {final_score:.3f}" if initial_score is not None else
                f"HMM smoothing score: {final_score:.3f}",
                "SUCCESS"
            )
            context.smoothed_assignments = smoothed_assignments
            metrics['final_score'] = final_score
            if initial_score is not None:
                metrics['improvement'] = final_score - initial_score
        else:
            tprint("HMM smoothing did not improve, keeping fused assignments", "INFO")
            context.smoothed_assignments = base_assignments
            metrics.setdefault('final_score', initial_score)
            metrics.setdefault('improvement', 0.0)

        metrics.setdefault('iterations', 1)
        metrics.setdefault('method', 'data_driven_optimization')
        metrics.setdefault('fusion_metadata', context.fusion_metadata)
        metrics['hmm_transitions'] = smoothing_metadata.get('transmat', [])
        metrics['smoothing_metadata'] = smoothing_metadata
        context.optimization_metrics = metrics

        tprint(
            f"Progressive optimization completed - Final score: {metrics.get('final_score', final_score):.3f}",
            "SUCCESS"
        )

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
            
            # Calculate consensus score using shared utilities
            consensus_score = 0.0
            try:
                consensus_metrics = self.metrics_calculator.calculate_consensus_metrics(
                    assignments, verbose=False
                )
                consensus_score = consensus_metrics.get('overall_consensus', 0.0)
            except Exception:
                # Fallback to simple consensus calculation
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

    def _calculate_final_quality_metrics(self, features: np.ndarray, assignments: np.ndarray) -> Dict[str, Any]:
        """Calculate final quality metrics for clustering results."""
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

    def _update_learned_weights(
        self,
        clustering_result: Dict[str, Any],
        clustering_metrics: Dict[str, Any],
    ) -> None:
        """Update learned metric weights using latest clustering run outputs."""
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

    def _summarize_results(self, context: ClusteringContext) -> Dict[str, Any]:
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
                'feature_optimization': {
                    'original_features': context.original_features.shape[1],
                    'optimized_features': optimized_features.shape[1],
                    'reduction_ratio': optimized_features.shape[1] / context.original_features.shape[1],
                    'method': 'pca_mle'
                }
            }
        }

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
            max_k = min(40, n_samples // 10, n_features * 2, rank_X - 1, n_samples - 5)  # Evidence-driven cap
            tprint(f"Evidence-driven K search with cap={max_k} (n_samples={n_samples}, n_features={n_features}, rank={rank_X})", "INFO")
            
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
            from sklearn.model_selection import StratifiedKFold
            from sklearn.feature_selection import SelectKBest, mutual_info_classif
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            
            tprint("  🔄 Performing k-fold cross-validation feature selection with M1 hardware optimization...", "INFO")
            
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
            
            # Cross-validation setup
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
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
            
            # Step 2: Find optimal regime mapping using Hungarian algorithm
            regime_mapping = self._find_optimal_regime_mapping(tas_centroids, nas_centroids)
            
            # Step 3: Calculate semantic divergence using mapped regimes
            semantic_assignments = self._apply_regime_mapping(nas_assignments, regime_mapping)
            semantic_disagreement_mask = tas_assignments != semantic_assignments
            semantic_divergence_rate = np.mean(semantic_disagreement_mask)
            
            # Step 4: Calculate confidence scores for divergence detection
            confidence_scores = self._calculate_divergence_confidence_scores(
                features, tas_assignments, nas_assignments, semantic_disagreement_mask
            )
            
            # Step 5: Calculate comprehensive mapping quality metrics
            mapping_quality_metrics = self._calculate_mapping_quality(tas_centroids, nas_centroids, regime_mapping)
            
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
    
    def _calculate_neighborhood_consistency(self, features: np.ndarray, sample_idx: int, tas_regime: int, nas_regime: int) -> float:
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
            # Reduced redundancy by focusing on complementary metrics
            weights = [
                0.20,  # Silhouette improvement (primary cluster separation metric) - reduced from 0.25
                0.00,  # Calinski-Harabasz improvement (REMOVED - redundant with Silhouette)
                0.00,  # Davies-Bouldin improvement (REMOVED - redundant with Silhouette)
                0.20,  # Balance improvement (regime distribution - unique metric)
                0.20,  # Within-regime CV improvement (intra-regime stability - unique metric)
                0.20,  # Between-regime CV improvement (inter-regime divergence - unique metric) - reduced from 0.25
                0.20   # Temporal improvement (smoothness - unique metric) - increased from 0.10
            ]
            
            return weights
            
        except Exception as e:
            tprint(f"Pareto weights calculation failed: {e}")
            return [0.25, 0.00, 0.00, 0.20, 0.20, 0.25, 0.10]  # Default weights
    
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
    
    def _extract_regime_assignments(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract TAS and NAS regime assignments from pipeline state."""
        try:
            pipeline_state = getattr(self, 'pipeline_state', {}) or {}
            if not isinstance(pipeline_state, dict):
                raise ValueError("Pipeline state is missing or invalid")

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
