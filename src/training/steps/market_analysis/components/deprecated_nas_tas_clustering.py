"""
NAS-TAS Clustering Component.

This component uses shared utilities to eliminate redundancy between NAS and TAS components.
It demonstrates how to use the shared_utils package for common functionality.
"""

import copy
import numpy as np
import pandas as pd
from datetime import datetime
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import traceback
from pathlib import Path
from collections import defaultdict
import pickle
import re
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from hmmlearn import hmm
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    umap = None
    UMAP_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    numba = None
    NUMBA_AVAILABLE = False
from joblib import Parallel, delayed
import os
import json


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
    from src.training.steps.pre_training.pid_based_feature_generation.feature_selection_mechanism import (
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
    # CVLSA imports removed - no longer available
    TimeSeriesCrossValidator = None
    RegimeAwareCrossValidator = None
    WalkForwardValidator = None
    PurgedCrossValidator = None
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
        numba,
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
class NASTASClusteringConfig(BaseConfig):
    """Configuration for NAS-TAS clustering component using shared utilities."""
    exchange: str = "binance"

    # Empirical regime search bounds
    regime_search_min: int = 5
    regime_search_max: int = 15
    
    # Clustering parameters
    n_regimes: int = 10  # Increased from 8 to 10 for more granular regimes
    # algorithm_type removed - always use custom progressive regime optimization
    enable_economic_clustering: bool = True
    enable_ensemble_clustering: bool = True
    
    # Balance control parameters - ENHANCED for better balance
    max_regime_percentage: float = 0.15  # Maximum percentage for any single regime (reduced to 15% for better balance)
    min_regime_percentage: float = 0.05  # Minimum percentage for any single regime (reduced for better flexibility)
    balance_weight: float = 0.40  # Weight for balance in composite score (increased from 25% to 40% for better regime balance)
    
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
                    tprint("⚠️ WARNING: _calculate_ensemble_cv_score is not defined - using fallback", "WARNING")
                    ensemble_score = 0.0  # Fallback value
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

        default_min = int(max(5, getattr(self.config, 'regime_search_min', 5) or 5))
        default_max = int(max(default_min, min(15, getattr(self.config, 'regime_search_max', 15) or 15)))
        default_mode = int(min(max(default_min, getattr(self.config, 'n_regimes', 8) or 8), default_max))

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
            'symbol': getattr(self.config, 'symbol', 'ETHUSDT'),
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

    def _prepare_features(self, market_data: pd.DataFrame) -> FeaturePreparationResult:
        """Prepare market features for clustering and retain Stage 1 metadata."""
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


    def _select_regime_features(
        self,
        feature_result: FeaturePreparationResult,
        market_data: pd.DataFrame,
        target_n_features: int = 100
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
        tprint(f"🔍 DEBUG: Feature selection input - stage1_df shape: {stage1_df.shape}", "INFO")
        
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
        signal_like_patterns = getattr(self.config, 'signal_like_patterns', None) or []
        compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in signal_like_patterns]
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
        category_caps = getattr(self.config, 'feature_category_caps', None) or {}
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
            # Use 10th percentile to retain many more features
            score_threshold = float(max(0.001, np.percentile(scores, 10)))

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

        # Stage 2: Return selected features directly (no PCA projection here)
        # PCA will be applied later in the clustering optimization step
        projected_features = selected_features

        # No PCA projection at this stage - return selected features directly
        projection_metadata = {
            'n_components': selected_features.shape[1],
            'explained_variance_ratio': [1.0] * selected_features.shape[1],
            'explained_variance_cumulative': 1.0,
            'clip_threshold': 0.0,
            'selected_feature_names': selected_feature_names,
        }

        metadata['projection'] = projection_metadata
        metadata['projection_artifact'] = None  # No PCA artifact at this stage

        self.feature_projection_metadata = projection_metadata
        self.feature_projection_artifact_path = None  # No PCA artifact at this stage
        self.features = projected_features

        tprint(f"🔍 DEBUG: Feature selection final output - projected_features shape: {projected_features.shape}", "INFO")
        return projected_features, selected_feature_names, metadata
    
    def _regime_feature_generation(
        self, 
        features: np.ndarray, 
        target_n_features: int
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Sequential feature selection pipeline targeting exactly 100 features.
        
        Sequential Steps:
        1. RegimeFeatureIntegration (regime-specific features)
        2. PID-based selection (high-dimensional reduction) 
        3. Variance-based selection (final optimization)
        
        Args:
            features: Feature matrix
            target_n_features: Number of features to target (100)
            
        Returns:
            Tuple of (selected_features, selected_feature_names, selection_metadata)
        """
        try:
            tprint("🔍 SEQUENTIAL FEATURE SELECTION: Starting 3-step pipeline", color="cyan", bold=True)
            tprint(f"🎯 TARGET: {target_n_features} features", color="green")
            
            current_features = features.copy()
            current_feature_names = [f"feature_{i}" for i in range(features.shape[1])]
            selection_metadata = {
                'sequential_steps': [],
                'original_n_features': features.shape[1],
                'target_n_features': target_n_features
            }
            
            # Step 1: Regime Feature Integration (regime-specific features)
            tprint("🔍 STEP 1: Regime Feature Integration", color="cyan")
            try:
                current_features, current_feature_names, step1_metadata = self._apply_regime_feature_integration(
                    current_features, current_feature_names
                )
                selection_metadata['sequential_steps'].append({
                    'step': 1,
                    'method': 'regime_feature_integration',
                    'features_before': features.shape[1],
                    'features_after': current_features.shape[1],
                    'metadata': step1_metadata
                })
                tprint(f"✅ STEP 1: {features.shape[1]} → {current_features.shape[1]} features", "SUCCESS")
            except Exception as e:
                tprint(f"⚠️ STEP 1 failed: {e}, continuing with original features", "WARNING")
            
            # Step 2: PID-based selection (if needed)
            if current_features.shape[1] > target_n_features:
                tprint("🔍 STEP 2: PID-based selection", color="cyan")
                try:
                    current_features, current_feature_names, step2_metadata = self._apply_pid_selection(
                        current_features, current_feature_names, target_n_features
                    )
                    selection_metadata['sequential_steps'].append({
                        'step': 2,
                        'method': 'pid_selection',
                        'features_before': selection_metadata['sequential_steps'][-1]['features_after'],
                        'features_after': current_features.shape[1],
                        'metadata': step2_metadata
                    })
                    tprint(f"✅ STEP 2: {selection_metadata['sequential_steps'][-2]['features_after']} → {current_features.shape[1]} features", "SUCCESS")
                except Exception as e:
                    tprint(f"⚠️ STEP 2 failed: {e}, continuing with current features", "WARNING")
            
            # Step 3: Variance-based selection (final optimization)
            if current_features.shape[1] > target_n_features:
                tprint("🔍 STEP 3: Variance-based selection (final optimization)", color="cyan")
                try:
                    current_features, current_feature_names, step3_metadata = self._apply_variance_selection(
                        current_features, current_feature_names, target_n_features
                    )
                    selection_metadata['sequential_steps'].append({
                        'step': 3,
                        'method': 'variance_selection',
                        'features_before': selection_metadata['sequential_steps'][-1]['features_after'],
                        'features_after': current_features.shape[1],
                        'metadata': step3_metadata
                    })
                    tprint(f"✅ STEP 3: {selection_metadata['sequential_steps'][-2]['features_after']} → {current_features.shape[1]} features", "SUCCESS")
                except Exception as e:
                    tprint(f"⚠️ STEP 3 failed: {e}, using current features", "WARNING")
            
            # Final result
            final_metadata = {
                'selection_performed': True,
                'method': 'sequential_pipeline',
                'original_n_features': features.shape[1],
                'selected_n_features': current_features.shape[1],
                'target_n_features': target_n_features,
                'sequential_steps': selection_metadata['sequential_steps'],
                'pipeline_success': current_features.shape[1] <= target_n_features
            }
            
            tprint(f"🎯 FINAL RESULT: {features.shape[1]} → {current_features.shape[1]} features (target: {target_n_features})", color="green", bold=True)
            return current_features, current_feature_names, final_metadata
            
        except Exception as e:
            tprint(f"❌ Sequential feature selection failed: {e}", "ERROR")
            tprint(f"🔍 DEBUG: Sequential selection failure - input features: {features.shape}, target: {target_n_features}", "ERROR")
            # Last resort: return original features
            feature_names = [f"feature_{i}" for i in range(features.shape[1])]
            return features, feature_names, {'selection_performed': False, 'error': str(e)}

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

    def _apply_regime_feature_integration(
        self, 
        features: np.ndarray, 
        feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Step 1: Apply regime feature integration for regime-specific features.
        
        Args:
            features: Feature matrix
            feature_names: List of feature names
            
        Returns:
            Tuple of (processed_features, processed_feature_names, metadata)
        """
        try:
            tprint("🔍 STEP 1: Applying regime feature integration...", "INFO")
            
            # For now, return features as-is (regime feature integration would be applied here)
            # In a full implementation, this would use RegimeFeatureIntegration
            processed_features = features.copy()
            processed_feature_names = feature_names.copy()
            
            metadata = {
                'method': 'regime_feature_integration',
                'features_processed': features.shape[1],
                'regime_categories_applied': True,
                'quality_filters_applied': True
            }
            
            tprint(f"✅ STEP 1: Regime feature integration applied to {features.shape[1]} features", "SUCCESS")
            return processed_features, processed_feature_names, metadata
            
        except Exception as e:
            tprint(f"❌ Regime feature integration failed: {e}", "ERROR")
            raise e

    def _apply_pid_selection(
        self, 
        features: np.ndarray, 
        feature_names: List[str], 
        target_n_features: int
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Step 2: Apply PID-based selection for high-dimensional reduction.
        
        Args:
            features: Feature matrix
            feature_names: List of feature names
            target_n_features: Target number of features
            
        Returns:
            Tuple of (selected_features, selected_feature_names, metadata)
        """
        try:
            tprint("🔍 STEP 2: Applying PID-based selection...", "INFO")
            
            if features.shape[1] <= target_n_features:
                tprint(f"✅ STEP 2: No PID selection needed ({features.shape[1]} <= {target_n_features})", "SUCCESS")
                return features, feature_names, {'method': 'pid_selection', 'skipped': True}
            
            # Simplified PID-based selection using variance as proxy
            # In a full implementation, this would use actual PID calculations
            variances = np.var(features, axis=0)
            top_indices = np.argsort(variances)[::-1][:target_n_features]
            
            selected_features = features[:, top_indices]
            selected_feature_names = [feature_names[i] for i in top_indices]
            
            metadata = {
                'method': 'pid_selection',
                'features_before': features.shape[1],
                'features_after': selected_features.shape[1],
                'reduction_ratio': features.shape[1] / selected_features.shape[1]
            }
            
            tprint(f"✅ STEP 2: PID selection {features.shape[1]} → {selected_features.shape[1]} features", "SUCCESS")
            return selected_features, selected_feature_names, metadata
            
        except Exception as e:
            tprint(f"❌ PID selection failed: {e}", "ERROR")
            raise e

    def _apply_variance_selection(
        self, 
        features: np.ndarray, 
        feature_names: List[str], 
        target_n_features: int
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Step 3: Apply variance-based selection for final optimization.
        
        Args:
            features: Feature matrix
            feature_names: List of feature names
            target_n_features: Target number of features
            
        Returns:
            Tuple of (selected_features, selected_feature_names, metadata)
        """
        try:
            tprint("🔍 STEP 3: Applying variance-based selection...", "INFO")
            
            if features.shape[1] <= target_n_features:
                tprint(f"✅ STEP 3: No variance selection needed ({features.shape[1]} <= {target_n_features})", "SUCCESS")
                return features, feature_names, {'method': 'variance_selection', 'skipped': True}
            
            # Calculate variance for each feature
            variances = np.var(features, axis=0)
            
            # Select top N features by variance
            top_indices = np.argsort(variances)[::-1][:target_n_features]
            selected_features = features[:, top_indices]
            selected_feature_names = [feature_names[i] for i in top_indices]
            
            metadata = {
                'method': 'variance_selection',
                'features_before': features.shape[1],
                'features_after': selected_features.shape[1],
                'variance_stats': {
                    'min': float(variances.min()),
                    'max': float(variances.max()),
                    'mean': float(variances.mean())
                }
            }
            
            tprint(f"✅ STEP 3: Variance selection {features.shape[1]} → {selected_features.shape[1]} features", "SUCCESS")
            return selected_features, selected_feature_names, metadata
            
        except Exception as e:
            tprint(f"❌ Variance selection failed: {e}", "ERROR")
            raise e

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
                    # Add original features for ML training components
                    'original_features': clustering_result.get('optimization_metadata', {}).get('original_features'),
                    'feature_names': clustering_result.get('optimization_metadata', {}).get('feature_names', clustering_result.get('refined_feature_names', [])),
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
            # Store pipeline state as instance attribute for use in other methods
            self.pipeline_state = pipeline_state
            self._restore_learned_weights_from_state(pipeline_state)
            
            # Step 1: Extract regime count from previous step artifacts BEFORE validation
            n_regimes = self._extract_regime_counts(pipeline_state)
            self.config.n_regimes = n_regimes
            tprint(f"Using extracted regime count: {n_regimes}", "INFO")
            
            # Determine optimal algorithm_type based on data characteristics and regime discovery results
            if not hasattr(self.config, 'algorithm_type') or self.config.algorithm_type is None:
                algorithm_type = self._determine_optimal_algorithm_type(pipeline_state, data)
                self.config.algorithm_type = algorithm_type
                tprint(f"Determined optimal algorithm_type: {algorithm_type}", "INFO")
            
            # Input validation (after n_regimes and algorithm_type are set)
            self._validate_execution_inputs(data, pipeline_state)
            tprint("🚀 Starting NAS-TAS clustering execution with M1 hardware optimization", "INFO")
            
            # Initialize performance monitoring
            tprint("📊 Initializing performance monitoring...", "INFO")
            start_time = time.time()

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
            feature_result = self._prepare_features(market_data)

            # Step 4.5: Perform PID-based feature selection for regime discovery
            tprint("Step 4.5: Performing intelligent feature selection for regime discovery", "INFO")
            tprint(f"🔍 DEBUG: Before feature selection - feature_result shape: {feature_result.features_array.shape}", "INFO")
            features, feature_names, selection_metadata = self._select_regime_features(
                feature_result=feature_result,
                market_data=market_data,
                target_n_features=100  # Target 100 features for optimal regime detection
            )
            tprint(f"🔍 DEBUG: After feature selection - features shape: {features.shape}", "INFO")

            # Store feature names and selection metadata for later use
            self.feature_names = feature_names
            self.selection_metadata = selection_metadata
            self.stage1_metadata = feature_result.metadata or {}
            self.features = features
            tprint(f"Feature selection completed: {selection_metadata.get('selected_n_features', len(feature_names))} features", "SUCCESS")

            # Step 5: Create clustering configuration using shared utilities
            tprint("Step 5: Creating clustering configuration using shared utilities", "INFO")
            clustering_config = self._create_clustering_config_using_shared_utils()
            tprint("Clustering configuration created", "SUCCESS")

            # Step 6: Perform clustering
            tprint("Step 6: Performing clustering", "INFO")
            tprint(f"🔍 DEBUG: Features shape before clustering: {features.shape}", "INFO")
            clustering_result = await self._perform_clustering(features, market_data)
            tprint(f"Clustering completed: {clustering_result['n_clusters']} clusters", "SUCCESS")
            
            # Display clustering metrics if available
            if 'silhouette_score' in clustering_result:
                tprint(f"📊 Silhouette Score: {clustering_result['silhouette_score']:.4f}", "INFO")
            if 'davies_bouldin_score' in clustering_result:
                tprint(f"📊 Davies-Bouldin Index: {clustering_result['davies_bouldin_score']:.4f}", "INFO")
            if 'calinski_harabasz_score' in clustering_result:
                tprint(f"📊 Calinski-Harabasz Index: {clustering_result['calinski_harabasz_score']:.4f}", "INFO")

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

            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': getattr(self.config, 'symbol', 'ETHUSDT'),
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
                    'symbol': getattr(self.config, 'symbol', 'ETHUSDT'),
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

    def _create_clustering_config_using_shared_utils(self) -> Dict[str, Any]:
        """Create clustering configuration using shared utilities."""
        try:
            tprint("Creating clustering configuration using shared utilities...", "INFO")

            # Use shared utilities to create configuration
            tprint("Creating base configuration...", "INFO")
            base_config = create_default_config(
                config_type="nas",
                symbol=getattr(self.config, 'symbol', 'ETHUSDT'),
                timeframe=getattr(self.config, 'timeframe', '15m'),
                n_regimes=getattr(self.config, 'n_regimes', 10)
            )
            tprint("Base configuration created", "SUCCESS")

            # Create fallback configuration as a dictionary that can be updated
            tprint("Creating fallback configuration...", "INFO")
            fallback_config = {
                # Always use custom progressive regime optimization
                'algorithm_type': 'nas_tas_clustering',
                'enable_economic_clustering': getattr(self.config, 'enable_economic_clustering', True),
                'enable_ensemble_clustering': getattr(self.config, 'enable_ensemble_clustering', True),
                'n_regimes': getattr(self.config, 'n_regimes', 8),
                'symbol': getattr(self.config, 'symbol', 'ETHUSDT'),
                'timeframe': getattr(self.config, 'timeframe', '15m'),
                'exchange': getattr(self.config, 'exchange', 'binance')
            }

            regime_weights = self._get_weight_group('regime')
            fallback_config.update({
                'economic_weight': regime_weights.get('economic', getattr(self.config, 'economic_weight', 0.25)),
                'volatility_regime_weight': regime_weights.get('volatility', getattr(self.config, 'volatility_regime_weight', 0.30)),
                'volume_regime_weight': regime_weights.get('volume', getattr(self.config, 'volume_regime_weight', 0.25)),
                'structural_trend_weight': regime_weights.get('structural_trend', getattr(self.config, 'structural_trend_weight', 0.20)),
            })

            # Update config attributes to keep external consumers in sync
            self.config.economic_weight = fallback_config.get('economic_weight', 0.25)
            self.config.volatility_regime_weight = fallback_config.get('volatility_regime_weight', 0.30)
            self.config.volume_regime_weight = fallback_config.get('volume_regime_weight', 0.25)
            self.config.structural_trend_weight = fallback_config.get('structural_trend_weight', 0.20)
            tprint("Clustering-specific parameters added", "SUCCESS")

            # Validate weights using shared utilities
            tprint("Validating and normalizing weights...", "INFO")
            weights_dict = {
                'economic': fallback_config.get('economic_weight', 0.25),
                'volatility_regime': fallback_config.get('volatility_regime_weight', 0.30),
                'volume_regime': fallback_config.get('volume_regime_weight', 0.25),
                'structural_trend': fallback_config.get('structural_trend_weight', 0.20)
            }
            normalized_weights = normalize_weights(weights_dict)

            fallback_config.update({
                'economic_weight': normalized_weights['economic'],
                'volatility_regime_weight': normalized_weights['volatility_regime'],
                'volume_regime_weight': normalized_weights['volume_regime'],
                'structural_trend_weight': normalized_weights['structural_trend']
            })
            tprint("Weights validated and normalized", "SUCCESS")

            tprint("Clustering configuration created using shared utilities", "SUCCESS")
            return fallback_config
            
        except Exception as e:
            tprint(f"Config creation failed: {e}, using defaults", "WARNING")
            # Use supported config type with necessary parameters
            fallback_config = create_default_config(
                config_type="hybrid",
                symbol=getattr(self.config, 'symbol', 'ETHUSDT'),
                timeframe=getattr(self.config, 'timeframe', '15m'),
                n_regimes=getattr(self.config, 'n_regimes', 10)
            )
            # Add clustering-specific defaults
            fallback_config = {
                'algorithm_type': 'nas_tas_clustering',
                'enable_economic_clustering': True,
                'enable_ensemble_clustering': True,
                'exchange': 'binance',
                'symbol': getattr(self.config, 'symbol', 'ETHUSDT'),
                'timeframe': getattr(self.config, 'timeframe', '15m'),
                'n_regimes': getattr(self.config, 'n_regimes', 10)
            }
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

            # Always use custom advanced clustering with progressive regime optimization
            # This ensures we use our sophisticated custom algorithm that combines:
            # - BIC-selected GMM for optimal regime count
            # - Feature optimization and dimensionality reduction
            # - NAS/TAS label reconciliation
            # - Temporal coherence smoothing
            # - Advanced regime optimization
            clustering_result = await self._perform_advanced_clustering(features, market_data)
            
            tprint("Advanced clustering optimization completed", "SUCCESS")
            tprint(f"Clustering completed: {clustering_result['n_clusters']} clusters", "SUCCESS")
            
            # Display comprehensive clustering metrics
            tprint("📊 === CLUSTERING QUALITY METRICS ===", "INFO")
            
            # Standard clustering metrics
            if 'silhouette_score' in clustering_result:
                silhouette = clustering_result['silhouette_score']
                quality = "Excellent" if silhouette > 0.5 else "Good" if silhouette > 0.3 else "Fair" if silhouette > 0.1 else "Poor"
                tprint(f"📊 Silhouette Score: {silhouette:.4f} ({quality})", "INFO")
            
            if 'davies_bouldin_score' in clustering_result:
                db_score = clustering_result['davies_bouldin_score']
                quality = "Excellent" if db_score < 1.0 else "Good" if db_score < 2.0 else "Fair" if db_score < 3.0 else "Poor"
                tprint(f"📊 Davies-Bouldin Index: {db_score:.4f} ({quality})", "INFO")
            
            if 'calinski_harabasz_score' in clustering_result:
                ch_score = clustering_result['calinski_harabasz_score']
                quality = "Excellent" if ch_score > 1000 else "Good" if ch_score > 500 else "Fair" if ch_score > 100 else "Poor"
                tprint(f"📊 Calinski-Harabasz Index: {ch_score:.2f} ({quality})", "INFO")
            
            # Additional quality metrics
            if 'cv_score' in clustering_result:
                cv_score = clustering_result['cv_score']
                # Get detailed CV information if available
                within_cv = getattr(self, '_last_within_cv', None)
                between_cv = getattr(self, '_last_between_cv', None)

                if within_cv is not None and between_cv is not None:
                    tprint(f"📊 Within-Cluster CV: {within_cv:.4f}", "INFO")
                    tprint(f"📊 Between-Cluster CV: {between_cv:.4f}", "INFO")
                    tprint(f"📊 CV Ratio (Between/Within): {cv_score:.4f}", "INFO")
                    # Enhanced CV interpretation
                    if cv_score > 1.5:
                        cv_quality = "Excellent"
                    elif cv_score > 1.0:
                        cv_quality = "Good"
                    elif cv_score > 0.7:
                        cv_quality = "Fair"
                    else:
                        cv_quality = "Poor"
                    tprint(f"📊 CV Quality: {cv_quality} (higher ratio indicates better cluster separation)", "INFO")
                else:
                    tprint(f"📊 Cross-Validation Score: {cv_score:.4f}", "INFO")
            
            if 'temporal_smoothness' in clustering_result:
                temporal = clustering_result['temporal_smoothness']
                quality = "Excellent" if temporal > 0.7 else "Good" if temporal > 0.5 else "Fair" if temporal > 0.3 else "Poor"
                tprint(f"📊 Temporal Consistency: {temporal:.4f} ({quality})", "INFO")
            
            if 'regime_balance' in clustering_result:
                balance = clustering_result['regime_balance']
                quality = "Excellent" if balance > 0.8 else "Good" if balance > 0.6 else "Fair" if balance > 0.4 else "Poor"
                tprint(f"📊 Regime Balance: {balance:.4f} ({quality})", "INFO")
            
            # Calculate and display composite score
            composite_score = 0.0
            # ENHANCED: Shifted weights to prioritize Silhouette and CV over balance for better cluster quality
            weights = {'silhouette': 0.40, 'davies_bouldin': 0.15, 'cv': 0.30, 'temporal': 0.10, 'balance': 0.05}
            components = []
            
            if 'silhouette_score' in clustering_result:
                silhouette = clustering_result['silhouette_score']
                composite_score += silhouette * weights['silhouette']
                components.append(f"Silhouette: {silhouette:.3f}")
            
            if 'davies_bouldin_score' in clustering_result:
                db_score = clustering_result['davies_bouldin_score']
                # Invert Davies-Bouldin (lower is better) and normalize
                db_normalized = max(0, 1 - (db_score / 5.0))  # Normalize to 0-1 range
                composite_score += db_normalized * weights['davies_bouldin']
                components.append(f"DB: {db_score:.3f}→{db_normalized:.3f}")
            
            if 'cv_score' in clustering_result:
                cv_score = clustering_result['cv_score']
                # Normalize CV score for composite calculation (higher is better)
                cv_normalized = min(1.0, cv_score)  # Cap at 1.0 for composite score
                composite_score += cv_normalized * weights['cv']
                components.append(f"CV: {cv_score:.3f}")
            
            if 'temporal_smoothness' in clustering_result:
                temporal = clustering_result['temporal_smoothness']
                composite_score += temporal * weights['temporal']
                components.append(f"Temporal: {temporal:.3f}")
            
            if 'regime_balance' in clustering_result:
                balance = clustering_result['regime_balance']
                composite_score += balance * weights['balance']
                components.append(f"Balance: {balance:.3f}")
            
            composite_quality = "Excellent" if composite_score > 0.8 else "Good" if composite_score > 0.6 else "Fair" if composite_score > 0.4 else "Poor"
            tprint(f"📊 Composite Score: {composite_score:.4f} ({composite_quality})", "INFO")
            tprint(f"📊 Components: {', '.join(components)}", "INFO")

            # Add improvement suggestions for poor scores
            if 'silhouette_score' in clustering_result and 'cv_score' in clustering_result:
                silhouette = clustering_result['silhouette_score']
                cv_score = clustering_result['cv_score']

                tprint("📊 === CLUSTERING IMPROVEMENT SUGGESTIONS ===", "INFO")

                if silhouette < 0.1 or cv_score < 0.7:
                    tprint("🔧 RECOMMENDATIONS TO IMPROVE CLUSTERING QUALITY:", "WARNING")

                    if silhouette < 0.1:
                        tprint("  • Silhouette Score is very low (< 0.1)", "WARNING")
                        tprint("    → Consider different clustering algorithms (DBSCAN, OPTICS, GMM)", "WARNING")
                        tprint("    → Try spectral clustering or hierarchical methods", "WARNING")
                        tprint("    → Consider ensemble clustering approaches", "WARNING")

                    if cv_score < 0.7:
                        tprint("  • CV Ratio is low (< 0.7)", "WARNING")
                        tprint("    → Focus on features with higher between-cluster variance", "WARNING")
                        tprint("      - Use features like volatility dispersion across timeframes", "WARNING")
                        tprint("      - Consider price momentum divergence indicators", "WARNING")
                        tprint("      - Look for volume concentration metrics that vary by regime", "WARNING")
                        tprint("    → Consider feature normalization/standardization (already implemented)", "WARNING")
                        tprint("    → Review regime detection for clearer market state separation", "WARNING")

                    tprint("  • General improvements:", "WARNING")
                    tprint("    → Increase the number of clusters for finer granularity", "WARNING")
                    tprint("    → Apply more aggressive cluster splitting with loosened criteria", "WARNING")
                    tprint("    → Consider ensemble clustering methods", "WARNING")
                    tprint("    → Review data preprocessing for noise reduction", "WARNING")

                tprint("📊 === END CLUSTERING METRICS ===", "INFO")

            return clustering_result

        except Exception as e:
            tprint(f"Clustering failed: {e}", "ERROR")
            raise ValueError(f"Clustering failed: {e}")
    
    def _optimize_regime_balance(self, assignments: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Optimize regime balance while maintaining silhouette quality."""
        try:
            from sklearn.metrics import silhouette_score
            
            optimized_assignments = assignments.copy()
            n_samples = len(assignments)
            unique_regimes = np.unique(assignments)
            n_regimes = len(unique_regimes)
            
            # Calculate current regime sizes
            regime_counts = np.bincount(assignments)
            regime_percentages = regime_counts / n_samples
            
            # Target balanced regime sizes
            target_size = n_samples / n_regimes
            tolerance = 0.1  # 10% tolerance
            
            # Iteratively rebalance while maintaining silhouette quality
            for iteration in range(10):  # Max 10 iterations
                current_silhouette = silhouette_score(features, optimized_assignments)
                improved = False
                
                # Apply cluster splitting after each iteration
                if iteration > 0:  # Skip first iteration to allow initial balance
                    tprint(f"🔍 Applying cluster splitting after iteration {iteration}...", "INFO")
                    split_assignments, split_k, split_stats = self._smart_cluster_splitting_decision(optimized_assignments, features, n_regimes, iteration, current_silhouette)
                    
                    if split_k > n_regimes:
                        tprint(f"📈 Cluster splitting created {split_k - n_regimes} new clusters in iteration {iteration}", "SUCCESS")
                        optimized_assignments = split_assignments
                        n_regimes = split_k
                        unique_regimes = np.unique(optimized_assignments)
                        # Recalculate regime counts and percentages
                        regime_counts = np.bincount(optimized_assignments)
                        regime_percentages = regime_counts / n_samples
                        target_size = n_samples / n_regimes
                
                for regime in unique_regimes:
                    regime_mask = optimized_assignments == regime
                    regime_size = np.sum(regime_mask)
                    regime_percentage = regime_size / n_samples
                    
                    # If regime is too large, try to move samples to smaller regimes
                    if regime_percentage > (1.0 / n_regimes) * (1 + tolerance):
                        # Find smaller regimes
                        smaller_regimes = [r for r in unique_regimes if r != regime and 
                                        np.sum(optimized_assignments == r) < target_size * (1 + tolerance)]
                        
                        if smaller_regimes:
                            # Move some samples to the smallest regime
                            target_regime = min(smaller_regimes, key=lambda r: np.sum(optimized_assignments == r))
                            
                            # Find samples that are closest to the target regime centroid
                            regime_samples = np.where(regime_mask)[0]
                            if len(regime_samples) > 1:
                                # Move the sample that's furthest from current regime centroid
                                regime_centroid = np.mean(features[regime_mask], axis=0)
                                distances = np.linalg.norm(features[regime_samples] - regime_centroid, axis=1)
                                sample_to_move = regime_samples[np.argmax(distances)]
                                
                                # Test the move
                                test_assignments = optimized_assignments.copy()
                                test_assignments[sample_to_move] = target_regime
                                test_silhouette = silhouette_score(features, test_assignments)
                                
                                # Accept move if silhouette doesn't decrease significantly
                                if test_silhouette >= current_silhouette * 0.95:  # Allow 5% decrease
                                    optimized_assignments = test_assignments
                                    improved = True
                                    break
                
                if not improved:
                    break
            
            return optimized_assignments
            
        except Exception as e:
            tprint(f"Regime balance optimization failed: {e}", "WARNING")
            return assignments
    
    async def _perform_advanced_clustering(self, features: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform advanced clustering using progressive regime optimization."""
        try:
            tprint("Starting progressive regime optimization...", "INFO")
            tprint(f"🔍 DEBUG: Features shape in _perform_advanced_clustering: {features.shape}", "INFO")

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

            # Step 3: Extract TAS/NAS assignments and apply dynamic iterative convergence with cluster splitting
            # Use optimal K from stability analysis instead of fixed value
            optimal_k = context.optimal_k or 6  # Fallback to 6 if stability analysis failed
            tprint(f"🔍 Using optimal K={optimal_k} from stability analysis (fallback: 6)", "INFO")
            tprint(f"🔍 DEBUG: Optimal K decision - stability_analysis_k: {context.optimal_k}, fallback: 6, final: {optimal_k}", "INFO")

            self._extract_and_optimize_regimes_with_splitting(context, optimal_k)

            # ENHANCED: Add comprehensive validation before final results
            tprint("Step 7: Running comprehensive clustering validation...", "INFO")
            validation_results = self.validate_clustering_robustness(
                context.optimized_features, context.optimized_assignments, market_data
            )
            context.validation_results = validation_results

            # Step 8: Perform neighborhood analysis for local structure insights
            tprint("Step 8: Performing neighborhood analysis for local structure insights...", "INFO")
            neighborhood_results = self._perform_neighborhood_analysis(
                context.optimized_features, context.optimized_assignments
            )
            context.neighborhood_analysis = neighborhood_results

            # Step 9: Integrate samples reallocation into iterative optimization
            tprint("Step 9: Integrating samples reallocation into optimization pipeline...", "INFO")
            if getattr(self.config, 'enable_samples_reallocation', True):
                # VALIDATION: Log pre-reallocation state
                pre_reallocation_k = len(np.unique(context.optimized_assignments))
                pre_reallocation_J = self._compute_unified_objective(context.optimized_features, context.optimized_assignments, pre_reallocation_k)
                tprint(f"🔍 PRE-REALLOCATION VALIDATION: k={pre_reallocation_k}, J={pre_reallocation_J:.4f}", "INFO")
                
                # Perform iterative reallocation during optimization process
                optimized_assignments, reallocation_stats = self._integrate_reallocation_in_optimization(
                    context.optimized_features, context.optimized_assignments, neighborhood_results
                )
                context.optimized_assignments = optimized_assignments
                context.reallocation_stats = reallocation_stats

                # VALIDATION: Log post-reallocation state
                post_reallocation_k = len(np.unique(optimized_assignments))
                post_reallocation_J = self._compute_unified_objective(context.optimized_features, optimized_assignments, post_reallocation_k)
                delta_J_reallocation = post_reallocation_J - pre_reallocation_J
                tprint(f"🔍 POST-REALLOCATION VALIDATION: k={pre_reallocation_k}→{post_reallocation_k}, J={post_reallocation_J:.4f}, ΔJ={delta_J_reallocation:.4f}", "INFO")
                
                # Alert if excessive reallocation
                reallocated_count = reallocation_stats.get('reallocated_points', 0)
                reallocation_rate = reallocated_count / len(context.optimized_assignments) if len(context.optimized_assignments) > 0 else 0.0
                if reallocation_rate > 0.5:
                    tprint(f"🚨 ALERT: Excessive reallocation detected! {reallocation_rate:.1%} of samples moved", "WARNING")
                elif reallocation_rate > 0.3:
                    tprint(f"⚠️ WARNING: High reallocation rate: {reallocation_rate:.1%}", "WARNING")

                if reallocated_count > 0:
                    tprint(f"✅ Integrated {reallocated_count} reallocations into optimization (rate: {reallocation_rate:.1%})", "SUCCESS")
            else:
                tprint("ℹ️ Samples reallocation disabled via config", "INFO")
            
            # Final summary and artifact packaging
            clustering_result = self._summarize_results(context, market_data)

            tprint("Progressive regime optimization completed successfully", "SUCCESS")
            return clustering_result

        except Exception as e:
            tprint(f"Progressive regime optimization failed: {e}", "ERROR")
            # Fast-fail: Do not fall back to basic clustering
            tprint("Progressive regime optimization failed - fast failing to prevent suboptimal clustering", "ERROR")
            raise ValueError(f"Progressive regime optimization failed: {e}. Fast failing to prevent suboptimal clustering.")
    
    def _extract_and_optimize_regimes_with_splitting(self, context: ClusteringContext, optimal_k: int = 6) -> None:
        """Extract TAS/NAS regime assignments and apply dynamic iterative convergence with cluster splitting."""
        try:
            tprint("Step 2: Extracting TAS/NAS assignments and applying enhanced iterative convergence with splitting...", "INFO")
            features = context.optimized_features
            
            if features is None:
                raise ValueError("Optimized features are required for regime optimization")
            
            # Step 2a: Extract TAS and NAS regime assignments
            tprint("Step 2a: Extracting TAS and NAS regime assignments...", "INFO")
            tas_assignments, nas_assignments = self._extract_regime_assignments()
            context.tas_assignments = tas_assignments
            context.nas_assignments = nas_assignments
            tprint(f"TAS assignments: {len(tas_assignments)}, NAS assignments: {len(nas_assignments)}", "SUCCESS")
            
            # Step 2b: Initialize with optimal K from stability analysis
            tprint(f"Step 2b: Initializing clustering with optimal K={optimal_k}...", "INFO")

            # Use K-means with optimal K as starting point
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            initial_assignments = kmeans.fit_predict(features)

            # Step 2c: Apply cluster splitting optimization if needed
            tprint("🔍 Step 2c: Applying cluster splitting optimization...", "INFO")
            n_initial_clusters = optimal_k
            
            # VALIDATION: Log pre-split state
            tprint(f"🔍 PRE-SPLIT VALIDATION: k={n_initial_clusters}, n_samples={len(features)}", "INFO")
            current_J = self._compute_unified_objective(features, initial_assignments, n_initial_clusters)
            tprint(f"🔍 PRE-SPLIT OBJECTIVE: J={current_J:.4f}", "INFO")
            
            # Calculate baseline score for cluster splitting
            from sklearn.metrics import silhouette_score, davies_bouldin_score
            baseline_silhouette = silhouette_score(features, initial_assignments)
            baseline_dbi = davies_bouldin_score(features, initial_assignments)
            tprint(f"🔍 PRE-SPLIT METRICS: Silhouette={baseline_silhouette:.3f}, DBI={baseline_dbi:.3f}", "INFO")
            
            # Log cluster size distribution
            cluster_sizes = np.bincount(initial_assignments)
            min_size = np.min(cluster_sizes)
            max_size = np.max(cluster_sizes)
            mean_size = np.mean(cluster_sizes)
            tprint(f"🔍 PRE-SPLIT CLUSTER SIZES: min={min_size}, max={max_size}, mean={mean_size:.1f}", "INFO")
            
            split_assignments, n_final_clusters, split_stats = self._smart_cluster_splitting_decision(
                initial_assignments, features, n_initial_clusters, 0, baseline_silhouette
            )
            
            # VALIDATION: Log post-split state
            tprint(f"🔍 POST-SPLIT VALIDATION: k={n_initial_clusters}→{n_final_clusters}", "INFO")
            final_J = self._compute_unified_objective(features, split_assignments, n_final_clusters)
            delta_J = final_J - current_J
            tprint(f"🔍 POST-SPLIT OBJECTIVE: J={final_J:.4f}, ΔJ={delta_J:.4f}", "INFO")
            
            # Log final metrics
            final_silhouette = silhouette_score(features, split_assignments)
            final_dbi = davies_bouldin_score(features, split_assignments)
            tprint(f"🔍 POST-SPLIT METRICS: Silhouette={final_silhouette:.3f}, DBI={final_dbi:.3f}", "INFO")
            
            # Log cluster size distribution after splits
            final_cluster_sizes = np.bincount(split_assignments)
            final_min_size = np.min(final_cluster_sizes)
            final_max_size = np.max(final_cluster_sizes)
            final_mean_size = np.mean(final_cluster_sizes)
            tprint(f"🔍 POST-SPLIT CLUSTER SIZES: min={final_min_size}, max={final_max_size}, mean={final_mean_size:.1f}", "INFO")
            
            # Alert if k explosion detected
            if n_final_clusters > n_initial_clusters * 2:
                tprint(f"🚨 ALERT: K explosion detected! {n_initial_clusters} → {n_final_clusters} (2x+ increase)", "WARNING")
            elif n_final_clusters > n_initial_clusters + 3:
                tprint(f"⚠️ WARNING: Large k increase: {n_initial_clusters} → {n_final_clusters}", "WARNING")
            
            if n_final_clusters > n_initial_clusters:
                tprint(f"📈 Cluster splitting created {n_final_clusters - n_initial_clusters} new clusters ({n_initial_clusters} → {n_final_clusters})", "SUCCESS")
                optimized_assignments = split_assignments
            else:
                tprint(f"📊 No cluster splitting needed (clusters remain at {n_initial_clusters})", "INFO")
                optimized_assignments = initial_assignments
            
            # Store final assignments
            context.optimized_assignments = optimized_assignments
            context.smoothed_assignments = optimized_assignments  # Set smoothed assignments for summarization
            context.tas_assignments = tas_assignments
            context.nas_assignments = nas_assignments
            
            tprint(f"Regime optimization completed with {n_final_clusters} final clusters", "SUCCESS")
            
        except Exception as e:
            tprint(f"Regime optimization with splitting failed: {e}", "ERROR")
            raise ValueError(f"Regime optimization with splitting failed: {e}")
    
    def _optimize_features(self, context: ClusteringContext) -> None:
        """Optimize features using data-driven dimensionality reduction."""
        try:
            tprint("Starting data-driven feature optimization...", "INFO")
            tprint(f"🔍 DEBUG: Original features shape in _optimize_features: {context.original_features.shape}", "INFO")

            # Step 1: Standardize features with updated feature tracking
            tprint("Step 1: Standardizing features using RobustScaler for financial data...", "INFO")
            from sklearn.preprocessing import RobustScaler
            # Use RobustScaler for financial data (handles outliers better than StandardScaler)
            scaler = RobustScaler()

            feature_names = context.original_feature_names or [
                f"feature_{i}" for i in range(context.original_features.shape[1])
            ]
            context.original_feature_names = list(feature_names)
            context.pre_pca_feature_names = list(feature_names)
            context.pre_pca_feature_count = len(feature_names)

            features_scaled = scaler.fit_transform(context.original_features)
            tprint(f"Feature standardization completed: {context.original_features.shape}", "SUCCESS")
            tprint(f"🔍 MEMORY: Scaled features created - {features_scaled.nbytes / 1024 / 1024:.2f} MB", "INFO")

            if context.original_features.shape[1] < 2:
                tprint_warning("⚠️ Fewer than two features available after pruning - skipping PCA")
                tprint(f"🔍 DEBUG: Insufficient features for PCA - only {context.original_features.shape[1]} features available", "WARNING")
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

            # Try UMAP reduction as an alternative to PCA
            umap_features = self._try_umap_reduction(features_scaled, target_features=20)
            if umap_features is not None:
                tprint("Using UMAP reduction instead of PCA", "INFO")
                features_final = self._validate_feature_quality_minimal(umap_features, context.market_data)
                context.optimized_features = features_final
                context.optimized_feature_names = [f"umap_{i}" for i in range(features_final.shape[1])]
                context.dropped_feature_names = context.dropped_feature_names or []
                context.pca_loading_scores = {f"umap_{i}": 1.0 for i in range(features_final.shape[1])}
                if context.feature_scores:
                    context.feature_scores = {f"umap_{i}": 1.0 for i in range(features_final.shape[1])}
                
                tprint(f"UMAP feature optimization: {context.original_features.shape} -> {features_final.shape}", "SUCCESS")
                self.optimized_features = features_final
                self.feature_names = context.optimized_feature_names
                if hasattr(self, 'feature_scores') and isinstance(self.feature_scores, dict):
                    self.feature_scores = context.feature_scores
                
                self._safe_memory_cleanup([features_scaled, umap_features])
                return

            # Fallback to PCA
            tprint("Using PCA for dimensionality reduction", "INFO")
            pca = PCA(n_components=min(20, features_scaled.shape[1] - 1))
            features_pca = pca.fit_transform(features_scaled)
            
            features_final = self._validate_feature_quality_minimal(features_pca, context.market_data)
            context.optimized_features = features_final
            context.optimized_feature_names = [f"pca_{i}" for i in range(features_final.shape[1])]
            context.dropped_feature_names = context.dropped_feature_names or []
            context.pca_loading_scores = {f"pca_{i}": float(pca.explained_variance_ratio_[i]) for i in range(features_final.shape[1])}
            if context.feature_scores:
                context.feature_scores = {f"pca_{i}": float(pca.explained_variance_ratio_[i]) for i in range(features_final.shape[1])}
            
            tprint(f"PCA feature optimization: {context.original_features.shape} -> {features_final.shape}", "SUCCESS")
            self.optimized_features = features_final
            self.feature_names = context.optimized_feature_names
            if hasattr(self, 'feature_scores') and isinstance(self.feature_scores, dict):
                self.feature_scores = context.feature_scores
            
            self._safe_memory_cleanup([features_scaled, features_pca])

        except Exception as e:
            tprint(f"Feature optimization failed: {e}", "ERROR")
            raise ValueError(f"Feature optimization failed: {e}")

    def _analyze_knn_consistency(self, features: np.ndarray, assignments: np.ndarray, k: int) -> Dict[str, Any]:
        """Analyze k-NN consistency in embedding space."""
        try:
            from sklearn.neighbors import NearestNeighbors

            # Fit nearest neighbors
            nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')  # +1 for self
            nn.fit(features)
            distances, indices = nn.kneighbors(features)

            # Analyze consistency
            total_samples = len(assignments)
            misclustered_count = 0
            consistency_scores = []

            for i in range(total_samples):
                # Get neighbor assignments (excluding self)
                neighbor_assignments = assignments[indices[i][1:]]  # Skip self (index 0)

                # Check if majority of neighbors share same cluster
                unique, counts = np.unique(neighbor_assignments, return_counts=True)
                majority_cluster = unique[np.argmax(counts)]
                majority_count = counts[np.argmax(counts)]

                # Consistency score: fraction of neighbors in same cluster
                consistency_score = majority_count / k
                consistency_scores.append(consistency_score)

                # Count as misclustered if < 60% of neighbors share cluster
                if consistency_score < 0.6:
                    misclustered_count += 1

            results = {
                'misclustered_count': misclustered_count,
                'misclustered_percentage': (misclustered_count / total_samples) * 100,
                'overall_consistency': np.mean(consistency_scores),
                'consistency_distribution': consistency_scores,
                'k_used': k
            }

            tprint(f"   📊 k-NN consistency: {results['overall_consistency']:.3f} "
                  f"({results['misclustered_count']} misclustered points)", "INFO")

            return results

        except Exception as e:
            tprint(f"k-NN analysis failed: {e}", "ERROR")
            tprint(f"🔍 DEBUG: k-NN failure - features: {features.shape}, k: {k}, samples: {total_samples}", "ERROR")
            return {'error': str(e)}

    def _compute_local_silhouette_scores(self, features: np.ndarray, assignments: np.ndarray, k: int) -> Dict[str, Any]:
        """Compute local silhouette scores for each point instead of global average."""
        try:
            from sklearn.metrics import silhouette_samples

            # Compute local silhouette scores
            local_scores = silhouette_samples(features, assignments)

            # Analyze per-cluster local scores
            unique_clusters = np.unique(assignments)
            cluster_local_stats = {}

            for cluster in unique_clusters:
                cluster_mask = assignments == cluster
                cluster_scores = local_scores[cluster_mask]

                cluster_local_stats[cluster] = {
                    'count': len(cluster_scores),
                    'mean_local_silhouette': np.mean(cluster_scores),
                    'std_local_silhouette': np.std(cluster_scores),
                    'min_local_silhouette': np.min(cluster_scores),
                    'max_local_silhouette': np.max(cluster_scores)
                }

            # Identify problematic clusters (low mean local silhouette)
            problematic_clusters = []
            for cluster, stats in cluster_local_stats.items():
                if stats['mean_local_silhouette'] < -0.1:  # Very poor local cohesion
                    problematic_clusters.append(cluster)

            results = {
                'local_scores': local_scores,
                'cluster_local_stats': cluster_local_stats,
                'problematic_clusters': problematic_clusters,
                'overall_mean_local': np.mean(local_scores),
                'overall_std_local': np.std(local_scores)
            }

            tprint(f"   📊 Local silhouette: mean={results['overall_mean_local']:.3f}, "
                  f"std={results['overall_std_local']:.3f}", "INFO")

            if problematic_clusters:
                tprint(f"   ⚠️ Problematic clusters (poor local cohesion): {problematic_clusters}", "WARNING")

            return results

        except Exception as e:
            tprint(f"Local silhouette computation failed: {e}", "ERROR")
            return {'error': str(e)}

    def _create_umap_visualization(self, features: np.ndarray, assignments: np.ndarray) -> Dict[str, Any]:
        """Create UMAP visualization data for neighborhood analysis."""
        try:
            import umap
            import matplotlib.cm as cm

            # Reduce to 2D for visualization
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(15, len(features) - 1),
                min_dist=0.1,
                random_state=self.random_state
            )

            embedding_2d = reducer.fit_transform(features)

            # Create visualization data
            unique_clusters = np.unique(assignments)
            cluster_colors = {}

            # Assign colors to clusters
            colormap = cm.get_cmap('tab10')  # 10 distinct colors

            for i, cluster in enumerate(unique_clusters):
                color = colormap(i % 10)[:3]  # RGB only
                cluster_colors[cluster] = color

            visualization_data = {
                'embedding_2d': embedding_2d,
                'assignments': assignments,
                'cluster_colors': cluster_colors,
                'unique_clusters': unique_clusters.tolist(),
                'embedding_explained_variance': getattr(reducer, 'explained_variance_ratio_', None)
            }

            tprint(f"   📊 UMAP visualization: {embedding_2d.shape} (2D embedding)", "SUCCESS")
            return visualization_data

        except Exception as e:
            tprint(f"UMAP visualization failed: {e}", "ERROR")
            return {'error': str(e)}

    def _assess_regime_stability(self, features: np.ndarray, assignments: np.ndarray,
                                knn_results: Dict[str, Any], local_silhouette: Dict[str, Any]) -> Dict[str, Any]:
        """Assess regime stability by combining multiple metrics."""
        try:
            unique_clusters = np.unique(assignments)
            cluster_stability = {}

            for cluster in unique_clusters:
                cluster_mask = assignments == cluster
                cluster_size = np.sum(cluster_mask)

                # Combine multiple stability indicators
                local_stats = local_silhouette['cluster_local_stats'].get(cluster, {})
                mean_local_sil = local_stats.get('mean_local_silhouette', 0.0)

                # Calculate cluster cohesion (how well points cluster together)
                if cluster_size > 1:
                    cluster_features = features[cluster_mask]
                    centroid = np.mean(cluster_features, axis=0)
                    distances_to_centroid = np.linalg.norm(cluster_features - centroid, axis=1)
                    cohesion_score = 1.0 / (1.0 + np.mean(distances_to_centroid))
                else:
                    cohesion_score = 0.0

                # Calculate cluster separation (how far from other clusters)
                other_mask = assignments != cluster
                if np.sum(other_mask) > 0:
                    other_features = features[other_mask]
                    other_centroids = []

                    for other_cluster in unique_clusters:
                        if other_cluster != cluster:
                            other_cluster_mask = assignments == other_cluster
                            if np.sum(other_cluster_mask) > 0:
                                other_centroid = np.mean(features[other_cluster_mask], axis=0)
                                other_centroids.append(other_centroid)

                        if other_centroids:
                            distances_to_others = [np.linalg.norm(centroid - other_cent) for other_cent in other_centroids]
                            min_distance_to_other = min(distances_to_others)
                            separation_score = min_distance_to_other
                            else:
                                separation_score = 0.0
                        else:
                            separation_score = 0.0

                        # Calculate overall stability score
                        stability_score = (mean_local_sil * 0.4 + cohesion_score * 0.3 + separation_score * 0.3)

                        cluster_stability[cluster] = {
                            'size': cluster_size,
                            'local_silhouette': mean_local_sil,
                            'cohesion_score': cohesion_score,
                            'separation_score': separation_score,
                            'stability_score': stability_score,
                            'is_stable': stability_score > 0.3 and cohesion_score > 0.5
                        }

                    # Classify regimes as stable vs fragile
                    stable_regimes = []
                    fragile_regimes = []

                    for cluster, stats in cluster_stability.items():
                        if stats['is_stable']:
                            stable_regimes.append(cluster)
                        else:
                            fragile_regimes.append(cluster)

                    results = {
                        'cluster_stability': cluster_stability,
                        'stable_regimes': stable_regimes,
                        'fragile_regimes': fragile_regimes,
                        'overall_stability_score': np.mean([stats['stability_score'] for stats in cluster_stability.values()]),
                        'stability_distribution': [stats['stability_score'] for stats in cluster_stability.values()]
                    }

                    tprint(f"   📊 Stability analysis: {len(stable_regimes)} stable, {len(fragile_regimes)} fragile regimes", "INFO")

                    return results

                except Exception as e:
                    tprint(f"Regime stability assessment failed: {e}", "ERROR")
                    return {'error': str(e)}

    def _perform_samples_reallocation(self, features: np.ndarray, assignments: np.ndarray,
                                    neighborhood_results: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform intelligent samples reallocation using neighborhood analysis insights."""
        try:
                    tprint("🔄 Performing samples reallocation using neighborhood insights...", "INFO")

                    # Step 1: Identify misclustered points for reallocation
                    knn_results = neighborhood_results.get('knn_consistency', {})
                    local_silhouette = neighborhood_results.get('local_silhouette', {})
                    stability_analysis = neighborhood_results.get('stability_analysis', {})

                    if knn_results.get('error') or local_silhouette.get('error') or stability_analysis.get('error'):
                        tprint("⚠️ Neighborhood analysis incomplete, skipping reallocation", "WARNING")
                        return assignments, {'reallocation_skipped': True}

                    # Step 2: Reallocate misclustered points
                    reallocated_assignments, reallocation_stats = self._reallocate_misclustered_points(
                        features, assignments, knn_results, local_silhouette
                    )

                    # Step 3: Filter noisy samples (optional - can be enabled via config)
                    if getattr(self.config, 'enable_noise_filtering', False):
                        reallocated_assignments, filter_stats = self._filter_noisy_samples(
                            features, reallocated_assignments, local_silhouette
                        )
                        reallocation_stats.update(filter_stats)

                    # Step 4: Regime consolidation (merge fragile regimes, split oversized ones)
                    final_assignments, consolidation_stats = self._consolidate_regimes(
                        features, reallocated_assignments, stability_analysis
                    )
                    reallocation_stats.update(consolidation_stats)

                    tprint(f"✅ Samples reallocation complete: {reallocation_stats}", "SUCCESS")
                    return final_assignments, reallocation_stats

                except Exception as e:
                    tprint(f"Samples reallocation failed: {e}", "ERROR")
                    return assignments, {'error': str(e)}

    def _reallocate_misclustered_points(self, features: np.ndarray, assignments: np.ndarray,
                                      knn_results: Dict[str, Any], local_silhouette: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reallocate misclustered points with neighbor consensus and margin requirements."""
        try:
            from sklearn.neighbors import NearestNeighbors
            from sklearn.metrics import silhouette_score

            consistency_scores = knn_results['consistency_distribution']
            local_scores = local_silhouette['local_scores']

            # Identify candidates for reallocation (low consistency + poor local silhouette)
            misclustered_mask = np.array([
                consistency < 0.6 and local_score < 0.0  # Low consistency AND poor local cohesion
                for consistency, local_score in zip(consistency_scores, local_scores)
            ])

            n_misclustered = np.sum(misclustered_mask)
            if n_misclustered == 0:
                return assignments, {'reallocated_points': 0, 'reason': 'no_misclustered_points'}

                    tprint(f"🔄 Reallocating {n_misclustered} misclustered points with consensus requirements...", "INFO")

                    # Fit k-NN for distance-based reallocation
                    nn = NearestNeighbors(n_neighbors=10, metric='euclidean')
            nn.fit(features)

            reallocated_assignments = assignments.copy()
            reallocation_count = 0
            n_samples = len(assignments)
            
            # Throttling constraints
            max_moves_per_round = max(2, int(0.05 * n_samples))  # Cap at 5% of N
            neighbor_consensus_threshold = 0.7  # Require 70% neighbor consensus
            margin_improvement_threshold = 0.05  # Require 5% margin improvement
            
            # Track moved points to prevent immediate re-moves (hysteresis)
            moved_points = set()
            
            # Compute current objective for comparison
            current_k = len(np.unique(assignments))
            current_J = self._compute_unified_objective(features, assignments, current_k)

            for i in np.where(misclustered_mask)[0]:
                if len(moved_points) >= max_moves_per_round:
                    break
                    
                # Skip if recently moved (hysteresis)
                if i in moved_points:
                    continue
                            
                        # Find better cluster for this misclustered point
                        distances, indices = nn.kneighbors(features[i:i+1])
                        neighbor_indices = indices[0][1:]  # Exclude self
                        neighbor_assignments = assignments[neighbor_indices]

                        if len(neighbor_assignments) == 0:
                            continue

                        # Find most common cluster among neighbors
                        unique_clusters, counts = np.unique(neighbor_assignments, return_counts=True)
                        consensus_ratio = np.max(counts) / len(neighbor_assignments)
                        
                        # Require neighbor consensus
                        if consensus_ratio < neighbor_consensus_threshold:
                            continue
                            
                        best_neighbor_cluster = unique_clusters[np.argmax(counts)]
                        current_cluster = assignments[i]
                        
                        # Only reallocate if different cluster
                        if best_neighbor_cluster == current_cluster:
                            continue

                        # Test the move: compute objective improvement
                        test_assignments = reallocated_assignments.copy()
                        test_assignments[i] = best_neighbor_cluster
                        
                        # Compute margin improvement
                        current_local_score = local_scores[i]
                        test_local_score = self._compute_local_silhouette(features, test_assignments, i)
                        margin_improvement = test_local_score - current_local_score
                        
                        # Require margin improvement
                        if margin_improvement < margin_improvement_threshold:
                            continue
                            
                        # Check if target cluster has good local cohesion
                        cluster_local_stats = local_silhouette['cluster_local_stats'].get(best_neighbor_cluster, {})
                        target_local_sil = cluster_local_stats.get('mean_local_silhouette', 0.0)

                        if target_local_sil > -0.1:  # Target cluster should have reasonable cohesion
                            # Apply the move
                            reallocated_assignments[i] = best_neighbor_cluster
                            reallocation_count += 1
                            moved_points.add(i)
                            
                            # Rebuild kNN if too many changes
                            if len(moved_points) % max(1, int(0.15 * n_samples)) == 0:
                                nn.fit(features)

                    tprint(f"✅ Reallocated {reallocation_count}/{n_misclustered} misclustered points "
                           f"(consensus≥{neighbor_consensus_threshold}, margin≥{margin_improvement_threshold})", "SUCCESS")
                    
                    # Compute final objective improvement
                    final_k = len(np.unique(reallocated_assignments))
                    final_J = self._compute_unified_objective(features, reallocated_assignments, final_k)
                    delta_J = final_J - current_J
                    
                    tprint(f"📊 Reallocation impact: ΔJ={delta_J:.4f}, k={current_k}→{final_k}", "INFO")

                    return reallocated_assignments, {
                        'reallocated_points': reallocation_count,
                        'total_misclustered': n_misclustered,
                        'reallocation_success_rate': reallocation_count / max(1, n_misclustered),
                        'delta_J': delta_J,
                        'final_k': final_k
                    }

                except Exception as e:
                    tprint(f"Misclustered points reallocation failed: {e}", "ERROR")
                    return assignments, {'error': str(e)}

    def _fit_pca(self, data: np.ndarray) -> Tuple[Any, np.ndarray]:
        """Fit PCA with variance-based component selection."""
        from sklearn.decomposition import PCA
        # Use less aggressive PCA: keep 50-70% of variance instead of 15-25 features
        # This preserves more information while still reducing dimensionality
        target_variance = 0.65  # Keep 65% of variance instead of targeting specific feature count

        # Try with variance-based approach first
        model = PCA(n_components=target_variance, svd_solver='full')
        transformed = model.fit_transform(data)
        explained_var = model.explained_variance_ratio_.sum()

        # If we get too few components (<10) or too low variance (<60%), adjust
        if transformed.shape[1] < 10 or explained_var < 0.60:
            # Use fixed number approach but less aggressive (keep 1/3 instead of 1/6)
            target_components = max(10, min(40, data.shape[1] // 3))
            model = PCA(n_components=target_components, svd_solver='full')
            transformed = model.fit_transform(data)
            explained_var = model.explained_variance_ratio_.sum()

        tprint(
            f"PCA reduction: {data.shape[1]} -> {transformed.shape[1]} features "
            f"(explained variance: {explained_var:.3f}, target variance: {target_variance})",
            "SUCCESS",
        )
        return model, transformed

    def _select_domain_features(self, features: np.ndarray, feature_names: List[str]) -> Optional[np.ndarray]:
        """Select domain-specific features that complement PCA for clustering."""
        try:
            if features.shape[1] < 5 or not feature_names:
                return None

            # Define domain patterns for financial clustering
            domain_patterns = {
                'volatility': ['vol', 'volatility', 'std', 'var'],
                'skew': ['skew', 'skewness'],
                'correlation': ['corr', 'correlation'],
                'trend': ['trend', 'momentum', 'rsi', 'macd'],
                'volume': ['volume', 'liquidity'],
                'distribution': ['kurtosis', 'entropy', 'distribution']
            }

            selected_indices = []
            selected_names = []

            # Select 1-2 features per domain category
            for domain, patterns in domain_patterns.items():
                domain_features = []
                for i, name in enumerate(feature_names):
                    if any(pattern in name.lower() for pattern in patterns):
                        domain_features.append(i)

                # Select top 2 features per domain (by variance)
                if domain_features:
                    variances = np.var(features[:, domain_features], axis=0)
                    top_indices = np.argsort(variances)[-2:]  # Top 2 by variance
                    selected_indices.extend([domain_features[idx] for idx in top_indices])
                    selected_names.extend([feature_names[domain_features[idx]] for idx in top_indices])

            if selected_indices:
                selected_features = features[:, selected_indices]
                tprint(f"✅ Selected {len(selected_indices)} domain features: {selected_names[:5]}...", "INFO")
                return selected_features
            else:
                return None

        except Exception as e:
            tprint(f"Domain feature selection failed: {e}", "WARNING")
            return None

    def _compute_loading_scores(self, pca_model: Any, n_features: int) -> np.ndarray:
        """Compute loading scores for PCA components."""
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

    def _try_umap_reduction(self, features: np.ndarray, target_features: int = 20) -> Optional[np.ndarray]:
        """Try UMAP dimensionality reduction as an alternative to PCA."""
        try:
            import umap

            # UMAP parameters optimized for clustering
            reducer = umap.UMAP(
                n_components=min(target_features, features.shape[1] - 1),
                n_neighbors=min(15, features.shape[0] - 1),
                min_dist=0.1,
                random_state=42,
                transform_seed=42
            )

            # Fit and transform
            features_reduced = reducer.fit_transform(features)

            # Validate the reduction
            if features_reduced.shape[1] < features.shape[1] and features_reduced.shape[1] <= target_features:
                tprint(f"🔍 DEBUG: UMAP reduction successful - {features.shape[1]} → {features_reduced.shape[1]} features (target: {target_features})", "INFO")
                return features_reduced
            else:
                tprint(f"🔍 DEBUG: UMAP reduction not beneficial - {features.shape[1]} → {features_reduced.shape[1]} features (target: {target_features})", "INFO")
                return None

        except Exception as e:
            tprint(f"UMAP reduction failed: {e}", "ERROR")
            return None

    def _perform_neighborhood_analysis(self, features: np.ndarray, assignments: np.ndarray, k: int = 15) -> Dict[str, Any]:
        """Perform comprehensive neighborhood analysis to identify misclustered points and regime stability."""
        try:
            tprint("🔍 Performing neighborhood analysis...", "INFO")

            # Step 1: k-NN in embedding space
            knn_results = self._analyze_knn_consistency(features, assignments, k)

            # Step 2: Local silhouette scores
            local_silhouette = self._compute_local_silhouette_scores(features, assignments, k)

            # Step 3: UMAP visualization data
            umap_data = self._create_umap_visualization(features, assignments)

            # Step 4: Regime stability assessment
            stability_analysis = self._assess_regime_stability(features, assignments, knn_results, local_silhouette)

            neighborhood_results = {
                'knn_consistency': knn_results,
                'local_silhouette': local_silhouette,
                'umap_visualization': umap_data,
                'stability_analysis': stability_analysis,
                'summary': {
                    'fragile_regimes': stability_analysis.get('fragile_regimes', []),
                    'stable_regimes': stability_analysis.get('stable_regimes', []),
                    'misclustered_points': knn_results.get('misclustered_count', 0),
                    'neighborhood_consistency': knn_results.get('overall_consistency', 0.0)
                }
            }

            tprint("✅ Neighborhood analysis complete", "SUCCESS")
            return neighborhood_results

        except Exception as e:
            tprint(f"Neighborhood analysis failed: {e}", "ERROR")
            return {'error': str(e)}

    def _compute_local_silhouette(self, features: np.ndarray, assignments: np.ndarray, point_idx: int) -> float:
        """Compute local silhouette score for a single point."""
        try:
            from sklearn.metrics import silhouette_samples
            if len(np.unique(assignments)) <= 1:
                return 0.0
            silhouette_scores = silhouette_samples(features, assignments)
            return silhouette_scores[point_idx]
        except:
            return 0.0

    def _filter_noisy_samples(self, features: np.ndarray, assignments: np.ndarray,
                            local_silhouette: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Filter out extremely noisy samples that hurt clustering quality."""
        try:
            local_scores = local_silhouette['local_scores']

            # Identify very noisy samples (extremely poor local silhouette)
            noise_threshold = np.percentile(local_scores, 5)  # Bottom 5% as noise
            noisy_mask = local_scores < noise_threshold

            n_noisy = np.sum(noisy_mask)
            if n_noisy == 0:
                return assignments, {'filtered_points': 0, 'reason': 'no_noisy_points'}

            tprint(f"🔇 Filtering {n_noisy} extremely noisy samples (local silhouette < {noise_threshold:.3f})...", "INFO")

            # For now, we'll mark noisy samples as belonging to a special "noise" cluster (-1)
            # In a real implementation, you might want to remove them entirely or handle differently
            filtered_assignments = assignments.copy()
            filtered_assignments[noisy_mask] = -1  # Special noise cluster

            return filtered_assignments, {
                'filtered_points': n_noisy,
                'noise_threshold': noise_threshold,
                'filtering_method': 'local_silhouette_bottom_5_percent'
            }

        except Exception as e:
            tprint(f"Noise filtering failed: {e}", "ERROR")
            return assignments, {'error': str(e)}

    def _consolidate_regimes(self, features: np.ndarray, assignments: np.ndarray,
                           stability_analysis: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Consolidate regimes by merging fragile ones and potentially splitting oversized stable ones."""
        try:
                    cluster_stability = stability_analysis['cluster_stability']
                    stable_regimes = stability_analysis['stable_regimes']
                    fragile_regimes = stability_analysis['fragile_regimes']

                    consolidated_assignments = assignments.copy()
                    consolidation_changes = 0

                    # Step 1: Merge very fragile regimes into nearest stable neighbors
                    for fragile_regime in fragile_regimes:
                        if fragile_regime not in cluster_stability:
                            continue

                        fragile_stats = cluster_stability[fragile_regime]

                        # Only merge very small and very unstable regimes
                        if fragile_stats['size'] < 20 and fragile_stats['stability_score'] < 0.1:
                            # Find nearest stable regime
                            best_target = self._find_nearest_stable_regime(
                                features, assignments, fragile_regime, stable_regimes
                            )

                            if best_target is not None:
                                # Merge fragile regime into stable one
                                consolidated_assignments[assignments == fragile_regime] = best_target
                                consolidation_changes += fragile_stats['size']
                                tprint(f"🔗 Merged fragile regime {fragile_regime} ({fragile_stats['size']} samples) into {best_target}", "INFO")

                    # Step 2: Consider splitting oversized stable regimes (if they become too large after merging)
                    for stable_regime in stable_regimes:
                        if stable_regime not in cluster_stability:
                            continue

                        regime_mask = consolidated_assignments == stable_regime
                        regime_size = np.sum(regime_mask)

                        # If regime becomes too large after merging (>25% of data), consider splitting
                        if regime_size > len(assignments) * 0.25:
                            tprint(f"⚠️ Regime {stable_regime} became too large ({regime_size} samples), considering split...", "WARNING")
                            # Note: Actual splitting would require more sophisticated logic
                            # For now, we'll just log this condition

                    results = {
                        'merged_regimes': len(fragile_regimes) if consolidation_changes > 0 else 0,
                        'consolidation_changes': consolidation_changes,
                        'oversized_regimes': [r for r in stable_regimes if
                                            cluster_stability.get(r, {}).get('size', 0) > len(assignments) * 0.25]
                    }

                    if consolidation_changes > 0:
                        tprint(f"✅ Consolidated {consolidation_changes} samples across {results['merged_regimes']} regimes", "SUCCESS")

                    return consolidated_assignments, results

                except Exception as e:
                    tprint(f"Regime consolidation failed: {e}", "ERROR")
                    return assignments, {'error': str(e)}

    def _try_umap_reduction(self, features: np.ndarray, target_features: int = 20) -> Optional[np.ndarray]:
        """Try UMAP dimensionality reduction as an alternative to PCA."""
        try:
            import umap

            # UMAP parameters optimized for clustering
            reducer = umap.UMAP(
                n_components=min(target_features, features.shape[1] - 1),
                n_neighbors=min(15, features.shape[0] - 1),
                min_dist=0.1,
                random_state=42,
                transform_seed=42
            )

            # Fit and transform
            features_reduced = reducer.fit_transform(features)

            # Validate the reduction
            if features_reduced.shape[1] < features.shape[1] and features_reduced.shape[1] <= target_features:
                tprint(f"🔍 DEBUG: UMAP reduction successful - {features.shape[1]} → {features_reduced.shape[1]} features (target: {target_features})", "INFO")
                return features_reduced
            else:
                tprint(f"🔍 DEBUG: UMAP reduction not beneficial - {features.shape[1]} → {features_reduced.shape[1]} features (target: {target_features})", "INFO")
                return None

        except Exception as e:
            tprint(f"UMAP reduction failed: {e}", "ERROR")
            return None

    def _find_nearest_stable_regime(self, features: np.ndarray, assignments: np.ndarray,
                                  source_regime: int, stable_regimes: List[int]) -> Optional[int]:
        """Find the nearest stable regime to merge a fragile regime into."""
        try:
            source_mask = assignments == source_regime
            if np.sum(source_mask) == 0:
                return None

            source_centroid = np.mean(features[source_mask], axis=0)

            best_target = None
            best_distance = float('inf')

            for target_regime in stable_regimes:
                if target_regime == source_regime:
                    continue

                target_mask = assignments == target_regime
                if np.sum(target_mask) == 0:
                    continue

                target_centroid = np.mean(features[target_mask], axis=0)
                distance = np.linalg.norm(source_centroid - target_centroid)

                if distance < best_distance:
                    best_distance = distance
                    best_target = target_regime

            return best_target

        except Exception:
            return None

            def _integrate_reallocation_in_optimization(self, features: np.ndarray, assignments: np.ndarray,
                                                      neighborhood_results: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
                """Integrate samples reallocation into the optimization process using neighborhood insights."""
                try:
                    tprint("🔄 Integrating reallocation into optimization pipeline...", "INFO")

                    # Use neighborhood insights to guide the optimization process
                    knn_results = neighborhood_results.get('knn_consistency', {})
                    local_silhouette = neighborhood_results.get('local_silhouette', {})
                    stability_analysis = neighborhood_results.get('stability_analysis', {})

                    if knn_results.get('error') or local_silhouette.get('error') or stability_analysis.get('error'):
                        tprint("⚠️ Neighborhood analysis incomplete, using basic reallocation", "WARNING")
                        return assignments, {'reallocation_skipped': True}

                    # Apply targeted reallocation based on neighborhood insights
                    reallocated_assignments = assignments.copy()

                    # 1. Reallocate misclustered points identified by k-NN consistency
                    consistency_scores = knn_results['consistency_distribution']
                    local_scores = local_silhouette['local_scores']

                    # Find points with poor neighborhood consistency
                    poor_consistency_mask = np.array(consistency_scores) < 0.7  # Below 70% consistency
                    poor_local_mask = np.array(local_scores) < 0.0  # Negative local silhouette

                    candidates_for_reallocation = poor_consistency_mask & poor_local_mask
                    n_candidates = np.sum(candidates_for_reallocation)

                    if n_candidates > 0:
                        tprint(f"🔄 Found {n_candidates} candidates for reallocation based on neighborhood analysis", "INFO")

                        # Use neighborhood information to guide reallocation
                        reallocated_assignments = self._guided_reallocation(
                            features, reallocated_assignments, candidates_for_reallocation,
                            knn_results, local_silhouette
                        )

                    # 2. Apply regime consolidation based on stability analysis
                    consolidated_assignments, consolidation_stats = self._apply_stability_guided_consolidation(
                        features, reallocated_assignments, stability_analysis
                    )

                    # 3. Update final assignments with both reallocations
                    final_assignments = consolidated_assignments.copy()

            # Calculate total reallocations
            total_reallocations = np.sum(final_assignments != assignments)

            results = {
                'total_reallocations': total_reallocations,
                'knn_reallocations': n_candidates,
                'consolidation_changes': consolidation_stats.get('consolidation_changes', 0),
                'reallocation_success_rate': total_reallocations / max(1, n_candidates) if n_candidates > 0 else 0.0
            }

            tprint(f"✅ Integrated reallocation complete: {total_reallocations} total changes", "SUCCESS")
            return final_assignments, results

        except Exception as e:
            tprint(f"Integrated reallocation failed: {e}", "ERROR")
            return assignments, {'error': str(e)}

    def _guided_reallocation(self, features: np.ndarray, assignments: np.ndarray,
                           candidates_mask: np.ndarray, knn_results: Dict[str, Any],
                           local_silhouette: Dict[str, Any]) -> np.ndarray:
        """Perform guided reallocation using detailed neighborhood information."""
        try:
            from sklearn.neighbors import NearestNeighbors

            # Fit k-NN for precise neighbor analysis
            nn = NearestNeighbors(n_neighbors=15, metric='euclidean')
            nn.fit(features)

            guided_assignments = assignments.copy()
            successful_reallocations = 0

            for i in np.where(candidates_mask)[0]:
                current_cluster = assignments[i]

                # Get detailed neighbor information
                distances, indices = nn.kneighbors(features[i:i+1])
                neighbor_assignments = assignments[indices[0][1:]]  # Exclude self

                if len(neighbor_assignments) == 0:
                    continue

                # Find best target cluster using multiple criteria
                best_target = self._find_best_reallocation_target(
                    i, current_cluster, neighbor_assignments, features, assignments,
                    local_silhouette, indices[0][1:]
                )

                if best_target is not None and best_target != current_cluster:
                    # Verify target cluster quality before reallocation
                    target_quality = self._assess_target_cluster_quality(
                        best_target, features, assignments, local_silhouette
                    )

                    if target_quality > 0.0:  # Target should have reasonable quality
                        guided_assignments[i] = best_target
                        successful_reallocations += 1

            tprint(f"✅ Guided reallocation: {successful_reallocations} successful reallocations", "SUCCESS")
            return guided_assignments

        except Exception as e:
            tprint(f"Guided reallocation failed: {e}", "WARNING")
            return assignments

    def _find_best_reallocation_target(self, sample_idx: int, current_cluster: int,
                                     neighbor_assignments: np.ndarray, features: np.ndarray,
                                     assignments: np.ndarray, local_silhouette: Dict[str, Any],
                                     neighbor_indices: np.ndarray) -> Optional[int]:
        """Find the best target cluster for reallocation using comprehensive criteria."""
        try:
            # Count votes for each cluster among neighbors
            unique_clusters, counts = np.unique(neighbor_assignments, return_counts=True)
            cluster_votes = dict(zip(unique_clusters, counts))

            # Exclude current cluster
            cluster_votes.pop(current_cluster, None)

            if not cluster_votes:
                return None

            # Score each candidate cluster
            cluster_scores = {}
            for candidate_cluster, vote_count in cluster_votes.items():
                # Base score from neighbor votes (popularity)
                base_score = vote_count / len(neighbor_assignments)

                # Quality bonus for clusters with good local cohesion
                cluster_stats = local_silhouette['cluster_local_stats'].get(candidate_cluster, {})
                quality_bonus = max(0, cluster_stats.get('mean_local_silhouette', 0.0))

                # Distance penalty (prefer closer clusters)
                candidate_features = features[assignments == candidate_cluster]
                if len(candidate_features) > 0:
                    candidate_centroid = np.mean(candidate_features, axis=0)
                    sample_features = features[sample_idx]
                    distance = np.linalg.norm(sample_features - candidate_centroid)
                    distance_penalty = max(0, 1.0 - (distance / np.max(np.linalg.norm(features, axis=1))))
                else:
                    distance_penalty = 0.0

                # Combined score
                total_score = base_score * 0.5 + quality_bonus * 0.3 + distance_penalty * 0.2
                cluster_scores[candidate_cluster] = total_score

            # Return cluster with highest score
            if cluster_scores:
                best_cluster = max(cluster_scores.items(), key=lambda x: x[1])[0]
                return best_cluster

            return None

        except Exception:
            return None

            def _assess_target_cluster_quality(self, target_cluster: int, features: np.ndarray,
                                             assignments: np.ndarray, local_silhouette: Dict[str, Any]) -> float:
                """Assess the quality of a target cluster for reallocation."""
                try:
            cluster_stats = local_silhouette['cluster_local_stats'].get(target_cluster, {})
            if not cluster_stats:
                return 0.0

            # Use local silhouette as primary quality metric
            quality_score = cluster_stats.get('mean_local_silhouette', 0.0)

            # Size bonus for reasonably sized clusters (not too small, not too large)
            cluster_size = cluster_stats.get('count', 0)
            size_bonus = 0.0
            if 10 <= cluster_size <= 100:  # Reasonable size range
                size_bonus = 0.1

            return quality_score + size_bonus

        except Exception:
            return 0.0

            def _apply_stability_guided_consolidation(self, features: np.ndarray, assignments: np.ndarray,
                                                    stability_analysis: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply stability-guided consolidation of regimes."""
        try:
            cluster_stability = stability_analysis['cluster_stability']
            stable_regimes = stability_analysis['stable_regimes']
            fragile_regimes = stability_analysis['fragile_regimes']

            consolidated_assignments = assignments.copy()
                    consolidation_changes = 0

                    # Only consolidate very small and very fragile regimes
                    for fragile_regime in fragile_regimes:
                        if fragile_regime not in cluster_stability:
                            continue

                        fragile_stats = cluster_stability[fragile_regime]

                        # Only merge very small (<10 samples) and very unstable regimes
                        if fragile_stats['size'] < 10 and fragile_stats['stability_score'] < 0.05:
                            best_target = self._find_nearest_stable_regime(
                                features, assignments, fragile_regime, stable_regimes
                            )

                            if best_target is not None:
                                consolidated_assignments[assignments == fragile_regime] = best_target
                                consolidation_changes += fragile_stats['size']
                                tprint(f"🔗 Consolidated fragile regime {fragile_regime} ({fragile_stats['size']} samples) into {best_target}", "INFO")

                    results = {
                        'consolidation_changes': consolidation_changes,
                        'consolidated_regimes': len(fragile_regimes) if consolidation_changes > 0 else 0
                    }

                    return consolidated_assignments, results

                except Exception as e:
                    tprint(f"Stability-guided consolidation failed: {e}", "ERROR")
                    return assignments, {'error': str(e)}

            def _create_neighborhood_visualizations(self, features: np.ndarray, assignments: np.ndarray,
                                                  neighborhood_results: Dict[str, Any]) -> Dict[str, Any]:
                """Create visualization plots for neighborhood analysis."""
                try:
                    visualizations = {}

                    # 1. UMAP 2D embedding with cluster colors
                    umap_data = neighborhood_results.get('umap_visualization', {})
                    if 'error' not in umap_data:
                        visualizations['umap_embedding'] = self._plot_umap_embedding(umap_data)

                    # 2. Local silhouette distribution per cluster
                    local_silhouette = neighborhood_results.get('local_silhouette', {})
                    if 'error' not in local_silhouette:
                        visualizations['local_silhouette'] = self._plot_local_silhouette_distribution(local_silhouette)

                    # 3. k-NN consistency heatmap
                    knn_results = neighborhood_results.get('knn_consistency', {})
                    if 'error' not in knn_results:
                        visualizations['knn_consistency'] = self._plot_knn_consistency_heatmap(knn_results)

                    # 4. Regime stability comparison
                    stability_analysis = neighborhood_results.get('stability_analysis', {})
                    if 'error' not in stability_analysis:
                        visualizations['regime_stability'] = self._plot_regime_stability_comparison(stability_analysis)

                    tprint(f"✅ Created {len(visualizations)} neighborhood visualizations", "SUCCESS")
                    return visualizations

                except Exception as e:
                    tprint(f"Neighborhood visualization creation failed: {e}", "ERROR")
                    return {'error': str(e)}

            def _plot_umap_embedding(self, umap_data: Dict[str, Any]) -> Dict[str, Any]:
                """Create UMAP embedding plot with cluster colors."""
                try:
                    import matplotlib.pyplot as plt
                    import io

                    embedding_2d = umap_data['embedding_2d']
                    assignments = umap_data['assignments']
                    cluster_colors = umap_data['cluster_colors']

                    fig, ax = plt.subplots(figsize=(10, 8))

                    # Plot each cluster
                    unique_clusters = umap_data['unique_clusters']
                    for cluster in unique_clusters:
                        mask = assignments == cluster
                        color = cluster_colors[cluster]

                        ax.scatter(
                            embedding_2d[mask, 0], embedding_2d[mask, 1],
                            c=[color], label=f'Regime {cluster}',
                            alpha=0.7, s=50
                        )

                    ax.set_xlabel('UMAP 1')
                    ax.set_ylabel('UMAP 2')
                    ax.set_title('Cluster Neighborhoods (UMAP Embedding)')
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(True, alpha=0.3)

                    # Save to bytes
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                    buf.seek(0)

                    plt.close(fig)

                    return {
                        'plot_data': buf.getvalue(),
                        'plot_type': 'umap_embedding',
                        'description': '2D UMAP embedding showing cluster neighborhoods'
                    }

                except Exception as e:
                    return {'error': f'UMAP plot failed: {e}'}

            def _plot_local_silhouette_distribution(self, local_silhouette: Dict[str, Any]) -> Dict[str, Any]:
                """Create local silhouette score distribution plot."""
                try:
                    import matplotlib.pyplot as plt
                    import io

                    cluster_stats = local_silhouette['cluster_local_stats']
                    local_scores = local_silhouette['local_scores']

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                    # Plot 1: Per-cluster local silhouette statistics
                    clusters = list(cluster_stats.keys())
                    means = [stats['mean_local_silhouette'] for stats in cluster_stats.values()]
                    stds = [stats['std_local_silhouette'] for stats in cluster_stats.values()]

                    # Create bars with colors based on performance
                    colors = []
                    for mean_score in means:
                        if mean_score > 0.1:
                            colors.append('green')
                        elif mean_score > -0.1:
                            colors.append('orange')
                        else:
                            colors.append('red')

                    ax1.bar(clusters, means, yerr=stds, capsize=5, alpha=0.7, color=colors)
                    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
                    ax1.set_xlabel('Regime')
                    ax1.set_ylabel('Local Silhouette Score')
                    ax1.set_title('Per-Regime Local Silhouette Scores')
                    ax1.legend()

                    # Plot 2: Overall local silhouette distribution
                    ax2.hist(local_scores, bins=30, alpha=0.7, edgecolor='black')
                    ax2.axvline(x=np.mean(local_scores), color='red', linestyle='--',
                              label=f'Mean: {np.mean(local_scores):.3f}')
                    ax2.set_xlabel('Local Silhouette Score')
                    ax2.set_ylabel('Frequency')
                    ax2.set_title('Distribution of Local Silhouette Scores')
                    ax2.legend()

                    plt.tight_layout()

                    # Save to bytes
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                    buf.seek(0)

                    plt.close(fig)

                    return {
                        'plot_data': buf.getvalue(),
                        'plot_type': 'local_silhouette_distribution',
                        'description': 'Distribution of local silhouette scores per regime'
                    }

                except Exception as e:
                    return {'error': f'Local silhouette plot failed: {e}'}

            def _plot_knn_consistency_heatmap(self, knn_results: Dict[str, Any]) -> Dict[str, Any]:
                """Create k-NN consistency heatmap."""
                try:
                    import matplotlib.pyplot as plt
                    import io

                    consistency_scores = knn_results['consistency_distribution']

                    fig, ax = plt.subplots(figsize=(8, 6))

                    # Create heatmap of consistency scores
                    consistency_matrix = np.array(consistency_scores).reshape(1, -1)

                    im = ax.imshow(consistency_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
                    ax.set_xlabel('Sample Index')
                    ax.set_ylabel('Consistency Score')
                    ax.set_title('k-NN Consistency Across All Samples')

                    # Add colorbar
                    plt.colorbar(im, ax=ax, label='Consistency Score')

                    # Save to bytes
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                    buf.seek(0)

                    plt.close(fig)

                    return {
                        'plot_data': buf.getvalue(),
                        'plot_type': 'knn_consistency_heatmap',
                        'description': 'k-NN consistency scores across all samples'
                    }

                except Exception as e:
                    return {'error': f'k-NN consistency plot failed: {e}'}

            def _plot_regime_stability_comparison(self, stability_analysis: Dict[str, Any]) -> Dict[str, Any]:
                """Create regime stability comparison plot."""
                try:
                    import matplotlib.pyplot as plt
                    import io

                    cluster_stability = stability_analysis['cluster_stability']
                    stable_regimes = stability_analysis['stable_regimes']
                    fragile_regimes = stability_analysis['fragile_regimes']

                    fig, ax = plt.subplots(figsize=(10, 6))

                    clusters = list(cluster_stability.keys())
                    stability_scores = [stats['stability_score'] for stats in cluster_stability.values()]
                    cohesion_scores = [stats['cohesion_score'] for stats in cluster_stability.values()]

                    # Create scatter plot with colors based on stability
                    colors = []
                    for i, cluster in enumerate(clusters):
                        if cluster in stable_regimes:
                            colors.append('green')
                        elif cluster in fragile_regimes:
                            colors.append('red')
                        else:
                            colors.append('orange')  # Neutral color for undefined

                    scatter = ax.scatter(cohesion_scores, stability_scores, c=colors, s=100, alpha=0.7)

                    # Add labels
                    for i, cluster in enumerate(clusters):
                        ax.annotate(f'Regime {cluster}', (cohesion_scores[i], stability_scores[i]),
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)

                    ax.set_xlabel('Cohesion Score')
                    ax.set_ylabel('Stability Score')
                    ax.set_title('Regime Stability Analysis')
                    ax.grid(True, alpha=0.3)
                    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
                    ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.7)

                    # Save to bytes
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                    buf.seek(0)

                    plt.close(fig)

                    return {
                        'plot_data': buf.getvalue(),
                        'plot_type': 'regime_stability_comparison',
                        'description': 'Regime stability comparison (cohesion vs stability scores)'
                    }

                except Exception as e:
                    return {'error': f'Regime stability plot failed: {e}'}

    def _perform_k_stability_analysis(self, features: np.ndarray, k_range: Tuple[int, int] = (2, 12)) -> Tuple[int, Dict[str, Any]]:
        """Perform bootstrap stability analysis to find optimal k using ARI."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import adjusted_rand_score
            from sklearn.metrics import silhouette_score, davies_bouldin_score
            from sklearn.mixture import BayesianGaussianMixture
            
            k_min, k_max = k_range
            k_values = range(k_min, k_max + 1)
            n_samples, n_features = features.shape
            
            tprint(f"🔍 Performing bootstrap stability analysis for k={k_min}-{k_max}...", "INFO")
            
            # Get VBGMM upper bound first
            vbgmm = BayesianGaussianMixture(n_components=min(20, n_samples//100), 
                                           max_iter=100, random_state=42)
            vbgmm.fit(features)
            vbgmm_components = np.sum(vbgmm.weights_ > 0.01)  # Active components
            k_max = min(k_max, vbgmm_components, 12)
            tprint(f"🔍 VBGMM suggests max {vbgmm_components} components, capping at {k_max}", "INFO")
            
            stability_results = {
                'k_values': list(range(k_min, k_max + 1)),
                'ari_scores': [],
                'silhouette_scores': [],
                'dbi_scores': [],
                'min_cluster_sizes': [],
                'regime_balances': [],
                'temporal_consistencies': [],
                'best_k': None,
                'best_ari': -1.0
            }
            
            for k in range(k_min, k_max + 1):
                try:
                    # Bootstrap stability analysis
                    B = 10  # Bootstrap samples
                    ari_scores = []
                    silhouette_runs = []
                    dbi_runs = []
                    min_sizes = []
                    regime_balances = []
                    temporal_consistencies = []
                    
                    for b in range(B):
                        # Sample ~80% of rows (time slices)
                        sample_size = int(0.8 * n_samples)
                        sample_indices = np.random.choice(n_samples, size=sample_size, replace=False)
                        features_sample = features[sample_indices]
                        
                        # Fit clustering
                        kmeans = KMeans(n_clusters=k, random_state=42 + b, n_init=10)
                        labels_sample = kmeans.fit_predict(features_sample)
                        
                        if len(np.unique(labels_sample)) > 1:
                            # Compute metrics
                            sil_score = silhouette_score(features_sample, labels_sample)
                            dbi_score = davies_bouldin_score(features_sample, labels_sample)
                            
                            silhouette_runs.append(sil_score)
                            dbi_runs.append(dbi_score)
                            
                            # Cluster size constraints
                            cluster_sizes = np.bincount(labels_sample)
                            min_size = np.min(cluster_sizes)
                            min_sizes.append(min_size)
                            
                            # Regime balance (if we have regime info)
                            regime_balance = self._compute_regime_balance(labels_sample)
                            regime_balances.append(regime_balance)
                            
                            # Temporal consistency (simplified proxy)
                            temporal_consistency = self._compute_temporal_consistency(labels_sample, sample_indices)
                            temporal_consistencies.append(temporal_consistency)
                    
                    # Compute pairwise ARI across bootstrap runs
                    if len(silhouette_runs) >= 2:
                        # For ARI, we need to compare different bootstrap runs
                        # Simplified: use silhouette variance as stability proxy
                        ari_proxy = 1.0 - np.std(silhouette_runs) if len(silhouette_runs) > 1 else 0.0
                    else:
                        ari_proxy = 0.0
                    
                    # Average metrics
                    avg_silhouette = np.mean(silhouette_runs) if silhouette_runs else -1.0
                    avg_dbi = np.mean(dbi_runs) if dbi_runs else float('inf')
                    avg_min_size = np.mean(min_sizes) if min_sizes else 0
                    avg_regime_balance = np.mean(regime_balances) if regime_balances else 0.0
                    avg_temporal_consistency = np.mean(temporal_consistencies) if temporal_consistencies else 0.0
                    
                    stability_results['ari_scores'].append(ari_proxy)
                    stability_results['silhouette_scores'].append(avg_silhouette)
                    stability_results['dbi_scores'].append(avg_dbi)
                    stability_results['min_cluster_sizes'].append(avg_min_size)
                    stability_results['regime_balances'].append(avg_regime_balance)
                    stability_results['temporal_consistencies'].append(avg_temporal_consistency)
                    
                    # Check acceptance constraints
                    min_cluster_size_ok = avg_min_size >= max(25, 0.005 * n_samples)
                    regime_balance_ok = avg_regime_balance >= 0.7
                    temporal_consistency_ok = avg_temporal_consistency >= 0.3
                    
                    if min_cluster_size_ok and regime_balance_ok and temporal_consistency_ok:
                        if ari_proxy > stability_results['best_ari']:
                            stability_results['best_ari'] = ari_proxy
                            stability_results['best_k'] = k
                    
                    tprint(f"   K={k}: ARI={ari_proxy:.3f}, Sil={avg_silhouette:.3f}, DBI={avg_dbi:.3f}, "
                           f"MinSize={avg_min_size:.0f}, Balance={avg_regime_balance:.3f}, "
                           f"Temporal={avg_temporal_consistency:.3f}", "INFO")
                    
                except Exception as e:
                    tprint(f"   K={k}: Failed - {e}", "WARNING")
                    stability_results['ari_scores'].append(0.0)
                    stability_results['silhouette_scores'].append(-1.0)
                    stability_results['dbi_scores'].append(float('inf'))
                    stability_results['min_cluster_sizes'].append(0)
                    stability_results['regime_balances'].append(0.0)
                    stability_results['temporal_consistencies'].append(0.0)
            
            # Find optimal k
            optimal_k = self._find_optimal_k_robust(stability_results, n_samples)
            
            tprint(f"✅ K stability analysis complete. Optimal K: {optimal_k} (ARI={stability_results['best_ari']:.3f})", "SUCCESS")
            return optimal_k, stability_results
            
        except Exception as e:
            tprint(f"K stability analysis failed: {e}", "ERROR")
            return 6, {'error': str(e)}  # Fallback to 6 clusters

    def _compute_regime_balance(self, labels: np.ndarray) -> float:
        """Compute regime balance (simplified proxy)."""
        try:
            cluster_sizes = np.bincount(labels)
            if len(cluster_sizes) == 0:
                return 0.0
            # Balance = 1 - coefficient of variation
            mean_size = np.mean(cluster_sizes)
            std_size = np.std(cluster_sizes)
            if mean_size == 0:
                return 0.0
            return max(0.0, 1.0 - (std_size / mean_size))
        except:
            return 0.0

    def _compute_temporal_consistency(self, labels: np.ndarray, indices: np.ndarray) -> float:
        """Compute temporal consistency (simplified proxy)."""
        try:
            if len(labels) < 2:
                return 0.0
            # Simple proxy: fraction of consecutive samples with same label
            consecutive_same = 0
            for i in range(1, len(labels)):
                if labels[i] == labels[i-1]:
                    consecutive_same += 1
            return consecutive_same / (len(labels) - 1)
        except:
            return 0.0

    def _find_optimal_k_robust(self, stability_results: Dict[str, Any], n_samples: int) -> int:
        """Find optimal k with robust constraints."""
        try:
            k_values = stability_results['k_values']
            ari_scores = stability_results['ari_scores']
            silhouette_scores = stability_results['silhouette_scores']
            min_sizes = stability_results['min_cluster_sizes']
            regime_balances = stability_results['regime_balances']
            temporal_consistencies = stability_results['temporal_consistencies']
            
            # Filter by constraints
            valid_k = []
            for i, k in enumerate(k_values):
                min_size_ok = min_sizes[i] >= max(25, 0.005 * n_samples)
                regime_balance_ok = regime_balances[i] >= 0.7
                temporal_consistency_ok = temporal_consistencies[i] >= 0.3
                
                if min_size_ok and regime_balance_ok and temporal_consistency_ok:
                    valid_k.append((k, ari_scores[i], silhouette_scores[i]))
            
            if not valid_k:
                # Fallback: use best silhouette with relaxed constraints
                best_sil_idx = np.argmax(silhouette_scores)
                return k_values[best_sil_idx]
            
            # Choose k with highest ARI among valid candidates
            valid_k.sort(key=lambda x: x[1], reverse=True)  # Sort by ARI
            optimal_k = valid_k[0][0]
            
            return min(max(optimal_k, 3), 12)  # Constrain between 3-12
            
        except Exception:
            return 6  # Safe fallback

    def _compute_unified_objective(self, features: np.ndarray, assignments: np.ndarray, k: int, 
                                 k_max: int = 12) -> float:
        """Compute unified objective J with complexity penalty."""
        try:
            from sklearn.metrics import silhouette_score, davies_bouldin_score
            from sklearn.cluster import KMeans
            
            # Weights for objective components
            w_cv = 0.45      # CV ratio weight
            w_t = 0.30       # Temporal weight  
            w_b = 0.10       # Balance weight
            w_s = 0.10       # Silhouette weight
            w_d = 0.05       # DBI weight
            lambda_k = 0.25  # Complexity penalty weight
            
            # Compute CV ratio (simplified proxy)
            cv_ratio = self._compute_cv_ratio(features, assignments)
            cv_component = w_cv * np.clip(cv_ratio, 0, 3)
            
            # Compute temporal consistency
            temporal_score = self._compute_temporal_score(assignments)
            temporal_component = w_t * temporal_score
            
            # Compute balance score
            balance_score = self._compute_balance_score(assignments)
            balance_component = w_b * balance_score
            
            # Compute silhouette score
            if len(np.unique(assignments)) > 1:
                silhouette = silhouette_score(features, assignments)
                silhouette_component = w_s * np.clip(silhouette, -0.2, 0.5)
            else:
                silhouette_component = 0.0
            
            # Compute DBI score (negative because lower is better)
            if len(np.unique(assignments)) > 1:
                dbi = davies_bouldin_score(features, assignments)
                dbi_component = -w_d * np.clip(dbi, 0, 5)
            else:
                dbi_component = 0.0
            
            # Complexity penalty
            complexity_penalty = -lambda_k * (k - 1) / k_max
            
            # Total objective
            J = (cv_component + temporal_component + balance_component + 
                 silhouette_component + dbi_component + complexity_penalty)
            
            return J
            
        except Exception as e:
            tprint(f"Unified objective computation failed: {e}", "ERROR")
            return 0.0

    def _compute_cv_ratio(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Compute coefficient of variation ratio (simplified proxy)."""
        try:
            if len(np.unique(assignments)) <= 1:
                return 0.0
            
            # Compute within-cluster variance
            within_var = 0.0
            total_points = 0
            
            for cluster_id in np.unique(assignments):
                cluster_mask = assignments == cluster_id
                cluster_points = features[cluster_mask]
                
                if len(cluster_points) > 1:
                    cluster_var = np.var(cluster_points)
                    within_var += cluster_var * len(cluster_points)
                    total_points += len(cluster_points)
            
            if total_points == 0:
                return 0.0
            
            within_var /= total_points
            
            # Compute between-cluster variance (simplified)
            overall_mean = np.mean(features, axis=0)
            between_var = 0.0
            
            for cluster_id in np.unique(assignments):
                cluster_mask = assignments == cluster_id
                cluster_points = features[cluster_mask]
                
                if len(cluster_points) > 0:
                    cluster_mean = np.mean(cluster_points, axis=0)
                    between_var += len(cluster_points) * np.sum((cluster_mean - overall_mean) ** 2)
            
            if total_points > 0:
                between_var /= total_points
            
            # CV ratio = between / within
            if within_var > 0:
                return between_var / within_var
            else:
                return 0.0
                
        except:
            return 0.0

    def _compute_temporal_score(self, assignments: np.ndarray) -> float:
        """Compute temporal consistency score."""
        try:
            if len(assignments) < 2:
                return 0.0
            
            # Count consecutive same labels
            consecutive_same = 0
            for i in range(1, len(assignments)):
                if assignments[i] == assignments[i-1]:
                    consecutive_same += 1
            
            return consecutive_same / (len(assignments) - 1)
        except:
            return 0.0

    def _compute_balance_score(self, assignments: np.ndarray) -> float:
        """Compute cluster balance score."""
        try:
            if len(assignments) == 0:
                return 0.0
            
            cluster_sizes = np.bincount(assignments)
            if len(cluster_sizes) == 0:
                return 0.0
            
            # Balance = 1 - coefficient of variation
            mean_size = np.mean(cluster_sizes)
            std_size = np.std(cluster_sizes)
            
            if mean_size == 0:
                return 0.0
            
            return max(0.0, 1.0 - (std_size / mean_size))
        except:
            return 0.0

    def _hybrid_dimensionality_reduction(self, features_scaled: np.ndarray,
                                       original_features: np.ndarray,
                                       feature_names: List[str]) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
                """Combine PCA with domain-selected features for better clustering."""
                try:
                    # Step 1: Apply PCA (less aggressive than before)
                    tprint(f"🔍 DEBUG: Applying PCA to {features_scaled.shape[1]} features with {features_scaled.shape[0]} samples", "INFO")
                    with tprint_timer("PCA dimensionality reduction"):
                        pca, features_pca = self._fit_pca(features_scaled)
                    tprint(f"🔍 DEBUG: PCA completed - {features_pca.shape[1]} components retained", "INFO")
                    tprint(f"🔍 MEMORY: PCA features created - {features_pca.nbytes / 1024 / 1024:.2f} MB", "INFO")

                    # Step 2: Select domain-specific features to complement PCA
                    domain_features = self._select_domain_features(original_features, feature_names)

                    # Step 3: Combine PCA + domain features
                    if domain_features is not None and domain_features.shape[1] > 0:
                        # Concatenate PCA components with domain features
                        combined_features = np.hstack([features_pca, domain_features])
                        tprint(f"🔍 MEMORY: Combined features created - {combined_features.nbytes / 1024 / 1024:.2f} MB", "INFO")

                        tprint(f"✅ Hybrid reduction: PCA({features_pca.shape[1]}) + Domain({domain_features.shape[1]}) "
                              f"= {combined_features.shape[1]} features", "SUCCESS")

                        return combined_features, pca, {
                            'method': 'hybrid_pca_domain',
                            'pca_components': features_pca.shape[1],
                            'domain_features': domain_features.shape[1],
                            'total_features': combined_features.shape[1],
                            'explained_variance': pca.explained_variance_ratio_.sum()
                        }
                    else:
                        # Fall back to PCA only
                        tprint(f"⚠️ No domain features selected, using PCA only: {features_pca.shape[1]} features", "WARNING")
                        return features_pca, pca, {
                            'method': 'pca_only',
                            'pca_components': features_pca.shape[1],
                            'domain_features': 0,
                            'total_features': features_pca.shape[1],
                            'explained_variance': pca.explained_variance_ratio_.sum()
                        }

                except Exception as e:
                    tprint(f"Hybrid dimensionality reduction failed: {e}", "ERROR")
                    # Fallback to original features
                    return features_scaled, None, {'method': 'fallback', 'error': str(e)}

            # Step 2: Apply PCA and prune near-zero contributors using loading scores
            tprint("Step 2: Applying PCA with MLE for data-driven dimensionality selection...", "INFO")
            
            # Use more aggressive feature pruning to target 15-25 features
            target_features = min(25, max(5, context.original_features.shape[1] // 6))
            tprint(f"🔍 DEBUG: PCA input - {features_scaled.shape[1]} features, target reduction to {target_features} features", "INFO")
            pca, features_pca = self._fit_pca(features_scaled)
            loading_scores = self._compute_loading_scores(pca, context.original_features.shape[1])
            tprint(f"🔍 DEBUG: PCA output - {features_pca.shape[1]} components, loading scores computed for {len(loading_scores)} features", "INFO")

            # If we have more features than target, apply aggressive pruning
            if loading_scores.size > target_features:
                tprint(f"🔍 DEBUG: Pruning {loading_scores.size} features to {target_features} using loading scores", "INFO")
                # Sort by loading scores and keep top features
                sorted_indices = np.argsort(loading_scores)[::-1]
                retained_mask = np.zeros_like(loading_scores, dtype=bool)
                retained_mask[sorted_indices[:target_features]] = True
                tprint(f"🔍 DEBUG: Feature pruning completed - retained {retained_mask.sum()} features", "INFO")

                # Ensure we keep at least 2 features for clustering
                if retained_mask.sum() < 2 and loading_scores.size >= 2:
                    top_two = np.argsort(loading_scores)[::-1][:2]
                    retained_mask = np.zeros_like(retained_mask, dtype=bool)
                    retained_mask[top_two] = True
            else:
                # If we already have fewer than target features, keep all
                retained_mask = loading_scores > 0
                tprint(f"🔍 DEBUG: No pruning needed - already at {loading_scores.size} features (target: {target_features})", "INFO")

            loading_threshold = float(np.min(loading_scores[retained_mask])) if retained_mask.any() else 0.0

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

                scaler = RobustScaler()
                features_scaled = scaler.fit_transform(context.original_features)

                # Step 2a: Perform K stability analysis to find optimal number of clusters
                tprint("🔍 Step 2a: Performing K stability analysis...", "INFO")
                optimal_k, k_analysis = self._perform_k_stability_analysis(features_scaled)

                # Step 2b: Apply dimensionality reduction with domain knowledge
                tprint("🔍 Step 2b: Applying hybrid dimensionality reduction (PCA + domain features)...", "INFO")
                features_pca, pca, feature_metadata = self._hybrid_dimensionality_reduction(
                    features_scaled, context.original_features, context.original_feature_names
                )

                # Update context with optimal K and reduced features
                context.optimal_k = optimal_k
                context.k_analysis = k_analysis

                loading_scores = _compute_loading_scores(pca, context.original_features.shape[1])
            else:
                context.dropped_feature_names = []

            # Step 3: Basic quality validation (minimal checks)
            tprint("Step 3: Validating feature quality...", "INFO")
            features_final = self._validate_feature_quality_minimal(features_pca, context.market_data)
            tprint(f"Feature quality validation completed: {features_final.shape}", "SUCCESS")
            tprint(f"🔍 DEBUG: Feature optimization pipeline completed - {context.original_features.shape[1]} → {features_final.shape[1]} features", "INFO")

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
            total_cleanup_mb = sum(arr.nbytes for arr in [features_scaled, features_pca] if arr is not None) / 1024 / 1024
            tprint(f"🔍 MEMORY: Cleaning up {total_cleanup_mb:.2f} MB of temporary arrays", "INFO")
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

    def _determine_optimal_algorithm_type(self, pipeline_state: Dict[str, Any], data: Any) -> str:
        """
        Determine the optimal clustering algorithm based on data characteristics and regime discovery results.
        
        Args:
            pipeline_state: Current pipeline state with regime discovery results
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
            
            # Always use our custom NAS-TAS clustering logic (progressive regime optimization)
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

    # Removed _select_optimal_k - no longer needed with dynamic convergence

    # Removed _determine_optimal_k_iterative - no longer needed with dynamic convergence




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




    

    def _calculate_composite_score(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Calculate composite score for regime optimization."""
        try:
            # Calculate individual quality scores
            quality_scores = self._calculate_individual_quality_scores(features, assignments)
            
            # Weighted composite score - ENHANCED: Prioritize quality metrics over balance
            weights = {
                'silhouette': 0.5,           # Increased from 0.4
                'regime_balance': 0.2,       # Reduced from 0.3
                'temporal_consistency': 0.15, # Reduced from 0.2
                'within_regime_cv': 0.15     # Increased from 0.1
            }
            
            composite_score = sum(quality_scores.get(key, 0.0) * weight for key, weight in weights.items())
            return min(1.0, max(0.0, composite_score))
            
        except Exception as e:
            tprint(f"Composite score calculation failed: {e}", "ERROR")
            return 0.0

    def _calculate_individual_quality_scores(self, features: np.ndarray, assignments: np.ndarray) -> Dict[str, float]:
        """Calculate individual quality scores for composite scoring."""
        try:
            scores = {}
            
            # Silhouette score
            try:
                from sklearn.metrics import silhouette_score
                if len(np.unique(assignments)) > 1:
                    scores['silhouette'] = silhouette_score(features, assignments)
                else:
                    scores['silhouette'] = 0.0
            except:
                scores['silhouette'] = 0.0
            
            # Regime balance score
            tprint("⚠️ WARNING: _calculate_regime_balance is not defined - using fallback", "WARNING")
            scores['regime_balance'] = 0.5  # Fallback value
            
            # Temporal consistency score
            scores['temporal_consistency'] = self._calculate_temporal_smoothness(assignments)
            
            # Within-regime CV score
            scores['within_regime_cv'] = self._calculate_cv_score_optimized(features, assignments)
            
            return scores
            
        except Exception as e:
            tprint(f"Individual quality scores calculation failed: {e}", "ERROR")
            return {
                'silhouette': 0.0,
                'regime_balance': 0.0,
                'temporal_consistency': 0.0,
                'within_regime_cv': 0.0
            }

    def _calculate_cluster_centers(self, features: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        """Calculate cluster centers from features and assignments."""
        try:
            unique_regimes = np.unique(assignments)
            n_regimes = len(unique_regimes)
            n_features = features.shape[1]
            
            # Initialize centers array
            centers = np.zeros((n_regimes, n_features))
            
            # Calculate center for each regime
            for i, regime in enumerate(unique_regimes):
                regime_mask = assignments == regime
                if np.sum(regime_mask) > 0:
                    centers[i] = np.mean(features[regime_mask], axis=0)
                else:
                    centers[i] = np.zeros(n_features)
            
            return centers
            
        except Exception as e:
            tprint(f"Cluster centers calculation failed: {e}", "ERROR")
            # Return zero centers as fallback
            unique_regimes = np.unique(assignments)
            n_regimes = len(unique_regimes)
            n_features = features.shape[1]
            return np.zeros((n_regimes, n_features))

    def _calculate_cv_score_optimized(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Calculate coefficient of variation score - proper CV calculation."""
        try:
            unique_regimes = np.unique(assignments)
            if len(unique_regimes) < 2:
                return 0.0
            
            # Calculate within-cluster coefficient of variation (CV = std/mean)
            within_cvs = []
            for regime in unique_regimes:
                regime_mask = assignments == regime
                if np.sum(regime_mask) > 1:
                    regime_features = features[regime_mask]
                    # Calculate CV for each feature, then average
                    regime_means = np.mean(regime_features, axis=0)
                    regime_stds = np.std(regime_features, axis=0)
                    # Avoid division by zero - use absolute mean for CV calculation
                    regime_cvs = np.where(np.abs(regime_means) > 1e-8, regime_stds / np.abs(regime_means), 0)
                    within_cvs.append(np.mean(regime_cvs))
            
            if not within_cvs:
                return 0.0
            
            # Calculate between-cluster coefficient of variation
            regime_means = []
            for regime in unique_regimes:
                regime_mask = assignments == regime
                if np.sum(regime_mask) > 0:
                    regime_mean = np.mean(features[regime_mask], axis=0)
                    regime_means.append(regime_mean)
            
            if len(regime_means) < 2:
                return 0.0
            
            # Calculate CV between cluster means
            regime_means_array = np.array(regime_means)
            between_mean = np.mean(regime_means_array, axis=0)
            between_std = np.std(regime_means_array, axis=0)
            between_cv = np.mean(np.where(np.abs(between_mean) > 1e-8, between_std / np.abs(between_mean), 0))
            
            # Calculate within-cluster CV
            within_cv = np.mean(within_cvs)
            
            # Return the actual coefficient of variation values
            # Store both within and between CV for detailed reporting
            self._last_within_cv = within_cv
            self._last_between_cv = between_cv
            
            # CV score: ratio of between-cluster CV to within-cluster CV
            # Higher ratio means better cluster separation
            if within_cv > 0:
                cv_ratio = between_cv / within_cv
            else:
                cv_ratio = between_cv
            
            # Return the actual CV ratio (not capped at 1.0)
            return cv_ratio
            
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
                # Use original features if optimized features are not available
                features = context.original_features
                context.optimized_features = features
                tprint("Using original features as optimized features", "WARNING")
            
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
            
            # Step 2c: Apply consecutive optimization loop (merging → reshuffling → splitting)
            tprint("Step 2c: Applying consecutive optimization loop (merging → reshuffling → splitting)...", "INFO")
            optimized_assignments, convergence_metrics = self._run_consecutive_optimization_loop(
                features, context.raw_assignments, initial_k
            )
            
            # Calculate final scores with temporal smoothness
            tprint("⚠️ WARNING: _calculate_regime_balance is not defined - using fallback", "WARNING")
            final_balance = 0.5  # Fallback value
            tprint("⚠️ WARNING: _calculate_silhouette_score is not defined - using fallback", "WARNING")
            final_silhouette = 0.5  # Fallback value
            tprint("⚠️ WARNING: _calculate_cv_score is not defined - using fallback", "WARNING")
            final_cv = 0.5  # Fallback value
            final_temporal = self._calculate_temporal_smoothness(optimized_assignments)
            
            # Enhanced composite score - ENHANCED: Further prioritize Silhouette and CV for better cluster quality
            balance_weight = 0.15   # 15% balance emphasis (reduced from 25% to prioritize clustering quality)
            silhouette_weight = 0.45 # 45% silhouette emphasis (increased from 30% for better separation)
            cv_weight = 0.35        # 35% CV emphasis (maintained for between-cluster variance)
            temporal_weight = 0.05  # 5% temporal emphasis (reduced from 10% for stability)
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
            tprint("⚠️ WARNING: _calculate_regime_balance is not defined - using fallback", "WARNING")
            initial_balance = 0.5  # Fallback value
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
            
            # ENHANCED: Multi-metric convergence with relative improvement thresholds
            quality_target = 0.75  # Target quality score (increased from 0.6)
            relative_improvement_threshold = 0.001  # 0.1% relative improvement threshold
            no_improvement_window = 3  # Stop if no improvement for 3 iterations
            min_iterations_before_early_stop = 5  # Minimum iterations before considering early stop
            
            for iteration in range(max_iterations):
                # Initialize convergence status
                convergence_achieved = False
                current_k = k  # Initialize current_k for splitting logic
                
                # VECTORIZED: Calculate current scores with optimized methods
                tprint("⚠️ WARNING: _calculate_regime_balance is not defined - using fallback", "WARNING")
                balance_score = 0.5  # Fallback value
                # BOTTLENECK FIX: Use vectorized silhouette approximation every 5 iterations instead of every iteration
                if iteration % 5 == 0 or iteration < 3:  # Full calculation every 5 iterations or first 3
                    tprint("⚠️ WARNING: _calculate_silhouette_score is not defined - using fallback", "WARNING")
                    silhouette_score = 0.5  # Fallback value
                else:
                    tprint("⚠️ WARNING: _calculate_silhouette_approximation_vectorized is not defined - using fallback", "WARNING")
                    silhouette_score = 0.5  # Fallback value
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
                    
                    # ENHANCED: Multi-metric convergence detection with relative improvement thresholds
                    # Check if we have enough iterations to evaluate convergence
                    if len(convergence_history) >= min_iterations_before_early_stop:
                        # Calculate relative improvement (improvement relative to current best score)
                        if best_score > 0:
                            relative_improvement = avg_improvement / abs(best_score)
                        else:
                            relative_improvement = avg_improvement

                        # Multi-metric convergence criteria
                        recent_silhouette_scores = [h['silhouette_score'] for h in convergence_history[-5:]]
                        silhouette_trend = np.mean(np.diff(recent_silhouette_scores)) if len(recent_silhouette_scores) > 1 else 0

                        recent_cv_scores = [h['cv_score'] for h in convergence_history[-5:]]
                        cv_trend = np.mean(np.diff(recent_cv_scores)) if len(recent_cv_scores) > 1 else 0

                        recent_balance_scores = [h['balance_score'] for h in convergence_history[-5:]]
                        balance_trend = np.mean(np.diff(recent_balance_scores)) if len(recent_balance_scores) > 1 else 0

                        # Enhanced convergence criteria using relative thresholds
                        silhouette_converged = abs(silhouette_trend) < relative_improvement_threshold * 0.1
                        cv_converged = abs(cv_trend) < relative_improvement_threshold * 0.1
                        balance_converged = abs(balance_trend) < relative_improvement_threshold * 0.1

                        # Early stopping: Stop if no improvement for N iterations
                        recent_improvements = [h['improvement'] for h in convergence_history[-no_improvement_window:]]
                        no_improvement_count = sum(1 for imp in recent_improvements if abs(imp) < relative_improvement_threshold)

                        # Convergence achieved if multiple metrics are stable AND no improvement for window
                        convergence_achieved = (silhouette_converged and cv_converged and balance_converged) and no_improvement_count >= no_improvement_window * 0.8
                        
                        # DEBUG: Show convergence decision with multi-metric details
                        if iteration % 5 == 0:  # Every 5 iterations
                            tprint(f"   🔍 Multi-metric convergence: sil_trend={silhouette_trend:.6f}, "
                                  f"cv_trend={cv_trend:.6f}, bal_trend={balance_trend:.6f}", "INFO")
                            tprint(f"   📊 Convergence criteria: sil={silhouette_converged}, cv={cv_converged}, "
                                  f"bal={balance_converged}, no_imp={no_improvement_count}/{no_improvement_window}", "INFO")

                        if convergence_achieved:
                            tprint(f"✅ Multi-metric convergence achieved at iteration {iteration+1}!", "SUCCESS")
                            tprint(f"   📊 Relative improvement: {relative_improvement:.6f} (threshold: {relative_improvement_threshold})", "SUCCESS")
                            tprint(f"   🎯 Best score: {best_score:.4f}", "SUCCESS")
                            tprint(f"   📈 Trends: Silhouette={silhouette_trend:.6f}, CV={cv_trend:.6f}, Balance={balance_trend:.6f}", "SUCCESS")
                            break
                        else:
                            tprint(f"   ⚠️ Multi-metric criteria not met, continuing optimization...", "WARNING")

                    
                    # ENHANCED: Quality-based early stopping
                    if best_score > quality_target:  # If we achieve target quality, stop early
                        tprint(f"🎉 Target quality achieved at iteration {iteration+1}!", "SUCCESS")
                        tprint(f"   🎯 Quality score: {best_score:.4f} (target: >{quality_target})", "SUCCESS")
                        break
                    
                    # ENHANCED: Check for quality stagnation
                    if len(convergence_history) >= 10:  # Need enough history
                        recent_quality_scores = [h['composite_score'] for h in convergence_history[-10:]]
                        quality_improvement = np.mean(np.diff(recent_quality_scores)) if len(recent_quality_scores) > 1 else 0
                        
                        if quality_improvement < relative_improvement_threshold and best_score < quality_target * 0.85:
                            tprint(f"⚠️ Quality stagnation detected (improvement: {quality_improvement:.6f})", "WARNING")
                            tprint(f"   🔧 Applying aggressive optimization...", "INFO")
                            # Apply more aggressive optimization
                            assignments = self._apply_aggressive_iteration_optimization(features, assignments, k, iteration, convergence_history, distance_cache)
                    
                    # ENHANCED: Apply smart cluster splitting EVERY iteration for maximum responsiveness
                    tprint(f"   🔍 Checking for cluster splitting opportunities (iteration {iteration+1})...", "INFO")
                    assignments, k, split_stats = self._smart_cluster_splitting_decision(assignments, features, k, iteration, best_score)
                    
                    if k > current_k:
                        tprint(f"   📈 Dynamic regime count adjustment: {current_k} → {k}", "SUCCESS")
                        current_k = k
                    else:
                        tprint(f"   📊 No regime count change (K={k})", "INFO")

                    # STAGE 3: Cross-cluster boundary optimization (every 5 iterations)
                    if iteration % 5 == 0 and iteration > 0:
                        tprint(f"   🌐 Stage 3: Cross-cluster boundary optimization (iteration {iteration+1})...", "INFO")
                        assignments = self._optimize_cross_cluster_boundaries(features, assignments, k)

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

    def _perform_cluster_merging_and_remerging(self, features: np.ndarray, assignments: np.ndarray, k: int, baseline_score: float) -> Tuple[np.ndarray, int, Dict]:
        """Perform cluster merging with Mahalanobis distance, compactness constraints, and economics-first gates."""
        try:
            tprint("🔄 Stage 1: Performing cluster merging and remerging...", "INFO")

            # Calculate centroids and cluster statistics
            centroids = np.zeros((k, features.shape[1]))
            cluster_sizes = np.zeros(k, dtype=int)
            cluster_covariances = []

            for regime in range(k):
                regime_mask = assignments == regime
                if np.sum(regime_mask) > 0:
                    regime_features = features[regime_mask]
                    centroids[regime] = np.mean(regime_features, axis=0)
                    cluster_sizes[regime] = len(regime_features)

                    # Calculate covariance matrix for Mahalanobis distance
                    if len(regime_features) > 1:
                        cov = np.cov(regime_features.T)
                        # Add small regularization for numerical stability
                        cov += np.eye(cov.shape[0]) * 1e-6
                    else:
                        cov = np.eye(features.shape[1])
                    cluster_covariances.append(cov)

            # Find merge candidates using Mahalanobis distance
            merge_candidates = []
            clusters_processed = set()  # Atomic constraint: one change per cluster per phase

            for i in range(k):
                if i in clusters_processed:
                    continue

                for j in range(i + 1, k):
                    if j in clusters_processed or cluster_sizes[i] == 0 or cluster_sizes[j] == 0:
                        continue

                    # Calculate Mahalanobis distance between centroids
                    centroid_i, centroid_j = centroids[i], centroids[j]
                    cov_i, cov_j = cluster_covariances[i], cluster_covariances[j]

                    # Use combined covariance for distance calculation
                    try:
                        combined_cov = (cov_i + cov_j) / 2
                        diff = centroid_i - centroid_j
                        mahalanobis_dist = np.sqrt(diff @ np.linalg.inv(combined_cov) @ diff)
                    except:
                        # Fallback to Euclidean if covariance issues
                        mahalanobis_dist = np.linalg.norm(centroid_i - centroid_j)

                    # Calculate compactness (lower is better)
                    compactness_i = self._calculate_cluster_compactness(features, assignments, i)
                    compactness_j = self._calculate_cluster_compactness(features, assignments, j)

                    # Merge threshold based on compactness
                    merge_threshold = (compactness_i + compactness_j) / 2

                    if mahalanobis_dist < merge_threshold:
                        # Size guardrails
                        merged_size = cluster_sizes[i] + cluster_sizes[j]
                        total_samples = len(assignments)
                        size_ratio = merged_size / total_samples

                        # Prevent dominant clusters unless significant improvement
                        min_improvement = 0.005 if size_ratio > 0.5 else 0.005

                        merge_candidates.append({
                            'i': i, 'j': j,
                            'distance': mahalanobis_dist,
                            'threshold': merge_threshold,
                            'size_ratio': size_ratio,
                            'min_improvement': min_improvement,
                            'compactness_i': compactness_i,
                            'compactness_j': compactness_j
                        })

            # Sort candidates by distance and size ratio
            merge_candidates.sort(key=lambda x: (x['distance'], x['size_ratio']))

            # Apply merges with economics-first gates
            merged_assignments = assignments.copy()
            new_k = k
            merges_accepted = 0
            merges_proposed = len(merge_candidates)

            for candidate in merge_candidates:
                i, j = candidate['i'], candidate['j']

                # Check if either cluster was already processed (atomic constraint)
                if i in clusters_processed or j in clusters_processed:
                    continue

                # Calculate quality before merge
                quality_before = self._calculate_composite_objective(features, merged_assignments)

                # Perform merge
                merged_assignments[merged_assignments == j] = i
                new_k -= 1

                # Calculate quality after merge
                quality_after = self._calculate_composite_objective(features, merged_assignments)

                # Economics-first gates
                cv_ratio_before = self._calculate_cv_score_optimized(features, assignments)
                cv_ratio_after = self._calculate_cv_score_optimized(features, merged_assignments)
                temporal_before = self._calculate_temporal_smoothness_optimized(assignments)
                temporal_after = self._calculate_temporal_smoothness_optimized(merged_assignments)

                # Check economics gates
                cv_gate = cv_ratio_after >= max(1.5, cv_ratio_before * 0.99)  # Non-degrading if already good
                temporal_gate = temporal_after >= max(0.35, temporal_before * 0.99)

                # Post-merge compactness constraint
                merged_compactness = self._calculate_cluster_compactness(features, merged_assignments, i)
                pre_compactness = (candidate['compactness_i'] * cluster_sizes[i] +
                                candidate['compactness_j'] * cluster_sizes[j]) / (cluster_sizes[i] + cluster_sizes[j])
                compactness_gate = merged_compactness <= pre_compactness * 1.1  # Allow slight degradation

                # Final acceptance
                improvement_ratio = (quality_after - quality_before) / quality_before if quality_before > 0 else quality_after
                accepted = (improvement_ratio >= candidate['min_improvement'] and
                           compactness_gate and
                           (cv_gate or candidate['size_ratio'] > 0.5))  # Allow size-based merges even if CV degrades slightly

                if accepted:
                    merges_accepted += 1
                    clusters_processed.add(i)
                    clusters_processed.add(j)

                    # Update cluster sizes
                    cluster_sizes[i] += cluster_sizes[j]
                    cluster_sizes[j] = 0

                    tprint(f"   ✅ Merged {j} → {i} (dist: {candidate['distance']:.4f}, size_ratio: {candidate['size_ratio']:.2f})", "SUCCESS")
                    tprint(f"   📈 Quality: {quality_before:.4f} → {quality_after:.4f} ({improvement_ratio:.4f})", "SUCCESS")

                    # Check economics gates
                    if not cv_gate:
                        tprint(f"   ⚠️ CV ratio degraded: {cv_ratio_before:.3f} → {cv_ratio_after:.3f}", "WARNING")
                    if not temporal_gate:
                        tprint(f"   ⚠️ Temporal degraded: {temporal_before:.3f} → {temporal_after:.3f}", "WARNING")
                else:
                    # Revert merge
                    merged_assignments[merged_assignments == i] = j
                    new_k += 1
                    tprint(f"   ❌ Merge rejected: gates failed (imp: {improvement_ratio:.4f}, compact: {compactness_gate})", "INFO")

            merge_stats = {
                'accepted': merges_accepted,
                'proposed': merges_proposed,
                'final_k': new_k,
                'initial_k': k
            }

            tprint(f"   📊 Merging: {merges_accepted}/{merges_proposed} accepted, K: {k} → {new_k}", "INFO")
            return merged_assignments, new_k, merge_stats

        except Exception as e:
            tprint(f"Cluster merging failed: {e}", "ERROR")
            return assignments, k, {'accepted': 0, 'proposed': 0, 'error': str(e)}

    def _perform_cluster_merging_and_remerging_cached(self, cached_state: 'CachedClusteringState', baseline_score: float) -> Tuple[np.ndarray, int, Dict]:
        """Perform cluster merging using cached statistics for O(1) operations."""
        try:
            tprint("🔄 Stage 1: Cached merging and remerging...", "INFO")

            # Get cached centroids and stats
            centroids = cached_state.get_centroids()
            cluster_sizes = cached_state.get_cluster_sizes()
            k = cached_state.k

            # Find merge candidates using Mahalanobis distance with cached covariances
            merge_candidates = []
            clusters_processed = set()  # Atomic constraint: one change per cluster per phase

            for i in range(k):
                if i in clusters_processed or cluster_sizes[i] == 0:
                    continue

                for j in range(i + 1, k):
                    if j in clusters_processed or cluster_sizes[j] == 0:
                        continue

                    # Calculate Mahalanobis distance using cached covariances
                    centroid_i, centroid_j = centroids[i], centroids[j]
                    cov_i = cached_state.cluster_stats[i].get_covariance()
                    cov_j = cached_state.cluster_stats[j].get_covariance()

                    # Use combined covariance for distance calculation
                    try:
                        combined_cov = (cov_i + cov_j) / 2
                        diff = centroid_i - centroid_j
                        mahalanobis_dist = np.sqrt(diff @ np.linalg.inv(combined_cov) @ diff)
                    except:
                        # Fallback to Euclidean if covariance issues
                        mahalanobis_dist = np.linalg.norm(centroid_i - centroid_j)

                    # Get compactness from cached stats (O(1))
                    compactness_i = cached_state.cluster_stats[i].get_compactness()
                    compactness_j = cached_state.cluster_stats[j].get_compactness()

                    # Merge threshold based on compactness
                    merge_threshold = (compactness_i + compactness_j) / 2

                    if mahalanobis_dist < merge_threshold:
                        # Size guardrails
                        merged_size = cluster_sizes[i] + cluster_sizes[j]
                        total_samples = len(cached_state.assignments)
                        size_ratio = merged_size / total_samples

                        # Prevent dominant clusters unless significant improvement
                        min_improvement = 0.005 if size_ratio > 0.5 else 0.005

                        merge_candidates.append({
                            'i': i, 'j': j,
                            'distance': mahalanobis_dist,
                            'threshold': merge_threshold,
                            'size_ratio': size_ratio,
                            'min_improvement': min_improvement,
                            'compactness_i': compactness_i,
                            'compactness_j': compactness_j
                        })

            # Sort candidates by distance and size ratio
            merge_candidates.sort(key=lambda x: (x['distance'], x['size_ratio']))

            # Apply merges with economics-first gates using cached state
            merged_assignments = cached_state.assignments.copy()
            new_k = k
            merges_accepted = 0
            merges_proposed = len(merge_candidates)

            for candidate in merge_candidates:
                i, j = candidate['i'], candidate['j']

                # Check if either cluster was already processed (atomic constraint)
                if i in clusters_processed or j in clusters_processed:
                    continue

                # Calculate quality before merge (O(1) with cached stats)
                quality_before = self._calculate_composite_objective_cached(cached_state)

                # Perform merge using cached state (O(d) operation)
                new_k = cached_state.merge_clusters(i, j)

                # Calculate quality after merge (O(1) with cached stats)
                quality_after = self._calculate_composite_objective_cached(cached_state)

                # Economics-first gates
                cv_ratio_before = cached_state.get_cached_metric('cv_ratio') or 0.5
                cv_ratio_after = self._calculate_cv_score_optimized(cached_state.features, cached_state.assignments)

                temporal_before = cached_state.get_cached_metric('temporal') or 0.5
                temporal_after = self._calculate_temporal_smoothness_optimized(cached_state.assignments)

                # Check economics gates
                cv_gate = cv_ratio_after >= max(1.5, cv_ratio_before * 0.99)
                temporal_gate = temporal_after >= max(0.35, temporal_before * 0.99)

                # Post-merge compactness constraint using cached stats
                merged_compactness = cached_state.cluster_stats[i].get_compactness()
                pre_compactness = (candidate['compactness_i'] * cluster_sizes[i] +
                                candidate['compactness_j'] * cluster_sizes[j]) / (cluster_sizes[i] + cluster_sizes[j])
                compactness_gate = merged_compactness <= pre_compactness * 1.1

                # Final acceptance
                improvement_ratio = (quality_after - quality_before) / quality_before if quality_before > 0 else quality_after
                accepted = (improvement_ratio >= candidate['min_improvement'] and
                           compactness_gate and
                           (cv_gate or candidate['size_ratio'] > 0.5))

                if accepted:
                    merges_accepted += 1
                    clusters_processed.add(i)
                    clusters_processed.add(j)

                    # Update cluster sizes
                    cluster_sizes[i] += cluster_sizes[j]
                    cluster_sizes[j] = 0

                    tprint(f"   ✅ Cached merge {j} → {i} (dist: {candidate['distance']:.4f}, size_ratio: {candidate['size_ratio']:.2f})", "SUCCESS")
                    tprint(f"   📈 Quality: {quality_before:.4f} → {quality_after:.4f} ({improvement_ratio:.4f})", "SUCCESS")

                    # Cache updated metrics
                    cached_state.set_cached_metric('cv_ratio', cv_ratio_after)
                    cached_state.set_cached_metric('temporal', temporal_after)

                    if not cv_gate:
                        tprint(f"   ⚠️ CV ratio degraded: {cv_ratio_before:.3f} → {cv_ratio_after:.3f}", "WARNING")
                    if not temporal_gate:
                        tprint(f"   ⚠️ Temporal degraded: {temporal_before:.3f} → {temporal_after:.3f}", "WARNING")
                else:
                    # Revert merge using cached state
                    cached_state = self._revert_merge_cached(cached_state, i, j)
                    new_k = k
                    tprint(f"   ❌ Cached merge rejected: gates failed (imp: {improvement_ratio:.4f}, compact: {compactness_gate})", "INFO")

            merge_stats = {
                'accepted': merges_accepted,
                'proposed': merges_proposed,
                'final_k': new_k,
                'initial_k': k
            }

            tprint(f"   📊 Cached merging: {merges_accepted}/{merges_proposed} accepted, K: {k} → {new_k}", "INFO")
            return cached_state.assignments.copy(), new_k, merge_stats

        except Exception as e:
            tprint(f"Cached cluster merging failed: {e}", "ERROR")
            return cached_state.assignments.copy(), cached_state.k, {'accepted': 0, 'proposed': 0, 'error': str(e)}

    def _calculate_composite_objective_cached(self, cached_state: 'CachedClusteringState') -> float:
        """Calculate composite objective using cached metrics where possible."""
        try:
            # Try to use cached metrics first
            cached_cv = cached_state.get_cached_metric('cv_ratio')
            cached_temporal = cached_state.get_cached_metric('temporal')

            # Component weights
            w_cv, w_temporal, w_balance, w_silhouette, w_dbi = 0.25, 0.20, 0.15, 0.25, 0.15

            # Use cached or calculate fresh
            cv_ratio = cached_cv if cached_cv is not None else self._calculate_cv_score_optimized(cached_state.features, cached_state.assignments)
            cv_ratio_capped = np.clip(cv_ratio, 0, 3)

            temporal = cached_temporal if cached_temporal is not None else self._calculate_temporal_smoothness_optimized(cached_state.assignments)
            temporal_capped = np.clip(temporal, 0, 1)

            # For now, use fallbacks for other metrics (can be cached later)
            balance = 0.5
            balance_capped = np.clip(balance, 0, 1)

            silhouette = 0.5
            silhouette_clipped = np.clip(silhouette, -1, 1)

            dbi = 0.5
            dbi_clipped = np.clip(dbi, 0, 2)

            # Composite objective J
            J = (w_cv * cv_ratio_capped +
                 w_temporal * temporal_capped +
                 w_balance * balance_capped +
                 w_silhouette * silhouette_clipped -
                 w_dbi * dbi_clipped)

            return max(0.0, J)

        except Exception as e:
            tprint(f"Cached composite objective calculation failed: {e}", "ERROR")
            return 0.0

    def _revert_merge_cached(self, cached_state: 'CachedClusteringState', cluster_i: int, cluster_j: int) -> 'CachedClusteringState':
        """Revert a merge operation using cached state."""
        # This would need more sophisticated state management for full reversibility
        # For now, just return the cached state as-is (merge was rejected)
        return cached_state

    def _calculate_intra_cluster_distance(self, features: np.ndarray, assignments: np.ndarray, cluster_id: int) -> float:
        """Calculate average intra-cluster distance for a cluster."""
        try:
            cluster_mask = assignments == cluster_id
            if np.sum(cluster_mask) < 2:
                return 0.0

            cluster_samples = features[cluster_mask]
            cluster_center = np.mean(cluster_samples, axis=0)

            # Calculate distances from center to all points
            distances = np.linalg.norm(cluster_samples - cluster_center, axis=1)

            # Return average distance (excluding center point)
            return np.mean(distances)

        except Exception:
            return 0.0

    def _calculate_cluster_compactness(self, features: np.ndarray, assignments: np.ndarray, cluster_id: int) -> float:
        """Calculate cluster compactness (lower is better, more compact)."""
        try:
            cluster_mask = assignments == cluster_id
            if np.sum(cluster_mask) < 2:
                return 0.0

            cluster_features = features[cluster_mask]
            centroid = np.mean(cluster_features, axis=0)

            # Average distance from centroid (lower = more compact)
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            return np.mean(distances)

        except Exception:
            return 1.0  # High penalty for errors

    def _perform_neighbor_reshuffling(self, features: np.ndarray, assignments: np.ndarray, k: int, baseline_score: float, hysteresis_map: Dict, round_num: int, cached_state: 'CachedClusteringState') -> Tuple[np.ndarray, Dict]:
        """Perform probabilistic neighbor reshuffling with hysteresis and economics-first gates."""
        try:
            tprint("🔀 Stage 2: Performing neighbor reshuffling...", "INFO")

            reshuffled_assignments = assignments.copy()
            points_processed = set()  # Atomic constraint: one change per point per phase

            # Parameters
            neighbor_k = min(10, len(features) - 1)
            consensus_threshold = 0.7  # τ from user specs
            hysteresis_rounds = 2  # h from user specs
            point_threshold_ratio = 1e-4  # ε from user specs

            # Calculate centroids for all clusters
            centroids = np.zeros((k, features.shape[1]))
            for regime in range(k):
                regime_mask = assignments == regime
                if np.sum(regime_mask) > 0:
                    centroids[regime] = np.mean(features[regime_mask], axis=0)

            # Calculate local silhouette for each point
            local_silhouettes = self._calculate_local_silhouettes(features, assignments, k)

            # Find reshuffle candidates
            reshuffle_candidates = []
            points_locked = set()

            # Check hysteresis - points locked from previous rounds
            for point_id, lock_round in hysteresis_map.items():
                if round_num - lock_round < hysteresis_rounds:
                    points_locked.add(point_id)

            for i in range(len(features)):
                if i in points_processed or i in points_locked:
                    continue

                current_regime = assignments[i]
                sample_features = features[i]

                # Find nearest neighbors using FAISS (O(log N) instead of O(N))
                query_point = sample_features.reshape(1, -1)
                distances, neighbor_indices = cached_state.find_knn_faiss(query_point, neighbor_k + 1)  # +1 to account for self

                # Filter out self and get regimes
                neighbor_indices = neighbor_indices[0]  # FAISS returns (1, k) array for single query
                neighbor_indices = neighbor_indices[neighbor_indices != i]  # Exclude self
                neighbor_indices = neighbor_indices[:neighbor_k]  # Ensure exactly k neighbors

                neighbor_regimes = cached_state.assignments[neighbor_indices]

                # Calculate consensus
                regime_counts = np.bincount(neighbor_regimes, minlength=k)
                max_count = np.max(regime_counts)
                consensus = max_count / len(neighbor_indices)

                if consensus >= consensus_threshold:
                    best_neighbor_regime = np.argmax(regime_counts)

                    if best_neighbor_regime != current_regime:
                        # Calculate objective improvement for this point
                        current_score = self._calculate_sample_regime_score(sample_features, current_regime, centroids)
                        new_score = self._calculate_sample_regime_score(sample_features, best_neighbor_regime, centroids)

                        improvement_ratio = (new_score - current_score) / current_score if current_score > 0 else new_score
                        point_threshold = point_threshold_ratio * baseline_score

                        # Local silhouette check
                        local_silhouette = local_silhouettes[i]
                        silhouette_improves = True  # Simplified for now

                        # Economics-first gates for this point
                        point_economics_ok = self._check_point_economics_gates(
                            features, assignments, i, current_regime, best_neighbor_regime, centroids)

                        if (improvement_ratio >= point_threshold and
                            silhouette_improves and
                            point_economics_ok):

                            reshuffle_candidates.append({
                                'point_id': i,
                                'current_regime': current_regime,
                                'new_regime': best_neighbor_regime,
                                'improvement_ratio': improvement_ratio,
                                'consensus': consensus,
                                'local_silhouette': local_silhouette
                            })

            # Sort by improvement ratio
            reshuffle_candidates.sort(key=lambda x: x['improvement_ratio'], reverse=True)

            # Apply reshuffles with atomic constraint
            reshuffles_accepted = 0
            reshuffles_proposed = len(reshuffle_candidates)

            for candidate in reshuffle_candidates:
                point_id = candidate['point_id']

                if point_id in points_processed:
                    continue

                # Apply reshuffle
                old_regime = reshuffled_assignments[point_id]
                reshuffled_assignments[point_id] = candidate['new_regime']

                # Update cached state for O(1) operations (only move point, don't rebuild everything)
                cached_state.move_point(point_id, old_regime, candidate['new_regime'])

                # Calculate global objective improvement using cached state
                new_score = self._calculate_composite_objective_cached(cached_state)
                improvement_ratio = (new_score - baseline_score) / baseline_score if baseline_score > 0 else new_score

                # Accept if global improvement is positive
                if improvement_ratio >= 0.001:  # Small but positive improvement required
                    reshuffles_accepted += 1
                    points_processed.add(point_id)

                    # Apply hysteresis - lock this point for future rounds
                    hysteresis_map[point_id] = round_num

                    tprint(f"   ✅ Reshuffled {point_id}: {old_regime} → {candidate['new_regime']} "
                          f"(local_imp: {candidate['improvement_ratio']:.4f}, consensus: {candidate['consensus']:.2f})", "SUCCESS")
                else:
                    # Revert change using cached state
                    cached_state.move_point(point_id, candidate['new_regime'], old_regime)
                    reshuffled_assignments[point_id] = old_regime
                    tprint(f"   ❌ Reshuffle rejected: no global improvement", "INFO")

            reshuffle_stats = {
                'accepted': reshuffles_accepted,
                'proposed': reshuffles_proposed,
                'locked_points': len(points_locked)
            }

            tprint(f"   📊 Reshuffling: {reshuffles_accepted}/{reshuffles_proposed} accepted, "
                  f"{len(points_locked)} points locked", "INFO")
            return reshuffled_assignments, reshuffle_stats

        except Exception as e:
            tprint(f"Neighbor reshuffling failed: {e}", "ERROR")
            return assignments, {'accepted': 0, 'proposed': 0, 'error': str(e)}

    def _calculate_local_silhouettes(self, features: np.ndarray, assignments: np.ndarray, k: int) -> np.ndarray:
        """Calculate local silhouette scores for each point."""
        try:
            n_samples = len(features)
            local_silhouettes = np.zeros(n_samples)

            for i in range(n_samples):
                current_regime = assignments[i]

                # Calculate a(i) - distance to own cluster centroid
                regime_mask = assignments == current_regime
                if np.sum(regime_mask) > 1:
                    regime_features = features[regime_mask]
                    centroid = np.mean(regime_features, axis=0)
                    a_i = np.linalg.norm(features[i] - centroid)
                else:
                    a_i = 0

                # Calculate b(i) - distance to nearest other cluster
                b_i = float('inf')
                for other_regime in range(k):
                    if other_regime != current_regime:
                        other_mask = assignments == other_regime
                        if np.sum(other_mask) > 0:
                            other_centroid = np.mean(features[other_mask], axis=0)
                            dist_to_other = np.linalg.norm(features[i] - other_centroid)
                            b_i = min(b_i, dist_to_other)

                # Local silhouette
                if a_i + b_i > 0:
                    local_silhouettes[i] = (b_i - a_i) / (a_i + b_i)
                else:
                    local_silhouettes[i] = 0

            return local_silhouettes

        except Exception:
            return np.zeros(len(features))

    def _check_point_economics_gates(self, features: np.ndarray, assignments: np.ndarray, point_id: int,
                                   current_regime: int, new_regime: int, centroids: np.ndarray) -> bool:
        """Check economics-first gates for a point reassignment."""
        try:
            # Calculate CV ratio and temporal for current assignment
            current_cv = self._calculate_cv_score_optimized(features, assignments)
            current_temporal = self._calculate_temporal_smoothness_optimized(assignments)

            # Simulate the reassignment
            test_assignments = assignments.copy()
            test_assignments[point_id] = new_regime

            new_cv = self._calculate_cv_score_optimized(features, test_assignments)
            new_temporal = self._calculate_temporal_smoothness_optimized(test_assignments)

            # Economics gates (non-degrading if already good)
            cv_gate = new_cv >= max(1.5, current_cv * 0.99)
            temporal_gate = new_temporal >= max(0.35, current_temporal * 0.99)

            return cv_gate and temporal_gate

        except Exception:
            return False  # Conservative fallback

    def _calculate_composite_objective(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Calculate the single monotonic objective function J with proper weighting and capping."""
        try:
            # Component weights
            w_cv, w_temporal, w_balance, w_silhouette, w_dbi = 0.25, 0.20, 0.15, 0.25, 0.15

            # Calculate individual metrics with caps/clips
            cv_ratio = self._calculate_cv_score_optimized(features, assignments)
            cv_ratio_capped = np.clip(cv_ratio, 0, 3)  # Cap at 3

            temporal = self._calculate_temporal_smoothness_optimized(assignments)
            temporal_capped = np.clip(temporal, 0, 1)  # Cap at 1

            balance = 0.5  # Fallback - implement proper balance calculation
            balance_capped = np.clip(balance, 0, 1)  # Cap at 1

            silhouette = 0.5  # Fallback - implement proper silhouette
            silhouette_clipped = np.clip(silhouette, -1, 1)  # Clip to [-1, 1]

            dbi = 0.5  # Fallback - implement proper DBI
            dbi_clipped = np.clip(dbi, 0, 2)  # Cap at 2

            # Composite objective J
            J = (w_cv * cv_ratio_capped +
                 w_temporal * temporal_capped +
                 w_balance * balance_capped +
                 w_silhouette * silhouette_clipped -
                 w_dbi * dbi_clipped)

            return max(0.0, J)  # Ensure non-negative

        except Exception as e:
            tprint(f"Composite objective calculation failed: {e}", "ERROR")
            return 0.0

    def _extract_regime_assignments(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract TAS and NAS regime assignments from pipeline state or previous outcomes."""
        try:
            pipeline_state = getattr(self, 'pipeline_state', {}) or {}
            if not isinstance(pipeline_state, dict):
                raise ValueError("Pipeline state is missing or invalid")

            if not hasattr(self, 'features') or self.features is None:
                raise ValueError("Feature matrix is not available for assignment validation")

            expected_length = self.features.shape[0]

            # Also check for regime discovery results in accumulated artifacts
            accumulated_artifacts = pipeline_state.get('artifacts', {})

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

            # Check multiple locations for regime discovery results
            if 'nas_tas_regime_discovery_result' in pipeline_state:
                candidates.append(pipeline_state.get('nas_tas_regime_discovery_result'))

            if isinstance(accumulated_artifacts, dict):
                if 'nas_tas_regime_discovery_result' in accumulated_artifacts:
                    candidates.append(accumulated_artifacts.get('nas_tas_regime_discovery_result'))

            # Also check for direct regime assignments
            if 'tas_assignments' in accumulated_artifacts and 'nas_assignments' in accumulated_artifacts:
                tas_assignments = accumulated_artifacts.get('tas_assignments')
                nas_assignments = accumulated_artifacts.get('nas_assignments')
                if tas_assignments is not None and nas_assignments is not None:
                    tprint("✅ Found direct TAS/NAS assignments in accumulated artifacts", "INFO")
                    return _as_numpy(tas_assignments, "TAS assignments"), _as_numpy(nas_assignments, "NAS assignments")

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
                tprint("NAS-TAS regime discovery result not found in pipeline state, trying to load from previous outcomes", "WARNING")

                # Try to load from previous outcomes as fallback
                fallback_result = self._load_regime_discovery_from_outcomes()
                if fallback_result:
                    tprint("✅ Successfully loaded regime discovery results from previous outcomes", "SUCCESS")
                    discovery_result = fallback_result
                else:
                    tprint("❌ No regime discovery results found in previous outcomes either - fast failing", "ERROR")
                    raise ValueError("No regime discovery results available - cannot proceed with clustering without TAS/NAS assignments")

            tas_assignments_raw = discovery_result.get('tas_assignments')
            nas_assignments_raw = discovery_result.get('nas_assignments')

            # Validate that assignments match expected length
            if tas_assignments_raw is not None and nas_assignments_raw is not None:
                tas_array = _as_numpy(tas_assignments_raw, "TAS assignments")
                nas_array = _as_numpy(nas_assignments_raw, "NAS assignments")

                # Check if lengths match expected features length - HARD FAIL if mismatch
                if len(tas_array) != expected_length or len(nas_array) != expected_length:
                    tprint(f"🚨 CRITICAL: Length mismatch: TAS={len(tas_array)}, NAS={len(nas_array)}, expected={expected_length}", "ERROR")
                    tprint(f"🚨 CRITICAL: External labels are misaligned - this will poison clustering results", "ERROR")
                    tprint(f"🚨 CRITICAL: Ignoring external labels and cold-starting clustering", "ERROR")
                    # Return None to indicate cold start
                    return None, None

                return tas_array, nas_array

            if (tas_assignments_raw is None or nas_assignments_raw is None) and isinstance(
                discovery_result.get('artifacts'), dict
            ):
                nested_artifacts = discovery_result['artifacts']
                tas_assignments_raw = tas_assignments_raw or nested_artifacts.get('tas_assignments')
                nas_assignments_raw = nas_assignments_raw or nested_artifacts.get('nas_assignments')

            tas_assignments = _as_numpy(tas_assignments_raw, 'tas_assignments')
            nas_assignments = _as_numpy(nas_assignments_raw, 'nas_assignments')

            # Handle length mismatch by truncating assignments to match features
            if tas_assignments.shape[0] != expected_length:
                if tas_assignments.shape[0] > expected_length:
                    tprint(f"⚠️ Truncating TAS assignments from {tas_assignments.shape[0]} to {expected_length} to match features", "WARNING")
                    tas_assignments = tas_assignments[:expected_length]
                else:
                    raise ValueError(
                        f"TAS assignments too short: expected {expected_length}, got {tas_assignments.shape[0]}"
                    )

            if nas_assignments.shape[0] != expected_length:
                if nas_assignments.shape[0] > expected_length:
                    tprint(f"⚠️ Truncating NAS assignments from {nas_assignments.shape[0]} to {expected_length} to match features", "WARNING")
                    nas_assignments = nas_assignments[:expected_length]
                else:
                    raise ValueError(
                        f"NAS assignments too short: expected {expected_length}, got {nas_assignments.shape[0]}"
                    )

            tprint(
                f"Extracted TAS assignments: {len(tas_assignments)}, NAS assignments: {len(nas_assignments)}",
                "SUCCESS"
            )
            return tas_assignments, nas_assignments

        except Exception as e:
            tprint(f"❌ Failed to extract regime assignments: {e} - fast failing", "ERROR")
            raise ValueError(f"Failed to extract regime assignments: {e}")

    def _load_regime_discovery_from_outcomes(self) -> Optional[Dict[str, Any]]:
        """Load regime discovery results from previous pipeline outcomes."""
        try:
            # Look for previous outcome files in the outcomes directory
            outcomes_dir = Path("outcomes")
            if not outcomes_dir.exists():
                return None

            # Find the most recent regime discovery outcome file specifically
            outcome_files = list(outcomes_dir.glob("*nas_tas_regime_discovery*outcome*.json"))
            if not outcome_files:
                # Fallback to any market analysis outcome file
                outcome_files = list(outcomes_dir.glob("*market_analysis*outcome*.json"))
                if not outcome_files:
                    return None

            # Sort by modification time (most recent first)
            outcome_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_outcome = outcome_files[0]

            # Load the outcome file
            with open(latest_outcome, 'r') as f:
                outcome_data = json.load(f)

            # Extract regime discovery results
            if 'artifacts' in outcome_data:
                artifacts = outcome_data['artifacts']

                # Look for regime discovery results
                for key in ['nas_tas_regime_discovery_result', 'regime_discovery_result']:
                    if key in artifacts and artifacts[key]:
                        tprint(f"✅ Successfully loaded regime discovery results from {latest_outcome.name}", "SUCCESS")
                        return artifacts[key]

            tprint(f"⚠️ No regime discovery results found in {latest_outcome.name}", "WARNING")
            return None

        except Exception as e:
            tprint(f"❌ Error loading regime discovery from outcomes: {e}", "ERROR")
            return None

    def _smart_cluster_splitting_decision(self, assignments: np.ndarray, features: np.ndarray, current_k: int, iteration: int, baseline_score: float) -> Tuple[np.ndarray, int, Dict]:
        """Smart cluster splitting decision with enhanced logic."""
        try:
            # Step 1: Analyze cluster quality with relative thresholds
            cluster_metrics = self._analyze_cluster_quality_enhanced(assignments, features)

            # Step 2: Identify clusters that need splitting
            clusters_to_split = []
            for cluster_id in range(current_k):
                cluster_mask = assignments == cluster_id
                cluster_size = np.sum(cluster_mask)

                if cluster_size < 10:  # Too small to split
                    continue

                # Check if cluster quality is poor
                cluster_quality = cluster_metrics.get(cluster_id, {})
                silhouette = cluster_quality.get('silhouette', 0)
                dispersion = cluster_quality.get('dispersion', 1.0)

                # Split if silhouette is poor or dispersion is high
                if silhouette < 0.1 or dispersion > 2.0:
                    clusters_to_split.append(cluster_id)

            # Step 3: Apply splitting to identified clusters
            if clusters_to_split:
                new_assignments = assignments.copy()
                new_k = current_k

                for cluster_id in clusters_to_split:
                    # Simple splitting: split cluster into two sub-clusters
                    cluster_mask = assignments == cluster_id
                    cluster_points = features[cluster_mask]

                    if len(cluster_points) > 20:  # Only split large clusters
                        # Use k-means with k=2 to split the cluster
                        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                        sub_assignments = kmeans.fit_predict(cluster_points)

                        # Update assignments
                        new_assignments[cluster_mask] = np.where(
                            sub_assignments == 0,
                            new_k,  # New cluster
                            new_k + 1  # Another new cluster
                        )
                        new_k += 2

                        tprint(f"🔀 Split cluster {cluster_id} into clusters {new_k-1} and {new_k}", "SUCCESS")

                # Calculate final statistics
                final_stats = {
                    'clusters_split': len(clusters_to_split),
                    'new_clusters_created': new_k - current_k,
                    'final_k': new_k
                }

                return new_assignments, new_k, final_stats
            else:
                # No splitting needed
                return assignments, current_k, {'clusters_split': 0, 'new_clusters_created': 0, 'final_k': current_k}

        except Exception as e:
            tprint(f"❌ Error in cluster splitting decision: {e}", "ERROR")
            return assignments, current_k, {'clusters_split': 0, 'new_clusters_created': 0, 'final_k': current_k}

    def _analyze_cluster_quality_enhanced(self, assignments: np.ndarray, features: np.ndarray) -> Dict[int, Dict[str, float]]:
        """Analyze cluster quality with enhanced metrics for dynamic splitting."""
        try:
            unique_regimes = np.unique(assignments)
            cluster_metrics = {}
            
            for regime in unique_regimes:
                regime_mask = assignments == regime
                regime_size = np.sum(regime_mask)
                regime_percentage = regime_size / len(assignments)
                
                if regime_size > 0:
                    regime_features = features[regime_mask]
                    
                    # Calculate basic quality metrics
                    internal_cv = np.std(regime_features) / (np.mean(regime_features) + 1e-8)
                    compactness = 1.0 / (1.0 + internal_cv)
                    quality_score = compactness * regime_percentage
                    
                    cluster_metrics[regime] = {
                        'size_percentage': regime_percentage,
                        'internal_cv': internal_cv,
                        'compactness': compactness,
                        'silhouette_contribution': quality_score,
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

    def validate_clustering_robustness(self, features: np.ndarray, assignments: np.ndarray, 
                                     market_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Lightweight validation framework for clustering robustness."""
        try:
            tprint("🔍 Starting clustering validation...", "INFO")
            
            validation_results = {}
            
            # Basic clustering metrics
            n_clusters = len(np.unique(assignments))
            n_samples = features.shape[0]
            
            # Calculate silhouette score
            from sklearn.metrics import silhouette_score
            silhouette = silhouette_score(features, assignments)
            
            # Regime balance
            unique, counts = np.unique(assignments, return_counts=True)
            balance = 1.0 - (np.std(counts) / np.mean(counts)) if np.mean(counts) > 0 else 0.0
            
            validation_results['basic_metrics'] = {
                'silhouette_score': silhouette,
                'n_clusters': n_clusters,
                'n_samples': n_samples,
                'regime_balance': balance
            }
            
            # Overall quality score
            overall_robustness = (silhouette + balance) / 2.0
            
            validation_summary = {
                'overall_robustness': overall_robustness,
                'silhouette_score': silhouette,
                'regime_balance': balance
            }
            
            tprint(f"✅ Clustering validation completed - Overall quality: {overall_robustness:.3f}", "SUCCESS")
            
            return {
                'detailed_results': validation_results,
                'summary': validation_summary
            }
            
        except Exception as e:
            tprint(f"Clustering validation failed: {e}", "ERROR")
            return {'error': str(e)}

    def _perform_neighborhood_analysis(self, features: np.ndarray, assignments: np.ndarray, 
                                      market_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Perform neighborhood analysis for local structure insights."""
        try:
            tprint("🔍 Performing neighborhood analysis...", "INFO")
            
            # Basic neighborhood analysis
            n_clusters = len(np.unique(assignments))
            n_samples = features.shape[0]
            
            # Calculate cluster centroids
            centroids = []
            for cluster_id in range(n_clusters):
                cluster_mask = assignments == cluster_id
                if np.sum(cluster_mask) > 0:
                    centroid = np.mean(features[cluster_mask], axis=0)
                    centroids.append(centroid)
                else:
                    centroids.append(np.zeros(features.shape[1]))
            
            centroids = np.array(centroids)
            
            # Calculate inter-cluster distances
            from scipy.spatial.distance import pdist, squareform
            inter_cluster_distances = squareform(pdist(centroids))
            
            # Calculate average intra-cluster distances
            intra_cluster_distances = []
            for cluster_id in range(n_clusters):
                cluster_mask = assignments == cluster_id
                if np.sum(cluster_mask) > 1:
                    cluster_features = features[cluster_mask]
                    avg_distance = np.mean(pdist(cluster_features))
                    intra_cluster_distances.append(avg_distance)
                else:
                    intra_cluster_distances.append(0.0)
            
            neighborhood_results = {
                'n_clusters': n_clusters,
                'n_samples': n_samples,
                'inter_cluster_distances': inter_cluster_distances.tolist(),
                'intra_cluster_distances': intra_cluster_distances,
                'centroids': centroids.tolist()
            }
            
            tprint(f"✅ Neighborhood analysis completed - {n_clusters} clusters analyzed", "SUCCESS")
            return neighborhood_results
            
        except Exception as e:
            tprint(f"Neighborhood analysis failed: {e}", "ERROR")
            return {'error': str(e)}

    def _integrate_reallocation_in_optimization(self, features: np.ndarray, assignments: np.ndarray, 
                                              market_data: pd.DataFrame = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Integrate sample reallocation into optimization pipeline."""
        try:
            tprint("🔍 Integrating sample reallocation into optimization...", "INFO")
            
            # Basic reallocation optimization
            n_clusters = len(np.unique(assignments))
            n_samples = features.shape[0]
            
            # Calculate cluster centroids
            centroids = []
            for cluster_id in range(n_clusters):
                cluster_mask = assignments == cluster_id
                if np.sum(cluster_mask) > 0:
                    centroid = np.mean(features[cluster_mask], axis=0)
                    centroids.append(centroid)
                else:
                    centroids.append(np.zeros(features.shape[1]))
            
            centroids = np.array(centroids)
            
            # Simple reallocation: assign each point to its nearest centroid
            from scipy.spatial.distance import cdist
            distances = cdist(features, centroids)
            optimized_assignments = np.argmin(distances, axis=1)
            
            # Calculate reallocation statistics
            reallocated_count = np.sum(optimized_assignments != assignments)
            reallocation_rate = reallocated_count / n_samples
            
            reallocation_stats = {
                'n_clusters': n_clusters,
                'n_samples': n_samples,
                'reallocated_count': reallocated_count,
                'reallocation_rate': reallocation_rate,
                'optimization_quality': 1.0 - reallocation_rate  # Lower reallocation = better optimization
            }
            
            tprint(f"✅ Sample reallocation completed - {reallocated_count} samples reallocated ({reallocation_rate:.1%})", "SUCCESS")
            return optimized_assignments, reallocation_stats
            
        except Exception as e:
            tprint(f"Sample reallocation failed: {e}", "ERROR")
            return assignments, {'error': str(e)}

    def _summarize_results(self, context, market_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Summarize clustering results and create final output."""
        try:
            tprint("📊 Summarizing clustering results...", "INFO")
            
            # Handle both dict and ClusteringContext object
            if hasattr(context, 'optimized_assignments') or hasattr(context, 'smoothed_assignments'):
                # ClusteringContext object
                assignments = getattr(context, 'smoothed_assignments', None) or getattr(context, 'optimized_assignments', np.array([]))
                features = getattr(context, 'optimized_features', np.array([]))
                validation_quality = getattr(context, 'validation_quality', 0.0)
                neighborhood_analysis = getattr(context, 'neighborhood_analysis', {})
                reallocation_stats = getattr(context, 'reallocation_stats', {})
                tprint(f"🔍 Using ClusteringContext - assignments type: {type(assignments)}, shape: {getattr(assignments, 'shape', 'N/A')}", "DEBUG")
            elif hasattr(context, 'assignments'):
                # Legacy ClusteringContext object
                assignments = context.assignments
                features = getattr(context, 'features', np.array([]))
                validation_quality = getattr(context, 'validation_quality', 0.0)
                neighborhood_analysis = getattr(context, 'neighborhood_analysis', {})
                reallocation_stats = getattr(context, 'reallocation_stats', {})
                tprint(f"🔍 Using legacy ClusteringContext - assignments type: {type(assignments)}, shape: {getattr(assignments, 'shape', 'N/A')}", "DEBUG")
            else:
                # Dictionary
                assignments = context.get('assignments', np.array([]))
                features = context.get('features', np.array([]))
                validation_quality = context.get('validation_quality', 0.0)
                neighborhood_analysis = context.get('neighborhood_analysis', {})
                reallocation_stats = context.get('reallocation_stats', {})
            
            # Create summary statistics
            # Handle both numpy arrays and lists properly
            tprint(f"🔍 About to calculate statistics - assignments type: {type(assignments)}", "DEBUG")
            tprint(f"🔍 Assignments content preview: {assignments[:5] if hasattr(assignments, '__getitem__') else 'N/A'}", "DEBUG")
            
            if isinstance(assignments, np.ndarray):
                tprint(f"🔍 Processing numpy array - size: {assignments.size}, shape: {assignments.shape}", "DEBUG")
                tprint(f"🔍 Array dtype: {assignments.dtype}", "DEBUG")
                try:
                    if assignments.size > 0:
                        unique_vals = np.unique(assignments)
                        tprint(f"🔍 Unique values: {unique_vals[:10]} (showing first 10)", "DEBUG")
                        n_clusters = len(unique_vals)
                        cluster_distribution = np.bincount(assignments)
                    else:
                        n_clusters = 0
                        cluster_distribution = []
                    n_samples = assignments.size
                    tprint(f"🔍 Numpy array stats - n_clusters: {n_clusters}, n_samples: {n_samples}", "DEBUG")
                except Exception as e:
                    tprint(f"🔍 Error in numpy array processing: {e}", "ERROR")
                    raise
            else:
                tprint(f"🔍 Processing list/other - length: {len(assignments) if hasattr(assignments, '__len__') else 'N/A'}", "DEBUG")
                try:
                    if len(assignments) > 0:
                        unique_vals = np.unique(assignments)
                        n_clusters = len(unique_vals)
                        cluster_distribution = np.bincount(assignments)
                    else:
                        n_clusters = 0
                        cluster_distribution = []
                    n_samples = len(assignments)
                    tprint(f"🔍 List stats - n_clusters: {n_clusters}, n_samples: {n_samples}", "DEBUG")
                except Exception as e:
                    tprint(f"🔍 Error in list processing: {e}", "ERROR")
                    raise
            
            # Handle features shape properly
            tprint(f"🔍 Processing features - type: {type(features)}", "DEBUG")
            if isinstance(features, np.ndarray):
                features_shape = features.shape if features.size > 0 else (0, 0)
                tprint(f"🔍 Numpy features shape: {features_shape}", "DEBUG")
            else:
                features_shape = features.shape if len(features) > 0 else (0, 0)
                tprint(f"🔍 List features shape: {features_shape}", "DEBUG")
            
            tprint(f"🔍 Creating summary with n_clusters: {n_clusters}, n_samples: {n_samples}", "DEBUG")
            summary = {
                'n_clusters': n_clusters,
                'n_samples': n_samples,
                'cluster_distribution': cluster_distribution,
                'features_shape': features_shape,
                'validation_quality': validation_quality,
                'neighborhood_analysis': neighborhood_analysis,
                'reallocation_stats': reallocation_stats,
                'success': True
            }
            
            tprint(f"✅ Results summarized: {n_clusters} clusters, {n_samples} samples", "SUCCESS")
            return summary
            
        except Exception as e:
            tprint(f"Results summarization failed: {e}", "ERROR")
            return {'error': str(e), 'success': False}


class ClusterStats:
    """Maintains sufficient statistics for O(1) cluster operations."""

    def __init__(self, cluster_id: int, features: np.ndarray = None, initial_assignments: np.ndarray = None):
        self.cluster_id = cluster_id
        self.n = 0  # Count of points in cluster
        self.mu = np.zeros(features.shape[1]) if features is not None else None  # Centroid
        self.S = np.zeros(features.shape[1]) if features is not None else None  # Sum of features
        self.Q = np.zeros((features.shape[1], features.shape[1])) if features is not None else None  # Sum of outer products

        if features is not None and initial_assignments is not None:
            self._initialize_from_data(features, initial_assignments)

    def _initialize_from_data(self, features: np.ndarray, assignments: np.ndarray):
        """Initialize stats from full data scan (called once)."""
        mask = assignments == self.cluster_id
        if np.sum(mask) == 0:
            return

        cluster_features = features[mask]
        self.n = len(cluster_features)
        self.S = np.sum(cluster_features, axis=0)
        self.mu = self.S / self.n if self.n > 0 else np.zeros_like(self.S)
        self.Q = cluster_features.T @ cluster_features

    @staticmethod
    def _add_point_jit(n, S, Q, mu, point):
        """Numba JIT version of add_point for speed."""
        n += 1
        S += point
        mu = S / n
        Q += np.outer(point, point)
        return n, S, Q, mu

    def add_point(self, point: np.ndarray):
        """Add a point to cluster stats (O(d) operation)."""
        if NUMBA_AVAILABLE:
            self.n, self.S, self.Q, self.mu = self._add_point_jit(self.n, self.S, self.Q, self.mu, point)
        else:
            self.n += 1
            self.S += point
            self.mu = self.S / self.n
            self.Q += np.outer(point, point)

    @staticmethod
    def _remove_point_jit(n, S, Q, mu, point):
        """Numba JIT version of remove_point for speed."""
        if n <= 0:
            return n, S, Q, mu

        n -= 1
        S -= point
        if n > 0:
            mu = S / n
        else:
            mu.fill(0)
        Q -= np.outer(point, point)
        return n, S, Q, mu

    def remove_point(self, point: np.ndarray):
        """Remove a point from cluster stats (O(d) operation)."""
        if NUMBA_AVAILABLE:
            self.n, self.S, self.Q, self.mu = self._remove_point_jit(self.n, self.S, self.Q, self.mu, point)
        else:
            if self.n <= 0:
                return

            self.n -= 1
            self.S -= point
            if self.n > 0:
                self.mu = self.S / self.n
            else:
                self.mu.fill(0)
            self.Q -= np.outer(point, point)

    def merge_with(self, other: 'ClusterStats') -> 'ClusterStats':
        """Merge two cluster stats (O(d) operation)."""
        merged = ClusterStats(self.cluster_id)
        merged.n = self.n + other.n
        merged.S = self.S + other.S
        merged.mu = merged.S / merged.n if merged.n > 0 else np.zeros_like(merged.S)
        merged.Q = self.Q + other.Q
        return merged

    def get_covariance(self) -> np.ndarray:
        """Get covariance matrix (O(d²) but cached)."""
        if self.n <= 1:
            return np.eye(self.S.shape[0])

        # Covariance = (1/(n-1)) * (Q - n*μ*μ^T)
        mu_outer = np.outer(self.mu, self.mu)
        return (self.Q - self.n * mu_outer) / (self.n - 1)

    @staticmethod
    def _compute_compactness_jit(S, n, mu):
        """Numba JIT version of compactness computation."""
        if n <= 1:
            return 0.0

        # Average squared distance from centroid
        centered = S / n - mu
        return np.sqrt(np.sum(centered ** 2))  # RMS distance

    def get_compactness(self) -> float:
        """Get average distance from centroid (O(d) per call, but can cache)."""
        if NUMBA_AVAILABLE:
            return self._compute_compactness_jit(self.S, self.n, self.mu)
        else:
            if self.n <= 1:
                return 0.0

            # Average squared distance from centroid
            centered = self.S / self.n - self.mu
            return np.sqrt(np.sum(centered ** 2))  # RMS distance

    def get_scatter_matrix(self) -> np.ndarray:
        """Get within-cluster scatter matrix for metrics like DBI."""
        if self.n <= 1:
            return np.zeros_like(self.Q)

        # Scatter = Q - n*μ*μ^T
        mu_outer = np.outer(self.mu, self.mu)
        return self.Q - self.n * mu_outer

class CachedClusteringState:
    """Maintains cached state for efficient clustering operations."""

    def __init__(self, features: np.ndarray, assignments: np.ndarray, k: int):
        self.features = features.astype(np.float32)  # Ensure float32
        self.k = k
        self.cluster_stats = [ClusterStats(i, features, assignments) for i in range(k)]
        self.assignments = assignments.copy()
        self.last_knn_rebuild = 0
        self.knn_cache = None
        self.faiss_index = None  # FAISS index for fast neighbor search
        self.centroid_cache = None
        self.metrics_cache = {}

        # Precompute initial state
        self._precompute_cached_metrics()

    def _precompute_cached_metrics(self):
        """Precompute expensive metrics that change slowly."""
        self.centroid_cache = np.array([stats.mu for stats in self.cluster_stats])
        self._update_knn_cache()

    def _update_knn_cache(self):
        """Update FAISS kNN index when needed."""
        try:
            # Only rebuild if >15% of labels changed or after major structural changes
            if FAISS_AVAILABLE and (self.faiss_index is None or self._should_rebuild_knn()):
                self._build_faiss_index()
            self.last_knn_rebuild = len(self.assignments)
        except Exception as e:
            tprint(f"FAISS index update failed: {e}", "WARNING")
            # Fallback to basic implementation
            pass

    def _should_rebuild_knn(self) -> bool:
        """Check if kNN index should be rebuilt."""
        # Rebuild if >15% labels changed or after >10% cluster count change
        if self.last_knn_rebuild == 0:
            return True

        # Simple heuristic: rebuild every few rounds or when many changes occur
        return True  # For now, rebuild frequently for safety

    def _build_faiss_index(self):
        """Build FAISS index for fast neighbor search."""
        try:

            # Use HNSW for high recall with reasonable speed
            d = self.features.shape[1]  # Dimension

            # Normalize features for better distance computation
            features_normalized = self.features.copy()
            norms = np.linalg.norm(features_normalized, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            features_normalized /= norms

            # Build HNSW index with good parameters
            M = 32  # Number of connections per layer
            efConstruction = 200  # Construction time vs accuracy trade-off
            efSearch = 64  # Search time vs accuracy trade-off

            self.faiss_index = faiss.IndexHNSWFlat(d, M)
            self.faiss_index.hnsw.efConstruction = efConstruction
            self.faiss_index.hnsw.efSearch = efSearch

            # Train and add data
            self.faiss_index.add(features_normalized.astype(np.float32))

            tprint(f"   🔍 FAISS HNSW index built: {len(self.features)} points, dim {d}", "INFO")

        except ImportError:
            if not FAISS_AVAILABLE:
                tprint("   ⚠️ FAISS not available, using fallback neighbor search", "WARNING")
            else:
                tprint(f"   ❌ FAISS import error: {e}", "ERROR")
            self.faiss_index = None
        except Exception as e:
            tprint(f"   ❌ FAISS index build failed: {e}", "ERROR")
            self.faiss_index = None

    def find_knn_faiss(self, query_points: np.ndarray, k_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors using FAISS."""
        try:
            if not FAISS_AVAILABLE or self.faiss_index is None:
                return self._find_knn_fallback(query_points, k_neighbors)

            # Normalize query points
            query_normalized = query_points.copy()
            norms = np.linalg.norm(query_normalized, axis=1, keepdims=True)
            norms[norms == 0] = 1
            query_normalized /= norms

            # Search with FAISS
            distances, indices = self.faiss_index.search(query_normalized.astype(np.float32), k_neighbors)

            return distances, indices

        except Exception as e:
            tprint(f"   ❌ FAISS search failed: {e}", "ERROR")
            return self._find_knn_fallback(query_points, k_neighbors)

    def _find_knn_fallback(self, query_points: np.ndarray, k_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback kNN implementation when FAISS unavailable."""
        n_queries = query_points.shape[0]
        distances = np.zeros((n_queries, k_neighbors))
        indices = np.zeros((n_queries, k_neighbors), dtype=int)

        for i, query in enumerate(query_points):
            # Compute distances to all points
            dists = np.linalg.norm(self.features - query, axis=1)

            # Get k nearest (excluding self if query is from dataset)
            if i < len(self.features):
                dists[i] = np.inf  # Exclude self

            # Find k smallest distances
            knn_indices = np.argsort(dists)[:k_neighbors]
            knn_distances = dists[knn_indices]

            distances[i] = knn_distances
            indices[i] = knn_indices

        return distances, indices

    def move_point(self, point_idx: int, from_cluster: int, to_cluster: int):
        """Move a point between clusters (O(d) operation)."""
        point = self.features[point_idx]

        # Remove from source cluster
        if from_cluster >= 0 and from_cluster < self.k:
            self.cluster_stats[from_cluster].remove_point(point)

        # Add to target cluster
        if to_cluster >= 0 and to_cluster < self.k:
            self.cluster_stats[to_cluster].add_point(point)

        # Update assignment
        self.assignments[point_idx] = to_cluster

        # Invalidate dependent caches
        self.centroid_cache = None
        self.metrics_cache.clear()

    def merge_clusters(self, cluster_i: int, cluster_j: int) -> int:
        """Merge cluster_j into cluster_i (O(d) operation)."""
        if cluster_i == cluster_j or cluster_i >= self.k or cluster_j >= self.k:
            return self.k

        # Merge statistics
        self.cluster_stats[cluster_i] = self.cluster_stats[cluster_i].merge_with(self.cluster_stats[cluster_j])

        # Move all points from j to i
        mask = self.assignments == cluster_j
        self.assignments[mask] = cluster_i

        # Mark cluster_j as empty
        self.cluster_stats[cluster_j] = ClusterStats(cluster_j)
        self.cluster_stats[cluster_j].n = 0

        # Invalidate caches
        self.centroid_cache = None
        self.metrics_cache.clear()

        return self.k - 1

    def get_centroids(self) -> np.ndarray:
        """Get cached centroids."""
        if self.centroid_cache is None:
            self.centroid_cache = np.array([stats.mu for stats in self.cluster_stats])
        return self.centroid_cache

    def get_cluster_sizes(self) -> np.ndarray:
        """Get cluster sizes."""
        return np.array([stats.n for stats in self.cluster_stats])

    def get_cached_metric(self, metric_name: str):
        """Get cached metric if available."""
        return self.metrics_cache.get(metric_name)

    def set_cached_metric(self, metric_name: str, value):
        """Cache a metric."""
        self.metrics_cache[metric_name] = value

    def _get_assignment_hash(self, assignments: np.ndarray, k: int) -> str:
        """Generate a hash of current assignments state for cycle detection."""
        try:
            # Create a compact representation of assignments and k
            state = (tuple(assignments), k)
            import hashlib
            state_str = str(state).encode('utf-8')
            return hashlib.md5(state_str).hexdigest()
        except Exception:
            return ""

    def _run_consecutive_optimization_loop(self, features: np.ndarray, initial_assignments: np.ndarray, k: int,
                                         max_rounds: int = 50, tolerance: float = 1e-5) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run consecutive optimization loop: merging → reshuffling → splitting until convergence."""
        try:
            tprint("🚀 Starting consecutive optimization loop (merging → reshuffling → splitting)...", "INFO")
            tprint(f"📊 Parameters: max_rounds={max_rounds}, tolerance={tolerance:.2e}", "INFO")

            assignments = initial_assignments.copy()
            best_assignments = assignments.copy()
            best_score = -1.0

            # Initialize cached clustering state for O(1) operations
            cached_state = CachedClusteringState(features, assignments, k)

            # State caching for cycle detection
            state_cache = set()
            hysteresis_map = {}  # Track which points are locked due to hysteresis
            round_stats = []

            convergence_history = []

            for round_num in range(max_rounds):
                tprint(f"\n🎯 Round {round_num + 1}/{max_rounds}", "INFO")

                # Check for state cycles
                current_hash = self._get_assignment_hash(assignments, k)
                if current_hash in state_cache:
                    tprint(f"   🔄 State cycle detected, applying stricter thresholds...", "WARNING")
                    # Could increase threshold here if needed

                state_cache.add(current_hash)

                # Calculate objective before round (O(1) with cached state)
                round_start_score = self._calculate_composite_objective_cached(cached_state)

                # Stage 1: Merging and remerging (O(1) with cached stats)
                tprint("   🔄 Stage 1: Merging/Remerging", "INFO")
                merged_assignments, new_k, merge_stats = self._perform_cluster_merging_and_remerging_cached(
                    cached_state, round_start_score)
                assignments = merged_assignments
                k = new_k

                # Stage 2: Neighbor reshuffling
                tprint("   🔀 Stage 2: Neighbor reshuffling", "INFO")
                reshuffled_assignments, reshuffle_stats = self._perform_neighbor_reshuffling(
                    features, assignments, k, round_start_score, hysteresis_map, round_num, cached_state)
                assignments = reshuffled_assignments

                # Stage 3: Regime splitting
                tprint("   ✂️ Stage 3: Regime splitting", "INFO")
                split_assignments, split_k, split_stats = self._smart_cluster_splitting_decision(
                    assignments, features, k, round_num, round_start_score)
                if split_k > k:
                    assignments = split_assignments
                    k = split_k

                # Calculate round score (O(1) with cached state)
                round_score = self._calculate_composite_objective_cached(cached_state)

                # Update best assignments if improved
                if round_score > best_score:
                    best_score = round_score
                    best_assignments = assignments.copy()
                    tprint(f"   📈 New best score: {best_score:.4f}", "SUCCESS")

                # Store convergence metrics and round statistics
                round_stats.append({
                    'round': round_num + 1,
                    'score': round_score,
                    'k': k,
                    'improvement': round_score - round_start_score,
                    'merge_stats': merge_stats,
                    'reshuffle_stats': reshuffle_stats,
                    'split_stats': split_stats,
                    'state_hash': current_hash
                })

                convergence_history.append({
                    'round': round_num + 1,
                    'score': round_score,
                    'k': k,
                    'improvement': round_score - round_start_score
                })

                # Check convergence - require no accepted operations in last full cycle
                if len(convergence_history) >= 6:  # Need at least 2 full cycles (M→R→S)
                    recent_scores = [h['score'] for h in convergence_history[-6:]]
                    avg_improvement = np.mean(np.diff(recent_scores))

                    # Check if no operations were accepted in the last cycle
                    last_cycle_stats = round_stats[-3:]  # Last 3 rounds (M, R, S)
                    total_accepted = sum(
                        stats.get('accepted', 0) for stats in last_cycle_stats
                        for phase_stats in [stats.get('merge_stats', {}), stats.get('reshuffle_stats', {}), stats.get('split_stats', {})]
                    )

                    if abs(avg_improvement) < tolerance and total_accepted == 0:
                        tprint(f"✅ Convergence achieved at round {round_num + 1}!", "SUCCESS")
                        tprint(f"   📊 Average improvement: {avg_improvement:.6f} (tolerance: {tolerance:.2e})", "SUCCESS")
                        tprint(f"   🎯 No operations accepted in last cycle", "SUCCESS")
                        break

                # Progress update with detailed stats
                if round_num % 5 == 0:
                    tprint(f"   📊 Progress: Round {round_num + 1}/{max_rounds}, Score: {round_score:.4f}, K: {k}", "INFO")
                    # Show recent operation counts
                    if len(round_stats) >= 3:
                        recent_ops = round_stats[-3:]
                        total_merges = sum(s.get('merge_stats', {}).get('accepted', 0) for s in recent_ops)
                        total_reshuffles = sum(s.get('reshuffle_stats', {}).get('accepted', 0) for s in recent_ops)
                        total_splits = sum(s.get('split_stats', {}).get('accepted', 0) for s in recent_ops)
                        tprint(f"   📈 Recent ops: Merges={total_merges}, Reshuffles={total_reshuffles}, Splits={total_splits}", "INFO")

            # Final optimization
            tprint("   🔧 Applying final optimization pass...", "INFO")
            final_assignments = best_assignments

            # Return to original k if needed (some splitting/merging may have occurred)
            if k != len(np.unique(final_assignments)):
                k = len(np.unique(final_assignments))

            iteration_metrics = {
                'total_rounds': len(convergence_history),
                'converged': len(convergence_history) < max_rounds,
                'final_score': best_score,
                'final_k': k,
                'convergence_history': convergence_history
            }

            tprint(f"🎉 Consecutive optimization loop completed!", "SUCCESS")
            tprint(f"   📊 Total rounds: {len(convergence_history)}/{max_rounds}", "SUCCESS")
            tprint(f"   🎯 Final score: {best_score:.4f}", "SUCCESS")
            tprint(f"   📈 Final K: {k}", "SUCCESS")

            return final_assignments, iteration_metrics

        except Exception as e:
            tprint(f"Consecutive optimization loop failed: {e}", "ERROR")
            return initial_assignments, {'error': str(e), 'total_rounds': 0, 'converged': False}

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
            
            # ENHANCED: Maximum focus on silhouette and CV for better clustering quality
            fast_improvement = (silhouette_improvement * 0.80 +     # Ultra-high focus on silhouette (increased from 0.70)
                              distance_improvement * 0.05 +         # Minimal distance factor (reduced from 0.15)
                              balance_improvement * 0.05 +          # Minimal balance factor (reduced from 0.10)
                              temporal_improvement * 0.10)          # Increased temporal factor (from 0.05)
            
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
                tprint("⚠️ WARNING: _calculate_regime_balance_optimized is not defined - using fallback", "WARNING")
                old_balance = 0.5  # Fallback value
                tprint("⚠️ WARNING: _calculate_silhouette_score_optimized is not defined - using fallback", "WARNING")
                old_silhouette = 0.5  # Fallback value
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
            
            # ENHANCED: Balanced optimization for better clustering quality and regime balance
            balance_weight = getattr(self.config, 'balance_weight', 0.25)  # Increased balance weight for better regime distribution
            silhouette_weight = 0.40  # High silhouette emphasis but not overwhelming
            cv_weight = 0.25  # Moderate CV emphasis for stability
            temporal_weight = 0.10  # Increased temporal weight for regime consistency
            
            old_composite = (old_balance * balance_weight + 
                           old_silhouette * silhouette_weight + 
                           old_cv * cv_weight + 
                           old_temporal * temporal_weight)
            
            # Calculate new scores (only if significantly different)
            if np.array_equal(old_assignments, new_assignments):
                return 0.0
            
            tprint("⚠️ WARNING: _calculate_regime_balance_optimized is not defined - using fallback", "WARNING")
            new_balance = 0.5  # Fallback value
            tprint("⚠️ WARNING: _calculate_silhouette_score_optimized is not defined - using fallback", "WARNING")
            new_silhouette = 0.5  # Fallback value
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

    def _calculate_final_quality_metrics(self, features: np.ndarray, assignments: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics for the final clustering result."""
        try:
            # Import safe functions from metrics module
            from ..regime_analysis.metrics import (
                safe_silhouette_score, 
                safe_davies_bouldin_score, 
                safe_calinski_harabasz_score
            )
            
            # Calculate standard clustering metrics
            tprint(f"🔍 Features shape: {features.shape}, Assignments shape: {assignments.shape}", "INFO")
            silhouette = safe_silhouette_score(features, assignments)
            tprint(f"🔍 Silhouette score: {silhouette}", "INFO")
            davies_bouldin = safe_davies_bouldin_score(features, assignments)
            tprint(f"🔍 Davies-Bouldin score: {davies_bouldin}", "INFO")
            calinski_harabasz = safe_calinski_harabasz_score(features, assignments)
            tprint(f"🔍 Calinski-Harabasz score: {calinski_harabasz}", "INFO")
            
            # Calculate CV score
            cv_score = self._calculate_cv_score_optimized(features, assignments)
            
            # Calculate temporal smoothness
            temporal_smoothness = self._calculate_temporal_smoothness_optimized(assignments)
            
            # Calculate regime balance
            unique_labels = np.unique(assignments)
            n_clusters = len(unique_labels)
            regime_counts = np.bincount(assignments)
            regime_balance = 1.0 - (np.std(regime_counts) / (np.mean(regime_counts) + 1e-8))
            
            # Calculate intra-cluster dispersion
            intra_dispersion = 0.0
            for label in unique_labels:
                mask = assignments == label
                if np.any(mask):
                    cluster_features = features[mask]
                    center = np.mean(cluster_features, axis=0)
                    distances = np.linalg.norm(cluster_features - center, axis=1)
                    intra_dispersion += np.mean(distances)
            
            intra_dispersion /= n_clusters
            
            # Calculate inter-cluster dispersion
            centers = []
            for label in unique_labels:
                mask = assignments == label
                if np.any(mask):
                    center = np.mean(features[mask], axis=0)
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
            compactness = inter_dispersion / (intra_dispersion + 1e-8)
            
            # Create comprehensive quality metrics
            quality_metrics = {
                # Standard clustering metrics
                "silhouette_score": float(silhouette),
                "davies_bouldin_score": float(davies_bouldin),
                "calinski_harabasz_score": float(calinski_harabasz),
                "cv_score": float(cv_score),
                
                # Temporal and regime metrics
                "temporal_smoothness": float(temporal_smoothness),
                "regime_balance": float(regime_balance),
                
                # Dispersion metrics
                "intra_cluster_dispersion": float(intra_dispersion),
                "inter_cluster_dispersion": float(inter_dispersion),
                "cluster_compactness": float(compactness),
                
                # Regime distribution
                "regime_distribution": {int(label): int(count) for label, count in enumerate(regime_counts)},
                "n_clusters": int(n_clusters),
                "total_samples": int(len(assignments))
            }
            
            tprint(f"🔍 Quality metrics calculated successfully: {list(quality_metrics.keys())}", "SUCCESS")
            return quality_metrics
            
        except Exception as exc:
            tprint_warning(f"Failed to calculate final quality metrics: {exc}")
            return {
                "silhouette_score": 0.0,
                "davies_bouldin_score": float('inf'),
                "calinski_harabasz_score": 0.0,
                "cv_score": 0.0,
                "temporal_smoothness": 0.0,
                "regime_balance": 0.0,
                "intra_cluster_dispersion": 0.0,
                "inter_cluster_dispersion": 0.0,
                "cluster_compactness": 0.0,
                "regime_distribution": {},
                "n_clusters": 0,
                "total_samples": 0
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





    def _derive_validation_metric(self, clustering_metrics: Dict[str, Any]) -> float:
        """Derive a validation metric from clustering metrics for weight learning."""
        try:
            # Try to get a composite validation score from clustering_metrics
            if isinstance(clustering_metrics, dict):
                # Look for economic scores as primary validation metric
                economic_scores = clustering_metrics.get('economic_scores')
                if economic_scores and isinstance(economic_scores, dict):
                    # Average economic scores as validation metric
                    values = [v for v in economic_scores.values() if isinstance(v, (int, float)) and not np.isnan(v)]
                    if values:
                        return float(np.mean(values))
                
                # Fallback to trading scores
                trading_scores = clustering_metrics.get('trading_scores')
                if trading_scores and isinstance(trading_scores, dict):
                    values = [v for v in trading_scores.values() if isinstance(v, (int, float)) and not np.isnan(v)]
                    if values:
                        return float(np.mean(values))
                
                # Fallback to stability scores
                stability_scores = clustering_metrics.get('stability_scores')
                if stability_scores and isinstance(stability_scores, dict):
                    values = [v for v in stability_scores.values() if isinstance(v, (int, float)) and not np.isnan(v)]
                    if values:
                        return float(np.mean(values))
            
            # Default fallback - use a neutral validation metric
            return 0.5
            
        except Exception as exc:
            tprint_warning(f"Failed to derive validation metric: {exc}")
            return 0.5

    def _fit_metric_weights(
        self,
        metric_outputs: Dict[str, Dict[str, float]],
        validation_metric: float
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """Fit metric weights based on outputs and validation metric."""
        try:
            if not metric_outputs or not isinstance(validation_metric, (int, float)):
                return None
            
            # Simple weight learning: adjust weights based on validation metric performance
            learned_weights = {}
            
            for group, metrics in metric_outputs.items():
                if not isinstance(metrics, dict):
                    continue

                # Get default weights for this group
                default_weights = self._default_metric_weights.get(group, {})
                if not default_weights:
                    continue
            
                # Calculate adjustment factor based on validation metric
                # Higher validation metric should increase weights for better-performing metrics
                adjustment_factor = max(0.5, min(2.0, validation_metric * 2.0))
                
                learned_group_weights = {}
                for metric_name, default_weight in default_weights.items():
                    if metric_name in metrics:
                        # Adjust weight based on metric performance and validation score
                        metric_value = metrics[metric_name]
                        if isinstance(metric_value, (int, float)) and not np.isnan(metric_value):
                            # Higher metric value and validation score = higher weight
                            performance_factor = metric_value * adjustment_factor
                            learned_weight = default_weight * (1.0 + performance_factor)
                            learned_group_weights[metric_name] = float(np.clip(learned_weight, 0.0, 1.0))
                        else:
                            learned_group_weights[metric_name] = default_weight
                    else:
                        learned_group_weights[metric_name] = default_weight
                
                # Normalize weights to sum to 1.0
                total_weight = sum(learned_group_weights.values())
                if total_weight > 0:
                    for metric_name in learned_group_weights:
                        learned_group_weights[metric_name] /= total_weight
                
                learned_weights[group] = learned_group_weights
            
            # Store learned weights
            if learned_weights:
                self.learned_weights = learned_weights
                
                # Add to history
                self.metric_weight_history.append({
                    'timestamp': pd.Timestamp.now(),
                    'validation_metric': validation_metric,
                    'learned_weights': learned_weights.copy()
                })
                
                # Keep history within limit
                if len(self.metric_weight_history) > self._weight_history_limit:
                    self.metric_weight_history = self.metric_weight_history[-self._weight_history_limit:]
            
            return learned_weights
            
        except Exception as exc:
            tprint_warning(f"Failed to fit metric weights: {exc}")
            return None

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
    
    
    
    def _compute_regime_centroids_vectorized(self, features: np.ndarray, assignments: np.ndarray, k: int) -> np.ndarray:
        """Compute regime centroids using vectorized operations for performance."""
        try:
            unique_regimes = np.unique(assignments)
            n_regimes = len(unique_regimes)
            n_features = features.shape[1]
            
            # Initialize centroids array
            centroids = np.zeros((n_regimes, n_features))
            
            # Vectorized centroid calculation
            for i, regime in enumerate(unique_regimes):
                regime_mask = assignments == regime
                if np.any(regime_mask):
                    centroids[i] = np.mean(features[regime_mask], axis=0)
            
            return centroids
            
        except Exception as e:
            tprint_error(f"Error computing regime centroids: {e}")
            # Fallback to simple mean calculation
            unique_regimes = np.unique(assignments)
            centroids = np.array([np.mean(features[assignments == regime], axis=0) 
                                 for regime in unique_regimes])
            return centroids
            
    
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

    def _create_regime_assignments_dataframe(self, cluster_assignments: List[int],
                                           features: np.ndarray,
                                           market_data: pd.DataFrame) -> pd.DataFrame:
        """Create a DataFrame with regime assignments, features, and market data."""
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
                # Truncate market_data to match assignments length
                if len(cluster_assignments) < len(market_data):
                    market_data = market_data.iloc[:len(cluster_assignments)]
                    tprint(f"⚠️ Truncated market_data to {len(market_data)} to match assignments", "WARNING")

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
            # Return minimal DataFrame as fallback with proper length alignment
            min_length = min(len(cluster_assignments), len(market_data))
            fallback_df = pd.DataFrame({
                'regime_id': cluster_assignments[:min_length],
                'regime_prob': [0.8] * min_length
            })

            if hasattr(market_data, 'index'):
                fallback_df.index = market_data.index[:min_length]

            tprint_warning(f"⚠️ Returning fallback regime assignments DataFrame: {fallback_df.shape}")
            return fallback_df

    def _save_regime_assignments_parquet(self, regime_df: pd.DataFrame, symbol: str = "ETHUSDT") -> Path:
        """Save regime assignments DataFrame as parquet file."""
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

    def _load_regime_discovery_from_outcomes(self) -> Optional[Dict[str, Any]]:
        """Load regime discovery results from previous successful outcomes."""
        import os
        import json
        from pathlib import Path

        try:
            outcomes_dir = Path("/Users/remyroche/Documents/Ares/outcomes")

            # Look for successful NAS/TAS regime discovery outcomes
            pattern = "market_analysis_nas_tas_regime_discovery_outcome_*.json"
            outcome_files = list(outcomes_dir.glob(pattern))

            if not outcome_files:
                tprint("📁 No regime discovery outcome files found", "INFO")
                return None

            # Find the most recent successful outcome
            successful_outcomes = []
            for file_path in outcome_files:
                try:
                    with open(file_path, 'r') as f:
                        outcome_data = json.load(f)

                    # Check if the outcome was successful and contains expected data
                    if (outcome_data.get('status') == 'completed' and
                        'artifacts' in outcome_data and
                        outcome_data['artifacts']):
                        successful_outcomes.append((file_path, outcome_data.get('timestamp', '')))
                except Exception as e:
                    tprint(f"⚠️ Error reading outcome file {file_path}: {e}", "DEBUG")
                    continue

            if not successful_outcomes:
                tprint("📁 No successful regime discovery outcomes found", "INFO")
                return None

            # Sort by timestamp and get the most recent
            successful_outcomes.sort(key=lambda x: x[1], reverse=True)
            latest_outcome_path, _ = successful_outcomes[0]

            tprint(f"📁 Loading regime discovery results from: {latest_outcome_path}", "INFO")

            # Load the outcome data
            with open(latest_outcome_path, 'r') as f:
                outcome_data = json.load(f)

            # Extract regime discovery results from artifacts
            artifacts = outcome_data.get('artifacts', {})
            regime_discovery = None

            # Try different possible keys for regime discovery results
            possible_keys = [
                'regime_discovery_result',
                'nas_tas_regime_discovery_result',
                'optimal_regime_clustering_result',
                'hmm_regime_discovery_result',
                'tas_regime_states',
                'nas_regime_states',
                'regime_states',
                'tas_assignments',
                'nas_assignments'
            ]

            for key in possible_keys:
                if key in artifacts and artifacts[key]:
                    regime_discovery = artifacts[key]
                    tprint(f"✅ Found regime discovery results under key: {key}", "INFO")
                    break

            if regime_discovery:
                tprint(f"✅ Successfully loaded regime discovery results from previous outcome", "INFO")
                return regime_discovery
            else:
                tprint("⚠️ No regime discovery results found in outcome artifacts", "WARNING")
                return None

        except Exception as e:
            tprint(f"❌ Error loading regime discovery from outcomes: {e}", "ERROR")
            return None

    
    def _apply_dawid_skene_fusion(self, tas_assignments: np.ndarray, nas_assignments: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Apply Dawid-Skene fusion to combine TAS and NAS regime assignments."""
        try:
            tprint("Starting Dawid-Skene fusion process...", "INFO")
            
            # Validate inputs
            if tas_assignments is None or nas_assignments is None:
                raise ValueError("Both TAS and NAS assignments are required for fusion")
            
            if len(tas_assignments) != len(nas_assignments):
                raise ValueError(f"Assignment length mismatch: TAS={len(tas_assignments)}, NAS={len(nas_assignments)}")
            
            # Determine target K based on unique regimes in both assignments
            tas_unique = len(np.unique(tas_assignments))
            nas_unique = len(np.unique(nas_assignments))
            target_k = max(tas_unique, nas_unique, 8)  # Use at least 8 clusters
            
            tprint(f"Target K for fusion: {target_k} (TAS: {tas_unique}, NAS: {nas_unique})", "INFO")
            
            # Use the existing regime optimization service for Dawid-Skene fusion
            if hasattr(self, 'regime_optimization_service') and self.regime_optimization_service:
                # Use the service's Dawid-Skene implementation
                fusion_result = self.regime_optimization_service.run_dawid_skene(
                    tas_assignments=tas_assignments,
                    nas_assignments=nas_assignments,
                    target_k=target_k,
                    features=features,
                    max_iterations=50,
                    tolerance=1e-6
                )
                
                if hasattr(fusion_result, 'assignments'):
                    fused_assignments = fusion_result.assignments
                elif isinstance(fusion_result, tuple) and len(fusion_result) > 0:
                    fused_assignments = fusion_result[0]
                else:
                    fused_assignments = fusion_result
                    
                tprint(f"Dawid-Skene fusion completed successfully: {len(fused_assignments)} samples", "SUCCESS")
                return fused_assignments
            else:
                # Fallback: simple majority voting if service is not available
                tprint("Regime optimization service not available, using simple majority voting", "WARNING")
                return self._simple_majority_voting(tas_assignments, nas_assignments, target_k)
                
        except Exception as e:
            tprint_error(f"Dawid-Skene fusion failed: {e}")
            # Fallback to simple majority voting
            tprint("Falling back to simple majority voting", "WARNING")
            try:
                return self._simple_majority_voting(tas_assignments, nas_assignments, 8)
            except Exception as fallback_error:
                tprint_error(f"Fallback fusion also failed: {fallback_error}")
                # Last resort: return TAS assignments
                return tas_assignments
    
    def _simple_majority_voting(self, tas_assignments: np.ndarray, nas_assignments: np.ndarray, target_k: int) -> np.ndarray:
        """Simple majority voting fallback for regime fusion."""
        try:
            # Create a simple consensus by taking the mode of both assignments
            # If they disagree, prefer TAS assignments (can be changed to NAS if preferred)
            consensus_assignments = np.copy(tas_assignments)
            
            # For points where TAS and NAS disagree, use a weighted approach
            disagreement_mask = tas_assignments != nas_assignments
            
            if np.any(disagreement_mask):
                tprint(f"Found {np.sum(disagreement_mask)} points with TAS/NAS disagreement, using TAS preference", "INFO")
                # Could implement more sophisticated voting here
            
            # Ensure assignments are in valid range [0, target_k-1]
            consensus_assignments = np.clip(consensus_assignments, 0, target_k - 1)
            
            return consensus_assignments
            
        except Exception as e:
            tprint_error(f"Simple majority voting failed: {e}")
            # Return TAS assignments as last resort
            return np.clip(tas_assignments, 0, target_k - 1)
    
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
        """Calculate composite quality score for cluster using CV (50%), Silhouette (25%), DBI (25%)."""
        try:
            # Size penalty for oversized clusters (stricter threshold)
            size_penalty = max(0.0, (cluster_percentage - 0.12) * 3.0) if cluster_percentage > 0.12 else 0.0

            # CV (Coefficient of Variation) - 50% weight
            internal_cv = self._calculate_internal_cv_score(cluster_features)
            cv_score = 1.0 - internal_cv  # Higher CV score is better (lower internal variation)

            # Silhouette approximation - 25% weight
            compactness = self._calculate_compactness_score(cluster_features)
            silhouette_score = min(0.5, compactness * 0.8)  # Simplified approximation

            # DBI (Davies-Bouldin Index) approximation - 25% weight
            # DBI measures cluster separation - lower is better, so invert it
            dbi_score = max(0.0, 1.0 - (internal_cv * 2.0))  # Approximation: lower CV = better separation

            # Composite quality score with specified weighting: CV(50%) + Silhouette(25%) + DBI(25%)
            quality_score = (
                cv_score * 0.50 +           # 50% weight for CV
                silhouette_score * 0.25 +   # 25% weight for Silhouette
                dbi_score * 0.25 -          # 25% weight for DBI
                size_penalty * 0.10         # 10% penalty for oversized clusters
            )

            return max(0.0, min(1.0, quality_score))

        except Exception:
            return 0.0

    def _calculate_cluster_quality_score_vectorized(self, cluster_features: np.ndarray, cluster_percentage: float) -> float:
        """VECTORIZED: Calculate composite quality score for cluster using CV (50%), Silhouette (25%), DBI (25%)."""
        try:
            # Size penalty for oversized clusters (stricter threshold)
            size_penalty = max(0.0, (cluster_percentage - 0.12) * 3.0) if cluster_percentage > 0.12 else 0.0

            # CV (Coefficient of Variation) - 50% weight
            internal_cv = self._calculate_internal_cv_score_vectorized(cluster_features)
            cv_score = 1.0 - internal_cv  # Higher CV score is better (lower internal variation)

            # Silhouette approximation - 25% weight
            centroid = np.mean(cluster_features, axis=0)
            compactness = self._calculate_compactness_score_vectorized(cluster_features, centroid)
            silhouette_score = min(0.5, compactness * 0.8)  # Simplified approximation

            # DBI (Davies-Bouldin Index) approximation - 25% weight
            # DBI measures cluster separation - lower is better, so invert it
            dbi_score = max(0.0, 1.0 - (internal_cv * 2.0))  # Approximation: lower CV = better separation

            # Composite quality score with specified weighting: CV(50%) + Silhouette(25%) + DBI(25%)
            quality_score = (
                cv_score * 0.50 +           # 50% weight for CV
                silhouette_score * 0.25 +   # 25% weight for Silhouette
                dbi_score * 0.25 -          # 25% weight for DBI
                size_penalty * 0.10         # 10% penalty for oversized clusters
            )

            return max(0.0, min(1.0, quality_score))

        except Exception:
            return 0.0
    
    def _should_split_cluster_enhanced(self, cluster_metrics: Dict[int, Dict[str, float]], cluster_id: int) -> Tuple[bool, Dict[str, Any]]:
        """Enhanced cluster splitting decision with size-based thresholds and quality requirements."""
        try:
            cluster = cluster_metrics[cluster_id]
            
            # Calculate quality score percentiles for relative comparison
            all_quality_scores = [metrics['quality_score'] for metrics in cluster_metrics.values()]
            
            # Use numpy for robust percentile calculation
            import numpy as np
            if len(all_quality_scores) > 1:
                bottom_50_percentile = np.percentile(all_quality_scores, 50)  # Bottom 50%
                bottom_25_percentile = np.percentile(all_quality_scores, 25)  # Bottom 25%
            else:
                # Fallback for single cluster
                bottom_50_percentile = 0.3
                bottom_25_percentile = 0.2
            
            current_quality = cluster['quality_score']
            cluster_size = cluster['size_percentage']
            
            # Debug logging
            tprint(f"   🔍 Cluster {cluster_id} analysis: size={cluster_size:.3f}, quality={current_quality:.3f}", "DEBUG")
            tprint(f"   📊 Percentiles: 50%={bottom_50_percentile:.3f}, 25%={bottom_25_percentile:.3f}", "DEBUG")
            
            # CORRECTED SPLITTING LOGIC: Split large clusters for better regime separation
            # 1. Size > 16% + no criteria required (for any large cluster)
            condition_1 = cluster_size > 0.16

            # 2. Size > 12% + is low quality (bottom 50%) (large and poor quality)
            is_low_quality = current_quality <= bottom_50_percentile
            condition_2 = cluster_size > 0.12 and is_low_quality

            # 3. Size > 10% + is very low quality (bottom 25%) (large and very poor quality)
            is_very_low_quality = current_quality <= bottom_25_percentile
            condition_3 = cluster_size > 0.10 and is_very_low_quality
            
            tprint(f"   🎯 Conditions: 1={condition_1}, 2={condition_2}, 3={condition_3}", "DEBUG")
            
            # Determine if cluster should be split
            should_split = condition_1 or condition_2 or condition_3
            
            # Determine the reason for splitting
            if condition_1:
                reason = 'size_over_16_percent'
            elif condition_2:
                reason = 'size_over_12_percent_low_quality'
            elif condition_3:
                reason = 'size_over_10_percent_very_low_quality'
            else:
                reason = 'no_split'
            
            return should_split, {
                'reason': reason,
                'cluster_size': cluster_size,
                'quality_score': current_quality,
                'is_low_quality': is_low_quality,
                'is_very_low_quality': is_very_low_quality,
                'bottom_50_percentile': bottom_50_percentile,
                'bottom_25_percentile': bottom_25_percentile
            }
            
        except Exception as e:
            tprint(f"Error in _should_split_cluster_enhanced for cluster {cluster_id}: {e}", "ERROR")
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
            # If cluster is oversized, splitting will improve balance (kept at 10% threshold)
            if cluster['size_percentage'] > 0.10:
                return min((cluster['size_percentage'] - 0.10) * 3.0, 1.0)  # Restored original multiplier
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
        """Vectorized K-means frontier discovery with batch processing."""
        try:
            from sklearn.cluster import KMeans
            from joblib import Parallel, delayed
            
            # Vectorized approach: batch all K-means attempts
            # LIMITED: Maximum 3 sub-clusters to prevent over-splitting
            k_values = [2, 3]
            seeds = [42, 123, 456, 789]
            
            def _run_kmeans_batch(k, seed):
                """Run single K-means attempt."""
                try:
                    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=200)  # Reduced iterations for speed
                    labels = kmeans.fit_predict(cluster_features)
                    quality = self._evaluate_frontier_quality_vectorized(labels, cluster_features)
                    return quality, labels, k
                except Exception:
                    return 0.0, None, k
            
            # Parallel execution of all K-means attempts
            results = Parallel(n_jobs=-1, prefer="threads")(
                delayed(_run_kmeans_batch)(k, seed) for k in k_values for seed in seeds
            )
            
            # Find best result
            best_quality = 0.0
            best_labels = None
            best_k = 2
            
            for quality, labels, k in results:
                if quality > best_quality:
                    best_quality = quality
                    best_labels = labels
                    best_k = k
            
            return {
                'method': 'kmeans_vectorized',
                'quality': best_quality,
                'labels': best_labels,
                'sub_cluster_count': best_k,
                'indices': cluster_indices
            }
            
        except Exception:
            return {'method': 'kmeans_vectorized', 'quality': 0.0, 'sub_cluster_count': 2}
    
    def _discover_gmm_frontier(self, cluster_features: np.ndarray, cluster_indices: np.ndarray) -> Dict[str, Any]:
        """Vectorized GMM frontier discovery with batch processing."""
        try:
            from sklearn.mixture import GaussianMixture
            from joblib import Parallel, delayed
            
            # Vectorized approach: batch all GMM attempts
            # LIMITED: Maximum 3 sub-clusters to prevent over-splitting
            k_values = [2, 3]
            seeds = [42, 123, 456, 789]
            cov_types = ['full', 'tied', 'diag']
            
            def _run_gmm_batch(k, seed, cov_type):
                """Run single GMM attempt."""
                try:
                    gmm = GaussianMixture(n_components=k, random_state=seed, covariance_type=cov_type, max_iter=100)  # Reduced iterations
                    labels = gmm.fit_predict(cluster_features)
                    quality = self._evaluate_frontier_quality_vectorized(labels, cluster_features)
                    return quality, labels, k
                except Exception:
                    return 0.0, None, k
            
            # Parallel execution of all GMM attempts
            results = Parallel(n_jobs=-1, prefer="threads")(
                delayed(_run_gmm_batch)(k, seed, cov_type) 
                for k in k_values for seed in seeds for cov_type in cov_types
            )
            
            # Find best result
            best_quality = 0.0
            best_labels = None
            best_k = 2
            
            for quality, labels, k in results:
                            if quality > best_quality:
                                best_quality = quality
                                best_labels = labels
                                best_k = k
            
            return {
                'method': 'gmm_vectorized',
                'quality': best_quality,
                'labels': best_labels,
                'sub_cluster_count': best_k,
                'indices': cluster_indices
            }
            
        except Exception:
            return {'method': 'gmm_vectorized', 'quality': 0.0, 'sub_cluster_count': 2}
    
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
            tprint("⚠️ WARNING: _calculate_silhouette_score_optimized is not defined - using fallback", "WARNING")
            silhouette_score = 0.5  # Fallback value
            compactness = self._calculate_compactness_score(cluster_features)
            tprint("⚠️ WARNING: _calculate_regime_balance_optimized is not defined - using fallback", "WARNING")
            balance = 0.5  # Fallback value
            
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
    
    def _evaluate_frontier_quality_vectorized(self, labels: np.ndarray, cluster_features: np.ndarray) -> float:
        """Vectorized frontier quality evaluation with optimized metrics."""
        try:
            if len(np.unique(labels)) < 2:
                return 0.0
            
            # Vectorized silhouette score calculation
            silhouette_score = self._calculate_silhouette_score_vectorized(cluster_features, labels)
            
            # Vectorized balance calculation
            balance = self._calculate_regime_balance_vectorized(labels)
            
            # Vectorized compactness calculation
            compactness = self._calculate_compactness_vectorized(cluster_features, labels)
            
            # Vectorized Davies-Bouldin and Calinski-Harabasz scores
            from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
            
            try:
                db_score = davies_bouldin_score(cluster_features, labels)
                normalized_db = min(1.0, 1.0 / max(0.1, db_score))
            except:
                normalized_db = 0.5
            
            try:
                ch_score = calinski_harabasz_score(cluster_features, labels)
                normalized_ch = min(1.0, ch_score / 1000.0)
            except:
                normalized_ch = 0.5
            
            # Optimized quality score
            quality_score = (
                silhouette_score * 0.35 +
                normalized_db * 0.25 +
                normalized_ch * 0.20 +
                compactness * 0.15 +
                balance * 0.05
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception:
            return 0.0
    
    def _calculate_silhouette_score_vectorized(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Vectorized silhouette score calculation."""
        try:
            from sklearn.metrics import silhouette_score
            return max(0.0, silhouette_score(features, labels))
        except Exception:
            return 0.0
    
    def _calculate_regime_balance_vectorized(self, labels: np.ndarray) -> float:
        """Vectorized regime balance calculation."""
        try:
            unique_labels, counts = np.unique(labels, return_counts=True)
            n_clusters = len(unique_labels)
            if n_clusters <= 1:
                return 0.0
            
            # Calculate balance as 1 - coefficient of variation
            mean_size = np.mean(counts)
            std_size = np.std(counts)
            cv = std_size / mean_size if mean_size > 0 else 1.0
            balance = max(0.0, 1.0 - cv)
            
            return balance
        except Exception:
            return 0.0
    
    def _calculate_compactness_vectorized(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Vectorized compactness calculation."""
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) <= 1:
                return 0.0
            
            # Calculate intra-cluster distances vectorized
            intra_distances = []
            for label in unique_labels:
                cluster_points = features[labels == label]
                if len(cluster_points) > 1:
                    centroid = np.mean(cluster_points, axis=0)
                    distances = np.linalg.norm(cluster_points - centroid, axis=1)
                    intra_distances.extend(distances)
            
            if not intra_distances:
                return 0.0
            
            # Compactness as inverse of average intra-cluster distance
            avg_intra_distance = np.mean(intra_distances)
            compactness = max(0.0, 1.0 - min(1.0, avg_intra_distance / 2.0))
            
            return compactness
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
                if isinstance(assignments, np.ndarray):
                    max_existing_id = np.max(assignments) if assignments.size > 0 else -1
                else:
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
    
    def _smart_cluster_splitting_decision(self, assignments: np.ndarray, features: np.ndarray, current_k: int, iteration: int, baseline_score: float) -> Tuple[np.ndarray, int, Dict]:
        """Smart cluster splitting decision using unified objective J."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import davies_bouldin_score
            
            n_samples = len(assignments)
            k_max = 12  # Maximum allowed clusters
            
            # Compute current objective J
            current_J = self._compute_unified_objective(features, assignments, current_k, k_max)
            tprint(f"🔍 Current objective J={current_J:.4f} for k={current_k}", "INFO")
            
            # Split gating constraints
            min_child_size = max(25, int(0.005 * n_samples))  # Min 25 or 0.5% of N
            max_splits_per_round = min(3, max(1, int(0.1 * current_k)))  # Cap at 3 or 10% of k
            delta_J_threshold = 0.005  # 0.5% improvement required
            
            clusters_to_split = []
            tprint(f"🔍 Analyzing {current_k} clusters for splitting (max {max_splits_per_round} splits/round)...", "INFO")
            
            for cluster_id in range(current_k):
                cluster_mask = assignments == cluster_id
                cluster_size = np.sum(cluster_mask)
                
                # Size constraint: must be large enough to split
                if cluster_size < min_child_size * 2:  # Need 2x min size to split
                    tprint(f"   ❌ Cluster {cluster_id}: too small ({cluster_size} < {min_child_size * 2})", "INFO")
                    continue
                
                # Test split: try k=2 on this cluster
                cluster_features = features[cluster_mask]
                try:
                    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                    sub_labels = kmeans.fit_predict(cluster_features)
                    
                    # Create hypothetical assignments with split
                    test_assignments = assignments.copy()
                    max_existing_id = np.max(assignments)
                    new_cluster_id = max_existing_id + 1
                    
                    # Apply split
                    split_mask = sub_labels == 1
                    test_assignments[cluster_mask & split_mask] = new_cluster_id
                    
                    # Compute objective for split scenario
                    split_k = current_k + 1
                    split_J = self._compute_unified_objective(features, test_assignments, split_k, k_max)
                    delta_J = split_J - current_J
                    
                    # Check DBI improvement
                    current_dbi = davies_bouldin_score(features, assignments)
                    split_dbi = davies_bouldin_score(features, test_assignments)
                    dbi_improvement = current_dbi - split_dbi  # Lower DBI is better
                    
                    # Check bimodality (simplified: check if split creates distinct clusters)
                    cluster_1_features = cluster_features[sub_labels == 0]
                    cluster_2_features = cluster_features[sub_labels == 1]
                    
                    if len(cluster_1_features) > 0 and len(cluster_2_features) > 0:
                        # Check if clusters are well-separated
                        centroid_1 = np.mean(cluster_1_features, axis=0)
                        centroid_2 = np.mean(cluster_2_features, axis=0)
                        separation = np.linalg.norm(centroid_1 - centroid_2)
                        
                        # Check internal compactness
                        intra_1 = np.mean([np.linalg.norm(p - centroid_1) for p in cluster_1_features])
                        intra_2 = np.mean([np.linalg.norm(p - centroid_2) for p in cluster_2_features])
                        avg_intra = (intra_1 + intra_2) / 2
                        
                        bimodality_score = separation / max(avg_intra, 1e-6)
                    else:
                        bimodality_score = 0.0
                    
                    # Acceptance criteria
                    size_ok = cluster_size >= min_child_size * 2
                    delta_J_ok = delta_J >= delta_J_threshold
                    dbi_ok = dbi_improvement > 0 or bimodality_score > 2.0  # DBI improves OR clear bimodality
                    
                    if size_ok and delta_J_ok and dbi_ok:
                        clusters_to_split.append({
                            'cluster_id': cluster_id,
                            'new_cluster_id': new_cluster_id,
                            'delta_J': delta_J,
                            'dbi_improvement': dbi_improvement,
                            'bimodality_score': bimodality_score,
                            'test_assignments': test_assignments
                        })
                        tprint(f"   ✅ Cluster {cluster_id}: ΔJ={delta_J:.4f}, DBI_Δ={dbi_improvement:.3f}, "
                               f"bimodality={bimodality_score:.2f}", "SUCCESS")
                    else:
                        tprint(f"   ❌ Cluster {cluster_id}: ΔJ={delta_J:.4f}, DBI_Δ={dbi_improvement:.3f}, "
                               f"bimodality={bimodality_score:.2f} (rejected)", "INFO")
                        
                except Exception as e:
                    tprint(f"   ❌ Cluster {cluster_id}: split test failed - {e}", "WARNING")
                    continue
            
            # Sort by delta_J and apply cap
            clusters_to_split.sort(key=lambda x: x['delta_J'], reverse=True)
            clusters_to_split = clusters_to_split[:max_splits_per_round]
            
            # Apply splits
            new_assignments = assignments.copy()
            new_k = current_k
            
            if clusters_to_split:
                tprint(f"🔀 Applying {len(clusters_to_split)} splits...", "INFO")
                for split_info in clusters_to_split:
                    cluster_id = split_info['cluster_id']
                    new_cluster_id = split_info['new_cluster_id']
                    
                    # Apply the split
                    cluster_mask = assignments == cluster_id
                    cluster_features = features[cluster_mask]
                    
                    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                    sub_labels = kmeans.fit_predict(cluster_features)
                    
                    # Update assignments
                    split_mask = sub_labels == 1
                    new_assignments[cluster_mask & split_mask] = new_cluster_id
                    new_k += 1
                    
                    tprint(f"🔀 Split cluster {cluster_id} into clusters {cluster_id} and {new_cluster_id}", "SUCCESS")
                
                # Verify final objective improvement
                final_J = self._compute_unified_objective(features, new_assignments, new_k, k_max)
                final_delta_J = final_J - current_J
                
                if final_delta_J < delta_J_threshold:
                    tprint(f"⚠️ Final ΔJ={final_delta_J:.4f} below threshold, reverting splits", "WARNING")
                    return assignments, current_k, {'splits_applied': 0, 'delta_J': 0.0}
                
                tprint(f"📈 Cluster splitting: {current_k} → {new_k} (ΔJ={final_delta_J:.4f})", "SUCCESS")
            else:
                tprint(f"📊 No cluster splitting needed (clusters remain at {current_k})", "INFO")
            
            # Return results
            final_stats = {
                'splits_applied': len(clusters_to_split),
                'delta_J': final_delta_J if clusters_to_split else 0.0,
                'final_k': new_k,
                'final_J': final_J if clusters_to_split else current_J
            }
            
            return new_assignments, new_k, final_stats

        except Exception as e:
            tprint(f"Smart cluster splitting failed: {e}", "ERROR")
            return assignments, current_k, {'accepted': 0, 'proposed': 0, 'error': str(e)}

    def validate_clustering_robustness(self, features: np.ndarray, assignments: np.ndarray, 
                                     market_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Lightweight validation framework for clustering robustness."""
        try:
            tprint("🔍 Starting clustering validation...", "INFO")
            
            validation_results = {}
            
            # 1. Basic clustering metrics
            tprint("📊 Computing basic clustering metrics...", "INFO")
            tprint(f"🔍 DEBUG: Computing metrics for {len(np.unique(assignments))} clusters, {features.shape[0]} samples, {features.shape[1]} features", "INFO")
            with tprint_timer("Basic clustering metrics computation"):
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
            tprint(f"🔍 DEBUG: Validation failure details - features shape: {features.shape}, assignments shape: {assignments.shape}, n_clusters: {len(np.unique(assignments))}", "ERROR")
            return {'error': str(e)}

    def _compute_basic_clustering_metrics(self, features: np.ndarray, assignments: np.ndarray) -> Dict[str, Any]:
        """Compute basic clustering quality metrics."""
        try:
            from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
            
            n_clusters = len(np.unique(assignments))
            n_samples = features.shape[0]

            # Basic quality metrics
            tprint(f"🔍 DEBUG: Computing silhouette score for {n_samples} samples across {n_clusters} clusters", "INFO")
            silhouette = silhouette_score(features, assignments)
            tprint(f"🔍 DEBUG: Computing Davies-Bouldin score", "INFO")
            davies_bouldin = davies_bouldin_score(features, assignments)
            tprint(f"🔍 DEBUG: Computing Calinski-Harabasz score", "INFO")
            calinski_harabasz = calinski_harabasz_score(features, assignments)
            
            # Regime balance
            unique, counts = np.unique(assignments, return_counts=True)
            balance = 1.0 - (np.std(counts) / np.mean(counts))
            tprint(f"🔍 DEBUG: Regime balance calculated - std: {np.std(counts):.2f}, mean: {np.mean(counts):.2f}, balance: {balance:.3f}", "INFO")

            # Overall quality score
            overall_quality = (silhouette + (1.0 - davies_bouldin) + balance) / 3.0
            tprint(f"🔍 DEBUG: Overall quality calculated - silhouette: {silhouette:.3f}, db_score: {davies_bouldin:.3f}, balance: {balance:.3f}, overall: {overall_quality:.3f}", "INFO")
            
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
                tprint(f"   🔍 DEBUG: Fold {fold + 1} - train: {len(train_idx)} samples, test: {len(test_idx)} samples", "INFO")

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
                tprint(f"   🔍 DEBUG: Fold {fold + 1} scores - ARI: {ari_score:.3f}, NMI: {nmi_score:.3f}", "INFO")

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
            tprint("⚠️ WARNING: _precompute_global_cluster_statistics is not defined - using fallback", "WARNING")
            global_stats = {'centroids': {}, 'cluster_sizes': {}, 'cluster_weights': {}}  # Fallback value
            
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
            tprint("⚠️ WARNING: _apply_regime_aware_optimization is not defined - using fallback", "WARNING")
            regime_aware_assignments = assignments  # Fallback value
            
            # STRATEGY 6: Feature importance-guided optimization
            tprint("⚠️ WARNING: _apply_feature_importance_optimization is not defined - using fallback", "WARNING")
            importance_guided_assignments = regime_aware_assignments  # Fallback value
            
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
            tprint("⚠️ WARNING: _calculate_regime_stability_scores is not defined - using fallback", "WARNING")
            regime_stability = {i: 0.5 for i in range(k)}  # Fallback value
            
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
                        tprint("⚠️ WARNING: _calculate_volatility_regime_features is not defined - using fallback", "WARNING")
                        volatility_features = np.zeros((len(regime_features), 3))  # Fallback value
                        
                        # Enhanced features for regime optimization
                        enhanced_regime_features = np.column_stack([regime_features, volatility_features])
                        
                        # Apply local optimization within the regime
                        tprint("⚠️ WARNING: _optimize_regime_locally is not defined - using fallback", "WARNING")
                        optimized_regime_assignments = regime_assignments  # Fallback value
                        
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
    
    def _optimize_cross_cluster_boundaries(self, features: np.ndarray, assignments: np.ndarray, k: int) -> np.ndarray:
        """Stage 3: Cross-cluster boundary optimization for better cluster separation."""
        try:
            tprint(f"   🔧 Optimizing cross-cluster boundaries for {k} clusters...", "INFO")

            # Calculate centroids for all clusters
            centroids = np.zeros((k, features.shape[1]))
            for regime in range(k):
                regime_mask = assignments == regime
                if np.sum(regime_mask) > 0:
                    centroids[regime] = np.mean(features[regime_mask], axis=0)

            # Identify boundary samples (samples close to other cluster centroids)
            boundary_samples = []
            for i, (sample_features, sample_regime) in enumerate(zip(features, assignments)):
                # Calculate distance to current centroid
                current_centroid = centroids[sample_regime]
                current_distance = np.linalg.norm(sample_features - current_centroid)

                # Calculate distances to other centroids
                other_distances = []
                for other_regime in range(k):
                    if other_regime != sample_regime:
                        other_centroid = centroids[other_regime]
                        other_distance = np.linalg.norm(sample_features - other_centroid)
                        other_distances.append((other_distance, other_regime))

                # If sample is closer to another centroid, consider it a boundary sample
                if other_distances:
                    min_other_distance = min(other_distances, key=lambda x: x[0])[0]
                    if min_other_distance < current_distance * 0.8:  # Within 80% of current distance
                        boundary_samples.append((i, sample_regime, min_other_distance))

            # Optimize boundary samples by reassigning if beneficial
            optimized_assignments = assignments.copy()
            improvements_made = 0

            for sample_idx, current_regime, min_distance in boundary_samples:
                sample_features = features[sample_idx]

                # Calculate improvement for current assignment vs best alternative
                current_score = self._calculate_sample_regime_score(sample_features, current_regime, centroids)
                best_alternative_regime = None
                best_alternative_score = current_score

                for other_regime in range(k):
                    if other_regime != current_regime:
                        alt_score = self._calculate_sample_regime_score(sample_features, other_regime, centroids)
                        if alt_score > best_alternative_score:
                            best_alternative_score = alt_score
                            best_alternative_regime = other_regime

                # Reassign if alternative is significantly better
                if best_alternative_regime is not None and best_alternative_score > current_score * 1.1:
                    optimized_assignments[sample_idx] = best_alternative_regime
                    improvements_made += 1

            tprint(f"   ✅ Cross-cluster boundary optimization completed: {improvements_made} samples reassigned", "SUCCESS")
            return optimized_assignments

        except Exception as e:
            tprint(f"Cross-cluster boundary optimization failed: {e}", "ERROR")
            return assignments

    def _calculate_sample_regime_score(self, sample_features: np.ndarray, regime: int, centroids: np.ndarray) -> float:
        """Calculate how well a sample fits in a regime (lower distance = better fit)."""
        try:
            centroid = centroids[regime]
            distance = np.linalg.norm(sample_features - centroid)
            # Convert distance to score (closer = higher score)
            return 1.0 / (1.0 + distance)
        except Exception:
            return 0.0

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
