"""
Feature Service for NAS-TAS Clustering.

This module provides feature preparation, scaling, and embedding services
that wrap FeaturePreprocessor, FeatureSelector, and FeatureAnalyzer.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

# Mac M1 Hardware Optimizations
HARDWARE_OPTIMIZATIONS_AVAILABLE = False
try:
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    HARDWARE_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    pass

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error
)

from .shared_utils import (
    get_logger,
    prepare_market_features,
    FeatureConfig
)

# Import comprehensive utility functions
from src.utils.common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
    safe_float, safe_int, validate_finite, validate_positive, validate_range,
    safe_rolling, safe_groupby_operation, safe_apply_function, safe_filter_dataframe,
    create_summary_statistics, format_bytes, chunked_iterable, parallel_map,
    timed_operation, get_current_datetime, format_datetime, parse_datetime,
    ensure_directory, safe_file_exists, safe_json_dump, safe_json_load,
    optimize_dataframe_dtypes, calculate_data_quality_metrics, get_dataframe_info,
    create_data_quality_report, math_safe, validate_correlation_matrix,
    safe_matrix_inverse, safe_kelly_calculation, safe_weighted_average,
    safe_percentage_change, safe_resample, align_dataframes,
    validate_dataframe_schema, guard_dataframe_nulls, sanitize_string,
    memory_checkpoint, gpu_context, optimize_memory, get_memory_usage,
    validate_file_path, get_file_size, check_disk_space, get_logger,
    integrate_with_m1_optimizers, cleanup_m1_optimizers, get_m1_gpu_manager,
    get_m1_memory_optimizer, get_m1_cpu_optimizer, is_m1_available, is_mps_available
)

from src.utils.common_utilities import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    analyze_nan_values_detailed, safe_apply_with_validation, safe_aggregate_data,
    safe_merge_dataframes, safe_drop_columns, safe_fillna, safe_dropna,
    safe_reset_index, safe_sort_values, safe_groupby_agg, safe_pivot_table,
    safe_melt_dataframe, safe_concat_dataframes, safe_join_dataframes,
    safe_apply_custom_function, safe_transform_dataframe, safe_validate_dataframe,
    safe_export_dataframe, safe_import_dataframe, safe_compress_dataframe,
    safe_decompress_dataframe, safe_serialize_dataframe, safe_deserialize_dataframe,
    calculate_data_quality_score, detect_data_anomalies, validate_data_consistency,
    clean_data_automatically, standardize_data_format, validate_data_types,
    check_data_completeness, validate_data_ranges, detect_outliers,
    validate_data_relationships, check_data_duplicates, validate_data_integrity,
    optimize_dataframe_performance, reduce_memory_usage, optimize_dtypes,
    compress_dataframe, decompress_dataframe, cache_dataframe, load_cached_dataframe,
    get_hardware_info, optimize_for_hardware, get_memory_usage, get_cpu_usage,
    get_gpu_usage, optimize_memory_allocation, optimize_cpu_usage, optimize_gpu_usage
)

from src.utils.math_validation import (
    MathValidationError, safe_divide as math_safe_divide, safe_log as math_safe_log,
    safe_sqrt as math_safe_sqrt, safe_power as math_safe_power,
    validate_finite as math_validate_finite, validate_positive as math_validate_positive,
    validate_range as math_validate_range, validate_numeric_array as math_validate_numeric_array,
    validate_array_finite, validate_scalar_finite, validate_matrix_finite,
    safe_matrix_operations, validate_correlation_matrix as math_validate_correlation_matrix,
    safe_eigenvalue_decomposition, safe_svd_decomposition, safe_cholesky_decomposition
)

# Import hardware utilities
try:
    from src.utils.hardware.optimization_decorators import (
        smart_cache, auto_optimize, memory_efficient, performance_tracked
    )
    from src.utils.hardware.memory_optimized_decorators import (
        memory_optimized, comprehensive_memory_optimization, MemoryOptimizationLevel
    )
    from src.utils.hardware.integrated_hardware_manager import get_integrated_hardware_manager
    from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager
    from src.utils.hardware.vectorbt_gpu_accelerator import VectorBTRollingOptimizer, UnifiedVectorizationManager
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    smart_cache = lambda *args, **kwargs: lambda f: f
    auto_optimize = lambda *args, **kwargs: lambda f: f
    memory_efficient = lambda *args, **kwargs: lambda f: f
    performance_tracked = lambda *args, **kwargs: lambda f: f
    memory_optimized = lambda *args, **kwargs: lambda f: f
    comprehensive_memory_optimization = lambda *args, **kwargs: lambda f: f
    MemoryOptimizationLevel = type('MemoryOptimizationLevel', (), {})
    get_integrated_hardware_manager = lambda: None
    UnifiedHardwareManager = None
    VectorBTRollingOptimizer = None
    UnifiedVectorizationManager = None
    HARDWARE_OPTIMIZATION_AVAILABLE = False

# Import ML common utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
    from src.utils.ml_common.optimization.grid_utils import GridSearchOptimizer
    from src.utils.ml_common.optimization.hpo_utils import HPOConfig, HPOOptimizer
    from src.utils.ml_common.cross_validation import PurgedKFold, TimeSeriesSplit
    from src.utils.ml_common.model_validation import ModelValidator, ValidationMetrics
    from src.utils.ml_common.feature_importance import SHAPExplainer, LIMEExplainer
    from src.utils.ml_common.data_leakage import DataLeakageDetector
    from src.utils.ml_common.lookahead_bias import LookaheadBiasDetector
    ML_COMMON_AVAILABLE = True
except ImportError:
    BayesianTPEOptimizer = None
    GridSearchOptimizer = None
    HPOConfig = None
    HPOOptimizer = None
    PurgedKFold = None
    TimeSeriesSplit = None
    ModelValidator = None
    ValidationMetrics = None
    SHAPExplainer = None
    LIMEExplainer = None
    DataLeakageDetector = None
    LookaheadBiasDetector = None
    ML_COMMON_AVAILABLE = False

# Import data utilities
try:
    from src.utils.data.klines_parquet import KlinesParquetManager
    from src.utils.data.unified_data_utils import UnifiedDataManager
    from src.utils.data.feature_engineer import FeatureEngineer
    from src.utils.data.historical_data_pipeline import HistoricalDataPipeline
    DATA_UTILS_AVAILABLE = True
except ImportError:
    KlinesParquetManager = None
    UnifiedDataManager = None
    FeatureEngineer = None
    HistoricalDataPipeline = None
    DATA_UTILS_AVAILABLE = False

# Import artifact manager
try:
    from src.utils.artifact_manager import ArtifactManager
    from src.utils.enhanced_artifact_manager import EnhancedArtifactManager
    ARTIFACT_MANAGER_AVAILABLE = True
except ImportError:
    ArtifactManager = None
    EnhancedArtifactManager = None
    ARTIFACT_MANAGER_AVAILABLE = False

@dataclass
class FeaturePreparationResult:
    """Result from feature preparation."""
    features: np.ndarray
    feature_names: List[str]
    feature_scores: Dict[str, float]
    dropped_features: List[str]
    preparation_time: float
    metadata: Dict[str, Any]

class FeatureService:
    """
    Feature service that wraps FeaturePreprocessor, FeatureSelector, and FeatureAnalyzer.

    Responsibilities:
    - Wrap FeaturePreprocessor, FeatureSelector, and FeatureAnalyzer
    - Handle scaling (RobustScaler), PCA/UMAP embedding
    - Expose API: prepare_features(data) → clean feature matrix ready for clustering
    """

    def __init__(self, verbose: bool = True, enable_hardware_optimization: bool = True, 
                 enable_ml_optimization: bool = True, enable_data_validation: bool = True):
        """Initialize the enhanced feature service with comprehensive utility integrations."""
        self.verbose = verbose
        self.logger = get_logger('FeatureService')
        self.enable_hardware_optimization = enable_hardware_optimization and HARDWARE_OPTIMIZATION_AVAILABLE
        self.enable_ml_optimization = enable_ml_optimization and ML_COMMON_AVAILABLE
        self.enable_data_validation = enable_data_validation

        # Feature preparation components
        self.scaler = None
        self.pca = None
        self.umap_reducer = None

        # Initialize hardware manager if available
        if self.enable_hardware_optimization:
            try:
                self.hardware_manager = get_integrated_hardware_manager()
                self.vectorbt_optimizer = VectorBTRollingOptimizer() if VectorBTRollingOptimizer else None
                self.vectorization_manager = UnifiedVectorizationManager() if UnifiedVectorizationManager else None
                tprint_info("Hardware optimization enabled for feature service")
            except Exception as e:
                tprint_warning(f"Failed to initialize hardware manager: {e}")
                self.hardware_manager = None
                self.vectorbt_optimizer = None
                self.vectorization_manager = None
        else:
            self.hardware_manager = None
            self.vectorbt_optimizer = None
            self.vectorization_manager = None

        # Hardware service integration
        try:
            from .hardware_service import HardwareService
            self.hardware_service = HardwareService(verbose=self.verbose)
            self.hardware_integration_enabled = True
        except ImportError:
            self.hardware_service = None
            self.hardware_integration_enabled = False

        # Mac M1 Hardware Optimizations
        self.memory_optimizer = None
        self.cpu_optimizer = None

        if HARDWARE_OPTIMIZATIONS_AVAILABLE:
            try:
                self.memory_optimizer = get_m1_memory_optimizer(memory_limit_gb=2.0)  # Conservative limit for feature processing
                self.cpu_optimizer = get_m1_cpu_optimizer()
                self.cpu_optimizer.set_conservative_mode()  # Use conservative mode for feature processing
                tprint("🧠 Mac M1 hardware optimizations initialized for feature service", "INFO")
            except Exception as e:
                tprint(f"⚠️ Failed to initialize hardware optimizations: {e}", "WARNING")

        # Initialize ML optimization components if available
        if self.enable_ml_optimization:
            try:
                self.bayesian_optimizer = BayesianTPEOptimizer() if BayesianTPEOptimizer else None
                self.grid_optimizer = GridSearchOptimizer() if GridSearchOptimizer else None
                self.hpo_optimizer = HPOOptimizer() if HPOOptimizer else None
                self.model_validator = ModelValidator() if ModelValidator else None
                self.data_leakage_detector = DataLeakageDetector() if DataLeakageDetector else None
                self.lookahead_bias_detector = LookaheadBiasDetector() if LookaheadBiasDetector else None
                self.shap_explainer = SHAPExplainer() if SHAPExplainer else None
                self.lime_explainer = LIMEExplainer() if LIMEExplainer else None
                tprint_info("ML optimization enabled for feature service")
            except Exception as e:
                tprint_warning(f"Failed to initialize ML optimization: {e}")
                self.bayesian_optimizer = None
                self.grid_optimizer = None
                self.hpo_optimizer = None
                self.model_validator = None
                self.data_leakage_detector = None
                self.lookahead_bias_detector = None
                self.shap_explainer = None
                self.lime_explainer = None
        else:
            self.bayesian_optimizer = None
            self.grid_optimizer = None
            self.hpo_optimizer = None
            self.model_validator = None
            self.data_leakage_detector = None
            self.lookahead_bias_detector = None
            self.shap_explainer = None
            self.lime_explainer = None

        # Initialize data utilities if available
        if DATA_UTILS_AVAILABLE:
            try:
                self.klines_manager = KlinesParquetManager() if KlinesParquetManager else None
                self.data_manager = UnifiedDataManager() if UnifiedDataManager else None
                self.feature_engineer = FeatureEngineer() if FeatureEngineer else None
                self.historical_pipeline = HistoricalDataPipeline() if HistoricalDataPipeline else None
                tprint_info("Data utilities enabled for feature service")
            except Exception as e:
                tprint_warning(f"Failed to initialize data utilities: {e}")
                self.klines_manager = None
                self.data_manager = None
                self.feature_engineer = None
                self.historical_pipeline = None
        else:
            self.klines_manager = None
            self.data_manager = None
            self.feature_engineer = None
            self.historical_pipeline = None

        # Initialize artifact manager if available
        if ARTIFACT_MANAGER_AVAILABLE:
            try:
                self.artifact_manager = EnhancedArtifactManager() if EnhancedArtifactManager else ArtifactManager()
                tprint_info("Artifact manager enabled for feature service")
            except Exception as e:
                tprint_warning(f"Failed to initialize artifact manager: {e}")
                self.artifact_manager = None
        else:
            self.artifact_manager = None

        # Feature tracking
        self.feature_history = []
        self.performance_metrics = {
            "total_preparation_time": 0.0,
            "scaling_time": 0.0,
            "embedding_time": 0.0,
            "feature_reduction_rate": 0.0,
            "hardware_accelerations": 0,
            "memory_optimizations": 0,
            "data_quality_checks": 0,
            "ml_optimizations": 0,
            "artifact_saves": 0
        }

    @performance_tracked(log_performance=True, track_memory=True) if HARDWARE_OPTIMIZATION_AVAILABLE else lambda x: x
    @memory_optimized(level=MemoryOptimizationLevel.BALANCED) if HARDWARE_OPTIMIZATION_AVAILABLE else lambda x: x
    async def prepare_features(
        self,
        market_data: pd.DataFrame,
        config: Any = None
    ) -> FeaturePreparationResult:
        """
        Enhanced feature preparation with comprehensive utility integrations.

        Args:
            market_data: Market data for feature extraction
            config: Configuration parameters

        Returns:
            FeaturePreparationResult with clean feature matrix and comprehensive metadata
        """
        try:
            start_time = time.time()
            tprint_info("Starting enhanced feature preparation with comprehensive utility integrations")

            # Step 1: Data validation and quality checks
            if self.enable_data_validation:
                tprint_info("Performing comprehensive data validation and quality checks")
                validation_results = await self._validate_input_data(market_data)
                if not validation_results.get('valid', False):
                    tprint_warning(f"Data validation failed: {validation_results.get('errors', [])}")
                self.performance_metrics["data_quality_checks"] += 1

            # Step 2: Data preprocessing and optimization
            tprint_info("Applying data preprocessing and optimization")
            optimized_data = await self._preprocess_and_optimize_data(market_data)

            # Step 3: Start memory monitoring for feature preparation
            if self.memory_optimizer:
                try:
                    self.memory_optimizer.start_monitoring()
                    tprint_info("Memory monitoring started for feature preparation")
                except Exception as e:
                    tprint_warning(f"Memory monitoring failed: {e}")

            # Step 4: Optimize market data for memory efficiency
            if self.memory_optimizer and hasattr(optimized_data, 'memory_usage'):
                try:
                    optimized_data = self.memory_optimizer.optimize_dataframe_memory(optimized_data)
                    tprint_info("Market data memory optimized for feature preparation")
                except Exception as e:
                    tprint_warning(f"Data optimization failed: {e}")

            # Step 5: Extract features using shared utilities
            feature_config = self._create_enhanced_feature_config(config)
            shared_result = await self._prepare_features_shared(optimized_data, feature_config)

            # Step 6: Validate shared result
            if shared_result is None or not hasattr(shared_result, 'features') or shared_result.features is None:
                raise ValueError("Shared feature preparation returned None or invalid result")

            if shared_result.features.size == 0:
                raise ValueError("Shared feature preparation returned empty features array")

            # Step 7: Apply enhanced scaling and normalization
            scaled_features, scaling_time = await self._apply_enhanced_scaling(shared_result.features)

            # Step 8: Apply dimensionality reduction (PCA/UMAP) with ML optimization
            final_features, embedding_time = await self._apply_enhanced_embedding(
                scaled_features, shared_result.feature_names, config
            )

            # Step 9: Apply ML-specific optimizations if available
            if self.enable_ml_optimization:
                tprint_info("Applying ML-specific optimizations")
                final_features = await self._apply_ml_optimizations(final_features, optimized_data)
                self.performance_metrics["ml_optimizations"] += 1

            # Step 10: Validate final features with comprehensive checks
            validation_results = await self._validate_final_features(final_features, optimized_data)

            # Step 11: Calculate feature importance if available
            feature_importance = await self._calculate_feature_importance(final_features, optimized_data)

            # Step 12: Record performance metrics
            total_time = time.time() - start_time
            self.performance_metrics["total_preparation_time"] = total_time
            self.performance_metrics["scaling_time"] = scaling_time
            self.performance_metrics["embedding_time"] = embedding_time

            # Calculate feature reduction rate
            original_count = shared_result.features.shape[1]
            final_count = final_features.shape[1]
            reduction_rate = (original_count - final_count) / original_count
            self.performance_metrics["feature_reduction_rate"] = reduction_rate

            # Step 13: Generate comprehensive metadata
            metadata = await self._generate_comprehensive_metadata(
                original_count, final_count, reduction_rate, scaling_time, embedding_time,
                validation_results, feature_importance, optimized_data
            )

            # Step 14: Create feature names
            feature_names = self._generate_feature_names(final_features, shared_result.feature_names)

            # Step 15: Create enhanced result
            result = FeaturePreparationResult(
                features=final_features,
                feature_names=feature_names,
                feature_scores=feature_importance,
                dropped_features=[],
                preparation_time=total_time,
                metadata=metadata
            )

            # Step 16: Track feature history and save artifacts
            self._track_feature_preparation(result)
            await self._save_feature_artifacts(result, config)

            # Step 17: Final memory cleanup
            if self.memory_optimizer:
                try:
                    self.memory_optimizer.force_garbage_collection()
                    tprint_info("Final memory cleanup completed for feature preparation")
                except Exception as e:
                    tprint_warning(f"Final cleanup failed: {e}")

            # Step 18: Stop memory monitoring
            if self.memory_optimizer:
                try:
                    self.memory_optimizer.stop_monitoring()
                    tprint_info("Memory monitoring stopped for feature preparation")
                except Exception as e:
                    tprint_warning(f"Memory monitoring stop failed: {e}")

            # Step 19: Log comprehensive summary
            tprint_success(f"Enhanced feature preparation completed in {total_time:.2f}s")
            tprint_info(f"Features: {original_count} → {final_count} (reduction: {reduction_rate:.1%})")
            tprint_info(f"Data quality checks: {self.performance_metrics['data_quality_checks']}")
            tprint_info(f"ML optimizations: {self.performance_metrics['ml_optimizations']}")
            tprint_info(f"Hardware accelerations: {self.performance_metrics['hardware_accelerations']}")

            return result

        except Exception as e:
            tprint_error(f"Enhanced feature preparation failed: {e}")
            raise ValueError(f"Enhanced feature preparation failed: {e}")

    def _create_feature_config(self, config: Any) -> FeatureConfig:
        """Create feature configuration from provided config."""
        return FeatureConfig(
            feature_categories=getattr(config, 'feature_categories', [
                'regime_volatility',
                'regime_volume',
                'regime_structural_trend',
                'regime_statistical'
            ]),
            use_standardized_features=getattr(config, 'use_standardized_features', True),
            drop_highly_correlated=getattr(config, 'drop_highly_correlated', True),
            correlation_threshold=getattr(config, 'correlation_threshold', 0.95)
        )

    def _create_enhanced_feature_config(self, config: Any) -> FeatureConfig:
        """Create enhanced feature configuration with comprehensive options."""
        return FeatureConfig(
            n_features=getattr(config, 'n_features', 50),
            use_pca=getattr(config, 'use_pca', True),
            pca_components=getattr(config, 'pca_components', 20),
            use_umap=getattr(config, 'use_umap', False),
            umap_components=getattr(config, 'umap_components', 10),
            scaler_type=getattr(config, 'scaler_type', 'robust'),
            enable_hardware_optimization=self.enable_hardware_optimization,
            enable_ml_optimization=self.enable_ml_optimization,
            enable_data_validation=self.enable_data_validation,
            enable_feature_engineering=getattr(config, 'enable_feature_engineering', True),
            enable_anomaly_detection=getattr(config, 'enable_anomaly_detection', True),
            enable_outlier_detection=getattr(config, 'enable_outlier_detection', True),
            enable_data_quality_checks=getattr(config, 'enable_data_quality_checks', True),
            enable_memory_optimization=getattr(config, 'enable_memory_optimization', True),
            enable_caching=getattr(config, 'enable_caching', True),
            feature_selection_method=getattr(config, 'feature_selection_method', 'variance_threshold'),
            feature_selection_threshold=getattr(config, 'feature_selection_threshold', 0.01),
            feature_correlation_threshold=getattr(config, 'feature_correlation_threshold', 0.95),
            feature_importance_method=getattr(config, 'feature_importance_method', 'mutual_info'),
            handle_missing_values=getattr(config, 'handle_missing_values', 'median'),
            handle_outliers=getattr(config, 'handle_outliers', 'iqr'),
            outlier_threshold=getattr(config, 'outlier_threshold', 1.5),
            validate_data_consistency=getattr(config, 'validate_data_consistency', True),
            check_data_leakage=getattr(config, 'check_data_leakage', True),
            check_lookahead_bias=getattr(config, 'check_lookahead_bias', True),
            use_parallel_processing=getattr(config, 'use_parallel_processing', True),
            max_workers=getattr(config, 'max_workers', 4),
            chunk_size=getattr(config, 'chunk_size', 1000)
        )

    async def _validate_input_data(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate input data with comprehensive checks."""
        try:
            validation_results = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'data_quality_metrics': {}
            }

            # Basic validation
            if market_data is None or len(market_data) == 0:
                validation_results['valid'] = False
                validation_results['errors'].append("Market data is None or empty")
                return validation_results

            # Data quality metrics
            if self.enable_data_validation:
                try:
                    data_quality_metrics = calculate_data_quality_metrics(market_data)
                    validation_results['data_quality_metrics'] = data_quality_metrics

                    # Check for data anomalies
                    if self.enable_ml_optimization and self.data_leakage_detector:
                        try:
                            leakage_score = self.data_leakage_detector.detect_leakage(market_data)
                            if leakage_score > 0.1:
                                validation_results['warnings'].append(f"Potential data leakage detected: {leakage_score:.3f}")
                        except Exception as e:
                            validation_results['warnings'].append(f"Data leakage detection failed: {e}")

                    # Check for lookahead bias
                    if self.enable_ml_optimization and self.lookahead_bias_detector:
                        try:
                            bias_score = self.lookahead_bias_detector.detect_bias(market_data)
                            if bias_score > 0.05:
                                validation_results['warnings'].append(f"Potential lookahead bias detected: {bias_score:.3f}")
                        except Exception as e:
                            validation_results['warnings'].append(f"Lookahead bias detection failed: {e}")

                except Exception as e:
                    validation_results['warnings'].append(f"Data quality analysis failed: {e}")

            return validation_results

        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Validation failed: {e}"],
                'warnings': [],
                'data_quality_metrics': {}
            }

    async def _preprocess_and_optimize_data(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess and optimize data for feature preparation."""
        try:
            tprint_info("Preprocessing and optimizing data")

            # Create a copy to avoid modifying original data
            processed_data = market_data.copy()

            # Apply data quality improvements
            if self.enable_data_validation:
                # Clean data automatically
                processed_data = clean_data_automatically(processed_data)

                # Standardize data format
                processed_data = standardize_data_format(processed_data)

                # Validate data types
                processed_data = validate_data_types(processed_data)

            # Optimize data types and memory usage
            if self.enable_hardware_optimization:
                processed_data = optimize_dataframe_dtypes(processed_data)
                processed_data = reduce_memory_usage(processed_data)

            # Apply hardware-specific optimizations
            if self.hardware_manager:
                try:
                    processed_data = self.hardware_manager.optimize_dataframe(processed_data)
                except Exception as e:
                    tprint_warning(f"Hardware optimization failed: {e}")

            return processed_data

        except Exception as e:
            tprint_warning(f"Data preprocessing failed: {e}")
            return market_data

    async def _apply_enhanced_scaling(self, features: np.ndarray) -> Tuple[np.ndarray, float]:
        """Apply enhanced scaling with hardware optimization."""
        try:
            start_time = time.time()
            tprint_info("Applying enhanced feature scaling")

            # Validate input features
            math_validate_numeric_array(features, "scaling_features")

            # Apply memory optimization if hardware service is available
            if self.hardware_integration_enabled and self.hardware_service:
                try:
                    features, optimization_info = self.hardware_service.optimize_memory(features)
                    if optimization_info.get("hardware_optimization_used", False):
                        self.performance_metrics["memory_optimizations"] += 1
                        tprint_info("Memory optimization applied during scaling")
                except Exception as e:
                    tprint_warning(f"Memory optimization failed during scaling: {e}")

            # Use hardware-optimized scaling if available
            if self.enable_hardware_optimization and self.vectorization_manager:
                try:
                    scaled_features = self.vectorization_manager.scale_features(features, method='robust')
                    scaling_time = time.time() - start_time
                    tprint_success(f"Hardware-optimized scaling completed in {scaling_time:.3f}s")
                    return scaled_features, scaling_time
                except Exception as e:
                    tprint_warning(f"Hardware-optimized scaling failed: {e}")

            # Fallback to standard scaling
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
            scaled_features = self.scaler.fit_transform(features)

            scaling_time = time.time() - start_time
            tprint_success(f"Standard scaling completed in {scaling_time:.3f}s")
            return scaled_features, scaling_time

        except Exception as e:
            tprint_error(f"Enhanced feature scaling failed: {e}")
            raise

    async def _apply_enhanced_embedding(
        self,
        features: np.ndarray,
        feature_names: List[str],
        config: Any
    ) -> Tuple[np.ndarray, float]:
        """Apply enhanced dimensionality reduction with ML optimization."""
        try:
            start_time = time.time()
            tprint_info("Applying enhanced dimensionality reduction")

            # Check if dimensionality reduction is needed
            n_features = features.shape[1]
            n_samples = features.shape[0]
            target_features = getattr(config, 'target_features', min(20, n_features - 1))

            if n_features <= target_features:
                tprint_info(f"No reduction needed: {n_features} features")
                return features, 0.0

            # Use hardware-optimized embedding if available
            if self.enable_hardware_optimization and self.vectorization_manager:
                try:
                    # Try UMAP first
                    if getattr(config, 'use_umap', False):
                        reduced_features = self.vectorization_manager.reduce_dimensions(
                            features, method='umap', n_components=target_features
                        )
                    else:
                        # Use PCA
                        reduced_features = self.vectorization_manager.reduce_dimensions(
                            features, method='pca', n_components=target_features
                        )
                    
                    embedding_time = time.time() - start_time
                    tprint_success(f"Hardware-optimized embedding completed in {embedding_time:.3f}s")
                    return reduced_features, embedding_time
                except Exception as e:
                    tprint_warning(f"Hardware-optimized embedding failed: {e}")

            # Fallback to standard embedding methods
            return await self._apply_standard_embedding(features, feature_names, config)

        except Exception as e:
            tprint_error(f"Enhanced embedding failed: {e}")
            return features, 0.0

    async def _apply_ml_optimizations(self, features: np.ndarray, market_data: pd.DataFrame) -> np.ndarray:
        """Apply ML-specific optimizations to features."""
        try:
            if not self.enable_ml_optimization:
                return features

            tprint_info("Applying ML-specific optimizations")

            # Feature importance analysis
            if self.shap_explainer:
                try:
                    importance_scores = self.shap_explainer.calculate_feature_importance(features, market_data)
                    tprint_info(f"Feature importance calculated using SHAP")
                except Exception as e:
                    tprint_warning(f"SHAP analysis failed: {e}")

            # Model validation
            if self.model_validator:
                try:
                    validation_results = self.model_validator.validate_features(features, market_data)
                    tprint_info(f"Feature validation completed")
                except Exception as e:
                    tprint_warning(f"Feature validation failed: {e}")

            return features

        except Exception as e:
            tprint_warning(f"ML optimizations failed: {e}")
            return features

    async def _validate_final_features(self, features: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate final features with comprehensive checks."""
        try:
            validation_results = {
                'valid': True,
                'passed': True,
                'errors': [],
                'warnings': [],
                'metrics': {}
            }

            # Basic validation
            if features is None or features.size == 0:
                validation_results['valid'] = False
                validation_results['passed'] = False
                validation_results['errors'].append("Features are None or empty")
                return validation_results

            # Validate finite values
            try:
                math_validate_numeric_array(features, "final_features")
                validation_results['metrics']['finite_values'] = True
            except Exception as e:
                validation_results['warnings'].append(f"Non-finite values detected: {e}")

            # Check for data leakage
            if self.enable_ml_optimization and self.data_leakage_detector:
                try:
                    leakage_score = self.data_leakage_detector.detect_leakage(features)
                    validation_results['metrics']['leakage_score'] = leakage_score
                    if leakage_score > 0.1:
                        validation_results['warnings'].append(f"Potential data leakage: {leakage_score:.3f}")
                except Exception as e:
                    validation_results['warnings'].append(f"Leakage detection failed: {e}")

            return validation_results

        except Exception as e:
            return {
                'valid': False,
                'passed': False,
                'errors': [f"Validation failed: {e}"],
                'warnings': [],
                'metrics': {}
            }

    async def _calculate_feature_importance(self, features: np.ndarray, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance using available methods."""
        try:
            if not self.enable_ml_optimization:
                return {}

            feature_importance = {}

            # Use SHAP if available
            if self.shap_explainer:
                try:
                    importance_scores = self.shap_explainer.calculate_feature_importance(features, market_data)
                    feature_importance.update(importance_scores)
                except Exception as e:
                    tprint_warning(f"SHAP importance calculation failed: {e}")

            # Use LIME if available
            if self.lime_explainer:
                try:
                    lime_scores = self.lime_explainer.calculate_feature_importance(features, market_data)
                    feature_importance.update(lime_scores)
                except Exception as e:
                    tprint_warning(f"LIME importance calculation failed: {e}")

            return feature_importance

        except Exception as e:
            tprint_warning(f"Feature importance calculation failed: {e}")
            return {}

    async def _generate_comprehensive_metadata(
        self, original_count: int, final_count: int, reduction_rate: float,
        scaling_time: float, embedding_time: float, validation_results: Dict[str, Any],
        feature_importance: Dict[str, float], market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate comprehensive metadata for the feature preparation result."""
        try:
            metadata = {
                "original_feature_count": original_count,
                "final_feature_count": final_count,
                "feature_reduction_rate": reduction_rate,
                "scaling_time": scaling_time,
                "scaling_method": "robust",
                "embedding_time": embedding_time,
                "embedding_method": self._get_embedding_method(),
                "validation_passed": validation_results.get("passed", True),
                "validation_results": validation_results,
                "feature_importance": feature_importance,
                "performance_metrics": self.performance_metrics,
                "hardware_optimization_enabled": self.enable_hardware_optimization,
                "ml_optimization_enabled": self.enable_ml_optimization,
                "data_validation_enabled": self.enable_data_validation,
                "timestamp": get_current_datetime(),
                "data_shape": market_data.shape,
                "data_types": market_data.dtypes.to_dict() if hasattr(market_data, 'dtypes') else {}
            }

            # Add hardware-specific metadata
            if self.hardware_manager:
                try:
                    hardware_info = self.hardware_manager.get_optimization_info()
                    metadata["hardware_info"] = hardware_info
                except Exception as e:
                    tprint_warning(f"Failed to get hardware info: {e}")

            return metadata

        except Exception as e:
            tprint_warning(f"Metadata generation failed: {e}")
            return {}

    def _generate_feature_names(self, features: np.ndarray, original_names: List[str]) -> List[str]:
        """Generate appropriate feature names for the final features."""
        try:
            n_features = features.shape[1]
            
            if n_features == len(original_names):
                # No dimensionality reduction, use original names
                return original_names
            else:
                # Dimensionality reduction applied, create meaningful names
                method = self._get_embedding_method()
                if method == "PCA":
                    if hasattr(self, 'pca') and self.pca is not None:
                        return [f"PC{i+1}_var{self.pca.explained_variance_ratio_[i]:.3f}"
                               for i in range(n_features)]
                    else:
                        return [f"PC{i+1}" for i in range(n_features)]
                elif method == "UMAP":
                    return [f"UMAP_dim{i+1}" for i in range(n_features)]
                else:
                    return [f"embedding_{i+1}" for i in range(n_features)]

        except Exception as e:
            tprint_warning(f"Feature name generation failed: {e}")
            return [f"feature_{i+1}" for i in range(features.shape[1])]

    async def _save_feature_artifacts(self, result: FeaturePreparationResult, config: Any) -> None:
        """Save feature preparation artifacts."""
        try:
            if not self.artifact_manager:
                return

            tprint_info("Saving feature preparation artifacts")

            artifacts = {
                'features': result.features,
                'feature_names': result.feature_names,
                'feature_scores': result.feature_scores,
                'metadata': result.metadata,
                'performance_metrics': self.performance_metrics
            }

            step_name = f"feature_preparation_{get_current_datetime().strftime('%Y%m%d_%H%M%S')}"
            success = self.artifact_manager.save_artifacts(artifacts, step_name, result.metadata)

            if success:
                self.performance_metrics["artifact_saves"] += 1
                tprint_success(f"Feature artifacts saved successfully")
            else:
                tprint_warning(f"Failed to save feature artifacts")

        except Exception as e:
            tprint_warning(f"Artifact saving failed: {e}")

    async def _prepare_features_shared(
        self,
        market_data: pd.DataFrame,
        feature_config: FeatureConfig
    ):
        """Prepare features using shared utilities."""
        try:
            tprint("📊 Preparing features using shared utilities", "INFO")

            # Validate inputs
            if market_data is None or len(market_data) == 0:
                raise ValueError("Market data is None or empty in shared feature preparation")

            if feature_config is None:
                raise ValueError("Feature config is None in shared feature preparation")

            # Use shared feature preparation
            result = prepare_market_features(
                market_data=market_data,
                feature_config=feature_config,
                return_metadata=True
            )

            # Validate result
            if result is None:
                raise ValueError("Shared feature preparation returned None")

            # Handle both return types: FeaturePreparationResult or numpy array
            if hasattr(result, 'features_array'):
                # It's a FeaturePreparationResult from shared_utils
                features = result.features_array
                feature_names = list(result.features_df.columns) if hasattr(result, 'features_df') and result.features_df is not None else []
                feature_scores = {}
                dropped_features = []
                metadata = result.metadata if hasattr(result, 'metadata') else {}
                preparation_time = 0.0
            elif hasattr(result, 'features'):
                # It's a FeaturePreparationResult from feature_service
                features = result.features
                feature_names = result.feature_names
                feature_scores = result.feature_scores if hasattr(result, 'feature_scores') else {}
                dropped_features = result.dropped_features if hasattr(result, 'dropped_features') else []
                metadata = result.metadata if hasattr(result, 'metadata') else {}
                preparation_time = result.preparation_time if hasattr(result, 'preparation_time') else 0.0
            else:
                # It's a numpy array
                features = result
                feature_names = []
                feature_scores = {}
                dropped_features = []
                metadata = {}
                preparation_time = 0.0

            tprint(f"✅ Shared utilities prepared {features.shape[1]} features", "SUCCESS")

            # Create a proper FeaturePreparationResult object for consistency
            return FeaturePreparationResult(
                features=features,
                feature_names=feature_names,
                feature_scores=feature_scores,
                dropped_features=dropped_features,
                preparation_time=preparation_time,
                metadata={
                    **metadata,
                    "scaling_method": "robust",  # Default scaling method for shared utilities
                    "embedding_method": "none"   # No embedding applied in shared utilities
                }
            )

        except Exception as e:
            tprint(f"❌ Shared feature preparation failed: {e}", "ERROR")
            raise

    async def _apply_scaling(self, features: np.ndarray) -> Tuple[np.ndarray, float]:
        """Apply scaling to features with hardware optimization."""
        try:
            start_time = time.time()
            tprint("⚖️ Applying feature scaling", "INFO")

            # Apply memory optimization if hardware service is available
            if self.hardware_integration_enabled and self.hardware_service:
                try:
                    features, optimization_info = self.hardware_service.optimize_memory(features)
                    if optimization_info.get("hardware_optimization_used", False):
                        self.performance_metrics["memory_optimizations"] += 1
                        tprint(f"🧠 Memory optimization applied during scaling", "SUCCESS")
                except Exception as e:
                    tprint(f"⚠️ Memory optimization failed during scaling: {e}", "WARNING")

            # Import and initialize scaler
            from sklearn.preprocessing import RobustScaler

            # Use RobustScaler for financial data (handles outliers well)
            self.scaler = RobustScaler()

            # Fit and transform
            scaled_features = self.scaler.fit_transform(features)

            scaling_time = time.time() - start_time

            tprint(f"✅ Scaling completed in {scaling_time:.3f}s", "SUCCESS")
            return scaled_features, scaling_time

        except Exception as e:
            tprint(f"❌ Feature scaling failed: {e}", "ERROR")
            raise

    async def _apply_embedding(
        self,
        features: np.ndarray,
        feature_names: List[str],
        config: Any
    ) -> Tuple[np.ndarray, float]:
        """Apply dimensionality reduction (PCA/UMAP)."""
        try:
            start_time = time.time()
            tprint("🗺️ Applying dimensionality reduction", "INFO")

            # Check memory pressure before dimensionality reduction
            try:
                from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
                memory_optimizer = get_m1_memory_optimizer()
                memory_pressure = getattr(memory_optimizer, 'memory_pressure', 0.0)

                if memory_pressure > 0.85:  # Very high memory pressure threshold
                    tprint(f"🧠 Very high memory pressure detected ({memory_pressure:.2f}), skipping dimensionality reduction", "WARNING")
                    return features, 0.0
            except Exception as e:
                tprint(f"Could not check memory pressure: {e}, proceeding with dimensionality reduction", "WARNING")

            # Check if dimensionality reduction is needed
            n_features = features.shape[1]
            n_samples = features.shape[0]
            target_features = getattr(config, 'target_features', min(20, n_features - 1))

            # Log embedding configuration
            tprint(f"  📊 Input: {n_samples} samples × {n_features} features", "DEBUG")
            tprint(f"  🎯 Target: {target_features} components", "DEBUG")
            tprint(f"  📉 Max reduction: {((n_features - target_features) / n_features * 100):.1f}%", "DEBUG")

            if n_features <= target_features:
                tprint(f"📊 No reduction needed: {n_features} features", "INFO")
                return features, 0.0

            # Try UMAP first (better for non-linear relationships)
            tprint(f"🔍 Attempting UMAP reduction...", "DEBUG")
            umap_features = await self._try_umap_reduction(features, target_features)

            if umap_features is not None:
                embedding_time = time.time() - start_time
                tprint(f"✅ UMAP reduction: {n_features} → {umap_features.shape[1]} features", "SUCCESS")
                tprint(f"  📊 Dimensionality reduction: {((n_features - umap_features.shape[1]) / n_features * 100):.1f}%", "INFO")
                return umap_features, embedding_time

            # Fallback to PCA
            tprint(f"🔄 UMAP not available, using PCA fallback", "INFO")
            pca_features = await self._apply_pca_reduction(features, target_features, feature_names)

            embedding_time = time.time() - start_time
            tprint(f"✅ PCA reduction: {n_features} → {pca_features.shape[1]} features", "SUCCESS")
            tprint(f"  📊 Dimensionality reduction: {((n_features - pca_features.shape[1]) / n_features * 100):.1f}%", "INFO")
            return pca_features, embedding_time

        except Exception as e:
            tprint(f"❌ Dimensionality reduction failed: {e}", "ERROR")
            tprint("⚠️ Returning original features", "WARNING")
            return features, 0.0

    async def _try_umap_reduction(self, features: np.ndarray, target_features: int) -> Optional[np.ndarray]:
        """Try UMAP reduction as primary method with hardware acceleration."""
        try:
            import umap  # type: ignore

            if not hasattr(umap, 'UMAP'):
                return None

            # Apply hardware acceleration if available
            if self.hardware_integration_enabled and self.hardware_service:
                try:
                    # Try to use GPU acceleration for UMAP
                    neighbors_result, acceleration_info = self.hardware_service.accelerate_neighbors(
                        features, n_neighbors=min(15, features.shape[0] // 10)
                    )

                    if acceleration_info.get("hardware_acceleration_used", False):
                        self.performance_metrics["hardware_accelerations"] += 1
                        tprint(f"🏎️ Hardware acceleration used for UMAP neighbors computation", "SUCCESS")
                except Exception as e:
                    tprint(f"⚠️ Hardware acceleration failed for UMAP: {e}", "WARNING")

            # Initialize UMAP reducer
            self.umap_reducer = umap.UMAP(
                n_components=target_features,
                random_state=42,
                n_neighbors=min(15, features.shape[0] // 10),
                min_dist=0.1,
                metric='euclidean'
            )

            # Fit and transform
            reduced_features = self.umap_reducer.fit_transform(features)

            return reduced_features

        except ImportError:
            tprint("📦 UMAP not available, using PCA fallback", "INFO")
            return None
        except Exception as e:
            tprint(f"⚠️ UMAP reduction failed: {e}, using PCA fallback", "WARNING")
            return None

    async def _apply_pca_reduction(self, features: np.ndarray, target_features: int, feature_names: List[str] = None) -> np.ndarray:
        """Apply PCA reduction as fallback method."""
        try:
            from sklearn.decomposition import PCA

            # Check memory pressure before PCA fitting
            try:
                from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
                memory_optimizer = get_m1_memory_optimizer()
                memory_pressure = getattr(memory_optimizer, 'memory_pressure', 0.0)

                if memory_pressure > 0.8:  # High memory pressure threshold
                    tprint(f"🧠 High memory pressure detected ({memory_pressure:.2f}), using simplified PCA", "WARNING")
                    # Use fewer components to reduce memory usage
                    target_features = min(target_features, 5)
                    tprint(f"📉 Reduced target components to {target_features} due to memory pressure", "INFO")
            except Exception as e:
                tprint(f"Could not check memory pressure: {e}, proceeding with normal PCA", "WARNING")

            # Log PCA initialization details
            n_samples, n_features = features.shape
            tprint(f"🔧 Initializing PCA reduction", "INFO")
            tprint(f"  📊 Input features: {n_features} dimensions, {n_samples} samples", "DEBUG")
            tprint(f"  🎯 Target components: {target_features}", "DEBUG")
            tprint(f"  🔄 Random state: 42 (for reproducibility)", "DEBUG")

            # Initialize PCA
            self.pca = PCA(n_components=target_features, random_state=42)

            # Log PCA fitting process
            tprint(f"🔍 Fitting PCA to data...", "INFO")

            # Fit and transform
            reduced_features = self.pca.fit_transform(features)

            tprint(f"  ✅ PCA fitting completed", "DEBUG")

            # Log PCA results and analysis
            explained_variance_ratio = self.pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            total_variance_explained = cumulative_variance[-1]

            tprint(f"📈 PCA Reduction Results:", "INFO")
            tprint(f"  📊 Original features: {n_features} → Reduced features: {reduced_features.shape[1]}", "INFO")
            tprint(f"  📉 Total variance explained: {total_variance_explained:.4f} ({total_variance_explained*100:.2f}%)", "INFO")
            tprint(f"  📊 Feature reduction: {((n_features - reduced_features.shape[1]) / n_features * 100):.1f}%", "INFO")

            # Log component-wise variance explained with feature contributions
            tprint(f"🔍 Component Analysis:", "DEBUG")
            for i, (var_ratio, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
                feature_name = f"PC{i+1}_var{var_ratio:.3f}"
                tprint(f"  {feature_name}: {var_ratio:.4f} ({var_ratio*100:.2f}%) | Cumulative: {cum_var:.4f} ({cum_var*100:.2f}%)", "DEBUG")

                # Analyze which original features contribute most to this component
                if hasattr(self.pca, 'components_') and feature_names and len(feature_names) > 0:
                    component_loadings = self.pca.components_[i]
                    # Get top contributing features (absolute values)
                    top_features_idx = np.argsort(np.abs(component_loadings))[-5:][::-1]  # Top 5

                    tprint(f"    🎯 Top contributing features:", "DEBUG")
                    for j, feat_idx in enumerate(top_features_idx):
                        if feat_idx < len(feature_names):
                            feat_name = feature_names[feat_idx]
                            loading = component_loadings[feat_idx]
                            # Categorize feature type
                            feat_type = self._categorize_feature(feat_name)
                            tprint(f"      {j+1}. {feat_name} ({feat_type}): {loading:.4f}", "DEBUG")

            # Log top components that explain most variance
            top_components = np.argsort(explained_variance_ratio)[::-1][:5]
            tprint(f"🏆 Top 5 Components by Variance:", "DEBUG")
            for i, comp_idx in enumerate(top_components):
                feature_name = f"PC{comp_idx+1}_var{explained_variance_ratio[comp_idx]:.3f}"
                tprint(f"  {i+1}. {feature_name}: {explained_variance_ratio[comp_idx]:.4f} ({explained_variance_ratio[comp_idx]*100:.2f}%)", "DEBUG")

            # Analyze feature type composition for top components
            tprint(f"📊 PCA Component Feature Analysis:", "INFO")
            for i in range(min(3, reduced_features.shape[1])):  # Analyze top 3 components
                if hasattr(self.pca, 'components_') and feature_names and len(feature_names) > 0:
                    component_loadings = self.pca.components_[i]
                    # Get all contributing features (not just top 5)
                    feature_contributions = []
                    for j, loading in enumerate(component_loadings):
                        if j < len(feature_names):
                            feat_name = feature_names[j]
                            feat_type = self._categorize_feature(feat_name)
                            feature_contributions.append((feat_name, feat_type, abs(loading)))

                    # Sort by contribution strength
                    feature_contributions.sort(key=lambda x: x[2], reverse=True)

                    # Count feature types
                    type_counts = {}
                    for _, feat_type, _ in feature_contributions:
                        type_counts[feat_type] = type_counts.get(feat_type, 0) + 1

                    # Show composition
                    component_name = f"PC{i+1}_var{explained_variance_ratio[i]:.3f}"
                    tprint(f"  🎯 {component_name} composition:", "INFO")
                    for feat_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / len(feature_contributions)) * 100
                        tprint(f"    {feat_type}: {count} features ({percentage:.1f}%)", "INFO")

            # Log data quality metrics
            reduced_mean = np.mean(reduced_features, axis=0)
            reduced_std = np.std(reduced_features, axis=0)
            tprint(f"📊 Reduced Feature Statistics:", "DEBUG")
            tprint(f"  Mean range: [{np.min(reduced_mean):.4f}, {np.max(reduced_mean):.4f}]", "DEBUG")
            tprint(f"  Std range: [{np.min(reduced_std):.4f}, {np.max(reduced_std):.4f}]", "DEBUG")

            # Check for potential issues
            if total_variance_explained < 0.8:
                tprint(f"⚠️ Low variance explained ({total_variance_explained:.2f}%) - consider more components", "WARNING")

            if np.any(np.isnan(reduced_features)):
                tprint(f"❌ NaN values detected in reduced features!", "ERROR")

            if np.any(np.isinf(reduced_features)):
                tprint(f"❌ Infinite values detected in reduced features!", "ERROR")

            return reduced_features

        except Exception as e:
            tprint(f"❌ PCA reduction failed: {e}", "ERROR")
            raise

    def _validate_features(self, features: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate prepared features."""
        try:
            tprint("🔍 Validating prepared features", "INFO")

            validation_results = {
                "valid": True,
                "issues": [],
                "warnings": []
            }

            # Check basic properties with validation
            if not validate_finite(features.shape[0], "feature_count"):
                validation_results["valid"] = False
                validation_results["issues"].append("Invalid feature count")
            elif features.shape[0] == 0:
                validation_results["valid"] = False
                validation_results["issues"].append("No samples in features")

            if not validate_finite(features.shape[1], "feature_dimensions"):
                validation_results["valid"] = False
                validation_results["issues"].append("Invalid feature dimensions")
            elif features.shape[1] == 0:
                validation_results["valid"] = False
                validation_results["issues"].append("No features available")

            if features.shape[0] < 10:
                validation_results["warnings"].append("Very few samples for clustering")

            if features.shape[1] < 2:
                validation_results["valid"] = False
                validation_results["issues"].append("Insufficient features for clustering")

            # Check for NaN values with safe operations
            nan_count = int(np.sum(np.isnan(features)))
            if nan_count > 0:
                validation_results["warnings"].append(f"Features contain {nan_count} NaN values")

            # Check for infinite values with safe operations
            inf_count = int(np.sum(np.isinf(features)))
            if inf_count > 0:
                validation_results["warnings"].append(f"Features contain {inf_count} infinite values")

            # Check feature variance (avoid constant features) with safe math
            try:
                feature_variances = np.var(features, axis=0)
                constant_features = int(np.sum(feature_variances < 1e-8))
                if constant_features > 0:
                    validation_results["warnings"].append(f"{constant_features} constant features detected")
            except Exception as e:
                validation_results["warnings"].append(f"Could not calculate feature variances: {e}")

            tprint(f"✅ Feature validation completed: {len(validation_results['issues'])} issues, {len(validation_results['warnings'])} warnings", "SUCCESS")
            return validation_results

        except Exception as e:
            tprint(f"❌ Feature validation failed: {e}", "ERROR")
            return {"valid": False, "issues": [f"Validation error: {e}"], "warnings": []}

    def _categorize_feature(self, feature_name: str) -> str:
        """Categorize a feature by its name to identify type (volatility, momentum, trend, etc.)."""
        feature_name_lower = feature_name.lower()

        # Volatility indicators
        if any(term in feature_name_lower for term in ['vol', 'volatility', 'atr', 'std', 'dev', 'range', 'bb', 'bollinger']):
            return "VOLATILITY"

        # Momentum indicators
        elif any(term in feature_name_lower for term in ['rsi', 'momentum', 'roc', 'rate_of_change', 'stoch', 'stochastic', 'williams', 'cci']):
            return "MOMENTUM"

        # Trend indicators
        elif any(term in feature_name_lower for term in ['ma', 'moving_average', 'ema', 'sma', 'trend', 'macd', 'adx', 'dmi', 'aroon']):
            return "TREND"

        # Volume indicators
        elif any(term in feature_name_lower for term in ['volume', 'vol', 'obv', 'ad', 'accumulation', 'distribution', 'mfi', 'money_flow']):
            return "VOLUME"

        # Price-based features
        elif any(term in feature_name_lower for term in ['price', 'close', 'open', 'high', 'low', 'return', 'change', 'pct']):
            return "PRICE"

        # Statistical features
        elif any(term in feature_name_lower for term in ['skew', 'kurt', 'stat', 'corr', 'correlation', 'beta', 'alpha']):
            return "STATISTICAL"

        # Regime features
        elif any(term in feature_name_lower for term in ['regime', 'state', 'phase', 'cycle']):
            return "REGIME"

        # Technical patterns
        elif any(term in feature_name_lower for term in ['pattern', 'signal', 'crossover', 'breakout', 'support', 'resistance']):
            return "PATTERN"

        # Default category
        else:
            return "OTHER"

    def _get_embedding_method(self) -> str:
        """Get the current embedding method name."""
        if self.umap_reducer is not None:
            return "UMAP"
        elif self.pca is not None:
            return "PCA"
        else:
            return "None"

    def _track_feature_preparation(self, result: FeaturePreparationResult):
        """Track feature preparation for analysis."""
        try:
            self.feature_history.append({
                "timestamp": time.time(),
                "original_features": result.metadata["original_feature_count"],
                "final_features": result.metadata["final_feature_count"],
                "preparation_time": result.preparation_time,
                "scaling_method": result.metadata["scaling_method"],
                "embedding_method": result.metadata["embedding_method"],
                "validation_issues": len(result.metadata["validation_results"]["issues"]),
                "validation_warnings": len(result.metadata["validation_results"]["warnings"])
            })

            # Keep only last 10 entries
            if len(self.feature_history) > 10:
                self.feature_history = self.feature_history[-10:]

        except Exception as e:
            tprint(f"⚠️ Feature tracking failed: {e}", "WARNING")

    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get feature preparation statistics."""
        if not self.feature_history:
            return {"message": "No feature preparation history available"}

        # Calculate statistics across all preparations
        prep_times = [h["preparation_time"] for h in self.feature_history]
        feature_counts = [h["final_features"] for h in self.feature_history]

        return {
            "total_preparations": len(self.feature_history),
            "average_preparation_time": np.mean(prep_times),
            "min_preparation_time": np.min(prep_times),
            "max_preparation_time": np.max(prep_times),
            "average_feature_count": np.mean(feature_counts),
            "min_feature_count": np.min(feature_counts),
            "max_feature_count": np.max(feature_counts),
            "performance_metrics": self.performance_metrics,
            "recent_history": self.feature_history[-3:]  # Last 3 preparations
        }

    def clear_feature_cache(self):
        """Clear feature preparation cache and reset state."""
        try:
            self.scaler = None
            self.pca = None
            self.umap_reducer = None
            self.feature_history.clear()

            tprint("🧹 Feature cache cleared", "INFO")

        except Exception as e:
            tprint(f"⚠️ Cache clearing failed: {e}", "WARNING")

    async def prepare_features_for_clustering(
        self,
        market_data: pd.DataFrame,
        clustering_config: Any = None
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Prepare features specifically for clustering.

        Args:
            market_data: Market data for feature extraction
            clustering_config: Clustering-specific configuration

        Returns:
            Tuple of (features, feature_names, metadata)
        """
        try:
            # Use clustering-specific configuration if provided
            if clustering_config:
                config = clustering_config
            else:
                # Create default clustering configuration
                config = type('Config', (), {
                    'feature_categories': ['regime_volatility', 'regime_volume', 'regime_structural_trend', 'regime_statistical'],
                    'use_standardized_features': True,
                    'drop_highly_correlated': True,
                    'correlation_threshold': 0.95,
                    'target_features': 20
                })()

            # Prepare features
            result = await self.prepare_features(market_data, config)

            return (
                result.features,
                result.feature_names,
                result.metadata
            )

        except Exception as e:
            tprint(f"❌ Clustering feature preparation failed: {e}", "ERROR")
            raise
