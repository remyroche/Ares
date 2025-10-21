"""
Shared utilities for clustering components.

This module provides essential utilities that were previously in the deleted shared_utils folder.
Enhanced with comprehensive utility integrations for improved functionality and performance.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import time
from datetime import datetime

# Import comprehensive utility modules
from src.utils.common_operations import (
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

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_structured,
    tprint_timer, tprint_logged, LogLevel, TimestampFormat
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


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


def get_scaler(scaler_type: str):
    """Get the appropriate scaler based on type."""
    if scaler_type == 'robust':
        from sklearn.preprocessing import RobustScaler
        return RobustScaler()
    elif scaler_type == 'standard':
        from sklearn.preprocessing import StandardScaler
        return StandardScaler()
    elif scaler_type == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        return MinMaxScaler()
    elif scaler_type == 'normalizer':
        from sklearn.preprocessing import Normalizer
        return Normalizer()
    else:
        from sklearn.preprocessing import RobustScaler
        return RobustScaler()


def handle_outliers_iqr(data: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
    """Handle outliers using IQR method."""
    try:
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Clip outliers instead of removing them
        return data.clip(lower=lower_bound, upper=upper_bound, axis=1)
    except Exception as e:
        tprint_warning(f"Failed to handle outliers with IQR: {e}")
        return data


def handle_outliers_zscore(data: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Handle outliers using Z-score method."""
    try:
        from scipy import stats
        z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))
        data_clean = data.copy()
        data_clean[z_scores > threshold] = np.nan
        return data_clean.fillna(data.median())
    except Exception as e:
        tprint_warning(f"Failed to handle outliers with Z-score: {e}")
        return data


def apply_variance_threshold(features: np.ndarray, feature_names: List[str], threshold: float) -> Tuple[np.ndarray, List[str]]:
    """Apply variance threshold for feature selection."""
    try:
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=threshold)
        features_selected = selector.fit_transform(features)
        selected_indices = selector.get_support(indices=True)
        selected_names = [feature_names[i] for i in selected_indices]
        return features_selected, selected_names
    except Exception as e:
        tprint_warning(f"Failed to apply variance threshold: {e}")
        return features, feature_names


def remove_correlated_features(features: np.ndarray, feature_names: List[str], threshold: float) -> Tuple[np.ndarray, List[str]]:
    """Remove highly correlated features."""
    try:
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(features.T)
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                if abs(corr_matrix[i, j]) > threshold:
                    high_corr_pairs.append((i, j))
        
        # Remove one feature from each highly correlated pair
        features_to_remove = set()
        for i, j in high_corr_pairs:
            # Keep the feature with higher variance
            var_i = np.var(features[:, i])
            var_j = np.var(features[:, j])
            if var_i < var_j:
                features_to_remove.add(i)
            else:
                features_to_remove.add(j)
        
        # Create mask for features to keep
        keep_mask = [i not in features_to_remove for i in range(features.shape[1])]
        
        return features[:, keep_mask], [name for i, name in enumerate(feature_names) if keep_mask[i]]
    except Exception as e:
        tprint_warning(f"Failed to remove correlated features: {e}")
        return features, feature_names


def calculate_feature_importance(features: np.ndarray, feature_names: List[str], method: str) -> Dict[str, float]:
    """Calculate feature importance using specified method."""
    try:
        if method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression
            # Create dummy target for mutual information
            target = np.random.randn(features.shape[0])
            importance_scores = mutual_info_regression(features, target)
        elif method == 'variance':
            importance_scores = np.var(features, axis=0)
        elif method == 'random':
            importance_scores = np.random.rand(features.shape[1])
        else:
            importance_scores = np.ones(features.shape[1])
        
        # Normalize scores
        importance_scores = importance_scores / np.sum(importance_scores)
        
        return {name: score for name, score in zip(feature_names, importance_scores)}
    except Exception as e:
        tprint_warning(f"Failed to calculate feature importance: {e}")
        return {name: 1.0 for name in feature_names}


@dataclass
class FeatureConfig:
    """Enhanced configuration for feature preparation with comprehensive options."""
    n_features: int = 50
    use_pca: bool = True
    pca_components: int = 20
    use_umap: bool = False
    umap_components: int = 10
    scaler_type: str = 'robust'
    # Enhanced configuration options
    enable_hardware_optimization: bool = True
    enable_ml_optimization: bool = True
    enable_data_validation: bool = True
    enable_feature_engineering: bool = True
    enable_anomaly_detection: bool = True
    enable_outlier_detection: bool = True
    enable_data_quality_checks: bool = True
    enable_memory_optimization: bool = True
    enable_caching: bool = True
    # Feature engineering options
    feature_selection_method: str = 'variance_threshold'
    feature_selection_threshold: float = 0.01
    feature_correlation_threshold: float = 0.95
    feature_importance_method: str = 'mutual_info'
    # Data preprocessing options
    handle_missing_values: str = 'median'  # 'median', 'mean', 'drop', 'interpolate'
    handle_outliers: str = 'iqr'  # 'iqr', 'zscore', 'isolation_forest', 'none'
    outlier_threshold: float = 1.5
    # Validation options
    validate_data_consistency: bool = True
    check_data_leakage: bool = True
    check_lookahead_bias: bool = True
    # Performance options
    use_parallel_processing: bool = True
    max_workers: int = 4
    chunk_size: int = 1000


@dataclass
class FeaturePreparationResult:
    """Enhanced result from feature preparation with comprehensive metadata."""
    features: np.ndarray
    feature_names: List[str]
    scaler: Any
    pca: Optional[Any] = None
    umap: Optional[Any] = None
    feature_scores: Dict[str, float] = None
    # Enhanced result metadata
    original_features: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    data_quality_metrics: Optional[Dict[str, Any]] = None
    preprocessing_steps: Optional[List[str]] = None
    validation_results: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    hardware_optimization_info: Optional[Dict[str, Any]] = None
    memory_usage: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    warnings: Optional[List[str]] = None
    errors: Optional[List[str]] = None


@performance_tracked(log_performance=True, track_memory=True) if HARDWARE_OPTIMIZATION_AVAILABLE else lambda x: x
@memory_optimized(level=MemoryOptimizationLevel.BALANCED) if HARDWARE_OPTIMIZATION_AVAILABLE else lambda x: x
def prepare_market_features(
    market_data: pd.DataFrame,
    config: FeatureConfig
) -> FeaturePreparationResult:
    """
    Enhanced market feature preparation with comprehensive utility integrations.
    
    Args:
        market_data: Market data DataFrame
        config: Enhanced feature configuration
        
    Returns:
        FeaturePreparationResult with prepared features and comprehensive metadata
    """
    logger = get_logger('prepare_market_features')
    start_time = time.time()
    warnings = []
    errors = []
    preprocessing_steps = []
    
    try:
        tprint_info("Starting enhanced market feature preparation")
        
        # Initialize hardware manager if available
        hardware_manager = None
        if config.enable_hardware_optimization and HARDWARE_OPTIMIZATION_AVAILABLE:
            try:
                hardware_manager = get_integrated_hardware_manager()
                tprint_info("Hardware optimization enabled for feature preparation")
            except Exception as e:
                warnings.append(f"Failed to initialize hardware manager: {e}")
        
        # Step 1: Data validation and quality checks
        if config.enable_data_validation:
            tprint_info("Performing data validation and quality checks")
            preprocessing_steps.append("data_validation")
            
            # Validate input data
            if not validate_dataframe_columns(market_data, []):
                warnings.append("DataFrame validation failed")
            
            # Analyze data quality
            data_quality_metrics = calculate_data_quality_metrics(market_data)
            tprint_structured("Data Quality Metrics", data_quality_metrics)
            
            # Check for data anomalies
            if config.enable_anomaly_detection:
                anomalies = detect_data_anomalies(market_data)
                if anomalies:
                    warnings.append(f"Data anomalies detected: {len(anomalies)}")
        
        # Step 2: Data preprocessing
        tprint_info("Performing data preprocessing")
        preprocessing_steps.append("data_preprocessing")
        
        # Handle missing values
        if config.handle_missing_values != 'none':
            original_data = market_data.copy()
            if config.handle_missing_values == 'median':
                market_data = safe_fillna(market_data, method='median')
            elif config.handle_missing_values == 'mean':
                market_data = safe_fillna(market_data, method='mean')
            elif config.handle_missing_values == 'interpolate':
                market_data = market_data.interpolate()
            elif config.handle_missing_values == 'drop':
                market_data = safe_dropna(market_data)
            preprocessing_steps.append(f"missing_values_{config.handle_missing_values}")
        
        # Handle outliers
        if config.handle_outliers != 'none':
            if config.handle_outliers == 'iqr':
                market_data = handle_outliers_iqr(market_data, threshold=config.outlier_threshold)
            elif config.handle_outliers == 'zscore':
                market_data = handle_outliers_zscore(market_data, threshold=config.outlier_threshold)
            preprocessing_steps.append(f"outliers_{config.handle_outliers}")
        
        # Step 3: Feature extraction
        tprint_info("Extracting numerical features")
        preprocessing_steps.append("feature_extraction")
        
        # Extract numerical features with validation
        features = market_data.select_dtypes(include=[np.number]).values
        math_validate_numeric_array(features, "market_features")
        
        # Store original features
        original_features = features.copy()
        
        # Generate feature names
        feature_names = [f"feature_{i}" for i in range(features.shape[1])]
        
        # Step 4: Feature engineering
        if config.enable_feature_engineering:
            tprint_info("Performing feature engineering")
            preprocessing_steps.append("feature_engineering")
            
            # Feature selection
            if config.feature_selection_method == 'variance_threshold':
                features, feature_names = apply_variance_threshold(features, feature_names, config.feature_selection_threshold)
            elif config.feature_selection_method == 'correlation_threshold':
                features, feature_names = remove_correlated_features(features, feature_names, config.feature_correlation_threshold)
        
        # Step 5: Scaling
        tprint_info(f"Applying {config.scaler_type} scaling")
        preprocessing_steps.append(f"scaling_{config.scaler_type}")
        
        scaler = get_scaler(config.scaler_type)
        features_scaled = scaler.fit_transform(features)
        math_validate_numeric_array(features_scaled, "scaled_features")
        
        # Step 6: Dimensionality reduction
        pca = None
        umap = None
        
        if config.use_pca and config.pca_components < features_scaled.shape[1]:
            tprint_info(f"Applying PCA with {config.pca_components} components")
            preprocessing_steps.append("pca")
            
            from sklearn.decomposition import PCA
            pca = PCA(n_components=config.pca_components)
            features_scaled = pca.fit_transform(features_scaled)
            feature_names = [f"pca_{i}" for i in range(config.pca_components)]
        
        if config.use_umap:
            tprint_info(f"Applying UMAP with {config.umap_components} components")
            preprocessing_steps.append("umap")
            
            try:
                import umap
                umap = umap.UMAP(n_components=config.umap_components)
                features_scaled = umap.fit_transform(features_scaled)
                feature_names = [f"umap_{i}" for i in range(config.umap_components)]
            except ImportError:
                warnings.append("UMAP not available, skipping UMAP transformation")
        
        # Step 7: Feature importance calculation
        feature_importance = None
        if config.enable_ml_optimization and ML_COMMON_AVAILABLE:
            try:
                feature_importance = calculate_feature_importance(features_scaled, feature_names, config.feature_importance_method)
            except Exception as e:
                warnings.append(f"Failed to calculate feature importance: {e}")
        
        # Step 8: Final validation
        if config.enable_data_validation:
            tprint_info("Performing final validation")
            preprocessing_steps.append("final_validation")
            
            # Check for data leakage
            if config.check_data_leakage and DataLeakageDetector:
                try:
                    leakage_detector = DataLeakageDetector()
                    leakage_score = leakage_detector.detect_leakage(features_scaled)
                    if leakage_score > 0.1:
                        warnings.append(f"Potential data leakage detected: {leakage_score:.3f}")
                except Exception as e:
                    warnings.append(f"Data leakage detection failed: {e}")
            
            # Check for lookahead bias
            if config.check_lookahead_bias and LookaheadBiasDetector:
                try:
                    bias_detector = LookaheadBiasDetector()
                    bias_score = bias_detector.detect_bias(features_scaled)
                    if bias_score > 0.05:
                        warnings.append(f"Potential lookahead bias detected: {bias_score:.3f}")
                except Exception as e:
                    warnings.append(f"Lookahead bias detection failed: {e}")
        
        # Step 9: Performance metrics
        processing_time = time.time() - start_time
        memory_usage = get_memory_usage() if HARDWARE_OPTIMIZATION_AVAILABLE else {}
        
        performance_metrics = {
            'processing_time': processing_time,
            'memory_usage': memory_usage,
            'n_features_original': original_features.shape[1] if original_features is not None else 0,
            'n_features_final': features_scaled.shape[1],
            'n_samples': features_scaled.shape[0]
        }
        
        # Step 10: Hardware optimization info
        hardware_optimization_info = None
        if hardware_manager:
            try:
                hardware_optimization_info = hardware_manager.get_optimization_info()
            except Exception as e:
                warnings.append(f"Failed to get hardware optimization info: {e}")
        
        # Calculate feature scores
        feature_scores = {name: 1.0 for name in feature_names}
        if feature_importance:
            for name, importance in feature_importance.items():
                if name in feature_scores:
                    feature_scores[name] = importance
        
        tprint_success(f"Feature preparation completed in {processing_time:.2f} seconds")
        
        return FeaturePreparationResult(
            features=features_scaled,
            feature_names=feature_names,
            scaler=scaler,
            pca=pca,
            umap=umap,
            feature_scores=feature_scores,
            original_features=original_features,
            feature_importance=feature_importance,
            data_quality_metrics=data_quality_metrics if config.enable_data_validation else None,
            preprocessing_steps=preprocessing_steps,
            validation_results={'warnings': warnings, 'errors': errors},
            performance_metrics=performance_metrics,
            hardware_optimization_info=hardware_optimization_info,
            memory_usage=memory_usage,
            processing_time=processing_time,
            warnings=warnings,
            errors=errors
        )
        
    except Exception as e:
        error_msg = f"Feature preparation failed: {e}"
        tprint_error(error_msg)
        logger.error(error_msg)
        errors.append(error_msg)
        raise


@performance_tracked(log_performance=True, track_memory=True) if HARDWARE_OPTIMIZATION_AVAILABLE else lambda x: x
def calculate_consensus_metrics(
    cluster_assignments: np.ndarray,
    market_data: pd.DataFrame
) -> Dict[str, float]:
    """Calculate enhanced consensus metrics for clustering results."""
    try:
        tprint_info("Calculating consensus metrics")
        
        # Validate inputs
        math_validate_numeric_array(cluster_assignments, "cluster_assignments")
        validate_dataframe_columns(market_data, [])
        
        n_clusters = len(np.unique(cluster_assignments))
        n_samples = len(cluster_assignments)
        
        # Calculate consensus score based on cluster stability
        consensus_score = calculate_cluster_consensus_score(cluster_assignments, market_data)
        
        # Calculate stability score based on cluster consistency
        stability_score = calculate_cluster_stability_score(cluster_assignments, market_data)
        
        # Calculate additional consensus metrics
        cluster_balance = calculate_cluster_balance(cluster_assignments)
        cluster_separation = calculate_cluster_separation(cluster_assignments, market_data)
        
        metrics = {
            'n_clusters': n_clusters,
            'n_samples': n_samples,
            'consensus_score': consensus_score,
            'stability_score': stability_score,
            'cluster_balance': cluster_balance,
            'cluster_separation': cluster_separation,
            'overall_consensus': (consensus_score + stability_score + cluster_balance + cluster_separation) / 4
        }
        
        tprint_success(f"Consensus metrics calculated: {metrics}")
        return metrics
        
    except Exception as e:
        tprint_error(f"Failed to calculate consensus metrics: {e}")
        return {'n_clusters': 0, 'n_samples': 0, 'consensus_score': 0.0, 'stability_score': 0.0}


def calculate_cluster_consensus_score(cluster_assignments: np.ndarray, market_data: pd.DataFrame) -> float:
    """Calculate cluster consensus score."""
    try:
        # Calculate intra-cluster similarity
        n_clusters = len(np.unique(cluster_assignments))
        if n_clusters <= 1:
            return 0.0
        
        total_similarity = 0.0
        for cluster_id in np.unique(cluster_assignments):
            cluster_mask = cluster_assignments == cluster_id
            cluster_data = market_data[cluster_mask]
            
            if len(cluster_data) > 1:
                # Calculate average pairwise correlation within cluster
                numeric_data = cluster_data.select_dtypes(include=[np.number])
                if len(numeric_data.columns) > 1:
                    corr_matrix = numeric_data.corr()
                    # Get upper triangle of correlation matrix
                    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    avg_correlation = upper_tri.stack().mean()
                    total_similarity += avg_correlation if not np.isnan(avg_correlation) else 0.0
        
        return total_similarity / n_clusters
    except Exception:
        return 0.0


def calculate_cluster_stability_score(cluster_assignments: np.ndarray, market_data: pd.DataFrame) -> float:
    """Calculate cluster stability score."""
    try:
        # Calculate cluster size consistency
        cluster_sizes = [np.sum(cluster_assignments == i) for i in np.unique(cluster_assignments)]
        if len(cluster_sizes) <= 1:
            return 0.0
        
        # Calculate coefficient of variation (lower is more stable)
        mean_size = np.mean(cluster_sizes)
        std_size = np.std(cluster_sizes)
        cv = std_size / mean_size if mean_size > 0 else 1.0
        
        # Convert to stability score (1 - normalized CV)
        stability_score = max(0.0, 1.0 - cv)
        return stability_score
    except Exception:
        return 0.0


def calculate_cluster_balance(cluster_assignments: np.ndarray) -> float:
    """Calculate cluster balance score."""
    try:
        n_clusters = len(np.unique(cluster_assignments))
        if n_clusters <= 1:
            return 0.0
        
        cluster_sizes = [np.sum(cluster_assignments == i) for i in np.unique(cluster_assignments)]
        total_samples = len(cluster_assignments)
        expected_size = total_samples / n_clusters
        
        # Calculate balance as 1 - normalized standard deviation
        size_deviations = [abs(size - expected_size) for size in cluster_sizes]
        max_deviation = total_samples * 0.5  # Maximum possible deviation
        balance_score = 1.0 - (np.mean(size_deviations) / max_deviation)
        
        return max(0.0, min(1.0, balance_score))
    except Exception:
        return 0.0


def calculate_cluster_separation(cluster_assignments: np.ndarray, market_data: pd.DataFrame) -> float:
    """Calculate cluster separation score."""
    try:
        n_clusters = len(np.unique(cluster_assignments))
        if n_clusters <= 1:
            return 0.0
        
        # Calculate inter-cluster distances
        numeric_data = market_data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) == 0:
            return 0.0
        
        cluster_centers = []
        for cluster_id in np.unique(cluster_assignments):
            cluster_mask = cluster_assignments == cluster_id
            cluster_data = numeric_data[cluster_mask]
            if len(cluster_data) > 0:
                cluster_centers.append(cluster_data.mean().values)
        
        if len(cluster_centers) <= 1:
            return 0.0
        
        # Calculate average pairwise distance between cluster centers
        total_distance = 0.0
        pair_count = 0
        
        for i in range(len(cluster_centers)):
            for j in range(i + 1, len(cluster_centers)):
                distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                total_distance += distance
                pair_count += 1
        
        if pair_count == 0:
            return 0.0
        
        avg_distance = total_distance / pair_count
        
        # Normalize separation score (higher distance = better separation)
        # Use a simple normalization based on data variance
        data_variance = np.var(numeric_data.values)
        normalized_separation = min(1.0, avg_distance / np.sqrt(data_variance))
        
        return normalized_separation
    except Exception:
        return 0.0


@performance_tracked(log_performance=True, track_memory=True) if HARDWARE_OPTIMIZATION_AVAILABLE else lambda x: x
def calculate_disagreement_metrics(
    cluster_assignments: np.ndarray,
    market_data: pd.DataFrame
) -> Dict[str, float]:
    """Calculate enhanced disagreement metrics for clustering results."""
    try:
        tprint_info("Calculating disagreement metrics")
        
        # Validate inputs
        math_validate_numeric_array(cluster_assignments, "cluster_assignments")
        
        # Calculate disagreement score based on cluster boundary uncertainty
        disagreement_score = calculate_cluster_disagreement_score(cluster_assignments, market_data)
        
        # Calculate uncertainty score based on cluster assignment confidence
        uncertainty_score = calculate_cluster_uncertainty_score(cluster_assignments, market_data)
        
        # Calculate boundary instability
        boundary_instability = calculate_boundary_instability(cluster_assignments, market_data)
        
        metrics = {
            'disagreement_score': disagreement_score,
            'uncertainty_score': uncertainty_score,
            'boundary_instability': boundary_instability,
            'overall_disagreement': (disagreement_score + uncertainty_score + boundary_instability) / 3
        }
        
        tprint_success(f"Disagreement metrics calculated: {metrics}")
        return metrics
        
    except Exception as e:
        tprint_error(f"Failed to calculate disagreement metrics: {e}")
        return {'disagreement_score': 0.0, 'uncertainty_score': 0.0, 'boundary_instability': 0.0}


def calculate_cluster_disagreement_score(cluster_assignments: np.ndarray, market_data: pd.DataFrame) -> float:
    """Calculate cluster disagreement score based on boundary uncertainty."""
    try:
        n_clusters = len(np.unique(cluster_assignments))
        if n_clusters <= 1:
            return 0.0
        
        # Calculate pairwise cluster distances
        numeric_data = market_data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) == 0:
            return 0.0
        
        cluster_centers = []
        for cluster_id in np.unique(cluster_assignments):
            cluster_mask = cluster_assignments == cluster_id
            cluster_data = numeric_data[cluster_mask]
            if len(cluster_data) > 0:
                cluster_centers.append(cluster_data.mean().values)
        
        if len(cluster_centers) <= 1:
            return 0.0
        
        # Calculate minimum distance between clusters
        min_distance = float('inf')
        for i in range(len(cluster_centers)):
            for j in range(i + 1, len(cluster_centers)):
                distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                min_distance = min(min_distance, distance)
        
        # Calculate average cluster spread
        total_spread = 0.0
        for cluster_id in np.unique(cluster_assignments):
            cluster_mask = cluster_assignments == cluster_id
            cluster_data = numeric_data[cluster_mask]
            if len(cluster_data) > 1:
                cluster_center = cluster_data.mean().values
                distances = [np.linalg.norm(point - cluster_center) for point in cluster_data.values]
                total_spread += np.mean(distances)
        
        avg_spread = total_spread / n_clusters
        
        # Disagreement is higher when clusters are close relative to their spread
        disagreement = avg_spread / min_distance if min_distance > 0 else 1.0
        return min(1.0, disagreement)
    except Exception:
        return 0.0


def calculate_cluster_uncertainty_score(cluster_assignments: np.ndarray, market_data: pd.DataFrame) -> float:
    """Calculate cluster uncertainty score based on assignment confidence."""
    try:
        # Calculate uncertainty based on cluster size distribution
        cluster_sizes = [np.sum(cluster_assignments == i) for i in np.unique(cluster_assignments)]
        total_samples = len(cluster_assignments)
        
        # Calculate entropy of cluster distribution
        probabilities = [size / total_samples for size in cluster_sizes if size > 0]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
        # Normalize entropy (max entropy is log2(n_clusters))
        max_entropy = np.log2(len(cluster_sizes))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy
    except Exception:
        return 0.0


def calculate_boundary_instability(cluster_assignments: np.ndarray, market_data: pd.DataFrame) -> float:
    """Calculate boundary instability score."""
    try:
        # This is a simplified version - in practice, you'd need multiple clustering runs
        # For now, we'll calculate based on cluster density variations
        n_clusters = len(np.unique(cluster_assignments))
        if n_clusters <= 1:
            return 0.0
        
        numeric_data = market_data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) == 0:
            return 0.0
        
        cluster_densities = []
        for cluster_id in np.unique(cluster_assignments):
            cluster_mask = cluster_assignments == cluster_id
            cluster_data = numeric_data[cluster_mask]
            if len(cluster_data) > 1:
                # Calculate average distance to cluster center
                cluster_center = cluster_data.mean().values
                distances = [np.linalg.norm(point - cluster_center) for point in cluster_data.values]
                avg_distance = np.mean(distances)
                cluster_densities.append(1.0 / (1.0 + avg_distance))  # Higher density = lower distance
        
        if len(cluster_densities) <= 1:
            return 0.0
        
        # Calculate coefficient of variation of densities
        mean_density = np.mean(cluster_densities)
        std_density = np.std(cluster_densities)
        cv = std_density / mean_density if mean_density > 0 else 1.0
        
        return min(1.0, cv)
    except Exception:
        return 0.0


@performance_tracked(log_performance=True, track_memory=True) if HARDWARE_OPTIMIZATION_AVAILABLE else lambda x: x
def calculate_economic_scores(
    cluster_assignments: np.ndarray,
    market_data: pd.DataFrame
) -> Dict[str, float]:
    """Calculate enhanced economic scores for clustering results."""
    try:
        tprint_info("Calculating economic scores")
        
        # Validate inputs
        math_validate_numeric_array(cluster_assignments, "cluster_assignments")
        
        # Calculate economic score based on market data characteristics
        economic_score = calculate_market_economic_score(cluster_assignments, market_data)
        
        # Calculate trading score based on trading characteristics
        trading_score = calculate_trading_characteristics_score(cluster_assignments, market_data)
        
        # Calculate profitability potential
        profitability_score = calculate_profitability_potential(cluster_assignments, market_data)
        
        # Calculate market efficiency score
        efficiency_score = calculate_market_efficiency_score(cluster_assignments, market_data)
        
        metrics = {
            'economic_score': economic_score,
            'trading_score': trading_score,
            'profitability_score': profitability_score,
            'efficiency_score': efficiency_score,
            'overall_economic': (economic_score + trading_score + profitability_score + efficiency_score) / 4
        }
        
        tprint_success(f"Economic scores calculated: {metrics}")
        return metrics
        
    except Exception as e:
        tprint_error(f"Failed to calculate economic scores: {e}")
        return {'economic_score': 0.0, 'trading_score': 0.0, 'profitability_score': 0.0, 'efficiency_score': 0.0}


def calculate_market_economic_score(cluster_assignments: np.ndarray, market_data: pd.DataFrame) -> float:
    """Calculate market economic score based on clustering quality."""
    try:
        # This is a simplified economic score calculation
        # In practice, you'd use more sophisticated economic indicators
        
        n_clusters = len(np.unique(cluster_assignments))
        if n_clusters <= 1:
            return 0.0
        
        # Calculate cluster size balance (more balanced = better economic structure)
        cluster_sizes = [np.sum(cluster_assignments == i) for i in np.unique(cluster_assignments)]
        total_samples = len(cluster_assignments)
        expected_size = total_samples / n_clusters
        
        size_deviations = [abs(size - expected_size) for size in cluster_sizes]
        balance_score = 1.0 - (np.mean(size_deviations) / expected_size)
        
        return max(0.0, min(1.0, balance_score))
    except Exception:
        return 0.0


def calculate_trading_characteristics_score(cluster_assignments: np.ndarray, market_data: pd.DataFrame) -> float:
    """Calculate trading characteristics score."""
    try:
        # Check if we have trading-related columns
        trading_columns = ['volume', 'close', 'open', 'high', 'low']
        available_trading_cols = [col for col in trading_columns if col in market_data.columns]
        
        if not available_trading_cols:
            return 0.0
        
        # Calculate trading pattern consistency within clusters
        total_consistency = 0.0
        cluster_count = 0
        
        for cluster_id in np.unique(cluster_assignments):
            cluster_mask = cluster_assignments == cluster_id
            cluster_data = market_data[cluster_mask]
            
            if len(cluster_data) > 1:
                # Calculate coefficient of variation for trading metrics
                trading_metrics = cluster_data[available_trading_cols]
                cv_scores = []
                
                for col in available_trading_cols:
                    values = trading_metrics[col].dropna()
                    if len(values) > 1:
                        mean_val = values.mean()
                        std_val = values.std()
                        cv = std_val / mean_val if mean_val != 0 else 1.0
                        cv_scores.append(cv)
                
                if cv_scores:
                    avg_cv = np.mean(cv_scores)
                    consistency = 1.0 / (1.0 + avg_cv)  # Lower CV = higher consistency
                    total_consistency += consistency
                    cluster_count += 1
        
        return total_consistency / cluster_count if cluster_count > 0 else 0.0
    except Exception:
        return 0.0


def calculate_profitability_potential(cluster_assignments: np.ndarray, market_data: pd.DataFrame) -> float:
    """Calculate profitability potential score."""
    try:
        # This is a simplified profitability calculation
        # In practice, you'd use more sophisticated financial metrics
        
        n_clusters = len(np.unique(cluster_assignments))
        if n_clusters <= 1:
            return 0.0
        
        # Calculate cluster diversity (more diverse clusters = better profitability potential)
        cluster_centers = []
        for cluster_id in np.unique(cluster_assignments):
            cluster_mask = cluster_assignments == cluster_id
            cluster_data = market_data[cluster_mask]
            
            if len(cluster_data) > 0:
                numeric_data = cluster_data.select_dtypes(include=[np.number])
                if len(numeric_data.columns) > 0:
                    cluster_centers.append(numeric_data.mean().values)
        
        if len(cluster_centers) <= 1:
            return 0.0
        
        # Calculate average pairwise distance between cluster centers
        total_distance = 0.0
        pair_count = 0
        
        for i in range(len(cluster_centers)):
            for j in range(i + 1, len(cluster_centers)):
                distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                total_distance += distance
                pair_count += 1
        
        if pair_count == 0:
            return 0.0
        
        avg_distance = total_distance / pair_count
        
        # Normalize based on data variance
        numeric_data = market_data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 0:
            data_variance = np.var(numeric_data.values)
            normalized_distance = min(1.0, avg_distance / np.sqrt(data_variance))
            return normalized_distance
        
        return 0.0
    except Exception:
        return 0.0


def calculate_market_efficiency_score(cluster_assignments: np.ndarray, market_data: pd.DataFrame) -> float:
    """Calculate market efficiency score."""
    try:
        # This is a simplified efficiency calculation
        # In practice, you'd use more sophisticated efficiency metrics
        
        n_clusters = len(np.unique(cluster_assignments))
        if n_clusters <= 1:
            return 0.0
        
        # Calculate cluster size efficiency (more uniform sizes = better efficiency)
        cluster_sizes = [np.sum(cluster_assignments == i) for i in np.unique(cluster_assignments)]
        total_samples = len(cluster_assignments)
        expected_size = total_samples / n_clusters
        
        # Calculate efficiency as inverse of size variance
        size_variance = np.var(cluster_sizes)
        max_variance = (total_samples / 2) ** 2  # Maximum possible variance
        efficiency = 1.0 - (size_variance / max_variance)
        
        return max(0.0, min(1.0, efficiency))
    except Exception:
        return 0.0


@performance_tracked(log_performance=True, track_memory=True) if HARDWARE_OPTIMIZATION_AVAILABLE else lambda x: x
def calculate_trading_scores(
    cluster_assignments: np.ndarray,
    market_data: pd.DataFrame
) -> Dict[str, float]:
    """Calculate enhanced trading scores for clustering results."""
    try:
        tprint_info("Calculating trading scores")
        
        # Validate inputs
        math_validate_numeric_array(cluster_assignments, "cluster_assignments")
        
        # Calculate trading score based on trading characteristics
        trading_score = calculate_trading_characteristics_score(cluster_assignments, market_data)
        
        # Calculate profitability score
        profitability_score = calculate_profitability_potential(cluster_assignments, market_data)
        
        # Calculate risk-adjusted score
        risk_adjusted_score = calculate_risk_adjusted_score(cluster_assignments, market_data)
        
        # Calculate liquidity score
        liquidity_score = calculate_liquidity_score(cluster_assignments, market_data)
        
        metrics = {
            'trading_score': trading_score,
            'profitability_score': profitability_score,
            'risk_adjusted_score': risk_adjusted_score,
            'liquidity_score': liquidity_score,
            'overall_trading': (trading_score + profitability_score + risk_adjusted_score + liquidity_score) / 4
        }
        
        tprint_success(f"Trading scores calculated: {metrics}")
        return metrics
        
    except Exception as e:
        tprint_error(f"Failed to calculate trading scores: {e}")
        return {'trading_score': 0.0, 'profitability_score': 0.0, 'risk_adjusted_score': 0.0, 'liquidity_score': 0.0}


def calculate_risk_adjusted_score(cluster_assignments: np.ndarray, market_data: pd.DataFrame) -> float:
    """Calculate risk-adjusted trading score."""
    try:
        # This is a simplified risk calculation
        # In practice, you'd use more sophisticated risk metrics
        
        n_clusters = len(np.unique(cluster_assignments))
        if n_clusters <= 1:
            return 0.0
        
        # Calculate risk as cluster volatility
        cluster_volatilities = []
        for cluster_id in np.unique(cluster_assignments):
            cluster_mask = cluster_assignments == cluster_id
            cluster_data = market_data[cluster_mask]
            
            if len(cluster_data) > 1:
                numeric_data = cluster_data.select_dtypes(include=[np.number])
                if len(numeric_data.columns) > 0:
                    # Calculate average volatility across numeric columns
                    volatilities = []
                    for col in numeric_data.columns:
                        values = numeric_data[col].dropna()
                        if len(values) > 1:
                            volatility = values.std() / values.mean() if values.mean() != 0 else 1.0
                            volatilities.append(volatility)
                    
                    if volatilities:
                        cluster_volatilities.append(np.mean(volatilities))
        
        if not cluster_volatilities:
            return 0.0
        
        # Risk-adjusted score is inverse of average volatility
        avg_volatility = np.mean(cluster_volatilities)
        risk_adjusted_score = 1.0 / (1.0 + avg_volatility)
        
        return min(1.0, risk_adjusted_score)
    except Exception:
        return 0.0


def calculate_liquidity_score(cluster_assignments: np.ndarray, market_data: pd.DataFrame) -> float:
    """Calculate liquidity score based on trading volume."""
    try:
        if 'volume' not in market_data.columns:
            return 0.0
        
        # Calculate liquidity as average volume per cluster
        cluster_liquidity = []
        for cluster_id in np.unique(cluster_assignments):
            cluster_mask = cluster_assignments == cluster_id
            cluster_data = market_data[cluster_mask]
            
            if len(cluster_data) > 0:
                volume_data = cluster_data['volume'].dropna()
                if len(volume_data) > 0:
                    avg_volume = volume_data.mean()
                    cluster_liquidity.append(avg_volume)
        
        if not cluster_liquidity:
            return 0.0
        
        # Normalize liquidity score
        max_liquidity = max(cluster_liquidity)
        normalized_liquidity = np.mean(cluster_liquidity) / max_liquidity if max_liquidity > 0 else 0.0
        
        return min(1.0, normalized_liquidity)
    except Exception:
        return 0.0


@performance_tracked(log_performance=True, track_memory=True) if HARDWARE_OPTIMIZATION_AVAILABLE else lambda x: x
def calculate_stability_scores(
    cluster_assignments: np.ndarray,
    market_data: pd.DataFrame
) -> Dict[str, float]:
    """Calculate enhanced stability scores for clustering results."""
    try:
        tprint_info("Calculating stability scores")
        
        # Validate inputs
        math_validate_numeric_array(cluster_assignments, "cluster_assignments")
        
        # Calculate stability score based on cluster consistency
        stability_score = calculate_cluster_stability_score(cluster_assignments, market_data)
        
        # Calculate consistency score
        consistency_score = calculate_cluster_consistency_score(cluster_assignments, market_data)
        
        # Calculate robustness score
        robustness_score = calculate_cluster_robustness_score(cluster_assignments, market_data)
        
        # Calculate temporal stability (if time series data available)
        temporal_stability = calculate_temporal_stability(cluster_assignments, market_data)
        
        metrics = {
            'stability_score': stability_score,
            'consistency_score': consistency_score,
            'robustness_score': robustness_score,
            'temporal_stability': temporal_stability,
            'overall_stability': (stability_score + consistency_score + robustness_score + temporal_stability) / 4
        }
        
        tprint_success(f"Stability scores calculated: {metrics}")
        return metrics
        
    except Exception as e:
        tprint_error(f"Failed to calculate stability scores: {e}")
        return {'stability_score': 0.0, 'consistency_score': 0.0, 'robustness_score': 0.0, 'temporal_stability': 0.0}


def calculate_cluster_consistency_score(cluster_assignments: np.ndarray, market_data: pd.DataFrame) -> float:
    """Calculate cluster consistency score."""
    try:
        # This is a simplified consistency calculation
        # In practice, you'd use more sophisticated consistency metrics
        
        n_clusters = len(np.unique(cluster_assignments))
        if n_clusters <= 1:
            return 0.0
        
        # Calculate consistency based on cluster size uniformity
        cluster_sizes = [np.sum(cluster_assignments == i) for i in np.unique(cluster_assignments)]
        total_samples = len(cluster_assignments)
        expected_size = total_samples / n_clusters
        
        # Calculate coefficient of variation of cluster sizes
        mean_size = np.mean(cluster_sizes)
        std_size = np.std(cluster_sizes)
        cv = std_size / mean_size if mean_size > 0 else 1.0
        
        # Consistency is inverse of coefficient of variation
        consistency = 1.0 / (1.0 + cv)
        
        return min(1.0, consistency)
    except Exception:
        return 0.0


def calculate_cluster_robustness_score(cluster_assignments: np.ndarray, market_data: pd.DataFrame) -> float:
    """Calculate cluster robustness score."""
    try:
        # This is a simplified robustness calculation
        # In practice, you'd use more sophisticated robustness metrics
        
        n_clusters = len(np.unique(cluster_assignments))
        if n_clusters <= 1:
            return 0.0
        
        # Calculate robustness based on cluster separation
        numeric_data = market_data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) == 0:
            return 0.0
        
        cluster_centers = []
        for cluster_id in np.unique(cluster_assignments):
            cluster_mask = cluster_assignments == cluster_id
            cluster_data = numeric_data[cluster_mask]
            
            if len(cluster_data) > 0:
                cluster_centers.append(cluster_data.mean().values)
        
        if len(cluster_centers) <= 1:
            return 0.0
        
        # Calculate minimum distance between cluster centers
        min_distance = float('inf')
        for i in range(len(cluster_centers)):
            for j in range(i + 1, len(cluster_centers)):
                distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                min_distance = min(min_distance, distance)
        
        # Calculate average cluster spread
        total_spread = 0.0
        for cluster_id in np.unique(cluster_assignments):
            cluster_mask = cluster_assignments == cluster_id
            cluster_data = numeric_data[cluster_mask]
            
            if len(cluster_data) > 1:
                cluster_center = cluster_data.mean().values
                distances = [np.linalg.norm(point - cluster_center) for point in cluster_data.values]
                total_spread += np.mean(distances)
        
        avg_spread = total_spread / n_clusters
        
        # Robustness is ratio of minimum distance to average spread
        robustness = min_distance / avg_spread if avg_spread > 0 else 0.0
        
        return min(1.0, robustness)
    except Exception:
        return 0.0


def calculate_temporal_stability(cluster_assignments: np.ndarray, market_data: pd.DataFrame) -> float:
    """Calculate temporal stability score."""
    try:
        # This is a simplified temporal stability calculation
        # In practice, you'd use more sophisticated temporal analysis
        
        # For now, return a placeholder value
        # In a real implementation, you'd analyze how cluster assignments change over time
        return 0.5
    except Exception:
        return 0.0


class MetricsCalculator:
    """Enhanced calculator for various clustering metrics with comprehensive utility integrations."""
    
    def __init__(self, enable_hardware_optimization: bool = True, enable_ml_optimization: bool = True):
        self.logger = get_logger('MetricsCalculator')
        self.enable_hardware_optimization = enable_hardware_optimization and HARDWARE_OPTIMIZATION_AVAILABLE
        self.enable_ml_optimization = enable_ml_optimization and ML_COMMON_AVAILABLE
        
        # Initialize hardware manager if available
        if self.enable_hardware_optimization:
            try:
                self.hardware_manager = get_integrated_hardware_manager()
                tprint_info("Hardware optimization enabled for metrics calculator")
            except Exception as e:
                tprint_warning(f"Failed to initialize hardware manager: {e}")
                self.hardware_manager = None
        else:
            self.hardware_manager = None
        
        # Initialize ML optimization components if available
        if self.enable_ml_optimization:
            try:
                self.model_validator = ModelValidator() if ModelValidator else None
                self.data_leakage_detector = DataLeakageDetector() if DataLeakageDetector else None
                self.lookahead_bias_detector = LookaheadBiasDetector() if LookaheadBiasDetector else None
                tprint_info("ML optimization enabled for metrics calculator")
            except Exception as e:
                tprint_warning(f"Failed to initialize ML optimization: {e}")
                self.model_validator = None
                self.data_leakage_detector = None
                self.lookahead_bias_detector = None
    
    @performance_tracked(log_performance=True, track_memory=True) if HARDWARE_OPTIMIZATION_AVAILABLE else lambda x: x
    def calculate_all_metrics(
        self,
        cluster_assignments: np.ndarray,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate all available metrics with enhanced functionality."""
        try:
            tprint_info("Starting comprehensive metrics calculation")
            start_time = time.time()
            
            # Validate inputs
            math_validate_numeric_array(cluster_assignments, "cluster_assignments")
            validate_dataframe_columns(market_data, [])
            
            # Initialize metrics dictionary
            metrics = {
                'calculation_timestamp': get_current_datetime(),
                'input_validation': {
                    'n_clusters': len(np.unique(cluster_assignments)),
                    'n_samples': len(cluster_assignments),
                    'n_features': market_data.shape[1],
                    'data_types': market_data.dtypes.to_dict()
                }
            }
            
            # Calculate consensus metrics
            tprint_info("Calculating consensus metrics")
            consensus_metrics = calculate_consensus_metrics(cluster_assignments, market_data)
            metrics['consensus'] = consensus_metrics
            
            # Calculate disagreement metrics
            tprint_info("Calculating disagreement metrics")
            disagreement_metrics = calculate_disagreement_metrics(cluster_assignments, market_data)
            metrics['disagreement'] = disagreement_metrics
            
            # Calculate economic scores
            tprint_info("Calculating economic scores")
            economic_metrics = calculate_economic_scores(cluster_assignments, market_data)
            metrics['economic'] = economic_metrics
            
            # Calculate trading scores
            tprint_info("Calculating trading scores")
            trading_metrics = calculate_trading_scores(cluster_assignments, market_data)
            metrics['trading'] = trading_metrics
            
            # Calculate stability scores
            tprint_info("Calculating stability scores")
            stability_metrics = calculate_stability_scores(cluster_assignments, market_data)
            metrics['stability'] = stability_metrics
            
            # Calculate additional ML metrics if available
            if self.enable_ml_optimization:
                tprint_info("Calculating ML-specific metrics")
                ml_metrics = self._calculate_ml_metrics(cluster_assignments, market_data)
                metrics['ml_specific'] = ml_metrics
            
            # Calculate data quality metrics
            tprint_info("Calculating data quality metrics")
            data_quality_metrics = self._calculate_data_quality_metrics(cluster_assignments, market_data)
            metrics['data_quality'] = data_quality_metrics
            
            # Calculate performance metrics
            processing_time = time.time() - start_time
            memory_usage = get_memory_usage() if HARDWARE_OPTIMIZATION_AVAILABLE else {}
            
            metrics['performance'] = {
                'processing_time': processing_time,
                'memory_usage': memory_usage,
                'hardware_optimization_enabled': self.enable_hardware_optimization,
                'ml_optimization_enabled': self.enable_ml_optimization
            }
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(metrics)
            metrics['overall_score'] = overall_score
            
            tprint_success(f"Comprehensive metrics calculation completed in {processing_time:.2f} seconds")
            return metrics
            
        except Exception as e:
            tprint_error(f"Metrics calculation failed: {e}")
            self.logger.error(f"Metrics calculation failed: {e}")
            return {'error': str(e), 'timestamp': get_current_datetime()}
    
    def _calculate_ml_metrics(self, cluster_assignments: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate ML-specific metrics."""
        try:
            ml_metrics = {}
            
            # Data leakage detection
            if self.data_leakage_detector:
                try:
                    leakage_score = self.data_leakage_detector.detect_leakage(market_data)
                    ml_metrics['data_leakage_score'] = leakage_score
                except Exception as e:
                    tprint_warning(f"Data leakage detection failed: {e}")
            
            # Lookahead bias detection
            if self.lookahead_bias_detector:
                try:
                    bias_score = self.lookahead_bias_detector.detect_bias(market_data)
                    ml_metrics['lookahead_bias_score'] = bias_score
                except Exception as e:
                    tprint_warning(f"Lookahead bias detection failed: {e}")
            
            # Model validation metrics
            if self.model_validator:
                try:
                    validation_metrics = self.model_validator.validate_clustering(cluster_assignments, market_data)
                    ml_metrics['model_validation'] = validation_metrics
                except Exception as e:
                    tprint_warning(f"Model validation failed: {e}")
            
            return ml_metrics
            
        except Exception as e:
            tprint_warning(f"ML metrics calculation failed: {e}")
            return {}
    
    def _calculate_data_quality_metrics(self, cluster_assignments: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data quality metrics."""
        try:
            # Basic data quality metrics
            data_quality_metrics = calculate_data_quality_metrics(market_data)
            
            # Cluster-specific data quality
            cluster_quality_metrics = {}
            for cluster_id in np.unique(cluster_assignments):
                cluster_mask = cluster_assignments == cluster_id
                cluster_data = market_data[cluster_mask]
                
                if len(cluster_data) > 0:
                    cluster_quality = calculate_data_quality_metrics(cluster_data)
                    cluster_quality_metrics[f'cluster_{cluster_id}'] = cluster_quality
            
            return {
                'overall_quality': data_quality_metrics,
                'cluster_quality': cluster_quality_metrics
            }
            
        except Exception as e:
            tprint_warning(f"Data quality metrics calculation failed: {e}")
            return {}
    
    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall clustering quality score."""
        try:
            # Extract key scores from different metric categories
            scores = []
            
            # Consensus scores
            if 'consensus' in metrics:
                consensus = metrics['consensus']
                if 'overall_consensus' in consensus:
                    scores.append(consensus['overall_consensus'])
            
            # Economic scores
            if 'economic' in metrics:
                economic = metrics['economic']
                if 'overall_economic' in economic:
                    scores.append(economic['overall_economic'])
            
            # Trading scores
            if 'trading' in metrics:
                trading = metrics['trading']
                if 'overall_trading' in trading:
                    scores.append(trading['overall_trading'])
            
            # Stability scores
            if 'stability' in metrics:
                stability = metrics['stability']
                if 'overall_stability' in stability:
                    scores.append(stability['overall_stability'])
            
            # Calculate weighted average
            if scores:
                # Equal weights for now, but could be made configurable
                weights = [1.0] * len(scores)
                overall_score = np.average(scores, weights=weights)
                return min(1.0, max(0.0, overall_score))
            
            return 0.0
            
        except Exception as e:
            tprint_warning(f"Overall score calculation failed: {e}")
            return 0.0


class CharacteristicsGenerator:
    """Enhanced generator for cluster characteristics with comprehensive utility integrations."""
    
    def __init__(self, enable_hardware_optimization: bool = True, enable_ml_optimization: bool = True):
        self.logger = get_logger('CharacteristicsGenerator')
        self.enable_hardware_optimization = enable_hardware_optimization and HARDWARE_OPTIMIZATION_AVAILABLE
        self.enable_ml_optimization = enable_ml_optimization and ML_COMMON_AVAILABLE
        
        # Initialize hardware manager if available
        if self.enable_hardware_optimization:
            try:
                self.hardware_manager = get_integrated_hardware_manager()
                tprint_info("Hardware optimization enabled for characteristics generator")
            except Exception as e:
                tprint_warning(f"Failed to initialize hardware manager: {e}")
                self.hardware_manager = None
        else:
            self.hardware_manager = None
    
    @performance_tracked(log_performance=True, track_memory=True) if HARDWARE_OPTIMIZATION_AVAILABLE else lambda x: x
    def generate_characteristics(
        self,
        cluster_assignments: np.ndarray,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate comprehensive cluster characteristics with enhanced analysis."""
        try:
            tprint_info("Starting comprehensive cluster characteristics generation")
            start_time = time.time()
            
            # Validate inputs
            math_validate_numeric_array(cluster_assignments, "cluster_assignments")
            validate_dataframe_columns(market_data, [])
            
            n_clusters = len(np.unique(cluster_assignments))
            characteristics = {
                'generation_timestamp': get_current_datetime(),
                'n_clusters': n_clusters,
                'n_samples': len(cluster_assignments),
                'cluster_sizes': [np.sum(cluster_assignments == i) for i in range(n_clusters)],
                'cluster_characteristics': {},
                'summary_statistics': {},
                'data_quality_metrics': {},
                'performance_metrics': {}
            }
            
            # Generate characteristics for each cluster
            for i in range(n_clusters):
                cluster_mask = cluster_assignments == i
                cluster_data = market_data[cluster_mask]
                
                if len(cluster_data) > 0:
                    cluster_char = self._generate_cluster_characteristics(i, cluster_data, cluster_mask)
                    characteristics['cluster_characteristics'][f'cluster_{i}'] = cluster_char
                else:
                    tprint_warning(f"Cluster {i} is empty")
                    characteristics['cluster_characteristics'][f'cluster_{i}'] = {
                        'size': 0,
                        'empty_cluster': True
                    }
            
            # Generate summary statistics
            characteristics['summary_statistics'] = self._generate_summary_statistics(characteristics)
            
            # Generate data quality metrics
            characteristics['data_quality_metrics'] = self._generate_data_quality_metrics(cluster_assignments, market_data)
            
            # Generate performance metrics
            processing_time = time.time() - start_time
            memory_usage = get_memory_usage() if HARDWARE_OPTIMIZATION_AVAILABLE else {}
            
            characteristics['performance_metrics'] = {
                'processing_time': processing_time,
                'memory_usage': memory_usage,
                'hardware_optimization_enabled': self.enable_hardware_optimization,
                'ml_optimization_enabled': self.enable_ml_optimization
            }
            
            tprint_success(f"Cluster characteristics generation completed in {processing_time:.2f} seconds")
            return characteristics
            
        except Exception as e:
            tprint_error(f"Characteristics generation failed: {e}")
            self.logger.error(f"Characteristics generation failed: {e}")
            return {'n_clusters': 0, 'cluster_sizes': [], 'cluster_characteristics': {}, 'error': str(e)}
    
    def _generate_cluster_characteristics(self, cluster_id: int, cluster_data: pd.DataFrame, cluster_mask: np.ndarray) -> Dict[str, Any]:
        """Generate comprehensive characteristics for a single cluster."""
        try:
            characteristics = {
                'cluster_id': cluster_id,
                'size': np.sum(cluster_mask),
                'size_percentage': (np.sum(cluster_mask) / len(cluster_mask)) * 100,
                'basic_statistics': {},
                'trading_characteristics': {},
                'data_quality': {},
                'feature_analysis': {}
            }
            
            # Basic statistics
            characteristics['basic_statistics'] = self._calculate_basic_statistics(cluster_data)
            
            # Trading characteristics
            characteristics['trading_characteristics'] = self._calculate_trading_characteristics(cluster_data)
            
            # Data quality metrics
            characteristics['data_quality'] = self._calculate_cluster_data_quality(cluster_data)
            
            # Feature analysis
            characteristics['feature_analysis'] = self._analyze_cluster_features(cluster_data)
            
            return characteristics
            
        except Exception as e:
            tprint_warning(f"Failed to generate characteristics for cluster {cluster_id}: {e}")
            return {
                'cluster_id': cluster_id,
                'size': np.sum(cluster_mask),
                'error': str(e)
            }
    
    def _calculate_basic_statistics(self, cluster_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistical characteristics for a cluster."""
        try:
            numeric_data = cluster_data.select_dtypes(include=[np.number])
            
            if len(numeric_data.columns) == 0:
                return {'no_numeric_data': True}
            
            stats = {}
            for col in numeric_data.columns:
                values = numeric_data[col].dropna()
                if len(values) > 0:
                    stats[col] = {
                        'count': len(values),
                        'mean': safe_mean(values),
                        'std': safe_std(values),
                        'min': values.min(),
                        'max': values.max(),
                        'median': values.median(),
                        'q25': values.quantile(0.25),
                        'q75': values.quantile(0.75),
                        'skewness': values.skew() if len(values) > 2 else 0.0,
                        'kurtosis': values.kurtosis() if len(values) > 2 else 0.0
                    }
            
            return stats
            
        except Exception as e:
            tprint_warning(f"Failed to calculate basic statistics: {e}")
            return {}
    
    def _calculate_trading_characteristics(self, cluster_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trading-specific characteristics for a cluster."""
        try:
            trading_chars = {}
            
            # Volume characteristics
            if 'volume' in cluster_data.columns:
                volume_data = cluster_data['volume'].dropna()
                if len(volume_data) > 0:
                    trading_chars['volume'] = {
                        'mean': safe_mean(volume_data),
                        'std': safe_std(volume_data),
                        'min': volume_data.min(),
                        'max': volume_data.max(),
                        'median': volume_data.median()
                    }
            
            # Price characteristics
            price_columns = ['close', 'open', 'high', 'low']
            for col in price_columns:
                if col in cluster_data.columns:
                    price_data = cluster_data[col].dropna()
                    if len(price_data) > 0:
                        trading_chars[col] = {
                            'mean': safe_mean(price_data),
                            'std': safe_std(price_data),
                            'min': price_data.min(),
                            'max': price_data.max(),
                            'median': price_data.median()
                        }
            
            # Price range characteristics
            if all(col in cluster_data.columns for col in ['high', 'low']):
                high_data = cluster_data['high'].dropna()
                low_data = cluster_data['low'].dropna()
                if len(high_data) > 0 and len(low_data) > 0:
                    price_ranges = high_data - low_data
                    trading_chars['price_range'] = {
                        'mean': safe_mean(price_ranges),
                        'std': safe_std(price_ranges),
                        'min': price_ranges.min(),
                        'max': price_ranges.max()
                    }
            
            return trading_chars
            
        except Exception as e:
            tprint_warning(f"Failed to calculate trading characteristics: {e}")
            return {}
    
    def _calculate_cluster_data_quality(self, cluster_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data quality metrics for a cluster."""
        try:
            # Calculate data quality metrics
            data_quality = calculate_data_quality_metrics(cluster_data)
            
            # Calculate additional cluster-specific quality metrics
            numeric_data = cluster_data.select_dtypes(include=[np.number])
            
            quality_metrics = {
                'overall_quality': data_quality,
                'missing_data_percentage': (cluster_data.isnull().sum().sum() / cluster_data.size) * 100,
                'numeric_columns': len(numeric_data.columns),
                'total_columns': len(cluster_data.columns)
            }
            
            # Calculate feature correlation within cluster
            if len(numeric_data.columns) > 1:
                try:
                    corr_matrix = numeric_data.corr()
                    # Calculate average absolute correlation
                    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    avg_correlation = upper_tri.stack().abs().mean()
                    quality_metrics['avg_correlation'] = avg_correlation if not np.isnan(avg_correlation) else 0.0
                except Exception as e:
                    tprint_warning(f"Failed to calculate correlation: {e}")
                    quality_metrics['avg_correlation'] = 0.0
            
            return quality_metrics
            
        except Exception as e:
            tprint_warning(f"Failed to calculate cluster data quality: {e}")
            return {}
    
    def _analyze_cluster_features(self, cluster_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature characteristics within a cluster."""
        try:
            numeric_data = cluster_data.select_dtypes(include=[np.number])
            
            if len(numeric_data.columns) == 0:
                return {'no_numeric_features': True}
            
            feature_analysis = {
                'n_features': len(numeric_data.columns),
                'feature_names': list(numeric_data.columns),
                'feature_importance': {},
                'feature_correlations': {}
            }
            
            # Calculate feature importance (simplified version)
            for col in numeric_data.columns:
                values = numeric_data[col].dropna()
                if len(values) > 1:
                    # Use variance as a simple importance measure
                    importance = values.var()
                    feature_analysis['feature_importance'][col] = importance
            
            # Calculate feature correlations
            if len(numeric_data.columns) > 1:
                try:
                    corr_matrix = numeric_data.corr()
                    feature_analysis['feature_correlations'] = corr_matrix.to_dict()
                except Exception as e:
                    tprint_warning(f"Failed to calculate feature correlations: {e}")
            
            return feature_analysis
            
        except Exception as e:
            tprint_warning(f"Failed to analyze cluster features: {e}")
            return {}
    
    def _generate_summary_statistics(self, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics across all clusters."""
        try:
            cluster_chars = characteristics['cluster_characteristics']
            
            summary = {
                'total_clusters': len(cluster_chars),
                'total_samples': sum(char.get('size', 0) for char in cluster_chars.values()),
                'cluster_size_distribution': {},
                'average_cluster_size': 0.0,
                'cluster_size_std': 0.0
            }
            
            if cluster_chars:
                cluster_sizes = [char.get('size', 0) for char in cluster_chars.values()]
                summary['average_cluster_size'] = safe_mean(cluster_sizes)
                summary['cluster_size_std'] = safe_std(cluster_sizes)
                summary['cluster_size_distribution'] = {
                    'min': min(cluster_sizes),
                    'max': max(cluster_sizes),
                    'median': np.median(cluster_sizes)
                }
            
            return summary
            
        except Exception as e:
            tprint_warning(f"Failed to generate summary statistics: {e}")
            return {}
    
    def _generate_data_quality_metrics(self, cluster_assignments: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate data quality metrics for the entire dataset."""
        try:
            # Overall data quality
            overall_quality = calculate_data_quality_metrics(market_data)
            
            # Cluster-specific quality metrics
            cluster_quality = {}
            for cluster_id in np.unique(cluster_assignments):
                cluster_mask = cluster_assignments == cluster_id
                cluster_data = market_data[cluster_mask]
                
                if len(cluster_data) > 0:
                    cluster_quality[f'cluster_{cluster_id}'] = calculate_data_quality_metrics(cluster_data)
            
            return {
                'overall_quality': overall_quality,
                'cluster_quality': cluster_quality
            }
            
        except Exception as e:
            tprint_warning(f"Failed to generate data quality metrics: {e}")
            return {}