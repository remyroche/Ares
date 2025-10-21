"""
Clustering Utilities Module

This module provides utility functions for clustering operations including distance metrics,
kNN graph building, centroid updates, and incremental BCSS/WCSS calculations.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import time
from datetime import datetime

# Import utility modules
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
    get_m1_memory_optimizer, get_m1_cpu_optimizer, is_m1_available, is_mps_available,
    # Additional common operations utilities
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    analyze_nan_values_detailed, safe_apply_with_validation, safe_aggregate_data,
    safe_merge_dataframes, safe_drop_columns, safe_fillna, safe_dropna,
    safe_reset_index, safe_sort_values, safe_groupby_agg, safe_pivot_table,
    safe_melt_dataframe, safe_concat_dataframes, safe_join_dataframes,
    safe_apply_custom_function, safe_transform_dataframe, safe_validate_dataframe,
    safe_export_dataframe, safe_import_dataframe, safe_compress_dataframe,
    safe_decompress_dataframe, safe_serialize_dataframe, safe_deserialize_dataframe
)

# Import common utilities
from src.utils.common_utilities import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    analyze_nan_values_detailed, safe_apply_with_validation, safe_aggregate_data,
    safe_merge_dataframes, safe_drop_columns, safe_fillna, safe_dropna,
    safe_reset_index, safe_sort_values, safe_groupby_agg, safe_pivot_table,
    safe_melt_dataframe, safe_concat_dataframes, safe_join_dataframes,
    safe_apply_custom_function, safe_transform_dataframe, safe_validate_dataframe,
    safe_export_dataframe, safe_import_dataframe, safe_compress_dataframe,
    safe_decompress_dataframe, safe_serialize_dataframe, safe_deserialize_dataframe,
    # Data quality utilities
    calculate_data_quality_score, detect_data_anomalies, validate_data_consistency,
    clean_data_automatically, standardize_data_format, validate_data_types,
    check_data_completeness, validate_data_ranges, detect_outliers,
    validate_data_relationships, check_data_duplicates, validate_data_integrity,
    # Performance utilities
    optimize_dataframe_performance, reduce_memory_usage, optimize_dtypes,
    compress_dataframe, decompress_dataframe, cache_dataframe, load_cached_dataframe,
    # Hardware optimization utilities
    get_hardware_info, optimize_for_hardware, get_memory_usage, get_cpu_usage,
    get_gpu_usage, optimize_memory_allocation, optimize_cpu_usage, optimize_gpu_usage
)

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_structured,
    tprint_timer, tprint_logged, LogLevel, TimestampFormat
)

from src.utils.math_validation import (
    MathValidationError, safe_divide as math_safe_divide, safe_log as math_safe_log,
    safe_sqrt as math_safe_sqrt, safe_power as math_safe_power,
    validate_finite as math_validate_finite, validate_positive as math_validate_positive,
    validate_range as math_validate_range, validate_numeric_array as math_validate_numeric_array
)

# Import hardware utilities
try:
    from src.utils.hardware.m1_gpu_utils import is_m1_available as hw_is_m1_available, is_mps_available as hw_is_mps_available
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer as hw_get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer as hw_get_m1_cpu_optimizer
    from src.utils.hardware.integrated_hardware_manager import get_integrated_hardware_manager
    from src.utils.hardware.optimization_decorators import (
        smart_cache, auto_optimize, memory_efficient, performance_tracked
    )
    from src.utils.hardware.memory_optimized_decorators import (
        memory_optimized, comprehensive_memory_optimization, MemoryOptimizationLevel
    )
    from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager
    from src.utils.hardware.vectorbt_gpu_accelerator import VectorBTRollingOptimizer, UnifiedVectorizationManager
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    hw_is_m1_available = lambda: False
    hw_is_mps_available = lambda: False
    hw_get_m1_memory_optimizer = lambda: None
    hw_get_m1_cpu_optimizer = lambda: None
    get_integrated_hardware_manager = lambda: None
    smart_cache = lambda *args, **kwargs: lambda f: f
    auto_optimize = lambda *args, **kwargs: lambda f: f
    memory_efficient = lambda *args, **kwargs: lambda f: f
    performance_tracked = lambda *args, **kwargs: lambda f: f
    memory_optimized = lambda *args, **kwargs: lambda f: f
    comprehensive_memory_optimization = lambda *args, **kwargs: lambda f: f
    MemoryOptimizationLevel = type('MemoryOptimizationLevel', (), {})
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

logger = logging.getLogger(__name__)

class ClusteringUtils:
    """Utility functions for clustering operations and calculations."""

    def __init__(self, enable_hardware_optimization: bool = True, enable_ml_optimization: bool = True):
        """Initialize clustering utilities with enhanced capabilities."""
        self.logger = logger
        self.enable_hardware_optimization = enable_hardware_optimization and HARDWARE_OPTIMIZATION_AVAILABLE
        self.enable_ml_optimization = enable_ml_optimization and ML_COMMON_AVAILABLE
        
        # Initialize hardware manager if available
        if self.enable_hardware_optimization:
            try:
                self.hardware_manager = get_integrated_hardware_manager()
                self.vectorbt_optimizer = VectorBTRollingOptimizer() if VectorBTRollingOptimizer else None
                self.vectorization_manager = UnifiedVectorizationManager() if UnifiedVectorizationManager else None
                tprint_info("Hardware optimization enabled for clustering utilities")
            except Exception as e:
                tprint_warning(f"Failed to initialize hardware optimization: {e}")
                self.hardware_manager = None
                self.vectorbt_optimizer = None
                self.vectorization_manager = None
        else:
            self.hardware_manager = None
            self.vectorbt_optimizer = None
            self.vectorization_manager = None
        
        # Initialize ML optimization components if available
        if self.enable_ml_optimization:
            try:
                self.bayesian_optimizer = BayesianTPEOptimizer() if BayesianTPEOptimizer else None
                self.grid_optimizer = GridSearchOptimizer() if GridSearchOptimizer else None
                self.hpo_optimizer = HPOOptimizer() if HPOOptimizer else None
                self.model_validator = ModelValidator() if ModelValidator else None
                self.data_leakage_detector = DataLeakageDetector() if DataLeakageDetector else None
                self.lookahead_bias_detector = LookaheadBiasDetector() if LookaheadBiasDetector else None
                tprint_info("ML optimization enabled for clustering utilities")
            except Exception as e:
                tprint_warning(f"Failed to initialize ML optimization: {e}")
                self.bayesian_optimizer = None
                self.grid_optimizer = None
                self.hpo_optimizer = None
                self.model_validator = None
                self.data_leakage_detector = None
                self.lookahead_bias_detector = None
        else:
            self.bayesian_optimizer = None
            self.grid_optimizer = None
            self.hpo_optimizer = None
            self.model_validator = None
            self.data_leakage_detector = None
            self.lookahead_bias_detector = None
        
        # Initialize data utilities if available
        if DATA_UTILS_AVAILABLE:
            try:
                self.klines_manager = KlinesParquetManager() if KlinesParquetManager else None
                self.data_manager = UnifiedDataManager() if UnifiedDataManager else None
                self.feature_engineer = FeatureEngineer() if FeatureEngineer else None
                tprint_info("Data utilities enabled for clustering utilities")
            except Exception as e:
                tprint_warning(f"Failed to initialize data utilities: {e}")
                self.klines_manager = None
                self.data_manager = None
                self.feature_engineer = None
        else:
            self.klines_manager = None
            self.data_manager = None
            self.feature_engineer = None
        
        # Initialize artifact manager if available
        if ARTIFACT_MANAGER_AVAILABLE:
            try:
                self.artifact_manager = EnhancedArtifactManager() if EnhancedArtifactManager else ArtifactManager()
                tprint_info("Artifact manager enabled for clustering utilities")
            except Exception as e:
                tprint_warning(f"Failed to initialize artifact manager: {e}")
                self.artifact_manager = None
        else:
            self.artifact_manager = None

    def calculate_euclidean_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        try:
            with tprint_timer("euclidean_distance_calculation"):
                tprint_debug("Calculating Euclidean distance between points")
                result = float(np.linalg.norm(point1 - point2))
                tprint_performance("euclidean_distance", 0.001)  # Small operation
                return result
        except Exception as e:
            tprint_error(f"Failed to calculate Euclidean distance: {e}")
            self.logger.error(f"Failed to calculate Euclidean distance: {e}")
            return float('inf')

    def calculate_cosine_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate cosine distance between two points."""
        try:
            dot_product = np.dot(point1, point2)
            norm1 = np.linalg.norm(point1)
            norm2 = np.linalg.norm(point2)

            if norm1 == 0.0 or norm2 == 0.0:
                return 1.0  # Maximum distance for zero vectors

            cosine_similarity = dot_product / (norm1 * norm2)
            return 1.0 - cosine_similarity
        except Exception as e:
            self.logger.error(f"Failed to calculate cosine distance: {e}")
            return 1.0

    def calculate_manhattan_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate Manhattan distance between two points."""
        try:
            return float(np.sum(np.abs(point1 - point2)))
        except Exception as e:
            self.logger.error(f"Failed to calculate Manhattan distance: {e}")
            return float('inf')

    def build_knn_graph(self, data: np.ndarray, k: int = 5,
                       distance_metric: str = 'euclidean') -> Tuple[np.ndarray, np.ndarray]:
        """Build k-nearest neighbors graph for the given data."""
        try:
            tprint_info(f"Building kNN graph with k={k}, metric={distance_metric}")
            n_samples = data.shape[0]

            # Validate inputs using utility functions
            math_validate_numeric_array(data, "clustering_data")
            validate_positive(k, "k_value")
            validate_range(k, min_val=1, max_val=n_samples, name="k_value")

            if distance_metric == 'euclidean':
                distance_func = self.calculate_euclidean_distance
            elif distance_metric == 'cosine':
                distance_func = self.calculate_cosine_distance
            elif distance_metric == 'manhattan':
                distance_func = self.calculate_manhattan_distance
            else:
                tprint_warning(f"Unknown distance metric: {distance_metric}, using euclidean")
                distance_func = self.calculate_euclidean_distance

            # Initialize distance matrix and indices using safe operations
            distances = np.zeros((n_samples, k), dtype=np.float64)
            indices = np.zeros((n_samples, k), dtype=np.int32)

            # Use parallel processing for large datasets
            if n_samples > 1000:
                tprint_debug(f"Using parallel processing for {n_samples} samples")

                def process_sample(i):
                    point_distances = []
                    for j in range(n_samples):
                        if i != j:
                            dist = distance_func(data[i], data[j])
                            point_distances.append((dist, j))

                    # Sort by distance and take k nearest
                    point_distances.sort(key=lambda x: x[0])
                    return [(dist, neighbor_idx) for dist, neighbor_idx in point_distances[:k]]

                # Use parallel processing utility
                results = parallel_map(process_sample, range(n_samples), max_workers=min(4, n_samples//100))

                for i, result in enumerate(results):
                    for idx, (dist, neighbor_idx) in enumerate(result):
                        distances[i, idx] = dist
                        indices[i, idx] = neighbor_idx
            else:
                # For each point, find k nearest neighbors
                for i in range(n_samples):
                    point_distances = []

                    for j in range(n_samples):
                        if i != j:
                            dist = distance_func(data[i], data[j])
                            point_distances.append((dist, j))

                    # Sort by distance and take k nearest
                    point_distances.sort(key=lambda x: x[0])
                    for idx, (dist, neighbor_idx) in enumerate(point_distances[:k]):
                        distances[i, idx] = dist
                        indices[i, idx] = neighbor_idx

            tprint_success(f"Successfully built kNN graph for {n_samples} samples")
            return distances, indices

        except Exception as e:
            tprint_error(f"Failed to build kNN graph: {e}")
            self.logger.error(f"Failed to build kNN graph: {e}")
            return np.array([]), np.array([])

    def update_centroids_incremental(self, data: np.ndarray, labels: np.ndarray,
                                   centroids: np.ndarray, new_point: np.ndarray,
                                   new_label: int) -> np.ndarray:
        """Incrementally update centroids when adding a new point."""
        try:
            n_clusters = len(np.unique(labels))
            n_features = data.shape[1]

            # Initialize with current centroids if not provided
            if centroids is None:
                centroids = self._initialize_centroids(data, labels)

            # Count points per cluster
            cluster_counts = np.bincount(labels, minlength=n_clusters)

            # Update centroids for the new point's cluster
            old_count = cluster_counts[new_label]
            new_count = old_count + 1

            # Incremental centroid update formula
            centroids[new_label] = (centroids[new_label] * old_count + new_point) / new_count

            return centroids

        except Exception as e:
            self.logger.error(f"Failed to update centroids incrementally: {e}")
            return centroids

    def _initialize_centroids(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Initialize centroids from existing data and labels."""
        try:
            n_clusters = len(np.unique(labels))
            n_features = data.shape[1]
            centroids = np.zeros((n_clusters, n_features))

            for cluster_id in range(n_clusters):
                cluster_points = data[labels == cluster_id]
                if len(cluster_points) > 0:
                    centroids[cluster_id] = np.mean(cluster_points, axis=0)
                else:
                    # Handle empty clusters
                    centroids[cluster_id] = np.random.rand(n_features)

            return centroids

        except Exception as e:
            self.logger.error(f"Failed to initialize centroids: {e}")
            return np.array([])

    def calculate_bcss_incremental(self, data: np.ndarray, labels: np.ndarray,
                                 centroids: np.ndarray, new_point: np.ndarray,
                                 new_label: int) -> float:
        """Calculate Between-Cluster Sum of Squares (BCSS) incrementally."""
        try:
            # Overall centroid (mean of all points)
            overall_centroid = np.mean(data, axis=0)

            # Current BCSS
            current_bcss = 0.0
            for i, centroid in enumerate(centroids):
                cluster_points = data[labels == i]
                if len(cluster_points) > 0:
                    cluster_size = len(cluster_points)
                    current_bcss += cluster_size * np.sum((centroid - overall_centroid) ** 2)

            # Calculate new centroids including the new point
            n_clusters = len(centroids)
            cluster_counts = np.bincount(labels, minlength=n_clusters)
            new_centroids = centroids.copy()

            # Update centroid for new point's cluster
            old_count = cluster_counts[new_label]
            new_count = old_count + 1
            new_centroids[new_label] = (centroids[new_label] * old_count + new_point) / new_count

            # New BCSS with updated centroids
            new_bcss = 0.0
            new_overall_centroid = (overall_centroid * len(data) + new_point) / (len(data) + 1)
            new_cluster_counts = cluster_counts.copy()
            new_cluster_counts[new_label] += 1

            for i, centroid in enumerate(new_centroids):
                cluster_size = new_cluster_counts[i]
                new_bcss += cluster_size * np.sum((centroid - new_overall_centroid) ** 2)

            return new_bcss

        except Exception as e:
            self.logger.error(f"Failed to calculate BCSS incrementally: {e}")
            return 0.0

    def calculate_wcss_incremental(self, data: np.ndarray, labels: np.ndarray,
                                 centroids: np.ndarray, new_point: np.ndarray,
                                 new_label: int) -> float:
        """Calculate Within-Cluster Sum of Squares (WCSS) incrementally."""
        try:
            # Current WCSS
            current_wcss = 0.0
            for i, centroid in enumerate(centroids):
                cluster_points = data[labels == i]
                if len(cluster_points) > 0:
                    current_wcss += np.sum((cluster_points - centroid) ** 2)

            # Update centroid for new point's cluster
            n_clusters = len(centroids)
            cluster_counts = np.bincount(labels, minlength=n_clusters)
            new_centroids = centroids.copy()

            old_count = cluster_counts[new_label]
            new_count = old_count + 1
            new_centroids[new_label] = (centroids[new_label] * old_count + new_point) / new_count

            # New WCSS including the new point
            new_wcss = current_wcss + np.sum((new_point - new_centroids[new_label]) ** 2)

            return new_wcss

        except Exception as e:
            self.logger.error(f"Failed to calculate WCSS incrementally: {e}")
            return 0.0

    def encode_labels(self, labels: np.ndarray, label_mapping: Optional[Dict[int, int]] = None) -> Tuple[np.ndarray, Dict[int, int]]:
        """Encode cluster labels to consecutive integers starting from 0."""
        try:
            unique_labels = np.unique(labels)
            if label_mapping is None:
                label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}

            encoded_labels = np.array([label_mapping[label] for label in labels])
            return encoded_labels, label_mapping

        except Exception as e:
            self.logger.error(f"Failed to encode labels: {e}")
            return labels, {}

    def decode_labels(self, encoded_labels: np.ndarray, label_mapping: Dict[int, int]) -> np.ndarray:
        """Decode cluster labels back to original values."""
        try:
            # Create reverse mapping
            reverse_mapping = {v: k for k, v in label_mapping.items()}
            decoded_labels = np.array([reverse_mapping[label] for label in encoded_labels])
            return decoded_labels

        except Exception as e:
            self.logger.error(f"Failed to decode labels: {e}")
            return encoded_labels

    def calculate_cluster_statistics(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive statistics for each cluster."""
        try:
            unique_labels = np.unique(labels)
            statistics = {}

            for label in unique_labels:
                cluster_data = data[labels == label]
                if len(cluster_data) > 0:
                    statistics[int(label)] = {
                        'size': len(cluster_data),
                        'mean': np.mean(cluster_data, axis=0).tolist(),
                        'std': np.std(cluster_data, axis=0).tolist(),
                        'min': np.min(cluster_data, axis=0).tolist(),
                        'max': np.max(cluster_data, axis=0).tolist(),
                        'centroid': np.mean(cluster_data, axis=0).tolist()
                    }

            return statistics

        except Exception as e:
            self.logger.error(f"Failed to calculate cluster statistics: {e}")
            return {}

    def normalize_features(self, data: np.ndarray, method: str = 'standard') -> np.ndarray:
        """Normalize features using specified method with enhanced validation and optimization."""
        try:
            # Validate input data using math validation utilities
            math_validate_numeric_array(data, "normalization_data")
            
            # Use hardware optimization if available
            if self.enable_hardware_optimization and self.hardware_manager:
                with self.hardware_manager.optimize_operation("normalization"):
                    return self._normalize_features_optimized(data, method)
            else:
                return self._normalize_features_standard(data, method)

        except Exception as e:
            tprint_error(f"Failed to normalize features: {e}")
            self.logger.error(f"Failed to normalize features: {e}")
            return data

    def _normalize_features_standard(self, data: np.ndarray, method: str) -> np.ndarray:
        """Standard normalization implementation."""
        if method == 'standard':
            # Z-score normalization
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            std = np.where(std == 0, 1, std)  # Avoid division by zero
            return (data - mean) / std

        elif method == 'minmax':
            # Min-max normalization
            min_val = np.min(data, axis=0)
            max_val = np.max(data, axis=0)
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1, range_val)  # Avoid division by zero
            return (data - min_val) / range_val

        elif method == 'robust':
            # Robust normalization using median and IQR
            median = np.median(data, axis=0)
            q75, q25 = np.percentile(data, [75, 25], axis=0)
            iqr = q75 - q25
            iqr = np.where(iqr == 0, 1, iqr)  # Avoid division by zero
            return (data - median) / iqr

        else:
            self.logger.warning(f"Unknown normalization method: {method}")
            return data

    def _normalize_features_optimized(self, data: np.ndarray, method: str) -> np.ndarray:
        """Hardware-optimized normalization implementation."""
        # Use vectorized operations and hardware acceleration
        if self.vectorization_manager:
            return self.vectorization_manager.normalize_features(data, method)
        else:
            return self._normalize_features_standard(data, method)

    @memory_optimized(level=MemoryOptimizationLevel.BALANCED) if HARDWARE_OPTIMIZATION_AVAILABLE else lambda x: x
    def optimize_clustering_data(self, data: np.ndarray, labels: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Optimize data for clustering with comprehensive validation and preprocessing."""
        try:
            tprint_info("Starting comprehensive data optimization for clustering")
            
            # Data quality analysis
            if isinstance(data, np.ndarray):
                nan_analysis = analyze_nan_values_detailed(data)
                tprint_structured("Data Quality Analysis", nan_analysis)
                
                # Clean data if needed
                if nan_analysis['total_nans'] > 0:
                    tprint_warning(f"Found {nan_analysis['total_nans']} NaN values, cleaning data")
                    data = safe_fillna(data, method='median')
            
            # Validate data consistency
            if self.data_leakage_detector:
                leakage_score = self.data_leakage_detector.detect_leakage(data)
                if leakage_score > 0.1:  # Threshold for data leakage
                    tprint_warning(f"Potential data leakage detected: {leakage_score:.3f}")
            
            # Check for lookahead bias
            if self.lookahead_bias_detector:
                bias_score = self.lookahead_bias_detector.detect_bias(data)
                if bias_score > 0.05:  # Threshold for lookahead bias
                    tprint_warning(f"Potential lookahead bias detected: {bias_score:.3f}")
            
            # Optimize data types and memory usage
            if self.enable_hardware_optimization:
                data = optimize_dataframe_dtypes(pd.DataFrame(data)) if hasattr(data, 'dtype') else data
                data = optimize_memory(data)
            
            # Normalize features
            data = self.normalize_features(data, method='robust')
            
            # Validate final data
            math_validate_numeric_array(data, "optimized_clustering_data")
            
            tprint_success("Data optimization completed successfully")
            return data, labels
            
        except Exception as e:
            tprint_error(f"Failed to optimize clustering data: {e}")
            self.logger.error(f"Failed to optimize clustering data: {e}")
            return data, labels

    @performance_tracked(log_performance=True, track_memory=True) if HARDWARE_OPTIMIZATION_AVAILABLE else lambda x: x
    def enhanced_knn_graph(self, data: np.ndarray, k: int = 5, distance_metric: str = 'euclidean',
                          use_vectorization: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced kNN graph building with vectorization and hardware optimization."""
        try:
            tprint_info(f"Building enhanced kNN graph with k={k}, metric={distance_metric}")
            
            # Use vectorization manager if available
            if use_vectorization and self.vectorization_manager:
                return self.vectorization_manager.build_knn_graph(data, k, distance_metric)
            else:
                return self.build_knn_graph(data, k, distance_metric)
                
        except Exception as e:
            tprint_error(f"Failed to build enhanced kNN graph: {e}")
            return self.build_knn_graph(data, k, distance_metric)

    def calculate_enhanced_cluster_metrics(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Calculate enhanced cluster metrics with comprehensive analysis."""
        try:
            tprint_info("Calculating enhanced cluster metrics")
            
            # Basic cluster statistics
            basic_stats = self.calculate_cluster_statistics(data, labels)
            
            # Enhanced metrics using ML utilities
            enhanced_metrics = {
                'basic_statistics': basic_stats,
                'silhouette_score': None,
                'calinski_harabasz_score': None,
                'davies_bouldin_score': None,
                'data_quality_score': None,
                'optimization_recommendations': []
            }
            
            # Calculate silhouette score if possible
            try:
                from sklearn.metrics import silhouette_score
                enhanced_metrics['silhouette_score'] = silhouette_score(data, labels)
            except ImportError:
                tprint_warning("scikit-learn not available for silhouette score")
            except Exception as e:
                tprint_warning(f"Failed to calculate silhouette score: {e}")
            
            # Calculate data quality score
            if self.data_manager:
                enhanced_metrics['data_quality_score'] = calculate_data_quality_score(data)
            
            # Generate optimization recommendations
            if self.hardware_manager:
                recommendations = self.hardware_manager.get_optimization_recommendations(data)
                enhanced_metrics['optimization_recommendations'] = recommendations
            
            tprint_success("Enhanced cluster metrics calculated successfully")
            return enhanced_metrics
            
        except Exception as e:
            tprint_error(f"Failed to calculate enhanced cluster metrics: {e}")
            return {'basic_statistics': self.calculate_cluster_statistics(data, labels)}

    def save_clustering_artifacts(self, artifacts: Dict[str, Any], step_name: str) -> bool:
        """Save clustering artifacts using the artifact manager."""
        try:
            if not self.artifact_manager:
                tprint_warning("Artifact manager not available, skipping artifact save")
                return False
            
            tprint_info(f"Saving clustering artifacts for step: {step_name}")
            
            # Save artifacts with metadata
            metadata = {
                'step_name': step_name,
                'timestamp': get_current_datetime(),
                'hardware_optimization_enabled': self.enable_hardware_optimization,
                'ml_optimization_enabled': self.enable_ml_optimization
            }
            
            success = self.artifact_manager.save_artifacts(artifacts, step_name, metadata)
            
            if success:
                tprint_success(f"Successfully saved artifacts for step: {step_name}")
            else:
                tprint_warning(f"Failed to save artifacts for step: {step_name}")
            
            return success
            
        except Exception as e:
            tprint_error(f"Failed to save clustering artifacts: {e}")
            return False

    def load_clustering_artifacts(self, step_name: str) -> Optional[Dict[str, Any]]:
        """Load clustering artifacts using the artifact manager."""
        try:
            if not self.artifact_manager:
                tprint_warning("Artifact manager not available, skipping artifact load")
                return None
            
            tprint_info(f"Loading clustering artifacts for step: {step_name}")
            
            artifacts = self.artifact_manager.load_artifacts(step_name)
            
            if artifacts:
                tprint_success(f"Successfully loaded artifacts for step: {step_name}")
            else:
                tprint_warning(f"No artifacts found for step: {step_name}")
            
            return artifacts
            
        except Exception as e:
            tprint_error(f"Failed to load clustering artifacts: {e}")
            return None
