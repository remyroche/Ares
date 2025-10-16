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
    get_m1_memory_optimizer, get_m1_cpu_optimizer, is_m1_available, is_mps_available
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
except ImportError:
    hw_is_m1_available = lambda: False
    hw_is_mps_available = lambda: False
    hw_get_m1_memory_optimizer = lambda: None
    hw_get_m1_cpu_optimizer = lambda: None

# Import ML common utilities (removed unused imports that cause linter errors)

logger = logging.getLogger(__name__)


class ClusteringUtils:
    """Utility functions for clustering operations and calculations."""

    def __init__(self):
        """Initialize clustering utilities."""
        self.logger = logger

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
        """Normalize features using specified method."""
        try:
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

        except Exception as e:
            self.logger.error(f"Failed to normalize features: {e}")
            return data
