"""
Basic clustering metrics calculation for HMM clustering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import logging
from dataclasses import dataclass

try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_enhanced_matrix_operations,
        get_batch_matrix_processor
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False

try:
    from src.utils.hardware import (
        get_hardware_accelerator,
        get_memory_manager,
        get_performance_monitor
    )
    HARDWARE_ACCELERATION_AVAILABLE = True
except ImportError:
    HARDWARE_ACCELERATION_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BasicMetricsResult:
    """Result of basic metrics calculation."""
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    n_clusters: int
    n_valid_points: int
    n_noise_points: int
    average_cluster_cv: float
    cluster_size_cv: float
    execution_time: float
    matrix_ops_used: bool
    hardware_acceleration_used: bool


class BasicClusteringMetrics:
    """Basic clustering metrics calculator with hardware acceleration."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the metrics calculator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize hardware acceleration if available
        self.hardware_accelerator = None
        self.memory_manager = None
        self.performance_monitor = None
        
        if HARDWARE_ACCELERATION_AVAILABLE:
            try:
                self.hardware_accelerator = get_hardware_accelerator()
                self.memory_manager = get_memory_manager()
                self.performance_monitor = get_performance_monitor()
                self.logger.info("✅ Hardware acceleration initialized for metrics")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware acceleration not available for metrics: {e}")
        
        # Initialize matrix operations if available
        self.matrix_ops = None
        self.vectorized_core = None
        self.enhanced_ops = None
        self.batch_processor = None
        
        if MATRIX_OPERATIONS_AVAILABLE:
            try:
                self.matrix_ops = get_unified_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                self.enhanced_ops = get_enhanced_matrix_operations()
                self.batch_processor = get_batch_matrix_processor()
                self.logger.info("✅ Matrix operations initialized for metrics")
            except Exception as e:
                self.logger.warning(f"⚠️ Matrix operations not available for metrics: {e}")

    def calculate_basic_metrics(self, features: np.ndarray, labels: np.ndarray) -> BasicMetricsResult:
        """Calculate basic clustering metrics.

        Args:
            features: Feature matrix
            labels: Cluster labels

        Returns:
            BasicMetricsResult with calculated metrics
        """
        import time
        start_time = time.time()
        
        try:
            # Monitor performance
            if self.performance_monitor:
                self.performance_monitor.start_monitoring("basic_metrics_calculation")
            
            # Filter out noise points for metrics calculation
            valid_mask = labels != -1
            valid_features = features[valid_mask]
            valid_labels = labels[valid_mask]
            
            if len(valid_labels) == 0:
                return self._create_empty_result(start_time)
            
            unique_labels = np.unique(valid_labels)
            n_clusters = len(unique_labels)
            
            if n_clusters < 2:
                return self._create_single_cluster_result(start_time, len(valid_features), len(features) - len(valid_features))
            
            # Calculate metrics using matrix operations if available
            if self.matrix_ops is not None:
                metrics = self._calculate_metrics_with_matrix_ops(valid_features, valid_labels, n_clusters)
            else:
                metrics = self._calculate_metrics_standard(valid_features, valid_labels, n_clusters)
            
            # Calculate cluster size coefficient of variation
            cluster_size_cv = self._calculate_cluster_size_cv(valid_labels)
            
            # Calculate average cluster coefficient of variation
            average_cluster_cv = self._calculate_average_cluster_cv(valid_features, valid_labels)
            
            # Stop performance monitoring
            perf_metrics = {}
            if self.performance_monitor:
                perf_metrics = self.performance_monitor.stop_monitoring("basic_metrics_calculation")
            
            execution_time = time.time() - start_time
            
            return BasicMetricsResult(
                silhouette=metrics['silhouette'],
                davies_bouldin=metrics['davies_bouldin'],
                calinski_harabasz=metrics['calinski_harabasz'],
                n_clusters=n_clusters,
                n_valid_points=len(valid_features),
                n_noise_points=len(features) - len(valid_features),
                average_cluster_cv=average_cluster_cv,
                cluster_size_cv=cluster_size_cv,
                execution_time=execution_time,
                matrix_ops_used=self.matrix_ops is not None,
                hardware_acceleration_used=self.hardware_accelerator is not None
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Basic metrics calculation failed: {e}")
            return self._create_error_result(str(e), execution_time)

    def _calculate_metrics_with_matrix_ops(self, features: np.ndarray, labels: np.ndarray, n_clusters: int) -> Dict[str, float]:
        """Calculate metrics using matrix operations.

        Args:
            features: Valid feature matrix
            labels: Valid cluster labels
            n_clusters: Number of clusters

        Returns:
            Dictionary of calculated metrics
        """
        try:
            # Use matrix operations for distance calculations
            if hasattr(self.matrix_ops, 'silhouette_score_gpu'):
                silhouette = self.matrix_ops.silhouette_score_gpu(features, labels)
            else:
                silhouette = silhouette_score(features, labels)
            
            if hasattr(self.matrix_ops, 'davies_bouldin_score_gpu'):
                davies_bouldin = self.matrix_ops.davies_bouldin_score_gpu(features, labels)
            else:
                davies_bouldin = davies_bouldin_score(features, labels)
            
            if hasattr(self.matrix_ops, 'calinski_harabasz_score_gpu'):
                calinski_harabasz = self.matrix_ops.calinski_harabasz_score_gpu(features, labels)
            else:
                calinski_harabasz = calinski_harabasz_score(features, labels)
            
            return {
                'silhouette': float(silhouette),
                'davies_bouldin': float(davies_bouldin),
                'calinski_harabasz': float(calinski_harabasz)
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Matrix operations metrics calculation failed: {e}")
            return self._calculate_metrics_standard(features, labels, n_clusters)

    def _calculate_metrics_standard(self, features: np.ndarray, labels: np.ndarray, n_clusters: int) -> Dict[str, float]:
        """Calculate metrics using standard sklearn implementations.

        Args:
            features: Valid feature matrix
            labels: Valid cluster labels
            n_clusters: Number of clusters

        Returns:
            Dictionary of calculated metrics
        """
        try:
            silhouette = silhouette_score(features, labels)
            davies_bouldin = davies_bouldin_score(features, labels)
            calinski_harabasz = calinski_harabasz_score(features, labels)
            
            return {
                'silhouette': float(silhouette),
                'davies_bouldin': float(davies_bouldin),
                'calinski_harabasz': float(calinski_harabasz)
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Standard metrics calculation failed: {e}")
            return {
                'silhouette': 0.0,
                'davies_bouldin': 10.0,
                'calinski_harabasz': 0.0
            }

    def _calculate_cluster_size_cv(self, labels: np.ndarray) -> float:
        """Calculate coefficient of variation of cluster sizes.

        Args:
            labels: Cluster labels

        Returns:
            Coefficient of variation of cluster sizes
        """
        try:
            unique_labels = np.unique(labels)
            cluster_sizes = [np.sum(labels == label) for label in unique_labels]
            
            if len(cluster_sizes) < 2:
                return 0.0
            
            mean_size = np.mean(cluster_sizes)
            std_size = np.std(cluster_sizes)
            
            if mean_size == 0:
                return 0.0
            
            return float(std_size / mean_size)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cluster size CV calculation failed: {e}")
            return 0.0

    def _calculate_average_cluster_cv(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate average coefficient of variation within clusters.

        Args:
            features: Feature matrix
            labels: Cluster labels

        Returns:
            Average cluster coefficient of variation
        """
        try:
            unique_labels = np.unique(labels)
            cluster_cvs = []
            
            for label in unique_labels:
                cluster_features = features[labels == label]
                if len(cluster_features) > 1:
                    cluster_cv = self._calculate_feature_cv(cluster_features)
                    cluster_cvs.append(cluster_cv)
            
            return float(np.mean(cluster_cvs)) if cluster_cvs else 0.0
            
        except Exception as e:
            self.logger.warning(f"⚠️ Average cluster CV calculation failed: {e}")
            return 0.0

    def _calculate_feature_cv(self, features: np.ndarray) -> float:
        """Calculate coefficient of variation for a cluster's features.

        Args:
            features: Cluster features

        Returns:
            Coefficient of variation
        """
        try:
            if len(features) < 2:
                return 0.0
            
            # Calculate CV for each feature dimension
            feature_cvs = []
            for i in range(features.shape[1]):
                feature_values = features[:, i]
                mean_val = np.mean(feature_values)
                std_val = np.std(feature_values)
                
                if mean_val != 0:
                    cv = std_val / abs(mean_val)
                    feature_cvs.append(cv)
            
            return float(np.mean(feature_cvs)) if feature_cvs else 0.0
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature CV calculation failed: {e}")
            return 0.0

    def _create_empty_result(self, start_time: float) -> BasicMetricsResult:
        """Create empty result for no valid clusters.

        Args:
            start_time: Start time for execution time calculation

        Returns:
            Empty BasicMetricsResult
        """
        import time
        execution_time = time.time() - start_time
        
        return BasicMetricsResult(
            silhouette=0.0,
            davies_bouldin=10.0,
            calinski_harabasz=0.0,
            n_clusters=0,
            n_valid_points=0,
            n_noise_points=0,
            average_cluster_cv=0.0,
            cluster_size_cv=0.0,
            execution_time=execution_time,
            matrix_ops_used=self.matrix_ops is not None,
            hardware_acceleration_used=self.hardware_accelerator is not None
        )

    def _create_single_cluster_result(self, start_time: float, n_valid_points: int, n_noise_points: int) -> BasicMetricsResult:
        """Create result for single cluster case.

        Args:
            start_time: Start time for execution time calculation
            n_valid_points: Number of valid points
            n_noise_points: Number of noise points

        Returns:
            Single cluster BasicMetricsResult
        """
        import time
        execution_time = time.time() - start_time
        
        return BasicMetricsResult(
            silhouette=0.0,
            davies_bouldin=10.0,
            calinski_harabasz=0.0,
            n_clusters=1,
            n_valid_points=n_valid_points,
            n_noise_points=n_noise_points,
            average_cluster_cv=0.0,
            cluster_size_cv=0.0,
            execution_time=execution_time,
            matrix_ops_used=self.matrix_ops is not None,
            hardware_acceleration_used=self.hardware_accelerator is not None
        )

    def _create_error_result(self, error_message: str, execution_time: float) -> BasicMetricsResult:
        """Create error result.

        Args:
            error_message: Error message
            execution_time: Execution time

        Returns:
            Error BasicMetricsResult
        """
        return BasicMetricsResult(
            silhouette=0.0,
            davies_bouldin=10.0,
            calinski_harabasz=0.0,
            n_clusters=0,
            n_valid_points=0,
            n_noise_points=0,
            average_cluster_cv=0.0,
            cluster_size_cv=0.0,
            execution_time=execution_time,
            matrix_ops_used=self.matrix_ops is not None,
            hardware_acceleration_used=self.hardware_accelerator is not None
        )

    def batch_calculate_metrics(self, features_list: List[np.ndarray], 
                              labels_list: List[np.ndarray]) -> List[BasicMetricsResult]:
        """Calculate metrics for multiple clustering results in batch.

        Args:
            features_list: List of feature matrices
            labels_list: List of cluster label arrays

        Returns:
            List of BasicMetricsResult objects
        """
        results = []
        
        try:
            if self.batch_processor is not None:
                # Use batch processing if available
                results = self._batch_calculate_with_processor(features_list, labels_list)
            else:
                # Standard batch processing
                for features, labels in zip(features_list, labels_list):
                    result = self.calculate_basic_metrics(features, labels)
                    results.append(result)
            
            self.logger.info(f"✅ Batch calculated metrics for {len(results)} clustering results")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Batch metrics calculation failed: {e}")
            return results

    def _batch_calculate_with_processor(self, features_list: List[np.ndarray], 
                                      labels_list: List[np.ndarray]) -> List[BasicMetricsResult]:
        """Calculate metrics using batch processor.

        Args:
            features_list: List of feature matrices
            labels_list: List of cluster label arrays

        Returns:
            List of BasicMetricsResult objects
        """
        try:
            # Use batch processor for efficient calculation
            batch_results = self.batch_processor.calculate_basic_metrics_batch(
                features_list, labels_list
            )
            
            # Convert to BasicMetricsResult objects
            results = []
            for i, (features, labels) in enumerate(zip(features_list, labels_list)):
                if i < len(batch_results):
                    batch_result = batch_results[i]
                    result = BasicMetricsResult(
                        silhouette=batch_result.get('silhouette', 0.0),
                        davies_bouldin=batch_result.get('davies_bouldin', 10.0),
                        calinski_harabasz=batch_result.get('calinski_harabasz', 0.0),
                        n_clusters=batch_result.get('n_clusters', 0),
                        n_valid_points=batch_result.get('n_valid_points', 0),
                        n_noise_points=batch_result.get('n_noise_points', 0),
                        average_cluster_cv=batch_result.get('average_cluster_cv', 0.0),
                        cluster_size_cv=batch_result.get('cluster_size_cv', 0.0),
                        execution_time=batch_result.get('execution_time', 0.0),
                        matrix_ops_used=True,
                        hardware_acceleration_used=True
                    )
                    results.append(result)
                else:
                    # Fallback to individual calculation
                    result = self.calculate_basic_metrics(features, labels)
                    results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.warning(f"⚠️ Batch processor failed: {e}")
            # Fallback to standard batch processing
            results = []
            for features, labels in zip(features_list, labels_list):
                result = self.calculate_basic_metrics(features, labels)
                results.append(result)
            return results