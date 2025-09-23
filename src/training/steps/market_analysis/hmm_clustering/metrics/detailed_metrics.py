"""
Detailed clustering metrics calculation for HMM clustering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
import logging
from dataclasses import dataclass
import time

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

from .basic_metrics import BasicClusteringMetrics, BasicMetricsResult

logger = logging.getLogger(__name__)


@dataclass
class DetailedMetricsResult:
    """Result of detailed metrics calculation."""
    # Basic metrics
    basic_metrics: BasicMetricsResult
    
    # Advanced metrics
    adjusted_rand_index: float
    normalized_mutual_info: float
    homogeneity_score: float
    completeness_score: float
    v_measure_score: float
    
    # Cluster quality metrics
    cluster_separation: float
    cluster_compactness: float
    cluster_density: float
    cluster_connectivity: float
    
    # Size distribution metrics
    size_distribution: Dict[str, Any]
    size_balance_score: float
    size_entropy: float
    
    # Separation metrics
    inter_cluster_distance: float
    intra_cluster_distance: float
    separation_ratio: float
    
    # Execution metadata
    execution_time: float
    matrix_ops_used: bool
    hardware_acceleration_used: bool


class DetailedClusteringMetrics:
    """Detailed clustering metrics calculator with comprehensive analysis."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the detailed metrics calculator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize basic metrics calculator
        self.basic_metrics_calc = BasicClusteringMetrics(config)
        
        # Initialize hardware acceleration if available
        self.hardware_accelerator = None
        self.memory_manager = None
        self.performance_monitor = None
        
        if HARDWARE_ACCELERATION_AVAILABLE:
            try:
                self.hardware_accelerator = get_hardware_accelerator()
                self.memory_manager = get_memory_manager()
                self.performance_monitor = get_performance_monitor()
                self.logger.info("✅ Hardware acceleration initialized for detailed metrics")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware acceleration not available for detailed metrics: {e}")
        
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
                self.logger.info("✅ Matrix operations initialized for detailed metrics")
            except Exception as e:
                self.logger.warning(f"⚠️ Matrix operations not available for detailed metrics: {e}")

    def calculate_detailed_metrics(self, features: np.ndarray, labels: np.ndarray, 
                                 reference_labels: Optional[np.ndarray] = None) -> DetailedMetricsResult:
        """Calculate detailed clustering metrics.

        Args:
            features: Feature matrix
            labels: Cluster labels
            reference_labels: Reference labels for comparison (optional)

        Returns:
            DetailedMetricsResult with comprehensive metrics
        """
        start_time = time.time()
        
        try:
            # Monitor performance
            if self.performance_monitor:
                self.performance_monitor.start_monitoring("detailed_metrics_calculation")
            
            # Calculate basic metrics first
            basic_metrics = self.basic_metrics_calc.calculate_basic_metrics(features, labels)
            
            # Filter out noise points for detailed metrics
            valid_mask = labels != -1
            valid_features = features[valid_mask]
            valid_labels = labels[valid_mask]
            
            if len(valid_labels) == 0:
                return self._create_empty_detailed_result(basic_metrics, start_time)
            
            unique_labels = np.unique(valid_labels)
            n_clusters = len(unique_labels)
            
            if n_clusters < 2:
                return self._create_single_cluster_detailed_result(basic_metrics, start_time)
            
            # Calculate advanced metrics
            advanced_metrics = self._calculate_advanced_metrics(valid_features, valid_labels, reference_labels)
            
            # Calculate cluster quality metrics
            quality_metrics = self._calculate_cluster_quality_metrics(valid_features, valid_labels)
            
            # Calculate size distribution metrics
            size_metrics = self._calculate_size_distribution_metrics(valid_labels)
            
            # Calculate separation metrics
            separation_metrics = self._calculate_separation_metrics(valid_features, valid_labels)
            
            # Stop performance monitoring
            perf_metrics = {}
            if self.performance_monitor:
                perf_metrics = self.performance_monitor.stop_monitoring("detailed_metrics_calculation")
            
            execution_time = time.time() - start_time
            
            return DetailedMetricsResult(
                basic_metrics=basic_metrics,
                adjusted_rand_index=advanced_metrics['adjusted_rand_index'],
                normalized_mutual_info=advanced_metrics['normalized_mutual_info'],
                homogeneity_score=advanced_metrics['homogeneity_score'],
                completeness_score=advanced_metrics['completeness_score'],
                v_measure_score=advanced_metrics['v_measure_score'],
                cluster_separation=quality_metrics['cluster_separation'],
                cluster_compactness=quality_metrics['cluster_compactness'],
                cluster_density=quality_metrics['cluster_density'],
                cluster_connectivity=quality_metrics['cluster_connectivity'],
                size_distribution=size_metrics['size_distribution'],
                size_balance_score=size_metrics['size_balance_score'],
                size_entropy=size_metrics['size_entropy'],
                inter_cluster_distance=separation_metrics['inter_cluster_distance'],
                intra_cluster_distance=separation_metrics['intra_cluster_distance'],
                separation_ratio=separation_metrics['separation_ratio'],
                execution_time=execution_time,
                matrix_ops_used=self.matrix_ops is not None,
                hardware_acceleration_used=self.hardware_accelerator is not None
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Detailed metrics calculation failed: {e}")
            return self._create_error_detailed_result(str(e), execution_time)

    def _calculate_advanced_metrics(self, features: np.ndarray, labels: np.ndarray, 
                                  reference_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate advanced clustering metrics.

        Args:
            features: Valid feature matrix
            labels: Valid cluster labels
            reference_labels: Reference labels for comparison

        Returns:
            Dictionary of advanced metrics
        """
        try:
            advanced_metrics = {}
            
            # Calculate metrics that require reference labels
            if reference_labels is not None:
                # Filter reference labels to match valid features
                valid_ref_labels = reference_labels[reference_labels != -1]
                
                if len(valid_ref_labels) == len(labels):
                    try:
                        advanced_metrics['adjusted_rand_index'] = float(adjusted_rand_score(valid_ref_labels, labels))
                    except Exception:
                        advanced_metrics['adjusted_rand_index'] = 0.0
                    
                    try:
                        advanced_metrics['normalized_mutual_info'] = float(normalized_mutual_info_score(valid_ref_labels, labels))
                    except Exception:
                        advanced_metrics['normalized_mutual_info'] = 0.0
                else:
                    advanced_metrics['adjusted_rand_index'] = 0.0
                    advanced_metrics['normalized_mutual_info'] = 0.0
            else:
                advanced_metrics['adjusted_rand_index'] = 0.0
                advanced_metrics['normalized_mutual_info'] = 0.0
            
            # Calculate homogeneity, completeness, and V-measure
            try:
                from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
                
                if reference_labels is not None and len(valid_ref_labels) == len(labels):
                    advanced_metrics['homogeneity_score'] = float(homogeneity_score(valid_ref_labels, labels))
                    advanced_metrics['completeness_score'] = float(completeness_score(valid_ref_labels, labels))
                    advanced_metrics['v_measure_score'] = float(v_measure_score(valid_ref_labels, labels))
                else:
                    advanced_metrics['homogeneity_score'] = 0.0
                    advanced_metrics['completeness_score'] = 0.0
                    advanced_metrics['v_measure_score'] = 0.0
                    
            except Exception as e:
                self.logger.warning(f"⚠️ V-measure calculation failed: {e}")
                advanced_metrics['homogeneity_score'] = 0.0
                advanced_metrics['completeness_score'] = 0.0
                advanced_metrics['v_measure_score'] = 0.0
            
            return advanced_metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Advanced metrics calculation failed: {e}")
            return {
                'adjusted_rand_index': 0.0,
                'normalized_mutual_info': 0.0,
                'homogeneity_score': 0.0,
                'completeness_score': 0.0,
                'v_measure_score': 0.0
            }

    def _calculate_cluster_quality_metrics(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate cluster quality metrics.

        Args:
            features: Valid feature matrix
            labels: Valid cluster labels

        Returns:
            Dictionary of cluster quality metrics
        """
        try:
            unique_labels = np.unique(labels)
            
            # Calculate cluster centroids
            centroids = {}
            for label in unique_labels:
                cluster_features = features[labels == label]
                centroids[label] = np.mean(cluster_features, axis=0)
            
            # Calculate cluster separation (average distance between centroids)
            cluster_separation = self._calculate_cluster_separation(centroids)
            
            # Calculate cluster compactness (average within-cluster distance)
            cluster_compactness = self._calculate_cluster_compactness(features, labels, centroids)
            
            # Calculate cluster density
            cluster_density = self._calculate_cluster_density(features, labels)
            
            # Calculate cluster connectivity
            cluster_connectivity = self._calculate_cluster_connectivity(features, labels)
            
            return {
                'cluster_separation': cluster_separation,
                'cluster_compactness': cluster_compactness,
                'cluster_density': cluster_density,
                'cluster_connectivity': cluster_connectivity
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cluster quality metrics calculation failed: {e}")
            return {
                'cluster_separation': 0.0,
                'cluster_compactness': 0.0,
                'cluster_density': 0.0,
                'cluster_connectivity': 0.0
            }

    def _calculate_cluster_separation(self, centroids: Dict[int, np.ndarray]) -> float:
        """Calculate average separation between cluster centroids.

        Args:
            centroids: Dictionary of cluster centroids

        Returns:
            Average cluster separation
        """
        try:
            if len(centroids) < 2:
                return 0.0
            
            centroid_list = list(centroids.values())
            total_distance = 0.0
            count = 0
            
            for i in range(len(centroid_list)):
                for j in range(i + 1, len(centroid_list)):
                    distance = np.linalg.norm(centroid_list[i] - centroid_list[j])
                    total_distance += distance
                    count += 1
            
            return float(total_distance / count) if count > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cluster separation calculation failed: {e}")
            return 0.0

    def _calculate_cluster_compactness(self, features: np.ndarray, labels: np.ndarray, 
                                     centroids: Dict[int, np.ndarray]) -> float:
        """Calculate average cluster compactness.

        Args:
            features: Feature matrix
            labels: Cluster labels
            centroids: Cluster centroids

        Returns:
            Average cluster compactness
        """
        try:
            total_compactness = 0.0
            count = 0
            
            for label, centroid in centroids.items():
                cluster_mask = labels == label
                cluster_features = features[cluster_mask]
                
                if len(cluster_features) > 0:
                    distances = np.linalg.norm(cluster_features - centroid, axis=1)
                    cluster_compactness = np.mean(distances)
                    total_compactness += cluster_compactness
                    count += 1
            
            return float(total_compactness / count) if count > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cluster compactness calculation failed: {e}")
            return 0.0

    def _calculate_cluster_density(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate average cluster density.

        Args:
            features: Feature matrix
            labels: Cluster labels

        Returns:
            Average cluster density
        """
        try:
            unique_labels = np.unique(labels)
            total_density = 0.0
            
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_features = features[cluster_mask]
                
                if len(cluster_features) > 1:
                    # Calculate pairwise distances within cluster
                    distances = []
                    for i in range(len(cluster_features)):
                        for j in range(i + 1, len(cluster_features)):
                            dist = np.linalg.norm(cluster_features[i] - cluster_features[j])
                            distances.append(dist)
                    
                    if distances:
                        avg_distance = np.mean(distances)
                        # Density is inverse of average distance
                        density = 1.0 / (1.0 + avg_distance)
                        total_density += density
            
            return float(total_density / len(unique_labels)) if len(unique_labels) > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cluster density calculation failed: {e}")
            return 0.0

    def _calculate_cluster_connectivity(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate cluster connectivity.

        Args:
            features: Feature matrix
            labels: Cluster labels

        Returns:
            Cluster connectivity score
        """
        try:
            # Simplified connectivity calculation
            # This could be enhanced with graph-based connectivity measures
            
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 0.0
            
            # Calculate connectivity as the ratio of inter-cluster to intra-cluster distances
            intra_distances = []
            inter_distances = []
            
            for i, label_a in enumerate(unique_labels):
                mask_a = labels == label_a
                features_a = features[mask_a]
                
                # Intra-cluster distances
                if len(features_a) > 1:
                    for j in range(len(features_a)):
                        for k in range(j + 1, len(features_a)):
                            dist = np.linalg.norm(features_a[j] - features_a[k])
                            intra_distances.append(dist)
                
                # Inter-cluster distances
                for j, label_b in enumerate(unique_labels[i+1:], i+1):
                    mask_b = labels == label_b
                    features_b = features[mask_b]
                    
                    for feat_a in features_a:
                        for feat_b in features_b:
                            dist = np.linalg.norm(feat_a - feat_b)
                            inter_distances.append(dist)
            
            if not intra_distances or not inter_distances:
                return 0.0
            
            avg_intra = np.mean(intra_distances)
            avg_inter = np.mean(inter_distances)
            
            # Connectivity is higher when inter-cluster distances are much larger than intra-cluster
            connectivity = avg_inter / (avg_intra + 1e-8)
            
            return float(connectivity)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cluster connectivity calculation failed: {e}")
            return 0.0

    def _calculate_size_distribution_metrics(self, labels: np.ndarray) -> Dict[str, Any]:
        """Calculate size distribution metrics.

        Args:
            labels: Cluster labels

        Returns:
            Dictionary of size distribution metrics
        """
        try:
            unique_labels = np.unique(labels)
            cluster_sizes = [np.sum(labels == label) for label in unique_labels]
            
            size_distribution = {
                'cluster_sizes': cluster_sizes,
                'mean_size': float(np.mean(cluster_sizes)),
                'std_size': float(np.std(cluster_sizes)),
                'min_size': int(np.min(cluster_sizes)),
                'max_size': int(np.max(cluster_sizes)),
                'size_range': int(np.max(cluster_sizes) - np.min(cluster_sizes))
            }
            
            # Calculate size balance score (inverse of coefficient of variation)
            mean_size = np.mean(cluster_sizes)
            std_size = np.std(cluster_sizes)
            size_balance_score = 1.0 - (std_size / (mean_size + 1e-8))
            size_balance_score = max(0.0, min(1.0, size_balance_score))
            
            # Calculate size entropy
            total_points = len(labels)
            probabilities = [size / total_points for size in cluster_sizes]
            size_entropy = -sum(p * np.log2(p + 1e-8) for p in probabilities)
            
            return {
                'size_distribution': size_distribution,
                'size_balance_score': float(size_balance_score),
                'size_entropy': float(size_entropy)
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Size distribution metrics calculation failed: {e}")
            return {
                'size_distribution': {'cluster_sizes': [], 'mean_size': 0.0, 'std_size': 0.0},
                'size_balance_score': 0.0,
                'size_entropy': 0.0
            }

    def _calculate_separation_metrics(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate separation metrics.

        Args:
            features: Valid feature matrix
            labels: Valid cluster labels

        Returns:
            Dictionary of separation metrics
        """
        try:
            unique_labels = np.unique(labels)
            
            # Calculate centroids
            centroids = {}
            for label in unique_labels:
                cluster_features = features[labels == label]
                centroids[label] = np.mean(cluster_features, axis=0)
            
            # Calculate inter-cluster distance
            inter_cluster_distance = self._calculate_cluster_separation(centroids)
            
            # Calculate intra-cluster distance
            intra_cluster_distance = self._calculate_cluster_compactness(features, labels, centroids)
            
            # Calculate separation ratio
            separation_ratio = inter_cluster_distance / (intra_cluster_distance + 1e-8)
            
            return {
                'inter_cluster_distance': inter_cluster_distance,
                'intra_cluster_distance': intra_cluster_distance,
                'separation_ratio': separation_ratio
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Separation metrics calculation failed: {e}")
            return {
                'inter_cluster_distance': 0.0,
                'intra_cluster_distance': 0.0,
                'separation_ratio': 0.0
            }

    def _create_empty_detailed_result(self, basic_metrics: BasicMetricsResult, start_time: float) -> DetailedMetricsResult:
        """Create empty detailed result for no valid clusters.

        Args:
            basic_metrics: Basic metrics result
            start_time: Start time for execution time calculation

        Returns:
            Empty DetailedMetricsResult
        """
        execution_time = time.time() - start_time
        
        return DetailedMetricsResult(
            basic_metrics=basic_metrics,
            adjusted_rand_index=0.0,
            normalized_mutual_info=0.0,
            homogeneity_score=0.0,
            completeness_score=0.0,
            v_measure_score=0.0,
            cluster_separation=0.0,
            cluster_compactness=0.0,
            cluster_density=0.0,
            cluster_connectivity=0.0,
            size_distribution={'cluster_sizes': [], 'mean_size': 0.0, 'std_size': 0.0},
            size_balance_score=0.0,
            size_entropy=0.0,
            inter_cluster_distance=0.0,
            intra_cluster_distance=0.0,
            separation_ratio=0.0,
            execution_time=execution_time,
            matrix_ops_used=self.matrix_ops is not None,
            hardware_acceleration_used=self.hardware_accelerator is not None
        )

    def _create_single_cluster_detailed_result(self, basic_metrics: BasicMetricsResult, start_time: float) -> DetailedMetricsResult:
        """Create detailed result for single cluster case.

        Args:
            basic_metrics: Basic metrics result
            start_time: Start time for execution time calculation

        Returns:
            Single cluster DetailedMetricsResult
        """
        execution_time = time.time() - start_time
        
        return DetailedMetricsResult(
            basic_metrics=basic_metrics,
            adjusted_rand_index=0.0,
            normalized_mutual_info=0.0,
            homogeneity_score=0.0,
            completeness_score=0.0,
            v_measure_score=0.0,
            cluster_separation=0.0,
            cluster_compactness=0.0,
            cluster_density=0.0,
            cluster_connectivity=0.0,
            size_distribution={'cluster_sizes': [basic_metrics.n_valid_points], 'mean_size': float(basic_metrics.n_valid_points), 'std_size': 0.0},
            size_balance_score=1.0,
            size_entropy=0.0,
            inter_cluster_distance=0.0,
            intra_cluster_distance=0.0,
            separation_ratio=0.0,
            execution_time=execution_time,
            matrix_ops_used=self.matrix_ops is not None,
            hardware_acceleration_used=self.hardware_accelerator is not None
        )

    def _create_error_detailed_result(self, error_message: str, execution_time: float) -> DetailedMetricsResult:
        """Create error detailed result.

        Args:
            error_message: Error message
            execution_time: Execution time

        Returns:
            Error DetailedMetricsResult
        """
        # Create empty basic metrics for error case
        basic_metrics = BasicMetricsResult(
            silhouette=0.0,
            davies_bouldin=10.0,
            calinski_harabasz=0.0,
            n_clusters=0,
            n_valid_points=0,
            n_noise_points=0,
            average_cluster_cv=0.0,
            cluster_size_cv=0.0,
            execution_time=0.0,
            matrix_ops_used=False,
            hardware_acceleration_used=False
        )
        
        return DetailedMetricsResult(
            basic_metrics=basic_metrics,
            adjusted_rand_index=0.0,
            normalized_mutual_info=0.0,
            homogeneity_score=0.0,
            completeness_score=0.0,
            v_measure_score=0.0,
            cluster_separation=0.0,
            cluster_compactness=0.0,
            cluster_density=0.0,
            cluster_connectivity=0.0,
            size_distribution={'cluster_sizes': [], 'mean_size': 0.0, 'std_size': 0.0},
            size_balance_score=0.0,
            size_entropy=0.0,
            inter_cluster_distance=0.0,
            intra_cluster_distance=0.0,
            separation_ratio=0.0,
            execution_time=execution_time,
            matrix_ops_used=False,
            hardware_acceleration_used=False
        )