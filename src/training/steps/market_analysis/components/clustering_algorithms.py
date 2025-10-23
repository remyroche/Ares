"""
Clustering algorithms for NAS-TAS regime analysis.

This module provides specialized clustering algorithms with optimization
and validation for financial time series data.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import abstractmethod
import time

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

from src.utils.tprint import (
    tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_performance, tprint_structured
)

from .memory_manager import MemoryManager, memory_checkpoint
from .clustering_config import ClusteringConfig

@dataclass
class ClusteringResult:
    """Result of clustering operation."""
    labels: np.ndarray
    n_clusters: int
    algorithm: str
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'labels': self.labels.tolist(),
            'n_clusters': self.n_clusters,
            'algorithm': self.algorithm,
            'metrics': self.metrics,
            'metadata': self.metadata,
            'execution_time': self.execution_time
        }

class BaseClusteringAlgorithm:
    """Base class for clustering algorithms."""

    def __init__(self, config: ClusteringConfig, memory_manager: Optional[MemoryManager] = None):
        """Initialize clustering algorithm."""
        self.config = config
        self.memory_manager = memory_manager or MemoryManager()
        self.scaler = StandardScaler()

    @abstractmethod
    def fit_predict(self, features: np.ndarray) -> ClusteringResult:
        """
        Fit clustering algorithm and predict labels.
        
        This is an abstract method that must be implemented by subclasses.
        
        Args:
            features: Input feature matrix of shape (n_samples, n_features)
            
        Returns:
            ClusteringResult containing labels, metrics, and metadata
        """
        pass

    def _calculate_metrics(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering quality metrics."""
        try:
            metrics = {}

            # Basic metrics
            if len(np.unique(labels)) > 1:
                metrics['silhouette_score'] = silhouette_score(features, labels)
                metrics['davies_bouldin_score'] = davies_bouldin_score(features, labels)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(features, labels)
            else:
                metrics['silhouette_score'] = 0.0
                metrics['davies_bouldin_score'] = float('inf')
                metrics['calinski_harabasz_score'] = 0.0

            # Additional metrics
            metrics['n_clusters'] = len(np.unique(labels))
            metrics['n_samples'] = len(labels)

            return metrics

        except Exception as e:
            tprint_warning(f"Failed to calculate metrics: {e}")
            return {'error': str(e)}

    def _preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """Preprocess features for clustering."""
        try:
            # Validate features
            finite_mask = np.isfinite(features)
            if not finite_mask.all():
                tprint_warning("Non-finite values found in features, cleaning...")
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # Scale features
            if self.config.use_standardized_features:
                features_scaled = self.scaler.fit_transform(features)
            else:
                features_scaled = features.copy()

            # Optimize memory usage
            features_scaled = self.memory_manager.optimize_memory_usage(features_scaled)

            return features_scaled

        except Exception as e:
            tprint_error(f"Feature preprocessing failed: {e}")
            raise

class GaussianMixtureClustering(BaseClusteringAlgorithm):
    """Gaussian Mixture Model clustering algorithm."""

    def __init__(self, config: ClusteringConfig, memory_manager: Optional[MemoryManager] = None):
        """Initialize GMM clustering."""
        super().__init__(config, memory_manager)
        self.model = None

    def fit_predict(self, features: np.ndarray) -> ClusteringResult:
        """Fit GMM and predict labels."""
        start_time = time.time()

        try:
            with memory_checkpoint("gmm_clustering", self.memory_manager):
                # Preprocess features
                features_scaled = self._preprocess_features(features)

                # Initialize GMM
                self.model = GaussianMixture(
                    n_components=self.config.n_regimes,
                    random_state=42,
                    max_iter=100,
                    tol=1e-6
                )

                # Fit model
                self.model.fit(features_scaled)

                # Predict labels
                labels = self.model.predict(features_scaled)

                # Calculate metrics
                metrics = self._calculate_metrics(features_scaled, labels)

                # Create result
                result = ClusteringResult(
                    labels=labels,
                    n_clusters=self.config.n_regimes,
                    algorithm="gaussian_mixture",
                    metrics=metrics,
                    metadata={
                        'converged': self.model.converged_,
                        'n_iter': self.model.n_iter_,
                        'aic': self.model.aic(features_scaled),
                        'bic': self.model.bic(features_scaled)
                    },
                    execution_time=time.time() - start_time
                )

                tprint_success(f"GMM clustering completed: {result.n_clusters} clusters")
                return result

        except Exception as e:
            tprint_error(f"GMM clustering failed: {e}")
            raise

class KMeansClustering(BaseClusteringAlgorithm):
    """K-Means clustering algorithm."""

    def __init__(self, config: ClusteringConfig, memory_manager: Optional[MemoryManager] = None):
        """Initialize K-Means clustering."""
        super().__init__(config, memory_manager)
        self.model = None

    def fit_predict(self, features: np.ndarray) -> ClusteringResult:
        """Fit K-Means and predict labels."""
        start_time = time.time()

        try:
            with memory_checkpoint("kmeans_clustering", self.memory_manager):
                # Preprocess features
                features_scaled = self._preprocess_features(features)

                # Initialize K-Means
                self.model = KMeans(
                    n_clusters=self.config.n_regimes,
                    random_state=42,
                    max_iter=100,
                    tol=1e-6,
                    n_init=10
                )

                # Fit model
                self.model.fit(features_scaled)

                # Predict labels
                labels = self.model.labels_

                # Calculate metrics
                metrics = self._calculate_metrics(features_scaled, labels)

                # Create result
                result = ClusteringResult(
                    labels=labels,
                    n_clusters=self.config.n_regimes,
                    algorithm="kmeans",
                    metrics=metrics,
                    metadata={
                        'inertia': self.model.inertia_,
                        'n_iter': self.model.n_iter_,
                        'centers': self.model.cluster_centers_.tolist()
                    },
                    execution_time=time.time() - start_time
                )

                tprint_success(f"K-Means clustering completed: {result.n_clusters} clusters")
                return result

        except Exception as e:
            tprint_error(f"K-Means clustering failed: {e}")
            raise

class AgglomerativeClusteringAlgorithm(BaseClusteringAlgorithm):
    """Agglomerative clustering algorithm."""

    def __init__(self, config: ClusteringConfig, memory_manager: Optional[MemoryManager] = None):
        """Initialize Agglomerative clustering."""
        super().__init__(config, memory_manager)
        self.model = None

    def fit_predict(self, features: np.ndarray) -> ClusteringResult:
        """Fit Agglomerative clustering and predict labels."""
        start_time = time.time()

        try:
            with memory_checkpoint("agglomerative_clustering", self.memory_manager):
                # Preprocess features
                features_scaled = self._preprocess_features(features)

                # Initialize Agglomerative clustering
                self.model = AgglomerativeClustering(
                    n_clusters=self.config.n_regimes,
                    linkage='ward'
                )

                # Fit and predict
                labels = self.model.fit_predict(features_scaled)

                # Calculate metrics
                metrics = self._calculate_metrics(features_scaled, labels)

                # Create result
                result = ClusteringResult(
                    labels=labels,
                    n_clusters=self.config.n_regimes,
                    algorithm="agglomerative",
                    metrics=metrics,
                    metadata={
                        'linkage': 'ward',
                        'n_leaves': self.model.n_leaves_,
                        'n_components': self.model.n_components_
                    },
                    execution_time=time.time() - start_time
                )

                tprint_success(f"Agglomerative clustering completed: {result.n_clusters} clusters")
                return result

        except Exception as e:
            tprint_error(f"Agglomerative clustering failed: {e}")
            raise

class AdaptiveClusteringAlgorithm(BaseClusteringAlgorithm):
    """Adaptive clustering that selects the best algorithm based on data characteristics."""

    def __init__(self, config: ClusteringConfig, memory_manager: Optional[MemoryManager] = None):
        """Initialize adaptive clustering."""
        super().__init__(config, memory_manager)
        self.algorithms = {
            'gmm': GaussianMixtureClustering(config, memory_manager),
            'kmeans': KMeansClustering(config, memory_manager),
            'agglomerative': AgglomerativeClusteringAlgorithm(config, memory_manager)
        }
        self.selected_algorithm = None

    def fit_predict(self, features: np.ndarray) -> ClusteringResult:
        """Fit adaptive clustering and predict labels."""
        start_time = time.time()

        try:
            with memory_checkpoint("adaptive_clustering", self.memory_manager):
                # Preprocess features
                features_scaled = self._preprocess_features(features)

                # Select best algorithm based on data characteristics
                best_algorithm = self._select_best_algorithm(features_scaled)
                self.selected_algorithm = best_algorithm

                # Run selected algorithm
                result = self.algorithms[best_algorithm].fit_predict(features_scaled)

                # Update metadata
                result.algorithm = f"adaptive_{best_algorithm}"
                result.metadata['selected_algorithm'] = best_algorithm
                result.metadata['algorithm_selection_criteria'] = self._get_selection_criteria(features_scaled)
                result.execution_time = time.time() - start_time

                tprint_success(f"Adaptive clustering completed using {best_algorithm}")
                return result

        except Exception as e:
            tprint_error(f"Adaptive clustering failed: {e}")
            raise

    def _select_best_algorithm(self, features: np.ndarray) -> str:
        """Select the best algorithm based on data characteristics."""
        try:
            # Analyze data characteristics
            n_samples, n_features = features.shape
            data_density = n_samples / n_features

            # Selection criteria
            if data_density > 10 and n_features < 50:
                # High density, low dimensions -> GMM
                return 'gmm'
            elif n_samples > 1000:
                # Large dataset -> K-Means
                return 'kmeans'
            else:
                # Default -> Agglomerative
                return 'agglomerative'

        except Exception as e:
            tprint_warning(f"Algorithm selection failed: {e}, using GMM")
            return 'gmm'

    def _get_selection_criteria(self, features: np.ndarray) -> Dict[str, Any]:
        """Get algorithm selection criteria."""
        return {
            'n_samples': features.shape[0],
            'n_features': features.shape[1],
            'data_density': features.shape[0] / features.shape[1],
            'selected_algorithm': self.selected_algorithm
        }

class ClusteringAlgorithmFactory:
    """Factory for creating clustering algorithms."""

    @staticmethod
    def create_algorithm(
        algorithm_type: str,
        config: ClusteringConfig,
        memory_manager: Optional[MemoryManager] = None
    ) -> BaseClusteringAlgorithm:
        """Create clustering algorithm based on type."""
        algorithms = {
            'gaussian_mixture': GaussianMixtureClustering,
            'kmeans': KMeansClustering,
            'agglomerative': AgglomerativeClusteringAlgorithm,
            'adaptive_clustering': AdaptiveClusteringAlgorithm
        }

        if algorithm_type not in algorithms:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")

        return algorithms[algorithm_type](config, memory_manager)

    @staticmethod
    def get_available_algorithms() -> List[str]:
        """Get list of available algorithms."""
        return [
            'gaussian_mixture',
            'kmeans',
            'agglomerative',
            'adaptive_clustering'
        ]

def create_clustering_algorithm(
    algorithm_type: str,
    config: ClusteringConfig,
    memory_manager: Optional[MemoryManager] = None
) -> BaseClusteringAlgorithm:
    """Create clustering algorithm with error handling."""
    try:
        return ClusteringAlgorithmFactory.create_algorithm(algorithm_type, config, memory_manager)
    except Exception as e:
        tprint_error(f"Failed to create clustering algorithm: {e}")
        raise
