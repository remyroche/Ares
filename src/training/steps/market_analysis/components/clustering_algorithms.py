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

    def fit(self, features: np.ndarray) -> 'BaseClusteringAlgorithm':
        """
        Fit the clustering algorithm to the data.
        
        Args:
            features: Input feature matrix of shape (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        # Default implementation - subclasses can override
        self.fit_predict(features)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            features: Input feature matrix of shape (n_samples, n_features)
            
        Returns:
            Cluster labels
        """
        # Default implementation - subclasses should override
        result = self.fit_predict(features)
        return result.labels

    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """
        Get cluster centers if available.
        
        Returns:
            Cluster centers array or None if not available
        """
        # Default implementation - subclasses should override
        return None

    def get_model_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        # Default implementation - subclasses should override
        return {}

    def validate_input(self, features: np.ndarray) -> bool:
        """
        Validate input features.
        
        Args:
            features: Input feature matrix
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not isinstance(features, np.ndarray):
                tprint_error("Features must be a numpy array")
                return False
            
            if features.ndim != 2:
                tprint_error("Features must be 2-dimensional")
                return False
            
            if features.shape[0] < 2:
                tprint_error("Need at least 2 samples for clustering")
                return False
            
            if features.shape[1] < 1:
                tprint_error("Need at least 1 feature for clustering")
                return False
            
            if not np.isfinite(features).all():
                tprint_warning("Non-finite values found in features")
                return False
            
            return True
            
        except Exception as e:
            tprint_error(f"Input validation failed: {e}")
            return False

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
            # Validate input
            if not self.validate_input(features):
                raise ValueError("Invalid input features for GMM clustering")

            with memory_checkpoint("gmm_clustering", self.memory_manager):
                # Preprocess features
                features_scaled = self._preprocess_features(features)

                # Validate preprocessed features
                if features_scaled.shape[0] < self.config.n_regimes:
                    tprint_warning(f"Not enough samples ({features_scaled.shape[0]}) for {self.config.n_regimes} clusters, reducing to {features_scaled.shape[0] - 1}")
                    n_components = max(2, features_scaled.shape[0] - 1)
                else:
                    n_components = self.config.n_regimes

                # Initialize GMM with error handling
                try:
                    self.model = GaussianMixture(
                        n_components=n_components,
                        random_state=42,
                        max_iter=100,
                        tol=1e-6
                    )
                except Exception as e:
                    tprint_error(f"Failed to initialize GMM: {e}")
                    raise

                # Fit model with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        self.model.fit(features_scaled)
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        tprint_warning(f"GMM fit attempt {attempt + 1} failed: {e}, retrying...")
                        # Reduce components and try again
                        n_components = max(2, n_components - 1)
                        self.model = GaussianMixture(
                            n_components=n_components,
                            random_state=42 + attempt,
                            max_iter=100,
                            tol=1e-6
                        )

                # Predict labels
                labels = self.model.predict(features_scaled)

                # Calculate metrics
                metrics = self._calculate_metrics(features_scaled, labels)

                # Create result
                result = ClusteringResult(
                    labels=labels,
                    n_clusters=n_components,
                    algorithm="gaussian_mixture",
                    metrics=metrics,
                    metadata={
                        'converged': self.model.converged_,
                        'n_iter': self.model.n_iter_,
                        'aic': self.model.aic(features_scaled),
                        'bic': self.model.bic(features_scaled),
                        'original_n_regimes': self.config.n_regimes,
                        'actual_n_components': n_components
                    },
                    execution_time=time.time() - start_time
                )

                tprint_success(f"GMM clustering completed: {result.n_clusters} clusters")
                return result

        except Exception as e:
            tprint_error(f"GMM clustering failed: {e}")
            # Return error result instead of raising
            return ClusteringResult(
                labels=np.zeros(len(features), dtype=int),
                n_clusters=1,
                algorithm="gaussian_mixture",
                metrics={'error': str(e)},
                metadata={'error': str(e), 'failed': True},
                execution_time=time.time() - start_time
            )

    def fit(self, features: np.ndarray) -> 'GaussianMixtureClustering':
        """Fit GMM to the data."""
        if not self.validate_input(features):
            raise ValueError("Invalid input features")
        
        features_scaled = self._preprocess_features(features)
        
        self.model = GaussianMixture(
            n_components=self.config.n_regimes,
            random_state=42,
            max_iter=100,
            tol=1e-6
        )
        
        self.model.fit(features_scaled)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        if not self.validate_input(features):
            raise ValueError("Invalid input features")
        
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)

    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """Get cluster centers (means)."""
        if self.model is None:
            return None
        return self.model.means_

    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        if self.model is None:
            return {}
        
        return {
            'n_components': self.model.n_components,
            'converged': self.model.converged_,
            'n_iter': self.model.n_iter_,
            'means': self.model.means_.tolist() if self.model.means_ is not None else None,
            'covariances': self.model.covariances_.tolist() if self.model.covariances_ is not None else None
        }

    def validate_model(self) -> bool:
        """Validate the fitted GMM model."""
        if self.model is None:
            tprint_error("GMM model is not fitted")
            return False
        
        try:
            # Check if model converged
            if not self.model.converged_:
                tprint_warning("GMM model did not converge")
                return False
            
            # Check if model has required attributes
            if not hasattr(self.model, 'means_'):
                tprint_error("GMM model missing means_ attribute")
                return False
            
            if not hasattr(self.model, 'covariances_'):
                tprint_error("GMM model missing covariances_ attribute")
                return False
            
            # Check means validity
            if self.model.means_ is None:
                tprint_error("GMM means are None")
                return False
            
            # Check covariances validity
            if self.model.covariances_ is None:
                tprint_error("GMM covariances are None")
                return False
            
            return True
            
        except Exception as e:
            tprint_error(f"GMM model validation failed: {e}")
            return False

    def get_cluster_quality(self) -> Dict[str, float]:
        """Get GMM cluster quality metrics."""
        if not self.validate_model():
            return {'error': 'Invalid GMM model'}
        
        try:
            # Get AIC and BIC if model is fitted
            aic = getattr(self.model, 'aic', lambda x: float('inf'))(self.model.means_)
            bic = getattr(self.model, 'bic', lambda x: float('inf'))(self.model.means_)
            
            # Calculate log-likelihood
            log_likelihood = getattr(self.model, 'score', lambda x: 0.0)(self.model.means_)
            
            return {
                'aic': aic,
                'bic': bic,
                'log_likelihood': log_likelihood,
                'n_components': self.model.n_components,
                'converged': self.model.converged_,
                'n_iter': self.model.n_iter_
            }
            
        except Exception as e:
            tprint_error(f"GMM quality calculation failed: {e}")
            return {'error': str(e)}

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
            # Validate input
            if not self.validate_input(features):
                raise ValueError("Invalid input features for K-Means clustering")

            with memory_checkpoint("kmeans_clustering", self.memory_manager):
                # Preprocess features
                features_scaled = self._preprocess_features(features)

                # Validate preprocessed features
                if features_scaled.shape[0] < self.config.n_regimes:
                    tprint_warning(f"Not enough samples ({features_scaled.shape[0]}) for {self.config.n_regimes} clusters, reducing to {features_scaled.shape[0] - 1}")
                    n_clusters = max(2, features_scaled.shape[0] - 1)
                else:
                    n_clusters = self.config.n_regimes

                # Initialize K-Means with error handling
                try:
                    self.model = KMeans(
                        n_clusters=n_clusters,
                        random_state=42,
                        max_iter=100,
                        tol=1e-6,
                        n_init=10
                    )
                except Exception as e:
                    tprint_error(f"Failed to initialize K-Means: {e}")
                    raise

                # Fit model with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        self.model.fit(features_scaled)
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        tprint_warning(f"K-Means fit attempt {attempt + 1} failed: {e}, retrying...")
                        # Reduce clusters and try again
                        n_clusters = max(2, n_clusters - 1)
                        self.model = KMeans(
                            n_clusters=n_clusters,
                            random_state=42 + attempt,
                            max_iter=100,
                            tol=1e-6,
                            n_init=10
                        )

                # Predict labels
                labels = self.model.labels_

                # Calculate metrics
                metrics = self._calculate_metrics(features_scaled, labels)

                # Create result
                result = ClusteringResult(
                    labels=labels,
                    n_clusters=n_clusters,
                    algorithm="kmeans",
                    metrics=metrics,
                    metadata={
                        'inertia': self.model.inertia_,
                        'n_iter': self.model.n_iter_,
                        'centers': self.model.cluster_centers_.tolist(),
                        'original_n_regimes': self.config.n_regimes,
                        'actual_n_clusters': n_clusters
                    },
                    execution_time=time.time() - start_time
                )

                tprint_success(f"K-Means clustering completed: {result.n_clusters} clusters")
                return result

        except Exception as e:
            tprint_error(f"K-Means clustering failed: {e}")
            # Return error result instead of raising
            return ClusteringResult(
                labels=np.zeros(len(features), dtype=int),
                n_clusters=1,
                algorithm="kmeans",
                metrics={'error': str(e)},
                metadata={'error': str(e), 'failed': True},
                execution_time=time.time() - start_time
            )

    def fit(self, features: np.ndarray) -> 'KMeansClustering':
        """Fit K-Means to the data."""
        if not self.validate_input(features):
            raise ValueError("Invalid input features")
        
        features_scaled = self._preprocess_features(features)
        
        self.model = KMeans(
            n_clusters=self.config.n_regimes,
            random_state=42,
            max_iter=100,
            tol=1e-6,
            n_init=10
        )
        
        self.model.fit(features_scaled)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        if not self.validate_input(features):
            raise ValueError("Invalid input features")
        
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)

    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """Get cluster centers."""
        if self.model is None:
            return None
        return self.model.cluster_centers_

    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        if self.model is None:
            return {}
        
        return {
            'n_clusters': self.model.n_clusters,
            'inertia': self.model.inertia_,
            'n_iter': self.model.n_iter_,
            'centers': self.model.cluster_centers_.tolist() if self.model.cluster_centers_ is not None else None
        }

    def validate_model(self) -> bool:
        """Validate the fitted model."""
        if self.model is None:
            tprint_error("Model is not fitted")
            return False
        
        try:
            # Check if model has required attributes
            if not hasattr(self.model, 'cluster_centers_'):
                tprint_error("Model missing cluster_centers_ attribute")
                return False
            
            if not hasattr(self.model, 'labels_'):
                tprint_error("Model missing labels_ attribute")
                return False
            
            # Check cluster centers validity
            if self.model.cluster_centers_ is None:
                tprint_error("Cluster centers are None")
                return False
            
            # Check for empty clusters
            unique_labels = np.unique(self.model.labels_)
            if len(unique_labels) != self.model.n_clusters:
                tprint_warning(f"Expected {self.model.n_clusters} clusters, got {len(unique_labels)}")
            
            return True
            
        except Exception as e:
            tprint_error(f"Model validation failed: {e}")
            return False

    def get_cluster_quality(self) -> Dict[str, float]:
        """Get cluster quality metrics."""
        if not self.validate_model():
            return {'error': 'Invalid model'}
        
        try:
            # Calculate inertia (within-cluster sum of squares)
            inertia = self.model.inertia_
            
            # Calculate silhouette score if possible
            try:
                from sklearn.metrics import silhouette_score
                if hasattr(self.model, 'labels_') and len(np.unique(self.model.labels_)) > 1:
                    silhouette = silhouette_score(self.model.labels_.reshape(-1, 1), self.model.labels_)
                else:
                    silhouette = 0.0
            except:
                silhouette = 0.0
            
            return {
                'inertia': inertia,
                'silhouette_score': silhouette,
                'n_clusters': self.model.n_clusters,
                'converged': True
            }
            
        except Exception as e:
            tprint_error(f"Quality calculation failed: {e}")
            return {'error': str(e)}

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
            # Validate input
            if not self.validate_input(features):
                raise ValueError("Invalid input features for Agglomerative clustering")

            with memory_checkpoint("agglomerative_clustering", self.memory_manager):
                # Preprocess features
                features_scaled = self._preprocess_features(features)

                # Validate preprocessed features
                if features_scaled.shape[0] < self.config.n_regimes:
                    tprint_warning(f"Not enough samples ({features_scaled.shape[0]}) for {self.config.n_regimes} clusters, reducing to {features_scaled.shape[0] - 1}")
                    n_clusters = max(2, features_scaled.shape[0] - 1)
                else:
                    n_clusters = self.config.n_regimes

                # Initialize Agglomerative clustering with error handling
                try:
                    self.model = AgglomerativeClustering(
                        n_clusters=n_clusters,
                        linkage='ward'
                    )
                except Exception as e:
                    tprint_error(f"Failed to initialize Agglomerative clustering: {e}")
                    raise

                # Fit and predict with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        labels = self.model.fit_predict(features_scaled)
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        tprint_warning(f"Agglomerative fit attempt {attempt + 1} failed: {e}, retrying...")
                        # Reduce clusters and try again
                        n_clusters = max(2, n_clusters - 1)
                        self.model = AgglomerativeClustering(
                            n_clusters=n_clusters,
                            linkage='ward'
                        )

                # Calculate metrics
                metrics = self._calculate_metrics(features_scaled, labels)

                # Create result
                result = ClusteringResult(
                    labels=labels,
                    n_clusters=n_clusters,
                    algorithm="agglomerative",
                    metrics=metrics,
                    metadata={
                        'linkage': 'ward',
                        'n_leaves': self.model.n_leaves_,
                        'n_components': self.model.n_components_,
                        'original_n_regimes': self.config.n_regimes,
                        'actual_n_clusters': n_clusters
                    },
                    execution_time=time.time() - start_time
                )

                tprint_success(f"Agglomerative clustering completed: {result.n_clusters} clusters")
                return result

        except Exception as e:
            tprint_error(f"Agglomerative clustering failed: {e}")
            # Return error result instead of raising
            return ClusteringResult(
                labels=np.zeros(len(features), dtype=int),
                n_clusters=1,
                algorithm="agglomerative",
                metrics={'error': str(e)},
                metadata={'error': str(e), 'failed': True},
                execution_time=time.time() - start_time
            )

    def fit(self, features: np.ndarray) -> 'AgglomerativeClusteringAlgorithm':
        """Fit Agglomerative clustering to the data."""
        if not self.validate_input(features):
            raise ValueError("Invalid input features")
        
        features_scaled = self._preprocess_features(features)
        
        self.model = AgglomerativeClustering(
            n_clusters=self.config.n_regimes,
            linkage='ward'
        )
        
        self.model.fit(features_scaled)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        if not self.validate_input(features):
            raise ValueError("Invalid input features")
        
        features_scaled = self.scaler.transform(features)
        return self.model.fit_predict(features_scaled)

    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """Get cluster centers (not available for Agglomerative)."""
        # Agglomerative clustering doesn't have explicit cluster centers
        return None

    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        if self.model is None:
            return {}
        
        return {
            'n_clusters': self.model.n_clusters,
            'linkage': self.model.linkage,
            'n_leaves': self.model.n_leaves_,
            'n_components': self.model.n_components_
        }

    def validate_model(self) -> bool:
        """Validate the fitted Agglomerative model."""
        if self.model is None:
            tprint_error("Agglomerative model is not fitted")
            return False
        
        try:
            # Check if model has required attributes
            if not hasattr(self.model, 'n_leaves_'):
                tprint_error("Agglomerative model missing n_leaves_ attribute")
                return False
            
            if not hasattr(self.model, 'n_components_'):
                tprint_error("Agglomerative model missing n_components_ attribute")
                return False
            
            # Check n_leaves validity
            if self.model.n_leaves_ <= 0:
                tprint_error("Invalid n_leaves_ value")
                return False
            
            return True
            
        except Exception as e:
            tprint_error(f"Agglomerative model validation failed: {e}")
            return False

    def get_cluster_quality(self) -> Dict[str, float]:
        """Get Agglomerative cluster quality metrics."""
        if not self.validate_model():
            return {'error': 'Invalid Agglomerative model'}
        
        try:
            return {
                'n_clusters': self.model.n_clusters,
                'n_leaves': self.model.n_leaves_,
                'n_components': self.model.n_components_,
                'linkage': self.model.linkage
            }
            
        except Exception as e:
            tprint_error(f"Agglomerative quality calculation failed: {e}")
            return {'error': str(e)}

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
            # Validate input
            if not self.validate_input(features):
                raise ValueError("Invalid input features for Adaptive clustering")

            with memory_checkpoint("adaptive_clustering", self.memory_manager):
                # Preprocess features
                features_scaled = self._preprocess_features(features)

                # Select best algorithm based on data characteristics
                best_algorithm = self._select_best_algorithm(features_scaled)
                self.selected_algorithm = best_algorithm

                # Try the selected algorithm first
                try:
                    result = self.algorithms[best_algorithm].fit_predict(features_scaled)
                    
                    # Check if the result is valid
                    if result.metadata.get('failed', False):
                        raise ValueError(f"Selected algorithm {best_algorithm} failed")
                    
                except Exception as e:
                    tprint_warning(f"Selected algorithm {best_algorithm} failed: {e}, trying alternatives...")
                    
                    # Try alternative algorithms
                    alternative_algorithms = [alg for alg in self.algorithms.keys() if alg != best_algorithm]
                    result = None
                    
                    for alt_algorithm in alternative_algorithms:
                        try:
                            tprint_info(f"Trying alternative algorithm: {alt_algorithm}")
                            result = self.algorithms[alt_algorithm].fit_predict(features_scaled)
                            
                            if not result.metadata.get('failed', False):
                                self.selected_algorithm = alt_algorithm
                                tprint_info(f"Successfully used alternative algorithm: {alt_algorithm}")
                                break
                                
                        except Exception as alt_e:
                            tprint_warning(f"Alternative algorithm {alt_algorithm} also failed: {alt_e}")
                            continue
                    
                    # If all algorithms failed, create error result
                    if result is None or result.metadata.get('failed', False):
                        raise ValueError("All clustering algorithms failed")

                # Update metadata
                result.algorithm = f"adaptive_{self.selected_algorithm}"
                result.metadata['selected_algorithm'] = self.selected_algorithm
                result.metadata['algorithm_selection_criteria'] = self._get_selection_criteria(features_scaled)
                result.metadata['original_selection'] = best_algorithm
                result.metadata['fallback_used'] = self.selected_algorithm != best_algorithm
                result.execution_time = time.time() - start_time

                tprint_success(f"Adaptive clustering completed using {self.selected_algorithm}")
                return result

        except Exception as e:
            tprint_error(f"Adaptive clustering failed: {e}")
            # Return error result instead of raising
            return ClusteringResult(
                labels=np.zeros(len(features), dtype=int),
                n_clusters=1,
                algorithm="adaptive_clustering",
                metrics={'error': str(e)},
                metadata={'error': str(e), 'failed': True, 'selected_algorithm': self.selected_algorithm},
                execution_time=time.time() - start_time
            )

    def fit(self, features: np.ndarray) -> 'AdaptiveClusteringAlgorithm':
        """Fit adaptive clustering to the data."""
        if not self.validate_input(features):
            raise ValueError("Invalid input features")
        
        features_scaled = self._preprocess_features(features)
        
        # Select best algorithm
        best_algorithm = self._select_best_algorithm(features_scaled)
        self.selected_algorithm = best_algorithm
        
        # Fit selected algorithm
        self.algorithms[best_algorithm].fit(features_scaled)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        if self.selected_algorithm is None:
            raise ValueError("Model must be fitted before prediction")
        
        if not self.validate_input(features):
            raise ValueError("Invalid input features")
        
        return self.algorithms[self.selected_algorithm].predict(features)

    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """Get cluster centers from selected algorithm."""
        if self.selected_algorithm is None:
            return None
        
        return self.algorithms[self.selected_algorithm].get_cluster_centers()

    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        if self.selected_algorithm is None:
            return {}
        
        params = self.algorithms[self.selected_algorithm].get_model_params()
        params['selected_algorithm'] = self.selected_algorithm
        return params

    def validate_model(self) -> bool:
        """Validate the adaptive clustering model."""
        if self.selected_algorithm is None:
            tprint_error("No algorithm selected for adaptive clustering")
            return False
        
        try:
            # Validate the selected algorithm's model
            return self.algorithms[self.selected_algorithm].validate_model()
            
        except Exception as e:
            tprint_error(f"Adaptive model validation failed: {e}")
            return False

    def get_cluster_quality(self) -> Dict[str, float]:
        """Get adaptive clustering quality metrics."""
        if self.selected_algorithm is None:
            return {'error': 'No algorithm selected'}
        
        try:
            quality = self.algorithms[self.selected_algorithm].get_cluster_quality()
            quality['selected_algorithm'] = self.selected_algorithm
            return quality
            
        except Exception as e:
            tprint_error(f"Adaptive quality calculation failed: {e}")
            return {'error': str(e)}

    def get_algorithm_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all available algorithms."""
        try:
            performance = {}
            for alg_name, alg_instance in self.algorithms.items():
                try:
                    if hasattr(alg_instance, 'get_cluster_quality'):
                        quality = alg_instance.get_cluster_quality()
                        performance[alg_name] = quality
                    else:
                        performance[alg_name] = {'error': 'No quality method available'}
                except Exception as e:
                    performance[alg_name] = {'error': str(e)}
            
            return performance
            
        except Exception as e:
            tprint_error(f"Algorithm performance calculation failed: {e}")
            return {'error': str(e)}

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
