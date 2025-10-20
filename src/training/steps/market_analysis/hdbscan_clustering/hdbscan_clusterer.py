"""
HDBSCAN Clusterer

This module provides a legacy HDBSCAN clustering implementation for
regime discovery, serving as a fallback when the optimized version
is not available.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import time
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
import warnings

logger = logging.getLogger(__name__)

@dataclass
class HDBSCANClustererConfig:
    """Configuration for HDBSCAN clustering."""
    # Core HDBSCAN parameters
    min_cluster_size: int = 20
    min_samples: int = 5
    cluster_selection_epsilon: float = 0.0
    cluster_selection_method: str = 'eom'  # 'eom', 'leaf'
    
    # Distance and metric
    metric: str = 'euclidean'
    alpha: float = 1.0
    algorithm: str = 'auto'  # 'auto', 'ball_tree', 'kd_tree', 'brute'
    
    # Memory and performance
    memory: Optional[str] = None
    n_jobs: int = 1
    
    # Validation parameters
    min_silhouette_score: float = 0.1
    max_clusters: int = 20
    min_clusters: int = 2
    
    # Noise handling
    handle_noise: bool = True
    noise_strategy: str = 'keep'  # 'keep', 'knn_assign', 'causal_smooth'
    
    # Parameter optimization
    enable_optimization: bool = False
    optimization_metric: str = 'silhouette'  # 'silhouette', 'calinski_harabasz', 'davies_bouldin'
    param_search_space: Dict[str, List] = None
    
    # Validation
    validate_input: bool = True
    min_samples_for_clustering: int = 10

class HDBSCANClusterer:
    """
    Legacy HDBSCAN clusterer for regime discovery.
    
    Provides a fallback implementation when the optimized version
    is not available, with basic parameter optimization and validation.
    """
    
    def __init__(self, config: Optional[HDBSCANClustererConfig] = None):
        """
        Initialize HDBSCAN clusterer.
        
        Args:
            config: Configuration for clustering
        """
        self.config = config or HDBSCANClustererConfig()
        self.clusterer = None
        self.clustering_stats = {}
        self.best_params = None
        self.best_score = -np.inf
        
        # Set default parameter search space
        if self.config.param_search_space is None:
            self.config.param_search_space = {
                'min_cluster_size': [10, 15, 20, 25, 30],
                'min_samples': [3, 5, 7, 10],
                'cluster_selection_epsilon': [0.0, 0.1, 0.2, 0.3]
            }
    
    def cluster_data(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Cluster data using HDBSCAN.
        
        Args:
            features_df: Input features DataFrame
            
        Returns:
            Tuple of (cluster_labels, clustering_info)
        """
        try:
            logger.info("🔍 Starting HDBSCAN clustering...")
            
            # Validate input
            if self.config.validate_input:
                features_df = self._validate_input(features_df)
            
            # Convert to numpy array
            features = features_df.values
            
            # Check minimum samples
            if len(features) < self.config.min_samples_for_clustering:
                logger.warning(f"⚠️ Insufficient samples for clustering: {len(features)} < {self.config.min_samples_for_clustering}")
                return np.zeros(len(features), dtype=int), {'error': 'insufficient_samples'}
            
            # Optimize parameters if enabled
            if self.config.enable_optimization:
                best_params = self._optimize_parameters(features)
            else:
                best_params = self._get_default_params()
            
            # Perform clustering
            cluster_labels, clustering_info = self._perform_clustering(features, best_params)
            
            # Handle noise if enabled
            if self.config.handle_noise:
                cluster_labels = self._handle_noise_points(cluster_labels, features)
            
            # Validate clustering results
            self._validate_clustering(cluster_labels, features)
            
            # Calculate clustering statistics
            self.clustering_stats = self._calculate_clustering_stats(cluster_labels, features, clustering_info)
            
            logger.info(f"✅ HDBSCAN clustering completed. Found {len(np.unique(cluster_labels[cluster_labels != -1]))} clusters")
            
            return cluster_labels, clustering_info
            
        except Exception as e:
            logger.error(f"❌ HDBSCAN clustering failed: {e}")
            # Return single cluster as fallback
            return np.zeros(len(features_df), dtype=int), {'error': str(e)}
    
    def _validate_input(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Validate input features."""
        try:
            # Check for NaN values
            if features_df.isnull().any().any():
                logger.warning("⚠️ Found NaN values, filling with 0")
                features_df = features_df.fillna(0)
            
            # Check for infinite values
            if np.isinf(features_df.values).sum() > 0:
                logger.warning("⚠️ Found infinite values, clipping")
                features_df = features_df.replace([np.inf, -np.inf], [np.finfo(np.float64).max, np.finfo(np.float64).min])
            
            # Check for constant columns
            constant_cols = features_df.columns[features_df.nunique() <= 1]
            if len(constant_cols) > 0:
                logger.warning(f"⚠️ Removing constant columns: {constant_cols.tolist()}")
                features_df = features_df.drop(columns=constant_cols)
            
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Input validation failed: {e}")
            return features_df
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default HDBSCAN parameters."""
        return {
            'min_cluster_size': self.config.min_cluster_size,
            'min_samples': self.config.min_samples,
            'cluster_selection_epsilon': self.config.cluster_selection_epsilon,
            'cluster_selection_method': self.config.cluster_selection_method,
            'metric': self.config.metric,
            'alpha': self.config.alpha,
            'algorithm': self.config.algorithm,
            'memory': self.config.memory,
            'n_jobs': self.config.n_jobs
        }
    
    def _optimize_parameters(self, features: np.ndarray) -> Dict[str, Any]:
        """Optimize HDBSCAN parameters using grid search."""
        try:
            logger.info("🔧 Optimizing HDBSCAN parameters...")
            
            best_params = None
            best_score = -np.inf
            
            # Generate parameter combinations
            param_combinations = self._generate_param_combinations()
            
            logger.info(f"Testing {len(param_combinations)} parameter combinations...")
            
            for i, params in enumerate(param_combinations):
                try:
                    # Perform clustering with current parameters
                    cluster_labels, _ = self._perform_clustering(features, params)
                    
                    # Calculate validation score
                    score = self._calculate_validation_score(cluster_labels, features)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Completed {i + 1}/{len(param_combinations)} combinations. Best score: {best_score:.3f}")
                
                except Exception as e:
                    logger.debug(f"Parameter combination failed: {e}")
                    continue
            
            if best_params is None:
                logger.warning("⚠️ Parameter optimization failed, using default parameters")
                best_params = self._get_default_params()
            else:
                logger.info(f"✅ Best parameters found: {best_params} (score: {best_score:.3f})")
            
            self.best_params = best_params
            self.best_score = best_score
            
            return best_params
            
        except Exception as e:
            logger.error(f"❌ Parameter optimization failed: {e}")
            return self._get_default_params()
    
    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate parameter combinations for grid search."""
        try:
            import itertools
            
            # Get parameter names and values
            param_names = list(self.config.param_search_space.keys())
            param_values = list(self.config.param_search_space.values())
            
            # Generate all combinations
            combinations = list(itertools.product(*param_values))
            
            # Convert to list of dictionaries
            param_combinations = []
            for combo in combinations:
                params = dict(zip(param_names, combo))
                # Add fixed parameters
                params.update({
                    'cluster_selection_method': self.config.cluster_selection_method,
                    'metric': self.config.metric,
                    'alpha': self.config.alpha,
                    'algorithm': self.config.algorithm,
                    'memory': self.config.memory,
                    'n_jobs': self.config.n_jobs
                })
                param_combinations.append(params)
            
            return param_combinations
            
        except Exception as e:
            logger.error(f"❌ Parameter combination generation failed: {e}")
            return [self._get_default_params()]
    
    def _perform_clustering(self, features: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform HDBSCAN clustering with given parameters."""
        try:
            # Import HDBSCAN
            try:
                import hdbscan
            except ImportError:
                logger.error("❌ HDBSCAN not available. Please install: pip install hdbscan")
                raise ImportError("HDBSCAN not available")
            
            # Create clusterer
            clusterer = hdbscan.HDBSCAN(**params)
            
            # Perform clustering
            start_time = time.time()
            cluster_labels = clusterer.fit_predict(features)
            clustering_time = time.time() - start_time
            
            # Create clustering info
            clustering_info = {
                'clusterer': clusterer,
                'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                'n_noise_points': list(cluster_labels).count(-1),
                'clustering_time': clustering_time,
                'cluster_persistence': getattr(clusterer, 'cluster_persistence_', None),
                'condensed_tree': getattr(clusterer, 'condensed_tree_', None),
                'mst': getattr(clusterer, 'mst_', None),
                'glosh_scores': getattr(clusterer, 'glosh_scores_', None),
                'cluster_centers': self._calculate_cluster_centers(features, cluster_labels),
                'cluster_sizes': self._calculate_cluster_sizes(cluster_labels)
            }
            
            return cluster_labels, clustering_info
            
        except Exception as e:
            logger.error(f"❌ HDBSCAN clustering failed: {e}")
            raise
    
    def _calculate_cluster_centers(self, features: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
        """Calculate cluster centers."""
        try:
            unique_labels = np.unique(cluster_labels)
            unique_labels = unique_labels[unique_labels != -1]  # Remove noise
            
            if len(unique_labels) == 0:
                return np.array([])
            
            centers = []
            for label in unique_labels:
                mask = cluster_labels == label
                if mask.sum() > 0:
                    center = np.mean(features[mask], axis=0)
                    centers.append(center)
            
            return np.array(centers) if centers else np.array([])
            
        except Exception as e:
            logger.debug(f"Cluster center calculation failed: {e}")
            return np.array([])
    
    def _calculate_cluster_sizes(self, cluster_labels: np.ndarray) -> Dict[int, int]:
        """Calculate cluster sizes."""
        try:
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            return dict(zip(unique_labels, counts))
        except Exception as e:
            logger.debug(f"Cluster size calculation failed: {e}")
            return {}
    
    def _calculate_validation_score(self, cluster_labels: np.ndarray, features: np.ndarray) -> float:
        """Calculate validation score for clustering."""
        try:
            # Remove noise points for validation
            valid_mask = cluster_labels != -1
            if valid_mask.sum() < 2:
                return -np.inf
            
            valid_labels = cluster_labels[valid_mask]
            valid_features = features[valid_mask]
            
            if len(set(valid_labels)) < 2:
                return -np.inf
            
            # Calculate score based on optimization metric
            if self.config.optimization_metric == 'silhouette':
                score = silhouette_score(valid_features, valid_labels)
            elif self.config.optimization_metric == 'calinski_harabasz':
                score = calinski_harabasz_score(valid_features, valid_labels)
            elif self.config.optimization_metric == 'davies_bouldin':
                score = -davies_bouldin_score(valid_features, valid_labels)  # Negative because lower is better
            else:
                score = silhouette_score(valid_features, valid_labels)
            
            return score
            
        except Exception as e:
            logger.debug(f"Validation score calculation failed: {e}")
            return -np.inf
    
    def _handle_noise_points(self, cluster_labels: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Handle noise points using specified strategy."""
        try:
            noise_mask = cluster_labels == -1
            n_noise = noise_mask.sum()
            
            if n_noise == 0:
                return cluster_labels
            
            logger.info(f"Handling {n_noise} noise points using {self.config.noise_strategy} strategy")
            
            if self.config.noise_strategy == 'keep':
                # Keep noise points as -1
                return cluster_labels
            
            elif self.config.noise_strategy == 'knn_assign':
                # Assign noise points to nearest cluster
                return self._assign_noise_to_nearest_cluster(cluster_labels, features, noise_mask)
            
            elif self.config.noise_strategy == 'causal_smooth':
                # Apply causal smoothing to noise points
                return self._causal_smooth_noise(cluster_labels, noise_mask)
            
            else:
                logger.warning(f"⚠️ Unknown noise strategy: {self.config.noise_strategy}")
                return cluster_labels
            
        except Exception as e:
            logger.error(f"❌ Noise handling failed: {e}")
            return cluster_labels
    
    def _assign_noise_to_nearest_cluster(self, cluster_labels: np.ndarray, features: np.ndarray, noise_mask: np.ndarray) -> np.ndarray:
        """Assign noise points to nearest cluster using KNN."""
        try:
            # Get non-noise points and their labels
            valid_mask = ~noise_mask
            valid_features = features[valid_mask]
            valid_labels = cluster_labels[valid_mask]
            
            if len(valid_features) == 0:
                return cluster_labels
            
            # Get noise points
            noise_features = features[noise_mask]
            
            # Find nearest neighbors
            knn = NearestNeighbors(n_neighbors=min(5, len(valid_features)), metric=self.config.metric)
            knn.fit(valid_features)
            
            # Find nearest neighbors for noise points
            distances, indices = knn.kneighbors(noise_features)
            
            # Assign to most common label among neighbors
            new_labels = cluster_labels.copy()
            for i, noise_idx in enumerate(np.where(noise_mask)[0]):
                neighbor_labels = valid_labels[indices[i]]
                unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
                most_common_label = unique_labels[np.argmax(counts)]
                new_labels[noise_idx] = most_common_label
            
            return new_labels
            
        except Exception as e:
            logger.error(f"❌ Noise assignment failed: {e}")
            return cluster_labels
    
    def _causal_smooth_noise(self, cluster_labels: np.ndarray, noise_mask: np.ndarray) -> np.ndarray:
        """Apply causal smoothing to noise points."""
        try:
            new_labels = cluster_labels.copy()
            
            # Find noise points
            noise_indices = np.where(noise_mask)[0]
            
            for noise_idx in noise_indices:
                # Look at previous non-noise points
                prev_mask = (np.arange(len(cluster_labels)) < noise_idx) & (~noise_mask)
                if prev_mask.sum() > 0:
                    prev_labels = cluster_labels[prev_mask]
                    # Use most recent non-noise label
                    new_labels[noise_idx] = prev_labels[-1]
                else:
                    # Look at next non-noise points
                    next_mask = (np.arange(len(cluster_labels)) > noise_idx) & (~noise_mask)
                    if next_mask.sum() > 0:
                        next_labels = cluster_labels[next_mask]
                        # Use first future non-noise label
                        new_labels[noise_idx] = next_labels[0]
                    else:
                        # No non-noise points found, keep as noise
                        new_labels[noise_idx] = -1
            
            return new_labels
            
        except Exception as e:
            logger.error(f"❌ Causal smoothing failed: {e}")
            return cluster_labels
    
    def _validate_clustering(self, cluster_labels: np.ndarray, features: np.ndarray):
        """Validate clustering results."""
        try:
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise_points = list(cluster_labels).count(-1)
            
            # Check minimum clusters
            if n_clusters < self.config.min_clusters:
                logger.warning(f"⚠️ Too few clusters: {n_clusters} < {self.config.min_clusters}")
            
            # Check maximum clusters
            if n_clusters > self.config.max_clusters:
                logger.warning(f"⚠️ Too many clusters: {n_clusters} > {self.config.max_clusters}")
            
            # Check silhouette score
            if n_clusters >= 2:
                try:
                    valid_mask = cluster_labels != -1
                    if valid_mask.sum() >= 2:
                        valid_labels = cluster_labels[valid_mask]
                        valid_features = features[valid_mask]
                        
                        if len(set(valid_labels)) >= 2:
                            silhouette = silhouette_score(valid_features, valid_labels)
                            if silhouette < self.config.min_silhouette_score:
                                logger.warning(f"⚠️ Low silhouette score: {silhouette:.3f} < {self.config.min_silhouette_score}")
                except Exception as e:
                    logger.debug(f"Silhouette score validation failed: {e}")
            
            logger.info(f"✅ Clustering validation completed. Clusters: {n_clusters}, Noise: {n_noise_points}")
            
        except Exception as e:
            logger.error(f"❌ Clustering validation failed: {e}")
    
    def _calculate_clustering_stats(self, cluster_labels: np.ndarray, features: np.ndarray, clustering_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate clustering statistics."""
        try:
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise_points = list(cluster_labels).count(-1)
            n_samples = len(cluster_labels)
            
            stats = {
                'n_clusters': n_clusters,
                'n_noise_points': n_noise_points,
                'n_samples': n_samples,
                'noise_ratio': n_noise_points / n_samples if n_samples > 0 else 0,
                'clustering_time': clustering_info.get('clustering_time', 0),
                'cluster_sizes': clustering_info.get('cluster_sizes', {}),
                'best_params': self.best_params,
                'best_score': self.best_score
            }
            
            # Calculate silhouette score if possible
            if n_clusters >= 2:
                try:
                    valid_mask = cluster_labels != -1
                    if valid_mask.sum() >= 2:
                        valid_labels = cluster_labels[valid_mask]
                        valid_features = features[valid_mask]
                        
                        if len(set(valid_labels)) >= 2:
                            stats['silhouette_score'] = silhouette_score(valid_features, valid_labels)
                        else:
                            stats['silhouette_score'] = 0.0
                    else:
                        stats['silhouette_score'] = 0.0
                except Exception as e:
                    logger.debug(f"Silhouette score calculation failed: {e}")
                    stats['silhouette_score'] = 0.0
            else:
                stats['silhouette_score'] = 0.0
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Clustering stats calculation failed: {e}")
            return {'error': str(e)}
    
    def approximate_predict_with_fallback(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Predict cluster labels and probabilities for new data points.
        
        This method provides approximate prediction capabilities for HDBSCAN,
        which doesn't have a direct predict method.
        
        Args:
            features: Feature matrix for prediction (n_samples, n_features)
            
        Returns:
            Tuple of (labels, probabilities, method_used)
        """
        try:
            if self.clusterer is None:
                logger.warning("⚠️ No trained clusterer available, using random assignment")
                return self._random_fallback(features)
            
            # Try to use the clusterer's approximate_predict method if available
            if hasattr(self.clusterer, 'approximate_predict'):
                try:
                    labels, probabilities = self.clusterer.approximate_predict(features)
                    return labels, probabilities, "hdbscan_approximate_predict"
                except Exception as e:
                    logger.debug(f"HDBSCAN approximate_predict failed: {e}")
            
            # Fallback to distance-based assignment
            return self._distance_based_prediction(features)
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return self._random_fallback(features)
    
    def _distance_based_prediction(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """Predict using distance-based assignment."""
        try:
            # Get cluster centers from training
            if hasattr(self, 'cluster_centers') and self.cluster_centers is not None:
                centers = self.cluster_centers
            else:
                logger.warning("⚠️ No cluster centers available, using random assignment")
                return self._random_fallback(features)
            
            if len(centers) == 0:
                return self._random_fallback(features)
            
            # Calculate distances to cluster centers
            distances = np.sqrt(((features[:, np.newaxis] - centers[np.newaxis, :]) ** 2).sum(axis=2))
            
            # Assign to closest cluster
            labels = np.argmin(distances, axis=1)
            
            # Calculate probabilities based on distance
            min_distances = np.min(distances, axis=1, keepdims=True)
            probabilities = min_distances / (distances + 1e-10)
            probabilities = np.max(probabilities, axis=1)
            
            return labels, probabilities, "distance_based"
            
        except Exception as e:
            logger.error(f"❌ Distance-based prediction failed: {e}")
            return self._random_fallback(features)
    
    def _random_fallback(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """Fallback to random assignment when all other methods fail."""
        try:
            n_samples = len(features)
            
            # Random labels (assuming 2-5 clusters)
            n_clusters = np.random.randint(2, 6)
            labels = np.random.randint(0, n_clusters, n_samples)
            
            # Random probabilities
            probabilities = np.random.uniform(0.1, 0.9, n_samples)
            
            return labels, probabilities, "random_fallback"
            
        except Exception as e:
            logger.error(f"❌ Random fallback failed: {e}")
            # Ultimate fallback
            n_samples = len(features)
            return np.zeros(n_samples), np.ones(n_samples), "ultimate_fallback"
    
    def get_clustering_stats(self) -> Dict[str, Any]:
        """Get clustering statistics."""
        return self.clustering_stats.copy()
    
    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get best parameters from optimization."""
        return self.best_params
    
    def get_best_score(self) -> float:
        """Get best score from optimization."""
        return self.best_score