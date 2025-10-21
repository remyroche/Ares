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

# Import tprint utilities for extensive logging
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, 
    tprint_debug, tprint_performance, tprint_progress, tprint_timer,
    tprint_logged, tprint_data_preview, LogLevel
)

# Import enhanced hardware optimization tools
from src.utils.hardware import (
    smart_cache, auto_optimize, memory_efficient, performance_tracked,
    optimize_dataframe_default, optimize_numpy_array_default
)

# Import common operations and utilities
from src.utils.common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, 
    safe_numeric_operation, optimize_dataframe_memory
)
from src.utils.common_utilities import (
    safe_dataframe_operation as safe_df_op,
    validate_dataframe_columns as validate_df_cols
)
from src.utils.math_validation import (
    validate_finite, safe_divide, safe_log, safe_sqrt, safe_power,
    validate_array, validate_numeric_range, safe_statistical_operation
)

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
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def __init__(self, config: Optional[HDBSCANClustererConfig] = None):
        """
        Initialize HDBSCAN clusterer.
        
        Args:
            config: Configuration for clustering
        """
        tprint_info("🔧 Initializing HDBSCAN clusterer")
        start_time = time.perf_counter()
        
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
            tprint_debug(f"Set default parameter search space: {self.config.param_search_space}")
        
        init_time = time.perf_counter() - start_time
        tprint_success(f"✅ HDBSCAN clusterer initialized in {init_time:.3f}s")
    
    @smart_cache(ttl=3600)  # Cache clustering results for 1 hour
    @auto_optimize(optimize_inputs=True, optimize_outputs=True)
    @memory_efficient(memory_threshold_mb=200.0, auto_cleanup=True)
    @performance_tracked(log_performance=True, track_memory=True)
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
            
            # Data preview for clustering input
            tprint_data_preview(features_df, "clustering_input_features", max_rows=5, level="DEBUG")
            tprint_data_preview(features, "clustering_input_array", max_rows=5, level="DEBUG")
            
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
            
            # Data preview of clustering results
            tprint_data_preview(cluster_labels, "raw_cluster_labels", max_rows=10, level="INFO")
            tprint_data_preview(clustering_info, "clustering_info", level="DEBUG")
            
            # Handle noise if enabled
            if self.config.handle_noise:
                cluster_labels = self._handle_noise_points(cluster_labels, features)
                tprint_data_preview(cluster_labels, "post_noise_handling_labels", max_rows=10, level="DEBUG")
            
            # Validate clustering results
            self._validate_clustering(cluster_labels, features)
            
            # Calculate clustering statistics
            self.clustering_stats = self._calculate_clustering_stats(cluster_labels, features, clustering_info)
            
            # Final data preview
            tprint_data_preview(cluster_labels, "final_cluster_labels", max_rows=10, level="INFO")
            tprint_data_preview(self.clustering_stats, "clustering_stats", level="DEBUG")
            
            logger.info(f"✅ HDBSCAN clustering completed. Found {len(np.unique(cluster_labels[cluster_labels != -1]))} clusters")
            
            return cluster_labels, clustering_info
            
        except Exception as e:
            logger.error(f"❌ HDBSCAN clustering failed: {e}")
            # Return single cluster as fallback
            return np.zeros(len(features_df), dtype=int), {'error': str(e)}
    
    @tprint_logged(LogLevel.DEBUG, include_args=True, include_result=True)
    def _validate_input(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Validate input features with enhanced validation using common utilities."""
        try:
            tprint_debug(f"🔍 Validating input features: {features_df.shape}")
            
            # Use safe dataframe operation for validation
            def validate_and_clean_dataframe(df):
                tprint_debug("🧹 Starting dataframe validation and cleaning")
                
                # Check for NaN values using math validation
                if df.isnull().any().any():
                    tprint_warning("⚠️ Found NaN values, filling with 0")
                    df = df.fillna(0)
                
                # Check for infinite values using math validation
                if np.isinf(df.values).sum() > 0:
                    tprint_warning("⚠️ Found infinite values, clipping")
                    df = df.replace([np.inf, -np.inf], [np.finfo(np.float64).max, np.finfo(np.float64).min])
                
                # Validate finite values using math validation
                df_values = df.values
                if not validate_finite(df_values):
                    tprint_warning("⚠️ Found non-finite values, applying safe operations")
                    df_values = np.where(np.isfinite(df_values), df_values, 0)
                    df = pd.DataFrame(df_values, columns=df.columns, index=df.index)
                
                # Check for constant columns
                constant_cols = df.columns[df.nunique() <= 1]
                if len(constant_cols) > 0:
                    tprint_warning(f"⚠️ Removing constant columns: {constant_cols.tolist()}")
                    df = df.drop(columns=constant_cols)
                
                # Validate numeric range
                for col in df.columns:
                    if not validate_numeric_range(df[col].values, min_val=-1e10, max_val=1e10):
                        tprint_warning(f"⚠️ Column {col} has values outside expected range, clipping")
                        df[col] = df[col].clip(-1e10, 1e10)
                
                tprint_debug(f"✅ Dataframe validation completed: {df.shape}")
                return df
            
            # Use safe dataframe operation
            validated_df = safe_dataframe_operation(features_df, validate_and_clean_dataframe)
            
            # Data preview after validation
            tprint_data_preview(validated_df, "validated_features", max_rows=5, level="DEBUG")
            
            # Optimize memory usage
            optimized_df = optimize_dataframe_memory(validated_df)
            
            # Data preview after optimization
            tprint_data_preview(optimized_df, "optimized_features", max_rows=5, level="DEBUG")
            
            tprint_success(f"✅ Input validation completed: {optimized_df.shape}")
            return optimized_df
            
        except Exception as e:
            tprint_error(f"❌ Input validation failed: {e}")
            return features_df
    
    @tprint_logged(LogLevel.DEBUG, include_result=True)
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default HDBSCAN parameters."""
        tprint_debug("📋 Getting default HDBSCAN parameters")
        
        params = {
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
        
        tprint_debug(f"Default parameters: {params}")
        return params
    
    @tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
    def _optimize_parameters(self, features: np.ndarray) -> Dict[str, Any]:
        """Optimize HDBSCAN parameters using grid search."""
        try:
            tprint_info("🔧 Optimizing HDBSCAN parameters...")
            start_time = time.perf_counter()
            
            best_params = None
            best_score = -np.inf
            
            # Generate parameter combinations
            param_combinations = self._generate_param_combinations()
            
            tprint_info(f"Testing {len(param_combinations)} parameter combinations...")
            
            for i, params in enumerate(param_combinations):
                try:
                    # Perform clustering with current parameters
                    cluster_labels, _ = self._perform_clustering(features, params)
                    
                    # Calculate validation score
                    score = self._calculate_validation_score(cluster_labels, features)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        tprint_debug(f"New best score: {best_score:.3f} with params: {params}")
                    
                    if (i + 1) % 10 == 0:
                        tprint_progress(f"Completed {i + 1}/{len(param_combinations)} combinations. Best score: {best_score:.3f}")
                
                except Exception as e:
                    tprint_debug(f"Parameter combination failed: {e}")
                    continue
            
            if best_params is None:
                tprint_warning("⚠️ Parameter optimization failed, using default parameters")
                best_params = self._get_default_params()
            else:
                tprint_success(f"✅ Best parameters found: {best_params} (score: {best_score:.3f})")
            
            self.best_params = best_params
            self.best_score = best_score
            
            opt_time = time.perf_counter() - start_time
            tprint_performance(f"Parameter optimization completed in {opt_time:.3f}s")
            
            return best_params
            
        except Exception as e:
            tprint_error(f"❌ Parameter optimization failed: {e}")
            return self._get_default_params()
    
    @tprint_logged(LogLevel.DEBUG, include_result=True)
    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate parameter combinations for grid search."""
        try:
            tprint_debug("🔄 Generating parameter combinations for grid search")
            import itertools
            
            # Get parameter names and values
            param_names = list(self.config.param_search_space.keys())
            param_values = list(self.config.param_search_space.values())
            
            tprint_debug(f"Parameter names: {param_names}")
            tprint_debug(f"Parameter values: {param_values}")
            
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
            
            tprint_debug(f"Generated {len(param_combinations)} parameter combinations")
            return param_combinations
            
        except Exception as e:
            tprint_error(f"❌ Parameter combination generation failed: {e}")
            return [self._get_default_params()]
    
    @tprint_logged(LogLevel.DEBUG, include_args=True, include_result=True)
    def _perform_clustering(self, features: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform HDBSCAN clustering with given parameters."""
        try:
            tprint_debug(f"🔍 Performing HDBSCAN clustering with params: {params}")
            
            # Import HDBSCAN
            try:
                import hdbscan
                tprint_debug("✅ HDBSCAN imported successfully")
            except ImportError:
                tprint_error("❌ HDBSCAN not available. Please install: pip install hdbscan")
                raise ImportError("HDBSCAN not available")
            
            # Create clusterer
            clusterer = hdbscan.HDBSCAN(**params)
            tprint_debug("✅ HDBSCAN clusterer created")
            
            # Perform clustering
            start_time = time.perf_counter()
            cluster_labels = clusterer.fit_predict(features)
            clustering_time = time.perf_counter() - start_time
            
            tprint_debug(f"Clustering completed in {clustering_time:.3f}s")
            
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
            
            tprint_debug(f"Clustering info: {clustering_info['n_clusters']} clusters, {clustering_info['n_noise_points']} noise points")
            
            return cluster_labels, clustering_info
            
        except Exception as e:
            tprint_error(f"❌ HDBSCAN clustering failed: {e}")
            raise
    
    @tprint_logged(LogLevel.DEBUG, include_args=True, include_result=True)
    def _calculate_cluster_centers(self, features: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
        """Calculate cluster centers with enhanced math validation."""
        try:
            tprint_debug(f"📊 Calculating cluster centers for {features.shape[0]} samples")
            
            # Validate inputs
            if not validate_finite(features) or not validate_finite(cluster_labels):
                tprint_debug("Non-finite values in features or labels, returning empty centers")
                return np.array([])
            
            unique_labels = np.unique(cluster_labels)
            unique_labels = unique_labels[unique_labels != -1]  # Remove noise
            
            if len(unique_labels) == 0:
                tprint_debug("No valid clusters found, returning empty centers")
                return np.array([])
            
            tprint_debug(f"Found {len(unique_labels)} unique clusters: {unique_labels}")
            
            centers = []
            for label in unique_labels:
                mask = cluster_labels == label
                if mask.sum() > 0:
                    cluster_features = features[mask]
                    
                    # Validate cluster features
                    if not validate_finite(cluster_features):
                        tprint_debug(f"Non-finite values in cluster {label}, skipping")
                        continue
                    
                    # Calculate center using safe operations
                    def calculate_center():
                        return np.mean(cluster_features, axis=0)
                    
                    center = safe_statistical_operation(calculate_center, default=None)
                    
                    if center is not None and validate_finite(center):
                        centers.append(center)
                        tprint_debug(f"✅ Calculated center for cluster {label}")
                    else:
                        tprint_debug(f"Invalid center for cluster {label}, skipping")
            
            result = np.array(centers) if centers else np.array([])
            tprint_debug(f"✅ Cluster centers calculation completed: {len(centers)} centers")
            return result
            
        except Exception as e:
            tprint_debug(f"Cluster center calculation failed: {e}")
            return np.array([])
    
    @tprint_logged(LogLevel.DEBUG, include_args=True, include_result=True)
    def _calculate_cluster_sizes(self, cluster_labels: np.ndarray) -> Dict[int, int]:
        """Calculate cluster sizes."""
        try:
            tprint_debug(f"📏 Calculating cluster sizes for {len(cluster_labels)} labels")
            
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            sizes = dict(zip(unique_labels, counts))
            
            tprint_debug(f"Cluster sizes: {sizes}")
            return sizes
            
        except Exception as e:
            tprint_debug(f"Cluster size calculation failed: {e}")
            return {}
    
    @tprint_logged(LogLevel.DEBUG, include_args=True, include_result=True)
    def _calculate_validation_score(self, cluster_labels: np.ndarray, features: np.ndarray) -> float:
        """Calculate validation score for clustering with enhanced math validation."""
        try:
            tprint_debug(f"📊 Calculating validation score using {self.config.optimization_metric} metric")
            
            # Remove noise points for validation
            valid_mask = cluster_labels != -1
            if valid_mask.sum() < 2:
                tprint_debug("Insufficient valid samples for validation")
                return -np.inf
            
            valid_labels = cluster_labels[valid_mask]
            valid_features = features[valid_mask]
            
            if len(set(valid_labels)) < 2:
                tprint_debug("Insufficient unique clusters for validation")
                return -np.inf
            
            # Validate inputs using math validation
            if not validate_finite(valid_features):
                tprint_debug("Non-finite values in features, skipping validation")
                return -np.inf
            
            if not validate_finite(valid_labels):
                tprint_debug("Non-finite values in labels, skipping validation")
                return -np.inf
            
            # Calculate score based on optimization metric with safe operations
            def calculate_safe_score():
                if self.config.optimization_metric == 'silhouette':
                    return silhouette_score(valid_features, valid_labels)
                elif self.config.optimization_metric == 'calinski_harabasz':
                    return calinski_harabasz_score(valid_features, valid_labels)
                elif self.config.optimization_metric == 'davies_bouldin':
                    return -davies_bouldin_score(valid_features, valid_labels)  # Negative because lower is better
                else:
                    return silhouette_score(valid_features, valid_labels)
            
            # Use safe statistical operation
            score = safe_statistical_operation(calculate_safe_score, default=-np.inf)
            
            # Validate score is finite
            if not validate_finite(np.array([score])):
                tprint_debug("Non-finite validation score, returning -inf")
                return -np.inf
            
            tprint_debug(f"✅ Validation score calculated: {score:.4f}")
            return score
            
        except Exception as e:
            tprint_debug(f"Validation score calculation failed: {e}")
            return -np.inf
    
    @tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
    def _handle_noise_points(self, cluster_labels: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Handle noise points using specified strategy."""
        try:
            noise_mask = cluster_labels == -1
            n_noise = noise_mask.sum()
            
            if n_noise == 0:
                tprint_debug("No noise points to handle")
                return cluster_labels
            
            tprint_info(f"🔧 Handling {n_noise} noise points using {self.config.noise_strategy} strategy")
            
            if self.config.noise_strategy == 'keep':
                # Keep noise points as -1
                tprint_debug("Keeping noise points as -1")
                return cluster_labels
            
            elif self.config.noise_strategy == 'knn_assign':
                # Assign noise points to nearest cluster
                tprint_debug("Assigning noise points to nearest cluster using kNN")
                return self._assign_noise_to_nearest_cluster(cluster_labels, features, noise_mask)
            
            elif self.config.noise_strategy == 'causal_smooth':
                # Apply causal smoothing to noise points
                tprint_debug("Applying causal smoothing to noise points")
                return self._causal_smooth_noise(cluster_labels, noise_mask)
            
            else:
                tprint_warning(f"⚠️ Unknown noise strategy: {self.config.noise_strategy}")
                return cluster_labels
            
        except Exception as e:
            tprint_error(f"❌ Noise handling failed: {e}")
            return cluster_labels
    
    @tprint_logged(LogLevel.DEBUG, include_args=True, include_result=True)
    def _assign_noise_to_nearest_cluster(self, cluster_labels: np.ndarray, features: np.ndarray, noise_mask: np.ndarray) -> np.ndarray:
        """Assign noise points to nearest cluster using KNN."""
        try:
            tprint_debug(f"🔍 Assigning {noise_mask.sum()} noise points to nearest clusters using KNN")
            
            # Get non-noise points and their labels
            valid_mask = ~noise_mask
            valid_features = features[valid_mask]
            valid_labels = cluster_labels[valid_mask]
            
            if len(valid_features) == 0:
                tprint_warning("No valid features for KNN assignment")
                return cluster_labels
            
            # Get noise points
            noise_features = features[noise_mask]
            
            # Find nearest neighbors
            knn = NearestNeighbors(n_neighbors=min(5, len(valid_features)), metric=self.config.metric)
            knn.fit(valid_features)
            tprint_debug(f"KNN fitted with {knn.n_neighbors} neighbors")
            
            # Find nearest neighbors for noise points
            distances, indices = knn.kneighbors(noise_features)
            tprint_debug(f"Found nearest neighbors for {len(noise_features)} noise points")
            
            # Assign to most common label among neighbors
            new_labels = cluster_labels.copy()
            for i, noise_idx in enumerate(np.where(noise_mask)[0]):
                neighbor_labels = valid_labels[indices[i]]
                unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
                most_common_label = unique_labels[np.argmax(counts)]
                new_labels[noise_idx] = most_common_label
            
            tprint_success(f"✅ Assigned {noise_mask.sum()} noise points to nearest clusters")
            return new_labels
            
        except Exception as e:
            tprint_error(f"❌ Noise assignment failed: {e}")
            return cluster_labels
    
    @tprint_logged(LogLevel.DEBUG, include_args=True, include_result=True)
    def _causal_smooth_noise(self, cluster_labels: np.ndarray, noise_mask: np.ndarray) -> np.ndarray:
        """Apply causal smoothing to noise points."""
        try:
            tprint_debug(f"🔄 Applying causal smoothing to {noise_mask.sum()} noise points")
            
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
                    tprint_debug(f"Smoothed noise point {noise_idx} using previous label {prev_labels[-1]}")
                else:
                    # Look at next non-noise points
                    next_mask = (np.arange(len(cluster_labels)) > noise_idx) & (~noise_mask)
                    if next_mask.sum() > 0:
                        next_labels = cluster_labels[next_mask]
                        # Use first future non-noise label
                        new_labels[noise_idx] = next_labels[0]
                        tprint_debug(f"Smoothed noise point {noise_idx} using next label {next_labels[0]}")
                    else:
                        # No non-noise points found, keep as noise
                        new_labels[noise_idx] = -1
                        tprint_debug(f"Kept noise point {noise_idx} as noise (no neighbors)")
            
            tprint_success(f"✅ Causal smoothing completed")
            return new_labels
            
        except Exception as e:
            tprint_error(f"❌ Causal smoothing failed: {e}")
            return cluster_labels
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def _validate_clustering(self, cluster_labels: np.ndarray, features: np.ndarray):
        """Validate clustering results."""
        try:
            tprint_info("🔍 Validating clustering results")
            
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise_points = list(cluster_labels).count(-1)
            
            tprint_debug(f"Clustering stats: {n_clusters} clusters, {n_noise_points} noise points")
            
            # Check minimum clusters
            if n_clusters < self.config.min_clusters:
                tprint_warning(f"⚠️ Too few clusters: {n_clusters} < {self.config.min_clusters}")
            
            # Check maximum clusters
            if n_clusters > self.config.max_clusters:
                tprint_warning(f"⚠️ Too many clusters: {n_clusters} > {self.config.max_clusters}")
            
            # Check silhouette score
            if n_clusters >= 2:
                try:
                    valid_mask = cluster_labels != -1
                    if valid_mask.sum() >= 2:
                        valid_labels = cluster_labels[valid_mask]
                        valid_features = features[valid_mask]
                        
                        if len(set(valid_labels)) >= 2:
                            silhouette = silhouette_score(valid_features, valid_labels)
                            tprint_debug(f"Silhouette score: {silhouette:.3f}")
                            if silhouette < self.config.min_silhouette_score:
                                tprint_warning(f"⚠️ Low silhouette score: {silhouette:.3f} < {self.config.min_silhouette_score}")
                except Exception as e:
                    tprint_debug(f"Silhouette score validation failed: {e}")
            
            tprint_success(f"✅ Clustering validation completed. Clusters: {n_clusters}, Noise: {n_noise_points}")
            
        except Exception as e:
            tprint_error(f"❌ Clustering validation failed: {e}")
    
    @tprint_logged(LogLevel.DEBUG, include_args=True, include_result=True)
    def _calculate_clustering_stats(self, cluster_labels: np.ndarray, features: np.ndarray, clustering_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate clustering statistics."""
        try:
            tprint_debug("📊 Calculating clustering statistics")
            
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
            
            tprint_debug(f"Basic stats: {n_clusters} clusters, {n_noise_points} noise, {n_samples} samples")
            
            # Calculate silhouette score if possible
            if n_clusters >= 2:
                try:
                    valid_mask = cluster_labels != -1
                    if valid_mask.sum() >= 2:
                        valid_labels = cluster_labels[valid_mask]
                        valid_features = features[valid_mask]
                        
                        if len(set(valid_labels)) >= 2:
                            silhouette = silhouette_score(valid_features, valid_labels)
                            stats['silhouette_score'] = silhouette
                            tprint_debug(f"Silhouette score: {silhouette:.3f}")
                        else:
                            stats['silhouette_score'] = 0.0
                    else:
                        stats['silhouette_score'] = 0.0
                except Exception as e:
                    tprint_debug(f"Silhouette score calculation failed: {e}")
                    stats['silhouette_score'] = 0.0
            else:
                stats['silhouette_score'] = 0.0
            
            tprint_success(f"✅ Clustering statistics calculated: {len(stats)} metrics")
            return stats
            
        except Exception as e:
            tprint_error(f"❌ Clustering stats calculation failed: {e}")
            return {'error': str(e)}
    
    @tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
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
            tprint_info(f"🔮 Predicting cluster labels for {features.shape[0]} samples")
            
            if self.clusterer is None:
                tprint_warning("⚠️ No trained clusterer available, using random assignment")
                return self._random_fallback(features)
            
            # Try to use the clusterer's approximate_predict method if available
            if hasattr(self.clusterer, 'approximate_predict'):
                try:
                    tprint_debug("Using HDBSCAN approximate_predict method")
                    labels, probabilities = self.clusterer.approximate_predict(features)
                    tprint_success(f"✅ HDBSCAN approximate_predict completed: {len(labels)} predictions")
                    return labels, probabilities, "hdbscan_approximate_predict"
                except Exception as e:
                    tprint_debug(f"HDBSCAN approximate_predict failed: {e}")
            
            # Try enhanced distance-based prediction with better probability estimation
            tprint_debug("Using enhanced distance-based prediction")
            return self._enhanced_distance_based_prediction(features)
            
        except Exception as e:
            tprint_error(f"❌ Prediction failed: {e}")
            return self._random_fallback(features)
    
    @tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
    def predict_regime_probabilities(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Predict regime probabilities with detailed confidence measures.
        
        Args:
            features: Feature matrix for prediction (n_samples, n_features)
            
        Returns:
            Tuple of (labels, probabilities, confidence_info)
        """
        try:
            tprint_info("🔮 Predicting regime probabilities...")
            start_time = time.perf_counter()
            
            # Get basic prediction
            labels, probabilities, method = self.approximate_predict_with_fallback(features)
            tprint_debug(f"Basic prediction completed using {method}")
            
            # Calculate confidence measures
            confidence_info = self._calculate_prediction_confidence(features, labels, probabilities)
            tprint_debug(f"Confidence measures calculated: {len(confidence_info)} metrics")
            
            # Enhance probabilities with uncertainty quantification
            enhanced_probabilities = self._enhance_probability_estimation(features, labels, probabilities, confidence_info)
            tprint_debug(f"Probability enhancement completed")
            
            pred_time = time.perf_counter() - start_time
            tprint_success(f"✅ Regime prediction completed using {method} in {pred_time:.3f}s")
            
            return labels, enhanced_probabilities, confidence_info
            
        except Exception as e:
            tprint_error(f"❌ Regime probability prediction failed: {e}")
            return self._random_fallback(features)
    
    def predict_out_of_sample(self, features: np.ndarray, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Predict out-of-sample regime assignments with confidence filtering.
        
        Args:
            features: Feature matrix for prediction (n_samples, n_features)
            confidence_threshold: Minimum confidence threshold for predictions
            
        Returns:
            Dictionary with prediction results and confidence measures
        """
        try:
            logger.info(f"🔮 Predicting out-of-sample regimes (confidence threshold: {confidence_threshold})...")
            
            # Get predictions
            labels, probabilities, confidence_info = self.predict_regime_probabilities(features)
            
            # Filter by confidence
            high_confidence_mask = probabilities >= confidence_threshold
            n_high_confidence = np.sum(high_confidence_mask)
            
            # Calculate prediction quality metrics
            prediction_quality = self._assess_prediction_quality(features, labels, probabilities, confidence_info)
            
            results = {
                'labels': labels,
                'probabilities': probabilities,
                'confidence_info': confidence_info,
                'high_confidence_mask': high_confidence_mask,
                'n_high_confidence': n_high_confidence,
                'confidence_ratio': n_high_confidence / len(features),
                'prediction_quality': prediction_quality,
                'recommendations': self._generate_prediction_recommendations(confidence_info, prediction_quality)
            }
            
            logger.info(f"✅ Out-of-sample prediction completed. High confidence: {n_high_confidence}/{len(features)}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Out-of-sample prediction failed: {e}")
            return {'error': str(e)}
    
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
    
    def _enhanced_distance_based_prediction(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """Enhanced distance-based prediction with better probability estimation and math validation."""
        try:
            # Validate input features
            if not validate_finite(features):
                logger.warning("⚠️ Non-finite values in features, using random assignment")
                return self._random_fallback(features)
            
            # Get cluster centers from training
            if hasattr(self, 'cluster_centers') and self.cluster_centers is not None:
                centers = self.cluster_centers
            else:
                logger.warning("⚠️ No cluster centers available, using random assignment")
                return self._random_fallback(features)
            
            if len(centers) == 0:
                return self._random_fallback(features)
            
            # Validate cluster centers
            if not validate_finite(centers):
                logger.warning("⚠️ Non-finite values in cluster centers, using random assignment")
                return self._random_fallback(features)
            
            # Calculate distances to cluster centers using safe operations
            def calculate_distances():
                return np.sqrt(((features[:, np.newaxis] - centers[np.newaxis, :]) ** 2).sum(axis=2))
            
            distances = safe_statistical_operation(calculate_distances, default=None)
            if distances is None or not validate_finite(distances):
                logger.warning("⚠️ Distance calculation failed, using random assignment")
                return self._random_fallback(features)
            
            # Assign to closest cluster
            labels = np.argmin(distances, axis=1)
            
            # Enhanced probability calculation using softmax with safe operations
            def calculate_probabilities():
                # Convert distances to similarities (inverse relationship)
                max_distance = safe_statistical_operation(lambda: np.max(distances), default=1.0)
                similarities = max_distance - distances + 1e-10  # Add small value to avoid division by zero
                
                # Apply softmax to get probabilities
                std_similarities = safe_statistical_operation(lambda: np.std(similarities), default=1.0)
                exp_similarities = np.exp(similarities / std_similarities)  # Normalize by standard deviation
                
                # Safe division for probabilities
                sum_exp = np.sum(exp_similarities, axis=1, keepdims=True)
                probabilities = safe_divide(exp_similarities, sum_exp, default=1.0/len(centers))
                
                return probabilities
            
            probabilities_matrix = safe_statistical_operation(calculate_probabilities, default=None)
            if probabilities_matrix is None or not validate_finite(probabilities_matrix):
                logger.warning("⚠️ Probability calculation failed, using uniform probabilities")
                probabilities_matrix = np.ones((len(features), len(centers))) / len(centers)
            
            # Get maximum probability for each sample
            max_probabilities = np.max(probabilities_matrix, axis=1)
            
            # Validate final probabilities
            if not validate_finite(max_probabilities):
                logger.warning("⚠️ Non-finite probabilities, using uniform probabilities")
                max_probabilities = np.ones(len(features)) / len(centers)
            
            return labels, max_probabilities, "enhanced_distance_based"
            
        except Exception as e:
            logger.error(f"❌ Enhanced distance-based prediction failed: {e}")
            return self._distance_based_prediction(features)
    
    def _calculate_prediction_confidence(self, features: np.ndarray, labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, Any]:
        """Calculate confidence measures for predictions."""
        try:
            confidence_info = {}
            
            # Basic confidence metrics
            confidence_info['avg_probability'] = np.mean(probabilities)
            confidence_info['min_probability'] = np.min(probabilities)
            confidence_info['max_probability'] = np.max(probabilities)
            confidence_info['probability_std'] = np.std(probabilities)
            
            # Confidence distribution
            high_conf = np.sum(probabilities >= 0.7)
            medium_conf = np.sum((probabilities >= 0.4) & (probabilities < 0.7))
            low_conf = np.sum(probabilities < 0.4)
            
            confidence_info['high_confidence_count'] = high_conf
            confidence_info['medium_confidence_count'] = medium_conf
            confidence_info['low_confidence_count'] = low_conf
            confidence_info['high_confidence_ratio'] = high_conf / len(probabilities)
            
            # Feature-based confidence (if we have training data)
            if hasattr(self, 'cluster_centers') and self.cluster_centers is not None:
                # Calculate distance-based confidence
                distances = np.sqrt(((features[:, np.newaxis] - self.cluster_centers[np.newaxis, :]) ** 2).sum(axis=2))
                min_distances = np.min(distances, axis=1)
                avg_distance = np.mean(min_distances)
                
                confidence_info['avg_distance_to_centers'] = avg_distance
                confidence_info['distance_confidence'] = 1.0 / (1.0 + avg_distance)  # Higher distance = lower confidence
            
            # Regime distribution confidence
            unique_labels, counts = np.unique(labels, return_counts=True)
            regime_balance = 1.0 - np.std(counts) / (np.mean(counts) + 1e-10)
            confidence_info['regime_balance'] = regime_balance
            
            return confidence_info
            
        except Exception as e:
            logger.error(f"❌ Confidence calculation failed: {e}")
            return {}
    
    def _enhance_probability_estimation(self, features: np.ndarray, labels: np.ndarray, 
                                      probabilities: np.ndarray, confidence_info: Dict[str, Any]) -> np.ndarray:
        """Enhance probability estimation with uncertainty quantification."""
        try:
            enhanced_probabilities = probabilities.copy()
            
            # Apply confidence-based adjustment
            avg_confidence = confidence_info.get('avg_probability', 0.5)
            confidence_std = confidence_info.get('probability_std', 0.1)
            
            # Adjust probabilities based on confidence distribution
            if confidence_std > 0:
                # Normalize probabilities to account for uncertainty
                z_scores = (probabilities - avg_confidence) / confidence_std
                # Apply sigmoid function to bound probabilities
                enhanced_probabilities = 1.0 / (1.0 + np.exp(-z_scores))
            
            # Apply regime balance adjustment
            regime_balance = confidence_info.get('regime_balance', 1.0)
            if regime_balance < 0.5:  # Unbalanced regime distribution
                # Reduce confidence for less balanced predictions
                enhanced_probabilities *= (0.5 + regime_balance)
            
            # Ensure probabilities are in valid range
            enhanced_probabilities = np.clip(enhanced_probabilities, 0.0, 1.0)
            
            return enhanced_probabilities
            
        except Exception as e:
            logger.error(f"❌ Probability enhancement failed: {e}")
            return probabilities
    
    def _assess_prediction_quality(self, features: np.ndarray, labels: np.ndarray, 
                                 probabilities: np.ndarray, confidence_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of predictions."""
        try:
            quality = {}
            
            # Basic quality metrics
            quality['n_predictions'] = len(labels)
            quality['n_unique_regimes'] = len(np.unique(labels))
            quality['avg_probability'] = np.mean(probabilities)
            
            # Prediction consistency
            high_conf_mask = probabilities >= 0.7
            if np.sum(high_conf_mask) > 0:
                high_conf_labels = labels[high_conf_mask]
                quality['high_conf_regime_diversity'] = len(np.unique(high_conf_labels))
            else:
                quality['high_conf_regime_diversity'] = 0
            
            # Feature space coverage
            if len(features) > 1:
                feature_std = np.std(features, axis=0)
                quality['feature_diversity'] = np.mean(feature_std)
                quality['feature_coverage'] = np.sum(feature_std > 0) / len(feature_std)
            else:
                quality['feature_diversity'] = 0.0
                quality['feature_coverage'] = 0.0
            
            # Prediction stability (if we have multiple samples)
            if len(labels) > 1:
                label_changes = np.sum(np.diff(labels) != 0)
                quality['prediction_stability'] = 1.0 - (label_changes / (len(labels) - 1))
            else:
                quality['prediction_stability'] = 1.0
            
            # Overall quality score
            quality_score = (
                quality['avg_probability'] * 0.3 +
                quality['prediction_stability'] * 0.3 +
                quality['feature_coverage'] * 0.2 +
                (quality['high_conf_regime_diversity'] / max(quality['n_unique_regimes'], 1)) * 0.2
            )
            quality['overall_quality_score'] = quality_score
            
            return quality
            
        except Exception as e:
            logger.error(f"❌ Prediction quality assessment failed: {e}")
            return {}
    
    def _generate_prediction_recommendations(self, confidence_info: Dict[str, Any], 
                                           prediction_quality: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations based on prediction results."""
        try:
            recommendations = {}
            
            # Confidence-based recommendations
            high_conf_ratio = confidence_info.get('high_confidence_ratio', 0.0)
            if high_conf_ratio > 0.8:
                recommendations['confidence_level'] = 'high'
                recommendations['confidence_message'] = 'High confidence predictions - suitable for trading decisions'
            elif high_conf_ratio > 0.5:
                recommendations['confidence_level'] = 'medium'
                recommendations['confidence_message'] = 'Medium confidence predictions - use with caution'
            else:
                recommendations['confidence_level'] = 'low'
                recommendations['confidence_message'] = 'Low confidence predictions - avoid trading decisions'
            
            # Quality-based recommendations
            overall_quality = prediction_quality.get('overall_quality_score', 0.0)
            if overall_quality > 0.8:
                recommendations['quality_level'] = 'excellent'
                recommendations['quality_message'] = 'Excellent prediction quality'
            elif overall_quality > 0.6:
                recommendations['quality_level'] = 'good'
                recommendations['quality_message'] = 'Good prediction quality'
            else:
                recommendations['quality_level'] = 'poor'
                recommendations['quality_message'] = 'Poor prediction quality - consider retraining'
            
            # Regime balance recommendations
            regime_balance = confidence_info.get('regime_balance', 1.0)
            if regime_balance < 0.3:
                recommendations['regime_balance'] = 'unbalanced'
                recommendations['regime_message'] = 'Unbalanced regime distribution - consider adjusting parameters'
            else:
                recommendations['regime_balance'] = 'balanced'
                recommendations['regime_message'] = 'Balanced regime distribution'
            
            # Overall recommendation
            if high_conf_ratio > 0.7 and overall_quality > 0.7:
                recommendations['overall'] = 'proceed'
                recommendations['overall_message'] = 'Predictions are reliable for use'
            elif high_conf_ratio > 0.5 and overall_quality > 0.5:
                recommendations['overall'] = 'caution'
                recommendations['overall_message'] = 'Use predictions with caution and additional validation'
            else:
                recommendations['overall'] = 'avoid'
                recommendations['overall_message'] = 'Avoid using these predictions for trading decisions'
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Recommendation generation failed: {e}")
            return {}
    
    def _random_fallback(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """Fallback to random assignment when all other methods fail."""
        try:
            n_samples = len(features)
            
            # Data preview for fallback case
            tprint_data_preview(features, "fallback_input_features", max_rows=5, level="WARNING")
            
            # Random labels (assuming 2-5 clusters)
            n_clusters = np.random.randint(2, 6)
            labels = np.random.randint(0, n_clusters, n_samples)
            
            # Random probabilities
            probabilities = np.random.uniform(0.1, 0.9, n_samples)
            
            # Data preview of fallback results
            tprint_data_preview(labels, "fallback_labels", max_rows=10, level="WARNING")
            tprint_data_preview(probabilities, "fallback_probabilities", max_rows=10, level="WARNING")
            
            return labels, probabilities, "random_fallback"
            
        except Exception as e:
            logger.error(f"❌ Random fallback failed: {e}")
            # Ultimate fallback
            n_samples = len(features)
            ultimate_labels = np.zeros(n_samples)
            ultimate_probabilities = np.ones(n_samples)
            
            # Data preview of ultimate fallback
            tprint_data_preview(ultimate_labels, "ultimate_fallback_labels", max_rows=10, level="ERROR")
            tprint_data_preview(ultimate_probabilities, "ultimate_fallback_probabilities", max_rows=10, level="ERROR")
            
            return ultimate_labels, ultimate_probabilities, "ultimate_fallback"
    
    @tprint_logged(LogLevel.DEBUG, include_result=True)
    def get_clustering_stats(self) -> Dict[str, Any]:
        """Get clustering statistics."""
        tprint_debug("📊 Retrieving clustering statistics")
        stats = self.clustering_stats.copy()
        tprint_debug(f"Retrieved {len(stats)} clustering statistics")
        return stats
    
    @tprint_logged(LogLevel.DEBUG, include_result=True)
    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get best parameters from optimization."""
        tprint_debug("🔧 Retrieving best parameters from optimization")
        params = self.best_params
        if params:
            tprint_debug(f"Best parameters: {params}")
        else:
            tprint_debug("No best parameters available")
        return params
    
    @tprint_logged(LogLevel.DEBUG, include_result=True)
    def get_best_score(self) -> float:
        """Get best score from optimization."""
        tprint_debug("📈 Retrieving best score from optimization")
        score = self.best_score
        tprint_debug(f"Best score: {score:.4f}")
        return score