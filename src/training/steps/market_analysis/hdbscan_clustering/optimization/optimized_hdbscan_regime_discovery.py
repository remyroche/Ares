"""
Optimized HDBSCAN Regime Discovery

This module provides a comprehensive, optimized HDBSCAN regime discovery system
that integrates all optimization components for maximum performance.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import optimization components
from .enhanced_memory_optimizer import (
    EnhancedMemoryOptimizer,
    MemoryOptimizationConfig,
    create_enhanced_memory_optimizer
)

from .enhanced_hyperparameter_optimizer import (
    EnhancedHyperparameterOptimizer,
    HDBSCANHyperparameterConfig,
    create_enhanced_hyperparameter_optimizer
)

from .enhanced_vectorized_processor import (
    EnhancedVectorizedProcessor,
    VectorizedProcessingConfig,
    create_enhanced_vectorized_processor
)

from .features_common_integration import (
    FeaturesCommonHDBSCANIntegration,
    FeaturesCommonIntegrationConfig,
    create_features_common_hdbscan_integration
)

# Import feature generation systems
from src.feature_generation.categories.entropy import create_default_entropy_generators
from src.feature_generation.categories.spectral_wavelet import create_default_spectral_wavelet_generators
from src.feature_generation.categories.regime_features import create_default_regime_generators

# Import HDBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    hdbscan = None

logger = logging.getLogger(__name__)

@dataclass
class OptimizedRegimeResult:
    """Result of optimized HDBSCAN regime discovery."""
    cluster_labels: np.ndarray
    cluster_probabilities: np.ndarray
    n_clusters: int
    n_noise_points: int
    cluster_persistence: Optional[np.ndarray] = None
    condensed_tree: Optional[Any] = None
    mst: Optional[Any] = None
    glosh_scores: Optional[np.ndarray] = None
    cluster_centers: Optional[np.ndarray] = None
    cluster_sizes: Optional[np.ndarray] = None
    noise_ratio: float = 0.0
    
    # Performance metrics
    silhouette_score: Optional[float] = None
    calinski_harabasz_score: Optional[float] = None
    davies_bouldin_score: Optional[float] = None
    
    # Processing information
    processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    optimization_stats: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None

@dataclass
class OptimizedHDBSCANRegimeDiscoveryConfig:
    """Configuration for optimized HDBSCAN regime discovery."""
    # Core HDBSCAN parameters
    min_cluster_size: int = 10
    min_samples: int = 5
    cluster_selection_epsilon: float = 0.0
    cluster_selection_method: str = 'eom'
    metric: str = 'euclidean'
    alpha: float = 1.0
    
    # Optimization settings
    enable_hyperparameter_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_vectorized_processing: bool = True
    enable_features_common: bool = True
    
    # Feature generation
    enable_entropy_features: bool = True
    enable_spectral_features: bool = True
    enable_regime_features: bool = True
    enable_normalization_features: bool = True
    
    # Performance settings
    enable_parallel_processing: bool = True
    max_memory_gb: float = 8.0
    chunk_size: int = 1000
    n_jobs: int = -1
    
    # Evaluation metrics
    primary_metric: str = 'silhouette'
    enable_cross_validation: bool = True
    cv_folds: int = 5
    
    # Advanced settings
    enable_feature_selection: bool = True
    feature_selection_method: str = 'mrmr'  # 'mrmr', 'lasso', 'mutual_info'
    max_features: int = 50
    feature_selection_threshold: float = 0.01

class OptimizedHDBSCANRegimeDiscovery:
    """
    Optimized HDBSCAN regime discovery with comprehensive optimization.
    
    This class integrates all optimization components for maximum performance:
    - Memory & Data Processing Optimization
    - Hyperparameter Optimization
    - Vectorized Computations
    - Features Common Integration
    - Feature Selection
    - Performance Monitoring
    """
    
    def __init__(self, config: Optional[OptimizedHDBSCANRegimeDiscoveryConfig] = None):
        """Initialize the optimized HDBSCAN regime discovery."""
        self.config = config or OptimizedHDBSCANRegimeDiscoveryConfig()
        
        # Initialize optimization components
        self._initialize_optimization_components()
        
        # Initialize feature generators
        self._initialize_feature_generators()
        
        # Performance tracking
        self.performance_stats = {
            'total_processing_time': 0.0,
            'feature_generation_time': 0.0,
            'hyperparameter_optimization_time': 0.0,
            'clustering_time': 0.0,
            'post_processing_time': 0.0,
            'memory_optimizations': 0,
            'vectorized_operations': 0,
            'caching_hits': 0,
            'optimization_improvements': 0
        }
        
        logger.info("✅ OptimizedHDBSCANRegimeDiscovery initialized")
    
    def _initialize_optimization_components(self):
        """Initialize all optimization components."""
        # Memory optimizer
        if self.config.enable_memory_optimization:
            self.memory_optimizer = create_enhanced_memory_optimizer(
                max_memory_gb=self.config.max_memory_gb,
                enable_memory_optimization=True,
                enable_data_validation=True,
                enable_safe_operations=True,
                enable_memory_monitoring=True
            )
        else:
            self.memory_optimizer = None
        
        # Hyperparameter optimizer
        if self.config.enable_hyperparameter_optimization:
            self.hyperparameter_optimizer = create_enhanced_hyperparameter_optimizer(
                optimization_strategy="hybrid",
                n_trials=50,
                primary_metric=self.config.primary_metric,
                enable_parallel=self.config.enable_parallel_processing,
                memory_efficient=True
            )
        else:
            self.hyperparameter_optimizer = None
        
        # Vectorized processor
        if self.config.enable_vectorized_processing:
            self.vectorized_processor = create_enhanced_vectorized_processor(
                enable_vectorbt=True,
                enable_gpu=False,
                enable_parallel=self.config.enable_parallel_processing,
                memory_efficient=True,
                max_memory_gb=self.config.max_memory_gb,
                chunk_size=self.config.chunk_size
            )
        else:
            self.vectorized_processor = None
        
        # Features common integration
        if self.config.enable_features_common:
            self.features_common_integration = create_features_common_hdbscan_integration(
                enable_unified_vectorization=True,
                enable_vectorbt_optimization=True,
                enable_automatic_scaling=True,
                enable_performance_monitoring=True,
                enable_caching=True,
                optimization_level="high",
                memory_efficient=True,
                max_memory_gb=self.config.max_memory_gb
            )
        else:
            self.features_common_integration = None
    
    def _initialize_feature_generators(self):
        """Initialize feature generators."""
        self.feature_generators = []
        
        # Entropy features
        if self.config.enable_entropy_features:
            entropy_generators = create_default_entropy_generators()
            self.feature_generators.extend(entropy_generators)
        
        # Spectral features
        if self.config.enable_spectral_features:
            spectral_generators = create_default_spectral_wavelet_generators()
            self.feature_generators.extend(spectral_generators)
        
        # Regime features
        if self.config.enable_regime_features:
            regime_generators = create_default_regime_generators()
            self.feature_generators.extend(regime_generators)
    
    def discover_regimes(self, data: pd.DataFrame, 
                        labels: Optional[np.ndarray] = None) -> OptimizedRegimeResult:
        """
        Discover regimes using optimized HDBSCAN clustering.
        
        Args:
            data: Input data for regime discovery
            labels: Optional labels for supervised evaluation
            
        Returns:
            OptimizedRegimeResult with clustering results and performance metrics
        """
        start_time = time.time()
        
        logger.info(f"🚀 Starting optimized regime discovery for {data.shape[0]} samples")
        
        # Step 1: Feature generation with optimization
        features_df = self._generate_optimized_features(data)
        feature_generation_time = time.time() - start_time
        self.performance_stats['feature_generation_time'] += feature_generation_time
        
        # Step 2: Hyperparameter optimization
        hyperparameter_start = time.time()
        best_params = self._optimize_hyperparameters(features_df, labels)
        hyperparameter_time = time.time() - hyperparameter_start
        self.performance_stats['hyperparameter_optimization_time'] += hyperparameter_time
        
        # Step 3: Feature selection
        if self.config.enable_feature_selection:
            features_df = self._select_optimal_features(features_df, labels)
        
        # Step 4: Optimized clustering
        clustering_start = time.time()
        cluster_labels, clustering_info = self._perform_optimized_clustering(
            features_df, best_params
        )
        clustering_time = time.time() - clustering_start
        self.performance_stats['clustering_time'] += clustering_time
        
        # Step 5: Post-processing and evaluation
        post_processing_start = time.time()
        result = self._create_optimized_result(
            cluster_labels, clustering_info, features_df, labels
        )
        post_processing_time = time.time() - post_processing_start
        self.performance_stats['post_processing_time'] += post_processing_time
        
        # Update total processing time
        total_time = time.time() - start_time
        self.performance_stats['total_processing_time'] += total_time
        result.processing_time = total_time
        
        logger.info(f"✅ Regime discovery completed: {total_time:.2f}s, "
                   f"{result.n_clusters} clusters found")
        
        return result
    
    def _generate_optimized_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features with comprehensive optimization."""
        if self.features_common_integration:
            # Use features_common integration for maximum optimization
            return self.features_common_integration.process_data_with_features_common(data)
        else:
            # Fallback to basic feature generation
            features_df = data.copy()
            
            for generator in self.feature_generators:
                try:
                    feature_result = generator.generate(data)
                    
                    if isinstance(feature_result, pd.DataFrame):
                        features_df = pd.concat([features_df, feature_result], axis=1)
                    elif isinstance(feature_result, pd.Series):
                        features_df[feature_result.name] = feature_result
                    
                except Exception as e:
                    logger.warning(f"⚠️ Feature generation failed: {e}")
                    continue
            
            return features_df
    
    def _optimize_hyperparameters(self, features_df: pd.DataFrame, 
                                 labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Optimize HDBSCAN hyperparameters."""
        if not self.hyperparameter_optimizer:
            # Use default parameters
            return {
                'min_cluster_size': self.config.min_cluster_size,
                'min_samples': self.config.min_samples,
                'cluster_selection_epsilon': self.config.cluster_selection_epsilon,
                'cluster_selection_method': self.config.cluster_selection_method,
                'metric': self.config.metric,
                'alpha': self.config.alpha
            }
        
        # Perform hyperparameter optimization
        optimization_results = self.hyperparameter_optimizer.optimize_hyperparameters(
            features_df, labels
        )
        
        return optimization_results.get('best_params', {
            'min_cluster_size': self.config.min_cluster_size,
            'min_samples': self.config.min_samples,
            'cluster_selection_epsilon': self.config.cluster_selection_epsilon,
            'cluster_selection_method': self.config.cluster_selection_method,
            'metric': self.config.metric,
            'alpha': self.config.alpha
        })
    
    def _select_optimal_features(self, features_df: pd.DataFrame, 
                               labels: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Select optimal features for clustering."""
        # Simple feature selection based on variance and correlation
        # Remove low variance features
        variance_threshold = 0.01
        high_variance_features = features_df.var() > variance_threshold
        features_df = features_df.loc[:, high_variance_features]
        
        # Remove highly correlated features
        correlation_threshold = 0.95
        corr_matrix = features_df.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
        features_df = features_df.drop(columns=to_drop)
        
        # Limit to max_features if specified
        if len(features_df.columns) > self.config.max_features:
            # Select top features by variance
            feature_variance = features_df.var().sort_values(ascending=False)
            top_features = feature_variance.head(self.config.max_features).index
            features_df = features_df[top_features]
        
        logger.info(f"✅ Feature selection: {len(features_df.columns)} features selected")
        
        return features_df
    
    def _perform_optimized_clustering(self, features_df: pd.DataFrame, 
                                    hdbscan_params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform optimized HDBSCAN clustering."""
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN not available")
        
        # Use vectorized processor if available
        if self.vectorized_processor:
            cluster_labels, clustering_info = self.vectorized_processor.optimized_hdbscan_clustering(
                features_df, **hdbscan_params
            )
        else:
            # Standard HDBSCAN clustering
            clusterer = hdbscan.HDBSCAN(**hdbscan_params)
            cluster_labels = clusterer.fit_predict(features_df)
            clustering_info = {
                'clusterer': clusterer,
                'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                'n_noise_points': list(cluster_labels).count(-1)
            }
        
        return cluster_labels, clustering_info
    
    def _create_optimized_result(self, cluster_labels: np.ndarray, 
                               clustering_info: Dict[str, Any], 
                               features_df: pd.DataFrame,
                               labels: Optional[np.ndarray] = None) -> OptimizedRegimeResult:
        """Create optimized result with comprehensive metrics."""
        # Basic clustering information
        n_clusters = clustering_info.get('n_clusters', 0)
        n_noise_points = clustering_info.get('n_noise_points', 0)
        noise_ratio = n_noise_points / len(cluster_labels) if len(cluster_labels) > 0 else 0.0
        
        # Calculate evaluation metrics
        silhouette_score = None
        calinski_harabasz_score = None
        davies_bouldin_score = None
        
        if n_clusters > 1:
            try:
                from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
                
                # Remove noise points for evaluation
                valid_mask = cluster_labels != -1
                if valid_mask.sum() > 1:
                    valid_features = features_df[valid_mask]
                    valid_labels = cluster_labels[valid_mask]
                    
                    if len(set(valid_labels)) > 1:
                        silhouette_score = silhouette_score(valid_features, valid_labels)
                        calinski_harabasz_score = calinski_harabasz_score(valid_features, valid_labels)
                        davies_bouldin_score = davies_bouldin_score(valid_features, valid_labels)
            except Exception as e:
                logger.warning(f"⚠️ Evaluation metrics calculation failed: {e}")
        
        # Get performance statistics
        optimization_stats = self.get_performance_stats()
        
        # Calculate feature importance (simple variance-based)
        feature_importance = {}
        if len(features_df.columns) > 0:
            feature_variance = features_df.var()
            total_variance = feature_variance.sum()
            if total_variance > 0:
                feature_importance = (feature_variance / total_variance).to_dict()
        
        # Calculate cluster probabilities using the clusterer's method
        cluster_probabilities = self._calculate_cluster_probabilities(
            cluster_labels, clustering_info, features_df
        )
        
        return OptimizedRegimeResult(
            cluster_labels=cluster_labels,
            cluster_probabilities=cluster_probabilities,
            n_clusters=n_clusters,
            n_noise_points=n_noise_points,
            cluster_persistence=clustering_info.get('cluster_persistence'),
            condensed_tree=clustering_info.get('condensed_tree'),
            mst=clustering_info.get('mst'),
            glosh_scores=clustering_info.get('glosh_scores'),
            cluster_centers=clustering_info.get('cluster_centers'),
            cluster_sizes=clustering_info.get('cluster_sizes'),
            noise_ratio=noise_ratio,
            silhouette_score=silhouette_score,
            calinski_harabasz_score=calinski_harabasz_score,
            davies_bouldin_score=davies_bouldin_score,
            optimization_stats=optimization_stats,
            feature_importance=feature_importance
        )
    
    def _calculate_cluster_probabilities(self, 
                                       cluster_labels: np.ndarray,
                                       clustering_info: Dict[str, Any],
                                       features_df: pd.DataFrame) -> np.ndarray:
        """Calculate cluster probabilities for each sample."""
        try:
            # Get cluster centers
            cluster_centers = clustering_info.get('cluster_centers')
            if cluster_centers is None or len(cluster_centers) == 0:
                # Fallback to uniform probabilities
                return np.ones(len(cluster_labels))
            
            # Calculate distances to cluster centers
            distances = np.sqrt(((features_df.values[:, np.newaxis] - cluster_centers[np.newaxis, :]) ** 2).sum(axis=2))
            
            # Convert distances to probabilities using softmax
            # Lower distance = higher probability
            max_distances = np.max(distances, axis=1, keepdims=True)
            normalized_distances = distances / (max_distances + 1e-10)
            
            # Apply softmax to get probabilities
            exp_distances = np.exp(-normalized_distances)
            probabilities = exp_distances / np.sum(exp_distances, axis=1, keepdims=True)
            
            # For each sample, get the probability of its assigned cluster
            cluster_probabilities = np.zeros(len(cluster_labels))
            for i, label in enumerate(cluster_labels):
                if label != -1:  # Not noise
                    # Find the index of the assigned cluster
                    unique_labels = np.unique(cluster_labels)
                    unique_labels = unique_labels[unique_labels != -1]
                    if label in unique_labels:
                        cluster_idx = np.where(unique_labels == label)[0][0]
                        cluster_probabilities[i] = probabilities[i, cluster_idx]
                    else:
                        cluster_probabilities[i] = 0.0
                else:
                    cluster_probabilities[i] = 0.0  # Noise has zero probability
            
            return cluster_probabilities
            
        except Exception as e:
            logger.error(f"❌ Cluster probability calculation failed: {e}")
            # Fallback to uniform probabilities
            return np.ones(len(cluster_labels))
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add component-specific stats
        if self.memory_optimizer:
            memory_stats = self.memory_optimizer.get_memory_stats()
            stats['memory_optimizer_stats'] = memory_stats
        
        if self.hyperparameter_optimizer:
            hyperparameter_stats = self.hyperparameter_optimizer.get_optimization_results()
            stats['hyperparameter_optimizer_stats'] = hyperparameter_stats
        
        if self.vectorized_processor:
            vectorized_stats = self.vectorized_processor.get_performance_stats()
            stats['vectorized_processor_stats'] = vectorized_stats
        
        if self.features_common_integration:
            features_common_stats = self.features_common_integration.get_performance_stats()
            stats['features_common_stats'] = features_common_stats
        
        return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_processing_time': 0.0,
            'feature_generation_time': 0.0,
            'hyperparameter_optimization_time': 0.0,
            'clustering_time': 0.0,
            'post_processing_time': 0.0,
            'memory_optimizations': 0,
            'vectorized_operations': 0,
            'caching_hits': 0,
            'optimization_improvements': 0
        }
        
        # Reset component stats
        if self.memory_optimizer:
            self.memory_optimizer.reset_stats()
        
        if self.hyperparameter_optimizer:
            self.hyperparameter_optimizer.reset_optimization()
        
        if self.vectorized_processor:
            self.vectorized_processor.reset_stats()
        
        if self.features_common_integration:
            self.features_common_integration.reset_performance_stats()

# Convenience function
def create_optimized_hdbscan_regime_discovery(
    min_cluster_size: int = 10,
    min_samples: int = 5,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = 'eom',
    metric: str = 'euclidean',
    enable_hyperparameter_optimization: bool = True,
    enable_memory_optimization: bool = True,
    enable_vectorized_processing: bool = True,
    enable_features_common: bool = True,
    enable_feature_selection: bool = True,
    max_features: int = 50,
    max_memory_gb: float = 8.0,
    n_jobs: int = -1
) -> OptimizedHDBSCANRegimeDiscovery:
    """
    Create an optimized HDBSCAN regime discovery with specified configuration.
    
    Args:
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Minimum samples for HDBSCAN
        cluster_selection_epsilon: Cluster selection epsilon for HDBSCAN
        cluster_selection_method: Cluster selection method for HDBSCAN
        metric: Distance metric for HDBSCAN
        enable_hyperparameter_optimization: Enable hyperparameter optimization
        enable_memory_optimization: Enable memory optimization
        enable_vectorized_processing: Enable vectorized processing
        enable_features_common: Enable features_common integration
        enable_feature_selection: Enable feature selection
        max_features: Maximum number of features to use
        max_memory_gb: Maximum memory usage in GB
        n_jobs: Number of parallel jobs
        
    Returns:
        OptimizedHDBSCANRegimeDiscovery instance
    """
    config = OptimizedHDBSCANRegimeDiscoveryConfig(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        metric=metric,
        enable_hyperparameter_optimization=enable_hyperparameter_optimization,
        enable_memory_optimization=enable_memory_optimization,
        enable_vectorized_processing=enable_vectorized_processing,
        enable_features_common=enable_features_common,
        enable_feature_selection=enable_feature_selection,
        max_features=max_features,
        max_memory_gb=max_memory_gb,
        n_jobs=n_jobs
    )
    
    return OptimizedHDBSCANRegimeDiscovery(config)