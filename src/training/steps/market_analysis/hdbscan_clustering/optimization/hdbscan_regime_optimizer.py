"""
HDBSCAN Regime Optimizer

This module integrates the optimized regime feature processing with HDBSCAN clustering
to provide a complete optimized regime discovery solution.

Key Features:
- Integration with existing HDBSCAN clustering pipeline
- Optimized feature processing for regime discovery
- Memory and computation efficient clustering
- Performance monitoring and statistics
- VectorBT acceleration for distance calculations
- Regime quality metrics and validation
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time
import hdbscan
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import warnings

# Import optimization modules
from .optimized_regime_feature_processor import (
    OptimizedRegimeFeatureProcessor,
    OptimizedRegimeFeatureProcessorConfig,
    create_optimized_regime_feature_processor
)
from .optimized_dimensionality_reducer import (
    OptimizedDimensionalityReducer,
    DimensionalityReductionConfig,
    create_optimized_dimensionality_reducer
)

# Import optimization utilities
from src.utils.common_operations import (
    memory_monitor, safe_divide, safe_mean, safe_std,
    validate_finite, force_garbage_collection
)
from src.utils.hardware import optimize_dataframe_default, get_memory_usage
from src.utils.math_validation import validate_positive, validate_range
from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error, tprint_performance

logger = logging.getLogger(__name__)

@dataclass
class HDBSCANRegimeOptimizerConfig:
    """Configuration for HDBSCAN regime optimization."""
    # Feature processing configuration
    enable_feature_processing: bool = True
    k_features: int = 50
    selection_method: str = 'hybrid'
    enable_sampling: bool = True
    sample_size: int = 1000
    
    # Dimensionality reduction configuration
    enable_dimensionality_reduction: bool = True
    primary_method: str = 'pca'
    fallback_method: str = 'pca'
    pca_variance_threshold: float = 0.95
    
    # HDBSCAN configuration
    min_cluster_size: int = 20
    min_samples: int = 10
    cluster_selection_epsilon: Optional[float] = None
    metric: str = 'euclidean'
    algorithm: str = 'auto'
    leaf_size: int = 40
    n_jobs: int = -1
    
    # Memory and performance optimization
    memory_efficient: bool = True
    chunk_size: int = 1000
    max_memory_gb: float = 8.0
    enable_vectorbt: bool = True
    enable_gpu: bool = False
    
    # Regime-specific parameters
    regime_detection_method: str = 'volatility'
    n_regime_classes: int = 3
    regime_window: int = 20
    
    # Quality metrics
    enable_quality_metrics: bool = True
    quality_metrics: List[str] = None

    def __post_init__(self):
        if self.quality_metrics is None:
            self.quality_metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']

class HDBSCANRegimeOptimizer:
    """
    Optimized HDBSCAN regime discovery system that integrates feature processing,
    dimensionality reduction, and clustering for efficient regime discovery.
    """
    
    def __init__(self, config: Optional[HDBSCANRegimeOptimizerConfig] = None):
        """Initialize the HDBSCAN regime optimizer."""
        self.config = config or HDBSCANRegimeOptimizerConfig()
        
        # Initialize components
        self._initialize_feature_processor()
        self._initialize_dimensionality_reducer()
        self._initialize_hdbscan_clusterer()
        
        # Performance tracking
        self.performance_stats = {
            'total_optimization_time': 0.0,
            'feature_processing_time': 0.0,
            'dimensionality_reduction_time': 0.0,
            'clustering_time': 0.0,
            'quality_metrics_time': 0.0,
            'final_features_count': 0,
            'reduced_features_count': 0,
            'n_clusters': 0,
            'n_noise_points': 0,
            'memory_usage_mb': 0.0,
            'quality_metrics': {}
        }
        
        tprint_info("✅ HDBSCANRegimeOptimizer initialized")
    
    def _initialize_feature_processor(self):
        """Initialize the optimized regime feature processor."""
        processor_config = OptimizedRegimeFeatureProcessorConfig(
            k_features=self.config.k_features,
            selection_method=self.config.selection_method,
            enable_feature_selection=True,
            enable_sampling=self.config.enable_sampling,
            sample_size=self.config.sample_size,
            memory_efficient=self.config.memory_efficient,
            chunk_size=self.config.chunk_size,
            max_memory_gb=self.config.max_memory_gb,
            enable_vectorbt=self.config.enable_vectorbt,
            enable_gpu=self.config.enable_gpu,
            regime_detection_method=self.config.regime_detection_method,
            n_regime_classes=self.config.n_regime_classes,
            regime_window=self.config.regime_window
        )
        self.feature_processor = OptimizedRegimeFeatureProcessor(processor_config)
    
    def _initialize_dimensionality_reducer(self):
        """Initialize the optimized dimensionality reducer."""
        reducer_config = DimensionalityReductionConfig(
            primary_method=self.config.primary_method,
            fallback_method=self.config.fallback_method,
            pca_variance_threshold=self.config.pca_variance_threshold,
            memory_efficient=self.config.memory_efficient,
            chunk_size=self.config.chunk_size,
            max_memory_gb=self.config.max_memory_gb,
            enable_vectorbt=self.config.enable_vectorbt,
            enable_gpu=self.config.enable_gpu
        )
        self.dimensionality_reducer = OptimizedDimensionalityReducer(reducer_config)
    
    def _initialize_hdbscan_clusterer(self):
        """Initialize the HDBSCAN clusterer."""
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.min_cluster_size,
            min_samples=self.config.min_samples,
            cluster_selection_epsilon=self.config.cluster_selection_epsilon,
            metric=self.config.metric,
            algorithm=self.config.algorithm,
            leaf_size=self.config.leaf_size,
            n_jobs=self.config.n_jobs
        )
    
    def optimize_regime_discovery(self, data: pd.DataFrame, 
                                 symbol: str, 
                                 timeframe: str,
                                 target: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Optimize regime discovery using HDBSCAN with efficient feature processing.
        
        Args:
            data: OHLCV data
            symbol: Trading symbol
            timeframe: Data timeframe
            target: Target variable (optional, will create pseudo-target for regime discovery)
            
        Returns:
            Dictionary with clustering results, regime labels, and performance metrics
        """
        start_time = time.time()
        
        with memory_monitor("HDBSCAN Regime Optimization"):
            tprint_info(f"🚀 Starting optimized regime discovery for {symbol} {timeframe}")
            
            # Step 1: Feature Processing
            if self.config.enable_feature_processing:
                tprint_info("🔄 Step 1: Feature Processing")
                processing_start = time.time()
                features_df = self.feature_processor.process_features(data, symbol, timeframe, target)
                processing_time = time.time() - processing_start
                self.performance_stats['feature_processing_time'] = processing_time
                
                tprint_success(f"✅ Feature processing completed: {features_df.shape[1]} features in {processing_time:.2f}s")
            else:
                # Use raw data as features
                features_df = data.select_dtypes(include=[np.number])
                self.performance_stats['feature_processing_time'] = 0.0
                tprint_info("ℹ️ Feature processing disabled, using raw numeric data")
            
            # Step 2: Dimensionality Reduction
            if self.config.enable_dimensionality_reduction:
                tprint_info("🔄 Step 2: Dimensionality Reduction")
                reduction_start = time.time()
                reduced_features = self.dimensionality_reducer.reduce_dimensions(features_df)
                reduction_time = time.time() - reduction_start
                self.performance_stats['dimensionality_reduction_time'] = reduction_time
                
                tprint_success(f"✅ Dimensionality reduction completed: {reduced_features.shape[1]} features in {reduction_time:.2f}s")
            else:
                reduced_features = features_df
                self.performance_stats['dimensionality_reduction_time'] = 0.0
                tprint_info("ℹ️ Dimensionality reduction disabled")
            
            # Step 3: HDBSCAN Clustering
            tprint_info("🔄 Step 3: HDBSCAN Clustering")
            clustering_start = time.time()
            
            # Prepare data for clustering
            clustering_data = self._prepare_clustering_data(reduced_features)
            
            # Perform clustering
            cluster_labels = self.clusterer.fit_predict(clustering_data)
            
            clustering_time = time.time() - clustering_start
            self.performance_stats['clustering_time'] = clustering_time
            
            # Analyze clustering results
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise_points = list(cluster_labels).count(-1)
            
            self.performance_stats['n_clusters'] = n_clusters
            self.performance_stats['n_noise_points'] = n_noise_points
            
            tprint_success(f"✅ HDBSCAN clustering completed: {n_clusters} clusters, {n_noise_points} noise points in {clustering_time:.2f}s")
            
            # Step 4: Quality Metrics (if enabled)
            quality_metrics = {}
            if self.config.enable_quality_metrics and n_clusters > 1:
                tprint_info("🔄 Step 4: Quality Metrics")
                metrics_start = time.time()
                quality_metrics = self._calculate_quality_metrics(clustering_data, cluster_labels)
                metrics_time = time.time() - metrics_start
                self.performance_stats['quality_metrics_time'] = metrics_time
                
                tprint_success(f"✅ Quality metrics calculated in {metrics_time:.2f}s")
            else:
                self.performance_stats['quality_metrics_time'] = 0.0
                tprint_info("ℹ️ Quality metrics disabled or insufficient clusters")
            
            # Step 5: Create regime labels
            regime_labels = self._create_regime_labels(cluster_labels, data.index)
            
            # Update performance stats
            total_time = time.time() - start_time
            self._update_performance_stats(features_df, reduced_features, total_time, quality_metrics)
            
            # Create results
            results = {
                'regime_labels': regime_labels,
                'cluster_labels': cluster_labels,
                'features_df': features_df,
                'reduced_features_df': reduced_features,
                'clustering_data': clustering_data,
                'n_clusters': n_clusters,
                'n_noise_points': n_noise_points,
                'quality_metrics': quality_metrics,
                'performance_stats': self.performance_stats.copy(),
                'config': self.config
            }
            
            tprint_success(f"✅ Optimized regime discovery completed in {total_time:.2f}s")
            
            return results
    
    def _prepare_clustering_data(self, features_df: pd.DataFrame) -> np.ndarray:
        """Prepare data for clustering with proper scaling and validation."""
        # Remove any remaining NaN values
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')
        
        # Scale features for clustering
        scaler = StandardScaler()
        clustering_data = scaler.fit_transform(features_df)
        
        # Validate data
        validate_finite(clustering_data)
        
        return clustering_data
    
    def _calculate_quality_metrics(self, clustering_data: np.ndarray, 
                                  cluster_labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering quality metrics."""
        quality_metrics = {}
        
        # Filter out noise points for metrics calculation
        valid_mask = cluster_labels != -1
        if valid_mask.sum() < 2:
            tprint_warning("⚠️ Too few valid points for quality metrics")
            return quality_metrics
        
        valid_data = clustering_data[valid_mask]
        valid_labels = cluster_labels[valid_mask]
        
        # Calculate metrics
        if 'silhouette' in self.config.quality_metrics:
            try:
                quality_metrics['silhouette_score'] = silhouette_score(valid_data, valid_labels)
            except Exception as e:
                tprint_warning(f"⚠️ Failed to calculate silhouette score: {e}")
                quality_metrics['silhouette_score'] = 0.0
        
        if 'calinski_harabasz' in self.config.quality_metrics:
            try:
                quality_metrics['calinski_harabasz_score'] = calinski_harabasz_score(valid_data, valid_labels)
            except Exception as e:
                tprint_warning(f"⚠️ Failed to calculate Calinski-Harabasz score: {e}")
                quality_metrics['calinski_harabasz_score'] = 0.0
        
        if 'davies_bouldin' in self.config.quality_metrics:
            try:
                quality_metrics['davies_bouldin_score'] = davies_bouldin_score(valid_data, valid_labels)
            except Exception as e:
                tprint_warning(f"⚠️ Failed to calculate Davies-Bouldin score: {e}")
                quality_metrics['davies_bouldin_score'] = 0.0
        
        return quality_metrics
    
    def _create_regime_labels(self, cluster_labels: np.ndarray, 
                            index: pd.Index) -> pd.Series:
        """Create regime labels from cluster labels."""
        # Convert cluster labels to regime labels
        regime_labels = pd.Series(cluster_labels, index=index)
        
        # Map noise points to a special regime
        regime_labels = regime_labels.replace(-1, -1)  # Keep noise as -1
        
        return regime_labels
    
    def _update_performance_stats(self, features_df: pd.DataFrame, 
                                reduced_features_df: pd.DataFrame,
                                total_time: float,
                                quality_metrics: Dict[str, float]):
        """Update performance statistics."""
        self.performance_stats['total_optimization_time'] = total_time
        self.performance_stats['final_features_count'] = features_df.shape[1]
        self.performance_stats['reduced_features_count'] = reduced_features_df.shape[1]
        self.performance_stats['quality_metrics'] = quality_metrics
        
        # Calculate memory usage
        memory_usage = reduced_features_df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        self.performance_stats['memory_usage_mb'] = memory_usage
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add component stats
        if hasattr(self, 'feature_processor'):
            stats['feature_processor_stats'] = self.feature_processor.get_performance_stats()
        
        if hasattr(self, 'dimensionality_reducer'):
            stats['dimensionality_reducer_stats'] = self.dimensionality_reducer.get_performance_stats()
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_optimization_time': 0.0,
            'feature_processing_time': 0.0,
            'dimensionality_reduction_time': 0.0,
            'clustering_time': 0.0,
            'quality_metrics_time': 0.0,
            'final_features_count': 0,
            'reduced_features_count': 0,
            'n_clusters': 0,
            'n_noise_points': 0,
            'memory_usage_mb': 0.0,
            'quality_metrics': {}
        }
        
        # Reset component stats
        if hasattr(self, 'feature_processor'):
            self.feature_processor.reset_stats()
        
        if hasattr(self, 'dimensionality_reducer'):
            self.dimensionality_reducer.reset_stats()

# Convenience function for easy usage
def create_hdbscan_regime_optimizer(
    k_features: int = 50,
    selection_method: str = 'hybrid',
    min_cluster_size: int = 20,
    min_samples: int = 10,
    enable_dimensionality_reduction: bool = True,
    primary_method: str = 'pca',
    enable_quality_metrics: bool = True,
    memory_efficient: bool = True,
    enable_vectorbt: bool = True,
    enable_gpu: bool = False
) -> HDBSCANRegimeOptimizer:
    """
    Create an HDBSCAN regime optimizer with specified configuration.
    
    Args:
        k_features: Number of features to select
        selection_method: Selection method ('mrmr', 'lasso', 'hybrid')
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Minimum samples for HDBSCAN
        enable_dimensionality_reduction: Enable dimensionality reduction
        primary_method: Primary dimensionality reduction method
        enable_quality_metrics: Enable quality metrics calculation
        memory_efficient: Enable memory optimization
        enable_vectorbt: Enable VectorBT acceleration
        enable_gpu: Enable GPU acceleration
        
    Returns:
        HDBSCANRegimeOptimizer instance
    """
    config = HDBSCANRegimeOptimizerConfig(
        k_features=k_features,
        selection_method=selection_method,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        enable_dimensionality_reduction=enable_dimensionality_reduction,
        primary_method=primary_method,
        enable_quality_metrics=enable_quality_metrics,
        memory_efficient=memory_efficient,
        enable_vectorbt=enable_vectorbt,
        enable_gpu=enable_gpu
    )
    
    return HDBSCANRegimeOptimizer(config)

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(1000) * 0.01),
        'high': 100 + np.cumsum(np.random.randn(1000) * 0.01) + np.abs(np.random.randn(1000) * 0.5),
        'low': 100 + np.cumsum(np.random.randn(1000) * 0.01) - np.abs(np.random.randn(1000) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(1000) * 0.01),
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Create HDBSCAN regime optimizer
    optimizer = create_hdbscan_regime_optimizer(
        k_features=30,
        selection_method='hybrid',
        min_cluster_size=20,
        min_samples=10,
        enable_dimensionality_reduction=True,
        primary_method='pca',
        enable_quality_metrics=True,
        memory_efficient=True,
        enable_vectorbt=True
    )
    
    # Optimize regime discovery
    results = optimizer.optimize_regime_discovery(data, symbol="BTCUSDT", timeframe="15m")
    
    print(f"Regime discovery results:")
    print(f"  - Clusters: {results['n_clusters']}")
    print(f"  - Noise points: {results['n_noise_points']}")
    print(f"  - Quality metrics: {results['quality_metrics']}")
    print(f"  - Performance stats: {results['performance_stats']}")
