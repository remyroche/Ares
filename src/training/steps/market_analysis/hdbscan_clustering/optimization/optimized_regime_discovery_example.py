"""
Optimized HDBSCAN Regime Discovery Example

This module provides comprehensive examples of how to use the optimized
HDBSCAN regime discovery system with all optimization components.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional
import time

# Import the optimized regime discovery
from .optimized_hdbscan_regime_discovery import (
    OptimizedHDBSCANRegimeDiscovery,
    OptimizedRegimeResult,
    OptimizedHDBSCANRegimeDiscoveryConfig,
    create_optimized_hdbscan_regime_discovery
)

logger = logging.getLogger(__name__)

class OptimizedRegimeDiscoveryExample:
    """
    Comprehensive example of optimized HDBSCAN regime discovery.
    
    This class demonstrates how to use all optimization components
    for maximum performance in regime discovery.
    """
    
    def __init__(self):
        """Initialize the example."""
        self.regime_discovery = None
        self.results = {}
        
    def basic_example(self, data: pd.DataFrame) -> OptimizedRegimeResult:
        """
        Basic example of optimized regime discovery.
        
        Args:
            data: Input data for regime discovery
            
        Returns:
            OptimizedRegimeResult with clustering results
        """
        print("=== Basic Optimized Regime Discovery Example ===")
        
        # Create optimized regime discovery with default settings
        self.regime_discovery = create_optimized_hdbscan_regime_discovery()
        
        # Discover regimes
        start_time = time.time()
        result = self.regime_discovery.discover_regimes(data)
        processing_time = time.time() - start_time
        
        # Display results
        print(f"✅ Regime discovery completed in {processing_time:.2f}s")
        print(f"📊 Found {result.n_clusters} clusters")
        print(f"🔍 Noise points: {result.n_noise_points} ({result.noise_ratio:.1%})")
        
        if result.silhouette_score is not None:
            print(f"📈 Silhouette score: {result.silhouette_score:.3f}")
        
        if result.calinski_harabasz_score is not None:
            print(f"📈 Calinski-Harabasz score: {result.calinski_harabasz_score:.3f}")
        
        if result.davies_bouldin_score is not None:
            print(f"📈 Davies-Bouldin score: {result.davies_bouldin_score:.3f}")
        
        return result
    
    def advanced_example(self, data: pd.DataFrame) -> OptimizedRegimeResult:
        """
        Advanced example with custom configuration.
        
        Args:
            data: Input data for regime discovery
            
        Returns:
            OptimizedRegimeResult with clustering results
        """
        print("=== Advanced Optimized Regime Discovery Example ===")
        
        # Create custom configuration
        config = OptimizedHDBSCANRegimeDiscoveryConfig(
            # HDBSCAN parameters
            min_cluster_size=15,
            min_samples=7,
            cluster_selection_epsilon=0.1,
            cluster_selection_method='eom',
            metric='euclidean',
            
            # Optimization settings
            enable_hyperparameter_optimization=True,
            enable_memory_optimization=True,
            enable_vectorized_processing=True,
            enable_features_common=True,
            
            # Feature settings
            enable_entropy_features=True,
            enable_spectral_features=True,
            enable_regime_features=True,
            enable_normalization_features=True,
            enable_feature_selection=True,
            max_features=30,
            
            # Performance settings
            enable_parallel_processing=True,
            max_memory_gb=16.0,
            chunk_size=2000,
            n_jobs=-1,
            
            # Evaluation settings
            primary_metric='silhouette',
            enable_cross_validation=True,
            cv_folds=5
        )
        
        # Create optimized regime discovery
        self.regime_discovery = OptimizedHDBSCANRegimeDiscovery(config)
        
        # Discover regimes
        start_time = time.time()
        result = self.regime_discovery.discover_regimes(data)
        processing_time = time.time() - start_time
        
        # Display detailed results
        print(f"✅ Advanced regime discovery completed in {processing_time:.2f}s")
        print(f"📊 Found {result.n_clusters} clusters")
        print(f"🔍 Noise points: {result.n_noise_points} ({result.noise_ratio:.1%})")
        
        # Performance metrics
        if result.silhouette_score is not None:
            print(f"📈 Silhouette score: {result.silhouette_score:.3f}")
        
        if result.calinski_harabasz_score is not None:
            print(f"📈 Calinski-Harabasz score: {result.calinski_harabasz_score:.3f}")
        
        if result.davies_bouldin_score is not None:
            print(f"📈 Davies-Bouldin score: {result.davies_bouldin_score:.3f}")
        
        # Performance statistics
        stats = self.regime_discovery.get_performance_stats()
        print(f"\n📊 Performance Statistics:")
        print(f"  Total processing time: {stats['total_processing_time']:.2f}s")
        print(f"  Feature generation time: {stats['feature_generation_time']:.2f}s")
        print(f"  Hyperparameter optimization time: {stats['hyperparameter_optimization_time']:.2f}s")
        print(f"  Clustering time: {stats['clustering_time']:.2f}s")
        print(f"  Post-processing time: {stats['post_processing_time']:.2f}s")
        
        # Memory optimizations
        if 'memory_optimizer_stats' in stats:
            memory_stats = stats['memory_optimizer_stats']
            print(f"  Memory optimizations: {memory_stats.get('memory_optimizations', 0)}")
            print(f"  Memory savings: {memory_stats.get('memory_savings_mb', 0):.2f}MB")
        
        # Vectorized operations
        if 'vectorized_processor_stats' in stats:
            vectorized_stats = stats['vectorized_processor_stats']
            print(f"  Vectorized operations: {vectorized_stats.get('vectorized_operations', 0)}")
            print(f"  VectorBT usage rate: {vectorized_stats.get('vectorbt_usage_rate', 0):.1%}")
        
        # Features common integration
        if 'features_common_stats' in stats:
            features_common_stats = stats['features_common_stats']
            print(f"  Features common operations: {features_common_stats.get('vectorized_operations', 0)}")
            print(f"  Normalization operations: {features_common_stats.get('normalization_operations', 0)}")
        
        return result
    
    def performance_comparison_example(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Performance comparison between different optimization levels.
        
        Args:
            data: Input data for regime discovery
            
        Returns:
            Dictionary with performance comparison results
        """
        print("=== Performance Comparison Example ===")
        
        results = {}
        
        # Test 1: No optimization
        print("\n🧪 Testing without optimization...")
        config_no_opt = OptimizedHDBSCANRegimeDiscoveryConfig(
            enable_hyperparameter_optimization=False,
            enable_memory_optimization=False,
            enable_vectorized_processing=False,
            enable_features_common=False,
            enable_feature_selection=False
        )
        
        regime_discovery_no_opt = OptimizedHDBSCANRegimeDiscovery(config_no_opt)
        start_time = time.time()
        result_no_opt = regime_discovery_no_opt.discover_regimes(data)
        time_no_opt = time.time() - start_time
        
        results['no_optimization'] = {
            'time': time_no_opt,
            'n_clusters': result_no_opt.n_clusters,
            'silhouette_score': result_no_opt.silhouette_score,
            'processing_time': result_no_opt.processing_time
        }
        
        print(f"  Time: {time_no_opt:.2f}s, Clusters: {result_no_opt.n_clusters}")
        
        # Test 2: Basic optimization
        print("\n🧪 Testing with basic optimization...")
        config_basic = OptimizedHDBSCANRegimeDiscoveryConfig(
            enable_hyperparameter_optimization=True,
            enable_memory_optimization=True,
            enable_vectorized_processing=False,
            enable_features_common=False,
            enable_feature_selection=False
        )
        
        regime_discovery_basic = OptimizedHDBSCANRegimeDiscovery(config_basic)
        start_time = time.time()
        result_basic = regime_discovery_basic.discover_regimes(data)
        time_basic = time.time() - start_time
        
        results['basic_optimization'] = {
            'time': time_basic,
            'n_clusters': result_basic.n_clusters,
            'silhouette_score': result_basic.silhouette_score,
            'processing_time': result_basic.processing_time
        }
        
        print(f"  Time: {time_basic:.2f}s, Clusters: {result_basic.n_clusters}")
        
        # Test 3: Full optimization
        print("\n🧪 Testing with full optimization...")
        config_full = OptimizedHDBSCANRegimeDiscoveryConfig(
            enable_hyperparameter_optimization=True,
            enable_memory_optimization=True,
            enable_vectorized_processing=True,
            enable_features_common=True,
            enable_feature_selection=True
        )
        
        regime_discovery_full = OptimizedHDBSCANRegimeDiscovery(config_full)
        start_time = time.time()
        result_full = regime_discovery_full.discover_regimes(data)
        time_full = time.time() - start_time
        
        results['full_optimization'] = {
            'time': time_full,
            'n_clusters': result_full.n_clusters,
            'silhouette_score': result_full.silhouette_score,
            'processing_time': result_full.processing_time
        }
        
        print(f"  Time: {time_full:.2f}s, Clusters: {result_full.n_clusters}")
        
        # Calculate improvements
        basic_improvement = (time_no_opt - time_basic) / time_no_opt * 100
        full_improvement = (time_no_opt - time_full) / time_no_opt * 100
        
        print(f"\n📊 Performance Improvements:")
        print(f"  Basic optimization: {basic_improvement:.1f}% faster")
        print(f"  Full optimization: {full_improvement:.1f}% faster")
        
        results['improvements'] = {
            'basic_improvement': basic_improvement,
            'full_improvement': full_improvement
        }
        
        return results
    
    def feature_importance_example(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Example of feature importance analysis.
        
        Args:
            data: Input data for regime discovery
            
        Returns:
            Dictionary with feature importance scores
        """
        print("=== Feature Importance Analysis Example ===")
        
        # Create optimized regime discovery
        self.regime_discovery = create_optimized_hdbscan_regime_discovery(
            enable_feature_selection=True,
            max_features=20
        )
        
        # Discover regimes
        result = self.regime_discovery.discover_regimes(data)
        
        # Display feature importance
        if result.feature_importance:
            print(f"\n📊 Feature Importance (Top 10):")
            sorted_features = sorted(
                result.feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            for feature, importance in sorted_features:
                print(f"  {feature}: {importance:.3f}")
        
        return result.feature_importance or {}
    
    def memory_optimization_example(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Example of memory optimization benefits.
        
        Args:
            data: Input data for regime discovery
            
        Returns:
            Dictionary with memory optimization results
        """
        print("=== Memory Optimization Example ===")
        
        # Test with memory optimization
        config_with_memory = OptimizedHDBSCANRegimeDiscoveryConfig(
            enable_memory_optimization=True,
            max_memory_gb=4.0,
            chunk_size=500
        )
        
        regime_discovery_with_memory = OptimizedHDBSCANRegimeDiscovery(config_with_memory)
        result_with_memory = regime_discovery_with_memory.discover_regimes(data)
        
        # Test without memory optimization
        config_without_memory = OptimizedHDBSCANRegimeDiscoveryConfig(
            enable_memory_optimization=False,
            max_memory_gb=8.0,
            chunk_size=1000
        )
        
        regime_discovery_without_memory = OptimizedHDBSCANRegimeDiscovery(config_without_memory)
        result_without_memory = regime_discovery_without_memory.discover_regimes(data)
        
        # Compare memory usage
        stats_with = regime_discovery_with_memory.get_performance_stats()
        stats_without = regime_discovery_without_memory.get_performance_stats()
        
        print(f"\n📊 Memory Optimization Results:")
        
        if 'memory_optimizer_stats' in stats_with:
            memory_stats = stats_with['memory_optimizer_stats']
            print(f"  Memory optimizations: {memory_stats.get('memory_optimizations', 0)}")
            print(f"  Memory savings: {memory_stats.get('memory_savings_mb', 0):.2f}MB")
            print(f"  Peak memory usage: {memory_stats.get('peak_memory_mb', 0):.2f}MB")
        
        return {
            'with_memory_optimization': stats_with,
            'without_memory_optimization': stats_without
        }

# Convenience function
def run_optimized_regime_discovery_example(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Run comprehensive optimized regime discovery example.
    
    Args:
        data: Input data for regime discovery
        
    Returns:
        Dictionary with all example results
    """
    example = OptimizedRegimeDiscoveryExample()
    
    results = {}
    
    # Basic example
    results['basic'] = example.basic_example(data)
    
    # Advanced example
    results['advanced'] = example.advanced_example(data)
    
    # Performance comparison
    results['performance_comparison'] = example.performance_comparison_example(data)
    
    # Feature importance
    results['feature_importance'] = example.feature_importance_example(data)
    
    # Memory optimization
    results['memory_optimization'] = example.memory_optimization_example(data)
    
    return results

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Generate sample financial data
    data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Add some regime-like structure
    data['regime_1'] = np.sin(np.arange(n_samples) * 0.1) + np.random.randn(n_samples) * 0.1
    data['regime_2'] = np.cos(np.arange(n_samples) * 0.05) + np.random.randn(n_samples) * 0.1
    
    print("🚀 Starting Optimized HDBSCAN Regime Discovery Examples")
    print(f"📊 Data shape: {data.shape}")
    
    # Run examples
    results = run_optimized_regime_discovery_example(data)
    
    print("\n✅ All examples completed successfully!")
    print(f"📊 Results summary:")
    print(f"  Basic clusters: {results['basic'].n_clusters}")
    print(f"  Advanced clusters: {results['advanced'].n_clusters}")
    print(f"  Performance improvement: {results['performance_comparison']['improvements']['full_improvement']:.1f}%")
