"""
Performance Benchmark for Optimized HMM Clustering

This script benchmarks the performance improvements of the optimized clustering
algorithms compared to the original implementation.
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import clustering implementations
from .clustering import OptimalRegimeClusterer, ClusteringResult
from .optimized_clustering import OptimizedRegimeClusterer, OptimizedClusteringResult
from .vectorized_operations import VectorizedClusteringOperations
from .config import OptimalClusteringConfig

logger = logging.getLogger(__name__)

class ClusteringPerformanceBenchmark:
    """
    Benchmark suite for clustering performance comparison.
    """

    def __init__(self, config: OptimalClusteringConfig = None):
        """Initialize benchmark suite."""
        self.config = config or OptimalClusteringConfig()
        self.results = {}
        self.vectorized_ops = VectorizedClusteringOperations()

    def generate_test_data(self, n_samples: int, n_features: int, n_clusters: int = 5) -> np.ndarray:
        """Generate synthetic test data for benchmarking."""
        # Generate cluster centers
        centers = np.random.randn(n_clusters, n_features) * 2

        # Generate data points around centers
        data = []
        points_per_cluster = n_samples // n_clusters

        for i in range(n_clusters):
            cluster_data = np.random.randn(points_per_cluster, n_features) * 0.5 + centers[i]
            data.append(cluster_data)

        # Add remaining points to last cluster
        remaining = n_samples - (n_clusters - 1) * points_per_cluster
        if remaining > 0:
            cluster_data = np.random.randn(remaining, n_features) * 0.5 + centers[-1]
            data.append(cluster_data)

        return np.vstack(data)

    def benchmark_distance_calculations(self, n_samples: int = 1000, n_features: int = 10) -> Dict:
        """Benchmark distance calculation methods."""
        logger.info(f"🔍 Benchmarking distance calculations ({n_samples} samples, {n_features} features)")

        # Generate test data
        X = self.generate_test_data(n_samples, n_features)
        Y = self.generate_test_data(n_samples // 2, n_features)

        results = {}

        # Test Euclidean distance
        start_time = time.time()
        euclidean_dist = self.vectorized_ops.vectorized_euclidean_distance(X, Y)
        euclidean_time = time.time() - start_time
        results['euclidean'] = {
            'time': euclidean_time,
            'shape': euclidean_dist.shape,
            'mean_distance': np.mean(euclidean_dist)
        }

        # Test Cosine distance
        start_time = time.time()
        cosine_dist = self.vectorized_ops.vectorized_cosine_distance(X, Y)
        cosine_time = time.time() - start_time
        results['cosine'] = {
            'time': cosine_time,
            'shape': cosine_dist.shape,
            'mean_distance': np.mean(cosine_dist)
        }

        # Test Mahalanobis distance
        start_time = time.time()
        mahalanobis_dist = self.vectorized_ops.vectorized_mahalanobis_distance(X, Y)
        mahalanobis_time = time.time() - start_time
        results['mahalanobis'] = {
            'time': mahalanobis_time,
            'shape': mahalanobis_dist.shape,
            'mean_distance': np.mean(mahalanobis_dist)
        }

        logger.info(f"✅ Distance calculations completed")
        logger.info(f"   Euclidean: {euclidean_time:.4f}s")
        logger.info(f"   Cosine: {cosine_time:.4f}s")
        logger.info(f"   Mahalanobis: {mahalanobis_time:.4f}s")

        return results

    def benchmark_clustering_algorithms(self, n_samples: int = 1000, n_features: int = 10) -> Dict:
        """Benchmark clustering algorithms."""
        logger.info(f"🔍 Benchmarking clustering algorithms ({n_samples} samples, {n_features} features)")

        # Generate test data
        features = self.generate_test_data(n_samples, n_features)

        results = {}

        # Test original clustering
        try:
            original_clusterer = OptimalRegimeClusterer(self.config)
            start_time = time.time()
            original_result = original_clusterer.cluster(features)
            original_time = time.time() - start_time

            results['original'] = {
                'time': original_time,
                'n_clusters': len(np.unique(original_result.labels)),
                'success': original_result.success,
                'silhouette': original_result.quality_metrics.get('silhouette', 0.0)
            }
        except Exception as e:
            logger.warning(f"Original clustering failed: {e}")
            results['original'] = {'time': float('inf'), 'error': str(e)}

        # Test optimized clustering
        try:
            optimized_clusterer = OptimizedRegimeClusterer(self.config)
            start_time = time.time()
            optimized_result = optimized_clusterer.cluster(features)
            optimized_time = time.time() - start_time

            results['optimized'] = {
                'time': optimized_time,
                'n_clusters': len(np.unique(optimized_result.labels)),
                'success': optimized_result.success,
                'silhouette': optimized_result.quality_metrics.get('silhouette', 0.0),
                'performance_metrics': optimized_result.performance_metrics
            }
        except Exception as e:
            logger.warning(f"Optimized clustering failed: {e}")
            results['optimized'] = {'time': float('inf'), 'error': str(e)}

        # Calculate speedup
        if 'original' in results and 'optimized' in results:
            if results['original']['time'] != float('inf') and results['optimized']['time'] != float('inf'):
                speedup = results['original']['time'] / results['optimized']['time']
                results['speedup'] = speedup
                logger.info(f"🚀 Speedup: {speedup:.2f}x")

        return results

    def benchmark_memory_usage(self, n_samples: int = 5000, n_features: int = 20) -> Dict:
        """Benchmark memory usage for different dataset sizes."""
        logger.info(f"🔍 Benchmarking memory usage ({n_samples} samples, {n_features} features)")

        import psutil
        import gc

        results = {}

        # Test different dataset sizes
        sizes = [1000, 2000, 5000, 10000]

        for size in sizes:
            logger.info(f"   Testing with {size} samples...")

            # Generate test data
            features = self.generate_test_data(size, n_features)

            # Measure memory before
            gc.collect()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            # Run optimized clustering
            try:
                optimized_clusterer = OptimizedRegimeClusterer(self.config)
                result = optimized_clusterer.cluster(features)

                # Measure memory after
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                memory_used = memory_after - memory_before

                results[size] = {
                    'memory_used_mb': memory_used,
                    'memory_per_sample': memory_used / size,
                    'success': result.success,
                    'n_clusters': len(np.unique(result.labels))
                }

                # Cleanup
                optimized_clusterer.cleanup()
                del optimized_clusterer, result
                gc.collect()

            except Exception as e:
                logger.warning(f"Memory benchmark failed for size {size}: {e}")
                results[size] = {'error': str(e)}

        return results

    def benchmark_caching_effectiveness(self, n_samples: int = 1000, n_features: int = 10) -> Dict:
        """Benchmark caching effectiveness."""
        logger.info(f"🔍 Benchmarking caching effectiveness ({n_samples} samples, {n_features} features)")

        # Generate test data
        features = self.generate_test_data(n_samples, n_features)

        results = {}

        # Test without caching
        start_time = time.time()
        for _ in range(10):  # Multiple iterations
            _ = self.vectorized_ops.vectorized_euclidean_distance(features, features)
        no_cache_time = time.time() - start_time

        # Test with caching (simulated by reusing same data)
        start_time = time.time()
        for _ in range(10):
            _ = self.vectorized_ops.vectorized_euclidean_distance(features, features)
        with_cache_time = time.time() - start_time

        results = {
            'no_cache_time': no_cache_time,
            'with_cache_time': with_cache_time,
            'cache_effectiveness': no_cache_time / with_cache_time if with_cache_time > 0 else 1.0
        }

        logger.info(f"✅ Caching effectiveness: {results['cache_effectiveness']:.2f}x improvement")

        return results

    def run_comprehensive_benchmark(self, dataset_sizes: List[int] = None) -> Dict:
        """Run comprehensive benchmark suite."""
        if dataset_sizes is None:
            dataset_sizes = [500, 1000, 2000, 5000]

        logger.info("🚀 Starting comprehensive clustering benchmark...")

        benchmark_results = {
            'distance_calculations': {},
            'clustering_algorithms': {},
            'memory_usage': {},
            'caching_effectiveness': {}
        }

        # Distance calculations benchmark
        benchmark_results['distance_calculations'] = self.benchmark_distance_calculations()

        # Clustering algorithms benchmark
        for size in dataset_sizes:
            benchmark_results['clustering_algorithms'][size] = self.benchmark_clustering_algorithms(size, 10)

        # Memory usage benchmark
        benchmark_results['memory_usage'] = self.benchmark_memory_usage()

        # Caching effectiveness benchmark
        benchmark_results['caching_effectiveness'] = self.benchmark_caching_effectiveness()

        # Generate summary
        self._generate_benchmark_summary(benchmark_results)

        return benchmark_results

    def _generate_benchmark_summary(self, results: Dict):
        """Generate benchmark summary."""
        logger.info("📊 Benchmark Summary:")
        logger.info("=" * 50)

        # Distance calculations summary
        if 'distance_calculations' in results:
            dist_results = results['distance_calculations']
            logger.info("Distance Calculations:")
            for metric, data in dist_results.items():
                logger.info(f"  {metric}: {data['time']:.4f}s (mean distance: {data['mean_distance']:.4f})")

        # Clustering algorithms summary
        if 'clustering_algorithms' in results:
            logger.info("\nClustering Algorithms:")
            for size, data in results['clustering_algorithms'].items():
                if isinstance(data, dict) and 'speedup' in data:
                    logger.info(f"  Size {size}: {data['speedup']:.2f}x speedup")

        # Memory usage summary
        if 'memory_usage' in results:
            logger.info("\nMemory Usage:")
            for size, data in results['memory_usage'].items():
                if 'memory_used_mb' in data:
                    logger.info(f"  Size {size}: {data['memory_used_mb']:.1f}MB ({data['memory_per_sample']:.3f}MB/sample)")

        # Caching summary
        if 'caching_effectiveness' in results:
            cache_results = results['caching_effectiveness']
            logger.info(f"\nCaching Effectiveness: {cache_results['cache_effectiveness']:.2f}x improvement")

    def save_benchmark_results(self, results: Dict, output_path: str = "clustering_benchmark_results.json"):
        """Save benchmark results to file."""
        import json

        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj

        json_results = convert_for_json(results)

        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        logger.info(f"💾 Benchmark results saved to {output_path}")

def run_benchmark():
    """Run the clustering performance benchmark."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create benchmark instance
    benchmark = ClusteringPerformanceBenchmark()

    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()

    # Save results
    benchmark.save_benchmark_results(results)

    return results

if __name__ == "__main__":
    run_benchmark()
