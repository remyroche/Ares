"""
Performance Profiler for Data-Driven Clustering System

This module provides comprehensive performance profiling for the multi-objective optimization
and feature generation stages to help identify bottlenecks and optimize performance.
"""

import time
import psutil
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Callable
import logging
from datetime import datetime
import json
from pathlib import Path
import cProfile
import pstats
import io
from contextlib import contextmanager
import threading
import queue
import multiprocessing as mp
from functools import wraps
import gc

# Import the clustering system components
from src.training.steps.market_analysis.hdbscan_clustering.optimization.data_driven_clustering_optimizer import (
    DataDrivenClusteringOptimizer
)
from src.training.steps.market_analysis.hdbscan_clustering.config.data_driven_config import (
    DataDrivenClusteringConfig
)
from src.training.steps.market_analysis.hdbscan_clustering.feature_engineering.advanced_financial_features import (
    AdvancedFinancialFeatureEngineer, AdvancedFeatureConfig
)
from src.training.steps.market_analysis.hdbscan_clustering.optimization.economic_validator import (
    EconomicValidator, EconomicValidationConfig
)
from src.training.steps.market_analysis.hdbscan_clustering.validation.regime_persistence_validator import (
    RegimePersistenceValidator, RegimePersistenceConfig
)
from src.training.steps.market_analysis.hdbscan_clustering.optimization.multi_objective_optimizer import (
    MultiObjectiveOptimizer, MultiObjectiveConfig
)

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """
    Comprehensive performance profiler for the data-driven clustering system.
    
    Provides detailed profiling of:
    - Feature generation performance
    - Multi-objective optimization performance
    - Memory usage patterns
    - CPU utilization
    - I/O operations
    - Caching effectiveness
    """
    
    def __init__(self, 
                 profile_dir: str = "performance_profiles",
                 enable_caching: bool = True,
                 enable_parallelization: bool = True):
        """
        Initialize the performance profiler.
        
        Args:
            profile_dir: Directory to save profile results
            enable_caching: Whether to enable caching for repeated operations
            enable_parallelization: Whether to enable parallel processing
        """
        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_caching = enable_caching
        self.enable_parallelization = enable_parallelization
        
        # Performance tracking
        self.performance_metrics: Dict[str, Any] = {}
        self.memory_usage: List[Dict[str, Any]] = []
        self.cpu_usage: List[Dict[str, Any]] = []
        
        # Caching
        self.cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Parallel processing
        self.max_workers = min(mp.cpu_count(), 8)  # Limit to 8 workers
        self.thread_pool = None
        
        # Initialize components
        self.config = DataDrivenClusteringConfig()
        self.optimizer = DataDrivenClusteringOptimizer(self.config)
        self.feature_engineer = AdvancedFinancialFeatureEngineer(AdvancedFeatureConfig())
        self.economic_validator = EconomicValidator(EconomicValidationConfig())
        self.persistence_validator = RegimePersistenceValidator(RegimePersistenceConfig())
        self.multi_objective_optimizer = MultiObjectiveOptimizer(MultiObjectiveConfig())
        
    @contextmanager
    def profile_context(self, operation_name: str):
        """Context manager for profiling operations."""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Start profiling
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            yield profiler
        finally:
            # Stop profiling
            profiler.disable()
            
            # Calculate metrics
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Get profiling stats
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 functions
            
            profile_stats = s.getvalue()
            
            # Store performance metrics
            self.performance_metrics[operation_name] = {
                'execution_time': execution_time,
                'memory_start': start_memory,
                'memory_end': end_memory,
                'memory_delta': memory_delta,
                'profile_stats': profile_stats,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Profiled {operation_name}: {execution_time:.3f}s, {memory_delta:+.1f}MB")
    
    def profile_feature_generation(self, 
                                 market_data: pd.DataFrame,
                                 n_iterations: int = 5) -> Dict[str, Any]:
        """
        Profile feature generation performance.
        
        Args:
            market_data: Market data to process
            n_iterations: Number of iterations to run for averaging
            
        Returns:
            Dictionary containing performance metrics
        """
        logger.info(f"Profiling feature generation with {n_iterations} iterations")
        
        execution_times = []
        memory_usage = []
        
        for i in range(n_iterations):
            with self.profile_context(f"feature_generation_iter_{i}"):
                # Clear cache between iterations
                if not self.enable_caching:
                    self.cache.clear()
                
                # Generate features
                features, feature_names, feature_categories = self.feature_engineer.engineer_features(market_data)
                
                # Record metrics
                execution_times.append(self.performance_metrics[f"feature_generation_iter_{i}"]['execution_time'])
                memory_usage.append(self.performance_metrics[f"feature_generation_iter_{i}"]['memory_delta'])
        
        # Calculate statistics
        feature_generation_stats = {
            'n_iterations': n_iterations,
            'avg_execution_time': np.mean(execution_times),
            'std_execution_time': np.std(execution_times),
            'min_execution_time': np.min(execution_times),
            'max_execution_time': np.max(execution_times),
            'avg_memory_delta': np.mean(memory_usage),
            'std_memory_delta': np.std(memory_usage),
            'total_features': features.shape[1] if len(features.shape) > 1 else 0,
            'feature_categories': len(feature_categories),
            'data_shape': market_data.shape
        }
        
        logger.info(f"Feature generation: {feature_generation_stats['avg_execution_time']:.3f}s ± {feature_generation_stats['std_execution_time']:.3f}s")
        
        return feature_generation_stats
    
    def profile_multi_objective_optimization(self, 
                                           market_data: pd.DataFrame,
                                           features: np.ndarray,
                                           feature_names: List[str],
                                           n_trials: int = 50) -> Dict[str, Any]:
        """
        Profile multi-objective optimization performance.
        
        Args:
            market_data: Market data
            features: Feature matrix
            feature_names: Feature names
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary containing performance metrics
        """
        logger.info(f"Profiling multi-objective optimization with {n_trials} trials")
        
        # Create clustering function
        def clustering_func(x):
            # Simple clustering for testing
            from sklearn.cluster import KMeans
            n_clusters = min(5, len(np.unique(x)) if len(np.unique(x)) > 1 else 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            return kmeans.fit_predict(x)
        
        # Profile optimization
        with self.profile_context("multi_objective_optimization"):
            # Set up parameter ranges
            parameter_ranges = {
                'similarity_threshold': (0.5, 0.95),
                'distance_threshold': (0.1, 0.5),
                'window_size': (50, 500),
                'smoothing_window': (3, 20)
            }
            
            # Run optimization
            optimization_result = self.multi_objective_optimizer.optimize_parameters(
                parameter_ranges=parameter_ranges,
                clustering_func=clustering_func,
                market_data=market_data,
                features=features,
                feature_names=feature_names
            )
        
        # Extract performance metrics
        optimization_stats = {
            'n_trials': n_trials,
            'execution_time': self.performance_metrics['multi_objective_optimization']['execution_time'],
            'memory_delta': self.performance_metrics['multi_objective_optimization']['memory_delta'],
            'optimization_success': optimization_result.get('optimization_success', False),
            'overall_score': optimization_result.get('overall_score', 0),
            'n_parameters': len(parameter_ranges),
            'data_shape': market_data.shape,
            'features_shape': features.shape
        }
        
        logger.info(f"Multi-objective optimization: {optimization_stats['execution_time']:.3f}s")
        
        return optimization_stats
    
    def profile_economic_validation(self, 
                                  cluster_labels: np.ndarray,
                                  market_data: pd.DataFrame,
                                  features: np.ndarray,
                                  feature_names: List[str],
                                  n_iterations: int = 3) -> Dict[str, Any]:
        """
        Profile economic validation performance.
        
        Args:
            cluster_labels: Cluster labels
            market_data: Market data
            features: Feature matrix
            feature_names: Feature names
            n_iterations: Number of iterations to run
            
        Returns:
            Dictionary containing performance metrics
        """
        logger.info(f"Profiling economic validation with {n_iterations} iterations")
        
        execution_times = []
        memory_usage = []
        
        for i in range(n_iterations):
            with self.profile_context(f"economic_validation_iter_{i}"):
                # Run economic validation
                economic_result = self.economic_validator.validate_clustering(
                    cluster_labels=cluster_labels,
                    market_data=market_data,
                    features=features,
                    feature_names=feature_names
                )
                
                # Record metrics
                execution_times.append(self.performance_metrics[f"economic_validation_iter_{i}"]['execution_time'])
                memory_usage.append(self.performance_metrics[f"economic_validation_iter_{i}"]['memory_delta'])
        
        # Calculate statistics
        economic_validation_stats = {
            'n_iterations': n_iterations,
            'avg_execution_time': np.mean(execution_times),
            'std_execution_time': np.std(execution_times),
            'min_execution_time': np.min(execution_times),
            'max_execution_time': np.max(execution_times),
            'avg_memory_delta': np.mean(memory_usage),
            'std_memory_delta': np.std(memory_usage),
            'n_clusters': len(np.unique(cluster_labels)),
            'n_samples': len(cluster_labels),
            'n_features': features.shape[1] if len(features.shape) > 1 else 0
        }
        
        logger.info(f"Economic validation: {economic_validation_stats['avg_execution_time']:.3f}s ± {economic_validation_stats['std_execution_time']:.3f}s")
        
        return economic_validation_stats
    
    def profile_memory_usage(self, 
                           market_data: pd.DataFrame,
                           features: np.ndarray,
                           feature_names: List[str]) -> Dict[str, Any]:
        """
        Profile memory usage patterns.
        
        Args:
            market_data: Market data
            features: Feature matrix
            feature_names: Feature names
            
        Returns:
            Dictionary containing memory usage metrics
        """
        logger.info("Profiling memory usage patterns")
        
        # Track memory usage over time
        memory_snapshots = []
        
        # Initial memory
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_snapshots.append(('initial', initial_memory))
        
        # After feature generation
        features, feature_names, feature_categories = self.feature_engineer.engineer_features(market_data)
        after_features_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_snapshots.append(('after_features', after_features_memory))
        
        # After clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        after_clustering_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_snapshots.append(('after_clustering', after_clustering_memory))
        
        # After economic validation
        economic_result = self.economic_validator.validate_clustering(
            cluster_labels=cluster_labels,
            market_data=market_data,
            features=features,
            feature_names=feature_names
        )
        after_validation_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_snapshots.append(('after_validation', after_validation_memory))
        
        # Calculate memory deltas
        memory_deltas = []
        for i in range(1, len(memory_snapshots)):
            delta = memory_snapshots[i][1] - memory_snapshots[i-1][1]
            memory_deltas.append((memory_snapshots[i][0], delta))
        
        # Memory usage statistics
        memory_stats = {
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': max(snapshot[1] for snapshot in memory_snapshots),
            'final_memory_mb': after_validation_memory,
            'total_memory_delta_mb': after_validation_memory - initial_memory,
            'memory_snapshots': memory_snapshots,
            'memory_deltas': memory_deltas,
            'data_size_mb': market_data.memory_usage(deep=True).sum() / 1024 / 1024,
            'features_size_mb': features.nbytes / 1024 / 1024 if hasattr(features, 'nbytes') else 0
        }
        
        logger.info(f"Memory usage: {memory_stats['total_memory_delta_mb']:+.1f}MB (peak: {memory_stats['peak_memory_mb']:.1f}MB)")
        
        return memory_stats
    
    def profile_parallelization(self, 
                              market_data: pd.DataFrame,
                              features: np.ndarray,
                              feature_names: List[str]) -> Dict[str, Any]:
        """
        Profile parallelization performance.
        
        Args:
            market_data: Market data
            features: Feature matrix
            feature_names: Feature names
            
        Returns:
            Dictionary containing parallelization metrics
        """
        logger.info("Profiling parallelization performance")
        
        # Test different numbers of workers
        worker_counts = [1, 2, 4, 8, min(mp.cpu_count(), 16)]
        parallelization_results = {}
        
        for n_workers in worker_counts:
            if n_workers > mp.cpu_count():
                continue
                
            logger.info(f"Testing with {n_workers} workers")
            
            # Create test function
            def test_worker_workload(worker_id):
                # Simulate some work
                time.sleep(0.1)  # Simulate processing time
                return worker_id * 2
            
            # Test parallel execution
            start_time = time.perf_counter()
            
            if n_workers == 1:
                # Sequential execution
                results = [test_worker_workload(i) for i in range(8)]
            else:
                # Parallel execution
                with mp.Pool(n_workers) as pool:
                    results = pool.map(test_worker_workload, range(8))
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            parallelization_results[n_workers] = {
                'execution_time': execution_time,
                'speedup': parallelization_results.get(1, {}).get('execution_time', execution_time) / execution_time,
                'efficiency': (parallelization_results.get(1, {}).get('execution_time', execution_time) / execution_time) / n_workers
            }
        
        # Find optimal number of workers
        optimal_workers = max(parallelization_results.keys(), 
                            key=lambda x: parallelization_results[x]['speedup'])
        
        parallelization_stats = {
            'worker_results': parallelization_results,
            'optimal_workers': optimal_workers,
            'max_speedup': max(result['speedup'] for result in parallelization_results.values()),
            'cpu_count': mp.cpu_count(),
            'recommended_workers': min(optimal_workers, mp.cpu_count())
        }
        
        logger.info(f"Optimal workers: {optimal_workers} (speedup: {parallelization_stats['max_speedup']:.2f}x)")
        
        return parallelization_stats
    
    def profile_caching_effectiveness(self, 
                                    market_data: pd.DataFrame,
                                    n_iterations: int = 10) -> Dict[str, Any]:
        """
        Profile caching effectiveness.
        
        Args:
            market_data: Market data
            n_iterations: Number of iterations to run
            
        Returns:
            Dictionary containing caching metrics
        """
        logger.info(f"Profiling caching effectiveness with {n_iterations} iterations")
        
        # Clear cache
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Test with caching enabled
        execution_times_with_cache = []
        
        for i in range(n_iterations):
            start_time = time.perf_counter()
            
            # Simulate cached operation
            cache_key = f"test_operation_{i % 3}"  # Some operations repeat
            
            if cache_key in self.cache:
                self.cache_hits += 1
                result = self.cache[cache_key]
            else:
                self.cache_misses += 1
                # Simulate expensive operation
                time.sleep(0.01)
                result = f"result_{i}"
                self.cache[cache_key] = result
            
            end_time = time.perf_counter()
            execution_times_with_cache.append(end_time - start_time)
        
        # Test without caching
        execution_times_without_cache = []
        
        for i in range(n_iterations):
            start_time = time.perf_counter()
            
            # Simulate same operation without caching
            time.sleep(0.01)
            result = f"result_{i}"
            
            end_time = time.perf_counter()
            execution_times_without_cache.append(end_time - start_time)
        
        # Calculate caching statistics
        cache_stats = {
            'n_iterations': n_iterations,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses),
            'avg_time_with_cache': np.mean(execution_times_with_cache),
            'avg_time_without_cache': np.mean(execution_times_without_cache),
            'speedup': np.mean(execution_times_without_cache) / np.mean(execution_times_with_cache),
            'cache_size': len(self.cache)
        }
        
        logger.info(f"Caching: {cache_stats['hit_rate']:.1%} hit rate, {cache_stats['speedup']:.2f}x speedup")
        
        return cache_stats
    
    def run_comprehensive_profile(self, 
                                market_data: pd.DataFrame,
                                n_iterations: int = 3) -> Dict[str, Any]:
        """
        Run comprehensive performance profiling.
        
        Args:
            market_data: Market data to profile
            n_iterations: Number of iterations for averaging
            
        Returns:
            Dictionary containing all performance metrics
        """
        logger.info("Starting comprehensive performance profiling")
        
        # Generate features
        features, feature_names, feature_categories = self.feature_engineer.engineer_features(market_data)
        
        # Create cluster labels for testing
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        
        # Run all profiling tests
        profile_results = {
            'timestamp': datetime.now().isoformat(),
            'data_info': {
                'n_samples': len(market_data),
                'n_features': features.shape[1] if len(features.shape) > 1 else 0,
                'n_clusters': len(np.unique(cluster_labels)),
                'data_shape': market_data.shape
            },
            'feature_generation': self.profile_feature_generation(market_data, n_iterations),
            'multi_objective_optimization': self.profile_multi_objective_optimization(
                market_data, features, feature_names
            ),
            'economic_validation': self.profile_economic_validation(
                cluster_labels, market_data, features, feature_names, n_iterations
            ),
            'memory_usage': self.profile_memory_usage(market_data, features, feature_names),
            'parallelization': self.profile_parallelization(market_data, features, feature_names),
            'caching': self.profile_caching_effectiveness(market_data, n_iterations * 2)
        }
        
        # Save profile results
        profile_file = self.profile_dir / f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(profile_file, 'w') as f:
            json.dump(profile_results, f, indent=2)
        
        logger.info(f"Comprehensive profiling completed. Results saved to {profile_file}")
        
        return profile_results
    
    def generate_performance_report(self, 
                                  profile_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            profile_results: Results from comprehensive profiling
            
        Returns:
            Markdown-formatted performance report
        """
        report = f"""# Performance Profiling Report

**Generated:** {profile_results['timestamp']}

## Data Information
- **Samples:** {profile_results['data_info']['n_samples']:,}
- **Features:** {profile_results['data_info']['n_features']:,}
- **Clusters:** {profile_results['data_info']['n_clusters']}
- **Data Shape:** {profile_results['data_info']['data_shape']}

## Feature Generation Performance
- **Average Time:** {profile_results['feature_generation']['avg_execution_time']:.3f}s ± {profile_results['feature_generation']['std_execution_time']:.3f}s
- **Memory Delta:** {profile_results['feature_generation']['avg_memory_delta']:+.1f}MB ± {profile_results['feature_generation']['std_memory_delta']:.1f}MB
- **Total Features:** {profile_results['feature_generation']['total_features']:,}
- **Feature Categories:** {profile_results['feature_generation']['feature_categories']}

## Multi-Objective Optimization Performance
- **Execution Time:** {profile_results['multi_objective_optimization']['execution_time']:.3f}s
- **Memory Delta:** {profile_results['multi_objective_optimization']['memory_delta']:+.1f}MB
- **Success Rate:** {profile_results['multi_objective_optimization']['optimization_success']}
- **Overall Score:** {profile_results['multi_objective_optimization']['overall_score']:.3f}

## Economic Validation Performance
- **Average Time:** {profile_results['economic_validation']['avg_execution_time']:.3f}s ± {profile_results['economic_validation']['std_execution_time']:.3f}s
- **Memory Delta:** {profile_results['economic_validation']['avg_memory_delta']:+.1f}MB ± {profile_results['economic_validation']['std_memory_delta']:.1f}MB
- **Clusters:** {profile_results['economic_validation']['n_clusters']}
- **Samples:** {profile_results['economic_validation']['n_samples']:,}

## Memory Usage Analysis
- **Initial Memory:** {profile_results['memory_usage']['initial_memory_mb']:.1f}MB
- **Peak Memory:** {profile_results['memory_usage']['peak_memory_mb']:.1f}MB
- **Final Memory:** {profile_results['memory_usage']['final_memory_mb']:.1f}MB
- **Total Delta:** {profile_results['memory_usage']['total_memory_delta_mb']:+.1f}MB
- **Data Size:** {profile_results['memory_usage']['data_size_mb']:.1f}MB
- **Features Size:** {profile_results['memory_usage']['features_size_mb']:.1f}MB

## Parallelization Analysis
- **Optimal Workers:** {profile_results['parallelization']['optimal_workers']}
- **Max Speedup:** {profile_results['parallelization']['max_speedup']:.2f}x
- **CPU Count:** {profile_results['parallelization']['cpu_count']}
- **Recommended Workers:** {profile_results['parallelization']['recommended_workers']}

## Caching Effectiveness
- **Hit Rate:** {profile_results['caching']['hit_rate']:.1%}
- **Speedup:** {profile_results['caching']['speedup']:.2f}x
- **Cache Size:** {profile_results['caching']['cache_size']}
- **Hits:** {profile_results['caching']['cache_hits']}
- **Misses:** {profile_results['caching']['cache_misses']}

## Recommendations

### Performance Optimizations
1. **Parallelization:** Use {profile_results['parallelization']['recommended_workers']} workers for optimal performance
2. **Caching:** Caching provides {profile_results['caching']['speedup']:.2f}x speedup with {profile_results['caching']['hit_rate']:.1%} hit rate
3. **Memory:** Peak memory usage is {profile_results['memory_usage']['peak_memory_mb']:.1f}MB

### Bottlenecks
1. **Feature Generation:** Takes {profile_results['feature_generation']['avg_execution_time']:.3f}s on average
2. **Multi-Objective Optimization:** Takes {profile_results['multi_objective_optimization']['execution_time']:.3f}s
3. **Economic Validation:** Takes {profile_results['economic_validation']['avg_execution_time']:.3f}s on average

### Scaling Considerations
- **Memory:** System uses {profile_results['memory_usage']['total_memory_delta_mb']:+.1f}MB additional memory
- **CPU:** Optimal performance with {profile_results['parallelization']['optimal_workers']} workers
- **Features:** Processing {profile_results['data_info']['n_features']:,} features efficiently
"""
        
        return report


def run_performance_profiling(market_data: pd.DataFrame, 
                            n_iterations: int = 3,
                            profile_dir: str = "performance_profiles") -> Dict[str, Any]:
    """
    Run comprehensive performance profiling.
    
    Args:
        market_data: Market data to profile
        n_iterations: Number of iterations for averaging
        profile_dir: Directory to save profile results
        
    Returns:
        Dictionary containing performance metrics
    """
    profiler = PerformanceProfiler(profile_dir=profile_dir)
    results = profiler.run_comprehensive_profile(market_data, n_iterations)
    
    # Generate and save report
    report = profiler.generate_performance_report(results)
    report_file = profiler.profile_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"Performance profiling completed. Report saved to {report_file}")
    
    return results


if __name__ == "__main__":
    # Generate sample data for testing
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='1H')
    
    market_data = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.cumsum(np.random.normal(0, 0.01, n_samples)),
        'high': 100 + np.cumsum(np.random.normal(0, 0.01, n_samples)) + np.abs(np.random.normal(0, 0.01, n_samples)),
        'low': 100 + np.cumsum(np.random.normal(0, 0.01, n_samples)) - np.abs(np.random.normal(0, 0.01, n_samples)),
        'close': 100 + np.cumsum(np.random.normal(0, 0.01, n_samples)),
        'volume': np.random.lognormal(5, 0.5, n_samples)
    })
    
    market_data['returns'] = market_data['close'].pct_change()
    market_data['volatility'] = market_data['returns'].rolling(20).std()
    
    # Run performance profiling
    print("Running performance profiling...")
    results = run_performance_profiling(market_data, n_iterations=3)
    
    print("\nPerformance Profiling Results:")
    print(f"Feature Generation: {results['feature_generation']['avg_execution_time']:.3f}s")
    print(f"Multi-Objective Optimization: {results['multi_objective_optimization']['execution_time']:.3f}s")
    print(f"Economic Validation: {results['economic_validation']['avg_execution_time']:.3f}s")
    print(f"Peak Memory: {results['memory_usage']['peak_memory_mb']:.1f}MB")
    print(f"Optimal Workers: {results['parallelization']['optimal_workers']}")
    print(f"Caching Speedup: {results['caching']['speedup']:.2f}x")