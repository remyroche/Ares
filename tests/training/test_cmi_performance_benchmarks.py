"""
Performance benchmarks for CMI Complementarity Enhancements

This module provides comprehensive performance testing for all the
advanced technical improvements in the CMI complementarity integration.
"""

import pytest
import numpy as np
import pandas as pd
import time
import psutil
import gc
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.training.steps.pre_training.unified_data_driven_pipeline.utils.cmi_enhancements import (
    DensityAwareKSelector,
    AdaptiveDecomposition,
    RobustSynergyEstimator,
    ThreadSafeCMICache,
    CachedInteractionSelector,
    EarlyStoppingDeltaPerf,
    AnalyticalNoiseFloor,
    SafeMPSComputation,
    MemoryAwareCacheManager,
    SmoothFamilyBudgetAllocator,
    SmartDegradationHandler,
    CMIComplementarityEnhancements
)


class TestPerformanceBenchmarks:
    """Performance benchmarks for CMI enhancements."""
    
    def test_density_aware_k_selection_performance(self):
        """Benchmark density-aware k-selection performance."""
        selector = DensityAwareKSelector(base_k=5)
        
        # Test with different data sizes
        sizes = [100, 500, 1000, 2000]
        times = []
        
        for size in sizes:
            np.random.seed(42)
            X = np.random.normal(0, 1, size)
            Y = np.random.normal(0, 1, size)
            A = np.random.uniform(0, 1, size)
            
            start_time = time.time()
            k = selector.select_k(X, Y, A)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            # Should complete within reasonable time
            assert end_time - start_time < 5.0  # 5 seconds max
        
        # Performance should scale reasonably
        print(f"Density-aware k-selection times: {times}")
        assert all(t < 5.0 for t in times)
    
    def test_adaptive_decomposition_performance(self):
        """Benchmark adaptive decomposition performance."""
        decomposer = AdaptiveDecomposition(max_dims=2)
        
        # Test with different channel counts
        channel_counts = [2, 3, 5, 10]
        times = []
        
        for n_channels in channel_counts:
            np.random.seed(42)
            n_samples = 1000
            A_multi_channel = np.random.multivariate_normal(
                np.zeros(n_channels),
                np.eye(n_channels),
                n_samples
            )
            
            start_time = time.time()
            result = decomposer.decompose(A_multi_channel)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            # Should complete within reasonable time
            assert end_time - start_time < 10.0  # 10 seconds max
        
        print(f"Adaptive decomposition times: {times}")
        assert all(t < 10.0 for t in times)
    
    def test_robust_synergy_estimation_performance(self):
        """Benchmark robust synergy estimation performance."""
        estimator = RobustSynergyEstimator(n_bootstrap=50)
        
        # Test with different sample sizes
        sizes = [100, 500, 1000]
        times = []
        
        for size in sizes:
            np.random.seed(42)
            Xi = np.random.normal(0, 1, size)
            Xj = np.random.normal(0, 1, size)
            Y = np.random.normal(0, 1, size)
            A = np.random.uniform(0, 1, size)
            
            start_time = time.time()
            synergy = estimator.estimate_synergy_with_confidence(Xi, Xj, Y, A)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            # Should complete within reasonable time
            assert end_time - start_time < 15.0  # 15 seconds max
        
        print(f"Robust synergy estimation times: {times}")
        assert all(t < 15.0 for t in times)
    
    def test_thread_safe_cache_performance(self):
        """Benchmark thread-safe cache performance."""
        cache = ThreadSafeCMICache(maxsize=1000)
        
        # Test cache operations
        start_time = time.time()
        
        # Put operations
        for i in range(1000):
            cache.put(f"key_{i}", f"value_{i}")
        
        # Get operations
        for i in range(1000):
            result = cache.get(f"key_{i}")
            assert result == f"value_{i}"
        
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 5.0  # 5 seconds max
        print(f"Thread-safe cache operations time: {end_time - start_time:.3f}s")
    
    def test_cached_interaction_selection_performance(self):
        """Benchmark cached interaction selection performance."""
        selector = CachedInteractionSelector(cache_size=1000)
        
        # Test with different feature counts
        feature_counts = [5, 10, 20, 50]
        times = []
        
        for n_features in feature_counts:
            np.random.seed(42)
            n_samples = 500
            
            features = pd.DataFrame({
                f'feature_{i}': np.random.normal(0, 1, n_samples)
                for i in range(n_features)
            })
            
            Y = np.random.normal(0, 1, n_samples)
            A = np.random.uniform(0, 1, n_samples)
            
            start_time = time.time()
            interactions = selector.select_interactions(features, Y, A, budget=10)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            # Should complete within reasonable time
            assert end_time - start_time < 20.0  # 20 seconds max
        
        print(f"Cached interaction selection times: {times}")
        assert all(t < 20.0 for t in times)
    
    def test_early_stopping_delta_perf_performance(self):
        """Benchmark early stopping ΔPerf validation performance."""
        validator = EarlyStoppingDeltaPerf(patience=3)
        
        # Test with different candidate counts
        candidate_counts = [5, 10, 20, 50]
        times = []
        
        for n_candidates in candidate_counts:
            np.random.seed(42)
            n_samples = 500
            
            features = pd.DataFrame({
                f'feature_{i}': np.random.normal(0, 1, n_samples)
                for i in range(n_candidates)
            })
            
            Y = np.random.normal(0, 1, n_samples)
            A = np.random.uniform(0, 1, (n_samples, 1))
            
            candidates = [f'feature_{i}' for i in range(n_candidates)]
            
            start_time = time.time()
            delta_scores = validator.validate_delta_perf(candidates, features, Y, A)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            # Should complete within reasonable time
            assert end_time - start_time < 30.0  # 30 seconds max
        
        print(f"Early stopping ΔPerf validation times: {times}")
        assert all(t < 30.0 for t in times)
    
    def test_analytical_noise_floor_performance(self):
        """Benchmark analytical noise floor estimation performance."""
        estimator = AnalyticalNoiseFloor(confidence=0.9)
        
        # Test with different sample sizes
        sizes = [100, 500, 1000, 2000, 5000]
        times = []
        
        for size in sizes:
            np.random.seed(42)
            X = np.random.normal(0, 1, size)
            Y = np.random.normal(0, 1, size)
            A = np.random.uniform(0, 1, (size, 1))
            
            start_time = time.time()
            noise_floor = estimator.estimate_noise_floor(X, Y, A)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            # Should complete very quickly (analytical)
            assert end_time - start_time < 1.0  # 1 second max
        
        print(f"Analytical noise floor estimation times: {times}")
        assert all(t < 1.0 for t in times)
    
    def test_safe_mps_computation_performance(self):
        """Benchmark safe MPS computation performance."""
        mps = SafeMPSComputation()
        
        # Test with different data sizes
        sizes = [100, 500, 1000, 2000]
        times = []
        
        for size in sizes:
            np.random.seed(42)
            X = np.random.normal(0, 1, size)
            Y = np.random.normal(0, 1, size)
            A = np.random.uniform(0, 1, size)
            
            start_time = time.time()
            result = mps.safe_computation(X, Y, A)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            # Should complete within reasonable time
            assert end_time - start_time < 5.0  # 5 seconds max
        
        print(f"Safe MPS computation times: {times}")
        assert all(t < 5.0 for t in times)
    
    def test_memory_aware_cache_performance(self):
        """Benchmark memory-aware cache performance."""
        manager = MemoryAwareCacheManager()
        
        # Test cache warming
        np.random.seed(42)
        n_samples = 1000
        
        features = pd.DataFrame({
            f'feature_{i}': np.random.normal(0, 1, n_samples)
            for i in range(20)
        })
        
        Y = np.random.normal(0, 1, n_samples)
        A = np.random.uniform(0, 1, n_samples)
        
        start_time = time.time()
        manager.warm_cache(features, Y, A)
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 10.0  # 10 seconds max
        print(f"Cache warming time: {end_time - start_time:.3f}s")
    
    def test_smooth_family_budget_performance(self):
        """Benchmark smooth family budget allocation performance."""
        allocator = SmoothFamilyBudgetAllocator(
            total_budget=100, min_budget=2, max_budget=20
        )
        
        # Test with different family counts
        family_counts = [5, 10, 20, 50]
        times = []
        
        for n_families in family_counts:
            family_scores = {
                f'family_{i}': np.random.random()
                for i in range(n_families)
            }
            
            start_time = time.time()
            budgets = allocator.allocate_budgets(family_scores)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            # Should complete very quickly
            assert end_time - start_time < 1.0  # 1 second max
        
        print(f"Smooth family budget allocation times: {times}")
        assert all(t < 1.0 for t in times)
    
    def test_smart_degradation_performance(self):
        """Benchmark smart degradation handler performance."""
        handler = SmartDegradationHandler(threshold=1e-6)
        
        # Test with different data sizes
        sizes = [100, 500, 1000, 2000]
        times = []
        
        for size in sizes:
            np.random.seed(42)
            A = np.random.uniform(0, 1, (size, 2))
            
            start_time = time.time()
            is_degenerate = handler.is_degenerate_A(A)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            # Should complete very quickly
            assert end_time - start_time < 1.0  # 1 second max
        
        print(f"Smart degradation detection times: {times}")
        assert all(t < 1.0 for t in times)
    
    def test_full_enhancements_performance(self):
        """Benchmark full enhancements application performance."""
        enhancements = CMIComplementarityEnhancements()
        
        # Test with different data sizes
        sizes = [100, 500, 1000]
        times = []
        
        for size in sizes:
            np.random.seed(42)
            
            features = pd.DataFrame({
                f'feature_{i}': np.random.normal(0, 1, size)
                for i in range(10)
            })
            
            Y = np.random.normal(0, 1, size)
            A = np.random.uniform(0, 1, (size, 2))
            
            start_time = time.time()
            results = enhancements.apply_enhancements(features, Y, A)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            # Should complete within reasonable time
            assert end_time - start_time < 30.0  # 30 seconds max
        
        print(f"Full enhancements application times: {times}")
        assert all(t < 30.0 for t in times)
    
    def test_memory_usage_benchmarks(self):
        """Benchmark memory usage for different components."""
        # Get initial memory usage
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Test cache memory usage
        cache = ThreadSafeCMICache(maxsize=1000)
        
        # Fill cache
        for i in range(1000):
            cache.put(f"key_{i}", f"value_{i}" * 100)  # Large values
        
        cache_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        cache_memory_usage = cache_memory - initial_memory
        
        print(f"Cache memory usage: {cache_memory_usage:.2f} MB")
        
        # Test memory-efficient batch validation
        validator = EarlyStoppingDeltaPerf(batch_size=5)
        
        np.random.seed(42)
        n_samples = 1000
        n_candidates = 50
        
        features = pd.DataFrame({
            f'feature_{i}': np.random.normal(0, 1, n_samples)
            for i in range(n_candidates)
        })
        
        Y = np.random.normal(0, 1, n_samples)
        A = np.random.uniform(0, 1, (n_samples, 1))
        
        candidates = [f'feature_{i}' for i in range(n_candidates)]
        
        # Monitor memory during batch validation
        gc.collect()  # Clean up before test
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        delta_scores = validator._memory_efficient_batch_validation(
            candidates, features, Y, A
        )
        
        gc.collect()  # Clean up after test
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before
        
        print(f"Batch validation memory usage: {memory_usage:.2f} MB")
        
        # Memory usage should be reasonable
        assert cache_memory_usage < 100  # 100 MB max for cache
        assert memory_usage < 50  # 50 MB max for batch validation
    
    def test_scalability_benchmarks(self):
        """Benchmark scalability with increasing data sizes."""
        # Test density-aware k-selection scalability
        selector = DensityAwareKSelector(base_k=5)
        
        sizes = [100, 500, 1000, 2000, 5000]
        times = []
        
        for size in sizes:
            np.random.seed(42)
            X = np.random.normal(0, 1, size)
            Y = np.random.normal(0, 1, size)
            A = np.random.uniform(0, 1, size)
            
            start_time = time.time()
            k = selector.select_k(X, Y, A)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        # Check that time complexity is reasonable
        print(f"Scalability times: {times}")
        
        # Time should not grow too rapidly
        for i in range(1, len(times)):
            growth_factor = times[i] / times[i-1]
            assert growth_factor < 5.0  # Should not grow more than 5x per size doubling
    
    def test_concurrent_access_performance(self):
        """Benchmark concurrent access performance."""
        import threading
        import queue
        
        # Test thread-safe cache with concurrent access
        cache = ThreadSafeCMICache(maxsize=1000)
        results_queue = queue.Queue()
        
        def worker(thread_id, n_operations=100):
            start_time = time.time()
            
            for i in range(n_operations):
                # Put operation
                cache.put(f"thread_{thread_id}_key_{i}", f"value_{thread_id}_{i}")
                
                # Get operation
                result = cache.get(f"thread_{thread_id}_key_{i}")
                
                # Verify result
                assert result == f"value_{thread_id}_{i}"
            
            end_time = time.time()
            results_queue.put((thread_id, end_time - start_time))
        
        # Create multiple threads
        n_threads = 5
        threads = []
        
        for i in range(n_threads):
            thread = threading.Thread(target=worker, args=(i, 50))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # All threads should complete successfully
        assert len(results) == n_threads
        
        # Print performance results
        for thread_id, time_taken in results:
            print(f"Thread {thread_id} time: {time_taken:.3f}s")
        
        # All threads should complete within reasonable time
        assert all(time_taken < 10.0 for _, time_taken in results)


if __name__ == "__main__":
    # Run performance benchmarks
    pytest.main([__file__, "-v", "-s"])