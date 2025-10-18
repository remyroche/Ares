"""
Test suite for CMI Complementarity Enhancements

This module tests all the advanced technical improvements identified
in the CMI complementarity integration analysis.
"""

import pytest
import numpy as np
import pandas as pd
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


class TestDensityAwareKSelector:
    """Test density-aware k-selection for KSG estimator."""
    
    def test_density_aware_k_selection(self):
        """Test density-aware k selection."""
        selector = DensityAwareKSelector(base_k=5)
        
        # Create data with varying density
        np.random.seed(42)
        n_samples = 1000
        X = np.random.normal(0, 1, n_samples)
        Y = np.random.normal(0, 1, n_samples)
        A = np.random.uniform(0, 1, n_samples)
        
        k = selector.select_k(X, Y, A)
        
        # Should return a reasonable k value
        assert 3 <= k <= 10
        assert isinstance(k, int)
    
    def test_high_density_variation(self):
        """Test k selection with high density variation."""
        selector = DensityAwareKSelector(base_k=5)
        
        # Create data with high density variation
        np.random.seed(42)
        n_samples = 1000
        
        # Mix of high and low density regions
        X = np.concatenate([
            np.random.normal(0, 0.1, n_samples // 2),  # High density
            np.random.normal(0, 2.0, n_samples // 2)     # Low density
        ])
        Y = np.random.normal(0, 1, n_samples)
        A = np.random.uniform(0, 1, n_samples)
        
        k = selector.select_k(X, Y, A)
        
        # Should use smaller k for high density variation
        assert k <= 5


class TestAdaptiveDecomposition:
    """Test adaptive decomposition using PCA or ICA."""
    
    def test_gaussian_data_pca(self):
        """Test PCA for Gaussian data."""
        decomposer = AdaptiveDecomposition(max_dims=2)
        
        # Create Gaussian data
        np.random.seed(42)
        n_samples = 1000
        A_multi_channel = np.random.multivariate_normal(
            [0, 0, 0], 
            [[1, 0.5, 0.3], [0.5, 1, 0.2], [0.3, 0.2, 1]], 
            n_samples
        )
        
        result = decomposer.decompose(A_multi_channel)
        
        # Should return 2D result
        assert result.shape[1] == 2
        assert result.shape[0] == n_samples
    
    def test_non_gaussian_data_ica(self):
        """Test ICA for non-Gaussian data."""
        decomposer = AdaptiveDecomposition(max_dims=2)
        
        # Create non-Gaussian data (exponential)
        np.random.seed(42)
        n_samples = 1000
        A_multi_channel = np.column_stack([
            np.random.exponential(1, n_samples),
            np.random.exponential(2, n_samples),
            np.random.exponential(0.5, n_samples)
        ])
        
        result = decomposer.decompose(A_multi_channel)
        
        # Should return 2D result
        assert result.shape[1] == 2
        assert result.shape[0] == n_samples
    
    def test_information_loss_threshold(self):
        """Test information loss threshold handling."""
        decomposer = AdaptiveDecomposition(max_dims=2, info_loss_threshold=0.15)
        
        # Create data with high information loss
        np.random.seed(42)
        n_samples = 1000
        
        # Highly correlated channels (high information loss when reduced)
        A_multi_channel = np.column_stack([
            np.random.normal(0, 1, n_samples),
            np.random.normal(0, 1, n_samples) + 0.1 * np.random.normal(0, 1, n_samples),
            np.random.normal(0, 1, n_samples) + 0.1 * np.random.normal(0, 1, n_samples)
        ])
        
        result = decomposer.decompose(A_multi_channel)
        
        # Should return 2D result
        assert result.shape[1] == 2
        assert result.shape[0] == n_samples


class TestRobustSynergyEstimator:
    """Test robust synergy estimation with bootstrap confidence intervals."""
    
    def test_synergy_estimation(self):
        """Test synergy estimation."""
        estimator = RobustSynergyEstimator(n_bootstrap=50)
        
        np.random.seed(42)
        n_samples = 500
        
        # Create correlated features
        Xi = np.random.normal(0, 1, n_samples)
        Xj = Xi + 0.5 * np.random.normal(0, 1, n_samples)
        Y = Xi + Xj + 0.3 * np.random.normal(0, 1, n_samples)
        A = np.random.uniform(0, 1, n_samples)
        
        synergy = estimator.estimate_synergy_with_confidence(Xi, Xj, Y, A)
        
        # Should return a numeric value
        assert isinstance(synergy, (int, float))
        assert not np.isnan(synergy)
    
    def test_stratified_bootstrap(self):
        """Test stratified bootstrap indices."""
        estimator = RobustSynergyEstimator()
        
        np.random.seed(42)
        n_samples = 1000
        
        # Create imbalanced Y
        Y = np.concatenate([
            np.zeros(800),  # 80% class 0
            np.ones(200)    # 20% class 1
        ])
        
        indices = estimator._stratified_bootstrap_indices(Y)
        
        # Should preserve class balance
        assert len(indices) == n_samples
        assert len(np.unique(indices)) <= n_samples  # Some duplicates allowed


class TestThreadSafeCMICache:
    """Test thread-safe LRU cache."""
    
    def test_cache_operations(self):
        """Test basic cache operations."""
        cache = ThreadSafeCMICache(maxsize=3)
        
        # Test put and get
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test cache eviction
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        cache.put("key4", "value4")  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key4") == "value4"
    
    def test_thread_safety(self):
        """Test thread safety with concurrent access."""
        import threading
        import time
        
        cache = ThreadSafeCMICache(maxsize=100)
        results = []
        
        def worker(thread_id):
            for i in range(10):
                key = f"thread_{thread_id}_key_{i}"
                value = f"value_{thread_id}_{i}"
                cache.put(key, value)
                retrieved = cache.get(key)
                results.append((thread_id, i, retrieved == value))
                time.sleep(0.001)  # Small delay to increase chance of race conditions
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check that all operations succeeded
        assert all(result[2] for result in results)


class TestCachedInteractionSelector:
    """Test cached interaction selection."""
    
    def test_interaction_selection(self):
        """Test interaction selection with caching."""
        selector = CachedInteractionSelector(cache_size=100)
        
        np.random.seed(42)
        n_samples = 500
        
        # Create features
        features = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(0, 1, n_samples),
            'feature_3': np.random.normal(0, 1, n_samples),
            'feature_4': np.random.normal(0, 1, n_samples)
        })
        
        Y = np.random.normal(0, 1, n_samples)
        A = np.random.uniform(0, 1, n_samples)
        
        interactions = selector.select_interactions(features, Y, A, budget=3)
        
        # Should return list of interaction pairs
        assert isinstance(interactions, list)
        assert len(interactions) <= 3
        
        for interaction in interactions:
            assert len(interaction) == 2  # Pair of features
            assert all(isinstance(feat, str) for feat in interaction)


class TestEarlyStoppingDeltaPerf:
    """Test early stopping ΔPerf validation."""
    
    def test_early_stopping(self):
        """Test early stopping logic."""
        validator = EarlyStoppingDeltaPerf(patience=3)
        
        np.random.seed(42)
        n_samples = 500
        
        # Create features
        features = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(0, 1, n_samples),
            'feature_3': np.random.normal(0, 1, n_samples),
            'feature_4': np.random.normal(0, 1, n_samples),
            'feature_5': np.random.normal(0, 1, n_samples)
        })
        
        Y = np.random.normal(0, 1, n_samples)
        A = np.random.uniform(0, 1, (n_samples, 1))
        
        candidates = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        
        delta_scores = validator.validate_delta_perf(candidates, features, Y, A)
        
        # Should return delta scores
        assert isinstance(delta_scores, dict)
        assert len(delta_scores) <= len(candidates)  # Early stopping might reduce this
    
    def test_memory_efficient_batch_validation(self):
        """Test memory-efficient batch validation."""
        validator = EarlyStoppingDeltaPerf(batch_size=2)
        
        np.random.seed(42)
        n_samples = 1000
        
        # Create features
        features = pd.DataFrame({
            f'feature_{i}': np.random.normal(0, 1, n_samples)
            for i in range(10)
        })
        
        Y = np.random.normal(0, 1, n_samples)
        A = np.random.uniform(0, 1, (n_samples, 1))
        
        candidates = [f'feature_{i}' for i in range(10)]
        
        delta_scores = validator._memory_efficient_batch_validation(
            candidates, features, Y, A
        )
        
        # Should return delta scores for all candidates
        assert isinstance(delta_scores, dict)
        assert len(delta_scores) == len(candidates)


class TestAnalyticalNoiseFloor:
    """Test analytical noise floor estimation."""
    
    def test_noise_floor_estimation(self):
        """Test noise floor estimation."""
        estimator = AnalyticalNoiseFloor(confidence=0.9)
        
        np.random.seed(42)
        n_samples = 1000
        
        X = np.random.normal(0, 1, n_samples)
        Y = np.random.normal(0, 1, n_samples)
        A = np.random.uniform(0, 1, (n_samples, 1))
        
        noise_floor = estimator.estimate_noise_floor(X, Y, A)
        
        # Should return a positive value
        assert noise_floor > 0
        assert isinstance(noise_floor, (int, float))
        assert not np.isnan(noise_floor)


class TestSafeMPSComputation:
    """Test safe MPS computation with CPU fallback."""
    
    def test_mps_initialization(self):
        """Test MPS initialization."""
        mps = SafeMPSComputation()
        
        # Should initialize without errors
        assert hasattr(mps, 'use_mps')
        assert hasattr(mps, 'device')
    
    def test_safe_computation(self):
        """Test safe computation with fallback."""
        mps = SafeMPSComputation()
        
        np.random.seed(42)
        n_samples = 100
        
        X = np.random.normal(0, 1, n_samples)
        Y = np.random.normal(0, 1, n_samples)
        A = np.random.uniform(0, 1, n_samples)
        
        result = mps.safe_computation(X, Y, A)
        
        # Should return a numeric result
        assert isinstance(result, (int, float))
        assert not np.isnan(result)


class TestMemoryAwareCacheManager:
    """Test memory-aware cache management."""
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        manager = MemoryAwareCacheManager()
        
        # Should initialize without errors
        assert hasattr(manager, 'cache')
        assert hasattr(manager, 'cache_warmed')
    
    def test_cache_warming(self):
        """Test cache warming."""
        manager = MemoryAwareCacheManager()
        
        np.random.seed(42)
        n_samples = 100
        
        features = pd.DataFrame({
            f'feature_{i}': np.random.normal(0, 1, n_samples)
            for i in range(5)
        })
        
        Y = np.random.normal(0, 1, n_samples)
        A = np.random.uniform(0, 1, n_samples)
        
        # Should complete without errors
        manager.warm_cache(features, Y, A)
        
        # Should mark cache as warmed
        assert manager.cache_warmed


class TestSmoothFamilyBudgetAllocator:
    """Test smooth family budget allocation."""
    
    def test_budget_allocation(self):
        """Test budget allocation."""
        allocator = SmoothFamilyBudgetAllocator(
            total_budget=60, min_budget=2, max_budget=20
        )
        
        # Create family scores
        family_scores = {
            'price_features': 0.8,
            'volume_features': 0.6,
            'technical_features': 0.4,
            'momentum_features': 0.2
        }
        
        budgets = allocator.allocate_budgets(family_scores)
        
        # Should return budgets for all families
        assert len(budgets) == len(family_scores)
        assert all(isinstance(budget, int) for budget in budgets.values())
        
        # Should respect min/max constraints
        assert all(2 <= budget <= 20 for budget in budgets.values())
        
        # Should sum to total budget (approximately)
        total_allocated = sum(budgets.values())
        assert abs(total_allocated - 60) <= 4  # Allow some rounding error


class TestSmartDegradationHandler:
    """Test smart degradation handler."""
    
    def test_degenerate_detection(self):
        """Test degenerate A detection."""
        handler = SmartDegradationHandler(threshold=1e-6)
        
        # Test constant A (degenerate)
        A_constant = np.ones((100, 1))
        assert handler.is_degenerate_A(A_constant)
        
        # Test normal A (not degenerate)
        A_normal = np.random.normal(0, 1, (100, 1))
        assert not handler.is_degenerate_A(A_normal)
    
    def test_rank_deficiency_detection(self):
        """Test rank deficiency detection."""
        handler = SmartDegradationHandler()
        
        # Create rank-deficient matrix
        A_rank_deficient = np.column_stack([
            np.random.normal(0, 1, 100),
            np.random.normal(0, 1, 100),
            np.random.normal(0, 1, 100)  # Third column is linear combination
        ])
        A_rank_deficient[:, 2] = A_rank_deficient[:, 0] + A_rank_deficient[:, 1]
        
        assert handler.is_degenerate_A(A_rank_deficient)


class TestCMIComplementarityEnhancements:
    """Test main enhancement class."""
    
    def test_enhancements_initialization(self):
        """Test enhancements initialization."""
        enhancements = CMIComplementarityEnhancements()
        
        # Should initialize all components
        assert hasattr(enhancements, 'density_selector')
        assert hasattr(enhancements, 'decomposition')
        assert hasattr(enhancements, 'synergy_estimator')
        assert hasattr(enhancements, 'interaction_selector')
        assert hasattr(enhancements, 'delta_perf_validator')
        assert hasattr(enhancements, 'noise_floor_estimator')
        assert hasattr(enhancements, 'mps_computation')
        assert hasattr(enhancements, 'cache_manager')
        assert hasattr(enhancements, 'budget_allocator')
        assert hasattr(enhancements, 'degradation_handler')
    
    def test_apply_enhancements(self):
        """Test applying all enhancements."""
        enhancements = CMIComplementarityEnhancements()
        
        np.random.seed(42)
        n_samples = 500
        
        # Create test data
        features = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(0, 1, n_samples),
            'feature_3': np.random.normal(0, 1, n_samples)
        })
        
        Y = np.random.normal(0, 1, n_samples)
        A = np.random.uniform(0, 1, (n_samples, 2))
        
        results = enhancements.apply_enhancements(features, Y, A)
        
        # Should return enhancement results
        assert isinstance(results, dict)
        assert 'k' in results
        assert 'A_reduced' in results
        assert 'cache_manager' in results
        assert 'synergy_estimator' in results
        assert 'interaction_selector' in results
        assert 'delta_perf_validator' in results
        assert 'noise_floor_estimator' in results
        assert 'mps_computation' in results
        assert 'budget_allocator' in results
        
        # Check specific results
        assert isinstance(results['k'], int)
        assert results['A_reduced'].shape[0] == n_samples
        assert results['A_reduced'].shape[1] <= 2


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
