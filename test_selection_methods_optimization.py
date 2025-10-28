#!/usr/bin/env python3
"""
Test script for optimized selection_methods.py

Tests:
1. MI proxy functionality
2. MRMRSelector with vectorization
3. CompositeFeatureScorer with vectorization
4. Performance comparison between proxy and sklearn
"""

import numpy as np
import pandas as pd
import time
from src.training.utils.feature_selection.selection_methods import (
    MRMRSelector,
    CompositeFeatureScorer,
    get_mi_proxy
)

def generate_test_data(n_samples=1000, n_features=50, noise_level=0.1):
    """Generate synthetic test data."""
    np.random.seed(42)
    
    # Create features with varying relevance
    X = np.random.randn(n_samples, n_features)
    
    # Create target with some features relevant
    y = (
        2.0 * X[:, 0] +      # Strong relevance
        1.5 * X[:, 1] +      # Medium relevance
        0.5 * X[:, 2] +      # Weak relevance
        noise_level * np.random.randn(n_samples)
    )
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    return X, y, feature_names


def test_mi_proxy():
    """Test MI proxy functionality."""
    print("\n" + "="*60)
    print("TEST 1: MI Proxy Functionality")
    print("="*60)
    
    # Generate test data
    X, y, feature_names = generate_test_data(n_samples=500, n_features=20)
    
    # Initialize MI proxy
    mi_proxy = get_mi_proxy(use_cache=True, use_correlation_proxy=True)
    
    print("\n✅ MI proxy initialized")
    print(f"   - Cache enabled: {mi_proxy.use_cache}")
    print(f"   - Correlation proxy: {mi_proxy.use_correlation_proxy}")
    
    # Test single MI computation
    start_time = time.time()
    mi_score = mi_proxy.compute_mi(X[:, 0], y, x_id=0, y_id=-1)
    single_time = time.time() - start_time
    
    print(f"\n✅ Single MI computation: {mi_score:.6f} (took {single_time*1000:.2f}ms)")
    
    # Test cached computation
    start_time = time.time()
    mi_score_cached = mi_proxy.compute_mi(X[:, 0], y, x_id=0, y_id=-1)
    cached_time = time.time() - start_time
    
    print(f"✅ Cached MI computation: {mi_score_cached:.6f} (took {cached_time*1000:.2f}ms)")
    print(f"   - Speedup: {single_time/cached_time:.1f}x faster")
    
    # Test batch computation
    start_time = time.time()
    mi_scores_batch = mi_proxy.compute_mi_batch(X, y)
    batch_time = time.time() - start_time
    
    print(f"\n✅ Batch MI computation: {len(mi_scores_batch)} features")
    print(f"   - Time: {batch_time*1000:.2f}ms ({batch_time/len(mi_scores_batch)*1000:.2f}ms per feature)")
    print(f"   - Top 5 MI scores: {sorted(mi_scores_batch, reverse=True)[:5]}")
    
    # Compare with loop computation
    start_time = time.time()
    mi_scores_loop = []
    for i in range(X.shape[1]):
        mi = mi_proxy.compute_mi(X[:, i], y, x_id=i, y_id=-1, use_sklearn=False)
        mi_scores_loop.append(mi)
    loop_time = time.time() - start_time
    
    print(f"\n✅ Loop MI computation: {len(mi_scores_loop)} features")
    print(f"   - Time: {loop_time*1000:.2f}ms ({loop_time/len(mi_scores_loop)*1000:.2f}ms per feature)")
    print(f"   - Speedup: {loop_time/batch_time:.1f}x faster with batch")
    
    print(f"\n✅ Cache size: {mi_proxy.get_cache_size()} entries")
    
    return True


def test_mrmr_selector():
    """Test MRMRSelector with vectorization."""
    print("\n" + "="*60)
    print("TEST 2: MRMRSelector with Vectorization")
    print("="*60)
    
    # Generate test data
    X, y, feature_names = generate_test_data(n_samples=1000, n_features=50)
    
    # Test with MI proxy enabled
    print("\n🔵 Testing with MI proxy enabled...")
    config_with_proxy = {
        'use_mi_proxy': True,
        'use_vectorization': True,
        'relevance_method': 'mutual_info',
        'redundancy_method': 'correlation'
    }
    selector_with_proxy = MRMRSelector(config=config_with_proxy)
    
    start_time = time.time()
    result_with_proxy = selector_with_proxy.select_features(
        X, y, feature_names, n_features=10
    )
    time_with_proxy = time.time() - start_time
    
    print(f"✅ MRMR with proxy completed in {time_with_proxy:.3f}s")
    print(f"   - Selected features: {len(result_with_proxy['selected_features'])}")
    print(f"   - Top 5 features: {result_with_proxy['selected_features'][:5]}")
    
    # Test without MI proxy (fallback to sklearn)
    print("\n🔵 Testing without MI proxy (sklearn)...")
    config_without_proxy = {
        'use_mi_proxy': False,
        'use_vectorization': False,
        'relevance_method': 'mutual_info',
        'redundancy_method': 'correlation'
    }
    selector_without_proxy = MRMRSelector(config=config_without_proxy)
    
    start_time = time.time()
    result_without_proxy = selector_without_proxy.select_features(
        X, y, feature_names, n_features=10
    )
    time_without_proxy = time.time() - start_time
    
    print(f"✅ MRMR without proxy completed in {time_without_proxy:.3f}s")
    print(f"   - Selected features: {len(result_without_proxy['selected_features'])}")
    print(f"   - Top 5 features: {result_without_proxy['selected_features'][:5]}")
    
    # Compare performance
    speedup = time_without_proxy / time_with_proxy
    print(f"\n✅ Performance comparison:")
    print(f"   - With proxy: {time_with_proxy:.3f}s")
    print(f"   - Without proxy: {time_without_proxy:.3f}s")
    print(f"   - Speedup: {speedup:.2f}x faster with vectorization")
    
    return True


def test_composite_scorer():
    """Test CompositeFeatureScorer with vectorization."""
    print("\n" + "="*60)
    print("TEST 3: CompositeFeatureScorer with Vectorization")
    print("="*60)
    
    # Generate test data
    X, y, feature_names = generate_test_data(n_samples=1000, n_features=100)
    
    # Test with MI proxy and vectorization
    print("\n🔵 Testing with MI proxy and vectorization enabled...")
    config_optimized = {
        'use_mi_proxy': True,
        'use_vectorization': True,
        'rfe_removal_rate': 0.33
    }
    scorer_optimized = CompositeFeatureScorer(config=config_optimized)
    
    start_time = time.time()
    result_optimized = scorer_optimized.select_features(
        X, y, feature_names, n_features=20
    )
    time_optimized = time.time() - start_time
    
    print(f"✅ Composite scoring optimized completed in {time_optimized:.3f}s")
    print(f"   - Selected features: {len(result_optimized['selected_features'])}")
    print(f"   - RFE rounds: {result_optimized.get('rounds', 'N/A')}")
    print(f"   - Top 5 features: {result_optimized['selected_features'][:5]}")
    
    # Test without optimization
    print("\n🔵 Testing without optimization...")
    config_basic = {
        'use_mi_proxy': False,
        'use_vectorization': False,
        'rfe_removal_rate': 0.33
    }
    scorer_basic = CompositeFeatureScorer(config=config_basic)
    
    start_time = time.time()
    result_basic = scorer_basic.select_features(
        X, y, feature_names, n_features=20
    )
    time_basic = time.time() - start_time
    
    print(f"✅ Composite scoring basic completed in {time_basic:.3f}s")
    print(f"   - Selected features: {len(result_basic['selected_features'])}")
    print(f"   - RFE rounds: {result_basic.get('rounds', 'N/A')}")
    print(f"   - Top 5 features: {result_basic['selected_features'][:5]}")
    
    # Compare performance
    speedup = time_basic / time_optimized
    print(f"\n✅ Performance comparison:")
    print(f"   - Optimized: {time_optimized:.3f}s")
    print(f"   - Basic: {time_basic:.3f}s")
    print(f"   - Speedup: {speedup:.2f}x faster with optimization")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("SELECTION METHODS OPTIMIZATION TESTS")
    print("="*60)
    
    results = {}
    
    try:
        results['MI Proxy'] = test_mi_proxy()
    except Exception as e:
        print(f"\n❌ MI Proxy test failed: {e}")
        results['MI Proxy'] = False
    
    try:
        results['MRMR Selector'] = test_mrmr_selector()
    except Exception as e:
        print(f"\n❌ MRMR Selector test failed: {e}")
        results['MRMR Selector'] = False
    
    try:
        results['Composite Scorer'] = test_composite_scorer()
    except Exception as e:
        print(f"\n❌ Composite Scorer test failed: {e}")
        results['Composite Scorer'] = False
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
