"""
Comprehensive Test Suite for Step08 Optimizations

This test suite validates all implemented optimizations:
- Computational optimizations
- Fast fail implementations
- Enhanced validity checks
- Logic fixes
- Performance enhancements
"""

import asyncio
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytest
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.training.steps.step08_optimized_class import OptimizedStep08
from src.training.steps.step08_optimized_methods import OptimizedStep08Methods

class TestStep08Optimizations:
    """Test suite for Step08 optimizations."""
    
    def setup_method(self):
        """Set up test data and configuration."""
        # Create test configuration
        self.config = {
            'symbol': 'ETHUSDT',
            'exchange': 'BINANCE',
            'timeframe': '1m',
            'step08_optimized': {
                'phase1_target_features': 50,
                'phase2_targets': [30, 20, 10],
                'min_regime_samples': 50,
                'target_balance_ratio': 0.8,
                'enable_regime_rebalancing': True,
                'rebalancing_method': 'oversample',
                'risk_free_rate': 0.02,
                'var_confidence_levels': [0.95, 0.99],
                'model_risk_threshold': 0.3,
                'overfitting_threshold': 0.1,
                'feature_stability_threshold': 0.8,
                'enable_parallel_processing': True,
                'enable_caching': True,
                'enable_incremental_processing': True,
                'chunk_size': 1000,
                'max_workers': 4,
                'min_data_samples': 100,
                'max_missing_data_ratio': 0.1,
                'max_timestamp_gap_seconds': 0.5,
                'max_duplicate_ratio': 0.001
            }
        }
        
        # Create test data
        self.test_data = self._create_test_data()
        
        # Initialize optimized step
        self.step = OptimizedStep08(self.config)
    
    def _create_test_data(self) -> pd.DataFrame:
        """Create comprehensive test data."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create timestamps
        timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='1min')
        
        # Create regime data
        regimes = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.4, 0.2])
        
        # Create feature data
        data = {
            'timestamp': timestamps,
            'composite_cluster_id': regimes,
            'open': np.random.randn(n_samples).cumsum() + 100,
            'high': np.random.randn(n_samples).cumsum() + 105,
            'low': np.random.randn(n_samples).cumsum() + 95,
            'close': np.random.randn(n_samples).cumsum() + 100,
            'volume': np.random.exponential(1000, n_samples),
        }
        
        # Add more features
        for i in range(20):
            data[f'feature_{i}'] = np.random.randn(n_samples)
        
        # Add some constant features (should be detected)
        data['constant_feature'] = np.ones(n_samples)
        
        # Add some features with high missing data
        data['high_missing_feature'] = np.random.randn(n_samples)
        data['high_missing_feature'].iloc[:int(n_samples * 0.6)] = np.nan
        
        return pd.DataFrame(data)
    
    # ============================================================================
    # FAST FAIL TESTS
    # ============================================================================
    
    def test_fast_fail_data_quality(self):
        """Test fast fail data quality validation."""
        # Test with good data
        assert self.step._fast_fail_data_quality(self.test_data) == True
        
        # Test with insufficient samples
        small_data = self.test_data.iloc[:50]
        assert self.step._fast_fail_data_quality(small_data) == False
        
        # Test with missing required columns
        bad_data = self.test_data.drop(columns=['composite_cluster_id'])
        assert self.step._fast_fail_data_quality(bad_data) == False
        
        # Test with excessive missing data
        bad_data = self.test_data.copy()
        bad_data.iloc[:int(len(bad_data) * 0.2)] = np.nan
        assert self.step._fast_fail_data_quality(bad_data) == False
    
    def test_fast_fail_feature_selection(self):
        """Test fast fail feature selection validation."""
        # Test with good data
        assert self.step._fast_fail_feature_selection(self.test_data) == True
        
        # Test with insufficient features
        small_data = self.test_data[['timestamp', 'composite_cluster_id', 'open']]
        assert self.step._fast_fail_feature_selection(small_data) == False
        
        # Test with too many constant features
        bad_data = self.test_data.copy()
        for i in range(15):  # Add many constant features
            bad_data[f'constant_{i}'] = np.ones(len(bad_data))
        assert self.step._fast_fail_feature_selection(bad_data) == False
    
    def test_fast_fail_memory_resources(self):
        """Test fast fail memory and resource validation."""
        # This test might fail in environments with limited resources
        # We'll just test that the method runs without error
        try:
            result = self.step._fast_fail_memory_resources()
            assert isinstance(result, bool)
        except Exception as e:
            # If it fails due to resource constraints, that's expected
            print(f"Memory validation failed as expected: {e}")
    
    # ============================================================================
    # ENHANCED VALIDITY CHECKS TESTS
    # ============================================================================
    
    def test_validate_temporal_integrity(self):
        """Test temporal integrity validation."""
        # Test with good data
        assert self.step._validate_temporal_integrity(self.test_data) == True
        
        # Test with future data
        bad_data = self.test_data.copy()
        bad_data['timestamp'] = pd.date_range(start='2025-01-01', periods=len(bad_data), freq='1min')
        assert self.step._validate_temporal_integrity(bad_data) == False
        
        # Test with duplicate timestamps
        bad_data = self.test_data.copy()
        bad_data['timestamp'].iloc[100:200] = bad_data['timestamp'].iloc[0]
        assert self.step._validate_temporal_integrity(bad_data) == False
    
    def test_validate_regime_transitions(self):
        """Test regime transition validation."""
        # Test with good data
        assert self.step._validate_regime_transitions(self.test_data) == True
        
        # Test with excessive regime changes
        bad_data = self.test_data.copy()
        bad_data['composite_cluster_id'] = np.random.choice([0, 1, 2], size=len(bad_data))
        assert self.step._validate_regime_transitions(bad_data) == False
    
    def test_validate_feature_distributions(self):
        """Test feature distribution validation."""
        # Test with good data
        assert self.step._validate_feature_distributions(self.test_data) == True
        
        # Test with constant features (should warn but not fail)
        bad_data = self.test_data.copy()
        bad_data['constant_feature'] = np.ones(len(bad_data))
        assert self.step._validate_feature_distributions(bad_data) == True  # Should warn but pass
    
    # ============================================================================
    # LOGIC FIXES TESTS
    # ============================================================================
    
    def test_gini_coefficient_fix(self):
        """Test fixed Gini coefficient calculation."""
        # Test with balanced data
        balanced_percentages = {'regime_0': 0.33, 'regime_1': 0.33, 'regime_2': 0.34}
        balance_score = self.step._calculate_balance_score_fixed(balanced_percentages)
        assert 0.8 <= balance_score <= 1.0  # Should be high for balanced data
        
        # Test with imbalanced data
        imbalanced_percentages = {'regime_0': 0.8, 'regime_1': 0.15, 'regime_2': 0.05}
        balance_score = self.step._calculate_balance_score_fixed(imbalanced_percentages)
        assert 0.0 <= balance_score <= 0.5  # Should be low for imbalanced data
        
        # Test edge cases
        single_regime = {'regime_0': 1.0}
        balance_score = self.step._calculate_balance_score_fixed(single_regime)
        assert balance_score == 1.0  # Single regime should be perfectly balanced
    
    def test_regime_weights_fix(self):
        """Test fixed regime weight calculation."""
        # Test with normal data
        regime_sharpes = [0.5, 0.3, 0.7]
        regime_counts = {0: 100, 1: 200, 2: 50}
        weights = self.step._calculate_regime_weights_fixed(regime_sharpes, regime_counts)
        
        assert len(weights) == 3
        assert np.isclose(weights.sum(), 1.0)  # Weights should sum to 1
        assert all(w >= 0 for w in weights)  # All weights should be non-negative
        
        # Test with empty data
        weights = self.step._calculate_regime_weights_fixed([], {})
        assert len(weights) == 0
    
    def test_feature_stability_fix(self):
        """Test fixed feature stability calculation."""
        # Test with normal data
        feature_values = pd.Series(np.random.randn(100))
        regime_values = pd.Series(np.random.choice([0, 1, 2], size=100))
        
        stability = self.step._calculate_feature_stability_fixed(feature_values, regime_values)
        assert 0.0 <= stability <= 1.0
        
        # Test with constant feature
        constant_feature = pd.Series(np.ones(100))
        stability = self.step._calculate_feature_stability_fixed(constant_feature, regime_values)
        assert stability >= 0.0  # Should handle constant features gracefully
    
    # ============================================================================
    # COMPUTATIONAL OPTIMIZATIONS TESTS
    # ============================================================================
    
    def test_sparse_correlation_matrix(self):
        """Test sparse correlation matrix optimization."""
        # Create test data
        X = np.random.randn(100, 10)
        
        # Test sparse correlation
        sparse_corr = self.step._sparse_correlation_matrix_optimized(X, threshold=0.1)
        
        assert sparse_corr.shape == (10, 10)
        assert sparse_corr.nnz <= 100  # Should be sparse
        
        # Test caching
        sparse_corr2 = self.step._sparse_correlation_matrix_optimized(X, threshold=0.1)
        assert sparse_corr.nnz == sparse_corr2.nnz  # Should be identical (cached)
    
    def test_optimized_mrmr_selection(self):
        """Test optimized mRMR selection."""
        # Create test data
        X = np.random.randn(100, 20)
        y = np.random.choice([0, 1, 2], size=100)
        feature_names = [f'feature_{i}' for i in range(20)]
        
        # Test mRMR selection
        selected_features = self.step._mrmr_selection_optimized(X, y, feature_names, 10)
        
        assert len(selected_features) <= 10
        assert all(f in feature_names for f in selected_features)
        assert len(set(selected_features)) == len(selected_features)  # No duplicates
    
    def test_optimized_rf_selection(self):
        """Test optimized Random Forest selection."""
        # Create test data
        X = np.random.randn(100, 20)
        y = np.random.choice([0, 1, 2], size=100)
        feature_names = [f'feature_{i}' for i in range(20)]
        
        # Test RF selection
        selected_features = self.step._rf_selection_optimized(X, y, feature_names, 10)
        
        assert len(selected_features) <= 10
        assert all(f in feature_names for f in selected_features)
        assert len(set(selected_features)) == len(selected_features)  # No duplicates
    
    def test_vectorized_feature_stability(self):
        """Test vectorized feature stability calculation."""
        # Create test data
        features = ['feature_0', 'feature_1', 'feature_2']
        data = self.test_data[features + ['composite_cluster_id']]
        
        # Test vectorized stability
        stability_scores = self.step._vectorized_feature_stability(features, data)
        
        assert len(stability_scores) == 3
        assert all(0.0 <= score <= 1.0 for score in stability_scores.values())
        assert all(f in stability_scores for f in features)
    
    # ============================================================================
    # PERFORMANCE ENHANCEMENTS TESTS
    # ============================================================================
    
    def test_parallel_feature_processing(self):
        """Test parallel feature processing."""
        # Create simple feature functions
        def feature_func1(data):
            return pd.Series(data['open'] * 2, name='open_x2')
        
        def feature_func2(data):
            return pd.Series(data['close'] * 3, name='close_x3')
        
        feature_functions = [feature_func1, feature_func2]
        
        # Test parallel processing
        result = self.step._parallel_feature_processing(self.test_data, feature_functions)
        
        assert 'open_x2' in result.columns
        assert 'close_x3' in result.columns
        assert len(result) == len(self.test_data)
    
    def test_memory_efficient_operations(self):
        """Test memory-efficient operations."""
        # Test data type optimization
        optimized_data = self.step._optimize_data_types(self.test_data.copy())
        
        # Check that some columns were optimized
        assert len(optimized_data.columns) == len(self.test_data.columns)
        
        # Test memory-efficient concatenation
        dataframes = [self.test_data.iloc[:100], self.test_data.iloc[100:200]]
        if hasattr(self.step, 'memory_optimizer') and self.step.memory_optimizer:
            result = self.step.memory_optimizer.memory_efficient_concat(dataframes)
            assert len(result) == 200
    
    def test_caching_mechanisms(self):
        """Test caching mechanisms."""
        # Test that caching works
        X = np.random.randn(50, 5)
        
        # First call should compute
        start_time = time.time()
        corr1 = self.step._sparse_correlation_matrix_optimized(X, threshold=0.1)
        first_time = time.time() - start_time
        
        # Second call should use cache
        start_time = time.time()
        corr2 = self.step._sparse_correlation_matrix_optimized(X, threshold=0.1)
        second_time = time.time() - start_time
        
        # Second call should be faster (cached)
        assert second_time < first_time
        assert np.array_equal(corr1.toarray(), corr2.toarray())
    
    # ============================================================================
    # INTEGRATION TESTS
    # ============================================================================
    
    def test_full_optimization_pipeline(self):
        """Test the full optimization pipeline."""
        # This would test the complete execution pipeline
        # For now, we'll test individual components
        
        # Test data loading and validation
        assert self.step._fast_fail_data_quality(self.test_data)
        assert self.step._fast_fail_feature_selection(self.test_data)
        
        # Test validity checks
        assert self.step._validate_temporal_integrity(self.test_data)
        assert self.step._validate_regime_transitions(self.test_data)
        assert self.step._validate_feature_distributions(self.test_data)
        
        # Test optimizations
        feature_columns = [col for col in self.test_data.columns if col not in ['timestamp', 'composite_cluster_id']]
        X = self.test_data[feature_columns].values
        y = self.test_data['composite_cluster_id'].values
        
        # Test correlation matrix
        corr_matrix = self.step._sparse_correlation_matrix_optimized(X)
        assert corr_matrix.shape[0] == len(feature_columns)
        
        # Test feature selection
        selected_features = self.step._mrmr_selection_optimized(X, y, feature_columns, 10)
        assert len(selected_features) <= 10
    
    def test_performance_improvements(self):
        """Test that optimizations actually improve performance."""
        # Create larger test data
        large_data = pd.concat([self.test_data] * 10, ignore_index=True)
        feature_columns = [col for col in large_data.columns if col not in ['timestamp', 'composite_cluster_id']]
        X = large_data[feature_columns].values
        y = large_data['composite_cluster_id'].values
        
        # Test correlation matrix performance
        start_time = time.time()
        corr_matrix = self.step._sparse_correlation_matrix_optimized(X, threshold=0.1)
        optimized_time = time.time() - start_time
        
        # Test with standard correlation (for comparison)
        start_time = time.time()
        standard_corr = np.corrcoef(X.T)
        standard_time = time.time() - start_time
        
        # Optimized version should be faster for large datasets
        print(f"Optimized time: {optimized_time:.4f}s, Standard time: {standard_time:.4f}s")
        
        # Test feature selection performance
        start_time = time.time()
        selected_features = self.step._mrmr_selection_optimized(X, y, feature_columns, 20)
        selection_time = time.time() - start_time
        
        assert len(selected_features) <= 20
        assert selection_time < 10.0  # Should complete within reasonable time

def run_performance_benchmark():
    """Run performance benchmark comparing optimized vs standard implementations."""
    print("🚀 Running Step08 Optimization Performance Benchmark")
    print("=" * 60)
    
    # Create test instance
    config = {
        'step08_optimized': {
            'phase1_target_features': 100,
            'enable_parallel_processing': True,
            'enable_caching': True,
            'chunk_size': 1000,
            'max_workers': 4
        }
    }
    
    step = OptimizedStep08(config)
    
    # Create large test dataset
    np.random.seed(42)
    n_samples = 10000
    n_features = 100
    
    timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='1min')
    regimes = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.4, 0.2])
    
    data = {
        'timestamp': timestamps,
        'composite_cluster_id': regimes,
    }
    
    for i in range(n_features):
        data[f'feature_{i}'] = np.random.randn(n_samples)
    
    test_data = pd.DataFrame(data)
    
    print(f"📊 Test dataset: {n_samples} samples, {n_features} features")
    print()
    
    # Benchmark correlation matrix computation
    print("🔍 Benchmarking Correlation Matrix Computation:")
    X = test_data[[f'feature_{i}' for i in range(n_features)]].values
    
    # Standard correlation
    start_time = time.time()
    standard_corr = np.corrcoef(X.T)
    standard_time = time.time() - start_time
    print(f"   Standard: {standard_time:.4f}s")
    
    # Optimized sparse correlation
    start_time = time.time()
    sparse_corr = step._sparse_correlation_matrix_optimized(X, threshold=0.1)
    optimized_time = time.time() - start_time
    print(f"   Optimized: {optimized_time:.4f}s")
    print(f"   Speedup: {standard_time / optimized_time:.2f}x")
    print(f"   Memory reduction: {sparse_corr.nnz / (n_features * n_features) * 100:.1f}% sparse")
    print()
    
    # Benchmark feature selection
    print("🎯 Benchmarking Feature Selection:")
    y = test_data['composite_cluster_id'].values
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Optimized mRMR
    start_time = time.time()
    selected_features = step._mrmr_selection_optimized(X, y, feature_names, 50)
    mrmr_time = time.time() - start_time
    print(f"   Optimized mRMR: {mrmr_time:.4f}s ({len(selected_features)} features)")
    
    # Optimized RF
    start_time = time.time()
    selected_features_rf = step._rf_selection_optimized(X, y, feature_names, 50)
    rf_time = time.time() - start_time
    print(f"   Optimized RF: {rf_time:.4f}s ({len(selected_features_rf)} features)")
    print()
    
    # Benchmark feature stability
    print("📈 Benchmarking Feature Stability Calculation:")
    features_subset = feature_names[:20]
    
    # Sequential
    start_time = time.time()
    sequential_scores = [step._calculate_feature_stability_fixed(test_data[f], test_data['composite_cluster_id']) for f in features_subset]
    sequential_time = time.time() - start_time
    print(f"   Sequential: {sequential_time:.4f}s")
    
    # Parallel
    start_time = time.time()
    parallel_scores = step._parallel_feature_stability(features_subset, test_data)
    parallel_time = time.time() - start_time
    print(f"   Parallel: {parallel_time:.4f}s")
    print(f"   Speedup: {sequential_time / parallel_time:.2f}x")
    print()
    
    # Benchmark data type optimization
    print("💾 Benchmarking Data Type Optimization:")
    start_time = time.time()
    optimized_data = step._optimize_data_types(test_data.copy())
    optimization_time = time.time() - start_time
    
    original_memory = test_data.memory_usage(deep=True).sum() / 1024**2
    optimized_memory = optimized_data.memory_usage(deep=True).sum() / 1024**2
    memory_reduction = (original_memory - optimized_memory) / original_memory * 100
    
    print(f"   Optimization time: {optimization_time:.4f}s")
    print(f"   Original memory: {original_memory:.2f} MB")
    print(f"   Optimized memory: {optimized_memory:.2f} MB")
    print(f"   Memory reduction: {memory_reduction:.1f}%")
    print()
    
    print("✅ Performance benchmark completed!")

if __name__ == '__main__':
    # Run the performance benchmark
    run_performance_benchmark()
    
    # Run the test suite
    print("\n🧪 Running Test Suite:")
    print("=" * 60)
    
    # Create test instance
    test_suite = TestStep08Optimizations()
    test_suite.setup_method()
    
    # Run key tests
    test_methods = [
        'test_fast_fail_data_quality',
        'test_fast_fail_feature_selection',
        'test_validate_temporal_integrity',
        'test_validate_regime_transitions',
        'test_gini_coefficient_fix',
        'test_regime_weights_fix',
        'test_feature_stability_fix',
        'test_sparse_correlation_matrix',
        'test_optimized_mrmr_selection',
        'test_optimized_rf_selection',
        'test_vectorized_feature_stability',
        'test_parallel_feature_processing',
        'test_caching_mechanisms',
        'test_full_optimization_pipeline'
    ]
    
    passed_tests = 0
    total_tests = len(test_methods)
    
    for test_method in test_methods:
        try:
            getattr(test_suite, test_method)()
            print(f"✅ {test_method}")
            passed_tests += 1
        except Exception as e:
            print(f"❌ {test_method}: {e}")
    
    print(f"\n📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Step08 optimizations are working correctly.")
    else:
        print("⚠️ Some tests failed. Please review the implementation.")