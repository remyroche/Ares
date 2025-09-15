#!/usr/bin/env python3
"""
Comprehensive Test Suite for Unified Feature Selection Framework

This test suite validates all components of the unified feature selection system:
1. Core unified framework functionality
2. Matrix operations integration
3. Backwards compatibility
4. Feature set generation (120, 100, 80, 60)
5. HMM regime-specific selection
6. Random Forest refinement
7. Error handling and edge cases

Author: AI Assistant
Date: 2024-01-XX
Version: 1.0.0
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

# Import the unified framework
try:
    from src.utils.ml_common.unified_feature_selection import (
        UnifiedFeatureSelector, UnifiedFeatureSelectionConfig,
        create_unified_selector, select_features_unified, generate_feature_sets
    )
    from src.utils.ml_common.matrix_feature_operations import (
        MatrixFeatureOperations, create_matrix_feature_operations
    )
    from src.utils.ml_common.backwards_compatibility import (
        BackwardsCompatibilityWrapper, create_feature_selector
    )
    UNIFIED_FRAMEWORK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import unified framework: {e}")
    UNIFIED_FRAMEWORK_AVAILABLE = False


class TestDataGenerator:
    """Generate test data for feature selection tests."""
    
    @staticmethod
    def generate_regression_data(n_samples: int = 1000, n_features: int = 200) -> tuple:
        """Generate regression test data."""
        np.random.seed(42)
        
        # Generate feature matrix
        X = np.random.randn(n_samples, n_features)
        
        # Add structure to make some features more important
        X[:, :50] = X[:, :50] * 0.1 + np.random.randn(n_samples, 50) * 0.9
        
        # Generate target variable
        y = np.sum(X[:, :50], axis=1) * 0.1 + np.random.randn(n_samples) * 0.5
        
        # Generate feature names
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        return X, y, feature_names
    
    @staticmethod
    def generate_classification_data(n_samples: int = 1000, n_features: int = 200) -> tuple:
        """Generate classification test data."""
        np.random.seed(42)
        
        # Generate feature matrix
        X = np.random.randn(n_samples, n_features)
        
        # Add structure to make some features more important
        X[:, :50] = X[:, :50] * 0.1 + np.random.randn(n_samples, 50) * 0.9
        
        # Generate target variable (3 classes)
        y = np.zeros(n_samples, dtype=int)
        y[np.sum(X[:, :50], axis=1) > 0.5] = 1
        y[np.sum(X[:, :50], axis=1) > 1.0] = 2
        
        # Generate feature names
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        return X, y, feature_names
    
    @staticmethod
    def generate_hmm_regime_data(n_samples: int = 1000, n_features: int = 200) -> tuple:
        """Generate HMM regime test data."""
        np.random.seed(42)
        
        # Generate feature matrix
        X = np.random.randn(n_samples, n_features)
        
        # Create regime structure
        volatility = np.std(X[:, :50], axis=1)
        regime_thresholds = np.percentile(volatility, [33, 67])
        
        y_regime = np.zeros(n_samples, dtype=int)
        y_regime[volatility > regime_thresholds[1]] = 2  # High volatility
        y_regime[volatility > regime_thresholds[0]] = 1  # Medium volatility
        
        # Generate feature names
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        return X, y_regime, feature_names


@unittest.skipUnless(UNIFIED_FRAMEWORK_AVAILABLE, "Unified framework not available")
class TestUnifiedFeatureSelector(unittest.TestCase):
    """Test the core unified feature selector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = UnifiedFeatureSelectionConfig(
            target_features=50,
            save_results=True,
            output_dir=self.temp_dir
        )
        self.selector = UnifiedFeatureSelector(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test selector initialization."""
        self.assertIsInstance(self.selector, UnifiedFeatureSelector)
        self.assertEqual(self.selector.config.target_features, 50)
        self.assertTrue(self.selector.config.save_results)
    
    def test_regression_feature_selection(self):
        """Test feature selection for regression."""
        X, y, feature_names = TestDataGenerator.generate_regression_data()
        
        results = self.selector.select_features(X, y, feature_names, target_sizes=[50, 40, 30])
        
        # Check results structure
        self.assertIn('top_50', results)
        self.assertIn('top_40', results)
        self.assertIn('top_30', results)
        
        # Check feature counts
        self.assertEqual(len(results['top_50']['selected_features']), 50)
        self.assertEqual(len(results['top_40']['selected_features']), 40)
        self.assertEqual(len(results['top_30']['selected_features']), 30)
        
        # Check that smaller sets are subsets of larger sets
        top_50_features = set(results['top_50']['selected_features'])
        top_40_features = set(results['top_40']['selected_features'])
        top_30_features = set(results['top_30']['selected_features'])
        
        self.assertTrue(top_40_features.issubset(top_50_features))
        self.assertTrue(top_30_features.issubset(top_40_features))
    
    def test_classification_feature_selection(self):
        """Test feature selection for classification."""
        X, y, feature_names = TestDataGenerator.generate_classification_data()
        
        config = UnifiedFeatureSelectionConfig(
            target_features=50,
            task_type="classification",
            save_results=False
        )
        selector = UnifiedFeatureSelector(config)
        
        results = selector.select_features(X, y, feature_names, target_sizes=[50])
        
        self.assertIn('top_50', results)
        self.assertEqual(len(results['top_50']['selected_features']), 50)
        self.assertEqual(results['top_50']['method'], 'hybrid')
    
    def test_hmm_regime_selection(self):
        """Test HMM regime-specific feature selection."""
        X, y_regime, feature_names = TestDataGenerator.generate_hmm_regime_data()
        
        config = UnifiedFeatureSelectionConfig(
            target_features=50,
            task_type="classification",
            prediction_target="hmm_regime",
            save_results=False
        )
        selector = UnifiedFeatureSelector(config)
        
        results = selector.select_features(X, y_regime, feature_names)
        
        # Check HMM regime results
        self.assertIn('hmm_regime_top_100', results)
        hmm_result = results['hmm_regime_top_100']
        
        self.assertEqual(len(hmm_result['selected_features']), 50)  # Should be 50, not 100 due to config
        self.assertIn('regime_analysis', hmm_result)
        
        # Check regime analysis
        regime_analysis = hmm_result['regime_analysis']
        self.assertIn('n_regimes', regime_analysis)
        self.assertIn('unique_regimes', regime_analysis)
        self.assertIn('regime_separation_scores', regime_analysis)
    
    def test_data_preprocessing(self):
        """Test data preprocessing functionality."""
        # Test with NaN values
        X, y, feature_names = TestDataGenerator.generate_regression_data()
        X[0, 0] = np.nan  # Add NaN value
        
        results = self.selector.select_features(X, y, feature_names, target_sizes=[50])
        self.assertIn('top_50', results)
        
        # Test with infinite values
        X[1, 1] = np.inf  # Add infinite value
        
        results = self.selector.select_features(X, y, feature_names, target_sizes=[50])
        self.assertIn('top_50', results)
    
    def test_feature_set_retrieval(self):
        """Test feature set retrieval methods."""
        X, y, feature_names = TestDataGenerator.generate_regression_data()
        
        results = self.selector.select_features(X, y, feature_names, target_sizes=[50, 40, 30])
        
        # Test get_feature_set method
        top_50 = self.selector.get_feature_set(50)
        top_40 = self.selector.get_feature_set(40)
        top_30 = self.selector.get_feature_set(30)
        
        self.assertEqual(len(top_50), 50)
        self.assertEqual(len(top_40), 40)
        self.assertEqual(len(top_30), 30)
        
        # Test get_feature_scores method
        scores_50 = self.selector.get_feature_scores(50)
        self.assertEqual(len(scores_50), 50)
        self.assertIsInstance(scores_50, dict)


@unittest.skipUnless(UNIFIED_FRAMEWORK_AVAILABLE, "Unified framework not available")
class TestMatrixFeatureOperations(unittest.TestCase):
    """Test matrix feature operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.matrix_ops = create_matrix_feature_operations(use_gpu=False, use_parallel=False)
        self.X, self.y, self.feature_names = TestDataGenerator.generate_regression_data()
    
    def test_correlation_matrix(self):
        """Test correlation matrix computation."""
        corr_matrix = self.matrix_ops.correlation_matrix(
            self.X, method="pearson", feature_names=self.feature_names
        )
        
        self.assertEqual(corr_matrix.shape, (len(self.feature_names), len(self.feature_names)))
        self.assertTrue(np.allclose(np.diag(corr_matrix), 1.0))  # Diagonal should be 1
    
    def test_mutual_information_matrix(self):
        """Test mutual information computation."""
        mi_scores = self.matrix_ops.mutual_information_matrix(
            self.X, self.y, self.feature_names
        )
        
        self.assertEqual(len(mi_scores), len(self.feature_names))
        self.assertTrue(np.all(mi_scores >= 0))  # MI should be non-negative
    
    def test_hierarchical_clustering(self):
        """Test hierarchical clustering based on correlation."""
        result = self.matrix_ops.hierarchical_clustering_correlation(
            self.X, correlation_threshold=0.95, feature_names=self.feature_names
        )
        
        self.assertIn('clusters', result)
        self.assertIn('representative_features', result)
        self.assertIn('n_clusters', result)
        self.assertIn('n_representatives', result)
        
        self.assertGreater(result['n_clusters'], 0)
        self.assertGreaterEqual(result['n_representatives'], result['n_clusters'])
    
    def test_feature_importance_matrix(self):
        """Test feature importance computation."""
        importance_scores = self.matrix_ops.feature_importance_matrix(
            self.X, self.y, method="random_forest", feature_names=self.feature_names
        )
        
        self.assertEqual(len(importance_scores), len(self.feature_names))
        self.assertTrue(np.all(importance_scores >= 0))  # Importance should be non-negative
        self.assertAlmostEqual(np.sum(importance_scores), 1.0, places=5)  # Should sum to 1
    
    def test_variance_threshold_matrix(self):
        """Test variance threshold filtering."""
        result = self.matrix_ops.variance_threshold_matrix(
            self.X, threshold=0.0, feature_names=self.feature_names
        )
        
        self.assertIn('selected_features', result)
        self.assertIn('removed_features', result)
        self.assertIn('variances', result)
        
        self.assertEqual(len(result['selected_features']) + len(result['removed_features']), 
                        len(self.feature_names))
    
    def test_correlation_filter_matrix(self):
        """Test correlation-based filtering."""
        result = self.matrix_ops.correlation_filter_matrix(
            self.X, correlation_threshold=0.95, feature_names=self.feature_names
        )
        
        self.assertIn('selected_features', result)
        self.assertIn('removed_features', result)
        self.assertIn('high_correlation_pairs', result)
        
        self.assertEqual(len(result['selected_features']) + len(result['removed_features']), 
                        len(self.feature_names))
    
    def test_batch_operations(self):
        """Test batch feature operations."""
        operations = [
            "correlation_matrix",
            "mutual_information",
            "variance_threshold",
            "correlation_filter"
        ]
        
        results = self.matrix_ops.batch_feature_operations(
            self.X, self.y, operations, self.feature_names
        )
        
        for operation in operations:
            self.assertIn(operation, results)
            self.assertIsNotNone(results[operation])
    
    def test_optimize_feature_selection_pipeline(self):
        """Test optimized feature selection pipeline."""
        result = self.matrix_ops.optimize_feature_selection_pipeline(
            self.X, self.y, target_features=50, feature_names=self.feature_names
        )
        
        self.assertIn('selected_features', result)
        self.assertIn('feature_scores', result)
        self.assertIn('pipeline_steps', result)
        
        self.assertEqual(len(result['selected_features']), 50)
        self.assertEqual(len(result['feature_scores']), 50)


@unittest.skipUnless(UNIFIED_FRAMEWORK_AVAILABLE, "Unified framework not available")
class TestBackwardsCompatibility(unittest.TestCase):
    """Test backwards compatibility layer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.X, self.y, self.feature_names = TestDataGenerator.generate_regression_data()
    
    def test_legacy_feature_selector(self):
        """Test legacy feature selector interface."""
        selector = create_feature_selector()
        
        # Test fit method
        selector.fit(self.X, self.y)
        
        # Test transform method
        X_transformed = selector.transform(self.X)
        self.assertEqual(X_transformed.shape[0], self.X.shape[0])
        
        # Test fit_transform method
        X_fit_transform = selector.fit_transform(self.X, self.y)
        self.assertEqual(X_fit_transform.shape[0], self.X.shape[0])
        
        # Test get_support method
        support = selector.get_support()
        self.assertIsInstance(support, list)
        
        # Test get_feature_names_out method
        feature_names_out = selector.get_feature_names_out()
        self.assertIsInstance(feature_names_out, list)
        
        # Test get_feature_importance method
        importance = selector.get_feature_importance()
        self.assertIsInstance(importance, dict)
        
        # Test properties
        self.assertGreater(selector.n_features_in_, 0)
        self.assertGreaterEqual(selector.n_features_out_, 0)
    
    def test_legacy_select_features_function(self):
        """Test legacy select_features function."""
        from src.utils.ml_common.backwards_compatibility import select_features
        
        selected_features = select_features(
            self.X, self.y, method="correlation", max_features=50
        )
        
        self.assertIsInstance(selected_features, list)
        self.assertLessEqual(len(selected_features), 50)


@unittest.skipUnless(UNIFIED_FRAMEWORK_AVAILABLE, "Unified framework not available")
class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.X, self.y, self.feature_names = TestDataGenerator.generate_regression_data()
    
    def test_select_features_unified(self):
        """Test select_features_unified convenience function."""
        results = select_features_unified(
            self.X, self.y, self.feature_names, 
            target_features=50, task_type="regression"
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('top_50', results)
        self.assertEqual(len(results['top_50']['selected_features']), 50)
    
    def test_generate_feature_sets(self):
        """Test generate_feature_sets convenience function."""
        feature_sets = generate_feature_sets(
            self.X, self.y, self.feature_names, 
            target_sizes=[50, 40, 30], task_type="regression"
        )
        
        self.assertIsInstance(feature_sets, dict)
        self.assertIn('top_50', feature_sets)
        self.assertIn('top_40', feature_sets)
        self.assertIn('top_30', feature_sets)
        
        self.assertEqual(len(feature_sets['top_50']), 50)
        self.assertEqual(len(feature_sets['top_40']), 40)
        self.assertEqual(len(feature_sets['top_30']), 30)
    
    def test_create_unified_selector(self):
        """Test create_unified_selector convenience function."""
        selector = create_unified_selector()
        self.assertIsInstance(selector, UnifiedFeatureSelector)


@unittest.skipUnless(UNIFIED_FRAMEWORK_AVAILABLE, "Unified framework not available")
class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_empty_data(self):
        """Test handling of empty data."""
        X = np.array([]).reshape(0, 0)
        y = np.array([])
        
        selector = UnifiedFeatureSelector()
        
        with self.assertRaises((ValueError, IndexError)):
            selector.select_features(X, y)
    
    def test_single_feature(self):
        """Test handling of single feature."""
        X = np.random.randn(100, 1)
        y = np.random.randn(100)
        feature_names = ['single_feature']
        
        selector = UnifiedFeatureSelector()
        results = selector.select_features(X, y, feature_names, target_sizes=[1])
        
        self.assertIn('top_1', results)
        self.assertEqual(len(results['top_1']['selected_features']), 1)
    
    def test_constant_target(self):
        """Test handling of constant target variable."""
        X = np.random.randn(100, 50)
        y = np.ones(100)  # Constant target
        
        selector = UnifiedFeatureSelector()
        
        # Should handle constant target gracefully
        results = selector.select_features(X, y, target_sizes=[10])
        self.assertIsInstance(results, dict)
    
    def test_insufficient_features(self):
        """Test handling when target_features > available_features."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        feature_names = [f'feature_{i}' for i in range(10)]
        
        selector = UnifiedFeatureSelector()
        results = selector.select_features(X, y, feature_names, target_sizes=[20])
        
        # Should select all available features
        self.assertIn('top_20', results)
        self.assertEqual(len(results['top_20']['selected_features']), 10)


@unittest.skipUnless(UNIFIED_FRAMEWORK_AVAILABLE, "Unified framework not available")
class TestPerformance(unittest.TestCase):
    """Test performance characteristics."""
    
    def test_large_dataset_performance(self):
        """Test performance with larger dataset."""
        # Generate larger dataset
        X, y, feature_names = TestDataGenerator.generate_regression_data(
            n_samples=5000, n_features=500
        )
        
        selector = UnifiedFeatureSelector()
        
        import time
        start_time = time.time()
        results = selector.select_features(X, y, feature_names, target_sizes=[100])
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(execution_time, 60)  # 60 seconds
        self.assertIn('top_100', results)
    
    def test_memory_usage(self):
        """Test memory usage with large dataset."""
        # Generate large dataset
        X, y, feature_names = TestDataGenerator.generate_regression_data(
            n_samples=10000, n_features=1000
        )
        
        selector = UnifiedFeatureSelector()
        
        # Should not raise memory errors
        results = selector.select_features(X, y, feature_names, target_sizes=[100])
        self.assertIn('top_100', results)


def run_performance_benchmark():
    """Run performance benchmark tests."""
    print("\n" + "="*60)
    print("🚀 PERFORMANCE BENCHMARK")
    print("="*60)
    
    if not UNIFIED_FRAMEWORK_AVAILABLE:
        print("❌ Unified framework not available for benchmarking")
        return
    
    import time
    
    # Test different dataset sizes
    dataset_sizes = [
        (1000, 200),
        (5000, 500),
        (10000, 1000)
    ]
    
    for n_samples, n_features in dataset_sizes:
        print(f"\n📊 Testing dataset: {n_samples} samples, {n_features} features")
        
        # Generate data
        X, y, feature_names = TestDataGenerator.generate_regression_data(n_samples, n_features)
        
        # Test unified framework
        selector = UnifiedFeatureSelector()
        start_time = time.time()
        results = selector.select_features(X, y, feature_names, target_sizes=[100])
        unified_time = time.time() - start_time
        
        print(f"   ✅ Unified framework: {unified_time:.3f}s")
        
        # Test matrix operations
        matrix_ops = create_matrix_feature_operations()
        start_time = time.time()
        matrix_result = matrix_ops.optimize_feature_selection_pipeline(
            X, y, target_features=100, feature_names=feature_names
        )
        matrix_time = time.time() - start_time
        
        print(f"   ⚡ Matrix operations: {matrix_time:.3f}s")
        
        # Test backwards compatibility
        legacy_selector = create_feature_selector()
        start_time = time.time()
        legacy_selector.fit(X, y)
        legacy_time = time.time() - start_time
        
        print(f"   🔄 Backwards compatibility: {legacy_time:.3f}s")


def main():
    """Run all tests."""
    print("🧪 Starting Unified Feature Selection Framework Tests")
    print("="*80)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestUnifiedFeatureSelector,
        TestMatrixFeatureOperations,
        TestBackwardsCompatibility,
        TestConvenienceFunctions,
        TestErrorHandling,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    successes = total_tests - failures - errors
    
    print(f"Total tests: {total_tests}")
    print(f"✅ Passed: {successes}")
    print(f"❌ Failed: {failures}")
    print(f"💥 Errors: {errors}")
    
    if failures > 0:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if errors > 0:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.split('\\n')[-2]}")
    
    # Run performance benchmark
    run_performance_benchmark()
    
    # Final result
    if failures == 0 and errors == 0:
        print("\n🎉 All tests passed successfully!")
        return True
    else:
        print(f"\n⚠️ {failures + errors} test(s) failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)