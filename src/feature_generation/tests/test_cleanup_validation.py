"""
Test Cleanup Validation

This module tests that the duplicate cleanup didn't break any functionality
and that the new base class methods work correctly.
"""

import unittest
import pandas as pd
import numpy as np
from typing import List, Optional

from ..core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
from ..core.optimization_mixin import OptimizationMixin
from ..core.rolling_operations_mixin import RollingOperationsMixin
from ..core.generator_factory import GeneratorFactory

class TestCleanupValidation(unittest.TestCase):
    """Test that cleanup didn't break functionality."""

    def setUp(self):
        """Set up test data."""
        # Create sample data
        np.random.seed(42)
        self.data = pd.DataFrame({
            'close': np.random.randn(1000) + 100,
            'volume': np.random.randint(1000, 10000, 1000),
            'high': np.random.randn(1000) + 105,
            'low': np.random.randn(1000) + 95
        })

        # Create test config
        self.config = FeatureConfig(
            name="test_feature",
            category=FeatureCategory.CUSTOM,
            description="Test feature for cleanup validation",
            required_columns=["close"],
            optional_columns=["volume", "high", "low"]
        )

    def test_vectorized_generator_inheritance(self):
        """Test that VectorizedFeatureGenerator has the required methods."""
        generator = VectorizedFeatureGenerator(self.config)

        # Check that methods exist
        self.assertTrue(hasattr(generator, 'optimize_dataframe_processing'))
        self.assertTrue(hasattr(generator, 'vectorized_rolling_operations'))

        # Check that methods are callable
        self.assertTrue(callable(generator.optimize_dataframe_processing))
        self.assertTrue(callable(generator.vectorized_rolling_operations))

    def test_optimize_dataframe_processing(self):
        """Test optimize_dataframe_processing method."""
        generator = VectorizedFeatureGenerator(self.config)

        # Test with sample data
        result = generator.optimize_dataframe_processing(self.data)

        # Check that result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check that result has same shape
        self.assertEqual(result.shape, self.data.shape)

        # Check that result has same columns
        self.assertEqual(list(result.columns), list(self.data.columns))

    def test_vectorized_rolling_operations(self):
        """Test vectorized_rolling_operations method."""
        generator = VectorizedFeatureGenerator(self.config)

        # Test with sample data
        result = generator.vectorized_rolling_operations(
            self.data,
            operations=['mean', 'std'],
            windows=[20, 50],
            columns=['close']
        )

        # Check that result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check that new columns were added
        expected_columns = ['close_mean_20', 'close_std_20', 'close_mean_50', 'close_std_50']
        for col in expected_columns:
            self.assertIn(col, result.columns)

    def test_optimization_mixin(self):
        """Test OptimizationMixin functionality."""
        class TestGenerator(VectorizedFeatureGenerator, OptimizationMixin):
            def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                return data['close']

        generator = TestGenerator(self.config)

        # Test optimize_dataframe_processing
        result = generator.optimize_dataframe_processing(self.data)
        self.assertIsInstance(result, pd.DataFrame)

        # Test chunked_processing
        def test_func(data):
            return data['close'].rolling(20).mean()

        result = generator.chunked_processing(self.data, test_func)
        self.assertIsInstance(result, pd.Series)

    def test_rolling_operations_mixin(self):
        """Test RollingOperationsMixin functionality."""
        class TestGenerator(VectorizedFeatureGenerator, RollingOperationsMixin):
            def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                return data['close']

        generator = TestGenerator(self.config)

        # Test rolling operations
        result = generator.rolling_mean(self.data['close'], window=20)
        self.assertIsInstance(result, pd.Series)

        result = generator.rolling_std(self.data['close'], window=20)
        self.assertIsInstance(result, pd.Series)

        # Test batch operations
        operations = [
            {'column': 'close', 'operation': 'mean', 'window': 20, 'name': 'close_mean_20'},
            {'column': 'close', 'operation': 'std', 'window': 20, 'name': 'close_std_20'}
        ]

        result = generator.batch_rolling_operations(self.data, operations)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('close_mean_20', result.columns)
        self.assertIn('close_std_20', result.columns)

    def test_generator_factory(self):
        """Test GeneratorFactory functionality."""
        factory = GeneratorFactory()

        # Test creating a vectorized generator
        generator = factory.create_vectorized_generator(
            name="test_sma",
            category=FeatureCategory.CUSTOM,
            required_columns=["close"],
            window=20
        )

        self.assertIsNotNone(generator)
        self.assertIsInstance(generator, VectorizedFeatureGenerator)

        # Test creating an optimized generator
        generator = factory.create_optimized_generator(
            name="test_optimized",
            category=FeatureCategory.CUSTOM,
            required_columns=["close"]
        )

        self.assertIsNotNone(generator)
        self.assertIsInstance(generator, VectorizedFeatureGenerator)

    def test_performance_stats(self):
        """Test performance statistics tracking."""
        generator = VectorizedFeatureGenerator(self.config)

        # Perform some operations
        generator.optimize_dataframe_processing(self.data)
        generator.vectorized_rolling_operations(
            self.data, ['mean'], [20], ['close']
        )

        # Check performance stats
        stats = generator.get_performance_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_generations', stats)
        self.assertIn('successful_generations', stats)

    def test_memory_optimization(self):
        """Test memory optimization functionality."""
        class TestGenerator(VectorizedFeatureGenerator, OptimizationMixin):
            def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                return data['close']

        generator = TestGenerator(self.config)

        # Test with large data
        large_data = pd.DataFrame({
            'close': np.random.randn(10000) + 100,
            'volume': np.random.randint(1000, 10000, 10000)
        })

        result = generator.optimize_dataframe_processing(large_data)
        self.assertIsInstance(result, pd.DataFrame)

        # Check optimization stats
        stats = generator.get_optimization_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('memory_optimizations', stats)

    def test_error_handling(self):
        """Test error handling in base class methods."""
        generator = VectorizedFeatureGenerator(self.config)

        # Test with invalid data
        with self.assertRaises(Exception):
            generator.optimize_dataframe_processing(None)

        # Test with empty DataFrame
        empty_data = pd.DataFrame()
        result = generator.optimize_dataframe_processing(empty_data)
        self.assertIsInstance(result, pd.DataFrame)

    def test_backward_compatibility(self):
        """Test that existing code still works."""
        # This test ensures that the cleanup didn't break existing functionality
        generator = VectorizedFeatureGenerator(self.config)

        # Test that methods work as expected
        result = generator.optimize_dataframe_processing(self.data)
        self.assertIsInstance(result, pd.DataFrame)

        result = generator.vectorized_rolling_operations(
            self.data, ['mean'], [20], ['close']
        )
        self.assertIsInstance(result, pd.DataFrame)

        # Test that performance stats are available
        stats = generator.get_performance_stats()
        self.assertIsInstance(stats, dict)

if __name__ == '__main__':
    unittest.main()
