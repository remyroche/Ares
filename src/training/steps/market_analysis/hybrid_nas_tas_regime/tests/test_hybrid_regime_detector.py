"""
Tests for Hybrid NAS-TAS Regime Detector

Comprehensive tests for the core hybrid regime detection functionality.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import json
from pathlib import Path

from ..config.hybrid_regime_config import HybridRegimeConfig
from ..core.hybrid_regime_detector import HybridNASTASRegimeDetector, HybridRegimeResult


class TestHybridRegimeDetector(unittest.TestCase):
    """Test cases for Hybrid NAS-TAS Regime Detector."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = HybridRegimeConfig(n_regimes=4)
        self.detector = HybridNASTASRegimeDetector(self.config)

        # Create sample market data
        self.sample_data = self._create_sample_market_data(1000)

    def _create_sample_market_data(self, n_samples: int) -> pd.DataFrame:
        """Create sample market data for testing."""
        np.random.seed(42)  # For reproducible tests

        # Create time series data
        timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='1H')

        # Generate price data with different regimes
        prices = []
        current_price = 100.0

        for i in range(n_samples):
            # Simulate different volatility regimes
            if i < 250:  # Low volatility regime
                volatility = 0.01
                trend = 0.0001
            elif i < 500:  # Medium volatility regime
                volatility = 0.02
                trend = -0.0002
            elif i < 750:  # High volatility regime
                volatility = 0.05
                trend = 0.0003
            else:  # Mixed regime
                volatility = 0.03
                trend = 0.0001

            # Generate price movement
            price_change = np.random.normal(trend, volatility)
            current_price *= (1 + price_change)
            prices.append(current_price)

        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.lognormal(10, 1, n_samples)
        })

        return data

    def test_initialization(self):
        """Test detector initialization."""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.config.n_regimes, 4)
        self.assertIsNotNone(self.detector.tas_integration)
        self.assertIsNotNone(self.detector.nas_integration)
        self.assertIsNotNone(self.detector.economic_evaluator)

    def test_preprocess_market_data(self):
        """Test market data preprocessing."""
        # Test with DataFrame
        processed = self.detector._preprocess_market_data(self.sample_data)
        self.assertIsInstance(processed, pd.DataFrame)
        self.assertEqual(len(processed), len(self.sample_data))

        # Test with numpy array
        data_array = self.sample_data[['open', 'high', 'low', 'close', 'volume']].values
        processed_array = self.detector._preprocess_market_data(data_array)
        self.assertIsInstance(processed_array, pd.DataFrame)

        # Test error handling
        with self.assertRaises(ValueError):
            self.detector._preprocess_market_data("invalid_data")

    def test_detect_regimes_basic(self):
        """Test basic regime detection functionality."""
        result = self.detector.detect_regimes(self.sample_data)

        # Check result structure
        self.assertIsInstance(result, HybridRegimeResult)
        self.assertTrue(result.success)
        self.assertEqual(len(result.regime_predictions), len(self.sample_data))
        self.assertEqual(result.regime_probabilities.shape[0], len(self.sample_data))
        self.assertEqual(len(result.economic_significance_scores), self.config.n_regimes)
        self.assertEqual(len(result.financial_relevance_scores), self.config.n_regimes)

        # Check data types
        self.assertIsInstance(result.regime_predictions, np.ndarray)
        self.assertIsInstance(result.regime_probabilities, np.ndarray)
        self.assertIsInstance(result.economic_significance_scores, np.ndarray)
        self.assertIsInstance(result.financial_relevance_scores, np.ndarray)

    def test_detect_regimes_with_timestamps(self):
        """Test regime detection with custom timestamps."""
        timestamps = np.array([datetime.now() + timedelta(hours=i) for i in range(len(self.sample_data))])

        result = self.detector.detect_regimes(self.sample_data, timestamps)

        self.assertTrue(result.success)
        self.assertEqual(len(result.regime_predictions), len(self.sample_data))

    def test_feature_combination_strategies(self):
        """Test different feature combination strategies."""
        strategies = [
            self.config.combination_strategy.WEIGHTED_AVERAGE,
            self.config.combination_strategy.ENSEMBLE_VOTING,
            self.config.combination_strategy.ADAPTIVE_FUSION
        ]

        for strategy in strategies:
            with self.subTest(strategy=strategy):
                config = HybridRegimeConfig(
                    n_regimes=3,
                    combination_strategy=strategy
                )
                detector = HybridNASTASRegimeDetector(config)

                result = detector.detect_regimes(self.sample_data)
                self.assertTrue(result.success)

    def test_clustering_algorithms(self):
        """Test different clustering algorithms."""
        algorithms = ['adaptive', 'kmeans', 'gmm', 'agglomerative']

        for algorithm in algorithms:
            with self.subTest(algorithm=algorithm):
                config = HybridRegimeConfig(
                    n_regimes=3,
                    clustering_config={'algorithm': algorithm}
                )
                detector = HybridNASTASRegimeDetector(config)

                result = detector.detect_regimes(self.sample_data)
                self.assertTrue(result.success)

    def test_economic_evaluation(self):
        """Test economic significance evaluation."""
        # Create data with clear economic patterns
        economic_data = self._create_economic_test_data()

        result = self.detector.detect_regimes(economic_data, validate_economic_significance=True)

        self.assertTrue(result.success)
        self.assertGreater(np.mean(result.economic_significance_scores), 0.5)

    def test_financial_relevance(self):
        """Test financial relevance evaluation."""
        # Create data with clear financial patterns
        financial_data = self._create_financial_test_data()

        result = self.detector.detect_regimes(financial_data, validate_financial_relevance=True)

        self.assertTrue(result.success)
        self.assertGreater(np.mean(result.financial_relevance_scores), 0.5)

    def test_regime_stability_calculation(self):
        """Test regime stability calculation."""
        result = self.detector.detect_regimes(self.sample_data)

        self.assertTrue(result.success)
        self.assertEqual(len(result.regime_stability_scores), self.config.n_regimes)

        # Stability scores should be between 0 and 1
        self.assertTrue(all(0 <= score <= 1 for score in result.regime_stability_scores))

    def test_transition_probabilities(self):
        """Test transition probability calculation."""
        result = self.detector.detect_regimes(self.sample_data)

        self.assertTrue(result.success)
        n_regimes = self.config.n_regimes

        # Transition matrix should be n_regimes x n_regimes
        self.assertEqual(result.transition_probabilities.shape, (n_regimes, n_regimes))

        # Rows should sum to 1 (probabilities)
        row_sums = np.sum(result.transition_probabilities, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with empty data
        empty_data = pd.DataFrame()
        result = self.detector.detect_regimes(empty_data)
        self.assertFalse(result.success)

        # Test with very small dataset
        small_data = self._create_sample_market_data(5)
        result = self.detector.detect_regimes(small_data)
        self.assertTrue(result.success)  # Should handle gracefully

        # Test with invalid data types
        result = self.detector.detect_regimes("invalid_data")
        self.assertFalse(result.success)

    def test_fallback_mechanisms(self):
        """Test fallback mechanisms when components fail."""
        # Create detector with minimal configuration
        config = HybridRegimeConfig(n_regimes=2)
        detector = HybridNASTASRegimeDetector(config)

        result = detector.detect_regimes(self.sample_data)
        self.assertTrue(result.success)  # Should use fallback mechanisms

    def test_performance_tracking(self):
        """Test execution time tracking."""
        result = self.detector.detect_regimes(self.sample_data)

        self.assertTrue(result.success)
        self.assertGreater(result.execution_time, 0)
        self.assertLess(result.execution_time, 60)  # Should complete within 60 seconds

    def _create_economic_test_data(self) -> pd.DataFrame:
        """Create test data with clear economic patterns."""
        np.random.seed(42)
        n_samples = 500
        timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='1H')

        # Create data with distinct volatility regimes
        prices = []
        current_price = 100.0

        for i in range(n_samples):
            # High volatility regime (first 250 samples)
            if i < 250:
                volatility = 0.05
            else:
                volatility = 0.01  # Low volatility regime

            price_change = np.random.normal(0, volatility)
            current_price *= (1 + price_change)
            prices.append(current_price)

        return pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.lognormal(10, 1, n_samples)
        })

    def _create_financial_test_data(self) -> pd.DataFrame:
        """Create test data with clear financial patterns."""
        np.random.seed(42)
        n_samples = 500
        timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='1H')

        # Create data with trending behavior
        base_prices = 100 + np.linspace(0, 50, n_samples)  # Strong upward trend

        prices = []
        for i, base_price in enumerate(base_prices):
            # Add noise with varying magnitude
            noise = np.random.normal(0, base_price * 0.02)
            prices.append(base_price + noise)

        return pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.lognormal(10, 1, n_samples)
        })


class TestHybridRegimeIntegration(unittest.TestCase):
    """Integration tests for hybrid regime detection."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.config = HybridRegimeConfig(n_regimes=4)
        self.detector = HybridNASTASRegimeDetector(self.config)
        self.sample_data = self._create_large_sample_data(2000)

    def _create_large_sample_data(self, n_samples: int) -> pd.DataFrame:
        """Create larger sample dataset for integration testing."""
        np.random.seed(42)
        timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='1H')

        # Create more complex multi-regime data
        prices = []
        current_price = 100.0

        for i in range(n_samples):
            # Four distinct regimes
            if i % 4 == 0:  # Regime 0: High volatility, downward trend
                volatility = 0.04
                trend = -0.001
            elif i % 4 == 1:  # Regime 1: Low volatility, upward trend
                volatility = 0.01
                trend = 0.0005
            elif i % 4 == 2:  # Regime 2: Medium volatility, sideways
                volatility = 0.02
                trend = 0.0001
            else:  # Regime 3: High volatility, random
                volatility = 0.06
                trend = 0.0002

            price_change = np.random.normal(trend, volatility)
            current_price *= (1 + price_change)
            prices.append(current_price)

        return pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.lognormal(10, 1, n_samples)
        })

    def test_end_to_end_detection(self):
        """Test complete end-to-end regime detection."""
        result = self.detector.detect_regimes(
            self.sample_data,
            validate_economic_significance=True,
            validate_financial_relevance=True
        )

        self.assertTrue(result.success)
        self.assertEqual(len(result.regime_predictions), len(self.sample_data))

        # Verify all components of result are present
        self.assertGreater(len(result.economic_significance_scores), 0)
        self.assertGreater(len(result.financial_relevance_scores), 0)
        self.assertGreater(len(result.regime_stability_scores), 0)
        self.assertEqual(result.transition_probabilities.shape[0], self.config.n_regimes)

    def test_regime_consistency(self):
        """Test that regime detection is consistent across runs."""
        # Run detection multiple times
        results = []
        for i in range(3):
            result = self.detector.detect_regimes(self.sample_data)
            if result.success:
                results.append(result)

        self.assertGreater(len(results), 0)

        # Check consistency in number of regimes detected
        n_regimes_list = [len(set(r.regime_predictions)) for r in results]
        self.assertEqual(len(set(n_regimes_list)), 1)  # All should detect same number

        # Check consistency in economic significance scores
        econ_scores = [r.economic_significance_scores for r in results]
        avg_econ_scores = np.mean(econ_scores, axis=0)
        std_econ_scores = np.std(econ_scores, axis=0)

        # Standard deviation should be reasonable
        self.assertLess(np.mean(std_econ_scores), 0.2)


if __name__ == '__main__':
    unittest.main()