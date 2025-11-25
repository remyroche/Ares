"""
Comprehensive tests for Regime Probability Aggregator.

Tests all aggregation methods, isotonic calibration, and integration workflows.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.utils.regime_probability_aggregator import (
    RegimeProbabilityAggregator,
    AggregationMethod,
    compare_aggregation_methods
)


@pytest.fixture
def synthetic_regime_data():
    """
    Generate synthetic regime probabilities and forward returns.

    Simulates a 3-regime system:
    - Regime 0: Low risk (mean return = +0.002)
    - Regime 1: Medium risk (mean return = +0.001)
    - Regime 2: High risk (mean return = -0.003)
    """
    np.random.seed(42)
    n_samples = 1000
    n_regimes = 3

    # Generate regime probabilities (sum to 1)
    regime_probs = np.random.dirichlet(alpha=np.ones(n_regimes), size=n_samples)

    # Generate forward returns based on regime probabilities
    regime_means = np.array([0.002, 0.001, -0.003])  # Low, medium, high risk
    regime_stds = np.array([0.01, 0.02, 0.04])

    forward_returns = np.zeros(n_samples)
    for i in range(n_samples):
        # Sample from regime mixture
        regime_idx = np.random.choice(n_regimes, p=regime_probs[i])
        forward_returns[i] = np.random.normal(
            regime_means[regime_idx],
            regime_stds[regime_idx]
        )

    # Regime statistics (ground truth)
    regime_stats = {
        0: {
            'mean_return': 0.002,
            'std_return': 0.01,
            'sharpe_ratio': 0.2,
            'count': 333
        },
        1: {
            'mean_return': 0.001,
            'std_return': 0.02,
            'sharpe_ratio': 0.05,
            'count': 333
        },
        2: {
            'mean_return': -0.003,
            'std_return': 0.04,
            'sharpe_ratio': -0.075,
            'count': 334
        }
    }

    return {
        'regime_probs': regime_probs,
        'forward_returns': forward_returns,
        'regime_stats': regime_stats,
        'n_regimes': n_regimes
    }


class TestRegimeProbabilityAggregator:
    """Tests for RegimeProbabilityAggregator class."""

    def test_initialization(self):
        """Test aggregator initialization."""
        aggregator = RegimeProbabilityAggregator(
            method=AggregationMethod.EXPECTED_RETURN,
            isotonic_calibration=True
        )

        assert aggregator.method == AggregationMethod.EXPECTED_RETURN
        assert aggregator.isotonic_calibration is True
        assert aggregator.is_fitted is False

    def test_expected_return_aggregation(self, synthetic_regime_data):
        """Test expected return weighted aggregation."""
        aggregator = RegimeProbabilityAggregator(
            method=AggregationMethod.EXPECTED_RETURN,
            isotonic_calibration=False
        )

        aggregator.fit(
            synthetic_regime_data['regime_probs'],
            synthetic_regime_data['forward_returns'],
            synthetic_regime_data['regime_stats']
        )

        assert aggregator.is_fitted
        assert aggregator.regime_weights is not None
        assert len(aggregator.regime_weights) == synthetic_regime_data['n_regimes']

        # Transform
        scores = aggregator.transform(synthetic_regime_data['regime_probs'])

        assert len(scores) == len(synthetic_regime_data['forward_returns'])
        assert np.all((scores >= 0) & (scores <= 1))
        assert np.all(np.isfinite(scores))

    def test_expected_sharpe_aggregation(self, synthetic_regime_data):
        """Test expected Sharpe ratio weighted aggregation."""
        aggregator = RegimeProbabilityAggregator(
            method=AggregationMethod.EXPECTED_SHARPE,
            isotonic_calibration=False
        )

        aggregator.fit(
            synthetic_regime_data['regime_probs'],
            synthetic_regime_data['forward_returns'],
            synthetic_regime_data['regime_stats']
        )

        assert aggregator.is_fitted
        assert aggregator.regime_weights is not None

        # Transform
        scores = aggregator.transform(synthetic_regime_data['regime_probs'])

        assert len(scores) == len(synthetic_regime_data['forward_returns'])
        assert np.all((scores >= 0) & (scores <= 1))

    def test_logistic_regression_aggregation(self, synthetic_regime_data):
        """Test logistic regression aggregation."""
        aggregator = RegimeProbabilityAggregator(
            method=AggregationMethod.LOGISTIC_REGRESSION,
            isotonic_calibration=False
        )

        aggregator.fit(
            synthetic_regime_data['regime_probs'],
            synthetic_regime_data['forward_returns']
        )

        assert aggregator.is_fitted
        assert aggregator.model is not None

        # Transform
        scores = aggregator.transform(synthetic_regime_data['regime_probs'])

        assert len(scores) == len(synthetic_regime_data['forward_returns'])
        assert np.all((scores >= 0) & (scores <= 1))

    def test_neural_network_aggregation(self, synthetic_regime_data):
        """Test neural network aggregation."""
        aggregator = RegimeProbabilityAggregator(
            method=AggregationMethod.NEURAL_NETWORK,
            isotonic_calibration=False
        )

        aggregator.fit(
            synthetic_regime_data['regime_probs'],
            synthetic_regime_data['forward_returns']
        )

        assert aggregator.is_fitted
        assert aggregator.model is not None

        # Transform
        scores = aggregator.transform(synthetic_regime_data['regime_probs'])

        assert len(scores) == len(synthetic_regime_data['forward_returns'])
        assert np.all((scores >= 0) & (scores <= 1))

    def test_pca_aggregation(self, synthetic_regime_data):
        """Test PCA first component aggregation."""
        aggregator = RegimeProbabilityAggregator(
            method=AggregationMethod.PCA_FIRST,
            isotonic_calibration=False
        )

        aggregator.fit(
            synthetic_regime_data['regime_probs'],
            synthetic_regime_data['forward_returns']
        )

        assert aggregator.is_fitted
        assert aggregator.model is not None

        # Transform
        scores = aggregator.transform(synthetic_regime_data['regime_probs'])

        assert len(scores) == len(synthetic_regime_data['forward_returns'])
        assert np.all((scores >= 0) & (scores <= 1))

    def test_isotonic_calibration(self, synthetic_regime_data):
        """Test isotonic calibration."""
        # Without calibration
        aggregator_no_calib = RegimeProbabilityAggregator(
            method=AggregationMethod.EXPECTED_RETURN,
            isotonic_calibration=False
        )
        aggregator_no_calib.fit(
            synthetic_regime_data['regime_probs'],
            synthetic_regime_data['forward_returns'],
            synthetic_regime_data['regime_stats']
        )
        scores_no_calib = aggregator_no_calib.transform(
            synthetic_regime_data['regime_probs']
        )

        # With calibration
        aggregator_calib = RegimeProbabilityAggregator(
            method=AggregationMethod.EXPECTED_RETURN,
            isotonic_calibration=True
        )
        aggregator_calib.fit(
            synthetic_regime_data['regime_probs'],
            synthetic_regime_data['forward_returns'],
            synthetic_regime_data['regime_stats']
        )
        scores_calib = aggregator_calib.transform(
            synthetic_regime_data['regime_probs']
        )

        # Both should be in [0, 1]
        assert np.all((scores_no_calib >= 0) & (scores_no_calib <= 1))
        assert np.all((scores_calib >= 0) & (scores_calib <= 1))

        # Calibration should exist
        assert aggregator_calib.isotonic_regressor is not None

        # Scores should be different (calibration changes values)
        assert not np.allclose(scores_no_calib, scores_calib)

    def test_fit_transform(self, synthetic_regime_data):
        """Test fit_transform convenience method."""
        aggregator = RegimeProbabilityAggregator(
            method=AggregationMethod.EXPECTED_RETURN
        )

        scores = aggregator.fit_transform(
            synthetic_regime_data['regime_probs'],
            synthetic_regime_data['forward_returns'],
            synthetic_regime_data['regime_stats']
        )

        assert aggregator.is_fitted
        assert len(scores) == len(synthetic_regime_data['forward_returns'])
        assert np.all((scores >= 0) & (scores <= 1))

    def test_evaluate(self, synthetic_regime_data):
        """Test evaluation metrics."""
        aggregator = RegimeProbabilityAggregator(
            method=AggregationMethod.EXPECTED_RETURN
        )

        aggregator.fit(
            synthetic_regime_data['regime_probs'],
            synthetic_regime_data['forward_returns'],
            synthetic_regime_data['regime_stats']
        )

        metrics = aggregator.evaluate(
            synthetic_regime_data['regime_probs'],
            synthetic_regime_data['forward_returns']
        )

        # Check all metrics exist
        assert 'roc_auc' in metrics
        assert 'mse' in metrics
        assert 'correlation' in metrics
        assert 'spearman' in metrics

        # Check reasonable values
        assert 0.0 <= metrics['roc_auc'] <= 1.0
        assert metrics['mse'] >= 0.0
        assert -1.0 <= metrics['correlation'] <= 1.0
        assert -1.0 <= metrics['spearman'] <= 1.0

    def test_save_load(self, synthetic_regime_data):
        """Test saving and loading aggregator."""
        aggregator = RegimeProbabilityAggregator(
            method=AggregationMethod.EXPECTED_RETURN
        )

        aggregator.fit(
            synthetic_regime_data['regime_probs'],
            synthetic_regime_data['forward_returns'],
            synthetic_regime_data['regime_stats']
        )

        # Transform before save
        scores_before = aggregator.transform(synthetic_regime_data['regime_probs'])

        # Save
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            aggregator.save(temp_path)

            # Load
            loaded_aggregator = RegimeProbabilityAggregator.load(temp_path)

            # Transform after load
            scores_after = loaded_aggregator.transform(
                synthetic_regime_data['regime_probs']
            )

            # Scores should be identical
            np.testing.assert_array_almost_equal(scores_before, scores_after)

            # Metadata should match
            assert loaded_aggregator.method == aggregator.method
            assert loaded_aggregator.n_regimes == aggregator.n_regimes
            assert loaded_aggregator.is_fitted

        finally:
            Path(temp_path).unlink()

    def test_transform_without_fit_raises(self, synthetic_regime_data):
        """Test that transform without fit raises error."""
        aggregator = RegimeProbabilityAggregator(
            method=AggregationMethod.EXPECTED_RETURN
        )

        with pytest.raises(RuntimeError, match="not fitted"):
            aggregator.transform(synthetic_regime_data['regime_probs'])

    def test_save_without_fit_raises(self, synthetic_regime_data):
        """Test that save without fit raises error."""
        aggregator = RegimeProbabilityAggregator(
            method=AggregationMethod.EXPECTED_RETURN
        )

        with tempfile.NamedTemporaryFile(suffix='.pkl') as f:
            with pytest.raises(RuntimeError, match="Cannot save unfitted"):
                aggregator.save(f.name)

    def test_wrong_number_of_regimes_raises(self, synthetic_regime_data):
        """Test that wrong number of regimes raises error."""
        aggregator = RegimeProbabilityAggregator(
            method=AggregationMethod.EXPECTED_RETURN
        )

        aggregator.fit(
            synthetic_regime_data['regime_probs'],
            synthetic_regime_data['forward_returns'],
            synthetic_regime_data['regime_stats']
        )

        # Try to transform with wrong number of regimes
        wrong_probs = np.random.rand(100, 5)  # 5 regimes instead of 3

        with pytest.raises(ValueError, match="Expected"):
            aggregator.transform(wrong_probs)

    def test_regime_stats_computed_when_not_provided(self, synthetic_regime_data):
        """Test that regime stats are computed when not provided."""
        aggregator = RegimeProbabilityAggregator(
            method=AggregationMethod.EXPECTED_RETURN
        )

        # Fit without providing regime_stats
        aggregator.fit(
            synthetic_regime_data['regime_probs'],
            synthetic_regime_data['forward_returns'],
            regime_stats=None  # Will be computed
        )

        assert aggregator.is_fitted
        assert aggregator.regime_stats is not None
        assert len(aggregator.regime_stats) == synthetic_regime_data['n_regimes']


class TestCompareAggregationMethods:
    """Tests for method comparison functionality."""

    def test_compare_all_methods(self, synthetic_regime_data):
        """Test comparison of all aggregation methods."""
        comparison_df = compare_aggregation_methods(
            synthetic_regime_data['regime_probs'],
            synthetic_regime_data['forward_returns'],
            synthetic_regime_data['regime_stats']
        )

        # Check all methods are present
        assert len(comparison_df) == len(AggregationMethod)

        # Check all metrics are present
        assert 'roc_auc' in comparison_df.columns
        assert 'mse' in comparison_df.columns
        assert 'correlation' in comparison_df.columns
        assert 'spearman' in comparison_df.columns

        # Check sorted by ROC-AUC descending
        assert comparison_df['roc_auc'].is_monotonic_decreasing

    def test_compare_specific_methods(self, synthetic_regime_data):
        """Test comparison of specific methods."""
        methods_to_compare = [
            AggregationMethod.EXPECTED_RETURN,
            AggregationMethod.EXPECTED_SHARPE
        ]

        comparison_df = compare_aggregation_methods(
            synthetic_regime_data['regime_probs'],
            synthetic_regime_data['forward_returns'],
            synthetic_regime_data['regime_stats'],
            methods=methods_to_compare
        )

        assert len(comparison_df) == 2
        assert 'expected_return' in comparison_df.index
        assert 'expected_sharpe' in comparison_df.index


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_regime(self):
        """Test handling of single regime."""
        regime_probs = np.ones((100, 1))  # All probability on one regime
        forward_returns = np.random.randn(100) * 0.01

        aggregator = RegimeProbabilityAggregator(
            method=AggregationMethod.EXPECTED_RETURN
        )

        # Should handle single regime
        aggregator.fit(regime_probs, forward_returns)
        scores = aggregator.transform(regime_probs)

        assert len(scores) == 100
        # All scores should be similar (only one regime)
        assert np.std(scores) < 0.1

    def test_nans_in_data(self):
        """Test handling of NaN values."""
        regime_probs = np.random.rand(100, 3)
        regime_probs = regime_probs / regime_probs.sum(axis=1, keepdims=True)

        forward_returns = np.random.randn(100) * 0.01

        # Add some NaNs
        regime_probs[10:20] = np.nan
        forward_returns[30:40] = np.nan

        aggregator = RegimeProbabilityAggregator(
            method=AggregationMethod.EXPECTED_RETURN
        )

        # Should handle NaNs by removing them
        aggregator.fit(regime_probs, forward_returns)

        # Transform should work on valid data
        valid_probs = regime_probs[~np.isnan(regime_probs).any(axis=1)]
        scores = aggregator.transform(valid_probs)

        assert len(scores) > 0
        assert np.all(np.isfinite(scores))

    def test_perfect_correlation(self):
        """Test with perfect correlation between probabilities and returns."""
        np.random.seed(42)
        n_samples = 100

        # Create perfectly correlated data
        regime_probs = np.zeros((n_samples, 2))
        forward_returns = np.zeros(n_samples)

        for i in range(n_samples):
            if i % 2 == 0:
                regime_probs[i] = [0.9, 0.1]
                forward_returns[i] = 0.01  # Positive return
            else:
                regime_probs[i] = [0.1, 0.9]
                forward_returns[i] = -0.01  # Negative return

        aggregator = RegimeProbabilityAggregator(
            method=AggregationMethod.LOGISTIC_REGRESSION
        )

        aggregator.fit(regime_probs, forward_returns)
        metrics = aggregator.evaluate(regime_probs, forward_returns)

        # Should have perfect prediction
        assert metrics['roc_auc'] > 0.95
        assert abs(metrics['correlation']) > 0.8


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_full_workflow(self, synthetic_regime_data):
        """Test complete workflow: fit -> transform -> evaluate -> save -> load."""
        # 1. Fit
        aggregator = RegimeProbabilityAggregator(
            method=AggregationMethod.EXPECTED_RETURN,
            isotonic_calibration=True
        )

        aggregator.fit(
            synthetic_regime_data['regime_probs'],
            synthetic_regime_data['forward_returns'],
            synthetic_regime_data['regime_stats']
        )

        # 2. Transform
        scores = aggregator.transform(synthetic_regime_data['regime_probs'])

        # 3. Evaluate
        metrics = aggregator.evaluate(
            synthetic_regime_data['regime_probs'],
            synthetic_regime_data['forward_returns']
        )

        assert metrics['roc_auc'] > 0.5  # Better than random

        # 4. Save
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            aggregator.save(temp_path)

            # 5. Load
            loaded_aggregator = RegimeProbabilityAggregator.load(temp_path)

            # 6. Verify
            loaded_scores = loaded_aggregator.transform(
                synthetic_regime_data['regime_probs']
            )

            np.testing.assert_array_almost_equal(scores, loaded_scores)

        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
