#!/usr/bin/env python3
"""Unit tests for feature residualization and RMI-based selection functions."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest


class TestResidualizeTarget:
    """Tests for _residualize_target function."""
    
    def test_basic_differencing(self):
        """Test that target is correctly differenced."""
        from scripts.specialist_feature_diagnostics import _residualize_target
        
        y = pd.Series([1.0, 2.0, 4.0, 7.0, 11.0])
        result = _residualize_target(y, lag=1)
        
        # Expected: [2-1, 4-2, 7-4, 11-7] = [1, 2, 3, 4]
        expected = pd.Series([1.0, 2.0, 3.0, 4.0], index=[1, 2, 3, 4])
        pd.testing.assert_series_equal(result, expected)
    
    def test_length_reduction(self):
        """Test that length is reduced by lag amount."""
        from scripts.specialist_feature_diagnostics import _residualize_target
        
        y = pd.Series(np.random.randn(100))
        result = _residualize_target(y, lag=1)
        
        assert len(result) == 99
    
    def test_lag_2(self):
        """Test differencing with lag=2."""
        from scripts.specialist_feature_diagnostics import _residualize_target
        
        y = pd.Series([1.0, 2.0, 4.0, 7.0, 11.0])
        result = _residualize_target(y, lag=2)
        
        # Expected: [4-1, 7-2, 11-4] = [3, 5, 7]
        expected = pd.Series([3.0, 5.0, 7.0], index=[2, 3, 4])
        pd.testing.assert_series_equal(result, expected)


class TestResidualizeFeature:
    """Tests for _residualize_feature function."""
    
    def test_residuals_zero_mean(self):
        """Test that AR(1) residuals have approximately zero mean."""
        from scripts.specialist_feature_diagnostics import _residualize_feature
        
        np.random.seed(42)
        # Create AR(1) process: x_t = 0.8 * x_{t-1} + noise
        n = 500
        x = np.zeros(n)
        x[0] = np.random.randn()
        for i in range(1, n):
            x[i] = 0.8 * x[i-1] + np.random.randn()
        
        series = pd.Series(x)
        residuals = _residualize_feature(series)
        
        # Residuals should have mean close to 0
        assert abs(residuals.dropna().mean()) < 0.5
    
    def test_residuals_uncorrelated_with_lag(self):
        """Test that residuals are uncorrelated with lagged values."""
        from scripts.specialist_feature_diagnostics import _residualize_feature
        
        np.random.seed(42)
        n = 500
        x = np.cumsum(np.random.randn(n))  # Random walk
        
        series = pd.Series(x)
        residuals = _residualize_feature(series)
        
        # Correlation between residuals and lagged original should be lower
        valid = ~residuals.isna()
        if valid.sum() > 10:
            lagged = series.shift(1)
            corr = residuals[valid].corr(lagged[valid])
            assert abs(corr) < 0.3  # Much lower than original autocorrelation
    
    def test_fallback_on_insufficient_data(self):
        """Test fallback to raw when insufficient data."""
        from scripts.specialist_feature_diagnostics import _residualize_feature
        
        x = pd.Series([1.0, 2.0, 3.0])  # Only 3 points
        result = _residualize_feature(x)
        
        # Should return original when < 10 valid points
        pd.testing.assert_series_equal(result, x)


class TestComputeInnovationZscore:
    """Tests for _compute_innovation_zscore function."""
    
    def test_zscore_approximate_standard_normal(self):
        """Test that innovation z-scores have approximately unit variance."""
        from scripts.specialist_feature_diagnostics import _compute_innovation_zscore
        
        np.random.seed(42)
        x = pd.Series(np.cumsum(np.random.randn(500)))
        
        innovation = _compute_innovation_zscore(x, window=20)
        
        # Skip first few values due to warm-up
        valid = innovation.iloc[25:].dropna()
        
        # Should have variance close to 1
        assert 0.5 < valid.std() < 2.0
    
    def test_handles_constant_series(self):
        """Test that constant series doesn't cause division by zero."""
        from scripts.specialist_feature_diagnostics import _compute_innovation_zscore
        
        x = pd.Series([5.0] * 100)
        result = _compute_innovation_zscore(x, window=20)
        
        # Should not have infinities
        assert not np.any(np.isinf(result.values))
    
    def test_window_parameter(self):
        """Test that window parameter is respected."""
        from scripts.specialist_feature_diagnostics import _compute_innovation_zscore
        
        np.random.seed(42)
        x = pd.Series(np.cumsum(np.random.randn(100)))
        
        # Different windows should produce different results
        result_10 = _compute_innovation_zscore(x, window=10)
        result_30 = _compute_innovation_zscore(x, window=30)
        
        # Results should differ (not perfectly equal)
        assert not result_10.equals(result_30)


class TestDoubleResidualizeFeatures:
    """Tests for _double_residualize_features function."""
    
    def test_output_shape(self):
        """Test that output preserves feature columns."""
        from scripts.specialist_feature_diagnostics import _double_residualize_features
        
        np.random.seed(42)
        X = pd.DataFrame({
            'feat_a': np.cumsum(np.random.randn(200)),
            'feat_b': np.cumsum(np.random.randn(200)),
            'feat_c': np.random.randn(200),
        })
        y = pd.Series(np.random.randn(200))
        
        X_res, y_res, info = _double_residualize_features(X, y)
        
        assert list(X_res.columns) == ['feat_a', 'feat_b', 'feat_c']
        assert len(X_res) == len(y_res)
        assert info['samples_lost'] > 0  # At least 1 sample lost due to differencing
    
    def test_index_alignment(self):
        """Test that output indices are aligned."""
        from scripts.specialist_feature_diagnostics import _double_residualize_features
        
        np.random.seed(42)
        X = pd.DataFrame({
            'a': np.random.randn(100),
            'b': np.random.randn(100),
        })
        y = pd.Series(np.random.randn(100))
        
        X_res, y_res, _ = _double_residualize_features(X, y)
        
        pd.testing.assert_index_equal(X_res.index, y_res.index)


class TestSelectBestFeatureFromClusterRMI:
    """Tests for _select_best_feature_from_cluster_rmi function."""
    
    def test_selects_informative_feature(self):
        """Test that RMI-based selection picks the most informative feature."""
        from scripts.specialist_feature_diagnostics import _select_best_feature_from_cluster_rmi
        
        np.random.seed(42)
        n = 500
        
        # True signal
        signal = np.random.randn(n)
        
        # Feature A: correlated with signal
        feat_a = signal + 0.5 * np.random.randn(n)
        
        # Feature B: pure noise
        feat_b = np.random.randn(n)
        
        # Feature C: noise with small correlation
        feat_c = 0.1 * signal + 0.9 * np.random.randn(n)
        
        X = pd.DataFrame({
            'informative': feat_a,
            'noise': feat_b,
            'weak': feat_c,
        })
        y = pd.Series(signal)
        
        best = _select_best_feature_from_cluster_rmi(
            X, y, 
            features=['informative', 'noise', 'weak'],
            n_neighbors=3,
            n_subsamples=5
        )
        
        # Should select the informative feature
        assert best == 'informative'
    
    def test_single_feature_returns_itself(self):
        """Test that single-feature clusters return that feature."""
        from scripts.specialist_feature_diagnostics import _select_best_feature_from_cluster_rmi
        
        X = pd.DataFrame({'only_one': np.random.randn(100)})
        y = pd.Series(np.random.randn(100))
        
        result = _select_best_feature_from_cluster_rmi(X, y, ['only_one'])
        assert result == 'only_one'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
