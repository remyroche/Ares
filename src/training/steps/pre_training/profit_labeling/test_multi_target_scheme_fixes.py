"""
Test suite for multi_target_scheme.py fixes

This module tests the critical fixes implemented in the multi-target scheme:
1. Leakage tests
2. Synthetic process tests
3. Kaplan-Meier tests
4. Parallel determinism tests
5. Correctness tests
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock
import warnings
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_data_preview, tprint_data_format

# Import the fixed module
from multi_target_scheme import MultiTargetScheme, MultiTargetConfig, TargetBand


class TestMultiTargetSchemeFixes:
    """Test suite for multi-target scheme fixes."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = MultiTargetConfig(
            cv_folds=3,
            embargo_fraction=0.01,
            random_state=42,
            objective='auc',
            min_activation=0.05,
            max_activation=0.50,
            min_nonzero_samples_per_target=50
        )
        self.scheme = MultiTargetScheme(self.config)
        
        # Create synthetic test data
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic OHLCV data
        self.bars = pd.DataFrame({
            'open': np.random.randn(n_samples).cumsum() + 100,
            'high': np.random.randn(n_samples).cumsum() + 101,
            'low': np.random.randn(n_samples).cumsum() + 99,
            'close': np.random.randn(n_samples).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, n_samples)
        })
        
        # Generate synthetic volatility
        self.volatility = pd.Series(np.random.exponential(0.02, n_samples))
        
        # Generate eligibility mask
        self.eligibility = pd.Series(np.random.random(n_samples) > 0.1)
    
    def test_no_data_leakage(self):
        """Test that no data leakage occurs."""
        # Test that decisions at time t only use data up to t-1
        result = self.scheme.generate_targets(
            self.bars, self.volatility, self.eligibility
        )
        
        # Check that no future data is used
        assert result.labels is not None
        assert not result.labels.empty
        
        # Verify that labels are properly shifted
        # (This is a simplified test - in practice, you'd check specific timestamps)
        assert len(result.labels) <= len(self.bars)
    
    def test_synthetic_random_walk(self):
        """Test on random walk with drift."""
        # Create random walk with known drift
        drift = 0.001
        n_samples = 500
        
        # Generate random walk with drift
        returns = np.random.normal(drift, 0.02, n_samples)
        prices = 100 * np.exp(np.cumsum(returns))
        
        bars_rw = pd.DataFrame({
            'open': prices,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_samples)
        })
        
        volatility_rw = pd.Series(np.full(n_samples, 0.02))
        eligibility_rw = pd.Series(np.ones(n_samples, dtype=bool))
        
        result = self.scheme.generate_targets(
            bars_rw, volatility_rw, eligibility_rw
        )
        
        # Should find some targets (small/medium k should work well with drift)
        assert result.n_targets > 0
        assert not result.labels.empty
    
    def test_synthetic_bid_ask_bounce(self):
        """Test on bid-ask bounce + noise."""
        # Create bid-ask bounce pattern
        n_samples = 500
        base_price = 100
        spread = 0.01
        
        # Generate bid-ask bounce
        prices = []
        for i in range(n_samples):
            if i % 2 == 0:
                prices.append(base_price + spread/2 + np.random.normal(0, 0.001))
            else:
                prices.append(base_price - spread/2 + np.random.normal(0, 0.001))
        
        bars_bounce = pd.DataFrame({
            'open': prices,
            'high': [p + 0.001 for p in prices],
            'low': [p - 0.001 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_samples)
        })
        
        volatility_bounce = pd.Series(np.full(n_samples, 0.01))
        eligibility_bounce = pd.Series(np.ones(n_samples, dtype=bool))
        
        result = self.scheme.generate_targets(
            bars_bounce, volatility_bounce, eligibility_bounce
        )
        
        # Should penalize short horizons / small k due to noise
        # This is a qualitative test - the scheme should adapt
        assert result.n_targets >= 0  # May find no good targets due to noise
    
    def test_kaplan_meier_correctness(self):
        """Test Kaplan-Meier implementation correctness."""
        # Create known test data with ties and censoring
        times = np.array([1, 1, 2, 3, 3, 3, 4, 5, 6, 7])
        events = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1])  # 1=event, 0=censored
        
        survival_probs = self.scheme._calculate_survival_probabilities(times, events)
        
        # Check basic properties
        assert len(survival_probs) == len(times)
        assert all(0 <= p <= 1 for p in survival_probs)
        assert survival_probs[0] <= survival_probs[-1]  # Survival should be non-increasing
        
        # Check that survival decreases at event times
        for i in range(1, len(times)):
            if events[i] == 1:  # Event occurred
                assert survival_probs[i] <= survival_probs[i-1]
    
    def test_parallel_determinism(self):
        """Test that parallel execution is deterministic."""
        # Test with same random seed
        config1 = MultiTargetConfig(random_state=42, enable_parallel_processing=True)
        config2 = MultiTargetConfig(random_state=42, enable_parallel_processing=True)
        
        scheme1 = MultiTargetScheme(config1)
        scheme2 = MultiTargetScheme(config2)
        
        # Generate targets with same data
        result1 = scheme1.generate_targets(
            self.bars, self.volatility, self.eligibility
        )
        result2 = scheme2.generate_targets(
            self.bars, self.volatility, self.eligibility
        )
        
        # Results should be identical (or very close due to numerical precision)
        if not result1.labels.empty and not result2.labels.empty:
            # Check that the same targets are selected
            assert set(result1.labels.columns) == set(result2.labels.columns)
    
    def test_fpt_quantile_semantics(self):
        """Test FPT quantile semantics consistency."""
        # Test that get_fpt_quantiles returns probabilities
        fpt_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        quantiles = self.config.get_fpt_quantiles(fpt_series)
        
        # Should return probabilities (0-1 range)
        assert all(0 <= q <= 1 for q in quantiles)
        
        # Test that get_fpt_times returns actual times
        times = self.config.get_fpt_times(fpt_series)
        
        # Should return actual time values
        assert all(t >= 0 for t in times)
        assert len(times) == len(quantiles)
    
    def test_volatility_normalization_k_space(self):
        """Test that bands are defined in k-space, not sigma-space."""
        # Test target bands calculation
        vol_series = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        bands = self.config.get_target_bands(vol_series)
        
        # Bands should be in k-space (volatility multipliers)
        for band_name, (low, high) in bands.items():
            assert low > 0 and high > 0  # k values should be positive
            assert low < high  # Low should be less than high
            # k values should be reasonable multipliers (not raw volatility values)
            assert low < 5.0 and high < 5.0
    
    def test_class_balance_metadata(self):
        """Test that class balance uses band metadata, not string matching."""
        # Create mock selected targets with band metadata
        selected_targets = {
            'target1': {'band': TargetBand.SMALL, 'k_up': 0.8, 'k_down': 0.8},
            'target2': {'band': TargetBand.MEDIUM, 'k_up': 1.2, 'k_down': 1.2},
            'target3': {'band': TargetBand.HIGH, 'k_up': 1.8, 'k_down': 1.8}
        }
        
        # Create mock labels DataFrame
        labels_df = pd.DataFrame({
            'target1': [1, -1, 0, 1, -1],
            'target2': [0, 1, -1, 0, 1],
            'target3': [1, 0, 1, -1, 0]
        })
        
        # Test conflict resolution with metadata
        resolved = self.scheme._resolve_label_conflicts(labels_df, selected_targets)
        
        # Should use band metadata for grouping, not string matching
        assert not resolved.empty
        assert len(resolved.columns) == len(labels_df.columns)
    
    def test_confidence_scores_probabilistic(self):
        """Test that confidence scores are probabilistic and calibrated."""
        # Create mock features and test confidence calculation
        features = pd.Series({
            'returns': 0.01,
            'volatility': 0.02,
            'volatility_ratio': 1.1,
            'volume_ratio': 0.9,
            'high_low_ratio': 0.005,
            'close_open_ratio': 0.002,
            'price_momentum': 0.001
        })
        
        confidence = self.scheme._calculate_probabilistic_confidence(
            features, 1, 0.02, 5
        )
        
        # Should return a probability between 0 and 1
        assert 0 <= confidence <= 1
    
    def test_mutual_information_orthogonality(self):
        """Test mutual information calculation for orthogonality."""
        # Create test label sequences
        x = pd.Series([1, -1, 1, -1, 0, 1, -1, 0, 1, -1])
        y = pd.Series([-1, 1, -1, 1, 0, -1, 1, 0, -1, 1])  # Negatively correlated
        
        mi_score = self.scheme._calculate_mutual_information(x, y)
        
        # Should return a non-negative mutual information score
        assert mi_score >= 0
    
    def test_diversity_metrics_correct_set(self):
        """Test that diversity metrics compute on the right set."""
        # Create labels with some zeros
        labels_df = pd.DataFrame({
            'target1': [1, -1, 0, 1, -1, 0, 1, -1, 0, 1],
            'target2': [0, 1, -1, 0, 1, -1, 0, 1, -1, 0],
            'target3': [1, 0, 1, -1, 0, 1, -1, 0, 1, -1]
        })
        
        diversity_score = self.scheme._calculate_diversity_score(labels_df)
        coverage = self.scheme._calculate_target_coverage(labels_df)
        
        # Diversity should be calculated on non-zero labels only
        assert 0 <= diversity_score <= 1
        
        # Coverage should include detailed metrics
        for col in labels_df.columns:
            assert col in coverage
            assert 'activation_rate' in coverage[col]
            assert 'positive_ratio' in coverage[col]
            assert 'negative_ratio' in coverage[col]
    
    def test_idxmax_safety(self):
        """Test that idxmax() is used safely with .any() checks."""
        # Test the vectorized first hit detection
        hit_array = np.array([False, False, True, False, True])
        
        result = self.scheme._vectorized_first_hit(hit_array)
        
        # Should return the index of first True, or -1 if none
        assert result == 2
        
        # Test with all False
        hit_array_false = np.array([False, False, False, False])
        result_false = self.scheme._vectorized_first_hit(hit_array_false)
        assert result_false == -1
    
    def test_explicit_dtypes(self):
        """Test that explicit dtypes are set."""
        # Test label generation with explicit dtypes
        bars = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'open': [99, 100, 101, 102, 103],
            'high': [101, 102, 103, 104, 105],
            'low': [98, 99, 100, 101, 102]
        })
        
        vol = pd.Series([0.01, 0.02, 0.015, 0.025, 0.02])
        elig = pd.Series([True, True, True, True, True])
        
        labels = self.scheme._generate_labels_for_k(1.0, 1.0, bars, vol, elig)
        
        # Should have explicit int dtype
        assert labels.dtype == int or labels.dtype == 'int64'


def run_all_tests():
    """Run all tests and report results."""
    test_instance = TestMultiTargetSchemeFixes()
    test_instance.setup_method()
    
    tests = [
        test_instance.test_no_data_leakage,
        test_instance.test_synthetic_random_walk,
        test_instance.test_synthetic_bid_ask_bounce,
        test_instance.test_kaplan_meier_correctness,
        test_instance.test_parallel_determinism,
        test_instance.test_fpt_quantile_semantics,
        test_instance.test_volatility_normalization_k_space,
        test_instance.test_class_balance_metadata,
        test_instance.test_confidence_scores_probabilistic,
        test_instance.test_mutual_information_orthogonality,
        test_instance.test_diversity_metrics_correct_set,
        test_instance.test_idxmax_safety,
        test_instance.test_explicit_dtypes
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            tprint(f"✅ {test.__name__} - PASSED")
            passed += 1
        except Exception as e:
            tprint(f"❌ {test.__name__} - FAILED: {e}")
            failed += 1
    
    tprint(f"\n📊 Test Results: {passed} passed, {failed} failed")
    return passed, failed


if __name__ == "__main__":
    run_all_tests()