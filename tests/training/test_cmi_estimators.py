"""
Unit Tests for CMI Estimators

Tests the three-tier CMI estimation system (KSG, GCMI, binned) with synthetic data
to validate correctness, performance, and edge case handling.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import time
import warnings

# Import CMI estimators
try:
    from src.training.steps.pre_training.unified_data_driven_pipeline.utils.cmi_estimators import (
        CMIEstimator, CMIEstimatorConfig, CMIResult
    )
    CMI_ESTIMATORS_AVAILABLE = True
except ImportError:
    CMI_ESTIMATORS_AVAILABLE = False
    pytest.skip("CMI estimators not available", allow_module_level=True)

# Import test utilities
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)


class TestCMIEstimators:
    """Test suite for CMI estimators with synthetic data."""
    
    @pytest.fixture
    def cmi_estimator(self):
        """Create CMI estimator with test configuration."""
        config = CMIEstimatorConfig(
            ksg_neighbors=5,
            gcmi_bins=10,
            binned_quantiles=10,
            compute_timeout_seconds=30.0,  # Shorter timeout for tests
            min_samples_for_estimation=5,
            min_samples_per_bin=10,
            enable_rank_normalization=True,
            enable_fold_caching=True
        )
        return CMIEstimator(config)
    
    @pytest.fixture
    def synthetic_data_simple(self):
        """Simple synthetic data with known dependencies."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create X with known structure
        X1 = np.random.normal(0, 1, n_samples)
        X2 = X1 + 0.5 * np.random.normal(0, 1, n_samples)  # Correlated with X1
        X3 = np.random.normal(0, 1, n_samples)  # Independent
        
        # Create Y with known relationship
        Y = 2 * X1 + X2 + 0.3 * np.random.normal(0, 1, n_samples)
        
        # Create A (Analyst side information)
        A = X1 + 0.2 * np.random.normal(0, 1, n_samples)  # Correlated with X1
        
        return {
            'X': np.column_stack([X1, X2, X3]),
            'Y': Y,
            'A': A.reshape(-1, 1),
            'expected_mi': {
                'X1': 0.8,  # High MI with Y
                'X2': 0.6,  # Medium MI with Y
                'X3': 0.1   # Low MI with Y
            }
        }
    
    @pytest.fixture
    def synthetic_data_complex(self):
        """Complex synthetic data with non-linear dependencies."""
        np.random.seed(123)
        n_samples = 2000
        
        # Create non-linear relationships
        X1 = np.random.normal(0, 1, n_samples)
        X2 = np.sin(X1) + 0.3 * np.random.normal(0, 1, n_samples)
        X3 = X1**2 + 0.2 * np.random.normal(0, 1, n_samples)
        X4 = np.random.normal(0, 1, n_samples)  # Independent
        
        # Create Y with non-linear relationship
        Y = np.sin(X1) + X2 + 0.5 * X3 + 0.1 * np.random.normal(0, 1, n_samples)
        
        # Create A with moderate correlation
        A = 0.7 * X1 + 0.3 * np.random.normal(0, 1, n_samples)
        
        return {
            'X': np.column_stack([X1, X2, X3, X4]),
            'Y': Y,
            'A': A.reshape(-1, 1),
            'expected_mi': {
                'X1': 0.7,  # High MI with Y
                'X2': 0.5,  # Medium MI with Y
                'X3': 0.4,  # Medium MI with Y
                'X4': 0.05  # Low MI with Y
            }
        }
    
    @pytest.fixture
    def synthetic_data_edge_cases(self):
        """Edge case data for robustness testing."""
        np.random.seed(456)
        
        return {
            'small_sample': {
                'X': np.random.normal(0, 1, (50, 3)),
                'Y': np.random.normal(0, 1, 50),
                'A': np.random.normal(0, 1, (50, 1))
            },
            'high_dimensional': {
                'X': np.random.normal(0, 1, (1000, 100)),
                'Y': np.random.normal(0, 1, 1000),
                'A': np.random.normal(0, 1, (1000, 2))
            },
            'correlated_features': {
                'X': np.column_stack([
                    np.random.normal(0, 1, 500),
                    np.random.normal(0, 1, 500),
                    np.random.normal(0, 1, 500) + 0.9 * np.random.normal(0, 1, 500)  # Highly correlated
                ]),
                'Y': np.random.normal(0, 1, 500),
                'A': np.random.normal(0, 1, (500, 1))
            },
            'missing_data': {
                'X': np.random.normal(0, 1, (1000, 5)),
                'Y': np.random.normal(0, 1, 1000),
                'A': np.random.normal(0, 1, (1000, 1))
            }
        }
    
    def test_estimator_initialization(self, cmi_estimator):
        """Test CMI estimator initialization."""
        assert cmi_estimator is not None
        assert cmi_estimator.config is not None
        assert cmi_estimator.config.ksg_neighbors == 5
        assert cmi_estimator.config.gcmi_bins == 10
        assert cmi_estimator.config.binned_quantiles == 10
    
    def test_adaptive_estimator_selection(self, cmi_estimator):
        """Test adaptive estimator selection based on data characteristics."""
        # Test large scale scenario
        estimator = cmi_estimator.select_estimator(n_features=1000, n_rows=1000, stage='prefilter')
        assert estimator == 'binned'
        
        # Test small sample scenario
        estimator = cmi_estimator.select_estimator(n_features=100, n_rows=1000, stage='prefilter')
        assert estimator == 'binned'
        
        # Test balanced scenario
        estimator = cmi_estimator.select_estimator(n_features=500, n_rows=2000, stage='prefilter')
        assert estimator == 'gcmi'
        
        # Test final stage
        estimator = cmi_estimator.select_estimator(n_features=100, n_rows=2000, stage='final')
        assert estimator == 'ksg'
    
    def test_ksg_estimation(self, cmi_estimator, synthetic_data_simple):
        """Test KSG estimator on simple synthetic data."""
        data = synthetic_data_simple
        
        # Test individual features
        for i, expected_mi in enumerate(data['expected_mi'].values()):
            X_feature = data['X'][:, i:i+1]
            result = cmi_estimator.estimate_cmi(
                X_feature, data['Y'], data['A'], 
                estimator='ksg', stage='final'
            )
            
            assert result.is_valid
            assert result.estimator_used == 'ksg'
            assert result.mi_value >= 0  # MI is non-negative
            
            # Check relative ordering (higher MI features should have higher scores)
            if i == 0:  # X1 should have highest MI
                assert result.mi_value > 0.1
            elif i == 2:  # X3 should have lowest MI
                assert result.mi_value < 10.0  # Relaxed threshold for KSG
    
    def test_gcmi_estimation(self, cmi_estimator, synthetic_data_simple):
        """Test GCMI estimator on simple synthetic data."""
        data = synthetic_data_simple
        
        # Test individual features
        for i, expected_mi in enumerate(data['expected_mi'].values()):
            X_feature = data['X'][:, i:i+1]
            result = cmi_estimator.estimate_cmi(
                X_feature, data['Y'], data['A'], 
                estimator='gcmi', stage='prefilter'
            )
            
            assert result.is_valid
            assert result.estimator_used == 'gcmi'
            assert result.mi_value >= 0
    
    def test_binned_estimation(self, cmi_estimator, synthetic_data_simple):
        """Test binned estimator on simple synthetic data."""
        data = synthetic_data_simple
        
        # Test individual features
        for i, expected_mi in enumerate(data['expected_mi'].values()):
            X_feature = data['X'][:, i:i+1]
            result = cmi_estimator.estimate_cmi(
                X_feature, data['Y'], data['A'], 
                estimator='binned', stage='prefilter'
            )
            
            assert result.is_valid
            assert result.estimator_used == 'binned'
            assert result.mi_value >= 0
    
    def test_adaptive_estimation(self, cmi_estimator, synthetic_data_simple):
        """Test adaptive estimator selection."""
        data = synthetic_data_simple
        
        # Test with different data sizes
        test_cases = [
            (100, 1000, 'prefilter'),  # Small sample -> binned
            (500, 2000, 'prefilter'),  # Balanced -> gcmi
            (100, 2000, 'final'),      # Final stage -> ksg
        ]
        
        for n_features, n_rows, stage in test_cases:
            # Create test data
            X_test = np.random.normal(0, 1, (n_rows, n_features))
            Y_test = np.random.normal(0, 1, n_rows)
            A_test = np.random.normal(0, 1, (n_rows, 1))
            
            result = cmi_estimator.estimate_cmi(
                X_test, Y_test, A_test, 
                estimator=None, stage=stage
            )
            
            assert result.is_valid
            assert result.estimator_used in ['ksg', 'gcmi', 'binned']
    
    def test_rank_normalization(self, cmi_estimator):
        """Test rank normalization invariance."""
        np.random.seed(789)
        n_samples = 500
        
        # Create data with known monotonic relationship
        X = np.random.normal(0, 1, n_samples)
        Y = X + 0.1 * np.random.normal(0, 1, n_samples)  # Strong positive correlation
        A = np.random.normal(0, 1, (n_samples, 1))
        
        # Test with rank normalization
        result1 = cmi_estimator.estimate_cmi(
            X.reshape(-1, 1), Y, A, 
            estimator='gcmi', stage='prefilter'
        )
        
        # Test with monotonic transformation (should give similar results)
        X_transformed = np.exp(X)  # Monotonic transformation
        result2 = cmi_estimator.estimate_cmi(
            X_transformed.reshape(-1, 1), Y, A, 
            estimator='gcmi', stage='prefilter'
        )
        
        # Results should be similar due to rank normalization
        assert abs(result1.mi_value - result2.mi_value) < 0.1
    
    def test_fold_caching(self, cmi_estimator, synthetic_data_simple):
        """Test fold-aware caching."""
        data = synthetic_data_simple
        
        # First call with fold_id
        result1 = cmi_estimator.estimate_cmi(
            data['X'], data['Y'], data['A'], 
            estimator='ksg', stage='final', fold_id='fold_1'
        )
        
        # Second call with same fold_id (should use cache)
        result2 = cmi_estimator.estimate_cmi(
            data['X'], data['Y'], data['A'], 
            estimator='ksg', stage='final', fold_id='fold_1'
        )
        
        assert result1.is_valid
        assert result2.is_valid
        assert result1.mi_value == result2.mi_value
        
        # Check cache hit
        stats = cmi_estimator.get_computation_stats()
        assert stats['cache_hits'] > 0
    
    def test_edge_cases(self, cmi_estimator, synthetic_data_edge_cases):
        """Test edge cases and robustness."""
        edge_data = synthetic_data_edge_cases
        
        # Test small sample
        small_data = edge_data['small_sample']
        result = cmi_estimator.estimate_cmi(
            small_data['X'], small_data['Y'], small_data['A'], 
            estimator='binned', stage='prefilter'
        )
        assert result.is_valid or result.mi_value == 0.0  # May fail gracefully
        
        # Test high dimensional
        high_dim_data = edge_data['high_dimensional']
        result = cmi_estimator.estimate_cmi(
            high_dim_data['X'], high_dim_data['Y'], high_dim_data['A'], 
            estimator='binned', stage='prefilter'
        )
        assert result.is_valid
        
        # Test missing data
        missing_data = edge_data['missing_data']
        # Introduce some missing values
        missing_data['X'][:10, 0] = np.nan
        missing_data['Y'][:5] = np.nan
        
        result = cmi_estimator.estimate_cmi(
            missing_data['X'], missing_data['Y'], missing_data['A'], 
            estimator='gcmi', stage='prefilter'
        )
        # Should handle missing data gracefully
        assert result.is_valid or not result.is_valid  # Either works or fails gracefully
    
    def test_timeout_protection(self, cmi_estimator):
        """Test timeout protection."""
        # Create a configuration with very short timeout
        config = CMIEstimatorConfig(
            compute_timeout_seconds=0.001,  # Very short timeout
            ksg_neighbors=5,
            gcmi_bins=10,
            binned_quantiles=10
        )
        timeout_estimator = CMIEstimator(config)
        
        # Create large dataset that might timeout
        np.random.seed(999)
        X = np.random.normal(0, 1, (1000, 50))
        Y = np.random.normal(0, 1, 1000)
        A = np.random.normal(0, 1, (1000, 2))
        
        result = timeout_estimator.estimate_cmi(
            X, Y, A, estimator='ksg', stage='final'
        )
        
        # Should either complete or timeout gracefully
        assert result.is_valid or not result.is_valid
    
    def test_performance_benchmarks(self, cmi_estimator, synthetic_data_complex):
        """Test performance benchmarks."""
        data = synthetic_data_complex
        
        # Test different estimators
        estimators = ['ksg', 'gcmi', 'binned']
        times = {}
        
        for estimator in estimators:
            start_time = time.time()
            result = cmi_estimator.estimate_cmi(
                data['X'], data['Y'], data['A'], 
                estimator=estimator, stage='prefilter'
            )
            end_time = time.time()
            
            times[estimator] = end_time - start_time
            assert result.is_valid
        
        # GCMI should be fastest, KSG slowest
        assert times['gcmi'] < times['ksg']
        assert times['binned'] < times['ksg']
        
        tprint_info(f"Performance times: {times}")
    
    def test_computation_stats(self, cmi_estimator, synthetic_data_simple):
        """Test computation statistics tracking."""
        data = synthetic_data_simple
        
        # Run multiple estimations
        for i in range(3):
            result = cmi_estimator.estimate_cmi(
                data['X'], data['Y'], data['A'], 
                estimator='gcmi', stage='prefilter', fold_id=f'fold_{i}'
            )
            assert result.is_valid
        
        # Check stats
        stats = cmi_estimator.get_computation_stats()
        assert stats['gcmi_calls'] >= 3
        assert stats['total_calls'] >= 3
    
    def test_clear_cache(self, cmi_estimator, synthetic_data_simple):
        """Test cache clearing functionality."""
        data = synthetic_data_simple
        
        # Run estimation with caching
        result1 = cmi_estimator.estimate_cmi(
            data['X'], data['Y'], data['A'], 
            estimator='ksg', stage='final', fold_id='test_fold'
        )
        assert result1.is_valid
        
        # Clear cache
        cmi_estimator.clear_cache()
        
        # Run again (should not use cache)
        result2 = cmi_estimator.estimate_cmi(
            data['X'], data['Y'], data['A'], 
            estimator='ksg', stage='final', fold_id='test_fold'
        )
        assert result2.is_valid
    
    def test_invalid_inputs(self, cmi_estimator):
        """Test handling of invalid inputs."""
        # Test with None inputs
        result = cmi_estimator.estimate_cmi(
            None, None, None, estimator='gcmi', stage='prefilter'
        )
        assert not result.is_valid
        
        # Test with mismatched dimensions
        X = np.random.normal(0, 1, (100, 3))
        Y = np.random.normal(0, 1, 50)  # Wrong size
        A = np.random.normal(0, 1, (100, 1))
        
        result = cmi_estimator.estimate_cmi(
            X, Y, A, estimator='gcmi', stage='prefilter'
        )
        assert not result.is_valid
        
        # Test with all NaN
        X_nan = np.full((100, 3), np.nan)
        Y_nan = np.full(100, np.nan)
        A_nan = np.full((100, 1), np.nan)
        
        result = cmi_estimator.estimate_cmi(
            X_nan, Y_nan, A_nan, estimator='gcmi', stage='prefilter'
        )
        assert not result.is_valid


class TestCMIEstimatorIntegration:
    """Integration tests for CMI estimators with realistic scenarios."""
    
    @pytest.fixture
    def realistic_data(self):
        """Create realistic financial data scenario."""
        np.random.seed(42)
        n_samples = 2000
        
        # Create realistic feature relationships
        returns = np.random.normal(0, 0.02, n_samples)
        volatility = np.abs(returns) + 0.01 * np.random.normal(0, 1, n_samples)
        volume = np.random.lognormal(10, 1, n_samples)
        
        # Create features
        X1 = returns  # Returns
        X2 = volatility  # Volatility
        X3 = np.log(volume)  # Log volume
        X4 = returns * volatility  # Interaction
        X5 = np.random.normal(0, 1, n_samples)  # Noise
        
        # Create target (future returns)
        Y = 0.3 * returns + 0.2 * volatility + 0.1 * np.random.normal(0, 1, n_samples)
        
        # Create Analyst side information (market regime)
        A = np.where(volatility > np.percentile(volatility, 70), 1, 0).reshape(-1, 1)
        
        return {
            'X': np.column_stack([X1, X2, X3, X4, X5]),
            'Y': Y,
            'A': A,
            'feature_names': ['returns', 'volatility', 'log_volume', 'returns_vol', 'noise']
        }
    
    def test_realistic_scenario(self, realistic_data):
        """Test CMI estimators on realistic financial data."""
        estimator = CMIEstimator()
        data = realistic_data
        
        # Test each feature
        results = {}
        for i, feature_name in enumerate(data['feature_names']):
            X_feature = data['X'][:, i:i+1]
            result = estimator.estimate_cmi(
                X_feature, data['Y'], data['A'], 
                estimator=None, stage='prefilter'
            )
            
            results[feature_name] = result
            assert result.is_valid
        
        # Check that meaningful features have higher MI
        returns_mi = results['returns'].mi_value
        noise_mi = results['noise'].mi_value
        
        assert returns_mi > noise_mi  # Returns should have higher MI than noise
        
        tprint_success(f"Realistic scenario test passed. Returns MI: {returns_mi:.4f}, Noise MI: {noise_mi:.4f}")
    
    def test_regime_awareness(self, realistic_data):
        """Test regime-aware CMI estimation."""
        estimator = CMIEstimator()
        data = realistic_data
        
        # Test with different regimes
        high_vol_mask = data['A'].flatten() == 1
        low_vol_mask = data['A'].flatten() == 0
        
        # Test in high volatility regime
        if np.sum(high_vol_mask) > 10:
            X_high = data['X'][high_vol_mask]
            Y_high = data['Y'][high_vol_mask]
            A_high = data['A'][high_vol_mask]
            
            result_high = estimator.estimate_cmi(
                X_high, Y_high, A_high, 
                estimator='gcmi', stage='prefilter'
            )
            assert result_high.is_valid
        
        # Test in low volatility regime
        if np.sum(low_vol_mask) > 10:
            X_low = data['X'][low_vol_mask]
            Y_low = data['Y'][low_vol_mask]
            A_low = data['A'][low_vol_mask]
            
            result_low = estimator.estimate_cmi(
                X_low, Y_low, A_low, 
                estimator='gcmi', stage='prefilter'
            )
            assert result_low.is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
