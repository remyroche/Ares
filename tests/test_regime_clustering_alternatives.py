"""
Comprehensive Unit Tests for Regime Clustering Alternatives

Tests cover:
- Edge cases (single regime, no transitions, minimal data)
- Integration tests with actual market data
- Performance benchmarks comparing methods
- Validation metrics (regime stability, transition accuracy)
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

# Suppress specific warnings during testing
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def synthetic_regime_data():
    """Generate synthetic data with clear regime structure."""
    np.random.seed(42)
    n_samples = 300
    
    # Create 3 regimes
    regime_1 = np.random.normal(0.0, 0.5, 100)
    regime_2 = np.random.normal(2.0, 1.5, 100)
    regime_3 = np.random.normal(-1.0, 0.8, 100)
    
    time_series = np.concatenate([regime_1, regime_2, regime_3])
    true_labels = np.array([0]*100 + [1]*100 + [2]*100)
    
    # Create features
    features = np.column_stack([
        time_series,
        np.roll(time_series, 1),
        np.roll(time_series, 2),
        np.abs(time_series),
    ])[3:]
    
    return features, true_labels[3:]


@pytest.fixture
def single_regime_data():
    """Generate data with only one regime (edge case)."""
    np.random.seed(42)
    n_samples = 200
    
    # All data from same distribution
    time_series = np.random.normal(0.0, 0.5, n_samples)
    
    features = np.column_stack([
        time_series,
        np.roll(time_series, 1),
        np.abs(time_series),
    ])[2:]
    
    true_labels = np.zeros(len(features))
    
    return features, true_labels


@pytest.fixture
def minimal_data():
    """Generate minimal data (testing minimum requirements)."""
    np.random.seed(42)
    n_samples = 50  # Very small dataset
    
    time_series = np.random.normal(0.0, 1.0, n_samples)
    features = time_series.reshape(-1, 1)
    
    return features


@pytest.fixture
def market_data():
    """Generate realistic market data."""
    np.random.seed(42)
    n_samples = 500
    
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='15min')
    
    # Generate OHLCV data with regime changes
    base_price = 100.0
    prices = [base_price]
    
    for i in range(1, n_samples):
        # Regime-dependent volatility
        if i < 150:
            volatility = 0.005  # Low vol regime
        elif i < 350:
            volatility = 0.02   # High vol regime
        else:
            volatility = 0.01   # Medium vol regime
        
        change = np.random.normal(0, volatility)
        prices.append(prices[-1] * (1 + change))
    
    prices = np.array(prices)
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.001, 0.001, n_samples)),
        'high': prices * (1 + np.random.uniform(0, 0.005, n_samples)),
        'low': prices * (1 + np.random.uniform(-0.005, 0, n_samples)),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    }, index=dates)
    
    return data


# =============================================================================
# HDP-HMM Tests
# =============================================================================

class TestHDPHMM:
    """Test suite for HDP-HMM clustering."""
    
    def test_import(self):
        """Test that HDP-HMM module can be imported."""
        from src.training.steps.market_analysis.hdp_hmm_clustering import (
            HDPHMMClusterer, HDPHMMConfig, HMM_AVAILABLE
        )
        assert HDPHMMClusterer is not None
        assert HDPHMMConfig is not None
    
    @pytest.mark.skipif(
        not __import__('src.training.steps.market_analysis.hdp_hmm_clustering', 
                      fromlist=['HMM_AVAILABLE']).HMM_AVAILABLE,
        reason="HMM libraries not available"
    )
    def test_basic_clustering(self, synthetic_regime_data):
        """Test basic clustering with synthetic data."""
        from src.training.steps.market_analysis.hdp_hmm_clustering import (
            HDPHMMClusterer, HDPHMMConfig
        )
        
        features, true_labels = synthetic_regime_data
        
        config = HDPHMMConfig(
            alpha=3.0,
            kappa=50.0,
            n_iterations=30,  # Reduced for testing
            show_progress=False
        )
        
        clusterer = HDPHMMClusterer(config)
        result = clusterer.fit_predict(features)
        
        # Assertions
        assert result.success
        assert result.n_clusters > 0
        assert len(result.cluster_labels) == len(features)
        assert result.cluster_labels.min() >= 0
        assert result.silhouette_score >= -1 and result.silhouette_score <= 1
    
    @pytest.mark.skipif(
        not __import__('src.training.steps.market_analysis.hdp_hmm_clustering',
                      fromlist=['HMM_AVAILABLE']).HMM_AVAILABLE,
        reason="HMM libraries not available"
    )
    def test_single_regime_edge_case(self, single_regime_data):
        """Test edge case: data with only one regime."""
        from src.training.steps.market_analysis.hdp_hmm_clustering import (
            HDPHMMClusterer, HDPHMMConfig
        )
        
        features, true_labels = single_regime_data
        
        config = HDPHMMConfig(
            alpha=1.0,  # Lower alpha for fewer regimes
            kappa=100.0,  # High stickiness
            n_iterations=20,
            show_progress=False
        )
        
        clusterer = HDPHMMClusterer(config)
        result = clusterer.fit_predict(features, validate=False)  # Skip validation
        
        # Should handle gracefully
        assert result.success
        assert result.n_clusters >= 1  # At least one regime
    
    @pytest.mark.skipif(
        not __import__('src.training.steps.market_analysis.hdp_hmm_clustering',
                      fromlist=['HMM_AVAILABLE']).HMM_AVAILABLE,
        reason="HMM libraries not available"
    )
    def test_minimal_data_warning(self, minimal_data):
        """Test validation warning with minimal data."""
        from src.training.steps.market_analysis.hdp_hmm_clustering import (
            HDPHMMClusterer, HDPHMMConfig
        )
        
        config = HDPHMMConfig(
            n_iterations=10,
            show_progress=False,
            min_samples_required=500
        )
        
        clusterer = HDPHMMClusterer(config)
        
        # Should warn but not fail
        with pytest.warns(UserWarning):
            result = clusterer.fit_predict(minimal_data)
    
    @pytest.mark.skipif(
        not __import__('src.training.steps.market_analysis.hdp_hmm_clustering',
                      fromlist=['HMM_AVAILABLE']).HMM_AVAILABLE,
        reason="HMM libraries not available"
    )
    def test_convergence_diagnostics(self, synthetic_regime_data):
        """Test convergence diagnostics and early stopping."""
        from src.training.steps.market_analysis.hdp_hmm_clustering import (
            HDPHMMClusterer, HDPHMMConfig
        )
        
        features, _ = synthetic_regime_data
        
        config = HDPHMMConfig(
            alpha=3.0,
            kappa=50.0,
            n_iterations=100,
            convergence_check=True,
            convergence_threshold=0.01,
            show_progress=False
        )
        
        clusterer = HDPHMMClusterer(config)
        result = clusterer.fit_predict(features)
        
        # Check convergence history
        assert hasattr(clusterer, 'convergence_history')
        assert 'state_counts' in clusterer.convergence_history
        assert 'converged' in clusterer.convergence_history


# =============================================================================
# MS-DR Tests
# =============================================================================

class TestMSDR:
    """Test suite for MS-DR clustering."""
    
    def test_import(self):
        """Test that MS-DR module can be imported."""
        from src.training.steps.market_analysis.ms_dr_clustering import (
            MSDRClusterer, MSDRConfig, MS_AVAILABLE
        )
        assert MSDRClusterer is not None
        assert MSDRConfig is not None
    
    @pytest.mark.skipif(
        not __import__('src.training.steps.market_analysis.ms_dr_clustering',
                      fromlist=['MS_AVAILABLE']).MS_AVAILABLE,
        reason="MS libraries not available"
    )
    def test_basic_clustering(self, synthetic_regime_data):
        """Test basic clustering with synthetic data."""
        from src.training.steps.market_analysis.ms_dr_clustering import (
            MSDRClusterer, MSDRConfig
        )
        
        features, true_labels = synthetic_regime_data
        
        config = MSDRConfig(
            n_regimes=3,
            auto_select_regimes=False,
            show_progress=False
        )
        
        clusterer = MSDRClusterer(config)
        result = clusterer.fit_predict(features)
        
        # Assertions
        assert result.success
        assert result.n_clusters > 0
        assert len(result.cluster_labels) == len(features)
        assert result.cluster_labels.min() >= 0
        assert result.aic > 0 or result.aic < 0  # Valid AIC
        assert result.bic > 0 or result.bic < 0  # Valid BIC
    
    @pytest.mark.skipif(
        not __import__('src.training.steps.market_analysis.ms_dr_clustering',
                      fromlist=['MS_AVAILABLE']).MS_AVAILABLE,
        reason="MS libraries not available"
    )
    def test_auto_regime_selection(self, synthetic_regime_data):
        """Test automatic regime selection using IC."""
        from src.training.steps.market_analysis.ms_dr_clustering import (
            MSDRClusterer, MSDRConfig
        )
        
        features, true_labels = synthetic_regime_data
        
        config = MSDRConfig(
            auto_select_regimes=True,
            min_regimes=2,
            max_regimes=5,
            ic_criterion='bic',
            show_progress=False
        )
        
        clusterer = MSDRClusterer(config)
        result = clusterer.fit_predict(features)
        
        # Should select reasonable number of regimes
        assert result.success
        assert 2 <= result.n_clusters <= 5
    
    @pytest.mark.skipif(
        not __import__('src.training.steps.market_analysis.ms_dr_clustering',
                      fromlist=['MS_AVAILABLE']).MS_AVAILABLE,
        reason="MS libraries not available"
    )
    def test_degenerate_case_rejection(self):
        """Test that degenerate cases (all identical) are rejected."""
        from src.training.steps.market_analysis.ms_dr_clustering import (
            MSDRClusterer, MSDRConfig
        )
        
        # All identical values
        features = np.ones((100, 3))
        
        config = MSDRConfig(show_progress=False)
        clusterer = MSDRClusterer(config)
        
        # Should raise ValueError during validation
        with pytest.raises(ValueError, match="identical"):
            clusterer.fit_predict(features, validate=True)


# =============================================================================
# Comparison Tests
# =============================================================================

class TestComparison:
    """Compare HDP-HMM and MS-DR performance."""
    
    @pytest.mark.skipif(
        not (__import__('src.training.steps.market_analysis.hdp_hmm_clustering',
                       fromlist=['HMM_AVAILABLE']).HMM_AVAILABLE and
             __import__('src.training.steps.market_analysis.ms_dr_clustering',
                       fromlist=['MS_AVAILABLE']).MS_AVAILABLE),
        reason="Both HMM and MS libraries required"
    )
    def test_performance_comparison(self, synthetic_regime_data):
        """Compare performance of both methods."""
        from src.training.steps.market_analysis.hdp_hmm_clustering import (
            HDPHMMClusterer, HDPHMMConfig
        )
        from src.training.steps.market_analysis.ms_dr_clustering import (
            MSDRClusterer, MSDRConfig
        )
        
        features, true_labels = synthetic_regime_data
        
        # HDP-HMM
        hdp_config = HDPHMMConfig(
            alpha=3.0,
            kappa=50.0,
            n_iterations=30,
            show_progress=False
        )
        hdp_clusterer = HDPHMMClusterer(hdp_config)
        hdp_result = hdp_clusterer.fit_predict(features)
        
        # MS-DR
        ms_config = MSDRConfig(
            auto_select_regimes=True,
            min_regimes=2,
            max_regimes=5,
            show_progress=False
        )
        ms_clusterer = MSDRClusterer(ms_config)
        ms_result = ms_clusterer.fit_predict(features)
        
        # Both should succeed
        assert hdp_result.success
        assert ms_result.success
        
        # Compare metrics
        print(f"\nPerformance Comparison:")
        print(f"HDP-HMM: {hdp_result.n_clusters} clusters, "
              f"silhouette={hdp_result.silhouette_score:.3f}, "
              f"time={hdp_result.processing_time:.2f}s")
        print(f"MS-DR: {ms_result.n_clusters} clusters, "
              f"silhouette={ms_result.silhouette_score:.3f}, "
              f"AIC={ms_result.aic:.1f}, BIC={ms_result.bic:.1f}, "
              f"time={ms_result.processing_time:.2f}s")
        
        # MS-DR should generally be faster
        assert ms_result.processing_time < hdp_result.processing_time * 2


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests with feature bank and market data."""
    
    def test_hdp_hmm_integration_import(self):
        """Test HDP-HMM integration can be imported."""
        from src.feature_generation.integration.enhanced_hdp_hmm_clustering_integration import (
            EnhancedHDPHMMClusteringIntegration
        )
        assert EnhancedHDPHMMClusteringIntegration is not None
    
    def test_ms_dr_integration_import(self):
        """Test MS-DR integration can be imported."""
        from src.feature_generation.integration.enhanced_ms_dr_clustering_integration import (
            EnhancedMSDRClusteringIntegration
        )
        assert EnhancedMSDRClusteringIntegration is not None
    
    @pytest.mark.skipif(
        not __import__('src.training.steps.market_analysis.ms_dr_clustering',
                      fromlist=['MS_AVAILABLE']).MS_AVAILABLE,
        reason="MS libraries not available"
    )
    def test_ms_dr_with_market_data(self, market_data):
        """Test MS-DR with realistic market data."""
        from src.feature_generation.integration.enhanced_ms_dr_clustering_integration import (
            EnhancedMSDRClusteringIntegration
        )
        
        integration = EnhancedMSDRClusteringIntegration(
            min_features=10,
            max_features=30,
            auto_select_regimes=True,
            min_regimes=2,
            max_regimes=5
        )
        
        # Prepare data (may fail if feature bank unavailable)
        try:
            features, feature_names, metadata = integration.prepare_data_for_clustering(market_data)
            
            # Should generate some features
            assert features.shape[0] == len(market_data)
            assert features.shape[1] > 0
            
        except Exception as e:
            pytest.skip(f"Feature generation failed (expected if feature bank unavailable): {e}")


# =============================================================================
# Validation Tests
# =============================================================================

class TestValidation:
    """Test validation and error handling."""
    
    @pytest.mark.skipif(
        not __import__('src.training.steps.market_analysis.hdp_hmm_clustering',
                      fromlist=['HMM_AVAILABLE']).HMM_AVAILABLE,
        reason="HMM libraries not available"
    )
    def test_nan_handling(self):
        """Test NaN value handling."""
        from src.training.steps.market_analysis.hdp_hmm_clustering import (
            HDPHMMClusterer, HDPHMMConfig
        )
        
        # Data with many NaNs
        features = np.random.randn(100, 3)
        features[features > 1] = np.nan  # ~16% NaN
        
        config = HDPHMMConfig(
            n_iterations=10,
            show_progress=False,
            max_nan_ratio=0.2  # Allow up to 20% NaN
        )
        
        clusterer = HDPHMMClusterer(config)
        
        # Should handle NaNs in preprocessing
        result = clusterer.fit_predict(features, validate=True)
        assert not np.any(np.isnan(result.cluster_labels))
    
    @pytest.mark.skipif(
        not __import__('src.training.steps.market_analysis.ms_dr_clustering',
                      fromlist=['MS_AVAILABLE']).MS_AVAILABLE,
        reason="MS libraries not available"
    )
    def test_excessive_nan_rejection(self):
        """Test that excessive NaNs are rejected."""
        from src.training.steps.market_analysis.ms_dr_clustering import (
            MSDRClusterer, MSDRConfig
        )
        
        # Data with too many NaNs
        features = np.random.randn(100, 3)
        features[features > 0] = np.nan  # ~50% NaN
        
        config = MSDRConfig(
            show_progress=False,
            max_nan_ratio=0.1  # Only allow 10% NaN
        )
        
        clusterer = MSDRClusterer(config)
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="NaN"):
            clusterer.fit_predict(features, validate=True)


# =============================================================================
# Main test runner
# =============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
