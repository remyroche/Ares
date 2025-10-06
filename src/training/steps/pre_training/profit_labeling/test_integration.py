"""
Integration Test for Volatility-Aware Multi-Horizon Profit Labeling System

This module provides integration tests to ensure the system works correctly
with the existing codebase and produces high-quality labels.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import unittest
from typing import Dict, Any

# Import the volatility-aware labeling system
from .volatility_aware_labeler import (
    VolatilityAwareMultiHorizonLabeler,
    VolatilityAwareConfig,
    LabelingResult
)

from .bar_construction import BarConstructionConfig, BarType
from .volatility_modeling import VolatilityConfig, VolatilityMethod
from .noise_gating import NoiseGatingConfig, NoiseGateType
from .quality_scoring import QualityScoringConfig
from .multi_target_scheme import MultiTargetConfig, TargetBand

# Import utilities
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning


def generate_test_data(n_samples: int = 1000, 
                      start_date: str = "2023-01-01",
                      timeframe_minutes: int = 15) -> pd.DataFrame:
    """Generate test data for integration testing."""
    # Create datetime index
    start_dt = pd.to_datetime(start_date)
    timestamps = pd.date_range(
        start=start_dt,
        periods=n_samples,
        freq=f'{timeframe_minutes}T'
    )
    
    # Generate realistic price data
    np.random.seed(42)
    base_price = 100.0
    
    # Generate returns with trend and volatility
    returns = np.random.normal(0, 0.02, n_samples)
    
    # Add volatility clustering
    volatility = np.ones(n_samples) * 0.02
    for i in range(1, n_samples):
        volatility[i] = 0.95 * volatility[i-1] + 0.05 * 0.02 + 0.01 * np.random.normal()
    
    returns = returns * volatility
    
    # Calculate prices
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = []
    for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
        # Generate intraday volatility
        intraday_vol = volatility[i] * 0.1
        
        # Generate OHLC
        open_price = price
        high_price = price * (1 + abs(np.random.normal(0, intraday_vol)))
        low_price = price * (1 - abs(np.random.normal(0, intraday_vol)))
        close_price = price * (1 + np.random.normal(0, intraday_vol))
        
        # Ensure OHLC relationships
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Generate volume
        volume = np.random.lognormal(10, 0.5)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df


class TestVolatilityAwareLabeling(unittest.TestCase):
    """Test cases for volatility-aware labeling system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data = generate_test_data(n_samples=2000)
        self.config = VolatilityAwareConfig(
            min_data_points=500,
            enable_caching=False,  # Disable caching for tests
            parallel_processing=False  # Disable parallel processing for tests
        )
        self.labeler = VolatilityAwareMultiHorizonLabeler(self.config)
    
    def test_basic_functionality(self):
        """Test basic functionality of the labeling system."""
        tprint_info("🧪 Testing basic functionality")
        
        # Generate labels
        result = self.labeler.generate_labels(self.test_data)
        
        # Check result type
        self.assertIsInstance(result, LabelingResult)
        
        # Check basic properties
        self.assertIsInstance(result.labels, pd.DataFrame)
        self.assertIsInstance(result.confidence_scores, pd.DataFrame)
        self.assertIsInstance(result.eligibility_masks, pd.DataFrame)
        self.assertIsInstance(result.quality_scores, dict)
        
        # Check processing time
        self.assertGreaterEqual(result.processing_time, 0)
        
        tprint_success("✅ Basic functionality test passed")
    
    def test_data_validation(self):
        """Test data validation."""
        tprint_info("🧪 Testing data validation")
        
        # Test with empty data
        empty_data = pd.DataFrame()
        result = self.labeler.generate_labels(empty_data)
        self.assertTrue(result.labels.empty)
        
        # Test with insufficient data
        small_data = self.test_data.head(10)
        result = self.labeler.generate_labels(small_data)
        self.assertTrue(result.labels.empty)
        
        # Test with missing columns
        incomplete_data = self.test_data.drop(columns=['volume'])
        result = self.labeler.generate_labels(incomplete_data)
        self.assertTrue(result.labels.empty)
        
        tprint_success("✅ Data validation test passed")
    
    def test_bar_construction(self):
        """Test bar construction component."""
        tprint_info("🧪 Testing bar construction")
        
        from .bar_construction import EventBasedBarConstructor, BarConstructionConfig
        
        # Test dollar bars
        config = BarConstructionConfig(bar_type=BarType.DOLLAR, bar_size=100000.0)
        constructor = EventBasedBarConstructor(config)
        result = constructor.construct_bars(self.test_data)
        
        self.assertIsNotNone(result.cleaned_bars)
        self.assertGreaterEqual(result.n_cleaned_bars, 0)
        self.assertGreaterEqual(result.data_quality_score, 0)
        
        tprint_success("✅ Bar construction test passed")
    
    def test_volatility_modeling(self):
        """Test volatility modeling component."""
        tprint_info("🧪 Testing volatility modeling")
        
        from .volatility_modeling import VolatilityModeler, VolatilityConfig
        
        # Test combined volatility
        config = VolatilityConfig(method=VolatilityMethod.COMBINED)
        modeler = VolatilityModeler(config)
        result = modeler.model_volatility(self.test_data)
        
        self.assertIsNotNone(result.volatility_series)
        self.assertGreaterEqual(result.mean_volatility, 0)
        self.assertGreaterEqual(result.volatility_consistency, 0)
        
        tprint_success("✅ Volatility modeling test passed")
    
    def test_noise_gating(self):
        """Test noise gating component."""
        tprint_info("🧪 Testing noise gating")
        
        from .noise_gating import NoiseGatingFilter, NoiseGatingConfig
        from .volatility_modeling import VolatilityModeler, VolatilityConfig
        
        # Generate volatility
        vol_config = VolatilityConfig()
        vol_modeler = VolatilityModeler(vol_config)
        vol_result = vol_modeler.model_volatility(self.test_data)
        
        # Test noise gating
        config = NoiseGatingConfig(gate_type=NoiseGateType.COMBINED)
        filter_obj = NoiseGatingFilter(config)
        result = filter_obj.filter_noise(self.test_data, vol_result.volatility_series)
        
        self.assertIsNotNone(result.eligibility_mask)
        self.assertGreaterEqual(result.eligibility_ratio, 0)
        self.assertLessEqual(result.eligibility_ratio, 1)
        
        tprint_success("✅ Noise gating test passed")
    
    def test_quality_scoring(self):
        """Test quality scoring component."""
        tprint_info("🧪 Testing quality scoring")
        
        from .quality_scoring import LabelQualityScorer, QualityScoringConfig
        
        # Generate sample labels
        sample_labels = pd.DataFrame({
            'target_1': np.random.choice([-1, 0, 1], size=len(self.test_data), p=[0.3, 0.4, 0.3]),
            'target_2': np.random.choice([-1, 0, 1], size=len(self.test_data), p=[0.2, 0.6, 0.2])
        }, index=self.test_data.index)
        
        sample_confidence = pd.DataFrame({
            'target_1': np.random.uniform(0, 1, size=len(self.test_data)),
            'target_2': np.random.uniform(0, 1, size=len(self.test_data))
        }, index=self.test_data.index)
        
        sample_eligibility = pd.DataFrame({
            'target_1': np.random.choice([True, False], size=len(self.test_data), p=[0.8, 0.2]),
            'target_2': np.random.choice([True, False], size=len(self.test_data), p=[0.7, 0.3])
        }, index=self.test_data.index)
        
        # Test quality scoring
        config = QualityScoringConfig()
        scorer = LabelQualityScorer(config)
        result = scorer.assess_quality(sample_labels, sample_confidence, sample_eligibility, self.test_data)
        
        self.assertIsInstance(result, dict)
        for target_name, quality_metrics in result.items():
            self.assertGreaterEqual(quality_metrics.lqs_score, 0)
            self.assertLessEqual(quality_metrics.lqs_score, 1)
        
        tprint_success("✅ Quality scoring test passed")
    
    def test_multi_target_scheme(self):
        """Test multi-target scheme component."""
        tprint_info("🧪 Testing multi-target scheme")
        
        from .multi_target_scheme import MultiTargetScheme, MultiTargetConfig
        from .volatility_modeling import VolatilityModeler, VolatilityConfig
        from .noise_gating import NoiseGatingFilter, NoiseGatingConfig
        
        # Generate volatility
        vol_config = VolatilityConfig()
        vol_modeler = VolatilityModeler(vol_config)
        vol_result = vol_modeler.model_volatility(self.test_data)
        
        # Generate eligibility mask
        noise_config = NoiseGatingConfig()
        noise_filter = NoiseGatingFilter(noise_config)
        elig_result = noise_filter.filter_noise(self.test_data, vol_result.volatility_series)
        
        # Test multi-target scheme
        config = MultiTargetConfig(
            small_band=(0.5, 0.8),
            medium_band=(0.8, 1.2),
            high_band=(1.2, 1.8),
            max_targets_per_band=1
        )
        scheme = MultiTargetScheme(config)
        result = scheme.generate_targets(self.test_data, vol_result.volatility_series, elig_result.eligibility_mask)
        
        self.assertIsNotNone(result.labels)
        self.assertIsNotNone(result.confidence_scores)
        self.assertIsNotNone(result.eligibility_masks)
        self.assertGreaterEqual(result.n_targets, 0)
        
        tprint_success("✅ Multi-target scheme test passed")
    
    def test_end_to_end_integration(self):
        """Test end-to-end integration."""
        tprint_info("🧪 Testing end-to-end integration")
        
        # Generate labels
        result = self.labeler.generate_labels(self.test_data)
        
        # Check that we get some results
        if not result.labels.empty:
            self.assertGreater(result.n_targets, 0)
            self.assertGreater(result.n_samples, 0)
            
            # Check label values are valid
            for col in result.labels.columns:
                unique_values = set(result.labels[col].dropna().unique())
                self.assertTrue(unique_values.issubset({-1, 0, 1}))
            
            # Check confidence scores are in valid range
            for col in result.confidence_scores.columns:
                if not result.confidence_scores[col].empty:
                    min_conf = result.confidence_scores[col].min()
                    max_conf = result.confidence_scores[col].max()
                    self.assertGreaterEqual(min_conf, 0)
                    self.assertLessEqual(max_conf, 1)
            
            # Check eligibility masks are boolean
            for col in result.eligibility_masks.columns:
                if not result.eligibility_masks[col].empty:
                    unique_values = set(result.eligibility_masks[col].dropna().unique())
                    self.assertTrue(unique_values.issubset({True, False}))
        
        tprint_success("✅ End-to-end integration test passed")
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        tprint_info("🧪 Testing configuration validation")
        
        # Test with custom configuration
        custom_config = VolatilityAwareConfig(
            min_data_points=100,
            enable_caching=True,
            parallel_processing=True
        )
        
        custom_labeler = VolatilityAwareMultiHorizonLabeler(custom_config)
        result = custom_labeler.generate_labels(self.test_data)
        
        self.assertIsInstance(result, LabelingResult)
        self.assertEqual(result.config_used, custom_config)
        
        tprint_success("✅ Configuration validation test passed")
    
    def test_performance_requirements(self):
        """Test performance requirements."""
        tprint_info("🧪 Testing performance requirements")
        
        # Test processing time
        result = self.labeler.generate_labels(self.test_data)
        
        # Should complete within reasonable time (adjust as needed)
        max_processing_time = 60.0  # seconds
        self.assertLess(result.processing_time, max_processing_time)
        
        # Test memory usage (basic check)
        if not result.labels.empty:
            memory_usage = result.labels.memory_usage(deep=True).sum()
            self.assertLess(memory_usage, 100 * 1024 * 1024)  # 100MB limit
        
        tprint_success("✅ Performance requirements test passed")


def run_integration_tests():
    """Run all integration tests."""
    tprint_info("🚀 Running Integration Tests")
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestVolatilityAwareLabeling)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    if result.wasSuccessful():
        tprint_success("✅ All integration tests passed!")
    else:
        tprint_warning(f"⚠️ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        for failure in result.failures:
            tprint_warning(f"   → FAIL: {failure[0]}")
        for error in result.errors:
            tprint_warning(f"   → ERROR: {error[0]}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run integration tests
    success = run_integration_tests()
    
    if success:
        print("\n🎉 All tests passed! The volatility-aware labeling system is ready for use.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")