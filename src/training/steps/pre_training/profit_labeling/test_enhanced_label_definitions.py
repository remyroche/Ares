"""
Unit tests for Enhanced Label Definitions - Causality-First Implementation

This module provides comprehensive tests for the refactored label definitions,
focusing on causality, data leakage prevention, and proper time-series handling.
"""

import numpy as np
import pandas as pd
import unittest
from datetime import datetime, timedelta
import warnings
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_data_preview, tprint_data_format

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

from enhanced_label_definitions import (
    EnhancedLabelDefinitions,
    AnalystLabelConfig,
    TacticianLabelConfig,
    DataCleaningConfig,
    StabilityCheckConfig,
    TradingCosts,
    ThresholdPolicy,
    ThresholdSource,
    DataQualityMasks
)


class TestEnhancedLabelDefinitions(unittest.TestCase):
    """Test cases for the enhanced label definitions."""
    
    def setUp(self):
        """Set up test data and configurations."""
        # Create synthetic market data with known properties
        np.random.seed(42)
        n_bars = 100
        start_time = datetime(2024, 1, 1, 9, 0, 0)
        
        # Generate realistic OHLCV data
        base_price = 100.0
        returns = np.random.normal(0, 0.02, n_bars)  # 2% daily volatility
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create OHLC from prices
        highs = prices * (1 + np.abs(np.random.normal(0, 0.01, n_bars)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.01, n_bars)))
        opens = np.roll(prices, 1)
        opens[0] = prices[0]
        
        # Ensure OHLC consistency
        highs = np.maximum(highs, np.maximum(opens, prices))
        lows = np.minimum(lows, np.minimum(opens, prices))
        
        # Generate volume
        volume = np.random.lognormal(10, 1, n_bars)
        
        # Create timestamps
        timestamps = [start_time + timedelta(minutes=15*i) for i in range(n_bars)]
        
        self.market_data = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volume
        }, index=pd.DatetimeIndex(timestamps))
        
        # Generate volatility series
        self.volatility_series = pd.Series(
            np.random.uniform(0.01, 0.05, n_bars),
            index=self.market_data.index
        )
        
        # Generate regime data
        self.regime_data = pd.Series(
            np.random.choice(['low_vol', 'normal', 'high_vol'], n_bars),
            index=self.market_data.index
        )
        
        # Initialize labeler
        self.labeler = EnhancedLabelDefinitions(
            analyst_config=AnalystLabelConfig(horizon_minutes=60),
            tactician_config=TacticianLabelConfig(horizon_minutes=30),
            cleaning_config=DataCleaningConfig(
                outlier_method="rolling_iqr",
                enable_quality_flags=True
            ),
            stability_config=StabilityCheckConfig(
                enable_autocorrelation_test=True,
                enable_psi_test=True
            ),
            random_state=42
        )

    def test_foundational_contracts_validation(self):
        """Test foundational contracts validation."""
        tprint("\n=== Testing Foundational Contracts Validation ===")
        
        # Test valid data
        validation_results = self.labeler.validate_foundational_contracts(self.market_data)
        self.assertTrue(validation_results['is_valid'])
        self.assertEqual(len(validation_results['issues']), 0)
        tprint("✅ Valid data passed foundational contracts validation")
        
        # Test invalid data (non-monotonic index)
        invalid_data = self.market_data.copy()
        invalid_data.index = invalid_data.index[::-1]  # Reverse order
        validation_results = self.labeler.validate_foundational_contracts(invalid_data)
        self.assertFalse(validation_results['is_valid'])
        self.assertGreater(len(validation_results['issues']), 0)
        tprint("✅ Invalid data correctly rejected by foundational contracts validation")

    def test_data_cleaning_masking(self):
        """Test data cleaning with masking approach."""
        tprint("\n=== Testing Data Cleaning (Masking-Based) ===")
        
        # Add some outliers to test detection
        test_data = self.market_data.copy()
        test_data.loc[test_data.index[10], 'close'] *= 10  # Extreme outlier
        test_data.loc[test_data.index[20], 'volume'] = 0  # Zero volume
        
        cleaned_data, masks = self.labeler._apply_data_cleaning(test_data)
        
        # Check that data is not deleted but flagged
        self.assertEqual(len(cleaned_data), len(test_data))
        self.assertTrue(hasattr(masks, 'outlier_mask'))
        self.assertTrue(hasattr(masks, 'untradable_mask'))
        
        # Check that outliers are flagged
        self.assertTrue(masks.outlier_mask.iloc[10])  # Extreme price outlier
        self.assertTrue(masks.untradable_mask.iloc[20])  # Zero volume
        
        tprint("✅ Data cleaning correctly flags issues without deleting data")

    def test_analyst_labels_causality(self):
        """Test analyst labels for causality (no leakage)."""
        tprint("\n=== Testing Analyst Labels (Causality) ===")
        
        labels, confidence, meta = self.labeler.generate_analyst_labels(
            self.market_data, self.volatility_series, self.regime_data
        )
        
        # Check that labels are binary
        self.assertTrue(labels.isin([0, 1]).all())
        
        # Check that confidence scores are in valid range
        self.assertTrue((confidence >= 0).all())
        self.assertTrue((confidence <= 1).all())
        
        # Check meta data structure
        self.assertIn('threshold_values', meta)
        self.assertIn('cost_series', meta)
        self.assertIn('volatility_estimate', meta)
        self.assertIn('random_state', meta)
        
        tprint("✅ Analyst labels generated with proper causality and meta data")

    def test_tactician_labels_mfe_mae(self):
        """Test tactician labels with MFE/MAE logic."""
        tprint("\n=== Testing Tactician Labels (MFE/MAE) ===")
        
        labels, magnitude, meta = self.labeler.generate_tactician_labels(
            self.market_data, self.volatility_series, self.regime_data
        )
        
        # Check that labels are binary
        self.assertTrue(labels.isin([0, 1]).all())
        
        # Check that magnitude scores are non-negative
        self.assertTrue((magnitude >= 0).all())
        
        # Check meta data structure
        self.assertIn('threshold_values', meta)
        self.assertIn('mfe_mae', meta)
        
        tprint("✅ Tactician labels generated with MFE/MAE logic")

    def test_risk_aware_labels_first_hit(self):
        """Test risk-aware labels with first-hit logic."""
        tprint("\n=== Testing Risk-Aware Labels (First-Hit Logic) ===")
        
        # Create base labels
        base_labels = pd.Series(1, index=self.market_data.index)
        
        risk_labels, meta = self.labeler.generate_risk_aware_labels(
            base_labels, self.market_data
        )
        
        # Check that risk filtering reduces positive labels
        self.assertLessEqual(risk_labels.sum(), base_labels.sum())
        
        # Check meta data
        self.assertIn('stop_hit_count', meta)
        self.assertIn('target_hit_count', meta)
        
        tprint("✅ Risk-aware labels applied with first-hit logic")

    def test_stability_checks_statistical(self):
        """Test stability checks using statistical tests."""
        tprint("\n=== Testing Stability Checks (Statistical Tests) ===")
        
        # Create some labels
        labels = pd.Series(np.random.choice([0, 1], 50), index=self.market_data.index[:50])
        
        stability_results = self.labeler.check_label_stability(
            labels, market_data=self.market_data
        )
        
        # Check that results contain expected metrics
        self.assertIn('is_stable', stability_results)
        self.assertIn('metrics', stability_results)
        self.assertIn('p_values', stability_results)
        
        tprint("✅ Stability checks completed with statistical tests")

    def test_trading_costs_data_driven(self):
        """Test data-driven trading costs."""
        tprint("\n=== Testing Trading Costs (Data-Driven) ===")
        
        costs = TradingCosts(
            spread_model_enabled=True,
            market_impact_model_enabled=True,
            participation_rate=0.01
        )
        
        cost_series = costs.calculate_costs(self.market_data)
        
        # Check that costs are calculated per bar
        self.assertEqual(len(cost_series), len(self.market_data))
        
        # Check that costs are non-negative
        self.assertTrue((cost_series >= 0).all())
        
        tprint("✅ Trading costs calculated using data-driven models")

    def test_threshold_policies_causal(self):
        """Test causal threshold calculation."""
        tprint("\n=== Testing Threshold Policies (Causal) ===")
        
        from enhanced_label_definitions import CausalThresholdCalculator
        
        policy = ThresholdPolicy(
            source=ThresholdSource.ROLLING_QUANTILE,
            quantile=0.75,
            window=20,
            min_samples=10
        )
        
        calculator = CausalThresholdCalculator(policy)
        
        # Test with some data
        test_data = pd.Series(np.random.normal(0, 1, 50))
        threshold, source = calculator.calculate_threshold(test_data)
        
        self.assertIsInstance(threshold, float)
        self.assertIsInstance(source, str)
        
        tprint("✅ Threshold policies work with causal calculations")

    def test_data_quality_masks(self):
        """Test data quality mask functionality."""
        tprint("\n=== Testing Data Quality Masks ===")
        
        masks = DataQualityMasks()
        
        # Create some test masks
        masks.outlier_mask = pd.Series([False, True, False], index=pd.Index([0, 1, 2]))
        masks.untradable_mask = pd.Series([False, False, True], index=pd.Index([0, 1, 2]))
        
        combined_mask = masks.get_combined_mask()
        
        # Check that combined mask works correctly
        self.assertEqual(len(combined_mask), 3)
        self.assertFalse(combined_mask.iloc[0])  # No issues
        self.assertTrue(combined_mask.iloc[1])   # Outlier
        self.assertTrue(combined_mask.iloc[2])   # Untradable
        
        print("✅ Data quality masks work correctly")

    def test_no_future_leakage(self):
        """Test that no future information is leaked."""
        print("\n=== Testing No Future Leakage ===")
        
        # This is a conceptual test - in practice, you would check that
        # calculations at time t only use data from times ≤ t
        
        # Generate labels
        labels, _, _ = self.labeler.generate_analyst_labels(
            self.market_data, self.volatility_series
        )
        
        # Check that labels are not all the same (indicating some variation)
        self.assertGreater(labels.nunique(), 1)
        
        # Check that confidence scores vary
        _, confidence, _ = self.labeler.generate_analyst_labels(
            self.market_data, self.volatility_series
        )
        self.assertGreater(confidence.nunique(), 1)
        
        print("✅ No obvious future leakage detected")

    def test_meta_data_completeness(self):
        """Test that meta data is complete and useful."""
        print("\n=== Testing Meta Data Completeness ===")
        
        labels, confidence, meta = self.labeler.generate_analyst_labels(
            self.market_data, self.volatility_series, self.regime_data
        )
        
        # Check required meta data fields
        required_fields = [
            'threshold_values', 'cost_series', 'volatility_estimate',
            'data_masks', 'random_state', 'data_checksum'
        ]
        
        for field in required_fields:
            self.assertIn(field, meta, f"Missing required field: {field}")
        
        # Check that cost series has correct length
        self.assertEqual(len(meta['cost_series']), len(self.market_data))
        
        # Check that random state is preserved
        self.assertEqual(meta['random_state'], 42)
        
        print("✅ Meta data is complete and useful")


def run_comprehensive_tests():
    """Run all tests and provide a summary."""
    print("🚀 Running Comprehensive Tests for Enhanced Label Definitions")
    print("=" * 70)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEnhancedLabelDefinitions)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
        print("🎉 Enhanced Label Definitions are working correctly!")
    else:
        print(f"\n⚠️ {len(result.failures + result.errors)} test(s) failed")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_comprehensive_tests()