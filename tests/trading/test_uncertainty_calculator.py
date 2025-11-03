"""
Unit Tests for UncertaintyCalculator

Tests the uncertainty calculation logic including:
- Ensemble variance calculation
- Model disagreement measurement
- Confidence degradation tracking
- Combined uncertainty metrics
"""

import unittest
from datetime import datetime
from typing import List, Dict
import numpy as np
import pandas as pd

from src.utils.ml_common.uncertainty_calculator import (
    UncertaintyCalculator,
    create_uncertainty_calculator,
    get_global_uncertainty_calculator
)


class TestUncertaintyCalculator(unittest.TestCase):
    """Test suite for UncertaintyCalculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'variance_weight': 0.4,
            'disagreement_weight': 0.4,
            'confidence_weight': 0.2,
            'degradation_window': 8,
            'degradation_method': 'relative_change'
        }
        self.calculator = UncertaintyCalculator(self.config)
    
    def test_ensemble_variance_low_uncertainty(self):
        """Test ensemble variance with low uncertainty (predictions close together)."""
        # Predictions very close together -> low variance
        predictions = [0.70, 0.71, 0.69, 0.70, 0.72]
        
        variance = self.calculator.calculate_ensemble_variance(predictions)
        
        # Should be very low variance
        self.assertGreater(variance, 0.0)
        self.assertLess(variance, 0.01)
    
    def test_ensemble_variance_high_uncertainty(self):
        """Test ensemble variance with high uncertainty (predictions spread out)."""
        # Predictions spread out -> high variance
        predictions = [0.3, 0.8, 0.5, 0.9, 0.4]
        
        variance = self.calculator.calculate_ensemble_variance(predictions)
        
        # Should have higher variance
        self.assertGreater(variance, 0.01)
    
    def test_ensemble_variance_empty_predictions(self):
        """Test ensemble variance with empty predictions."""
        predictions = []
        
        variance = self.calculator.calculate_ensemble_variance(predictions)
        
        # Should return 0.0 for empty list
        self.assertEqual(variance, 0.0)
    
    def test_ensemble_variance_single_prediction(self):
        """Test ensemble variance with single prediction."""
        predictions = [0.75]
        
        variance = self.calculator.calculate_ensemble_variance(predictions)
        
        # Single prediction -> zero variance
        self.assertEqual(variance, 0.0)
    
    def test_ensemble_variance_numpy_array(self):
        """Test ensemble variance with numpy array input."""
        predictions = np.array([0.70, 0.71, 0.69, 0.72])
        
        variance = self.calculator.calculate_ensemble_variance(predictions)
        
        # Should handle numpy arrays and return valid float
        self.assertIsInstance(variance, float)
        self.assertGreaterEqual(variance, 0.0)  # Can be very small for close predictions
        self.assertLess(variance, 1.0)  # Should be reasonable
    
    def test_model_disagreement_low(self):
        """Test model disagreement with similar predictions."""
        predictions = {
            'lightgbm': 0.70,
            'catboost': 0.71,
            'xgboost': 0.69
        }
        
        disagreement = self.calculator.calculate_model_disagreement(predictions)
        
        # Low disagreement
        self.assertGreater(disagreement, 0.0)
        self.assertLess(disagreement, 0.1)
    
    def test_model_disagreement_high(self):
        """Test model disagreement with divergent predictions."""
        predictions = {
            'lightgbm': 0.3,
            'catboost': 0.9,
            'xgboost': 0.5
        }
        
        disagreement = self.calculator.calculate_model_disagreement(predictions)
        
        # High disagreement
        self.assertGreater(disagreement, 0.2)
    
    def test_model_disagreement_insufficient_models(self):
        """Test model disagreement with only one model."""
        predictions = {'lightgbm': 0.75}
        
        disagreement = self.calculator.calculate_model_disagreement(predictions)
        
        # Should return 0.0 for single model
        self.assertEqual(disagreement, 0.0)
    
    def test_model_disagreement_numpy_values(self):
        """Test model disagreement with numpy array predictions."""
        predictions = {
            'model1': np.array([0.70, 0.71, 0.72]),
            'model2': np.array([0.68, 0.69, 0.70]),
            'model3': np.array([0.73, 0.74, 0.75])
        }
        
        disagreement = self.calculator.calculate_model_disagreement(predictions)
        
        # Should handle numpy arrays by taking mean
        self.assertIsInstance(disagreement, float)
        self.assertGreaterEqual(disagreement, 0.0)
        self.assertLessEqual(disagreement, 1.0)
    
    def test_confidence_degradation_decreasing(self):
        """Test confidence degradation with decreasing confidence."""
        # Confidence dropping over time
        confidence_series = [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45]
        
        degradation = self.calculator.calculate_confidence_degradation(confidence_series)
        
        # Should be negative (degradation)
        self.assertLess(degradation, 0.0)
        
        # Should be about -43.75% = (0.45 - 0.8) / 0.8
        expected = (0.45 - 0.8) / 0.8
        self.assertAlmostEqual(degradation, expected, places=2)
    
    def test_confidence_degradation_increasing(self):
        """Test confidence degradation with increasing confidence."""
        # Confidence improving over time
        confidence_series = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
        
        degradation = self.calculator.calculate_confidence_degradation(confidence_series)
        
        # Should be positive (improvement)
        self.assertGreater(degradation, 0.0)
        
        # Should be about +70% = (0.85 - 0.5) / 0.5
        expected = (0.85 - 0.5) / 0.5
        self.assertAlmostEqual(degradation, expected, places=2)
    
    def test_confidence_degradation_stable(self):
        """Test confidence degradation with stable confidence."""
        # Confidence stable over time
        confidence_series = [0.7, 0.71, 0.69, 0.70, 0.71, 0.70, 0.69, 0.70]
        
        degradation = self.calculator.calculate_confidence_degradation(confidence_series)
        
        # Should be near zero
        self.assertAlmostEqual(degradation, 0.0, delta=0.05)
    
    def test_confidence_degradation_with_window(self):
        """Test confidence degradation with custom window."""
        confidence_series = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45]
        
        # Use only last 4 values
        degradation = self.calculator.calculate_confidence_degradation(
            confidence_series,
            window=4
        )
        
        # Should calculate from last 4: [0.55, 0.5, 0.45] -> (0.45 - 0.55) / 0.55
        # Actually uses 4 values: [0.55, 0.5, 0.45, 0.4] but we only have 0.45 as last
        # Window of 4 takes last 4 elements: [0.55, 0.5, 0.45]
        # Wait, we have 10 elements, last 4 are: [0.6, 0.55, 0.5, 0.45]
        expected = (0.45 - 0.6) / 0.6
        self.assertAlmostEqual(degradation, expected, places=2)
    
    def test_confidence_degradation_absolute_method(self):
        """Test confidence degradation with absolute change method."""
        confidence_series = [0.8, 0.75, 0.7, 0.65]
        
        # Use absolute change method
        degradation = self.calculator.calculate_confidence_degradation(
            confidence_series,
            method='absolute_change'
        )
        
        # Should be absolute difference: 0.65 - 0.8 = -0.15
        self.assertAlmostEqual(degradation, -0.15, places=2)
    
    def test_combine_uncertainty_metrics_all_provided(self):
        """Test combining all uncertainty metrics."""
        variance = 0.2
        disagreement = 0.3
        confidence_degradation = -0.4  # 40% drop
        
        combined = self.calculator.combine_uncertainty_metrics(
            variance=variance,
            disagreement=disagreement,
            confidence_degradation=confidence_degradation
        )
        
        # Combined should be weighted average of normalized metrics
        # variance: 0.2 (clipped to 0.2)
        # disagreement: 0.3
        # degradation uncertainty: clip(-(-0.4), 0, 1) = 0.4
        # combined = (0.2*0.4 + 0.3*0.4 + 0.4*0.2) / 1.0 = 0.28
        expected = (0.2 * 0.4 + 0.3 * 0.4 + 0.4 * 0.2) / 1.0
        self.assertAlmostEqual(combined, expected, places=2)
    
    def test_combine_uncertainty_metrics_partial(self):
        """Test combining with only some metrics provided."""
        disagreement = 0.5
        
        combined = self.calculator.combine_uncertainty_metrics(
            disagreement=disagreement
        )
        
        # Should only use disagreement with its weight
        # combined = 0.5 (normalized by disagreement_weight)
        self.assertAlmostEqual(combined, disagreement, places=2)
    
    def test_combine_uncertainty_metrics_none_provided(self):
        """Test combining with no metrics provided."""
        combined = self.calculator.combine_uncertainty_metrics()
        
        # Should return 0.0 when nothing provided
        self.assertEqual(combined, 0.0)
    
    def test_comprehensive_metrics_calculation(self):
        """Test comprehensive metrics calculation."""
        ensemble_predictions = [0.7, 0.72, 0.68, 0.71]
        model_predictions = {
            'lightgbm': 0.70,
            'catboost': 0.72,
            'xgboost': 0.68
        }
        confidence_history = [0.8, 0.75, 0.7, 0.65]
        
        metrics = self.calculator.calculate_comprehensive_metrics(
            ensemble_predictions=ensemble_predictions,
            model_predictions=model_predictions,
            confidence_history=confidence_history
        )
        
        # Verify all metrics are present
        self.assertIn('ensemble_variance', metrics)
        self.assertIn('model_disagreement', metrics)
        self.assertIn('confidence_degradation', metrics)
        self.assertIn('combined_uncertainty', metrics)
        self.assertIn('timestamp', metrics)
        
        # Verify all metrics are valid floats
        self.assertIsInstance(metrics['ensemble_variance'], float)
        self.assertIsInstance(metrics['model_disagreement'], float)
        self.assertIsInstance(metrics['confidence_degradation'], float)
        self.assertIsInstance(metrics['combined_uncertainty'], float)
        
        # Verify ranges
        self.assertGreaterEqual(metrics['ensemble_variance'], 0.0)
        self.assertGreaterEqual(metrics['model_disagreement'], 0.0)
        self.assertLessEqual(metrics['model_disagreement'], 1.0)
        self.assertGreaterEqual(metrics['combined_uncertainty'], 0.0)
        self.assertLessEqual(metrics['combined_uncertainty'], 1.0)
    
    def test_factory_function(self):
        """Test factory function creates calculator correctly."""
        calculator = create_uncertainty_calculator(self.config)
        
        self.assertIsInstance(calculator, UncertaintyCalculator)
        self.assertEqual(calculator.variance_weight, 0.4)
        self.assertEqual(calculator.disagreement_weight, 0.4)
    
    def test_global_calculator_singleton(self):
        """Test global calculator is singleton."""
        calc1 = get_global_uncertainty_calculator()
        calc2 = get_global_uncertainty_calculator()
        
        # Should return same instance
        self.assertIs(calc1, calc2)


# Test runner
def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestUncertaintyCalculator)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("=" * 80)
    print("UncertaintyCalculator Unit Tests")
    print("=" * 80)
    print()
    
    result = run_tests()
    
    print()
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print()
    
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
        
    exit(0 if result.wasSuccessful() else 1)

