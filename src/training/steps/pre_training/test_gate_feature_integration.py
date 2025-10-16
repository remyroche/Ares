"""
Test Gate Feature Integration

This module provides comprehensive tests for the gate feature integration system.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any

from src.training.steps.pre_training.gate_feature_integration import (
    GateFeaturePipelineManager, GateFeatureConfig, GateFeatureResult,
    GateStatus, GateFeatureType, create_gate_manager
)
from src.training.steps.pre_training.gate_feature_pipeline_integration import (
    GateFeaturePipelineIntegration, create_gate_feature_integration,
    integrate_gate_features_with_pipeline
)


class TestGateFeatureIntegration(unittest.TestCase):
    """Test cases for gate feature integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_features = pd.DataFrame({
            'feature_1': np.random.randn(1000),
            'feature_2': np.random.randn(1000),
            'feature_3': np.random.randn(1000),
            'feature_4': np.random.randn(1000),
            'feature_5': np.random.randn(1000)
        })
        
        self.sample_targets = pd.Series(np.random.randint(0, 2, 1000))
        
        # Create problematic features for testing
        self.problematic_features = pd.DataFrame({
            'good_feature': np.random.randn(1000),
            'high_nan_feature': [np.nan if i % 3 == 0 else np.random.randn() for i in range(1000)],
            'low_variance_feature': np.ones(1000) + np.random.normal(0, 0.001, 1000),
            'correlated_feature': self.sample_features['feature_1'] + np.random.normal(0, 0.1, 1000)
        })
        
        self.problematic_targets = pd.Series(np.random.randint(0, 2, 1000))
    
    def test_gate_feature_manager_initialization(self):
        """Test gate feature manager initialization."""
        manager = create_gate_manager()
        self.assertIsInstance(manager, GateFeaturePipelineManager)
        self.assertTrue(manager.is_gate_protection_enabled())
    
    def test_gate_feature_config(self):
        """Test gate feature configuration."""
        config = GateFeatureConfig(
            enable_gate_protection=True,
            max_gate_features_per_base=5,
            min_gate_ic_improvement=0.01
        )
        
        self.assertTrue(config.enable_gate_protection)
        self.assertEqual(config.max_gate_features_per_base, 5)
        self.assertEqual(config.min_gate_ic_improvement, 0.01)
    
    def test_quality_gate_validation(self):
        """Test quality gate validation."""
        manager = create_gate_manager()
        results = manager.evaluate_gate_features(self.sample_features, self.sample_targets)
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # Check that we have quality gate results
        quality_results = [r for r in results if r.gate_type == GateFeatureType.QUALITY_GATE]
        self.assertGreater(len(quality_results), 0)
    
    def test_correlation_gate_validation(self):
        """Test correlation gate validation."""
        manager = create_gate_manager()
        results = manager.evaluate_gate_features(self.sample_features, self.sample_targets)
        
        # Check for correlation gate results
        correlation_results = [r for r in results if r.gate_type == GateFeatureType.CORRELATION_GATE]
        self.assertGreater(len(correlation_results), 0)
    
    def test_variance_gate_validation(self):
        """Test variance gate validation."""
        manager = create_gate_manager()
        results = manager.evaluate_gate_features(self.sample_features, self.sample_targets)
        
        # Check for variance gate results
        variance_results = [r for r in results if r.gate_type == GateFeatureType.VARIANCE_GATE]
        self.assertGreater(len(variance_results), 0)
    
    def test_gate_feature_selection(self):
        """Test gate feature selection."""
        manager = create_gate_manager()
        selected_features = manager.select_gate_features(self.sample_features, self.sample_targets)
        
        self.assertIsInstance(selected_features, list)
        self.assertLessEqual(len(selected_features), manager.config.max_gate_features_per_base)
        
        # Check that selected features exist in the original data
        for feature in selected_features:
            self.assertIn(feature, self.sample_features.columns)
    
    def test_problematic_data_handling(self):
        """Test handling of problematic data."""
        manager = create_gate_manager()
        results = manager.evaluate_gate_features(self.problematic_features, self.problematic_targets)
        
        # Should detect issues with problematic data
        failed_results = [r for r in results if r.status == GateStatus.FAILED]
        warning_results = [r for r in results if r.status == GateStatus.WARNING]
        
        # Should have some failed or warning results due to problematic data
        self.assertGreater(len(failed_results) + len(warning_results), 0)
    
    def test_gate_status_tracking(self):
        """Test gate status tracking."""
        manager = create_gate_manager()
        
        # Initial status
        status = manager.get_gate_status()
        self.assertIn('enabled', status)
        self.assertIn('active_gates', status)
        self.assertIn('total_evaluations', status)
        
        # Evaluate gates
        manager.evaluate_gate_features(self.sample_features, self.sample_targets)
        
        # Check updated status
        updated_status = manager.get_gate_status()
        self.assertGreater(updated_status['total_evaluations'], status['total_evaluations'])
    
    def test_gate_protection_enable_disable(self):
        """Test enabling and disabling gate protection."""
        manager = create_gate_manager()
        
        # Initially enabled
        self.assertTrue(manager.is_gate_protection_enabled())
        
        # Disable
        manager.disable_gate_protection()
        self.assertFalse(manager.is_gate_protection_enabled())
        
        # Enable
        manager.enable_gate_protection()
        self.assertTrue(manager.is_gate_protection_enabled())
    
    def test_pipeline_integration(self):
        """Test pipeline integration."""
        integration = create_gate_feature_integration()
        
        # Test data
        test_data = {
            'features': self.sample_features,
            'targets': self.sample_targets
        }
        
        # Process through integration
        result = integration.process(test_data)
        
        self.assertTrue(result.success)
        self.assertIn('gate_features', result.data)
        self.assertIn('gate_status', result.data)
    
    def test_corrective_measures(self):
        """Test corrective measures for problematic data."""
        integration = create_gate_feature_integration()
        
        # Test with problematic data
        test_data = {
            'features': self.problematic_features,
            'targets': self.problematic_targets
        }
        
        result = integration.process(test_data)
        
        # Should handle problematic data
        self.assertTrue(result.success)
        
        # Check that corrective measures were applied
        if 'gate_features' in result.data:
            gate_results = result.data['gate_features']['gate_results']
            self.assertIsInstance(gate_results, list)
    
    def test_gate_statistics(self):
        """Test gate statistics tracking."""
        integration = create_gate_feature_integration()
        
        # Process some data
        test_data = {
            'features': self.sample_features,
            'targets': self.sample_targets
        }
        
        integration.process(test_data)
        
        # Get statistics
        stats = integration.get_gate_statistics()
        
        self.assertIn('total_evaluations', stats)
        self.assertIn('success_rate', stats)
        self.assertIn('gate_manager_status', stats)
    
    def test_integration_convenience_function(self):
        """Test the convenience integration function."""
        test_data = {
            'features': self.sample_features,
            'targets': self.sample_targets
        }
        
        result = integrate_gate_features_with_pipeline(test_data)
        
        self.assertIn('gate_features', result)
        self.assertIn('gate_status', result)
    
    def test_gate_feature_result_creation(self):
        """Test gate feature result creation."""
        result = GateFeatureResult(
            feature_name="test_feature",
            gate_type=GateFeatureType.QUALITY_GATE,
            status=GateStatus.PASSED,
            score=0.95,
            threshold=0.8,
            message="Test passed"
        )
        
        self.assertEqual(result.feature_name, "test_feature")
        self.assertEqual(result.gate_type, GateFeatureType.QUALITY_GATE)
        self.assertEqual(result.status, GateStatus.PASSED)
        self.assertEqual(result.score, 0.95)
        self.assertEqual(result.threshold, 0.8)
        self.assertEqual(result.message, "Test passed")
    
    def test_gate_feature_types(self):
        """Test all gate feature types."""
        gate_types = [
            GateFeatureType.QUALITY_GATE,
            GateFeatureType.STABILITY_GATE,
            GateFeatureType.PERFORMANCE_GATE,
            GateFeatureType.DATA_INTEGRITY_GATE,
            GateFeatureType.FEATURE_IMPORTANCE_GATE,
            GateFeatureType.CORRELATION_GATE,
            GateFeatureType.VARIANCE_GATE,
            GateFeatureType.OUTLIER_GATE
        ]
        
        for gate_type in gate_types:
            self.assertIsInstance(gate_type.value, str)
            self.assertGreater(len(gate_type.value), 0)
    
    def test_gate_status_types(self):
        """Test all gate status types."""
        status_types = [
            GateStatus.PASSED,
            GateStatus.FAILED,
            GateStatus.WARNING,
            GateStatus.SKIPPED
        ]
        
        for status in status_types:
            self.assertIsInstance(status.value, str)
            self.assertGreater(len(status.value), 0)


def run_gate_feature_tests():
    """Run all gate feature integration tests."""
    print("🧪 Running Gate Feature Integration Tests...")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGateFeatureIntegration)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        print("✅ All gate feature integration tests passed!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_gate_feature_tests()
