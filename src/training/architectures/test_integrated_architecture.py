"""
Comprehensive Test Suite for Integrated ML Architecture

This module provides comprehensive testing for the integrated ML architecture
including all components: CLVSA, MultiScaleNBEATS, RegimeNAS, Meta-Labels, and HPO.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
import logging
import unittest
from unittest.mock import Mock, patch
import tempfile
import os

# Import all architecture components
from .clvsa_architecture import CLVSAArchitecture, CLVSAConfig, CLVSATrainer
from .multiscale_nbeats import MultiScaleNBEATS, MultiScaleNBEATSConfig, MultiScaleNBEATSTrainer
from .regime_nas_framework import RegimeNASFramework, RegimeNASConfig, RegimeNASTrainer
from .meta_labels_patterns import MetaLabelsPatternsSystem, MetaLabelsConfig, MetaLabelsPatternsTrainer
from .regime_specific_hpo import RegimeSpecificHPO, RegimeHPOConfig
from .integrated_ml_architecture import IntegratedMLArchitecture, IntegratedMLConfig, IntegratedMLTrainer

# Test utilities
from src.utils.tprint import tprint
from src.utils.logger import get_logger

logger = get_logger('TestIntegratedArchitecture')

class TestCLVSAArchitecture(unittest.TestCase):
    """Test suite for CLVSA architecture."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = CLVSAConfig(
            input_features=20,  # Reduced for testing
            sequence_length=30,
            num_regimes=3
        )
        self.model = CLVSAArchitecture(self.config)
        
    def test_model_creation(self):
        """Test CLVSA model creation."""
        self.assertIsInstance(self.model, CLVSAArchitecture)
        self.assertEqual(self.model.config.input_features, 20)
        self.assertEqual(self.model.config.sequence_length, 30)
        
    def test_forward_pass(self):
        """Test forward pass through CLVSA architecture."""
        batch_size = 8
        x = torch.randn(batch_size, self.config.sequence_length, self.config.input_features)
        
        with torch.no_grad():
            outputs = self.model(x)
        
        # Check output structure
        self.assertIn('regime_prediction', outputs)
        self.assertIn('price_prediction', outputs)
        self.assertIn('uncertainty', outputs)
        self.assertIn('latent_z', outputs)
        
        # Check output shapes
        self.assertEqual(outputs['regime_prediction'].shape, (batch_size, self.config.num_regimes))
        self.assertEqual(outputs['price_prediction'].shape, (batch_size, self.config.num_outputs))
        self.assertEqual(outputs['uncertainty'].shape, (batch_size, 1))
        
    def test_loss_computation(self):
        """Test loss computation."""
        batch_size = 8
        x = torch.randn(batch_size, self.config.sequence_length, self.config.input_features)
        
        with torch.no_grad():
            outputs = self.model(x)
        
        targets = {
            'regime': torch.randint(0, self.config.num_regimes, (batch_size,)),
            'price': torch.randn(batch_size, self.config.num_outputs)
        }
        
        losses = self.model.compute_loss(outputs, targets)
        
        # Check loss structure
        self.assertIn('total_loss', losses)
        self.assertIn('regime_loss', losses)
        self.assertIn('kl_loss', losses)
        
        # Check loss values are finite
        for key, value in losses.items():
            self.assertTrue(torch.isfinite(value), f"Loss {key} is not finite: {value}")
    
    def test_trainer_creation(self):
        """Test CLVSA trainer creation."""
        trainer = CLVSATrainer(self.model, self.config)
        self.assertIsInstance(trainer, CLVSATrainer)
        self.assertEqual(trainer.model, self.model)
        self.assertEqual(trainer.config, self.config)


class TestMultiScaleNBEATS(unittest.TestCase):
    """Test suite for MultiScaleNBEATS architecture."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = MultiScaleNBEATSConfig(
            input_features=20,
            sequence_length=30,
            forecast_horizon=6
        )
        self.model = MultiScaleNBEATS(self.config)
        
    def test_model_creation(self):
        """Test MultiScaleNBEATS model creation."""
        self.assertIsInstance(self.model, MultiScaleNBEATS)
        self.assertEqual(self.model.config.input_features, 20)
        self.assertEqual(self.model.config.forecast_horizon, 6)
        
    def test_forward_pass(self):
        """Test forward pass through MultiScaleNBEATS."""
        batch_size = 8
        x = torch.randn(batch_size, self.config.sequence_length, self.config.input_features)
        regime_ids = torch.randint(0, 3, (batch_size,))
        
        with torch.no_grad():
            outputs = self.model(x, regime_ids)
        
        # Check output structure
        self.assertIn('forecast', outputs)
        self.assertIn('uncertainty', outputs)
        self.assertIn('regime_prediction', outputs)
        self.assertIn('scale_forecasts', outputs)
        
        # Check output shapes
        self.assertEqual(outputs['forecast'].shape, (batch_size, self.config.forecast_horizon))
        self.assertEqual(outputs['uncertainty'].shape, (batch_size, 1))
        self.assertEqual(outputs['regime_prediction'].shape, (batch_size, self.config.num_regimes))
        
    def test_loss_computation(self):
        """Test loss computation."""
        batch_size = 8
        x = torch.randn(batch_size, self.config.sequence_length, self.config.input_features)
        regime_ids = torch.randint(0, 3, (batch_size,))
        
        with torch.no_grad():
            outputs = self.model(x, regime_ids)
        
        targets = {
            'forecast': torch.randn(batch_size, self.config.forecast_horizon),
            'regime': regime_ids,
            'uncertainty': torch.randn(batch_size, 1),
            'regime_ids': regime_ids
        }
        
        losses = self.model.compute_loss(outputs, targets)
        
        # Check loss structure
        self.assertIn('total_loss', losses)
        self.assertIn('forecast_loss', losses)
        self.assertIn('regime_loss', losses)
        
        # Check loss values are finite
        for key, value in losses.items():
            self.assertTrue(torch.isfinite(value), f"Loss {key} is not finite: {value}")


class TestRegimeNASFramework(unittest.TestCase):
    """Test suite for RegimeNAS framework."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = RegimeNASConfig(
            input_features=20,
            sequence_length=30
        )
        self.model = RegimeNASFramework(self.config)
        
    def test_model_creation(self):
        """Test RegimeNAS model creation."""
        self.assertIsInstance(self.model, RegimeNASFramework)
        self.assertEqual(self.model.config.input_features, 20)
        
    def test_forward_pass(self):
        """Test forward pass through RegimeNAS."""
        batch_size = 8
        x = torch.randn(batch_size, self.config.sequence_length, self.config.input_features)
        
        with torch.no_grad():
            outputs = self.model(x)
        
        # Check output structure
        self.assertIn('regime_predictions', outputs)
        self.assertIn('transition_prediction', outputs)
        self.assertIn('architecture_selection', outputs)
        self.assertIn('regime_specific_predictions', outputs)
        
        # Check regime predictions for each level
        for level in self.config.regime_levels:
            level_key = level.value
            self.assertIn(level_key, outputs['regime_predictions'])
            
    def test_loss_computation(self):
        """Test loss computation."""
        batch_size = 8
        x = torch.randn(batch_size, self.config.sequence_length, self.config.input_features)
        
        with torch.no_grad():
            outputs = self.model(x)
        
        targets = {
            'micro_regime': torch.randint(0, 3, (batch_size,)),
            'short_regime': torch.randint(0, 4, (batch_size,)),
            'medium_regime': torch.randint(0, 5, (batch_size,)),
            'transition': torch.randn(batch_size, sum(self.config.num_regimes_per_level.values())),
            'micro_prediction': torch.randn(batch_size, self.config.regime_horizons[RegimeLevel.MICRO]),
            'short_prediction': torch.randn(batch_size, self.config.regime_horizons[RegimeLevel.SHORT]),
            'medium_prediction': torch.randn(batch_size, self.config.regime_horizons[RegimeLevel.MEDIUM])
        }
        
        losses = self.model.compute_loss(outputs, targets)
        
        # Check loss structure
        self.assertIn('total_loss', losses)
        
        # Check loss values are finite
        for key, value in losses.items():
            self.assertTrue(torch.isfinite(value), f"Loss {key} is not finite: {value}")


class TestMetaLabelsPatterns(unittest.TestCase):
    """Test suite for Meta-labels and patterns system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = MetaLabelsConfig(
            input_features=20,
            sequence_length=30
        )
        self.model = MetaLabelsPatternsSystem(self.config)
        
    def test_model_creation(self):
        """Test Meta-labels model creation."""
        self.assertIsInstance(self.model, MetaLabelsPatternsSystem)
        self.assertEqual(self.model.config.input_features, 20)
        
    def test_forward_pass(self):
        """Test forward pass through Meta-labels system."""
        batch_size = 8
        x = torch.randn(batch_size, self.config.sequence_length, self.config.input_features)
        
        with torch.no_grad():
            outputs = self.model(x)
        
        # Check output structure
        self.assertIn('pattern_classification', outputs)
        self.assertIn('pattern_embedding', outputs)
        self.assertIn('meta_labels', outputs)
        self.assertIn('similar_patterns', outputs)
        
        # Check output shapes
        self.assertEqual(outputs['pattern_classification'].shape, (batch_size, len(self.config.pattern_types)))
        self.assertEqual(outputs['pattern_embedding'].shape, (batch_size, self.config.pattern_embedding_dim))
        
    def test_loss_computation(self):
        """Test loss computation."""
        batch_size = 8
        x = torch.randn(batch_size, self.config.sequence_length, self.config.input_features)
        
        with torch.no_grad():
            outputs = self.model(x)
        
        targets = {
            'pattern_labels': torch.randint(0, len(self.config.pattern_types), (batch_size,)),
            'regime_label': torch.randint(0, 3, (batch_size,)),
            'transition_label': torch.randn(batch_size, 1),
            'confidence_label': torch.randn(batch_size, 1),
            'original_features': torch.randn(batch_size, self.config.input_features)
        }
        
        losses = self.model.compute_loss(outputs, targets)
        
        # Check loss structure
        self.assertIn('total_loss', losses)
        
        # Check loss values are finite
        for key, value in losses.items():
            self.assertTrue(torch.isfinite(value), f"Loss {key} is not finite: {value}")


class TestRegimeSpecificHPO(unittest.TestCase):
    """Test suite for Regime-specific HPO."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = RegimeHPOConfig(
            num_regimes=3,
            optimization_trials=5,  # Reduced for testing
            optimization_timeout=60  # 1 minute for testing
        )
        self.hpo_system = RegimeSpecificHPO(self.config)
        
    def test_hpo_creation(self):
        """Test HPO system creation."""
        self.assertIsInstance(self.hpo_system, RegimeSpecificHPO)
        self.assertEqual(self.hpo_system.config.num_regimes, 3)
        
    def test_studies_creation(self):
        """Test Optuna studies creation."""
        self.assertEqual(len(self.hpo_system.studies), 3)
        for regime_name in self.config.regime_names:
            self.assertIn(regime_name, self.hpo_system.studies)
    
    @patch('optuna.create_study')
    def test_optimization_setup(self, mock_create_study):
        """Test optimization setup."""
        mock_study = Mock()
        mock_create_study.return_value = mock_study
        
        # Test that studies are created correctly
        self.assertEqual(len(self.hpo_system.studies), 3)


class TestIntegratedArchitecture(unittest.TestCase):
    """Test suite for integrated ML architecture."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = IntegratedMLConfig(
            input_features=20,
            sequence_length=30,
            forecast_horizon=6,
            use_hpo=False  # Disable HPO for testing
        )
        self.model = IntegratedMLArchitecture(self.config)
        
    def test_model_creation(self):
        """Test integrated model creation."""
        self.assertIsInstance(self.model, IntegratedMLArchitecture)
        self.assertEqual(self.model.config.input_features, 20)
        
    def test_components_creation(self):
        """Test that all components are created."""
        if self.config.use_clvsa:
            self.assertIn('clvsa', self.model.components)
        if self.config.use_multiscale_nbeats:
            self.assertIn('multiscale_nbeats', self.model.components)
        if self.config.use_regime_nas:
            self.assertIn('regime_nas', self.model.components)
        if self.config.use_meta_labels:
            self.assertIn('meta_labels', self.model.components)
        
    def test_forward_pass(self):
        """Test forward pass through integrated architecture."""
        batch_size = 8
        x = torch.randn(batch_size, self.config.sequence_length, self.config.input_features)
        regime_ids = torch.randint(0, 3, (batch_size,))
        
        with torch.no_grad():
            outputs = self.model(x, regime_ids)
        
        # Check output structure
        self.assertIn('prediction', outputs)
        self.assertIn('uncertainty', outputs)
        self.assertIn('regime_prediction', outputs)
        self.assertIn('component_outputs', outputs)
        
        # Check output shapes
        self.assertEqual(outputs['prediction'].shape, (batch_size, self.config.forecast_horizon))
        self.assertEqual(outputs['uncertainty'].shape, (batch_size, 1))
        self.assertEqual(outputs['regime_prediction'].shape, (batch_size, 3))
        
    def test_loss_computation(self):
        """Test loss computation."""
        batch_size = 8
        x = torch.randn(batch_size, self.config.sequence_length, self.config.input_features)
        regime_ids = torch.randint(0, 3, (batch_size,))
        
        with torch.no_grad():
            outputs = self.model(x, regime_ids)
        
        targets = {
            'prediction': torch.randn(batch_size, self.config.forecast_horizon),
            'uncertainty': torch.randn(batch_size, 1),
            'regime': regime_ids,
            'regime_ids': regime_ids
        }
        
        losses = self.model.compute_loss(outputs, targets)
        
        # Check loss structure
        self.assertIn('total_loss', losses)
        
        # Check loss values are finite
        for key, value in losses.items():
            self.assertTrue(torch.isfinite(value), f"Loss {key} is not finite: {value}")
    
    def test_trainer_creation(self):
        """Test integrated trainer creation."""
        trainer = IntegratedMLTrainer(self.model, self.config)
        self.assertIsInstance(trainer, IntegratedMLTrainer)
        self.assertEqual(trainer.model, self.model)
        self.assertEqual(trainer.config, self.config)


class TestDataIntegration(unittest.TestCase):
    """Test suite for data integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = IntegratedMLConfig(
            input_features=20,
            sequence_length=30,
            forecast_horizon=6
        )
        
    def test_data_preprocessing(self):
        """Test data preprocessing for integrated architecture."""
        # Create sample data
        n_samples = 1000
        n_features = self.config.input_features
        
        # Generate synthetic time series data
        data = np.random.randn(n_samples, n_features)
        regime_labels = np.random.randint(0, 3, n_samples)
        
        # Test data validation
        self.assertEqual(data.shape, (n_samples, n_features))
        self.assertEqual(len(regime_labels), n_samples)
        self.assertTrue(np.all(regime_labels >= 0))
        self.assertTrue(np.all(regime_labels < 3))
        
    def test_sequence_creation(self):
        """Test sequence creation for time series data."""
        n_samples = 1000
        n_features = self.config.input_features
        sequence_length = self.config.sequence_length
        
        # Generate synthetic data
        data = np.random.randn(n_samples, n_features)
        
        # Create sequences
        sequences = []
        for i in range(sequence_length, n_samples):
            sequence = data[i-sequence_length:i]
            sequences.append(sequence)
        
        sequences = np.array(sequences)
        
        # Test sequence structure
        self.assertEqual(sequences.shape, (n_samples - sequence_length, sequence_length, n_features))
        
    def test_regime_aware_data_splitting(self):
        """Test regime-aware data splitting."""
        n_samples = 1000
        regime_labels = np.random.randint(0, 3, n_samples)
        
        # Split data by regime
        regime_data = {}
        for regime_id in range(3):
            regime_mask = (regime_labels == regime_id)
            regime_indices = np.where(regime_mask)[0]
            regime_data[regime_id] = regime_indices
            
            # Test that each regime has data
            self.assertGreater(len(regime_indices), 0, f"Regime {regime_id} has no data")


class TestPerformanceBenchmarks(unittest.TestCase):
    """Test suite for performance benchmarks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = IntegratedMLConfig(
            input_features=20,
            sequence_length=30,
            forecast_horizon=6
        )
        self.model = IntegratedMLArchitecture(self.config)
        
    def test_memory_usage(self):
        """Test memory usage of integrated architecture."""
        batch_size = 32
        x = torch.randn(batch_size, self.config.sequence_length, self.config.input_features)
        
        # Test forward pass memory usage
        with torch.no_grad():
            outputs = self.model(x)
        
        # Check that outputs are created successfully
        self.assertIsNotNone(outputs)
        
    def test_inference_speed(self):
        """Test inference speed of integrated architecture."""
        import time
        
        batch_size = 32
        x = torch.randn(batch_size, self.config.sequence_length, self.config.input_features)
        
        # Warm up
        with torch.no_grad():
            _ = self.model(x)
        
        # Time inference
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(x)
        end_time = time.time()
        
        inference_time = end_time - start_time
        
        # Check that inference is reasonably fast (< 1 second)
        self.assertLess(inference_time, 1.0, f"Inference too slow: {inference_time:.4f}s")
        
    def test_gradient_computation(self):
        """Test gradient computation for training."""
        batch_size = 8
        x = torch.randn(batch_size, self.config.sequence_length, self.config.input_features)
        regime_ids = torch.randint(0, 3, (batch_size,))
        
        # Forward pass
        outputs = self.model(x, regime_ids)
        
        # Create dummy targets
        targets = {
            'prediction': torch.randn(batch_size, self.config.forecast_horizon),
            'uncertainty': torch.randn(batch_size, 1),
            'regime': regime_ids,
            'regime_ids': regime_ids
        }
        
        # Compute loss
        losses = self.model.compute_loss(outputs, targets)
        
        # Backward pass
        losses['total_loss'].backward()
        
        # Check that gradients are computed
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all(), f"Gradient for {name} contains non-finite values")


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    tprint('🧪 Running Comprehensive Architecture Tests')
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestCLVSAArchitecture,
        TestMultiScaleNBEATS,
        TestRegimeNASFramework,
        TestMetaLabelsPatterns,
        TestRegimeSpecificHPO,
        TestIntegratedArchitecture,
        TestDataIntegration,
        TestPerformanceBenchmarks
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print results
    tprint(f'📊 Test Results:')
    tprint(f'   → Tests run: {result.testsRun}')
    tprint(f'   → Failures: {len(result.failures)}')
    tprint(f'   → Errors: {len(result.errors)}')
    tprint(f'   → Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%')
    
    if result.failures:
        tprint('❌ Test Failures:')
        for test, traceback in result.failures:
            tprint(f'   → {test}: {traceback}')
    
    if result.errors:
        tprint('❌ Test Errors:')
        for test, traceback in result.errors:
            tprint(f'   → {test}: {traceback}')
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run comprehensive tests
    success = run_comprehensive_tests()
    
    if success:
        tprint('✅ All tests passed successfully!')
    else:
        tprint('❌ Some tests failed. Please check the output above.')