"""
Test Suite for Models Training ModularComponent Integration

This module provides comprehensive tests for the ModularComponent architecture
in the models training pipeline, including unit tests, integration tests,
and performance tests.

Test Coverage:
- Core ModularComponent functionality
- BaseModelsTrainingComponent features
- Migration utilities
- AnalystTrainingPipelineModular
- Error handling and edge cases
- Performance monitoring
- State management
- Configuration management
"""

import unittest
import logging
import time
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd

# Import the components to test
from src.training.steps.models_training.unified_data_driven_pipeline.core.modular_architecture import (
    ModularComponent, ExampleModularComponent, ValidationLevel, ValidationResult,
    ErrorInfo, ErrorSeverity, ErrorCategory, create_modular_component
)

from src.training.steps.models_training.unified_data_driven_pipeline.core.migration_utils import (
    ModelsTrainingMigrationUtils, analyze_component, validate_migration_compatibility,
    create_component_wrapper, migrate_component
)

from src.training.steps.models_training.components.base_component import BaseModelsTrainingComponent
from src.training.steps.models_training.components.analyst_training_pipeline_modular import (
    AnalystTrainingPipelineModular, AnalystModelType, create_analyst_training_pipeline
)


class TestModularComponent(unittest.TestCase):
    """Test cases for ModularComponent base class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger('test')
        self.config = {
            'test_param': 'test_value',
            'memory_limit_mb': 512,
            'slow_operation_threshold': 1.0
        }
    
    def test_initialization(self):
        """Test component initialization."""
        component = ExampleModularComponent('test_component', self.config, self.logger)
        
        self.assertEqual(component.name, 'test_component')
        self.assertEqual(component.get_config('test_param'), 'test_value')
        self.assertFalse(component.is_initialized())
    
    def test_initialization_success(self):
        """Test successful initialization."""
        component = ExampleModularComponent('test_component', self.config, self.logger)
        
        result = component.initialize()
        self.assertTrue(result)
        self.assertTrue(component.is_initialized())
    
    def test_config_management(self):
        """Test configuration management."""
        component = ExampleModularComponent('test_component', self.config, self.logger)
        
        # Test get_config
        self.assertEqual(component.get_config('test_param'), 'test_value')
        self.assertEqual(component.get_config('nonexistent', 'default'), 'default')
        
        # Test update_config
        component.update_config({'new_param': 'new_value'})
        self.assertEqual(component.get_config('new_param'), 'new_value')
    
    def test_state_management(self):
        """Test state management."""
        component = ExampleModularComponent('test_component', self.config, self.logger)
        component.initialize()
        
        # Test set_state and get_state
        component.set_state('test_key', 'test_value')
        self.assertEqual(component.get_state('test_key'), 'test_value')
        self.assertEqual(component.get_state('nonexistent', 'default'), 'default')
        
        # Test has_state
        self.assertTrue(component.has_state('test_key'))
        self.assertFalse(component.has_state('nonexistent'))
        
        # Test get_all_state
        all_state = component.get_all_state()
        self.assertIn('test_key', all_state)
        
        # Test remove_state
        removed_value = component.remove_state('test_key')
        self.assertEqual(removed_value, 'test_value')
        self.assertFalse(component.has_state('test_key'))
    
    def test_ml_state_management(self):
        """Test ML-specific state management."""
        component = ExampleModularComponent('test_component', self.config, self.logger)
        component.initialize()
        
        # Test ML state
        component.set_ml_state('model_weights', {'layer1': np.array([1, 2, 3])})
        self.assertIsNotNone(component.get_ml_state('model_weights'))
        
        # Test training progress
        component.update_training_progress(1, {'accuracy': 0.9, 'loss': 0.1})
        progress = component.get_ml_state('training_progress')
        self.assertIn(1, progress)
        self.assertEqual(progress[1]['accuracy'], 0.9)
    
    def test_performance_monitoring(self):
        """Test performance monitoring."""
        component = ExampleModularComponent('test_component', self.config, self.logger)
        component.initialize()
        
        # Test performance stats
        stats = component.get_performance_stats()
        self.assertIn('total_operations', stats)
        self.assertIn('success_rate', stats)
        self.assertEqual(stats['total_operations'], 0)
        
        # Test performance summary
        summary = component.get_performance_summary()
        self.assertIn('performance_grade', summary)
        self.assertIn('recommendations', summary)
    
    def test_health_monitoring(self):
        """Test health monitoring."""
        component = ExampleModularComponent('test_component', self.config, self.logger)
        
        # Test status before initialization
        status = component.get_status()
        self.assertFalse(status['initialized'])
        self.assertEqual(status['health'], 'not_initialized')
        
        # Test status after initialization
        component.initialize()
        status = component.get_status()
        self.assertTrue(status['initialized'])
        self.assertEqual(status['health'], 'healthy')
        
        # Test health report
        health = component.get_health_report()
        self.assertIn('overall_health', health)
        self.assertIn('health_score', health)
    
    def test_serialization(self):
        """Test component serialization."""
        component = ExampleModularComponent('test_component', self.config, self.logger)
        component.initialize()
        component.set_state('test_key', 'test_value')
        
        # Test serialize
        serialized = component.serialize()
        self.assertIn('component_class', serialized)
        self.assertIn('name', serialized)
        self.assertIn('config', serialized)
        self.assertIn('state', serialized)
        
        # Test deserialize
        new_component = ExampleModularComponent('new_component')
        new_component.deserialize(serialized)
        self.assertEqual(new_component.name, 'test_component')
        self.assertEqual(new_component.get_state('test_key'), 'test_value')
    
    def test_file_serialization(self):
        """Test file-based serialization."""
        component = ExampleModularComponent('test_component', self.config, self.logger)
        component.initialize()
        component.set_state('test_key', 'test_value')
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            # Test save_to_file
            component.save_to_file(filepath)
            self.assertTrue(os.path.exists(filepath))
            
            # Test load_from_file
            new_component = ExampleModularComponent('new_component')
            new_component.load_from_file(filepath)
            self.assertEqual(new_component.name, 'test_component')
            self.assertEqual(new_component.get_state('test_key'), 'test_value')
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_validation(self):
        """Test input validation."""
        component = ExampleModularComponent('test_component', self.config, self.logger)
        component.initialize()
        
        # Test valid data
        valid_data = pd.DataFrame({'required_column': [1, 2, 3, 4, 5]})
        result = component.validate_input(valid_data)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
        
        # Test invalid data
        invalid_data = pd.DataFrame({'wrong_column': [1, 2, 3]})
        result = component.validate_input(invalid_data)
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)
    
    def test_safe_processing(self):
        """Test safe processing with error handling."""
        component = ExampleModularComponent('test_component', self.config, self.logger)
        component.initialize()
        
        # Test successful processing
        data = pd.DataFrame({'required_column': [1, 2, 3, 4, 5]})
        result = component._safe_process(data)
        self.assertIsNotNone(result)
        
        # Test processing with invalid data
        invalid_data = None
        with self.assertRaises(ValueError):
            component._safe_process(invalid_data)
    
    def test_cleanup(self):
        """Test component cleanup."""
        component = ExampleModularComponent('test_component', self.config, self.logger)
        component.initialize()
        component.set_state('test_key', 'test_value')
        
        # Test cleanup
        component.cleanup()
        self.assertFalse(component.is_initialized())
        self.assertEqual(len(component.get_all_state()), 0)


class TestBaseModelsTrainingComponent(unittest.TestCase):
    """Test cases for BaseModelsTrainingComponent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger('test')
        self.config = {
            'model': {
                'type': 'neural_network',
                'architecture': 'simple'
            },
            'training': {
                'epochs': 10,
                'batch_size': 32,
                'learning_rate': 0.001
            },
            'validation': {
                'split': 0.2,
                'metrics': ['accuracy', 'loss']
            }
        }
    
    def test_initialization(self):
        """Test component initialization."""
        component = BaseModelsTrainingComponent('test_component', self.config, self.logger)
        
        self.assertEqual(component.name, 'test_component')
        self.assertEqual(component.model_config['type'], 'neural_network')
        self.assertEqual(component.training_config['epochs'], 10)
    
    def test_ml_state_management(self):
        """Test ML-specific state management."""
        component = BaseModelsTrainingComponent('test_component', self.config, self.logger)
        component.initialize()
        
        # Test ML state
        component.set_ml_state('model_weights', {'layer1': np.array([1, 2, 3])})
        self.assertIsNotNone(component.get_ml_state('model_weights'))
        
        # Test training progress
        component.update_training_progress(1, {'accuracy': 0.9, 'loss': 0.1})
        progress = component.get_ml_state('training_progress')
        self.assertIn(1, progress)
    
    def test_training_lifecycle(self):
        """Test training lifecycle methods."""
        component = BaseModelsTrainingComponent('test_component', self.config, self.logger)
        component.initialize()
        
        # Test start training
        result = component.start_training()
        self.assertTrue(result)
        self.assertTrue(component.get_ml_state('training_started'))
        
        # Test stop training
        component.stop_training()
        self.assertFalse(component.get_ml_state('training_started'))
    
    def test_training_summary(self):
        """Test training summary generation."""
        component = BaseModelsTrainingComponent('test_component', self.config, self.logger)
        component.initialize()
        
        summary = component.get_training_summary()
        self.assertIn('component_name', summary)
        self.assertIn('training_state', summary)
        self.assertIn('ml_state', summary)
        self.assertIn('performance_stats', summary)


class TestAnalystTrainingPipelineModular(unittest.TestCase):
    """Test cases for AnalystTrainingPipelineModular."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger('test')
        self.config = {
            'model': {
                'base_models': ['tcn', 'lightgbm', 'ridge'],
                'ensemble_method': 'voting'
            },
            'training': {
                'epochs': 5,
                'batch_size': 32,
                'learning_rate': 0.001
            },
            'validation': {
                'split': 0.2,
                'metrics': ['accuracy', 'precision', 'recall']
            },
            'regime_aware': True,
            'timeframe': '5m'
        }
        
        # Create sample training data
        self.training_data = {
            'X_train': pd.DataFrame({
                'feature1': np.random.randn(100),
                'feature2': np.random.randn(100),
                'feature3': np.random.randn(100)
            }),
            'y_train': np.random.randint(0, 2, 100),
            'X_val': pd.DataFrame({
                'feature1': np.random.randn(20),
                'feature2': np.random.randn(20),
                'feature3': np.random.randn(20)
            }),
            'y_val': np.random.randint(0, 2, 20)
        }
    
    def test_initialization(self):
        """Test component initialization."""
        component = AnalystTrainingPipelineModular('test_pipeline', self.config, self.logger)
        
        self.assertEqual(component.name, 'test_pipeline')
        self.assertEqual(len(component.analyst_config.model_types), 3)
        self.assertTrue(component.analyst_config.regime_aware)
    
    def test_model_config_initialization(self):
        """Test model configuration initialization."""
        component = AnalystTrainingPipelineModular('test_pipeline', self.config, self.logger)
        component.initialize()
        
        model_configs = component.get_ml_state('model_configs')
        self.assertIn('tcn', model_configs)
        self.assertIn('lightgbm', model_configs)
        self.assertIn('ridge', model_configs)
    
    def test_training_data_validation(self):
        """Test training data validation."""
        component = AnalystTrainingPipelineModular('test_pipeline', self.config, self.logger)
        component.initialize()
        
        # Test valid data
        result = component._validate_training_data(self.training_data)
        self.assertTrue(result)
        
        # Test invalid data
        invalid_data = {'X_train': self.training_data['X_train']}
        result = component._validate_training_data(invalid_data)
        self.assertFalse(result)
    
    def test_base_models_training(self):
        """Test base models training."""
        component = AnalystTrainingPipelineModular('test_pipeline', self.config, self.logger)
        component.initialize()
        
        result = component._train_base_models(self.training_data)
        self.assertTrue(result['success'])
        self.assertGreater(len(result['models']), 0)
    
    def test_ensemble_training(self):
        """Test ensemble model training."""
        component = AnalystTrainingPipelineModular('test_pipeline', self.config, self.logger)
        component.initialize()
        
        # First train base models
        base_result = component._train_base_models(self.training_data)
        
        # Then train ensemble
        ensemble_result = component._train_ensemble_model(self.training_data, base_result['models'])
        self.assertTrue(ensemble_result['success'])
        self.assertIsNotNone(ensemble_result['ensemble_model'])
    
    def test_full_training_pipeline(self):
        """Test full training pipeline."""
        component = AnalystTrainingPipelineModular('test_pipeline', self.config, self.logger)
        component.initialize()
        
        result = component._process_data(self.training_data)
        self.assertIsNotNone(result)
        self.assertTrue(result.success)
        self.assertGreater(len(result.models), 0)
        self.assertIsNotNone(result.ensemble_model)
    
    def test_training_summary(self):
        """Test training summary generation."""
        component = AnalystTrainingPipelineModular('test_pipeline', self.config, self.logger)
        component.initialize()
        
        summary = component.get_training_summary()
        self.assertIn('analyst_config', summary)
        self.assertIn('training_models', summary)
        self.assertIn('ensemble_trained', summary)


class TestMigrationUtils(unittest.TestCase):
    """Test cases for migration utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger('test')
        self.utils = ModelsTrainingMigrationUtils(self.logger)
    
    def test_component_analysis(self):
        """Test component analysis."""
        analysis = self.utils.analyze_component(ExampleModularComponent)
        
        self.assertEqual(analysis.component_name, 'ExampleModularComponent')
        self.assertTrue(analysis.has_init)
        self.assertTrue(analysis.has_process)
        self.assertGreater(analysis.compatibility_score, 0.5)
    
    def test_migration_compatibility_validation(self):
        """Test migration compatibility validation."""
        # Test compatible component
        result = self.utils.validate_migration_compatibility(ExampleModularComponent)
        self.assertTrue(result)
        
        # Test incompatible component (no __init__)
        class IncompatibleComponent:
            def process(self, data):
                return data
        
        result = self.utils.validate_migration_compatibility(IncompatibleComponent)
        self.assertFalse(result)
    
    def test_component_wrapper_creation(self):
        """Test component wrapper creation."""
        wrapper_class = self.utils.create_component_wrapper(ExampleModularComponent)
        
        self.assertTrue(issubclass(wrapper_class, ModularComponent))
        self.assertEqual(wrapper_class.__name__, 'ExampleModularComponentModularWrapper')
    
    def test_component_migration(self):
        """Test component migration."""
        result = self.utils.migrate_component(ExampleModularComponent)
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.migrated_component)
        self.assertGreater(result.compatibility_score, 0.5)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger('test')
        self.config = {
            'model': {
                'base_models': ['tcn', 'lightgbm'],
                'ensemble_method': 'voting'
            },
            'training': {
                'epochs': 3,
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }
        
        self.training_data = {
            'X_train': pd.DataFrame({
                'feature1': np.random.randn(50),
                'feature2': np.random.randn(50),
                'feature3': np.random.randn(50)
            }),
            'y_train': np.random.randint(0, 2, 50),
            'X_val': pd.DataFrame({
                'feature1': np.random.randn(10),
                'feature2': np.random.randn(10),
                'feature3': np.random.randn(10)
            }),
            'y_val': np.random.randint(0, 2, 10)
        }
    
    def test_end_to_end_training(self):
        """Test end-to-end training pipeline."""
        # Create component
        component = create_analyst_training_pipeline(self.config, self.logger)
        
        # Initialize
        self.assertTrue(component.initialize())
        
        # Process data
        result = component.process(self.training_data)
        
        # Verify result
        self.assertIsNotNone(result)
        self.assertTrue(result.success)
        self.assertGreater(len(result.models), 0)
        self.assertIsNotNone(result.ensemble_model)
        
        # Cleanup
        component.cleanup()
    
    def test_performance_monitoring(self):
        """Test performance monitoring during training."""
        component = create_analyst_training_pipeline(self.config, self.logger)
        component.initialize()
        
        # Process data
        result = component.process(self.training_data)
        
        # Check performance stats
        stats = component.get_performance_stats()
        self.assertGreater(stats['total_operations'], 0)
        self.assertGreater(stats['success_rate'], 0)
        
        # Check health
        health = component.get_health_report()
        self.assertIn('overall_health', health)
        self.assertGreater(health['health_score'], 0)
    
    def test_state_persistence(self):
        """Test state persistence and recovery."""
        component = create_analyst_training_pipeline(self.config, self.logger)
        component.initialize()
        
        # Process data
        result = component.process(self.training_data)
        
        # Serialize
        serialized = component.serialize()
        
        # Create new component and deserialize
        new_component = create_analyst_training_pipeline(self.config, self.logger)
        new_component.deserialize(serialized)
        
        # Verify state
        self.assertEqual(new_component.name, component.name)
        self.assertTrue(new_component.get_ml_state('base_models_trained'))


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestModularComponent))
    test_suite.addTest(unittest.makeSuite(TestBaseModelsTrainingComponent))
    test_suite.addTest(unittest.makeSuite(TestAnalystTrainingPipelineModular))
    test_suite.addTest(unittest.makeSuite(TestMigrationUtils))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)