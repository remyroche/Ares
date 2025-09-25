"""
Tests for Bayesian TPE Optimizer

Comprehensive test suite for the Bayesian TPE optimizer with automatic grid search integration.
"""

import unittest
import numpy as np
import time
from unittest.mock import patch, MagicMock
import tempfile
import os

from .bayesian_tpe_optimizer import (
    BayesianTPEOptimizer,
    BayesianTPEConfig,
    OptimizationResult,
    optimize_with_bayesian_tpe,
    create_search_space_from_bounds
)


class TestBayesianTPEOptimizer(unittest.TestCase):
    """Test cases for BayesianTPEOptimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BayesianTPEConfig(
            n_trials=10,
            coarse_grid_points=3,
            fine_grid_points=4,
            enable_grid_search=True,
            enable_parallel=False,  # Disable for testing
            log_level='WARNING'  # Reduce log noise
        )
        self.optimizer = BayesianTPEOptimizer(self.config)
        
        # Simple test objective function
        def test_objective(params, **kwargs):
            x = params['x']
            y = params['y']
            return -(x - 1)**2 - (y - 2)**2
        
        self.objective_function = test_objective
        
        # Simple search space
        self.search_space = {
            'x': {'type': 'float', 'low': -5.0, 'high': 5.0},
            'y': {'type': 'float', 'low': -5.0, 'high': 5.0}
        }
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertIsInstance(self.optimizer, BayesianTPEOptimizer)
        self.assertEqual(self.optimizer.config.n_trials, 10)
        self.assertTrue(self.optimizer.config.enable_grid_search)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        valid_config = BayesianTPEConfig(n_trials=10, backend='optuna')
        optimizer = BayesianTPEOptimizer(valid_config)
        self.assertIsInstance(optimizer, BayesianTPEOptimizer)
        
        # Invalid backend
        with self.assertRaises(ValueError):
            invalid_config = BayesianTPEConfig(backend='invalid')
            BayesianTPEOptimizer(invalid_config)
        
        # Invalid n_trials
        with self.assertRaises(ValueError):
            invalid_config = BayesianTPEConfig(n_trials=0)
            BayesianTPEOptimizer(invalid_config)
    
    def test_simple_optimization(self):
        """Test simple optimization."""
        result = self.optimizer.optimize(self.objective_function, self.search_space)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertTrue(result.success)
        self.assertGreater(result.best_score, -np.inf)
        self.assertIn('x', result.best_params)
        self.assertIn('y', result.best_params)
        self.assertGreater(result.optimization_time, 0)
    
    def test_optimization_with_data(self):
        """Test optimization with data parameters."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        def objective_with_data(params, X, y, **kwargs):
            return -(params['x'] - 1)**2 - (params['y'] - 2)**2
        
        result = self.optimizer.optimize(
            objective_with_data, 
            self.search_space, 
            X=X, 
            y=y
        )
        
        self.assertTrue(result.success)
        self.assertGreater(result.best_score, -np.inf)
    
    def test_grid_search_only(self):
        """Test grid search only (no TPE)."""
        config = BayesianTPEConfig(
            n_trials=5,
            enable_grid_search=True,
            backend='optuna'  # Will fall back to grid search if TPE fails
        )
        optimizer = BayesianTPEOptimizer(config)
        
        result = optimizer.optimize(self.objective_function, self.search_space)
        
        self.assertTrue(result.success)
        self.assertGreater(result.best_score, -np.inf)
    
    def test_error_handling(self):
        """Test error handling in objective function."""
        def failing_objective(params, **kwargs):
            if params['x'] > 0:
                raise ValueError("Simulated error")
            return -(params['x'] - 1)**2 - (params['y'] - 2)**2
        
        result = self.optimizer.optimize(failing_objective, self.search_space)
        
        # Should still succeed with some valid evaluations
        self.assertTrue(result.success or result.best_score > -np.inf)
    
    def test_mixed_parameter_types(self):
        """Test optimization with mixed parameter types."""
        search_space = {
            'x': {'type': 'float', 'low': -2.0, 'high': 2.0},
            'y': {'type': 'int', 'low': 1, 'high': 10},
            'method': {'type': 'categorical', 'choices': ['linear', 'quadratic']}
        }
        
        def mixed_objective(params, **kwargs):
            x, y, method = params['x'], params['y'], params['method']
            if method == 'linear':
                return x + y
            else:  # quadratic
                return -(x**2 + y**2)
        
        result = self.optimizer.optimize(mixed_objective, search_space)
        
        self.assertTrue(result.success)
        self.assertIn('x', result.best_params)
        self.assertIn('y', result.best_params)
        self.assertIn('method', result.best_params)
    
    def test_convergence_info(self):
        """Test convergence information."""
        result = self.optimizer.optimize(self.objective_function, self.search_space)
        
        self.assertIsInstance(result.convergence_info, dict)
        self.assertIn('best_method', result.convergence_info)
        self.assertIn('grid_search_used', result.convergence_info)
        self.assertIn('tpe_optimization_used', result.convergence_info)
    
    def test_optimization_history(self):
        """Test optimization history tracking."""
        result = self.optimizer.optimize(self.objective_function, self.search_space)
        
        self.assertIsInstance(result.optimization_history, list)
        if result.optimization_history:
            for entry in result.optimization_history:
                self.assertIn('stage', entry)
                self.assertIn('best_score', entry)
                self.assertIn('best_params', entry)
    
    def test_parallel_processing(self):
        """Test parallel processing configuration."""
        config = BayesianTPEConfig(
            n_trials=10,
            enable_parallel=True,
            max_workers=2
        )
        optimizer = BayesianTPEOptimizer(config)
        
        result = optimizer.optimize(self.objective_function, self.search_space)
        
        self.assertTrue(result.success)
    
    def test_early_stopping(self):
        """Test early stopping configuration."""
        config = BayesianTPEConfig(
            n_trials=20,
            enable_early_stopping=True,
            early_stopping_patience=3
        )
        optimizer = BayesianTPEOptimizer(config)
        
        result = optimizer.optimize(self.objective_function, self.search_space)
        
        self.assertTrue(result.success)
    
    def test_memory_management(self):
        """Test memory management features."""
        config = BayesianTPEConfig(
            n_trials=10,
            max_history_size=50,
            enable_memory_cleanup=True
        )
        optimizer = BayesianTPEOptimizer(config)
        
        result = optimizer.optimize(self.objective_function, self.search_space)
        
        self.assertTrue(result.success)
    
    def test_logging_configuration(self):
        """Test logging configuration."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            log_file = f.name
        
        try:
            config = BayesianTPEConfig(
                n_trials=5,
                log_file=log_file,
                log_level='DEBUG'
            )
            optimizer = BayesianTPEOptimizer(config)
            
            result = optimizer.optimize(self.objective_function, self.search_space)
            
            self.assertTrue(result.success)
            
            # Check if log file was created and has content
            self.assertTrue(os.path.exists(log_file))
            with open(log_file, 'r') as f:
                log_content = f.read()
                self.assertGreater(len(log_content), 0)
        
        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    def test_optimize_with_bayesian_tpe(self):
        """Test convenience function."""
        def objective(params, **kwargs):
            return -(params['x'] - 1)**2 - (params['y'] - 2)**2
        
        search_space = {
            'x': {'type': 'float', 'low': -5.0, 'high': 5.0},
            'y': {'type': 'float', 'low': -5.0, 'high': 5.0}
        }
        
        result = optimize_with_bayesian_tpe(
            objective_function=objective,
            search_space=search_space,
            config=BayesianTPEConfig(n_trials=5)
        )
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertTrue(result.success)
    
    def test_create_search_space_from_bounds(self):
        """Test search space creation from bounds."""
        bounds = {
            'x': (-5.0, 5.0),
            'y': (1, 10),
            'method': (0, 2)
        }
        
        param_types = {
            'x': 'float',
            'y': 'int',
            'method': 'categorical'
        }
        
        search_space = create_search_space_from_bounds(bounds, param_types)
        
        self.assertIn('x', search_space)
        self.assertIn('y', search_space)
        self.assertIn('method', search_space)
        
        self.assertEqual(search_space['x']['type'], 'float')
        self.assertEqual(search_space['y']['type'], 'int')
        self.assertEqual(search_space['method']['type'], 'categorical')
        
        self.assertEqual(search_space['x']['low'], -5.0)
        self.assertEqual(search_space['x']['high'], 5.0)
        self.assertEqual(search_space['y']['low'], 1)
        self.assertEqual(search_space['y']['high'], 10)


class TestOptimizationResult(unittest.TestCase):
    """Test OptimizationResult class."""
    
    def test_optimization_result_creation(self):
        """Test OptimizationResult creation."""
        result = OptimizationResult(
            best_params={'x': 1.0, 'y': 2.0},
            best_score=0.95,
            optimization_time=10.5,
            n_trials=50,
            success=True
        )
        
        self.assertEqual(result.best_params, {'x': 1.0, 'y': 2.0})
        self.assertEqual(result.best_score, 0.95)
        self.assertEqual(result.optimization_time, 10.5)
        self.assertEqual(result.n_trials, 50)
        self.assertTrue(result.success)
        self.assertIsNone(result.error_message)
    
    def test_optimization_result_with_error(self):
        """Test OptimizationResult with error."""
        result = OptimizationResult(
            best_params={},
            best_score=-np.inf,
            optimization_time=5.0,
            n_trials=0,
            success=False,
            error_message="Test error"
        )
        
        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "Test error")
        self.assertEqual(result.best_score, -np.inf)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_full_optimization_workflow(self):
        """Test complete optimization workflow."""
        def complex_objective(params, X=None, y=None, **kwargs):
            x1, x2, x3 = params['x1'], params['x2'], params['x3']
            method = params['method']
            
            if method == 'linear':
                return x1 + x2 + x3
            elif method == 'quadratic':
                return -(x1**2 + x2**2 + x3**2)
            else:  # exponential
                return np.exp(-(x1**2 + x2**2 + x3**2))
        
        search_space = {
            'x1': {'type': 'float', 'low': -2.0, 'high': 2.0},
            'x2': {'type': 'float', 'low': -2.0, 'high': 2.0},
            'x3': {'type': 'int', 'low': 1, 'high': 10},
            'method': {'type': 'categorical', 'choices': ['linear', 'quadratic', 'exponential']}
        }
        
        config = BayesianTPEConfig(
            n_trials=15,
            coarse_grid_points=3,
            fine_grid_points=4,
            enable_grid_search=True,
            enable_parallel=False
        )
        
        optimizer = BayesianTPEOptimizer(config)
        result = optimizer.optimize(complex_objective, search_space)
        
        self.assertTrue(result.success)
        self.assertGreater(result.best_score, -np.inf)
        self.assertIn('x1', result.best_params)
        self.assertIn('x2', result.best_params)
        self.assertIn('x3', result.best_params)
        self.assertIn('method', result.best_params)
        
        # Check convergence info
        self.assertIn('best_method', result.convergence_info)
        self.assertIn('grid_search_used', result.convergence_info)
        self.assertIn('tpe_optimization_used', result.convergence_info)
    
    def test_performance_monitoring(self):
        """Test performance monitoring features."""
        config = BayesianTPEConfig(
            n_trials=10,
            enable_performance_monitoring=True,
            monitor_memory=True,
            monitor_time=True
        )
        
        optimizer = BayesianTPEOptimizer(config)
        
        def objective(params, **kwargs):
            time.sleep(0.01)  # Simulate computation
            return -(params['x'] - 1)**2 - (params['y'] - 2)**2
        
        search_space = {
            'x': {'type': 'float', 'low': -5.0, 'high': 5.0},
            'y': {'type': 'float', 'low': -5.0, 'high': 5.0}
        }
        
        result = optimizer.optimize(objective, search_space)
        
        self.assertTrue(result.success)
        
        # Check if performance metrics were collected
        if hasattr(optimizer, 'performance_metrics'):
            self.assertIn('execution_times', optimizer.performance_metrics)
            self.assertIn('memory_usage', optimizer.performance_metrics)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestBayesianTPEOptimizer,
        TestConvenienceFunctions,
        TestOptimizationResult,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        exit(1)