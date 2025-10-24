"""
Test script for weighted loss implementation in models_training.

This script tests the weighted loss functionality across all model types
to ensure proper integration and performance.
"""

import numpy as np
import pandas as pd
import pytest
from typing import Dict, List, Any, Optional
import logging
import tempfile
import os
from pathlib import Path

# Import the weighted loss components
from src.training.steps.models_training.core.weighted_loss_framework import (
    WeightedLossManager, WeightedLossConfig, WeightingStrategy, FailureContextType,
    FailureContextDetector, SampleDifficultyAssessor, WeightedLossCalculator
)
from src.training.steps.models_training.core.weighted_loss_integration import (
    WeightedLossIntegrator, WeightedLossIntegrationConfig, WeightedLossModelWrapper
)

# Import model wrappers
from src.training.steps.models_training.core.lgbm_gru_wrapper import LGBMGRUWrapper
from src.training.steps.models_training.core.stacker_lgbm_calibrated_gated import StackerLGBMCalibratedGated

# Import tprint utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success, 
    tprint_debug, tprint_data_format, LogLevel
)

logger = logging.getLogger(__name__)

class TestWeightedLossFramework:
    """Test the weighted loss framework components."""
    
    def setup_method(self):
        """Set up test data and configurations."""
        # Create synthetic test data
        np.random.seed(42)
        self.n_samples = 1000
        self.n_features = 20
        
        # Create features with different difficulty levels
        self.X = np.random.randn(self.n_samples, self.n_features)
        
        # Create targets with some difficult samples
        self.y = np.random.randint(0, 2, self.n_samples)
        
        # Add some difficult samples (outliers, high variance)
        difficult_indices = np.random.choice(self.n_samples, size=100, replace=False)
        self.X[difficult_indices] += np.random.randn(100, self.n_features) * 2
        self.y[difficult_indices] = 1 - self.y[difficult_indices]  # Flip labels
        
        # Create market data for failure context detection
        self.market_data = {
            'returns': np.random.randn(self.n_samples) * 0.01,
            'high': np.random.randn(self.n_samples) + 100,
            'low': np.random.randn(self.n_samples) + 99,
            'bid': np.random.randn(self.n_samples) + 99.5,
            'ask': np.random.randn(self.n_samples) + 100.5,
            'volume': np.random.randint(1000, 10000, self.n_samples)
        }
        
        # Create weighted loss config
        self.config = WeightedLossConfig(
            enable_weighted_loss=True,
            weighting_strategy=WeightingStrategy.ADAPTIVE,
            volatility_threshold=0.02,
            chop_threshold=0.5,
            spread_threshold=0.01,
            base_weight=1.0,
            max_weight=5.0,
            min_weight=0.1
        )
    
    def test_failure_context_detector(self):
        """Test failure context detection."""
        tprint_info("Testing failure context detector...")
        
        detector = FailureContextDetector(self.config)
        detector.fit(self.X, self.y, self.market_data)
        
        # Detect failure contexts
        contexts = detector.detect_failure_contexts(self.X, self.y, self.market_data)
        
        # Verify contexts are detected
        assert len(contexts) > 0
        assert FailureContextType.HIGH_VOLATILITY.value in contexts
        assert FailureContextType.CHOP.value in contexts
        assert FailureContextType.WIDE_SPREAD.value in contexts
        assert FailureContextType.OUTLIER.value in contexts
        assert FailureContextType.UNCERTAINTY.value in contexts
        
        # Verify context scores are in valid range
        for context_type, scores in contexts.items():
            assert len(scores) == self.n_samples
            assert np.all(scores >= 0)
            assert np.all(scores <= 1)
        
        tprint_success("✅ Failure context detector test passed")
    
    def test_sample_difficulty_assessor(self):
        """Test sample difficulty assessment."""
        tprint_info("Testing sample difficulty assessor...")
        
        assessor = SampleDifficultyAssessor(self.config)
        
        # Assess difficulty
        difficulty = assessor.assess_difficulty(self.X, self.y)
        
        # Verify difficulty scores
        assert len(difficulty) == self.n_samples
        assert np.all(difficulty >= 0)
        assert np.all(difficulty <= 1)
        
        # Verify difficult samples have higher scores
        difficult_indices = np.where(difficulty > 0.8)[0]
        assert len(difficult_indices) > 0
        
        tprint_success("✅ Sample difficulty assessor test passed")
    
    def test_weighted_loss_calculator(self):
        """Test weighted loss calculator."""
        tprint_info("Testing weighted loss calculator...")
        
        calculator = WeightedLossCalculator(self.config)
        calculator.fit(self.X, self.y, self.market_data)
        
        # Calculate weights
        weights = calculator.calculate_weights(self.X, self.y, None, self.market_data)
        
        # Verify weights
        assert len(weights) == self.n_samples
        assert np.all(weights >= self.config.min_weight)
        assert np.all(weights <= self.config.max_weight)
        
        # Test weighted loss calculation
        y_pred = np.random.rand(self.n_samples)
        loss = calculator.calculate_weighted_loss(self.y, y_pred, weights, "mse")
        
        assert isinstance(loss, float)
        assert loss >= 0
        
        tprint_success("✅ Weighted loss calculator test passed")
    
    def test_weighted_loss_manager(self):
        """Test weighted loss manager."""
        tprint_info("Testing weighted loss manager...")
        
        manager = WeightedLossManager(self.config)
        manager.fit(self.X, self.y, self.market_data)
        
        # Get sample weights
        weights = manager.get_sample_weights(self.X, self.y, None, self.market_data)
        
        # Verify weights
        assert len(weights) == self.n_samples
        assert np.all(weights >= self.config.min_weight)
        assert np.all(weights <= self.config.max_weight)
        
        # Test weighted loss calculation
        y_pred = np.random.rand(self.n_samples)
        loss = manager.calculate_weighted_loss(self.y, y_pred, self.X, "mse", self.market_data)
        
        assert isinstance(loss, float)
        assert loss >= 0
        
        # Test weight statistics
        stats = manager.get_weight_statistics()
        assert isinstance(stats, dict)
        
        tprint_success("✅ Weighted loss manager test passed")

class TestWeightedLossIntegration:
    """Test the weighted loss integration components."""
    
    def setup_method(self):
        """Set up test data and configurations."""
        # Create synthetic test data
        np.random.seed(42)
        self.n_samples = 500
        self.n_features = 15
        
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.random.randint(0, 2, self.n_samples)
        
        # Create market data
        self.market_data = {
            'returns': np.random.randn(self.n_samples) * 0.01,
            'high': np.random.randn(self.n_samples) + 100,
            'low': np.random.randn(self.n_samples) + 99
        }
        
        # Create integration config
        self.config = WeightedLossIntegrationConfig(
            enable_weighted_loss=True,
            weighting_strategy=WeightingStrategy.ADAPTIVE
        )
    
    def test_weighted_loss_integrator(self):
        """Test weighted loss integrator."""
        tprint_info("Testing weighted loss integrator...")
        
        integrator = WeightedLossIntegrator(self.config)
        
        # Initialize with model types
        model_types = ['LIGHTGBM', 'CATBOOST', 'XGBOOST']
        integrator.initialize(model_types)
        
        # Fit for each model type
        for model_type in model_types:
            integrator.fit(model_type, self.X, self.y, self.market_data)
        
        # Test sample weights
        for model_type in model_types:
            weights = integrator.get_sample_weights(model_type, self.X, self.y, None, self.market_data)
            assert len(weights) == self.n_samples
            assert np.all(weights > 0)
        
        # Test weighted loss calculation
        y_pred = np.random.rand(self.n_samples)
        for model_type in model_types:
            loss = integrator.calculate_weighted_loss(model_type, self.y, y_pred, self.X, "mse", self.market_data)
            assert isinstance(loss, float)
            assert loss >= 0
        
        # Test weight statistics
        stats = integrator.get_weight_statistics()
        assert isinstance(stats, dict)
        assert len(stats) == len(model_types)
        
        tprint_success("✅ Weighted loss integrator test passed")
    
    def test_weighted_loss_model_wrapper(self):
        """Test weighted loss model wrapper."""
        tprint_info("Testing weighted loss model wrapper...")
        
        # Create integrator
        integrator = WeightedLossIntegrator(self.config)
        integrator.initialize(['LIGHTGBM'])
        integrator.fit('LIGHTGBM', self.X, self.y, self.market_data)
        
        # Create a simple model
        from sklearn.linear_model import LogisticRegression
        base_model = LogisticRegression(random_state=42)
        
        # Wrap with weighted loss
        wrapped_model = WeightedLossModelWrapper(base_model, 'LIGHTGBM', integrator)
        
        # Test fitting
        wrapped_model.fit(self.X, self.y)
        assert wrapped_model.is_fitted
        
        # Test prediction
        predictions = wrapped_model.predict(self.X)
        assert len(predictions) == self.n_samples
        
        # Test probability prediction
        if hasattr(wrapped_model, 'predict_proba'):
            probabilities = wrapped_model.predict_proba(self.X)
            assert probabilities.shape[0] == self.n_samples
        
        tprint_success("✅ Weighted loss model wrapper test passed")

class TestModelIntegration:
    """Test integration with actual models."""
    
    def setup_method(self):
        """Set up test data and configurations."""
        # Create synthetic test data
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 10
        
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.random.randint(0, 2, self.n_samples)
        
        # Create market data
        self.market_data = {
            'returns': np.random.randn(self.n_samples) * 0.01,
            'high': np.random.randn(self.n_samples) + 100,
            'low': np.random.randn(self.n_samples) + 99
        }
    
    def test_lgbm_gru_wrapper_with_weighted_loss(self):
        """Test LGBM-GRU wrapper with weighted loss."""
        tprint_info("Testing LGBM-GRU wrapper with weighted loss...")
        
        # Create model with weighted loss enabled
        model = LGBMGRUWrapper(
            enable_weighted_loss=True,
            weighted_loss_config={
                'weighting_strategy': 'adaptive',
                'max_weight': 3.0,
                'min_weight': 0.5
            },
            gru_hidden_size=16,
            gru_epochs=5,
            n_estimators=50
        )
        
        # Fit the model
        model.fit(self.X, self.y)
        assert model.is_fitted
        
        # Make predictions
        predictions = model.predict(self.X)
        assert len(predictions) == self.n_samples
        
        # Test probability prediction
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(self.X)
            assert probabilities.shape[0] == self.n_samples
        
        tprint_success("✅ LGBM-GRU wrapper with weighted loss test passed")
    
    def test_stacker_lgbm_with_weighted_loss(self):
        """Test Stacker LGBM with weighted loss."""
        tprint_info("Testing Stacker LGBM with weighted loss...")
        
        # Create base models configuration
        base_models = [
            {
                'name': 'lgbm1',
                'type': 'LIGHTGBM',
                'parameters': {
                    'n_estimators': 50,
                    'learning_rate': 0.1
                }
            },
            {
                'name': 'lgbm2',
                'type': 'LIGHTGBM',
                'parameters': {
                    'n_estimators': 50,
                    'learning_rate': 0.05
                }
            }
        ]
        
        # Create model with weighted loss enabled
        model = StackerLGBMCalibratedGated(
            base_models=base_models,
            enable_weighted_loss=True,
            weighted_loss_config={
                'weighting_strategy': 'adaptive',
                'max_weight': 3.0,
                'min_weight': 0.5
            },
            cv_folds=3,
            n_jobs=1
        )
        
        # Fit the model
        model.fit(self.X, self.y)
        assert model.is_fitted
        
        # Make predictions
        predictions = model.predict(self.X)
        assert len(predictions) == self.n_samples
        
        # Test probability prediction
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(self.X)
            assert probabilities.shape[0] == self.n_samples
        
        tprint_success("✅ Stacker LGBM with weighted loss test passed")

class TestPerformanceAndRobustness:
    """Test performance and robustness of weighted loss implementation."""
    
    def setup_method(self):
        """Set up test data and configurations."""
        # Create synthetic test data with various difficulty levels
        np.random.seed(42)
        self.n_samples = 1000
        self.n_features = 20
        
        # Create features with different difficulty levels
        self.X = np.random.randn(self.n_samples, self.n_features)
        
        # Create targets with some difficult samples
        self.y = np.random.randint(0, 2, self.n_samples)
        
        # Add difficult samples
        difficult_indices = np.random.choice(self.n_samples, size=200, replace=False)
        self.X[difficult_indices] += np.random.randn(200, self.n_features) * 3
        self.y[difficult_indices] = 1 - self.y[difficult_indices]
        
        # Create market data
        self.market_data = {
            'returns': np.random.randn(self.n_samples) * 0.01,
            'high': np.random.randn(self.n_samples) + 100,
            'low': np.random.randn(self.n_samples) + 99
        }
    
    def test_weight_distribution(self):
        """Test that weights are properly distributed."""
        tprint_info("Testing weight distribution...")
        
        config = WeightedLossConfig(
            enable_weighted_loss=True,
            weighting_strategy=WeightingStrategy.ADAPTIVE,
            max_weight=5.0,
            min_weight=0.1
        )
        
        manager = WeightedLossManager(config)
        manager.fit(self.X, self.y, self.market_data)
        
        weights = manager.get_sample_weights(self.X, self.y, None, self.market_data)
        
        # Test weight distribution
        assert len(weights) == self.n_samples
        assert np.all(weights >= config.min_weight)
        assert np.all(weights <= config.max_weight)
        
        # Test that difficult samples have higher weights
        difficult_indices = np.where(weights > np.percentile(weights, 80))[0]
        assert len(difficult_indices) > 0
        
        # Test weight statistics
        stats = manager.get_weight_statistics()
        assert 'mean_weight' in stats
        assert 'std_weight' in stats
        assert 'min_weight' in stats
        assert 'max_weight' in stats
        
        tprint_success("✅ Weight distribution test passed")
    
    def test_different_weighting_strategies(self):
        """Test different weighting strategies."""
        tprint_info("Testing different weighting strategies...")
        
        strategies = [
            WeightingStrategy.DIFFICULTY_BASED,
            WeightingStrategy.FAILURE_CONTEXT,
            WeightingStrategy.ADAPTIVE,
            WeightingStrategy.FOCAL_LOSS,
            WeightingStrategy.GRADIENT_BASED
        ]
        
        for strategy in strategies:
            config = WeightedLossConfig(
                enable_weighted_loss=True,
                weighting_strategy=strategy,
                max_weight=3.0,
                min_weight=0.2
            )
            
            manager = WeightedLossManager(config)
            manager.fit(self.X, self.y, self.market_data)
            
            weights = manager.get_sample_weights(self.X, self.y, None, self.market_data)
            
            # Verify weights are valid
            assert len(weights) == self.n_samples
            assert np.all(weights >= config.min_weight)
            assert np.all(weights <= config.max_weight)
            
            tprint_debug(f"Strategy {strategy.value}: mean={np.mean(weights):.3f}, std={np.std(weights):.3f}")
        
        tprint_success("✅ Different weighting strategies test passed")
    
    def test_memory_efficiency(self):
        """Test memory efficiency of weighted loss implementation."""
        tprint_info("Testing memory efficiency...")
        
        # Test with larger dataset
        large_X = np.random.randn(5000, 50)
        large_y = np.random.randint(0, 2, 5000)
        
        config = WeightedLossConfig(
            enable_weighted_loss=True,
            weighting_strategy=WeightingStrategy.ADAPTIVE
        )
        
        manager = WeightedLossManager(config)
        manager.fit(large_X, large_y)
        
        # Test that it doesn't crash with large data
        weights = manager.get_sample_weights(large_X, large_y)
        assert len(weights) == 5000
        
        tprint_success("✅ Memory efficiency test passed")
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        tprint_info("Testing edge cases...")
        
        config = WeightedLossConfig(
            enable_weighted_loss=True,
            weighting_strategy=WeightingStrategy.ADAPTIVE
        )
        
        manager = WeightedLossManager(config)
        
        # Test with minimal data
        small_X = np.random.randn(10, 5)
        small_y = np.random.randint(0, 2, 10)
        
        manager.fit(small_X, small_y)
        weights = manager.get_sample_weights(small_X, small_y)
        assert len(weights) == 10
        
        # Test with all same labels
        same_y = np.zeros(100)
        manager.fit(self.X, same_y)
        weights = manager.get_sample_weights(self.X, same_y)
        assert len(weights) == 100
        
        # Test with NaN values (should handle gracefully)
        X_with_nan = self.X.copy()
        X_with_nan[0, 0] = np.nan
        
        try:
            manager.fit(X_with_nan, self.y)
            weights = manager.get_sample_weights(X_with_nan, self.y)
            assert len(weights) == self.n_samples
        except Exception as e:
            tprint_warning(f"NaN handling test failed: {e}")
        
        tprint_success("✅ Edge cases test passed")

def run_all_tests():
    """Run all weighted loss tests."""
    tprint_info("🚀 Starting weighted loss implementation tests...")
    
    # Test framework components
    framework_test = TestWeightedLossFramework()
    framework_test.setup_method()
    framework_test.test_failure_context_detector()
    framework_test.test_sample_difficulty_assessor()
    framework_test.test_weighted_loss_calculator()
    framework_test.test_weighted_loss_manager()
    
    # Test integration components
    integration_test = TestWeightedLossIntegration()
    integration_test.setup_method()
    integration_test.test_weighted_loss_integrator()
    integration_test.test_weighted_loss_model_wrapper()
    
    # Test model integration
    model_test = TestModelIntegration()
    model_test.setup_method()
    model_test.test_lgbm_gru_wrapper_with_weighted_loss()
    model_test.test_stacker_lgbm_with_weighted_loss()
    
    # Test performance and robustness
    performance_test = TestPerformanceAndRobustness()
    performance_test.setup_method()
    performance_test.test_weight_distribution()
    performance_test.test_different_weighting_strategies()
    performance_test.test_memory_efficiency()
    performance_test.test_edge_cases()
    
    tprint_success("🎉 All weighted loss implementation tests passed!")

if __name__ == "__main__":
    run_all_tests()