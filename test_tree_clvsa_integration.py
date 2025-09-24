#!/usr/bin/env python3
"""
Test Tree CLVSA Integration

This script tests the integration of CLVSA architecture with tree models
to ensure all tree models are properly wrapped by default.
"""

import numpy as np
import pandas as pd
import sys
import os
import logging
from typing import Dict, Any, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_data(n_samples: int = 1000, n_features: int = 20) -> tuple:
    """Create sample data for testing."""
    np.random.seed(42)
    
    # Create features
    X = np.random.randn(n_samples, n_features)
    
    # Create target with some relationship to features
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + X[:, 2] * 0.2 + 
         np.random.randn(n_samples) * 0.1)
    
    # Create regime labels
    regimes = np.random.choice(['high_vol', 'low_vol', 'trending', 'mean_reverting'], 
                              n_samples, p=[0.3, 0.2, 0.3, 0.2])
    
    return X, y, regimes

def test_tree_clvsa_wrapper():
    """Test the Tree CLVSA wrapper directly."""
    logger.info("🧪 Testing Tree CLVSA wrapper...")
    
    try:
        from src.training.steps.model_training.tree_clvsa_wrapper import (
            TreeCLVSAWrapper, TreeCLVSAConfig, create_tree_clvsa_wrapper
        )
        from sklearn.ensemble import RandomForestRegressor
        
        # Create sample data
        X, y, regimes = create_sample_data(500, 15)
        
        # Create base model
        base_model = RandomForestRegressor(n_estimators=50, random_state=42)
        
        # Create CLVSA wrapper
        config = TreeCLVSAConfig(
            attention_dim=32,
            use_temporal_attention=True,
            regime_aware=True,
            ensemble_attention=True
        )
        
        wrapper = TreeCLVSAWrapper(base_model, config)
        
        # Test fitting
        logger.info("   Fitting Tree CLVSA wrapper...")
        wrapper.fit(X, y, regimes=regimes)
        
        # Test prediction
        logger.info("   Testing predictions...")
        predictions = wrapper.predict(X[:100], regimes=regimes[:100])
        
        # Test attention weights
        attention_weights = wrapper.get_attention_weights()
        
        # Verify results
        assert len(predictions) == 100, f"Expected 100 predictions, got {len(predictions)}"
        assert 'feature_attention' in attention_weights, "Feature attention weights missing"
        assert attention_weights['feature_attention'] is not None, "Feature attention weights are None"
        
        logger.info("✅ Tree CLVSA wrapper test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Tree CLVSA wrapper test failed: {e}")
        return False

def test_model_factory_integration():
    """Test the model factory integration with CLVSA."""
    logger.info("🏭 Testing Model Factory CLVSA integration...")
    
    try:
        from src.utils.ml_common.models.model_factory import (
            EnhancedModelFactory, ModelConfig, ModelType
        )
        
        # Create model factory
        factory = EnhancedModelFactory()
        
        # Test different tree models
        tree_models = [
            ModelType.RANDOM_FOREST,
            ModelType.LIGHTGBM,
            ModelType.XGBOOST,
            ModelType.CATBOOST,
            ModelType.EXTRA_TREES,
            ModelType.HIST_GRADIENT_BOOSTING
        ]
        
        # Create sample data
        X, y, regimes = create_sample_data(300, 10)
        
        for model_type in tree_models:
            logger.info(f"   Testing {model_type.value}...")
            
            # Create model config
            config = ModelConfig(
                model_type=model_type,
                model_name=f"test_{model_type.value.lower()}",
                model_params={
                    'use_clvsa': True,  # Enable CLVSA by default
                    'attention_dim': 32,
                    'use_temporal_attention': True,
                    'regime_aware': True,
                    'ensemble_attention': True,
                    'memory_efficient': True
                }
            )
            
            # Create model
            model = factory.create_model(config)
            
            # Verify it's wrapped with CLVSA
            model_type_name = type(model).__name__
            assert 'TreeCLVSAWrapper' in model_type_name or 'CLVSAAttentionWrapper' in model_type_name, \
                f"Model {model_type.value} not wrapped with CLVSA: {model_type_name}"
            
            # Test fitting
            model.fit(X, y, regimes=regimes)
            
            # Test prediction
            predictions = model.predict(X[:50], regimes=regimes[:50])
            assert len(predictions) == 50, f"Expected 50 predictions, got {len(predictions)}"
            
            logger.info(f"   ✅ {model_type.value} CLVSA integration successful")
        
        logger.info("✅ Model Factory CLVSA integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model Factory CLVSA integration test failed: {e}")
        return False

def test_clvsa_disabled():
    """Test that CLVSA can be disabled."""
    logger.info("🚫 Testing CLVSA disabled option...")
    
    try:
        from src.utils.ml_common.models.model_factory import (
            EnhancedModelFactory, ModelConfig, ModelType
        )
        
        # Create model factory
        factory = EnhancedModelFactory()
        
        # Create model config with CLVSA disabled
        config = ModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            model_name="test_rf_no_clvsa",
            model_params={
                'use_clvsa': False,  # Disable CLVSA
                'n_estimators': 50
            }
        )
        
        # Create model
        model = factory.create_model(config)
        
        # Verify it's NOT wrapped with CLVSA
        model_type_name = type(model).__name__
        assert 'TreeCLVSAWrapper' not in model_type_name, \
            f"Model should not be wrapped with CLVSA: {model_type_name}"
        assert 'RandomForestRegressor' in model_type_name, \
            f"Expected RandomForestRegressor, got {model_type_name}"
        
        # Test that it still works
        X, y, _ = create_sample_data(100, 5)
        model.fit(X, y)
        predictions = model.predict(X[:10])
        assert len(predictions) == 10, f"Expected 10 predictions, got {len(predictions)}"
        
        logger.info("✅ CLVSA disabled test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ CLVSA disabled test failed: {e}")
        return False

def test_attention_weights():
    """Test that attention weights are properly computed."""
    logger.info("🎯 Testing attention weights computation...")
    
    try:
        from src.training.steps.model_training.tree_clvsa_wrapper import (
            TreeCLVSAWrapper, TreeCLVSAConfig
        )
        from sklearn.ensemble import RandomForestRegressor
        
        # Create sample data with clear feature importance
        np.random.seed(42)
        X = np.random.randn(200, 5)
        # Make first feature very important
        y = X[:, 0] * 2.0 + np.random.randn(200) * 0.1
        
        # Create model
        base_model = RandomForestRegressor(n_estimators=20, random_state=42)
        config = TreeCLVSAConfig(feature_selection_method='mutual_info')
        wrapper = TreeCLVSAWrapper(base_model, config)
        
        # Fit model
        wrapper.fit(X, y)
        
        # Get attention weights
        attention_weights = wrapper.get_attention_weights()
        feature_attention = attention_weights['feature_attention']
        
        # Verify attention weights
        assert feature_attention is not None, "Feature attention weights are None"
        assert len(feature_attention) == X.shape[1], f"Expected {X.shape[1]} attention weights, got {len(feature_attention)}"
        assert np.allclose(np.sum(feature_attention), 1.0, atol=1e-6), "Attention weights should sum to 1"
        
        # First feature should have highest attention (it's most important)
        assert feature_attention[0] > feature_attention[1], "First feature should have higher attention"
        
        logger.info("✅ Attention weights test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Attention weights test failed: {e}")
        return False

def test_performance_comparison():
    """Test that CLVSA-wrapped models perform at least as well as base models."""
    logger.info("📊 Testing performance comparison...")
    
    try:
        from src.training.steps.model_training.tree_clvsa_wrapper import (
            TreeCLVSAWrapper, TreeCLVSAConfig
        )
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score
        
        # Create sample data
        X, y, _ = create_sample_data(200, 10)
        
        # Split data
        split_idx = len(X) // 2
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Base model
        base_model = RandomForestRegressor(n_estimators=50, random_state=42)
        base_model.fit(X_train, y_train)
        base_predictions = base_model.predict(X_test)
        base_score = r2_score(y_test, base_predictions)
        
        # CLVSA-wrapped model
        config = TreeCLVSAConfig(attention_dim=16, memory_efficient=True)
        clvsa_model = TreeCLVSAWrapper(RandomForestRegressor(n_estimators=50, random_state=42), config)
        clvsa_model.fit(X_train, y_train)
        clvsa_predictions = clvsa_model.predict(X_test)
        clvsa_score = r2_score(y_test, clvsa_predictions)
        
        # CLVSA should perform at least as well (allowing for some variance)
        logger.info(f"   Base model R²: {base_score:.4f}")
        logger.info(f"   CLVSA model R²: {clvsa_score:.4f}")
        
        # Allow for some variance in performance
        assert clvsa_score >= base_score - 0.05, f"CLVSA model underperformed: {clvsa_score} < {base_score - 0.05}"
        
        logger.info("✅ Performance comparison test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance comparison test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting Tree CLVSA Integration Tests")
    
    tests = [
        ("Tree CLVSA Wrapper", test_tree_clvsa_wrapper),
        ("Model Factory Integration", test_model_factory_integration),
        ("CLVSA Disabled", test_clvsa_disabled),
        ("Attention Weights", test_attention_weights),
        ("Performance Comparison", test_performance_comparison)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Tree CLVSA integration is working correctly.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)