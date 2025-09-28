#!/usr/bin/env python3
"""
Test Script for PatchTST Integration with Tree Models

This script tests the comprehensive PatchTST integration, including:
1. Automatic PatchTST wrapping of tree models
2. PatchTST element caching and reuse
3. Memory optimization
4. Performance improvements
"""

import logging
import numpy as np
import pandas as pd
import time
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_data(n_samples: int = 1000, n_features: int = 20) -> pd.DataFrame:
    """Create sample market data for testing."""
    np.random.seed(42)

    # Create base price data
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='1min')
    base_price = 100.0

    # Generate price movements
    price_changes = np.random.normal(0, 0.001, n_samples)
    prices = []
    current_price = base_price

    for change in price_changes:
        current_price *= (1 + change)
        prices.append(current_price)

    # Create OHLCV data
    high_prices = prices * (1 + np.random.uniform(0, 0.002, n_samples))
    low_prices = prices * (1 - np.random.uniform(0, 0.002, n_samples))
    open_prices = prices * (1 + np.random.uniform(-0.001, 0.001, n_samples))
    volumes = np.random.lognormal(10, 1, n_samples)

    # Create feature data
    features_data = {}
    for i in range(n_features):
        features_data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)

    # Combine into DataFrame
    market_data = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': prices,
        'volume': volumes,
        **features_data
    })

    return market_data

def test_clvsa_automatic_wrapping():
    """Test automatic CLVSA wrapping of tree models."""
    logger.info("🧪 Testing automatic CLVSA wrapping of tree models...")

    from src.utils.ml_common.models.model_factory import EnhancedModelFactory, ModelConfig, ModelType
    from src.utils.ml_common.models.tree_clvsa_wrapper import TreePatchTSTWrapper

    # Create model factory
    factory = EnhancedModelFactory()

    # Test Random Forest wrapping
    config = ModelConfig(
        model_type=ModelType.RANDOM_FOREST,
        model_name="test_rf",
        model_params={
            'use_patchtst': True,  # Should automatically enable PatchTST
            'n_estimators': 50,
            'max_depth': 5
        }
    )

    model = factory.create_model(config)

    # Verify it's wrapped with CLVSA
    assert isinstance(model, TreePatchTSTWrapper), "Random Forest should be automatically wrapped with PatchTST"
    assert hasattr(model, 'patch_model'), "PatchTST wrapper should have patch model"
    assert model.config.lookback > 0, "PatchTST enhancement should be enabled by default"

    logger.info("✅ Random Forest automatic PatchTST wrapping: PASSED")

    # Test XGBoost wrapping
    config = ModelConfig(
        model_type=ModelType.XGBOOST,
        model_name="test_xgb",
        model_params={
            'use_patchtst': True,
            'n_estimators': 50,
            'max_depth': 5
        }
    )

    model = factory.create_model(config)

    assert isinstance(model, TreePatchTSTWrapper), "XGBoost should be automatically wrapped with PatchTST"
    assert hasattr(model, 'patch_model'), "PatchTST wrapper should have patch model"

    logger.info("✅ XGBoost automatic PatchTST wrapping: PASSED")

    # Test disabling CLVSA
    config = ModelConfig(
        model_type=ModelType.LIGHTGBM,
        model_name="test_lgb_no_clvsa",
        model_params={
            'use_patchtst': False,  # Explicitly disable
            'n_estimators': 50
        }
    )

    model = factory.create_model(config)

    # Should be the base model, not wrapped
    from lightgbm import LGBMRegressor
    assert isinstance(model, LGBMRegressor), "LightGBM should not be wrapped when use_patchtst=False"
    assert not isinstance(model, TreePatchTSTWrapper), "LightGBM should not be PatchTST wrapped when disabled"

    logger.info("✅ PatchTST disable functionality: PASSED")

    return True

def test_patchtst_caching():
    """Test PatchTST element caching system."""
    logger.info("🧪 Testing PatchTST element caching system...")

    from src.utils.ml_common.models.cvlsa_cache import get_global_clvsa_cache, CLVSACacheConfig

    # Create cache with test configuration
    cache_config = CLVSACacheConfig(
        max_cache_size=10,
        max_memory_mb=50.0,
        ttl_seconds=300,
        enable_persistence=False  # Disable for testing
    )

    cache_manager = get_global_clvsa_cache(cache_config)

    # Create test data
    market_data = create_sample_data(100, 10)

    # Create feature configuration
    feature_config = {
        'input_dim': market_data.shape[1],
        'output_dim': 4,
        'seq_length': len(market_data),
        'cross_view_attention': True,
        'use_multi_scale_attention': True,
        'memory_efficient': True,
        'use_m1_gpu': False  # Disable GPU for testing
    }

    # Test cache storage and retrieval
    import torch
    features = {
        'price': torch.randn(100, 50),
        'volume': torch.randn(100, 30),
        'trend': torch.randn(100, 40),
        'momentum': torch.randn(100, 25)
    }
    predictions = torch.randn(100, 4)
    attention_weights = {
        'cross_view': np.random.randn(8, 100, 100),
        'temporal': np.random.randn(8, 100, 100)
    }

    # Store in cache
    cache_key = cache_manager.store(market_data, feature_config, features, predictions, attention_weights)

    assert cache_key is not None, "Cache key should not be None"
    assert cache_key in cache_manager.cache, "Entry should be in cache"

    # Retrieve from cache
    retrieved = cache_manager.retrieve(market_data, feature_config)

    assert retrieved is not None, "Should retrieve cached data"
    ret_features, ret_predictions, ret_attention = retrieved

    # Verify data integrity
    assert len(ret_features) == len(features), "Feature keys should match"
    assert ret_predictions.shape == predictions.shape, "Predictions shape should match"
    assert len(ret_attention) == len(attention_weights), "Attention weights keys should match"

    logger.info("✅ CLVSA caching storage and retrieval: PASSED")

    # Test cache miss
    different_config = feature_config.copy()
    different_config['output_dim'] = 8  # Different config

    miss_result = cache_manager.retrieve(market_data, different_config)
    assert miss_result is None, "Should return None for cache miss"

    logger.info("✅ CLVSA cache miss handling: PASSED")

    # Test cache statistics
    stats = cache_manager.get_stats()
    assert stats['cache_size'] == 1, "Cache should contain 1 entry"
    assert stats['hit_count'] == 1, "Should have 1 cache hit"
    assert stats['miss_count'] == 1, "Should have 1 cache miss"

    logger.info("✅ CLVSA cache statistics: PASSED")

    return True

def test_memory_optimization():
    """Test memory optimization features."""
    logger.info("🧪 Testing memory optimization features...")

    from src.utils.ml_common.models.cvlsa_cache import CLVSACacheConfig, CLVSACacheManager
    import torch

    # Create cache with small memory limit to test optimization
    cache_config = CLVSACacheConfig(
        max_cache_size=3,
        max_memory_mb=10.0,  # Small limit to trigger optimization
        ttl_seconds=300,
        enable_persistence=False
    )

    cache_manager = CLVSACacheManager(cache_config)

    # Create multiple test entries to exceed memory limit
    market_data = create_sample_data(50, 5)

    base_config = {
        'input_dim': market_data.shape[1],
        'output_dim': 4,
        'seq_length': len(market_data),
        'cross_view_attention': True,
        'use_multi_scale_attention': False,
        'memory_efficient': True,
        'use_m1_gpu': False
    }

    # Store multiple entries to test memory management
    for i in range(5):
        config = base_config.copy()
        config['output_dim'] = 4 + i  # Make each config unique

        features = {
            'price': torch.randn(50, 20 + i),
            'volume': torch.randn(50, 15 + i)
        }
        predictions = torch.randn(50, 4 + i)
        attention_weights = {'cross_view': np.random.randn(4, 50, 50)}

        cache_manager.store(market_data, config, features, predictions, attention_weights)

    # Check that memory optimization kicked in
    stats = cache_manager.get_stats()
    assert stats['cache_size'] <= 3, "Cache size should be limited by max_cache_size"
    assert stats['eviction_count'] > 0, "Some entries should have been evicted"

    logger.info("✅ Memory optimization and eviction: PASSED")

    # Test memory pool functionality
    tensor1 = torch.randn(100, 50)
    tensor2 = torch.randn(100, 50)  # Same shape as tensor1

    # Optimize first tensor
    optimized1 = cache_manager._optimize_tensor_memory(tensor1, "test1")
    assert optimized1.shape == tensor1.shape, "Tensor shape should be preserved"

    # Optimize second tensor (should reuse from pool if possible)
    optimized2 = cache_manager._optimize_tensor_memory(tensor2, "test2")

    logger.info("✅ Memory pool optimization: PASSED")

    return True

def test_end_to_end_integration():
    """Test end-to-end CLVSA integration with training and prediction."""
    logger.info("🧪 Testing end-to-end CLVSA integration...")

    from src.utils.ml_common.models.model_factory import EnhancedModelFactory, ModelConfig, ModelType
    from src.utils.ml_common.models.tree_clvsa_wrapper import TreePatchTSTWrapper
    import torch

    # Create training data
    X = np.random.randn(200, 10)
    y = np.random.choice([0, 1], 200)  # Binary classification

    # Create model with CLVSA enhancement
    factory = EnhancedModelFactory()

    config = ModelConfig(
        model_type=ModelType.RANDOM_FOREST_CLASSIFIER,
        model_name="test_clvsa_rf",
        model_params={
            'use_patchtst': True,
            'n_estimators': 20,  # Small number for quick testing
            'max_depth': 3,
            'clvsa_config': {
                'attention_dim': 32,
                'fusion_method': 'attention',
                'memory_efficient': True
            }
        }
    )

    model = factory.create_model(config)

    # Verify model structure
    assert isinstance(model, TreeCLVSAWrapper), "Model should be CLVSA wrapped"
    assert model.config.enable_cvlsa_enhancement, "CLVSA should be enabled"
    assert model.is_classifier, "Should be a classifier"

    # Test training
    logger.info("Training CLVSA-enhanced model...")
    model.fit(X, y)

    assert model.is_fitted, "Model should be fitted"
    assert hasattr(model, 'training_metadata'), "Should have training metadata"
    assert 'cvlsa_enabled' in model.training_metadata, "Training metadata should include CVLSA info"

    # Test prediction
    predictions = model.predict(X[:10])
    assert len(predictions) == 10, "Should return predictions for all test samples"

    # Test probability prediction
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X[:10])
        assert probabilities.shape[0] == 10, "Should return probabilities for all test samples"

    logger.info("✅ End-to-end CLVSA integration: PASSED")

    # Test feature importance
    importance = model.get_feature_importance()
    assert 'cvlsa_attention' in importance or 'tree_importance' in importance, "Should have feature importance"

    logger.info("✅ Feature importance extraction: PASSED")

    return True

def test_performance_comparison():
    """Test performance comparison between CLVSA and non-CLVSA models."""
    logger.info("🧪 Testing performance comparison...")

    from src.utils.ml_common.models.model_factory import EnhancedModelFactory, ModelConfig, ModelType

    # Create larger dataset for meaningful comparison
    X = np.random.randn(500, 15)
    y = np.random.randn(500)  # Regression task

    factory = EnhancedModelFactory()

    # Test CLVSA-enhanced model
    clvsa_config = ModelConfig(
        model_type=ModelType.RANDOM_FOREST,
        model_name="clvsa_rf",
        model_params={
            'n_estimators': 30,
            'max_depth': 5
        }
    )

    # Test regular model
    regular_config = ModelConfig(
        model_type=ModelType.RANDOM_FOREST,
        model_name="regular_rf",
        model_params={
            'use_clvsa': False,
            'n_estimators': 30,
            'max_depth': 5
        }
    )

    # Train both models
    start_time = time.time()
    clvsa_model = factory.create_model(clvsa_config)
    clvsa_model.fit(X, y)
    clvsa_time = time.time() - start_time

    start_time = time.time()
    regular_model = factory.create_model(regular_config)
    regular_model.fit(X, y)
    regular_time = time.time() - start_time

    # Make predictions
    test_X = X[:50]

    start_time = time.time()
    clvsa_predictions = clvsa_model.predict(test_X)
    clvsa_pred_time = time.time() - start_time

    start_time = time.time()
    regular_predictions = regular_model.predict(test_X)
    regular_pred_time = time.time() - start_time

    logger.info(f"CLVSA Training Time: {clvsa_time:.3f}s")
    logger.info(f"Regular Training Time: {regular_time:.3f}s")
    logger.info(f"CLVSA Prediction Time: {clvsa_pred_time:.3f}s")
    logger.info(f"Regular Prediction Time: {regular_pred_time:.3f}s")

    # CLVSA should provide more sophisticated predictions
    assert len(clvsa_predictions) == len(regular_predictions), "Both should return same number of predictions"

    logger.info("✅ Performance comparison completed")

    return True

def main():
    """Run all tests."""
    logger.info("🚀 Starting comprehensive CLVSA integration tests...")

    try:
        # Run all tests
        tests = [
            test_clvsa_automatic_wrapping,
            test_clvsa_caching,
            test_memory_optimization,
            test_end_to_end_integration,
            test_performance_comparison
        ]

        passed = 0
        failed = 0

        for test in tests:
            try:
                if test():
                    passed += 1
                    logger.info(f"✅ {test.__name__}: PASSED")
                else:
                    failed += 1
                    logger.error(f"❌ {test.__name__}: FAILED")
            except Exception as e:
                failed += 1
                logger.error(f"❌ {test.__name__}: FAILED with exception: {e}")

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total Tests: {len(tests)}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success Rate: {passed/len(tests)*100:.1f}%")
        logger.info(f"{'='*60}")

        if failed == 0:
            logger.info("🎉 All tests passed! CLVSA integration is working correctly.")
            return True
        else:
            logger.error(f"❌ {failed} tests failed. Please check the implementation.")
            return False

    except Exception as e:
        logger.error(f"❌ Test suite failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)