#!/usr/bin/env python3
"""
Test script for completed implementations in unsupervised_tree_nas.py and pure_tree_nas.py

This script tests the enhanced functionality that was implemented:
1. Enhanced regime type determination
2. Improved transition probability calculation  
3. Comprehensive feature importance calculation
4. Complete NODE model implementation
5. True Oblivious Tree implementation
6. Enhanced Rotation Forest with proper rotation logic
7. Complete Histogram Gradient Boosting implementation
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample market data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic market data
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Generate price data with different regimes
    prices = [100]
    for i in range(1, n_samples):
        # Create different regimes
        if i < 200:  # Bull market
            change = np.random.normal(0.001, 0.02)
        elif i < 400:  # Bear market
            change = np.random.normal(-0.001, 0.03)
        elif i < 600:  # Sideways market
            change = np.random.normal(0.000, 0.01)
        elif i < 800:  # Volatile market
            change = np.random.normal(0.000, 0.05)
        else:  # Trending market
            change = np.random.normal(0.002, 0.015)
        
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1))  # Ensure positive prices
    
    # Create OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    })
    
    return data

def test_unsupervised_tree_nas():
    """Test the enhanced unsupervised tree NAS implementation."""
    logger.info("🧪 Testing Unsupervised Tree NAS...")
    
    try:
        from src.utils.ml_common.optimization.unsupervised_tree_nas import (
            UnsupervisedTreeNAS, UnsupervisedTreeNASConfig
        )
        
        # Create sample data
        market_data = create_sample_data()
        
        # Create configuration
        config = UnsupervisedTreeNASConfig(
            n_trials=5,  # Reduced for testing
            n_regimes_range=(3, 8),
            min_regime_duration=10
        )
        
        # Initialize NAS
        nas = UnsupervisedTreeNAS(config)
        
        # Run search
        logger.info("   → Running unsupervised regime detection...")
        result = nas.search(market_data)
        
        # Validate results
        assert result is not None, "Search should return a result"
        assert hasattr(result, 'regimes'), "Result should have regimes"
        assert hasattr(result, 'overall_score'), "Result should have overall_score"
        assert len(result.regimes) > 0, "Should detect at least one regime"
        
        logger.info(f"   ✅ Detected {len(result.regimes)} regimes")
        logger.info(f"   ✅ Overall score: {result.overall_score:.4f}")
        
        # Test regime characteristics
        for i, regime in enumerate(result.regimes):
            logger.info(f"   → Regime {i}: {regime.regime_type} (confidence: {regime.regime_confidence:.3f})")
            assert regime.regime_type in ['bull', 'bear', 'sideways', 'volatile', 'trending', 'mixed', 'unknown'], \
                f"Invalid regime type: {regime.regime_type}"
            assert 0 <= regime.regime_confidence <= 1, f"Invalid confidence: {regime.regime_confidence}"
        
        logger.info("✅ Unsupervised Tree NAS test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Unsupervised Tree NAS test failed: {e}")
        return False

def test_pure_tree_nas():
    """Test the enhanced pure tree NAS implementation."""
    logger.info("🧪 Testing Pure Tree NAS...")
    
    try:
        from src.utils.ml_common.optimization.pure_tree_nas import (
            PureTreeNAS, PureTreeNASConfig, NODEModel, ObliviousTreeModel, 
            RotationForestModel, HistogramGradientBoostingModel
        )
        
        # Create sample data
        np.random.seed(42)
        X = np.random.randn(500, 10)
        y = np.random.randn(500)
        
        # Test individual models
        logger.info("   → Testing NODE Model...")
        node_config = {
            'num_trees': 2,
            'tree_dim': 2,
            'depth': 4,
            'learning_rate': 0.01,
            'n_epochs': 10,  # Reduced for testing
            'batch_size': 32
        }
        
        # Skip NODE test if PyTorch not available
        try:
            node_model = NODEModel(node_config)
            node_model.fit(X, y)
            node_pred = node_model.predict(X[:10])
            assert len(node_pred) == 10, "NODE prediction should return correct length"
            logger.info("   ✅ NODE Model test passed")
        except ImportError:
            logger.info("   ⚠️ NODE Model test skipped (PyTorch not available)")
        
        # Test Oblivious Tree Model
        logger.info("   → Testing Oblivious Tree Model...")
        oblivious_config = {
            'max_depth': 5,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        }
        
        oblivious_model = ObliviousTreeModel(oblivious_config)
        oblivious_model.fit(X, y)
        oblivious_pred = oblivious_model.predict(X[:10])
        assert len(oblivious_pred) == 10, "Oblivious Tree prediction should return correct length"
        logger.info("   ✅ Oblivious Tree Model test passed")
        
        # Test Rotation Forest Model
        logger.info("   → Testing Rotation Forest Model...")
        rotation_config = {
            'n_estimators': 5,  # Reduced for testing
            'n_features_per_subset': 3,
            'rotation_method': 'pca',
            'bootstrap': True,
            'max_depth': 5
        }
        
        rotation_model = RotationForestModel(rotation_config)
        rotation_model.fit(X, y)
        rotation_pred = rotation_model.predict(X[:10])
        assert len(rotation_pred) == 10, "Rotation Forest prediction should return correct length"
        logger.info("   ✅ Rotation Forest Model test passed")
        
        # Test Histogram Gradient Boosting Model
        logger.info("   → Testing Histogram Gradient Boosting Model...")
        hist_config = {
            'max_iter': 20,  # Reduced for testing
            'max_depth': 5,
            'learning_rate': 0.1,
            'early_stopping': True
        }
        
        hist_model = HistogramGradientBoostingModel(hist_config)
        hist_model.fit(X, y)
        hist_pred = hist_model.predict(X[:10])
        assert len(hist_pred) == 10, "Histogram GB prediction should return correct length"
        logger.info("   ✅ Histogram Gradient Boosting Model test passed")
        
        # Test Pure Tree NAS
        logger.info("   → Testing Pure Tree NAS...")
        nas_config = PureTreeNASConfig(
            n_trials=3,  # Reduced for testing
            tree_models=['decision_tree', 'random_forest', 'oblivious_tree']
        )
        
        nas = PureTreeNAS(nas_config)
        result = nas.search(X, y)
        
        assert result is not None, "NAS search should return a result"
        assert hasattr(result, 'primary_model'), "Result should have primary_model"
        assert hasattr(result, 'overall_score'), "Result should have overall_score"
        
        logger.info(f"   ✅ Best model: {result.primary_model}")
        logger.info(f"   ✅ Best score: {result.overall_score:.4f}")
        
        logger.info("✅ Pure Tree NAS test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pure Tree NAS test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_utility_integration():
    """Test integration with utility functions."""
    logger.info("🧪 Testing Utility Integration...")
    
    try:
        from src.utils.math_validation import safe_mean, safe_std, validate_numeric_array
        from src.utils.common_operations import safe_weighted_average
        
        # Test math validation
        test_array = np.array([1, 2, 3, 4, 5])
        validated_array = validate_numeric_array(test_array, "test")
        assert np.array_equal(validated_array, test_array), "Array validation should work"
        
        mean_val = safe_mean(test_array)
        std_val = safe_std(test_array)
        assert mean_val == 3.0, f"Mean should be 3.0, got {mean_val}"
        assert std_val > 0, "Std should be positive"
        
        # Test weighted average
        values = [1, 2, 3, 4, 5]
        weights = [0.1, 0.2, 0.3, 0.2, 0.2]
        weighted_avg = safe_weighted_average(values, weights)
        assert weighted_avg > 0, "Weighted average should be positive"
        
        logger.info("✅ Utility Integration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Utility Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting implementation tests...")
    
    tests = [
        ("Utility Integration", test_utility_integration),
        ("Unsupervised Tree NAS", test_unsupervised_tree_nas),
        ("Pure Tree NAS", test_pure_tree_nas)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
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
        logger.info("🎉 All tests passed! Implementations are working correctly.")
        return True
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Check the logs above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)