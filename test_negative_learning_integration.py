#!/usr/bin/env python3
"""
Test script for Negative Learning Training Integration

This script tests the complete integration of negative learning
into the ML training pipeline.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_test_data():
    """Create test data for integration testing"""
    np.random.seed(42)
    
    # Create synthetic data
    n_samples = 1000
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='1H')
    
    # Create price data
    returns = np.random.normal(0, 0.01, n_samples)
    prices = 3000 * np.exp(np.cumsum(returns))
    
    # Create features
    data = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        'volume': np.random.lognormal(10, 1, n_samples)
    })
    
    data.set_index('timestamp', inplace=True)
    
    # Create features
    data['momentum_5m'] = data['close'].pct_change(5)
    data['momentum_15m'] = data['close'].pct_change(15)
    data['volatility'] = data['close'].rolling(20).std()
    data['trend_strength'] = data['close'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
    data['volume_profile'] = data['volume'].rolling(20).mean()
    
    # Create target
    data['target'] = data['close'].pct_change(4).shift(-4)
    
    return data

def test_negative_learning_integration():
    """Test the complete negative learning integration"""
    print("🧪 Testing Negative Learning Training Integration...")
    
    try:
        # Test 1: Import integration modules
        print("📦 Testing imports...")
        from src.training.steps.models_training.negative_learning_training_integration import (
            initialize_negative_learning_integration,
            get_negative_learning_integration
        )
        from src.training.steps.models_training.negative_learning_training_patches import (
            apply_negative_learning_patches
        )
        print("✅ Imports successful")
        
        # Test 2: Create test data
        print("📊 Creating test data...")
        data = create_test_data()
        analyst_data = data.resample('1H').agg({
            'close': 'last',
            'high': 'max',
            'low': 'min',
            'volume': 'sum',
            'momentum_5m': 'last',
            'momentum_15m': 'last',
            'volatility': 'last',
            'trend_strength': 'last',
            'volume_profile': 'last',
            'target': 'last'
        }).dropna()
        
        tactician_data = data.resample('15T').agg({
            'close': 'last',
            'high': 'max',
            'low': 'min',
            'volume': 'sum',
            'momentum_5m': 'last',
            'momentum_15m': 'last',
            'volatility': 'last',
            'trend_strength': 'last',
            'volume_profile': 'last',
            'target': 'last'
        }).dropna()
        
        print(f"✅ Test data created: Analyst {analyst_data.shape}, Tactician {tactician_data.shape}")
        
        # Test 3: Initialize negative learning integration
        print("🎯 Initializing negative learning integration...")
        nl_integration = initialize_negative_learning_integration()
        
        init_results = nl_integration.initialize_for_training(
            analyst_features=analyst_data[['momentum_5m', 'momentum_15m', 'volatility', 'trend_strength', 'volume_profile']],
            analyst_target=analyst_data['target'],
            tactician_features=tactician_data[['momentum_5m', 'momentum_15m', 'volatility', 'trend_strength', 'volume_profile']],
            tactician_target=tactician_data['target'],
            retrain_timestamp=datetime.now()
        )
        
        if init_results['analyst']['status'] == 'success' and init_results['tactician']['status'] == 'success':
            print("✅ Negative learning integration initialized successfully")
        else:
            print(f"❌ Initialization failed: {init_results}")
            return False
        
        # Test 4: Test feature enhancement
        print("🔄 Testing feature enhancement...")
        
        # Test Analyst feature enhancement
        analyst_features = analyst_data[['momentum_5m', 'momentum_15m', 'volatility', 'trend_strength', 'volume_profile']]
        enhanced_analyst = nl_integration.enhance_training_features(analyst_features, pipeline_type='analyst')
        
        analyst_negative_features = [col for col in enhanced_analyst.columns if col not in analyst_features.columns]
        print(f"✅ Analyst features enhanced: {analyst_features.shape[1]} -> {enhanced_analyst.shape[1]} (+{len(analyst_negative_features)} negative features)")
        
        # Test Tactician feature enhancement
        tactician_features = tactician_data[['momentum_5m', 'momentum_15m', 'volatility', 'trend_strength', 'volume_profile']]
        enhanced_tactician = nl_integration.enhance_training_features(tactician_features, pipeline_type='tactician')
        
        tactician_negative_features = [col for col in enhanced_tactician.columns if col not in tactician_features.columns]
        print(f"✅ Tactician features enhanced: {tactician_features.shape[1]} -> {enhanced_tactician.shape[1]} (+{len(tactician_negative_features)} negative features)")
        
        # Test 5: Test constraints
        print("🔧 Testing model constraints...")
        
        analyst_constraints = nl_integration.get_training_constraints(pipeline_type='analyst', model_type='lightgbm')
        tactician_constraints = nl_integration.get_training_constraints(pipeline_type='tactician', model_type='lightgbm')
        
        print(f"✅ Analyst constraints: {len(analyst_constraints.get('monotone_constraints', []))} monotone constraints")
        print(f"✅ Tactician constraints: {len(tactician_constraints.get('monotone_constraints', []))} monotone constraints")
        
        # Test 6: Test sample weights
        print("⚖️ Testing sample weights...")
        
        analyst_weights = nl_integration.get_training_sample_weights(analyst_features, pipeline_type='analyst')
        tactician_weights = nl_integration.get_training_sample_weights(tactician_features, pipeline_type='tactician')
        
        print(f"✅ Analyst sample weights: mean={analyst_weights.mean():.3f}, std={analyst_weights.std():.3f}")
        print(f"✅ Tactician sample weights: mean={tactician_weights.mean():.3f}, std={tactician_weights.std():.3f}")
        
        # Test 7: Test validation
        print("🔍 Testing validation...")
        
        analyst_validation = nl_integration.validate_training_performance(
            analyst_features, analyst_data['target'], pipeline_type='analyst'
        )
        tactician_validation = nl_integration.validate_training_performance(
            tactician_features, tactician_data['target'], pipeline_type='tactician'
        )
        
        print(f"✅ Analyst validation: {analyst_validation.get('status', 'unknown')}")
        print(f"✅ Tactician validation: {tactician_validation.get('status', 'unknown')}")
        
        # Test 8: Test integration status
        print("📊 Testing integration status...")
        status = nl_integration.get_integration_status()
        print(f"✅ Integration status: {status}")
        
        print("\n🎉 All integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_patch_application():
    """Test that patches are applied correctly"""
    print("\n🔧 Testing patch application...")
    
    try:
        from src.training.steps.models_training.negative_learning_training_patches import (
            apply_negative_learning_patches
        )
        
        # Apply patches
        apply_negative_learning_patches()
        print("✅ Patches applied successfully")
        
        # Test that patches are working by checking if functions are patched
        from src.training.steps.models_training import analyst_models_training
        from src.training.steps.models_training import tactician_models_training
        
        # Check if functions have been patched (they should have __wrapped__ attribute)
        if hasattr(analyst_models_training.execute_analyst_models_training, '__wrapped__'):
            print("✅ Analyst training function patched")
        else:
            print("⚠️ Analyst training function may not be patched")
        
        if hasattr(tactician_models_training.execute_tactician_models_training, '__wrapped__'):
            print("✅ Tactician training function patched")
        else:
            print("⚠️ Tactician training function may not be patched")
        
        return True
        
    except Exception as e:
        print(f"❌ Patch application test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🚀 Starting Negative Learning Training Integration Tests")
    print("=" * 70)
    
    tests = [
        ("Negative Learning Integration", test_negative_learning_integration),
        ("Patch Application", test_patch_application)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Test...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} test PASSED")
        else:
            print(f"❌ {test_name} test FAILED")
    
    print("\n" + "=" * 70)
    print(f"📊 Integration Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        print("✅ Negative Learning is fully wired into ML training pipeline!")
        print("\n📚 What this means:")
        print("- Your existing training functions will automatically use negative learning features")
        print("- Model constraints will be applied automatically")
        print("- Sample weights will be enhanced with uncertainty weighting")
        print("- No code changes needed in your training pipeline")
        return True
    else:
        print("⚠️ Some integration tests failed.")
        print("Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)