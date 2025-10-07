#!/usr/bin/env python3
"""
Test script for Negative Learning Plugin

This script tests the negative learning plugin implementation
with synthetic ETHUSDT data to verify functionality.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_synthetic_ethusdt_data(n_days=30):
    """Create synthetic ETHUSDT data for testing"""
    np.random.seed(42)
    
    # Generate minute-level data
    n_periods = n_days * 24 * 60
    returns = np.random.normal(0, 0.001, n_periods)  # 0.1% per minute volatility
    prices = 3000 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_periods, freq='1min'),
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_periods))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_periods))),
        'close': prices,
        'volume': np.random.lognormal(8, 1, n_periods)
    })
    
    data.set_index('timestamp', inplace=True)
    return data

def create_analyst_features(data):
    """Create Analyst (1h) features"""
    # Resample to 1h
    analyst_data = data.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    features = analyst_data.copy()
    
    # HTF parent features
    features['trend_strength'] = analyst_data['close'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
    features['volatility_regime'] = analyst_data['close'].rolling(20).std()
    features['volume_profile'] = analyst_data['volume'].rolling(20).mean()
    features['momentum_htf'] = analyst_data['close'].pct_change(20)
    
    return features

def create_tactician_features(data):
    """Create Tactician (15m) features"""
    # Resample to 15m
    tactician_data = data.resample('15T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    features = tactician_data.copy()
    
    # Fast features
    features['momentum_5m'] = tactician_data['close'].pct_change(5)
    features['momentum_15m'] = tactician_data['close'].pct_change(15)
    features['vwap'] = (tactician_data['high'] + tactician_data['low'] + tactician_data['close']) / 3
    features['vwap_distance'] = (tactician_data['close'] - features['vwap']) / features['vwap']
    
    return features

def test_negative_learning_plugin():
    """Test the core negative learning plugin"""
    print("🧪 Testing Negative Learning Plugin...")
    
    try:
        from src.feature_generation.categories.negative_learning import (
            NegativeLearningPlugin, 
            NegativeLearningConfig
        )
        
        # Create test data
        data = create_synthetic_ethusdt_data(7)  # 1 week of data
        analyst_features = create_analyst_features(data)
        analyst_target = analyst_features['close'].pct_change(4).shift(-4)  # 4h forward returns
        
        # Initialize plugin
        config = NegativeLearningConfig(
            max_negative_features=4,
            enable_gated_twins=True,
            enable_exception_interactions=True
        )
        
        plugin = NegativeLearningPlugin(config)
        
        # Fit and transform
        train_data = analyst_features.iloc[:100]  # First 100 hours
        plugin.fit(train_data, analyst_target.iloc[:100], ['trend_strength', 'volatility_regime'])
        
        enhanced_features = plugin.transform(analyst_features)
        
        # Check results
        original_features = analyst_features.columns.tolist()
        negative_features = [col for col in enhanced_features.columns if col not in original_features]
        
        print(f"✅ Plugin test passed:")
        print(f"   - Original features: {len(original_features)}")
        print(f"   - Enhanced features: {len(enhanced_features.columns)}")
        print(f"   - Negative features: {len(negative_features)}")
        print(f"   - Negative features: {negative_features}")
        
        return True
        
    except Exception as e:
        print(f"❌ Plugin test failed: {e}")
        return False

def test_pipeline_integration():
    """Test the pipeline integration"""
    print("\n🧪 Testing Pipeline Integration...")
    
    try:
        from src.feature_generation.categories.negative_learning_pipeline_integration import (
            create_negative_learning_integrator
        )
        
        # Create test data
        data = create_synthetic_ethusdt_data(14)  # 2 weeks of data
        analyst_features = create_analyst_features(data)
        tactician_features = create_tactician_features(data)
        
        analyst_target = analyst_features['close'].pct_change(4).shift(-4)
        tactician_target = tactician_features['close'].pct_change(4).shift(-4)
        
        # Create integrator
        integrator = create_negative_learning_integrator()
        
        # Initialize
        init_results = integrator.initialize_negative_learning(
            analyst_features=analyst_features,
            analyst_target=analyst_target,
            tactician_features=tactician_features,
            tactician_target=tactician_target
        )
        
        # Check initialization
        if not init_results['analyst']['status'] == 'success':
            print(f"❌ Analyst initialization failed: {init_results['analyst']}")
            return False
            
        if not init_results['tactician']['status'] == 'success':
            print(f"❌ Tactician initialization failed: {init_results['tactician']}")
            return False
        
        # Test feature enhancement
        enhanced_analyst, enhanced_tactician = integrator.get_enhanced_features(
            analyst_features, tactician_features
        )
        
        # Check results
        analyst_negative = [col for col in enhanced_analyst.columns if col not in analyst_features.columns]
        tactician_negative = [col for col in enhanced_tactician.columns if col not in tactician_features.columns]
        
        print(f"✅ Integration test passed:")
        print(f"   - Analyst enhanced: {analyst_features.shape[1]} -> {enhanced_analyst.shape[1]}")
        print(f"   - Tactician enhanced: {tactician_features.shape[1]} -> {enhanced_tactician.shape[1]}")
        print(f"   - Analyst negative features: {len(analyst_negative)}")
        print(f"   - Tactician negative features: {len(tactician_negative)}")
        
        # Test model configurations
        model_configs = integrator.get_model_configs()
        print(f"   - Model configs generated: {len(model_configs)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_examples():
    """Test the ETHUSDT examples"""
    print("\n🧪 Testing ETHUSDT Examples...")
    
    try:
        from src.feature_generation.categories.negative_learning_examples import (
            ETHUSDTNegativeLearningExamples
        )
        
        examples = ETHUSDTNegativeLearningExamples()
        
        # Test momentum × high volatility example
        result1 = examples.example_1_momentum_high_volatility()
        print(f"✅ Example 1 (Momentum × High Vol): {result1['description']}")
        
        # Test VWAP × wide spread example
        result2 = examples.example_2_vwap_widespread()
        print(f"✅ Example 2 (VWAP × Wide Spread): {result2['description']}")
        
        # Test RSI × chop example
        result3 = examples.example_3_rsi_chop()
        print(f"✅ Example 3 (RSI × Chop): {result3['description']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Examples test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Negative Learning Plugin Tests")
    print("=" * 60)
    
    tests = [
        ("Core Plugin", test_negative_learning_plugin),
        ("Pipeline Integration", test_pipeline_integration),
        ("ETHUSDT Examples", test_examples)
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
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Negative Learning Plugin is ready to use.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)