#!/usr/bin/env python3
"""
Test script to verify SR features are properly generated and categorized.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_test_data():
    """Create test OHLCV data."""
    np.random.seed(42)
    n_periods = 1000
    
    # Generate realistic price data
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_periods)  # 2% daily volatility
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    # Create OHLCV data
    data = []
    for i, close in enumerate(prices):
        # Generate realistic OHLC from close
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = close * (1 + np.random.normal(0, 0.005))
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.index = pd.date_range(start='2024-01-01', periods=len(df), freq='1H')
    
    return df

def test_sr_feature_generation():
    """Test SR feature generation."""
    print("🧪 Testing SR feature generation...")
    
    # Create test data
    test_data = create_test_data()
    print(f"✅ Created test data with {len(test_data)} periods")
    
    # Test fallback SR feature generation
    try:
        from src.training.steps.step3_hmm_regime_discovery import _generate_fallback_sr_features
        
        sr_features = _generate_fallback_sr_features(test_data)
        
        if not sr_features.empty:
            print(f"✅ Generated {len(sr_features.columns)} SR features:")
            for col in sr_features.columns:
                print(f"  - {col}")
            
            # Check for key features
            expected_features = [
                'distance_to_support', 'distance_to_resistance',
                'normalized_distance_to_support', 'normalized_distance_to_resistance',
                'sr_proximity_score', 'support_strength_score', 'resistance_strength_score',
                'clarity_factor', 'directional_pressure', 'sr_score', 'delta_sr_score'
            ]
            
            missing_features = [f for f in expected_features if f not in sr_features.columns]
            if missing_features:
                print(f"⚠️ Missing features: {missing_features}")
            else:
                print("✅ All expected SR features generated")
            
            # Check for NaN values
            nan_counts = sr_features.isna().sum()
            if nan_counts.sum() > 0:
                print(f"⚠️ Found NaN values: {nan_counts[nan_counts > 0].to_dict()}")
            else:
                print("✅ No NaN values in SR features")
                
        else:
            print("❌ Failed to generate SR features")
            return False
            
    except Exception as e:
        print(f"❌ Error testing SR feature generation: {e}")
        return False
    
    return True

def test_sr_block_categorization():
    """Test SR block feature categorization."""
    print("\n🧪 Testing SR block feature categorization...")
    
    try:
        from src.training.steps.step3_hmm_regime_discovery import _assign_block, _select_block_features
        
        # Create test data with SR features
        test_data = create_test_data()
        sr_features = _generate_fallback_sr_features(test_data)
        
        if sr_features.empty:
            print("❌ No SR features to test categorization")
            return False
        
        # Test feature assignment
        print("🔍 Testing feature assignment to blocks:")
        for feature in sr_features.columns:
            block = _assign_block(feature)
            print(f"  {feature} -> {block}")
        
        # Test block selection
        print("\n🔍 Testing SR block feature selection:")
        sr_block_features = _select_block_features(sr_features, "support_resistance", 5)
        
        if not sr_block_features.empty:
            print(f"✅ Selected {len(sr_block_features.columns)} features for SR block:")
            for col in sr_block_features.columns:
                print(f"  - {col}")
        else:
            print("❌ No features selected for SR block")
            return False
            
    except Exception as e:
        print(f"❌ Error testing SR block categorization: {e}")
        return False
    
    return True

def test_enhanced_sr_features():
    """Test enhanced SR features from SRBreakoutPredictor."""
    print("\n🧪 Testing enhanced SR features from SRBreakoutPredictor...")
    
    try:
        from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
        
        # Create test data
        test_data = create_test_data()
        
        # Initialize SR predictor
        config = {
            "sr_breakout_predictor": {
                "enable_sr_breakout_tactics": True,
                "sr_proximity_threshold": 0.02,
                "breakout_confidence_threshold": 0.7,
                "enhanced_sr_detection": {
                    "enable_fractal_analysis": True,
                    "enable_volume_weighted_levels": True,
                    "enable_atr_based_activation": True
                }
            }
        }
        
        sr_predictor = SRBreakoutPredictor(config)
        
        # Test comprehensive SR feature calculation
        enhanced_features = sr_predictor.calculate_comprehensive_sr_features(test_data)
        
        if enhanced_features:
            print(f"✅ Generated {len(enhanced_features)} enhanced SR features:")
            for feature_name, feature_data in enhanced_features.items():
                if isinstance(feature_data, pd.Series):
                    print(f"  - {feature_name}: {len(feature_data)} values, mean={feature_data.mean():.4f}")
                else:
                    print(f"  - {feature_name}: {type(feature_data)}")
        else:
            print("❌ No enhanced SR features generated")
            return False
            
    except Exception as e:
        print(f"❌ Error testing enhanced SR features: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🚀 Starting SR feature tests...\n")
    
    tests = [
        ("SR Feature Generation", test_sr_feature_generation),
        ("SR Block Categorization", test_sr_block_categorization),
        ("Enhanced SR Features", test_enhanced_sr_features),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"📋 Running test: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"{'✅ PASSED' if result else '❌ FAILED'}: {test_name}\n")
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}\n")
            results.append((test_name, False))
    
    # Summary
    print("📊 Test Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status}: {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! SR features should work correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)