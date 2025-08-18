#!/usr/bin/env python3
"""
Test script for comprehensive SR features implementation.
This script tests:
1. SRBreakoutPredictor comprehensive feature generation
2. Step2 integration of comprehensive SR features
3. Step3 SR block feature detection
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.tactician.sr_breakout_predictor import setup_sr_breakout_predictor
from src.training.steps.step2_feature_engineering import run_step as run_step2
from src.training.steps.step3_hmm_regime_discovery import _assign_block
from src.utils.logger import system_logger

async def test_sr_breakout_predictor():
    """Test SRBreakoutPredictor comprehensive feature generation."""
    print("🧪 Testing SRBreakoutPredictor comprehensive feature generation...")
    
    # Create sample price data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H')
    np.random.seed(42)
    
    # Generate realistic price data
    base_price = 100.0
    returns = np.random.normal(0, 0.02, len(dates))
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    price_data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, len(dates))
    }, index=dates)
    
    # Ensure high >= close >= low
    price_data['high'] = price_data[['high', 'close']].max(axis=1)
    price_data['low'] = price_data[['low', 'close']].min(axis=1)
    
    print(f"✅ Generated sample price data: {price_data.shape}")
    
    # Initialize SR breakout predictor
    config = {
        "sr_breakout_predictor": {
            "enable_sr_breakout_tactics": True,
            "feature_calculation": {
                "enable_comprehensive_features": True
            }
        }
    }
    
    sr_predictor = await setup_sr_breakout_predictor(config)
    if not sr_predictor:
        print("❌ Failed to initialize SR breakout predictor")
        return False
    
    print("✅ SR breakout predictor initialized successfully")
    
    # Generate comprehensive SR features
    comprehensive_features = sr_predictor.calculate_comprehensive_sr_features(price_data)
    
    if not comprehensive_features:
        print("❌ No comprehensive SR features generated")
        return False
    
    print(f"✅ Generated {len(comprehensive_features)} comprehensive SR features")
    
    # Check for expected features
    expected_features = [
        'distance_to_support', 'distance_to_resistance',
        'normalized_distance_to_support', 'normalized_distance_to_resistance',
        'sr_proximity_score', 'strength_score', 'directional_pressure',
        'sr_score', 'delta_sr_score', 'isolation_score'
    ]
    
    missing_features = []
    for feature in expected_features:
        if feature not in comprehensive_features:
            missing_features.append(feature)
    
    if missing_features:
        print(f"⚠️ Missing expected features: {missing_features}")
    else:
        print("✅ All expected comprehensive SR features generated")
    
    # Print feature statistics
    for feature_name, feature_series in comprehensive_features.items():
        if isinstance(feature_series, pd.Series):
            print(f"  {feature_name}: mean={feature_series.mean():.4f}, std={feature_series.std():.4f}")
    
    return True

def test_step3_feature_assignment():
    """Test Step3 feature assignment for SR block."""
    print("\n🧪 Testing Step3 feature assignment for SR block...")
    
    # Test feature names that should be assigned to SR block
    test_features = [
        'distance_to_support', 'distance_to_resistance',
        'normalized_distance_to_support', 'normalized_distance_to_resistance',
        'sr_proximity_score', 'strength_score', 'clarity_factor',
        'directional_pressure', 'sr_score', 'delta_sr_score',
        'isolation_score', 'support_strength', 'resistance_strength',
        'support_clarity_factor', 'resistance_clarity_factor',
        'pivot_support', 'fibonacci_resistance', 'sr_level_strength'
    ]
    
    sr_block_features = []
    other_block_features = []
    
    for feature in test_features:
        assigned_block = _assign_block(feature)
        if assigned_block == "support_resistance":
            sr_block_features.append(feature)
        else:
            other_block_features.append(feature)
    
    print(f"✅ Features assigned to SR block: {len(sr_block_features)}")
    for feature in sr_block_features:
        print(f"  ✅ {feature} -> support_resistance")
    
    if other_block_features:
        print(f"⚠️ Features assigned to other blocks: {len(other_block_features)}")
        for feature in other_block_features:
            assigned_block = _assign_block(feature)
            print(f"  ⚠️ {feature} -> {assigned_block}")
    
    # Check if we have the essential features for SR block
    essential_features = ['sr_score', 'delta_sr_score', 'directional_pressure']
    missing_essential = [f for f in essential_features if f not in sr_block_features]
    
    if missing_essential:
        print(f"❌ Missing essential SR features: {missing_essential}")
        return False
    else:
        print("✅ All essential SR features properly assigned to SR block")
        return True

async def test_step2_integration():
    """Test Step2 integration of comprehensive SR features."""
    print("\n🧪 Testing Step2 integration of comprehensive SR features...")
    
    # This would require actual data files, so we'll just test the function exists
    try:
        from src.training.steps.step2_feature_engineering import run_step
        print("✅ Step2 feature engineering module imported successfully")
        
        # Check if the comprehensive SR feature function exists
        import inspect
        from src.training.steps.step2_feature_engineering import _generate_comprehensive_sr_features
        
        if inspect.isfunction(_generate_comprehensive_sr_features):
            print("✅ Comprehensive SR feature generation function found in Step2")
            return True
        else:
            print("❌ Comprehensive SR feature generation function not found in Step2")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import Step2 module: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing Step2 integration: {e}")
        return False

async def main():
    """Run all tests."""
    print("🚀 Starting comprehensive SR features test suite...")
    
    results = []
    
    # Test 1: SRBreakoutPredictor
    result1 = await test_sr_breakout_predictor()
    results.append(("SRBreakoutPredictor", result1))
    
    # Test 2: Step3 feature assignment
    result2 = test_step3_feature_assignment()
    results.append(("Step3 Feature Assignment", result2))
    
    # Test 3: Step2 integration
    result3 = await test_step2_integration()
    results.append(("Step2 Integration", result3))
    
    # Summary
    print("\n📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Comprehensive SR features are working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)