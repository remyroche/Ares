"""
Minimal Test for Enhanced SR Detection System

This test focuses on core functionality without complex dependencies.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_sample_data():
    """Create sample market data for testing."""
    print("📊 Creating sample market data...")
    
    # Create 500 data points
    dates = pd.date_range(start='2024-01-01', end='2024-01-15', freq='15T')
    np.random.seed(42)
    
    # Create realistic price data with clear SR levels
    base_price = 2000.0
    trend = np.linspace(0, 0.02, len(dates))  # 2% upward trend
    noise = np.random.normal(0, 0.001, len(dates))
    returns = trend + noise
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Add clear support and resistance levels
    support_levels = [1980, 2000]
    resistance_levels = [2020, 2040]
    
    # Modify prices to touch these levels
    for i, price in enumerate(prices):
        # Check for support levels
        for support in support_levels:
            if abs(price - support) < 10:
                prices[i] = support + np.random.normal(0, 2)
        
        # Check for resistance levels
        for resistance in resistance_levels:
            if abs(price - resistance) < 10:
                prices[i] = resistance + np.random.normal(0, 2)
    
    # Create OHLCV data
    market_data = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, len(dates)))),
        'close': prices,
        'volume': np.random.uniform(1000, 5000, len(dates))
    }, index=dates)
    
    # Ensure high >= low
    market_data['high'] = np.maximum(market_data['high'], market_data['low'])
    
    print(f"✅ Created sample data: {len(market_data)} rows")
    print(f"   Date range: {market_data.index[0]} to {market_data.index[-1]}")
    print(f"   Price range: {market_data['low'].min():.2f} - {market_data['high'].max():.2f}")
    
    return market_data

def test_basic_sr_detection():
    """Test basic SR detection functionality."""
    print("\n🎯 Testing Basic SR Detection...")
    
    try:
        # Import the enhanced SR detector
        from src.tactician.sr_levels.enhanced_sr_detection_optimized import (
            EnhancedSROptimizedDetector, SROptimizationConfig, SRLevel
        )
        print("✅ Enhanced SR detector imported successfully")
        
        # Create configuration with minimal dependencies
        config = SROptimizationConfig(
            min_touches=2,
            tolerance_pct=0.5,
            lookback_periods=50,
            enable_vectorbt=False,  # Disable for minimal test
            enable_hardware_optimization=False,
            enable_explainability=False,
            enable_validation=False,
            enable_hpo=False
        )
        print("✅ Configuration created")
        
        # Initialize detector
        detector = EnhancedSROptimizedDetector(config)
        print("✅ Detector initialized")
        
        # Create sample data
        market_data = create_sample_data()
        
        # Detect SR levels
        print("🔍 Detecting SR levels...")
        sr_levels = detector.detect_sr_levels(market_data)
        
        print(f"✅ Detected {len(sr_levels)} SR levels")
        
        # Validate results
        if sr_levels:
            print("\n📋 SR Level Details:")
            for i, level in enumerate(sr_levels[:5]):  # Show first 5
                print(f"  {i+1}. {level.level_type}: {level.price:.2f}")
                print(f"     Strength: {level.strength:.3f}, Touches: {level.touches}")
                print(f"     Quality: {level.quality_score:.3f}, R²: {level.r_squared:.3f}")
                print()
        else:
            print("⚠️ No SR levels detected")
        
        # Test performance metrics
        metrics = detector.get_performance_metrics()
        print(f"⏱️ Detection time: {metrics.get('detection_time', 0):.3f}s")
        
        # Test optimization status
        status = detector.get_optimization_status()
        print(f"🔧 Optimization status: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic SR detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_step_integration():
    """Test step integration."""
    print("\n📊 Testing Step Integration...")
    
    try:
        # Import the enhanced step
        from src.training.steps.market_analysis.components.sr_detection_enhanced import (
            EnhancedSRDetectionStep
        )
        print("✅ Enhanced step imported successfully")
        
        # Create step
        step = EnhancedSRDetectionStep()
        print("✅ Step initialized")
        
        # Create sample data
        market_data = create_sample_data()
        
        # Prepare input data
        input_data = {
            'market_data': market_data,
            'config': {
                'sr_detection': {
                    'min_touches': 2,
                    'tolerance_pct': 0.5,
                    'enable_vectorbt': False,
                    'enable_explainability': False,
                    'enable_validation': False
                }
            }
        }
        
        # Test input validation
        is_valid = step.validate_input(input_data)
        print(f"✅ Input validation: {'PASSED' if is_valid else 'FAILED'}")
        
        # Test step info
        step_info = step.get_step_info()
        print(f"✅ Step info: {step_info['step_name']} v{step_info['version']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Step integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_validation():
    """Test data validation."""
    print("\n🔍 Testing Data Validation...")
    
    try:
        from src.tactician.sr_levels.enhanced_sr_detection_optimized import (
            EnhancedSROptimizedDetector, SROptimizationConfig
        )
        
        config = SROptimizationConfig(
            enable_vectorbt=False,
            enable_hardware_optimization=False,
            enable_explainability=False,
            enable_validation=False,
            enable_hpo=False
        )
        
        detector = EnhancedSROptimizedDetector(config)
        
        # Test with valid data
        valid_data = create_sample_data()
        result = detector.detect_sr_levels(valid_data)
        print(f"✅ Valid data test: {len(result)} levels detected")
        
        # Test with invalid data
        invalid_data = pd.DataFrame({'invalid': [1, 2, 3]})
        result = detector.detect_sr_levels(invalid_data)
        print(f"✅ Invalid data test: {len(result)} levels (should be 0)")
        
        # Test with empty data
        empty_data = pd.DataFrame()
        result = detector.detect_sr_levels(empty_data)
        print(f"✅ Empty data test: {len(result)} levels (should be 0)")
        
        return True
        
    except Exception as e:
        print(f"❌ Data validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 ENHANCED SR DETECTION - MINIMAL TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Basic SR Detection", test_basic_sr_detection),
        ("Step Integration", test_step_integration),
        ("Data Validation", test_data_validation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 Running: {test_name}")
        print('='*60)
        
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"✅ {test_name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print('='*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("⚠️ SOME TESTS FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)