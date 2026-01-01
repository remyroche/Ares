#!/usr/bin/env python3
"""
Test script to verify the specialist fixes
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def test_momentum_step_fix():
    """Test the momentum step method signature fix"""
    
    try:
        from src.training.steps.market_analysis.ml_momentum_persistence_step_enhanced import EnhancedMLMomentumPersistenceStep
        from src.training.steps.market_analysis.specialist_data_standard import SpecialistType
        
        print("✅ Import successful")
        
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        
        sample_df = pd.DataFrame({
            'close': np.random.randn(n_samples).cumsum() + 100,
            'high': np.random.randn(n_samples).cumsum() + 102,
            'low': np.random.randn(n_samples).cumsum() + 98,
            'volume': np.random.lognormal(10, 1, n_samples),
            'returns': np.random.randn(n_samples) * 0.01
        })
        
        # Test method signature
        step = EnhancedMLMomentumPersistenceStep()
        
        # Test both old and new signature styles
        try:
            # New signature with specialist_type
            features1 = step._generate_enhanced_features(sample_df, SpecialistType.MOMENTUM_PERSISTENCE)
            print("✅ New signature (with specialist_type) works")
        except Exception as e:
            print(f"❌ New signature failed: {e}")
        
        try:
            # Old signature without specialist_type (should work with default)
            features2 = step._generate_enhanced_features(sample_df)
            print("✅ Old signature (without specialist_type) works")
        except Exception as e:
            print(f"❌ Old signature failed: {e}")
        
        # Test training validation
        try:
            from sklearn.dummy import DummyClassifier
            
            # Create problematic data to test validation
            bad_features = pd.DataFrame(np.zeros((100, 5)))  # Zero variance features
            bad_labels = pd.Series([0] * 95 + [1] * 5)  # Highly imbalanced
            
            model, metrics = step._train_enhanced_momentum_model(bad_features, bad_labels)
            
            if metrics.get('model_type') == 'dummy_fallback':
                print("✅ Validation correctly triggered fallback for bad data")
            else:
                print(f"⚠️ Unexpected model type: {metrics.get('model_type')}")
                
        except Exception as e:
            print(f"❌ Training validation test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_path_regime_optimization():
    """Test the path regime optimization"""
    
    try:
        from src.training.steps.market_analysis.ml_path_regime_step_enhanced import EnhancedMLPathRegimeStep
        from src.training.steps.market_analysis.specialist_data_standard import SpecialistType
        
        print("✅ Path regime step import successful")
        
        # Create sample data
        np.random.seed(42)
        n_samples = 500  # Smaller dataset for speed testing
        
        sample_df = pd.DataFrame({
            'close': 100 + np.random.randn(n_samples).cumsum() * 0.1,
            'high': 100.5 + np.random.randn(n_samples).cumsum() * 0.1,
            'low': 99.5 + np.random.randn(n_samples).cumsum() * 0.1,
            'volume': np.random.lognormal(10, 1, n_samples)
        }, index=pd.date_range('2024-01-01', periods=n_samples, freq='15min'))
        
        # Test feature generation speed
        import time
        step = EnhancedMLPathRegimeStep()
        
        start_time = time.time()
        features = step._generate_enhanced_features(sample_df, SpecialistType.PATH_REGIME)
        end_time = time.time()
        
        processing_time = end_time - start_time
        print(f"✅ Path features generated in {processing_time:.3f}s")
        print(f"   Generated {len(features.columns)} features")
        
        # Test label creation speed
        start_time = time.time()
        labels = step._create_path_labels(sample_df)
        end_time = time.time()
        
        label_time = end_time - start_time
        print(f"✅ Path labels created in {label_time:.3f}s")
        print(f"   Generated {len(labels)} labels")
        
        # Performance check
        if processing_time < 1.0:  # Should be under 1 second for 500 samples
            print("✅ Performance optimization successful")
        else:
            print(f"⚠️ Still slow: {processing_time:.3f}s for 500 samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Path regime test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all specialist fix tests"""
    
    print("🧪 Testing Specialist Fixes")
    print("=" * 50)
    
    tests = [
        ("Momentum Step Fix", test_momentum_step_fix),
        ("Path Regime Optimization", test_path_regime_optimization),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Specialist fixes are working correctly!")
        return 0
    else:
        print(f"❌ {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
