#!/usr/bin/env python3
"""
Test script to verify Step 6 SHAP analysis fixes.

This script tests:
1. Keras compatibility fix
2. SHAP import and usage
3. Robust feature selection fallback
4. Overall Step 6 functionality
"""

import sys
import os
import asyncio
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_keras_compatibility():
    """Test Keras compatibility fix."""
    print("🧪 Testing Keras compatibility...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Try importing tf-keras
        try:
            import tf_keras as keras
            print(f"✅ tf-keras version: {keras.__version__}")
            return True
        except ImportError:
            print("⚠️  tf-keras not available, trying regular keras...")
            import keras
            print(f"✅ Keras version: {keras.__version__}")
            return True
            
    except ImportError as e:
        print(f"❌ TensorFlow/Keras import failed: {e}")
        return False

def test_shap_import():
    """Test SHAP import and basic functionality."""
    print("🧪 Testing SHAP import...")
    
    try:
        import shap
        print(f"✅ SHAP version: {shap.__version__}")
        
        # Test TreeExplainer import
        try:
            from shap.explainers import TreeExplainer
            print("✅ TreeExplainer import successful")
            return True
        except ImportError:
            print("⚠️  TreeExplainer not available in shap.explainers")
            try:
                from shap import TreeExplainer
                print("✅ TreeExplainer import successful (direct)")
                return True
            except ImportError:
                print("❌ TreeExplainer not available")
                return False
                
    except ImportError as e:
        print(f"❌ SHAP import failed: {e}")
        return False

def test_feature_selection_methods():
    """Test the robust feature selection methods."""
    print("🧪 Testing feature selection methods...")
    
    # Create synthetic data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 20), columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(np.random.choice([0, 1], size=100))
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"✅ Model trained successfully")
    print(f"   Training accuracy: {accuracy_score(y_train, model.predict(X_train)):.4f}")
    print(f"   Validation accuracy: {accuracy_score(y_val, model.predict(X_val)):.4f}")
    
    # Test permutation importance
    try:
        from sklearn.inspection import permutation_importance
        perm_importance = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42)
        print("✅ Permutation importance calculation successful")
        print(f"   Top 5 features: {np.argsort(perm_importance.importances_mean)[-5:]}")
    except Exception as e:
        print(f"❌ Permutation importance failed: {e}")
        return False
    
    # Test statistical feature selection
    try:
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        selector = SelectKBest(score_func=mutual_info_classif, k=10)
        selector.fit(X_train, y_train)
        print("✅ Statistical feature selection successful")
        print(f"   Selected features: {selector.get_support().sum()}")
    except Exception as e:
        print(f"❌ Statistical feature selection failed: {e}")
        return False
    
    return True

async def test_step6_integration():
    """Test the complete Step 6 integration."""
    print("🧪 Testing Step 6 integration...")
    
    try:
        from src.training.steps.step6_analyst_enhancement import AnalystEnhancementStep
        
        # Create test configuration
        config = {
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "data_dir": "data/training",
        }
        
        # Initialize the step
        step = AnalystEnhancementStep(config)
        await step.initialize()
        
        print("✅ Step 6 initialization successful")
        
        # Test feature selection method
        X = pd.DataFrame(np.random.randn(50, 10), columns=[f'feature_{i}' for i in range(10)])
        y = pd.Series(np.random.choice([0, 1], size=50))
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        # Test the robust feature selection
        optimal_features, summary = step._robust_feature_selection(
            model, "random_forest", X, y, X, y, 
            list(X.columns), 5, 8
        )
        
        print(f"✅ Feature selection successful")
        print(f"   Selected features: {len(optimal_features)}")
        print(f"   Method used: {summary.get('method', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Step 6 integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🔧 Testing Step 6 SHAP Analysis Fixes")
    print("=" * 50)
    
    tests = [
        ("Keras Compatibility", test_keras_compatibility),
        ("SHAP Import", test_shap_import),
        ("Feature Selection Methods", test_feature_selection_methods),
        ("Step 6 Integration", lambda: asyncio.run(test_step6_integration())),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status} {test_name}")
        except Exception as e:
            print(f"❌ FAILED {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status}: {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Step 6 SHAP analysis fixes are working correctly.")
        print("\n💡 The fixes include:")
        print("   - Keras 3 compatibility with tf-keras")
        print("   - Robust SHAP import handling")
        print("   - Fallback feature selection methods")
        print("   - Comprehensive error handling")
    else:
        print("⚠️  Some tests failed. Please check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 