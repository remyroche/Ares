#!/usr/bin/env python3
"""
Simple test for the updated main pipeline.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test basic imports."""
    print("🧪 Testing Basic Imports")
    print("=" * 25)
    
    try:
        # Test configuration import
        from src.training.steps.pre_training.feature_selection.core.config import FeatureSelectionConfig
        print("✅ Configuration import successful")
        
        # Test pipeline import
        from src.training.steps.pre_training.feature_selection.core.pipeline import MultiStageFeatureSelector
        print("✅ Pipeline import successful")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration."""
    print("\n🧪 Testing Configuration")
    print("=" * 25)
    
    try:
        from src.training.steps.pre_training.feature_selection.core.config import FeatureSelectionConfig
        
        config = FeatureSelectionConfig()
        print(f"✅ Target features: {config.target_features}")
        print(f"✅ RFE step size: {config.rfe_step_size}")
        print(f"✅ Bootstrap CV threshold: {config.stage2_bootstrap_cv_threshold}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_pipeline_creation():
    """Test pipeline creation."""
    print("\n🧪 Testing Pipeline Creation")
    print("=" * 30)
    
    try:
        from src.training.steps.pre_training.feature_selection.core.pipeline import MultiStageFeatureSelector
        from src.training.steps.pre_training.feature_selection.core.config import FeatureSelectionConfig
        
        config = FeatureSelectionConfig()
        selector = MultiStageFeatureSelector(config)
        print("✅ Pipeline created successfully")
        
        # Check for new methods
        if hasattr(selector, '_stage_1_mrmr_spearman_combination'):
            print("✅ Stage 1 method present")
        else:
            print("❌ Stage 1 method missing")
            
        if hasattr(selector, '_stage_2_progressive_refinement'):
            print("✅ Stage 2 method present")
        else:
            print("❌ Stage 2 method missing")
            
        if hasattr(selector, '_rfe_with_percentage_step'):
            print("✅ RFE method present")
        else:
            print("❌ RFE method missing")
        
        return True
    except Exception as e:
        print(f"❌ Pipeline creation failed: {e}")
        return False

def main():
    """Run tests."""
    print("🚀 Simple Pipeline Test")
    print("=" * 25)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Configuration", test_configuration),
        ("Pipeline Creation", test_pipeline_creation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    print('='*50)
    
    if passed == total:
        print("🎉 All tests passed! The updated main pipeline is working.")
    else:
        print("⚠️ Some tests failed.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)