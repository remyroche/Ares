#!/usr/bin/env python3
"""
Minimal test to isolate the logger issue.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_config_only():
    """Test only configuration import."""
    print("🧪 Testing Configuration Only")
    print("=" * 30)
    
    try:
        from src.training.steps.pre_training.feature_selection.core.config import FeatureSelectionConfig
        print("✅ Configuration import successful")
        
        config = FeatureSelectionConfig()
        print(f"✅ Target features: {config.target_features}")
        print(f"✅ RFE step size: {config.rfe_step_size}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_pipeline_import():
    """Test pipeline import step by step."""
    print("\n🧪 Testing Pipeline Import Step by Step")
    print("=" * 40)
    
    try:
        print("Step 1: Importing basic modules...")
        import pandas as pd
        import numpy as np
        print("✅ Basic modules imported")
        
        print("Step 2: Importing configuration...")
        from src.training.steps.pre_training.feature_selection.core.config import FeatureSelectionConfig
        print("✅ Configuration imported")
        
        print("Step 3: Importing pipeline...")
        from src.training.steps.pre_training.feature_selection.core.pipeline import MultiStageFeatureSelector
        print("✅ Pipeline imported")
        
        print("Step 4: Creating instance...")
        config = FeatureSelectionConfig()
        selector = MultiStageFeatureSelector(config)
        print("✅ Pipeline instance created")
        
        return True
    except Exception as e:
        print(f"❌ Pipeline import failed at step: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run minimal tests."""
    print("🚀 Minimal Test")
    print("=" * 15)
    
    tests = [
        ("Configuration Only", test_config_only),
        ("Pipeline Import", test_pipeline_import)
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
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)