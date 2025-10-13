#!/usr/bin/env python3
"""
Validation script for the Unified Data-Driven Feature Pipeline

This script validates that the implementation is working correctly.
"""

import sys
import os
import traceback

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test core imports
        from core.config import create_default_config, UnifiedPipelineConfig
        print("✓ Core config imports successful")
        
        from core.unified_pipeline import create_unified_pipeline, process_features
        print("✓ Core pipeline imports successful")
        
        # Test time series CV imports
        from time_series_cv import create_purged_embargoed_cv, PurgedEmbargoedConfig
        print("✓ Time series CV imports successful")
        
        # Test statistical analysis imports
        from statistical_analysis import StatisticalAnalysisFramework
        print("✓ Statistical analysis imports successful")
        
        # Test feature selection imports
        from feature_selection import create_default_objectives, MultiObjectiveFeatureSelector
        print("✓ Feature selection imports successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        import pandas as pd
        
        # Create sample data
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        targets = pd.Series(np.random.randn(100))
        
        # Test configuration creation
        from core.config import create_default_config
        config = create_default_config()
        print("✓ Configuration creation successful")
        
        # Test statistical framework
        from statistical_analysis import StatisticalAnalysisFramework
        framework = StatisticalAnalysisFramework()
        characteristics = framework.analyze_data_characteristics(data)
        print("✓ Statistical analysis successful")
        
        # Test time series CV
        from time_series_cv import create_purged_embargoed_cv
        cv = create_purged_embargoed_cv(n_splits=3, test_size=0.2, train_size=0.6)
        splits = cv.split(data, targets=targets)
        print(f"✓ Time series CV successful: {len(splits)} splits generated")
        
        # Test multi-objective selector
        from feature_selection import create_default_objectives, MultiObjectiveFeatureSelector
        objectives = create_default_objectives()
        selector = MultiObjectiveFeatureSelector(objectives=objectives, max_features=5, min_features=2)
        result = selector.select_features(data, targets)
        print(f"✓ Feature selection successful: {len(result.selected_features)} features selected")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_pipeline_integration():
    """Test pipeline integration."""
    print("\nTesting pipeline integration...")
    
    try:
        import numpy as np
        import pandas as pd
        
        # Create sample data
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200),
            'feature3': np.random.randn(200),
            'feature4': np.random.randn(200),
            'feature5': np.random.randn(200)
        })
        targets = pd.Series(np.random.randn(200))
        
        # Test pipeline creation
        from core.unified_pipeline import create_unified_pipeline, create_default_config
        config = create_default_config()
        config.feature_selection.multi_objective.max_features = 3
        config.feature_selection.multi_objective.min_features = 2
        config.feature_selection.cv_config.n_splits = 3
        
        pipeline = create_unified_pipeline(config)
        print("✓ Pipeline creation successful")
        
        # Test pipeline processing
        result = pipeline.process(data, targets)
        print(f"✓ Pipeline processing successful: {len(result.selected_features)} features selected")
        print(f"  Processing time: {result.processing_time:.2f}s")
        print(f"  Objective values: {result.objective_values}")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    print("=" * 60)
    print("UNIFIED DATA-DRIVEN FEATURE PIPELINE VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Pipeline Integration", test_pipeline_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        
        if test_func():
            print(f"✓ {test_name} PASSED")
            passed += 1
        else:
            print(f"✗ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print(f"VALIDATION SUMMARY: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The implementation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())