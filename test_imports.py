#!/usr/bin/env python3
"""
Simple test script to verify enhanced feature selection imports
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test importing the enhanced feature selection modules."""
    print("🧪 Testing Enhanced Feature Selection Imports")
    print("=" * 50)
    
    success_count = 0
    total_tests = 0
    
    # Test 1: Multi-objective selector with enhanced methods
    total_tests += 1
    try:
        from src.training.steps.pre_training.unified_data_driven_pipeline.feature_selection.multi_objective_selector import (
            MultiObjectiveFeatureSelector, create_default_objectives
        )
        print("✅ Multi-objective selector with enhanced methods imported")
        success_count += 1
    except Exception as e:
        print(f"❌ Multi-objective selector import failed: {e}")
    
    # Test 2: Enhanced feature selection methods
    total_tests += 1
    try:
        from src.feature_selection.advanced.improved_mrmr import ImprovedMRMR
        print("✅ Improved mRMR imported")
        success_count += 1
    except Exception as e:
        print(f"❌ Improved mRMR import failed: {e}")
    
    # Test 3: VectorBT selectors
    total_tests += 1
    try:
        from src.feature_selection.vectorbt.vectorbt_mrmr_selector import VectorBTMRMRSelector
        print("✅ VectorBT mRMR selector imported")
        success_count += 1
    except Exception as e:
        print(f"❌ VectorBT mRMR selector import failed: {e}")
    
    # Test 4: VectorBT RFE selector
    total_tests += 1
    try:
        from src.feature_selection.vectorbt.vectorbt_rfe_selector import VectorBTRFESelector
        print("✅ VectorBT RFE selector imported")
        success_count += 1
    except Exception as e:
        print(f"❌ VectorBT RFE selector import failed: {e}")
    
    # Test 5: VectorBT LASSO selector
    total_tests += 1
    try:
        from src.feature_selection.vectorbt.vectorbt_regularization import VectorBTRegularizationSelector
        print("✅ VectorBT LASSO selector imported")
        success_count += 1
    except Exception as e:
        print(f"❌ VectorBT LASSO selector import failed: {e}")
    
    # Test 6: Enhanced ensemble selector
    total_tests += 1
    try:
        from src.feature_selection.advanced.enhanced_ensemble_selector import EnhancedEnsembleAdvancedSelector
        print("✅ Enhanced ensemble selector imported")
        success_count += 1
    except Exception as e:
        print(f"❌ Enhanced ensemble selector import failed: {e}")
    
    # Test 7: Enhanced advanced selector
    total_tests += 1
    try:
        from src.feature_selection.advanced.enhanced_advanced_selector import EnhancedAdvancedFeatureSelector
        print("✅ Enhanced advanced selector imported")
        success_count += 1
    except Exception as e:
        print(f"❌ Enhanced advanced selector import failed: {e}")
    
    # Test 8: Pipeline configuration
    total_tests += 1
    try:
        from src.training.steps.pre_training.unified_data_driven_pipeline.core.config import (
            FeatureSelectionConfig, UnifiedPipelineConfig
        )
        print("✅ Enhanced pipeline configuration imported")
        success_count += 1
    except Exception as e:
        print(f"❌ Pipeline configuration import failed: {e}")
    
    # Test 9: Check if enhanced methods are available in multi-objective selector
    total_tests += 1
    try:
        from src.training.steps.pre_training.unified_data_driven_pipeline.feature_selection.multi_objective_selector import (
            MultiObjectiveFeatureSelector
        )
        
        # Check if enhanced methods are available
        selector = MultiObjectiveFeatureSelector([])
        enhanced_methods = ['_improved_mrmr_selection', '_vectorbt_mrmr_selection', 
                           '_vectorbt_rfe_selection', '_vectorbt_lasso_selection',
                           '_enhanced_ensemble_selection', '_enhanced_advanced_selection']
        
        available_methods = []
        for method in enhanced_methods:
            if hasattr(selector, method):
                available_methods.append(method)
        
        if available_methods:
            print(f"✅ Enhanced methods available in multi-objective selector: {len(available_methods)}")
            success_count += 1
        else:
            print("❌ No enhanced methods found in multi-objective selector")
    except Exception as e:
        print(f"❌ Enhanced methods check failed: {e}")
    
    print(f"\n📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Enhanced feature selection integration is working.")
        return True
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)