#!/usr/bin/env python3
"""
Simple test script for the enhanced multi-stage feature selection pipeline.
This test focuses on the configuration and basic structure without requiring all dependencies.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_configuration():
    """Test the configuration system."""
    print("🧪 Testing Enhanced Pipeline Configuration")
    print("=" * 50)
    
    try:
        from src.training.steps.pre_training.feature_selection.core.config import (
            FeatureSelectionConfig, NewPipelineConfig
        )
        
        # Test NewPipelineConfig
        print("📊 Testing NewPipelineConfig...")
        new_config = NewPipelineConfig()
        
        print(f"   ✅ enable_new_pipeline: {new_config.enable_new_pipeline}")
        print(f"   ✅ stage1_mrmr_weight: {new_config.stage1_mrmr_weight}")
        print(f"   ✅ stage1_spearman_weight: {new_config.stage1_spearman_weight}")
        print(f"   ✅ stage1_target_ratio: {new_config.stage1_target_ratio}")
        print(f"   ✅ stage2_initial_batch_size: {new_config.stage2_initial_batch_size}")
        print(f"   ✅ stage2_medium_batch_size: {new_config.stage2_medium_batch_size}")
        print(f"   ✅ stage2_final_batch_size: {new_config.stage2_final_batch_size}")
        
        # Test FeatureSelectionConfig with new pipeline
        print("\n📊 Testing FeatureSelectionConfig with new pipeline...")
        config = FeatureSelectionConfig()
        
        print(f"   ✅ enable_new_pipeline: {config.enable_new_pipeline}")
        print(f"   ✅ target_features: {config.target_features}")
        print(f"   ✅ stage1_mrmr_weight: {config.stage1_mrmr_weight}")
        print(f"   ✅ stage1_spearman_weight: {config.stage1_spearman_weight}")
        
        # Test configuration updates
        print("\n📊 Testing configuration updates...")
        config.enable_new_pipeline = True
        config.target_features = 50
        config.stage1_mrmr_weight = 0.8
        config.stage1_spearman_weight = 0.2
        
        print(f"   ✅ Updated enable_new_pipeline: {config.enable_new_pipeline}")
        print(f"   ✅ Updated target_features: {config.target_features}")
        print(f"   ✅ Updated stage1_mrmr_weight: {config.stage1_mrmr_weight}")
        print(f"   ✅ Updated stage1_spearman_weight: {config.stage1_spearman_weight}")
        
        print("\n✅ Configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imports():
    """Test that all required modules can be imported."""
    print("\n🧪 Testing Module Imports")
    print("=" * 30)
    
    try:
        # Test core imports
        print("📊 Testing core imports...")
        from src.training.steps.pre_training.feature_selection.core.config import FeatureSelectionConfig
        from src.training.steps.pre_training.feature_selection.core.pipeline import MultiStageFeatureSelector
        print("   ✅ Core modules imported successfully")
        
        # Test enhanced pipeline import
        print("📊 Testing enhanced pipeline import...")
        from src.training.steps.pre_training.feature_selection.core.enhanced_pipeline import EnhancedMultiStageFeatureSelector
        print("   ✅ Enhanced pipeline imported successfully")
        
        # Test VectorBT imports (may fail if not available)
        print("📊 Testing VectorBT imports...")
        try:
            from src.feature_selection.vectorbt.vectorbt_mrmr_selector import VectorBTMRMRSelector
            print("   ✅ VectorBT mRMR selector imported successfully")
        except ImportError as e:
            print(f"   ⚠️ VectorBT mRMR selector not available: {e}")
        
        print("\n✅ Import tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_structure():
    """Test the pipeline structure and method availability."""
    print("\n🧪 Testing Pipeline Structure")
    print("=" * 35)
    
    try:
        from src.training.steps.pre_training.feature_selection.core.enhanced_pipeline import EnhancedMultiStageFeatureSelector
        from src.training.steps.pre_training.feature_selection.core.config import FeatureSelectionConfig
        
        # Create selector
        config = FeatureSelectionConfig()
        config.enable_new_pipeline = True
        config.target_features = 60
        
        selector = EnhancedMultiStageFeatureSelector(config)
        
        # Test method availability
        print("📊 Testing method availability...")
        methods_to_test = [
            'select_features',
            '_stage_1_mrmr_spearman_combination',
            '_stage_2_progressive_refinement',
            '_calculate_mrmr_scores',
            '_calculate_spearman_scores',
            '_calculate_ensemble_feature_scores',
            '_determine_batch_size',
            '_select_features_to_remove'
        ]
        
        for method_name in methods_to_test:
            if hasattr(selector, method_name):
                print(f"   ✅ {method_name}: Available")
            else:
                print(f"   ❌ {method_name}: Missing")
                return False
        
        print("\n✅ Pipeline structure tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Enhanced Multi-Stage Feature Selection Pipeline Tests")
    print("=" * 60)
    
    tests = [
        ("Configuration", test_configuration),
        ("Imports", test_imports),
        ("Pipeline Structure", test_pipeline_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed}/{total} tests passed")
    print('='*60)
    
    if passed == total:
        print("🎉 All tests passed! The enhanced pipeline is ready to use.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)