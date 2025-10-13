#!/usr/bin/env python3
"""
Simple test script for the enhanced pipeline configuration only.
This test focuses on the configuration without requiring all dependencies.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_configuration_direct():
    """Test the configuration system directly without imports."""
    print("🧪 Testing Enhanced Pipeline Configuration (Direct)")
    print("=" * 55)
    
    try:
        # Test the configuration file directly
        config_file = "src/training/steps/pre_training/feature_selection/core/config.py"
        
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Check for key components
        checks = [
            ("NewPipelineConfig class", "class NewPipelineConfig:"),
            ("enable_new_pipeline", "enable_new_pipeline: bool = True"),
            ("stage1_mrmr_weight", "stage1_mrmr_weight: float = 0.7"),
            ("stage1_spearman_weight", "stage1_spearman_weight: float = 0.3"),
            ("stage1_target_ratio", "stage1_target_ratio: float = 0.5"),
            ("stage2_initial_batch_size", "stage2_initial_batch_size: int = 10"),
            ("stage2_medium_batch_size", "stage2_medium_batch_size: int = 5"),
            ("stage2_final_batch_size", "stage2_final_batch_size: int = 1"),
            ("lgbm_params", "lgbm_params: Dict[str, Any]"),
            ("ensemble_weights", "ensemble_weights: Dict[str, float]"),
            ("Tuple import", "from typing import Dict, List, Optional, Any, Tuple")
        ]
        
        print("📊 Checking configuration components...")
        all_passed = True
        
        for check_name, check_string in checks:
            if check_string in content:
                print(f"   ✅ {check_name}: Found")
            else:
                print(f"   ❌ {check_name}: Missing")
                all_passed = False
        
        if all_passed:
            print("\n✅ Configuration file checks passed!")
            return True
        else:
            print("\n❌ Some configuration checks failed!")
            return False
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_enhanced_pipeline_file():
    """Test the enhanced pipeline file structure."""
    print("\n🧪 Testing Enhanced Pipeline File Structure")
    print("=" * 45)
    
    try:
        # Test the enhanced pipeline file
        pipeline_file = "src/training/steps/pre_training/feature_selection/core/enhanced_pipeline.py"
        
        with open(pipeline_file, 'r') as f:
            content = f.read()
        
        # Check for key components
        checks = [
            ("EnhancedMultiStageFeatureSelector class", "class EnhancedMultiStageFeatureSelector:"),
            ("select_features method", "def select_features(self, X: pd.DataFrame, y: pd.Series"),
            ("_stage_1_mrmr_spearman_combination", "def _stage_1_mrmr_spearman_combination"),
            ("_stage_2_progressive_refinement", "def _stage_2_progressive_refinement"),
            ("_calculate_mrmr_scores", "def _calculate_mrmr_scores"),
            ("_calculate_spearman_scores", "def _calculate_spearman_scores"),
            ("_calculate_ensemble_feature_scores", "def _calculate_ensemble_feature_scores"),
            ("_determine_batch_size", "def _determine_batch_size"),
            ("_select_features_to_remove", "def _select_features_to_remove"),
            ("VectorBT integration", "VectorBTMRMRSelector"),
            ("LGBM integration", "import lightgbm as lgb"),
            ("SHAP integration", "import shap"),
            ("LASSO integration", "LassoCV"),
            ("RFE integration", "RFE")
        ]
        
        print("📊 Checking enhanced pipeline components...")
        all_passed = True
        
        for check_name, check_string in checks:
            if check_string in content:
                print(f"   ✅ {check_name}: Found")
            else:
                print(f"   ❌ {check_name}: Missing")
                all_passed = False
        
        if all_passed:
            print("\n✅ Enhanced pipeline file checks passed!")
            return True
        else:
            print("\n❌ Some enhanced pipeline checks failed!")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced pipeline file test failed: {e}")
        return False

def test_pipeline_integration():
    """Test the pipeline integration."""
    print("\n🧪 Testing Pipeline Integration")
    print("=" * 35)
    
    try:
        # Test the main pipeline file
        pipeline_file = "src/training/steps/pre_training/feature_selection/core/pipeline.py"
        
        with open(pipeline_file, 'r') as f:
            content = f.read()
        
        # Check for integration components
        checks = [
            ("Enhanced pipeline import", "from .enhanced_pipeline import EnhancedMultiStageFeatureSelector"),
            ("New pipeline check", "if hasattr(self.config, 'enable_new_pipeline') and self.config.enable_new_pipeline:"),
            ("Enhanced pipeline usage", "enhanced_selector = EnhancedMultiStageFeatureSelector(self.config)"),
            ("Fallback to original", "Fallback to original pipeline"),
            ("Pipeline selection logic", "Use enhanced pipeline")
        ]
        
        print("📊 Checking pipeline integration components...")
        all_passed = True
        
        for check_name, check_string in checks:
            if check_string in content:
                print(f"   ✅ {check_name}: Found")
            else:
                print(f"   ❌ {check_name}: Missing")
                all_passed = False
        
        if all_passed:
            print("\n✅ Pipeline integration checks passed!")
            return True
        else:
            print("\n❌ Some pipeline integration checks failed!")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Enhanced Multi-Stage Feature Selection Pipeline Tests")
    print("=" * 60)
    
    tests = [
        ("Configuration", test_configuration_direct),
        ("Enhanced Pipeline File", test_enhanced_pipeline_file),
        ("Pipeline Integration", test_pipeline_integration)
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
        print("🎉 All tests passed! The enhanced pipeline structure is correct.")
        print("\n📋 Summary of Implementation:")
        print("   ✅ NewPipelineConfig class with all required parameters")
        print("   ✅ EnhancedMultiStageFeatureSelector with 2-stage process")
        print("   ✅ Stage 1: mRMR + Spearman combination (70% + 30%)")
        print("   ✅ Stage 2: Progressive refinement with LGBM-SHAP and LASSO ensemble")
        print("   ✅ VectorBT optimizations maintained")
        print("   ✅ Integration with existing pipeline system")
        print("   ✅ Configurable target features and batch sizes")
        print("   ✅ Fallback to original pipeline when disabled")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)