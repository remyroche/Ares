#!/usr/bin/env python3
"""
Test script to verify import fixes
"""

import sys
import traceback

def test_import(module_name):
    """Test importing a module and report results."""
    try:
        __import__(module_name)
        print(f"✅ {module_name} - SUCCESS")
        return True
    except Exception as e:
        print(f"❌ {module_name} - FAILED: {e}")
        return False

def main():
    """Test the problematic imports."""
    print("🔍 Testing import fixes...")
    
    # Test the modules that were failing
    test_modules = [
        "src.training.steps.pre_training.analyst_profit_labeler",
        "src.training.steps.pre_training.unified_data_driven_pipeline.steps.feature_generation_data_validation_step",
        "src.training.steps.pre_training.unified_data_driven_pipeline.steps.feature_generation_feature_generation_step",
        "src.training.steps.pre_training.unified_data_driven_pipeline.steps.feature_generation_feature_selection_step",
        "src.training.steps.pre_training.unified_data_driven_pipeline.steps.feature_generation_final_validation_step",
        "src.training.steps.pre_training.unified_data_driven_pipeline.steps.feature_generation_interaction_generation_step",
        "src.training.steps.pre_training.unified_data_driven_pipeline.steps.feature_generation_labeling_integration_step",
        "src.training.steps.pre_training.unified_data_driven_pipeline.steps.feature_generation_lookback_optimization_step",
        "src.training.steps.pre_training.unified_data_driven_pipeline.steps.feature_generation_period_optimization_step",
        "src.training.steps.pre_training.unified_data_driven_pipeline.steps.feature_generation_period_lookback_optimization_step",
        "src.training.steps.pre_training.unified_data_driven_pipeline.steps.feature_generation_vectorization_step",
        "src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.interactive_feature_generation_component",
        "src.training.steps.pre_training.components.final_feature_selection",
        "src.training.steps.pre_training.feature_lookback_optimization.feature_lookback_optimization"
    ]
    
    success_count = 0
    total_count = len(test_modules)
    
    for module in test_modules:
        if test_import(module):
            success_count += 1
    
    print(f"\n📊 Results: {success_count}/{total_count} modules imported successfully")
    
    if success_count == total_count:
        print("🎉 All import issues have been resolved!")
        return 0
    else:
        print("⚠️ Some import issues remain")
        return 1

if __name__ == "__main__":
    sys.exit(main())
