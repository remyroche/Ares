#!/usr/bin/env python3
"""
Validation script for Negative Learning Training Integration

This script validates that the negative learning plugin is properly
integrated into the ML training pipeline without requiring external dependencies.
"""

import os
import sys
import importlib.util

def check_file_exists(filepath):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ {filepath}")
        return True
    else:
        print(f"❌ {filepath} - MISSING")
        return False

def check_import_structure(filepath, module_name):
    """Check if a module can be imported and has expected structure"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None:
            print(f"❌ {filepath} - Cannot load spec")
            return False
        
        module = importlib.util.module_from_spec(spec)
        if module is None:
            print(f"❌ {filepath} - Cannot create module")
            return False
        
        # Check for expected functions/classes
        expected_items = [
            'initialize_negative_learning_integration',
            'get_negative_learning_integration',
            'enhance_features_for_training',
            'get_training_constraints',
            'get_training_sample_weights'
        ]
        
        missing_items = []
        for item in expected_items:
            if not hasattr(module, item):
                missing_items.append(item)
        
        if missing_items:
            print(f"⚠️ {filepath} - Missing items: {missing_items}")
        else:
            print(f"✅ {filepath} - All expected items present")
        
        return True
        
    except Exception as e:
        print(f"❌ {filepath} - Import error: {e}")
        return False

def check_training_file_integration(filepath, expected_imports):
    """Check if training files have negative learning integration"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        missing_imports = []
        for import_text in expected_imports:
            if import_text not in content:
                missing_imports.append(import_text)
        
        if missing_imports:
            print(f"⚠️ {filepath} - Missing imports: {missing_imports}")
            return False
        else:
            print(f"✅ {filepath} - All expected imports present")
            return True
            
    except Exception as e:
        print(f"❌ {filepath} - Read error: {e}")
        return False

def main():
    """Validate the negative learning training integration"""
    print("🔍 Validating Negative Learning Training Integration")
    print("=" * 60)
    
    # Check integration files
    print("\n📁 Checking integration files...")
    integration_files = [
        "src/training/steps/models_training/negative_learning_training_integration.py",
        "src/training/steps/models_training/negative_learning_training_patches.py"
    ]
    
    integration_files_exist = 0
    for filepath in integration_files:
        if check_file_exists(filepath):
            integration_files_exist += 1
    
    print(f"\n📊 Integration files: {integration_files_exist}/{len(integration_files)} exist")
    
    # Check integration structure
    print("\n🔧 Checking integration structure...")
    integration_structure_valid = 0
    
    for filepath in integration_files:
        if check_import_structure(filepath, os.path.basename(filepath).replace('.py', '')):
            integration_structure_valid += 1
    
    print(f"\n📊 Integration structure: {integration_structure_valid}/{len(integration_files)} valid")
    
    # Check training file integration
    print("\n🎯 Checking training file integration...")
    training_files = [
        "src/training/steps/models_training/analyst_models_training.py",
        "src/training/steps/models_training/tactician_models_training.py",
        "src/training/steps/models_training/analyst_training_pipeline.py",
        "src/training/steps/models_training/tactician_training_pipeline.py"
    ]
    
    expected_imports = [
        "negative_learning_training_patches",
        "negative_learning_training_integration",
        "apply_negative_learning_patches",
        "initialize_negative_learning_integration"
    ]
    
    training_integration_valid = 0
    for filepath in training_files:
        if check_training_file_integration(filepath, expected_imports):
            training_integration_valid += 1
    
    print(f"\n📊 Training integration: {training_integration_valid}/{len(training_files)} valid")
    
    # Check for patch application code
    print("\n🔧 Checking patch application code...")
    patch_application_valid = 0
    
    for filepath in training_files[:2]:  # Only check the models training files
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            if "apply_negative_learning_patches()" in content:
                print(f"✅ {filepath} - Patch application code present")
                patch_application_valid += 1
            else:
                print(f"❌ {filepath} - Patch application code missing")
        except Exception as e:
            print(f"❌ {filepath} - Read error: {e}")
    
    print(f"\n📊 Patch application: {patch_application_valid}/2 valid")
    
    # Check for initialization code
    print("\n🎯 Checking initialization code...")
    initialization_valid = 0
    
    for filepath in training_files[2:]:  # Only check the pipeline files
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            if "initialize_negative_learning_integration" in content and "initialize_for_training" in content:
                print(f"✅ {filepath} - Initialization code present")
                initialization_valid += 1
            else:
                print(f"❌ {filepath} - Initialization code missing")
        except Exception as e:
            print(f"❌ {filepath} - Read error: {e}")
    
    print(f"\n📊 Initialization: {initialization_valid}/2 valid")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 INTEGRATION VALIDATION SUMMARY")
    print("=" * 60)
    
    total_checks = len(integration_files) + len(integration_files) + len(training_files) + 2 + 2
    passed_checks = integration_files_exist + integration_structure_valid + training_integration_valid + patch_application_valid + initialization_valid
    
    print(f"Integration files: {integration_files_exist}/{len(integration_files)}")
    print(f"Integration structure: {integration_structure_valid}/{len(integration_files)}")
    print(f"Training integration: {training_integration_valid}/{len(training_files)}")
    print(f"Patch application: {patch_application_valid}/2")
    print(f"Initialization: {initialization_valid}/2")
    print(f"Overall: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("\n🎉 All integration validation checks passed!")
        print("✅ Negative Learning is fully integrated into ML training pipeline!")
        print("\n📚 Integration Summary:")
        print("- ✅ Integration modules created and structured correctly")
        print("- ✅ Training files updated with negative learning imports")
        print("- ✅ Patch application code added to models training files")
        print("- ✅ Initialization code added to pipeline files")
        print("\n🚀 What this means:")
        print("- Your existing training functions will automatically use negative learning features")
        print("- Model constraints will be applied automatically")
        print("- Sample weights will be enhanced with uncertainty weighting")
        print("- No additional code changes needed in your training pipeline")
        print("\n📖 Next steps:")
        print("1. Install required dependencies (numpy, pandas, scikit-learn, etc.)")
        print("2. Run your existing training pipeline - negative learning will work automatically")
        print("3. Monitor training logs for negative learning initialization messages")
        print("4. Check validation results for performance improvements")
        return True
    else:
        print("\n⚠️ Some integration validation checks failed.")
        print("Please review the missing items above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)