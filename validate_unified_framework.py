#!/usr/bin/env python3
"""
Validation Script for Unified Feature Selection Framework

This script validates the structure and imports of the unified framework
without requiring external dependencies like numpy or sklearn.

Author: AI Assistant
Date: 2024-01-XX
Version: 1.0.0
"""

import sys
import os
from pathlib import Path

def validate_file_structure():
    """Validate that all required files exist."""
    print("🔍 Validating file structure...")
    
    required_files = [
        "src/utils/ml_common/unified_feature_selection.py",
        "src/utils/ml_common/matrix_feature_operations.py", 
        "src/utils/ml_common/backwards_compatibility.py",
        "unified_feature_selection_demo.py",
        "test_unified_feature_selection.py",
        "UNIFIED_FEATURE_SELECTION_README.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"  ✅ {file_path}")
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    
    print("  ✅ All required files exist")
    return True

def validate_imports():
    """Validate that imports work correctly."""
    print("\n🔍 Validating imports...")
    
    # Add src to path
    sys.path.insert(0, str(Path.cwd() / "src"))
    
    try:
        # Test unified feature selection imports
        print("  Testing unified_feature_selection imports...")
        from utils.ml_common.unified_feature_selection import (
            UnifiedFeatureSelector, UnifiedFeatureSelectionConfig,
            create_unified_selector, select_features_unified, generate_feature_sets
        )
        print("    ✅ UnifiedFeatureSelector imported successfully")
        print("    ✅ UnifiedFeatureSelectionConfig imported successfully")
        print("    ✅ Convenience functions imported successfully")
        
        # Test matrix operations imports
        print("  Testing matrix_feature_operations imports...")
        from utils.ml_common.matrix_feature_operations import (
            MatrixFeatureOperations, create_matrix_feature_operations
        )
        print("    ✅ MatrixFeatureOperations imported successfully")
        print("    ✅ create_matrix_feature_operations imported successfully")
        
        # Test backwards compatibility imports
        print("  Testing backwards_compatibility imports...")
        from utils.ml_common.backwards_compatibility import (
            BackwardsCompatibilityWrapper, create_feature_selector
        )
        print("    ✅ BackwardsCompatibilityWrapper imported successfully")
        print("    ✅ create_feature_selector imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"    ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"    ❌ Unexpected error: {e}")
        return False

def validate_class_structure():
    """Validate that classes have required methods."""
    print("\n🔍 Validating class structure...")
    
    try:
        from utils.ml_common.unified_feature_selection import UnifiedFeatureSelector, UnifiedFeatureSelectionConfig
        
        # Test UnifiedFeatureSelector
        print("  Testing UnifiedFeatureSelector...")
        config = UnifiedFeatureSelectionConfig()
        selector = UnifiedFeatureSelector(config)
        
        required_methods = [
            'select_features', 'get_feature_set', 'get_hmm_regime_features',
            'get_feature_scores', '_prepare_data', '_perform_feature_selection'
        ]
        
        for method in required_methods:
            if hasattr(selector, method):
                print(f"    ✅ {method} method exists")
            else:
                print(f"    ❌ {method} method missing")
                return False
        
        # Test UnifiedFeatureSelectionConfig
        print("  Testing UnifiedFeatureSelectionConfig...")
        required_attributes = [
            'target_features', 'task_type', 'prediction_target',
            'primary_method', 'use_matrix_operations'
        ]
        
        for attr in required_attributes:
            if hasattr(config, attr):
                print(f"    ✅ {attr} attribute exists")
            else:
                print(f"    ❌ {attr} attribute missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"    ❌ Error validating class structure: {e}")
        return False

def validate_configuration():
    """Validate configuration options."""
    print("\n🔍 Validating configuration...")
    
    try:
        from utils.ml_common.unified_feature_selection import UnifiedFeatureSelectionConfig
        
        # Test default configuration
        config = UnifiedFeatureSelectionConfig()
        print(f"  ✅ Default target_features: {config.target_features}")
        print(f"  ✅ Default task_type: {config.task_type}")
        print(f"  ✅ Default prediction_target: {config.prediction_target}")
        print(f"  ✅ Default primary_method: {config.primary_method}")
        
        # Test custom configuration
        custom_config = UnifiedFeatureSelectionConfig(
            target_features=100,
            task_type="classification",
            prediction_target="hmm_regime",
            primary_method="hybrid"
        )
        print(f"  ✅ Custom target_features: {custom_config.target_features}")
        print(f"  ✅ Custom task_type: {custom_config.task_type}")
        print(f"  ✅ Custom prediction_target: {custom_config.prediction_target}")
        print(f"  ✅ Custom primary_method: {custom_config.primary_method}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Error validating configuration: {e}")
        return False

def validate_backwards_compatibility():
    """Validate backwards compatibility layer."""
    print("\n🔍 Validating backwards compatibility...")
    
    try:
        from utils.ml_common.backwards_compatibility import (
            BackwardsCompatibilityWrapper, create_feature_selector
        )
        
        # Test legacy interface
        selector = create_feature_selector()
        print("    ✅ Legacy selector created successfully")
        
        # Test required methods
        required_methods = [
            'fit', 'transform', 'fit_transform', 'get_support',
            'get_feature_names_out', 'get_feature_importance'
        ]
        
        for method in required_methods:
            if hasattr(selector, method):
                print(f"    ✅ {method} method exists")
            else:
                print(f"    ❌ {method} method missing")
                return False
        
        # Test properties
        required_properties = ['n_features_in_', 'n_features_out_']
        for prop in required_properties:
            if hasattr(selector, prop):
                print(f"    ✅ {prop} property exists")
            else:
                print(f"    ❌ {prop} property missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"    ❌ Error validating backwards compatibility: {e}")
        return False

def validate_documentation():
    """Validate documentation completeness."""
    print("\n🔍 Validating documentation...")
    
    try:
        readme_path = Path("UNIFIED_FEATURE_SELECTION_README.md")
        if readme_path.exists():
            content = readme_path.read_text()
            
            required_sections = [
                "# Unified Feature Selection Framework",
                "## Overview",
                "## Quick Start",
                "## API Reference",
                "## Examples",
                "## Configuration"
            ]
            
            for section in required_sections:
                if section in content:
                    print(f"    ✅ {section} section exists")
                else:
                    print(f"    ❌ {section} section missing")
                    return False
            
            print("    ✅ Documentation is complete")
            return True
        else:
            print("    ❌ README file not found")
            return False
            
    except Exception as e:
        print(f"    ❌ Error validating documentation: {e}")
        return False

def validate_demo_script():
    """Validate demo script structure."""
    print("\n🔍 Validating demo script...")
    
    try:
        demo_path = Path("unified_feature_selection_demo.py")
        if demo_path.exists():
            content = demo_path.read_text()
            
            required_functions = [
                "def generate_sample_data",
                "def demonstrate_unified_framework",
                "def demonstrate_random_forest_refinement",
                "def demonstrate_hmm_regime_selection",
                "def main"
            ]
            
            for func in required_functions:
                if func in content:
                    print(f"    ✅ {func} function exists")
                else:
                    print(f"    ❌ {func} function missing")
                    return False
            
            print("    ✅ Demo script is complete")
            return True
        else:
            print("    ❌ Demo script not found")
            return False
            
    except Exception as e:
        print(f"    ❌ Error validating demo script: {e}")
        return False

def validate_test_suite():
    """Validate test suite structure."""
    print("\n🔍 Validating test suite...")
    
    try:
        test_path = Path("test_unified_feature_selection.py")
        if test_path.exists():
            content = test_path.read_text()
            
            required_classes = [
                "class TestUnifiedFeatureSelector",
                "class TestMatrixFeatureOperations",
                "class TestBackwardsCompatibility",
                "class TestConvenienceFunctions",
                "class TestErrorHandling",
                "class TestPerformance"
            ]
            
            for test_class in required_classes:
                if test_class in content:
                    print(f"    ✅ {test_class} exists")
                else:
                    print(f"    ❌ {test_class} missing")
                    return False
            
            print("    ✅ Test suite is complete")
            return True
        else:
            print("    ❌ Test suite not found")
            return False
            
    except Exception as e:
        print(f"    ❌ Error validating test suite: {e}")
        return False

def main():
    """Main validation function."""
    print("🎯 Unified Feature Selection Framework Validation")
    print("="*60)
    
    validation_results = []
    
    # Run all validations
    validation_results.append(("File Structure", validate_file_structure()))
    validation_results.append(("Imports", validate_imports()))
    validation_results.append(("Class Structure", validate_class_structure()))
    validation_results.append(("Configuration", validate_configuration()))
    validation_results.append(("Backwards Compatibility", validate_backwards_compatibility()))
    validation_results.append(("Documentation", validate_documentation()))
    validation_results.append(("Demo Script", validate_demo_script()))
    validation_results.append(("Test Suite", validate_test_suite()))
    
    # Print summary
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for test_name, result in validation_results:
        if result:
            print(f"✅ {test_name}: PASSED")
            passed += 1
        else:
            print(f"❌ {test_name}: FAILED")
            failed += 1
    
    print(f"\nTotal: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All validations passed! The unified framework is ready.")
        return True
    else:
        print(f"\n⚠️ {failed} validation(s) failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)