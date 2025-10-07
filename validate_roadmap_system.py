#!/usr/bin/env python3
"""
Simple validation script for End-to-End Roadmap System

This script validates the structure and imports of the roadmap system
without requiring external dependencies.
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and report status."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (NOT FOUND)")
        return False

def check_directory_structure():
    """Check the directory structure of the roadmap system."""
    print("=== Checking Directory Structure ===")
    
    required_files = [
        ("config/end_to_end_roadmap_config.yaml", "System Configuration"),
        ("src/end_to_end_roadmap.py", "Main Integration File"),
        ("src/feature_engineering/data_contracts.py", "Data Contracts"),
        ("src/feature_engineering/feature_registry.py", "Feature Registry"),
        ("src/feature_engineering/transforms.py", "Transform System"),
        ("src/feature_engineering/lookback_selection.py", "Lookback Selection"),
        ("src/feature_engineering/interactions.py", "Interaction Engine"),
        ("src/feature_engineering/assembly_dag.py", "Assembly DAG"),
        ("src/models/patch_gru.py", "Patch/GRU Model"),
        ("src/validation/walkforward_validation.py", "Validation System"),
        ("src/monitoring/retrain_monitoring.py", "Monitoring System"),
        ("src/ci/validators.py", "CI/CD Validators"),
        ("src/deployment/rollout_plan.py", "Rollout Plan"),
        ("src/training/steps/pre_training/end_to_end_roadmap_generation/end_to_end_roadmap_component.py", "Roadmap Component"),
        ("END_TO_END_ROADMAP_README.md", "Documentation")
    ]
    
    passed = 0
    total = len(required_files)
    
    for filepath, description in required_files:
        if check_file_exists(filepath, description):
            passed += 1
    
    print(f"\nDirectory Structure: {passed}/{total} files found")
    return passed == total

def check_import_structure():
    """Check the import structure of Python files."""
    print("\n=== Checking Import Structure ===")
    
    python_files = [
        "src/end_to_end_roadmap.py",
        "src/feature_engineering/data_contracts.py",
        "src/feature_engineering/feature_registry.py",
        "src/feature_engineering/transforms.py",
        "src/feature_engineering/lookback_selection.py",
        "src/feature_engineering/interactions.py",
        "src/feature_engineering/assembly_dag.py",
        "src/models/patch_gru.py",
        "src/validation/walkforward_validation.py",
        "src/monitoring/retrain_monitoring.py",
        "src/ci/validators.py",
        "src/deployment/rollout_plan.py"
    ]
    
    passed = 0
    total = len(python_files)
    
    for filepath in python_files:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                
                # Check for basic Python syntax
                if 'class ' in content and 'def ' in content:
                    print(f"✅ {filepath}: Valid Python structure")
                    passed += 1
                else:
                    print(f"⚠️ {filepath}: Missing class/function definitions")
            except Exception as e:
                print(f"❌ {filepath}: Error reading file - {e}")
        else:
            print(f"❌ {filepath}: File not found")
    
    print(f"\nImport Structure: {passed}/{total} files valid")
    return passed == total

def check_configuration():
    """Check configuration files."""
    print("\n=== Checking Configuration ===")
    
    config_file = "config/end_to_end_roadmap_config.yaml"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                content = f.read()
            
            # Check for key configuration sections
            required_sections = [
                'system:',
                'menus:',
                'interactions_locked:',
                'model:',
                'labeling:',
                'calendar:',
                'transforms:',
                'monitoring:',
                'feature_gates:'
            ]
            
            found_sections = 0
            for section in required_sections:
                if section in content:
                    found_sections += 1
                    print(f"✅ Found section: {section}")
                else:
                    print(f"❌ Missing section: {section}")
            
            print(f"\nConfiguration: {found_sections}/{len(required_sections)} sections found")
            return found_sections == len(required_sections)
            
        except Exception as e:
            print(f"❌ Error reading config file: {e}")
            return False
    else:
        print(f"❌ Config file not found: {config_file}")
        return False

def check_documentation():
    """Check documentation completeness."""
    print("\n=== Checking Documentation ===")
    
    readme_file = "END_TO_END_ROADMAP_README.md"
    if os.path.exists(readme_file):
        try:
            with open(readme_file, 'r') as f:
                content = f.read()
            
            # Check for key documentation sections
            required_sections = [
                '# End-to-End Roadmap System',
                '## System Architecture',
                '## Feature Families',
                '## Transform Types',
                '## Interaction Engine',
                '## Usage',
                '## Validation',
                '## Monitoring',
                '## Deployment',
                '## CI/CD'
            ]
            
            found_sections = 0
            for section in required_sections:
                if section in content:
                    found_sections += 1
                    print(f"✅ Found section: {section}")
                else:
                    print(f"❌ Missing section: {section}")
            
            print(f"\nDocumentation: {found_sections}/{len(required_sections)} sections found")
            return found_sections == len(required_sections)
            
        except Exception as e:
            print(f"❌ Error reading documentation: {e}")
            return False
    else:
        print(f"❌ Documentation not found: {readme_file}")
        return False

def check_component_integration():
    """Check component integration structure."""
    print("\n=== Checking Component Integration ===")
    
    component_file = "src/training/steps/pre_training/end_to_end_roadmap_generation/end_to_end_roadmap_component.py"
    if os.path.exists(component_file):
        try:
            with open(component_file, 'r') as f:
                content = f.read()
            
            # Check for key component features
            required_features = [
                'class EndToEndRoadmapComponent',
                'async def execute',
                'def _load_and_validate_market_data',
                'def _get_target_variable',
                'def _validate_generation_results',
                'def _create_comprehensive_artifacts'
            ]
            
            found_features = 0
            for feature in required_features:
                if feature in content:
                    found_features += 1
                    print(f"✅ Found feature: {feature}")
                else:
                    print(f"❌ Missing feature: {feature}")
            
            print(f"\nComponent Integration: {found_features}/{len(required_features)} features found")
            return found_features == len(required_features)
            
        except Exception as e:
            print(f"❌ Error reading component file: {e}")
            return False
    else:
        print(f"❌ Component file not found: {component_file}")
        return False

def main():
    """Run all validation checks."""
    print("End-to-End Roadmap System Validation")
    print("=" * 50)
    
    checks = [
        ("Directory Structure", check_directory_structure),
        ("Import Structure", check_import_structure),
        ("Configuration", check_configuration),
        ("Documentation", check_documentation),
        ("Component Integration", check_component_integration)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        try:
            if check_func():
                passed += 1
                print(f"✅ {check_name} PASSED")
            else:
                print(f"❌ {check_name} FAILED")
        except Exception as e:
            print(f"❌ {check_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Validation Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All validations passed! The end-to-end roadmap system is properly structured.")
        print("\nNext steps:")
        print("1. Install required dependencies (pandas, numpy, scikit-learn, etc.)")
        print("2. Run the full test suite: python3 test_end_to_end_roadmap.py")
        print("3. Integrate the component into your training pipeline")
        return 0
    else:
        print("⚠️ Some validations failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())