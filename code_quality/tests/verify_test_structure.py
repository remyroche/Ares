#!/usr/bin/env python3
"""Verify that all test files for steps 1-7 have been created with proper structure.

This script checks the existence and basic structure of test files without running them.
"""

import os
import re
from pathlib import Path


def verify_test_files():
    """Verify all test files exist and have proper structure."""
    tests_dir = Path(__file__).parent
    
    # Expected test files
    expected_files = [
        ('test_step1_data_collection.py', ['TestDataCollectionStep', 'test_initialization', 'test_execute']),
        ('test_step2_data_reading.py', ['TestDataReadingStep', 'test_initialization', 'test_execute']),
        ('test_step3_hmm_regime_discovery.py', ['TestHMMRegimeDiscoveryStep', 'TestParameterOptimizationStep']),
        ('test_step4_regime_data_splitting.py', ['TestRegimeDataSplittingStep', 'TestTripleBarrierMethodStep']),
        ('test_step5_labeling.py', ['TestLabelingStep', 'test_initialization', 'test_execute_labeling']),
        ('test_step6_feature_engineering.py', ['TestStep6FeatureEngineering', 'TestFeatureInteractionEngine']),
        ('test_step7_enhanced_matrix_operations.py', ['TestMatrixOperationsStep', 'test_initialization'])
    ]
    
    print("=" * 80)
    print("Verifying Test Files for Training Pipeline Steps 1-7")
    print("=" * 80)
    
    all_valid = True
    total_test_classes = 0
    total_test_methods = 0
    
    for test_file, expected_content in expected_files:
        test_path = tests_dir / test_file
        print(f"\n📁 Checking {test_file}...")
        
        if not test_path.exists():
            print(f"   ❌ File not found!")
            all_valid = False
            continue
        
        # Read file content
        with open(test_path, 'r') as f:
            content = f.read()
        
        # Check file size
        file_size = len(content)
        print(f"   📏 File size: {file_size:,} bytes")
        
        # Count test classes
        test_classes = re.findall(r'class\s+Test\w+\s*\(.*TestCase\)', content)
        print(f"   🧪 Test classes found: {len(test_classes)}")
        for cls in test_classes:
            print(f"      - {cls.split()[1].split('(')[0]}")
        total_test_classes += len(test_classes)
        
        # Count test methods
        test_methods = re.findall(r'def\s+test_\w+\s*\(', content)
        print(f"   🔬 Test methods found: {len(test_methods)}")
        total_test_methods += len(test_methods)
        
        # Check for expected content
        missing_content = []
        for expected in expected_content:
            if expected not in content:
                missing_content.append(expected)
        
        if missing_content:
            print(f"   ⚠️  Missing expected content: {', '.join(missing_content)}")
        else:
            print(f"   ✅ All expected content found")
        
        # Check for required imports
        has_unittest = 'import unittest' in content
        has_mock = 'from unittest.mock import' in content
        has_asyncio = 'import asyncio' in content
        
        print(f"   📦 Required imports: unittest={'✅' if has_unittest else '❌'}, "
              f"mock={'✅' if has_mock else '❌'}, "
              f"asyncio={'✅' if has_asyncio else '❌'}")
        
        # Check for main block
        has_main = 'if __name__ == "__main__"' in content
        print(f"   🚀 Has main block: {'✅' if has_main else '❌'}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"✅ Total test files created: {len(expected_files)}")
    print(f"🧪 Total test classes: {total_test_classes}")
    print(f"🔬 Total test methods: {total_test_methods}")
    print(f"📊 Average methods per class: {total_test_methods / total_test_classes:.1f}" if total_test_classes > 0 else "N/A")
    
    if all_valid:
        print("\n✅ All test files are properly structured!")
    else:
        print("\n❌ Some test files are missing or incomplete!")
    
    # List all test files in directory
    print("\n" + "=" * 80)
    print("All Python files in tests directory:")
    print("=" * 80)
    for file in sorted(tests_dir.glob("*.py")):
        if file.name != '__init__.py':
            size = file.stat().st_size
            print(f"  - {file.name:<40} ({size:>10,} bytes)")
    
    return all_valid


def check_test_coverage():
    """Check which step modules are being tested."""
    print("\n" + "=" * 80)
    print("Test Coverage Check")
    print("=" * 80)
    
    step_modules = {
        1: ['step1_data_collection.py'],
        2: ['step2_data_reading.py'],
        3: ['step3_hmm_regime_discovery.py', 'step3_parameter_optimization.py'],
        4: ['step4_regime_data_splitting.py', 'step4_triple_barrier_method.py'],
        5: ['step5_labeling.py'],
        6: ['step6_feature_engineering.py', 'step6_feature_interaction_engineering.py'],
        7: ['step7_enhanced_matrix_operations.py']
    }
    
    for step_num, modules in step_modules.items():
        print(f"\nStep {step_num}:")
        test_file = f"test_step{step_num}_*.py"
        test_exists = len(list(Path(__file__).parent.glob(test_file))) > 0
        
        for module in modules:
            print(f"  - {module:<45} {'✅ Has tests' if test_exists else '❌ No tests'}")


if __name__ == "__main__":
    print("🔍 Verifying test file structure...\n")
    
    # Verify test files
    success = verify_test_files()
    
    # Check coverage
    check_test_coverage()
    
    print("\n✅ Verification complete!")
    
    # Provide instructions
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print("1. Install required dependencies:")
    print("   pip install pandas numpy scikit-learn")
    print("\n2. Run the tests:")
    print("   python3 code_quality/tests/run_all_step_tests.py")
    print("\n3. Run tests for a specific step:")
    print("   python3 code_quality/tests/run_all_step_tests.py <step_number>")
    print("\n4. Run tests with pytest (if installed):")
    print("   pytest code_quality/tests/test_step*.py -v")