#!/usr/bin/env python3
from src.utils.tprint import tprint

"""Verify that all test files for steps 1-7 have been created with proper structure.

This script checks the existence and basic structure of test files without running them.
"""

import re
from pathlib import Path


def verify_test_files():
    """Verify all test files exist and have proper structure."""
    tests_dir = Path(__file__).parent

    # Expected test files
    expected_files = [
        ("test_step1_data_collection.py", ["TestDataCollectionStep", "test_initialization", "test_execute"]),
        ("test_step2_data_reading.py", ["TestDataReadingStep", "test_initialization", "test_execute"]),
        ("test_step3_hmm_regime_discovery.py", ["TestHMMRegimeDiscoveryStep", "TestParameterOptimizationStep"]),
        ("test_step4_regime_data_splitting.py", ["TestRegimeDataSplittingStep", "TestTripleBarrierMethodStep"]),
        ("test_step5_labeling.py", ["TestLabelingStep", "test_initialization", "test_execute_labeling"]),
        ("test_step6_feature_engineering.py", ["TestStep6FeatureEngineering", "TestFeatureInteractionEngine"]),
        ("test_step7_enhanced_matrix_operations.py", ["TestMatrixOperationsStep", "test_initialization"]),
    ]

    tprint("=" * 80)
    tprint("Verifying Test Files for Training Pipeline Steps 1-7")
    tprint("=" * 80)

    all_valid = True
    total_test_classes = 0
    total_test_methods = 0

    for test_file, expected_content in expected_files:
        test_path = tests_dir / test_file
        tprint(f"\n📁 Checking {test_file}...")

        if not test_path.exists():
            tprint("   ❌ File not found!")
            all_valid = False
            continue

        # Read file content
        with open(test_path) as f:
            content = f.read()

        # Check file size
        file_size = len(content)
        tprint(f"   📏 File size: {file_size:,} bytes")

        # Count test classes
        test_classes = re.findall(r"class\s+Test\w+\s*\(.*TestCase\)", content)
        tprint(f"   🧪 Test classes found: {len(test_classes)}")
        for cls in test_classes:
            tprint(f"      - {cls.split()[1].split('(')[0]}")
        total_test_classes += len(test_classes)

        # Count test methods
        test_methods = re.findall(r"def\s+test_\w+\s*\(", content)
        tprint(f"   🔬 Test methods found: {len(test_methods)}")
        total_test_methods += len(test_methods)

        # Check for expected content
        missing_content = []
        for expected in expected_content:
            if expected not in content:
                missing_content.append(expected)

        if missing_content:
            tprint(f"   ⚠️  Missing expected content: {', '.join(missing_content)}")
        else:
            tprint("   ✅ All expected content found")

        # Check for required imports
        has_unittest = "import unittest" in content
        has_mock = "from unittest.mock import" in content
        has_asyncio = "import asyncio" in content

        tprint(f"   📦 Required imports: unittest={'✅' if has_unittest else '❌'}, "
              f"mock={'✅' if has_mock else '❌'}, "
              f"asyncio={'✅' if has_asyncio else '❌'}")

        # Check for main block
        has_main = 'if __name__ == "__main__"' in content
        tprint(f"   🚀 Has main block: {'✅' if has_main else '❌'}")

    # Summary
    tprint("\n" + "=" * 80)
    tprint("Summary")
    tprint("=" * 80)
    tprint(f"✅ Total test files created: {len(expected_files)}")
    tprint(f"🧪 Total test classes: {total_test_classes}")
    tprint(f"🔬 Total test methods: {total_test_methods}")
    tprint(f"📊 Average methods per class: {total_test_methods / total_test_classes:.1f}" if total_test_classes > 0 else "N/A")

    if all_valid:
        tprint("\n✅ All test files are properly structured!")
    else:
        tprint("\n❌ Some test files are missing or incomplete!")

    # List all test files in directory
    tprint("\n" + "=" * 80)
    tprint("All Python files in tests directory:")
    tprint("=" * 80)
    for file in sorted(tests_dir.glob("*.py")):
        if file.name != "__init__.py":
            size = file.stat().st_size
            tprint(f"  - {file.name:<40} ({size:>10,} bytes)")

    return all_valid


def check_test_coverage():
    """Check which step modules are being tested."""
    tprint("\n" + "=" * 80)
    tprint("Test Coverage Check")
    tprint("=" * 80)

    step_modules = {
        1: ["step1_data_collection.py"],
        2: ["step2_data_reading.py"],
        3: ["step3_hmm_regime_discovery.py", "step3_parameter_optimization.py"],
        4: ["step4_regime_data_splitting.py", "step4_triple_barrier_method.py"],
        5: ["step5_labeling.py"],
        6: ["step6_feature_engineering.py", "step6_feature_interaction_engineering.py"],
        7: ["step7_enhanced_matrix_operations.py"],
    }

    for step_num, modules in step_modules.items():
        tprint(f"\nStep {step_num}:")
        test_file = f"test_step{step_num}_*.py"
        test_exists = len(list(Path(__file__).parent.glob(test_file))) > 0

        for module in modules:
            tprint(f"  - {module:<45} {'✅ Has tests' if test_exists else '❌ No tests'}")


if __name__ == "__main__":
    tprint("🔍 Verifying test file structure...\n")

    # Verify test files
    success = verify_test_files()

    # Check coverage
    check_test_coverage()

    tprint("\n✅ Verification complete!")

    # Provide instructions
    tprint("\n" + "=" * 80)
    tprint("Next Steps:")
    tprint("=" * 80)
    tprint("1. Install required dependencies:")
    tprint("   pip install pandas numpy scikit-learn")
    tprint("\n2. Run the tests:")
    tprint("   python3 code_quality/tests/run_all_step_tests.py")
    tprint("\n3. Run tests for a specific step:")
    tprint("   python3 code_quality/tests/run_all_step_tests.py <step_number>")
    tprint("\n4. Run tests with pytest (if installed):")
    tprint("   pytest code_quality/tests/test_step*.py -v")
