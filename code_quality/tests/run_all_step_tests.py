#!/usr/bin/env python3
"""Run all unit tests for training pipeline steps 1-7.

This script discovers and runs all test files for the training pipeline steps.
"""

import sys
import unittest
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def run_all_tests():
    """Discover and run all tests in the tests directory."""
    # Get the tests directory
    tests_dir = Path(__file__).parent

    # Create test loader
    loader = unittest.TestLoader()

    # Discover all test files for steps 1-7
    test_files = [
        "test_step1_data_collection.py",
        "test_step2_data_reading.py",
        "test_step3_hmm_regime_discovery.py",
        "test_step4_regime_data_splitting.py",
        "test_step5_labeling.py",
        "test_step6_feature_engineering.py",
        "test_step7_enhanced_matrix_operations.py",
    ]

    # Create test suite
    suite = unittest.TestSuite()

    print("=" * 80)
    print("Running Unit Tests for Training Pipeline Steps 1-7")
    print("=" * 80)

    # Load tests from each file
    for test_file in test_files:
        test_path = tests_dir / test_file
        if test_path.exists():
            print(f"\n📁 Loading tests from {test_file}...")
            try:
                # Import the test module
                module_name = test_file[:-3]  # Remove .py extension
                test_module = __import__(module_name, fromlist=[""])

                # Load tests from the module
                module_tests = loader.loadTestsFromModule(test_module)
                suite.addTests(module_tests)

                # Count tests
                test_count = module_tests.countTestCases()
                print(f"   ✅ Loaded {test_count} tests from {test_file}")

            except Exception as e:
                print(f"   ❌ Error loading tests from {test_file}: {e}")
        else:
            print(f"   ⚠️  Test file not found: {test_file}")

    # Run the tests
    print("\n" + "=" * 80)
    print("Running Tests...")
    print("=" * 80 + "\n")

    # Create test runner with verbosity
    runner = unittest.TextTestRunner(verbosity=2)

    # Run the test suite
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Total tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\n❌ Failed Tests:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(chr(10))[0]}")

    if result.errors:
        print("\n💥 Test Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split(chr(10))[0]}")

    # Return success/failure
    return result.wasSuccessful()


def run_specific_step_tests(step_number):
    """Run tests for a specific step."""
    test_file_map = {
        1: "test_step1_data_collection.py",
        2: "test_step2_data_reading.py",
        3: "test_step3_hmm_regime_discovery.py",
        4: "test_step4_regime_data_splitting.py",
        5: "test_step5_labeling.py",
        6: "test_step6_feature_engineering.py",
        7: "test_step7_enhanced_matrix_operations.py",
    }

    if step_number not in test_file_map:
        print(f"❌ Invalid step number: {step_number}. Must be between 1 and 7.")
        return False

    test_file = test_file_map[step_number]
    tests_dir = Path(__file__).parent
    test_path = tests_dir / test_file

    if not test_path.exists():
        print(f"❌ Test file not found: {test_file}")
        return False

    print(f"Running tests for Step {step_number} ({test_file})...")

    # Create test loader and runner
    loader = unittest.TestLoader()
    runner = unittest.TextTestRunner(verbosity=2)

    # Load and run tests
    try:
        module_name = test_file[:-3]
        test_module = __import__(module_name, fromlist=[""])
        suite = loader.loadTestsFromModule(test_module)
        result = runner.run(suite)
        return result.wasSuccessful()
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        try:
            step_num = int(sys.argv[1])
            success = run_specific_step_tests(step_num)
        except ValueError:
            print("Usage: python run_all_step_tests.py [step_number]")
            print("  step_number: Optional, 1-7 to run tests for specific step")
            success = False
    else:
        # Run all tests
        success = run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
