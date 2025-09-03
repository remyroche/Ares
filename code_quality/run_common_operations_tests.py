#!/usr/bin/env python3
"""
Run comprehensive tests for common_operations module.

This script runs all unit tests and generates a coverage report.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_tests():
    """Run all common_operations tests."""
    print("=" * 80)
    print("Running Common Operations Tests")
    print("=" * 80)

    # Discover and run tests
    loader = unittest.TestLoader()
    # Use relative path from this file's location
    start_dir = Path(__file__).parent / "tests"
    suite = loader.discover(str(start_dir), pattern="test_common_operations.py")

    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return 0
    print("\n❌ Some tests failed!")
    if result.failures:
        print("\nFailures:")
        for test, _traceback in result.failures:
            print(f"  - {test}")
    if result.errors:
        print("\nErrors:")
        for test, _traceback in result.errors:
            print(f"  - {test}")
    return 1

def run_with_coverage():
    """Run tests with coverage reporting."""
    try:
        import coverage
    except ImportError:
        print("Coverage module not installed. Install with: pip install coverage")
        print("Running tests without coverage...")
        return run_tests()

    print("Running tests with coverage...")

    # Start coverage
    cov = coverage.Coverage()
    cov.start()

    # Run tests
    exit_code = run_tests()

    # Stop coverage and generate report
    cov.stop()
    cov.save()

    print("\n" + "=" * 80)
    print("Coverage Report")
    print("=" * 80)

    # Print coverage report
    cov.report(include=["src/utils/common_operations.py"])

    # Generate HTML report
    print("\nGenerating HTML coverage report...")
    cov.html_report(include=["src/utils/common_operations.py"], directory="coverage_html")
    print("HTML coverage report generated in: coverage_html/")

    return exit_code

if __name__ == "__main__":
    # Check if coverage is requested
    if "--coverage" in sys.argv:
        exit_code = run_with_coverage()
    else:
        exit_code = run_tests()
        print("\nTip: Run with --coverage flag to get coverage report")

    sys.exit(exit_code)
