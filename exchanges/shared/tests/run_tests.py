#!/usr/bin/env python3
"""
Test runner for shared exchange utilities.

This script runs all unit tests for the shared exchange utilities.
"""

import sys
import subprocess
import os
from pathlib import Path


def run_tests():
    """Run all tests for the shared exchange utilities."""
    # Get the directory containing this script
    test_dir = Path(__file__).parent
    
    # Change to the test directory
    os.chdir(test_dir)
    
    # Run pytest with appropriate options
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--asyncio-mode=auto",  # Auto-detect async tests
        "--disable-warnings",  # Disable warnings for cleaner output
        "."
    ]
    
    print("Running tests for shared exchange utilities...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        return 0
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 50)
        print("❌ Some tests failed!")
        return e.returncode
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        return 1


def run_specific_test(test_file):
    """Run a specific test file."""
    test_dir = Path(__file__).parent
    os.chdir(test_dir)
    
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
        "--disable-warnings",
        test_file
    ]
    
    print(f"Running specific test: {test_file}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 50)
        print("✅ Test passed!")
        return 0
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 50)
        print("❌ Test failed!")
        return e.returncode


def run_coverage():
    """Run tests with coverage reporting."""
    test_dir = Path(__file__).parent
    os.chdir(test_dir)
    
    cmd = [
        sys.executable, "-m", "pytest",
        "--cov=..",  # Coverage for parent directory
        "--cov-report=html",  # HTML coverage report
        "--cov-report=term-missing",  # Terminal coverage report
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
        "--disable-warnings",
        "."
    ]
    
    print("Running tests with coverage...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 50)
        print("✅ Tests with coverage completed!")
        print("📊 HTML coverage report generated in htmlcov/")
        return 0
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 50)
        print("❌ Tests failed!")
        return e.returncode


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--coverage":
            return run_coverage()
        elif sys.argv[1].endswith(".py"):
            return run_specific_test(sys.argv[1])
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python run_tests.py [--coverage] [test_file.py]")
            return 1
    else:
        return run_tests()


if __name__ == "__main__":
    sys.exit(main())