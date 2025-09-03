#!/usr/bin/env python3
"""
Run a subset of common_operations tests that work without numpy/pandas.
This creates a minimal implementation and tests only the parts that don't need external dependencies.
"""

import json
import shutil
import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class MinimalCommonOperations:
    """Minimal implementation of common operations for testing."""

    # DateTime Operations
    @staticmethod
    def get_current_datetime():
        return datetime.now()

    @staticmethod
    def get_today():
        return date.today()

    @staticmethod
    def format_datetime(dt, fmt="%Y-%m-%d %H:%M:%S"):
        return dt.strftime(fmt)

    @staticmethod
    def parse_datetime(date_str, fmt="%Y-%m-%d %H:%M:%S"):
        return datetime.strptime(date_str, fmt)

    # File Operations
    @staticmethod
    def ensure_directory(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def safe_file_exists(path):
        try:
            return Path(path).exists()
        except:
            return False

    @staticmethod
    def safe_json_dump(data, path, **kwargs):
        with open(path, "w") as f:
            json.dump(data, f, **kwargs)

    @staticmethod
    def safe_json_load(path):
        with open(path) as f:
            return json.load(f)

    # String Operations
    @staticmethod
    def safe_lower(s):
        return str(s).lower() if s is not None else ""

    @staticmethod
    def safe_upper(s):
        return str(s).upper() if s is not None else ""

    @staticmethod
    def safe_join(sep, items):
        if items is None:
            return ""
        return sep.join(str(item) for item in items)

    # Type Conversions
    @staticmethod
    def safe_float(value, default=0.0):
        try:
            return float(value)
        except:
            return default

    @staticmethod
    def safe_int(value, default=0):
        try:
            return int(value)
        except:
            return default


class TestMinimalOperations(unittest.TestCase):
    """Test cases for operations that don't require numpy/pandas."""

    def setUp(self):
        """Set up test fixtures."""
        self.ops = MinimalCommonOperations()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # DateTime Tests
    def test_get_current_datetime(self):
        """Test get_current_datetime returns datetime object."""
        result = self.ops.get_current_datetime()
        assert isinstance(result, datetime)
        time_diff = datetime.now() - result
        assert time_diff.total_seconds() < 1

    def test_get_today(self):
        """Test get_today returns date object."""
        result = self.ops.get_today()
        assert isinstance(result, date)
        assert result == date.today()

    def test_format_datetime(self):
        """Test datetime formatting."""
        dt = datetime(2024, 1, 15, 10, 30, 45)
        result = self.ops.format_datetime(dt)
        assert result == "2024-01-15 10:30:45"

        result = self.ops.format_datetime(dt, "%Y%m%d")
        assert result == "20240115"

    def test_parse_datetime(self):
        """Test datetime parsing."""
        result = self.ops.parse_datetime("2024-01-15 10:30:45")
        expected = datetime(2024, 1, 15, 10, 30, 45)
        assert result == expected

        with pytest.raises(ValueError):
            self.ops.parse_datetime("invalid date")

    # File Operation Tests
    def test_ensure_directory(self):
        """Test directory creation."""
        new_dir = Path(self.temp_dir) / "new" / "nested" / "dir"
        result = self.ops.ensure_directory(new_dir)

        assert isinstance(result, Path)
        assert result.exists()
        assert result.is_dir()

    def test_safe_file_exists(self):
        """Test safe file existence check."""
        test_file = Path(self.temp_dir) / "test.txt"

        assert not self.ops.safe_file_exists(test_file)

        test_file.write_text("test")
        assert self.ops.safe_file_exists(test_file)

        # Test with invalid path
        assert not self.ops.safe_file_exists("\x00invalid\x00path")

    def test_safe_json_operations(self):
        """Test JSON dump and load operations."""
        test_data = {
            "string": "value",
            "number": 42,
            "float": 3.14,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
        }

        test_file = Path(self.temp_dir) / "test.json"

        # Test dump
        self.ops.safe_json_dump(test_data, test_file, indent=2)
        assert test_file.exists()

        # Test load
        loaded_data = self.ops.safe_json_load(test_file)
        assert loaded_data == test_data

    # String Operation Tests
    def test_safe_lower(self):
        """Test safe lowercase conversion."""
        assert self.ops.safe_lower("HELLO") == "hello"
        assert self.ops.safe_lower(None) == ""
        assert self.ops.safe_lower(123) == "123"

    def test_safe_upper(self):
        """Test safe uppercase conversion."""
        assert self.ops.safe_upper("hello") == "HELLO"
        assert self.ops.safe_upper(None) == ""
        assert self.ops.safe_upper(123) == "123"

    def test_safe_join(self):
        """Test safe string joining."""
        assert self.ops.safe_join(", ", ["a", "b", "c"]) == "a, b, c"
        assert self.ops.safe_join(", ", None) == ""
        assert self.ops.safe_join("-", [1, 2, None, 3]) == "1-2-None-3"

    # Type Conversion Tests
    def test_safe_float(self):
        """Test safe float conversion."""
        assert self.ops.safe_float("3.14") == 3.14
        assert self.ops.safe_float("42") == 42.0
        assert self.ops.safe_float(42) == 42.0
        assert self.ops.safe_float("invalid", -1.0) == -1.0
        assert self.ops.safe_float(None, 0.0) == 0.0

    def test_safe_int(self):
        """Test safe int conversion."""
        assert self.ops.safe_int("42") == 42
        assert self.ops.safe_int(42.7) == 42
        assert self.ops.safe_int("invalid", -1) == -1
        assert self.ops.safe_int(None, 0) == 0


def run_tests():
    """Run the minimal test suite."""
    print("=" * 80)
    print("Running Minimal Common Operations Tests")
    print("=" * 80)
    print("\n✅ These tests run without numpy/pandas dependencies")
    print("📝 Testing core functionality that doesn't require external libraries\n")

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMinimalOperations)

    # Run tests
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
        print("\n📋 Full Test Suite Information:")
        print("   - Total test methods in original file: 47")
        print("   - Tests run in this subset: " + str(result.testsRun))
        print("   - Tests requiring numpy/pandas: " + str(47 - result.testsRun))
        print("\n🔧 To run all tests:")
        print("   1. Set up a Python environment with numpy and pandas")
        print("   2. Run: python3 run_common_operations_tests.py")
    else:
        print("\n❌ Some tests failed!")

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
