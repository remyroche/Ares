#!/usr/bin/env python3
"""
Run subset of real common_operations tests by mocking only numpy/pandas imports.
This tests the actual implementation for functions that don't need these dependencies.
"""

import asyncio
import shutil
import sys
import tempfile
import types
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock

# Mock numpy and pandas before any imports
numpy_mock = types.ModuleType("numpy")
numpy_mock.nan = float("nan")
numpy_mock.integer = int
numpy_mock.floating = float

pandas_mock = types.ModuleType("pandas")
pandas_mock.DataFrame = MagicMock
pandas_mock.Series = MagicMock
pandas_mock.core = types.ModuleType("core")
pandas_mock.core.window = types.ModuleType("window")
pandas_mock.core.window.Rolling = MagicMock

sys.modules["numpy"] = numpy_mock
sys.modules["pandas"] = pandas_mock

# Now we can import the real module
sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest

from src.utils.common_operations import (
    chunked_iterable,
    create_async_task,
    # File operations
    ensure_directory,
    format_bytes,
    format_datetime,
    generate_cache_key,
    # Hashing operations
    generate_hash,
    # DateTime operations
    get_current_datetime,
    get_today,
    parse_datetime,
    # Collection operations
    safe_append,
    safe_dict_get,
    safe_file_exists,
    # Type conversions
    safe_float,
    safe_int,
    safe_join,
    safe_json_dump,
    safe_json_load,
    # String operations
    safe_lower,
    # Async operations
    safe_sleep,
    safe_upper,
    validate_numeric_range,
)


class TestRealDateTimeOperations(unittest.TestCase):
    """Test real datetime utility functions."""

    def test_get_current_datetime(self):
        """Test get_current_datetime returns datetime object."""
        result = get_current_datetime()
        assert isinstance(result, datetime)
        time_diff = datetime.now() - result
        assert time_diff.total_seconds() < 1

    def test_get_today(self):
        """Test get_today returns date object."""
        result = get_today()
        assert isinstance(result, date)
        assert result == date.today()

    def test_format_datetime(self):
        """Test datetime formatting."""
        dt = datetime(2024, 1, 15, 10, 30, 45)

        result = format_datetime(dt)
        assert result == "2024-01-15 10:30:45"

        result = format_datetime(dt, "%Y%m%d")
        assert result == "20240115"

    def test_parse_datetime(self):
        """Test datetime parsing."""
        result = parse_datetime("2024-01-15 10:30:45")
        expected = datetime(2024, 1, 15, 10, 30, 45)
        assert result == expected

        with pytest.raises(ValueError):
            parse_datetime("invalid date")


class TestRealFileOperations(unittest.TestCase):
    """Test real file operation utilities."""

    def setUp(self):
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_ensure_directory(self):
        """Test directory creation."""
        new_dir = Path(self.temp_dir) / "new" / "nested" / "dir"
        result = ensure_directory(new_dir)

        assert isinstance(result, Path)
        assert result.exists()
        assert result.is_dir()

    def test_safe_file_exists(self):
        """Test safe file existence check."""
        test_file = Path(self.temp_dir) / "test.txt"

        assert not safe_file_exists(test_file)

        test_file.write_text("test")
        assert safe_file_exists(test_file)

    def test_safe_json_operations(self):
        """Test JSON dump and load operations."""
        test_data = {
            "string": "value",
            "number": 42,
            "list": [1, 2, 3],
        }

        test_file = Path(self.temp_dir) / "test.json"

        safe_json_dump(test_data, test_file, indent=2)
        assert test_file.exists()

        loaded_data = safe_json_load(test_file)
        assert loaded_data == test_data


class TestRealStringOperations(unittest.TestCase):
    """Test real string utility functions."""

    def test_safe_lower(self):
        """Test safe lowercase conversion."""
        assert safe_lower("HELLO") == "hello"
        assert safe_lower(None) == ""
        assert safe_lower(123) == "123"

    def test_safe_upper(self):
        """Test safe uppercase conversion."""
        assert safe_upper("hello") == "HELLO"
        assert safe_upper(None) == ""
        assert safe_upper(123) == "123"

    def test_safe_join(self):
        """Test safe string joining."""
        assert safe_join(", ", ["a", "b", "c"]) == "a, b, c"
        assert safe_join(", ", None) == ""


class TestRealTypeConversions(unittest.TestCase):
    """Test real type conversion utilities."""

    def test_safe_float(self):
        """Test safe float conversion."""
        assert safe_float("3.14") == 3.14
        assert safe_float("42") == 42.0
        assert safe_float("invalid", -1.0) == -1.0
        assert safe_float(None, 0.0) == 0.0

    def test_safe_int(self):
        """Test safe int conversion."""
        assert safe_int("42") == 42
        assert safe_int(42.7) == 42
        assert safe_int("invalid", -1) == -1


class TestRealHashingOperations(unittest.TestCase):
    """Test real hashing operations."""

    def test_generate_hash(self):
        """Test hash generation."""
        hash1 = generate_hash("test string")
        assert len(hash1) == 32  # MD5 length

        hash2 = generate_hash("test string")
        assert hash1 == hash2  # Same input, same hash

        hash3 = generate_hash("different")
        assert hash1 != hash3

    def test_generate_cache_key(self):
        """Test cache key generation."""
        key = generate_cache_key("features", "BTCUSDT", "1h")
        assert len(key) == 16  # Default max_length

        key2 = generate_cache_key("features", "BTCUSDT", "1h")
        assert key == key2  # Consistent


class TestRealCollectionOperations(unittest.TestCase):
    """Test real collection operations."""

    def test_safe_append(self):
        """Test safe list append."""
        lst = safe_append(None, "item")
        assert lst == ["item"]

        lst = safe_append(lst, "item2")
        assert lst == ["item", "item2"]

    def test_safe_dict_get(self):
        """Test safe dictionary get."""
        result = safe_dict_get(None, "key", "default")
        assert result == "default"

        d = {"key": "value"}
        result = safe_dict_get(d, "key", "default")
        assert result == "value"


class TestRealUtilityOperations(unittest.TestCase):
    """Test real utility operations."""

    def test_format_bytes(self):
        """Test byte formatting."""
        assert format_bytes(0) == "0.00 B"
        assert format_bytes(1024) == "1.00 KB"
        assert format_bytes(1024 * 1024) == "1.00 MB"

    def test_chunked_iterable(self):
        """Test iterable chunking."""
        items = list(range(10))
        chunks = chunked_iterable(items, 3)
        assert len(chunks) == 4
        assert chunks[0] == [0, 1, 2]
        assert chunks[3] == [9]

    def test_validate_numeric_range(self):
        """Test numeric range validation."""
        assert validate_numeric_range(5, 0, 10)
        assert not validate_numeric_range(-1, 0, 10)


class TestRealAsyncOperations(unittest.TestCase):
    """Test real async operations."""

    def test_safe_sleep(self):
        """Test async sleep wrapper."""
        async def test_sleep():
            start = asyncio.get_event_loop().time()
            await safe_sleep(0.01)
            elapsed = asyncio.get_event_loop().time() - start
            assert elapsed >= 0.01
            assert elapsed < 0.1

        asyncio.run(test_sleep())

    def test_create_async_task(self):
        """Test async task creation."""
        async def background_job():
            await safe_sleep(0.01)
            return "done"

        async def test_task():
            task = create_async_task(background_job())
            assert isinstance(task, asyncio.Task)
            result = await task
            assert result == "done"

        asyncio.run(test_task())


def run_tests():
    """Run the real subset test suite."""
    print("=" * 80)
    print("Running Real Common Operations Tests (Subset)")
    print("=" * 80)
    print("\n✅ Testing actual src/utils/common_operations.py implementation")
    print("📝 Running tests for functions that don't require numpy/pandas\n")

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestRealDateTimeOperations,
        TestRealFileOperations,
        TestRealStringOperations,
        TestRealTypeConversions,
        TestRealHashingOperations,
        TestRealCollectionOperations,
        TestRealUtilityOperations,
        TestRealAsyncOperations,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

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
        print("\n📊 Test Coverage:")
        print("   - DateTime operations: ✅")
        print("   - File operations: ✅")
        print("   - String operations: ✅")
        print("   - Type conversions: ✅")
        print("   - Hashing operations: ✅")
        print("   - Collection operations: ✅")
        print("   - Utility operations: ✅")
        print("   - Async operations: ✅")
        print("\n⚠️  Not tested (require numpy/pandas):")
        print("   - DataFrame operations")
        print("   - Numeric operations (mean, std)")
        print("   - Parquet operations")
        print("   - Data validation")
        print("   - MLflow operations")
    else:
        print("\n❌ Some tests failed!")
        if result.failures:
            print("\nFailures:")
            for test, _traceback in result.failures:
                print(f"  - {test}")
        if result.errors:
            print("\nErrors:")
            for test, _traceback in result.errors:
                print(f"  - {test}")

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
