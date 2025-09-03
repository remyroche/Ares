#!/usr/bin/env python3
"""
Run subset of real common_operations tests by mocking only numpy/pandas imports.
This tests the actual implementation for functions that don't need these dependencies.
"""

import sys
import unittest
import tempfile
import shutil
import json
import asyncio
from datetime import datetime, date
from pathlib import Path
import logging
from unittest.mock import MagicMock, patch
import types

# Mock numpy and pandas before any imports
numpy_mock = types.ModuleType('numpy')
numpy_mock.nan = float('nan')
numpy_mock.integer = int
numpy_mock.floating = float

pandas_mock = types.ModuleType('pandas')
pandas_mock.DataFrame = MagicMock
pandas_mock.Series = MagicMock
pandas_mock.core = types.ModuleType('core')
pandas_mock.core.window = types.ModuleType('window')
pandas_mock.core.window.Rolling = MagicMock

sys.modules['numpy'] = numpy_mock
sys.modules['pandas'] = pandas_mock

# Now we can import the real module
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.common_operations import (
    # DateTime operations
    get_current_datetime,
    get_today,
    format_datetime,
    parse_datetime,
    # File operations
    ensure_directory,
    safe_file_exists,
    safe_json_dump,
    safe_json_load,
    # String operations
    safe_lower,
    safe_upper,
    safe_join,
    # Type conversions
    safe_float,
    safe_int,
    # Hashing operations
    generate_hash,
    generate_cache_key,
    # Collection operations
    safe_append,
    safe_extend,
    safe_dict_get,
    safe_dict_items,
    safe_defaultdict,
    safe_counter,
    safe_deque,
    # Logging operations
    get_logger,
    setup_basic_logging,
    # Utility operations
    format_bytes,
    chunked_iterable,
    validate_numeric_range,
    # Async operations
    safe_sleep,
    safe_gather,
    create_async_task,
)


class TestRealDateTimeOperations(unittest.TestCase):
    """Test real datetime utility functions."""
    
    def test_get_current_datetime(self):
        """Test get_current_datetime returns datetime object."""
        result = get_current_datetime()
        self.assertIsInstance(result, datetime)
        time_diff = datetime.now() - result
        self.assertLess(time_diff.total_seconds(), 1)
    
    def test_get_today(self):
        """Test get_today returns date object."""
        result = get_today()
        self.assertIsInstance(result, date)
        self.assertEqual(result, date.today())
    
    def test_format_datetime(self):
        """Test datetime formatting."""
        dt = datetime(2024, 1, 15, 10, 30, 45)
        
        result = format_datetime(dt)
        self.assertEqual(result, "2024-01-15 10:30:45")
        
        result = format_datetime(dt, "%Y%m%d")
        self.assertEqual(result, "20240115")
    
    def test_parse_datetime(self):
        """Test datetime parsing."""
        result = parse_datetime("2024-01-15 10:30:45")
        expected = datetime(2024, 1, 15, 10, 30, 45)
        self.assertEqual(result, expected)
        
        with self.assertRaises(ValueError):
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
        
        self.assertIsInstance(result, Path)
        self.assertTrue(result.exists())
        self.assertTrue(result.is_dir())
    
    def test_safe_file_exists(self):
        """Test safe file existence check."""
        test_file = Path(self.temp_dir) / "test.txt"
        
        self.assertFalse(safe_file_exists(test_file))
        
        test_file.write_text("test")
        self.assertTrue(safe_file_exists(test_file))
    
    def test_safe_json_operations(self):
        """Test JSON dump and load operations."""
        test_data = {
            "string": "value",
            "number": 42,
            "list": [1, 2, 3],
        }
        
        test_file = Path(self.temp_dir) / "test.json"
        
        safe_json_dump(test_data, test_file, indent=2)
        self.assertTrue(test_file.exists())
        
        loaded_data = safe_json_load(test_file)
        self.assertEqual(loaded_data, test_data)


class TestRealStringOperations(unittest.TestCase):
    """Test real string utility functions."""
    
    def test_safe_lower(self):
        """Test safe lowercase conversion."""
        self.assertEqual(safe_lower("HELLO"), "hello")
        self.assertEqual(safe_lower(None), "")
        self.assertEqual(safe_lower(123), "123")
    
    def test_safe_upper(self):
        """Test safe uppercase conversion."""
        self.assertEqual(safe_upper("hello"), "HELLO")
        self.assertEqual(safe_upper(None), "")
        self.assertEqual(safe_upper(123), "123")
    
    def test_safe_join(self):
        """Test safe string joining."""
        self.assertEqual(safe_join(", ", ["a", "b", "c"]), "a, b, c")
        self.assertEqual(safe_join(", ", None), "")


class TestRealTypeConversions(unittest.TestCase):
    """Test real type conversion utilities."""
    
    def test_safe_float(self):
        """Test safe float conversion."""
        self.assertEqual(safe_float("3.14"), 3.14)
        self.assertEqual(safe_float("42"), 42.0)
        self.assertEqual(safe_float("invalid", -1.0), -1.0)
        self.assertEqual(safe_float(None, 0.0), 0.0)
    
    def test_safe_int(self):
        """Test safe int conversion."""
        self.assertEqual(safe_int("42"), 42)
        self.assertEqual(safe_int(42.7), 42)
        self.assertEqual(safe_int("invalid", -1), -1)


class TestRealHashingOperations(unittest.TestCase):
    """Test real hashing operations."""
    
    def test_generate_hash(self):
        """Test hash generation."""
        hash1 = generate_hash("test string")
        self.assertEqual(len(hash1), 32)  # MD5 length
        
        hash2 = generate_hash("test string")
        self.assertEqual(hash1, hash2)  # Same input, same hash
        
        hash3 = generate_hash("different")
        self.assertNotEqual(hash1, hash3)
    
    def test_generate_cache_key(self):
        """Test cache key generation."""
        key = generate_cache_key("features", "BTCUSDT", "1h")
        self.assertEqual(len(key), 16)  # Default max_length
        
        key2 = generate_cache_key("features", "BTCUSDT", "1h")
        self.assertEqual(key, key2)  # Consistent


class TestRealCollectionOperations(unittest.TestCase):
    """Test real collection operations."""
    
    def test_safe_append(self):
        """Test safe list append."""
        lst = safe_append(None, "item")
        self.assertEqual(lst, ["item"])
        
        lst = safe_append(lst, "item2")
        self.assertEqual(lst, ["item", "item2"])
    
    def test_safe_dict_get(self):
        """Test safe dictionary get."""
        result = safe_dict_get(None, "key", "default")
        self.assertEqual(result, "default")
        
        d = {"key": "value"}
        result = safe_dict_get(d, "key", "default")
        self.assertEqual(result, "value")


class TestRealUtilityOperations(unittest.TestCase):
    """Test real utility operations."""
    
    def test_format_bytes(self):
        """Test byte formatting."""
        self.assertEqual(format_bytes(0), "0.00 B")
        self.assertEqual(format_bytes(1024), "1.00 KB")
        self.assertEqual(format_bytes(1024 * 1024), "1.00 MB")
    
    def test_chunked_iterable(self):
        """Test iterable chunking."""
        items = list(range(10))
        chunks = chunked_iterable(items, 3)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0], [0, 1, 2])
        self.assertEqual(chunks[3], [9])
    
    def test_validate_numeric_range(self):
        """Test numeric range validation."""
        self.assertTrue(validate_numeric_range(5, 0, 10))
        self.assertFalse(validate_numeric_range(-1, 0, 10))


class TestRealAsyncOperations(unittest.TestCase):
    """Test real async operations."""
    
    def test_safe_sleep(self):
        """Test async sleep wrapper."""
        async def test_sleep():
            start = asyncio.get_event_loop().time()
            await safe_sleep(0.01)
            elapsed = asyncio.get_event_loop().time() - start
            self.assertGreaterEqual(elapsed, 0.01)
            self.assertLess(elapsed, 0.1)
        
        asyncio.run(test_sleep())
    
    def test_create_async_task(self):
        """Test async task creation."""
        async def background_job():
            await safe_sleep(0.01)
            return "done"
        
        async def test_task():
            task = create_async_task(background_job())
            self.assertIsInstance(task, asyncio.Task)
            result = await task
            self.assertEqual(result, "done")
        
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
            for test, traceback in result.failures:
                print(f"  - {test}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}")
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())