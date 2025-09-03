"""
Comprehensive unit tests for common_operations module.

Tests all functions including edge cases, error conditions, and expected behaviors.
"""

import unittest
import tempfile
import shutil
import json
import asyncio
from datetime import datetime, date
from pathlib import Path
from unittest.mock import patch, MagicMock
import logging
import argparse

import numpy as np
import pandas as pd

# Import the module to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.common_operations import *


class TestDateTimeOperations(unittest.TestCase):
    """Test datetime utility functions."""
    
    def test_get_current_datetime(self):
        """Test get_current_datetime returns datetime object."""
        result = get_current_datetime()
        self.assertIsInstance(result, datetime)
        # Should be recent (within last minute)
        time_diff = datetime.now() - result
        self.assertLess(time_diff.total_seconds(), 60)
    
    def test_get_today(self):
        """Test get_today returns date object."""
        result = get_today()
        self.assertIsInstance(result, date)
        self.assertEqual(result, date.today())
    
    def test_format_datetime(self):
        """Test datetime formatting."""
        dt = datetime(2024, 1, 15, 10, 30, 45)
        
        # Test default format
        result = format_datetime(dt)
        self.assertEqual(result, "2024-01-15 10:30:45")
        
        # Test custom format
        result = format_datetime(dt, "%Y%m%d")
        self.assertEqual(result, "20240115")
        
        # Test ISO format
        result = format_datetime(dt, "%Y-%m-%dT%H:%M:%S")
        self.assertEqual(result, "2024-01-15T10:30:45")
    
    def test_parse_datetime(self):
        """Test datetime parsing."""
        # Test default format
        result = parse_datetime("2024-01-15 10:30:45")
        expected = datetime(2024, 1, 15, 10, 30, 45)
        self.assertEqual(result, expected)
        
        # Test custom format
        result = parse_datetime("20240115", "%Y%m%d")
        expected = datetime(2024, 1, 15, 0, 0, 0)
        self.assertEqual(result, expected)
        
        # Test invalid format
        with self.assertRaises(ValueError):
            parse_datetime("invalid date")


class TestDataFrameOperations(unittest.TestCase):
    """Test DataFrame utility functions."""
    
    def test_create_empty_dataframe(self):
        """Test empty DataFrame creation."""
        columns = ['a', 'b', 'c']
        df = create_empty_dataframe(columns)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), columns)
        self.assertEqual(len(df), 0)
    
    def test_safe_fillna(self):
        """Test safe fillna operation."""
        # Test with NaN values
        df = pd.DataFrame({'a': [1, np.nan, 3], 'b': [np.nan, 2, np.nan]})
        result = safe_fillna(df, 0)
        
        self.assertEqual(result['a'].tolist(), [1.0, 0.0, 3.0])
        self.assertEqual(result['b'].tolist(), [0.0, 2.0, 0.0])
        
        # Test with different fill value
        result = safe_fillna(df, -1)
        self.assertEqual(result['a'].tolist(), [1.0, -1.0, 3.0])
        
        # Test with no NaN values
        df_no_nan = pd.DataFrame({'a': [1, 2, 3]})
        result = safe_fillna(df_no_nan, 0)
        pd.testing.assert_frame_equal(result, df_no_nan)
    
    def test_safe_rolling(self):
        """Test safe rolling window creation."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        
        # Test basic rolling
        rolling = safe_rolling(df, window=3)
        self.assertIsInstance(rolling, pd.core.window.Rolling)
        
        # Test rolling mean calculation
        result = rolling.mean()
        expected = [np.nan, np.nan, 2.0, 3.0, 4.0]
        np.testing.assert_array_almost_equal(result['a'].values, expected)
        
        # Test with min_periods
        rolling = safe_rolling(df, window=3, min_periods=1)
        result = rolling.mean()
        expected = [1.0, 1.5, 2.0, 3.0, 4.0]
        np.testing.assert_array_almost_equal(result['a'].values, expected)
    
    def test_safe_copy(self):
        """Test safe DataFrame copying."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        # Test deep copy (default)
        copy_df = safe_copy(df)
        copy_df.iloc[0, 0] = 999
        self.assertEqual(df.iloc[0, 0], 1)  # Original unchanged
        
        # Test shallow copy
        copy_df = safe_copy(df, deep=False)
        self.assertIsNot(copy_df, df)
    
    def test_safe_resample(self):
        """Test safe time series resampling."""
        # Create time series data
        dates = pd.date_range('2024-01-01', periods=24, freq='H')
        df = pd.DataFrame({
            'close': range(24),
            'volume': range(100, 124)
        }, index=dates)
        
        # Test with default aggregations
        result = safe_resample(df, '4H')
        self.assertEqual(len(result), 6)
        self.assertEqual(result.iloc[0]['close'], 3)  # Last value in first 4H
        self.assertEqual(result.iloc[0]['volume'], 406)  # Sum of first 4 values
        
        # Test with custom aggregations
        agg_dict = {'close': 'mean', 'volume': 'max'}
        result = safe_resample(df, '4H', agg_dict)
        self.assertEqual(result.iloc[0]['close'], 1.5)  # Mean of 0,1,2,3
        self.assertEqual(result.iloc[0]['volume'], 103)  # Max of 100,101,102,103
        
        # Test with non-datetime index
        df_no_dt = pd.DataFrame({'a': [1, 2, 3]})
        with self.assertRaises(ValueError):
            safe_resample(df_no_dt, '1H')


class TestNumericOperations(unittest.TestCase):
    """Test numeric utility functions."""
    
    def test_safe_mean(self):
        """Test safe mean calculation."""
        # Test with list
        self.assertEqual(safe_mean([1, 2, 3, 4, 5]), 3.0)
        
        # Test with numpy array
        arr = np.array([1, 2, 3, 4, 5])
        self.assertEqual(safe_mean(arr), 3.0)
        
        # Test with pandas Series
        series = pd.Series([1, 2, 3, 4, 5])
        self.assertEqual(safe_mean(series), 3.0)
        
        # Test with empty input
        self.assertTrue(np.isnan(safe_mean([])))
        
        # Test with NaN values
        self.assertEqual(safe_mean([1, np.nan, 3]), 2.0)
    
    def test_safe_std(self):
        """Test safe standard deviation calculation."""
        # Test with list
        result = safe_std([1, 2, 3, 4, 5])
        self.assertAlmostEqual(result, 1.4142135623730951)
        
        # Test with empty input
        self.assertTrue(np.isnan(safe_std([])))
        
        # Test with single value
        self.assertEqual(safe_std([5]), 0.0)
        
        # Test with NaN values
        result = safe_std([1, np.nan, 3])
        self.assertAlmostEqual(result, 1.0)


class TestFileOperations(unittest.TestCase):
    """Test file operation utilities."""
    
    def setUp(self):
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test.json"
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_ensure_directory(self):
        """Test directory creation."""
        # Test creating new directory
        new_dir = Path(self.temp_dir) / "new" / "nested" / "dir"
        result = ensure_directory(new_dir)
        
        self.assertIsInstance(result, Path)
        self.assertTrue(result.exists())
        self.assertTrue(result.is_dir())
        
        # Test with existing directory
        result2 = ensure_directory(new_dir)
        self.assertEqual(result, result2)
        
        # Test with string path
        str_dir = str(Path(self.temp_dir) / "string_dir")
        result = ensure_directory(str_dir)
        self.assertTrue(result.exists())
    
    def test_safe_file_exists(self):
        """Test safe file existence check."""
        # Test non-existent file
        self.assertFalse(safe_file_exists(self.test_file))
        
        # Create file
        self.test_file.write_text("test")
        self.assertTrue(safe_file_exists(self.test_file))
        
        # Test with string path
        self.assertTrue(safe_file_exists(str(self.test_file)))
        
        # Test with invalid path (should not raise)
        self.assertFalse(safe_file_exists("\0invalid\0path"))
    
    def test_safe_json_operations(self):
        """Test JSON dump and load operations."""
        test_data = {
            "string": "value",
            "number": 42,
            "float": 3.14,
            "list": [1, 2, 3],
            "nested": {"key": "value"}
        }
        
        # Test dump
        safe_json_dump(test_data, self.test_file, indent=2)
        self.assertTrue(self.test_file.exists())
        
        # Test load
        loaded_data = safe_json_load(self.test_file)
        self.assertEqual(loaded_data, test_data)
        
        # Test load non-existent file
        with self.assertRaises(FileNotFoundError):
            safe_json_load("non_existent.json")


class TestParquetOperations(unittest.TestCase):
    """Test parquet file operations."""
    
    def setUp(self):
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': ['x', 'y', 'z']
        })
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_safe_to_parquet(self):
        """Test safe parquet writing."""
        file_path = Path(self.temp_dir) / "test.parquet"
        
        # Test successful write
        success = safe_to_parquet(self.test_df, file_path)
        self.assertTrue(success)
        self.assertTrue(file_path.exists())
        
        # Test write with compression
        file_path2 = Path(self.temp_dir) / "test_compressed.parquet"
        success = safe_to_parquet(self.test_df, file_path2, compression='snappy')
        self.assertTrue(success)
    
    def test_safe_read_parquet(self):
        """Test safe parquet reading."""
        file_path = Path(self.temp_dir) / "test.parquet"
        self.test_df.to_parquet(file_path)
        
        # Test successful read
        df = safe_read_parquet(file_path)
        pd.testing.assert_frame_equal(df, self.test_df)
        
        # Test read with columns
        df = safe_read_parquet(file_path, columns=['a', 'c'])
        self.assertEqual(list(df.columns), ['a', 'c'])
        
        # Test read non-existent file
        df = safe_read_parquet("non_existent.parquet")
        self.assertTrue(df.empty)
    
    def test_list_parquet_files(self):
        """Test listing parquet files."""
        # Create test files
        (Path(self.temp_dir) / "file1.parquet").touch()
        (Path(self.temp_dir) / "file2.parquet").touch()
        (Path(self.temp_dir) / "subdir").mkdir()
        (Path(self.temp_dir) / "subdir" / "file3.parquet").touch()
        (Path(self.temp_dir) / "other.txt").touch()
        
        # Test recursive listing
        files = list_parquet_files(self.temp_dir, recursive=True)
        self.assertEqual(len(files), 3)
        self.assertTrue(all(f.suffix == '.parquet' for f in files))
        
        # Test non-recursive listing
        files = list_parquet_files(self.temp_dir, recursive=False)
        self.assertEqual(len(files), 2)


class TestHashingOperations(unittest.TestCase):
    """Test hashing and cache key operations."""
    
    def test_generate_hash(self):
        """Test hash generation for different data types."""
        # Test string hashing
        hash1 = generate_hash("test string")
        self.assertEqual(len(hash1), 32)  # MD5 length
        
        # Test same string gives same hash
        hash2 = generate_hash("test string")
        self.assertEqual(hash1, hash2)
        
        # Test different string gives different hash
        hash3 = generate_hash("different string")
        self.assertNotEqual(hash1, hash3)
        
        # Test SHA256
        hash_sha = generate_hash("test", algorithm="sha256")
        self.assertEqual(len(hash_sha), 64)  # SHA256 length
        
        # Test DataFrame hashing
        df = pd.DataFrame({'a': [1, 2, 3]})
        hash_df = generate_hash(df)
        self.assertEqual(len(hash_df), 32)
        
        # Test invalid algorithm
        with self.assertRaises(ValueError):
            generate_hash("test", algorithm="invalid")
    
    def test_generate_cache_key(self):
        """Test cache key generation."""
        # Test basic cache key
        key = generate_cache_key("features", "BTCUSDT", "1h")
        self.assertEqual(len(key), 16)  # Default max_length
        
        # Test consistent generation
        key2 = generate_cache_key("features", "BTCUSDT", "1h")
        self.assertEqual(key, key2)
        
        # Test different inputs
        key3 = generate_cache_key("features", "ETHUSDT", "1h")
        self.assertNotEqual(key, key3)
        
        # Test custom length
        key_long = generate_cache_key("test", "data", max_length=32)
        self.assertEqual(len(key_long), 32)


class TestAsyncOperations(unittest.TestCase):
    """Test async utility functions."""
    
    def test_safe_sleep(self):
        """Test async sleep wrapper."""
        async def test_sleep():
            start = asyncio.get_event_loop().time()
            await safe_sleep(0.1)
            elapsed = asyncio.get_event_loop().time() - start
            self.assertGreaterEqual(elapsed, 0.1)
            self.assertLess(elapsed, 0.2)
        
        asyncio.run(test_sleep())
    
    def test_safe_gather(self):
        """Test safe gathering of coroutines."""
        async def task(n):
            await safe_sleep(0.01)
            return n * 2
        
        async def failing_task():
            raise ValueError("Test error")
        
        async def test_gather():
            # Test successful tasks
            results = await safe_gather(task(1), task(2), task(3))
            self.assertEqual(results, [2, 4, 6])
            
            # Test with exception
            results = await safe_gather(task(1), failing_task(), task(3))
            self.assertEqual(results[0], 2)
            self.assertIsInstance(results[1], ValueError)
            self.assertEqual(results[2], 6)
            
            # Test with return_exceptions=False
            with self.assertRaises(ValueError):
                await safe_gather(task(1), failing_task(), return_exceptions=False)
        
        asyncio.run(test_gather())
    
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


class TestCollectionOperations(unittest.TestCase):
    """Test collection utility functions."""
    
    def test_safe_append(self):
        """Test safe list append."""
        # Test with None
        lst = safe_append(None, "item")
        self.assertEqual(lst, ["item"])
        
        # Test with existing list
        lst = safe_append(lst, "item2")
        self.assertEqual(lst, ["item", "item2"])
        
        # Test with different types
        lst = safe_append([], 42)
        self.assertEqual(lst, [42])
    
    def test_safe_extend(self):
        """Test safe list extend."""
        # Test with None
        lst = safe_extend(None, [1, 2, 3])
        self.assertEqual(lst, [1, 2, 3])
        
        # Test with existing list
        lst = safe_extend([0], [1, 2, 3])
        self.assertEqual(lst, [0, 1, 2, 3])
    
    def test_safe_dict_get(self):
        """Test safe dictionary get."""
        # Test with None dict
        result = safe_dict_get(None, "key", "default")
        self.assertEqual(result, "default")
        
        # Test with existing key
        d = {"key": "value"}
        result = safe_dict_get(d, "key", "default")
        self.assertEqual(result, "value")
        
        # Test with missing key
        result = safe_dict_get(d, "missing", "default")
        self.assertEqual(result, "default")
    
    def test_safe_dict_items(self):
        """Test safe dictionary items."""
        # Test with None
        items = safe_dict_items(None)
        self.assertEqual(items, [])
        
        # Test with dict
        d = {"a": 1, "b": 2}
        items = safe_dict_items(d)
        self.assertEqual(sorted(items), [("a", 1), ("b", 2)])
    
    def test_safe_defaultdict(self):
        """Test safe defaultdict creation."""
        dd = safe_defaultdict(list)
        dd["key"].append(1)
        self.assertEqual(dd["key"], [1])
        self.assertEqual(dd["missing"], [])
    
    def test_safe_counter(self):
        """Test safe Counter creation."""
        # Test with None
        c = safe_counter()
        self.assertEqual(len(c), 0)
        
        # Test with items
        c = safe_counter(['a', 'b', 'a', 'c', 'b', 'a'])
        self.assertEqual(c['a'], 3)
        self.assertEqual(c['b'], 2)
    
    def test_safe_deque(self):
        """Test safe deque creation."""
        # Test with None
        d = safe_deque()
        self.assertEqual(len(d), 0)
        
        # Test with items and maxlen
        d = safe_deque([1, 2, 3], maxlen=2)
        self.assertEqual(list(d), [2, 3])


class TestStringOperations(unittest.TestCase):
    """Test string utility functions."""
    
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
        self.assertEqual(safe_join("-", [1, 2, None, 3]), "1-2-None-3")


class TestLoggingOperations(unittest.TestCase):
    """Test logging utility functions."""
    
    def test_get_logger(self):
        """Test logger creation."""
        logger = get_logger("test_logger")
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "test_logger")
        
        # Test same name returns same logger
        logger2 = get_logger("test_logger")
        self.assertIs(logger, logger2)
    
    def test_setup_basic_logging(self):
        """Test basic logging setup."""
        # Clear existing handlers
        logging.root.handlers = []
        
        setup_basic_logging(logging.DEBUG)
        
        # Check that handler was added
        self.assertEqual(len(logging.root.handlers), 1)
        self.assertEqual(logging.root.level, logging.DEBUG)


class TestValidationOperations(unittest.TestCase):
    """Test validation utility functions."""
    
    def test_validate_dataframe(self):
        """Test DataFrame validation."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        
        # Test with all columns present
        self.assertTrue(validate_dataframe(df, ['a', 'b']))
        
        # Test with missing columns
        self.assertFalse(validate_dataframe(df, ['a', 'd']))
        
        # Test with None
        self.assertFalse(validate_dataframe(None, ['a']))
        
        # Test with empty DataFrame
        self.assertFalse(validate_dataframe(pd.DataFrame(), ['a']))
    
    def test_validate_numeric_range(self):
        """Test numeric range validation."""
        self.assertTrue(validate_numeric_range(5, 0, 10))
        self.assertTrue(validate_numeric_range(0, 0, 10))
        self.assertTrue(validate_numeric_range(10, 0, 10))
        self.assertFalse(validate_numeric_range(-1, 0, 10))
        self.assertFalse(validate_numeric_range(11, 0, 10))
    
    def test_validate_dataframe_schema(self):
        """Test DataFrame schema validation."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.0, 2.0, 3.0],
            'str_col': ['a', 'b', 'c']
        })
        
        # Test required columns only
        is_valid, errors = validate_dataframe_schema(df, ['int_col', 'float_col'])
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])
        
        # Test with missing columns
        is_valid, errors = validate_dataframe_schema(df, ['int_col', 'missing_col'])
        self.assertFalse(is_valid)
        self.assertTrue(any('Missing columns' in e for e in errors))
        
        # Test with column types
        is_valid, errors = validate_dataframe_schema(
            df, 
            ['int_col'], 
            column_types={'int_col': np.integer}
        )
        self.assertTrue(is_valid)
    
    def test_validate_data_quality(self):
        """Test data quality validation."""
        # Test good data
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        
        quality = validate_data_quality(df)
        self.assertTrue(quality['is_valid'])
        self.assertEqual(quality['total_rows'], 5)
        self.assertEqual(quality['total_columns'], 2)
        self.assertEqual(len(quality['issues']), 0)
        
        # Test data with high NaN ratio
        df_nan = pd.DataFrame({
            'a': [1, np.nan, np.nan, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        
        quality = validate_data_quality(df_nan, max_nan_ratio=0.1)
        self.assertFalse(quality['is_valid'])
        self.assertTrue(any(issue['type'] == 'high_nan_ratio' for issue in quality['issues']))
        
        # Test data with duplicates
        df_dup = pd.DataFrame({
            'a': [1, 1, 1, 1, 1],
            'b': [10, 10, 10, 10, 10]
        })
        
        quality = validate_data_quality(df_dup, check_duplicates=True)
        self.assertFalse(quality['is_valid'])
        self.assertTrue(any(issue['type'] == 'duplicates' for issue in quality['issues']))


class TestUtilityOperations(unittest.TestCase):
    """Test utility operations."""
    
    def test_timed_operation(self):
        """Test operation timing decorator."""
        @timed_operation("test_operation")
        def slow_function():
            time.sleep(0.1)
            return "done"
        
        with self.assertLogs(level='INFO') as cm:
            result = slow_function()
            self.assertEqual(result, "done")
            self.assertTrue(any("Starting test_operation" in log for log in cm.output))
            self.assertTrue(any("Completed test_operation" in log for log in cm.output))
        
        @timed_operation("failing_operation")
        def failing_function():
            raise ValueError("Test error")
        
        with self.assertLogs(level='ERROR') as cm:
            with self.assertRaises(ValueError):
                failing_function()
            self.assertTrue(any("Failed failing_operation" in log for log in cm.output))
    
    def test_format_bytes(self):
        """Test byte formatting."""
        self.assertEqual(format_bytes(0), "0.00 B")
        self.assertEqual(format_bytes(1023), "1023.00 B")
        self.assertEqual(format_bytes(1024), "1.00 KB")
        self.assertEqual(format_bytes(1024 * 1024), "1.00 MB")
        self.assertEqual(format_bytes(1024 * 1024 * 1024), "1.00 GB")
        self.assertEqual(format_bytes(1024 * 1024 * 1024 * 1024), "1.00 TB")
        self.assertEqual(format_bytes(1024 * 1024 * 1024 * 1024 * 1024), "1.00 PB")
    
    def test_chunked_iterable(self):
        """Test iterable chunking."""
        # Test basic chunking
        items = list(range(10))
        chunks = chunked_iterable(items, 3)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0], [0, 1, 2])
        self.assertEqual(chunks[1], [3, 4, 5])
        self.assertEqual(chunks[2], [6, 7, 8])
        self.assertEqual(chunks[3], [9])
        
        # Test exact chunks
        items = list(range(9))
        chunks = chunked_iterable(items, 3)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(len(chunk) == 3 for chunk in chunks))
        
        # Test empty list
        chunks = chunked_iterable([], 3)
        self.assertEqual(chunks, [])
    
    def test_parallel_map(self):
        """Test parallel mapping."""
        def square(x):
            return x * x
        
        items = list(range(10))
        results = parallel_map(square, items, max_workers=2)
        expected = [x * x for x in items]
        self.assertEqual(results, expected)
        
        # Test with None max_workers
        results = parallel_map(square, items)
        self.assertEqual(results, expected)


class TestTypeConversions(unittest.TestCase):
    """Test type conversion utilities."""
    
    def test_safe_float(self):
        """Test safe float conversion."""
        self.assertEqual(safe_float("3.14"), 3.14)
        self.assertEqual(safe_float("42"), 42.0)
        self.assertEqual(safe_float(42), 42.0)
        self.assertEqual(safe_float("invalid", -1.0), -1.0)
        self.assertEqual(safe_float(None, 0.0), 0.0)
    
    def test_safe_int(self):
        """Test safe int conversion."""
        self.assertEqual(safe_int("42"), 42)
        self.assertEqual(safe_int(42.7), 42)
        self.assertEqual(safe_int("invalid", -1), -1)
        self.assertEqual(safe_int(None, 0), 0)


class TestMLflowOperations(unittest.TestCase):
    """Test MLflow integration helpers."""
    
    @patch('mlflow.active_run')
    @patch('mlflow.log_metric')
    def test_safe_log_metric(self, mock_log_metric, mock_active_run):
        """Test safe metric logging."""
        # Test with active run
        mock_active_run.return_value = MagicMock()
        safe_log_metric("accuracy", 0.95)
        mock_log_metric.assert_called_once_with("accuracy", 0.95, None)
        
        # Test with step
        mock_log_metric.reset_mock()
        safe_log_metric("loss", 0.05, step=10)
        mock_log_metric.assert_called_once_with("loss", 0.05, 10)
        
        # Test without active run
        mock_active_run.return_value = None
        mock_log_metric.reset_mock()
        safe_log_metric("accuracy", 0.95)  # Should not raise
        mock_log_metric.assert_not_called()
    
    @patch('mlflow.active_run')
    @patch('mlflow.log_params')
    def test_safe_log_params(self, mock_log_params, mock_active_run):
        """Test safe parameter logging."""
        params = {"learning_rate": 0.01, "batch_size": 32}
        
        # Test with active run
        mock_active_run.return_value = MagicMock()
        safe_log_params(params)
        mock_log_params.assert_called_once_with(params)
        
        # Test without active run
        mock_active_run.return_value = None
        mock_log_params.reset_mock()
        safe_log_params(params)  # Should not raise
        mock_log_params.assert_not_called()
    
    @patch('mlflow.active_run')
    @patch('mlflow.log_artifact')
    def test_safe_log_artifact(self, mock_log_artifact, mock_active_run):
        """Test safe artifact logging."""
        # Test with active run
        mock_active_run.return_value = MagicMock()
        safe_log_artifact("model.pkl")
        mock_log_artifact.assert_called_once_with("model.pkl")
        
        # Test with Path object
        mock_log_artifact.reset_mock()
        safe_log_artifact(Path("model.pkl"))
        mock_log_artifact.assert_called_once_with("model.pkl")
        
        # Test without active run
        mock_active_run.return_value = None
        mock_log_artifact.reset_mock()
        safe_log_artifact("model.pkl")  # Should not raise
        mock_log_artifact.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)