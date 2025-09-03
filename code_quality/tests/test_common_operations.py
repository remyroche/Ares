"""
Comprehensive unit tests for common_operations module.

Tests all functions including edge cases, error conditions, and expected behaviors.
"""

import asyncio
import logging
import shutil

# Import the module to test
import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest

from src.utils.common_operations import * as *_src_utils_common_operations


class TestDateTimeOperations(unittest.TestCase):
    """Test datetime utility functions."""

    def test_get_current_datetime(self):
        """Test get_current_datetime returns datetime object."""
        result = get_current_datetime()
        assert isinstance(result, datetime)
        # Should be recent (within last minute)
        time_diff = datetime.now() - result
        assert time_diff.total_seconds() < 60

    def test_get_today(self):
        """Test get_today returns date object."""
        result = get_today()
        assert isinstance(result, date)
        assert result == date.today()

    def test_format_datetime(self):
        """Test datetime formatting."""
        dt = datetime(2024, 1, 15, 10, 30, 45)

        # Test default format
        result = format_datetime(dt)
        assert result == "2024-01-15 10:30:45"

        # Test custom format
        result = format_datetime(dt, "%Y%m%d")
        assert result == "20240115"

        # Test ISO format
        result = format_datetime(dt, "%Y-%m-%dT%H:%M:%S")
        assert result == "2024-01-15T10:30:45"

    def test_parse_datetime(self):
        """Test datetime parsing."""
        # Test default format
        result = parse_datetime("2024-01-15 10:30:45")
        expected = datetime(2024, 1, 15, 10, 30, 45)
        assert result == expected

        # Test custom format
        result = parse_datetime("20240115", "%Y%m%d")
        expected = datetime(2024, 1, 15, 0, 0, 0)
        assert result == expected

        # Test invalid format
        with pytest.raises(ValueError):
            parse_datetime("invalid date")


class TestDataFrameOperations(unittest.TestCase):
    """Test DataFrame utility functions."""

    def test_create_empty_dataframe(self):
        """Test empty DataFrame creation."""
        columns = ["a", "b", "c"]
        df = create_empty_dataframe(columns)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == columns
        assert len(df) == 0

    def test_safe_fillna(self):
        """Test safe fillna operation."""
        # Test with NaN values
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, np.nan]})
        result = safe_fillna(df, 0)

        assert result["a"].tolist() == [1.0, 0.0, 3.0]
        assert result["b"].tolist() == [0.0, 2.0, 0.0]

        # Test with different fill value
        result = safe_fillna(df, -1)
        assert result["a"].tolist() == [1.0, -1.0, 3.0]

        # Test with no NaN values
        df_no_nan = pd.DataFrame({"a": [1, 2, 3]})
        result = safe_fillna(df_no_nan, 0)
        pd.testing.assert_frame_equal(result, df_no_nan)

    def test_safe_rolling(self):
        """Test safe rolling window creation."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

        # Test basic rolling
        rolling = safe_rolling(df, window=3)
        assert isinstance(rolling, pd.core.window.Rolling)

        # Test rolling mean calculation
        result = rolling.mean()
        expected = [np.nan, np.nan, 2.0, 3.0, 4.0]
        np.testing.assert_array_almost_equal(result["a"].values, expected)

        # Test with min_periods
        rolling = safe_rolling(df, window=3, min_periods=1)
        result = rolling.mean()
        expected = [1.0, 1.5, 2.0, 3.0, 4.0]
        np.testing.assert_array_almost_equal(result["a"].values, expected)

    def test_safe_copy(self):
        """Test safe DataFrame copying."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Test deep copy (default)
        copy_df = safe_copy(df)
        copy_df.iloc[0, 0] = 999
        assert df.iloc[0, 0] == 1  # Original unchanged

        # Test shallow copy
        copy_df = safe_copy(df, deep=False)
        assert copy_df is not df

    def test_safe_resample(self):
        """Test safe time series resampling."""
        # Create time series data
        dates = pd.date_range("2024-01-01", periods=24, freq="H")
        df = pd.DataFrame({
            "close": range(24),
            "volume": range(100, 124),
        }, index=dates)

        # Test with default aggregations
        result = safe_resample(df, "4H")
        assert len(result) == 6
        assert result.iloc[0]["close"] == 3  # Last value in first 4H
        assert result.iloc[0]["volume"] == 406  # Sum of first 4 values

        # Test with custom aggregations
        agg_dict = {"close": "mean", "volume": "max"}
        result = safe_resample(df, "4H", agg_dict)
        assert result.iloc[0]["close"] == 1.5  # Mean of 0,1,2,3
        assert result.iloc[0]["volume"] == 103  # Max of 100,101,102,103

        # Test with non-datetime index
        df_no_dt = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError):
            safe_resample(df_no_dt, "1H")


class TestNumericOperations(unittest.TestCase):
    """Test numeric utility functions."""

    def test_safe_mean(self):
        """Test safe mean calculation."""
        # Test with list
        assert safe_mean([1, 2, 3, 4, 5]) == 3.0

        # Test with numpy array
        arr = np.array([1, 2, 3, 4, 5])
        assert safe_mean(arr) == 3.0

        # Test with pandas Series
        series = pd.Series([1, 2, 3, 4, 5])
        assert safe_mean(series) == 3.0

        # Test with empty input
        assert np.isnan(safe_mean([]))

        # Test with NaN values
        assert safe_mean([1, np.nan, 3]) == 2.0

    def test_safe_std(self):
        """Test safe standard deviation calculation."""
        # Test with list
        result = safe_std([1, 2, 3, 4, 5])
        self.assertAlmostEqual(result, 1.4142135623730951)

        # Test with empty input
        assert np.isnan(safe_std([]))

        # Test with single value
        assert safe_std([5]) == 0.0

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

        assert isinstance(result, Path)
        assert result.exists()
        assert result.is_dir()

        # Test with existing directory
        result2 = ensure_directory(new_dir)
        assert result == result2

        # Test with string path
        str_dir = str(Path(self.temp_dir) / "string_dir")
        result = ensure_directory(str_dir)
        assert result.exists()

    def test_safe_file_exists(self):
        """Test safe file existence check."""
        # Test non-existent file
        assert not safe_file_exists(self.test_file)

        # Create file
        self.test_file.write_text("test")
        assert safe_file_exists(self.test_file)

        # Test with string path
        assert safe_file_exists(str(self.test_file))

        # Test with invalid path (should not raise)
        assert not safe_file_exists("\x00invalid\x00path")

    def test_safe_json_operations(self):
        """Test JSON dump and load operations."""
        test_data = {
            "string": "value",
            "number": 42,
            "float": 3.14,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
        }

        # Test dump
        safe_json_dump(test_data, self.test_file, indent=2)
        assert self.test_file.exists()

        # Test load
        loaded_data = safe_json_load(self.test_file)
        assert loaded_data == test_data

        # Test load non-existent file
        with pytest.raises(FileNotFoundError):
            safe_json_load("non_existent.json")


class TestParquetOperations(unittest.TestCase):
    """Test parquet file operations."""

    def setUp(self):
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": ["x", "y", "z"],
        })

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_safe_to_parquet(self):
        """Test safe parquet writing."""
        file_path = Path(self.temp_dir) / "test.parquet"

        # Test successful write
        success = safe_to_parquet(self.test_df, file_path)
        assert success
        assert file_path.exists()

        # Test write with compression
        file_path2 = Path(self.temp_dir) / "test_compressed.parquet"
        success = safe_to_parquet(self.test_df, file_path2, compression="snappy")
        assert success

    def test_safe_read_parquet(self):
        """Test safe parquet reading."""
        file_path = Path(self.temp_dir) / "test.parquet"
        self.test_df.to_parquet(file_path)

        # Test successful read
        df = safe_read_parquet(file_path)
        pd.testing.assert_frame_equal(df, self.test_df)

        # Test read with columns
        df = safe_read_parquet(file_path, columns=["a", "c"])
        assert list(df.columns) == ["a", "c"]

        # Test read non-existent file
        df = safe_read_parquet("non_existent.parquet")
        assert df.empty

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
        assert len(files) == 3
        assert all(f.suffix == ".parquet" for f in files)

        # Test non-recursive listing
        files = list_parquet_files(self.temp_dir, recursive=False)
        assert len(files) == 2


class TestHashingOperations(unittest.TestCase):
    """Test hashing and cache key operations."""

    def test_generate_hash(self):
        """Test hash generation for different data types."""
        # Test string hashing
        hash1 = generate_hash("test string")
        assert len(hash1) == 32  # MD5 length

        # Test same string gives same hash
        hash2 = generate_hash("test string")
        assert hash1 == hash2

        # Test different string gives different hash
        hash3 = generate_hash("different string")
        assert hash1 != hash3

        # Test SHA256
        hash_sha = generate_hash("test", algorithm="sha256")
        assert len(hash_sha) == 64  # SHA256 length

        # Test DataFrame hashing
        df = pd.DataFrame({"a": [1, 2, 3]})
        hash_df = generate_hash(df)
        assert len(hash_df) == 32

        # Test invalid algorithm
        with pytest.raises(ValueError):
            generate_hash("test", algorithm="invalid")

    def test_generate_cache_key(self):
        """Test cache key generation."""
        # Test basic cache key
        key = generate_cache_key("features", "BTCUSDT", "1h")
        assert len(key) == 16  # Default max_length

        # Test consistent generation
        key2 = generate_cache_key("features", "BTCUSDT", "1h")
        assert key == key2

        # Test different inputs
        key3 = generate_cache_key("features", "ETHUSDT", "1h")
        assert key != key3

        # Test custom length
        key_long = generate_cache_key("test", "data", max_length=32)
        assert len(key_long) == 32


class TestAsyncOperations(unittest.TestCase):
    """Test async utility functions."""

    def test_safe_sleep(self):
        """Test async sleep wrapper."""
        async def test_sleep():
            start = asyncio.get_event_loop().time()
            await safe_sleep(0.1)
            elapsed = asyncio.get_event_loop().time() - start
            assert elapsed >= 0.1
            assert elapsed < 0.2

        asyncio.run(test_sleep())

    def test_safe_gather(self):
        """Test safe gathering of coroutines."""
        async def task(n):
            await safe_sleep(0.01)
            return n * 2

        async def failing_task():
            msg = "Test error"
            raise ValueError(msg)

        async def test_gather():
            # Test successful tasks
            results = await safe_gather(task(1), task(2), task(3))
            assert results == [2, 4, 6]

            # Test with exception
            results = await safe_gather(task(1), failing_task(), task(3))
            assert results[0] == 2
            assert isinstance(results[1], ValueError)
            assert results[2] == 6

            # Test with return_exceptions=False
            with pytest.raises(ValueError):
                await safe_gather(task(1), failing_task(), return_exceptions=False)

        asyncio.run(test_gather())

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


class TestCollectionOperations(unittest.TestCase):
    """Test collection utility functions."""

    def test_safe_append(self):
        """Test safe list append."""
        # Test with None
        lst = safe_append(None, "item")
        assert lst == ["item"]

        # Test with existing list
        lst = safe_append(lst, "item2")
        assert lst == ["item", "item2"]

        # Test with different types
        lst = safe_append([], 42)
        assert lst == [42]

    def test_safe_extend(self):
        """Test safe list extend."""
        # Test with None
        lst = safe_extend(None, [1, 2, 3])
        assert lst == [1, 2, 3]

        # Test with existing list
        lst = safe_extend([0], [1, 2, 3])
        assert lst == [0, 1, 2, 3]

    def test_safe_dict_get(self):
        """Test safe dictionary get."""
        # Test with None dict
        result = safe_dict_get(None, "key", "default")
        assert result == "default"

        # Test with existing key
        d = {"key": "value"}
        result = safe_dict_get(d, "key", "default")
        assert result == "value"

        # Test with missing key
        result = safe_dict_get(d, "missing", "default")
        assert result == "default"

    def test_safe_dict_items(self):
        """Test safe dictionary items."""
        # Test with None
        items = safe_dict_items(None)
        assert items == []

        # Test with dict
        d = {"a": 1, "b": 2}
        items = safe_dict_items(d)
        assert sorted(items) == [("a", 1), ("b", 2)]

    def test_safe_defaultdict(self):
        """Test safe defaultdict creation."""
        dd = safe_defaultdict(list)
        dd["key"].append(1)
        assert dd["key"] == [1]
        assert dd["missing"] == []

    def test_safe_counter(self):
        """Test safe Counter creation."""
        # Test with None
        c = safe_counter()
        assert len(c) == 0

        # Test with items
        c = safe_counter(["a", "b", "a", "c", "b", "a"])
        assert c["a"] == 3
        assert c["b"] == 2

    def test_safe_deque(self):
        """Test safe deque creation."""
        # Test with None
        d = safe_deque()
        assert len(d) == 0

        # Test with items and maxlen
        d = safe_deque([1, 2, 3], maxlen=2)
        assert list(d) == [2, 3]


class TestStringOperations(unittest.TestCase):
    """Test string utility functions."""

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
        assert safe_join("-", [1, 2, None, 3]) == "1-2-None-3"


class TestLoggingOperations(unittest.TestCase):
    """Test logging utility functions."""

    def test_get_logger(self):
        """Test logger creation."""
        logger = get_logger("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"

        # Test same name returns same logger
        logger2 = get_logger("test_logger")
        assert logger is logger2

    def test_setup_basic_logging(self):
        """Test basic logging setup."""
        # Clear existing handlers
        logging.root.handlers = []

        setup_basic_logging(logging.DEBUG)

        # Check that handler was added
        assert len(logging.root.handlers) == 1
        assert logging.root.level == logging.DEBUG


class TestValidationOperations(unittest.TestCase):
    """Test validation utility functions."""

    def test_validate_dataframe(self):
        """Test DataFrame validation."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})

        # Test with all columns present
        assert validate_dataframe(df, ["a", "b"])

        # Test with missing columns
        assert not validate_dataframe(df, ["a", "d"])

        # Test with None
        assert not validate_dataframe(None, ["a"])

        # Test with empty DataFrame
        assert not validate_dataframe(pd.DataFrame(), ["a"])

    def test_validate_numeric_range(self):
        """Test numeric range validation."""
        assert validate_numeric_range(5, 0, 10)
        assert validate_numeric_range(0, 0, 10)
        assert validate_numeric_range(10, 0, 10)
        assert not validate_numeric_range(-1, 0, 10)
        assert not validate_numeric_range(11, 0, 10)

    def test_validate_dataframe_schema(self):
        """Test DataFrame schema validation."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.0, 2.0, 3.0],
            "str_col": ["a", "b", "c"],
        })

        # Test required columns only
        is_valid, errors = validate_dataframe_schema(df, ["int_col", "float_col"])
        assert is_valid
        assert errors == []

        # Test with missing columns
        is_valid, errors = validate_dataframe_schema(df, ["int_col", "missing_col"])
        assert not is_valid
        assert any("Missing columns" in e for e in errors)

        # Test with column types
        is_valid, errors = validate_dataframe_schema(
            df,
            ["int_col"],
            column_types={"int_col": np.integer},
        )
        assert is_valid

    def test_validate_data_quality(self):
        """Test data quality validation."""
        # Test good data
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
        })

        quality = validate_data_quality(df)
        assert quality["is_valid"]
        assert quality["total_rows"] == 5
        assert quality["total_columns"] == 2
        assert len(quality["issues"]) == 0

        # Test data with high NaN ratio
        df_nan = pd.DataFrame({
            "a": [1, np.nan, np.nan, 4, 5],
            "b": [10, 20, 30, 40, 50],
        })

        quality = validate_data_quality(df_nan, max_nan_ratio=0.1)
        assert not quality["is_valid"]
        assert any(issue["type"] == "high_nan_ratio" for issue in quality["issues"])

        # Test data with duplicates
        df_dup = pd.DataFrame({
            "a": [1, 1, 1, 1, 1],
            "b": [10, 10, 10, 10, 10],
        })

        quality = validate_data_quality(df_dup, check_duplicates=True)
        assert not quality["is_valid"]
        assert any(issue["type"] == "duplicates" for issue in quality["issues"])


class TestUtilityOperations(unittest.TestCase):
    """Test utility operations."""

    def test_timed_operation(self):
        """Test operation timing decorator."""
        @timed_operation("test_operation")
        def slow_function():
            time.sleep(0.1)
            return "done"

        with self.assertLogs(level="INFO") as cm:
            result = slow_function()
            assert result == "done"
            assert any("Starting test_operation" in log for log in cm.output)
            assert any("Completed test_operation" in log for log in cm.output)

        @timed_operation("failing_operation")
        def failing_function():
            msg = "Test error"
            raise ValueError(msg)

        with self.assertLogs(level="ERROR") as cm:
            with pytest.raises(ValueError):
                failing_function()
            assert any("Failed failing_operation" in log for log in cm.output)

    def test_format_bytes(self):
        """Test byte formatting."""
        assert format_bytes(0) == "0.00 B"
        assert format_bytes(1023) == "1023.00 B"
        assert format_bytes(1024) == "1.00 KB"
        assert format_bytes(1024 * 1024) == "1.00 MB"
        assert format_bytes(1024 * 1024 * 1024) == "1.00 GB"
        assert format_bytes(1024 * 1024 * 1024 * 1024) == "1.00 TB"
        assert format_bytes(1024 * 1024 * 1024 * 1024 * 1024) == "1.00 PB"

    def test_chunked_iterable(self):
        """Test iterable chunking."""
        # Test basic chunking
        items = list(range(10))
        chunks = chunked_iterable(items, 3)
        assert len(chunks) == 4
        assert chunks[0] == [0, 1, 2]
        assert chunks[1] == [3, 4, 5]
        assert chunks[2] == [6, 7, 8]
        assert chunks[3] == [9]

        # Test exact chunks
        items = list(range(9))
        chunks = chunked_iterable(items, 3)
        assert len(chunks) == 3
        assert all(len(chunk) == 3 for chunk in chunks)

        # Test empty list
        chunks = chunked_iterable([], 3)
        assert chunks == []

    def test_parallel_map(self):
        """Test parallel mapping."""
        def square(x):
            return x * x

        items = list(range(10))
        results = parallel_map(square, items, max_workers=2)
        expected = [x * x for x in items]
        assert results == expected

        # Test with None max_workers
        results = parallel_map(square, items)
        assert results == expected


class TestTypeConversions(unittest.TestCase):
    """Test type conversion utilities."""

    def test_safe_float(self):
        """Test safe float conversion."""
        assert safe_float("3.14") == 3.14
        assert safe_float("42") == 42.0
        assert safe_float(42) == 42.0
        assert safe_float("invalid", -1.0) == -1.0
        assert safe_float(None, 0.0) == 0.0

    def test_safe_int(self):
        """Test safe int conversion."""
        assert safe_int("42") == 42
        assert safe_int(42.7) == 42
        assert safe_int("invalid", -1) == -1
        assert safe_int(None, 0) == 0


class TestMLflowOperations(unittest.TestCase):
    """Test MLflow integration helpers."""

    @patch("mlflow.active_run")
    @patch("mlflow.log_metric")
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

    @patch("mlflow.active_run")
    @patch("mlflow.log_params")
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

    @patch("mlflow.active_run")
    @patch("mlflow.log_artifact")
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
