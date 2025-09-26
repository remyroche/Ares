from src.utils.tprint import tprint

"""
Comprehensive unit tests for common_operations module.

Tests all functions including edge cases, error conditions, and expected behaviors.
"""

import asyncio
import logging
import shutil
import sys
import tempfile
import time
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    import pytest
except ImportError:
    pytest = None
    tprint("Warning: pytest not available")

from src.utils.common_operations import *
import collections
import json


class TestCommonOperations(unittest.TestCase):
    """Consolidated test class for all common operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test.json"
        self.test_df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": ["x", "y", "z"],
        })

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    # DateTime Operations Tests
    def test_datetime_operations(self):
        """Test datetime utility functions."""
        # Test get_current_datetime
        result = get_current_datetime()
        assert isinstance(result, datetime)
        time_diff = datetime.now() - result
        assert time_diff.total_seconds() < 60

        # Test get_today
        result = get_today()
        assert isinstance(result, date)
        assert result == date.today()

        # Test format_datetime
        dt = datetime(2024, 1, 15, 10, 30, 45)
        assert format_datetime(dt) == "2024-01-15 10:30:45"
        assert format_datetime(dt, "%Y%m%d") == "20240115"
        assert format_datetime(dt, "%Y-%m-%dT%H:%M:%S") == "2024-01-15T10:30:45"

        # Test parse_datetime
        result = parse_datetime("2024-01-15 10:30:45")
        expected = datetime(2024, 1, 15, 10, 30, 45)
        assert result == expected

        result = parse_datetime("20240115", "%Y%m%d")
        expected = datetime(2024, 1, 15, 0, 0, 0)
        assert result == expected

        if pytest:
            with pytest.raises(ValueError):
                parse_datetime("invalid date")


    # DataFrame Operations Tests
    def test_dataframe_operations(self):
        """Test DataFrame utility functions."""
        # Test create_empty_dataframe
        columns = ["a", "b", "c"]
        df = create_empty_dataframe(columns)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == columns
        assert len(df) == 0

        # Test safe_fillna
        df_nan = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, np.nan]})
        result = safe_fillna(df_nan, 0)
        assert result["a"].tolist() == [1.0, 0.0, 3.0]
        assert result["b"].tolist() == [0.0, 2.0, 0.0]

        result = safe_fillna(df_nan, -1)
        assert result["a"].tolist() == [1.0, -1.0, 3.0]

        # Test with no NaN values
        df_no_nan = pd.DataFrame({"a": [1, 2, 3]})
        result = safe_fillna(df_no_nan, 0)
        pd.testing.assert_frame_equal(result, df_no_nan)

        # Test safe_rolling
        df_rolling = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        rolling = safe_rolling(df_rolling, window=3)
        assert isinstance(rolling, pd.core.window.Rolling)

        result = rolling.mean()
        expected = [np.nan, np.nan, 2.0, 3.0, 4.0]
        np.testing.assert_array_almost_equal(result["a"].values, expected)

        # Test with min_periods
        rolling = safe_rolling(df_rolling, window=3, min_periods=1)
        result = rolling.mean()
        expected = [1.0, 1.5, 2.0, 3.0, 4.0]
        np.testing.assert_array_almost_equal(result["a"].values, expected)

        # Test safe_copy
        df_copy = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        copy_df = safe_copy(df_copy)
        copy_df.iloc[0, 0] = 999
        assert df_copy.iloc[0, 0] == 1  # Original unchanged

        copy_df = safe_copy(df_copy, deep=False)
        assert copy_df is not df_copy

        # Test safe_resample
        dates = pd.date_range("2024-01-01", periods=24, freq="H")
        df_ts = pd.DataFrame({
            "close": range(24),
            "volume": range(100, 124),
        }, index=dates)

        result = safe_resample(df_ts, "4H")
        assert len(result) == 6
        assert result.iloc[0]["close"] == 3
        assert result.iloc[0]["volume"] == 406

        agg_dict = {"close": "mean", "volume": "max"}
        result = safe_resample(df_ts, "4H", agg_dict)
        assert result.iloc[0]["close"] == 1.5
        assert result.iloc[0]["volume"] == 103

        if pytest:
            df_no_dt = pd.DataFrame({"a": [1, 2, 3]})
            with pytest.raises(ValueError):
                safe_resample(df_no_dt, "1H")


    # Numeric Operations Tests
    def test_numeric_operations(self):
        """Test numeric utility functions."""
        # Test safe_mean
        assert safe_mean([1, 2, 3, 4, 5]) == 3.0
        assert safe_mean(np.array([1, 2, 3, 4, 5])) == 3.0
        assert safe_mean(pd.Series([1, 2, 3, 4, 5])) == 3.0
        assert np.isnan(safe_mean([]))
        assert safe_mean([1, np.nan, 3]) == 2.0

        # Test safe_std
        result = safe_std([1, 2, 3, 4, 5])
        self.assertAlmostEqual(result, 1.4142135623730951)
        assert np.isnan(safe_std([]))
        assert safe_std([5]) == 0.0
        result = safe_std([1, np.nan, 3])
        self.assertAlmostEqual(result, 1.0)


    # File Operations Tests
    def test_file_operations(self):
        """Test file operation utilities."""
        # Test ensure_directory
        new_dir = Path(self.temp_dir) / "new" / "nested" / "dir"
        result = ensure_directory(new_dir)
        assert isinstance(result, Path)
        assert result.exists()
        assert result.is_dir()

        result2 = ensure_directory(new_dir)
        assert result == result2

        str_dir = str(Path(self.temp_dir) / "string_dir")
        result = ensure_directory(str_dir)
        assert result.exists()

        # Test safe_file_exists
        assert not safe_file_exists(self.test_file)
        self.test_file.write_text("test")
        assert safe_file_exists(self.test_file)
        assert safe_file_exists(str(self.test_file))
        assert not safe_file_exists("\x00invalid\x00path")

        # Test JSON operations
        test_data = {
            "string": "value",
            "number": 42,
            "float": 3.14,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
        }

        safe_json_dump(test_data, self.test_file, indent=2)
        assert self.test_file.exists()

        loaded_data = safe_json_load(self.test_file)
        assert loaded_data == test_data

        if pytest:
            with pytest.raises(FileNotFoundError):
                safe_json_load("non_existent.json")


    # Parquet Operations Tests
    def test_parquet_operations(self):
        """Test parquet file operations."""
        file_path = Path(self.temp_dir) / "test.parquet"

        # Test safe_to_parquet
        success = safe_to_parquet(self.test_df, file_path)
        assert success
        assert file_path.exists()

        file_path2 = Path(self.temp_dir) / "test_compressed.parquet"
        success = safe_to_parquet(self.test_df, file_path2, compression="snappy")
        assert success

        # Test safe_read_parquet
        df = safe_read_parquet(file_path)
        pd.testing.assert_frame_equal(df, self.test_df)

        df = safe_read_parquet(file_path, columns=["a", "c"])
        assert list(df.columns) == ["a", "c"]

        df = safe_read_parquet("non_existent.parquet")
        assert df.empty

        # Test list_parquet_files
        (Path(self.temp_dir) / "file1.parquet").touch()
        (Path(self.temp_dir) / "file2.parquet").touch()
        (Path(self.temp_dir) / "subdir").mkdir()
        (Path(self.temp_dir) / "subdir" / "file3.parquet").touch()
        (Path(self.temp_dir) / "other.txt").touch()

        files = list_parquet_files(self.temp_dir, recursive=True)
        assert len(files) == 3
        assert all(f.suffix == ".parquet" for f in files)

        files = list_parquet_files(self.temp_dir, recursive=False)
        assert len(files) == 2


    # Hashing Operations Tests
    def test_hashing_operations(self):
        """Test hashing and cache key operations."""
        # Test generate_hash
        hash1 = generate_hash("test string")
        assert len(hash1) == 32  # MD5 length

        hash2 = generate_hash("test string")
        assert hash1 == hash2

        hash3 = generate_hash("different string")
        assert hash1 != hash3

        hash_sha = generate_hash("test", algorithm="sha256")
        assert len(hash_sha) == 64  # SHA256 length

        df = pd.DataFrame({"a": [1, 2, 3]})
        hash_df = generate_hash(df)
        assert len(hash_df) == 32

        if pytest:
            with pytest.raises(ValueError):
                generate_hash("test", algorithm="invalid")

        # Test generate_cache_key
        key = generate_cache_key("features", "BTCUSDT", "1h")
        assert len(key) == 16  # Default max_length

        key2 = generate_cache_key("features", "BTCUSDT", "1h")
        assert key == key2

        key3 = generate_cache_key("features", "ETHUSDT", "1h")
        assert key != key3

        key_long = generate_cache_key("test", "data", max_length=32)
        assert len(key_long) == 32


    # Async Operations Tests
    def test_async_operations(self):
        """Test async utility functions."""
        async def test_sleep():
            start = asyncio.get_event_loop().time()
            await safe_sleep(0.1)
            elapsed = asyncio.get_event_loop().time() - start
            assert elapsed >= 0.1
            assert elapsed < 0.2

        asyncio.run(test_sleep())

        async def task(n):
            await safe_sleep(0.01)
            return n * 2

        async def failing_task():
            raise ValueError("Test error")

        async def test_gather():
            results = await safe_gather(task(1), task(2), task(3))
            assert results == [2, 4, 6]

            results = await safe_gather(task(1), failing_task(), task(3))
            assert results[0] == 2
            assert isinstance(results[1], ValueError)
            assert results[2] == 6

            if pytest:
                with pytest.raises(ValueError):
                    await safe_gather(task(1), failing_task(), return_exceptions=False)

        asyncio.run(test_gather())

        async def background_job():
            await safe_sleep(0.01)
            return "done"

        async def test_task():
            task = create_async_task(background_job())
            assert isinstance(task, asyncio.Task)
            result = await task
            assert result == "done"

        asyncio.run(test_task())


    # Collection Operations Tests
    def test_collection_operations(self):
        """Test collection utility functions."""
        # Test safe_append
        lst = safe_append(None, "item")
        assert lst == ["item"]

        lst = safe_append(lst, "item2")
        assert lst == ["item", "item2"]

        lst = safe_append([], 42)
        assert lst == [42]

        # Test safe_extend
        lst = safe_extend(None, [1, 2, 3])
        assert lst == [1, 2, 3]

        lst = safe_extend([0], [1, 2, 3])
        assert lst == [0, 1, 2, 3]

        # Test safe_dict_get
        result = safe_dict_get(None, "key", "default")
        assert result == "default"

        d = {"key": "value"}
        result = safe_dict_get(d, "key", "default")
        assert result == "value"

        result = safe_dict_get(d, "missing", "default")
        assert result == "default"

        # Test safe_dict_items
        items = safe_dict_items(None)
        assert items == []

        d = {"a": 1, "b": 2}
        items = safe_dict_items(d)
        assert sorted(items) == [("a", 1), ("b", 2)]

        # Test safe_defaultdict
        dd = safe_defaultdict(list)
        dd["key"].append(1)
        assert dd["key"] == [1]
        assert dd["missing"] == []

        # Test safe_counter
        c = safe_counter()
        assert len(c) == 0

        c = safe_counter(["a", "b", "a", "c", "b", "a"])
        assert c["a"] == 3
        assert c["b"] == 2

        # Test safe_deque
        d = safe_deque()
        assert len(d) == 0

        d = safe_deque([1, 2, 3], maxlen=2)
        assert list(d) == [2, 3]


    # String Operations Tests
    def test_string_operations(self):
        """Test string utility functions."""
        assert safe_lower("HELLO") == "hello"
        assert safe_lower(None) == ""
        assert safe_lower(123) == "123"

        assert safe_upper("hello") == "HELLO"
        assert safe_upper(None) == ""
        assert safe_upper(123) == "123"

        assert safe_join(", ", ["a", "b", "c"]) == "a, b, c"
        assert safe_join(", ", None) == ""
        assert safe_join("-", [1, 2, None, 3]) == "1-2-None-3"

    # Logging Operations Tests
    def test_logging_operations(self):
        """Test logging utility functions."""
        logger = get_logger("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"

        logger2 = get_logger("test_logger")
        assert logger is logger2

        # Clear existing handlers
        logging.root.handlers = []
        setup_basic_logging(logging.DEBUG)
        assert len(logging.root.handlers) == 1
        assert logging.root.level == logging.DEBUG


    # Validation Operations Tests
    def test_validation_operations(self):
        """Test validation utility functions."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})

        # Test validate_dataframe
        assert validate_dataframe(df, ["a", "b"])
        assert not validate_dataframe(df, ["a", "d"])
        assert not validate_dataframe(None, ["a"])
        assert not validate_dataframe(pd.DataFrame(), ["a"])

        # Test validate_numeric_range
        assert validate_numeric_range(5, 0, 10)
        assert validate_numeric_range(0, 0, 10)
        assert validate_numeric_range(10, 0, 10)
        assert not validate_numeric_range(-1, 0, 10)
        assert not validate_numeric_range(11, 0, 10)

        # Test validate_dataframe_schema
        df_schema = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.0, 2.0, 3.0],
            "str_col": ["a", "b", "c"],
        })

        is_valid, errors = validate_dataframe_schema(df_schema, ["int_col", "float_col"])
        assert is_valid
        assert errors == []

        is_valid, errors = validate_dataframe_schema(df_schema, ["int_col", "missing_col"])
        assert not is_valid
        assert any("Missing columns" in e for e in errors)

        is_valid, errors = validate_dataframe_schema(
            df_schema,
            ["int_col"],
            column_types={"int_col": np.integer},
        )
        assert is_valid

        # Test validate_data_quality
        df_good = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
        })

        quality = validate_data_quality(df_good)
        assert quality["is_valid"]
        assert quality["total_rows"] == 5
        assert quality["total_columns"] == 2
        assert len(quality["issues"]) == 0

        df_nan = pd.DataFrame({
            "a": [1, np.nan, np.nan, 4, 5],
            "b": [10, 20, 30, 40, 50],
        })

        quality = validate_data_quality(df_nan, max_nan_ratio=0.1)
        assert not quality["is_valid"]
        assert any(issue["type"] == "high_nan_ratio" for issue in quality["issues"])

        df_dup = pd.DataFrame({
            "a": [1, 1, 1, 1, 1],
            "b": [10, 10, 10, 10, 10],
        })

        quality = validate_data_quality(df_dup, check_duplicates=True)
        assert not quality["is_valid"]
        assert any(issue["type"] == "duplicates" for issue in quality["issues"])


    # Utility Operations Tests
    def test_utility_operations(self):
        """Test utility operations."""
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
            raise ValueError("Test error")

        with self.assertLogs(level="ERROR") as cm:
            if pytest:
                with pytest.raises(ValueError):
                    failing_function()
            assert any("Failed failing_operation" in log for log in cm.output)

        # Test format_bytes
        assert format_bytes(0) == "0.00 B"
        assert format_bytes(1023) == "1023.00 B"
        assert format_bytes(1024) == "1.00 KB"
        assert format_bytes(1024 * 1024) == "1.00 MB"
        assert format_bytes(1024 * 1024 * 1024) == "1.00 GB"
        assert format_bytes(1024 * 1024 * 1024 * 1024) == "1.00 TB"
        assert format_bytes(1024 * 1024 * 1024 * 1024 * 1024) == "1.00 PB"

        # Test chunked_iterable
        items = list(range(10))
        chunks = chunked_iterable(items, 3)
        assert len(chunks) == 4
        assert chunks[0] == [0, 1, 2]
        assert chunks[1] == [3, 4, 5]
        assert chunks[2] == [6, 7, 8]
        assert chunks[3] == [9]

        items = list(range(9))
        chunks = chunked_iterable(items, 3)
        assert len(chunks) == 3
        assert all(len(chunk) == 3 for chunk in chunks)

        chunks = chunked_iterable([], 3)
        assert chunks == []

        # Test parallel_map
        def square(x):
            return x * x

        items = list(range(10))
        results = parallel_map(square, items, max_workers=2)
        expected = [x * x for x in items]
        assert results == expected

        results = parallel_map(square, items)
        assert results == expected

    # Type Conversions Tests
    def test_type_conversions(self):
        """Test type conversion utilities."""
        assert safe_float("3.14") == 3.14
        assert safe_float("42") == 42.0
        assert safe_float(42) == 42.0
        assert safe_float("invalid", -1.0) == -1.0
        assert safe_float(None, 0.0) == 0.0

        assert safe_int("42") == 42
        assert safe_int(42.7) == 42
        assert safe_int("invalid", -1) == -1
        assert safe_int(None, 0) == 0

    # MLflow Operations Tests
    @patch("mlflow.active_run")
    @patch("mlflow.log_metric")
    @patch("mlflow.log_params")
    @patch("mlflow.log_artifact")
    def test_mlflow_operations(self, mock_log_artifact, mock_log_params, mock_log_metric, mock_active_run):
        """Test MLflow integration helpers."""
        # Test safe_log_metric
        mock_active_run.return_value = MagicMock()
        safe_log_metric("accuracy", 0.95)
        mock_log_metric.assert_called_once_with("accuracy", 0.95, None)

        mock_log_metric.reset_mock()
        safe_log_metric("loss", 0.05, step=10)
        mock_log_metric.assert_called_once_with("loss", 0.05, 10)

        mock_active_run.return_value = None
        mock_log_metric.reset_mock()
        safe_log_metric("accuracy", 0.95)
        mock_log_metric.assert_not_called()

        # Test safe_log_params
        params = {"learning_rate": 0.01, "batch_size": 32}
        mock_active_run.return_value = MagicMock()
        safe_log_params(params)
        mock_log_params.assert_called_once_with(params)

        mock_active_run.return_value = None
        mock_log_params.reset_mock()
        safe_log_params(params)
        mock_log_params.assert_not_called()

        # Test safe_log_artifact
        mock_active_run.return_value = MagicMock()
        safe_log_artifact("model.pkl")
        mock_log_artifact.assert_called_once_with("model.pkl")

        mock_log_artifact.reset_mock()
        safe_log_artifact(Path("model.pkl"))
        mock_log_artifact.assert_called_once_with("model.pkl")

        mock_active_run.return_value = None
        mock_log_artifact.reset_mock()
        safe_log_artifact("model.pkl")
        mock_log_artifact.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
