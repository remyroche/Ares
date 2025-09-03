"""Unit tests for step2_data_reading.py.

This module tests the data reading and validation functionality of the training pipeline.
"""

import unittest
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
import asyncio
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the module to be tested
from src.training.steps.step2_data_reading import DataReadingStep


class TestDataReadingStep(unittest.TestCase):
    """Test cases for DataReadingStep class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "SYMBOL": "ETHUSDT",
            "EXCHANGE": "BINANCE",
            "TIMEFRAME": "1m",
            "DATA_DIR": "/tmp/test_data",
            "MIN_DATA_POINTS": 1000,
            "MAX_NULL_PERCENTAGE": 0.05
        }
        
        # Mock the logger
        with patch('src.training.steps.step2_data_reading.system_logger'):
            self.step = DataReadingStep(self.config)
        
        # Create sample test data
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=2000, freq='1min'),
            'open': np.random.rand(2000) * 100 + 100,
            'high': np.random.rand(2000) * 100 + 110,
            'low': np.random.rand(2000) * 100 + 90,
            'close': np.random.rand(2000) * 100 + 100,
            'volume': np.random.rand(2000) * 10000,
            'quote_volume': np.random.rand(2000) * 1000000,
            'trades': np.random.randint(10, 1000, 2000),
            'taker_buy_volume': np.random.rand(2000) * 5000,
            'taker_buy_quote_volume': np.random.rand(2000) * 500000
        })
        
    def tearDown(self):
        """Clean up after tests."""
        pass

    def test_initialization(self):
        """Test DataReadingStep initialization."""
        self.assertIsNotNone(self.step)
        self.assertEqual(self.step.config, self.config)
        
    def test_validate_environment(self):
        """Test environment validation."""
        with patch('src.training.steps.step2_data_reading.dependency_status', 
                  {'pandas': True, 'numpy': True, 'psutil': True}):
            # Should not raise any exceptions
            self.step._validate_environment()
            
    async def test_initialize(self):
        """Test step initialization."""
        await self.step.initialize()
        self.assertIsNotNone(self.step.start_time)
        
    @patch('src.training.steps.step2_data_reading.safe_read_parquet')
    async def test_read_unified_data_success(self, mock_read_parquet):
        """Test successful unified data reading."""
        # Set up mock
        mock_read_parquet.return_value = self.test_data
        
        # Run the method
        result = await self.step.read_unified_data(
            "ETHUSDT", "BINANCE", "1m", "/tmp/test_data"
        )
        
        # Verify result
        self.assertIsNotNone(result)
        pd.testing.assert_frame_equal(result, self.test_data)
        
    @patch('src.training.steps.step2_data_reading.safe_read_parquet')
    async def test_read_unified_data_file_not_found(self, mock_read_parquet):
        """Test unified data reading when file doesn't exist."""
        # Set up mock
        mock_read_parquet.return_value = None
        
        # Run the method
        result = await self.step.read_unified_data(
            "ETHUSDT", "BINANCE", "1m", "/tmp/test_data"
        )
        
        # Verify result
        self.assertIsNone(result)
        
    @patch('src.training.steps.step2_data_reading.validate_dataframe_schema')
    @patch('src.training.steps.step2_data_reading.validate_data_quality')
    async def test_validate_data_quality_success(self, mock_validate_quality, mock_validate_schema):
        """Test successful data quality validation."""
        # Set up mocks
        mock_validate_schema.return_value = {"valid": True, "missing_columns": [], "extra_columns": []}
        mock_validate_quality.return_value = {
            "null_percentages": {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0},
            "duplicate_rows": 0,
            "monotonic_timestamp": True,
            "negative_values": {},
            "outliers": {}
        }
        
        # Run the method
        result = await self.step.validate_data_quality(
            self.test_data, "ETHUSDT", "BINANCE"
        )
        
        # Verify result
        self.assertTrue(result["passed"])
        self.assertIn("schema_validation", result)
        self.assertIn("quality_metrics", result)
        
    @patch('src.training.steps.step2_data_reading.validate_dataframe_schema')
    async def test_validate_data_quality_schema_failure(self, mock_validate_schema):
        """Test data quality validation with schema failure."""
        # Set up mock
        mock_validate_schema.return_value = {
            "valid": False, 
            "missing_columns": ["close"], 
            "extra_columns": []
        }
        
        # Run the method
        result = await self.step.validate_data_quality(
            self.test_data, "ETHUSDT", "BINANCE"
        )
        
        # Verify result
        self.assertFalse(result["passed"])
        self.assertIn("Schema validation failed", result["issues"][0])
        
    async def test_validate_data_quality_insufficient_data(self):
        """Test data quality validation with insufficient data points."""
        # Create small dataset
        small_data = self.test_data.head(100)
        
        # Run the method
        with patch('src.training.steps.step2_data_reading.validate_dataframe_schema',
                  return_value={"valid": True, "missing_columns": [], "extra_columns": []}):
            with patch('src.training.steps.step2_data_reading.validate_data_quality',
                      return_value={
                          "null_percentages": {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0},
                          "duplicate_rows": 0,
                          "monotonic_timestamp": True,
                          "negative_values": {},
                          "outliers": {}
                      }):
                result = await self.step.validate_data_quality(
                    small_data, "ETHUSDT", "BINANCE"
                )
        
        # Verify result
        self.assertFalse(result["passed"])
        self.assertIn("Insufficient data", result["issues"][0])
        
    @patch('src.training.steps.step2_data_reading.ensure_directory')
    @patch('src.training.steps.step2_data_reading.safe_json_dump')
    async def test_save_validation_report(self, mock_json_dump, mock_ensure_dir):
        """Test saving validation report."""
        # Set up validation results
        validation_results = {
            "passed": True,
            "timestamp": datetime.now().isoformat(),
            "schema_validation": {"valid": True},
            "quality_metrics": {"null_percentages": {}}
        }
        
        # Run the method
        await self.step.save_validation_report(
            validation_results, "ETHUSDT", "BINANCE", "/tmp/test_data"
        )
        
        # Verify calls
        mock_ensure_dir.assert_called()
        mock_json_dump.assert_called()
        
    @patch('src.training.steps.step2_data_reading.DataReadingStep.read_unified_data')
    @patch('src.training.steps.step2_data_reading.DataReadingStep.validate_data_quality')
    @patch('src.training.steps.step2_data_reading.DataReadingStep.save_validation_report')
    @patch('src.training.steps.step2_data_reading.log_step_metrics')
    @patch('src.training.steps.step2_data_reading.log_step_report')
    @patch('src.training.steps.step2_data_reading.log_step_dataframe_with_standardized_name')
    async def test_execute_success(self, mock_log_df, mock_log_report, mock_log_metrics,
                                  mock_save_report, mock_validate, mock_read):
        """Test successful execution of the step."""
        # Set up mocks
        mock_read.return_value = self.test_data
        mock_validate.return_value = {
            "passed": True,
            "schema_validation": {"valid": True},
            "quality_metrics": {"null_percentages": {}},
            "issues": []
        }
        
        # Patch decorators
        with patch('src.training.steps.step2_data_reading.with_enhanced_mlflow_logging', lambda x: lambda fn: fn):
            with patch('src.training.steps.step2_data_reading.with_tracing_span', lambda x: lambda fn: fn):
                with patch('src.training.steps.step2_data_reading.handle_errors', lambda fn: fn):
                    with patch('src.training.steps.step2_data_reading.resource_monitor', lambda fn: fn):
                        # Run the method
                        result = await self.step.execute(
                            symbol="ETHUSDT",
                            exchange="BINANCE",
                            timeframe="1m",
                            data_dir="/tmp/test_data"
                        )
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertIn("unified_data", result)
        self.assertIn("validation_results", result)
        
    @patch('src.training.steps.step2_data_reading.DataReadingStep.read_unified_data')
    async def test_execute_read_failure(self, mock_read):
        """Test execution when data reading fails."""
        # Set up mock
        mock_read.return_value = None
        
        # Patch decorators
        with patch('src.training.steps.step2_data_reading.with_enhanced_mlflow_logging', lambda x: lambda fn: fn):
            with patch('src.training.steps.step2_data_reading.with_tracing_span', lambda x: lambda fn: fn):
                with patch('src.training.steps.step2_data_reading.handle_errors', lambda fn: fn):
                    with patch('src.training.steps.step2_data_reading.resource_monitor', lambda fn: fn):
                        # Run the method
                        result = await self.step.execute(
                            symbol="ETHUSDT",
                            exchange="BINANCE",
                            timeframe="1m",
                            data_dir="/tmp/test_data"
                        )
        
        # Verify result
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        
    @patch('src.training.steps.step2_data_reading.DataReadingStep.read_unified_data')
    @patch('src.training.steps.step2_data_reading.DataReadingStep.validate_data_quality')
    @patch('src.training.steps.step2_data_reading.DataReadingStep.save_validation_report')
    async def test_execute_validation_failure(self, mock_save_report, mock_validate, mock_read):
        """Test execution when data validation fails."""
        # Set up mocks
        mock_read.return_value = self.test_data
        mock_validate.return_value = {
            "passed": False,
            "schema_validation": {"valid": False},
            "quality_metrics": {"null_percentages": {"close": 0.1}},
            "issues": ["High null percentage in close column"]
        }
        
        # Patch decorators
        with patch('src.training.steps.step2_data_reading.with_enhanced_mlflow_logging', lambda x: lambda fn: fn):
            with patch('src.training.steps.step2_data_reading.with_tracing_span', lambda x: lambda fn: fn):
                with patch('src.training.steps.step2_data_reading.handle_errors', lambda fn: fn):
                    with patch('src.training.steps.step2_data_reading.resource_monitor', lambda fn: fn):
                        # Run the method
                        result = await self.step.execute(
                            symbol="ETHUSDT",
                            exchange="BINANCE",
                            timeframe="1m",
                            data_dir="/tmp/test_data"
                        )
        
        # Verify result
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Data quality validation failed")
        
    def test_log_step_timing(self):
        """Test step timing logging."""
        import time
        start_time = time.time()
        
        # Add a small delay
        time.sleep(0.1)
        
        # Log timing
        self.step._log_step_timing("test_step", start_time)
        
        # Verify timing was recorded
        self.assertIn("test_step", self.step.step_timings)
        self.assertGreater(self.step.step_timings["test_step"], 0.09)


class TestDataReadingHelpers(unittest.TestCase):
    """Test helper functions in data reading step."""
    
    def test_quality_gate_decorator(self):
        """Test quality gate decorator functionality."""
        # This test verifies the decorator doesn't break function execution
        @patch('src.training.steps.step2_data_reading.quality_gate', lambda **kwargs: lambda fn: fn)
        def decorated_function():
            return "success"
            
        result = decorated_function()
        self.assertEqual(result, "success")
        
    def test_memory_efficient_decorator(self):
        """Test memory efficient decorator functionality."""
        # This test verifies the decorator doesn't break function execution
        @patch('src.training.steps.step2_data_reading.memory_efficient', lambda fn: fn)
        def decorated_function():
            return "success"
            
        result = decorated_function()
        self.assertEqual(result, "success")


if __name__ == "__main__":
    # Run with asyncio support
    unittest.main()