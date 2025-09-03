"""Unit tests for step1_data_collection.py.

This module tests the data collection functionality of the training pipeline.
"""

import unittest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import asyncio
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the module to be tested
from src.training.steps.step1_data_collection import DataCollectionStep


class TestDataCollectionStep(unittest.TestCase):
    """Test cases for DataCollectionStep class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "timeframe": "1m",
            "lookback_days": 30,
            "project_version": "1.0.0"
        }
        
        # Mock the logger
        with patch('src.training.steps.step1_data_collection.system_logger'):
            self.step = DataCollectionStep(self.config)
        
    def tearDown(self):
        """Clean up after tests."""
        pass

    def test_initialization(self):
        """Test DataCollectionStep initialization."""
        self.assertIsNotNone(self.step)
        self.assertEqual(self.step.config, self.config)
        
    def test_validate_environment(self):
        """Test environment validation."""
        with patch('src.training.steps.step1_data_collection.dependency_status', {'pandas': True, 'numpy': True}):
            # Should not raise any exceptions
            self.step._validate_environment()
            
    @patch('src.training.steps.step1_data_collection.download_all_data_with_consolidation')
    def test_run_data_collection_success(self, mock_download):
        """Test successful data collection."""
        # Set up mock
        mock_download.return_value = True
        
        # Run the async method
        async def run_test():
            training_input = {
                "symbol": "ETHUSDT",
                "exchange": "BINANCE",
                "timeframe": "1m",
                "lookback_days": 30
            }
            result = await self.step._run_data_collection(training_input, "/tmp/test_data")
            return result
            
        result = asyncio.run(run_test())
        self.assertTrue(result)
        
    @patch('src.training.steps.step1_data_collection.download_all_data_with_consolidation')
    def test_run_data_collection_failure(self, mock_download):
        """Test data collection failure handling."""
        # Set up mock to raise exception
        mock_download.side_effect = Exception("Download failed")
        
        # Run the async method
        async def run_test():
            training_input = {
                "symbol": "ETHUSDT",
                "exchange": "BINANCE",
                "timeframe": "1m",
                "lookback_days": 30
            }
            result = await self.step._run_data_collection(training_input, "/tmp/test_data")
            return result
            
        result = asyncio.run(run_test())
        self.assertFalse(result)
        
    @patch('pandas.read_parquet')
    @patch('os.path.exists')
    def test_run_standardized_quality_check_success(self, mock_exists, mock_read_parquet):
        """Test successful quality check."""
        # Set up mocks
        mock_exists.return_value = True
        
        # Create mock dataframe
        mock_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1min'),
            'open': np.random.rand(100) * 100,
            'high': np.random.rand(100) * 100,
            'low': np.random.rand(100) * 100,
            'close': np.random.rand(100) * 100,
            'volume': np.random.rand(100) * 1000
        })
        mock_read_parquet.return_value = mock_df
        
        # Run the async method
        async def run_test():
            result = await self.step._run_standardized_quality_check(
                "ETHUSDT", "BINANCE", "1m", "/tmp/test_data"
            )
            return result
            
        result = asyncio.run(run_test())
        self.assertTrue(result)
        
    @patch('os.path.exists')
    def test_run_standardized_quality_check_missing_files(self, mock_exists):
        """Test quality check with missing files."""
        # Set up mock
        mock_exists.return_value = False
        
        # Run the async method
        async def run_test():
            result = await self.step._run_standardized_quality_check(
                "ETHUSDT", "BINANCE", "1m", "/tmp/test_data"
            )
            return result
            
        result = asyncio.run(run_test())
        self.assertFalse(result)
        
    @patch('src.training.steps.step1_data_collection.DataCollectionStep._run_data_collection')
    @patch('src.training.steps.step1_data_collection.DataCollectionStep._run_standardized_quality_check')
    @patch('src.training.steps.step1_data_collection.DataCollectionStep._log_step1_artifacts_and_report')
    def test_execute_success(self, mock_log_artifacts, mock_quality_check, mock_data_collection):
        """Test successful execution of the step."""
        # Set up mocks
        mock_data_collection.return_value = asyncio.Future()
        mock_data_collection.return_value.set_result(True)
        
        mock_quality_check.return_value = asyncio.Future()
        mock_quality_check.return_value.set_result(True)
        
        mock_log_artifacts.return_value = asyncio.Future()
        mock_log_artifacts.return_value.set_result(None)
        
        # Run the async method
        async def run_test():
            training_input = {
                "symbol": "ETHUSDT",
                "exchange": "BINANCE",
                "timeframe": "1m",
                "lookback_days": 30
            }
            pipeline_state = {}
            
            # Patch the decorator
            with patch('src.training.steps.step1_data_collection.with_enhanced_mlflow_logging', lambda x: lambda fn: fn):
                result = await self.step.execute(training_input, pipeline_state)
            
            return result
            
        result = asyncio.run(run_test())
        
        self.assertTrue(result["data_collection_completed"])
        self.assertTrue(result["quality_check_passed"])
        
    @patch('src.training.steps.step1_data_collection.DataCollectionStep._run_data_collection')
    @patch('src.training.steps.step1_data_collection.DataCollectionStep._log_step1_artifacts_and_report')
    def test_execute_data_collection_failure(self, mock_log_artifacts, mock_data_collection):
        """Test execution when data collection fails."""
        # Set up mocks
        mock_data_collection.return_value = asyncio.Future()
        mock_data_collection.return_value.set_result(False)
        
        mock_log_artifacts.return_value = asyncio.Future()
        mock_log_artifacts.return_value.set_result(None)
        
        # Run the async method
        async def run_test():
            training_input = {
                "symbol": "ETHUSDT",
                "exchange": "BINANCE",
                "timeframe": "1m",
                "lookback_days": 30
            }
            pipeline_state = {}
            
            # Patch the decorator
            with patch('src.training.steps.step1_data_collection.with_enhanced_mlflow_logging', lambda x: lambda fn: fn):
                result = await self.step.execute(training_input, pipeline_state)
            
            return result
            
        result = asyncio.run(run_test())
        
        self.assertFalse(result["data_collection_completed"])
        self.assertFalse(result["quality_check_passed"])
        
    def test_execute_missing_required_parameters(self):
        """Test execution with missing required parameters."""
        async def run_test():
            training_input = {
                # Missing symbol and exchange
                "timeframe": "1m",
                "lookback_days": 30
            }
            pipeline_state = {}
            
            # Patch the decorator
            with patch('src.training.steps.step1_data_collection.with_enhanced_mlflow_logging', lambda x: lambda fn: fn):
                with patch('src.training.steps.step1_data_collection.DataCollectionStep._log_step1_artifacts_and_report', 
                          return_value=asyncio.Future()):
                    result = await self.step.execute(training_input, pipeline_state)
            
            return result
            
        result = asyncio.run(run_test())
        
        self.assertFalse(result["data_collection_completed"])
        self.assertFalse(result["quality_check_passed"])
        
    @patch('src.training.steps.step1_data_collection.create_detailed_step_report')
    @patch('src.training.steps.step1_data_collection.log_step_report')
    @patch('src.training.steps.step1_data_collection.log_step_metrics')
    def test_log_step1_artifacts_and_report(self, mock_log_metrics, mock_log_report, mock_create_report):
        """Test artifact and report logging."""
        # Set up mocks
        mock_create_report.return_value = {"report": "data"}
        mock_log_report.return_value = "report_name"
        
        # Run the async method
        async def run_test():
            training_input = {
                "symbol": "ETHUSDT",
                "exchange": "BINANCE",
                "timeframe": "1m",
                "lookback_days": 30
            }
            pipeline_state = {
                "data_collection_completed": True,
                "quality_check_passed": True
            }
            
            await self.step._log_step1_artifacts_and_report(training_input, pipeline_state)
            
        asyncio.run(run_test())
        
        # Verify methods were called
        self.assertTrue(mock_create_report.called)
        self.assertTrue(mock_log_report.called)
        self.assertTrue(mock_log_metrics.called)


class TestDataCollectionIntegration(unittest.TestCase):
    """Integration tests for data collection step."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = Path("/tmp/test_data_collection")
        self.test_data_dir.mkdir(exist_ok=True)
        
    def tearDown(self):
        """Clean up test data."""
        import shutil
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)
            
    @patch('src.training.steps.step1_data_collection.download_all_data_with_consolidation')
    def test_full_pipeline_integration(self, mock_download):
        """Test full pipeline integration."""
        # Create test data files
        test_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=1000, freq='1min'),
            'open': np.random.rand(1000) * 100 + 100,
            'high': np.random.rand(1000) * 100 + 110,
            'low': np.random.rand(1000) * 100 + 90,
            'close': np.random.rand(1000) * 100 + 100,
            'volume': np.random.rand(1000) * 10000
        })
        
        # Save test files
        klines_file = self.test_data_dir / "BINANCE_ETHUSDT_1m_klines.parquet"
        test_df.to_parquet(klines_file)
        
        # Mock download to return success
        mock_download.return_value = True
        
        # Run the test
        async def run_test():
            config = {
                "symbol": "ETHUSDT",
                "exchange": "BINANCE",
                "timeframe": "1m",
                "lookback_days": 30,
                "project_version": "1.0.0"
            }
            
            with patch('src.training.steps.step1_data_collection.system_logger'):
                step = DataCollectionStep(config)
                
            training_input = {
                "symbol": "ETHUSDT",
                "exchange": "BINANCE",
                "timeframe": "1m",
                "lookback_days": 30,
                "data_dir": str(self.test_data_dir)
            }
            pipeline_state = {}
            
            # Patch various dependencies
            with patch('src.training.steps.step1_data_collection.with_enhanced_mlflow_logging', lambda x: lambda fn: fn):
                with patch('src.training.steps.step1_data_collection.pipeline_standards.build_path', 
                          return_value=str(self.test_data_dir)):
                    with patch('src.training.steps.step1_data_collection.pipeline_standards.generate_file_name',
                              side_effect=lambda t, e, s, *args: f"{e}_{s}_1m_{t}.parquet"):
                        with patch('src.training.steps.step1_data_collection.pipeline_standards.standardize_timestamp',
                                  side_effect=lambda df, col: df):
                            result = await step.execute(training_input, pipeline_state)
            
            return result
            
        result = asyncio.run(run_test())
        
        # Verify results
        self.assertTrue(result["data_collection_completed"])
        # Quality check might fail due to missing aggtrades file, which is expected in this test


if __name__ == "__main__":
    unittest.main()