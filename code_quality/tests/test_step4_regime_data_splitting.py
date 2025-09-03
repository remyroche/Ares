"""Unit tests for step4_regime_data_splitting.py and step4_triple_barrier_method.py.

This module tests the regime data splitting and triple barrier method functionality.
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

# Import the modules to be tested
from src.training.steps.step4_regime_data_splitting import RegimeDataSplittingStep
from src.training.steps.step4_triple_barrier_method import TripleBarrierMethodStep


class TestRegimeDataSplittingStep(unittest.TestCase):
    """Test cases for RegimeDataSplittingStep class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "SYMBOL": "ETHUSDT",
            "EXCHANGE": "BINANCE", 
            "TIMEFRAME": "1m",
            "DATA_DIR": "/tmp/test_data",
            "MIN_REGIME_DURATION": 100
        }
        
        # Mock the logger
        with patch('src.training.steps.step4_regime_data_splitting.system_logger'):
            self.step = RegimeDataSplittingStep(self.config)
        
        # Create sample test data with regimes
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=2000, freq='1min'),
            'open': np.random.rand(2000) * 100 + 100,
            'high': np.random.rand(2000) * 100 + 110,
            'low': np.random.rand(2000) * 100 + 90,
            'close': np.random.rand(2000) * 100 + 100,
            'volume': np.random.rand(2000) * 10000,
            'regime': np.array([0] * 500 + [1] * 500 + [2] * 500 + [0] * 500)
        })
        
    def tearDown(self):
        """Clean up after tests."""
        pass

    def test_initialization(self):
        """Test RegimeDataSplittingStep initialization."""
        self.assertIsNotNone(self.step)
        self.assertEqual(self.step.config, self.config)
        
    def test_validate_environment(self):
        """Test environment validation."""
        with patch('src.training.steps.step4_regime_data_splitting.dependency_status', 
                  {'pandas': True, 'numpy': True}):
            # Should not raise any exceptions
            self.step._validate_environment()
            
    async def test_initialize(self):
        """Test step initialization."""
        await self.step.initialize()
        self.assertIsNotNone(self.step.start_time)
        
    @patch('src.training.steps.step4_regime_data_splitting.pandas.read_parquet')
    async def test_load_regime_data_success(self, mock_read_parquet):
        """Test successful regime data loading."""
        # Set up mock
        mock_read_parquet.return_value = self.test_data
        
        # Run the method
        result = await self.step._load_regime_data(
            "ETHUSDT", "BINANCE", "1m", "/tmp/test_data"
        )
        
        # Verify result
        self.assertIsNotNone(result)
        pd.testing.assert_frame_equal(result, self.test_data)
        
    @patch('src.training.steps.step4_regime_data_splitting.pandas.read_parquet')
    async def test_load_regime_data_file_not_found(self, mock_read_parquet):
        """Test regime data loading when file doesn't exist."""
        # Set up mock to raise FileNotFoundError
        mock_read_parquet.side_effect = FileNotFoundError("File not found")
        
        # Run the method
        result = await self.step._load_regime_data(
            "ETHUSDT", "BINANCE", "1m", "/tmp/test_data"
        )
        
        # Verify result
        self.assertIsNone(result)
        
    async def test_create_unified_regime_dataset(self):
        """Test creating unified regime dataset."""
        # Run the method
        result = await self.step._create_unified_regime_dataset(
            self.test_data, [0, 1, 2], "/tmp/test_data", "ETHUSDT", "BINANCE", "1m"
        )
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn("unified_data", result)
        self.assertIn("regime_stats", result)
        self.assertIn("saved_path", result)
        
    def test_calculate_regime_statistics(self):
        """Test regime statistics calculation."""
        # Run the method
        stats = self.step._calculate_regime_statistics(self.test_data, [0, 1, 2])
        
        # Verify statistics
        self.assertIsInstance(stats, dict)
        self.assertEqual(len(stats), 3)  # 3 regimes
        for regime_id in [0, 1, 2]:
            self.assertIn(regime_id, stats)
            self.assertIn("count", stats[regime_id])
            self.assertIn("duration_minutes", stats[regime_id])
            self.assertIn("mean_volume", stats[regime_id])
            
    @patch('src.training.steps.step4_regime_data_splitting.ensure_directory')
    @patch('src.training.steps.step4_regime_data_splitting.safe_json_dump')
    async def test_save_regime_metadata(self, mock_json_dump, mock_ensure_dir):
        """Test saving regime metadata."""
        # Run the method
        await self.step._save_regime_metadata(
            [0, 1, 2], "/tmp/test_data", "ETHUSDT", "BINANCE", "1m"
        )
        
        # Verify calls
        mock_ensure_dir.assert_called()
        mock_json_dump.assert_called()
        
    @patch('src.training.steps.step4_regime_data_splitting.RegimeDataSplittingStep._load_regime_data')
    @patch('src.training.steps.step4_regime_data_splitting.RegimeDataSplittingStep._create_unified_regime_dataset')
    @patch('src.training.steps.step4_regime_data_splitting.RegimeDataSplittingStep._save_regime_metadata')
    @patch('src.training.steps.step4_regime_data_splitting.log_step_metrics')
    @patch('src.training.steps.step4_regime_data_splitting.log_step_report')
    async def test_split_data_by_regimes_success(self, mock_log_report, mock_log_metrics,
                                                mock_save_metadata, mock_create_unified, mock_load):
        """Test successful regime data splitting."""
        # Set up mocks
        mock_load.return_value = self.test_data
        mock_create_unified.return_value = {
            "unified_data": self.test_data,
            "regime_stats": {0: {"count": 1000}, 1: {"count": 500}, 2: {"count": 500}},
            "saved_path": "/tmp/test_data/unified_regime_data.parquet"
        }
        
        # Patch decorators
        with patch('src.training.steps.step4_regime_data_splitting.with_enhanced_mlflow_logging', lambda x: lambda fn: fn):
            with patch('src.training.steps.step4_regime_data_splitting.with_tracing_span', lambda x: lambda fn: fn):
                with patch('src.training.steps.step4_regime_data_splitting.quality_gate', lambda **kwargs: lambda fn: fn):
                    with patch('src.training.steps.step4_regime_data_splitting.handle_errors', lambda fn: fn):
                        with patch('src.training.steps.step4_regime_data_splitting.resource_monitor', lambda fn: fn):
                            # Run the method
                            result = await self.step.split_data_by_regimes(
                                symbol="ETHUSDT",
                                exchange="BINANCE",
                                timeframe="1m",
                                data_dir="/tmp/test_data"
                            )
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertIn("unified_data", result)
        self.assertIn("regime_stats", result)
        
    @patch('src.training.steps.step4_regime_data_splitting.RegimeDataSplittingStep._load_regime_data')
    async def test_split_data_by_regimes_load_failure(self, mock_load):
        """Test regime data splitting when loading fails."""
        # Set up mock
        mock_load.return_value = None
        
        # Patch decorators
        with patch('src.training.steps.step4_regime_data_splitting.with_enhanced_mlflow_logging', lambda x: lambda fn: fn):
            with patch('src.training.steps.step4_regime_data_splitting.with_tracing_span', lambda x: lambda fn: fn):
                with patch('src.training.steps.step4_regime_data_splitting.quality_gate', lambda **kwargs: lambda fn: fn):
                    with patch('src.training.steps.step4_regime_data_splitting.handle_errors', lambda fn: fn):
                        with patch('src.training.steps.step4_regime_data_splitting.resource_monitor', lambda fn: fn):
                            # Run the method
                            result = await self.step.split_data_by_regimes(
                                symbol="ETHUSDT",
                                exchange="BINANCE",
                                timeframe="1m",
                                data_dir="/tmp/test_data"
                            )
        
        # Verify result
        self.assertFalse(result["success"])
        self.assertIn("error", result)


class TestTripleBarrierMethodStep(unittest.TestCase):
    """Test cases for TripleBarrierMethodStep class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "SYMBOL": "ETHUSDT",
            "EXCHANGE": "BINANCE",
            "TIMEFRAME": "1m",
            "DATA_DIR": "/tmp/test_data",
            "PROFIT_TARGET": 0.02,
            "STOP_LOSS": 0.01,
            "MAX_HOLDING_PERIOD": 100
        }
        
        # Mock the logger
        with patch('src.training.steps.step4_triple_barrier_method.system_logger'):
            self.step = TripleBarrierMethodStep(self.config)
        
        # Create sample test data
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=2000, freq='1min'),
            'open': np.random.rand(2000) * 100 + 100,
            'high': np.random.rand(2000) * 100 + 110,
            'low': np.random.rand(2000) * 100 + 90,
            'close': np.random.rand(2000) * 100 + 100,
            'volume': np.random.rand(2000) * 10000
        })
        
    def tearDown(self):
        """Clean up after tests."""
        pass

    def test_initialization(self):
        """Test TripleBarrierMethodStep initialization."""
        self.assertIsNotNone(self.step)
        self.assertEqual(self.step.config, self.config)
        
    def test_initialize_components(self):
        """Test component initialization."""
        with patch('src.training.steps.step4_triple_barrier_method.OptimizedTripleBarrierLabeling') as mock_labeler:
            mock_labeler.return_value = Mock()
            self.step._initialize_components()
            # Should initialize without errors
            
    async def test_initialize(self):
        """Test step initialization."""
        await self.step.initialize()
        self.assertIsNotNone(self.step.start_time)
        
    @patch('pandas.read_parquet')
    async def test_load_data_success(self, mock_read_parquet):
        """Test successful data loading."""
        # Set up mock
        mock_read_parquet.return_value = self.test_data
        
        # Run the method
        result = await self.step._load_data("/tmp/test_data/test.parquet")
        
        # Verify result
        pd.testing.assert_frame_equal(result, self.test_data)
        
    @patch('pandas.read_parquet')
    async def test_load_data_failure(self, mock_read_parquet):
        """Test data loading failure."""
        # Set up mock to raise exception
        mock_read_parquet.side_effect = FileNotFoundError("File not found")
        
        # Run the method
        result = await self.step._load_data("/tmp/test_data/test.parquet")
        
        # Verify result
        self.assertIsNone(result)
        
    async def test_apply_triple_barrier_with_labeler(self):
        """Test applying triple barrier method with labeler."""
        # Mock the labeler
        self.step.triple_barrier_labeler = Mock()
        self.step.triple_barrier_labeler.label_data = AsyncMock(return_value=pd.DataFrame({
            'timestamp': self.test_data['timestamp'],
            'label': np.random.choice([-1, 0, 1], size=len(self.test_data)),
            'return': np.random.randn(len(self.test_data)) * 0.01
        }))
        
        # Run the method
        result = await self.step._apply_triple_barrier(
            self.test_data, 0.02, 0.01, 100
        )
        
        # Verify result
        self.assertIsNotNone(result)
        self.assertIn('label', result.columns)
        
    async def test_apply_triple_barrier_without_labeler(self):
        """Test applying triple barrier method without labeler."""
        # Remove the labeler
        self.step.triple_barrier_labeler = None
        
        # Run the method
        result = await self.step._apply_triple_barrier(
            self.test_data, 0.02, 0.01, 100
        )
        
        # Verify result (should use fallback implementation)
        self.assertIsNotNone(result)
        self.assertIn('label', result.columns)
        
    @patch('src.training.steps.step4_triple_barrier_method.TripleBarrierMethodStep._load_data')
    @patch('src.training.steps.step4_triple_barrier_method.TripleBarrierMethodStep._apply_triple_barrier')
    @patch('src.training.steps.step4_triple_barrier_method.TripleBarrierMethodStep._save_labeled_data')
    @patch('src.training.steps.step4_triple_barrier_method.log_step_metrics')
    @patch('src.training.steps.step4_triple_barrier_method.log_step_report')
    async def test_execute_success(self, mock_log_report, mock_log_metrics,
                                  mock_save, mock_apply, mock_load):
        """Test successful execution of triple barrier method."""
        # Set up mocks
        mock_load.return_value = self.test_data
        labeled_data = self.test_data.copy()
        labeled_data['label'] = np.random.choice([-1, 0, 1], size=len(self.test_data))
        mock_apply.return_value = labeled_data
        mock_save.return_value = True
        
        # Patch decorators
        with patch('src.training.steps.step4_triple_barrier_method.with_enhanced_mlflow_logging', lambda x: lambda fn: fn):
            with patch('src.training.steps.step4_triple_barrier_method.with_tracing_span', lambda x: lambda fn: fn):
                with patch('src.training.steps.step4_triple_barrier_method.handle_errors', lambda fn: fn):
                    with patch('src.training.steps.step4_triple_barrier_method.resource_monitor', lambda fn: fn):
                        # Run the method
                        result = await self.step.execute(
                            symbol="ETHUSDT",
                            exchange="BINANCE",
                            timeframe="1m",
                            data_dir="/tmp/test_data"
                        )
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertIn("labeled_data", result)
        self.assertIn("label_stats", result)
        
    def test_calculate_label_statistics(self):
        """Test label statistics calculation."""
        # Create labeled data
        labeled_data = self.test_data.copy()
        labeled_data['label'] = np.array([1] * 500 + [0] * 1000 + [-1] * 500)
        
        # Run the method
        stats = self.step._calculate_label_statistics(labeled_data)
        
        # Verify statistics
        self.assertIn("total_labels", stats)
        self.assertIn("buy_signals", stats)
        self.assertIn("sell_signals", stats)
        self.assertIn("no_action", stats)
        self.assertEqual(stats["total_labels"], 2000)
        self.assertEqual(stats["buy_signals"], 500)
        self.assertEqual(stats["sell_signals"], 500)
        self.assertEqual(stats["no_action"], 1000)


class TestRegimeTripleBarrierIntegration(unittest.TestCase):
    """Integration tests for regime splitting and triple barrier pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = Path("/tmp/test_regime_triple_barrier")
        self.test_data_dir.mkdir(exist_ok=True)
        
    def tearDown(self):
        """Clean up test data."""
        import shutil
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)
            
    async def test_regime_to_triple_barrier_pipeline(self):
        """Test integration between regime splitting and triple barrier."""
        # Create configuration
        config = {
            "SYMBOL": "ETHUSDT",
            "EXCHANGE": "BINANCE",
            "TIMEFRAME": "1m",
            "DATA_DIR": str(self.test_data_dir),
            "MIN_REGIME_DURATION": 100,
            "PROFIT_TARGET": 0.02,
            "STOP_LOSS": 0.01,
            "MAX_HOLDING_PERIOD": 100
        }
        
        # Initialize steps with mocked dependencies
        with patch('src.training.steps.step4_regime_data_splitting.system_logger'):
            with patch('src.training.steps.step4_triple_barrier_method.system_logger'):
                regime_step = RegimeDataSplittingStep(config)
                barrier_step = TripleBarrierMethodStep(config)
        
        # Create test data with regimes
        test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=1000, freq='1min'),
            'open': np.random.rand(1000) * 100 + 100,
            'high': np.random.rand(1000) * 100 + 110,
            'low': np.random.rand(1000) * 100 + 90,
            'close': np.random.rand(1000) * 100 + 100,
            'volume': np.random.rand(1000) * 10000,
            'regime': np.array([0] * 300 + [1] * 400 + [2] * 300)
        })
        
        # Test that both steps can be initialized
        await regime_step.initialize()
        await barrier_step.initialize()
        
        # Verify initialization
        self.assertIsNotNone(regime_step.start_time)
        self.assertIsNotNone(barrier_step.start_time)


if __name__ == "__main__":
    unittest.main()