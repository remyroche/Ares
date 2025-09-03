"""Unit tests for step5_labeling.py.

This module tests the labeling functionality of the training pipeline.
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
from src.training.steps.step5_labeling import LabelingStep


class TestLabelingStep(unittest.TestCase):
    """Test cases for LabelingStep class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "SYMBOL": "ETHUSDT",
            "EXCHANGE": "BINANCE",
            "TIMEFRAME": "1m",
            "DATA_DIR": "/tmp/test_data",
            "USE_META_LABELING": True,
            "LABELING_CONFIG": {
                "profit_target": 0.02,
                "stop_loss": 0.01,
                "max_holding_period": 100,
                "min_sample_weight": 0.1
            }
        }
        
        # Mock the logger
        with patch('src.training.steps.step5_labeling.system_logger'):
            self.step = LabelingStep(self.config)
        
        # Create sample test data
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=2000, freq='1min'),
            'open': np.random.rand(2000) * 100 + 100,
            'high': np.random.rand(2000) * 100 + 110,
            'low': np.random.rand(2000) * 100 + 90,
            'close': np.random.rand(2000) * 100 + 100,
            'volume': np.random.rand(2000) * 10000,
            'label': np.random.choice([-1, 0, 1], size=2000)
        })
        
    def tearDown(self):
        """Clean up after tests."""
        pass

    def test_initialization(self):
        """Test LabelingStep initialization."""
        self.assertIsNotNone(self.step)
        self.assertEqual(self.step.config, self.config)
        
    def test_validate_environment(self):
        """Test environment validation."""
        with patch('src.training.steps.step5_labeling.dependency_status', 
                  {'pandas': True, 'numpy': True, 'psutil': True}):
            # Should not raise any exceptions
            self.step._validate_environment()
            
    def test_initialize_components(self):
        """Test component initialization."""
        with patch('src.training.steps.step5_labeling.meta_labeling_system') as mock_meta:
            mock_meta.MetaLabelingSystem = Mock(return_value=Mock())
            self.step._initialize_components()
            # Should initialize without errors
            
    async def test_initialize(self):
        """Test step initialization."""
        await self.step.initialize()
        self.assertIsNotNone(self.step.start_time)
        
    @patch('pandas.read_parquet')
    async def test_load_data_with_labels_success(self, mock_read_parquet):
        """Test successful data loading with labels."""
        # Set up mock
        mock_read_parquet.return_value = self.test_data
        
        # Run the method
        result = await self.step._load_data_with_labels(
            "ETHUSDT", "BINANCE", "1m", "/tmp/test_data"
        )
        
        # Verify result
        self.assertIsNotNone(result)
        pd.testing.assert_frame_equal(result, self.test_data)
        
    @patch('pandas.read_parquet')
    async def test_load_data_with_labels_file_not_found(self, mock_read_parquet):
        """Test data loading when file doesn't exist."""
        # Set up mock to raise exception
        mock_read_parquet.side_effect = FileNotFoundError("File not found")
        
        # Run the method
        result = await self.step._load_data_with_labels(
            "ETHUSDT", "BINANCE", "1m", "/tmp/test_data"
        )
        
        # Verify result
        self.assertIsNone(result)
        
    async def test_create_meta_labels_with_system(self):
        """Test meta label creation with meta labeling system."""
        # Mock the meta labeling system
        self.step.meta_labeling_system = Mock()
        self.step.meta_labeling_system.generate_meta_labels = AsyncMock(return_value={
            "meta_labels": np.random.rand(len(self.test_data)),
            "confidence_scores": np.random.rand(len(self.test_data)),
            "feature_importance": {"volume": 0.3, "volatility": 0.5}
        })
        
        # Run the method
        result = await self.step._create_meta_labels(self.test_data)
        
        # Verify result
        self.assertIsNotNone(result)
        self.assertIn("meta_label", result.columns)
        self.assertIn("confidence", result.columns)
        
    async def test_create_meta_labels_without_system(self):
        """Test meta label creation without meta labeling system."""
        # Remove the meta labeling system
        self.step.meta_labeling_system = None
        
        # Run the method
        result = await self.step._create_meta_labels(self.test_data)
        
        # Verify result (should use fallback)
        self.assertIsNotNone(result)
        self.assertIn("meta_label", result.columns)
        self.assertIn("confidence", result.columns)
        
    def test_calculate_label_statistics(self):
        """Test label statistics calculation."""
        # Create test data with labels
        labeled_data = self.test_data.copy()
        labeled_data['meta_label'] = np.random.rand(len(labeled_data))
        labeled_data['confidence'] = np.random.rand(len(labeled_data))
        
        # Run the method
        stats = self.step._calculate_label_statistics(labeled_data)
        
        # Verify statistics
        self.assertIn("total_samples", stats)
        self.assertIn("buy_signals", stats)
        self.assertIn("sell_signals", stats)
        self.assertIn("no_action", stats)
        self.assertIn("avg_confidence", stats)
        self.assertIn("label_distribution", stats)
        
    @patch('src.training.steps.step5_labeling.ensure_directory')
    @patch('pandas.DataFrame.to_parquet')
    async def test_save_labeled_data(self, mock_to_parquet, mock_ensure_dir):
        """Test saving labeled data."""
        # Run the method
        result = await self.step._save_labeled_data(
            self.test_data, "/tmp/test_data", "ETHUSDT", "BINANCE", "1m"
        )
        
        # Verify calls and result
        mock_ensure_dir.assert_called()
        mock_to_parquet.assert_called()
        self.assertIsNotNone(result)
        self.assertIn("/tmp/test_data", result)
        
    @patch('src.training.steps.step5_labeling.LabelingStep._load_data_with_labels')
    @patch('src.training.steps.step5_labeling.LabelingStep._create_meta_labels')
    @patch('src.training.steps.step5_labeling.LabelingStep._save_labeled_data')
    @patch('src.training.steps.step5_labeling.log_step_metrics')
    @patch('src.training.steps.step5_labeling.log_step_report')
    @patch('src.training.steps.step5_labeling.log_step_dataframe_with_standardized_name')
    async def test_execute_labeling_success(self, mock_log_df, mock_log_report, mock_log_metrics,
                                          mock_save, mock_create_meta, mock_load):
        """Test successful labeling execution."""
        # Set up mocks
        mock_load.return_value = self.test_data
        labeled_data = self.test_data.copy()
        labeled_data['meta_label'] = np.random.rand(len(labeled_data))
        labeled_data['confidence'] = np.random.rand(len(labeled_data))
        mock_create_meta.return_value = labeled_data
        mock_save.return_value = "/tmp/test_data/labeled_data.parquet"
        
        # Patch decorators
        with patch('src.training.steps.step5_labeling.handle_errors', lambda fn: fn):
            with patch('src.training.steps.step5_labeling.memory_efficient', lambda fn: fn):
                with patch('src.training.steps.step5_labeling.resource_monitor', lambda fn: fn):
                    with patch('src.training.steps.step5_labeling.secure_data_processing', lambda fn: fn):
                        with patch('src.training.steps.step5_labeling.validate_data_structure', lambda fn: fn):
                            # Run the method
                            result = await self.step.execute_labeling(
                                symbol="ETHUSDT",
                                exchange="BINANCE",
                                timeframe="1m",
                                data_dir="/tmp/test_data",
                                force_rerun=False
                            )
        
        # Verify result
        self.assertTrue(result)
        
    @patch('src.training.steps.step5_labeling.LabelingStep._load_data_with_labels')
    async def test_execute_labeling_load_failure(self, mock_load):
        """Test labeling execution when data loading fails."""
        # Set up mock
        mock_load.return_value = None
        
        # Patch decorators
        with patch('src.training.steps.step5_labeling.handle_errors', lambda fn: fn):
            with patch('src.training.steps.step5_labeling.memory_efficient', lambda fn: fn):
                with patch('src.training.steps.step5_labeling.resource_monitor', lambda fn: fn):
                    with patch('src.training.steps.step5_labeling.secure_data_processing', lambda fn: fn):
                        with patch('src.training.steps.step5_labeling.validate_data_structure', lambda fn: fn):
                            # Run the method
                            result = await self.step.execute_labeling(
                                symbol="ETHUSDT",
                                exchange="BINANCE",
                                timeframe="1m",
                                data_dir="/tmp/test_data",
                                force_rerun=False
                            )
        
        # Verify result
        self.assertFalse(result)
        
    def test_validate_labels(self):
        """Test label validation."""
        # Create test data with valid labels
        valid_data = self.test_data.copy()
        valid_data['meta_label'] = np.random.rand(len(valid_data))
        valid_data['confidence'] = np.random.rand(len(valid_data))
        
        # Test valid labels
        result = self.step._validate_labels(valid_data)
        self.assertTrue(result)
        
        # Create test data with missing labels
        invalid_data = self.test_data.copy()
        
        # Test invalid labels
        result = self.step._validate_labels(invalid_data)
        self.assertFalse(result)
        
    async def test_generate_dynamic_labels(self):
        """Test dynamic label generation."""
        # This tests the new dynamic labeling feature
        self.step.config["ENABLE_DYNAMIC_LABELING"] = True
        
        # Mock the dynamic labeling components
        with patch.object(self.step, '_apply_regime_aware_triple_barrier') as mock_barrier:
            mock_barrier.return_value = self.test_data.copy()
            
            # Run method (if implemented)
            # result = await self.step._generate_dynamic_labels(self.test_data)
            # self.assertIsNotNone(result)


class TestLabelingHelpers(unittest.TestCase):
    """Test helper functions in labeling step."""
    
    def test_meta_label_fallback(self):
        """Test meta label fallback implementation."""
        # Create test data
        data = pd.DataFrame({
            'close': np.random.rand(100) * 100,
            'label': np.random.choice([-1, 0, 1], size=100)
        })
        
        # Test fallback meta labeling (method implementation dependent)
        # This would test the fallback logic when meta_labeling_system is not available
        
    def test_confidence_score_calculation(self):
        """Test confidence score calculation."""
        # Create test data with various scenarios
        data = pd.DataFrame({
            'close': np.random.rand(100) * 100,
            'volume': np.random.rand(100) * 10000,
            'label': np.random.choice([-1, 0, 1], size=100)
        })
        
        # Test confidence calculation logic


class TestLabelingIntegration(unittest.TestCase):
    """Integration tests for labeling pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = Path("/tmp/test_labeling")
        self.test_data_dir.mkdir(exist_ok=True)
        
    def tearDown(self):
        """Clean up test data."""
        import shutil
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)
            
    async def test_labeling_with_regime_data(self):
        """Test labeling integration with regime data."""
        # Create configuration
        config = {
            "SYMBOL": "ETHUSDT",
            "EXCHANGE": "BINANCE",
            "TIMEFRAME": "1m",
            "DATA_DIR": str(self.test_data_dir),
            "USE_META_LABELING": True,
            "LABELING_CONFIG": {
                "profit_target": 0.02,
                "stop_loss": 0.01,
                "max_holding_period": 100
            }
        }
        
        # Initialize step with mocked dependencies
        with patch('src.training.steps.step5_labeling.system_logger'):
            step = LabelingStep(config)
        
        # Create test data with regime labels
        test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=1000, freq='1min'),
            'open': np.random.rand(1000) * 100 + 100,
            'high': np.random.rand(1000) * 100 + 110,
            'low': np.random.rand(1000) * 100 + 90,
            'close': np.random.rand(1000) * 100 + 100,
            'volume': np.random.rand(1000) * 10000,
            'regime': np.array([0] * 300 + [1] * 400 + [2] * 300),
            'label': np.random.choice([-1, 0, 1], size=1000)
        })
        
        # Save test data
        test_file = self.test_data_dir / "BINANCE_ETHUSDT_1m_labeled.parquet"
        test_data.to_parquet(test_file)
        
        # Test that labeling can process regime data
        with patch('src.training.steps.step5_labeling.pandas.read_parquet', return_value=test_data):
            labeled_data = await step._load_data_with_labels(
                "ETHUSDT", "BINANCE", "1m", str(self.test_data_dir)
            )
        
        self.assertIsNotNone(labeled_data)
        self.assertIn('regime', labeled_data.columns)
        self.assertIn('label', labeled_data.columns)


if __name__ == "__main__":
    unittest.main()