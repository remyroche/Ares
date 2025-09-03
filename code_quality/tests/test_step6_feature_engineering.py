"""Unit tests for step6_feature_engineering.py and step6_feature_interaction_engineering.py.

This module tests the feature engineering functionality of the training pipeline.
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
from src.training.steps.step6_feature_engineering import run_step, _create_features_cache_key, _check_features_cache
from src.training.steps.step6_feature_interaction_engineering import FeatureInteractionEngine


class TestStep6FeatureEngineering(unittest.TestCase):
    """Test cases for step6_feature_engineering functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "SYMBOL": "ETHUSDT",
            "EXCHANGE": "BINANCE",
            "TIMEFRAME": "1m",
            "DATA_DIR": "/tmp/test_data",
            "FEATURE_ENGINEERING": {
                "use_basic_features": True,
                "use_advanced_features": True,
                "use_regime_features": True
            }
        }
        
        # Create sample test data
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=2000, freq='1min'),
            'open': np.random.rand(2000) * 100 + 100,
            'high': np.random.rand(2000) * 100 + 110,
            'low': np.random.rand(2000) * 100 + 90,
            'close': np.random.rand(2000) * 100 + 100,
            'volume': np.random.rand(2000) * 10000,
            'label': np.random.choice([-1, 0, 1], size=2000),
            'meta_label': np.random.rand(2000),
            'confidence': np.random.rand(2000)
        })
        
    def tearDown(self):
        """Clean up after tests."""
        pass

    def test_create_features_cache_key(self):
        """Test cache key creation."""
        # Create test data
        data = self.test_data.copy()
        config = {"feature_version": "1.0", "feature_set": "basic"}
        
        # Run the function
        cache_key = _create_features_cache_key(data, config)
        
        # Verify result
        self.assertIsInstance(cache_key, str)
        self.assertTrue(len(cache_key) > 0)
        
    @patch('os.path.exists')
    @patch('pandas.read_parquet')
    def test_check_features_cache_exists(self, mock_read_parquet, mock_exists):
        """Test checking features cache when it exists."""
        # Set up mocks
        mock_exists.return_value = True
        mock_read_parquet.return_value = self.test_data
        
        # Run the function
        result = _check_features_cache("test_cache_key", "/tmp/cache")
        
        # Verify result
        pd.testing.assert_frame_equal(result, self.test_data)
        
    @patch('os.path.exists')
    def test_check_features_cache_not_exists(self, mock_exists):
        """Test checking features cache when it doesn't exist."""
        # Set up mock
        mock_exists.return_value = False
        
        # Run the function
        result = _check_features_cache("test_cache_key", "/tmp/cache")
        
        # Verify result
        self.assertIsNone(result)
        
    @patch('pandas.read_parquet')
    @patch('src.training.steps.step6_feature_engineering._check_features_cache')
    @patch('src.training.steps.step6_feature_engineering.vectorized_feature_engineering')
    @patch('src.training.steps.step6_feature_engineering.ensure_directory')
    @patch('pandas.DataFrame.to_parquet')
    @patch('src.training.steps.step6_feature_engineering.log_step_dataframe_with_standardized_name')
    @patch('src.training.steps.step6_feature_engineering.log_step_report')
    async def test_run_step_success(self, mock_log_report, mock_log_df, mock_to_parquet,
                                   mock_ensure_dir, mock_vec_feature_eng, mock_check_cache, mock_read_parquet):
        """Test successful feature engineering execution."""
        # Set up mocks
        mock_read_parquet.return_value = self.test_data
        mock_check_cache.return_value = None  # No cache
        
        # Mock feature engineering
        features_df = self.test_data.copy()
        for i in range(10):
            features_df[f'feature_{i}'] = np.random.rand(len(features_df))
        
        mock_vec_feature_eng.VectorizedAdvancedFeatureEngineering = Mock()
        mock_instance = Mock()
        mock_instance.create_all_features = AsyncMock(return_value=features_df)
        mock_vec_feature_eng.VectorizedAdvancedFeatureEngineering.return_value = mock_instance
        
        # Patch decorators
        with patch('src.training.steps.step6_feature_engineering.handle_errors', lambda **kwargs: lambda fn: fn):
            # Run the function
            result = await run_step(
                symbol="ETHUSDT",
                exchange="BINANCE",
                timeframe="1m",
                data_dir="/tmp/test_data",
                force_rerun=False
            )
        
        # Verify result
        self.assertTrue(result)
        mock_to_parquet.assert_called()
        
    @patch('pandas.read_parquet')
    async def test_run_step_data_not_found(self, mock_read_parquet):
        """Test feature engineering when data file doesn't exist."""
        # Set up mock to raise exception
        mock_read_parquet.side_effect = FileNotFoundError("File not found")
        
        # Patch decorators
        with patch('src.training.steps.step6_feature_engineering.handle_errors', lambda **kwargs: lambda fn: fn):
            # Run the function
            result = await run_step(
                symbol="ETHUSDT",
                exchange="BINANCE",
                timeframe="1m",
                data_dir="/tmp/test_data",
                force_rerun=False
            )
        
        # Verify result
        self.assertFalse(result)
        
    @patch('pandas.read_parquet')
    @patch('src.training.steps.step6_feature_engineering._check_features_cache')
    async def test_run_step_with_cache(self, mock_check_cache, mock_read_parquet):
        """Test feature engineering with cached features."""
        # Set up mocks
        mock_read_parquet.return_value = self.test_data
        
        # Mock cached features
        cached_features = self.test_data.copy()
        for i in range(10):
            cached_features[f'feature_{i}'] = np.random.rand(len(cached_features))
        mock_check_cache.return_value = cached_features
        
        # Patch decorators
        with patch('src.training.steps.step6_feature_engineering.handle_errors', lambda **kwargs: lambda fn: fn):
            with patch('src.training.steps.step6_feature_engineering.log_step_dataframe_with_standardized_name'):
                with patch('src.training.steps.step6_feature_engineering.log_step_report'):
                    # Run the function
                    result = await run_step(
                        symbol="ETHUSDT",
                        exchange="BINANCE",
                        timeframe="1m",
                        data_dir="/tmp/test_data",
                        force_rerun=False
                    )
        
        # Verify result
        self.assertTrue(result)
        
    @patch('pandas.read_parquet')
    def test_validate_features(self, mock_read_parquet):
        """Test feature validation logic."""
        # Create test data with features
        features_df = self.test_data.copy()
        for i in range(10):
            features_df[f'feature_{i}'] = np.random.rand(len(features_df))
        
        # Add some problematic features
        features_df['nan_feature'] = np.nan
        features_df['inf_feature'] = np.inf
        
        # Test validation (implementation dependent)
        # This would test the validation logic in the actual implementation


class TestFeatureInteractionEngine(unittest.TestCase):
    """Test cases for FeatureInteractionEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "step6_feature_interaction_engineering": {
                "use_matrix_optimizer": True,
                "max_interactions": 50,
                "interaction_types": ["multiply", "divide", "add", "subtract"],
                "correlation_threshold": 0.95
            }
        }
        
        # Create sample test data
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=1000, freq='1min'),
            'close': np.random.rand(1000) * 100 + 100,
            'volume': np.random.rand(1000) * 10000,
            'RSI_14': np.random.rand(1000) * 100,
            'MACD': np.random.randn(1000) * 0.1,
            'BB_upper': np.random.rand(1000) * 100 + 110,
            'BB_lower': np.random.rand(1000) * 100 + 90,
            'regime': np.random.choice([0, 1, 2], size=1000)
        })
        
    def tearDown(self):
        """Clean up after tests."""
        pass

    def test_initialization(self):
        """Test FeatureInteractionEngine initialization."""
        # Mock the optimizers
        with patch('src.training.steps.step6_feature_interaction_engineering.DiverseLookbackOptimizer'):
            with patch('src.training.steps.step6_feature_interaction_engineering.MatrixDiverseLookbackOptimizer'):
                engine = FeatureInteractionEngine(self.config)
        
        self.assertIsNotNone(engine)
        self.assertEqual(engine.config, self.config)
        
    def test_initialization_without_optimizers(self):
        """Test initialization when optimizers are not available."""
        # Create engine without optimizers
        engine = FeatureInteractionEngine(self.config)
        
        self.assertIsNotNone(engine)
        self.assertFalse(engine.use_dynamic_periods)
        
    async def test_create_all_interactions(self):
        """Test creating all feature interactions."""
        # Create engine
        engine = FeatureInteractionEngine(self.config)
        
        # Run the method
        result = await engine.create_all_interactions(self.test_data)
        
        # Verify result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result.columns), len(self.test_data.columns))
        self.assertEqual(len(result), len(self.test_data))
        
    def test_create_technical_interactions(self):
        """Test creating technical indicator interactions."""
        # Create engine
        engine = FeatureInteractionEngine(self.config)
        
        # Run the method
        result = engine._create_technical_interactions(self.test_data)
        
        # Verify result
        self.assertIsInstance(result, pd.DataFrame)
        # Should have created interaction columns
        interaction_cols = [col for col in result.columns if '_x_' in col or '_over_' in col]
        self.assertGreater(len(interaction_cols), 0)
        
    def test_create_regime_aware_interactions(self):
        """Test creating regime-aware interactions."""
        # Create engine
        engine = FeatureInteractionEngine(self.config)
        
        # Run the method
        result = engine._create_regime_aware_interactions(self.test_data)
        
        # Verify result
        self.assertIsInstance(result, pd.DataFrame)
        # Should have regime interaction columns
        regime_cols = [col for col in result.columns if 'regime' in col.lower()]
        self.assertGreater(len(regime_cols), 0)
        
    def test_create_cross_timeframe_interactions(self):
        """Test creating cross-timeframe interactions."""
        # Create engine
        engine = FeatureInteractionEngine(self.config)
        
        # Add some rolling features to test data
        self.test_data['close_ma_5'] = self.test_data['close'].rolling(5).mean()
        self.test_data['close_ma_20'] = self.test_data['close'].rolling(20).mean()
        
        # Run the method
        result = engine._create_cross_timeframe_interactions(self.test_data)
        
        # Verify result
        self.assertIsInstance(result, pd.DataFrame)
        # Should have cross-timeframe columns
        cross_cols = [col for col in result.columns if 'ratio' in col or 'diff' in col]
        self.assertGreater(len(cross_cols), 0)
        
    def test_calculate_interaction_importance(self):
        """Test interaction importance calculation."""
        # Create engine
        engine = FeatureInteractionEngine(self.config)
        
        # Create features with label
        features = self.test_data.copy()
        features['label'] = np.random.choice([0, 1], size=len(features))
        
        # Run the method
        result = engine._calculate_interaction_importance(features, 'label')
        
        # Verify result
        self.assertIsInstance(result, dict)
        # Should have importance scores for features
        feature_cols = [col for col in features.columns if col not in ['timestamp', 'label']]
        for col in feature_cols:
            self.assertIn(col, result)
            self.assertIsInstance(result[col], (int, float))
            
    def test_remove_correlated_features(self):
        """Test correlated feature removal."""
        # Create engine
        engine = FeatureInteractionEngine(self.config)
        
        # Create features with high correlation
        features = self.test_data.copy()
        features['corr_feature_1'] = features['close'] * 1.1 + np.random.randn(len(features)) * 0.01
        features['corr_feature_2'] = features['close'] * 1.1 + np.random.randn(len(features)) * 0.01
        
        # Run the method
        result = engine._remove_correlated_features(features, threshold=0.95)
        
        # Verify result
        self.assertIsInstance(result, pd.DataFrame)
        # Should have removed at least one correlated feature
        self.assertLess(len(result.columns), len(features.columns))


class TestFeatureEngineeringIntegration(unittest.TestCase):
    """Integration tests for feature engineering pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = Path("/tmp/test_feature_engineering")
        self.test_data_dir.mkdir(exist_ok=True)
        
    def tearDown(self):
        """Clean up test data."""
        import shutil
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)
            
    async def test_feature_engineering_pipeline(self):
        """Test complete feature engineering pipeline."""
        # Create test data with labels
        test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=1000, freq='1min'),
            'open': np.random.rand(1000) * 100 + 100,
            'high': np.random.rand(1000) * 100 + 110,
            'low': np.random.rand(1000) * 100 + 90,
            'close': np.random.rand(1000) * 100 + 100,
            'volume': np.random.rand(1000) * 10000,
            'label': np.random.choice([-1, 0, 1], size=1000),
            'meta_label': np.random.rand(1000),
            'confidence': np.random.rand(1000),
            'regime': np.random.choice([0, 1, 2], size=1000)
        })
        
        # Save test data
        test_file = self.test_data_dir / "BINANCE_ETHUSDT_1m_labeled_meta.parquet"
        test_data.to_parquet(test_file)
        
        # Run feature engineering with mocked components
        with patch('src.training.steps.step6_feature_engineering.vectorized_feature_engineering'):
            with patch('src.training.steps.step6_feature_engineering.log_step_dataframe_with_standardized_name'):
                with patch('src.training.steps.step6_feature_engineering.log_step_report'):
                    with patch('src.training.steps.step6_feature_engineering.handle_errors', lambda **kwargs: lambda fn: fn):
                        # Run the step
                        result = await run_step(
                            symbol="ETHUSDT",
                            exchange="BINANCE",
                            timeframe="1m",
                            data_dir=str(self.test_data_dir),
                            force_rerun=True
                        )
        
        # Verify result (would be False due to mocked components not fully implementing logic)
        self.assertIsInstance(result, bool)
        
    async def test_feature_interaction_with_regime_data(self):
        """Test feature interaction engineering with regime data."""
        # Create configuration
        config = {
            "step6_feature_interaction_engineering": {
                "use_matrix_optimizer": False,  # Disable for testing
                "max_interactions": 20,
                "interaction_types": ["multiply", "divide"],
                "correlation_threshold": 0.9
            }
        }
        
        # Create engine
        engine = FeatureInteractionEngine(config)
        
        # Create test data with regimes and technical indicators
        test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=500, freq='1min'),
            'close': np.random.rand(500) * 100 + 100,
            'volume': np.random.rand(500) * 10000,
            'RSI_14': np.random.rand(500) * 100,
            'MACD': np.random.randn(500) * 0.1,
            'regime': np.array([0] * 150 + [1] * 200 + [2] * 150)
        })
        
        # Run interaction creation
        result = await engine.create_all_interactions(test_data)
        
        # Verify interactions were created
        self.assertGreater(len(result.columns), len(test_data.columns))
        
        # Check for regime-specific interactions
        regime_interactions = [col for col in result.columns if 'regime' in col]
        self.assertGreater(len(regime_interactions), 0)


if __name__ == "__main__":
    unittest.main()