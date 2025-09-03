"""Unit tests for step7_enhanced_matrix_operations.py.

This module tests the enhanced matrix operations functionality of the training pipeline.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the module to be tested
from src.training.steps.step7_enhanced_matrix_operations import (
    MatrixOperationsStep,
    run_step,
)


class TestMatrixOperationsStep(unittest.TestCase):
    """Test cases for MatrixOperationsStep class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "SYMBOL": "ETHUSDT",
            "EXCHANGE": "BINANCE",
            "TIMEFRAME": "1m",
            "DATA_DIR": "/tmp/test_data",
            "MATRIX_OPERATIONS": {
                "pca_components": 0.95,
                "correlation_threshold": 0.95,
                "use_gpu": False,
                "batch_size": 1000,
            },
        }

        # Mock the logger
        with patch("src.training.steps.step7_enhanced_matrix_operations.system_logger"):
            self.step = MatrixOperationsStep(self.config)

        # Create sample test data with features
        self.test_data = pd.DataFrame({
            "timestamp": pd.date_range(start="2024-01-01", periods=1000, freq="1min"),
            "close": np.random.rand(1000) * 100 + 100,
            "volume": np.random.rand(1000) * 10000,
            "label": np.random.choice([-1, 0, 1], size=1000),
        })

        # Add features
        for i in range(20):
            self.test_data[f"feature_{i}"] = np.random.randn(1000)

    def tearDown(self):
        """Clean up after tests."""

    def test_initialization(self):
        """Test MatrixOperationsStep initialization."""
        assert self.step is not None
        assert self.step.config == self.config

    def test_validate_environment(self):
        """Test environment validation."""
        with patch("src.training.steps.step7_enhanced_matrix_operations.dependency_status",
                  {"pandas": True, "numpy": True}):
            # Should not raise any exceptions
            self.step._validate_environment()

    def test_initialize_components(self):
        """Test component initialization."""
        with patch("src.training.steps.step7_enhanced_matrix_operations.enhanced_matrix_operations") as mock_matrix:
            mock_matrix.MatrixOperationsManager = Mock(return_value=Mock())
            with patch("src.training.steps.step7_enhanced_matrix_operations.feature_engineering_optimizer") as mock_optimizer:
                mock_optimizer.FeatureEngineeringOptimizer = Mock(return_value=Mock())
                with patch("src.training.steps.step7_enhanced_matrix_operations.timeframe_relevance_analyzer") as mock_analyzer:
                    mock_analyzer.TimeframeRelevanceAnalyzer = Mock(return_value=Mock())
                    self.step._initialize_components()

        # Should initialize without errors

    async def test_initialize(self):
        """Test step initialization."""
        await self.step.initialize()
        # Should complete without errors

    @patch("pandas.read_parquet")
    async def test_load_features_success(self, mock_read_parquet):
        """Test successful feature loading."""
        # Set up mock
        mock_read_parquet.return_value = self.test_data

        # Run the method
        result = await self.step._load_features(
            "ETHUSDT", "BINANCE", "1m", "/tmp/test_data",
        )

        # Verify result
        assert result is not None
        pd.testing.assert_frame_equal(result, self.test_data)

    @patch("pandas.read_parquet")
    async def test_load_features_file_not_found(self, mock_read_parquet):
        """Test feature loading when file doesn't exist."""
        # Set up mock to raise exception
        mock_read_parquet.side_effect = FileNotFoundError("File not found")

        # Run the method
        result = await self.step._load_features(
            "ETHUSDT", "BINANCE", "1m", "/tmp/test_data",
        )

        # Verify result
        assert result is None

    async def test_perform_matrix_operations_with_manager(self):
        """Test matrix operations with matrix manager."""
        # Mock the matrix manager
        self.step.matrix_manager = Mock()
        mock_results = {
            "correlation_matrix": np.random.rand(10, 10),
            "pca_components": np.random.randn(10, 5),
            "feature_importance": np.random.rand(10),
            "covariance_matrix": np.random.rand(10, 10),
        }
        self.step.matrix_manager.perform_all_operations = AsyncMock(return_value=mock_results)

        # Run the method
        result = await self.step._perform_matrix_operations(self.test_data)

        # Verify result
        assert isinstance(result, dict)
        assert "correlation_matrix" in result
        assert "pca_components" in result

    async def test_perform_matrix_operations_without_manager(self):
        """Test matrix operations without matrix manager."""
        # Remove the matrix manager
        self.step.matrix_manager = None

        # Run the method
        result = await self.step._perform_matrix_operations(self.test_data)

        # Verify result (should use fallback)
        assert isinstance(result, dict)
        assert "correlation_matrix" in result

    async def test_analyze_feature_relationships(self):
        """Test feature relationship analysis."""
        # Create matrix results
        matrix_results = {
            "correlation_matrix": np.random.rand(10, 10),
            "feature_importance": np.random.rand(10),
        }

        # Run the method
        result = await self.step._analyze_feature_relationships(
            self.test_data, matrix_results,
        )

        # Verify result
        assert isinstance(result, dict)
        assert "highly_correlated_pairs" in result
        assert "feature_clusters" in result

    def test_calculate_feature_statistics(self):
        """Test feature statistics calculation."""
        # Run the method
        stats = self.step._calculate_feature_statistics(self.test_data)

        # Verify statistics
        assert isinstance(stats, dict)
        assert "n_features" in stats
        assert "n_samples" in stats
        assert "missing_values" in stats
        assert "feature_types" in stats

    @patch("src.training.steps.step7_enhanced_matrix_operations.ensure_directory")
    @patch("src.training.steps.step7_enhanced_matrix_operations.safe_json_dump")
    @patch("numpy.save")
    async def test_save_matrix_results(self, mock_np_save, mock_json_dump, mock_ensure_dir):
        """Test saving matrix results."""
        # Create test results
        matrix_results = {
            "correlation_matrix": np.random.rand(10, 10),
            "pca_components": np.random.randn(10, 5),
            "feature_importance": np.random.rand(10),
            "analysis": {"test": "data"},
        }

        # Run the method
        result = await self.step._save_matrix_results(
            matrix_results, "/tmp/test_data", "ETHUSDT", "BINANCE", "1m",
        )

        # Verify calls and result
        mock_ensure_dir.assert_called()
        mock_np_save.assert_called()
        mock_json_dump.assert_called()
        assert isinstance(result, dict)
        assert "correlation_matrix" in result

    @patch("src.training.steps.step7_enhanced_matrix_operations.MatrixOperationsStep._load_features")
    @patch("src.training.steps.step7_enhanced_matrix_operations.MatrixOperationsStep._perform_matrix_operations")
    @patch("src.training.steps.step7_enhanced_matrix_operations.MatrixOperationsStep._analyze_feature_relationships")
    @patch("src.training.steps.step7_enhanced_matrix_operations.MatrixOperationsStep._save_matrix_results")
    @patch("src.training.steps.step7_enhanced_matrix_operations.log_step_metrics")
    @patch("src.training.steps.step7_enhanced_matrix_operations.log_step_report")
    async def test_execute_success(self, mock_log_report, mock_log_metrics,
                                  mock_save, mock_analyze, mock_perform, mock_load):
        """Test successful matrix operations execution."""
        # Set up mocks
        mock_load.return_value = self.test_data
        mock_perform.return_value = {
            "correlation_matrix": np.random.rand(10, 10),
            "pca_components": np.random.randn(10, 5),
        }
        mock_analyze.return_value = {
            "highly_correlated_pairs": [],
            "feature_clusters": {},
        }
        mock_save.return_value = {
            "correlation_matrix": "/tmp/test_data/correlation.npy",
            "pca_components": "/tmp/test_data/pca.npy",
        }

        # Run the method
        result = await self.step.execute(
            symbol="ETHUSDT",
            exchange="BINANCE",
            timeframe="1m",
            data_dir="/tmp/test_data",
        )

        # Verify result
        assert result["success"]
        assert "matrix_results" in result
        assert "analysis_results" in result

    @patch("src.training.steps.step7_enhanced_matrix_operations.MatrixOperationsStep._load_features")
    async def test_execute_load_failure(self, mock_load):
        """Test execution when feature loading fails."""
        # Set up mock
        mock_load.return_value = None

        # Run the method
        result = await self.step.execute(
            symbol="ETHUSDT",
            exchange="BINANCE",
            timeframe="1m",
            data_dir="/tmp/test_data",
        )

        # Verify result
        assert not result["success"]
        assert "error" in result


class TestRunStep(unittest.TestCase):
    """Test cases for run_step function."""

    @patch("src.training.steps.step7_enhanced_matrix_operations.MatrixOperationsStep")
    async def test_run_step_success(self, mock_step_class):
        """Test successful run_step execution."""
        # Create mock step instance
        mock_step = Mock()
        mock_step.initialize = AsyncMock()
        mock_step.execute = AsyncMock(return_value={"success": True})
        mock_step_class.return_value = mock_step

        # Run the function
        result = await run_step(
            symbol="ETHUSDT",
            exchange="BINANCE",
            timeframe="1m",
            data_dir="/tmp/test_data",
            force_rerun=False,
        )

        # Verify result
        assert result

    @patch("src.training.steps.step7_enhanced_matrix_operations.MatrixOperationsStep")
    async def test_run_step_failure(self, mock_step_class):
        """Test run_step execution failure."""
        # Create mock step instance
        mock_step = Mock()
        mock_step.initialize = AsyncMock()
        mock_step.execute = AsyncMock(return_value={"success": False})
        mock_step_class.return_value = mock_step

        # Run the function
        result = await run_step(
            symbol="ETHUSDT",
            exchange="BINANCE",
            timeframe="1m",
            data_dir="/tmp/test_data",
            force_rerun=False,
        )

        # Verify result
        assert not result


class TestMatrixOperationsHelpers(unittest.TestCase):
    """Test helper functions in matrix operations step."""

    def test_fallback_correlation_calculation(self):
        """Test fallback correlation matrix calculation."""
        # Create test data
        data = pd.DataFrame({
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "feature_3": np.random.randn(100),
        })

        # Calculate correlation (fallback method)
        corr_matrix = data.corr().values

        # Verify result
        assert corr_matrix.shape == (3, 3)
        assert np.allclose(np.diag(corr_matrix), 1.0)

    def test_fallback_pca_calculation(self):
        """Test fallback PCA calculation."""
        # Create test data
        data = np.random.randn(100, 10)

        # Perform simple PCA (fallback method)
        # Center the data
        centered = data - np.mean(data, axis=0)

        # Compute covariance
        cov = np.cov(centered.T)

        # Verify covariance shape
        assert cov.shape == (10, 10)


class TestMatrixOperationsIntegration(unittest.TestCase):
    """Integration tests for matrix operations pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = Path("/tmp/test_matrix_operations")
        self.test_data_dir.mkdir(exist_ok=True)

    def tearDown(self):
        """Clean up test data."""
        import shutil
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)

    async def test_matrix_operations_pipeline(self):
        """Test complete matrix operations pipeline."""
        # Create test data with many features
        test_data = pd.DataFrame({
            "timestamp": pd.date_range(start="2024-01-01", periods=500, freq="1min"),
            "close": np.random.rand(500) * 100 + 100,
            "volume": np.random.rand(500) * 10000,
            "label": np.random.choice([-1, 0, 1], size=500),
        })

        # Add many features
        for i in range(50):
            test_data[f"feature_{i}"] = np.random.randn(500)

        # Save test data
        test_file = self.test_data_dir / "BINANCE_ETHUSDT_1m_features.parquet"
        test_data.to_parquet(test_file)

        # Create configuration
        config = {
            "SYMBOL": "ETHUSDT",
            "EXCHANGE": "BINANCE",
            "TIMEFRAME": "1m",
            "DATA_DIR": str(self.test_data_dir),
            "MATRIX_OPERATIONS": {
                "pca_components": 0.95,
                "correlation_threshold": 0.9,
            },
        }

        # Initialize step with mocked dependencies
        with patch("src.training.steps.step7_enhanced_matrix_operations.system_logger"):
            step = MatrixOperationsStep(config)

        # Run matrix operations
        features = await step._load_features(
            "ETHUSDT", "BINANCE", "1m", str(self.test_data_dir),
        )

        assert features is not None
        assert len(features) == 500

        # Test matrix calculations
        matrix_results = await step._perform_matrix_operations(features)

        assert isinstance(matrix_results, dict)
        assert "correlation_matrix" in matrix_results

    async def test_feature_importance_analysis(self):
        """Test feature importance analysis in matrix operations."""
        # Create test data with known relationships
        n_samples = 1000

        # Create correlated features
        base_feature = np.random.randn(n_samples)

        test_data = pd.DataFrame({
            "timestamp": pd.date_range(start="2024-01-01", periods=n_samples, freq="1min"),
            "feature_1": base_feature,
            "feature_2": base_feature * 0.9 + np.random.randn(n_samples) * 0.1,  # Highly correlated
            "feature_3": np.random.randn(n_samples),  # Independent
            "feature_4": -base_feature * 0.8 + np.random.randn(n_samples) * 0.2,  # Negatively correlated
            "label": (base_feature > 0).astype(int),  # Label correlated with base_feature
        })

        # Create step
        config = {"MATRIX_OPERATIONS": {"correlation_threshold": 0.8}}
        with patch("src.training.steps.step7_enhanced_matrix_operations.system_logger"):
            step = MatrixOperationsStep(config)

        # Analyze relationships
        matrix_results = {"correlation_matrix": test_data.corr().values}
        analysis = await step._analyze_feature_relationships(test_data, matrix_results)

        # Verify high correlation detected
        assert "highly_correlated_pairs" in analysis
        assert len(analysis["highly_correlated_pairs"]) > 0


if __name__ == "__main__":
    unittest.main()
