"""Unit tests for step3_hmm_regime_discovery.py and step3_parameter_optimization.py.

This module tests the HMM regime discovery and parameter optimization functionality.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the modules to be tested
from src.training.steps.step3_hmm_regime_discovery import HMMRegimeDiscoveryStep
from src.training.steps.step3_parameter_optimization import ParameterOptimizationStep


class TestHMMRegimeDiscoveryStep(unittest.TestCase):
    """Test cases for HMMRegimeDiscoveryStep class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "SYMBOL": "ETHUSDT",
            "EXCHANGE": "BINANCE",
            "TIMEFRAME": "1m",
            "MIN_REGIME_DURATION": 100,
            "MAX_REGIMES": 5,
            "sr_breakout_predictor": {
                "use_optimized_params": True,
            },
        }

        # Mock the logger and sr_breakout_predictor
        with patch("src.training.steps.step3_hmm_regime_discovery.system_logger"):
            with patch("src.training.steps.step3_hmm_regime_discovery.sr_breakout_predictor"):
                self.step = HMMRegimeDiscoveryStep(self.config)

        # Create sample test data
        self.test_data = pd.DataFrame({
            "timestamp": pd.date_range(start="2024-01-01", periods=2000, freq="1min"),
            "open": np.random.rand(2000) * 100 + 100,
            "high": np.random.rand(2000) * 100 + 110,
            "low": np.random.rand(2000) * 100 + 90,
            "close": np.random.rand(2000) * 100 + 100,
            "volume": np.random.rand(2000) * 10000,
            "quote_volume": np.random.rand(2000) * 1000000,
            "trades": np.random.randint(10, 1000, 2000),
            "taker_buy_volume": np.random.rand(2000) * 5000,
            "taker_buy_quote_volume": np.random.rand(2000) * 500000,
        })

    def tearDown(self):
        """Clean up after tests."""

    def test_initialization(self):
        """Test HMMRegimeDiscoveryStep initialization."""
        assert self.step is not None
        assert self.step.config == self.config

    def test_validate_environment(self):
        """Test environment validation."""
        with patch("src.training.steps.step3_hmm_regime_discovery.dependency_status",
                  {"pandas": True, "numpy": True, "psutil": True}):
            # Should not raise any exceptions
            self.step._validate_environment()

    def test_initialize_components(self):
        """Test component initialization."""
        with patch("src.training.steps.step3_hmm_regime_discovery.sr_breakout_predictor") as mock_sr:
            mock_sr.SRBreakoutPredictor = Mock(return_value=Mock())
            self.step._initialize_components()
            # Should initialize without errors

    @patch("src.training.steps.step3_hmm_regime_discovery.pandas")
    async def test_prepare_data_for_hmm_success(self, mock_pd):
        """Test successful data preparation for HMM."""
        # Set up mock
        mock_pd.DataFrame = pd.DataFrame

        # Create pipeline state with data
        pipeline_state = {
            "unified_data": self.test_data,
        }

        # Run the method
        result = await self.step._prepare_data_for_hmm(pipeline_state)

        # Verify result
        assert result is not None
        assert isinstance(result, pd.DataFrame)

    async def test_prepare_data_for_hmm_missing_data(self):
        """Test data preparation when unified data is missing."""
        # Create pipeline state without data
        pipeline_state = {}

        # Run the method
        result = await self.step._prepare_data_for_hmm(pipeline_state)

        # Verify result
        assert result is None

    @patch("src.training.steps.step3_hmm_regime_discovery.HMMRegimeDiscoveryStep._prepare_data_for_hmm")
    @patch("src.training.steps.step3_hmm_regime_discovery.HMMRegimeDiscoveryStep._run_hmm_discovery")
    @patch("src.training.steps.step3_hmm_regime_discovery.HMMRegimeDiscoveryStep._validate_regimes")
    @patch("src.training.steps.step3_hmm_regime_discovery.HMMRegimeDiscoveryStep._save_regime_results")
    async def test_execute_success(self, mock_save, mock_validate, mock_run_hmm, mock_prepare):
        """Test successful execution of HMM regime discovery."""
        # Set up mocks
        mock_prepare.return_value = self.test_data
        mock_run_hmm.return_value = {
            "regimes": [0, 0, 1, 1, 2, 2] * 333 + [0, 0],  # 2000 items
            "regime_stats": {
                "0": {"count": 666, "mean_return": 0.001},
                "1": {"count": 666, "mean_return": -0.001},
                "2": {"count": 668, "mean_return": 0.002},
            },
        }
        mock_validate.return_value = True
        mock_save.return_value = True

        # Create input
        training_input = {
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "timeframe": "1m",
        }
        pipeline_state = {
            "unified_data": self.test_data,
        }

        # Patch decorators
        with patch("src.training.steps.step3_hmm_regime_discovery.handle_errors", lambda **kwargs: lambda fn: fn):
            # Run the method
            result = await self.step.execute(training_input, pipeline_state)

        # Verify result
        assert result["success"]
        assert "regimes" in result
        assert "regime_stats" in result

    @patch("src.training.steps.step3_hmm_regime_discovery.HMMRegimeDiscoveryStep._prepare_data_for_hmm")
    async def test_execute_data_preparation_failure(self, mock_prepare):
        """Test execution when data preparation fails."""
        # Set up mock
        mock_prepare.return_value = None

        # Create input
        training_input = {
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "timeframe": "1m",
        }
        pipeline_state = {}

        # Patch decorators
        with patch("src.training.steps.step3_hmm_regime_discovery.handle_errors", lambda **kwargs: lambda fn: fn):
            # Run the method
            result = await self.step.execute(training_input, pipeline_state)

        # Verify result
        assert not result["success"]
        assert "error" in result

    async def test_run_hmm_discovery(self):
        """Test HMM discovery execution."""
        # This is a placeholder test as the actual implementation would require HMMLearn
        with patch("src.training.steps.step3_hmm_regime_discovery.HMMRegimeDiscoveryStep._fit_hmm_model") as mock_fit:
            mock_fit.return_value = {
                "regimes": np.array([0, 1, 0, 1, 2] * 400),
                "model": Mock(),
            }

            # Run method (if it exists)
            # result = await self.step._run_hmm_discovery(self.test_data)
            # self.assertIsNotNone(result)

    def test_validate_regimes(self):
        """Test regime validation logic."""
        # Create sample regimes
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2] * 100)

        # Test validation (method implementation dependent)
        # result = self.step._validate_regimes(regimes)
        # self.assertTrue(result)


class TestParameterOptimizationStep(unittest.TestCase):
    """Test cases for ParameterOptimizationStep class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "parameter_optimization": {
                "n_components_range": [2, 3, 4, 5],
                "covariance_type": ["full", "diag"],
                "n_iter": 100,
                "random_state": 42,
            },
        }

        # Mock the logger
        with patch("src.training.steps.step3_parameter_optimization.system_logger"):
            self.step = ParameterOptimizationStep(self.config)

    def tearDown(self):
        """Clean up after tests."""

    def test_initialization(self):
        """Test ParameterOptimizationStep initialization."""
        assert self.step is not None
        assert self.step.config == self.config

    @patch("src.training.steps.step3_parameter_optimization.secure_step_execution", lambda fn: fn)
    def test_initialize_components(self):
        """Test component initialization."""
        # Should not raise any exceptions
        self.step._initialize_components()

    async def test_initialize(self):
        """Test step initialization."""
        # Patch decorators
        with patch("src.training.steps.step3_parameter_optimization.handle_errors", lambda **kwargs: lambda fn: fn):
            with patch("src.training.steps.step3_parameter_optimization.secure_step_execution", lambda fn: fn):
                result = await self.step.initialize()
                assert result

    @patch("src.training.steps.step3_parameter_optimization.ParameterOptimizationStep._optimize_hmm_parameters")
    @patch("src.training.steps.step3_parameter_optimization.ParameterOptimizationStep._optimize_feature_parameters")
    @patch("src.training.steps.step3_parameter_optimization.ParameterOptimizationStep._save_optimization_results")
    async def test_execute_success(self, mock_save, mock_optimize_features, mock_optimize_hmm):
        """Test successful parameter optimization execution."""
        # Set up mocks
        mock_optimize_hmm.return_value = {
            "best_n_components": 3,
            "best_covariance_type": "full",
            "best_score": 0.95,
        }
        mock_optimize_features.return_value = {
            "best_features": ["return", "volume", "volatility"],
            "feature_importance": {"return": 0.5, "volume": 0.3, "volatility": 0.2},
        }
        mock_save.return_value = True

        # Patch decorators
        with patch("src.training.steps.step3_parameter_optimization.monitor_step_execution", lambda fn: fn):
            with patch("src.training.steps.step3_parameter_optimization.secure_step_execution", lambda fn: fn):
                with patch("src.training.steps.step3_parameter_optimization.validate_pipeline_step", lambda fn: fn):
                    with patch("src.training.steps.step3_parameter_optimization.handle_errors", lambda **kwargs: lambda fn: fn):
                        # Run the method
                        result = await self.step.execute()

        # Verify result
        assert result

    async def test_execute_optimization_failure(self):
        """Test parameter optimization execution failure."""
        # Patch decorators and force an exception
        with patch("src.training.steps.step3_parameter_optimization.monitor_step_execution", lambda fn: fn):
            with patch("src.training.steps.step3_parameter_optimization.secure_step_execution", lambda fn: fn):
                with patch("src.training.steps.step3_parameter_optimization.validate_pipeline_step", lambda fn: fn):
                    with patch("src.training.steps.step3_parameter_optimization.handle_errors", lambda **kwargs: lambda fn: fn):
                        with patch.object(self.step, "_optimize_hmm_parameters", side_effect=Exception("Optimization failed")):
                            # Run the method
                            result = await self.step.execute()

        # Verify result
        assert not result

    def test_save_optimization_results(self):
        """Test saving optimization results."""
        # Create sample results

        # Test save functionality (implementation dependent)
        # result = self.step._save_optimization_results(results)
        # self.assertTrue(result)


class TestHMMRegimeIntegration(unittest.TestCase):
    """Integration tests for HMM regime discovery pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = Path("/tmp/test_hmm_regime")
        self.test_data_dir.mkdir(exist_ok=True)

    def tearDown(self):
        """Clean up test data."""
        import shutil
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)

    async def test_hmm_parameter_optimization_integration(self):
        """Test integration between HMM discovery and parameter optimization."""
        # Create configuration
        config = {
            "SYMBOL": "ETHUSDT",
            "EXCHANGE": "BINANCE",
            "TIMEFRAME": "1m",
            "parameter_optimization": {
                "enabled": True,
                "n_components_range": [2, 3, 4],
                "cv_folds": 3,
            },
        }

        # Initialize steps with mocked dependencies
        with patch("src.training.steps.step3_hmm_regime_discovery.system_logger"):
            with patch("src.training.steps.step3_parameter_optimization.system_logger"):
                HMMRegimeDiscoveryStep(config)
                opt_step = ParameterOptimizationStep(config)

        # Run optimization first
        with patch.object(opt_step, "_optimize_hmm_parameters", return_value={"best_n_components": 3}):
            with patch.object(opt_step, "_save_optimization_results", return_value=True):
                # Test that optimization can be initialized
                opt_initialized = await opt_step.initialize()
                assert opt_initialized


if __name__ == "__main__":
    unittest.main()
