"""
Tests for Hybrid Regime Orchestrator

Tests for the main orchestrator that replaces HMM clustering functionality.
"""

import importlib
import sys
import types
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from datetime import datetime
import tempfile
import json
from pathlib import Path

from ..config.hybrid_regime_config import HybridRegimeConfig
try:
    from ..integration.hybrid_orchestrator import HybridRegimeOrchestrator
except ImportError:  # pragma: no cover - allow tests to run with partial dependencies
    HybridRegimeOrchestrator = None


class TestHybridOrchestrator(unittest.TestCase):
    """Test cases for Hybrid Regime Orchestrator."""

    def setUp(self):
        """Set up test fixtures."""
        if HybridRegimeOrchestrator is None:
            self.skipTest("HybridRegimeOrchestrator dependencies unavailable")
        self.config = HybridRegimeConfig(n_regimes=4)
        self.orchestrator = HybridRegimeOrchestrator(self.config)

        # Create sample market data
        self.sample_data = self._create_sample_market_data(500)

    def _create_sample_market_data(self, n_samples: int) -> pd.DataFrame:
        """Create sample market data for testing."""
        np.random.seed(42)
        timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='1H')

        # Simple price series
        prices = 100 + np.cumsum(np.random.normal(0, 0.01, n_samples))

        return pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, n_samples)
        })

    def test_initialization(self):
        """Test orchestrator initialization."""
        self.assertIsNotNone(self.orchestrator)
        self.assertIsNotNone(self.orchestrator.hybrid_detector)
        self.assertIsNotNone(self.orchestrator.regime_tagger)

    def test_detect_regimes(self):
        """Test regime detection through orchestrator."""
        result = self.orchestrator.detect_regimes(
            self.sample_data,
            symbol="TEST",
            exchange="test_exchange",
            timeframe="1h",
            save_results=False
        )

        self.assertTrue(result['success'])
        self.assertIn('regime_data', result)
        self.assertIn('economic_analysis', result)
        self.assertIn('financial_analysis', result)
        self.assertIn('performance_metrics', result)

        # Check regime data structure
        regime_data = result['regime_data']
        self.assertIn('predictions', regime_data)
        self.assertIn('probabilities', regime_data)
        self.assertEqual(len(regime_data['predictions']), len(self.sample_data))

    def test_tag_existing_data(self):
        """Test data tagging functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample data file
            data_file = Path(temp_dir) / "test_data.csv"
            self.sample_data.to_csv(data_file, index=False)

            # Tag data
            tagging_result = self.orchestrator.tag_existing_data(
                str(Path(temp_dir)),
                file_pattern="*.csv"
            )

            # Should complete successfully
            self.assertIsInstance(tagging_result, dict)

    def test_create_regime_aware_dataset(self):
        """Test creation of regime-aware dataset."""
        # First detect regimes
        detection_result = self.orchestrator.detect_regimes(
            self.sample_data,
            save_results=False
        )

        # Create regime-aware dataset
        regime_dataset = self.orchestrator.create_regime_aware_dataset(
            self.sample_data, detection_result
        )

        self.assertIsInstance(regime_dataset, pd.DataFrame)
        self.assertIn('regime_id', regime_dataset.columns)
        self.assertIn('regime_confidence', regime_dataset.columns)
        self.assertIn('economic_significance', regime_dataset.columns)
        self.assertIn('financial_relevance', regime_dataset.columns)

    def test_split_by_regime(self):
        """Test splitting dataset by regime."""
        # First detect regimes
        detection_result = self.orchestrator.detect_regimes(
            self.sample_data,
            save_results=False
        )

        # Create split regime datasets
        split_datasets = self.orchestrator.create_regime_aware_dataset(
            self.sample_data, detection_result, split_by_regime=True
        )

        self.assertIsInstance(split_datasets, dict)
        self.assertGreater(len(split_datasets), 0)

        # Each split should be a DataFrame
        for regime_name, regime_data in split_datasets.items():
            self.assertIsInstance(regime_data, pd.DataFrame)
            self.assertIn('regime_id', regime_data.columns)

    def test_get_regime_summary(self):
        """Test regime summary generation."""
        # Test without detection results
        summary = self.orchestrator.get_regime_summary()
        self.assertIn('error', summary)

        # Test with detection results
        self.orchestrator.detect_regimes(self.sample_data, save_results=False)
        summary = self.orchestrator.get_regime_summary()

        if 'error' not in summary:
            self.assertIn('n_regimes', summary)
            self.assertIn('avg_economic_significance', summary)
            self.assertIn('avg_financial_relevance', summary)

    def test_save_load_model(self):
        """Test model save and load functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_file = Path(temp_dir) / "test_model.pkl"

            # Save model
            self.orchestrator.save_model(str(model_file))

            # Create new orchestrator and load model
            new_orchestrator = HybridRegimeOrchestrator(self.config)
            new_orchestrator.load_model(str(model_file))

            # Should load successfully
            self.assertIsNotNone(new_orchestrator.performance_history)

    def test_performance_history_tracking(self):
        """Test performance history tracking."""
        # Initially empty
        self.assertEqual(len(self.orchestrator.performance_history), 0)

        # After detection, should have entry
        self.orchestrator.detect_regimes(self.sample_data, save_results=False)
        self.assertGreater(len(self.orchestrator.performance_history), 0)

        # Check structure of performance entry
        latest_entry = self.orchestrator.performance_history[-1]
        self.assertIn('timestamp', latest_entry)
        self.assertIn('symbol', latest_entry)
        self.assertIn('exchange', latest_entry)
        self.assertIn('timeframe', latest_entry)
        self.assertIn('n_regimes', latest_entry)
        self.assertIn('avg_economic_significance', latest_entry)
        self.assertIn('execution_time', latest_entry)

    def test_error_handling(self):
        """Test error handling in orchestrator."""
        # Test with invalid data
        invalid_result = self.orchestrator.detect_regimes("invalid_data")
        self.assertFalse(invalid_result['success'])
        self.assertIn('error', invalid_result)

        # Test tagging with non-existent directory
        tagging_result = self.orchestrator.tag_existing_data("/nonexistent/path")
        self.assertFalse(tagging_result['success'])

    def test_different_configurations(self):
        """Test orchestrator with different configurations."""
        configs = [
            HybridRegimeConfig(n_regimes=3),
            HybridRegimeConfig(n_regimes=5, combination_strategy=HybridRegimeConfig.RegimeCombinationStrategy.ECONOMIC_PRIORITY),
            HybridRegimeConfig(n_regimes=6, combination_strategy=HybridRegimeConfig.RegimeCombinationStrategy.MULTI_OBJECTIVE)
        ]

        for config in configs:
            with self.subTest(n_regimes=config.n_regimes):
                orchestrator = HybridRegimeOrchestrator(config)
                result = orchestrator.detect_regimes(self.sample_data, save_results=False)

                if result['success']:
                    self.assertEqual(
                        len(set(result['regime_data']['predictions'])),
                        config.n_regimes
                    )


class TestEnhancedHybridOrchestratorConfig(unittest.TestCase):
    """Tests for the enhanced hybrid orchestrator configuration toggles."""

    def test_boolean_feature_toggles_propagate(self):
        """Ensure boolean flags on the config are accessible as attributes."""

        config = HybridRegimeConfig(
            n_regimes=3,
            enable_multi_timeframe=False,
            use_unified_search=False,
            use_signal_generation=False,
        )
        if not hasattr(config, "search_config"):
            config.search_config = {}

        module_path = "src.training.steps.market_analysis.hybrid_nas_tas_regime.enhanced_hybrid_orchestrator"

        # Provide stub modules required for importing the enhanced orchestrator
        components_pkg = types.ModuleType("components")
        tas_module = types.ModuleType("tas_integration")
        nas_module = types.ModuleType("nas_integration")
        setattr(tas_module, "TASIntegrationComponent", MagicMock)
        setattr(nas_module, "NASIntegrationComponent", MagicMock)

        sys.modules.setdefault("src.training.steps.market_analysis.hybrid_nas_tas_regime.components", components_pkg)
        sys.modules.setdefault(
            "src.training.steps.market_analysis.hybrid_nas_tas_regime.components.tas_integration",
            tas_module
        )
        sys.modules.setdefault(
            "src.training.steps.market_analysis.hybrid_nas_tas_regime.components.nas_integration",
            nas_module
        )

        utility_module = types.ModuleType("enhanced_utility_integration")
        setattr(utility_module, "EnhancedUtilityIntegration", MagicMock)
        setattr(utility_module, "UtilityIntegrationConfig", MagicMock)
        setattr(utility_module, "create_enhanced_utility_integration", MagicMock(return_value=MagicMock()))
        sys.modules.setdefault(
            "src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.enhanced_utility_integration",
            utility_module
        )

        data_module = types.ModuleType("enhanced_data_integration")
        setattr(data_module, "EnhancedDataIntegration", MagicMock)
        setattr(data_module, "DataIntegrationConfig", MagicMock)
        setattr(data_module, "create_enhanced_data_integration", MagicMock(return_value=MagicMock()))
        sys.modules.setdefault(
            "src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.enhanced_data_integration",
            data_module
        )

        ml_module = types.ModuleType("enhanced_ml_integration")
        setattr(ml_module, "EnhancedMLIntegration", MagicMock)
        setattr(ml_module, "MLIntegrationConfig", MagicMock)
        setattr(ml_module, "create_enhanced_ml_integration", MagicMock(return_value=MagicMock()))
        sys.modules.setdefault(
            "src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.enhanced_ml_integration",
            ml_module
        )

        enhanced_module = importlib.import_module(module_path)
        EnhancedHybridOrchestrator = enhanced_module.EnhancedHybridOrchestrator

        with patch.object(EnhancedHybridOrchestrator, "_initialize_enhanced_utilities", return_value=None), \
             patch(f"{module_path}.TASIntegrationComponent", return_value=MagicMock()), \
             patch(f"{module_path}.NASIntegrationComponent", return_value=MagicMock()), \
             patch(f"{module_path}.create_unified_search_manager", return_value=MagicMock()), \
             patch(f"{module_path}.create_unified_clustering_algorithm", return_value=MagicMock()):
            orchestrator = EnhancedHybridOrchestrator(config)

        self.assertFalse(orchestrator.enable_multi_timeframe)
        self.assertFalse(orchestrator.use_unified_search)
        self.assertFalse(orchestrator.use_signal_generation)


if __name__ == '__main__':
    unittest.main()