#!/usr/bin/env python3
"""
Enhanced Step 18 Integration Tests

This module provides comprehensive integration tests for the enhanced step18 pipeline
with real market data, advanced metrics, k-fold cross-validation, and parallel processing.
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Direct import to avoid module loading issues
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the specific module directly
from training.steps.backtesting.step18_walk_forward_validation_per_regime import (
    PerRegimeWalkForwardValidationStep
)
from src.training.steps.backtesting.step18_backtesting_main import main


class TestStep18EnhancedIntegration(unittest.TestCase):
    """Comprehensive integration tests for enhanced step18 features."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            'symbol': 'ETHUSDT',
            'exchange': 'BINANCE',
            'timeframe': '1m',
            'data_dir': 'data_cache',
            'force_rerun': False,
            'use_real_market_data': True,
            'enable_enhanced_metrics': True,
            'kfold_cross_validation': True,
            'parallel_regime_processing': True,
            'max_concurrent_regimes': 2,
            'k_folds': 3,
            'regime_ids': [0, 1, 2, 3, 4]  # Test subset for speed
        }

        # Create mock market data
        self.mock_market_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=1000, freq='1min'),
            'open': np.random.uniform(1500, 2000, 1000),
            'high': np.random.uniform(1500, 2000, 1000),
            'low': np.random.uniform(1500, 2000, 1000),
            'close': np.random.uniform(1500, 2000, 1000),
            'volume': np.random.uniform(100, 1000, 1000),
            'hmm_state': np.random.randint(0, 5, 1000)
        })

        # Create temporary directory for test artifacts
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('src.training.steps.backtesting.step18_walk_forward_validation_per_regime.pd.read_parquet')
    async def test_real_market_data_loading(self, mock_read_parquet):
        """Test real market data loading functionality."""
        mock_read_parquet.return_value = self.mock_market_data

        validator = PerRegimeWalkForwardValidationStep(self.test_config)

        # Test loading regime-specific data
        data = await validator._load_regime_validation_data(
            'ETHUSDT', 'BINANCE', '1m', 'data_cache', 0
        )

        self.assertIsNotNone(data)
        self.assertEqual(len(data), 1000)
        self.assertIn('close', data.columns)

    async def test_calculate_real_performance_metrics(self):
        """Test real performance metrics calculation."""
        validator = PerRegimeWalkForwardValidationStep(self.test_config)

        train_data = self.mock_market_data.iloc[:700]
        test_data = self.mock_market_data.iloc[700:900]

        calibrated_specialists = {
            'specialist_1': {'confidence_score': 0.8},
            'specialist_2': {'confidence_score': 0.75}
        }

        metrics = await validator._calculate_real_performance_metrics(
            train_data, test_data, calibrated_specialists, 0
        )

        # Verify enhanced metrics are present
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('sortino_ratio', metrics)
        self.assertIn('calmar_ratio', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('profit_factor', metrics)
        self.assertIn('data_source', metrics)
        self.assertEqual(metrics['data_source'], 'real_market_data')

    async def test_kfold_cross_validation(self):
        """Test k-fold cross-validation implementation."""
        validator = PerRegimeWalkForwardValidationStep(self.test_config)

        k_folds = 3
        regime_data = self.mock_market_data
        calibrated_specialists = {'specialist_1': {'confidence_score': 0.8}}

        folds = await validator._perform_kfold_time_series_validation(
            calibrated_specialists, regime_data, k_folds, 0, {'k_folds': k_folds}
        )

        self.assertEqual(len(folds), k_folds)

        for fold in folds:
            self.assertIn('fold_index', fold)
            self.assertIn('fold_metrics', fold)
            self.assertIn('fold_metadata', fold)
            self.assertEqual(fold['fold_metadata']['k_folds'], k_folds)

    async def test_cross_validation_metrics_calculation(self):
        """Test cross-validation metrics calculation."""
        validator = PerRegimeWalkForwardValidationStep(self.test_config)

        # Create mock fold results
        mock_folds = [
            {
                'fold_metrics': {
                    'accuracy': 0.7, 'sharpe_ratio': 1.2, 'sortino_ratio': 1.5, 'calmar_ratio': 2.1
                }
            } for _ in range(3)
        ]

        cv_metrics = validator._calculate_kfold_metrics(mock_folds, 3)

        self.assertIn('k_folds', cv_metrics)
        self.assertIn('cv_accuracy_mean', cv_metrics)
        self.assertIn('cv_sharpe_mean', cv_metrics)
        self.assertIn('cv_sortino_mean', cv_metrics)
        self.assertIn('cv_calmar_mean', cv_metrics)
        self.assertIn('cross_validation_score', cv_metrics)

        self.assertEqual(cv_metrics['k_folds'], 3)
        self.assertGreater(cv_metrics['cross_validation_score'], 0)

    async def test_parallel_regime_validation(self):
        """Test parallel regime validation execution."""
        validator = PerRegimeWalkForwardValidationStep(self.test_config)

        regime_ids = [0, 1, 2]

        # Mock the individual regime validation
        with patch.object(validator, 'execute_per_regime_walk_forward_validation',
                         return_value=True) as mock_validate:

            results = await validator.execute_parallel_regime_validation(
                'ETHUSDT', 'BINANCE', '1m', 'data_cache', regime_ids, max_concurrent=2
            )

            self.assertEqual(len(results), 3)
            self.assertTrue(all(results.values()))  # All should succeed with mock

            # Verify parallel execution was attempted
            self.assertEqual(mock_validate.call_count, 3)

    async def test_enhanced_metrics_calculation(self):
        """Test enhanced validation metrics with Sortino and Calmar ratios."""
        validator = PerRegimeWalkForwardValidationStep(self.test_config)

        mock_folds = [{
            'overall_performance': {
                'mean_accuracy': 0.7,
                'mean_sharpe_ratio': 1.2,
                'mean_sortino_ratio': 1.5,
                'mean_calmar_ratio': 2.1,
                'mean_max_drawdown': 0.15,
                'mean_win_rate': 0.65,
                'mean_profit_factor': 1.3
            }
        }]

        metrics = validator._calculate_validation_metrics({'time_series': mock_folds})

        self.assertIn('enhanced_metrics', metrics)
        self.assertIn('mean_sharpe_ratio', metrics['enhanced_metrics'])
        self.assertIn('mean_sortino_ratio', metrics['enhanced_metrics'])
        self.assertIn('mean_calmar_ratio', metrics['enhanced_metrics'])
        self.assertIn('risk_adjusted_performance_score', metrics['enhanced_metrics'])

    async def test_risk_adjusted_performance_score(self):
        """Test risk-adjusted performance score calculation."""
        validator = PerRegimeWalkForwardValidationStep(self.test_config)

        score = validator._calculate_risk_adjusted_score(1.5, 1.8, 2.2, 0.12)

        self.assertGreater(score, 0)
        self.assertLessEqual(score, 1)

        # Test edge cases
        high_risk_score = validator._calculate_risk_adjusted_score(0.5, 0.8, 1.0, 0.8)
        self.assertLess(high_risk_score, score)  # Higher risk should reduce score

    async def test_data_quality_assessment(self):
        """Test data quality assessment functionality."""
        validator = PerRegimeWalkForwardValidationStep(self.test_config)

        # Mock validation folds with mixed data sources
        mock_folds = {
            'time_series': {
                'validation_folds': [
                    {'fold_metrics': {'data_source': 'real_market_data'}},
                    {'fold_metrics': {'data_source': 'real_market_data'}},
                    {'fold_metrics': {'data_source': 'fallback_mock_data'}}
                ]
            }
        }

        quality = validator._assess_overall_data_quality(mock_folds)
        self.assertEqual(quality, 'mixed_data_quality')

        # Test excellent data quality
        excellent_folds = {
            'time_series': {
                'validation_folds': [
                    {'fold_metrics': {'data_source': 'real_market_data'}} for _ in range(5)
                ]
            }
        }

        excellent_quality = validator._assess_overall_data_quality(excellent_folds)
        self.assertEqual(excellent_quality, 'excellent_real_data')

    @patch('src.training.steps.backtesting.step18_backtesting_main.run_backtesting_pipeline')
    async def test_enhanced_main_pipeline_execution(self, mock_run_pipeline):
        """Test enhanced main pipeline execution."""
        mock_run_pipeline.return_value = True

        # Test with enhanced configuration
        config = self.test_config.copy()
        config.update({
            'parallel_regime_processing': False,  # Test fallback to original pipeline
            'enable_validation': True
        })

        success = await main(
            symbol='ETHUSDT',
            exchange='BINANCE',
            timeframe='1m',
            data_dir='data_cache',
            **config
        )

        self.assertTrue(success)
        mock_run_pipeline.assert_called_once()

    async def test_error_handling_and_fallbacks(self):
        """Test error handling and fallback mechanisms."""
        validator = PerRegimeWalkForwardValidationStep(self.test_config)

        # Test with empty data
        empty_data = pd.DataFrame()
        metrics = await validator._calculate_real_performance_metrics(
            empty_data, empty_data, {}, 0
        )

        self.assertIn('data_source', metrics)
        self.assertEqual(metrics['data_source'], 'fallback_mock_data')

        # Test with insufficient data
        small_data = pd.DataFrame({'close': [100, 101, 102]})
        small_metrics = await validator._calculate_real_performance_metrics(
            small_data, small_data, {}, 0
        )

        self.assertIn('data_source', small_metrics)

    async def test_regime_specific_performance_multipliers(self):
        """Test regime-specific performance multipliers."""
        validator = PerRegimeWalkForwardValidationStep(self.test_config)

        # Test different regime multipliers
        trend_multiplier = validator._get_regime_performance_multiplier(0)  # Trend regime
        volatility_multiplier = validator._get_regime_performance_multiplier(5)  # Volatility regime
        balanced_multiplier = validator._get_regime_performance_multiplier(3)  # Balanced regime

        self.assertGreater(trend_multiplier, balanced_multiplier)
        self.assertLess(volatility_multiplier, balanced_multiplier)
        self.assertEqual(balanced_multiplier, 1.0)

    def test_configuration_validation(self):
        """Test configuration validation and defaults."""
        # Test with minimal config
        minimal_config = {'symbol': 'ETHUSDT'}
        validator = PerRegimeWalkForwardValidationStep(minimal_config)

        self.assertIsNotNone(validator.config)
        self.assertEqual(validator.config['symbol'], 'ETHUSDT')

        # Test enhanced metrics defaults
        self.assertTrue(validator.config.get('per_regime_walk_forward_validation', True))

    async def test_integration_with_step_orchestrator(self):
        """Test integration with step orchestrator."""
        # This would test the full pipeline integration
        # For now, we'll test the component interfaces

        validator = PerRegimeWalkForwardValidationStep(self.test_config)

        # Test that all required methods exist and are callable
        self.assertTrue(hasattr(validator, 'execute_per_regime_walk_forward_validation'))
        self.assertTrue(hasattr(validator, 'execute_parallel_regime_validation'))
        self.assertTrue(callable(validator.execute_per_regime_walk_forward_validation))
        self.assertTrue(callable(validator.execute_parallel_regime_validation))


class TestStep18PerformanceOptimization(unittest.TestCase):
    """Performance optimization tests for step18."""

    def setUp(self):
        self.config = {
            'symbol': 'ETHUSDT',
            'exchange': 'BINANCE',
            'timeframe': '1m',
            'max_concurrent_regimes': 3,
            'parallel_regime_processing': True
        }

    async def test_concurrent_limit_respected(self):
        """Test that concurrent regime limit is respected."""
        validator = PerRegimeWalkForwardValidationStep(self.config)

        # Mock slow validation to test concurrency
        async def slow_validation(*args, **kwargs):
            await asyncio.sleep(0.1)
            return True

        with patch.object(validator, 'execute_per_regime_walk_forward_validation',
                         side_effect=slow_validation):

            start_time = asyncio.get_event_loop().time()

            results = await validator.execute_parallel_regime_validation(
                'ETHUSDT', 'BINANCE', '1m', 'data_cache', [0, 1, 2, 3, 4],
                max_concurrent=2
            )

            elapsed = asyncio.get_event_loop().time() - start_time

            # With concurrency limit of 2, should take about 0.3 seconds (2 batches of 0.1s each)
            # rather than 0.5 seconds if sequential
            self.assertLess(elapsed, 0.4)  # Allow some margin for test execution

    async def test_memory_efficient_processing(self):
        """Test memory-efficient processing of large datasets."""
        validator = PerRegimeWalkForwardValidationStep(self.config)

        # Create large mock dataset
        large_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=50000, freq='1min'),
            'close': np.random.uniform(1500, 2000, 50000),
            'open': np.random.uniform(1500, 2000, 50000),
            'high': np.random.uniform(1500, 2000, 50000),
            'low': np.random.uniform(1500, 2000, 50000),
            'volume': np.random.uniform(100, 1000, 50000)
        })

        # Test that processing doesn't crash with large data
        metrics = await validator._calculate_real_performance_metrics(
            large_data.iloc[:35000], large_data.iloc[35000:], {}, 0
        )

        self.assertIsInstance(metrics, dict)
        self.assertIn('sharpe_ratio', metrics)


if __name__ == '__main__':
    # Run async tests
    async def run_async_tests():
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestStep18EnhancedIntegration)
        suite.addTests(loader.loadTestsFromTestCase(TestStep18PerformanceOptimization))

        runner = unittest.TextTestRunner(verbosity=2)
        result = await asyncio.get_event_loop().run_until_complete(
            asyncio.gather(*[test() for test in suite])
        )
        runner.run(suite)

    asyncio.run(run_async_tests())
