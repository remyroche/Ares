"""
VectorBT Integration Tests

This module provides comprehensive tests for the VectorBT-enhanced components
to ensure they work correctly and provide expected performance improvements.

Run tests with:
    python -m pytest test_vectorbt_integration.py -v
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

# Import VectorBT components
from .vectorbt_backtesting_engine import (
    VectorBTBacktestingEngine, VectorBTBacktestConfig, BacktestMode,
    run_vectorbt_backtest, create_vectorbt_config
)
from .vectorbt_financial_metrics import (
    VectorBTFinancialMetrics, FinancialMetricsConfig,
    calculate_financial_metrics, create_metrics_config
)
from .vectorbt_portfolio_optimization import (
    VectorBTPortfolioOptimizer, OptimizationConfig, OptimizationMethod,
    optimize_portfolio, create_optimization_config
)
from .unified_vectorization_manager import (
    UnifiedVectorizationManager, OperationType, OptimizationStrategy,
    optimize_vectorbt_backtesting, optimize_vectorbt_metrics, optimize_vectorbt_portfolio
)

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
logger = logging.getLogger(__name__)


class TestVectorBTBacktesting(unittest.TestCase):
    """Test VectorBT backtesting engine."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_periods = 100
        self.n_assets = 3
        
        # Generate test data
        self.returns = np.random.normal(0.001, 0.02, (self.n_periods, self.n_assets))
        self.prices = 100 * (1 + self.returns).cumprod(axis=0)
        self.signals = np.random.choice([-1, 0, 1], size=(self.n_periods, self.n_assets))
        self.timestamps = pd.date_range(start='2020-01-01', periods=self.n_periods, freq='1min')
        
        # Create engine
        self.config = create_vectorbt_config(initial_capital=100000.0)
        self.engine = VectorBTBacktestingEngine(self.config)
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        self.assertIsNotNone(self.engine)
        self.assertIsNotNone(self.engine.config)
        self.assertEqual(self.engine.config.initial_capital, 100000.0)
    
    def test_cpu_backtesting(self):
        """Test CPU backtesting."""
        result = self.engine.run_backtest(
            self.signals, 
            self.prices, 
            self.timestamps,
            mode=BacktestMode.VECTORBT_CPU
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.portfolio_values)
        self.assertIsNotNone(result.returns)
        self.assertIsNotNone(result.performance_metrics)
        self.assertGreater(len(result.portfolio_values), 0)
        self.assertGreater(result.performance_metrics['total_return'], -1.0)  # Not total loss
    
    def test_parallel_backtesting(self):
        """Test parallel backtesting."""
        result = self.engine.run_backtest(
            self.signals, 
            self.prices, 
            self.timestamps,
            mode=BacktestMode.VECTORBT_PARALLEL
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.portfolio_values)
        self.assertIsNotNone(result.performance_metrics)
    
    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        result = self.engine.run_backtest(
            self.signals, 
            self.prices, 
            self.timestamps
        )
        
        metrics = result.performance_metrics
        
        # Check required metrics exist
        required_metrics = [
            'total_return', 'annualized_return', 'volatility', 
            'sharpe_ratio', 'max_drawdown', 'final_portfolio_value'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
    
    def test_risk_metrics(self):
        """Test risk metrics calculation."""
        result = self.engine.run_backtest(
            self.signals, 
            self.prices, 
            self.timestamps
        )
        
        risk_metrics = result.risk_metrics
        
        # Check required risk metrics exist
        required_risk_metrics = [
            'volatility', 'max_drawdown', 'sharpe_ratio', 'sortino_ratio'
        ]
        
        for metric in required_risk_metrics:
            self.assertIn(metric, risk_metrics)
            self.assertIsInstance(risk_metrics[metric], (int, float))
    
    def test_convenience_function(self):
        """Test convenience function."""
        result = run_vectorbt_backtest(
            self.signals, 
            self.prices, 
            self.timestamps,
            config=self.config
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.portfolio_values)
        self.assertIsNotNone(result.performance_metrics)


class TestVectorBTFinancialMetrics(unittest.TestCase):
    """Test VectorBT financial metrics."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_periods = 100
        
        # Generate test data
        self.returns = np.random.normal(0.001, 0.02, self.n_periods)
        self.portfolio_values = 100000 * (1 + self.returns).cumprod()
        self.timestamps = pd.date_range(start='2020-01-01', periods=self.n_periods, freq='1min')
        
        # Create calculator
        self.config = create_metrics_config(risk_free_rate=0.02)
        self.calculator = VectorBTFinancialMetrics(self.config)
    
    def test_calculator_initialization(self):
        """Test calculator initialization."""
        self.assertIsNotNone(self.calculator)
        self.assertIsNotNone(self.calculator.config)
        self.assertEqual(self.calculator.config.risk_free_rate, 0.02)
    
    def test_comprehensive_metrics(self):
        """Test comprehensive metrics calculation."""
        metrics = self.calculator.calculate_comprehensive_metrics(
            self.portfolio_values, 
            self.returns, 
            timestamps=self.timestamps
        )
        
        self.assertIsNotNone(metrics)
        self.assertIsInstance(metrics, dict)
        self.assertGreater(len(metrics), 0)
    
    def test_return_metrics(self):
        """Test return metrics."""
        metrics = self.calculator.calculate_comprehensive_metrics(
            self.portfolio_values, 
            self.returns
        )
        
        # Check return metrics exist
        return_metrics = [
            'total_return', 'annualized_return', 'cumulative_return'
        ]
        
        for metric in return_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
    
    def test_risk_metrics(self):
        """Test risk metrics."""
        metrics = self.calculator.calculate_comprehensive_metrics(
            self.portfolio_values, 
            self.returns
        )
        
        # Check risk metrics exist
        risk_metrics = [
            'volatility', 'var_95', 'cvar_95', 'skewness', 'kurtosis'
        ]
        
        for metric in risk_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
    
    def test_risk_adjusted_metrics(self):
        """Test risk-adjusted metrics."""
        metrics = self.calculator.calculate_comprehensive_metrics(
            self.portfolio_values, 
            self.returns
        )
        
        # Check risk-adjusted metrics exist
        risk_adjusted_metrics = [
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio'
        ]
        
        for metric in risk_adjusted_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
    
    def test_benchmark_comparison(self):
        """Test benchmark comparison."""
        benchmark_values = 100000 * (1 + np.random.normal(0.0008, 0.015, self.n_periods)).cumprod()
        
        metrics = self.calculator.calculate_comprehensive_metrics(
            self.portfolio_values, 
            self.returns, 
            benchmark_values=benchmark_values
        )
        
        # Check benchmark metrics exist
        benchmark_metrics = [
            'alpha', 'beta', 'tracking_error', 'relative_performance'
        ]
        
        for metric in benchmark_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
    
    def test_convenience_function(self):
        """Test convenience function."""
        metrics = calculate_financial_metrics(
            self.portfolio_values, 
            self.returns, 
            timestamps=self.timestamps
        )
        
        self.assertIsNotNone(metrics)
        self.assertIsInstance(metrics, dict)
        self.assertGreater(len(metrics), 0)


class TestVectorBTPortfolioOptimization(unittest.TestCase):
    """Test VectorBT portfolio optimization."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_periods = 100
        self.n_assets = 5
        
        # Generate test data
        self.returns = np.random.normal(0.001, 0.02, (self.n_periods, self.n_assets))
        self.asset_names = [f'Asset_{i+1}' for i in range(self.n_assets)]
        
        # Create optimizer
        self.config = create_optimization_config(
            method=OptimizationMethod.MEAN_VARIANCE,
            risk_aversion=1.0
        )
        self.optimizer = VectorBTPortfolioOptimizer(self.config)
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        self.assertIsNotNone(self.optimizer)
        self.assertIsNotNone(self.optimizer.config)
        self.assertEqual(self.optimizer.config.method, OptimizationMethod.MEAN_VARIANCE)
    
    def test_mean_variance_optimization(self):
        """Test mean-variance optimization."""
        result = self.optimizer.optimize_portfolio(
            self.returns, 
            asset_names=self.asset_names
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.weights)
        self.assertEqual(len(result.weights), self.n_assets)
        self.assertAlmostEqual(np.sum(result.weights), 1.0, places=5)  # Weights sum to 1
        self.assertGreaterEqual(result.weights.min(), 0.0)  # No negative weights
        self.assertLessEqual(result.weights.max(), 1.0)  # No weights > 1
    
    def test_risk_parity_optimization(self):
        """Test risk parity optimization."""
        config = create_optimization_config(method=OptimizationMethod.RISK_PARITY)
        optimizer = VectorBTPortfolioOptimizer(config)
        
        result = optimizer.optimize_portfolio(
            self.returns, 
            asset_names=self.asset_names
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.weights)
        self.assertEqual(len(result.weights), self.n_assets)
        self.assertAlmostEqual(np.sum(result.weights), 1.0, places=5)
    
    def test_equal_weight_optimization(self):
        """Test equal weight optimization."""
        config = create_optimization_config(method=OptimizationMethod.EQUAL_WEIGHT)
        optimizer = VectorBTPortfolioOptimizer(config)
        
        result = optimizer.optimize_portfolio(
            self.returns, 
            asset_names=self.asset_names
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.weights)
        self.assertEqual(len(result.weights), self.n_assets)
        self.assertAlmostEqual(np.sum(result.weights), 1.0, places=5)
        
        # All weights should be equal
        expected_weight = 1.0 / self.n_assets
        for weight in result.weights:
            self.assertAlmostEqual(weight, expected_weight, places=5)
    
    def test_min_variance_optimization(self):
        """Test minimum variance optimization."""
        config = create_optimization_config(method=OptimizationMethod.MIN_VARIANCE)
        optimizer = VectorBTPortfolioOptimizer(config)
        
        result = optimizer.optimize_portfolio(
            self.returns, 
            asset_names=self.asset_names
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.weights)
        self.assertEqual(len(result.weights), self.n_assets)
        self.assertAlmostEqual(np.sum(result.weights), 1.0, places=5)
    
    def test_max_sharpe_optimization(self):
        """Test maximum Sharpe ratio optimization."""
        config = create_optimization_config(method=OptimizationMethod.MAX_SHARPE)
        optimizer = VectorBTPortfolioOptimizer(config)
        
        result = optimizer.optimize_portfolio(
            self.returns, 
            asset_names=self.asset_names
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.weights)
        self.assertEqual(len(result.weights), self.n_assets)
        self.assertAlmostEqual(np.sum(result.weights), 1.0, places=5)
        self.assertIsNotNone(result.sharpe_ratio)
    
    def test_convenience_function(self):
        """Test convenience function."""
        result = optimize_portfolio(
            self.returns,
            method=OptimizationMethod.MEAN_VARIANCE,
            asset_names=self.asset_names
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.weights)
        self.assertEqual(len(result.weights), self.n_assets)
        self.assertAlmostEqual(np.sum(result.weights), 1.0, places=5)


class TestUnifiedVectorizationManager(unittest.TestCase):
    """Test unified vectorization manager with VectorBT."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_periods = 100
        self.n_assets = 3
        
        # Generate test data
        self.returns = np.random.normal(0.001, 0.02, (self.n_periods, self.n_assets))
        self.prices = 100 * (1 + self.returns).cumprod(axis=0)
        self.signals = np.random.choice([-1, 0, 1], size=(self.n_periods, self.n_assets))
        self.timestamps = pd.date_range(start='2020-01-01', periods=self.n_periods, freq='1min')
        
        # Create manager
        self.manager = UnifiedVectorizationManager()
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        self.assertIsNotNone(self.manager)
        self.assertIsNotNone(self.manager.hardware_caps)
        self.assertIsNotNone(self.manager.optimization_stats)
    
    def test_vectorbt_backtesting_operation(self):
        """Test VectorBT backtesting through unified manager."""
        data = {
            'signals': self.signals,
            'prices': self.prices,
            'timestamps': self.timestamps
        }
        
        result = self.manager.optimize_operation(
            OperationType.VECTORBT_BACKTESTING,
            data
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.result)
        self.assertIsNotNone(result.strategy_used)
        self.assertGreater(result.computation_time, 0)
    
    def test_vectorbt_metrics_operation(self):
        """Test VectorBT metrics through unified manager."""
        portfolio_values = 100000 * (1 + self.returns.sum(axis=1)).cumprod()
        
        data = {
            'portfolio_values': portfolio_values,
            'returns': self.returns.sum(axis=1),
            'timestamps': self.timestamps
        }
        
        result = self.manager.optimize_operation(
            OperationType.VECTORBT_METRICS,
            data
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.result)
        self.assertIsNotNone(result.strategy_used)
        self.assertGreater(result.computation_time, 0)
    
    def test_vectorbt_portfolio_optimization_operation(self):
        """Test VectorBT portfolio optimization through unified manager."""
        data = {
            'returns': self.returns,
            'asset_names': [f'Asset_{i+1}' for i in range(self.n_assets)]
        }
        
        result = self.manager.optimize_operation(
            OperationType.VECTORBT_PORTFOLIO_OPTIMIZATION,
            data
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.result)
        self.assertIsNotNone(result.strategy_used)
        self.assertGreater(result.computation_time, 0)
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        # Test VectorBT backtesting convenience function
        result = optimize_vectorbt_backtesting(
            self.signals,
            self.prices,
            self.timestamps
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.result)
        
        # Test VectorBT metrics convenience function
        portfolio_values = 100000 * (1 + self.returns.sum(axis=1)).cumprod()
        
        result = optimize_vectorbt_metrics(
            portfolio_values,
            self.returns.sum(axis=1),
            timestamps=self.timestamps
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.result)
        
        # Test VectorBT portfolio optimization convenience function
        result = optimize_vectorbt_portfolio(
            self.returns,
            asset_names=[f'Asset_{i+1}' for i in range(self.n_assets)]
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.result)
    
    def test_optimization_stats(self):
        """Test optimization statistics."""
        stats = self.manager.get_optimization_stats()
        
        self.assertIsNotNone(stats)
        self.assertIn('total_operations', stats)
        self.assertIn('available_optimizations', stats)
        self.assertIn('vectorbt_backtesting', stats['available_optimizations'])
        self.assertIn('vectorbt_metrics', stats['available_optimizations'])
        self.assertIn('vectorbt_portfolio_optimization', stats['available_optimizations'])


class TestPerformanceComparison(unittest.TestCase):
    """Test performance comparison between different approaches."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_periods = 500
        self.n_assets = 5
        
        # Generate test data
        self.returns = np.random.normal(0.001, 0.02, (self.n_periods, self.n_assets))
        self.prices = 100 * (1 + self.returns).cumprod(axis=0)
        self.signals = np.random.choice([-1, 0, 1], size=(self.n_periods, self.n_assets))
        self.timestamps = pd.date_range(start='2020-01-01', periods=self.n_periods, freq='1min')
    
    def test_backtesting_performance(self):
        """Test backtesting performance."""
        config = create_vectorbt_config(initial_capital=100000.0)
        engine = VectorBTBacktestingEngine(config)
        
        # Test CPU mode
        start_time = time.time()
        result_cpu = engine.run_backtest(
            self.signals, 
            self.prices, 
            self.timestamps,
            mode=BacktestMode.VECTORBT_CPU
        )
        cpu_time = time.time() - start_time
        
        # Test parallel mode
        start_time = time.time()
        result_parallel = engine.run_backtest(
            self.signals, 
            self.prices, 
            self.timestamps,
            mode=BacktestMode.VECTORBT_PARALLEL
        )
        parallel_time = time.time() - start_time
        
        # Both should complete successfully
        self.assertIsNotNone(result_cpu)
        self.assertIsNotNone(result_parallel)
        self.assertGreater(cpu_time, 0)
        self.assertGreater(parallel_time, 0)
        
        # Results should be similar
        self.assertAlmostEqual(
            result_cpu.performance_metrics['total_return'],
            result_parallel.performance_metrics['total_return'],
            places=2
        )
    
    def test_metrics_performance(self):
        """Test metrics calculation performance."""
        portfolio_values = 100000 * (1 + self.returns.sum(axis=1)).cumprod()
        
        config = create_metrics_config()
        calculator = VectorBTFinancialMetrics(config)
        
        start_time = time.time()
        metrics = calculator.calculate_comprehensive_metrics(
            portfolio_values,
            self.returns.sum(axis=1),
            timestamps=self.timestamps
        )
        metrics_time = time.time() - start_time
        
        self.assertIsNotNone(metrics)
        self.assertGreater(len(metrics), 0)
        self.assertGreater(metrics_time, 0)
        self.assertLess(metrics_time, 10.0)  # Should complete within 10 seconds
    
    def test_optimization_performance(self):
        """Test portfolio optimization performance."""
        config = create_optimization_config(method=OptimizationMethod.MEAN_VARIANCE)
        optimizer = VectorBTPortfolioOptimizer(config)
        
        start_time = time.time()
        result = optimizer.optimize_portfolio(
            self.returns,
            asset_names=[f'Asset_{i+1}' for i in range(self.n_assets)]
        )
        optimization_time = time.time() - start_time
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.weights)
        self.assertGreater(optimization_time, 0)
        self.assertLess(optimization_time, 30.0)  # Should complete within 30 seconds


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestVectorBTBacktesting,
        TestVectorBTFinancialMetrics,
        TestVectorBTPortfolioOptimization,
        TestUnifiedVectorizationManager,
        TestPerformanceComparison
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("🧪 Running VectorBT Integration Tests...")
    print("="*60)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
        print("VectorBT integration is working correctly.")
    else:
        print("\n❌ Some tests failed!")
        print("Please check the test output for details.")
    
    print("="*60)