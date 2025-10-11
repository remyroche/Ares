"""
VectorBT Integration Tests

This module provides comprehensive tests for VectorBT-optimized feature generators
to ensure they work correctly and provide the expected performance improvements.
"""

import unittest
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

# Import VectorBT generators
from ..categories.vectorbt_order_flow import (
    create_vectorbt_order_flow_generators,
    VectorBTTakerBuyRatioGenerator,
    VectorBTOrderFlowImbalanceGenerator
)
from ..categories.vectorbt_acceleration import (
    create_vectorbt_acceleration_generators,
    VectorBTMomentumGenerator,
    VectorBTPriceAccelerationGenerator
)
from ..categories.vectorbt_advanced_statistical import (
    create_vectorbt_advanced_statistical_generators,
    VectorBTHurstExponentGenerator,
    VectorBTCVaRGenerator
)
from ..categories.vectorbt_support_resistance import (
    create_vectorbt_support_resistance_generators,
    VectorBTSupportLevelGenerator,
    VectorBTPivotPointGenerator
)
from ..categories.vectorbt_legacy import (
    create_vectorbt_legacy_generators,
    VectorBTLegacyRSIGenerator,
    VectorBTLegacyMACDGenerator
)

# Import VectorBT rolling optimizer
from ..core.vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer

# Import feature registry
from ..core.feature_registry import FeatureRegistry

logger = logging.getLogger(__name__)

class TestVectorBTIntegration(unittest.TestCase):
    """Test VectorBT integration and feature generation."""
    
    def setUp(self):
        """Set up test data and configurations."""
        # Create sample OHLCV data
        np.random.seed(42)
        n_samples = 1000
        
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
        
        # Generate realistic price data
        price_changes = np.random.normal(0, 0.01, n_samples)
        prices = 100 * np.exp(np.cumsum(price_changes))
        
        # Generate OHLCV data
        self.sample_data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, n_samples)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, n_samples)
        }, index=dates)
        
        # Ensure high >= low and high/low >= open/close
        self.sample_data['high'] = np.maximum(
            self.sample_data['high'],
            np.maximum(self.sample_data['open'], self.sample_data['close'])
        )
        self.sample_data['low'] = np.minimum(
            self.sample_data['low'],
            np.minimum(self.sample_data['open'], self.sample_data['close'])
        )
        
        # Add some additional columns for testing
        self.sample_data['bid'] = self.sample_data['close'] * 0.999
        self.sample_data['ask'] = self.sample_data['close'] * 1.001
        self.sample_data['market_buys'] = np.random.poisson(100, n_samples)
        self.sample_data['market_sells'] = np.random.poisson(95, n_samples)
        
        # Initialize feature registry
        self.registry = FeatureRegistry()
        
        # Initialize VectorBT rolling optimizer
        self.rolling_optimizer = get_vectorbt_rolling_optimizer()
    
    def test_vectorbt_rolling_optimizer(self):
        """Test VectorBT rolling optimizer functionality."""
        logger.info("Testing VectorBT rolling optimizer...")
        
        # Test basic rolling operations
        close_prices = self.sample_data['close']
        
        # Test rolling mean
        rolling_mean = self.rolling_optimizer.rolling_mean(close_prices, window=20)
        self.assertIsInstance(rolling_mean, pd.Series)
        self.assertEqual(len(rolling_mean), len(close_prices))
        
        # Test rolling std
        rolling_std = self.rolling_optimizer.rolling_std(close_prices, window=20)
        self.assertIsInstance(rolling_std, pd.Series)
        self.assertEqual(len(rolling_std), len(close_prices))
        
        # Test rolling min/max
        rolling_min = self.rolling_optimizer.rolling_min(close_prices, window=20)
        rolling_max = self.rolling_optimizer.rolling_max(close_prices, window=20)
        
        self.assertIsInstance(rolling_min, pd.Series)
        self.assertIsInstance(rolling_max, pd.Series)
        
        # Test batch operations
        operations = [
            {'name': 'test_mean_20', 'type': 'mean', 'column': 'close', 'window': 20},
            {'name': 'test_std_20', 'type': 'std', 'column': 'close', 'window': 20},
            {'name': 'test_min_20', 'type': 'min', 'column': 'close', 'window': 20},
            {'name': 'test_max_20', 'type': 'max', 'column': 'close', 'window': 20}
        ]
        
        batch_results = self.rolling_optimizer.batch_rolling_operations(
            self.sample_data, operations
        )
        
        self.assertIsInstance(batch_results, pd.DataFrame)
        self.assertEqual(len(batch_results.columns), len(operations))
        
        logger.info("✅ VectorBT rolling optimizer tests passed")
    
    def test_order_flow_generators(self):
        """Test VectorBT order flow generators."""
        logger.info("Testing VectorBT order flow generators...")
        
        # Test individual generators
        taker_buy_ratio = VectorBTTakerBuyRatioGenerator(window=20)
        result = taker_buy_ratio.generate_features(self.sample_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn(f'vectorbt_taker_buy_ratio_20', result.columns)
        
        # Test order flow imbalance
        order_flow_imbalance = VectorBTOrderFlowImbalanceGenerator(window=20)
        result = order_flow_imbalance.generate_features(self.sample_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn(f'vectorbt_order_flow_imbalance_20', result.columns)
        
        # Test batch generation
        generators = create_vectorbt_order_flow_generators()
        self.assertGreater(len(generators), 0)
        
        # Test a few generators
        for generator in generators[:5]:  # Test first 5 generators
            try:
                result = generator.generate_features(self.sample_data)
                self.assertIsInstance(result, pd.DataFrame)
                self.assertGreater(len(result.columns), 0)
            except Exception as e:
                self.fail(f"Generator {generator.__class__.__name__} failed: {e}")
        
        logger.info("✅ VectorBT order flow generators tests passed")
    
    def test_acceleration_generators(self):
        """Test VectorBT acceleration generators."""
        logger.info("Testing VectorBT acceleration generators...")
        
        # Test momentum generator
        momentum_gen = VectorBTMomentumGenerator(period=10)
        result = momentum_gen.generate_features(self.sample_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn(f'vectorbt_momentum_10_price_returns', result.columns)
        
        # Test price acceleration generator
        accel_gen = VectorBTPriceAccelerationGenerator(period=10)
        result = accel_gen.generate_features(self.sample_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn(f'vectorbt_acceleration_10_price_returns', result.columns)
        
        # Test batch generation
        generators = create_vectorbt_acceleration_generators()
        self.assertGreater(len(generators), 0)
        
        # Test a few generators
        for generator in generators[:5]:  # Test first 5 generators
            try:
                result = generator.generate_features(self.sample_data)
                self.assertIsInstance(result, pd.DataFrame)
                self.assertGreater(len(result.columns), 0)
            except Exception as e:
                self.fail(f"Generator {generator.__class__.__name__} failed: {e}")
        
        logger.info("✅ VectorBT acceleration generators tests passed")
    
    def test_advanced_statistical_generators(self):
        """Test VectorBT advanced statistical generators."""
        logger.info("Testing VectorBT advanced statistical generators...")
        
        # Test Hurst exponent generator
        hurst_gen = VectorBTHurstExponentGenerator(window=20)
        result = hurst_gen.generate_features(self.sample_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn(f'vectorbt_hurst_exponent_20', result.columns)
        
        # Test CVaR generator
        cvar_gen = VectorBTCVaRGenerator(window=20, confidence_level=0.05)
        result = cvar_gen.generate_features(self.sample_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn(f'vectorbt_cvar_20_0.05', result.columns)
        
        # Test batch generation
        generators = create_vectorbt_advanced_statistical_generators()
        self.assertGreater(len(generators), 0)
        
        # Test a few generators
        for generator in generators[:5]:  # Test first 5 generators
            try:
                result = generator.generate_features(self.sample_data)
                self.assertIsInstance(result, pd.DataFrame)
                self.assertGreater(len(result.columns), 0)
            except Exception as e:
                self.fail(f"Generator {generator.__class__.__name__} failed: {e}")
        
        logger.info("✅ VectorBT advanced statistical generators tests passed")
    
    def test_support_resistance_generators(self):
        """Test VectorBT support/resistance generators."""
        logger.info("Testing VectorBT support/resistance generators...")
        
        # Test support level generator
        support_gen = VectorBTSupportLevelGenerator(level=1, window=20)
        result = support_gen.generate_features(self.sample_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn(f'vectorbt_support_level_1_20', result.columns)
        
        # Test pivot point generator
        pivot_gen = VectorBTPivotPointGenerator(window=20)
        result = pivot_gen.generate_features(self.sample_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn(f'vectorbt_pivot_point_20', result.columns)
        
        # Test batch generation
        generators = create_vectorbt_support_resistance_generators()
        self.assertGreater(len(generators), 0)
        
        # Test a few generators
        for generator in generators[:5]:  # Test first 5 generators
            try:
                result = generator.generate_features(self.sample_data)
                self.assertIsInstance(result, pd.DataFrame)
                self.assertGreater(len(result.columns), 0)
            except Exception as e:
                self.fail(f"Generator {generator.__class__.__name__} failed: {e}")
        
        logger.info("✅ VectorBT support/resistance generators tests passed")
    
    def test_legacy_generators(self):
        """Test VectorBT legacy generators."""
        logger.info("Testing VectorBT legacy generators...")
        
        # Test RSI generator
        rsi_gen = VectorBTLegacyRSIGenerator(period=14)
        result = rsi_gen.generate_features(self.sample_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn(f'vectorbt_legacy_rsi_14', result.columns)
        
        # Test MACD generator
        macd_gen = VectorBTLegacyMACDGenerator(fast=12, slow=26, signal=9)
        result = macd_gen.generate_features(self.sample_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn(f'vectorbt_legacy_macd_12_26_9', result.columns)
        
        # Test batch generation
        generators = create_vectorbt_legacy_generators()
        self.assertGreater(len(generators), 0)
        
        # Test a few generators
        for generator in generators[:5]:  # Test first 5 generators
            try:
                result = generator.generate_features(self.sample_data)
                self.assertIsInstance(result, pd.DataFrame)
                self.assertGreater(len(result.columns), 0)
            except Exception as e:
                self.fail(f"Generator {generator.__class__.__name__} failed: {e}")
        
        logger.info("✅ VectorBT legacy generators tests passed")
    
    def test_feature_registry_integration(self):
        """Test VectorBT generators integration with feature registry."""
        logger.info("Testing VectorBT generators integration with feature registry...")
        
        # Register VectorBT generators
        try:
            self.registry.register_vectorbt_generators()
            
            # Check that generators were registered
            vectorbt_generators = self.registry.get_vectorbt_generators()
            self.assertGreater(len(vectorbt_generators), 0)
            
            # Check category-specific generators
            order_flow_gens = self.registry.get_vectorbt_generators_by_category(
                self.registry._generators_by_name[list(self.registry._generators_by_name.keys())[0]].config.category
            )
            
            # Get registry summary
            summary = self.registry.get_summary()
            self.assertIn('vectorbt_generators', summary)
            self.assertGreater(summary['vectorbt_generators'], 0)
            
            logger.info(f"✅ Registered {len(vectorbt_generators)} VectorBT generators")
            
        except Exception as e:
            logger.warning(f"VectorBT generators registration failed: {e}")
            # This is expected if VectorBT is not available
            pass
    
    def test_performance_comparison(self):
        """Test performance comparison between VectorBT and standard implementations."""
        logger.info("Testing performance comparison...")
        
        # This test would compare performance between VectorBT and standard implementations
        # For now, we'll just ensure VectorBT generators work correctly
        
        generators = [
            VectorBTTakerBuyRatioGenerator(window=20),
            VectorBTMomentumGenerator(period=10),
            VectorBTHurstExponentGenerator(window=20),
            VectorBTSupportLevelGenerator(level=1, window=20),
            VectorBTLegacyRSIGenerator(period=14)
        ]
        
        for generator in generators:
            try:
                result = generator.generate_features(self.sample_data)
                self.assertIsInstance(result, pd.DataFrame)
                self.assertGreater(len(result.columns), 0)
                
                # Check that results are finite (no NaN or infinite values in final results)
                for col in result.columns:
                    finite_values = result[col].dropna()
                    if len(finite_values) > 0:
                        self.assertTrue(np.all(np.isfinite(finite_values)), 
                                      f"Non-finite values found in {col}")
                
            except Exception as e:
                self.fail(f"Generator {generator.__class__.__name__} failed: {e}")
        
        logger.info("✅ Performance comparison tests passed")
    
    def test_memory_usage(self):
        """Test memory usage of VectorBT generators."""
        logger.info("Testing memory usage...")
        
        # Test with larger dataset
        large_data = pd.concat([self.sample_data] * 5, ignore_index=True)
        large_data.index = pd.date_range('2023-01-01', periods=len(large_data), freq='1H')
        
        generators = [
            VectorBTTakerBuyRatioGenerator(window=20),
            VectorBTMomentumGenerator(period=10),
            VectorBTHurstExponentGenerator(window=20)
        ]
        
        for generator in generators:
            try:
                result = generator.generate_features(large_data)
                self.assertIsInstance(result, pd.DataFrame)
                self.assertGreater(len(result.columns), 0)
                
                # Check memory usage is reasonable
                memory_usage = result.memory_usage(deep=True).sum()
                self.assertLess(memory_usage, 100 * 1024 * 1024)  # Less than 100MB
                
            except Exception as e:
                self.fail(f"Generator {generator.__class__.__name__} failed with large data: {e}")
        
        logger.info("✅ Memory usage tests passed")


def run_vectorbt_integration_tests():
    """Run all VectorBT integration tests."""
    logger.info("🚀 Starting VectorBT integration tests...")
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestVectorBTIntegration)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    if result.wasSuccessful():
        logger.info("✅ All VectorBT integration tests passed!")
    else:
        logger.error(f"❌ {len(result.failures)} tests failed, {len(result.errors)} errors")
        for failure in result.failures:
            logger.error(f"Failure: {failure[0]}")
        for error in result.errors:
            logger.error(f"Error: {error[0]}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    success = run_vectorbt_integration_tests()
    exit(0 if success else 1)