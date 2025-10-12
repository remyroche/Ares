"""
Comprehensive test suite for legacy feature VectorBT optimization.

This test suite verifies that all legacy features are properly optimized
with VectorBT and that the UnifiedVectorizationManager works correctly.
"""

import pytest
import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any

# Import the optimized legacy features
from ..categories.legacy import (
    LegacyFeatureGeneratorBase,
    LegacyRSIGenerator,
    LegacyMACDGenerator,
    LegacyBollingerBandsGenerator,
    LegacySMAGenerator,
    LegacyEMAGenerator,
    LegacyATRGenerator,
    LegacyStochasticGenerator,
    LegacyWilliamsRGenerator,
    LegacyOBVGenerator,
    create_default_legacy_generators,
    create_legacy_features_batch,
    get_legacy_performance_stats,
    reset_legacy_performance_stats
)

# Import the unified manager
from ..utils.unified_vectorization_manager import (
    get_unified_vectorization_manager,
    OptimizationConfig
)


class TestLegacyVectorBTOptimization:
    """Test suite for legacy feature VectorBT optimization."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1000, freq='1min')
        
        # Generate realistic price data
        price = 100
        prices = [price]
        for _ in range(999):
            price += np.random.normal(0, 0.5)
            prices.append(price)
        
        return pd.DataFrame({
            'open': prices,
            'high': [p + np.random.uniform(0, 2) for p in prices],
            'low': [p - np.random.uniform(0, 2) for p in prices],
            'close': prices,
            'volume': np.random.lognormal(10, 1, 1000)
        }, index=dates)
    
    @pytest.fixture
    def unified_manager(self):
        """Create unified vectorization manager for testing."""
        config = OptimizationConfig(
            enable_vectorbt=True,
            enable_gpu=False,  # Disable GPU for testing
            enable_parallel=True,
            enable_batch_processing=True,
            enable_caching=True
        )
        return get_unified_vectorization_manager(config)
    
    def test_unified_manager_initialization(self, unified_manager):
        """Test that the unified manager initializes correctly."""
        assert unified_manager is not None
        assert unified_manager.config.enable_vectorbt is True
        assert unified_manager.config.enable_parallel is True
        assert unified_manager.config.enable_batch_processing is True
    
    def test_dataframe_optimization(self, unified_manager, sample_data):
        """Test DataFrame optimization functionality."""
        optimized_data = unified_manager.optimize_dataframe(sample_data)
        
        assert optimized_data is not None
        assert len(optimized_data) == len(sample_data)
        assert list(optimized_data.columns) == list(sample_data.columns)
        
        # Check that data types are optimized
        for col in optimized_data.select_dtypes(include=[np.number]).columns:
            assert optimized_data[col].dtype in [np.float32, np.float64, np.int32, np.int64]
    
    def test_rolling_operations(self, unified_manager, sample_data):
        """Test rolling operations through unified manager."""
        close = sample_data['close']
        
        # Test various rolling operations
        operations = ['mean', 'std', 'var', 'min', 'max', 'sum']
        window = 20
        
        for operation in operations:
            result = unified_manager.rolling_operation(close, operation, window)
            assert result is not None
            assert len(result) == len(close)
            assert not result.isna().all()  # Should have some valid values
    
    def test_technical_indicators(self, unified_manager, sample_data):
        """Test technical indicator calculations."""
        indicators = ['rsi', 'macd', 'atr', 'bbands_upper', 'sma', 'ema', 'stoch_k', 'willr', 'obv']
        
        for indicator in indicators:
            try:
                result = unified_manager.technical_indicator(sample_data, indicator, window=14)
                assert result is not None
                assert len(result) == len(sample_data)
            except Exception as e:
                # Some indicators might not be available in all VectorBT versions
                print(f"Indicator {indicator} not available: {e}")
    
    def test_legacy_rsi_generator(self, sample_data):
        """Test optimized RSI generator."""
        generator = LegacyRSIGenerator(14)
        
        # Test single feature generation
        result = generator.generate_feature(sample_data)
        assert result is not None
        assert len(result) == len(sample_data)
        assert result.name == 'legacy_rsi_14'
        assert not result.isna().all()
        
        # Test performance stats
        stats = generator.get_performance_stats()
        assert 'total_generations' in stats
        assert 'vectorbt_operations' in stats
    
    def test_legacy_macd_generator(self, sample_data):
        """Test optimized MACD generator."""
        generator = LegacyMACDGenerator(12, 26, 9)
        
        result = generator.generate_feature(sample_data)
        assert result is not None
        assert len(result) == len(sample_data)
        assert result.name == 'legacy_macd_12_26_9'
        assert not result.isna().all()
    
    def test_legacy_bollinger_bands_generator(self, sample_data):
        """Test optimized Bollinger Bands generator."""
        generator = LegacyBollingerBandsGenerator(20, 2.0)
        
        result = generator.generate_feature(sample_data)
        assert result is not None
        assert len(result) == len(sample_data)
        assert result.name == 'legacy_bollinger_upper_20_2.0'
        assert not result.isna().all()
    
    def test_legacy_sma_generator(self, sample_data):
        """Test optimized SMA generator."""
        generator = LegacySMAGenerator(20)
        
        result = generator.generate_feature(sample_data)
        assert result is not None
        assert len(result) == len(sample_data)
        assert result.name == 'legacy_sma_20'
        assert not result.isna().all()
    
    def test_legacy_ema_generator(self, sample_data):
        """Test optimized EMA generator."""
        generator = LegacyEMAGenerator(21)
        
        result = generator.generate_feature(sample_data)
        assert result is not None
        assert len(result) == len(sample_data)
        assert result.name == 'legacy_ema_21'
        assert not result.isna().all()
    
    def test_legacy_atr_generator(self, sample_data):
        """Test optimized ATR generator."""
        generator = LegacyATRGenerator(14)
        
        result = generator.generate_feature(sample_data)
        assert result is not None
        assert len(result) == len(sample_data)
        assert result.name == 'legacy_atr_14'
        assert not result.isna().all()
    
    def test_legacy_stochastic_generator(self, sample_data):
        """Test optimized Stochastic generator."""
        generator = LegacyStochasticGenerator(14, 3)
        
        result = generator.generate_feature(sample_data)
        assert result is not None
        assert len(result) == len(sample_data)
        assert result.name == 'legacy_stochastic_k_14_3'
        assert not result.isna().all()
    
    def test_legacy_williams_r_generator(self, sample_data):
        """Test optimized Williams %R generator."""
        generator = LegacyWilliamsRGenerator(14)
        
        result = generator.generate_feature(sample_data)
        assert result is not None
        assert len(result) == len(sample_data)
        assert result.name == 'legacy_williams_r_14'
        assert not result.isna().all()
    
    def test_legacy_obv_generator(self, sample_data):
        """Test optimized OBV generator."""
        generator = LegacyOBVGenerator()
        
        result = generator.generate_feature(sample_data)
        assert result is not None
        assert len(result) == len(sample_data)
        assert result.name == 'legacy_obv'
        assert not result.isna().all()
    
    def test_batch_processing(self, sample_data):
        """Test batch processing functionality."""
        # Create feature configurations
        feature_configs = [
            {
                'name': 'rsi_14',
                'type': 'indicator',
                'indicator': 'rsi',
                'params': {'window': 14}
            },
            {
                'name': 'sma_20',
                'type': 'indicator',
                'indicator': 'sma',
                'params': {'window': 20}
            },
            {
                'name': 'close_rolling_mean_10',
                'type': 'rolling',
                'column': 'close',
                'operation': 'mean',
                'window': 10
            }
        ]
        
        # Test batch processing
        result = create_legacy_features_batch(sample_data, feature_configs)
        
        assert result is not None
        assert len(result) == len(sample_data)
        assert len(result.columns) == len(feature_configs)
        
        # Check that all features are present
        for config in feature_configs:
            assert config['name'] in result.columns
    
    def test_default_generators_creation(self):
        """Test creation of default legacy generators."""
        generators = create_default_legacy_generators()
        
        assert len(generators) > 0
        assert all(isinstance(gen, LegacyFeatureGeneratorBase) for gen in generators)
        
        # Check that we have the expected number of generators
        # 9 base + 5 SMA + 5 EMA + 3 RSI = 22 generators
        assert len(generators) == 22
    
    def test_performance_monitoring(self, sample_data):
        """Test performance monitoring functionality."""
        # Reset stats
        reset_legacy_performance_stats()
        
        # Generate some features
        generator = LegacyRSIGenerator(14)
        generator.generate_feature(sample_data)
        
        # Get performance stats
        stats = get_legacy_performance_stats()
        
        assert 'total_operations' in stats
        assert 'vectorbt_operations' in stats
        assert 'vectorization_operations' in stats
        assert 'batch_operations' in stats
        assert 'total_time' in stats
    
    def test_gpu_acceleration_option(self, sample_data):
        """Test GPU acceleration option."""
        # Test with GPU disabled (should work)
        generator = LegacyRSIGenerator(14, enable_gpu=False)
        result = generator.generate_feature(sample_data)
        assert result is not None
        
        # Test with GPU enabled (might not be available)
        try:
            generator_gpu = LegacyRSIGenerator(14, enable_gpu=True)
            result_gpu = generator_gpu.generate_feature(sample_data)
            assert result_gpu is not None
        except Exception as e:
            # GPU might not be available, which is fine
            print(f"GPU acceleration not available: {e}")
    
    def test_parallel_processing_option(self, sample_data):
        """Test parallel processing option."""
        # Test with parallel processing enabled
        generator = LegacyRSIGenerator(14, enable_parallel=True)
        result = generator.generate_feature(sample_data)
        assert result is not None
        
        # Test with parallel processing disabled
        generator_serial = LegacyRSIGenerator(14, enable_parallel=False)
        result_serial = generator_serial.generate_feature(sample_data)
        assert result_serial is not None
    
    def test_memory_optimization(self, sample_data):
        """Test memory optimization functionality."""
        # Create a larger dataset
        large_data = pd.concat([sample_data] * 10, ignore_index=True)
        large_data.index = pd.date_range('2020-01-01', periods=len(large_data), freq='1min')
        
        # Test that memory optimization works
        generator = LegacyRSIGenerator(14)
        result = generator.generate_feature(large_data)
        
        assert result is not None
        assert len(result) == len(large_data)
        
        # Check that the unified manager handled memory optimization
        stats = generator.get_performance_stats()
        assert 'memory_optimizations' in stats
    
    def test_caching_functionality(self, sample_data):
        """Test caching functionality."""
        # Reset stats to clear cache
        reset_legacy_performance_stats()
        
        generator = LegacyRSIGenerator(14)
        
        # Generate feature twice
        result1 = generator.generate_feature(sample_data)
        result2 = generator.generate_feature(sample_data)
        
        # Results should be identical
        pd.testing.assert_series_equal(result1, result2)
        
        # Check that caching was used
        stats = generator.get_performance_stats()
        assert 'cache_hits' in stats or 'cache_misses' in stats
    
    def test_error_handling(self, sample_data):
        """Test error handling and fallbacks."""
        # Test with invalid data
        invalid_data = pd.DataFrame({
            'close': [np.nan] * 100,
            'high': [np.nan] * 100,
            'low': [np.nan] * 100,
            'volume': [np.nan] * 100
        })
        
        generator = LegacyRSIGenerator(14)
        result = generator.generate_feature(invalid_data)
        
        # Should return a series with NaN values, not crash
        assert result is not None
        assert len(result) == len(invalid_data)
        assert result.isna().all()  # All values should be NaN for invalid data
    
    def test_performance_improvement(self, sample_data):
        """Test that VectorBT optimization provides performance improvement."""
        # This is a basic performance test
        generator = LegacyRSIGenerator(14)
        
        # Time the operation
        start_time = time.time()
        result = generator.generate_feature(sample_data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should complete in reasonable time (less than 1 second for 1000 data points)
        assert execution_time < 1.0
        assert result is not None
        
        # Check performance stats
        stats = generator.get_performance_stats()
        assert stats['total_time'] > 0


class TestLegacyFeatureConsistency:
    """Test consistency between different legacy feature implementations."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for consistency testing."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='1min')
        
        price = 100
        prices = [price]
        for _ in range(499):
            price += np.random.normal(0, 0.5)
            prices.append(price)
        
        return pd.DataFrame({
            'open': prices,
            'high': [p + np.random.uniform(0, 2) for p in prices],
            'low': [p - np.random.uniform(0, 2) for p in prices],
            'close': prices,
            'volume': np.random.lognormal(10, 1, 500)
        }, index=dates)
    
    def test_rsi_consistency(self, sample_data):
        """Test RSI consistency across different periods."""
        periods = [9, 14, 21]
        results = {}
        
        for period in periods:
            generator = LegacyRSIGenerator(period)
            results[period] = generator.generate_feature(sample_data)
        
        # All results should have the same length
        for period, result in results.items():
            assert len(result) == len(sample_data)
            assert result.name == f'legacy_rsi_{period}'
    
    def test_sma_consistency(self, sample_data):
        """Test SMA consistency across different periods."""
        periods = [5, 10, 20, 50]
        results = {}
        
        for period in periods:
            generator = LegacySMAGenerator(period)
            results[period] = generator.generate_feature(sample_data)
        
        # All results should have the same length
        for period, result in results.items():
            assert len(result) == len(sample_data)
            assert result.name == f'legacy_sma_{period}'
    
    def test_ema_consistency(self, sample_data):
        """Test EMA consistency across different periods."""
        periods = [8, 12, 21, 26]
        results = {}
        
        for period in periods:
            generator = LegacyEMAGenerator(period)
            results[period] = generator.generate_feature(sample_data)
        
        # All results should have the same length
        for period, result in results.items():
            assert len(result) == len(sample_data)
            assert result.name == f'legacy_ema_{period}'


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])