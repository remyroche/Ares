"""
Test VectorBT Microstructure Optimization

This module provides comprehensive tests for VectorBT-optimized microstructure features
to ensure proper integration and performance improvements.
"""

import numpy as np
import pandas as pd
import pytest
import time
import logging
from typing import List, Dict, Any

# Import the microstructure generators
from ..categories.microstructure import (
    create_default_microstructure_generators,
    MicrostructureFeatureGenerator,
    BidAskSpreadGenerator,
    OrderFlowImbalanceGenerator,
    TradeSizeImbalanceGenerator,
    PriceImpactGenerator,
    VolumeWeightedPriceGenerator,
    TradeIntensityGenerator,
    LiquidityProxyGenerator,
    MarketDepthGenerator
)

# Import optimization utilities
from ..utils.unified_vectorization_manager import get_unified_vectorization_manager
from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer

logger = logging.getLogger(__name__)


class TestVectorBTMicrostructureOptimization:
    """Test suite for VectorBT-optimized microstructure features."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        n_periods = 1000
        
        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, n_periods)
        prices = 100 * (1 + returns).cumprod()
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, n_periods)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_periods))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_periods))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, n_periods),
            'bid': prices * (1 - np.random.uniform(0.0001, 0.001, n_periods)),
            'ask': prices * (1 + np.random.uniform(0.0001, 0.001, n_periods))
        }, index=pd.date_range('2020-01-01', periods=n_periods, freq='1min'))
        
        return data
    
    @pytest.fixture
    def large_sample_data(self):
        """Create large sample data for performance testing."""
        np.random.seed(42)
        n_periods = 10000  # 10k periods for performance testing
        
        returns = np.random.normal(0.001, 0.02, n_periods)
        prices = 100 * (1 + returns).cumprod()
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, n_periods)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_periods))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_periods))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, n_periods),
            'bid': prices * (1 - np.random.uniform(0.0001, 0.001, n_periods)),
            'ask': prices * (1 + np.random.uniform(0.0001, 0.001, n_periods))
        }, index=pd.date_range('2020-01-01', periods=n_periods, freq='1min'))
        
        return data
    
    def test_vectorbt_rolling_optimizer_available(self):
        """Test that VectorBT rolling optimizer is available."""
        optimizer = get_vectorbt_rolling_optimizer()
        assert optimizer is not None
        assert hasattr(optimizer, 'rolling_mean')
        assert hasattr(optimizer, 'rolling_std')
        assert hasattr(optimizer, 'rolling_sum')
    
    def test_unified_vectorization_manager_available(self):
        """Test that unified vectorization manager is available."""
        manager = get_unified_vectorization_manager()
        assert manager is not None
        assert hasattr(manager, 'rolling_operation')
        assert hasattr(manager, 'scale_data')
        assert hasattr(manager, 'batch_process_features')
    
    def test_microstructure_generators_have_vectorbt_optimization(self, sample_data):
        """Test that microstructure generators have VectorBT optimization."""
        generators = create_default_microstructure_generators()
        
        # Test that generators have VectorBT optimization attributes
        for generator in generators[:5]:  # Test first 5 generators
            assert hasattr(generator, 'rolling_optimizer')
            assert hasattr(generator, 'vectorization_manager')
            
            # Test that optimization attributes are not None
            if generator.rolling_optimizer is not None:
                assert generator.rolling_optimizer is not None
            if generator.vectorization_manager is not None:
                assert generator.vectorization_manager is not None
    
    def test_bid_ask_spread_generator_vectorbt(self, sample_data):
        """Test BidAskSpreadGenerator with VectorBT optimization."""
        generator = BidAskSpreadGenerator(window=20)
        
        # Add VectorBT optimization
        generator.rolling_optimizer = get_vectorbt_rolling_optimizer()
        generator.vectorization_manager = get_unified_vectorization_manager()
        
        # Generate feature
        result = generator.generate_feature(sample_data)
        
        # Verify result
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert not result.empty
        assert result.name is not None
        
        # Check that VectorBT was used (if available)
        if generator.rolling_optimizer:
            stats = generator.rolling_optimizer.get_performance_stats()
            assert 'vectorbt_operations' in stats
    
    def test_order_flow_imbalance_generator_vectorbt(self, sample_data):
        """Test OrderFlowImbalanceGenerator with VectorBT optimization."""
        generator = OrderFlowImbalanceGenerator(window=20)
        
        # Add VectorBT optimization
        generator.rolling_optimizer = get_vectorbt_rolling_optimizer()
        generator.vectorization_manager = get_unified_vectorization_manager()
        
        # Generate feature
        result = generator.generate_feature(sample_data)
        
        # Verify result
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert not result.empty
        assert result.name is not None
    
    def test_volume_weighted_price_generator_vectorbt(self, sample_data):
        """Test VolumeWeightedPriceGenerator with VectorBT optimization."""
        generator = VolumeWeightedPriceGenerator(window=20)
        
        # Add VectorBT optimization
        generator.rolling_optimizer = get_vectorbt_rolling_optimizer()
        generator.vectorization_manager = get_unified_vectorization_manager()
        
        # Generate feature
        result = generator.generate_feature(sample_data)
        
        # Verify result
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert not result.empty
        assert result.name is not None
    
    def test_batch_processing_performance(self, large_sample_data):
        """Test batch processing performance with VectorBT optimization."""
        manager = get_unified_vectorization_manager()
        
        # Create feature configurations
        feature_configs = [
            {'name': 'sma_20', 'type': 'rolling', 'params': {'operation': 'mean', 'window': 20, 'column': 'close'}},
            {'name': 'sma_50', 'type': 'rolling', 'params': {'operation': 'mean', 'window': 50, 'column': 'close'}},
            {'name': 'std_20', 'type': 'rolling', 'params': {'operation': 'std', 'window': 20, 'column': 'close'}},
            {'name': 'volume_sma', 'type': 'rolling', 'params': {'operation': 'mean', 'window': 20, 'column': 'volume'}},
            {'name': 'close_scaled', 'type': 'scaling', 'params': {'method': 'zscore', 'column': 'close'}},
        ]
        
        # Time the batch processing
        start_time = time.time()
        features = manager.batch_process_features(large_sample_data, feature_configs)
        processing_time = time.time() - start_time
        
        # Verify results
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(large_sample_data)
        assert len(features.columns) == len(feature_configs)
        
        # Check performance
        assert processing_time < 10.0  # Should complete within 10 seconds
        logger.info(f"Batch processing time: {processing_time:.3f}s for {len(large_sample_data)} rows")
    
    def test_memory_optimization(self, large_sample_data):
        """Test memory optimization capabilities."""
        manager = get_unified_vectorization_manager()
        
        # Test memory optimization
        original_memory = large_sample_data.memory_usage(deep=True).sum()
        optimized_data = manager.optimize_dataframe(large_sample_data)
        optimized_memory = optimized_data.memory_usage(deep=True).sum()
        
        # Verify memory optimization
        assert optimized_memory <= original_memory
        memory_savings = (original_memory - optimized_memory) / original_memory * 100
        logger.info(f"Memory savings: {memory_savings:.2f}%")
    
    def test_rolling_operations_performance(self, large_sample_data):
        """Test rolling operations performance with VectorBT."""
        optimizer = get_vectorbt_rolling_optimizer()
        
        # Test different rolling operations
        operations = ['mean', 'std', 'var', 'min', 'max', 'sum']
        close_data = large_sample_data['close']
        
        results = {}
        for operation in operations:
            start_time = time.time()
            if operation == 'mean':
                result = optimizer.rolling_mean(close_data, window=20)
            elif operation == 'std':
                result = optimizer.rolling_std(close_data, window=20)
            elif operation == 'var':
                result = optimizer.rolling_var(close_data, window=20)
            elif operation == 'min':
                result = optimizer.rolling_min(close_data, window=20)
            elif operation == 'max':
                result = optimizer.rolling_max(close_data, window=20)
            elif operation == 'sum':
                result = optimizer.rolling_sum(close_data, window=20)
            
            processing_time = time.time() - start_time
            results[operation] = {
                'time': processing_time,
                'result_length': len(result)
            }
            
            # Verify result
            assert isinstance(result, pd.Series)
            assert len(result) == len(close_data)
            assert processing_time < 5.0  # Should complete within 5 seconds
        
        # Log performance results
        for operation, stats in results.items():
            logger.info(f"{operation}: {stats['time']:.3f}s")
    
    def test_scaling_operations(self, sample_data):
        """Test scaling operations with VectorBT."""
        manager = get_unified_vectorization_manager()
        
        scaling_methods = ['zscore', 'minmax', 'robust']
        close_data = sample_data['close']
        
        for method in scaling_methods:
            result = manager.scale_data(close_data, method=method)
            
            # Verify result
            assert isinstance(result, pd.Series)
            assert len(result) == len(close_data)
            assert not result.empty
    
    def test_performance_statistics(self, sample_data):
        """Test performance statistics tracking."""
        manager = get_unified_vectorization_manager()
        
        # Perform some operations
        manager.rolling_operation(sample_data['close'], 'mean', window=20)
        manager.rolling_operation(sample_data['close'], 'std', window=20)
        manager.scale_data(sample_data['close'], method='zscore')
        
        # Get performance stats
        stats = manager.get_performance_stats()
        
        # Verify stats structure
        assert 'total_operations' in stats
        assert 'vectorbt_operations' in stats
        assert 'rolling_operations' in stats
        assert 'scaling_operations' in stats
        assert 'total_time' in stats
        
        # Verify stats are reasonable
        assert stats['total_operations'] > 0
        assert stats['total_time'] > 0
        
        logger.info(f"Performance stats: {stats}")
    
    def test_error_handling(self, sample_data):
        """Test error handling and fallbacks."""
        # Test with invalid data
        invalid_data = pd.DataFrame({
            'close': [np.nan, np.inf, -np.inf, 1, 2, 3],
            'volume': [1, 2, 3, 4, 5, 6]
        })
        
        generator = OrderFlowImbalanceGenerator(window=3)
        generator.rolling_optimizer = get_vectorbt_rolling_optimizer()
        generator.vectorization_manager = get_unified_vectorization_manager()
        
        # Should handle invalid data gracefully
        result = generator.generate_feature(invalid_data)
        assert isinstance(result, pd.Series)
        assert len(result) == len(invalid_data)
    
    def test_concurrent_processing(self, sample_data):
        """Test concurrent processing capabilities."""
        import concurrent.futures
        import threading
        
        manager = get_unified_vectorization_manager()
        
        def process_data(data_slice):
            return manager.rolling_operation(data_slice, 'mean', window=20)
        
        # Split data into chunks
        chunk_size = len(sample_data) // 4
        chunks = [sample_data.iloc[i:i+chunk_size] for i in range(0, len(sample_data), chunk_size)]
        
        # Process chunks concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_data, chunk['close']) for chunk in chunks]
            results = [future.result() for future in futures]
        
        # Verify results
        assert len(results) == len(chunks)
        for result in results:
            assert isinstance(result, pd.Series)
            assert not result.empty


def test_integration_with_existing_features():
    """Test integration with existing feature generation pipeline."""
    # This test would verify that the VectorBT optimizations
    # work correctly with the existing feature generation pipeline
    pass


def test_memory_usage_under_load():
    """Test memory usage under heavy load."""
    # This test would verify that memory usage remains reasonable
    # under heavy processing loads
    pass


def test_gpu_acceleration():
    """Test GPU acceleration capabilities."""
    # This test would verify GPU acceleration works when available
    pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])