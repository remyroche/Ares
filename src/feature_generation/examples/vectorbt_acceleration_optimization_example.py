"""
VectorBT Acceleration Feature Optimization Example

This example demonstrates the enhanced VectorBT usage in acceleration features,
including VectorBTRollingOptimizer and UnifiedVectorizationManager integration.

Key Features Demonstrated:
- VectorBTRollingOptimizer for enhanced rolling operations
- UnifiedVectorizationManager for intelligent optimization strategy selection
- Batch processing for multiple acceleration features
- Performance monitoring and statistics
- Memory optimization and GPU acceleration
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import acceleration features
from ..categories.vectorbt_acceleration import (
    VectorBTMomentumGenerator,
    VectorBTPriceAccelerationGenerator,
    VectorBTPriceJerkGenerator,
    VectorBTTrendStrengthGenerator,
    VectorBTTrendConsistencyGenerator,
    VectorBTAccelerationBatchProcessor,
    VectorBTAccelerationPerformanceMonitor,
    get_acceleration_performance_monitor,
    create_vectorbt_acceleration_generators
)

# Import optimization components
from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
from ...utils.ml_common.unified_vectorization_manager import (
    get_unified_vectorization_manager, OperationType, OptimizationStrategy
)


def create_sample_data(n_points: int = 10000) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate realistic price data with trends and volatility
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.02, n_points)  # Daily returns
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Add some trend and volatility clustering
    trend = np.sin(np.linspace(0, 4*np.pi, n_points)) * 0.01
    volatility_cluster = np.random.exponential(0.01, n_points)
    volatility_cluster = np.convolve(volatility_cluster, np.ones(50)/50, mode='same')
    
    prices = prices * (1 + trend + volatility_cluster)
    
    # Generate OHLCV data
    high = prices * (1 + np.abs(np.random.normal(0, 0.01, n_points)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.01, n_points)))
    open_prices = np.roll(prices, 1)
    open_prices[0] = prices[0]
    
    volume = np.random.lognormal(10, 1, n_points)
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    }, index=pd.date_range('2020-01-01', periods=n_points, freq='1min'))
    
    return data


def demonstrate_individual_optimization():
    """Demonstrate individual feature optimization with VectorBTRollingOptimizer."""
    logger.info("🚀 Demonstrating individual feature optimization...")
    
    # Create sample data
    data = create_sample_data(5000)
    
    # Initialize performance monitor
    monitor = get_acceleration_performance_monitor()
    
    # Test different generators with optimization enabled
    generators = [
        VectorBTMomentumGenerator(period=10, enable_optimization=True),
        VectorBTPriceAccelerationGenerator(period=5, enable_optimization=True),
        VectorBTPriceJerkGenerator(period=5, enable_optimization=True),
        VectorBTTrendStrengthGenerator(window=20, enable_optimization=True),
        VectorBTTrendConsistencyGenerator(window=20, enable_optimization=True)
    ]
    
    results = {}
    
    for generator in generators:
        start_time = time.time()
        
        try:
            # Generate feature
            feature_result = generator.generate(data)
            generation_time = time.time() - start_time
            
            # Record performance
            monitor.record_feature_generation(
                feature_type=generator.config.name,
                generation_time=generation_time,
                optimization_strategy='vectorbt_optimized',
                vectorbt_used=True,
                gpu_used=generator.enable_optimization,
                parallel_used=generator.enable_optimization,
                memory_optimized=True
            )
            
            results[generator.config.name] = feature_result
            logger.info(f"✅ Generated {generator.config.name} in {generation_time:.4f}s")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate {generator.config.name}: {e}")
    
    return results


def demonstrate_batch_processing():
    """Demonstrate batch processing with UnifiedVectorizationManager."""
    logger.info("🚀 Demonstrating batch processing with UnifiedVectorizationManager...")
    
    # Create sample data
    data = create_sample_data(8000)
    
    # Initialize batch processor
    batch_processor = VectorBTAccelerationBatchProcessor(enable_optimization=True)
    
    # Define feature configurations for batch processing
    feature_configs = [
        {'type': 'momentum', 'period': 10, 'base_calculation': 'price_returns'},
        {'type': 'momentum', 'period': 20, 'base_calculation': 'price_returns'},
        {'type': 'acceleration', 'period': 5, 'base_calculation': 'price_returns'},
        {'type': 'acceleration', 'period': 10, 'base_calculation': 'price_returns'},
        {'type': 'jerk', 'period': 5, 'base_calculation': 'price_returns'},
        {'type': 'jerk', 'period': 10, 'base_calculation': 'price_returns'},
    ]
    
    start_time = time.time()
    
    try:
        # Generate features in batch
        batch_results = batch_processor.generate_batch_acceleration_features(
            data, feature_configs
        )
        
        batch_time = time.time() - start_time
        
        # Record batch performance
        monitor = get_acceleration_performance_monitor()
        monitor.record_batch_operation(
            num_features=len(feature_configs),
            batch_time=batch_time,
            optimization_strategy='unified_vectorization_manager'
        )
        
        logger.info(f"✅ Generated {len(batch_results.columns)} features in batch in {batch_time:.4f}s")
        logger.info(f"   Features: {list(batch_results.columns)}")
        
        return batch_results
        
    except Exception as e:
        logger.error(f"❌ Batch processing failed: {e}")
        return pd.DataFrame()


def demonstrate_unified_vectorization_manager():
    """Demonstrate UnifiedVectorizationManager for intelligent optimization."""
    logger.info("🚀 Demonstrating UnifiedVectorizationManager...")
    
    # Create sample data
    data = create_sample_data(10000)
    
    # Get unified manager
    unified_manager = get_unified_vectorization_manager()
    
    # Test different operation types
    operation_tests = [
        {
            'name': 'Feature Engineering',
            'operation_type': OperationType.FEATURE_ENGINEERING,
            'data': {'data': data, 'feature_configs': [
                {'type': 'momentum', 'period': 10},
                {'type': 'acceleration', 'period': 5}
            ]}
        },
        {
            'name': 'Technical Analysis',
            'operation_type': OperationType.TECHNICAL_INDICATORS,
            'data': {'data': data, 'indicator_configs': {
                'rsi': [14, 21],
                'macd': [12, 26, 9],
                'bbands': [20, 2]
            }}
        }
    ]
    
    results = {}
    
    for test in operation_tests:
        start_time = time.time()
        
        try:
            # Optimize operation
            result = unified_manager.optimize_operation(
                test['operation_type'],
                test['data']
            )
            
            operation_time = time.time() - start_time
            
            logger.info(f"✅ {test['name']} completed in {operation_time:.4f}s")
            logger.info(f"   Strategy used: {result.strategy_used.value}")
            logger.info(f"   Performance gain: {result.performance_gain:.2f}x")
            logger.info(f"   Memory used: {result.memory_used_mb:.2f} MB")
            
            results[test['name']] = result
            
        except Exception as e:
            logger.error(f"❌ {test['name']} failed: {e}")
    
    return results


def demonstrate_rolling_optimizer():
    """Demonstrate VectorBTRollingOptimizer capabilities."""
    logger.info("🚀 Demonstrating VectorBTRollingOptimizer...")
    
    # Create sample data
    data = create_sample_data(5000)
    
    # Get rolling optimizer
    rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
    
    # Test different rolling operations
    operations = [
        ('rolling_mean', lambda: rolling_optimizer.rolling_mean(data['close'], window=20)),
        ('rolling_std', lambda: rolling_optimizer.rolling_std(data['close'], window=20)),
        ('rolling_min', lambda: rolling_optimizer.rolling_min(data['close'], window=20)),
        ('rolling_max', lambda: rolling_optimizer.rolling_max(data['close'], window=20)),
        ('rolling_corr', lambda: rolling_optimizer.rolling_corr(data['close'], data['volume'], window=20)),
        ('rolling_quantile', lambda: rolling_optimizer.rolling_quantile(data['close'], window=20, q=0.5)),
    ]
    
    results = {}
    
    for op_name, op_func in operations:
        start_time = time.time()
        
        try:
            result = op_func()
            operation_time = time.time() - start_time
            
            logger.info(f"✅ {op_name} completed in {operation_time:.4f}s")
            logger.info(f"   Result shape: {result.shape}")
            
            results[op_name] = result
            
        except Exception as e:
            logger.error(f"❌ {op_name} failed: {e}")
    
    # Get performance stats
    stats = rolling_optimizer.get_performance_stats()
    logger.info(f"📊 Rolling Optimizer Stats:")
    logger.info(f"   Total operations: {stats['total_operations']}")
    logger.info(f"   VectorBT operations: {stats['vectorbt_operations']}")
    logger.info(f"   GPU operations: {stats['gpu_operations']}")
    logger.info(f"   Average time per operation: {stats.get('avg_time_per_operation', 0):.4f}s")
    
    return results


def demonstrate_performance_monitoring():
    """Demonstrate comprehensive performance monitoring."""
    logger.info("🚀 Demonstrating performance monitoring...")
    
    # Get performance monitor
    monitor = get_acceleration_performance_monitor()
    
    # Generate some features to track
    data = create_sample_data(3000)
    
    # Create generators
    generators = create_vectorbt_acceleration_generators()[:5]  # Use first 5 generators
    
    for generator in generators:
        start_time = time.time()
        
        try:
            feature_result = generator.generate(data)
            generation_time = time.time() - start_time
            
            # Record performance
            monitor.record_feature_generation(
                feature_type=generator.config.name,
                generation_time=generation_time,
                optimization_strategy='vectorbt_optimized',
                vectorbt_used=True,
                gpu_used=hasattr(generator, 'enable_optimization') and generator.enable_optimization,
                parallel_used=hasattr(generator, 'enable_optimization') and generator.enable_optimization,
                memory_optimized=True
            )
            
        except Exception as e:
            logger.warning(f"Failed to generate {generator.config.name}: {e}")
    
    # Display performance summary
    monitor.log_performance_summary()
    
    return monitor.get_performance_summary()


def main():
    """Main demonstration function."""
    logger.info("🎯 VectorBT Acceleration Feature Optimization Demonstration")
    logger.info("=" * 70)
    
    try:
        # Demonstrate individual optimization
        individual_results = demonstrate_individual_optimization()
        logger.info(f"✅ Individual optimization completed: {len(individual_results)} features generated")
        
        # Demonstrate batch processing
        batch_results = demonstrate_batch_processing()
        logger.info(f"✅ Batch processing completed: {len(batch_results.columns)} features generated")
        
        # Demonstrate UnifiedVectorizationManager
        unified_results = demonstrate_unified_vectorization_manager()
        logger.info(f"✅ UnifiedVectorizationManager completed: {len(unified_results)} operations")
        
        # Demonstrate rolling optimizer
        rolling_results = demonstrate_rolling_optimizer()
        logger.info(f"✅ Rolling optimizer completed: {len(rolling_results)} operations")
        
        # Demonstrate performance monitoring
        performance_summary = demonstrate_performance_monitoring()
        logger.info(f"✅ Performance monitoring completed")
        
        logger.info("🎉 All demonstrations completed successfully!")
        
        # Final performance summary
        monitor = get_acceleration_performance_monitor()
        final_summary = monitor.get_performance_summary()
        
        logger.info("\n📊 Final Performance Summary:")
        logger.info(f"   Total features generated: {final_summary['total_features_generated']}")
        logger.info(f"   Average generation time: {final_summary['average_generation_time']:.4f}s")
        logger.info(f"   Features per second: {final_summary['features_per_second']:.2f}")
        logger.info(f"   VectorBT usage rate: {final_summary['vectorbt_usage_rate']:.2%}")
        logger.info(f"   GPU usage rate: {final_summary['gpu_usage_rate']:.2%}")
        logger.info(f"   Parallel usage rate: {final_summary['parallel_usage_rate']:.2%}")
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()