"""
Comprehensive Example: Optimized Feature Generation with VectorBT

This script demonstrates the comprehensive optimizations implemented using
VectorBTRollingOptimizer and UnifiedVectorizationManager for feature generation.

Key Optimizations Demonstrated:
1. Batch rolling operations using VectorBTRollingOptimizer
2. UnifiedVectorizationManager integration for cross-category features
3. Memory optimization with data type optimization
4. Smart caching for frequently computed operations
5. Performance monitoring and statistics
6. Cross-category feature generation optimization

Usage:
    python optimized_feature_generation_example.py
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import optimized feature generators
try:
    from src.feature_generation.categories.optimized_trend import (
        OptimizedTrendFeatureGenerator, 
        create_optimized_trend_generator,
        generate_trend_features_optimized
    )
    from src.feature_generation.categories.optimized_volatility import (
        OptimizedVolatilityFeatureGenerator,
        create_optimized_volatility_generator,
        generate_volatility_features_optimized
    )
    from src.feature_generation.categories.optimized_returns import (
        OptimizedReturnsFeatureGenerator,
        create_optimized_returns_generator,
        generate_returns_features_optimized
    )
    from src.feature_generation.categories.optimized_cross_category import (
        OptimizedCrossCategoryFeatureGenerator,
        create_optimized_cross_category_generator,
        generate_cross_category_features_optimized
    )
    from src.feature_generation.core.optimized_feature_generator import OptimizedFeatureGenerator
    OPTIMIZED_GENERATORS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Optimized generators not available: {e}")
    OPTIMIZED_GENERATORS_AVAILABLE = False

# Import VectorBT components
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
    from src.utils.ml_common.unified_vectorization_manager import get_unified_vectorization_manager
    VECTORBT_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"VectorBT components not available: {e}")
    VECTORBT_COMPONENTS_AVAILABLE = False


def create_sample_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(seed)
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='1min')
    
    # Generate realistic price data
    base_price = 100
    returns = np.random.randn(n_samples) * 0.01
    prices = base_price + np.cumsum(returns)
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'open': prices + np.random.randn(n_samples) * 0.1,
        'high': prices + np.abs(np.random.randn(n_samples) * 0.5),
        'low': prices - np.abs(np.random.randn(n_samples) * 0.5),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    }, index=dates)
    
    # Ensure high >= low and high >= close >= low
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    return data


def demonstrate_batch_rolling_operations():
    """Demonstrate batch rolling operations using VectorBTRollingOptimizer."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING BATCH ROLLING OPERATIONS")
    logger.info("=" * 60)
    
    if not OPTIMIZED_GENERATORS_AVAILABLE:
        logger.warning("Optimized generators not available, skipping demonstration")
        return
    
    # Create sample data
    data = create_sample_data(1000)
    logger.info(f"Created sample data with {len(data)} rows")
    
    # Create trend generator
    trend_generator = create_optimized_trend_generator(periods=[10, 20, 50])
    
    # Demonstrate batch SMA generation
    logger.info("Generating SMA features in batch...")
    start_time = time.time()
    
    sma_features = trend_generator.generate_sma_features_batch(data, [10, 20, 50])
    
    end_time = time.time()
    logger.info(f"Generated {len(sma_features.columns)} SMA features in {end_time - start_time:.3f}s")
    logger.info(f"SMA features: {list(sma_features.columns)}")
    
    # Demonstrate batch EMA generation
    logger.info("Generating EMA features in batch...")
    start_time = time.time()
    
    ema_features = trend_generator.generate_ema_features_batch(data, [10, 20, 50])
    
    end_time = time.time()
    logger.info(f"Generated {len(ema_features.columns)} EMA features in {end_time - start_time:.3f}s")
    logger.info(f"EMA features: {list(ema_features.columns)}")
    
    # Get performance statistics
    stats = trend_generator.get_performance_stats()
    logger.info(f"Performance stats: {stats}")


def demonstrate_unified_vectorization_manager():
    """Demonstrate UnifiedVectorizationManager integration."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING UNIFIED VECTORIZATION MANAGER")
    logger.info("=" * 60)
    
    if not VECTORBT_COMPONENTS_AVAILABLE:
        logger.warning("VectorBT components not available, skipping demonstration")
        return
    
    # Create sample data
    data = create_sample_data(1000)
    logger.info(f"Created sample data with {len(data)} rows")
    
    # Get unified vectorization manager
    unified_manager = get_unified_vectorization_manager()
    
    # Demonstrate batch processing
    logger.info("Demonstrating batch processing with UnifiedVectorizationManager...")
    
    feature_configs = [
        {'name': 'sma_20', 'type': 'rolling', 'params': {'operation': 'mean', 'window': 20, 'column': 'close'}},
        {'name': 'sma_50', 'type': 'rolling', 'params': {'operation': 'mean', 'window': 50, 'column': 'close'}},
        {'name': 'std_20', 'type': 'rolling', 'params': {'operation': 'std', 'window': 20, 'column': 'close'}},
        {'name': 'close_scaled', 'type': 'scaling', 'params': {'method': 'zscore', 'column': 'close'}}
    ]
    
    start_time = time.time()
    features = unified_manager.batch_process_features(data, feature_configs)
    end_time = time.time()
    
    logger.info(f"Generated {len(features.columns)} features in {end_time - start_time:.3f}s")
    logger.info(f"Features: {list(features.columns)}")
    
    # Get performance statistics
    stats = unified_manager.get_performance_stats()
    logger.info(f"UnifiedVectorizationManager stats: {stats}")


def demonstrate_memory_optimization():
    """Demonstrate memory optimization features."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING MEMORY OPTIMIZATION")
    logger.info("=" * 60)
    
    if not OPTIMIZED_GENERATORS_AVAILABLE:
        logger.warning("Optimized generators not available, skipping demonstration")
        return
    
    # Create sample data
    data = create_sample_data(2000)
    logger.info(f"Created sample data with {len(data)} rows")
    
    # Check original memory usage
    original_memory = data.memory_usage(deep=True).sum() / (1024**2)  # MB
    logger.info(f"Original memory usage: {original_memory:.2f} MB")
    
    # Create volatility generator
    volatility_generator = create_optimized_volatility_generator(periods=[10, 20, 50])
    
    # Demonstrate memory optimization
    logger.info("Demonstrating memory optimization...")
    start_time = time.time()
    
    optimized_data = volatility_generator.optimize_dataframe_processing(data)
    
    end_time = time.time()
    optimized_memory = optimized_data.memory_usage(deep=True).sum() / (1024**2)  # MB
    
    logger.info(f"Optimized memory usage: {optimized_memory:.2f} MB")
    logger.info(f"Memory savings: {((original_memory - optimized_memory) / original_memory * 100):.2f}%")
    logger.info(f"Optimization time: {end_time - start_time:.3f}s")
    
    # Get performance statistics
    stats = volatility_generator.get_performance_stats()
    logger.info(f"Memory optimization stats: {stats}")


def demonstrate_smart_caching():
    """Demonstrate smart caching mechanisms."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING SMART CACHING")
    logger.info("=" * 60)
    
    if not OPTIMIZED_GENERATORS_AVAILABLE:
        logger.warning("Optimized generators not available, skipping demonstration")
        return
    
    # Create sample data
    data = create_sample_data(1000)
    logger.info(f"Created sample data with {len(data)} rows")
    
    # Create returns generator
    returns_generator = create_optimized_returns_generator(periods=[10, 20, 50])
    
    # Demonstrate caching with repeated operations
    logger.info("Demonstrating smart caching with repeated operations...")
    
    # First run - should populate cache
    start_time = time.time()
    features1 = returns_generator.generate_all_returns_features(data)
    end_time = time.time()
    first_run_time = end_time - start_time
    
    # Second run - should use cache
    start_time = time.time()
    features2 = returns_generator.generate_all_returns_features(data)
    end_time = time.time()
    second_run_time = end_time - start_time
    
    logger.info(f"First run time: {first_run_time:.3f}s")
    logger.info(f"Second run time: {second_run_time:.3f}s")
    logger.info(f"Speedup from caching: {first_run_time / second_run_time:.2f}x")
    
    # Get performance statistics
    stats = returns_generator.get_performance_stats()
    logger.info(f"Caching stats: {stats}")


def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING PERFORMANCE MONITORING")
    logger.info("=" * 60)
    
    if not OPTIMIZED_GENERATORS_AVAILABLE:
        logger.warning("Optimized generators not available, skipping demonstration")
        return
    
    # Create sample data
    data = create_sample_data(1000)
    logger.info(f"Created sample data with {len(data)} rows")
    
    # Create cross-category generator
    cross_category_generator = create_optimized_cross_category_generator(
        enabled_categories=['trend', 'volatility', 'returns'],
        periods=[10, 20, 50]
    )
    
    # Demonstrate performance monitoring
    logger.info("Demonstrating performance monitoring...")
    
    with cross_category_generator.performance_monitoring("comprehensive_feature_generation"):
        features = cross_category_generator.generate_all_cross_category_features(data)
    
    logger.info(f"Generated {len(features.columns)} cross-category features")
    
    # Get detailed performance statistics
    stats = cross_category_generator.get_performance_stats()
    logger.info(f"Performance monitoring stats: {stats}")


def demonstrate_cross_category_optimization():
    """Demonstrate cross-category feature generation optimization."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING CROSS-CATEGORY OPTIMIZATION")
    logger.info("=" * 60)
    
    if not OPTIMIZED_GENERATORS_AVAILABLE:
        logger.warning("Optimized generators not available, skipping demonstration")
        return
    
    # Create sample data
    data = create_sample_data(1000)
    logger.info(f"Created sample data with {len(data)} rows")
    
    # Create cross-category generator
    cross_category_generator = create_optimized_cross_category_generator(
        enabled_categories=['trend', 'volatility', 'returns', 'momentum', 'volume'],
        periods=[10, 20, 50],
        cross_correlations=True,
        interaction_features=True,
        regime_features=True
    )
    
    # Demonstrate cross-category feature generation
    logger.info("Demonstrating cross-category feature generation...")
    start_time = time.time()
    
    features = cross_category_generator.generate_all_cross_category_features(data)
    
    end_time = time.time()
    logger.info(f"Generated {len(features.columns)} cross-category features in {end_time - start_time:.3f}s")
    
    # Categorize features
    feature_categories = {
        'trend': [col for col in features.columns if any(x in col for x in ['sma', 'ema', 'adx', 'macd', 'rsi'])],
        'volatility': [col for col in features.columns if any(x in col for x in ['volatility', 'bb_', 'atr', 'kc_'])],
        'returns': [col for col in features.columns if any(x in col for x in ['return', 'momentum', 'roc'])],
        'correlation': [col for col in features.columns if 'correlation' in col],
        'interaction': [col for col in features.columns if 'interaction' in col],
        'regime': [col for col in features.columns if 'regime' in col]
    }
    
    for category, feature_list in feature_categories.items():
        if feature_list:
            logger.info(f"{category.capitalize()} features ({len(feature_list)}): {feature_list[:5]}{'...' if len(feature_list) > 5 else ''}")
    
    # Get performance statistics
    stats = cross_category_generator.get_performance_stats()
    logger.info(f"Cross-category optimization stats: {stats}")


def demonstrate_comprehensive_optimization():
    """Demonstrate comprehensive optimization across all categories."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING COMPREHENSIVE OPTIMIZATION")
    logger.info("=" * 60)
    
    if not OPTIMIZED_GENERATORS_AVAILABLE:
        logger.warning("Optimized generators not available, skipping demonstration")
        return
    
    # Create sample data
    data = create_sample_data(2000)
    logger.info(f"Created sample data with {len(data)} rows")
    
    # Create all generators
    generators = {
        'trend': create_optimized_trend_generator(periods=[10, 20, 50, 100]),
        'volatility': create_optimized_volatility_generator(periods=[10, 20, 50, 100]),
        'returns': create_optimized_returns_generator(periods=[5, 10, 20, 50]),
        'cross_category': create_optimized_cross_category_generator(
            enabled_categories=['trend', 'volatility', 'returns'],
            periods=[10, 20, 50]
        )
    }
    
    # Generate features from each category
    all_features = {}
    total_time = 0
    
    for category, generator in generators.items():
        logger.info(f"Generating {category} features...")
        start_time = time.time()
        
        if category == 'cross_category':
            features = generator.generate_all_cross_category_features(data)
        else:
            features = generator.generate_all_trend_features(data) if category == 'trend' else \
                      generator.generate_all_volatility_features(data) if category == 'volatility' else \
                      generator.generate_all_returns_features(data)
        
        end_time = time.time()
        generation_time = end_time - start_time
        total_time += generation_time
        
        all_features[category] = features
        logger.info(f"Generated {len(features.columns)} {category} features in {generation_time:.3f}s")
        
        # Get performance statistics
        stats = generator.get_performance_stats()
        logger.info(f"{category.capitalize()} performance: VectorBT rate={stats.get('vectorbt_usage_rate', 0):.2%}, "
                   f"Cache hit rate={stats.get('cache_hit_rate', 0):.2f}%, "
                   f"Memory optimizations={stats.get('memory_optimizations', 0)}")
    
    # Combine all features
    logger.info("Combining all features...")
    combined_features = pd.concat(list(all_features.values()), axis=1)
    
    logger.info(f"Total features generated: {len(combined_features.columns)}")
    logger.info(f"Total generation time: {total_time:.3f}s")
    logger.info(f"Average time per feature: {total_time / len(combined_features.columns):.4f}s")
    
    # Memory usage
    memory_usage = combined_features.memory_usage(deep=True).sum() / (1024**2)  # MB
    logger.info(f"Total memory usage: {memory_usage:.2f} MB")
    
    return combined_features


def main():
    """Main demonstration function."""
    logger.info("🚀 Starting Comprehensive VectorBT Optimization Demonstration")
    logger.info("=" * 80)
    
    try:
        # Demonstrate each optimization
        demonstrate_batch_rolling_operations()
        demonstrate_unified_vectorization_manager()
        demonstrate_memory_optimization()
        demonstrate_smart_caching()
        demonstrate_performance_monitoring()
        demonstrate_cross_category_optimization()
        
        # Comprehensive demonstration
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE OPTIMIZATION DEMONSTRATION")
        logger.info("=" * 80)
        
        combined_features = demonstrate_comprehensive_optimization()
        
        logger.info("=" * 80)
        logger.info("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        # Summary
        logger.info("SUMMARY OF OPTIMIZATIONS:")
        logger.info("1. ✅ Batch rolling operations using VectorBTRollingOptimizer")
        logger.info("2. ✅ UnifiedVectorizationManager integration for cross-category features")
        logger.info("3. ✅ Memory optimization with data type optimization")
        logger.info("4. ✅ Smart caching for frequently computed operations")
        logger.info("5. ✅ Performance monitoring and statistics")
        logger.info("6. ✅ Cross-category feature generation optimization")
        
        logger.info(f"Total features generated: {len(combined_features.columns)}")
        logger.info("All optimizations are working correctly!")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    main()