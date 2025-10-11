"""
Comprehensive Usage Example for Unified Feature Generation System

This example demonstrates how to use the unified feature generation system
for various trading and financial analysis tasks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

# Import the unified feature generation system
from src.feature_generation import (
    # Core classes
    FeatureBank,
    FeatureBankConfig,
    FeatureCategory,
    
    # Convenience functions
    generate_features_by_category,
    generate_all_features,
    get_feature_summary,
    validate_feature_data,
    
    # Category-specific generators
    ReturnsFeatureGenerator,
    MomentumFeatureGenerator,
    VolumeFeatureGenerator,
    VolatilityFeatureGenerator,
    TrendFeatureGenerator,
    
    # Optimization
    LookbackOptimizer,
    FeatureOptimizationConfig,
    OptimizationMethod,
    
    # Matrix integration
    enable_matrix_acceleration,
    get_matrix_processor,
    
    # Backwards compatibility
    LegacyFeatureAdapter,
    migrate_legacy_features
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample OHLCV data for demonstration."""
    np.random.seed(42)
    
    # Generate price data with trend and volatility
    returns = np.random.normal(0.001, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    })
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # Add a target variable (next period return)
    data['target'] = data['close'].pct_change().shift(-1)
    
    return data.set_index('timestamp')

def example_basic_usage():
    """Example 1: Basic usage with convenience functions."""
    logger.info("=== Example 1: Basic Usage ===")
    
    # Create sample data
    data = create_sample_data(500)
    logger.info(f"Created sample data with shape: {data.shape}")
    
    # Generate features by category using convenience function
    features = generate_features_by_category(
        data=data,
        categories=['returns', 'momentum', 'volume'],
        lookback_optimization=False
    )
    
    logger.info(f"Generated {len(features.columns)} features")
    logger.info(f"Feature columns: {list(features.columns)}")
    
    return features

def example_advanced_usage():
    """Example 2: Advanced usage with feature bank configuration."""
    logger.info("=== Example 2: Advanced Usage ===")
    
    # Create sample data
    data = create_sample_data(1000)
    
    # Configure feature bank
    config = FeatureBankConfig(
        enable_matrix_operations=True,
        enable_gpu_acceleration=True,
        enable_lookback_optimization=True,
        enable_parallel_processing=True,
        max_workers=4,
        cache_results=True
    )
    
    # Initialize feature bank
    bank = FeatureBank(config)
    
    # Generate features with optimization
    features = bank.generate_features(
        data=data,
        categories=[FeatureCategory.RETURNS, FeatureCategory.MOMENTUM, FeatureCategory.VOLUME],
        lookback_optimization=True,
        target_column='target'
    )
    
    logger.info(f"Generated {len(features.columns)} features with optimization")
    
    # Get performance statistics
    stats = bank.get_performance_stats()
    logger.info(f"Performance stats: {stats}")
    
    return features

def example_category_specific_generation():
    """Example 3: Category-specific feature generation."""
    logger.info("=== Example 3: Category-Specific Generation ===")
    
    # Create sample data
    data = create_sample_data(500)
    
    # Create specific generators
    returns_gen = ReturnsFeatureGenerator()
    momentum_gen = MomentumFeatureGenerator()
    volume_gen = VolumeFeatureGenerator()
    
    # Generate features individually
    returns_features = returns_gen.generate(data)
    momentum_features = momentum_gen.generate(data)
    volume_features = volume_gen.generate(data)
    
    logger.info(f"Returns features: {returns_features.name}")
    logger.info(f"Momentum features: {momentum_features.name}")
    logger.info(f"Volume features: {volume_features.name}")
    
    # Combine features
    all_features = pd.concat([
        returns_features.data,
        momentum_features.data,
        volume_features.data
    ], axis=1)
    
    logger.info(f"Combined features shape: {all_features.shape}")
    
    return all_features

def example_feature_bank_management():
    """Example 4: Feature bank management and exploration."""
    logger.info("=== Example 4: Feature Bank Management ===")
    
    # Initialize feature bank
    bank = FeatureBank()
    
    # List available categories
    categories = bank.list_categories()
    logger.info(f"Available categories: {[cat.value for cat in categories]}")
    
    # List features in each category
    for category in categories:
        features = bank.list_features(category)
        logger.info(f"{category.value}: {len(features)} features")
        if features:
            logger.info(f"  Features: {features[:5]}...")  # Show first 5
    
    # Get feature summary
    summary = bank.get_feature_summary()
    logger.info(f"Feature summary: {summary}")
    
    # Get specific generator
    rsi_generator = bank.get_generator_by_name("rsi_14")
    if rsi_generator:
        logger.info(f"Found RSI generator: {rsi_generator.config.description}")
    
    return summary

def example_lookback_optimization():
    """Example 5: Lookback optimization."""
    logger.info("=== Example 5: Lookback Optimization ===")
    
    # Create sample data
    data = create_sample_data(1000)
    
    # Configure optimization
    opt_config = FeatureOptimizationConfig(
        min_lookback=5,
        max_lookback=50,
        optimization_method=OptimizationMethod.STATISTICAL_ANALYSIS,
        parallel_processing=True
    )
    
    # Initialize optimizer
    optimizer = LookbackOptimizer(opt_config)
    
    # Get some generators to optimize
    bank = FeatureBank()
    generators = bank.get_generators_by_category(FeatureCategory.RETURNS)[:3]  # First 3
    
    # Optimize lookback periods
    optimal_lookbacks = optimizer.optimize_multiple_features(
        generators=generators,
        data=data,
        target_column='target'
    )
    
    logger.info(f"Optimal lookbacks: {optimal_lookbacks}")
    
    # Get optimization summary
    summary = optimizer.get_optimization_summary(optimal_lookbacks)
    logger.info(f"Optimization summary: {summary}")
    
    return optimal_lookbacks

def example_matrix_operations_integration():
    """Example 6: Matrix operations integration."""
    logger.info("=== Example 6: Matrix Operations Integration ===")
    
    # Enable matrix acceleration
    enable_matrix_acceleration(True)
    
    # Get matrix processor
    processor = get_matrix_processor(enable_gpu=True, enable_parallel=True)
    
    # Create sample data
    data = create_sample_data(2000)  # Larger dataset for matrix operations
    
    # Get generators
    bank = FeatureBank()
    generators = bank.get_generators_by_category(FeatureCategory.TREND)[:5]  # First 5 trend features
    
    # Process features with matrix optimization
    results = processor.process_features(generators, data)
    
    logger.info(f"Processed {len(results)} features with matrix operations")
    
    # Check which features were successful
    successful_features = [r.name for r in results if r.success]
    failed_features = [r.name for r in results if not r.success]
    
    logger.info(f"Successful features: {successful_features}")
    if failed_features:
        logger.warning(f"Failed features: {failed_features}")
    
    return results

def example_backwards_compatibility():
    """Example 7: Backwards compatibility with legacy code."""
    logger.info("=== Example 7: Backwards Compatibility ===")
    
    # Create legacy adapter
    adapter = LegacyFeatureAdapter()
    
    # List available legacy functions
    legacy_functions = adapter.list_available_legacy_functions()
    logger.info(f"Available legacy functions: {legacy_functions[:10]}...")  # Show first 10
    
    # Create sample data
    data = create_sample_data(500)
    
    # Create legacy generator
    try:
        sma_generator = adapter.create_legacy_generator(
            function_name='sma',
            category=FeatureCategory.TREND,
            description='Legacy Simple Moving Average',
            required_columns=['close'],
            period=20
        )
        
        # Generate feature using legacy function
        result = sma_generator.generate(data)
        logger.info(f"Generated legacy SMA feature: {result.name}")
        logger.info(f"Feature success: {result.success}")
        
    except Exception as e:
        logger.warning(f"Legacy function not available: {e}")
    
    # Example of migrating legacy configuration
    legacy_config = {
        'sma_20': {
            'category': 'trend',
            'description': 'Simple Moving Average 20',
            'required_columns': ['close'],
            'parameters': {'period': 20}
        },
        'ema_12': {
            'category': 'trend',
            'description': 'Exponential Moving Average 12',
            'required_columns': ['close'],
            'parameters': {'period': 12}
        }
    }
    
    try:
        migrated_generators = migrate_legacy_features(legacy_config)
        logger.info(f"Migrated {len(migrated_generators)} legacy features")
        
        # Generate features using migrated generators
        for generator in migrated_generators:
            result = generator.generate(data)
            logger.info(f"Migrated feature {generator.config.name}: {result.success}")
            
    except Exception as e:
        logger.warning(f"Legacy migration not available: {e}")
    
    return legacy_config

def example_data_validation():
    """Example 8: Data validation and feature requirements."""
    logger.info("=== Example 8: Data Validation ===")
    
    # Create sample data with missing columns
    data = create_sample_data(500)
    data_missing = data.drop(columns=['volume'])  # Remove volume column
    
    # Validate data for different categories
    validation_results = validate_feature_data(
        data=data_missing,
        categories=['returns', 'momentum', 'volume']
    )
    
    logger.info(f"Validation results: {validation_results}")
    
    # Check which features are valid/invalid
    valid_features = validation_results['valid_features']
    invalid_features = validation_results['invalid_features']
    missing_columns = validation_results['missing_columns']
    
    logger.info(f"Valid features: {len(valid_features)}")
    logger.info(f"Invalid features: {len(invalid_features)}")
    
    if missing_columns:
        logger.info(f"Missing columns: {missing_columns}")
    
    return validation_results

def example_feature_export_import():
    """Example 9: Feature configuration export/import."""
    logger.info("=== Example 9: Feature Export/Import ===")
    
    from src.feature_generation.convenience import export_feature_config, import_feature_config
    
    # Export feature configuration
    export_file = "/tmp/feature_config.json"
    export_feature_config(
        output_file=export_file,
        categories=['returns', 'momentum'],
        include_parameters=True
    )
    
    logger.info(f"Exported feature configuration to {export_file}")
    
    # Import feature configuration
    try:
        imported_config = import_feature_config(export_file)
        logger.info(f"Imported configuration with {len(imported_config['features'])} features")
        
        # Show some imported features
        feature_names = list(imported_config['features'].keys())[:5]
        logger.info(f"Sample imported features: {feature_names}")
        
    except Exception as e:
        logger.warning(f"Import failed: {e}")
    
    return export_file

def example_performance_comparison():
    """Example 10: Performance comparison between methods."""
    logger.info("=== Example 10: Performance Comparison ===")
    
    import time

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    
    # Create larger dataset for performance testing
    data = create_sample_data(5000)
    
    # Test 1: Basic feature generation
    start_time = time.time()
    features_basic = generate_features_by_category(
        data=data,
        categories=['returns', 'momentum'],
        lookback_optimization=False
    )
    basic_time = time.time() - start_time
    
    # Test 2: Feature generation with matrix operations
    enable_matrix_acceleration(True)
    start_time = time.time()
    features_matrix = generate_features_by_category(
        data=data,
        categories=['returns', 'momentum'],
        lookback_optimization=False
    )
    matrix_time = time.time() - start_time
    
    # Test 3: Feature generation with optimization
    start_time = time.time()
    features_optimized = generate_features_by_category(
        data=data,
        categories=['returns', 'momentum'],
        lookback_optimization=True,
        target_column='target'
    )
    optimized_time = time.time() - start_time
    
    # Compare results
    logger.info(f"Basic generation time: {basic_time:.3f}s")
    logger.info(f"Matrix operations time: {matrix_time:.3f}s")
    logger.info(f"Optimized generation time: {optimized_time:.3f}s")
    
    logger.info(f"Basic features: {len(features_basic.columns)}")
    logger.info(f"Matrix features: {len(features_matrix.columns)}")
    logger.info(f"Optimized features: {len(features_optimized.columns)}")
    
    # Calculate speedup
    if basic_time > 0:
        matrix_speedup = basic_time / matrix_time
        logger.info(f"Matrix operations speedup: {matrix_speedup:.2f}x")
    
    return {
        'basic_time': basic_time,
        'matrix_time': matrix_time,
        'optimized_time': optimized_time,
        'basic_features': len(features_basic.columns),
        'matrix_features': len(features_matrix.columns),
        'optimized_features': len(features_optimized.columns)
    }

def main():
    """Run all examples."""
    logger.info("🚀 Starting Unified Feature Generation System Examples")
    
    try:
        # Run all examples
        example_basic_usage()
        example_advanced_usage()
        example_category_specific_generation()
        example_feature_bank_management()
        example_lookback_optimization()
        example_matrix_operations_integration()
        example_backwards_compatibility()
        example_data_validation()
        example_feature_export_import()
        example_performance_comparison()
        
        logger.info("✅ All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        raise

if __name__ == "__main__":
    main()
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
