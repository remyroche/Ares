"""
Example Usage of Data-Driven Feature Selection System

This module demonstrates how to use the data-driven feature selection system
to select the most promising features from the feature bank for the lookback
optimization system.

Key Examples:
- Development configuration (fast, less thorough)
- Production configuration (thorough, robust)
- Custom configuration with specific parameters
- Integration with existing pipeline
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import the feature selection system
from .feature_selector import (
    DataDrivenFeatureSelector,
    select_features_development,
    select_features_production,
    select_features_custom
)
from .config import (

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
    DataDrivenFeatureSelectionConfig,
    create_development_config,
    create_production_config,
    create_custom_config
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_market_data(n_days: int = 2000) -> pd.DataFrame:
    """Generate sample market data for demonstration."""
    logger.info(f"Generating sample market data for {n_days} days...")
    
    # Generate price data
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.02, n_days)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    high_low_noise = np.random.uniform(0.001, 0.005, n_days)
    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.001, 0.001, n_days)),
        'high': prices * (1 + high_low_noise),
        'low': prices * (1 - high_low_noise),
        'close': prices,
        'volume': np.random.uniform(1000000, 5000000, n_days)
    })
    
    # Add some technical indicators
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['rsi_14'] = compute_rsi(df['close'], 14)
    df['bb_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
    df['bb_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
    
    return df


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI indicator."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = self._vectorbt_rolling_operation(gain, "mean", period)
    avg_loss = self._vectorbt_rolling_operation(loss, "mean", period)
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)


def generate_sample_targets(data: pd.DataFrame) -> np.ndarray:
    """Generate sample target variables."""
    # Generate future returns as targets
    future_returns = data['close'].pct_change(5).shift(-5)
    return future_returns.fillna(0).values


async def example_development_configuration():
    """Example using development configuration (fast, less thorough)."""
    print("\n" + "="*80)
    print("DEVELOPMENT CONFIGURATION EXAMPLE")
    print("="*80)
    
    # Generate sample data
    market_data = generate_sample_market_data(1000)
    targets = generate_sample_targets(market_data)
    
    # Data availability information
    data_availability = {
        'book_data': 0.95,  # 95% availability
        'tick_data': 0.90,  # 90% availability
        'volume_data': 1.0   # 100% availability
    }
    
    print(f"📊 Data shape: {market_data.shape}")
    print(f"🎯 Target length: {len(targets)}")
    print(f"📈 Data availability: {data_availability}")
    
    # Run feature selection with development config
    print("\n🚀 Running feature selection with development configuration...")
    result = await select_features_development(market_data, targets, data_availability)
    
    # Display results
    print(f"\n✅ Feature selection completed in {result.total_execution_time:.3f}s")
    print(f"📊 Features evaluated: {result.total_features_evaluated}")
    print(f"📊 Features selected: {result.total_features_selected}")
    print(f"💰 Budget utilization: {result.budget_utilization:.1%}")
    print(f"📈 Coverage achieved: {sum(result.coverage_achieved.values())}/{len(result.coverage_achieved)} families")
    
    if result.final_feature_names:
        print(f"\n🎯 Final features selected:")
        for i, feature in enumerate(result.final_feature_names[:10], 1):  # Show first 10
            print(f"  {i:2d}. {feature}")
        if len(result.final_feature_names) > 10:
            print(f"  ... and {len(result.final_feature_names) - 10} more")
    
    return result


async def example_production_configuration():
    """Example using production configuration (thorough, robust)."""
    print("\n" + "="*80)
    print("PRODUCTION CONFIGURATION EXAMPLE")
    print("="*80)
    
    # Generate sample data
    market_data = generate_sample_market_data(2000)
    targets = generate_sample_targets(market_data)
    
    # Data availability information
    data_availability = {
        'book_data': 0.98,  # 98% availability
        'tick_data': 0.95,  # 95% availability
        'volume_data': 1.0   # 100% availability
    }
    
    print(f"📊 Data shape: {market_data.shape}")
    print(f"🎯 Target length: {len(targets)}")
    print(f"📈 Data availability: {data_availability}")
    
    # Run feature selection with production config
    print("\n🚀 Running feature selection with production configuration...")
    result = await select_features_production(market_data, targets, data_availability)
    
    # Display results
    print(f"\n✅ Feature selection completed in {result.total_execution_time:.3f}s")
    print(f"📊 Features evaluated: {result.total_features_evaluated}")
    print(f"📊 Features selected: {result.total_features_selected}")
    print(f"💰 Budget utilization: {result.budget_utilization:.1%}")
    print(f"📈 Coverage achieved: {sum(result.coverage_achieved.values())}/{len(result.coverage_achieved)} families")
    
    if result.final_feature_names:
        print(f"\n🎯 Final features selected:")
        for i, feature in enumerate(result.final_feature_names[:15], 1):  # Show first 15
            print(f"  {i:2d}. {feature}")
        if len(result.final_feature_names) > 15:
            print(f"  ... and {len(result.final_feature_names) - 15} more")
    
    return result


async def example_custom_configuration():
    """Example using custom configuration with specific parameters."""
    print("\n" + "="*80)
    print("CUSTOM CONFIGURATION EXAMPLE")
    print("="*80)
    
    # Generate sample data
    market_data = generate_sample_market_data(1500)
    targets = generate_sample_targets(market_data)
    
    # Create custom configuration
    config = create_custom_config(
        phase1_overrides={
            'probe_days': 30,
            'subset_ratio': 0.5,
            'momentum_lookbacks': [5, 10, 15, 20],
            'volatility_lookbacks': [6, 12, 18, 24]
        },
        phase2_overrides={
            'n_samples': 1000,
            'warmup': 500,
            'enable_stability_test': True
        },
        budget_overrides={
            'max_features_pre_selection': 80,
            'max_final_features': 50,
            'lambda_cost': 0.15
        },
        final_selection_overrides={
            'target_feature_count': 40,
            'n_bootstrap_samples': 150,
            'fdr_q_value': 0.10
        }
    )
    
    print(f"📊 Data shape: {market_data.shape}")
    print(f"🎯 Target length: {len(targets)}")
    print(f"⚙️ Custom configuration created")
    
    # Run feature selection with custom config
    print("\n🚀 Running feature selection with custom configuration...")
    result = await select_features_custom(market_data, targets, config)
    
    # Display results
    print(f"\n✅ Feature selection completed in {result.total_execution_time:.3f}s")
    print(f"📊 Features evaluated: {result.total_features_evaluated}")
    print(f"📊 Features selected: {result.total_features_selected}")
    print(f"💰 Budget utilization: {result.budget_utilization:.1%}")
    print(f"📈 Coverage achieved: {sum(result.coverage_achieved.values())}/{len(result.coverage_achieved)} families")
    
    if result.final_feature_names:
        print(f"\n🎯 Final features selected:")
        for i, feature in enumerate(result.final_feature_names[:12], 1):  # Show first 12
            print(f"  {i:2d}. {feature}")
        if len(result.final_feature_names) > 12:
            print(f"  ... and {len(result.final_feature_names) - 12} more")
    
    return result


async def example_pipeline_integration():
    """Example showing integration with existing pipeline."""
    print("\n" + "="*80)
    print("PIPELINE INTEGRATION EXAMPLE")
    print("="*80)
    
    # Generate sample data
    market_data = generate_sample_market_data(2500)
    targets = generate_sample_targets(market_data)
    
    print(f"📊 Data shape: {market_data.shape}")
    print(f"🎯 Target length: {len(targets)}")
    
    # Create selector instance
    selector = DataDrivenFeatureSelector()
    
    # Run feature selection
    print("\n🚀 Running feature selection...")
    result = await selector.select_features(market_data, targets)
    
    # Display results
    print(f"\n✅ Feature selection completed in {result.total_execution_time:.3f}s")
    print(f"📊 Features evaluated: {result.total_features_evaluated}")
    print(f"📊 Features selected: {result.total_features_selected}")
    
    # Get performance summary
    performance = selector.get_performance_summary()
    print(f"\n📈 Performance Summary:")
    print(f"  Matrix ops used: {performance['matrix_ops_used']}")
    print(f"  Hardware accelerated ops: {performance['hardware_accelerated_ops']}")
    print(f"  Memory efficient ops: {performance['memory_efficient_ops']}")
    print(f"  Bayesian optimizations: {performance['bayesian_optimizations']}")
    
    # Save results
    print(f"\n💾 Saving results...")
    success = selector.save_results(result, "feature_selection_results.json")
    if success:
        print("✅ Results saved successfully")
    
    # Load results
    print(f"\n📂 Loading results...")
    loaded_result = selector.load_results("feature_selection_results.json")
    if loaded_result:
        print("✅ Results loaded successfully")
        print(f"📊 Loaded {len(loaded_result.final_feature_names)} features")
    
    return result


async def example_performance_comparison():
    """Example comparing different configurations."""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON EXAMPLE")
    print("="*80)
    
    # Generate sample data
    market_data = generate_sample_market_data(1500)
    targets = generate_sample_targets(market_data)
    
    print(f"📊 Data shape: {market_data.shape}")
    print(f"🎯 Target length: {len(targets)}")
    
    # Test different configurations
    configs = [
        ("Development", create_development_config()),
        ("Production", create_production_config()),
        ("Custom", create_custom_config(
            phase1_overrides={'probe_days': 25},
            budget_overrides={'max_features_pre_selection': 100}
        ))
    ]
    
    results = {}
    
    for config_name, config in configs:
        print(f"\n🔧 Testing {config_name} configuration...")
        
        try:
            selector = DataDrivenFeatureSelector(config)
            result = await selector.select_features(market_data, targets)
            results[config_name] = result
            
            print(f"✅ {config_name}: {result.total_execution_time:.3f}s, {result.total_features_selected} features")
            
        except Exception as e:
            print(f"❌ {config_name} failed: {e}")
            results[config_name] = None
    
    # Compare results
    print(f"\n📊 Performance Comparison:")
    print(f"{'Configuration':<15} {'Time (s)':<10} {'Features':<10} {'Budget %':<10} {'Coverage':<10}")
    print("-" * 65)
    
    for config_name, result in results.items():
        if result:
            coverage = sum(result.coverage_achieved.values())
            total_families = len(result.coverage_achieved)
            print(f"{config_name:<15} {result.total_execution_time:<10.3f} {result.total_features_selected:<10} {result.budget_utilization:<10.1%} {coverage}/{total_families}")
        else:
            print(f"{config_name:<15} {'FAILED':<10} {'N/A':<10} {'N/A':<10} {'N/A'}")
    
    return results


async def main():
    """Main example function."""
    print("Data-Driven Feature Selection System - Usage Examples")
    print("=" * 80)
    
    try:
        # Example 1: Development configuration
        print("\n1. Development Configuration")
        print("-" * 40)
        await example_development_configuration()
        
        # Example 2: Production configuration
        print("\n2. Production Configuration")
        print("-" * 40)
        await example_production_configuration()
        
        # Example 3: Custom configuration
        print("\n3. Custom Configuration")
        print("-" * 40)
        await example_custom_configuration()
        
        # Example 4: Pipeline integration
        print("\n4. Pipeline Integration")
        print("-" * 40)
        await example_pipeline_integration()
        
        # Example 5: Performance comparison
        print("\n5. Performance Comparison")
        print("-" * 40)
        await example_performance_comparison()
        
        print("\n" + "=" * 80)
        print("✅ All examples completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Example execution failed: {e}")
        print(f"\n❌ Example execution failed: {e}")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())