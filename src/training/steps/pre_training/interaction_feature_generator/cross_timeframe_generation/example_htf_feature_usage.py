"""
Example Usage of the New HTF Base Features System

This script demonstrates how to use the refactored htf_base_features module
with dynamic feature generation and lookback optimization.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Import the new HTF base features system
from .htf_base_features import (
    generate_htf_features,
    optimize_htf_lookbacks,
    get_feature_generator,
    get_base_feature_func,
    resample_to_htf,
    DynamicFeatureGenerator
)

# Import FeatureCategory if available
try:
    from src.feature_generation.core.feature_generator import FeatureCategory
    FEATURE_CATEGORIES_AVAILABLE = True
except ImportError:
    FEATURE_CATEGORIES_AVAILABLE = False
    print("⚠️ FeatureCategory not available - some examples will be skipped")


def create_sample_data(n_rows: int = 1000) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate random walk price data
    close_prices = 100 + np.random.randn(n_rows).cumsum()
    
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_rows, freq='5min'),
        'open': close_prices + np.random.randn(n_rows) * 0.5,
        'high': close_prices + np.abs(np.random.randn(n_rows) * 1.0),
        'low': close_prices - np.abs(np.random.randn(n_rows) * 1.0),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, n_rows)
    })
    
    # Set timestamp as index
    data.set_index('timestamp', inplace=True)
    
    return data


def example_1_basic_feature_generation():
    """Example 1: Basic dynamic feature generation."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Dynamic Feature Generation")
    print("="*80)
    
    # Create sample data
    data = create_sample_data(500)
    print(f"✅ Created sample data: {data.shape}")
    
    if not FEATURE_CATEGORIES_AVAILABLE:
        print("⚠️ Skipping - FeatureCategory not available")
        return
    
    # Generate features dynamically
    print("\n📊 Generating features...")
    features_df = generate_htf_features(
        data=data,
        categories=[
            FeatureCategory.MOMENTUM,
            FeatureCategory.VOLATILITY,
            FeatureCategory.TREND,
            FeatureCategory.OSCILLATOR
        ]
    )
    
    print(f"✅ Generated {features_df.shape[1]} features")
    print(f"\nSample feature names (first 10):")
    for i, col in enumerate(list(features_df.columns[:10]), 1):
        print(f"  {i}. {col}")
    
    return features_df


def example_2_lookback_optimization():
    """Example 2: Optimize lookback periods for features."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Lookback Period Optimization")
    print("="*80)
    
    # Create sample data
    data = create_sample_data(1000)
    print(f"✅ Created sample data: {data.shape}")
    
    if not FEATURE_CATEGORIES_AVAILABLE:
        print("⚠️ Skipping - FeatureCategory not available")
        return
    
    # Generate features
    print("\n📊 Generating features...")
    features_df = generate_htf_features(data)
    
    if features_df.empty:
        print("⚠️ No features generated")
        return
    
    # Add a synthetic target column
    data['target'] = np.log(data['close'] / data['close'].shift(1)).shift(-1)
    
    # Combine data and features
    combined_data = pd.concat([data, features_df], axis=1)
    
    # Select a few features to optimize
    feature_columns = list(features_df.columns[:5])
    print(f"\n🎯 Optimizing lookback periods for features: {feature_columns}")
    
    # Optimize lookback periods
    optimization_results = optimize_htf_lookbacks(
        data=combined_data,
        feature_columns=feature_columns,
        target_column='target',
        lookback_range=(5, 100)  # Smaller range for faster testing
    )
    
    # Display results
    print("\n📈 Optimization Results:")
    print("-" * 80)
    for feature_name, result in optimization_results.items():
        print(f"\nFeature: {feature_name}")
        print(f"  Best Lookback Period: {result['best_lookback_period']}")
        print(f"  Best Score: {result['best_score']:.4f}")
        print(f"  Method: {result['method']}")
    
    return optimization_results


def example_3_direct_generator_usage():
    """Example 3: Use DynamicFeatureGenerator directly."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Direct Generator Usage")
    print("="*80)
    
    # Create sample data
    data = create_sample_data(500)
    print(f"✅ Created sample data: {data.shape}")
    
    # Get the global feature generator
    generator = get_feature_generator()
    print(f"✅ Got feature generator (initialized: {generator._initialized})")
    
    if not generator._initialized:
        print("⚠️ Generator not properly initialized")
        return
    
    # Generate features with custom settings
    print("\n📊 Generating features with custom exclude patterns...")
    features_df = generator.generate_features(
        data=data,
        categories=[FeatureCategory.MOMENTUM] if FEATURE_CATEGORIES_AVAILABLE else None,
        exclude_patterns=['wavelet', 'autoencoder', 'regime']
    )
    
    print(f"✅ Generated {features_df.shape[1]} features")
    
    # Optimize a single feature
    if not features_df.empty and FEATURE_CATEGORIES_AVAILABLE:
        feature_name = features_df.columns[0]
        print(f"\n🎯 Optimizing lookback for feature: {feature_name}")
        
        # Add target
        data['target'] = np.log(data['close'] / data['close'].shift(1)).shift(-1)
        combined_data = pd.concat([data, features_df], axis=1)
        
        result = generator.optimize_feature_lookback(
            data=combined_data,
            feature_name=feature_name,
            target_column='target',
            lookback_range=(5, 100),
            method='coarse_to_refine'
        )
        
        print(f"\n📈 Optimization Result:")
        print(f"  Best Lookback Period: {result['best_lookback_period']}")
        print(f"  Best Score: {result['best_score']:.4f}")
        print(f"  Method: {result['method']}")
    
    return generator


def example_4_backward_compatibility():
    """Example 4: Backward compatibility with old interface."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Backward Compatibility")
    print("="*80)
    
    # Create sample data
    data = create_sample_data(500)
    print(f"✅ Created sample data: {data.shape}")
    
    # Use the old interface (now with dynamic generation)
    print("\n📊 Using get_base_feature_func (backward compatible)...")
    
    # Get feature function (old way)
    rsi_func = get_base_feature_func('rsi', lookback_period=14)
    
    # Compute feature
    rsi_series = rsi_func(data)
    print(f"✅ Computed RSI series: {len(rsi_series)} values")
    print(f"   Non-null values: {rsi_series.notna().sum()}")
    
    # Resample to HTF (unchanged)
    print("\n📊 Resampling to HTF...")
    htf_rsi = resample_to_htf(
        base_series=rsi_series,
        lookback_minutes=60,
        family='oscillators'
    )
    print(f"✅ Resampled to HTF: {len(htf_rsi)} values")
    
    return rsi_series, htf_rsi


def example_5_complete_workflow():
    """Example 5: Complete workflow from data to optimized HTF features."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Complete Workflow")
    print("="*80)
    
    # Step 1: Create or load data
    print("\n📊 Step 1: Create sample data")
    data = create_sample_data(1000)
    print(f"✅ Data shape: {data.shape}")
    
    if not FEATURE_CATEGORIES_AVAILABLE:
        print("⚠️ Skipping - FeatureCategory not available")
        return
    
    # Step 2: Generate base features
    print("\n📊 Step 2: Generate base features")
    features_df = generate_htf_features(
        data=data,
        categories=[
            FeatureCategory.MOMENTUM,
            FeatureCategory.VOLATILITY,
            FeatureCategory.TREND
        ]
    )
    print(f"✅ Generated {features_df.shape[1]} base features")
    
    if features_df.empty:
        print("⚠️ No features generated")
        return
    
    # Step 3: Add target column
    print("\n📊 Step 3: Add target column")
    data['target'] = np.log(data['close'] / data['close'].shift(1)).shift(-1)
    print(f"✅ Target column added")
    
    # Step 4: Combine data and features
    print("\n📊 Step 4: Combine data and features")
    combined_data = pd.concat([data, features_df], axis=1)
    print(f"✅ Combined data shape: {combined_data.shape}")
    
    # Step 5: Select top features for optimization
    print("\n📊 Step 5: Select features for optimization")
    # In real scenario, you'd use correlation or other methods
    selected_features = list(features_df.columns[:10])
    print(f"✅ Selected {len(selected_features)} features")
    
    # Step 6: Optimize lookback periods
    print("\n📊 Step 6: Optimize lookback periods")
    optimization_results = optimize_htf_lookbacks(
        data=combined_data,
        feature_columns=selected_features,
        target_column='target',
        lookback_range=(5, 100)
    )
    print(f"✅ Optimized {len(optimization_results)} features")
    
    # Step 7: Apply optimized lookbacks and resample to HTF
    print("\n📊 Step 7: Resample to HTF frequencies")
    htf_features = {}
    
    for feature_name, result in list(optimization_results.items())[:3]:  # First 3 for demo
        # Get the feature series
        feature_series = combined_data[feature_name]
        
        # Resample to HTF
        htf_feature = resample_to_htf(
            base_series=feature_series,
            lookback_minutes=result['best_lookback_period'],
            family='trend_level_vol'  # Example family
        )
        
        htf_features[f"htf_{feature_name}_{result['best_lookback_period']}m"] = htf_feature
        print(f"  ✅ {feature_name}: {result['best_lookback_period']}min -> {len(htf_feature)} values")
    
    # Step 8: Create final feature matrix
    print("\n📊 Step 8: Create final HTF feature matrix")
    htf_df = pd.DataFrame(htf_features)
    print(f"✅ Final HTF features: {htf_df.shape}")
    print(f"\nHTF Feature names:")
    for col in htf_df.columns:
        print(f"  - {col}")
    
    return htf_df, optimization_results


def run_all_examples():
    """Run all examples."""
    print("\n" + "="*80)
    print("HTF BASE FEATURES - NEW SYSTEM EXAMPLES")
    print("="*80)
    print("\nThis script demonstrates the new dynamic feature generation")
    print("and lookback optimization system.\n")
    
    try:
        # Example 1: Basic feature generation
        example_1_basic_feature_generation()
        
        # Example 2: Lookback optimization
        example_2_lookback_optimization()
        
        # Example 3: Direct generator usage
        example_3_direct_generator_usage()
        
        # Example 4: Backward compatibility
        example_4_backward_compatibility()
        
        # Example 5: Complete workflow
        example_5_complete_workflow()
        
        print("\n" + "="*80)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_examples()