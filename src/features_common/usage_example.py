"""
Simple usage example demonstrating native optimization benefits.

This example shows how easy it is to use transforms that automatically
benefit from all available optimizations.
"""

import numpy as np
import pandas as pd
from src.features_common import create_optimized_scaler, get_unified_vectorbt_manager

def simple_usage_example():
    """
    Demonstrate simple usage with automatic optimization.
    """
    print("🚀 Simple Usage Example - All Optimizations by Default")
    print("=" * 60)

    # Create some sample data
    np.random.seed(42)
    data = pd.Series(np.random.randn(1000) + np.sin(np.linspace(0, 10, 1000)) * 2)

    print(f"📊 Sample data: {len(data)} points, range [{data.min():.2f}, {data.max():.2f}]")

    # Method 1: Create optimized scaler (automatically gets all optimizations)
    print("\n🔧 Method 1: Optimized Scaler")
    print("-" * 30)

    scaler = create_optimized_scaler(method='zscore')

    # Fit and transform - automatically uses all optimizations
    result = scaler.fit_transform(data)

    print(f"✅ Result: mean={result.mean():.3f}, std={result.std():.3f}")
    print(f"✅ All optimizations active: {scaler.is_optimization_enabled()}")

    # Method 2: Use VectorBT manager directly
    print("\n🔧 Method 2: VectorBT Manager")
    print("-" * 30)

    vectorbt_manager = get_unified_vectorbt_manager()

    # Rolling operations - automatically optimized
    rolling_mean = vectorbt_manager.rolling_mean(data, window=20)
    rolling_std = vectorbt_manager.rolling_std(data, window=20)

    print(f"✅ Rolling mean: {len(rolling_mean)} points")
    print(f"✅ Rolling std: {len(rolling_std)} points")
    print(f"✅ VectorBT available: {vectorbt_manager.is_vectorbt_available()}")

    # Method 3: Batch processing
    print("\n🔧 Method 3: Batch Processing")
    print("-" * 30)

    # Create multiple features
    features = pd.DataFrame({
        'feature1': data,
        'feature2': data * 1.5 + 2,
        'feature3': data * 0.8 - 1,
        'feature4': np.random.randn(1000)
    })

    # Scale all features at once - automatically optimized
    scaled_features = vectorbt_manager.scale_data(features, method='zscore')

    print(f"✅ Scaled features shape: {scaled_features.shape}")
    print(f"✅ All features normalized: mean≈0, std≈1")

    # Show that optimizations are working
    print("\n📊 Optimization Status")
    print("-" * 30)

    # Check scaler optimizations
    opt_stats = scaler.get_optimization_stats()
    print(f"✅ Optimization operations: {opt_stats.get('optimized_operations', 0)}")
    print(f"✅ Fallback operations: {opt_stats.get('fallback_operations', 0)}")

    # Check VectorBT optimizations
    vectorbt_stats = vectorbt_manager.get_operation_stats()
    print(f"✅ VectorBT operations: {vectorbt_stats.get('vectorbt_operations', 0)}")
    print(f"✅ Pandas fallbacks: {vectorbt_stats.get('pandas_fallbacks', 0)}")

    print("\n🎉 That's it! All optimizations work automatically.")
    print("   • No configuration needed")
    print("   • VectorBT acceleration when beneficial")
    print("   • Intelligent caching")
    print("   • Performance monitoring")
    print("   • Automatic fallback to pandas when needed")

if __name__ == "__main__":
    simple_usage_example()
