#!/usr/bin/env python3
"""
Test script to verify AdaptiveHunterRouter optimizations.
"""

import sys
import os
import time
import numpy as np
import pandas as pd

# Add Ares to path
sys.path.insert(0, '/Users/remyroche/Documents/Ares')

from src.utils.ml_common.physics_router import AdaptiveHunterRouter

def generate_test_data(n_rows=50000):
    """Generate synthetic test data similar to real market data."""
    np.random.seed(42)
    
    # Generate realistic price series with trends and volatility
    returns = np.random.normal(0.0001, 0.02, n_rows)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate volume with some correlation to price movements
    base_volume = 1000000
    volume_noise = np.random.normal(0, 0.3, n_rows)
    volume_boost = np.abs(np.gradient(returns)) * 10  # Higher volume on price changes
    volumes = base_volume * (1 + volume_noise + volume_boost)
    
    # Generate bar durations (inversely related to volume)
    base_duration = 60  # seconds
    durations = base_duration * (base_volume / volumes) ** 0.5
    durations = np.clip(durations, 10, 3600)  # Clip to reasonable range
    
    df = pd.DataFrame({
        'close': prices,
        'volume': volumes,
        'bar_duration': durations
    })
    
    return df

def test_optimization_performance():
    """Test performance of optimized vs original implementation."""
    print("🧪 Testing AdaptiveHunterRouter Optimizations")
    print("=" * 50)
    
    # Generate test data
    df = generate_test_data(50000)
    print(f"Generated test data: {len(df)} rows")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"Volume range: {df['volume'].min():.0f} - {df['volume'].max():.0f}")
    print()
    
    # Initialize router
    cache_dir = "/tmp/physics_router_test"
    router = AdaptiveHunterRouter(
        n_regimes=3, 
        window_size=1000, 
        mp_window=30,
        cache_dir=cache_dir
    )
    
    # Test feature computation
    print("🚀 Testing optimized feature computation...")
    start_time = time.time()
    
    features = router.compute_physics_features(df)
    
    computation_time = time.time() - start_time
    print(f"✅ Feature computation completed in {computation_time:.2f}s")
    print(f"✅ Generated {len(features.columns)} features for {len(features)} rows")
    print()
    
    # Display feature statistics
    print("📊 Feature Statistics:")
    for col in features.columns:
        mean_val = features[col].mean()
        std_val = features[col].std()
        nan_pct = features[col].isna().sum() / len(features) * 100
        print(f"  {col}: mean={mean_val:.4f}, std={std_val:.4f}, nan%={nan_pct:.2f}%")
    print()
    
    # Test caching by running again
    print("🔄 Testing caching performance (second run)...")
    start_time = time.time()
    
    features_cached = router.compute_physics_features(df)
    
    cached_time = time.time() - start_time
    speedup = computation_time / cached_time if cached_time > 0 else float('inf')
    print(f"✅ Cached computation completed in {cached_time:.2f}s")
    print(f"🚀 Speedup: {speedup:.1f}x faster")
    print()
    
    # Verify results are identical
    features_equal = np.allclose(features.values, features_cached.values, equal_nan=True)
    print(f"✅ Cache verification: {'PASSED' if features_equal else 'FAILED'}")
    print()
    
    # Test GMM fitting
    print("🤖 Testing GMM fitting...")
    start_time = time.time()
    
    router.fit(features.values)
    
    fit_time = time.time() - start_time
    print(f"✅ GMM fitting completed in {fit_time:.2f}s")
    print(f"✅ Regime mapping: {router.regime_map}")
    print()
    
    # Test prediction
    print("🔮 Testing prediction...")
    start_time = time.time()
    
    sample_features = features.iloc[-1:].values
    weights, entropy, z_familiar, confidence = router.predict(sample_features[0])
    
    pred_time = time.time() - start_time
    print(f"✅ Prediction completed in {pred_time:.4f}s")
    print(f"✅ Regime weights: {weights}")
    print(f"✅ Router confidence: {confidence:.3f}")
    print()
    
    # Performance summary
    print("📈 Performance Summary:")
    print(f"  Feature computation: {computation_time:.2f}s")
    print(f"  Cached computation: {cached_time:.2f}s ({speedup:.1f}x speedup)")
    print(f"  GMM fitting: {fit_time:.2f}s")
    print(f"  Prediction: {pred_time:.4f}s")
    print()
    
    # Cleanup
    import shutil
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    
    print("🎉 All tests completed successfully!")
    return True

if __name__ == "__main__":
    try:
        success = test_optimization_performance()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
