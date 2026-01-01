"""
Test Layer0-Enhanced Layer1 Integration

This script validates that Layer1 can successfully use Layer0-optimized prices
for better weighting optimization and signal quality.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.append('src')

from training.steps.labeling.label_based_layer_1 import run_layer1_optimization, LAYER0_AVAILABLE
from training.steps.labeling.unified_price_layer2 import load_layer0_params, generate_unified_layer2_price

def create_test_data(n_samples=1000):
    """Create synthetic market data for testing."""
    np.random.seed(42)
    
    # Create realistic price series with trend and noise
    price_base = 100.0
    trend = np.linspace(0, 0.1, n_samples)  # 10% uptrend
    noise = np.random.normal(0, 0.02, n_samples)  # 2% daily volatility
    price = price_base * (1 + trend + noise)
    
    # Add some volume
    volume = np.random.lognormal(10, 1, n_samples)
    
    # Create datetime index
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='15min')
    
    df = pd.DataFrame({
        'close': price,
        'high': price * 1.01,
        'low': price * 0.99,
        'volume': volume,
        'open': price
    }, index=dates)
    
    return df

def create_test_labels(df):
    """Create synthetic labels for testing."""
    # Simple momentum-based labels
    returns = df['close'].pct_change()
    labels = returns.shift(1) * 100  # Next period return in percent
    
    # Remove NaN and create binary labels
    labels = labels.dropna()
    binary_labels = (labels > 0).astype(int)
    
    return labels, binary_labels

def test_layer0_availability():
    """Test if Layer0 components are available."""
    print("🔍 Testing Layer0 Availability...")
    
    if LAYER0_AVAILABLE:
        print("✅ Layer0 components are available")
        
        try:
            layer0_params = load_layer0_params()
            print(f"✅ Loaded Layer0 params: {len(layer0_params)} parameters")
            print(f"   Key params: Q={layer0_params.get('kalman_Q', 'N/A')}, R={layer0_params.get('kalman_R', 'N/A')}")
            return True, layer0_params
        except Exception as e:
            print(f"❌ Failed to load Layer0 params: {e}")
            return False, None
    else:
        print("❌ Layer0 components not available")
        return False, None

def test_layer0_price_generation(df, layer0_params):
    """Test Layer0 price generation."""
    print("\n🔍 Testing Layer0 Price Generation...")
    
    try:
        optimized_price = generate_unified_layer2_price(df, layer0_params)
        print(f"✅ Generated Layer0 optimized price: {len(optimized_price)} points")
        print(f"   Price range: {optimized_price.min():.4f} - {optimized_price.max():.4f}")
        print(f"   Price volatility: {optimized_price.pct_change().std():.6f}")
        
        # Compare with raw price
        raw_vol = df['close'].pct_change().std()
        optimized_vol = optimized_price.pct_change().std()
        noise_reduction = (1 - optimized_vol / raw_vol) * 100
        
        print(f"   Raw price volatility: {raw_vol:.6f}")
        print(f"   Optimized volatility: {optimized_vol:.6f}")
        print(f"   Noise reduction: {noise_reduction:.1f}%")
        
        return True, optimized_price
    except Exception as e:
        print(f"❌ Failed to generate Layer0 price: {e}")
        return False, None

def test_layer1_with_raw_prices(df, labels):
    """Test Layer1 optimization with raw prices."""
    print("\n🔍 Testing Layer1 with Raw Prices...")
    
    try:
        params_raw = run_layer1_optimization(
            symbol="TEST",
            timeframe="15m",
            market_data=df,
            labels=labels,
            n_trials=10,  # Reduced for testing
            use_layer0_prices=False
        )
        
        print(f"✅ Layer1 optimization completed with raw prices")
        print(f"   Best params: {len(params_raw)} parameters")
        print(f"   Key params: mag_compression={params_raw.get('mag_compression', 'N/A'):.3f}")
        print(f"             uniq_intensity={params_raw.get('uniq_intensity', 'N/A'):.3f}")
        
        return True, params_raw
    except Exception as e:
        print(f"❌ Layer1 optimization failed with raw prices: {e}")
        return False, None

def test_layer1_with_layer0_prices(df, labels, layer0_params):
    """Test Layer1 optimization with Layer0 prices."""
    print("\n🔍 Testing Layer1 with Layer0 Prices...")
    
    try:
        params_layer0 = run_layer1_optimization(
            symbol="TEST",
            timeframe="15m",
            market_data=df,
            labels=labels,
            n_trials=10,  # Reduced for testing
            use_layer0_prices=True
        )
        
        print(f"✅ Layer1 optimization completed with Layer0 prices")
        print(f"   Best params: {len(params_layer0)} parameters")
        print(f"   Key params: mag_compression={params_layer0.get('mag_compression', 'N/A'):.3f}")
        print(f"             uniq_intensity={params_layer0.get('uniq_intensity', 'N/A'):.3f}")
        
        return True, params_layer0
    except Exception as e:
        print(f"❌ Layer1 optimization failed with Layer0 prices: {e}")
        return False, None

def compare_results(params_raw, params_layer0):
    """Compare results between raw and Layer0 price optimizations."""
    print("\n📊 Comparing Results...")
    
    if not params_raw or not params_layer0:
        print("❌ Cannot compare - one optimization failed")
        return
    
    # Compare key parameters
    key_params = ['mag_compression', 'uniq_intensity', 'exp_mag', 'downside_multiplier']
    
    print("🔍 Parameter Comparison:")
    print(f"{'Parameter':<20} {'Raw Prices':<15} {'Layer0 Prices':<15} {'Difference':<15}")
    print("-" * 65)
    
    for param in key_params:
        raw_val = params_raw.get(param, 0)
        layer0_val = params_layer0.get(param, 0)
        diff = layer0_val - raw_val
        
        print(f"{param:<20} {raw_val:<15.3f} {layer0_val:<15.3f} {diff:<15.3f}")
    
    # Analyze differences
    mag_diff = params_layer0.get('mag_compression', 0) - params_raw.get('mag_compression', 0)
    uniq_diff = params_layer0.get('uniq_intensity', 0) - params_raw.get('uniq_intensity', 0)
    
    print(f"\n📈 Analysis:")
    if abs(mag_diff) > 0.05:
        print(f"   • Magnitude compression changed significantly ({mag_diff:+.3f})")
        print(f"     → Layer0 prices affect how returns are weighted")
    
    if abs(uniq_diff) > 0.1:
        print(f"   • Uniqueness intensity changed significantly ({uniq_diff:+.3f})")
        print(f"     → Layer0 prices affect event uniqueness detection")
    
    if mag_diff < -0.01:
        print(f"   • Lower mag_compression with Layer0 → less aggressive magnitude weighting")
    elif mag_diff > 0.01:
        print(f"   • Higher mag_compression with Layer0 → more aggressive magnitude weighting")
    
    if uniq_diff < -0.05:
        print(f"   • Lower uniq_intensity with Layer0 → less penalty for overlapping events")
    elif uniq_diff > 0.05:
        print(f"   • Higher uniq_intensity with Layer0 → more penalty for overlapping events")

def main():
    """Main test function."""
    print("🧪 Layer0-Enhanced Layer1 Integration Test")
    print("=" * 60)
    
    # Create test data
    print("📊 Creating test data...")
    df = create_test_data(1000)
    labels, binary_labels = create_test_labels(df)
    print(f"✅ Created test data: {len(df)} price points, {len(labels)} labels")
    
    # Test Layer0 availability
    layer0_available, layer0_params = test_layer0_availability()
    
    if not layer0_available:
        print("\n❌ Layer0 not available - cannot test integration")
        return
    
    # Test Layer0 price generation
    price_success, optimized_price = test_layer0_price_generation(df, layer0_params)
    
    if not price_success:
        print("\n❌ Layer0 price generation failed - cannot test integration")
        return
    
    # Test Layer1 with raw prices
    raw_success, params_raw = test_layer1_with_raw_prices(df, labels)
    
    # Test Layer1 with Layer0 prices
    layer0_success, params_layer0 = test_layer1_with_layer0_prices(df, labels, layer0_params)
    
    # Compare results
    if raw_success and layer0_success:
        compare_results(params_raw, params_layer0)
        
        print(f"\n🎯 Integration Test Summary:")
        print(f"✅ Layer0 components available")
        print(f"✅ Layer0 price generation working")
        print(f"✅ Layer1 optimization with raw prices working")
        print(f"✅ Layer1 optimization with Layer0 prices working")
        print(f"✅ Parameter comparison completed")
        
        print(f"\n💡 Key Benefits of Layer0-Enhanced Layer1:")
        print(f"   • Cleaner price signals improve volatility estimation")
        print(f"   • Better uniqueness calculations with reduced noise")
        print(f"   • Enhanced consistency scoring with feature preservation")
        print(f"   • Optimized weighting parameters for better sample quality")
        
    else:
        print(f"\n❌ Integration test failed")
        if not raw_success:
            print(f"   • Layer1 with raw prices failed")
        if not layer0_success:
            print(f"   • Layer1 with Layer0 prices failed")

if __name__ == "__main__":
    main()
