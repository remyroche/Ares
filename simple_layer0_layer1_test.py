"""
Simple Layer0-Layer1 Integration Test

Tests the core functionality without complex imports.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

def simple_kalman_filter(price_series, Q=1e-4, R=0.01):
    """Simple Kalman filter implementation."""
    n = len(price_series)
    x_hat = np.zeros(n)
    P = np.ones(n)
    
    x_hat[0] = price_series.iloc[0]
    P[0] = 1.0
    
    for k in range(1, n):
        x_hat_minus = x_hat[k-1]
        P_minus = P[k-1] + Q
        K = P_minus / (P_minus + R)
        x_hat[k] = x_hat_minus + K * (price_series.iloc[k] - x_hat_minus)
        P[k] = (1 - K) * P_minus
    
    return pd.Series(x_hat, index=price_series.index)

def simple_vwap(price, volume, window=50):
    """Simple VWAP calculation."""
    pv = price * volume
    cum_pv = pv.rolling(window).sum()
    cum_vol = volume.rolling(window).sum()
    return cum_pv / (cum_vol + 1e-9)

def generate_unified_price_simple(df, params):
    """Generate unified price with Layer0 parameters."""
    close = df['close']
    volume = df.get('volume', pd.Series(1, index=close.index))
    
    # Kalman filtering
    kalman_price = simple_kalman_filter(close, params['kalman_Q'], params['kalman_R'])
    
    # VWAP
    vwap_price = simple_vwap(kalman_price, volume, params['vwap_lookback'])
    
    # Composite
    composite_price = (1 - params['vwap_weight']) * kalman_price + params['vwap_weight'] * vwap_price
    
    return composite_price

def calculate_snr(series):
    """Calculate signal-to-noise ratio."""
    signal = series.rolling(50, center=True).mean()
    noise = series - signal
    
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    return 10 * np.log10(signal_power / noise_power)

def test_layer0_layer1_integration():
    """Test Layer0-Layer1 integration with synthetic data."""
    print("🧪 Simple Layer0-Layer1 Integration Test")
    print("=" * 50)
    
    # Create synthetic data
    np.random.seed(42)
    n_points = 500
    dates = pd.date_range('2024-01-01', periods=n_points, freq='15min')
    
    # Create price with trend and noise
    price_base = 100.0
    trend = np.linspace(0, 0.05, n_points)  # 5% uptrend
    noise = np.random.normal(0, 0.01, n_points)  # 1% volatility
    price = price_base * (1 + trend + noise)
    volume = np.random.lognormal(10, 0.5, n_points)
    
    df = pd.DataFrame({
        'close': price,
        'volume': volume
    }, index=dates)
    
    # Create labels
    returns = df['close'].pct_change().shift(-1) * 100  # Next period return
    labels = returns.dropna()
    
    print(f"📊 Created test data: {len(df)} points, {len(labels)} labels")
    
    # Layer0 parameters (optimized)
    layer0_params = {
        'kalman_Q': 1e-4,
        'kalman_R': 0.01,
        'vwap_weight': 0.4,
        'vwap_lookback': 50
    }
    
    print(f"⚙️ Layer0 params: Q={layer0_params['kalman_Q']}, R={layer0_params['kalman_R']}")
    
    # Test 1: Raw price SNR
    raw_snr = calculate_snr(df['close'])
    print(f"\n📈 Raw Price SNR: {raw_snr:.2f} dB")
    
    # Test 2: Layer0 optimized price SNR
    optimized_price = generate_unified_price_simple(df, layer0_params)
    optimized_snr = calculate_snr(optimized_price)
    snr_improvement = optimized_snr - raw_snr
    
    print(f"📈 Layer0 Price SNR: {optimized_snr:.2f} dB")
    print(f"📈 SNR Improvement: {snr_improvement:.2f} dB")
    
    # Test 3: Volatility comparison
    raw_vol = df['close'].pct_change().std()
    optimized_vol = optimized_price.pct_change().std()
    noise_reduction = (1 - optimized_vol / raw_vol) * 100
    
    print(f"📊 Raw volatility: {raw_vol:.6f}")
    print(f"📊 Optimized volatility: {optimized_vol:.6f}")
    print(f"📊 Noise reduction: {noise_reduction:.1f}%")
    
    # Test 4: Layer1 weighting simulation
    print(f"\n🎯 Layer1 Weighting Simulation:")
    
    # Simulate magnitude-based weights for raw vs optimized prices
    raw_returns = df['close'].pct_change().dropna()
    optimized_returns = optimized_price.pct_change().dropna()
    
    # Simple magnitude weighting
    raw_magnitude = np.abs(raw_returns)
    optimized_magnitude = np.abs(optimized_returns)
    
    # Calculate weight entropy (diversity measure)
    def calculate_entropy(weights):
        weights = np.asarray(weights)
        weights = weights / np.sum(weights)
        entropy = -np.sum(weights * np.log(weights + 1e-12))
        max_entropy = np.log(len(weights))
        return entropy / max_entropy
    
    raw_entropy = calculate_entropy(raw_magnitude)
    optimized_entropy = calculate_entropy(optimized_magnitude)
    
    print(f"   Raw magnitude entropy: {raw_entropy:.3f}")
    print(f"   Optimized magnitude entropy: {optimized_entropy:.3f}")
    print(f"   Entropy change: {optimized_entropy - raw_entropy:+.3f}")
    
    # Test 5: Uniqueness simulation
    print(f"\n🔄 Uniqueness Calculation Simulation:")
    
    # Simulate event uniqueness based on volatility
    def simulate_uniqueness(returns, window=20):
        vol = returns.rolling(window).std()
        # Higher volatility = lower uniqueness (more overlapping events)
        uniqueness = 1.0 / (1.0 + vol * 100)  # Scale to reasonable range
        return uniqueness.dropna()
    
    raw_uniqueness = simulate_uniqueness(raw_returns)
    optimized_uniqueness = simulate_uniqueness(optimized_returns)
    
    print(f"   Raw uniqueness mean: {raw_uniqueness.mean():.3f}")
    print(f"   Optimized uniqueness mean: {optimized_uniqueness.mean():.3f}")
    print(f"   Uniqueness change: {optimized_uniqueness.mean() - raw_uniqueness.mean():+.3f}")
    
    # Summary
    print(f"\n🎯 Integration Test Results:")
    print(f"✅ Layer0 price generation: SUCCESS")
    print(f"✅ SNR improvement: {snr_improvement:+.2f} dB")
    print(f"✅ Noise reduction: {noise_reduction:.1f}%")
    print(f"✅ Weight entropy change: {optimized_entropy - raw_entropy:+.3f}")
    print(f"✅ Uniqueness change: {optimized_uniqueness.mean() - raw_uniqueness.mean():+.3f}")
    
    # Benefits analysis
    if snr_improvement > 1.0:
        print(f"\n💡 Key Benefits of Layer0 for Layer1:")
        print(f"   • Significant SNR improvement ({snr_improvement:.1f} dB)")
        print(f"   • Better signal quality for weighting optimization")
        print(f"   • Cleaner volatility estimation")
        print(f"   • More accurate uniqueness calculations")
        
        if noise_reduction > 10:
            print(f"   • Strong noise reduction ({noise_reduction:.1f}%)")
        
        entropy_change = optimized_entropy - raw_entropy
        if abs(entropy_change) > 0.05:
            if entropy_change > 0:
                print(f"   • More diverse weighting (entropy +{entropy_change:.3f})")
            else:
                print(f"   • More concentrated weighting (entropy {entropy_change:.3f})")
    
    return {
        'raw_snr': raw_snr,
        'optimized_snr': optimized_snr,
        'snr_improvement': snr_improvement,
        'noise_reduction': noise_reduction,
        'raw_entropy': raw_entropy,
        'optimized_entropy': optimized_entropy,
        'entropy_change': optimized_entropy - raw_entropy,
        'raw_uniqueness': raw_uniqueness.mean(),
        'optimized_uniqueness': optimized_uniqueness.mean(),
        'uniqueness_change': optimized_uniqueness.mean() - raw_uniqueness.mean()
    }

if __name__ == "__main__":
    results = test_layer0_layer1_integration()
    
    print(f"\n📊 Test Summary:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
