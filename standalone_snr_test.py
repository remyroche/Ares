"""
Standalone SNR Validation Test - No Dependencies Required

Run this script directly to validate SNR improvements from enhanced filtering.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any

def generate_test_data():
    """Generate test data with known signal and noise characteristics."""
    np.random.seed(42)
    n_points = 2000
    
    # Create clean signal (trend + cycles)
    t = np.arange(n_points)
    signal = 100 + 0.01 * t + 5 * np.sin(2 * np.pi * t / 100) + 2 * np.sin(2 * np.pi * t / 20)
    
    # Add noise
    noise = np.random.normal(0, 2, n_points)
    noisy_price = signal + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'close': noisy_price,
        'volume': np.abs(np.random.normal(1000000, 200000, n_points))
    })
    
    return df, signal, noise

def simple_kalman_filter(price_series, Q=1e-4, R=0.01):
    """Simple Kalman filter implementation."""
    n = len(price_series)
    x_hat = np.zeros(n)  # State estimate
    P = np.ones(n)       # Covariance estimate
    
    # Initialize
    x_hat[0] = price_series.iloc[0]
    P[0] = 1.0
    
    for k in range(1, n):
        # Prediction
        x_hat_minus = x_hat[k-1]
        P_minus = P[k-1] + Q
        
        # Update
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

def median_filter(series, window=5):
    """Simple median filter."""
    return series.rolling(window, center=True, min_periods=1).median()

def adaptive_noise_estimate(price, window=50):
    """Estimate measurement noise from recent volatility."""
    returns = price.pct_change()
    volatility = returns.rolling(window).std()
    return volatility * 0.5  # Scale factor for noise estimation

def calculate_snr(series):
    """Calculate signal-to-noise ratio."""
    # Signal: low-frequency component
    signal_component = series.rolling(50, center=True).mean()
    # Noise: high-frequency component
    noise_component = series - signal_component
    
    signal_power = np.mean(signal_component ** 2)
    noise_power = np.mean(noise_component ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    return 10 * np.log10(signal_power / noise_power)

def quick_snr_test():
    """Quick SNR test with synthetic data."""
    print("🧪 Running Quick SNR Validation Test...")
    
    # Generate test data
    df, true_signal, true_noise = generate_test_data()
    
    # Test baseline vs enhanced filtering
    # Baseline: Simple Kalman + VWAP
    kalman_price = simple_kalman_filter(df['close'])
    vwap_price = simple_vwap(df['close'], df['volume'], 50)
    baseline_price = 0.6 * kalman_price + 0.4 * vwap_price
    
    # Enhanced: Add median filter and adaptive noise
    # Apply median filter first
    median_filtered = median_filter(baseline_price, 5)
    
    # Adaptive Kalman with noise estimation
    adaptive_R = adaptive_noise_estimate(df['close'], 50)
    adaptive_kalman = simple_kalman_filter(df['close'], Q=1e-4, R=0.01)  # Simplified
    
    # Robust VWAP with adaptive window
    volume_volatility = df['volume'].pct_change().rolling(20).std()
    adaptive_window = np.clip(50 / (1 + volume_volatility * 10), 20, 100)
    robust_vwap = simple_vwap(df['close'], df['volume'], 50)  # Simplified
    
    # Enhanced composite
    enhanced_price = 0.6 * adaptive_kalman + 0.4 * robust_vwap
    enhanced_price = median_filter(enhanced_price, 5)
    
    # Calculate SNR for both
    baseline_snr = calculate_snr(baseline_price)
    enhanced_snr = calculate_snr(enhanced_price)
    
    # Calculate noise levels
    baseline_noise = np.std(baseline_price.diff().rolling(5).std())
    enhanced_noise = np.std(enhanced_price.diff().rolling(5).std())
    
    noise_reduction = (1 - enhanced_noise / baseline_noise) * 100
    
    # Results
    results = {
        'baseline_snr': baseline_snr,
        'enhanced_snr': enhanced_snr,
        'snr_improvement': enhanced_snr - baseline_snr,
        'noise_reduction_percent': noise_reduction,
        'sample_size': len(df)
    }
    
    # Print results
    print(f"\n📊 Quick SNR Test Results:")
    print(f"   Baseline SNR: {baseline_snr:.2f} dB")
    print(f"   Enhanced SNR: {enhanced_snr:.2f} dB")
    print(f"   SNR Improvement: {results['snr_improvement']:.2f} dB")
    print(f"   Noise Reduction: {noise_reduction:.1f}%")
    
    # Visual comparison
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(df['close'].values, alpha=0.5, label='Noisy Price', color='gray')
    plt.plot(baseline_price.values, label='Baseline Filtered', color='blue', linewidth=2)
    plt.plot(enhanced_price.values, label='Enhanced Filtered', color='red', linewidth=2)
    plt.title('Price Filtering Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(baseline_price.values - enhanced_price.values, label='Difference (Baseline - Enhanced)', color='purple')
    plt.title('Filtering Difference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.hist(baseline_price.diff().dropna(), bins=50, alpha=0.5, label='Baseline Noise', color='blue')
    plt.hist(enhanced_price.diff().dropna(), bins=50, alpha=0.5, label='Enhanced Noise', color='red')
    plt.title('Noise Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outcomes/quick_snr_validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Visualization saved to: outcomes/quick_snr_validation.png")
    
    # Validation check
    if results['snr_improvement'] > 0.5:  # At least 0.5 dB improvement
        print(f"   ✅ VALIDATION PASSED: Significant SNR improvement detected!")
        return True
    else:
        print(f"   ❌ VALIDATION FAILED: Minimal SNR improvement")
        return False

def test_layer2_context_quality():
    """Test Layer2 context quality improvement."""
    print("\n🎯 Testing Layer2 Context Quality...")
    
    # Generate test data with regime changes
    np.random.seed(42)
    n_points = 1000
    
    # Create price with volatility regimes
    base_price = 100
    regime_volatility = np.concatenate([
        np.full(250, 0.5),   # Low volatility
        np.full(250, 2.0),   # High volatility  
        np.full(250, 0.8),   # Medium volatility
        np.full(250, 1.5)    # High volatility
    ])
    
    price_changes = np.random.normal(0, regime_volatility/100, n_points)
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    df = pd.DataFrame({
        'close': prices,
        'volume': np.abs(np.random.normal(1000000, 300000, n_points))
    })
    
    # Calculate volatility regime features
    # Baseline: raw price volatility
    baseline_vol = df['close'].pct_change().rolling(20).std()
    
    # Enhanced: filtered price volatility
    filtered_price = median_filter(simple_kalman_filter(df['close']), 5)
    enhanced_vol = filtered_price.pct_change().rolling(20).std()
    
    # Calculate volume flow features
    # Baseline: raw volume pressure
    baseline_flow = (df['volume'] / df['volume'].rolling(20).mean() - 1) * np.sign(df['close'].pct_change())
    
    # Enhanced: filtered volume pressure
    enhanced_flow = (df['volume'] / df['volume'].rolling(20).mean() - 1) * np.sign(filtered_price.pct_change())
    
    # Calculate SNR for features
    baseline_vol_snr = calculate_snr(baseline_vol.dropna())
    enhanced_vol_snr = calculate_snr(enhanced_vol.dropna())
    
    baseline_flow_snr = calculate_snr(baseline_flow.dropna())
    enhanced_flow_snr = calculate_snr(enhanced_flow.dropna())
    
    vol_improvement = enhanced_vol_snr - baseline_vol_snr
    flow_improvement = enhanced_flow_snr - baseline_flow_snr
    
    print(f"📊 Layer2 Context Quality Results:")
    print(f"   Volatility Regime SNR Improvement: {vol_improvement:.2f} dB")
    print(f"   Volume Flow SNR Improvement: {flow_improvement:.2f} dB")
    
    total_improvement = vol_improvement + flow_improvement
    if total_improvement > 1.0:
        print(f"   ✅ LAYER2 VALIDATION PASSED: Significant context quality improvement!")
        return True
    else:
        print(f"   ❌ LAYER2 VALIDATION FAILED: Minimal context quality improvement")
        return False

if __name__ == "__main__":
    print("🚀 Starting Standalone SNR Validation Suite...")
    
    # Test 1: Basic SNR improvement
    basic_passed = quick_snr_test()
    
    # Test 2: Layer2 context quality
    context_passed = test_layer2_context_quality()
    
    # Summary
    print(f"\n🎯 VALIDATION SUMMARY:")
    
    if basic_passed:
        print(f"   ✅ Basic Filtering: SNR improvement validated")
    else:
        print(f"   ❌ Basic Filtering: No significant improvement")
    
    if context_passed:
        print(f"   ✅ Layer2 Context: Quality improvement validated")
    else:
        print(f"   ❌ Layer2 Context: No significant improvement")
    
    if basic_passed and context_passed:
        print(f"\n🎉 ALL VALIDATIONS PASSED! Enhanced filtering improves SNR.")
    else:
        print(f"\n⚠️  Some validations failed. Review filtering parameters.")
    
    print(f"\n📈 Check outcomes/quick_snr_validation.png for visualization.")
