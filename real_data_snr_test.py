"""
Real Data SNR Validation - Test with Actual Market Data

This script tests SNR improvements using real market data to provide
more realistic validation of enhanced filtering benefits.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any
import glob

def load_real_market_data(symbol="ETHUSDT"):
    """Load real market data for SNR validation."""
    # Try to find data in standard locations
    data_files = glob.glob(f"historical_data/**/{symbol.lower()}/**/*.parquet", recursive=True)
    
    if not data_files:
        print(f"❌ No real data found for {symbol}")
        return None
    
    try:
        df = pd.read_parquet(data_files[0])
        print(f"✅ Loaded real data: {len(df)} records from {data_files[0]}")
        
        # Ensure required columns
        required_cols = ['close', 'volume']
        if not all(col in df.columns for col in required_cols):
            print(f"❌ Missing required columns: {[c for c in required_cols if c not in df.columns]}")
            return None
        
        # Use recent data for testing
        df = df.tail(5000)  # Last 5k records
        print(f"📊 Using recent {len(df)} records for testing")
        
        return df
        
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return None

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

def median_filter(series, window=5):
    """Simple median filter."""
    return series.rolling(window, center=True, min_periods=1).median()

def calculate_snr_improvement(df):
    """Calculate SNR improvement with enhanced filtering on real data."""
    print("🧪 Testing SNR Improvement on Real Market Data...")
    
    # Baseline: Simple Kalman + VWAP
    baseline_kalman = simple_kalman_filter(df['close'], Q=1e-4, R=0.01)
    baseline_vwap = simple_vwap(df['close'], df['volume'], 50)
    baseline_price = 0.6 * baseline_kalman + 0.4 * baseline_vwap
    
    # Enhanced: Multi-stage filtering
    # Stage 1: Median filter for outlier removal
    outlier_removed = median_filter(df['close'], 7)
    
    # Stage 2: Kalman with optimized parameters
    enhanced_kalman = simple_kalman_filter(outlier_removed, Q=5e-5, R=0.005)
    
    # Stage 3: VWAP
    enhanced_vwap = simple_vwap(enhanced_kalman, df['volume'], 50)
    
    # Stage 4: Composite with final median smoothing
    enhanced_composite = 0.7 * enhanced_kalman + 0.3 * enhanced_vwap
    enhanced_price = median_filter(enhanced_composite, 5)
    
    # Calculate SNR using different methods
    def calculate_snr(series):
        # Method 1: Signal/Noise power ratio
        signal = series.rolling(50, center=True).mean()
        noise = series - signal
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        return 10 * np.log10(signal_power / noise_power)
    
    def calculate_sharpe_like_snr(series):
        # Method 2: Sharpe-like SNR (mean/std)
        returns = series.pct_change().dropna()
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        return np.abs(np.mean(returns)) / np.std(returns)
    
    def calculate_trend_snr(series):
        # Method 3: Trend vs noise
        trend = series.ewm(span=100).mean()
        noise = series - trend
        trend_var = np.var(trend.dropna())
        noise_var = np.var(noise.dropna())
        
        if noise_var == 0:
            return float('inf')
        
        return 10 * np.log10(trend_var / noise_var)
    
    # Calculate SNR using all methods
    baseline_snrs = {
        'power_snr': calculate_snr(baseline_price),
        'sharpe_snr': calculate_sharpe_like_snr(baseline_price),
        'trend_snr': calculate_trend_snr(baseline_price)
    }
    
    enhanced_snrs = {
        'power_snr': calculate_snr(enhanced_price),
        'sharpe_snr': calculate_sharpe_like_snr(enhanced_price),
        'trend_snr': calculate_trend_snr(enhanced_price)
    }
    
    # Calculate improvements
    improvements = {}
    for method in baseline_snrs:
        improvement = enhanced_snrs[method] - baseline_snrs[method]
        improvements[method] = improvement
    
    # Calculate noise reduction
    baseline_noise = np.std(baseline_price.diff().rolling(5).std())
    enhanced_noise = np.std(enhanced_price.diff().rolling(5).std())
    noise_reduction = (1 - enhanced_noise / baseline_noise) * 100
    
    # Calculate smoothing effectiveness
    baseline_smoothness = 1 / (1 + np.std(baseline_price.diff().rolling(10).std()))
    enhanced_smoothness = 1 / (1 + np.std(enhanced_price.diff().rolling(10).std()))
    smoothness_improvement = (enhanced_smoothness - baseline_smoothness) * 100
    
    print(f"📊 Real Data SNR Results:")
    print(f"   Power SNR Improvement: {improvements['power_snr']:.2f} dB")
    print(f"   Sharpe SNR Improvement: {improvements['sharpe_snr']:.4f}")
    print(f"   Trend SNR Improvement: {improvements['trend_snr']:.2f} dB")
    print(f"   Noise Reduction: {noise_reduction:.1f}%")
    print(f"   Smoothness Improvement: {smoothness_improvement:.1f}%")
    
    # Visual comparison
    plt.figure(figsize=(16, 12))
    
    plt.subplot(4, 1, 1)
    plt.plot(df['close'].values[-1000:], alpha=0.3, label='Raw Price', color='gray')
    plt.plot(baseline_price.values[-1000:], label='Baseline Filtered', color='blue', linewidth=2)
    plt.plot(enhanced_price.values[-1000:], label='Enhanced Filtered', color='red', linewidth=2)
    plt.title('Real Data: Price Filtering Comparison (Last 1000 points)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 1, 2)
    plt.plot(baseline_price.values[-1000:] - enhanced_price.values[-1000:], 
             label='Difference (Baseline - Enhanced)', color='purple')
    plt.title('Real Data: Filtering Difference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 1, 3)
    baseline_returns = baseline_price.pct_change().dropna()[-1000:]
    enhanced_returns = enhanced_price.pct_change().dropna()[-1000:]
    plt.hist(baseline_returns, bins=50, alpha=0.5, label='Baseline Returns', color='blue', density=True)
    plt.hist(enhanced_returns, bins=50, alpha=0.5, label='Enhanced Returns', color='red', density=True)
    plt.title('Real Data: Return Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 1, 4)
    # Show volatility reduction
    baseline_vol = baseline_price.pct_change().rolling(20).std()[-1000:]
    enhanced_vol = enhanced_price.pct_change().rolling(20).std()[-1000:]
    plt.plot(baseline_vol.values, label='Baseline Volatility', color='blue', alpha=0.7)
    plt.plot(enhanced_vol.values, label='Enhanced Volatility', color='red', alpha=0.7)
    plt.title('Real Data: Volatility Smoothing')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outcomes/real_data_snr_validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Visualization saved to: outcomes/real_data_snr_validation.png")
    
    # Validation criteria
    total_improvement = sum(improvements.values())
    significant_methods = sum(1 for imp in improvements.values() if imp > 0)
    
    results = {
        'improvements': improvements,
        'noise_reduction': noise_reduction,
        'smoothness_improvement': smoothness_improvement,
        'total_improvement': total_improvement,
        'significant_methods': significant_methods,
        'sample_size': len(df)
    }
    
    if significant_methods >= 2 and noise_reduction > 5:
        print(f"   ✅ REAL DATA VALIDATION PASSED: Significant improvements detected!")
        return True
    else:
        print(f"   ❌ REAL DATA VALIDATION FAILED: Minimal improvements")
        return False

def test_layer2_context_improvement(df):
    """Test Layer2 context generation improvement."""
    print("\n🎯 Testing Layer2 Context Improvement...")
    
    # Generate volatility regime context
    # Baseline: raw price volatility
    baseline_vol = df['close'].pct_change().rolling(20).std()
    
    # Enhanced: filtered price volatility
    filtered_close = median_filter(simple_kalman_filter(df['close']), 5)
    enhanced_vol = filtered_close.pct_change().rolling(20).std()
    
    # Generate volume flow context
    # Baseline: raw volume pressure
    baseline_flow = (df['volume'] / df['volume'].rolling(20).mean() - 1) * np.sign(df['close'].pct_change())
    
    # Enhanced: filtered volume pressure
    enhanced_flow = (df['volume'] / df['volume'].rolling(20).mean() - 1) * np.sign(filtered_close.pct_change())
    
    # Calculate context quality metrics
    def calculate_context_snr(context_series):
        signal = context_series.rolling(50, center=True).mean()
        noise = context_series - signal
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        return 10 * np.log10(signal_power / noise_power)
    
    baseline_vol_snr = calculate_context_snr(baseline_vol.dropna())
    enhanced_vol_snr = calculate_context_snr(enhanced_vol.dropna())
    
    baseline_flow_snr = calculate_context_snr(baseline_flow.dropna())
    enhanced_flow_snr = calculate_context_snr(enhanced_flow.dropna())
    
    vol_improvement = enhanced_vol_snr - baseline_vol_snr
    flow_improvement = enhanced_flow_snr - baseline_flow_snr
    
    print(f"📊 Layer2 Context Results:")
    print(f"   Volatility Regime SNR Improvement: {vol_improvement:.2f} dB")
    print(f"   Volume Flow SNR Improvement: {flow_improvement:.2f} dB")
    
    total_context_improvement = vol_improvement + flow_improvement
    
    if total_context_improvement > 1.0:
        print(f"   ✅ LAYER2 CONTEXT PASSED: Significant quality improvement!")
        return True
    else:
        print(f"   ❌ LAYER2 CONTEXT FAILED: Minimal improvement")
        return False

if __name__ == "__main__":
    print("🚀 Starting Real Data SNR Validation Suite...")
    
    # Load real market data
    df = load_real_market_data("ETHUSDT")
    
    if df is None:
        print("❌ Cannot proceed without real data")
        exit(1)
    
    # Test 1: SNR improvement on real data
    snr_passed = calculate_snr_improvement(df)
    
    # Test 2: Layer2 context improvement
    context_passed = test_layer2_context_improvement(df)
    
    # Summary
    print(f"\n🎯 REAL DATA VALIDATION SUMMARY:")
    
    if snr_passed:
        print(f"   ✅ SNR Enhancement: Real data improvement validated")
    else:
        print(f"   ❌ SNR Enhancement: No significant improvement in real data")
    
    if context_passed:
        print(f"   ✅ Layer2 Context: Real data quality improvement validated")
    else:
        print(f"   ❌ Layer2 Context: No significant improvement in real data")
    
    if snr_passed and context_passed:
        print(f"\n🎉 REAL DATA VALIDATION PASSED!")
        print(f"   📈 Enhanced filtering provides measurable benefits with real market data")
        print(f"   🔬 Validation confirms SNR and context quality improvements")
    else:
        print(f"\n⚠️  Real data validation failed.")
        print(f"   💡 Consider adjusting filtering parameters for specific market conditions")
    
    print(f"\n📈 Check outcomes/real_data_snr_validation.png for detailed visualization.")
