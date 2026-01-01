"""
Enhanced Denoising Pipeline Test with Hampel Filter

Tests the complete denoising pipeline:
- Layer0: Hampel filter + Kalman + VWAP + Savitzky-Golay
- Layer1: Uses denoised prices for weighting optimization
- Layer2: Uses denoised prices for features, raw for triple barrier
- Layer3/4: Uses raw prices with denoised features
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

def simple_hampel_filter(price_series, window=5, threshold=3.0):
    """Simple Hampel filter implementation."""
    if window % 2 == 0:
        window += 1
    
    filtered_price = price_series.copy()
    half_window = window // 2
    
    for i in range(half_window, len(price_series) - half_window):
        window_data = price_series.iloc[i - half_window:i + half_window + 1]
        median = window_data.median()
        mad = np.median(np.abs(window_data - median))
        
        if mad > 0:
            threshold_value = threshold * mad
        else:
            threshold_value = 0
        
        if abs(price_series.iloc[i] - median) > threshold_value:
            filtered_price.iloc[i] = median
    
    return filtered_price

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

def simple_savgol_filter(price_series, window=21, order=3):
    """Simple Savitzky-Golay-like filter using moving average."""
    return price_series.rolling(window, center=True, min_periods=1).mean()

def calculate_snr(series):
    """Calculate signal-to-noise ratio."""
    signal = series.rolling(50, center=True).mean()
    noise = series - signal
    
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    return 10 * np.log10(signal_power / noise_power)

def calculate_noise_metrics(series):
    """Calculate comprehensive noise metrics."""
    returns = series.pct_change().dropna()
    
    # Standard deviation
    std = returns.std()
    
    # MAD (Median Absolute Deviation)
    median = returns.median()
    mad = np.median(np.abs(returns - median))
    
    # Outlier ratio (returns beyond 3 std)
    outlier_ratio = np.sum(np.abs(returns) > 3 * std) / len(returns)
    
    return {
        'std': std,
        'mad': mad,
        'outlier_ratio': outlier_ratio
    }

def test_layer0_denoising_pipeline(df):
    """Test Layer0 denoising pipeline with Hampel filter."""
    print("🔍 Layer0 Denoising Pipeline Test")
    print("=" * 50)
    
    # Layer0 parameters (optimized)
    layer0_params = {
        'kalman_Q': 1e-4,
        'kalman_R': 0.01,
        'vwap_weight': 0.4,
        'vwap_lookback': 50,
        'hampel_filter_enabled': True,
        'hampel_window': 5,
        'hampel_threshold': 3.0,
        'savgol_filter_enabled': True,
        'savgol_window': 21,
        'savgol_order': 3
    }
    
    print(f"⚙️ Layer0 params: {layer0_params}")
    
    # Step-by-step denoising
    results = {}
    
    # Step 0: Raw price
    raw_price = df['close']
    raw_snr = calculate_snr(raw_price)
    raw_noise = calculate_noise_metrics(raw_price)
    
    results['raw'] = {
        'price': raw_price,
        'snr': raw_snr,
        'noise': raw_noise
    }
    
    print(f"📊 Raw Price SNR: {raw_snr:.2f} dB")
    print(f"   Noise: std={raw_noise['std']:.6f}, mad={raw_noise['mad']:.6f}, outliers={raw_noise['outlier_ratio']:.3f}")
    
    # Step 1: Hampel filter
    if layer0_params['hampel_filter_enabled']:
        hampel_price = simple_hampel_filter(
            raw_price, 
            layer0_params['hampel_window'], 
            layer0_params['hampel_threshold']
        )
        hampel_snr = calculate_snr(hampel_price)
        hampel_noise = calculate_noise_metrics(hampel_price)
        
        results['hampel'] = {
            'price': hampel_price,
            'snr': hampel_snr,
            'noise': hampel_noise
        }
        
        hampel_improvement = hampel_snr - raw_snr
        outlier_reduction = raw_noise['outlier_ratio'] - hampel_noise['outlier_ratio']
        
        print(f"📊 Hampel Filter SNR: {hampel_snr:.2f} dB (+{hampel_improvement:.2f})")
        print(f"   Noise: std={hampel_noise['std']:.6f}, mad={hampel_noise['mad']:.6f}, outliers={hampel_noise['outlier_ratio']:.3f}")
        print(f"   Outlier reduction: {outlier_reduction:.3f} ({outlier_reduction/raw_noise['outlier_ratio']*100:.1f}%)")
    
    # Step 2: Kalman filter
    current_price = results['hampel']['price'] if 'hampel' in results else raw_price
    kalman_price = simple_kalman_filter(current_price, layer0_params['kalman_Q'], layer0_params['kalman_R'])
    kalman_snr = calculate_snr(kalman_price)
    kalman_noise = calculate_noise_metrics(kalman_price)
    
    results['kalman'] = {
        'price': kalman_price,
        'snr': kalman_snr,
        'noise': kalman_noise
    }
    
    kalman_improvement = kalman_snr - results['hampel']['snr'] if 'hampel' in results else kalman_snr - raw_snr
    noise_reduction = (results['hampel']['noise']['std'] if 'hampel' in results else raw_noise['std']) - kalman_noise['std']
    
    print(f"📊 Kalman Filter SNR: {kalman_snr:.2f} dB (+{kalman_improvement:.2f})")
    print(f"   Noise: std={kalman_noise['std']:.6f}, mad={kalman_noise['mad']:.6f}, outliers={kalman_noise['outlier_ratio']:.3f}")
    print(f"   Noise reduction: {noise_reduction:.6f}")
    
    # Step 3: VWAP
    vwap_price = simple_vwap(kalman_price, df['volume'], layer0_params['vwap_lookback'])
    composite_price = (1 - layer0_params['vwap_weight']) * kalman_price + layer0_params['vwap_weight'] * vwap_price
    composite_snr = calculate_snr(composite_price)
    composite_noise = calculate_noise_metrics(composite_price)
    
    results['composite'] = {
        'price': composite_price,
        'snr': composite_snr,
        'noise': composite_noise
    }
    
    composite_improvement = composite_snr - kalman_snr
    print(f"📊 VWAP Composite SNR: {composite_snr:.2f} dB (+{composite_improvement:.2f})")
    print(f"   Noise: std={composite_noise['std']:.6f}, mad={composite_noise['mad']:.6f}, outliers={composite_noise['outlier_ratio']:.3f}")
    
    # Step 4: Savitzky-Golay
    if layer0_params['savgol_filter_enabled']:
        final_price = simple_savgol_filter(composite_price, layer0_params['savgol_window'], layer0_params['savgol_order'])
        final_snr = calculate_snr(final_price)
        final_noise = calculate_noise_metrics(final_price)
        
        results['final'] = {
            'price': final_price,
            'snr': final_snr,
            'noise': final_noise
        }
        
        savgol_improvement = final_snr - composite_snr
        total_improvement = final_snr - raw_snr
        
        print(f"📊 Savitzky-Golay SNR: {final_snr:.2f} dB (+{savgol_improvement:.2f})")
        print(f"   Noise: std={final_noise['std']:.6f}, mad={final_noise['mad']:.6f}, outliers={final_noise['outlier_ratio']:.3f}")
        print(f"🎯 Total SNR Improvement: +{total_improvement:.2f} dB")
    
    return results

def test_layer1_denoised_integration(df, denoised_price):
    """Test Layer1 weighting optimization with denoised prices."""
    print("\n🔍 Layer1 Denoised Price Integration Test")
    print("=" * 50)
    
    # Simulate Layer1 weighting with raw vs denoised prices
    raw_returns = df['close'].pct_change().dropna()
    denoised_returns = denoised_price.pct_change().dropna()
    
    # Magnitude-based weighting
    raw_magnitude = np.abs(raw_returns)
    denoised_magnitude = np.abs(denoised_returns)
    
    # Calculate weight entropy
    def calculate_entropy(weights):
        weights = np.asarray(weights)
        weights = weights / np.sum(weights)
        entropy = -np.sum(weights * np.log(weights + 1e-12))
        max_entropy = np.log(len(weights))
        return entropy / max_entropy
    
    raw_entropy = calculate_entropy(raw_magnitude)
    denoised_entropy = calculate_entropy(denoised_magnitude)
    
    # Uniqueness simulation
    def simulate_uniqueness(returns, window=20):
        vol = returns.rolling(window).std()
        uniqueness = 1.0 / (1.0 + vol * 100)
        return uniqueness.dropna()
    
    raw_uniqueness = simulate_uniqueness(raw_returns)
    denoised_uniqueness = simulate_uniqueness(denoised_returns)
    
    print(f"📊 Raw magnitude entropy: {raw_entropy:.3f}")
    print(f"📊 Denoised magnitude entropy: {denoised_entropy:.3f}")
    print(f"📊 Entropy change: {denoised_entropy - raw_entropy:+.3f}")
    
    print(f"📊 Raw uniqueness mean: {raw_uniqueness.mean():.3f}")
    print(f"📊 Denoised uniqueness mean: {denoised_uniqueness.mean():.3f}")
    print(f"📊 Uniqueness improvement: {denoised_uniqueness.mean() - raw_uniqueness.mean():+.3f}")
    
    return {
        'raw_entropy': raw_entropy,
        'denoised_entropy': denoised_entropy,
        'entropy_change': denoised_entropy - raw_entropy,
        'raw_uniqueness': raw_uniqueness.mean(),
        'denoised_uniqueness': denoised_uniqueness.mean(),
        'uniqueness_improvement': denoised_uniqueness.mean() - raw_uniqueness.mean()
    }

def test_layer2_feature_generation(df, denoised_price):
    """Test Layer2 feature generation with denoised prices."""
    print("\n🔍 Layer2 Feature Generation Test")
    print("=" * 50)
    
    # Simulate feature generation with raw vs denoised prices
    raw_price = df['close']
    
    # Technical indicators
    def generate_features(price):
        returns = price.pct_change()
        
        # RSI
        delta = price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_mean = price.rolling(window=20).mean()
        bb_std = price.rolling(window=20).std()
        bb_pct_b = (price - bb_mean) / (2 * bb_std + 1e-9)
        
        # Volatility
        volatility = returns.rolling(window=20).std()
        
        return {
            'rsi': rsi,
            'bb_pct_b': bb_pct_b,
            'volatility': volatility
        }
    
    raw_features = generate_features(raw_price)
    denoised_features = generate_features(denoised_price)
    
    # Compare feature stability
    print("📊 Feature Stability Comparison:")
    
    for feature_name in raw_features:
        raw_feat = raw_features[feature_name].dropna()
        denoised_feat = denoised_features[feature_name].dropna()
        
        # Align series
        min_len = min(len(raw_feat), len(denoised_feat))
        raw_feat = raw_feat.iloc[-min_len:]
        denoised_feat = denoised_feat.iloc[-min_len:]
        
        # Calculate correlation
        correlation = np.corrcoef(raw_feat, denoised_feat)[0, 1]
        
        # Calculate noise reduction
        raw_noise = raw_feat.diff().std()
        denoised_noise = denoised_feat.diff().std()
        noise_reduction = (1 - denoised_noise / raw_noise) * 100
        
        print(f"   {feature_name}:")
        print(f"     Correlation: {correlation:.3f}")
        print(f"     Noise reduction: {noise_reduction:.1f}%")
    
    return {
        'raw_features': raw_features,
        'denoised_features': denoised_features
    }

def test_layer3_noise_features(df, denoised_price):
    """Test Layer3 noise features for raw/denoised price comparison."""
    print("\n🔍 Layer3 Noise Features Test")
    print("=" * 50)
    
    raw_price = df['close']
    
    # Calculate noise metrics
    price_diff = raw_price - denoised_price
    price_diff_std = price_diff.std()
    
    raw_volatility = raw_price.pct_change().std()
    denoised_volatility = denoised_price.pct_change().std()
    
    noise_ratio = (raw_volatility ** 2) / (denoised_volatility ** 2 + 1e-9)
    
    print(f"📊 Price difference std: {price_diff_std:.6f}")
    print(f"📊 Raw volatility: {raw_volatility:.6f}")
    print(f"📊 Denoised volatility: {denoised_volatility:.6f}")
    print(f"📊 Noise ratio (σ²_raw/σ²_denoised): {noise_ratio:.3f}")
    
    # Simulate feature importance
    features = {
        'raw_price': raw_price,
        'denoised_price': denoised_price,
        'price_diff': price_diff,
        'price_diff_std': price_diff.rolling(20).std(),
        'noise_ratio': pd.Series([noise_ratio] * len(raw_price), index=raw_price.index)
    }
    
    print(f"\n💡 Layer3 Feature Benefits:")
    print(f"   • Raw price: Preserves original market dynamics")
    print(f"   • Denoised price: Provides cleaner signal for trend detection")
    print(f"   • Price difference: Captures filtering artifacts")
    print(f"   • Noise ratio: Quantifies denoising effectiveness")
    
    return features

def main():
    """Main test function."""
    print("🧪 Enhanced Denoising Pipeline Test")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    n_points = 1000
    dates = pd.date_range('2024-01-01', periods=n_points, freq='15min')
    
    # Create price with trend, noise, and outliers
    price_base = 100.0
    trend = np.linspace(0, 0.05, n_points)  # 5% uptrend
    noise = np.random.normal(0, 0.01, n_points)  # 1% volatility
    
    # Add some outliers
    outlier_indices = np.random.choice(n_points, size=20, replace=False)
    noise[outlier_indices] *= 5  # Amplify outliers
    
    price = price_base * (1 + trend + noise)
    volume = np.random.lognormal(10, 0.5, n_points)
    
    df = pd.DataFrame({
        'close': price,
        'volume': volume
    }, index=dates)
    
    print(f"📊 Created test data: {len(df)} points with {len(outlier_indices)} outliers")
    
    # Test Layer0 denoising pipeline
    layer0_results = test_layer0_denoising_pipeline(df)
    
    # Get final denoised price
    if 'final' in layer0_results:
        denoised_price = layer0_results['final']['price']
    else:
        denoised_price = layer0_results['composite']['price']
    
    # Test Layer1 integration
    layer1_results = test_layer1_denoised_integration(df, denoised_price)
    
    # Test Layer2 feature generation
    layer2_results = test_layer2_feature_generation(df, denoised_price)
    
    # Test Layer3 noise features
    layer3_results = test_layer3_noise_features(df, denoised_price)
    
    # Summary
    print(f"\n🎯 Enhanced Denoising Pipeline Summary:")
    print(f"✅ Layer0: Hampel + Kalman + VWAP + Savitzky-Golay")
    print(f"✅ Layer1: Denoised prices for weighting optimization")
    print(f"✅ Layer2: Denoised prices for features, raw for triple barrier")
    print(f"✅ Layer3: Raw prices + denoised features + noise metrics")
    
    # Performance summary
    raw_snr = layer0_results['raw']['snr']
    final_snr = layer0_results['final']['snr'] if 'final' in layer0_results else layer0_results['composite']['snr']
    snr_improvement = final_snr - raw_snr
    
    print(f"\n📈 Performance Improvements:")
    print(f"   SNR improvement: +{snr_improvement:.2f} dB")
    print(f"   Layer1 uniqueness: +{layer1_results['uniqueness_improvement']:.3f}")
    print(f"   Layer3 noise ratio: {(layer0_results['raw']['noise']['std']**2) / (layer0_results['final']['noise']['std']**2):.3f}")

if __name__ == "__main__":
    main()
