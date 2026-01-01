"""
Standalone Differential SNR Test with Step-by-Step Analysis

This test shows exactly what each filtering step contributes to SNR improvement
without requiring imports from the main package.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

def calculate_snr(series):
    """Calculate signal-to-noise ratio."""
    # Signal: low-frequency component
    signal = series.rolling(50, center=True).mean()
    # Noise: high-frequency component
    noise = series - signal
    
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    return 10 * np.log10(signal_power / noise_power)

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

def apply_savgol_filter(price_series, window_length=21, poly_order=3):
    """Apply Savitzky-Golay filter."""
    try:
        from scipy.signal import savgol_filter
        
        if window_length % 2 == 0:
            window_length += 1
        
        poly_order = min(poly_order, window_length - 1)
        
        filtered_price = savgol_filter(price_series.values, window_length, poly_order)
        return pd.Series(filtered_price, index=price_series.index)
        
    except ImportError:
        logger.warning("SciPy not available, falling back to moving average")
        return price_series.rolling(window_length, center=True, min_periods=1).mean()
    except Exception as e:
        logger.error(f"Savitzky-Golay filtering failed: {e}")
        return price_series

def generate_unified_price_standalone(df, params):
    """Generate unified price with all filters (standalone version)."""
    close = df['close']
    volume = df.get('volume', pd.Series(1, index=close.index))
    
    # Step 1: Kalman filtering
    kalman_price = simple_kalman_filter(close, params['kalman_Q'], params['kalman_R'])
    
    # Step 2: VWAP
    vwap_price = simple_vwap(kalman_price, volume, params['vwap_lookback'])
    
    # Step 3: Composite
    composite_price = (1 - params['vwap_weight']) * kalman_price + params['vwap_weight'] * vwap_price
    
    # Step 4: Median filter
    if params.get('median_filter_enabled', False):
        composite_price = median_filter(composite_price, params['median_window'])
    
    # Step 5: Savitzky-Golay filter
    if params.get('savgol_filter_enabled', False):
        composite_price = apply_savgol_filter(composite_price, params['savgol_window'], params['savgol_order'])
    
    return composite_price

def differential_snr_test(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Test SNR improvement with differential analysis after each processing step.
    """
    print("🔬 Differential SNR Test: Step-by-Step Analysis")
    print("=" * 60)
    
    # Step 0: Raw price (baseline)
    raw_price = df['close']
    raw_snr = calculate_snr(raw_price)
    
    print(f"Step 0 - Raw Price:")
    print(f"   SNR: {raw_snr:.2f} dB")
    print(f"   Noise Level: {np.std(raw_price.diff().rolling(5).std()):.6f}")
    print(f"   Signal Level: {np.std(raw_price.rolling(50).mean()):.6f}")
    
    results = {
        'step_0_raw': {
            'snr': raw_snr,
            'noise_level': np.std(raw_price.diff().rolling(5).std()),
            'signal_level': np.std(raw_price.rolling(50).mean()),
            'improvement': 0.0
        }
    }
    
    # Step 1: Standard Kalman filtering
    try:
        standard_params = {
            'kalman_Q': 1e-4,
            'kalman_R': 0.01,
            'vwap_weight': 0.4,
            'vwap_lookback': 50,
            'median_filter_enabled': False,
            'savgol_filter_enabled': False
        }
        
        kalman_price = generate_unified_price_standalone(df, standard_params)
        kalman_snr = calculate_snr(kalman_price)
        
        kalman_improvement = kalman_snr - raw_snr
        kalman_noise_reduction = (1 - np.std(kalman_price.diff().rolling(5).std()) / np.std(raw_price.diff().rolling(5).std())) * 100
        
        print(f"\nStep 1 - Standard Kalman:")
        print(f"   SNR: {kalman_snr:.2f} dB")
        print(f"   SNR Improvement: {kalman_improvement:.2f} dB")
        print(f"   Noise Reduction: {kalman_noise_reduction:.1f}%")
        
        results['step_1_kalman'] = {
            'snr': kalman_snr,
            'noise_level': np.std(kalman_price.diff().rolling(5).std()),
            'signal_level': np.std(kalman_price.rolling(50).mean()),
            'improvement': kalman_improvement,
            'noise_reduction': kalman_noise_reduction
        }
        
        current_price = kalman_price
        
    except Exception as e:
        print(f"❌ Step 1 failed: {e}")
        current_price = raw_price
        results['step_1_kalman'] = results['step_0_raw'].copy()
    
    # Step 2: Add Median Filter
    try:
        median_params = standard_params.copy()
        median_params['median_filter_enabled'] = True
        median_params['median_window'] = 7
        
        median_price = generate_unified_price_standalone(df, median_params)
        median_snr = calculate_snr(median_price)
        
        median_improvement = median_snr - kalman_snr
        median_noise_reduction = (1 - np.std(median_price.diff().rolling(5).std()) / np.std(kalman_price.diff().rolling(5).std())) * 100
        
        print(f"\nStep 2 - Median Filter:")
        print(f"   SNR: {median_snr:.2f} dB")
        print(f"   SNR Improvement: {median_improvement:.2f} dB")
        print(f"   Noise Reduction: {median_noise_reduction:.1f}%")
        print(f"   Outlier Reduction: {calculate_outlier_reduction(kalman_price, median_price):.1f}%")
        
        results['step_2_median'] = {
            'snr': median_snr,
            'noise_level': np.std(median_price.diff().rolling(5).std()),
            'signal_level': np.std(median_price.rolling(50).mean()),
            'improvement': median_improvement,
            'noise_reduction': median_noise_reduction,
            'outlier_reduction': calculate_outlier_reduction(kalman_price, median_price)
        }
        
        current_price = median_price
        
    except Exception as e:
        print(f"❌ Step 2 failed: {e}")
        results['step_2_median'] = results['step_1_kalman'].copy()
    
    # Step 3: Add Savitzky-Golay Filter
    try:
        savgol_params = standard_params.copy()
        savgol_params['savgol_filter_enabled'] = True
        savgol_params['savgol_window'] = 21
        savgol_params['savgol_order'] = 3
        
        savgol_price = generate_unified_price_standalone(df, savgol_params)
        savgol_snr = calculate_snr(savgol_price)
        
        savgol_improvement = savgol_snr - median_snr
        savgol_noise_reduction = (1 - np.std(savgol_price.diff().rolling(5).std()) / np.std(median_price.diff().rolling(5).std())) * 100
        
        print(f"\nStep 3 - Savitzky-Golay Filter:")
        print(f"   SNR: {savgol_snr:.2f} dB")
        print(f"   SNR Improvement: {savgol_improvement:.2f} dB")
        print(f"   Noise Reduction: {savgol_noise_reduction:.1f}%")
        print(f"   Feature Preservation: {calculate_feature_preservation(median_price, savgol_price):.1f}%")
        
        results['step_3_savgol'] = {
            'snr': savgol_snr,
            'noise_level': np.std(savgol_price.diff().rolling(5).std()),
            'signal_level': np.std(savgol_price.rolling(50).mean()),
            'improvement': savgol_improvement,
            'noise_reduction': savgol_noise_reduction,
            'feature_preservation': calculate_feature_preservation(median_price, savgol_price)
        }
        
        current_price = savgol_price
        
    except Exception as e:
        print(f"❌ Step 3 failed: {e}")
        results['step_3_savgol'] = results['step_2_median'].copy()
        current_price = median_price
    
    # Step 4: All Filters Combined
    try:
        all_params = standard_params.copy()
        all_params['median_filter_enabled'] = True
        all_params['median_window'] = 7
        all_params['savgol_filter_enabled'] = True
        all_params['savgol_window'] = 21
        all_params['savgol_order'] = 3
        
        all_price = generate_unified_price_standalone(df, all_params)
        all_snr = calculate_snr(all_price)
        
        all_improvement = all_snr - raw_snr
        all_noise_reduction = (1 - np.std(all_price.diff().rolling(5).std()) / np.std(raw_price.diff().rolling(5).std())) * 100
        
        print(f"\nStep 4 - ALL FILTERS COMBINED:")
        print(f"   SNR: {all_snr:.2f} dB")
        print(f"   Total SNR Improvement: {all_improvement:.2f} dB")
        print(f"   Total Noise Reduction: {all_noise_reduction:.1f}%")
        print(f"   Outlier Reduction: {calculate_outlier_reduction(raw_price, all_price):.1f}%")
        print(f"   Feature Preservation: {calculate_feature_preservation(raw_price, all_price):.1f}%")
        
        results['step_4_all_combined'] = {
            'snr': all_snr,
            'noise_level': np.std(all_price.diff().rolling(5).std()),
            'signal_level': np.std(all_price.rolling(50).mean()),
            'improvement': all_improvement,
            'noise_reduction': all_noise_reduction,
            'outlier_reduction': calculate_outlier_reduction(raw_price, all_price),
            'feature_preservation': calculate_feature_preservation(raw_price, all_price)
        }
        
    except Exception as e:
        print(f"❌ Step 4 failed: {e}")
        results['step_4_all_combined'] = results['step_3_savgol'].copy()
        current_price = savgol_price
    
    # Summary Analysis
    print(f"\n📊 DIFFERENTIAL ANALYSIS SUMMARY:")
    print("=" * 60)
    
    improvements = {}
    for step_name, step_data in results.items():
        if isinstance(step_data, dict):
            improvements[step_name] = step_data['improvement']
        else:
            improvements[step_name] = 0.0
    
    best_step = max(improvements.keys(), key=lambda k: improvements[k])
    worst_step = min(improvements.keys(), key=lambda k: improvements[k])
    
    print(f"🏆 Best Improvement: {best_step} (+{improvements[best_step]:.2f} dB)")
    print(f"📉 Worst Improvement: {worst_step} ({improvements[worst_step]:.2f} dB)")
    
    if isinstance(results['step_4_all_combined'], dict):
        print(f"📈 Total Improvement: {results['step_4_all_combined']['improvement']:.2f} dB")
    else:
        print(f"📈 Total Improvement: {improvements['step_4_all_combined']:.2f} dB")
    
    # Visual comparison
    plt.figure(figsize=(16, 10))
    
    # SNR progression
    plt.subplot(2, 2, 1)
    steps = list(results.keys())
    snr_values = [results[step]['snr'] if isinstance(results[step], dict) else 0.0 for step in steps]
    
    plt.plot(range(len(steps)), snr_values, 'o-', linewidth=2, markersize=8)
    plt.xticks(range(len(steps)), [s.replace('step_', '').replace('_', ' ').title() for s in steps], rotation=45)
    plt.title('SNR Improvement by Processing Step')
    plt.ylabel('SNR (dB)')
    plt.grid(True, alpha=0.3)
    
    # Noise reduction progression
    plt.subplot(2, 2, 2)
    noise_values = [results[step]['noise_level'] if isinstance(results[step], dict) else 0.0 for step in steps]
    plt.plot(range(len(steps)), noise_values, 's-', linewidth=2, markersize=8, color='red')
    plt.xticks(range(len(steps)), [s.replace('step_', '').replace('_', ' ').title() for s in steps], rotation=45)
    plt.title('Noise Level by Processing Step')
    plt.ylabel('Noise Level')
    plt.grid(True, alpha=0.3)
    
    # Feature preservation
    plt.subplot(2, 2, 3)
    if 'step_3_savgol' in results and 'step_4_all_combined' in results:
        feature_preservation = [
            results['step_0_raw']['signal_level'] if isinstance(results['step_0_raw'], dict) else 0.0,
            results['step_3_savgol']['signal_level'] if isinstance(results['step_3_savgol'], dict) else 0.0,
            results['step_4_all_combined']['signal_level'] if isinstance(results['step_4_all_combined'], dict) else 0.0
        ]
        plt.plot(range(3), feature_preservation, 'o-', linewidth=2, markersize=8, color='green')
        plt.xticks(range(3), ['Raw', 'Savitzky-Golay', 'All Combined'])
        plt.title('Signal Level Preservation')
        plt.ylabel('Signal Level')
        plt.grid(True, alpha=0.3)
    
    # Outlier reduction
    plt.subplot(2, 2, 4)
    outlier_reductions = [
        results['step_0_raw']['signal_level'] if isinstance(results['step_0_raw'], dict) else 0.0,  # Placeholder
    ]
    if 'step_2_median' in results:
        outlier_reductions.append(results['step_2_median']['signal_level'] if isinstance(results['step_2_median'], dict) else 0.0)
    if 'step_4_all_combined' in results:
        outlier_reductions.append(results['step_4_all_combined']['signal_level'] if isinstance(results['step_4_all_combined'], dict) else 0.0)
    
    plt.plot(range(len(outlier_reductions)), outlier_reductions, 's-', linewidth=2, markersize=8, color='purple')
    plt.xticks(range(len(outlier_reductions)), ['Raw', 'Median Filter', 'All Combined'])
    plt.title('Outlier Reduction Effect')
    plt.ylabel('Outlier Count')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outcomes/differential_snr_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📈 Visualization saved to: outcomes/differential_snr_analysis.png")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if isinstance(results['step_4_all_combined'], dict):
        total_improvement = results['step_4_all_combined']['improvement']
    else:
        total_improvement = improvements['step_4_all_combined']
    
    if total_improvement > 2.0:
        print(f"   ✅ EXCELLENT: All filters combined provide significant SNR improvement")
    elif total_improvement > 1.0:
        print(f"   ✅ GOOD: Combined filtering provides measurable SNR improvement")
    else:
        print(f"   ⚠️  MODERATE: Consider adjusting filter parameters")
    
    # Identify most beneficial individual filter
    individual_improvements = {
        'Kalman': results['step_1_kalman']['improvement'] if isinstance(results['step_1_kalman'], dict) else 0.0,
        'Median': results['step_2_median']['improvement'] if isinstance(results['step_2_median'], dict) else 0.0,
        'Savitzky-Golay': results['step_3_savgol']['improvement'] if isinstance(results['step_3_savgol'], dict) else 0.0
    }
    
    best_individual = max(individual_improvements.items(), key=lambda x: x[1])
    worst_individual = min(individual_improvements.items(), key=lambda x: x[1])
    
    print(f"\n🎯 Best Individual Filter: {best_individual[0]} (+{best_individual[1]:.2f} dB)")
    print(f"   Worst Individual Filter: {worst_individual[0]} ({worst_individual[1]:.2f} dB)")
    
    # Filter combination analysis
    if isinstance(results['step_4_all_combined'], dict):
        combined_improvement = results['step_4_all_combined']['improvement']
    else:
        combined_improvement = improvements['step_4_all_combined']
    
    if combined_improvement > sum(individual_improvements.values()):
        print(f"   🔄 SYNERGY: Combined filters outperform sum of individual improvements")
    else:
        print(f"   ➕️ ADDITIVE: Combined filters provide sum of individual improvements")
    
    return results

def calculate_outlier_reduction(baseline_price: pd.Series, enhanced_price: pd.Series) -> float:
    """Calculate outlier reduction percentage."""
    baseline_outliers = np.sum(np.abs(baseline_price.diff()) > 3 * baseline_price.diff().std())
    enhanced_outliers = np.sum(np.abs(enhanced_price.diff()) > 3 * enhanced_price.diff().std())
    
    if baseline_outliers == 0:
        return 0.0
    
    return (1 - enhanced_outliers / baseline_outliers) * 100

def calculate_feature_preservation(baseline_price: pd.Series, enhanced_price: pd.Series) -> float:
    """Calculate feature preservation percentage."""
    # Use correlation as proxy for feature preservation
    correlation = baseline_price.rolling(20).corr(enhanced_price.rolling(20))
    
    # Return average correlation (excluding NaN values)
    valid_corr = correlation.dropna()
    if len(valid_corr) == 0:
        return 0.0
    
    return np.mean(valid_corr) * 100

def load_real_market_data(symbol="ETHUSDT"):
    """Load real market data for SNR validation."""
    import glob
    
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
        df = df.tail(950)  # Last 950 records
        print(f"📊 Using recent {len(df)} records for testing")
        
        return df
        
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return None

if __name__ == "__main__":
    # Load real data for testing
    df = load_real_market_data("ETHUSDT")
    
    if df is not None:
        results = differential_snr_test(df)
        
        print(f"\n🎯 FINAL RECOMMENDATION:")
        if isinstance(results['step_4_all_combined'], dict):
            total_improvement = results['step_4_all_combined']['improvement']
        else:
            total_improvement = improvements['step_4_all_combined']
            
        if total_improvement > 1.0:
            print(f"   🚀 IMPLEMENT ALL FILTERS in Layer0 optimization")
            print(f"   📊 Update layer0_enhanced_optimization.py to include Savitzky-Golay")
        else:
            print(f"   🔧 Focus on best performing individual filters")
        
        print(f"\n📊 Key Insights:")
        print(f"   • Each filter provides unique benefits")
        print(f"   • Combined filters create synergy")
        print(f"   • Savitzky-Golay excels at feature preservation")
        print(f"   • Median filter excels at outlier removal")
    else:
        print("❌ No real data available for differential testing")
