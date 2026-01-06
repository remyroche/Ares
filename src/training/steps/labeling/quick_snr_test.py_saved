"""
Quick SNR Validation Test - Immediate Proof of Enhancement

This script provides a fast way to validate SNR improvements without
complex setup. Run this to see immediate evidence that enhanced filtering works.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def quick_snr_test() -> Dict[str, Any]:
    """
    Quick SNR test with synthetic data to prove enhancement works.
    
    Returns:
        Test results showing SNR improvement
    """
    print("🧪 Running Quick SNR Validation Test...")
    
    # Generate test data with known signal + noise
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
    
    # Test baseline vs enhanced filtering
    try:
        from .unified_price_layer2 import generate_unified_layer2_price
        
        # Baseline: Simple Kalman + VWAP
        baseline_params = {
            'kalman_Q': 1e-4,
            'kalman_R': 0.01,
            'vwap_weight': 0.4,
            'vwap_lookback': 50,
            'median_filter_enabled': False,
            'adaptive_kalman_enabled': False,
            'robust_vwap_enabled': False
        }
        
        # Enhanced: All filters enabled
        enhanced_params = {
            'kalman_Q': 1e-4,
            'kalman_R': 0.01,
            'vwap_weight': 0.4,
            'vwap_lookback': 50,
            'median_filter_enabled': True,
            'median_window': 5,
            'adaptive_kalman_enabled': True,
            'robust_vwap_enabled': True
        }
        
        # Generate filtered prices
        baseline_price = generate_unified_layer2_price(df, baseline_params)
        enhanced_price = generate_unified_layer2_price(df, enhanced_params)
        
        # Calculate SNR for both
        def calculate_snr(series):
            # Signal: low-frequency component
            signal_component = series.rolling(50, center=True).mean()
            # Noise: high-frequency component
            noise_component = series - signal_component
            
            signal_power = np.mean(signal_component ** 2)
            noise_power = np.mean(noise_component ** 2)
            
            if noise_power == 0:
                return float('inf')
            
            return 10 * np.log10(signal_power / noise_power)
        
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
            'sample_size': n_points
        }
        
        # Print results
        print(f"\n📊 Quick SNR Test Results:")
        print(f"   Baseline SNR: {baseline_snr:.2f} dB")
        print(f"   Enhanced SNR: {enhanced_snr:.2f} dB")
        print(f"   SNR Improvement: {results['snr_improvement']:.2f} dB")
        print(f"   Noise Reduction: {noise_reduction:.1f}%")
        
        # Visual comparison
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(noisy_price, alpha=0.5, label='Noisy Price', color='gray')
        plt.plot(baseline_price, label='Baseline Filtered', color='blue', linewidth=2)
        plt.plot(enhanced_price, label='Enhanced Filtered', color='red', linewidth=2)
        plt.title('Price Filtering Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(baseline_price - enhanced_price, label='Difference (Baseline - Enhanced)', color='purple')
        plt.title('Filtering Difference')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outcomes/quick_snr_validation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Visualization saved to: outcomes/quick_snr_validation.png")
        
        # Validation check
        if results['snr_improvement'] > 0.5:  # At least 0.5 dB improvement
            print(f"   ✅ VALIDATION PASSED: Significant SNR improvement detected!")
        else:
            print(f"   ❌ VALIDATION FAILED: Minimal SNR improvement")
        
        return results
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return {'error': str(e)}

def run_layer2_context_snr_test() -> Dict[str, Any]:
    """
    Test SNR improvement in Layer2 context generation.
    
    Returns:
        Context SNR test results
    """
    print("\n🎯 Running Layer2 Context SNR Test...")
    
    try:
        from .orthogonal_label_generation import VolatilityCusumEvents, VolumeCusumEvents
        
        # Generate test data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=1000, freq='1H')
        
        # Create price with volatility regimes
        base_price = 100
        regime_volatility = np.concatenate([
            np.full(250, 0.5),   # Low volatility
            np.full(250, 2.0),   # High volatility  
            np.full(250, 0.8),   # Medium volatility
            np.full(250, 1.5)    # High volatility
        ])
        
        price_changes = np.random.normal(0, regime_volatility/100, 1000)
        prices = base_price * np.exp(np.cumsum(price_changes))
        
        df = pd.DataFrame({
            'close': prices,
            'volume': np.abs(np.random.normal(1000000, 300000, 1000))
        }, index=dates)
        
        # Test baseline vs enhanced generators
        baseline_vol = VolatilityCusumEvents(use_unified_price=False)
        enhanced_vol = VolatilityCusumEvents(use_unified_price=True)
        
        baseline_flow = VolumeCusumEvents(use_unified_price=False)
        enhanced_flow = VolumeCusumEvents(use_unified_price=True)
        
        # Generate context
        baseline_vol_context = baseline_vol.generate_probabilities(df)
        enhanced_vol_context = enhanced_vol.generate_probabilities(df)
        
        baseline_flow_context = baseline_flow.generate_flow_metrics(df)
        enhanced_flow_context = enhanced_flow.generate_flow_metrics(df)
        
        # Calculate SNR for context features
        def calculate_context_snr(context_df):
            snr_values = {}
            for col in context_df.columns:
                series = context_df[col].dropna()
                if len(series) > 0:
                    signal = series.rolling(20, center=True).mean()
                    noise = series - signal
                    signal_power = np.mean(signal ** 2)
                    noise_power = np.mean(noise ** 2)
                    snr = 10 * np.log10(signal_power / (noise_power + 1e-9))
                    snr_values[col] = snr
            return snr_values
        
        baseline_vol_snr = calculate_context_snr(baseline_vol_context)
        enhanced_vol_snr = calculate_context_snr(enhanced_vol_context)
        
        baseline_flow_snr = calculate_context_snr(baseline_flow_context)
        enhanced_flow_snr = calculate_context_snr(enhanced_flow_context)
        
        # Calculate improvements
        vol_improvements = {}
        for key in baseline_vol_snr:
            if key in enhanced_vol_snr:
                vol_improvements[key] = enhanced_vol_snr[key] - baseline_vol_snr[key]
        
        flow_improvements = {}
        for key in baseline_flow_snr:
            if key in enhanced_flow_snr:
                flow_improvements[key] = enhanced_flow_snr[key] - baseline_flow_snr[key]
        
        results = {
            'volatility_context_snr_improvements': vol_improvements,
            'flow_context_snr_improvements': flow_improvements,
            'avg_vol_improvement': np.mean(list(vol_improvements.values())) if vol_improvements else 0,
            'avg_flow_improvement': np.mean(list(flow_improvements.values())) if flow_improvements else 0
        }
        
        print(f"📊 Layer2 Context SNR Results:")
        print(f"   Volatility Context SNR Improvement: {results['avg_vol_improvement']:.2f} dB")
        print(f"   Flow Context SNR Improvement: {results['avg_flow_improvement']:.2f} dB")
        
        # Validation
        total_improvement = results['avg_vol_improvement'] + results['avg_flow_improvement']
        if total_improvement > 1.0:  # At least 1 dB total improvement
            print(f"   ✅ LAYER2 VALIDATION PASSED: Significant context SNR improvement!")
        else:
            print(f"   ❌ LAYER2 VALIDATION FAILED: Minimal context SNR improvement")
        
        return results
        
    except Exception as e:
        print(f"   ❌ Layer2 test failed: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    # Run quick validation
    print("🚀 Starting SNR Validation Suite...")
    
    # Test 1: Basic SNR improvement
    basic_results = quick_snr_test()
    
    # Test 2: Layer2 context SNR improvement
    context_results = run_layer2_context_snr_test()
    
    # Summary
    print(f"\n🎯 VALIDATION SUMMARY:")
    
    if 'error' not in basic_results and basic_results.get('snr_improvement', 0) > 0.5:
        print(f"   ✅ Basic Filtering: SNR improved by {basic_results['snr_improvement']:.2f} dB")
    else:
        print(f"   ❌ Basic Filtering: No significant improvement")
    
    if 'error' not in context_results and context_results.get('avg_vol_improvement', 0) + context_results.get('avg_flow_improvement', 0) > 1.0:
        print(f"   ✅ Layer2 Context: SNR improved by {context_results['avg_vol_improvement'] + context_results['avg_flow_improvement']:.2f} dB")
    else:
        print(f"   ❌ Layer2 Context: No significant improvement")
    
    print(f"\n📈 Run complete! Check outcomes/quick_snr_validation.png for visualization.")
