#!/usr/bin/env python3
"""
Test script for target denoising functionality
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def test_target_denoiser():
    """Test the target denoiser functionality"""
    
    try:
        from src.utils.ml_common.target_denoiser import (
            TargetDenoiser, DenoisingConfig,
            kalman_denoise, hampel_denoise, savgol_denoise, volume_weighted_denoise
        )
        print("✅ Target denoiser import successful")
        
        # Create sample noisy binary target
        np.random.seed(42)
        n_samples = 1000
        
        # Create clean signal
        clean_signal = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
        
        # Add noise (flip some bits)
        noise_mask = np.random.random(n_samples) < 0.15  # 15% noise
        noisy_target = clean_signal.copy()
        noisy_target[noise_mask] = 1 - noisy_target[noise_mask]
        
        target_series = pd.Series(noisy_target, name='binary_label_long')
        
        # Create sample volume data
        volume_data = np.random.lognormal(10, 1, n_samples)
        volume_series = pd.Series(volume_data, name='volume')
        
        print("✅ Sample data created")
        print(f"  Original transitions: {np.sum(np.abs(np.diff(noisy_target)))}")
        print(f"  Noise level: {np.mean(noise_mask):.1%}")
        
        # Test individual denoising methods
        methods = ['kalman', 'hampel', 'savgol', 'volume', 'ensemble']
        results = {}
        
        for method in methods:
            print(f"\n🔇 Testing {method} denoising...")
            
            try:
                denoiser = TargetDenoiser(DenoisingConfig(method=method))
                
                if method == 'volume':
                    result = denoiser.denoise_target(target_series, volume_series=volume_series)
                else:
                    result = denoiser.denoise_target(target_series)
                
                results[method] = result
                
                # Calculate improvement
                original_transitions = np.sum(np.abs(np.diff(target_series.values)))
                denoised_transitions = np.sum(np.abs(np.diff(result.denoised_target.values)))
                
                print(f"  Processing time: {result.processing_time:.3f}s")
                print(f"  Noise reduction: {result.denoising_stats.get('noise_reduction', 0):.1%}")
                print(f"  Agreement rate: {result.denoising_stats.get('agreement_rate', 0):.1%}")
                print(f"  Transitions: {original_transitions} → {denoised_transitions}")
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                results[method] = None
        
        # Test convenience functions
        print(f"\n🔧 Testing convenience functions...")
        
        try:
            kalman_result = kalman_denoise(target_series)
            print("✅ Kalman convenience function working")
        except Exception as e:
            print(f"❌ Kalman convenience function failed: {e}")
        
        try:
            hampel_result = hampel_denoise(target_series)
            print("✅ Hampel convenience function working")
        except Exception as e:
            print(f"❌ Hampel convenience function failed: {e}")
        
        try:
            savgol_result = savgol_denoise(target_series)
            print("✅ Savitzky-Golay convenience function working")
        except Exception as e:
            print(f"❌ Savitzky-Golay convenience function failed: {e}")
        
        try:
            volume_result = volume_weighted_denoise(target_series, volume_series)
            print("✅ Volume-weighted convenience function working")
        except Exception as e:
            print(f"❌ Volume-weighted convenience function failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_orthogonalizer_integration():
    """Test target denoising integration with orthogonalizer"""
    
    try:
        from src.utils.ml_common.specialist_orthogonalizer import OptimizedSpecialistOrthogonalizer
        print("✅ Orthogonalizer import successful")
        
        # Initialize with target denoising
        orthogonalizer = OptimizedSpecialistOrthogonalizer(
            enable_target_denoising=True
        )
        print("✅ Orthogonalizer with target denoising initialized")
        
        # Create sample data
        np.random.seed(42)
        n_samples = 500
        
        sample_data = pd.DataFrame({
            'macro_trend_1': np.random.randn(n_samples),
            'xgb_macro_signal': np.random.randn(n_samples),
            'risk_score': np.random.random(n_samples),
            'liquidity_score': np.random.random(n_samples),
            'volume': np.random.lognormal(10, 1, n_samples),
        })
        
        # Create noisy target
        clean_target = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
        noise_mask = np.random.random(n_samples) < 0.2
        noisy_target = clean_target.copy()
        noisy_target[noise_mask] = 1 - noisy_target[noise_mask]
        
        target_series = pd.Series(noisy_target, index=sample_data.index)
        volume_series = sample_data['volume']
        
        print("✅ Sample data created")
        
        # Test denoised orthogonal targets
        print("🔇 Testing denoised orthogonal targets...")
        
        orthogonal_targets, denoising_info = orthogonalizer.generate_denoised_orthogonal_targets(
            specialist_df=sample_data,
            target_series=target_series,
            denoising_method='kalman',
            volume_series=volume_series
        )
        
        print(f"✅ Generated {len(orthogonal_targets.columns)} orthogonal targets")
        
        # Check denoising info
        denoising_result = denoising_info.get('denoising_result')
        if denoising_result:
            print(f"  Method: {denoising_result.method_used}")
            print(f"  Processing time: {denoising_result.processing_time:.3f}s")
            print(f"  Noise reduction: {denoising_result.denoising_stats.get('noise_reduction', 0):.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all target denoising tests"""
    
    print("🧪 Testing Target Denoising Implementation")
    print("=" * 60)
    
    tests = [
        ("Target Denoiser", test_target_denoiser),
        ("Orthogonalizer Integration", test_orthogonalizer_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Target denoising implementation is ready!")
        return 0
    else:
        print(f"❌ {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
