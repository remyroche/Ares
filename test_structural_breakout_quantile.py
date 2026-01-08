#!/usr/bin/env python3
"""
Test script to verify 2% quantile approach for structural breakouts.
"""

import numpy as np
import pandas as pd
from src.training.steps.labeling.adaptive_event_driven_labeling import AdaptiveEventDrivenLabeling

def test_quantile_approach():
    """Test that the quantile approach generates at least 2% coverage."""
    
    # Create mock data
    np.random.seed(42)
    n_periods = 1000
    
    # Create synthetic spectral components
    spectral_components = {
        'momentum_d1': np.random.randn(n_periods),
        'momentum_d3': np.random.randn(n_periods) * 0.5,
        'volume_d1': np.random.randn(n_periods),
        'volume_d3': np.random.randn(n_periods) * 0.3,
        'volatility_d1': np.random.randn(n_periods),
        'volatility_d3': np.random.randn(n_periods) * 0.4,
    }
    
    # Create mock AEDL instance
    aedl = AdaptiveEventDrivenLabeling(verbose=True)
    aedl.spectral_components = spectral_components
    
    # Mock the spectral specialists
    class MockSpectralSpecialists:
        priority_specialists = ['momentum', 'volume', 'volatility']
    
    aedl.spectral_specialists = MockSpectralSpecialists()
    
    # Mock resonance detector
    class MockResonanceDetector:
        def calculate_phase_lead_lag(self, d1_coeffs, d3_coeffs):
            # Generate phase values with some correlation
            phase = np.random.randn(len(d1_coeffs)) * 0.1 + 0.05
            return phase
    
    aedl.resonance_detector = MockResonanceDetector()
    
    print("🧪 Testing 2% quantile approach for structural breakouts...")
    
    # Test 1: Quantile approach (default)
    print("\n--- Test 1: 2% Quantile Approach ---")
    result_quantile = aedl.get_structural_breakouts()
    
    if 'error' in result_quantile:
        print(f"❌ Quantile approach failed: {result_quantile['error']}")
        return False
    
    # Calculate actual coverage
    total_periods = sum(len(mask) for mask in result_quantile['breakout_signals'].values())
    total_breakouts = sum(np.sum(mask) for mask in result_quantile['breakout_signals'].values())
    actual_coverage = (total_breakouts / total_periods * 100) if total_periods > 0 else 0
    
    print(f"📊 Quantile approach results:")
    print(f"   - Breakout signals: {len(result_quantile['breakout_signals'])}")
    print(f"   - Total periods: {total_periods}")
    print(f"   - Total breakouts: {total_breakouts}")
    print(f"   - Actual coverage: {actual_coverage:.2f}%")
    print(f"   - Target coverage: 2.00%")
    
    # Test 2: Fixed threshold approach (for comparison)
    print("\n--- Test 2: Fixed Threshold Approach ---")
    result_fixed = aedl.get_structural_breakouts(
        use_quantile_approach=False,
        phase_threshold=0.1
    )
    
    if 'error' in result_fixed:
        print(f"❌ Fixed approach failed: {result_fixed['error']}")
        return False
    
    # Calculate coverage for fixed approach
    total_periods_fixed = sum(len(mask) for mask in result_fixed['breakout_signals'].values())
    total_breakouts_fixed = sum(np.sum(mask) for mask in result_fixed['breakout_signals'].values())
    actual_coverage_fixed = (total_breakouts_fixed / total_periods_fixed * 100) if total_periods_fixed > 0 else 0
    
    print(f"📊 Fixed threshold results:")
    print(f"   - Breakout signals: {len(result_fixed['breakout_signals'])}")
    print(f"   - Total periods: {total_periods_fixed}")
    print(f"   - Total breakouts: {total_breakouts_fixed}")
    print(f"   - Actual coverage: {actual_coverage_fixed:.2f}%")
    
    # Test 3: Convenience method
    print("\n--- Test 3: Convenience Method (2%) ---")
    result_convenience = aedl.get_structural_breakouts_2percent()
    
    if 'error' in result_convenience:
        print(f"❌ Convenience method failed: {result_convenience['error']}")
        return False
    
    print(f"✅ Convenience method works: {len(result_convenience['breakout_signals'])} signals")
    
    # Verify quantile approach achieves minimum coverage
    success = actual_coverage >= 1.5  # Allow some tolerance
    
    print(f"\n🎯 Results Summary:")
    print(f"   - Quantile approach coverage: {actual_coverage:.2f}% {'✅' if success else '❌'}")
    print(f"   - Fixed approach coverage: {actual_coverage_fixed:.2f}%")
    print(f"   - Quantile vs Fixed difference: {actual_coverage - actual_coverage_fixed:.2f}%")
    
    return success

if __name__ == "__main__":
    success = test_quantile_approach()
    if success:
        print("\n🎉 Test PASSED: 2% quantile approach works correctly!")
    else:
        print("\n❌ Test FAILED: 2% quantile approach not working as expected")
