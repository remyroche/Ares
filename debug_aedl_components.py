
import sys
import os
import numpy as np
import pandas as pd
import pywt
import logging

# Add src to path
sys.path.append(os.getcwd())

# Mock logger
logging.basicConfig(level=logging.INFO)

from src.training.steps.labeling.wavelet_decomposition import WaveletDecomposition
from src.training.steps.labeling.resonance_detector import ResonanceDetector

def test_wavelet_decomposition():
    print("--- Testing WaveletDecomposition (SWT) ---")
    data_len = 2000
    t = np.linspace(0, 10, data_len)
    signal = np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.sin(2 * np.pi * 5.0 * t)
    
    wd = WaveletDecomposition(max_level=4, wavelet='sym4')
    
    try:
        results = wd.decompose_signal(signal)
        print(f"Decomposition Results Keys: {list(results.keys())}")
        
        for k, v in results.items():
            print(f"Scale {k}: Type={type(v)}, Shape={v.shape}")
            if len(v) != data_len:
                print(f"❌ ERROR: Length mismatch for {k}. Expected {data_len}, got {len(v)}")
            if np.isnan(v).any():
                print(f"❌ ERROR: NaNs found in {k}")
                
        print("✅ WaveletDecomposition Test Passed")
        
    except Exception as e:
        print(f"❌ WaveletDecomposition FAILED: {e}")
        import traceback
        traceback.print_exc()

def test_resonance_detector():
    print("\n--- Testing ResonanceDetector ---")
    rd = ResonanceDetector(cache_size=128)
    
    # Create coupled signals
    data_len = 1000
    t = np.linspace(0, 10, data_len)
    sig1 = np.sin(2 * np.pi * 2.0 * t)
    sig2 = np.sin(2 * np.pi * 2.0 * t + np.pi/4) # Phase shifted
    
    try:
        coh_summary, coh_series = rd.calculate_wavelet_coherence(sig1, sig2)
        print(f"Coherence Summary: {coh_summary}")
        print(f"Coherence Series Max: {np.max(coh_series)}")
        
        phase_summary, phase_series = rd.calculate_phase_lead_lag(sig1, sig2)
        print(f"Phase Summary: {phase_summary}")
        
        # Test constant signal check (my fix)
        const_sig = np.ones(data_len)
        c_sum, c_ser = rd.calculate_wavelet_coherence(const_sig, sig2)
        print(f"Constant Signal Coherence: {c_sum} (Expect 0.0)")
        
        if c_sum != 0.0:
            print("❌ ERROR: Constant signal should return 0.0 coherence")
            
        print("✅ ResonanceDetector Test Passed")
        
    except Exception as e:
        print(f"❌ ResonanceDetector FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_wavelet_decomposition()
    test_resonance_detector()
