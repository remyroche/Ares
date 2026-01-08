import pandas as pd
import numpy as np
import sys
import os

# Mocking tprint for the script
def tprint_info(msg): print(f"[INFO] {msg}")
def tprint_success(msg): print(f"[SUCCESS] {msg}")
def tprint_warning(msg): print(f"[WARNING] {msg}")
def tprint_error(msg): print(f"[ERROR] {msg}")

from src.training.steps.labeling.orthogonal_label_generation import _wavelet_denoise, apply_layer2_price_processing
from src.training.steps.labeling.label_based_layer_2 import LabelBasedLayer2

def test_wavelet_level():
    tprint_info("Testing Wavelet Level Update...")
    signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    
    # Check default level in _wavelet_denoise
    # Since we changed the default to 2, calling without level should use 2
    import inspect
    sig = inspect.signature(_wavelet_denoise)
    default_level = sig.parameters['level'].default
    tprint_info(f"Default level in _wavelet_denoise: {default_level}")
    assert default_level == 2, f"Expected 2, got {default_level}"
    
    # Check default level in apply_layer2_price_processing
    sig2 = inspect.signature(apply_layer2_price_processing)
    default_level2 = sig2.parameters['wavelet_level'].default
    tprint_info(f"Default level in apply_layer2_price_processing: {default_level2}")
    assert default_level2 == 2, f"Expected 2, got {default_level2}"
    tprint_success("Wavelet level test passed!")

def test_memory_management():
    tprint_info("Testing Memory & Cache Management...")
    # Initialize LabelBasedLayer2 with mock config
    class MockConfig:
        def __init__(self, size): self._size = size
        def get(self, k, default): return self._size if k == "global_cache_size" else default
    
    lb2 = LabelBasedLayer2(verbose=True)
    lb2.config = MockConfig(2)
    
    # Fill caches
    lb2._global_feature_cache = {'a': 1, 'b': 2, 'c': 3}
    lb2._all_candidate_assessments = [1] * 6000
    
    tprint_info(f"Initial cache size: {len(lb2._global_feature_cache)}")
    tprint_info(f"Initial assessment size: {len(lb2._all_candidate_assessments)}")
    
    lb2._cleanup_memory()
    
    tprint_info(f"Final cache size: {len(lb2._global_feature_cache)}")
    tprint_info(f"Final assessment size: {len(lb2._all_candidate_assessments)}")
    
    assert len(lb2._global_feature_cache) == 2, f"Expected 2, got {len(lb2._global_feature_cache)}"
    assert len(lb2._all_candidate_assessments) == 0, f"Expected 0, got {len(lb2._all_candidate_assessments)}"
    tprint_success("Memory management test passed!")

if __name__ == "__main__":
    try:
        test_wavelet_level()
        test_memory_management()
        tprint_success("\n🚀 All verifications passed!")
    except Exception as e:
        tprint_error(f"Verification failed: {e}")
        sys.exit(1)
