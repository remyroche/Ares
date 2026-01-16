
import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.getcwd())

try:
    from src.training.steps.labeling.resonance_detector import ResonanceDetector
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def reproduce():
    print("--- Reproducing Issue ---")

    # 1. Setup Dummy Data
    # Create spectral components mimicking d1, d2, d3, d4
    t = np.linspace(0, 10, 100)
    # Create correlated signals for d1 and d3 (should trigger resonance)
    d1 = np.sin(t)
    d3 = np.sin(t) # Perfect correlation

    spectral_components = {
        'spec1_d1': d1,
        'spec1_d3': d3,
        'spec1_d2': np.random.randn(100),
        'spec1_d4': np.random.randn(100)
    }

    detector = ResonanceDetector(verbose=False)

    # 2. Run compute_all_resonances
    # This should find 'spec1_d1_d3_resonance'
    resonance_analysis = detector.compute_all_resonances(spectral_components)

    print(f"\nResonance Analysis Keys: {list(resonance_analysis.keys())}")

    if not resonance_analysis:
        print("Error: No resonance analysis computed.")
        return

    first_key = list(resonance_analysis.keys())[0]
    first_val = resonance_analysis[first_key]

    print(f"Type of value for {first_key}: {type(first_val)}")
    if isinstance(first_val, dict):
        print(f"Keys in value: {list(first_val.keys())}")

    # 3. Simulate AdaptiveEventDrivenLabeling logic
    print("\n--- Simulating AdaptiveEventDrivenLabeling logic ---")

    high_resonance_periods = {}
    resonance_threshold = 0.7

    for resonance_name, resonance_scores in resonance_analysis.items():
        # This is the line from get_harmonic_entries
        is_ndarray = isinstance(resonance_scores, np.ndarray)
        print(f"Checking {resonance_name}: isinstance(scores, np.ndarray) = {is_ndarray}")

        if is_ndarray:
            high_resonance_mask = resonance_scores > resonance_threshold
            high_resonance_periods[resonance_name] = high_resonance_mask

    print(f"\nHigh Resonance Periods Count: {len(high_resonance_periods)}")

    if len(high_resonance_periods) == 0:
        print("BUG CONFIRMED: high_resonance_periods is empty because of type mismatch.")
    else:
        print("Bug not reproduced (logic worked?).")

if __name__ == "__main__":
    reproduce()
