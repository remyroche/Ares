
import numpy as np
import time
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

try:
    from src.utils.entropy_optimized import lempel_ziv_complexity_numba
except ImportError as e:
    print(f"Error importing: {e}")
    sys.exit(1)

def run_repro():
    N = 1000
    print(f"Testing with N={N}...")

    # Case 1: All zeros (Low Complexity)
    data1 = np.zeros(N)

    # Case 2: Random (High Complexity)
    np.random.seed(42)
    data2 = np.random.randn(N)

    # Run
    print("Running on Data 1 (All Zeros)...")
    res1 = lempel_ziv_complexity_numba(data1, normalize=False)

    print("Running on Data 2 (Random)...")
    res2 = lempel_ziv_complexity_numba(data2, normalize=False)

    # Compare last value
    c1 = res1[-1]
    c2 = res2[-1]

    print(f"Complexity 1 (Zeros): {c1}")
    print(f"Complexity 2 (Random): {c2}")

    if c1 < c2:
        print("✅ SUCCESS: Random data has higher complexity than Zeros.")
    else:
        print("❌ FAIL: Random data complexity not higher?")

    if np.array_equal(res1, res2):
         print("❌ FAIL: Arrays are identical.")
    else:
         print("✅ SUCCESS: Arrays differ.")

    # Performance
    print(f"\nPerformance check with N=10000...")
    data_perf = np.random.randn(10000)
    start = time.time()
    _ = lempel_ziv_complexity_numba(data_perf, normalize=False)
    dur = time.time() - start
    print(f"Time: {dur:.4f}s")

if __name__ == "__main__":
    run_repro()
