
import time
import numpy as np
import pandas as pd
import extreme_price_movements.fast_funcs as ff
from extreme_price_movements.feature_transforms import CausalFeatureTransformer

def benchmark():
    # Setup Data
    # 3 years hourly data ~ 26k rows. 300 symbols.
    N = 26000
    M = 100 # Reduced from 300 to be quicker for benchmark
    print(f"Generating synthetic data ({N}x{M})...")
    data = np.random.randn(N, M).astype(np.float32)
    # Add some outliers
    data[::1000] *= 10.0
    # Make positive for log transform
    data = np.abs(data) + 1.0
    
    df = pd.DataFrame(data)
    
    # 1. Baseline
    transformer = CausalFeatureTransformer(winsor_qt=0.02, roll_window=720) # 30 days
    
    print("\nRunning Baseline (Quantile-based)...")
    start = time.time()
    res_base = transformer.transform(df, "test")
    dt_base = time.time() - start
    print(f"Baseline Time: {dt_base:.4f}s")
    
    # 2. Optimized (Parametric Mean/Std)
    print("\nRunning Optimized (Mean/Std-based)...")
    start = time.time()
    
    # Implementation of Optimized Logic inline for testing
    mat = np.ascontiguousarray(df.to_numpy(dtype=np.float32, copy=False))
    mat = np.arcsinh(mat)
    
    window = 720
    # Rolling Mean/Std
    # We can use ff._numba_rolling_mean/std_nan_safe if we wrap them
    # But for speed let's use a specialized kernel or just reuse existing 1D applied to frame?
    # ff._numba_rolling_zscore_parallel actually computes mean/std internally!
    # But it returns Z directly.
    # We need Mean/Std separate.
    
    # Let's use ff.apply_to_frame logic but parallelized?
    # For benchmark we can just use the existing fast_funcs wrappers which iterate cols (slow-ish but O(N)).
    
    # Actually, let's write what the optimized generic implementation would look like:
    # We need rolling mean/std efficiently.
    # ff.numba_zscore computes them. 
    
    # Approximation:
    # 1. Log
    # 2. Z-Score (using robust window)
    # 3. Clip Z-Score to +/- K
    # 4. Return Clipped Z-Score?
    # Wait, the original pipeline is: Log -> Winsor -> Z-Score.
    # Result is a Z-score of the Winsorized data.
    # If we Clip based on the Unwinsorized Z-score, then the result IS the Clipped Z-Score (roughly).
    # Z_robust = (X - Mean) / Std.  Clip(Z_robust, -K, K).
    # Baseline: X_wins = Clip(X, Q_lo, Q_hi). Z_final = (X_wins - Mean_wins) / Std_wins.
    
    # These are slightly different.
    # But if X is mostly normal, Z_robust ~ Z_final.
    # Let's see if (Log -> RollingZ -> Clip) is a good enough proxy.
    
    # Efficient Implementation:
    # mat = np.arcsinh(mat)
    # z_scores = ff._numba_rolling_zscore_parallel(mat, window)
    # z_clipped = np.clip(z_scores, -2.5, 2.5) # 2.5 sigma ~ 1% tails
    
    z_scores = ff._numba_rolling_zscore_parallel(mat, window)
    res_opt_mat = np.clip(z_scores, -2.5, 2.5)
    res_opt = pd.DataFrame(res_opt_mat, index=df.index, columns=df.columns)
    
    dt_opt = time.time() - start
    print(f"Optimized Time: {dt_opt:.4f}s")
    print(f"Speedup: {dt_base / dt_opt:.2f}x")
    
    # Compare correctness (Correlation)
    # Just take first column
    c1 = res_base.iloc[:, 0]
    c2 = res_opt.iloc[:, 0]
    corr = c1.corr(c2)
    print(f"\nCorrelation between Baseline and Optimized: {corr:.4f}")
    
if __name__ == "__main__":
    benchmark()
