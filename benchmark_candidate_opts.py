
import time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import binary_dilation
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler

# --- Config ---
T = 50000
N = 200
OOF_TRAIN_SIZE = 200000  # Conservative estimate for "250k" samples
N_FEATURES = 100
EXPANSION_OFFSETS = [-4, -3, -2, -1, 1, 2, 3, 4] # +/- 4
ET_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "min_samples_leaf": 30,
    "max_features": "sqrt",
    "n_jobs": 3,
    "random_state": 42
}

def benchmark_ranking():
    print("\n--- Benchmark 1: Ranking vs Thresholding ---")
    data = np.random.randn(T, N).astype(np.float32)
    df = pd.DataFrame(data)

    # Baseline: Rank every time
    start = time.time()
    for _ in range(5):
        ranks = df.rank(axis=1, pct=True)
        mask = ranks > 0.95
    end = time.time()
    baseline_time = (end - start) / 5
    print(f"Baseline (Rank every time): {baseline_time:.4f}s per call")

    # Optimized: Pre-compute rank, then threshold
    start = time.time()
    ranks_pre = df.rank(axis=1, pct=True)
    prep_time = time.time() - start

    start = time.time()
    for _ in range(5):
        mask = ranks_pre > 0.95
    end = time.time()
    optimized_time = (end - start) / 5
    print(f"Optimized (Threshold pre-ranked): {optimized_time:.4f}s per call")
    print(f"Speedup: {baseline_time / optimized_time:.1f}x")

def benchmark_expansion():
    print("\n--- Benchmark 2: Expansion (Shift Loop vs Binary Dilation) ---")
    mask_data = (np.random.randn(T, N) > 1.6).astype(bool)
    df_mask = pd.DataFrame(mask_data)

    # Baseline: Pandas Shift Loop
    start = time.time()
    for _ in range(5):
        expanded = df_mask.copy()
        for i in EXPANSION_OFFSETS:
            expanded |= df_mask.shift(i).fillna(False)
    end = time.time()
    baseline_time = (end - start) / 5
    print(f"Baseline (Pandas shift loop): {baseline_time:.4f}s per call")

    # Optimized: Binary Dilation
    # Structure: [1] * 9 along axis 0
    structure = np.ones((9, 1), dtype=bool)

    start = time.time()
    for _ in range(5):
        # binary_dilation expects numpy array
        # Scipy binary_dilation is generic, let's see if it's faster
        # Note: binary_dilation with structure covers +/- 4 automatically if structure is length 9 centered.
        expanded_arr = binary_dilation(mask_data, structure=structure)
    end = time.time()
    optimized_time = (end - start) / 5
    print(f"Optimized (Binary Dilation): {optimized_time:.4f}s per call")
    print(f"Speedup: {baseline_time / optimized_time:.1f}x")

def benchmark_models():
    print("\n--- Benchmark 3 & 4: Feature Selection & Model Training ---")
    # Generate X, y
    X = np.random.randn(OOF_TRAIN_SIZE, N_FEATURES).astype(np.float32)
    y = np.random.randn(OOF_TRAIN_SIZE).astype(np.float32)
    # Add some signal
    y += 0.1 * X[:, 0] - 0.1 * X[:, 1]

    # Pre-scaling (usually part of pipeline, but let's assume pre-scaled for model bench)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Benchmark Feature Selection (Ridge)
    start = time.time()
    sel_model = Ridge(alpha=1.0)
    sel_model.fit(X_scaled, y)
    coefs = np.abs(sel_model.coef_)
    top_k = np.argpartition(coefs, -20)[-20:]
    fs_time = time.time() - start
    print(f"Ridge Feature Selection time: {fs_time:.4f}s")

    # Benchmark Ridge Training (for Two-Stage)
    start = time.time()
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled[:, top_k], y)
    ridge.predict(X_scaled[:, top_k]) # OOF predict
    ridge_time = time.time() - start
    print(f"Ridge Train+Predict time (on top 20 feats): {ridge_time:.4f}s")

    # Benchmark ExtraTrees Training
    start = time.time()
    et = ExtraTreesRegressor(**ET_PARAMS)
    et.fit(X_scaled[:, top_k], y)
    et.predict(X_scaled[:, top_k])
    et_time = time.time() - start
    print(f"ExtraTrees Train+Predict time (on top 20 feats): {et_time:.4f}s")

    print(f"ET / Ridge Ratio: {et_time / ridge_time:.1f}x")
    print(f"Feature Selection Overhead vs ET: {fs_time / et_time:.1%}")

if __name__ == "__main__":
    benchmark_ranking()
    benchmark_expansion()
    benchmark_models()
