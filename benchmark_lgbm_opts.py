
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor

# --- Config ---
N_SAMPLES = 250000
N_FEATURES = 50
BASE_PARAMS = {
    "n_estimators": 200,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "max_depth": 6,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "colsample_bytree": 0.8,
    "n_jobs": 4,
    "random_state": 42,
    "verbosity": -1
}

def benchmark():
    print(f"Generating data: {N_SAMPLES} samples, {N_FEATURES} features...")
    X = np.random.randn(N_SAMPLES, N_FEATURES).astype(np.float32)
    # Add some noise and signal
    y = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.1 * X[:, 2] * X[:, 3] + np.random.randn(N_SAMPLES)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Baseline (GBDT + 255 bins)
    print("\n--- Baseline (GBDT, max_bin=255) ---")
    start = time.time()
    lgbm = LGBMRegressor(**BASE_PARAMS, boosting_type='gbdt', max_bin=255)
    lgbm.fit(X_scaled, y)
    pred = lgbm.predict(X_scaled)
    base_time = time.time() - start
    print(f"Time: {base_time:.4f}s")
    print(f"R2: {r2_score(y, pred):.4f}")

    # 2. GOSS
    print("\n--- GOSS (max_bin=255) ---")
    start = time.time()
    # GOSS requires subsample=1.0 (implied, can't subsample rows randomly)
    lgbm_goss = LGBMRegressor(**BASE_PARAMS, boosting_type='goss', max_bin=255, subsample=1.0)
    lgbm_goss.fit(X_scaled, y)
    pred_goss = lgbm_goss.predict(X_scaled)
    goss_time = time.time() - start
    print(f"Time: {goss_time:.4f}s")
    print(f"R2: {r2_score(y, pred_goss):.4f}")
    print(f"Speedup vs Base: {base_time / goss_time:.2f}x")

    # 3. Reduced Binning (max_bin=63)
    print("\n--- Reduced Bins (GBDT, max_bin=63) ---")
    start = time.time()
    lgbm_bins = LGBMRegressor(**BASE_PARAMS, boosting_type='gbdt', max_bin=63)
    lgbm_bins.fit(X_scaled, y)
    pred_bins = lgbm_bins.predict(X_scaled)
    bins_time = time.time() - start
    print(f"Time: {bins_time:.4f}s")
    print(f"R2: {r2_score(y, pred_bins):.4f}")
    print(f"Speedup vs Base: {base_time / bins_time:.2f}x")

    # 4. GOSS + Reduced Bins
    print("\n--- GOSS + Reduced Bins (max_bin=63) ---")
    start = time.time()
    lgbm_combo = LGBMRegressor(**BASE_PARAMS, boosting_type='goss', max_bin=63, subsample=1.0)
    lgbm_combo.fit(X_scaled, y)
    pred_combo = lgbm_combo.predict(X_scaled)
    combo_time = time.time() - start
    print(f"Time: {combo_time:.4f}s")
    print(f"R2: {r2_score(y, pred_combo):.4f}")
    print(f"Speedup vs Base: {base_time / combo_time:.2f}x")

if __name__ == "__main__":
    benchmark()
