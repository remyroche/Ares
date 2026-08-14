
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

# --- Config ---
N_SAMPLES = 250000
N_FEATURES = 50
ET_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "min_samples_leaf": 30,
    "max_features": "sqrt",
    "n_jobs": 4,
    "random_state": 42
}
LGBM_PARAMS = {
    "n_estimators": 200,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_samples": 30,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "n_jobs": 4,
    "random_state": 42,
    "verbosity": -1
}

def benchmark():
    if LGBMRegressor is None:
        print("LightGBM not installed.")
        return

    print(f"Generating data: {N_SAMPLES} samples, {N_FEATURES} features...")
    X = np.random.randn(N_SAMPLES, N_FEATURES).astype(np.float32)
    # Add some noise and signal
    y = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.1 * X[:, 2] * X[:, 3] + np.random.randn(N_SAMPLES)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n--- ExtraTrees ---")
    start = time.time()
    et = ExtraTreesRegressor(**ET_PARAMS)
    et.fit(X_scaled, y)
    et_pred = et.predict(X_scaled)
    et_time = time.time() - start
    print(f"Time: {et_time:.4f}s")
    print(f"R2: {r2_score(y, et_pred):.4f}")

    print("\n--- LightGBM (Regularized) ---")
    start = time.time()
    lgbm = LGBMRegressor(**LGBM_PARAMS)
    lgbm.fit(X_scaled, y)
    lgbm_pred = lgbm.predict(X_scaled)
    lgbm_time = time.time() - start
    print(f"Time: {lgbm_time:.4f}s")
    print(f"R2: {r2_score(y, lgbm_pred):.4f}")

    print(f"\nSpeedup: {et_time / lgbm_time:.1f}x")

if __name__ == "__main__":
    benchmark()
