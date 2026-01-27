
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

def verify_lightgbm():
    print("\n--- Verifying LightGBM (sklearn API) ---")
    X = np.random.rand(100, 5)
    offset = np.full(100, 10.0) # Large offset to be obvious
    y_true = np.random.rand(100) + offset # y around 10.5

    model = LGBMRegressor(n_estimators=10, random_state=42, verbose=-1)
    model.fit(X, y_true, init_score=offset)

    # Predict without passing init_score (sklearn API default)
    preds = model.predict(X)

    mean_pred = np.mean(preds)
    print(f"Mean Prediction (no init_score passed): {mean_pred:.4f}")
    print(f"Target Mean (approx 10.5): {np.mean(y_true):.4f}")
    print(f"Residual Mean (approx 0.5): {np.mean(y_true - offset):.4f}")

    if abs(mean_pred - 0.5) < 1.0:
        print(">> LGBMRegressor.predict(X) returns RESIDUAL (Sum of Trees).")
        print(">> To get full prediction, you MUST add offset manually.")
    else:
        print(">> LGBMRegressor.predict(X) returns FULL value (includes some offset?).")

def verify_xgboost():
    print("\n--- Verifying XGBoost (sklearn API) ---")
    X = np.random.rand(100, 5)
    offset = np.full(100, 10.0)
    y_true = np.random.rand(100) + offset

    model = XGBRegressor(n_estimators=10, random_state=42)
    model.fit(X, y_true, base_margin=offset)

    # Predict with base_margin
    preds_with = model.predict(X, base_margin=offset)

    # Predict without base_margin
    preds_without = model.predict(X) # Defaults to base_score=0.5 usually

    print(f"Mean Prediction (with base_margin): {np.mean(preds_with):.4f}")
    print(f"Mean Prediction (without base_margin): {np.mean(preds_without):.4f}")

    if abs(np.mean(preds_with) - np.mean(y_true)) < 0.1:
        print(">> XGBRegressor.predict(X, base_margin=...) returns FULL value.")
    else:
        print(">> XGBRegressor.predict(X, base_margin=...) returns something else.")

    if abs(np.mean(preds_without) - 0.5) < 1.0:
        print(">> XGBRegressor.predict(X) (no margin) returns RESIDUAL (+ default base_score).")
    else:
        print(">> XGBRegressor.predict(X) (no margin) returns FULL value (baked in?).")

def verify_catboost():
    print("\n--- Verifying CatBoost (sklearn API) ---")
    X = np.random.rand(100, 5)
    offset = np.full(100, 10.0)
    y_true = np.random.rand(100) + offset

    model = CatBoostRegressor(n_estimators=10, random_state=42, verbose=0, allow_writing_files=False)
    model.fit(X, y_true, baseline=offset)

    # Predict without baseline
    preds_without = model.predict(X)

    # Predict with baseline? CatBoost sklearn predict doesn't accept 'baseline' argument directly usually
    # unless using Pool. Let's check if we can pass it.
    try:
        from catboost import Pool
        pool = Pool(X, baseline=offset)
        preds_with = model.predict(pool)
        print(f"Mean Prediction (with baseline in Pool): {np.mean(preds_with):.4f}")
    except Exception as e:
        print(f"Could not test Pool: {e}")
        preds_with = np.zeros_like(preds_without)

    print(f"Mean Prediction (no baseline passed): {np.mean(preds_without):.4f}")

    if abs(np.mean(preds_with) - np.mean(y_true)) < 0.1:
        print(">> CatBoost predict(Pool(baseline=...)) returns FULL value.")

    if abs(np.mean(preds_without) - 0.5) < 1.0:
        print(">> CatBoost predict(X) returns RESIDUAL.")
    else:
        print(">> CatBoost predict(X) returns FULL value.")

if __name__ == "__main__":
    verify_lightgbm()
    verify_xgboost()
    verify_catboost()
