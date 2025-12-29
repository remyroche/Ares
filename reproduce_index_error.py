
import numpy as np
import pandas as pd
import lightgbm as lgb

def cci_objective(y_true, y_pred):
    """Custom CCI objective for LightGBM"""
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))  # Sigmoid
    # Calculate concordance correlation
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)

    # This is the line suspected:
    cov_yy = np.cov(y_true, y_pred)[0, 1]

    var_y_true = np.var(y_true)
    var_y_pred = np.var(y_pred)

    cci = (2 * cov_yy) / (var_y_true + var_y_pred + (y_true_mean - y_pred_mean)**2 + 1e-9)

    # Convert to gradient/hessian format for LightGBM
    grad = -cci  # Negative for maximization
    hess = np.ones_like(grad)
    return grad, hess

# Case where np.cov returns a 0-d array?
# If inputs are empty?
try:
    print("Testing empty input...")
    cci_objective(np.array([]), np.array([]))
except Exception as e:
    print(f"Empty input error: {e}")

# Case where inputs are scalar (0-d array)?
try:
    print("Testing scalar input...")
    # np.cov treats 0-d array as 1 sample? No, it expects at least 1-d usually.
    # But if y_true is passed as scalar?
    cci_objective(np.array(1.0), np.array(0.5))
except Exception as e:
    print(f"Scalar input error: {e}")
