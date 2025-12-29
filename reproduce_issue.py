
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

def cci_objective(y_true, y_pred):
    """Custom CCI objective for LightGBM"""
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))  # Sigmoid
    # Calculate concordance correlation
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)

    cov_yy = np.cov(y_true, y_pred)[0, 1]
    var_y_true = np.var(y_true)
    var_y_pred = np.var(y_pred)

    cci = (2 * cov_yy) / (var_y_true + var_y_pred + (y_true_mean - y_pred_mean)**2 + 1e-9)

    # Convert to gradient/hessian format for LightGBM
    grad = -cci  # Negative for maximization
    hess = np.ones_like(grad)
    return grad, hess

# Create synthetic data
X = pd.DataFrame(np.random.rand(100, 5), columns=[f'feat_{i}' for i in range(5)])
y = pd.Series(np.random.randint(0, 2, 100))

# Sparse case: 1 sample
X_sparse = X.iloc[:1]
y_sparse = y.iloc[:1]

print(f"Testing with sparse data (N={len(X_sparse)})...")
try:
    # This should fail if y_true has length 1 for np.cov
    grad, hess = cci_objective(y_sparse.values, np.array([0.5]))
    print(f"Success! grad={grad}, hess={hess}")
except Exception as e:
    print(f"Caught expected error: {e}")
    import traceback
    traceback.print_exc()

# Another sparse case: 2 samples (constant y)
X_sparse2 = X.iloc[:2]
y_sparse2 = pd.Series([0, 0]) # Constant

print(f"\nTesting with constant y (N={len(X_sparse2)})...")
try:
    grad, hess = cci_objective(y_sparse2.values, np.array([0.5, 0.6]))
    print(f"Success! grad={grad}, hess={hess}")
except Exception as e:
    print(f"Caught error: {e}")
    import traceback
    traceback.print_exc()
