
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

params_cci = {
    'objective': cci_objective,
    'metric': 'auc',
    'n_estimators': 10,
    'verbosity': -1
}

model_cci = lgb.LGBMClassifier(**params_cci)

print("Training with scalar gradient return...")
try:
    model_cci.fit(X, y, eval_set=[(X, y)])
    print("Training success!")
except Exception as e:
    print(f"Training failed: {e}")
    import traceback
    traceback.print_exc()

# Sparse data test
X_sparse = X.iloc[:5]
y_sparse = y.iloc[:5] # Small batch
print("\nTraining with small batch...")
try:
    model_cci.fit(X_sparse, y_sparse, eval_set=[(X_sparse, y_sparse)])
    print("Training success!")
except Exception as e:
    print(f"Training failed: {e}")
    import traceback
    traceback.print_exc()
