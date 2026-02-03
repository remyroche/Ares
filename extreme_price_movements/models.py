import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, LogisticRegression

def make_elasticnet_reg(alpha=1e-3, l1_ratio=0.2):
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("reg", ElasticNet(
            alpha=float(alpha),
            l1_ratio=float(l1_ratio),
            fit_intercept=True,
            max_iter=5000,
            random_state=42
        ))
    ])

def make_exhaustion_model(C=1.0, l1_ratio=0.3):
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=float(l1_ratio),
            C=float(C),
            max_iter=2000,
            random_state=42
        ))
    ])

def map_pred_to_score(pred_ret, mode="tanh", scale=10.0):
    x = float(pred_ret) * float(scale)
    if mode == "tanh":
        return float(np.tanh(max(0.0, x)))
    if mode == "relu":
        return float(max(0.0, x))
    raise ValueError("mode must be 'tanh' or 'relu'")
