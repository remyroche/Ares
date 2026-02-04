import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, LogisticRegression
from .utils import tprint

def make_elasticnet_reg(alpha=1e-3, l1_ratio=0.2):
    tprint(f"Entering function: make_elasticnet_reg in models.py")
    tprint(f"make_elasticnet_reg params: alpha={alpha}, l1_ratio={l1_ratio}")
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
    tprint(f"Entering function: make_exhaustion_model in models.py")
    tprint(f"make_exhaustion_model params: C={C}, l1_ratio={l1_ratio}")
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
    tprint(f"Entering function: map_pred_to_score in models.py")
    tprint(f"map_pred_to_score input: pred_ret={pred_ret}, mode={mode}, scale={scale}")
    x = float(pred_ret) * float(scale)
    if mode == "tanh":
        res = float(np.tanh(max(0.0, x)))
        tprint(f"map_pred_to_score result (tanh): {res}")
        return res
    if mode == "relu":
        res = float(max(0.0, x))
        tprint(f"map_pred_to_score result (relu): {res}")
        return res
    raise ValueError("mode must be 'tanh' or 'relu'")
