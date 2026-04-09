from .utils import tprint
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, LogisticRegression

def make_elasticnet_reg(alpha=1e-3, l1_ratio=0.2):
    tprint(f"Entering function: make_elasticnet_reg in {__name__}")
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



