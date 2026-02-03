import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, HuberRegressor
from sklearn.preprocessing import StandardScaler
from extreme_price_movements.utils import tprint

class RuleCleaner:
    def __init__(self, corr_thr=0.8):
        self.corr_thr = float(corr_thr)
        self.keep_cols_ = None

    def fit(self, X_df: pd.DataFrame, coef_by_col: dict):
        cols = list(X_df.columns)
        if len(cols) <= 1:
            self.keep_cols_ = cols
            return self

        corr = X_df.corr().abs()
        # Ensure we have a writable array
        corr_vals = corr.to_numpy(copy=True)
        np.fill_diagonal(corr_vals, 0.0)
        corr = pd.DataFrame(corr_vals, index=corr.index, columns=corr.columns)

        strength = pd.Series({c: abs(float(coef_by_col.get(c, 0.0))) for c in cols})
        ordered = strength.sort_values(ascending=False).index.tolist()

        keep = []
        dropped = set()
        for c in ordered:
            if c in dropped:
                continue
            keep.append(c)
            high = corr.index[corr[c] > self.corr_thr].tolist()
            for h in high:
                dropped.add(h)

        self.keep_cols_ = keep
        return self

    def transform(self, X_df: pd.DataFrame) -> pd.DataFrame:
        if self.keep_cols_ is None:
            return X_df
        cols = [c for c in self.keep_cols_ if c in X_df.columns]
        return X_df[cols].copy()

def compute_mr_weights(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """
    Computes sample weights for MR model favoring exhaustion and snapback.
    w_mr = 1 + a * p_exh_lag1 + b * clip(|ret1h_z|,0,zmax) + c * clip(vol_z24,0,vmax)
    Optionally downweight strong trend.
    """
    # defaults
    a = 2.0  # weight for exhaustion
    b = 0.5  # weight for return deviation
    c = 0.5  # weight for vol
    zmax = 4.0
    vmax = 4.0

    p_exh = df.get("p_exh_lag1", 0.0)
    ret_z = df.get("a_ret1h_z", 0.0).abs().clip(upper=zmax)
    vol_z = df.get("a_volz", 0.0).clip(upper=vmax)
    g_trend = df.get("G_TREND", 0.0)

    w = 1.0 + (a * p_exh) + (b * ret_z) + (c * vol_z)

    # downweight strong trend
    w *= (1.0 - 0.5 * g_trend)

    # Clip and normalize
    w = w.clip(0.25, 4.0)
    w = w * (len(w) / w.sum())
    return w.to_numpy(dtype=np.float32)

class MRModel:
    def __init__(self, lasso_alpha=0.001, huber_epsilon=1.35, rule_clean_corr=0.8):
        self.lasso_alpha = lasso_alpha
        self.huber_epsilon = huber_epsilon
        self.rule_clean_corr = rule_clean_corr
        self.model = None
        self.cleaner = None
        self.selected_features = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray = None):
        # 1. Lasso Selection
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

        lasso = Lasso(alpha=self.lasso_alpha, random_state=42)
        lasso.fit(X_scaled, y) # Lasso doesn't support sample_weight usually? It does in sklearn > 0.23

        # Get coefs
        coefs = dict(zip(X.columns, lasso.coef_))
        selected = [c for c, v in coefs.items() if abs(v) > 1e-5]

        if not selected:
            tprint("MR Model: Lasso dropped all features. Using top 5 by correlation.")
            corrs = X.corrwith(pd.Series(y, index=X.index)).abs().sort_values(ascending=False)
            selected = corrs.head(5).index.tolist()
            coefs = {c: 1.0 for c in selected}

        # 2. RuleCleaner
        self.cleaner = RuleCleaner(corr_thr=self.rule_clean_corr)
        self.cleaner.fit(X[selected], coefs)
        final_cols = self.cleaner.keep_cols_
        self.selected_features = final_cols

        # 3. Huber Regressor
        # Scale again? Pipeline handles it.
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", HuberRegressor(epsilon=self.huber_epsilon, max_iter=200))
        ])

        self.model.fit(X[final_cols], y, reg__sample_weight=sample_weight)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.selected_features is None:
            raise ValueError("MR Model not fitted")

        X_sel = X[self.selected_features]
        # Missing cols handling?
        # Assuming X has them.
        return self.model.predict(X_sel)
