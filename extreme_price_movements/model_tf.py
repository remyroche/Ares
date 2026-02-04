import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from extreme_price_movements.utils import tprint
from extreme_price_movements.feature_selection_extreme_events import mdi_feature_selection_leakage_safe

def compute_tf_weights(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """
    Computes sample weights for TF model favoring continuation.
    w_tf = 1 + a*(1 - p_exh_lag1) + b*clip(|trend_pct|) + c*G_TREND
    Downweight high vol.
    """
    tprint(f"Entering function: compute_tf_weights in model_tf.py")
    a = 2.0
    b = 1.0
    c = 1.0
    cap = 0.05

    p_exh = df.get("p_exh_lag1", 0.0)
    trend = df.get("a_trend", 0.0).abs().clip(upper=cap) * 100 # scale up? trend is usually small pct
    g_trend = df.get("G_TREND", 0.0)
    g_vol = df.get("G_VOL", 0.0)

    w = 1.0 + (a * (1.0 - p_exh)) + (b * trend) + (c * g_trend)
    w *= (1.0 - 0.3 * g_vol)

    w = w.clip(0.25, 4.0)
    w = w * (len(w) / w.sum())
    return w.to_numpy(dtype=np.float32)

class TFModel:
    def __init__(self, lasso_alpha=0.001):
        tprint(f"Entering function: __init__ in model_tf.py")
        self.lasso_alpha = lasso_alpha
        self.models = []
        self.selected_features = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray = None):
        # 1. Feature Selection (MDI)
        tprint(f"Entering function: fit in model_tf.py")

        n_samples = len(X)
        n_select = min(60, max(1, n_samples // 100))
        tprint(f"TFModel: Running MDI feature selection. Target features={n_select}")

        base_selector = ExtraTreesRegressor(
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=50,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )

        sel_res = mdi_feature_selection_leakage_safe(
            X=X,
            y=y,
            base_model=base_selector,
            sample_weight=sample_weight,
            top_n_precluster=n_select,
            keep_top_per_cluster=1,
            use_quantile_transform_for_corr=True
        )

        self.selected_features = sel_res.selected_features
        tprint(f"TFModel: Selected {len(self.selected_features)} features.")
        X_sel = X[self.selected_features]

        # 2. Train Ensemble
        # 15 models: varying alpha and l1_ratio
        alphas = np.logspace(-4, -2, 5) # 1e-4 to 1e-2
        l1_ratios = [0.1, 0.5, 0.9]

        self.models = []
        for alpha in alphas:
            for l1 in l1_ratios:
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("reg", ElasticNet(alpha=alpha, l1_ratio=l1, random_state=42, max_iter=2000))
                ])
                pipe.fit(X_sel, y, reg__sample_weight=sample_weight)
                self.models.append(pipe)

        return self

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, float]:
        """
        Returns (trimmed_median_preds, dispersion_metric)
        """
        tprint(f"Entering function: predict in model_tf.py")
        if not self.models or self.selected_features is None:
            raise ValueError("TF Model not fitted")

        X_sel = X[self.selected_features]

        preds_mat = []
        for model in self.models:
            preds_mat.append(model.predict(X_sel))

        preds_mat = np.array(preds_mat) # (n_models, n_samples)

        # Trimmed Median (remove top/bottom 2 models?)
        # Or simple median. User said "trimmed median".
        # Let's sort along axis 0 and take mean of center 50%?
        # Or just median. Median is robust.
        # "trimmed mean" is common. "trimmed median" is redundant unless we mean mean of truncated set.
        # I'll implement trimmed mean of the middle 50% (IQR range).

        # Sort along models
        preds_sorted = np.sort(preds_mat, axis=0)
        n_models = len(self.models) # 15
        lower = int(n_models * 0.25)
        upper = int(n_models * 0.75)

        # Take mean of middle slice
        trimmed_preds = preds_sorted[lower:upper, :].mean(axis=0)

        # Dispersion = IQR
        iqr = preds_sorted[upper, :] - preds_sorted[lower, :]
        dispersion = np.mean(iqr) # Scalar metric for the batch?
        # Or per-sample dispersion? The user said "tf_dispersion = IQR(preds) (very useful diagnostic + potential gating)"
        # This implies we might want to return per-sample dispersion or just log it.
        # But prediction usually returns 1D array.
        # I'll return the predictions and the MEAN dispersion of this batch as a diagnostic.

        return trimmed_preds, dispersion
