import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from extreme_price_movements.utils import tprint
from extreme_price_movements.feature_selection_extreme_events import (
    mdi_feature_selection_v3,
    mdi_feature_selection_v4_topk
)

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
    trend = df.get("a_trend", df.get("trend_pct", 0.0)).abs().clip(upper=cap) * 100 # scale up? trend is usually small pct
    g_trend = df.get("G_TREND", 0.0)
    g_vol = df.get("G_VOL", 0.0)

    try:
        # Diagnostic logging for weights inputs
        p_exh_mean = p_exh.mean() if hasattr(p_exh, "mean") else p_exh
        trend_mean = trend.mean() if hasattr(trend, "mean") else trend
        tprint(f"TF Weights inputs: p_exh mean={p_exh_mean:.4f}, trend mean={trend_mean:.4f}")
    except Exception as e:
        tprint(f"TF Weights inputs logging failed: {e}")

    w = 1.0 + (a * (1.0 - p_exh)) + (b * trend) + (c * g_trend)
    w *= (1.0 - 0.3 * g_vol)

    w = w.clip(0.25, 4.0)
    w = w * (len(w) / w.sum())

    try:
        tprint(f"Computed TF weights: min={w.min():.4f}, max={w.max():.4f}, mean={w.mean():.4f}")
    except Exception as e:
        tprint(f"TF Weights output logging failed: {e}")

    return w.to_numpy(dtype=np.float32)

class TFModel:
    def __init__(self, lasso_alpha=0.001):
        tprint(f"Entering function: __init__ in model_tf.py")
        self.lasso_alpha = lasso_alpha
        self.models = []
        self.selected_features = None
        self.last_dispersion = None  # Store dispersion from last prediction

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray = None):
        # 1. Feature Selection (MDI)
        tprint(f"Entering function: fit in model_tf.py. X shape: {X.shape}, y shape: {y.shape}")

        n_samples = len(X)
        n_select = min(60, max(1, n_samples // 100))
        tprint(f"TFModel: Running MDI feature selection. Target features={n_select}")

        base_selector = ExtraTreesRegressor(
            n_estimators=500, # Increased per v3 request
            max_depth=None,   # Let v3 suggest depth
            min_samples_leaf=50,
            max_features='sqrt',
            n_jobs=2,
            random_state=42
        )

        # Two-stage feature selection:
        # Stage 1: Use v3 to get 2x target features
        # Stage 2: Use v4_topk to refine to target count
        n_stage1 = min(X.shape[1], n_select * 2)
        
        sel_res_stage1 = mdi_feature_selection_v3(
            X, y,
            base_model=base_selector,
            sample_weight=sample_weight,
            analysis_n_estimators=500,
            end_features=n_stage1,
            cumulative_cap=0.98,
            min_share=0.001,
            min_features=5,
            max_features_pct=0.8
        )
        
        # If we got more than target features, refine with v4_topk
        if len(sel_res_stage1.selected_features) > n_select:
            tprint(f"TFModel: Refining {len(sel_res_stage1.selected_features)} features with v4_topk")
            X_stage1 = X[sel_res_stage1.selected_features]
            
            sel_res = mdi_feature_selection_v4_topk(
                X_stage1, y,
                base_model=base_selector,
                sample_weight=sample_weight,
                topk_weight=0.3
            )
            self.selected_features = sel_res.selected_features[:n_select]
        else:
            self.selected_features = sel_res_stage1.selected_features[:n_select]

        tprint(f"TFModel: Selected {len(self.selected_features)} features.")
        if len(self.selected_features) > 0:
             tprint(f"Top features: {self.selected_features[:5]}")

        X_sel = X[self.selected_features]

        # 2. Train Ensemble
        # 15 models: varying alpha and l1_ratio
        alphas = np.logspace(-4, -2, 5) # 1e-4 to 1e-2
        l1_ratios = [0.1, 0.5, 0.9]

        tprint("TFModel: Starting training of 15 ElasticNet models...")
        self.models = []
        for alpha in alphas:
            for l1 in l1_ratios:
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("reg", ElasticNet(alpha=alpha, l1_ratio=l1, random_state=42, max_iter=2000))
                ])
                pipe.fit(X_sel, y, reg__sample_weight=sample_weight)
                self.models.append(pipe)

        tprint(f"TFModel: Finished training {len(self.models)} models.")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns trimmed mean predictions across ensemble.
        Dispersion metric stored in self.last_dispersion attribute.
        """
        tprint(f"Entering function: predict in model_tf.py. X shape: {X.shape}")
        if not self.models or self.selected_features is None:
            raise ValueError("TF Model not fitted")

        X_sel = X[self.selected_features]

        preds_mat = []
        for model in self.models:
            preds_mat.append(model.predict(X_sel))

        preds_mat = np.array(preds_mat) # (n_models, n_samples)

        # Sort along models
        preds_sorted = np.sort(preds_mat, axis=0)
        n_models = len(self.models) # 15
        lower = int(n_models * 0.25)
        upper = int(n_models * 0.75)

        # Take mean of middle slice (trimmed mean)
        trimmed_preds = preds_sorted[lower:upper, :].mean(axis=0)

        # Compute dispersion = IQR (stored for diagnostics)
        iqr = preds_sorted[upper, :] - preds_sorted[lower, :]
        self.last_dispersion = np.mean(iqr)

        tprint(f"TFModel prediction: mean_pred={trimmed_preds.mean():.4f}, dispersion={self.last_dispersion:.4f}")
        return trimmed_preds
