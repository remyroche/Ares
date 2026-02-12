import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, HuberRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from extreme_price_movements.utils import tprint
from extreme_price_movements.feature_selection_extreme_events import (
    mdi_feature_selection_v3,
    mdi_feature_selection_v4_topk
)


def compute_mr_weights(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """
    Computes sample weights for MR model favoring exhaustion and snapback.
    w_mr = 1 + a * p_exh_lag1 + b * clip(|ret1h_z|,0,zmax) + c * clip(vol_z24,0,vmax)
    Optionally downweight strong trend.
    """
    tprint(f"Entering function: compute_mr_weights in model_mr.py")
    # defaults
    a = 2.0  # weight for exhaustion
    b = 0.5  # weight for return deviation
    c = 0.5  # weight for vol
    zmax = 4.0
    vmax = 4.0

    p_exh = df.get("p_exh_lag1", 0.0)
    ret_z = df.get("a_ret1h_z", df.get("ret1h_z", 0.0)).abs().clip(upper=zmax)
    vol_z = df.get("a_volz", df.get("vol_z", 0.0)).clip(upper=vmax)
    g_trend = df.get("G_TREND", 0.0)

    w = 1.0 + (a * p_exh) + (b * ret_z) + (c * vol_z)

    # downweight strong trend
    w *= (1.0 - 0.5 * g_trend)

    # Clip and normalize
    w = w.clip(0.25, 4.0)
    w = w * (len(w) / w.sum())
    tprint(f"MR Weights: min={w.min():.4f}, mean={w.mean():.4f}, max={w.max():.4f}")
    return w.to_numpy(dtype=np.float32)

class MRModel:
    def __init__(self, lasso_alpha=0.001, huber_epsilon=1.35):
        tprint(f"Entering function: __init__ in model_mr.py")
        self.lasso_alpha = lasso_alpha
        self.huber_epsilon = huber_epsilon
        self.model = None
        self.selected_features = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray = None):
        # 1. Feature Selection (MDI)
        tprint(f"Entering function: fit in model_mr.py")

        tprint(f"MRModel.fit: Input X shape: {X.shape}, y shape: {y.shape}")
        n_samples = len(X)
        n_select = min(60, max(1, n_samples // 100))
        tprint(f"MRModel: Running MDI feature selection. Target features={n_select}")

        base_selector = ExtraTreesRegressor(
            n_estimators=500, # Increased per v3
            max_depth=None,   # Let v3 suggest depth
            min_samples_leaf=50,
            max_features='sqrt',
            n_jobs=-1,
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
            tprint(f"MRModel: Refining {len(sel_res_stage1.selected_features)} features with v4_topk")
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

        tprint(f"MRModel: Selected {len(self.selected_features)} features.")
        X_sel = X[self.selected_features]

        # 2. Huber Regressor
        # Scale again? Pipeline handles it.
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", HuberRegressor(epsilon=self.huber_epsilon, max_iter=200))
        ])

        self.model.fit(X_sel, y, reg__sample_weight=sample_weight)
        tprint("MRModel.fit: HuberRegressor fit complete.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        tprint(f"Entering function: predict in model_mr.py")
        tprint(f"MRModel.predict: predicting for {len(X)} samples.")
        if self.model is None or self.selected_features is None:
            raise ValueError("MR Model not fitted")

        X_sel = X[self.selected_features]
        # Missing cols handling?
        # Assuming X has them.
        return self.model.predict(X_sel)
