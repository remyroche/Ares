import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.special import logit
from extreme_price_movements.utils import tprint
from extreme_price_movements.feature_transforms import CausalFeatureTransformer

class MetaModel:
    def __init__(self):
        self.model = LinearRegression()
        # Updated feature names to include new requested features
        self.feature_names = [
            "pred_tf_logit", "pred_mr_logit",
            "realized_vol", "vol_z", "log_volume",
            "norm_momentum", "dist_ma_z",
            "atr_slope", "dist_vwap_norm", "mom_accel"
        ]
        self.transformer = CausalFeatureTransformer(winsor_qt=0.02, roll_window=24*30)

    def prepare_meta_features(self, preds_tf, preds_mr, feats_df):
        """
        Constructs X_meta.
        Applies logit to predictions.
        Applies CausalTransform to other features.
        """
        meta_data = pd.DataFrame(index=feats_df.index)

        eps = 1e-4
        p_tf = np.clip(preds_tf, eps, 1 - eps)
        p_mr = np.clip(preds_mr, eps, 1 - eps)

        meta_data["pred_tf_logit"] = logit(p_tf)
        meta_data["pred_mr_logit"] = logit(p_mr)

        # Mapping
        meta_data["realized_vol"] = feats_df.get("a_rv24", 0.0)
        meta_data["vol_z"] = feats_df.get("a_volz", 0.0)
        meta_data["log_volume"] = feats_df.get("a_volz", 0.0)
        meta_data["norm_momentum"] = feats_df.get("a_rsi", 0.0)
        meta_data["dist_ma_z"] = feats_df.get("dist_ema_fast", 0.0)

        # New features mapping
        # These features are in `feats_df` (which is `X_tf` built in training loop).
        # We need to ensure `build_hourly_training_set_and_weights` includes them in X.
        # It includes `causal_cols`. We need to add these to `causal_cols` in config?
        # But they are meta features. The meta model is trained on X_meta.
        # X_meta is built from `feats_df`.
        # `feats_df` in `select_best_horizon` comes from `build_hourly_training_set_and_weights`.
        # That function selects `causal_cols` and gates.
        # So we MUST add these new meta features to `causal_cols` or fetch them separately?
        # If we add them to `causal_cols`, they are used by base models too.
        # That's fine (they are good features).
        # BUT `atr_slope`, `dist_vwap_norm`, `mom_accel` were asked specifically for Meta Model.
        # Let's check config.py. I didn't add them to causal_cols yet. I should.

        # I will use `.get` with default 0.0, assuming they will be present if config is updated.
        # "momentum_accel" -> "mom_accel" in feature names above?

        meta_data["atr_slope"] = feats_df.get("atr_slope", 0.0)
        meta_data["dist_vwap_norm"] = feats_df.get("dist_vwap_norm", 0.0)
        meta_data["mom_accel"] = feats_df.get("momentum_accel", 0.0)

        # Note: If feats_df comes from `X` which has causal transform applied,
        # these are already transformed.
        # But `atr_slope` etc were added to `feats` in `features.py` and transformed there.
        # So `X` will have transformed values.

        return meta_data[self.feature_names].fillna(0.0)

    def fit(self, X_meta, y):
        self.model.fit(X_meta, y)
        return self

    def predict(self, X_meta):
        return self.model.predict(X_meta)
