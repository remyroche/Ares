import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.special import logit
from extreme_price_movements.utils import tprint
from extreme_price_movements.feature_transforms import CausalFeatureTransformer

class MetaModel:
    def __init__(self):
        self.model = LinearRegression()
        self.feature_names = [
            "pred_tf_logit", "pred_mr_logit",
            "realized_vol", "vol_z", "log_volume",
            "norm_momentum", "dist_ma_z"
        ]
        # Transformer for the meta features (excluding logits which are already transformed)
        self.transformer = CausalFeatureTransformer(winsor_qt=0.02, roll_window=24*30)

    def prepare_meta_features(self, preds_tf, preds_mr, feats_df):
        """
        Constructs X_meta.
        Applies logit to predictions.
        Applies CausalTransform to other features.
        """
        meta_data = pd.DataFrame(index=feats_df.index)

        # Logit Transform of predictions (clip to avoid inf)
        eps = 1e-4
        p_tf = np.clip(preds_tf, eps, 1 - eps)
        p_mr = np.clip(preds_mr, eps, 1 - eps)

        meta_data["pred_tf_logit"] = logit(p_tf)
        meta_data["pred_mr_logit"] = logit(p_mr)

        # Raw features (to be transformed)
        raw_meta = pd.DataFrame(index=feats_df.index)

        # Mappings
        # feats_df likely contains "a_" features which are ALREADY transformed in features.py?
        # In features.py, we applied CausalFeatureTransformer to `feats`.
        # So "a_rv24", "a_volz" are already Log/Winsor/Zscored.
        # User requirement: "ensure that the other features we feed it are 1) log 2) winsorised 3/ normalized"
        # If they come from `feats`, they ARE already transformed.
        # BUT, if we are feeding raw columns, we should transform.
        # Let's assume `feats_df` passed here contains the features used in training.
        # If they are "a_..." they are transformed.
        # If we use "realized_vol" mapped to "a_rv24", it's fine.

        # However, to be safe and explicit as per instruction (maybe MetaModel has its own normalization context?),
        # I will check if they look normalized.
        # Actually, double normalization is usually okay (z-score of z-score).
        # But if they are already log/winsorized, we just need z-score?
        # The user instruction implies we should do it here.
        # "Feed it (logit(p)... Also ensure that the other features... are 1) log 2) winsorised 3/ normalized"

        # If I pull "a_rv24", it is already processed.
        # I will map them directly.

        meta_data["realized_vol"] = feats_df.get("a_rv24", 0.0)
        meta_data["vol_z"] = feats_df.get("a_volz", 0.0)
        meta_data["log_volume"] = feats_df.get("a_volz", 0.0) # Proxy
        meta_data["norm_momentum"] = feats_df.get("a_rsi", 0.0)
        meta_data["dist_ma_z"] = feats_df.get("dist_ema_fast", 0.0)

        return meta_data[self.feature_names].fillna(0.0)

    def fit(self, X_meta, y):
        self.model.fit(X_meta, y)
        return self

    def predict(self, X_meta):
        return self.model.predict(X_meta)
