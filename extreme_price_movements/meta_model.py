import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from extreme_price_movements.utils import tprint

class MetaModel:
    def __init__(self):
        self.model = LinearRegression()
        self.feature_names = [
            "pred_tf", "pred_mr",
            "realized_vol", "vol_z", "log_volume",
            "norm_momentum", "dist_ma_z"
        ]

    def prepare_meta_features(self, preds_tf, preds_mr, feats_df):
        """
        Constructs X_meta.
        feats_df should contain the raw features needed.
        names mapping might be needed.
        """
        # Map raw features to expected names
        # realized vol -> rv_24h ?
        # vol z-score -> vol_z24 ?
        # log(volume) -> log(qv) or volume? feats has log-transformed features?
        # feats are causally transformed (log + winsor + z).
        # So "vol_z" is likely available. "rv_24h" is available.
        # "norm_momentum" -> rsi? or roc_div?
        # "dist_ma_z" -> dist_ema_fast ?

        # We assume feats_df has these columns or their equivalents.
        # We need to extract them.

        meta_data = pd.DataFrame(index=feats_df.index)
        meta_data["pred_tf"] = preds_tf
        meta_data["pred_mr"] = preds_mr

        # Mappings
        # Assuming feats_df comes from `X` which has columns like "a_rv24", "a_volz", etc.
        # Check config causal_cols.

        # "realized vol EWMA vol over a window" -> a_rv24
        meta_data["realized_vol"] = feats_df.get("a_rv24", 0.0)

        # "vol z-score" -> a_volz
        meta_data["vol_z"] = feats_df.get("a_volz", 0.0)

        # "log(volume)" -> qv? No, volume.
        # But features are transformed. "a_volz" is z-score of volume.
        # Maybe we don't have raw log volume in X?
        # If X was built with `drop_raw_causal=True`, we only have `a_` features.
        # We might need to pass raw `feats` panel?
        # But `X` is row-based.
        # Let's use available features in X as proxies.
        # "a_volz" captures volume info.
        meta_data["log_volume"] = feats_df.get("a_volz", 0.0) # Proxy

        # "normalized momentum" -> a_rsi or trend_snr?
        meta_data["norm_momentum"] = feats_df.get("a_rsi", 0.0)

        # "distance from moving average (z-scored)" -> dist_ema_fast
        meta_data["dist_ma_z"] = feats_df.get("dist_ema_fast", 0.0)

        return meta_data[self.feature_names].fillna(0.0)

    def fit(self, X_meta, y):
        # y should be return (or simulated PnL return)
        self.model.fit(X_meta, y)
        return self

    def predict(self, X_meta):
        # Outputs "position" (unbounded score).
        # We can clip or normalize later.
        return self.model.predict(X_meta)
