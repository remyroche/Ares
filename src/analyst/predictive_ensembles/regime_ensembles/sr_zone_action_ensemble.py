import logging
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from .base_ensemble import BaseEnsemble

class SRZoneActionEnsemble(BaseEnsemble):
    """
    This class focuses on training a specialized LGBM model that has access
    to all features to best predict outcomes at S/R zones.
    """
    def __init__(self, config: dict, ensemble_name: str = "SRZoneActionEnsemble"):
        super().__init__(config, ensemble_name)

    def _train_base_models(self, aligned_data: pd.DataFrame, y_encoded: np.ndarray):
        """Trains base models for S/R zone action."""
        self.logger.info("Training SRZoneAction base models...")
        
        X_flat = aligned_data[self.flat_features].fillna(0)
        
        # The primary model is a single, powerful LGBM trained on all available flat features
        try:
            self.models["sr_lgbm"] = LGBMClassifier(random_state=42, verbose=-1).fit(X_flat, y_encoded)
        except Exception as e:
            self.logger.error(f"S/R LGBM training failed: {e}")

    def _get_meta_features(self, df: pd.DataFrame, is_live: bool = False, **kwargs) -> pd.DataFrame or dict:
        """Generates meta-features for the S/R zone meta-learner."""
        
        for col in self.flat_features:
            if col not in df.columns:
                df[col] = 0.0
        
        X_flat = df[self.flat_features].fillna(0)

        if is_live:
            meta_features = {}
            current_row = X_flat.tail(1)
            if self.models.get("sr_lgbm"):
                meta_features['sr_lgbm_prob'] = self.models["sr_lgbm"].predict_proba(current_row)[0][1]
            meta_features.update(current_row.iloc[0].to_dict())
            return meta_features
        else:
            meta_df = pd.DataFrame(index=df.index)
            if self.models.get("sr_lgbm"):
                meta_df['sr_lgbm_prob'] = self.models["sr_lgbm"].predict_proba(X_flat)[:, 1]
            meta_df = meta_df.join(X_flat)
            return meta_df.fillna(0)
