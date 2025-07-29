import logging
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

from .base_ensemble import BaseEnsemble

class HighImpactCandleEnsemble(BaseEnsemble):
    """
    This class focuses on defining the base models relevant for predicting
    the outcome of high-impact candles, such as TabNet.
    """
    def __init__(self, config: dict, ensemble_name: str = "HighImpactCandleEnsemble"):
        super().__init__(config, ensemble_name)

    def _train_base_models(self, aligned_data: pd.DataFrame, y_encoded: np.ndarray):
        """Trains base models for high-impact candles."""
        self.logger.info("Training HighImpactCandle base models...")
        
        X_flat = aligned_data[self.flat_features].fillna(0)

        # TabNet Model
        try:
            self.models["tabnet"] = self._train_tabnet_model(X_flat, y_encoded)
        except Exception as e:
            self.logger.error(f"TabNet training failed: {e}")
        
        # A second, standard LGBM model for diversity
        try:
            self.models["candle_lgbm"] = LGBMClassifier(random_state=42, verbose=-1).fit(X_flat, y_encoded)
        except Exception as e:
            self.logger.error(f"Candle LGBM training failed: {e}")

    def _get_meta_features(self, df: pd.DataFrame, is_live: bool = False, **kwargs) -> pd.DataFrame or dict:
        """Generates meta-features for the high-impact candle meta-learner."""
        
        for col in self.flat_features:
            if col not in df.columns:
                df[col] = 0.0
        
        X_flat = df[self.flat_features].fillna(0)

        if is_live:
            meta_features = {}
            current_row = X_flat.tail(1)
            if self.models.get("tabnet"):
                meta_features['tabnet_prob'] = np.max(self.models["tabnet"].predict_proba(current_row.values))
            if self.models.get("candle_lgbm"):
                meta_features['candle_lgbm_prob'] = self.models["candle_lgbm"].predict_proba(current_row)[0][1]
            meta_features.update(current_row.iloc[0].to_dict())
            return meta_features
        else:
            meta_df = pd.DataFrame(index=df.index)
            if self.models.get("tabnet"):
                meta_df['tabnet_prob'] = np.max(self.models["tabnet"].predict_proba(X_flat.values), axis=1)
            if self.models.get("candle_lgbm"):
                meta_df['candle_lgbm_prob'] = self.models["candle_lgbm"].predict_proba(X_flat)[:, 1]
            meta_df = meta_df.join(X_flat)
            return meta_df.fillna(0)
            
    def _train_tabnet_model(self, X_flat, y_flat_encoded):
        try:
            model = TabNetClassifier()
            model.fit(X_flat.values, y_flat_encoded, max_epochs=50, patience=20, batch_size=1024, virtual_batch_size=128)
            return model
        except Exception as e:
            self.logger.error(f"TabNet training failed: {e}")
            return None
