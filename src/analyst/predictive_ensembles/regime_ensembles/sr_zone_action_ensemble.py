import logging
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

from .base_ensemble import BaseEnsemble

class SRZoneActionEnsemble(BaseEnsemble):
    """
    ## CHANGE: Diversified the base models for more robust predictions.
    ## This ensemble no longer relies on a single model. It now trains and combines
    ## signals from a specialized LGBM, a TabNet classifier, and a rule-based
    ## volume check to make a more resilient final decision.
    """
    def __init__(self, config: dict, ensemble_name: str = "SRZoneActionEnsemble"):
        super().__init__(config, ensemble_name)
        self.models = {
            "sr_lgbm": None,
            "sr_tabnet": None,
            "volume_check": True # Rule-based model, no training needed
        }

    def _train_base_models(self, aligned_data: pd.DataFrame, y_encoded: np.ndarray):
        """Trains multiple diverse base models for S/R zone action."""
        self.logger.info("Training SRZoneAction base models...")
        
        X_flat = aligned_data[self.flat_features].fillna(0)
        
        # Train LGBM Model
        try:
            self.models["sr_lgbm"] = LGBMClassifier(random_state=42, verbose=-1).fit(X_flat, y_encoded)
        except Exception as e:
            self.logger.error(f"S/R LGBM training failed: {e}")

        # Train TabNet Model for a different perspective
        try:
            tabnet_model = TabNetClassifier()
            tabnet_model.fit(X_flat.values, y_encoded, max_epochs=50, patience=20, batch_size=1024)
            self.models["sr_tabnet"] = tabnet_model
        except Exception as e:
            self.logger.error(f"S/R TabNet training failed: {e}")

    def _get_meta_features(self, df: pd.DataFrame, is_live: bool = False, **kwargs) -> pd.DataFrame or dict:
        """Generates meta-features from all base models for the S/R zone meta-learner."""
        
        for col in self.flat_features:
            if col not in df.columns:
                df[col] = 0.0
        
        X_flat = df[self.flat_features].fillna(0)

        if is_live:
            meta_features = {}
            current_row = X_flat.tail(1)
            if self.models.get("sr_lgbm"):
                meta_features['sr_lgbm_prob'] = self.models["sr_lgbm"].predict_proba(current_row)[0][1]
            if self.models.get("sr_tabnet"):
                meta_features['sr_tabnet_prob'] = np.max(self.models["sr_tabnet"].predict_proba(current_row.values))
            
            # Add rule-based feature
            volume_ma = df['volume'].rolling(20).mean().iloc[-1]
            meta_features['is_high_volume_at_sr'] = int(current_row['volume'].iloc[-1] > volume_ma * 1.5 and current_row.get('Is_SR_Interacting', 0).iloc[-1] == 1)

            meta_features.update(current_row.iloc[0].to_dict())
            return meta_features
        else:
            meta_df = pd.DataFrame(index=df.index)
            if self.models.get("sr_lgbm"):
                meta_df['sr_lgbm_prob'] = self.models["sr_lgbm"].predict_proba(X_flat)[:, 1]
            if self.models.get("sr_tabnet"):
                meta_df['sr_tabnet_prob'] = np.max(self.models["sr_tabnet"].predict_proba(X_flat.values), axis=1)

            # Add rule-based feature
            volume_ma = df['volume'].rolling(20).mean()
            meta_df['is_high_volume_at_sr'] = ((df['volume'] > volume_ma * 1.5) & (df.get('Is_SR_Interacting', 0) == 1)).astype(int)

            meta_df = meta_df.join(X_flat)
            return meta_df.fillna(0)
