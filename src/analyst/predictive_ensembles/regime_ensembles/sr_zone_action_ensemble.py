import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

from .base_ensemble import BaseEnsemble


class SRZoneActionEnsemble(BaseEnsemble):
    """
    This ensemble trains and combines signals from a specialized LGBM,
    a TabNet classifier, and a rule-based volume check to make a more
    resilient final decision on S/R interactions.
    """

    def __init__(self, config: dict, ensemble_name: str = "SRZoneActionEnsemble"):
        super().__init__(config, ensemble_name)
        self.models = {
            "sr_lgbm": None,
            "sr_tabnet": None,
            "order_flow_lgbm": None,
        }

    def _train_base_models(self, aligned_data: pd.DataFrame, y_encoded: np.ndarray):
        """Trains multiple diverse base models for S/R zone action."""
        self.logger.info("Training SRZoneAction base models...")

        X_flat = aligned_data[self.flat_features].fillna(0)

        # Train a general LGBM on all flat features
        lgbm_params = self._tune_hyperparameters(
            LGBMClassifier,
            self._get_lgbm_search_space,
            X_flat,
            y_encoded,
        )
        self.models["sr_lgbm"] = self._train_with_smote(
            LGBMClassifier(**lgbm_params, random_state=42, verbose=-1),
            X_flat,
            y_encoded,
        )

        # Train TabNet Model for a different perspective
        try:
            tabnet = TabNetClassifier()
            tabnet.fit(
                X_flat.values,
                y_encoded,
                max_epochs=50,
                patience=20,
                batch_size=1024,
            )
            self.models["sr_tabnet"] = tabnet
        except Exception as e:
            self.logger.error(f"S/R TabNet training failed: {e}")

        # Train a specialized Order Flow LGBM
        X_of = aligned_data[self.order_flow_features].fillna(0)
        of_params = self._tune_hyperparameters(
            LGBMClassifier,
            self._get_lgbm_search_space,
            X_of,
            y_encoded,
        )
        self.models["order_flow_lgbm"] = self._train_with_smote(
            LGBMClassifier(**of_params, random_state=42, verbose=-1),
            X_of,
            y_encoded,
        )

    def _get_meta_features(
        self,
        df: pd.DataFrame,
        is_live: bool = False,
        **kwargs,
    ) -> pd.DataFrame | dict:
        """Generates meta-features from all base models for the S/R zone meta-learner."""

        X_flat = df[self.flat_features].fillna(0)
        X_of = df[self.order_flow_features].fillna(0)

        if is_live:
            meta_features = {}
            current_row_flat = X_flat.tail(1)
            current_row_of = X_of.tail(1)

            if self.models.get("sr_lgbm"):
                meta_features["sr_lgbm_prob"] = np.max(
                    self.models["sr_lgbm"].predict_proba(current_row_flat),
                )
            if self.models.get("sr_tabnet"):
                meta_features["sr_tabnet_prob"] = np.max(
                    self.models["sr_tabnet"].predict_proba(current_row_flat.values),
                )
            if self.models.get("order_flow_lgbm"):
                meta_features["of_lgbm_prob"] = np.max(
                    self.models["order_flow_lgbm"].predict_proba(current_row_of),
                )

            # Add all raw features for the meta-learner to consider
            meta_features.update(current_row_flat.iloc[0].to_dict())
            return meta_features
        meta_df = pd.DataFrame(index=df.index)
        if self.models.get("sr_lgbm"):
            meta_df["sr_lgbm_prob"] = np.max(
                self.models["sr_lgbm"].predict_proba(X_flat),
                axis=1,
            )
        if self.models.get("sr_tabnet"):
            meta_df["sr_tabnet_prob"] = np.max(
                self.models["sr_tabnet"].predict_proba(X_flat.values),
                axis=1,
            )
        if self.models.get("order_flow_lgbm"):
            meta_df["of_lgbm_prob"] = np.max(
                self.models["order_flow_lgbm"].predict_proba(X_of),
                axis=1,
            )

        # Join all raw features
        meta_df = meta_df.join(X_flat)
        return meta_df.fillna(0)
