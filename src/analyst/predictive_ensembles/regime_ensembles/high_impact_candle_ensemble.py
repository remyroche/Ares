import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.naive_bayes import GaussianNB

from src.utils.warning_symbols import (
    failed,
)

from .base_ensemble import BaseEnsemble


class HighImpactCandleEnsemble(BaseEnsemble):
    """
    This ensemble now combines signals from TabNet, a specialized LGBM, an order flow
    model, and a Naive Bayes classifier for a robust, multi-faceted prediction.
    """

    def __init__(self, config: dict, ensemble_name: str = "HighImpactCandleEnsemble"):
        super().__init__(config, ensemble_name)
        self.models = {
            "tabnet": None,
            "candle_lgbm": None,
            "order_flow_lgbm": None,
            "naive_bayes": None,
        }

    def _train_base_models(self, aligned_data: pd.DataFrame, y_encoded: np.ndarray):
        """Trains multiple diverse base models for high-impact candles."""
        self.logger.info("Training HighImpactCandle base models...")
        X_flat = aligned_data[self.flat_features].fillna(0)

        # TabNet Model
        try:
            tabnet = TabNetClassifier()
            tabnet.fit(
                X_flat.values,
                y_encoded,
                max_epochs=50,
                patience=20,
                batch_size=1024,
            )
            self.models["tabnet"] = tabnet
        except Exception:
            self.print(failed("Candle TabNet training failed: {e}"))

        # General LGBM Model
        lgbm_params = self._tune_hyperparameters(
            LGBMClassifier,
            self._get_lgbm_search_space,
            X_flat,
            y_encoded,
        )
        self.models["candle_lgbm"] = self._train_with_smote(
            LGBMClassifier(**lgbm_params, random_state=42, verbose=-1),
            X_flat,
            y_encoded,
        )

        # Specialized Order Flow LGBM
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

        # Gaussian Naive Bayes for a probabilistic, non-tree-based perspective
        self.logger.info("Training Gaussian Naive Bayes model...")
        self.models["naive_bayes"] = self._train_with_smote(
            GaussianNB(),
            X_flat,
            y_encoded,
        )

    def _get_meta_features(
        self,
        df: pd.DataFrame,
        is_live: bool = False,
        **kwargs,
    ) -> pd.DataFrame | dict:
        """Generates meta-features from all base models for the high-impact candle meta-learner."""
        X_flat = df[self.flat_features].fillna(0)
        X_of = df[self.order_flow_features].fillna(0)
        meta_label_cols = [
            c for c in df.columns if any(
                c.startswith(prefix) and c.endswith(suffix)
                for suffix in [
                    "STRONG_TREND_CONTINUATION",
                    "RANGE_MEAN_REVERSION",
                    "EXHAUSTION_REVERSAL",
                ]
                for prefix in ["1m_", "5m_", "15m_", "30m_"]
            )
        ]

        if is_live:
            meta = {}
            current_row_flat = X_flat.tail(1)
            current_row_of = X_of.tail(1)
            if self.models.get("tabnet"):
                meta["tabnet_prob"] = np.max(
                    self.models["tabnet"].predict_proba(current_row_flat.values),
                )
            if self.models.get("candle_lgbm"):
                meta["candle_lgbm_prob"] = np.max(
                    self.models["candle_lgbm"].predict_proba(current_row_flat),
                )
            if self.models.get("order_flow_lgbm"):
                meta["of_lgbm_prob"] = np.max(
                    self.models["order_flow_lgbm"].predict_proba(current_row_of),
                )
            if self.models.get("naive_bayes"):
                meta["nb_prob"] = np.max(
                    self.models["naive_bayes"].predict_proba(current_row_flat),
                )
            meta.update(current_row_flat.iloc[0].to_dict())
            for col in meta_label_cols:
                try:
                    meta[col] = int(df[col].iloc[-1])
                except (KeyError, IndexError):
                    meta[col] = 0
            if meta_label_cols:
                self.logger.info(f"HighImpactCandleEnsemble meta-learner live features include meta-labels: {meta_label_cols}")
            return meta
        meta_df = pd.DataFrame(index=df.index)
        if self.models.get("tabnet"):
            meta_df["tabnet_prob"] = np.max(
                self.models["tabnet"].predict_proba(X_flat.values),
                axis=1,
            )
        if self.models.get("candle_lgbm"):
            meta_df["candle_lgbm_prob"] = np.max(
                self.models["candle_lgbm"].predict_proba(X_flat),
                axis=1,
            )
        if self.models.get("order_flow_lgbm"):
            meta_df["of_lgbm_prob"] = np.max(
                self.models["order_flow_lgbm"].predict_proba(X_of),
                axis=1,
            )
        if self.models.get("naive_bayes"):
            meta_df["nb_prob"] = np.max(
                self.models["naive_bayes"].predict_proba(X_flat),
                axis=1,
            )
        meta_df = meta_df.join(X_flat)
        if meta_label_cols:
            meta_df = meta_df.join(df[meta_label_cols].astype(float))
            self.logger.info(f"HighImpactCandleEnsemble meta-learner train features include meta-labels: {meta_label_cols}")
        return meta_df.fillna(0)
