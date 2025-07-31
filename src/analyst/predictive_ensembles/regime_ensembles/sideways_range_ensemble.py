import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from .base_ensemble import BaseEnsemble


class SidewaysRangeEnsemble(BaseEnsemble):
    def __init__(self, config: dict, ensemble_name: str = "SidewaysRangeEnsemble"):
        super().__init__(config, ensemble_name)
        self.model_config = {"kmeans_clusters": 4}
        self.models = {"clustering": None, "order_flow_lgbm": None, "svm": None}

    def _train_base_models(self, aligned_data: pd.DataFrame, y_encoded: np.ndarray):
        self.logger.info("Training SidewaysRange base models...")
        X_flat = aligned_data[self.flat_features].fillna(0)

        try:
            cluster_features = X_flat[["close", "volume", "ATR"]]
            self.models["clustering"] = KMeans(
                n_clusters=self.model_config["kmeans_clusters"],
                random_state=42,
                n_init=10,
            ).fit(cluster_features)
        except Exception as e:
            self.logger.error(f"KMeans training failed: {e}")

        X_of = aligned_data[self.order_flow_features].fillna(0)
        of_params = self._tune_hyperparameters(
            LGBMClassifier, self._get_lgbm_search_space, X_of, y_encoded
        )
        self.models["order_flow_lgbm"] = self._train_with_smote(
            LGBMClassifier(**of_params, random_state=42, verbose=-1), X_of, y_encoded
        )

        sample_size = min(len(X_flat), 2000)
        X_sample = X_flat.sample(n=sample_size, random_state=42)
        y_sample = pd.Series(y_encoded).loc[X_sample.index].values
        svm_params = self._tune_hyperparameters(
            SVC, self._get_svm_search_space, X_sample, y_sample
        )
        self.models["svm"] = self._train_with_smote(
            SVC(**svm_params, random_state=42), X_flat, y_encoded
        )

    def _get_meta_features(
        self, df: pd.DataFrame, is_live: bool = False, **kwargs
    ) -> pd.DataFrame | dict:
        X_flat = df[self.flat_features].fillna(0)
        X_of = df[self.order_flow_features].fillna(0)

        if is_live:
            meta = {}
            current_row_flat = X_flat.tail(1)
            current_row_of = X_of.tail(1)
            if self.models.get("clustering"):
                meta["price_cluster"] = self.models["clustering"].predict(
                    current_row_flat[["close", "volume", "ATR"]]
                )[0]
            if self.models.get("order_flow_lgbm"):
                meta["of_lgbm_prob"] = np.max(
                    self.models["order_flow_lgbm"].predict_proba(current_row_of)
                )
            if self.models.get("svm"):
                meta["svm_prob"] = np.max(
                    self.models["svm"].predict_proba(current_row_flat)
                )
            meta.update(current_row_flat.iloc[0].to_dict())
            return meta
        else:
            meta_df = pd.DataFrame(index=df.index)
            if self.models.get("clustering"):
                meta_df["price_cluster"] = self.models["clustering"].predict(
                    X_flat[["close", "volume", "ATR"]]
                )
            if self.models.get("order_flow_lgbm"):
                meta_df["of_lgbm_prob"] = np.max(
                    self.models["order_flow_lgbm"].predict_proba(X_of), axis=1
                )
            if self.models.get("svm"):
                meta_df["svm_prob"] = np.max(
                    self.models["svm"].predict_proba(X_flat), axis=1
                )
            meta_df = meta_df.join(X_flat)
            return meta_df.fillna(0)
