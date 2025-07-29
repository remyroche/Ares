import logging
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans

from .base_ensemble import BaseEnsemble

class SidewaysRangeEnsemble(BaseEnsemble):
    """
    This class focuses solely on defining the base models relevant for a
    sideways market, like KMeans clustering and a specialized order flow model.
    """
    def __init__(self, config: dict, ensemble_name: str = "SidewaysRangeEnsemble"):
        super().__init__(config, ensemble_name)
        self.model_config = {"kmeans_clusters": 4}

    def _train_base_models(self, aligned_data: pd.DataFrame, y_encoded: np.ndarray):
        """Trains base models specific to sideways markets."""
        self.logger.info("Training SidewaysRange base models...")
        
        # Use the full, unified feature set for training
        X_flat = aligned_data[self.flat_features].fillna(0)

        # KMeans Clustering Model
        try:
            cluster_features = X_flat[['close', 'volume', 'ATR']]
            self.models["clustering"] = KMeans(
                n_clusters=self.model_config["kmeans_clusters"], random_state=42, n_init=10
            ).fit(cluster_features)
        except Exception as e:
            self.logger.error(f"KMeans training failed: {e}")

        # Order Flow Model (LGBM)
        try:
            self.models["order_flow_lgbm"] = LGBMClassifier(random_state=42, verbose=-1).fit(X_flat, y_encoded)
        except Exception as e:
            self.logger.error(f"Order Flow LGBM training failed: {e}")

    def _get_meta_features(self, df: pd.DataFrame, is_live: bool = False, **kwargs) -> pd.DataFrame or dict:
        """Generates meta-features for the sideways market meta-learner."""
        
        # Ensure all necessary features are present
        for col in self.flat_features:
            if col not in df.columns:
                df[col] = 0.0
        
        X_flat = df[self.flat_features].fillna(0)

        if is_live:
            meta_features = {}
            current_row = X_flat.tail(1)
            if self.models.get("clustering"):
                cluster_data = current_row[['close', 'volume', 'ATR']]
                meta_features['price_cluster'] = self.models["clustering"].predict(cluster_data)[0]
            if self.models.get("order_flow_lgbm"):
                meta_features['order_flow_prob'] = self.models["order_flow_lgbm"].predict_proba(current_row)[0][1]
            # Add all raw features for the meta-learner to consider
            meta_features.update(current_row.iloc[0].to_dict())
            return meta_features
        else:
            meta_df = pd.DataFrame(index=df.index)
            if self.models.get("clustering"):
                cluster_data = X_flat[['close', 'volume', 'ATR']]
                meta_df['price_cluster'] = self.models["clustering"].predict(cluster_data)
            if self.models.get("order_flow_lgbm"):
                meta_df['order_flow_prob'] = self.models["order_flow_lgbm"].predict_proba(X_flat)[:, 1]
            
            # Join all raw features
            meta_df = meta_df.join(X_flat)
            return meta_df.fillna(0)
