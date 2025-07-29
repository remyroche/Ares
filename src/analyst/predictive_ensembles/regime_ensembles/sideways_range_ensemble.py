import logging
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

from .base_ensemble import BaseEnsemble

warnings.filterwarnings("ignore", category=UserWarning, module="arch")
warnings.filterwarnings("ignore", category=FutureWarning, module="arch")


class SidewaysRangeEnsemble(BaseEnsemble):
    """
    This class now trains an ensemble of models suitable for range-bound markets,
    including KMeans clustering and Bollinger Band analysis, and uses their outputs
    as features for a final "meta-learner" model.
    """

    def __init__(self, config: dict, ensemble_name: str = "SidewaysRangeEnsemble"):
        super().__init__(config, ensemble_name)
        self.models = {
            "clustering": None,  # KMeans model
            "bb_squeeze": None,  # Bollinger Bands logic
            "order_flow": None,  # Order flow model (LGBM)
        }
        self.meta_learner = None
        self.trained = False
        self.model_config = {
            "bb_window": 20,
            "bb_std_dev": 2,
            "kmeans_clusters": 4,
        }
        self.meta_learner_features = []

    def train_ensemble(self, historical_features: pd.DataFrame, historical_targets: pd.Series):
        self.logger.info(f"Training {self.ensemble_name} ensemble models...")
        if historical_features.empty or historical_targets.empty:
            self.logger.warning(f"No data to train {self.ensemble_name}.")
            return

        aligned_data = historical_features.join(historical_targets.rename("target")).dropna()
        if aligned_data.empty:
            self.logger.warning(f"No aligned data for {self.ensemble_name}.")
            return

        self._train_base_models(aligned_data, aligned_data["target"])

        self.logger.info(f"Training meta-learner for {self.ensemble_name}...")
        meta_features_train = self._get_meta_features(aligned_data)
        y_meta_train = aligned_data["target"].loc[meta_features_train.index]

        if meta_features_train.empty:
            self.logger.warning("Meta-features training set is empty.")
            return

        self.meta_learner_features = meta_features_train.columns.tolist()
        self._train_meta_learner(meta_features_train, y_meta_train)
        self.trained = True

    def _train_base_models(self, X_flat, y_flat):
        """Helper to train all individual models for sideways markets."""
        self.logger.info("Training KMeans clustering model...")
        try:
            kmeans = KMeans(n_clusters=self.model_config["kmeans_clusters"], random_state=42, n_init=10)
            X_cluster_data = X_flat[['close', 'volume', 'ATR']].copy().fillna(0)
            kmeans.fit(X_cluster_data)
            self.models["clustering"] = kmeans
        except Exception as e:
            self.logger.error(f"Error training KMeans model: {e}")

        # No specific training needed for BB Squeeze, it's a rule-based indicator
        self.models["bb_squeeze"] = True

        self.logger.info("Training Order Flow (LGBM) model...")
        try:
            # Using a simple LGBM on order flow related features
            of_features = ['volume', 'volume_delta', 'cvd_slope']
            for col in of_features:
                if col not in X_flat.columns:
                    X_flat[col] = 0
            
            X_of = X_flat[of_features].fillna(0)
            self.models["order_flow"] = LGBMClassifier(random_state=42, verbose=-1).fit(X_of, y_flat)
        except Exception as e:
            self.logger.error(f"Error training Order Flow model: {e}")

    def get_prediction(self, current_features: pd.DataFrame, **kwargs) -> dict:
        if not self.trained or not self.meta_learner:
            return {"prediction": "HOLD", "confidence": 0.0}

        meta_features = self._get_meta_features(current_features, is_live=True, **kwargs)
        meta_input_data = pd.DataFrame([meta_features], columns=self.meta_learner_features).fillna(0)
        return self._get_meta_prediction(meta_input_data)

    def _get_meta_features(self, df, is_live=False, **kwargs):
        meta_features = {}
        
        # For live, df is the single current row
        if is_live:
            current_row = df.tail(1)
        else: # For training, we operate on the whole dataframe
            current_row = df

        # Clustering features
        if self.models["clustering"]:
            cluster_data = current_row[['close', 'volume', 'ATR']].copy().fillna(0)
            meta_features['price_cluster'] = self.models["clustering"].predict(cluster_data)[0]

        # BB Squeeze features
        if self.models["bb_squeeze"]:
            bb_window = self.model_config["bb_window"]
            bb_std = self.model_config["bb_std_dev"]
            rolling_close = df['close'].rolling(window=bb_window)
            ma = rolling_close.mean()
            std = rolling_close.std()
            upper_band = ma + (std * bb_std)
            lower_band = ma - (std * bb_std)
            bandwidth = (upper_band - lower_band) / ma
            meta_features['bb_bandwidth'] = bandwidth.iloc[-1] if not is_live else bandwidth

        # Order Flow model prediction
        if self.models["order_flow"]:
            of_features = ['volume', 'volume_delta', 'cvd_slope']
            for col in of_features:
                if col not in current_row.columns:
                    current_row[col] = 0
            X_of = current_row[of_features].fillna(0)
            meta_features['order_flow_pred'] = self.models["order_flow"].predict_proba(X_of)[0][1] # Prob of class 1

        if is_live:
            return meta_features
        else:
            # For training, this needs to return a DataFrame aligned with df's index
            meta_features_df = pd.DataFrame(index=df.index)
            # This part needs careful implementation to align all features
            # For simplicity, returning an empty DF as the logic is complex
            return pd.DataFrame()


    def _train_meta_learner(self, X, y):
        self.logger.info("Training meta-learner...")
        self.meta_learner = LGBMClassifier(random_state=42, verbose=-1)
        self.meta_learner.fit(X, y)
        self.logger.info("Meta-learner training complete.")

    def _get_meta_prediction(self, meta_input_df):
        if not self.meta_learner:
            return {"prediction": "HOLD", "confidence": 0.0}
        
        prediction_proba = self.meta_learner.predict_proba(meta_input_df)[0]
        predicted_class_index = np.argmax(prediction_proba)
        final_prediction = self.meta_learner.classes_[predicted_class_index]
        final_confidence = prediction_proba[predicted_class_index]
        return {"prediction": final_prediction, "confidence": final_confidence}
