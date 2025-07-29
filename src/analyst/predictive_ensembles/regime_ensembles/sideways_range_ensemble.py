import logging
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans

from .base_ensemble import BaseEnsemble

warnings.filterwarnings("ignore", category=UserWarning, module="arch")
warnings.filterwarnings("ignore", category=FutureWarning, module="arch")


class SidewaysRangeEnsemble(BaseEnsemble):
    """
    ## CHANGE: Enriched the feature set and fully implemented training logic.
    ## This ensemble now incorporates classic oscillators (RSI, Stochastic), range analysis,
    ## and a volume profile proxy to better predict behavior in sideways markets.
    ## The meta-feature generation for training is now fully implemented.
    """

    def __init__(self, config: dict, ensemble_name: str = "SidewaysRangeEnsemble"):
        super().__init__(config, ensemble_name)
        self.models = {
            "clustering": None,
            "order_flow": None,
        }
        self.meta_learner = None
        self.trained = False
        self.model_config = {
            "bb_window": 20,
            "bb_std_dev": 2,
            "kmeans_clusters": 4,
            "rsi_period": 14,
            "stoch_period": 14,
            "range_window": 50,
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

        self.logger.info("Training Order Flow (LGBM) model...")
        try:
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

        # For live prediction, we need a window of data to calculate rolling features
        history_window = kwargs.get("klines_df", current_features)
        
        meta_features = self._get_meta_features(history_window, is_live=True)
        meta_input_data = pd.DataFrame([meta_features], columns=self.meta_learner_features).fillna(0)
        return self._get_meta_prediction(meta_input_data)

    def _get_meta_features(self, df, is_live=False):
        """
        ## CHANGE: Fully implemented meta-feature generation for both training and live prediction.
        ## This function now calculates all base model outputs and new technical indicators.
        """
        if is_live:
            # For live prediction, we operate on the last row of the provided dataframe
            current_row_df = df.tail(1)
        else:
            # For training, we operate on the entire dataframe
            current_row_df = df

        meta_features_df = pd.DataFrame(index=current_row_df.index)

        # --- Base Model Features ---
        if self.models["clustering"]:
            cluster_data = current_row_df[['close', 'volume', 'ATR']].copy().fillna(0)
            meta_features_df['price_cluster'] = self.models["clustering"].predict(cluster_data)

        if self.models["order_flow"]:
            of_features = ['volume', 'volume_delta', 'cvd_slope']
            for col in of_features:
                if col not in current_row_df.columns:
                    current_row_df[col] = 0
            X_of = current_row_df[of_features].fillna(0)
            # Predict probability of the first class
            meta_features_df['order_flow_pred'] = self.models["order_flow"].predict_proba(X_of)[:, 0]

        # --- New Technical Indicator Features ---
        # Bollinger Bands
        bb_window = self.model_config["bb_window"]
        rolling_close = df['close'].rolling(window=bb_window)
        ma = rolling_close.mean()
        std = rolling_close.std()
        meta_features_df['bb_bandwidth'] = ((ma + (std * 2)) - (ma - (std * 2))) / ma
        meta_features_df['is_bb_squeeze'] = (meta_features_df['bb_bandwidth'] < meta_features_df['bb_bandwidth'].rolling(100).min()).astype(int)

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.model_config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.model_config['rsi_period']).mean()
        rs = gain / loss
        meta_features_df['rsi'] = 100 - (100 / (1 + rs))

        # Stochastic Oscillator
        low_min = df['low'].rolling(window=self.model_config['stoch_period']).min()
        high_max = df['high'].rolling(window=self.model_config['stoch_period']).max()
        meta_features_df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
        meta_features_df['stoch_d'] = meta_features_df['stoch_k'].rolling(window=3).mean()

        # Position in Range
        range_low = df['low'].rolling(window=self.model_config['range_window']).min()
        range_high = df['high'].rolling(window=self.model_config['range_window']).max()
        meta_features_df['position_in_range'] = (df['close'] - range_low) / (range_high - range_low)

        # Distance to Point of Control (Volume Profile Proxy)
        poc = df['close'].iloc[df['volume'].rolling(self.model_config['range_window']).apply(lambda x: x.idxmax(), raw=True).astype(int)]
        poc.index = df.index
        meta_features_df['distance_to_poc'] = (df['close'] - poc) / df['close']

        if is_live:
            # Return a dictionary for the single live row
            return meta_features_df.iloc[-1].to_dict()
        else:
            # Return the full DataFrame for training
            return meta_features_df.dropna()

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
