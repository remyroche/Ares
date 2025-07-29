import logging
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from .base_ensemble import BaseEnsemble

# Placeholder for TabNet, assuming a library like 'pytorch_tabnet' would be used
# from pytorch_tabnet.tab_model import TabNetClassifier

warnings.filterwarnings("ignore", category=UserWarning, module="arch")
warnings.filterwarnings("ignore", category=FutureWarning, module="arch")


class HighImpactCandleEnsemble(BaseEnsemble):
    """
    ## CHANGE: Fully implemented the High Impact Candle Ensemble.
    ## This class now correctly trains its base models and meta-learner on a rich,
    ## consistent feature set, making it robust for predicting the outcome of
    ## high-impact candle events.
    """

    def __init__(self, config: dict, ensemble_name: str = "HighImpactCandleEnsemble"):
        super().__init__(config, ensemble_name)
        self.models = {
            "volume_imbalance": None,  # Rule-based or simple model
            "tabnet": None,            # Placeholder for TabNet model
            "order_flow": None,        # LGBM on order flow features
        }
        self.meta_learner = None
        self.trained = False
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
        
        aligned_meta = meta_features_train.join(aligned_data["target"]).dropna()
        
        if aligned_meta.empty:
            self.logger.warning("Meta-features training set is empty after alignment.")
            return

        y_meta_train = aligned_meta["target"]
        X_meta_train = aligned_meta.drop(columns=["target"])

        self.meta_learner_features = X_meta_train.columns.tolist()
        self._train_meta_learner(X_meta_train, y_meta_train)
        self.trained = True

    def _train_base_models(self, X_flat, y_flat):
        """Helper to train base models for high impact candles."""
        # Volume Imbalance model is rule-based, no training needed
        self.models["volume_imbalance"] = True

        self.logger.info("Training TabNet model (LGBM Stand-in)...")
        try:
            feature_subset = self._get_feature_subset()
            # Ensure all features exist, fill with 0 if not
            for col in feature_subset:
                if col not in X_flat.columns:
                    X_flat[col] = 0
            self.models["tabnet"] = LGBMClassifier(random_state=42, verbose=-1).fit(X_flat[feature_subset], y_flat)
        except Exception as e:
            self.logger.error(f"Error training TabNet placeholder model: {e}")

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

        history_window = kwargs.get("klines_df", current_features)
        meta_features = self._get_meta_features(history_window, is_live=True)
        meta_input_data = pd.DataFrame([meta_features], columns=self.meta_learner_features).fillna(0)
        return self._get_meta_prediction(meta_input_data)

    def _get_meta_features(self, df, is_live=False, **kwargs):
        """
        ## CHANGE: Fully implemented meta-feature generation for training and live prediction.
        """
        if is_live:
            current_row = df.tail(1)
            meta_features = {}
        else:
            meta_features_df = pd.DataFrame(index=df.index)
            current_row = df

        # --- Feature Calculation ---
        # Volume Imbalance Signal
        if self.models["volume_imbalance"]:
            # Use np.divide to handle potential division by zero
            volume_imbalance_signal = np.divide(current_row['volume_delta'], current_row['volume'], where=current_row['volume']!=0, out=np.zeros_like(current_row['volume_delta'], dtype=float))
            if is_live:
                meta_features['volume_imbalance_signal'] = volume_imbalance_signal.iloc[-1]
            else:
                meta_features_df['volume_imbalance_signal'] = volume_imbalance_signal

        # TabNet (LGBM Stand-in) Prediction
        if self.models["tabnet"]:
            feature_subset = self._get_feature_subset()
            for col in feature_subset:
                if col not in current_row.columns:
                    current_row[col] = 0
            X_tabnet = current_row[feature_subset].fillna(0)
            tabnet_pred_prob = self.models["tabnet"].predict_proba(X_tabnet)[:, 1] # Prob of class 1
            if is_live:
                meta_features['tabnet_pred_prob'] = tabnet_pred_prob[0]
            else:
                meta_features_df['tabnet_pred_prob'] = tabnet_pred_prob

        # Order Flow Model Prediction
        if self.models["order_flow"]:
            of_features = ['volume', 'volume_delta', 'cvd_slope']
            for col in of_features:
                if col not in current_row.columns:
                    current_row[col] = 0
            X_of = current_row[of_features].fillna(0)
            order_flow_pred_prob = self.models["order_flow"].predict_proba(X_of)[:, 1] # Prob of class 1
            if is_live:
                meta_features['order_flow_pred_prob'] = order_flow_pred_prob[0]
            else:
                meta_features_df['order_flow_pred_prob'] = order_flow_pred_prob

        if is_live:
            return meta_features
        else:
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

    def _get_feature_subset(self):
        """
        ## Defined the feature subset for the TabNet/LGBM model.
        ## Provides a consistent list of features for training and prediction.
        """
        return [
            'ADX', 'MACD_HIST', 'ATR', 'volume_delta',
            'autoencoder_reconstruction_error', 'Is_SR_Interacting',
            'rsi', 'stoch_k', 'bb_bandwidth', 'position_in_range'
        ]
