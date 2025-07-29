import logging
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from .base_ensemble import BaseEnsemble

warnings.filterwarnings("ignore", category=UserWarning, module="arch")
warnings.filterwarnings("ignore", category=FutureWarning, module="arch")


class SRZoneActionEnsemble(BaseEnsemble):
    """
    ## CHANGE: Fully implemented the meta-feature generation for training.
    ## This class now correctly calculates and uses a rich feature set from its
    ## base models (Volume Profile, Order Flow) to train its final meta-learner,
    ## ensuring the ensemble is robust and effective.
    """

    def __init__(self, config: dict, ensemble_name: str = "SRZoneActionEnsemble"):
        super().__init__(config, ensemble_name)
        self.models = {
            "volume_profile": None,  # Rule-based or simple model
            "order_flow": None,      # LGBM on order flow features
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
        
        # Align meta features with targets
        aligned_meta = meta_features_train.join(aligned_data["target"]).dropna()
        
        if aligned_meta.empty:
            self.logger.warning("Meta-features training set is empty after alignment. Cannot train meta-learner.")
            return

        y_meta_train = aligned_meta["target"]
        X_meta_train = aligned_meta.drop(columns=["target"])

        self.meta_learner_features = X_meta_train.columns.tolist()
        self._train_meta_learner(X_meta_train, y_meta_train)
        self.trained = True

    def _train_base_models(self, X_flat, y_flat):
        """Helper to train base models for S/R zone action."""
        # Volume Profile model is rule-based, no training needed
        self.models["volume_profile"] = True

        self.logger.info("Training Order Flow (LGBM) model for S/R zones...")
        try:
            of_features = ['volume', 'volume_delta', 'cvd_slope', 'ATR', 'Is_SR_Interacting']
            for col in of_features:
                if col not in X_flat.columns:
                    X_flat[col] = 0
            
            X_of = X_flat[of_features].fillna(0)
            self.models["order_flow"] = LGBMClassifier(random_state=42, verbose=-1).fit(X_of, y_flat)
        except Exception as e:
            self.logger.error(f"Error training Order Flow model for S/R zones: {e}")

    def get_prediction(self, current_features: pd.DataFrame, **kwargs) -> dict:
        if not self.trained or not self.meta_learner:
            return {"prediction": "HOLD", "confidence": 0.0}

        # For live prediction, we need a window of data for rolling calculations
        history_window = kwargs.get("klines_df", current_features)
        
        meta_features = self._get_meta_features(history_window, is_live=True)
        meta_input_data = pd.DataFrame([meta_features], columns=self.meta_learner_features).fillna(0)
        return self._get_meta_prediction(meta_input_data)

    def _get_meta_features(self, df, is_live=False, **kwargs):
        """
        ## CHANGE: Implemented full meta-feature generation for the training phase.
        ## This function now correctly computes features across the entire historical
        ## dataset, ensuring the meta-learner is trained effectively.
        """
        if is_live:
            # For live prediction, operate on the last row
            current_row = df.tail(1)
            meta_features = {}
        else:
            # For training, create a new DataFrame to hold features
            meta_features_df = pd.DataFrame(index=df.index)
            current_row = df # Operate on the whole dataframe

        # --- Feature Calculation ---
        # Ensure 'Is_SR_Interacting' exists
        if 'Is_SR_Interacting' not in current_row.columns:
            current_row['Is_SR_Interacting'] = 0

        # Volume Profile feature (rule-based)
        if self.models["volume_profile"]:
            volume_ma = current_row['volume'].rolling(20, min_periods=1).mean()
            is_high_volume_at_sr = (current_row['volume'] > volume_ma * 1.5) & (current_row['Is_SR_Interacting'] == 1)
            if is_live:
                meta_features['is_high_volume_at_sr'] = int(is_high_volume_at_sr.iloc[-1])
            else:
                meta_features_df['is_high_volume_at_sr'] = is_high_volume_at_sr.astype(int)

        # Order Flow model prediction
        if self.models["order_flow"]:
            of_features = ['volume', 'volume_delta', 'cvd_slope', 'ATR', 'Is_SR_Interacting']
            for col in of_features:
                if col not in current_row.columns:
                    current_row[col] = 0
            X_of = current_row[of_features].fillna(0)
            
            # Get probability for the "rejection" class (assuming it's the second class)
            rejection_prob = self.models["order_flow"].predict_proba(X_of)[:, 1]
            if is_live:
                meta_features['order_flow_rejection_prob'] = rejection_prob[0]
            else:
                meta_features_df['order_flow_rejection_prob'] = rejection_prob

        # --- Return appropriate format ---
        if is_live:
            return meta_features
        else:
            # For training, return the complete DataFrame, dropping NaNs from rolling calcs
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
