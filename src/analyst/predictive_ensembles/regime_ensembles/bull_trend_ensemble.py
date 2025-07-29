import logging
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from arch import arch_model
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Input,
    LayerNormalization,
    MultiHeadSelfAttention,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

from .base_ensemble import BaseEnsemble

warnings.filterwarnings("ignore", category=UserWarning, module="arch")
warnings.filterwarnings("ignore", category=FutureWarning, module="arch")


class BullTrendEnsemble(BaseEnsemble):
    """
    Predictive Ensemble specifically designed for BULL_TREND market regimes.
    Combines LSTM, Transformer, GARCH, and LightGBM models into a meta-learner.
    """

    def __init__(self, config: dict, ensemble_name: str = "BullTrendEnsemble"):
        super().__init__(config, ensemble_name)
        self.models = {"lstm": None, "transformer": None, "garch": None, "lgbm": None}
        self.meta_learner = None
        self.trained = False
        self.dl_config = {
            "sequence_length": 20,
            "lstm_units": 50,
            "transformer_heads": 2,
            "transformer_key_dim": 32,
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "epochs": 50,
            "batch_size": 32,
            "l1_reg_strength": self.config.get("meta_learner_l1_reg", 0.001),
            "l2_reg_strength": self.config.get("meta_learner_l2_reg", 0.001),
        }
        self.expected_dl_features = [
            "close", "volume", "ADX", "MACD_HIST", "ATR", "volume_delta",
            "autoencoder_reconstruction_error", "Is_SR_Interacting",
        ]
        self.meta_learner_features = []

    def _prepare_sequence_data(self, df: pd.DataFrame, target_series: pd.Series = None):
        features_df = df[self.expected_dl_features].copy().fillna(0)
        sequence_length = self.dl_config["sequence_length"]
        X, y = [], []
        for i in range(len(features_df) - sequence_length + 1):
            X.append(features_df.iloc[i : (i + sequence_length)].values)
            if target_series is not None:
                y.append(target_series.iloc[i + sequence_length - 1])
        if not X:
            return np.array([]), np.array([])
        return np.array(X), np.array(y) if y else None

    def train_ensemble(self, historical_features: pd.DataFrame, historical_targets: pd.Series):
        self.logger.info(f"Training {self.ensemble_name} ensemble models...")
        if historical_features.empty:
            return

        for col in self.expected_dl_features:
            if col not in historical_features.columns:
                historical_features[col] = 0.0
        
        aligned_data = historical_features.join(historical_targets.rename("target")).dropna()
        if aligned_data.empty:
            return

        X_seq, y_seq = self._prepare_sequence_data(aligned_data, aligned_data["target"])
        X_flat = aligned_data[self.expected_dl_features].copy().fillna(0)
        
        if X_seq.size == 0:
            return

        self._train_base_models(X_seq, y_seq, X_flat, aligned_data["target"], aligned_data)
        
        meta_features_train = self._get_meta_features(X_seq, X_flat, aligned_data)
        aligned_meta = meta_features_train.join(aligned_data["target"]).dropna()
        
        if aligned_meta.empty:
            return

        y_meta_train = aligned_meta["target"]
        X_meta_train = aligned_meta.drop(columns=["target"])

        self.meta_learner_features = X_meta_train.columns.tolist()
        self._train_meta_learner(X_meta_train, y_meta_train)
        self.trained = True

    def _train_base_models(self, X_seq, y_seq, X_flat, y_flat, aligned_data):
        unique_targets, y_seq_encoded = np.unique(y_seq, return_inverse=True)
        num_classes = len(unique_targets)

        # Train DL models
        self.models["lstm"] = self._train_dl_model(X_seq, y_seq_encoded, num_classes, is_transformer=False)
        self.models["transformer"] = self._train_dl_model(X_seq, y_seq_encoded, num_classes, is_transformer=True)

        # Train GARCH
        returns = aligned_data['close'].pct_change().dropna()
        if len(returns) > 100:
            try:
                garch = arch_model(returns, vol='Garch', p=1, q=1)
                self.models["garch"] = garch.fit(disp='off')
            except Exception as e:
                self.logger.error(f"GARCH training failed: {e}")

        # Train LightGBM
        try:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            self.models["lgbm"] = [
                LGBMClassifier(random_state=42, verbose=-1).fit(X_flat.iloc[train_idx], y_flat.iloc[train_idx])
                for train_idx, _ in kf.split(X_flat, y_flat)
            ]
        except Exception as e:
            self.logger.error(f"LGBM training failed: {e}")

    def _train_dl_model(self, X_seq, y_seq_encoded, num_classes, is_transformer=False):
        try:
            input_layer = Input(shape=(X_seq.shape[1], X_seq.shape[2]))
            if is_transformer:
                x = MultiHeadSelfAttention(num_heads=self.dl_config["transformer_heads"], key_dim=self.dl_config["transformer_key_dim"])(input_layer, input_layer)
                x = LayerNormalization(epsilon=1e-6)(input_layer + x)
                x = Dense(X_seq.shape[2], activation="relu")(x)
                x = LayerNormalization(epsilon=1e-6)(x + x)
                x = tf.keras.layers.Flatten()(x)
            else: # LSTM
                x = LSTM(self.dl_config["lstm_units"], return_sequences=True)(input_layer)
                x = Dropout(self.dl_config["dropout_rate"])(x)
                x = LSTM(self.dl_config["lstm_units"] // 2)(x)
                x = Dropout(self.dl_config["dropout_rate"])(x)
            
            output_layer = Dense(num_classes, activation='softmax')(x)
            model = Model(inputs=input_layer, outputs=output_layer)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            model.fit(X_seq, y_seq_encoded, epochs=self.dl_config["epochs"], batch_size=self.dl_config["batch_size"], verbose=0)
            return model
        except Exception as e:
            self.logger.error(f"DL Model training failed: {e}")
            return None

    def get_prediction(self, current_features: pd.DataFrame, **kwargs) -> dict:
        if not self.trained:
            return {"prediction": "HOLD", "confidence": 0.0}
        
        X_seq, _ = self._prepare_sequence_data(current_features)
        if X_seq.size == 0: return {"prediction": "HOLD", "confidence": 0.0}
        
        X_flat = current_features[self.expected_dl_features].tail(1).fillna(0)
        meta_features = self._get_meta_features(X_seq, X_flat, current_features, is_live=True, **kwargs)
        meta_input = pd.DataFrame([meta_features], columns=self.meta_learner_features).fillna(0)
        
        return self._get_meta_prediction(meta_input)

    def _get_meta_features(self, X_seq, X_flat, full_df, is_live=False, **kwargs):
        """
        ## CHANGE: Fully implemented meta-feature generation for the training phase.
        This function now correctly computes and aligns features from all base models
        and advanced indicators across the entire historical dataset.
        """
        if is_live:
            # --- Live Prediction Logic ---
            meta_features = {}
            if self.models["lstm"]: meta_features['lstm_conf'] = np.max(self.models["lstm"].predict(X_seq, verbose=0))
            if self.models["transformer"]: meta_features['transformer_conf'] = np.max(self.models["transformer"].predict(X_seq, verbose=0))
            if self.models["lgbm"]: meta_features['lgbm_proba'] = np.max(np.mean([m.predict_proba(X_flat) for m in self.models["lgbm"]], axis=0))
            if self.models["garch"]: meta_features['garch_volatility'] = self.models["garch"].forecast(horizon=1).variance.iloc[-1,0]
            
            meta_features.update(self._get_wyckoff_features(full_df, is_live))
            meta_features.update(self._get_multi_timeframe_features(full_df, is_live))
            return meta_features
        else:
            # --- Training Logic ---
            meta_index = full_df.index[self.dl_config["sequence_length"] - 1:]
            meta_features_df = pd.DataFrame(index=meta_index)
            
            # DL Models
            if self.models["lstm"]: meta_features_df['lstm_conf'] = np.max(self.models["lstm"].predict(X_seq, verbose=0), axis=1)
            if self.models["transformer"]: meta_features_df['transformer_conf'] = np.max(self.models["transformer"].predict(X_seq, verbose=0), axis=1)
            
            # Align flat features with sequence model outputs
            X_flat_aligned = X_flat.loc[meta_index]
            if self.models["lgbm"]:
                probas = np.mean([m.predict_proba(X_flat_aligned) for m in self.models["lgbm"]], axis=0)
                meta_features_df['lgbm_proba'] = np.max(probas, axis=1)

            if self.models["garch"]:
                meta_features_df['garch_volatility'] = self.models["garch"].conditional_volatility.reindex(meta_index, method='ffill')

            # Advanced Features
            meta_features_df = meta_features_df.join(self._get_wyckoff_features(full_df, is_live))
            meta_features_df = meta_features_df.join(self._get_multi_timeframe_features(full_df, is_live))

            return meta_features_df.fillna(0)

    def _train_meta_learner(self, X, y):
        self.meta_learner = LGBMClassifier(random_state=42, verbose=-1)
        self.meta_learner.fit(X, y)
        self.logger.info(f"{self.ensemble_name} meta-learner trained.")

    def _get_meta_prediction(self, meta_input_df):
        if not self.meta_learner: return {"prediction": "HOLD", "confidence": 0.0}
        proba = self.meta_learner.predict_proba(meta_input_df)[0]
        idx = np.argmax(proba)
        return {"prediction": self.meta_learner.classes_[idx], "confidence": proba[idx]}

    # --- Feature Helpers ---
    def _get_wyckoff_features(self, df, is_live):
        vol_ma = df["volume"].rolling(20).mean()
        high_vol = (df["volume"] > vol_ma * 1.5).astype(int)
        return pd.DataFrame({'wyckoff_high_volume': high_vol}) if not is_live else {'wyckoff_high_volume': high_vol.iloc[-1]}

    def _get_multi_timeframe_features(self, df, is_live):
        ma_short = df["close"].rolling(10).mean()
        ma_long = df["close"].rolling(50).mean()
        aligned = (ma_short > ma_long).astype(int)
        return pd.DataFrame({'ma_aligned': aligned}) if not is_live else {'ma_aligned': aligned.iloc[-1]}
