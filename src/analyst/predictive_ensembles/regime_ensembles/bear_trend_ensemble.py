import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from arch import arch_model
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, MultiHeadSelfAttention, LayerNormalization
from tensorflow.keras.models import Model

from .base_ensemble import BaseEnsemble

class BearTrendEnsemble(BaseEnsemble):
    def __init__(self, config: dict, ensemble_name: str = "BearTrendEnsemble"):
        super().__init__(config, ensemble_name)
        self.dl_config = {
            "sequence_length": 20, "lstm_units": 50, "transformer_heads": 2,
            "transformer_key_dim": 32, "dropout_rate": 0.2, "epochs": 50, "batch_size": 32,
        }
        
        # CHANGE: Updated feature sets to include new alternative data and autoencoder error.
        self.expected_dl_features = [
            "close", "volume", "ADX", "MACD_HIST", "ATR", "volume_delta",
            "autoencoder_reconstruction_error", "oi_roc", "funding_rate"
        ]
        self.flat_feature_subset = [
            "ADX", "MACD_HIST", "ATR", "volume_delta", "rsi", "stoch_k",
            "autoencoder_reconstruction_error", "oi_roc", "funding_rate", "total_liquidations"
        ]

    def _train_base_models(self, aligned_data: pd.DataFrame, y_encoded: np.ndarray):
        # (This method's logic is identical to the Bull Trend ensemble)
        X_seq, y_seq_aligned_encoded = self._prepare_sequence_data(aligned_data, pd.Series(y_encoded, index=aligned_data.index))
        X_flat = aligned_data[self.flat_feature_subset].fillna(0)
        num_classes = len(np.unique(y_encoded))

        self.models["lstm"] = self._train_dl_model(X_seq, y_seq_aligned_encoded, num_classes, is_transformer=False)
        self.models["transformer"] = self._train_dl_model(X_seq, y_seq_aligned_encoded, num_classes, is_transformer=True)
        self.models["tabnet"] = self._train_tabnet_model(X_flat, y_encoded)
        
        returns = aligned_data['close'].pct_change().dropna()
        if len(returns) > 100:
            try:
                self.models["garch"] = arch_model(returns, vol='Garch', p=1, q=1).fit(disp='off')
            except Exception as e: self.logger.error(f"GARCH training failed: {e}")
        try:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            self.models["lgbm"] = [LGBMClassifier(random_state=42, verbose=-1).fit(X_flat.iloc[train_idx], y_encoded[train_idx]) for train_idx, _ in kf.split(X_flat)]
        except Exception as e: self.logger.error(f"LGBM training failed: {e}")


    def _get_meta_features(self, df: pd.DataFrame, is_live: bool = False, **kwargs) -> pd.DataFrame or dict:
        # (This method's logic is identical to the Bull Trend ensemble)
        base_model_features = self._get_base_model_predictions(df, is_live)
        raw_features_to_include = [
            "autoencoder_reconstruction_error", "oi_roc", "funding_rate", 
            "total_liquidations", "liquidation_ratio"
        ]
        if is_live:
            for col in raw_features_to_include:
                base_model_features[col] = df[col].iloc[-1] if col in df.columns else 0.0
            return base_model_features
        else:
            meta_df = base_model_features
            for col in raw_features_to_include:
                if col in df.columns:
                    meta_df = meta_df.join(df[[col]])
            return meta_df.fillna(0)

    def _get_base_model_predictions(self, df: pd.DataFrame, is_live: bool):
        if is_live:
            meta_features = {}
            X_seq, _ = self._prepare_sequence_data(df)
            X_flat = df[self.flat_feature_subset].tail(1).fillna(0)
            if self.models["lstm"]: meta_features['lstm_conf'] = np.max(self.models["lstm"].predict(X_seq, verbose=0))
            if self.models["transformer"]: meta_features['transformer_conf'] = np.max(self.models["transformer"].predict(X_seq, verbose=0))
            if self.models["tabnet"]: meta_features['tabnet_proba'] = np.max(self.models["tabnet"].predict_proba(X_flat.values))
            if self.models["lgbm"]: meta_features['lgbm_proba'] = np.max(np.mean([m.predict_proba(X_flat) for m in self.models["lgbm"]], axis=0))
            if self.models["garch"]: meta_features['garch_volatility'] = self.models["garch"].forecast(horizon=1).variance.iloc[-1,0]
            return meta_features
        else:
            meta_index = df.index[self.dl_config["sequence_length"] - 1:]
            meta_df = pd.DataFrame(index=meta_index)
            X_seq, _ = self._prepare_sequence_data(df)
            X_flat_aligned = df[self.flat_feature_subset].loc[meta_index].fillna(0)
            
            if self.models["lstm"]: meta_df['lstm_conf'] = np.max(self.models["lstm"].predict(X_seq, verbose=0), axis=1)
            if self.models["transformer"]: meta_df['transformer_conf'] = np.max(self.models["transformer"].predict(X_seq, verbose=0), axis=1)
            if self.models["tabnet"]: meta_df['tabnet_proba'] = np.max(self.models["tabnet"].predict_proba(X_flat_aligned.values), axis=1)
            if self.models["lgbm"]: meta_df['lgbm_proba'] = np.max(np.mean([m.predict_proba(X_flat_aligned) for m in self.models["lgbm"]], axis=0), axis=1)
            if self.models["garch"]: meta_df['garch_volatility'] = self.models["garch"].conditional_volatility.reindex(meta_index, method='ffill')
            return meta_df
            
    def _prepare_sequence_data(self, df: pd.DataFrame, target_series: pd.Series = None):
        for col in self.expected_dl_features:
            if col not in df.columns:
                df[col] = 0.0
        features_df = df[self.expected_dl_features].copy().fillna(0)
        seq_len = self.dl_config["sequence_length"]
        X, y = [], []
        for i in range(len(features_df) - seq_len + 1):
            X.append(features_df.iloc[i:i+seq_len].values)
            if target_series is not None:
                y.append(target_series.iloc[i + seq_len - 1])
        return np.array(X), np.array(y) if target_series is not None else None

    def _train_dl_model(self, X_seq, y_seq_encoded, num_classes, is_transformer=False):
        try:
            inputs = Input(shape=(X_seq.shape[1], X_seq.shape[2]))
            if is_transformer:
                x = MultiHeadSelfAttention(num_heads=self.dl_config["transformer_heads"], key_dim=self.dl_config["transformer_key_dim"])(inputs, inputs)
                x = LayerNormalization(epsilon=1e-6)(inputs + x)
                x = tf.keras.layers.Flatten()(x)
            else:
                x = LSTM(self.dl_config["lstm_units"])(inputs)
            x = Dropout(self.dl_config["dropout_rate"])(x)
            outputs = Dense(num_classes, activation='softmax')(x)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            model.fit(X_seq, y_seq_encoded, epochs=self.dl_config["epochs"], batch_size=self.dl_config["batch_size"], verbose=0)
            return model
        except Exception as e:
            self.logger.error(f"DL Model training failed: {e}")
            return None

    def _train_tabnet_model(self, X_flat, y_flat_encoded):
        try:
            model = TabNetClassifier()
            model.fit(X_flat.values, y_flat_encoded, max_epochs=50, patience=20, batch_size=1024, virtual_batch_size=128)
            return model
        except Exception as e:
            self.logger.error(f"TabNet training failed: {e}")
            return None
