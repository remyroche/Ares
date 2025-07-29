import numpy as np
import pandas as pd
import tensorflow as tf
from arch import arch_model
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, MultiHeadSelfAttention, LayerNormalization
from tensorflow.keras.models import Model

from .base_ensemble import BaseEnsemble

class BullTrendEnsemble(BaseEnsemble):
    """
    ## CHANGE: Simplified to use the unified feature set from BaseEnsemble.
    ## This class now focuses solely on defining the base models relevant for a bull trend.
    """
    def __init__(self, config: dict, ensemble_name: str = "BullTrendEnsemble"):
        super().__init__(config, ensemble_name)
        self.dl_config = {"sequence_length": 20, "lstm_units": 50, "transformer_heads": 2, "transformer_key_dim": 32, "dropout_rate": 0.2, "epochs": 50, "batch_size": 32}

    def _train_base_models(self, aligned_data: pd.DataFrame, y_encoded: np.ndarray):
        X_seq, y_seq_aligned_encoded = self._prepare_sequence_data(aligned_data, pd.Series(y_encoded, index=aligned_data.index))
        X_flat = aligned_data[self.flat_features].fillna(0)
        num_classes = len(np.unique(y_encoded))

        self.models["lstm"] = self._train_dl_model(X_seq, y_seq_aligned_encoded, num_classes, is_transformer=False)
        self.models["transformer"] = self._train_dl_model(X_seq, y_seq_aligned_encoded, num_classes, is_transformer=True)
        self.models["tabnet"] = self._train_tabnet_model(X_flat, y_encoded)
        
        returns = aligned_data['close'].pct_change().dropna()
        if len(returns) > 100:
            try: self.models["garch"] = arch_model(returns, vol='Garch', p=1, q=1).fit(disp='off')
            except Exception as e: self.logger.error(f"GARCH training failed: {e}")
        try:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            self.models["lgbm"] = [LGBMClassifier(random_state=42, verbose=-1).fit(X_flat.iloc[train_idx], y_encoded[train_idx]) for train_idx, _ in kf.split(X_flat)]
        except Exception as e: self.logger.error(f"LGBM training failed: {e}")

    def _get_meta_features(self, df: pd.DataFrame, is_live: bool = False, **kwargs) -> pd.DataFrame or dict:
        base_preds = self._get_base_model_predictions(df, is_live)
        raw_features = df[self.flat_features + ['oi_value', 'funding_rate_ma']].tail(1) if is_live else df[self.flat_features + ['oi_value', 'funding_rate_ma']]
        
        if is_live:
            base_preds.update(raw_features.iloc[0].to_dict())
            return base_preds
        else:
            return base_preds.join(raw_features).fillna(0)

    def _get_base_model_predictions(self, df: pd.DataFrame, is_live: bool):
        seq_len = self.dl_config["sequence_length"]
        if is_live:
            meta = {}
            X_seq, _ = self._prepare_sequence_data(df)
            X_flat = df[self.flat_features].tail(1).fillna(0)
            if self.models.get("lstm"): meta['lstm_conf'] = np.max(self.models["lstm"].predict(X_seq, verbose=0))
            if self.models.get("transformer"): meta['transformer_conf'] = np.max(self.models["transformer"].predict(X_seq, verbose=0))
            if self.models.get("tabnet"): meta['tabnet_proba'] = np.max(self.models["tabnet"].predict_proba(X_flat.values))
            if self.models.get("lgbm"): meta['lgbm_proba'] = np.max(np.mean([m.predict_proba(X_flat) for m in self.models["lgbm"]], axis=0))
            if self.models.get("garch"): meta['garch_volatility'] = self.models["garch"].forecast(horizon=1).variance.iloc[-1,0]
            return meta
        else:
            meta_df = pd.DataFrame(index=df.index[seq_len - 1:])
            X_seq, _ = self._prepare_sequence_data(df)
            X_flat_aligned = df[self.flat_features].loc[meta_df.index].fillna(0)
            if self.models.get("lstm"): meta_df['lstm_conf'] = np.max(self.models["lstm"].predict(X_seq, verbose=0), axis=1)
            if self.models.get("transformer"): meta_df['transformer_conf'] = np.max(self.models["transformer"].predict(X_seq, verbose=0), axis=1)
            if self.models.get("tabnet"): meta_df['tabnet_proba'] = np.max(self.models["tabnet"].predict_proba(X_flat_aligned.values), axis=1)
            if self.models.get("lgbm"): meta_df['lgbm_proba'] = np.max(np.mean([m.predict_proba(X_flat_aligned) for m in self.models["lgbm"]], axis=0), axis=1)
            if self.models.get("garch"): meta_df['garch_volatility'] = self.models["garch"].conditional_volatility.reindex(meta_df.index, method='ffill')
            return meta_df

    def _prepare_sequence_data(self, df: pd.DataFrame, target_series: pd.Series = None):
        features_df = df[self.sequence_features].copy().fillna(0)
        seq_len = self.dl_config["sequence_length"]
        X, y = [], []
        for i in range(len(features_df) - seq_len + 1):
            X.append(features_df.iloc[i:i+seq_len].values)
            if target_series is not None: y.append(target_series.iloc[i + seq_len - 1])
        return np.array(X), np.array(y) if target_series is not None else None

    def _train_dl_model(self, X_seq, y_seq_encoded, num_classes, is_transformer=False):
        try:
            inputs = Input(shape=(X_seq.shape[1], X_seq.shape[2]))
            x = MultiHeadSelfAttention(num_heads=self.dl_config["transformer_heads"], key_dim=self.dl_config["transformer_key_dim"])(inputs, inputs) if is_transformer else LSTM(self.dl_config["lstm_units"])(inputs)
            if is_transformer: x = LayerNormalization(epsilon=1e-6)(inputs + x); x = tf.keras.layers.Flatten()(x)
            x = Dropout(self.dl_config["dropout_rate"])(x)
            outputs = Dense(num_classes, activation='softmax')(x)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            model.fit(X_seq, y_seq_encoded, epochs=self.dl_config["epochs"], batch_size=self.dl_config["batch_size"], verbose=0)
            return model
        except Exception as e: self.logger.error(f"DL Model training failed: {e}"); return None

    def _train_tabnet_model(self, X_flat, y_flat_encoded):
        try:
            model = TabNetClassifier(); model.fit(X_flat.values, y_flat_encoded, max_epochs=50, patience=20, batch_size=1024)
            return model
        except Exception as e: self.logger.error(f"TabNet training failed: {e}"); return None
