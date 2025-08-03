from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model
from keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Flatten,
    Input,
    LayerNormalization,
    MultiHeadAttention,
)
from keras.models import Model
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

from .base_ensemble import BaseEnsemble


class BearTrendEnsemble(BaseEnsemble):
    """
    This version is a mirror of the Bull Trend ensemble, ensuring symmetrical logic
    and using a diverse set of hyperparameter-tuned base models.
    """

    def __init__(self, config: dict, ensemble_name: str = "BearTrendEnsemble"):
        super().__init__(config, ensemble_name)
        self.dl_config = {
            "sequence_length": 20,
            "lstm_units": 50,
            "transformer_heads": 2,
            "transformer_key_dim": 32,
            "dropout_rate": 0.2,
            "epochs": 50,
            "batch_size": 32,
        }
        self.models = {
            "lstm": None,
            "transformer": None,
            "garch": None,
            "tabnet": None,
            "order_flow_lgbm": None,
            "logistic_regression": None,
        }

    def _train_base_models(self, aligned_data: pd.DataFrame, y_encoded: np.ndarray):
        # (This method's logic is identical to the Bull Trend ensemble)
        X_seq, y_seq_aligned_encoded = self._prepare_sequence_data(
            aligned_data,
            pd.Series(y_encoded, index=aligned_data.index),
        )
        X_flat = aligned_data[self.flat_features].fillna(0)
        num_classes = len(np.unique(y_encoded))

        if X_seq.size > 0:
            self.models["lstm"] = self._train_dl_model(
                X_seq,
                y_seq_aligned_encoded,
                num_classes,
                is_transformer=False,
            )
            self.models["transformer"] = self._train_dl_model(
                X_seq,
                y_seq_aligned_encoded,
                num_classes,
                is_transformer=True,
            )

        self.models["tabnet"] = self._train_tabnet_model(X_flat, y_encoded)

        self.logger.info("Tuning and training specialized Order Flow LGBM...")
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

        # Logistic Regression with L1-L2 regularization
        self.logger.info(
            "Training Logistic Regression model with L1-L2 regularization...",
        )
        self.models["logistic_regression"] = self._train_with_smote(
            self._get_regularized_logistic_regression(),
            X_flat,
            y_encoded,
        )

        returns = aligned_data["close"].pct_change().dropna()
        if len(returns) > 100:
            try:
                self.models["garch"] = arch_model(returns, vol="Garch", p=1, q=1).fit(
                    disp="off",
                )
            except Exception as e:
                self.logger.error(f"GARCH training failed: {e}")

    def _get_meta_features(
        self,
        df: pd.DataFrame,
        is_live: bool = False,
        **kwargs,
    ) -> pd.DataFrame | dict:
        # (This method's logic is identical to the Bull Trend ensemble)
        base_preds = self._get_base_model_predictions(df, is_live)
        raw_features_to_include = self.flat_features + ["oi_value", "funding_rate_ma"]

        if is_live:
            current_row = df.tail(1)
            for col in raw_features_to_include:
                base_preds[col] = (
                    current_row[col].iloc[0] if col in current_row.columns else 0.0
                )
            return base_preds
        meta_df = base_preds
        for col in raw_features_to_include:
            if col in df.columns:
                meta_df = meta_df.join(df[[col]])
        return meta_df.fillna(0)

    def _get_base_model_predictions(self, df: pd.DataFrame, is_live: bool):
        seq_len = self.dl_config["sequence_length"]
        X_seq, _ = self._prepare_sequence_data(df)
        X_flat = df[self.flat_features].fillna(0)
        X_of = df[self.order_flow_features].fillna(0)

        if is_live:
            meta = {}
            if self.models.get("lstm") and X_seq.size > 0:
                meta["lstm_conf"] = np.max(
                    self.models["lstm"].predict(X_seq, verbose=0),
                )
            if self.models.get("transformer") and X_seq.size > 0:
                meta["transformer_conf"] = np.max(
                    self.models["transformer"].predict(X_seq, verbose=0),
                )
            if self.models.get("tabnet"):
                meta["tabnet_proba"] = np.max(
                    self.models["tabnet"].predict_proba(X_flat.tail(1).values),
                )
            if self.models.get("order_flow_lgbm"):
                meta["of_lgbm_prob"] = np.max(
                    self.models["order_flow_lgbm"].predict_proba(X_of.tail(1)),
                )
            if self.models.get("logistic_regression"):
                meta["log_reg_prob"] = np.max(
                    self.models["logistic_regression"].predict_proba(X_flat.tail(1)),
                )
            if self.models.get("garch"):
                meta["garch_volatility"] = (
                    self.models["garch"].forecast(horizon=1).variance.iloc[-1, 0]
                )
            return meta
        meta_df = pd.DataFrame(index=df.index[seq_len - 1 :])
        X_flat_aligned = X_flat.loc[meta_df.index]
        X_of_aligned = X_of.loc[meta_df.index]
        if self.models.get("lstm") and X_seq.size > 0:
            meta_df["lstm_conf"] = np.max(
                self.models["lstm"].predict(X_seq, verbose=0),
                axis=1,
            )
        if self.models.get("transformer") and X_seq.size > 0:
            meta_df["transformer_conf"] = np.max(
                self.models["transformer"].predict(X_seq, verbose=0),
                axis=1,
            )
        if self.models.get("tabnet"):
            meta_df["tabnet_proba"] = np.max(
                self.models["tabnet"].predict_proba(X_flat_aligned.values),
                axis=1,
            )
        if self.models.get("order_flow_lgbm"):
            meta_df["of_lgbm_prob"] = np.max(
                self.models["order_flow_lgbm"].predict_proba(X_of_aligned),
                axis=1,
            )
        if self.models.get("logistic_regression"):
            meta_df["log_reg_prob"] = np.max(
                self.models["logistic_regression"].predict_proba(X_flat_aligned),
                axis=1,
            )
        if self.models.get("garch"):
            meta_df["garch_volatility"] = self.models[
                "garch"
            ].conditional_volatility.reindex(meta_df.index, method="ffill")
        return meta_df

    def _prepare_sequence_data(self, df: pd.DataFrame, target_series: pd.Series = None):
        for col in self.sequence_features:
            if col not in df.columns:
                df[col] = 0.0
        features_df = df[self.sequence_features].copy().fillna(0)
        seq_len = self.dl_config["sequence_length"]
        X: list[Any] = []
        y: list[Any] = []
        if len(features_df) < seq_len:
            return np.array(X), np.array(y)
        for i in range(int(len(features_df) - seq_len + 1)):
            X.append(features_df.iloc[i : i + seq_len].values)
            if target_series is not None:
                y.append(target_series.iloc[i + seq_len - 1])
        return np.array(X), np.array(y) if target_series is not None else None

    def _train_dl_model(self, X_seq, y_seq_encoded, num_classes, is_transformer=False):
        try:
            inputs = Input(shape=(X_seq.shape[1], X_seq.shape[2]))
            x = (
                MultiHeadAttention(
                    num_heads=self.dl_config["transformer_heads"],
                    key_dim=self.dl_config["transformer_key_dim"],
                )(inputs, inputs)
                if is_transformer
                else LSTM(self.dl_config["lstm_units"])(inputs)
            )
            if is_transformer:
                x = LayerNormalization(epsilon=1e-6)(inputs + x)
                x = Flatten()(x)
            x = Dropout(self.dl_config["dropout_rate"])(x)
            outputs = Dense(num_classes, activation="softmax")(x)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
            model.fit(
                X_seq,
                y_seq_encoded,
                epochs=self.dl_config["epochs"],
                batch_size=self.dl_config["batch_size"],
                verbose=0,
            )
            return model
        except Exception as e:
            self.logger.error(f"DL Model training failed: {e}")
            return None

    def _train_tabnet_model(self, X_flat, y_flat_encoded):
        try:
            model = TabNetClassifier()
            model.fit(
                X_flat.values,
                y_flat_encoded,
                max_epochs=50,
                patience=20,
                batch_size=1024,
            )
            return model
        except Exception as e:
            self.logger.error(f"TabNet training failed: {e}")
            return None
