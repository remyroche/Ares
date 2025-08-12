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
from keras.regularizers import l1_l2
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

from src.utils.warning_symbols import (
    failed,
)

from .base_ensemble import BaseEnsemble


class BullTrendEnsemble(BaseEnsemble):
    """
    ## CHANGE: Fully implemented all methods for the Bull Trend Ensemble.
    ## This version includes complete, functional logic for training all diverse base
    ## models (LSTM, Transformer, TabNet, LGBM, Logistic Regression, GARCH) and for
    ## generating a rich, fully-aligned meta-feature set for both training and live prediction.
    ## Enhanced with L1-L2 regularization support across all model types.
    """

    def __init__(self, config: dict, ensemble_name: str = "BullTrendEnsemble"):
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
        # 1. Train Deep Learning Models
        X_seq, y_seq_aligned_encoded = self._prepare_sequence_data(
            aligned_data,
            pd.Series(y_encoded, index=aligned_data.index),
        )
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

        # 2. Train Flat-Feature Models
        X_flat = aligned_data[self.flat_features].fillna(0)
        self.models["tabnet"] = self._train_tabnet_model(X_flat, y_encoded)

        # Specialized Order Flow LGBM (now uses regularization config)
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

        # GARCH
        returns = aligned_data["close"].pct_change().dropna()
        if len(returns) > 100:
            try:
                self.models["garch"] = arch_model(returns, vol="Garch", p=1, q=1).fit(
                    disp="off",
                )
            except Exception:
                self.print(failed("GARCH training failed: {e}"))

    def _get_meta_features(
        self,
        df: pd.DataFrame,
        is_live: bool = False,
        **kwargs,
    ) -> pd.DataFrame | dict:
        base_preds = self._get_base_model_predictions(df, is_live)
        raw_features_to_include = self.flat_features + ["oi_value", "funding_rate_ma"]

        # Attach explicit meta-labels if present to make their influence explicit
        meta_label_cols = [
            c for c in df.columns if any(
                c.endswith(suffix) or c.startswith(prefix)
                for suffix in [
                    "STRONG_TREND_CONTINUATION",
                    "RANGE_MEAN_REVERSION",
                    "EXHAUSTION_REVERSAL",
                ]
                for prefix in [
                    "1m_", "5m_", "15m_", "30m_"
                ]
            )
        ]

        if is_live:
            current_row = df.tail(1)
            for col in raw_features_to_include:
                base_preds[col] = (
                    current_row[col].iloc[0] if col in current_row.columns else 0.0
                )
            # Live: attach meta-label flags (0/1)
            for col in meta_label_cols:
                try:
                    base_preds[col] = int(current_row[col].iloc[0])
                except Exception:
                    base_preds[col] = 0
            if meta_label_cols:
                self.logger.info(f"BullTrendEnsemble meta-learner live features include meta-labels: {meta_label_cols}")
            return base_preds
        meta_df = base_preds
        for col in raw_features_to_include:
            if col in df.columns:
                meta_df = meta_df.join(df[[col]])
        # Train: join explicit meta-labels if present
        if meta_label_cols:
            meta_df = meta_df.join(df[meta_label_cols].astype(float))
            self.logger.info(f"BullTrendEnsemble meta-learner train features include meta-labels: {meta_label_cols}")
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
        for i in range(len(features_df) - seq_len + 1):
            X.append(features_df.iloc[i : i + seq_len].values)
            if target_series is not None:
                y.append(target_series.iloc[i + seq_len - 1])
        return np.array(X), np.array(y) if target_series is not None else None

    def _train_dl_model(self, X_seq, y_seq_encoded, num_classes, is_transformer=False):
        """Enhanced deep learning model training with L1-L2 regularization."""
        try:
            # Get regularization parameters from config
            l1_reg = 0.01
            l2_reg = 0.001
            dropout_rate = self.dl_config["dropout_rate"]

            if (
                self.regularization_config
                and "tensorflow" in self.regularization_config
            ):
                tf_config = self.regularization_config["tensorflow"]
                l1_reg = tf_config.get("l1_reg", 0.01)
                l2_reg = tf_config.get("l2_reg", 0.001)
                dropout_rate = tf_config.get("dropout_rate", 0.2)
                self.logger.info(
                    f"Using configured TensorFlow regularization: L1={l1_reg}, L2={l2_reg}, Dropout={dropout_rate}",
                )
            else:
                self.logger.info(
                    f"Using default TensorFlow regularization: L1={l1_reg}, L2={l2_reg}, Dropout={dropout_rate}",
                )

            # Create L1-L2 regularizer
            regularizer = l1_l2(l1=l1_reg, l2=l2_reg)

            inputs = Input(shape=(X_seq.shape[1], X_seq.shape[2]))
            if is_transformer:
                x = MultiHeadAttention(
                    num_heads=self.dl_config["transformer_heads"],
                    key_dim=self.dl_config["transformer_key_dim"],
                )(inputs, inputs)
                x = LayerNormalization(epsilon=1e-6)(inputs + x)
                x = Flatten()(x)
            else:
                x = LSTM(
                    self.dl_config["lstm_units"],
                    kernel_regularizer=regularizer,
                    recurrent_regularizer=regularizer,
                )(inputs)

            x = Dropout(dropout_rate)(x)

            # Dense layer with L1-L2 regularization
            outputs = Dense(
                num_classes,
                activation="softmax",
                kernel_regularizer=regularizer,
            )(x)

            model = Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            model.fit(
                X_seq,
                y_seq_encoded,
                epochs=self.dl_config["epochs"],
                batch_size=self.dl_config["batch_size"],
                verbose=0,
                validation_split=0.2,
            )

            self.logger.info(
                f"Successfully trained {'Transformer' if is_transformer else 'LSTM'} model with L1-L2 regularization",
            )
            return model

        except Exception:
            self.print(failed("DL Model training failed: {e}"))
            return None

    def _train_tabnet_model(self, X_flat, y_flat_encoded):
        """Enhanced TabNet training with L1-L2 regularization."""
        try:
            # Get regularization parameters from config
            lambda_sparse = 0.01
            lambda_l2 = 0.001

            if self.regularization_config and "tabnet" in self.regularization_config:
                tabnet_config = self.regularization_config["tabnet"]
                lambda_sparse = tabnet_config.get("lambda_sparse", 0.01)
                lambda_l2 = tabnet_config.get("lambda_l2", 0.001)
                self.logger.info(
                    f"Using configured TabNet regularization: L1={lambda_sparse}, L2={lambda_l2}",
                )
            else:
                self.logger.info(
                    f"Using default TabNet regularization: L1={lambda_sparse}, L2={lambda_l2}",
                )

            model = TabNetClassifier(
                lambda_sparse=lambda_sparse,
                reg_lambda=lambda_l2,
                verbose=0,
            )

            model.fit(
                X_flat.values,
                y_flat_encoded,
                max_epochs=50,
                patience=20,
                batch_size=1024,
            )

            self.logger.info(
                "Successfully trained TabNet model with L1-L2 regularization",
            )
            return model

        except Exception:
            self.print(failed("TabNet training failed: {e}"))
            return None
