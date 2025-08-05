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


class VolatileRegimeEnsemble(BaseEnsemble):
    """
    This ensemble specializes in detecting and predicting during volatile market conditions.
    It combines signals from multiple models optimized for high volatility periods.
    """

    def __init__(self, config: dict, ensemble_name: str = "VolatileRegimeEnsemble"):
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
        """Trains multiple diverse base models for volatile regime detection."""
        self.logger.info("Training VolatileRegime base models...")

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

        # GARCH Model for volatility modeling
        try:
            self.logger.info("Training GARCH model for volatility modeling...")
            self.models["garch"] = self._train_garch_model(aligned_data, y_encoded)
        except Exception as e:
            self.logger.error(f"GARCH training failed: {e}")

        self.logger.info("✅ VolatileRegime base models training completed")

    def _prepare_sequence_data(self, df: pd.DataFrame, target_series: pd.Series = None):
        """Prepare sequence data for deep learning models."""
        try:
            sequence_length = self.dl_config["sequence_length"]

            # Prepare features for sequence
            feature_cols = self.flat_features + self.order_flow_features
            X = df[feature_cols].fillna(0).values

            # Create sequences
            X_seq = []
            y_seq = []

            for i in range(sequence_length, len(X)):
                X_seq.append(X[i - sequence_length : i])
                if target_series is not None:
                    y_seq.append(target_series.iloc[i])

            if len(X_seq) > 0:
                return np.array(X_seq), np.array(y_seq)
            return np.array([]), np.array([])

        except Exception as e:
            self.logger.error(f"Error preparing sequence data: {e}")
            return np.array([]), np.array([])

    def _train_dl_model(self, X_seq, y_seq_encoded, num_classes, is_transformer=False):
        """Train deep learning model (LSTM or Transformer)."""
        try:
            if len(X_seq) == 0:
                return None

            input_shape = (X_seq.shape[1], X_seq.shape[2])

            if is_transformer:
                return self._build_transformer_model(input_shape, num_classes)
            return self._build_lstm_model(input_shape, num_classes)

        except Exception as e:
            self.logger.error(f"Error training DL model: {e}")
            return None

    def _build_lstm_model(self, input_shape, num_classes):
        """Build LSTM model."""
        try:
            inputs = Input(shape=input_shape)

            # LSTM layers
            x = LSTM(self.dl_config["lstm_units"], return_sequences=True)(inputs)
            x = Dropout(self.dl_config["dropout_rate"])(x)
            x = LSTM(self.dl_config["lstm_units"] // 2)(x)
            x = Dropout(self.dl_config["dropout_rate"])(x)

            # Dense layers
            x = Dense(64, activation="relu")(x)
            x = Dropout(self.dl_config["dropout_rate"])(x)
            x = Dense(32, activation="relu")(x)

            # Output layer
            outputs = Dense(num_classes, activation="softmax")(x)

            model = Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            # Train the model
            model.fit(
                X_seq,
                y_seq_encoded,
                epochs=self.dl_config["epochs"],
                batch_size=self.dl_config["batch_size"],
                validation_split=0.2,
                verbose=0,
            )

            return model

        except Exception as e:
            self.logger.error(f"Error building LSTM model: {e}")
            return None

    def _build_transformer_model(self, input_shape, num_classes):
        """Build Transformer model."""
        try:
            inputs = Input(shape=input_shape)

            # Multi-head attention
            x = MultiHeadAttention(
                num_heads=self.dl_config["transformer_heads"],
                key_dim=self.dl_config["transformer_key_dim"],
            )(inputs, inputs)
            x = LayerNormalization()(x)
            x = Dropout(self.dl_config["dropout_rate"])(x)

            # Feed-forward network
            x = Dense(128, activation="relu")(x)
            x = Dropout(self.dl_config["dropout_rate"])(x)
            x = Dense(input_shape[1])(x)
            x = LayerNormalization()(x)

            # Global average pooling
            x = Flatten()(x)
            x = Dense(64, activation="relu")(x)
            x = Dropout(self.dl_config["dropout_rate"])(x)
            x = Dense(32, activation="relu")(x)

            # Output layer
            outputs = Dense(num_classes, activation="softmax")(x)

            model = Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            # Train the model
            model.fit(
                X_seq,
                y_seq_encoded,
                epochs=self.dl_config["epochs"],
                batch_size=self.dl_config["batch_size"],
                validation_split=0.2,
                verbose=0,
            )

            return model

        except Exception as e:
            self.logger.error(f"Error building Transformer model: {e}")
            return None

    def _train_tabnet_model(self, X_flat, y_flat_encoded):
        """Train TabNet model."""
        try:
            tabnet = TabNetClassifier()
            tabnet.fit(
                X_flat.values,
                y_flat_encoded,
                max_epochs=50,
                patience=20,
                batch_size=1024,
            )
            return tabnet
        except Exception as e:
            self.logger.error(f"TabNet training failed: {e}")
            return None

    def _train_garch_model(self, aligned_data, y_encoded):
        """Train GARCH model for volatility modeling."""
        try:
            # Use returns for GARCH modeling
            returns = aligned_data["close"].pct_change().dropna()

            # Fit GARCH model
            garch_model = arch_model(returns, vol="GARCH", p=1, q=1)
            fitted_model = garch_model.fit(disp="off")

            return fitted_model
        except Exception as e:
            self.logger.error(f"GARCH model training failed: {e}")
            return None

    def _generate_meta_features(self, aligned_data: pd.DataFrame) -> pd.DataFrame:
        """Generate meta-features specific to volatile regime detection."""
        meta_features = pd.DataFrame(index=aligned_data.index)

        # Volatility-specific features
        if "volatility_20" in aligned_data.columns:
            meta_features["volatility_percentile"] = (
                aligned_data["volatility_20"].rolling(100).rank(pct=True)
            )
            meta_features["volatility_acceleration"] = aligned_data[
                "volatility_20"
            ].diff()
            meta_features["volatility_momentum"] = aligned_data[
                "volatility_20"
            ] - aligned_data["volatility_20"].shift(5)

        # Volume volatility features
        if "volume" in aligned_data.columns:
            meta_features["volume_volatility"] = (
                aligned_data["volume"].rolling(20).std()
            )
            meta_features["volume_volatility_ratio"] = (
                meta_features["volume_volatility"]
                / aligned_data["volume"].rolling(20).mean()
            )

        # Price volatility features
        if "close" in aligned_data.columns:
            meta_features["price_volatility"] = (
                aligned_data["close"].pct_change().rolling(20).std()
            )
            meta_features["price_volatility_percentile"] = (
                meta_features["price_volatility"].rolling(100).rank(pct=True)
            )

        # Regime-specific features
        if "volatility_regime" in aligned_data.columns:
            meta_features["volatility_regime_numeric"] = aligned_data[
                "volatility_regime"
            ]

        # Fill NaN values
        meta_features = meta_features.fillna(0)

        return meta_features

    def predict(self, current_features: pd.DataFrame) -> tuple[float, float]:
        """Make prediction for volatile regime."""
        if not self.trained:
            self.logger.warning("VolatileRegime ensemble not trained")
            return 0.5, 0.5

        try:
            # Get base model predictions
            base_predictions = self._get_base_model_predictions(
                current_features,
                is_live=True,
            )

            if not base_predictions:
                return 0.5, 0.5

            # Calculate ensemble prediction
            predictions = list(base_predictions.values())
            confidences = [0.8] * len(predictions)  # Default confidence

            # Weighted average of predictions
            weighted_pred = np.average(predictions, weights=confidences)
            ensemble_confidence = np.mean(confidences)

            return weighted_pred, ensemble_confidence

            # Volatility LGBM
            if self.models["volatility_lgbm"]:
                pred = self.models["volatility_lgbm"].predict_proba(X_flat)[0]
                predictions["volatility_lgbm"] = pred[
                    1
                ]  # Probability of positive class
                confidences["volatility_lgbm"] = np.max(pred)

            # TabNet
            if self.models["volatility_tabnet"]:
                pred = self.models["volatility_tabnet"].predict_proba(X_flat.values)[0]
                predictions["volatility_tabnet"] = pred[1]
                confidences["volatility_tabnet"] = np.max(pred)

            # Order Flow LGBM
            if self.models["order_flow_lgbm"]:
                pred = self.models["order_flow_lgbm"].predict_proba(X_of)[0]
                predictions["order_flow_lgbm"] = pred[1]
                confidences["order_flow_lgbm"] = np.max(pred)

            # Random Forest
            if self.models["random_forest"]:
                pred = self.models["random_forest"].predict_proba(X_flat)[0]
                predictions["random_forest"] = pred[1]
                confidences["random_forest"] = np.max(pred)

            # SVM
            if self.models["svm"]:
                pred = self.models["svm"].predict_proba(X_flat)[0]
                predictions["svm"] = pred[1]
                confidences["svm"] = np.max(pred)

            # Combine predictions using weighted average
            if predictions:
                # Use confidence as weight
                total_weight = sum(confidences.values())
                if total_weight > 0:
                    weighted_prediction = (
                        sum(
                            pred * confidences[model]
                            for model, pred in predictions.items()
                        )
                        / total_weight
                    )
                    overall_confidence = np.mean(list(confidences.values()))
                else:
                    weighted_prediction = np.mean(list(predictions.values()))
                    overall_confidence = 0.5
            else:
                weighted_prediction = 0.5
                overall_confidence = 0.5

            return weighted_prediction, overall_confidence

        except Exception as e:
            self.logger.error(f"Error in VolatileRegime prediction: {e}")
            return 0.5, 0.5
