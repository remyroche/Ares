import os
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump, load
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import CONFIG
from src.utils.logger import system_logger

from .regime_ensembles.bear_trend_ensemble import BearTrendEnsemble
from .regime_ensembles.bull_trend_ensemble import BullTrendEnsemble
from .regime_ensembles.sideways_range_ensemble import SidewaysRangeEnsemble
from .regime_ensembles.volatile_regime_ensemble import VolatileRegimeEnsemble


class RegimePredictiveEnsembles:
    """
    Orchestrates the training and prediction workflows for all specialized ensembles.
    Now includes checkpointing for ensemble models and a sophisticated global meta-learner
    for final prediction combining outputs from all regime-specific ensembles and market context.
    """

    def __init__(self, config):
        self.config = config.get("analyst", {})
        self.logger = system_logger.getChild("PredictiveEnsembles.Orchestrator")

        # Initialize all possible ensemble instances - updated for new regime classification
        self.regime_ensembles = {
            "BULL_TREND": BullTrendEnsemble(config, "BullTrendEnsemble"),
            "BEAR_TREND": BearTrendEnsemble(config, "BearTrendEnsemble"),
            "SIDEWAYS_RANGE": SidewaysRangeEnsemble(config, "SidewaysRangeEnsemble"),
            "VOLATILE_REGIME": VolatileRegimeEnsemble(config, "VolatileRegimeEnsemble"),
        }
        # Model storage path is now dynamic based on checkpoint config
        self.model_storage_dir = os.path.join(
            CONFIG["CHECKPOINT_DIR"],
            "analyst_models",
            "ensembles",
        )
        os.makedirs(self.model_storage_dir, exist_ok=True)

        # Global Meta-Learner for final decision
        self.global_meta_learner: LGBMClassifier | None = None
        self.global_meta_scaler: StandardScaler | None = None
        self.global_meta_label_encoder: LabelEncoder | None = None
        self.global_meta_learner_path = os.path.join(
            self.model_storage_dir,
            "global_meta_learner.joblib",
        )
        self.global_meta_scaler_path = os.path.join(
            self.model_storage_dir,
            "global_meta_scaler.joblib",
        )
        self.global_meta_label_encoder_path = os.path.join(
            self.model_storage_dir,
            "global_meta_label_encoder.joblib",
        )

        # Load global meta-learner if exists
        self._load_global_meta_learner()

        # Configuration for the global meta-learner
        self.global_meta_config = self.config.get(
            "global_meta_learner",
            {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "verbose": -1,
            },
        )
        self.overall_confidence_threshold = self.config.get(
            "overall_confidence_threshold",
            0.55,
        )

    def train_all_models(
        self,
        asset: str,
        prepared_data: pd.DataFrame,
        model_path_prefix: str | None = None,
    ):
        """
        Orchestrates the training of all regime-specific ensembles.
        It splits the prepared data by regime and passes the relevant slice to each ensemble.
        After individual ensembles are trained, it trains a global meta-learner.

        Args:
            asset (str): The trading asset (e.g., "BTCUSDT").
            prepared_data (pd.DataFrame): The full prepared historical data with 'regime' and 'target' columns.
            model_path_prefix (str, optional): A prefix for saving models (e.g., includes fold_id).
        """
        self.logger.info(
            f"Orchestrator: Starting training for all ensembles for asset {asset} (prefix: {model_path_prefix})...",
        )

        if (
            "Market_Regime_Label" not in prepared_data.columns
            or "target" not in prepared_data.columns
        ):
            self.logger.error(
                "Prepared data is missing 'Market_Regime_Label' or 'target' column. Halting training.",
            )
            return

        # Prepare data for global meta-learner training
        meta_learner_data = []  # To store inputs for the global meta-learner

        for regime_key, ensemble_instance in self.regime_ensembles.items():
            self.logger.info(f"--- Processing ensemble for {regime_key} ---")

            # Filter data for the current regime
            regime_data = prepared_data[
                prepared_data["Market_Regime_Label"] == regime_key
            ]

            if regime_data.empty or len(regime_data["target"].unique()) < 2:
                self.logger.warning(
                    f"Insufficient or single-class data for {regime_key}. Skipping training.",
                )
                continue

            historical_features = regime_data.drop(
                columns=["target", "Market_Regime_Label"],
                errors="ignore",
            )
            historical_targets = regime_data["target"]

            # Construct the specific model path for this ensemble and fold
            model_file_name = f"{regime_key.lower()}_ensemble.joblib"
            if model_path_prefix:
                full_model_path = f"{model_path_prefix}{model_file_name}"
            else:
                full_model_path = os.path.join(
                    self.model_storage_dir,
                    f"final_{model_file_name}",
                )

            # Try to load the model first
            if os.path.exists(full_model_path):
                self.logger.info(
                    f"Attempting to load {regime_key} ensemble from {full_model_path}...",
                )
                if ensemble_instance.load_model(full_model_path):
                    self.logger.info(f"Successfully loaded {regime_key} ensemble.")
                else:
                    self.logger.warning(
                        f"Failed to load {regime_key} ensemble from {full_model_path}. Retraining.",
                    )
                    ensemble_instance.train_ensemble(
                        historical_features,
                        historical_targets,
                    )
            else:
                # If not loaded, train the model
                ensemble_instance.train_ensemble(
                    historical_features,
                    historical_targets,
                )

            # Save the trained model
            if ensemble_instance.trained:
                ensemble_instance.save_model(full_model_path)

                # Collect predictions from this trained ensemble for meta-learner training
                # Use the full historical_features (not just regime_data) to get predictions for all time points
                # This ensures the meta-learner has a consistent time index.
                ensemble_predictions_on_full_data = (
                    ensemble_instance.get_prediction_on_historical_data(
                        historical_features,
                    )
                )

                # Add regime key to prediction outputs for meta-learner
                for idx, row in ensemble_predictions_on_full_data.iterrows():
                    meta_learner_data.append(
                        {
                            "timestamp": idx,
                            "regime": regime_key,
                            "prediction": row["prediction"],
                            "confidence": row["confidence"],
                            "true_target": prepared_data.loc[
                                idx,
                                "target",
                            ],  # Get true target for this timestamp
                            # Add other high-level features here if needed for meta-learner
                            # e.g., prepared_data.loc[idx, 'market_health_score']
                        },
                    )
            else:
                self.logger.warning(
                    f"Ensemble {regime_key} was not trained/loaded successfully. Skipping for meta-learner.",
                )

        # Train the global meta-learner
        if meta_learner_data:
            self._train_global_meta_learner(meta_learner_data)
        else:
            self.logger.warning(
                "No data collected for global meta-learner training. Skipping.",
            )

    def get_all_predictions(
        self,
        asset: str,
        current_features: pd.DataFrame,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Gets a prediction by identifying the current regime and delegating to the
        appropriate trained ensemble. The final prediction is made by the global meta-learner.
        """
        primary_regime = self.get_current_regime(current_features)

        # Collect predictions and confidences from all individual ensembles
        ensemble_predictions_for_meta = {}
        ensemble_confidences_for_meta = {}
        combined_base_predictions = {}  # To return all base model predictions

        for regime_key, ensemble_instance in self.regime_ensembles.items():
            if not ensemble_instance.trained:
                # Attempt to load the final model if not already loaded (e.g., at startup)
                final_model_file_name = os.path.join(
                    self.model_storage_dir,
                    f"final_{regime_key.lower()}_ensemble.joblib",
                )
                if not ensemble_instance.load_model(final_model_file_name):
                    self.logger.warning(
                        f"Could not load final model for {regime_key}. Skipping its prediction.",
                    )
                    continue

            prediction_output = ensemble_instance.get_prediction(
                current_features,
                **kwargs,
            )

            # Store raw predictions and confidence for meta-learner input
            ensemble_predictions_for_meta[regime_key] = prediction_output.get(
                "prediction",
                "HOLD",
            )
            ensemble_confidences_for_meta[regime_key] = prediction_output.get(
                "confidence",
                0.0,
            )

            # Get detailed base predictions from each ensemble and combine
            if hasattr(ensemble_instance, "_get_meta_features"):
                base_preds_dict = ensemble_instance._get_meta_features(
                    current_features,
                    is_live=True,
                    **kwargs,
                )
                for model_name, pred_value in base_preds_dict.items():
                    # Prefix model names with regime to avoid clashes if model names are not unique
                    unique_model_name = f"{regime_key}_{model_name}"
                    combined_base_predictions[unique_model_name] = pred_value

        # Make final prediction using the global meta-learner
        final_prediction, final_confidence = self._predict_with_global_meta_learner(
            primary_regime,
            ensemble_predictions_for_meta,
            ensemble_confidences_for_meta,
            current_features,
        )

        # Include the ensemble's current weights in the output (from the primary ensemble or a combined view)
        current_ensemble_weights_snapshot = {
            regime: ens.ensemble_weights if hasattr(ens, "ensemble_weights") else {}
            for regime, ens in self.regime_ensembles.items()
        }

        return {
            "prediction": final_prediction,
            "confidence": final_confidence,
            "regime": primary_regime,
            "base_predictions": combined_base_predictions,
            "ensemble_weights": current_ensemble_weights_snapshot,
        }

    def _train_global_meta_learner(self, meta_learner_raw_data: list[dict[str, Any]]):
        """
        Trains the global meta-learner using outputs from individual ensembles
        and high-level market context.
        """
        self.logger.info("Training global meta-learner...")

        meta_df = pd.DataFrame(meta_learner_raw_data)
        meta_df.set_index("timestamp", inplace=True)
        meta_df.sort_index(inplace=True)

        # Prepare features for the meta-learner
        # Inputs: predictions and confidences from each ensemble, plus market context

        # One-hot encode the 'regime' column
        meta_df = pd.get_dummies(meta_df, columns=["regime"], prefix="regime")

        # Create columns for each ensemble's prediction and confidence
        # Initialize columns to 0 or 'HOLD' to handle cases where an ensemble might not have predicted
        all_regimes = list(self.regime_ensembles.keys())
        for r in all_regimes:
            meta_df[f"{r}_prediction"] = meta_df.apply(
                lambda row: row["prediction"] if row[f"regime_{r}"] == 1 else "HOLD",
                axis=1,
            )
            meta_df[f"{r}_confidence"] = meta_df.apply(
                lambda row: row["confidence"] if row[f"regime_{r}"] == 1 else 0.0,
                axis=1,
            )

        # Drop original 'prediction' and 'confidence' columns as they are now split by regime
        meta_df.drop(columns=["prediction", "confidence"], inplace=True)

        # One-hot encode the prediction columns (BUY, SELL, HOLD)
        prediction_cols = [f"{r}_prediction" for r in all_regimes]
        for col in prediction_cols:
            meta_df = pd.get_dummies(meta_df, columns=[col], prefix=col)

        # Define features (X) and target (y) for the meta-learner
        meta_features = [
            col
            for col in meta_df.columns
            if col.startswith(
                (
                    "regime_",
                    "BULL_TREND_confidence",
                    "BEAR_TREND_confidence",
                    "SIDEWAYS_RANGE_confidence",
                    "VOLATILE_REGIME_confidence",
                ),
            )
            or "_prediction_" in col
        ]

        # Add high-level market context features (these need to be present in prepared_data)
        # For simplicity, I'll assume they are available in prepared_data and can be merged by index.
        # In a real scenario, you'd need to ensure these features are passed into meta_learner_raw_data.
        # Example: if 'market_health_score' was added to meta_learner_raw_data
        # meta_features.append('market_health_score')

        X_meta = meta_df[meta_features].copy()
        y_meta = meta_df["true_target"].copy()

        # Handle potential missing columns from meta_features (e.g., if a regime never occurred)
        for col in meta_features:
            if col not in X_meta.columns:
                X_meta[col] = 0  # Add missing columns with default value

        # Encode target labels (BUY, SELL, HOLD) to integers
        self.global_meta_label_encoder = LabelEncoder()
        y_encoded = self.global_meta_label_encoder.fit_transform(y_meta)

        # Scale features
        self.global_meta_scaler = StandardScaler()
        X_scaled = self.global_meta_scaler.fit_transform(X_meta)
        X_scaled_df = pd.DataFrame(X_scaled, index=X_meta.index, columns=X_meta.columns)

        # Train the global meta-learner
        self.global_meta_learner = LGBMClassifier(
            **self.global_meta_config,
            random_state=42,
        )

        # Use cross-validation for more robust training
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        for train_index, val_index in skf.split(X_scaled_df, y_encoded):
            X_train, X_val = X_scaled_df.iloc[train_index], X_scaled_df.iloc[val_index]
            y_train, y_val = y_encoded[train_index], y_encoded[val_index]
            self.global_meta_learner.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[LGBMClassifier.early_stopping(10, verbose=False)],
            )  # Early stopping

        self.logger.info("Global meta-learner trained successfully.")
        self._save_global_meta_learner()

    def _predict_with_global_meta_learner(
        self,
        primary_regime: str,
        ensemble_predictions: dict[str, str],
        ensemble_confidences: dict[str, float],
        current_features: pd.DataFrame,
    ) -> tuple[str, float]:
        """
        Uses the trained global meta-learner to make the final prediction.
        """
        if (
            not self.global_meta_learner
            or not self.global_meta_scaler
            or not self.global_meta_label_encoder
        ):
            self.logger.warning(
                "Global meta-learner not trained/loaded. Defaulting to HOLD.",
            )
            return "HOLD", 0.0

        # Prepare input for the global meta-learner for the current timestep
        meta_input_data = {"regime": primary_regime}

        # Add predictions and confidences from each ensemble
        all_regimes = list(self.regime_ensembles.keys())
        for r in all_regimes:
            pred = ensemble_predictions.get(r, "HOLD")
            conf = ensemble_confidences.get(r, 0.0)
            meta_input_data[f"{r}_prediction"] = pred
            meta_input_data[f"{r}_confidence"] = conf

        # Convert to DataFrame for scaling and prediction
        meta_input_df = pd.DataFrame([meta_input_data])

        # One-hot encode the 'regime' column
        meta_input_df = pd.get_dummies(
            meta_input_df,
            columns=["regime"],
            prefix="regime",
        )

        # One-hot encode the prediction columns (BUY, SELL, HOLD)
        prediction_cols_for_dummies = [f"{r}_prediction" for r in all_regimes]
        for col in prediction_cols_for_dummies:
            meta_input_df = pd.get_dummies(meta_input_df, columns=[col], prefix=col)

        # Ensure all columns that the meta-learner was trained on are present, fill missing with 0
        trained_features = (
            self.global_meta_scaler.feature_names_in_
            if hasattr(self.global_meta_scaler, "feature_names_in_")
            else []
        )

        # Reindex to match the training columns, filling missing with 0
        X_meta_live = meta_input_df.reindex(columns=trained_features, fill_value=0)

        # Scale the features
        X_meta_live_scaled = self.global_meta_scaler.transform(X_meta_live)

        # Make prediction
        proba = self.global_meta_learner.predict_proba(X_meta_live_scaled)[0]
        predicted_label_idx = np.argmax(proba)

        final_prediction = self.global_meta_label_encoder.inverse_transform(
            [predicted_label_idx],
        )[0]
        final_confidence = proba[predicted_label_idx]

        # Apply overall confidence threshold
        if final_confidence < self.overall_confidence_threshold:
            final_prediction = "HOLD"
            self.logger.info(
                f"Global meta-learner confidence ({final_confidence:.2f}) below threshold ({self.overall_confidence_threshold}). Final decision: HOLD.",
            )

        return final_prediction, final_confidence

    def _save_global_meta_learner(self):
        """Saves the global meta-learner and its scaler/encoder."""
        try:
            dump(self.global_meta_learner, self.global_meta_learner_path)
            dump(self.global_meta_scaler, self.global_meta_scaler_path)
            dump(self.global_meta_label_encoder, self.global_meta_label_encoder_path)
            self.logger.info(
                "Global meta-learner, scaler, and label encoder saved successfully.",
            )
        except Exception as e:
            self.logger.error(
                f"Error saving global meta-learner components: {e}",
                exc_info=True,
            )

    def _load_global_meta_learner(self):
        """Loads the global meta-learner and its scaler/encoder."""
        if (
            os.path.exists(self.global_meta_learner_path)
            and os.path.exists(self.global_meta_scaler_path)
            and os.path.exists(self.global_meta_label_encoder_path)
        ):
            try:
                self.global_meta_learner = load(self.global_meta_learner_path)
                self.global_meta_scaler = load(self.global_meta_scaler_path)
                self.global_meta_label_encoder = load(
                    self.global_meta_label_encoder_path,
                )
                self.logger.info(
                    "Global meta-learner, scaler, and label encoder loaded.",
                )
            except Exception as e:
                self.logger.error(
                    f"Error loading global meta-learner components: {e}",
                    exc_info=True,
                )
                self.global_meta_learner = None  # Reset on failure
                self.global_meta_scaler = None
                self.global_meta_label_encoder = None
        else:
            self.logger.info(
                "Global meta-learner components not found. Will train on first run.",
            )

    def get_current_regime(self, current_features: pd.DataFrame) -> str:
        """Determines the most recent market regime from the feature data."""
        if (
            not current_features.empty
            and "Market_Regime_Label" in current_features.columns
        ):
            return current_features["Market_Regime_Label"].iloc[-1]
        return "UNKNOWN"

    def save_model(self, ensemble_instance: Any, path: str):
        """Saves a trained ensemble instance to a file."""
        try:
            dump(ensemble_instance, path)
            self.logger.info(f"Successfully saved trained ensemble to {path}")
        except Exception as e:
            self.logger.error(
                f"Error saving ensemble model to {path}: {e}",
                exc_info=True,
            )

    def load_model(self, ensemble_instance: Any, path: str) -> bool:
        """Loads a trained ensemble instance from a file."""
        if not os.path.exists(path):
            return False
        try:
            loaded_ensemble = load(path)
            # Update the existing instance's state instead of replacing it
            ensemble_instance.__dict__.update(loaded_ensemble.__dict__)
            ensemble_instance.trained = True  # Ensure 'trained' flag is set
            self.logger.info(f"Successfully loaded pre-trained ensemble from {path}")
            return True
        except Exception as e:
            self.logger.error(
                f"Error loading ensemble model from {path}: {e}",
                exc_info=True,
            )
            return False

    def load_weights(self, weights: dict[str, Any]):
        """Loads updated weights into the ensembles for dynamic weighting."""
        for regime, ensemble_weights in weights.items():
            if regime in self.regime_ensembles:
                # Assuming BaseEnsemble has an attribute 'ensemble_weights'
                self.regime_ensembles[regime].ensemble_weights = ensemble_weights

    def get_current_weights(self) -> dict[str, Any]:
        """Returns the current weights of all ensembles."""
        return {
            regime: ens.ensemble_weights
            for regime, ens in self.regime_ensembles.items()
        }
