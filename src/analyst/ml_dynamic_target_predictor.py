import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from datetime import datetime


class MLDynamicTargetPredictor:
    """
    Machine Learning-based dynamic target predictor for SR fade and breakout orders.

    This class predicts optimal take-profit and stop-loss levels based on:
    - Current market conditions
    - Support/Resistance strength
    - Volatility patterns
    - Order flow dynamics
    - Historical success rates at different target levels

    The predictor continuously learns and adapts to market conditions,
    replacing fixed ATR multipliers with ML-based dynamic targets.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("analyst", {}).get("ml_dynamic_target_predictor", {})
        self.logger = logging.getLogger(self.__class__.__name__)

        # Model configuration
        self.retrain_interval_hours = self.config.get("retrain_interval_hours", 24)
        self.lookback_window = self.config.get("lookback_window", 200)
        self.min_samples_for_training = self.config.get("min_samples_for_training", 500)
        self.validation_split = self.config.get("validation_split", 0.2)

        # Target prediction bounds
        self.min_tp_multiplier = self.config.get("min_tp_multiplier", 0.5)
        self.max_tp_multiplier = self.config.get("max_tp_multiplier", 6.0)
        self.min_sl_multiplier = self.config.get("min_sl_multiplier", 0.2)
        self.max_sl_multiplier = self.config.get("max_sl_multiplier", 2.0)

        # Fallback values
        self.fallback_tp_multiplier = self.config.get("fallback_tp_multiplier", 2.0)
        self.fallback_sl_multiplier = self.config.get("fallback_sl_multiplier", 0.5)

        # Model storage
        checkpoint_dir = config.get("CHECKPOINT_DIR", "checkpoints")
        self.model_dir = os.path.join(checkpoint_dir, "ml_target_models")
        os.makedirs(self.model_dir, exist_ok=True)

        # Models for different scenarios
        self.models = {
            "sr_fade_long_tp": None,
            "sr_fade_long_sl": None,
            "sr_fade_short_tp": None,
            "sr_fade_short_sl": None,
            "sr_breakout_long_tp": None,
            "sr_breakout_long_sl": None,
            "sr_breakout_short_tp": None,
            "sr_breakout_short_sl": None,
        }

        # Feature scalers
        self.scalers = {}
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()

        # Last training time
        self.last_training_time = None

        # Load existing models if available
        self._load_models()

    def predict_dynamic_targets(
        self,
        signal_type: str,
        technical_analysis_data: Dict[str, Any],
        current_features: pd.DataFrame,
        current_atr: float,
    ) -> Dict[str, float]:
        """
        Predict dynamic targets for SR fade and breakout orders.

        Args:
            signal_type: One of ["SR_FADE_LONG", "SR_FADE_SHORT", "SR_BREAKOUT_LONG", "SR_BREAKOUT_SHORT"]
            technical_analysis_data: Current technical analysis data
            current_features: Current feature vector for ML prediction
            current_atr: Current ATR value

        Returns:
            Dict containing predicted target and stop loss multipliers
        """
        try:
            # Extract relevant features for prediction
            prediction_features = self._extract_prediction_features(
                technical_analysis_data, current_features, current_atr
            )

            # Get model names for this signal type
            tp_model_name, sl_model_name = self._get_model_names(signal_type)

            # Predict take profit multiplier
            tp_multiplier = self._predict_single_target(
                tp_model_name, prediction_features, "tp"
            )

            # Predict stop loss multiplier
            sl_multiplier = self._predict_single_target(
                sl_model_name, prediction_features, "sl"
            )

            # Apply bounds and validation
            tp_multiplier = np.clip(
                tp_multiplier, self.min_tp_multiplier, self.max_tp_multiplier
            )
            sl_multiplier = np.clip(
                sl_multiplier, self.min_sl_multiplier, self.max_sl_multiplier
            )

            # Calculate actual prices
            entry_price = self._get_entry_price(signal_type, technical_analysis_data)
            if entry_price is None:
                return self._get_fallback_targets(current_atr)

            if "LONG" in signal_type:
                take_profit = entry_price + (current_atr * tp_multiplier)
                stop_loss = entry_price - (current_atr * sl_multiplier)
            else:  # SHORT
                take_profit = entry_price - (current_atr * tp_multiplier)
                stop_loss = entry_price + (current_atr * sl_multiplier)

            self.logger.info(
                f"ML Dynamic Targets - {signal_type}: TP={tp_multiplier:.2f}x ATR, "
                f"SL={sl_multiplier:.2f}x ATR (Entry: {entry_price:.6f})"
            )

            return {
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "tp_multiplier": tp_multiplier,
                "sl_multiplier": sl_multiplier,
                "entry_price": entry_price,
                "prediction_confidence": self._calculate_prediction_confidence(
                    tp_model_name, sl_model_name
                ),
            }

        except Exception as e:
            self.logger.error(f"Error predicting dynamic targets: {e}", exc_info=True)
            return self._get_fallback_targets(current_atr, technical_analysis_data)

    def train_models(self, historical_data: pd.DataFrame, force_retrain: bool = False):
        """
        Train ML models for dynamic target prediction.

        Args:
            historical_data: Historical data with features and outcomes
            force_retrain: Force retraining even if recently trained
        """
        # Check if retraining is needed
        if not force_retrain and self._should_skip_training():
            return

        try:
            self.logger.info("Starting ML dynamic target model training...")

            # Prepare training data for each signal type
            training_datasets = self._prepare_training_data(historical_data)

            for signal_type, dataset in training_datasets.items():
                if len(dataset) < self.min_samples_for_training:
                    self.logger.warning(
                        f"Insufficient data for {signal_type}: {len(dataset)} samples "
                        f"(minimum: {self.min_samples_for_training})"
                    )
                    continue

                self._train_signal_models(signal_type, dataset)

            # Save all trained models
            self._save_models()
            self.last_training_time = datetime.now()

            self.logger.info("ML dynamic target model training completed successfully")

        except Exception as e:
            self.logger.error(f"Error training ML target models: {e}", exc_info=True)

    def _extract_prediction_features(
        self,
        technical_analysis_data: Dict[str, Any],
        current_features: pd.DataFrame,
        current_atr: float,
    ) -> np.ndarray:
        """Extract relevant features for target prediction."""
        features = []

        # ATR-based features
        features.append(current_atr)
        features.append(
            technical_analysis_data.get("close", 0) / current_atr
            if current_atr > 0
            else 0
        )

        # Price-based features
        features.extend(
            [
                technical_analysis_data.get("close", 0),
                technical_analysis_data.get("high", 0),
                technical_analysis_data.get("low", 0),
                technical_analysis_data.get("volume", 0),
            ]
        )

        # Technical indicators from current_features (last row)
        if not current_features.empty:
            latest_features = current_features.iloc[-1]
            indicator_features = [
                "ADX",
                "MACD_HIST",
                "rsi",
                "stoch_k",
                "bb_bandwidth",
                "volume_delta",
                "OBV",
                "CMF",
                "position_in_range",
            ]

            for feature in indicator_features:
                features.append(latest_features.get(feature, 0))
        else:
            # Add zeros if no current features available
            features.extend([0] * 9)

        # SR interaction features
        features.extend(
            [
                technical_analysis_data.get("is_sr_support_interaction", 0),
                technical_analysis_data.get("is_sr_resistance_interaction", 0),
            ]
        )

        # Market regime features
        if (
            not current_features.empty
            and "Market_Regime_Label" in current_features.columns
        ):
            regime = current_features.iloc[-1].get("Market_Regime_Label", "UNKNOWN")
            regime_encoding = {
                "BULL_TREND": [1, 0, 0, 0],
                "BEAR_TREND": [0, 1, 0, 0],
                "SIDEWAYS_RANGE": [0, 0, 1, 0],
                "HIGH_IMPACT_CANDLE": [0, 0, 0, 1],
            }
            features.extend(regime_encoding.get(regime, [0, 0, 0, 0]))
        else:
            features.extend([0, 0, 0, 0])

        return np.array(features).reshape(1, -1)

    def _get_model_names(self, signal_type: str) -> Tuple[str, str]:
        """Get model names for TP and SL prediction."""
        base_name = signal_type.lower()
        return f"{base_name}_tp", f"{base_name}_sl"

    def _predict_single_target(
        self, model_name: str, features: np.ndarray, target_type: str
    ) -> float:
        """Predict a single target (TP or SL) using the specified model."""
        model = self.models.get(model_name)
        scaler = self.scalers.get(model_name)

        if model is None or scaler is None:
            self.logger.warning(f"Model {model_name} not available, using fallback")
            return (
                self.fallback_tp_multiplier
                if target_type == "tp"
                else self.fallback_sl_multiplier
            )

        try:
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            return float(prediction)
        except Exception as e:
            self.logger.error(f"Error predicting with model {model_name}: {e}")
            return (
                self.fallback_tp_multiplier
                if target_type == "tp"
                else self.fallback_sl_multiplier
            )

    def _get_entry_price(
        self, signal_type: str, technical_analysis_data: Dict[str, Any]
    ) -> Optional[float]:
        """Get entry price based on signal type."""
        if signal_type == "SR_FADE_LONG":
            return technical_analysis_data.get("low")
        elif signal_type == "SR_FADE_SHORT":
            return technical_analysis_data.get("high")
        elif signal_type == "SR_BREAKOUT_LONG":
            return technical_analysis_data.get("high")
        elif signal_type == "SR_BREAKOUT_SHORT":
            return technical_analysis_data.get("low")
        return technical_analysis_data.get("close")

    def _get_fallback_targets(
        self,
        current_atr: float,
        technical_analysis_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Return fallback targets when ML prediction fails."""
        if technical_analysis_data:
            entry_price = technical_analysis_data.get("close", 0)
            return {
                "take_profit": entry_price
                + (current_atr * self.fallback_tp_multiplier),
                "stop_loss": entry_price - (current_atr * self.fallback_sl_multiplier),
                "tp_multiplier": self.fallback_tp_multiplier,
                "sl_multiplier": self.fallback_sl_multiplier,
                "entry_price": entry_price,
                "prediction_confidence": 0.0,
            }
        else:
            return {
                "tp_multiplier": self.fallback_tp_multiplier,
                "sl_multiplier": self.fallback_sl_multiplier,
                "prediction_confidence": 0.0,
            }

    def _prepare_training_data(
        self, historical_data: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Prepare training datasets for each signal type."""
        # This would analyze historical trades and their outcomes
        # to create training data for optimal target prediction

        training_datasets = {}
        signal_types = [
            "SR_FADE_LONG",
            "SR_FADE_SHORT",
            "SR_BREAKOUT_LONG",
            "SR_BREAKOUT_SHORT",
        ]

        for signal_type in signal_types:
            # Filter data for this signal type
            signal_data = historical_data[
                historical_data.get("target_sr", "") == signal_type
            ].copy()

            if len(signal_data) > 0:
                # Add optimal multiplier calculations based on historical outcomes
                signal_data = self._calculate_optimal_multipliers(
                    signal_data, signal_type
                )
                training_datasets[signal_type] = signal_data

        return training_datasets

    def _calculate_optimal_multipliers(
        self, signal_data: pd.DataFrame, signal_type: str
    ) -> pd.DataFrame:
        """Calculate optimal TP/SL multipliers based on historical outcomes."""
        # This would analyze what TP/SL levels would have been optimal
        # for each historical trade setup

        optimal_data = signal_data.copy()

        # Simulate different multiplier values and find optimal ones
        # based on risk-reward ratios and success rates

        # Placeholder implementation - in practice, this would be more sophisticated
        for idx in optimal_data.index:
            # row = optimal_data.loc[idx]
            # atr = row.get("ATR", 1.0)

            # Calculate optimal multipliers based on future price action
            # This is a simplified version - real implementation would analyze
            # forward-looking windows to find optimal targets

            optimal_tp_mult = np.random.uniform(1.0, 4.0)  # Placeholder
            optimal_sl_mult = np.random.uniform(0.3, 1.5)  # Placeholder

            optimal_data.loc[idx, "optimal_tp_multiplier"] = optimal_tp_mult
            optimal_data.loc[idx, "optimal_sl_multiplier"] = optimal_sl_mult

        return optimal_data

    def _train_signal_models(self, signal_type: str, dataset: pd.DataFrame):
        """Train models for a specific signal type."""
        tp_model_name, sl_model_name = self._get_model_names(signal_type)

        # Prepare features
        features = self._prepare_features_for_training(dataset)

        # Train TP model
        if "optimal_tp_multiplier" in dataset.columns:
            y_tp = dataset["optimal_tp_multiplier"].values
            self._train_single_model(tp_model_name, features, y_tp)

        # Train SL model
        if "optimal_sl_multiplier" in dataset.columns:
            y_sl = dataset["optimal_sl_multiplier"].values
            self._train_single_model(sl_model_name, features, y_sl)

    def _prepare_features_for_training(self, dataset: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix for training."""
        # Extract same features as used in prediction
        feature_columns = [
            "ATR",
            "close",
            "high",
            "low",
            "volume",
            "ADX",
            "MACD_HIST",
            "rsi",
            "stoch_k",
            "bb_bandwidth",
            "volume_delta",
            "OBV",
            "CMF",
            "position_in_range",
        ]

        features = []
        for col in feature_columns:
            if col in dataset.columns:
                features.append(dataset[col].fillna(0).values)
            else:
                features.append(np.zeros(len(dataset)))

        return np.column_stack(features)

    def _train_single_model(self, model_name: str, X: np.ndarray, y: np.ndarray):
        """Train a single model for target prediction."""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.validation_split, random_state=42
            )

            # Scale features
            scaler = self.scalers[model_name]
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Create ensemble of models
            models = [
                LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
                RandomForestRegressor(n_estimators=50, random_state=42),
                LinearRegression(),
            ]

            best_model = None
            best_score = -np.inf

            for model in models:
                model.fit(X_train_scaled, y_train)
                score = model.score(X_test_scaled, y_test)

                if score > best_score:
                    best_score = score
                    best_model = model

            self.models[model_name] = best_model

            # Log training results
            y_pred = best_model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            self.logger.info(
                f"Trained {model_name}: R2={r2:.3f}, MSE={mse:.4f}, "
                f"Model: {type(best_model).__name__}"
            )

        except Exception as e:
            self.logger.error(f"Error training model {model_name}: {e}")

    def _calculate_prediction_confidence(
        self, tp_model_name: str, sl_model_name: str
    ) -> float:
        """Calculate confidence in the current prediction."""
        # This could be based on model performance metrics, ensemble agreement, etc.
        # For now, return a placeholder confidence
        tp_available = self.models.get(tp_model_name) is not None
        sl_available = self.models.get(sl_model_name) is not None

        if tp_available and sl_available:
            return 0.8
        elif tp_available or sl_available:
            return 0.5
        else:
            return 0.0

    def _should_skip_training(self) -> bool:
        """Check if training should be skipped based on last training time."""
        if self.last_training_time is None:
            return False

        time_since_training = datetime.now() - self.last_training_time
        return time_since_training.total_seconds() < (
            self.retrain_interval_hours * 3600
        )

    def _save_models(self):
        """Save all trained models and scalers."""
        try:
            for model_name, model in self.models.items():
                if model is not None:
                    model_path = os.path.join(
                        self.model_dir, f"{model_name}_model.joblib"
                    )
                    joblib.dump(model, model_path)

                    scaler_path = os.path.join(
                        self.model_dir, f"{model_name}_scaler.joblib"
                    )
                    joblib.dump(self.scalers[model_name], scaler_path)

            # Save metadata
            metadata = {
                "last_training_time": self.last_training_time,
                "model_names": list(self.models.keys()),
            }
            metadata_path = os.path.join(self.model_dir, "metadata.joblib")
            joblib.dump(metadata, metadata_path)

            self.logger.info(f"Saved ML target models to {self.model_dir}")

        except Exception as e:
            self.logger.error(f"Error saving models: {e}")

    def _load_models(self):
        """Load existing trained models and scalers."""
        try:
            metadata_path = os.path.join(self.model_dir, "metadata.joblib")
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.last_training_time = metadata.get("last_training_time")

                for model_name in self.models.keys():
                    model_path = os.path.join(
                        self.model_dir, f"{model_name}_model.joblib"
                    )
                    scaler_path = os.path.join(
                        self.model_dir, f"{model_name}_scaler.joblib"
                    )

                    if os.path.exists(model_path) and os.path.exists(scaler_path):
                        self.models[model_name] = joblib.load(model_path)
                        self.scalers[model_name] = joblib.load(scaler_path)

                self.logger.info(f"Loaded ML target models from {self.model_dir}")

        except Exception as e:
            self.logger.error(f"Error loading models: {e}")

    def get_model_status(self) -> Dict[str, Any]:
        """Get status information about the ML target models."""
        status = {
            "models_trained": sum(
                1 for model in self.models.values() if model is not None
            ),
            "total_models": len(self.models),
            "last_training_time": self.last_training_time,
            "model_details": {},
        }

        for name, model in self.models.items():
            status["model_details"][name] = {
                "trained": model is not None,
                "model_type": type(model).__name__ if model is not None else None,
            }

        return status
