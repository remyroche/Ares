import os
from datetime import datetime
from typing import Any

import joblib
import pandas as pd

from src.utils.error_handler import (
    handle_errors,
    handle_file_operations,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class MLDynamicTargetPredictor:
    """
    Enhanced ML dynamic target predictor with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize ML dynamic target predictor with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("MLDynamicTargetPredictor")

        # Model state
        self.model: Any | None = None
        self.is_trained: bool = False
        self.last_training_time: datetime | None = None
        self.model_performance: dict[str, float] = {}

        # Configuration
        self.predictor_config: dict[str, Any] = self.config.get(
            "ml_dynamic_target_predictor",
            {},
        )
        self.model_path: str = self.predictor_config.get(
            "model_path",
            "models/dynamic_target_predictor.joblib",
        )
        self.retrain_interval_hours: int = self.predictor_config.get(
            "retrain_interval_hours",
            24,
        )
        # Import the centralized lookback window function
        from src.config import get_lookback_window
        self.lookback_window: int = get_lookback_window()
        self.min_samples_for_training: int = self.predictor_config.get(
            "min_samples_for_training",
            500,
        )

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid ML dynamic target predictor configuration"),
            AttributeError: (False, "Missing required predictor parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="ML dynamic target predictor initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize ML dynamic target predictor with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing ML Dynamic Target Predictor...")

            # Load predictor configuration
            await self._load_predictor_configuration()

            # Initialize model parameters
            await self._initialize_model_parameters()

            # Load existing model if available
            await self._load_existing_model()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error(
                    "Invalid configuration for ML dynamic target predictor",
                )
                return False

            self.logger.info(
                "✅ ML Dynamic Target Predictor initialization completed successfully",
            )
            return True

        except Exception as e:
            self.logger.error(
                f"❌ ML Dynamic Target Predictor initialization failed: {e}",
            )
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="predictor configuration loading",
    )
    async def _load_predictor_configuration(self) -> None:
        """Load predictor configuration."""
        try:
            # Set default predictor parameters
            self.predictor_config.setdefault(
                "model_path",
                "models/dynamic_target_predictor.joblib",
            )
            self.predictor_config.setdefault("retrain_interval_hours", 24)
            self.predictor_config.setdefault("min_samples_for_training", 500)
            self.predictor_config.setdefault("validation_split", 0.2)
            self.predictor_config.setdefault("min_tp_multiplier", 0.5)
            self.predictor_config.setdefault("max_tp_multiplier", 6.0)
            self.predictor_config.setdefault("min_sl_multiplier", 0.2)
            self.predictor_config.setdefault("max_sl_multiplier", 2.0)

            # Use centralized lookback window
            from src.config import get_lookback_window
            lookback_days = get_lookback_window()
            self.logger.info(f"📊 Using lookback window: {lookback_days} days")

            self.logger.info("Predictor configuration loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading predictor configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="model parameters initialization",
    )
    async def _initialize_model_parameters(self) -> None:
        """Initialize model parameters."""
        try:
            # Initialize model parameters
            self.model_path = self.predictor_config["model_path"]
            self.retrain_interval_hours = self.predictor_config[
                "retrain_interval_hours"
            ]
            # Use centralized lookback window
            from src.config import get_lookback_window
            self.lookback_window = get_lookback_window()
            self.min_samples_for_training = self.predictor_config[
                "min_samples_for_training"
            ]

            self.logger.info("Model parameters initialized")

        except Exception as e:
            self.logger.error(f"Error initializing model parameters: {e}")

    @handle_file_operations(
        default_return=None,
        context="model loading",
    )
    async def _load_existing_model(self) -> None:
        """Load existing trained model."""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.is_trained = True
                self.logger.info(f"Loaded existing model from: {self.model_path}")
            else:
                self.logger.info("No existing model found, will train new model")

        except Exception as e:
            self.logger.error(f"Error loading existing model: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate ML dynamic target predictor configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            required_keys = [
                "model_path",
                "retrain_interval_hours",
                "min_samples_for_training",
            ]
            for key in required_keys:
                if key not in self.predictor_config:
                    self.logger.error(
                        f"Missing required predictor configuration key: {key}",
                    )
                    return False

            # Validate parameter ranges
            if self.retrain_interval_hours < 1:
                self.logger.error("retrain_interval_hours must be at least 1")
                return False

            # Validate lookback window (now handled centrally)
            from src.config import get_lookback_window
            lookback_days = get_lookback_window()
            if lookback_days < 50:
                self.logger.error("lookback_window must be at least 50")
                return False

            if self.min_samples_for_training < 100:
                self.logger.error("min_samples_for_training must be at least 100")
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_specific_errors(
        error_handlers={
            ConnectionError: (None, "Failed to connect to data source"),
            TimeoutError: (None, "Model training timed out"),
            ValueError: (None, "Invalid training data"),
        },
        default_return=None,
        context="model training",
    )
    async def train_model(
        self,
        training_data: pd.DataFrame,
    ) -> dict[str, Any] | None:
        """
        Train the dynamic target prediction model with enhanced error handling.

        Args:
            training_data: Training data DataFrame

        Returns:
            Optional[Dict[str, Any]]: Training results or None if failed
        """
        try:
            if training_data.empty:
                self.logger.error("Empty training data provided")
                return None

            if len(training_data) < self.min_samples_for_training:
                self.logger.error(
                    f"Insufficient training data: {len(training_data)} < {self.min_samples_for_training}",
                )
                return None

            self.logger.info("Starting model training...")

            # Prepare features and targets
            features, targets = await self._prepare_training_data(training_data)
            if features is None or targets is None:
                return None

            # Train model
            training_result = await self._train_model_internal(features, targets)
            if not training_result:
                return None

            # Save model
            await self._save_model()

            # Update state
            self.is_trained = True
            self.last_training_time = datetime.now()

            self.logger.info("✅ Model training completed successfully")
            return training_result

        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="training data preparation",
    )
    async def _prepare_training_data(
        self,
        data: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        """
        Prepare features and targets for training.

        Args:
            data: Raw training data

        Returns:
            Optional[Tuple[pd.DataFrame, pd.DataFrame]]: Features and targets or None
        """
        try:
            # Ensure required columns exist
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return None

            # Create features
            features = data.copy()

            # Add technical indicators
            features["returns"] = data["close"].pct_change()
            features["volatility"] = features["returns"].rolling(window=20).std()
            features["rsi"] = self._calculate_rsi(data["close"])
            features["macd"] = self._calculate_macd(data["close"])

            # Add price-based features
            features["price_position"] = (data["close"] - data["low"]) / (
                data["high"] - data["low"]
            )
            features["volume_ratio"] = (
                data["volume"] / data["volume"].rolling(window=20).mean()
            )

            # Create targets (TP/SL ratios)
            targets = pd.DataFrame()
            targets["tp_ratio"] = self._calculate_tp_ratios(data)
            targets["sl_ratio"] = self._calculate_sl_ratios(data)

            # Remove NaN values
            features = features.dropna()
            targets = targets.dropna()

            # Align indices
            common_index = features.index.intersection(targets.index)
            features = features.loc[common_index]
            targets = targets.loc[common_index]

            return features, targets

        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="RSI calculation",
    )
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI indicator.

        Args:
            prices: Price series
            period: RSI period

        Returns:
            pd.Series: RSI values
        """
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return pd.Series(index=prices.index)

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="MACD calculation",
    )
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.Series:
        """
        Calculate MACD indicator.

        Args:
            prices: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            pd.Series: MACD values
        """
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()
            return macd - signal_line
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return pd.Series(index=prices.index)

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="TP ratio calculation",
    )
    def _calculate_tp_ratios(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate take profit ratios based on future price movements.

        Args:
            data: Market data

        Returns:
            pd.Series: TP ratios
        """
        try:
            # Look ahead for optimal TP levels
            future_highs = data["high"].shift(-1).rolling(window=5).max()
            current_price = data["close"]
            tp_ratios = (future_highs - current_price) / current_price

            # Apply bounds
            min_tp = self.predictor_config["min_tp_multiplier"]
            max_tp = self.predictor_config["max_tp_multiplier"]
            tp_ratios = tp_ratios.clip(min_tp, max_tp)

            return tp_ratios

        except Exception as e:
            self.logger.error(f"Error calculating TP ratios: {e}")
            return pd.Series(index=data.index)

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="SL ratio calculation",
    )
    def _calculate_sl_ratios(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate stop loss ratios based on future price movements.

        Args:
            data: Market data

        Returns:
            pd.Series: SL ratios
        """
        try:
            # Look ahead for optimal SL levels
            future_lows = data["low"].shift(-1).rolling(window=5).min()
            current_price = data["close"]
            sl_ratios = (current_price - future_lows) / current_price

            # Apply bounds
            min_sl = self.predictor_config["min_sl_multiplier"]
            max_sl = self.predictor_config["max_sl_multiplier"]
            sl_ratios = sl_ratios.clip(min_sl, max_sl)

            return sl_ratios

        except Exception as e:
            self.logger.error(f"Error calculating SL ratios: {e}")
            return pd.Series(index=data.index)

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="internal model training",
    )
    async def _train_model_internal(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
    ) -> dict[str, Any] | None:
        """
        Train the model internally.

        Args:
            features: Feature DataFrame
            targets: Target DataFrame

        Returns:
            Optional[Dict[str, Any]]: Training results or None
        """
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error, r2_score
            from sklearn.model_selection import train_test_split

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features,
                targets,
                test_size=0.2,
                random_state=42,
            )

            # Train TP model
            tp_model = RandomForestRegressor(n_estimators=100, random_state=42)
            tp_model.fit(X_train, y_train["tp_ratio"])
            tp_pred = tp_model.predict(X_test)
            tp_mse = mean_squared_error(y_test["tp_ratio"], tp_pred)
            tp_r2 = r2_score(y_test["tp_ratio"], tp_pred)

            # Train SL model
            sl_model = RandomForestRegressor(n_estimators=100, random_state=42)
            sl_model.fit(X_train, y_train["sl_ratio"])
            sl_pred = sl_model.predict(X_test)
            sl_mse = mean_squared_error(y_test["sl_ratio"], sl_pred)
            sl_r2 = r2_score(y_test["sl_ratio"], sl_pred)

            # Store models
            self.model = {"tp_model": tp_model, "sl_model": sl_model}

            # Store performance metrics
            self.model_performance = {
                "tp_mse": tp_mse,
                "tp_r2": tp_r2,
                "sl_mse": sl_mse,
                "sl_r2": sl_r2,
            }

            return {
                "tp_mse": tp_mse,
                "tp_r2": tp_r2,
                "sl_mse": sl_mse,
                "sl_r2": sl_r2,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
            }

        except Exception as e:
            self.logger.error(f"Error training model internally: {e}")
            return None

    @handle_file_operations(
        default_return=None,
        context="model saving",
    )
    async def _save_model(self) -> None:
        """Save trained model to file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

            # Save model
            joblib.dump(self.model, self.model_path)
            self.logger.info(f"Model saved to: {self.model_path}")

        except Exception as e:
            self.logger.error(f"Error saving model: {e}")

    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid input data for prediction"),
            AttributeError: (None, "Model not properly trained"),
        },
        default_return=None,
        context="target prediction",
    )
    async def predict_targets(
        self,
        market_data: pd.DataFrame,
        current_price: float,
    ) -> dict[str, Any] | None:
        """
        Predict dynamic targets with enhanced error handling.

        Args:
            market_data: Recent market data
            current_price: Current market price

        Returns:
            Optional[Dict[str, Any]]: Predicted targets or None if failed
        """
        try:
            if not self.is_trained or not self.model:
                self.logger.error("Model not trained")
                return None

            if market_data.empty:
                self.logger.error("Empty market data provided")
                return None

            self.logger.info("Predicting dynamic targets...")

            # Prepare features
            features = await self._prepare_prediction_features(market_data)
            if features is None:
                return None

            # Make predictions
            tp_ratio = self.model["tp_model"].predict(features.iloc[-1:])[0]
            sl_ratio = self.model["sl_model"].predict(features.iloc[-1:])[0]

            # Apply bounds
            tp_ratio = max(
                self.predictor_config["min_tp_multiplier"],
                min(tp_ratio, self.predictor_config["max_tp_multiplier"]),
            )
            sl_ratio = max(
                self.predictor_config["min_sl_multiplier"],
                min(sl_ratio, self.predictor_config["max_sl_multiplier"]),
            )

            # Calculate target prices
            tp_price = current_price * (1 + tp_ratio)
            sl_price = current_price * (1 - sl_ratio)

            prediction_result = {
                "take_profit_price": tp_price,
                "stop_loss_price": sl_price,
                "tp_ratio": tp_ratio,
                "sl_ratio": sl_ratio,
                "current_price": current_price,
                "prediction_time": datetime.now(),
                "model_performance": self.model_performance,
            }

            self.logger.info("✅ Dynamic targets predicted successfully")
            return prediction_result

        except Exception as e:
            self.logger.error(f"Error predicting targets: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="prediction features preparation",
    )
    async def _prepare_prediction_features(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """
        Prepare features for prediction.

        Args:
            data: Market data

        Returns:
            Optional[pd.DataFrame]: Prepared features or None
        """
        try:
            # Use same feature preparation as training
            features = data.copy()

            # Add technical indicators
            features["returns"] = data["close"].pct_change()
            features["volatility"] = features["returns"].rolling(window=20).std()
            features["rsi"] = self._calculate_rsi(data["close"])
            features["macd"] = self._calculate_macd(data["close"])

            # Add price-based features
            features["price_position"] = (data["close"] - data["low"]) / (
                data["high"] - data["low"]
            )
            features["volume_ratio"] = (
                data["volume"] / data["volume"].rolling(window=20).mean()
            )

            # Remove NaN values
            features = features.dropna()

            return features

        except Exception as e:
            self.logger.error(f"Error preparing prediction features: {e}")
            return None

    def get_model_performance(self) -> dict[str, float]:
        """
        Get model performance metrics.

        Returns:
            Dict[str, float]: Model performance scores
        """
        return self.model_performance.copy()

    def is_model_trained(self) -> bool:
        """
        Check if model is trained.

        Returns:
            bool: True if model is trained, False otherwise
        """
        return self.is_trained

    def get_last_training_time(self) -> datetime | None:
        """
        Get last training time.

        Returns:
            Optional[datetime]: Last training time or None
        """
        return self.last_training_time

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="ML dynamic target predictor cleanup",
    )
    async def stop(self) -> None:
        """Stop the ML dynamic target predictor component."""
        self.logger.info("🛑 Stopping ML Dynamic Target Predictor...")

        try:
            # Save model if trained
            if self.is_trained and self.model:
                await self._save_model()

            self.logger.info("✅ ML Dynamic Target Predictor stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping ML dynamic target predictor: {e}")
