import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
import numpy as np

from src.utils.error_handler import (
    handle_errors,
    handle_file_operations,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class MLConfidencePredictor:
    """
    ML Confidence Predictor that generates predictions with confidence scores
    for price increases and expected price decreases in table format.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize ML Confidence Predictor with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("MLConfidencePredictor")

        # Model state
        self.model: Any | None = None
        self.is_trained: bool = False
        self.last_training_time: datetime | None = None
        self.model_performance: dict[str, float] = {}

        # Configuration
        self.predictor_config: dict[str, Any] = self.config.get(
            "ml_confidence_predictor",
            {},
        )
        self.model_path: str = self.predictor_config.get(
            "model_path",
            "models/confidence_predictor.joblib",
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

        # Confidence score levels for price increases
        self.price_increase_levels: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        # Expected price decrease levels (before the increase)
        self.price_decrease_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid ML confidence predictor configuration"),
            AttributeError: (False, "Missing required predictor parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="ML confidence predictor initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize ML Confidence Predictor with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing ML Confidence Predictor...")

            # Load predictor configuration
            await self._load_predictor_configuration()

            # Initialize model parameters
            await self._initialize_model_parameters()

            # Load existing model if available
            await self._load_existing_model()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error(
                    "Invalid configuration for ML confidence predictor",
                )
                return False

            self.logger.info(
                "✅ ML Confidence Predictor initialization completed successfully",
            )
            return True

        except Exception as e:
            self.logger.error(
                f"❌ ML Confidence Predictor initialization failed: {e}",
            )
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="predictor configuration loading",
    )
    async def _load_predictor_configuration(self) -> None:
        """Load predictor configuration."""
        # Set default predictor parameters
        self.predictor_config.setdefault("model_path", "models/confidence_predictor.joblib")
        self.predictor_config.setdefault("retrain_interval_hours", 24)
        self.predictor_config.setdefault("min_samples_for_training", 500)
        self.predictor_config.setdefault("confidence_threshold", 0.6)
        self.predictor_config.setdefault("max_prediction_horizon", 24)  # hours

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="model parameters initialization",
    )
    async def _initialize_model_parameters(self) -> None:
        """Initialize model parameters."""
        # Ensure model directory exists
        model_dir = os.path.dirname(self.model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        # Initialize performance metrics
        self.model_performance = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

    @handle_file_operations(
        default_return=None,
        context="model loading",
    )
    async def _load_existing_model(self) -> None:
        """Load existing model if available."""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.is_trained = True
                self.logger.info("✅ Loaded existing confidence predictor model")
            except Exception as e:
                self.logger.warning(f"Failed to load existing model: {e}")
                self.model = None
                self.is_trained = False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate predictor configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate required parameters
            required_params = [
                "model_path",
                "retrain_interval_hours",
                "min_samples_for_training",
            ]
            
            for param in required_params:
                if param not in self.predictor_config:
                    self.logger.error(f"Missing required parameter: {param}")
                    return False

            # Validate parameter values
            if self.predictor_config["retrain_interval_hours"] <= 0:
                self.logger.error("Retrain interval must be positive")
                return False

            if self.predictor_config["min_samples_for_training"] < 100:
                self.logger.error("Minimum samples for training must be at least 100")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
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
        Train the confidence predictor model.

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

            self.logger.info("Training confidence predictor model...")

            # Prepare training data
            prepared_data = await self._prepare_training_data(training_data)
            if prepared_data is None:
                return None

            features, targets = prepared_data

            # Train model
            training_result = await self._train_model_internal(features, targets)
            if training_result is None:
                return None

            # Save model
            await self._save_model()

            self.last_training_time = datetime.now()
            self.is_trained = True

            self.logger.info("✅ Confidence predictor model training completed")
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
        Prepare training data for confidence prediction.

        Args:
            data: Raw market data

        Returns:
            Optional[Tuple[pd.DataFrame, pd.DataFrame]]: Features and targets or None
        """
        try:
            # Create features DataFrame
            features = data.copy()

            # Add technical indicators
            features["returns"] = data["close"].pct_change()
            features["volatility"] = features["returns"].rolling(window=20).std()
            features["rsi"] = self._calculate_rsi(data["close"])
            features["macd"] = self._calculate_macd(data["close"])
            features["bollinger_upper"] = self._calculate_bollinger_bands(data["close"])[0]
            features["bollinger_lower"] = self._calculate_bollinger_bands(data["close"])[1]

            # Add price-based features
            features["price_position"] = (data["close"] - data["low"]) / (
                data["high"] - data["low"]
            )
            features["volume_ratio"] = (
                data["volume"] / data["volume"].rolling(window=20).mean()
            )

            # Add momentum features
            features["momentum_5"] = data["close"].pct_change(5)
            features["momentum_10"] = data["close"].pct_change(10)
            features["momentum_20"] = data["close"].pct_change(20)

            # Create targets for confidence prediction
            targets = pd.DataFrame()
            
            # Generate confidence scores for each price increase level
            for increase_level in self.price_increase_levels:
                # Calculate if price increased by this level within prediction horizon
                future_returns = data["close"].shift(-self.predictor_config["max_prediction_horizon"]) / data["close"] - 1
                confidence_target = (future_returns >= increase_level / 100).astype(float)
                targets[f"confidence_{increase_level}"] = confidence_target

            # Generate expected price decrease targets
            for decrease_level in self.price_decrease_levels:
                # Calculate if price decreased by this level before increase
                future_returns = data["close"].shift(-self.predictor_config["max_prediction_horizon"]) / data["close"] - 1
                decrease_target = (future_returns <= -decrease_level / 100).astype(float)
                targets[f"expected_decrease_{decrease_level}"] = decrease_target

            # Remove rows with NaN values
            features = features.dropna()
            targets = targets.dropna()

            # Align features and targets
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
        """Calculate RSI indicator."""
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
        """Calculate MACD indicator."""
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
        context="Bollinger Bands calculation",
    )
    def _calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: int = 2,
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            return upper_band, lower_band
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")
            return pd.Series(index=prices.index), pd.Series(index=prices.index)

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
        Train the internal model.

        Args:
            features: Feature DataFrame
            targets: Target DataFrame

        Returns:
            Optional[Dict[str, Any]]: Training results or None
        """
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.2, random_state=42
            )

            # Initialize models for each target
            models = {}
            performance_metrics = {}

            for target_col in targets.columns:
                # Create model for this target
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1,
                )

                # Train model
                model.fit(X_train, y_train[target_col])

                # Evaluate model
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test[target_col], y_pred)
                r2 = r2_score(y_test[target_col], y_pred)

                models[target_col] = model
                performance_metrics[target_col] = {
                    "mse": mse,
                    "r2": r2,
                }

            # Store models
            self.model = models
            self.model_performance = performance_metrics

            # Calculate overall performance
            avg_r2 = np.mean([metrics["r2"] for metrics in performance_metrics.values()])
            avg_mse = np.mean([metrics["mse"] for metrics in performance_metrics.values()])

            training_result = {
                "models_trained": len(models),
                "avg_r2_score": avg_r2,
                "avg_mse": avg_mse,
                "performance_metrics": performance_metrics,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
            }

            return training_result

        except Exception as e:
            self.logger.error(f"Error in internal model training: {e}")
            return None

    @handle_file_operations(
        default_return=None,
        context="model saving",
    )
    async def _save_model(self) -> None:
        """Save the trained model."""
        if self.model:
            joblib.dump(self.model, self.model_path)
            self.logger.info("✅ Model saved successfully")

    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid input data for prediction"),
            AttributeError: (None, "Model not properly trained"),
        },
        default_return=None,
        context="confidence prediction",
    )
    async def predict_confidence_table(
        self,
        market_data: pd.DataFrame,
        current_price: float,
    ) -> dict[str, Any] | None:
        """
        Predict confidence scores and expected price decreases in table format.

        Args:
            market_data: Recent market data
            current_price: Current market price

        Returns:
            Optional[Dict[str, Any]]: Prediction table or None if failed
        """
        try:
            if not self.is_trained or not self.model:
                self.logger.error("Model not trained")
                return None

            if market_data.empty:
                self.logger.error("Empty market data provided")
                return None

            self.logger.info("Generating confidence prediction table...")

            # Prepare features
            features = await self._prepare_prediction_features(market_data)
            if features is None:
                return None

            # Make predictions for all confidence levels
            confidence_predictions = {}
            expected_decrease_predictions = {}

            # Predict confidence scores for price increases
            for increase_level in self.price_increase_levels:
                target_col = f"confidence_{increase_level}"
                if target_col in self.model:
                    confidence = self.model[target_col].predict(features.iloc[-1:])[0]
                    confidence_predictions[increase_level] = max(0.0, min(1.0, confidence))

            # Predict expected price decreases
            for decrease_level in self.price_decrease_levels:
                target_col = f"expected_decrease_{decrease_level}"
                if target_col in self.model:
                    decrease_prob = self.model[target_col].predict(features.iloc[-1:])[0]
                    expected_decrease_predictions[decrease_level] = max(0.0, min(1.0, decrease_prob))

            # Create prediction table
            prediction_table = {
                "timestamp": datetime.now(),
                "current_price": current_price,
                "confidence_scores": confidence_predictions,
                "expected_decreases": expected_decrease_predictions,
                "model_performance": self.model_performance,
            }

            self.logger.info("✅ Confidence prediction table generated successfully")
            return prediction_table

        except Exception as e:
            self.logger.error(f"Error generating confidence prediction table: {e}")
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
            features["bollinger_upper"] = self._calculate_bollinger_bands(data["close"])[0]
            features["bollinger_lower"] = self._calculate_bollinger_bands(data["close"])[1]

            # Add price-based features
            features["price_position"] = (data["close"] - data["low"]) / (
                data["high"] - data["low"]
            )
            features["volume_ratio"] = (
                data["volume"] / data["volume"].rolling(window=20).mean()
            )

            # Add momentum features
            features["momentum_5"] = data["close"].pct_change(5)
            features["momentum_10"] = data["close"].pct_change(10)
            features["momentum_20"] = data["close"].pct_change(20)

            # Remove rows with NaN values
            features = features.dropna()

            return features

        except Exception as e:
            self.logger.error(f"Error preparing prediction features: {e}")
            return None

    def get_model_performance(self) -> dict[str, float]:
        """Get model performance metrics."""
        return self.model_performance

    def is_model_trained(self) -> bool:
        """Check if model is trained."""
        return self.is_trained

    def get_last_training_time(self) -> datetime | None:
        """Get last training time."""
        return self.last_training_time

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="ML confidence predictor cleanup",
    )
    async def stop(self) -> None:
        """Clean up resources."""
        try:
            self.logger.info("Stopping ML Confidence Predictor...")
            # Cleanup code here if needed
            self.logger.info("✅ ML Confidence Predictor stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping ML Confidence Predictor: {e}")


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="ML confidence predictor setup",
)
async def setup_ml_confidence_predictor(
    config: dict[str, Any] | None = None,
) -> MLConfidencePredictor | None:
    """
    Setup ML Confidence Predictor.

    Args:
        config: Configuration dictionary

    Returns:
        Optional[MLConfidencePredictor]: Initialized predictor or None
    """
    try:
        if config is None:
            config = {}

        predictor = MLConfidencePredictor(config)
        if await predictor.initialize():
            return predictor
        else:
            return None

    except Exception as e:
        system_logger.error(f"Error setting up ML Confidence Predictor: {e}")
        return None
