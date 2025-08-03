# src/analyst/regime_classifier.py
import os  # For path manipulation
from datetime import datetime
from typing import Any

import joblib  # For saving/loading models
import numpy as np
import pandas as pd

from src.analyst.sr_analyzer import (
    SRLevelAnalyzer,
)  # Assuming sr_analyzer.py is accessible
from src.utils.error_handler import (
    handle_errors,
    handle_file_operations,
    handle_specific_errors,
)
from src.utils.logger import system_logger  # Centralized logger


class MarketRegimeClassifier:
    """
    Enhanced market regime classifier with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any], sr_analyzer: SRLevelAnalyzer) -> None:
        """
        Initialize market regime classifier with enhanced type safety.

        Args:
            config: Configuration dictionary
            sr_analyzer: Support/resistance analyzer component
        """
        self.config: dict[str, Any] = config
        self.sr_analyzer: SRLevelAnalyzer = sr_analyzer
        self.logger = system_logger.getChild("MarketRegimeClassifier")

        # Model state
        self.models: dict[str, Any] = {}
        self.is_trained: bool = False
        self.last_training_time: datetime | None = None
        self.model_performance: dict[str, float] = {}

        # Configuration
        self.checkpoint_dir: str = self.config.get("CHECKPOINT_DIR", "checkpoints")
        self.model_prefix: str = self.config.get(
            "REGIME_CLASSIFIER_MODEL_PREFIX",
            "regime_classifier_",
        )

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid regime classifier configuration"),
            AttributeError: (False, "Missing required classifier parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="regime classifier initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize regime classifier with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Market Regime Classifier...")

            # Load existing models if available
            await self._load_existing_models()

            # Initialize feature engineering
            await self._initialize_feature_engineering()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for regime classifier")
                return False

            self.logger.info(
                "✅ Market Regime Classifier initialization completed successfully",
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ Market Regime Classifier initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate regime classifier configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            required_keys = ["CHECKPOINT_DIR", "REGIME_CLASSIFIER_MODEL_PREFIX"]
            for key in required_keys:
                if key not in self.config:
                    self.logger.error(f"Missing required configuration key: {key}")
                    return False

            # Validate checkpoint directory
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                self.logger.info(f"Created checkpoint directory: {self.checkpoint_dir}")

            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_file_operations(
        default_return=None,
        context="model loading",
    )
    async def _load_existing_models(self) -> None:
        """Load existing trained models from checkpoint directory."""
        try:
            if not os.path.exists(self.checkpoint_dir):
                self.logger.info("No checkpoint directory found, will train new models")
                return

            # Look for existing model files
            model_files = [
                f
                for f in os.listdir(self.checkpoint_dir)
                if f.startswith(self.model_prefix) and f.endswith(".joblib")
            ]

            for model_file in model_files:
                model_path = os.path.join(self.checkpoint_dir, model_file)
                try:
                    model = joblib.load(model_path)
                    model_name = model_file.replace(self.model_prefix, "").replace(
                        ".joblib",
                        "",
                    )
                    self.models[model_name] = model
                    self.logger.info(f"Loaded existing model: {model_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to load model {model_file}: {e}")

            if self.models:
                self.is_trained = True
                self.logger.info(f"Loaded {len(self.models)} existing models")

        except Exception as e:
            self.logger.error(f"Error loading existing models: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="feature engineering initialization",
    )
    async def _initialize_feature_engineering(self) -> None:
        """Initialize feature engineering components."""
        try:
            # Initialize feature engineering parameters
            from src.config import get_lookback_window
            self.feature_params = {
                "lookback_window": get_lookback_window(),
                "volatility_window": self.config.get("volatility_window", 20),
                "momentum_window": self.config.get("momentum_window", 14),
            }

            self.logger.info(f"📊 Using lookback window: {self.feature_params['lookback_window']} days")
            self.logger.info("Feature engineering initialized")

        except Exception as e:
            self.logger.error(f"Error initializing feature engineering: {e}")

    @handle_specific_errors(
        error_handlers={
            ConnectionError: (None, "Failed to connect to data source"),
            TimeoutError: (None, "Training operation timed out"),
            ValueError: (None, "Invalid training data"),
        },
        default_return=None,
        context="regime classification training",
    )
    async def train_models(
        self,
        training_data: pd.DataFrame,
    ) -> dict[str, Any] | None:
        """
        Train regime classification models with enhanced error handling.

        Args:
            training_data: Training data DataFrame

        Returns:
            Optional[Dict[str, Any]]: Training results or None if failed
        """
        try:
            if training_data.empty:
                self.logger.error("Training data is empty")
                return None

            self.logger.info("Starting regime classification model training...")

            # Prepare features
            features = await self._prepare_features(training_data)
            if features is None:
                return None

            # Train multiple models
            training_results = {}

            # Train Random Forest
            rf_results = await self._train_random_forest(features)
            if rf_results:
                training_results["random_forest"] = rf_results

            # Train SVM
            svm_results = await self._train_svm(features)
            if svm_results:
                training_results["svm"] = svm_results

            # Train Neural Network
            nn_results = await self._train_neural_network(features)
            if nn_results:
                training_results["neural_network"] = nn_results

            # Save models
            await self._save_models()

            self.is_trained = True
            self.last_training_time = datetime.now()

            self.logger.info(
                "✅ Regime classification model training completed successfully",
            )
            return training_results

        except Exception as e:
            self.logger.error(f"Error training regime classification models: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="feature preparation",
    )
    async def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame | None:
        """
        Prepare features for regime classification.

        Args:
            data: Input data DataFrame

        Returns:
            Optional[pd.DataFrame]: Prepared features or None if failed
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

            # Calculate technical indicators
            features = data.copy()

            # Price-based features
            features["returns"] = data["close"].pct_change()
            features["log_returns"] = np.log(data["close"] / data["close"].shift(1))
            features["volatility"] = (
                features["returns"]
                .rolling(window=self.feature_params["volatility_window"])
                .std()
            )

            # Volume-based features
            features["volume_ma"] = data["volume"].rolling(window=20).mean()
            features["volume_ratio"] = data["volume"] / features["volume_ma"]

            # Momentum features
            features["rsi"] = self._calculate_rsi(data["close"])
            features["macd"] = self._calculate_macd(data["close"])

            # Support/Resistance features
            sr_features = await self._get_sr_features(data)
            if sr_features is not None:
                features = pd.concat([features, sr_features], axis=1)

            # Remove NaN values
            features = features.dropna()

            return features

        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
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
        context="SR features",
    )
    async def _get_sr_features(self, data: pd.DataFrame) -> pd.DataFrame | None:
        """
        Get support/resistance features.

        Args:
            data: Market data

        Returns:
            Optional[pd.DataFrame]: SR features or None
        """
        try:
            if self.sr_analyzer:
                sr_analysis = self.sr_analyzer.analyze(data)
                if sr_analysis:
                    # Extract SR features
                    sr_features = pd.DataFrame(index=data.index)
                    sr_features["sr_levels_count"] = len(sr_analysis.get("levels", []))
                    sr_features["price_to_sr_distance"] = 0.0  # Placeholder
                    return sr_features
            return None
        except Exception as e:
            self.logger.error(f"Error getting SR features: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="random forest training",
    )
    async def _train_random_forest(
        self,
        features: pd.DataFrame,
    ) -> dict[str, Any] | None:
        """
        Train Random Forest model.

        Args:
            features: Feature DataFrame

        Returns:
            Optional[Dict[str, Any]]: Training results
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split

            # Prepare target variable (regime labels)
            target = self._generate_regime_labels(features)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features.drop(["target"], axis=1, errors="ignore"),
                target,
                test_size=0.2,
                random_state=42,
            )

            # Train model
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)

            # Evaluate model
            train_score = rf_model.score(X_train, y_train)
            test_score = rf_model.score(X_test, y_test)

            # Store model
            self.models["random_forest"] = rf_model
            self.model_performance["random_forest"] = test_score

            return {
                "train_score": train_score,
                "test_score": test_score,
                "feature_importance": dict(
                    zip(X_train.columns, rf_model.feature_importances_, strict=False),
                ),
            }

        except Exception as e:
            self.logger.error(f"Error training Random Forest: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="SVM training",
    )
    async def _train_svm(self, features: pd.DataFrame) -> dict[str, Any] | None:
        """
        Train SVM model.

        Args:
            features: Feature DataFrame

        Returns:
            Optional[Dict[str, Any]]: Training results
        """
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.svm import SVC

            # Prepare target variable
            target = self._generate_regime_labels(features)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features.drop(["target"], axis=1, errors="ignore"),
                target,
                test_size=0.2,
                random_state=42,
            )

            # Train model
            svm_model = SVC(kernel="rbf", random_state=42)
            svm_model.fit(X_train, y_train)

            # Evaluate model
            train_score = svm_model.score(X_train, y_train)
            test_score = svm_model.score(X_test, y_test)

            # Store model
            self.models["svm"] = svm_model
            self.model_performance["svm"] = test_score

            return {"train_score": train_score, "test_score": test_score}

        except Exception as e:
            self.logger.error(f"Error training SVM: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="neural network training",
    )
    async def _train_neural_network(
        self,
        features: pd.DataFrame,
    ) -> dict[str, Any] | None:
        """
        Train Neural Network model.

        Args:
            features: Feature DataFrame

        Returns:
            Optional[Dict[str, Any]]: Training results
        """
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.neural_network import MLPClassifier

            # Prepare target variable
            target = self._generate_regime_labels(features)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features.drop(["target"], axis=1, errors="ignore"),
                target,
                test_size=0.2,
                random_state=42,
            )

            # Train model
            nn_model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=42,
                max_iter=500,
            )
            nn_model.fit(X_train, y_train)

            # Evaluate model
            train_score = nn_model.score(X_train, y_train)
            test_score = nn_model.score(X_test, y_test)

            # Store model
            self.models["neural_network"] = nn_model
            self.model_performance["neural_network"] = test_score

            return {"train_score": train_score, "test_score": test_score}

        except Exception as e:
            self.logger.error(f"Error training Neural Network: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="regime label generation",
    )
    def _generate_regime_labels(self, features: pd.DataFrame) -> pd.Series:
        """
        Generate regime labels based on market conditions.

        Args:
            features: Feature DataFrame

        Returns:
            pd.Series: Regime labels
        """
        try:
            # Simple regime classification based on volatility and returns
            volatility = features["volatility"]
            returns = features["returns"]

            # Define regime thresholds
            high_vol = volatility.quantile(0.75)
            low_vol = volatility.quantile(0.25)

            # Generate labels
            labels = pd.Series(index=features.index, data="normal")
            labels[(volatility > high_vol) & (returns > 0)] = "bull_volatile"
            labels[(volatility > high_vol) & (returns < 0)] = "bear_volatile"
            labels[(volatility < low_vol) & (returns > 0)] = "bull_calm"
            labels[(volatility < low_vol) & (returns < 0)] = "bear_calm"

            return labels

        except Exception as e:
            self.logger.error(f"Error generating regime labels: {e}")
            return pd.Series(index=features.index, data="normal")

    @handle_file_operations(
        default_return=None,
        context="model saving",
    )
    async def _save_models(self) -> None:
        """Save trained models to checkpoint directory."""
        try:
            for model_name, model in self.models.items():
                model_path = os.path.join(
                    self.checkpoint_dir,
                    f"{self.model_prefix}{model_name}.joblib",
                )
                joblib.dump(model, model_path)
                self.logger.info(f"Saved model: {model_name}")

        except Exception as e:
            self.logger.error(f"Error saving models: {e}")

    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid input data for classification"),
            AttributeError: (None, "Model not properly trained"),
        },
        default_return=None,
        context="regime classification",
    )
    async def classify_regime(self, data: pd.DataFrame) -> dict[str, Any] | None:
        """
        Classify market regime with enhanced error handling.

        Args:
            data: Market data DataFrame

        Returns:
            Optional[Dict[str, Any]]: Classification results or None if failed
        """
        try:
            if not self.is_trained or not self.models:
                self.logger.error("Models not trained")
                return None

            # Prepare features
            features = await self._prepare_features(data)
            if features is None:
                return None

            # Make predictions with all models
            predictions = {}
            probabilities = {}

            for model_name, model in self.models.items():
                try:
                    # Remove target column if present
                    X = features.drop(["target"], axis=1, errors="ignore")

                    # Make prediction
                    pred = model.predict(X.iloc[-1:])[0]
                    prob = (
                        model.predict_proba(X.iloc[-1:])[0]
                        if hasattr(model, "predict_proba")
                        else None
                    )

                    predictions[model_name] = pred
                    if prob is not None:
                        probabilities[model_name] = prob

                except Exception as e:
                    self.logger.warning(
                        f"Error making prediction with {model_name}: {e}",
                    )

            # Ensemble prediction (majority vote)
            if predictions:
                ensemble_pred = max(
                    set(predictions.values()),
                    key=list(predictions.values()).count,
                )

                result = {
                    "ensemble_prediction": ensemble_pred,
                    "individual_predictions": predictions,
                    "probabilities": probabilities,
                    "confidence": self._calculate_confidence(
                        predictions,
                        probabilities,
                    ),
                    "timestamp": datetime.now(),
                }

                return result

            return None

        except Exception as e:
            self.logger.error(f"Error classifying regime: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=0.0,
        context="confidence calculation",
    )
    def _calculate_confidence(
        self,
        predictions: dict[str, str],
        probabilities: dict[str, list[float]],
    ) -> float:
        """
        Calculate confidence in ensemble prediction.

        Args:
            predictions: Individual model predictions
            probabilities: Model prediction probabilities

        Returns:
            float: Confidence score
        """
        try:
            if not predictions:
                return 0.0

            # Calculate agreement among models
            unique_predictions = set(predictions.values())
            if len(unique_predictions) == 1:
                # All models agree
                base_confidence = 0.9
            else:
                # Models disagree
                base_confidence = 0.5

            # Adjust confidence based on probabilities if available
            if probabilities:
                avg_max_prob = np.mean([max(prob) for prob in probabilities.values()])
                final_confidence = (base_confidence + avg_max_prob) / 2
            else:
                final_confidence = base_confidence

            return min(1.0, max(0.0, final_confidence))

        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5

    def get_model_performance(self) -> dict[str, float]:
        """
        Get model performance metrics.

        Returns:
            Dict[str, float]: Model performance scores
        """
        return self.model_performance.copy()

    def is_model_trained(self) -> bool:
        """
        Check if models are trained.

        Returns:
            bool: True if models are trained, False otherwise
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
        context="regime classifier cleanup",
    )
    async def stop(self) -> None:
        """Stop the regime classifier component."""
        self.logger.info("🛑 Stopping Market Regime Classifier...")

        try:
            # Save models if not already saved
            if self.models and self.is_trained:
                await self._save_models()

            self.logger.info("✅ Market Regime Classifier stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping regime classifier: {e}")
