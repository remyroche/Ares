from __future__ import annotations

"""
Enhanced Scenario-Based Predictor for Tactician

Implements advanced probabilistic scenario analysis with:
- All step07 technical indicators
- 15-minute look-ahead period
- Fractal scenario definitions (linear progression)
- Full step17 optimization for all parameters
- Complete migration from existing system
"""

import logging
from datetime import datetime
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import talib
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

# Simple logger setup
logger = logging.getLogger(__name__)


# Simple error handling decorator
def handle_errors(func):
    """Simple error handling decorator."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Error in {func.__name__}: {e}")
            return None

    return wrapper


class EnhancedScenarioBasedPredictor:
    """
    Enhanced scenario-based predictor with fractal scenarios and comprehensive technical indicators.

    Fractal Scenarios (Linear Progression):
    - Profit Zones: 0.25%, 0.5%, 0.75%, 1.0%, 1.25%, 1.5%, 1.75%, 2.0%
    - Risk Zones: -0.25%, -0.5%, -0.75%, -1.0%, -1.25%, -1.5%, -1.75%, -2.0%
    - Neutral: No scenario triggered within 15 minutes
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize enhanced scenario-based predictor.

        Args:
            config: Configuration dictionary with step17 optimization parameters
        """
        self.config = config
        self.logger = logger

        # Load step17 optimization parameters
        step17_config = config.get("step17_optimization", {})
        scenario_config = step17_config.get("enhanced_scenario_analysis", {})

        # Fractal scenario definitions (configurable for step17)
        self.scenarios = self._create_fractal_scenarios(scenario_config)

        # Time limit for scenario evaluation (15 minutes)
        self.time_limit_minutes = scenario_config.get("time_limit_minutes", 15)

        # Model configuration (configurable for step17)
        self.model_config = {
            "n_estimators": scenario_config.get("n_estimators", 200),
            "learning_rate": scenario_config.get("learning_rate", 0.05),
            "max_depth": scenario_config.get("max_depth", 8),
            "num_leaves": scenario_config.get("num_leaves", 63),
            "subsample": scenario_config.get("subsample", 0.8),
            "colsample_bytree": scenario_config.get("colsample_bytree", 0.8),
            "random_state": scenario_config.get("random_state", 42),
            "verbose": -1,
        }

        # Enhanced decision thresholds (configurable for step17)
        self.decision_thresholds = {
            "profit_zone_combined": scenario_config.get(
                "profit_zone_combined_threshold", 0.6
            ),
            "risk_zone_combined": scenario_config.get(
                "risk_zone_combined_threshold", 0.2
            ),
            "exit_risk_threshold": scenario_config.get("exit_risk_threshold", 0.5),
            "neutral_threshold": scenario_config.get("neutral_threshold", 0.3),
            "confidence_threshold": scenario_config.get("confidence_threshold", 0.7),
            "profit_risk_ratio": scenario_config.get(
                "profit_risk_ratio_threshold", 2.0
            ),
            "scenario_dominance": scenario_config.get(
                "scenario_dominance_threshold", 0.4
            ),
        }

        # Step7 technical indicator parameters (configurable for step17)
        self.technical_indicators = {
            "RSI": {
                "lookback_period": scenario_config.get("rsi_lookback_period", 14),
                "overbought_threshold": scenario_config.get(
                    "rsi_overbought_threshold", 70
                ),
                "oversold_threshold": scenario_config.get("rsi_oversold_threshold", 30),
            },
            "MACD": {
                "fast_period": scenario_config.get("macd_fast_period", 12),
                "slow_period": scenario_config.get("macd_slow_period", 26),
                "signal_period": scenario_config.get("macd_signal_period", 9),
            },
            "Bollinger_Bands": {
                "lookback_period": scenario_config.get("bb_lookback_period", 20),
                "std_dev": scenario_config.get("bb_std_dev", 2.0),
                "squeeze_threshold": scenario_config.get("bb_squeeze_threshold", 0.2),
            },
            "SMA": {
                "short_period": scenario_config.get("sma_short_period", 10),
                "long_period": scenario_config.get("sma_long_period", 30),
            },
            "EMA": {
                "short_period": scenario_config.get("ema_short_period", 10),
                "long_period": scenario_config.get("ema_long_period", 30),
            },
            "ATR": {
                "lookback_period": scenario_config.get("atr_lookback_period", 14),
            },
            "Stochastic": {
                "k_period": scenario_config.get("stoch_k_period", 14),
                "d_period": scenario_config.get("stoch_d_period", 3),
                "overbought": scenario_config.get("stoch_overbought", 80),
                "oversold": scenario_config.get("stoch_oversold", 20),
            },
            "ADX": {
                "lookback_period": scenario_config.get("adx_lookback_period", 14),
                "threshold": scenario_config.get("adx_threshold", 25),
            },
            "CCI": {
                "lookback_period": scenario_config.get("cci_lookback_period", 14),
                "constant": scenario_config.get("cci_constant", 0.015),
            },
        }

        # Feature engineering parameters (configurable for step17)
        self.feature_config = {
            "lookback_periods": scenario_config.get("lookback_periods", 20),
            "volatility_window": scenario_config.get("volatility_window", 20),
            "volume_ma_period": scenario_config.get("volume_ma_period", 10),
            "price_momentum_periods": scenario_config.get(
                "price_momentum_periods", [5, 10, 20]
            ),
            "volatility_periods": scenario_config.get(
                "volatility_periods", [5, 10, 20]
            ),
        }

        # Model state
        self.model = None
        self.is_trained = False
        self.last_training_time: datetime | None = None
        self.feature_importance: dict[str, float] = {}
        self.model_performance: dict[str, float] = {}

    def _create_fractal_scenarios(
        self, scenario_config: dict[str, Any]
    ) -> dict[int, dict[str, Any]]:
        """
        Create fractal scenarios with linear progression.

        Args:
            scenario_config: Scenario configuration

        Returns:
            dict: Fractal scenario definitions
        """
        scenarios = {}
        scenario_id = 0

        # Profit zones (0.25% to 2.0% in 0.25% increments)
        profit_targets = [0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02]
        for i, profit_target in enumerate(profit_targets):
            scenarios[scenario_id] = {
                "name": f"Profit Zone {i+1} ({profit_target*100:.1f}%)",
                "profit_target": scenario_config.get(
                    f"profit_zone_{i+1}_target", profit_target
                ),
                "stop_loss": scenario_config.get(
                    f"profit_zone_{i+1}_stop_loss", -0.005
                ),
                "description": f"Price moves up by {profit_target*100:.1f}% before moving down by 0.5%",
                "zone_type": "profit",
                "zone_level": i + 1,
            }
            scenario_id += 1

        # Risk zones (-0.25% to -2.0% in 0.25% increments)
        risk_targets = [
            -0.0025,
            -0.005,
            -0.0075,
            -0.01,
            -0.0125,
            -0.015,
            -0.0175,
            -0.02,
        ]
        for i, risk_target in enumerate(risk_targets):
            scenarios[scenario_id] = {
                "name": f"Risk Zone {i+1} ({abs(risk_target)*100:.1f}%)",
                "profit_target": scenario_config.get(f"risk_zone_{i+1}_target", 0.005),
                "stop_loss": scenario_config.get(
                    f"risk_zone_{i+1}_stop_loss", risk_target
                ),
                "description": f"Price moves down by {abs(risk_target)*100:.1f}% before moving up by 0.5%",
                "zone_type": "risk",
                "zone_level": i + 1,
            }
            scenario_id += 1

        # Neutral scenario
        scenarios[scenario_id] = {
            "name": "Neutral",
            "profit_target": scenario_config.get("neutral_target", 0.0),
            "stop_loss": scenario_config.get("neutral_stop_loss", 0.0),
            "description": "No scenario triggered within time limit",
            "zone_type": "neutral",
            "zone_level": 0,
        }

        return scenarios

    async def initialize(self) -> bool:
        """
        Initialize enhanced scenario-based predictor.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Enhanced Scenario-Based Predictor...")

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error(
                    "Invalid configuration for enhanced scenario predictor"
                )
                return False

            # Initialize model
            self.model = lgb.LGBMClassifier(**self.model_config)

            self.logger.info(
                "✅ Enhanced Scenario-Based Predictor initialized successfully"
            )
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Enhanced Scenario-Based Predictor initialization failed: {e}"
            )
            return False

    def _validate_configuration(self) -> bool:
        """
        Validate enhanced scenario predictor configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate scenarios
            for scenario_id, scenario in self.scenarios.items():
                if scenario["zone_type"] != "neutral":
                    if (
                        scenario["profit_target"] <= 0
                        and scenario["zone_type"] == "profit"
                    ):
                        self.logger.error(
                            f"Invalid profit target for scenario {scenario_id}"
                        )
                        return False

                    if scenario["stop_loss"] >= 0 and scenario["zone_type"] == "risk":
                        self.logger.error(
                            f"Invalid stop loss for scenario {scenario_id}"
                        )
                        return False

            # Validate time limit
            if self.time_limit_minutes <= 0:
                self.logger.error("Invalid time limit")
                return False

            # Validate thresholds
            for threshold_name, threshold in self.decision_thresholds.items():
                if threshold < 0 or threshold > 1:
                    self.logger.error(f"Invalid threshold for {threshold_name}")
                    return False

            # Validate technical indicator parameters
            for indicator_name, params in self.technical_indicators.items():
                for param_name, param_value in params.items():
                    if param_value <= 0:
                        self.logger.error(
                            f"Invalid parameter for {indicator_name}.{param_name}"
                        )
                        return False

            return True

        except Exception as e:
            self.logger.exception(f"❌ Configuration validation failed: {e}")
            return False

    def extract_comprehensive_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Extract comprehensive features using all step07 technical indicators.

        Args:
            market_data: Market data with OHLCV

        Returns:
            np.ndarray: Comprehensive feature array
        """
        try:
            features = []

            if len(market_data) < max(self.feature_config["lookback_periods"], 50):
                # Not enough data, return default features
                return np.array([0.5] * 150)  # Increased feature count

            # Price-based features
            close_prices = market_data["close"].values
            high_prices = market_data["high"].values
            low_prices = market_data["low"].values
            open_prices = market_data["open"].values
            volumes = market_data["volume"].values

            # Current price and recent prices
            current_price = close_prices[-1]

            # 1. Price momentum features
            for period in self.feature_config["price_momentum_periods"]:
                if len(close_prices) >= period:
                    momentum = (current_price - close_prices[-period]) / close_prices[
                        -period
                    ]
                    features.append(momentum)
                else:
                    features.append(0.0)

            # 2. Volatility features
            returns = np.diff(close_prices) / close_prices[:-1]
            for period in self.feature_config["volatility_periods"]:
                if len(returns) >= period:
                    volatility = np.std(returns[-period:])
                    features.append(volatility)
                else:
                    features.append(0.0)

            # 3. Volume features
            volume_trend = (
                (volumes[-1] - volumes[-5]) / volumes[-5] if volumes[-5] > 0 else 0
            )
            volume_ma_ratio = (
                volumes[-1]
                / np.mean(volumes[-self.feature_config["volume_ma_period"] :])
                if np.mean(volumes[-self.feature_config["volume_ma_period"] :]) > 0
                else 1.0
            )
            features.extend([volume_trend, volume_ma_ratio])

            # 4. RSI
            rsi_params = self.technical_indicators["RSI"]
            rsi = talib.RSI(close_prices, timeperiod=rsi_params["lookback_period"])
            features.append(rsi[-1] / 100 if not np.isnan(rsi[-1]) else 0.5)

            # 5. MACD
            macd_params = self.technical_indicators["MACD"]
            macd, macd_signal, macd_hist = talib.MACD(
                close_prices,
                fastperiod=macd_params["fast_period"],
                slowperiod=macd_params["slow_period"],
                signalperiod=macd_params["signal_period"],
            )
            features.extend(
                [
                    macd[-1] if not np.isnan(macd[-1]) else 0.0,
                    macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0.0,
                    macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0.0,
                ]
            )

            # 6. Bollinger Bands
            bb_params = self.technical_indicators["Bollinger_Bands"]
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                close_prices,
                timeperiod=bb_params["lookback_period"],
                nbdevup=bb_params["std_dev"],
                nbdevdn=bb_params["std_dev"],
            )
            bb_position = (
                (current_price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
                if bb_upper[-1] != bb_lower[-1]
                else 0.5
            )
            bb_squeeze = (
                (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
                if bb_middle[-1] > 0
                else 0.0
            )
            features.extend(
                [
                    bb_position if not np.isnan(bb_position) else 0.5,
                    bb_squeeze if not np.isnan(bb_squeeze) else 0.0,
                ]
            )

            # 7. SMA
            sma_params = self.technical_indicators["SMA"]
            sma_short = talib.SMA(close_prices, timeperiod=sma_params["short_period"])
            sma_long = talib.SMA(close_prices, timeperiod=sma_params["long_period"])
            sma_ratio = sma_short[-1] / sma_long[-1] if sma_long[-1] > 0 else 1.0
            features.append(sma_ratio if not np.isnan(sma_ratio) else 1.0)

            # 8. EMA
            ema_params = self.technical_indicators["EMA"]
            ema_short = talib.EMA(close_prices, timeperiod=ema_params["short_period"])
            ema_long = talib.EMA(close_prices, timeperiod=ema_params["long_period"])
            ema_ratio = ema_short[-1] / ema_long[-1] if ema_long[-1] > 0 else 1.0
            features.append(ema_ratio if not np.isnan(ema_ratio) else 1.0)

            # 9. ATR
            atr_params = self.technical_indicators["ATR"]
            atr = talib.ATR(
                high_prices,
                low_prices,
                close_prices,
                timeperiod=atr_params["lookback_period"],
            )
            atr_normalized = atr[-1] / current_price if current_price > 0 else 0.0
            features.append(atr_normalized if not np.isnan(atr_normalized) else 0.0)

            # 10. Stochastic
            stoch_params = self.technical_indicators["Stochastic"]
            stoch_k, stoch_d = talib.STOCH(
                high_prices,
                low_prices,
                close_prices,
                fastk_period=stoch_params["k_period"],
                slowk_period=stoch_params["d_period"],
                slowd_period=stoch_params["d_period"],
            )
            features.extend(
                [
                    stoch_k[-1] / 100 if not np.isnan(stoch_k[-1]) else 0.5,
                    stoch_d[-1] / 100 if not np.isnan(stoch_d[-1]) else 0.5,
                ]
            )

            # 11. ADX
            adx_params = self.technical_indicators["ADX"]
            adx = talib.ADX(
                high_prices,
                low_prices,
                close_prices,
                timeperiod=adx_params["lookback_period"],
            )
            features.append(adx[-1] / 100 if not np.isnan(adx[-1]) else 0.5)

            # 12. CCI
            cci_params = self.technical_indicators["CCI"]
            cci = talib.CCI(
                high_prices,
                low_prices,
                close_prices,
                timeperiod=cci_params["lookback_period"],
            )
            # Normalize CCI to 0-1 range
            cci_normalized = (cci[-1] + 300) / 600 if not np.isnan(cci[-1]) else 0.5
            features.append(np.clip(cci_normalized, 0, 1))

            # 13. Additional price-based features
            price_range = (high_prices[-1] - low_prices[-1]) / current_price
            upper_shadow = (high_prices[-1] - current_price) / current_price
            lower_shadow = (current_price - low_prices[-1]) / current_price
            body_size = abs(close_prices[-1] - open_prices[-1]) / current_price

            features.extend([price_range, upper_shadow, lower_shadow, body_size])

            # 14. Latest return
            latest_return = (
                (current_price - close_prices[-2]) / close_prices[-2]
                if len(close_prices) > 1
                else 0.0
            )
            features.append(latest_return)

            # 15. Price acceleration (second derivative)
            if len(close_prices) >= 3:
                return_1 = (close_prices[-1] - close_prices[-2]) / close_prices[-2]
                return_2 = (close_prices[-2] - close_prices[-3]) / close_prices[-3]
                acceleration = return_1 - return_2
                features.append(acceleration)
            else:
                features.append(0.0)

            return np.array(features)

        except Exception as e:
            self.logger.exception(f"❌ Comprehensive feature extraction failed: {e}")
            return np.array([0.5] * 150)

    @handles_errors
    def prepare_scenario_targets(
        self,
        X: np.ndarray,
        market_data: pd.DataFrame,
        base_price_column: str = "close",
    ) -> np.ndarray:
        """
        Label each data point with the scenario that occurred first.

        Args:
            X: Feature array
            market_data: Market data with OHLCV
            base_price_column: Column to use for price calculations

        Returns:
            np.ndarray: Scenario labels for each data point
        """
        try:
            if len(X) != len(market_data):
                msg = "Feature array and market data must have same length"
                raise ValueError(msg)

            scenario_labels = []
            prices = market_data[base_price_column].values

            for i in range(len(X)):
                # Look ahead to see which scenario occurs first
                scenario = self._determine_first_scenario(
                    prices[i:],
                    i,
                    self.time_limit_minutes,
                )
                scenario_labels.append(scenario)

            return np.array(scenario_labels)

        except Exception as e:
            self.logger.exception(f"❌ Scenario labeling failed: {e}")
            return np.full(len(X), len(self.scenarios) - 1)  # Default to neutral

    def _determine_first_scenario(
        self,
        future_prices: np.ndarray,
        current_index: int,
        time_limit: int,
    ) -> int:
        """
        Determine which scenario occurs first in the future price data.

        Args:
            future_prices: Future price data
            current_index: Current data point index
            time_limit: Maximum look-ahead periods

        Returns:
            int: Scenario label
        """
        try:
            if len(future_prices) < 2:
                return len(self.scenarios) - 1  # Neutral if not enough data

            current_price = future_prices[0]
            look_ahead_prices = future_prices[
                1 : min(len(future_prices), time_limit + 1)
            ]

            # Check each scenario in order of preference
            for scenario_id in range(len(self.scenarios) - 1):  # Exclude neutral
                scenario = self.scenarios[scenario_id]

                if self._scenario_triggered(
                    look_ahead_prices,
                    current_price,
                    scenario,
                ):
                    return scenario_id

            return len(self.scenarios) - 1  # Neutral if no scenario triggered

        except Exception as e:
            self.logger.exception(f"❌ Scenario determination failed: {e}")
            return len(self.scenarios) - 1

    def _scenario_triggered(
        self,
        prices: np.ndarray,
        current_price: float,
        scenario: dict[str, Any],
    ) -> bool:
        """
        Check if a specific scenario is triggered in the price data.

        Args:
            prices: Future price data
            current_price: Current price
            scenario: Scenario definition

        Returns:
            bool: True if scenario is triggered
        """
        try:
            profit_target = scenario["profit_target"]
            stop_loss = scenario["stop_loss"]

            # Calculate price changes relative to current price
            price_changes = (prices - current_price) / current_price

            # Check if profit target is hit before stop loss
            for price_change in price_changes:
                if scenario["zone_type"] == "profit":
                    if price_change >= profit_target:
                        return True
                    if price_change <= stop_loss:
                        return False
                elif scenario["zone_type"] == "risk":
                    if price_change <= stop_loss:
                        return True
                    if price_change >= profit_target:
                        return False

            return False

        except Exception as e:
            self.logger.exception(f"❌ Scenario trigger check failed: {e}")
            return False

    @handles_errors
    async def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        market_data: pd.DataFrame | None = None,
    ) -> bool:
        """
        Train the enhanced scenario prediction model.

        Args:
            X_train: Training features
            y_train: Training scenario labels
            X_val: Validation features
            y_val: Validation scenario labels
            market_data: Market data for feature engineering

        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            self.logger.info("Training enhanced scenario prediction model...")

            # Prepare scenario targets if not provided
            if market_data is not None and len(y_train) == len(X_train):
                y_train = self.prepare_scenario_targets(X_train, market_data)

            # Split validation data if not provided
            if X_val is None or y_val is None:
                X_train_split, X_val, y_train_split, y_val = train_test_split(
                    X_train,
                    y_train,
                    test_size=0.2,
                    random_state=42,
                    stratify=y_train,
                )
            else:
                X_train_split, y_train_split = X_train, y_train

            # Train model
            self.model.fit(
                X_train_split,
                y_train_split,
                eval_set=[(X_val, y_val)],
                eval_metric="multi_logloss",
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )

            # Calculate feature importance
            self.feature_importance = dict(
                zip(
                    [f"feature_{i}" for i in range(X_train.shape[1])],
                    self.model.feature_importances_,
                    strict=False,
                )
            )

            # Calculate performance metrics
            y_pred = self.model.predict(X_val)
            y_pred_proba = self.model.predict_proba(X_val)

            self.model_performance = {
                "accuracy": accuracy_score(y_val, y_pred),
                "log_loss": log_loss(y_val, y_pred_proba),
                "n_samples": len(X_train),
                "n_features": X_train.shape[1],
                "n_scenarios": len(self.scenarios),
            }

            self.is_trained = True
            self.last_training_time = datetime.now()

            self.logger.info(
                f"✅ Enhanced model trained successfully. Accuracy: {self.model_performance['accuracy']:.3f}"
            )
            return True

        except Exception as e:
            self.logger.exception(f"❌ Enhanced model training failed: {e}")
            return False

    @handles_errors
    async def predict_scenarios(
        self,
        X: np.ndarray,
        market_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Generate enhanced scenario predictions.

        Args:
            X: Feature array
            market_data: Market data (optional, for additional context)

        Returns:
            dict: Enhanced scenario predictions with probabilities and metadata
        """
        try:
            if not self.is_trained:
                self.logger.warning(
                    "Enhanced model not trained, using fallback predictions"
                )
                return self._generate_enhanced_fallback_predictions(X)

            # Generate probability predictions
            probabilities = self.model.predict_proba(X)

            # Get most likely scenario
            predicted_scenario = self.model.predict(X)[0]

            # Calculate scenario-specific metrics
            scenario_analysis = self._analyze_enhanced_scenario_probabilities(
                probabilities[0]
            )

            # Calculate confidence score
            confidence = self._calculate_enhanced_confidence(probabilities[0])

            return {
                "probabilities": dict(
                    zip(range(len(probabilities[0])), probabilities[0], strict=False)
                ),
                "predicted_scenario": predicted_scenario,
                "scenario_name": self.scenarios[predicted_scenario]["name"],
                "confidence": confidence,
                "scenario_analysis": scenario_analysis,
                "metadata": {
                    "model_type": "enhanced_scenario_based",
                    "generation_timestamp": datetime.now().isoformat(),
                    "is_trained": self.is_trained,
                    "last_training_time": (
                        self.last_training_time.isoformat()
                        if self.last_training_time
                        else None
                    ),
                    "n_scenarios": len(self.scenarios),
                    "time_limit_minutes": self.time_limit_minutes,
                },
            }

        except Exception as e:
            self.logger.exception(f"❌ Enhanced scenario prediction failed: {e}")
            return self._generate_enhanced_fallback_predictions(X)

    def _analyze_enhanced_scenario_probabilities(
        self, probabilities: np.ndarray
    ) -> dict[str, Any]:
        """
        Analyze enhanced scenario probabilities for decision making.

        Args:
            probabilities: Probability array for each scenario

        Returns:
            dict: Enhanced analysis results
        """
        try:
            # Calculate combined probabilities by zone type
            profit_zone_probs = []
            risk_zone_probs = []

            for scenario_id, scenario in self.scenarios.items():
                if scenario["zone_type"] == "profit":
                    profit_zone_probs.append(probabilities[scenario_id])
                elif scenario["zone_type"] == "risk":
                    risk_zone_probs.append(probabilities[scenario_id])

            profit_zone_prob = sum(profit_zone_probs)
            risk_zone_prob = sum(risk_zone_probs)
            neutral_prob = probabilities[len(self.scenarios) - 1]

            # Determine dominant zone
            if profit_zone_prob > risk_zone_prob and profit_zone_prob > neutral_prob:
                dominant_zone = "profit"
            elif risk_zone_prob > profit_zone_prob and risk_zone_prob > neutral_prob:
                dominant_zone = "risk"
            else:
                dominant_zone = "neutral"

            # Calculate enhanced metrics
            risk_reward_ratio = profit_zone_prob / (risk_zone_prob + 1e-8)
            profit_risk_difference = profit_zone_prob - risk_zone_prob

            # Calculate scenario dominance
            max_prob = max(probabilities)
            scenario_dominance = max_prob / (sum(probabilities) + 1e-8)

            # Calculate zone distribution
            zone_distribution = {
                "profit_zones": len(profit_zone_probs),
                "risk_zones": len(risk_zone_probs),
                "profit_probabilities": profit_zone_probs,
                "risk_probabilities": risk_zone_probs,
            }

            return {
                "profit_zone_probability": profit_zone_prob,
                "risk_zone_probability": risk_zone_prob,
                "neutral_probability": neutral_prob,
                "dominant_zone": dominant_zone,
                "risk_reward_ratio": risk_reward_ratio,
                "profit_risk_difference": profit_risk_difference,
                "scenario_dominance": scenario_dominance,
                "zone_distribution": zone_distribution,
                "max_probability": max_prob,
                "probability_entropy": -np.sum(
                    probabilities * np.log(probabilities + 1e-8)
                ),
            }

        except Exception as e:
            self.logger.exception(f"❌ Enhanced scenario analysis failed: {e}")
            return {
                "profit_zone_probability": 0.0,
                "risk_zone_probability": 0.0,
                "neutral_probability": 1.0,
                "dominant_zone": "neutral",
                "risk_reward_ratio": 0.0,
                "profit_risk_difference": 0.0,
                "scenario_dominance": 0.0,
                "zone_distribution": {
                    "profit_zones": 0,
                    "risk_zones": 0,
                    "profit_probabilities": [],
                    "risk_probabilities": [],
                },
                "max_probability": 0.0,
                "probability_entropy": 0.0,
            }

    def _calculate_enhanced_confidence(self, probabilities: np.ndarray) -> float:
        """
        Calculate enhanced confidence score based on probability distribution.

        Args:
            probabilities: Probability array

        Returns:
            float: Enhanced confidence score (0-1)
        """
        try:
            # Use entropy-based confidence with scenario dominance
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
            max_entropy = np.log(len(probabilities))

            # Base confidence from entropy
            base_confidence = 1 - (entropy / max_entropy)

            # Boost confidence based on scenario dominance
            max_prob = max(probabilities)
            dominance_boost = max_prob * 0.3

            # Final confidence
            confidence = base_confidence + dominance_boost

            return np.clip(confidence, 0.0, 1.0)

        except Exception as e:
            self.logger.exception(f"❌ Enhanced confidence calculation failed: {e}")
            return 0.5

    def _generate_enhanced_fallback_predictions(self, X: np.ndarray) -> dict[str, Any]:
        """
        Generate enhanced fallback predictions when model is not trained.

        Args:
            X: Feature array

        Returns:
            dict: Enhanced fallback predictions
        """
        try:
            # Simple heuristic-based predictions
            n_scenarios = len(self.scenarios)
            base_prob = 1.0 / n_scenarios

            # Slightly favor neutral scenario
            probabilities = [base_prob * 0.8] * (n_scenarios - 1) + [base_prob * 1.4]

            return {
                "probabilities": dict(
                    zip(range(n_scenarios), probabilities, strict=False)
                ),
                "predicted_scenario": n_scenarios - 1,  # Neutral
                "scenario_name": self.scenarios[n_scenarios - 1]["name"],
                "confidence": 0.3,
                "scenario_analysis": {
                    "profit_zone_probability": base_prob * 8,
                    "risk_zone_probability": base_prob * 8,
                    "neutral_probability": base_prob * 1.4,
                    "dominant_zone": "neutral",
                    "risk_reward_ratio": 1.0,
                    "profit_risk_difference": 0.0,
                    "scenario_dominance": base_prob * 1.4,
                    "zone_distribution": {
                        "profit_zones": 8,
                        "risk_zones": 8,
                        "profit_probabilities": [base_prob * 0.8] * 8,
                        "risk_probabilities": [base_prob * 0.8] * 8,
                    },
                    "max_probability": base_prob * 1.4,
                    "probability_entropy": np.log(n_scenarios),
                },
                "metadata": {
                    "model_type": "enhanced_scenario_based_fallback",
                    "generation_timestamp": datetime.now().isoformat(),
                    "is_trained": False,
                    "last_training_time": None,
                    "n_scenarios": n_scenarios,
                    "time_limit_minutes": self.time_limit_minutes,
                },
            }

        except Exception as e:
            self.logger.exception(
                f"❌ Enhanced fallback prediction generation failed: {e}"
            )
            return {
                "probabilities": dict.fromkeys(range(n_scenarios), 1.0 / n_scenarios),
                "predicted_scenario": n_scenarios - 1,
                "scenario_name": "Neutral",
                "confidence": 0.0,
                "scenario_analysis": {
                    "profit_zone_probability": 0.5,
                    "risk_zone_probability": 0.5,
                    "neutral_probability": 0.0,
                    "dominant_zone": "neutral",
                    "risk_reward_ratio": 1.0,
                    "profit_risk_difference": 0.0,
                    "scenario_dominance": 0.0,
                    "zone_distribution": {
                        "profit_zones": 0,
                        "risk_zones": 0,
                        "profit_probabilities": [],
                        "risk_probabilities": [],
                    },
                    "max_probability": 0.0,
                    "probability_entropy": 0.0,
                },
                "metadata": {
                    "model_type": "enhanced_scenario_based_error",
                    "generation_timestamp": datetime.now().isoformat(),
                    "is_trained": False,
                    "last_training_time": None,
                    "n_scenarios": n_scenarios,
                    "time_limit_minutes": self.time_limit_minutes,
                },
            }

    def get_enhanced_configuration_summary(self) -> dict[str, Any]:
        """
        Get enhanced configuration summary for step17 optimization.

        Returns:
            dict: Enhanced configuration summary
        """
        return {
            "scenarios": self.scenarios,
            "time_limit_minutes": self.time_limit_minutes,
            "model_config": self.model_config,
            "decision_thresholds": self.decision_thresholds,
            "technical_indicators": self.technical_indicators,
            "feature_config": self.feature_config,
            "is_trained": self.is_trained,
            "model_performance": self.model_performance,
            "feature_importance": self.feature_importance,
            "n_scenarios": len(self.scenarios),
        }
