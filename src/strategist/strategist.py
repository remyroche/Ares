from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class Strategist:
    """
    Enhanced strategist that orchestrates ML regime classification and model selection.

    Main responsibilities:
    1. Run ML regime classifying model
    2. Determine market regimes (bullish, bearish, sideways, SR, candle)
    3. Based on regime, run proper ML model with ml_confidence_predictor.py
    4. Pass information to Analyst and collect volatility data for tactician/position_sizer.py

    Note: Position sizing is entirely handled by tactician/position_sizer.py
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize strategist with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("Strategist")

        # Strategy state
        self.current_strategy: dict[str, Any] | None = None
        self.strategy_history: list[dict[str, Any]] = []
        self.last_strategy_update: datetime | None = None
        self.current_regime: str | None = None
        self.regime_confidence: float = 0.0

        # Configuration
        from src.config_optuna import get_parameter_value

        self.strategy_config: dict[str, Any] = self.config.get("strategist", {})
        self.risk_config: dict[str, Any] = self.config.get("risk_management", {})

        # Performance tracking
        self.strategy_performance: dict[str, float] = {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }

        # ML Regime Classifier integration
        self.regime_classifier = None
        self.enable_regime_classification: bool = get_parameter_value(
            "strategist_parameters.enable_regime_classification",
            True,
        )

        # ML Confidence Predictor integration
        self.ml_confidence_predictor = None
        self.enable_ml_predictions: bool = get_parameter_value(
            "strategist_parameters.enable_ml_predictions",
            True,
        )

        # Volatility targeting information (passed to tactician)
        self.volatility_info = None
        self.enable_volatility_targeting: bool = get_parameter_value(
            "strategist_parameters.enable_volatility_targeting",
            True,
        )

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid strategist configuration"),
            AttributeError: (False, "Missing required strategy parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="strategist initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize strategist with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        self.logger.info("Initializing Strategist...")

        # Load strategy configuration
        await self._load_strategy_configuration()

        # Initialize risk management
        await self._initialize_risk_management()

        # Initialize ML Regime Classifier
        if self.enable_regime_classification:
            await self._initialize_regime_classifier()

        # Initialize ML Confidence Predictor
        if self.enable_ml_predictions:
            await self._initialize_ml_confidence_predictor()

        # Note: Volatility targeting is handled by tactician/position_sizer.py
        # Strategist only collects volatility information to pass to tactician

        # Validate configuration
        if not self._validate_configuration():
            self.logger.error("Invalid configuration for strategist")
            return False

        self.logger.info("✅ Strategist initialization completed successfully")
        return True

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="strategy configuration loading",
    )
    async def _load_strategy_configuration(self) -> None:
        """Load strategy configuration."""
        # Set default strategy parameters
        self.strategy_config.setdefault("max_position_size", 0.1)
        self.strategy_config.setdefault("max_daily_trades", 10)
        self.strategy_config.setdefault("min_confidence_threshold", 0.6)
        self.strategy_config.setdefault("strategy_timeout_seconds", 30)
        self.strategy_config.setdefault("regime_update_interval", 300)  # 5 minutes

        self.logger.info("Strategy configuration loaded successfully")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="ML regime classifier initialization",
    )
    async def _initialize_regime_classifier(self) -> None:
        """Initialize ML Regime Classifier for market regime determination."""
        from src.analyst.unified_regime_classifier import UnifiedRegimeClassifier

        self.regime_classifier = UnifiedRegimeClassifier(self.config)
        if not self.regime_classifier.load_models():
            self.logger.info(
                "No existing regime classifier models found. Will train when needed.",
            )
        else:
            self.logger.info("✅ ML Regime Classifier initialized successfully")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="ML confidence predictor initialization",
    )
    async def _initialize_ml_confidence_predictor(self) -> None:
        """Initialize ML Confidence Predictor for regime-specific predictions."""
        from src.analyst.ml_confidence_predictor import (
            setup_ml_confidence_predictor,
        )

        self.ml_confidence_predictor = await setup_ml_confidence_predictor(
            self.config,
        )
        if self.ml_confidence_predictor:
            await self.ml_confidence_predictor.initialize()
            self.logger.info("✅ ML Confidence Predictor initialized successfully")
        else:
            self.logger.error("❌ Failed to initialize ML Confidence Predictor")

    # Volatility targeting is handled by tactician/position_sizer.py
    # This method is removed as volatility targeting decisions belong to the tactician

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="risk management initialization",
    )
    async def _initialize_risk_management(self) -> None:
        """Initialize risk management components."""
        try:
            # Set default risk parameters
            self.risk_config.setdefault("max_daily_loss", 0.02)
            self.risk_config.setdefault("max_position_risk", 0.01)
            self.risk_config.setdefault("max_correlation", 0.7)
            self.risk_config.setdefault("stop_loss_multiplier", 2.0)

            self.logger.info("Risk management initialized")

        except Exception as e:
            self.logger.error(f"Error initializing risk management: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate strategist configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            required_strategy_keys = [
                "max_position_size",
                "max_daily_trades",
                "min_confidence_threshold",
            ]
            for key in required_strategy_keys:
                if key not in self.strategy_config:
                    self.logger.error(
                        f"Missing required strategy configuration key: {key}",
                    )
                    return False

            required_risk_keys = [
                "max_daily_loss",
                "max_position_risk",
                "stop_loss_multiplier",
            ]
            for key in required_risk_keys:
                if key not in self.risk_config:
                    self.logger.error(f"Missing required risk configuration key: {key}")
                    return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_specific_errors(
        error_handlers={
            ConnectionError: (None, "Failed to connect to data source"),
            TimeoutError: (None, "Strategy generation timed out"),
            ValueError: (None, "Invalid market intelligence data"),
        },
        default_return=None,
        context="strategy generation",
    )
    async def generate_strategy(
        self,
        market_data: pd.DataFrame,
        current_price: float,
    ) -> dict[str, Any] | None:
        """
        Generate trading strategy based on ML regime classification and predictions.

        Args:
            market_data: Market data DataFrame with OHLCV
            current_price: Current market price

        Returns:
            Optional[Dict[str, Any]]: Generated strategy or None if failed
        """
        try:
            self.logger.info("Generating trading strategy...")

            # Step 1: Run ML regime classification
            regime_info = await self._classify_market_regime(market_data)
            if not regime_info:
                self.logger.error("Failed to classify market regime")
                return None

            # Step 2: Determine market regime and update state
            self.current_regime = regime_info["regime"]
            self.regime_confidence = regime_info["confidence"]

            self.logger.info(
                f"📊 Current market regime: {self.current_regime} (confidence: {self.regime_confidence:.2f})",
            )

            # Step 3: Run appropriate ML model based on regime
            ml_predictions = await self._run_regime_specific_ml_model(
                market_data,
                current_price,
                regime_info,
            )
            if not ml_predictions:
                self.logger.warning(
                    "No ML predictions available, using fallback strategy",
                )
                ml_predictions = self._generate_fallback_predictions(current_price)

            # Step 4: Collect volatility information for tactician
            volatility_info = await self._collect_volatility_info(market_data)

            # Step 5: Generate comprehensive strategy (volatility decisions handled by tactician)
            strategy = await self._generate_comprehensive_strategy(
                regime_info,
                ml_predictions,
                volatility_info,
                current_price,
            )

            # Update strategy state
            self.current_strategy = strategy
            self.strategy_history.append(strategy)
            self.last_strategy_update = datetime.now()

            self.logger.info("✅ Strategy generated successfully")
            return strategy

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="market regime classification",
    )
    async def _classify_market_regime(
        self,
        market_data: pd.DataFrame,
    ) -> dict[str, Any] | None:
        """
        Classify the current market regime using ML regime classifier.

        Args:
            market_data: Market data DataFrame

        Returns:
            Optional[Dict[str, Any]]: Regime classification results
        """
        try:
            if not self.regime_classifier:
                self.logger.error("Regime classifier not initialized")
                return None

            # Check if regime classifier is trained
            if not self.regime_classifier.trained:
                self.logger.info("Regime classifier not trained. Training now...")
                success = await self.regime_classifier.train_complete_system(
                    market_data,
                )
                if not success:
                    self.logger.error("Failed to train regime classifier")
                    return None

            # Predict regime
            regime, confidence, additional_info = self.regime_classifier.predict_regime(
                market_data,
            )

            regime_info = {
                "regime": regime,
                "confidence": confidence,
                "additional_info": additional_info,
                "classification_time": datetime.now(),
                "system_status": self.regime_classifier.get_system_status(),
            }

            self.logger.info(
                f"🎯 Market regime classified as: {regime} (confidence: {confidence:.2f})",
            )
            return regime_info

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="regime-specific ML model execution",
    )
    async def _run_regime_specific_ml_model(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        regime_info: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Run the appropriate ML model based on the current market regime.

        Args:
            market_data: Market data DataFrame
            current_price: Current market price
            regime_info: Regime classification information

        Returns:
            Optional[Dict[str, Any]]: ML predictions
        """
        if not self.ml_confidence_predictor:
            self.logger.error("ML confidence predictor not initialized")
            return None

        regime = regime_info["regime"]
        confidence = regime_info["confidence"]

        # Adjust prediction parameters based on regime
        prediction_params = self._get_regime_specific_prediction_params(
            regime,
            confidence,
        )

        # Run ML confidence prediction
        ml_predictions = (
            await self.ml_confidence_predictor.predict_confidence_table(
                market_data,
                current_price,
            )
        )

        if ml_predictions:
            # Add regime-specific adjustments
            ml_predictions["regime"] = regime
            ml_predictions["regime_confidence"] = confidence
            ml_predictions["prediction_params"] = prediction_params
            ml_predictions["regime_adjustments"] = self._apply_regime_adjustments(
                ml_predictions,
                regime,
                confidence,
            )

        return ml_predictions

    def _get_regime_specific_prediction_params(
        self,
        regime: str,
        confidence: float,
    ) -> dict[str, Any]:
        """
        Get regime-specific prediction parameters.

        Args:
            regime: Current market regime
            confidence: Regime classification confidence

        Returns:
            Dict[str, Any]: Regime-specific parameters
        """
        try:
            base_params = {
                "confidence_threshold": 0.6,
                "risk_adjustment": 1.0,
                "position_size_multiplier": 1.0,
                "stop_loss_multiplier": 1.0,
                "take_profit_multiplier": 1.0,
            }

            # Adjust parameters based on regime
            if regime == "BULL":
                base_params.update(
                    {
                        "confidence_threshold": 0.5,  # Lower threshold for bullish markets
                        "position_size_multiplier": 1.2,
                        "take_profit_multiplier": 1.3,
                    },
                )
            elif regime == "BEAR":
                base_params.update(
                    {
                        "confidence_threshold": 0.7,  # Higher threshold for bearish markets
                        "risk_adjustment": 0.8,
                        "position_size_multiplier": 0.8,
                        "stop_loss_multiplier": 0.8,
                    },
                )
            elif regime == "SIDEWAYS":
                base_params.update(
                    {
                        "confidence_threshold": 0.65,
                        "position_size_multiplier": 0.9,
                        "take_profit_multiplier": 0.9,
                    },
                )
            elif regime == "SIDEWAYS":
                base_params.update(
                    {
                        "confidence_threshold": 0.8,  # Very high threshold for sideways regimes
                        "risk_adjustment": 0.6,
                        "position_size_multiplier": 0.5,
                        "stop_loss_multiplier": 0.7,
                    },
                )
            elif regime == "VOLATILE":
                base_params.update(
                    {
                        "confidence_threshold": 0.75,
                        "position_size_multiplier": 1.1,
                        "take_profit_multiplier": 1.2,
                    },
                )

            # Adjust based on confidence
            if confidence < 0.5:
                base_params["position_size_multiplier"] *= 0.8
                base_params["risk_adjustment"] *= 0.8

            return base_params

        except Exception as e:
            self.logger.error(f"Error getting regime-specific parameters: {e}")
            return {"confidence_threshold": 0.6, "risk_adjustment": 1.0}

    def _apply_regime_adjustments(
        self,
        ml_predictions: dict[str, Any],
        regime: str,
        confidence: float,
    ) -> dict[str, Any]:
        """
        Apply regime-specific adjustments to ML predictions.

        Args:
            ml_predictions: Original ML predictions
            regime: Current market regime
            confidence: Regime confidence

        Returns:
            Dict[str, Any]: Adjusted predictions
        """
        try:
            adjustments = {
                "confidence_scores_adjusted": {},
                "expected_decreases_adjusted": {},
                "position_sizing_adjusted": {},
                "risk_parameters_adjusted": {},
            }

            # Adjust confidence scores based on regime
            confidence_scores = ml_predictions.get("confidence_scores", {})
            for level, score in confidence_scores.items():
                if regime == "BULL":
                    adjusted_score = min(
                        1.0,
                        score * 1.1,
                    )  # Boost confidence in bullish regime
                elif regime == "BEAR":
                    adjusted_score = score * 0.9  # Reduce confidence in bearish regime
                elif regime == "SIDEWAYS":
                    adjusted_score = score * 0.8  # Significantly reduce confidence
                else:
                    adjusted_score = score

                adjustments["confidence_scores_adjusted"][level] = adjusted_score

            # Adjust adversarial confidences based on regime
            adversarial_confidences = ml_predictions.get("adversarial_confidences", {})
            for level, prob in adversarial_confidences.items():
                if regime == "BEAR":
                    adjusted_prob = min(1.0, prob * 1.2)  # Increase bearish probability
                elif regime == "BULL":
                    adjusted_prob = prob * 0.8  # Decrease bearish probability
                else:
                    adjusted_prob = prob

                adjustments["adversarial_confidences_adjusted"][level] = adjusted_prob

            # Apply confidence multiplier
            confidence_multiplier = max(0.5, confidence)
            for key in [
                "confidence_scores_adjusted",
                "adversarial_confidences_adjusted",
            ]:
                for level in adjustments[key]:
                    adjustments[key][level] *= confidence_multiplier

            return adjustments

        except Exception as e:
            self.logger.error(f"Error applying regime adjustments: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="volatility information collection",
    )
    async def _collect_volatility_info(
        self,
        market_data: pd.DataFrame,
    ) -> dict[str, Any] | None:
        """
        Collect volatility information for tactician/position_sizer.py.
        Note: Actual volatility targeting decisions are made by the tactician.

        Args:
            market_data: Market data DataFrame

        Returns:
            Optional[Dict[str, Any]]: Volatility information for tactician
        """
        try:
            # Calculate basic volatility metrics for tactician
            returns = market_data["close"].pct_change().dropna()
            current_volatility = returns.std() * np.sqrt(252)  # Annualized volatility

            # Calculate additional volatility metrics
            volatility_info = {
                "current_volatility": current_volatility,
                "volatility_percentile": self._calculate_volatility_percentile(returns),
                "volatility_trend": self._calculate_volatility_trend(returns),
                "market_data": {
                    "returns": returns.tolist()[-20:],  # Last 20 returns
                    "price_data": market_data[
                        ["open", "high", "low", "close", "volume"]
                    ]
                    .tail(20)
                    .to_dict("records"),
                },
                "calculation_time": datetime.now(),
                "note": "Volatility targeting decisions handled by tactician/position_sizer.py",
            }

            self.logger.info(
                f"📊 Collected volatility info - Current volatility: {current_volatility:.3f}",
            )
            return volatility_info

        except Exception as e:
            self.logger.error(f"Error collecting volatility info: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="comprehensive strategy generation",
    )
    async def _generate_comprehensive_strategy(
        self,
        regime_info: dict[str, Any],
        ml_predictions: dict[str, Any],
        volatility_info: dict[str, Any],
        current_price: float,
    ) -> dict[str, Any] | None:
        """
        Generate comprehensive strategy combining all components.

        Args:
            regime_info: Market regime information
            ml_predictions: ML predictions
            volatility_info: Volatility targeting information
            current_price: Current market price

        Returns:
            Optional[Dict[str, Any]]: Comprehensive strategy
        """
        try:
            # Calculate risk parameters
            risk_parameters = await self._calculate_risk_parameters(
                regime_info,
                volatility_info,
            )

            # Note: All position sizing handled by tactician/position_sizer.py

            # Generate entry signals
            entry_signals = await self._generate_entry_signals(
                regime_info,
                ml_predictions,
            )

            # Generate exit signals
            exit_signals = await self._generate_exit_signals(
                regime_info,
                ml_predictions,
            )

            # Combine into comprehensive strategy
            strategy = {
                "regime_info": regime_info,
                "ml_predictions": ml_predictions,
                "volatility_info": volatility_info,  # Passed to tactician for position sizing
                "entry_signals": entry_signals,
                "exit_signals": exit_signals,
                "risk_parameters": risk_parameters,
                "position_sizing_note": "All position sizing handled by tactician/position_sizer.py",
                "confidence_score": self._calculate_confidence_score(
                    regime_info,
                    ml_predictions,
                ),
                "generation_time": datetime.now(),
                "valid_until": datetime.now() + timedelta(minutes=30),
                "strategy_metadata": {
                    "regime": regime_info["regime"],
                    "regime_confidence": regime_info["confidence"],
                    "ml_model_used": "ml_confidence_predictor",
                    "volatility_info_collected": self.enable_volatility_targeting,
                    "note": "Volatility targeting decisions handled by tactician/position_sizer.py",
                },
            }

            # Validate strategy
            if not self._validate_strategy(strategy):
                self.logger.error("Generated strategy validation failed")
                return None

            return strategy

        except Exception as e:
            self.logger.error(f"Error generating comprehensive strategy: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="risk parameter calculation",
    )
    async def _calculate_risk_parameters(
        self,
        regime_info: dict[str, Any],
        volatility_info: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Calculate risk parameters based on regime and volatility.

        Args:
            regime_info: Market regime information
            volatility_info: Volatility targeting information

        Returns:
            Optional[Dict[str, Any]]: Risk parameters
        """
        try:
            regime = regime_info["regime"]
            regime_confidence = regime_info["confidence"]

            # Base risk parameters
            risk_params = {
                "max_daily_loss": self.risk_config["max_daily_loss"],
                "stop_loss_distance": 0.01,  # 1% base stop loss
                "take_profit_distance": 0.02,  # 2% base take profit
                "max_correlation": self.risk_config["max_correlation"],
                "note": "Position sizing handled by tactician/position_sizer.py",
            }

            # Adjust based on regime
            if regime == "BEAR":
                risk_params["stop_loss_distance"] *= 0.8  # Tighter stop loss
                risk_params["take_profit_distance"] *= 0.7  # Lower take profit
            elif regime == "BULL":
                risk_params["take_profit_distance"] *= 1.3  # Higher take profit
            elif regime == "SIDEWAYS":
                risk_params["stop_loss_distance"] *= 0.6  # Very tight stop loss
                risk_params["take_profit_distance"] *= 0.5  # Lower take profit

            # Note: Volatility-based position sizing handled by tactician/position_sizer.py
            # This provides basic risk parameters for the tactician to use

            # Adjust based on regime confidence
            if regime_confidence < 0.5:
                risk_params["stop_loss_distance"] *= 0.9

            return risk_params

        except Exception as e:
            self.logger.error(f"Error calculating risk parameters: {e}")
            return None

    # Position sizing is entirely handled by tactician/position_sizer.py
    # This method is removed as position sizing decisions belong to the tactician

    def _calculate_volatility_percentile(self, returns: pd.Series) -> float:
        """
        Calculate volatility percentile based on historical data.

        Args:
            returns: Price returns series

        Returns:
            float: Volatility percentile (0-1)
        """
        try:
            if len(returns) < 20:
                return 0.5  # Default to median if insufficient data

            # Calculate rolling volatility
            rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
            current_vol = rolling_vol.iloc[-1]

            # Calculate percentile
            percentile = (rolling_vol < current_vol).mean()
            return float(percentile)

        except Exception as e:
            self.logger.error(f"Error calculating volatility percentile: {e}")
            return 0.5

    def _calculate_volatility_trend(self, returns: pd.Series) -> str:
        """
        Calculate volatility trend direction.

        Args:
            returns: Price returns series

        Returns:
            str: Trend direction ('increasing', 'decreasing', 'stable')
        """
        try:
            if len(returns) < 40:
                return "stable"  # Default if insufficient data

            # Calculate rolling volatility
            rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)

            # Compare recent vs earlier volatility
            recent_vol = rolling_vol.tail(10).mean()
            earlier_vol = rolling_vol.head(10).mean()

            if recent_vol > earlier_vol * 1.1:
                return "increasing"
            if recent_vol < earlier_vol * 0.9:
                return "decreasing"
            return "stable"

        except Exception as e:
            self.logger.error(f"Error calculating volatility trend: {e}")
            return "stable"

    def _get_regime_adjustment(self, regime: str) -> float:
        """
        Get position size adjustment based on market regime.

        Args:
            regime: Market regime string

        Returns:
            float: Position size adjustment multiplier
        """
        try:
            regime_adjustments = {
                "BULL": 1.2,  # Increase position size in bullish regime
                "BEAR": 0.8,  # Decrease position size in bearish regime
                "SIDEWAYS": 1.0,  # No adjustment in sideways regime
                # Legacy S/R/Candle code removed: 0.5,  # Significantly reduce position size
                # Legacy S/R/Candle code removed: 1.1,  # Slight increase for SR action
            }

            return regime_adjustments.get(regime, 1.0)

        except Exception as e:
            self.logger.error(f"Error calculating regime adjustment: {e}")
            return 1.0

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="entry signal generation",
    )
    async def _generate_entry_signals(
        self,
        regime_info: dict[str, Any],
        ml_predictions: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Generate entry signals based on regime and ML predictions.

        Args:
            regime_info: Market regime information
            ml_predictions: ML predictions

        Returns:
            Optional[Dict[str, Any]]: Entry signals
        """
        try:
            regime = regime_info["regime"]
            confidence_scores = ml_predictions.get("confidence_scores", {})
            adversarial_confidences = ml_predictions.get("adversarial_confidences", {})

            entry_signals = {
                "long_conditions": [],
                "short_conditions": [],
                "confidence": 0.5,
                "timeframe": "1m",
            }

            # Generate long entry conditions based on ML confidence
            for increase_level, confidence in confidence_scores.items():
                if confidence >= 0.7:  # High confidence threshold
                    entry_signals["long_conditions"].append(
                        {
                            "type": "ml_high_confidence",
                            "confidence": confidence,
                            "price_increase_target": float(increase_level),
                            "regime": regime,
                        },
                    )

            # Generate short entry conditions based on adversarial confidences
            for decrease_level, probability in adversarial_confidences.items():
                if probability >= 0.7:  # High probability threshold
                    entry_signals["short_conditions"].append(
                        {
                            "type": "ml_high_probability_decrease",
                            "probability": probability,
                            "price_decrease_target": float(decrease_level),
                            "regime": regime,
                        },
                    )

            # Adjust confidence based on regime
            if regime == "BULL":
                entry_signals["confidence"] *= 1.1
            elif regime == "BEAR":
                entry_signals["confidence"] *= 0.9
            elif regime == "SIDEWAYS":
                entry_signals["confidence"] *= 0.7

            return entry_signals

        except Exception as e:
            self.logger.error(f"Error generating entry signals: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="exit signal generation",
    )
    async def _generate_exit_signals(
        self,
        regime_info: dict[str, Any],
        ml_predictions: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Generate exit signals based on regime and ML predictions.

        Args:
            regime_info: Market regime information
            ml_predictions: ML predictions

        Returns:
            Optional[Dict[str, Any]]: Exit signals
        """
        try:
            regime = regime_info["regime"]
            adversarial_confidences = ml_predictions.get("adversarial_confidences", {})

            exit_signals = {
                "take_profit_levels": [],
                "stop_loss_levels": [],
                "trailing_stop": True,
                "time_based_exit": False,
            }

            # Calculate take profit levels based on ML predictions
            confidence_scores = ml_predictions.get("confidence_scores", {})
            for increase_level, confidence in confidence_scores.items():
                if confidence >= 0.6:
                    exit_signals["take_profit_levels"].append(
                        {
                            "level": 1 + (float(increase_level) / 100),
                            "weight": confidence,
                            "regime": regime,
                        },
                    )

            # Calculate stop loss levels based on adversarial confidences
            for decrease_level, probability in adversarial_confidences.items():
                if probability >= 0.6:
                    exit_signals["stop_loss_levels"].append(
                        {
                            "level": 1 - (float(decrease_level) / 100),
                            "weight": probability,
                            "regime": regime,
                        },
                    )

            # Adjust based on regime
            if regime == "SIDEWAYS":
                exit_signals["trailing_stop"] = True
                exit_signals["time_based_exit"] = True

            return exit_signals

        except Exception as e:
            self.logger.error(f"Error generating exit signals: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=0.5,
        context="confidence score calculation",
    )
    def _calculate_confidence_score(
        self,
        regime_info: dict[str, Any],
        ml_predictions: dict[str, Any],
    ) -> float:
        """
        Calculate confidence score for the strategy.

        Args:
            regime_info: Market regime information
            ml_predictions: ML predictions

        Returns:
            float: Confidence score between 0 and 1
        """
        try:
            confidence = 0.5  # Base confidence

            # Adjust based on regime confidence
            regime_confidence = regime_info.get("confidence", 0.5)
            confidence += (regime_confidence - 0.5) * 0.3

            # Adjust based on ML predictions
            confidence_scores = ml_predictions.get("confidence_scores", {})
            if confidence_scores:
                avg_ml_confidence = sum(confidence_scores.values()) / len(
                    confidence_scores,
                )
                confidence += (avg_ml_confidence - 0.5) * 0.4

            # Adjust based on regime type
            regime = regime_info.get("regime", "SIDEWAYS")
            if regime == "BULL":
                confidence += 0.1
            elif regime == "BEAR":
                confidence -= 0.1
            elif regime == "SIDEWAYS":
                confidence -= 0.2

            return max(0.0, min(1.0, confidence))

        except Exception as e:
            self.logger.error(f"Error calculating confidence score: {e}")
            return 0.5

    def _generate_fallback_predictions(self, current_price: float) -> dict[str, Any]:
        """
        Generate fallback predictions when ML model is unavailable.

        Args:
            current_price: Current market price

        Returns:
            Dict[str, Any]: Fallback predictions
        """
        try:
            return {
                "price_target_confidences": {
                    "0.5%": 0.6,  # 0.5% increase with 60% confidence
                    "1.0%": 0.5,  # 1.0% increase with 50% confidence
                    "1.5%": 0.4,  # 1.5% increase with 40% confidence
                },
                "adversarial_confidences": {
                    "0.5%": 0.4,  # 0.5% decrease with 40% probability
                    "1.0%": 0.3,  # 1.0% decrease with 30% probability
                    "1.5%": 0.2,  # 1.5% decrease with 20% probability
                },
                "fallback": True,
                "generation_time": datetime.now(),
            }

        except Exception as e:
            self.logger.error(f"Error generating fallback predictions: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="strategy validation",
    )
    def _validate_strategy(self, strategy: dict[str, Any]) -> bool:
        """
        Validate generated strategy.

        Args:
            strategy: Strategy to validate

        Returns:
            bool: True if strategy is valid, False otherwise
        """
        try:
            required_keys = [
                "regime_info",
                "ml_predictions",
                "entry_signals",
                "exit_signals",
                "risk_parameters",
                "position_sizing",
                "confidence_score",
            ]
            for key in required_keys:
                if key not in strategy:
                    self.logger.error(f"Missing required strategy key: {key}")
                    return False

            # Validate confidence score
            confidence = strategy.get("confidence_score", 0)
            if confidence < self.strategy_config["min_confidence_threshold"]:
                self.logger.warning(f"Strategy confidence too low: {confidence}")
                return False

            # Note: Position sizing validation handled by tactician/position_sizer.py

            return True

        except Exception as e:
            self.logger.error(f"Error validating strategy: {e}")
            return False

    def get_current_strategy(self) -> dict[str, Any] | None:
        """
        Get current strategy.

        Returns:
            Optional[Dict[str, Any]]: Current strategy or None
        """
        return self.current_strategy.copy() if self.current_strategy else None

    def get_strategy_history(self) -> list[dict[str, Any]]:
        """
        Get strategy history.

        Returns:
            List[Dict[str, Any]]: Strategy history
        """
        return self.strategy_history.copy()

    def get_last_strategy_update(self) -> datetime | None:
        """
        Get last strategy update time.

        Returns:
            Optional[datetime]: Last update time or None
        """
        return self.last_strategy_update

    def get_current_regime(self) -> tuple[str | None, float]:
        """
        Get current market regime and confidence.

        Returns:
            Tuple[str | None, float]: Current regime and confidence
        """
        return self.current_regime, self.regime_confidence

    def get_strategy_performance(self) -> dict[str, float]:
        """
        Get strategy performance metrics.

        Returns:
            Dict[str, float]: Performance metrics
        """
        return self.strategy_performance.copy()

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="strategist cleanup",
    )
    async def stop(self) -> None:
        """Stop the strategist component."""
        self.logger.info("🛑 Stopping Strategist...")

        try:
            # Save strategy history
            if self.strategy_history:
                self.logger.info(
                    f"Saving {len(self.strategy_history)} strategy records",
                )

            # Clear current state
            self.current_strategy = None
            self.last_strategy_update = None
            self.current_regime = None
            self.regime_confidence = 0.0

            self.logger.info("✅ Strategist stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping strategist: {e}")
