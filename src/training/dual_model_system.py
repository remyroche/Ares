# src/training/dual_model_system.py

import os
from datetime import datetime
from typing import Any

import pandas as pd

# Import ML Confidence Predictor
from src.analyst.ml_confidence_predictor import MLConfidencePredictor
from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    execution_error,
    failed,
    initialization_error,
    invalid,
)
from src.utils.confidence import aggregate_directional_confidences


class DualModelSystem:
    """
    Dual Model System for trading decisions.

    Analyst Model: Decides IF we enter/exit a trade (multi-timeframe: 30m/15m/5m)
    Tactician Model: Decides WHEN we enter/exit a trade (1m timeframe)

    Both models use ml_confidence_predictor.py for predictions.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize Dual Model System.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("DualModelSystem")
        # Backward-compatibility shim for legacy self.print calls
        # to avoid AttributeError during transitional cleanup.
        if not hasattr(self, "print"):

            def _shim_print(message: str) -> None:
                try:
                    self.logger.error(str(message))
                except Exception:
                    pass

            self.print = _shim_print  # type: ignore[attr-defined]

        # Model state
        self.analyst_model: Any | None = None
        self.tactician_model: Any | None = None
        self.ml_confidence_predictor: MLConfidencePredictor | None = None
        self.is_initialized: bool = False

        # Configuration
        self.dual_model_config: dict[str, Any] = self.config.get(
            "dual_model_system",
            {},
        )

        # Analyst model configuration (IF decisions) - multi-timeframe
        self.analyst_timeframes: list[str] = self.dual_model_config.get(
            "analyst_timeframes",
            ["30m", "15m", "5m"],
        )
        self.analyst_confidence_threshold: float = self.dual_model_config.get(
            "analyst_confidence_threshold",
            0.5,  # ENTER signal threshold
        )

        # Tactician model configuration (WHEN decisions) - 1m timeframe
        self.tactician_timeframes: list[str] = self.dual_model_config.get(
            "tactician_timeframes",
            ["1m"],
        )
        self.tactician_confidence_threshold: float = self.dual_model_config.get(
            "tactician_confidence_threshold",
            0.6,  # Minimum average confidence for both models
        )

        # Signal management
        self.enter_signal_validity_duration: int = self.dual_model_config.get(
            "enter_signal_validity_duration",
            120,  # 2 minutes in seconds
        )
        self.signal_check_interval: int = self.dual_model_config.get(
            "signal_check_interval",
            10,  # 10 seconds
        )

        # Confidence thresholds for signals
        self.neutral_signal_threshold: float = self.dual_model_config.get(
            "neutral_signal_threshold",
            0.5,  # NEUTRAL signal when confidence drops below 0.5
        )
        self.close_signal_threshold: float = self.dual_model_config.get(
            "close_signal_threshold",
            0.4,  # CLOSE signal when confidence drops below 0.4
        )

        # Position management thresholds
        self.position_close_confidence_threshold: float = self.dual_model_config.get(
            "position_close_confidence_threshold",
            0.6,  # Close positions when tactician confidence drops below 0.6
        )

        # Signal tracking
        self.current_enter_signal: dict[str, Any] | None = None
        self.signal_history: list[dict[str, Any]] = []

        # Ensemble configuration
        self.enable_ensemble_analysis: bool = self.dual_model_config.get(
            "enable_ensemble_analysis",
            True,
        )

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid dual model system configuration"),
            AttributeError: (False, "Missing required dual model parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="dual model system initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize Dual Model System with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Dual Model System...")

            # Load dual model configuration
            await self._load_dual_model_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for dual model system")
                return False

            # Initialize ML Confidence Predictor
            await self._initialize_ml_confidence_predictor()

            # Initialize Analyst Model (multi-timeframe)
            await self._initialize_analyst_model()

            # Initialize Tactician Model (1m timeframe)
            await self._initialize_tactician_model()

            self.is_initialized = True
            self.logger.info(
                "✅ Dual Model System initialization completed successfully",
            )
            return True

        except Exception:
            self.logger.exception("❌ Dual Model System initialization failed")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="dual model configuration loading",
    )
    async def _load_dual_model_configuration(self) -> None:
        """Load dual model configuration."""
        try:
            # Set default dual model parameters
            self.dual_model_config.setdefault(
                "analyst_timeframes",
                ["30m", "15m", "5m"],
            )
            self.dual_model_config.setdefault("tactician_timeframes", ["1m"])
            self.dual_model_config.setdefault("analyst_confidence_threshold", 0.5)
            self.dual_model_config.setdefault("tactician_confidence_threshold", 0.6)
            self.dual_model_config.setdefault("enter_signal_validity_duration", 120)
            self.dual_model_config.setdefault("signal_check_interval", 10)
            self.dual_model_config.setdefault("neutral_signal_threshold", 0.5)
            self.dual_model_config.setdefault("close_signal_threshold", 0.4)
            self.dual_model_config.setdefault(
                "position_close_confidence_threshold",
                0.6,
            )
            self.dual_model_config.setdefault("enable_ensemble_analysis", True)

            # Update configuration
            self.analyst_timeframes = self.dual_model_config["analyst_timeframes"]
            self.tactician_timeframes = self.dual_model_config["tactician_timeframes"]
            self.analyst_confidence_threshold = self.dual_model_config[
                "analyst_confidence_threshold"
            ]
            self.tactician_confidence_threshold = self.dual_model_config[
                "tactician_confidence_threshold"
            ]
            self.enter_signal_validity_duration = self.dual_model_config[
                "enter_signal_validity_duration"
            ]
            self.signal_check_interval = self.dual_model_config["signal_check_interval"]
            self.neutral_signal_threshold = self.dual_model_config[
                "neutral_signal_threshold"
            ]
            self.close_signal_threshold = self.dual_model_config[
                "close_signal_threshold"
            ]
            self.position_close_confidence_threshold = self.dual_model_config[
                "position_close_confidence_threshold"
            ]
            self.enable_ensemble_analysis = self.dual_model_config[
                "enable_ensemble_analysis"
            ]

            self.logger.info("Dual model configuration loaded successfully")

        except Exception as e:
            error_msg = f"Error loading dual model configuration: {e}"
            self.logger.error(error_msg)

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate dual model configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate analyst timeframes
            if not self.analyst_timeframes:
                self.logger.error("Analyst timeframes cannot be empty")
                return False

            # Validate tactician timeframes
            if not self.tactician_timeframes:
                self.logger.error("Tactician timeframes cannot be empty")
                return False

            # Validate confidence thresholds
            if not (0.0 <= self.analyst_confidence_threshold <= 1.0):
                self.logger.error(
                    "Analyst confidence threshold must be between 0 and 1",
                )
                return False

            if not (0.0 <= self.tactician_confidence_threshold <= 1.0):
                self.logger.error(
                    "Tactician confidence threshold must be between 0 and 1",
                )
                return False

            # Validate signal validity duration
            if self.enter_signal_validity_duration <= 0:
                self.logger.error("Enter signal validity duration must be positive")
                return False

            # Validate signal check interval
            if self.signal_check_interval <= 0:
                self.logger.error("Signal check interval must be positive")
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            error_msg = f"Error validating dual model configuration: {e}"
            self.logger.error(error_msg)
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="ML confidence predictor initialization",
    )
    async def _initialize_ml_confidence_predictor(self) -> None:
        """Initialize ML Confidence Predictor with meta-labeling integration."""
        try:
            # Get configuration for ML confidence predictor with meta-labeling and feature engineering
            ml_config = self.config.get(
                "ml_confidence_predictor",
                {
                    "enhanced_training_integration": True,
                    "model_path": "models/ml_confidence_predictor",
                    "min_samples_for_training": 1000,
                    "confidence_threshold": 0.6,
                    "max_prediction_horizon": 1,
                    "meta_labeling": {
                        "enable_analyst_labels": True,
                        "enable_tactician_labels": True,
                        "pattern_detection": {
                            "volatility_threshold": 0.02,
                            "momentum_threshold": 0.01,
                            "volume_threshold": 1.5,
                        },
                        "entry_prediction": {
                            "prediction_horizon": 5,
                            "max_adverse_excursion": 0.02,
                        },
                    },
                    "feature_engineering": {
                        "enable_advanced_features": True,
                        "enable_multi_timeframe_features": True,
                        "enable_autoencoder_features": True,
                        "enable_legacy_features": True,
                        "feature_cache_duration": 300,  # 5 minutes
                        "enable_feature_selection": True,
                        "max_features": 500,
                        "multi_timeframe_feature_engineering": {
                            "enable_mtf_features": True,
                            "enable_timeframe_adaptation": True,
                        },
                    },
                    "enhanced_order_manager": {
                        "enable_enhanced_order_manager": True,
                        "enable_async_order_executor": True,
                        "enable_chase_micro_breakout": True,
                        "enable_limit_order_return": True,
                        "enable_partial_fill_management": True,
                        "max_order_retries": 3,
                        "order_timeout_seconds": 30,
                        "slippage_tolerance": 0.001,
                        "volume_threshold": 1.5,
                        "momentum_threshold": 0.02,
                        "execution_strategies": {
                            "immediate": {"max_slippage": 0.001, "timeout_seconds": 30},
                            "batch": {"batch_size": 0.1, "batch_interval": 5},
                            "twap": {"duration_minutes": 10, "intervals": 20},
                            "vwap": {"volume_threshold": 1.5, "price_deviation": 0.002},
                            "iceberg": {"iceberg_qty": 0.1, "display_qty": 0.01},
                            "adaptive": {
                                "dynamic_slippage": True,
                                "market_impact_aware": True,
                            },
                        },
                    },
                    "model_training": {
                        "enable_continuous_training": True,
                        "enable_adaptive_training": True,
                        "enable_incremental_training": True,
                        "training_interval_hours": 24,
                        "min_samples_for_retraining": 1000,
                        "performance_degradation_threshold": 0.1,
                        "enable_model_calibration": True,
                        "enable_ensemble_training": True,
                        "enable_regime_specific_training": True,
                        "enable_multi_timeframe_training": True,
                        "enable_dual_model_training": True,
                        "enable_confidence_calibration": True,
                        "training_strategies": {
                            "continuous": {"batch_size": 1000, "learning_rate": 0.001},
                            "adaptive": {
                                "dynamic_lr": True,
                                "performance_threshold": 0.7,
                            },
                            "incremental": {
                                "update_frequency": 100,
                                "memory_size": 10000,
                            },
                            "full": {"epochs": 100, "validation_split": 0.2},
                        },
                    },
                },
            )

            self.ml_confidence_predictor = MLConfidencePredictor(ml_config)
            await self.ml_confidence_predictor.initialize()

            self.logger.info(
                "✅ ML Confidence Predictor with meta-labeling initialized successfully",
            )

        except Exception:
            self.print(
                initialization_error("Error initializing ML Confidence Predictor: {e}"),
            )

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="analyst model initialization",
    )
    async def _initialize_analyst_model(self) -> None:
        """Initialize Analyst Model for IF decisions (multi-timeframe)."""
        try:
            # Load analyst model from training results
            analyst_model_path = "models/analyst_model.pkl"

            if os.path.exists(analyst_model_path):
                import pickle

                with open(analyst_model_path, "rb") as f:
                    self.analyst_model = pickle.load(f)
                self.logger.info("Analyst model loaded successfully")
            else:
                self.logger.warning(
                    "Analyst model not found, will use ML Confidence Predictor",
                )
                self.analyst_model = None

        except Exception as e:
            error_msg = f"Error initializing Analyst model: {e}"
            self.logger.exception(error_msg)
            self.print(initialization_error(error_msg))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="tactician model initialization",
    )
    async def _initialize_tactician_model(self) -> None:
        """Initialize Tactician Model for WHEN decisions (1m timeframe)."""
        try:
            # Load tactician model from training results
            tactician_model_path = "models/tactician_model.pkl"

            if os.path.exists(tactician_model_path):
                import pickle

                with open(tactician_model_path, "rb") as f:
                    self.tactician_model = pickle.load(f)
                self.logger.info("Tactician model loaded successfully")
            else:
                self.logger.warning(
                    "Tactician model not found, will use ML Confidence Predictor",
                )
                self.tactician_model = None

        except Exception as e:
            error_msg = f"Error initializing Tactician model: {e}"
            self.logger.exception(error_msg)
            self.print(initialization_error(error_msg))

    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid market data for decision making"),
            AttributeError: (None, "Models not properly initialized"),
        },
        default_return=None,
        context="dual model decision making",
    )
    async def make_trading_decision(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        current_position: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Make trading decision using dual model system.

        Args:
            market_data: Market data for analysis
            current_price: Current asset price
            current_position: Current position information (if any)

        Returns:
            Dictionary with trading decision
        """
        try:
            if not self.is_initialized:
                msg = "Dual Model System not initialized"
                raise ValueError(msg)

            self.logger.info("🎯 Making Dual Model Trading Decision")

            # Check if we have an open position for exit logic
            if current_position:
                return await self._make_exit_decision(
                    market_data,
                    current_price,
                    current_position,
                )

            return await self._make_entry_decision(market_data, current_price)

        except Exception as e:
            error_msg = f"Error making trading decision: {e}"
            self.logger.error(error_msg)
            self.print(error(error_msg))
            return self._get_fallback_decision()

    async def _make_entry_decision(
        self,
        market_data: pd.DataFrame,
        current_price: float,
    ) -> dict[str, Any]:
        """Make entry decision using dual model system."""
        try:
            # Step 1: Analyst Model - IF decision (multi-timeframe)
            analyst_decision = await self._get_analyst_decision(
                market_data,
                current_price,
            )

            # Check if we have a valid ENTER signal
            if not analyst_decision["should_trade"]:
                return {
                    "action": "HOLD",
                    "signal": "HOLD",
                    "reason": "Analyst model: No clear trading opportunity",
                    "analyst_confidence": analyst_decision["confidence"],
                    "timestamp": datetime.now().isoformat(),
                }

            # Step 2: Tactician Model - WHEN decision (1m timeframe)
            tactician_decision = await self._get_tactician_decision(
                market_data,
                current_price,
                analyst_decision,
            )

            # Calculate final confidence using the specified formula
            final_conf_agg = aggregate_directional_confidences([
                {"direction": analyst_decision.get("direction", "HOLD"), "confidence": float(analyst_decision.get("confidence", 0.0))},
                {"direction": (analyst_decision.get("direction", "HOLD") if tactician_decision.get("should_execute") else "HOLD"), "confidence": float(tactician_decision.get("confidence", 0.0))},
            ])
            final_confidence = float(final_conf_agg.get("confidence", 0.0))
            final_direction = final_conf_agg.get("direction", analyst_decision.get("direction", "HOLD"))

            # Determine if we should execute the trade
            should_execute = final_confidence > 0.216  # Minimum threshold

            if should_execute:
                # Store the ENTER signal
                self.current_enter_signal = {
                    "timestamp": datetime.now(),
                    "analyst_confidence": analyst_decision["confidence"],
                    "tactician_confidence": tactician_decision["confidence"],
                    "final_confidence": final_confidence,
                    "direction": final_direction,
                    "strategy": analyst_decision["strategy"],
                }

                # Combine decisions
                final_decision = {
                    "action": "ENTRY",
                    "signal": "ENTER",
                    "direction": final_direction,
                    "strategy": analyst_decision["strategy"],
                    "analyst_confidence": analyst_decision["confidence"],
                    "tactician_confidence": tactician_decision["confidence"],
                    "final_confidence": final_confidence,
                    "normalized_confidence": self._calculate_normalized_confidence(
                        final_confidence,
                    ),
                    "entry_timing": tactician_decision["timing_signal"],
                    "reason": f"Final confidence: {final_confidence:.3f} > 0.216",
                    "analyst_timeframes": self.analyst_timeframes,
                    "tactician_timeframes": self.tactician_timeframes,
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                final_decision = {
                    "action": "HOLD",
                    "signal": "HOLD",
                    "reason": f"Final confidence: {final_confidence:.3f} <= 0.216",
                    "analyst_confidence": analyst_decision["confidence"],
                    "tactician_confidence": tactician_decision["confidence"],
                    "final_confidence": final_confidence,
                    "timestamp": datetime.now().isoformat(),
                }

            return final_decision

        except Exception as e:
            error_msg = f"Error making entry decision: {e}"
            self.logger.error(error_msg)
            self.print(error(error_msg))
            return self._get_fallback_decision()

    async def _make_exit_decision(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        current_position: dict[str, Any],
    ) -> dict[str, Any]:
        """Make exit decision using dual model system."""
        try:
            # Step 1: Analyst Model - IF exit decision
            analyst_exit_decision = await self._get_analyst_exit_decision(
                market_data,
                current_price,
                current_position,
            )

            # Step 2: Tactician Model - WHEN exit decision
            tactician_exit_decision = await self._get_tactician_exit_decision(
                market_data,
                current_price,
                analyst_exit_decision,
            )

            # Determine exit signal based on analyst confidence
            analyst_confidence = analyst_exit_decision["confidence"]

            if analyst_confidence < self.close_signal_threshold:
                exit_signal = "CLOSE"
                exit_action = "EXIT"
            elif analyst_confidence < self.neutral_signal_threshold:
                exit_signal = "NEUTRAL"
                # Only close if tactician confidence is also low
                if (
                    tactician_exit_decision["confidence"]
                    < self.position_close_confidence_threshold
                ):
                    exit_action = "PARTIAL_EXIT"
                else:
                    exit_action = "HOLD_POSITION"
            else:
                exit_signal = "HOLD"
                exit_action = "HOLD_POSITION"

            # Combine decisions
            return {
                "action": exit_action,
                "signal": exit_signal,
                "exit_type": analyst_exit_decision["exit_type"],
                "strategy": analyst_exit_decision["strategy"],
                "analyst_confidence": analyst_exit_decision["confidence"],
                "tactician_confidence": tactician_exit_decision["confidence"],
                "exit_timing": tactician_exit_decision["timing_signal"],
                "exit_reason": tactician_exit_decision["reason"],
                "analyst_timeframes": self.analyst_timeframes,
                "tactician_timeframes": self.tactician_timeframes,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            error_msg = f"Error making exit decision: {e}"
            self.logger.error(error_msg)
            self.print(error(error_msg))
            return self._get_fallback_decision()

    async def _get_analyst_decision(
        self,
        market_data: pd.DataFrame,
        current_price: float,
    ) -> dict[str, Any]:
        """Get Analyst model decision for IF we should trade (multi-timeframe)."""
        try:
            # Use ML Confidence Predictor for analyst decision
            if self.ml_confidence_predictor:
                # Use the new dual model system prediction method
                analyst_predictions = (
                    await self.ml_confidence_predictor.predict_for_dual_model_system(
                        market_data=market_data,
                        current_price=current_price,
                        model_type="analyst",
                    )
                )

                if analyst_predictions:
                    return self._analyze_analyst_confidence(
                        analyst_predictions,
                        current_price,
                    )
                # Fallback to original method
                confidence_predictions = (
                    await self.ml_confidence_predictor.predict_confidence_table(
                        market_data,
                        current_price,
                    )
                )

                if confidence_predictions:
                    return self._analyze_analyst_confidence(
                        confidence_predictions,
                        current_price,
                    )

            # Fallback to model-based decision
            if self.analyst_model:
                return await self._get_model_based_analyst_decision(
                    market_data,
                    current_price,
                )

            # Final fallback
            return {
                "should_trade": False,
                "direction": "HOLD",
                "strategy": "UNKNOWN",
                "confidence": 0.5,
                "reason": "No analyst model available",
            }

        except Exception as e:
            self.print(error("Error getting analyst decision: {e}"))
            return {
                "should_trade": False,
                "direction": "HOLD",
                "strategy": "ERROR",
                "confidence": 0.0,
                "reason": f"Analyst decision error: {e}",
            }

    def _analyze_analyst_confidence(
        self,
        confidence_predictions: dict[str, Any],
        current_price: float,
    ) -> dict[str, Any]:
        """Analyze confidence predictions for analyst decision (multi-timeframe)."""
        try:
            price_target_confidences = confidence_predictions.get(
                "price_target_confidences",
                {},
            )
            adversarial_confidences = confidence_predictions.get(
                "adversarial_confidences",
                {},
            )

            # Find the highest confidence score for price action above 0.3%
            # where adversarial movement is less than 50% of it
            best_confidence = 0.0

            for target_str, confidence in price_target_confidences.items():
                target = float(target_str.replace("%", ""))

                # Check if target is above 0.3%
                if target >= 0.3:
                    # Find corresponding adversarial confidence
                    adversarial_key = f"{target}%"
                    adversarial_confidence = adversarial_confidences.get(
                        adversarial_key,
                        0.0,
                    )

                    # Check if adversarial movement is less than 50% of the target confidence
                    if adversarial_confidence < (confidence * 0.5):
                        best_confidence = max(confidence, best_confidence)

            # If no suitable target found, use overall confidence
            if best_confidence == 0.0:
                if price_target_confidences:
                    best_confidence = max(price_target_confidences.values())
                else:
                    best_confidence = 0.5

            # Determine direction and strategy
            if best_confidence > self.analyst_confidence_threshold:
                direction = "LONG"
                should_trade = True
                strategy = "BULLISH"
            elif best_confidence < (1 - self.analyst_confidence_threshold):
                direction = "SHORT"
                should_trade = True
                strategy = "BEARISH"
            else:
                direction = "HOLD"
                should_trade = False
                strategy = "NEUTRAL"

            return {
                "should_trade": should_trade,
                "direction": direction,
                "strategy": strategy,
                "confidence": best_confidence,
                "price_target_confidences": price_target_confidences,
                "adversarial_confidences": adversarial_confidences,
                "reason": f"Analyst confidence: {best_confidence:.2f}",
            }

        except Exception as e:
            self.print(error("Error analyzing analyst confidence: {e}"))
            return {
                "should_trade": False,
                "direction": "HOLD",
                "strategy": "ERROR",
                "confidence": 0.0,
                "reason": f"Confidence analysis error: {e}",
            }

    async def _get_model_based_analyst_decision(
        self,
        market_data: pd.DataFrame,
        current_price: float,
    ) -> dict[str, Any]:
        """Get analyst decision using ML confidence predictor with meta-labeling."""
        try:
            if not self.ml_confidence_predictor:
                return {
                    "should_trade": False,
                    "direction": "HOLD",
                    "strategy": "ERROR",
                    "confidence": 0.0,
                    "reason": "No ML confidence predictor available",
                }

            # Get predictions with meta-labeling for analyst timeframes
            analyst_predictions = {}
            analyst_meta_labels = {}

            for timeframe in self.analyst_timeframes:
                # Use meta-labeling enhanced predictions
                predictions = (
                    await self.ml_confidence_predictor.predict_with_meta_labeling(
                        market_data,
                        timeframe,
                    )
                )
                analyst_predictions[timeframe] = predictions

                # Extract meta-labels
                if "meta_labels" in predictions:
                    analyst_meta_labels[timeframe] = predictions["meta_labels"]

            # Analyze confidence across timeframes with meta-labeling
            decision = self._analyze_analyst_confidence(
                analyst_predictions,
                current_price,
            )

            # Add meta-labeling information
            decision["meta_labels"] = analyst_meta_labels
            decision["prediction_enhanced"] = True

            return decision

        except Exception as e:
            self.print(error("Error getting model-based analyst decision: {e}"))
            return {
                "should_trade": False,
                "direction": "HOLD",
                "strategy": "ERROR",
                "confidence": 0.0,
                "reason": f"Model-based decision error: {e}",
            }

    async def _get_tactician_decision(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        analyst_decision: dict[str, Any],
    ) -> dict[str, Any]:
        """Get Tactician model decision for WHEN we should trade (1m timeframe)."""
        try:
            # Use ML Confidence Predictor for tactician decision
            if self.ml_confidence_predictor:
                # Use the new dual model system prediction method
                tactician_predictions = (
                    await self.ml_confidence_predictor.predict_for_dual_model_system(
                        market_data=market_data,
                        current_price=current_price,
                        model_type="tactician",
                    )
                )

                if tactician_predictions:
                    return self._analyze_tactician_confidence(
                        tactician_predictions,
                        current_price,
                        analyst_decision,
                    )
                # Fallback to original method
                confidence_predictions = (
                    await self.ml_confidence_predictor.predict_confidence_table(
                        market_data,
                        current_price,
                    )
                )

                if confidence_predictions:
                    return self._analyze_tactician_confidence(
                        confidence_predictions,
                        current_price,
                        analyst_decision,
                    )

            # Fallback to model-based decision
            if self.tactician_model:
                return await self._get_model_based_tactician_decision(
                    market_data,
                    current_price,
                    analyst_decision,
                )

            # Final fallback
            return {
                "should_execute": False,
                "timing_signal": 0.5,
                "confidence": 0.5,
                "reason": "No tactician model available",
            }

        except Exception as e:
            self.print(error("Error getting tactician decision: {e}"))
            return {
                "should_execute": False,
                "timing_signal": 0.0,
                "confidence": 0.0,
                "reason": f"Tactician decision error: {e}",
            }

    def _analyze_tactician_confidence(
        self,
        confidence_predictions: dict[str, Any],
        current_price: float,
        analyst_decision: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze confidence predictions for tactician decision (1m timeframe)."""
        try:
            price_target_confidences = confidence_predictions.get(
                "price_target_confidences",
                {},
            )

            # Focus on short-term targets for timing (1m timeframe)
            short_term_targets = {
                k: v
                for k, v in price_target_confidences.items()
                if float(k.replace("%", "")) <= 1.0
            }

            if short_term_targets:
                # Calculate timing confidence from short-term targets
                timing_confidence = sum(short_term_targets.values()) / len(
                    short_term_targets,
                )
            else:
                timing_confidence = 0.5

            # Determine if we should execute based on timing confidence
            should_execute = timing_confidence > self.tactician_confidence_threshold

            # Adjust based on analyst direction
            direction = analyst_decision.get("direction", "HOLD")
            if direction == "SHORT":
                # For short positions, invert the confidence
                timing_confidence = 1.0 - timing_confidence
                should_execute = timing_confidence > self.tactician_confidence_threshold

            return {
                "should_execute": should_execute,
                "timing_signal": timing_confidence,
                "confidence": timing_confidence,
                "short_term_targets": short_term_targets,
                "reason": f"Tactician timing confidence: {timing_confidence:.2f}",
            }

        except Exception as e:
            self.print(error("Error analyzing tactician confidence: {e}"))
            return {
                "should_execute": False,
                "timing_signal": 0.0,
                "confidence": 0.0,
                "reason": f"Tactician confidence analysis error: {e}",
            }

    async def _get_model_based_tactician_decision(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        analyst_decision: dict[str, Any],
    ) -> dict[str, Any]:
        """Get tactician decision using ML confidence predictor with meta-labeling."""
        try:
            if not self.ml_confidence_predictor:
                return {
                    "should_execute": False,
                    "timing_signal": 0.0,
                    "confidence": 0.0,
                    "reason": "No ML confidence predictor available",
                }

            # Get predictions with meta-labeling for tactician (1m timeframe)
            tactician_predictions = (
                await self.ml_confidence_predictor.predict_with_meta_labeling(
                    market_data,
                    "1m",
                )
            )

            # Extract meta-labels
            tactician_meta_labels = tactician_predictions.get("meta_labels", {})

            # Analyze tactician confidence with meta-labeling
            decision = self._analyze_tactician_confidence(
                tactician_predictions,
                current_price,
                analyst_decision,
            )

            # Add meta-labeling information
            decision["meta_labels"] = tactician_meta_labels
            decision["prediction_enhanced"] = True

            return decision

        except Exception as e:
            self.print(error("Error getting model-based tactician decision: {e}"))
            return {
                "should_execute": False,
                "timing_signal": 0.0,
                "confidence": 0.0,
                "reason": f"Model-based tactician decision error: {e}",
            }

    async def _get_analyst_exit_decision(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        current_position: dict[str, Any],
    ) -> dict[str, Any]:
        """Get Analyst model exit decision."""
        try:
            # Use ML Confidence Predictor for exit analysis
            if self.ml_confidence_predictor:
                confidence_predictions = (
                    await self.ml_confidence_predictor.predict_confidence_table(
                        market_data,
                        current_price,
                    )
                )

                if confidence_predictions:
                    return self._analyze_analyst_exit_confidence(
                        confidence_predictions,
                        current_price,
                        current_position,
                    )

            # Fallback
            return {
                "should_exit": False,
                "exit_type": "HOLD",
                "strategy": "HOLD",
                "confidence": 0.5,
                "reason": "No clear exit signal",
            }

        except Exception as e:
            self.print(error("Error getting analyst exit decision: {e}"))
            return {
                "should_exit": False,
                "exit_type": "HOLD",
                "strategy": "ERROR",
                "confidence": 0.0,
                "reason": f"Analyst exit decision error: {e}",
            }

    def _analyze_analyst_exit_confidence(
        self,
        confidence_predictions: dict[str, Any],
        current_price: float,
        current_position: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze confidence predictions for analyst exit decision."""
        try:
            price_target_confidences = confidence_predictions.get(
                "price_target_confidences",
                {},
            )
            adversarial_confidences = confidence_predictions.get(
                "adversarial_confidences",
                {},
            )

            position_type = current_position.get("type", "LONG")

            # Analyze exit conditions based on position type
            if position_type == "LONG":
                # For long positions, check if we should take profit or stop loss
                if price_target_confidences:
                    # Check if we've reached profit targets
                    profit_targets = {
                        k: v
                        for k, v in price_target_confidences.items()
                        if float(k.replace("%", "")) > 0
                    }
                    if profit_targets:
                        max_profit_confidence = max(profit_targets.values())
                        if max_profit_confidence > self.analyst_confidence_threshold:
                            return {
                                "should_exit": True,
                                "exit_type": "TAKE_PROFIT",
                                "strategy": "PROFIT_TAKING",
                                "confidence": max_profit_confidence,
                                "reason": f"Profit target reached: {max_profit_confidence:.2f}",
                            }

                # Check stop loss conditions
                if adversarial_confidences:
                    stop_loss_confidence = max(adversarial_confidences.values())
                    if stop_loss_confidence > 0.7:  # High confidence for stop loss
                        return {
                            "should_exit": True,
                            "exit_type": "STOP_LOSS",
                            "strategy": "RISK_MANAGEMENT",
                            "confidence": stop_loss_confidence,
                            "reason": f"Stop loss triggered: {stop_loss_confidence:.2f}",
                        }

            elif position_type == "SHORT":
                # For short positions, check if we should take profit or stop loss
                if adversarial_confidences:
                    # Check if we've reached profit targets (price went down)
                    profit_targets = {
                        k: v
                        for k, v in adversarial_confidences.items()
                        if float(k.replace("%", "")) > 0
                    }
                    if profit_targets:
                        max_profit_confidence = max(profit_targets.values())
                        if max_profit_confidence > self.analyst_confidence_threshold:
                            return {
                                "should_exit": True,
                                "exit_type": "TAKE_PROFIT",
                                "strategy": "PROFIT_TAKING",
                                "confidence": max_profit_confidence,
                                "reason": f"Profit target reached: {max_profit_confidence:.2f}",
                            }

                # Check stop loss conditions (price went up)
                if price_target_confidences:
                    stop_loss_confidence = max(price_target_confidences.values())
                    if stop_loss_confidence > 0.7:  # High confidence for stop loss
                        return {
                            "should_exit": True,
                            "exit_type": "STOP_LOSS",
                            "strategy": "RISK_MANAGEMENT",
                            "confidence": stop_loss_confidence,
                            "reason": f"Stop loss triggered: {stop_loss_confidence:.2f}",
                        }

            return {
                "should_exit": False,
                "exit_type": "HOLD",
                "strategy": "HOLD",
                "confidence": 0.5,
                "reason": "No clear exit signal",
            }

        except Exception as e:
            self.print(error("Error analyzing analyst exit confidence: {e}"))
            return {
                "should_exit": False,
                "exit_type": "HOLD",
                "strategy": "ERROR",
                "confidence": 0.0,
                "reason": f"Exit confidence analysis error: {e}",
            }

    async def _get_tactician_exit_decision(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        analyst_exit_decision: dict[str, Any],
    ) -> dict[str, Any]:
        """Get Tactician model exit decision."""
        try:
            # Use ML Confidence Predictor for exit timing
            if self.ml_confidence_predictor:
                confidence_predictions = (
                    await self.ml_confidence_predictor.predict_confidence_table(
                        market_data,
                        current_price,
                    )
                )

                if confidence_predictions:
                    return self._analyze_tactician_exit_confidence(
                        confidence_predictions,
                        current_price,
                        analyst_exit_decision,
                    )

            # Fallback
            return {
                "should_execute": False,
                "timing_signal": 0.5,
                "confidence": 0.5,
                "reason": "No clear exit timing",
            }

        except Exception as e:
            self.print(error("Error getting tactician exit decision: {e}"))
            return {
                "should_execute": False,
                "timing_signal": 0.0,
                "confidence": 0.0,
                "reason": f"Tactician exit decision error: {e}",
            }

    def _analyze_tactician_exit_confidence(
        self,
        confidence_predictions: dict[str, Any],
        current_price: float,
        analyst_exit_decision: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze confidence predictions for tactician exit decision."""
        try:
            exit_type = analyst_exit_decision.get("exit_type", "HOLD")

            if exit_type == "TAKE_PROFIT":
                # For take profit, look for high confidence in short-term targets
                price_target_confidences = confidence_predictions.get(
                    "price_target_confidences",
                    {},
                )
                short_term_targets = {
                    k: v
                    for k, v in price_target_confidences.items()
                    if float(k.replace("%", "")) <= 0.5
                }

                if short_term_targets:
                    timing_confidence = max(short_term_targets.values())
                else:
                    timing_confidence = 0.5

            elif exit_type == "STOP_LOSS":
                # For stop loss, look for high confidence in adverse movements
                adversarial_confidences = confidence_predictions.get(
                    "adversarial_confidences",
                    {},
                )
                short_term_adversarial = {
                    k: v
                    for k, v in adversarial_confidences.items()
                    if float(k.replace("%", "")) <= 0.5
                }

                if short_term_adversarial:
                    timing_confidence = max(short_term_adversarial.values())
                else:
                    timing_confidence = 0.5
            else:
                timing_confidence = 0.5

            should_execute = timing_confidence > self.tactician_confidence_threshold

            return {
                "should_execute": should_execute,
                "timing_signal": timing_confidence,
                "confidence": timing_confidence,
                "reason": f"Exit timing confidence: {timing_confidence:.2f}",
            }

        except Exception as e:
            self.print(error("Error analyzing tactician exit confidence: {e}"))
            return {
                "should_execute": False,
                "timing_signal": 0.0,
                "confidence": 0.0,
                "reason": f"Exit confidence analysis error: {e}",
            }

    def _calculate_final_confidence(
        self,
        analyst_confidence: float,
        tactician_confidence: float,
    ) -> float:
        """Calculate final confidence using the specified formula."""
        try:
            # Final_Confidence = Calibrated_Analyst_Score * Calibrated_Tactician_Score^2
            return analyst_confidence * (tactician_confidence**2)

        except Exception as e:
            error_msg = f"Error calculating final confidence: {e}"
            self.logger.error(error_msg)
            self.print(error(error_msg))
            return 0.0

    def _calculate_normalized_confidence(self, final_confidence: float) -> float:
        """Calculate normalized confidence for position sizing."""
        try:
            # Normalized_Confidence = (Final_Confidence - 0.216) / 0.784
            normalized_confidence = (final_confidence - 0.216) / 0.784
            return max(0.0, min(1.0, normalized_confidence))  # Clamp between 0 and 1

        except Exception as e:
            error_msg = f"Error calculating normalized confidence: {e}"
            self.logger.error(error_msg)
            self.print(error(error_msg))
            return 0.0

    def is_enter_signal_valid(self) -> bool:
        """Check if the current ENTER signal is still valid (within 2 minutes)."""
        try:
            if self.current_enter_signal is None:
                return False

            signal_time = self.current_enter_signal["timestamp"]
            current_time = datetime.now()
            time_diff = (current_time - signal_time).total_seconds()

            return time_diff <= self.enter_signal_validity_duration

        except Exception as e:
            error_msg = f"Error checking enter signal validity: {e}"
            self.logger.error(error_msg)
            self.print(error(error_msg))
            return False

    def get_current_signal(self) -> dict[str, Any] | None:
        """Get the current signal information."""
        return self.current_enter_signal

    def clear_current_signal(self) -> None:
        """Clear the current signal."""
        self.current_enter_signal = None

    def _get_fallback_decision(self) -> dict[str, Any]:
        """Get fallback decision when models fail."""
        return {
            "action": "HOLD",
            "signal": "HOLD",
            "reason": "Fallback decision - models unavailable",
            "analyst_confidence": 0.0,
            "tactician_confidence": 0.0,
            "final_confidence": 0.0,
            "timestamp": datetime.now().isoformat(),
        }

    def _determine_execution_strategy(
        self,
        normalized_confidence: float,
        analyst_decision: dict[str, Any],
        tactician_decision: dict[str, Any],
    ) -> str:
        """Determine optimal order execution strategy based on confidence and market conditions."""
        try:
            # High confidence scenarios
            if normalized_confidence > 0.8:
                return "immediate"  # High confidence, execute immediately

            # Medium confidence scenarios
            if normalized_confidence > 0.5:
                # Check for volatility conditions
                if analyst_decision.get("volatility", "low") == "high":
                    return "twap"  # High volatility, use TWAP
                return "batch"  # Medium confidence, use batch execution

            # Low confidence scenarios
            if normalized_confidence > 0.2:
                return "vwap"  # Low confidence, use VWAP for better price

            # Very low confidence scenarios
            return "iceberg"  # Very low confidence, use iceberg to minimize impact

        except Exception as e:
            error_msg = f"Error determining execution strategy: {e}"
            self.logger.exception(error_msg)
            self.print(execution_error(error_msg))
            return "immediate"  # Default to immediate execution

    def _calculate_recommended_quantity(self, normalized_confidence: float) -> float:
        """Calculate recommended order quantity based on normalized confidence."""
        try:
            # Base quantity calculation using Kelly criterion
            base_quantity = 0.05  # 5% base position size

            # Scale by normalized confidence
            recommended_quantity = base_quantity * (1 + normalized_confidence)

            # Cap at maximum position size
            max_quantity = 0.3  # 30% maximum position size
            return min(recommended_quantity, max_quantity)

        except Exception as e:
            error_msg = f"Error calculating recommended quantity: {e}"
            self.logger.error(error_msg)
            self.print(error(error_msg))
            return 0.05  # Default to 5%

    def _calculate_recommended_leverage(self, normalized_confidence: float) -> float:
        """Calculate recommended leverage based on normalized confidence."""
        try:
            # Leverage range: 10x to 100x
            min_leverage = 10.0
            max_leverage = 100.0

            # Scale leverage by normalized confidence
            return min_leverage + (max_leverage - min_leverage) * normalized_confidence

        except Exception as e:
            error_msg = f"Error calculating recommended leverage: {e}"
            self.logger.error(error_msg)
            self.print(error(error_msg))
            return 20.0  # Default to 20x leverage

    def _determine_execution_priority(self, normalized_confidence: float) -> int:
        """Determine execution priority based on normalized confidence."""
        try:
            # Priority range: 1 (lowest) to 10 (highest)
            if normalized_confidence > 0.8:
                return 10  # Highest priority
            if normalized_confidence > 0.6:
                return 8  # High priority
            if normalized_confidence > 0.4:
                return 6  # Medium priority
            if normalized_confidence > 0.2:
                return 4  # Low priority
            return 2  # Lowest priority

        except Exception as e:
            error_msg = f"Error determining execution priority: {e}"
            self.logger.exception(error_msg)
            self.print(execution_error(error_msg))
            return 5  # Default to medium priority

    async def trigger_model_training(
        self,
        training_data: pd.DataFrame,
        training_type: str = "continuous",
        force_training: bool = False,
    ) -> dict[str, Any]:
        """
        Trigger model training for the dual model system.

        Args:
            training_data: Historical data for training
            training_type: Type of training ("continuous", "adaptive", "incremental", "full")
            force_training: Force training regardless of conditions

        Returns:
            Dictionary containing training results
        """
        try:
            if not self.ml_confidence_predictor:
                return {
                    "success": False,
                    "error": "ML confidence predictor not available",
                }

            # Trigger training through ML confidence predictor
            training_result = await self.ml_confidence_predictor.trigger_model_training(
                training_data,
                training_type,
                force_training,
            )

            if training_result.get("success", False):
                # Update system state after successful training
                await self._update_system_after_training(training_result)

            return training_result

        except Exception as e:
            self.print(error("Error triggering model training: {e}"))
            return {"success": False, "error": str(e)}

    async def _update_system_after_training(
        self,
        training_result: dict[str, Any],
    ) -> None:
        """Update system state after successful training."""
        try:
            # Refresh models
            if self.ml_confidence_predictor:
                await (
                    self.ml_confidence_predictor.refresh_models_from_enhanced_training()
                )

            # Update training state
            self.last_training_update = datetime.now()

            # Log training success
            self.logger.info(
                f"✅ Model training completed successfully: {training_result.get('training_type', 'unknown')}",
            )

        except Exception as e:
            error_msg = f"Error updating system after training: {e}"
            self.logger.error(error_msg)
            self.print(error(error_msg))

    def get_training_status(self) -> dict[str, Any]:
        """Get training status for the dual model system."""
        try:
            training_status = {}

            # Get ML confidence predictor training status
            if self.ml_confidence_predictor:
                training_status["ml_confidence_predictor"] = (
                    self.ml_confidence_predictor.get_training_status()
                )

            # Add dual model system specific training info
            training_status["dual_model_system"] = {
                "last_training_update": self.last_training_update.isoformat()
                if hasattr(self, "last_training_update") and self.last_training_update
                else None,
                "analyst_models_loaded": self.analyst_model is not None,
                "tactician_models_loaded": self.tactician_model is not None,
                "ml_confidence_predictor_loaded": self.ml_confidence_predictor
                is not None,
                "training_config": self.config.get("model_training", {}),
            }

            return training_status

        except Exception as e:
            self.print(error("Error getting training status: {e}"))
            return {"error": str(e)}

    async def update_model_performance(
        self,
        performance_metrics: dict[str, Any],
    ) -> None:
        """Update model performance metrics."""
        try:
            if self.ml_confidence_predictor:
                await self.ml_confidence_predictor.update_model_performance(
                    performance_metrics,
                )

            # Update dual model system performance tracking
            if not hasattr(self, "performance_history"):
                self.performance_history = []

            self.performance_history.append(
                {"timestamp": datetime.now(), "metrics": performance_metrics},
            )

            # Keep only last 100 performance records
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]

        except Exception as e:
            error_msg = f"Error updating model performance: {e}"
            self.logger.error(error_msg)
            self.print(error(error_msg))

    def should_trigger_training(self) -> bool:
        """Check if training should be triggered."""
        try:
            if self.ml_confidence_predictor:
                return self.ml_confidence_predictor._should_trigger_training()
            return False

        except Exception as e:
            error_msg = f"Error checking training trigger: {e}"
            self.logger.error(error_msg)
            self.print(error(error_msg))
            return False

    def get_system_info(self) -> dict[str, Any]:
        """Get information about the dual model system."""
        return {
            "analyst_timeframes": self.analyst_timeframes,
            "tactician_timeframes": self.tactician_timeframes,
            "analyst_confidence_threshold": self.analyst_confidence_threshold,
            "tactician_confidence_threshold": self.tactician_confidence_threshold,
            "enter_signal_validity_duration": self.enter_signal_validity_duration,
            "signal_check_interval": self.signal_check_interval,
            "neutral_signal_threshold": self.neutral_signal_threshold,
            "close_signal_threshold": self.close_signal_threshold,
            "position_close_confidence_threshold": self.position_close_confidence_threshold,
            "enable_ensemble_analysis": self.enable_ensemble_analysis,
            "is_initialized": self.is_initialized,
            "analyst_model_loaded": self.analyst_model is not None,
            "tactician_model_loaded": self.tactician_model is not None,
            "ml_confidence_predictor_loaded": self.ml_confidence_predictor is not None,
            "current_signal_valid": self.is_enter_signal_valid(),
            "description": "Dual model system for trading decisions",
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="dual model system cleanup",
    )
    async def stop(self) -> None:
        """Stop the dual model system."""
        self.logger.info("🛑 Stopping Dual Model System...")

        try:
            # Stop ML Confidence Predictor
            if self.ml_confidence_predictor:
                await self.ml_confidence_predictor.stop()

            # Clear models
            self.analyst_model = None
            self.tactician_model = None
            self.ml_confidence_predictor = None
            self.is_initialized = False
            self.current_enter_signal = None

            self.logger.info("✅ Dual Model System stopped successfully")

        except Exception as e:
            error_msg = f"Error stopping dual model system: {e}"
            self.logger.error(error_msg)
            self.print(error(error_msg))


# Global dual model system instance
dual_model_system: DualModelSystem | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="dual model system setup",
)
async def setup_dual_model_system(
    config: dict[str, Any] | None = None,
) -> DualModelSystem | None:
    """
    Setup global dual model system.

    Args:
        config: Optional configuration dictionary

    Returns:
        Optional[DualModelSystem]: Global dual model system instance
    """
    try:
        global dual_model_system

        if config is None:
            config = {
                "dual_model_system": {
                    "analyst_timeframes": ["30m", "15m", "5m"],
                    "tactician_timeframes": ["1m"],
                    "analyst_confidence_threshold": 0.5,
                    "tactician_confidence_threshold": 0.6,
                    "enter_signal_validity_duration": 120,
                    "signal_check_interval": 10,
                    "neutral_signal_threshold": 0.5,
                    "close_signal_threshold": 0.4,
                    "position_close_confidence_threshold": 0.6,
                    "enable_ensemble_analysis": True,
                },
            }

        # Create dual model system
        dual_model_system = DualModelSystem(config)

        # Initialize dual model system
        success = await dual_model_system.initialize()
        if success:
            return dual_model_system
        return None

    except Exception as e:
        print(f"Error setting up dual model system: {e}")
        return None
