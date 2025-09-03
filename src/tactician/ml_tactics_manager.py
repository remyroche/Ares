# src/tactician/ml_tactics_manager.py

import asyncio
import copy
from datetime import datetime
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from src.utils.centralized_decorators import validate_data_quality
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import failed, invalid, warning


class MLTacticsManager:
    """ML Tactics Manager responsible for ML-based tactics and decision making.

    This module handles all ML tactics logic and decision making.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize ML tactics manager.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("MLTacticsManager")

        # ML tactics state
        self.is_initialized: bool = False
        self.ml_predictions: dict[str, Any] = {}
        self.ml_decisions: dict[str, Any] = {}

        # Configuration from step17 optimization results
        self.ml_config: dict[str, Any] = self.config.get("ml_tactics_manager", {})

        # Load step17 optimized parameters
        step17_config = self.config.get("step17_optimization", {})
        ml_tactics_optimization = step17_config.get("ml_tactics", {})

        # Load optimized ML tactics parameters
        self.enable_ml_tactics: bool = ml_tactics_optimization.get(
            "enable_ml_tactics", True
        )
        self.confidence_threshold: float = ml_tactics_optimization.get(
            "confidence_threshold", 0.7
        )
        self.regime_threshold: float = ml_tactics_optimization.get(
            "regime_threshold", 0.6
        )

        # Load additional optimized parameters
        self.ml_weight: float = ml_tactics_optimization.get("ml_weight", 0.8)
        self.regime_weight: float = ml_tactics_optimization.get("regime_weight", 0.2)
        self.confidence_boost_factor: float = ml_tactics_optimization.get(
            "confidence_boost_factor", 1.2
        )
        self.risk_adjustment_factor: float = ml_tactics_optimization.get(
            "risk_adjustment_factor", 1.0
        )

        # NEW: Multi-output prediction models
        self.multi_output_models: dict[str, Any] = {}
        self.is_trained: bool = False
        self.last_training_time: datetime | None = None

        # NEW: Barrier configuration (50% and 25% of Analyst barriers)
        self.barrier_config = {
            "fifty_percent": {
                "profit_target_multiplier": 0.5,
                "stop_loss_multiplier": 0.5,
                "timeframe": "1m",  # Shorter timeframe than Analyst
            },
            "twenty_five_percent": {
                "profit_target_multiplier": 0.25,
                "stop_loss_multiplier": 0.25,
                "timeframe": "1m",  # Shorter timeframe than Analyst
            },
            "fifty_percent_5m": {
                "profit_target_multiplier": 0.5,
                "stop_loss_multiplier": 0.5,
                "timeframe": "5m",  # 5-minute timeframe
            },
            "twenty_five_percent_5m": {
                "profit_target_multiplier": 0.25,
                "stop_loss_multiplier": 0.25,
                "timeframe": "5m",  # 5-minute timeframe
            },
        }

        # NEW: Confidence thresholds for green light signals (MTF unified)
        self.green_light_thresholds = {
            "fifty_percent": ml_tactics_optimization.get(
                "fifty_percent_threshold", 0.75
            ),
            "twenty_five_percent": ml_tactics_optimization.get(
                "twenty_five_percent_threshold", 0.8
            ),
            "combined_threshold": ml_tactics_optimization.get(
                "combined_threshold", 0.7
            ),
        }

        # NEW: Exit thresholds (MTF unified)
        self.exit_thresholds = {
            "fifty_percent": ml_tactics_optimization.get(
                "exit_fifty_percent_threshold", 0.4
            ),
            "twenty_five_percent": ml_tactics_optimization.get(
                "exit_twenty_five_percent_threshold", 0.35
            ),
            "combined_exit_threshold": ml_tactics_optimization.get(
                "combined_exit_threshold", 0.45
            ),
        }

        # NEW: Combined confidence weights (Analyst + Tactician confidences)
        self.confidence_weights = {
            "analyst_weight": ml_tactics_optimization.get(
                "analyst_confidence_weight", 0.3
            ),
            "fifty_percent_1m_weight": ml_tactics_optimization.get(
                "fifty_percent_1m_weight", 0.25
            ),
            "twenty_five_percent_1m_weight": ml_tactics_optimization.get(
                "twenty_five_percent_1m_weight", 0.15
            ),
            "fifty_percent_5m_weight": ml_tactics_optimization.get(
                "fifty_percent_5m_weight", 0.2
            ),
            "twenty_five_percent_5m_weight": ml_tactics_optimization.get(
                "twenty_five_percent_5m_weight", 0.1
            ),
        }

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid ML tactics manager configuration"),
            AttributeError: (False, "Missing required ML tactics parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="ML tactics manager initialization",
    )
    async def initialize(self) -> bool:
        """Initialize ML tactics manager.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing ML Tactics Manager...")

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error(
                    invalid("Invalid configuration for ML tactics manager")
                )
                return False

            # Initialize ML models
            await self._initialize_ml_models()

            self.is_initialized = True
            self.logger.info("✅ ML Tactics Manager initialized successfully")
            return True

        except Exception as e:
            self.logger.error(
                failed(f"❌ ML Tactics Manager initialization failed: {e}")
            )
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """Validate ML tactics manager configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            if self.confidence_threshold <= 0 or self.confidence_threshold > 1:
                self.logger.error(invalid("Invalid confidence_threshold configuration"))
                return False

            if self.regime_threshold <= 0 or self.regime_threshold > 1:
                self.logger.error(invalid("Invalid regime_threshold configuration"))
                return False

            if self.ml_weight <= 0 or self.ml_weight > 1:
                self.logger.error(invalid("Invalid ml_weight configuration"))
                return False

            if self.regime_weight <= 0 or self.regime_weight > 1:
                self.logger.error(invalid("Invalid regime_weight configuration"))
                return False

            # Validate barrier configuration
            for barrier_type, config in self.barrier_config.items():
                if (
                    config["profit_target_multiplier"] <= 0
                    or config["stop_loss_multiplier"] <= 0
                ):
                    self.logger.error(
                        invalid(f"Invalid barrier configuration for {barrier_type}")
                    )
                    return False

            # Validate thresholds
            for threshold_type, threshold in self.green_light_thresholds.items():
                if threshold <= 0 or threshold > 1:
                    self.logger.error(
                        invalid(f"Invalid green light threshold for {threshold_type}")
                    )
                    return False

            for threshold_type, threshold in self.exit_thresholds.items():
                if threshold <= 0 or threshold > 1:
                    self.logger.error(
                        invalid(f"Invalid exit threshold for {threshold_type}")
                    )
                    return False

            # Validate confidence weights
            total_weight = sum(self.confidence_weights.values())
            if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
                self.logger.error(
                    invalid(f"Confidence weights must sum to 1.0, got {total_weight}")
                )
                return False

            for weight_name, weight in self.confidence_weights.items():
                if weight < 0 or weight > 1:
                    self.logger.error(
                        invalid(
                            f"Invalid confidence weight for {weight_name}: {weight}"
                        )
                    )
                    return False

            return True

        except Exception as e:
            self.logger.error(failed(f"Configuration validation failed: {e}"))
            return False

    def refresh_step17_configuration(self, step17_results: dict[str, Any]) -> None:
        """Refresh configuration from step17 optimization results. This method is called
        automatically when step17 completes.

        Args:
            step17_results: Step17 optimization results
        """
        try:
            if "ml_tactics" in step17_results:
                ml_tactics_optimization = step17_results["ml_tactics"]

                # Update ML tactics parameters
                self.enable_ml_tactics = ml_tactics_optimization.get(
                    "enable_ml_tactics", self.enable_ml_tactics
                )
                self.confidence_threshold = ml_tactics_optimization.get(
                    "confidence_threshold", self.confidence_threshold
                )
                self.regime_threshold = ml_tactics_optimization.get(
                    "regime_threshold", self.regime_threshold
                )

                # Update additional parameters
                self.ml_weight = ml_tactics_optimization.get(
                    "ml_weight", self.ml_weight
                )
                self.regime_weight = ml_tactics_optimization.get(
                    "regime_weight", self.regime_weight
                )
                self.confidence_boost_factor = ml_tactics_optimization.get(
                    "confidence_boost_factor", self.confidence_boost_factor
                )
                self.risk_adjustment_factor = ml_tactics_optimization.get(
                    "risk_adjustment_factor", self.risk_adjustment_factor
                )

                # Update barrier and threshold configurations
                self.barrier_config = {
                    "fifty_percent": {
                        "profit_target_multiplier": ml_tactics_optimization.get(
                            "fifty_percent_profit_target_multiplier", 0.5
                        ),
                        "stop_loss_multiplier": ml_tactics_optimization.get(
                            "fifty_percent_stop_loss_multiplier", 0.5
                        ),
                        "timeframe": ml_tactics_optimization.get(
                            "fifty_percent_timeframe", "1m"
                        ),
                    },
                    "twenty_five_percent": {
                        "profit_target_multiplier": ml_tactics_optimization.get(
                            "twenty_five_percent_profit_target_multiplier", 0.25
                        ),
                        "stop_loss_multiplier": ml_tactics_optimization.get(
                            "twenty_five_percent_stop_loss_multiplier", 0.25
                        ),
                        "timeframe": ml_tactics_optimization.get(
                            "twenty_five_percent_timeframe", "1m"
                        ),
                    },
                    "fifty_percent_5m": {
                        "profit_target_multiplier": ml_tactics_optimization.get(
                            "fifty_percent_5m_profit_target_multiplier", 0.5
                        ),
                        "stop_loss_multiplier": ml_tactics_optimization.get(
                            "fifty_percent_5m_stop_loss_multiplier", 0.5
                        ),
                        "timeframe": ml_tactics_optimization.get(
                            "fifty_percent_5m_timeframe", "5m"
                        ),
                    },
                    "twenty_five_percent_5m": {
                        "profit_target_multiplier": ml_tactics_optimization.get(
                            "twenty_five_percent_5m_profit_target_multiplier", 0.25
                        ),
                        "stop_loss_multiplier": ml_tactics_optimization.get(
                            "twenty_five_percent_5m_stop_loss_multiplier", 0.25
                        ),
                        "timeframe": ml_tactics_optimization.get(
                            "twenty_five_percent_5m_timeframe", "5m"
                        ),
                    },
                }
                self.green_light_thresholds = {
                    "fifty_percent": ml_tactics_optimization.get(
                        "fifty_percent_threshold", 0.75
                    ),
                    "twenty_five_percent": ml_tactics_optimization.get(
                        "twenty_five_percent_threshold", 0.8
                    ),
                    "combined_threshold": ml_tactics_optimization.get(
                        "combined_threshold", 0.7
                    ),
                }
                self.exit_thresholds = {
                    "fifty_percent": ml_tactics_optimization.get(
                        "exit_fifty_percent_threshold", 0.4
                    ),
                    "twenty_five_percent": ml_tactics_optimization.get(
                        "exit_twenty_five_percent_threshold", 0.35
                    ),
                    "combined_exit_threshold": ml_tactics_optimization.get(
                        "combined_exit_threshold", 0.45
                    ),
                }
                self.confidence_weights = {
                    "analyst_weight": ml_tactics_optimization.get(
                        "analyst_confidence_weight", 0.3
                    ),
                    "fifty_percent_1m_weight": ml_tactics_optimization.get(
                        "fifty_percent_1m_weight", 0.25
                    ),
                    "twenty_five_percent_1m_weight": ml_tactics_optimization.get(
                        "twenty_five_percent_1m_weight", 0.15
                    ),
                    "fifty_percent_5m_weight": ml_tactics_optimization.get(
                        "fifty_percent_5m_weight", 0.2
                    ),
                    "twenty_five_percent_5m_weight": ml_tactics_optimization.get(
                        "twenty_five_percent_5m_weight", 0.1
                    ),
                }

                self.logger.info(
                    "✅ ML tactics manager configuration refreshed from step17 results"
                )

        except Exception as e:
            self.logger.error(f"Error refreshing step17 configuration: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="ML models initialization",
    )
    async def _initialize_ml_models(self) -> bool:
        """Initialize multi-output prediction models.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing multi-output prediction models...")

            # Initialize models for each barrier type
            for barrier_type in [
                "fifty_percent",
                "twenty_five_percent",
                "fifty_percent_5m",
                "twenty_five_percent_5m",
            ]:
                self.multi_output_models[barrier_type] = {
                    "model": None,
                    "calibrator": None,
                    "is_trained": False,
                    "feature_importance": {},
                    "performance_metrics": {},
                }

            # Load pre-trained models if available
            await self._load_pretrained_models()

            # If no pre-trained models, use fallback models
            if not self.is_trained:
                self.logger.warning(
                    "No pre-trained models found, using fallback models"
                )
                await self._initialize_fallback_models()

            self.logger.info("✅ Multi-output prediction models initialized")
            return True

        except Exception as e:
            self.logger.error(failed(f"❌ ML models initialization failed: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="pre-trained models loading",
    )
    async def _load_pretrained_models(self) -> bool:
        """Load pre-trained multi-output models.

        Returns:
            bool: True if models loaded successfully, False otherwise
        """
        try:
            # This would load actual trained models from disk
            # For now, we'll use fallback models
            self.logger.info("Loading pre-trained models (fallback mode)")
            return False

        except Exception as e:
            self.logger.error(failed(f"❌ Failed to load pre-trained models: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="fallback models initialization",
    )
    async def _initialize_fallback_models(self) -> bool:
        """Initialize fallback models for testing.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing fallback models...")

            # Create simple fallback models for each barrier type
            for barrier_type in [
                "fifty_percent",
                "twenty_five_percent",
                "fifty_percent_5m",
                "twenty_five_percent_5m",
            ]:
                self.multi_output_models[barrier_type]["is_trained"] = True
                self.multi_output_models[barrier_type]["model"] = "fallback"

            self.is_trained = True
            self.logger.info("✅ Fallback models initialized")
            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Fallback models initialization failed: {e}"))
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid ML tactics parameters"),
            AttributeError: (False, "Missing ML tactics components"),
            KeyError: (False, "Missing required ML tactics data"),
        },
        default_return=False,
        context="ML tactics execution",
    )
    async def execute_ml_tactics(
        self,
        tactics_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute ML-based tactics.

        Args:
            tactics_input: ML tactics input parameters

        Returns:
            dict: ML tactics results
        """
        try:
            self.logger.info("🤖 Executing ML tactics...")

            # Validate tactics input
            if not self._validate_tactics_input(tactics_input):
                return {}

            # Get ML predictions
            ml_predictions = self._get_ml_predictions()

            if not ml_predictions:
                self.logger.warning(warning("⚠️ No ML predictions available"))
                return {}

            # Apply regime and location tactics
            regime_tactics = self._apply_regime_and_location_tactics(ml_predictions)

            # Make ML entry decisions
            entry_decisions = self._make_ml_entry_decisions(ml_predictions)

            # Make ML sizing decisions
            sizing_decisions = self._make_ml_sizing_decisions(ml_predictions)

            # Make ML leverage decisions
            leverage_decisions = self._make_ml_leverage_decisions(ml_predictions)

            # Make ML directional decisions
            directional_decisions = self._make_ml_directional_decisions(ml_predictions)

            # Make ML liquidation risk decisions
            liquidation_decisions = self._make_ml_liquidation_risk_decisions(
                ml_predictions
            )

            # Calculate position size
            position_size = await self._calculate_position_size(ml_predictions)

            # Calculate leverage
            leverage = await self._calculate_leverage(ml_predictions)

            # Combine all results
            ml_results = {
                "regime_tactics": regime_tactics,
                "entry_decisions": entry_decisions,
                "sizing_decisions": sizing_decisions,
                "leverage_decisions": leverage_decisions,
                "directional_decisions": directional_decisions,
                "liquidation_decisions": liquidation_decisions,
                "position_size": position_size,
                "leverage": leverage,
                "ml_predictions": ml_predictions,
                "timestamp": datetime.now(),
            }

            self.ml_decisions = ml_results
            self.logger.info("✅ ML tactics execution completed successfully")

            return ml_results

        except Exception as e:
            self.logger.error(failed(f"❌ ML tactics execution failed: {e}"))
            return {}

    @validate_data_quality(
        required_columns=None,  # This method validates dict input, not DataFrame
        min_rows=1,
        max_null_ratio=0.0,
        check_duplicates=False,
        check_timestamps=False,
        context="ML tactics input validation",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="ML tactics input validation",
    )
    def _validate_tactics_input(self, tactics_input: dict[str, Any]) -> bool:
        """Validate ML tactics input parameters.

        Args:
            tactics_input: ML tactics input parameters

        Returns:
            bool: True if input is valid, False otherwise
        """
        try:
            required_fields = ["symbol", "exchange", "timeframe", "current_price"]

            for field in required_fields:
                if field not in tactics_input:
                    self.logger.error(
                        f"Missing required ML tactics input field: {field}",
                    )
                    return False

            # Validate specific field values
            if tactics_input.get("current_price", 0) <= 0:
                self.logger.error(invalid("Invalid current_price value"))
                return False

            return True

        except Exception as e:
            self.logger.error(failed(f"ML tactics input validation failed: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="ML predictions retrieval",
    )
    def _get_ml_predictions(self) -> dict[str, Any] | None:
        """Get ML predictions.

        Returns:
            dict: ML predictions or None if not available
        """
        try:
            # This would typically retrieve ML predictions from the analyst or other sources
            # For now, return mock predictions
            return {
                "regime_prediction": {
                    "BULL_TREND": 0.7,
                    "BEAR_TREND": 0.2,
                    "SIDEWAYS_RANGE": 0.1,
                },
                "location_prediction": {
                    "NEAR_SUPPORT": 0.8,
                    "NEAR_RESISTANCE": 0.1,
                    "MIDDLE": 0.1,
                },
                "entry_prediction": {
                    "confidence": 0.85,
                    "direction": "LONG",
                    "strength": 0.8,
                },
                "sizing_prediction": {
                    "confidence": 0.75,
                    "size_multiplier": 1.2,
                    "risk_level": "MEDIUM",
                },
                "leverage_prediction": {
                    "confidence": 0.7,
                    "leverage_multiplier": 1.5,
                    "risk_level": "HIGH",
                },
                "directional_prediction": {
                    "confidence": 0.8,
                    "direction": "UP",
                    "strength": 0.75,
                },
                "liquidation_risk_prediction": {
                    "confidence": 0.6,
                    "risk_level": "LOW",
                    "time_to_liquidation": 24,
                },
            }

        except Exception as e:
            self.logger.error(failed(f"❌ Failed to get ML predictions: {e}"))
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="regime and location tactics application",
    )
    def _apply_regime_and_location_tactics(
        self,
        regime_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply regime and location tactics.

        Args:
            regime_info: Regime information

        Returns:
            dict: Regime and location tactics
        """
        try:
            regime_prediction = regime_info.get("regime_prediction", {})
            location_prediction = regime_info.get("location_prediction", {})

            # Determine dominant regime
            dominant_regime = max(regime_prediction.items(), key=lambda x: x[1])[0]
            regime_confidence = regime_prediction.get(dominant_regime, 0)

            # Determine location
            dominant_location = max(location_prediction.items(), key=lambda x: x[1])[0]
            location_confidence = location_prediction.get(dominant_location, 0)

            # Apply regime-based tactics
            regime_tactics = self._get_regime_tactics(
                dominant_regime,
                regime_confidence,
            )

            # Apply location-based tactics
            location_tactics = self._get_location_tactics(
                dominant_location,
                location_confidence,
            )

            return {
                "dominant_regime": dominant_regime,
                "regime_confidence": regime_confidence,
                "dominant_location": dominant_location,
                "location_confidence": location_confidence,
                "regime_tactics": regime_tactics,
                "location_tactics": location_tactics,
                "combined_tactics": self._combine_regime_location_tactics(
                    regime_tactics,
                    location_tactics,
                ),
            }

        except Exception as e:
            self.logger.exception(
                f"❌ Regime and location tactics application failed: {e}",
            )
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="ML entry decisions making",
    )
    def _make_ml_entry_decisions(
        self,
        ml_predictions: dict[str, Any],
    ) -> dict[str, Any]:
        """Make ML-based entry decisions.

        Args:
            ml_predictions: ML predictions

        Returns:
            dict: Entry decisions
        """
        try:
            entry_prediction = ml_predictions.get("entry_prediction", {})

            confidence = entry_prediction.get("confidence", 0)
            direction = entry_prediction.get("direction", "NEUTRAL")
            strength = entry_prediction.get("strength", 0)

            # Determine entry decision based on confidence and direction
            if confidence >= self.confidence_threshold:
                if direction == "LONG" and strength > 0.6:
                    decision = "ENTER_LONG"
                elif direction == "SHORT" and strength > 0.6:
                    decision = "ENTER_SHORT"
                else:
                    decision = "HOLD"
            else:
                decision = "HOLD_LOW_CONFIDENCE"

            return {
                "decision": decision,
                "confidence": confidence,
                "direction": direction,
                "strength": strength,
                "reasoning": f"ML prediction: {direction} with {confidence:.2f} confidence",
            }

        except Exception as e:
            self.logger.error(failed(f"❌ ML entry decisions making failed: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="ML sizing decisions making",
    )
    def _make_ml_sizing_decisions(
        self,
        ml_predictions: dict[str, Any],
    ) -> dict[str, Any]:
        """Make ML-based sizing decisions.

        Args:
            ml_predictions: ML predictions

        Returns:
            dict: Sizing decisions
        """
        try:
            sizing_prediction = ml_predictions.get("sizing_prediction", {})

            confidence = sizing_prediction.get("confidence", 0)
            size_multiplier = sizing_prediction.get("size_multiplier", 1.0)
            risk_level = sizing_prediction.get("risk_level", "MEDIUM")

            # Determine sizing decision based on confidence and risk
            if confidence >= self.confidence_threshold:
                if risk_level == "LOW":
                    adjusted_multiplier = size_multiplier * 1.2
                elif risk_level == "HIGH":
                    adjusted_multiplier = size_multiplier * 0.8
                else:
                    adjusted_multiplier = size_multiplier

                decision = "ADJUST_SIZE"
            else:
                adjusted_multiplier = 1.0
                decision = "MAINTAIN_SIZE"

            return {
                "decision": decision,
                "confidence": confidence,
                "size_multiplier": adjusted_multiplier,
                "risk_level": risk_level,
                "reasoning": f"ML sizing: {adjusted_multiplier:.2f}x with {confidence:.2f} confidence",
            }

        except Exception as e:
            self.logger.error(failed(f"❌ ML sizing decisions making failed: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="ML leverage decisions making",
    )
    def _make_ml_leverage_decisions(
        self,
        ml_predictions: dict[str, Any],
    ) -> dict[str, Any]:
        """Make ML-based leverage decisions.

        Args:
            ml_predictions: ML predictions

        Returns:
            dict: Leverage decisions
        """
        try:
            leverage_prediction = ml_predictions.get("leverage_prediction", {})

            confidence = leverage_prediction.get("confidence", 0)
            leverage_multiplier = leverage_prediction.get("leverage_multiplier", 1.0)
            risk_level = leverage_prediction.get("risk_level", "MEDIUM")

            # Determine leverage decision based on confidence and risk
            if confidence >= self.confidence_threshold:
                if risk_level == "LOW":
                    adjusted_leverage = leverage_multiplier * 1.3
                elif risk_level == "HIGH":
                    adjusted_leverage = leverage_multiplier * 0.7
                else:
                    adjusted_leverage = leverage_multiplier

                decision = "ADJUST_LEVERAGE"
            else:
                adjusted_leverage = 1.0
                decision = "MAINTAIN_LEVERAGE"

            return {
                "decision": decision,
                "confidence": confidence,
                "leverage_multiplier": adjusted_leverage,
                "risk_level": risk_level,
                "reasoning": f"ML leverage: {adjusted_leverage:.2f}x with {confidence:.2f} confidence",
            }

        except Exception as e:
            self.logger.error(failed(f"❌ ML leverage decisions making failed: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="ML directional decisions making",
    )
    def _make_ml_directional_decisions(
        self,
        ml_predictions: dict[str, Any],
    ) -> dict[str, Any]:
        """Make ML-based directional decisions.

        Args:
            ml_predictions: ML predictions

        Returns:
            dict: Directional decisions
        """
        try:
            directional_prediction = ml_predictions.get("directional_prediction", {})

            confidence = directional_prediction.get("confidence", 0)
            direction = directional_prediction.get("direction", "NEUTRAL")
            strength = directional_prediction.get("strength", 0)

            # Determine directional decision based on confidence and direction
            if confidence >= self.confidence_threshold:
                if direction == "UP" and strength > 0.6:
                    decision = "BULLISH"
                elif direction == "DOWN" and strength > 0.6:
                    decision = "BEARISH"
                else:
                    decision = "NEUTRAL"
            else:
                decision = "UNCERTAIN"

            return {
                "decision": decision,
                "confidence": confidence,
                "direction": direction,
                "strength": strength,
                "reasoning": f"ML direction: {direction} with {confidence:.2f} confidence",
            }

        except Exception as e:
            self.logger.error(failed(f"❌ ML directional decisions making failed: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="ML liquidation risk decisions making",
    )
    def _make_ml_liquidation_risk_decisions(
        self,
        ml_predictions: dict[str, Any],
    ) -> dict[str, Any]:
        """Make ML-based liquidation risk decisions.

        Args:
            ml_predictions: ML predictions

        Returns:
            dict: Liquidation risk decisions
        """
        try:
            liquidation_prediction = ml_predictions.get(
                "liquidation_risk_prediction",
                {},
            )

            confidence = liquidation_prediction.get("confidence", 0)
            risk_level = liquidation_prediction.get("risk_level", "MEDIUM")
            time_to_liquidation = liquidation_prediction.get("time_to_liquidation", 24)

            # Determine liquidation risk decision based on confidence and risk
            if confidence >= self.confidence_threshold:
                if risk_level == "HIGH":
                    decision = "REDUCE_POSITION"
                elif risk_level == "MEDIUM":
                    decision = "MONITOR_CLOSELY"
                else:
                    decision = "MAINTAIN_POSITION"
            else:
                decision = "UNCERTAIN_RISK"

            return {
                "decision": decision,
                "confidence": confidence,
                "risk_level": risk_level,
                "time_to_liquidation": time_to_liquidation,
                "reasoning": f"ML liquidation risk: {risk_level} with {confidence:.2f} confidence",
            }

        except Exception as e:
            self.logger.exception(
                f"❌ ML liquidation risk decisions making failed: {e}",
            )
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="position size calculation",
    )
    async def _calculate_position_size(
        self,
        ml_predictions: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate position size based on ML predictions.

        Args:
            ml_predictions: ML predictions

        Returns:
            dict: Position size calculation results
        """
        try:
            sizing_decisions = self._make_ml_sizing_decisions(ml_predictions)

            base_position_size = 0.05  # 5% base position size
            size_multiplier = sizing_decisions.get("size_multiplier", 1.0)

            calculated_size = base_position_size * size_multiplier

            # Apply risk limits
            max_position_size = 0.3  # 30% maximum position size
            calculated_size = min(calculated_size, max_position_size)

            return {
                "base_size": base_position_size,
                "size_multiplier": size_multiplier,
                "calculated_size": calculated_size,
                "max_size": max_position_size,
                "confidence": sizing_decisions.get("confidence", 0),
                "decision": sizing_decisions.get("decision", "MAINTAIN_SIZE"),
            }

        except Exception as e:
            self.logger.error(failed(f"❌ Position size calculation failed: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="leverage calculation",
    )
    async def _calculate_leverage(
        self,
        ml_predictions: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate leverage based on ML predictions.

        Args:
            ml_predictions: ML predictions

        Returns:
            dict: Leverage calculation results
        """
        try:
            leverage_decisions = self._make_ml_leverage_decisions(ml_predictions)

            base_leverage = 1.0  # 1x base leverage
            leverage_multiplier = leverage_decisions.get("leverage_multiplier", 1.0)

            calculated_leverage = base_leverage * leverage_multiplier

            # Apply leverage limits
            max_leverage = 10.0  # 10x maximum leverage
            calculated_leverage = min(calculated_leverage, max_leverage)

            return {
                "base_leverage": base_leverage,
                "leverage_multiplier": leverage_multiplier,
                "calculated_leverage": calculated_leverage,
                "max_leverage": max_leverage,
                "confidence": leverage_decisions.get("confidence", 0),
                "decision": leverage_decisions.get("decision", "MAINTAIN_LEVERAGE"),
            }

        except Exception as e:
            self.logger.error(failed(f"❌ Leverage calculation failed: {e}"))
            return {}

    # Helper methods for regime and location tactics

    def _get_regime_tactics(self, regime: str, confidence: float) -> dict[str, Any]:
        """Get tactics for a specific regime."""
        tactics = {
            "BULL_TREND": {"position_multiplier": 1.2, "risk_tolerance": "HIGH"},
            "BEAR_TREND": {"position_multiplier": 0.8, "risk_tolerance": "LOW"},
            "SIDEWAYS_RANGE": {"position_multiplier": 1.0, "risk_tolerance": "MEDIUM"},
        }
        return tactics.get(
            regime,
            {"position_multiplier": 1.0, "risk_tolerance": "MEDIUM"},
        )

    def _get_location_tactics(self, location: str, confidence: float) -> dict[str, Any]:
        """Get tactics for a specific location."""
        tactics = {
            "NEAR_SUPPORT": {"entry_aggression": "HIGH", "stop_distance": "TIGHT"},
            "NEAR_RESISTANCE": {"entry_aggression": "LOW", "stop_distance": "WIDE"},
            "MIDDLE": {"entry_aggression": "MEDIUM", "stop_distance": "MEDIUM"},
        }
        return tactics.get(
            location,
            {"entry_aggression": "MEDIUM", "stop_distance": "MEDIUM"},
        )

    def _combine_regime_location_tactics(
        self,
        regime_tactics: dict[str, Any],
        location_tactics: dict[str, Any],
    ) -> dict[str, Any]:
        """Combine regime and location tactics."""
        return {
            "position_multiplier": regime_tactics.get("position_multiplier", 1.0),
            "risk_tolerance": regime_tactics.get("risk_tolerance", "MEDIUM"),
            "entry_aggression": location_tactics.get("entry_aggression", "MEDIUM"),
            "stop_distance": location_tactics.get("stop_distance", "MEDIUM"),
        }

    def get_ml_decisions(self) -> dict[str, Any]:
        """Get the latest ML decisions.

        Returns:
            dict: ML decisions
        """
        return self.ml_decisions.copy()

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="ML tactics manager cleanup",
    )
    async def stop(self) -> None:
        """Stop the ML tactics manager and cleanup resources."""
        try:
            self.logger.info("🛑 Stopping ML Tactics Manager...")
            self.is_initialized = False
            self.logger.info("✅ ML Tactics Manager stopped successfully")

        except Exception as e:
            self.logger.error(failed(f"❌ Failed to stop ML Tactics Manager: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="ML tactics manager cleanup",
    )
    async def cleanup(self) -> None:
        """Cleanup ML tactics manager resources."""
        try:
            self.logger.info("Cleaning up ML Tactics Manager...")
            await self.stop()
            self.ml_decisions.clear()
            self.logger.info("✅ ML Tactics Manager cleanup completed")
        except Exception as e:
            self.logger.error(f"Error cleaning up ML Tactics Manager: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="multi-output predictions generation",
    )
    async def generate_multi_output_predictions(
        self,
        market_data: pd.DataFrame,
        analyst_barriers: dict[str, float],
        symbol: str,
        timeframe: str,
        analyst_confidence: float = 0.5,
    ) -> dict[str, Any]:
        """Generate multi-output predictions for 50% and 25% barriers.

        Args:
            market_data: Market data with OHLCV
            analyst_barriers: Analyst's barrier values (for reference)
            symbol: Trading symbol
            timeframe: Current timeframe

        Returns:
            dict: Multi-output predictions with confidence scores and directions
        """
        try:
            if not self.is_trained:
                self.logger.warning("Models not trained, using fallback predictions")
                return self._generate_fallback_predictions()

            # Calculate Tactician barriers (50% and 25% of Analyst barriers)
            tactician_barriers = self._calculate_tactician_barriers(analyst_barriers)

            # Generate predictions for each barrier type
            predictions = {}

            for barrier_type in [
                "fifty_percent",
                "twenty_five_percent",
                "fifty_percent_5m",
                "twenty_five_percent_5m",
            ]:
                barrier_prediction = await self._generate_barrier_prediction(
                    barrier_type=barrier_type,
                    market_data=market_data,
                    barriers=tactician_barriers[barrier_type],
                    symbol=symbol,
                    timeframe=timeframe,
                )

                if barrier_prediction:
                    predictions[barrier_type] = barrier_prediction

            # Calculate combined confidence and green light signal
            combined_confidence = self._calculate_combined_confidence(
                predictions, analyst_confidence
            )
            green_light_signal = self._evaluate_green_light_signal(
                predictions, combined_confidence
            )

            # Add metadata
            result = {
                **predictions,
                "combined_confidence": combined_confidence,
                "green_light_signal": green_light_signal,
                "metadata": {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "generation_timestamp": datetime.now().isoformat(),
                    "model_type": "tactician_multi_output",
                    "barrier_config": self.barrier_config,
                },
            }

            self.logger.info(
                f"Generated multi-output predictions for {symbol}: {green_light_signal['signal']}"
            )
            return result

        except Exception as e:
            self.logger.error(
                failed(f"❌ Multi-output predictions generation failed: {e}")
            )
            return self._generate_fallback_predictions()

    def _calculate_tactician_barriers(
        self, analyst_barriers: dict[str, float]
    ) -> dict[str, dict[str, float]]:
        """Calculate Tactician barriers as 50% and 25% of Analyst barriers.

        Args:
            analyst_barriers: Analyst's barrier values

        Returns:
            dict: Tactician barriers for 50% and 25% levels
        """
        try:
            # Extract Analyst barriers
            analyst_upper = analyst_barriers.get("upper_barrier", 0.02)
            analyst_lower = analyst_barriers.get("lower_barrier", -0.01)

            tactician_barriers = {}

            # Calculate 50% barriers
            tactician_barriers["fifty_percent"] = {
                "upper_barrier": analyst_upper
                * self.barrier_config["fifty_percent"]["profit_target_multiplier"],
                "lower_barrier": analyst_lower
                * self.barrier_config["fifty_percent"]["stop_loss_multiplier"],
                "timeframe": self.barrier_config["fifty_percent"]["timeframe"],
            }

            # Calculate 25% barriers
            tactician_barriers["twenty_five_percent"] = {
                "upper_barrier": analyst_upper
                * self.barrier_config["twenty_five_percent"][
                    "profit_target_multiplier"
                ],
                "lower_barrier": analyst_lower
                * self.barrier_config["twenty_five_percent"]["stop_loss_multiplier"],
                "timeframe": self.barrier_config["twenty_five_percent"]["timeframe"],
            }

            # Calculate 50% barriers (5m)
            tactician_barriers["fifty_percent_5m"] = {
                "upper_barrier": analyst_upper
                * self.barrier_config["fifty_percent_5m"]["profit_target_multiplier"],
                "lower_barrier": analyst_lower
                * self.barrier_config["fifty_percent_5m"]["stop_loss_multiplier"],
                "timeframe": self.barrier_config["fifty_percent_5m"]["timeframe"],
            }

            # Calculate 25% barriers (5m)
            tactician_barriers["twenty_five_percent_5m"] = {
                "upper_barrier": analyst_upper
                * self.barrier_config["twenty_five_percent_5m"][
                    "profit_target_multiplier"
                ],
                "lower_barrier": analyst_lower
                * self.barrier_config["twenty_five_percent_5m"]["stop_loss_multiplier"],
                "timeframe": self.barrier_config["twenty_five_percent_5m"]["timeframe"],
            }

            return tactician_barriers

        except Exception as e:
            self.logger.error(failed(f"❌ Barrier calculation failed: {e}"))
            return {
                "fifty_percent": {
                    "upper_barrier": 0.01,
                    "lower_barrier": -0.005,
                    "timeframe": "1m",
                },
                "twenty_five_percent": {
                    "upper_barrier": 0.005,
                    "lower_barrier": -0.0025,
                    "timeframe": "1m",
                },
                "fifty_percent_5m": {
                    "upper_barrier": 0.01,
                    "lower_barrier": -0.005,
                    "timeframe": "5m",
                },
                "twenty_five_percent_5m": {
                    "upper_barrier": 0.005,
                    "lower_barrier": -0.0025,
                    "timeframe": "5m",
                },
            }

    async def _generate_barrier_prediction(
        self,
        barrier_type: str,
        market_data: pd.DataFrame,
        barriers: dict[str, float],
        symbol: str,
        timeframe: str,
    ) -> dict[str, Any]:
        """Generate prediction for a specific barrier type.

        Args:
            barrier_type: "fifty_percent" or "twenty_five_percent"
            market_data: Market data
            barriers: Barrier values
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            dict: Barrier prediction with confidence and direction
        """
        try:
            # Extract features from market data
            features = self._extract_features(market_data)

            # Generate prediction using model
            if self.multi_output_models[barrier_type]["model"] == "fallback":
                # Use fallback prediction
                confidence = self._generate_fallback_confidence(barrier_type, features)
                direction = self._determine_direction(features)
            else:
                # Use actual model prediction
                confidence = self._predict_with_model(barrier_type, features)
                direction = self._determine_direction(features)

            # Apply calibration if available
            if self.multi_output_models[barrier_type]["calibrator"]:
                confidence = self._calibrate_prediction(barrier_type, confidence)

            # Validate confidence
            confidence = np.clip(confidence, 0.0, 1.0)

            return {
                "confidence": confidence,
                "direction": direction,
                "upper_barrier": barriers["upper_barrier"],
                "lower_barrier": barriers["lower_barrier"],
                "timeframe": barriers["timeframe"],
                "barrier_type": barrier_type,
            }

        except Exception as e:
            self.logger.error(
                failed(f"❌ Barrier prediction failed for {barrier_type}: {e}")
            )
            return None

    def _extract_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Extract features from market data for prediction.

        Args:
            market_data: Market data with OHLCV

        Returns:
            np.ndarray: Feature array
        """
        try:
            features = []

            if len(market_data) < 20:
                # Not enough data, return default features
                return np.array([0.5] * 10)

            # Price-based features
            close_prices = market_data["close"].values
            high_prices = market_data["high"].values
            low_prices = market_data["low"].values
            volumes = market_data["volume"].values

            # Calculate technical indicators
            # Price momentum
            price_momentum = (close_prices[-1] - close_prices[-5]) / close_prices[-5]
            features.append(price_momentum)

            # Volatility
            returns = np.diff(close_prices) / close_prices[:-1]
            volatility = np.std(returns[-20:])
            features.append(volatility)

            # Volume trend
            volume_trend = (
                (volumes[-1] - volumes[-5]) / volumes[-5] if volumes[-5] > 0 else 0
            )
            features.append(volume_trend)

            # Price range
            price_range = (high_prices[-1] - low_prices[-1]) / close_prices[-1]
            features.append(price_range)

            # Moving averages
            ma_short = np.mean(close_prices[-5:])
            ma_long = np.mean(close_prices[-20:])
            ma_ratio = ma_short / ma_long if ma_long > 0 else 1.0
            features.append(ma_ratio)

            # RSI-like indicator
            gains = np.where(returns > 0, returns, 0)
            losses = np.where(returns < 0, -returns, 0)
            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
            rs = avg_gain / avg_loss if avg_loss > 0 else 1.0
            rsi = 100 - (100 / (1 + rs))
            features.append(rsi / 100)  # Normalize to 0-1

            # Additional features
            features.extend(
                [
                    close_prices[-1] / close_prices[-2] - 1,  # Latest return
                    (
                        np.mean(volumes[-5:]) / np.mean(volumes[-20:])
                        if np.mean(volumes[-20:]) > 0
                        else 1.0
                    ),  # Volume ratio
                    (high_prices[-1] - close_prices[-1])
                    / close_prices[-1],  # Upper shadow
                    (close_prices[-1] - low_prices[-1])
                    / close_prices[-1],  # Lower shadow
                ]
            )

            return np.array(features)

        except Exception as e:
            self.logger.error(failed(f"❌ Feature extraction failed: {e}"))
            return np.array([0.5] * 10)

    def _generate_fallback_confidence(
        self, barrier_type: str, features: np.ndarray
    ) -> float:
        """Generate fallback confidence score.

        Args:
            barrier_type: Barrier type
            features: Feature array

        Returns:
            float: Confidence score
        """
        try:
            # Simple heuristic-based confidence
            base_confidence = 0.5

            # Adjust based on price momentum
            if len(features) > 0:
                momentum = features[0]
                if abs(momentum) > 0.01:  # Strong momentum
                    base_confidence += 0.2
                elif abs(momentum) > 0.005:  # Moderate momentum
                    base_confidence += 0.1

            # Adjust based on volatility
            if len(features) > 1:
                volatility = features[1]
                if volatility < 0.01:  # Low volatility
                    base_confidence += 0.1
                elif volatility > 0.03:  # High volatility
                    base_confidence -= 0.1

            # Adjust based on RSI
            if len(features) > 5:
                rsi = features[5]
                if 0.3 < rsi < 0.7:  # Neutral RSI
                    base_confidence += 0.1
                elif rsi < 0.2 or rsi > 0.8:  # Extreme RSI
                    base_confidence -= 0.1

            # Adjust for barrier type (25% barriers need higher confidence)
            if barrier_type == "twenty_five_percent":
                base_confidence *= 0.9  # Slightly lower for smaller barriers

            return np.clip(base_confidence, 0.0, 1.0)

        except Exception as e:
            self.logger.error(failed(f"❌ Fallback confidence generation failed: {e}"))
            return 0.5

    def _determine_direction(self, features: np.ndarray) -> str:
        """Determine price direction based on features.

        Args:
            features: Feature array

        Returns:
            str: "UP" or "DOWN"
        """
        try:
            if len(features) > 0:
                momentum = features[0]
                if momentum > 0:
                    return "UP"
                else:
                    return "DOWN"
            else:
                return "UP"  # Default direction

        except Exception as e:
            self.logger.error(failed(f"❌ Direction determination failed: {e}"))
            return "UP"

    def _predict_with_model(self, barrier_type: str, features: np.ndarray) -> float:
        """Predict confidence using actual model.

        Args:
            barrier_type: Barrier type
            features: Feature array

        Returns:
            float: Confidence score
        """
        try:
            # This would use the actual trained model
            # For now, return fallback confidence
            return self._generate_fallback_confidence(barrier_type, features)

        except Exception as e:
            self.logger.error(failed(f"❌ Model prediction failed: {e}"))
            return 0.5

    def _calibrate_prediction(self, barrier_type: str, confidence: float) -> float:
        """Calibrate prediction using calibrator.

        Args:
            barrier_type: Barrier type
            confidence: Raw confidence

        Returns:
            float: Calibrated confidence
        """
        try:
            # This would use the actual calibrator
            # For now, return original confidence
            return confidence

        except Exception as e:
            self.logger.error(failed(f"❌ Prediction calibration failed: {e}"))
            return confidence

    def _calculate_combined_confidence(
        self, predictions: dict[str, Any], analyst_confidence: float = 0.5
    ) -> float:
        """Calculate combined confidence from Analyst and Tactician predictions.

        Args:
            predictions: Tactician predictions dictionary
            analyst_confidence: Analyst confidence score

        Returns:
            float: Combined confidence score
        """
        try:
            # Start with Analyst confidence
            combined_confidence = (
                analyst_confidence * self.confidence_weights["analyst_weight"]
            )

            # Add Tactician confidences with their respective weights
            for barrier_type, prediction in predictions.items():
                if prediction and "confidence" in prediction:
                    confidence = prediction["confidence"]

                    # Get weight for this barrier type
                    if barrier_type == "fifty_percent":
                        weight = self.confidence_weights["fifty_percent_1m_weight"]
                    elif barrier_type == "twenty_five_percent":
                        weight = self.confidence_weights[
                            "twenty_five_percent_1m_weight"
                        ]
                    elif barrier_type == "fifty_percent_5m":
                        weight = self.confidence_weights["fifty_percent_5m_weight"]
                    elif barrier_type == "twenty_five_percent_5m":
                        weight = self.confidence_weights[
                            "twenty_five_percent_5m_weight"
                        ]
                    else:
                        weight = 0.0

                    combined_confidence += confidence * weight

            return np.clip(combined_confidence, 0.0, 1.0)

        except Exception as e:
            self.logger.error(failed(f"❌ Combined confidence calculation failed: {e}"))
            return 0.5

    def _evaluate_green_light_signal(
        self, predictions: dict[str, Any], combined_confidence: float
    ) -> dict[str, Any]:
        """Evaluate green light signal based on predictions and thresholds.

        Args:
            predictions: Predictions dictionary
            combined_confidence: Combined confidence score

        Returns:
            dict: Green light signal evaluation
        """
        try:
            # Check individual barrier thresholds (MTF unified)
            fifty_percent_ok = False
            twenty_five_percent_ok = False

            # Check 50% barriers (both 1m and 5m)
            fifty_percent_confidences = []
            if "fifty_percent" in predictions and predictions["fifty_percent"]:
                fifty_percent_confidences.append(
                    predictions["fifty_percent"]["confidence"]
                )
            if "fifty_percent_5m" in predictions and predictions["fifty_percent_5m"]:
                fifty_percent_confidences.append(
                    predictions["fifty_percent_5m"]["confidence"]
                )

            if fifty_percent_confidences:
                fifty_percent_ok = (
                    max(fifty_percent_confidences)
                    >= self.green_light_thresholds["fifty_percent"]
                )

            # Check 25% barriers (both 1m and 5m)
            twenty_five_percent_confidences = []
            if (
                "twenty_five_percent" in predictions
                and predictions["twenty_five_percent"]
            ):
                twenty_five_percent_confidences.append(
                    predictions["twenty_five_percent"]["confidence"]
                )
            if (
                "twenty_five_percent_5m" in predictions
                and predictions["twenty_five_percent_5m"]
            ):
                twenty_five_percent_confidences.append(
                    predictions["twenty_five_percent_5m"]["confidence"]
                )

            if twenty_five_percent_confidences:
                twenty_five_percent_ok = (
                    max(twenty_five_percent_confidences)
                    >= self.green_light_thresholds["twenty_five_percent"]
                )

            # Check combined threshold
            combined_ok = (
                combined_confidence >= self.green_light_thresholds["combined_threshold"]
            )

            # Determine signal
            if fifty_percent_ok and twenty_five_percent_ok and combined_ok:
                signal = "GREEN_LIGHT"
                reason = "All thresholds met"
            elif combined_ok:
                signal = "YELLOW_LIGHT"
                reason = "Combined threshold met, individual thresholds partial"
            else:
                signal = "RED_LIGHT"
                reason = "Thresholds not met"

            return {
                "signal": signal,
                "reason": reason,
                "fifty_percent_ok": fifty_percent_ok,
                "twenty_five_percent_ok": twenty_five_percent_ok,
                "combined_ok": combined_ok,
                "combined_confidence": combined_confidence,
                "thresholds": self.green_light_thresholds,
            }

        except Exception as e:
            self.logger.error(failed(f"❌ Green light signal evaluation failed: {e}"))
            return {
                "signal": "RED_LIGHT",
                "reason": "Evaluation failed",
                "fifty_percent_ok": False,
                "twenty_five_percent_ok": False,
                "combined_ok": False,
                "combined_confidence": 0.0,
                "thresholds": self.green_light_thresholds,
            }

    def _generate_fallback_predictions(self) -> dict[str, Any]:
        """Generate fallback predictions when models are not available.

        Returns:
            dict: Fallback predictions
        """
        return {
            "fifty_percent": {
                "confidence": 0.5,
                "direction": "UP",
                "upper_barrier": 0.01,
                "lower_barrier": -0.005,
                "timeframe": "1m",
                "barrier_type": "fifty_percent",
            },
            "twenty_five_percent": {
                "confidence": 0.5,
                "direction": "UP",
                "upper_barrier": 0.005,
                "lower_barrier": -0.0025,
                "timeframe": "1m",
                "barrier_type": "twenty_five_percent",
            },
            "fifty_percent_5m": {
                "confidence": 0.5,
                "direction": "UP",
                "upper_barrier": 0.01,
                "lower_barrier": -0.005,
                "timeframe": "5m",
                "barrier_type": "fifty_percent_5m",
            },
            "twenty_five_percent_5m": {
                "confidence": 0.5,
                "direction": "UP",
                "upper_barrier": 0.005,
                "lower_barrier": -0.0025,
                "timeframe": "5m",
                "barrier_type": "twenty_five_percent_5m",
            },
            "combined_confidence": 0.5,
            "green_light_signal": {
                "signal": "RED_LIGHT",
                "reason": "Fallback mode",
                "fifty_percent_ok": False,
                "twenty_five_percent_ok": False,
                "combined_ok": False,
                "combined_confidence": 0.5,
                "thresholds": self.green_light_thresholds,
            },
            "metadata": {
                "model_type": "fallback",
                "generation_timestamp": datetime.now().isoformat(),
            },
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="exit signal evaluation",
    )
    async def evaluate_exit_signal(
        self, current_predictions: dict[str, Any], position_context: dict[str, Any]
    ) -> dict[str, Any]:
        """Evaluate exit signal based on current predictions and position context.

        Args:
            current_predictions: Current multi-output predictions
            position_context: Current position context

        Returns:
            dict: Exit signal evaluation
        """
        try:
            combined_confidence = current_predictions.get("combined_confidence", 0.5)

            # Check exit thresholds (MTF unified)
            fifty_percent_exit = False
            twenty_five_percent_exit = False

            # Check 50% barriers (both 1m and 5m)
            fifty_percent_confidences = []
            if (
                "fifty_percent" in current_predictions
                and current_predictions["fifty_percent"]
            ):
                fifty_percent_confidences.append(
                    current_predictions["fifty_percent"]["confidence"]
                )
            if (
                "fifty_percent_5m" in current_predictions
                and current_predictions["fifty_percent_5m"]
            ):
                fifty_percent_confidences.append(
                    current_predictions["fifty_percent_5m"]["confidence"]
                )

            if fifty_percent_confidences:
                fifty_percent_exit = (
                    min(fifty_percent_confidences)
                    <= self.exit_thresholds["fifty_percent"]
                )

            # Check 25% barriers (both 1m and 5m)
            twenty_five_percent_confidences = []
            if (
                "twenty_five_percent" in current_predictions
                and current_predictions["twenty_five_percent"]
            ):
                twenty_five_percent_confidences.append(
                    current_predictions["twenty_five_percent"]["confidence"]
                )
            if (
                "twenty_five_percent_5m" in current_predictions
                and current_predictions["twenty_five_percent_5m"]
            ):
                twenty_five_percent_confidences.append(
                    current_predictions["twenty_five_percent_5m"]["confidence"]
                )

            if twenty_five_percent_confidences:
                twenty_five_percent_exit = (
                    min(twenty_five_percent_confidences)
                    <= self.exit_thresholds["twenty_five_percent"]
                )

            combined_exit = (
                combined_confidence <= self.exit_thresholds["combined_exit_threshold"]
            )

            # Determine exit signal
            if combined_exit or (fifty_percent_exit and twenty_five_percent_exit):
                exit_signal = "EXIT"
                reason = "Confidence below exit thresholds"
            elif fifty_percent_exit or twenty_five_percent_exit:
                exit_signal = "PARTIAL_EXIT"
                reason = "Partial confidence below exit thresholds"
            else:
                exit_signal = "HOLD"
                reason = "Confidence above exit thresholds"

            return {
                "exit_signal": exit_signal,
                "reason": reason,
                "fifty_percent_exit": fifty_percent_exit,
                "twenty_five_percent_exit": twenty_five_percent_exit,
                "combined_exit": combined_exit,
                "combined_confidence": combined_confidence,
                "exit_thresholds": self.exit_thresholds,
            }

        except Exception as e:
            self.logger.error(failed(f"❌ Exit signal evaluation failed: {e}"))
            return {
                "exit_signal": "HOLD",
                "reason": "Evaluation failed",
                "fifty_percent_exit": False,
                "twenty_five_percent_exit": False,
                "combined_exit": False,
                "combined_confidence": 0.5,
                "exit_thresholds": self.exit_thresholds,
            }


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="ML tactics manager setup",
)
async def setup_ml_tactics_manager(
    config: dict[str, Any] | None = None,
) -> MLTacticsManager | None:
    """Setup and return a configured MLTacticsManager instance.

    Args:
        config: Configuration dictionary

    Returns:
        MLTacticsManager: Configured ML tactics manager instance
    """
    try:
        manager = MLTacticsManager(config or {})
        if await manager.initialize():
            return manager
        return None
    except Exception as e:
        system_logger.exception(failed(f"Failed to setup ML Tactics Manager: {e}"))
        return None
