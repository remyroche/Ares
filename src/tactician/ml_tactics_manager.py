# src/tactician/ml_tactics_manager.py

from datetime import datetime
from typing import Any

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    failed,
    invalid,
    warning,
)


class MLTacticsManager:
    """
    ML Tactics Manager responsible for ML-based tactics and decision making.
    This module handles all ML tactics logic and decision making.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize ML tactics manager.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("MLTacticsManager")

        # ML tactics state
        self.is_initialized: bool = False
        self.ml_predictions: dict[str, Any] = {}
        self.ml_decisions: dict[str, Any] = {}

        # Configuration
        self.ml_config: dict[str, Any] = self.config.get("ml_tactics_manager", {})
        self.enable_ml_tactics: bool = self.ml_config.get("enable_ml_tactics", True)
        self.confidence_threshold: float = self.ml_config.get(
            "confidence_threshold",
            0.7,
        )
        self.regime_threshold: float = self.ml_config.get("regime_threshold", 0.6)

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
        """
        Initialize ML tactics manager.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing ML Tactics Manager...")

            # Validate configuration
            if not self._validate_configuration():
                self.print(invalid("Invalid configuration for ML tactics manager"))
                return False

            # Initialize ML models
            await self._initialize_ml_models()

            self.is_initialized = True
            self.logger.info("✅ ML Tactics Manager initialized successfully")
            return True

        except Exception:
            self.print(failed("❌ ML Tactics Manager initialization failed: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate ML tactics manager configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            if self.confidence_threshold <= 0 or self.confidence_threshold > 1:
                self.print(invalid("Invalid confidence_threshold configuration"))
                return False

            if self.regime_threshold <= 0 or self.regime_threshold > 1:
                self.print(invalid("Invalid regime_threshold configuration"))
                return False

            return True

        except Exception:
            self.print(failed("Configuration validation failed: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="ML models initialization",
    )
    async def _initialize_ml_models(self) -> None:
        """Initialize ML prediction models."""
        try:
            # Initialize ML prediction models here
            # This would typically load pre-trained models for various ML predictions
            self.logger.info("✅ ML prediction models initialized")

        except Exception:
            self.print(failed("❌ Failed to initialize ML models: {e}"))
            raise

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
        """
        Execute ML-based tactics.

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
                self.print(warning("⚠️ No ML predictions available"))
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
                ml_predictions,
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

        except Exception:
            self.print(failed("❌ ML tactics execution failed: {e}"))
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="ML tactics input validation",
    )
    def _validate_tactics_input(self, tactics_input: dict[str, Any]) -> bool:
        """
        Validate ML tactics input parameters.

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
                self.print(invalid("Invalid current_price value"))
                return False

            return True

        except Exception:
            self.print(failed("ML tactics input validation failed: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="ML predictions retrieval",
    )
    def _get_ml_predictions(self) -> dict[str, Any] | None:
        """
        Get ML predictions.

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

        except Exception:
            self.print(failed("❌ Failed to get ML predictions: {e}"))
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
        """
        Apply regime and location tactics.

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
        """
        Make ML-based entry decisions.

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

        except Exception:
            self.print(failed("❌ ML entry decisions making failed: {e}"))
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
        """
        Make ML-based sizing decisions.

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

        except Exception:
            self.print(failed("❌ ML sizing decisions making failed: {e}"))
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
        """
        Make ML-based leverage decisions.

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

        except Exception:
            self.print(failed("❌ ML leverage decisions making failed: {e}"))
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
        """
        Make ML-based directional decisions.

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

        except Exception:
            self.print(failed("❌ ML directional decisions making failed: {e}"))
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
        """
        Make ML-based liquidation risk decisions.

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
        """
        Calculate position size based on ML predictions.

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

        except Exception:
            self.print(failed("❌ Position size calculation failed: {e}"))
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
        """
        Calculate leverage based on ML predictions.

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

        except Exception:
            self.print(failed("❌ Leverage calculation failed: {e}"))
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
        """
        Get the latest ML decisions.

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

        except Exception:
            self.print(failed("❌ Failed to stop ML Tactics Manager: {e}"))


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="ML tactics manager setup",
)
async def setup_ml_tactics_manager(
    config: dict[str, Any] | None = None,
) -> MLTacticsManager | None:
    """
    Setup and return a configured MLTacticsManager instance.

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
    except Exception:
        system_logger.exception(failed("Failed to setup ML Tactics Manager: {e}"))
        return None
