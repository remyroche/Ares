# src/tactician/leverage_sizer.py

"""
Simplified Leverage Sizer for high leverage trading.
Uses ML confidence scores, liquidation risk model, and market health analysis.
"""

from datetime import datetime
from src.utils.logger import system_logger
import contextlib
from typing import Any

from src.config_optuna import get_parameter_value
from src.core.decorators import handles_errors
import copy
import asyncio

class LeverageSizer:
    """
    Simplified leverage sizer that uses ML confidence scores and liquidation risk model
    to set leverage between 10x and 100x.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("LeverageSizer")
        # Backward-compatibility shim for legacy self.print calls
        if not hasattr(self, "print"):
            def _shim_print(message: str) -> None:
                with contextlib.suppress(Exception):
                    self.logger.error(str(message))

            self.print = _shim_print  # type: ignore[attr-defined]

        # Load configuration from step17 optimization results
        self.leverage_config: dict[str, Any] = self.config.get("leverage_sizing", {})
        
        # Load step17 optimized parameters
        step17_config = self.config.get("step17_optimization", {})
        leverage_optimization = step17_config.get("leverage", {})
        
        # Load optimized leverage parameters
        self.min_leverage: float = leverage_optimization.get("min_leverage", 10.0)
        self.max_leverage: float = leverage_optimization.get("max_leverage", 100.0)
        self.confidence_threshold: float = leverage_optimization.get("confidence_threshold", 0.6)
        self.liquidation_buffer: float = leverage_optimization.get("liquidation_buffer", 0.05)
        
        # NEW: Combined confidence threshold for leverage sizing (optimizable in step17)
        self.leverage_combined_threshold: float = leverage_optimization.get("leverage_combined_threshold", 0.75)
        
        # Load optimized component weights
        self.ml_weight: float = leverage_optimization.get("ml_weight", 0.6)
        self.liquidation_weight: float = leverage_optimization.get("liquidation_weight", 0.4)
        
        # Load additional optimized parameters
        self.leverage_multiplier: float = leverage_optimization.get("leverage_multiplier", 1.0)
        self.risk_adjustment_factor: float = leverage_optimization.get("risk_adjustment_factor", 1.0)
        self.confidence_boost_threshold: float = leverage_optimization.get("confidence_boost_threshold", 0.8)
        self.max_risk_leverage: float = leverage_optimization.get("max_risk_leverage", 50.0)

        self.is_initialized: bool = False
        self.leverage_sizing_history: list[dict[str, Any]] = []

    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid leverage sizer configuration"),
            AttributeError: (False, "Missing required leverage parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="leverage sizer initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the leverage sizer."""
        self.logger.info("Initializing leverage sizer...")

        # Validate configuration
        if not self._validate_configuration():
            return False

        self.is_initialized = True
        self.logger.info("✅ Leverage sizer initialized successfully")
        return True

    def _validate_configuration(self) -> bool:
        """Validate leverage sizer configuration."""
        try:
            required_keys = [
                "min_leverage",
                "max_leverage",
                "confidence_threshold",
                "liquidation_buffer",
            ]
            for key in required_keys:
                if not hasattr(self, key):
                    self.logger.error(f"Missing required configuration key: {key}")
                    return False

            # Validate values
            if self.min_leverage <= 0 or self.min_leverage >= self.max_leverage:
                self.logger.error("Invalid leverage range configuration")
                return False

            if self.liquidation_buffer <= 0 or self.liquidation_buffer >= 1:
                self.logger.error("Invalid liquidation_buffer configuration")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    def refresh_step17_configuration(self, step17_results: dict[str, Any]) -> None:
        """
        Refresh configuration from step17 optimization results.
        This method is called automatically when step17 completes.
        
        Args:
            step17_results: Step17 optimization results
        """
        try:
            if "leverage" in step17_results:
                leverage_optimization = step17_results["leverage"]
                
                # Update leverage parameters
                self.min_leverage = leverage_optimization.get("min_leverage", self.min_leverage)
                self.max_leverage = leverage_optimization.get("max_leverage", self.max_leverage)
                self.confidence_threshold = leverage_optimization.get("confidence_threshold", self.confidence_threshold)
                self.liquidation_buffer = leverage_optimization.get("liquidation_buffer", self.liquidation_buffer)
                
                # Update component weights
                self.ml_weight = leverage_optimization.get("ml_weight", self.ml_weight)
                self.liquidation_weight = leverage_optimization.get("liquidation_weight", self.liquidation_weight)
                
                # Update additional parameters
                self.leverage_multiplier = leverage_optimization.get("leverage_multiplier", self.leverage_multiplier)
                self.risk_adjustment_factor = leverage_optimization.get("risk_adjustment_factor", self.risk_adjustment_factor)
                self.confidence_boost_threshold = leverage_optimization.get("confidence_boost_threshold", self.confidence_boost_threshold)
                self.max_risk_leverage = leverage_optimization.get("max_risk_leverage", self.max_risk_leverage)
                
                self.logger.info("✅ Leverage sizer configuration refreshed from step17 results")
                
        except Exception as e:
            self.logger.error(f"Error refreshing step17 configuration: {e}")

    @handles_errors(
        error_handlers={
            ValueError: (None, "Invalid input data for leverage sizing"),
            AttributeError: (None, "Sizer not properly initialized"),
        },
        default_return={},
        context="leverage sizing calculation",
    )
    async def calculate_leverage(
        self,
        ml_predictions: dict[str, Any],
        current_price: float = 0.0,
        account_balance: float = 1000.0,
        analyst_confidence: float = 0.5,
        tactician_confidence: float = 0.5,
        market_health_analysis: dict[str, Any] | None = None,
        strategist_risk_parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Calculate leverage using ML confidence scores and liquidation risk model.

        Args:
            ml_predictions: ML model predictions
            current_price: Current market price
            account_balance: Account balance
            analyst_confidence: Analyst confidence score
            tactician_confidence: Tactician confidence score
            market_health_analysis: Market health analysis
            strategist_risk_parameters: Strategist risk parameters

        Returns:
            dict[str, Any]: Leverage sizing analysis
        """
        if not self.is_initialized:
            self.logger.error("Leverage sizer not initialized")
            return {}

        try:
            # NEW: Extract combined confidence from Tactician multi-output predictions
            combined_confidence = ml_predictions.get("combined_confidence", 0.5)
            
            # Extract ML confidence scores (for backward compatibility)
            price_target_confidences = ml_predictions.get("price_target_confidences", {})
            adversarial_confidences = ml_predictions.get("adversarial_confidences", {})

            # NEW: Use combined confidence for leverage sizing if available
            if combined_confidence >= self.leverage_combined_threshold:
                # Calculate base ML leverage
                ml_leverage = self._calculate_ml_leverage(
                    price_target_confidences, adversarial_confidences,
                )

                # Calculate liquidation risk-adjusted leverage
                liquidation_leverage = self._calculate_liquidation_safe_leverage(
                    current_price, account_balance, market_health_analysis,
                )

                # Calculate weighted leverage
                final_leverage = self._calculate_weighted_leverage(
                    ml_leverage, liquidation_leverage,
                )

                # Apply market-health and strategist risk modifiers
                final_leverage = self._apply_leverage_modifiers(
                    final_leverage,
                    market_health_analysis=market_health_analysis,
                    strategist_risk_parameters=strategist_risk_parameters,
                    analyst_confidence=analyst_confidence,
                    tactician_confidence=tactician_confidence,
                )
            else:
                # If combined confidence is below threshold, use minimum leverage
                final_leverage = self.min_leverage
                ml_leverage = self.min_leverage
                liquidation_leverage = self.min_leverage

            # Create leverage sizing analysis
            leverage_analysis = {
                "timestamp": datetime.now(),
                "current_price": current_price,
                "account_balance": account_balance,
                "ml_leverage": ml_leverage,
                "liquidation_leverage": liquidation_leverage,
                "final_leverage": final_leverage,
                "combined_confidence": combined_confidence,
                "leverage_combined_threshold": self.leverage_combined_threshold,
                "price_target_confidences": price_target_confidences,
                "adversarial_confidences": adversarial_confidences,
                "market_health_modifiers": (market_health_analysis or {}),
                "strategist_risk_parameters": (strategist_risk_parameters or {}),
                "leverage_reason": self._generate_leverage_reason(
                    final_leverage,
                    ml_leverage,
                    liquidation_leverage,
                    price_target_confidences,
                    adversarial_confidences,
                    combined_confidence,
                ),
            }

            # Store in history
            self.leverage_sizing_history.append(leverage_analysis)
            if len(self.leverage_sizing_history) > 100:  # Keep last 100 entries
                self.leverage_sizing_history = self.leverage_sizing_history[-100:]

            self.logger.info(f"✅ Leverage calculated: {final_leverage:.1f}x")
            return leverage_analysis

        except Exception as e:
            self.logger.error(f"Error calculating leverage: {e}")
            return {}

    def _calculate_ml_leverage(
        self,
        price_target_confidences: dict[str, float],
        adversarial_confidences: dict[str, float],
    ) -> float:
        """Calculate leverage based on ML confidence scores."""
        try:
            # Get average confidence for target levels (0.5% to 2.0%)
            target_levels = [0.5, 1.0, 1.5, 2.0]
            confidences = []

            for level in target_levels:
                closest_level = min(
                    price_target_confidences.keys(),
                    key=lambda x: abs(float(x.replace("%", "")) - level),
                )
                confidence = price_target_confidences.get(closest_level, 0.5)
                confidences.append(confidence)

            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences)

            # Get average adverse risk
            adverse_risks = []
            for level in target_levels:
                closest_level = min(
                    adversarial_confidences.keys(),
                    key=lambda x: abs(float(x.replace("%", "")) - level),
                )
                risk = adversarial_confidences.get(closest_level, 0.3)
                adverse_risks.append(risk)

            avg_adverse_risk = sum(adverse_risks) / len(adverse_risks)

            # Calculate confidence-based leverage
            confidence_factor = avg_confidence / self.confidence_threshold
            risk_factor = 1.0 - avg_adverse_risk

            # Base leverage calculation
            base_leverage = (
                self.min_leverage
                + (self.max_leverage - self.min_leverage)
                * confidence_factor
                * risk_factor
            )

            # Ensure within bounds
            return max(
                self.min_leverage, min(self.max_leverage, base_leverage),
            )

        except (ValueError, TypeError, KeyError) as e:
            self.logger.exception(f"Error calculating ML leverage: {e}")
            return self.min_leverage

    def _calculate_liquidation_safe_leverage(
        self,
        current_price: float,
        account_balance: float,
        market_health_analysis: dict[str, Any] | None,
    ) -> float:
        """Calculate safe leverage to avoid liquidation."""
        try:
            # Base liquidation calculation
            # Assume worst-case scenario: 10% price move against position
            worst_case_move = 0.10

            # Adjust for market volatility
            if market_health_analysis:
                vol_analysis = market_health_analysis.get("volatility_analysis", {})
                current_vol = float(vol_analysis.get("current_volatility", 0.02))

                # If volatility is high, increase the worst-case scenario
                if current_vol > 0.03:
                    worst_case_move = min(0.20, current_vol * 2)
                elif current_vol > 0.02:
                    worst_case_move = 0.15

            # Calculate safe leverage with buffer
            safe_leverage = (1.0 - self.liquidation_buffer) / worst_case_move

            # Ensure within bounds
            return max(
                self.min_leverage, min(self.max_leverage, safe_leverage),
            )

        except (ValueError, TypeError) as e:
            self.logger.exception(f"Error calculating liquidation safe leverage: {e}")
            return self.min_leverage

    def _calculate_weighted_leverage(
        self,
        ml_leverage: float,
        liquidation_leverage: float,
    ) -> float:
        """Calculate weighted leverage using ML and liquidation risk models."""
        try:
            # Calculate weighted leverage
            weighted_leverage = (
                ml_leverage * self.ml_weight
                + liquidation_leverage * self.liquidation_weight
            ) / (self.ml_weight + self.liquidation_weight)

            # Ensure within bounds
            return max(
                self.min_leverage, min(self.max_leverage, weighted_leverage),
            )

        except Exception as e:
            self.logger.exception(f"Error calculating weighted leverage: {e}")
            return self.min_leverage

    def _apply_leverage_modifiers(
        self,
        base_leverage: float,
        *,
        market_health_analysis: dict[str, Any] | None,
        strategist_risk_parameters: dict[str, Any] | None,
        analyst_confidence: float,
        tactician_confidence: float,
    ) -> float:
        """Adjust leverage based on market health and risk parameters."""
        try:
            adjusted = base_leverage

            # Apply market health modifiers
            if market_health_analysis:
                volatility_modifier = market_health_analysis.get("volatility_modifier", 1.0)
                liquidity_modifier = market_health_analysis.get("liquidity_modifier", 1.0)
                stress_modifier = market_health_analysis.get("stress_modifier", 1.0)

                adjusted *= volatility_modifier * liquidity_modifier * stress_modifier

            # Apply strategist risk parameters
            if strategist_risk_parameters:
                risk_modifier = strategist_risk_parameters.get("leverage_modifier", 1.0)
                adjusted *= risk_modifier

            # Apply confidence modifiers
            confidence_modifier = (analyst_confidence + tactician_confidence) / 2
            adjusted *= confidence_modifier

            # Ensure within bounds
            return max(
                self.min_leverage, min(self.max_leverage, adjusted),
            )

        except Exception as e:
            self.logger.exception(f"Error applying leverage modifiers: {e}")
            return base_leverage

    def _generate_leverage_reason(
        self,
        final_leverage: float,
        ml_leverage: float,
        liquidation_leverage: float,
        price_target_confidences: dict[str, float],
        adversarial_confidences: dict[str, float],
        combined_confidence: float = 0.5,
    ) -> str:
        """Generate reason for leverage sizing decision."""
        try:
            # Get average confidence and risk
            key_levels = [0.5, 1.0, 1.5, 2.0]
            avg_confidence = 0.0
            avg_risk = 0.0

            for level in key_levels:
                closest_level = min(
                    price_target_confidences.keys(),
                    key=lambda x: abs(float(x.replace("%", "")) - level),
                )
                confidence = price_target_confidences.get(closest_level, 0.5)
                risk = adversarial_confidences.get(closest_level, 0.3)
                avg_confidence += confidence
                avg_risk += risk

            avg_confidence /= len(key_levels)
            avg_risk /= len(key_levels)

            # NEW: Include combined confidence in leverage reason
            if combined_confidence < self.leverage_combined_threshold:
                return (
                    f"Leverage: {final_leverage:.1f}x (minimum due to low combined confidence "
                    f"{combined_confidence:.2f} below threshold {self.leverage_combined_threshold:.2f})"
                )
            
            return (
                f"Leverage: {final_leverage:.1f}x "
                f"(ML: {ml_leverage:.1f}x, Liquidation: {liquidation_leverage:.1f}x, "
                f"Combined Confidence: {combined_confidence:.3f}, Risk: {avg_risk:.3f})"
            )

        except Exception as e:
            self.logger.exception(f"Error generating leverage reason: {e}")
            return f"Leverage: {final_leverage:.1f}x (Error generating reason)"

    def get_leverage_sizing_history(
        self,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get leverage sizing history."""
        if limit:
            return self.leverage_sizing_history[-limit:]
        return self.leverage_sizing_history.copy()

    @handles_errors(fallback=None)
    async def stop(self) -> None:
        """Stop the leverage sizer."""
        try:
            self.logger.info("Stopping leverage sizer...")
            self.is_initialized = False
            self.logger.info("✅ Leverage sizer stopped successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to stop leverage sizer: {e}")

    @handles_errors(fallback=None)
    async def cleanup(self) -> None:
        """Cleanup leverage sizer resources."""
        try:
            self.logger.info("Cleaning up leverage sizer...")
            await self.stop()
            self.leverage_sizing_history.clear()
            self.logger.info("✅ Leverage sizer cleanup completed")
        except Exception as e:
            self.logger.error(f"Error cleaning up leverage sizer: {e}")

@handles_errors(fallback=None)
async def setup_leverage_sizer(
    config: dict[str, Any] | None = None,
) -> LeverageSizer | None:
    """
    Setup and return a configured LeverageSizer instance.

    Args:
        config: Configuration dictionary

    Returns:
        LeverageSizer: Configured leverage sizer instance
    """
    try:
        if config is None:
            config = {}

        leverage_sizer = LeverageSizer(config)
        if await leverage_sizer.initialize():
            return leverage_sizer
        return None
    except Exception as e:
        system_logger.exception(f"Failed to setup leverage sizer: {e}")
        return None
