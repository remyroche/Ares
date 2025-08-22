# src/tactician/position_sizer.py

"""
Simplified Position Sizer for high leverage trading.
Uses ML confidence scores and Kelly criterion for position sizing.
"""

from datetime import datetime
from src.utils.logger import system_logger
from typing import Any, Optional, Dict, List
import contextlib

from src.config_optuna import get_parameter_value
from src.utils.confidence import normalize_dual_confidence
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.warning_symbols import error, initialization_error, missing
from src.utils.data_quality_decorators import validate_data_quality


class PositionSizer:
    """
    Simplified position sizer that uses ML confidence scores and Kelly criterion
    for position sizing decisions.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config
        self.logger = system_logger.getChild("PositionSizer")
        # Backward-compatibility shim for legacy self.print calls
        if not hasattr(self, "print"):

            def _shim_print(message: str) -> None:
                with contextlib.suppress(Exception):
                    self.logger.error(str(message))

            self.print = _shim_print  # type: ignore[attr-defined]

        # Load configuration

        self.sizing_config: Dict[str, Any] = self.config.get("position_sizing", {})
        self.kelly_multiplier: float = get_parameter_value(
            "position_sizing_parameters.kelly_multiplier",
            0.25,
        )
        self.max_position_size: float = get_parameter_value(
            "position_sizing_parameters.max_position_size",
            0.5,
        )
        self.min_position_size: float = get_parameter_value(
            "position_sizing_parameters.min_position_size",
            0.01,
        )
        self.confidence_threshold: float = get_parameter_value(
            "confidence_thresholds.base_entry_threshold",
            0.6,
        )

        # Component weights
        self.ml_weight: float = self.sizing_config.get("ml_weight", 0.7)
        self.kelly_weight: float = self.sizing_config.get("kelly_weight", 0.3)

        self.is_initialized: bool = False
        self.position_sizing_history: List[Dict[str, Any]] = []

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid position sizer configuration"),
            AttributeError: (False, "Missing required sizing parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="position sizer initialization",
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="position sizer initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the position sizer."""
        self.logger.info("Initializing position sizer...")

        # Validate configuration
        if not self._validate_configuration():
            return False

        self.is_initialized = True
        self.logger.info("✅ Position sizer initialized successfully")
        return True

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="configuration validation",
    )

    def _validate_configuration(self) -> bool:
        """Validate position sizer configuration."""
        try:
            required_keys = [
                "kelly_multiplier",
                "max_position_size",
                "min_position_size",
            ]
            for key in required_keys:
                if key not in self.sizing_config:
                    self.print(missing(f"Missing required configuration key: {key}"))
                    return False

            if self.max_position_size <= self.min_position_size:
                self.logger.error(
                    "max_position_size must be greater than min_position_size",
                )
                return False

            if self.kelly_multiplier <= 0 or self.kelly_multiplier > 1:
                self.print(error("kelly_multiplier must be between 0 and 1"))
                return False

            return True

        except Exception as e:
            self.print(error(f"Error validating configuration: {e}"))
            return False

    @validate_data_quality(
        required_columns=None,  # This method validates dict input, not DataFrame
        min_rows=1,
        max_null_ratio=0.0,
        check_duplicates=False,
        check_timestamps=False,
        context="position sizing calculation input validation"
    )
    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid input data for position sizing"),
            AttributeError: (None, "Sizer not properly initialized"),
        },
        default_return=None,
        context="position sizing calculation",
    )
    async def calculate_position_size(
        self,
        ml_predictions: Dict[str, Any],
        current_price: float = 0.0,
        account_balance: float = 1000.0,
        analyst_confidence: float = 0.5,
        tactician_confidence: float = 0.5,
        market_health_analysis: Optional[Dict[str, Any]] = None,
        strategist_risk_parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate position size using ML confidence scores and Kelly criterion.

        Args:
            ml_predictions: ML confidence predictions from ml_confidence_predictor
            current_price: Current market price
            account_balance: Account balance for position sizing
            market_health_analysis: Aggregated indicators from Analyst's MarketHealthAnalyzer
            strategist_risk_parameters: Risk parameters produced by Strategist (fed via Analyst)

        Returns:
            dict[str, Any]: Position sizing analysis
        """
        if not self.is_initialized:
            self.print(initialization_error("Position sizer not initialized"))
            return None

        self.logger.info("Calculating position size using ML intelligence...")

        try:
            # Extract ML confidence scores
            price_target_confidences = ml_predictions.get(
                "price_target_confidences",
                {},
            )
            adversarial_confidences = ml_predictions.get("adversarial_confidences", {})
            directional_confidence = ml_predictions.get("directional_confidence", {})

            # Calculate base Kelly criterion position size
            kelly_position_size = self._calculate_kelly_position_size(
                price_target_confidences, adversarial_confidences,
            )

            # Calculate ML-based position size
            ml_position_size = self._calculate_ml_position_size(
                price_target_confidences, adversarial_confidences,
            )

            # Calculate weighted position size
            final_position_size = self._calculate_weighted_position_size(
                kelly_position_size, ml_position_size,
            )

            # Apply market-health and strategist risk modifiers (volatility/liquidity/stress aware)
            final_position_size = self._apply_position_size_modifiers(
                final_position_size,
                market_health_analysis=market_health_analysis,
                strategist_risk_parameters=strategist_risk_parameters,
                analyst_confidence=analyst_confidence,
                tactician_confidence=tactician_confidence,
            )

            # Create position sizing analysis
            sizing_analysis = {
                "timestamp": datetime.now(),
                "current_price": current_price,
                "account_balance": account_balance,
                "kelly_position_size": kelly_position_size,
                "ml_position_size": ml_position_size,
                "final_position_size": final_position_size,
                "price_target_confidences": price_target_confidences,
                "adversarial_confidences": adversarial_confidences,
                "directional_confidence": directional_confidence,
                "market_health_modifiers": (market_health_analysis or {}),
                "strategist_risk_parameters": (strategist_risk_parameters or {}),
                "sizing_reason": self._generate_sizing_reason(
                    final_position_size,
                    kelly_position_size,
                    ml_position_size,
                    price_target_confidences,
                    adversarial_confidences,
                ),
            }

            # Store in history
            self.position_sizing_history.append(sizing_analysis)
            if len(self.position_sizing_history) > 100:  # Keep last 100 entries
                self.position_sizing_history = self.position_sizing_history[-100:]

            self.logger.info(f"✅ Position size calculated: {final_position_size:.4f}")
            return sizing_analysis

        except Exception as e:
            self.print(error(f"Error calculating position size: {e}"))
            return None

    def _calculate_kelly_position_size(
        self,
        price_target_confidences: Dict[str, float],
        adversarial_confidences: Dict[str, float],
    ) -> float:
        """Calculate position size using Kelly criterion based on ML confidence scores."""
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

            # CORRECT Kelly criterion: f = (bp - q) / b
            # where b = odds received, p = probability of win, q = probability of loss
            # For our case: b = 1 (1:1 odds), so f = p - q
            # where p = avg_confidence (probability of win)
            # and q = avg_adverse_risk (probability of loss)

            # Ensure probabilities are valid (0 <= p, q <= 1 and p + q <= 1)
            p = max(0.0, min(1.0, avg_confidence))
            q = max(0.0, min(1.0, avg_adverse_risk))

            # If p + q > 1, normalize them
            if p + q > 1.0:
                total = p + q
                p = p / total
                q = q / total

            # Calculate Kelly fraction
            kelly_fraction = p - q

            # Apply Kelly multiplier for conservative sizing
            kelly_position_size = kelly_fraction * self.kelly_multiplier

            # Ensure within bounds
            return max(
                self.min_position_size, min(self.max_position_size, kelly_position_size),
            )

        except (ValueError, TypeError, KeyError) as e:
            self.logger.exception(f"Error calculating Kelly position size: {e}")
            return self.min_position_size
        except ZeroDivisionError as e:
            self.logger.exception(f"Division by zero in Kelly calculation: {e}")
            return self.min_position_size

    def _calculate_ml_position_size(
        self,
        price_target_confidences: Dict[str, float],
        adversarial_confidences: Dict[str, float],
    ) -> float:
        """Calculate position size based on ML confidence scores."""
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

            # Calculate ML-based position size
            # Higher confidence and lower risk = larger position
            confidence_factor = avg_confidence / self.confidence_threshold
            risk_factor = 1.0 - avg_adverse_risk

            # Base position size calculation
            base_position_size = (
                self.min_position_size
                + (self.max_position_size - self.min_position_size)
                * confidence_factor
                * risk_factor
            )

            # Ensure within bounds
            return max(
                self.min_position_size, min(self.max_position_size, base_position_size),
            )

        except (ValueError, TypeError, KeyError) as e:
            self.logger.exception(f"Error calculating ML position size: {e}")
            return self.min_position_size
        except ZeroDivisionError as e:
            self.logger.exception(f"Division by zero in ML position calculation: {e}")
            return self.min_position_size

    def _calculate_weighted_position_size(
        self,
        kelly_position_size: float,
        ml_position_size: float,
    ) -> float:
        """Calculate weighted position size using Kelly criterion and ML confidence."""
        try:
            # Calculate weighted position size
            weighted_size = (
                kelly_position_size * self.kelly_weight
                + ml_position_size * self.ml_weight
            ) / (self.kelly_weight + self.ml_weight)

            return max(
                self.min_position_size, min(self.max_position_size, weighted_size),
            )

        except Exception as e:
            self.print(error(f"Error calculating weighted position size: {e}"))
            return kelly_position_size

    def _apply_position_size_modifiers(
        self,
        base_size: float,
        *,
        market_health_analysis: Optional[Dict[str, Any]],
        strategist_risk_parameters: Optional[Dict[str, Any]],
        analyst_confidence: float,
        tactician_confidence: float,
    ) -> float:
        """Adjust position size based on market health (vol/liquidity/stress), strategist risk, and dynamic confidence."""
        try:
            adjusted = base_size

            # Market health: downscale size under high volatility or stress; upscale when healthy
            if market_health_analysis:
                vol = market_health_analysis.get("volatility_analysis", {})
                stress = market_health_analysis.get("stress_analysis", {})
                liq = market_health_analysis.get("liquidity_analysis", {})

                current_vol = float(vol.get("current_volatility", 0.02))
                vol_regime = vol.get("volatility_regime", "normal")
                stress_level = float(stress.get("stress_level", 0.5))  # 0..1
                liquidity_score = float(liq.get("liquidity_score", 0.5))  # 0..1

                # Volatility adjustment
                if vol_regime in ("high", "extreme") or current_vol > 0.03:
                    adjusted *= 0.6
                elif vol_regime == "low" and current_vol < 0.015:
                    adjusted *= 1.1

                # Stress adjustment
                if stress_level >= 0.8:
                    adjusted *= 0.4
                elif stress_level >= 0.6:
                    adjusted *= 0.6
                elif stress_level >= 0.4:
                    adjusted *= 0.8

                # Liquidity adjustment
                if liquidity_score < 0.3:
                    adjusted *= 0.6
                elif liquidity_score > 0.7:
                    adjusted *= 1.05

            # Strategist risk parameters: respect max risk caps without using fixed TP/SL distances
            if strategist_risk_parameters:
                # Example: cap size based on max daily loss or risk per trade signals
                max_position_risk = float(
                    strategist_risk_parameters.get("max_position_risk", 0.01),
                )
                # Ensure final size does not exceed configured max_position_size
                configured_max = float(self.max_position_size)
                adjusted = min(adjusted, configured_max)
                # If max_position_risk is very small, reduce size further
                if max_position_risk <= 0.005:
                    adjusted *= 0.8

            # Dynamic confidence-based modulation (analyst and tactician)
            # Use dual confidence similar to monitor normalization
            _, normalized = normalize_dual_confidence(
                analyst_confidence, tactician_confidence,
            )
            # Scale position by a gentle factor around 1.0 (0.8..1.2)
            conf_scale = 0.8 + 0.4 * normalized
            adjusted *= conf_scale

            return max(self.min_position_size, min(self.max_position_size, adjusted))
        except Exception as e:
            self.print(error(f"Error applying size modifiers: {e}"))
            return max(self.min_position_size, min(self.max_position_size, base_size))

    def _generate_sizing_reason(
        self,
        final_position_size: float,
        kelly_position_size: float,
        ml_position_size: float,
        price_target_confidences: Dict[str, float],
        adversarial_confidences: Dict[str, float],
    ) -> str:
        """Generate reason for position sizing decision."""
        try:
            # Get average confidence and risk
            key_levels = [0.5, 1.0, 1.5, 2.0]
            confidences = []
            risks = []

            for level in key_levels:
                closest_confidence = min(
                    price_target_confidences.keys(),
                    key=lambda x: abs(float(x.replace("%", "")) - level),
                )
                closest_risk = min(
                    adversarial_confidences.keys(),
                    key=lambda x: abs(float(x.replace("%", "")) - level),
                )
                confidences.append(
                    price_target_confidences.get(closest_confidence, 0.5),
                )
                risks.append(adversarial_confidences.get(closest_risk, 0.3))

            avg_confidence = sum(confidences) / len(confidences)
            avg_risk = sum(risks) / len(risks)

            if final_position_size >= self.max_position_size * 0.8:
                return f"Maximum position size due to high confidence ({avg_confidence:.2f}) and low risk ({avg_risk:.2f})"
            if final_position_size >= self.max_position_size * 0.5:
                return f"Large position size based on Kelly criterion ({kelly_position_size:.3f}) and ML confidence ({ml_position_size:.3f})"
            if final_position_size >= self.min_position_size * 2:
                return "Moderate position size with balanced risk-reward profile"
            return f"Conservative position size due to low confidence ({avg_confidence:.2f}) or high risk ({avg_risk:.2f})"

        except Exception as e:
            self.print(error(f"Error generating sizing reason: {e}"))
            return "Position size calculated using ML intelligence and Kelly criterion"

    def _generate_dual_confidence_sizing_reason(
        self,
        final_position_size: float,
        final_confidence: float,
        normalized_confidence: float,
        analyst_confidence: float,
        tactician_confidence: float,
        p_avg: float,
        b_avg: float,
        fractional_kelly_pct: float,
    ) -> str:
        """Generate sizing reason for dual confidence system."""
        try:
            return (
                f"Position size: {final_position_size:.4f} "
                f"(Final confidence: {final_confidence:.3f}, Normalized: {normalized_confidence:.3f}) "
                f"Analyst: {analyst_confidence:.2f}, Tactician: {tactician_confidence:.2f} "
                f"Kelly: p_avg={p_avg:.2f}, b_avg={b_avg:.2f}, frac_kelly={fractional_kelly_pct:.3f}"
            )

        except Exception as e:
            self.logger.exception(
                f"Error generating dual confidence sizing reason: {e}",
            )
            return f"Position size: {final_position_size:.4f} (Error generating reason)"

    def _get_historical_performance(self) -> tuple[float, float]:
        """Get historical performance data for Kelly criterion calculation."""
        try:
            # Use local sizing history as a proxy when available
            # Expect entries with keys: {"pnl": float}
            history = self.position_sizing_history[-500:]  # recent window
            if not history:
                return 0.5, 1.5

            pnls = [float(h.get("pnl", 0.0)) for h in history if "pnl" in h]
            if not pnls:
                return 0.5, 1.5

            wins = [p for p in pnls if p > 0]
            losses = [-p for p in pnls if p < 0]

            num_trades = len(pnls)
            win_rate = (len(wins) / num_trades) if num_trades > 0 else 0.5
            avg_win = (sum(wins) / len(wins)) if wins else 1.0
            avg_loss = (sum(losses) / len(losses)) if losses else 1.0
            payoff = (avg_win / max(avg_loss, 1e-9)) if avg_loss else 1.5

            # Conservative shrinkage towards priors
            alpha = min(1.0, num_trades / 200.0)  # confidence weight up to 200 trades
            p_avg = (1 - alpha) * 0.5 + alpha * win_rate
            b_avg = (1 - alpha) * 1.5 + alpha * payoff

            # Clamp to reasonable bounds
            p_avg = max(0.3, min(0.7, p_avg))
            b_avg = max(0.8, min(2.5, b_avg))

            return p_avg, b_avg
        except Exception as e:
            self.print(error(f"Error getting historical performance: {e}"))
            return 0.5, 1.5  # Default fallback values

    def get_position_sizing_history(
        self,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get position sizing history."""
        if limit:
            return self.position_sizing_history[-limit:]
        return self.position_sizing_history.copy()

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="position sizer cleanup",
    )
    async def stop(self) -> None:
        """Stop the position sizer."""
        try:
            self.logger.info("Stopping position sizer...")
            self.is_initialized = False
            self.logger.info("✅ Position sizer stopped successfully")
        except Exception as e:
            self.print(error(f"Error stopping position sizer: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="position sizer cleanup",
    )
    async def cleanup(self) -> None:
        """Cleanup position sizer resources."""
        try:
            self.logger.info("Cleaning up position sizer...")
            await self.stop()
            self.position_sizing_history.clear()
            self.logger.info("✅ Position sizer cleanup completed")
        except Exception as e:
            self.logger.error(f"Error cleaning up position sizer: {e}")


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="position sizer setup",
)
async def setup_position_sizer(
    config: Optional[Dict[str, Any]] = None,
) -> Optional[PositionSizer]:
    """
    Setup position sizer.

    Args:
        config: Configuration dictionary

    Returns:
        Optional[PositionSizer]: Initialized position sizer or None
    """
    try:
        if config is None:
            config = {}

        position_sizer = PositionSizer(config)

        if await position_sizer.initialize():
            return position_sizer
        return None

    except Exception as e:
        system_logger.exception(error(f"Error setting up position sizer: {e}"))
        return None
