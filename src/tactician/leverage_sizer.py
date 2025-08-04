# src/tactician/leverage_sizer.py

from datetime import datetime
from typing import Any, Optional

from src.analyst.liquidation_risk_model import ProbabilisticLiquidationRiskModel
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.config_loader import load_leverage_sizing_config



class LeverageSizer:
    """
    Enhanced leverage sizing system that incorporates multiple indicators:
    - Confidence scores from ML models
    - Volatility measures (ATR, realized volatility)
    - Liquidation risk assessment
    - Dynamic risk management based on performance
    - Opportunity type detection (S/R levels, breakouts, etc.)
    """

    def __init__(self, config: dict[str, Any], state_manager=None):
        self.config = config
        self.state_manager = state_manager
        self.risk_config = config.get("risk_management", {})
        self.leverage_config = self.risk_config.get("leverage_sizing", {})
        self.dynamic_config = self.risk_config.get("dynamic_risk_management", {})
        self.logger = system_logger.getChild("LeverageSizer")

        # Initialize liquidation risk model
        self.liquidation_risk_model = ProbabilisticLiquidationRiskModel(
            self.risk_config.get("liquidation_risk", {})
        )

        # Performance tracking
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.current_exposure = 0.0

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=1,
        context="leverage calculation",
    )
    def calculate_leverage(
        self,
        base_leverage: float,
        max_leverage_cap: int,
        confidence: float = 0.0,
        market_conditions: dict[str, Any] = None,
        position_size: float = None,
        current_price: float = None,
        side: str = "long",
    ) -> dict[str, Any]:
        """
        Calculate optimal leverage based on multiple indicators.

        Args:
            base_leverage: Base leverage to start with
            max_leverage_cap: Maximum allowed leverage
            confidence: ML confidence score (0.0 to 1.0)
            market_conditions: Market data including volatility indicators
            position_size: Position size for liquidation risk calculation
            current_price: Current asset price for liquidation risk calculation
            side: 'long' or 'short'

        Returns:
            Dict containing leverage and metadata
        """
        try:
            # Start with base leverage
            leverage = base_leverage

            # Apply confidence-based leverage adjustments
            confidence_leverage = self._apply_confidence_leverage_adjustment(
                leverage,
                confidence,
            )

            # Apply volatility-based leverage adjustment
            volatility_leverage = self._apply_volatility_leverage_adjustment(
                confidence_leverage,
                market_conditions,
            )

            # Apply opportunity-based leverage adjustment
            opportunity_leverage = self._apply_opportunity_leverage_adjustment(
                market_conditions,
            )

            # Apply liquidation risk-based leverage adjustment
            liquidation_leverage = self._apply_liquidation_risk_leverage_adjustment(
                opportunity_leverage,
                position_size,
                current_price,
                side,
                market_conditions,
            )

            # Apply dynamic risk management to leverage
            dynamic_leverage = self._apply_dynamic_risk_leverage_adjustment(
                liquidation_leverage,
            )

            # Ensure leverage doesn't exceed maximum cap
            final_leverage = min(int(dynamic_leverage), max_leverage_cap)

            # Calculate liquidation safety score if position size is provided
            lss_metrics = {}
            if position_size and current_price:
                lss_metrics = self._calculate_liquidation_safety(
                    current_price,
                    position_size,
                    final_leverage,
                    side,
                    market_conditions,
                )

            return {
                "leverage": final_leverage,
                "confidence_score": confidence,
                "base_leverage": base_leverage,
                "confidence_multiplier": self._get_confidence_leverage_multiplier(confidence),
                "volatility_multiplier": self._get_volatility_leverage_multiplier(
                    market_conditions,
                ),
                "opportunity_multiplier": self._get_opportunity_leverage_multiplier(
                    market_conditions,
                ),
                "liquidation_multiplier": self._get_liquidation_leverage_multiplier(
                    lss_metrics,
                ),
                "risk_multiplier": self._get_risk_leverage_multiplier(),
                "max_leverage_cap": max_leverage_cap,
                "liquidation_safety_score": lss_metrics.get("lss", 50.0),
                "distance_to_liquidation": lss_metrics.get("distance_to_liquidation", 0.0),
                "calculation_time": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error calculating leverage: {e}")
            return {"leverage": 1, "error": str(e)}

    def _apply_confidence_leverage_adjustment(
        self,
        leverage: float,
        confidence: float,
    ) -> float:
        """Apply confidence-based leverage adjustments."""
        if not self.leverage_config.get("confidence_based_leverage", {}).get(
            "enable_confidence_scaling",
            True,
        ):
            return leverage

        thresholds = self.leverage_config.get("confidence_based_leverage", {}).get(
            "confidence_thresholds",
            {},
        )
        multipliers = self.leverage_config.get("confidence_based_leverage", {}).get(
            "leverage_multipliers",
            {},
        )

        # Determine confidence level
        if confidence >= thresholds.get("very_high_confidence", 0.95):
            multiplier = multipliers.get("very_high_confidence", 2.0)
        elif confidence >= thresholds.get("high_confidence", 0.85):
            multiplier = multipliers.get("high_confidence", 1.8)
        elif confidence >= thresholds.get("medium_confidence", 0.75):
            multiplier = multipliers.get("medium_confidence", 1.5)
        else:
            multiplier = multipliers.get("low_confidence", 1.0)

        return leverage * multiplier

    def _apply_volatility_leverage_adjustment(
        self,
        leverage: float,
        market_conditions: dict[str, Any],
    ) -> float:
        """Apply volatility-based leverage adjustment."""
        if not self.leverage_config.get("volatility_based_leverage", {}).get(
            "enable_volatility_scaling",
            True,
        ):
            return leverage

        # Get ATR or volatility measure
        atr = market_conditions.get("atr", 0.0) if market_conditions else 0.0
        current_price = (
            market_conditions.get("current_price", 1.0) if market_conditions else 1.0
        )
        atr_ratio = atr / current_price if current_price > 0 else 0.0

        # Get realized volatility if available
        realized_vol = (
            market_conditions.get("realized_volatility_30d", 0.0)
            if market_conditions
            else 0.0
        )

        # Use the higher of ATR ratio or realized volatility
        volatility_measure = max(atr_ratio, realized_vol)

        thresholds = self.leverage_config.get("volatility_based_leverage", {}).get(
            "volatility_thresholds",
            {},
        )
        multipliers = self.leverage_config.get("volatility_based_leverage", {}).get(
            "leverage_multipliers",
            {},
        )

        # Determine volatility level
        if volatility_measure <= thresholds.get("low_volatility", 0.02):
            multiplier = multipliers.get("low_volatility", 1.3)  # Higher leverage in low vol
        elif volatility_measure <= thresholds.get("medium_volatility", 0.05):
            multiplier = multipliers.get("medium_volatility", 1.0)
        else:
            multiplier = multipliers.get("high_volatility", 0.6)  # Lower leverage in high vol

        return leverage * multiplier


    def _apply_opportunity_leverage_adjustment(
        self,
        leverage: float,
        market_conditions: dict[str, Any],
    ) -> float:
        """Apply opportunity-based leverage adjustment."""
        if not self.leverage_config.get("opportunity_based_leverage", {}).get(
            "enable_opportunity_scaling",
            True,
        ):
            return leverage

        opportunity_type = (
            market_conditions.get("opportunity_type", "STANDARD")
            if market_conditions
            else "STANDARD"
        )
        multipliers = self.leverage_config.get("opportunity_based_leverage", {}).get(
            "opportunity_multipliers",
            {},
        )

        # Enhanced leverage for specific opportunity types
        if opportunity_type in ["SR_FADE", "SR_BREAKOUT"]:
            multiplier = multipliers.get("sr_opportunity", 2.0)
        elif opportunity_type == "BREAKOUT":
            multiplier = multipliers.get("breakout", 1.8)
        elif opportunity_type == "REVERSAL":
            multiplier = multipliers.get("reversal", 1.5)
        else:
            multiplier = multipliers.get("standard", 1.0)

        # Additional adjustments for specific market conditions
        if market_conditions:
            # S/R zone leverage boost
            if market_conditions.get("near_sr_zone", False):
                sr_boost = multipliers.get("sr_zone", 1.5)
                multiplier *= sr_boost

            # Huge candle leverage boost
            if market_conditions.get("huge_candle", False):
                huge_candle_boost = multipliers.get("huge_candle", 2.0)
                multiplier *= huge_candle_boost

            # Momentum leverage boost
            if market_conditions.get("strong_momentum", False):
                momentum_boost = multipliers.get("momentum", 1.3)
                multiplier *= momentum_boost

        return leverage * multiplier

    def _apply_liquidation_risk_leverage_adjustment(
        self,
        leverage: float,
        position_size: float,
        current_price: float,
        side: str,
        market_conditions: dict[str, Any],
    ) -> float:
        """Apply liquidation risk-based leverage adjustment."""
        if not self.leverage_config.get("liquidation_risk_leverage", {}).get(
            "enable_liquidation_scaling",
            True,
        ):
            return leverage

        if not position_size or not current_price:
            return leverage

        # Calculate liquidation safety score
        atr = market_conditions.get("atr", 0.0) if market_conditions else 0.0
        lss = self.liquidation_risk_model.calculate_lss(
            current_price,
            position_size,
            leverage,
            side,
            atr,
        )

        # Apply LSS-based adjustment
        thresholds = self.leverage_config.get("liquidation_risk_leverage", {}).get(
            "lss_thresholds",
            {},
        )
        multipliers = self.leverage_config.get("liquidation_risk_leverage", {}).get(
            "lss_multipliers",
            {},
        )

        if lss >= thresholds.get("very_safe", 80):
            multiplier = multipliers.get("very_safe", 1.2)
        elif lss >= thresholds.get("safe", 60):
            multiplier = multipliers.get("safe", 1.0)
        elif lss >= thresholds.get("moderate", 40):
            multiplier = multipliers.get("moderate", 0.8)
        else:
            multiplier = multipliers.get("dangerous", 0.5)

        return leverage * multiplier

    def _apply_dynamic_risk_leverage_adjustment(self, leverage: float) -> float:
        """Apply dynamic risk management to leverage calculation."""
        if not self.dynamic_config.get("enable_dynamic_risk", True):
            return leverage

        # Apply drawdown-based leverage reduction
        if self.max_drawdown >= 0.4:
            leverage *= 0.2
        elif self.max_drawdown >= 0.3:
            leverage *= 0.5
        elif self.max_drawdown >= 0.2:
            leverage *= 0.7
        elif self.max_drawdown >= 0.1:
            leverage *= 0.9

        # Apply daily loss-based leverage reduction
        if self.daily_pnl <= -0.10:
            leverage *= 0.2
        elif self.daily_pnl <= -0.08:
            leverage *= 0.5
        elif self.daily_pnl <= -0.05:
            leverage *= 0.8

        return leverage

    def _calculate_liquidation_safety(
        self,
        current_price: float,
        position_size: float,
        leverage: int,
        side: str,
        market_conditions: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate liquidation safety metrics."""
        atr = market_conditions.get("atr", 0.0) if market_conditions else 0.0
        
        return self.liquidation_risk_model.get_risk_metrics(
            current_price,
            position_size,
            leverage,
            side,
            atr,
        )

    def _get_confidence_leverage_multiplier(self, confidence: float) -> float:
        """Get confidence-based leverage multiplier."""
        thresholds = self.leverage_config.get("confidence_based_leverage", {}).get(
            "confidence_thresholds",
            {},
        )
        multipliers = self.leverage_config.get("confidence_based_leverage", {}).get(
            "leverage_multipliers",
            {},
        )

        if confidence >= thresholds.get("very_high_confidence", 0.95):
            return multipliers.get("very_high_confidence", 2.0)
        if confidence >= thresholds.get("high_confidence", 0.85):
            return multipliers.get("high_confidence", 1.8)
        if confidence >= thresholds.get("medium_confidence", 0.75):
            return multipliers.get("medium_confidence", 1.5)
        return multipliers.get("low_confidence", 1.0)

    def _get_volatility_leverage_multiplier(self, market_conditions: dict[str, Any]) -> float:
        """Get volatility-based leverage multiplier."""
        atr = market_conditions.get("atr", 0.0) if market_conditions else 0.0
        current_price = (
            market_conditions.get("current_price", 1.0) if market_conditions else 1.0
        )
        atr_ratio = atr / current_price if current_price > 0 else 0.0

        # Get realized volatility if available
        realized_vol = (
            market_conditions.get("realized_volatility_30d", 0.0)
            if market_conditions
            else 0.0
        )

        # Use the higher of ATR ratio or realized volatility
        volatility_measure = max(atr_ratio, realized_vol)

        thresholds = self.leverage_config.get("volatility_based_leverage", {}).get(
            "volatility_thresholds",
            {},
        )
        multipliers = self.leverage_config.get("volatility_based_leverage", {}).get(
            "leverage_multipliers",
            {},
        )

        if volatility_measure <= thresholds.get("low_volatility", 0.02):
            return multipliers.get("low_volatility", 1.3)
        if volatility_measure <= thresholds.get("medium_volatility", 0.05):
            return multipliers.get("medium_volatility", 1.0)
        return multipliers.get("high_volatility", 0.6)


    def _get_opportunity_leverage_multiplier(self, market_conditions: dict[str, Any]) -> float:
        """Get opportunity-based leverage multiplier."""
        opportunity_type = (
            market_conditions.get("opportunity_type", "STANDARD")
            if market_conditions
            else "STANDARD"
        )
        multipliers = self.leverage_config.get("opportunity_based_leverage", {}).get(
            "opportunity_multipliers",
            {},
        )

        base_multiplier = multipliers.get(opportunity_type, 1.0)
        
        # Apply additional adjustments
        if market_conditions:
            if market_conditions.get("near_sr_zone", False):
                base_multiplier *= multipliers.get("sr_zone", 1.5)
            if market_conditions.get("huge_candle", False):
                base_multiplier *= multipliers.get("huge_candle", 2.0)
            if market_conditions.get("strong_momentum", False):
                base_multiplier *= multipliers.get("momentum", 1.3)

        return base_multiplier

    def _get_liquidation_leverage_multiplier(self, lss_metrics: dict[str, Any]) -> float:
        """Get liquidation risk-based leverage multiplier."""
        lss = lss_metrics.get("lss", 50.0)
        
        thresholds = self.leverage_config.get("liquidation_risk_leverage", {}).get(
            "lss_thresholds",
            {},
        )
        multipliers = self.leverage_config.get("liquidation_risk_leverage", {}).get(
            "lss_multipliers",
            {},
        )

        if lss >= thresholds.get("very_safe", 80):
            return multipliers.get("very_safe", 1.2)
        if lss >= thresholds.get("safe", 60):
            return multipliers.get("safe", 1.0)
        if lss >= thresholds.get("moderate", 40):
            return multipliers.get("moderate", 0.8)
        return multipliers.get("dangerous", 0.5)

    def _get_risk_leverage_multiplier(self) -> float:
        """Get current risk-based leverage multiplier."""
        # This would be calculated based on current drawdown and daily loss
        drawdown_multiplier = 1.0
        daily_loss_multiplier = 1.0

        # Apply drawdown adjustment
        if self.max_drawdown >= 0.4:
            drawdown_multiplier = 0.2
        elif self.max_drawdown >= 0.3:
            drawdown_multiplier = 0.5
        elif self.max_drawdown >= 0.2:
            drawdown_multiplier = 0.7
        elif self.max_drawdown >= 0.1:
            drawdown_multiplier = 0.9

        # Apply daily loss adjustment
        if self.daily_pnl <= -0.10:
            daily_loss_multiplier = 0.2
        elif self.daily_pnl <= -0.08:
            daily_loss_multiplier = 0.5
        elif self.daily_pnl <= -0.05:
            daily_loss_multiplier = 0.8

        return min(drawdown_multiplier, daily_loss_multiplier)

    def update_performance_metrics(self, daily_pnl: float, max_drawdown: float):
        """Update performance metrics for dynamic risk management."""
        self.daily_pnl = daily_pnl
        self.max_drawdown = max_drawdown

    def update_exposure(self, new_exposure: float):
        """Update current total exposure."""
        self.current_exposure = new_exposure

    def get_leverage_summary(self) -> dict[str, Any]:
        """Get summary of leverage sizing rules and current state."""
        return {
            "current_exposure": self.current_exposure,
            "daily_pnl": self.daily_pnl,
            "max_drawdown": self.max_drawdown,
            "risk_multiplier": self._get_risk_leverage_multiplier(),
            "leverage_config": self.leverage_config,
            "dynamic_config": self.dynamic_config,
        }
