# src/tactician/leverage_sizer.py

"""
Simplified Leverage Sizer for high leverage trading.
Uses ML confidence scores, liquidation risk model, and market health analysis.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger


class LeverageSizer:
    """
    Simplified leverage sizer that uses ML confidence scores, liquidation risk model,
    and market health analysis to set leverage between 10x and 100x.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("LeverageSizer")

        # Load configuration
        self.leverage_config: dict[str, Any] = self.config.get("leverage_sizing", {})
        self.max_leverage: float = self.leverage_config.get("max_leverage", 100.0)
        self.min_leverage: float = self.leverage_config.get("min_leverage", 10.0)
        self.confidence_threshold: float = self.leverage_config.get("confidence_threshold", 0.7)
        self.risk_tolerance: float = self.leverage_config.get("risk_tolerance", 0.3)
        
        # Component weights
        self.ml_weight: float = self.leverage_config.get("ml_weight", 0.5)
        self.liquidation_risk_weight: float = self.leverage_config.get("liquidation_risk_weight", 0.3)
        self.market_health_weight: float = self.leverage_config.get("market_health_weight", 0.2)
        
        self.is_initialized: bool = False
        self.leverage_sizing_history: List[dict[str, Any]] = []

    @handle_specific_errors(
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
        try:
            self.logger.info("Initializing leverage sizer...")

            # Validate configuration
            if not self._validate_configuration():
                return False

            self.is_initialized = True
            self.logger.info("✅ Leverage sizer initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing leverage sizer: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """Validate leverage sizer configuration."""
        try:
            required_keys = ["max_leverage", "min_leverage", "confidence_threshold"]
            for key in required_keys:
                if key not in self.leverage_config:
                    self.logger.error(f"Missing required configuration key: {key}")
                    return False

            if self.max_leverage <= self.min_leverage:
                self.logger.error("max_leverage must be greater than min_leverage")
                return False

            if self.confidence_threshold <= 0 or self.confidence_threshold > 1:
                self.logger.error("confidence_threshold must be between 0 and 1")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid input data for leverage sizing"),
            AttributeError: (None, "Sizer not properly initialized"),
        },
        default_return=None,
        context="leverage sizing calculation",
    )
    async def calculate_leverage(
        self,
        ml_predictions: dict[str, Any],
        liquidation_risk_analysis: Optional[dict[str, Any]] = None,
        market_health_analysis: Optional[dict[str, Any]] = None,
        current_price: float = 0.0,
        target_direction: str = "long",
    ) -> dict[str, Any]:
        """
        Calculate leverage using ML confidence scores, liquidation risk analysis, and market health.
        
        Args:
            ml_predictions: ML confidence predictions from ml_confidence_predictor
            liquidation_risk_analysis: Liquidation risk analysis from liquidation_risk_model
            market_health_analysis: Market health analysis from market_health_analyzer
            current_price: Current market price
            target_direction: Target direction ("long" or "short")
            
        Returns:
            dict[str, Any]: Leverage sizing analysis
        """
        try:
            if not self.is_initialized:
                self.logger.error("Leverage sizer not initialized")
                return None

            self.logger.info(f"Calculating leverage for {target_direction} position...")

            # Extract ML confidence scores
            movement_confidence = ml_predictions.get("movement_confidence_scores", {})
            adverse_movement_risks = ml_predictions.get("adverse_movement_risks", {})
            directional_confidence = ml_predictions.get("directional_confidence", {})

            # Calculate base leverage from ML confidence
            ml_leverage = self._calculate_ml_leverage(movement_confidence, adverse_movement_risks)

            # Get liquidation risk leverage recommendations
            liquidation_leverage = self._extract_liquidation_leverage(liquidation_risk_analysis)

            # Get market health leverage adjustment
            market_health_leverage = self._extract_market_health_leverage(market_health_analysis)

            # Calculate weighted leverage
            final_leverage = self._calculate_weighted_leverage(
                ml_leverage,
                liquidation_leverage,
                market_health_leverage,
            )

            # Create leverage sizing analysis
            leverage_analysis = {
                "timestamp": datetime.now(),
                "current_price": current_price,
                "target_direction": target_direction,
                "ml_leverage": ml_leverage,
                "liquidation_leverage": liquidation_leverage,
                "market_health_leverage": market_health_leverage,
                "final_leverage": final_leverage,
                "ml_confidence_scores": movement_confidence,
                "adverse_movement_risks": adverse_movement_risks,
                "directional_confidence": directional_confidence,
                "leverage_reason": self._generate_leverage_reason(
                    final_leverage, ml_leverage, liquidation_leverage, market_health_leverage, movement_confidence, adverse_movement_risks
                ),
            }

            # Store in history
            self.leverage_sizing_history.append(leverage_analysis)
            if len(self.leverage_sizing_history) > 100:  # Keep last 100 entries
                self.leverage_sizing_history = self.leverage_sizing_history[-100:]

            self.logger.info(f"✅ Leverage calculated: {final_leverage:.2f}x")
            return leverage_analysis

        except Exception as e:
            self.logger.error(f"Error calculating leverage: {e}")
            return None

    def _calculate_ml_leverage(
        self, movement_confidence: dict[str, float], adverse_movement_risks: dict[str, float]
    ) -> float:
        """Calculate leverage based on ML confidence scores."""
        try:
            # Get average confidence for target levels (0.5% to 2.0%)
            target_levels = [0.5, 1.0, 1.5, 2.0]
            confidences = []
            
            for level in target_levels:
                closest_level = min(movement_confidence.keys(), key=lambda x: abs(float(x) - level))
                confidence = movement_confidence.get(closest_level, 0.5)
                confidences.append(confidence)
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences)
            
            # Get average adverse risk
            adverse_risks = []
            for level in target_levels:
                closest_level = min(adverse_movement_risks.keys(), key=lambda x: abs(float(x) - level))
                risk = adverse_movement_risks.get(closest_level, 0.3)
                adverse_risks.append(risk)
            
            avg_adverse_risk = sum(adverse_risks) / len(adverse_risks)
            
            # Calculate leverage based on confidence and risk
            # Higher confidence and lower risk = higher leverage
            confidence_factor = avg_confidence / self.confidence_threshold
            risk_factor = 1.0 - avg_adverse_risk
            
            # Base leverage calculation (10x to 100x range)
            base_leverage = self.min_leverage + (self.max_leverage - self.min_leverage) * confidence_factor * risk_factor
            
            # Apply risk tolerance adjustment
            risk_adjusted_leverage = base_leverage * (1.0 - self.risk_tolerance)
            
            return max(self.min_leverage, min(self.max_leverage, risk_adjusted_leverage))
            
        except Exception as e:
            self.logger.error(f"Error calculating ML leverage: {e}")
            return self.min_leverage

    def _extract_liquidation_leverage(self, liquidation_risk_analysis: Optional[dict[str, Any]]) -> float:
        """Extract leverage recommendations from liquidation risk analysis."""
        try:
            if not liquidation_risk_analysis:
                return self.min_leverage
            
            # Get safe leverage levels
            safe_leverage_levels = liquidation_risk_analysis.get("safe_leverage_levels", {})
            
            if not safe_leverage_levels:
                return self.min_leverage
            
            # Get average safe leverage
            safe_leverages = []
            for leverage_data in safe_leverage_levels.values():
                safe_leverage = leverage_data.get("safe_leverage", self.min_leverage)
                safe_leverages.append(safe_leverage)
            
            if safe_leverages:
                avg_safe_leverage = sum(safe_leverages) / len(safe_leverages)
                return max(self.min_leverage, min(self.max_leverage, avg_safe_leverage))
            else:
                return self.min_leverage
            
        except Exception as e:
            self.logger.error(f"Error extracting liquidation leverage: {e}")
            return self.min_leverage

    def _extract_market_health_leverage(self, market_health_analysis: Optional[dict[str, Any]]) -> float:
        """Extract leverage adjustment from market health analysis."""
        try:
            if not market_health_analysis:
                return self.min_leverage
            
            # Get volatility analysis
            volatility_analysis = market_health_analysis.get("volatility_analysis", {})
            current_volatility = volatility_analysis.get("current_volatility", 0.02)
            historical_volatility = volatility_analysis.get("historical_volatility", 0.02)
            volatility_regime = volatility_analysis.get("volatility_regime", "normal")
            
            # Get liquidity analysis
            liquidity_analysis = market_health_analysis.get("liquidity_analysis", {})
            liquidity_score = liquidity_analysis.get("liquidity_score", 0.5)
            bid_ask_spread = liquidity_analysis.get("bid_ask_spread", 0.001)
            market_depth = liquidity_analysis.get("market_depth", 0.5)
            
            # Get market stress analysis
            stress_analysis = market_health_analysis.get("stress_analysis", {})
            stress_level = stress_analysis.get("stress_level", 0.5)
            stress_regime = stress_analysis.get("stress_regime", "normal")
            
            # Calculate volatility factor with regime consideration
            volatility_factor = self._calculate_volatility_factor(
                current_volatility, historical_volatility, volatility_regime
            )
            
            # Calculate liquidity factor with multiple indicators
            liquidity_factor = self._calculate_liquidity_factor(
                liquidity_score, bid_ask_spread, market_depth
            )
            
            # Calculate stress factor with regime consideration
            stress_factor = self._calculate_stress_factor(stress_level, stress_regime)
            
            # Combine factors with weighted average
            market_health_factor = (
                volatility_factor * 0.4 +  # Volatility has highest weight
                liquidity_factor * 0.35 +  # Liquidity is second most important
                stress_factor * 0.25       # Stress is least important
            )
            
            # Calculate market health leverage
            market_health_leverage = self.min_leverage + (self.max_leverage - self.min_leverage) * market_health_factor
            
            return max(self.min_leverage, min(self.max_leverage, market_health_leverage))
            
        except Exception as e:
            self.logger.error(f"Error extracting market health leverage: {e}")
            return self.min_leverage

    def _calculate_volatility_factor(self, current_vol: float, historical_vol: float, regime: str) -> float:
        """Calculate volatility factor with regime consideration."""
        try:
            # Define volatility thresholds
            low_vol_threshold = 0.01    # 1%
            normal_vol_threshold = 0.03  # 3%
            high_vol_threshold = 0.05    # 5%
            extreme_vol_threshold = 0.08 # 8%
            
            # Calculate volatility ratio (current vs historical)
            vol_ratio = current_vol / max(historical_vol, 0.001)
            
            # Base factor based on current volatility
            if current_vol <= low_vol_threshold:
                base_factor = 1.0  # Full leverage in low volatility
            elif current_vol <= normal_vol_threshold:
                base_factor = 0.9  # Slight reduction
            elif current_vol <= high_vol_threshold:
                base_factor = 0.7  # Moderate reduction
            elif current_vol <= extreme_vol_threshold:
                base_factor = 0.4  # Significant reduction
            else:
                base_factor = 0.2  # Extreme reduction
            
            # Adjust based on volatility regime
            if regime == "low_volatility":
                base_factor *= 1.1  # Increase leverage in low vol regime
            elif regime == "high_volatility":
                base_factor *= 0.8  # Decrease leverage in high vol regime
            elif regime == "extreme_volatility":
                base_factor *= 0.5  # Significant decrease in extreme vol
            
            # Adjust based on volatility ratio (current vs historical)
            if vol_ratio > 1.5:  # Current vol is 50% higher than historical
                base_factor *= 0.8
            elif vol_ratio < 0.7:  # Current vol is 30% lower than historical
                base_factor *= 1.1
            
            return max(0.1, min(1.0, base_factor))
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility factor: {e}")
            return 0.5

    def _calculate_liquidity_factor(self, liquidity_score: float, bid_ask_spread: float, market_depth: float) -> float:
        """Calculate liquidity factor with multiple indicators."""
        try:
            # Define liquidity thresholds
            tight_spread = 0.0005  # 0.05%
            normal_spread = 0.001   # 0.1%
            wide_spread = 0.002     # 0.2%
            
            # Calculate spread factor
            if bid_ask_spread <= tight_spread:
                spread_factor = 1.0
            elif bid_ask_spread <= normal_spread:
                spread_factor = 0.9
            elif bid_ask_spread <= wide_spread:
                spread_factor = 0.7
            else:
                spread_factor = 0.5
            
            # Calculate depth factor
            depth_factor = market_depth  # Direct use of market depth score
            
            # Calculate overall liquidity factor
            liquidity_factor = (liquidity_score * 0.4 + spread_factor * 0.4 + depth_factor * 0.2)
            
            return max(0.1, min(1.0, liquidity_factor))
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity factor: {e}")
            return 0.5

    def _calculate_stress_factor(self, stress_level: float, regime: str) -> float:
        """Calculate stress factor with regime consideration."""
        try:
            # Define stress thresholds
            low_stress = 0.2
            normal_stress = 0.5
            high_stress = 0.7
            extreme_stress = 0.9
            
            # Base factor based on stress level
            if stress_level <= low_stress:
                base_factor = 1.0  # Full leverage in low stress
            elif stress_level <= normal_stress:
                base_factor = 0.9  # Slight reduction
            elif stress_level <= high_stress:
                base_factor = 0.7  # Moderate reduction
            elif stress_level <= extreme_stress:
                base_factor = 0.4  # Significant reduction
            else:
                base_factor = 0.2  # Extreme reduction
            
            # Adjust based on stress regime
            if regime == "low_stress":
                base_factor *= 1.1  # Increase leverage in low stress
            elif regime == "high_stress":
                base_factor *= 0.8  # Decrease leverage in high stress
            elif regime == "extreme_stress":
                base_factor *= 0.5  # Significant decrease in extreme stress
            elif regime == "crisis":
                base_factor *= 0.3  # Minimal leverage in crisis
            
            return max(0.1, min(1.0, base_factor))
            
        except Exception as e:
            self.logger.error(f"Error calculating stress factor: {e}")
            return 0.5

    def _calculate_weighted_leverage(
        self,
        ml_leverage: float,
        liquidation_leverage: float,
        market_health_leverage: float,
    ) -> float:
        """Calculate weighted leverage using component indicators."""
        try:
            # Calculate weighted leverage
            weighted_leverage = (
                ml_leverage * self.ml_weight +
                liquidation_leverage * self.liquidation_risk_weight +
                market_health_leverage * self.market_health_weight
            ) / (self.ml_weight + self.liquidation_risk_weight + self.market_health_weight)
            
            return max(self.min_leverage, min(self.max_leverage, weighted_leverage))
            
        except Exception as e:
            self.logger.error(f"Error calculating weighted leverage: {e}")
            return ml_leverage

    def _generate_leverage_reason(
        self,
        final_leverage: float,
        ml_leverage: float,
        liquidation_leverage: float,
        market_health_leverage: float,
        movement_confidence: dict[str, float],
        adverse_movement_risks: dict[str, float],
    ) -> str:
        """Generate reason for leverage decision."""
        try:
            # Get average confidence and risk
            key_levels = [0.5, 1.0, 1.5, 2.0]
            confidences = []
            risks = []
            
            for level in key_levels:
                closest_confidence = min(movement_confidence.keys(), key=lambda x: abs(float(x) - level))
                closest_risk = min(adverse_movement_risks.keys(), key=lambda x: abs(float(x) - level))
                confidences.append(movement_confidence.get(closest_confidence, 0.5))
                risks.append(adverse_movement_risks.get(closest_risk, 0.3))
            
            avg_confidence = sum(confidences) / len(confidences)
            avg_risk = sum(risks) / len(risks)
            
            if final_leverage >= self.max_leverage * 0.8:
                return f"Maximum leverage due to high confidence ({avg_confidence:.2f}) and low risk ({avg_risk:.2f})"
            elif final_leverage >= self.max_leverage * 0.5:
                return f"High leverage based on ML confidence ({ml_leverage:.2f}x) and liquidation safety ({liquidation_leverage:.2f}x)"
            elif final_leverage >= self.min_leverage * 2:
                return f"Moderate leverage with balanced risk-reward profile"
            else:
                return f"Conservative leverage due to low confidence ({avg_confidence:.2f}) or high risk ({avg_risk:.2f})"
                
        except Exception as e:
            self.logger.error(f"Error generating leverage reason: {e}")
            return "Leverage calculated using ML intelligence and liquidation risk analysis"

    def get_leverage_sizing_history(self, limit: Optional[int] = None) -> List[dict[str, Any]]:
        """Get leverage sizing history."""
        if limit:
            return self.leverage_sizing_history[-limit:]
        return self.leverage_sizing_history.copy()

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="leverage sizer cleanup",
    )
    async def stop(self) -> None:
        """Stop the leverage sizer."""
        try:
            self.logger.info("Stopping leverage sizer...")
            self.is_initialized = False
            self.logger.info("✅ Leverage sizer stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping leverage sizer: {e}")


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="leverage sizer setup",
)
async def setup_leverage_sizer(
    config: dict[str, Any] | None = None,
) -> LeverageSizer | None:
    """
    Setup leverage sizer.

    Args:
        config: Configuration dictionary

    Returns:
        Optional[LeverageSizer]: Initialized leverage sizer or None
    """
    try:
        if config is None:
            config = {}

        leverage_sizer = LeverageSizer(config)
        
        if await leverage_sizer.initialize():
            return leverage_sizer
        else:
            return None

    except Exception as e:
        system_logger.error(f"Error setting up leverage sizer: {e}")
        return None
