# src/tactician/leverage_sizer.py

"""
Leverage Sizer for high leverage trading.
Uses ML confidence scores, liquidation risk model, and intelligence from other components.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger


class LeverageSizer:
    """
    Leverage sizer that uses ML confidence scores, liquidation risk model, and intelligence
    from Strategist, Analyst, and Governor components.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("LeverageSizer")

        # Load configuration
        self.leverage_config: dict[str, Any] = self.config.get("leverage_sizing", {})
        self.max_leverage: float = self.leverage_config.get("max_leverage", 10.0)
        self.min_leverage: float = self.leverage_config.get("min_leverage", 1.0)
        self.confidence_threshold: float = self.leverage_config.get("confidence_threshold", 0.7)
        self.risk_tolerance: float = self.leverage_config.get("risk_tolerance", 0.3)
        
        # Load combined sizing configuration
        self.combined_sizing_config: dict[str, Any] = self._load_combined_sizing_config()
        
        # Component weights
        self.ml_weight: float = self.leverage_config.get("ml_weight", 0.4)
        self.liquidation_risk_weight: float = self.leverage_config.get("liquidation_risk_weight", 0.3)
        self.strategist_weight: float = self.leverage_config.get("strategist_weight", 0.2)
        self.analyst_weight: float = self.leverage_config.get("analyst_weight", 0.1)
        
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

            # Validate combined sizing config
            if not self._validate_combined_sizing_config():
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

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="combined sizing config loading",
    )
    def _load_combined_sizing_config(self) -> dict[str, Any]:
        """Load combined sizing configuration from YAML file."""
        try:
            config_path = "config/combined_sizing.yaml"
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            self.logger.info(f"Loaded combined sizing config from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Error loading combined sizing config: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="combined sizing config validation",
    )
    def _validate_combined_sizing_config(self) -> bool:
        """Validate combined sizing configuration."""
        try:
            if not self.combined_sizing_config:
                self.logger.error("Combined sizing config is empty")
                return False

            # Check for required sections
            required_sections = ["indicators", "weights", "thresholds"]
            for section in required_sections:
                if section not in self.combined_sizing_config:
                    self.logger.error(f"Missing required section in combined sizing config: {section}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating combined sizing config: {e}")
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
        strategist_results: Optional[dict[str, Any]] = None,
        analyst_results: Optional[dict[str, Any]] = None,
        governor_results: Optional[dict[str, Any]] = None,
        current_price: float = 0.0,
        target_direction: str = "long",
    ) -> dict[str, Any]:
        """
        Calculate leverage using ML confidence scores, liquidation risk analysis, and component intelligence.
        
        Args:
            ml_predictions: ML confidence predictions from ml_confidence_predictor
            liquidation_risk_analysis: Liquidation risk analysis from liquidation_risk_model
            strategist_results: Results from Strategist component
            analyst_results: Results from Analyst component
            governor_results: Results from Governor component
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

            # Get component indicators
            strategist_indicators = self._extract_strategist_leverage_indicators(strategist_results)
            analyst_indicators = self._extract_analyst_leverage_indicators(analyst_results)
            governor_indicators = self._extract_governor_leverage_indicators(governor_results)

            # Calculate weighted leverage
            weighted_leverage = self._calculate_weighted_leverage(
                ml_leverage,
                liquidation_leverage,
                strategist_indicators,
                analyst_indicators,
                governor_indicators,
            )

            # Apply combined sizing indicators
            final_leverage = self._apply_combined_leverage_indicators(
                weighted_leverage,
                strategist_indicators,
                analyst_indicators,
                governor_indicators,
            )

            # Create leverage sizing analysis
            leverage_analysis = {
                "timestamp": datetime.now(),
                "current_price": current_price,
                "target_direction": target_direction,
                "ml_leverage": ml_leverage,
                "liquidation_leverage": liquidation_leverage,
                "weighted_leverage": weighted_leverage,
                "final_leverage": final_leverage,
                "component_indicators": {
                    "strategist": strategist_indicators,
                    "analyst": analyst_indicators,
                    "governor": governor_indicators,
                },
                "ml_confidence_scores": movement_confidence,
                "adverse_movement_risks": adverse_movement_risks,
                "directional_confidence": directional_confidence,
                "leverage_reason": self._generate_leverage_reason(
                    final_leverage, ml_leverage, liquidation_leverage, movement_confidence, adverse_movement_risks
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
            
            # Base leverage calculation
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

    def _extract_strategist_leverage_indicators(self, strategist_results: Optional[dict[str, Any]]) -> dict[str, float]:
        """Extract leverage indicators from Strategist results."""
        try:
            if not strategist_results:
                return {}
            
            indicators = {}
            
            # Extract strategy confidence
            strategy = strategist_results.get("strategy", {})
            indicators["strategy_confidence"] = strategy.get("confidence", 0.5)
            indicators["risk_score"] = strategy.get("risk_score", 0.5)
            
            # Extract leverage recommendations
            ml_strategy = strategy.get("ml_strategy", {})
            leverage_recommendations = ml_strategy.get("leverage_recommendations", {})
            indicators["recommended_leverage"] = leverage_recommendations.get("recommended_leverage", self.min_leverage)
            indicators["max_safe_leverage"] = leverage_recommendations.get("max_safe_leverage", self.max_leverage)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error extracting Strategist leverage indicators: {e}")
            return {}

    def _extract_analyst_leverage_indicators(self, analyst_results: Optional[dict[str, Any]]) -> dict[str, float]:
        """Extract leverage indicators from Analyst results."""
        try:
            if not analyst_results:
                return {}
            
            indicators = {}
            
            # Extract market analysis
            market_analysis = analyst_results.get("market_analysis", {})
            indicators["market_sentiment"] = market_analysis.get("sentiment", 0.5)
            indicators["trend_strength"] = market_analysis.get("trend_strength", 0.5)
            indicators["volatility"] = market_analysis.get("volatility", 0.02)
            
            # Extract volatility-based leverage adjustment
            volatility = indicators["volatility"]
            if volatility > 0.05:  # High volatility
                indicators["volatility_leverage_factor"] = 0.5
            elif volatility > 0.03:  # Medium volatility
                indicators["volatility_leverage_factor"] = 0.8
            else:  # Low volatility
                indicators["volatility_leverage_factor"] = 1.0
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error extracting Analyst leverage indicators: {e}")
            return {}

    def _extract_governor_leverage_indicators(self, governor_results: Optional[dict[str, Any]]) -> dict[str, float]:
        """Extract leverage indicators from Governor results."""
        try:
            if not governor_results:
                return {}
            
            indicators = {}
            
            # Extract governance decisions
            governance_decisions = governor_results.get("governance_decisions", {})
            
            # Extract leverage decisions
            leverage_decisions = governance_decisions.get("leverage_decisions", {})
            if leverage_decisions:
                # Get average recommended leverage
                leverages = [decision.get("recommended_leverage", self.min_leverage) for decision in leverage_decisions.values()]
                indicators["governor_recommended_leverage"] = sum(leverages) / len(leverages) if leverages else self.min_leverage
            else:
                indicators["governor_recommended_leverage"] = self.min_leverage
            
            # Extract liquidation risk decisions
            liquidation_risk_decisions = governance_decisions.get("liquidation_risk_decisions", {})
            if liquidation_risk_decisions:
                # Count safe leverage decisions
                safe_decisions = sum(1 for decision in liquidation_risk_decisions.values() 
                                  if decision.get("action", "") in ["enter_position", "enter_position_cautious"])
                total_decisions = len(liquidation_risk_decisions)
                indicators["liquidation_safety_ratio"] = safe_decisions / total_decisions if total_decisions > 0 else 0.5
            else:
                indicators["liquidation_safety_ratio"] = 0.5
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error extracting Governor leverage indicators: {e}")
            return {}

    def _calculate_weighted_leverage(
        self,
        ml_leverage: float,
        liquidation_leverage: float,
        strategist_indicators: dict[str, float],
        analyst_indicators: dict[str, float],
        governor_indicators: dict[str, float],
    ) -> float:
        """Calculate weighted leverage using component indicators."""
        try:
            # Base leverage from ML and liquidation risk
            base_leverage = (ml_leverage * self.ml_weight + liquidation_leverage * self.liquidation_risk_weight) / (self.ml_weight + self.liquidation_risk_weight)
            
            # Strategist adjustment
            strategist_adjustment = 1.0
            if strategist_indicators:
                strategy_confidence = strategist_indicators.get("strategy_confidence", 0.5)
                recommended_leverage = strategist_indicators.get("recommended_leverage", self.min_leverage)
                strategist_adjustment = (strategy_confidence * 0.7) + (recommended_leverage / self.max_leverage * 0.3)
            
            # Analyst adjustment
            analyst_adjustment = 1.0
            if analyst_indicators:
                volatility_factor = analyst_indicators.get("volatility_leverage_factor", 1.0)
                market_sentiment = analyst_indicators.get("market_sentiment", 0.5)
                analyst_adjustment = volatility_factor * (0.7 + market_sentiment * 0.3)
            
            # Governor adjustment
            governor_adjustment = 1.0
            if governor_indicators:
                governor_recommended = governor_indicators.get("governor_recommended_leverage", self.min_leverage)
                liquidation_safety = governor_indicators.get("liquidation_safety_ratio", 0.5)
                governor_adjustment = (governor_recommended / self.max_leverage * 0.6) + (liquidation_safety * 0.4)
            
            # Calculate weighted leverage
            weighted_leverage = base_leverage * (
                self.ml_weight +
                self.liquidation_risk_weight +
                self.strategist_weight * strategist_adjustment +
                self.analyst_weight * analyst_adjustment +
                self.governor_weight * governor_adjustment
            )
            
            return max(self.min_leverage, min(self.max_leverage, weighted_leverage))
            
        except Exception as e:
            self.logger.error(f"Error calculating weighted leverage: {e}")
            return ml_leverage

    def _apply_combined_leverage_indicators(
        self,
        weighted_leverage: float,
        strategist_indicators: dict[str, float],
        analyst_indicators: dict[str, float],
        governor_indicators: dict[str, float],
    ) -> float:
        """Apply combined leverage indicators from config."""
        try:
            final_leverage = weighted_leverage
            
            # Apply volatility adjustment
            if analyst_indicators:
                volatility = analyst_indicators.get("volatility", 0.02)
                volatility_threshold = self.combined_sizing_config.get("thresholds", {}).get("volatility", 0.03)
                if volatility > volatility_threshold:
                    # Reduce leverage for high volatility
                    volatility_factor = volatility_threshold / volatility
                    final_leverage *= volatility_factor
            
            # Apply risk score adjustment
            if strategist_indicators:
                risk_score = strategist_indicators.get("risk_score", 0.5)
                risk_threshold = self.combined_sizing_config.get("thresholds", {}).get("risk_score", 0.7)
                if risk_score > risk_threshold:
                    # Reduce leverage for high risk
                    risk_factor = risk_threshold / risk_score
                    final_leverage *= risk_factor
            
            # Apply liquidation safety adjustment
            if governor_indicators:
                liquidation_safety = governor_indicators.get("liquidation_safety_ratio", 0.5)
                safety_threshold = self.combined_sizing_config.get("thresholds", {}).get("liquidation_safety", 0.6)
                if liquidation_safety < safety_threshold:
                    # Reduce leverage for low liquidation safety
                    safety_factor = liquidation_safety / safety_threshold
                    final_leverage *= safety_factor
            
            return max(self.min_leverage, min(self.max_leverage, final_leverage))
            
        except Exception as e:
            self.logger.error(f"Error applying combined leverage indicators: {e}")
            return weighted_leverage

    def _generate_leverage_reason(
        self,
        final_leverage: float,
        ml_leverage: float,
        liquidation_leverage: float,
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
