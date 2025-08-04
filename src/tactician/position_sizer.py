src/tactician/position_sizer.py

"""
Position Sizer for high leverage trading.
Uses ML confidence scores, Kelly criterion, and intelligence from other components.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger


class PositionSizer:
    """
    Position sizer that uses ML confidence scores, Kelly criterion, and intelligence
    from Strategist, Analyst, and Governor components.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("PositionSizer")

        # Load configuration
        self.sizing_config: dict[str, Any] = self.config.get("position_sizing", {})
        self.kelly_multiplier: float = self.sizing_config.get("kelly_multiplier", 0.25)
        self.max_position_size: float = self.sizing_config.get("max_position_size", 0.5)
        self.min_position_size: float = self.sizing_config.get("min_position_size", 0.01)
        self.confidence_threshold: float = self.sizing_config.get("confidence_threshold", 0.6)
        
        # Load combined sizing configuration
        self.combined_sizing_config: dict[str, Any] = self._load_combined_sizing_config()
        
        # Component weights
        self.ml_weight: float = self.sizing_config.get("ml_weight", 0.4)
        self.strategist_weight: float = self.sizing_config.get("strategist_weight", 0.3)
        self.analyst_weight: float = self.sizing_config.get("analyst_weight", 0.2)
        self.governor_weight: float = self.sizing_config.get("governor_weight", 0.1)
        
        self.is_initialized: bool = False
        self.position_sizing_history: List[dict[str, Any]] = []

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid position sizer configuration"),
            AttributeError: (False, "Missing required sizing parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="position sizer initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the position sizer."""
        try:
            self.logger.info("Initializing position sizer...")

            # Validate configuration
            if not self._validate_configuration():
                return False

            # Validate combined sizing config
            if not self._validate_combined_sizing_config():
                return False

            self.is_initialized = True
            self.logger.info("✅ Position sizer initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing position sizer: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """Validate position sizer configuration."""
        try:
            required_keys = ["kelly_multiplier", "max_position_size", "min_position_size"]
            for key in required_keys:
                if key not in self.sizing_config:
                    self.logger.error(f"Missing required configuration key: {key}")
                    return False

            if self.max_position_size <= self.min_position_size:
                self.logger.error("max_position_size must be greater than min_position_size")
                return False

            if self.kelly_multiplier <= 0 or self.kelly_multiplier > 1:
                self.logger.error("kelly_multiplier must be between 0 and 1")
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
            ValueError: (None, "Invalid input data for position sizing"),
            AttributeError: (None, "Sizer not properly initialized"),
        },
        default_return=None,
        context="position sizing calculation",
    )
    async def calculate_position_size(
        self,
        ml_predictions: dict[str, Any],
        strategist_results: Optional[dict[str, Any]] = None,
        analyst_results: Optional[dict[str, Any]] = None,
        governor_results: Optional[dict[str, Any]] = None,
        current_price: float = 0.0,
        account_balance: float = 1000.0,
    ) -> dict[str, Any]:
        """
        Calculate position size using ML confidence scores, Kelly criterion, and component intelligence.
        
        Args:
            ml_predictions: ML confidence predictions from ml_confidence_predictor
            strategist_results: Results from Strategist component
            analyst_results: Results from Analyst component
            governor_results: Results from Governor component
            current_price: Current market price
            account_balance: Account balance for position sizing
            
        Returns:
            dict[str, Any]: Position sizing analysis
        """
        try:
            if not self.is_initialized:
                self.logger.error("Position sizer not initialized")
                return None

            self.logger.info("Calculating position size using ML intelligence...")

            # Extract ML confidence scores
            movement_confidence = ml_predictions.get("movement_confidence_scores", {})
            adverse_movement_risks = ml_predictions.get("adverse_movement_risks", {})
            directional_confidence = ml_predictions.get("directional_confidence", {})

            # Calculate base Kelly criterion position size
            kelly_position_size = self._calculate_kelly_position_size(movement_confidence, adverse_movement_risks)

            # Get component indicators
            strategist_indicators = self._extract_strategist_indicators(strategist_results)
            analyst_indicators = self._extract_analyst_indicators(analyst_results)
            governor_indicators = self._extract_governor_indicators(governor_results)

            # Calculate weighted position size
            weighted_position_size = self._calculate_weighted_position_size(
                kelly_position_size,
                strategist_indicators,
                analyst_indicators,
                governor_indicators,
            )

            # Apply combined sizing indicators
            final_position_size = self._apply_combined_sizing_indicators(
                weighted_position_size,
                strategist_indicators,
                analyst_indicators,
                governor_indicators,
            )

            # Create position sizing analysis
            sizing_analysis = {
                "timestamp": datetime.now(),
                "current_price": current_price,
                "account_balance": account_balance,
                "kelly_position_size": kelly_position_size,
                "weighted_position_size": weighted_position_size,
                "final_position_size": final_position_size,
                "component_indicators": {
                    "strategist": strategist_indicators,
                    "analyst": analyst_indicators,
                    "governor": governor_indicators,
                },
                "ml_confidence_scores": movement_confidence,
                "adverse_movement_risks": adverse_movement_risks,
                "directional_confidence": directional_confidence,
                "sizing_reason": self._generate_sizing_reason(
                    final_position_size, kelly_position_size, movement_confidence, adverse_movement_risks
                ),
            }

            # Store in history
            self.position_sizing_history.append(sizing_analysis)
            if len(self.position_sizing_history) > 100:  # Keep last 100 entries
                self.position_sizing_history = self.position_sizing_history[-100:]

            self.logger.info(f"✅ Position size calculated: {final_position_size:.4f}")
            return sizing_analysis

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return None

    def _calculate_kelly_position_size(
        self, movement_confidence: dict[str, float], adverse_movement_risks: dict[str, float]
    ) -> float:
        """Calculate position size using Kelly criterion based on ML confidence scores."""
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
            
            # Kelly criterion: f = (bp - q) / b
            # where b = odds received, p = probability of win, q = probability of loss
            # For our case: b = 1 (1:1 odds), p = avg_confidence, q = avg_adverse_risk
            kelly_fraction = avg_confidence - avg_adverse_risk
            
            # Apply Kelly multiplier for conservative sizing
            kelly_position_size = kelly_fraction * self.kelly_multiplier
            
            # Ensure within bounds
            kelly_position_size = max(self.min_position_size, min(self.max_position_size, kelly_position_size))
            
            return kelly_position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly position size: {e}")
            return self.min_position_size

    def _extract_strategist_indicators(self, strategist_results: Optional[dict[str, Any]]) -> dict[str, float]:
        """Extract relevant indicators from Strategist results."""
        try:
            if not strategist_results:
                return {}
            
            indicators = {}
            
            # Extract strategy confidence
            strategy = strategist_results.get("strategy", {})
            indicators["strategy_confidence"] = strategy.get("confidence", 0.5)
            indicators["risk_score"] = strategy.get("risk_score", 0.5)
            
            # Extract position sizing recommendations
            ml_strategy = strategy.get("ml_strategy", {})
            position_sizing = ml_strategy.get("position_sizing", {})
            indicators["recommended_size"] = position_sizing.get("recommended_size", 0.1)
            indicators["max_size"] = position_sizing.get("max_size", 0.5)
            
            # Extract risk parameters
            risk_parameters = ml_strategy.get("risk_parameters", {})
            indicators["stop_loss"] = risk_parameters.get("stop_loss", 0.02)
            indicators["take_profit"] = risk_parameters.get("take_profit", 0.03)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error extracting Strategist indicators: {e}")
            return {}

    def _extract_analyst_indicators(self, analyst_results: Optional[dict[str, Any]]) -> dict[str, float]:
        """Extract relevant indicators from Analyst results."""
        try:
            if not analyst_results:
                return {}
            
            indicators = {}
            
            # Extract market analysis
            market_analysis = analyst_results.get("market_analysis", {})
            indicators["market_sentiment"] = market_analysis.get("sentiment", 0.5)
            indicators["trend_strength"] = market_analysis.get("trend_strength", 0.5)
            indicators["volatility"] = market_analysis.get("volatility", 0.02)
            
            # Extract ML confidence scores
            ml_confidence_scores = analyst_results.get("ml_confidence_scores", {})
            if ml_confidence_scores:
                # Get average confidence for key levels
                key_levels = [0.5, 1.0, 1.5, 2.0]
                confidences = []
                for level in key_levels:
                    closest_level = min(ml_confidence_scores.keys(), key=lambda x: abs(float(x) - level))
                    confidence = ml_confidence_scores.get(closest_level, 0.5)
                    confidences.append(confidence)
                indicators["avg_ml_confidence"] = sum(confidences) / len(confidences)
            else:
                indicators["avg_ml_confidence"] = 0.5
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error extracting Analyst indicators: {e}")
            return {}

    def _extract_governor_indicators(self, governor_results: Optional[dict[str, Any]]) -> dict[str, float]:
        """Extract relevant indicators from Governor results."""
        try:
            if not governor_results:
                return {}
            
            indicators = {}
            
            # Extract governance decisions
            governance_decisions = governor_results.get("governance_decisions", {})
            
            # Extract entry decisions
            entry_decisions = governance_decisions.get("entry_decisions", {})
            if entry_decisions:
                # Count positive entry decisions
                positive_entries = sum(1 for decision in entry_decisions.values() 
                                    if decision.get("should_enter", False))
                total_entries = len(entry_decisions)
                indicators["entry_confidence"] = positive_entries / total_entries if total_entries > 0 else 0.5
            else:
                indicators["entry_confidence"] = 0.5
            
            # Extract position sizing decisions
            position_sizing_decisions = governance_decisions.get("position_sizing_decisions", {})
            if position_sizing_decisions:
                # Get average recommended size
                sizes = [decision.get("recommended_size", 0.1) for decision in position_sizing_decisions.values()]
                indicators["governor_recommended_size"] = sum(sizes) / len(sizes) if sizes else 0.1
            else:
                indicators["governor_recommended_size"] = 0.1
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error extracting Governor indicators: {e}")
            return {}

    def _calculate_weighted_position_size(
        self,
        kelly_position_size: float,
        strategist_indicators: dict[str, float],
        analyst_indicators: dict[str, float],
        governor_indicators: dict[str, float],
    ) -> float:
        """Calculate weighted position size using component indicators."""
        try:
            # Base position size from Kelly criterion
            base_size = kelly_position_size
            
            # Strategist adjustment
            strategist_adjustment = 1.0
            if strategist_indicators:
                strategy_confidence = strategist_indicators.get("strategy_confidence", 0.5)
                recommended_size = strategist_indicators.get("recommended_size", 0.1)
                strategist_adjustment = (strategy_confidence * 0.7) + (recommended_size * 0.3)
            
            # Analyst adjustment
            analyst_adjustment = 1.0
            if analyst_indicators:
                market_sentiment = analyst_indicators.get("market_sentiment", 0.5)
                trend_strength = analyst_indicators.get("trend_strength", 0.5)
                avg_ml_confidence = analyst_indicators.get("avg_ml_confidence", 0.5)
                analyst_adjustment = (market_sentiment * 0.3) + (trend_strength * 0.3) + (avg_ml_confidence * 0.4)
            
            # Governor adjustment
            governor_adjustment = 1.0
            if governor_indicators:
                entry_confidence = governor_indicators.get("entry_confidence", 0.5)
                governor_recommended_size = governor_indicators.get("governor_recommended_size", 0.1)
                governor_adjustment = (entry_confidence * 0.6) + (governor_recommended_size * 0.4)
            
            # Calculate weighted position size
            weighted_size = base_size * (
                self.ml_weight +
                self.strategist_weight * strategist_adjustment +
                self.analyst_weight * analyst_adjustment +
                self.governor_weight * governor_adjustment
            )
            
            return max(self.min_position_size, min(self.max_position_size, weighted_size))
            
        except Exception as e:
            self.logger.error(f"Error calculating weighted position size: {e}")
            return kelly_position_size

    def _apply_combined_sizing_indicators(
        self,
        weighted_position_size: float,
        strategist_indicators: dict[str, float],
        analyst_indicators: dict[str, float],
        governor_indicators: dict[str, float],
    ) -> float:
        """Apply combined sizing indicators from config."""
        try:
            final_size = weighted_position_size
            
            # Apply volatility adjustment
            if analyst_indicators:
                volatility = analyst_indicators.get("volatility", 0.02)
                volatility_threshold = self.combined_sizing_config.get("thresholds", {}).get("volatility", 0.03)
                if volatility > volatility_threshold:
                    # Reduce position size for high volatility
                    volatility_factor = volatility_threshold / volatility
                    final_size *= volatility_factor
            
            # Apply risk score adjustment
            if strategist_indicators:
                risk_score = strategist_indicators.get("risk_score", 0.5)
                risk_threshold = self.combined_sizing_config.get("thresholds", {}).get("risk_score", 0.7)
                if risk_score > risk_threshold:
                    # Reduce position size for high risk
                    risk_factor = risk_threshold / risk_score
                    final_size *= risk_factor
            
            # Apply sentiment adjustment
            if analyst_indicators:
                sentiment = analyst_indicators.get("market_sentiment", 0.5)
                sentiment_threshold = self.combined_sizing_config.get("thresholds", {}).get("sentiment", 0.3)
                if sentiment < sentiment_threshold:
                    # Reduce position size for negative sentiment
                    sentiment_factor = sentiment / sentiment_threshold
                    final_size *= sentiment_factor
            
            return max(self.min_position_size, min(self.max_position_size, final_size))
            
        except Exception as e:
            self.logger.error(f"Error applying combined sizing indicators: {e}")
            return weighted_position_size

    def _generate_sizing_reason(
        self,
        final_position_size: float,
        kelly_position_size: float,
        movement_confidence: dict[str, float],
        adverse_movement_risks: dict[str, float],
    ) -> str:
        """Generate reason for position sizing decision."""
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
            
            if final_position_size >= self.max_position_size * 0.8:
                return f"Maximum position size due to high confidence ({avg_confidence:.2f}) and low risk ({avg_risk:.2f})"
            elif final_position_size >= self.max_position_size * 0.5:
                return f"Large position size based on Kelly criterion ({kelly_position_size:.3f}) and good conditions"
            elif final_position_size >= self.min_position_size * 2:
                return f"Moderate position size with balanced risk-reward profile"
            else:
                return f"Conservative position size due to low confidence ({avg_confidence:.2f}) or high risk ({avg_risk:.2f})"
                
        except Exception as e:
            self.logger.error(f"Error generating sizing reason: {e}")
            return "Position size calculated using ML intelligence"

    def get_position_sizing_history(self, limit: Optional[int] = None) -> List[dict[str, Any]]:
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
            self.logger.error(f"Error stopping position sizer: {e}")


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="position sizer setup",
)
async def setup_position_sizer(
    config: dict[str, Any] | None = None,
) -> PositionSizer | None:
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
        else:
            return None

    except Exception as e:
        system_logger.error(f"Error setting up position sizer: {e}")
        return None
