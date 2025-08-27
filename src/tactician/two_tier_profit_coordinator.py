# src/tactician/two_tier_profit_coordinator.py

"""
Two-Tier Profit Coordinator for coordinating profit predictions between Analyst and Tactician.
"""

from datetime import datetime
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors, handle_specific_errors


class TwoTierProfitCoordinator:
    """
    Coordinates profit predictions between Analyst and Tactician tiers.
    Combines predictions with weights and provides feedback loops.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the two-tier profit coordinator.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("TwoTierProfitCoordinator")
        
        # Configuration
        self.coordinator_config: dict[str, Any] = self.config.get("two_tier_profit_coordinator", {})
        self.analyst_weight: float = self.coordinator_config.get("analyst_weight", 0.7)
        self.tactician_weight: float = self.coordinator_config.get("tactician_weight", 0.3)
        self.confidence_threshold: float = self.coordinator_config.get("confidence_threshold", 0.6)
        
        # State
        self.coordination_history: list[dict[str, Any]] = []
        self.max_history: int = self.coordinator_config.get("max_history", 100)
        
        # Performance tracking
        self.analyst_performance: list[float] = []
        self.tactician_performance: list[float] = []
        self.combined_performance: list[float] = []

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid coordination parameters"),
            AttributeError: (False, "Missing coordination components"),
            KeyError: (False, "Missing required coordination data"),
        },
        default_return={},
        context="profit coordination",
    )
    async def coordinate_profit_predictions(
        self, 
        analyst_results: dict[str, Any], 
        tactician_results: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Coordinate and reconcile profit predictions between Analyst and Tactician tiers.

        Args:
            analyst_results: Analysis results from Analyst tier
            tactician_results: Execution results from Tactician tier

        Returns:
            dict: Coordinated profit predictions and confidence scores
        """
        try:
            self.logger.info("🔄 Coordinating profit predictions between Analyst and Tactician...")

            # Extract profit predictions from both tiers
            analyst_profit = analyst_results.get("profit_predictions", {})
            tactician_profit = tactician_results.get("profit_predictions", {})
            
            # Extract confidence scores
            analyst_confidence = analyst_results.get("enhanced_confidence", 0.5)
            tactician_confidence = tactician_results.get("enhanced_confidence", 0.5)

            # Combine profit predictions with dynamic weights
            combined_profit = self._combine_profit_predictions(
                analyst_profit=analyst_profit,
                tactician_profit=tactician_profit,
                analyst_confidence=analyst_confidence,
                tactician_confidence=tactician_confidence
            )

            # Calculate combined confidence
            combined_confidence = self._calculate_combined_confidence(
                analyst_results=analyst_results,
                tactician_results=tactician_results
            )

            # Create coordination results
            coordination_results = {
                "timestamp": datetime.now().isoformat(),
                "combined_profit": combined_profit,
                "combined_confidence": combined_confidence,
                "analyst_profit": analyst_profit,
                "tactician_profit": tactician_profit,
                "analyst_confidence": analyst_confidence,
                "tactician_confidence": tactician_confidence,
                "coordination_weights": {
                    "analyst_weight": self.analyst_weight,
                    "tactician_weight": self.tactician_weight
                },
                "coordination_status": "completed"
            }

            # Store coordination results
            self._store_coordination_results(coordination_results)

            self.logger.info("✅ Profit coordination completed successfully")
            return coordination_results

        except Exception as e:
            self.logger.error(f"❌ Profit coordination failed: {e}")
            return {}

    def _combine_profit_predictions(
        self, 
        analyst_profit: dict[str, Any], 
        tactician_profit: dict[str, Any],
        analyst_confidence: float,
        tactician_confidence: float
    ) -> dict[str, Any]:
        """
        Combine profit predictions from both tiers with dynamic weights.

        Args:
            analyst_profit: Profit predictions from Analyst
            tactician_profit: Profit predictions from Tactician
            analyst_confidence: Analyst confidence score
            tactician_confidence: Tactician confidence score

        Returns:
            dict: Combined profit predictions
        """
        try:
            # Extract profit values
            analyst_profit_value = analyst_profit.get("profit", 0.0)
            tactician_profit_value = tactician_profit.get("profit", 0.0)
            
            # Extract direction values
            analyst_direction = analyst_profit.get("direction", 0)
            tactician_direction = tactician_profit.get("direction", 0)
            
            # Extract high-value factors
            analyst_high_value = analyst_profit.get("high_value_trades", 0.0)
            tactician_high_value = tactician_profit.get("high_value_trades", 0.0)

            # Calculate dynamic weights based on confidence
            total_confidence = analyst_confidence + tactician_confidence
            if total_confidence > 0:
                dynamic_analyst_weight = analyst_confidence / total_confidence
                dynamic_tactician_weight = tactician_confidence / total_confidence
            else:
                dynamic_analyst_weight = self.analyst_weight
                dynamic_tactician_weight = self.tactician_weight

            # Combine profit predictions
            combined_profit_value = (
                dynamic_analyst_weight * analyst_profit_value +
                dynamic_tactician_weight * tactician_profit_value
            )

            # Combine direction (weighted average for continuous values, majority for discrete)
            if isinstance(analyst_direction, (int, float)) and isinstance(tactician_direction, (int, float)):
                combined_direction = (
                    dynamic_analyst_weight * analyst_direction +
                    dynamic_tactician_weight * tactician_direction
                )
            else:
                # For discrete directions, use confidence-weighted majority
                combined_direction = analyst_direction if analyst_confidence > tactician_confidence else tactician_direction

            # Combine high-value factors
            combined_high_value = (
                dynamic_analyst_weight * analyst_high_value +
                dynamic_tactician_weight * tactician_high_value
            )

            return {
                "profit": combined_profit_value,
                "direction": combined_direction,
                "high_value_trades": combined_high_value,
                "analyst_contribution": {
                    "profit": analyst_profit_value,
                    "direction": analyst_direction,
                    "high_value": analyst_high_value,
                    "weight": dynamic_analyst_weight
                },
                "tactician_contribution": {
                    "profit": tactician_profit_value,
                    "direction": tactician_direction,
                    "high_value": tactician_high_value,
                    "weight": dynamic_tactician_weight
                }
            }

        except Exception as e:
            self.logger.error(f"Error combining profit predictions: {e}")
            # Return analyst predictions as fallback
            return analyst_profit

    def _calculate_combined_confidence(
        self, 
        analyst_results: dict[str, Any], 
        tactician_results: dict[str, Any]
    ) -> float:
        """
        Calculate combined confidence score from both tiers.

        Args:
            analyst_results: Analysis results from Analyst
            tactician_results: Execution results from Tactician

        Returns:
            float: Combined confidence score
        """
        try:
            # Extract confidence scores
            analyst_confidence = analyst_results.get("enhanced_confidence", 0.5)
            tactician_confidence = tactician_results.get("enhanced_confidence", 0.5)
            
            # Extract market health and risk information
            market_health = analyst_results.get("market_health", {})
            liquidation_risk = analyst_results.get("liquidation_risk", {})
            
            # Base combined confidence (weighted average)
            base_combined_confidence = (
                self.analyst_weight * analyst_confidence +
                self.tactician_weight * tactician_confidence
            )
            
            # Adjust based on market health
            market_health_score = self._calculate_market_health_score(market_health)
            health_adjustment = (market_health_score - 0.5) * 0.2  # ±10% adjustment
            
            # Adjust based on liquidation risk
            risk_score = self._calculate_risk_score(liquidation_risk)
            risk_adjustment = (0.5 - risk_score) * 0.2  # ±10% adjustment (lower risk = higher confidence)
            
            # Calculate final combined confidence
            combined_confidence = base_combined_confidence + health_adjustment + risk_adjustment
            
            # Ensure confidence stays within [0, 1] range
            return max(0.0, min(1.0, combined_confidence))

        except Exception as e:
            self.logger.error(f"Error calculating combined confidence: {e}")
            return 0.5

    def _calculate_market_health_score(self, market_health: dict[str, Any]) -> float:
        """Calculate market health score from market health analysis."""
        try:
            if not market_health:
                return 0.5
            
            # Extract health indicators (implement based on your market health structure)
            volatility_score = market_health.get("volatility_score", 0.5)
            liquidity_score = market_health.get("liquidity_score", 0.5)
            trend_score = market_health.get("trend_score", 0.5)
            
            # Calculate weighted average
            health_score = (volatility_score + liquidity_score + trend_score) / 3
            return health_score

        except Exception as e:
            self.logger.error(f"Error calculating market health score: {e}")
            return 0.5

    def _calculate_risk_score(self, liquidation_risk: dict[str, Any]) -> float:
        """Calculate risk score from liquidation risk analysis."""
        try:
            if not liquidation_risk:
                return 0.5
            
            # Extract risk indicators (implement based on your risk structure)
            risk_level = liquidation_risk.get("risk_level", 0.5)
            liquidation_probability = liquidation_risk.get("liquidation_probability", 0.5)
            
            # Calculate weighted risk score
            risk_score = (risk_level + liquidation_probability) / 2
            return risk_score

        except Exception as e:
            self.logger.error(f"Error calculating risk score: {e}")
            return 0.5

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="coordination results storage",
    )
    def _store_coordination_results(self, coordination_results: dict[str, Any]) -> None:
        """Store coordination results for later analysis."""
        try:
            # Add to history
            self.coordination_history.append(coordination_results)
            
            # Limit history size
            if len(self.coordination_history) > self.max_history:
                self.coordination_history = self.coordination_history[-self.max_history:]
            
            self.logger.debug(f"Stored coordination results, history size: {len(self.coordination_history)}")

        except Exception as e:
            self.logger.error(f"Error storing coordination results: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="performance feedback",
    )
    async def update_performance_feedback(
        self, 
        actual_profit: float, 
        predicted_profit: float,
        execution_quality: float = 0.5
    ) -> dict[str, Any]:
        """
        Update performance feedback and adjust weights.

        Args:
            actual_profit: Actual realized profit
            predicted_profit: Predicted profit from coordination
            execution_quality: Quality of execution (0-1)

        Returns:
            dict: Updated performance metrics and weight adjustments
        """
        try:
            self.logger.info("📊 Updating performance feedback...")

            # Calculate prediction accuracy
            prediction_error = abs(actual_profit - predicted_profit)
            prediction_accuracy = max(0.0, 1.0 - prediction_error / max(abs(actual_profit), 0.01))

            # Update performance tracking
            self.combined_performance.append(prediction_accuracy)

            # Calculate recent performance (last 10 predictions)
            recent_performance = np.mean(self.combined_performance[-10:]) if self.combined_performance else 0.5

            # Adjust weights based on performance
            weight_adjustment = self._calculate_weight_adjustment(recent_performance, execution_quality)
            
            # Apply weight adjustments
            self.analyst_weight = max(0.3, min(0.8, self.analyst_weight + weight_adjustment["analyst"]))
            self.tactician_weight = max(0.2, min(0.7, self.tactician_weight + weight_adjustment["tactician"]))

            # Normalize weights
            total_weight = self.analyst_weight + self.tactician_weight
            self.analyst_weight /= total_weight
            self.tactician_weight /= total_weight

            feedback_results = {
                "timestamp": datetime.now().isoformat(),
                "actual_profit": actual_profit,
                "predicted_profit": predicted_profit,
                "prediction_accuracy": prediction_accuracy,
                "recent_performance": recent_performance,
                "execution_quality": execution_quality,
                "weight_adjustments": weight_adjustment,
                "updated_weights": {
                    "analyst_weight": self.analyst_weight,
                    "tactician_weight": self.tactician_weight
                }
            }

            self.logger.info(f"✅ Performance feedback updated. Recent performance: {recent_performance:.3f}")
            return feedback_results

        except Exception as e:
            self.logger.error(f"❌ Performance feedback update failed: {e}")
            return {}

    def _calculate_weight_adjustment(self, recent_performance: float, execution_quality: float) -> dict[str, float]:
        """Calculate weight adjustments based on performance and execution quality."""
        try:
            # Base adjustment from performance
            performance_adjustment = (recent_performance - 0.5) * 0.1  # ±5% adjustment
            
            # Execution quality adjustment
            quality_adjustment = (execution_quality - 0.5) * 0.05  # ±2.5% adjustment
            
            # Total adjustment
            total_adjustment = performance_adjustment + quality_adjustment
            
            # Distribute adjustment between tiers
            # Better performance increases analyst weight (direction decisions)
            # Better execution quality increases tactician weight (execution decisions)
            analyst_adjustment = total_adjustment * 0.7  # 70% of adjustment affects analyst
            tactician_adjustment = total_adjustment * 0.3  # 30% of adjustment affects tactician
            
            return {
                "analyst": analyst_adjustment,
                "tactician": tactician_adjustment
            }

        except Exception as e:
            self.logger.error(f"Error calculating weight adjustment: {e}")
            return {"analyst": 0.0, "tactician": 0.0}

    def get_coordination_summary(self) -> dict[str, Any]:
        """Get summary of coordination performance and current state."""
        try:
            recent_performance = np.mean(self.combined_performance[-10:]) if self.combined_performance else 0.5
            
            return {
                "current_weights": {
                    "analyst_weight": self.analyst_weight,
                    "tactician_weight": self.tactician_weight
                },
                "recent_performance": recent_performance,
                "total_predictions": len(self.combined_performance),
                "history_size": len(self.coordination_history),
                "confidence_threshold": self.confidence_threshold
            }

        except Exception as e:
            self.logger.error(f"Error getting coordination summary: {e}")
            return {}