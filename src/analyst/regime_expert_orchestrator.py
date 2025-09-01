# src/analyst/regime_expert_orchestrator.py


import asyncio
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.analyst.predictive_ensembles.ensemble_orchestrator import (
    RegimePredictiveEnsembles,
)
from src.analyst.regime_runtime import get_current_regime_info
# TransitionRegimeHandler and TransitionAnalysis have been removed
# as they were part of the deprecated bull/bear/sideways market classification


class RegimeExpertOrchestrator:
    """
    Orchestrates regime detection and expert selection using composite_cluster_id.
    Integrates with Step 9.5 (HMM-LM Generalist) and Step 10 (Event Transition Modeling).
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("RegimeExpertOrchestrator")

        # Initialize regime ensembles
        self.regime_ensembles = RegimePredictiveEnsembles(config)

        # Configuration for cluster to regime mapping
        self.cluster_mapping = config.get(
            "regime_mapping",
            {
                # Rare/Transition Conditions (-1)
                -1: "RARE_MARKET_CONDITIONS",
                # Core Trend Regimes (0-4)
                0: "STRONG_BULL_TREND",
                1: "MODERATE_BULL_TREND",
                2: "WEAK_BULL_TREND",
                3: "STRONG_BEAR_TREND",
                4: "MODERATE_BEAR_TREND",
                # Sideways/Range Regimes (5-8)
                5: "TIGHT_SIDEWAYS_RANGE",
                6: "WIDE_SIDEWAYS_RANGE",
                7: "ASCENDING_SIDEWAYS",
                8: "DESCENDING_SIDEWAYS",
                # Volatility Regimes (9-12)
                9: "HIGH_VOLATILITY_BULL",
                10: "HIGH_VOLATILITY_BEAR",
                11: "LOW_VOLATILITY_RANGE",
                12: "EXTREME_VOLATILITY",
                # Transition Regimes (13-16)
                13: "BULL_TO_BEAR_TRANSITION",
                14: "BEAR_TO_BULL_TRANSITION",
                15: "TREND_TO_SIDEWAYS",
                16: "SIDEWAYS_TO_TREND",
                # Specialized Regimes (17-19)
                17: "ACCUMULATION_PHASE",
                18: "DISTRIBUTION_PHASE",
                19: "BREAKOUT_PREPARATION",
            },
        )

        # Confidence thresholds
        self.min_regime_confidence = config.get("min_regime_confidence", 0.6)
        self.min_expert_confidence = config.get("min_expert_confidence", 0.5)

        # Integration flags
        self.use_step9_5 = config.get("use_step9_5", True)
        self.use_step10 = config.get("use_step10", True)

        # Transition handler removed - using advanced HMM categorization instead

        # Cache for regime predictions
        self.regime_cache = {}
        self.last_regime_update = None
        self.cache_ttl = 300  # 5 minutes

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="regime expert initialization",
    )
    @handle_errors(
        exceptions=(Exception,), default_return=None, context="current regime detection"
    )
    @handle_errors(
        exceptions=(Exception,), default_return=None, context="regime expert prediction"
    )
    async def get_regime_expert_prediction(
        self, current_features: pd.DataFrame, regime_info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get prediction from the current regime expert."""
        try:
            cluster_id = regime_info.get("cluster_id")

            # Special handling for cluster -1 (transitions)
            if cluster_id == -1:
                return await self._handle_transition_prediction(
                    current_features, regime_info
                )

            expert = regime_info.get("expert")
            if expert is None:
                self.logger.warning("No expert available for current regime")
                return None

            # Get prediction from the expert
            prediction_output = expert.get_prediction(current_features)

            return {
                "prediction": prediction_output.get("prediction", "HOLD"),
                "confidence": prediction_output.get(
                    "confidence", regime_info.get("confidence", 0.0)
                ),
                "regime": regime_info.get("regime_name"),
                "cluster_id": cluster_id,
                "expert_type": type(expert).__name__,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error getting regime expert prediction: {e}")
            return None

    async def _handle_transition_prediction(
        self, features: pd.DataFrame, regime_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle predictions during market transitions (cluster -1)."""
        try:
            # Get current intensity scores for all regimes
            intensity_scores = self._get_current_intensity_scores(regime_info)

            # Analyze the transition
            analysis = self.transition_handler.analyze_transition(
                intensity_scores=intensity_scores, current_features=features
            )

            # Get trading recommendation
            recommendation = self.transition_handler.get_trading_recommendation(
                analysis
            )

            # Combine predictions from multiple regime experts if intensity threshold is met
            if analysis.intensity_threshold_met:
                combined_prediction = await self._get_combined_regime_predictions(
                    analysis, features
                )
            else:
                combined_prediction = {
                    "error": "Insufficient regime intensity for trading"
                }

            return {
                "prediction": recommendation.get("action", "HOLD"),
                "confidence": analysis.confidence_score,
                "regime": "RARE_MARKET_CONDITIONS",
                "cluster_id": -1,
                "expert_type": "TransitionHandler",
                "timestamp": datetime.now().isoformat(),
                "transition_analysis": analysis,
                "trading_recommendation": recommendation,
                "combined_prediction": combined_prediction,
                "regime_weights": analysis.regime_weights,
            }

        except Exception as e:
            self.logger.error(f"Error handling transition prediction: {e}")
            return {
                "prediction": "HOLD",
                "confidence": 0.0,
                "regime": "RARE_MARKET_CONDITIONS",
                "cluster_id": -1,
                "expert_type": "TransitionHandler",
                "timestamp": datetime.now().isoformat(),
                "error": f"Transition prediction failed: {e}",
            }

    async def _get_combined_regime_predictions(
        self, analysis: TransitionAnalysis, features: pd.DataFrame
    ) -> Dict[str, Any]:
        """Get weighted predictions from multiple regime experts."""
        combined_prediction = {
            "weighted_prediction": 0.0,
            "individual_predictions": {},
            "regime_contributions": {},
        }

        total_weight = 0.0

        for regime_name, weight in analysis.regime_weights.items():
            if weight < 0.1:  # Skip regimes with very low weight
                continue

            # Map regime name back to cluster ID
            cluster_id = self._get_cluster_id_from_regime_name(regime_name)
            if cluster_id is None:
                continue

            # Get prediction from this regime's expert
            expert = self.get_regime_expert(cluster_id)
            if expert is None:
                continue

            try:
                prediction = expert.get_prediction(features)
                prediction_value = prediction.get("prediction", 0.0)

                # Weight the prediction
                weighted_prediction = prediction_value * weight
                combined_prediction["weighted_prediction"] += weighted_prediction
                total_weight += weight

                # Store individual predictions
                combined_prediction["individual_predictions"][regime_name] = {
                    "prediction": prediction_value,
                    "weight": weight,
                    "weighted_contribution": weighted_prediction,
                }

                combined_prediction["regime_contributions"][regime_name] = weight

            except Exception as e:
                self.logger.warning(
                    f"Error getting prediction from {regime_name} expert: {e}"
                )

        # Normalize the weighted prediction
        if total_weight > 0:
            combined_prediction["weighted_prediction"] /= total_weight

        return combined_prediction

    @handle_errors(
        exceptions=(Exception,), default_return=None, context="step09_5 integration"
    )
    async def integrate_step9_5_prediction(
        self, regime_info: Dict[str, Any], step09_5_prediction: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Integrate Step 9.5 (HMM-LM Generalist) predictions with regime expert."""
        try:
            if not self.use_step9_5 or step09_5_prediction is None:
                return None

            # Extract Step 9.5 predictions
            regime_transition_prob = step09_5_prediction.get(
                "regime_transition_prob", 0.0
            )
            price_direction = step09_5_prediction.get("price_direction", "SIDEWAYS")
            tpsl_probabilities = step09_5_prediction.get("tpsl_probabilities", {})

            # Combine with current regime expert prediction
            current_prediction = await self.get_regime_expert_prediction(
                step09_5_prediction.get("current_features", pd.DataFrame()), regime_info
            )

            if current_prediction is None:
                return None

            # Weight the predictions based on confidence
            step09_5_confidence = step09_5_prediction.get("confidence", 0.0)
            expert_confidence = current_prediction.get("confidence", 0.0)

            # Combined confidence (weighted average)
            combined_confidence = step09_5_confidence * 0.4 + expert_confidence * 0.6

            return {
                "strategic_prediction": current_prediction,
                "regime_transition_prob": regime_transition_prob,
                "price_direction": price_direction,
                "tpsl_probabilities": tpsl_probabilities,
                "combined_confidence": combined_confidence,
                "should_trade": combined_confidence > self.min_regime_confidence,
                "integration_type": "step09_5",
            }

        except Exception as e:
            self.logger.error(f"Error integrating Step 9.5 prediction: {e}")
            return None

    @handle_errors(
        exceptions=(Exception,), default_return=None, context="step10 integration"
    )
    async def integrate_step10_prediction(
        self, regime_info: Dict[str, Any], step10_prediction: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Integrate Step 10 (Event Transition Modeling) predictions for timing optimization."""
        try:
            if not self.use_step10 or step10_prediction is None:
                return None

            # Extract Step 10 predictions
            path_class = step10_prediction.get("path_class", "end_of_trend")
            optimal_timing = step10_prediction.get("optimal_timing", 0)
            event_confidence = step10_prediction.get("confidence", 0.0)

            # Get current regime expert prediction
            current_prediction = await self.get_regime_expert_prediction(
                step10_prediction.get("current_features", pd.DataFrame()), regime_info
            )

            if current_prediction is None:
                return None

            # Determine if we should execute based on path class and confidence
            should_execute = (
                path_class in ["beginning_of_trend", "continuation"]
                and event_confidence > self.min_expert_confidence
                and current_prediction.get("confidence", 0.0)
                > self.min_expert_confidence
            )

            return {
                "strategic_prediction": current_prediction,
                "path_class": path_class,
                "optimal_timing": optimal_timing,
                "event_confidence": event_confidence,
                "should_execute": should_execute,
                "execution_delay": optimal_timing,  # bars to wait before executing
                "integration_type": "step10",
            }

        except Exception as e:
            self.logger.error(f"Error integrating Step 10 prediction: {e}")
            return None

    @handle_errors(
        exceptions=(Exception,), default_return=None, context="two-tier decision system"
    )
    def _make_final_decision(
        self,
        strategic_decision: Dict[str, Any],
        step09_5_integration: Optional[Dict[str, Any]],
        step10_integration: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Make final trading decision based on all available information."""

        base_prediction = strategic_decision.get("prediction", "HOLD")
        base_confidence = strategic_decision.get("confidence", 0.0)

        # Default decision
        final_decision = {
            "action": base_prediction,
            "confidence": base_confidence,
            "timing": "immediate",
            "reason": "strategic_only",
        }

        # Apply Step 9.5 adjustments (regime transitions)
        if step09_5_integration and step09_5_integration.get("should_trade", False):
            transition_prob = step09_5_integration.get("regime_transition_prob", 0.0)
            if transition_prob > 0.7:  # High probability of regime change
                final_decision["action"] = "HOLD"
                final_decision["reason"] = "regime_transition_imminent"
                final_decision["confidence"] = transition_prob

        # Apply Step 10 adjustments (timing optimization)
        if step10_integration and step10_integration.get("should_execute", False):
            optimal_timing = step10_integration.get("optimal_timing", 0)
            if optimal_timing > 0:
                final_decision["timing"] = f"delay_{optimal_timing}_bars"
                final_decision["reason"] = "optimal_timing"
                final_decision["confidence"] = min(
                    final_decision["confidence"],
                    step10_integration.get("event_confidence", 0.0),
                )

        return final_decision

    @handle_errors(
        exceptions=(Exception,), default_return=False, context="continuous monitoring"
    )
    async def start_continuous_monitoring(
        self, exchange: str, symbol: str, timeframe: str
    ) -> bool:
        """Start continuous monitoring for regime changes and trading opportunities."""
        try:
            self.logger.info(
                f"Starting continuous monitoring for {exchange}:{symbol} on {timeframe}"
            )

            while True:
                # Get current regime and decision
                decision = await self.get_two_tier_decision(exchange, symbol, timeframe)

                if decision is not None:
                    final_decision = decision.get("final_decision", {})

                    if final_decision.get("action") != "HOLD":
                        self.logger.info(f"Trading signal: {final_decision}")
                        # Here you would trigger the actual trading execution
                        # await self.execute_trade_decision(decision)

                # Wait before next check
                await asyncio.sleep(60)  # Check every minute

        except Exception as e:
            self.logger.error(f"Error in continuous monitoring: {e}")
            return False


# Convenience function for easy integration