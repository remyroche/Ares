# src/analyst/regime_expert_orchestrator.py

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.analyst.predictive_ensembles.ensemble_orchestrator import (
import logging
    RegimePredictiveEnsembles,
)
from src.analyst.regime_runtime import get_current_regime_info
# TransitionRegimeHandler and TransitionAnalysis have been removed
# as they were part of the deprecated bull/bear/sideways market classification


class RegimeExpertOrchestrator:
    """"
    Orchestrates regime detection and expert selection using composite_cluster_id.
    Integrates with Step 9.5 (HMM-LM Generalist) and Step 10 (Event Transition Modeling).
    """"

    def __init__(self, config: dict[str, Any]):
        self.logger = logging.getLogger(self.__class__.__name__)
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
    async def initialize(self) -> bool:
        """Initialize the regime expert orchestrator."""
        try:
            self.logger.info("Initializing Regime Expert Orchestrator...")

            # Load regime ensembles
            # Note: This would typically load the trained models from Step 5
            self.logger.info("Regime Expert Orchestrator initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Regime Expert Orchestrator: {e}")
            return False

    def get_current_regime_from_cluster(self, cluster_id: int) -> str:
        """Map composite_cluster_id to regime name."""
        return self.cluster_mapping.get(cluster_id, "UNKNOWN")

    def get_regime_expert(self, cluster_id: int) -> Optional[Any]:
        """Get the appropriate regime expert for the given cluster ID."""
        regime_name = self.get_current_regime_from_cluster(cluster_id)
        return self.regime_ensembles.get_regime_expert(cluster_id)

    @handle_errors(
        exceptions=(Exception,), default_return=None, context="current regime detection"
    )
    async def get_current_regime_info(
        self, exchange: str, symbol: str, timeframe: str
    ) -> Optional[Dict[str, Any]]:
        """Get comprehensive current regime information."""
        try:
            # Get regime info from runtime
            regime_info = get_current_regime_info(exchange, symbol, timeframe)

            if regime_info is None or regime_info.get("cluster_id", -1) == -1:
                return None

            cluster_id = regime_info["cluster_id"]
            regime_name = self.get_current_regime_from_cluster(cluster_id)
            expert = self.get_regime_expert(cluster_id)

            # Get intensity confidence
            intensities = regime_info.get("intensities", {})
            confidence = intensities.get(cluster_id, 0.0)

            return {
                "cluster_id": cluster_id,
                "regime_name": regime_name,
                "expert": expert,
                "confidence": confidence,
                "intensities": intensities,
                "p_emerge": regime_info.get("p_emerge", {}),
                "exit_hazard": regime_info.get("exit_hazard"),
                "timestamp": regime_info.get("timestamp"),
                "exchange": exchange,
                "symbol": symbol,
                "timeframe": timeframe,
            }

        except Exception as e:
            self.logger.error(f"Error getting current regime info: {e}")
            return None

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

    def _get_current_intensity_scores(
        self, regime_info: Dict[str, Any]
    ) -> Dict[str, float]:
        """Get current intensity scores for all regimes."""
        intensities = regime_info.get("intensities", {})
        return {
            f"intensity_cluster_{cluster_id}": intensity
            for cluster_id, intensity in intensities.items()
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

            # Get prediction from this regime's expert'
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

    def _get_cluster_id_from_regime_name(self, regime_name: str) -> Optional[int]:
        """Map regime name back to cluster ID."""
        for cluster_id, name in self.cluster_mapping.items():
            if name == regime_name:
                return cluster_id
        return None

    @handle_errors(
        exceptions=(Exception,), default_return=None, context="step9_5 integration"
    )
    async def integrate_step9_5_prediction(
        self, regime_info: Dict[str, Any], step9_5_prediction: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Integrate Step 9.5 (HMM-LM Generalist) predictions with regime expert."""
        try:
            if not self.use_step9_5 or step9_5_prediction is None:
                return None

            # Extract Step 9.5 predictions
            regime_transition_prob = step9_5_prediction.get(
                "regime_transition_prob", 0.0
            )
            price_direction = step9_5_prediction.get("price_direction", "SIDEWAYS")
            tpsl_probabilities = step9_5_prediction.get("tpsl_probabilities", {})

            # Combine with current regime expert prediction
            current_prediction = await self.get_regime_expert_prediction(
                step9_5_prediction.get("current_features", pd.DataFrame()), regime_info
            )

            if current_prediction is None:
                return None

            # Weight the predictions based on confidence
            step9_5_confidence = step9_5_prediction.get("confidence", 0.0)
            expert_confidence = current_prediction.get("confidence", 0.0)

            # Combined confidence (weighted average)
            combined_confidence = step9_5_confidence * 0.4 + expert_confidence * 0.6

            return {
                "strategic_prediction": current_prediction,
                "regime_transition_prob": regime_transition_prob,
                "price_direction": price_direction,
                "tpsl_probabilities": tpsl_probabilities,
                "combined_confidence": combined_confidence,
                "should_trade": combined_confidence > self.min_regime_confidence,
                "integration_type": "step9_5",
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
    async def get_two_tier_decision(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        step9_5_prediction: Optional[Dict[str, Any]] = None,
        step10_prediction: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get two-tier decision combining regime expert with Step 9.5 and Step 10."""
        try:
            # Tier 1: Get current regime and expert
            regime_info = await self.get_current_regime_info(
                exchange, symbol, timeframe
            )
            if regime_info is None:
                self.logger.warning("Could not determine current regime")
                return None

            # Tier 1: Strategic decision from regime expert
            strategic_decision = await self.get_regime_expert_prediction(
                pd.DataFrame(),  # Current features would be passed here
                regime_info,
            )

            if strategic_decision is None:
                return None

            # Check if we should consider trading
            if strategic_decision.get("confidence", 0.0) < self.min_regime_confidence:
                return {
                    "decision": "HOLD",
                    "reason": "Insufficient regime confidence",
                    "regime_info": regime_info,
                    "strategic_decision": strategic_decision,
                    "tactical_decision": None,
                    "final_decision": "HOLD",
                }

            # Tier 2: Integrate Step 9.5 (regime transitions)
            step9_5_integration = None
            if step9_5_prediction is not None:
                step9_5_integration = await self.integrate_step9_5_prediction(
                    regime_info, step9_5_prediction
                )

            # Tier 2: Integrate Step 10 (event timing)
            step10_integration = None
            if step10_prediction is not None:
                step10_integration = await self.integrate_step10_prediction(
                    regime_info, step10_prediction
                )

            # Make final decision
            final_decision = self._make_final_decision(
                strategic_decision, step9_5_integration, step10_integration
            )

            return {
                "regime_info": regime_info,
                "strategic_decision": strategic_decision,
                "step9_5_integration": step9_5_integration,
                "step10_integration": step10_integration,
                "final_decision": final_decision,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error getting two-tier decision: {e}")
            return None

    def _make_final_decision(
        self,
        strategic_decision: Dict[str, Any],
        step9_5_integration: Optional[Dict[str, Any]],
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
        if step9_5_integration and step9_5_integration.get("should_trade", False):
            transition_prob = step9_5_integration.get("regime_transition_prob", 0.0)
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
async def get_regime_expert_decision(
    exchange: str, symbol: str, timeframe: str, config: dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Get regime expert decision for the given parameters."""
    orchestrator = RegimeExpertOrchestrator(config)
    await orchestrator.initialize()

    return await orchestrator.get_two_tier_decision(exchange, symbol, timeframe)
