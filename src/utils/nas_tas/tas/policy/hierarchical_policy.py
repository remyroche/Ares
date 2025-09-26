"""Hierarchical tactical policy graph for TAS execution."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StrategyDecision:
    target_exposure: float
    rebalance_interval_minutes: int
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionDirective:
    slice_count: int
    participation_rate: float
    venue_preferences: List[str] = field(default_factory=list)
    urgency: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HierarchicalPolicyConfig:
    exposure_bands: Dict[str, float] = field(
        default_factory=lambda: {
            "conservative": 0.2,
            "balanced": 0.4,
            "aggressive": 0.6,
        }
    )
    tactical_rebalance_minutes: int = 30
    microstructure_learning_rate: float = 0.1
    default_slice_count: int = 4
    default_participation_rate: float = 0.15
    venue_universe: List[str] = field(default_factory=lambda: ["ARCA", "EDGX", "IEX", "NYSE"])


class HierarchicalPolicyGraph:
    """Encapsulates strategy vs execution separation for TAS."""

    def __init__(self, config: Optional[HierarchicalPolicyConfig] = None) -> None:
        self.config = config or HierarchicalPolicyConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def plan_strategy(self, regime_info: Optional[Dict[str, Any]] = None) -> StrategyDecision:
        """Map regime & market descriptors to a tactical exposure band."""

        regime_info = regime_info or {}
        risk_score = float(regime_info.get("risk_score", 0.5))
        volatility = float(regime_info.get("volatility", 0.02))
        liquidity = float(regime_info.get("liquidity", 1.0))

        if risk_score < 0.3 or volatility > 0.04:
            band_key = "conservative"
        elif risk_score < 0.6:
            band_key = "balanced"
        else:
            band_key = "aggressive"

        target_exposure = self.config.exposure_bands.get(band_key, 0.3) * liquidity
        confidence = max(0.1, min(0.95, regime_info.get("confidence", 0.6)))

        decision = StrategyDecision(
            target_exposure=target_exposure,
            rebalance_interval_minutes=self.config.tactical_rebalance_minutes,
            confidence=confidence,
            metadata={
                "band": band_key,
                "risk_score": risk_score,
                "volatility": volatility,
                "liquidity": liquidity,
            },
        )
        self.logger.debug("Strategy decision: %s", decision)
        return decision

    def plan_execution(
        self,
        strategy: StrategyDecision,
        microstructure: Optional[Dict[str, Any]] = None,
    ) -> ExecutionDirective:
        """Generate microstructure-aware execution guidance."""

        microstructure = microstructure or {}
        spread = float(microstructure.get("spread", 0.01))
        queue_position = float(microstructure.get("queue_position", 0.5))
        imbalance = float(microstructure.get("volume_imbalance", 0.0))

        slice_count = max(1, int(self.config.default_slice_count * (1 + spread / 0.02)))
        participation_rate = min(0.5, self.config.default_participation_rate * (1 + abs(imbalance)))
        urgency = max(0.1, min(0.9, 1 - queue_position))

        # Simple learning-to-rank style venue ordering using spread/imbalance features
        venue_scores = {}
        for venue in self.config.venue_universe:
            venue_quality = float(microstructure.get("venue_quality", {}).get(venue, 0.5))
            venue_scores[venue] = venue_quality - self.config.microstructure_learning_rate * spread
        ranked_venues = [venue for venue, _ in sorted(venue_scores.items(), key=lambda item: item[1], reverse=True)]

        directive = ExecutionDirective(
            slice_count=slice_count,
            participation_rate=participation_rate,
            urgency=urgency,
            venue_preferences=ranked_venues,
            metadata={
                "spread": spread,
                "queue_position": queue_position,
                "volume_imbalance": imbalance,
            },
        )
        self.logger.debug("Execution directive: %s", directive)
        return directive

    def build_plan(
        self,
        regime_info: Optional[Dict[str, Any]] = None,
        microstructure: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        strategy = self.plan_strategy(regime_info)
        execution = self.plan_execution(strategy, microstructure)
        return {
            "strategy": strategy,
            "execution": execution,
        }


__all__ = [
    "HierarchicalPolicyGraph",
    "HierarchicalPolicyConfig",
    "StrategyDecision",
    "ExecutionDirective",
]
