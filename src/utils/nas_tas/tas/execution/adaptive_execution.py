"""Adaptive execution layer that reacts to microstructure signals."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExecutionObservation:
    spread: float
    queue_position: float
    volume_imbalance: float
    volatility: float
    venue_quality: Dict[str, float] = field(default_factory=dict)


@dataclass
class AdaptiveExecutionConfig:
    almgren_chrisk_lambda: float = 1e-6
    min_slices: int = 2
    max_slices: int = 12
    participation_floor: float = 0.05
    participation_ceiling: float = 0.5
    learning_rate: float = 0.1


@dataclass
class OrderSchedule:
    slices: List[float]
    expected_shortfall: float
    venue_weights: Dict[str, float]


class AdaptiveExecutionLayer:
    """Plan order slicing & routing using microstructure awareness."""

    def __init__(self, config: Optional[AdaptiveExecutionConfig] = None) -> None:
        self.config = config or AdaptiveExecutionConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_schedule(
        self,
        notional: float,
        observation: ExecutionObservation,
        directive_urgency: float,
        venue_preferences: Optional[List[str]] = None,
    ) -> OrderSchedule:
        slice_count = self._determine_slice_count(observation, directive_urgency)
        participation = self._determine_participation(observation)
        slices = self._almgren_chriss_schedule(notional, slice_count, observation.volatility)

        venue_weights = self._rank_venues(observation, venue_preferences)
        shortfall = self._estimate_shortfall(slices, observation, participation)

        schedule = OrderSchedule(slices=slices, expected_shortfall=shortfall, venue_weights=venue_weights)
        self.logger.debug("Adaptive schedule generated: %s", schedule)
        return schedule

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _determine_slice_count(self, observation: ExecutionObservation, urgency: float) -> int:
        spread_factor = 1 + observation.spread / 0.02
        urgency_factor = 1 + urgency
        raw_count = self.config.min_slices * spread_factor * urgency_factor
        return int(np.clip(raw_count, self.config.min_slices, self.config.max_slices))

    def _determine_participation(self, observation: ExecutionObservation) -> float:
        imbalance_adjustment = 1 + abs(observation.volume_imbalance)
        participation = self.config.participation_floor * imbalance_adjustment
        return float(np.clip(participation, self.config.participation_floor, self.config.participation_ceiling))

    def _almgren_chriss_schedule(self, notional: float, slices: int, volatility: float) -> List[float]:
        time_horizon = slices
        gamma = self.config.almgren_chrisk_lambda
        dt = 1 / max(time_horizon, 1)
        optimal_rate = (notional / time_horizon) * np.exp(-gamma * np.arange(time_horizon) * dt)
        return optimal_rate.tolist()

    def _rank_venues(
        self,
        observation: ExecutionObservation,
        venue_preferences: Optional[List[str]],
    ) -> Dict[str, float]:
        venues = venue_preferences or list(observation.venue_quality.keys())
        if not venues:
            return {}
        scores = {}
        for venue in venues:
            quality = observation.venue_quality.get(venue, 0.5)
            scores[venue] = quality - self.config.learning_rate * observation.spread
        total = sum(scores.values()) or 1.0
        return {venue: score / total for venue, score in scores.items()}

    def _estimate_shortfall(
        self,
        slices: List[float],
        observation: ExecutionObservation,
        participation: float,
    ) -> float:
        impact = observation.spread * participation
        variance_term = observation.volatility * np.sqrt(len(slices))
        return float(impact + variance_term)


__all__ = [
    "AdaptiveExecutionLayer",
    "AdaptiveExecutionConfig",
    "ExecutionObservation",
    "OrderSchedule",
]
