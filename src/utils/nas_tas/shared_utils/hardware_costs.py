"""Hardware-aware evaluation helpers for NAS/TAS architectures."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HardwareConstraintConfig:
    latency_budget_ms: float = 5.0
    tail_latency_budget_ms: float = 10.0
    cold_start_budget_ms: float = 60.0
    memory_budget_mb: float = 4096.0


class HardwareCostEvaluator:
    """Estimate latency/memory costs and enforce production constraints."""

    def __init__(self, config: HardwareConstraintConfig) -> None:
        self.config = config

    def estimate(self, params: Dict[str, Any], metrics: Dict[str, float]) -> Dict[str, float]:
        parameter_count = self._resolve_numeric(
            metrics.get("parameter_count"),
            metrics.get("n_parameters"),
            params.get("parameter_count"),
        )
        depth = self._resolve_numeric(params.get("depth"), params.get("num_layers"), 1)
        width = self._resolve_numeric(params.get("width"), params.get("hidden_units"), 32)

        latency_ms = float(metrics.get("latency_ms") or self._estimate_latency(parameter_count, depth, width))
        tail_latency_ms = float(
            metrics.get("tail_latency_ms")
            or latency_ms * (1.0 + 0.2 * np.log1p(depth))
        )
        cold_start_ms = float(
            metrics.get("cold_start_ms")
            or latency_ms + 5.0 * np.sqrt(parameter_count or 1.0)
        )
        memory_mb = float(
            metrics.get("memory_mb")
            or (parameter_count or 0.0) * 4.0 / (1024 ** 2)
        )

        return {
            "latency_ms": latency_ms,
            "tail_latency_ms": tail_latency_ms,
            "cold_start_ms": cold_start_ms,
            "memory_mb": memory_mb,
        }

    def validate(self, costs: Dict[str, float]) -> bool:
        return (
            costs.get("latency_ms", float("inf")) <= self.config.latency_budget_ms
            and costs.get("tail_latency_ms", float("inf")) <= self.config.tail_latency_budget_ms
            and costs.get("cold_start_ms", float("inf")) <= self.config.cold_start_budget_ms
            and costs.get("memory_mb", float("inf")) <= self.config.memory_budget_mb
        )

    def constraint_penalty(self, costs: Dict[str, float]) -> float:
        penalties = []
        penalties.append(self._penalty(costs.get("latency_ms"), self.config.latency_budget_ms))
        penalties.append(self._penalty(costs.get("tail_latency_ms"), self.config.tail_latency_budget_ms))
        penalties.append(self._penalty(costs.get("cold_start_ms"), self.config.cold_start_budget_ms))
        penalties.append(self._penalty(costs.get("memory_mb"), self.config.memory_budget_mb))
        return float(sum(penalties))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_numeric(self, *values: Any) -> float:
        for value in values:
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return 0.0

    def _estimate_latency(self, parameter_count: float, depth: float, width: float) -> float:
        parameter_count = max(parameter_count, 1.0)
        depth = max(depth, 1.0)
        width = max(width, 1.0)
        base_latency = parameter_count ** 0.5 / 1000.0
        depth_factor = np.log1p(depth)
        width_factor = np.log1p(width)
        return float(base_latency * depth_factor * width_factor * 1000.0)

    def _penalty(self, value: Any, budget: float) -> float:
        if value is None:
            return 0.0
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        if value <= budget:
            return 0.0
        return (value - budget) / max(budget, 1e-8)


__all__ = [
    "HardwareCostEvaluator",
    "HardwareConstraintConfig",
]
