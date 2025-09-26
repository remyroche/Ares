"""Light-weight timeframe optimisation helpers.

The previous module depended on many unavailable research components and hid
failures by returning low scores.  The refactored version implements a small
heuristic optimiser that relies only on pandas and numpy, performs strict input
validation and raises :class:`OptimizationError` when optimisation cannot be
completed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("AutomaticTimeframeOptimizer")


class OptimizationError(RuntimeError):
    """Raised when timeframe optimisation fails."""


class ModelType(str, Enum):
    ANALYST = "analyst"
    TACTICIAN = "tactician"


@dataclass(frozen=True)
class OptimalTimeframeConfig:
    """Minimal configuration returned by the optimiser."""

    time_horizons: Dict[str, int] = field(default_factory=dict)
    profit_targets: Dict[str, float] = field(default_factory=dict)
    transaction_cost: float = 0.0008


@dataclass(frozen=True)
class ModelOptimizationParameters:
    """Parameter bundle controlling the heuristic search space."""

    short_horizons: Tuple[int, ...]
    medium_horizons: Tuple[int, ...]
    profit_targets: Dict[str, float]
    liquidity_window: int = 5

    def __post_init__(self) -> None:
        if not self.short_horizons or not self.medium_horizons:
            raise ValueError("Horizon candidates must be non-empty")
        if len(self.short_horizons) != len(self.medium_horizons):
            raise ValueError("short and medium horizons must have the same length")
        for horizon in (*self.short_horizons, *self.medium_horizons):
            if horizon <= 0:
                raise ValueError("Horizon lengths must be positive integers")
        if not self.profit_targets:
            raise ValueError("At least one profit target must be configured")
        for name, value in self.profit_targets.items():
            if value <= 0:
                raise ValueError(f"Profit target '{name}' must be positive")
        if self.liquidity_window <= 0:
            raise ValueError("liquidity_window must be greater than zero")


@dataclass
class OptimizationResult:
    model_type: ModelType
    optimal_config: OptimalTimeframeConfig
    optimization_score: float
    validation_score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_type": self.model_type.value,
            "optimization_score": self.optimization_score,
            "validation_score": self.validation_score,
            "timestamp": self.timestamp.isoformat(),
            "optimal_config": {
                "time_horizons": self.optimal_config.time_horizons,
                "profit_targets": self.optimal_config.profit_targets,
                "transaction_cost": self.optimal_config.transaction_cost,
            },
        }


DEFAULT_PARAMETERS: Dict[ModelType, ModelOptimizationParameters] = {
    ModelType.ANALYST: ModelOptimizationParameters(
        short_horizons=(2, 3, 4),
        medium_horizons=(6, 8, 10),
        profit_targets={"micro": 0.003, "small": 0.006, "medium": 0.009},
    ),
    ModelType.TACTICIAN: ModelOptimizationParameters(
        short_horizons=(1, 2, 3),
        medium_horizons=(4, 6, 8),
        profit_targets={"micro": 0.004, "small": 0.007, "medium": 0.010},
    ),
}


class AutomaticTimeframeOptimizer:
    """Simple heuristic based timeframe optimiser."""

    def __init__(
        self,
        parameters: Optional[Mapping[ModelType | str, ModelOptimizationParameters]] = None,
    ) -> None:
        self.optimization_enabled = True
        self._cache: Dict[ModelType, OptimizationResult] = {}
        self._parameters: Dict[ModelType, ModelOptimizationParameters] = dict(DEFAULT_PARAMETERS)
        if parameters:
            for key, value in parameters.items():
                model_type = _normalise_model_type(key)
                self._parameters[model_type] = value

    def optimize_for_model(
        self,
        model_type: ModelType,
        market_data: pd.DataFrame,
        force_optimization: bool = False,
    ) -> OptimizationResult:
        if not isinstance(market_data, pd.DataFrame):
            raise OptimizationError("market_data must be a pandas DataFrame")
        if market_data.empty:
            raise OptimizationError("market_data must contain rows")
        if not {"close", "volume"}.issubset(market_data.columns):
            raise OptimizationError("market_data must contain 'close' and 'volume' columns")

        if not self.optimization_enabled and not force_optimization:
            raise OptimizationError("Timeframe optimisation has been disabled")

        if not force_optimization and model_type in self._cache:
            return self._cache[model_type]

        params = self._parameters[model_type]
        close = pd.to_numeric(market_data["close"], errors="coerce")
        if close.isna().all():
            raise OptimizationError("close column does not contain numeric values")
        volume = pd.to_numeric(market_data["volume"], errors="coerce")
        if volume.isna().all():
            raise OptimizationError("volume column does not contain numeric values")

        scores = self._score_candidates(close.astype(float), params)

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if not np.isfinite(best_score):  # pragma: no cover - guard for extreme input
            raise OptimizationError("Failed to derive a valid optimisation score")

        optimal_config = OptimalTimeframeConfig(
            time_horizons={
                "short": params.short_horizons[best_idx],
                "medium": params.medium_horizons[best_idx],
            },
            profit_targets=dict(params.profit_targets),
        )
        validation_score = self._validate_candidate(volume.astype(float), params, optimal_config)

        result = OptimizationResult(
            model_type=model_type,
            optimal_config=optimal_config,
            optimization_score=best_score,
            validation_score=validation_score,
        )
        self._cache[model_type] = result
        logger.info(
            "Optimised timeframe for %s – score %.4f validation %.4f",
            model_type.value,
            best_score,
            validation_score,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def clear_cache(self) -> None:
        """Forget previously computed optimisation results."""

        self._cache.clear()

    def _score_candidates(self, close: pd.Series, params: ModelOptimizationParameters) -> np.ndarray:
        returns = close.pct_change().dropna()
        if returns.empty:
            raise OptimizationError("Not enough data to calculate returns")

        scores: List[float] = []
        for short, medium in zip(params.short_horizons, params.medium_horizons):
            short_vol = returns.rolling(short).std().dropna()
            medium_vol = returns.rolling(medium).std().dropna()
            if short_vol.empty or medium_vol.empty:
                scores.append(float("nan"))
                continue
            combined = short_vol.iloc[-1] * 0.6 + medium_vol.iloc[-1] * 0.4
            scores.append(float(combined))
        arr = np.array(scores, dtype=float)
        if np.all(np.isnan(arr)):
            raise OptimizationError("Could not compute optimisation scores for any candidate")
        return np.nan_to_num(arr, nan=0.0)

    def _validate_candidate(
        self,
        volume: pd.Series,
        params: ModelOptimizationParameters,
        config: OptimalTimeframeConfig,
    ) -> float:
        """Simple sanity check that penalises extreme spreads between horizons."""

        short = config.time_horizons.get("short", 0)
        medium = config.time_horizons.get("medium", 0)
        if medium <= short:
            raise OptimizationError("medium horizon must be greater than short horizon")
        spread_penalty = max(0.0, (medium - short) / max(short, 1) - 1.5)
        rolling_liquidity = volume.rolling(params.liquidity_window).mean().dropna()
        if rolling_liquidity.empty:
            raise OptimizationError("Not enough data to evaluate liquidity")
        liquidity = float(rolling_liquidity.iloc[-1])
        if not np.isfinite(liquidity) or liquidity <= 0:
            raise OptimizationError("volume based liquidity check failed")
        volume_mean = float(volume.mean())
        if not np.isfinite(volume_mean) or volume_mean <= 0:
            raise OptimizationError("volume mean must be positive")
        liquidity_score = min(1.0, liquidity / (volume_mean + 1e-9))
        return max(0.0, 1.0 - spread_penalty) * liquidity_score


def optimize_timeframes_for_training(
    market_data: pd.DataFrame,
    model_types: Iterable[ModelType | str] | ModelType | str = (ModelType.ANALYST, ModelType.TACTICIAN),
) -> Dict[ModelType, OptimizationResult]:
    optimizer = AutomaticTimeframeOptimizer()
    normalised = _normalise_model_types(model_types)
    return {model_type: optimizer.optimize_for_model(model_type, market_data) for model_type in normalised}


def get_optimal_timeframes_for_models(
    results: Dict[ModelType | str, OptimizationResult]
) -> Dict[str, Dict[str, object]]:
    return {(_normalise_model_type(key)).value: result.to_dict() for key, result in results.items()}


def _normalise_model_types(
    model_types: Iterable[ModelType | str] | ModelType | str,
) -> List[ModelType]:
    if isinstance(model_types, (ModelType, str)):
        return [_normalise_model_type(model_types)]
    return [_normalise_model_type(model_type) for model_type in model_types]


def _normalise_model_type(model_type: ModelType | str) -> ModelType:
    if isinstance(model_type, ModelType):
        return model_type
    try:
        return ModelType(model_type.lower())
    except ValueError as exc:  # pragma: no cover - guard clause
        raise OptimizationError(f"Unknown model type '{model_type}'") from exc


__all__ = [
    "AutomaticTimeframeOptimizer",
    "ModelType",
    "ModelOptimizationParameters",
    "OptimizationError",
    "OptimizationResult",
    "OptimalTimeframeConfig",
    "get_optimal_timeframes_for_models",
    "optimize_timeframes_for_training",
]
