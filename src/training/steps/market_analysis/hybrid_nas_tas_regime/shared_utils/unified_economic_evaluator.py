"""Unified Economic Evaluator for NAS and TAS Systems.

This module provides lightweight yet comprehensive economic evaluation helpers
for the hybrid NAS/TAS stack.  The original implementation that shipped with
this repository was partially truncated which caused imports to fail and left
callers without the convenience helpers they expect (for instance
``quick_economic_evaluation`` and regime-level scoring utilities).  The module
below rebuilds the public API in a compact form that focuses on deterministic
metrics, graceful fallbacks, and compatibility with the wider codebase.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EvaluationType(Enum):
    """Supported evaluation flavours."""

    ECONOMIC_SIGNIFICANCE = "economic_significance"
    TRADING_VIABILITY = "trading_viability"
    RISK_ADJUSTED = "risk_adjusted"
    COMBINED = "combined"


class ArchitectureType(Enum):
    """Architecture flavours used by NAS/TAS components."""

    NEURAL = "neural"
    TREE = "tree"
    HYBRID = "hybrid"


@dataclass
class EconomicEvaluationConfig:
    """Configuration container used across NAS/TAS utilities."""

    evaluation_types: List[EvaluationType] = field(
        default_factory=lambda: [
            EvaluationType.ECONOMIC_SIGNIFICANCE,
            EvaluationType.TRADING_VIABILITY,
        ]
    )
    architecture_type: ArchitectureType = ArchitectureType.HYBRID
    significance_threshold: float = 0.05
    risk_free_rate: float = 0.02
    min_regime_duration: int = 10
    price_impact_threshold: float = 0.5
    volume_threshold: float = 0.5
    volatility_threshold: float = 0.5
    trend_threshold: float = 0.5
    efficiency_threshold: float = 0.5
    return_threshold: float = 0.0
    transaction_cost_bps: float = 1.0
    slippage_bps: float = 0.5
    position_size_threshold: float = 0.01
    max_position_size: float = 0.1
    enable_economic_indicators: bool = True
    enable_position_aware_analysis: bool = True
    enable_bootstrap_analysis: bool = False
    enable_regime_specific_analysis: bool = True
    enable_logging: bool = True
    confidence_interval: float = 0.95
    extra_parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation of the configuration."""

        data = dict(self.__dict__)
        data["evaluation_types"] = [item.value for item in self.evaluation_types]
        data["architecture_type"] = self.architecture_type.value
        return data


@dataclass
class EconomicMetrics:
    """Rich set of metrics returned by the economic evaluator."""

    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    economic_significance: float
    trading_viability: float
    overall_score: float
    return_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    trading_frequency: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0


@dataclass
class EconomicSignificanceResult:
    """Compact summary used by validators and examples."""

    overall_score: float
    significance_level: str
    component_scores: Dict[str, float]
    regime_scores: Optional[np.ndarray] = None
    evaluation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class EconomicEvaluationResult:
    """Full evaluation output consumed by orchestration layers."""

    overall_score: float
    significance_level: str
    economic_metrics: EconomicMetrics
    evaluation_summary: Dict[str, Any]
    architecture_type: ArchitectureType
    total_evaluation_time: float
    regime_scores: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable dictionary."""

        return {
            "overall_score": self.overall_score,
            "significance_level": self.significance_level,
            "economic_metrics": dict(self.economic_metrics.__dict__),
            "evaluation_summary": self.evaluation_summary,
            "architecture_type": self.architecture_type.value,
            "total_evaluation_time": self.total_evaluation_time,
            "regime_scores": self.regime_scores.tolist()
            if isinstance(self.regime_scores, np.ndarray)
            else self.regime_scores,
            "metadata": self.metadata,
            "success": self.success,
            "error_message": self.error_message,
        }


class UnifiedEconomicEvaluator:
    """Primary entry point used by NAS/TAS components."""

    def __init__(self, config: Optional[EconomicEvaluationConfig] = None) -> None:
        self.config = config or EconomicEvaluationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        if self.config.enable_logging:
            self.logger.info(
                "🚀 UnifiedEconomicEvaluator initialised",
                extra={"evaluation_types": [e.value for e in self.config.evaluation_types]},
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(
        self,
        market_data: Union[pd.DataFrame, np.ndarray, Sequence[Sequence[float]]],
        regime_predictions: Optional[Union[np.ndarray, Sequence[int]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> EconomicEvaluationResult:
        """Evaluate economic significance and trading viability.

        The method purposefully accepts a flexible signature because legacy
        callers pass positional arguments in different orders.  Extra positional
        arguments are interpreted as ``regime_probabilities`` followed by
        ``timestamps`` and ``returns`` if they are not provided explicitly via
        keyword arguments.
        """

        regime_probabilities = kwargs.pop("regime_probabilities", None)
        timestamps = kwargs.pop("timestamps", None)
        returns = kwargs.pop("returns", None)
        predictions = kwargs.pop("predictions", None)
        architecture_type = kwargs.pop("architecture_type", None)
        model_metadata = kwargs.pop("model_metadata", None)

        extra_args = list(args)
        # First optional positional argument → probabilities (unless it looks like timestamps)
        if regime_probabilities is None and extra_args:
            candidate = extra_args[0]
            if self._looks_like_timestamps(candidate):
                timestamps = extra_args.pop(0)
            else:
                regime_probabilities = extra_args.pop(0)
        # Second optional positional argument → timestamps
        if timestamps is None and extra_args:
            candidate = extra_args[0]
            if self._looks_like_timestamps(candidate):
                timestamps = extra_args.pop(0)
            else:
                returns = returns or extra_args.pop(0)
        # Third optional positional argument → returns
        if returns is None and extra_args:
            returns = extra_args.pop(0)

        start_time = time.time()
        df = self._normalise_market_data(market_data, timestamps)
        returns_array = self._prepare_returns(df, returns)
        regime_predictions = self._prepare_regime_predictions(regime_predictions, len(df))
        regime_probabilities = self._prepare_regime_probabilities(
            regime_probabilities, len(df), len(np.unique(regime_predictions))
        )

        metrics = self._calculate_metrics(
            returns_array, regime_predictions, regime_probabilities, predictions
        )
        significance_level = self._score_to_level(metrics.overall_score)
        regime_scores = self._calculate_regime_scores(
            regime_predictions, metrics.economic_significance
        )

        evaluation_summary = {
            "mean_return": float(np.mean(returns_array)),
            "annualised_return": float(np.mean(returns_array) * 252),
            "risk_free_rate": self.config.risk_free_rate,
            "evaluation_types": [etype.value for etype in self.config.evaluation_types],
            "significance_threshold": self.config.significance_threshold,
            "trading_viability_threshold": self.config.trend_threshold,
        }

        architecture = self._resolve_architecture_type(architecture_type)
        metadata = {
            "n_samples": len(df),
            "n_regimes": int(len(regime_scores)) if regime_scores is not None else 0,
            "regime_scores": regime_scores,
            "evaluation_config": self.config.to_dict(),
            "model_metadata": model_metadata or {},
            "additional_arguments": {
                "predictions_provided": predictions is not None,
                "extra_args_count": len(extra_args),
                **kwargs,
            },
        }

        execution_time = time.time() - start_time
        result = EconomicEvaluationResult(
            overall_score=metrics.overall_score,
            significance_level=significance_level,
            economic_metrics=metrics,
            evaluation_summary=evaluation_summary,
            architecture_type=architecture,
            total_evaluation_time=execution_time,
            regime_scores=regime_scores,
            metadata=metadata,
            success=True,
            error_message=None,
        )

        if self.config.enable_logging:
            self.logger.info(
                "✅ Unified economic evaluation completed",
                extra={
                    "overall_score": metrics.overall_score,
                    "significance_level": significance_level,
                    "execution_time": execution_time,
                },
            )

        return result

    def evaluate_regimes(
        self,
        market_data: Union[pd.DataFrame, np.ndarray, Sequence[Sequence[float]]],
        regime_predictions: Union[np.ndarray, Sequence[int]],
        regime_probabilities: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Return regime-level economic significance scores."""

        result = self.evaluate(
            market_data,
            regime_predictions,
            regime_probabilities,
            **kwargs,
        )
        if isinstance(result.regime_scores, np.ndarray):
            return result.regime_scores
        if isinstance(result.regime_scores, dict):
            ordered = [result.regime_scores.get(idx, 0.0) for idx in sorted(result.regime_scores)]
            return np.asarray(ordered, dtype=float)
        return np.full(len(np.unique(regime_predictions)), result.overall_score)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalise_market_data(
        self,
        market_data: Union[pd.DataFrame, np.ndarray, Sequence[Sequence[float]]],
        timestamps: Optional[Union[Sequence[Any], np.ndarray]],
    ) -> pd.DataFrame:
        if isinstance(market_data, pd.DataFrame):
            df = market_data.copy()
        else:
            array = np.asarray(market_data)
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            columns = ["open", "high", "low", "close", "volume"]
            column_count = min(array.shape[1], len(columns))
            df = pd.DataFrame(array[:, :column_count], columns=columns[:column_count])
        if timestamps is not None and "timestamp" not in df.columns:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(np.asarray(timestamps))
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.date_range(
                end=pd.Timestamp.utcnow(), periods=len(df), freq="min"
            )
        df = df.sort_values("timestamp").reset_index(drop=True)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(method="ffill").fillna(method="bfill")
        return df

    def _prepare_returns(
        self,
        df: pd.DataFrame,
        returns: Optional[Union[np.ndarray, Sequence[float]]],
    ) -> np.ndarray:
        if returns is not None:
            array = np.asarray(returns, dtype=float)
            if array.ndim > 1:
                array = array.ravel()
            if len(array) != len(df):
                array = self._align_length(array, len(df))
            return array
        close = df["close"].astype(float)
        pct = close.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return pct.to_numpy(dtype=float)

    def _prepare_regime_predictions(
        self,
        regime_predictions: Optional[Union[np.ndarray, Sequence[int]]],
        length: int,
    ) -> np.ndarray:
        if regime_predictions is None:
            return np.zeros(length, dtype=int)
        labels = np.asarray(regime_predictions, dtype=int)
        if labels.ndim != 1:
            labels = labels.ravel()
        if len(labels) != length:
            labels = self._align_length(labels, length).astype(int)
        return labels

    def _prepare_regime_probabilities(
        self,
        regime_probabilities: Optional[Union[np.ndarray, Sequence[Sequence[float]]]],
        length: int,
        n_regimes: int,
    ) -> Optional[np.ndarray]:
        if regime_probabilities is None:
            return None
        probs = np.asarray(regime_probabilities, dtype=float)
        if probs.ndim == 1:
            probs = probs.reshape(-1, 1)
        if probs.shape[0] != length:
            probs = self._align_length(probs, length)
        if probs.shape[1] != n_regimes:
            adjusted = np.zeros((length, n_regimes), dtype=float)
            cols = min(probs.shape[1], n_regimes)
            adjusted[:, :cols] = probs[:, :cols]
            probs = adjusted
        row_sums = probs.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            probs = np.divide(probs, row_sums, out=np.full_like(probs, 1.0 / n_regimes), where=row_sums > 0)
        return probs

    def _calculate_metrics(
        self,
        returns: np.ndarray,
        regime_predictions: np.ndarray,
        regime_probabilities: Optional[np.ndarray],
        predictions: Optional[Any],
    ) -> EconomicMetrics:
        epsilon = 1e-12
        mean_return = float(np.mean(returns))
        excess_returns = returns - self.config.risk_free_rate / 252.0
        volatility = float(np.std(returns) + epsilon)
        sharpe_ratio = float(np.mean(excess_returns) / (np.std(excess_returns) + epsilon))
        downside = returns.copy()
        downside[downside > 0] = 0
        sortino = float(np.mean(excess_returns) / (np.std(downside) + epsilon))
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / (running_max + epsilon)
        max_drawdown = float(np.min(drawdowns))
        annualised_return = mean_return * 252
        calmar = float(annualised_return / (abs(max_drawdown) + epsilon))
        information_ratio = float(np.mean(excess_returns) / (volatility + epsilon))
        win_rate = float(np.mean(returns > 0)) if len(returns) else 0.0
        positive = returns[returns > 0]
        negative = returns[returns < 0]
        gross_profit = float(np.sum(positive))
        gross_loss = float(np.abs(np.sum(negative)))
        profit_factor = float(gross_profit / (gross_loss + epsilon))
        transition_count = np.sum(regime_predictions[1:] != regime_predictions[:-1])
        trading_frequency = float(transition_count / max(len(returns), 1))

        economic_significance = self._score_from_threshold(
            annualised_return, self.config.significance_threshold
        )
        viability_components = [
            self._score_from_threshold(win_rate, 0.5),
            self._score_from_threshold(profit_factor, 1.5),
            self._score_from_threshold(-max_drawdown, -0.2),
        ]
        trading_viability = float(np.clip(np.mean(viability_components), 0.0, 1.0))
        overall_score = float(np.clip(0.6 * economic_significance + 0.4 * trading_viability, 0.0, 1.0))

        return EconomicMetrics(
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            economic_significance=economic_significance,
            trading_viability=trading_viability,
            overall_score=overall_score,
            return_ratio=annualised_return,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=information_ratio,
            trading_frequency=trading_frequency,
            win_rate=win_rate,
            profit_factor=profit_factor,
        )

    def _calculate_regime_scores(
        self,
        regime_predictions: np.ndarray,
        economic_significance: float,
    ) -> np.ndarray:
        if regime_predictions.size == 0:
            return np.zeros(0, dtype=float)
        unique_regimes = np.unique(regime_predictions)
        scores = np.full(len(unique_regimes), economic_significance, dtype=float)
        return scores

    @staticmethod
    def _align_length(array: np.ndarray, target_length: int) -> np.ndarray:
        if len(array) == target_length:
            return array
        if len(array) == 0:
            return np.zeros(target_length)
        if len(array) > target_length:
            return array[-target_length:]
        padding = np.repeat(array[-1], target_length - len(array))
        return np.concatenate([array, padding])

    @staticmethod
    def _looks_like_timestamps(candidate: Any) -> bool:
        try:
            array = np.asarray(candidate)
            if array.ndim == 0:
                return False
            return np.issubdtype(array.dtype, np.datetime64) or array.dtype == object
        except Exception:
            return False

    @staticmethod
    def _score_from_threshold(value: float, threshold: float) -> float:
        if threshold == 0:
            return float(np.clip(0.5 + 0.5 * np.tanh(value), 0.0, 1.0))
        scaled = (value - threshold) / (abs(threshold) + 1e-12)
        return float(np.clip(0.5 + 0.5 * np.tanh(scaled), 0.0, 1.0))

    @staticmethod
    def _score_to_level(score: float) -> str:
        if score >= 0.8:
            return "excellent"
        if score >= 0.6:
            return "good"
        if score >= 0.4:
            return "fair"
        return "poor"

    @staticmethod
    def _resolve_architecture_type(architecture_type: Optional[Union[str, ArchitectureType]]) -> ArchitectureType:
        if isinstance(architecture_type, ArchitectureType):
            return architecture_type
        if isinstance(architecture_type, str):
            lowered = architecture_type.lower()
            for member in ArchitectureType:
                if member.value == lowered:
                    return member
        return ArchitectureType.HYBRID


class UnifiedEconomicSignificanceEvaluator:
    """Thin wrapper that exposes significance-specific helpers."""

    def __init__(self, config: Optional[EconomicEvaluationConfig] = None) -> None:
        self.base_evaluator = UnifiedEconomicEvaluator(config)
        self.config = self.base_evaluator.config
        self.logger = logging.getLogger(self.__class__.__name__)

    def evaluate(
        self,
        market_data: Union[pd.DataFrame, np.ndarray, Sequence[Sequence[float]]],
        regime_predictions: Optional[Union[np.ndarray, Sequence[int]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> EconomicSignificanceResult:
        result = self.base_evaluator.evaluate(
            market_data, regime_predictions, *args, **kwargs
        )
        significance = EconomicSignificanceResult(
            overall_score=result.overall_score,
            significance_level=result.significance_level,
            component_scores={
                "economic_significance": result.economic_metrics.economic_significance,
                "trading_viability": result.economic_metrics.trading_viability,
                "sharpe_ratio": result.economic_metrics.sharpe_ratio,
                "profit_factor": result.economic_metrics.profit_factor,
            },
            regime_scores=result.regime_scores,
            evaluation_time=result.total_evaluation_time,
            metadata=result.metadata,
            success=result.success,
            error_message=result.error_message,
        )
        if self.config.enable_logging:
            self.logger.info(
                "📊 Economic significance evaluation", extra={"score": significance.overall_score}
            )
        return significance


# Convenience factories -------------------------------------------------

def create_unified_economic_evaluator(
    config: Optional[EconomicEvaluationConfig] = None,
) -> UnifiedEconomicEvaluator:
    return UnifiedEconomicEvaluator(config)


def quick_economic_evaluation(
    market_data: Union[pd.DataFrame, np.ndarray, Sequence[Sequence[float]]],
    regime_predictions: Optional[Union[np.ndarray, Sequence[int]]] = None,
    config: Optional[EconomicEvaluationConfig] = None,
) -> EconomicEvaluationResult:
    evaluator = create_unified_economic_evaluator(config)
    return evaluator.evaluate(market_data, regime_predictions)


__all__ = [
    "UnifiedEconomicEvaluator",
    "UnifiedEconomicSignificanceEvaluator",
    "EconomicEvaluationConfig",
    "EconomicEvaluationResult",
    "EconomicSignificanceResult",
    "EconomicMetrics",
    "EvaluationType",
    "ArchitectureType",
    "create_unified_economic_evaluator",
    "quick_economic_evaluation",
]
