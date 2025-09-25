"""Unified NAS & TAS regime detection utilities."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .unified_regime_config import RegimeDetectionMethod, UnifiedRegimeConfig
from .unified_result import UnifiedRegimeResult

try:  # pragma: no cover - optional colourful logging
    from src.utils.tprint import (
        tprint,
        tprint_debug,
        tprint_error,
        tprint_info,
        tprint_success,
        tprint_warning,
    )

    _TPRINT_AVAILABLE = True
except Exception:  # pragma: no cover - fallback used in tests
    _TPRINT_AVAILABLE = False

    def _plain_print(prefix: str, *args: Any) -> None:
        print(prefix, *args)

    def tprint(*args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        print(*args)

    def tprint_info(*args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        _plain_print("INFO:", *args)

    def tprint_debug(*args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        _plain_print("DEBUG:", *args)

    def tprint_warning(*args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        _plain_print("WARNING:", *args)

    def tprint_error(*args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        _plain_print("ERROR:", *args)

    def tprint_success(*args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        _plain_print("SUCCESS:", *args)


@dataclass
class DetectionPerformance:
    """Snapshot of a detection run used for lightweight telemetry."""

    timestamp: float
    detection_time: float
    n_samples: int
    n_regimes: int
    method: str
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedRegimeDetector:
    """High level orchestrator that blends NAS & TAS regime logic.

    The historical codebase contained a very large implementation in this module
    that ended up getting truncated which made the launcher crash at import
    time.  This rewritten version focuses on a reliable, lightweight core that
    produces deterministic yet meaningful placeholder regimes.  It keeps the
    public API intact so higher level training and monitoring components keep
    working while the team iterates on richer models.
    """

    def __init__(self, config: Optional[UnifiedRegimeConfig] = None) -> None:
        self.config = config or UnifiedRegimeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.performance_history: List[DetectionPerformance] = []
        self.last_result: Optional[UnifiedRegimeResult] = None

        tprint_info(
            "🚀 UnifiedRegimeDetector initialised",
            f"method={self.config.detection_method.value}",
            f"regimes={self.config.n_regimes}",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect_regimes(
        self,
        market_data: Union[pd.DataFrame, np.ndarray, Iterable[Iterable[float]]],
        timestamps: Optional[Union[np.ndarray, Iterable[Any]]] = None,
    ) -> UnifiedRegimeResult:
        """Run the simplified unified regime detector.

        The implementation favours robustness over sophistication: it performs
        a couple of basic statistical transforms to produce regime labels and
        attaches quality metrics that downstream systems expect.  The method
        mirrors the signature used across the project so existing callers do
        not require any adjustments.
        """

        start_time = time.time()
        try:
            data_frame = self._normalise_market_data(market_data, timestamps)
            feature_frame = self._compute_features(data_frame)
            regime_labels, regime_probabilities = self._assign_regimes(feature_frame)

            economic_scores = self._compute_economic_significance(feature_frame)
            trading_scores = self._compute_trading_viability(feature_frame)
            stability_scores = self._compute_stability_scores(regime_labels)
            transition_probabilities = self._compute_transition_matrix(regime_labels)

            execution_time = time.time() - start_time

            result = UnifiedRegimeResult(
                success=True,
                regime_predictions=regime_labels,
                regime_probabilities=regime_probabilities,
                economic_significance_scores=economic_scores,
                trading_viability_scores=trading_scores,
                regime_stability_scores=stability_scores,
                transition_probabilities=transition_probabilities,
                micro_regimes=None,
                performance_metrics={
                    "detection_time": execution_time,
                    "n_samples": len(regime_labels),
                    "method": self.config.detection_method.value,
                },
                execution_time=execution_time,
                system_type="unified",
                architecture_used=self.config.detection_method.value,
                metadata=self._build_metadata(feature_frame, regime_labels, execution_time),
                error_message=None,
            )

            self._record_performance(result)
            self.last_result = result

            tprint_success(
                "✅ Unified regime detection completed",
                f"regimes_detected={len(np.unique(regime_labels))}",
                f"execution_time={execution_time:.3f}s",
            )
            return result
        except Exception as exc:  # pragma: no cover - defensive logging path
            execution_time = time.time() - start_time
            tprint_error(f"Unified regime detection failed: {exc}")
            self.logger.exception("Unified regime detection failure")
            failure = UnifiedRegimeResult(
                success=False,
                regime_predictions=np.array([]),
                regime_probabilities=np.array([]),
                economic_significance_scores=np.array([]),
                trading_viability_scores=np.array([]),
                regime_stability_scores=np.array([]),
                transition_probabilities=np.array([]),
                execution_time=execution_time,
                metadata={"error": str(exc)},
                error_message=str(exc),
            )
            self._record_performance(failure)
            self.last_result = failure
            return failure

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Return aggregate statistics for recent detections."""

        if not self.performance_history:
            return {"total_runs": 0}

        detection_times = [p.detection_time for p in self.performance_history]
        success_rates = [1.0 if p.success else 0.0 for p in self.performance_history]

        metrics = {
            "total_runs": len(self.performance_history),
            "average_detection_time": float(np.mean(detection_times)),
            "fastest_detection_time": float(np.min(detection_times)),
            "slowest_detection_time": float(np.max(detection_times)),
            "success_rate": float(np.mean(success_rates)),
            "last_method": self.performance_history[-1].method,
            "last_samples": self.performance_history[-1].n_samples,
        }

        return metrics

    def save_results(self, result: UnifiedRegimeResult, filepath: Union[str, Path]) -> None:
        """Persist a detection result to JSON for reproducibility."""

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        serialisable = self._to_serialisable(result.to_dict())
        with path.open("w", encoding="utf-8") as handle:
            json.dump(serialisable, handle, indent=2)

        tprint_debug(f"💾 Unified regime result saved to {path}")

    def load_results(self, filepath: Union[str, Path]) -> UnifiedRegimeResult:
        """Load a previously persisted detection result."""

        path = Path(filepath)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        result = UnifiedRegimeResult.from_dict(data)
        self.last_result = result
        tprint_debug(f"📂 Unified regime result loaded from {path}")
        return result

    def summarize_result(self, result: Optional[UnifiedRegimeResult] = None) -> Dict[str, Any]:
        """Return a lightweight summary for dashboards and logs."""

        target = result or self.last_result
        if target is None:
            return {"success": False, "error": "no_result"}
        return target.get_summary()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record_performance(self, result: UnifiedRegimeResult) -> None:
        detection_time = result.execution_time
        n_samples = int(len(result.regime_predictions))
        performance = DetectionPerformance(
            timestamp=time.time(),
            detection_time=float(detection_time),
            n_samples=n_samples,
            n_regimes=int(result.metadata.get("n_regimes", 0) if result.metadata else 0),
            method=self.config.detection_method.value,
            success=result.success,
            metadata=result.metadata or {},
        )
        self.performance_history.append(performance)
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    def _normalise_market_data(
        self,
        market_data: Union[pd.DataFrame, np.ndarray, Iterable[Iterable[float]]],
        timestamps: Optional[Union[np.ndarray, Iterable[Any]]],
    ) -> pd.DataFrame:
        if isinstance(market_data, pd.DataFrame):
            df = market_data.copy()
        else:
            array = np.asarray(list(market_data))
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            if array.shape[0] == 0:
                raise ValueError("Market data is empty")

            default_columns = ["open", "high", "low", "close", "volume"]
            column_count = min(array.shape[1], len(default_columns))
            columns = default_columns[:column_count]
            df = pd.DataFrame(array[:, :column_count], columns=columns)

        if timestamps is not None:
            ts_array = np.asarray(list(timestamps))
            if len(ts_array) != len(df):
                raise ValueError("Timestamp length does not match market data length")
            df = df.copy()
            df["timestamp"] = pd.to_datetime(ts_array)
        elif "timestamp" not in df.columns:
            df["timestamp"] = pd.date_range(
                end=pd.Timestamp.utcnow(), periods=len(df), freq="min"
            )

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if numeric_columns.empty:
            raise ValueError("No numeric columns available for regime detection")

        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df.fillna(method="ffill").fillna(method="bfill")
        return df

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_df = df.select_dtypes(include=[np.number]).copy()
        if "close" not in feature_df.columns:
            first_numeric = feature_df.columns[0]
            feature_df["close"] = feature_df[first_numeric]

        feature_df["returns"] = feature_df["close"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        feature_df["volatility"] = (
            feature_df["returns"].rolling(window=10, min_periods=1).std().fillna(0.0)
        )
        feature_df["momentum"] = (
            feature_df["close"].pct_change(periods=5).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        )
        feature_df["trend_strength"] = (
            feature_df["momentum"].rolling(window=5, min_periods=1).mean().fillna(0.0)
        )
        feature_df["volume_zscore"] = (
            (feature_df.get("volume", feature_df["close"]).rolling(window=20, min_periods=1).mean())
        )
        feature_df = feature_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return feature_df

    def _assign_regimes(
        self, feature_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_regimes = max(2, int(self.config.n_regimes))
        returns = feature_df["returns"].to_numpy()

        quantiles = np.linspace(0, 100, num=n_regimes + 1)
        thresholds = np.percentile(returns, quantiles)
        # Avoid identical thresholds by adding a tiny epsilon when needed
        for idx in range(1, len(thresholds)):
            if thresholds[idx] <= thresholds[idx - 1]:
                thresholds[idx] = thresholds[idx - 1] + 1e-8

        regime_labels = np.digitize(returns, thresholds[1:-1], right=False).astype(int)
        centres = 0.5 * (thresholds[:-1] + thresholds[1:])
        probabilities = self._compute_probabilities(returns, centres)
        return regime_labels, probabilities

    def _compute_probabilities(self, returns: np.ndarray, centres: np.ndarray) -> np.ndarray:
        probabilities: List[np.ndarray] = []
        epsilon = 1e-8
        for value in returns:
            distances = np.abs(value - centres) + epsilon
            weights = 1.0 / distances
            probs = weights / np.sum(weights)
            probabilities.append(probs)
        return np.asarray(probabilities)

    def _compute_economic_significance(self, feature_df: pd.DataFrame) -> np.ndarray:
        returns = feature_df["returns"].to_numpy()
        mean = np.mean(returns)
        std = np.std(returns) + 1e-8
        z_scores = (returns - mean) / std
        scaled = 0.5 + 0.5 * np.tanh(z_scores)
        return np.clip(scaled, 0.0, 1.0)

    def _compute_trading_viability(self, feature_df: pd.DataFrame) -> np.ndarray:
        volatility = feature_df["volatility"].to_numpy()
        max_vol = np.max(volatility) + 1e-8
        viability = 1.0 - (volatility / max_vol)
        momentum = feature_df["momentum"].to_numpy()
        momentum_scaled = 0.5 + 0.5 * np.tanh(momentum)
        return np.clip(0.6 * viability + 0.4 * momentum_scaled, 0.0, 1.0)

    def _compute_stability_scores(self, regimes: np.ndarray) -> np.ndarray:
        stability = np.ones_like(regimes, dtype=float)
        streak = 1
        for idx in range(1, len(regimes)):
            if regimes[idx] == regimes[idx - 1]:
                streak += 1
            else:
                streak = 1
            stability[idx] = min(1.0, streak / max(self.config.min_regime_samples, 1))
        return stability

    def _compute_transition_matrix(self, regimes: np.ndarray) -> np.ndarray:
        n_regimes = max(2, int(self.config.n_regimes))
        matrix = np.zeros((n_regimes, n_regimes), dtype=float)
        for prev, curr in zip(regimes[:-1], regimes[1:]):
            if 0 <= prev < n_regimes and 0 <= curr < n_regimes:
                matrix[int(prev), int(curr)] += 1.0

        row_sums = matrix.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
        return matrix

    def _build_metadata(
        self, feature_df: pd.DataFrame, regimes: np.ndarray, execution_time: float
    ) -> Dict[str, Any]:
        return {
            "n_regimes": int(len(np.unique(regimes))),
            "execution_time": float(execution_time),
            "detection_method": self.config.detection_method.value,
            "mean_return": float(feature_df["returns"].mean()),
            "std_return": float(feature_df["returns"].std()),
            "mean_volatility": float(feature_df["volatility"].mean()),
            "timestamp": time.time(),
        }

    def _to_serialisable(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, dict):
            return {k: self._to_serialisable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_serialisable(v) for v in value]
        return value


__all__ = ["UnifiedRegimeDetector", "UnifiedRegimeResult"]
