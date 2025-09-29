"""Neural Architecture Search engine built on the shared NAS/TAS base."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...common_operations import (
    calculate_data_quality_metrics,
    guard_dataframe_nulls,
    memory_checkpoint,
    optimize_dataframe_dtypes,
    validate_dataframe_columns,
)
from ...data.klines_parquet import validate_klines_data
from ...math_validation import (
    safe_divide,
    safe_log,
    safe_mean,
    safe_power,
    safe_sqrt,
    safe_std,
    safe_weighted_average,
)
from ...tprint import (
    LogLevel,
    tprint_debug,
    tprint_error,
    tprint_info,
    tprint_logged,
    tprint_timer,
)
from .base_engine import BaseSearchEngine


@tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
class NASEngine(BaseSearchEngine):
    """Neural Architecture Search engine leveraging :class:`BaseSearchEngine`."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config, engine_name="NASEngine")
        self.search_context_name = "architecture_search"
        self.evaluation_context_name = "architecture_evaluation"

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @tprint_timer("Data Loading and Validation")
    def load_and_validate_data(
        self,
        symbol: str = "ETHUSDT",
        interval: str = "1m",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Load market data using the shared parquet manager and validate it."""

        tprint_info(f"📊 Loading data for {symbol} {interval}")
        try:
            with memory_checkpoint("data_loading"):
                data = self.klines_manager.read_data(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date,
                    data_type="processed",
                )

            if data is None or data.empty:
                tprint_error(f"❌ No data loaded for {symbol} {interval}")
                return None

            tprint_debug(f"🔢 Loaded {len(data)} rows for NAS evaluation")

            validation_result = validate_klines_data(data)
            if not validation_result.get("valid", False):
                tprint_error(
                    f"❌ Data validation failed: {validation_result.get('errors', 'unknown')}"
                )
                return None

            quality_metrics = calculate_data_quality_metrics(data)
            tprint_info(f"📈 Data quality metrics: {quality_metrics}")

            data = optimize_dataframe_dtypes(data)
            data = guard_dataframe_nulls(data, threshold=0.1)

            return data
        except Exception as exc:  # pragma: no cover - defensive logging
            tprint_error(f"❌ Error loading data: {exc}")
            self.logger.exception("Data loading error")
            return None

    # ------------------------------------------------------------------
    # Search orchestration
    # ------------------------------------------------------------------

    @tprint_timer("Architecture Search")
    def search_architectures(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        *,
        optimization_method: str = "bayesian_tpe",
        n_trials: int = 100,
    ) -> Dict[str, Any]:
        """Run the shared search loop and expose NAS specific naming."""

        required_cols = ["open", "high", "low", "close", "volume"]
        if not validate_dataframe_columns(data, required_cols):
            tprint_error("❌ Invalid data columns for architecture search")
            return {}

        results = self.run_search(
            data,
            search_space,
            optimization_method=optimization_method,
            n_trials=n_trials,
        )
        results["best_architecture"] = results.pop("best_params")
        return results

    # ------------------------------------------------------------------
    # Base hook implementations
    # ------------------------------------------------------------------

    def _create_feature_matrix(self, data: pd.DataFrame, **_: Any) -> np.ndarray:
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            feature_data = data[numeric_cols].values
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)

            normalized_features = self.matrix_ops.normalize_matrix(feature_data)
            polynomial_features = self.enhanced_matrix_ops.add_polynomial_features(
                normalized_features, degree=2
            )
            return polynomial_features
        except Exception as exc:  # pragma: no cover - defensive logging
            tprint_error(f"❌ Error creating feature matrix: {exc}")
            return np.array([])

    def _compute_score(
        self, feature_matrix: np.ndarray, params: Dict[str, Any], **_: Any
    ) -> float:
        try:
            complexity = params.get("complexity", 1.0)
            depth = params.get("depth", 1)
            width = params.get("width", 1)

            base_score = self.vectorized_core.compute_performance_metric(
                feature_matrix, complexity, depth, width
            )

            complexity_factor = safe_power(complexity, 0.5)
            depth_factor = safe_log(depth + 1)
            width_factor = safe_sqrt(width)

            return safe_weighted_average(
                [base_score, complexity_factor, depth_factor, width_factor],
                [0.7, 0.1, 0.1, 0.1],
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            tprint_error(f"❌ Error computing architecture score: {exc}")
            return 0.0

    def _calculate_metrics(self, trials: List[Dict[str, Any]]) -> Dict[str, Any]:
        metrics = super()._calculate_metrics(trials)
        if not trials:
            return metrics

        scores = [trial["score"] for trial in trials]
        metrics.update(
            {
                "improvement_rate": self._calculate_improvement_rate(scores),
                "convergence_metric": self._calculate_convergence_metric(scores),
            }
        )
        return metrics

    # ------------------------------------------------------------------
    # NAS specific metric helpers
    # ------------------------------------------------------------------

    def _calculate_improvement_rate(self, scores: List[float]) -> float:
        if len(scores) < 2:
            return 0.0
        improvements = sum(
            1 for previous, current in zip(scores[:-1], scores[1:]) if current > previous
        )
        return safe_divide(improvements, len(scores) - 1)

    def _calculate_convergence_metric(self, scores: List[float]) -> float:
        if len(scores) < 10:
            return 0.0
        window = max(1, len(scores) // 5)
        recent_scores = np.array(scores[-window:], dtype=float)
        mean_score = safe_mean(recent_scores)
        if mean_score == 0:
            return 0.0
        std_score = safe_std(recent_scores)
        coefficient = safe_divide(std_score, abs(mean_score))
        return 1.0 - coefficient


def create_nas_engine(config: Optional[Dict[str, Any]] = None) -> NASEngine:
    """Compatibility helper mirroring the historical factory function."""

    return NASEngine(config)
