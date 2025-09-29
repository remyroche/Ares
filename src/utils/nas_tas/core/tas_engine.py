"""Trading Architecture Search engine built on the shared NAS/TAS base."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...common_operations import (
    calculate_data_quality_metrics,
    create_data_quality_report,
    guard_dataframe_nulls,
    memory_checkpoint,
    optimize_dataframe_dtypes,
    safe_copy,
    validate_dataframe_columns,
)
from ...data.klines_parquet import validate_klines_data
from ...math_validation import (
    safe_divide,
    safe_kelly_calculation,
    safe_mean,
    safe_power,
    safe_sqrt,
    safe_std,
    safe_weighted_average,
    validate_numeric_array,
)
from ...tprint import (
    LogLevel,
    tprint_debug,
    tprint_error,
    tprint_info,
    tprint_logged,
    tprint_structured,
    tprint_timer,
)
from ..ml_common.optimization.regime_specific_tpsl_optimizer import (
    RegimeSpecificTPSLOptimizer,
)
from ..ml_common.optimization.shared_utils.evaluation_metrics import (
    FinancialMetricCalculator,
)
from ...data.basic_returns_engineer import BasicReturnsEngineer
from ...data.feature_engineer import FeatureEngineer
from ...data.gap_detector import GapDetector
from ...data.unified_data_utils import UnifiedDataUtils
from ...data.processing.data_processing import DataProcessor
from ...matrix_operations.convenience import MatrixConvenience
from .base_engine import BaseSearchEngine


@tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
class TASEngine(BaseSearchEngine):
    """Trading Architecture Search engine built on :class:`BaseSearchEngine`."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config, engine_name="TASEngine")
        self.search_context_name = "strategy_search"
        self.evaluation_context_name = "strategy_evaluation"

        # TAS specific utilities
        self.data_processor = DataProcessor()
        self.returns_engineer = BasicReturnsEngineer()
        self.feature_engineer = FeatureEngineer()
        self.gap_detector = GapDetector()
        self.unified_data_utils = UnifiedDataUtils()
        self.matrix_convenience = MatrixConvenience()
        self.regime_tpsl_optimizer = RegimeSpecificTPSLOptimizer()
        self.financial_metric_calculator = FinancialMetricCalculator()

        self.strategy_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @tprint_timer("Data Loading and Processing")
    def load_and_process_data(
        self,
        symbol: str = "ETHUSDT",
        interval: str = "1m",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        apply_feature_engineering: bool = True,
    ) -> Optional[pd.DataFrame]:
        """Load raw data and run the TAS processing pipeline."""

        tprint_info(f"📊 Loading and processing data for {symbol} {interval}")
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

            validation_result = validate_klines_data(data)
            if not validation_result.get("valid", False):
                tprint_error(
                    f"❌ Data validation failed: {validation_result.get('errors', 'unknown')}"
                )
                return None

            tprint_info(f"📈 Data quality metrics: {calculate_data_quality_metrics(data)}")

            with memory_checkpoint("data_processing"):
                processed = self._process_trading_data(
                    data, apply_feature_engineering=apply_feature_engineering
                )

            if processed is None or processed.empty:
                tprint_error("❌ Data processing failed")
                return None

            processed = optimize_dataframe_dtypes(processed)
            processed = guard_dataframe_nulls(processed, threshold=0.1)

            report = create_data_quality_report(processed)
            tprint_structured(report, LogLevel.INFO)

            return processed
        except Exception as exc:  # pragma: no cover - defensive logging
            tprint_error(f"❌ Error loading and processing data: {exc}")
            self.logger.exception("Data loading and processing error")
            return None

    def _process_trading_data(
        self, data: pd.DataFrame, *, apply_feature_engineering: bool
    ) -> Optional[pd.DataFrame]:
        try:
            processed = safe_copy(data)

            with memory_checkpoint("returns_engineering"):
                processed = self.returns_engineer.add_basic_returns(processed)

            with memory_checkpoint("gap_detection"):
                gaps = self.gap_detector.detect_gaps(processed)
                if gaps:
                    tprint_info(f"🔍 Detected {len(gaps)} gaps in data")

            if apply_feature_engineering:
                with memory_checkpoint("feature_engineering"):
                    processed = self.feature_engineer.add_technical_indicators(processed)
                    processed = self.feature_engineer.add_price_features(processed)
                    processed = self.feature_engineer.add_volume_features(processed)
                    processed = self.feature_engineer.add_time_features(processed)

            with memory_checkpoint("unified_processing"):
                processed = self.unified_data_utils.standardize_data(processed)
                processed = self.unified_data_utils.add_derived_features(processed)

            required_cols = ["open", "high", "low", "close", "volume"]
            if not validate_dataframe_columns(processed, required_cols):
                tprint_error("❌ Processed data missing required columns")
                return None

            return processed
        except Exception as exc:  # pragma: no cover - defensive logging
            tprint_error(f"❌ Error processing trading data: {exc}")
            return None

    # ------------------------------------------------------------------
    # Search orchestration
    # ------------------------------------------------------------------

    @tprint_timer("Strategy Search")
    def search_strategies(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        *,
        optimization_method: str = "bayesian_tpe",
        n_trials: int = 100,
        include_regime_specific: bool = True,
    ) -> Dict[str, Any]:
        """Execute the shared search loop with TAS specific behaviour."""

        required_cols = ["open", "high", "low", "close", "volume"]
        if not validate_dataframe_columns(data, required_cols):
            tprint_error("❌ Invalid data columns for strategy search")
            return {}

        regime_analysis = None
        if include_regime_specific:
            regime_analysis = self._analyze_regimes(data)

        results = self.run_search(
            data,
            search_space,
            optimization_method=optimization_method,
            n_trials=n_trials,
            extra_context={"regime_analysis": regime_analysis} if regime_analysis else None,
        )
        results["best_strategy"] = results.pop("best_params")
        results["regime_analysis"] = regime_analysis

        self.strategy_history.append(results)
        return results

    # ------------------------------------------------------------------
    # Regime analysis
    # ------------------------------------------------------------------

    def _analyze_regimes(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            price_data = data[["open", "high", "low", "close", "volume"]].values

            with memory_checkpoint("regime_analysis"):
                rolling_returns = self.matrix_ops.calculate_rolling_returns(price_data)
                volatility = self.matrix_ops.calculate_rolling_volatility(rolling_returns)
                trend_strength = self.matrix_ops.calculate_trend_strength(price_data)
                regime_features = np.column_stack([volatility, trend_strength])
                regimes = self.vectorized_core.classify_regimes(regime_features)

            regime_stats: Dict[str, Any] = {}
            unique_regimes = np.unique(regimes)
            for regime in unique_regimes:
                mask = regimes == regime
                regime_data = data[mask]
                if regime_data.empty:
                    continue
                regime_stats[f"regime_{regime}"] = {
                    "count": len(regime_data),
                    "percentage": safe_divide(len(regime_data), len(data)) * 100,
                    "avg_volatility": safe_mean(volatility[mask]),
                    "avg_trend": safe_mean(trend_strength[mask]),
                    "avg_return": safe_mean(rolling_returns[mask]),
                }

            tprint_info(f"🔍 Detected {len(unique_regimes)} market regimes")
            return {
                "regimes": regimes,
                "regime_stats": regime_stats,
                "features": {
                    "volatility": volatility,
                    "trend_strength": trend_strength,
                    "returns": rolling_returns,
                },
            }
        except Exception as exc:  # pragma: no cover - defensive logging
            tprint_error(f"❌ Error in regime analysis: {exc}")
            return {}

    # ------------------------------------------------------------------
    # Base hook implementations
    # ------------------------------------------------------------------

    def _create_feature_matrix(self, data: pd.DataFrame, **_: Any) -> np.ndarray:
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            feature_data = data[numeric_cols].values
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
            feature_data = validate_numeric_array(feature_data, "strategy_features")

            normalized_features = self.matrix_ops.normalize_matrix(feature_data)
            technical_features = self.enhanced_matrix_ops.add_technical_features(
                normalized_features
            )
            trading_features = self.matrix_convenience.add_trading_features(
                technical_features
            )
            return trading_features
        except Exception as exc:  # pragma: no cover - defensive logging
            tprint_error(f"❌ Error creating strategy feature matrix: {exc}")
            return np.array([])

    def _compute_score(
        self,
        feature_matrix: np.ndarray,
        params: Dict[str, Any],
        **extra: Any,
    ) -> float:
        try:
            regime_analysis = extra.get("regime_analysis")
            entry_threshold = params.get("entry_threshold", 0.5)
            exit_threshold = params.get("exit_threshold", 0.5)
            risk_factor = params.get("risk_factor", 1.0)
            position_size = params.get("position_size", 0.1)

            base_score = self.vectorized_core.compute_strategy_performance(
                feature_matrix, entry_threshold, exit_threshold
            )

            regime_adjustment = 1.0
            if regime_analysis and "regime_stats" in regime_analysis:
                regime_adjustment = self._calculate_regime_adjustment(
                    regime_analysis["regime_stats"], params
                )

            risk_adjustment = safe_power(risk_factor, 0.5)
            position_adjustment = safe_sqrt(position_size)

            return safe_weighted_average(
                [base_score, regime_adjustment, risk_adjustment, position_adjustment],
                [0.6, 0.2, 0.1, 0.1],
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            tprint_error(f"❌ Error computing strategy score: {exc}")
            return 0.0

    def _calculate_metrics(self, trials: List[Dict[str, Any]]) -> Dict[str, Any]:
        metrics = super()._calculate_metrics(trials)
        if not trials:
            return metrics

        scores = [trial["score"] for trial in trials]
        metrics.update(
            {
                "improvement_rate": self._calculate_strategy_improvement_rate(scores),
                "convergence_metric": self._calculate_strategy_convergence_metric(scores),
                "risk_adjusted_score": self._calculate_risk_adjusted_score(scores),
            }
        )
        return metrics

    # ------------------------------------------------------------------
    # TAS specific metric helpers
    # ------------------------------------------------------------------

    def _calculate_regime_adjustment(
        self, regime_stats: Dict[str, Any], params: Dict[str, Any]
    ) -> float:
        try:
            adjustments: List[float] = []
            for stats in regime_stats.values():
                if not isinstance(stats, dict):
                    continue
                volatility = stats.get("avg_volatility", 0.0)
                trend = stats.get("avg_trend", 0.0)
                percentage = stats.get("percentage", 0.0)
                regime_factor = safe_weighted_average([volatility, abs(trend)], [0.7, 0.3])
                adjustments.append(safe_divide(regime_factor * percentage, 100.0))
            return safe_mean(np.array(adjustments)) if adjustments else 1.0
        except Exception as exc:  # pragma: no cover - defensive logging
            tprint_debug(f"⚠️ Error calculating regime adjustment: {exc}")
            return 1.0

    def _calculate_strategy_improvement_rate(self, scores: List[float]) -> float:
        if len(scores) < 2:
            return 0.0
        improvements = sum(
            1 for previous, current in zip(scores[:-1], scores[1:]) if current > previous
        )
        return safe_divide(improvements, len(scores) - 1)

    def _calculate_strategy_convergence_metric(self, scores: List[float]) -> float:
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

    def _calculate_risk_adjusted_score(self, scores: List[float]) -> float:
        if len(scores) < 2:
            return safe_mean(np.array(scores))
        scores_array = np.array(scores, dtype=float)
        metrics = self.financial_metric_calculator.calculate(
            predictions=scores_array,
            targets=scores_array,
            returns=scores_array,
        )
        mean_score = safe_mean(scores_array)
        kelly_fraction = safe_kelly_calculation(
            win_rate=safe_divide(np.sum(scores_array > 0), len(scores_array)),
            avg_win=safe_mean(scores_array[scores_array > 0]) if np.any(scores_array > 0) else 0.0,
            avg_loss=abs(safe_mean(scores_array[scores_array < 0]))
            if np.any(scores_array < 0)
            else 0.0,
        )
        return mean_score * (1 + getattr(metrics, "sharpe_ratio", 0.0) + kelly_fraction)
