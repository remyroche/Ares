"""Shared Pandera schemas for pre-training data validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Union

import math

import numpy as np
import pandas as pd
from pandas import Series
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import pandera as pa
    from pandera import errors as pa_errors
except ModuleNotFoundError:  # pragma: no cover - fallback for test environments without pandera
    from ._pandera_stub import pandera_stub as pa
    from ._pandera_stub import errors as pa_errors

try:  # pragma: no cover - optional utility import
    from ..quantitative_validation import calculate_information_coefficient
except Exception:  # pragma: no cover - guard against circular imports or absence
    calculate_information_coefficient = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------
try:
    from statsmodels.stats.multitest import multipletests
except ModuleNotFoundError:  # pragma: no cover - optional dependency for multiple testing adjustments
    multipletests = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..multi_horizon_profit_labeler import MultiHorizonConfig


__all__ = [
    "RAW_OHLCV_SCHEMA",
    "LABELED_DATASET_SCHEMA",
    "ENGINEERED_FEATURE_SCHEMA",
    "SCHEMA_REGISTRY",
    "SchemaValidationException",
    "validate_raw_ohlcv",
    "validate_labeled_dataset",
    "validate_engineered_features",
    "enforce_feature_temporal_alignment",
    "schema_metadata",
    "HypothesisTracker",
    "apply_multiple_testing_correction",
    "report_hypothesis_count",
    "SplitAwareScaler",
]


RAW_OHLCV_SCHEMA = pa.DataFrameSchema(
    columns={
        "open": pa.Column(pa.Float, coerce=True, nullable=False),
        "high": pa.Column(pa.Float, coerce=True, nullable=False),
        "low": pa.Column(pa.Float, coerce=True, nullable=False),
        "close": pa.Column(pa.Float, coerce=True, nullable=False),
        "volume": pa.Column(pa.Float, coerce=True, nullable=False),
    },
    index=pa.Index(pa.DateTime, coerce=True, nullable=False),
    strict=False,
    coerce=True,
)
"""Schema describing the normalized OHLCV frame produced by the loaders."""


LABELED_DATASET_SCHEMA = pa.DataFrameSchema(
    columns={
        "immediate_opportunity": pa.Column(pa.Int64, coerce=True, nullable=False),
        "short_term_opportunity": pa.Column(pa.Int64, coerce=True, nullable=False),
        "leverage_adjusted_score": pa.Column(pa.Float, coerce=True, nullable=False),
    },
    index=pa.Index(pa.DateTime, coerce=True, nullable=False),
    strict=False,
    coerce=True,
)
"""Schema describing the downstream-ready multi-horizon labeling frame."""


ENGINEERED_FEATURE_SCHEMA = pa.DataFrameSchema(
    columns={
        r"^.+$": pa.Column(pa.Float64, regex=True, required=False, coerce=True, nullable=False),
    },
    index=pa.Index(pa.DateTime, coerce=True, nullable=False),
    strict=False,
    coerce=True,
)
"""Schema describing engineered feature matrices used throughout the pipeline."""


SCHEMA_REGISTRY: Dict[str, Dict[str, str]] = {
    "raw_ohlcv": {
        "name": "Raw OHLCV Frame",
        "version": "1.0",
        "description": "Normalized OHLCV input with datetime index and float OHLCV columns.",
    },
    "labeled_dataset": {
        "name": "Multi-horizon Label Dataset",
        "version": "1.0",
        "description": "Label matrix exposing standardized opportunity targets for downstream use.",
    },
    "engineered_features": {
        "name": "Engineered Feature Frame",
        "version": "1.0",
        "description": "Feature matrix with datetime index and numeric feature columns.",
    },
}


@dataclass
class SchemaValidationException(RuntimeError):
    """Exception raised when a dataframe fails schema validation."""

    schema_key: str
    context: str
    original_error: pa_errors.SchemaError

    def __post_init__(self) -> None:
        super().__init__(format_schema_error(self.schema_key, self.context, self.original_error))


@dataclass
class HypothesisTracker:
    """Track outcomes for multiple hypothesis tests performed during validation.

    Downstream selection and diagnostics steps should instantiate this helper at the
    beginning of a batch of statistical tests and call :meth:`record` for each
    hypothesis. The tracker aggregates counters, stores raw p-values, and exposes
    helpers for common multiple-testing adjustments so diagnostics can be reported
    alongside schema validation artifacts.
    """

    accepted: int = 0
    rejected: int = 0
    inconclusive: int = 0
    pvalues: List[float] = field(default_factory=list)
    metadata: List[Dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        *,
        p_value: Optional[float],
        rejected: bool,
        metadata: Optional[Dict[str, Any]] = None,
        inconclusive: bool = False,
    ) -> None:
        """Register a single hypothesis evaluation.

        Args:
            p_value: Raw p-value for the hypothesis; ``None`` when unavailable.
            rejected: Whether the null hypothesis was rejected under the nominal
                significance level before multiple-testing correction.
            metadata: Optional contextual diagnostics (e.g., hypothesis labels or
                summary statistics) that will be attached to exported artifacts.
            inconclusive: Flag the test as inconclusive rather than accepted when
                the null hypothesis cannot be resolved decisively.
        """

        if rejected:
            self.rejected += 1
        elif inconclusive:
            self.inconclusive += 1
        else:
            self.accepted += 1

        if p_value is not None:
            self.pvalues.append(p_value)

        if metadata is not None:
            enriched_metadata = dict(metadata)
        else:
            enriched_metadata = {}
        enriched_metadata.update({"rejected": rejected, "p_value": p_value, "inconclusive": inconclusive})
        self.metadata.append(enriched_metadata)

    @property
    def total_hypotheses(self) -> int:
        """Total number of hypotheses that have been logged."""

        return self.accepted + self.rejected + self.inconclusive

    def bonferroni_threshold(self, alpha: float) -> float:
        """Return the Bonferroni-adjusted per-test significance threshold."""

        total = max(self.total_hypotheses, 1)
        return alpha / total

    def bonferroni_reject(self, p_value: float, alpha: float) -> bool:
        """Determine rejection under a Bonferroni-adjusted threshold."""

        return p_value <= self.bonferroni_threshold(alpha)

    def fdr_adjusted_pvalues(self, *, alpha: float = 0.05, method: str = "fdr_bh") -> List[float]:
        """Compute FDR-adjusted p-values for recorded hypotheses.

        Returns an empty list if no p-values have been recorded. Requires
        ``statsmodels`` for the underlying multiple testing procedure.
        """

        if not self.pvalues:
            return []
        if multipletests is None:  # pragma: no cover - exercised only when dependency missing
            raise RuntimeError(
                "statsmodels is required to compute FDR-adjusted p-values but is not installed."
            )

        _, corrected_pvalues, _, _ = multipletests(self.pvalues, alpha=alpha, method=method)
        return list(corrected_pvalues)


def apply_multiple_testing_correction(
    horizon_metrics: Mapping[str, Mapping[str, Any]],
    *,
    alpha: float = 0.05,
    method: str = "fdr_bh",
) -> Dict[str, Dict[str, Any]]:
    """Augment horizon metrics with multiple-testing adjustments.

    Parameters
    ----------
    horizon_metrics
        Mapping of horizon identifiers to metric dictionaries containing at
        least a ``p_value`` entry. The returned structure mirrors the input but
        adds corrected p-values, rejection flags, and metadata describing the
        correction.
    alpha
        Significance level used for the multiple-testing procedure.
    method
        Multiple-testing method understood by :func:`statsmodels.stats.multitest.multipletests`.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping with enriched metric dictionaries. Entries lacking a finite
        ``p_value`` receive metadata indicating that no adjustment was applied.
    """

    corrected: Dict[str, Dict[str, Any]] = {}
    if not horizon_metrics:
        return corrected

    ordered_metrics: List[Tuple[str, float]] = []
    for horizon, metrics in horizon_metrics.items():
        horizon_key = str(horizon)
        metrics_copy: Dict[str, Any] = dict(metrics) if isinstance(metrics, Mapping) else {}
        corrected[horizon_key] = metrics_copy

        raw_p = metrics_copy.get("p_value")
        if isinstance(raw_p, (int, float)) and math.isfinite(raw_p):
            ordered_metrics.append((horizon_key, float(raw_p)))

    hypothesis_count = len(ordered_metrics)
    base_metadata = {
        "method": method,
        "alpha": alpha,
        "hypothesis_count": hypothesis_count,
        "bonferroni_threshold": alpha / hypothesis_count if hypothesis_count else alpha,
        "dependency": "statsmodels" if multipletests is not None else "benjamini_hochberg_fallback",
    }

    adjusted_values: List[float] = []
    rejection_flags: List[bool] = []
    if hypothesis_count:
        if multipletests is not None:
            reject, adjusted, _, _ = multipletests(
                [value for _, value in ordered_metrics], alpha=alpha, method=method
            )
            rejection_flags = [bool(flag) for flag in reject]
            adjusted_values = [float(value) for value in adjusted]
        else:
            if method != "fdr_bh":  # pragma: no cover - guarded by tests to use default method
                raise RuntimeError(
                    "statsmodels is unavailable; only the 'fdr_bh' method is supported as a fallback."
                )
            fallback_adjusted = _bh_fdr_adjustment({key: value for key, value in ordered_metrics})
            adjusted_values = [fallback_adjusted[key] for key, _ in ordered_metrics]
            rejection_flags = [value <= alpha for value in adjusted_values]

    for index, (horizon_key, _raw_p) in enumerate(ordered_metrics):
        metrics = corrected[horizon_key]
        metrics["adjusted_p_value"] = float(adjusted_values[index])
        metrics["reject_null_corrected"] = bool(rejection_flags[index])
        metadata = dict(base_metadata)
        metadata["adjustment_applied"] = True
        metrics["multiple_testing_correction"] = metadata

    for horizon_key, metrics in corrected.items():
        if "multiple_testing_correction" in metrics:
            continue
        metadata = dict(base_metadata)
        metadata["adjustment_applied"] = False
        metrics["multiple_testing_correction"] = metadata
        metrics.setdefault("reject_null_corrected", False)

    return corrected


class SplitAwareScaler(TransformerMixin, BaseEstimator):
    """Wrap a scaler to enforce split-aware fitting and transformations."""

    def __init__(
        self,
        base_scaler: BaseEstimator,
        split_indices: Optional[Mapping[str, Sequence[int]]] = None,
    ) -> None:
        if not hasattr(base_scaler, "fit") or not hasattr(base_scaler, "transform"):
            raise TypeError("base_scaler must implement fit and transform methods")
        self.base_scaler = base_scaler
        self.split_indices: Optional[Dict[str, np.ndarray]] = None
        if split_indices is not None:
            self.split_indices = self.normalize_split_indices(split_indices)
        self._fitted: bool = False
        self._fitted_split: Optional[str] = None

    @staticmethod
    def normalize_split_indices(
        split_indices: Mapping[str, Sequence[int]]
    ) -> Dict[str, np.ndarray]:
        """Validate and normalize split metadata into integer index arrays."""

        if split_indices is None:
            raise ValueError("Split metadata must be provided")

        normalized: Dict[str, np.ndarray] = {}
        for split_name, indices in split_indices.items():
            if indices is None:
                raise ValueError(f"Split '{split_name}' has no indices")
            array = np.array(indices, dtype=int, copy=True)
            if array.ndim != 1:
                raise ValueError(
                    f"Split '{split_name}' indices must be 1-dimensional, got shape {array.shape}"
                )
            normalized[split_name] = array
        if not normalized:
            raise ValueError("At least one split must be provided")
        return normalized

    def set_split_indices(
        self, split_indices: Mapping[str, Sequence[int]]
    ) -> "SplitAwareScaler":
        """Assign split metadata after instantiation."""

        self.split_indices = self.normalize_split_indices(split_indices)
        return self

    def get_split_indices(self, split: str) -> np.ndarray:
        """Return indices for the requested split."""

        indices = self._require_split_indices().get(split)
        if indices is None:
            raise KeyError(f"Split '{split}' not found in metadata")
        return indices

    def fit(self, X: Any, y: Any = None, *, split: str = "train") -> "SplitAwareScaler":
        """Fit the underlying scaler using data from the specified split."""

        if split != "train":
            raise ValueError("SplitAwareScaler.fit may only be called with split='train'")
        indices = self.get_split_indices(split)
        X_subset = self._subset(X, indices)
        y_subset = self._subset(y, indices)
        self.base_scaler.fit(X_subset, y_subset)
        self._fitted = True
        self._fitted_split = split
        return self

    def transform(self, X: Any, *, split: str = "train") -> Any:
        """Transform the specified split with the fitted scaler."""

        if not self._fitted:
            raise RuntimeError("SplitAwareScaler must be fitted before calling transform")
        indices = self.get_split_indices(split)
        X_subset = self._subset(X, indices)
        return self.base_scaler.transform(X_subset)

    def fit_transform(self, X: Any, y: Any = None, *, split: str = "train") -> Any:
        """Fit and transform the training split."""

        if split != "train":
            raise ValueError("fit_transform is only permitted for the 'train' split")
        self.fit(X, y=y, split=split)
        return self.transform(X, split=split)

    @property
    def is_fitted(self) -> bool:
        """Whether the underlying scaler has been fitted."""

        return self._fitted

    def _require_split_indices(self) -> Dict[str, np.ndarray]:
        if self.split_indices is None:
            raise ValueError("Split metadata has not been provided")
        return self.split_indices

    @staticmethod
    def _subset(data: Any, indices: np.ndarray) -> Any:
        if data is None:
            return None
        if hasattr(data, "iloc"):
            return data.iloc[indices]
        array = np.asarray(data)
        return array[indices]

def _serialize_failure_cases(error: pa_errors.SchemaError, limit: int = 3) -> str:
    if hasattr(error, "failure_cases"):
        failure_cases = getattr(error, "failure_cases")
        if isinstance(failure_cases, pd.DataFrame) and not failure_cases.empty:
            head = failure_cases.head(limit)
            serialized = ", ".join(
                f"{row.get('check', 'check')}→{row.get('failure_case')}" for row in head.to_dict("records")
            )
            remaining = len(failure_cases) - len(head)
            if remaining > 0:
                serialized += f", … (+{remaining} more)"
            return serialized
    return str(error)


def format_schema_error(schema_key: str, context: str, error: pa_errors.SchemaError) -> str:
    registry_entry = SCHEMA_REGISTRY.get(schema_key, {"name": schema_key, "version": "unknown"})
    schema_name = registry_entry.get("name", schema_key)
    schema_version = registry_entry.get("version", "unknown")
    location = f" in {context}" if context else ""
    details = _serialize_failure_cases(error)
    return (
        f"Schema validation failed for {schema_name} v{schema_version}{location}: {details}. "
        "Ensure the dataframe has the required columns, dtypes, and datetime index."
    )


def _raise(schema_key: str, context: str, error: pa_errors.SchemaError) -> None:
    raise SchemaValidationException(schema_key=schema_key, context=context, original_error=error)


def _validate(
    schema: pa.DataFrameSchema,
    schema_key: str,
    df: pd.DataFrame,
    *,
    context: str,
    lazy: bool = False,
) -> pd.DataFrame:
    try:
        return schema.validate(df, lazy=lazy)
    except pa_errors.SchemaError as err:  # pragma: no cover - exercised in tests raising SchemaValidationException
        _raise(schema_key, context, err)
    return df


def validate_raw_ohlcv(df: pd.DataFrame, *, context: str, lazy: bool = False) -> pd.DataFrame:
    """Validate raw OHLCV input frames."""

    return _validate(RAW_OHLCV_SCHEMA, "raw_ohlcv", df, context=context, lazy=lazy)


def validate_labeled_dataset(df: pd.DataFrame, *, context: str, lazy: bool = False) -> pd.DataFrame:
    """Validate labeled dataset frames."""

    return _validate(LABELED_DATASET_SCHEMA, "labeled_dataset", df, context=context, lazy=lazy)


def validate_engineered_features(df: pd.DataFrame, *, context: str, lazy: bool = False) -> pd.DataFrame:
    """Validate engineered feature frames."""

    return _validate(ENGINEERED_FEATURE_SCHEMA, "engineered_features", df, context=context, lazy=lazy)


_Numeric = (int, float, np.integer, np.floating)


def _normalize_target_shifts(target_shifts: Optional[Mapping[str, Any]]) -> Tuple[Dict[str, int], int]:
    normalized: Dict[str, int] = {}
    if not target_shifts:
        return normalized, 1

    min_shift = None
    for raw_name, raw_value in target_shifts.items():
        name = str(raw_name)
        try:
            shift = int(raw_value)
        except (TypeError, ValueError):
            raise ValueError(f"Target '{name}' has non-integer shift {raw_value!r}") from None
        if shift < 1:
            raise ValueError(f"Target '{name}' has shift {shift} < 1; labels must be shifted forward at least one period")
        normalized[name] = shift
        if min_shift is None or shift < min_shift:
            min_shift = shift

    return normalized, (min_shift or 1)


def _count_leading_nulls(series: pd.Series) -> int:
    values = series.to_numpy()
    if values.size == 0:
        return 0
    is_null = pd.isna(values)
    if not is_null.any():
        return 0
    non_null_idx = np.flatnonzero(~is_null)
    if non_null_idx.size == 0:
        return int(values.size)
    return int(non_null_idx[0])


def _collect_reported_lags(metadata: Any) -> Tuple[Dict[str, int], Optional[int]]:
    if not metadata:
        return {}, None

    column_candidates: Dict[str, List[int]] = {}
    frame_level: List[int] = []

    def _walk(node: Any, path: List[str]) -> None:
        if isinstance(node, Mapping):
            raw_lag = node.get("max_lag")
            if isinstance(raw_lag, _Numeric) and not math.isnan(float(raw_lag)):
                lag = int(raw_lag)
                key = path[-1] if path else "__frame__"
                column_candidates.setdefault(key, []).append(lag)
            for key, value in node.items():
                if key == "max_lag":
                    continue
                _walk(value, path + [str(key)])
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            for idx, item in enumerate(node):
                _walk(item, path + [f"[{idx}]"])

    _walk(metadata, [])

    resolved: Dict[str, int] = {}
    for key, lags in column_candidates.items():
        if key == "__frame__":
            frame_level.append(int(max(lags)))
        else:
            resolved[key] = int(max(lags))

    return resolved, (max(frame_level) if frame_level else None)


def _candidate_column_keys(column: str) -> List[str]:
    candidates = {column}
    if "." in column:
        candidates.add(column.split(".")[-1])
    if "/" in column:
        candidates.add(column.split("/")[-1])
    if ":" in column:
        candidates.add(column.split(":")[-1])
    if "__" in column:
        candidates.add(column.split("__")[-1])
    if "_" in column:
        candidates.add(column.split("_", 1)[-1])
    return list(candidates)


def enforce_feature_temporal_alignment(
    features: pd.DataFrame,
    *,
    context: str = "",
    target_shifts: Optional[Mapping[str, Any]] = None,
    feature_metadata: Optional[Any] = None,
) -> Dict[str, Dict[str, int]]:
    """Ensure engineered features honour minimum lag requirements.

    Args:
        features: Feature dataframe to validate.
        context: Human readable context for error messages.
        target_shifts: Mapping of target names to their forward shifts.
        feature_metadata: Optional metadata describing feature lags.

    Returns:
        Mapping of column name to observed temporal alignment metadata.

    Raises:
        ValueError: When features expose contemporaneous data.
    """

    if features is None or features.empty:
        return {}

    _, min_shift = _normalize_target_shifts(target_shifts)
    reported_map, global_reported = _collect_reported_lags(feature_metadata)

    resolved_reported: Dict[str, int] = {}
    unmatched_global: List[int] = []
    for key, lag in reported_map.items():
        matched = False
        for candidate in _candidate_column_keys(key):
            if candidate in features.columns:
                resolved_reported[candidate] = max(resolved_reported.get(candidate, lag), lag)
                matched = True
        if not matched:
            unmatched_global.append(lag)

    if global_reported is not None:
        unmatched_global.append(global_reported)

    if unmatched_global:
        global_max = max(unmatched_global)
        if global_max < 1:
            location = f" ({context})" if context else ""
            raise ValueError(
                f"Feature metadata{location} reports max_lag {global_max} < 1; "
                "all features must be lagged by at least one period"
            )
    else:
        global_max = None

    metadata: Dict[str, Dict[str, int]] = {}
    location = f" in {context}" if context else ""
    required_msg = f"minimum lag >= 1 (target min shift {min_shift})"

    for column in features.columns:
        observed_lag = _count_leading_nulls(features[column])
        reported_lag = resolved_reported.get(column)
        if reported_lag is None and global_max is not None:
            reported_lag = global_max

        column_meta: Dict[str, int] = {"observed_lag": observed_lag}
        if reported_lag is not None:
            column_meta["reported_lag"] = reported_lag
        metadata[column] = column_meta

        effective_lag = max(observed_lag, reported_lag) if reported_lag is not None else observed_lag

        if reported_lag is not None and reported_lag < 1:
            raise ValueError(
                f"Feature '{column}'{location} metadata reports max_lag {reported_lag} < 1; "
                f"expected {required_msg}"
            )

        if effective_lag < 1:
            raise ValueError(
                f"Feature '{column}'{location} exposes contemporaneous values (lag={effective_lag}); "
                f"expected {required_msg}"
            )

    return metadata


def schema_metadata(*schema_keys: str) -> Dict[str, Dict[str, str]]:
    """Collect schema metadata entries for artifact recording."""

    metadata: Dict[str, Dict[str, str]] = {}
    for key in schema_keys:
        if key in SCHEMA_REGISTRY:
            metadata[key] = dict(SCHEMA_REGISTRY[key])
    return metadata


def report_hypothesis_count(
    config: "MultiHorizonConfig",
    *,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Summarise the hypothesis volume implied by the configuration.

    The total hypothesis count is derived from the Cartesian product of active
    horizons, declared transaction-cost scenarios, and configured regime
    variants. A Bonferroni-adjusted threshold is reported for convenience so
    downstream diagnostics can surface the stricter per-test significance
    level.
    """

    def _count_numeric_horizons(weights: Any) -> Tuple[int, List[str]]:
        if weights is None:
            return 0, []
        numeric_items = [
            (str(name), float(value))
            for name, value in vars(weights).items()
            if isinstance(value, (int, float))
        ]
        active = [name for name, value in numeric_items if value > 0]
        if active:
            return len(active), active
        return len(numeric_items), [name for name, _ in numeric_items]

    def _count_configurations(container: Any, attribute_names: Sequence[str]) -> Tuple[int, Optional[str]]:
        if container is None:
            return 0, None
        for attr in attribute_names:
            value = getattr(container, attr, None)
            if isinstance(value, Mapping):
                return len(value), attr
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return len(value), attr
        return 0, None

    horizon_count, active_horizons = _count_numeric_horizons(getattr(config, "horizon_weights", None))
    if horizon_count == 0:
        horizon_count = 1  # fall back to a single horizon assumption

    transaction_container = getattr(config, "transaction_costs", None)
    transaction_count, transaction_attr = _count_configurations(
        transaction_container,
        ("scenarios", "scenario_configs", "cost_scenarios", "scenario_grid", "configurations"),
    )
    if transaction_container is not None and transaction_count == 0:
        transaction_count = 1
    elif transaction_container is None:
        transaction_count = 1

    regime_count = 1
    regime_attr: Optional[str] = None
    if getattr(config, "enable_regime_aware_labeling", False):
        regime_container = getattr(config, "regime_config", None)
        computed_count, regime_attr = _count_configurations(
            regime_container,
            ("regimes", "regime_states", "regime_templates", "clusters", "configurations"),
        )
        if computed_count:
            regime_count = computed_count

    total_hypotheses = int(horizon_count * transaction_count * regime_count)
    bonferroni_threshold = alpha / total_hypotheses if total_hypotheses else alpha

    return {
        "horizon_count": int(horizon_count),
        "transaction_cost_scenarios": int(transaction_count),
        "regime_configurations": int(regime_count),
        "total_hypotheses": total_hypotheses,
        "alpha": alpha,
        "bonferroni_threshold": bonferroni_threshold,
        "details": {
            "active_horizons": active_horizons,
            "transaction_attribute": transaction_attr,
            "regime_attribute": regime_attr,
        },
    }


def _should_skip_object(obj: Any) -> bool:
    """Return ``True`` when an object should be skipped during recursion."""

    module = getattr(obj, "__module__", "")
    if module.startswith("pandas") or module.startswith("numpy"):
        return True
    return False


def extract_p_value_mapping(results: Any) -> Dict[str, float]:
    """Recursively extract p-values from nested selection results.

    The function walks dictionaries, dataclass ``__dict__`` objects, and
    sequences, collecting any values stored under ``p_value`` keys or nested
    ``p_values`` mappings.  The return value maps a dotted path (representing
    the traversal route) to the numeric p-value discovered at that location.
    Non-numeric or non-finite entries are ignored.
    """

    extracted: Dict[str, float] = {}
    visited: Set[int] = set()

    def _walk(obj: Any, path: str) -> None:
        if obj is None:
            return
        if _should_skip_object(obj):
            return

        if isinstance(obj, Mapping):
            obj_id = id(obj)
            if obj_id in visited:
                return
            visited.add(obj_id)
            for key, value in obj.items():
                key_str = str(key)
                key_lower = key_str.lower()
                next_path = f"{path}.{key_str}" if path else key_str

                if key_lower in {"p_value", "pvalue"}:
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        extracted[path or key_str] = float(value)
                    continue

                if key_lower in {"p_values", "pvalues"} and isinstance(value, Mapping):
                    for inner_key, inner_val in value.items():
                        if isinstance(inner_val, (int, float)) and math.isfinite(inner_val):
                            inner_path = f"{next_path}.{inner_key}"
                            extracted[inner_path] = float(inner_val)
                    continue

                _walk(value, next_path)
            return

        if isinstance(obj, (list, tuple, set)):
            obj_id = id(obj)
            if obj_id in visited:
                return
            visited.add(obj_id)
            for idx, item in enumerate(obj):
                next_path = f"{path}[{idx}]" if path else f"[{idx}]"
                _walk(item, next_path)
            return

        if hasattr(obj, "__dict__") and not _should_skip_object(obj):
            _walk(vars(obj), path)

    _walk(results, "")
    return extracted


def _bh_fdr_adjustment(p_values: Dict[str, float]) -> Dict[str, float]:
    """Apply Benjamini-Hochberg FDR correction to a mapping of p-values."""

    if not p_values:
        return {}

    items: Sequence[Tuple[str, float]] = tuple((key, float(value)) for key, value in p_values.items())
    sorted_indices = sorted(range(len(items)), key=lambda idx: items[idx][1])
    adjusted: Dict[str, float] = {}

    n = len(items)
    prev = 1.0
    for rank, index in enumerate(reversed(sorted_indices), start=1):
        key, value = items[index]
        corrected = min(prev, (value * n) / (n - rank + 1))
        corrected = float(max(0.0, min(1.0, corrected)))
        adjusted[key] = corrected
        prev = corrected

    # Preserve original order for readability
    return {key: adjusted.get(key, float(value)) for key, value in items}


def _normalise_p_values(results: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    """Normalise raw result mappings into string→float p-value mappings."""

    if results is None:
        return {}
    if isinstance(results, Mapping):
        flattened = extract_p_value_mapping(results)
        if flattened:
            return flattened
        normalised: Dict[str, float] = {}
        for key, value in results.items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                normalised[str(key)] = float(value)
        return normalised
    return {}


def track_and_control_hypotheses(
    horizon_results: Optional[Mapping[str, Any]] = None,
    feature_results: Optional[Mapping[str, Any]] = None,
    lookback_results: Optional[Mapping[str, Any]] = None,
    *,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Aggregate hypothesis counts and apply FDR correction to p-values.

    Parameters
    ----------
    horizon_results
        Mapping containing horizon-related hypothesis test results.
    feature_results
        Mapping containing feature-level hypothesis test results.
    lookback_results
        Mapping containing lookback optimisation hypothesis test results.
    alpha
        Significance level used when counting significant hypotheses.

    Returns
    -------
    Dict[str, Any]
        Structured report describing hypothesis volumes, adjusted p-values,
        significant counts before/after correction, and a warning message that
        callers should log for visibility.
    """

    corrected_horizon_metrics: Dict[str, Dict[str, Any]] = {}
    horizon_metrics_input: Optional[Mapping[str, Any]] = None
    if isinstance(horizon_results, Mapping):
        candidate_metrics = {
            str(key): dict(value)
            for key, value in horizon_results.items()
            if isinstance(value, Mapping) and "p_value" in value
        }
        if candidate_metrics:
            corrected_horizon_metrics = apply_multiple_testing_correction(
                candidate_metrics, alpha=alpha
            )
            horizon_metrics_input = candidate_metrics

    horizon_p = _normalise_p_values(horizon_metrics_input or horizon_results)
    feature_p = _normalise_p_values(feature_results)
    lookback_p = _normalise_p_values(lookback_results)

    breakdown = {
        "horizons": len(horizon_p),
        "features": len(feature_p),
        "lookbacks": len(lookback_p),
    }
    total_hypotheses = sum(breakdown.values())

    adjusted = {
        "horizons": _bh_fdr_adjustment(horizon_p),
        "features": _bh_fdr_adjustment(feature_p),
        "lookbacks": _bh_fdr_adjustment(lookback_p),
    }

    significant_before = {
        "horizons": sum(1 for value in horizon_p.values() if value < alpha),
        "features": sum(1 for value in feature_p.values() if value < alpha),
        "lookbacks": sum(1 for value in lookback_p.values() if value < alpha),
    }
    significant_after = {
        "horizons": sum(1 for value in adjusted["horizons"].values() if value < alpha),
        "features": sum(1 for value in adjusted["features"].values() if value < alpha),
        "lookbacks": sum(1 for value in adjusted["lookbacks"].values() if value < alpha),
    }
    significant_before["total"] = sum(significant_before.values())
    significant_after["total"] = sum(significant_after.values())

    warning: str = ""
    if total_hypotheses > 0:
        warning = (
            f"⚠️ Multiple testing detected across {total_hypotheses} hypotheses. "
            "Applied Benjamini–Hochberg FDR correction to control the false discovery rate."
        )
        if total_hypotheses > 100:
            warning = (
                f"⚠️ Multiple testing detected across {total_hypotheses} hypotheses (exceeds 100). "
                "Applied Benjamini–Hochberg FDR correction to control the false discovery rate."
            )

    report = {
        "hypothesis_breakdown": breakdown,
        "total_hypotheses": total_hypotheses,
        "raw_p_values": {
            "horizons": horizon_p,
            "features": feature_p,
            "lookbacks": lookback_p,
        },
        "adjusted_p_values": adjusted,
        "significant_counts": {
            "before": significant_before,
            "after": significant_after,
        },
        "warning": warning,
    }

    if corrected_horizon_metrics:
        report["corrected_horizon_metrics"] = corrected_horizon_metrics

    return report
_RNGInput = Union[int, np.random.Generator, np.random.RandomState]


def _build_validation_series(series: Union[pd.Series, Series, np.ndarray], index: pd.Index) -> Series:
    if isinstance(series, pd.Series):
        return series.reindex(index)
    return Series(series, index=index)


def _default_ic_scorer(model: Any, X: pd.DataFrame, y: Series) -> float:
    predictions = model.predict(X)
    prediction_series = _build_validation_series(predictions, X.index)
    target_series = _build_validation_series(y, X.index)
    aligned_predictions, aligned_targets = prediction_series.align(target_series, join="inner")
    if aligned_predictions.empty or aligned_targets.empty:
        return 0.0

    if calculate_information_coefficient is not None:
        try:
            return float(calculate_information_coefficient(aligned_predictions, aligned_targets))
        except Exception:
            pass

    ranked_predictions = aligned_predictions.rank(method="average")
    ranked_targets = aligned_targets.rank(method="average")
    correlation = ranked_predictions.corr(ranked_targets)
    if correlation is None or np.isnan(correlation):
        return 0.0
    return float(correlation)


def _get_rng(random_state: Optional[_RNGInput]) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, np.random.RandomState):  # pragma: no branch - compatibility guard
        seed = random_state.randint(0, 2**32 - 1)
        return np.random.default_rng(seed)
    return np.random.default_rng(random_state)


def block_permutation_importance(
    model: Any,
    X_val: pd.DataFrame,
    y_val: Series,
    *,
    block_size: int,
    n_repeats: int = 10,
    scoring_func: Optional[Callable[[Any, pd.DataFrame, Series], float]] = None,
    random_state: Optional[_RNGInput] = None,
) -> Series:
    """Compute block-wise permutation importance on validation data.

    The helper preserves temporal ordering inside each permuted block and aggregates
    repeated score drops into a feature-importance series.
    """

    if block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if n_repeats <= 0:
        raise ValueError("n_repeats must be a positive integer")
    if X_val.empty:
        return Series(dtype=float)

    scorer = scoring_func or _default_ic_scorer
    y_series = _build_validation_series(y_val, X_val.index)
    baseline_score = scorer(model, X_val, y_series)

    rng = _get_rng(random_state)
    importance_scores: Dict[str, float] = {}

    n_samples = len(X_val)
    block_starts = list(range(0, n_samples, block_size))
    block_slices = [slice(start, min(start + block_size, n_samples)) for start in block_starts]

    for column in X_val.columns:
        drops: list[float] = []
        column_values = X_val[column].to_numpy(copy=True)
        if len(column_values) == 0:
            importance_scores[column] = 0.0
            continue

        for _ in range(n_repeats):
            permuted_frame = X_val.copy()
            permutation_order = rng.permutation(len(block_slices))
            permuted_blocks = [column_values[block_slices[idx]] for idx in permutation_order]
            permuted_column = (
                np.concatenate(permuted_blocks)
                if permuted_blocks
                else column_values.copy()
            )
            permuted_frame[column] = permuted_column
            permuted_score = scorer(model, permuted_frame, y_series)
            drops.append(baseline_score - permuted_score)

        importance_scores[column] = float(np.mean(drops)) if drops else 0.0

    return Series(importance_scores).sort_values(ascending=False)

