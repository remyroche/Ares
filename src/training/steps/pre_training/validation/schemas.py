"""Shared Pandera schemas for pre-training data validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple

import math

import pandas as pd

try:
    import pandera as pa
    from pandera import errors as pa_errors
except ModuleNotFoundError:  # pragma: no cover - fallback for test environments without pandera
    from ._pandera_stub import pandera_stub as pa
    from ._pandera_stub import errors as pa_errors


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


def schema_metadata(*schema_keys: str) -> Dict[str, Dict[str, str]]:
    """Collect schema metadata entries for artifact recording."""

    metadata: Dict[str, Dict[str, str]] = {}
    for key in schema_keys:
        if key in SCHEMA_REGISTRY:
            metadata[key] = dict(SCHEMA_REGISTRY[key])
    return metadata


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

    horizon_p = _normalise_p_values(horizon_results)
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

    return report

