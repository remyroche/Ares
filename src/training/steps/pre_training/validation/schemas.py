"""Shared Pandera schemas for pre-training data validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

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
    columns={},
    dtype=pa.Float64,
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

