"""Shared Pandera schemas for pre-training data validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import pandera as pa
    from pandera import errors as pa_errors
except ModuleNotFoundError:  # pragma: no cover - fallback for test environments without pandera
    from ._pandera_stub import pandera_stub as pa
    from ._pandera_stub import errors as pa_errors

try:
    from statsmodels.stats.multitest import multipletests
except ModuleNotFoundError:  # pragma: no cover - optional dependency for multiple testing adjustments
    multipletests = None  # type: ignore[assignment]


__all__ = [
    "RAW_OHLCV_SCHEMA",
    "LABELED_DATASET_SCHEMA",
    "ENGINEERED_FEATURE_SCHEMA",
    "SCHEMA_REGISTRY",
    "SchemaValidationException",
    "validate_raw_ohlcv",
    "validate_labeled_dataset",
    "validate_engineered_features",
    "schema_metadata",
    "HypothesisTracker",
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

