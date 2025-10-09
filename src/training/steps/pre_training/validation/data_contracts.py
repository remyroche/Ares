"""Data contract schemas and validators for pre-training pipeline artifacts."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

# Import core utilities
try:
    from ...utils.tprint import tprint, tprint_debug, tprint_error, tprint_info, tprint_warning
    from ...utils.common_operations import (
        validate_dataframe, validate_positive, validate_range, safe_divide,
        timed_operation, format_bytes, get_dataframe_info, calculate_data_quality_metrics
    )
    from ...utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    # Import matrix operations for feature correlation analysis
    from ...utils.matrix_operations import (
        safe_correlation_matrix, matrix_correlation_analysis, optimize_dataframe,
        get_unified_matrix_operations, get_vectorized_processing_core, batch_matrix_multiply
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError as e:
    # Fallback imports if utils are not available
    MATRIX_OPERATIONS_AVAILABLE = False
    def tprint(*args, **kwargs): pass
    def tprint_debug(*args, **kwargs): pass
    def tprint_error(*args, **kwargs): pass
    def tprint_info(*args, **kwargs): pass
    def tprint_warning(*args, **kwargs): pass
    def validate_dataframe(df): return isinstance(df, pd.DataFrame) and not df.empty
    def validate_positive(value, name="value"): return value if value > 0 else 0.0
    def validate_range(value, min_val=None, max_val=None, name="value"): return value
    def safe_divide(a, b, default=0.0): return a / b if b != 0 else default
    def timed_operation(func): return func
    def format_bytes(bytes_value): return f"{bytes_value}B"
    def get_dataframe_info(df): return {}
    def calculate_data_quality_metrics(df): return {}
    def get_m1_memory_optimizer(): return None
    # Matrix operations fallbacks
    def safe_correlation_matrix(df): return df.corr() if hasattr(df, 'corr') else None
    def matrix_correlation_analysis(*args, **kwargs): return {}
    def optimize_dataframe(df): return df
    def get_unified_matrix_operations(): return None
    def get_vectorized_processing_core(): return None
    def batch_matrix_multiply(*args, **kwargs): return None

try:
    from pydantic import (
        BaseModel,
        ConfigDict,
        Field,
        ValidationInfo,
        ValidationError,
        field_validator,
    )
except ModuleNotFoundError:  # pragma: no cover - pydantic should be available in production
    BaseModel = object  # type: ignore
    ConfigDict = dict  # type: ignore
    Field = lambda *args, **kwargs: None  # type: ignore
    ValidationError = Exception  # type: ignore
    ValidationInfo = object  # type: ignore
    field_validator = lambda *args, **kwargs: (lambda func: func)  # type: ignore

from .schemas import (
    SchemaValidationException,
    enforce_feature_temporal_alignment,
    validate_engineered_features,
    validate_labeled_dataset,
    validate_raw_ohlcv,
)


def _normalize_shift_mapping(raw: Any) -> Dict[str, int]:
    if not isinstance(raw, Mapping):
        return {}
    normalized: Dict[str, int] = {}
    for key, value in raw.items():
        try:
            normalized[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _extract_target_shifts(payload: Mapping[str, Any]) -> Dict[str, int]:
    shifts: Dict[str, int] = {}
    if not isinstance(payload, Mapping):
        return shifts

    shifts.update(_normalize_shift_mapping(payload.get("target_shifts")))

    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        shifts.update(_normalize_shift_mapping(metadata.get("target_shifts")))

    target_params = payload.get("target_parameters")
    if isinstance(target_params, Mapping):
        for name, params in target_params.items():
            if isinstance(params, Mapping):
                if "target_shift" in params:
                    try:
                        shifts[str(name)] = int(params["target_shift"])
                    except (TypeError, ValueError):
                        continue

    return shifts


class DataContractValidationError(RuntimeError):
    """Raised when a payload fails to satisfy a data contract schema."""

    def __init__(self, context: str, errors: Iterable[str]):
        self.context = context
        self.errors = list(errors)
        message = self._build_message(context, self.errors)
        super().__init__(message)

    @staticmethod
    def _build_message(context: str, errors: Sequence[str]) -> str:
        joined = "; ".join(errors) if errors else "unknown validation failure"
        return f"Data contract validation failed for {context}: {joined}"


class _BaseContractModel(BaseModel):
    """Base model enabling pandas types and consistent config."""

    context: str = Field(default="", exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="ignore",
    )


class ValidationResultsSchema(_BaseContractModel):
    """Schema for validation results blocks embedded in artifacts."""

    is_valid: bool
    issues: List[str] = Field(default_factory=list)


class LabeledDataSchema(_BaseContractModel):
    """Schema describing multi-horizon labeling artifacts."""

    labeled_data: pd.DataFrame
    labels: pd.DataFrame
    confidence_scores: Optional[pd.DataFrame] = None
    eligibility_masks: Optional[pd.DataFrame] = None
    sigma_payoffs: Optional[pd.DataFrame] = None
    quality_scores: Dict[str, Any] = Field(default_factory=dict)
    horizon_weights: Dict[str, float] = Field(default_factory=dict)
    target_columns: Sequence[str] = Field(default_factory=list)
    normalization_factors: Dict[str, Any] = Field(default_factory=dict)
    execution_timing: Dict[str, Any] = Field(default_factory=dict)
    method: Optional[str] = None
    balancing_applied: Optional[bool] = None
    sample_weights: Optional[Any] = None
    validation_results: ValidationResultsSchema
    smoothing_settings: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    market_data: Optional[pd.DataFrame] = None
    market_data_batches: Optional[Sequence[Any]] = None

    @field_validator(
        "labeled_data",
        "labels",
        "confidence_scores",
        "eligibility_masks",
        "sigma_payoffs",
        "market_data",
        mode="before",
    )
    def _ensure_dataframe(cls, value: Any) -> Optional[pd.DataFrame]:
        if value is None:
            return None
        if not isinstance(value, pd.DataFrame):
            raise ValueError("expected pandas.DataFrame")
        return value

    @field_validator("target_columns", mode="before")
    def _ensure_target_columns(cls, value: Any) -> Sequence[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            return list(value)
        raise ValueError("target_columns must be a sequence of strings")

    @field_validator("labels", mode="after")
    def _labels_shape_alignment(cls, value: pd.DataFrame, info: ValidationInfo) -> pd.DataFrame:
        labeled = None
        if info and info.data:
            labeled = info.data.get("labeled_data")
        if isinstance(labeled, pd.DataFrame) and isinstance(value, pd.DataFrame):
            if labeled.shape != value.shape:
                raise ValueError("labeled_data and labels must have matching shape")
        return value


class FeaturesSchema(_BaseContractModel):
    """Schema describing interactive feature generation outputs."""

    features: pd.DataFrame
    feature_names: Sequence[str] = Field(default_factory=list)
    selected_features: Sequence[str] = Field(default_factory=list)
    interaction_features: Optional[pd.DataFrame] = None
    cross_timeframe_features: Optional[pd.DataFrame] = None
    execution_time: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    success: Optional[bool] = None
    error_message: Optional[str] = None

    @field_validator(
        "features",
        "interaction_features",
        "cross_timeframe_features",
        mode="before",
    )
    def _ensure_dataframe(cls, value: Any) -> Optional[pd.DataFrame]:
        if value is None:
            return None
        if not isinstance(value, pd.DataFrame):
            raise ValueError("expected pandas.DataFrame")
        return value

    @field_validator("feature_names", "selected_features", mode="before")
    def _ensure_string_sequence(cls, value: Any) -> Sequence[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            return list(value)
        raise ValueError("expected a sequence of strings")


class SelectionResultSchema(_BaseContractModel):
    """Schema describing final feature selection results."""

    final_features: Sequence[str]
    stage_1_features: Sequence[str] = Field(default_factory=list)
    stage_2_features: Sequence[str] = Field(default_factory=list)
    stage_3_features: Sequence[str] = Field(default_factory=list)
    feature_counts: Dict[str, int] = Field(default_factory=dict)
    stage_scores: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    selection_time: Optional[float] = None
    is_unsupervised: Optional[bool] = None
    hypothesis_report: Dict[str, Any] = Field(default_factory=dict)
    horizon_p_values: Dict[str, float] = Field(default_factory=dict)
    feature_p_values: Dict[str, float] = Field(default_factory=dict)
    lookback_p_values: Dict[str, float] = Field(default_factory=dict)
    adjusted_p_values: Dict[str, Dict[str, float]] = Field(default_factory=dict)

    @field_validator(
        "final_features",
        "stage_1_features",
        "stage_2_features",
        "stage_3_features",
        mode="before",
    )
    def _ensure_string_sequence(cls, value: Any) -> Sequence[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            return list(value)
        raise ValueError("expected a sequence of strings")

    @field_validator(
        "horizon_p_values",
        "feature_p_values",
        "lookback_p_values",
        mode="before",
    )
    def _ensure_float_mapping(cls, value: Any) -> Dict[str, float]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            cleaned: Dict[str, float] = {}
            for key, item in value.items():
                if not isinstance(item, (int, float)):
                    raise ValueError("p-values must be numeric")
                cleaned[str(key)] = float(item)
            return cleaned
        raise ValueError("expected mapping of p-values")

    @field_validator("adjusted_p_values", mode="before")
    def _ensure_nested_float_mapping(cls, value: Any) -> Dict[str, Dict[str, float]]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("adjusted_p_values must be a mapping")
        cleaned: Dict[str, Dict[str, float]] = {}
        for category, sub_mapping in value.items():
            if sub_mapping is None:
                cleaned[str(category)] = {}
                continue
            if not isinstance(sub_mapping, Mapping):
                raise ValueError("adjusted_p_values entries must be mappings")
            cleaned_category: Dict[str, float] = {}
            for key, item in sub_mapping.items():
                if not isinstance(item, (int, float)):
                    raise ValueError("adjusted p-values must be numeric")
                cleaned_category[str(key)] = float(item)
            cleaned[str(category)] = cleaned_category
        return cleaned

    @field_validator("hypothesis_report", mode="before")
    def _ensure_hypothesis_report(cls, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("hypothesis_report must be a mapping")
        report = dict(value)
        total = report.get("total_hypotheses")
        warning = report.get("warning")
        if isinstance(total, (int, float)) and total > 100 and not warning:
            report["warning"] = (
                f"⚠️ Multiple testing detected across {int(total)} hypotheses (exceeds 100)."
            )
        return report


def _format_pydantic_errors(error: ValidationError) -> List[str]:  # pragma: no cover - exercised via tests
    messages: List[str] = []
    for err in error.errors():
        loc = ".".join(str(part) for part in err.get("loc", []) if part != "context")
        msg = err.get("msg", "invalid value")
        messages.append(f"{loc}: {msg}" if loc else msg)
    return messages


def _wrap_pandera_error(context: str, error: SchemaValidationException) -> DataContractValidationError:
    return DataContractValidationError(context, [str(error)])


@timed_operation
def validate_multi_horizon_labeling_result(
    payload: Mapping[str, Any],
    *,
    context: str,
) -> Dict[str, Any]:
    """Validate the multi-horizon labeling artifact structure and contents."""

    tprint_info(f"Validating multi-horizon labeling result for {context}")

    target_shifts = _extract_target_shifts(payload)
    tprint_debug(f"Extracted target shifts: {target_shifts}")

    # Validate payload structure
    if not isinstance(payload, Mapping):
        tprint_error(f"Payload must be a mapping, got {type(payload)}")
        raise DataContractValidationError(context, ["Payload must be a mapping"])

    # Check data quality metrics
    memory_optimizer = get_m1_memory_optimizer()
    if memory_optimizer:
        memory_optimizer.start_monitoring()

    # Initialize matrix operations for correlation analysis
    matrix_ops = get_unified_matrix_operations() if MATRIX_OPERATIONS_AVAILABLE else None
    vectorized_core = get_vectorized_processing_core() if MATRIX_OPERATIONS_AVAILABLE else None

    tprint_debug(f"Matrix operations available for data contracts: {MATRIX_OPERATIONS_AVAILABLE}")

    try:
        model = LabeledDataSchema.model_validate({**payload, "context": context})
    except ValidationError as err:
        tprint_error(f"Pydantic validation failed for {context}")
        raise DataContractValidationError(context, _format_pydantic_errors(err)) from err

    validated_payload = dict(model.model_dump(exclude={"context"}))
    tprint_debug(f"Validated payload structure for {context}")

    try:
        # Validate labeled data with quality checks and correlation analysis
        if validate_dataframe(model.labeled_data):
            quality_metrics = calculate_data_quality_metrics(model.labeled_data)
            tprint_debug(f"Labeled data quality: {quality_metrics}")

            # Use matrix operations for correlation analysis if available
            if matrix_ops and len(model.labeled_data.columns) > 1:
                try:
                    corr_matrix = safe_correlation_matrix(model.labeled_data)
                    if corr_matrix is not None:
                        correlation_analysis = matrix_correlation_analysis(
                            model.labeled_data.values, method='correlation'
                        )
                        tprint_debug(f"Feature correlation analysis completed for {len(model.labeled_data.columns)} features")
                        validated_payload["correlation_analysis"] = correlation_analysis
                except Exception as e:
                    tprint_debug(f"Correlation analysis failed: {e}")

        validated_payload["labeled_data"] = validate_labeled_dataset(
            model.labeled_data,
            context=f"{context}.labeled_data",
        )
        validated_payload["labels"] = validate_labeled_dataset(
            model.labels,
            context=f"{context}.labels",
        )

        if model.confidence_scores is not None:
            validated_payload["confidence_scores"] = validate_engineered_features(
                model.confidence_scores,
                context=f"{context}.confidence_scores",
            )
            enforce_feature_temporal_alignment(
                validated_payload["confidence_scores"],
                context=f"{context}.confidence_scores",
                target_shifts=target_shifts,
                feature_metadata=payload.get("confidence_scores_metadata")
                or payload.get("feature_metadata"),
            )

        if model.market_data is not None:
            validated_payload["market_data"] = validate_raw_ohlcv(
                model.market_data,
                context=f"{context}.market_data",
            )

        if model.market_data_batches is not None:
            validated_batches = []
            for idx, batch in enumerate(model.market_data_batches):
                if isinstance(batch, pd.DataFrame):
                    validated_batches.append(
                        validate_raw_ohlcv(
                            batch,
                            context=f"{context}.market_data_batches[{idx}]",
                        )
                    )
                else:
                    tprint_error(f"Invalid batch type at index {idx}: {type(batch)}")
                    raise DataContractValidationError(
                        context,
                        [
                            f"market_data_batches[{idx}] must be a pandas.DataFrame, got {type(batch).__name__}",
                        ],
                    )
            validated_payload["market_data_batches"] = validated_batches

        tprint_info(f"Successfully validated multi-horizon labeling result for {context}")
    except SchemaValidationException as schema_error:
        tprint_error(f"Schema validation failed for {context}: {schema_error}")
        raise _wrap_pandera_error(context, schema_error) from schema_error

    return validated_payload


@timed_operation
def validate_feature_artifact(
    payload: Mapping[str, Any],
    *,
    context: str,
) -> Dict[str, Any]:
    """Validate the interactive feature generation artifact."""

    tprint_info(f"Validating feature artifact for {context}")
    
    # FIXED: Initialize matrix operations variables for feature correlation analysis
    matrix_ops = get_unified_matrix_operations() if MATRIX_OPERATIONS_AVAILABLE else None
    vectorized_core = get_vectorized_processing_core() if MATRIX_OPERATIONS_AVAILABLE else None

    target_shifts = _extract_target_shifts(payload)
    feature_metadata = payload.get("feature_metadata")
    interaction_metadata = payload.get("interaction_feature_metadata")
    cross_timeframe_metadata = payload.get("cross_timeframe_feature_metadata")

    tprint_debug(f"Feature metadata keys: {list(feature_metadata.keys()) if feature_metadata else 'None'}")

    # Validate payload structure
    if not isinstance(payload, Mapping):
        tprint_error(f"Payload must be a mapping, got {type(payload)}")
        raise DataContractValidationError(context, ["Payload must be a mapping"])

    try:
        model = FeaturesSchema.model_validate({**payload, "context": context})
    except ValidationError as err:
        tprint_error(f"Pydantic validation failed for {context}")
        raise DataContractValidationError(context, _format_pydantic_errors(err)) from err

    validated_payload = dict(model.model_dump(exclude={"context"}))
    tprint_debug(f"Validated feature payload structure for {context}")

    try:
        # Validate main features with quality checks and correlation analysis
        if validate_dataframe(model.features):
            quality_metrics = calculate_data_quality_metrics(model.features)
            tprint_debug(f"Features quality: {quality_metrics}")
            info = get_dataframe_info(model.features)
            tprint_debug(f"Features info: shape={info.get('shape', 'unknown')}, memory={format_bytes(info.get('memory_usage', 0))}")

            # Use matrix operations for feature correlation analysis
            if matrix_ops and len(model.features.columns) > 1:
                try:
                    # Optimize DataFrame using vectorized core first
                    if vectorized_core:
                        model.features = optimize_dataframe(model.features)

                    corr_matrix = safe_correlation_matrix(model.features)
                    if corr_matrix is not None:
                        correlation_analysis = matrix_correlation_analysis(
                            model.features.values, method='correlation'
                        )
                        tprint_debug(f"Feature correlation analysis completed for {len(model.features.columns)} features")
                        validated_payload["feature_correlation_analysis"] = correlation_analysis
                except Exception as e:
                    tprint_debug(f"Feature correlation analysis failed: {e}")

        validated_payload["features"] = validate_engineered_features(
            model.features,
            context=f"{context}.features",
        )
        enforce_feature_temporal_alignment(
            validated_payload["features"],
            context=f"{context}.features",
            target_shifts=target_shifts,
            feature_metadata=feature_metadata,
        )

        if model.interaction_features is not None:
            validated_payload["interaction_features"] = validate_engineered_features(
                model.interaction_features,
                context=f"{context}.interaction_features",
            )
            enforce_feature_temporal_alignment(
                validated_payload["interaction_features"],
                context=f"{context}.interaction_features",
                target_shifts=target_shifts,
                feature_metadata=interaction_metadata,
            )

        if model.cross_timeframe_features is not None:
            validated_payload["cross_timeframe_features"] = validate_engineered_features(
                model.cross_timeframe_features,
                context=f"{context}.cross_timeframe_features",
            )
            enforce_feature_temporal_alignment(
                validated_payload["cross_timeframe_features"],
                context=f"{context}.cross_timeframe_features",
                target_shifts=target_shifts,
                feature_metadata=cross_timeframe_metadata,
            )

        tprint_info(f"Successfully validated feature artifact for {context}")
    except SchemaValidationException as schema_error:
        tprint_error(f"Schema validation failed for {context}: {schema_error}")
        raise _wrap_pandera_error(context, schema_error) from schema_error

    return validated_payload


def validate_selection_artifact(
    payload: Mapping[str, Any],
    *,
    context: str,
) -> Dict[str, Any]:
    """Validate the final feature selection result payload."""

    try:
        model = SelectionResultSchema.model_validate({**payload, "context": context})
    except ValidationError as err:
        raise DataContractValidationError(context, _format_pydantic_errors(err)) from err

    return dict(model.model_dump(exclude={"context"}))


__all__ = [
    "DataContractValidationError",
    "LabeledDataSchema",
    "FeaturesSchema",
    "SelectionResultSchema",
    "validate_multi_horizon_labeling_result",
    "validate_feature_artifact",
    "validate_selection_artifact",
]
