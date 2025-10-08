"""Data contract schemas and validators for pre-training pipeline artifacts."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

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
    validate_engineered_features,
    validate_labeled_dataset,
    validate_raw_ohlcv,
)


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


def _format_pydantic_errors(error: ValidationError) -> List[str]:  # pragma: no cover - exercised via tests
    messages: List[str] = []
    for err in error.errors():
        loc = ".".join(str(part) for part in err.get("loc", []) if part != "context")
        msg = err.get("msg", "invalid value")
        messages.append(f"{loc}: {msg}" if loc else msg)
    return messages


def _wrap_pandera_error(context: str, error: SchemaValidationException) -> DataContractValidationError:
    return DataContractValidationError(context, [str(error)])


def validate_multi_horizon_labeling_result(
    payload: Mapping[str, Any],
    *,
    context: str,
) -> Dict[str, Any]:
    """Validate the multi-horizon labeling artifact structure and contents."""

    try:
        model = LabeledDataSchema.model_validate({**payload, "context": context})
    except ValidationError as err:
        raise DataContractValidationError(context, _format_pydantic_errors(err)) from err

    validated_payload = dict(model.model_dump(exclude={"context"}))

    try:
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
                    raise DataContractValidationError(
                        context,
                        [
                            f"market_data_batches[{idx}] must be a pandas.DataFrame, got {type(batch).__name__}",
                        ],
                    )
            validated_payload["market_data_batches"] = validated_batches
    except SchemaValidationException as schema_error:
        raise _wrap_pandera_error(context, schema_error) from schema_error

    return validated_payload


def validate_feature_artifact(
    payload: Mapping[str, Any],
    *,
    context: str,
) -> Dict[str, Any]:
    """Validate the interactive feature generation artifact."""

    try:
        model = FeaturesSchema.model_validate({**payload, "context": context})
    except ValidationError as err:
        raise DataContractValidationError(context, _format_pydantic_errors(err)) from err

    validated_payload = dict(model.model_dump(exclude={"context"}))

    try:
        validated_payload["features"] = validate_engineered_features(
            model.features,
            context=f"{context}.features",
        )
        if model.interaction_features is not None:
            validated_payload["interaction_features"] = validate_engineered_features(
                model.interaction_features,
                context=f"{context}.interaction_features",
            )
        if model.cross_timeframe_features is not None:
            validated_payload["cross_timeframe_features"] = validate_engineered_features(
                model.cross_timeframe_features,
                context=f"{context}.cross_timeframe_features",
            )
    except SchemaValidationException as schema_error:
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
