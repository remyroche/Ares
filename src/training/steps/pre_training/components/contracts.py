"""Typed contracts for pre-training pipeline components."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field, fields
from functools import lru_cache
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Tuple, Type, TypeVar, Union

from src.training.steps.pre_training.validation.data_contracts import (
    validate_feature_artifact,
    validate_multi_horizon_labeling_result,
    validate_selection_artifact,
)


def _copy_mapping(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return a shallow dictionary copy from an optional mapping."""

    if not value:
        return {}
    return dict(value)


@lru_cache(None)
def _field_names(cls: Type[Any]) -> Tuple[str, ...]:
    """Return dataclass field names excluding the internal ``extra`` field."""

    return tuple(f.name for f in fields(cls) if f.init and f.name != "extra")


class _MappingBackedDataclass(MutableMapping[str, Any]):
    """Mixin providing mapping-like behaviour for dataclass payloads."""

    extra: Dict[str, Any]

    def __post_init__(self) -> None:  # pragma: no cover - defensive normalisation
        if getattr(self, "extra", None) is None:
            object.__setattr__(self, "extra", {})

    # -- MutableMapping protocol -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        if key in _field_names(type(self)):
            return getattr(self, key)
        return self.extra[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key in _field_names(type(self)):
            setattr(self, key, value)
        else:
            self.extra[key] = value

    def __delitem__(self, key: str) -> None:
        if key in _field_names(type(self)):
            setattr(self, key, None)
        else:
            del self.extra[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.as_payload())

    def __len__(self) -> int:
        return len(self.as_payload())

    # -- Dictionary compatibility helpers ---------------------------------------
    def as_payload(self) -> Dict[str, Any]:
        """Return the payload as a plain dictionary."""

        payload: Dict[str, Any] = {}
        for name in _field_names(type(self)):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        payload.update(self.extra)
        return payload

    def get(self, key: str, default: Any = None) -> Any:  # noqa: D401 - dict compatible
        if key in _field_names(type(self)):
            value = getattr(self, key)
            return value if value is not None else default
        return self.extra.get(key, default)

    def setdefault(self, key: str, default: Any = None) -> Any:  # noqa: D401
        if key in _field_names(type(self)):
            current = getattr(self, key)
            if current is None:
                setattr(self, key, default)
                return default
            return current
        return self.extra.setdefault(key, default)

    def update(self, other: Optional[Mapping[str, Any]] = None, /, **kwargs: Any) -> None:  # noqa: D401
        for key, value in dict(other or {}, **kwargs).items():
            self[key] = value

    def copy(self) -> Dict[str, Any]:  # noqa: D401
        return self.as_payload()


@dataclass
class ArtifactBundle(_MappingBackedDataclass):
    """Base class for typed artifact payloads."""

    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenericArtifacts(ArtifactBundle):
    """Fallback artifact bundle when no specific contract is provided."""


@dataclass
class MultiHorizonArtifacts(ArtifactBundle):
    """Artifacts produced by the multi-horizon profit labeler."""

    multi_horizon_labeling_result: Dict[str, Any] = field(default_factory=dict)
    labeling_report: Dict[str, Any] = field(default_factory=dict)
    standardized_output: Optional[Dict[str, Any]] = None
    validated_schemas: Optional[Dict[str, Any]] = None


@dataclass
class FinalFeatureSelectionArtifacts(ArtifactBundle):
    """Artifacts produced by the final feature selection component."""

    final_feature_selection_result: Dict[str, Any] = field(default_factory=dict)
    validated_schemas: Optional[Dict[str, Any]] = None


@dataclass
class InteractiveFeatureArtifacts(ArtifactBundle):
    """Artifacts produced by the interactive feature generation component."""

    interactive_feature_generation_result: Dict[str, Any] = field(default_factory=dict)
    stage_results: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    validated_schemas: Optional[Dict[str, Any]] = None


@dataclass
class FeatureLookbackArtifacts(ArtifactBundle):
    """Artifacts produced by the feature lookback optimization component."""

    feature_lookback_optimization_summary: Dict[str, Any] = field(default_factory=dict)
    feature_lookback_optimization_result: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineState(_MappingBackedDataclass):
    """Typed representation of the mutable pipeline state passed between components."""

    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    timeframe: str = "15m"
    data_dir: Optional[str] = None
    outcomes_dir: Optional[str] = None
    data_locator: Optional[Any] = None
    data_dir_key: Optional[str] = None
    outcomes_dir_key: Optional[str] = None
    random_seed: Optional[int] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    custom_params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:  # pragma: no cover - normalisation guard
        super().__post_init__()
        object.__setattr__(self, "artifacts", _copy_mapping(self.artifacts))
        object.__setattr__(self, "custom_params", _copy_mapping(self.custom_params))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))

    @classmethod
    def from_mapping(cls, mapping: Optional[Mapping[str, Any]]) -> "PipelineState":
        """Construct a pipeline state from an arbitrary mapping."""

        if isinstance(mapping, cls):
            return mapping

        source: Dict[str, Any] = dict(mapping or {})
        extras = {k: v for k, v in source.items() if k not in _field_names(cls)}
        init_kwargs = {k: source[k] for k in source.keys() if k in _field_names(cls)}
        state = cls(**init_kwargs)
        state.extra.update(extras)
        return state

    @classmethod
    def ensure(cls, value: Optional[Union["PipelineState", Mapping[str, Any]]]) -> "PipelineState":
        """Return a :class:`PipelineState` instance for the supplied value."""

        if isinstance(value, cls):
            return value
        return cls.from_mapping(value)

    def to_dict(self) -> Dict[str, Any]:
        """Return the state as a plain dictionary."""

        return self.as_payload()


ArtifactsT = TypeVar("ArtifactsT", bound=ArtifactBundle)


Validator = Callable[[ArtifactBundle], ArtifactBundle]


def _validate_multi_horizon(bundle: MultiHorizonArtifacts) -> MultiHorizonArtifacts:
    if bundle.multi_horizon_labeling_result:
        bundle.multi_horizon_labeling_result = validate_multi_horizon_labeling_result(
            bundle.multi_horizon_labeling_result,
            context="components.multi_horizon_labeling_result",
        )
    return bundle


def _validate_interactive_features(
    bundle: InteractiveFeatureArtifacts,
) -> InteractiveFeatureArtifacts:
    if bundle.interactive_feature_generation_result:
        bundle.interactive_feature_generation_result = validate_feature_artifact(
            bundle.interactive_feature_generation_result,
            context="components.interactive_feature_generation_result",
        )
    return bundle


def _validate_final_selection(
    bundle: FinalFeatureSelectionArtifacts,
) -> FinalFeatureSelectionArtifacts:
    if bundle.final_feature_selection_result:
        bundle.final_feature_selection_result = validate_selection_artifact(
            bundle.final_feature_selection_result,
            context="components.final_feature_selection_result",
        )
    return bundle


_ARTIFACT_VALIDATORS: Dict[Type[ArtifactBundle], Validator] = {
    MultiHorizonArtifacts: _validate_multi_horizon,
    InteractiveFeatureArtifacts: _validate_interactive_features,
    FinalFeatureSelectionArtifacts: _validate_final_selection,
}


def validate_artifact_bundle(bundle: ArtifactBundle) -> ArtifactBundle:
    """Validate a typed artifact bundle against its registered contract."""

    validator = _ARTIFACT_VALIDATORS.get(type(bundle))
    if validator is None:
        return bundle
    return validator(bundle)


__all__ = [
    "ArtifactBundle",
    "ArtifactsT",
    "FeatureLookbackArtifacts",
    "FinalFeatureSelectionArtifacts",
    "GenericArtifacts",
    "InteractiveFeatureArtifacts",
    "MultiHorizonArtifacts",
    "PipelineState",
    "validate_artifact_bundle",
]

