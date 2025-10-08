"""Typed contracts for pre-training pipeline components."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, TypedDict, Union, cast
import copy

class MultiHorizonArtifacts(TypedDict, total=False):
    """Artifacts produced by the multi-horizon profit labeler."""

    multi_horizon_labeling_result: Dict[str, Any]
    labeling_report: Dict[str, Any]
    standardized_output: Dict[str, Any]


class FeatureLookbackArtifacts(TypedDict, total=False):
    """Artifacts produced by the feature lookback optimization component."""

    feature_lookback_optimization_result: Dict[str, Any]
    feature_lookback_optimization_summary: Dict[str, Any]


class InteractiveFeatureArtifacts(TypedDict, total=False):
    """Artifacts produced by the interactive feature generation component."""

    interactive_feature_generation_result: Dict[str, Any]
    stage_results: Any
    performance_metrics: Any
    artifacts: Any


class FinalSelectionArtifacts(TypedDict, total=False):
    """Artifacts produced by the final feature selection component."""

    final_feature_selection_result: Dict[str, Any]


ComponentArtifacts = Union[
    Dict[str, Any],
    MultiHorizonArtifacts,
    FeatureLookbackArtifacts,
    InteractiveFeatureArtifacts,
    FinalSelectionArtifacts,
]


_MULTI_H_KEYS = {
    "multi_horizon_labeling_result",
    "labeling_report",
    "standardized_output",
}
_FLO_KEYS = {
    "feature_lookback_optimization_result",
    "feature_lookback_optimization_summary",
}
_INTERACTIVE_KEYS = {
    "interactive_feature_generation_result",
    "stage_results",
    "performance_metrics",
    "artifacts",
}
_FINAL_SELECTION_KEYS = {"final_feature_selection_result"}


_ALLOWED_PIPELINE_KEYS = {
    "symbol",
    "exchange",
    "timeframe",
    "data_dir",
    "custom_params",
    "quality_thresholds",
    "regime_cache_path",
    "regime_data_splitting_result",
    "model_type",
}
_ALLOWED_PIPELINE_ARTIFACT_KEYS = (
    _MULTI_H_KEYS
    | _FLO_KEYS
    | _INTERACTIVE_KEYS
    | _FINAL_SELECTION_KEYS
)


class PipelineStateMapping(Mapping[str, Any]):
    """Mapping interface for :class:`PipelineState`."""

    def __init__(self, state: "PipelineState") -> None:
        self._state = state

    def __getitem__(self, key: str) -> Any:  # pragma: no cover - delegated
        return self._state._get_item(key)

    def __iter__(self) -> Iterator[str]:  # pragma: no cover - delegated
        return iter(self._state.keys())

    def __len__(self) -> int:  # pragma: no cover - delegated
        return len(list(self._state.keys()))


@dataclass(frozen=True)
class PipelineState:
    """Typed representation of the pre-training pipeline state.

    The pipeline state exposes core execution context fields in addition to the
    artifacts emitted by individual components.  Consumers may access the data
    either via attributes (``state.symbol``) or mapping style access
    (``state['symbol']``).  Unknown keys raise :class:`KeyError` to prevent
    silent schema drift.
    """

    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    timeframe: str = "1h"
    data_dir: str = "historical_data"
    custom_params: Dict[str, Any] = field(default_factory=dict)
    quality_thresholds: Dict[str, float] = field(default_factory=dict)
    regime_cache_path: Optional[str] = None
    regime_data_splitting_result: Optional[Dict[str, Any]] = None
    model_type: Optional[str] = None
    multi_horizon: Optional[MultiHorizonArtifacts] = None
    feature_lookback: Optional[FeatureLookbackArtifacts] = None
    interactive: Optional[InteractiveFeatureArtifacts] = None
    final_selection: Optional[FinalSelectionArtifacts] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _ensure_str(self.symbol, "symbol"))
        object.__setattr__(self, "exchange", _ensure_str(self.exchange, "exchange"))
        object.__setattr__(self, "timeframe", _ensure_str(self.timeframe, "timeframe"))
        object.__setattr__(self, "data_dir", _ensure_str(self.data_dir, "data_dir"))

        custom_params = dict(self.custom_params or {})
        object.__setattr__(self, "custom_params", custom_params)

        quality_thresholds = {
            str(key): float(value)
            for key, value in (self.quality_thresholds or {}).items()
        }
        object.__setattr__(self, "quality_thresholds", quality_thresholds)

        if self.regime_cache_path is not None:
            object.__setattr__(
                self,
                "regime_cache_path",
                _ensure_str(self.regime_cache_path, "regime_cache_path"),
            )

        if self.model_type is not None and not isinstance(self.model_type, str):
            raise ValueError("PipelineState field 'model_type' must be a string when provided")

        if self.multi_horizon is not None:
            object.__setattr__(
                self,
                "multi_horizon",
                validate_multi_horizon_artifacts(dict(self.multi_horizon)),
            )
        if self.feature_lookback is not None:
            object.__setattr__(
                self,
                "feature_lookback",
                validate_feature_lookback_artifacts(dict(self.feature_lookback)),
            )
        if self.interactive is not None:
            object.__setattr__(
                self,
                "interactive",
                validate_interactive_artifacts(dict(self.interactive)),
            )
        if self.final_selection is not None:
            object.__setattr__(
                self,
                "final_selection",
                validate_final_selection_artifacts(dict(self.final_selection)),
            )

    # Mapping helpers -------------------------------------------------
    def as_mapping(self) -> Mapping[str, Any]:
        return PipelineStateMapping(self)

    def _get_item(self, key: str) -> Any:
        if key not in PIPELINE_STATE_KEYS:
            raise KeyError(f"Unknown pipeline state key: {key}")

        base = self._base_dict()
        if key in base:
            return base[key]
        if self.multi_horizon and key in self.multi_horizon:
            return self.multi_horizon[key]  # type: ignore[index]
        if self.feature_lookback and key in self.feature_lookback:
            return self.feature_lookback[key]  # type: ignore[index]
        if self.interactive and key in self.interactive:
            return self.interactive[key]  # type: ignore[index]
        if self.final_selection and key in self.final_selection:
            return self.final_selection[key]  # type: ignore[index]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self._get_item(key)
        except KeyError:
            if key in PIPELINE_STATE_KEYS:
                return default
            raise

    def __getitem__(self, key: str) -> Any:
        return self._get_item(key)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if key not in PIPELINE_STATE_KEYS:
            return False
        try:
            self._get_item(key)
            return True
        except KeyError:
            return False

    def keys(self) -> Iterable[str]:
        yield from self._base_dict().keys()
        if self.multi_horizon:
            yield from self.multi_horizon.keys()
        if self.feature_lookback:
            yield from self.feature_lookback.keys()
        if self.interactive:
            yield from self.interactive.keys()
        if self.final_selection:
            yield from self.final_selection.keys()

    def items(self) -> Iterable[tuple[str, Any]]:
        for key in self.keys():
            yield key, self[key]

    # Copy helpers ----------------------------------------------------
    def copy(self) -> "PipelineState":
        return PipelineState(**self._copy_data())

    def copy_with(self, **kwargs: Any) -> "PipelineState":
        data = self._copy_data()
        data.update(kwargs)
        return PipelineState(**data)

    def with_multi_horizon(self, artifacts: MultiHorizonArtifacts) -> "PipelineState":
        return self.copy_with(multi_horizon=validate_multi_horizon_artifacts(dict(artifacts)))

    def with_feature_lookback(self, artifacts: FeatureLookbackArtifacts) -> "PipelineState":
        return self.copy_with(feature_lookback=validate_feature_lookback_artifacts(dict(artifacts)))

    def with_interactive(self, artifacts: InteractiveFeatureArtifacts) -> "PipelineState":
        return self.copy_with(interactive=validate_interactive_artifacts(dict(artifacts)))

    def with_final_selection(self, artifacts: FinalSelectionArtifacts) -> "PipelineState":
        return self.copy_with(final_selection=validate_final_selection_artifacts(dict(artifacts)))

    def with_regime_split(self, regime_split: Optional[Dict[str, Any]]) -> "PipelineState":
        return self.copy_with(regime_data_splitting_result=regime_split)

    def with_custom_params(self, params: Mapping[str, Any]) -> "PipelineState":
        return self.copy_with(custom_params=dict(params))

    def with_quality_thresholds(self, thresholds: Mapping[str, Any]) -> "PipelineState":
        return self.copy_with(quality_thresholds={str(k): float(v) for k, v in thresholds.items()})

    def _base_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "timeframe": self.timeframe,
            "data_dir": self.data_dir,
            "custom_params": self.custom_params,
            "quality_thresholds": self.quality_thresholds,
            "regime_cache_path": self.regime_cache_path,
            "regime_data_splitting_result": self.regime_data_splitting_result,
            "model_type": self.model_type,
        }

    def _copy_data(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "timeframe": self.timeframe,
            "data_dir": self.data_dir,
            "custom_params": copy.deepcopy(self.custom_params),
            "quality_thresholds": copy.deepcopy(self.quality_thresholds),
            "regime_cache_path": self.regime_cache_path,
            "regime_data_splitting_result": copy.deepcopy(self.regime_data_splitting_result),
            "model_type": self.model_type,
            "multi_horizon": copy.deepcopy(self.multi_horizon),
            "feature_lookback": copy.deepcopy(self.feature_lookback),
            "interactive": copy.deepcopy(self.interactive),
            "final_selection": copy.deepcopy(self.final_selection),
        }


PIPELINE_STATE_KEYS = _ALLOWED_PIPELINE_KEYS | _ALLOWED_PIPELINE_ARTIFACT_KEYS


def _ensure_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"PipelineState field '{field_name}' must be a non-empty string")
    return value


def _validate_artifact_keys(artifacts: Mapping[str, Any], allowed: Iterable[str], *, require_any: bool = False) -> Dict[str, Any]:
    data = dict(artifacts)
    unknown = set(data.keys()) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown artifact keys: {sorted(unknown)}")
    if require_any and not data:
        raise ValueError("Artifacts payload must not be empty")
    return data


def validate_multi_horizon_artifacts(artifacts: Mapping[str, Any]) -> MultiHorizonArtifacts:
    data = _validate_artifact_keys(artifacts, _MULTI_H_KEYS)
    if "multi_horizon_labeling_result" not in data:
        raise ValueError("multi_horizon_labeling_result artifact is required")
    return cast(MultiHorizonArtifacts, data)


def validate_feature_lookback_artifacts(artifacts: Mapping[str, Any]) -> FeatureLookbackArtifacts:
    data = _validate_artifact_keys(artifacts, _FLO_KEYS)
    if "feature_lookback_optimization_result" not in data:
        raise ValueError("feature_lookback_optimization_result artifact is required")
    return cast(FeatureLookbackArtifacts, data)


def validate_interactive_artifacts(artifacts: Mapping[str, Any]) -> InteractiveFeatureArtifacts:
    data = _validate_artifact_keys(artifacts, _INTERACTIVE_KEYS)
    if "interactive_feature_generation_result" not in data:
        raise ValueError("interactive_feature_generation_result artifact is required")
    return cast(InteractiveFeatureArtifacts, data)


def validate_final_selection_artifacts(artifacts: Mapping[str, Any]) -> FinalSelectionArtifacts:
    data = _validate_artifact_keys(artifacts, _FINAL_SELECTION_KEYS)
    if "final_feature_selection_result" not in data:
        raise ValueError("final_feature_selection_result artifact is required")
    return cast(FinalSelectionArtifacts, data)


def validate_component_artifacts(artifacts: Mapping[str, Any]) -> Dict[str, Any]:
    data = dict(artifacts)
    for key in data.keys():
        if not isinstance(key, str):
            raise ValueError("Artifact keys must be strings")
    return data


def pipeline_state_from_mapping(mapping: Mapping[str, Any]) -> PipelineState:
    data = dict(mapping)
    unknown = set(data.keys()) - (PIPELINE_STATE_KEYS)
    if unknown:
        raise ValueError(f"Unknown pipeline state keys: {sorted(unknown)}")

    base_kwargs: Dict[str, Any] = {
        "symbol": data.get("symbol", "ETHUSDT"),
        "exchange": data.get("exchange", "binance"),
        "timeframe": data.get("timeframe", "1h"),
        "data_dir": data.get("data_dir", "historical_data"),
        "custom_params": dict(data.get("custom_params", {})),
        "quality_thresholds": dict(data.get("quality_thresholds", {})),
        "regime_cache_path": data.get("regime_cache_path"),
        "regime_data_splitting_result": data.get("regime_data_splitting_result"),
        "model_type": data.get("model_type"),
    }

    multi_h_keys = {k: data[k] for k in _MULTI_H_KEYS if k in data}
    if multi_h_keys:
        base_kwargs["multi_horizon"] = validate_multi_horizon_artifacts(multi_h_keys)

    flo_keys = {k: data[k] for k in _FLO_KEYS if k in data}
    if flo_keys:
        base_kwargs["feature_lookback"] = validate_feature_lookback_artifacts(flo_keys)

    interactive_keys = {k: data[k] for k in _INTERACTIVE_KEYS if k in data}
    if interactive_keys:
        base_kwargs["interactive"] = validate_interactive_artifacts(interactive_keys)

    final_keys = {k: data[k] for k in _FINAL_SELECTION_KEYS if k in data}
    if final_keys:
        base_kwargs["final_selection"] = validate_final_selection_artifacts(final_keys)

    return PipelineState(**base_kwargs)
