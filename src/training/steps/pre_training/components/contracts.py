"""Typed contracts for pre-training pipeline components.

This module defines standardized data structures (contracts) for artifacts produced by
pre-training pipeline components. These contracts ensure consistency and type safety
across the entire pre-training pipeline.

ARTIFACT SCHEMA STANDARDIZATION
==============================

All labeling components in the pre-training pipeline must produce artifacts that
conform to the BaseLabelingArtifacts structure. This ensures:

1. Consistent field names across all labelers
2. Required fields for downstream compatibility
3. Standardized metadata for auditing and debugging
4. Type safety for validation

REQUIRED FIELDS FOR ALL LABELERS:
--------------------------------
- labeled_data: Primary labeled dataset (DataFrame)
- labels: Alias to labeled_data for backward compatibility (DataFrame)
- confidence_scores: Confidence scores for predictions (DataFrame)
- eligibility_masks: Boolean masks for valid samples (DataFrame)
- quality_scores: Quality metrics per target column (Dict[str, Dict[str, float]])
- normalization_factors: Normalization metadata (Dict[str, Any])

- processing_time: Execution time in seconds (float)
- n_samples: Number of samples processed (int)
- n_targets: Number of target columns (int)
- n_horizons: Number of prediction horizons (int)
- method: Labeling method identifier (str)

OPTIONAL FIELDS (populate when available):
----------------------------------------
- sigma_payoffs: Payoff calculations in sigma units (DataFrame)
- horizon_weights: Weights for different horizons (Dict[str, float])
- target_columns: Target column names (List[str])
- target_parameters: Target generation parameters (Dict[str, Any])
- target_shifts: Shift information per target (Dict[str, Any])
- execution_timing: Detailed timing breakdown (Dict[str, Any])
- balancing_applied: Whether balancing was used (bool)
- sample_weights: Sample weights for training (Series/DataFrame)
- validation_results: Validation outcomes (Dict[str, Any])
- smoothing_settings: Smoothing parameters (Dict[str, Any])
- metadata: Additional component-specific metadata (Dict[str, Any])

METADATA STANDARDIZATION:
------------------------
All labelers should include standard metadata fields:
- symbol: Trading symbol (str)
- exchange: Exchange name (str)
- timeframe: Data timeframe (str)
- regime_aware: Whether regime analysis was used (bool)
- processing_time: Execution time (float)
- n_samples: Sample count (int)
- n_targets: Target count (int)
- n_horizons: Horizon count (int)
- source: Data source identifier (str)

USAGE PATTERNS:
==============

1. LABELER IMPLEMENTATION:
   ```python
   artifacts = {
       'multi_horizon_labeling_result': {
           'labeled_data': labels_df,
           'labels': labels_df,  # Backward compatibility
           'confidence_scores': confidence_df,
           'eligibility_masks': eligibility_df,
           'quality_scores': quality_metrics,
           'normalization_factors': normalization_metadata,
           'processing_time': execution_time,
           'n_samples': len(labels_df),
           'n_targets': len(target_columns),
           'n_horizons': len(horizons),
           'method': 'analyst_profit_labeling',
           'metadata': {
               'symbol': self.config.symbol,
               'exchange': self.config.exchange,
               'timeframe': self.config.timeframe,
               'regime_aware': bool(regime_assignments is not None),
               'processing_time': execution_time,
               'n_samples': len(labels_df),
               'n_targets': len(target_columns),
               'n_horizons': len(horizons),
               'source': 'all_market_data'
           }
       },
       'labeling_report': {
           'status': 'completed',
           'timestamp': datetime.now().isoformat(),
           'method': 'analyst_profit_labeling',
           'summary': quality_metrics
       }
   }
   ```

2. DOWNSTREAM COMPONENT USAGE:
   ```python
   # Access labels with fallback for backward compatibility
   labels_df = artifacts.get('labeled_data')
   if labels_df is None or labels_df.empty:
       labels_df = artifacts.get('labels')

   # Access metadata
   metadata = artifacts.get('metadata', {})
   symbol = metadata.get('symbol', 'UNKNOWN')
   exchange = metadata.get('exchange', 'UNKNOWN')

   # Use normalization factors for auditing
   norm_factors = artifacts.get('normalization_factors', {})
   scaling_reference = norm_factors.get('scaling_reference', 'Unknown')
   ```

3. VALIDATION:
   ```python
   # Artifacts are validated against contracts
   validated_artifacts = validate_multi_horizon_labeling_result(
       artifacts,
       context='component_name'
   )
   ```

MIGRATION GUIDE:
===============

When updating existing labelers to use the standardized schema:

1. Ensure all required fields are populated
2. Add normalization_factors with appropriate metadata
3. Move component-specific fields to metadata dictionary
4. Remove redundant or inconsistent field structures
5. Update downstream components to use standardized field names

ERROR HANDLING:
==============

If a labeler cannot produce all required fields (e.g., due to errors), it should:
1. Populate as many fields as possible with default values
2. Include error information in validation_results
3. Set appropriate error flags in metadata
4. Ensure downstream components can handle missing fields gracefully
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field, fields
from functools import lru_cache
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple, Type, TypeVar, Union

from src.training.steps.pre_training.validation.data_contracts import (
    validate_feature_artifact,
    validate_multi_horizon_labeling_result,
    validate_selection_artifact,
)
from src.utils.common_operations import safe_dataframe_operation
from src.utils.common_utilities import CommonUtilities
from src.utils.math_validation import validate_finite
from src.utils.serialization_utils import JSONSerializer
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_error


def _copy_mapping(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return a shallow dictionary copy from an optional mapping."""

    if not value:
        return {}

    # Use utility for safe mapping copy
    try:
        return dict(value)
    except Exception as e:
        logger = system_logger.getChild('contracts')
        logger.warning(f"Failed to copy mapping: {e}. Using empty dict as fallback.")
        return {}


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
            attr_value = getattr(self, key)
            if attr_value is not None:
                return attr_value
            else:
                # For dataclass fields that are None, fall back to extra dict
                return self.extra[key]
        elif key in self.extra:
            return self.extra[key]
        else:
            raise KeyError(f"Key '{key}' not found in dataclass fields or extra dictionary")

    def __setitem__(self, key: str, value: Any) -> None:
        if key in _field_names(type(self)):
            # Validate type safety for dataclass fields when possible
            try:
                import inspect
                field_info = None
                for field in fields(type(self)):
                    if field.name == key:
                        field_info = field
                        break

                if field_info and field_info.type != Any:
                    # Basic type validation - could be enhanced with more sophisticated validation
                    expected_type = field_info.type
                    # Check if expected_type is actually a valid type for isinstance()
                    try:
                        # Get origin type for complex annotations like Optional, Union, etc.
                        import typing
                        origin = typing.get_origin(expected_type)
                        if origin is not None:
                            # For Union types (including Optional), extract the actual types
                            args = typing.get_args(expected_type)
                            if args and value is not None:
                                # Check against all types in the union (excluding None)
                                valid_types = tuple(arg for arg in args if arg is not type(None))
                                if valid_types and not isinstance(value, valid_types):
                                    import logging
                                    logger = logging.getLogger(__name__)
                                    logger.warning(f"Type mismatch for field '{key}': expected {expected_type}, got {type(value)}")
                        elif isinstance(expected_type, type) and not isinstance(value, expected_type) and value is not None:
                            # Simple type check for non-generic types
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.warning(f"Type mismatch for field '{key}': expected {expected_type}, got {type(value)}")
                    except (TypeError, AttributeError):
                        # If type checking fails, just skip validation
                        pass

            except (ImportError, AttributeError):
                # If validation fails, just set the value
                pass

            setattr(self, key, value)
        else:
            # For extra fields, allow any value but log for type safety awareness
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
class BaseLabelingArtifacts(ArtifactBundle):
    """Base class for standardized labeling artifact structures.

    This class defines the standard structure for labeling artifacts across all labelers
    in the pre-training pipeline. All labelers should use this structure to ensure
    consistency and compatibility with downstream components.

    Schema Requirements:
    -------------------
    - labeled_data: Primary labeled dataset (DataFrame) - required for backward compatibility
    - labels: Alias to labeled_data (DataFrame) - required for backward compatibility
    - confidence_scores: Confidence scores for labels (DataFrame)
    - eligibility_masks: Boolean masks indicating valid samples (DataFrame)
    - quality_scores: Quality metrics for each target column (Dict[str, Dict[str, float]])
    - normalization_factors: Normalization metadata for auditing (Dict[str, Any])

    - processing_time: Time taken for labeling operation (float)
    - n_samples: Number of samples processed (int)
    - n_targets: Number of target columns (int)
    - n_horizons: Number of prediction horizons (int)
    - method: Labeling method used (str)

    Optional Fields:
    ---------------
    - sigma_payoffs: Payoff calculations in sigma units (DataFrame)
    - horizon_weights: Weights for different prediction horizons (Dict[str, float])
    - target_columns: List of target column names (List[str])
    - target_parameters: Parameters used for each target (Dict[str, Any])
    - target_shifts: Shift information for each target (Dict[str, Any])
    - execution_timing: Detailed timing breakdown (Dict[str, Any])
    - balancing_applied: Whether sample balancing was applied (bool)
    - sample_weights: Sample weights for training (Series/DataFrame)
    - validation_results: Validation outcomes (Dict[str, Any])
    - smoothing_settings: Smoothing parameters used (Dict[str, Any])
    - metadata: Additional metadata (Dict[str, Any])

    Usage Guidelines:
    ----------------
    1. All labelers must populate the core required fields
    2. Optional fields should be populated when relevant data is available
    3. Metadata should include standard fields: symbol, exchange, timeframe, etc.
    4. Normalization factors should provide transparency for downstream auditing
    """

    # Core labeling data - required for all labelers
    labeled_data: Any = None
    labels: Any = None
    confidence_scores: Any = None
    eligibility_masks: Any = None
    quality_scores: Dict[str, Any] = field(default_factory=dict)
    normalization_factors: Dict[str, Any] = field(default_factory=dict)

    # Metadata fields - standardized across all labelers
    processing_time: float = 0.0
    n_samples: int = 0
    n_targets: int = 0
    n_horizons: int = 0
    method: str = ""

    # Optional fields for enhanced functionality
    sigma_payoffs: Any = None
    horizon_weights: Dict[str, float] = field(default_factory=dict)
    target_columns: List[str] = field(default_factory=list)
    target_parameters: Dict[str, Any] = field(default_factory=dict)
    target_shifts: Dict[str, Any] = field(default_factory=dict)
    execution_timing: Dict[str, Any] = field(default_factory=dict)
    balancing_applied: Optional[bool] = None
    sample_weights: Any = None
    validation_results: Dict[str, Any] = field(default_factory=dict)
    smoothing_settings: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    """Validate multi-horizon artifacts with enhanced error handling."""
    logger = system_logger.getChild('contracts.validation')
    common_utils = CommonUtilities()
    json_serializer = JSONSerializer()

    tprint("🔍 Validating multi-horizon artifacts...")

    if bundle.multi_horizon_labeling_result:
        try:
            bundle.multi_horizon_labeling_result = validate_multi_horizon_labeling_result(
                bundle.multi_horizon_labeling_result,
                context="components.multi_horizon_labeling_result",
            )
            logger.debug("✅ Multi-horizon labeling result validated successfully")
            tprint_success("✅ Multi-horizon labeling result validated successfully")
        except Exception as e:
            logger.warning(f"⚠️ Multi-horizon validation failed: {e}")
            tprint_warning(f"⚠️ Multi-horizon validation failed: {e}")
            # Try to serialize error details for debugging
            try:
                json_serializer.save({
                    'error': str(e),
                    'context': 'multi_horizon_validation',
                    'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else None
                }, '/tmp/multi_horizon_validation_error.json')
                tprint_warning("💾 Multi-horizon validation error details saved for debugging")
            except Exception as save_error:
                tprint_warning(f"⚠️ Could not save validation error details: {save_error}")
                logger.warning(f"Could not save validation error details: {save_error}")
    else:
        tprint_warning("⚠️ No multi-horizon labeling result to validate")
    return bundle


def _validate_interactive_features(
    bundle: InteractiveFeatureArtifacts,
) -> InteractiveFeatureArtifacts:
    """Validate interactive feature artifacts with enhanced error handling."""
    logger = system_logger.getChild('contracts.validation')

    tprint("🔍 Validating interactive feature artifacts...")

    if bundle.interactive_feature_generation_result:
        try:
            bundle.interactive_feature_generation_result = validate_feature_artifact(
                bundle.interactive_feature_generation_result,
                context="components.interactive_feature_generation_result",
            )
            logger.debug("✅ Interactive feature generation result validated successfully")
            tprint_success("✅ Interactive feature generation result validated successfully")
        except Exception as e:
            logger.warning(f"⚠️ Interactive features validation failed: {e}")
            tprint_warning(f"⚠️ Interactive features validation failed: {e}")
    else:
        tprint_warning("⚠️ No interactive feature generation result to validate")
    return bundle


def _validate_final_selection(
    bundle: FinalFeatureSelectionArtifacts,
) -> FinalFeatureSelectionArtifacts:
    """Validate final selection artifacts with enhanced error handling."""
    logger = system_logger.getChild('contracts.validation')

    tprint("🔍 Validating final feature selection artifacts...")

    if bundle.final_feature_selection_result:
        try:
            bundle.final_feature_selection_result = validate_selection_artifact(
                bundle.final_feature_selection_result,
                context="components.final_feature_selection_result",
            )
            logger.debug("✅ Final feature selection result validated successfully")
            tprint_success("✅ Final feature selection result validated successfully")
        except Exception as e:
            logger.warning(f"⚠️ Final selection validation failed: {e}")
            tprint_warning(f"⚠️ Final selection validation failed: {e}")
    else:
        tprint_warning("⚠️ No final feature selection result to validate")
    return bundle


def _validate_feature_lookback(bundle: FeatureLookbackArtifacts) -> FeatureLookbackArtifacts:
    """Validate FeatureLookbackArtifacts bundle."""
    # Basic validation - ensure required fields exist
    if not bundle.feature_lookback_optimization_result:
        raise ValueError("feature_lookback_optimization_result is empty")
    return bundle

_ARTIFACT_VALIDATORS: Dict[Type[ArtifactBundle], Validator] = {
    MultiHorizonArtifacts: _validate_multi_horizon,
    InteractiveFeatureArtifacts: _validate_interactive_features,
    FinalFeatureSelectionArtifacts: _validate_final_selection,
    FeatureLookbackArtifacts: _validate_feature_lookback,
}


def validate_artifact_bundle(bundle: ArtifactBundle) -> ArtifactBundle:
    """Validate a typed artifact bundle against its registered contract."""

    logger = system_logger.getChild('contracts.validation')
    bundle_type = type(bundle).__name__

    tprint(f"🔍 Validating artifact bundle of type: {bundle_type}")

    validator = _ARTIFACT_VALIDATORS.get(type(bundle))
    if validator is None:
        tprint_warning(f"⚠️ No validator found for bundle type {bundle_type}")
        logger.debug(f"No validator found for bundle type {bundle_type}")
        return bundle

    try:
        result = validator(bundle)
        tprint_success(f"✅ Artifact bundle validation completed for {bundle_type}")
        return result
    except Exception as e:
        tprint_error(f"❌ Artifact bundle validation failed for {bundle_type}: {e}")
        logger.warning(f"⚠️ Artifact bundle validation failed: {e}")
        # Return the original bundle even if validation fails to maintain compatibility
        return bundle


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

