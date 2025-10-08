"""Typed contracts shared by pre-training components and orchestrators."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, TypedDict

import numpy as np
import pandas as pd

from src.training.config.data_locator import DataLocator
from src.utils.random_seeding import SeededRNGs


@dataclass
class ValidationResults:
    """Basic validation outcome shared across artifacts."""

    is_valid: bool
    issues: Sequence[str] = field(default_factory=list)


@dataclass
class MultiHorizonLabelingResult:
    """Structured payload returned by the multi-horizon labeler."""

    labeled_data: pd.DataFrame
    labels: pd.DataFrame
    confidence_scores: Optional[pd.DataFrame] = None
    eligibility_masks: Optional[pd.DataFrame] = None
    sigma_payoffs: Optional[pd.DataFrame] = None
    quality_scores: Dict[str, Any] = field(default_factory=dict)
    horizon_weights: Mapping[str, float] = field(default_factory=dict)
    target_columns: Sequence[str] = field(default_factory=list)
    normalization_factors: Dict[str, Any] = field(default_factory=dict)
    method: Optional[str] = None
    balancing_applied: Optional[bool] = None
    sample_weights: Optional[Any] = None
    validation_results: Optional[ValidationResults] = None
    smoothing_settings: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    market_data: Optional[pd.DataFrame] = None
    market_data_batches: Optional[Sequence[pd.DataFrame]] = None


@dataclass
class StandardizedLabelingOutput:
    """Unified labeling format consumed by downstream components."""

    labels: pd.DataFrame
    weights: Mapping[str, float]
    target_columns: Sequence[str] = field(default_factory=list)
    quality_scores: Dict[str, Any] = field(default_factory=dict)
    confidence_scores: Optional[pd.DataFrame] = None
    eligibility_masks: Optional[pd.DataFrame] = None
    sigma_payoffs: Optional[pd.DataFrame] = None
    sample_weights: Optional[Any] = None
    normalization_factors: Dict[str, Any] = field(default_factory=dict)
    validation_results: Optional[ValidationResults] = None
    smoothing_settings: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiHorizonArtifacts:
    """Complete artifact bundle emitted by the multi-horizon labeler."""

    labeling_result: MultiHorizonLabelingResult
    labeling_report: Mapping[str, Any]
    standardized_output: Optional[StandardizedLabelingOutput] = None
    validated_schemas: Optional[Mapping[str, Any]] = None


@dataclass
class FeatureLookbackSummary:
    """High level summary of feature lookback optimisation."""

    timestamp: str
    symbol: str
    exchange: str
    timeframe: str
    optimization_results: Mapping[str, Any]


@dataclass
class FeatureLookbackOptimizationResult:
    """Detailed feature lookback optimisation artifact."""

    optimization_results: Mapping[str, Any]
    summary: FeatureLookbackSummary
    component_type: str = "feature_lookback_optimization"
    timestamp: Optional[str] = None


@dataclass
class FeatureLookbackArtifacts:
    """Artifacts emitted by the feature lookback optimisation component."""

    summary: FeatureLookbackSummary
    result: FeatureLookbackOptimizationResult


@dataclass
class FinalSelectionResult:
    """Structured payload produced by the final feature selection step."""

    symbol: str
    exchange: str
    timeframe: str
    data_dir: str
    feature_selection_config: Mapping[str, Any]
    execution_mode: str
    success: bool
    stage_reduction: Mapping[str, int]
    hardware_performance: Mapping[str, Any]
    validated_schemas: Optional[Mapping[str, Any]] = None
    final_features: Sequence[str] = field(default_factory=list)
    stage_1_features: Sequence[str] = field(default_factory=list)
    stage_2_features: Sequence[str] = field(default_factory=list)
    stage_3_features: Sequence[str] = field(default_factory=list)
    feature_counts: Mapping[str, int] = field(default_factory=dict)
    stage_scores: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    selection_time: Optional[float] = None
    is_unsupervised: Optional[bool] = None


@dataclass
class FinalSelectionArtifacts:
    """Artifacts emitted by the final feature selection component."""

    result: FinalSelectionResult
    validated_schemas: Optional[Mapping[str, Any]] = None


class PipelineState(TypedDict, total=False):
    """Shared pipeline state propagated between orchestrator steps."""

    symbol: str
    exchange: str
    timeframe: str
    data_dir: str
    data_cache_dir: str
    artifacts_dir: str
    generated_dir: str
    outcomes_dir: str
    final_feature_selection_dir: str
    data_dir_key: str
    cache_dir_key: str
    artifacts_dir_key: str
    generated_dir_key: str
    outcomes_dir_key: str
    final_feature_selection_dir_key: str
    data_locator: DataLocator
    custom_params: Dict[str, Any]
    quality_thresholds: Dict[str, float]
    market_data_batch_size: int
    market_data_window_days: int
    random_seed: int
    python_rng: random.Random
    numpy_rng: np.random.Generator
    seeded_rngs: SeededRNGs
    regime_cache_path: str
    regime_data_splitting_result: Any
    multi_horizon_labeling_result: MultiHorizonLabelingResult
    labeling_report: Mapping[str, Any]
    standardized_output: StandardizedLabelingOutput
    feature_lookback_optimization_summary: FeatureLookbackSummary
    feature_lookback_optimization_result: FeatureLookbackOptimizationResult
    final_feature_selection_result: FinalSelectionResult
    validated_schemas: Mapping[str, Any]
    artifacts: MutableMapping[str, Any]


__all__ = [
    "PipelineState",
    "ValidationResults",
    "MultiHorizonLabelingResult",
    "StandardizedLabelingOutput",
    "MultiHorizonArtifacts",
    "FeatureLookbackSummary",
    "FeatureLookbackOptimizationResult",
    "FeatureLookbackArtifacts",
    "FinalSelectionResult",
    "FinalSelectionArtifacts",
]
