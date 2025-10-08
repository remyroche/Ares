"""
Multi-Horizon Profit Labeler Component for Pre-Training Pipeline.

This component integrates the VolatilityAwareMultiHorizonLabeler with regime data splitting
to create differentiated profit labels for different market regimes.
"""

import asyncio
import copy
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, AsyncIterator, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.logger import system_logger
from src.training.config.data_locator import DataLocator as PipelineDataLocator
from src.training.steps.pre_training.artifacts.manifest import (
    ArtifactManifest,
    DataLocator as ArtifactDataLocator,
)
from .settings import get_pre_training_settings
from src.training.common.artifact_persistence import SaveReport

try:
    from src.utils.data.klines_parquet import get_klines_manager
except Exception:  # pragma: no cover - defensive guard for optional dependency
    get_klines_manager = None  # type: ignore[assignment]

# Import the volatility-aware multi-horizon labeler
try:
    from src.training.steps.pre_training.profit_labeling.volatility_aware_labeler import (
        VolatilityAwareMultiHorizonLabeler,
        VolatilityAwareConfig,
        LabelingResult,
        create_enhanced_analyst_labeler,
        create_enhanced_tactician_labeler,
        LabelDefinitionType,
    )
except (ImportError, SyntaxError):  # pragma: no cover - defensive fallback for optional dependency
    class VolatilityAwareConfig:  # type: ignore[no-redef]
        """Fallback configuration stub when volatility labeler is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.params = kwargs

    class LabelingResult:  # type: ignore[no-redef]
        """Minimal stub replicating the interface used by the labeler."""

        def __init__(self, **kwargs: Any) -> None:
            self.labels = kwargs.get('labels', pd.DataFrame())
            self.confidence_scores = kwargs.get('confidence_scores', pd.DataFrame())
            self.eligibility_masks = kwargs.get('eligibility_masks', pd.DataFrame())
            self.sigma_payoffs = kwargs.get('sigma_payoffs', pd.DataFrame())
            self.training_labels = kwargs.get('training_labels', pd.DataFrame())
            self.normalization_factors = kwargs.get('normalization_factors', {})
            self.quality_scores = kwargs.get('quality_scores', {})
            self.n_samples = kwargs.get('n_samples', 0)
            self.n_targets = kwargs.get('n_targets', 0)
            self.n_horizons = kwargs.get('n_horizons', 0)
            self.processing_time = kwargs.get('processing_time', 0.0)

    class _UnavailableVolatilityLabeler:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs
            self.config = None

        def generate_labels(self, *args: Any, **kwargs: Any) -> LabelingResult:
            raise RuntimeError("Volatility-aware labeler is unavailable in this environment")

    def create_enhanced_analyst_labeler(*args: Any, **kwargs: Any) -> _UnavailableVolatilityLabeler:  # type: ignore[no-redef]
        return _UnavailableVolatilityLabeler(*args, **kwargs)

    def create_enhanced_tactician_labeler(*args: Any, **kwargs: Any) -> _UnavailableVolatilityLabeler:  # type: ignore[no-redef]
        return _UnavailableVolatilityLabeler(*args, **kwargs)

    class VolatilityAwareMultiHorizonLabeler(_UnavailableVolatilityLabeler):  # type: ignore[no-redef]
        pass

    LabelDefinitionType = str  # type: ignore[no-redef]
from src.training.steps.pre_training.standardized_labeling_interface import (
    assert_labels_sigma_scaled,
    validate_dataframe_schema
)
from src.training.steps.pre_training.validation.schemas import (
    SchemaValidationException,
    report_hypothesis_count,
    enforce_feature_temporal_alignment,
    schema_metadata,
    validate_engineered_features,
    validate_labeled_dataset,
    validate_raw_ohlcv,
)
from src.training.steps.pre_training.validation.cv import (
    WalkForwardFold,
    purged_walk_forward_cv,
    validate_cv_no_leakage,
)
from src.training.steps.pre_training.validation.data_contracts import (
    DataContractValidationError,
    validate_multi_horizon_labeling_result,
)
from src.training.steps.pre_training.column_naming import (
    ColumnNamespace,
    ensure_dataframe_namespace,
    ensure_namespace,
    filter_namespace_columns,
    strip_namespace,
)

# Import the label balancing system
try:
    from src.training.steps.pre_training.profit_labeling.label_balancing import (
        ComprehensiveBalancingSystem,
        BalancingConfig,
        WeightingConfig,
        RegimeConfig,
        ValidationFairnessConfig,
        DEFAULT_BALANCING_CONFIG,
        DEFAULT_WEIGHTING_CONFIG,
        DEFAULT_REGIME_CONFIG,
        DEFAULT_FAIRNESS_CONFIG
    )
    BALANCING_SYSTEM_AVAILABLE = True
except (ImportError, SyntaxError):
    BALANCING_SYSTEM_AVAILABLE = False
    ComprehensiveBalancingSystem = None  # type: ignore[assignment]
    BalancingConfig = None  # type: ignore[assignment]
    WeightingConfig = None  # type: ignore[assignment]
    RegimeConfig = None  # type: ignore[assignment]
    ValidationFairnessConfig = None  # type: ignore[assignment]
    DEFAULT_BALANCING_CONFIG = None  # type: ignore[assignment]
    DEFAULT_WEIGHTING_CONFIG = None  # type: ignore[assignment]
    DEFAULT_REGIME_CONFIG = None  # type: ignore[assignment]
    DEFAULT_FAIRNESS_CONFIG = None  # type: ignore[assignment]

# Import base component
from src.training.steps.pre_training.components.base_component import BasePreTrainingComponent, ComponentConfig, ComponentResult
from src.training.steps.pre_training.components.contracts import MultiHorizonArtifacts, PipelineState


def _normalize_for_hash(value: Any) -> Any:
    """Normalize complex structures into hash-friendly primitives."""
    if isinstance(value, dict):
        return {str(k): _normalize_for_hash(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Series):
        return _normalize_for_hash(value.to_dict())
    if isinstance(value, pd.Index):
        return _normalize_for_hash(list(value))
    if isinstance(value, pd.DataFrame):
        return {
            'columns': _normalize_for_hash(list(value.columns)),
            'index': _normalize_for_hash(value.index.tolist()),
            'data': _normalize_for_hash(value.to_dict(orient='list')),
        }
    if isinstance(value, np.ndarray):
        return _normalize_for_hash(value.tolist())
    return value


def _json_default(value: Any) -> Any:
    normalized = _normalize_for_hash(value)
    if isinstance(normalized, (dict, list, str, int, float, bool)) or normalized is None:
        return normalized
    return str(normalized)


def _compute_outcome_digest(
    symbol: str,
    exchange: str,
    timeframe: str,
    outcome_payload: Dict[str, Any],
) -> str:
    payload = {
        'symbol': symbol,
        'exchange': exchange,
        'timeframe': timeframe,
        'outcome': _normalize_for_hash(outcome_payload),
    }
    serialized = json.dumps(payload, sort_keys=True, default=_json_default, ensure_ascii=False)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def _build_outcome_filename(
    symbol: str,
    exchange: str,
    timeframe: str,
    outcome_payload: Dict[str, Any],
) -> Tuple[str, str]:
    digest = _compute_outcome_digest(symbol, exchange, timeframe, outcome_payload)
    filename = (
        f"market_analysis_multi_horizon_profit_labeler_outcome_"
        f"{symbol}_{exchange}_{timeframe}_{digest[:16]}.json"
    )
    return filename, digest


def _persist_labeling_outcome(
    *,
    base_dir: Path,
    symbol: str,
    exchange: str,
    timeframe: str,
    outcome_payload: Dict[str, Any],
    logger: logging.Logger,
    correlation_id: Optional[str] = None,
) -> Tuple[SaveReport, bool]:
    """Persist the labeling outcome using a deterministic, idempotent strategy."""

    base_dir.mkdir(parents=True, exist_ok=True)
    filename, digest = _build_outcome_filename(symbol, exchange, timeframe, outcome_payload)
    path = base_dir / filename

    serialized = json.dumps(outcome_payload, indent=2, default=_json_default, ensure_ascii=False).encode('utf-8')
    file_size = len(serialized)
    skipped = False

    start = perf_counter()
    with open(path, 'wb') as handle:
        handle.write(serialized)
    duration = perf_counter() - start

    report = SaveReport(
        paths={'labeling_outcome': str(path)},
        bytes={'labeling_outcome': 0 if skipped else file_size},
        duration=duration,
        checksum={'labeling_outcome': digest},
        correlation_id=correlation_id or str(uuid.uuid4()),
    )

    log_payload = {
        'event': 'labeling_outcome_save',
        'correlation_id': report.correlation_id,
        'symbol': symbol,
        'exchange': exchange,
        'timeframe': timeframe,
        'path': report.paths['labeling_outcome'],
        'bytes_written': report.bytes['labeling_outcome'],
        'file_size': file_size,
        'checksum': digest,
        'skipped': skipped,
        'duration_sec': duration,
    }
    try:
        logger.info(json.dumps(log_payload, ensure_ascii=False))
    except TypeError:
        logger.info(log_payload)

    return report, skipped


def _ensure_labeling_contract(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a payload that satisfies the downstream data contract defaults."""

    normalized = dict(payload)
    required_columns = ['immediate_opportunity', 'short_term_opportunity', 'leverage_adjusted_score']
    labeled_frame = normalized.get('labeled_data')
    labels_entry = normalized.get('labels')

    def _coerce_frame(source: Any) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            frame = source.copy()
        elif isinstance(source, Mapping):
            frame = pd.DataFrame(source)
        elif isinstance(source, (list, tuple)):
            frame = pd.DataFrame(source)
        else:
            frame = pd.DataFrame()

        for column in required_columns:
            if column not in frame.columns:
                frame[column] = 0

        if frame.empty:
            frame.index = pd.DatetimeIndex([], tz='UTC')
        else:
            index = pd.to_datetime(frame.index, utc=True, errors='coerce')
            if index.isna().all():
                index = pd.date_range(
                    end=pd.Timestamp.utcnow().tz_localize('UTC'),
                    periods=len(frame),
                    freq='H'
                )
            frame.index = index

        return frame

    labels_frame = _coerce_frame(labels_entry)
    normalized['labels'] = labels_frame

    labeled_frame = _coerce_frame(labeled_frame if labeled_frame is not None else labels_frame)
    normalized['labeled_data'] = labeled_frame
    validation = normalized.setdefault('validation_results', {})
    if isinstance(validation, Mapping):
        validation = dict(validation)
    validation.setdefault('is_valid', True)
    normalized['validation_results'] = validation
    normalized.setdefault('metadata', {})
    return normalized


@dataclass
class HorizonWeightsConfig:
    """Configuration for horizon weights in multi-horizon labeling."""
    micro: float = 0.0   # 0% - disabled for now
    small: float = 0.5   # 50% - immediate opportunities
    medium: float = 0.3  # 30% - short-term opportunities
    high: float = 0.2    # 20% - longer-term opportunities


@dataclass
class TransactionCostConfig:
    """Configuration for transaction cost modeling."""
    maker_fee: float = 0.0002  # 0.02% maker fee
    taker_fee: float = 0.0004  # 0.04% taker fee
    slippage_bps: float = 2.0  # 2 basis points slippage
    enable_cost_adjustment: bool = True
    
    def total_roundtrip_cost(self, is_aggressive: bool = True) -> float:
        """Calculate total round-trip transaction cost."""
        fee = self.taker_fee if is_aggressive else self.maker_fee
        slippage = self.slippage_bps / 10000.0
        return 2 * (fee + slippage)  # Round-trip


@dataclass
class TemporalValidationConfig:
    """Configuration for temporal validation."""
    enable_temporal_validation: bool = True
    enable_purging: bool = True
    purge_window_hours: int = 24
    embargo_window_hours: int = 12
    train_ratio: float = 0.70
    validation_ratio: float = 0.20
    test_ratio: float = 0.10
    walk_forward_folds: int = 3
    validate_distribution: bool = True


@dataclass
class MultiHorizonConfig:
    """Configuration for multi-horizon profit labeling."""

    # Timeframe settings
    timeframe: str = "1h"  # Updated to 1h for analyst
    base_period_minutes: float = 60.0  # Updated to 60 minutes for 1h timeframe
    
    # Horizon weights configuration
    horizon_weights: HorizonWeightsConfig = None
    
    # Transaction cost configuration
    transaction_costs: TransactionCostConfig = None
    
    # Temporal validation configuration
    temporal_validation: TemporalValidationConfig = None
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.horizon_weights is None:
            self.horizon_weights = HorizonWeightsConfig()
        if self.transaction_costs is None:
            self.transaction_costs = TransactionCostConfig()
        if self.temporal_validation is None:
            self.temporal_validation = TemporalValidationConfig()

    # Volatility-aware labeling settings
    enable_volatility_normalization: bool = True
    enable_noise_gating: bool = True
    enable_quality_scoring: bool = True
    enable_multi_target_scheme: bool = True

    # Regime integration settings
    enable_regime_aware_labeling: bool = True
    regime_column: str = "regime_state"

    # Enhanced label settings
    enable_enhanced_labels: bool = True
    label_definition_type: str = "analyst"  # "analyst", "tactician"

    # Output settings
    min_data_points: int = 1000
    save_intermediate_results: bool = True
    generate_reports: bool = True

    # Market data streaming controls
    market_data_batch_size: Optional[int] = None
    market_data_window_days: Optional[int] = None

    # Quality thresholds
    min_auc_threshold: float = 0.55
    max_auc_std_threshold: float = 0.03
    min_psi_threshold: float = 0.1
    max_flip_rate_threshold: float = 0.15
    min_balance_threshold: float = 0.35
    max_balance_threshold: float = 0.65

    # Label balancing and weighting settings
    enable_label_balancing: bool = True
    enable_sample_weighting: bool = True
    enable_regime_balancing: bool = True

    # Balancing configuration (can be customized)
    balancing_config: BalancingConfig = None
    weighting_config: WeightingConfig = None
    regime_config: RegimeConfig = None
    fairness_config: ValidationFairnessConfig = None
    data_locator: Optional[PipelineDataLocator] = None
    data_dir_key: str = "market_data"
    outcomes_dir_key: str = "multi_horizon_outcomes"


def validate_and_prepare_dataframe(
    df: pd.DataFrame,
    name: str = "DataFrame",
    duplicate_threshold: Optional[float] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Validate and prepare a DataFrame for processing.

    Args:
        df: DataFrame to validate and prepare
        name: Name of the DataFrame for logging
        duplicate_threshold: Optional threshold for duplicate index share
        metrics: Optional dict that will be populated with quality metrics

    Returns:
        Cleaned and validated DataFrame
    """
    if df is None or df.empty:
        tprint_warning(f"⚠️ {name} is empty or None")
        return df

    quality_metrics = metrics if metrics is not None else {}
    quality_metrics['row_count'] = len(df)

    # Check for duplicate indices
    if df.index.has_duplicates:
        dup_count = df.index.duplicated().sum()
        duplicate_share = float(dup_count / len(df)) if len(df) else 0.0
        quality_metrics['duplicate_count'] = int(dup_count)
        quality_metrics['duplicate_index_share'] = duplicate_share
        threshold = duplicate_threshold if duplicate_threshold is not None else 0.0

        if duplicate_threshold is None or duplicate_share > threshold:
            tprint_warning(
                f"⚠️ {name} has {dup_count} duplicate indices ({duplicate_share:.2%}), removing duplicates (keeping first)"
            )
            df = df[~df.index.duplicated(keep='first')]
            quality_metrics['deduplicated'] = True
        else:
            tprint_info(
                f"ℹ️ {name} has {dup_count} duplicate indices ({duplicate_share:.2%}) within threshold {threshold:.2%}; retaining duplicates"
            )
            quality_metrics['deduplicated'] = False
    else:
        quality_metrics['duplicate_count'] = 0
        quality_metrics['duplicate_index_share'] = 0.0
        quality_metrics['deduplicated'] = False

    # Ensure index is sorted
    if not df.index.is_monotonic_increasing:
        tprint_info(f"📊 Sorting {name} by index")
        df = df.sort_index()
    
    tprint(f"✅ {name} validated: {len(df)} rows, {len(df.columns)} columns")
    return df


class MultiHorizonProfitLabeler:
    """
    Multi-Horizon Profit Labeler that integrates volatility-aware labeling with regime data.

    This class creates differentiated profit labels for different market regimes,
    ensuring that the labeling process accounts for regime-specific behaviors.
    """

    def __init__(self, config: MultiHorizonConfig = None):
        """Initialize multi-horizon profit labeler."""
        self.config = config or MultiHorizonConfig()
        self.logger = logging.getLogger('MultiHorizonProfitLabeler')
        self.quality_thresholds: Dict[str, float] = {}
        self._settings = get_pre_training_settings()
        self.pipeline_data_locator: Optional[PipelineDataLocator] = self.config.data_locator

        # Initialize the volatility-aware labeler
        if self.config.enable_enhanced_labels:
            if self.config.label_definition_type.lower() == "analyst":
                self.volatility_labeler = create_enhanced_analyst_labeler()
                tprint_info("   → Enhanced Analyst labels: Enabled")
            elif self.config.label_definition_type.lower() == "tactician":
                self.volatility_labeler = create_enhanced_tactician_labeler()
                tprint_info("   → Enhanced Tactician labels: Enabled")
            else:
                self.volatility_labeler = VolatilityAwareMultiHorizonLabeler(self._create_volatility_config())
                tprint_warning(f"   → Unknown label type '{self.config.label_definition_type}', using standard labels")
        else:
            self.volatility_labeler = VolatilityAwareMultiHorizonLabeler(self._create_volatility_config())
            tprint_info("   → Enhanced labels: Disabled")
        
        # Log transaction cost configuration
        if self.config.transaction_costs.enable_cost_adjustment:
            cost = self.config.transaction_costs.total_roundtrip_cost()
            tprint_info(f"   → Transaction cost adjustment: Enabled (round-trip: {cost:.4%})")
        
        # Log temporal validation configuration
        if self.config.temporal_validation.enable_temporal_validation:
            tprint_info(f"   → Temporal validation: Enabled")
            if self.config.temporal_validation.enable_purging:
                tprint_info(f"   → Purging: {self.config.temporal_validation.purge_window_hours}h window")

        # Initialize the balancing system if available
        self.balancing_system = None
        if BALANCING_SYSTEM_AVAILABLE and (self.config.enable_label_balancing or self.config.enable_sample_weighting):
            # Set default configurations if not provided
            balancing_config = self.config.balancing_config or DEFAULT_BALANCING_CONFIG
            weighting_config = self.config.weighting_config or DEFAULT_WEIGHTING_CONFIG
            regime_config = self.config.regime_config or DEFAULT_REGIME_CONFIG
            fairness_config = self.config.fairness_config or DEFAULT_FAIRNESS_CONFIG

            self.balancing_system = ComprehensiveBalancingSystem(
                balancing_config, weighting_config, regime_config, fairness_config
            )

            tprint_success("✅ Label balancing system initialized")
        else:
            tprint_info("ℹ️ Label balancing system disabled or not available")

        # Initialize artifact helpers
        self.artifact_locator = ArtifactDataLocator(self._settings.artifacts_root)
        self.artifact_manifest = ArtifactManifest()

        tprint_success("🚀 Multi-Horizon Profit Labeler initialized")
        tprint_info(f"   → Timeframe: {self.config.timeframe}")
        tprint_info(f"   → Regime-aware: {self.config.enable_regime_aware_labeling}")
        tprint_info(f"   → Volatility normalization: {self.config.enable_volatility_normalization}")
        tprint_info(f"   → Label balancing: {self.config.enable_label_balancing}")
        tprint_info(f"   → Sample weighting: {self.config.enable_sample_weighting}")

    def _create_volatility_config(self) -> VolatilityAwareConfig:
        """Create volatility-aware configuration from multi-horizon config."""
        return VolatilityAwareConfig(
            min_data_points=self.config.min_data_points,
            generate_reports=self.config.generate_reports,
            save_intermediate_results=self.config.save_intermediate_results,
            min_auc_threshold=self.config.min_auc_threshold,
            max_auc_std_threshold=self.config.max_auc_std_threshold,
            temporal_validation=self.config.temporal_validation
        )

    def _apply_namespace_conventions(self, labeling_result: LabelingResult) -> LabelingResult:
        """Ensure all labeling artifacts use the standardized namespaces."""

        labels_df = getattr(labeling_result, 'labels', None)
        if isinstance(labels_df, pd.DataFrame) and not labels_df.empty:
            labeling_result.labels = ensure_dataframe_namespace(labels_df, ColumnNamespace.TARGET)

        training_labels = getattr(labeling_result, 'training_labels', None)
        if isinstance(training_labels, pd.DataFrame) and not training_labels.empty:
            labeling_result.training_labels = ensure_dataframe_namespace(
                training_labels, ColumnNamespace.TARGET
            )
        confidence_scores = getattr(labeling_result, 'confidence_scores', None)
        if isinstance(confidence_scores, pd.DataFrame) and not confidence_scores.empty:
            labeling_result.confidence_scores = ensure_dataframe_namespace(
                confidence_scores, ColumnNamespace.LABEL
            )
        eligibility_masks = getattr(labeling_result, 'eligibility_masks', None)
        if isinstance(eligibility_masks, pd.DataFrame) and not eligibility_masks.empty:
            labeling_result.eligibility_masks = ensure_dataframe_namespace(
                eligibility_masks, ColumnNamespace.LABEL
            )
        sigma_payoffs = getattr(labeling_result, 'sigma_payoffs', None)
        if isinstance(sigma_payoffs, pd.DataFrame) and not sigma_payoffs.empty:
            labeling_result.sigma_payoffs = ensure_dataframe_namespace(
                sigma_payoffs, ColumnNamespace.TARGET
            )
        return labeling_result

    def _adjust_returns_for_transaction_costs(
        self,
        labeling_result: LabelingResult
    ) -> LabelingResult:
        """
        Adjust label returns for transaction costs.
        
        Args:
            labeling_result: Original labeling result
        
        Returns:
            LabelingResult with cost-adjusted labels
        """
        if not self.config.transaction_costs.enable_cost_adjustment:
            return labeling_result
        
        tprint_info("💰 Adjusting labels for transaction costs...")
        
        # Get round-trip cost
        roundtrip_cost = self.config.transaction_costs.total_roundtrip_cost()
        
        # Adjust sigma-normalized labels
        # Since labels are already sigma-normalized, we need to adjust them proportionally
        adjusted_labels = labeling_result.labels.copy()
        
        # Subtract cost from raw returns before normalization was applied
        # This is an approximation - ideally we'd adjust before sigma normalization
        if 'sigma_payoffs' in dir(labeling_result) and not labeling_result.sigma_payoffs.empty:
            # We have access to sigma payoffs, adjust those
            adjusted_sigma = labeling_result.sigma_payoffs - roundtrip_cost
            
            # Update normalization factors
            if labeling_result.normalization_factors:
                updated_factors = copy.deepcopy(labeling_result.normalization_factors)
                if 'cost_adjustment' not in updated_factors:
                    updated_factors['cost_adjustment'] = {}
                updated_factors['cost_adjustment']['roundtrip_cost'] = roundtrip_cost
                updated_factors['cost_adjustment']['maker_fee'] = self.config.transaction_costs.maker_fee
                updated_factors['cost_adjustment']['taker_fee'] = self.config.transaction_costs.taker_fee
                updated_factors['cost_adjustment']['slippage_bps'] = self.config.transaction_costs.slippage_bps
            else:
                updated_factors = {
                    'cost_adjustment': {
                        'roundtrip_cost': roundtrip_cost,
                        'maker_fee': self.config.transaction_costs.maker_fee,
                        'taker_fee': self.config.transaction_costs.taker_fee,
                        'slippage_bps': self.config.transaction_costs.slippage_bps
                    }
                }
        else:
            # No sigma payoffs available, apply percentage adjustment to labels
            # This is a conservative approximation
            cost_factor = 1.0 - (roundtrip_cost / 2.0)  # Reduce signal strength proportionally
            adjusted_labels = adjusted_labels * cost_factor
            updated_factors = labeling_result.normalization_factors or {}
        
        # Create new result with adjusted labels
        adjusted_result = LabelingResult(
            labels=adjusted_labels,
            confidence_scores=labeling_result.confidence_scores,
            eligibility_masks=labeling_result.eligibility_masks,
            sigma_payoffs=adjusted_sigma if 'adjusted_sigma' in locals() else labeling_result.sigma_payoffs,
            training_labels=adjusted_labels.copy(),
            normalization_factors=updated_factors,
            quality_scores=labeling_result.quality_scores,
            n_samples=labeling_result.n_samples,
            n_targets=labeling_result.n_targets,
            processing_time=labeling_result.processing_time
        )

        if hasattr(labeling_result, 'execution_timing'):
            adjusted_result.execution_timing = copy.deepcopy(getattr(labeling_result, 'execution_timing'))

        adjusted_result = self._apply_namespace_conventions(adjusted_result)
        tprint_success(f"✅ Transaction cost adjustment applied (round-trip: {roundtrip_cost:.4%})")

        return adjusted_result
    
    def _create_temporal_splits(self, data: pd.DataFrame) -> List[WalkForwardFold]:
        """Create purged, embargoed walk-forward folds for temporal validation."""

        config = self.config.temporal_validation

        if not config.enable_temporal_validation:
            return [
                WalkForwardFold(
                    fold=0,
                    train=data.copy(),
                    validation=data.iloc[0:0].copy(),
                    test=data.iloc[0:0].copy(),
                )
            ]

        if not data.index.is_monotonic_increasing:
            data = data.sort_index()

        tprint_info("📊 Creating walk-forward validation folds...")

        folds = list(
            purged_walk_forward_cv(
                data,
                n_splits=max(1, int(config.walk_forward_folds)),
                train_ratio=config.train_ratio,
                validation_ratio=config.validation_ratio,
                test_ratio=config.test_ratio,
                purge_window_hours=config.purge_window_hours,
                embargo_window_hours=config.embargo_window_hours,
            )
        )

        if not folds:
            tprint_warning(
                "⚠️ Insufficient data for walk-forward validation; returning single training fold"
            )
            return [
                WalkForwardFold(
                    fold=0,
                    train=data.copy(),
                    validation=data.iloc[0:0].copy(),
                    test=data.iloc[0:0].copy(),
                )
            ]

        validate_cv_no_leakage(
            [fold.to_mapping() for fold in folds],
            purge_window_hours=config.purge_window_hours,
            embargo_window_hours=config.embargo_window_hours,
        )

        for fold in folds:
            mapping = fold.to_mapping()
            tprint_info(
                "   → Fold %d: train=%d, val=%d, test=%d",
                fold.fold,
                len(mapping['train']),
                len(mapping['validation']),
                len(mapping['test']),
            )

        tprint_success(
            f"✅ Generated {len(folds)} walk-forward folds "
            f"(purge={config.purge_window_hours}h, embargo={config.embargo_window_hours}h)"
        )

        return folds

    async def execute_labeling(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: Optional[str] = None,
        regime_data: Optional[Dict[str, Any]] = None,
        quality_thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Execute multi-horizon profit labeling.

        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            exchange: Exchange name (e.g., 'binance')
            timeframe: Timeframe for labeling (e.g., '15m')
            data_dir: Directory containing historical data
            regime_data: Optional regime data for regime-aware labeling

        Returns:
            Dictionary containing labeling results and metadata
        """
        try:
            tprint_info(f"🏷️ Starting multi-horizon profit labeling for {symbol} on {exchange}")
            tprint_info(f"⏰ Timeframe: {timeframe}")

            thresholds = quality_thresholds or self.quality_thresholds or {}
            if quality_thresholds is not None:
                self.quality_thresholds = thresholds

            locator = self.pipeline_data_locator or self.config.data_locator
            if data_dir is None and locator:
                data_dir = str(locator.data_path(self.config.data_dir_key))
                self.pipeline_data_locator = locator

            if data_dir is None:
                raise ValueError("data_dir must be provided or resolvable via DataLocator")

            # Load market data
            tprint_info("📊 Loading market data...")
            market_data_batches: List[pd.DataFrame] = []

            if self.config.market_data_batch_size or self.config.market_data_window_days:
                async for batch in self._load_market_data(
                    symbol,
                    exchange,
                    timeframe,
                    data_dir,
                    batch_size=self.config.market_data_batch_size,
                    window_days=self.config.market_data_window_days,
                ):
                    market_data_batches.append(batch)

                if not market_data_batches:
                    tprint_error(f"❌ No market data available for {symbol} {timeframe}")
                    raise ValueError(f"No market data available for {symbol} {timeframe}")

                market_data = pd.concat(market_data_batches, axis=0).sort_index()
                market_data = market_data[~market_data.index.duplicated(keep="first")]
            else:
                market_data = None
                async for batch in self._load_market_data(
                    symbol,
                    exchange,
                    timeframe,
                    data_dir,
                ):
                    market_data = batch
                    break

                if market_data is None or market_data.empty:
                    tprint_error(f"❌ No market data available for {symbol} {timeframe}")
                    raise ValueError(f"No market data available for {symbol} {timeframe}")

            market_data = validate_raw_ohlcv(
                market_data,
                context="multi_horizon_profit_labeler.market_data"
            )

            tprint_success(
                f"✅ Market data loaded: {len(market_data)} rows, {len(market_data.columns)} columns"
            )
            if self.config.market_data_batch_size or self.config.market_data_window_days:
                tprint_info(
                    "   → Batches processed: "
                    f"{len(market_data_batches)} (batch_size={self.config.market_data_batch_size}, "
                    f"window_days={self.config.market_data_window_days})"
                )

            # Apply regime-aware labeling if enabled and regime data is available
            if self.config.enable_regime_aware_labeling and regime_data:
                tprint_info("🎭 Applying regime-aware labeling...")
                labeling_result = await self._execute_regime_aware_labeling(market_data, regime_data)
                tprint_success("✅ Regime-aware labeling completed")
            else:
                tprint_info("📊 Using standard volatility-aware labeling...")
                labeling_result = self.volatility_labeler.generate_labels(market_data)
                tprint_success("✅ Standard labeling completed")

            labeling_result = self._apply_namespace_conventions(labeling_result)

            # Apply transaction cost adjustment if enabled
            labeling_result = self._adjust_returns_for_transaction_costs(labeling_result)

            # Apply label balancing and sample weighting if enabled
            tprint_info("⚖️ Applying label balancing and sample weighting...")
            balanced_labeling_result = await self._apply_balancing_and_weighting(
                labeling_result, market_data, regime_data
            )
            balanced_labeling_result = self._apply_namespace_conventions(balanced_labeling_result)
            tprint_success("✅ Balancing and weighting completed")

            # Generate comprehensive report using the profit labeling report generator
            tprint_info("📋 Generating comprehensive report...")
            report = await self._generate_comprehensive_report(
                balanced_labeling_result, symbol, exchange, timeframe, regime_data
            )
            tprint_success("✅ Report generation completed")

            # Map target columns to expected names for feature lookback optimization compatibility
            mapping_metrics: Dict[str, Any] = {}
            mapped_labels = self._map_target_columns_for_feature_optimization(
                balanced_labeling_result.labels,
                duplicate_threshold=thresholds.get('duplicate_index'),
                quality_metrics=mapping_metrics,
            )
            mapped_labels = validate_labeled_dataset(
                mapped_labels,
                context="multi_horizon_profit_labeler.labeled_data"
            )

            walk_forward_folds = self._create_temporal_splits(mapped_labels)
            walk_forward_summary = self._summarize_walk_forward_folds(
                walk_forward_folds,
                mapped_labels,
            )

            # Create properly structured artifacts that feature lookback optimization expects
            # The feature lookback optimization expects 'labeled_data' or 'labels' keys
            tprint_info("📋 Creating comprehensive artifacts structure for downstream components")

            # Calculate horizon weights based on target quality and balance
            horizon_weights = self._calculate_horizon_weights(balanced_labeling_result, mapped_labels)
            tprint_info(f"⚖️ Calculated horizon weights: {horizon_weights}")

            # Extract target columns for feature optimization
            target_columns = self._extract_target_columns_for_optimization(mapped_labels)
            tprint_info(f"🎯 Identified target columns for feature optimization: {target_columns}")

            multi_target_result = getattr(balanced_labeling_result, 'multi_target_result', None)
            target_parameters = getattr(multi_target_result, 'target_parameters', {}) if multi_target_result else {}
            target_shifts = getattr(balanced_labeling_result, 'target_shifts', {}) or (
                getattr(multi_target_result, 'target_shifts', {}) if multi_target_result else {}
            )

            feature_frames: Dict[str, pd.DataFrame] = {}
            feature_metadata: Dict[str, Dict[str, int]] = {}

            # Validate that we have the required data for downstream components
            tprint_info("🔍 Validating data structure for downstream compatibility...")
            confidence_scores_df = balanced_labeling_result.confidence_scores
            if isinstance(confidence_scores_df, pd.DataFrame) and not confidence_scores_df.empty:
                feature_frames['confidence_scores'] = confidence_scores_df.shift(1)
                feature_metadata['confidence_scores'] = {'max_lag': 1}

            validation_results = self._validate_downstream_compatibility(
                mapped_labels,
                horizon_weights,
                target_columns,
                target_parameters=target_parameters,
                target_shifts=target_shifts,
                feature_frames=feature_frames if feature_frames else None,
                feature_metadata=feature_metadata if feature_metadata else None,
            )
            validation_results['walk_forward_cv'] = walk_forward_summary
            validation_results['walk_forward_folds'] = len(walk_forward_summary)
            if not validation_results['is_valid']:
                tprint_warning(f"⚠️ Downstream compatibility issues detected: {validation_results['issues']}")
            else:
                tprint_success("✅ Data structure validated for downstream compatibility")

            # Build smoothing metadata for downstream consumers
            smoothing_metadata = self._build_smoothing_metadata(balanced_labeling_result)
            normalization_factors = copy.deepcopy(balanced_labeling_result.normalization_factors or {})
            execution_timing = copy.deepcopy(getattr(balanced_labeling_result, 'execution_timing', {}))

            # Validate engineered scoring frames
            if isinstance(confidence_scores_df, pd.DataFrame) and not confidence_scores_df.empty:
                confidence_scores_df = validate_engineered_features(
                    confidence_scores_df,
                    context="multi_horizon_profit_labeler.confidence_scores"
                )
                alignment_metadata = enforce_feature_temporal_alignment(
                    confidence_scores_df,
                    context="multi_horizon_profit_labeler.confidence_scores",
                    target_shifts=target_shifts,
                    feature_metadata=feature_metadata.get('confidence_scores') if feature_metadata else None,
                )
                confidence_scores_df.attrs.setdefault('temporal_alignment', alignment_metadata)

            # Create enhanced artifacts with comprehensive metadata for downstream components
            artifacts = {
                'multi_horizon_labeling_result': {
                    'labeled_data': mapped_labels,  # This is what feature lookback optimization expects
                    'labels': mapped_labels,  # Backward compatibility
                    'confidence_scores': confidence_scores_df,
                    'eligibility_masks': balanced_labeling_result.eligibility_masks,
                    'sigma_payoffs': balanced_labeling_result.sigma_payoffs,
                    'quality_scores': balanced_labeling_result.quality_scores,
                    'horizon_weights': horizon_weights,  # New: weights for different horizons
                    'target_columns': target_columns,    # New: target columns for optimization
                    'target_parameters': target_parameters,
                    'target_shifts': target_shifts,
                    'normalization_factors': normalization_factors,
                    'execution_timing': execution_timing,
                    'method': 'multi_horizon_profit_labeling',
                    'balancing_applied': self.config.enable_label_balancing or self.config.enable_sample_weighting,
                    'sample_weights': getattr(balanced_labeling_result, 'sample_weights', None),  # Sample weights for training
                    'validation_results': validation_results,  # Downstream compatibility validation
                    'walk_forward_summary': walk_forward_summary,
                    'smoothing_settings': smoothing_metadata['settings'],
                    'metadata': {
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': timeframe,
                        'regime_aware': self.config.enable_regime_aware_labeling and regime_data is not None,
                        'processing_time': balanced_labeling_result.processing_time,
                        'n_samples': balanced_labeling_result.n_samples,
                        'n_targets': balanced_labeling_result.n_targets,
                        'n_horizons': balanced_labeling_result.n_horizons,
                        'target_distribution': self._calculate_target_distribution(mapped_labels),
                        'quality_summary': self._summarize_quality_scores(balanced_labeling_result.quality_scores),
                        'downstream_ready': validation_results['is_valid'],
                        'forward_return_smoothing': smoothing_metadata,
                        'target_shifts': target_shifts,
                        'min_target_shift': min(target_shifts.values()) if target_shifts else None,
                        'execution_timing': execution_timing,
                        'walk_forward_folds': len(walk_forward_summary),
                    },
                    'market_data': market_data,
                    'market_data_batches': tuple(market_data_batches),
                },
                'labeling_report': report,
                'standardized_output': {  # New: standardized format for all downstream steps
                    'labels': mapped_labels,
                    'weights': horizon_weights,
                    'target_columns': target_columns,
                    'quality_scores': balanced_labeling_result.quality_scores,
                    'confidence_scores': balanced_labeling_result.confidence_scores,
                    'eligibility_masks': balanced_labeling_result.eligibility_masks,
                    'sigma_payoffs': balanced_labeling_result.sigma_payoffs,
                    'sample_weights': getattr(balanced_labeling_result, 'sample_weights', None),
                    'normalization_factors': normalization_factors,
                    'validation_results': validation_results,
                    'walk_forward_summary': walk_forward_summary,
                    'smoothing_settings': smoothing_metadata['settings'],
                    'metadata': {
                        'source_component': 'multi_horizon_profit_labeler',
                        'creation_time': datetime.now().isoformat(),
                        'pipeline_ready': validation_results['is_valid'],
                        'downstream_compatibility': validation_results,
                        'forward_return_smoothing': smoothing_metadata,
                        'execution_timing': execution_timing,
                        'walk_forward_folds': len(walk_forward_summary),
                    }
                }
            }

            try:
                artifacts['multi_horizon_labeling_result'] = validate_multi_horizon_labeling_result(
                    artifacts['multi_horizon_labeling_result'],
                    context='multi_horizon_profit_labeler.artifacts.multi_horizon_labeling_result',
                )
            except DataContractValidationError as contract_error:
                tprint_error(f"❌ Data contract validation error: {contract_error}")
                raise

            validation_metadata = {
                'inputs': {
                    'market_data': schema_metadata('raw_ohlcv').get('raw_ohlcv')
                },
                'outputs': {
                    'labeled_data': schema_metadata('labeled_dataset').get('labeled_dataset')
                },
                'derived': {}
            }

            if isinstance(confidence_scores_df, pd.DataFrame) and not confidence_scores_df.empty:
                validation_metadata['derived']['confidence_scores'] = schema_metadata('engineered_features').get('engineered_features')

            artifacts.setdefault('validated_schemas', validation_metadata)
            mh_metadata = artifacts['multi_horizon_labeling_result'].setdefault('metadata', {})
            mh_metadata['validated_schemas'] = validation_metadata
            std_metadata = artifacts.get('standardized_output', {}).setdefault('metadata', {})
            std_metadata['validated_schemas'] = validation_metadata

            hypothesis_statistics = report_hypothesis_count(self.config)
            mh_metadata.setdefault('hypothesis_statistics', hypothesis_statistics)
            std_metadata.setdefault('hypothesis_statistics', dict(hypothesis_statistics))

            mh_metadata = artifacts['multi_horizon_labeling_result'].setdefault('metadata', {})
            if thresholds:
                mh_metadata.setdefault('quality_thresholds', thresholds)
            if mapping_metrics:
                mh_metadata.setdefault('quality_metrics', {})['mapping'] = mapping_metrics.get(
                    'labels_df for mapping',
                    mapping_metrics,
                )

            tprint_success(f"✅ Comprehensive artifacts structure created with {len(artifacts)} main sections")

            tprint_success(f"✅ Multi-horizon labeling completed for {symbol}")
            tprint_info(f"   → Samples: {labeling_result.n_samples}")
            tprint_info(f"   → Targets: {labeling_result.n_targets}")
            tprint_info(f"   → Processing time: {labeling_result.processing_time:.2f}s")

            return artifacts

        except Exception as e:
            tprint_error(f"❌ Multi-horizon labeling failed: {e}")
            import traceback
            tprint_error(f"🔍 Error details: {traceback.format_exc()}")

            error_smoothing_metadata = self._build_smoothing_metadata()

            # Create error artifacts with proper structure for downstream components
            error_artifacts = {
                'multi_horizon_labeling_result': {
                    'labeled_data': pd.DataFrame(),
                    'labels': pd.DataFrame(),
                    'confidence_scores': pd.DataFrame(),
                    'eligibility_masks': pd.DataFrame(),
                    'sigma_payoffs': pd.DataFrame(),
                    'quality_scores': {},
                    'horizon_weights': {},
                    'target_columns': [],
                    'normalization_factors': {},
                    'method': 'multi_horizon_profit_labeling',
                    'balancing_applied': False,
                    'sample_weights': None,
                    'validation_results': {'is_valid': False, 'issues': [f'Labeling failed: {str(e)}']},
                    'smoothing_settings': error_smoothing_metadata['settings'],
                    'metadata': {
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': timeframe,
                        'regime_aware': False,
                        'processing_time': 0.0,
                        'n_samples': 0,
                        'n_targets': 0,
                        'n_horizons': 0,
                        'target_distribution': {},
                        'quality_summary': {},
                        'downstream_ready': False,
                        'forward_return_smoothing': error_smoothing_metadata,
                        'error': str(e),
                        'error_traceback': traceback.format_exc()
                    }
                },
                'labeling_report': {
                    'status': 'failed',
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'timestamp': datetime.now().isoformat()
                },
                'standardized_output': {
                    'labels': pd.DataFrame(),
                    'weights': {},
                    'target_columns': [],
                    'quality_scores': {},
                    'confidence_scores': pd.DataFrame(),
                    'eligibility_masks': pd.DataFrame(),
                    'sigma_payoffs': pd.DataFrame(),
                    'sample_weights': None,
                    'normalization_factors': {},
                    'validation_results': {'is_valid': False, 'issues': [f'Labeling failed: {str(e)}']},
                    'smoothing_settings': error_smoothing_metadata['settings'],
                    'metadata': {
                        'source_component': 'multi_horizon_profit_labeler',
                        'creation_time': datetime.now().isoformat(),
                        'pipeline_ready': False,
                        'downstream_compatibility': {'is_valid': False, 'issues': [f'Labeling failed: {str(e)}']},
                        'forward_return_smoothing': error_smoothing_metadata,
                        'error': str(e)
                    }
                }
            }
            
            tprint_error("❌ Error artifacts created for downstream components")
            return error_artifacts

    def _validate_downstream_compatibility(
        self,
        labels_df: pd.DataFrame,
        horizon_weights: Dict[str, float],
        target_columns: List[str],
        *,
        target_parameters: Optional[Dict[str, Dict[str, Any]]] = None,
        target_shifts: Optional[Dict[str, int]] = None,
        feature_frames: Optional[Dict[str, pd.DataFrame]] = None,
        feature_metadata: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> Dict[str, Any]:
        """
        Validate that the labeling results are compatible with downstream components.
        
        Args:
            labels_df: DataFrame with labels
            horizon_weights: Dictionary of horizon weights
            target_columns: List of target column names
            
        Returns:
            Dictionary with validation results
        """
        try:
            tprint_info("🔍 Validating downstream compatibility...")
            
            issues = []
            is_valid = True
            
            # Check if we have labels
            if labels_df is None or labels_df.empty:
                issues.append("No labels available")
                is_valid = False
            else:
                tprint_info(f"✅ Labels available: {len(labels_df)} rows, {len(labels_df.columns)} columns")
            
            # Check if we have target columns
            if not target_columns:
                issues.append("No target columns identified")
                is_valid = False
            else:
                tprint_info(f"✅ Target columns identified: {target_columns}")
                
                # Check if target columns exist in labels
                missing_targets = [col for col in target_columns if col not in labels_df.columns]
                if missing_targets:
                    issues.append(f"Missing target columns in labels: {missing_targets}")
                    is_valid = False
                else:
                    tprint_info("✅ All target columns present in labels")
                    try:
                        assert_labels_sigma_scaled(labels_df[target_columns])
                        tprint_success("✅ Target labels confirmed to be σ-normalized")
                    except ValueError as scaling_error:
                        issues.append(str(scaling_error))
                        is_valid = False
                        tprint_warning(f"⚠️ Label scaling validation failed: {scaling_error}")

            # Check if we have horizon weights
            if not horizon_weights:
                issues.append("No horizon weights calculated")
                is_valid = False
            else:
                tprint_info(f"✅ Horizon weights available: {horizon_weights}")

            min_required_shift = 1
            recorded_shifts: List[int] = []
            if target_shifts:
                recorded_shifts = [int(shift) for shift in target_shifts.values() if shift is not None]
            elif target_parameters:
                recorded_shifts = [
                    int(params.get('target_shift', 1))
                    for params in target_parameters.values()
                    if isinstance(params, dict)
                ]

            if recorded_shifts:
                min_required_shift = max(1, min(recorded_shifts))
                if min_required_shift < 1:
                    issues.append("Target metadata reports non-positive target_shift")
                    is_valid = False
                else:
                    tprint_info(f"✅ Minimum target shift recorded: {min_required_shift}")
            else:
                tprint_warning("⚠️ Target shift metadata unavailable; assuming minimum shift of 1")

            if feature_metadata:
                for name, metadata in feature_metadata.items():
                    reported_lag = int(metadata.get('max_lag', 0)) if metadata else 0
                    if reported_lag < 1:
                        issues.append(f"Feature '{name}' reports max_lag < 1")
                        is_valid = False
                    elif reported_lag < min_required_shift:
                        issues.append(
                            f"Feature '{name}' reports max_lag {reported_lag} < required shift {min_required_shift}"
                        )
                        is_valid = False
                    else:
                        tprint_info(
                            f"✅ Feature '{name}' metadata passes lag check (max_lag={reported_lag})"
                        )

            if feature_frames:
                for name, frame in feature_frames.items():
                    if not isinstance(frame, pd.DataFrame) or frame.empty:
                        continue
                    leading_window = frame.iloc[:min_required_shift]
                    if not leading_window.isna().all().all():
                        issues.append(
                            f"Feature frame '{name}' contains non-null values within the first {min_required_shift} rows"
                        )
                        is_valid = False
                    else:
                        tprint_info(
                            f"✅ Feature frame '{name}' contains no contemporaneous values in the first {min_required_shift} rows"
                        )

            # Check data quality
            if not labels_df.empty:
                # Check for sufficient non-null values
                for col in target_columns:
                    if col in labels_df.columns:
                        non_null_count = labels_df[col].notna().sum()
                        total_count = len(labels_df)
                        null_ratio = 1 - (non_null_count / total_count) if total_count > 0 else 1.0
                        
                        if null_ratio > 0.5:  # More than 50% null values
                            issues.append(f"High null ratio in target '{col}': {null_ratio:.2%}")
                            is_valid = False
                        else:
                            tprint_info(f"✅ Target '{col}' has good data quality: {null_ratio:.2%} null values")
            
            validation_result = {
                'is_valid': is_valid,
                'issues': issues,
                'labels_shape': labels_df.shape if not labels_df.empty else (0, 0),
                'target_columns_count': len(target_columns),
                'horizon_weights_count': len(horizon_weights),
                'data_quality_score': 1.0 - (len(issues) / 10.0),  # Simple quality score
                'min_target_shift': min_required_shift,
            }
            
            if is_valid:
                tprint_success("✅ Downstream compatibility validation passed")
            else:
                tprint_warning(f"⚠️ Downstream compatibility validation failed: {len(issues)} issues")
            
            return validation_result
            
        except Exception as e:
            tprint_error(f"❌ Error during downstream compatibility validation: {e}")
            return {
                'is_valid': False,
                'issues': [f'Validation error: {str(e)}'],
                'labels_shape': (0, 0),
                'target_columns_count': 0,
                'horizon_weights_count': 0,
                'data_quality_score': 0.0
            }

    async def _apply_balancing_and_weighting(self, labeling_result: LabelingResult,
                                           market_data: pd.DataFrame,
                                           regime_data: Optional[Dict[str, Any]] = None) -> LabelingResult:
        """
        Apply label balancing and sample weighting to the labeling result.

        Args:
            labeling_result: Original labeling result
            market_data: Market data used for labeling
            regime_data: Optional regime data

        Returns:
            LabelingResult with balanced and weighted labels
        """
        if not self.balancing_system:
            tprint_info("ℹ️ Balancing system not available, returning original labels")
            return labeling_result

        try:
            tprint_info("⚖️ Applying label balancing and sample weighting...")
            tprint_info(f"   → Original samples: {labeling_result.n_samples}")
            tprint_info(f"   → Original targets: {labeling_result.n_targets}")

            # Prepare data for balancing
            # Use market data as features (exclude target columns and metadata)
            exclude_cols = filter_namespace_columns(market_data.columns, ColumnNamespace.TARGET)
            exclude_cols.extend(['sample_weight', 'timestamp'])

            feature_cols = [col for col in market_data.columns if col not in exclude_cols]
            X = market_data[feature_cols]

            # Extract targets from labeling result
            if labeling_result.labels is not None and not labeling_result.labels.empty:
                # Use the first target column for balancing (can be extended for multi-target)
                target_cols = filter_namespace_columns(labeling_result.labels.columns, ColumnNamespace.TARGET)
                if not target_cols:
                    target_cols = filter_namespace_columns(labeling_result.labels.columns, ColumnNamespace.LABEL)
                if target_cols:
                    y = labeling_result.labels[target_cols[0]]
                else:
                    # Fallback: create a simple target from the first column
                    fallback_col = labeling_result.labels.columns[0]
                    namespaced = ensure_namespace(fallback_col, ColumnNamespace.TARGET)
                    if fallback_col != namespaced:
                        labeling_result.labels = labeling_result.labels.rename(columns={fallback_col: namespaced})
                        fallback_col = namespaced
                    y = labeling_result.labels[fallback_col] if not labeling_result.labels.empty else pd.Series()
            else:
                tprint_warning("⚠️ No labels available for balancing")
                return labeling_result

            # Extract existing sample weights if available
            sample_weight = market_data.get('sample_weight')

            # Prepare additional features for weighting
            additional_features = {}

            # Add regime information if available
            if regime_data and 'regime_data' in regime_data:
                regime_info = regime_data['regime_data']
                if 'regime_states' in regime_info and len(regime_info['regime_states']) == len(market_data):
                    additional_features['regime'] = pd.Series(regime_info['regime_states'], index=market_data.index)

            # Add volatility information if available in market data
            volatility_cols = [col for col in market_data.columns if 'volatility' in col.lower()]
            if volatility_cols:
                additional_features['volatility'] = market_data[volatility_cols[0]]

            # Apply balancing and weighting
            tprint_warning("⚠️ IMPORTANT: Balancing is applied to the entire dataset. In production, apply balancing "
                          "separately to train/validation splits to avoid data leakage during cross-validation.")
            tprint_info("🔄 Executing balancing algorithm...")
            X_balanced, y_balanced, final_weights = self.balancing_system.balance_and_weight(
                X, y, sample_weight, additional_features
            )

            # Validate balancing results
            if len(y_balanced) == 0:
                tprint_warning("⚠️ Balancing resulted in empty dataset, returning original labels")
                return labeling_result
            
            if len(y_balanced) < 100:  # Minimum threshold for reliable analysis
                tprint_warning(f"⚠️ Balancing resulted in very small dataset ({len(y_balanced)} samples), proceeding with caution")

            # Create new labeling result with balanced data
            balanced_result = LabelingResult(
                labels=pd.DataFrame({target_cols[0]: y_balanced}, index=y_balanced.index),
                confidence_scores=labeling_result.confidence_scores,
                eligibility_masks=labeling_result.eligibility_masks,
                sigma_payoffs=pd.DataFrame(),
                training_labels=pd.DataFrame({target_cols[0]: y_balanced}, index=y_balanced.index),
                normalization_factors=labeling_result.normalization_factors.copy(),
                quality_scores=labeling_result.quality_scores,
                n_samples=len(y_balanced),
                n_targets=labeling_result.n_targets,
                processing_time=labeling_result.processing_time
            )

            balanced_result.smoothing_settings = getattr(labeling_result, 'smoothing_settings', {})
            if hasattr(labeling_result, 'execution_timing'):
                balanced_result.execution_timing = copy.deepcopy(getattr(labeling_result, 'execution_timing'))
            if not labeling_result.sigma_payoffs.empty:
                balanced_result.sigma_payoffs = labeling_result.sigma_payoffs.reindex(y_balanced.index)

            balanced_result.multi_target_result = getattr(labeling_result, 'multi_target_result', None)
            balanced_result.target_shifts = getattr(labeling_result, 'target_shifts', {})
            balanced_result.target_parameters = getattr(labeling_result, 'target_parameters', {})

            if balanced_result.normalization_factors:
                normalization_factors = copy.deepcopy(balanced_result.normalization_factors)
                if 'sigma_payoffs' in normalization_factors and isinstance(normalization_factors['sigma_payoffs'], pd.DataFrame):
                    normalization_factors['sigma_payoffs'] = normalization_factors['sigma_payoffs'].reindex(y_balanced.index)
                raw_payoffs = normalization_factors.get('raw_payoffs')
                if isinstance(raw_payoffs, pd.DataFrame):
                    normalization_factors['raw_payoffs'] = raw_payoffs.reindex(y_balanced.index)
                balanced_result.normalization_factors = normalization_factors

            # Add balancing metadata
            balanced_result.balancing_applied = True
            balanced_result.original_samples = labeling_result.n_samples
            balanced_result.balanced_samples = len(y_balanced)
            balanced_result.sample_weights = final_weights

            # Calculate balancing statistics
            original_distribution = y.value_counts().to_dict() if not y.empty else {}
            balanced_distribution = y_balanced.value_counts().to_dict()
            
            tprint_success(f"✅ Balancing completed: {labeling_result.n_samples} → {len(y_balanced)} samples")
            tprint_info(f"📊 Original class distribution: {original_distribution}")
            tprint_info(f"📊 Balanced class distribution: {balanced_distribution}")
            
            # Check if balancing improved class balance
            if original_distribution and balanced_distribution:
                original_balance = min(original_distribution.values()) / max(original_distribution.values()) if max(original_distribution.values()) > 0 else 0
                balanced_balance = min(balanced_distribution.values()) / max(balanced_distribution.values()) if max(balanced_distribution.values()) > 0 else 0

                if balanced_balance > original_balance:
                    tprint_success(f"✅ Class balance improved: {original_balance:.3f} → {balanced_balance:.3f}")
                else:
                    tprint_warning(f"⚠️ Class balance may have worsened: {original_balance:.3f} → {balanced_balance:.3f}")

            return self._apply_namespace_conventions(balanced_result)

        except Exception as e:
            tprint_warning(f"⚠️ Balancing failed: {e}, returning original labels")
            tprint_info("🔍 This is not critical - downstream components can work with unbalanced data")
            return labeling_result

    async def _load_market_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        *,
        batch_size: Optional[int] = None,
        window_days: Optional[int] = None,
    ) -> Iterable[pd.DataFrame]:
        """Load market data for the specified symbol and timeframe.

        Returns an iterable of DataFrame batches to support streaming
        consumption by downstream components.
        """

        if get_klines_manager is None:
            message = (
                "kline_parquet utilities are not available. "
                "Ensure src.utils.data.klines_parquet can be imported."
            )
            self.logger.error(message)
            tprint_error(f"❌ {message}")
            raise RuntimeError(message)

        tprint_info(f"📊 Loading market data for {symbol} {timeframe} from {data_dir}")

        manager = get_klines_manager(data_dir)

        symbol_variants = list(dict.fromkeys([symbol, symbol.upper(), symbol.lower()]))
        timeframe_variants = list(dict.fromkeys([timeframe, timeframe.lower(), timeframe.upper()]))
        data_type_variants = ["processed", "raw"]

        load_errors: List[str] = []

        for sym in symbol_variants:
            for tf in timeframe_variants:
                for data_type in data_type_variants:
                    streamed = False
                    async for batch in self._stream_market_data_batches(
                        manager,
                        sym,
                        tf,
                        data_type,
                        batch_size=batch_size,
                        window_days=window_days,
                        load_errors=load_errors,
                    ):
                        streamed = True
                        yield batch
                    if streamed:
                        return

        error_message = (
            f"No market data available for {symbol} on {exchange} with timeframe {timeframe}."
        )
        if load_errors:
            for msg in load_errors[-5:]:  # Log the most recent errors for context
                self.logger.error(msg)
        self.logger.error(error_message)
        tprint_error(f"❌ {error_message}")
        raise FileNotFoundError(error_message)

    async def _stream_market_data_batches(
        self,
        manager,
        symbol: str,
        timeframe: str,
        data_type: str,
        *,
        batch_size: Optional[int],
        window_days: Optional[int],
        load_errors: List[str],
    ) -> AsyncIterator[pd.DataFrame]:
        """Yield prepared market data batches for the requested parameters."""

        tprint_info(
            f"🔍 Attempting klines_parquet load for {symbol}/{timeframe} [{data_type}]"
        )

        try:
            if window_days:
                async for chunk in self._stream_by_date_window(
                    manager,
                    symbol,
                    timeframe,
                    data_type,
                    window_days,
                    batch_size,
                    load_errors,
                ):
                    yield chunk
                return

            raw_df = await asyncio.to_thread(
                manager.read_data,
                symbol,
                timeframe,
                None,
                None,
                data_type,
            )
        except Exception as load_error:  # pragma: no cover - defensive guard
            error_msg = (
                f"Failed to load {symbol}/{timeframe} ({data_type}) via klines_parquet: {load_error}"
            )
            self.logger.warning(error_msg)
            load_errors.append(error_msg)
            return

        if raw_df is None or raw_df.empty:
            info_msg = (
                f"klines_parquet returned no data for {symbol}/{timeframe} ({data_type})"
            )
            self.logger.info(info_msg)
            load_errors.append(info_msg)
            return

        try:
            prepared = self._prepare_market_data_frame(raw_df)
        except Exception as prep_error:
            prep_msg = (
                f"Loaded data for {symbol}/{timeframe} ({data_type}) could not be prepared: {prep_error}"
            )
            self.logger.warning(prep_msg)
            load_errors.append(prep_msg)
            return

        tprint_success(
            f"✅ Loaded {len(prepared)} rows via klines_parquet for {symbol} {timeframe}"
        )

        for chunk in self._split_market_data_batches(prepared, batch_size=batch_size):
            yield chunk

    async def _stream_by_date_window(
        self,
        manager,
        symbol: str,
        timeframe: str,
        data_type: str,
        window_days: int,
        batch_size: Optional[int],
        load_errors: List[str],
    ) -> AsyncIterator[pd.DataFrame]:
        """Yield market data batches by iterating over date windows."""

        date_info = None
        try:
            date_info = manager.get_data_info(symbol, timeframe, data_type)
        except Exception as info_error:  # pragma: no cover - defensive guard
            self.logger.debug(f"ℹ️ Could not retrieve data info for streaming: {info_error}")

        start_date, end_date = self._extract_date_range(date_info)
        if not start_date or not end_date:
            return

        current_start = start_date
        while current_start < end_date:
            current_end = min(current_start + pd.Timedelta(days=window_days), end_date)
            try:
                raw_df = await asyncio.to_thread(
                    manager.read_data,
                    symbol,
                    timeframe,
                    current_start.to_pydatetime(),
                    current_end.to_pydatetime(),
                    data_type,
                )
            except Exception as load_error:  # pragma: no cover - defensive guard
                error_msg = (
                    f"Failed to load {symbol}/{timeframe} ({data_type}) for {current_start}→{current_end}: {load_error}"
                )
                self.logger.warning(error_msg)
                load_errors.append(error_msg)
                current_start = current_end
                continue

            if raw_df is None or raw_df.empty:
                current_start = current_end
                continue

            try:
                prepared = self._prepare_market_data_frame(raw_df)
            except Exception as prep_error:
                prep_msg = (
                    f"Window {current_start}→{current_end} for {symbol}/{timeframe} could not be prepared: {prep_error}"
                )
                self.logger.warning(prep_msg)
                load_errors.append(prep_msg)
                current_start = current_end
                continue

            for chunk in self._split_market_data_batches(prepared, batch_size=batch_size):
                yield chunk

            current_start = current_end

    def _split_market_data_batches(
        self,
        data: pd.DataFrame,
        *,
        batch_size: Optional[int]
    ) -> Iterable[pd.DataFrame]:
        """Split a prepared market data frame into row-based batches."""

        if batch_size is None or batch_size <= 0 or len(data) <= batch_size:
            yield data.copy()
            return

        for start in range(0, len(data), batch_size):
            end = start + batch_size
            yield data.iloc[start:end].copy()

    def _extract_date_range(self, info: Optional[Dict[str, Any]]) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Parse the available date range from manager metadata."""

        if not info:
            return None, None

        date_range = info.get('date_range') if isinstance(info, dict) else None
        if not date_range:
            return None, None

        start_value = None
        end_value = None

        if isinstance(date_range, dict):
            start_value = date_range.get('start') or date_range.get('from')
            end_value = date_range.get('end') or date_range.get('to')
        elif isinstance(date_range, (list, tuple)) and len(date_range) >= 2:
            start_value, end_value = date_range[0], date_range[1]

        if not start_value or not end_value:
            return None, None

        start_ts = pd.to_datetime(start_value, utc=True, errors='coerce')
        end_ts = pd.to_datetime(end_value, utc=True, errors='coerce')

        if pd.isna(start_ts) or pd.isna(end_ts):
            return None, None

        if start_ts.tzinfo is not None:
            start_ts = start_ts.tz_convert(None)
        if end_ts.tzinfo is not None:
            end_ts = end_ts.tz_convert(None)

        return start_ts, end_ts

    def _prepare_market_data_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure loaded market data is indexed and typed as expected by the labeler."""
        if data is None or data.empty:
            raise ValueError("Loaded market data is empty")

        df = data.copy()

        if "timestamp" in df.columns:
            timestamp_series = df.pop("timestamp")
        elif "open_time" in df.columns:
            timestamp_series = df.pop("open_time")
        elif df.index.name == "timestamp":
            timestamp_series = df.index
        else:
            timestamp_series = df.index

        ts = pd.to_datetime(timestamp_series, utc=True, errors="coerce")
        if ts.isnull().any():
            # Try integer timestamps (milliseconds/seconds)
            numeric_ts = pd.to_numeric(timestamp_series, errors="coerce")
            if numeric_ts.notnull().all():
                unit = "ms" if numeric_ts.max() > 10**12 else "s"
                ts = pd.to_datetime(numeric_ts, unit=unit, utc=True, errors="coerce")
        if ts.isnull().all():
            raise ValueError("Unable to parse timestamps for market data")

        ts_index = pd.DatetimeIndex(ts)

        valid_mask = ~pd.isna(ts_index)
        if not valid_mask.all():
            df = df.loc[valid_mask]
            ts_index = ts_index[valid_mask]
        if ts_index.empty:
            raise ValueError("Market data contains no valid timestamps")

        if ts_index.tz is not None:
            ts_index = ts_index.tz_convert(None)
        else:
            ts_index = ts_index.tz_localize(None)

        df.index = ts_index

        # Normalize column names
        normalized_columns = {col: col.lower() for col in df.columns}
        df = df.rename(columns=normalized_columns)

        volume_candidates = [
            "volume",
            "volume_usdt",
            "quote_volume",
            "vol",
        ]
        if "volume" not in df.columns:
            for candidate in volume_candidates:
                if candidate in df.columns:
                    df["volume"] = df.pop(candidate)
                    break

        required_columns = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Market data missing required columns: {missing}")

        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=required_columns)
        if df.empty:
            raise ValueError("Market data contains no valid OHLCV rows after cleaning")

        return df

    async def _execute_regime_aware_labeling(self, market_data: pd.DataFrame, regime_data: Dict[str, Any]) -> LabelingResult:
        """
        Execute regime-aware labeling that creates differentiated labels for different regimes.

        Args:
            market_data: Market data
            regime_data: Regime data from regime_data_splitting

        Returns:
            LabelingResult with regime-differentiated labels
        """
        try:
            tprint_info("🎭 Executing regime-aware labeling")

            # Extract regime assignments from regime data
            regime_assignments = self._extract_regime_assignments(market_data, regime_data)
            if regime_assignments is None:
                tprint_warning("⚠️ No regime assignments found, falling back to standard labeling")
                return self.volatility_labeler.generate_labels(market_data)

            # Get unique regimes
            regimes = np.unique(regime_assignments[~pd.isna(regime_assignments)])
            tprint_info(f"📊 Found {len(regimes)} distinct regimes")

            if len(regimes) == 0:
                tprint_warning("⚠️ No valid regime assignments, falling back to standard labeling")
                return self.volatility_labeler.generate_labels(market_data)

            # Create regime-specific labels using the volatility-aware labeler for each regime
            regime_labels = {}
            regime_quality_scores = {}
            regime_execution_timing: Dict[str, Dict[str, Any]] = {}
            regime_sigma_payoffs = {}
            regime_normalization_factors = {}
            total_processing_time = 0.0

            for regime in regimes:
                tprint_info(f"🏷️ Processing regime {regime}")

                # Filter data for this regime
                regime_mask = regime_assignments == regime
                regime_data_subset = market_data[regime_mask].copy()

                if len(regime_data_subset) < self.config.min_data_points:
                    tprint_warning(f"⚠️ Insufficient data for regime {regime}: {len(regime_data_subset)} samples")
                    continue

                # Generate labels for this regime using the volatility-aware labeler
                regime_result = self.volatility_labeler.generate_labels(regime_data_subset)

                if not regime_result.labels.empty:
                    # Add regime suffix to column names to differentiate between regimes
                    regime_labels[regime] = regime_result.labels.add_suffix(f'_regime_{regime}')
                    if not regime_result.sigma_payoffs.empty:
                        regime_sigma_payoffs[regime] = regime_result.sigma_payoffs.add_suffix(f'_regime_{regime}')
                    if regime_result.normalization_factors:
                        regime_normalization_factors[regime] = regime_result.normalization_factors
                    regime_quality_scores.update({
                        f"{target}_regime_{regime}": quality_score
                        for target, quality_score in regime_result.quality_scores.items()
                    })
                    if hasattr(regime_result, 'execution_timing'):
                        regime_execution_timing[regime] = copy.deepcopy(getattr(regime_result, 'execution_timing'))
                    total_processing_time += regime_result.processing_time

            # Combine regime-specific labels
            if regime_labels:
                combined_labels = pd.concat(regime_labels.values(), axis=1)
                combined_sigma_payoffs = pd.concat(regime_sigma_payoffs.values(), axis=1) if regime_sigma_payoffs else pd.DataFrame(index=combined_labels.index)
                combined_normalization = {'regimes': regime_normalization_factors}

                # Create combined result with proper metadata
                combined_result = LabelingResult(
                    labels=combined_labels,
                    confidence_scores=pd.DataFrame(index=combined_labels.index),
                    eligibility_masks=pd.DataFrame(index=combined_labels.index),
                    sigma_payoffs=combined_sigma_payoffs,
                    training_labels=combined_labels.copy(),
                    normalization_factors=combined_normalization,
                    quality_scores=regime_quality_scores,
                    n_samples=len(combined_labels),
                    n_targets=len([col for col in combined_labels.columns if 'target' in col]),
                    processing_time=total_processing_time
                )

                if regime_execution_timing:
                    # Use the first regime's execution timing as representative metadata
                    combined_result.execution_timing = copy.deepcopy(next(iter(regime_execution_timing.values())))

                tprint_success(f"✅ Regime-aware labeling completed for {len(regime_labels)} regimes")
                return combined_result
            else:
                tprint_warning("⚠️ No valid regime-specific labels generated, falling back to standard labeling")
                return self.volatility_labeler.generate_labels(market_data)

        except Exception as e:
            tprint_error(f"❌ Regime-aware labeling failed: {e}")
            # Fall back to standard labeling
            return self.volatility_labeler.generate_labels(market_data)

    def _extract_regime_assignments(self, market_data: pd.DataFrame, regime_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extract regime assignments from regime data.

        Args:
            market_data: Market data
            regime_data: Regime data from regime_data_splitting

        Returns:
            Array of regime assignments or None if not found
        """
        try:
            # Try to get regime assignments from regime data
            if 'regime_data' in regime_data:
                regime_info = regime_data['regime_data']

                # Check if regime states are directly available
                if 'regime_states' in regime_info:
                    regime_states = regime_info['regime_states']
                    if len(regime_states) == len(market_data):
                        return regime_states

                # Check if market data in regime data has regime column
                if 'market_data' in regime_info and regime_info['market_data'] is not None:
                    regime_market_data = regime_info['market_data']
                    if self.config.regime_column in regime_market_data.columns:
                        return regime_market_data[self.config.regime_column].values

            # Check if regime assignments are in the market data itself
            if self.config.regime_column in market_data.columns:
                return market_data[self.config.regime_column].values

            tprint_warning(f"⚠️ No regime assignments found in regime data or market data")
            return None

        except Exception as e:
            tprint_warning(f"⚠️ Error extracting regime assignments: {e}")
            return None

    def _map_target_columns_for_feature_optimization(
        self,
        labels_df: pd.DataFrame,
        duplicate_threshold: Optional[float] = None,
        quality_metrics: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Map target column names to expected names for feature lookback optimization compatibility.

        Feature lookback optimization expects specific target column names like:
        - 'leverage_adjusted_score'
        - 'immediate_opportunity'
        - 'short_term_opportunity'

        This method maps the generated target columns (like 'small_k0.50_a1.00') to these expected names.
        """
        try:
            if labels_df is None or labels_df.empty:
                tprint_warning("⚠️ No labels to map for feature optimization compatibility")
                return labels_df

            # Validate and prepare the DataFrame
            metrics_entry: Dict[str, Any] = {}
            if quality_metrics is not None:
                metrics_entry = quality_metrics.setdefault('labels_df for mapping', {})
            labels_df = validate_and_prepare_dataframe(
                labels_df,
                "labels_df for mapping",
                duplicate_threshold=duplicate_threshold,
                metrics=metrics_entry,
            )
            mapped_df = labels_df.copy()
            tprint_info(f"🔄 Mapping {len(labels_df.columns)} target columns for feature optimization compatibility")

            # Define mapping from generated column patterns to expected names
            column_mappings = {
                'leverage_adjusted_score': [],
                'immediate_opportunity': [],
                'short_term_opportunity': []
            }

            column_bases = {col: strip_namespace(col)[0] for col in labels_df.columns}

            # Priority 1: Map small band targets to immediate_opportunity (shortest horizon)
            # Handle both regular targets and regime-specific targets
            small_targets = [
                col for col, base in column_bases.items()
                if base.startswith('small_') and '_regime_' not in base
            ]
            small_regime_targets = [
                col for col, base in column_bases.items()
                if base.startswith('small_') and '_regime_' in base
            ]

            # Use regular targets first, then regime targets if no regular targets available
            if small_targets:
                best_small_target = self._select_best_target_by_pattern(small_targets, labels_df, 'small')
                if best_small_target:
                    column_mappings['immediate_opportunity'].append(best_small_target)
            elif small_regime_targets:
                # Use the first regime target (could be improved to select best regime)
                best_small_target = self._select_best_target_by_pattern(small_regime_targets, labels_df, 'small')
                if best_small_target:
                    column_mappings['immediate_opportunity'].append(best_small_target)

            # Priority 2: Map medium band targets to short_term_opportunity (medium horizon)
            medium_targets = [
                col for col, base in column_bases.items()
                if base.startswith('medium_') and '_regime_' not in base
            ]
            medium_regime_targets = [
                col for col, base in column_bases.items()
                if base.startswith('medium_') and '_regime_' in base
            ]

            if medium_targets:
                best_medium_target = self._select_best_target_by_pattern(medium_targets, labels_df, 'medium')
                if best_medium_target:
                    column_mappings['short_term_opportunity'].append(best_medium_target)
            elif medium_regime_targets:
                best_medium_target = self._select_best_target_by_pattern(medium_regime_targets, labels_df, 'medium')
                if best_medium_target:
                    column_mappings['short_term_opportunity'].append(best_medium_target)

            # Priority 3: Map high band targets to leverage_adjusted_score (longest horizon)
            high_targets = [
                col for col, base in column_bases.items()
                if base.startswith('high_') and '_regime_' not in base
            ]
            high_regime_targets = [
                col for col, base in column_bases.items()
                if base.startswith('high_') and '_regime_' in base
            ]

            if high_targets:
                best_high_target = self._select_best_target_by_pattern(high_targets, labels_df, 'high')
                if best_high_target:
                    column_mappings['leverage_adjusted_score'].append(best_high_target)
            elif high_regime_targets:
                best_high_target = self._select_best_target_by_pattern(high_regime_targets, labels_df, 'high')
                if best_high_target:
                    column_mappings['leverage_adjusted_score'].append(best_high_target)

            # Apply the mappings
            for expected_name, source_columns in column_mappings.items():
                if source_columns:
                    # Use the first (best) source column
                    source_col = source_columns[0]
                    if source_col in mapped_df.columns:
                        expected_col = expected_name
                        mapped_df[expected_col] = mapped_df[source_col]
                        tprint_info(f"✅ Mapped '{source_col}' → '{expected_col}'")

            tprint_info(f"✅ Target column mapping completed. Original: {len(labels_df.columns)}, Mapped: {len(mapped_df.columns)}")

            return mapped_df

        except Exception as e:
            tprint_warning(f"⚠️ Error mapping target columns: {e}")
            # Return original dataframe if mapping fails
            return labels_df

    def _select_best_target_by_pattern(self, target_columns: List[str], labels_df: pd.DataFrame, pattern: str) -> Optional[str]:
        """
        Select the best target column from a list of candidates based on pattern and data quality.

        Args:
            target_columns: List of column names matching the pattern
            labels_df: DataFrame with the labels
            pattern: Pattern type ('small', 'medium', 'high')

        Returns:
            Best column name or None if no suitable column found
        """
        try:
            if not target_columns:
                return None

            # For now, select the first target in the list
            # In a more sophisticated implementation, we could analyze label quality,
            # balance, predictability, etc. to select the best target
            selected_target = target_columns[0]

            # Validate that the selected target has reasonable data
            if selected_target in labels_df.columns:
                target_data = labels_df[selected_target].dropna()

                # Check if we have enough non-null values
                if len(target_data) > 100:  # Minimum threshold for reliable analysis
                    tprint_info(f"✅ Selected '{selected_target}' as best {pattern} target")
                    return selected_target

            tprint_warning(f"⚠️ No suitable {pattern} target found among {len(target_columns)} candidates")
            return None

        except Exception as e:
            tprint_warning(f"⚠️ Error selecting best target for pattern {pattern}: {e}")
            return target_columns[0] if target_columns else None

    def _calculate_horizon_weights(self, labeling_result: LabelingResult, labels_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate weights for different horizons based on target quality and balance.

        Args:
            labeling_result: The labeling result object
            labels_df: DataFrame with the labels

        Returns:
            Dictionary mapping horizon names to weights
        """
        try:
            tprint_info("⚖️ Calculating horizon weights based on target quality and balance")

            weights = {}

            # Get base weights from configuration
            base_weights = {
                'micro': self.config.horizon_weights.micro,
                'small': self.config.horizon_weights.small,
                'medium': self.config.horizon_weights.medium,
                'high': self.config.horizon_weights.high
            }
            tprint_info(f"📊 Using base horizon weights from config: {base_weights}")

            # Adjust weights based on quality scores if available
            if labeling_result.quality_scores:
                quality_scores = labeling_result.quality_scores

                aliases = {
                    'small': ('small_', 'immediate_opportunity', 'directional_confidence'),
                    'medium': ('medium_', 'short_term_opportunity'),
                    'high': ('high_', 'leverage_adjusted_score', 'overall_opportunity'),
                }

                column_bases = {col: strip_namespace(col)[0].lower() for col in labels_df.columns}

                def _match_targets(pattern: str) -> List[str]:
                    patterns = aliases.get(pattern, ())
                    return [
                        col
                        for col, base in column_bases.items()
                        if any(alias in base for alias in patterns)
                    ]

                # Find targets for each horizon pattern
                small_targets = _match_targets('small')
                medium_targets = _match_targets('medium')
                high_targets = _match_targets('high')

                # Calculate average quality for each horizon
                small_quality = self._calculate_average_quality(small_targets, quality_scores)
                medium_quality = self._calculate_average_quality(medium_targets, quality_scores)
                high_quality = self._calculate_average_quality(high_targets, quality_scores)

                # Normalize quality scores to weights
                total_quality = small_quality + medium_quality + high_quality
                if total_quality > 0:
                    weights['small'] = small_quality / total_quality * 0.6  # 60% max allocation
                    weights['medium'] = medium_quality / total_quality * 0.3  # 30% max allocation
                    weights['high'] = high_quality / total_quality * 0.2   # 20% max allocation
                else:
                    weights = base_weights.copy()
            else:
                weights = base_weights.copy()

            # Ensure minimum weights for active horizons
            for horizon in ['small', 'medium', 'high']:
                if weights.get(horizon, 0) < 0.1:
                    weights[horizon] = 0.1

            # Normalize to sum to 1.0
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}

            tprint_success(f"✅ Horizon weights calculated: {weights}")
            return weights

        except Exception as e:
            tprint_warning(f"⚠️ Error calculating horizon weights: {e}")
            return {'small': 0.5, 'medium': 0.3, 'high': 0.2}

    def _calculate_average_quality(self, target_columns: List[str], quality_scores: Dict[str, Any]) -> float:
        """Calculate average quality score for a list of targets."""
        if not target_columns or not quality_scores:
            return 0.0

        qualities = []
        for target in target_columns:
            base_name = strip_namespace(target)[0]
            quality_entry = quality_scores.get(target) or quality_scores.get(base_name)
            if quality_entry:
                quality = quality_entry
                if hasattr(quality, 'overall_quality'):
                    qualities.append(quality.overall_quality)
                elif isinstance(quality, dict) and 'overall_quality' in quality:
                    qualities.append(quality['overall_quality'])

        return np.mean(qualities) if qualities else 0.0

    def _extract_target_columns_for_optimization(self, labels_df: pd.DataFrame) -> List[str]:
        """
        Extract target columns that should be used for feature optimization.

        Args:
            labels_df: DataFrame with the labels

        Returns:
            List of target column names for optimization
        """
        try:
            tprint_info("🎯 Extracting target columns for feature optimization")

            target_columns = []
            column_bases = {col: strip_namespace(col)[0] for col in labels_df.columns}

            # Priority order for target selection
            priority_patterns = [
                'immediate_opportunity',  # Mapped immediate targets
                'short_term_opportunity', # Mapped short-term targets
                'leverage_adjusted_score', # Mapped leverage targets
                'small_',                 # Small horizon targets
                'medium_',                # Medium horizon targets
                'high_'                   # High horizon targets
            ]

            for pattern in priority_patterns:
                matching_columns = [
                    col for col, base in column_bases.items() if pattern in base
                ]
                if matching_columns:
                    # Select the best target for this pattern
                    if pattern in ['immediate_opportunity', 'short_term_opportunity', 'leverage_adjusted_score']:
                        target_columns.append(matching_columns[0])  # Use mapped targets directly
                    else:
                        # For pattern-based targets, select the best one
                        best_target = self._select_best_target_by_pattern(matching_columns, labels_df, pattern.replace('_', ''))
                        if best_target:
                            target_columns.append(best_target)

            # Remove duplicates while preserving order
            seen = set()
            unique_targets = []
            for target in target_columns:
                if target not in seen:
                    seen.add(target)
                    unique_targets.append(target)

            tprint_success(f"✅ Extracted {len(unique_targets)} target columns for optimization: {unique_targets}")
            return unique_targets

        except Exception as e:
            tprint_warning(f"⚠️ Error extracting target columns: {e}")
            # Fallback: return first few columns that look like targets
            target_like_columns = [
                col for col, base in column_bases.items()
                if any(token in base.lower() for token in ['target', 'opportunity', 'score'])
            ]
            return target_like_columns[:3] if target_like_columns else []

    def _calculate_target_distribution(self, labels_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate distribution statistics for targets."""
        try:
            if labels_df is None or labels_df.empty:
                return {}

            distribution = {}
            for col in labels_df.columns:
                if is_numeric_dtype(labels_df[col]):
                    values = labels_df[col].dropna()
                    if len(values) > 0:
                        distribution[col] = {
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'min': float(values.min()),
                            'max': float(values.max()),
                            'non_null_count': int(len(values)),
                            'class_balance': self._calculate_class_balance(values)
                        }

            return distribution

        except Exception as e:
            tprint_warning(f"⚠️ Error calculating target distribution: {e}")
            return {}

    def _summarize_walk_forward_folds(
        self,
        folds: Iterable[WalkForwardFold],
        labels: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """Summarize walk-forward folds with distribution statistics."""

        summaries: List[Dict[str, Any]] = []
        for fold in folds:
            mapping = fold.to_mapping()
            validation_index = mapping['validation'].index
            test_index = mapping['test'].index

            validation_labels = labels.reindex(validation_index)
            test_labels = labels.reindex(test_index)

            summaries.append(
                {
                    'fold': fold.fold,
                    'train_rows': len(mapping['train']),
                    'validation_rows': len(mapping['validation']),
                    'test_rows': len(mapping['test']),
                    'train_window': self._index_range(mapping['train'].index),
                    'validation_window': self._index_range(validation_index),
                    'test_window': self._index_range(test_index),
                    'validation_distribution': self._calculate_target_distribution(validation_labels),
                    'test_distribution': self._calculate_target_distribution(test_labels),
                }
            )

        return summaries

    @staticmethod
    def _index_range(index: pd.Index) -> Optional[Dict[str, Any]]:
        """Return start/end timestamps for a datetime index."""

        if not isinstance(index, pd.DatetimeIndex) or index.empty:
            return None

        return {
            'start': index[0].isoformat(),
            'end': index[-1].isoformat(),
        }

    def _calculate_class_balance(self, values: pd.Series) -> Dict[str, float]:
        """Calculate class balance for a target series."""
        try:
            if len(values) == 0:
                return {}

            # For continuous targets, calculate balance across quantiles
            quantiles = values.quantile([0.25, 0.5, 0.75])
            balance = {
                'positive_ratio': float((values > 0).mean()),
                'negative_ratio': float((values < 0).mean()),
                'zero_ratio': float((values == 0).mean()),
                'q25': float(quantiles[0.25]),
                'q50': float(quantiles[0.5]),
                'q75': float(quantiles[0.75])
            }

            return balance

        except Exception as e:
            return {}

    def _summarize_quality_scores(self, quality_scores: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize quality scores across all targets."""
        try:
            if not quality_scores:
                return {}

            summary = {
                'n_targets_with_quality': len(quality_scores),
                'average_quality': 0.0,
                'quality_range': {'min': 1.0, 'max': 0.0},
                'quality_distribution': {}
            }

            quality_values = []
            for target_name, quality in quality_scores.items():
                if hasattr(quality, 'overall_quality'):
                    q_val = quality.overall_quality
                elif isinstance(quality, dict) and 'overall_quality' in quality:
                    q_val = quality['overall_quality']
                else:
                    continue

                quality_values.append(q_val)
                summary['quality_range']['min'] = min(summary['quality_range']['min'], q_val)
                summary['quality_range']['max'] = max(summary['quality_range']['max'], q_val)

            if quality_values:
                summary['average_quality'] = float(np.mean(quality_values))

                # Quality distribution
                q25, q50, q75 = np.percentile(quality_values, [25, 50, 75])
                summary['quality_distribution'] = {
                    'q25': float(q25),
                    'median': float(q50),
                    'q75': float(q75)
                }

            return summary

        except Exception as e:
            tprint_warning(f"⚠️ Error summarizing quality scores: {e}")
            return {}

    def _build_smoothing_metadata(self, labeling_result: Optional[LabelingResult] = None) -> Dict[str, Any]:
        """Create a standardized smoothing metadata payload."""
        try:
            smoothing_cfg = getattr(
                self.volatility_labeler.config.multi_target,
                'forward_return_smoothing',
                None
            )

            settings = {}
            if labeling_result is not None:
                settings = getattr(labeling_result, 'smoothing_settings', {}) or {}

            enabled = bool(smoothing_cfg and getattr(smoothing_cfg, 'enabled', False))
            config_dict = asdict(smoothing_cfg) if smoothing_cfg else {}

            metadata = {
                'enabled': enabled,
                'method': 'ewm_halflife' if enabled else None,
                'aggregation': 'exponential_weighted_mean' if enabled else None,
                'settings': settings,
                'applied_columns': list(settings.keys()),
                'config': config_dict
            }

            if not settings:
                metadata['applied_columns'] = []

            return metadata

        except Exception as e:
            tprint_warning(f"⚠️ Error building smoothing metadata: {e}")
            return {
                'enabled': False,
                'method': None,
                'aggregation': None,
                'settings': {},
                'applied_columns': [],
                'config': {}
            }

    async def _generate_comprehensive_report(
        self,
        labeling_result: LabelingResult,
        symbol: str,
        exchange: str,
        timeframe: str,
        regime_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive labeling report with regime-aware analysis."""
        try:
            # Import the profit labeling report generator
            from src.training.steps.pre_training.profit_labeling.profit_labeling_report_generator import (
                ProfitLabelingReportGenerator, ProfitLabelingReport
            )

            tprint_info("📋 Generating comprehensive profit labeling report")

            # Create the report generator
            report_generator = ProfitLabelingReportGenerator()

            # Prepare the labeling result data for the report generator
            labeling_result_data = {
                'multi_horizon_labeling_result': {
                    'labeled_data': labeling_result.labels,
                    'confidence_scores': labeling_result.confidence_scores,
                    'eligibility_masks': labeling_result.eligibility_masks,
                    'quality_scores': labeling_result.quality_scores,
                    'metadata': {
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': timeframe,
                        'regime_aware': self.config.enable_regime_aware_labeling and regime_data is not None,
                        'processing_time': labeling_result.processing_time,
                        'n_samples': labeling_result.n_samples,
                        'n_targets': labeling_result.n_targets,
                        'n_horizons': labeling_result.n_horizons,
                        'balancing_applied': getattr(labeling_result, 'balancing_applied', False),
                        'original_samples': getattr(labeling_result, 'original_samples', labeling_result.n_samples),
                        'balanced_samples': getattr(labeling_result, 'balanced_samples', labeling_result.n_samples)
                    }
                },
                'labeling_report': self._generate_basic_labeling_report(labeling_result, symbol, exchange, timeframe)
            }

            # Generate the comprehensive report
            comprehensive_report = report_generator.generate_report(
                labeling_result=labeling_result_data,
                regime_data=regime_data,
                output_directory="profit_labeling_reports"
            )

            # Convert the report object to dictionary for pipeline compatibility
            report_dict = {
                'status': 'completed',
                'symbol': comprehensive_report.symbol,
                'exchange': comprehensive_report.exchange,
                'timeframe': comprehensive_report.timeframe,
                'timestamp': comprehensive_report.timestamp.isoformat(),
                'processing_time': comprehensive_report.processing_time,
                'statistics': {
                    'n_samples': comprehensive_report.n_samples,
                    'n_targets': comprehensive_report.n_targets,
                    'n_horizons': comprehensive_report.n_horizons,
                    'label_distribution': comprehensive_report.label_distribution,
                    'balancing_applied': getattr(labeling_result, 'balancing_applied', False),
                    'original_samples': getattr(labeling_result, 'original_samples', comprehensive_report.n_samples),
                    'balanced_samples': getattr(labeling_result, 'balanced_samples', comprehensive_report.n_samples)
                },
                'quality_scores': comprehensive_report.quality_scores,
                'regime_statistics': comprehensive_report.regime_statistics,
                'feature_lookback_compatibility': comprehensive_report.feature_lookback_compatibility,
                'recommendations': comprehensive_report.recommendations
            }

            tprint_success("✅ Comprehensive profit labeling report generated")
            return report_dict

        except Exception as e:
            tprint_warning(f"⚠️ Error generating comprehensive report: {e}")
            # Fall back to basic report generation
            return self._generate_basic_labeling_report(labeling_result, symbol, exchange, timeframe)

    def _generate_basic_labeling_report(
        self,
        labeling_result: LabelingResult,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """Generate basic labeling report as fallback."""
        try:
            report = {
                'status': 'completed',
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'processing_time': labeling_result.processing_time,
                'statistics': {
                    'n_samples': labeling_result.n_samples,
                    'n_targets': labeling_result.n_targets,
                    'n_horizons': labeling_result.n_horizons,
                    'label_distribution': labeling_result.label_distribution
                },
                'quality_summary': {}
            }

            # Add quality scores summary
            if labeling_result.quality_scores:
                quality_summary = {}
                for target_name, quality_score in labeling_result.quality_scores.items():
                    quality_summary[target_name] = {
                        'overall_quality': quality_score.overall_quality,
                        'predictability': quality_score.predictability,
                        'stability': quality_score.stability,
                        'balance': quality_score.balance,
                        'auc_mean': quality_score.auc_mean,
                        'class_balance': quality_score.class_balance
                    }
                report['quality_summary'] = quality_summary

            return report

        except Exception as e:
            tprint_warning(f"⚠️ Error generating basic report: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


class MultiHorizonProfitLabelerComponent(BasePreTrainingComponent):
    """
    Component wrapper for Multi-Horizon Profit Labeler.

    This component integrates with the pre-training pipeline and handles
    regime-aware profit labeling with proper error handling and reporting.
    """

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the multi-horizon profit labeler component."""
        super().__init__(config)
        self.labeler = None
        self.quality_thresholds: Dict[str, float] = {}

        # Create configuration from component config
        mh_config = MultiHorizonConfig()

        # Override with custom parameters if provided
        if config and config.custom_params:
            for key, value in config.custom_params.items():
                if hasattr(mh_config, key):
                    setattr(mh_config, key, value)
            thresholds = config.custom_params.get('quality_thresholds')
            if isinstance(thresholds, dict):
                self.quality_thresholds = thresholds

        self.labeler = MultiHorizonProfitLabeler(mh_config)
        if self.quality_thresholds:
            self.labeler.quality_thresholds = self.quality_thresholds

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['multi_horizon_labeling_result', 'labeling_report']

    async def execute(self, data: Any, pipeline_state: PipelineState) -> ComponentResult:
        """
        Execute multi-horizon profit labeling as a component.

        Args:
            data: Input data (typically None for this component)
            pipeline_state: Current pipeline state

        Returns:
            ComponentResult with labeling results
        """
        try:
            pipeline_state = PipelineState.ensure(pipeline_state)
            # Extract parameters from pipeline state
            symbol = pipeline_state.get('symbol', 'ETHUSDT')
            exchange = pipeline_state.get('exchange', 'binance')
            timeframe = pipeline_state.get('timeframe', '1h')  # Updated to 1h for analyst

            data_locator: Optional[PipelineDataLocator] = pipeline_state.get('data_locator')
            if data_locator:
                self.labeler.config.data_locator = data_locator
                self.labeler.pipeline_data_locator = data_locator
            data_dir_key = pipeline_state.get('data_dir_key', self.labeler.config.data_dir_key)
            outcomes_dir_key = pipeline_state.get('outcomes_dir_key', self.labeler.config.outcomes_dir_key)
            if data_locator:
                self.labeler.config.data_dir_key = data_dir_key
                self.labeler.config.outcomes_dir_key = outcomes_dir_key

            data_dir = pipeline_state.get('data_dir')
            if not data_dir and data_locator:
                data_dir = str(data_locator.data_path(data_dir_key))

            # Extract regime data from pipeline state if available
            regime_data = pipeline_state.get('regime_data_splitting_result')

            # Execute labeling with regime data if available
            labeling_result = await self.labeler.execute_labeling(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                regime_data=regime_data,
                quality_thresholds=self.quality_thresholds or pipeline_state.get('quality_thresholds')
            )
            validation_metadata = labeling_result.get('validated_schemas', {})

            # Save artifacts persistently for other components to use
            artifacts_saved = False
            artifact_save_error: Optional[str] = None
            artifact_digest: Optional[str] = None
            artifact_path: Optional[str] = None
            artifact_save_skipped = False
            save_report: Optional[SaveReport] = None
            try:
                artifacts_payload = dict(labeling_result)
                mh_payload = artifacts_payload.get('multi_horizon_labeling_result', {})
                artifacts_payload['multi_horizon_labeling_result'] = _ensure_labeling_contract(mh_payload)

                outcome_metadata = {
                    'component_type': 'multi_horizon_profit_labeler',
                    'random_seed': pipeline_state.get('random_seed'),
                }
                outcome_data = {
                    'config': {
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': timeframe,
                    },
                    'artifacts': artifacts_payload,
                    'metadata': outcome_metadata,
                }
                if validation_metadata:
                    outcome_data['metadata']['validated_schemas'] = validation_metadata

                outcomes_dir_value = pipeline_state.get('outcomes_dir')
                if outcomes_dir_value:
                    outcomes_dir = Path(outcomes_dir_value)
                elif data_locator:
                    outcomes_dir = data_locator.artifacts_path(outcomes_dir_key, ensure_exists=True)
                else:
                    outcomes_dir = self.labeler._settings.outcomes_root
                    outcomes_dir.mkdir(parents=True, exist_ok=True)

                artifact_base_name = 'market_analysis_multi_horizon_profit_labeler_outcome'
                save_report, artifact_save_skipped = _persist_labeling_outcome(
                    base_dir=Path(outcomes_dir),
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    outcome_payload=outcome_data,
                    logger=self.logger,
                )

                artifact_path = save_report.paths['labeling_outcome']
                artifact_digest = save_report.checksum['labeling_outcome']
                version = artifact_digest[:16] if artifact_digest else Path(artifact_path).stem
                outcome_metadata['artifact_digest'] = artifact_digest
                artifacts_saved = True

                logical_name = ArtifactDataLocator.build_logical_name(
                    artifact_base_name,
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                )
                artifact_manifest = getattr(self.labeler, 'artifact_manifest', None)
                if artifact_manifest is not None:
                    try:
                        artifact_manifest.register(
                            logical_name=logical_name,
                            path=Path(artifact_path),
                            version=version,
                            checksum=artifact_digest,
                        )
                    except Exception as register_error:  # pragma: no cover - manifest failures are non-fatal
                        tprint_warning(
                            f"⚠️ Failed to register outcome in manifest: {register_error}"
                        )

                tprint_info(f"💾 Labeling outcome saved to {artifact_path}")

            except Exception as e:
                tprint_warning(f"⚠️ Failed to save outcome: {e}")
                artifact_save_error = str(e)

            artifacts_bundle = MultiHorizonArtifacts(
                multi_horizon_labeling_result=artifacts_payload.get('multi_horizon_labeling_result', {}),
                labeling_report=artifacts_payload.get('labeling_report', {}),
                standardized_output=artifacts_payload.get('standardized_output'),
                validated_schemas=artifacts_payload.get('validated_schemas'),
            )
            extras = {
                key: value
                for key, value in artifacts_payload.items()
                if key not in {
                    'multi_horizon_labeling_result',
                    'labeling_report',
                    'standardized_output',
                    'validated_schemas',
                }
            }
            if extras:
                artifacts_bundle.extra.update(extras)

            return ComponentResult(
                success=True,
                artifacts=artifacts_bundle,
                metadata={
                    'component_type': 'multi_horizon_profit_labeler',
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'artifacts_saved': artifacts_saved,
                    'artifact_save_skipped': artifact_save_skipped,
                    **({'artifact_save_correlation_id': save_report.correlation_id} if save_report else {}),
                    **({'artifact_persistence_report': asdict(save_report)} if save_report else {}),
                    'validated_schemas': validation_metadata,
                    **({'artifact_digest': artifact_digest} if artifact_digest else {}),
                    **({'artifact_path': artifact_path} if artifact_path else {}),
                    **({'artifact_save_error': artifact_save_error} if artifact_save_error else {})
                }
            )

        except SchemaValidationException as validation_error:
            error_message = str(validation_error)
            tprint_error(f"❌ Schema validation error in multi-horizon labeler: {error_message}")
            return ComponentResult(
                success=False,
                artifacts=MultiHorizonArtifacts(),
                error_message=error_message,
                metadata={
                    'component_type': 'multi_horizon_profit_labeler',
                    'schema_error': {
                        'schema_key': validation_error.schema_key,
                        'context': validation_error.context,
                        'schema_metadata': schema_metadata(validation_error.schema_key).get(validation_error.schema_key)
                    }
                }
            )

        except Exception as e:
            tprint_error(f"❌ Multi-horizon profit labeler component failed: {e}")
            return ComponentResult(
                success=False,
                artifacts=MultiHorizonArtifacts(
                    multi_horizon_labeling_result={},
                    labeling_report={
                        'status': 'failed',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat(),
                    },
                ),
                error_message=str(e),
                metadata={'component_type': 'multi_horizon_profit_labeler'}
            )