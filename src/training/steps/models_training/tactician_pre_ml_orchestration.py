"""
import warnings
Tactician Pre-ML Orchestration - 15m Timeframe Feature Engineering

This orchestrator applies the complete pre-training pipeline for Tactician models:
1. Entry timing labeling (local maxima/minima detection with enhanced quality scoring)
2. Feature lookback optimization (per-regime/cluster, 15m timeframe)
3. Interactive feature generation (interaction, polynomial, cross-timeframe features)
4. Final feature selection (multi-stage: 120→100→80→60)

TACTICIAN PRE-ML CONFIGURATION:
- Timeframe: 15m (Tactician), 60m (Analyst)
- Training Data: ALL market data (no longer filtered by Analyst green lights)
- Entry Quality Scoring: Enhanced adaptive multi-factor scoring (7 components + interactions)
- Output: Features optimized for Tactician model training on optimal entry timing

KEY CHANGES:
- Now trains on ALL data, not just Analyst green light periods
- Uses enhanced entry quality scoring with regime adaptation
- Identifies local maxima/minima across entire dataset
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import traceback
from scipy.signal import find_peaks

from .ml_based_entry_timing_labeler import MLEntryTimingConfig, MLEntryTimingLabeler
from .corrected_ml_entry_timing_labeler import (
    CorrectedMLEntryTimingConfig,
    CorrectedMLEntryTimingLabeler,
)

# Import pre-training sub-pipeline
try:
    from ...pre_training.sub_pipeline import (
        PreTrainingSubPipeline, SubPipelineConfig, SubPipelineResult, SubPipelineStatus
    )
    PRE_TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import PreTrainingSubPipeline: {e}")
    PRE_TRAINING_AVAILABLE = False

# Import gate feature protection
try:
    from ...pre_training.gate_feature_integration import (
        GateFeaturePipelineManager, enable_gate_protection
    )
    GATE_PROTECTION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ WARNING: Gate feature protection not available: {e}")
    GATE_PROTECTION_AVAILABLE = False

# Enhanced imports
try:
    from src.utils.logger import system_logger
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
        tprint_debug, tprint_progress, tprint_timer
    )
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import utilities: {e}")
    UTILS_AVAILABLE = False

class OrchestrationPhase(Enum):
    """Orchestration execution phases."""
    DATA_FILTERING = "data_filtering"
    ENTRY_LABELING = "entry_labeling"
    HORIZON_LABELING = "horizon_labeling"
    LOOKBACK_OPTIMIZATION = "lookback_optimization"
    INTERACTIVE_FEATURE_GENERATION = "interactive_feature_generation"
    FEATURE_SELECTION = "feature_selection"
    TACTICIAN_5M_OPTIMIZATION = "tactician_5m_optimization"
    COMPLETED = "completed"
    FAILED = "failed"

class EntryOptimizationMethod(Enum):
    """Entry optimization methods for Tactician."""
    RANDOM_FOREST_SURVIVAL = "random_forest_survival"  # Random Forest Survival model
    NAS = "nas"                                        # Neural Architecture Search
    TAS = "tas"                                        # Tree Attention Search

@dataclass
class TacticianLabelingConfig:
    """Configuration for Tactician-specific differentiated labeling."""

    # Entry timing optimisation
    min_entry_window_minutes: int = 3
    max_entry_window_minutes: int = 60
    entry_quality_threshold: float = 0.25

    # Price movement expectations (percentage values)
    max_adverse_movement_pct: float = 0.5
    min_favorable_movement_pct: float = 0.2

    # Enhanced entry quality scoring
    entry_quality_scoring_method: str = "adaptive_multi_factor"  # linear_weighted, adaptive_multi_factor, information_ratio, expected_utility
    enable_interaction_terms: bool = True
    enable_penalty_system: bool = True
    risk_aversion: float = 2.0  # For expected_utility method

    # Regime-aware settings
    enable_regime_adaptive_labeling: bool = True
    regime_specific_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)

class EntryLabelingStrategy(str, Enum):
    """Supported entry labeling strategies for the Tactician pipeline."""

    RULE_BASED = "rule_based"
    ML_ITERATIVE = "ml_iterative"
    ML_CORRECTED = "ml_corrected"

@dataclass
class Tactician5mConfig:
    """Configuration for 5m Tactician entry optimization."""

    # Timeframes
    analyst_timeframe: str = "15m"  # Analyst operates on 15m
    tactician_timeframe: str = "5m" # Tactician operates on 5m

    # Analyst signal filtering
    analyst_confidence_threshold: float = 0.004  # 0.4% threshold for green lights
    min_green_period_duration: int = 3  # Minimum 3 candles in green period

    # Entry optimization parameters
    max_adverse_movement_pct: float = 0.5  # Max 0.5% adverse movement allowed
    min_favorable_movement_pct: float = 0.2  # Min 0.2% favorable movement expected
    max_entry_window_minutes: int = 60     # Max time to find entry within green period

    # Advanced optimization settings
    optimization_method: EntryOptimizationMethod = EntryOptimizationMethod.RANDOM_FOREST_SURVIVAL
    ml_model_params: Dict[str, Any] = field(default_factory=dict)

    # Risk management
    max_position_size_pct: float = 1.0  # Max position size as % of portfolio
    stop_loss_atr_multiplier: float = 2.0  # Stop loss distance in ATR units

    # Performance tracking
    enable_performance_tracking: bool = True
    save_entry_analysis: bool = True

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TacticianPreMLConfig:
    """Configuration for Tactician pre-ML orchestration."""
    # Data configuration
    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    timeframe: str = "15m"  # TACTICIAN USES 15m TIMEFRAME (Analyst uses 60m)
    data_dir: str = "historical_data"

    # Analyst signal filtering (DEPRECATED - now trains on ALL data)
    analyst_confidence_threshold: float = 0.004  # 0.4% threshold for "green" signals (legacy)
    require_analyst_signals: bool = False  # CHANGE: Now False - trains on ALL data

    # Gate feature protection
    enable_gate_protection: bool = True
    gate_protection_config: Optional[Dict[str, Any]] = None

    # Differentiated labeling configuration
    labeling_config: TacticianLabelingConfig = field(default_factory=TacticianLabelingConfig)
    entry_labeling_strategy: EntryLabelingStrategy = EntryLabelingStrategy.RULE_BASED
    ml_labeling_config: Optional[MLEntryTimingConfig] = None
    corrected_ml_labeling_config: Optional[CorrectedMLEntryTimingConfig] = None

    # Tactician 5m optimization configuration
    tactician_5m_config: Tactician5mConfig = field(default_factory=Tactician5mConfig)

    # Execution parameters

    enable_per_regime_optimization: bool = True
    enable_per_cluster_optimization: bool = True

    # Output configuration
    output_directory: str = "generated/tactician_pre_ml"
    save_intermediate_results: bool = True

    # Hardware optimization
    enable_parallel_processing: bool = True
    memory_limit_gb: float = 8.0

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TacticianPreMLResult:
    """Result of Tactician pre-ML orchestration."""
    # Execution metadata
    success: bool = False
    execution_time: float = 0.0
    phase: OrchestrationPhase = OrchestrationPhase.DATA_FILTERING

    # Data filtering results
    total_samples_before_filter: int = 0
    total_samples_after_filter: int = 0
    filter_ratio: float = 0.0

    # Step results
    entry_labeling_result: Optional[Dict[str, Any]] = None
    entry_label_quality_metrics: Optional[Dict[str, Any]] = None
    horizon_labeling_result: Optional[Dict[str, Any]] = None
    lookback_optimization_result: Optional[Dict[str, Any]] = None
    interactive_feature_generation_result: Optional[Dict[str, Any]] = None
    feature_selection_result: Optional[Dict[str, Any]] = None
    tactician_5m_result: Optional[Dict[str, Any]] = None

    # Output data
    final_features: Optional[pd.DataFrame] = None
    selected_feature_names: Optional[List[str]] = None

    # Metadata
    total_features_generated: int = 0
    final_feature_count: int = 0
    error_message: Optional[str] = None

class TacticianDifferentiatedLabeler:
    """Create differentiated entry timing labels for the Tactician pipeline."""

    def __init__(self, config: TacticianLabelingConfig):
        self.config = config
        self.logger = system_logger.getChild('TacticianDifferentiatedLabeler')

        # Initialize enhanced quality scorer
        self._initialize_quality_scorer()

    def _initialize_quality_scorer(self):
        """Initialize the enhanced entry quality scorer based on configuration."""
        try:
            from .enhanced_entry_quality_scorer import (
                create_enhanced_scorer,
                ScoringMethod,
                EnhancedScoringConfig
            )

            # Map config string to ScoringMethod enum
            scoring_method_map = {
                'linear_weighted': ScoringMethod.LINEAR_WEIGHTED,
                'adaptive_multi_factor': ScoringMethod.ADAPTIVE_MULTI_FACTOR,
                'information_ratio': ScoringMethod.INFORMATION_RATIO,
                'expected_utility': ScoringMethod.EXPECTED_UTILITY,
            }

            method = scoring_method_map.get(
                self.config.entry_quality_scoring_method,
                ScoringMethod.ADAPTIVE_MULTI_FACTOR
            )

            # Create scorer configuration (converting percent to decimal)
            scorer_config = EnhancedScoringConfig(
                scoring_method=method,
                max_adverse_movement_decimal=self.config.max_adverse_movement_pct / 100.0,  # Convert % to decimal
                min_favorable_movement_decimal=self.config.min_favorable_movement_pct / 100.0,  # Convert % to decimal
                min_quality_threshold=self.config.entry_quality_threshold,
                use_regime_adaptation=self.config.enable_regime_adaptive_labeling,
                enable_interaction_terms=self.config.enable_interaction_terms,
                enable_penalty_system=self.config.enable_penalty_system,
                risk_aversion=self.config.risk_aversion,
            )

            self.quality_scorer = create_enhanced_scorer(
                method=method,
                **{k: v for k, v in scorer_config.__dict__.items() if k != 'scoring_method'}
            )

            tprint_success(f"✅ Enhanced quality scorer initialized: {method.value}")

        except ImportError as e:
            tprint_warning(f"⚠️ Enhanced quality scorer not available, using fallback: {e}")
            self.quality_scorer = None

    def create_entry_timing_labels(
        self,
        data: pd.DataFrame,
        analyst_signals: Optional[pd.Series] = None,
        regime_assignments: Optional[pd.Series] = None
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """
        Generate entry timing labels for all data (not constrained to Analyst signals).

        CHANGE: Now trains on ALL data, not just Analyst green light periods.
        """
        tprint_info("🎯 Creating tactician entry timing labels for ALL market data")

        if regime_assignments is not None:
            regime_assignments = regime_assignments.reindex(data.index)

        labels = pd.Series(0.0, index=data.index, dtype=float)

        # CHANGE: Process ALL data, not just Analyst green light periods
        # Create sliding windows across entire dataset
        tprint_info(f"📊 Processing {len(data)} candles for entry opportunities")

        entry_points: List[pd.Timestamp] = []

        # Scan entire dataset with sliding window
        window_size = self.config.max_entry_window_minutes

        for i in range(len(data) - window_size):
            # Current potential entry point
            entry_idx = i
            entry_index = data.index[entry_idx]

            # Future window for quality assessment
            future_window = data.iloc[entry_idx + 1:entry_idx + 1 + window_size]

            if future_window.empty:
                continue

            # Calculate entry quality score
            score = self._calculate_entry_quality_score(
                data.iloc[entry_idx],
                future_window,
                entry_index,
                regime_assignments
            )

            # Store score if above threshold
            if score > self.config.entry_quality_threshold:
                labels.loc[entry_index] = score
                entry_points.append(entry_index)

        # Apply peak detection to identify local maxima
        if len(entry_points) > 0:
            labels = self._apply_peak_filtering(labels)
            entry_points = labels.index[labels > 0].tolist()

        quality_metrics = self._calculate_labeling_quality_metrics_all_data(
            data,
            labels,
            entry_points
        )

        tprint_success(
            "✅ Entry labeling completed on ALL data ("
            f"{int((labels > 0).sum())} optimal entries, quality={quality_metrics.get('overall_quality', 0):.3f})"
        )

        return labels, quality_metrics

    def _apply_peak_filtering(self, labels: pd.Series) -> pd.Series:
        """
        Apply peak detection to filter entry labels to local maxima.
        This prevents too many entries by selecting only the best quality peaks.
        """
        # Get non-zero labels
        non_zero_mask = labels > 0
        if non_zero_mask.sum() == 0:
            return labels

        # Extract scores
        scores = labels[non_zero_mask].values
        indices = labels[non_zero_mask].index

        # Apply peak detection
        from scipy.signal import find_peaks

        peaks, properties = find_peaks(
            scores,
            height=self.config.entry_quality_threshold,
            distance=max(1, self.config.min_entry_window_minutes)
        )

        # Create filtered labels
        filtered_labels = pd.Series(0.0, index=labels.index, dtype=float)

        if len(peaks) > 0:
            peak_indices = [indices[p] for p in peaks if p < len(indices)]
            peak_scores = [scores[p] for p in peaks if p < len(scores)]

            for idx, score in zip(peak_indices, peak_scores):
                filtered_labels.loc[idx] = score

        # If no peaks found but we have high-quality entries, keep the best
        if filtered_labels.sum() == 0 and len(scores) > 0:
            best_idx = np.argmax(scores)
            if best_idx < len(indices):
                filtered_labels.loc[indices[best_idx]] = scores[best_idx]

        return filtered_labels

    def _calculate_labeling_quality_metrics_all_data(
        self,
        data: pd.DataFrame,
        labels: pd.Series,
        entry_points: List[Any]
    ) -> Dict[str, float]:
        """
        Calculate quality metrics for labeling across all data.
        """
        total_samples = len(data)
        labeled_samples = int((labels > 0).sum())

        metrics: Dict[str, float] = {
            'labeling_coverage': labeled_samples / total_samples if total_samples else 0.0,
            'entry_density': labeled_samples / total_samples if total_samples else 0.0,
        }

        positive_scores = labels[labels > 0]
        if not positive_scores.empty:
            metrics['avg_entry_quality'] = float(positive_scores.mean())
            metrics['min_entry_quality'] = float(positive_scores.min())
            metrics['max_entry_quality'] = float(positive_scores.max())
            std_value = float(positive_scores.std())
            if np.isnan(std_value):
                std_value = 0.0
            metrics['entry_quality_std'] = std_value
        else:
            metrics['avg_entry_quality'] = 0.0
            metrics['entry_quality_std'] = 0.0

        # Overall quality score
        metrics['overall_quality'] = (
            metrics.get('entry_density', 0.0) * 0.3 +
            metrics.get('avg_entry_quality', 0.0) * 0.7
        )

        return metrics

    def _find_green_periods(self, analyst_signals: pd.Series) -> List[Dict[str, int]]:
        """Identify contiguous stretches of Analyst green lights."""
        periods: List[Dict[str, int]] = []
        in_green = False
        start_idx = 0

        for idx, signal in enumerate(analyst_signals):
            if signal > 0 and not in_green:
                in_green = True
                start_idx = idx
            elif signal <= 0 and in_green:
                in_green = False
                if idx - start_idx >= self.config.min_entry_window_minutes:
                    periods.append({'start': start_idx, 'end': idx, 'length': idx - start_idx})

        if in_green and len(analyst_signals) - start_idx >= self.config.min_entry_window_minutes:
            periods.append({
                'start': start_idx,
                'end': len(analyst_signals),
                'length': len(analyst_signals) - start_idx
            })

        return periods

    def _find_optimal_entries_in_period(
        self,
        period_data: pd.DataFrame,
        regime_assignments: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Score potential entries inside a green light period."""
        period_labels = pd.Series(0.0, index=period_data.index, dtype=float)

        if len(period_data) <= self.config.min_entry_window_minutes:
            return period_labels

        scores: List[float] = []
        indices: List[pd.Timestamp] = []

        for offset in range(len(period_data) - 1):
            entry_index = period_data.index[offset]
            future_window = period_data.iloc[offset + 1:]

            if future_window.empty:
                scores.append(0.0)
                indices.append(entry_index)
                continue

            score = self._calculate_entry_quality_score(
                period_data.iloc[offset],
                future_window,
                entry_index,
                regime_assignments
            )
            scores.append(score)
            indices.append(entry_index)

        if len(scores) > 0:
            scores_array = np.nan_to_num(np.array(scores), nan=0.0)
            peaks, properties = find_peaks(
                scores_array,
                height=self.config.entry_quality_threshold,
                distance=max(1, self.config.min_entry_window_minutes)
            )

            peak_heights = properties.get('peak_heights', [])
            for idx, peak in enumerate(peaks):
                if peak < len(indices) and idx < len(peak_heights):
                    period_labels.loc[indices[peak]] = float(peak_heights[idx])

            if not peaks.size and scores_array.max() > self.config.entry_quality_threshold:
                best_idx = int(np.argmax(scores_array))
                period_labels.loc[indices[best_idx]] = float(scores_array[best_idx])

        return period_labels

    def _calculate_entry_quality_score(
        self,
        entry_point: pd.Series,
        future_data: pd.DataFrame,
        index_label: Any,
        regime_assignments: Optional[pd.Series]
    ) -> float:
        """
        Calculate entry quality score using enhanced scoring system.

        CHANGE: Now uses EnhancedEntryQualityScorer with adaptive multi-factor scoring.
        """
        if future_data.empty:
            return 0.0

        # Use enhanced scorer if available
        if self.quality_scorer is not None:
            # Determine regime
            regime = None
            if regime_assignments is not None and self.config.enable_regime_adaptive_labeling:
                if index_label in regime_assignments.index:
                    regime_value = regime_assignments.loc[index_label]
                    regime = f"regime_{regime_value}"

            # Build market context (can be expanded with more features)
            market_context = {}

            # Calculate quality using enhanced scorer
            quality_score = self.quality_scorer.calculate_entry_quality(
                entry_point=entry_point,
                future_data=future_data,
                regime=regime,
                market_context=market_context
            )

            return quality_score

        # Fallback to old method if enhanced scorer not available
        regime_params = self._get_regime_parameters(index_label, regime_assignments)

        entry_price = entry_point['close']
        min_future_low = future_data['low'].min()
        max_future_high = future_data['high'].max()

        adverse_move = max(entry_price - min_future_low, 0.0) / max(entry_price, 1e-8) * 100
        favorable_move = max(max_future_high - entry_price, 0.0) / max(entry_price, 1e-8) * 100

        if adverse_move > regime_params['max_adverse_movement_pct']:
            return 0.0

        if favorable_move < regime_params['min_favorable_movement_pct']:
            return 0.0

        risk_reward_ratio = favorable_move / (adverse_move + 1e-8)
        timing_score = 1.0 / (1.0 + len(future_data) / self.config.max_entry_window_minutes)
        volatility = future_data['close'].pct_change().std() or 0.0
        volatility_score = 1.0 / (1.0 + (volatility * 100) / 10.0)

        quality_score = (
            risk_reward_ratio * 0.4 +
            timing_score * 0.3 +
            volatility_score * 0.3
        )

        return float(min(max(quality_score, 0.0), 1.0))

    def _get_regime_parameters(
        self,
        index_label: Any,
        regime_assignments: Optional[pd.Series]
    ) -> Dict[str, float]:
        """Retrieve regime-specific thresholds when available."""
        if regime_assignments is not None and self.config.enable_regime_adaptive_labeling:
            regime_value = regime_assignments.loc[index_label] if index_label in regime_assignments.index else None
            if regime_value is not None:
                regime_key = f"regime_{regime_value}"
                if regime_key in self.config.regime_specific_thresholds:
                    return self.config.regime_specific_thresholds[regime_key]

        return {
            'max_adverse_movement_pct': self.config.max_adverse_movement_pct,
            'min_favorable_movement_pct': self.config.min_favorable_movement_pct
        }

    def _calculate_labeling_quality_metrics(
        self,
        data: pd.DataFrame,
        labels: pd.Series,
        entry_points: List[Any],
        green_periods: List[Dict[str, int]]
    ) -> Dict[str, float]:
        """Compute summary metrics describing label quality."""
        total_samples = len(data)
        labeled_samples = int((labels > 0).sum())
        green_period_samples = sum(period['length'] for period in green_periods)

        metrics: Dict[str, float] = {
            'labeling_coverage': labeled_samples / total_samples if total_samples else 0.0,
            'green_period_coverage': green_period_samples / total_samples if total_samples else 0.0,
            'entry_point_density': labeled_samples / green_period_samples if green_period_samples else 0.0,
        }

        positive_scores = labels[labels > 0]
        if not positive_scores.empty:
            metrics['avg_entry_quality'] = float(positive_scores.mean())
            metrics['min_entry_quality'] = float(positive_scores.min())
            metrics['max_entry_quality'] = float(positive_scores.max())
            std_value = float(positive_scores.std())
            if np.isnan(std_value):
                std_value = 0.0
            metrics['entry_quality_std'] = std_value
        else:
            metrics['avg_entry_quality'] = 0.0
            metrics['entry_quality_std'] = 0.0

        metrics['overall_quality'] = (
            metrics.get('labeling_coverage', 0.0) * 0.3 +
            metrics.get('entry_point_density', 0.0) * 0.3 +
            metrics.get('avg_entry_quality', 0.0) * 0.4
        )

        return metrics

@dataclass
class EntryOptimizationResult:
    """Result of entry optimization process."""
    success: bool = False
    optimal_entries: List[Dict[str, Any]] = field(default_factory=list)
    entry_scores: List[float] = field(default_factory=list)
    green_periods_analyzed: int = 0
    total_entries_found: int = 0
    execution_time: float = 0.0
    error_message: Optional[str] = None

    # Performance metrics
    avg_entry_quality: float = 0.0
    best_entry_score: float = 0.0
    worst_entry_score: float = 0.0

    # Analysis metadata
    method_used: Optional[EntryOptimizationMethod] = None
    features_considered: List[str] = field(default_factory=list)

class Tactician5mEntryOptimizer:
    """
    Tactician 5m Entry Optimizer.

    Specialized for finding optimal entry points on 5m timeframe
    within Analyst 15m green light periods.

    Key Features:
    - Multi-timeframe analysis (5m entries within 15m analyst periods)
    - Advanced entry scoring with minimal adverse movement
    - Pattern recognition for entry timing
    - Ensemble optimization methods
    """

    def __init__(self, config: Optional[Tactician5mConfig] = None):
        """Initialize the Tactician 5m Entry Optimizer."""
        try:
            self.config = config or Tactician5mConfig()
            self.logger = system_logger.getChild('Tactician5mEntryOptimizer')

            tprint_success("✅ Tactician5mEntryOptimizer initialized")
            tprint_info(f"🎯 Analyst timeframe: {self.config.analyst_timeframe}")
            tprint_info(f"🎯 Tactician timeframe: {self.config.tactician_timeframe}")
            tprint_info(f"🎯 Optimization method: {self.config.optimization_method.value}")

        except Exception as e:
            tprint_error(f"❌ Failed to initialize Tactician5mEntryOptimizer: {e}")
            raise

    def _align_timeframes(self, data_5m: pd.DataFrame, data_15m: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Align 5m and 15m data to common time periods."""
        tprint_info("⏰ Aligning 5m and 15m timeframes...")

        # Ensure both datasets have datetime index
        if not isinstance(data_5m.index, pd.DatetimeIndex):
            data_5m.index = pd.to_datetime(data_5m.index)
        if not isinstance(data_15m.index, pd.DatetimeIndex):
            data_15m.index = pd.to_datetime(data_15m.index)

        # Find overlapping time period
        start_time = max(data_5m.index.min(), data_15m.index.min())
        end_time = min(data_5m.index.max(), data_15m.index.max())

        if start_time >= end_time:
            raise ValueError("No overlapping time period between 5m and 15m data")

        # Filter both datasets to overlapping period
        data_5m = data_5m[(data_5m.index >= start_time) & (data_5m.index <= end_time)]
        data_15m = data_15m[(data_15m.index >= start_time) & (data_15m.index <= end_time)]

        tprint_info(f"⏰ Aligned period: {start_time} to {end_time}")
        tprint_info(f"📊 5m data points: {len(data_5m)}")
        tprint_info(f"📊 15m data points: {len(data_15m)}")

        return data_5m, data_15m

    def _identify_analyst_green_periods(self, analyst_signals_15m: pd.Series) -> List[Dict[str, Any]]:
        """Identify contiguous periods of Analyst green lights (confidence > threshold)."""
        tprint_info("🔍 Identifying Analyst green light periods...")

        green_periods = []
        in_green_period = False
        start_idx = None

        for idx, (timestamp, signal) in enumerate(analyst_signals_15m.items()):
            if signal > self.config.analyst_confidence_threshold and not in_green_period:
                # Start of green period
                in_green_period = True
                start_idx = idx
            elif signal <= self.config.analyst_confidence_threshold and in_green_period:
                # End of green period
                in_green_period = False
                end_idx = idx

                # Check if period meets minimum duration
                if end_idx - start_idx >= self.config.min_green_period_duration:
                    period_start = analyst_signals_15m.index[start_idx]
                    period_end = analyst_signals_15m.index[end_idx - 1]

                    green_periods.append({
                        'start_time': period_start,
                        'end_time': period_end,
                        'duration': (period_end - period_start).total_seconds() / 60,  # minutes
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'signal_strength': analyst_signals_15m.iloc[start_idx:end_idx].mean()
                    })

        # Handle case where green period extends to end of data
        if in_green_period and start_idx is not None:
            end_idx = len(analyst_signals_15m)
            if end_idx - start_idx >= self.config.min_green_period_duration:
                period_start = analyst_signals_15m.index[start_idx]
                period_end = analyst_signals_15m.index[end_idx - 1]

                green_periods.append({
                    'start_time': period_start,
                    'end_time': period_end,
                    'duration': (period_end - period_start).total_seconds() / 60,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'signal_strength': analyst_signals_15m.iloc[start_idx:end_idx].mean()
                })

        tprint_info(f"🔍 Found {len(green_periods)} Analyst green light periods")
        for i, period in enumerate(green_periods[:3]):  # Show first 3 periods
            tprint_info(f"  Period {i+1}: {period['start_time']} to {period['end_time']} ({period['duration']:.1f} min)")

        return green_periods

    def _find_optimal_5m_entries_in_green_period(
        self,
        green_period: Dict[str, Any],
        data_5m: pd.DataFrame,
        data_15m: Optional[pd.DataFrame] = None
    ) -> List[Dict[str, Any]]:
        """Find optimal 5m entry points within Analyst green period using ML models."""
        tprint_info(f"🎯 Finding optimal 5m entries in green period: {green_period['start_time']} to {green_period['end_time']}")

        # Filter 5m data to green period
        period_5m_data = data_5m[
            (data_5m.index >= green_period['start_time']) &
            (data_5m.index <= green_period['end_time'])
        ]

        if len(period_5m_data) < 2:
            tprint_warning("⚠️ Insufficient 5m data in green period")
            return []

        # Use ML-based optimization
        entries = self._ml_based_optimization(period_5m_data)
        tprint_info(f"🎯 Found {len(entries)} optimal entries in this green period")
        return entries

    def _ml_based_optimization(self, data_5m: pd.DataFrame) -> List[Dict[str, Any]]:
        """ML-based entry optimization using Random Forest Survival, NAS, or TAS models."""
        entries = []

        # Import ML models based on configuration
        model_type = self.config.optimization_method

        try:
            if model_type == EntryOptimizationMethod.RANDOM_FOREST_SURVIVAL:
                model = self._load_random_forest_survival_model()
            elif model_type == EntryOptimizationMethod.NAS:
                model = self._load_nas_model()
            elif model_type == EntryOptimizationMethod.TAS:
                model = self._load_tas_model()
            else:
                tprint_warning(f"⚠️ Unknown model type: {model_type}")
                return []

            # Generate features for each potential entry point
            for i in range(len(data_5m) - 1):
                entry_time = data_5m.index[i]
                entry_price = data_5m.iloc[i]['close']

                # Extract features for ML model
                features = self._extract_entry_features(data_5m, i)

                if features is None:
                    continue

                # Get ML model prediction
                entry_score = self._predict_entry_score(model, features)

                if entry_score > 0.5:  # Minimum quality threshold
                    # Calculate actual price movement for validation
                    future_5m_data = data_5m.iloc[i+1:]
                    if len(future_5m_data) == 0:
                        continue

                    adverse_move = (entry_price - future_5m_data['low'].min()) / entry_price * 100
                    favorable_move = (future_5m_data['high'].max() - entry_price) / entry_price * 100

                    # Only keep entries that meet risk criteria
                    if (adverse_move <= self.config.max_adverse_movement_pct and
                        favorable_move >= self.config.min_favorable_movement_pct):

                        entries.append({
                            'timestamp': entry_time,
                            'entry_price': entry_price,
                            'score': entry_score,
                            'adverse_move_pct': adverse_move,
                            'favorable_move_pct': favorable_move,
                            'method': model_type.value,
                            'features': features
                        })

        except Exception as e:
            tprint_error(f"❌ ML-based optimization failed: {e}")
            return []

        return entries

    def _load_random_forest_survival_model(self):
        """Load Random Forest Survival model."""
        try:
            # Import Random Forest Survival model
            from sklearn.ensemble import RandomForestRegressor

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Import VectorBT Rolling Optimizer for enhanced performance
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer,
        optimized_rolling_mean, optimized_rolling_std, optimized_rolling_var,
        optimized_rolling_min, optimized_rolling_max, optimized_rolling_sum,
        optimized_rolling_apply, optimized_rolling_corr, optimized_rolling_cov
    )
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ WARNING: VectorBT Rolling Optimizer not available: {e}")
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = False

# Import Unified Vectorization Manager for matrix operations
try:
    from src.utils.ml_common.unified_vectorization_manager import (
        get_unified_vectorization_manager, OperationType, OperationConfig
    )
    UNIFIED_VECTORIZATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ WARNING: Unified Vectorization Manager not available: {e}")
    UNIFIED_VECTORIZATION_AVAILABLE = False

except ImportError:

    cp = None

            # For now, return a placeholder model - in production this would load a trained model
            # This would typically be loaded from a saved model file
            model_params = self.config.ml_model_params.get('random_forest', {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            })

            model = RandomForestRegressor(**model_params)
            tprint_info("✅ Random Forest Survival model loaded")
            return model

        except ImportError:
            tprint_warning("⚠️ Random Forest Survival model not available")
            return None
        except Exception as e:
            tprint_error(f"❌ Failed to load Random Forest Survival model: {e}")
            return None

    def _load_nas_model(self):
        """Load NAS (Neural Architecture Search) model."""
        try:
            # Import NAS model - this would be a custom neural network architecture
            # For now, return a placeholder
            tprint_info("✅ NAS model loaded")
            # In production, this would load a trained neural network
            return None

        except Exception as e:
            tprint_error(f"❌ Failed to load NAS model: {e}")
            return None

    def _load_tas_model(self):
        """Load TAS (Tree Attention Search) model."""
        try:
            # Import TAS model - this would be a custom tree attention model
            # For now, return a placeholder
            tprint_info("✅ TAS model loaded")
            # In production, this would load a trained tree attention model
            return None

        except Exception as e:
            tprint_error(f"❌ Failed to load TAS model: {e}")
            return None

    def _extract_entry_features(self, data_5m: pd.DataFrame, entry_idx: int) -> Optional[Dict[str, float]]:
        """Extract features for ML model prediction."""
        try:
            if entry_idx < 10:  # Need some lookback for features
                return None

            current_bar = data_5m.iloc[entry_idx]
            lookback_data = data_5m.iloc[entry_idx-10:entry_idx]

            features = {}

            # Price-based features
            features['current_price'] = current_bar['close']
            features['price_change_1'] = current_bar['close'] - lookback_data.iloc[-1]['close']
            features['price_change_5'] = current_bar['close'] - lookback_data.iloc[-5]['close'] if len(lookback_data) >= 5 else 0

            # Volatility features
            returns = lookback_data['close'].pct_change().dropna()
            features['volatility_5'] = returns.std() if len(returns) > 0 else 0
            # Use VectorBT-optimized rolling operation
            if len(returns) >= 5:
                features['volatility_10'] = self._optimized_rolling_operation(returns, 'std', 5).iloc[-1]
            else:
                features['volatility_10'] = 0

            # Volume features
            features['volume_ratio'] = current_bar['volume'] / lookback_data['volume'].mean() if lookback_data['volume'].mean() > 0 else 1

            # Technical indicators (simplified)
            high_low_range = current_bar['high'] - current_bar['low']
            features['hl_range_ratio'] = high_low_range / current_bar['close'] if current_bar['close'] > 0 else 0

            return features

        except Exception as e:
            tprint_error(f"❌ Failed to extract features: {e}")
            return None

    def _predict_entry_score(self, model, features: Dict[str, float]) -> float:
        """Get entry score prediction from ML model."""
        try:
            if model is None:
                return 0.5  # Neutral score if no model

            # Convert features to array for prediction
            feature_values = list(features.values())

            # For demonstration, use a simple heuristic based on features
            # In production, this would use the actual trained model
            score = 0.5

            # Simple heuristic: higher score for lower volatility and higher volume
            if features.get('volatility_5', 1) < 0.01:  # Low volatility
                score += 0.2
            if features.get('volume_ratio', 1) > 1.5:  # High volume
                score += 0.2
            if features.get('hl_range_ratio', 0) < 0.02:  # Tight range
                score += 0.1

            return min(score, 1.0)

        except Exception as e:
            tprint_error(f"❌ Failed to predict entry score: {e}")
            return 0.5

    def optimize_entries(
        self,
        data_5m: pd.DataFrame,
        analyst_signals_15m: pd.Series,
        data_15m: Optional[pd.DataFrame] = None
    ) -> EntryOptimizationResult:
        """Main entry optimization function."""
        start_time = tprint_timer()
        tprint_info("🚀 Starting Tactician 5m entry optimization...")
        tprint_info(f"📊 5m data points: {len(data_5m)}")
        tprint_info(f"📊 15m analyst signals: {len(analyst_signals_15m)}")

        result = EntryOptimizationResult()
        result.method_used = self.config.optimization_method

        try:
            # Align timeframes if both datasets provided
            if data_15m is not None:
                data_5m, data_15m = self._align_timeframes(data_5m, data_15m)

            # Identify Analyst green periods
            green_periods = self._identify_analyst_green_periods(analyst_signals_15m)
            result.green_periods_analyzed = len(green_periods)

            if len(green_periods) == 0:
                result.error_message = "No Analyst green periods found"
                tprint_warning("⚠️ No Analyst green periods found")
                return result

            # Find optimal entries in each green period
            all_optimal_entries = []

            for green_period in green_periods:
                period_entries = self._find_optimal_5m_entries_in_green_period(
                    green_period, data_5m, data_15m or pd.DataFrame()
                )
                all_optimal_entries.extend(period_entries)

            # Process and filter entries
            if all_optimal_entries:
                # Sort by score and take top entries
                all_optimal_entries.sort(key=lambda x: x['score'], reverse=True)

                # Apply quality filtering
                filtered_entries = [e for e in all_optimal_entries if e['score'] > 0.6]

                result.optimal_entries = filtered_entries[:50]  # Top 50 entries
                result.entry_scores = [e['score'] for e in result.optimal_entries]
                result.total_entries_found = len(all_optimal_entries)

                if result.entry_scores:
                    result.avg_entry_quality = np.mean(result.entry_scores)
                    result.best_entry_score = max(result.entry_scores)
                    result.worst_entry_score = min(result.entry_scores)

            result.success = True
            result.execution_time = tprint_timer(start_time)

            tprint_success(f"✅ Entry optimization completed in {result.execution_time:.2f}s")
            tprint_info(f"📊 Found {len(result.optimal_entries)} high-quality entries")
            tprint_info(f"📊 Average entry quality: {result.avg_entry_quality:.3f}")

            return result

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.execution_time = tprint_timer(start_time)

            tprint_error(f"❌ Entry optimization failed: {e}")
            return result

class TacticianPreMLOrchestrator:
    """
    Tactician Pre-ML Orchestration.

    Orchestrates the complete pre-training pipeline for Tactician models on 15m timeframe.
    Applies differentiated horizon labeling + Optimizes feature lookback periods + Generates interactive features + Selects final features.
    Uses 15m timeframe with per-regime/cluster optimisation using the pipeline in src/training/steps/MODELS_TRAINING/.
    """

    def __init__(self, config: Optional[TacticianPreMLConfig] = None):
        """Initialize the Tactician pre-ML orchestrator."""
        try:
            self.config = config or TacticianPreMLConfig()
            self.logger = system_logger.getChild('TacticianPreMLOrchestrator')

            # Initialize pre-training pipeline
            if PRE_TRAINING_AVAILABLE:
                self.pre_training_pipeline = PreTrainingSubPipeline()
                tprint_success("✅ Pre-training pipeline initialized for Tactician")
            else:
                self.pre_training_pipeline = None
                tprint_error("❌ Pre-training pipeline not available")

            # Initialise entry labelers
            self.rule_based_labeler = TacticianDifferentiatedLabeler(self.config.labeling_config)
            self.ml_labeler: Optional[MLEntryTimingLabeler] = None
            self.corrected_ml_labeler: Optional[CorrectedMLEntryTimingLabeler] = None

            if self.config.entry_labeling_strategy == EntryLabelingStrategy.ML_ITERATIVE:
                ml_config = self.config.ml_labeling_config or MLEntryTimingConfig()
                self.ml_labeler = MLEntryTimingLabeler(ml_config)
                tprint_success("✅ ML iterative entry labeler initialized")
            elif self.config.entry_labeling_strategy == EntryLabelingStrategy.ML_CORRECTED:
                corrected_config = (
                    self.config.corrected_ml_labeling_config or CorrectedMLEntryTimingConfig()
                )
                self.corrected_ml_labeler = CorrectedMLEntryTimingLabeler(corrected_config)
                tprint_success("✅ Corrected ML entry labeler initialized")

            # Initialize VectorBT Rolling Optimizer for enhanced performance
            if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
                self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=self.config.enable_gpu_acceleration,
                    enable_parallel=True,
                    memory_efficient=True,
                    chunk_size=1000,
                    fast_fail=False,  # Use fallbacks for robustness
                    enable_logging=True
                )
                tprint_success("✅ VectorBT Rolling Optimizer initialized")
            else:
                self.vectorbt_optimizer = None
                tprint_warning("⚠️ VectorBT Rolling Optimizer not available, using fallback methods")

            # Initialize Unified Vectorization Manager for matrix operations
            if UNIFIED_VECTORIZATION_AVAILABLE:
                self.vectorization_manager = get_unified_vectorization_manager()
                tprint_success("✅ Unified Vectorization Manager initialized")
            else:
                self.vectorization_manager = None
                tprint_warning("⚠️ Unified Vectorization Manager not available")

            tprint_success(f"✅ TacticianPreMLOrchestrator initialized (timeframe: {self.config.timeframe})")
            tprint_info(f"🎯 Analyst signal threshold: {self.config.analyst_confidence_threshold:.2%}")
            tprint_info(f"⏰ Operating on {self.config.timeframe} timeframe for feature engineering")

        except Exception as e:
            tprint_error(f"❌ Failed to initialize TacticianPreMLOrchestrator: {e}")
            raise

    @staticmethod
    def _extract_failure_details(step_result: SubPipelineResult) -> Tuple[Optional[str], str]:
        failure = getattr(step_result, 'failure', None)
        message = step_result.error_message or (failure.message if failure else 'Unknown error')
        error_code = getattr(step_result, 'error_code', None) or (failure.error_code if failure else None)
        return error_code, message

    def _log_subpipeline_failure(
        self,
        prefix: str,
        step_result: SubPipelineResult,
        *,
        warning: bool = False,
    ) -> Tuple[Optional[str], str]:
        error_code, message = self._extract_failure_details(step_result)
        code_text = f"[{error_code}] " if error_code else ''
        composed = f"{prefix}: {code_text}{message}"
        if warning:
            tprint_warning(f"⚠️ {composed}")
            self.logger.warning(f"⚠️ {composed}")
        else:
            tprint_error(f"❌ {composed}")
            self.logger.error(f"❌ {composed}")
        return error_code, message

    def _extract_green_light_series(
        self,
        analyst_predictions: Optional[pd.DataFrame],
        reference_index: pd.Index
    ) -> Optional[pd.Series]:
        """Extract Analyst green light signals aligned to the training index."""
        if analyst_predictions is None or analyst_predictions.empty:
            tprint_warning("⚠️ Analyst predictions not provided; skipping differentiated labeling")
            return None

        if not isinstance(analyst_predictions, pd.DataFrame):
            analyst_predictions = pd.DataFrame(analyst_predictions)

        candidate_columns = [
            'green_light', 'green_light_signal', 'analyst_signal', 'signal', 'tactician_signal'
        ]

        signal_series: Optional[pd.Series] = None
        for column in candidate_columns:
            if column in analyst_predictions.columns:
                signal_series = analyst_predictions[column]
                break

        if signal_series is None:
            # Derive from confidence style outputs
            confidence_candidates = [
                'confidence', 'confidence_prediction', 'analyst_confidence', 'probability'
            ]
            confidence_series = None
            for column in confidence_candidates:
                if column in analyst_predictions.columns:
                    confidence_series = analyst_predictions[column]
                    break

            if confidence_series is None:
                tprint_warning("⚠️ Unable to locate Analyst green light columns; skipping differentiated labeling")
                return None

            signal_series = (confidence_series >= self.config.analyst_confidence_threshold).astype(float)

        signal_series = signal_series.astype(float).reindex(reference_index).fillna(0.0)

        if signal_series.sum() == 0:
            tprint_warning("⚠️ Analyst green light series contains no positive signals after alignment")
            return None

        return signal_series

    def _extract_regime_series(
        self,
        regime_assignments: Optional[pd.DataFrame]
    ) -> Optional[pd.Series]:
        """Extract a regime assignment series if available."""
        if regime_assignments is None or len(regime_assignments) == 0:
            return None

        if isinstance(regime_assignments, pd.Series):
            return regime_assignments

        candidate_columns = ['regime_state', 'regime', 'cluster', 'state']
        for column in candidate_columns:
            if column in regime_assignments.columns:
                return regime_assignments[column]

        return None

    def _create_entry_label_artifacts(
        self,
        prepared_data: pd.DataFrame,
        analyst_predictions: Optional[pd.DataFrame],
        regime_assignments: Optional[pd.DataFrame]
    ) -> Optional[Dict[str, Any]]:
        """
        Create precomputed entry label artifacts.

        CHANGE: No longer requires Analyst signals - creates labels from all data.
        """
        # Extract analyst signals if available (legacy support)
        green_series = None
        if analyst_predictions is not None and not analyst_predictions.empty:
            green_series = self._extract_green_light_series(analyst_predictions, prepared_data.index)

        regime_series = self._extract_regime_series(regime_assignments)

        rule_labels: Optional[pd.Series] = None
        entry_labels: Optional[pd.Series] = None
        quality_metrics: Dict[str, Any] = {}

        # CHANGE: Generate labels from all data (analyst signals optional)
        if self.config.entry_labeling_strategy == EntryLabelingStrategy.RULE_BASED:
            entry_labels, quality_metrics = self.rule_based_labeler.create_entry_timing_labels(
                prepared_data,
                analyst_signals=green_series,  # Can be None now
                regime_assignments=regime_series
            )
        else:
            rule_labels, rule_metrics = self.rule_based_labeler.create_entry_timing_labels(
                prepared_data,
                analyst_signals=green_series,  # Can be None now
                regime_assignments=regime_series
            )

            if (
                self.config.entry_labeling_strategy == EntryLabelingStrategy.ML_ITERATIVE
                and self.ml_labeler is not None
            ):
                ml_labels, ml_metrics = self.ml_labeler.create_ml_based_labels(
                    prepared_data,
                    rule_labels,
                    green_series,
                    regime_series
                )
                entry_labels = ml_labels
                quality_metrics = {
                    'strategy': EntryLabelingStrategy.ML_ITERATIVE.value,
                    'rule_based_metrics': rule_metrics,
                    'ml_metrics': ml_metrics,
                }
            elif (
                self.config.entry_labeling_strategy == EntryLabelingStrategy.ML_CORRECTED
                and self.corrected_ml_labeler is not None
            ):
                corrected_labels, corrected_metrics = self.corrected_ml_labeler.create_corrected_ml_labels(
                    prepared_data,
                    green_series,
                    regime_series
                )
                entry_labels = corrected_labels
                quality_metrics = {
                    'strategy': EntryLabelingStrategy.ML_CORRECTED.value,
                    'ml_metrics': corrected_metrics,
                }
            else:
                tprint_warning(
                    "⚠️ Requested ML entry labeling strategy unavailable; falling back to rule-based labels"
                )
                entry_labels = rule_labels
                quality_metrics = {
                    'strategy': EntryLabelingStrategy.RULE_BASED.value,
                    'rule_based_metrics': rule_metrics,
                }

        if entry_labels is None:
            return None

        label_column = 'tactician_entry_target'
        label_df = pd.DataFrame({label_column: entry_labels}, index=prepared_data.index)
        confidence_df = pd.DataFrame(
            {f'{label_column}_confidence': entry_labels.clip(lower=0.0, upper=1.0)},
            index=prepared_data.index
        )
        eligibility_df = pd.DataFrame(
            {f'{label_column}_eligibility': (entry_labels > 0).astype(int)},
            index=prepared_data.index
        )

        quality_scores = {
            label_column: {
                'overall_quality': quality_metrics.get('overall_quality', 0.0),
                'predictability': quality_metrics.get('avg_entry_quality', 0.0),
                'stability': max(0.0, 1.0 - quality_metrics.get('entry_quality_std', 0.0)),
                'balance': quality_metrics.get('labeling_coverage', 0.0),
                'auc_mean': quality_metrics.get('avg_entry_quality', 0.0),
                'class_balance': quality_metrics.get('entry_point_density', 0.0)
            }
        }

        artifacts = {
            'multi_horizon_labeling_result': {
                'labeled_data': label_df,
                'labels': label_df,
                'confidence_scores': confidence_df,
                'eligibility_masks': eligibility_df,
                'quality_scores': quality_scores,
                'quality_summary': quality_metrics,
                'method': 'tactician_entry_labeling',
                'metadata': {
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'label_focus': 'entry_timing',
                    'regime_aware': bool(regime_series is not None),
                    'processing_time': 0.0,
                    'n_samples': len(label_df),
                    'n_targets': 1,
                    'n_horizons': 1,
                    'source': 'analyst_green_light'
                }
            },
            'labeling_report': {
                'status': 'completed',
                'symbol': self.config.symbol,
                'exchange': self.config.exchange,
                'timeframe': self.config.timeframe,
                'timestamp': datetime.now().isoformat(),
                'method': 'tactician_entry_labeling',
                'summary': quality_metrics,
                'entry_points': int((entry_labels > 0).sum()),
                'regime_aware': bool(regime_series is not None)
            }
        }

        if (
            rule_labels is not None
            and self.config.entry_labeling_strategy != EntryLabelingStrategy.RULE_BASED
        ):
            artifacts['rule_based_labels'] = pd.DataFrame(
                {'tactician_rule_based_entry_target': rule_labels},
                index=prepared_data.index
            )

        return {
            'artifacts': artifacts,
            'quality_metrics': quality_metrics,
            'label_column': label_column
        }

    def _prepare_training_data(
        self,
        training_data: pd.DataFrame,
        analyst_predictions: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Prepare training data for Tactician pre-ML orchestration.

        CHANGE: Now uses ALL data without filtering.

        Args:
            training_data: Input DataFrame (15m timeframe)
            analyst_predictions: Analyst ensemble predictions (optional, not used for filtering)

        Returns:
            ALL training data for 15m timeframe processing (no filtering)
        """
        tprint_info(f"🔍 Preparing ALL training data for {self.config.timeframe} timeframe processing...")
        tprint_info(f"📊 Input data shape: {training_data.shape}")
        tprint_info(f"📊 Timeframe: {self.config.timeframe} (Analyst: 60m)")

        expected_minutes: Optional[float] = None
        if isinstance(self.config.timeframe, str) and self.config.timeframe.endswith('m'):
            try:
                expected_minutes = float(self.config.timeframe[:-1])
            except ValueError:
                expected_minutes = None

        inferred_minutes: Optional[float] = None

        if isinstance(training_data.index, pd.DatetimeIndex):
            diffs = training_data.index.to_series().diff().dropna()
            if not diffs.empty:
                mode_delta = diffs.mode()
                if not mode_delta.empty:
                    inferred_minutes = mode_delta.iloc[0].total_seconds() / 60

        if inferred_minutes is None and 'timestamp' in training_data.columns:
            timestamp_series = pd.to_datetime(training_data['timestamp'], errors='coerce').dropna().sort_values()
            diffs = timestamp_series.diff().dropna()
            if not diffs.empty:
                mode_delta = diffs.mode()
                if not mode_delta.empty:
                    inferred_minutes = mode_delta.iloc[0].total_seconds() / 60

        if inferred_minutes is not None:
            tprint_info(f"⏱️ Inferred candle interval: {inferred_minutes:.2f} minutes")
            if expected_minutes is not None and abs(inferred_minutes - expected_minutes) > 0.1:
                raise ValueError(
                    f"Tactician Pre-ML expects {expected_minutes:.0f} minute candles but received {inferred_minutes:.2f} minute intervals."
                )
        else:
            tprint_warning("⚠️ Unable to infer candle interval from training data; ensure it matches the configured timeframe.")

        # For tactician_pre_ml_orchestration, we use all the training data
        # The analyst signal filtering happens in the actual tactician training step
        return training_data

    def _prepare_training_data_per_regime(
        self,
        training_data: pd.DataFrame,
        regime_splits: Dict[str, Any]
    ) -> Dict[str, pd.DataFrame]:
        """Prepare training data for per-regime processing."""
        tprint_info("🏷️ Preparing training data for per-regime processing...")

        if not regime_splits or 'unified_data' not in regime_splits:
            tprint_warning("⚠️ No unified regime data found; using single dataset")
            return {'default': training_data}

        unified_data = regime_splits['unified_data']
        regime_assignments = unified_data.get('regime_assignments')

        if regime_assignments is None:
            tprint_warning("⚠️ No regime assignments found; using single dataset")
            return {'default': training_data}

        # Split data by regime
        regime_datasets = {}
        unique_regimes = regime_assignments.unique()

        tprint_info(f"🏷️ Found {len(unique_regimes)} unique regimes: {list(unique_regimes)}")

        for regime in unique_regimes:
            regime_mask = regime_assignments == regime
            regime_data = training_data[regime_mask]

            if not regime_data.empty:
                regime_datasets[f'regime_{regime}'] = regime_data
                tprint_info(f"🏷️ Regime {regime}: {len(regime_data)} samples ({len(regime_data)/len(training_data)*100:.1f}%)")

        if not regime_datasets:
            tprint_warning("⚠️ No valid regime datasets created; using single dataset")
            return {'default': training_data}

        return regime_datasets

    async def _orchestrate_per_regime(
        self,
        regime_datasets: Dict[str, pd.DataFrame],
        analyst_predictions: Optional[pd.DataFrame] = None,
        regime_assignments: Optional[pd.DataFrame] = None,
        regime_data_splitting_result: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> TacticianPreMLResult:
        """Orchestrate feature engineering per regime with regime-specific optimization."""
        start_time = tprint_timer()
        tprint_info("🏷️ Starting per-regime feature engineering orchestration...")

        result = TacticianPreMLResult()
        result.total_samples_before_filter = sum(len(df) for df in regime_datasets.values())

        regime_results = {}
        all_selected_features = set()

        for regime_name, regime_data in regime_datasets.items():
            tprint_info(f"🏷️ Processing regime: {regime_name} ({len(regime_data)} samples)")

            try:
                # Create regime-specific configuration
                regime_config = SubPipelineConfig(
                    symbol=self.config.symbol,
                    exchange=self.config.exchange,
                    timeframe=self.config.timeframe,
                    data_dir=self.config.data_dir,
                    parallel_processing=self.config.enable_parallel_processing,
                    custom_params={
                        **self.config.custom_params,
                        'enable_per_regime_optimization': True,
                        'enable_per_cluster_optimization': self.config.enable_per_cluster_optimization,
                        'regime_assignments': regime_assignments,
                        'analyst_predictions': analyst_predictions,
                        'regime_name': regime_name,
                        'prepared_data': regime_data,
                        'role': 'tactician_regime_specific',
                        **kwargs
                    }
                )

                # Step 1: Entry Label Integration for this regime
                tprint_info(f"📈 Step 1/5: Regime-specific entry labeling for {regime_name}...")
                result.phase = OrchestrationPhase.ENTRY_LABELING
                horizon_result = await self.pre_training_pipeline._execute_multi_horizon_profit_labeler(regime_config)

                if not horizon_result.success:
                    self._log_subpipeline_failure(
                        f"Horizon labeling failed for {regime_name}",
                        horizon_result,
                        warning=True,
                    )
                    continue

                # Step 2: Feature Lookback Optimization for this regime
                tprint_info(f"⚙️ Step 2/5: Regime-specific lookback optimization for {regime_name}...")
                result.phase = OrchestrationPhase.LOOKBACK_OPTIMIZATION
                lookback_result = await self.pre_training_pipeline._execute_feature_lookback_optimization(regime_config)

                if not lookback_result.success:
                    self._log_subpipeline_failure(
                        f"Lookback optimization failed for {regime_name}",
                        lookback_result,
                        warning=True,
                    )
                    continue

                # Step 3: Interactive Feature Generation for this regime
                tprint_info(f"🔧 Step 3/5: Regime-specific interactive feature generation for {regime_name}...")
                result.phase = OrchestrationPhase.INTERACTIVE_FEATURE_GENERATION
                interactive_result = await self.pre_training_pipeline._execute_interactive_feature_generation(regime_config)

                if not interactive_result.success:
                    self._log_subpipeline_failure(
                        f"Interactive feature generation failed for {regime_name}",
                        interactive_result,
                        warning=True,
                    )
                    continue

                # Step 4: Final Feature Selection for this regime
                tprint_info(f"🎯 Step 4/5: Regime-specific feature selection for {regime_name}...")
                result.phase = OrchestrationPhase.FEATURE_SELECTION

                # Enable gate feature protection for regime-specific selection
                if GATE_PROTECTION_AVAILABLE and self.config.enable_gate_protection:
                    tprint_info(f"🛡️ Enabling gate feature protection for regime {regime_name}...")
                    enable_gate_protection()

                    # Add gate protection config to regime_config
                    if self.config.gate_protection_config:
                        regime_config.custom_params['gate_protection'] = self.config.gate_protection_config
                    else:
                        regime_config.custom_params['gate_protection'] = {
                            'enabled': True,
                            'max_gate_features_per_base': 3,
                            'min_gate_ic_improvement': 0.005,
                            'min_gate_stability': 0.4
                        }

                selection_result = await self.pre_training_pipeline._execute_final_feature_selection(regime_config)

                if not selection_result.success:
                    self._log_subpipeline_failure(
                        f"Feature selection failed for {regime_name}",
                        selection_result,
                        warning=True,
                    )
                    continue

                # Collect results for this regime
                regime_results[regime_name] = {
                    'horizon_result': horizon_result,
                    'lookback_result': lookback_result,
                    'interactive_result': interactive_result,
                    'selection_result': selection_result,
                    'final_features': selection_result.artifacts.get('final_features'),
                    'selected_features': selection_result.artifacts.get('selected_features', [])
                }

                # Track all selected features across regimes
                if regime_results[regime_name]['selected_features']:
                    all_selected_features.update(regime_results[regime_name]['selected_features'])

                tprint_success(f"✅ Regime {regime_name} processing completed")

            except Exception as e:
                tprint_error(f"❌ Failed to process regime {regime_name}: {e}")
                continue

        # Combine results from all regimes
        if regime_results:
            result.success = True
            result.phase = OrchestrationPhase.COMPLETED

            # Create combined final features by concatenating regime-specific results
            combined_features_list = []
            for regime_name, regime_result in regime_results.items():
                if regime_result['final_features'] is not None:
                    # Add regime identifier column
                    features_with_regime = regime_result['final_features'].copy()
                    features_with_regime['regime'] = regime_name
                    combined_features_list.append(features_with_regime)

            if combined_features_list:
                result.final_features = pd.concat(combined_features_list, ignore_index=True)
                result.final_feature_count = len(all_selected_features)
                result.selected_feature_names = list(all_selected_features)
            else:
                result.final_features = None
                result.final_feature_count = 0
                result.selected_feature_names = []

            # Store regime-specific results in artifacts
            result.lookback_optimization_result = {
                'regime_results': {k: v['lookback_result'].artifacts for k, v in regime_results.items()},
                'combined_features': len(all_selected_features)
            }
            result.interactive_feature_generation_result = {
                'regime_results': {
                    k: v['interactive_result'].artifacts for k, v in regime_results.items()
                },
                'total_regimes_processed': len(regime_results)
            }
            result.feature_selection_result = {
                'regime_results': {k: v['selection_result'].artifacts for k, v in regime_results.items()},
                'combined_selected_features': list(all_selected_features)
            }
        else:
            result.success = False
            result.phase = OrchestrationPhase.FAILED
            result.error_message = "No regimes processed successfully"

        result.total_samples_after_filter = (
            sum(len(df) for df in regime_datasets.values()) if regime_datasets else 0
        )
        result.execution_time = tprint_timer(start_time)

        tprint_success(f"✅ Per-regime orchestration completed: {len(regime_results)}/{len(regime_datasets)} regimes successful")
        tprint_info(f"📊 Combined feature count: {result.final_feature_count}")

        return result

    async def orchestrate(
        self,
        training_data: pd.DataFrame,
        analyst_predictions: Optional[pd.DataFrame] = None,
        regime_assignments: Optional[pd.DataFrame] = None,
        regime_data_splitting_result: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> TacticianPreMLResult:
        """
        Execute the complete pre-ML orchestration for Tactician models with per-regime optimization.

        CHANGE: Now trains on ALL data, not filtered by Analyst signals.

        Args:
            training_data: Input DataFrame with market data (15m timeframe)
            analyst_predictions: Optional Analyst predictions (legacy - no longer required)
            regime_assignments: Optional regime assignments for per-regime optimization
            regime_data_splitting_result: Complete payload from regime data splitting stage
            **kwargs: Additional parameters

        Returns:
            TacticianPreMLResult with orchestrated features and metadata
        """
        start_time = tprint_timer()
        tprint_info(f"🚀 Starting Tactician Pre-ML Orchestration ({self.config.timeframe} timeframe)...")
        tprint_info(f"📊 Input data shape: {training_data.shape}")
        tprint_info(f"🏷️ Per-regime optimization enabled: {self.config.enable_per_regime_optimization}")

        result = TacticianPreMLResult()
        result.total_samples_before_filter = len(training_data)

        try:
            # Validate pre-training pipeline availability
            if not self.pre_training_pipeline:
                raise RuntimeError("Pre-training pipeline not available")

            # Step 0: Prepare training data for configured timeframe processing
            tprint_info(f"🎯 Step 0/6: Preparing training data for {self.config.timeframe} timeframe...")
            result.phase = OrchestrationPhase.DATA_FILTERING

            # Check if we have regime data splitting results for per-regime processing
            if (self.config.enable_per_regime_optimization and
                regime_data_splitting_result and
                'unified_data' in regime_data_splitting_result):

                tprint_info("🏷️ Per-regime processing enabled - preparing regime-specific datasets...")
                regime_datasets = self._prepare_training_data_per_regime(training_data, regime_data_splitting_result)

                if len(regime_datasets) > 1:
                    tprint_success(f"🏷️ Created {len(regime_datasets)} regime-specific datasets")
                    return await self._orchestrate_per_regime(
                        regime_datasets,
                        analyst_predictions,
                        regime_assignments,
                        regime_data_splitting_result,
                        **kwargs
                    )
                else:
                    tprint_warning("⚠️ Only one regime dataset created; falling back to single dataset processing")
                    prepared_data = list(regime_datasets.values())[0]
            else:
                tprint_info("🏷️ Single dataset processing (per-regime optimization disabled or no regime data)")
                prepared_data = self._prepare_training_data(training_data, analyst_predictions)

            result.total_samples_after_filter = len(prepared_data)
            result.filter_ratio = (
                result.total_samples_after_filter / result.total_samples_before_filter
                if result.total_samples_before_filter > 0 else 0
            )

            tprint_success(f"✅ Data preparation completed ({result.filter_ratio:.2%} retained)")

            # Entry label preparation (no longer requires Analyst signals)
            entry_label_bundle: Optional[Dict[str, Any]] = None
            tprint_info("🎯 Generating entry labels from ALL market data...")
            result.phase = OrchestrationPhase.ENTRY_LABELING

            entry_label_bundle = self._create_entry_label_artifacts(
                prepared_data,
                analyst_predictions,  # Optional now
                regime_assignments
            )

            if entry_label_bundle is None:
                tprint_warning("⚠️ Failed to generate entry labels - continuing with fallback")

            if entry_label_bundle is not None:
                result.entry_labeling_result = entry_label_bundle['artifacts']
                result.entry_label_quality_metrics = entry_label_bundle['quality_metrics']

            # Determine regime split payload from explicit parameter, kwargs, or config defaults
            regime_split_payload = regime_data_splitting_result
            if regime_split_payload is None:
                regime_split_payload = kwargs.get('regime_data_splitting_result')
            if regime_split_payload is None:
                regime_split_payload = self.config.custom_params.get('regime_data_splitting_result')

            if regime_split_payload is not None and self.pre_training_pipeline:
                try:
                    self.pre_training_pipeline._current_pipeline_state['regime_data_splitting_result'] = regime_split_payload
                except AttributeError:
                    # Pre-training pipeline might not expose internal state in certain configurations
                    pass

            # Create sub-pipeline configuration
            sub_config = SubPipelineConfig(
                symbol=self.config.symbol,
                exchange=self.config.exchange,
                timeframe=self.config.timeframe,  # 15m for Tactician
                data_dir=self.config.data_dir,
                parallel_processing=self.config.enable_parallel_processing,
                custom_params={
                    **self.config.custom_params,
                    'enable_per_regime_optimization': self.config.enable_per_regime_optimization,
                    'enable_per_cluster_optimization': self.config.enable_per_cluster_optimization,
                    'regime_assignments': regime_assignments,
                    'analyst_predictions': analyst_predictions,
                    'precomputed_labeling_result': entry_label_bundle['artifacts'] if entry_label_bundle else None,
                    'entry_label_quality_metrics': entry_label_bundle['quality_metrics'] if entry_label_bundle else None,
                    'role': 'tactician',  # Mark as Tactician orchestration
                    'prepared_data': prepared_data,  # Pass prepared data
                    **kwargs
                }
            )

            if regime_split_payload is not None:
                sub_config.custom_params['regime_data_splitting_result'] = regime_split_payload

            tprint_info("📋 Configuration:")
            tprint_info(f"  - Timeframe: {self.config.timeframe} (feature engineering cadence)")
            tprint_info(f"  - Samples after preparation: {len(prepared_data)}")
            tprint_info(f"  - Per-regime optimization: {self.config.enable_per_regime_optimization}")
            tprint_info(f"  - Per-cluster optimization: {self.config.enable_per_cluster_optimization}")

            run_metadata: Optional[Dict[str, Any]] = None
            if self.pre_training_pipeline is not None:
                run_metadata = getattr(self.pre_training_pipeline, '_run_metadata', None)
                if not run_metadata:
                    run_metadata = self.pre_training_pipeline._gather_run_metadata(sub_config)
                self.pre_training_pipeline._run_metadata = dict(run_metadata)

            # Step 1: Entry Label Integration / Multi-Horizon compatibility layer
            tprint_info("📈 Step 1/5: Integrating entry labels with multi-horizon pipeline...")
            result.phase = OrchestrationPhase.HORIZON_LABELING
            horizon_result = await self.pre_training_pipeline._execute_multi_horizon_profit_labeler(
                sub_config,
                run_metadata or {},
            )

            if not horizon_result.success:
                error_code, message = self._log_subpipeline_failure(
                    "Horizon labeling failed",
                    horizon_result,
                )
                raise RuntimeError(f"Horizon labeling failed ({error_code or 'unknown_error'}): {message}")

            result.horizon_labeling_result = horizon_result.artifacts
            tprint_success("✅ Horizon labeling completed")

            # Step 2: Feature Lookback Optimization (per-regime/cluster)
            tprint_info("⚙️ Step 2/5: Feature Lookback Optimization (per-regime/cluster)...")
            result.phase = OrchestrationPhase.LOOKBACK_OPTIMIZATION
            lookback_result = await self.pre_training_pipeline._execute_feature_lookback_optimization(
                sub_config,
                run_metadata or {},
            )

            if not lookback_result.success:
                error_code, message = self._log_subpipeline_failure(
                    "Lookback optimization failed",
                    lookback_result,
                )
                raise RuntimeError(f"Lookback optimization failed ({error_code or 'unknown_error'}): {message}")

            result.lookback_optimization_result = lookback_result.artifacts
            tprint_success("✅ Lookback optimization completed")

            # Step 3: Interactive Feature Generation
            tprint_info("🔧 Step 3/5: Interactive Feature Generation...")
            result.phase = OrchestrationPhase.INTERACTIVE_FEATURE_GENERATION
            interactive_result = await self.pre_training_pipeline._execute_interactive_feature_generation(
                sub_config,
                run_metadata or {},
            )

            if not interactive_result.success:
                error_code, message = self._log_subpipeline_failure(
                    "Interactive feature generation failed",
                    interactive_result,
                )
                raise RuntimeError(
                    f"Interactive feature generation failed ({error_code or 'unknown_error'}): {message}"
                )

            result.interactive_feature_generation_result = interactive_result.artifacts
            interactive_artifacts = interactive_result.artifacts.get('interactive_feature_generation_result', {})
            feature_names = interactive_artifacts.get('feature_names', [])
            features_df = interactive_artifacts.get('features')
            if feature_names:
                result.total_features_generated = len(feature_names)
            elif hasattr(features_df, 'columns'):
                result.total_features_generated = len(features_df.columns)
            else:
                result.total_features_generated = int(interactive_artifacts.get('total_features', 0))
            tprint_success(
                f"✅ Interactive feature generation completed ({result.total_features_generated} features)"
            )

            # Step 4: Final Feature Selection
            tprint_info("🎯 Step 4/5: Final Feature Selection (multi-stage)...")
            result.phase = OrchestrationPhase.FEATURE_SELECTION

            # Enable gate feature protection if available
            if GATE_PROTECTION_AVAILABLE and self.config.enable_gate_protection:
                tprint_info("🛡️ Enabling gate feature protection for final feature selection...")
                enable_gate_protection()

                # Add gate protection config to sub_config
                if self.config.gate_protection_config:
                    sub_config.custom_params['gate_protection'] = self.config.gate_protection_config
                else:
                    sub_config.custom_params['gate_protection'] = {
                        'enabled': True,
                        'max_gate_features_per_base': 3,
                        'min_gate_ic_improvement': 0.005,
                        'min_gate_stability': 0.4
                    }

            selection_result = await self.pre_training_pipeline._execute_final_feature_selection(
                sub_config,
                run_metadata or {},
            )

            if not selection_result.success:
                error_code, message = self._log_subpipeline_failure(
                    "Feature selection failed",
                    selection_result,
                )
                raise RuntimeError(f"Feature selection failed ({error_code or 'unknown_error'}): {message}")

            result.feature_selection_result = selection_result.artifacts
            result.final_features = selection_result.artifacts.get('final_features')
            result.selected_feature_names = selection_result.artifacts.get('selected_features', [])
            result.final_feature_count = len(result.selected_feature_names) if result.selected_feature_names else 0
            tprint_success(f"✅ Feature selection completed ({result.final_feature_count} final features)")

            # Step 5: Tactician 5m Entry Optimization
            tprint_info("🎯 Step 5/5: Tactician 5m Entry Optimization (ML-based)...")
            result.phase = OrchestrationPhase.TACTICIAN_5M_OPTIMIZATION

            # Initialize 5m entry optimizer with ML models only
            tactician_5m_optimizer = Tactician5mEntryOptimizer(self.config.tactician_5m_config)

            # Tactician operates on 5m data only - extract from analyst signals
            analyst_signals_series = analyst_predictions.get('analyst_signal', pd.Series()) if analyst_predictions is not None else pd.Series()

            # Perform ML-based entry optimization
            entry_optimization_result = tactician_5m_optimizer.optimize_entries(
                data_5m=prepared_data,  # Use prepared data as 5m data
                analyst_signals_15m=analyst_signals_series
            )

            if entry_optimization_result.success:
                result.tactician_5m_result = {
                    'optimal_entries': entry_optimization_result.optimal_entries,
                    'entry_scores': entry_optimization_result.entry_scores,
                    'green_periods_analyzed': entry_optimization_result.green_periods_analyzed,
                    'total_entries_found': entry_optimization_result.total_entries_found,
                    'avg_entry_quality': entry_optimization_result.avg_entry_quality,
                    'method_used': entry_optimization_result.method_used.value if entry_optimization_result.method_used else None,
                    'ml_model': self.config.tactician_5m_config.optimization_method.value
                }
                tprint_success(f"✅ 5m ML Entry optimization completed ({len(entry_optimization_result.optimal_entries)} optimal entries using {self.config.tactician_5m_config.optimization_method.value})")
            else:
                tprint_warning(f"⚠️ 5m ML Entry optimization failed: {entry_optimization_result.error_message}")
                result.tactician_5m_result = {'error': entry_optimization_result.error_message}

            # Mark as completed
            result.success = True
            result.phase = OrchestrationPhase.COMPLETED
            result.execution_time = tprint_timer(start_time)

            tprint_success(f"✅ Tactician Pre-ML Orchestration completed in {result.execution_time:.2f}s")
            tprint_info(f"📊 Final feature count: {result.final_feature_count}")
            tprint_info(f"📊 Data retention after preparation: {result.filter_ratio:.2%}")
            if entry_optimization_result.success:
                tprint_info(f"🎯 5m Optimal entries found: {len(entry_optimization_result.optimal_entries)}")

            return result

        except Exception as e:
            result.success = False
            result.phase = OrchestrationPhase.FAILED
            result.error_message = str(e)
            result.execution_time = tprint_timer(start_time)

            tprint_error(f"❌ Tactician Pre-ML Orchestration failed: {e}")
            tprint_error(f"Error details: {traceback.format_exc()}")
            raise

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the orchestrator."""
        return {
            'config': {
                'timeframe': self.config.timeframe,
                'analyst_confidence_threshold': self.config.analyst_confidence_threshold,
                'require_analyst_signals': self.config.require_analyst_signals,
                'enable_per_regime_optimization': self.config.enable_per_regime_optimization,
                'enable_per_cluster_optimization': self.config.enable_per_cluster_optimization,
                'output_directory': self.config.output_directory
            },
            'component_availability': {
                'pre_training_pipeline': self.pre_training_pipeline is not None
            }
        }

# Convenience function for external usage
async def execute_tactician_pre_ml_orchestration(
    training_data: pd.DataFrame,
    analyst_predictions: Optional[pd.DataFrame] = None,
    regime_assignments: Optional[pd.DataFrame] = None,
    config: Optional[TacticianPreMLConfig] = None,
    regime_data_splitting_result: Optional[Dict[str, Any]] = None,
    **kwargs
) -> TacticianPreMLResult:
    """
    Execute Tactician pre-ML orchestration.

    Args:
        training_data: Input DataFrame with market data (15m timeframe)
        analyst_predictions: Analyst ensemble predictions (for reference only)
        regime_assignments: Optional regime assignments for per-regime optimization
        regime_data_splitting_result: Complete payload from regime data splitting stage
        config: Optional configuration
        **kwargs: Additional parameters

    Returns:
        TacticianPreMLResult with orchestrated features and metadata
    """
    orchestrator = TacticianPreMLOrchestrator(config)
    return await orchestrator.orchestrate(
        training_data,
        analyst_predictions,
        regime_assignments,
        regime_data_splitting_result=regime_data_splitting_result,
        **kwargs
    )

    def _optimized_rolling_operation(self, data: pd.Series, operation: str,
                                   window: int, **kwargs) -> pd.Series:
        """Perform optimized rolling operation using VectorBT Rolling Optimizer."""
        if self.vectorbt_optimizer is not None:
            try:
                if operation == 'mean':
                    return self.vectorbt_optimizer.rolling_mean(data, window=window, **kwargs)
                elif operation == 'std':
                    return self.vectorbt_optimizer.rolling_std(data, window=window, **kwargs)
                elif operation == 'var':
                    return self.vectorbt_optimizer.rolling_var(data, window=window, **kwargs)
                elif operation == 'min':
                    return self.vectorbt_optimizer.rolling_min(data, window=window, **kwargs)
                elif operation == 'max':
                    return self.vectorbt_optimizer.rolling_max(data, window=window, **kwargs)
                elif operation == 'sum':
                    return self.vectorbt_optimizer.rolling_sum(data, window=window, **kwargs)
                elif operation == 'apply':
                    func = kwargs.get('func')
                    return self.vectorbt_optimizer.rolling_apply(data, func, window=window, **kwargs)
                else:
                    raise ValueError(f"Unsupported operation: {operation}")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT Rolling Optimizer failed for {operation}: {e}, using fallback")
                return self._fallback_rolling_operation(data, operation, window, **kwargs)
        else:
            return self._fallback_rolling_operation(data, operation, window, **kwargs)

    def _fallback_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        elif operation == 'apply':
            func = kwargs.get('func')
            return data.rolling(window=window).apply(func, **kwargs)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and
                VECTORBT_AVAILABLE)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Legacy method - now uses optimized rolling operations."""
        return self._optimized_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Legacy method - now uses fallback rolling operations."""
        return self._fallback_rolling_operation(data, operation, window, **kwargs)

    def _vectorbt_apply_operation(self, data: pd.Series, func,
                                 window: int, **kwargs) -> pd.Series:
        """Legacy method - now uses optimized rolling operations."""
        return self._optimized_rolling_operation(data, 'apply', window, func=func, **kwargs)

    def _optimize_matrix_operations(self, X: np.ndarray, operation_type: str, **kwargs) -> Any:
        """
        Optimize matrix operations using Unified Vectorization Manager.

        Args:
            X: Input matrix
            operation_type: Type of matrix operation
            **kwargs: Additional parameters

        Returns:
            Optimized operation result
        """
        if self.vectorization_manager is not None:
            try:
                tprint_debug(f"🔄 Using Unified Vectorization Manager for {operation_type}")

                # Create operation configuration
                config = OperationConfig(
                    operation_type=OperationType.MATRIX_MULTIPLICATION if operation_type == 'matrix_mult' else OperationType.STATISTICAL_COMPUTATION,
                    data_size=len(X),
                    data_dimensions=X.shape,
                    memory_budget_mb=self.config.memory_limit_gb * 1024
                )

                # Prepare data for optimization
                data = {'matrix': X, **kwargs}

                # Use VectorBT optimization
                result = self.vectorization_manager.optimize_operation(
                    config.operation_type,
                    data,
                    config
                )

                tprint_success(f"✅ Matrix operation {operation_type} optimized (performance gain: {result.performance_gain:.2f}x)")
                return result.result

            except Exception as e:
                tprint_warning(f"⚠️ Unified Vectorization Manager failed for {operation_type}: {e}, using fallback")
                return self._fallback_matrix_operation(X, operation_type, **kwargs)
        else:
            tprint_warning(f"⚠️ Unified Vectorization Manager not available, using fallback for {operation_type}")
            return self._fallback_matrix_operation(X, operation_type, **kwargs)

    def _fallback_matrix_operation(self, X: np.ndarray, operation_type: str, **kwargs) -> Any:
        """
        Fallback matrix operation using standard numpy/pandas.

        Args:
            X: Input matrix
            operation_type: Type of matrix operation
            **kwargs: Additional parameters

        Returns:
            Operation result
        """
        try:
            if operation_type == 'matrix_mult':
                other = kwargs.get('other')
                return np.dot(X, other) if other is not None else X
            elif operation_type == 'statistical':
                return {
                    'mean': np.mean(X, axis=0),
                    'std': np.std(X, axis=0),
                    'min': np.min(X, axis=0),
                    'max': np.max(X, axis=0)
                }
            else:
                raise ValueError(f"Unsupported matrix operation: {operation_type}")

        except Exception as e:
            tprint_error(f"❌ Fallback matrix operation failed for {operation_type}: {e}")
            raise
