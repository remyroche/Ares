"""
Tactician Pre-ML Orchestration - 15m Timeframe Feature Engineering

This orchestrator applies the complete pre-training pipeline for Tactician models:
1. Applies differentiated horizon labeling + Optimizes feature lookback periods + Generates PID features + Selects final features
2. Uses 15m timeframe with per-regime/cluster optimisation
3. Uses the pipeline present in src/training/steps/MODELS_TRAINING/

TACTICIAN PRE-ML CONFIGURATION:
- Timeframe: 15m (as specified for tactician_pre_ml_orchestration step)
- Training Data: All market data (processed through the standard pre-training pipeline)
- Output: Features optimized for Tactician model training
- Per-regime optimization: Yes, using regime assignments from market_analysis
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import traceback
from scipy.signal import find_peaks

# Import pre-training sub-pipeline
try:
    from ...pre_training.sub_pipeline import (
        PreTrainingSubPipeline, SubPipelineConfig, SubPipelineResult, SubPipelineStatus
    )
    PRE_TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import PreTrainingSubPipeline: {e}")
    PRE_TRAINING_AVAILABLE = False

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
    PID_GENERATION = "pid_generation"
    FEATURE_SELECTION = "feature_selection"
    COMPLETED = "completed"
    FAILED = "failed"


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

    # Regime-aware settings
    enable_regime_adaptive_labeling: bool = True
    regime_specific_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class TacticianPreMLConfig:
    """Configuration for Tactician pre-ML orchestration."""
    # Data configuration
    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    timeframe: str = "15m"  # TACTICIAN PRE-ML USES 15m TIMEFRAME
    data_dir: str = "historical_data"
    
    # Analyst signal filtering
    analyst_confidence_threshold: float = 0.004  # 0.4% threshold for "green" signals
    require_analyst_signals: bool = True

    # Differentiated labeling configuration
    labeling_config: TacticianLabelingConfig = field(default_factory=TacticianLabelingConfig)

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
    pid_generation_result: Optional[Dict[str, Any]] = None
    feature_selection_result: Optional[Dict[str, Any]] = None
    
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

    def create_entry_timing_labels(
        self,
        data: pd.DataFrame,
        analyst_signals: pd.Series,
        regime_assignments: Optional[pd.Series] = None
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """Generate entry timing labels constrained to Analyst green light periods."""
        tprint_info("🎯 Creating tactician entry timing labels from Analyst signals")

        if not isinstance(analyst_signals, pd.Series):
            analyst_signals = pd.Series(analyst_signals, index=data.index)

        analyst_signals = analyst_signals.reindex(data.index).fillna(0.0)

        if regime_assignments is not None:
            regime_assignments = regime_assignments.reindex(data.index)

        labels = pd.Series(0.0, index=data.index, dtype=float)

        green_periods = self._find_green_periods(analyst_signals)
        tprint_info(f"📊 Found {len(green_periods)} Analyst green light periods")

        if len(green_periods) == 0:
            tprint_warning("⚠️ No green light periods identified; returning empty labels")
            return labels, {}

        entry_points: List[pd.Timestamp] = []

        for period in green_periods:
            period_slice = data.iloc[period['start']:period['end']]
            period_labels = self._find_optimal_entries_in_period(
                period_slice,
                regime_assignments,
            )

            labels.loc[period_slice.index] = period_labels
            entry_points.extend(period_slice.index[period_labels > 0].tolist())

        quality_metrics = self._calculate_labeling_quality_metrics(
            data,
            labels,
            entry_points,
            green_periods
        )

        tprint_success(
            "✅ Entry labeling completed ("
            f"{int((labels > 0).sum())} optimal entries, quality={quality_metrics.get('overall_quality', 0):.3f})"
        )

        return labels, quality_metrics

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
        """Estimate entry quality favouring minimal adverse movement."""
        if future_data.empty:
            return 0.0

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


class TacticianPreMLOrchestrator:
    """
    Tactician Pre-ML Orchestration.

    Orchestrates the complete pre-training pipeline for Tactician models on 15m timeframe.
    Applies differentiated horizon labeling + Optimizes feature lookback periods + Generates PID features + Selects final features.
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

            # Initialise differentiated entry labeler
            self.labeler = TacticianDifferentiatedLabeler(self.config.labeling_config)

            tprint_success(f"✅ TacticianPreMLOrchestrator initialized (timeframe: {self.config.timeframe})")
            tprint_info(f"🎯 Analyst signal threshold: {self.config.analyst_confidence_threshold:.2%}")
            tprint_info(f"⏰ Operating on {self.config.timeframe} timeframe for feature engineering")
            
        except Exception as e:
            tprint_error(f"❌ Failed to initialize TacticianPreMLOrchestrator: {e}")
            raise

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
        """Create precomputed entry label artifacts from Analyst signals."""
        green_series = self._extract_green_light_series(analyst_predictions, prepared_data.index)
        if green_series is None:
            return None

        regime_series = self._extract_regime_series(regime_assignments)

        entry_labels, quality_metrics = self.labeler.create_entry_timing_labels(
            prepared_data,
            green_series,
            regime_series
        )

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

        Args:
            training_data: Input DataFrame (15m timeframe)
            analyst_predictions: Analyst ensemble predictions (for reference only)

        Returns:
            Prepared DataFrame for 15m timeframe processing
        """
        tprint_info(f"🔍 Preparing training data for {self.config.timeframe} timeframe processing...")
        tprint_info(f"📊 Input data shape: {training_data.shape}")
        tprint_info(f"📊 Timeframe: {self.config.timeframe}")

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
                    tprint_warning(f"⚠️ Horizon labeling failed for {regime_name}: {horizon_result.error_message}")
                    continue

                # Step 2: Feature Lookback Optimization for this regime
                tprint_info(f"⚙️ Step 2/5: Regime-specific lookback optimization for {regime_name}...")
                result.phase = OrchestrationPhase.LOOKBACK_OPTIMIZATION
                lookback_result = await self.pre_training_pipeline._execute_feature_lookback_optimization(regime_config)

                if not lookback_result.success:
                    tprint_warning(f"⚠️ Lookback optimization failed for {regime_name}: {lookback_result.error_message}")
                    continue

                # Step 3: PID-Based Feature Generation for this regime
                tprint_info(f"🔧 Step 3/5: Regime-specific PID generation for {regime_name}...")
                result.phase = OrchestrationPhase.PID_GENERATION
                pid_result = await self.pre_training_pipeline._execute_pid_based_feature_generation(regime_config)

                if not pid_result.success:
                    tprint_warning(f"⚠️ PID generation failed for {regime_name}: {pid_result.error_message}")
                    continue

                # Step 4: Final Feature Selection for this regime
                tprint_info(f"🎯 Step 4/5: Regime-specific feature selection for {regime_name}...")
                result.phase = OrchestrationPhase.FEATURE_SELECTION
                selection_result = await self.pre_training_pipeline._execute_final_feature_selection(regime_config)

                if not selection_result.success:
                    tprint_warning(f"⚠️ Feature selection failed for {regime_name}: {selection_result.error_message}")
                    continue

                # Collect results for this regime
                regime_results[regime_name] = {
                    'horizon_result': horizon_result,
                    'lookback_result': lookback_result,
                    'pid_result': pid_result,
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
            result.pid_generation_result = {
                'regime_results': {k: v['pid_result'].artifacts for k, v in regime_results.items()},
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

        Args:
            training_data: Input DataFrame with market data (15m timeframe)
            analyst_predictions: Analyst ensemble predictions for filtering
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
            
            # Step 1: Entry Label Integration / Multi-Horizon compatibility layer
            tprint_info("📈 Step 1/5: Integrating entry labels with multi-horizon pipeline...")
            result.phase = OrchestrationPhase.HORIZON_LABELING
            horizon_result = await self.pre_training_pipeline._execute_multi_horizon_profit_labeler(sub_config)

            if not horizon_result.success:
                raise RuntimeError(f"Horizon labeling failed: {horizon_result.error_message}")
            
            result.horizon_labeling_result = horizon_result.artifacts
            tprint_success("✅ Horizon labeling completed")
            
            # Step 2: Feature Lookback Optimization (per-regime/cluster)
            tprint_info("⚙️ Step 2/5: Feature Lookback Optimization (per-regime/cluster)...")
            result.phase = OrchestrationPhase.LOOKBACK_OPTIMIZATION
            lookback_result = await self.pre_training_pipeline._execute_feature_lookback_optimization(sub_config)

            if not lookback_result.success:
                raise RuntimeError(f"Lookback optimization failed: {lookback_result.error_message}")
            
            result.lookback_optimization_result = lookback_result.artifacts
            tprint_success("✅ Lookback optimization completed")
            
            # Step 3: PID-Based Feature Generation
            tprint_info("🔧 Step 3/5: PID-Based Feature Generation...")
            result.phase = OrchestrationPhase.PID_GENERATION
            pid_result = await self.pre_training_pipeline._execute_pid_based_feature_generation(sub_config)

            if not pid_result.success:
                raise RuntimeError(f"PID generation failed: {pid_result.error_message}")
            
            result.pid_generation_result = pid_result.artifacts
            result.total_features_generated = pid_result.artifacts.get('total_features', 0)
            tprint_success(f"✅ PID generation completed ({result.total_features_generated} features)")
            
            # Step 4: Final Feature Selection
            tprint_info("🎯 Step 4/5: Final Feature Selection (multi-stage)...")
            result.phase = OrchestrationPhase.FEATURE_SELECTION
            selection_result = await self.pre_training_pipeline._execute_final_feature_selection(sub_config)

            if not selection_result.success:
                raise RuntimeError(f"Feature selection failed: {selection_result.error_message}")
            
            result.feature_selection_result = selection_result.artifacts
            result.final_features = selection_result.artifacts.get('final_features')
            result.selected_feature_names = selection_result.artifacts.get('selected_features', [])
            result.final_feature_count = len(result.selected_feature_names) if result.selected_feature_names else 0
            tprint_success(f"✅ Feature selection completed ({result.final_feature_count} final features)")
            
            # Mark as completed
            result.success = True
            result.phase = OrchestrationPhase.COMPLETED
            result.execution_time = tprint_timer(start_time)
            
            tprint_success(f"✅ Tactician Pre-ML Orchestration completed in {result.execution_time:.2f}s")
            tprint_info(f"📊 Final feature count: {result.final_feature_count}")
            tprint_info(f"📊 Data retention after preparation: {result.filter_ratio:.2%}")
            
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
