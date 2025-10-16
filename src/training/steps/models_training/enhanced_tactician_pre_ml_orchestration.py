"""
Enhanced Tactician Pre-ML Orchestration - 15m Timeframe Feature Engineering

This orchestrator applies the complete pre-training pipeline for Tactician models with
differentiated horizon labeling focused on optimal entry timing:

1. **Differentiated Horizon Labeling** - Tactician-specific labeling for optimal entry points
2. **Feature Lookback Period Optimization** - Per-regime/cluster optimization
3. **Interactive Feature Generation** - Interactive and control-inspired features for entry timing
4. **Final Feature Selection** - Per-regime feature selection

TACTICIAN-SPECIFIC CONFIGURATION:
- Timeframe: 15m (for Analyst signal integration)
- Training Data: Filtered by Analyst green lights (15m timeframe)
- Output: Features optimized for Tactician model training (5m execution)
- Per-regime optimization: Yes, using regime assignments from market_analysis
- Labeling Focus: Optimal entry timing (least price adversarial movement)

KEY DIFFERENCES FROM ANALYST:
- Labels focus on entry timing rather than directional prediction
- Trains on Analyst green light signals from 15m timeframe
- Optimizes for finding the best entry point within Analyst signals
- Uses different horizon labeling methodology
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import traceback
import warnings
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

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

# Import corrected ML-based labeling
try:
    from .corrected_ml_entry_timing_labeler import CorrectedMLEntryTimingLabeler, CorrectedMLEntryTimingConfig

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

# Import Unified Vectorization Manager
try:
    from src.feature_selection.vectorbt.vectorbt_unified_framework import (
        VectorBTUnifiedFramework, create_vectorbt_unified_framework
    )
    UNIFIED_VECTORIZATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ WARNING: Unified Vectorization Manager not available: {e}")
    UNIFIED_VECTORIZATION_AVAILABLE = False

except ImportError:

    cp = None
    ML_LABELING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Corrected ML-based labeling not available: {e}")
    ML_LABELING_AVAILABLE = False

class OrchestrationPhase(Enum):
    """Orchestration execution phases."""
    DATA_FILTERING = "data_filtering"
    ANALYST_SIGNAL_INTEGRATION = "analyst_signal_integration"
    DIFFERENTIATED_LABELING = "differentiated_labeling"
    LOOKBACK_OPTIMIZATION = "lookback_optimization"
    INTERACTIVE_FEATURE_GENERATION = "interactive_feature_generation"
    FEATURE_SELECTION = "feature_selection"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class TacticianLabelingConfig:
    """Configuration for Tactician-specific labeling."""
    # Entry timing optimization
    min_entry_window_minutes: int = 5  # Minimum window to find entry
    max_entry_window_minutes: int = 60  # Maximum window to find entry
    entry_quality_threshold: float = 0.7  # Quality threshold for entry points

    # Price movement analysis
    max_adverse_movement_pct: float = 0.5  # Max adverse movement allowed
    min_favorable_movement_pct: float = 0.2  # Min favorable movement required

    # Horizon settings
    lookback_horizons: List[int] = field(default_factory=lambda: [3, 6, 12, 24, 48])  # 15m periods
    forward_horizons: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])  # 15m periods

    # Regime-specific parameters
    enable_regime_adaptive_labeling: bool = True
    regime_specific_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)

@dataclass
class EnhancedTacticianPreMLConfig:
    """Enhanced configuration for Tactician pre-ML orchestration."""
    # Data configuration
    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    timeframe: str = "15m"  # TACTICIAN PRE-ML USES 15m TIMEFRAME
    data_dir: str = "historical_data"

    # Analyst signal integration
    analyst_confidence_threshold: float = 0.004  # 0.4% threshold for "green" signals
    require_analyst_signals: bool = True
    analyst_signal_lookback_hours: int = 24  # Hours to look back for Analyst signals

    # Tactician-specific labeling
    labeling_config: TacticianLabelingConfig = field(default_factory=TacticianLabelingConfig)

    # ML-based labeling
    enable_ml_labeling: bool = True
    ml_labeling_config: Optional[CorrectedMLEntryTimingConfig] = None

    # Execution parameters
    enable_per_regime_optimization: bool = False  # Tactician is NOT per-regime
    enable_per_cluster_optimization: bool = False  # Tactician is NOT per-cluster

    # Output configuration
    output_directory: str = "generated/enhanced_tactician_pre_ml"
    save_intermediate_results: bool = True

    # Hardware optimization
    enable_parallel_processing: bool = True
    memory_limit_gb: float = 8.0

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EnhancedTacticianPreMLResult:
    """Enhanced result of Tactician pre-ML orchestration."""
    # Execution metadata
    success: bool = False
    execution_time: float = 0.0
    phase: OrchestrationPhase = OrchestrationPhase.DATA_FILTERING

    # Data filtering results
    total_samples_before_filter: int = 0
    total_samples_after_analyst_filter: int = 0
    total_samples_after_labeling: int = 0
    analyst_signal_coverage: float = 0.0

    # Step results
    analyst_signal_integration_result: Optional[Dict[str, Any]] = None
    differentiated_labeling_result: Optional[Dict[str, Any]] = None
    lookback_optimization_result: Optional[Dict[str, Any]] = None
    interactive_feature_generation_result: Optional[Dict[str, Any]] = None
    feature_selection_result: Optional[Dict[str, Any]] = None

    # Output data
    final_features: Optional[pd.DataFrame] = None
    selected_feature_names: Optional[List[str]] = None
    entry_timing_labels: Optional[pd.Series] = None

    # Metadata
    total_features_generated: int = 0
    final_feature_count: int = 0
    labeling_quality_metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None

class TacticianDifferentiatedLabeler:
    """Tactician-specific labeling focused on optimal entry timing."""

    def __init__(self, config: TacticianLabelingConfig):
        self.config = config
        self.logger = system_logger.getChild('TacticianDifferentiatedLabeler')

    def create_entry_timing_labels(
        self,
        data: pd.DataFrame,
        analyst_signals: pd.Series,
        regime_assignments: Optional[pd.Series] = None
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Create differentiated labels focused on optimal entry timing.

        Args:
            data: Market data with OHLCV columns
            analyst_signals: Analyst green light signals (1 for green, 0 for red)
            regime_assignments: Optional regime assignments for adaptive labeling

        Returns:
            Tuple of (entry_timing_labels, quality_metrics)
        """
        tprint_info("🎯 Creating Tactician-specific entry timing labels...")

        labels = pd.Series(0, index=data.index, dtype=float)
        quality_metrics = {}

        # Find Analyst green light periods
        green_periods = self._find_green_periods(analyst_signals)
        tprint_info(f"📊 Found {len(green_periods)} green light periods")

        if len(green_periods) == 0:
            tprint_warning("⚠️ No green light periods found for labeling")
            return labels, quality_metrics

        # Process each green period to find optimal entry points
        entry_points = []
        for period in green_periods:
            period_data = data.iloc[period['start']:period['end']]
            period_labels = self._find_optimal_entries_in_period(
                period_data,
                period['start'],
                regime_assignments
            )
            labels.iloc[period['start']:period['end']] = period_labels
            entry_points.extend(period_labels[period_labels > 0].index.tolist())

        # Calculate quality metrics
        quality_metrics = self._calculate_labeling_quality_metrics(
            data, labels, entry_points, green_periods
        )

        tprint_success(f"✅ Created {len(entry_points)} optimal entry points")
        tprint_info(f"📊 Labeling quality: {quality_metrics.get('overall_quality', 0):.3f}")

        return labels, quality_metrics

    def _find_green_periods(self, analyst_signals: pd.Series) -> List[Dict[str, int]]:
        """Find continuous green light periods from Analyst signals."""
        green_periods = []
        in_green = False
        start_idx = 0

        for i, signal in enumerate(analyst_signals):
            if signal > 0 and not in_green:
                # Start of green period
                in_green = True
                start_idx = i
            elif signal == 0 and in_green:
                # End of green period
                in_green = False
                if i - start_idx >= 3:  # Minimum period length
                    green_periods.append({
                        'start': start_idx,
                        'end': i,
                        'length': i - start_idx
                    })

        # Handle case where period extends to end
        if in_green and len(analyst_signals) - start_idx >= 3:
            green_periods.append({
                'start': start_idx,
                'end': len(analyst_signals),
                'length': len(analyst_signals) - start_idx
            })

        return green_periods

    def _find_optimal_entries_in_period(
        self,
        period_data: pd.DataFrame,
        start_offset: int,
        regime_assignments: Optional[pd.Series] = None
    ) -> pd.Series:
        """Find optimal entry points within a green light period."""
        labels = pd.Series(0, index=period_data.index, dtype=float)

        if len(period_data) < self.config.min_entry_window_minutes:
            return labels

        # Calculate price movement metrics for each potential entry point
        entry_scores = []
        for i in range(len(period_data) - self.config.min_entry_window_minutes):
            entry_point = period_data.iloc[i]
            future_data = period_data.iloc[i+1:]

            # Calculate entry quality score
            score = self._calculate_entry_quality_score(
                entry_point, future_data, i + start_offset, regime_assignments
            )
            entry_scores.append(score)

        # Find peaks in entry scores (optimal entry points)
        if len(entry_scores) > 3:
            scores_array = np.array(entry_scores)
            peaks, _ = find_peaks(
                scores_array,
                height=self.config.entry_quality_threshold,
                distance=self.config.min_entry_window_minutes
            )

            # Set labels for optimal entry points
            for peak in peaks:
                if peak < len(labels):
                    labels.iloc[peak] = scores_array[peak]

        return labels

    def _calculate_entry_quality_score(
        self,
        entry_point: pd.Series,
        future_data: pd.DataFrame,
        absolute_index: int,
        regime_assignments: Optional[pd.Series] = None
    ) -> float:
        """Calculate quality score for a potential entry point."""
        if len(future_data) == 0:
            return 0.0

        # Get regime-specific parameters if available
        regime_params = self._get_regime_parameters(absolute_index, regime_assignments)

        # Calculate price movements
        entry_price = entry_point['close']
        future_highs = future_data['high']
        future_lows = future_data['low']
        future_closes = future_data['close']

        # Calculate adverse movement (worst case)
        max_adverse = (future_lows.min() - entry_price) / entry_price * 100
        max_adverse = max(max_adverse, 0)  # Only positive adverse movement

        # Calculate favorable movement (best case)
        max_favorable = (future_highs.max() - entry_price) / entry_price * 100
        max_favorable = max(max_favorable, 0)  # Only positive favorable movement

        # Calculate entry quality based on risk-reward ratio
        if max_adverse > regime_params['max_adverse_movement_pct']:
            return 0.0  # Too much adverse movement

        if max_favorable < regime_params['min_favorable_movement_pct']:
            return 0.0  # Not enough favorable movement

        # Calculate risk-reward ratio
        risk_reward_ratio = max_favorable / (max_adverse + 1e-8)

        # Calculate timing score (earlier entries are better)
        timing_score = 1.0 / (1.0 + len(future_data) / 100.0)

        # Calculate volatility-adjusted score
        volatility = future_data['close'].pct_change().std() * 100
        volatility_score = 1.0 / (1.0 + volatility / 10.0)

        # Combine scores
        quality_score = (
            risk_reward_ratio * 0.4 +
            timing_score * 0.3 +
            volatility_score * 0.3
        )

        return min(quality_score, 1.0)  # Cap at 1.0

    def _get_regime_parameters(
        self,
        index: int,
        regime_assignments: Optional[pd.Series]
    ) -> Dict[str, float]:
        """Get regime-specific parameters for labeling."""
        if regime_assignments is not None and index < len(regime_assignments):
            regime = regime_assignments.iloc[index]
            regime_key = f"regime_{regime}"

            if regime_key in self.config.regime_specific_thresholds:
                return self.config.regime_specific_thresholds[regime_key]

        # Default parameters
        return {
            'max_adverse_movement_pct': self.config.max_adverse_movement_pct,
            'min_favorable_movement_pct': self.config.min_favorable_movement_pct
        }

    def _calculate_labeling_quality_metrics(
        self,
        data: pd.DataFrame,
        labels: pd.Series,
        entry_points: List[int],
        green_periods: List[Dict[str, int]]
    ) -> Dict[str, float]:
        """Calculate quality metrics for the labeling process."""
        metrics = {}

        # Basic metrics
        total_samples = len(data)
        labeled_samples = (labels > 0).sum()
        green_period_samples = sum(period['length'] for period in green_periods)

        metrics['labeling_coverage'] = labeled_samples / total_samples if total_samples > 0 else 0
        metrics['green_period_coverage'] = green_period_samples / total_samples if total_samples > 0 else 0
        metrics['entry_point_density'] = len(entry_points) / green_period_samples if green_period_samples > 0 else 0

        # Quality distribution
        if labeled_samples > 0:
            quality_scores = labels[labels > 0]
            metrics['avg_entry_quality'] = quality_scores.mean()
            metrics['min_entry_quality'] = quality_scores.min()
            metrics['max_entry_quality'] = quality_scores.max()
            metrics['entry_quality_std'] = quality_scores.std()

        # Overall quality score
        metrics['overall_quality'] = (
            metrics.get('labeling_coverage', 0) * 0.3 +
            metrics.get('entry_point_density', 0) * 0.3 +
            metrics.get('avg_entry_quality', 0) * 0.4
        )

        return metrics

class TacticianPIDFeatureGenerator:
    """PID-based feature generation for Tactician entry timing."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild('TacticianPIDFeatureGenerator')

    def generate_pid_features(
        self,
        data: pd.DataFrame,
        entry_labels: pd.Series,
        regime_assignments: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Generate PID-based features for entry timing optimization."""
        tprint_info("🔧 Generating PID-based features for Tactician...")

        features = pd.DataFrame(index=data.index)

        # Price-based PID features
        price_pid_features = self._generate_price_pid_features(data)
        features = pd.concat([features, price_pid_features], axis=1)

        # Volume-based PID features
        volume_pid_features = self._generate_volume_pid_features(data)
        features = pd.concat([features, volume_pid_features], axis=1)

        # Volatility-based PID features
        volatility_pid_features = self._generate_volatility_pid_features(data)
        features = pd.concat([features, volatility_pid_features], axis=1)

        # Entry timing specific features
        entry_timing_features = self._generate_entry_timing_features(data, entry_labels)
        features = pd.concat([features, entry_timing_features], axis=1)

        # Regime-adaptive features
        if regime_assignments is not None:
            regime_features = self._generate_regime_adaptive_features(data, regime_assignments)
            features = pd.concat([features, regime_features], axis=1)

        tprint_success(f"✅ Generated {len(features.columns)} PID features")
        return features

    def _generate_price_pid_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate price-based PID features."""
        features = pd.DataFrame(index=data.index)

        # Price error (difference from moving average)
        for window in [5, 10, 20, 50]:
            ma = data['close'].rolling(window).mean()
            error = data['close'] - ma
            features[f'price_error_{window}'] = error

            # Proportional term
            features[f'price_p_{window}'] = error

            # Integral term (cumulative error)
            features[f'price_i_{window}'] = error.rolling(window).sum()

            # Derivative term (error rate of change)
            features[f'price_d_{window}'] = error.diff()

        # Price momentum PID
        for window in [3, 6, 12]:
            momentum = data['close'].pct_change(window)
            features[f'momentum_p_{window}'] = momentum
            features[f'momentum_i_{window}'] = momentum.rolling(window).sum()
            features[f'momentum_d_{window}'] = momentum.diff()

        return features

    def _generate_volume_pid_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volume-based PID features."""
        features = pd.DataFrame(index=data.index)

        # Volume error (difference from average volume)
        for window in [5, 10, 20]:
            avg_volume = data['volume'].rolling(window).mean()
            volume_error = data['volume'] - avg_volume
            features[f'volume_error_{window}'] = volume_error

            # Volume PID terms
            features[f'volume_p_{window}'] = volume_error
            features[f'volume_i_{window}'] = volume_error.rolling(window).sum()
            features[f'volume_d_{window}'] = volume_error.diff()

        # Volume-price relationship PID
        vwap = (data['volume'] * data['close']).rolling(20).sum() / data['volume'].rolling(20).sum()
        vwap_error = data['close'] - vwap
        features['vwap_error'] = vwap_error
        features['vwap_p'] = vwap_error
        features['vwap_i'] = vwap_error.rolling(20).sum()
        features['vwap_d'] = vwap_error.diff()

        return features

    def _generate_volatility_pid_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility-based PID features."""
        features = pd.DataFrame(index=data.index)

        # Calculate rolling volatility
        for window in [5, 10, 20]:
            returns = data['close'].pct_change()
            volatility = returns.rolling(window).std()

            # Volatility error (difference from target volatility)
            target_vol = volatility.rolling(window * 2).mean()
            vol_error = volatility - target_vol
            features[f'vol_error_{window}'] = vol_error

            # Volatility PID terms
            features[f'vol_p_{window}'] = vol_error
            features[f'vol_i_{window}'] = vol_error.rolling(window).sum()
            features[f'vol_d_{window}'] = vol_error.diff()

        return features

    def _generate_entry_timing_features(
        self,
        data: pd.DataFrame,
        entry_labels: pd.Series
    ) -> pd.DataFrame:
        """Generate entry timing specific features."""
        features = pd.DataFrame(index=data.index)

        # Time since last entry signal
        last_entry = entry_labels[entry_labels > 0].index
        if len(last_entry) > 0:
            time_since_entry = pd.Series(index=data.index, dtype=float)
            for i, idx in enumerate(data.index):
                prev_entries = last_entry[last_entry <= idx]
                if len(prev_entries) > 0:
                    time_since_entry.iloc[i] = (idx - prev_entries[-1]).total_seconds() / 60  # minutes
                else:
                    time_since_entry.iloc[i] = np.nan
            features['time_since_last_entry'] = time_since_entry

        # Entry signal strength
        features['entry_signal_strength'] = entry_labels

        # Distance to next potential entry
        next_entry = entry_labels[entry_labels > 0].index
        if len(next_entry) > 0:
            distance_to_entry = pd.Series(index=data.index, dtype=float)
            for i, idx in enumerate(data.index):
                future_entries = next_entry[next_entry > idx]
                if len(future_entries) > 0:
                    distance_to_entry.iloc[i] = (future_entries[0] - idx).total_seconds() / 60  # minutes
                else:
                    distance_to_entry.iloc[i] = np.nan
            features['distance_to_next_entry'] = distance_to_entry

        return features

    def _generate_regime_adaptive_features(
        self,
        data: pd.DataFrame,
        regime_assignments: pd.Series
    ) -> pd.DataFrame:
        """Generate regime-adaptive features."""
        features = pd.DataFrame(index=data.index)

        # Regime-specific price behavior
        for regime in regime_assignments.unique():
            regime_mask = regime_assignments == regime
            regime_data = data[regime_mask]

            if len(regime_data) > 10:  # Minimum data for meaningful features
                # Regime-specific volatility
                regime_vol = regime_data['close'].pct_change().std()
                features[f'regime_{regime}_volatility'] = regime_vol

                # Regime-specific momentum
                regime_momentum = regime_data['close'].pct_change(5).mean()
                features[f'regime_{regime}_momentum'] = regime_momentum

        return features

class EnhancedTacticianPreMLOrchestrator:
    """
    Enhanced Tactician Pre-ML Orchestration.

    Orchestrates the complete pre-training pipeline for Tactician models with
    differentiated labeling focused on optimal entry timing.
    """

    def __init__(self, config: Optional[EnhancedTacticianPreMLConfig] = None):
        """Initialize the enhanced Tactician pre-ML orchestrator."""
        try:
            self.config = config or EnhancedTacticianPreMLConfig()
            self.logger = system_logger.getChild('EnhancedTacticianPreMLOrchestrator')

            # Initialize components
            self.labeler = TacticianDifferentiatedLabeler(self.config.labeling_config)
            self.pid_generator = TacticianPIDFeatureGenerator(self.config.custom_params)

            # Initialize corrected ML-based labeling if enabled
            if self.config.enable_ml_labeling and ML_LABELING_AVAILABLE:
                ml_config = self.config.ml_labeling_config or CorrectedMLEntryTimingConfig()
                self.ml_labeler = CorrectedMLEntryTimingLabeler(ml_config)
                tprint_success("✅ Corrected ML-based labeling initialized")
            else:
                self.ml_labeler = None
                if self.config.enable_ml_labeling:
                    tprint_warning("⚠️ Corrected ML-based labeling requested but not available")

            # Initialize pre-training pipeline
            if PRE_TRAINING_AVAILABLE:
                self.pre_training_pipeline = PreTrainingSubPipeline()
                tprint_success("✅ Pre-training pipeline initialized for Enhanced Tactician")
            else:
                self.pre_training_pipeline = None
                tprint_error("❌ Pre-training pipeline not available")

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

            # Initialize Unified Vectorization Manager
            if UNIFIED_VECTORIZATION_AVAILABLE:
                self.vectorization_manager = create_vectorbt_unified_framework()
                tprint_success("✅ Unified Vectorization Manager initialized")
            else:
                self.vectorization_manager = None
                tprint_warning("⚠️ Unified Vectorization Manager not available")

            tprint_success(f"✅ EnhancedTacticianPreMLOrchestrator initialized (timeframe: {self.config.timeframe})")
            tprint_info(f"🎯 Analyst signal threshold: {self.config.analyst_confidence_threshold:.2%}")
            tprint_info(f"🎯 Entry window: {self.config.labeling_config.min_entry_window_minutes}-{self.config.labeling_config.max_entry_window_minutes} minutes")

        except Exception as e:
            tprint_error(f"❌ Failed to initialize EnhancedTacticianPreMLOrchestrator: {e}")
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
    ) -> Tuple[Optional[str], str]:
        error_code, message = self._extract_failure_details(step_result)
        code_text = f"[{error_code}] " if error_code else ''
        composed = f"{prefix}: {code_text}{message}"
        tprint_error(f"❌ {composed}")
        self.logger.error(f"❌ {composed}")
        return error_code, message

    def _integrate_analyst_signals(
        self,
        training_data: pd.DataFrame,
        analyst_predictions: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Integrate Analyst signals for Tactician training.

        Args:
            training_data: Input DataFrame (15m timeframe)
            analyst_predictions: Analyst ensemble predictions

        Returns:
            Tuple of (filtered_data, analyst_signals, integration_metrics)
        """
        tprint_info("🔗 Integrating Analyst signals for Tactician training...")

        if analyst_predictions is None:
            tprint_warning("⚠️ No analyst predictions provided, using all data")
            return training_data, pd.Series(0, index=training_data.index), {}

        # Extract analyst confidence scores
        if 'confidence' in analyst_predictions.columns:
            analyst_signals = analyst_predictions['confidence']
        elif 'prediction' in analyst_predictions.columns:
            analyst_signals = analyst_predictions['prediction']
        else:
            tprint_warning("⚠️ No confidence or prediction column found in analyst predictions")
            return training_data, pd.Series(0, index=training_data.index), {}

        # Align analyst signals with training data
        common_index = training_data.index.intersection(analyst_signals.index)
        if len(common_index) == 0:
            tprint_warning("⚠️ No common index between training data and analyst predictions")
            return training_data, pd.Series(0, index=training_data.index), {}

        aligned_signals = analyst_signals.reindex(training_data.index, fill_value=0)

        # Create binary signals based on threshold
        binary_signals = (aligned_signals >= self.config.analyst_confidence_threshold).astype(int)

        # Calculate integration metrics
        total_samples = len(training_data)
        green_samples = binary_signals.sum()
        coverage = green_samples / total_samples if total_samples > 0 else 0

        integration_metrics = {
            'total_samples': total_samples,
            'green_samples': green_samples,
            'coverage': coverage,
            'threshold_used': self.config.analyst_confidence_threshold
        }

        tprint_success(f"✅ Analyst signal integration completed")
        tprint_info(f"📊 Green signal coverage: {coverage:.2%} ({green_samples}/{total_samples})")

        return training_data, binary_signals, integration_metrics

    async def orchestrate(
        self,
        training_data: pd.DataFrame,
        analyst_predictions: Optional[pd.DataFrame] = None,
        regime_assignments: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> EnhancedTacticianPreMLResult:
        """
        Execute the enhanced pre-ML orchestration for Tactician models.

        Args:
            training_data: Input DataFrame with market data (15m timeframe)
            analyst_predictions: Analyst ensemble predictions for signal integration
            regime_assignments: Optional regime assignments for per-regime optimization
            **kwargs: Additional parameters

        Returns:
            EnhancedTacticianPreMLResult with orchestrated features and metadata
        """
        start_time = tprint_timer()
        tprint_info("🚀 Starting Enhanced Tactician Pre-ML Orchestration (15m timeframe)...")
        tprint_info(f"📊 Input data shape: {training_data.shape}")

        result = EnhancedTacticianPreMLResult()
        result.total_samples_before_filter = len(training_data)

        try:
            # Validate pre-training pipeline availability
            if not self.pre_training_pipeline:
                raise RuntimeError("Pre-training pipeline not available")

            # Step 0: Integrate Analyst signals
            tprint_info("🔗 Step 0/5: Integrating Analyst signals...")
            result.phase = OrchestrationPhase.ANALYST_SIGNAL_INTEGRATION

            filtered_data, analyst_signals, integration_metrics = self._integrate_analyst_signals(
                training_data, analyst_predictions
            )

            result.analyst_signal_integration_result = integration_metrics
            result.total_samples_after_analyst_filter = len(filtered_data)
            result.analyst_signal_coverage = integration_metrics.get('coverage', 0)

            tprint_success(f"✅ Analyst signal integration completed")

            # Step 1: Apply differentiated horizon labeling
            tprint_info("🎯 Step 1/5: Applying differentiated horizon labeling...")
            result.phase = OrchestrationPhase.DIFFERENTIATED_LABELING

            regime_series = None
            if regime_assignments is not None and 'regime' in regime_assignments.columns:
                regime_series = regime_assignments['regime']

            # First, create initial rule-based labels
            initial_labels, initial_metrics = self.labeler.create_entry_timing_labels(
                filtered_data, analyst_signals, regime_series
            )

            # Then, apply corrected ML-based labeling if enabled
            if self.ml_labeler is not None:
                tprint_info("🤖 Applying corrected ML-based labeling (peak/bottom detection)...")
                entry_labels, ml_metrics = self.ml_labeler.create_corrected_ml_labels(
                    filtered_data, analyst_signals, regime_series
                )

                # Combine metrics
                labeling_metrics = {
                    'initial_labeling': initial_metrics,
                    'ml_labeling': ml_metrics,
                    'overall_quality': ml_metrics.get('overall_quality', initial_metrics.get('overall_quality', 0))
                }
            else:
                entry_labels = initial_labels
                labeling_metrics = initial_metrics

            result.differentiated_labeling_result = labeling_metrics
            result.entry_timing_labels = entry_labels
            result.labeling_quality_metrics = labeling_metrics
            result.total_samples_after_labeling = len(filtered_data)

            tprint_success(f"✅ Differentiated labeling completed")
            tprint_info(f"📊 Labeling quality: {labeling_metrics.get('overall_quality', 0):.3f}")

            # Step 2: Feature Lookback Optimization (global, not per-regime)
            tprint_info("⚙️ Step 2/5: Feature Lookback Optimization (global)...")
            result.phase = OrchestrationPhase.LOOKBACK_OPTIMIZATION

            # Create sub-pipeline configuration for lookback optimization
            sub_config = SubPipelineConfig(
                symbol=self.config.symbol,
                exchange=self.config.exchange,
                timeframe=self.config.timeframe,
                data_dir=self.config.data_dir,
                parallel_processing=self.config.enable_parallel_processing,
                custom_params={
                    **self.config.custom_params,
                    'enable_per_regime_optimization': False,  # Tactician is NOT per-regime
                    'enable_per_cluster_optimization': False,  # Tactician is NOT per-cluster
                    'regime_assignments': None,  # Not used for Tactician
                    'role': 'tactician',
                    'prepared_data': filtered_data,
                    'entry_labels': entry_labels,
                    **kwargs
                }
            )

            lookback_result = await self.pre_training_pipeline._execute_feature_lookback_optimization(sub_config)

            if not lookback_result.success:
                error_code, message = self._log_subpipeline_failure(
                    "Lookback optimization failed",
                    lookback_result,
                )
                raise RuntimeError(f"Lookback optimization failed ({error_code or 'unknown_error'}): {message}")

            result.lookback_optimization_result = lookback_result.artifacts
            tprint_success("✅ Lookback optimization completed")

            # Step 3: Enhanced Interactive Feature Generation
            tprint_info("🔧 Step 3/5: Enhanced Interactive Feature Generation...")
            result.phase = OrchestrationPhase.INTERACTIVE_FEATURE_GENERATION

            # Generate PID-style features using our custom generator for backward compatibility
            pid_features = self.pid_generator.generate_pid_features(
                filtered_data, entry_labels, regime_series
            )

            # Also run the interactive feature generation pipeline for additional features
            interactive_result = await self.pre_training_pipeline._execute_interactive_feature_generation(sub_config)

            if not interactive_result.success:
                error_code, message = self._log_subpipeline_failure(
                    "Interactive feature generation failed",
                    interactive_result,
                )
                raise RuntimeError(
                    f"Interactive feature generation failed ({error_code or 'unknown_error'}): {message}"
                )

            interactive_artifacts = interactive_result.artifacts.get('interactive_feature_generation_result', {})
            standard_interactive_features = interactive_artifacts.get('features')

            if isinstance(standard_interactive_features, pd.DataFrame):
                combined_features = pd.concat([pid_features, standard_interactive_features], axis=1)
            else:
                combined_features = pid_features.copy()

            result.interactive_feature_generation_result = {
                **interactive_result.artifacts,
                'custom_interactive_features': pid_features,
                'combined_features': combined_features
            }

            feature_names = interactive_artifacts.get('feature_names', [])
            if feature_names:
                generated_count = len(feature_names)
            elif isinstance(standard_interactive_features, pd.DataFrame):
                generated_count = len(standard_interactive_features.columns)
            else:
                generated_count = int(interactive_artifacts.get('total_features', 0))

            result.total_features_generated = len(combined_features.columns)
            tprint_success(
                "✅ Enhanced interactive feature generation completed "
                f"({result.total_features_generated} combined features; {generated_count} interactive)"
            )

            # Step 4: Final Feature Selection (global, not per-regime)
            tprint_info("🎯 Step 4/5: Final Feature Selection (global)...")
            result.phase = OrchestrationPhase.FEATURE_SELECTION

            # Update sub-config with combined features
            sub_config.custom_params['combined_features'] = combined_features
            sub_config.custom_params['entry_labels'] = entry_labels
            sub_config.custom_params['enable_per_regime_optimization'] = False  # Tactician is NOT per-regime
            sub_config.custom_params['enable_per_cluster_optimization'] = False  # Tactician is NOT per-cluster

            selection_result = await self.pre_training_pipeline._execute_final_feature_selection(sub_config)

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

            # Mark as completed
            result.success = True
            result.phase = OrchestrationPhase.COMPLETED
            result.execution_time = tprint_timer(start_time)

            tprint_success(f"✅ Enhanced Tactician Pre-ML Orchestration completed in {result.execution_time:.2f}s")
            tprint_info(f"📊 Final feature count: {result.final_feature_count}")
            tprint_info(f"📊 Analyst signal coverage: {result.analyst_signal_coverage:.2%}")
            tprint_info(f"📊 Labeling quality: {result.labeling_quality_metrics.get('overall_quality', 0):.3f}")

            return result

        except Exception as e:
            result.success = False
            result.phase = OrchestrationPhase.FAILED
            result.error_message = str(e)
            result.execution_time = tprint_timer(start_time)

            tprint_error(f"❌ Enhanced Tactician Pre-ML Orchestration failed: {e}")
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
                'output_directory': self.config.output_directory,
                'labeling_config': {
                    'min_entry_window_minutes': self.config.labeling_config.min_entry_window_minutes,
                    'max_entry_window_minutes': self.config.labeling_config.max_entry_window_minutes,
                    'entry_quality_threshold': self.config.labeling_config.entry_quality_threshold,
                    'enable_regime_adaptive_labeling': self.config.labeling_config.enable_regime_adaptive_labeling
                }
            },
            'component_availability': {
                'pre_training_pipeline': self.pre_training_pipeline is not None,
                'differentiated_labeler': self.labeler is not None,
                'pid_generator': self.pid_generator is not None
            }
        }

# Convenience function for external usage
async def execute_enhanced_tactician_pre_ml_orchestration(
    training_data: pd.DataFrame,
    analyst_predictions: Optional[pd.DataFrame] = None,
    regime_assignments: Optional[pd.DataFrame] = None,
    config: Optional[EnhancedTacticianPreMLConfig] = None,
    **kwargs
) -> EnhancedTacticianPreMLResult:
    """
    Execute enhanced Tactician pre-ML orchestration.

    Args:
        training_data: Input DataFrame with market data (15m timeframe)
        analyst_predictions: Analyst ensemble predictions for signal integration
        regime_assignments: Optional regime assignments for per-regime optimization
        config: Optional configuration
        **kwargs: Additional parameters

    Returns:
        EnhancedTacticianPreMLResult with orchestrated features and metadata
    """
    orchestrator = EnhancedTacticianPreMLOrchestrator(config)
    return await orchestrator.orchestrate(training_data, analyst_predictions, regime_assignments, **kwargs)

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
