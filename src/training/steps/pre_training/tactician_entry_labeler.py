"""
import warnings
Tactician Entry Labeler - Differentiated Entry Timing Labels for Tactician Models

This module provides entry timing label generation for Tactician models,
using enhanced entry quality scoring with regime adaptation.

Key Features:
- 15m timeframe optimization for entry timing
- Local maxima/minima detection with peak filtering
- Enhanced entry quality scoring (adaptive multi-factor)
- Regime-aware labeling with adaptive thresholds
- Trains on ALL market data (not just Analyst green lights)
"""

import time
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from scipy.signal import find_peaks

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.logger import system_logger
from src.utils.common_operations import (
    validate_dataframe_columns,
    safe_dataframe_operation,
    validate_positive,
    validate_range,
    safe_int,
    safe_float,
    get_dataframe_info,
    create_data_quality_report,
    ensure_directory,
    safe_json_dump,
    safe_json_load,
    format_bytes,
    timed_operation,
    memory_checkpoint,
    optimize_memory,
    check_disk_space,
    safe_divide,
    safe_mean,
    safe_std,
    integrate_with_m1_optimizers,
    get_m1_gpu_manager,
    get_m1_memory_optimizer
)
from src.utils.common_utilities import (
    analyze_nan_values_detailed,
    format_nan_analysis_report,
    create_data_quality_report as create_detailed_quality_report,
    get_dataframe_info as get_detailed_dataframe_info
)
from src.utils.matrix_operations import (
    get_unified_matrix_operations,
    get_vectorized_processing_core,
    get_enhanced_matrix_operations,
    optimize_dataframe,
    vectorized_rolling_features,
    matrix_correlation_analysis,
    safe_correlation_matrix,
    compute_trading_indicators,
    get_hardware_performance_report
)

# Import VectorBT optimizer for enhanced performance
# from .profit_labeling.vectorbt_optimizer import (
#     get_vectorbt_optimizer, VectorBTConfig, optimized_rolling_mean,
#     optimized_rolling_std, optimized_volatility, optimized_returns
# )
from src.utils.ml_common.optimization.grid_utils import (
    generate_grid,
    build_coarse_grid_from_search_space,
    GridSearchOptimizer
)
from src.training.steps.pre_training.components.base_component import BasePreTrainingComponent, ComponentResult
from src.training.steps.pre_training.components.component_factory import ComponentConfig
from src.training.steps.pre_training.components.contracts import PipelineState
from src.training.steps.pre_training.components import ComponentFactory
from src.training.steps.pre_training.validation.schemas import validate_raw_ohlcv, SchemaValidationException

@dataclass
class TacticianLabelingConfig:
    """Configuration for Tactician-specific differentiated labeling."""

    # Entry timing optimization
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
    
    # VectorBT configuration
    vectorbt_config: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'threshold': 1000,
        'optimization_level': 'balanced'
    })

    # Trading direction settings
    enable_long_positions: bool = True   # Include long opportunities (buy when expecting price increase)
    enable_short_positions: bool = False  # Include short opportunities (sell when expecting price decrease)

    # VectorBT optimization settings
    # vectorbt_config: Optional[VectorBTConfig] = None

class TacticianDifferentiatedLabeler:
    """Create differentiated entry timing labels for the Tactician pipeline."""

    def __init__(self, config: TacticianLabelingConfig):
        self.config = config
        self.logger = system_logger.getChild('TacticianDifferentiatedLabeler')

        # Initialize matrix operations for enhanced data processing
        self.matrix_ops = get_unified_matrix_operations()
        self.vectorized_core = get_vectorized_processing_core()
        self.enhanced_matrix_ops = get_enhanced_matrix_operations()

        tprint_info(f"🧮 Matrix operations initialized: {self.matrix_ops.__class__.__name__}")

        # Initialize VectorBT optimizer for enhanced performance
        self._initialize_vectorbt_optimizer()

        # Initialize M1 optimizations if available
        self.m1_integration = integrate_with_m1_optimizers()
        if self.m1_integration.get('success', False):
            tprint_info(f"🧠 M1 optimizations initialized: GPU={'✅' if self.m1_integration.get('gpu_manager') else '❌'}, Memory={'✅' if self.m1_integration.get('memory_optimizer') else '❌'}")

        # Initialize enhanced quality scorer
        self._initialize_quality_scorer()

    def _initialize_quality_scorer(self):
        """Initialize the enhanced entry quality scorer based on configuration."""
        tprint_info("🎯 Initializing enhanced entry quality scorer")
        try:
            from src.training.steps.models_training.enhanced_entry_quality_scorer import (
                create_enhanced_scorer,
                ScoringMethod,
                EnhancedScoringConfig
            )
            tprint_info("✅ Enhanced entry quality scorer module imported successfully")

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
            tprint_info(f"📊 Using scoring method: {method.value}")

            # Create scorer configuration (converting percent to decimal)
            tprint_info("⚙️ Creating scorer configuration with regime adaptation and interaction terms")
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
            tprint_info(f"   → Regime adaptation: {self.config.enable_regime_adaptive_labeling}")
            tprint_info(f"   → Interaction terms: {self.config.enable_interaction_terms}")
            tprint_info(f"   → Penalty system: {self.config.enable_penalty_system}")
            tprint_info(f"   → Risk aversion: {self.config.risk_aversion}")

        except (ImportError, AttributeError, Exception) as e:
            tprint_warning(f"⚠️ Enhanced quality scorer not available, using fallback: {e}")
            self.quality_scorer = None

    def _initialize_vectorbt_optimizer(self):
        """Initialize VectorBT optimizer if available."""
        try:
            # Try to import VectorBT optimizer
            from .profit_labeling.vectorbt_optimizer import get_vectorbt_optimizer, VectorBTConfig
            tprint_info("⚡ Initializing VectorBT optimizer")
            
            vectorbt_config = VectorBTConfig(
                enable_vectorbt=True,
                vectorbt_threshold=self.config.vectorbt_config.get('threshold', 1000),
                performance_monitoring=True,
                memory_efficiency_mode=True
            )
            self.vectorbt_optimizer = get_vectorbt_optimizer(vectorbt_config)
            tprint_success(f"✅ VectorBT optimizer initialized: {self.vectorbt_optimizer.__class__.__name__}")
        except (ImportError, AttributeError, Exception) as e:
            tprint_warning(f"⚠️ VectorBT optimizer not available, using fallback methods: {e}")
            self.vectorbt_optimizer = None

    def _safe_vectorbt_operation(self, operation_name: str, operation_func, *args, **kwargs):
        """Safely execute VectorBT operations with fallback."""
        if self.vectorbt_optimizer is not None:
            try:
                return operation_func(*args, **kwargs)
            except Exception as e:
                tprint_warning(f"VectorBT {operation_name} failed: {e}, using fallback")
                return None
        else:
            tprint_info(f"VectorBT not available for {operation_name}, using fallback")
            return None

    def _calculate_peak_distance(self, current_idx, existing_idx):
        """Calculate time distance between two indices in minutes."""
        try:
            # Handle different index types (datetime, timestamp, etc.)
            if hasattr(current_idx, 'to_pydatetime'):
                current_time = current_idx.to_pydatetime()
            elif hasattr(current_idx, 'timestamp'):
                current_time = pd.Timestamp(current_idx).to_pydatetime()
            else:
                current_time = pd.Timestamp(current_idx).to_pydatetime()
                
            if hasattr(existing_idx, 'to_pydatetime'):
                existing_time = existing_idx.to_pydatetime()
            elif hasattr(existing_idx, 'timestamp'):
                existing_time = pd.Timestamp(existing_idx).to_pydatetime()
            else:
                existing_time = pd.Timestamp(existing_idx).to_pydatetime()
                
            return abs((current_time - existing_time).total_seconds() / 60)
        except Exception as e:
            tprint_warning(f"Error calculating peak distance: {e}")
            return float('inf')

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

        # Validate input data format and constraints
        try:
            data = validate_raw_ohlcv(data)
            tprint_info(f"✅ Input data validated: {len(data)} rows, {len(data.columns)} columns")
        except SchemaValidationException as e:
            tprint_error(f"❌ Input data validation failed: {e}")
            raise ValueError(f"Invalid input data format: {e}") from e

        # Validate input data quality using common operations and utilities
        data_quality = create_data_quality_report(data)
        detailed_quality = analyze_nan_values_detailed(data)

        if data_quality.get('quality_metrics', {}).get('missing_percentage', 0) > 50:
            tprint_warning(f"⚠️ High missing data percentage: {data_quality['quality_metrics']['missing_percentage']:.2f}%")

        # Log detailed NaN analysis if issues found
        if detailed_quality.get('total_nans', 0) > 0:
            nan_report = format_nan_analysis_report(detailed_quality, "  ")
            tprint_info(f"📊 NaN Analysis:\n{nan_report}")

        # Optimize data using matrix operations for better performance
        tprint_info(f"🧮 Optimizing data with matrix operations ({data.shape})")
        original_shape = data.shape
        optimized_data = optimize_dataframe(data)

        if optimized_data is not data:
            data = optimized_data
            tprint_success(f"✅ Data optimized: {original_shape} → {data.shape}")

        # Validate required columns for OHLCV data
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not validate_dataframe_columns(data, required_columns):
            missing_cols = set(required_columns) - set(data.columns)
            raise ValueError(f"Missing required OHLCV columns: {missing_cols}")

        # Validate volume data (should not be all zero or negative)
        if 'volume' in data.columns:
            zero_volume_count = safe_int((data['volume'] <= 0).sum())
            if zero_volume_count > 0:
                tprint_warning(f"⚠️ Found {zero_volume_count} rows with zero or negative volume")
            if zero_volume_count == len(data):
                tprint_error("❌ All volume values are zero or negative - cannot create entry labels")
                raise ValueError("Invalid volume data: all values are zero or negative")

        # Validate index monotonicity (timestamps should be sorted)
        if not data.index.is_monotonic_increasing:
            tprint_error("❌ Data index is not sorted by timestamp")
            raise ValueError("Data index must be sorted by timestamp for time-series operations")

        # Validate window size against data length
        window_size = self.config.max_entry_window_minutes
        if len(data) <= window_size:
            tprint_error(f"❌ Data length ({len(data)}) is too short for window size ({window_size})")
            raise ValueError(f"Insufficient data: need at least {window_size + 1} rows for window size {window_size}")

        if regime_assignments is not None:
            regime_assignments = regime_assignments.reindex(data.index)

        labels = pd.Series(0.0, index=data.index, dtype=float)

        # CHANGE: Process ALL data, not just Analyst green light periods
        # Create sliding windows across entire dataset
        tprint_info(f"📊 Processing {len(data)} candles for entry opportunities")

        # Vectorized approach to avoid O(n²) nested loops with DataFrame operations
        window_size = self.config.max_entry_window_minutes

        # Pre-allocate arrays for better performance
        entry_indices = data.index[:-window_size]  # All potential entry points
        future_window_starts = np.arange(1, len(data) - window_size + 1)  # Start indices for future windows
        future_window_ends = future_window_starts + window_size  # End indices for future windows

        # Vectorized quality score calculation with VectorBT optimization
        scores = np.zeros(len(entry_indices))

        # Use VectorBT for optimized rolling operations if data is large enough
        vectorbt_threshold = self.config.vectorbt_config.get('threshold', 1000)
        if self.vectorbt_optimizer is not None and len(data) >= vectorbt_threshold:
            tprint_info(f"⚡ Using VectorBT optimization for {len(data)} samples")
            scores = self._calculate_vectorized_quality_scores(
                data, entry_indices, future_window_starts, future_window_ends,
                regime_assignments, window_size
            )
        else:
            # Use standard approach for smaller datasets
            for i, (entry_idx, start_idx, end_idx) in enumerate(zip(
                range(len(entry_indices)),
                future_window_starts,
                future_window_ends
            )):
                entry_index = entry_indices[i]
                future_window = data.iloc[start_idx:end_idx]

                if not future_window.empty:
                    # Calculate entry quality score
                    score = self._calculate_entry_quality_score(
                        data.iloc[entry_idx],
                        future_window,
                        entry_index,
                        regime_assignments
                    )
                    scores[i] = score

        # Apply threshold and store results
        valid_entries = scores > self.config.entry_quality_threshold
        labels.loc[entry_indices[valid_entries]] = scores[valid_entries]

        entry_points = entry_indices[valid_entries].tolist()

        # Apply peak detection to identify local maxima
        if len(entry_points) > 0:
            labels = self._apply_peak_filtering(labels)
            entry_points = labels.index[labels > 0].tolist()

        quality_metrics = self._calculate_labeling_quality_metrics_all_data(
            data,
            labels,
            entry_points
        )

        # Log memory usage and data quality
        memory_info = optimize_memory()
        data_info = get_dataframe_info(data)
        tprint_info(f"📊 Data info: {data_info['shape']} shape, {format_bytes(data_info['memory_usage'])} memory")

        tprint_success(
            "✅ Entry labeling completed on ALL data ("
            f"{int((labels > 0).sum())} optimal entries, quality={quality_metrics.get('overall_quality', 0):.3f})"
        )

        return labels, quality_metrics

    def _apply_peak_filtering(self, labels: pd.Series) -> pd.Series:
        """
        Apply peak detection to filter entry labels to local maxima.
        This prevents too many entries by selecting only the best quality peaks.
        Uses VectorBT optimization for large datasets.
        """
        tprint_info("🔍 Applying peak filtering to entry labels")
        # Get non-zero labels
        non_zero_mask = labels > 0
        non_zero_count = int(non_zero_mask.sum())
        tprint_info(f"📊 Found {non_zero_count} non-zero labels to filter")

        if non_zero_count == 0:
            tprint_warning("⚠️ No non-zero labels found, returning original labels")
            return labels

        # Extract scores
        scores = labels[non_zero_mask].values
        indices = labels[non_zero_mask].index
        tprint_info(f"📈 Score range: {scores.min():.4f} - {scores.max():.4f}")

        # Use VectorBT for optimized peak detection on large datasets
        vectorbt_threshold = self.config.vectorbt_config.get('threshold', 1000)
        if self.vectorbt_optimizer is not None and len(scores) >= vectorbt_threshold:
            tprint_info(f"⚡ Using VectorBT optimization for peak detection on {len(scores)} scores")
            filtered_labels = self._vectorbt_peak_filtering(labels, scores, indices)
        else:
            # Use standard peak detection for smaller datasets
            tprint_info(f"🔧 Using standard peak detection for {len(scores)} scores")
            filtered_labels = self._standard_peak_filtering(labels, scores, indices)

        # Validate that we have usable training data
        final_entry_count = int((filtered_labels > 0).sum())
        tprint_info(f"🎯 Peak filtering completed: {non_zero_count} → {final_entry_count} entries")

        if final_entry_count == 0:
            error_msg = (
                "Peak filtering resulted in no usable entry labels for training. "
                f"Original entries: {len(scores)}, Peak threshold: {self.config.entry_quality_threshold}, "
                f"Min window: {self.config.min_entry_window_minutes} minutes. "
                "Consider lowering the entry quality threshold or minimum window requirements."
            )
            tprint_error(f"❌ {error_msg}")
            raise ValueError(error_msg)

        # Warn if we have very few entries (might indicate overly strict filtering)
        if final_entry_count < 10:
            warning_msg = (
                f"Peak filtering resulted in very few entry labels ({final_entry_count}). "
                "Training data may be insufficient for reliable model training. "
                "Consider adjusting entry quality threshold or minimum window requirements."
            )
            tprint_warning(f"⚠️ {warning_msg}")
            warnings.warn(warning_msg, UserWarning, stacklevel=2)
        else:
            tprint_success(f"✅ Peak filtering successful: {final_entry_count} high-quality entries selected")

        return filtered_labels

    def _standard_peak_filtering(self, labels: pd.Series, scores: np.ndarray, indices: pd.Index) -> pd.Series:
        """Standard peak detection using scipy."""
        tprint_info("🔍 Applying standard peak detection with scipy")
        # Apply peak detection
        peaks, properties = find_peaks(
            scores,
            height=self.config.entry_quality_threshold,
            distance=max(1, self.config.min_entry_window_minutes)
        )
        tprint_info(f"📊 Found {len(peaks)} peaks with height > {self.config.entry_quality_threshold}")

        # Create filtered labels
        filtered_labels = pd.Series(0.0, index=labels.index, dtype=float)

        if len(peaks) > 0:
            peak_indices = [indices[p] for p in peaks if p < len(indices)]
            peak_scores = [scores[p] for p in peaks if p < len(scores)]
            tprint_info(f"✅ Selected {len(peak_indices)} peaks for filtering")

            for idx, score in zip(peak_indices, peak_scores):
                filtered_labels.loc[idx] = score
        else:
            tprint_warning("⚠️ No peaks found with current threshold")

        # If no peaks found but we have high-quality entries, keep the best
        if filtered_labels.sum() == 0 and len(scores) > 0:
            # Find the best entry that meets the quality threshold
            valid_scores = scores[scores > self.config.entry_quality_threshold]
            if len(valid_scores) > 0:
                best_idx = np.argmax(scores)
                if best_idx < len(indices):
                    filtered_labels.loc[indices[best_idx]] = scores[best_idx]
                    tprint_info(f"🔄 Fallback: selected best entry with score {scores[best_idx]:.4f}")
            else:
                # If no entries meet threshold, lower it temporarily
                best_idx = np.argmax(scores)
                if best_idx < len(indices) and scores[best_idx] > 0:
                    filtered_labels.loc[indices[best_idx]] = scores[best_idx]
                    tprint_warning(f"⚠️ No entries met quality threshold, selected best available: {scores[best_idx]:.4f}")

        final_count = int((filtered_labels > 0).sum())
        tprint_info(f"🎯 Standard peak filtering result: {final_count} entries selected")
        return filtered_labels

    def _vectorbt_peak_filtering(self, labels: pd.Series, scores: np.ndarray, indices: pd.Index) -> pd.Series:
        """VectorBT optimized peak detection for large datasets."""
        tprint_info("⚡ Applying VectorBT optimized peak filtering")

        # Create a temporary series for VectorBT operations
        temp_series = pd.Series(scores, index=indices)
        tprint_info(f"📊 Processing {len(temp_series)} scores with VectorBT")

        # Use VectorBT rolling operations to identify local maxima
        # Calculate rolling max to identify peaks
        window_size = self.config.min_entry_window_minutes * 2 + 1
        tprint_info(f"🔍 Using rolling window size: {window_size}")
        rolling_max = self._safe_vectorbt_operation(
            'rolling_max',
            self.vectorbt_optimizer.rolling_max,
            temp_series, window=window_size
        )

        # Identify peaks where current value equals rolling max
        if rolling_max is not None:
            peak_mask = (temp_series == rolling_max) & (temp_series > self.config.entry_quality_threshold)
        else:
            # Fallback: use simple threshold-based peak detection
            peak_mask = temp_series > self.config.entry_quality_threshold
            tprint_info("🔄 Using fallback peak detection (threshold-based)")
        tprint_info(f"📈 Found {int(peak_mask.sum())} potential peaks")

        # Apply additional filtering to ensure minimum distance between peaks
        filtered_peaks = []
        filtered_scores = []
        
        if peak_mask.sum() > 0:
            peak_indices = temp_series[peak_mask].index
            peak_scores = temp_series[peak_mask].values
            tprint_info(f"🔄 Applying distance filtering to {len(peak_indices)} peaks")

            # Sort by score and apply distance filtering
            sorted_indices = np.argsort(peak_scores)[::-1]  # Sort by score descending

            for idx in sorted_indices:
                current_peak_idx = peak_indices[idx]
                current_score = peak_scores[idx]

                # Check distance from already selected peaks
                if not filtered_peaks:
                    filtered_peaks.append(current_peak_idx)
                    filtered_scores.append(current_score)
                    tprint_info(f"✅ Added first peak: score {current_score:.4f}")
                else:
                    # Calculate minimum distance to existing peaks
                    distances = [self._calculate_peak_distance(current_peak_idx, existing_idx)
                               for existing_idx in filtered_peaks]
                    min_distance = min(distances) if distances else float('inf')

                    if min_distance >= self.config.min_entry_window_minutes:
                        filtered_peaks.append(current_peak_idx)
                        filtered_scores.append(current_score)
                        tprint_info(f"✅ Added peak: score {current_score:.4f}, min_distance {min_distance:.1f}min")
                    else:
                        tprint_info(f"⏭️ Skipped peak: score {current_score:.4f}, min_distance {min_distance:.1f}min")
        else:
            tprint_warning("⚠️ No peaks found after VectorBT processing")

        # Create filtered labels
        filtered_labels = pd.Series(0.0, index=labels.index, dtype=float)

        if len(filtered_peaks) > 0:
            for idx, score in zip(filtered_peaks, filtered_scores):
                filtered_labels.loc[idx] = score
            tprint_info(f"✅ Applied {len(filtered_peaks)} filtered peaks to labels")
        elif len(scores) > 0:
            # Fallback: keep the best entry if no peaks found
            best_idx = np.argmax(scores)
            if best_idx < len(indices) and scores[best_idx] > 0:
                filtered_labels.loc[indices[best_idx]] = scores[best_idx]
                tprint_warning(f"🔄 VectorBT fallback: selected best entry with score {scores[best_idx]:.4f}")
            else:
                tprint_error("❌ No valid entries found for VectorBT fallback")

        final_count = int((filtered_labels > 0).sum())
        tprint_success(f"⚡ VectorBT peak filtering completed: {final_count} peaks selected")
        return filtered_labels

    def _calculate_vectorized_quality_scores(
        self,
        data: pd.DataFrame,
        entry_indices: pd.Index,
        future_window_starts: np.ndarray,
        future_window_ends: np.ndarray,
        regime_assignments: Optional[pd.Series],
        window_size: int
    ) -> np.ndarray:
        """
        Calculate quality scores using VectorBT optimized operations for large datasets.

        This method uses VectorBT's optimized rolling operations to significantly
        improve performance for large datasets while maintaining accuracy.
        """
        tprint_info("⚡ Calculating vectorized quality scores with VectorBT optimization")
        tprint_info(f"📊 Processing {len(entry_indices)} entry points with window size {window_size}")

        # Pre-calculate rolling statistics using VectorBT for better performance
        close_prices = data['close']
        high_prices = data['high']
        low_prices = data['low']
        tprint_info("📈 Pre-calculating rolling statistics with VectorBT")

        # Calculate rolling statistics using VectorBT
        volatility_window = min(20, window_size)
        tprint_info(f"🔍 Calculating volatility with window {volatility_window}")
        rolling_volatility = self._safe_vectorbt_operation(
            'calculate_volatility',
            self.vectorbt_optimizer.calculate_volatility,
            close_prices.pct_change(), window=volatility_window, annualize=False
        )

        # Calculate rolling price statistics
        tprint_info(f"📊 Calculating rolling price statistics with window {window_size}")
        rolling_max_high = self._safe_vectorbt_operation(
            'rolling_max',
            self.vectorbt_optimizer.rolling_max,
            high_prices, window=window_size
        )
        rolling_min_low = self._safe_vectorbt_operation(
            'rolling_min',
            self.vectorbt_optimizer.rolling_min,
            low_prices, window=window_size
        )
        rolling_mean_close = self._safe_vectorbt_operation(
            'rolling_mean',
            self.vectorbt_optimizer.rolling_mean,
            close_prices, window=window_size
        )

        # Pre-allocate scores array
        scores = np.zeros(len(entry_indices))
        tprint_info(f"🔄 Starting vectorized quality score calculation for {len(entry_indices)} entries")

        # Vectorized calculation of quality scores
        processed_count = 0
        for i, (entry_idx, start_idx, end_idx) in enumerate(zip(
            range(len(entry_indices)),
            future_window_starts,
            future_window_ends
        )):
            if end_idx > len(data):
                continue

            entry_index = entry_indices[i]
            entry_price = close_prices.iloc[entry_idx]

            # Get future window data
            future_window = data.iloc[start_idx:end_idx]
            if future_window.empty:
                continue

            processed_count += 1
            if processed_count % 1000 == 0:
                tprint_info(f"📊 Processed {processed_count}/{len(entry_indices)} entries")

            # Calculate price movements using pre-computed rolling statistics
            min_future_low = future_window['low'].min()
            max_future_high = future_window['high'].max()

            # Calculate adverse and favorable movements
            adverse_move = max(entry_price - min_future_low, 0.0) / max(entry_price, 1e-8) * 100
            favorable_move = max(max_future_high - entry_price, 0.0) / max(entry_price, 1e-8) * 100

            # Get regime parameters
            regime_params = self._get_regime_parameters(entry_index, regime_assignments)

            # Apply regime-specific thresholds (consistent with fallback method)
            if adverse_move > regime_params['max_adverse_movement_pct']:
                continue
            if favorable_move < regime_params['min_favorable_movement_pct']:
                continue

            # Calculate risk-reward ratio
            risk_reward_ratio = favorable_move / (adverse_move + 1e-8)

            # Calculate timing score (prefer shorter windows)
            timing_score = 1.0 / (1.0 + len(future_window) / self.config.max_entry_window_minutes)

            # Calculate volatility score using pre-computed rolling volatility
            if rolling_volatility is not None and i < len(rolling_volatility) and not pd.isna(rolling_volatility.iloc[entry_idx]):
                volatility = rolling_volatility.iloc[entry_idx]
                volatility_score = 1.0 / (1.0 + volatility * 100 / 10.0)
            else:
                # Fallback: calculate volatility from future window
                if len(future_window) >= 2:
                    returns = future_window['close'].pct_change().dropna()
                    volatility = returns.std() if not returns.empty else 0.0
                    volatility_score = 1.0 / (1.0 + volatility * 100 / 10.0)
                else:
                    volatility_score = 1.0

            # Calculate composite quality score
            quality_score = (
                risk_reward_ratio * 0.4 +
                timing_score * 0.3 +
                volatility_score * 0.3
            )

            scores[i] = float(min(max(quality_score, 0.0), 1.0))

        valid_scores = scores[scores > 0]
        tprint_success(f"⚡ Vectorized quality scores calculated: {len(scores)} total, {len(valid_scores)} valid")
        if len(valid_scores) > 0:
            tprint_info(f"📊 Score statistics: mean={valid_scores.mean():.4f}, max={valid_scores.max():.4f}")
        return scores

    def _calculate_labeling_quality_metrics_all_data(
        self,
        data: pd.DataFrame,
        labels: pd.Series,
        entry_points: List[Any]
    ) -> Dict[str, float]:
        """
        Calculate quality metrics for labeling across all data.
        """
        tprint_info("📊 Calculating labeling quality metrics for all data")
        total_samples = len(data)
        labeled_samples = int((labels > 0).sum())
        tprint_info(f"📈 Total samples: {total_samples}, Labeled samples: {labeled_samples}")

        metrics: Dict[str, float] = {
            'labeling_coverage': labeled_samples / total_samples if total_samples else 0.0,
            'entry_density': labeled_samples / total_samples if total_samples else 0.0,
        }

        positive_scores = labels[labels > 0]
        if not positive_scores.empty:
            metrics['avg_entry_quality'] = safe_float(safe_mean(positive_scores))
            metrics['min_entry_quality'] = safe_float(positive_scores.min())
            metrics['max_entry_quality'] = safe_float(positive_scores.max())
            std_value = safe_float(safe_std(positive_scores))
            if std_value == 0.0:  # safe_std returns 0.0 for empty or error cases
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

        tprint_success(f"✅ Quality metrics calculated: overall_quality={metrics['overall_quality']:.3f}")
        tprint_info(f"   → Labeling coverage: {metrics['labeling_coverage']:.3f}")
        tprint_info(f"   → Entry density: {metrics['entry_density']:.3f}")
        tprint_info(f"   → Avg entry quality: {metrics.get('avg_entry_quality', 0.0):.3f}")

        return metrics

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
        if len(future_data) == 0:
            tprint_warning("⚠️ Empty future data for entry quality calculation")
            return 0.0

        # Use enhanced scorer if available
        if self.quality_scorer is not None:
            return self._calculate_enhanced_quality_score(entry_point, future_data, index_label, regime_assignments)
        else:
            return self._calculate_fallback_quality_score(entry_point, future_data, index_label, regime_assignments)

    def _calculate_enhanced_quality_score(
        self,
        entry_point: pd.Series,
        future_data: pd.DataFrame,
        index_label: Any,
        regime_assignments: Optional[pd.Series]
    ) -> float:
        """Calculate quality score using enhanced scorer."""
        tprint_info("🎯 Using enhanced quality scorer for entry calculation")
        
        # Determine regime
        regime = None
        if regime_assignments is not None and self.config.enable_regime_adaptive_labeling:
            if index_label in regime_assignments.index:
                regime_value = regime_assignments.loc[index_label]
                regime = f"regime_{regime_value}"
                tprint_info(f"🎭 Using regime: {regime}")

        # Build market context (can be expanded with more features)
        market_context = {}

        # Calculate quality using enhanced scorer
        quality_score = self.quality_scorer.calculate_entry_quality(
            entry_point=entry_point,
            future_data=future_data,
            regime=regime,
            market_context=market_context
        )

        tprint_info(f"📊 Enhanced quality score: {quality_score:.4f}")
        return quality_score

    def _calculate_fallback_quality_score(
        self,
        entry_point: pd.Series,
        future_data: pd.DataFrame,
        index_label: Any,
        regime_assignments: Optional[pd.Series]
    ) -> float:
        """Calculate quality score using fallback method."""
        tprint_info("🔄 Using fallback quality scoring method")
        regime_params = self._get_regime_parameters(index_label, regime_assignments)

        entry_price = entry_point['close']
        min_future_low = future_data['low'].min()
        max_future_high = future_data['high'].max()

        adverse_move = max(entry_price - min_future_low, 0.0) / max(entry_price, 1e-8) * 100
        favorable_move = max(max_future_high - entry_price, 0.0) / max(entry_price, 1e-8) * 100

        tprint_info(f"📊 Price movements: adverse={adverse_move:.2f}%, favorable={favorable_move:.2f}%")

        # Apply regime-specific thresholds
        if adverse_move > regime_params['max_adverse_movement_pct']:
            tprint_info(f"❌ Adverse movement {adverse_move:.2f}% exceeds threshold {regime_params['max_adverse_movement_pct']:.2f}%")
            return 0.0

        if favorable_move < regime_params['min_favorable_movement_pct']:
            tprint_info(f"❌ Favorable movement {favorable_move:.2f}% below threshold {regime_params['min_favorable_movement_pct']:.2f}%")
            return 0.0

        # Calculate quality components
        risk_reward_ratio = favorable_move / (adverse_move + 1e-8)
        timing_score = safe_divide(1.0, 1.0 + safe_divide(len(future_data), self.config.max_entry_window_minutes), default=0.0)

        tprint_info(f"📊 Risk-reward ratio: {risk_reward_ratio:.2f}, Timing score: {timing_score:.3f}")

        # Calculate volatility score
        volatility_score = self._calculate_volatility_score(future_data)
        tprint_info(f"📊 Volatility score: {volatility_score:.3f}")

        # Calculate composite quality score
        quality_score = (
            risk_reward_ratio * 0.4 +
            timing_score * 0.3 +
            volatility_score * 0.3
        )

        final_score = float(min(max(quality_score, 0.0), 1.0))
        tprint_info(f"📊 Fallback quality score: {final_score:.4f}")
        return final_score

    def _calculate_volatility_score(self, future_data: pd.DataFrame) -> float:
        """Calculate volatility score with VectorBT optimization and fallback."""
        if len(future_data) >= 2:
            returns = future_data['close'].pct_change().dropna()
            if not returns.empty:
                # Use VectorBT optimized volatility calculation
                volatility_result = self._safe_vectorbt_operation(
                    'calculate_volatility',
                    self.vectorbt_optimizer.calculate_volatility,
                    returns, window=len(returns), annualize=False
                )
                volatility = volatility_result.iloc[-1] if volatility_result is not None and len(volatility_result) > 0 else returns.std()
            else:
                volatility = 0.0
        else:
            volatility = 0.0

        volatility_score = safe_divide(1.0, 1.0 + safe_divide(volatility * 100, 10.0), default=1.0)
        tprint_info(f"📊 Volatility: {volatility:.4f}, Volatility score: {volatility_score:.3f}")
        return volatility_score

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
                    tprint_info(f"🎭 Using regime-specific parameters for {regime_key}")
                    return self.config.regime_specific_thresholds[regime_key]
                else:
                    tprint_info(f"🎭 No specific parameters for regime {regime_key}, using defaults")

        tprint_info("🎭 Using default regime parameters")
        return {
            'max_adverse_movement_pct': self.config.max_adverse_movement_pct,
            'min_favorable_movement_pct': self.config.min_favorable_movement_pct
        }

class TacticianEntryLabelerComponent(BasePreTrainingComponent):
    """
    Component wrapper for Tactician Entry Labeler.

    This component integrates the TacticianDifferentiatedLabeler with the pre-training pipeline
    and handles proper error handling, reporting, and pipeline state management.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Tactician entry labeler component."""
        tprint_info("🚀 Initializing TacticianEntryLabelerComponent")
        super().__init__(config)
        self.logger = system_logger.getChild('TacticianEntryLabelerComponent')

        # Create Tactician-specific configuration
        tactician_config = TacticianLabelingConfig()
        tprint_info("⚙️ Created default Tactician labeling configuration")

        # Override with custom parameters if provided
        if self.config and isinstance(self.config, dict) and self.config.get('custom_params'):
            custom_params = self.config.get('custom_params', {})
            tprint_info(f"🔧 Applying {len(custom_params)} custom parameters")

            # Update parameters
            for key in ['min_entry_window_minutes', 'max_entry_window_minutes',
                       'entry_quality_threshold', 'max_adverse_movement_pct',
                       'min_favorable_movement_pct', 'entry_quality_scoring_method',
                       'enable_regime_adaptive_labeling']:
                if key in custom_params:
                    setattr(tactician_config, key, custom_params[key])
                    tprint_info(f"   → {key}: {custom_params[key]}")

            # Handle VectorBT configuration
            # if 'vectorbt_config' in custom_params:
            #     vectorbt_params = custom_params['vectorbt_config']
            #     tactician_config.vectorbt_config = VectorBTConfig(**vectorbt_params)
            #     tprint_info("⚡ Applied custom VectorBT configuration")
            # elif 'enable_vectorbt' in custom_params:
            #     # Create VectorBT config with basic settings
            #     tactician_config.vectorbt_config = VectorBTConfig(
            #         enable_vectorbt=custom_params.get('enable_vectorbt', True),
            #         vectorbt_threshold=custom_params.get('vectorbt_threshold', 1000),
            #         performance_monitoring=custom_params.get('performance_monitoring', True)
            #     )
            #     tprint_info("⚡ Created basic VectorBT configuration")

        # Create the labeler
        try:
            self.labeler = TacticianDifferentiatedLabeler(tactician_config)
            tprint_success("✅ TacticianEntryLabelerComponent initialized successfully")
        except Exception as e:
            tprint_error(f"❌ Failed to initialize TacticianEntryLabelerComponent: {e}")
            raise

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        artifacts = ['multi_horizon_labeling_result', 'labeling_report']
        tprint_info(f"📋 Required artifacts: {artifacts}")
        return artifacts

    async def execute(self, data: Any, pipeline_state: PipelineState) -> ComponentResult:
        """
        Execute Tactician entry labeling as a component.

        Args:
            data: Input data (typically market data DataFrame)
            pipeline_state: Current pipeline state

        Returns:
            ComponentResult with labeling results and artifacts
        """
        try:
            tprint_info("🚀 Starting Tactician Entry Labeling execution...")

            # Start timing
            start_time = time.time()

            # Extract data from pipeline state if not provided
            if data is None:
                data = pipeline_state.get('prepared_data')
                if data is None:
                    raise ValueError("No input data provided and no prepared_data in pipeline state")

            # Extract analyst signals and regime assignments if available
            analyst_predictions = pipeline_state.get('analyst_predictions')
            analyst_signals = None
            if analyst_predictions is not None:
                if isinstance(analyst_predictions, pd.DataFrame):
                    # Try to extract signals from various possible column names
                    for col in ['analyst_signal', 'green_light', 'signal', 'confidence']:
                        if col in analyst_predictions.columns:
                            analyst_signals = analyst_predictions[col]
                            break

            regime_assignments = pipeline_state.get('regime_assignments')
            if regime_assignments is not None:
                if isinstance(regime_assignments, pd.DataFrame):
                    regime_assignments = regime_assignments.iloc[:, 0]  # Take first column
                tprint_info(f"📊 Using regime assignments for adaptive labeling")

            # Generate labels with error handling
            try:
                labels, quality_metrics = self.labeler.create_entry_timing_labels(
                    data=data,
                    analyst_signals=analyst_signals,
                    regime_assignments=regime_assignments
                )
            except Exception as e:
                tprint_error(f"❌ Error during label generation: {e}")
                raise ValueError(f"Failed to generate entry timing labels: {e}") from e

            # Validate generated labels
            if labels is None or len(labels) == 0:
                raise ValueError("No labels generated - check data quality and configuration")
            
            if not isinstance(labels, pd.Series):
                raise ValueError(f"Expected labels to be pd.Series, got {type(labels)}")
            
            if len(labels) != len(data):
                raise ValueError(f"Label length ({len(labels)}) doesn't match data length ({len(data)})")

            # Calculate processing time
            processing_time = time.time() - start_time

            # Create labels DataFrame
            label_column = 'tactician_entry_target'
            label_df = pd.DataFrame({label_column: labels}, index=data.index)
            confidence_df = pd.DataFrame(
                {f'{label_column}_confidence': labels.clip(lower=0.0, upper=1.0)},
                index=data.index
            )
            eligibility_df = pd.DataFrame(
                {f'{label_column}_eligibility': (labels > 0).astype(int)},
                index=data.index
            )

            # Create quality scores in expected format
            quality_scores = {
                label_column: {
                    'overall_quality': quality_metrics.get('overall_quality', 0.0),
                    'predictability': quality_metrics.get('avg_entry_quality', 0.0),
                    'stability': max(0.0, 1.0 - quality_metrics.get('entry_quality_std', 0.0)),
                    'balance': quality_metrics.get('labeling_coverage', 0.0),
                    'auc_mean': quality_metrics.get('avg_entry_quality', 0.0),
                    'class_balance': quality_metrics.get('entry_density', 0.0)
                }
            }

            # Save labeled data to parquet file for persistence
            from pathlib import Path
            symbol = pipeline_state.get('symbol', 'UNKNOWN')
            exchange = pipeline_state.get('exchange', 'UNKNOWN')
            timeframe = pipeline_state.get('timeframe', 'UNKNOWN')
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

            artifacts_dir = Path('artifacts')
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            labeled_data_file = artifacts_dir / f'tactician_labeled_data_{symbol}_{exchange}_{timeframe}_{timestamp_str}.parquet'

            # Save labeled DataFrame to parquet
            if isinstance(label_df, pd.DataFrame) and not label_df.empty:
                label_df.to_parquet(labeled_data_file)
                tprint_success(f"✅ Saved tactician labeled data to {labeled_data_file}")

            # Create artifacts
            artifacts = {
                'multi_horizon_labeling_result': {
                    'labeled_data': label_df,
                    'labeled_data_file': str(labeled_data_file),  # Add file path for persistence
                    'labels': label_df,
                    'confidence_scores': confidence_df,
                    'eligibility_masks': eligibility_df,
                    'quality_scores': quality_scores,
                    'normalization_factors': {
                        'scaling_reference': 'Entry quality normalized scoring',
                        'quality_threshold': quality_metrics.get('quality_threshold', 0.0),
                        'balance_factor': quality_metrics.get('labeling_coverage', 0.0)
                    },
                    'processing_time': processing_time,
                    'n_samples': len(label_df),
                    'n_targets': 1,
                    'n_horizons': 1,
                    'method': 'tactician_entry_labeling',
                    'metadata': {
                        'symbol': pipeline_state.get('symbol', 'UNKNOWN'),
                        'exchange': pipeline_state.get('exchange', 'UNKNOWN'),
                        'timeframe': pipeline_state.get('timeframe', '15m'),
                        'label_focus': 'entry_timing',
                        'regime_aware': bool(regime_assignments is not None),
                        'processing_time': processing_time,
                        'n_samples': len(label_df),
                        'n_targets': 1,
                        'n_horizons': 1,
                        'source': 'all_market_data'
                    }
                },
                'labeling_report': {
                    'status': 'completed',
                    'timestamp': datetime.now().isoformat(),
                    'method': 'tactician_entry_labeling',
                    'summary': quality_metrics,
                    'entry_points': int((labels > 0).sum()),
                    'regime_aware': bool(regime_assignments is not None)
                }
            }

            # Create result
            result = ComponentResult(
                success=True,
                processed_data=label_df,
                metadata={
                    'component': 'tactician_entry_labeler',
                    'timeframe': self.config.get('parameters', {}).get('timeframe', '15m') if self.config and isinstance(self.config, dict) else '15m',
                    'artifacts': artifacts,
                    'n_entry_points': int((labels > 0).sum()),
                    'quality_metrics': quality_metrics,
                    'direction_settings': {
                        'enable_long_positions': self.labeler.config.enable_long_positions,
                        'enable_short_positions': self.labeler.config.enable_short_positions,
                    }
                }
            )

            # Generate outcome file with datetime stamp
            try:
                outcome_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                outcomes_dir = Path('outcomes')
                ensure_directory(outcomes_dir)

                outcome_filename = f"tactician_labeler_outcome_{outcome_timestamp}.json"
                outcome_path = outcomes_dir / outcome_filename

                # Create comprehensive outcome report with detailed statistics

                # Entry distribution analysis
                entry_distribution = {
                    'total_samples': len(labels),
                    'entry_points': int((labels > 0).sum()),
                    'non_entry_points': int((labels == 0).sum()),
                    'entry_rate': float((labels > 0).sum() / len(labels) * 100) if len(labels) > 0 else 0.0,
                    'entry_quality_stats': {
                        'mean': float(labels[labels > 0].mean()) if (labels > 0).sum() > 0 else 0.0,
                        'median': float(labels[labels > 0].median()) if (labels > 0).sum() > 0 else 0.0,
                        'std': float(labels[labels > 0].std()) if (labels > 0).sum() > 0 else 0.0,
                        'min': float(labels[labels > 0].min()) if (labels > 0).sum() > 0 else 0.0,
                        'max': float(labels[labels > 0].max()) if (labels > 0).sum() > 0 else 0.0,
                        'percentile_25': float(labels[labels > 0].quantile(0.25)) if (labels > 0).sum() > 0 else 0.0,
                        'percentile_75': float(labels[labels > 0].quantile(0.75)) if (labels > 0).sum() > 0 else 0.0,
                    }
                }

                # Regime-specific analysis if available
                regime_analysis = {}
                if regime_assignments is not None:
                    try:
                        regime_groups = pd.DataFrame({'label': labels, 'regime': regime_assignments})
                        for regime in regime_groups['regime'].unique():
                            regime_labels = regime_groups[regime_groups['regime'] == regime]['label']
                            regime_analysis[str(regime)] = {
                                'total_samples': int(len(regime_labels)),
                                'entry_points': int((regime_labels > 0).sum()),
                                'entry_rate': float((regime_labels > 0).sum() / len(regime_labels) * 100) if len(regime_labels) > 0 else 0.0,
                                'avg_entry_quality': float(regime_labels[regime_labels > 0].mean()) if (regime_labels > 0).sum() > 0 else 0.0,
                            }
                    except Exception as e:
                        regime_analysis['error'] = str(e)

                # Timing analysis
                timing_analysis = {
                    'entry_window': {
                        'min_minutes': self.labeler.config.min_entry_window_minutes,
                        'max_minutes': self.labeler.config.max_entry_window_minutes,
                        'avg_minutes': (self.labeler.config.min_entry_window_minutes + self.labeler.config.max_entry_window_minutes) / 2,
                    },
                    'movement_expectations': {
                        'max_adverse_pct': self.labeler.config.max_adverse_movement_pct,
                        'min_favorable_pct': self.labeler.config.min_favorable_movement_pct,
                        'risk_reward_ratio': self.labeler.config.min_favorable_movement_pct / self.labeler.config.max_adverse_movement_pct if self.labeler.config.max_adverse_movement_pct > 0 else 0.0,
                    }
                }

                # Data quality assessment
                data_quality = {
                    'input_data': {
                        'rows': len(data),
                        'columns': len(data.columns),
                        'date_range': {
                            'start': str(data.index.min()) if hasattr(data.index, 'min') else None,
                            'end': str(data.index.max()) if hasattr(data.index, 'max') else None,
                            'duration_days': float((data.index.max() - data.index.min()).total_seconds() / 86400) if hasattr(data.index, 'min') and hasattr(data.index, 'max') else None,
                        },
                        'missing_values': int(data.isnull().sum().sum()),
                        'missing_percentage': float(data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100),
                    },
                    'output_labels': {
                        'total_generated': len(labels),
                        'label_coverage': 100.0,  # All samples get a label (even if 0)
                        'valid_entries': int((labels > 0).sum()),
                        'valid_entry_rate': float((labels > 0).sum() / len(labels) * 100) if len(labels) > 0 else 0.0,
                    }
                }

                outcome_data = {
                    'component': 'tactician_entry_labeler',
                    'timestamp': datetime.now().isoformat(),
                    'execution_time': processing_time,
                    'timeframe': self.config.get('parameters', {}).get('timeframe', '15m') if self.config and isinstance(self.config, dict) else '15m',
                    'configuration': {
                        'min_entry_window_minutes': self.labeler.config.min_entry_window_minutes,
                        'max_entry_window_minutes': self.labeler.config.max_entry_window_minutes,
                        'entry_quality_threshold': self.labeler.config.entry_quality_threshold,
                        'max_adverse_movement_pct': self.labeler.config.max_adverse_movement_pct,
                        'min_favorable_movement_pct': self.labeler.config.min_favorable_movement_pct,
                        'entry_quality_scoring_method': self.labeler.config.entry_quality_scoring_method,
                        'enable_regime_adaptive_labeling': self.labeler.config.enable_regime_adaptive_labeling,
                        'enable_interaction_terms': self.labeler.config.enable_interaction_terms,
                        'enable_penalty_system': self.labeler.config.enable_penalty_system,
                        'risk_aversion': self.labeler.config.risk_aversion,
                        'vectorbt_config': {
                            'enable_vectorbt': self.labeler.config.vectorbt_config.get('enabled', False) if self.labeler.config.vectorbt_config else False,
                            'vectorbt_threshold': self.labeler.config.vectorbt_config.get('threshold', 1000) if self.labeler.config.vectorbt_config else 1000,
                            'performance_monitoring': self.labeler.config.vectorbt_config.get('optimization_level', 'balanced') if self.labeler.config.vectorbt_config else 'balanced',
                        } if self.labeler.config.vectorbt_config else None,
                    },
                    'results': {
                        'n_samples': len(label_df),
                        'n_entry_points': int((labels > 0).sum()),
                        'entry_density': quality_metrics.get('entry_density', 0.0),
                        'labeling_coverage': quality_metrics.get('labeling_coverage', 0.0),
                        'quality_metrics': quality_metrics,
                        'entry_distribution': entry_distribution,
                        'regime_analysis': regime_analysis,
                        'timing_analysis': timing_analysis,
                    },
                    'quality_scores': quality_scores,
                    'data_quality': data_quality,
                    'data_info': {
                        'input_rows': len(data),
                        'input_columns': len(data.columns),
                        'analyst_signals_available': analyst_signals is not None,
                        'analyst_signals_count': int(analyst_signals.sum()) if analyst_signals is not None and hasattr(analyst_signals, 'sum') else None,
                        'regime_assignments_available': regime_assignments is not None,
                        'regime_count': int(regime_assignments.nunique()) if regime_assignments is not None and hasattr(regime_assignments, 'nunique') else None,
                    },
                    'confidence_statistics': {
                        'mean_confidence': float(confidence_df.iloc[:, 0].mean()) if len(confidence_df) > 0 else 0.0,
                        'median_confidence': float(confidence_df.iloc[:, 0].median()) if len(confidence_df) > 0 else 0.0,
                        'min_confidence': float(confidence_df.iloc[:, 0].min()) if len(confidence_df) > 0 else 0.0,
                        'max_confidence': float(confidence_df.iloc[:, 0].max()) if len(confidence_df) > 0 else 0.0,
                    },
                    'eligibility_statistics': {
                        'eligible_samples': int(eligibility_df.iloc[:, 0].sum()) if len(eligibility_df) > 0 else 0,
                        'eligibility_rate': float(eligibility_df.iloc[:, 0].sum() / len(eligibility_df) * 100) if len(eligibility_df) > 0 else 0.0,
                    },
                    'vectorbt_performance': self.labeler.vectorbt_optimizer.get_performance_summary() if hasattr(self.labeler, 'vectorbt_optimizer') and self.labeler.vectorbt_optimizer is not None else None,
                    'status': 'success'
                }

                safe_json_dump(outcome_data, str(outcome_path))
                tprint_success(f"📄 Outcome file saved: {outcome_filename}")

            except Exception as outcome_error:
                tprint_warning(f"⚠️ Failed to save outcome file: {outcome_error}")
                # Don't fail the component if outcome file generation fails

            tprint_success("✅ Tactician Entry Labeling completed successfully")
            return result

        except Exception as e:
            tprint_error(f"❌ Tactician Entry Labeling failed: {e}")

            # Create detailed error information
            import traceback
            error_details = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc(),
                'component': 'tactician_entry_labeler',
                'timestamp': datetime.now().isoformat()
            }

            result = ComponentResult(
                success=False,
                error_message=str(e),
                metadata={
                    'component': 'tactician_entry_labeler',
                    'error_details': error_details
                }
            )
            return result

    def process(self, data: Any) -> Any:
        """Process the input data and return the result."""
        try:
            # Extract required data from the input
            if hasattr(data, 'data') and hasattr(data, 'regime_assignments'):
                # Data is already in the expected format
                return self.execute_tactician_entry_labeling(data.data, data.regime_assignments)
            elif isinstance(data, dict) and 'data' in data and 'regime_assignments' in data:
                # Data is a dictionary with the required keys
                return self.execute_tactician_entry_labeling(data['data'], data['regime_assignments'])
            else:
                # Try to extract data and regime_assignments from the input
                if hasattr(data, 'data'):
                    data_obj = data.data
                else:
                    data_obj = data
                
                if hasattr(data, 'regime_assignments'):
                    regime_assignments = data.regime_assignments
                else:
                    # Try to get regime assignments from the data object
                    if hasattr(data_obj, 'regime_assignments'):
                        regime_assignments = data_obj.regime_assignments
                    else:
                        raise ValueError("Could not find regime_assignments in the input data")
                
                return self.execute_tactician_entry_labeling(data_obj, regime_assignments)
        except Exception as e:
            self.logger.error(f"Error processing data in TacticianEntryLabelerComponent: {e}")
            raise

    def validate(self, data: Any) -> bool:
        """Validate the input data."""
        try:
            # Check if data is not None
            if data is None:
                self.logger.warning("Input data is None")
                return False
            
            # Check if we can extract the required data
            if hasattr(data, 'data') and hasattr(data, 'regime_assignments'):
                # Data is already in the expected format
                return True
            elif isinstance(data, dict) and 'data' in data and 'regime_assignments' in data:
                # Data is a dictionary with the required keys
                return True
            else:
                # Try to extract data and regime_assignments from the input
                if hasattr(data, 'data'):
                    data_obj = data.data
                else:
                    data_obj = data
                
                if hasattr(data, 'regime_assignments'):
                    regime_assignments = data.regime_assignments
                else:
                    # Try to get regime assignments from the data object
                    if hasattr(data_obj, 'regime_assignments'):
                        regime_assignments = data_obj.regime_assignments
                    else:
                        self.logger.warning("Could not find regime_assignments in the input data")
                        return False
                
                # Validate that we have the required data
                if data_obj is None:
                    self.logger.warning("Data object is None")
                    return False
                
                if regime_assignments is None:
                    self.logger.warning("Regime assignments are None")
                    return False
                
                return True
        except Exception as e:
            self.logger.error(f"Error validating data in TacticianEntryLabelerComponent: {e}")
            return False

# Register component with factory
def _register_tactician_entry_labeler():
    """Register the tactician entry labeler component with the factory."""
    try:
        from src.training.steps.pre_training.components import ComponentFactory
        ComponentFactory.register_component(
            'tactician_entry_labeler',
            TacticianEntryLabelerComponent
        )
    except ImportError:
        # Component factory not available, skip registration
        pass

# Register the component when module is imported
_register_tactician_entry_labeler()
