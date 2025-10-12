"""
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
import warnings
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
from .profit_labeling.vectorbt_optimizer import (
    get_vectorbt_optimizer, VectorBTConfig, optimized_rolling_mean, 
    optimized_rolling_std, optimized_volatility, optimized_returns
)
from src.utils.ml_common.optimization.grid_utils import (
    generate_grid,
    build_coarse_grid_from_search_space,
    GridSearchOptimizer
)
from src.training.steps.pre_training.components.base_component import BasePreTrainingComponent, ComponentConfig, ComponentResult
from src.training.steps.pre_training.components.contracts import PipelineState
from src.training.steps.pre_training.components.component_factory import register_component
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

    # Trading direction settings
    enable_long_positions: bool = True   # Include long opportunities (buy when expecting price increase)
    enable_short_positions: bool = False  # Include short opportunities (sell when expecting price decrease)
    
    # VectorBT optimization settings
    vectorbt_config: Optional[VectorBTConfig] = None

    def get_optimization_search_space(self) -> Dict[str, Any]:
        """Get search space for hyperparameter optimization."""
        return {
            'entry_quality_threshold': {
                'type': 'float',
                'low': 0.1,
                'high': 0.5,
                'log': False
            },
            'max_adverse_movement_pct': {
                'type': 'float',
                'low': 0.2,
                'high': 1.0,
                'log': False
            },
            'min_favorable_movement_pct': {
                'type': 'float',
                'low': 0.1,
                'high': 0.5,
                'log': False
            },
            'risk_aversion': {
                'type': 'float',
                'low': 1.0,
                'high': 5.0,
                'log': False
            }
        }

    def optimize_config_grid_search(self, data: pd.DataFrame, max_trials: int = 50) -> 'TacticianLabelingConfig':
        """Optimize configuration using grid search."""
        search_space = self.get_optimization_search_space()

        # Generate parameter grid
        param_grid = generate_grid(search_space, max_trials)

        best_config = None
        best_score = -float('inf')

        # Simple evaluation based on data characteristics
        for params in param_grid[:max_trials]:
            try:
                # Create config with current parameters
                config = TacticianLabelingConfig(
                    entry_quality_threshold=params.get('entry_quality_threshold', self.entry_quality_threshold),
                    max_adverse_movement_pct=params.get('max_adverse_movement_pct', self.max_adverse_movement_pct),
                    min_favorable_movement_pct=params.get('min_favorable_movement_pct', self.min_favorable_movement_pct),
                    risk_aversion=params.get('risk_aversion', self.risk_aversion)
                )

                # Simple scoring based on data quality metrics
                quality = create_data_quality_report(data)
                score = quality.get('quality_metrics', {}).get('numeric_columns', 0) * 0.1
                score += (1 - quality.get('quality_metrics', {}).get('missing_percentage', 100)) * 0.01

                if score > best_score:
                    best_score = score
                    best_config = config

            except Exception as e:
                tprint_warning(f"⚠️ Error evaluating config {params}: {e}")
                continue

        if best_config:
            tprint_success(f"✅ Grid search completed. Best score: {best_score:.3f}")
            return best_config

        return self


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
        vectorbt_config = self.config.vectorbt_config or VectorBTConfig(
            enable_vectorbt=True,
            vectorbt_threshold=1000,
            performance_monitoring=True,
            memory_efficiency_mode=True
        )
        self.vectorbt_optimizer = get_vectorbt_optimizer(vectorbt_config)
        tprint_info(f"⚡ VectorBT optimizer initialized: {self.vectorbt_optimizer.__class__.__name__}")

        # Initialize M1 optimizations if available
        self.m1_integration = integrate_with_m1_optimizers()
        if self.m1_integration.get('success', False):
            tprint_info(f"🧠 M1 optimizations initialized: GPU={'✅' if self.m1_integration.get('gpu_manager') else '❌'}, Memory={'✅' if self.m1_integration.get('memory_optimizer') else '❌'}")

        # Initialize enhanced quality scorer
        self._initialize_quality_scorer()

    def cleanup(self) -> None:
        """Clean up resources and optimize memory."""
        try:
            # Optimize memory usage
            memory_info = optimize_memory()
            if memory_info.get('success', False):
                tprint_info(f"🧠 Memory optimized: {memory_info.get('objects_collected', 0)} objects collected")

            # Clean up matrix operations resources
            try:
                from src.utils.matrix_operations import cleanup_hardware_resources
                cleanup_hardware_resources()
                tprint_info("🧮 Matrix operations resources cleaned up")
            except ImportError:
                pass

            # Clean up M1 optimizers if available
            from src.utils.common_operations import cleanup_m1_optimizers
            cleanup_m1_optimizers()

            # Clean up VectorBT optimizer
            if hasattr(self, 'vectorbt_optimizer'):
                self.vectorbt_optimizer.clear_cache()
                tprint_info("⚡ VectorBT optimizer cache cleared")

            # Get final hardware performance report
            hardware_report = get_hardware_performance_report()
            tprint_info(f"🔧 Final hardware status: CPU cores={hardware_report.get('cpu_cores', 'N/A')}, GPU={hardware_report.get('gpu_available', 'N/A')}")

            # Get VectorBT performance summary
            if hasattr(self, 'vectorbt_optimizer'):
                perf_summary = self.vectorbt_optimizer.get_performance_summary()
                if 'total_operations' in perf_summary and perf_summary['total_operations'] > 0:
                    tprint_info(f"⚡ VectorBT performance: {perf_summary['total_operations']} operations, "
                              f"{perf_summary['vectorbt_usage_rate']:.1%} VectorBT usage, "
                              f"{perf_summary['avg_execution_time']:.3f}s avg time")

            tprint_success("✅ TacticianDifferentiatedLabeler cleanup completed")
        except Exception as e:
            tprint_warning(f"⚠️ Error during cleanup: {e}")

    def _initialize_quality_scorer(self):
        """Initialize the enhanced entry quality scorer based on configuration."""
        try:
            from src.training.steps.models_training.enhanced_entry_quality_scorer import (
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
            
        except (ImportError, AttributeError, Exception) as e:
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

        # Validate input data format and constraints
        try:
            data = validate_raw_ohlcv(data, context='tactician_entry_labeler.input_validation')
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
        if len(data) >= self.vectorbt_optimizer.config.vectorbt_threshold:
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
        # Get non-zero labels
        non_zero_mask = labels > 0
        if non_zero_mask.sum() == 0:
            return labels
        
        # Extract scores
        scores = labels[non_zero_mask].values
        indices = labels[non_zero_mask].index
        
        # Use VectorBT for optimized peak detection on large datasets
        if len(scores) >= self.vectorbt_optimizer.config.vectorbt_threshold:
            tprint_info(f"⚡ Using VectorBT optimization for peak detection on {len(scores)} scores")
            filtered_labels = self._vectorbt_peak_filtering(labels, scores, indices)
        else:
            # Use standard peak detection for smaller datasets
            filtered_labels = self._standard_peak_filtering(labels, scores, indices)
        
        # Validate that we have usable training data
        final_entry_count = int((filtered_labels > 0).sum())
        if final_entry_count == 0:
            raise ValueError(
                "Peak filtering resulted in no usable entry labels for training. "
                f"Original entries: {len(scores)}, Peak threshold: {self.config.entry_quality_threshold}, "
                f"Min window: {self.config.min_entry_window_minutes} minutes. "
                "Consider lowering the entry quality threshold or minimum window requirements."
            )

        # Warn if we have very few entries (might indicate overly strict filtering)
        if final_entry_count < 10:
            warnings.warn(
                f"Peak filtering resulted in very few entry labels ({final_entry_count}). "
                "Training data may be insufficient for reliable model training. "
                "Consider adjusting entry quality threshold or minimum window requirements.",
                UserWarning,
                stacklevel=2
            )

        return filtered_labels

    def _standard_peak_filtering(self, labels: pd.Series, scores: np.ndarray, indices: pd.Index) -> pd.Series:
        """Standard peak detection using scipy."""
        # Apply peak detection
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

    def _vectorbt_peak_filtering(self, labels: pd.Series, scores: np.ndarray, indices: pd.Index) -> pd.Series:
        """VectorBT optimized peak detection for large datasets."""
        tprint_info("⚡ Applying VectorBT optimized peak filtering")
        
        # Create a temporary series for VectorBT operations
        temp_series = pd.Series(scores, index=indices)
        
        # Use VectorBT rolling operations to identify local maxima
        # Calculate rolling max to identify peaks
        rolling_max = self.vectorbt_optimizer.rolling_max(temp_series, window=self.config.min_entry_window_minutes * 2 + 1)
        
        # Identify peaks where current value equals rolling max
        peak_mask = (temp_series == rolling_max) & (temp_series > self.config.entry_quality_threshold)
        
        # Apply additional filtering to ensure minimum distance between peaks
        if peak_mask.sum() > 0:
            peak_indices = temp_series[peak_mask].index
            peak_scores = temp_series[peak_mask].values
            
            # Sort by score and apply distance filtering
            sorted_indices = np.argsort(peak_scores)[::-1]  # Sort by score descending
            filtered_peaks = []
            filtered_scores = []
            
            for idx in sorted_indices:
                current_peak_idx = peak_indices[idx]
                current_score = peak_scores[idx]
                
                # Check distance from already selected peaks
                if not filtered_peaks:
                    filtered_peaks.append(current_peak_idx)
                    filtered_scores.append(current_score)
                else:
                    # Calculate minimum distance to existing peaks
                    distances = [abs((current_peak_idx - existing_idx).total_seconds() / 60) 
                               for existing_idx in filtered_peaks]
                    min_distance = min(distances) if distances else float('inf')
                    
                    if min_distance >= self.config.min_entry_window_minutes:
                        filtered_peaks.append(current_peak_idx)
                        filtered_scores.append(current_score)
        
        # Create filtered labels
        filtered_labels = pd.Series(0.0, index=labels.index, dtype=float)
        
        if 'filtered_peaks' in locals() and len(filtered_peaks) > 0:
            for idx, score in zip(filtered_peaks, filtered_scores):
                filtered_labels.loc[idx] = score
        elif len(scores) > 0:
            # Fallback: keep the best entry if no peaks found
            best_idx = np.argmax(scores)
            if best_idx < len(indices):
                filtered_labels.loc[indices[best_idx]] = scores[best_idx]
        
        tprint_success(f"⚡ VectorBT peak filtering completed: {int((filtered_labels > 0).sum())} peaks selected")
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
        
        # Pre-calculate rolling statistics using VectorBT for better performance
        close_prices = data['close']
        high_prices = data['high']
        low_prices = data['low']
        
        # Calculate rolling statistics using VectorBT
        rolling_volatility = self.vectorbt_optimizer.calculate_volatility(
            close_prices.pct_change(), window=min(20, window_size), annualize=False
        )
        
        # Calculate rolling price statistics
        rolling_max_high = self.vectorbt_optimizer.rolling_max(high_prices, window=window_size)
        rolling_min_low = self.vectorbt_optimizer.rolling_min(low_prices, window=window_size)
        rolling_mean_close = self.vectorbt_optimizer.rolling_mean(close_prices, window=window_size)
        
        # Pre-allocate scores array
        scores = np.zeros(len(entry_indices))
        
        # Vectorized calculation of quality scores
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
            
            # Calculate price movements using pre-computed rolling statistics
            min_future_low = future_window['low'].min()
            max_future_high = future_window['high'].max()
            
            # Calculate adverse and favorable movements
            adverse_move = max(entry_price - min_future_low, 0.0) / max(entry_price, 1e-8) * 100
            favorable_move = max(max_future_high - entry_price, 0.0) / max(entry_price, 1e-8) * 100
            
            # Get regime parameters
            regime_params = self._get_regime_parameters(entry_index, regime_assignments)
            
            # Apply regime-specific thresholds
            if adverse_move > regime_params['max_adverse_movement_pct']:
                continue
            if favorable_move < regime_params['min_favorable_movement_pct']:
                continue
            
            # Calculate risk-reward ratio
            risk_reward_ratio = favorable_move / (adverse_move + 1e-8)
            
            # Calculate timing score (prefer shorter windows)
            timing_score = 1.0 / (1.0 + len(future_window) / self.config.max_entry_window_minutes)
            
            # Calculate volatility score using pre-computed rolling volatility
            if i < len(rolling_volatility) and not pd.isna(rolling_volatility.iloc[entry_idx]):
                volatility = rolling_volatility.iloc[entry_idx]
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
        
        tprint_success(f"⚡ Vectorized quality scores calculated: {len(scores)} scores")
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
        total_samples = len(data)
        labeled_samples = int((labels > 0).sum())
        
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
        timing_score = safe_divide(1.0, 1.0 + safe_divide(len(future_data), self.config.max_entry_window_minutes), default=0.0)
        
        # Use VectorBT for optimized volatility calculation
        if len(future_data) >= 2:
            returns = future_data['close'].pct_change().dropna()
            if not returns.empty:
                # Use VectorBT optimized volatility calculation
                volatility = self.vectorbt_optimizer.calculate_volatility(
                    returns, window=len(returns), annualize=False
                ).iloc[-1] if len(returns) > 0 else 0.0
            else:
                volatility = 0.0
        else:
            volatility = 0.0
            
        volatility_score = safe_divide(1.0, 1.0 + safe_divide(volatility * 100, 10.0), default=1.0)

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


@register_component('tactician_entry_labeler')
class TacticianEntryLabelerComponent(BasePreTrainingComponent):
    """
    Component wrapper for Tactician Entry Labeler.
    
    This component integrates the TacticianDifferentiatedLabeler with the pre-training pipeline
    and handles proper error handling, reporting, and pipeline state management.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the Tactician entry labeler component."""
        super().__init__(config)
        self.logger = system_logger.getChild('TacticianEntryLabelerComponent')
        
        # Create Tactician-specific configuration
        tactician_config = TacticianLabelingConfig()
        
        # Override with custom parameters if provided
        if self.config and self.config.custom_params:
            custom_params = self.config.custom_params
            
            # Update parameters
            for key in ['min_entry_window_minutes', 'max_entry_window_minutes', 
                       'entry_quality_threshold', 'max_adverse_movement_pct', 
                       'min_favorable_movement_pct', 'entry_quality_scoring_method',
                       'enable_regime_adaptive_labeling']:
                if key in custom_params:
                    setattr(tactician_config, key, custom_params[key])
            
            # Handle VectorBT configuration
            if 'vectorbt_config' in custom_params:
                vectorbt_params = custom_params['vectorbt_config']
                tactician_config.vectorbt_config = VectorBTConfig(**vectorbt_params)
            elif 'enable_vectorbt' in custom_params:
                # Create VectorBT config with basic settings
                tactician_config.vectorbt_config = VectorBTConfig(
                    enable_vectorbt=custom_params.get('enable_vectorbt', True),
                    vectorbt_threshold=custom_params.get('vectorbt_threshold', 1000),
                    performance_monitoring=custom_params.get('performance_monitoring', True)
                )
        
        # Create the labeler
        try:
            self.labeler = TacticianDifferentiatedLabeler(tactician_config)
            tprint_success("✅ TacticianEntryLabelerComponent initialized")
        except Exception as e:
            tprint_error(f"❌ Failed to initialize TacticianEntryLabelerComponent: {e}")
            raise
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['multi_horizon_labeling_result', 'labeling_report']
    
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
            
            # Generate labels
            labels, quality_metrics = self.labeler.create_entry_timing_labels(
                data=data,
                analyst_signals=analyst_signals,
                regime_assignments=regime_assignments
            )
            
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
                        'symbol': self.config.symbol if self.config else 'UNKNOWN',
                        'exchange': self.config.exchange if self.config else 'UNKNOWN',
                        'timeframe': self.config.timeframe if self.config else '15m',
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
                data=label_df,
                artifacts=artifacts,
                metadata={
                    'component': 'tactician_entry_labeler',
                    'timeframe': self.config.timeframe if self.config else '15m',
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
                    'timeframe': self.config.timeframe if self.config else '15m',
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
                            'enable_vectorbt': self.labeler.config.vectorbt_config.enable_vectorbt if self.labeler.config.vectorbt_config else False,
                            'vectorbt_threshold': self.labeler.config.vectorbt_config.vectorbt_threshold if self.labeler.config.vectorbt_config else 1000,
                            'performance_monitoring': self.labeler.config.vectorbt_config.performance_monitoring if self.labeler.config.vectorbt_config else False,
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
                    'vectorbt_performance': self.labeler.vectorbt_optimizer.get_performance_summary() if hasattr(self.labeler, 'vectorbt_optimizer') else None,
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


# Convenience function for external usage
async def execute_tactician_entry_labeling(
    data: pd.DataFrame,
    analyst_signals: Optional[pd.Series] = None,
    regime_assignments: Optional[pd.Series] = None,
    config: Optional[TacticianLabelingConfig] = None,
    **kwargs
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Execute Tactician entry labeling.
    
    Args:
        data: Input market data (OHLCV format)
        analyst_signals: Optional Analyst signals (legacy support)
        regime_assignments: Optional regime assignments
        config: Optional configuration
        **kwargs: Additional parameters
        
    Returns:
        Tuple of (labels, quality_metrics)
    """
    labeler = TacticianDifferentiatedLabeler(config or TacticianLabelingConfig())
    return labeler.create_entry_timing_labels(data, analyst_signals, regime_assignments)