"""
Analyst Profit Labeler - Specialized Single-Target Labeling for Analyst Models

This module provides a specialized profit labeling component for Analyst models,
using the VolatilityAwareMultiHorizonLabeler with Analyst-specific configurations.

Key Features:
- 15m timeframe optimization for strategic decision-making
- Single-target profit labeling (0.5% base threshold modulated by volatility)
- Single horizon (90 minutes = 6 periods of 15m) for strategic decision-making
- Optimal entry point detection - analyzes price variation over rolling windows
- Local extrema entry - finds local minima/maxima BEFORE price action as optimal entry point
- Volatility-aware threshold modulation
- Enhanced label quality scoring
- Per-regime/cluster optimization support
- Consolidated opportunity flagging (single flag per profitable move across horizons)
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.logger import system_logger
from src.utils.common_operations import (
    validate_dataframe_columns,
    safe_dataframe_operation,
    validate_positive,
    safe_int,
    get_dataframe_info,
    create_data_quality_report,
    ensure_directory,
    safe_json_dump,
    format_bytes,
    memory_checkpoint,
    optimize_memory,
    integrate_with_m1_optimizers
)
from src.utils.common_utilities import (
    analyze_nan_values_detailed,
    format_nan_analysis_report
)
from src.utils.matrix_operations import (
    get_unified_matrix_operations,
    optimize_dataframe,
    get_hardware_performance_report
)
from src.utils.ml_common.optimization.grid_utils import (
    generate_grid
)
from src.training.steps.pre_training.components.base_component import BasePreTrainingComponent, ComponentConfig, ComponentResult
from src.training.steps.pre_training.components.contracts import PipelineState
from src.training.steps.pre_training.components import ComponentFactory
from src.training.steps.pre_training.validation.schemas import validate_raw_ohlcv, SchemaValidationException

# Import the volatility-aware labeler
try:
    from src.training.steps.pre_training.profit_labeling.volatility_aware_labeler import (
        VolatilityAwareMultiHorizonLabeler,
        VolatilityAwareConfig,
        LabelingResult,
        LabelDefinitionType,
        create_enhanced_analyst_labeler,
    )
    VOLATILITY_LABELER_AVAILABLE = True
except (ImportError, SyntaxError):
    VOLATILITY_LABELER_AVAILABLE = False
    VolatilityAwareMultiHorizonLabeler = None
    VolatilityAwareConfig = None
    LabelingResult = None
    LabelDefinitionType = None
    create_enhanced_analyst_labeler = None

# Import advanced filters for 15m timeframe from feature engineering
try:
    from src.training.steps.feature_engineering.filters.advanced_filters_15m import (
        AdvancedFilters15m,
        AdvancedFiltersConfig,
        FilterResult,
        apply_advanced_filters_15m,
    )
    ADVANCED_FILTERS_AVAILABLE = True
except (ImportError, SyntaxError):
    ADVANCED_FILTERS_AVAILABLE = False
    AdvancedFilters15m = None
    AdvancedFiltersConfig = None
    FilterResult = None
    apply_advanced_filters_15m = None

@dataclass
class AnalystProfitLabelerConfig:
    """Configuration for Analyst profit labeling."""

    # Timeframe settings (Analyst operates on 15m)
    timeframe: str = "15m"
    base_period_minutes: int = 15

    # Horizon settings for Analyst (strategic decision-making)
    # Horizons are in MINUTES (must be >= timeframe period)
    # Single horizon default: 60 minutes (4 periods of 15m)
    horizons: List[int] = field(default_factory=lambda: [60])  # 60 minutes = 4 * 15m periods

    # Profit target (percentage points) - Simplified to single 0.5% threshold modulated by volatility
    # Note: This is a percentage point, not fractional return (0.005 = 0.5%)
    # The threshold will be dynamically adjusted based on market volatility
    target_profit: float = 0.6
    
    # For backward compatibility and metadata consistency, expose as target_profits list
    @property
    def target_profits(self) -> List[float]:
        """Return target_profit as a single-item list for API consistency."""
        return [self.target_profit]

    # Volatility-aware settings
    # Enable volatility normalization so base 0.5% scales with regime
    use_volatility_normalization: bool = True
    volatility_window: int = 20

    # Label quality thresholds
    min_label_quality: float = 0.6
    min_predictability: float = 0.55

    # Per-regime optimization
    enable_regime_adaptation: bool = True

    # Trading direction settings
    enable_long_positions: bool = True   # Include long opportunities (buy when expecting price increase)
    enable_short_positions: bool = False  # Include short opportunities (sell when expecting price decrease)

    # Advanced filtering for 15m timeframe
    enable_advanced_filters: bool = True
    advanced_filters_config: Optional[AdvancedFiltersConfig] = None
    # Optimal entry point detection
    # Strategy: Analyze price variation over rolling windows to find optimal entry points
    # 1. Check price variation over rolling windows (15m to 150m, up to 10 windows)
    # 2. If price variation clears opportunity threshold (0.5%), find local minima/maxima
    # 3. Flag the bar with highest price gap BEFORE the price action as optimal entry point
    enable_optimal_entry_detection: bool = True  # Enable optimal entry point detection
    entry_threshold: float = 0.5  # Minimum price variation to consider an opportunity (0.5%)
    find_highest_gap_entry: bool = True  # Find bar with highest price gap as entry point
    entry_point_strategy: str = "local_extrema"  # Find local minima/maxima before price action

    # Volatility modulation settings for dynamic threshold adjustment
    volatility_modulation: bool = True  # Enable volatility-based threshold adjustment
    min_threshold_multiplier: float = 0.5  # Minimum threshold multiplier (0.5x base)
    max_threshold_multiplier: float = 2.0  # Maximum threshold multiplier (2.0x base)
    volatility_sensitivity: float = 1.5  # Volatility sensitivity factor (k)
    max_windows: int = 10  # Support up to 10 rolling windows

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        tprint_info("🔧 Validating AnalystProfitLabelerConfig...")
        try:
            self._validate_horizon_timeframe_compatibility()
            tprint_success("✅ Configuration validation completed")
        except Exception as e:
            tprint_error(f"❌ Configuration validation failed: {e}")
            raise

    def _validate_horizon_timeframe_compatibility(self) -> None:
        """Validate that horizons are compatible with the timeframe."""
        tprint_info(f"🔍 Validating horizon-timeframe compatibility for timeframe: {self.timeframe}")

        if not self.horizons:
            tprint_error("❌ Horizons list cannot be empty")
            raise ValueError("Horizons list cannot be empty")

        # Parse timeframe to get base period in minutes
        try:
            timeframe_minutes = self._parse_timeframe_to_minutes(self.timeframe)
            tprint_info(f"📊 Timeframe '{self.timeframe}' parsed to {timeframe_minutes} minutes")
        except Exception as e:
            tprint_error(f"❌ Failed to parse timeframe '{self.timeframe}': {e}")
            raise

        # Check each horizon
        problematic_horizons = []
        for horizon in self.horizons:
            if horizon < timeframe_minutes:
                problematic_horizons.append((horizon, timeframe_minutes))
                tprint_warning(f"⚠️ Horizon {horizon}m is less than timeframe {timeframe_minutes}m")

        if problematic_horizons:
            horizon_strs = [f"{h}m (horizon) vs {tf}m (timeframe)" for h, tf in problematic_horizons]
            error_msg = (
                f"Horizon(s) incompatible with timeframe '{self.timeframe}': {', '.join(horizon_strs)}. "
                "Horizon must be >= timeframe period to ensure sufficient data for labeling."
            )
            tprint_error(f"❌ {error_msg}")
            raise ValueError(error_msg)

        tprint_success(f"✅ All {len(self.horizons)} horizons are compatible with timeframe {self.timeframe}")

    def _parse_timeframe_to_minutes(self, timeframe: str) -> int:
        """Parse timeframe string to minutes."""
        tprint_info(f"🔍 Parsing timeframe: {timeframe}")

        try:
            if timeframe.endswith('m'):
                minutes = safe_int(timeframe[:-1])
                tprint_info(f"📊 Parsed {timeframe} as {minutes} minutes")
                return minutes
            elif timeframe.endswith('h'):
                hours = safe_int(timeframe[:-1])
                validate_positive(hours, "hours in timeframe")
                minutes = hours * 60
                tprint_info(f"📊 Parsed {timeframe} as {hours} hours = {minutes} minutes")
                return minutes
            elif timeframe.endswith('d'):
                days = safe_int(timeframe[:-1])
                validate_positive(days, "days in timeframe")
                minutes = days * 60 * 24
                tprint_info(f"📊 Parsed {timeframe} as {days} days = {minutes} minutes")
                return minutes
            else:
                tprint_error(f"❌ Unsupported timeframe format: {timeframe}")
                raise ValueError(f"Unsupported timeframe format: {timeframe}")
        except (ValueError, TypeError) as e:
            tprint_error(f"❌ Invalid timeframe format '{timeframe}': {e}")
            raise ValueError(f"Invalid timeframe format '{timeframe}': {e}")

    def get_optimization_search_space(self) -> Dict[str, Any]:
        """Get search space for hyperparameter optimization."""
        tprint_info("🔍 Generating optimization search space for grid search")

        search_space = {
            'target_profit': {
                'type': 'float',
                'low': 0.3,
                'high': 1.0,
                'log': False
            },
            'volatility_window': {
                'type': 'int',
                'low': 10,
                'high': 50
            },
            'min_label_quality': {
                'type': 'float',
                'low': 0.4,
                'high': 0.8,
                'log': False
            },
            'min_predictability': {
                'type': 'float',
                'low': 0.4,
                'high': 0.8,
                'log': False
            }
        }

        tprint_success(f"✅ Generated search space with {len(search_space)} parameters")
        return search_space

    def optimize_config_grid_search(self, data: pd.DataFrame, max_trials: int = 50) -> 'AnalystProfitLabelerConfig':
        """Optimize configuration using grid search."""
        tprint_info(f"🔍 Starting grid search optimization with {max_trials} trials")

        try:
            search_space = self.get_optimization_search_space()

            # Generate parameter grid
            tprint_info("📊 Generating parameter grid...")
            param_grid = generate_grid(search_space, max_trials)
            tprint_success(f"✅ Generated {len(param_grid)} parameter combinations")

            best_config = None
            best_score = -float('inf')
            successful_trials = 0

            # Simple evaluation based on data characteristics
            tprint_info("🧪 Evaluating parameter combinations...")
            for i, params in enumerate(param_grid[:max_trials]):
                try:
                    # Create config with current parameters
                    config = AnalystProfitLabelerConfig(
                        horizons=self.horizons,
                        target_profit=params.get('target_profit', self.target_profit),
                        volatility_window=params.get('volatility_window', self.volatility_window),
                        min_label_quality=params.get('min_label_quality', self.min_label_quality),
                        min_predictability=params.get('min_predictability', self.min_predictability)
                    )

                    # Simple scoring based on data quality metrics
                    quality = create_data_quality_report(data)
                    score = quality.get('quality_metrics', {}).get('numeric_columns', 0) * 0.1
                    score += (1 - quality.get('quality_metrics', {}).get('missing_percentage', 100)) * 0.01

                    if score > best_score:
                        best_score = score
                        best_config = config
                        tprint_info(f"📈 New best score: {best_score:.3f} (trial {i+1})")

                    successful_trials += 1

                except Exception as e:
                    tprint_warning(f"⚠️ Error evaluating config {params}: {e}")
                    continue

            if best_config:
                tprint_success(f"✅ Grid search completed: {successful_trials}/{max_trials} successful trials, best score: {best_score:.3f}")
                return best_config
            else:
                tprint_warning("⚠️ No valid configurations found, returning original config")
                return self

        except Exception as e:
            tprint_error(f"❌ Grid search optimization failed: {e}")
            tprint_warning("⚠️ Returning original configuration due to optimization failure")
            return self

class AnalystProfitLabeler:
    """
    Analyst Profit Labeler - Consolidated opportunity labeling for Analyst models.

    This class wraps the VolatilityAwareMultiHorizonLabeler with Analyst-specific
    configurations and provides consolidated opportunity detection. Instead of creating
    multiple flags for the same price movement across different horizons and targets,
    it generates a single opportunity flag per profitable price move.

    Key Features:
    - Consolidated labeling (single flag per opportunity)
    - Multi-horizon analysis (60m, 120m, 240m, 360m)
    - Multi-target evaluation (0.5%, 0.7%, 1.0%, 1.5%, 2.0%, 2.5%)
    - Reduces noise from overlapping opportunity detection
    """

    def __init__(self, config: Optional[AnalystProfitLabelerConfig] = None):
        """Initialize the Analyst profit labeler."""
        self.config = config or AnalystProfitLabelerConfig()
        self.logger = system_logger.getChild('AnalystProfitLabeler')

        # Initialize matrix operations for enhanced data processing
        try:
            self.matrix_ops = get_unified_matrix_operations()
            tprint_info(f"🧮 Matrix operations initialized: {self.matrix_ops.__class__.__name__}")
        except Exception as e:
            tprint_error(f"❌ Failed to initialize matrix operations: {e}")
            raise RuntimeError(f"Matrix operations initialization failed: {e}") from e

        # Initialize M1 optimizations if available
        try:
            self.m1_integration = integrate_with_m1_optimizers()
            if self.m1_integration.get('success', False):
                tprint_info(f"🧠 M1 optimizations initialized: GPU={'✅' if self.m1_integration.get('gpu_manager') else '❌'}, Memory={'✅' if self.m1_integration.get('memory_optimizer') else '❌'}")
            else:
                tprint_warning("⚠️ M1 optimizations not available or failed to initialize")
        except Exception as e:
            tprint_warning(f"⚠️ Failed to initialize M1 optimizations: {e}")
            self.m1_integration = {'success': False}

        if not VOLATILITY_LABELER_AVAILABLE:
            raise RuntimeError(
                "VolatilityAwareMultiHorizonLabeler is not available. "
                "Please ensure the profit_labeling module is properly installed."
            )

        # Initialize advanced filters if available and enabled
        self.advanced_filters = None
        tprint_info(f"🔍 Advanced filters initialization: enabled={self.config.enable_advanced_filters}, available={ADVANCED_FILTERS_AVAILABLE}")
        if self.config.enable_advanced_filters and ADVANCED_FILTERS_AVAILABLE:
            try:
                filters_config = self.config.advanced_filters_config or AdvancedFiltersConfig()
                tprint_info(f"🔍 Creating AdvancedFilters15m with config: {filters_config}")
                self.advanced_filters = AdvancedFilters15m(filters_config)
                tprint_info("🔍 Advanced 15m filters initialized")
            except Exception as e:
                tprint_error(f"❌ Failed to initialize advanced filters: {e}")
                import traceback
                tprint_error(f"❌ Traceback: {traceback.format_exc()}")
                self.advanced_filters = None
                tprint_warning("⚠️ Continuing without advanced filters due to initialization failure")
        elif self.config.enable_advanced_filters and not ADVANCED_FILTERS_AVAILABLE:
            tprint_warning("⚠️ Advanced filters requested but not available")
        else:
            tprint_info("🔍 Advanced filters disabled or not available")

        # Create the underlying labeler with Analyst-specific config
        self.labeler = self._create_labeler()

        # Set up output directory for reports
        self.output_dir = Path('artifacts') / 'analyst_reports'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        tprint_success(f"✅ AnalystProfitLabeler initialized (consolidated mode: {len(self.config.horizons)} horizons × 1 target ({self.config.target_profit}%), matrix_ops: {type(self.matrix_ops).__name__})")

    def cleanup(self) -> None:
        """Clean up resources and optimize memory."""
        try:
            # Optimize memory usage
            memory_info = optimize_memory()
            if memory_info.get('success', False):
                tprint_info(f"🧠 Memory optimized: {memory_info.get('objects_collected', 0)} objects collected")

            # Clean up M1 optimizers if available
            from src.utils.common_operations import cleanup_m1_optimizers
            cleanup_m1_optimizers()

            tprint_success("✅ AnalystProfitLabeler cleanup completed")
        except Exception as e:
            tprint_warning(f"⚠️ Error during cleanup: {e}")

    def _create_labeler(self) -> Any:
        """Create and configure the VolatilityAwareMultiHorizonLabeler for Analyst."""
        tprint_info("🔧 Creating VolatilityAwareMultiHorizonLabeler for Analyst...")

        try:
            # Create Analyst-specific configuration
            tprint_info("📋 Initializing labeler configuration...")
            labeler_config = VolatilityAwareConfig()

            # Set label definition type to Analyst
            labeler_config.label_definition_type = LabelDefinitionType.ANALYST
            labeler_config.enable_enhanced_labels = True
            tprint_info("✅ Set label definition type to ANALYST with enhanced labels")

            # Configure bar construction to use TIME bars (we're working with OHLCV data)
            # TIME bars with bar_size = timeframe period will pass through the data as-is
            tprint_info("📊 Configuring bar construction for OHLCV data...")
            from src.training.steps.pre_training.profit_labeling.bar_construction import BarType
            labeler_config.bar_construction.bar_type = BarType.TIME
            labeler_config.bar_construction.bar_size = float(self.config.base_period_minutes)  # 15 minutes for 15m data
            labeler_config.bar_construction.min_bars_required = 10  # Lower threshold for OHLCV data
            tprint_info(f"✅ Bar construction: TIME bars, {self.config.base_period_minutes}min period, min 10 bars")

            # Configure noise gating to be less aggressive for OHLCV data
            # OHLCV data is already aggregated, so noise is less of an issue
            labeler_config.noise_gating.enabled = False  # Disable noise gating for OHLCV data
            tprint_info("✅ Disabled noise gating for OHLCV data")

            # Configure timeframe and horizons
            tprint_info(f"⏰ Configuring timeframe and horizons: {self.config.timeframe}, {self.config.horizons}")
            labeler_config.timeframe = self.config.timeframe
            # Note: Single-horizon labeler doesn't use multi_target config
            # Horizons will be handled by the adapter loop in generate_labels
            tprint_info(f"✅ Single-target: {len(self.config.horizons)} horizons, 1 target ({self.config.target_profit}%)")

            # Configure volatility settings
            tprint_info(f"📈 Configuring volatility: enabled={self.config.use_volatility_normalization}, window={self.config.volatility_window}")
            labeler_config.volatility.enabled = self.config.use_volatility_normalization
            labeler_config.volatility.window = self.config.volatility_window
            # Ensure threshold scaling never drops below base target for Analyst
            if hasattr(labeler_config, 'multi_target'):
                labeler_config.multi_target.min_threshold_multiplier = 1.0
                labeler_config.multi_target.max_threshold_multiplier = 2.0
            # Sensitivity: higher -> more scaling in volatile regimes
            if hasattr(labeler_config, 'volatility'):
                labeler_config.volatility.sensitivity = 1.2

            # Enable rate control: cap opportunities per day (data-driven calibration)
            if hasattr(labeler_config, 'rate_control'):
                labeler_config.rate_control.enabled = True
                labeler_config.rate_control.max_ops_per_day = 8

            # Configure quality scoring - tighten thresholds to prioritize higher-quality labels
            # Quality filtering is now enabled to avoid extremely noisy opportunities
            tprint_info("🎯 Configuring quality scoring (tighter thresholds)...")
            labeler_config.enable_quality_scoring = True
            labeler_config.quality_scoring.min_quality_threshold = 0.4
            labeler_config.quality_scoring.min_predictability = 0.4
            tprint_info("✅ Quality scoring: enabled strict filtering, thresholds=0.4")

            # Configure regime adaptation
            tprint_info(f"🔄 Configuring regime adaptation: enabled={self.config.enable_regime_adaptation}")
            labeler_config.regime_config.enabled = self.config.enable_regime_adaptation

            # Configure optimal entry point detection
            if hasattr(labeler_config, 'optimal_entry_detection'):
                tprint_info("🎯 Configuring optimal entry point detection...")
                labeler_config.optimal_entry_detection.enabled = self.config.enable_optimal_entry_detection
                labeler_config.optimal_entry_detection.entry_threshold = self.config.entry_threshold
                labeler_config.optimal_entry_detection.find_highest_gap_entry = self.config.find_highest_gap_entry
                labeler_config.optimal_entry_detection.entry_point_strategy = self.config.entry_point_strategy
                labeler_config.optimal_entry_detection.horizons = self.config.horizons
                labeler_config.optimal_entry_detection.target_profits = [self.config.target_profit]
                labeler_config.optimal_entry_detection.multi_size_thresholds = {'base': self.config.target_profit}
                labeler_config.optimal_entry_detection.max_windows = self.config.max_windows
                tprint_info(f"✅ Entry detection: enabled={self.config.enable_optimal_entry_detection}, threshold={self.config.entry_threshold}")
            else:
                tprint_warning("⚠️ Optimal entry detection not available in labeler config")

            # Store profit targets in labeler config for volatility-based labeling
            if not hasattr(labeler_config, 'analyst_profit_targets'):
                labeler_config.analyst_profit_targets = [self.config.target_profit]

            # Apply custom parameters
            if self.config.custom_params:
                tprint_info(f"🔧 Applying {len(self.config.custom_params)} custom parameters...")
                applied_count = 0
                for key, value in self.config.custom_params.items():
                    if hasattr(labeler_config, key):
                        setattr(labeler_config, key, value)
                        applied_count += 1
                tprint_success(f"✅ Applied {applied_count} custom parameters")

            # Create the VolatilityAwareMultiHorizonLabeler with our configuration
            tprint_info("🏗️ Creating VolatilityAwareMultiHorizonLabeler instance...")
            labeler = VolatilityAwareMultiHorizonLabeler(labeler_config)
            tprint_success("✅ VolatilityAwareMultiHorizonLabeler created successfully")
            return labeler

        except Exception as e:
            tprint_error(f"❌ Failed to create labeler: {e}")
            raise

    def _condense_labels_single_per_window(
        self,
        labels_bool: pd.Series,
        prices: pd.Series,
        window_minutes: int,
        base_period_minutes: int,
        long_only: bool = True
    ) -> pd.Series:
        """Reduce multiple signals to a single label per fixed time window, choosing local extrema.

        - If long_only=True: pick the local minimum close within each window (best entry).
        - If short allowed: pick min for long signals and max for short signals when a mixed window occurs.
        """
        try:
            if len(labels_bool) == 0:
                return labels_bool

            # Ensure alignment
            labels_aligned, prices_aligned = labels_bool.align(prices, join='inner')
            values = labels_aligned.astype(bool).values
            close_vals = prices_aligned.values

            bars_per_window = max(1, int((window_minutes + base_period_minutes - 1) // base_period_minutes))

            out = pd.Series(False, index=labels_aligned.index)
            signal_idx = np.flatnonzero(values)
            i = 0
            n = len(values)

            while i < len(signal_idx):
                s = signal_idx[i]
                window_end = min(n - 1, s + bars_per_window - 1)

                # Define window slice
                sl = slice(s, window_end + 1)

                # Choose extremum index within the window
                if long_only:
                    local_pos = int(np.argmin(close_vals[sl]))
                else:
                    # If short allowed, default to long extremum; extending to direction-aware is trivial
                    local_pos = int(np.argmin(close_vals[sl]))

                chosen = s + local_pos
                out.iloc[chosen] = True

                # Skip all signals covered by this window
                j = i + 1
                while j < len(signal_idx) and signal_idx[j] <= window_end:
                    j += 1
                i = j

            reduced = int(values.sum())
            final = int(out.sum())
            if reduced:
                tprint_info(
                    f"🔧 Applied single-label per {window_minutes}m window: {reduced} -> {final} ({(reduced-final)/reduced:.1%} reduction)"
                )
            else:
                tprint_info(f"🔧 Single-label window applied: 0 signals")

            # Reindex to original index if alignment trimmed anything
            out_full = pd.Series(False, index=labels_bool.index)
            out_full.loc[out.index] = out
            return out_full
        except Exception as e:
            tprint_warning(f"⚠️ Failed to condense labels per window: {e}")
            return labels_bool

    def generate_labels(
        self,
        data: pd.DataFrame,
        regime_assignments: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> LabelingResult:
        """
        Generate consolidated Analyst profit labels for the input data.

        This method checks across all configured horizons and profit targets to determine
        if each data point represents a profitable trading opportunity. Instead of creating
        multiple flags for the same price movement across different horizons/targets, it
        creates a single consolidated opportunity flag per profitable price move.

        Args:
            data: Input market data (OHLCV format)
            regime_assignments: Optional regime assignments for regime-aware labeling
            **kwargs: Additional parameters for the labeler

        Returns:
            LabelingResult with consolidated opportunity labels (single flag per profitable move),
            metadata including horizons/targets checked, and quality metrics
        """
        tprint_info(f"📈 Generating Analyst profit labels for {len(data)} samples...")

        try:
            # Initialize quality scores collection
            consolidated_quality_scores = {}
            
            # Validate input data quality using both common operations and utilities
            data_quality = create_data_quality_report(data)
            detailed_quality = analyze_nan_values_detailed(data)

            if data_quality.get('quality_metrics', {}).get('missing_percentage', 0) > 50:
                tprint_warning(f"⚠️ High missing data percentage: {data_quality['quality_metrics']['missing_percentage']:.2f}%")

            # Log detailed NaN analysis if issues found
            if detailed_quality.get('total_nans', 0) > 0:
                nan_report = format_nan_analysis_report(detailed_quality, "  ")
                tprint_info(f"📊 NaN Analysis:\n{nan_report}")

            # Validate required columns for OHLCV data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not validate_dataframe_columns(data, required_columns):
                missing_cols = set(required_columns) - set(data.columns)
                raise ValueError(f"Missing required OHLCV columns: {missing_cols}")

            # Validate data types and convert if necessary using safe operations
            for col in ['open', 'high', 'low', 'close']:
                if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                    tprint_warning(f"⚠️ Converting {col} to numeric")
                    data = safe_dataframe_operation(data, pd.to_numeric, col, errors='coerce')

            # Optimize data using matrix operations for better performance
            tprint_info(f"🧮 Optimizing data with matrix operations ({data.shape})")
            original_shape = data.shape
            try:
                optimized_data = optimize_dataframe(data)
                if optimized_data is not data:
                    data = optimized_data
                    tprint_success(f"✅ Data optimized: {original_shape} → {data.shape}")
                else:
                    tprint_info("ℹ️ Data optimization: no changes needed")
            except Exception as e:
                tprint_warning(f"⚠️ Data optimization failed, continuing with original data: {e}")
                # Continue with original data if optimization fails

            # Note: regime_assignments are not currently used by the underlying labeler
            # but are kept in the API for future enhancement
            if regime_assignments is not None:
                tprint_info(f"📊 Regime assignments provided but not yet integrated into labeling logic")

            # Apply advanced filters if enabled
            filter_result = None
            tprint_info(f"🔍 Advanced filters status: enabled={self.config.enable_advanced_filters}, available={ADVANCED_FILTERS_AVAILABLE}, instance={self.advanced_filters is not None}")
            if self.advanced_filters is not None:
                tprint_info("🔍 Applying advanced 15m filters before labeling")
                filter_result = self.advanced_filters.apply_filters(data)
            else:
                tprint_warning("⚠️ Advanced filters not available - skipping filter application")

            # Log filter results if filters were applied
            if filter_result is not None:
                tprint_info(f"   → Filter eligibility: {filter_result.eligibility_ratio:.1%} ({filter_result.n_eligible_samples}/{filter_result.n_total_samples})")

                # Apply filter mask to data (optional - can be used to pre-filter data)
                if filter_result.eligibility_ratio < 0.5:
                    tprint_warning(f"⚠️ Low filter eligibility: {filter_result.eligibility_ratio:.1%}")
                else:
                    tprint_success(f"✅ Filter eligibility good: {filter_result.eligibility_ratio:.1%}")

            # Use memory optimization context for label generation
            with memory_checkpoint("analyst_label_generation"):
                # Generate consolidated opportunity labels (single flag per profitable price move)
                tprint_info(f"🔄 Generating consolidated opportunity labels across {len(self.config.horizons)} horizons with {self.config.target_profit}% target...")

                # Check if any horizon/target combination shows a profitable opportunity
                consolidated_labels = pd.Series([False] * len(data), index=data.index, name='opportunity', dtype=bool)

                # Collect anchor times from Analyst labeler for Tactician (WHEN) integration
                anchor_times: List[pd.Timestamp] = []

                for horizon_idx, horizon_minutes in enumerate(self.config.horizons):
                    tprint_info(f"📈 Checking horizon {horizon_idx + 1}/{len(self.config.horizons)}: {horizon_minutes}min")

                    # Calculate lookahead period for this horizon
                    lookahead_bars = horizon_minutes // self.config.base_period_minutes
                    
                    # Create a copy of the labeler config to avoid mutating the original
                    horizon_config = self.labeler.config
                    original_lookahead = getattr(horizon_config, 'lookahead_periods', None)
                    horizon_config.lookahead_periods = lookahead_bars

                    # Generate labels for this horizon with profit targets
                    profit_targets = getattr(self.labeler.config, 'analyst_profit_targets', None)
                    # Use the full generate_labels method to get quality scores
                    horizon_result = self.labeler.generate_labels(
                        data=data,
                        price_column='close',
                        volatility_column=None,  # Will be calculated automatically
                        profit_targets=profit_targets
                    )
                    
                    # Restore original lookahead_periods to avoid side effects
                    if original_lookahead is not None:
                        horizon_config.lookahead_periods = original_lookahead

                    if horizon_result.success and horizon_result.labels is not None:
                        # Convert to Series if needed
                        if isinstance(horizon_result.labels, pd.DataFrame):
                            # For multi-target results, check if any target shows opportunity (value > 0)
                            horizon_series = horizon_result.labels.max(axis=1)  # Max across targets
                        else:
                            horizon_series = horizon_result.labels

                        # Mark as opportunity if this horizon shows profitability
                        if isinstance(horizon_series, pd.Series):
                            consolidated_labels = consolidated_labels | (horizon_series > 0).astype(bool)
                        
                        # Collect anchor times from metadata for Tactician entry labels
                        try:
                            win_meta = horizon_result.metadata.get('opportunity_windows', []) if hasattr(horizon_result, 'metadata') else []
                            for w in win_meta:
                                ts = w.get('anchor')
                                if ts is not None:
                                    anchor_times.append(pd.Timestamp(ts))
                        except Exception:
                            pass

                        # Collect quality scores from this horizon
                        if hasattr(horizon_result, 'quality_scores') and horizon_result.quality_scores:
                            for target_name, quality in horizon_result.quality_scores.items():
                                if target_name not in consolidated_quality_scores:
                                    consolidated_quality_scores[target_name] = []
                                consolidated_quality_scores[target_name].append(quality)

                        tprint_info(f"✅ Found {(horizon_series > 0).sum()} opportunities in {horizon_minutes}min horizon")
                    else:
                        tprint_warning(f"⚠️ Failed to generate labels for {horizon_minutes}min horizon")

                # Enforce a single label per 60m window and place at local extremum
                consolidated_labels = self._condense_labels_single_per_window(
                    labels_bool=consolidated_labels,
                    prices=data['close'] if 'close' in data.columns else data.iloc[:, 0],
                    window_minutes=60,
                    base_period_minutes=self.config.base_period_minutes,
                    long_only=self.config.enable_long_positions and not self.config.enable_short_positions
                )

                # Convert boolean series to integer (0 or 1)
                consolidated_labels = consolidated_labels.astype(int)

                # Build Tactician entry label series (1 at Analyst anchor bars)
                tactician_entry_labels = pd.Series(0, index=data.index, dtype=int)
                if anchor_times:
                    anchor_times_unique = sorted(set([ts for ts in anchor_times if ts in tactician_entry_labels.index]))
                    tactician_entry_labels.loc[anchor_times_unique] = 1

                # Track detailed opportunity statistics
                total_opportunities = consolidated_labels.sum()
                total_samples = len(consolidated_labels)
                opportunity_rate = total_opportunities / total_samples

                # Calculate days for per-day statistics
                days_in_data = total_samples / (24 * 4)  # 96 intervals per day for 15m data

                # Filter statistics (before any filtering is applied)
                filtered_out_opportunities = 0
                accepted_opportunities = total_opportunities

                tprint_info(f"🎯 DETAILED OPPORTUNITY STATISTICS (BEFORE QUALITY FILTERING):")
                tprint_info(f"   ┌─────────────────────────────────────────────────────┐")
                tprint_info(f"   │ Total opportunities found: {total_opportunities:,}")
                tprint_info(f"   │ Total samples processed: {total_samples:,}")
                tprint_info(f"   │ Opportunity rate: {opportunity_rate:.1%}")
                tprint_info(f"   │ Days represented: {days_in_data:.1f}")
                tprint_info(f"   │ Opportunities per day: {total_opportunities / days_in_data:.1f}")
                tprint_info(f"   └─────────────────────────────────────────────────────┘")
                tprint_info(f"   📊 Raw opportunity count: {total_opportunities:,} (before quality thresholds)")
                tprint_info(f"   📊 Opportunities per day: {total_opportunities / days_in_data:.1f} (before quality thresholds)")

                # Process consolidated quality scores
                final_quality_scores = {}
                if consolidated_quality_scores:
                    for target_name, quality_list in consolidated_quality_scores.items():
                        if quality_list:
                            # Average quality scores across horizons for this target
                            avg_quality = sum(getattr(q, 'overall_quality', 0) for q in quality_list) / len(quality_list)
                            avg_predictability = sum(getattr(q, 'predictability', 0) for q in quality_list) / len(quality_list)
                            avg_stability = sum(getattr(q, 'stability', 0) for q in quality_list) / len(quality_list)
                            avg_balance = sum(getattr(q, 'balance', 0) for q in quality_list) / len(quality_list)
                            
                            # Create a quality score object
                            class QualityScore:
                                def __init__(self, overall_quality, predictability, stability, balance):
                                    self.overall_quality = overall_quality
                                    self.predictability = predictability
                                    self.stability = stability
                                    self.balance = balance
                            
                            final_quality_scores[target_name] = QualityScore(
                                avg_quality, avg_predictability, avg_stability, avg_balance
                            )

                result = LabelingResult(
                    labels=consolidated_labels.astype('int8').to_frame('opportunity'),
                    metadata={
                        'n_horizons': len(self.config.horizons),
                        'horizons': self.config.horizons,
                        'target_profits': self.config.target_profits,
                        'base_period_minutes': self.config.base_period_minutes,
                        'consolidation_method': 'single_flag_per_profitable_move',
                        'total_opportunities': total_opportunities,
                        'filtered_out_opportunities': filtered_out_opportunities,
                        'accepted_opportunities': accepted_opportunities,
                        'opportunity_rate': opportunity_rate,
                        'days_in_data': days_in_data,
                        'opportunities_per_day': total_opportunities / days_in_data,
                        'quality_scores': final_quality_scores,
                        # Expose anchor timestamps and a ready-to-use Tactician entry label series
                        'tactician_entry_labels': tactician_entry_labels,
                        'opportunity_windows': anchor_times_unique if anchor_times else []
                    },
                    success=True,
                    quality_scores=final_quality_scores
                )

                # Apply filter mask to results if filters were used
                if filter_result is not None and hasattr(result, 'labels') and result.labels is not None:
                    if isinstance(result.labels, pd.DataFrame) and len(result.labels) > 0:
                        # Track opportunities before filtering
                        opportunities_before_filtering = result.labels['opportunity'].sum()
                        total_before_filtering = len(result.labels)

                        # Apply eligibility mask to labels
                        # DIAGNOSTIC: Check eligibility mask before applying
                        mask = filter_result.eligibility_mask
                        tprint_info(f"🔍 ELIGIBILITY MASK DIAGNOSTICS:")
                        tprint_info(f"   │ mask dtype: {mask.dtype}")
                        tprint_info(f"   │ mask length == data length: {len(mask) == len(data)}")
                        tprint_info(f"   │ mask index equals data index: {mask.index.equals(data.index)}")
                        tprint_info(f"   │ mask true ratio: {float(mask.mean()) if mask.dtype == bool and len(mask) else 'N/A'}")
                        
                        # Inspect a few mismatched timestamps if any
                        bad_idx = mask.index.symmetric_difference(data.index)
                        tprint_info(f"   │ index symmetric diff size: {len(bad_idx)}")
                        if len(bad_idx) > 0:
                            tprint_info(f"   │ Sample bad indices: {list(bad_idx[:5])}")
                        
                        tprint_info(f"   │ mask head: {mask.head()}")
                        tprint_info(f"   │ data head index: {data.head(3).index.tolist()}")
                        
                        # Ensure eligibility mask alignment with labels
                        if len(filter_result.eligibility_mask) != len(result.labels):
                            tprint_warning(f"⚠️ Eligibility mask length ({len(filter_result.eligibility_mask)}) doesn't match labels length ({len(result.labels)})")
                            # Reindex mask to match labels index
                            mask_series = pd.Series(filter_result.eligibility_mask, index=result.labels.index)
                            result.labels = result.labels[mask_series]
                        else:
                            result.labels = result.labels[filter_result.eligibility_mask]
                        
                        # Update LabelingResult counts to match filtered labels
                        result.n_samples = len(result.labels)
                        result.n_targets = len(result.labels.columns) if hasattr(result.labels, 'columns') else 1
                        result.n_horizons = len(self.config.horizons)

                        # Track opportunities after filtering
                        opportunities_after_filtering = result.labels['opportunity'].sum()
                        filtered_out_opportunities = opportunities_before_filtering - opportunities_after_filtering
                        accepted_opportunities = opportunities_after_filtering

                        # Apply eligibility mask to confidence scores if available
                        if hasattr(result, 'confidence_scores') and result.confidence_scores is not None:
                            if isinstance(result.confidence_scores, pd.DataFrame):
                                result.confidence_scores = result.confidence_scores[filter_result.eligibility_mask]

                        # Apply eligibility mask to eligibility masks if available
                        if hasattr(result, 'eligibility_masks') and result.eligibility_masks is not None:
                            if isinstance(result.eligibility_masks, pd.DataFrame):
                                result.eligibility_masks = result.eligibility_masks[filter_result.eligibility_mask]

                        # Update sample counts
                        result.n_samples = len(result.labels) if hasattr(result.labels, '__len__') else result.n_samples

                        # Update metadata with filtering statistics
                        if 'metadata' in result.__dict__:
                            result.metadata.update({
                                'opportunities_before_filtering': opportunities_before_filtering,
                                'opportunities_after_filtering': opportunities_after_filtering,
                                'filtered_out_opportunities': filtered_out_opportunities,
                                'accepted_opportunities': accepted_opportunities,
                                'filter_eligibility_ratio': filter_result.eligibility_ratio,
                                'n_eligible_samples': filter_result.n_eligible_samples,
                                'n_total_samples': filter_result.n_total_samples
                            })

                        # Log filtering results
                        tprint_info("🔍 FILTERING RESULTS:")
                        tprint_info("   ┌─────────────────────────────────────────────────────┐")
                        tprint_info(f"   │ Before filtering: {opportunities_before_filtering:,}/{total_before_filtering:,} ({opportunities_before_filtering/total_before_filtering:.1%})")
                        tprint_info(f"   │ After filtering: {opportunities_after_filtering:,}/{result.n_samples:,} ({opportunities_after_filtering/result.n_samples:.1%})")
                        tprint_info(f"   │ Filtered out: {filtered_out_opportunities:,} opportunities ({filtered_out_opportunities/opportunities_before_filtering:.1%})")
                        tprint_info(f"   │ Filter eligibility: {filter_result.eligibility_ratio:.1%}")
                        tprint_info("   └─────────────────────────────────────────────────────┘")
                    else:
                        tprint_warning("⚠️ Filter mask not applied - result labels is not a DataFrame or is empty")

            # Validate minimum sample count for training
            self._validate_labeling_result(result)

            # Log memory usage and data quality with matrix operations info
            try:
                optimize_memory()  # Optimize memory without storing result
                data_info = get_dataframe_info(data)
                hardware_report = get_hardware_performance_report()
                tprint_info(f"📊 Data info: {data_info['shape']} shape, {format_bytes(data_info['memory_usage'])} memory")
                tprint_info(f"🔧 Hardware performance: {hardware_report.get('cpu_cores', 'N/A')} cores, GPU: {hardware_report.get('gpu_available', 'N/A')}")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to get performance metrics: {e}")
                # Continue without performance metrics

            tprint_success(
                f"✅ Analyst labels generated: {result.n_samples} samples, "
                f"{result.n_targets} targets, {result.n_horizons} horizons"
            )

            # Generate comprehensive outcome report
            self._generate_comprehensive_report(result, data)

            return result

        except Exception as e:
            tprint_error(f"❌ Failed to generate Analyst labels: {e}")
            raise

    def _validate_labeling_result(self, result: LabelingResult) -> None:
        """Validate that labeling produced sufficient samples for training."""
        tprint_info(f"🔍 Validating labeling result: {result.n_samples} samples, {result.n_targets} targets")

        MIN_SAMPLES_PER_TARGET = 50  # Minimum samples needed per target for reliable training
        MIN_TOTAL_SAMPLES = 200     # Absolute minimum total samples

        if result.n_samples < MIN_TOTAL_SAMPLES:
            error_msg = (
                f"Insufficient samples for training: got {result.n_samples}, need at least {MIN_TOTAL_SAMPLES}. "
                f"Consider adjusting labeling parameters (horizons, thresholds, or data timeframe) to generate more labels."
            )
            tprint_error(f"❌ {error_msg}")
            raise ValueError(error_msg)

        tprint_success(f"✅ Sample count validation passed: {result.n_samples} >= {MIN_TOTAL_SAMPLES}")

        if result.n_targets is not None and result.n_targets > 0:
            samples_per_target = result.n_samples / result.n_targets
            tprint_info(f"📊 Samples per target: {samples_per_target:.1f} (minimum recommended: {MIN_SAMPLES_PER_TARGET})")

            if samples_per_target < MIN_SAMPLES_PER_TARGET:
                warning_msg = (
                    f"Low samples per target: {samples_per_target:.1f} per target, "
                    f"recommended minimum is {MIN_SAMPLES_PER_TARGET}. "
                    "Model training may be unreliable with insufficient samples per target."
                )
                tprint_warning(f"⚠️ {warning_msg}")
                warnings.warn(
                    warning_msg,
                    UserWarning,
                    stacklevel=2
                )
            else:
                tprint_success(f"✅ Samples per target validation passed: {samples_per_target:.1f} >= {MIN_SAMPLES_PER_TARGET}")

        tprint_success("✅ Labeling result validation completed successfully")

    def _generate_comprehensive_report(self, result: LabelingResult, original_data: pd.DataFrame) -> None:
        """Generate comprehensive analysis report of the labeling outcome."""
        tprint_info("📊 Generating comprehensive labeling outcome report...")

        try:
            # Create report data structure
            report = {
                'execution_summary': self._get_execution_summary(result, original_data),
                'data_quality_analysis': self._get_data_quality_analysis(original_data),
                'labeling_performance': self._get_labeling_performance_metrics(result),
                'horizon_analysis': self._get_horizon_analysis(result),
                'target_analysis': self._get_target_analysis(result),
                'quality_metrics': self._get_quality_metrics_analysis(result),
                'recommendations': self._get_recommendations(result)
            }

            # Print comprehensive report (skip if there are issues)
            try:
                self._print_comprehensive_report(report)
            except Exception as e:
                tprint_warning(f"⚠️ Failed to print comprehensive report: {e}")
                # Continue without the printed report

            # Save detailed report to JSON file
            self._save_detailed_report(report)

        except Exception as e:
            tprint_warning(f"⚠️ Failed to generate comprehensive report: {e}")
            # Continue without the report

    def _get_execution_summary(self, result: LabelingResult, original_data: pd.DataFrame) -> Dict[str, Any]:
        """Get execution summary metrics."""
        return {
            'input_data_shape': original_data.shape,
            'input_data_memory_mb': round(original_data.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            'input_date_range': {
                'start': str(original_data.index.min()),
                'end': str(original_data.index.max()),
                'duration_days': round((original_data.index.max() - original_data.index.min()).total_seconds() / 86400, 1)
            },
            'output_labels_shape': result.labels.shape if hasattr(result.labels, 'shape') else (0, 0),
            'output_labels_memory_mb': round(result.labels.memory_usage(deep=True).sum() / 1024 / 1024, 2) if hasattr(result.labels, 'memory_usage') else 0,
            'processing_time_seconds': result.processing_time or 0.0,
            'success_rate': 100.0 if result.success else 0.0
        }

    def _get_data_quality_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality metrics."""
        return {
            'missing_values_total': int(data.isna().sum().sum()),
            'missing_values_percentage': round(data.isna().sum().sum() / (data.shape[0] * data.shape[1]) * 100, 2),
            'duplicate_timestamps': int(data.index.duplicated().sum()),
            'data_completeness': round((1 - data.isna().sum().sum() / (data.shape[0] * data.shape[1])) * 100, 2),
            'price_volatility': round(data['close'].pct_change().std() * 100, 2) if 'close' in data.columns else None,
            'volume_analysis': {
                'mean_volume': int(data['volume'].mean()) if 'volume' in data.columns else None,
                'volume_std': int(data['volume'].std()) if 'volume' in data.columns else None,
                'zero_volume_count': int((data['volume'] == 0).sum()) if 'volume' in data.columns else None
            }
        }

    def _get_labeling_performance_metrics(self, result: LabelingResult) -> Dict[str, Any]:
        """Get performance metrics for the labeling process."""
        return {
            'total_samples_processed': result.n_samples,
            'total_targets_generated': result.n_targets or 0,
            'total_horizons_processed': result.n_horizons,
            'labels_per_sample': round(result.n_targets / max(result.n_samples, 1), 2) if result.n_targets else 0,
            'samples_per_horizon': round(result.n_samples / max(result.n_horizons, 1), 2),
            'processing_efficiency': round(result.n_samples / max(result.processing_time, 0.001), 2) if result.processing_time else 0,
            'memory_efficiency_mb_per_sample': round(getattr(result.labels, 'memory_usage', lambda: [0])().sum() / 1024 / 1024 / max(result.n_samples, 1), 4) if hasattr(result.labels, 'memory_usage') and result.n_samples > 0 else 0
        }

    def _get_horizon_analysis(self, result: LabelingResult) -> Dict[str, Any]:
        """Analyze results by horizon."""
        if not hasattr(result.labels, 'columns') or result.labels is None:
            return {'error': 'No horizon data available'}

        # Check if this is consolidated mode (single 'opportunity' column)
        if 'opportunity' in result.labels.columns and len(result.labels.columns) == 1:
            # Consolidated mode - create summary for all horizons
            opportunity_data = result.labels['opportunity']
            total_opportunities = int((opportunity_data == 1).sum()) if pd.api.types.is_numeric_dtype(opportunity_data) else 0
            total_labels = len(opportunity_data)
            
            horizon_analysis = {}
            for horizon_minutes in self.config.horizons:
                horizon_analysis[f"{horizon_minutes}min"] = {
                    'horizon_minutes': horizon_minutes,
                    'horizon_bars': horizon_minutes // self.config.base_period_minutes,
                    'total_labels': total_labels,
                    'positive_labels': total_opportunities,
                    'negative_labels': total_labels - total_opportunities,
                    'neutral_labels': 0,
                    'nan_labels': int(opportunity_data.isna().sum()),
                    'label_balance': round(abs(total_opportunities - (total_labels - total_opportunities)) / total_labels, 3) if total_labels > 0 else 0,
                    'positive_rate': round(total_opportunities / total_labels * 100, 2) if total_labels > 0 else 0,
                    'expected_labels': 2  # Expected for consolidated mode
                }
            return horizon_analysis

        # Original per-horizon analysis
        horizon_analysis = {}
        for col in result.labels.columns:
            if col.startswith('h'):
                horizon_minutes = int(col[1:])  # Extract minutes from 'h60' format
                horizon_data = result.labels[col]

                horizon_analysis[f"{horizon_minutes}min"] = {
                    'horizon_minutes': horizon_minutes,
                    'horizon_bars': horizon_minutes // self.config.base_period_minutes,
                    'total_labels': len(horizon_data),
                    'positive_labels': int((horizon_data == 1).sum()) if pd.api.types.is_numeric_dtype(horizon_data) else 0,
                    'negative_labels': int((horizon_data == 0).sum()) if pd.api.types.is_numeric_dtype(horizon_data) else 0,
                    'neutral_labels': int((horizon_data == -1).sum()) if pd.api.types.is_numeric_dtype(horizon_data) else 0,
                    'nan_labels': int(horizon_data.isna().sum()),
                    'label_balance': round(abs((horizon_data == 1).sum() - (horizon_data == 0).sum()) / len(horizon_data), 3) if pd.api.types.is_numeric_dtype(horizon_data) else 0,
                    'positive_rate': round((horizon_data == 1).sum() / len(horizon_data) * 100, 2) if pd.api.types.is_numeric_dtype(horizon_data) else 0
                }

        return horizon_analysis

    def _get_target_analysis(self, result: LabelingResult) -> Dict[str, Any]:
        """Analyze results by target profit levels."""
        if not hasattr(result.labels, 'columns') or result.labels is None:
            return {'error': 'No target data available'}

        # Check if this is consolidated mode (single 'opportunity' column)
        if 'opportunity' in result.labels.columns and len(result.labels.columns) == 1:
            # Consolidated mode - create target breakdown for configured targets
            opportunity_data = result.labels['opportunity']
            target_analysis = {}
            
            for target_profit in self.config.target_profits:
                target_analysis[f"{target_profit}%"] = {
                    'target_profit_pct': target_profit,
                    'expected_labels': 0,  # In consolidated mode, we don't have per-target breakdown
                    'note': 'Consolidated mode - single opportunity flag across all targets'
                }
            return target_analysis

        # Original per-target analysis
        target_analysis = {}
        for col in result.labels.columns:
            if col.startswith('h'):
                horizon_minutes = int(col[1:])
                target_data = result.labels[col]

                if pd.api.types.is_numeric_dtype(target_data):
                    # Calculate distribution across different label values
                    value_counts = target_data.value_counts()
                    target_analysis[f"{horizon_minutes}min"] = {
                        'horizon_minutes': horizon_minutes,
                        'unique_values': int(target_data.nunique()),
                        'value_distribution': {str(k): int(v) for k, v in value_counts.to_dict().items()},
                        'most_common_label': str(target_data.mode().iloc[0]) if len(target_data.mode()) > 0 else 'N/A',
                        'label_entropy': round(-sum((count/len(target_data)) * (count/len(target_data)) for count in value_counts) / len(value_counts), 3) if len(value_counts) > 0 else 0
                    }

        return target_analysis

    def _get_quality_metrics_analysis(self, result: LabelingResult) -> Dict[str, Any]:
        """Analyze quality metrics if available."""
        if not result.quality_scores:
            return {'status': 'No quality scores available'}

        quality_analysis = {'overall_status': 'Quality metrics available'}
        for target_name, quality in result.quality_scores.items():
            quality_analysis[target_name] = {
                'overall_quality': round(getattr(quality, 'overall_quality', 0), 4),
                'predictability': round(getattr(quality, 'predictability', 0), 4),
                'stability': round(getattr(quality, 'stability', 0), 4),
                'balance': round(getattr(quality, 'balance', 0), 4),
                'quality_passes_threshold': getattr(quality, 'overall_quality', 0) >= self.config.min_label_quality,
                'predictability_passes_threshold': getattr(quality, 'predictability', 0) >= self.config.min_predictability
            }

        return quality_analysis

    def _get_recommendations(self, result: LabelingResult) -> List[str]:
        """Generate recommendations based on the labeling results."""
        recommendations = []

        # Check sample count
        if result.n_samples < 1000:
            recommendations.append("⚠️ Low sample count - consider expanding data range or adjusting labeling parameters")

        # Check target count
        if result.n_targets is not None and result.n_targets < 2:
            recommendations.append("⚠️ Very few targets generated - may indicate overly strict quality thresholds")

        # Check horizon distribution
        if hasattr(result.labels, 'columns'):
            positive_cols = sum(1 for col in result.labels.columns if (result.labels[col] == 1).sum() > 0)
            if positive_cols == 0:
                recommendations.append("❌ No positive labels found - labeling may be too conservative")

        # Check quality scores
        if result.quality_scores:
            avg_quality = sum(getattr(q, 'overall_quality', 0) for q in result.quality_scores.values()) / len(result.quality_scores)
            if avg_quality < 0.5:
                recommendations.append("⚠️ Low average quality scores - consider adjusting quality thresholds")

        if not recommendations:
            recommendations.append("✅ Labeling results look good - all metrics within acceptable ranges")

        return recommendations

    def _print_comprehensive_report(self, report: Dict[str, Any]) -> None:
        """Print the comprehensive report in a readable format."""
        tprint_info("🎯 COMPREHENSIVE LABELING OUTCOME REPORT")
        tprint_info("=" * 60)

        # Execution Summary
        tprint_info("📋 EXECUTION SUMMARY:")
        exec_summary = report['execution_summary']
        tprint_info(f"   📊 Input Data: {exec_summary['input_data_shape']} shape, {exec_summary['input_data_memory_mb']}MB")
        tprint_info(f"   📅 Date Range: {exec_summary['input_date_range']['start']} to {exec_summary['input_date_range']['end']}")
        tprint_info(f"   ⏱️  Duration: {exec_summary['input_date_range']['duration_days']} days")
        tprint_info(f"   📈 Output Labels: {exec_summary['output_labels_shape']} shape, {exec_summary['output_labels_memory_mb']}MB")
        tprint_info(f"   ⚡ Processing Time: {exec_summary['processing_time_seconds']:.2f}s")
        tprint_info(f"   ✅ Success Rate: {exec_summary['success_rate']:.1f}%")

        # Data Quality Analysis
        tprint_info("\n🔍 DATA QUALITY ANALYSIS:")
        data_quality = report['data_quality_analysis']
        tprint_info(f"   📉 Missing Values: {data_quality['missing_values_total']} ({data_quality['missing_values_percentage']}%)")
        tprint_info(f"   ✅ Completeness: {data_quality['data_completeness']:.1f}%")
        tprint_info(f"   🔄 Duplicates: {data_quality['duplicate_timestamps']} timestamp duplicates")

        if data_quality['volume_analysis']['mean_volume']:
            vol_analysis = data_quality['volume_analysis']
            tprint_info(f"   📊 Volume: μ={vol_analysis['mean_volume']:,}, σ={vol_analysis['volume_std']:,}")
            tprint_info(f"   💹 Price Volatility: {data_quality['price_volatility']:.2f}%")

        # Labeling Performance
        tprint_info("\n⚡ LABELING PERFORMANCE:")
        perf_metrics = report['labeling_performance']
        tprint_info(f"   🎯 Samples Processed: {perf_metrics['total_samples_processed']:,}")
        tprint_info(f"   🏆 Targets Generated: {perf_metrics['total_targets_generated']:,}")
        tprint_info(f"   ⏰ Horizons Processed: {perf_metrics['total_horizons_processed']}")
        tprint_info(f"   📊 Labels/Sample: {perf_metrics['labels_per_sample']:.2f}")
        tprint_info(f"   ⚡ Efficiency: {perf_metrics['processing_efficiency']:.0f} samples/sec")

        # Horizon Analysis
        tprint_info("\n🌅 HORIZON ANALYSIS:")
        horizon_analysis = report['horizon_analysis']
        if 'error' not in horizon_analysis:
            for horizon_name, metrics in horizon_analysis.items():
                tprint_info(f"   📈 {horizon_name}:")
                tprint_info(f"      📊 Total: {metrics['total_labels']:,}, Positive: {metrics['positive_labels']:,} ({metrics['positive_rate']:.1f}%)")
                tprint_info(f"      ⚖️  Balance: {metrics['label_balance']:.3f}, NaN: {metrics['nan_labels']:,}")
        else:
            tprint_info(f"   ❌ {horizon_analysis['error']}")

        # Quality Metrics
        tprint_info("\n🎯 QUALITY METRICS:")
        quality_analysis = report['quality_metrics']
        if quality_analysis.get('status') != 'No quality scores available':
            for target_name, metrics in quality_analysis.items():
                if target_name != 'overall_status':
                    tprint_info(f"   📋 {target_name}:")
                    tprint_info(f"      🎯 Quality: {metrics['overall_quality']:.4f} {'✅' if metrics['quality_passes_threshold'] else '❌'}")
                    tprint_info(f"      🔮 Predictability: {metrics['predictability']:.4f} {'✅' if metrics['predictability_passes_threshold'] else '❌'}")
        else:
            tprint_info(f"   ❌ {quality_analysis['status']}")

        # Recommendations
        tprint_info("\n💡 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            tprint_info(f"   {rec}")

        tprint_info("=" * 60)
        tprint_success("✅ Comprehensive report generated successfully")

    def _save_detailed_report(self, report: Dict[str, Any]) -> None:
        """Save detailed report to JSON file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"analyst_labeling_report_{timestamp}.json"
            report_path = self.output_dir / report_filename

            # Ensure output directory exists
            report_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert any non-serializable objects to strings
            serializable_report = self._make_report_serializable(report)

            safe_json_dump(serializable_report, report_path, indent=2)

            tprint_success(f"💾 Detailed report saved to: {report_path}")

        except Exception as e:
            tprint_warning(f"⚠️ Failed to save detailed report: {e}")

    def _make_report_serializable(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Convert report to JSON-serializable format."""
        def convert_value(value):
            if isinstance(value, (int, float, str, bool, type(None))):
                return value
            elif isinstance(value, (list, tuple)):
                return [convert_value(item) for item in value]
            elif isinstance(value, dict):
                return {str(k): convert_value(v) for k, v in value.items()}
            elif hasattr(value, 'isoformat'):  # datetime
                return value.isoformat()
            else:
                return str(value)

        return {k: convert_value(v) for k, v in report.items()}

    def _save_human_readable_outcome(self, outcome_data: Dict[str, Any], json_path: Path) -> None:
        """Save outcome data in human-readable markdown format."""
        try:
            # Create markdown filename from JSON filename
            markdown_filename = json_path.stem.replace('analyst_labeler_outcome', 'analyst_labeler_report') + '.md'
            markdown_path = json_path.parent / markdown_filename

            # Generate markdown content
            markdown_content = self._generate_human_readable_content(outcome_data)

            with open(markdown_path, 'w') as f:
                f.write(markdown_content)

            tprint_success(f"📄 Human-readable report saved: {markdown_path}")

        except Exception as e:
            tprint_warning(f"⚠️ Failed to save human-readable outcome file: {e}")

    def _generate_human_readable_content(self, outcome_data: Dict[str, Any]) -> str:
        """Generate human-readable markdown content from outcome data."""
        content = []

        content.append("# Analyst Profit Labeler - Execution Report")
        content.append("")
        content.append(f"**Execution Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"**Status:** {'✅ SUCCESS' if outcome_data.get('status') == 'success' else '❌ FAILED'}")
        content.append("")

        # Configuration Section
        content.append("## 🔧 Configuration")
        config = outcome_data.get('configuration', {})
        content.append(f"- **Timeframe:** {outcome_data.get('timeframe', 'N/A')}")
        content.append(f"- **Horizons:** {config.get('horizons', [])} minutes")
        content.append(f"- **Target Profit:** {config.get('target_profit', 0.5)}% (volatility-modulated)")
        content.append(f"- **Base Period:** {config.get('base_period_minutes', 'N/A')} minutes")
        content.append(f"- **Volatility Normalization:** {'✅ Enabled' if config.get('use_volatility_normalization') else '❌ Disabled'}")
        content.append("")

        # Results Section
        content.append("## 📊 Results Summary")
        results = outcome_data.get('results', {})
        content.append(f"- **Total Samples:** {results.get('n_samples', 0):,}")
        content.append(f"- **Total Targets:** {results.get('n_targets', 0):,}")
        content.append(f"- **Horizons Processed:** {results.get('n_horizons', 0)}")
        execution_time = outcome_data.get('execution_time', 0)
        content.append(f"- **Processing Time:** {execution_time:.2f} seconds" if execution_time is not None else "- **Processing Time:** N/A")
        content.append("")
        
        # Daily Average Opportunities
        data_quality = outcome_data.get('data_quality', {})
        input_data = data_quality.get('input_data', {})
        duration_days = input_data.get('date_range', {}).get('duration_days', 0)
        total_samples = results.get('n_samples', 0)
        
        if duration_days > 0 and total_samples > 0:
            daily_avg_opportunities = total_samples / duration_days
            content.append("## 📈 Daily Opportunity Analysis")
            content.append(f"- **Data Duration:** {duration_days:.1f} days")
            content.append(f"- **Daily Average Opportunities:** {daily_avg_opportunities:.1f} opportunities/day")
            
            # Calculate opportunities that passed filters
            label_distribution = results.get('label_distribution', {})
            opportunity_labels = label_distribution.get('opportunity', {})
            if opportunity_labels:
                total_opportunities = opportunity_labels.get('total_labels', 0)
                positive_opportunities = opportunity_labels.get('value_counts', {}).get('1', 0)
                if total_opportunities > 0:
                    filter_pass_rate = (positive_opportunities / total_opportunities) * 100
                    daily_passing_opportunities = (positive_opportunities / duration_days) if duration_days > 0 else 0
                    content.append(f"- **Total Opportunities:** {total_opportunities:,}")
                    content.append(f"- **Opportunities Passing Filters:** {positive_opportunities:,} ({filter_pass_rate:.1f}%)")
                    content.append(f"- **Daily Average Passing Opportunities:** {daily_passing_opportunities:.1f} opportunities/day")
            content.append("")

        # Quality Metrics Section
        quality_metrics = outcome_data.get('quality_metrics', {})
        if quality_metrics:
            content.append("## 🎯 Quality Metrics")
            content.append("")
            for target_name, metrics in quality_metrics.items():
                content.append(f"### {target_name}")
                content.append("")
                content.append(f"- **Overall Quality:** {metrics.get('overall_quality', 0):.4f}")
                content.append(f"- **Predictability:** {metrics.get('predictability', 0):.4f}")
                content.append(f"- **Stability:** {metrics.get('stability', 0):.4f}")
                content.append(f"- **Balance:** {metrics.get('balance', 0):.4f}")
                content.append("")
        else:
            content.append("## 🎯 Quality Metrics")
            content.append("")
            content.append("No quality scores available")
            content.append("")

        # Data Quality Section
        data_quality = outcome_data.get('data_quality', {})
        if data_quality:
            content.append("## 🔍 Data Quality Analysis")
            content.append("")
            input_data = data_quality.get('input_data', {})
            output_labels = data_quality.get('output_labels', {})

            content.append("### Input Data")
            content.append("")
            content.append(f"- **Total Records:** {input_data.get('rows', 0):,}")
            content.append(f"- **Columns:** {input_data.get('columns', 0)}")
            content.append(f"- **Missing Values:** {input_data.get('missing_percentage', 0):.2f}%")
            content.append(f"- **Date Range:** {input_data.get('date_range', {}).get('start', 'N/A')} to {input_data.get('date_range', {}).get('end', 'N/A')}")
            content.append("")

            content.append("### Output Labels")
            content.append("")
            content.append(f"- **Labels Generated:** {output_labels.get('total_generated', 0):,}")
            content.append(f"- **Label Coverage:** {output_labels.get('label_coverage', 0):.1f}%")
            content.append(f"- **Targets per Sample:** {output_labels.get('targets_per_sample', 0):.2f}")
            content.append("")

        # Horizon Breakdown
        horizon_breakdown = results.get('horizon_breakdown', {})
        if horizon_breakdown:
            content.append("## 🌅 Horizon Analysis")
            content.append("")
            for horizon_name, metrics in horizon_breakdown.items():
                content.append(f"### {horizon_name}")
                content.append("")
                content.append(f"- **Expected Labels:** {metrics.get('expected_labels', 0):,}")
                content.append(f"- **Horizon Bars:** {metrics.get('horizon_bars', 0)}")
                content.append("")

        # Target Breakdown
        target_breakdown = results.get('target_breakdown', {})
        if target_breakdown:
            content.append("## 🎯 Target Analysis")
            content.append("")
            for target_name, metrics in target_breakdown.items():
                content.append(f"### {target_name}")
                content.append("")
                content.append(f"- **Expected Labels:** {metrics.get('expected_labels', 0):,}")
                content.append("")

        # Recommendations
        if outcome_data.get('status') == 'success':
            content.append("## 💡 Summary")
            content.append("✅ **Analyst profit labeling completed successfully!**")
            content.append("")
            content.append("The labeled dataset is ready for model training. Consider:")
            content.append("- Using the generated labels for training analyst models")
            content.append("- Analyzing quality metrics to ensure label reliability")
            content.append("- Reviewing horizon and target breakdowns for optimization opportunities")

        content.append("")
        content.append("---")
        content.append(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        return "\n".join(content)

    def get_label_summary(self, result: LabelingResult) -> Dict[str, Any]:
        """Get a summary of the labeling results."""
        tprint_info("📊 Generating label summary...")

        try:
            summary = {
                'n_samples': result.n_samples,
                'n_targets': result.n_targets,
                'n_horizons': result.n_horizons,
                'processing_time': result.processing_time,
                'quality_scores': {}
            }

            # Add quality scores
            if result.quality_scores:
                tprint_info(f"📈 Processing {len(result.quality_scores)} quality scores...")

                # Enhanced quality metrics logging
                tprint_info("📊 Detailed Quality Metrics Analysis:")
                for target_name, quality in result.quality_scores.items():
                    tprint_info(f"  🎯 Target '{target_name}':")
                    tprint_info(f"     📈 Overall Quality: {quality.overall_quality:.4f}")
                    tprint_info(f"     🔮 Predictability: {quality.predictability:.4f}")
                    tprint_info(f"     ⚖️  Stability: {quality.stability:.4f}")
                    tprint_info(f"     📊 Balance: {quality.balance:.4f}")

                    # Check against thresholds
                    meets_quality = quality.overall_quality >= self.labeler.config.min_label_quality
                    meets_predictability = quality.predictability >= self.labeler.config.min_predictability

                    tprint_info(f"     ✅ Quality Threshold (≥{self.labeler.config.min_label_quality}): {'PASS' if meets_quality else 'FAIL'}")
                    tprint_info(f"     🔮 Predictability Threshold (≥{self.labeler.config.min_predictability}): {'PASS' if meets_predictability else 'FAIL'}")
                    tprint_info(f"     🎯 Overall Status: {'✅ APPROVED' if meets_quality and meets_predictability else '❌ REJECTED'}")

                    summary['quality_scores'][target_name] = {
                        'overall_quality': quality.overall_quality,
                        'predictability': quality.predictability,
                        'stability': quality.stability,
                        'balance': quality.balance
                    }
                tprint_success(f"✅ Processed quality scores for {len(result.quality_scores)} targets")
            else:
                tprint_warning("⚠️ No quality scores available in result")

            tprint_success(f"✅ Label summary generated: {summary['n_samples']} samples, {summary['n_targets']} targets, {summary['n_horizons']} horizons")

            # Final decision summary
            # Compute expected labels directly from produced labels instead of relying on external breakdowns
            # Expected = number of positive opportunity flags in the final (post-filter) labels
            total_expected_labels = 0
            raw_opportunities = 0
            if hasattr(result, 'labels') and result.labels is not None:
                if isinstance(result.labels, pd.DataFrame):
                    # Count both raw positives and expected positives from the final labels
                    # Treat any numeric column where value == 1 as an opportunity
                    numeric_cols = [c for c in result.labels.columns if pd.api.types.is_numeric_dtype(result.labels[c])]
                    if numeric_cols:
                        expected_per_col = [(result.labels[c] == 1).sum() for c in numeric_cols]
                        total_expected_labels = sum(int(x) for x in expected_per_col)
                        raw_opportunities = total_expected_labels
                elif isinstance(result.labels, pd.Series):
                    total_expected_labels = int((result.labels == 1).sum())
                    raw_opportunities = total_expected_labels
            
            tprint_info("🎯 Final Labeling Decision:")
            tprint_info(f"   📊 Raw opportunities found: {raw_opportunities:,} (before quality filtering)")
            if raw_opportunities > 0:
                # Calculate days for per-day statistics
                total_samples = len(result.labels) if hasattr(result, 'labels') and result.labels is not None else 0
                days_in_data = total_samples / (24 * 4) if total_samples > 0 else 0
                raw_opportunities_per_day = raw_opportunities / days_in_data if days_in_data > 0 else 0
                tprint_info(f"   📊 Raw opportunities per day: {raw_opportunities_per_day:.1f} (before quality filtering)")
            else:
                tprint_info(f"   📊 Raw opportunities per day: 0.0 (before quality filtering)")
            
            if total_expected_labels > 0:
                tprint_success(f"✅ Analyst labels will be generated: {total_expected_labels} total labels (after quality filtering)")
                # When using consolidated labels, expected equals raw; keep the filtered-out line only if meaningful
                if raw_opportunities > total_expected_labels:
                    filtered_out = raw_opportunities - total_expected_labels
                    tprint_info(f"   📊 Filtered out: {filtered_out:,} opportunities ({filtered_out/raw_opportunities:.1%} of raw count)")
            else:
                tprint_warning("❌ No analyst labels will be generated - thresholds not met")
                tprint_info(f"   📋 Thresholds: Quality ≥{self.config.min_label_quality}, Predictability ≥{self.config.min_predictability}")
                if raw_opportunities > 0:
                    tprint_info(f"   📊 All {raw_opportunities:,} raw opportunities were filtered out due to quality thresholds")
                if result.quality_scores:
                    tprint_info("   📊 Consider lowering thresholds or using tactician labeler for more lenient labeling")
                else:
                    tprint_info("   📊 No quality scores available for assessment")

            return summary

        except Exception as e:
            tprint_error(f"❌ Failed to generate label summary: {e}")
            # Return a basic summary even if quality scores fail
            return {
                'n_samples': getattr(result, 'n_samples', 0),
                'n_targets': getattr(result, 'n_targets', 0),
                'n_horizons': getattr(result, 'n_horizons', 0),
                'processing_time': getattr(result, 'processing_time', 0),
                'quality_scores': {},
                'error': str(e)
            }

class AnalystProfitLabelerComponent(BasePreTrainingComponent):
    """
    Component wrapper for Analyst Profit Labeler.

    This component integrates the AnalystProfitLabeler with the pre-training pipeline
    and handles proper error handling, reporting, and pipeline state management.
    """

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the Analyst profit labeler component."""
        super().__init__(config)
        self.logger = system_logger.getChild('AnalystProfitLabelerComponent')

        # Create Analyst-specific configuration
        analyst_config = AnalystProfitLabelerConfig()

        # Override with custom parameters if provided
        if self.config and self.config.custom_params:
            custom_params = self.config.custom_params

            # Update timeframe if provided
            if 'timeframe' in custom_params:
                analyst_config.timeframe = custom_params['timeframe']
                # Update base period based on timeframe
                if analyst_config.timeframe.endswith('m'):
                    analyst_config.base_period_minutes = int(analyst_config.timeframe[:-1])
                elif analyst_config.timeframe.endswith('h'):
                    analyst_config.base_period_minutes = int(analyst_config.timeframe[:-1]) * 60

            # Update other parameters
            for key in ['horizons', 'min_label_quality', 'min_predictability', 'enable_advanced_filters']:
                if key in custom_params:
                    setattr(analyst_config, key, custom_params[key])
            
            # Handle target_profits -> target_profit mapping for backward compatibility
            if 'target_profits' in custom_params:
                target_profits = custom_params['target_profits']
                if isinstance(target_profits, list) and len(target_profits) > 0:
                    # Use the first target for single-target design
                    analyst_config.target_profit = target_profits[0]
                    if len(target_profits) > 1:
                        tprint_warning(f"⚠️ Multiple targets provided {target_profits}, using first target {target_profits[0]}% for single-target design")
                else:
                    analyst_config.target_profit = target_profits
            elif 'target_profit' in custom_params:
                analyst_config.target_profit = custom_params['target_profit']

            # Update advanced filters configuration if provided
            if 'advanced_filters_config' in custom_params and ADVANCED_FILTERS_AVAILABLE:
                filters_config_dict = custom_params['advanced_filters_config']
                if isinstance(filters_config_dict, dict):
                    analyst_config.advanced_filters_config = AdvancedFiltersConfig(**filters_config_dict)
                elif isinstance(filters_config_dict, AdvancedFiltersConfig):
                    analyst_config.advanced_filters_config = filters_config_dict

            # Store all custom params for the underlying labeler
            analyst_config.custom_params = custom_params

        # Create the labeler
        try:
            self.labeler = AnalystProfitLabeler(analyst_config)
            tprint_success("✅ AnalystProfitLabelerComponent initialized")
        except Exception as e:
            tprint_error(f"❌ Failed to initialize AnalystProfitLabelerComponent: {e}")
            raise

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        tprint_info("📋 Getting required artifacts list")
        artifacts = ['multi_horizon_labeling_result', 'labeling_report']
        tprint_success(f"✅ Required artifacts: {artifacts}")
        return artifacts

    async def execute(self, data: Any, pipeline_state: PipelineState) -> ComponentResult:
        """
        Execute Analyst profit labeling as a component.

        Args:
            data: Input data (typically market data DataFrame)
            pipeline_state: Current pipeline state

        Returns:
            ComponentResult with labeling results and artifacts
        """
        try:
            tprint_info("🚀 Starting Analyst Profit Labeling execution...")

            # Extract data from pipeline state if not provided
            if data is None:
                data = pipeline_state.get('prepared_data')
                if data is None:
                    raise ValueError("No input data provided and no prepared_data in pipeline state")

            # Validate OHLCV data format and constraints
            try:
                data = validate_raw_ohlcv(data)
                tprint_info(f"✅ Input data validated: {len(data)} rows, {len(data.columns)} columns")
            except SchemaValidationException as e:
                tprint_error(f"❌ Input data validation failed: {e}")
                raise ValueError(f"Invalid input data format: {e}") from e

            # Extract regime assignments if available
            regime_assignments = pipeline_state.get('regime_assignments')
            if regime_assignments is not None:
                tprint_info(f"📊 Using regime assignments: {len(regime_assignments)} regimes")

            # Generate labels using the consolidated labeling process
            labeling_result = self.labeler.generate_labels(
                data=data,
                regime_assignments=regime_assignments
            )

            # ==================== VALIDATION PIPELINE ====================
            tprint_info("🔍 Running validation pipeline...")
            
            # Import validation utilities
            try:
                from src.utils.ml_common.validation import (
                    validate_temporal_consistency,
                    validate_window_quality,
                    benchmark_stage,
                    BenchmarkConfig
                )
                VALIDATION_AVAILABLE = True
            except ImportError as e:
                tprint_warning(f"⚠️ Validation utilities not available: {e}")
                VALIDATION_AVAILABLE = False
            
            # Validation results container
            validation_results = {}
            
            if VALIDATION_AVAILABLE:
                try:
                    # 1. Temporal alignment validation
                    with benchmark_stage("analyst_temporal_validation") as temporal_metrics:
                        temporal_artifacts = {
                            'input_data': data,
                            'labels': labeling_result.labels if hasattr(labeling_result, 'labels') else None,
                            'tactician_entry_labels': getattr(labeling_result.metadata, 'tactician_entry_labels', None) if hasattr(labeling_result, 'metadata') else None
                        }
                        
                        # Filter out None artifacts
                        temporal_artifacts = {k: v for k, v in temporal_artifacts.items() if v is not None}
                        
                        if len(temporal_artifacts) > 1:
                            temporal_result = validate_temporal_consistency(
                                temporal_artifacts,
                                list(temporal_artifacts.keys()),
                                config={
                                    'require_exact_match': False,  # Allow some tolerance
                                    'tolerance_seconds': 300,  # 5 minutes tolerance
                                    'check_data_hash': False
                                }
                            )
                            validation_results['temporal'] = temporal_result
                            temporal_metrics.custom_metrics = {
                                'artifacts_validated': len(temporal_artifacts),
                                'temporal_alignment_passed': temporal_result['success']
                            }
                        else:
                            tprint_warning("⚠️ Insufficient artifacts for temporal validation")
                    
                    # 2. Window quality assessment
                    with benchmark_stage("analyst_window_validation") as window_metrics:
                        # Extract opportunity windows from metadata
                        opportunity_windows = []
                        if hasattr(labeling_result, 'metadata') and labeling_result.metadata:
                            if hasattr(labeling_result.metadata, 'opportunity_windows'):
                                opportunity_windows = labeling_result.metadata.opportunity_windows
                            elif isinstance(labeling_result.metadata, dict):
                                opportunity_windows = labeling_result.metadata.get('opportunity_windows', [])
                        
                        if opportunity_windows:
                            window_artifacts = {
                                'opportunity_windows': opportunity_windows,
                                'data': data
                            }
                            
                            window_result = validate_window_quality(
                                window_artifacts,
                                config={
                                    'require_min_windows': 1,
                                    'max_overlap_ratio': 0.2,  # Allow up to 20% overlap
                                    'min_coverage_ratio': 0.001,  # Very low threshold
                                    'strict_mode': False  # Don't fail on warnings
                                }
                            )
                            validation_results['windows'] = window_result
                            window_metrics.custom_metrics = {
                                'total_windows': window_result['results']['windows'].total_windows if window_result['results'] else 0,
                                'valid_windows': window_result['results']['windows'].valid_windows if window_result['results'] else 0,
                                'window_validation_passed': window_result['success']
                            }
                        else:
                            tprint_warning("⚠️ No opportunity windows found for validation")
                    
                    # Log validation summary
                    validation_passed = all(
                        result.get('success', False) 
                        for result in validation_results.values()
                    )
                    
                    if validation_passed:
                        tprint_success("✅ All validation checks passed")
                    else:
                        tprint_warning("⚠️ Some validation checks failed - see details above")
                        for check_name, result in validation_results.items():
                            if not result.get('success', False):
                                tprint_warning(f"   → {check_name}: {result.get('error', 'Unknown error')}")
                    
                except Exception as validation_error:
                    tprint_warning(f"⚠️ Validation pipeline failed: {validation_error}")
                    validation_results['error'] = str(validation_error)
            else:
                tprint_warning("⚠️ Skipping validation pipeline - utilities not available")

            # Create artifacts bundle
            from src.training.steps.pre_training.components.contracts import GenericArtifacts

            # Save labeled data to parquet file for persistence
            symbol = pipeline_state.get('symbol', 'UNKNOWN')
            exchange = pipeline_state.get('exchange', 'UNKNOWN')
            timeframe = pipeline_state.get('timeframe', 'UNKNOWN')
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

            artifacts_dir = Path('artifacts')
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            labeled_data_file = artifacts_dir / f'labeled_data_{symbol}_{exchange}_{timeframe}_{timestamp_str}.parquet'

            # Save labeled DataFrame to parquet
            if isinstance(labeling_result.labels, pd.DataFrame) and len(labeling_result.labels) > 0:
                labeling_result.labels.to_parquet(labeled_data_file)
                tprint_success(f"✅ Saved labeled data to {labeled_data_file}")
            else:
                # No file was saved, don't include the path
                labeled_data_file = None

            # GenericArtifacts just needs to be instantiated, then we add attributes
            artifacts = GenericArtifacts()
            artifacts.multi_horizon_labeling_result = {
                'labeled_data': labeling_result.labels,  # Keep in memory for pipeline continuity
                'labeled_data_file': str(labeled_data_file) if labeled_data_file is not None else None,  # Only add path if file exists
                'labels': labeling_result.labels,
                'confidence_scores': labeling_result.confidence_scores,
                'eligibility_masks': labeling_result.eligibility_masks,
                'quality_scores': labeling_result.quality_scores,
                'normalization_factors': labeling_result.normalization_factors,
                'processing_time': labeling_result.processing_time,
                'n_samples': labeling_result.n_samples,
                'n_targets': labeling_result.n_targets,
                'n_horizons': labeling_result.n_horizons,
                'method': 'analyst_profit_labeling',
            }
            artifacts.labeling_report = {
                'status': 'completed',
                'timestamp': datetime.now().isoformat(),
                'method': 'analyst_profit_labeling',
                'timeframe': self.labeler.config.timeframe if hasattr(self.labeler.config, 'timeframe') else '15m',
                'summary': self.labeler.get_label_summary(labeling_result),
                'validation_results': validation_results,  # Add validation results to report
                'horizons': self.labeler.config.horizons,
                'target_profits': self.labeler.config.target_profits,
            }

            # Calculate additional metrics
            opportunities_per_day = 0.0
            raw_opportunities_per_day = 0.0

            if labeling_result.labels is not None and isinstance(labeling_result.labels, pd.DataFrame) and len(labeling_result.labels) > 0:
                # Calculate opportunities per day (post-filtering)
                try:
                    dates = pd.to_datetime(labeling_result.labels.index)
                    if len(dates) > 1:
                        date_range_days = (dates.max() - dates.min()).days
                        if date_range_days > 0:
                            # Count positive labels (actual opportunities after quality filtering)
                            positive_labels = 0
                            for col in labeling_result.labels.columns:
                                if pd.api.types.is_numeric_dtype(labeling_result.labels[col]):
                                    positive_labels += (labeling_result.labels[col] == 1).sum()
                            opportunities_per_day = round(float(positive_labels / date_range_days), 2)
                            
                            # Get raw opportunity count from metadata if available
                            raw_opportunities = labeling_result.metadata.get('total_opportunities', positive_labels)
                            raw_opportunities_per_day = round(float(raw_opportunities / date_range_days), 2)
                            
                            # Report both raw and filtered counts
                            tprint_info(f"📊 Opportunity Analysis:")
                            tprint_info(f"   📈 Raw opportunities: {raw_opportunities:,} ({raw_opportunities_per_day:.1f} per day)")
                            tprint_info(f"   🎯 Filtered opportunities: {positive_labels:,} ({opportunities_per_day:.1f} per day)")
                            if raw_opportunities > 0:
                                filter_rate = (raw_opportunities - positive_labels) / raw_opportunities
                                tprint_info(f"   🔍 Quality filter rate: {filter_rate:.1%} filtered out")
                            
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to calculate opportunities per day: {e}")
                    opportunities_per_day = 0.0
                    raw_opportunities_per_day = 0.0

            # Create result
            result = ComponentResult(
                success=True,
                metadata={
                    'component': 'analyst_profit_labeler',
                    'timeframe': self.labeler.config.timeframe if hasattr(self.labeler.config, 'timeframe') else '15m',
                    'artifacts': artifacts,
                    'n_samples': labeling_result.n_samples,
                    'n_targets': labeling_result.n_targets,
                    'n_horizons': labeling_result.n_horizons,
                    'direction_settings': {
                        'enable_long_positions': self.labeler.config.enable_long_positions,
                        'enable_short_positions': self.labeler.config.enable_short_positions,
                    },
                    'opportunities_per_day': opportunities_per_day,
                    'raw_opportunities_per_day': raw_opportunities_per_day
                }
            )

            # Generate outcome file with datetime stamp
            try:
                outcome_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                outcomes_dir = Path('outcomes')
                ensure_directory(outcomes_dir)

                outcome_filename = f"analyst_labeler_outcome_{outcome_timestamp}.json"
                outcome_path = outcomes_dir / outcome_filename

                # Create comprehensive outcome report with detailed statistics
                # Always generate a report, even when no labels are created

                # Calculate label distribution statistics
                label_distribution = {}
                if labeling_result.labels is not None and isinstance(labeling_result.labels, pd.DataFrame):
                    for col in labeling_result.labels.columns:
                        label_vals = labeling_result.labels[col].dropna()
                        if len(label_vals) > 0:
                            label_distribution[col] = {
                                'total_labels': int(len(label_vals)),
                                'unique_values': int(label_vals.nunique()),
                                'value_counts': label_vals.value_counts().head(10).to_dict(),
                                'mean': float(label_vals.mean()) if pd.api.types.is_numeric_dtype(label_vals) else None,
                                'std': float(label_vals.std()) if pd.api.types.is_numeric_dtype(label_vals) else None,
                                'min': float(label_vals.min()) if pd.api.types.is_numeric_dtype(label_vals) else None,
                                'max': float(label_vals.max()) if pd.api.types.is_numeric_dtype(label_vals) else None,
                            }
                else:
                    # No labels generated - add explanation to report
                    label_distribution = {
                        'no_labels_generated': {
                            'reason': 'Quality thresholds not met',
                            'total_labels': 0,
                            'explanation': 'No analyst labels were generated due to quality thresholds not being met. This could be due to insufficient data quality, low predictability, or other quality metrics falling below the minimum thresholds.'
                        }
                    }

                # Calculate per-horizon and per-target breakdowns
                horizon_breakdown = {}
                if labeling_result.labels is not None and isinstance(labeling_result.labels, pd.DataFrame) and len(labeling_result.labels) > 0:
                    for i, horizon in enumerate(self.labeler.config.horizons):
                        horizon_breakdown[f"{horizon}min"] = {
                            'horizon_minutes': horizon,
                            'horizon_bars': horizon // self.labeler.config.base_period_minutes,
                            'expected_labels': getattr(labeling_result, 'n_targets', 0) // len(self.labeler.config.horizons) if getattr(labeling_result, 'n_targets', 0) > 0 else 0,
                        }
                else:
                    # No labels generated - add horizon breakdown explanation
                    for i, horizon in enumerate(self.labeler.config.horizons):
                        horizon_breakdown[f"{horizon}min"] = {
                            'horizon_minutes': horizon,
                            'horizon_bars': horizon // self.labeler.config.base_period_minutes,
                            'expected_labels': 0,
                            'actual_labels': 0,
                            'status': 'no_labels_generated'
                        }

                # Calculate target breakdown
                target_breakdown = {}
                if labeling_result.labels is not None and isinstance(labeling_result.labels, pd.DataFrame) and len(labeling_result.labels) > 0:
                    for i, target in enumerate(self.labeler.config.target_profits):
                        target_breakdown[f"{target}%"] = {
                            'target_profit_pct': target,
                            'expected_labels': getattr(labeling_result, 'n_targets', 0) // len(self.labeler.config.target_profits) if getattr(labeling_result, 'n_targets', 0) > 0 else 0,
                        }
                else:
                    # No labels generated - add target breakdown explanation
                    for i, target in enumerate(self.labeler.config.target_profits):
                        target_breakdown[f"{target}%"] = {
                            'target_profit_pct': target,
                            'expected_labels': 0,
                            'actual_labels': 0,
                            'status': 'no_labels_generated'
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
                        'total_generated': labeling_result.n_samples,
                        'label_coverage': float(labeling_result.n_samples / len(data) * 100) if len(data) > 0 else 0.0,
                        'targets_per_sample': float(labeling_result.n_targets / labeling_result.n_samples) if labeling_result.n_samples > 0 else 0.0,
                    }
                }

                outcome_data = {
                    'component': 'analyst_profit_labeler',
                    'timestamp': datetime.now().isoformat(),
                    'execution_time': labeling_result.processing_time,
                    'timeframe': self.labeler.config.timeframe if hasattr(self.labeler.config, 'timeframe') else '15m',
                    'configuration': {
                        'horizons': self.labeler.config.horizons,
                        'target_profits': self.labeler.config.target_profits,
                        'use_volatility_normalization': self.labeler.config.use_volatility_normalization,
                        'volatility_window': self.labeler.config.volatility_window,
                        'min_label_quality': self.labeler.config.min_label_quality,
                        'min_predictability': self.labeler.config.min_predictability,
                        'enable_regime_adaptation': self.labeler.config.enable_regime_adaptation,
                        'enable_long_positions': self.labeler.config.enable_long_positions,
                        'enable_short_positions': self.labeler.config.enable_short_positions,
                        'base_period_minutes': self.labeler.config.base_period_minutes,
                    },
                    'results': {
                        'n_samples': labeling_result.n_samples,
                        'n_targets': labeling_result.n_targets,
                        'n_horizons': labeling_result.n_horizons,
                        'label_summary': self.labeler.get_label_summary(labeling_result),
                        'label_distribution': label_distribution,
                        'horizon_breakdown': horizon_breakdown,
                        'target_breakdown': target_breakdown,
                        'labels_generated': labeling_result.labels is not None and isinstance(labeling_result.labels, pd.DataFrame) and len(labeling_result.labels) > 0,
                        'no_labels_reason': 'Quality thresholds not met' if labeling_result.labels is None or (isinstance(labeling_result.labels, pd.DataFrame) and len(labeling_result.labels) == 0) else None,
                        'tactician_entry_labels_count': int(labeling_result.metadata.get('tactician_entry_labels', pd.Series(dtype=int)).sum()) if isinstance(labeling_result.metadata.get('tactician_entry_labels'), pd.Series) else 0,
                    },
                    'quality_metrics': {
                        target_name: {
                            'overall_quality': float(quality.overall_quality) if quality.overall_quality is not None else None,
                            'predictability': float(quality.predictability) if quality.predictability is not None else None,
                            'stability': float(quality.stability) if quality.stability is not None else None,
                            'balance': float(quality.balance) if quality.balance is not None else None,
                        }
                        for target_name, quality in (labeling_result.quality_scores or {}).items()
                    },
                    'data_quality': data_quality,
                    'data_info': {
                        'input_rows': len(data),
                        'input_columns': len(data.columns),
                        'regime_assignments_available': regime_assignments is not None,
                        'regime_count': int(regime_assignments.nunique()) if regime_assignments is not None and hasattr(regime_assignments, 'nunique') else None,
                    },
                    'confidence_statistics': {
                        'total_confidence_scores': len(labeling_result.confidence_scores) if labeling_result.confidence_scores is not None else 0,
                        'confidence_columns': list(labeling_result.confidence_scores.columns) if labeling_result.confidence_scores is not None and hasattr(labeling_result.confidence_scores, 'columns') else [],
                    },
                    'eligibility_statistics': {
                        'total_eligibility_masks': len(labeling_result.eligibility_masks) if labeling_result.eligibility_masks is not None else 0,
                        'eligible_samples': int(labeling_result.eligibility_masks.sum().sum()) if labeling_result.eligibility_masks is not None and hasattr(labeling_result.eligibility_masks, 'sum') else 0,
                    },
                    'normalization_factors': labeling_result.normalization_factors if labeling_result.normalization_factors else {},
                    'status': 'success'
                }

                safe_json_dump(outcome_data, str(outcome_path))
                tprint_success(f"📄 Outcome file saved: {outcome_filename}")

                # Generate human-readable outcome file
                self.labeler._save_human_readable_outcome(outcome_data, outcome_path)

            except Exception as outcome_error:
                tprint_warning(f"⚠️ Failed to save outcome file: {outcome_error}")
                # Don't fail the component if outcome file generation fails

            tprint_success("✅ Analyst Profit Labeling completed successfully")
            return result

        except Exception as e:
            tprint_error(f"❌ Analyst Profit Labeling failed: {e}")

            result = ComponentResult(
                success=False,
                error_message=str(e),
                metadata={'component': 'analyst_profit_labeler'}
            )
            return result

    def process(self, data: Any) -> Any:
        """Process the input data and return the result."""
        try:
            # Extract required data from the input
            if hasattr(data, 'data') and hasattr(data, 'regime_assignments'):
                # Data is already in the expected format
                return self.execute_analyst_profit_labeling(data.data, data.regime_assignments)
            elif isinstance(data, dict) and 'data' in data and 'regime_assignments' in data:
                # Data is a dictionary with the required keys
                return self.execute_analyst_profit_labeling(data['data'], data['regime_assignments'])
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
                
                return self.execute_analyst_profit_labeling(data_obj, regime_assignments)
        except Exception as e:
            self.logger.error(f"Error processing data in AnalystProfitLabelerComponent: {e}")
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
            self.logger.error(f"Error validating data in AnalystProfitLabelerComponent: {e}")
            return False

# Convenience function for external usage
async def execute_analyst_profit_labeling(
    data: pd.DataFrame,
    regime_assignments: Optional[pd.DataFrame] = None,
    config: Optional[AnalystProfitLabelerConfig] = None,
    **kwargs
) -> LabelingResult:
    """
    Execute Analyst profit labeling.

    Args:
        data: Input market data (OHLCV format)
        regime_assignments: Optional regime assignments
        config: Optional configuration
        **kwargs: Additional parameters

    Returns:
        LabelingResult with labels and quality metrics
    """
    tprint_info("🚀 Starting execute_analyst_profit_labeling...")

    try:
        tprint_info(f"📊 Input data: {len(data)} rows, {len(data.columns)} columns")
        if regime_assignments is not None:
            tprint_info(f"📈 Regime assignments: {len(regime_assignments)} regimes")
        else:
            tprint_info("📈 No regime assignments provided")

        tprint_info("🔧 Creating AnalystProfitLabeler...")
        labeler = AnalystProfitLabeler(config)
        tprint_success("✅ AnalystProfitLabeler created successfully")

        tprint_info("📈 Generating labels...")
        result = labeler.generate_labels(data, regime_assignments, **kwargs)
        tprint_success(f"✅ Labels generated successfully: {result.n_samples} samples, {result.n_targets} targets")

        return result

    except Exception as e:
        tprint_error(f"❌ execute_analyst_profit_labeling failed: {e}")
        raise

# Register component with factory
def _register_analyst_profit_labeler():
    """Register the analyst profit labeler component with the factory."""
    try:
        from src.training.steps.pre_training.components import ComponentFactory
        ComponentFactory.register_component(
            'analyst_profit_labeler',
            AnalystProfitLabelerComponent
        )
    except ImportError:
        # Component factory not available, skip registration
        pass

# Register the component when module is imported
_register_analyst_profit_labeler()
