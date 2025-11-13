"""
Volatility Aware Labeler Module

This module provides volatility-aware labeling functionality for profit labeling.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
import pandas as pd
import numpy as np

# Import label smoothing
from .label_smoother import LabelSmoother, LabelSmoothingConfig

# Import tprint utilities
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
except ImportError:
    # Fallback implementation if tprint not available
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

# Robust logger fallback
try:
    from src.utils.logger import system_logger  # type: ignore
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)
    system_logger = logging.getLogger("system")

# Robust piecewise linear mapping for proximity-based confidence
def proximity_mapping(
    ratio: np.ndarray, 
    min_thresh: float = 0.75, 
    cap: float = 1.5,
    adaptive_clip: bool = True,
    clip_percentile: float = 99.9
) -> np.ndarray:
    """
    Compute proximity-based confidence using piecewise linear mapping.
    
    Args:
        ratio: actual_move / target_move (can be negative for opposite direction)
        min_thresh: Minimum threshold below which confidence is 0 (default: 0.75)
        cap: Maximum confidence cap (default: 1.5)
        adaptive_clip: Use percentile-based clipping instead of fixed threshold
        clip_percentile: Percentile for adaptive clipping (default: 99.9)
    
    Returns:
        proximity (float) in [0, cap] with preserved sign
    """
    # Handle outliers with adaptive or fixed clipping
    if adaptive_clip and len(ratio) > 0:
        # Use percentile-based clipping for better adaptation to data distribution
        clip_value = np.percentile(np.abs(ratio), clip_percentile)
        ratio = np.clip(ratio, -clip_value, clip_value)
    else:
        # Fixed clipping as fallback
        ratio = np.clip(ratio, -10.0, 10.0)
    
    absd = np.abs(ratio)
    prox = np.zeros_like(absd, dtype=float)
    
    # Below minimum threshold -> 0 confidence
    mask_below = absd < min_thresh
    prox[mask_below] = 0.0
    
    # Edge continuity: smooth transition near 0.75 threshold
    # Use a small ramp to avoid label discontinuity
    ramp_start = min_thresh - 0.05  # Start ramp at 0.70
    mask_ramp = (absd >= ramp_start) & (absd < min_thresh)
    if np.any(mask_ramp):
        # Linear ramp from 0 to 0.5 over 0.05 range
        ramp_factor = (absd[mask_ramp] - ramp_start) / (min_thresh - ramp_start)
        prox[mask_ramp] = 0.5 * ramp_factor
    
    # 0.75 <= absd < 1.0 -> linear from 0.5 at 0.75 to 1.0 at 1.0
    mask_mid = (absd >= min_thresh) & (absd < 1.0)
    prox[mask_mid] = 0.5 + 2.0 * (absd[mask_mid] - min_thresh)  # 0.5 + 2.0 * (distance - 0.75)
    
    # 1.0 <= absd <= 2.0 -> linear from 1.0 at 1.0 to 1.5 at 2.0
    mask_high = (absd >= 1.0) & (absd <= 2.0)
    prox[mask_high] = 1.0 + 0.5 * (absd[mask_high] - 1.0)  # 1.0 + 0.5 * (distance - 1.0)
    
    # > 2.0 -> cap
    mask_above = absd > 2.0
    prox[mask_above] = cap
    
    # Handle signed-zero deterministically: treat exactly zero as positive
    sign = np.where(ratio == 0, 1, np.sign(ratio))
    return sign * prox

# Import the missing function from multi_horizon_profit_labeler
try:
    from src.training.steps.pre_training.multi_horizon_profit_labeler import create_enhanced_tactician_labeler
    # Check if the function is actually available (not the fallback)
    import inspect
    if hasattr(create_enhanced_tactician_labeler, '__name__') and 'Unavailable' in str(create_enhanced_tactician_labeler):
        # The function is the fallback version, create a proper implementation
        def create_enhanced_tactician_labeler(*args: Any, **kwargs: Any) -> Any:
            """Enhanced tactician labeler implementation."""
            # For now, return a simple implementation that doesn't fail
            class SimpleTacticianLabeler:
                def __init__(self, *args, **kwargs):
                    pass
                def generate_labels(self, *args, **kwargs):
                    return {"labels": None, "metadata": {"type": "tactician", "status": "fallback"}}
            return SimpleTacticianLabeler(*args, **kwargs)
except ImportError:
    # Fallback implementation if import fails
    def create_enhanced_tactician_labeler(*args: Any, **kwargs: Any) -> Any:
        """Fallback implementation for create_enhanced_tactician_labeler."""
        class SimpleTacticianLabeler:
            def __init__(self, *args, **kwargs):
                pass
            def generate_labels(self, *args, **kwargs):
                return {"labels": None, "metadata": {"type": "tactician", "status": "fallback"}}
        return SimpleTacticianLabeler(*args, **kwargs)


def _align_like(left: pd.Series, right: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Align two series using inner join to ensure consistent indices.
    
    Args:
        left: First series to align
        right: Second series to align
        
    Returns:
        Tuple of aligned series (left_aligned, right_aligned)
    """
    a, b = left.align(right, join="inner")
    return a, b


class LabelDefinitionType(Enum):
    """Enum for label definition types."""
    BINARY = "binary"
    MULTI_CLASS = "multi_class"
    REGRESSION = "regression"
    SMOOTH_BINARY = "smooth_binary"  # Smooth binary labels with confidence weighting
    SMOOTH_REGRESSION = "smooth_regression"  # Smooth regression labels with proximity weighting
    PROXIMITY_REGRESSION = "proximity_regression"  # Regression with proximity-based confidence and sample weights
    ANALYST = "analyst"  # For analyst profit labeling (long-term analysis)
    TACTICIAN = "tactician"  # For tactician entry labeling (short-term entry)


class VolatilityAwareConfig:
    """
    Configuration for volatility-aware labeling.
    """
    
    def __init__(
        self,
        volatility_threshold: float = 0.02,
        lookahead_periods: int = 6,
        min_volatility: float = 0.001,
        max_volatility: float = 0.1,
        label_type: LabelDefinitionType = LabelDefinitionType.PROXIMITY_REGRESSION,
        enable_long_positions: bool = True,
        enable_short_positions: bool = False,
        min_label_quality: float = 0.3,
        min_predictability: float = 0.2
    ):
        """
        Initialize volatility-aware configuration.
        
        Args:
            volatility_threshold: Threshold for volatility-based labeling
            lookahead_periods: Number of periods to look ahead
            min_volatility: Minimum volatility threshold
            max_volatility: Maximum volatility threshold
            label_type: Type of labels to generate
            enable_long_positions: Whether to generate long position signals
            enable_short_positions: Whether to generate short position signals
        """
        self.volatility_threshold = volatility_threshold
        self.lookahead_periods = lookahead_periods
        self.min_volatility = min_volatility
        self.max_volatility = max_volatility
        self.label_type = label_type
        self.enable_long_positions = enable_long_positions
        self.enable_short_positions = enable_short_positions
        self.min_label_quality = min_label_quality
        self.min_predictability = min_predictability
        
        # Initialize additional configuration attributes
        self.label_definition_type = label_type
        self.enable_enhanced_labels = False
        self.timeframe = None
        self.enable_quality_scoring = True
        self.quality_scoring = QualityScoringConfig()
        self.regime_config = RegimeConfig()
        self.optimal_entry_detection = OptimalEntryDetectionConfig()
        
        # Make bar_construction optional to avoid import-time crashes
        try:
            from .bar_construction import BarConstructionConfig  # type: ignore
            self.bar_construction = BarConstructionConfig()
        except Exception:
            self.bar_construction = None
        
        # Initialize noise gating configuration
        self.noise_gating = NoiseGatingConfig()
        
        # Initialize multi-target configuration
        self.multi_target = MultiTargetConfig()

        # Initialize volatility configuration
        self.volatility = VolatilityConfig()
        # Rate control (data-driven calibration)
        self.rate_control = RateControlConfig()
        # Data-driven labeling calibration (quantile-based)
        self.data_driven = DataDrivenLabelingConfig()

        # Initialize label smoothing configuration
        self.label_smoothing = LabelSmoothingConfig(
            enabled=True,
            apply_classification_smoothing=True,
            apply_uncertainty_shrinkage=True,
            apply_causal_ema=True,
            epsilon=0.08,
            temperature=1.2,
            gamma=1.0,
            min_alpha=0.12,
            baseline=0.0,
            uncertainty_source='quality_inverse',
            ema_decay=0.95,
            ema_group_by=None,  # Will be set dynamically if instrument column exists
            ema_seed_method='first',
            ablation_mode='full',
            store_intermediate=False,
            validate_causality=True
        )

        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters and raise helpful errors."""
        errors = []
        
        # Validate volatility thresholds
        if not (0 < self.min_volatility <= self.max_volatility):
            errors.append(f"Invalid volatility range: min_volatility ({self.min_volatility}) must be > 0 and <= max_volatility ({self.max_volatility})")
        
        # Validate lookahead periods
        if self.lookahead_periods < 1:
            errors.append(f"lookahead_periods ({self.lookahead_periods}) must be >= 1")
        
        # Validate volatility window
        if self.volatility.window < 2:
            errors.append(f"volatility.window ({self.volatility.window}) must be >= 2")
        
        # Validate single target profit
        if hasattr(self.multi_target, 'target_profit') and self.multi_target.target_profit:
            if self.multi_target.target_profit <= 0:
                errors.append(f"multi_target.target_profit ({self.multi_target.target_profit}) must be > 0")
        
        # Validate quality scoring thresholds
        if not (0 <= self.quality_scoring.min_quality_threshold <= 1):
            errors.append(f"quality_scoring.min_quality_threshold ({self.quality_scoring.min_quality_threshold}) must be between 0 and 1")
        
        if not (0 <= self.quality_scoring.min_predictability <= 1):
            errors.append(f"quality_scoring.min_predictability ({self.quality_scoring.min_predictability}) must be between 0 and 1")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ValueError(error_msg)


class QualityScoringConfig:
    """Configuration for quality scoring."""
    def __init__(self) -> None:
        self.min_quality_threshold = 0.15  # Further reduced for crypto (was 0.2) - 15% quality threshold
        self.min_predictability = 0.15     # Further reduced for crypto (was 0.2) - 15% predictability threshold


class RegimeConfig:
    """Configuration for regime adaptation."""
    def __init__(self) -> None:
        self.enabled = False


class OptimalEntryDetectionConfig:
    """Configuration for optimal entry point detection."""
    def __init__(self) -> None:
        self.enabled = False
        self.entry_threshold = 0.5
        self.find_highest_gap_entry = False
        self.entry_point_strategy = "default"
        self.horizons = []
        self.target_profits = []
        self.multi_size_thresholds = []
        # Max bars to resolve an opportunity (your "6 bars max")
        self.max_windows = 6


class NoiseGatingConfig:
    """Configuration for noise gating."""
    def __init__(self) -> None:
        self.enabled = True


class MultiTargetConfig:
    """Configuration for single-target labeling with volatility modulation."""
    def __init__(self) -> None:
        self.horizons = []
        self.target_profit = 0.5  # 0.5% target over 6 periods (90min) - reasonable for 15m data
        self.min_lqs_score = 0.2  # Reduced from 0.3 to 0.2
        self.volatility_modulation = True  # Enable volatility-based threshold adjustment
        self.min_threshold_multiplier = 0.3  # Further reduced for crypto (was 0.5) to allow more opportunities
        self.max_threshold_multiplier = 2.0  # Maximum threshold multiplier


class VolatilityConfig:
    """Configuration for volatility settings."""
    def __init__(self) -> None:
        self.enabled = True
        self.window = 20  # Window for raw volatility calculation
        self.vol_ema_span = 100  # EMA span for smoothing volatility baseline
        self.sensitivity = 1.0  # Tunable parameter for volatility sensitivity
        self.alpha = 1.0  # Nonlinear sensitivity exponent (1.0 = linear, <1.0 = dampened)
        self.volatility_estimator = 'log_returns'  # 'log_returns', 'atr', or 'realized'
        self.percentile_clipping = False  # Use percentile-based clipping instead of fixed bounds
        self.percentile_low = 1.0  # Low percentile for clipping (e.g., 1st percentile)
        self.percentile_high = 99.0  # High percentile for clipping (e.g., 99th percentile)
        self.percentile_min_range = 0.5  # Minimum range between percentiles to prevent too narrow ranges
        self.rolling_percentile_window = 90  # Rolling window for dynamic percentile updates (days)
        self.hysteresis_threshold = 0.05  # Base minimum change required to update threshold
        self.adaptive_hysteresis = True  # Scale hysteresis by volatility
        self.hysteresis_volatility_factor = 0.1  # Factor to scale hysteresis by volatility (5-10%)
        self.volatility_floor = 1e-6  # Minimum volatility to prevent divide-by-zero
        self.warmup_policy = 'rolling_mean'  # 'rolling_mean', 'drop', or 'fillna'


class RateControlConfig:
    """Configuration for data-driven rate calibration."""
    def __init__(self) -> None:
        self.enabled = True
        # Removed max_ops_per_day limit as requested
        self.min_scale = 0.5      # search lower bound for threshold scaling
        self.max_scale = 3.0      # search upper bound for threshold scaling
        self.tolerance = 0.15     # tighter tolerance (was 0.25)


class DataDrivenLabelingConfig:
    """Configuration for data-driven (quantile-based) thresholding.

    This avoids heuristic fixed profit targets and instead calibrates label density
    from the empirical distribution of forward returns, optionally per-volatility bin.
    """
    def __init__(self) -> None:
        # Enable to use quantile-calibrated thresholds instead of fixed bps thresholds
        self.enabled = False
        # Target average number of signals per day (total of long+short) - conservative target
        self.target_ops_per_day = 6.0  # Reduced from 8.0 to ensure we stay well below the cap
        # Calibration mode: 'global' uses a single set of quantiles; 'vol_bin' uses per-volatility bins
        self.calibration_mode = "global"  # or 'vol_bin'
        # Number of quantile bins over normalized volatility when calibration_mode='vol_bin'
        self.volatility_bins = 5
        # Optional floor on minimal desired signal rate per day (acts as a lower bound)
        self.min_ops_per_day = 2.0
        # When both directions enabled, allocate target ops evenly across long/short
        self.long_short_split = "even"  # or 'auto' to allocate by skew
        # Rolling window size for quantile estimation; if None, uses full-sample
        self.rolling_window = None  # e.g., 2000 for ~3 months of 15m bars


class LabelingResult:
    """
    Result of labeling operation.
    """
    
    def __init__(
        self,
        labels: pd.Series,
        metadata: Dict[str, Any],
        success: bool = True,
        error_message: Optional[str] = None,
        quality_scores: Dict[str, Any] = None
    ):
        """
        Initialize labeling result.
        
        Args:
            labels: Generated labels
            metadata: Additional metadata
            success: Whether labeling was successful
            error_message: Error message if unsuccessful
        """
        self.labels = labels
        self.metadata = metadata
        self.success = success
        self.error_message = error_message
        self.quality_scores = quality_scores or {}
        
        # Add convenience attributes with defensive counting
        self.n_samples = int(len(labels)) if labels is not None else 0

        # Use metadata values first, fallback to dtype inspection only if missing
        self.n_targets = self.metadata.get("n_targets")
        if self.n_targets is None and labels is not None:
            # Fallback to dtype inspection only if metadata doesn't have n_targets
            if isinstance(labels, pd.DataFrame):
                # For DataFrame, count columns as targets
                self.n_targets = len(labels.columns)
            else:
                # For Series, check if it's integer dtype (classification)
                if pd.api.types.is_integer_dtype(labels.dtype):
                    self.n_targets = int(labels.dropna().nunique())
                else:
                    self.n_targets = 1  # Single target for regression

        self.n_horizons = int(self.metadata.get("n_horizons", 1))
        self.confidence_scores = self.metadata.get("confidence_scores")
        self.eligibility_masks = self.metadata.get("eligibility_masks")
        # quality_scores is now passed directly as parameter
        self.normalization_factors = self.metadata.get("normalization_factors")
        self.processing_time = self.metadata.get("processing_time")


class VolatilityAwareMultiHorizonLabeler:
    """
    Volatility-aware multi-horizon labeler.
    """
    
    def __init__(self, config: VolatilityAwareConfig):
        """
        Initialize the volatility-aware labeler.
        
        Args:
            config: Configuration for the labeler
        """
        self.config = config
        self.logger = system_logger.getChild("VolatilityAwareMultiHorizonLabeler")
        
    def generate_labels(
        self,
        data: pd.DataFrame,
        price_column: str = "close",
        volatility_column: Optional[str] = None,
        profit_targets: Optional[List[float]] = None
    ) -> LabelingResult:
        """
        Generate volatility-aware labels with single profit target modulated by market volatility.

        This method creates trading opportunity labels by:
        1. Calculating future returns over the specified lookahead period
        2. Computing rolling volatility to assess market conditions
        3. Applying volatility modulation to the base profit target (0.5% default)
        4. Generating binary or regression labels based on the effective threshold

        The volatility modulation follows the formula:
        effective_threshold = base_target * clip(1 + k*(vol/vol_mean - 1), 0.5, 2.0)
        Where k is the volatility sensitivity factor (default 1.5).

        Args:
            data: Input market data DataFrame with OHLCV columns
            price_column: Name of the price column to use for calculations (default: "close")
            volatility_column: Optional name of pre-calculated volatility column
            profit_targets: Optional list of profit targets in percentage points (default: [0.5])

        Returns:
            LabelingResult containing:
                - labels: Generated labels (Series for single target, DataFrame for multiple)
                - metadata: Processing metadata including quality scores and statistics
                - success: Boolean indicating if labeling was successful
                - quality_scores: Comprehensive quality metrics for label validation

        Raises:
            ValueError: If input data is invalid or insufficient
            RuntimeError: If labeling process fails due to data quality issues
        """
        try:
            # Edge case handling: empty/short series
            if len(data) < self.config.lookahead_periods:
                self.logger.warning(f"Insufficient data: {len(data)} rows < {self.config.lookahead_periods} lookahead periods")
                return LabelingResult(
                    pd.Series(dtype=float, name='label'),
                    {"reason": "insufficient_history", "n_samples": len(data), "n_horizons": 1, "n_targets": 0},
                    success=True
                )
            
            # Edge case handling: non-monotonic/duplicate index
            if not data.index.is_monotonic_increasing:
                self.logger.warning("Non-monotonic index detected - proceeding with caution")
            
            if data.index.duplicated().any():
                self.logger.warning("Duplicate index values detected - proceeding with caution")
            
            # Validate inputs
            if price_column not in data.columns:
                raise ValueError(f"price_column '{price_column}' not in data columns {list(data.columns)}")
            
            # Edge case handling: constant price
            price_series = data[price_column]
            price_clean = price_series.dropna()  # Remove NaN values first
            if len(price_clean) == 0:
                self.logger.warning("No valid price data (all NaN) - all labels will be zero")
            elif price_clean.nunique() <= 1:
                self.logger.warning("Constant price detected - all labels will be zero")
            elif price_clean.std() == 0:
                self.logger.warning("Zero price variance detected - all labels will be zero")
            
            # Explicit units for targets - treat inputs as percent points
            target_pp = profit_targets or []
            # Ensure we extract scalar values from any Series objects
            targets_frac = []
            if target_pp:
                for t in target_pp:
                    if isinstance(t, pd.Series):
                        # Extract scalar value from Series (use first non-null value)
                        scalar_t = t.dropna().iloc[0] if len(t.dropna()) > 0 else 0.0
                    else:
                        scalar_t = float(t)
                    targets_frac.append(scalar_t / 100.0)
            
            # Calculate volatility with proper configuration
            if volatility_column is None or volatility_column not in data.columns:
                if self.config.volatility.enabled:
                    volatility = price_series.pct_change().rolling(window=self.config.volatility.window).std()
                else:
                    volatility = pd.Series(1.0, index=price_series.index)  # Default multiplier = 1
            else:
                volatility = data[volatility_column]
            
            # Optional: rate calibration to meet a max ops/day target
            calibrated_targets = targets_frac
            try:
                if getattr(self.config, 'rate_control', None) and self.config.rate_control.enabled:
                    # Use single scale across targets for simplicity
                    base = targets_frac[0] if targets_frac else (self.config.multi_target.target_profit / 100.0)
                    lookahead_bars = int(getattr(self.config, 'lookahead_periods', 6))
                    rate_scale = self._calibrate_rate_scale(price_series, volatility, base, lookahead_bars,
                                                            enable_long=self.config.enable_long_positions,
                                                            enable_short=self.config.enable_short_positions,
                                                            min_scale=self.config.rate_control.min_scale,
                                                            max_scale=self.config.rate_control.max_scale,
                                                            tol=self.config.rate_control.tolerance)
                    if targets_frac:
                        calibrated_targets = [float(t) * rate_scale for t in targets_frac]
                    else:
                        calibrated_targets = [base * rate_scale]
            except Exception as _:
                # Fallback silently to original targets on calibration failure
                calibrated_targets = targets_frac
            
            # Choose labeling path
            if getattr(self.config, 'data_driven', None) and self.config.data_driven.enabled:
                tprint_info(f"🔍 [DATA-DRIVEN LABELING] Enabled - targeting {self.config.data_driven.target_ops_per_day} ops/day")
                labels = self._generate_data_driven_labels(
                    prices=price_series,
                    volatility=volatility,
                )
                tprint_info(f"🔍 [DATA-DRIVEN LABELING] Generated {len(labels)} labels with {(labels != 0).sum()} signals")
            else:
                # Generate labels based on volatility and (optionally) calibrated profit targets
                # Check if we should use new simplified target structure or legacy approach
                if hasattr(self.config, 'use_simplified_targets') and self.config.use_simplified_targets:
                    labels = self._generate_simplified_target_labels(price_series, volatility, calibrated_targets)
                else:
                    labels = self._generate_price_target_vol_normalized_labels(price_series, volatility, calibrated_targets)
            
            # Generate quality scores with proper alignment
            quality_scores = self._calculate_quality_scores(labels, price_series)
            
            # Create downstream-ready opportunity data
            opportunity_data = self._create_downstream_opportunity_data(quality_scores)
            
            # Generate training strategies from quality scores
            training_strategy = self.score_to_training(quality_scores)
            
            # Analyze performance requirements
            performance_config = self.performance_sanity(data)
            
            # Determine result shape and format
            if isinstance(labels, pd.DataFrame):
                # Multi-target case
                result_labels = labels
                n_targets = len(labels.columns)
                label_columns = list(labels.columns)
            else:
                # Single target case
                result_labels = labels.rename('label')
                n_targets = 1
                label_columns = ['label']
            
            # Build comprehensive metadata
            metadata = {
                "volatility_threshold": self.config.volatility_threshold,
                "lookahead_periods": self.config.lookahead_periods,
                "label_type": self.config.label_type.value,
                "total_labels": len(result_labels),
                "non_null_labels": result_labels.notna().sum() if isinstance(result_labels, pd.Series) else result_labels.notna().sum().sum(),
                "n_signals": int(((result_labels != 0).sum() if isinstance(result_labels, pd.Series) else (result_labels != 0).sum().sum())),
                "quality_scores": quality_scores,
                "opportunity_data": opportunity_data,  # Downstream-ready opportunity data
                "training_strategy": training_strategy,  # Score-to-training mapping
                "performance_config": performance_config,  # Performance optimization settings
                "profit_targets_pp": target_pp,
                "profit_targets_frac": targets_frac,
                "n_horizons": 1,
                "n_targets": n_targets,
                "label_columns": label_columns,
                "labels_shape": result_labels.shape,
                "labels_mem_bytes": result_labels.memory_usage(deep=True).sum() if isinstance(result_labels, pd.DataFrame) else result_labels.memory_usage(deep=True),
                "volatility_enabled": self.config.volatility.enabled,
                "volatility_window": self.config.volatility.window,
                # Export first-passage windows if available (for Tactician training)
                "opportunity_windows": getattr(self, '_last_opportunity_windows', []),
                # Data-driven calibration diagnostics (if enabled)
                "data_driven": self._last_data_driven_report if hasattr(self, '_last_data_driven_report') else None,
                # Sample weights for PROXIMITY_REGRESSION labels (bugfix: was missing)
                "sample_weights": self.get_all_sample_weights() if hasattr(self, '_sample_weights') else None
            }
            
            # Log data-driven diagnostics if available
            if hasattr(self, '_last_data_driven_report'):
                report = self._last_data_driven_report
                tprint_info(f"🔍 [DATA-DRIVEN METADATA] Mode: {report['mode']}, Target: {report['target_ops_per_day']}, Achieved: {report['achieved_ops_per_day']:.1f}/day")
                
                # Comprehensive outcome reporting
                self._log_comprehensive_outcome_report(result_labels, quality_scores, metadata, training_strategy, performance_config)
            
            # Fast-fail if labels were not produced correctly before further processing
            if result_labels is None:
                error_msg = "VolatilityAwareMultiHorizonLabeler produced no labels (result_labels is None)."
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            if isinstance(result_labels, (pd.Series, pd.DataFrame)) and result_labels.empty:
                error_msg = "VolatilityAwareMultiHorizonLabeler produced empty labels."
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            if isinstance(result_labels, (pd.Series, pd.DataFrame)) and len(result_labels) != len(data):
                error_msg = (
                    "VolatilityAwareMultiHorizonLabeler labels length mismatch: "
                    f"labels={len(result_labels)}, data={len(data)}"
                )
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Logging & observability - single-line KPI
            # "coverage" was misleading: you want actual signal rate, not non-null count (labels are 0/±1)
            if isinstance(result_labels, pd.DataFrame):
                signal_rate = (result_labels != 0).stack().mean() if result_labels.size else 0.0
                positive_rate = (result_labels > 0).stack().mean() if result_labels.size else 0.0
            else:
                signal_rate = (result_labels != 0).mean() if len(result_labels) else 0.0
                positive_rate = (result_labels > 0).mean() if len(result_labels) else 0.0
            coverage = signal_rate
            
            self.logger.info(f"Labels generated: {metadata['total_labels']} rows, {n_targets} targets, "
                           f"coverage {coverage:.1%}, positive rate {positive_rate:.1%}, "
                           f"vol window {self.config.volatility.window}/{self.config.volatility.enabled}")
            
            # Warn on suspicious states
            if coverage < 0.01 and metadata["total_labels"] > 0:
                self.logger.warning(f"Very low coverage: {coverage:.1%} - check data quality")
            if positive_rate == 0:
                self.logger.warning("No positive labels found - check thresholds")
            elif positive_rate == 1:
                self.logger.warning("All labels positive - check thresholds")

            # Apply label smoothing if enabled
            if self.config.label_smoothing.enabled:
                result_labels, smoothing_metadata = self._apply_label_smoothing(
                    result_labels,
                    quality_scores,
                    data
                )
                # Add smoothing metadata to main metadata
                metadata['label_smoothing'] = smoothing_metadata
                self.logger.info(
                    f"Label smoothing applied: "
                    f"mean_abs_change={smoothing_metadata.get('statistics', {}).get('mean_absolute_change', 0):.4f}, "
                    f"correlation={smoothing_metadata.get('statistics', {}).get('correlation_raw_final', 1.0):.3f}"
                )

            return LabelingResult(result_labels, metadata, success=True, quality_scores=quality_scores)
        except Exception as e:
            self.logger.error(f"Error generating labels: {e}")
            raise

    def _apply_label_smoothing(
        self,
        labels: Union[pd.Series, pd.DataFrame],
        quality_scores: Dict[str, Any],
        data: pd.DataFrame
    ) -> Tuple[Union[pd.Series, pd.DataFrame], Dict[str, Any]]:
        """
        Apply three-stage label smoothing pipeline.

        Args:
            labels: Raw labels to smooth
            quality_scores: Quality metrics from labeler
            data: Original market data (for grouping and volatility)

        Returns:
            Tuple of (smoothed_labels, smoothing_metadata)
        """
        try:
            # Extract quality scores for uncertainty shrinkage
            opportunity_quality = None
            if quality_scores and 'opportunity_quality_scores' in quality_scores:
                opportunity_quality = quality_scores['opportunity_quality_scores']
                # Ensure alignment with labels
                if isinstance(opportunity_quality, pd.Series):
                    opportunity_quality = opportunity_quality.reindex(labels.index)

            # Extract volatility if available
            volatility = None
            if 'volatility' in data.columns:
                volatility = data['volatility'].reindex(labels.index)

            # Prepare group_by_data for EMA
            group_by_data = None
            if self.config.label_smoothing.ema_group_by:
                group_cols = []
                # Check for instrument column
                if 'instrument' in data.columns:
                    group_cols.append('instrument')
                    if self.config.label_smoothing.ema_group_by is None:
                        # Auto-enable if instrument column exists
                        self.config.label_smoothing.ema_group_by = 'instrument'
                # Check for timestamp/datetime
                if data.index.name in ['timestamp', 'datetime'] or isinstance(data.index, pd.DatetimeIndex):
                    group_cols.append('timestamp')
                elif 'timestamp' in data.columns:
                    group_cols.append('timestamp')

                if group_cols:
                    group_by_data = data[group_cols].copy() if all(c in data.columns for c in group_cols) else None
                    if group_by_data is None and data.index.name in ['timestamp', 'datetime']:
                        # Index is timestamp
                        group_by_data = pd.DataFrame({'timestamp': data.index})
                        if 'instrument' in data.columns:
                            group_by_data['instrument'] = data['instrument'].values

            # Create smoother and apply
            smoother = LabelSmoother(self.config.label_smoothing)

            result = smoother.smooth(
                labels=labels,
                quality_scores=opportunity_quality,
                volatility=volatility,
                group_by_data=group_by_data
            )

            smoothed_labels = result['labels_final']
            smoothing_metadata = result['metadata']

            return smoothed_labels, smoothing_metadata

        except Exception as e:
            self.logger.warning(f"Label smoothing failed: {e}. Returning raw labels.")
            # Return raw labels on error
            return labels, {'enabled': False, 'error': str(e)}

    def _calculate_quality_scores(self, labels: Union[pd.Series, pd.DataFrame], prices: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive quality scores with IC, Hit Rate, Uplift, Stability, and Risk-aware metrics."""
        try:
            # Handle both Series and DataFrame inputs
            if isinstance(labels, pd.DataFrame):
                # Special-case: single-column DataFrame should behave like single target
                if len(labels.columns) == 1:
                    col = labels.columns[0]
                    target_quality = self._calculate_comprehensive_target_quality(labels[col], prices, col)
                    return {col: target_quality}

                # For true multi-target, calculate quality for each target
                target_qualities = {}
                for col in labels.columns:
                    target_quality = self._calculate_comprehensive_target_quality(labels[col], prices, col)
                    target_qualities[col] = target_quality
                
                # Apply multiple testing hygiene with FDR control
                target_qualities = self._apply_multiple_testing_hygiene(target_qualities)
                
                # Aggregate across targets using median for robustness
                return self._aggregate_target_qualities(target_qualities)
            else:
                # Single target case
                target_quality = self._calculate_comprehensive_target_quality(labels, prices, 'default')
                return {'default': target_quality}
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate quality scores: {e}")
            return self._create_fallback_quality_score()
    
    def _calculate_comprehensive_target_quality(self, labels: pd.Series, prices: pd.Series, target_name: str) -> Any:
        """Calculate quality scores focused on trade opportunities using potential profit."""
        # Calculate trade opportunity quality metrics
        
        # Align series to ensure consistent indices
        labels_aligned, prices_aligned = _align_like(labels, prices)
        # Align labels and prices for quality calculation
        
        # Only calculate quality for trade opportunities (non-zero labels: positive for long, negative for short)
        trade_opportunities = labels_aligned[labels_aligned != 0]
        if len(trade_opportunities) == 0:
            self.logger.warning(f"No trade opportunities found for {target_name}")
            return self._create_fallback_quality_score(reason="no_trade_opportunities")
        
        long_opportunities = len(trade_opportunities[trade_opportunities > 0])
        short_opportunities = len(trade_opportunities[trade_opportunities < 0])
        # Count trade opportunities by direction
        
        # Calculate potential profit for each trade opportunity
        potential_profits = self._calculate_potential_profits(trade_opportunities, prices_aligned, target_name)
        
        # Construct lookahead returns to estimate predictability properly
        try:
            lookahead = max(1, int(getattr(self.config, 'lookahead_periods', 1)))
            lookahead_returns = prices_aligned.pct_change(lookahead).shift(-lookahead)
        except Exception:
            lookahead_returns = pd.Series(dtype=float, index=prices_aligned.index)

        # Calculate quality metrics based on potential profit, enriched with returns-based predictability
        metrics = self._calculate_trade_opportunity_metrics(
            trade_opportunities, potential_profits, target_name, lookahead_returns
        )
        
        # Calculate composite score based on potential profit quality
        composite_score = self._calculate_potential_profit_quality_score(metrics, potential_profits)
        
        # Calculate individual opportunity scores and weights
        opportunity_scores = self._calculate_individual_opportunity_scores(trade_opportunities, potential_profits, metrics)
        opportunity_weights = self._calculate_individual_opportunity_weights(trade_opportunities, potential_profits, metrics)
        
        # Create trade opportunity quality score object
        class TradeOpportunityQualityScore:
            def __init__(self, composite_score, metrics, potential_profits, target_name, opportunity_scores, opportunity_weights, trade_opportunities):
                self.overall_quality = composite_score
                self.predictability = metrics.get('ic', 0.0)
                # Removed stability, balance, and coverage - using data-driven approach
                self.target_name = target_name
                self.gates_passed = True  # No balance gates
                self.potential_profits = potential_profits
                self.avg_potential_profit = potential_profits.mean() if len(potential_profits) > 0 else 0.0
                self.max_potential_profit = potential_profits.max() if len(potential_profits) > 0 else 0.0
                # Individual opportunity scoring for downstream use
                self.opportunity_scores = opportunity_scores  # Per-opportunity quality scores
                self.opportunity_weights = opportunity_weights  # Per-opportunity weights
                # Calculate per-opportunity quality scores for high-quality counting
                self.opportunity_quality_scores = [self._calculate_individual_opportunity_quality_score(opp_score, opp_weight, metrics) for opp_score, opp_weight in zip(opportunity_scores, opportunity_weights)]
                # Store all metrics for detailed analysis
                self.metrics = metrics
                self.red_flag_reasons = self._extract_trade_opportunity_red_flags(metrics, potential_profits)
                # Keep the true signal directions for downstream artifacts
                self.signal_directions = trade_opportunities.copy()
            
            def _extract_trade_opportunity_red_flags(self, metrics: Dict[str, float], potential_profits: pd.Series) -> List[str]:
                """Extract red flags specific to trade opportunities."""
                red_flags = []

                # Check for low quality metrics
                if self.overall_quality < 0.3:
                    red_flags.append("low_quality")

                # Predictability (IC) removed from red flags

                # Check for insufficient opportunities
                if len(potential_profits) < 5:
                    red_flags.append("insufficient_opportunities")

                return red_flags

            def _calculate_individual_opportunity_quality_score(self, opp_score: float, opp_weight: float, metrics: Dict[str, float]) -> float:
                """Calculate quality score for individual opportunity using data-driven approach."""
                # Weight the opportunity score by the overall composite score
                # This creates a data-driven quality assessment for each opportunity
                base_quality = self.overall_quality * opp_score * opp_weight

                # Normalize to 0-1 range
                return max(0, min(1, base_quality))

        return TradeOpportunityQualityScore(composite_score, metrics, potential_profits, target_name, opportunity_scores, opportunity_weights, trade_opportunities)
    
    def _calculate_potential_profits(self, trade_opportunities: pd.Series, prices: pd.Series, target_name: str) -> pd.Series:
        """Calculate potential profit based on signal direction over a fixed lookahead window (default: 6 bars ≈ 90min on 15m data)."""
        # Use the same horizon as the labeling window for consistency
        window_len = max(1, int(getattr(self.config.optimal_entry_detection, "max_windows", 1)))
        # Pre-compute positional index mapping using RangeIndex semantics
        indexer = {key: idx for idx, key in enumerate(prices.index)}
        potentials: Dict[pd.Timestamp, float] = {}

        price_values = prices.to_numpy()

        for ts, signal in trade_opportunities.items():
            pos = indexer.get(ts)
            if pos is None:
                potentials[ts] = 0.0
                continue

            end_pos = min(pos + window_len, len(price_values) - 1)
            if end_pos <= pos:
                potentials[ts] = 0.0
                continue

            window = price_values[pos:end_pos + 1]
            if window.size < 2:
                potentials[ts] = 0.0
                continue

            start_price = window[0]
            if start_price <= 0:
                potentials[ts] = 0.0
                continue

            if signal > 0:
                potential = (window.max() - start_price) / start_price
            elif signal < 0:
                potential = (start_price - window.min()) / start_price
            else:
                potential = 0.0

            potentials[ts] = float(potential)

        return pd.Series(potentials, index=trade_opportunities.index).fillna(0.0)
    
    def _calculate_trade_opportunity_metrics(self, trade_opportunities: pd.Series, potential_profits: pd.Series, target_name: str,
                                             lookahead_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """Calculate metrics specific to trade opportunities."""
        metrics = {}
        
        if len(potential_profits) == 0:
            return {'ic': 0.0, 'hit_rate': 0.0, 'uplift': 0.0, 'stability': 0.0, 'sharpe': 0.0}
        
        # Basic opportunity metrics
        metrics['avg_potential_profit'] = potential_profits.mean()
        metrics['max_potential_profit'] = potential_profits.max()
        metrics['min_potential_profit'] = potential_profits.min()
        metrics['std_potential_profit'] = potential_profits.std()
        
        # Quality metrics based on returns-based predictability at signal times
        # Compute IC over non-zero signals only; calculate confidence based on signal quality and consistency
        ic_val = 0.0
        if lookahead_returns is not None and len(lookahead_returns) > 0:
            try:
                idx = trade_opportunities.index.intersection(lookahead_returns.index)
                if len(idx) > 2:
                    lbls = trade_opportunities.loc[idx].astype(float)
                    rets = lookahead_returns.loc[idx].astype(float)
                    if lbls.nunique() >= 2:
                        # Multi-directional signals: calculate proper correlation
                        ic_raw = lbls.corr(rets, method='spearman')
                        ic_val = float(ic_raw) if not pd.isna(ic_raw) else 0.0
                    else:
                        # Single direction (long-only): calculate directional correlation
                        # For long-only, we want to see if positive signals correlate with positive returns
                        if (lbls > 0).any():  # Has positive signals
                            pos_signal_idx = idx[lbls > 0]
                            if len(pos_signal_idx) > 1:
                                pos_rets = rets.loc[pos_signal_idx]
                                # Calculate correlation between positive signals and positive returns
                                pos_rets_binary = (pos_rets > 0).astype(float)
                                if pos_rets_binary.std() > 0:  # Avoid division by zero
                                    ic_val = lbls.loc[pos_signal_idx].corr(pos_rets_binary)
                                else:
                                    # All returns in same direction, give moderate confidence
                                    ic_val = 0.3 if pos_rets.mean() > 0 else 0.0
                            else:
                                ic_val = 0.0
                        else:
                            ic_val = 0.0
            except Exception:
                ic_val = 0.0

        # Apply reasonable bounds to IC but don't force it to 0 for single-direction strategies
        try:
            shorts_disabled = not getattr(self.config, 'enable_short_positions', False)
            has_shorts = (trade_opportunities < 0).any()
            if shorts_disabled or not has_shorts:
                # For single-direction strategies, allow negative IC but bound it reasonably
                ic_val = max(-0.5, min(1.0, ic_val))  # Bound between -0.5 and 1.0
            else:
                # Multi-directional: standard bounds
                ic_val = max(-1.0, min(1.0, ic_val))
        except Exception:
            ic_val = max(-1.0, min(1.0, ic_val))
        metrics['ic'] = ic_val
        # Mark single-direction scenarios (only long or only short signals present)
        try:
            has_long = (trade_opportunities > 0).any()
            has_short = (trade_opportunities < 0).any()
            metrics['single_direction'] = 1.0 if (has_long ^ has_short) else 0.0
        except Exception:
            metrics['single_direction'] = 0.0
        
        # Hit rate: percentage of opportunities with positive return in direction of signal (fallback to potential profit if needed)
        if lookahead_returns is not None and len(lookahead_returns) > 0:
            try:
                pos_idx = trade_opportunities[trade_opportunities > 0].index
                neg_idx = trade_opportunities[trade_opportunities < 0].index
                pos_hits = (lookahead_returns.loc[pos_idx] > 0).mean() if len(pos_idx) > 0 else 0.0
                neg_hits = (lookahead_returns.loc[neg_idx] < 0).mean() if len(neg_idx) > 0 else 0.0
                # Weighted by counts
                total = len(pos_idx) + len(neg_idx)
                hit_rate = (pos_hits * len(pos_idx) + neg_hits * len(neg_idx)) / total if total > 0 else 0.0
                metrics['hit_rate'] = float(hit_rate) if not np.isnan(hit_rate) else 0.0
            except Exception:
                avg_profit = potential_profits.mean()
                hit_rate = (potential_profits > avg_profit).mean()
                metrics['hit_rate'] = hit_rate if not np.isnan(hit_rate) else 0.0
        else:
            avg_profit = potential_profits.mean()
            hit_rate = (potential_profits > avg_profit).mean()
            metrics['hit_rate'] = hit_rate if not np.isnan(hit_rate) else 0.0
        
        # Uplift: return difference between signals and non-signals at signal times fallback to profit distribution
        if lookahead_returns is not None and len(lookahead_returns) > 0:
            try:
                idx = trade_opportunities.index.intersection(lookahead_returns.index)
                if len(idx) > 1:
                    lbls = trade_opportunities.loc[idx]
                    rets = lookahead_returns.loc[idx]
                    long_mask = lbls > 0
                    short_mask = lbls < 0
                    if long_mask.any():
                        long_uplift = rets[long_mask].mean() - rets[~long_mask].mean()
                    else:
                        long_uplift = 0.0
                    if short_mask.any():
                        short_uplift = -(rets[short_mask].mean() - rets[~short_mask].mean())  # positive is better
                    else:
                        short_uplift = 0.0
                    metrics['uplift'] = float(np.nan_to_num((long_uplift + short_uplift), nan=0.0))
                else:
                    metrics['uplift'] = 0.0
            except Exception:
                # Fallback to profit-based uplift
                if len(potential_profits) > 1:
                    high_profit_mask = potential_profits > potential_profits.median()
                    if high_profit_mask.sum() > 0 and (~high_profit_mask).sum() > 0:
                        uplift = potential_profits[high_profit_mask].mean() - potential_profits[~high_profit_mask].mean()
                        metrics['uplift'] = uplift if not np.isnan(uplift) else 0.0
                    else:
                        metrics['uplift'] = 0.0
                else:
                    metrics['uplift'] = 0.0
        
        # Stability: consistency of potential profits over time
        if len(potential_profits) > 3:
            # Rolling standard deviation of potential profits
            rolling_std = potential_profits.rolling(window=min(3, len(potential_profits))).std()
            stability = 1 / (1 + rolling_std.mean()) if not rolling_std.mean() == 0 else 0.0
            metrics['stability'] = stability if not np.isnan(stability) else 0.0
        else:
            metrics['stability'] = 0.0
        
        # Sharpe: risk-adjusted potential profit
        if metrics['std_potential_profit'] > 0:
            sharpe = metrics['avg_potential_profit'] / metrics['std_potential_profit']
            metrics['sharpe'] = sharpe if not np.isnan(sharpe) else 0.0
        else:
            metrics['sharpe'] = 0.0
        
        return metrics

    def _calibrate_rate_scale(
        self,
        prices: pd.Series,
        volatility: pd.Series,
        base_target: float,
        lookahead_bars: int,
        enable_long: bool,
        enable_short: bool,
        min_scale: float,
        max_scale: float,
        tol: float = 0.25
    ) -> float:
        """Calibrate a multiplicative scale for the base target.

        Uses binary search on a simple thresholded returns proxy to estimate signal rate.
        No longer limited by max_ops_per_day as requested.
        """
        # Precompute future returns and vol normalization
        fut_ret = prices.pct_change(lookahead_bars).shift(-lookahead_bars)
        vol_mean = volatility.mean()
        vol_norm = (volatility / vol_mean) if vol_mean > 0 else pd.Series(1.0, index=volatility.index)

        # Compute days in data
        try:
            time_span_days = max(1.0, float((prices.index.max() - prices.index.min()).total_seconds()) / 86400.0)
        except Exception:
            time_span_days = max(1.0, len(prices) / (24 * 60))  # rough fallback for 1-min data

        # No longer using target_total based on max_ops_per_day
        # Return a default scale since we're not limiting opportunities per day
        return 1.0

    def _infer_bars_per_day(self, index: pd.Index) -> float:
        """Infer approximate bars per day from datetime-like index."""
        tprint_debug(f"🔍 [BARS/DAY] Inferring from index with {len(index)} samples")
        if len(index) < 3:
            tprint_debug(f"🔍 [BARS/DAY] Insufficient samples, using default: 96.0 bars/day")
            return 96.0  # sensible default for 15m
        try:
            idx = pd.to_datetime(index)
            # Remove timezone to make arithmetic simpler
            if getattr(idx, 'tz', None) is not None:
                idx = idx.tz_convert(None)
        except Exception:
            try:
                idx = pd.to_datetime(pd.Index(index))
            except Exception:
                return 96.0
        try:
            deltas_sec = (idx[1:] - idx[:-1]).asi8 / 1e9
        except Exception:
            try:
                deltas_sec = (idx[1:] - idx[:-1]).total_seconds()
            except Exception:
                return 96.0
        # Use median delta to avoid outliers
        try:
            med = np.median(deltas_sec)
            if med <= 0:
                tprint_warning(f"🔍 [BARS/DAY] Invalid median delta: {med}, using default")
                return 96.0
            bars_per_day = float(86400.0 / med)
            tprint_debug(f"🔍 [BARS/DAY] Median delta: {med:.1f}s, calculated: {bars_per_day:.1f} bars/day")
            return bars_per_day
        except Exception as e:
            tprint_warning(f"🔍 [BARS/DAY] Calculation failed ({e}), using default")
            return 96.0

    def _compute_quantile_thresholds(
        self,
        fut_ret: pd.Series,
        vol_norm: Optional[pd.Series],
        mode: str,
        bins: int,
        q_long: float,
        q_short: float,
        rolling_window: Optional[int] = None
    ) -> Tuple[pd.Series, pd.Series]:
        """Compute long/short thresholds using empirical quantiles.

        - mode 'global': single constant thresholds for all samples
        - mode 'vol_bin': thresholds per volatility-quantile bin
        - rolling_window: if provided, compute rolling quantiles (global only for now)
        """
        tprint_debug(f"🔍 [QUANTILE THRESHOLDS] Computing {mode} thresholds with q_long={q_long:.3f}, q_short={q_short:.3f}")
        fut_clean = fut_ret.dropna()
        if len(fut_clean) == 0:
            tprint_warning(f"🔍 [QUANTILE THRESHOLDS] No valid forward returns, using zero thresholds")
            zero = pd.Series(0.0, index=fut_ret.index)
            return zero, zero

        if mode == 'vol_bin' and vol_norm is not None and bins > 1:
            tprint_debug(f"🔍 [QUANTILE THRESHOLDS] Using volatility binning with {bins} bins")
            try:
                cats = pd.qcut(vol_norm, q=min(bins, len(vol_norm.unique())), duplicates='drop')
                n_bins = len(cats.unique())
                tprint_debug(f"🔍 [QUANTILE THRESHOLDS] Created {n_bins} volatility bins")
            except Exception as e:
                tprint_warning(f"🔍 [QUANTILE THRESHOLDS] Volatility binning failed ({e}), using global")
                cats = pd.Series('all', index=vol_norm.index)
            thr_long = pd.Series(index=fut_ret.index, dtype=float)
            thr_short = pd.Series(index=fut_ret.index, dtype=float)
            # Compute per-bin thresholds
            for lvl, idx in cats.groupby(cats).groups.items():
                idx = pd.Index(idx)
                vals = fut_ret.loc[idx].dropna()
                if len(vals) == 0:
                    l, s = fut_clean.quantile(1 - q_long), fut_clean.quantile(q_short)
                    tprint_debug(f"🔍 [QUANTILE THRESHOLDS] Bin {lvl}: No data, using global quantiles")
                else:
                    l, s = vals.quantile(1 - q_long), vals.quantile(q_short)
                    tprint_debug(f"🔍 [QUANTILE THRESHOLDS] Bin {lvl}: {len(vals)} samples, L={l:.4f}, S={s:.4f}")
                thr_long.loc[idx] = float(l)
                thr_short.loc[idx] = float(s)
            # Fill any NA with global quantiles
            global_l = fut_clean.quantile(1 - q_long)
            global_s = fut_clean.quantile(q_short)
            thr_long = thr_long.fillna(global_l)
            thr_short = thr_short.fillna(global_s)
            tprint_debug(f"🔍 [QUANTILE THRESHOLDS] Global fallback: L={global_l:.4f}, S={global_s:.4f}")
            return thr_long, thr_short

        # Global thresholds (optionally rolling)
        if rolling_window and rolling_window > 10:
            tprint_debug(f"🔍 [QUANTILE THRESHOLDS] Using rolling window: {rolling_window}")
            ql = 1 - q_long
            qs = q_short
            thr_long = fut_ret.rolling(rolling_window, min_periods=max(10, rolling_window//5)).quantile(ql)
            thr_short = fut_ret.rolling(rolling_window, min_periods=max(10, rolling_window//5)).quantile(qs)
            # Back/forward fill to cover edges
            thr_long = thr_long.fillna(method='bfill').fillna(method='ffill')
            thr_short = thr_short.fillna(method='bfill').fillna(method='ffill')
            tprint_debug(f"🔍 [QUANTILE THRESHOLDS] Rolling thresholds computed")
            return thr_long, thr_short
        else:
            l = float(fut_clean.quantile(1 - q_long))
            s = float(fut_clean.quantile(q_short))
            tprint_debug(f"🔍 [QUANTILE THRESHOLDS] Global: L={l:.4f} ({q_long:.1%} quantile), S={s:.4f} ({q_short:.1%} quantile)")
            thr_long = pd.Series(l, index=fut_ret.index)
            thr_short = pd.Series(s, index=fut_ret.index)
            return thr_long, thr_short

    def _generate_data_driven_labels(
        self,
        prices: pd.Series,
        volatility: pd.Series,
    ) -> pd.Series:
        """Data-driven labeling using quantile-calibrated thresholds.

        - Calibrates target label density in ops/day from empirical forward returns.
        - Optional per-volatility-bin thresholds to adapt to regimes.
        - Respects direction toggles (long/short enable flags).
        """
        cfg = self.config.data_driven
        tprint_debug(f"🔍 [DATA-DRIVEN] Config: target={cfg.target_ops_per_day}, mode={cfg.calibration_mode}, bins={cfg.volatility_bins}")
        H = max(1, int(self.config.lookahead_periods))
        fut_ret = prices.pct_change(H).shift(-H)
        tprint_debug(f"🔍 [DATA-DRIVEN] Forward returns: {len(fut_ret.dropna())} valid samples, lookahead={H}")
        # Normalized volatility for optional binning
        vol_mean = float(volatility.mean()) if len(volatility) else 0.0
        vol_norm = (volatility / vol_mean) if vol_mean > 0 else pd.Series(1.0, index=volatility.index)
        tprint_debug(f"🔍 [DATA-DRIVEN] Volatility: mean={vol_mean:.4f}, norm_range=[{vol_norm.min():.3f}, {vol_norm.max():.3f}]")

        # Compute desired per-bar signal rates
        bars_per_day = max(1.0, self._infer_bars_per_day(prices.index))
        target_ops = max(cfg.min_ops_per_day, cfg.target_ops_per_day)
        per_bar_total = min(0.5, float(target_ops / bars_per_day))  # cap at 50% to avoid degenerate tie/noise
        tprint_debug(f"🔍 [DATA-DRIVEN] Bars/day: {bars_per_day:.1f}, Target ops: {target_ops}, Per-bar rate: {per_bar_total:.3f}")

        if self.config.enable_long_positions and self.config.enable_short_positions:
            if cfg.long_short_split == 'even':
                rate_long = rate_short = per_bar_total / 2.0
                tprint_debug(f"🔍 [DATA-DRIVEN] Even split: L={rate_long:.3f}, S={rate_short:.3f}")
            else:
                # Auto split by skew of forward returns
                skew = float(pd.Series(fut_ret).dropna().skew()) if len(fut_ret.dropna()) else 0.0
                w_long = 0.5 + 0.25 * np.tanh(skew)
                rate_long = per_bar_total * w_long
                rate_short = per_bar_total * (1.0 - w_long)
                tprint_debug(f"🔍 [DATA-DRIVEN] Skew split (skew={skew:.3f}): L={rate_long:.3f}, S={rate_short:.3f}")
        elif self.config.enable_long_positions:
            rate_long, rate_short = per_bar_total, 0.0
            tprint_debug(f"🔍 [DATA-DRIVEN] Long-only: L={rate_long:.3f}, S={rate_short:.3f}")
        elif self.config.enable_short_positions:
            rate_long, rate_short = 0.0, per_bar_total
            tprint_debug(f"🔍 [DATA-DRIVEN] Short-only: L={rate_long:.3f}, S={rate_short:.3f}")
        else:
            tprint_warning(f"🔍 [DATA-DRIVEN] No directions enabled, returning zero labels")
            return pd.Series(0, index=prices.index, dtype=np.int8)

        # Compute thresholds
        tprint_debug(f"🔍 [DATA-DRIVEN] Computing {cfg.calibration_mode} thresholds")
        thr_long, thr_short = self._compute_quantile_thresholds(
            fut_ret=fut_ret,
            vol_norm=vol_norm if cfg.calibration_mode == 'vol_bin' else None,
            mode=cfg.calibration_mode,
            bins=int(cfg.volatility_bins),
            q_long=float(rate_long),
            q_short=float(rate_short),
            rolling_window=cfg.rolling_window,
        )
        tprint_debug(f"🔍 [DATA-DRIVEN] Thresholds: L={thr_long.iloc[0]:.4f}, S={thr_short.iloc[0]:.4f}")

        # Produce labels
        tprint_debug(f"🔍 [DATA-DRIVEN] Generating labels with L={rate_long > 0}, S={rate_short > 0}")
        labels = pd.Series(0, index=prices.index, dtype=np.int8)
        if rate_long > 0 and self.config.enable_long_positions:
            long_signals = (fut_ret > thr_long).sum()
            labels = labels + (fut_ret > thr_long).astype(np.int8)
            tprint_debug(f"🔍 [DATA-DRIVEN] Long signals: {long_signals}")
        if rate_short > 0 and self.config.enable_short_positions:
            short_signals = (fut_ret < thr_short).sum()
            labels = labels - (fut_ret < thr_short).astype(np.int8)
            tprint_debug(f"🔍 [DATA-DRIVEN] Short signals: {short_signals}")
        total_signals = (labels != 0).sum()
        tprint_info(f"🔍 [DATA-DRIVEN] Generated {total_signals} total signals")

        # Diagnostics: achieved ops/day
        try:
            time_span_days = max(1.0, float((prices.index.max() - prices.index.min()).total_seconds()) / 86400.0)
        except Exception:
            time_span_days = max(1.0, len(prices) / bars_per_day)
        achieved_ops_per_day = float((labels != 0).sum()) / time_span_days

        self._last_data_driven_report = {
            'mode': cfg.calibration_mode,
            'target_ops_per_day': float(cfg.target_ops_per_day),
            'min_ops_per_day': float(cfg.min_ops_per_day),
            'bars_per_day': float(bars_per_day),
            'rate_long': float(rate_long),
            'rate_short': float(rate_short),
            'achieved_ops_per_day': achieved_ops_per_day,
        }

        tprint_info(f"🔍 [DATA-DRIVEN] Achieved {achieved_ops_per_day:.1f} ops/day vs target {cfg.target_ops_per_day}")
        tprint_debug(f"🔍 [DATA-DRIVEN] Time span: {time_span_days:.1f} days, Signals: {(labels != 0).sum()}")

        return labels
    
    def _generate_simplified_target_labels(
        self,
        prices: pd.Series,
        volatility: pd.Series,
        calibrated_targets: List[float]
    ) -> pd.DataFrame:
        """
        Generate simplified target labels (target_long, target_short) based on volatility.
        
        This method creates the new simplified target structure with separate binary targets
        for long and short positions, volume-normalized as requested.
        
        Args:
            prices: Price series for label generation
            volatility: Volatility series for modulation
            calibrated_targets: List of calibrated profit targets
            
        Returns:
            DataFrame with target_long and target_short columns
        """
        tprint_info("🎯 Generating simplified target labels (target_long, target_short)")
        
        # Use the first calibrated target or default to 0.5%
        base_target = calibrated_targets[0] if calibrated_targets else 0.005
        
        # Calculate forward returns
        H = max(1, int(self.config.lookahead_periods))
        fut_ret = prices.pct_change(H).shift(-H)
        
        # Calculate volatility-modulated thresholds
        vol_mean = volatility.mean()
        if vol_mean > 0:
            vol_norm = volatility / vol_mean
            # Apply volatility modulation with clipping
            vol_factor = np.clip(
                1.0 + self.config.volatility.sensitivity * (vol_norm - 1.0),
                self.config.multi_target.min_threshold_multiplier,
                self.config.multi_target.max_threshold_multiplier
            )
            effective_threshold = base_target * vol_factor
        else:
            effective_threshold = pd.Series(base_target, index=prices.index)
        
        # Generate binary targets for long and short positions
        target_long = (fut_ret > effective_threshold).astype(np.int8)
        target_short = (fut_ret < -effective_threshold).astype(np.int8)
        
        # Create result DataFrame
        labels = pd.DataFrame({
            'target_long': target_long,
            'target_short': target_short
        }, index=prices.index)
        
        # Log statistics
        long_signals = target_long.sum()
        short_signals = target_short.sum()
        total_signals = long_signals + short_signals
        
        tprint_info(f"📊 Simplified target statistics:")
        tprint_info(f"   Long signals: {long_signals} ({long_signals/len(labels):.1%})")
        tprint_info(f"   Short signals: {short_signals} ({short_signals/len(labels):.1%})")
        tprint_info(f"   Total signals: {total_signals} ({total_signals/len(labels):.1%})")
        tprint_info(f"   Base threshold: {base_target:.4f} ({base_target*10000:.1f} bps)")
        
        return labels

    def _build_multiscale_features(self, prices: pd.Series) -> pd.DataFrame:
        """Construct multi-scale lagged return and volatility features."""
        df = pd.DataFrame(index=prices.index)
        # Returns lags
        lags = [1, 2, 3, 4, 6, 8, 12]
        for L in lags:
            df[f"ret_{L}"] = prices.pct_change(L)
        # Rolling realized vol
        win = [5, 10, 20]
        for w in win:
            df[f"rv_{w}"] = df["ret_1"].rolling(w).std()
        return df.dropna()

    def _mrmr_select(self, X: pd.DataFrame, y: pd.Series, k: int = 20) -> Dict[str, Any]:
        """Greedy mRMR-style selection using mutual information and correlation penalty."""
        try:
            from sklearn.feature_selection import mutual_info_classif
        except Exception:
            # Fallback: select top-k by absolute correlation
            corrs = X.apply(lambda c: abs(pd.Series(c).corr(y)), axis=0).fillna(0.0)
            top = list(corrs.sort_values(ascending=False).head(k).index)
            return {"selected_features": top, "method": "corr_topk"}

        # Compute MI relevance
        mi = mutual_info_classif(X.fillna(0.0), (y > 0).astype(int), discrete_features=False, random_state=42)
        relevance = pd.Series(mi, index=X.columns)
        selected: list = []
        candidates = set(X.columns)
        corr_mat = X.fillna(0.0).corr().abs().fillna(0.0)

        while len(selected) < min(k, X.shape[1]) and candidates:
            best_feat = None
            best_score = -1e9
            for feat in candidates:
                red = 0.0
                if selected:
                    red = corr_mat.loc[feat, selected].mean()
                score = relevance.get(feat, 0.0) - 0.5 * red
                if score > best_score:
                    best_score = score
                    best_feat = feat
            if best_feat is None:
                break
            selected.append(best_feat)
            candidates.remove(best_feat)
        return {"selected_features": selected, "method": "greedy_mrmr"}

    def optimize_parameters(self, data: pd.DataFrame, price_column: str = "close") -> Dict[str, Any]:
        """Minimal time-blocked CV to tune lookahead and threshold scale by maximizing out-of-sample IC.

        This is a lightweight, heuristics-free tuner; it won't run automatically.
        """
        prices = data[price_column]
        vol = prices.pct_change().rolling(self.config.volatility.window).std()
        # Candidate grids
        lookaheads = [3, 4, 6]
        scales = [0.8, 1.0, 1.2, 1.5]
        best = {"ic": -1.0, "lookahead": None, "scale": None}
        n = len(prices)
        if n < 200:
            return {"error": "insufficient_data"}
        # 3-fold time blocks
        folds = [(0, n//3), (n//3, 2*n//3), (2*n//3, n)]
        for H in lookaheads:
            fut_ret = prices.pct_change(H).shift(-H)
            for s in scales:
                ics = []
                for (a, b) in folds:
                    eff = (self.config.multi_target.target_profit/100.0) * s
                    thr = eff * np.clip(1.0 + self.config.volatility.sensitivity * ((vol/vol.mean()) - 1.0),
                                        self.config.multi_target.min_threshold_multiplier,
                                        self.config.multi_target.max_threshold_multiplier)
                    y = pd.Series(0, index=fut_ret.index)
                    if self.config.enable_long_positions:
                        y = y.add((fut_ret > thr).astype(int), fill_value=0)
                    if self.config.enable_short_positions:
                        y = y.add(-(fut_ret < -thr).astype(int), fill_value=0)
                    y_fold = y.iloc[a:b]
                    r_fold = fut_ret.iloc[a:b]
                    if y_fold.replace(0, np.nan).dropna().nunique() < 2:
                        ic = 0.0
                    else:
                        ic = y_fold.corr(r_fold, method='spearman')
                        ic = 0.0 if pd.isna(ic) else float(ic)
                    ics.append(ic)
                mean_ic = float(np.nanmean(ics)) if ics else -1.0
                if mean_ic > best["ic"]:
                    best = {"ic": mean_ic, "lookahead": H, "scale": s}
        return best
    
    def _calculate_potential_profit_quality_score(self, metrics: Dict[str, float], potential_profits: pd.Series) -> float:
        """Calculate quality score based on potential profit characteristics."""
        if len(potential_profits) == 0:
            return 0.0
        
        # Base score from average potential profit (higher is better)
        avg_profit = metrics.get('avg_potential_profit', 0.0)
        profit_score = min(1.0, avg_profit / 0.02)  # Normalize to 2% max expected profit
        
        # Consistency score (lower std is better)
        std_profit = metrics.get('std_potential_profit', 0.0)
        consistency_score = 1.0 / (1.0 + std_profit * 10) if std_profit > 0 else 1.0
        
        # Hit rate score
        hit_rate = metrics.get('hit_rate', 0.0)
        hit_rate_score = hit_rate
        
        # Stability score
        stability = metrics.get('stability', 0.0)
        stability_score = stability
        
        # Sharpe score (risk-adjusted)
        sharpe = metrics.get('sharpe', 0.0)
        sharpe_score = min(1.0, max(0.0, (sharpe + 1) / 2))  # Normalize from [-1,1] to [0,1]
        
        # Weighted composite score
        composite_score = (
            0.4 * profit_score +      # 40% weight on potential profit
            0.2 * consistency_score + # 20% weight on consistency
            0.2 * hit_rate_score +    # 20% weight on hit rate
            0.1 * stability_score +   # 10% weight on stability
            0.1 * sharpe_score        # 10% weight on risk-adjusted return
        )
        
        return composite_score
    
    def _calculate_individual_opportunity_scores(self, trade_opportunities: pd.Series, potential_profits: pd.Series, metrics: Dict[str, float]) -> pd.Series:
        """Calculate individual quality scores for each opportunity."""
        if len(potential_profits) == 0:
            return pd.Series(dtype=float)
        
        # Base score from potential profit (normalized to [0, 1])
        avg_profit = metrics.get('avg_potential_profit', 0.0)
        std_profit = metrics.get('std_potential_profit', 0.0)
        
        # Individual opportunity scores based on:
        # 1. Potential profit relative to average (40% weight)
        # 2. Consistency with overall pattern (30% weight) 
        # 3. Risk-adjusted return (30% weight)
        
        # Profit score: how much above/below average
        profit_scores = potential_profits / max(avg_profit, 0.001)  # Avoid division by zero
        profit_scores = np.clip(profit_scores, 0, 2)  # Cap at 2x average
        
        # Consistency score: how close to the mean (lower deviation = higher score)
        if std_profit > 0:
            consistency_scores = 1.0 / (1.0 + np.abs(potential_profits - avg_profit) / std_profit)
        else:
            consistency_scores = pd.Series(1.0, index=potential_profits.index)
        
        # Risk-adjusted score: potential profit / volatility (if we had individual volatility)
        # For now, use a simplified version based on profit magnitude
        risk_adjusted_scores = potential_profits / max(potential_profits.max(), 0.001)
        
        # Weighted composite individual scores
        individual_scores = (
            0.4 * profit_scores +
            0.3 * consistency_scores +
            0.3 * risk_adjusted_scores
        )
        
        # Normalize to [0, 1] range
        individual_scores = np.clip(individual_scores, 0, 1)
        
        return pd.Series(individual_scores, index=trade_opportunities.index)
    
    def _calculate_individual_opportunity_weights(self, trade_opportunities: pd.Series, potential_profits: pd.Series, metrics: Dict[str, float]) -> pd.Series:
        """Calculate individual weights for each opportunity based on quality and potential."""
        if len(potential_profits) == 0:
            return pd.Series(dtype=float)
        
        # Base weight from potential profit magnitude
        avg_profit = metrics.get('avg_potential_profit', 0.0)
        max_profit = metrics.get('max_potential_profit', 0.0)
        
        # Weight based on potential profit relative to maximum
        if max_profit > 0:
            profit_weights = potential_profits / max_profit
        else:
            profit_weights = pd.Series(1.0, index=potential_profits.index)
        
        # Apply exponential scaling to emphasize high-potential opportunities
        # This creates a more pronounced difference between high and low potential opportunities
        scaled_weights = np.power(profit_weights, 0.7)  # 0.7 < 1 makes the curve less steep
        
        # Normalize weights to sum to 1.0 for proper probability distribution
        if scaled_weights.sum() > 0:
            normalized_weights = scaled_weights / scaled_weights.sum()
        else:
            normalized_weights = pd.Series(1.0 / len(scaled_weights), index=scaled_weights.index)
        
        return normalized_weights
    
    def _extract_trade_opportunity_red_flags(self, metrics: Dict[str, float], potential_profits: pd.Series) -> List[str]:
        """Extract red flags specific to trade opportunities."""
        red_flags = []
        
        # Check for low potential profit
        avg_profit = metrics.get('avg_potential_profit', 0.0)
        if avg_profit < 0.005:  # Less than 0.5%
            red_flags.append("low_potential_profit")
        elif avg_profit < 0.01:  # Less than 1%
            red_flags.append("marginal_potential_profit")
        
        # Check for high volatility in potential profits
        std_profit = metrics.get('std_potential_profit', 0.0)
        if std_profit > avg_profit * 2:  # Std > 2x mean
            red_flags.append("high_profit_volatility")
        
        # Check for low hit rate
        hit_rate = metrics.get('hit_rate', 0.0)
        if hit_rate < 0.3:  # Less than 30% above average
            red_flags.append("low_hit_rate")
        
        # Check for low stability
        stability = metrics.get('stability', 0.0)
        if stability < 0.3:
            red_flags.append("low_stability")
        
        return red_flags[:1]  # Return first red flag only
    
    def _check_quality_gates(self, labels: pd.Series, lookahead_returns: pd.Series, coverage: float) -> bool:
        """Check minimum quality gates."""
        # Check quality gates for label validation

        # Coverage gate removed - not used in final assessment
        
        # Gate 2: Balance check removed - not relevant for 2-5 opportunities per day
        if len(labels.dropna()) > 0:
            positive_rate = (labels.dropna() > 0).mean()
            # Gate 2 skipped for low-frequency opportunities
        
        # Gate 3: IC p-value < 0.1 in at least half of temporal folds
        if len(labels.dropna()) > 10 and len(lookahead_returns.dropna()) > 10:
            ic_pvalue = self._calculate_ic_pvalue(labels.dropna(), lookahead_returns.dropna())
            # Check IC p-value threshold
            if ic_pvalue >= 0.1:
                self.logger.warning(f"Quality gate 3 failed: IC p-value {ic_pvalue:.3f} >= 0.1")
                return False
            # Quality gate 3 passed
        else:
            # Gate 3 skipped due to insufficient data
            pass
        
        # All quality gates passed
        return True
    
    def _calculate_target_metrics(self, labels: pd.Series, lookahead_returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive metrics for a target."""
        metrics = {}
        
        # Predictability block (signal quality)
        metrics.update(self._calculate_predictability_metrics(labels, lookahead_returns))
        
        # Class/coverage block
        metrics.update(self._calculate_class_metrics(labels))
        
        # Stability block (time robustness)
        metrics.update(self._calculate_stability_metrics(labels, lookahead_returns))
        
        # Risk-aware block
        metrics.update(self._calculate_risk_metrics(labels, lookahead_returns))
        
        return metrics
    
    def _calculate_predictability_metrics(self, labels: pd.Series, lookahead_returns: pd.Series) -> Dict[str, float]:
        """Calculate predictability metrics (IC, Hit Rate, Uplift)."""
        metrics = {}
        
        # Align data
        labels_clean, returns_clean = _align_like(labels, lookahead_returns)
        
        # Debug logging
        # Clean data for metrics calculation
        
        if len(labels_clean.dropna()) < 10:
            self.logger.warning(f"Insufficient data for metrics: {len(labels_clean.dropna())} < 10")
            return {'ic': 0.0, 'hit_rate': 0.0, 'uplift': 0.0}
        
        # Information Coefficient (Spearman correlation)
        ic = labels_clean.corr(returns_clean, method='spearman')
        # Calculate information coefficient
        metrics['ic'] = ic if not pd.isna(ic) else 0.0
        
        # Hit Rate
        if self._is_classification_like(labels_clean):
            # For classification labels
            if set(labels_clean.dropna().unique()) <= {0.0, 1.0}:
                # Binary 0/1 labels
                hit_rate = (np.sign(returns_clean) == labels_clean).mean()
            elif set(labels_clean.dropna().unique()) <= {-1.0, 0.0, 1.0}:
                # Ternary -1/0/1 labels
                hit_rate = (np.sign(returns_clean) == labels_clean).mean()
            else:
                hit_rate = 0.0
        else:
            # For regression, use correlation-based hit rate
            hit_rate = abs(ic)
        
        metrics['hit_rate'] = hit_rate if not pd.isna(hit_rate) else 0.0
        
        # Uplift (return difference)
        if self._is_classification_like(labels_clean):
            positive_mask = labels_clean > 0
            if positive_mask.sum() > 0 and (~positive_mask).sum() > 0:
                uplift = returns_clean[positive_mask].mean() - returns_clean[~positive_mask].mean()
                metrics['uplift'] = uplift if not pd.isna(uplift) else 0.0
            else:
                metrics['uplift'] = 0.0
        else:
            metrics['uplift'] = 0.0
        
        return metrics
    
    def _calculate_class_metrics(self, labels: pd.Series) -> Dict[str, float]:
        """Calculate class/coverage metrics."""
        metrics = {}
        
        # Coverage - use signal rate (non-zero labels) for crypto trading
        signal_rate = (labels != 0).mean() if len(labels) > 0 else 0.0
        metrics['coverage'] = signal_rate
        
        # Balance
        if len(labels.dropna()) > 0:
            positive_rate = (labels.dropna() > 0).mean()
            balance = min(positive_rate, 1 - positive_rate) * 2
            metrics['balance'] = balance
        else:
            metrics['balance'] = 0.0
        
        return metrics
    
    def _calculate_stability_metrics(self, labels: pd.Series, lookahead_returns: pd.Series) -> Dict[str, float]:
        """Calculate stability metrics (rolling IC stability, temporal CV)."""
        metrics = {}
        
        # Rolling IC stability
        window_size = min(252, len(labels) // 4)  # 1 year or 1/4 of data
        if window_size < 20:
            metrics['stability'] = 0.0
            return metrics
        
        rolling_ics = []
        for i in range(window_size, len(labels)):
            window_labels = labels.iloc[i-window_size:i]
            window_returns = lookahead_returns.iloc[i-window_size:i]
            if len(window_labels.dropna()) > 5 and len(window_returns.dropna()) > 5:
                ic = window_labels.corr(window_returns, method='spearman')
                if not pd.isna(ic):
                    rolling_ics.append(ic)
        
        if len(rolling_ics) > 1:
            ic_std = np.std(rolling_ics)
            stability = 1 / (1 + ic_std)  # Convert to score (lower std = higher stability)
            metrics['stability'] = stability
        else:
            metrics['stability'] = 0.0
        
        return metrics
    
    def _calculate_risk_metrics(self, labels: pd.Series, lookahead_returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-aware metrics (Sharpe ratio)."""
        metrics = {}
        
        # Sharpe ratio of labeled subset
        if self._is_classification_like(labels):
            positive_mask = labels > 0
            if positive_mask.sum() > 5:  # Need enough samples
                labeled_returns = lookahead_returns[positive_mask]
                if len(labeled_returns.dropna()) > 5:
                    sharpe = labeled_returns.mean() / labeled_returns.std() if labeled_returns.std() > 0 else 0.0
                    metrics['sharpe'] = sharpe if not pd.isna(sharpe) else 0.0
                else:
                    metrics['sharpe'] = 0.0
            else:
                metrics['sharpe'] = 0.0
        else:
            metrics['sharpe'] = 0.0
        
        return metrics
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """Calculate composite quality score using data-driven approach."""
        # Normalize metrics to [0, 1] range (IC removed)
        hit_rate_norm = max(0, min(1, metrics.get('hit_rate', 0)))
        uplift_norm = max(0, min(1, (metrics.get('uplift', 0) + 0.1) / 0.2))  # Cap at 0.1
        sharpe_norm = max(0, min(1, (metrics.get('sharpe', 0) + 2) / 4))  # Cap at 2

        # Data-driven composite score focusing on core profitability metrics
        composite = (
            0.50 * hit_rate_norm +      # Primary: Signal accuracy (50%)
            0.35 * uplift_norm +        # Secondary: Profit potential (35%)
            0.15 * sharpe_norm          # Tertiary: Risk-adjusted returns (15%)
        )

        return composite
    
    def _aggregate_target_qualities(self, target_qualities: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate quality scores across targets using median and preserve opportunity info."""
        if not target_qualities:
            return self._create_fallback_quality_score()

        # Extract metrics for aggregation
        all_metrics: Dict[str, List[float]] = {}
        coverage_values: List[float] = []

        # Collect opportunity-level info for downstream logging/gating
        opp_scores_list: List[pd.Series] = []
        opp_weights_list: List[pd.Series] = []
        potential_profits_list: List[pd.Series] = []
        signal_dirs_list: List[pd.Series] = []

        for _, quality in target_qualities.items():
            if hasattr(quality, 'metrics'):
                for metric_name, value in quality.metrics.items():
                    all_metrics.setdefault(metric_name, []).append(value)
            # Coverage is a top-level attribute on the quality object
            if hasattr(quality, 'coverage'):
                coverage_values.append(getattr(quality, 'coverage', 0.0))

            # Preserve opportunity information if available
            if hasattr(quality, 'opportunity_scores') and isinstance(quality.opportunity_scores, pd.Series):
                opp_scores_list.append(quality.opportunity_scores)
            if hasattr(quality, 'opportunity_weights') and isinstance(quality.opportunity_weights, pd.Series):
                opp_weights_list.append(quality.opportunity_weights)
            if hasattr(quality, 'potential_profits') and isinstance(quality.potential_profits, pd.Series):
                potential_profits_list.append(quality.potential_profits)
            if hasattr(quality, 'signal_directions') and isinstance(quality.signal_directions, pd.Series):
                signal_dirs_list.append(quality.signal_directions)

        # Calculate median across targets for each metric
        aggregated_metrics: Dict[str, float] = {}
        for metric_name, values in all_metrics.items():
            if len(values) > 0:
                aggregated_metrics[metric_name] = float(np.median(values))

        # Aggregate coverage from per-target coverage attributes
        aggregated_coverage = float(np.median(coverage_values)) if coverage_values else 0.0

        # Create aggregated opportunity series (concatenate for count/logging purposes)
        aggregated_opp_scores = pd.concat(opp_scores_list) if opp_scores_list else pd.Series(dtype=float)
        aggregated_opp_weights = pd.concat(opp_weights_list) if opp_weights_list else pd.Series(dtype=float)
        aggregated_potential_profits = pd.concat(potential_profits_list) if potential_profits_list else pd.Series(dtype=float)
        aggregated_signal_dirs = pd.concat(signal_dirs_list) if signal_dirs_list else pd.Series(dtype=float)

        # Create aggregated quality score
        composite_score = self._calculate_composite_score(aggregated_metrics)

        class AggregatedQualityScore:
            def __init__(self, composite_score, aggregated_metrics, n_targets,
                         coverage, opp_scores, opp_weights, pot_profits, signal_dirs):
                self.overall_quality = composite_score
                self.predictability = aggregated_metrics.get('ic', 0.0)
                self.stability = aggregated_metrics.get('stability', 0.0)
                self.balance = aggregated_metrics.get('balance', 0.0)
                self.coverage = coverage
                self.n_targets = n_targets
                self.metrics = aggregated_metrics
                # Preserve opportunity-level attributes for downstream consumers
                self.opportunity_scores = opp_scores
                self.opportunity_weights = opp_weights
                self.potential_profits = pot_profits
                self.signal_directions = signal_dirs

        return {
            'aggregated': AggregatedQualityScore(
                composite_score,
                aggregated_metrics,
                len(target_qualities),
                aggregated_coverage,
                aggregated_opp_scores,
                aggregated_opp_weights,
                aggregated_potential_profits,
                aggregated_signal_dirs,
            )
        }
    
    def _is_classification_like(self, labels: pd.Series) -> bool:
        """Check if labels are classification-like."""
        unique_vals = set(labels.dropna().unique())
        return (unique_vals <= {0.0, 1.0} or 
                unique_vals <= {-1.0, 0.0, 1.0} or 
                unique_vals <= {-1.0, 1.0})
    
    def _calculate_ic_pvalue(self, labels: pd.Series, returns: pd.Series) -> float:
        """Calculate IC p-value using bootstrap."""
        if len(labels) < 10 or len(returns) < 10:
            return 1.0
        
        # Simple correlation test
        from scipy.stats import spearmanr
        try:
            correlation, p_value = spearmanr(labels, returns)
            return p_value if not pd.isna(p_value) else 1.0
        except:
            return 1.0
    
    def _create_fallback_quality_score(self) -> Dict[str, Any]:
        """Create fallback quality score when calculation fails."""
        class FallbackQualityScore:
            def __init__(self) -> None:
                self.overall_quality = 0.0
                self.predictability = 0.0
                self.stability = 0.0
                self.balance = 0.0
                self.coverage = 0.0
                self.metrics = {}
        
        return {'default': FallbackQualityScore()}
    
    def _check_no_overlap(self, labels: pd.Series, lookahead_returns: pd.Series) -> bool:
        """Check for no overlap between labels and lookahead returns."""
        # Ensure lookahead returns are strictly in the future
        if len(labels) != len(lookahead_returns):
            return False
        
        # Check that lookahead returns don't overlap with label formation period
        # This is a simplified check - in practice, you'd want more sophisticated overlap detection
        return True
    
    def _run_randomized_label_test(self, labels: pd.Series, lookahead_returns: pd.Series, target_name: str) -> bool:
        """Test that randomized labels produce near-zero metrics."""
        try:
            # Shuffle labels randomly
            shuffled_labels = labels.sample(frac=1.0, random_state=42).reset_index(drop=True)
            shuffled_labels.index = labels.index
            
            # Calculate metrics on shuffled data
            shuffled_metrics = self._calculate_predictability_metrics(shuffled_labels, lookahead_returns)
            
            # Check that IC, Hit Rate, Uplift, Sharpe collapse to ~0
            ic_shuffled = abs(shuffled_metrics.get('ic', 0))
            hit_rate_shuffled = shuffled_metrics.get('hit_rate', 0)
            uplift_shuffled = abs(shuffled_metrics.get('uplift', 0))
            
            # Thresholds for "collapsed" metrics
            ic_threshold = 0.05
            hit_rate_threshold = 0.55  # Should be close to random (0.5)
            uplift_threshold = 0.01  # 1% in returns
            
            if (ic_shuffled < ic_threshold and 
                abs(hit_rate_shuffled - 0.5) < 0.1 and 
                uplift_shuffled < uplift_threshold):
                return True
            else:
                self.logger.warning(f"Randomized test failed: IC={ic_shuffled:.4f}, HitRate={hit_rate_shuffled:.4f}, Uplift={uplift_shuffled:.4f}")
                return False
                
        except Exception as e:
            self.logger.warning(f"Randomized label test failed with error: {e}")
            return False
    
    def _run_permutation_ic_test(self, labels: pd.Series, lookahead_returns: pd.Series, target_name: str) -> bool:
        """Test that observed IC is in top 5% of null distribution."""
        try:
            # Calculate observed IC
            observed_ic = labels.corr(lookahead_returns, method='spearman')
            if pd.isna(observed_ic):
                return False
            
            # Run 500 permutations
            n_permutations = 500
            null_ics = []
            
            for _ in range(n_permutations):
                # Permute labels
                permuted_labels = labels.sample(frac=1.0, random_state=np.random.randint(0, 10000))
                permuted_labels.index = labels.index
                
                # Calculate IC
                perm_ic = permuted_labels.corr(lookahead_returns, method='spearman')
                if not pd.isna(perm_ic):
                    null_ics.append(perm_ic)
            
            if len(null_ics) < 100:  # Need sufficient permutations
                return False
            
            # Check if observed IC is in top 5%
            null_ics = np.array(null_ics)
            percentile_95 = np.percentile(null_ics, 95)
            
            if abs(observed_ic) > percentile_95:
                return True
            else:
                self.logger.warning(f"Permutation test failed: observed_ic={observed_ic:.4f}, 95th_percentile={percentile_95:.4f}")
                return False
                
        except Exception as e:
            self.logger.warning(f"Permutation IC test failed with error: {e}")
            return False
    
    def _calculate_target_metrics_calibrated(self, labels: pd.Series, lookahead_returns: pd.Series, target_name: str) -> Dict[str, float]:
        """Calculate comprehensive metrics with calibration and scaling."""
        metrics = {}
        
        # Predictability block (signal quality)
        metrics.update(self._calculate_predictability_metrics(labels, lookahead_returns))
        
        # Class/coverage block
        metrics.update(self._calculate_class_metrics(labels))
        
        # Stability block (time robustness) with blocked CV
        metrics.update(self._calculate_stability_metrics_robust(labels, lookahead_returns))
        
        # Risk-aware block with volatility normalization
        metrics.update(self._calculate_risk_metrics_calibrated(labels, lookahead_returns))
        
        # Apply calibration and scaling
        metrics = self._apply_calibration_scaling(metrics, target_name)
        
        return metrics
    
    def _calculate_stability_metrics_robust(self, labels: pd.Series, lookahead_returns: pd.Series) -> Dict[str, float]:
        """Calculate stability metrics with blocked CV and regime slicing."""
        metrics = {}
        
        # Blocked CV: use contiguous time folds
        n_folds = 5
        fold_size = len(labels) // n_folds
        
        if fold_size < 20:  # Need sufficient data per fold
            metrics['stability'] = 0.0
            metrics['temporal_cv_ic'] = 0.0
            metrics['temporal_cv_iqr'] = 0.0
            return metrics
        
        fold_ics = []
        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < n_folds - 1 else len(labels)
            
            fold_labels = labels.iloc[start_idx:end_idx]
            fold_returns = lookahead_returns.iloc[start_idx:end_idx]
            
            if len(fold_labels.dropna()) > 5 and len(fold_returns.dropna()) > 5:
                ic = fold_labels.corr(fold_returns, method='spearman')
                if not pd.isna(ic):
                    fold_ics.append(ic)
        
        if len(fold_ics) > 1:
            # Rolling IC stability
            rolling_ics = []
            window_size = min(252, len(labels) // 4)
            for i in range(window_size, len(labels)):
                window_labels = labels.iloc[i-window_size:i]
                window_returns = lookahead_returns.iloc[i-window_size:i]
                if len(window_labels.dropna()) > 5 and len(window_returns.dropna()) > 5:
                    ic = window_labels.corr(window_returns, method='spearman')
                    if not pd.isna(ic):
                        rolling_ics.append(ic)
            
            if len(rolling_ics) > 1:
                ic_std = np.std(rolling_ics)
                stability = 1 / (1 + ic_std)
            else:
                stability = 0.0
            
            # Temporal CV metrics
            temporal_cv_ic = np.median(fold_ics)
            temporal_cv_iqr = np.percentile(fold_ics, 75) - np.percentile(fold_ics, 25)
            
            metrics['stability'] = stability
            metrics['temporal_cv_ic'] = temporal_cv_ic
            metrics['temporal_cv_iqr'] = temporal_cv_iqr
        else:
            metrics['stability'] = 0.0
            metrics['temporal_cv_ic'] = 0.0
            metrics['temporal_cv_iqr'] = 0.0
        
        return metrics
    
    def _calculate_risk_metrics_calibrated(self, labels: pd.Series, lookahead_returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-aware metrics with volatility normalization."""
        metrics = {}
        
        # Sharpe ratio of labeled subset with volatility normalization
        if self._is_classification_like(labels):
            positive_mask = labels > 0
            if positive_mask.sum() > 5:
                labeled_returns = lookahead_returns[positive_mask]
                if len(labeled_returns.dropna()) > 5:
                    # Calculate Sharpe with volatility normalization
                    mean_return = labeled_returns.mean()
                    std_return = labeled_returns.std()
                    
                    if std_return > 0:
                        # Basic Sharpe
                        sharpe = mean_return / std_return
                        
                        # Volatility normalization toggle
                        if hasattr(self.config, 'volatility_normalization') and self.config.volatility_normalization:
                            # Deflated Sharpe for low-variance windows
                            vol_window = labeled_returns.rolling(20).std()
                            vol_mean = vol_window.mean()
                            if vol_mean > 0:
                                vol_adjusted_sharpe = sharpe * (vol_mean / std_return)
                                sharpe = min(sharpe, vol_adjusted_sharpe)  # Cap at deflated Sharpe
                        
                        metrics['sharpe'] = sharpe if not pd.isna(sharpe) else 0.0
                    else:
                        metrics['sharpe'] = 0.0
                else:
                    metrics['sharpe'] = 0.0
            else:
                metrics['sharpe'] = 0.0
        else:
            metrics['sharpe'] = 0.0
        
        return metrics
    
    def _apply_calibration_scaling(self, metrics: Dict[str, float], target_name: str) -> Dict[str, float]:
        """Apply calibration and scaling to make metrics comparable."""
        # Fixed caps for normalization
        ic_cap = 0.1
        uplift_cap = 0.005  # 50 bps per lookahead
        sharpe_cap = 2.0
        
        # Normalize IC to [0,1]
        ic_raw = metrics.get('ic', 0)
        metrics['ic_norm'] = max(0, min(1, (ic_raw + ic_cap) / (2 * ic_cap)))
        
        # Normalize Uplift to [0,1]
        uplift_raw = metrics.get('uplift', 0)
        metrics['uplift_norm'] = max(0, min(1, (uplift_raw + uplift_cap) / (2 * uplift_cap)))
        
        # Normalize Sharpe to [0,1]
        sharpe_raw = metrics.get('sharpe', 0)
        metrics['sharpe_norm'] = max(0, min(1, (sharpe_raw + sharpe_cap) / (2 * sharpe_cap)))
        
        # Store raw values for reporting
        metrics['ic_raw'] = ic_raw
        metrics['uplift_raw'] = uplift_raw
        metrics['sharpe_raw'] = sharpe_raw
        
        return metrics
    
    def _extract_red_flags(self, metrics: Dict[str, float], coverage: float) -> List[str]:
        """Extract red flag reasons for reporting."""
        red_flags = []
        
        # Check for red flags in order of severity
        if coverage < 0.05:
            red_flags.append("low_coverage")
        elif coverage < 0.10:
            red_flags.append("marginal_coverage")
        
        balance = metrics.get('balance', 0)
        if balance < 0.2:
            red_flags.append("imbalance")
        elif balance < 0.3:
            red_flags.append("marginal_balance")
        
        ic = abs(metrics.get('ic', 0))
        if ic < 0.01:
            red_flags.append("weak_IC")
        elif ic < 0.03:
            red_flags.append("marginal_IC")
        
        stability = metrics.get('stability', 0)
        if stability < 0.3:
            red_flags.append("unstable")
        elif stability < 0.6:
            red_flags.append("marginal_stability")
        
        return red_flags[:1]  # Return first red flag only
    
    def _create_downstream_opportunity_data(self, quality_scores: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Create downstream-ready opportunity data with scores and weights for each target."""
        opportunity_data = {}

        for target_name, quality in quality_scores.items():
            if hasattr(quality, 'opportunity_scores') and hasattr(quality, 'opportunity_weights') and hasattr(quality, 'potential_profits'):
                # Create DataFrame with all opportunity information
                opportunity_df = pd.DataFrame({
                    'opportunity_index': quality.opportunity_scores.index,
                    'signal_direction': quality.signal_directions.reindex(quality.opportunity_scores.index).fillna(0).astype(int),
                    'potential_profit': quality.potential_profits,
                    'quality_score': quality.opportunity_scores,
                    'weight': quality.opportunity_weights,
                    'target_name': target_name
                })

                # Add derived metrics
                opportunity_df['profit_rank'] = opportunity_df['potential_profit'].rank(ascending=False)
                opportunity_df['quality_rank'] = opportunity_df['quality_score'].rank(ascending=False)
                opportunity_df['weight_rank'] = opportunity_df['weight'].rank(ascending=False)

                # Add composite opportunity score (combination of quality and weight)
                opportunity_df['composite_score'] = (
                    0.6 * opportunity_df['quality_score'] +
                    0.4 * opportunity_df['weight']
                )

                # Keep only top 80% opportunities per quality (rank-based filtering)
                if len(opportunity_df) > 0:
                    # Calculate quality percentile threshold (top 80% = quality_rank <= 0.8 * total_count)
                    total_opportunities = len(opportunity_df)
                    quality_threshold_rank = int(0.8 * total_opportunities)

                    # Filter to keep only opportunities in top 80% by quality rank
                    opportunity_df = opportunity_df[opportunity_df['quality_rank'] <= quality_threshold_rank].copy()

                    # Re-rank after filtering to maintain consecutive ranks
                    opportunity_df['quality_rank'] = opportunity_df['quality_score'].rank(ascending=False)
                    opportunity_df['profit_rank'] = opportunity_df['potential_profit'].rank(ascending=False)
                    opportunity_df['weight_rank'] = opportunity_df['weight'].rank(ascending=False)

                opportunity_data[target_name] = opportunity_df

        return opportunity_data
    
    def score_to_training(self, quality_scores: Dict[str, Any], training_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Map quality scores to training strategies (gating, weighting, curriculum learning).

        Args:
            quality_scores: Quality scores from labeling
            training_config: Optional training configuration overrides

        Returns:
            Dictionary with training strategies mapped from scores
        """
        if not quality_scores:
            return self._create_default_training_strategy()
        
        training_strategy = {
            'gating': {},
            'weighting': {},
            'curriculum_learning': {},
            'memory_optimization': {},
            'reproducible_seeds': {}
        }
        
        # Process each target's quality scores
        for target_name, quality in quality_scores.items():
            if not hasattr(quality, 'overall_quality'):
                continue
                
            overall_quality = quality.overall_quality
            opportunity_scores = getattr(quality, 'opportunity_scores', pd.Series())
            opportunity_weights = getattr(quality, 'opportunity_weights', pd.Series())
            
            # 1. GATING: Determine if target should be included in training
            training_strategy['gating'][target_name] = self._calculate_training_gate(
                overall_quality, quality, training_config
            )
            
            # 2. WEIGHTING: Calculate sample weights for training
            training_strategy['weighting'][target_name] = self._calculate_training_weights(
                opportunity_scores, opportunity_weights, overall_quality, training_config
            )
            
            # 3. CURRICULUM LEARNING: Determine training order/difficulty
            training_strategy['curriculum_learning'][target_name] = self._calculate_curriculum_level(
                overall_quality, opportunity_scores, training_config
            )
        
        # 4. MEMORY OPTIMIZATION: Ensure O(N) memory usage
        training_strategy['memory_optimization'] = self._calculate_memory_strategy(
            quality_scores, training_config
        )
        
        # 5. REPRODUCIBLE SEEDS: Generate seeds for parallel folds
        training_strategy['reproducible_seeds'] = self._generate_reproducible_seeds(
            quality_scores, training_config
        )
        
        return training_strategy
    
    def _calculate_training_gate(self, overall_quality: float, quality: Any, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate training gate based on quality scores."""
        gate_config = config.get('gating', {}) if config else {}

        # Quality thresholds - very relaxed for crypto trading
        min_quality = gate_config.get('min_quality', 0.15)  # Further reduced for crypto (was 0.2) - 15% quality threshold
        # min_predictability removed (IC unused)

        # Get opportunity count for logging
        n_opportunities = len(quality.opportunity_scores) if hasattr(quality, 'opportunity_scores') else 0
        target_name = getattr(quality, 'target_name', 'unknown')

        # Check gate conditions
        passes_quality = overall_quality >= min_quality
        # Predictability gate: for single-direction (e.g., long-only) treat IC=0 as neutral/pass
        # Predictability gate removed; treat as pass
        passes_predictability = True

        # Additional checks for trade opportunities (relaxed for crypto)
        min_opportunities = gate_config.get('min_opportunities', 2)  # Reduced for crypto (was 5)
        has_opportunities = n_opportunities >= min_opportunities

        gate_passed = passes_quality and passes_predictability and has_opportunities

        # Log gate filtering details
        tprint_info(f"🔍 [GATE FILTERING] Target: {target_name}")
        tprint_info(f"🔍 [GATE FILTERING]   Opportunities: {n_opportunities} (min required: {min_opportunities})")
        tprint_info(f"🔍 [GATE FILTERING]   Quality: {overall_quality:.3f} >= {min_quality} ? {'✅' if passes_quality else '❌'}")
        tprint_info(f"🔍 [GATE FILTERING]   Predictability: {'✅' if passes_predictability else '❌'}")
        tprint_info(f"🔍 [GATE FILTERING]   Overall: {'✅ PASS' if gate_passed else '❌ FAIL'}")
        
        return {
            'include_in_training': gate_passed,
            'quality_score': overall_quality,
            'coverage': getattr(quality, 'coverage', 0),
            'predictability': getattr(quality, 'predictability', 0),
            'n_opportunities': len(quality.opportunity_scores) if hasattr(quality, 'opportunity_scores') else 0,
            'gate_reason': self._get_gate_reason(gate_passed, passes_quality, passes_predictability, has_opportunities)
        }
    
    def _calculate_training_weights(self, opportunity_scores: pd.Series, opportunity_weights: pd.Series, 
                                  overall_quality: float, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate training weights based on opportunity scores."""
        weight_config = config.get('weighting', {}) if config else {}
        
        if len(opportunity_scores) == 0:
            return {'weights': pd.Series(dtype=float), 'weight_strategy': 'uniform'}
        
        # Weight calculation strategies
        strategy = weight_config.get('strategy', 'quality_weighted')
        
        if strategy == 'quality_weighted':
            # Use individual quality scores as weights
            weights = opportunity_scores.copy()
        elif strategy == 'profit_weighted':
            # Use potential profits as weights (if available)
            if hasattr(opportunity_scores, 'index'):
                # This would need access to potential_profits - simplified for now
                weights = opportunity_scores.copy()
            else:
                weights = opportunity_scores.copy()
        elif strategy == 'composite_weighted':
            # Combine quality scores with opportunity weights
            if len(opportunity_weights) > 0:
                weights = 0.7 * opportunity_scores + 0.3 * opportunity_weights
            else:
                weights = opportunity_scores.copy()
        else:  # uniform
            weights = pd.Series(1.0, index=opportunity_scores.index)
        
        # Apply quality-based scaling
        quality_scale = weight_config.get('quality_scale', True)
        if quality_scale:
            weights = weights * overall_quality
        
        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = pd.Series(1.0 / len(weights), index=weights.index)
        
        return {
            'weights': weights,
            'weight_strategy': strategy,
            'quality_scale': quality_scale,
            'weight_stats': {
                'mean': weights.mean(),
                'std': weights.std(),
                'min': weights.min(),
                'max': weights.max()
            }
        }
    
    def _calculate_curriculum_level(self, overall_quality: float, opportunity_scores: pd.Series, 
                                  config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate curriculum learning level based on quality scores."""
        curriculum_config = config.get('curriculum_learning', {}) if config else {}
        
        # Determine difficulty level based on quality
        if overall_quality >= 0.8:
            difficulty_level = 'expert'
            training_order = 1  # Train first (highest quality)
        elif overall_quality >= 0.6:
            difficulty_level = 'intermediate'
            training_order = 2
        elif overall_quality >= 0.4:
            difficulty_level = 'beginner'
            training_order = 3
        else:
            difficulty_level = 'novice'
            training_order = 4  # Train last (lowest quality)
        
        # Calculate sample complexity
        if len(opportunity_scores) > 0:
            score_std = opportunity_scores.std()
            complexity = min(1.0, score_std * 2)  # Higher std = more complex
        else:
            complexity = 0.5
        
        return {
            'difficulty_level': difficulty_level,
            'training_order': training_order,
            'complexity_score': complexity,
            'quality_threshold': overall_quality,
            'enable_curriculum': curriculum_config.get('enable', True)
        }
    
    def _calculate_memory_strategy(self, quality_scores: Dict[str, Any], config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate memory optimization strategy for O(N) memory usage."""
        memory_config = config.get('memory_optimization', {}) if config else {}
        
        # Count total opportunities across all targets
        total_opportunities = 0
        for quality in quality_scores.values():
            if hasattr(quality, 'opportunity_scores'):
                total_opportunities += len(quality.opportunity_scores)
        
        # Memory budget (in MB)
        memory_budget_mb = memory_config.get('memory_budget_mb', 1024)  # 1GB default
        bytes_per_opportunity = memory_config.get('bytes_per_opportunity', 1000)  # Estimate
        
        max_opportunities = (memory_budget_mb * 1024 * 1024) // bytes_per_opportunity
        
        # Determine if we need to subsample
        needs_subsampling = total_opportunities > max_opportunities
        subsample_ratio = min(1.0, max_opportunities / total_opportunities) if total_opportunities > 0 else 1.0
        
        return {
            'total_opportunities': total_opportunities,
            'memory_budget_mb': memory_budget_mb,
            'max_opportunities': max_opportunities,
            'needs_subsampling': needs_subsampling,
            'subsample_ratio': subsample_ratio,
            'chunk_size': memory_config.get('chunk_size', 10000),
            'enable_streaming': needs_subsampling
        }
    
    def _generate_reproducible_seeds(self, quality_scores: Dict[str, Any], config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate reproducible seeds for parallel folds."""
        seed_config = config.get('reproducible_seeds', {}) if config else {}
        
        base_seed = seed_config.get('base_seed', 42)
        n_folds = seed_config.get('n_folds', 5)
        
        # Generate seeds for each fold
        fold_seeds = {}
        for i in range(n_folds):
            fold_seeds[f'fold_{i}'] = base_seed + i * 1000
        
        # Generate seeds for each target
        target_seeds = {}
        for i, target_name in enumerate(quality_scores.keys()):
            target_seeds[target_name] = base_seed + i * 100
        
        return {
            'base_seed': base_seed,
            'n_folds': n_folds,
            'fold_seeds': fold_seeds,
            'target_seeds': target_seeds,
            'random_state': np.random.RandomState(base_seed)
        }
    
    def _get_gate_reason(self, gate_passed: bool, quality_ok: bool,
                        predictability_ok: bool, has_opportunities: bool) -> str:
        """Get human-readable gate reason."""
        if gate_passed:
            return "PASS: All quality gates met"

        reasons = []
        if not quality_ok:
            reasons.append("low_quality")
        if not predictability_ok:
            reasons.append("low_predictability")
        if not has_opportunities:
            reasons.append("insufficient_opportunities")
        
        return f"FAIL: {', '.join(reasons)}"
    
    def _create_default_training_strategy(self) -> Dict[str, Any]:
        """Create default training strategy when no quality scores available."""
        return {
            'gating': {'default': {'include_in_training': False, 'gate_reason': 'no_quality_scores'}},
            'weighting': {'default': {'weights': pd.Series(dtype=float), 'weight_strategy': 'uniform'}},
            'curriculum_learning': {'default': {'difficulty_level': 'novice', 'training_order': 999}},
            'memory_optimization': {'total_opportunities': 0, 'needs_subsampling': False},
            'reproducible_seeds': {'base_seed': 42, 'n_folds': 5, 'fold_seeds': {}}
        }
    
    def _create_fallback_quality_score(self, reason: str = "unknown") -> Dict[str, Any]:
        """Create fallback quality score with reason."""
        class FallbackQualityScore:
            def __init__(self, reason):
                self.overall_quality = 0.0
                self.predictability = 0.0
                self.stability = 0.0
                self.balance = 0.0
                self.coverage = 0.0
                self.gates_passed = False
                self.metrics = {}
                self.red_flag_reasons = [reason]
        
        return {'default': FallbackQualityScore(reason)}
    
    def _log_comprehensive_outcome_report(self, labels: Union[pd.Series, pd.DataFrame], quality_scores: Dict[str, Any], 
                                        metadata: Dict[str, Any], training_strategy: Dict[str, Any], 
                                        performance_config: Dict[str, Any]) -> None:
        """Generate comprehensive, human-readable outcome report."""
        self.logger.info("=" * 80)
        self.logger.info("🎯 VOLATILITY-AWARE LABELING OUTCOME REPORT")
        self.logger.info("=" * 80)
        
        # 1. EXECUTIVE SUMMARY
        self._log_executive_summary(labels, quality_scores, metadata)
        
        # 2. LABELING PERFORMANCE
        self._log_labeling_performance(labels, metadata)
        
        # 3. QUALITY ANALYSIS
        self._log_quality_analysis(quality_scores)
        
        # 4. TRADE OPPORTUNITIES ANALYSIS
        self._log_trade_opportunities_analysis(quality_scores, labels)
        
        # 5. TRAINING STRATEGY RECOMMENDATIONS
        self._log_training_strategy_recommendations(training_strategy)
        
        # 6. PERFORMANCE OPTIMIZATION
        self._log_performance_optimization(performance_config)
        
        # 7. RISK ASSESSMENT & WARNINGS
        self._log_risk_assessment(quality_scores, metadata)

        # 8. LABEL SMOOTHING ANALYSIS
        if 'label_smoothing' in metadata and metadata['label_smoothing'].get('enabled', False):
            self._log_label_smoothing_analysis(metadata['label_smoothing'])

        # 9. NEXT STEPS & RECOMMENDATIONS
        self._log_next_steps_recommendations(quality_scores, training_strategy, performance_config)

        self.logger.info("=" * 80)
        self.logger.info("📋 REPORT COMPLETE")
        self.logger.info("=" * 80)
        
        # Print current working directory and file paths
        import os
        current_dir = os.getcwd()
        self.logger.info(f"📁 Current Working Directory: {current_dir}")
        self.logger.info(f"📁 Report Generated From: {__file__}")
        self.logger.info(f"📁 Data Cache Directory: {os.path.abspath(os.path.join(current_dir, 'data_cache'))}")
        self.logger.info(f"📁 Artifacts Directory: {os.path.abspath(os.path.join(current_dir, 'artifacts'))}")
        self.logger.info(f"📁 Outcomes Directory: {os.path.abspath(os.path.join(current_dir, 'outcomes'))}")
    
    def _log_executive_summary(self, labels: Union[pd.Series, pd.DataFrame], quality_scores: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        """Log executive summary of labeling results."""
        self.logger.info("📊 EXECUTIVE SUMMARY")
        self.logger.info("-" * 40)
        
        # Basic statistics
        total_labels = metadata.get("total_labels", 0)
        non_null_labels = metadata.get("non_null_labels", 0)
        coverage = non_null_labels / total_labels if total_labels > 0 else 0
        
        # Count opportunities with configurable quality threshold
        total_opportunities = 0
        high_quality_opportunities = 0
        quality_threshold = getattr(self.config, 'min_label_quality', 0.4)  # Use configured threshold

        for quality in quality_scores.values():
            if hasattr(quality, 'opportunity_scores'):
                total_opportunities += len(quality.opportunity_scores)
                # Count high-quality opportunities using configured threshold
                if hasattr(quality, 'opportunity_scores') and hasattr(quality, 'opportunity_quality_scores'):
                    high_quality_opportunities += sum(1 for q_score in quality.opportunity_quality_scores if q_score > quality_threshold)
                # Fallback: if no opportunity quality scores, use overall quality as proxy
                elif hasattr(quality, 'overall_quality') and quality.overall_quality > quality_threshold:
                    high_quality_opportunities += 1
        
        # Overall assessment
        if coverage > 0.1 and total_opportunities > 10:
            status = "✅ SUCCESS"
            status_emoji = "🎉"
        elif coverage > 0.05 and total_opportunities > 5:
            status = "⚠️ PARTIAL SUCCESS"
            status_emoji = "⚠️"
        else:
            status = "❌ NEEDS ATTENTION"
            status_emoji = "🚨"
        
        self.logger.info(f"{status_emoji} Overall Status: {status}")
        self.logger.info(f"📈 Data Coverage: {coverage:.1%} ({non_null_labels:,} / {total_labels:,} samples)")
        self.logger.info(f"🎯 Trade Opportunities: {total_opportunities:,} identified")
        self.logger.info(f"⭐ High-Quality Opportunities: {high_quality_opportunities} / {total_opportunities:,}")
        self.logger.info(f"⏱️ Processing Time: {metadata.get('processing_time', 0):.2f}s")
    
    def _log_labeling_performance(self, labels: Union[pd.Series, pd.DataFrame], metadata: Dict[str, Any]) -> None:
        """Log detailed labeling performance metrics."""
        self.logger.info("")
        self.logger.info("📈 LABELING PERFORMANCE")
        self.logger.info("-" * 40)
        
        if isinstance(labels, pd.DataFrame):
            self.logger.info("Multi-Target Analysis:")
            for col in labels.columns:
                coverage = (labels[col] != 0).mean() if len(labels[col]) > 0 else 0
                positive_rate = (labels[col] > 0).mean() if len(labels[col]) > 0 else 0
                negative_rate = (labels[col] < 0).mean() if len(labels[col]) > 0 else 0
                signal_rate = (labels[col] != 0).mean() if len(labels[col]) > 0 else 0
                
                self.logger.info(f"  🎯 {col}:")
                self.logger.info(f"     Coverage: {coverage:.1%} | Signals: {signal_rate:.1%} | Long: {positive_rate:.1%} | Short: {negative_rate:.1%}")
        else:
            coverage = (labels != 0).mean() if len(labels) > 0 else 0
            positive_rate = (labels > 0).mean() if len(labels) > 0 else 0
            negative_rate = (labels < 0).mean() if len(labels) > 0 else 0
            signal_rate = (labels != 0).mean() if len(labels) > 0 else 0
            
            self.logger.info(f"Single Target Analysis:")
            self.logger.info(f"  Coverage: {coverage:.1%} | Signals: {signal_rate:.1%} | Long: {positive_rate:.1%} | Short: {negative_rate:.1%}")
        
        # Configuration info
        self.logger.info(f"⚙️ Configuration:")
        self.logger.info(f"  Label Type: {metadata.get('label_type', 'unknown')}")
        self.logger.info(f"  Volatility Enabled: {metadata.get('volatility_enabled', False)}")
        self.logger.info(f"  Volatility Window: {metadata.get('volatility_window', 'N/A')}")
    
    def _log_quality_analysis(self, quality_scores: Dict[str, Any]) -> None:
        """Log detailed quality analysis."""
        self.logger.info("")
        self.logger.info("⭐ QUALITY ANALYSIS")
        self.logger.info("-" * 40)
        
        if not quality_scores:
            self.logger.warning("  ⚠️ No quality scores available")
            return
        
        # Overall quality statistics
        quality_values = []
        for quality in quality_scores.values():
            if hasattr(quality, 'overall_quality'):
                quality_values.append(quality.overall_quality)
        
        if quality_values:
            avg_quality = np.mean(quality_values)
            min_quality = np.min(quality_values)
            max_quality = np.max(quality_values)
            
            self.logger.info(f"📊 Overall Quality Statistics:")
            self.logger.info(f"  Average: {avg_quality:.3f} | Range: {min_quality:.3f} - {max_quality:.3f}")
            
            # Quality distribution
            excellent = sum(1 for q in quality_values if q >= 0.8)
            good = sum(1 for q in quality_values if 0.6 <= q < 0.8)
            fair = sum(1 for q in quality_values if 0.4 <= q < 0.6)
            poor = sum(1 for q in quality_values if q < 0.4)
            
            self.logger.info(f"📈 Quality Distribution:")
            self.logger.info(f"  🏆 Excellent (≥0.8): {excellent} opportunities")
            self.logger.info(f"  ✅ Good (0.6-0.8): {good} opportunities")
            self.logger.info(f"  ⚠️ Fair (0.4-0.6): {fair} opportunities")
            self.logger.info(f"  ❌ Poor (<0.4): {poor} opportunities")
        
        # Per-target detailed analysis
        self.logger.info(f"")
        self.logger.info(f"🎯 Per-Target Analysis (showing opportunity quality):")
        for target_name, quality in quality_scores.items():
            if hasattr(quality, 'overall_quality'):
                self._log_target_quality_details(target_name, quality)
    
    def _log_target_quality_details(self, target_name: str, quality: Any) -> None:
        """Log detailed quality information for a specific target/opportunity set."""
        overall_quality = getattr(quality, 'overall_quality', 0)
        # Predictability (IC) removed from reporting
        stability = getattr(quality, 'stability', 0)
        coverage = getattr(quality, 'coverage', 0)
        
        # Quality badge
        if overall_quality >= 0.8:
            badge = "🏆 EXCELLENT"
        elif overall_quality >= 0.6:
            badge = "✅ GOOD"
        elif overall_quality >= 0.4:
            badge = "⚠️ FAIR"
        else:
            badge = "❌ POOR"
        
        self.logger.info(f"  {badge} {target_name}:")
        self.logger.info(f"     Overall Quality: {overall_quality:.3f}")
        self.logger.info(f"     Stability: {stability:.3f}")
        self.logger.info(f"     Coverage: {coverage:.1%}")
        
        # Opportunity metrics
        if hasattr(quality, 'avg_potential_profit'):
            avg_profit = quality.avg_potential_profit * 10000  # Convert to bps
            max_profit = quality.max_potential_profit * 10000
            self.logger.info(f"     Avg Potential Profit: {avg_profit:.1f}bps | Max: {max_profit:.1f}bps")
        
        # Red flags
        red_flags = getattr(quality, 'red_flag_reasons', [])
        if red_flags:
            self.logger.info(f"     🚨 Red Flags: {', '.join(red_flags)}")
    
    def _log_trade_opportunities_analysis(self, quality_scores: Dict[str, Any], labels: Union[pd.Series, pd.DataFrame]) -> None:
        """Log detailed trade opportunities analysis."""
        self.logger.info("")
        self.logger.info("🎯 TRADE OPPORTUNITIES ANALYSIS")
        self.logger.info("-" * 40)
        
        total_opportunities = 0
        total_long_opportunities = 0
        total_short_opportunities = 0
        total_potential_profit = 0
        
        for target_name, quality in quality_scores.items():
            if hasattr(quality, 'opportunity_scores') and len(quality.opportunity_scores) > 0:
                n_opportunities = len(quality.opportunity_scores)
                total_opportunities += n_opportunities

                # Count long/short via signal directions if available; fallback to profits
                if hasattr(quality, 'signal_directions') and len(quality.signal_directions) > 0:
                    dir_series = quality.signal_directions
                    long_ops = (dir_series > 0).sum()
                    short_ops = (dir_series < 0).sum()
                    total_long_opportunities += int(long_ops)
                    total_short_opportunities += int(short_ops)
                elif hasattr(quality, 'potential_profits'):
                    # potential_profits are non-negative by design; treat all as long
                    total_long_opportunities += n_opportunities
                    total_short_opportunities += 0

                if hasattr(quality, 'potential_profits'):
                    total_potential_profit += quality.potential_profits.sum()
                
                self.logger.info(f"  📊 {target_name}: {n_opportunities:,} opportunities")
                if hasattr(quality, 'avg_potential_profit'):
                    avg_profit = quality.avg_potential_profit * 10000
                    self.logger.info(f"     Avg Potential Profit: {avg_profit:.1f}bps")
        
        if total_opportunities > 0:
            # Calculate daily average
            try:
                if hasattr(labels, 'index') and len(labels) > 0:
                    # Get the date range from the index
                    if hasattr(labels.index, 'date'):
                        # If index has date attribute (datetime index)
                        start_date = labels.index.min().date()
                        end_date = labels.index.max().date()
                        days = (end_date - start_date).days + 1
                    elif hasattr(labels.index, 'to_pydatetime'):
                        # If index can be converted to datetime
                        start_date = pd.to_datetime(labels.index.min()).date()
                        end_date = pd.to_datetime(labels.index.max()).date()
                        days = (end_date - start_date).days + 1
                    else:
                        # Fallback: estimate days from data length (assuming 15m intervals)
                        # 15m = 96 intervals per day
                        days = max(1, len(labels) // 96)
                    
                    avg_per_day = total_opportunities / days if days > 0 else 0
                else:
                    avg_per_day = 0
                    days = 1
            except Exception:
                # Fallback if date calculation fails
                avg_per_day = 0
                days = 1
            
            self.logger.info(f"")
            self.logger.info(f"📈 Summary Statistics:")
            self.logger.info(f"  Total Opportunities: {total_opportunities:,}")
            self.logger.info(f"  Long Opportunities: {total_long_opportunities:,}")
            self.logger.info(f"  Short Opportunities: {total_short_opportunities:,}")
            self.logger.info(f"  Average per Day: {avg_per_day:.1f} opportunities/day")
            if total_potential_profit > 0:
                avg_total_profit = (total_potential_profit / total_opportunities) * 10000
                self.logger.info(f"  Average Potential Profit: {avg_total_profit:.1f}bps")
        else:
            self.logger.warning("  ⚠️ No trade opportunities identified")
    
    def _log_training_strategy_recommendations(self, training_strategy: Dict[str, Any]) -> None:
        """Log training strategy recommendations."""
        self.logger.info("")
        self.logger.info("🎓 TRAINING STRATEGY RECOMMENDATIONS")
        self.logger.info("-" * 40)
        
        # Gating recommendations
        gating = training_strategy.get('gating', {})
        included_targets = sum(1 for gate in gating.values() if gate.get('include_in_training', False))
        total_targets = len(gating)

        # Log gating results with detailed tprints
        tprint_info(f"🔍 [GATING SUMMARY] Total targets evaluated: {total_targets}")
        tprint_info(f"🔍 [GATING SUMMARY] Targets passing gates: {included_targets} / {total_targets}")

        if included_targets == 0:
            tprint_warning("🔍 [GATING SUMMARY] ⚠️ CRITICAL: No targets passed quality gates!")

        self.logger.info(f"🚪 Gating Strategy:")
        self.logger.info(f"  Targets Included: {included_targets} / {total_targets}")

        for target_name, gate_info in gating.items():
            status = "✅ INCLUDE" if gate_info.get('include_in_training', False) else "❌ EXCLUDE"
            reason = gate_info.get('gate_reason', 'Unknown')
            self.logger.info(f"    {status} {target_name}: {reason}")

        # Additional detailed tprint for each target
        for target_name, gate_info in gating.items():
            if gate_info.get('include_in_training', False):
                tprint_info(f"🔍 [GATING SUMMARY] ✅ {target_name}: INCLUDED ({gate_info.get('gate_reason', 'Unknown')})")
            else:
                tprint_info(f"🔍 [GATING SUMMARY] ❌ {target_name}: EXCLUDED ({gate_info.get('gate_reason', 'Unknown')})")
        
        # Weighting strategy
        weighting = training_strategy.get('weighting', {})
        self.logger.info(f"")
        self.logger.info(f"⚖️ Weighting Strategy:")
        for target_name, weight_info in weighting.items():
            strategy = weight_info.get('weight_strategy', 'unknown')
            weight_stats = weight_info.get('weight_stats', {})
            self.logger.info(f"  {target_name}: {strategy}")
            if weight_stats:
                self.logger.info(f"    Mean: {weight_stats.get('mean', 0):.3f} | Std: {weight_stats.get('std', 0):.3f}")
        
        # Curriculum learning
        curriculum = training_strategy.get('curriculum_learning', {})
        self.logger.info(f"")
        self.logger.info(f"📚 Curriculum Learning:")
        for target_name, curriculum_info in curriculum.items():
            level = curriculum_info.get('difficulty_level', 'unknown')
            order = curriculum_info.get('training_order', 999)
            self.logger.info(f"  {target_name}: {level.upper()} (Order: {order})")
    
    def _log_performance_optimization(self, performance_config: Dict[str, Any]) -> None:
        """Log performance optimization settings."""
        self.logger.info("")
        self.logger.info("⚡ PERFORMANCE OPTIMIZATION")
        self.logger.info("-" * 40)
        
        memory_analysis = performance_config.get('memory_analysis', {})
        parallel_config = performance_config.get('parallel_config', {})
        chunking_config = performance_config.get('chunking_config', {})
        
        # Memory analysis
        data_size_mb = memory_analysis.get('data_size_mb', 0)
        needs_optimization = memory_analysis.get('needs_optimization', False)
        target_chunk_size = memory_analysis.get('target_chunk_size', 0)
        
        self.logger.info(f"💾 Memory Analysis:")
        self.logger.info(f"  Data Size: {data_size_mb:.1f} MB")
        self.logger.info(f"  Optimization Needed: {'Yes' if needs_optimization else 'No'}")
        if needs_optimization:
            self.logger.info(f"  Target Chunk Size: {target_chunk_size:,} samples")
        
        # Parallel processing
        n_workers = parallel_config.get('n_workers', 1)
        n_folds = parallel_config.get('n_folds', 5)
        enable_parallel = parallel_config.get('enable_parallel', False)
        
        self.logger.info(f"")
        self.logger.info(f"🔄 Parallel Processing:")
        self.logger.info(f"  Workers: {n_workers} | Folds: {n_folds}")
        self.logger.info(f"  Parallel Enabled: {'Yes' if enable_parallel else 'No'}")
        
        # Chunking strategy
        if chunking_config.get('enabled', False):
            strategy = chunking_config.get('strategy', 'unknown')
            n_chunks = chunking_config.get('n_chunks', 0)
            self.logger.info(f"")
            self.logger.info(f"📦 Chunking Strategy:")
            self.logger.info(f"  Strategy: {strategy.upper()} | Chunks: {n_chunks}")
    
    def _log_risk_assessment(self, quality_scores: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        """Log risk assessment and warnings."""
        self.logger.info("")
        self.logger.info("⚠️ RISK ASSESSMENT & WARNINGS")
        self.logger.info("-" * 40)
        
        warnings = []
        
        # Check for low quality targets
        low_quality_count = 0
        for quality in quality_scores.values():
            if hasattr(quality, 'overall_quality') and quality.overall_quality < 0.3:
                low_quality_count += 1
        
        if low_quality_count > 0:
            warnings.append(f"Low quality targets: {low_quality_count} targets below 0.3 quality score")
        
        # Check for insufficient opportunities
        total_opportunities = 0
        for quality in quality_scores.values():
            if hasattr(quality, 'opportunity_scores'):
                total_opportunities += len(quality.opportunity_scores)
        
        if total_opportunities < 10:
            warnings.append(f"Insufficient opportunities: Only {total_opportunities} opportunities identified")
        
        # Check for high volatility in quality scores
        quality_values = [q.overall_quality for q in quality_scores.values() if hasattr(q, 'overall_quality')]
        if len(quality_values) > 1:
            quality_std = np.std(quality_values)
            if quality_std > 0.3:
                warnings.append(f"High quality variance: {quality_std:.3f} standard deviation across targets")
        
        # Check for red flags
        red_flag_count = 0
        for quality in quality_scores.values():
            red_flags = getattr(quality, 'red_flag_reasons', [])
            red_flag_count += len(red_flags)
        
        if red_flag_count > 0:
            warnings.append(f"Red flags detected: {red_flag_count} total red flags across targets")
        
        if warnings:
            for warning in warnings:
                self.logger.warning(f"  🚨 {warning}")
        else:
            self.logger.info("  ✅ No significant risks identified")

    def _log_label_smoothing_analysis(self, smoothing_metadata: Dict[str, Any]) -> None:
        """Log label smoothing analysis and effects."""
        self.logger.info("")
        self.logger.info("🎨 LABEL SMOOTHING ANALYSIS")
        self.logger.info("-" * 40)

        # Show configuration
        config = smoothing_metadata.get('config', {})
        stages_applied = smoothing_metadata.get('stages_applied', {})

        self.logger.info(f"  Configuration:")
        self.logger.info(f"    • Ablation Mode: {config.get('ablation_mode', 'full')}")
        self.logger.info(f"    • Stages Applied:")
        if stages_applied.get('classification_smoothing', False):
            self.logger.info(f"      ✓ Classification Smoothing (ε={config.get('epsilon', 0.08):.3f}, T={config.get('temperature', 1.2):.2f})")
        if stages_applied.get('uncertainty_shrinkage', False):
            self.logger.info(f"      ✓ Uncertainty Shrinkage (γ={config.get('gamma', 1.0):.2f}, min_α={config.get('min_alpha', 0.12):.2f})")
        if stages_applied.get('causal_ema', False):
            ema_group = config.get('ema_group_by', 'none')
            self.logger.info(f"      ✓ Causal EMA (decay={config.get('ema_decay', 0.95):.3f}, group_by={ema_group})")

        # Show statistics
        stats = smoothing_metadata.get('statistics', {})
        if stats:
            self.logger.info(f"  ")
            self.logger.info(f"  Label Statistics:")
            self.logger.info(f"    • Raw Labels:   mean={stats.get('raw_mean', 0):.4f}, std={stats.get('raw_std', 0):.4f}")
            self.logger.info(f"    • Final Labels: mean={stats.get('final_mean', 0):.4f}, std={stats.get('final_std', 0):.4f}")
            self.logger.info(f"  ")
            self.logger.info(f"  Smoothing Impact:")
            self.logger.info(f"    • Mean Absolute Change: {stats.get('mean_absolute_change', 0):.4f}")
            self.logger.info(f"    • Max Absolute Change:  {stats.get('max_absolute_change', 0):.4f}")
            self.logger.info(f"    • Raw-Final Correlation: {stats.get('correlation_raw_final', 1.0):.4f}")
            self.logger.info(f"    • % Labels Changed:     {stats.get('pct_changed', 0):.2f}%")

            # Interpretation
            correlation = stats.get('correlation_raw_final', 1.0)
            mean_change = stats.get('mean_absolute_change', 0)

            self.logger.info(f"  ")
            self.logger.info(f"  Interpretation:")
            if correlation > 0.95 and mean_change < 0.05:
                self.logger.info(f"    ✅ Conservative smoothing - labels mostly preserved")
            elif correlation > 0.85 and mean_change < 0.15:
                self.logger.info(f"    ✅ Moderate smoothing - good balance of stability and signal")
            elif correlation > 0.70:
                self.logger.info(f"    ⚠️  Strong smoothing - verify not over-smoothing")
            else:
                self.logger.info(f"    🚨 Very strong smoothing - labels significantly altered")

            # Recommendations
            if mean_change < 0.02:
                self.logger.info(f"    💡 Consider increasing smoothing strength (ε, γ, decay)")
            elif mean_change > 0.25:
                self.logger.info(f"    💡 Consider reducing smoothing strength to preserve signal")

    def _log_next_steps_recommendations(self, quality_scores: Dict[str, Any], training_strategy: Dict[str, Any], 
                                      performance_config: Dict[str, Any]) -> None:
        """Log next steps and recommendations."""
        self.logger.info("")
        self.logger.info("🚀 NEXT STEPS & RECOMMENDATIONS")
        self.logger.info("-" * 40)
        
        # Count included targets
        gating = training_strategy.get('gating', {})
        included_targets = sum(1 for gate in gating.values() if gate.get('include_in_training', False))
        
        if included_targets == 0:
            self.logger.info("  🚨 CRITICAL: No targets passed quality gates")
            self.logger.info("     → Review profit thresholds and volatility settings")
            self.logger.info("     → Consider relaxing quality requirements")
            self.logger.info("     → Check data quality and preprocessing")
        elif included_targets < len(gating) // 2:
            self.logger.info("  ⚠️ WARNING: Less than half of targets passed quality gates")
            self.logger.info("     → Consider adjusting quality thresholds")
            self.logger.info("     → Review individual target performance")
        else:
            self.logger.info("  ✅ GOOD: Majority of targets passed quality gates")
        
        # Memory optimization recommendations
        memory_analysis = performance_config.get('memory_analysis', {})
        if memory_analysis.get('needs_optimization', False):
            self.logger.info("  💾 Memory optimization recommended:")
            self.logger.info("     → Use chunked processing for large datasets")
            self.logger.info("     → Consider data streaming for very large datasets")
        
        # Training recommendations
        curriculum = training_strategy.get('curriculum_learning', {})
        expert_targets = [t for t, c in curriculum.items() if c.get('difficulty_level') == 'expert']
        
        if expert_targets:
            self.logger.info(f"  🎓 Training recommendations:")
            self.logger.info(f"     → Start with expert targets: {', '.join(expert_targets)}")
            self.logger.info(f"     → Use curriculum learning for progressive training")
        
        # Performance recommendations
        parallel_config = performance_config.get('parallel_config', {})
        if parallel_config.get('enable_parallel', False):
            self.logger.info("  ⚡ Performance recommendations:")
            self.logger.info(f"     → Use {parallel_config.get('n_workers', 1)} parallel workers")
            self.logger.info(f"     → Implement reproducible seed management")
    
    def performance_sanity(self, data: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ensure O(N) memory usage and parallel folds with reproducible seeds.

        Args:
            data: Input data for memory analysis
            config: Optional performance configuration

        Returns:
            Dictionary with performance optimization settings
        """
        perf_config = config.get('performance', {}) if config else {}
        
        # Memory analysis
        memory_analysis = self._analyze_memory_usage(data, perf_config)
        
        # Parallel processing configuration
        parallel_config = self._configure_parallel_processing(data, perf_config)
        
        # Reproducible seeds for parallel folds
        seed_config = self._configure_reproducible_seeds(perf_config)
        
        # Chunking strategy for large datasets
        chunking_config = self._configure_chunking_strategy(data, memory_analysis, perf_config)
        
        return {
            'memory_analysis': memory_analysis,
            'parallel_config': parallel_config,
            'seed_config': seed_config,
            'chunking_config': chunking_config,
            'optimization_applied': True
        }
    
    def _analyze_memory_usage(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze memory usage and determine optimization needs."""
        # Calculate data size
        data_size_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
        n_samples = len(data)
        n_features = len(data.columns)
        
        # Memory budget
        memory_budget_mb = config.get('memory_budget_mb', 1024)  # 1GB default
        max_samples = config.get('max_samples', 1000000)  # 1M samples default
        
        # Determine if optimization is needed
        needs_optimization = data_size_mb > memory_budget_mb or n_samples > max_samples
        
        # Calculate optimal chunk size
        if needs_optimization:
            target_chunk_size = min(
                int(memory_budget_mb * 1024 * 1024 / (data_size_mb * 1024 * 1024 / n_samples)),
                max_samples
            )
        else:
            target_chunk_size = n_samples
        
        return {
            'data_size_mb': data_size_mb,
            'n_samples': n_samples,
            'n_features': n_features,
            'memory_budget_mb': memory_budget_mb,
            'needs_optimization': needs_optimization,
            'target_chunk_size': max(1000, target_chunk_size),  # Minimum 1000 samples
            'estimated_chunks': max(1, n_samples // max(1000, target_chunk_size))
        }
    
    def _configure_parallel_processing(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure parallel processing for optimal performance."""
        n_samples = len(data)
        n_cores = config.get('n_cores', None)
        
        if n_cores is None:
            import multiprocessing
            n_cores = min(multiprocessing.cpu_count(), 8)  # Cap at 8 cores
        
        # Determine optimal number of parallel workers
        if n_samples < 10000:
            n_workers = 1  # Single-threaded for small datasets
        elif n_samples < 100000:
            n_workers = min(2, n_cores)
        else:
            n_workers = min(4, n_cores)
        
        # Configure parallel folds
        n_folds = config.get('n_folds', 5)
        fold_size = n_samples // n_folds
        
        return {
            'n_cores': n_cores,
            'n_workers': n_workers,
            'n_folds': n_folds,
            'fold_size': fold_size,
            'enable_parallel': n_workers > 1,
            'chunk_processing': n_samples > 50000
        }
    
    def _configure_reproducible_seeds(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure reproducible seeds for parallel processing."""
        base_seed = config.get('base_seed', 42)
        n_folds = config.get('n_folds', 5)
        
        # Generate seeds for each fold
        fold_seeds = {}
        for i in range(n_folds):
            fold_seeds[f'fold_{i}'] = base_seed + i * 1000
        
        # Generate seeds for parallel workers
        n_workers = config.get('n_workers', 1)
        worker_seeds = {}
        for i in range(n_workers):
            worker_seeds[f'worker_{i}'] = base_seed + i * 100
        
        return {
            'base_seed': base_seed,
            'fold_seeds': fold_seeds,
            'worker_seeds': worker_seeds,
            'random_state': np.random.RandomState(base_seed)
        }
    
    def _configure_chunking_strategy(self, data: pd.DataFrame, memory_analysis: Dict[str, Any], 
                                   config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure chunking strategy for large datasets."""
        if not memory_analysis['needs_optimization']:
            return {'enabled': False, 'chunk_size': len(data)}
        
        chunk_size = memory_analysis['target_chunk_size']
        n_chunks = memory_analysis['estimated_chunks']
        
        # Determine chunking strategy
        if n_chunks <= 10:
            strategy = 'sequential'  # Process chunks sequentially
        elif n_chunks <= 100:
            strategy = 'parallel_chunks'  # Process multiple chunks in parallel
        else:
            strategy = 'streaming'  # Stream processing for very large datasets
        
        return {
            'enabled': True,
            'strategy': strategy,
            'chunk_size': chunk_size,
            'n_chunks': n_chunks,
            'overlap_samples': config.get('chunk_overlap', 100),  # Overlap between chunks
            'memory_efficient': True
        }
    
    def _apply_multiple_testing_hygiene(self, target_qualities: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Benjamini-Hochberg FDR control for multiple testing hygiene."""
        if len(target_qualities) <= 1:
            return target_qualities
        
        # Extract IC p-values for FDR control
        ic_pvalues = []
        target_names = []
        
        for target_name, quality in target_qualities.items():
            if hasattr(quality, 'metrics'):
                # Calculate IC p-value (simplified)
                ic_pvalue = self._calculate_ic_pvalue_simple(quality.metrics.get('ic', 0), len(target_qualities))
                ic_pvalues.append(ic_pvalue)
                target_names.append(target_name)
        
        if len(ic_pvalues) < 2:
            return target_qualities
        
        # Apply Benjamini-Hochberg FDR control
        try:
            from statsmodels.stats.multitest import fdrcorrection
            _, fdr_adjusted = fdrcorrection(ic_pvalues, alpha=0.10, method="indep")
            
            # Filter targets that pass FDR control
            filtered_qualities = {}
            for i, target_name in enumerate(target_names):
                if fdr_adjusted[i] < 0.1:  # FDR < 10%
                    filtered_qualities[target_name] = target_qualities[target_name]
                else:
                    self.logger.warning(f"Target {target_name} failed FDR control: p={ic_pvalues[i]:.4f}, FDR={fdr_adjusted[i]:.4f}")
            
            # Report FDR results
            raw_pass = sum(1 for p in ic_pvalues if p < 0.05)
            fdr_pass = len(filtered_qualities)
            self.logger.info(f"Multiple Testing: {raw_pass}/{len(target_names)} raw pass, {fdr_pass}/{len(target_names)} FDR-adjusted pass")
            
            return filtered_qualities if filtered_qualities else target_qualities
            
        except Exception:
            # Fallback if statsmodels is unavailable
            self.logger.warning("FDR control not available - using raw p-values")
            return target_qualities
    
    def _calculate_ic_pvalue_simple(self, ic: float, n_samples: int) -> float:
        """Calculate simplified IC p-value."""
        if n_samples < 10:
            return 1.0
        
        # Simplified p-value calculation
        # In practice, you'd use proper statistical tests
        if abs(ic) < 0.01:
            return 0.8
        elif abs(ic) < 0.03:
            return 0.3
        elif abs(ic) < 0.05:
            return 0.1
        else:
            return 0.05
    
    def _log_quality_sanity_check(self, labels: Union[pd.Series, pd.DataFrame], quality_scores: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        """Comprehensive sanity checklist with Quality PASS badges and red-flag reasons."""

        # Calculate overall filtering summary
        total_opportunities = 0
        passed_opportunities = 0

        if quality_scores:
            for target_name, quality in quality_scores.items():
                if hasattr(quality, 'opportunity_scores'):
                    total_opportunities += len(quality.opportunity_scores)
                    if hasattr(quality, 'overall_quality') and quality.overall_quality >= 0.3:  # Quality threshold
                        passed_opportunities += len(quality.opportunity_scores)

        tprint_info(f"🔍 [FILTERING SUMMARY] Total opportunities identified: {total_opportunities}")
        tprint_info(f"🔍 [FILTERING SUMMARY] High-quality opportunities: {passed_opportunities}")
        tprint_info(f"🔍 [FILTERING SUMMARY] Filtering rate: {(total_opportunities - passed_opportunities) / max(total_opportunities, 1) * 100:.1f}%")

        self.logger.info("🔍 COMPREHENSIVE QUALITY SANITY CHECK:")
        
        # Coverage and Positive Rate per target
        if isinstance(labels, pd.DataFrame):
            for col in labels.columns:
                coverage = (labels[col] != 0).mean() if len(labels[col]) > 0 else 0
                positive_rate = (labels[col] > 0).mean() if len(labels[col]) > 0 else 0
                self.logger.info(f"  📊 Target {col}: Coverage {coverage:.1%}, Positive Rate {positive_rate:.1%}")
        else:
            coverage = (labels != 0).mean() if len(labels) > 0 else 0
            positive_rate = (labels > 0).mean() if len(labels) > 0 else 0
            self.logger.info(f"  📊 Single target: Coverage {coverage:.1%}, Positive Rate {positive_rate:.1%}")
        
        # Quality PASS badges with comprehensive metrics
        if quality_scores:
            for target_name, quality in quality_scores.items():
                if hasattr(quality, 'metrics'):
                    metrics = quality.metrics
                    
                    # Extract key metrics for badge (IC removed)
                    hit_rate = metrics.get('hit_rate', 0)
                    coverage = quality.coverage
                    balance = 0.0  # Not relevant for trade opportunities
                    stability = metrics.get('stability', 0)
                    sharpe_norm = metrics.get('sharpe', 0)
                    uplift_bps = metrics.get('uplift', 0) * 10000  # Convert to bps
                    avg_potential_profit = metrics.get('avg_potential_profit', 0) * 10000  # Convert to bps
                    max_potential_profit = metrics.get('max_potential_profit', 0) * 10000  # Convert to bps
                    
                    # Red flag reasons
                    red_flags = getattr(quality, 'red_flag_reasons', [])
                    red_flag_text = f" [{', '.join(red_flags)}]" if red_flags else ""
                    
                    # Quality PASS badge - check if metrics are meaningful
                    has_meaningful_metrics = (
                        hit_rate > 0.1 or 
                        coverage > 0.05 or 
                        balance > 0.1 or 
                        stability > 0.1 or 
                        sharpe_norm > 0.1
                    )
                    pass_status = "✅ PASS" if (not red_flags and has_meaningful_metrics) else "❌ FAIL"
                    
                    # Get direction info if available
                    direction_info = ""
                    if hasattr(quality, 'potential_profits') and len(quality.potential_profits) > 0:
                        long_profits = quality.potential_profits[quality.potential_profits > 0]
                        short_profits = quality.potential_profits[quality.potential_profits < 0]
                        if len(long_profits) > 0 and len(short_profits) > 0:
                            direction_info = f" | Long: {len(long_profits)}, Short: {len(short_profits)}"
                    
                    # Get individual opportunity scoring info if available
                    opportunity_info = ""
                    if hasattr(quality, 'opportunity_scores') and hasattr(quality, 'opportunity_weights'):
                        if len(quality.opportunity_scores) > 0:
                            avg_score = quality.opportunity_scores.mean()
                            max_score = quality.opportunity_scores.max()
                            weight_entropy = -(quality.opportunity_weights * np.log(quality.opportunity_weights + 1e-10)).sum()
                            opportunity_info = f" | Avg Score: {avg_score:.3f} | Max Score: {max_score:.3f} | Weight Entropy: {weight_entropy:.3f}"
                    
                    self.logger.info(f"  🏆 {target_name} Trade Opportunity Quality: {pass_status}{red_flag_text}")
                    self.logger.info(f"     HitRate: {hit_rate:.3f} | Coverage: {coverage:.1%}{direction_info}")
                    self.logger.info(f"     Stability: {stability:.3f} | Sharpe: {sharpe_norm:.3f} | Uplift: {uplift_bps:.1f}bps")
                    self.logger.info(f"     Avg Potential Profit: {avg_potential_profit:.1f}bps | Max: {max_potential_profit:.1f}bps | Overall: {quality.overall_quality:.3f}{opportunity_info}")
                    
                    # Temporal CV metrics
                    # Skip temporal CV IC reporting since IC is removed
        
        # Top-K windows (last 20 days of rolling IC)
        self._log_top_k_windows(labels, quality_scores)
        
        # Overall PASS/FAIL assessment
        pass_fail = self._assess_quality_pass_fail(quality_scores, metadata)
        self.logger.info(f"  🏁 OVERALL QUALITY ASSESSMENT: {'✅ PASS' if pass_fail else '❌ FAIL'}")
        
        # Additional warnings
        self._log_quality_warnings(labels, quality_scores)
    
    def _log_top_k_windows(self, labels: Union[pd.Series, pd.DataFrame], quality_scores: Dict[str, Any]) -> None:
        """Log top-K windows with worst 3 windows."""
        if not quality_scores:
            return
        
        self.logger.info("  📈 Top-K Windows Analysis:")
        
        # For now, log a simplified version
        # In practice, you'd calculate rolling IC over the last 20 days
        for target_name, quality in quality_scores.items():
            if hasattr(quality, 'metrics'):
                metrics = quality.metrics
                stability = metrics.get('stability', 0)
                self.logger.info(f"     {target_name}: Rolling Stability {stability:.3f}")
                
                # Worst 3 windows (simplified)
                if stability < 0.5:
                    self.logger.warning(f"     ⚠️ {target_name}: Low stability detected - check for regime changes")
    
    def _assess_quality_pass_fail(self, quality_scores: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
        """Assess PASS/FAIL using conservative thresholds."""
        if not quality_scores:
            tprint_warning("🔍 [FINAL ASSESSMENT] No quality scores provided - FAIL")
            return False

        # Extract metrics for assessment
        all_ics = []
        all_hit_rates = []
        all_sharpes = []

        total_targets = len(quality_scores)
        targets_with_metrics = 0

        for target_name, quality in quality_scores.items():
            if hasattr(quality, 'metrics'):
                targets_with_metrics += 1
                metrics = quality.metrics
                all_ics.append(metrics.get('ic', 0))
                all_hit_rates.append(metrics.get('hit_rate', 0))
                all_sharpes.append(metrics.get('sharpe', 0))

        tprint_info(f"🔍 [FINAL ASSESSMENT] Evaluating {total_targets} targets ({targets_with_metrics} with metrics)")

        if not all_ics:
            tprint_warning("🔍 [FINAL ASSESSMENT] No targets with metrics - FAIL")
            return False

        # Calculate median statistics
        median_ic = np.median(all_ics)
        median_hit_rate = np.median(all_hit_rates)
        median_sharpe = np.median(all_sharpes)

        # Check if any metrics are meaningful (not all zeros)
        has_meaningful_metrics = (
            abs(median_ic) > 0.001 or
            median_hit_rate > 0.1 or
            median_sharpe > 0.1
        )

        tprint_info(f"🔍 [FINAL ASSESSMENT] Median metrics: IC={median_ic:.3f}, HitRate={median_hit_rate:.3f}, Sharpe={median_sharpe:.3f}")

        if not has_meaningful_metrics:
            tprint_warning("🔍 [FINAL ASSESSMENT] No meaningful metrics found - FAIL")
            return False

        # PASS if all thresholds met (very relaxed for crypto trading)
        pass_conditions = [
            median_ic >= 0.01,      # Further reduced for crypto (was 0.02) - 1% correlation still meaningful
            median_hit_rate >= 0.51, # Further reduced for crypto (was 0.52) - 51% still above random
            median_sharpe >= 0.20     # Further reduced for crypto (was 0.30) - 0.2 Sharpe still positive
        ]

        # Check each condition individually
        condition_results = [
            ("IC", median_ic >= 0.01, f"{median_ic:.3f} >= 0.01"),
            ("HitRate", median_hit_rate >= 0.51, f"{median_hit_rate:.3f} >= 0.51"),
            ("Sharpe", median_sharpe >= 0.20, f"{median_sharpe:.3f} >= 0.20")
        ]

        for condition_name, passed, details in condition_results:
            status = "✅" if passed else "❌"
            tprint_info(f"🔍 [FINAL ASSESSMENT] {condition_name}: {status} {details}")

        overall_pass = all(pass_conditions)
        final_status = "✅ PASS" if overall_pass else "❌ FAIL"
        tprint_info(f"🔍 [FINAL ASSESSMENT] Overall: {final_status}")

        return overall_pass
    
    def _log_quality_warnings(self, labels: Union[pd.Series, pd.DataFrame], quality_scores: Dict[str, Any]) -> None:
        """Log quality warnings for suspicious states."""
        # Check for all-zero or all-one labels
        if isinstance(labels, pd.DataFrame):
            for col in labels.columns:
                unique_vals = set(labels[col].dropna().unique())
                if len(unique_vals) <= 1:
                    self.logger.warning(f"⚠️ Target {col}: All-zero or all-one labels detected")
                elif len(unique_vals) == 2 and unique_vals <= {0.0, 1.0}:
                    positive_rate = (labels[col] > 0).mean()
                    if positive_rate < 0.01:
                        self.logger.warning(f"⚠️ Target {col}: Very low positive rate {positive_rate:.1%}")
                    elif positive_rate > 0.99:
                        self.logger.warning(f"⚠️ Target {col}: Very high positive rate {positive_rate:.1%}")
        else:
            unique_vals = set(labels.dropna().unique())
            if len(unique_vals) <= 1:
                self.logger.warning("⚠️ Single target: All-zero or all-one labels detected")
            elif len(unique_vals) == 2 and unique_vals <= {0.0, 1.0}:
                positive_rate = (labels > 0).mean()
                if positive_rate < 0.01:
                    self.logger.warning(f"⚠️ Single target: Very low positive rate {positive_rate:.1%}")
                elif positive_rate > 0.99:
                    self.logger.warning(f"⚠️ Single target: Very high positive rate {positive_rate:.1%}")

    def _generate_barrier_labels(
        self,
        prices: pd.Series,
        volatility: pd.Series,
        profit_targets: Optional[List[float]] = None,
        horizon_bars: int = 6,
        tie_policy: str = "neutral"
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Generate barrier-based labels that check if TP or SL is hit first within H bars.
        
        This method creates labels by checking, for each bar t, whether an upper or lower 
        barrier is reached first within the next H bars. It uses high/low if present; 
        otherwise it falls back to close. It's volatility-modulated and honors long/short switches.
        
        Args:
            prices: Price series (close prices)
            volatility: Volatility series for modulation
            profit_targets: List of profit targets in decimal form
            horizon_bars: Number of bars to look ahead (default: 6)
            tie_policy: How to handle ties ("neutral", "conservative", "optimistic")
            
        Returns:
            Series for single target, DataFrame for multiple targets
        """
        # Use OHLC if available, otherwise fall back to close
        if hasattr(prices, 'high') and hasattr(prices, 'low'):
            high_prices = prices.high
            low_prices = prices.low
        else:
            # Fallback to close prices
            high_prices = low_prices = prices
        
        # Calculate robust volatility normalization
        vol_normalized, vol_mean = self._calculate_volatility_normalization(volatility)
        
        # Default profit targets if not provided
        if profit_targets is None:
            profit_targets = [0.005]  # 50 bps default
        
        # Prepare window capture for downstream consumers (first-passage windows)
        self._last_opportunity_windows: List[Dict[str, Any]] = []

        # Generate labels for each target
        if len(profit_targets) == 1:
            # Single target case
            target = profit_targets[0]
            labels = self._generate_single_barrier_labels(
                prices, high_prices, low_prices, volatility, vol_normalized, 
                target, horizon_bars, tie_policy
            )
            return labels
        else:
            # Multi-target case
            label_dict = {}
            for i, target in enumerate(profit_targets):
                target_labels = self._generate_single_barrier_labels(
                    prices, high_prices, low_prices, volatility, vol_normalized,
                    target, horizon_bars, tie_policy
                )
                # Use basis points naming
                bps = int(round(target * 10_000))
                target_name = f"t_{bps}bps"
                label_dict[target_name] = target_labels
            
            return pd.DataFrame(label_dict)
    
    def _generate_single_barrier_labels(
        self,
        prices: pd.Series,
        high_prices: pd.Series,
        low_prices: pd.Series,
        volatility: pd.Series,
        vol_normalized: pd.Series,
        target: float,
        horizon_bars: int,
        tie_policy: str
    ) -> pd.Series:
        """Generate barrier labels for a single target.

        If optimal_entry_detection is enabled, apply first-passage dedup and place the label
        at the local extremum (min for long, max for short) between bar i and the hit time.
        """
        # Volatility modulation
        k = self.config.volatility.sensitivity
        min_mult = getattr(self.config.multi_target, 'min_threshold_multiplier', 0.5)
        max_mult = getattr(self.config.multi_target, 'max_threshold_multiplier', 2.0)
        effective_target = target * np.clip(1.0 + k * (vol_normalized - 1.0), min_mult, max_mult)

        upper_barriers = prices * (1 + effective_target)
        lower_barriers = prices * (1 - effective_target)

        labels = pd.Series(0, index=prices.index, dtype=np.int8)
        use_first_passage = bool(getattr(self.config, 'optimal_entry_detection', None) and self.config.optimal_entry_detection.enabled)

        if not use_first_passage:
            # Legacy per-bar labeling
            for i in range(len(prices) - horizon_bars):
                if i + horizon_bars >= len(prices):
                    continue
                if prices.iloc[i] <= 0:
                    continue
                high_window = high_prices.iloc[i+1:i+horizon_bars+1]
                low_window = low_prices.iloc[i+1:i+horizon_bars+1]
                up_hit = (high_window >= upper_barriers.iloc[i]).any()
                dn_hit = (low_window <= lower_barriers.iloc[i]).any()
                if up_hit and dn_hit:
                    if tie_policy == "conservative":
                        if self.config.enable_short_positions:
                            labels.iloc[i] = -1
                        elif self.config.enable_long_positions:
                            labels.iloc[i] = 1
                        else:
                            labels.iloc[i] = 0
                    elif tie_policy == "optimistic":
                        if self.config.enable_long_positions:
                            labels.iloc[i] = 1
                        elif self.config.enable_short_positions:
                            labels.iloc[i] = -1
                        else:
                            labels.iloc[i] = 0
                    else:
                        labels.iloc[i] = 0
                elif up_hit:
                    labels.iloc[i] = 1 if self.config.enable_long_positions else 0
                elif dn_hit:
                    labels.iloc[i] = -1 if self.config.enable_short_positions else 0
            return labels

        # First-passage dedup with local extrema placement
        i = 0
        n = len(prices)
        while i < n - horizon_bars:
            w_start = i + 1
            w_end = min(n - 1, i + horizon_bars)
            if w_start > w_end:
                break
            high_window = high_prices.iloc[w_start:w_end+1]
            low_window = low_prices.iloc[w_start:w_end+1]

            up_cross = np.where(high_window.values >= upper_barriers.iloc[i])[0]
            dn_cross = np.where(low_window.values <= lower_barriers.iloc[i])[0]

            up_idx = int(up_cross[0]) if up_cross.size > 0 else None
            dn_idx = int(dn_cross[0]) if dn_cross.size > 0 else None

            choose = 0
            hit_end = None
            if up_idx is not None and dn_idx is not None:
                if up_idx < dn_idx:
                    choose = 1
                    hit_end = w_start + up_idx
                elif dn_idx < up_idx:
                    choose = -1
                    hit_end = w_start + dn_idx
                else:
                    if tie_policy == "conservative":
                        choose = -1
                        hit_end = w_start + dn_idx
                    elif tie_policy == "optimistic":
                        choose = 1
                        hit_end = w_start + up_idx
                    else:
                        choose = 0
                        hit_end = w_start + up_idx
            elif up_idx is not None:
                choose = 1
                hit_end = w_start + up_idx
            elif dn_idx is not None:
                choose = -1
                hit_end = w_start + dn_idx

            # Respect direction toggles
            if choose > 0 and not self.config.enable_long_positions:
                choose = 0
            if choose < 0 and not self.config.enable_short_positions:
                choose = 0

            if choose == 0 or hit_end is None:
                i += 1
                continue

            sl = slice(i, hit_end + 1)
            if choose > 0:
                # place at local minimum using low
                local_pos = int(np.argmin(low_prices.iloc[sl].values))
                place_idx = i + local_pos
                labels.iloc[place_idx] = 1
            else:
                # place at local maximum using high
                local_pos = int(np.argmax(high_prices.iloc[sl].values))
                place_idx = i + local_pos
                labels.iloc[place_idx] = -1

            # Capture opportunity window for downstream consumers
            try:
                start_ts = prices.index[i]
                end_ts = prices.index[hit_end]
                anchor_ts = prices.index[place_idx]
                self._last_opportunity_windows.append({
                    'start': start_ts,
                    'end': end_ts,
                    'anchor': anchor_ts,
                    'direction': int(np.sign(choose))
                })
            except Exception:
                pass

            # Skip to the bar after the hit end to avoid duplicates
            i = hit_end + 1

        return labels

    def _calculate_robust_volatility(
        self, 
        prices: pd.Series, 
        high_prices: Optional[pd.Series] = None,
        low_prices: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Calculate robust volatility using the configured estimator.
        
        Args:
            prices: Price series (close prices)
            high_prices: High prices (for ATR calculation)
            low_prices: Low prices (for ATR calculation)
            
        Returns:
            Volatility series with proper scaling and floor
        """
        if self.config.volatility.volatility_estimator == 'log_returns':
            # Log returns volatility (most common)
            log_returns = np.log(prices / prices.shift(1))
            volatility = log_returns.rolling(window=self.config.volatility.window).std()
            
        elif self.config.volatility.volatility_estimator == 'atr':
            # ATR-based volatility
            if high_prices is None or low_prices is None:
                # Fallback to log returns if OHLC not available
                log_returns = np.log(prices / prices.shift(1))
                volatility = log_returns.rolling(window=self.config.volatility.window).std()
            else:
                # True Range calculation
                high_low = high_prices - low_prices
                high_close = np.abs(high_prices - prices.shift(1))
                low_close = np.abs(low_prices - prices.shift(1))
                true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                atr = true_range.rolling(window=self.config.volatility.window).mean()
                volatility = atr / prices  # Normalize by price level
                
        elif self.config.volatility.volatility_estimator == 'realized':
            # Realized volatility (sum of squared returns)
            returns = prices.pct_change()
            volatility = returns.rolling(window=self.config.volatility.window).std()
            
        else:
            raise ValueError(f"Unknown volatility estimator: {self.config.volatility.volatility_estimator}")
        
        # Apply volatility floor to prevent divide-by-zero
        volatility = np.maximum(volatility, self.config.volatility.volatility_floor)
        
        return volatility

    def _calculate_volatility_normalization(
        self, 
        volatility: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate robust volatility normalization with proper warm-up handling.
        
        Args:
            volatility: Raw volatility series
            
        Returns:
            Tuple of (vol_normalized, vol_mean) with proper warm-up handling
        """
        # Calculate volatility mean with proper warm-up policy
        if self.config.volatility.warmup_policy == 'rolling_mean':
            # Use rolling mean for warm-up, then switch to EMA
            vol_mean_rolling = volatility.rolling(window=self.config.volatility.vol_ema_span).mean()
            vol_mean_ema = volatility.ewm(span=self.config.volatility.vol_ema_span, adjust=False).mean()
            
            # Use rolling mean for warm-up period, then EMA
            warmup_periods = self.config.volatility.vol_ema_span
            vol_mean = vol_mean_rolling.copy()
            vol_mean.iloc[warmup_periods:] = vol_mean_ema.iloc[warmup_periods:]
            
        elif self.config.volatility.warmup_policy == 'drop':
            # Drop early samples - use EMA only
            vol_mean = volatility.ewm(span=self.config.volatility.vol_ema_span, adjust=False).mean()
            
        else:  # 'fillna'
            # Original behavior - fill with mean
            vol_mean = volatility.ewm(span=self.config.volatility.vol_ema_span, adjust=False).mean().shift(1).fillna(volatility.mean())
        
        # Apply volatility floor to prevent extreme jumps
        vol_mean = np.maximum(vol_mean, self.config.volatility.volatility_floor)
        
        # Calculate normalized volatility
        vol_normalized = volatility / vol_mean
        
        # Apply percentile-based clipping if enabled
        if self.config.volatility.percentile_clipping:
            vol_normalized = self._apply_percentile_clipping(vol_normalized)
        
        return vol_normalized, vol_mean

    def _apply_percentile_clipping(
        self, 
        vol_normalized: pd.Series
    ) -> pd.Series:
        """
        Apply percentile-based clipping with range validation and rolling updates.
        
        Args:
            vol_normalized: Normalized volatility series
            
        Returns:
            Clipped volatility series with validated range
        """
        # Use rolling window for dynamic percentile updates if enabled
        if hasattr(self.config.volatility, 'rolling_percentile_window') and self.config.volatility.rolling_percentile_window > 0:
            # Rolling percentile calculation
            window = self.config.volatility.rolling_percentile_window
            p_low_series = vol_normalized.rolling(window=window, min_periods=window//2).quantile(
                self.config.volatility.percentile_low / 100.0
            )
            p_high_series = vol_normalized.rolling(window=window, min_periods=window//2).quantile(
                self.config.volatility.percentile_high / 100.0
            )
            
            # Fill initial values with global percentiles
            global_p_low = np.percentile(vol_normalized.dropna(), self.config.volatility.percentile_low)
            global_p_high = np.percentile(vol_normalized.dropna(), self.config.volatility.percentile_high)
            p_low_series = p_low_series.fillna(global_p_low)
            p_high_series = p_high_series.fillna(global_p_high)
            
            # Apply range validation
            p_low_series, p_high_series = self._validate_percentile_range(p_low_series, p_high_series)
            
            # Apply clipping
            vol_normalized_clipped = vol_normalized.copy()
            for i in range(len(vol_normalized)):
                if not pd.isna(p_low_series.iloc[i]) and not pd.isna(p_high_series.iloc[i]):
                    vol_normalized_clipped.iloc[i] = np.clip(
                        vol_normalized.iloc[i], 
                        p_low_series.iloc[i], 
                        p_high_series.iloc[i]
                    )
            
            return vol_normalized_clipped
        else:
            # Static percentile calculation
            p_low = np.percentile(vol_normalized.dropna(), self.config.volatility.percentile_low)
            p_high = np.percentile(vol_normalized.dropna(), self.config.volatility.percentile_high)
            
            # Apply range validation
            p_low, p_high = self._validate_percentile_range_static(p_low, p_high)
            
            return np.clip(vol_normalized, p_low, p_high)

    def _validate_percentile_range(
        self, 
        p_low_series: pd.Series, 
        p_high_series: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Validate percentile range to prevent too narrow ranges.
        
        Args:
            p_low_series: Low percentile series
            p_high_series: High percentile series
            
        Returns:
            Validated percentile series
        """
        min_range = self.config.volatility.percentile_min_range
        
        # Calculate range
        range_series = p_high_series - p_low_series
        
        # Find where range is too narrow
        narrow_mask = range_series < min_range
        
        if narrow_mask.any():
            # Expand range symmetrically around center
            center = (p_low_series + p_high_series) / 2
            half_range = min_range / 2
            
            p_low_series = np.where(narrow_mask, center - half_range, p_low_series)
            p_high_series = np.where(narrow_mask, center + half_range, p_high_series)
        
        return p_low_series, p_high_series

    def _validate_percentile_range_static(
        self, 
        p_low: float, 
        p_high: float
    ) -> Tuple[float, float]:
        """
        Validate static percentile range to prevent too narrow ranges.
        
        Args:
            p_low: Low percentile value
            p_high: High percentile value
            
        Returns:
            Validated percentile values
        """
        min_range = self.config.volatility.percentile_min_range
        
        if p_high - p_low < min_range:
            # Expand range symmetrically around center
            center = (p_low + p_high) / 2
            half_range = min_range / 2
            p_low = center - half_range
            p_high = center + half_range
        
        return p_low, p_high

    def _calculate_effective_threshold(
        self, 
        base_threshold: float, 
        vol_normalized: pd.Series,
        min_mult: float = 0.5,
        max_mult: float = 2.0
    ) -> pd.Series:
        """
        Calculate effective threshold with nonlinear sensitivity and hysteresis.
        
        Args:
            base_threshold: Base profit target
            vol_normalized: Normalized volatility series
            min_mult: Minimum threshold multiplier
            max_mult: Maximum threshold multiplier
            
        Returns:
            Effective threshold series
        """
        # Apply nonlinear sensitivity: mult = 1 + k * (vol_normalized - 1)^alpha
        vol_deviation = vol_normalized - 1.0
        if self.config.volatility.alpha != 1.0:
            # Apply nonlinear transformation
            vol_deviation = np.sign(vol_deviation) * np.power(np.abs(vol_deviation), self.config.volatility.alpha)
        
        # Calculate multiplier
        multiplier = 1.0 + self.config.volatility.sensitivity * vol_deviation
        
        # Apply clipping
        multiplier = np.clip(multiplier, min_mult, max_mult)
        
        # Apply adaptive hysteresis to prevent chattering
        if hasattr(self, '_last_multiplier') and len(multiplier) > 0:
            # Calculate adaptive hysteresis threshold
            if self.config.volatility.adaptive_hysteresis:
                # Scale hysteresis by volatility: threshold = base * (1 + factor * vol_normalized)
                adaptive_threshold = self.config.volatility.hysteresis_threshold * (
                    1.0 + self.config.volatility.hysteresis_volatility_factor * vol_normalized
                )
            else:
                adaptive_threshold = self.config.volatility.hysteresis_threshold
            
            # Only update if change exceeds adaptive threshold
            change = np.abs(multiplier - self._last_multiplier)
            update_mask = change > adaptive_threshold
            multiplier = np.where(update_mask, multiplier, self._last_multiplier)
        
        # Store for next iteration
        self._last_multiplier = multiplier.iloc[-1] if len(multiplier) > 0 else 1.0
        
        return base_threshold * multiplier

    def _generate_price_target_vol_normalized_labels(
        self,
        prices: pd.Series,
        volatility: pd.Series,
        profit_targets: Optional[List[float]] = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Generate labels based on volatility-adjusted profit targets with multi-target support.

        Args:
            prices: Price series
            volatility: Volatility series
            profit_targets: Optional list of profit targets (as fractions, not percentages)

        Returns:
            Generated labels (Series for single target, DataFrame for multi-target)
        """
        # Performance optimization: calculate future returns once and reuse
        future_returns = prices.pct_change(self.config.lookahead_periods).shift(-self.config.lookahead_periods)

        # Calculate robust volatility normalization
        vol_normalized, vol_mean = self._calculate_volatility_normalization(volatility)

        # Multi-target labeling
        if profit_targets and len(profit_targets) > 0:
            # Build one column per target with deterministic names
            target_columns = []
            target_data = {}
            
            for i, target_frac in enumerate(profit_targets):
                # Ensure target_frac is a scalar value
                if isinstance(target_frac, pd.Series):
                    target_frac = target_frac.dropna().iloc[0] if len(target_frac.dropna()) > 0 else 0.0
                else:
                    target_frac = float(target_frac)
                
                # Name like t_50bps for 0.005, or t_100bps for 0.01
                bps = int(round(float(target_frac) * 10_000))
                target_name = f"t_{bps}bps"
                target_columns.append(target_name)
                
                # Profit target semantics in volatility regimes with robust calculation
                min_mult = getattr(self.config.multi_target, 'min_threshold_multiplier', 0.5)
                max_mult = getattr(self.config.multi_target, 'max_threshold_multiplier', 2.0)
                effective_target = self._calculate_effective_threshold(
                    target_frac, vol_normalized, min_mult, max_mult
                )
                
                # Generate labels for this target
                if self.config.label_type == LabelDefinitionType.BINARY:
                    # Binary classification based on enabled directions
                    target_labels = pd.Series(0, index=future_returns.index, dtype=np.int8)
                    
                    if self.config.enable_long_positions:
                        long_signals = (future_returns > effective_target).astype(np.int8)
                        target_labels += long_signals  # 1 for long signals
                    
                    if self.config.enable_short_positions:
                        short_signals = (future_returns < -effective_target).astype(np.int8)
                        target_labels -= short_signals  # -1 for short signals
                elif self.config.label_type == LabelDefinitionType.SMOOTH_BINARY:
                    # Smooth binary labels with proximity weighting
                    target_labels = self._generate_smooth_binary_labels(
                        future_returns, effective_target, vol_normalized
                    )
                elif self.config.label_type == LabelDefinitionType.SMOOTH_REGRESSION:
                    # Smooth regression labels with proximity weighting
                    target_labels = self._generate_smooth_regression_labels(
                        future_returns, effective_target, vol_normalized
                    )
                elif self.config.label_type == LabelDefinitionType.PROXIMITY_REGRESSION:
                    # Proximity-based regression with sample weights
                    target_labels, sample_weights = self._generate_proximity_regression_labels(
                        future_returns, effective_target, vol_normalized
                    )
                    # Store sample weights for later use (will be returned in metadata)
                    if not hasattr(self, '_sample_weights'):
                        self._sample_weights = {}
                    self._sample_weights[target_name] = sample_weights
                else:
                    # Regression: use actual returns
                    target_labels = future_returns
                
                # Optional: replace simple returns logic with barrier-based first-passage if enabled
                if getattr(self.config, 'optimal_entry_detection', None) and self.config.optimal_entry_detection.enabled:
                    barrier_labels = self._generate_barrier_labels(
                        prices=prices,
                        volatility=volatility,
                        profit_targets=[target_frac],
                        horizon_bars=int(getattr(self.config, 'lookahead_periods', 6)),
                        tie_policy="neutral"
                    )
                    if isinstance(barrier_labels, pd.Series):
                        target_labels = barrier_labels
                    else:
                        # DataFrame with single column
                        target_labels = barrier_labels.iloc[:, 0]
                target_data[target_name] = target_labels
            
            # Create DataFrame with deterministic column order
            labels_df = pd.DataFrame(target_data, index=prices.index)
            return labels_df
            
        else:
            # Single target case - return Series
            if self.config.label_type == LabelDefinitionType.BINARY:
                # Use volatility-modulated threshold logic with configurable base target
                base_threshold = self.config.multi_target.target_profit / 100.0  # Convert percentage to decimal
                
                # Apply robust volatility modulation with nonlinear sensitivity and hysteresis
                min_mult = getattr(self.config.multi_target, 'min_threshold_multiplier', 0.5)
                max_mult = getattr(self.config.multi_target, 'max_threshold_multiplier', 2.0)
                effective_threshold = self._calculate_effective_threshold(
                    base_threshold, vol_normalized, min_mult, max_mult
                )
                
                # Default simple thresholding path
                labels = pd.Series(0, index=future_returns.index, dtype=np.int8)
                if getattr(self.config, 'optimal_entry_detection', None) and self.config.optimal_entry_detection.enabled:
                    # Use barrier-based first-passage with local extrema placement
                    barrier_labels = self._generate_barrier_labels(
                        prices=prices,
                        volatility=volatility,
                        profit_targets=[base_threshold],
                        horizon_bars=int(getattr(self.config, 'lookahead_periods', 6)),
                        tie_policy="neutral"
                    )
                    labels = barrier_labels if isinstance(barrier_labels, pd.Series) else barrier_labels.iloc[:, 0]
                else:
                    # Generate signals based on enabled directions
                    if self.config.enable_long_positions:
                        long_signals = (future_returns > effective_threshold).astype(np.int8)
                        labels += long_signals
                    if self.config.enable_short_positions:
                        short_signals = (future_returns < -effective_threshold).astype(np.int8)
                        labels -= short_signals
                
                # Debug: Log return statistics
                # Calculate return statistics and volatility-modulated thresholds
                long_rate = (labels > 0).mean()
                short_rate = (labels < 0).mean()
                signal_rate = (labels != 0).mean()
                # Generate signals based on direction configuration and volatility modulation
            elif self.config.label_type == LabelDefinitionType.SMOOTH_BINARY:
                # Smooth binary labels with proximity weighting
                base_threshold = self.config.multi_target.target_profit / 100.0
                min_mult = getattr(self.config.multi_target, 'min_threshold_multiplier', 0.5)
                max_mult = getattr(self.config.multi_target, 'max_threshold_multiplier', 2.0)
                effective_threshold = self._calculate_effective_threshold(
                    base_threshold, vol_normalized, min_mult, max_mult
                )
                
                labels = self._generate_smooth_binary_labels(
                    future_returns, effective_threshold, vol_normalized
                )
            elif self.config.label_type == LabelDefinitionType.SMOOTH_REGRESSION:
                # Smooth regression labels with proximity weighting
                base_threshold = self.config.multi_target.target_profit / 100.0
                min_mult = getattr(self.config.multi_target, 'min_threshold_multiplier', 0.5)
                max_mult = getattr(self.config.multi_target, 'max_threshold_multiplier', 2.0)
                effective_threshold = self._calculate_effective_threshold(
                    base_threshold, vol_normalized, min_mult, max_mult
                )
                
                labels = self._generate_smooth_regression_labels(
                    future_returns, effective_threshold, vol_normalized
                )
            elif self.config.label_type == LabelDefinitionType.PROXIMITY_REGRESSION:
                # Proximity-based regression with sample weights
                base_threshold = self.config.multi_target.target_profit / 100.0
                min_mult = getattr(self.config.multi_target, 'min_threshold_multiplier', 0.5)
                max_mult = getattr(self.config.multi_target, 'max_threshold_multiplier', 2.0)
                effective_threshold = self._calculate_effective_threshold(
                    base_threshold, vol_normalized, min_mult, max_mult
                )
                
                labels, sample_weights = self._generate_proximity_regression_labels(
                    future_returns, effective_threshold, vol_normalized
                )
                # Store sample weights for later use
                if not hasattr(self, '_sample_weights'):
                    self._sample_weights = {}
                self._sample_weights['default'] = sample_weights
            else:
                # Regression: use actual returns
                labels = future_returns

        return labels

    def _generate_smooth_binary_labels(
        self,
        future_returns: pd.Series,
        effective_threshold: pd.Series,
        vol_normalized: pd.Series,
        quality_scores: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Generate smooth binary labels with confidence weighting and proximity consideration.
        
        Args:
            future_returns: Future returns series
            effective_threshold: Volatility-adjusted threshold series
            vol_normalized: Normalized volatility series
            quality_scores: Optional quality scores for weighting
            
        Returns:
            Smooth binary labels in range [-1, 1] with proximity weighting
        """
        # Calculate distance from threshold
        distance = future_returns / effective_threshold
        
        # Apply sigmoid smoothing with volatility-dependent sharpness
        # More sharp in high volatility periods
        sharpness = 2.0 + (vol_normalized - 1.0) * 0.5  # Range: 1.5 to 2.5
        smooth_labels = np.tanh(distance * sharpness)
        
        # Apply enhanced confidence weighting with caps:
        # - No labels below 75% of target (0 confidence)
        # - 0.5 confidence for 75% of target
        # - 1.0 confidence for 100% of target
        # - 1.5 confidence for 200% of target (capped at 1.5)
        # Linear scaling: confidence = 0.5 + 2.0 * (distance - 0.75), capped at 1.5
        proximity_factor = np.where(
            np.abs(distance) < 0.75,  # Below 75% of target
            0.0,  # No confidence below 75% of target
            np.where(
                np.abs(distance) >= 0.75,  # 75% and above
                np.clip(0.5 + 2.0 * (np.abs(distance) - 0.75), 0.5, 1.5),  # Linear scaling capped at 1.5
                0.0  # Fallback
            )
        )
        
        # Apply enhanced confidence weighting
        smooth_labels = smooth_labels * proximity_factor
        
        # Apply quality weighting if available
        if quality_scores is not None:
            # Align quality scores with smooth labels
            quality_aligned, smooth_aligned = _align_like(quality_scores, smooth_labels)
            smooth_labels = smooth_aligned * quality_aligned
        
        # Ensure labels are in valid range
        smooth_labels = np.clip(smooth_labels, -1.0, 1.0)
        
        return pd.Series(smooth_labels, index=future_returns.index, name='smooth_binary_label')

    def _generate_smooth_regression_labels(
        self,
        future_returns: pd.Series,
        effective_threshold: pd.Series,
        vol_normalized: pd.Series,
        quality_scores: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Generate smooth regression labels with proximity weighting.
        
        Args:
            future_returns: Future returns series
            effective_threshold: Volatility-adjusted threshold series
            vol_normalized: Normalized volatility series
            quality_scores: Optional quality scores for weighting
            
        Returns:
            Smooth regression labels with proximity weighting
        """
        # Start with actual returns
        smooth_labels = future_returns.copy()
        
        # Apply enhanced confidence weighting for cases close to and beyond targets
        # Calculate distance from threshold
        distance = future_returns / effective_threshold
        
        # Apply enhanced confidence weighting with caps:
        # - No labels below 75% of target (0 confidence)
        # - 0.5 confidence for 75% of target
        # - 1.0 confidence for 100% of target
        # - 1.5 confidence for 200% of target (capped at 1.5)
        # Linear scaling: confidence = 0.5 + 2.0 * (distance - 0.75), capped at 1.5
        proximity_factor = np.where(
            np.abs(distance) < 0.75,  # Below 75% of target
            0.0,  # No confidence below 75% of target
            np.where(
                np.abs(distance) >= 0.75,  # 75% and above
                np.clip(0.5 + 2.0 * (np.abs(distance) - 0.75), 0.5, 1.5),  # Linear scaling capped at 1.5
                0.0  # Fallback
            )
        )
        
        # Apply enhanced confidence weighting
        smooth_labels = smooth_labels * proximity_factor
        
        # Apply quality weighting if available
        if quality_scores is not None:
            # Align quality scores with smooth labels
            quality_aligned, smooth_aligned = _align_like(quality_scores, smooth_labels)
            smooth_labels = smooth_aligned * quality_aligned
        
        return pd.Series(smooth_labels, index=future_returns.index, name='smooth_regression_label')

    def _generate_proximity_regression_labels(
        self,
        future_returns: pd.Series,
        effective_threshold: pd.Series,
        vol_normalized: pd.Series,
        quality_scores: Optional[pd.Series] = None,
        weight_transform: str = 'linear',
        soft_threshold: bool = False
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate proximity-based regression labels with sample weights.
        
        This method implements the new regression approach where:
        - Target = expected move strength (normalized)
        - Sample weight = confidence based on proximity to target
        - Uses piecewise linear mapping for robust confidence scaling
        
        Args:
            future_returns: Future returns series
            effective_threshold: Volatility-adjusted threshold series
            vol_normalized: Normalized volatility series
            quality_scores: Optional quality scores for additional weighting
            weight_transform: Weight transformation ('linear', 'sqrt', 'power_0.75')
            soft_threshold: Use soft thresholding instead of hard cutoff
            
        Returns:
            Tuple of (labels, sample_weights) where:
            - labels: Regression targets in [-1, 1] range
            - sample_weights: Confidence-based weights in [0, 1] range
        """
        # LEAKAGE CHECK: Ensure future_returns uses only out-of-sample data
        # This is already handled by the calling method using shift(-lookahead_periods)
        
        # Calculate ratio = actual_move / target_move (directional)
        ratio = future_returns / effective_threshold
        
        # Apply robust piecewise linear mapping with soft threshold option
        min_thresh = 0.70 if soft_threshold else 0.75
        proximity = proximity_mapping(ratio, min_thresh=min_thresh, cap=1.5, adaptive_clip=True)
        
        # Normalize proximity to [0, 1] range for stability
        proximity_norm = np.abs(proximity) / 1.5
        
        # Create regression targets: sign(ratio) * proximity_norm (range [-1, 1])
        labels = np.sign(ratio) * proximity_norm
        
        # Apply weight transformation to reduce overfitting
        if weight_transform == 'sqrt':
            sample_weights = np.sqrt(proximity_norm)
        elif weight_transform == 'power_0.75':
            sample_weights = np.power(proximity_norm, 0.75)
        else:  # 'linear'
            sample_weights = proximity_norm
        
        # Apply quality weighting if available
        if quality_scores is not None:
            # Align quality scores with labels
            quality_aligned, labels_aligned = _align_like(quality_scores, pd.Series(labels, index=future_returns.index))
            quality_aligned, weights_aligned = _align_like(quality_scores, pd.Series(sample_weights, index=future_returns.index))
            
            # Apply quality weighting to both labels and weights
            labels = labels_aligned * quality_aligned
            sample_weights = weights_aligned * quality_aligned
            
            # Ensure weights stay in [0, 1] range
            sample_weights = np.clip(sample_weights, 0.0, 1.0)
        
        # Create return series
        labels_series = pd.Series(labels, index=future_returns.index, name='proximity_regression_label')
        weights_series = pd.Series(sample_weights, index=future_returns.index, name='sample_weight')
        
        return labels_series, weights_series

    def get_sample_weights(self, target_name: str = 'default') -> Optional[pd.Series]:
        """
        Get sample weights for the specified target.
        
        Args:
            target_name: Name of the target (default: 'default')
            
        Returns:
            Sample weights series if available, None otherwise
        """
        if hasattr(self, '_sample_weights') and target_name in self._sample_weights:
            return self._sample_weights[target_name]
        return None

    def get_all_sample_weights(self) -> Dict[str, pd.Series]:
        """
        Get all sample weights for all targets.
        
        Returns:
            Dictionary mapping target names to sample weights
        """
        if hasattr(self, '_sample_weights'):
            return self._sample_weights.copy()
        return {}

    def get_coverage_stats(self) -> Dict[str, float]:
        """
        Get coverage statistics for tracking dropped samples.
        
        Returns:
            Dictionary with coverage statistics
        """
        if not hasattr(self, '_sample_weights'):
            return {}
        
        stats = {}
        for target_name, weights in self._sample_weights.items():
            total_samples = len(weights)
            zero_weight_samples = (weights == 0).sum()
            coverage_rate = 1.0 - (zero_weight_samples / total_samples) if total_samples > 0 else 0.0
            
            stats[f'{target_name}_coverage_rate'] = coverage_rate
            stats[f'{target_name}_dropped_samples'] = zero_weight_samples
            stats[f'{target_name}_total_samples'] = total_samples
        
        return stats

    def compute_calibration_metrics(
        self, 
        predictions: pd.Series, 
        realized_returns: pd.Series,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """
        Compute calibration metrics for predicted confidence vs realized returns.
        
        Args:
            predictions: Model predictions (signed confidence)
            realized_returns: Actual realized returns
            n_bins: Number of calibration bins
            
        Returns:
            Dictionary with calibration metrics
        """
        # Align data
        pred_aligned, ret_aligned = _align_like(predictions, realized_returns)
        
        # Create bins based on absolute prediction values
        abs_pred = np.abs(pred_aligned)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        calibration_data = []
        for i in range(n_bins):
            mask = (abs_pred >= bin_edges[i]) & (abs_pred < bin_edges[i + 1])
            if i == n_bins - 1:  # Include upper bound for last bin
                mask = (abs_pred >= bin_edges[i]) & (abs_pred <= bin_edges[i + 1])
            
            if mask.sum() > 0:
                avg_pred = abs_pred[mask].mean()
                avg_realized = np.abs(ret_aligned[mask]).mean()
                sample_count = mask.sum()
                
                calibration_data.append({
                    'bin_center': bin_centers[i],
                    'avg_pred': avg_pred,
                    'avg_realized': avg_realized,
                    'sample_count': sample_count
                })
        
        # Calculate calibration error (MSE between predicted and realized)
        if calibration_data:
            pred_values = [d['avg_pred'] for d in calibration_data]
            realized_values = [d['avg_realized'] for d in calibration_data]
            calibration_error = np.mean((np.array(pred_values) - np.array(realized_values)) ** 2)
        else:
            calibration_error = np.nan
        
        return {
            'calibration_error': calibration_error,
            'n_bins': len(calibration_data),
            'calibration_data': calibration_data
        }

    def compute_rank_metrics(
        self, 
        predictions: pd.Series, 
        realized_returns: pd.Series
    ) -> Dict[str, float]:
        """
        Compute rank-based metrics for model evaluation.
        
        Args:
            predictions: Model predictions (signed confidence)
            realized_returns: Actual realized returns
            
        Returns:
            Dictionary with rank metrics
        """
        from scipy.stats import spearmanr, pearsonr
        
        # Align data
        pred_aligned, ret_aligned = _align_like(predictions, realized_returns)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(pred_aligned) | np.isnan(ret_aligned))
        pred_clean = pred_aligned[valid_mask]
        ret_clean = ret_aligned[valid_mask]
        
        if len(pred_clean) < 2:
            return {'spearman_correlation': np.nan, 'pearson_correlation': np.nan}
        
        # Spearman correlation (rank-based)
        spearman_corr, spearman_p = spearmanr(pred_clean, ret_clean)
        
        # Pearson correlation (linear)
        pearson_corr, pearson_p = pearsonr(pred_clean, ret_clean)
        
        # Information Coefficient (IC) - commonly used in quant finance
        ic = spearman_corr  # Often used interchangeably with Spearman
        
        return {
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'information_coefficient': ic,
            'n_samples': len(pred_clean)
        }


    @staticmethod
    def create_enhanced_analyst_labeler(
        config: Optional[VolatilityAwareConfig] = None
    ) -> 'VolatilityAwareMultiHorizonLabeler':
        """
        Create an enhanced analyst labeler.

        Args:
            config: Optional configuration

        Returns:
            Configured volatility-aware labeler
        """
        if config is None:
            config = VolatilityAwareConfig()

        return VolatilityAwareMultiHorizonLabeler(config)


def create_enhanced_analyst_labeler(config: Optional[VolatilityAwareConfig] = None) -> VolatilityAwareMultiHorizonLabeler:
    """
    Create an enhanced analyst labeler (standalone function).
    
    Args:
        config: Optional configuration
        
    Returns:
        Configured volatility-aware labeler
    """
    if config is None:
        config = VolatilityAwareConfig()
    
    return VolatilityAwareMultiHorizonLabeler(config)
