"""
Advanced Filters for 15m Timeframe Profit Labeling

This module implements advanced filtering techniques specifically optimized for 15-minute
timeframe data to improve label quality and reduce noise in profit labeling.

Key Features:
1. Bar Efficiency Ratio - Measures directional price action vs. choppy conditions
2. Close-Location Value (CLV) - Tracks buying/selling pressure and control
3. ATR Volatility Ratio - Normalizes volatility for adaptive filtering
4. Trend Coherence Features - Ensures trend continuity and direction consistency

These filters are designed to work with the existing VolatilityAwareMultiHorizonLabeler
and integrate seamlessly with the current profit labeling pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import warnings

# Import existing utilities
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_mean, safe_std,
    validate_finite, validate_positive, validate_range, safe_correlation,
    safe_dataframe_operation, validate_dataframe_columns, get_dataframe_info,
    create_data_quality_report, optimize_memory, memory_checkpoint,
    integrate_with_m1_optimizers, get_m1_gpu_manager, get_m1_memory_optimizer
)
from src.utils.common_utilities import (
    analyze_nan_values_detailed, format_nan_analysis_report,
    create_data_quality_report as create_detailed_quality_report,
    get_dataframe_info as get_detailed_dataframe_info
)
from src.utils.math_validation import MathValidation, safe_divide as math_safe_divide
from src.utils.matrix_operations import (
    get_unified_matrix_operations, get_vectorized_processing_core,
    get_enhanced_matrix_operations, optimize_dataframe,
    vectorized_rolling_features, matrix_correlation_analysis,
    safe_correlation_matrix, compute_trading_indicators,
    get_hardware_performance_report
)
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager as get_gpu_manager
from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
from src.utils.serialization_utils import UniversalSerializer

# Import data utilities (optional)
try:
    from src.utils.data.klines_parquet import KlineParquetManager
    DATA_UTILS_AVAILABLE = True
except ImportError:
    DATA_UTILS_AVAILABLE = False
    KlineParquetManager = None

# Import ML common utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
    from src.utils.ml_common.validation.cross_validation import CrossValidator
    from src.utils.ml_common.validation.lookahead_bias_detector import LookaheadBiasDetector
    from src.utils.ml_common.optimization.pareto_optimizer import ParetoOptimizer

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

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False
    BayesianTPEOptimizer = None
    CrossValidator = None
    LookaheadBiasDetector = None
    ParetoOptimizer = None


class FilterType(Enum):
    """Enumeration of filter types."""
    EFFICIENCY_RATIO = "efficiency_ratio"
    CLV = "clv"
    ATR_RATIO = "atr_ratio"
    TREND_COHERENCE = "trend_coherence"
    COMBINED = "combined"


@dataclass
class AdvancedFiltersConfig:
    """Configuration for advanced 15m timeframe filters."""
    
    # Global enable/disable
    enabled: bool = True
    
    # Bar Efficiency Ratio settings
    enable_efficiency_ratio: bool = True
    efficiency_window: int = 3  # Rolling window for efficiency (2-4 bars = 30-60 minutes)
    efficiency_threshold_high: float = 0.6  # High efficiency = directional
    efficiency_threshold_low: float = 0.3   # Low efficiency = choppy
    
    # Close-Location Value (CLV) settings
    enable_clv: bool = True
    clv_window: int = 8  # Rolling window for CLV smoothing
    clv_threshold_positive: float = 0.2   # Sustained positive CLV = bullish
    clv_threshold_negative: float = -0.2  # Sustained negative CLV = bearish
    clv_volatility_threshold: float = 0.5  # Avoid when CLV fluctuates rapidly
    
    # ATR Volatility Ratio settings
    enable_atr_ratio: bool = True
    atr_short_window: int = 4   # Short-term ATR window (1 hour)
    atr_long_window: int = 20   # Long-term ATR window (5 hours)
    atr_ratio_threshold_high: float = 1.5  # Too jumpy
    # Removed atr_ratio_threshold_low - no "too quiet" filter
    
    # Trend Coherence settings
    enable_trend_coherence: bool = True
    direction_window: int = 8   # Window for direction consistency check
    min_direction_consistency: float = 0.6  # 60% of bars in same direction
    ema_period: int = 12        # EMA period for slope calculation
    min_slope_threshold: float = 0.001  # Minimum slope for trend continuity
    
    # Grading system (replaces cumulative filtering)
    use_grading_system: bool = True  # Use average grade instead of cumulative filters
    grade_threshold: float = 0.5  # Minimum average grade to pass (0.0-1.0)
    grade_weights: Dict[str, float] = field(default_factory=lambda: {
        'efficiency': 0.25,
        'clv': 0.25,
        'atr_ratio': 0.25,
        'trend_coherence': 0.25
    })
    
    # Legacy combined filtering (for backward compatibility)
    filter_type: FilterType = FilterType.COMBINED
    min_eligibility_ratio: float = 0.3  # Minimum ratio of eligible samples
    strict_mode: bool = False  # Use strict eligibility criteria
    
    # Quality checks
    min_eligible_samples: int = 50
    max_filter_failure_rate: float = 0.7  # Maximum allowed filter failure rate


@dataclass
class FilterResult:
    """Result container for advanced filtering."""
    
    # Core results
    eligibility_mask: pd.Series
    eligibility_ratio: float
    
    # Filter-specific results (legacy)
    efficiency_mask: Optional[pd.Series] = None
    clv_mask: Optional[pd.Series] = None
    atr_ratio_mask: Optional[pd.Series] = None
    trend_coherence_mask: Optional[pd.Series] = None
    
    # Grading system results
    average_grade: Optional[pd.Series] = None
    individual_grades: Dict[str, pd.Series] = field(default_factory=dict)
    
    # Statistics
    n_total_samples: int = 0
    n_eligible_samples: int = 0
    n_filtered_samples: int = 0
    
    # Filter performance
    filter_effectiveness: Dict[str, float] = field(default_factory=dict)
    filter_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Quality metrics
    overall_quality_score: float = 0.0
    noise_reduction_ratio: float = 0.0
    
    # Metadata
    config_used: AdvancedFiltersConfig = None
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class AdvancedFilters15m:
    """
    Advanced Filters for 15m Timeframe Profit Labeling
    
    This class implements sophisticated filtering techniques optimized for 15-minute
    timeframe data to improve label quality and reduce noise.
    
    Key Features:
    1. **Bar Efficiency Ratio**: Measures directional price action vs. choppy conditions
    2. **Close-Location Value (CLV)**: Tracks buying/selling pressure and control
    3. **ATR Volatility Ratio**: Normalizes volatility for adaptive filtering
    4. **Trend Coherence**: Ensures trend continuity and direction consistency
    """
    
    def __init__(self, config: Optional[AdvancedFiltersConfig] = None):
        """Initialize advanced filters for 15m timeframe."""
        self.config = config or AdvancedFiltersConfig()
        self.logger = logging.getLogger('AdvancedFilters15m')
        
        # Initialize matrix operations for enhanced data processing
        self.matrix_ops = get_unified_matrix_operations()
        self.vectorized_core = get_vectorized_processing_core()
        self.enhanced_matrix_ops = get_enhanced_matrix_operations()
        
        # Initialize M1 optimizations if available
        self.m1_integration = integrate_with_m1_optimizers()
        self.gpu_manager = get_gpu_manager() if self.m1_integration.get('success', False) else None
        self.memory_optimizer = get_m1_memory_optimizer() if self.m1_integration.get('success', False) else None
        
        # Initialize ML common utilities if available
        self.bayesian_optimizer = BayesianTPEOptimizer() if ML_COMMON_AVAILABLE else None
        self.cross_validator = CrossValidator() if ML_COMMON_AVAILABLE else None
        self.lookahead_detector = LookaheadBiasDetector() if ML_COMMON_AVAILABLE else None
        self.pareto_optimizer = ParetoOptimizer() if ML_COMMON_AVAILABLE else None
        
        # Initialize serialization utilities
        self.serializer = UniversalSerializer()
        
        # Initialize data quality tools
        self.math_validator = MathValidation()
        
        tprint_info("🔍 Advanced Filters for 15m Timeframe initialized")
        tprint_info(f"   → Efficiency ratio: {self.config.enable_efficiency_ratio}")
        tprint_info(f"   → CLV filtering: {self.config.enable_clv}")
        tprint_info(f"   → ATR ratio: {self.config.enable_atr_ratio}")
        tprint_info(f"   → Trend coherence: {self.config.enable_trend_coherence}")
        tprint_info(f"   → Matrix operations: {self.matrix_ops.__class__.__name__}")
        tprint_info(f"   → M1 optimizations: GPU={'✅' if self.gpu_manager else '❌'}, Memory={'✅' if self.memory_optimizer else '❌'}")
        tprint_info(f"   → ML Common: {'✅' if ML_COMMON_AVAILABLE else '❌'}")
    
    def apply_filters(self, data: pd.DataFrame) -> FilterResult:
        """
        Apply advanced filters to 15m timeframe data.
        
        Args:
            data: OHLCV data with 15m timeframe
            
        Returns:
            FilterResult with eligibility mask and statistics
        """
        start_time = datetime.now()
        tprint_info("🔍 Applying advanced 15m filters")
        
        # Validate input data using common utilities
        self._validate_input_data(data)
        
        # Optimize data using matrix operations
        tprint_info("🧮 Optimizing data with matrix operations")
        original_shape = data.shape
        optimized_data = optimize_dataframe(data)
        if optimized_data is not data:
            data = optimized_data
            tprint_success(f"✅ Data optimized: {original_shape} → {data.shape}")
        
        # Initialize result container
        result = FilterResult(
            eligibility_mask=pd.Series(True, index=data.index),
            eligibility_ratio=1.0,
            n_total_samples=len(data),
            config_used=self.config
        )
        
        try:
            # Use memory optimization context
            with memory_checkpoint("advanced_filters_15m"):
                if self.config.use_grading_system:
                    # Use grading system instead of cumulative filters
                    tprint_info("📊 Using grading system for filter evaluation")
                    grades = {}
                    
                    if self.config.enable_efficiency_ratio:
                        tprint_info("📊 Calculating efficiency ratio grade")
                        grades['efficiency'] = self._calculate_efficiency_grade(data)
                    
                    if self.config.enable_clv:
                        tprint_info("📊 Calculating CLV grade")
                        grades['clv'] = self._calculate_clv_grade(data)
                    
                    if self.config.enable_atr_ratio:
                        tprint_info("📊 Calculating ATR ratio grade")
                        grades['atr_ratio'] = self._calculate_atr_ratio_grade(data)
                    
                    if self.config.enable_trend_coherence:
                        tprint_info("📊 Calculating trend coherence grade")
                        grades['trend_coherence'] = self._calculate_trend_coherence_grade(data)
                    
                    # Calculate weighted average grade
                    if grades:
                        weights = self.config.grade_weights
                        weighted_grades = []
                        for filter_name, grade in grades.items():
                            weight = weights.get(filter_name, 0.0)
                            weighted_grades.append(grade * weight)
                        
                        average_grade = pd.concat(weighted_grades, axis=1).sum(axis=1)
                        result.eligibility_mask = average_grade >= self.config.grade_threshold
                        result.average_grade = average_grade
                        result.individual_grades = grades
                    else:
                        result.eligibility_mask = pd.Series(True, index=data.index)
                        result.average_grade = pd.Series(1.0, index=data.index)
                        result.individual_grades = {}
                else:
                    # Use legacy cumulative filtering
                    tprint_info("📊 Using legacy cumulative filtering")
                    # Apply individual filters
                    if self.config.enable_efficiency_ratio:
                        tprint_info("📊 Applying efficiency ratio filter")
                        result.efficiency_mask = self._apply_efficiency_ratio_filter(data)
                        result.filter_statistics['efficiency'] = self._calculate_efficiency_stats(data, result.efficiency_mask)
                    
                    if self.config.enable_clv:
                        tprint_info("📊 Applying CLV filter")
                        result.clv_mask = self._apply_clv_filter(data)
                        result.filter_statistics['clv'] = self._calculate_clv_stats(data, result.clv_mask)
                    
                    if self.config.enable_atr_ratio:
                        tprint_info("📊 Applying ATR ratio filter")
                        result.atr_ratio_mask = self._apply_atr_ratio_filter(data)
                        result.filter_statistics['atr_ratio'] = self._calculate_atr_ratio_stats(data, result.atr_ratio_mask)
                    
                    if self.config.enable_trend_coherence:
                        tprint_info("📊 Applying trend coherence filter")
                        result.trend_coherence_mask = self._apply_trend_coherence_filter(data)
                        result.filter_statistics['trend_coherence'] = self._calculate_trend_coherence_stats(data, result.trend_coherence_mask)
                    
                    # Combine filters based on configuration
                    result.eligibility_mask = self._combine_filters(result)
                
                result.eligibility_ratio = result.eligibility_mask.mean()
                result.n_eligible_samples = result.eligibility_mask.sum()
                result.n_filtered_samples = result.n_total_samples - result.n_eligible_samples
                
                # Calculate quality metrics
                result.overall_quality_score = self._calculate_overall_quality_score(result)
                result.noise_reduction_ratio = result.n_filtered_samples / result.n_total_samples
                
                # Calculate filter effectiveness
                result.filter_effectiveness = self._calculate_filter_effectiveness(result)
                
                # Validate results
                self._validate_filter_results(result)
                
                # Log memory usage and performance
                memory_info = optimize_memory()
                data_info = get_dataframe_info(data)
                hardware_report = get_hardware_performance_report()
                tprint_info(f"📊 Data info: {data_info['shape']} shape, {data_info.get('memory_usage', 'N/A')} memory")
                tprint_info(f"🔧 Hardware performance: {hardware_report.get('cpu_cores', 'N/A')} cores, GPU: {hardware_report.get('gpu_available', 'N/A')}")
                
                result.processing_time = (datetime.now() - start_time).total_seconds()
                
                tprint_success(f"✅ Advanced filters applied: {result.n_eligible_samples}/{result.n_total_samples} samples eligible ({result.eligibility_ratio:.1%})")
                
                return result
            
        except Exception as e:
            tprint_error(f"❌ Error applying advanced filters: {e}")
            raise
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data format and requirements using common utilities."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Use common utilities for validation
        if not validate_dataframe_columns(data, required_columns):
            missing_columns = set(required_columns) - set(data.columns)
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        min_required = max(self.config.efficiency_window, self.config.clv_window, 
                          self.config.atr_long_window, self.config.direction_window)
        if len(data) < min_required:
            raise ValueError(f"Insufficient data: need at least {min_required} samples")
        
        # Check for valid OHLCV data using safe operations
        for col in ['open', 'high', 'low', 'close']:
            if not pd.api.types.is_numeric_dtype(data[col]):
                tprint_warning(f"⚠️ Converting {col} to numeric")
                data = safe_dataframe_operation(data, pd.to_numeric, col, errors='coerce')
        
        # Validate OHLC relationships using math validation
        try:
            high_low_valid = (data['high'] >= data['low']).all()
            high_open_valid = (data['high'] >= data['open']).all()
            high_close_valid = (data['high'] >= data['close']).all()
            low_open_valid = (data['low'] <= data['open']).all()
            low_close_valid = (data['low'] <= data['close']).all()
            
            if not all([high_low_valid, high_open_valid, high_close_valid, low_open_valid, low_close_valid]):
                tprint_warning("⚠️ Found invalid OHLC relationships - data may need cleaning")
        except Exception as e:
            tprint_warning(f"⚠️ Error validating OHLC relationships: {e}")
        
        # Analyze data quality
        data_quality = create_data_quality_report(data)
        if data_quality.get('quality_metrics', {}).get('missing_percentage', 0) > 10:
            tprint_warning(f"⚠️ High missing data percentage: {data_quality['quality_metrics']['missing_percentage']:.2f}%")
        
        # Detailed NaN analysis if issues found
        nan_analysis = analyze_nan_values_detailed(data)
        if nan_analysis.get('total_nans', 0) > 0:
            nan_report = format_nan_analysis_report(nan_analysis, "  ")
            tprint_info(f"📊 NaN Analysis:\n{nan_report}")
    
    def _apply_efficiency_ratio_filter(self, data: pd.DataFrame) -> pd.Series:
        """
        Apply bar efficiency ratio filter using matrix operations.
        
        Efficiency_t = |close_t - open_t| / (high_t - low_t)
        High efficiency (>0.6) = directional, Low efficiency (<0.3) = choppy
        """
        # Calculate efficiency ratio for each bar using safe operations
        price_range = data['high'] - data['low']
        price_range = price_range.replace(0, np.nan)  # Avoid division by zero
        
        # Use safe division for efficiency calculation
        efficiency = np.abs(data['close'] - data['open'])
        efficiency = efficiency / price_range
        efficiency = efficiency.fillna(0)  # Set to 0 for zero-range bars
        efficiency = efficiency.replace([np.inf, -np.inf], 0)  # Replace infinite values with 0
        
        # Use vectorized rolling operations
        rolling_efficiency = vectorized_rolling_features(
            efficiency.values, 
            windows=self.config.efficiency_window, 
            operation='mean'
        )
        rolling_efficiency = pd.Series(rolling_efficiency, index=data.index)
        
        # Create eligibility mask using math validation
        efficiency_mask = self.math_validator.validate_finite(
            (rolling_efficiency >= self.config.efficiency_threshold_low) & 
            (rolling_efficiency <= 1.0)  # Cap at 1.0 (perfect efficiency)
        )
        
        tprint_info(f"   → Efficiency filter: {efficiency_mask.sum()}/{len(efficiency_mask)} samples passed")
        
        return efficiency_mask
    
    def _calculate_efficiency_grade(self, data: pd.DataFrame) -> pd.Series:
        """Calculate efficiency ratio grade (0.0-1.0) for grading system."""
        # Calculate efficiency ratio for each bar using safe operations
        price_range = data['high'] - data['low']
        price_range = price_range.replace(0, np.nan)  # Avoid division by zero
        
        # Use safe division for efficiency calculation
        efficiency = np.abs(data['close'] - data['open'])
        efficiency = efficiency / price_range
        efficiency = efficiency.fillna(0)  # Set to 0 for zero-range bars
        efficiency = efficiency.replace([np.inf, -np.inf], 0)  # Replace infinite values with 0
        
        # Use vectorized rolling operations
        rolling_efficiency = vectorized_rolling_features(
            efficiency.values, 
            windows=self.config.efficiency_window, 
            operation='mean'
        )
        rolling_efficiency = pd.Series(rolling_efficiency, index=data.index)
        
        # Convert to grade (0.0-1.0) where higher efficiency = higher grade
        # Normalize efficiency to 0-1 range, with 0.6+ efficiency = 1.0 grade
        efficiency_grade = np.clip(rolling_efficiency / 0.6, 0.0, 1.0)
        
        tprint_info(f"   → Efficiency grade: mean={efficiency_grade.mean():.3f}, std={efficiency_grade.std():.3f}")
        
        return efficiency_grade
    
    def _apply_clv_filter(self, data: pd.DataFrame) -> pd.Series:
        """
        Apply Close-Location Value (CLV) filter using matrix operations.
        
        CLV_t = (2*close_t - high_t - low_t) / (high_t - low_t)
        Sustained positive CLV → bullish control, sustained negative → bearish
        """
        # Calculate CLV for each bar using safe operations
        price_range = data['high'] - data['low']
        price_range = price_range.replace(0, np.nan)  # Avoid division by zero
        
        clv_numerator = 2 * data['close'] - data['high'] - data['low']
        clv = clv_numerator / price_range
        clv = clv.fillna(0)  # Set to 0 for zero-range bars
        clv = clv.replace([np.inf, -np.inf], 0)  # Replace infinite values with 0
        
        # Use vectorized rolling operations for mean and std
        rolling_clv = vectorized_rolling_features(
            clv.values, 
            windows=self.config.clv_window, 
            operation='mean'
        )
        rolling_clv = pd.Series(rolling_clv, index=data.index)
        
        clv_volatility = vectorized_rolling_features(
            clv.values, 
            windows=self.config.clv_window, 
            operation='std'
        )
        clv_volatility = pd.Series(clv_volatility, index=data.index)
        
        # Create eligibility mask using math validation
        clv_directional = self.math_validator.validate_finite(
            (rolling_clv >= self.config.clv_threshold_positive) | 
            (rolling_clv <= self.config.clv_threshold_negative)
        )
        clv_stable = self.math_validator.validate_finite(
            clv_volatility <= self.config.clv_volatility_threshold
        )
        
        clv_mask = clv_directional & clv_stable
        
        tprint_info(f"   → CLV filter: {clv_mask.sum()}/{len(clv_mask)} samples passed")
        
        return clv_mask
    
    def _calculate_clv_grade(self, data: pd.DataFrame) -> pd.Series:
        """Calculate CLV grade (0.0-1.0) for grading system."""
        # Calculate CLV for each bar using safe operations
        price_range = data['high'] - data['low']
        price_range = price_range.replace(0, np.nan)  # Avoid division by zero
        
        clv_numerator = 2 * data['close'] - data['high'] - data['low']
        clv = clv_numerator / price_range
        clv = clv.fillna(0)  # Set to 0 for zero-range bars
        clv = clv.replace([np.inf, -np.inf], 0)  # Replace infinite values with 0
        
        # Use vectorized rolling operations for mean and std
        rolling_clv = vectorized_rolling_features(
            clv.values, 
            windows=self.config.clv_window, 
            operation='mean'
        )
        rolling_clv = pd.Series(rolling_clv, index=data.index)
        
        clv_volatility = vectorized_rolling_features(
            clv.values, 
            windows=self.config.clv_window, 
            operation='std'
        )
        clv_volatility = pd.Series(clv_volatility, index=data.index)
        
        # Convert to grade (0.0-1.0) based on directional strength and stability
        # Higher grade for stronger directional CLV and lower volatility
        clv_strength = np.abs(rolling_clv)
        clv_stability = 1.0 - np.clip(clv_volatility / self.config.clv_volatility_threshold, 0.0, 1.0)
        clv_grade = (clv_strength * clv_stability).clip(0.0, 1.0)
        
        tprint_info(f"   → CLV grade: mean={clv_grade.mean():.3f}, std={clv_grade.std():.3f}")
        
        return clv_grade
    
    def _apply_atr_ratio_filter(self, data: pd.DataFrame) -> pd.Series:
        """
        Apply ATR volatility ratio filter using matrix operations.
        
        r_t = ATR_short / ATR_long
        Skip when r_t > 1.5-2.0 (too jumpy) or < 0.5 (too quiet)
        """
        # Calculate True Range using vectorized operations
        tr1 = data['high'] - data['low']
        tr2 = np.abs(data['high'] - data['close'].shift(1))
        tr3 = np.abs(data['low'] - data['close'].shift(1))
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Use vectorized rolling operations for ATR calculation
        atr_short = vectorized_rolling_features(
            true_range.values, 
            windows=self.config.atr_short_window, 
            operation='mean'
        )
        atr_short = pd.Series(atr_short, index=data.index)
        
        atr_long = vectorized_rolling_features(
            true_range.values, 
            windows=self.config.atr_long_window, 
            operation='mean'
        )
        atr_long = pd.Series(atr_long, index=data.index)
        
        # Calculate ATR ratio using safe division
        atr_ratio = atr_short / atr_long
        atr_ratio = atr_ratio.fillna(1.0)  # Fill NaN values with 1.0
        atr_ratio = atr_ratio.replace([np.inf, -np.inf], 1.0)  # Replace infinite values with 1.0
        
        # Create eligibility mask using math validation (only check upper bound - no "too quiet" filter)
        atr_ratio_mask = self.math_validator.validate_finite(
            atr_ratio <= self.config.atr_ratio_threshold_high
        )
        
        tprint_info(f"   → ATR ratio filter: {atr_ratio_mask.sum()}/{len(atr_ratio_mask)} samples passed")
        
        return atr_ratio_mask
    
    def _calculate_atr_ratio_grade(self, data: pd.DataFrame) -> pd.Series:
        """Calculate ATR ratio grade (0.0-1.0) for grading system."""
        # Calculate True Range using vectorized operations
        tr1 = data['high'] - data['low']
        tr2 = np.abs(data['high'] - data['close'].shift(1))
        tr3 = np.abs(data['low'] - data['close'].shift(1))
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Use vectorized rolling operations for ATR calculation
        atr_short = vectorized_rolling_features(
            true_range.values, 
            windows=self.config.atr_short_window, 
            operation='mean'
        )
        atr_short = pd.Series(atr_short, index=data.index)
        
        atr_long = vectorized_rolling_features(
            true_range.values, 
            windows=self.config.atr_long_window, 
            operation='mean'
        )
        atr_long = pd.Series(atr_long, index=data.index)
        
        # Calculate ATR ratio using safe division
        atr_ratio = atr_short / atr_long
        atr_ratio = atr_ratio.fillna(1.0)  # Fill NaN values with 1.0
        atr_ratio = atr_ratio.replace([np.inf, -np.inf], 1.0)  # Replace infinite values with 1.0
        
        # Convert to grade (0.0-1.0) where moderate volatility = higher grade
        # Grade decreases as ratio approaches the threshold (too jumpy)
        atr_grade = np.clip(1.0 - (atr_ratio / self.config.atr_ratio_threshold_high), 0.0, 1.0)
        
        tprint_info(f"   → ATR ratio grade: mean={atr_grade.mean():.3f}, std={atr_grade.std():.3f}")
        
        return atr_grade
    
    def _apply_trend_coherence_filter(self, data: pd.DataFrame) -> pd.Series:
        """
        Apply trend coherence filter using matrix operations.
        
        Combines direction consistency and EMA slope for trend continuity.
        """
        # Calculate direction consistency using vectorized operations
        close_direction = np.sign(data['close'].diff())
        
        # Use vectorized rolling operations for direction consistency
        direction_consistency = vectorized_rolling_features(
            close_direction.values, 
            windows=self.config.direction_window, 
            operation='consistency'
        )
        direction_consistency = pd.Series(direction_consistency, index=data.index)
        
        # Calculate EMA slope using vectorized operations
        ema = data['close'].ewm(span=self.config.ema_period, min_periods=1).mean()
        ema_slope = ema.diff()
        
        # Create eligibility mask using math validation
        direction_consistent = self.math_validator.validate_finite(
            direction_consistency >= self.config.min_direction_consistency
        )
        slope_positive = self.math_validator.validate_finite(
            ema_slope >= self.config.min_slope_threshold
        )
        
        trend_coherence_mask = direction_consistent & slope_positive
        
        tprint_info(f"   → Trend coherence filter: {trend_coherence_mask.sum()}/{len(trend_coherence_mask)} samples passed")
        
        return trend_coherence_mask
    
    def _calculate_trend_coherence_grade(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trend coherence grade (0.0-1.0) for grading system."""
        # Calculate direction consistency using vectorized operations
        close_direction = np.sign(data['close'].diff())
        
        # Use vectorized rolling operations for direction consistency
        direction_consistency = vectorized_rolling_features(
            close_direction.values, 
            windows=self.config.direction_window, 
            operation='consistency'
        )
        direction_consistency = pd.Series(direction_consistency, index=data.index)
        
        # Calculate EMA slope using vectorized operations
        ema = data['close'].ewm(span=self.config.ema_period, min_periods=1).mean()
        ema_slope = ema.diff()
        
        # Convert to grade (0.0-1.0) based on direction consistency and slope strength
        direction_grade = np.clip(direction_consistency, 0.0, 1.0)
        slope_grade = np.clip(ema_slope / self.config.min_slope_threshold, 0.0, 1.0)
        trend_coherence_grade = (direction_grade * slope_grade).clip(0.0, 1.0)
        
        tprint_info(f"   → Trend coherence grade: mean={trend_coherence_grade.mean():.3f}, std={trend_coherence_grade.std():.3f}")
        
        return trend_coherence_grade
    
    def cleanup(self) -> None:
        """Clean up resources and optimize memory."""
        try:
            # Optimize memory usage
            memory_info = optimize_memory()
            if memory_info.get('success', False):
                tprint_info(f"🧠 Memory optimized: {memory_info.get('objects_collected', 0)} objects collected")
            
            # Clean up M1 optimizers if available
            if self.memory_optimizer:
                self.memory_optimizer.cleanup()
            
            tprint_success("✅ AdvancedFilters15m cleanup completed")
        except Exception as e:
            tprint_warning(f"⚠️ Error during cleanup: {e}")
    
    def _combine_filters(self, result: FilterResult) -> pd.Series:
        """Combine individual filter results into final eligibility mask."""
        masks = []
        
        if result.efficiency_mask is not None:
            masks.append(result.efficiency_mask)
        if result.clv_mask is not None:
            masks.append(result.clv_mask)
        if result.atr_ratio_mask is not None:
            masks.append(result.atr_ratio_mask)
        if result.trend_coherence_mask is not None:
            masks.append(result.trend_coherence_mask)
        
        if not masks:
            # If no filters are enabled, return all True
            return pd.Series(True, index=result.eligibility_mask.index)
        
        # Combine masks based on filter type
        if self.config.filter_type == FilterType.COMBINED:
            # Use AND logic - all filters must pass
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = combined_mask & mask
        else:
            # Use OR logic - any filter can pass
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = combined_mask | mask
        
        return combined_mask
    
    def _calculate_efficiency_stats(self, data: pd.DataFrame, mask: pd.Series) -> Dict[str, float]:
        """Calculate efficiency ratio statistics using safe operations."""
        price_range = data['high'] - data['low']
        price_range = price_range.replace(0, np.nan)
        efficiency = np.abs(data['close'] - data['open'])
        efficiency = efficiency / price_range
        efficiency = efficiency.fillna(0)
        efficiency = efficiency.replace([np.inf, -np.inf], 0)
        
        return {
            'mean_efficiency': float(safe_mean(efficiency)),
            'std_efficiency': float(safe_std(efficiency)),
            'min_efficiency': float(efficiency.min()),
            'max_efficiency': float(efficiency.max()),
            'eligible_ratio': float(mask.mean()) if mask is not None else 0.0
        }
    
    def _calculate_clv_stats(self, data: pd.DataFrame, mask: pd.Series) -> Dict[str, float]:
        """Calculate CLV statistics using safe operations."""
        price_range = data['high'] - data['low']
        price_range = price_range.replace(0, np.nan)
        clv_numerator = 2 * data['close'] - data['high'] - data['low']
        clv = clv_numerator / price_range
        clv = clv.fillna(0)
        clv = clv.replace([np.inf, -np.inf], 0)
        
        return {
            'mean_clv': float(safe_mean(clv)),
            'std_clv': float(safe_std(clv)),
            'min_clv': float(clv.min()),
            'max_clv': float(clv.max()),
            'eligible_ratio': float(mask.mean()) if mask is not None else 0.0
        }
    
    def _calculate_atr_ratio_stats(self, data: pd.DataFrame, mask: pd.Series) -> Dict[str, float]:
        """Calculate ATR ratio statistics using safe operations."""
        tr1 = data['high'] - data['low']
        tr2 = np.abs(data['high'] - data['close'].shift(1))
        tr3 = np.abs(data['low'] - data['close'].shift(1))
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        atr_short = true_range.rolling(window=self.config.atr_short_window, min_periods=1).mean()
        atr_long = true_range.rolling(window=self.config.atr_long_window, min_periods=1).mean()
        atr_ratio = atr_short / atr_long
        atr_ratio = atr_ratio.fillna(1.0)  # Fill NaN values with 1.0
        atr_ratio = atr_ratio.replace([np.inf, -np.inf], 1.0)  # Replace infinite values with 1.0
        
        return {
            'mean_atr_ratio': float(safe_mean(atr_ratio)),
            'std_atr_ratio': float(safe_std(atr_ratio)),
            'min_atr_ratio': float(atr_ratio.min()),
            'max_atr_ratio': float(atr_ratio.max()),
            'eligible_ratio': float(mask.mean()) if mask is not None else 0.0
        }
    
    def _calculate_trend_coherence_stats(self, data: pd.DataFrame, mask: pd.Series) -> Dict[str, float]:
        """Calculate trend coherence statistics using safe operations."""
        close_direction = np.sign(data['close'].diff())
        direction_consistency = close_direction.rolling(window=self.config.direction_window, min_periods=1).apply(
            lambda x: (x == x.iloc[-1]).sum() / len(x) if len(x) > 0 else 0
        )
        
        ema = data['close'].ewm(span=self.config.ema_period, min_periods=1).mean()
        ema_slope = ema.diff()
        
        return {
            'mean_direction_consistency': float(safe_mean(direction_consistency)),
            'mean_ema_slope': float(safe_mean(ema_slope)),
            'std_ema_slope': float(safe_std(ema_slope)),
            'eligible_ratio': float(mask.mean()) if mask is not None else 0.0
        }
    
    def _calculate_overall_quality_score(self, result: FilterResult) -> float:
        """Calculate overall quality score based on filter results."""
        if result.n_total_samples == 0:
            return 0.0
        
        # Base score from eligibility ratio
        base_score = result.eligibility_ratio
        
        # Bonus for noise reduction
        noise_reduction_bonus = min(result.noise_reduction_ratio * 0.2, 0.2)
        
        # Penalty for too many filters failing
        if result.eligibility_ratio < self.config.min_eligibility_ratio:
            penalty = (self.config.min_eligibility_ratio - result.eligibility_ratio) * 0.5
        else:
            penalty = 0.0
        
        quality_score = base_score + noise_reduction_bonus - penalty
        return max(0.0, min(1.0, quality_score))
    
    def _calculate_filter_effectiveness(self, result: FilterResult) -> Dict[str, float]:
        """Calculate effectiveness of each filter."""
        effectiveness = {}
        
        for filter_name, stats in result.filter_statistics.items():
            if 'eligible_ratio' in stats:
                effectiveness[filter_name] = stats['eligible_ratio']
        
        return effectiveness
    
    def _validate_filter_results(self, result: FilterResult) -> None:
        """Validate filter results and raise warnings if needed."""
        if result.eligibility_ratio < self.config.min_eligibility_ratio:
            tprint_warning(f"⚠️ Low eligibility ratio: {result.eligibility_ratio:.1%} < {self.config.min_eligibility_ratio:.1%}")
        
        if result.n_eligible_samples < self.config.min_eligible_samples:
            tprint_warning(f"⚠️ Insufficient eligible samples: {result.n_eligible_samples} < {self.config.min_eligible_samples}")
        
        # Check for filter failure rates
        for filter_name, effectiveness in result.filter_effectiveness.items():
            failure_rate = 1.0 - effectiveness
            if failure_rate > self.config.max_filter_failure_rate:
                tprint_warning(f"⚠️ High failure rate for {filter_name}: {failure_rate:.1%} > {self.config.max_filter_failure_rate:.1%}")
        
        # Log data quality metrics
        if hasattr(result, 'config_used') and result.config_used:
            tprint_info(f"📊 Filter configuration: {result.config_used.filter_type.value} mode")
            tprint_info(f"📊 Quality score: {result.overall_quality_score:.3f}")
            tprint_info(f"📊 Noise reduction: {result.noise_reduction_ratio:.1%}")


# Convenience function for external usage
def apply_advanced_filters_15m(
    data: pd.DataFrame,
    config: Optional[AdvancedFiltersConfig] = None,
    **kwargs
) -> FilterResult:
    """
    Apply advanced filters to 15m timeframe data.
    
    Args:
        data: OHLCV data with 15m timeframe
        config: Optional configuration
        **kwargs: Additional parameters
        
    Returns:
        FilterResult with eligibility mask and statistics
    """
    tprint_info("🚀 Starting advanced filters 15m application")
    
    try:
        filter_system = AdvancedFilters15m(config)
        result = filter_system.apply_filters(data, **kwargs)
        
        # Cleanup resources
        filter_system.cleanup()
        
        tprint_success("✅ Advanced filters 15m application completed")
        return result
        
    except Exception as e:
        tprint_error(f"❌ Error in advanced filters 15m application: {e}")
        raise

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
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
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
