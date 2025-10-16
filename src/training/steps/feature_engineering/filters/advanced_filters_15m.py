"""
Advanced Filters 15m - Unified Filter System

This module provides a unified interface for applying advanced filters to 15-minute
timeframe data using a grading system instead of cumulative filtering.

Features:
1. Bar Efficiency Ratio - Measures directional price action vs. choppy conditions
2. Close-Location Value (CLV) - Tracks buying/selling pressure and control
3. ATR Volatility Ratio - Normalizes volatility for adaptive filtering
4. Trend Coherence - Ensures trend continuity and direction consistency

Uses a grading system with weighted average and single threshold for filtering.
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

# Import feature engineering modules
from src.training.steps.feature_engineering.price_action.bar_efficiency_ratio import (
    BarEfficiencyRatioFeature, BarEfficiencyConfig
)
from src.training.steps.feature_engineering.price_action.close_location_value import (
    CloseLocationValueFeature, CLVConfig
)
from src.training.steps.feature_engineering.volatility.atr_volatility_ratio import (
    ATRVolatilityRatioFeature, ATRVolatilityRatioConfig
)
from src.training.steps.feature_engineering.trend.trend_coherence import (
    TrendCoherenceFeature, TrendCoherenceConfig
)

# Import ML common utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
    from src.utils.ml_common.validation import CrossValidator
    from src.utils.ml_common.utils.lookahead_protection import LookaheadBiasDetector
    from src.utils.ml_common.optimization.pareto import ParetoOptimizer
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

    # Individual filter enable/disable
    enable_efficiency_ratio: bool = True
    enable_clv: bool = True
    enable_atr_ratio: bool = True
    enable_trend_coherence: bool = True

    # Grading system (primary filtering method)
    use_grading_system: bool = True  # Use average grade instead of cumulative filters
    grade_threshold: float = 0.2  # Minimum average grade to pass (0.0-1.0)
    grade_weights: Dict[str, float] = field(default_factory=lambda: {
        'efficiency': 0.25,
        'clv': 0.25,
        'atr_ratio': 0.25,
        'trend_coherence': 0.25
    })

    # Individual filter configurations
    efficiency_config: Optional[BarEfficiencyConfig] = None
    clv_config: Optional[CLVConfig] = None
    atr_config: Optional[ATRVolatilityRatioConfig] = None
    trend_config: Optional[TrendCoherenceConfig] = None

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

    # Grading system results
    average_grade: Optional[pd.Series] = None
    individual_grades: Dict[str, pd.Series] = field(default_factory=dict)

    # Legacy filter results
    efficiency_mask: Optional[pd.Series] = None
    clv_mask: Optional[pd.Series] = None
    atr_ratio_mask: Optional[pd.Series] = None
    trend_coherence_mask: Optional[pd.Series] = None

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
    Advanced Filters for 15m Timeframe - Unified Filter System

    This class provides a unified interface for applying advanced filters to 15-minute
    timeframe data using a grading system instead of cumulative filtering.
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

        # Initialize individual feature engines
        self.efficiency_engine = BarEfficiencyRatioFeature(self.config.efficiency_config)
        self.clv_engine = CloseLocationValueFeature(self.config.clv_config)
        self.atr_engine = ATRVolatilityRatioFeature(self.config.atr_config)
        self.trend_engine = TrendCoherenceFeature(self.config.trend_config)

        tprint_info("🔍 Advanced Filters for 15m Timeframe initialized")
        tprint_info(f"   → Efficiency ratio: {self.config.enable_efficiency_ratio}")
        tprint_info(f"   → CLV filtering: {self.config.enable_clv}")
        tprint_info(f"   → ATR ratio: {self.config.enable_atr_ratio}")
        tprint_info(f"   → Trend coherence: {self.config.enable_trend_coherence}")
        tprint_info(f"   → Grading system: {self.config.use_grading_system}")
        tprint_info(f"   → Grade threshold: {self.config.grade_threshold}")
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
                        efficiency_features = self.efficiency_engine.calculate_features(data)
                        if 'bar_efficiency_grade' in efficiency_features:
                            grades['efficiency'] = efficiency_features['bar_efficiency_grade']

                    if self.config.enable_clv:
                        tprint_info("📊 Calculating CLV grade")
                        clv_features = self.clv_engine.calculate_features(data)
                        if 'clv_grade' in clv_features:
                            grades['clv'] = clv_features['clv_grade']

                    if self.config.enable_atr_ratio:
                        tprint_info("📊 Calculating ATR ratio grade")
                        atr_features = self.atr_engine.calculate_features(data)
                        if 'atr_grade' in atr_features:
                            grades['atr_ratio'] = atr_features['atr_grade']

                    if self.config.enable_trend_coherence:
                        tprint_info("📊 Calculating trend coherence grade")
                        trend_features = self.trend_engine.calculate_features(data)
                        if 'trend_coherence_grade' in trend_features:
                            grades['trend_coherence'] = trend_features['trend_coherence_grade']

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

                        tprint_info(f"   → Average grade: mean={average_grade.mean():.3f}, std={average_grade.std():.3f}")
                        tprint_info(f"   → Grade threshold: {self.config.grade_threshold}")
                    else:
                        result.eligibility_mask = pd.Series(True, index=data.index)
                        result.average_grade = pd.Series(1.0, index=data.index)
                        result.individual_grades = {}

                result.eligibility_ratio = result.eligibility_mask.mean()
                result.n_eligible_samples = result.eligibility_mask.sum()
                result.n_filtered_samples = result.n_total_samples - result.n_eligible_samples

                # Calculate quality metrics
                result.overall_quality_score = self._calculate_overall_quality_score(result)
                result.noise_reduction_ratio = result.n_filtered_samples / result.n_total_samples

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

        min_required = max(20, 50)  # Minimum samples for reliable filtering
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

    def _calculate_overall_quality_score(self, result: FilterResult) -> float:
        """Calculate overall quality score based on filter results."""
        if result.n_total_samples == 0:
            return 0.0

        # Base score from eligibility ratio
        eligibility_score = result.eligibility_ratio

        # Bonus for good noise reduction (but not too much)
        noise_reduction_score = min(result.noise_reduction_ratio, 0.8)  # Cap at 80% reduction

        # Combine scores
        overall_score = (eligibility_score * 0.7) + (noise_reduction_score * 0.3)

        return min(overall_score, 1.0)

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
