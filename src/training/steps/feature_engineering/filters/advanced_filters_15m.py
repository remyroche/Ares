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
from src.utils.hardware import (
    get_integrated_hardware_manager, 
    get_comprehensive_optimizer,
    memory_optimized, 
    comprehensive_memory_optimization,
    optimize_dataframe, 
    optimize_array,
    m1_optimized,
    WorkloadCategory,
    MemoryOptimizationLevel
)
    get_unified_matrix_operations, get_vectorized_processing_core,
    get_enhanced_matrix_operations, optimize_dataframe,
    vectorized_rolling_features, matrix_correlation_analysis,
    safe_correlation_matrix, compute_trading_indicators,
    get_hardware_performance_report
)
 as get_gpu_manager
()
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

        # Validate OHLC relationships using math validation (lenient)
        try:
            high_low_valid = (data['high'] >= data['low']).all()
            high_open_valid = (data['high'] >= data['open']).all()
            high_close_valid = (data['high'] >= data['close']).all()
            low_open_valid = (data['low'] <= data['open']).all()
            low_close_valid = (data['low'] <= data['close']).all()

            if not all([high_low_valid, high_open_valid, high_close_valid, low_open_valid, low_close_valid]):
                tprint_warning("⚠️ Found invalid OHLC relationships - data may need cleaning")
                # Don't fail the pipeline, just warn and continue
                tprint_info("ℹ️ Continuing with data cleaning to fix OHLC relationships...")
                # Basic data cleaning
                data['high'] = data[['high', 'open', 'close']].max(axis=1)
                data['low'] = data[['low', 'open', 'close']].min(axis=1)
                tprint_success("✅ OHLC relationships corrected")
        except Exception as e:
            tprint_warning(f"⚠️ Error validating OHLC relationships: {e}")
            tprint_info("ℹ️ Continuing with data processing despite validation errors...")

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
            memory_info = get_integrated_hardware_manager().clear_all_caches()
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
