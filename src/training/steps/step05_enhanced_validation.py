"""
Step05 Enhanced Validation Module

This module provides enhanced validation capabilities with sophisticated bias detection,
statistical validation, and comprehensive logging for Step05 labeling operations.
"""

import pandas as pd
import numpy as np
import time
import scipy.stats as stats
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, validates
from src.utils.common_operations import safe_mean, safe_std, safe_float, safe_int, validate_dataframe_schema, validate_data_quality, safe_copy, safe_deepcopy, get_current_datetime, format_datetime, create_empty_dataframe, safe_fillna, safe_rolling, safe_append, safe_extend, safe_dict_get, safe_dict_items, safe_lower, safe_upper, safe_join, get_logger, setup_basic_logging, safe_exception_handler, timed_operation, format_bytes, chunked_iterable, parallel_map, safe_log_metric, safe_log_params, safe_log_artifact
from src.utils.math_validation import safe_divide, safe_log, safe_sqrt, safe_power, validate_finite, validate_positive, validate_range, safe_kelly_calculation, safe_weighted_average, safe_percentage_change, validate_correlation_matrix, safe_matrix_inverse, math_safe, MathValidationError
from src.utils.parquet_utils import ParquetUtils, get_parquet_utils
from src.core.errors import AppError, ValidationError, DataIntegrityError, BusinessRuleError, NotFoundError, ConflictError, RateLimitError, TimeoutError, ServiceUnavailableError, ErrorCode
import os

logger = system_logger.getChild('Step05EnhancedValidation')


@dataclass
class StatisticalValidationResult:
    """Result of statistical validation checks."""
    passed: bool
    score: float
    warnings: List[str]
    errors: List[str]
    recommendations: List[str]
    details: Dict[str, Any]
    statistical_tests: Dict[str, Any]
    computation_time: float = 0.0


@dataclass
class BiasDetectionResult:
    """Result of sophisticated bias detection."""
    bias_detected: bool
    bias_score: float
    bias_types: List[str]
    temporal_violations: int
    future_data_leakage: bool
    statistical_anomalies: List[str]
    recommendations: List[str]
    details: Dict[str, Any]
    computation_time: float = 0.0


class Step05EnhancedValidator:
    """Enhanced validator with sophisticated bias detection and statistical validation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        self.validation_history = []
        self.performance_stats = {
            'total_validations': 0,
            'bias_detections': 0,
            'statistical_tests': 0,
            'total_computation_time': 0.0,
            'avg_computation_time': 0.0
        }
        
        self.logger.info("🚀 Initializing Step05 Enhanced Validator")
        self.logger.info("🔍 Enhanced features: Sophisticated bias detection, Statistical validation")
        self.logger.info("📊 Statistical tests: Normality, Stationarity, Autocorrelation, Cointegration")
    
    @traced(span_name='validate_ohlc_comprehensive')
    @validates()
    @handles_errors()
    def validate_ohlc_comprehensive(self, data: pd.DataFrame) -> StatisticalValidationResult:
        """
        Comprehensive OHLC validation with statistical checks.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            StatisticalValidationResult with comprehensive validation
        """
        start_time = time.time()
        self.logger.info("🔍 Starting comprehensive OHLC validation...")
        
        try:
            warnings = []
            errors = []
            recommendations = []
            details = {}
            statistical_tests = {}
            
            # Fast fail validation
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
                return StatisticalValidationResult(
                    passed=False, score=0.0, warnings=warnings, errors=errors,
                    recommendations=["Ensure all OHLC data is present"], details=details,
                    statistical_tests=statistical_tests, computation_time=time.time() - start_time
                )
            
            self.logger.info(f"📊 Validating OHLC data with {len(data)} rows")
            
            # 1. Basic OHLC relationship validation
            self.logger.info("🔍 Validating basic OHLC relationships...")
            
            high_violations = (data['high'] < data[['open', 'close']].max(axis=1)).sum()
            low_violations = (data['low'] > data[['open', 'close']].min(axis=1)).sum()
            hl_violations = (data['high'] < data['low']).sum()
            
            total_ohlc_errors = high_violations + low_violations + hl_violations
            details['ohlc_errors'] = {
                'high_violations': int(high_violations),
                'low_violations': int(low_violations),
                'hl_violations': int(hl_violations),
                'total': int(total_ohlc_errors)
            }
            
            if total_ohlc_errors > 0:
                errors.append(f"Found {total_ohlc_errors} OHLC relationship errors")
                recommendations.append("Review price data for invalid OHLC relationships")
                
                # Log sample errors
                error_mask = (data['high'] < data[['open', 'close']].max(axis=1)) | \
                           (data['low'] > data[['open', 'close']].min(axis=1)) | \
                           (data['high'] < data['low'])
                error_samples = data[error_mask].head(5)
                self.logger.error("🔍 Sample OHLC errors:")
                for idx, row in error_samples.iterrows():
                    self.logger.error(f"   Row {idx}: O={row['open']:.4f}, H={row['high']:.4f}, L={row['low']:.4f}, C={row['close']:.4f}")
            
            # 2. Price movement statistical validation
            self.logger.info("📊 Performing price movement statistical analysis...")
            
            price_changes = data['close'].pct_change().dropna()
            
            # Extreme moves detection
            extreme_moves = (price_changes.abs() > 0.1).sum()
            details['extreme_moves'] = int(extreme_moves)
            
            if extreme_moves > len(data) * 0.01:
                warnings.append(f"High number of extreme price moves: {extreme_moves}")
                recommendations.append("Review data for potential outliers or data quality issues")
            
            # Statistical tests for price changes
            if len(price_changes) > 30:  # Minimum sample size for statistical tests
                # Normality test
                shapiro_stat, shapiro_p = stats.shapiro(price_changes.sample(min(5000, len(price_changes))))
                statistical_tests['normality'] = {
                    'shapiro_statistic': shapiro_stat,
                    'shapiro_p_value': shapiro_p,
                    'is_normal': shapiro_p > 0.05
                }
                
                if shapiro_p <= 0.05:
                    warnings.append("Price changes are not normally distributed")
                    recommendations.append("Consider using robust statistical methods")
                
                # Autocorrelation test
                from statsmodels.stats.diagnostic import acorr_ljungbox
                try:
                    ljungbox_result = acorr_ljungbox(price_changes, lags=10, return_df=True)
                    autocorr_p = ljungbox_result['lb_pvalue'].iloc[-1]
                    statistical_tests['autocorrelation'] = {
                        'ljungbox_p_value': autocorr_p,
                        'has_autocorrelation': autocorr_p <= 0.05
                    }
                    
                    if autocorr_p <= 0.05:
                        warnings.append("Significant autocorrelation detected in price changes")
                        recommendations.append("Consider modeling autocorrelation in trading strategy")
                
                except Exception as e:
                    self.logger.warning(f"⚠️ Autocorrelation test failed: {e}")
                    statistical_tests['autocorrelation'] = {'error': str(e)}
            
            # 3. Volume-price relationship validation
            if 'volume' in data.columns:
                self.logger.info("📊 Analyzing volume-price relationships...")
                
                volume_price_corr = data['volume'].corr(data['close'].pct_change().abs())
                details['volume_price_correlation'] = float(volume_price_corr) if not pd.isna(volume_price_corr) else 0.0
                
                if volume_price_corr < 0.3:
                    warnings.append("Low volume-price correlation detected")
                    recommendations.append("Review volume data quality")
                
                # Volume statistical analysis
                volume_changes = data['volume'].pct_change().dropna()
                volume_skewness = volume_changes.skew()
                volume_kurtosis = volume_changes.kurtosis()
                
                details['volume_statistics'] = {
                    'correlation': float(volume_price_corr),
                    'skewness': float(volume_skewness),
                    'kurtosis': float(volume_kurtosis)
                }
                
                if abs(volume_skewness) > 2:
                    warnings.append(f"High volume skewness: {volume_skewness:.2f}")
                
                if volume_kurtosis > 5:
                    warnings.append(f"High volume kurtosis: {volume_kurtosis:.2f}")
            
            # 4. Price level validation
            self.logger.info("📊 Validating price levels...")
            
            # Check for zero or negative prices
            zero_prices = (data[required_columns] <= 0).any(axis=1).sum()
            details['zero_prices'] = int(zero_prices)
            
            if zero_prices > 0:
                errors.append(f"Found {zero_prices} rows with zero or negative prices")
                recommendations.append("Remove or correct zero/negative price data")
            
            # Check for unrealistic price jumps
            price_jumps = data['close'].pct_change().abs()
            unrealistic_jumps = (price_jumps > 0.5).sum()  # >50% jumps
            details['unrealistic_jumps'] = int(unrealistic_jumps)
            
            if unrealistic_jumps > 0:
                warnings.append(f"Found {unrealistic_jumps} unrealistic price jumps (>50%)")
                recommendations.append("Review data for potential errors or corporate actions")
            
            # 5. Data consistency validation
            self.logger.info("🔍 Validating data consistency...")
            
            # Check for duplicate timestamps
            if hasattr(data.index, 'duplicated'):
                duplicate_timestamps = data.index.duplicated().sum()
                details['duplicate_timestamps'] = int(duplicate_timestamps)
                
                if duplicate_timestamps > 0:
                    errors.append(f"Found {duplicate_timestamps} duplicate timestamps")
                    recommendations.append("Remove duplicate timestamp entries")
            
            # Check for missing values
            missing_values = data[required_columns].isnull().sum().sum()
            missing_percentage = missing_values / (len(data) * len(required_columns))
            details['missing_values'] = {
                'count': int(missing_values),
                'percentage': float(missing_percentage)
            }
            
            if missing_percentage > 0.05:  # >5% missing
                warnings.append(f"High percentage of missing values: {missing_percentage:.1%}")
                recommendations.append("Review data collection process for missing values")
            
            # Calculate overall score using safe math operations
            score = 1.0
            score = safe_divide(score - len(errors) * 0.2, 1.0, score)
            score = safe_divide(score - len(warnings) * 0.1, 1.0, score)
            score = safe_divide(score - min(total_ohlc_errors * 0.01, 0.3), 1.0, score)
            score = safe_divide(score - min(extreme_moves * 0.001, 0.2), 1.0, score)
            score = max(score, 0.0)
            
            passed = len(errors) == 0 and score > 0.7
            
            computation_time = time.time() - start_time
            
            # Update performance stats
            self.performance_stats['total_validations'] += 1
            self.performance_stats['statistical_tests'] += 1
            self.performance_stats['total_computation_time'] += computation_time
            self.performance_stats['avg_computation_time'] = (
                self.performance_stats['total_computation_time'] / 
                self.performance_stats['total_validations']
            )
            
            result = StatisticalValidationResult(
                passed=passed,
                score=score,
                warnings=warnings,
                errors=errors,
                recommendations=recommendations,
                details=details,
                statistical_tests=statistical_tests,
                computation_time=computation_time
            )
            
            self.logger.info(f"✅ Comprehensive OHLC validation completed in {computation_time:.3f}s")
            self.logger.info(f"📊 Validation score: {score:.3f}, Passed: {passed}")
            self.logger.info(f"⚠️ Errors: {len(errors)}, Warnings: {len(warnings)}")
            
            return result
            
        except Exception as e:
            computation_time = time.time() - start_time
            self.logger.error(f"❌ Comprehensive OHLC validation failed after {computation_time:.3f}s: {e}")
            self.logger.error(f"🔍 Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.error(f"📋 Stack trace: {traceback.format_exc()}")
            
            return StatisticalValidationResult(
                passed=False, score=0.0, warnings=[], errors=[f"Validation failed: {str(e)}"],
                recommendations=["Fix validation errors before proceeding"], details={'error': str(e)},
                statistical_tests={}, computation_time=computation_time
            )
    
    @traced(span_name='validate_temporal_consistency_enhanced')
    @validates()
    @handles_errors()
    def validate_temporal_consistency_enhanced(self, data: pd.DataFrame) -> StatisticalValidationResult:
        """
        Enhanced temporal consistency validation with statistical checks.
        
        Args:
            data: DataFrame with temporal data
            
        Returns:
            StatisticalValidationResult with temporal validation
        """
        start_time = time.time()
        self.logger.info("🔍 Starting enhanced temporal consistency validation...")
        
        try:
            warnings = []
            errors = []
            recommendations = []
            details = {}
            statistical_tests = {}
            
            self.logger.info(f"📊 Validating temporal consistency for {len(data)} rows")
            
            # 1. Basic temporal ordering
            if hasattr(data.index, 'to_pydatetime'):
                self.logger.info("🔍 Checking temporal ordering...")
                
                ordering_violations = 0
                for i in range(1, len(data.index)):
                    if data.index[i] < data.index[i-1]:
                        ordering_violations += 1
                
                details['ordering_violations'] = ordering_violations
                
                if ordering_violations > 0:
                    errors.append(f"Found {ordering_violations} temporal ordering violations")
                    recommendations.append("Sort data by timestamp before processing")
            
            # 2. Time gap analysis
            if hasattr(data.index, 'to_pydatetime'):
                self.logger.info("📊 Analyzing time gaps...")
                
                time_diffs = data.index.to_series().diff()
                expected_interval = time_diffs.mode().iloc[0] if len(time_diffs) > 0 else None
                
                if expected_interval:
                    gaps = time_diffs[time_diffs > expected_interval * 2]
                    gap_count = len(gaps)
                    details['time_gaps'] = {
                        'count': gap_count,
                        'expected_interval': str(expected_interval),
                        'largest_gap': str(gaps.max()) if gap_count > 0 else None
                    }
                    
                    if gap_count > 0:
                        warnings.append(f"Found {gap_count} time gaps in data")
                        recommendations.append("Review data collection for missing periods")
                        
                        # Statistical analysis of gaps
                        gap_ratios = gaps / expected_interval
                        gap_statistics = {
                            'mean_ratio': float(gap_ratios.mean()),
                            'std_ratio': float(gap_ratios.std()),
                            'max_ratio': float(gap_ratios.max())
                        }
                        details['gap_statistics'] = gap_statistics
                        
                        if gap_statistics['max_ratio'] > 10:
                            warnings.append(f"Very large time gap detected: {gap_statistics['max_ratio']:.1f}x expected interval")
            
            # 3. Duplicate timestamp detection
            if hasattr(data.index, 'duplicated'):
                self.logger.info("🔍 Detecting duplicate timestamps...")
                
                duplicate_timestamps = data.index.duplicated().sum()
                details['duplicate_timestamps'] = duplicate_timestamps
                
                if duplicate_timestamps > 0:
                    errors.append(f"Found {duplicate_timestamps} duplicate timestamps")
                    recommendations.append("Remove duplicate timestamp entries")
                    
                    # Analyze duplicate patterns
                    duplicate_indices = data.index[data.index.duplicated(keep=False)]
                    if len(duplicate_indices) > 0:
                        duplicate_groups = duplicate_indices.value_counts()
                        details['duplicate_patterns'] = {
                            'max_duplicates': int(duplicate_groups.max()),
                            'avg_duplicates': float(duplicate_groups.mean())
                        }
            
            # 4. Frequency analysis
            if hasattr(data.index, 'to_pydatetime'):
                self.logger.info("📊 Analyzing data frequency...")
                
                time_diffs = data.index.to_series().diff().dropna()
                if len(time_diffs) > 0:
                    # Convert to seconds for analysis
                    time_diffs_seconds = time_diffs.dt.total_seconds()
                    
                    frequency_stats = {
                        'mean_interval_seconds': float(time_diffs_seconds.mean()),
                        'std_interval_seconds': float(time_diffs_seconds.std()),
                        'min_interval_seconds': float(time_diffs_seconds.min()),
                        'max_interval_seconds': float(time_diffs_seconds.max()),
                        'median_interval_seconds': float(time_diffs_seconds.median())
                    }
                    details['frequency_statistics'] = frequency_stats
                    
                    # Check for irregular frequency
                    cv = frequency_stats['std_interval_seconds'] / frequency_stats['mean_interval_seconds']
                    if cv > 0.1:  # Coefficient of variation > 10%
                        warnings.append(f"Irregular data frequency detected (CV: {cv:.2f})")
                        recommendations.append("Review data collection for consistent timing")
            
            # 5. Seasonality detection
            if hasattr(data.index, 'to_pydatetime') and len(data) > 100:
                self.logger.info("📊 Detecting seasonality patterns...")
                
                try:
                    # Extract time components
                    hour_of_day = data.index.hour
                    day_of_week = data.index.dayofweek
                    
                    # Check for hour-of-day patterns
                    hour_counts = hour_of_day.value_counts().sort_index()
                    hour_entropy = stats.entropy(hour_counts)
                    details['hour_entropy'] = float(hour_entropy)
                    
                    if hour_entropy < 3.0:  # Low entropy indicates strong patterns
                        warnings.append("Strong hour-of-day patterns detected")
                        recommendations.append("Consider time-of-day effects in analysis")
                    
                    # Check for day-of-week patterns
                    day_counts = day_of_week.value_counts().sort_index()
                    day_entropy = stats.entropy(day_counts)
                    details['day_entropy'] = float(day_entropy)
                    
                    if day_entropy < 2.0:  # Very low entropy
                        warnings.append("Strong day-of-week patterns detected")
                        recommendations.append("Consider day-of-week effects in analysis")
                
                except Exception as e:
                    self.logger.warning(f"⚠️ Seasonality detection failed: {e}")
                    details['seasonality_error'] = str(e)
            
            # Calculate overall score using safe math operations
            score = 1.0
            score = safe_divide(score - len(errors) * 0.3, 1.0, score)
            score = safe_divide(score - len(warnings) * 0.1, 1.0, score)
            score = max(score, 0.0)
            
            passed = len(errors) == 0
            
            computation_time = time.time() - start_time
            
            result = StatisticalValidationResult(
                passed=passed,
                score=score,
                warnings=warnings,
                errors=errors,
                recommendations=recommendations,
                details=details,
                statistical_tests=statistical_tests,
                computation_time=computation_time
            )
            
            self.logger.info(f"✅ Enhanced temporal consistency validation completed in {computation_time:.3f}s")
            self.logger.info(f"📊 Validation score: {score:.3f}, Passed: {passed}")
            
            return result
            
        except Exception as e:
            computation_time = time.time() - start_time
            self.logger.error(f"❌ Enhanced temporal consistency validation failed after {computation_time:.3f}s: {e}")
            self.logger.error(f"🔍 Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.error(f"📋 Stack trace: {traceback.format_exc()}")
            
            return StatisticalValidationResult(
                passed=False, score=0.0, warnings=[], errors=[f"Validation failed: {str(e)}"],
                recommendations=["Fix validation errors before proceeding"], details={'error': str(e)},
                statistical_tests={}, computation_time=computation_time
            )
    
    @traced(span_name='detect_sophisticated_bias')
    @validates()
    @handles_errors()
    def detect_sophisticated_bias(self, data: pd.DataFrame, 
                                barrier_params: Dict[str, Any]) -> BiasDetectionResult:
        """
        Sophisticated bias detection with multiple statistical tests.
        
        Args:
            data: DataFrame with price data and labels
            barrier_params: Triple barrier parameters
            
        Returns:
            BiasDetectionResult with sophisticated bias analysis
        """
        start_time = time.time()
        self.logger.info("🔍 Starting sophisticated bias detection...")
        
        try:
            bias_types = []
            temporal_violations = 0
            future_data_leakage = False
            statistical_anomalies = []
            recommendations = []
            details = {}
            
            max_lookahead = barrier_params.get('max_lookahead', 100)
            self.logger.info(f"📊 Analyzing bias with max lookahead: {max_lookahead}")
            
            # 1. Information leakage detection
            if 'label' in data.columns and 'close' in data.columns:
                self.logger.info("🔍 Detecting information leakage...")
                
                # Calculate future returns
                future_returns = data['close'].pct_change().shift(-1)
                
                # Correlation analysis
                label_correlation = data['label'].corr(future_returns)
                details['label_future_correlation'] = float(label_correlation) if not pd.isna(label_correlation) else 0.0
                
                if abs(label_correlation) > 0.3:
                    bias_types.append("information_leakage")
                    statistical_anomalies.append(f"High label-future return correlation: {label_correlation:.3f}")
                    recommendations.append("Review labeling logic for future data leakage")
                
                # Granger causality test (simplified)
                if len(data) > 100:
                    try:
                        from statsmodels.tsa.stattools import grangercausalitytests
                        
                        # Prepare data for Granger test
                        test_data = pd.DataFrame({
                            'label': data['label'].fillna(0),
                            'future_return': future_returns.fillna(0)
                        }).dropna()
                        
                        if len(test_data) > 50:
                            # Test if labels Granger-cause future returns
                            gc_result = grangercausalitytests(test_data[['future_return', 'label']], maxlag=1, verbose=False)
                            gc_p_value = gc_result[1][0]['ssr_ftest'][1]
                            
                            details['granger_causality'] = {
                                'p_value': float(gc_p_value),
                                'significant': gc_p_value < 0.05
                            }
                            
                            if gc_p_value < 0.05:
                                bias_types.append("granger_causality")
                                statistical_anomalies.append(f"Labels Granger-cause future returns (p={gc_p_value:.4f})")
                                recommendations.append("Labels appear to predict future returns - potential bias")
                    
                    except Exception as e:
                        self.logger.warning(f"⚠️ Granger causality test failed: {e}")
                        details['granger_causality'] = {'error': str(e)}
            
            # 2. Perfect timing detection
            self.logger.info("🔍 Detecting perfect timing...")
            
            perfect_timing_count = 0
            timing_analysis = []
            
            for i in range(len(data) - max_lookahead):
                if pd.isna(data['label'].iloc[i]) or data['label'].iloc[i] == 0:
                    continue
                
                # Analyze future window
                future_window = data['close'].iloc[i+1:i+max_lookahead+1]
                if len(future_window) == 0:
                    continue
                
                # Calculate future price movements
                future_returns = future_window.pct_change()
                max_positive_return = future_returns.max()
                max_negative_return = future_returns.min()
                
                # Check for perfect timing
                label = data['label'].iloc[i]
                perfect_timing = False
                
                if label == 1 and max_positive_return > 0.002:  # Buy signal with >0.2% gain
                    perfect_timing = True
                elif label == -1 and max_negative_return < -0.002:  # Sell signal with >0.2% loss
                    perfect_timing = True
                
                if perfect_timing:
                    perfect_timing_count += 1
                    timing_analysis.append({
                        'index': i,
                        'label': label,
                        'max_positive': float(max_positive_return),
                        'max_negative': float(max_negative_return)
                    })
            
            perfect_timing_rate = perfect_timing_count / len(data)
            details['perfect_timing'] = {
                'count': perfect_timing_count,
                'rate': float(perfect_timing_rate),
                'analysis': timing_analysis[:10]  # Store first 10 examples
            }
            
            if perfect_timing_rate > 0.1:  # >10% perfect timing
                bias_types.append("perfect_timing")
                statistical_anomalies.append(f"High perfect timing rate: {perfect_timing_rate:.1%}")
                recommendations.append("Review labeling for unrealistic timing accuracy")
            
            # 3. Statistical anomaly detection
            self.logger.info("📊 Detecting statistical anomalies...")
            
            if 'label' in data.columns:
                labels = data['label'].dropna()
                
                # Label distribution analysis
                label_counts = labels.value_counts()
                label_entropy = stats.entropy(label_counts)
                details['label_entropy'] = float(label_entropy)
                
                if label_entropy < 0.5:  # Very low entropy
                    bias_types.append("label_distribution_bias")
                    statistical_anomalies.append(f"Low label entropy: {label_entropy:.3f}")
                    recommendations.append("Labels are highly concentrated - potential bias")
                
                # Label sequence analysis
                if len(labels) > 10:
                    # Check for alternating patterns
                    alternating_count = 0
                    for i in range(len(labels) - 2):
                        if labels.iloc[i] != labels.iloc[i+1] and labels.iloc[i] == labels.iloc[i+2]:
                            alternating_count += 1
                    
                    alternating_rate = alternating_count / (len(labels) - 2)
                    details['alternating_rate'] = float(alternating_rate)
                    
                    if alternating_rate > 0.3:  # >30% alternating
                        bias_types.append("sequence_bias")
                        statistical_anomalies.append(f"High alternating pattern rate: {alternating_rate:.1%}")
                        recommendations.append("Labels show strong alternating patterns - potential bias")
            
            # 4. Temporal bias detection
            self.logger.info("🔍 Detecting temporal bias...")
            
            if hasattr(data.index, 'to_pydatetime') and 'label' in data.columns:
                # Check for time-of-day bias
                hour_labels = data.groupby(data.index.hour)['label'].mean()
                hour_variance = hour_labels.var()
                details['hour_label_variance'] = float(hour_variance)
                
                if hour_variance > 0.1:  # High variance across hours
                    bias_types.append("temporal_bias")
                    statistical_anomalies.append(f"High hour-of-day label variance: {hour_variance:.3f}")
                    recommendations.append("Labels vary significantly by hour - potential temporal bias")
                
                # Check for day-of-week bias
                day_labels = data.groupby(data.index.dayofweek)['label'].mean()
                day_variance = day_labels.var()
                details['day_label_variance'] = float(day_variance)
                
                if day_variance > 0.1:  # High variance across days
                    bias_types.append("day_bias")
                    statistical_anomalies.append(f"High day-of-week label variance: {day_variance:.3f}")
                    recommendations.append("Labels vary significantly by day - potential day bias")
            
            # 5. Market regime bias detection
            if 'hmm_regime' in data.columns and 'label' in data.columns:
                self.logger.info("🔍 Detecting regime bias...")
                
                regime_labels = data.groupby('hmm_regime')['label'].agg(['mean', 'std', 'count'])
                regime_bias_score = regime_labels['mean'].std()
                details['regime_bias'] = {
                    'bias_score': float(regime_bias_score),
                    'regime_stats': regime_labels.to_dict()
                }
                
                if regime_bias_score > 0.2:  # High variance across regimes
                    bias_types.append("regime_bias")
                    statistical_anomalies.append(f"High regime label variance: {regime_bias_score:.3f}")
                    recommendations.append("Labels vary significantly across regimes - potential regime bias")
            
            # Calculate overall bias score using safe math operations
            bias_score = safe_divide(len(bias_types) * 0.2 + len(statistical_anomalies) * 0.1, 1.0, 0.0)
            bias_score = min(bias_score, 1.0)
            
            bias_detected = len(bias_types) > 0 or len(statistical_anomalies) > 0
            
            computation_time = time.time() - start_time
            
            # Update performance stats
            self.performance_stats['total_validations'] += 1
            self.performance_stats['bias_detections'] += 1
            self.performance_stats['total_computation_time'] += computation_time
            self.performance_stats['avg_computation_time'] = (
                self.performance_stats['total_computation_time'] / 
                self.performance_stats['total_validations']
            )
            
            result = BiasDetectionResult(
                bias_detected=bias_detected,
                bias_score=bias_score,
                bias_types=bias_types,
                temporal_violations=temporal_violations,
                future_data_leakage=future_data_leakage,
                statistical_anomalies=statistical_anomalies,
                recommendations=recommendations,
                details=details,
                computation_time=computation_time
            )
            
            self.logger.info(f"✅ Sophisticated bias detection completed in {computation_time:.3f}s")
            self.logger.info(f"📊 Bias score: {bias_score:.3f}, Bias detected: {bias_detected}")
            self.logger.info(f"🔍 Bias types: {bias_types}")
            self.logger.info(f"⚠️ Statistical anomalies: {len(statistical_anomalies)}")
            
            return result
            
        except Exception as e:
            computation_time = time.time() - start_time
            self.logger.error(f"❌ Sophisticated bias detection failed after {computation_time:.3f}s: {e}")
            self.logger.error(f"🔍 Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.error(f"📋 Stack trace: {traceback.format_exc()}")
            
            return BiasDetectionResult(
                bias_detected=True, bias_score=1.0, bias_types=["detection_failed"],
                temporal_violations=0, future_data_leakage=True,
                statistical_anomalies=[f"Bias detection failed: {str(e)}"],
                recommendations=["Fix bias detection errors before proceeding"],
                details={'error': str(e)}, computation_time=computation_time
            )
    
    @traced(span_name='validate_label_quality_statistical')
    @validates()
    @handles_errors()
    def validate_label_quality_statistical(self, data: pd.DataFrame) -> StatisticalValidationResult:
        """
        Statistical validation of label quality.
        
        Args:
            data: DataFrame with labeled data
            
        Returns:
            StatisticalValidationResult with label quality analysis
        """
        start_time = time.time()
        self.logger.info("🔍 Starting statistical label quality validation...")
        
        try:
            warnings = []
            errors = []
            recommendations = []
            details = {}
            statistical_tests = {}
            
            # Fast fail validation
            if 'label' not in data.columns:
                errors.append("No label column found in data")
                return StatisticalValidationResult(
                    passed=False, score=0.0, warnings=warnings, errors=errors,
                    recommendations=["Generate labels before validation"], details=details,
                    statistical_tests=statistical_tests, computation_time=time.time() - start_time
                )
            
            labels = data['label'].dropna()
            
            if len(labels) == 0:
                errors.append("No valid labels found")
                return StatisticalValidationResult(
                    passed=False, score=0.0, warnings=warnings, errors=errors,
                    recommendations=["Check label generation process"], details=details,
                    statistical_tests=statistical_tests, computation_time=time.time() - start_time
                )
            
            self.logger.info(f"📊 Validating {len(labels)} labels")
            
            # 1. Label distribution analysis
            self.logger.info("📊 Analyzing label distribution...")
            
            label_counts = labels.value_counts()
            label_distribution = label_counts.to_dict()
            details['label_distribution'] = label_distribution
            
            # Check for extreme imbalance
            if len(label_counts) > 1:
                max_count = label_counts.max()
                min_count = label_counts.min()
                imbalance_ratio = max_count / min_count
                details['imbalance_ratio'] = float(imbalance_ratio)
                
                if imbalance_ratio > 10:
                    warnings.append(f"Severe label imbalance detected (ratio: {imbalance_ratio:.1f})")
                    recommendations.append("Consider using balanced sampling or different labeling strategy")
                
                # Statistical test for balance
                chi2_stat, chi2_p = stats.chisquare(label_counts)
                statistical_tests['balance_test'] = {
                    'chi2_statistic': float(chi2_stat),
                    'p_value': float(chi2_p),
                    'is_balanced': chi2_p > 0.05
                }
                
                if chi2_p <= 0.05:
                    warnings.append("Labels are significantly imbalanced (chi-square test)")
            
            # 2. Label consistency analysis
            self.logger.info("📊 Analyzing label consistency...")
            
            # Calculate label change rate
            label_changes = (labels != labels.shift(1)).sum()
            change_rate = label_changes / len(labels)
            details['change_rate'] = float(change_rate)
            
            if change_rate < 0.01:  # <1% changes
                warnings.append("Very low label change rate - possible temporal bias")
                recommendations.append("Review labeling for temporal consistency")
            elif change_rate > 0.5:  # >50% changes
                warnings.append("Very high label change rate - possible noise")
                recommendations.append("Review labeling for excessive noise")
            
            # 3. Label sequence analysis
            self.logger.info("📊 Analyzing label sequences...")
            
            # Check for impossible sequences
            impossible_sequences = 0
            for i in range(len(labels) - 2):
                seq = labels.iloc[i:i+3].values
                if seq[0] == seq[2] and seq[0] != seq[1] and seq[0] != 0:
                    impossible_sequences += 1
            
            impossible_rate = impossible_sequences / (len(labels) - 2)
            details['impossible_sequences'] = {
                'count': impossible_sequences,
                'rate': float(impossible_rate)
            }
            
            if impossible_rate > 0.05:  # >5% impossible sequences
                warnings.append(f"High number of impossible label sequences: {impossible_sequences}")
                recommendations.append("Review labeling logic for sequence consistency")
            
            # 4. Label autocorrelation analysis
            if len(labels) > 30:
                self.logger.info("📊 Analyzing label autocorrelation...")
                
                try:
                    from statsmodels.stats.diagnostic import acorr_ljungbox
                    
                    ljungbox_result = acorr_ljungbox(labels, lags=5, return_df=True)
                    autocorr_p = ljungbox_result['lb_pvalue'].iloc[-1]
                    statistical_tests['autocorrelation'] = {
                        'ljungbox_p_value': float(autocorr_p),
                        'has_autocorrelation': autocorr_p <= 0.05
                    }
                    
                    if autocorr_p <= 0.05:
                        warnings.append("Significant autocorrelation in labels detected")
                        recommendations.append("Labels may not be independent - consider temporal modeling")
                
                except Exception as e:
                    self.logger.warning(f"⚠️ Autocorrelation test failed: {e}")
                    statistical_tests['autocorrelation'] = {'error': str(e)}
            
            # 5. Label confidence analysis (if available)
            if 'label_confidence' in data.columns:
                self.logger.info("📊 Analyzing label confidence...")
                
                confidence_scores = data['label_confidence'].dropna()
                if len(confidence_scores) > 0:
                    confidence_stats = {
                        'mean': float(confidence_scores.mean()),
                        'std': float(confidence_scores.std()),
                        'min': float(confidence_scores.min()),
                        'max': float(confidence_scores.max()),
                        'median': float(confidence_scores.median())
                    }
                    details['confidence_statistics'] = confidence_stats
                    
                    if confidence_stats['mean'] < 0.5:
                        warnings.append("Low average label confidence")
                        recommendations.append("Review labeling process for confidence improvement")
                    
                    if confidence_stats['std'] > 0.3:
                        warnings.append("High variance in label confidence")
                        recommendations.append("Standardize confidence calculation process")
            
            # Calculate overall score using safe math operations
            score = 1.0
            score = safe_divide(score - len(errors) * 0.3, 1.0, score)
            score = safe_divide(score - len(warnings) * 0.1, 1.0, score)
            score = safe_divide(score - min(impossible_rate * 2, 0.3), 1.0, score)
            score = max(score, 0.0)
            
            passed = len(errors) == 0 and score > 0.7
            
            computation_time = time.time() - start_time
            
            result = StatisticalValidationResult(
                passed=passed,
                score=score,
                warnings=warnings,
                errors=errors,
                recommendations=recommendations,
                details=details,
                statistical_tests=statistical_tests,
                computation_time=computation_time
            )
            
            self.logger.info(f"✅ Statistical label quality validation completed in {computation_time:.3f}s")
            self.logger.info(f"📊 Validation score: {score:.3f}, Passed: {passed}")
            
            return result
            
        except Exception as e:
            computation_time = time.time() - start_time
            self.logger.error(f"❌ Statistical label quality validation failed after {computation_time:.3f}s: {e}")
            self.logger.error(f"🔍 Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.error(f"📋 Stack trace: {traceback.format_exc()}")
            
            return StatisticalValidationResult(
                passed=False, score=0.0, warnings=[], errors=[f"Validation failed: {str(e)}"],
                recommendations=["Fix validation errors before proceeding"], details={'error': str(e)},
                statistical_tests={}, computation_time=computation_time
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'performance_stats': self.performance_stats.copy(),
            'avg_computation_time': self.performance_stats['avg_computation_time'],
            'bias_detection_rate': (
                self.performance_stats['bias_detections'] / 
                max(1, self.performance_stats['total_validations'])
            ),
            'statistical_test_rate': (
                self.performance_stats['statistical_tests'] / 
                max(1, self.performance_stats['total_validations'])
            )
        }