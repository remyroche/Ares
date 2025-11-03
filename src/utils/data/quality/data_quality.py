"""
Unified Data Quality Framework

This module consolidates all data quality validation, cleaning, and analysis functionality
from multiple previous modules into a single, comprehensive framework.

Consolidated from:
- enhanced_data_quality_validator.py
- data_quality_framework.py
- data_qualification_base.py (validation parts)
"""

import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import psutil

# Import our custom utilities
import logging

# Import comprehensive duplicate analyzer
try:
    from src.utils.data.quality.comprehensive_duplicate_analyzer import (
        ComprehensiveDuplicateAnalyzer,
        analyze_duplicates_comprehensive
    )
    DUPLICATE_ANALYZER_AVAILABLE = True
except ImportError:
    DUPLICATE_ANALYZER_AVAILABLE = False
    ComprehensiveDuplicateAnalyzer = None

logger = logging.getLogger(__name__)

class OutlierSeverity(Enum):
    """Outlier severity levels."""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'

@dataclass
class QualityThresholds:
    """Quality validation thresholds."""
    max_nan_ratio: float = 0.0
    max_infinite_count: int = 0
    min_unique_values: int = 2
    max_constant_ratio: float = 0.95
    max_gap_hours: int = 48
    price_tolerance: float = 0.001
    volume_tolerance: float = 0.001
    max_correlation_threshold: float = 0.95
    min_feature_count: int = 40

@dataclass
class UnifiedMemoryConfig:
    """Unified memory management configuration across all components."""
    # Memory thresholds (percentage of available memory)
    threshold_percentage: float = 0.8  # 80% of available memory
    threshold_absolute_gb: float = 4.0  # 4GB absolute limit

    # Cleanup and garbage collection
    cleanup_frequency: int = 100  # operations
    gc_frequency: int = 50  # operations

    # Component-specific overrides
    component_overrides: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'DataStreamingManager': {'threshold_percentage': 0.8},
        'M1MemoryOptimizer': {'threshold_absolute_gb': 4.0},
        'FeatureSelection': {'threshold_percentage': 0.75},
        'HMMRegimeDetection': {'threshold_percentage': 0.85}
    })

    def get_effective_threshold(self, component_name: str = None, available_memory_gb: float = None) -> float:
        """Get the effective memory threshold for a component."""
        if available_memory_gb is None:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)

        # Get component-specific overrides
        if component_name and component_name in self.component_overrides:
            overrides = self.component_overrides[component_name]
            threshold_percentage = overrides.get('threshold_percentage', self.threshold_percentage)
            threshold_absolute_gb = overrides.get('threshold_absolute_gb', self.threshold_absolute_gb)
        else:
            threshold_percentage = self.threshold_percentage
            threshold_absolute_gb = self.threshold_absolute_gb

        # Return the more restrictive threshold
        percentage_limit = available_memory_gb * threshold_percentage
        return min(percentage_limit, threshold_absolute_gb)

    def should_cleanup(self, component_name: str, operation_count: int) -> bool:
        """Check if cleanup should be performed based on operation count."""
        if component_name in self.component_overrides:
            cleanup_freq = self.component_overrides[component_name].get('cleanup_frequency', self.cleanup_frequency)
        else:
            cleanup_freq = self.cleanup_frequency

        return operation_count % cleanup_freq == 0

    def should_gc(self, component_name: str, operation_count: int) -> bool:
        """Check if garbage collection should be performed."""
        if component_name in self.component_overrides:
            gc_freq = self.component_overrides[component_name].get('gc_frequency', self.gc_frequency)
        else:
            gc_freq = self.gc_frequency

        return operation_count % gc_freq == 0

@dataclass
class QualityResult:
    """Result of data quality validation."""
    passed: bool = True
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    quality_score: float = 100.0
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def add_issue(self, issue_type: str, description: str) -> None:
        """Add a quality issue."""
        self.issues.append(f'{issue_type}: {description}')
        self.passed = False

    def add_warning(self, warning_type: str, description: str) -> None:
        """Add a quality warning."""
        self.warnings.append(f'{warning_type}: {description}')

    def add_info(self, info_type: str, description: str) -> None:
        """Add informational message (not an issue or warning)."""
        if not hasattr(self, 'info_messages'):
            self.info_messages = []
        self.info_messages.append(f'{info_type}: {description}')

    def add_metric(self, name: str, value: Any) -> None:
        """Add a quality metric."""
        self.metrics[name] = value

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the validation result."""
        return {
            'passed': self.passed,
            'issue_count': len(self.issues),
            'warning_count': len(self.warnings),
            'quality_score': self.quality_score,
            'metrics': self.metrics,
            'issues': self.issues[:5],
            'warnings': self.warnings[:5]
        }

class DataQualityFramework:
    """Unified data quality framework with validation, cleaning, and profiling."""

    _instance = None
    _initialized = False
    _init_done = False

    def __new__(cls, thresholds: Optional[QualityThresholds] = None):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(DataQualityFramework, cls).__new__(cls)
        return cls._instance

    def __init__(self, thresholds: Optional[QualityThresholds] = None) -> None:
        """Initialize data quality framework (only once due to singleton)."""
        if DataQualityFramework._init_done:
            return

        start_time = time.time()
        self.logger = logging.getLogger('DataQualityFramework')
        self.thresholds = thresholds or QualityThresholds()

        # Initialize duplicate analyzer if available
        if DUPLICATE_ANALYZER_AVAILABLE:
            self.duplicate_analyzer = ComprehensiveDuplicateAnalyzer(self.logger)
            self.logger.info('✅ Comprehensive duplicate analyzer integrated')
        else:
            self.duplicate_analyzer = None
            self.logger.warning('⚠️ Comprehensive duplicate analyzer not available')

        # Quality policies
        self.quality_policies = {
            'strict_validation': True,
            'auto_clean': True,
            'profiling_enabled': True,
            'max_issues_critical': 0,
            'max_issues_high': 5,
            'max_issues_medium': 20,
            'max_issues_low': 100,
            'duplicate_analysis_enabled': DUPLICATE_ANALYZER_AVAILABLE
        }

        # Validation rules
        self.validation_rules = {
            'klines_schema': {
                'required_columns': ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
                'data_types': {
                    'timestamp': 'int64',
                    'open': 'float64',
                    'high': 'float64',
                    'low': 'float64',
                    'close': 'float64',
                    'volume': 'float64'
                },
                'constraints': {
                    'timestamp': {'min': 0, 'max': None},
                    'open': {'min': 0, 'max': None},
                    'high': {'min': 0, 'max': None},
                    'low': {'min': 0, 'max': None},
                    'close': {'min': 0, 'max': None},
                    'volume': {'min': 0, 'max': None}
                }
            },
            'features_schema': {
                'required_columns': ['timestamp'],
                'data_types': {'timestamp': 'int64'},
                'constraints': {'timestamp': {'min': 0, 'max': None}}
            },
            'labels_schema': {
                'required_columns': ['timestamp', 'label'],
                'data_types': {'timestamp': 'int64', 'label': 'int64'},
                'constraints': {
                    'timestamp': {'min': 0, 'max': None},
                    'label': {'min': 0, 'max': None}
                }
            }
        }

        self.logger.info('🔧 Unified Data Quality Framework initialized (singleton)')
        self._initialized = True
        DataQualityFramework._init_done = True

        # Add timing information (Numba-safe implementation)
        duration = time.time() - start_time
        try:
            from src.utils.tprint import tprint_performance
            tprint_performance("DataQualityFramework initialization", duration)
        except ImportError:
            # Fallback to basic logging (Numba-safe)
            self.logger.info(f"⏱️ DataQualityFramework initialized in {duration:.3f}s")

    def validate_dataframe_quality(self, df: pd.DataFrame, context: str = '') -> QualityResult:
        """Validate DataFrame quality with comprehensive checks."""
        start_time = time.time()
        result = QualityResult()

        if df is None or df.empty:
            result.add_issue('empty_data', 'DataFrame is None or empty')
            return result

        # Enhanced logging for debugging
        self.logger.info(f"🔍 Starting quality validation for {context}")
        self.logger.info(f"📊 DataFrame shape: {df.shape}")
        self.logger.info(f"📋 DataFrame columns ({len(df.columns)}): {list(df.columns)[:20]}{'...' if len(df.columns) > 20 else ''}")
        self.logger.info(f"📐 DataFrame index type: {type(df.index).__name__}")
        if hasattr(df.index, 'name'):
            self.logger.info(f"📐 DataFrame index name: {df.index.name}")
        self.logger.info(f"💾 Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

        result.add_metric('rows', len(df))
        result.add_metric('columns', len(df.columns))
        result.add_metric('memory_mb', df.memory_usage(deep=True).sum() / 1024 / 1024)

        # Run all validation checks
        self._validate_nan_values(df, result)
        self._validate_infinite_values(df, result)
        self._validate_constant_features(df, result)
        self._validate_price_anomalies(df, result)
        self._validate_timestamp_consistency(df, result)
        self._validate_data_types(df, result)
        self._validate_correlations(df, result)

        # Run comprehensive duplicate analysis if available
        if self.quality_policies.get('duplicate_analysis_enabled', False) and self.duplicate_analyzer:
            self._validate_duplicate_timestamps(df, result)

        # Store result for quality score calculation
        self._last_validation_result = result.metrics

        # Calculate quality score
        result.quality_score = self._calculate_quality_score(df)
        result.execution_time = time.time() - start_time

        self._log_validation_results(result, context)
        return result

    def _validate_nan_values(self, df: pd.DataFrame, result: QualityResult) -> None:
        """Validate NaN values in DataFrame with detailed per-column statistics."""
        # Define required OHLCV columns that MUST NOT have NaN values
        required_ohlcv_columns = {'open', 'high', 'low', 'close', 'volume'}
        
        # Define optional columns that CAN have NaN values
        optional_columns = {
            'quote_volume', 'quote_asset_volume',
            'trades_count', 'number_of_trades',
            'taker_buy_base_volume', 'taker_buy_base_asset_volume',
            'taker_buy_quote_volume', 'taker_buy_quote_asset_volume'
        }
        
        # Define calculated features that can legitimately have NaN values due to rolling calculations
        calculated_features = {
            'price_std', 'price_ma', 'price_ema', 'price_min', 'price_max',
            'volume_ma', 'volume_ratio', 'price_vs_ma', 'price_vs_ema',
            'year', 'month', 'day', 'hour', 'minute'  # Time-based features
        }

        # Calculate NaN statistics per column
        nan_counts = df.isnull().sum()
        total_rows = len(df)

        # Create detailed per-column statistics
        nan_stats = {}
        for col in df.columns:
            count = nan_counts[col]
            # Handle case where duplicate column names cause count to be a Series
            if hasattr(count, 'iloc'):
                count = count.iloc[0] if len(count) > 0 else 0
                # If still a Series after iloc, take the first value
                if hasattr(count, 'iloc'):
                    count = count.values[0] if len(count) > 0 else 0
            # Ensure count is a scalar
            if hasattr(count, '__iter__') and not isinstance(count, str):
                count = count[0] if len(count) > 0 else 0
            ratio = count / total_rows if total_rows > 0 else 0
            
            # Determine column type
            is_required = col in required_ohlcv_columns
            is_optional = col in optional_columns
            is_calculated = any(calc in col for calc in calculated_features)
            
            nan_stats[col] = {
                'count': int(count),
                'ratio': round(ratio, 4),
                'percentage': round(ratio * 100, 2),
                'is_required': is_required,
                'is_optional': is_optional,
                'is_calculated': is_calculated
            }

        # Calculate overall metrics
        total_nans = nan_counts.sum()
        overall_nan_ratio = total_nans / (total_rows * len(df.columns)) if total_rows > 0 and len(df.columns) > 0 else 0

        # Calculate metrics for REQUIRED columns only (excluding calculated and optional features)
        required_columns = [col for col in df.columns 
                          if col in required_ohlcv_columns 
                          and not any(calc in col for calc in calculated_features)]
        
        if required_columns:
            nan_counts_required = df[required_columns].isnull().sum()
            total_nans_required = nan_counts_required.sum()
            nan_ratio_required = total_nans_required / (len(df) * len(required_columns)) if len(df) > 0 and len(required_columns) > 0 else 0
        else:
            nan_ratio_required = 0
            total_nans_required = 0

        # Add metrics
        result.add_metric('nan_count', int(total_nans))
        result.add_metric('nan_ratio', round(overall_nan_ratio, 4))
        result.add_metric('nan_count_required', int(total_nans_required))
        result.add_metric('nan_ratio_required', round(nan_ratio_required, 4))
        result.add_metric('nan_stats_per_column', nan_stats)
        result.add_metric('nan_by_column', nan_counts.to_dict())

        # Enhanced logging for NaN analysis
        self.logger.info(f"📊 NaN Analysis: Total NaN count = {int(total_nans)}, Overall ratio = {overall_nan_ratio:.4f}")
        self.logger.info(f"📊 NaN Analysis (required OHLCV only): Total NaN count = {int(total_nans_required)}, Required ratio = {nan_ratio_required:.4f}")
        self.logger.info(f"📊 NaN threshold for required columns: {self.thresholds.max_nan_ratio}")
        
        # Quality gate - ONLY check required OHLCV columns
        if nan_ratio_required > self.thresholds.max_nan_ratio:
            self.logger.error(f"❌ REQUIRED columns have NaN ratio {nan_ratio_required:.4f} exceeds threshold {self.thresholds.max_nan_ratio}")
            result.add_issue('nan_values', f'Required OHLCV columns have NaN ratio {nan_ratio_required:.4f} exceeds threshold {self.thresholds.max_nan_ratio}')
        else:
            self.logger.info(f"✅ Required OHLCV columns are complete (NaN ratio: {nan_ratio_required:.4f})")
        
        # Report on optional columns as info only (not an issue)
        optional_cols_with_nans = [col for col in df.columns if col in optional_columns and nan_counts[col] > 0]
        if optional_cols_with_nans:
            self.logger.info(f"ℹ️ Optional columns with NaN values ({len(optional_cols_with_nans)}): {optional_cols_with_nans}")
            for col in optional_cols_with_nans[:5]:  # Show first 5
                pct = (nan_counts[col] / total_rows * 100) if total_rows > 0 else 0
                self.logger.info(f"   - {col}: {pct:.1f}% NaN")

        # Detailed per-column analysis for non-optional, non-calculated columns
        high_nan_columns = nan_counts[nan_counts > total_rows * 0.1]  # >10% NaN
        very_high_nan_columns = nan_counts[nan_counts > total_rows * 0.5]  # >50% NaN
        
        # Filter out optional and calculated columns from high NaN warnings
        high_nan_required = [col for col in high_nan_columns.index 
                            if col not in optional_columns and not any(calc in col for calc in calculated_features)]
        very_high_nan_required = [col for col in very_high_nan_columns.index 
                                 if col not in optional_columns and not any(calc in col for calc in calculated_features)]
        
        if len(high_nan_required) > 0:
            self.logger.warning(f"⚠️ Required columns with >10% NaN ({len(high_nan_required)}): {high_nan_required[:10]}")
        if len(very_high_nan_required) > 0:
            self.logger.error(f"❌ Required columns with >50% NaN ({len(very_high_nan_required)}): {very_high_nan_required[:10]}")

        # Categorize by NaN levels
        nan_categories = {
            'no_nan': [],  # 0% NaN
            'low_nan': [],  # 1-10% NaN
            'high_nan': [],  # 10-50% NaN
            'very_high_nan': [],  # >50% NaN
            'all_nan': []  # 100% NaN
        }

        for col in df.columns:
            ratio = nan_stats[col]['ratio']
            if ratio == 0:
                nan_categories['no_nan'].append(col)
            elif ratio <= 0.1:
                nan_categories['low_nan'].append(col)
            elif ratio <= 0.5:
                nan_categories['high_nan'].append(col)
            elif ratio < 1.0:
                nan_categories['very_high_nan'].append(col)
            else:
                nan_categories['all_nan'].append(col)

        # Add categorized metrics
        result.add_metric('nan_categories', nan_categories)

        # Add warnings and info messages based on categories
        if nan_categories['very_high_nan']:
            result.add_warning('very_high_nan_columns', f'Columns with >50% NaN: {nan_categories["very_high_nan"]}')

        if nan_categories['all_nan']:
            result.add_issue('all_nan_columns', f'Columns with 100% NaN: {nan_categories["all_nan"]}')

        # Separate calculated vs non-calculated features in warnings
        if not high_nan_columns.empty:
            calc_high_nan = [col for col in high_nan_columns.index if any(calc in col for calc in calculated_features)]
            non_calc_high_nan = [col for col in high_nan_columns.index if col not in calc_high_nan]

            if non_calc_high_nan:
                result.add_warning('high_nan_non_calc', f'Non-calculated columns with >10% NaN: {non_calc_high_nan}')

            if calc_high_nan:
                result.add_info('high_nan_calc', f'Calculated features with >10% NaN (may be expected): {calc_high_nan}')

        # Add summary statistics
        summary_stats = {
            'total_columns': len(df.columns),
            'columns_with_nan': len([col for col in df.columns if nan_counts[col] > 0]),
            'columns_no_nan': len(nan_categories['no_nan']),
            'columns_low_nan': len(nan_categories['low_nan']),
            'columns_high_nan': len(nan_categories['high_nan']),
            'columns_very_high_nan': len(nan_categories['very_high_nan']),
            'columns_all_nan': len(nan_categories['all_nan']),
            'required_columns_count': len(required_columns),
            'optional_columns_count': len([col for col in df.columns if col in optional_columns]),
            'calculated_features_count': len([col for col in df.columns if any(calc in col for calc in calculated_features)])
        }

        result.add_metric('nan_summary_stats', summary_stats)

    def _validate_infinite_values(self, df: pd.DataFrame, result: QualityResult) -> None:
        """Validate infinite values in DataFrame."""
        infinite_counts = {}
        total_infinites = 0

        for col in df.select_dtypes(include=[np.number]).columns:
            infinite_count = np.isinf(df[col]).sum()
            if infinite_count > 0:
                infinite_counts[col] = infinite_count
                total_infinites += infinite_count

        result.add_metric('infinite_count', total_infinites)
        result.add_metric('infinite_columns', infinite_counts)

        if total_infinites > self.thresholds.max_infinite_count:
            # Auto-fix infinite values in volume-related columns
            fixed_columns = []
            for col in infinite_counts.keys():
                if 'volume' in col.lower() and col in ['volume_return', 'volume_log_return']:
                    # Replace infinite values with reasonable bounds
                    if col == 'volume_return':
                        df[col] = df[col].replace([np.inf, -np.inf], [9.0, -9.0])
                    elif col == 'volume_log_return':
                        df[col] = df[col].replace([np.inf, -np.inf], [9.0, -9.0])
                    df[col] = df[col].fillna(0.0)
                    fixed_columns.append(col)

            if fixed_columns:
                result.add_warning('infinite_values_auto_fixed',
                                 f'Auto-fixed infinite values in columns: {fixed_columns}')
                # Re-count infinite values after fixing
                new_total_infinites = 0
                for col in df.select_dtypes(include=[np.number]).columns:
                    if col not in fixed_columns:  # Skip columns we already fixed
                        new_total_infinites += np.isinf(df[col]).sum()

                if new_total_infinites <= self.thresholds.max_infinite_count:
                    return  # Issue resolved
                else:
                    remaining_cols = []
                    for col in df.select_dtypes(include=[np.number]).columns:
                        if col not in fixed_columns and np.isinf(df[col]).sum() > 0:
                            remaining_cols.append(col)

                    result.add_issue('infinite_values',
                                   f'Found {new_total_infinites} infinite values in columns: {remaining_cols}')
            else:
                result.add_issue('infinite_values', f'Found {total_infinites} infinite values in columns: {list(infinite_counts.keys())}')

    def _validate_constant_features(self, df: pd.DataFrame, result: QualityResult) -> None:
        """Validate constant features in DataFrame with metadata awareness."""
        # Define metadata columns that are expected to be constant
        metadata_columns = {
            'exchange', 'symbol', 'timeframe', 'source', 'data_type',
            'version', 'collection_method', 'instrument_type'
        }

        # Define configuration columns that may be constant but are important
        # Note: aggtrades-derived features may be constant when aggtrades data is missing
        config_columns = {
            'trade_volume', 'trade_count', 'avg_price',
            'min_price', 'max_price', 'volume_ratio'
        }

        # Define columns that should be excluded from constant feature checks
        # when the underlying data source is missing
        excluded_constant_columns = {
            'trade_volume', 'trade_count', 'avg_price',
            'min_price', 'max_price', 'volume_ratio'
        }

        # Define data columns that should have variance
        data_columns = {
            'open', 'high', 'low', 'close', 'volume', 'price', 'quantity',
            'timestamp', 'trade_id', 'is_buyer_maker', 'quote_asset_volume',
            'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
        }

        constant_features = []
        low_variance_features = []
        expected_constants = []
        problematic_constants = []

        for col in df.columns:
            unique_count = df[col].nunique()

            # Skip excluded columns that may be constant due to missing data sources
            if col in excluded_constant_columns and unique_count <= 1:
                result.add_info('excluded_constant', f'Column {col} is constant but excluded from checks (likely missing aggtrades data)')
                continue

            # Categorize columns
            if col in metadata_columns:
                # Metadata columns are expected to be constant
                if unique_count == 1:
                    expected_constants.append(col)
                else:
                    # Metadata should be constant - this might be an issue
                    result.add_warning('metadata_variance', f'Metadata column {col} has {unique_count} unique values (expected 1)')
            elif col in config_columns:
                # Config columns may be constant but should be flagged for review
                if unique_count < self.thresholds.min_unique_values:
                    constant_features.append(col)
                    result.add_info('config_constant', f'Config column {col} is constant - verify if this is expected')
                elif unique_count < 5:
                    low_variance_features.append(col)
            elif col in data_columns:
                # Data columns should have variance
                if unique_count < self.thresholds.min_unique_values:
                    problematic_constants.append(col)
                    result.add_issue('data_constant', f'Data column {col} is constant - this may indicate data quality issues')
                elif unique_count < 5:
                    low_variance_features.append(col)
            else:
                # Unknown columns - use default logic
                if unique_count < self.thresholds.min_unique_values:
                    constant_features.append(col)
                elif unique_count < 5:
                    low_variance_features.append(col)

        # Add metrics with categorization
        result.add_metric('constant_features', constant_features)
        result.add_metric('low_variance_features', low_variance_features)
        result.add_metric('expected_constants', expected_constants)
        result.add_metric('problematic_constants', problematic_constants)

        # Only report issues for problematic constants, not expected ones
        if problematic_constants:
            result.add_issue('problematic_constants', f'Found {len(problematic_constants)} problematic constant data columns: {", ".join(problematic_constants)}')
        if low_variance_features:
            # Create detailed message with variance information
            variance_details = []
            for feature in low_variance_features[:5]:  # Show first 5
                try:
                    std_val = df[feature].std()
                    unique_val = df[feature].nunique()
                    variance_details.append(f"{feature}(unique={unique_val}, std={std_val:.6f})")
                except:
                    variance_details.append(f"{feature}(low_variance)")
            warning_msg = f'Found {len(low_variance_features)} low variance features: {", ".join(variance_details)}'
            if len(low_variance_features) > 5:
                warning_msg += f" ... and {len(low_variance_features) - 5} more"
            result.add_warning('low_variance_features', warning_msg)

        # Log expected constants as info, not issues
        if expected_constants:
            result.add_info('expected_constants', f'Found {len(expected_constants)} expected constant metadata columns: {", ".join(expected_constants)}')

        # Log all constant features for transparency
        if constant_features:
            all_constant_details = []
            for feature in constant_features[:10]:  # Show first 10
                try:
                    std_val = df[feature].std()
                    unique_val = df[feature].nunique()
                    all_constant_details.append(f"{feature}(unique={unique_val}, std={std_val:.6f})")
                except:
                    all_constant_details.append(f"{feature}(constant)")
            result.add_info('all_constant_features', f'All constant features: {", ".join(all_constant_details)}')

    def _validate_price_anomalies(self, df: pd.DataFrame, result: QualityResult) -> None:
        """Validate price anomalies in OHLC data."""
        price_columns = [col for col in ['open', 'high', 'low', 'close'] if col in df.columns]
        if not price_columns:
            return

        # Debug: Print price range information
        print(f"🔍 [PRICE_ANOMALY_DEBUG] Price tolerance threshold: {self.thresholds.price_tolerance}")
        for col in price_columns:
            if col in df.columns:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    # Convert to numeric if needed
                    try:
                        col_data_numeric = pd.to_numeric(col_data, errors='coerce')
                        if not col_data_numeric.isna().all():
                            print(f"🔍 [PRICE_ANOMALY_DEBUG] {col} range: min={col_data_numeric.min():.6f}, max={col_data_numeric.max():.6f}, mean={col_data_numeric.mean():.6f}")
                            negative_count = (col_data_numeric < -self.thresholds.price_tolerance).sum()
                            if negative_count > 0:
                                print(f"🔍 [PRICE_ANOMALY_DEBUG] {col} negative values: {negative_count} (below {self.thresholds.price_tolerance})")
                        else:
                            print(f"🔍 [PRICE_ANOMALY_DEBUG] {col} contains non-numeric data: {col_data.dtype}")
                    except Exception as e:
                        print(f"🔍 [PRICE_ANOMALY_DEBUG] {col} data type: {col_data.dtype}, sample: {col_data.head(3).tolist()}")

        anomalies = []
        negative_price_anomalies = 0
        high_low_inversions = 0
        close_outside_range = 0
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            # Convert price columns to numeric for validation
            numeric_row = {}
            for col in price_columns:
                try:
                    numeric_row[col] = pd.to_numeric(row[col], errors='coerce')
                except:
                    numeric_row[col] = np.nan
            
            # Check for negative prices
            for col in price_columns:
                if not pd.isna(numeric_row[col]) and numeric_row[col] < -self.thresholds.price_tolerance:
                    anomalies.append({'row': i, 'column': col, 'value': row[col], 'type': 'negative_price'})
                    negative_price_anomalies += 1

            # Check OHLC relationships if all columns are numeric
            if all(col in price_columns for col in ['open', 'high', 'low', 'close']):
                if (not pd.isna(numeric_row['high']) and not pd.isna(numeric_row['low']) and 
                    numeric_row['high'] < numeric_row['low']):
                    anomalies.append({'row': i, 'type': 'high_low_inversion', 'high': row['high'], 'low': row['low']})
                    high_low_inversions += 1
                if (not pd.isna(numeric_row['close']) and not pd.isna(numeric_row['high']) and not pd.isna(numeric_row['low']) and
                    (numeric_row['close'] > numeric_row['high'] or numeric_row['close'] < numeric_row['low'])):
                    anomalies.append({'row': i, 'type': 'close_outside_range', 'close': row['close'], 'high': row['high'], 'low': row['low']})
                    close_outside_range += 1

        # Debug: Print anomaly breakdown
        print(f"🔍 [PRICE_ANOMALY_DEBUG] Anomaly breakdown: negative_prices={negative_price_anomalies}, high_low_inversions={high_low_inversions}, close_outside_range={close_outside_range}")
        print(f"🔍 [PRICE_ANOMALY_DEBUG] Total anomalies: {len(anomalies)}")

        result.add_metric('price_anomalies', anomalies)
        if anomalies:
            result.add_issue('price_anomalies', f'Found {len(anomalies)} price anomalies')

    def _validate_timestamp_consistency(self, df: pd.DataFrame, result: QualityResult) -> None:
        """Validate timestamp consistency with klines-aware gap detection."""
        # Enhanced logging for timestamp detection
        self.logger.info(f"🕐 Timestamp validation: Checking for timestamp column or index")
        self.logger.info(f"🕐 DataFrame has 'timestamp' column: {'timestamp' in df.columns}")
        self.logger.info(f"🕐 DataFrame index name: {df.index.name}")
        self.logger.info(f"🕐 DataFrame index type: {type(df.index).__name__}")
        
        if 'timestamp' not in df.columns and df.index.name != 'timestamp':
            self.logger.warning(f"⚠️ Timestamp column 'timestamp' not found in DataFrame")
            self.logger.info(f"📋 Available columns: {list(df.columns)[:20]}")
            self.logger.info(f"📐 Index details: name={df.index.name}, type={type(df.index).__name__}")
            return

        issues = []
        try:
            # Handle both datetime64[ns] and int64 timestamps
            if df.index.name == 'timestamp' and isinstance(df.index, pd.DatetimeIndex):
                timestamps = df.index
                self.logger.info(f"🕐 Using DatetimeIndex as timestamp source")
            elif 'timestamp' in df.columns and df['timestamp'].dtype == 'datetime64[ns]':
                timestamps = df['timestamp']
                self.logger.info(f"🕐 Using 'timestamp' column (datetime64[ns]) as timestamp source")
            elif 'timestamp' in df.columns:
                timestamps = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
                self.logger.info(f"🕐 Converting 'timestamp' column to datetime using milliseconds")
            else:
                # Use DataFrame index if it's datetime
                if isinstance(df.index, pd.DatetimeIndex):
                    timestamps = df.index
                    self.logger.info(f"🕐 Using DatetimeIndex (no name) as timestamp source")
                else:
                    self.logger.warning(f"⚠️ No recognizable timestamp column or index found")
                    result.add_warning('timestamp_validation', 'No recognizable timestamp column or index found')
                    return

            invalid_timestamps = timestamps.isna().sum()
            if invalid_timestamps > 0:
                issues.append({'type': 'invalid_timestamps', 'count': invalid_timestamps})

            valid_timestamps = timestamps.dropna()
            if len(valid_timestamps) > 1:
                # Sort timestamps to ensure proper gap detection
                sorted_timestamps = valid_timestamps.sort_values()
                time_diffs = sorted_timestamps.diff().dropna()

                # Determine expected interval based on data characteristics
                median_diff = time_diffs.median()
                expected_interval_seconds = median_diff.total_seconds()

                # Only detect errors if gaps are superior to 65 seconds
                # This prevents false positives from small gaps that are likely processing artifacts
                gap_error_threshold = 65.0  # 65 seconds
                significant_gaps = time_diffs[time_diffs > pd.Timedelta(seconds=gap_error_threshold)]

                if not significant_gaps.empty:
                    # Filter out very small gaps that might be millisecond-level artifacts
                    real_gaps = significant_gaps[significant_gaps > pd.Timedelta(seconds=1)]
                    if not real_gaps.empty:
                        issues.append({
                            'type': 'large_gaps',
                            'count': len(real_gaps),
                            'max_gap_seconds': real_gaps.max().total_seconds(),
                            'expected_interval_seconds': expected_interval_seconds,
                            'gap_threshold_seconds': gap_error_threshold
                        })

            # Check for duplicate timestamps with detailed analysis
            duplicates = valid_timestamps.duplicated()
            duplicate_count = duplicates.sum()
            if duplicate_count > 0:
                # Get detailed information about duplicates
                duplicate_timestamps = valid_timestamps[duplicates]
                unique_duplicate_timestamps = duplicate_timestamps.unique()
                most_common_duplicates = duplicate_timestamps.value_counts().head(5)

                issues.append({
                    'type': 'duplicate_timestamps',
                    'count': duplicate_count,
                    'unique_duplicate_timestamps': len(unique_duplicate_timestamps),
                    'most_common_duplicates': most_common_duplicates.to_dict(),
                    'duplicate_percentage': (duplicate_count / len(valid_timestamps)) * 100
                })

            # Check for future timestamps (handle timezone properly)
            now = pd.Timestamp.now()

            # Handle DatetimeIndex vs Series
            if hasattr(valid_timestamps, 'dt'):
                ts_tz = valid_timestamps.dt.tz
            else:
                # For DatetimeIndex, check if it's timezone-aware
                ts_tz = valid_timestamps.tz

            if ts_tz is not None:
                now = now.tz_localize(ts_tz)
                future_timestamps = valid_timestamps[valid_timestamps > now]
            elif now.tz is not None:
                now = now.tz_convert('UTC')
                future_timestamps = valid_timestamps[valid_timestamps > now]
            else:
                future_timestamps = valid_timestamps[valid_timestamps > now]

            if not future_timestamps.empty:
                issues.append({'type': 'future_timestamps', 'count': len(future_timestamps)})

        except Exception as e:
            issues.append({'type': 'timestamp_parsing_error', 'error': str(e)})

        result.add_metric('timestamp_issues', issues)
        if issues:
            # Create detailed message with specific issue types
            issue_details = []
            for issue in issues:
                issue_type = issue.get('type', 'unknown')
                if issue_type == 'invalid_timestamps':
                    issue_details.append(f"{issue['count']} invalid timestamps")
                elif issue_type == 'large_gaps':
                    gap_seconds = issue.get('max_gap_seconds', 0)
                    if gap_seconds < 60:
                        issue_details.append(f"{issue['count']} small gaps (max {gap_seconds:.1f}s)")
                    else:
                        issue_details.append(f"{issue['count']} large gaps (max {gap_seconds/60:.1f} min)")
                elif issue_type == 'duplicate_timestamps':
                    duplicate_pct = issue.get('duplicate_percentage', 0)
                    issue_details.append(f"{issue['count']} duplicate timestamps ({duplicate_pct:.2f}% of data)")
                elif issue_type == 'future_timestamps':
                    issue_details.append(f"{issue['count']} future timestamps")
                elif issue_type == 'timestamp_parsing_error':
                    issue_details.append(f"parsing error: {issue['error']}")
                else:
                    issue_details.append(f"{issue_type}: {issue}")

            result.add_issue('timestamp_issues', f'Found {len(issues)} timestamp issues: {", ".join(issue_details)}')
        else:
            result.add_info('timestamp_consistency', 'No timestamp consistency issues found')

    def _validate_data_types(self, df: pd.DataFrame, result: QualityResult) -> None:
        """Validate data types in DataFrame."""
        issues = []
        for col in df.columns:
            try:
                if col in ['timestamp']:
                    # Accept both int64 and datetime64[ns] for timestamps
                    if not (pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col])):
                        issues.append({'column': col, 'expected': 'int64 or datetime64', 'actual': str(df[col].dtype)})
                elif col in ['open', 'high', 'low', 'close', 'volume']:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        issues.append({'column': col, 'expected': 'numeric', 'actual': str(df[col].dtype)})
            except Exception as e:
                issues.append({'column': col, 'error': f'Type validation error: {e}'})

        result.add_metric('data_type_issues', issues)
        if issues:
            result.add_issue('data_type_issues', f'Found {len(issues)} data type issues')

    def _validate_correlations(self, df: pd.DataFrame, result: QualityResult) -> None:
        """Validate correlations between numeric columns, excluding OHLCV and known correlated features."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        # Exclude OHLCV columns from correlation analysis
        ohlcv_columns = {'open', 'high', 'low', 'close', 'volume', 'timestamp'}

        # Define known correlated feature groups that should be excluded from warnings
        correlated_feature_groups = {
            # Price returns that are perfectly correlated (both use close price)
            'price_returns': {'close_return', 'close_log_return'},
            # Bollinger bands (all use same moving average calculation)
            'bollinger_bands': {'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position'},
            # Volume returns (both use volume data)
            'volume_returns': {'volume_return', 'volume_log_return'},
            # Price features (all derived from OHLC data)
            'price_features': {'price_range', 'price_range_pct', 'body_size', 'body_size_pct'},
            # Moving averages (similar calculations)
            'moving_averages': {'close_sma_5', 'close_sma_20', 'close_ema_12', 'close_ema_26'},
            # Lagged features (autocorrelated by design)
            'lagged_features': {col for col in df.columns if 'lag_' in col},
            # Future features (shouldn't correlate with current features for analysis)
            'future_features': {col for col in df.columns if 'future_' in col},
            # Expected OHLC correlations (avg_price is typically weighted average, min/max are naturally correlated)
            'expected_price_correlations': {'avg_price', 'min_price', 'max_price'},
        }

        # Flatten all correlated features to exclude
        excluded_features = ohlcv_columns.copy()
        for group in correlated_feature_groups.values():
            excluded_features.update(group)

        # Filter columns for analysis
        analysis_columns = [col for col in numeric_columns if col.lower() not in excluded_features]

        # Additional filtering: remove columns that are likely to be correlated due to similar naming
        filtered_analysis_columns = []
        for col in analysis_columns:
            # Skip if this column is part of a known correlated pattern
            skip_column = False
            col_lower = col.lower()

            # Check for RSI variations
            if 'rsi' in col_lower and any('rsi' in other.lower() for other in filtered_analysis_columns):
                skip_column = True
            # Check for MACD variations
            elif 'macd' in col_lower and any('macd' in other.lower() for other in filtered_analysis_columns):
                skip_column = True
            # Check for volatility variations
            elif 'volatility' in col_lower and any('volatility' in other.lower() for other in filtered_analysis_columns):
                skip_column = True
            # Check for ATR variations
            elif 'atr' in col_lower and any('atr' in other.lower() for other in filtered_analysis_columns):
                skip_column = True
            # Check for expected price correlations (avg_price with min_price/max_price)
            elif col in ['avg_price', 'min_price', 'max_price']:
                # Allow one of each type but skip correlations between them
                existing_price_cols = [c for c in filtered_analysis_columns if c in ['avg_price', 'min_price', 'max_price']]
                if existing_price_cols:
                    skip_column = True

            if not skip_column:
                filtered_analysis_columns.append(col)

        analysis_columns = filtered_analysis_columns

        if len(analysis_columns) < 2:
            result.add_info('correlation_analysis', f'Insufficient uncorrelated features for correlation analysis ({len(analysis_columns)} columns after filtering)')
            return

        try:
            corr_matrix = df[analysis_columns].corr()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > self.thresholds.max_correlation_threshold:
                        high_corr_pairs.append({
                            'col1': corr_matrix.columns[i],
                            'col2': corr_matrix.columns[j],
                            'correlation': corr_value
                        })

            result.add_metric('high_correlations', high_corr_pairs)
            result.add_metric('excluded_correlated_features', list(excluded_features))
            result.add_metric('analysis_columns', analysis_columns)

            if high_corr_pairs:
                # Filter out known timestamp correlations that are expected in crypto data
                filtered_pairs = []
                for pair in high_corr_pairs:
                    col1, col2 = pair['col1'], pair['col2']
                    # Skip timestamp correlations (open_time, close_time) as they're expected
                    if ('time' in col1.lower() and 'time' in col2.lower()) or \
                       (col1 == 'open_time' and col2 == 'close_time') or \
                       (col1 == 'close_time' and col2 == 'open_time'):
                        continue
                    filtered_pairs.append(pair)

                if filtered_pairs:
                    # Create detailed warning message with specific pairs
                    warning_msg = f'Found {len(filtered_pairs)} highly correlated column pairs (after excluding known correlated features):'
                    for pair in filtered_pairs[:3]:  # Show first 3 pairs to avoid spam
                        warning_msg += f" {pair['col1']}↔{pair['col2']}({pair['correlation']:.3f})"
                    if len(filtered_pairs) > 3:
                        warning_msg += f" ... and {len(filtered_pairs) - 3} more"
                    result.add_warning('high_correlations', warning_msg)
                else:
                    result.add_info('correlation_analysis', f'All high correlations are expected timestamp correlations in crypto data')
                result.add_info('correlation_filtering', f'Excluded {len(excluded_features)} known correlated features from analysis')
            else:
                result.add_info('correlation_analysis', f'No problematic correlations found in {len(analysis_columns)} analyzed features')
        except Exception as e:
            result.add_warning('correlation_calculation_error', f'Could not calculate correlations: {e}')

    def _validate_duplicate_timestamps(self, df: pd.DataFrame, result: QualityResult) -> None:
        """Validate for duplicate timestamps using comprehensive analysis."""
        if not self.duplicate_analyzer:
            return

        try:
            self.logger.info('🔍 Running comprehensive duplicate timestamp analysis...')

            # Run duplicate analysis
            analysis_result = self.duplicate_analyzer.analyze_duplicates(df)

            # Add comprehensive metrics
            result.add_metric('duplicate_analysis_available', True)
            result.add_metric('total_duplicate_records', analysis_result.total_duplicates)
            result.add_metric('duplicate_groups', analysis_result.duplicate_groups)
            result.add_metric('true_duplicates', analysis_result.true_duplicate_groups)
            result.add_metric('false_duplicates', analysis_result.false_duplicate_groups)
            result.add_metric('mixed_duplicates', analysis_result.mixed_duplicate_groups)

            # Add detailed duplicate analysis
            duplicate_details = {
                'summary_stats': analysis_result.summary_stats,
                'duplicate_type_distribution': analysis_result.summary_stats.get('duplicate_type_distribution', {}),
                'recommendations': analysis_result.recommendations
            }
            result.add_metric('duplicate_details', duplicate_details)

            # Add issues based on duplicate analysis
            if analysis_result.total_duplicates > 0:
                # Add warnings/info for different types of duplicates
                if analysis_result.false_duplicate_groups > 0:
                    result.add_issue('false_duplicates',
                                   f'Found {analysis_result.false_duplicate_groups} groups of false duplicates '
                                   '(same timestamp, different values) - requires investigation')

                if analysis_result.true_duplicate_groups > 0:
                    result.add_warning('true_duplicates',
                                     f'Found {analysis_result.true_duplicate_groups} groups of true duplicates '
                                     '(identical records) - safe to remove')

                if analysis_result.mixed_duplicate_groups > 0:
                    result.add_warning('mixed_duplicates',
                                     f'Found {analysis_result.mixed_duplicate_groups} groups of mixed duplicates '
                                     '- requires detailed analysis')

                # Add duplicate percentage for quality scoring
                duplicate_percentage = (analysis_result.total_duplicates / len(df)) * 100
                result.add_metric('duplicate_percentage', duplicate_percentage)

                # Add specific duplicate recommendations
                for recommendation in analysis_result.recommendations:
                    result.add_info('duplicate_recommendation', recommendation)

            self.logger.info(f'✅ Duplicate analysis completed: {analysis_result.total_duplicates} duplicates in {analysis_result.duplicate_groups} groups')

        except Exception as e:
            self.logger.warning(f'⚠️ Duplicate analysis failed: {e}')
            result.add_warning('duplicate_analysis_failed', f'Duplicate timestamp analysis failed: {e}')
            result.add_metric('duplicate_analysis_available', False)

    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score (0-100)."""
        try:
            score = 100.0

            # Penalize NaN values
            null_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
            score -= null_percentage * 0.5

            # Penalize duplicates (use enhanced duplicate analysis if available)
            duplicate_percentage = df.duplicated().sum() / len(df) * 100

            # Check if we have enhanced duplicate analysis results
            duplicate_metrics = getattr(self, '_last_validation_result', None)
            if duplicate_metrics and duplicate_metrics.get('duplicate_percentage'):
                # Use enhanced duplicate percentage which includes true/false duplicate distinction
                enhanced_dup_pct = duplicate_metrics['duplicate_percentage']
                duplicate_percentage = max(duplicate_percentage, enhanced_dup_pct)

            score -= duplicate_percentage * 0.3

            # Penalize infinite values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                infinite_ratio = np.isinf(df[numeric_cols]).sum().sum() / (len(df) * len(numeric_cols))
                score -= infinite_ratio * 100

            # Penalize negative prices
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in df.columns:
                    # Only check for negative values if the column is numeric
                    if pd.api.types.is_numeric_dtype(df[col]):
                        negative_ratio = (df[col] < 0).sum() / len(df)
                        score -= negative_ratio * 20

            return max(0.0, score)
        except Exception as e:
            self.logger.exception(f'Error calculating quality score: {e}')
            return 0.0

    def _log_validation_results(self, result: QualityResult, context: str) -> None:
        """Log validation results."""
        status = 'PASSED' if result.passed else 'FAILED'
        self.logger.info(f'Quality validation for {context}: {status} ({len(result.issues)} issues, {len(result.warnings)} warnings)')

        if result.issues:
            for issue in result.issues[:3]:
                self.logger.warning(f'  - {issue}')
            if len(result.issues) > 3:
                self.logger.warning(f'  ... and {len(result.issues) - 3} more issues')

        if result.warnings:
            for warning in result.warnings[:3]:
                self.logger.info(f'  - {warning}')
            if len(result.warnings) > 3:
                self.logger.info(f'  ... and {len(result.warnings) - 3} more warnings')

    def validate_data(self, data: pd.DataFrame, validation_rules: Optional[List[str]] = None) -> Dict[str, Any]:
        """Validate data according to specified validation rules."""
        if validation_rules is None:
            validation_rules = list(self.validation_rules.keys())

        validation_results = {
            'overall_passed': True,
            'passed_rules': 0,
            'failed_rules': 0,
            'total_rules': len(validation_rules),
            'rule_results': {},
            'critical_issues': 0,
            'high_issues': 0,
            'medium_issues': 0,
            'low_issues': 0,
            'errors': [],
            'warnings': []
        }

        for rule_name in validation_rules:
            if rule_name not in self.validation_rules:
                validation_results['warnings'].append(f'Unknown validation rule: {rule_name}')
                continue

            rule = self.validation_rules[rule_name]
            rule_result = self._apply_validation_rule(data, rule, rule_name)
            validation_results['rule_results'][rule_name] = rule_result

            if rule_result['passed']:
                validation_results['passed_rules'] += 1
            else:
                validation_results['failed_rules'] += 1
                validation_results['overall_passed'] = False

                for issue in rule_result['issues']:
                    severity = issue.get('severity', 'medium')
                    if severity == 'critical':
                        validation_results['critical_issues'] += 1
                    elif severity == 'high':
                        validation_results['high_issues'] += 1
                    elif severity == 'medium':
                        validation_results['medium_issues'] += 1
                    elif severity == 'low':
                        validation_results['low_issues'] += 1

        if not self._check_quality_policy_compliance(validation_results):
            validation_results['overall_passed'] = False

        self._log_validation_results_summary(validation_results)
        return validation_results

    def _apply_validation_rule(self, data: pd.DataFrame, rule: Dict[str, Any], rule_name: str) -> Dict[str, Any]:
        """Apply a specific validation rule to data."""
        rule_result = {'passed': True, 'issues': [], 'warnings': []}

        try:
            # Check required columns
            missing_columns = set(rule['required_columns']) - set(data.columns)
            if missing_columns:
                rule_result['passed'] = False
                rule_result['issues'].append({
                    'type': 'missing_columns',
                    'severity': 'critical',
                    'message': f'Missing required columns: {missing_columns}',
                    'details': list(missing_columns)
                })

            # Check data types
            for column, expected_type in rule['data_types'].items():
                if column in data.columns:
                    actual_type = str(data[column].dtype)
                    if actual_type != expected_type:
                        rule_result['warnings'].append({
                            'type': 'data_type_mismatch',
                            'severity': 'medium',
                            'message': f"Column '{column}' has type {actual_type}, expected {expected_type}",
                            'details': {'column': column, 'actual': actual_type, 'expected': expected_type}
                        })

            # Check constraints
            for column, constraints in rule['constraints'].items():
                if column in data.columns:
                    column_data = data[column]
                    if 'min' in constraints and constraints['min'] is not None:
                        min_violations = (column_data < constraints['min']).sum()
                        if min_violations > 0:
                            rule_result['issues'].append({
                                'type': 'constraint_violation',
                                'severity': 'high',
                                'message': f"Column '{column}' has {min_violations} values below minimum {constraints['min']}",
                                'details': {'column': column, 'violations': min_violations, 'min': constraints['min']}
                            })
                    if 'max' in constraints and constraints['max'] is not None:
                        max_violations = (column_data > constraints['max']).sum()
                        if max_violations > 0:
                            rule_result['issues'].append({
                                'type': 'constraint_violation',
                                'severity': 'high',
                                'message': f"Column '{column}' has {max_violations} values above maximum {constraints['max']}",
                                'details': {'column': column, 'violations': max_violations, 'max': constraints['max']}
                            })

            # Check for infinite values
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                if column in data.columns:
                    infinite_count = np.isinf(data[column]).sum()
                    if infinite_count > 0:
                        rule_result['issues'].append({
                            'type': 'infinite_values',
                            'severity': 'critical',
                            'message': f"Column '{column}' has {infinite_count} infinite values",
                            'details': {'column': column, 'count': infinite_count}
                        })

            # Special OHLC validation
            if rule_name == 'klines_schema' and all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                ohlc_violations = ((data['high'] < data['low']) |
                                 (data['high'] < data['open']) |
                                 (data['high'] < data['close']) |
                                 (data['low'] > data['open']) |
                                 (data['low'] > data['close'])).sum()
                if ohlc_violations > 0:
                    rule_result['issues'].append({
                        'type': 'ohlc_inconsistency',
                        'severity': 'high',
                        'message': f'OHLC data has {ohlc_violations} inconsistent rows',
                        'details': {'violations': ohlc_violations}
                    })

            if rule_result['issues']:
                rule_result['passed'] = False

        except Exception as e:
            rule_result['passed'] = False
            rule_result['issues'].append({
                'type': 'validation_error',
                'severity': 'critical',
                'message': f'Error during validation: {str(e)}',
                'details': {'error': str(e)}
            })

        return rule_result

    def _check_quality_policy_compliance(self, validation_results: Dict[str, Any]) -> bool:
        """Check if validation results comply with quality policies."""
        summary = validation_results
        if summary['critical_issues'] > self.quality_policies['max_issues_critical']:
            return False
        if summary['high_issues'] > self.quality_policies['max_issues_high']:
            return False
        if summary['medium_issues'] > self.quality_policies['max_issues_medium']:
            return False
        return not summary['low_issues'] > self.quality_policies['max_issues_low']

    def _log_validation_results_summary(self, results: Dict[str, Any]) -> None:
        """Log validation results summary."""
        if results['overall_passed']:
            self.logger.info(f"Data validation passed: {results['passed_rules']}/{results['total_rules']} rules passed")
        else:
            self.logger.error(f"Data validation failed: {results['failed_rules']}/{results['total_rules']} rules failed")
            self.logger.error(f"Issues: Critical={results['critical_issues']}, High={results['high_issues']}, Medium={results['medium_issues']}, Low={results['low_issues']}")

# Convenience functions for backwards compatibility
def quick_validate_dataframe(df: pd.DataFrame, context: str = '') -> QualityResult:
    """Quick validation of DataFrame quality."""
    framework = DataQualityFramework()
    return framework.validate_dataframe_quality(df, context)

def validate_unified_dataframe(df: pd.DataFrame, context: str = '') -> QualityResult:
    """Validate unified DataFrame quality."""
    framework = DataQualityFramework()
    return framework.validate_dataframe_quality(df, context)

def check_dataframe_health(df: pd.DataFrame) -> Dict[str, Any]:
    """Quick health check of DataFrame."""
    if df is None or df.empty:
        return {'healthy': False, 'reason': 'DataFrame is None or empty'}

    nan_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns)) if len(df) > 0 and len(df.columns) > 0 else 0
    infinite_count = sum((np.isinf(df[col]).sum() for col in df.select_dtypes(include=[np.number]).columns))

    health_status = {
        'healthy': True,
        'shape': df.shape,
        'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'nan_ratio': nan_ratio,
        'infinite_count': infinite_count,
        'issues': []
    }

    if nan_ratio > 0.1:
        health_status['healthy'] = False
        health_status['issues'].append('High NaN ratio')
    if infinite_count > 0:
        health_status['healthy'] = False
        health_status['issues'].append('Infinite values present')

    return health_status

class UnifiedMemoryManager:
    """Unified memory management across all components."""

    def __init__(self, config: UnifiedMemoryConfig = None):
        self.config = config or UnifiedMemoryConfig()
        self.logger = logging.getLogger(f"{__name__}.UnifiedMemoryManager")
        self.operation_counts = {}  # Track operations per component

    def get_memory_threshold(self, component_name: str) -> float:
        """Get memory threshold for a specific component."""
        try:
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            threshold = self.config.get_effective_threshold(component_name, available_memory_gb)

            self.logger.debug(f"Memory threshold for {component_name}: {threshold:.2f}GB (available: {available_memory_gb:.2f}GB)")
            return threshold
        except ImportError:
            self.logger.warning("psutil not available, using default threshold")
            return self.config.threshold_absolute_gb

    def check_memory_usage(self, component_name: str) -> Dict[str, Any]:
        """Check current memory usage and return status."""
        try:
            memory = psutil.virtual_memory()
            threshold = self.get_memory_threshold(component_name)

            usage_gb = (memory.total - memory.available) / (1024**3)
            usage_percentage = memory.percent / 100

            status = {
                'component': component_name,
                'usage_gb': usage_gb,
                'usage_percentage': usage_percentage,
                'threshold_gb': threshold,
                'threshold_percentage': self.config.threshold_percentage,
                'available_gb': memory.available / (1024**3),
                'total_gb': memory.total / (1024**3),
                'near_limit': usage_gb > threshold * 0.9,  # 90% of threshold
                'over_limit': usage_gb > threshold
            }

            if status['over_limit']:
                self.logger.warning(f"⚠️ {component_name} memory usage ({usage_gb:.2f}GB) exceeds threshold ({threshold:.2f}GB)")
            elif status['near_limit']:
                self.logger.info(f"ℹ️ {component_name} memory usage ({usage_gb:.2f}GB) approaching threshold ({threshold:.2f}GB)")

            return status

        except ImportError:
            self.logger.warning("psutil not available for memory monitoring")
            return {'component': component_name, 'error': 'psutil not available'}

    def should_cleanup(self, component_name: str) -> bool:
        """Check if component should perform cleanup."""
        if component_name not in self.operation_counts:
            self.operation_counts[component_name] = 0

        self.operation_counts[component_name] += 1
        return self.config.should_cleanup(component_name, self.operation_counts[component_name])

    def should_gc(self, component_name: str) -> bool:
        """Check if component should perform garbage collection."""
        if component_name not in self.operation_counts:
            self.operation_counts[component_name] = 0

        return self.config.should_gc(component_name, self.operation_counts[component_name])

    def perform_cleanup(self, component_name: str, cleanup_func: callable = None) -> bool:
        """Perform cleanup if needed."""
        if self.should_cleanup(component_name):
            self.logger.info(f"🧹 Performing cleanup for {component_name}")
            if cleanup_func:
                try:
                    cleanup_func()
                    return True
                except Exception as e:
                    self.logger.error(f"❌ Cleanup failed for {component_name}: {e}")
                    return False
            else:
                # Default cleanup: garbage collection
                import gc
                gc.collect()
                return True
        return False

    def get_memory_status_summary(self) -> Dict[str, Any]:
        """Get memory status summary for all tracked components."""
        summary = {
            'timestamp': datetime.now(),
            'components': {},
            'overall_status': 'healthy'
        }

        for component_name in self.operation_counts.keys():
            status = self.check_memory_usage(component_name)
            summary['components'][component_name] = status

            if status.get('over_limit', False):
                summary['overall_status'] = 'critical'
            elif status.get('near_limit', False) and summary['overall_status'] == 'healthy':
                summary['overall_status'] = 'warning'

        return summary

class SimpleSchemaValidator:
    """Simple schema usage validation without complexity."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SimpleSchemaValidator")
        self.field_usage = {}  # Track field usage: {schema_name: {field_name: [operations]}}

    def track_field_usage(self, schema_name: str, field_name: str, operation: str):
        """Track how a field is used."""
        if schema_name not in self.field_usage:
            self.field_usage[schema_name] = {}
        if field_name not in self.field_usage[schema_name]:
            self.field_usage[schema_name][field_name] = []
        self.field_usage[schema_name][field_name].append(operation)

    def validate_schema_usage(self, schema_name: str, schema_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Validate schema usage with simple checks."""
        usage = self.field_usage.get(schema_name, {})
        required_fields = schema_definition.get('required_columns', [])
        optional_fields = schema_definition.get('optional_columns', [])
        all_schema_fields = set(required_fields + optional_fields)

        # Find unused required fields
        unused_required = [field for field in required_fields if field not in usage]

        # Find used fields not in schema
        used_fields = set(usage.keys())
        missing_from_schema = used_fields - all_schema_fields

        # Calculate usage coverage
        coverage = len(usage) / len(required_fields) if required_fields else 1.0

        result = {
            'schema_name': schema_name,
            'unused_required_fields': unused_required,
            'missing_from_schema': list(missing_from_schema),
            'usage_coverage': coverage,
            'total_required_fields': len(required_fields),
            'used_fields': len(usage),
            'recommendations': []
        }

        # Generate simple recommendations
        if unused_required:
            result['recommendations'].append(f"Consider removing unused required fields: {unused_required}")
        if missing_from_schema:
            result['recommendations'].append(f"Add missing fields to schema: {list(missing_from_schema)}")
        if coverage < 0.8:
            result['recommendations'].append(f"Low field usage coverage ({coverage:.1%}), review schema design")

        return result

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get summary of all schema usage."""
        summary = {
            'timestamp': datetime.now(),
            'schemas': {},
            'total_schemas': len(self.field_usage),
            'total_fields_tracked': sum(len(fields) for fields in self.field_usage.values())
        }

        for schema_name, fields in self.field_usage.items():
            summary['schemas'][schema_name] = {
                'fields_used': len(fields),
                'total_operations': sum(len(ops) for ops in fields.values()),
                'most_used_fields': sorted(fields.items(), key=lambda x: len(x[1]), reverse=True)[:5]
            }

        return summary

# Create global instance for backwards compatibility
data_quality_framework = DataQualityFramework()
unified_memory_manager = UnifiedMemoryManager()
schema_validator = SimpleSchemaValidator()

# Alias for backwards compatibility
DataQualityAnalyzer = DataQualityFramework

# Convenience functions for duplicate analysis
def analyze_duplicates_enhanced(df: pd.DataFrame, timestamp_column: str = 'timestamp'):
    """Convenience function for enhanced duplicate analysis."""
    if DUPLICATE_ANALYZER_AVAILABLE:
        return analyze_duplicates_comprehensive(df, timestamp_column)
    else:
        raise ImportError("Comprehensive duplicate analyzer not available")

def resolve_duplicates_enhanced(df: pd.DataFrame, strategy: str = 'manual_review',
                               timestamp_column: str = 'timestamp'):
    """Convenience function for enhanced duplicate resolution (MANUAL REVIEW ONLY)."""
    if DUPLICATE_ANALYZER_AVAILABLE:
        if strategy != 'manual_review':
            raise ValueError("Only 'manual_review' strategy is supported. Automatic resolution is disabled.")
        analyzer = ComprehensiveDuplicateAnalyzer()
        return analyzer.resolve_duplicates(df, strategy, timestamp_column)
    else:
        raise ImportError("Comprehensive duplicate analyzer not available")

def validate_with_duplicate_analysis(df: pd.DataFrame, context: str = '') -> QualityResult:
    """Validate dataframe quality including comprehensive duplicate analysis."""
    return data_quality_framework.validate_dataframe_quality(df, context)

# Enhanced quality check with duplicate focus
def check_duplicate_quality(df: pd.DataFrame, context: str = '') -> Dict[str, Any]:
    """Perform quality check with focus on duplicate analysis."""
    result = data_quality_framework.validate_dataframe_quality(df, context)

    # Extract duplicate-specific information
    duplicate_info = {
        'has_duplicates': result.metrics.get('total_duplicate_records', 0) > 0,
        'duplicate_count': result.metrics.get('total_duplicate_records', 0),
        'duplicate_groups': result.metrics.get('duplicate_groups', 0),
        'true_duplicates': result.metrics.get('true_duplicates', 0),
        'false_duplicates': result.metrics.get('false_duplicates', 0),
        'mixed_duplicates': result.metrics.get('mixed_duplicates', 0),
        'duplicate_percentage': result.metrics.get('duplicate_percentage', 0.0),
        'duplicate_analysis_available': result.metrics.get('duplicate_analysis_available', False),
        'quality_score': result.quality_score,
        'duplicate_issues': [issue for issue in result.issues if 'duplicate' in issue.lower()],
        'duplicate_warnings': [warning for warning in result.warnings if 'duplicate' in warning.lower()],
        'recommendations': result.metrics.get('duplicate_details', {}).get('recommendations', [])
    }

    return duplicate_info
