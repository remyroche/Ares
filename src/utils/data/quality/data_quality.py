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

# Import our custom utilities
from src.utils.logger import system_logger

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

    def __init__(self, thresholds: Optional[QualityThresholds] = None) -> None:
        """Initialize data quality framework."""
        self.logger = system_logger.getChild('DataQualityFramework')
        self.thresholds = thresholds or QualityThresholds()
        
        # Quality policies
        self.quality_policies = {
            'strict_validation': True,
            'auto_clean': True,
            'profiling_enabled': True,
            'max_issues_critical': 0,
            'max_issues_high': 5,
            'max_issues_medium': 20,
            'max_issues_low': 100
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
        
        self.logger.info('🔧 Unified Data Quality Framework initialized')

    def validate_dataframe_quality(self, df: pd.DataFrame, context: str = '') -> QualityResult:
        """Validate DataFrame quality with comprehensive checks."""
        start_time = time.time()
        result = QualityResult()
        
        if df is None or df.empty:
            result.add_issue('empty_data', 'DataFrame is None or empty')
            return result
            
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
        
        # Calculate quality score
        result.quality_score = self._calculate_quality_score(df)
        result.execution_time = time.time() - start_time
        
        self._log_validation_results(result, context)
        return result

    def _validate_nan_values(self, df: pd.DataFrame, result: QualityResult) -> None:
        """Validate NaN values in DataFrame."""
        nan_counts = df.isnull().sum()
        total_nans = nan_counts.sum()
        nan_ratio = total_nans / (len(df) * len(df.columns)) if len(df) > 0 and len(df.columns) > 0 else 0
        
        result.add_metric('nan_count', total_nans)
        result.add_metric('nan_ratio', nan_ratio)
        result.add_metric('nan_by_column', nan_counts.to_dict())
        
        if nan_ratio > self.thresholds.max_nan_ratio:
            result.add_issue('nan_values', f'NaN ratio {nan_ratio:.4f} exceeds threshold {self.thresholds.max_nan_ratio}')
            
        high_nan_columns = nan_counts[nan_counts > len(df) * 0.1]
        if not high_nan_columns.empty:
            result.add_warning('high_nan_columns', f'Columns with >10% NaN: {list(high_nan_columns.index)}')

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
            result.add_issue('infinite_values', f'Found {total_infinites} infinite values in columns: {list(infinite_counts.keys())}')

    def _validate_constant_features(self, df: pd.DataFrame, result: QualityResult) -> None:
        """Validate constant features in DataFrame."""
        constant_features = []
        low_variance_features = []
        
        for col in df.columns:
            unique_count = df[col].nunique()
            if unique_count < self.thresholds.min_unique_values:
                constant_features.append(col)
            elif unique_count < 5:
                low_variance_features.append(col)
                
        result.add_metric('constant_features', constant_features)
        result.add_metric('low_variance_features', low_variance_features)
        
        if constant_features:
            result.add_issue('constant_features', f'Found {len(constant_features)} constant features: {constant_features}')
        if low_variance_features:
            result.add_warning('low_variance_features', f'Found {len(low_variance_features)} low variance features: {low_variance_features}')

    def _validate_price_anomalies(self, df: pd.DataFrame, result: QualityResult) -> None:
        """Validate price anomalies in OHLC data."""
        price_columns = [col for col in ['open', 'high', 'low', 'close'] if col in df.columns]
        if not price_columns:
            return
            
        anomalies = []
        for i in range(len(df)):
            row = df.iloc[i]
            for col in price_columns:
                if row[col] < -self.thresholds.price_tolerance:
                    anomalies.append({'row': i, 'column': col, 'value': row[col], 'type': 'negative_price'})
                    
            if all(col in price_columns for col in ['open', 'high', 'low', 'close']):
                if row['high'] < row['low']:
                    anomalies.append({'row': i, 'type': 'high_low_inversion', 'high': row['high'], 'low': row['low']})
                if row['close'] > row['high'] or row['close'] < row['low']:
                    anomalies.append({'row': i, 'type': 'close_outside_range', 'close': row['close'], 'high': row['high'], 'low': row['low']})
                    
        result.add_metric('price_anomalies', anomalies)
        if anomalies:
            result.add_issue('price_anomalies', f'Found {len(anomalies)} price anomalies')

    def _validate_timestamp_consistency(self, df: pd.DataFrame, result: QualityResult) -> None:
        """Validate timestamp consistency."""
        if 'timestamp' not in df.columns:
            return
            
        issues = []
        try:
            # Handle both datetime64[ns] and int64 timestamps
            if df['timestamp'].dtype == 'datetime64[ns]':
                timestamps = df['timestamp']
            else:
                timestamps = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
            
            invalid_timestamps = timestamps.isna().sum()
            if invalid_timestamps > 0:
                issues.append({'type': 'invalid_timestamps', 'count': invalid_timestamps})
            
            valid_timestamps = timestamps.dropna()
            if len(valid_timestamps) > 1:
                # Be more lenient with gaps - allow up to 2 hours for market data
                expected_interval = pd.Timedelta(minutes=1)
                time_diffs = valid_timestamps.diff().dropna()
                large_gaps = time_diffs[time_diffs > pd.Timedelta(hours=2)]
                if not large_gaps.empty:
                    issues.append({'type': 'large_gaps', 'count': len(large_gaps), 'max_gap_minutes': large_gaps.max().total_seconds() / 60})
            
            duplicates = valid_timestamps.duplicated()
            if duplicates.any():
                issues.append({'type': 'duplicate_timestamps', 'count': duplicates.sum()})
            
            # Check for future timestamps (handle timezone properly)
            now = pd.Timestamp.now()
            if valid_timestamps.dt.tz is not None:
                now = now.tz_localize('UTC')
            future_timestamps = valid_timestamps[valid_timestamps > now]
            if not future_timestamps.empty:
                issues.append({'type': 'future_timestamps', 'count': len(future_timestamps)})
                
        except Exception as e:
            issues.append({'type': 'timestamp_parsing_error', 'error': str(e)})
            
        result.add_metric('timestamp_issues', issues)
        if issues:
            result.add_issue('timestamp_issues', f'Found {len(issues)} timestamp issues')

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
        """Validate correlations between numeric columns, excluding OHLCV columns."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Exclude OHLCV columns from correlation analysis
        ohlcv_columns = {'open', 'high', 'low', 'close', 'volume', 'timestamp'}
        analysis_columns = [col for col in numeric_columns if col.lower() not in ohlcv_columns]
        
        if len(analysis_columns) < 2:
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
            if high_corr_pairs:
                result.add_warning('high_correlations', f'Found {len(high_corr_pairs)} highly correlated column pairs (excluding OHLCV)')
        except Exception as e:
            result.add_warning('correlation_calculation_error', f'Could not calculate correlations: {e}')

    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score (0-100)."""
        try:
            score = 100.0
            
            # Penalize NaN values
            null_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
            score -= null_percentage * 0.5
            
            # Penalize duplicates
            duplicate_percentage = df.duplicated().sum() / len(df) * 100
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

# Create global instance for backwards compatibility
data_quality_framework = DataQualityFramework()