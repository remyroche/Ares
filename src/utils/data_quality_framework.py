from __future__ import annotations
'\nComprehensive Data Quality Framework\n\nThis module provides a comprehensive data quality framework that includes:\n- Data validation and schema enforcement\n- Data quality scoring and metrics\n- Data cleaning and preprocessing\n- Data profiling and analysis\n- Quality policy management\n- Cross-step quality consistency\n'
from datetime import datetime
from enum import Enum
from typing import Any
import numpy as np
import pandas as pd
from .enhanced_outlier_handler import OutlierSeverity, enhanced_outlier_handler
from .logger import system_logger
from copy import copy

class DataQualityLevel(Enum):
    """Data quality issue severity levels."""
    CRITICAL = 'critical'
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'

class DataFormat(Enum):
    """Standard data formats."""
    KLINES = 'klines'
    FEATURES = 'features'
    LABELS = 'labels'
    PREDICTIONS = 'predictions'
    METADATA = 'metadata'
    CONFIG = 'config'

class DataQualityFramework:
    """Comprehensive data quality framework with validation, cleaning, and profiling."""

    def __init__(self) -> None:
        """Initialize data quality framework."""
        self.logger = system_logger.getChild('DataQualityFramework')
        self.outlier_handler = enhanced_outlier_handler
        self.quality_policies = {'strict_validation': True, 'auto_clean': True, 'profiling_enabled': True, 'max_issues_critical': 0, 'max_issues_high': 5, 'max_issues_medium': 20, 'max_issues_low': 100}
        self.validation_rules = {'klines_schema': {'required_columns': ['timestamp', 'open', 'high', 'low', 'close', 'volume'], 'data_types': {'timestamp': 'int64', 'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'}, 'constraints': {'timestamp': {'min': 0, 'max': None}, 'open': {'min': 0, 'max': None}, 'high': {'min': 0, 'max': None}, 'low': {'min': 0, 'max': None}, 'close': {'min': 0, 'max': None}, 'volume': {'min': 0, 'max': None}}}, 'features_schema': {'required_columns': ['timestamp'], 'data_types': {'timestamp': 'int64'}, 'constraints': {'timestamp': {'min': 0, 'max': None}}}, 'labels_schema': {'required_columns': ['timestamp', 'label'], 'data_types': {'timestamp': 'int64', 'label': 'int64'}, 'constraints': {'timestamp': {'min': 0, 'max': None}, 'label': {'min': 0, 'max': None}}}}
        self.default_cleaning_rules = {'outlier_handling': 'detect_only', 'outlier_config': {'method': 'iqr', 'threshold': 1.5, 'severity_threshold': 'medium', 'raise_errors': False}, 'null_handling': 'drop', 'duplicate_handling': 'drop_first', 'data_type_validation': True, 'schema_validation': True}
        self.logger.info('🔧 Comprehensive Data Quality Framework initialized')

    def clean_data(self, data: pd.DataFrame, cleaning_rules: dict[str, Any]=None) -> pd.DataFrame:
        """Clean data according to specified rules.

        Args:
            data: Data to clean
            cleaning_rules: Cleaning configuration (uses defaults if None)

        Returns:
            Cleaned data
        """
        if cleaning_rules is None:
            cleaning_rules = self.default_cleaning_rules.copy()
        self.logger.info(f'🧹 Starting data cleaning for {len(data)} rows')
        original_shape = data.shape
        cleaned_data = data.copy()
        if cleaning_rules.get('schema_validation', True):
            cleaned_data = self._validate_schema(cleaned_data, cleaning_rules)
        if cleaning_rules.get('data_type_validation', True):
            cleaned_data = self._validate_data_types(cleaned_data, cleaning_rules)
        cleaned_data = self._handle_nulls(cleaned_data, cleaning_rules)
        cleaned_data = self._handle_duplicates(cleaned_data, cleaning_rules)
        cleaned_data = self._handle_outliers(cleaned_data, cleaning_rules)
        final_shape = cleaned_data.shape
        rows_removed = original_shape[0] - final_shape[0]
        cols_removed = original_shape[1] - final_shape[1]
        self.logger.info('✅ Data cleaning completed')
        self.logger.info(f'   Original shape: {original_shape}')
        self.logger.info(f'   Final shape: {final_shape}')
        self.logger.info(f'   Rows removed: {rows_removed}')
        self.logger.info(f'   Columns removed: {cols_removed}')
        return cleaned_data

    def validate_data(self, data: pd.DataFrame, validation_rules: list[str]=None) -> dict[str, Any]:
        """Validate data according to specified validation rules.

        Args:
            data: Data to validate
            validation_rules: List of validation rule names to apply

        Returns:
            Validation results
        """
        if validation_rules is None:
            validation_rules = list(self.validation_rules.keys())
        validation_results = {'overall_passed': True, 'passed_rules': 0, 'failed_rules': 0, 'total_rules': len(validation_rules), 'rule_results': {}, 'critical_issues': 0, 'high_issues': 0, 'medium_issues': 0, 'low_issues': 0, 'errors': [], 'warnings': []}
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
        self._log_validation_results(validation_results)
        return validation_results

    def _apply_validation_rule(self, data: pd.DataFrame, rule: dict[str, Any], rule_name: str) -> dict[str, Any]:
        """Apply a specific validation rule to data."""
        rule_result = {'passed': True, 'issues': [], 'warnings': []}
        try:
            missing_columns = set(rule['required_columns']) - set(data.columns)
            if missing_columns:
                rule_result['passed'] = False
                rule_result['issues'].append({'type': 'missing_columns', 'severity': 'critical', 'message': f'Missing required columns: {missing_columns}', 'details': list(missing_columns)})
            for column, expected_type in rule['data_types'].items():
                if column in data.columns:
                    actual_type = str(data[column].dtype)
                    if actual_type != expected_type:
                        rule_result['warnings'].append({'type': 'data_type_mismatch', 'severity': 'medium', 'message': f"Column '{column}' has type {actual_type}, expected {expected_type}", 'details': {'column': column, 'actual': actual_type, 'expected': expected_type}})
            for column, constraints in rule['constraints'].items():
                if column in data.columns:
                    column_data = data[column]
                    if 'min' in constraints and constraints['min'] is not None:
                        min_violations = (column_data < constraints['min']).sum()
                        if min_violations > 0:
                            rule_result['issues'].append({'type': 'constraint_violation', 'severity': 'high', 'message': f"Column '{column}' has {min_violations} values below minimum {constraints['min']}", 'details': {'column': column, 'violations': min_violations, 'min': constraints['min']}})
                    if 'max' in constraints and constraints['max'] is not None:
                        max_violations = (column_data > constraints['max']).sum()
                        if max_violations > 0:
                            rule_result['issues'].append({'type': 'constraint_violation', 'severity': 'high', 'message': f"Column '{column}' has {max_violations} values above maximum {constraints['max']}", 'details': {'column': column, 'violations': max_violations, 'max': constraints['max']}})
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                if column in data.columns:
                    infinite_count = np.isinf(data[column]).sum()
                    if infinite_count > 0:
                        rule_result['issues'].append({'type': 'infinite_values', 'severity': 'critical', 'message': f"Column '{column}' has {infinite_count} infinite values", 'details': {'column': column, 'count': infinite_count}})
            if rule_name == 'klines_schema' and all((col in data.columns for col in ['open', 'high', 'low', 'close'])):
                ohlc_violations = ((data['high'] < data['low']) | (data['high'] < data['open']) | (data['high'] < data['close']) | (data['low'] > data['open']) | (data['low'] > data['close'])).sum()
                if ohlc_violations > 0:
                    rule_result['issues'].append({'type': 'ohlc_inconsistency', 'severity': 'high', 'message': f'OHLC data has {ohlc_violations} inconsistent rows', 'details': {'violations': ohlc_violations}})
            if rule_result['issues']:
                rule_result['passed'] = False
        except Exception as e:
            rule_result['passed'] = False
            rule_result['issues'].append({'type': 'validation_error', 'severity': 'critical', 'message': f'Error during validation: {str(e)}', 'details': {'error': str(e)}})
        return rule_result

    def _check_quality_policy_compliance(self, validation_results: dict[str, Any]) -> bool:
        """Check if validation results comply with quality policies."""
        summary = validation_results
        if summary['critical_issues'] > self.quality_policies['max_issues_critical']:
            return False
        if summary['high_issues'] > self.quality_policies['max_issues_high']:
            return False
        if summary['medium_issues'] > self.quality_policies['max_issues_medium']:
            return False
        return not summary['low_issues'] > self.quality_policies['max_issues_low']

    def _log_validation_results(self, results: dict[str, Any]) -> None:
        """Log validation results."""
        if results['overall_passed']:
            self.logger.info(f"Data validation passed: {results['passed_rules']}/{results['total_rules']} rules passed")
        else:
            self.logger.error(f"Data validation failed: {results['failed_rules']}/{results['total_rules']} rules failed")
            self.logger.error(f"Issues: Critical={results['critical_issues']}, High={results['high_issues']}, Medium={results['medium_issues']}, Low={results['low_issues']}")

    def _validate_schema(self, data: pd.DataFrame, rules: dict[str, Any]) -> pd.DataFrame:
        """Validate data schema."""
        try:
            validation_result = self.outlier_handler.validate_data_schema(data, 'klines')
            if not validation_result['valid']:
                self.logger.warning(f"Schema validation issues: {validation_result['errors']}")
                validation_result = self.outlier_handler.validate_data_schema(data, 'features')
                if not validation_result['valid']:
                    self.logger.warning(f"Features schema validation issues: {validation_result['errors']}")
            return data
        except Exception as e:
            self.logger.exception(f'Schema validation error: {e}')
            return data

    def _validate_data_types(self, data: pd.DataFrame, rules: dict[str, Any]) -> pd.DataFrame:
        """Validate and fix data types."""
        try:
            for col in data.columns:
                if col == 'timestamp' and data[col].dtype != 'int64':
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce').astype('Int64')
                        self.logger.info(f'Converted {col} to int64')
                    except:
                        self.logger.warning(f'Could not convert {col} to int64')
                elif col in ['open', 'high', 'low', 'close', 'volume']:
                    if data[col].dtype not in ['float64', 'float32']:
                        try:
                            data[col] = pd.to_numeric(data[col], errors='coerce')
                            self.logger.info(f'Converted {col} to numeric')
                        except:
                            self.logger.warning(f'Could not convert {col} to numeric')
            return data
        except Exception as e:
            self.logger.exception(f'Data type validation error: {e}')
            return data

    def _handle_nulls(self, data: pd.DataFrame, rules: dict[str, Any]) -> pd.DataFrame:
        """Handle null values according to rules."""
        try:
            null_handling = rules.get('null_handling', 'drop')
            if null_handling == 'drop':
                original_rows = len(data)
                data = data.dropna()
                rows_removed = original_rows - len(data)
                if rows_removed > 0:
                    self.logger.info(f'Removed {rows_removed} rows with null values')
            elif null_handling == 'fill':
                for col in data.columns:
                    if data[col].dtype in ['float64', 'float32', 'int64']:
                        data[col] = data[col].fillna(data[col].median())
                    else:
                        data[col] = data[col].fillna(data[col].mode()[0] if len(data[col].mode()) > 0 else 'unknown')
                self.logger.info('Filled null values with appropriate defaults')
            return data
        except Exception as e:
            self.logger.exception(f'Null handling error: {e}')
            return data

    def _handle_duplicates(self, data: pd.DataFrame, rules: dict[str, Any]) -> pd.DataFrame:
        """Handle duplicate values according to rules."""
        try:
            duplicate_handling = rules.get('duplicate_handling', 'drop_first')
            if duplicate_handling == 'drop_first':
                original_rows = len(data)
                data = data.drop_duplicates()
                rows_removed = original_rows - len(data)
                if rows_removed > 0:
                    self.logger.info(f'Removed {rows_removed} duplicate rows')
            elif duplicate_handling == 'drop_last':
                original_rows = len(data)
                data = data.drop_duplicates(keep='first')
                rows_removed = original_rows - len(data)
                if rows_removed > 0:
                    self.logger.info(f'Removed {rows_removed} duplicate rows')
            return data
        except Exception as e:
            self.logger.exception(f'Duplicate handling error: {e}')
            return data

    def _handle_outliers(self, data: pd.DataFrame, rules: dict[str, Any]) -> pd.DataFrame:
        """Handle outliers according to rules."""
        try:
            outlier_handling = rules.get('outlier_handling', 'detect_only')
            outlier_config = rules.get('outlier_config', {})
            if outlier_handling == 'detect_only':
                outliers = self.outlier_handler.detect_outliers(data, method=outlier_config.get('method', 'iqr'), threshold=outlier_config.get('threshold', 1.5), raise_errors=outlier_config.get('raise_errors', False))
                if outliers:
                    self.logger.info(f'Detected {len(outliers)} outlier groups')
                    for outlier in outliers:
                        self.logger.warning(f'  {outlier.column}: {len(outlier.indices)} values, severity={outlier.severity.value}')
            elif outlier_handling == 'remove':
                severity_threshold = outlier_config.get('severity_threshold', 'medium')
                severity_map = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
                threshold_level = severity_map.get(severity_threshold, 1)
                outliers = self.outlier_handler.detect_outliers(data, method=outlier_config.get('method', 'iqr'), threshold=outlier_config.get('threshold', 1.5), raise_errors=False)
                high_severity_outliers = [o for o in outliers if o.severity.value >= threshold_level]
                if high_severity_outliers:
                    outlier_indices = set()
                    for outlier in high_severity_outliers:
                        outlier_indices.update(outlier.indices)
                    original_rows = len(data)
                    data = data.drop(data.index[list(outlier_indices)])
                    rows_removed = original_rows - len(data)
                    self.logger.info(f'Removed {rows_removed} rows with {severity_threshold}+ severity outliers')
            elif outlier_handling == 'cap':
                outlier_config = rules.get('outlier_config', {})
                method = outlier_config.get('method', 'iqr')
                threshold = outlier_config.get('threshold', 1.5)
                for col in data.select_dtypes(include=[np.number]).columns:
                    if method == 'iqr':
                        Q1 = data[col].quantile(0.25)
                        Q3 = data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                        capped_lower = (data[col] == lower_bound).sum()
                        capped_upper = (data[col] == upper_bound).sum()
                        if capped_lower > 0 or capped_upper > 0:
                            self.logger.info(f'Capped {capped_lower + capped_upper} outliers in {col}')
                    elif method == 'zscore':
                        mean_val = data[col].mean()
                        std_val = data[col].std()
                        lower_bound = mean_val - threshold * std_val
                        upper_bound = mean_val + threshold * std_val
                        data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                        capped_lower = (data[col] == lower_bound).sum()
                        capped_upper = (data[col] == upper_bound).sum()
                        if capped_lower > 0 or capped_upper > 0:
                            self.logger.info(f'Capped {capped_lower + capped_upper} outliers in {col}')
            return data
        except Exception as e:
            self.logger.exception(f'Outlier handling error: {e}')
            return data

    def generate_quality_report(self, data: pd.DataFrame) -> dict[str, Any]:
        """Generate comprehensive data quality report.

        Args:
            data: Data to analyze

        Returns:
            Quality report
        """
        try:
            return {'timestamp': datetime.now().isoformat(), 'data_shape': data.shape, 'data_types': data.dtypes.to_dict(), 'null_analysis': self._analyze_nulls(data), 'duplicate_analysis': self._analyze_duplicates(data), 'outlier_analysis': self._analyze_outliers(data), 'data_quality_score': self._calculate_quality_score(data), 'recommendations': self._generate_recommendations(data)}
        except Exception as e:
            self.logger.exception(f'Error generating quality report: {e}')
            return {'error': str(e)}

    def _analyze_nulls(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze null values in data."""
        try:
            null_counts = data.isnull().sum()
            null_percentages = null_counts / len(data) * 100
            return {'total_null_values': null_counts.sum(), 'columns_with_nulls': null_counts[null_counts > 0].to_dict(), 'null_percentages': null_percentages[null_percentages > 0].to_dict(), 'worst_column': null_counts.idxmax() if null_counts.max() > 0 else None, 'worst_percentage': max(0, null_percentages.max())}
        except Exception as e:
            return {'error': str(e)}

    def _analyze_duplicates(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze duplicate values in data."""
        try:
            duplicate_rows = data.duplicated().sum()
            duplicate_percentage = duplicate_rows / len(data) * 100
            return {'duplicate_rows': duplicate_rows, 'duplicate_percentage': duplicate_percentage, 'has_duplicates': duplicate_rows > 0}
        except Exception as e:
            return {'error': str(e)}

    def _analyze_outliers(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze outliers in data."""
        try:
            outliers = self.outlier_handler.detect_outliers(data, method='iqr', threshold=1.5, raise_errors=False)
            if not outliers:
                return {'total_outlier_groups': 0, 'severity_distribution': {}}
            severity_counts = {}
            column_counts = {}
            for outlier in outliers:
                severity = outlier.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                column = outlier.column
                if column not in column_counts:
                    column_counts[column] = {'count': 0, 'total_values': 0}
                column_counts[column]['count'] += 1
                column_counts[column]['total_values'] += len(outlier.indices)
            return {'total_outlier_groups': len(outliers), 'severity_distribution': severity_counts, 'column_distribution': column_counts, 'worst_column': max(column_counts.items(), key=lambda x: x[1]['total_values'])[0] if column_counts else None}
        except Exception as e:
            return {'error': str(e)}

    def _calculate_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate overall data quality score (0-100)."""
        try:
            score = 100.0
            null_percentage = data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100
            score -= null_percentage * 0.5
            duplicate_percentage = data.duplicated().sum() / len(data) * 100
            score -= duplicate_percentage * 0.3
            outliers = self.outlier_handler.detect_outliers(data, method='iqr', threshold=1.5, raise_errors=False)
            if outliers:
                critical_outliers = len([o for o in outliers if o.severity == OutlierSeverity.CRITICAL])
                high_outliers = len([o for o in outliers if o.severity == OutlierSeverity.HIGH])
                score -= critical_outliers * 5.0
                score -= high_outliers * 2.0
            return max(0.0, score)
        except Exception as e:
            self.logger.exception(f'Error calculating quality score: {e}')
            return 0.0

    def _generate_recommendations(self, data: pd.DataFrame) -> list[str]:
        """Generate data quality improvement recommendations."""
        recommendations = []
        try:
            null_analysis = self._analyze_nulls(data)
            if null_analysis.get('worst_percentage', 0) > 10:
                recommendations.append(f"High null percentage in {null_analysis['worst_column']}: {null_analysis['worst_percentage']:.1f}%")
            duplicate_analysis = self._analyze_duplicates(data)
            if duplicate_analysis.get('has_duplicates', False):
                recommendations.append(f"Remove {duplicate_analysis['duplicate_rows']} duplicate rows")
            outlier_analysis = self._analyze_outliers(data)
            if outlier_analysis.get('total_outlier_groups', 0) > 0:
                severity_dist = outlier_analysis.get('severity_distribution', {})
                if severity_dist.get('critical', 0) > 0:
                    recommendations.append('Critical outliers detected - investigate data source')
                if severity_dist.get('high', 0) > 5:
                    recommendations.append('Many high-severity outliers - consider outlier removal')
            for col, dtype in data.dtypes.items():
                if col == 'timestamp' and dtype != 'int64':
                    recommendations.append(f'Convert {col} to int64 for timestamp consistency')
                elif col in ['open', 'high', 'low', 'close', 'volume'] and dtype not in ['float64', 'float32']:
                    recommendations.append(f'Convert {col} to numeric type for calculations')
            if len(data) < 1000:
                recommendations.append('Small dataset - consider collecting more data')
            if len(data.columns) > 100:
                recommendations.append('High-dimensional data - consider feature selection')
            return recommendations
        except Exception as e:
            self.logger.exception(f'Error generating recommendations: {e}')
            return ['Error generating recommendations']

    def format_data(self, data: pd.DataFrame, data_type: str='klines') -> pd.DataFrame:
        """Format data according to standardized formats.

        Args:
            data: Data to format
            data_type: Type of data (klines, features, etc.)

        Returns:
            Formatted data
        """
        formatted_data = data.copy()
        if data_type == 'klines':
            formatted_data = self._format_klines_data(formatted_data)
        elif data_type == 'features':
            formatted_data = self._format_features_data(formatted_data)
        elif data_type == 'labels':
            formatted_data = self._format_labels_data(formatted_data)
        else:
            self.logger.warning(f'Unknown data type for formatting: {data_type}')
        return formatted_data

    def _format_klines_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Format klines data."""
        formatted = data.copy()
        if 'timestamp' in formatted.columns:
            formatted['timestamp'] = pd.to_numeric(formatted['timestamp'], errors='coerce').astype('int64')
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_columns:
            if col in formatted.columns:
                formatted[col] = pd.to_numeric(formatted[col], errors='coerce').astype('float64')
        if 'timestamp' in formatted.columns:
            formatted = formatted.sort_values('timestamp').reset_index(drop=True)
        return formatted

    def _format_features_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Format features data."""
        formatted = data.copy()
        numeric_columns = formatted.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            formatted[col] = pd.to_numeric(formatted[col], errors='coerce').astype('float64')
        return formatted.replace([np.inf, -np.inf], np.nan)

    def _format_labels_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Format labels data."""
        formatted = data.copy()
        label_columns = [col for col in formatted.columns if 'label' in col.lower()]
        for col in label_columns:
            formatted[col] = pd.to_numeric(formatted[col], errors='coerce').astype('int64')
        return formatted

    def profile_data(self, data: pd.DataFrame) -> dict[str, Any]:
        """Generate comprehensive data profile.

        Args:
            data: Data to profile

        Returns:
            Data profile
        """
        if not self.quality_policies['profiling_enabled']:
            return {'profiling_disabled': True}
        profile = {'timestamp': datetime.now().isoformat(), 'data_shape': data.shape, 'memory_usage': data.memory_usage(deep=True).sum(), 'columns': {}, 'summary': {'total_rows': len(data), 'total_columns': len(data.columns), 'missing_values': data.isnull().sum().sum(), 'duplicate_rows': data.duplicated().sum(), 'numeric_columns': len(data.select_dtypes(include=[np.number]).columns), 'categorical_columns': len(data.select_dtypes(include=['object']).columns), 'datetime_columns': len(data.select_dtypes(include=['datetime']).columns)}}
        for column in data.columns:
            col_data = data[column]
            col_profile = {'dtype': str(col_data.dtype), 'missing_count': col_data.isnull().sum(), 'missing_ratio': col_data.isnull().sum() / len(col_data), 'unique_count': col_data.nunique(), 'unique_ratio': col_data.nunique() / len(col_data)}
            if pd.api.types.is_numeric_dtype(col_data):
                col_profile.update({'min': float(col_data.min()) if not col_data.isna().all() else None, 'max': float(col_data.max()) if not col_data.isna().all() else None, 'mean': float(col_data.mean()) if not col_data.isna().all() else None, 'median': float(col_data.median()) if not col_data.isna().all() else None, 'std': float(col_data.std()) if not col_data.isna().all() else None, 'zero_count': (col_data == 0).sum(), 'negative_count': (col_data < 0).sum(), 'infinite_count': np.isinf(col_data).sum()})
            elif pd.api.types.is_object_dtype(col_data):
                value_counts = col_data.value_counts()
                col_profile.update({'top_values': value_counts.head(5).to_dict(), 'empty_string_count': (col_data == '').sum(), 'whitespace_only_count': col_data.astype(str).str.strip().eq('').sum()})
            profile['columns'][column] = col_profile
        return profile

    def get_quality_report(self, data: pd.DataFrame, include_profile: bool=True) -> dict[str, Any]:
        """Generate comprehensive data quality report.

        Args:
            data: Data to analyze
            include_profile: Whether to include data profiling

        Returns:
            Quality report
        """
        report = {'timestamp': datetime.now().isoformat(), 'data_shape': data.shape, 'validation_results': self.validate_data(data), 'quality_score': self.calculate_quality_score(data)}
        if include_profile:
            report['data_profile'] = self.profile_data(data)
        report['quality_metrics'] = {'completeness': self._calculate_completeness_score(data), 'consistency': self._calculate_consistency_score(data), 'accuracy': self._calculate_accuracy_score(data), 'timeliness': self._calculate_timeliness_score(data)}
        return report

    def calculate_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate overall data quality score.

        Args:
            data: Data to score

        Returns:
            Quality score between 0 and 1
        """
        scores = []
        completeness = 1 - data.isnull().sum().sum() / (len(data) * len(data.columns))
        scores.append(completeness)
        consistency = 1 - data.duplicated().sum() / len(data)
        scores.append(consistency)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            infinite_ratio = np.isinf(data[numeric_cols]).sum().sum() / (len(data) * len(numeric_cols))
            validity = 1 - infinite_ratio
        else:
            validity = 1.0
        scores.append(validity)
        range_scores = []
        for col in numeric_cols:
            if col in ['open', 'high', 'low', 'close', 'volume']:
                negative_ratio = (data[col] < 0).sum() / len(data)
                range_scores.append(1 - negative_ratio)
        if range_scores:
            scores.append(np.mean(range_scores))
        return np.mean(scores)

    def _calculate_completeness_score(self, data: pd.DataFrame) -> float:
        """Calculate completeness score."""
        return 1 - data.isnull().sum().sum() / (len(data) * len(data.columns))

    def _calculate_consistency_score(self, data: pd.DataFrame) -> float:
        """Calculate consistency score."""
        return 1 - data.duplicated().sum() / len(data)

    def _calculate_accuracy_score(self, data: pd.DataFrame) -> float:
        """Calculate accuracy score."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return 1.0
        accuracy_scores = []
        for col in numeric_cols:
            if col in ['open', 'high', 'low', 'close']:
                if all((c in data.columns for c in ['open', 'high', 'low', 'close'])):
                    ohlc_valid = ((data['high'] >= data['low']) & (data['high'] >= data['open']) & (data['high'] >= data['close']) & (data['low'] <= data['open']) & (data['low'] <= data['close'])).mean()
                    accuracy_scores.append(ohlc_valid)
        return np.mean(accuracy_scores) if accuracy_scores else 1.0

    def _calculate_timeliness_score(self, data: pd.DataFrame) -> float:
        """Calculate timeliness score."""
        if 'timestamp' not in data.columns:
            return 1.0
        try:
            timestamps = pd.to_datetime(data['timestamp'], unit='s')
            now = pd.Timestamp.now()
            time_diff = abs((timestamps - now).dt.total_seconds())
            return 1 - min(time_diff.mean() / (365 * 24 * 3600), 1.0)
        except:
            return 0.5
data_quality_framework = DataQualityFramework()