"""
Enhanced Data Quality Validator with M1 Optimizations.

This module provides comprehensive data quality validation with M1/M2/M3 optimizations,
GPU acceleration, memory optimization, and CPU parallel processing for high-performance
data quality assessment.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

# M1 Optimizations
from .m1_gpu_utils import M1GPUManager
from .m1_memory_optimizer import M1MemoryOptimizer
from .m1_cpu_optimizer import M1CPUOptimizer
from .data_processing_utils import DataFrameValidator, DataFrameCleaner
from .ml_common.matrix_operations import get_enhanced_matrix_operations
from .math_validation import (
    validate_finite, safe_divide, safe_log, safe_sqrt, 
    MathValidationError
)

try:
    from .logger import system_logger
except ImportError:
    system_logger = logging.getLogger('EnhancedDataQualityValidator')

warnings.filterwarnings('ignore')

@dataclass
class QualityResult:
    """Represents data quality validation results."""
    passed: bool
    quality_score: float
    issues: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]
    recommendations: List[str]

class EnhancedDataQualityValidator:
    """Enhanced data quality validator with M1 optimizations and comprehensive checks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced data quality validator with M1 optimizations."""
        self.config = config or {}
        self.logger = system_logger.getChild('EnhancedDataQualityValidator')
        
        # Initialize M1 optimizations
        self.gpu_manager = M1GPUManager(self.config.get('gpu_config', {}))
        self.memory_optimizer = M1MemoryOptimizer(
            memory_limit_gb=self.config.get('memory_limit_gb', 8.0),
            enable_gc_tuning=self.config.get('enable_gc_tuning', True)
        )
        self.cpu_optimizer = M1CPUOptimizer(
            max_workers=self.config.get('max_workers'),
            enable_hyperthreading=self.config.get('enable_hyperthreading', True)
        )
        
        # Initialize matrix operations
        self.matrix_ops = get_enhanced_matrix_operations()
        
        # Initialize data processing utilities
        self.dataframe_validator = DataFrameValidator()
        self.dataframe_cleaner = DataFrameCleaner()
        
        # Quality thresholds
        self.thresholds = self.config.get('thresholds', {
            'max_nan_ratio': 0.3,
            'max_infinite_count': 100,
            'min_unique_values': 2,
            'price_tolerance': 0.1,
            'volume_tolerance': 0.5,
            'correlation_threshold': 0.95,
            'outlier_contamination': 0.1
        })
        
        self.logger.info("✅ Enhanced Data Quality Validator initialized with M1 optimizations")

    def validate_dataframe_quality(self, df: pd.DataFrame, context: str = '') -> QualityResult:
        """Comprehensive dataframe quality validation with M1 optimizations."""
        self.logger.info(f'🔍 Validating dataframe quality for {context}...')
        
        issues = []
        warnings = []
        metrics = {}
        recommendations = []
        
        try:
            # Memory-optimized validation
            with self.memory_optimizer.memory_checkpoint('dataframe_validation'):
                # Basic structure validation
                structure_result = self._validate_dataframe_structure(df)
                issues.extend(structure_result.get('issues', []))
                warnings.extend(structure_result.get('warnings', []))
                metrics.update(structure_result.get('metrics', {}))
                
                # Data type validation
                dtype_result = self._validate_data_types(df)
                issues.extend(dtype_result.get('issues', []))
                warnings.extend(dtype_result.get('warnings', []))
                metrics.update(dtype_result.get('metrics', {}))
                
                # Value validation with GPU acceleration
                value_result = self._validate_data_values(df)
                issues.extend(value_result.get('issues', []))
                warnings.extend(value_result.get('warnings', []))
                metrics.update(value_result.get('metrics', {}))
                
                # Statistical validation
                stats_result = self._validate_statistical_properties(df)
                issues.extend(stats_result.get('issues', []))
                warnings.extend(stats_result.get('warnings', []))
                metrics.update(stats_result.get('metrics', {}))
                
                # Correlation analysis with parallel processing
                corr_result = self._validate_correlations(df)
                issues.extend(corr_result.get('issues', []))
                warnings.extend(corr_result.get('warnings', []))
                metrics.update(corr_result.get('metrics', {}))
                
                # Outlier detection with GPU acceleration
                outlier_result = self._detect_outliers(df)
                issues.extend(outlier_result.get('issues', []))
                warnings.extend(outlier_result.get('warnings', []))
                metrics.update(outlier_result.get('metrics', {}))
                
                # Generate recommendations
                recommendations = self._generate_recommendations(issues, warnings, metrics)
                
                # Calculate overall quality score
                quality_score = self._calculate_quality_score(issues, warnings, metrics)
                
                passed = len(issues) == 0 and quality_score >= 0.7
                
                result = QualityResult(
                    passed=passed,
                    quality_score=quality_score,
                    issues=issues,
                    warnings=warnings,
                    metrics=metrics,
                    recommendations=recommendations
                )
                
                self.logger.info(f'✅ Dataframe quality validation completed: Score {quality_score:.3f}')
                return result
                
        except Exception as e:
            self.logger.error(f'❌ Dataframe quality validation failed: {e}')
            return QualityResult(
                passed=False,
                quality_score=0.0,
                issues=[f'Validation failed: {str(e)}'],
                warnings=[],
                metrics={},
                recommendations=['Fix validation errors and retry']
            )

    def _validate_dataframe_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataframe structure with comprehensive data type and row checks."""
        result = {'issues': [], 'warnings': [], 'metrics': {}}
        
        # Basic structure checks
        if df.empty:
            result['issues'].append('DataFrame is empty')
            return result
        
        if len(df.columns) == 0:
            result['issues'].append('DataFrame has no columns')
            return result
        
        # Memory-optimized size analysis
        memory_usage = df.memory_usage(deep=True).sum()
        result['metrics']['memory_usage_mb'] = memory_usage / 1024 / 1024
        result['metrics']['row_count'] = len(df)
        result['metrics']['column_count'] = len(df.columns)
        
        # Check for duplicate columns
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            result['warnings'].append(f'Duplicate columns found: {duplicate_cols}')
        
        # Check for completely empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            result['warnings'].append(f'Completely empty columns: {empty_cols}')
        
        # Check for completely empty rows
        empty_rows = df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            result['warnings'].append(f'Found {empty_rows} completely empty rows')
            result['metrics']['empty_rows_count'] = empty_rows
        
        # Comprehensive data type validation
        dtype_issues = []
        dtype_warnings = []
        dtype_metrics = {}
        
        # Analyze each column's data type
        for col in df.columns:
            col_series = df[col]
            col_dtype = str(col_series.dtype)
            
            # Check for object dtype (potential mixed types)
            if col_dtype == 'object':
                # Check if it's actually numeric data stored as object
                try:
                    numeric_conversion = pd.to_numeric(col_series, errors='coerce')
                    non_numeric_count = numeric_conversion.isnull().sum() - col_series.isnull().sum()
                    if non_numeric_count > 0:
                        dtype_warnings.append(f"Column '{col}' (object) contains {non_numeric_count} non-numeric values")
                    else:
                        dtype_warnings.append(f"Column '{col}' is numeric data stored as object - consider converting to numeric")
                except Exception:
                    dtype_warnings.append(f"Column '{col}' (object) may contain mixed data types")
            
            # Check for datetime columns
            elif 'datetime' in col_dtype:
                # Validate datetime format consistency
                try:
                    if col_series.notna().any():
                        # Check for invalid datetime values
                        invalid_dates = col_series.isnull().sum() - col_series.isna().sum()
                        if invalid_dates > 0:
                            dtype_issues.append(f"Column '{col}' contains {invalid_dates} invalid datetime values")
                except Exception:
                    dtype_warnings.append(f"Column '{col}' datetime validation failed")
            
            # Check for numeric columns
            elif pd.api.types.is_numeric_dtype(col_series):
                # Check for infinite values
                if col_series.dtype in ['float64', 'float32']:
                    inf_count = np.isinf(col_series).sum()
                    if inf_count > 0:
                        dtype_issues.append(f"Column '{col}' contains {inf_count} infinite values")
                
                # Check for negative values in columns that shouldn't have them
                if any(keyword in col.lower() for keyword in ['price', 'volume', 'amount', 'size', 'count']):
                    negative_count = (col_series < 0).sum()
                    if negative_count > 0:
                        dtype_issues.append(f"Column '{col}' contains {negative_count} negative values")
            
            # Check for boolean columns
            elif col_dtype == 'bool':
                # Check for mixed boolean types
                unique_values = col_series.dropna().unique()
                if len(unique_values) > 2:
                    dtype_warnings.append(f"Column '{col}' (bool) contains more than 2 unique values: {unique_values}")
        
        # Check for columns with all NaN values
        all_nan_cols = df.columns[df.isnull().all()].tolist()
        if all_nan_cols:
            dtype_issues.append(f'Columns with all NaN values: {all_nan_cols}')
        
        # Check for columns with only one unique value (excluding NaN)
        constant_cols = []
        for col in df.columns:
            unique_count = df[col].nunique(dropna=True)
            if unique_count <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            dtype_warnings.append(f'Columns with constant values: {constant_cols}')
        
        # Check for columns with very low variance (for numeric columns)
        low_variance_cols = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].notna().sum() > 1:  # Need at least 2 non-null values
                variance = df[col].var()
                if variance < 1e-10:  # Very low variance threshold
                    low_variance_cols.append(col)
        
        if low_variance_cols:
            dtype_warnings.append(f'Columns with very low variance: {low_variance_cols}')
        
        # Check for potential data type inconsistencies
        inconsistent_dtype_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if all non-null values can be converted to the same type
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    # Try to infer consistent type
                    try:
                        # Try numeric conversion
                        pd.to_numeric(non_null_values, errors='raise')
                        inconsistent_dtype_cols.append(f"{col} (should be numeric)")
                    except (ValueError, TypeError):
                        try:
                            # Try datetime conversion
                            pd.to_datetime(non_null_values, errors='raise')
                            inconsistent_dtype_cols.append(f"{col} (should be datetime)")
                        except (ValueError, TypeError):
                            # Check if it's boolean-like
                            unique_vals = set(str(v).lower() for v in non_null_values.unique())
                            if unique_vals.issubset({'true', 'false', '1', '0', 'yes', 'no'}):
                                inconsistent_dtype_cols.append(f"{col} (should be boolean)")
        
        if inconsistent_dtype_cols:
            dtype_warnings.append(f'Columns with inconsistent data types: {inconsistent_dtype_cols}')
        
        # Compile results
        result['issues'].extend(dtype_issues)
        result['warnings'].extend(dtype_warnings)
        result['metrics'].update({
            'dtype_analysis': {
                'total_columns': len(df.columns),
                'numeric_columns': len(numeric_cols),
                'object_columns': len(df.select_dtypes(include=['object']).columns),
                'datetime_columns': len(df.select_dtypes(include=['datetime']).columns),
                'boolean_columns': len(df.select_dtypes(include=['bool']).columns),
                'constant_columns': constant_cols,
                'low_variance_columns': low_variance_cols,
                'inconsistent_dtype_columns': inconsistent_dtype_cols
            }
        })
        
        return result

    def _validate_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data types with CPU optimization - focused on null value analysis."""
        result = {'issues': [], 'warnings': [], 'metrics': {}}
        
        # Parallel data type analysis focused on null values and basic type info
        def analyze_column_dtype(col):
            return {
                'name': col.name,
                'dtype': str(col.dtype),
                'is_numeric': pd.api.types.is_numeric_dtype(col),
                'is_datetime': pd.api.types.is_datetime64_any_dtype(col),
                'null_count': col.isnull().sum(),
                'null_ratio': col.isnull().sum() / len(col),
                'unique_count': col.nunique(dropna=True),
                'memory_usage': col.memory_usage(deep=True)
            }
        
        # Use CPU optimizer for parallel processing
        dtype_analysis = self.cpu_optimizer.parallel_apply(
            list(df.columns), analyze_column_dtype
        )
        
        result['metrics']['detailed_dtype_analysis'] = dtype_analysis
        
        # Check for high null ratios
        high_null_cols = []
        for analysis in dtype_analysis:
            if analysis['null_ratio'] > self.thresholds['max_nan_ratio']:
                result['warnings'].append(f"Column '{analysis['name']}' has {analysis['null_ratio']:.1%} null values")
                high_null_cols.append(analysis['name'])
        
        # Check for memory usage optimization opportunities
        memory_optimization_warnings = []
        for analysis in dtype_analysis:
            col_name = analysis['name']
            col_dtype = analysis['dtype']
            memory_usage = analysis['memory_usage']
            
            # Check for potential memory optimization
            if col_dtype == 'object' and analysis['is_numeric']:
                memory_optimization_warnings.append(f"Column '{col_name}' could be converted to numeric to save memory")
            elif col_dtype == 'float64' and analysis['unique_count'] < 1000:
                memory_optimization_warnings.append(f"Column '{col_name}' could use float32 to save memory")
            elif col_dtype == 'int64' and analysis['unique_count'] < 2**31:
                memory_optimization_warnings.append(f"Column '{col_name}' could use int32 to save memory")
        
        if memory_optimization_warnings:
            result['warnings'].extend(memory_optimization_warnings)
        
        # Check for columns with very few unique values (potential categorical)
        categorical_candidates = []
        for analysis in dtype_analysis:
            if analysis['unique_count'] < 20 and analysis['unique_count'] > 1:
                categorical_candidates.append(analysis['name'])
        
        if categorical_candidates:
            result['warnings'].append(f'Columns that might benefit from categorical dtype: {categorical_candidates}')
        
        return result

    def _validate_data_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data values with GPU acceleration."""
        result = {'issues': [], 'warnings': [], 'metrics': {}}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            result['warnings'].append('No numeric columns found for value validation')
            return result
        
        # GPU-accelerated value validation
        with self.gpu_manager.get_device_context() as device:
            for col in numeric_cols:
                try:
                    # Convert to tensor for GPU processing
                    values = df[col].dropna().values
                    if len(values) == 0:
                        continue
                    
                    # GPU-accelerated finite value check
                    if self.gpu_manager.is_available():
                        values_tensor = self.gpu_manager.to_tensor(values)
                        finite_mask = self.gpu_manager.is_finite(values_tensor)
                        finite_count = finite_mask.sum().item()
                    else:
                        finite_count = np.isfinite(values).sum()
                    
                    total_count = len(values)
                    finite_ratio = finite_count / total_count if total_count > 0 else 0
                    
                    if finite_ratio < 0.9:
                        result['warnings'].append(f"Column '{col}' has {1-finite_ratio:.1%} non-finite values")
                    
                    # Check for extreme values
                    if finite_count > 0:
                        finite_values = values[np.isfinite(values)]
                        if len(finite_values) > 0:
                            q1, q3 = np.percentile(finite_values, [25, 75])
                            iqr = q3 - q1
                            lower_bound = q1 - 3 * iqr
                            upper_bound = q3 + 3 * iqr
                            
                            outliers = ((finite_values < lower_bound) | (finite_values > upper_bound)).sum()
                            outlier_ratio = outliers / len(finite_values)
                            
                            if outlier_ratio > 0.1:
                                result['warnings'].append(f"Column '{col}' has {outlier_ratio:.1%} extreme outliers")
                
                except Exception as e:
                    result['warnings'].append(f"Error validating column '{col}': {str(e)}")
        
        return result

    def _validate_statistical_properties(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate statistical properties with matrix operations."""
        result = {'issues': [], 'warnings': [], 'metrics': {}}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return result
        
        # Use enhanced matrix operations for statistical analysis
        try:
            numeric_data = df[numeric_cols].dropna()
            
            if len(numeric_data) == 0:
                result['warnings'].append('No complete numeric data for statistical analysis')
                return result
            
            # Matrix-based statistical analysis
            mean_values = numeric_data.mean()
            std_values = numeric_data.std()
            
            # Check for constant features
            constant_features = std_values[std_values == 0].index.tolist()
            if constant_features:
                result['warnings'].append(f'Constant features detected: {constant_features}')
            
            # Check for low variance features
            low_variance_threshold = 0.01
            low_variance_features = std_values[std_values < low_variance_threshold].index.tolist()
            if low_variance_features:
                result['warnings'].append(f'Low variance features detected: {low_variance_features}')
            
            result['metrics']['statistical_summary'] = {
                'mean_values': mean_values.to_dict(),
                'std_values': std_values.to_dict(),
                'constant_features': constant_features,
                'low_variance_features': low_variance_features
            }
            
        except Exception as e:
            result['warnings'].append(f'Statistical analysis failed: {str(e)}')
        
        return result

    def _validate_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate correlations with parallel processing."""
        result = {'issues': [], 'warnings': [], 'metrics': {}}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return result
        
        try:
            # Use CPU optimizer for parallel correlation analysis
            numeric_data = df[numeric_cols].dropna()
            
            if len(numeric_data) == 0:
                return result
            
            # Parallel correlation matrix calculation
            corr_matrix = self.cpu_optimizer.parallel_apply(
                [numeric_data], lambda x: x.corr()
            )[0]
            
            # Find highly correlated pairs
            highly_correlated_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > self.thresholds['correlation_threshold']:
                        highly_correlated_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            if highly_correlated_pairs:
                result['warnings'].append(f'Found {len(highly_correlated_pairs)} highly correlated feature pairs')
            
            result['metrics']['correlation_analysis'] = {
                'highly_correlated_pairs': highly_correlated_pairs[:10],  # Limit to first 10
                'correlation_matrix_shape': corr_matrix.shape
            }
            
        except Exception as e:
            result['warnings'].append(f'Correlation analysis failed: {str(e)}')
        
        return result

    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers with GPU acceleration."""
        result = {'issues': [], 'warnings': [], 'metrics': {}}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return result
        
        try:
            numeric_data = df[numeric_cols].dropna()
            
            if len(numeric_data) < 10:  # Need sufficient data for outlier detection
                return result
            
            # GPU-accelerated outlier detection
            with self.gpu_manager.get_device_context() as device:
                if self.gpu_manager.is_available():
                    # Use GPU for Isolation Forest
                    data_tensor = self.gpu_manager.to_tensor(numeric_data.values)
                    
                    # Standardize data
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(numeric_data.values)
                    
                    # Isolation Forest for outlier detection
                    iso_forest = IsolationForest(
                        contamination=self.thresholds['outlier_contamination'],
                        random_state=42
                    )
                    outlier_labels = iso_forest.fit_predict(scaled_data)
                    
                    outlier_count = (outlier_labels == -1).sum()
                    outlier_ratio = outlier_count / len(outlier_labels)
                    
                    if outlier_ratio > 0.2:
                        result['warnings'].append(f'High outlier ratio detected: {outlier_ratio:.1%}')
                    
                    result['metrics']['outlier_analysis'] = {
                        'outlier_count': int(outlier_count),
                        'outlier_ratio': float(outlier_ratio),
                        'total_samples': len(outlier_labels)
                    }
                else:
                    # Fallback to CPU-based outlier detection
                    result['warnings'].append('GPU not available, using CPU-based outlier detection')
        
        except Exception as e:
            result['warnings'].append(f'Outlier detection failed: {str(e)}')
        
        return result

    def _generate_recommendations(self, issues: List[str], warnings: List[str], metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if issues:
            recommendations.append('Address critical issues before proceeding with analysis')
        
        if warnings:
            recommendations.append('Review warnings and consider data cleaning')
        
        # Memory usage recommendations
        memory_usage = metrics.get('memory_usage_mb', 0)
        if memory_usage > 1000:  # > 1GB
            recommendations.append('Consider data sampling or chunked processing for large datasets')
        
        # Correlation recommendations
        corr_analysis = metrics.get('correlation_analysis', {})
        if corr_analysis.get('highly_correlated_pairs'):
            recommendations.append('Consider feature selection to remove highly correlated features')
        
        # Outlier recommendations
        outlier_analysis = metrics.get('outlier_analysis', {})
        if outlier_analysis.get('outlier_ratio', 0) > 0.1:
            recommendations.append('Consider outlier treatment or robust statistical methods')
        
        return recommendations

    def _calculate_quality_score(self, issues: List[str], warnings: List[str], metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score."""
        base_score = 1.0
        
        # Deduct for issues
        base_score -= len(issues) * 0.2
        
        # Deduct for warnings
        base_score -= len(warnings) * 0.05
        
        # Deduct for high memory usage
        memory_usage = metrics.get('memory_usage_mb', 0)
        if memory_usage > 1000:
            base_score -= 0.1
        
        # Deduct for high outlier ratio
        outlier_analysis = metrics.get('outlier_analysis', {})
        outlier_ratio = outlier_analysis.get('outlier_ratio', 0)
        if outlier_ratio > 0.2:
            base_score -= 0.2
        
        return max(0.0, min(1.0, base_score))

    def automated_data_cleaning(self, df: pd.DataFrame, cleaning_config: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Automated data cleaning with M1 optimizations."""
        self.logger.info('🧹 Starting automated data cleaning...')
        
        cleaning_config = cleaning_config or {}
        cleaning_report = {'operations': [], 'rows_removed': 0, 'columns_modified': 0}
        
        try:
            with self.memory_optimizer.memory_checkpoint('data_cleaning'):
                cleaned_df = df.copy()
                original_rows = len(cleaned_df)
                
                # Remove completely empty columns
                if cleaning_config.get('remove_empty_columns', True):
                    empty_cols = cleaned_df.columns[cleaned_df.isnull().all()].tolist()
                    if empty_cols:
                        cleaned_df = cleaned_df.drop(columns=empty_cols)
                        cleaning_report['operations'].append(f'Removed {len(empty_cols)} empty columns')
                        cleaning_report['columns_modified'] += len(empty_cols)
                
                # Remove rows with all NaN values
                if cleaning_config.get('remove_all_nan_rows', True):
                    all_nan_rows = cleaned_df.isnull().all(axis=1).sum()
                    if all_nan_rows > 0:
                        cleaned_df = cleaned_df.dropna(how='all')
                        cleaning_report['operations'].append(f'Removed {all_nan_rows} rows with all NaN values')
                        cleaning_report['rows_removed'] += all_nan_rows
                
                # Handle infinite values
                if cleaning_config.get('handle_infinite_values', True):
                    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        inf_count = np.isinf(cleaned_df[col]).sum()
                        if inf_count > 0:
                            # Replace infinite values with NaN
                            cleaned_df[col] = cleaned_df[col].replace([np.inf, -np.inf], np.nan)
                            cleaning_report['operations'].append(f'Replaced {inf_count} infinite values in {col}')
                
                # Remove highly correlated features
                if cleaning_config.get('remove_correlated_features', False):
                    correlation_threshold = cleaning_config.get('correlation_threshold', 0.95)
                    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                    
                    if len(numeric_cols) > 1:
                        corr_matrix = cleaned_df[numeric_cols].corr()
                        high_corr_pairs = []
                        
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i + 1, len(corr_matrix.columns)):
                                if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                                    high_corr_pairs.append(corr_matrix.columns[j])
                        
                        if high_corr_pairs:
                            cleaned_df = cleaned_df.drop(columns=high_corr_pairs)
                            cleaning_report['operations'].append(f'Removed {len(high_corr_pairs)} highly correlated features')
                            cleaning_report['columns_modified'] += len(high_corr_pairs)
                
                final_rows = len(cleaned_df)
                cleaning_report['rows_removed'] += (original_rows - final_rows)
                cleaning_report['final_rows'] = final_rows
                cleaning_report['final_columns'] = len(cleaned_df.columns)
                
                self.logger.info(f'✅ Data cleaning completed: {cleaning_report["rows_removed"]} rows removed, {cleaning_report["columns_modified"]} columns modified')
                
                return cleaned_df, cleaning_report
                
        except Exception as e:
            self.logger.error(f'❌ Data cleaning failed: {e}')
            return df, {'error': str(e), 'operations': []}

    def missing_value_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive missing value analysis."""
        result = {
            'missing_counts': df.isnull().sum().to_dict(),
            'missing_ratios': (df.isnull().sum() / len(df)).to_dict(),
            'severity_assessment': {}
        }
        
        # Assess severity
        high_missing_cols = [col for col, ratio in result['missing_ratios'].items() if ratio > 0.5]
        medium_missing_cols = [col for col, ratio in result['missing_ratios'].items() if 0.1 < ratio <= 0.5]
        
        if high_missing_cols:
            result['severity_assessment']['severity_level'] = 'high'
            result['severity_assessment']['high_missing_columns'] = high_missing_cols
        elif medium_missing_cols:
            result['severity_assessment']['severity_level'] = 'medium'
            result['severity_assessment']['medium_missing_columns'] = medium_missing_cols
        else:
            result['severity_assessment']['severity_level'] = 'low'
        
        return result

    def automated_outlier_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Automated outlier detection with GPU acceleration."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {'outliers_detected': 0, 'recommendations': []}
        
        try:
            numeric_data = df[numeric_cols].dropna()
            
            if len(numeric_data) < 10:
                return {'outliers_detected': 0, 'recommendations': []}
            
            # Use Isolation Forest for outlier detection
            iso_forest = IsolationForest(
                contamination=self.thresholds['outlier_contamination'],
                random_state=42
            )
            outlier_labels = iso_forest.fit_predict(numeric_data)
            
            outlier_count = (outlier_labels == -1).sum()
            outlier_ratio = outlier_count / len(outlier_labels)
            
            return {
                'outliers_detected': int(outlier_count),
                'outlier_ratio': float(outlier_ratio),
                'recommendations': [
                    'Consider outlier treatment if ratio > 10%',
                    'Review outlier detection method if ratio > 20%'
                ]
            }
            
        except Exception as e:
            return {'error': str(e), 'outliers_detected': 0, 'recommendations': []}

    def feature_correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Feature correlation analysis with parallel processing."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'correlation_matrix': None, 'highly_correlated_pairs': []}
        
        try:
            numeric_data = df[numeric_cols].dropna()
            corr_matrix = numeric_data.corr()
            
            # Find highly correlated pairs
            highly_correlated_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > self.thresholds['correlation_threshold']:
                        highly_correlated_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': float(corr_value)
                        })
            
            return {
                'correlation_matrix': corr_matrix.to_dict(),
                'highly_correlated_pairs': highly_correlated_pairs,
                'correlation_threshold': self.thresholds['correlation_threshold']
            }
            
        except Exception as e:
            return {'error': str(e), 'correlation_matrix': None, 'highly_correlated_pairs': []}


class UnifiedDataQualityValidator(EnhancedDataQualityValidator):
    """Specialized validator for unified data format with enhanced optimizations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.logger = system_logger.getChild('UnifiedDataQualityValidator')
        
        # Unified data specific thresholds
        self.unified_thresholds = {
            'max_timestamp_gap_seconds': 3600,  # 1 hour
            'min_price_value': 0.001,
            'max_price_change_ratio': 0.5,
            'min_volume_value': 0,
            'max_volume_ratio': 10.0
        }
        
        self.logger.info("✅ Unified Data Quality Validator initialized")

    def validate_unified_data_quality(self, df: pd.DataFrame) -> QualityResult:
        """Validate unified data format with specialized checks."""
        self.logger.info('🔍 Validating unified data quality...')
        
        # Start with general validation
        general_result = self.validate_dataframe_quality(df, 'unified_data')
        
        # Add unified-specific validations
        unified_issues = []
        unified_warnings = []
        
        # Check for required unified columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            unified_issues.append(f'Missing required unified columns: {missing_columns}')
        
        # Validate timestamp continuity
        if 'timestamp' in df.columns:
            timestamp_issues = self._validate_timestamp_continuity(df['timestamp'])
            unified_issues.extend(timestamp_issues)
        
        # Validate OHLC relationships
        ohlc_issues = self._validate_ohlc_relationships(df)
        unified_issues.extend(ohlc_issues)
        
        # Validate price and volume ranges
        price_volume_issues = self._validate_price_volume_ranges(df)
        unified_issues.extend(price_volume_issues)
        
        # Combine results
        combined_issues = general_result.issues + unified_issues
        combined_warnings = general_result.warnings + unified_warnings
        
        # Recalculate quality score
        quality_score = self._calculate_quality_score(combined_issues, combined_warnings, general_result.metrics)
        
        return QualityResult(
            passed=len(combined_issues) == 0 and quality_score >= 0.7,
            quality_score=quality_score,
            issues=combined_issues,
            warnings=combined_warnings,
            metrics=general_result.metrics,
            recommendations=general_result.recommendations
        )

    def _validate_timestamp_continuity(self, timestamp_series: pd.Series) -> List[str]:
        """Validate timestamp continuity."""
        issues = []
        
        try:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(timestamp_series):
                timestamp_series = pd.to_datetime(timestamp_series)
            
            # Check for duplicate timestamps
            duplicates = timestamp_series.duplicated().sum()
            if duplicates > 0:
                issues.append(f'Found {duplicates} duplicate timestamps')
            
            # Check for large gaps
            sorted_timestamps = timestamp_series.sort_values()
            time_diffs = sorted_timestamps.diff().dt.total_seconds()
            large_gaps = time_diffs[time_diffs > self.unified_thresholds['max_timestamp_gap_seconds']]
            
            if len(large_gaps) > 0:
                issues.append(f'Found {len(large_gaps)} large timestamp gaps (>1 hour)')
            
        except Exception as e:
            issues.append(f'Timestamp validation failed: {str(e)}')
        
        return issues

    def _validate_ohlc_relationships(self, df: pd.DataFrame) -> List[str]:
        """Validate OHLC relationships."""
        issues = []
        
        ohlc_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in ohlc_cols):
            return issues
        
        try:
            # Check OHLC relationships
            invalid_high = (df['high'] < df[['open', 'close']].max(axis=1)).sum()
            invalid_low = (df['low'] > df[['open', 'close']].min(axis=1)).sum()
            
            if invalid_high > 0:
                issues.append(f'Found {invalid_high} rows where high < max(open, close)')
            
            if invalid_low > 0:
                issues.append(f'Found {invalid_low} rows where low > min(open, close)')
            
        except Exception as e:
            issues.append(f'OHLC validation failed: {str(e)}')
        
        return issues

    def _validate_price_volume_ranges(self, df: pd.DataFrame) -> List[str]:
        """Validate price and volume ranges."""
        issues = []
        
        try:
            # Validate price ranges
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in df.columns:
                    negative_prices = (df[col] <= 0).sum()
                    if negative_prices > 0:
                        issues.append(f'Found {negative_prices} non-positive prices in {col}')
            
            # Validate volume ranges
            if 'volume' in df.columns:
                negative_volumes = (df['volume'] < 0).sum()
                if negative_volumes > 0:
                    issues.append(f'Found {negative_volumes} negative volumes')
            
        except Exception as e:
            issues.append(f'Price/volume validation failed: {str(e)}')
        
        return issues


def quick_validate_dataframe(df: pd.DataFrame, context: str = '') -> QualityResult:
    """Quick validation of DataFrame quality."""
    validator = EnhancedDataQualityValidator()
    return validator.validate_dataframe_quality(df, context)