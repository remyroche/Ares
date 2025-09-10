"""
Feature Engineering Validation with M1 Optimizations.

This module provides comprehensive feature engineering validation with M1/M2/M3 optimizations,
GPU acceleration, memory optimization, and CPU parallel processing for high-performance
feature validation and quality assessment.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
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
    system_logger = logging.getLogger('FeatureEngineeringValidation')

warnings.filterwarnings('ignore')

@dataclass
class FeatureValidationResult:
    """Represents feature engineering validation results."""
    passed: bool
    quality_score: float
    issues: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]
    recommendations: List[str]
    feature_analysis: Dict[str, Any]

class FeatureEngineeringValidator:
    """Feature engineering validator with M1 optimizations and comprehensive checks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize feature engineering validator with M1 optimizations."""
        self.config = config or {}
        self.logger = system_logger.getChild('FeatureEngineeringValidator')
        
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
        
        # Feature validation thresholds
        self.thresholds = self.config.get('thresholds', {
            'max_nan_ratio': 0.3,
            'max_infinite_ratio': 0.05,
            'min_variance': 1e-8,
            'max_correlation': 0.95,
            'min_mutual_info': 0.01,
            'max_skewness': 5.0,
            'max_kurtosis': 10.0,
            'outlier_contamination': 0.1
        })
        
        self.logger.info("✅ Feature Engineering Validator initialized with M1 optimizations")

    def validate_features(self, features: pd.DataFrame, target: Optional[pd.Series] = None, 
                         context: str = '') -> FeatureValidationResult:
        """Comprehensive feature validation with M1 optimizations."""
        self.logger.info(f'🔍 Validating features for {context}...')
        
        issues = []
        warnings = []
        metrics = {}
        recommendations = []
        feature_analysis = {}
        
        try:
            # Memory-optimized validation
            with self.memory_optimizer.memory_checkpoint('feature_validation'):
                # Basic feature structure validation
                structure_result = self._validate_feature_structure(features)
                issues.extend(structure_result.get('issues', []))
                warnings.extend(structure_result.get('warnings', []))
                metrics.update(structure_result.get('metrics', {}))
                
                # Feature quality validation
                quality_result = self._validate_feature_quality(features)
                issues.extend(quality_result.get('issues', []))
                warnings.extend(quality_result.get('warnings', []))
                metrics.update(quality_result.get('metrics', {}))
                
                # Feature correlation analysis
                correlation_result = self._validate_feature_correlations(features)
                issues.extend(correlation_result.get('issues', []))
                warnings.extend(correlation_result.get('warnings', []))
                metrics.update(correlation_result.get('metrics', {}))
                
                # Feature distribution analysis
                distribution_result = self._validate_feature_distributions(features)
                issues.extend(distribution_result.get('issues', []))
                warnings.extend(distribution_result.get('warnings', []))
                metrics.update(distribution_result.get('metrics', {}))
                
                # Feature importance analysis (if target provided)
                if target is not None:
                    importance_result = self._validate_feature_importance(features, target)
                    issues.extend(importance_result.get('issues', []))
                    warnings.extend(importance_result.get('warnings', []))
                    metrics.update(importance_result.get('metrics', {}))
                
                # Feature stability analysis
                stability_result = self._validate_feature_stability(features)
                issues.extend(stability_result.get('issues', []))
                warnings.extend(stability_result.get('warnings', []))
                metrics.update(stability_result.get('metrics', {}))
                
                # Generate feature analysis summary
                feature_analysis = self._generate_feature_analysis(features, metrics)
                
                # Generate recommendations
                recommendations = self._generate_feature_recommendations(issues, warnings, metrics, feature_analysis)
                
                # Calculate overall quality score
                quality_score = self._calculate_feature_quality_score(issues, warnings, metrics, feature_analysis)
                
                passed = len(issues) == 0 and quality_score >= 0.7
                
                result = FeatureValidationResult(
                    passed=passed,
                    quality_score=quality_score,
                    issues=issues,
                    warnings=warnings,
                    metrics=metrics,
                    recommendations=recommendations,
                    feature_analysis=feature_analysis
                )
                
                self.logger.info(f'✅ Feature validation completed: Score {quality_score:.3f}')
                return result
                
        except Exception as e:
            self.logger.error(f'❌ Feature validation failed: {e}')
            return FeatureValidationResult(
                passed=False,
                quality_score=0.0,
                issues=[f'Validation failed: {str(e)}'],
                warnings=[],
                metrics={},
                recommendations=['Fix validation errors and retry'],
                feature_analysis={}
            )

    def _validate_feature_structure(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Validate feature structure with comprehensive data type and row checks."""
        result = {'issues': [], 'warnings': [], 'metrics': {}}
        
        # Basic structure checks
        if features.empty:
            result['issues'].append('Feature DataFrame is empty')
            return result
        
        if len(features.columns) == 0:
            result['issues'].append('Feature DataFrame has no columns')
            return result
        
        # Memory-optimized size analysis
        memory_usage = features.memory_usage(deep=True).sum()
        result['metrics']['memory_usage_mb'] = memory_usage / 1024 / 1024
        result['metrics']['feature_count'] = len(features.columns)
        result['metrics']['sample_count'] = len(features)
        
        # Check for duplicate feature names
        duplicate_features = features.columns[features.columns.duplicated()].tolist()
        if duplicate_features:
            result['issues'].append(f'Duplicate feature names found: {duplicate_features}')
        
        # Check for completely empty rows
        empty_rows = features.isnull().all(axis=1).sum()
        if empty_rows > 0:
            result['warnings'].append(f'Found {empty_rows} completely empty rows in features')
            result['metrics']['empty_rows_count'] = empty_rows
        
        # Comprehensive feature data type validation
        dtype_issues = []
        dtype_warnings = []
        
        # Analyze each feature's data type
        for col in features.columns:
            col_series = features[col]
            col_dtype = str(col_series.dtype)
            
            # Check for object dtype (potential mixed types)
            if col_dtype == 'object':
                # Check if it's actually numeric data stored as object
                try:
                    numeric_conversion = pd.to_numeric(col_series, errors='coerce')
                    non_numeric_count = numeric_conversion.isnull().sum() - col_series.isnull().sum()
                    if non_numeric_count > 0:
                        dtype_warnings.append(f"Feature '{col}' (object) contains {non_numeric_count} non-numeric values")
                    else:
                        dtype_warnings.append(f"Feature '{col}' is numeric data stored as object - consider converting to numeric")
                except Exception:
                    dtype_warnings.append(f"Feature '{col}' (object) may contain mixed data types")
            
            # Check for datetime columns (unusual for features)
            elif 'datetime' in col_dtype:
                dtype_warnings.append(f"Feature '{col}' is datetime type - consider if this is appropriate for ML features")
            
            # Check for numeric columns
            elif pd.api.types.is_numeric_dtype(col_series):
                # Check for infinite values
                if col_series.dtype in ['float64', 'float32']:
                    inf_count = np.isinf(col_series).sum()
                    if inf_count > 0:
                        dtype_issues.append(f"Feature '{col}' contains {inf_count} infinite values")
                
                # Check for negative values in features that shouldn't have them
                if any(keyword in col.lower() for keyword in ['price', 'volume', 'amount', 'size', 'count', 'length', 'distance']):
                    negative_count = (col_series < 0).sum()
                    if negative_count > 0:
                        dtype_issues.append(f"Feature '{col}' contains {negative_count} negative values")
            
            # Check for boolean columns
            elif col_dtype == 'bool':
                # Check for mixed boolean types
                unique_values = col_series.dropna().unique()
                if len(unique_values) > 2:
                    dtype_warnings.append(f"Feature '{col}' (bool) contains more than 2 unique values: {unique_values}")
        
        # Check for features with all NaN values
        all_nan_features = features.columns[features.isnull().all()].tolist()
        if all_nan_features:
            dtype_issues.append(f'Features with all NaN values: {all_nan_features}')
        
        # Check for completely constant features
        constant_features = []
        for col in features.columns:
            unique_count = features[col].nunique(dropna=True)
            if unique_count <= 1:
                constant_features.append(col)
        
        if constant_features:
            result['warnings'].append(f'Constant features detected: {constant_features}')
        
        # Check for features with very low variance (for numeric features)
        low_variance_features = []
        numeric_features = features.select_dtypes(include=[np.number]).columns
        for col in numeric_features:
            if features[col].notna().sum() > 1:  # Need at least 2 non-null values
                variance = features[col].var()
                if variance < 1e-10:  # Very low variance threshold
                    low_variance_features.append(col)
        
        if low_variance_features:
            dtype_warnings.append(f'Features with very low variance: {low_variance_features}')
        
        # Check for potential data type inconsistencies
        inconsistent_dtype_features = []
        for col in features.columns:
            if features[col].dtype == 'object':
                # Check if all non-null values can be converted to the same type
                non_null_values = features[col].dropna()
                if len(non_null_values) > 0:
                    # Try to infer consistent type
                    try:
                        # Try numeric conversion
                        pd.to_numeric(non_null_values, errors='raise')
                        inconsistent_dtype_features.append(f"{col} (should be numeric)")
                    except (ValueError, TypeError):
                        try:
                            # Try datetime conversion
                            pd.to_datetime(non_null_values, errors='raise')
                            inconsistent_dtype_features.append(f"{col} (should be datetime)")
                        except (ValueError, TypeError):
                            # Check if it's boolean-like
                            unique_vals = set(str(v).lower() for v in non_null_values.unique())
                            if unique_vals.issubset({'true', 'false', '1', '0', 'yes', 'no'}):
                                inconsistent_dtype_features.append(f"{col} (should be boolean)")
        
        if inconsistent_dtype_features:
            dtype_warnings.append(f'Features with inconsistent data types: {inconsistent_dtype_features}')
        
        # Check for features with very few unique values (potential categorical)
        categorical_candidates = []
        for col in features.columns:
            unique_count = features[col].nunique(dropna=True)
            if 1 < unique_count < 20:  # Between 2 and 19 unique values
                categorical_candidates.append(col)
        
        if categorical_candidates:
            dtype_warnings.append(f'Features that might benefit from categorical dtype: {categorical_candidates}')
        
        # Compile results
        result['issues'].extend(dtype_issues)
        result['warnings'].extend(dtype_warnings)
        result['metrics'].update({
            'constant_features': constant_features,
            'low_variance_features': low_variance_features,
            'categorical_candidates': categorical_candidates,
            'dtype_analysis': {
                'total_features': len(features.columns),
                'numeric_features': len(numeric_features),
                'object_features': len(features.select_dtypes(include=['object']).columns),
                'datetime_features': len(features.select_dtypes(include=['datetime']).columns),
                'boolean_features': len(features.select_dtypes(include=['bool']).columns),
                'constant_features': constant_features,
                'low_variance_features': low_variance_features,
                'inconsistent_dtype_features': inconsistent_dtype_features
            }
        })
        
        return result

    def _validate_feature_quality(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Validate feature quality with GPU acceleration."""
        result = {'issues': [], 'warnings': [], 'metrics': {}}
        
        numeric_features = features.select_dtypes(include=[np.number]).columns
        
        if len(numeric_features) == 0:
            result['warnings'].append('No numeric features found for quality validation')
            return result
        
        # GPU-accelerated quality validation
        with self.gpu_manager.get_device_context() as device:
            quality_metrics = {}
            
            for col in numeric_features:
                try:
                    values = features[col].dropna().values
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
                    
                    # Check for excessive NaN values
                    nan_ratio = 1 - finite_ratio
                    if nan_ratio > self.thresholds['max_nan_ratio']:
                        result['warnings'].append(f"Feature '{col}' has {nan_ratio:.1%} NaN values")
                    
                    # Check for infinite values
                    if self.gpu_manager.is_available():
                        inf_mask = ~finite_mask
                        inf_count = inf_mask.sum().item()
                    else:
                        inf_count = np.isinf(values).sum()
                    
                    inf_ratio = inf_count / total_count if total_count > 0 else 0
                    if inf_ratio > self.thresholds['max_infinite_ratio']:
                        result['warnings'].append(f"Feature '{col}' has {inf_ratio:.1%} infinite values")
                    
                    # Check for zero variance
                    if finite_count > 1:
                        finite_values = values[np.isfinite(values)]
                        variance = np.var(finite_values)
                        if variance < self.thresholds['min_variance']:
                            result['warnings'].append(f"Feature '{col}' has very low variance: {variance:.2e}")
                    
                    quality_metrics[col] = {
                        'nan_ratio': nan_ratio,
                        'infinite_ratio': inf_ratio,
                        'variance': variance if finite_count > 1 else 0,
                        'finite_count': finite_count,
                        'total_count': total_count
                    }
                
                except Exception as e:
                    result['warnings'].append(f"Error validating feature '{col}': {str(e)}")
            
            result['metrics']['feature_quality'] = quality_metrics
        
        return result

    def _validate_feature_correlations(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Validate feature correlations with parallel processing."""
        result = {'issues': [], 'warnings': [], 'metrics': {}}
        
        numeric_features = features.select_dtypes(include=[np.number]).columns
        
        if len(numeric_features) < 2:
            return result
        
        try:
            # Use CPU optimizer for parallel correlation analysis
            numeric_data = features[numeric_features].dropna()
            
            if len(numeric_data) == 0:
                result['warnings'].append('No complete numeric data for correlation analysis')
                return result
            
            # Parallel correlation matrix calculation
            corr_matrix = self.cpu_optimizer.parallel_apply(
                [numeric_data], lambda x: x.corr()
            )[0]
            
            # Find highly correlated feature pairs
            highly_correlated_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > self.thresholds['max_correlation']:
                        highly_correlated_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            if highly_correlated_pairs:
                result['warnings'].append(f'Found {len(highly_correlated_pairs)} highly correlated feature pairs')
            
            # Check for perfect correlations
            perfect_correlations = [pair for pair in highly_correlated_pairs if abs(pair['correlation']) >= 0.99]
            if perfect_correlations:
                result['issues'].append(f'Found {len(perfect_correlations)} perfectly correlated feature pairs')
            
            result['metrics']['correlation_analysis'] = {
                'highly_correlated_pairs': highly_correlated_pairs[:10],  # Limit to first 10
                'perfect_correlations': perfect_correlations,
                'correlation_matrix_shape': corr_matrix.shape
            }
            
        except Exception as e:
            result['warnings'].append(f'Correlation analysis failed: {str(e)}')
        
        return result

    def _validate_feature_distributions(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Validate feature distributions with GPU acceleration."""
        result = {'issues': [], 'warnings': [], 'metrics': {}}
        
        numeric_features = features.select_dtypes(include=[np.number]).columns
        
        if len(numeric_features) == 0:
            return result
        
        distribution_metrics = {}
        
        for col in numeric_features:
            try:
                values = features[col].dropna()
                if len(values) < 10:  # Need sufficient data for distribution analysis
                    continue
                
                # Calculate distribution statistics
                skewness = stats.skew(values)
                kurtosis = stats.kurtosis(values)
                
                # Check for extreme skewness
                if abs(skewness) > self.thresholds['max_skewness']:
                    result['warnings'].append(f"Feature '{col}' has extreme skewness: {skewness:.2f}")
                
                # Check for extreme kurtosis
                if abs(kurtosis) > self.thresholds['max_kurtosis']:
                    result['warnings'].append(f"Feature '{col}' has extreme kurtosis: {kurtosis:.2f}")
                
                # Check for normal distribution
                if len(values) > 30:  # Need sufficient data for normality test
                    try:
                        _, p_value = stats.normaltest(values)
                        is_normal = p_value > 0.05
                    except:
                        is_normal = False
                else:
                    is_normal = False
                
                distribution_metrics[col] = {
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'is_normal': is_normal,
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
                
            except Exception as e:
                result['warnings'].append(f"Error analyzing distribution for feature '{col}': {str(e)}")
        
        result['metrics']['distribution_analysis'] = distribution_metrics
        
        return result

    def _validate_feature_importance(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
        """Validate feature importance with parallel processing."""
        result = {'issues': [], 'warnings': [], 'metrics': {}}
        
        numeric_features = features.select_dtypes(include=[np.number]).columns
        
        if len(numeric_features) == 0:
            result['warnings'].append('No numeric features for importance analysis')
            return result
        
        try:
            # Align features and target
            common_index = features.index.intersection(target.index)
            if len(common_index) == 0:
                result['issues'].append('No common index between features and target')
                return result
            
            aligned_features = features.loc[common_index, numeric_features]
            aligned_target = target.loc[common_index]
            
            # Remove rows with NaN values
            complete_data = pd.concat([aligned_features, aligned_target], axis=1).dropna()
            if len(complete_data) == 0:
                result['issues'].append('No complete data for importance analysis')
                return result
            
            features_clean = complete_data[numeric_features]
            target_clean = complete_data[target.name]
            
            # Determine if target is continuous or categorical
            is_continuous = pd.api.types.is_numeric_dtype(target_clean) and target_clean.nunique() > 10
            
            # Calculate mutual information
            if is_continuous:
                mutual_info_scores = mutual_info_regression(features_clean, target_clean, random_state=42)
            else:
                mutual_info_scores = mutual_info_classif(features_clean, target_clean, random_state=42)
            
            # Create importance analysis
            importance_analysis = {}
            low_importance_features = []
            
            for i, feature in enumerate(numeric_features):
                if i < len(mutual_info_scores):
                    importance_score = mutual_info_scores[i]
                    importance_analysis[feature] = {
                        'mutual_info_score': importance_score,
                        'rank': i + 1
                    }
                    
                    if importance_score < self.thresholds['min_mutual_info']:
                        low_importance_features.append(feature)
            
            if low_importance_features:
                result['warnings'].append(f'Found {len(low_importance_features)} features with low importance')
            
            result['metrics']['importance_analysis'] = {
                'feature_importance': importance_analysis,
                'low_importance_features': low_importance_features,
                'target_type': 'continuous' if is_continuous else 'categorical'
            }
            
        except Exception as e:
            result['warnings'].append(f'Feature importance analysis failed: {str(e)}')
        
        return result

    def _validate_feature_stability(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Validate feature stability over time."""
        result = {'issues': [], 'warnings': [], 'metrics': {}}
        
        # Check if features have temporal structure
        if not hasattr(features.index, 'is_monotonic_increasing'):
            result['warnings'].append('Features do not have temporal index for stability analysis')
            return result
        
        numeric_features = features.select_dtypes(include=[np.number]).columns
        
        if len(numeric_features) == 0:
            return result
        
        try:
            # Split data into early and late periods
            split_point = len(features) // 2
            early_features = features.iloc[:split_point, :]
            late_features = features.iloc[split_point:, :]
            
            stability_metrics = {}
            
            for col in numeric_features:
                try:
                    early_values = early_features[col].dropna()
                    late_values = late_features[col].dropna()
                    
                    if len(early_values) < 10 or len(late_values) < 10:
                        continue
                    
                    # Compare means
                    early_mean = np.mean(early_values)
                    late_mean = np.mean(late_values)
                    mean_change = abs(late_mean - early_mean) / abs(early_mean) if early_mean != 0 else 0
                    
                    # Compare standard deviations
                    early_std = np.std(early_values)
                    late_std = np.std(late_values)
                    std_change = abs(late_std - early_std) / early_std if early_std != 0 else 0
                    
                    # Check for significant changes
                    if mean_change > 0.5:  # 50% change in mean
                        result['warnings'].append(f"Feature '{col}' shows significant mean change: {mean_change:.1%}")
                    
                    if std_change > 0.5:  # 50% change in std
                        result['warnings'].append(f"Feature '{col}' shows significant variance change: {std_change:.1%}")
                    
                    stability_metrics[col] = {
                        'mean_change': mean_change,
                        'std_change': std_change,
                        'early_mean': early_mean,
                        'late_mean': late_mean,
                        'early_std': early_std,
                        'late_std': late_std
                    }
                    
                except Exception as e:
                    result['warnings'].append(f"Error analyzing stability for feature '{col}': {str(e)}")
            
            result['metrics']['stability_analysis'] = stability_metrics
            
        except Exception as e:
            result['warnings'].append(f'Feature stability analysis failed: {str(e)}')
        
        return result

    def _generate_feature_analysis(self, features: pd.DataFrame, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive feature analysis summary."""
        analysis = {
            'total_features': len(features.columns),
            'numeric_features': len(features.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(features.select_dtypes(include=['object', 'category']).columns),
            'constant_features': len(metrics.get('constant_features', [])),
            'highly_correlated_pairs': len(metrics.get('correlation_analysis', {}).get('highly_correlated_pairs', [])),
            'low_importance_features': len(metrics.get('importance_analysis', {}).get('low_importance_features', [])),
            'memory_usage_mb': metrics.get('memory_usage_mb', 0)
        }
        
        # Feature quality summary
        feature_quality = metrics.get('feature_quality', {})
        if feature_quality:
            high_nan_features = [f for f, q in feature_quality.items() if q['nan_ratio'] > 0.1]
            high_inf_features = [f for f, q in feature_quality.items() if q['infinite_ratio'] > 0.01]
            
            analysis.update({
                'high_nan_features': len(high_nan_features),
                'high_inf_features': len(high_inf_features),
                'avg_nan_ratio': np.mean([q['nan_ratio'] for q in feature_quality.values()]),
                'avg_inf_ratio': np.mean([q['infinite_ratio'] for q in feature_quality.values()])
            })
        
        return analysis

    def _generate_feature_recommendations(self, issues: List[str], warnings: List[str], 
                                        metrics: Dict[str, Any], feature_analysis: Dict[str, Any]) -> List[str]:
        """Generate feature engineering recommendations."""
        recommendations = []
        
        if issues:
            recommendations.append('Address critical issues before proceeding with feature engineering')
        
        if warnings:
            recommendations.append('Review warnings and consider feature preprocessing')
        
        # Memory usage recommendations
        memory_usage = feature_analysis.get('memory_usage_mb', 0)
        if memory_usage > 1000:  # > 1GB
            recommendations.append('Consider feature selection or dimensionality reduction for large feature sets')
        
        # Correlation recommendations
        highly_correlated = feature_analysis.get('highly_correlated_pairs', 0)
        if highly_correlated > 5:
            recommendations.append('Consider removing highly correlated features to reduce multicollinearity')
        
        # Importance recommendations
        low_importance = feature_analysis.get('low_importance_features', 0)
        if low_importance > 0:
            recommendations.append(f'Consider removing {low_importance} features with low importance')
        
        # Quality recommendations
        high_nan = feature_analysis.get('high_nan_features', 0)
        if high_nan > 0:
            recommendations.append(f'Consider imputation or removal of {high_nan} features with high NaN ratios')
        
        # Distribution recommendations
        distribution_analysis = metrics.get('distribution_analysis', {})
        if distribution_analysis:
            skewed_features = [f for f, d in distribution_analysis.items() if abs(d['skewness']) > 2]
            if skewed_features:
                recommendations.append(f'Consider transformation for {len(skewed_features)} highly skewed features')
        
        return recommendations

    def _calculate_feature_quality_score(self, issues: List[str], warnings: List[str], 
                                       metrics: Dict[str, Any], feature_analysis: Dict[str, Any]) -> float:
        """Calculate overall feature quality score."""
        base_score = 1.0
        
        # Deduct for issues
        base_score -= len(issues) * 0.2
        
        # Deduct for warnings
        base_score -= len(warnings) * 0.05
        
        # Deduct for high memory usage
        memory_usage = feature_analysis.get('memory_usage_mb', 0)
        if memory_usage > 1000:
            base_score -= 0.1
        
        # Deduct for many highly correlated features
        highly_correlated = feature_analysis.get('highly_correlated_pairs', 0)
        if highly_correlated > 10:
            base_score -= 0.2
        
        # Deduct for many low importance features
        low_importance = feature_analysis.get('low_importance_features', 0)
        total_features = feature_analysis.get('total_features', 1)
        if low_importance / total_features > 0.5:
            base_score -= 0.2
        
        # Deduct for high NaN ratios
        avg_nan_ratio = feature_analysis.get('avg_nan_ratio', 0)
        if avg_nan_ratio > 0.2:
            base_score -= 0.2
        
        return max(0.0, min(1.0, base_score))

    def get_feature_recommendations(self, features: pd.DataFrame, target: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Get feature engineering recommendations."""
        validation_result = self.validate_features(features, target, 'feature_recommendations')
        
        return {
            'quality_score': validation_result.quality_score,
            'recommendations': validation_result.recommendations,
            'feature_analysis': validation_result.feature_analysis,
            'critical_issues': validation_result.issues,
            'warnings': validation_result.warnings
        }


def quick_validate_features(features: pd.DataFrame, target: Optional[pd.Series] = None, 
                          context: str = '') -> FeatureValidationResult:
    """Quick validation of features."""
    validator = FeatureEngineeringValidator()
    return validator.validate_features(features, target, context)