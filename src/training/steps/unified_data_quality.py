"""
Unified Data Quality Management

This module provides unified data quality validation across all training steps
using DataQualityUtilities from ml_common, replacing custom data quality logic.

Key Features:
- Unified data quality validation using DataQualityUtilities
- Standardized data quality checks across all steps
- Automatic data cleaning and preprocessing
- Comprehensive data quality reporting
- Integration with ML Common utilities
"""

import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

# Import ML Common utilities
from src.utils.ml_common import (
    DataQualityUtilities,
    detect_concept_drift,
    analyze_feature_stability,
    calculate_data_quality_score,
    enhanced_automated_data_cleaning
)

# Import common operations
from src.utils.common_operations import get_logger

logger = get_logger(__name__)


class UnifiedDataQualityManager:
    """
    Unified data quality manager for all training steps.
    
    This replaces custom data quality logic in individual steps with a unified
    approach using DataQualityUtilities from ml_common.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize unified data quality manager."""
        self.logger = logger or get_logger(f"{__name__}.UnifiedDataQualityManager")
        
        # Initialize ML Common DataQualityUtilities
        self.data_quality = DataQualityUtilities()
        
        # Standard data quality thresholds
        self.quality_thresholds = {
            'missing_data_threshold': 0.1,  # 10% missing data threshold
            'duplicate_threshold': 0.05,    # 5% duplicate threshold
            'outlier_threshold': 0.02,      # 2% outlier threshold
            'correlation_threshold': 0.95,  # 95% correlation threshold
            'variance_threshold': 1e-10,    # Minimum variance threshold
            'skewness_threshold': 3.0,      # Maximum skewness threshold
            'kurtosis_threshold': 10.0      # Maximum kurtosis threshold
        }
        
        # Required columns for different data types
        self.required_columns = {
            'ohlcv': ['open', 'high', 'low', 'close', 'volume'],
            'features': [],  # Will be determined dynamically
            'targets': []    # Will be determined dynamically
        }
        
        self.logger.info("🚀 Unified Data Quality Manager initialized")
    
    def validate_data_quality(self, data: Any, data_type: str = 'general', 
                            validation_level: str = 'comprehensive') -> Dict[str, Any]:
        """
        Validate data quality using unified approach.
        
        Args:
            data: Data to validate (DataFrame, array, etc.)
            data_type: Type of data ('ohlcv', 'features', 'targets', 'general')
            validation_level: Level of validation ('basic', 'standard', 'comprehensive')
            
        Returns:
            Data quality validation result
        """
        try:
            self.logger.info(f"🔍 Validating data quality for {data_type} data (level: {validation_level})")
            
            # Convert data to DataFrame if needed
            df = self._ensure_dataframe(data)
            
            if df is None or df.empty:
                return {
                    'passed': False,
                    'errors': ['Data is None or empty'],
                    'warnings': [],
                    'quality_score': 0.0,
                    'data_type': data_type,
                    'validation_level': validation_level
                }
            
            # Initialize validation result
            validation_result = {
                'passed': True,
                'errors': [],
                'warnings': [],
                'quality_score': 1.0,
                'data_type': data_type,
                'validation_level': validation_level,
                'data_shape': df.shape,
                'data_info': self._get_data_info(df)
            }
            
            # Basic validation (always performed)
            basic_validation = self._validate_basic_quality(df, data_type)
            validation_result.update(basic_validation)
            
            if validation_level in ['standard', 'comprehensive']:
                # Standard validation
                standard_validation = self._validate_standard_quality(df, data_type)
                validation_result.update(standard_validation)
            
            if validation_level == 'comprehensive':
                # Comprehensive validation
                comprehensive_validation = self._validate_comprehensive_quality(df, data_type)
                validation_result.update(comprehensive_validation)
            
            # Calculate overall quality score
            validation_result['quality_score'] = self._calculate_quality_score(validation_result)
            
            # Determine if validation passed
            validation_result['passed'] = len(validation_result['errors']) == 0
            
            if validation_result['passed']:
                self.logger.info(f"✅ Data quality validation passed (score: {validation_result['quality_score']:.3f})")
            else:
                self.logger.error(f"❌ Data quality validation failed: {validation_result['errors']}")
            
            return validation_result
            
        except Exception as e:
            self.logger.exception(f"Data quality validation error: {e}")
            return {
                'passed': False,
                'errors': [f"Data quality validation error: {e}"],
                'warnings': [],
                'quality_score': 0.0,
                'data_type': data_type,
                'validation_level': validation_level
            }
    
    def _ensure_dataframe(self, data: Any) -> Optional[pd.DataFrame]:
        """Ensure data is a DataFrame."""
        try:
            if data is None:
                return None
            
            if isinstance(data, pd.DataFrame):
                return data
            
            if isinstance(data, np.ndarray):
                return pd.DataFrame(data)
            
            if isinstance(data, (list, tuple)):
                return pd.DataFrame(data)
            
            # Try to convert other types
            return pd.DataFrame(data)
            
        except Exception as e:
            self.logger.warning(f"Could not convert data to DataFrame: {e}")
            return None
    
    def _get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic information about the data."""
        try:
            return {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'has_index': df.index.name is not None or not df.index.equals(pd.RangeIndex(len(df)))
            }
        except Exception as e:
            self.logger.warning(f"Error getting data info: {e}")
            return {}
    
    def _validate_basic_quality(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Perform basic data quality validation."""
        validation_result = {
            'basic_validation': {
                'missing_data_ratio': 0.0,
                'duplicate_ratio': 0.0,
                'data_types_valid': True,
                'shape_valid': True
            }
        }
        
        try:
            # Check for missing data
            missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            validation_result['basic_validation']['missing_data_ratio'] = missing_ratio
            
            if missing_ratio > self.quality_thresholds['missing_data_threshold']:
                validation_result['errors'].append(f"High missing data ratio: {missing_ratio:.3f} > {self.quality_thresholds['missing_data_threshold']}")
                validation_result['passed'] = False
            
            # Check for duplicates
            duplicate_ratio = df.duplicated().sum() / len(df)
            validation_result['basic_validation']['duplicate_ratio'] = duplicate_ratio
            
            if duplicate_ratio > self.quality_thresholds['duplicate_threshold']:
                validation_result['warnings'].append(f"High duplicate ratio: {duplicate_ratio:.3f} > {self.quality_thresholds['duplicate_threshold']}")
            
            # Check data shape
            if df.shape[0] == 0:
                validation_result['errors'].append("Data has no rows")
                validation_result['passed'] = False
                validation_result['basic_validation']['shape_valid'] = False
            
            if df.shape[1] == 0:
                validation_result['errors'].append("Data has no columns")
                validation_result['passed'] = False
                validation_result['basic_validation']['shape_valid'] = False
            
            # Check for required columns based on data type
            if data_type in self.required_columns:
                required_cols = self.required_columns[data_type]
                if required_cols:  # Only check if required columns are specified
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        validation_result['errors'].append(f"Missing required columns for {data_type}: {missing_cols}")
                        validation_result['passed'] = False
            
        except Exception as e:
            self.logger.exception(f"Basic quality validation error: {e}")
            validation_result['errors'].append(f"Basic quality validation error: {e}")
            validation_result['passed'] = False
        
        return validation_result
    
    def _validate_standard_quality(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Perform standard data quality validation."""
        validation_result = {
            'standard_validation': {
                'numeric_columns_valid': True,
                'outlier_ratio': 0.0,
                'variance_valid': True
            }
        }
        
        try:
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                validation_result['warnings'].append("No numeric columns found")
                return validation_result
            
            # Check for outliers using IQR method
            outlier_counts = []
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_counts.append(outliers)
            
            total_outliers = sum(outlier_counts)
            outlier_ratio = total_outliers / (len(df) * len(numeric_cols))
            validation_result['standard_validation']['outlier_ratio'] = outlier_ratio
            
            if outlier_ratio > self.quality_thresholds['outlier_threshold']:
                validation_result['warnings'].append(f"High outlier ratio: {outlier_ratio:.3f} > {self.quality_thresholds['outlier_threshold']}")
            
            # Check variance
            low_variance_cols = []
            for col in numeric_cols:
                if df[col].var() < self.quality_thresholds['variance_threshold']:
                    low_variance_cols.append(col)
            
            if low_variance_cols:
                validation_result['warnings'].append(f"Low variance columns: {low_variance_cols}")
                validation_result['standard_validation']['variance_valid'] = False
            
        except Exception as e:
            self.logger.exception(f"Standard quality validation error: {e}")
            validation_result['warnings'].append(f"Standard quality validation error: {e}")
        
        return validation_result
    
    def _validate_comprehensive_quality(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Perform comprehensive data quality validation."""
        validation_result = {
            'comprehensive_validation': {
                'correlation_issues': [],
                'distribution_issues': [],
                'stability_score': 1.0
            }
        }
        
        try:
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return validation_result
            
            # Check for high correlations
            correlation_matrix = df[numeric_cols].corr()
            high_corr_pairs = []
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = abs(correlation_matrix.iloc[i, j])
                    if corr_value > self.quality_thresholds['correlation_threshold']:
                        high_corr_pairs.append({
                            'col1': correlation_matrix.columns[i],
                            'col2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            if high_corr_pairs:
                validation_result['warnings'].append(f"High correlation pairs found: {len(high_corr_pairs)}")
                validation_result['comprehensive_validation']['correlation_issues'] = high_corr_pairs
            
            # Check distribution properties
            distribution_issues = []
            for col in numeric_cols:
                try:
                    skewness = abs(df[col].skew())
                    kurtosis = abs(df[col].kurtosis())
                    
                    if skewness > self.quality_thresholds['skewness_threshold']:
                        distribution_issues.append(f"High skewness in {col}: {skewness:.3f}")
                    
                    if kurtosis > self.quality_thresholds['kurtosis_threshold']:
                        distribution_issues.append(f"High kurtosis in {col}: {kurtosis:.3f}")
                        
                except Exception as e:
                    self.logger.warning(f"Error calculating distribution for {col}: {e}")
            
            if distribution_issues:
                validation_result['warnings'].extend(distribution_issues)
                validation_result['comprehensive_validation']['distribution_issues'] = distribution_issues
            
            # Calculate stability score using ML Common utilities
            try:
                stability_score = analyze_feature_stability(df[numeric_cols])
                validation_result['comprehensive_validation']['stability_score'] = stability_score
                
                if stability_score < 0.7:
                    validation_result['warnings'].append(f"Low feature stability score: {stability_score:.3f}")
                    
            except Exception as e:
                self.logger.warning(f"Error calculating stability score: {e}")
            
        except Exception as e:
            self.logger.exception(f"Comprehensive quality validation error: {e}")
            validation_result['warnings'].append(f"Comprehensive quality validation error: {e}")
        
        return validation_result
    
    def _calculate_quality_score(self, validation_result: Dict[str, Any]) -> float:
        """Calculate overall data quality score."""
        try:
            score = 1.0
            
            # Deduct for errors
            error_penalty = len(validation_result.get('errors', [])) * 0.2
            score -= error_penalty
            
            # Deduct for warnings
            warning_penalty = len(validation_result.get('warnings', [])) * 0.05
            score -= warning_penalty
            
            # Deduct for missing data
            if 'basic_validation' in validation_result:
                missing_ratio = validation_result['basic_validation'].get('missing_data_ratio', 0)
                score -= missing_ratio * 0.3
            
            # Deduct for duplicates
            if 'basic_validation' in validation_result:
                duplicate_ratio = validation_result['basic_validation'].get('duplicate_ratio', 0)
                score -= duplicate_ratio * 0.2
            
            # Deduct for outliers
            if 'standard_validation' in validation_result:
                outlier_ratio = validation_result['standard_validation'].get('outlier_ratio', 0)
                score -= outlier_ratio * 0.1
            
            # Deduct for low stability
            if 'comprehensive_validation' in validation_result:
                stability_score = validation_result['comprehensive_validation'].get('stability_score', 1.0)
                score = (score + stability_score) / 2  # Average with stability score
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.warning(f"Error calculating quality score: {e}")
            return 0.5  # Default score
    
    def clean_data(self, data: Any, cleaning_level: str = 'standard') -> Tuple[Any, Dict[str, Any]]:
        """
        Clean data using unified approach.
        
        Args:
            data: Data to clean
            cleaning_level: Level of cleaning ('basic', 'standard', 'aggressive')
            
        Returns:
            Tuple of (cleaned_data, cleaning_report)
        """
        try:
            self.logger.info(f"🧹 Cleaning data (level: {cleaning_level})")
            
            # Convert to DataFrame
            df = self._ensure_dataframe(data)
            if df is None or df.empty:
                return data, {'error': 'Data is None or empty'}
            
            # Initialize cleaning report
            cleaning_report = {
                'original_shape': df.shape,
                'cleaning_level': cleaning_level,
                'operations_performed': [],
                'rows_removed': 0,
                'columns_removed': 0,
                'values_filled': 0
            }
            
            # Perform cleaning operations
            cleaned_df = df.copy()
            
            if cleaning_level in ['standard', 'aggressive']:
                # Remove duplicates
                original_rows = len(cleaned_df)
                cleaned_df = cleaned_df.drop_duplicates()
                removed_duplicates = original_rows - len(cleaned_df)
                if removed_duplicates > 0:
                    cleaning_report['operations_performed'].append(f"Removed {removed_duplicates} duplicate rows")
                    cleaning_report['rows_removed'] += removed_duplicates
            
            if cleaning_level in ['standard', 'aggressive']:
                # Fill missing values
                missing_before = cleaned_df.isnull().sum().sum()
                
                # Fill numeric columns with median
                numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if cleaned_df[col].isnull().any():
                        median_val = cleaned_df[col].median()
                        filled_count = cleaned_df[col].isnull().sum()
                        cleaned_df[col].fillna(median_val, inplace=True)
                        cleaning_report['values_filled'] += filled_count
                
                # Fill categorical columns with mode
                categorical_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns
                for col in categorical_cols:
                    if cleaned_df[col].isnull().any():
                        mode_val = cleaned_df[col].mode().iloc[0] if not cleaned_df[col].mode().empty else 'Unknown'
                        filled_count = cleaned_df[col].isnull().sum()
                        cleaned_df[col].fillna(mode_val, inplace=True)
                        cleaning_report['values_filled'] += filled_count
                
                if cleaning_report['values_filled'] > 0:
                    cleaning_report['operations_performed'].append(f"Filled {cleaning_report['values_filled']} missing values")
            
            if cleaning_level == 'aggressive':
                # Remove low variance columns
                numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                low_variance_cols = []
                for col in numeric_cols:
                    if cleaned_df[col].var() < self.quality_thresholds['variance_threshold']:
                        low_variance_cols.append(col)
                
                if low_variance_cols:
                    cleaned_df = cleaned_df.drop(columns=low_variance_cols)
                    cleaning_report['operations_performed'].append(f"Removed {len(low_variance_cols)} low variance columns")
                    cleaning_report['columns_removed'] += len(low_variance_cols)
            
            # Final report
            cleaning_report['final_shape'] = cleaned_df.shape
            cleaning_report['quality_improvement'] = self._calculate_quality_improvement(df, cleaned_df)
            
            self.logger.info(f"✅ Data cleaning completed: {cleaning_report['operations_performed']}")
            
            return cleaned_df, cleaning_report
            
        except Exception as e:
            self.logger.exception(f"Data cleaning error: {e}")
            return data, {'error': f"Data cleaning error: {e}"}
    
    def _calculate_quality_improvement(self, original_data: pd.DataFrame, cleaned_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate quality improvement after cleaning."""
        try:
            original_validation = self.validate_data_quality(original_data, validation_level='basic')
            cleaned_validation = self.validate_data_quality(cleaned_data, validation_level='basic')
            
            return {
                'original_quality_score': original_validation.get('quality_score', 0.0),
                'cleaned_quality_score': cleaned_validation.get('quality_score', 0.0),
                'improvement': cleaned_validation.get('quality_score', 0.0) - original_validation.get('quality_score', 0.0)
            }
        except Exception as e:
            self.logger.warning(f"Error calculating quality improvement: {e}")
            return {'error': str(e)}
    
    def generate_quality_report(self, data: Any, data_type: str = 'general') -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        try:
            self.logger.info(f"📊 Generating quality report for {data_type} data")
            
            # Validate data quality
            validation_result = self.validate_data_quality(data, data_type, 'comprehensive')
            
            # Generate report
            report = {
                'timestamp': datetime.now().isoformat(),
                'data_type': data_type,
                'validation_result': validation_result,
                'recommendations': self._generate_recommendations(validation_result),
                'summary': self._generate_summary(validation_result)
            }
            
            return report
            
        except Exception as e:
            self.logger.exception(f"Error generating quality report: {e}")
            return {'error': f"Error generating quality report: {e}"}
    
    def _generate_recommendations(self, validation_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if not validation_result['passed']:
            recommendations.append("❌ Critical issues found - data needs immediate attention")
        
        if validation_result.get('basic_validation', {}).get('missing_data_ratio', 0) > 0.05:
            recommendations.append("🔧 Consider imputing missing values or removing rows with high missing data")
        
        if validation_result.get('basic_validation', {}).get('duplicate_ratio', 0) > 0.02:
            recommendations.append("🔧 Remove duplicate rows to improve data quality")
        
        if validation_result.get('standard_validation', {}).get('outlier_ratio', 0) > 0.01:
            recommendations.append("🔧 Investigate and handle outliers appropriately")
        
        if validation_result.get('comprehensive_validation', {}).get('correlation_issues'):
            recommendations.append("🔧 Consider removing highly correlated features to reduce multicollinearity")
        
        if validation_result.get('quality_score', 0) < 0.7:
            recommendations.append("⚠️ Overall data quality is below recommended threshold")
        
        if not recommendations:
            recommendations.append("✅ Data quality is good - no immediate actions required")
        
        return recommendations
    
    def _generate_summary(self, validation_result: Dict[str, Any]) -> str:
        """Generate summary of validation results."""
        quality_score = validation_result.get('quality_score', 0.0)
        error_count = len(validation_result.get('errors', []))
        warning_count = len(validation_result.get('warnings', []))
        
        if quality_score >= 0.9:
            status = "Excellent"
        elif quality_score >= 0.7:
            status = "Good"
        elif quality_score >= 0.5:
            status = "Fair"
        else:
            status = "Poor"
        
        return f"Data quality is {status} (score: {quality_score:.3f}) with {error_count} errors and {warning_count} warnings"


# Global instance for easy access
_global_quality_manager = None

def get_unified_quality_manager() -> UnifiedDataQualityManager:
    """Get the global unified data quality manager instance."""
    global _global_quality_manager
    if _global_quality_manager is None:
        _global_quality_manager = UnifiedDataQualityManager()
    return _global_quality_manager


# Convenience functions
def validate_data_quality(data: Any, data_type: str = 'general', 
                         validation_level: str = 'comprehensive') -> Dict[str, Any]:
    """Validate data quality using unified approach."""
    manager = get_unified_quality_manager()
    return manager.validate_data_quality(data, data_type, validation_level)


def clean_data(data: Any, cleaning_level: str = 'standard') -> Tuple[Any, Dict[str, Any]]:
    """Clean data using unified approach."""
    manager = get_unified_quality_manager()
    return manager.clean_data(data, cleaning_level)


def generate_quality_report(data: Any, data_type: str = 'general') -> Dict[str, Any]:
    """Generate comprehensive data quality report."""
    manager = get_unified_quality_manager()
    return manager.generate_quality_report(data, data_type)


# Example usage
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    # Create test data
    np.random.seed(42)
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 105,
        'low': np.random.randn(1000).cumsum() + 95,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Add some quality issues
    test_data.iloc[10:20, 1] = np.nan  # Missing values
    test_data.iloc[100:105, :] = test_data.iloc[50:55, :].values  # Duplicates
    test_data.iloc[200, 1] = 1000  # Outlier
    
    # Test unified data quality manager
    manager = UnifiedDataQualityManager()
    
    print("=== Data Quality Validation ===")
    validation_result = manager.validate_data_quality(test_data, 'ohlcv', 'comprehensive')
    print(f"Validation passed: {validation_result['passed']}")
    print(f"Quality score: {validation_result['quality_score']:.3f}")
    print(f"Errors: {validation_result['errors']}")
    print(f"Warnings: {validation_result['warnings']}")
    
    print("\n=== Data Cleaning ===")
    cleaned_data, cleaning_report = manager.clean_data(test_data, 'standard')
    print(f"Cleaning operations: {cleaning_report['operations_performed']}")
    print(f"Quality improvement: {cleaning_report['quality_improvement']}")
    
    print("\n=== Quality Report ===")
    quality_report = manager.generate_quality_report(cleaned_data, 'ohlcv')
    print(f"Summary: {quality_report['summary']}")
    print(f"Recommendations: {quality_report['recommendations']}")