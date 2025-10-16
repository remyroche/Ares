"""
Data Quality Utilities

This module provides comprehensive data quality assessment and cleaning utilities with memory-aware operations.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

logger = logging.getLogger(__name__)

class DataQualityUtilities:
    """Data quality utilities with memory management."""

    def __init__(self):
        """Initialize data quality utilities."""
        self.logger = logger.getChild('DataQualityUtilities')
        self.logger.info("🚀 Initializing DataQualityUtilities")

    def automated_data_cleaning(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        cleaning_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform automated data cleaning and quality assessment.

        Args:
            data: Input dataframe
            target_column: Name of target column (if applicable)
            cleaning_config: Configuration for cleaning operations

        Returns:
            Dictionary containing cleaning results and cleaned data
        """
        self.logger.info("🧹 Starting automated data cleaning")

        start_time = time.time()

        # Set default configuration
        if cleaning_config is None:
            cleaning_config = {
                'handle_missing': True,
                'handle_outliers': True,
                'handle_duplicates': True,
                'handle_inconsistencies': True,
                'imputation_method': 'auto',
                'outlier_method': 'auto',
                'duplicate_handling': 'drop'
            }

        # Create a copy of the data
        cleaned_data = data.copy()
        original_shape = data.shape

        cleaning_report = {
            'original_shape': original_shape,
            'cleaning_operations': [],
            'quality_metrics': {},
            'warnings': [],
            'errors': [],
            'cleaning_time': None,
            'success': False
        }

        try:
            # Step 1: Initial quality assessment
            quality_assessment = self._assess_data_quality(cleaned_data, target_column)
            cleaning_report['initial_quality'] = quality_assessment

            # Step 2: Handle missing values
            if cleaning_config.get('handle_missing', True):
                cleaned_data, missing_report = self._handle_missing_values(
                    cleaned_data, cleaning_config.get('imputation_method', 'auto')
                )
                cleaning_report['cleaning_operations'].append(missing_report)

            # Step 3: Handle outliers
            if cleaning_config.get('handle_outliers', True):
                cleaned_data, outlier_report = self._handle_outliers(
                    cleaned_data, target_column, cleaning_config.get('outlier_method', 'auto')
                )
                cleaning_report['cleaning_operations'].append(outlier_report)

            # Step 4: Handle duplicates
            if cleaning_config.get('handle_duplicates', True):
                cleaned_data, duplicate_report = self._handle_duplicates(
                    cleaned_data, cleaning_config.get('duplicate_handling', 'drop')
                )
                cleaning_report['cleaning_operations'].append(duplicate_report)

            # Step 5: Handle data inconsistencies
            if cleaning_config.get('handle_inconsistencies', True):
                cleaned_data, consistency_report = self._handle_data_inconsistencies(cleaned_data)
                cleaning_report['cleaning_operations'].append(consistency_report)

            # Step 6: Final quality assessment
            final_quality = self._assess_data_quality(cleaned_data, target_column)
            cleaning_report['final_quality'] = final_quality

            # Calculate improvement metrics
            cleaning_report['improvement_metrics'] = self._calculate_improvement_metrics(
                quality_assessment, final_quality
            )

            cleaning_report['final_shape'] = cleaned_data.shape
            cleaning_report['rows_removed'] = original_shape[0] - cleaned_data.shape[0]
            cleaning_report['columns_removed'] = original_shape[1] - cleaned_data.shape[1]

            cleaning_report['success'] = True

        except Exception as e:
            self.logger.error(f"❌ Automated data cleaning failed: {e}")
            cleaning_report['errors'].append(str(e))
            cleaning_report['success'] = False
            cleaned_data = data.copy()  # Return original data if cleaning fails

        cleaning_report['cleaning_time'] = time.time() - start_time
        cleaning_report['cleaned_data'] = cleaned_data

        self.logger.info(f"✅ Automated data cleaning completed in {cleaning_report['cleaning_time']:.3f}s")
        return cleaning_report

    def _assess_data_quality(self, data: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """Assess overall data quality."""
        assessment = {}

        try:
            # Basic statistics
            assessment['shape'] = data.shape
            assessment['dtypes'] = data.dtypes.to_dict()

            # Missing values assessment
            missing_stats = self._calculate_missing_statistics(data)
            assessment['missing_values'] = missing_stats

            # Duplicate assessment
            duplicate_stats = self._calculate_duplicate_statistics(data)
            assessment['duplicates'] = duplicate_stats

            # Outlier assessment
            outlier_stats = self._calculate_outlier_statistics(data)
            assessment['outliers'] = outlier_stats

            # Data type consistency
            type_consistency = self._assess_type_consistency(data)
            assessment['type_consistency'] = type_consistency

            # Feature correlation assessment
            if target_column and target_column in data.columns:
                correlation_stats = self._calculate_correlation_statistics(data, target_column)
                assessment['correlations'] = correlation_stats

            # Overall quality score
            assessment['quality_score'] = self._calculate_overall_quality_score(
                missing_stats, duplicate_stats, outlier_stats
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Quality assessment failed: {e}")
            assessment['error'] = str(e)

        return assessment

    def _calculate_missing_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate missing value statistics."""
        missing_info = {}

        try:
            missing_counts = data.isnull().sum()
            missing_percentages = (missing_counts / len(data)) * 100

            missing_info['total_missing'] = int(missing_counts.sum())
            missing_info['missing_percentage'] = float(missing_percentages.sum())
            missing_info['columns_with_missing'] = int((missing_counts > 0).sum())
            missing_info['missing_by_column'] = missing_counts.to_dict()
            missing_info['missing_percentage_by_column'] = missing_percentages.to_dict()

            # Identify columns with high missing rates
            high_missing_cols = missing_percentages[missing_percentages > 50].index.tolist()
            missing_info['high_missing_columns'] = high_missing_cols

        except Exception as e:
            missing_info['error'] = str(e)

        return missing_info

    def _calculate_duplicate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate duplicate statistics."""
        duplicate_info = {}

        try:
            duplicate_rows = data.duplicated().sum()
            duplicate_percentage = (duplicate_rows / len(data)) * 100

            duplicate_info['duplicate_rows'] = int(duplicate_rows)
            duplicate_info['duplicate_percentage'] = float(duplicate_percentage)

            # Check for duplicate columns
            duplicate_columns = []
            for i in range(len(data.columns)):
                for j in range(i + 1, len(data.columns)):
                    if data.iloc[:, i].equals(data.iloc[:, j]):
                        duplicate_columns.append((data.columns[i], data.columns[j]))

            duplicate_info['duplicate_columns'] = duplicate_columns

        except Exception as e:
            duplicate_info['error'] = str(e)

        return duplicate_info

    def _calculate_outlier_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate outlier statistics using IQR method."""
        outlier_info = {}

        try:
            numeric_data = data.select_dtypes(include=[np.number])
            outlier_counts = {}

            for col in numeric_data.columns:
                Q1 = numeric_data[col].quantile(0.25)
                Q3 = numeric_data[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = ((numeric_data[col] < lower_bound) | (numeric_data[col] > upper_bound)).sum()
                outlier_percentage = (outliers / len(numeric_data)) * 100

                outlier_counts[col] = {
                    'count': int(outliers),
                    'percentage': float(outlier_percentage),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }

            outlier_info['outlier_counts'] = outlier_counts
            outlier_info['total_outliers'] = sum(info['count'] for info in outlier_counts.values())

        except Exception as e:
            outlier_info['error'] = str(e)

        return outlier_info

    def _assess_type_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data type consistency."""
        type_info = {}

        try:
            for col in data.columns:
                unique_types = data[col].apply(type).unique()
                type_info[col] = {
                    'declared_type': str(data[col].dtype),
                    'unique_types': [str(t) for t in unique_types],
                    'is_consistent': len(unique_types) == 1
                }

            inconsistent_columns = [col for col, info in type_info.items() if not info['is_consistent']]
            type_info['inconsistent_columns'] = inconsistent_columns

        except Exception as e:
            type_info['error'] = str(e)

        return type_info

    def _calculate_correlation_statistics(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Calculate correlation statistics with target."""
        correlation_info = {}

        try:
            numeric_data = data.select_dtypes(include=[np.number])

            if target_column in numeric_data.columns:
                correlations = numeric_data.corr()[target_column].drop(target_column)

                correlation_info['correlations'] = correlations.to_dict()
                correlation_info['high_correlations'] = correlations[abs(correlations) > 0.8].to_dict()
                correlation_info['low_correlations'] = correlations[abs(correlations) < 0.1].to_dict()

        except Exception as e:
            correlation_info['error'] = str(e)

        return correlation_info

    def _calculate_overall_quality_score(self, missing_stats, duplicate_stats, outlier_stats) -> float:
        """Calculate overall data quality score (0-100)."""
        try:
            # Weights for different quality aspects
            weights = {
                'missing': 0.4,
                'duplicates': 0.3,
                'outliers': 0.3
            }

            # Missing score (higher is better)
            missing_percentage = missing_stats.get('missing_percentage', 100)
            missing_score = max(0, 100 - missing_percentage)

            # Duplicate score (higher is better)
            duplicate_percentage = duplicate_stats.get('duplicate_percentage', 100)
            duplicate_score = max(0, 100 - duplicate_percentage * 10)  # Penalize heavily

            # Outlier score (moderate outliers are acceptable)
            total_outliers = outlier_stats.get('total_outliers', len(outlier_stats.get('outlier_counts', {})) * 100)
            outlier_percentage = (total_outliers / max(1, sum(len(info) for info in [missing_stats, duplicate_stats, outlier_stats]))) * 100
            outlier_score = max(0, 100 - outlier_percentage * 2)

            overall_score = (
                weights['missing'] * missing_score +
                weights['duplicates'] * duplicate_score +
                weights['outliers'] * outlier_score
            )

            return float(overall_score)

        except Exception as e:
            self.logger.warning(f"⚠️ Quality score calculation failed: {e}")
            return 50.0  # Default neutral score

    def _handle_missing_values(self, data: pd.DataFrame, method: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle missing values using specified method."""
        report = {'operation': 'missing_value_handling', 'method': method}
        # Strategy:
        # - 'drop_rows': fast drop when few rows have NaNs
        # - 'drop_columns': drop columns with >50% NaNs
        # - 'mean_median': numeric median, categorical mode
        # - 'knn': KNNImputer for numeric features when many NaNs

        try:
            if method == 'auto':
                # Choose method based on missing percentage
                missing_percentage = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100

                if missing_percentage < 5:
                    method = 'drop_rows'
                elif missing_percentage < 20:
                    method = 'mean_median'
                else:
                    method = 'knn'

            if method == 'drop_rows':
                initial_shape = data.shape
                data = data.dropna()
                rows_removed = initial_shape[0] - data.shape[0]
                report['rows_removed'] = rows_removed

            elif method == 'drop_columns':
                initial_shape = data.shape
                missing_percentages = (data.isnull().sum() / len(data)) * 100
                columns_to_drop = missing_percentages[missing_percentages > 50].index
                data = data.drop(columns=columns_to_drop)
                report['columns_dropped'] = columns_to_drop.tolist()

            elif method == 'mean_median':
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                categorical_columns = data.select_dtypes(include=['object', 'category']).columns

                # Impute numeric with median
                if len(numeric_columns) > 0:
                    numeric_imputer = SimpleImputer(strategy='median')
                    data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])

                # Impute categorical with most frequent
                if len(categorical_columns) > 0:
                    categorical_imputer = SimpleImputer(strategy='most_frequent')
                    data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])

            elif method == 'knn':
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    imputer = KNNImputer(n_neighbors=5)
                    data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

            report['success'] = True

        except Exception as e:
            self.logger.warning(f"⚠️ Missing value handling failed: {e}")
            report['error'] = str(e)
            report['success'] = False

        return data, report

    def _handle_outliers(self, data: pd.DataFrame, target_column: str, method: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle outliers using specified method."""
        report = {'operation': 'outlier_handling', 'method': method}

        try:
            if method == 'auto':
                # Choose method based on data size
                if data.shape[0] > 10000:
                    method = 'iqr'
                else:
                    method = 'isolation_forest'

            if method == 'iqr':
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                initial_shape = data.shape

                for col in numeric_columns:
                    if col == target_column:
                        continue

                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    # Cap outliers instead of removing
                    data[col] = np.clip(data[col], lower_bound, upper_bound)

                report['method_details'] = 'IQR capping'

            elif method == 'isolation_forest':
                from sklearn.ensemble import IsolationForest

                numeric_columns = data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outlier_predictions = iso_forest.fit_predict(data[numeric_columns])

                    # Remove outliers
                    initial_shape = data.shape
                    data = data[outlier_predictions == 1]
                    outliers_removed = initial_shape[0] - data.shape[0]
                    report['outliers_removed'] = outliers_removed

            report['success'] = True

        except Exception as e:
            self.logger.warning(f"⚠️ Outlier handling failed: {e}")
            report['error'] = str(e)
            report['success'] = False

        return data, report

    def _handle_duplicates(self, data: pd.DataFrame, method: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle duplicate rows."""
        report = {'operation': 'duplicate_handling', 'method': method}

        try:
            initial_shape = data.shape

            if method == 'drop':
                data = data.drop_duplicates()
            elif method == 'keep_first':
                data = data.drop_duplicates(keep='first')
            elif method == 'keep_last':
                data = data.drop_duplicates(keep='last')

            duplicates_removed = initial_shape[0] - data.shape[0]
            report['duplicates_removed'] = duplicates_removed
            report['success'] = True

        except Exception as e:
            self.logger.warning(f"⚠️ Duplicate handling failed: {e}")
            report['error'] = str(e)
            report['success'] = False

        return data, report

    def _handle_data_inconsistencies(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle data inconsistencies."""
        report = {'operation': 'consistency_handling'}

        try:
            # Fix string inconsistencies
            string_columns = data.select_dtypes(include=['object', 'string']).columns

            for col in string_columns:
                # Strip whitespace
                data[col] = data[col].astype(str).str.strip()

                # Handle case inconsistencies (optional)
                if len(data[col].unique()) < 20:  # Only for low-cardinality columns
                    data[col] = data[col].str.lower()

            # Fix numeric inconsistencies
            numeric_columns = data.select_dtypes(include=[np.number]).columns

            for col in numeric_columns:
                # Replace infinite values
                data[col] = data[col].replace([np.inf, -np.inf], np.nan)

            report['string_columns_processed'] = string_columns.tolist()
            report['numeric_columns_processed'] = numeric_columns.tolist()
            report['success'] = True

        except Exception as e:
            self.logger.warning(f"⚠️ Consistency handling failed: {e}")
            report['error'] = str(e)
            report['success'] = False

        return data, report

    def _calculate_improvement_metrics(self, initial_quality: Dict, final_quality: Dict) -> Dict[str, Any]:
        """Calculate improvement metrics after cleaning."""
        improvements = {}

        try:
            # Missing values improvement
            if 'missing_values' in initial_quality and 'missing_values' in final_quality:
                initial_missing = initial_quality['missing_values'].get('total_missing', 0)
                final_missing = final_quality['missing_values'].get('total_missing', 0)
                improvements['missing_values_reduction'] = initial_missing - final_missing

            # Duplicate improvement
            if 'duplicates' in initial_quality and 'duplicates' in final_quality:
                initial_duplicates = initial_quality['duplicates'].get('duplicate_rows', 0)
                final_duplicates = final_quality['duplicates'].get('duplicate_rows', 0)
                improvements['duplicates_reduction'] = initial_duplicates - final_duplicates

            # Quality score improvement
            if 'quality_score' in initial_quality and 'quality_score' in final_quality:
                initial_score = initial_quality['quality_score']
                final_score = final_quality['quality_score']
                improvements['quality_score_improvement'] = final_score - initial_score

        except Exception as e:
            improvements['error'] = str(e)

        return improvements

    def check_data_quality(self, data: pd.DataFrame, target_column: Optional[str] = None,
                          quality_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Check data quality and return assessment results.

        Args:
            data: Input dataframe to check
            target_column: Name of target column (if applicable)
            quality_threshold: Minimum quality threshold (0-1)

        Returns:
            Dictionary containing quality assessment results
        """
        self.logger.info("🔍 Starting data quality check")

        try:
            # Perform comprehensive quality assessment
            quality_assessment = self._assess_data_quality(data, target_column)

            # Determine overall quality status
            quality_score = quality_assessment.get('quality_score', 0)
            quality_status = 'good' if quality_score >= quality_threshold else 'poor'

            # Create quality report
            quality_report = {
                'overall_quality_score': quality_score,
                'quality_status': quality_status,
                'quality_threshold': quality_threshold,
                'data_shape': data.shape,
                'assessment_timestamp': time.time(),
                'detailed_assessment': quality_assessment,
                'recommendations': self._generate_quality_recommendations(quality_assessment, quality_threshold)
            }

            # Log quality status
            if quality_status == 'good':
                self.logger.info(f"✅ Data quality check passed (score: {quality_score:.3f})")
            else:
                self.logger.warning(f"⚠️ Data quality check failed (score: {quality_score:.3f}, threshold: {quality_threshold})")

            return quality_report

        except Exception as e:
            self.logger.error(f"❌ Data quality check failed: {e}")
            return {
                'overall_quality_score': 0.0,
                'quality_status': 'error',
                'error': str(e),
                'assessment_timestamp': time.time()
            }

    def _generate_quality_recommendations(self, quality_assessment: Dict[str, Any],
                                        quality_threshold: float) -> List[str]:
        """Generate recommendations based on quality assessment."""
        recommendations = []

        try:
            # Missing values recommendations
            missing_info = quality_assessment.get('missing_values', {})
            missing_percentage = missing_info.get('missing_percentage', 0)

            if missing_percentage > 20:
                recommendations.append(f"🚨 High missing data rate ({missing_percentage:.1f}%) - consider imputation or feature removal")
            elif missing_percentage > 5:
                recommendations.append(f"⚠️ Moderate missing data rate ({missing_percentage:.1f}%) - consider imputation strategies")

            # Duplicate recommendations
            duplicate_info = quality_assessment.get('duplicates', {})
            duplicate_percentage = duplicate_info.get('duplicate_percentage', 0)

            if duplicate_percentage > 10:
                recommendations.append(f"🚨 High duplicate rate ({duplicate_percentage:.1f}%) - remove duplicates")
            elif duplicate_percentage > 1:
                recommendations.append(f"⚠️ Some duplicates detected ({duplicate_percentage:.1f}%) - consider removal")

            # Outlier recommendations
            outlier_info = quality_assessment.get('outliers', {})
            total_outliers = outlier_info.get('total_outliers', 0)

            if total_outliers > len(quality_assessment.get('shape', [0])[0]) * 0.1:
                recommendations.append("🚨 High outlier rate - consider outlier treatment")

            # Type consistency recommendations
            type_info = quality_assessment.get('type_consistency', {})
            inconsistent_columns = type_info.get('inconsistent_columns', [])

            if inconsistent_columns:
                recommendations.append(f"⚠️ Type inconsistencies in columns: {inconsistent_columns}")

            # Overall quality recommendations
            quality_score = quality_assessment.get('quality_score', 0)
            if quality_score < quality_threshold:
                recommendations.append(f"📈 Overall quality below threshold - comprehensive data cleaning recommended")

            if not recommendations:
                recommendations.append("✅ Data quality is acceptable - no major issues detected")

        except Exception as e:
            recommendations.append(f"❌ Error generating recommendations: {e}")

        return recommendations

# Global instance for easy access
_data_quality_instance = None

def get_data_quality_utilities() -> DataQualityUtilities:
    """Get global data quality utilities instance."""
    global _data_quality_instance
    if _data_quality_instance is None:
        _data_quality_instance = DataQualityUtilities()
    return _data_quality_instance

# Export key classes and functions
__all__ = ['DataQualityUtilities', 'get_data_quality_utilities']
