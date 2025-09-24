"""
Data Validation for TAS

Comprehensive data validation system for tree architecture search including
data quality assessment, consistency checks, and validation reporting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Import tprint for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)


class ValidationType(Enum):
    """Validation types."""
    DATA_QUALITY = "data_quality"
    CONSISTENCY = "consistency"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    TIMELINESS = "timeliness"
    REGIME_VALIDATION = "regime_validation"
    FEATURE_VALIDATION = "feature_validation"


@dataclass
class ValidationConfig:
    """Configuration for data validation."""
    
    # Validation types
    validation_types: List[ValidationType] = field(default_factory=lambda: [
        ValidationType.DATA_QUALITY,
        ValidationType.CONSISTENCY,
        ValidationType.COMPLETENESS,
        ValidationType.ACCURACY,
        ValidationType.TIMELINESS
    ])
    
    # Data quality thresholds
    max_missing_ratio: float = 0.1
    max_outlier_ratio: float = 0.05
    min_data_points: int = 100
    max_data_points: int = 1000000
    
    # Consistency checks
    check_ohlc_consistency: bool = True
    check_price_positive: bool = True
    check_volume_non_negative: bool = True
    check_timestamp_order: bool = True
    
    # Completeness checks
    check_required_columns: bool = True
    required_columns: List[str] = field(default_factory=lambda: ['open', 'high', 'low', 'close', 'volume'])
    check_data_continuity: bool = True
    
    # Accuracy checks
    check_price_accuracy: bool = True
    check_volume_accuracy: bool = True
    price_tolerance: float = 0.001
    volume_tolerance: float = 0.001
    
    # Timeliness checks
    check_timestamp_freshness: bool = True
    max_age_hours: int = 24
    check_frequency_consistency: bool = True
    
    # Regime validation
    enable_regime_validation: bool = True
    min_regime_duration: int = 10
    max_regime_transitions: int = 100
    
    # Feature validation
    enable_feature_validation: bool = True
    check_feature_correlations: bool = True
    max_correlation_threshold: float = 0.95
    check_feature_importance: bool = True
    
    # Output configuration
    save_validation_results: bool = True
    output_directory: str = "validation_results"
    generate_validation_report: bool = True


@dataclass
class ValidationResult:
    """Result of data validation."""
    
    # Validation results
    validation_passed: bool
    validation_score: float
    validation_details: Dict[str, Any]
    
    # Data quality metrics
    data_quality_score: float
    missing_data_ratio: float
    outlier_ratio: float
    consistency_score: float
    completeness_score: float
    accuracy_score: float
    timeliness_score: float
    
    # Validation issues
    critical_issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    
    # Validation metadata
    validation_time: float
    validation_types: List[str]
    data_shape: Tuple[int, int]
    data_range: Tuple[datetime, datetime]
    
    # Performance metrics
    validation_metrics: Dict[str, float]
    
    # Metadata
    config: ValidationConfig
    errors: List[str] = field(default_factory=list)


class DataValidator:
    """
    Comprehensive data validator for TAS.
    
    Provides data quality assessment, consistency checks,
    and validation reporting for tree architecture search.
    """
    
    def __init__(self, config: ValidationConfig):
        """Initialize data validator.
        
        Args:
            config: Validation configuration
        """
        tprint_info("🔍 Initializing Data Validator")
        tprint_debug(f"Configuration: {config}")
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        tprint_success("✅ Data Validator initialized")
        tprint_info(f"📊 Validation types: {[vt.value for vt in config.validation_types]}")
        tprint_info(f"📊 Data quality thresholds: missing={config.max_missing_ratio}, outliers={config.max_outlier_ratio}")
        self.logger.info("✅ Data Validator initialized")
        self.logger.info(f"📊 Validation types: {[vt.value for vt in config.validation_types]}")
        self.logger.info(f"📊 Data quality thresholds: missing={config.max_missing_ratio}, outliers={config.max_outlier_ratio}")
    
    def validate_data(self, data: pd.DataFrame, regime_data: Optional[Dict[str, Any]] = None, 
                     features: Optional[pd.DataFrame] = None) -> ValidationResult:
        """
        Validate data for TAS.
        
        Args:
            data: Input data to validate
            regime_data: Optional regime data
            features: Optional engineered features
            
        Returns:
            Validation result
        """
        tprint_info("🚀 Starting data validation")
        self.logger.info("🚀 Starting data validation")
        start_time = datetime.now()
        
        try:
            # Initialize validation results
            tprint_debug("📊 Initializing validation results...")
            validation_details = {}
            critical_issues = []
            warnings = []
            recommendations = []
            
            # Perform validation based on configuration
            tprint_info(f"🔍 Performing {len(self.config.validation_types)} validation types...")
            for validation_type in self.config.validation_types:
                tprint_debug(f"🔍 Validating {validation_type.value}...")
                if validation_type == ValidationType.DATA_QUALITY:
                    quality_result = self._validate_data_quality(data)
                    validation_details['data_quality'] = quality_result
                    tprint_debug(f"✅ Data quality validation completed")
                
                elif validation_type == ValidationType.CONSISTENCY:
                    consistency_result = self._validate_consistency(data)
                    validation_details['consistency'] = consistency_result
                    tprint_debug(f"✅ Consistency validation completed")
                
                elif validation_type == ValidationType.COMPLETENESS:
                    completeness_result = self._validate_completeness(data)
                    validation_details['completeness'] = completeness_result
                    tprint_debug(f"✅ Completeness validation completed")
                
                elif validation_type == ValidationType.ACCURACY:
                    accuracy_result = self._validate_accuracy(data)
                    validation_details['accuracy'] = accuracy_result
                    tprint_debug(f"✅ Accuracy validation completed")
                
                elif validation_type == ValidationType.TIMELINESS:
                    timeliness_result = self._validate_timeliness(data)
                    validation_details['timeliness'] = timeliness_result
                    tprint_debug(f"✅ Timeliness validation completed")
                
                elif validation_type == ValidationType.REGIME_VALIDATION and regime_data is not None:
                    regime_result = self._validate_regime_data(regime_data)
                    validation_details['regime_validation'] = regime_result
                    tprint_debug(f"✅ Regime validation completed")
                
                elif validation_type == ValidationType.FEATURE_VALIDATION and features is not None:
                    feature_result = self._validate_features(features)
                    validation_details['feature_validation'] = feature_result
                    tprint_debug(f"✅ Feature validation completed")
            
            # Collect issues and warnings
            tprint_debug("📋 Collecting issues and warnings...")
            for validation_name, validation_result in validation_details.items():
                if 'issues' in validation_result:
                    critical_issues.extend(validation_result['issues'])
                if 'warnings' in validation_result:
                    warnings.extend(validation_result['warnings'])
                if 'recommendations' in validation_result:
                    recommendations.extend(validation_result['recommendations'])
            
            # Calculate overall validation score
            tprint_debug("📊 Calculating overall validation score...")
            validation_score = self._calculate_validation_score(validation_details)
            validation_passed = validation_score >= 0.8 and len(critical_issues) == 0
            tprint_info(f"📊 Validation score: {validation_score:.3f}, Passed: {validation_passed}")
            
            # Calculate individual scores
            data_quality_score = validation_details.get('data_quality', {}).get('score', 0.0)
            consistency_score = validation_details.get('consistency', {}).get('score', 0.0)
            completeness_score = validation_details.get('completeness', {}).get('score', 0.0)
            accuracy_score = validation_details.get('accuracy', {}).get('score', 0.0)
            timeliness_score = validation_details.get('timeliness', {}).get('score', 0.0)
            
            # Calculate data quality metrics
            missing_data_ratio = validation_details.get('data_quality', {}).get('missing_ratio', 0.0)
            outlier_ratio = validation_details.get('data_quality', {}).get('outlier_ratio', 0.0)
            
            # Calculate validation time
            validation_time = (datetime.now() - start_time).total_seconds()
            
            # Create comprehensive result
            tprint_debug("📋 Creating comprehensive validation result...")
            result = ValidationResult(
                # Validation results
                validation_passed=validation_passed,
                validation_score=validation_score,
                validation_details=validation_details,
                
                # Data quality metrics
                data_quality_score=data_quality_score,
                missing_data_ratio=missing_data_ratio,
                outlier_ratio=outlier_ratio,
                consistency_score=consistency_score,
                completeness_score=completeness_score,
                accuracy_score=accuracy_score,
                timeliness_score=timeliness_score,
                
                # Validation issues
                critical_issues=critical_issues,
                warnings=warnings,
                recommendations=recommendations,
                
                # Validation metadata
                validation_time=validation_time,
                validation_types=[vt.value for vt in self.config.validation_types],
                data_shape=data.shape,
                data_range=(data.index[0], data.index[-1]) if isinstance(data.index, pd.DatetimeIndex) else (None, None),
                
                # Performance metrics
                validation_metrics=self._calculate_validation_metrics(validation_details),
                
                # Metadata
                config=self.config
            )
            
            # Save validation results if configured
            if self.config.save_validation_results:
                tprint_debug("💾 Saving validation results...")
                self._save_validation_results(result)
                tprint_success("✅ Validation results saved")
                self._save_validation_results(result)
            
            # Generate validation report if configured
            if self.config.generate_validation_report:
                tprint_debug("📄 Generating validation report...")
                self._generate_validation_report(result)
                tprint_success("✅ Validation report generated")
            
            tprint_success(f"✅ Data validation completed in {result.validation_time:.2f}s")
            tprint_info(f"📊 Validation passed: {result.validation_passed}")
            tprint_info(f"📊 Validation score: {result.validation_score:.3f}")
            tprint_info(f"📊 Critical issues: {len(result.critical_issues)}")
            tprint_info(f"📊 Warnings: {len(result.warnings)}")
            self.logger.info(f"✅ Data validation completed in {result.validation_time:.2f}s")
            self.logger.info(f"📊 Validation passed: {result.validation_passed}")
            self.logger.info(f"📊 Validation score: {result.validation_score:.3f}")
            self.logger.info(f"📊 Critical issues: {len(result.critical_issues)}")
            self.logger.info(f"📊 Warnings: {len(result.warnings)}")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Data validation failed: {e}")
            self.logger.error(f"❌ Data validation failed: {e}")
            raise
    
    def _validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality."""
        try:
            issues = []
            warnings = []
            recommendations = []
            
            # Check missing data
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if missing_ratio > self.config.max_missing_ratio:
                issues.append(f"High missing data ratio: {missing_ratio:.3f} > {self.config.max_missing_ratio}")
            elif missing_ratio > 0:
                warnings.append(f"Missing data detected: {missing_ratio:.3f}")
            
            # Check outliers
            outlier_ratio = 0
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outlier_ratio += (z_scores > 3).sum() / len(data)
            outlier_ratio /= len(numeric_cols) if len(numeric_cols) > 0 else 1
            
            if outlier_ratio > self.config.max_outlier_ratio:
                issues.append(f"High outlier ratio: {outlier_ratio:.3f} > {self.config.max_outlier_ratio}")
            elif outlier_ratio > 0:
                warnings.append(f"Outliers detected: {outlier_ratio:.3f}")
            
            # Check data size
            if len(data) < self.config.min_data_points:
                issues.append(f"Insufficient data points: {len(data)} < {self.config.min_data_points}")
            elif len(data) > self.config.max_data_points:
                warnings.append(f"Large dataset: {len(data)} > {self.config.max_data_points}")
            
            # Calculate quality score
            quality_score = 1.0 - missing_ratio - outlier_ratio
            quality_score = max(0.0, min(1.0, quality_score))
            
            return {
                'score': quality_score,
                'missing_ratio': missing_ratio,
                'outlier_ratio': outlier_ratio,
                'issues': issues,
                'warnings': warnings,
                'recommendations': recommendations
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Data quality validation failed: {e}")
            return {'score': 0.0, 'issues': [f'Validation error: {str(e)}'], 'warnings': [], 'recommendations': []}
    
    def _validate_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data consistency."""
        try:
            issues = []
            warnings = []
            recommendations = []
            
            # Check OHLC consistency
            if self.config.check_ohlc_consistency:
                if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                    # Check high >= max(open, close)
                    high_consistency = data['high'] >= np.maximum(data['open'], data['close'])
                    if not high_consistency.all():
                        issues.append("High price consistency issues")
                    
                    # Check low <= min(open, close)
                    low_consistency = data['low'] <= np.minimum(data['open'], data['close'])
                    if not low_consistency.all():
                        issues.append("Low price consistency issues")
            
            # Check price positivity
            if self.config.check_price_positive:
                price_cols = ['open', 'high', 'low', 'close']
                for col in price_cols:
                    if col in data.columns:
                        if (data[col] <= 0).any():
                            issues.append(f"Negative prices in {col}")
            
            # Check volume non-negativity
            if self.config.check_volume_non_negative:
                if 'volume' in data.columns:
                    if (data['volume'] < 0).any():
                        issues.append("Negative volume values")
            
            # Check timestamp order
            if self.config.check_timestamp_order:
                if isinstance(data.index, pd.DatetimeIndex):
                    if not data.index.is_monotonic_increasing:
                        issues.append("Timestamp order issues")
            
            # Calculate consistency score
            consistency_score = 1.0 - len(issues) / 10.0  # Normalize by expected number of checks
            consistency_score = max(0.0, min(1.0, consistency_score))
            
            return {
                'score': consistency_score,
                'issues': issues,
                'warnings': warnings,
                'recommendations': recommendations
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Consistency validation failed: {e}")
            return {'score': 0.0, 'issues': [f'Validation error: {str(e)}'], 'warnings': [], 'recommendations': []}
    
    def _validate_completeness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data completeness."""
        try:
            issues = []
            warnings = []
            recommendations = []
            
            # Check required columns
            if self.config.check_required_columns:
                missing_columns = [col for col in self.config.required_columns if col not in data.columns]
                if missing_columns:
                    issues.append(f"Missing required columns: {missing_columns}")
            
            # Check data continuity
            if self.config.check_data_continuity:
                if isinstance(data.index, pd.DatetimeIndex):
                    time_diffs = data.index.to_series().diff().dropna()
                    if len(time_diffs) > 0:
                        expected_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
                        irregular_intervals = (abs(time_diffs - expected_interval) > timedelta(seconds=30)).sum()
                        if irregular_intervals > 0:
                            warnings.append(f"Irregular intervals detected: {irregular_intervals}")
            
            # Calculate completeness score
            completeness_score = 1.0 - len(issues) / len(self.config.required_columns)
            completeness_score = max(0.0, min(1.0, completeness_score))
            
            return {
                'score': completeness_score,
                'issues': issues,
                'warnings': warnings,
                'recommendations': recommendations
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Completeness validation failed: {e}")
            return {'score': 0.0, 'issues': [f'Validation error: {str(e)}'], 'warnings': [], 'recommendations': []}
    
    def _validate_accuracy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data accuracy."""
        try:
            issues = []
            warnings = []
            recommendations = []
            
            # Check price accuracy
            if self.config.check_price_accuracy:
                price_cols = ['open', 'high', 'low', 'close']
                for col in price_cols:
                    if col in data.columns:
                        # Check for unrealistic price values
                        if data[col].max() / data[col].min() > 1000:  # Arbitrary threshold
                            warnings.append(f"Unrealistic price range in {col}")
            
            # Check volume accuracy
            if self.config.check_volume_accuracy:
                if 'volume' in data.columns:
                    # Check for unrealistic volume values
                    if data['volume'].max() / data['volume'].min() > 10000:  # Arbitrary threshold
                        warnings.append("Unrealistic volume range")
            
            # Calculate accuracy score
            accuracy_score = 1.0 - len(warnings) / 10.0  # Normalize by expected number of checks
            accuracy_score = max(0.0, min(1.0, accuracy_score))
            
            return {
                'score': accuracy_score,
                'issues': issues,
                'warnings': warnings,
                'recommendations': recommendations
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Accuracy validation failed: {e}")
            return {'score': 0.0, 'issues': [f'Validation error: {str(e)}'], 'warnings': [], 'recommendations': []}
    
    def _validate_timeliness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data timeliness."""
        try:
            issues = []
            warnings = []
            recommendations = []
            
            # Check timestamp freshness
            if self.config.check_timestamp_freshness:
                if isinstance(data.index, pd.DatetimeIndex):
                    latest_timestamp = data.index.max()
                    age_hours = (datetime.now() - latest_timestamp).total_seconds() / 3600
                    if age_hours > self.config.max_age_hours:
                        warnings.append(f"Data is {age_hours:.1f} hours old")
            
            # Check frequency consistency
            if self.config.check_frequency_consistency:
                if isinstance(data.index, pd.DatetimeIndex):
                    time_diffs = data.index.to_series().diff().dropna()
                    if len(time_diffs) > 0:
                        expected_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
                        irregular_intervals = (abs(time_diffs - expected_interval) > timedelta(seconds=30)).sum()
                        if irregular_intervals > 0:
                            warnings.append(f"Inconsistent frequency: {irregular_intervals} irregular intervals")
            
            # Calculate timeliness score
            timeliness_score = 1.0 - len(warnings) / 10.0  # Normalize by expected number of checks
            timeliness_score = max(0.0, min(1.0, timeliness_score))
            
            return {
                'score': timeliness_score,
                'issues': issues,
                'warnings': warnings,
                'recommendations': recommendations
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Timeliness validation failed: {e}")
            return {'score': 0.0, 'issues': [f'Validation error: {str(e)}'], 'warnings': [], 'recommendations': []}
    
    def _validate_regime_data(self, regime_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate regime data."""
        try:
            issues = []
            warnings = []
            recommendations = []
            
            # Check regime labels
            regime_labels = regime_data.get('regime_labels', [])
            if len(regime_labels) == 0:
                issues.append("No regime labels found")
            
            # Check regime duration
            if len(regime_labels) > 0:
                unique_regimes = np.unique(regime_labels)
                for regime_id in unique_regimes:
                    regime_mask = regime_labels == regime_id
                    regime_duration = np.sum(regime_mask)
                    if regime_duration < self.config.min_regime_duration:
                        warnings.append(f"Regime {regime_id} duration too short: {regime_duration}")
            
            # Check regime transitions
            transitions = 0
            for i in range(1, len(regime_labels)):
                if regime_labels[i] != regime_labels[i-1]:
                    transitions += 1
            
            if transitions > self.config.max_regime_transitions:
                warnings.append(f"Too many regime transitions: {transitions}")
            
            # Calculate regime validation score
            regime_score = 1.0 - len(issues) / 10.0  # Normalize by expected number of checks
            regime_score = max(0.0, min(1.0, regime_score))
            
            return {
                'score': regime_score,
                'issues': issues,
                'warnings': warnings,
                'recommendations': recommendations
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime validation failed: {e}")
            return {'score': 0.0, 'issues': [f'Validation error: {str(e)}'], 'warnings': [], 'recommendations': []}
    
    def _validate_features(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Validate engineered features."""
        try:
            issues = []
            warnings = []
            recommendations = []
            
            # Check feature correlations
            if self.config.check_feature_correlations:
                numeric_cols = features.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    correlations = features[numeric_cols].corr()
                    high_correlations = []
                    for i in range(len(correlations.columns)):
                        for j in range(i+1, len(correlations.columns)):
                            corr = abs(correlations.iloc[i, j])
                            if corr > self.config.max_correlation_threshold:
                                high_correlations.append((correlations.columns[i], correlations.columns[j], corr))
                    
                    if high_correlations:
                        warnings.append(f"High feature correlations: {len(high_correlations)} pairs")
            
            # Check feature importance
            if self.config.check_feature_importance:
                numeric_cols = features.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if features[col].var() == 0:
                        warnings.append(f"Zero variance feature: {col}")
            
            # Calculate feature validation score
            feature_score = 1.0 - len(warnings) / 10.0  # Normalize by expected number of checks
            feature_score = max(0.0, min(1.0, feature_score))
            
            return {
                'score': feature_score,
                'issues': issues,
                'warnings': warnings,
                'recommendations': recommendations
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature validation failed: {e}")
            return {'score': 0.0, 'issues': [f'Validation error: {str(e)}'], 'warnings': [], 'recommendations': []}
    
    def _calculate_validation_score(self, validation_details: Dict[str, Any]) -> float:
        """Calculate overall validation score."""
        try:
            scores = []
            for validation_name, validation_result in validation_details.items():
                if 'score' in validation_result:
                    scores.append(validation_result['score'])
            
            if scores:
                return np.mean(scores)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"⚠️ Validation score calculation failed: {e}")
            return 0.0
    
    def _calculate_validation_metrics(self, validation_details: Dict[str, Any]) -> Dict[str, float]:
        """Calculate validation metrics."""
        try:
            metrics = {}
            
            for validation_name, validation_result in validation_details.items():
                if 'score' in validation_result:
                    metrics[f'{validation_name}_score'] = validation_result['score']
                if 'missing_ratio' in validation_result:
                    metrics[f'{validation_name}_missing_ratio'] = validation_result['missing_ratio']
                if 'outlier_ratio' in validation_result:
                    metrics[f'{validation_name}_outlier_ratio'] = validation_result['outlier_ratio']
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Validation metrics calculation failed: {e}")
            return {}
    
    def _save_validation_results(self, result: ValidationResult):
        """Save validation results to file."""
        try:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save validation results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_results_{timestamp}.json"
            filepath = output_dir / filename
            
            import json
            validation_data = {
                'validation_passed': result.validation_passed,
                'validation_score': result.validation_score,
                'validation_details': result.validation_details,
                'data_quality_score': result.data_quality_score,
                'missing_data_ratio': result.missing_data_ratio,
                'outlier_ratio': result.outlier_ratio,
                'consistency_score': result.consistency_score,
                'completeness_score': result.completeness_score,
                'accuracy_score': result.accuracy_score,
                'timeliness_score': result.timeliness_score,
                'critical_issues': result.critical_issues,
                'warnings': result.warnings,
                'recommendations': result.recommendations,
                'validation_time': result.validation_time,
                'validation_types': result.validation_types,
                'data_shape': result.data_shape,
                'validation_metrics': result.validation_metrics
            }
            
            with open(filepath, 'w') as f:
                json.dump(validation_data, f, indent=2, default=str)
            
            self.logger.info(f"📁 Validation results saved to {filepath}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save validation results: {e}")
    
    def _generate_validation_report(self, result: ValidationResult):
        """Generate validation report."""
        try:
            output_dir = Path(self.config.output_directory)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = output_dir / f"validation_report_{timestamp}.txt"
            
            with open(report_file, 'w') as f:
                f.write("DATA VALIDATION REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Validation Status: {'PASSED' if result.validation_passed else 'FAILED'}\n")
                f.write(f"Overall Score: {result.validation_score:.3f}\n")
                f.write(f"Validation Time: {result.validation_time:.2f}s\n\n")
                
                f.write("DATA QUALITY METRICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Data Quality Score: {result.data_quality_score:.3f}\n")
                f.write(f"Missing Data Ratio: {result.missing_data_ratio:.3f}\n")
                f.write(f"Outlier Ratio: {result.outlier_ratio:.3f}\n")
                f.write(f"Consistency Score: {result.consistency_score:.3f}\n")
                f.write(f"Completeness Score: {result.completeness_score:.3f}\n")
                f.write(f"Accuracy Score: {result.accuracy_score:.3f}\n")
                f.write(f"Timeliness Score: {result.timeliness_score:.3f}\n\n")
                
                f.write("CRITICAL ISSUES\n")
                f.write("-" * 30 + "\n")
                for issue in result.critical_issues:
                    f.write(f"❌ {issue}\n")
                f.write("\n")
                
                f.write("WARNINGS\n")
                f.write("-" * 30 + "\n")
                for warning in result.warnings:
                    f.write(f"⚠️ {warning}\n")
                f.write("\n")
                
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 30 + "\n")
                for recommendation in result.recommendations:
                    f.write(f"💡 {recommendation}\n")
                f.write("\n")
            
            self.logger.info(f"📁 Validation report saved to {report_file}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate validation report: {e}")
    
    def export_validation_results(self, result: ValidationResult, filepath: str):
        """Export validation results to file."""
        try:
            validation_data = {
                'validation_passed': result.validation_passed,
                'validation_score': result.validation_score,
                'data_quality_score': result.data_quality_score,
                'critical_issues': result.critical_issues,
                'warnings': result.warnings,
                'recommendations': result.recommendations
            }
            
            with open(filepath, 'w') as f:
                json.dump(validation_data, f, indent=2, default=str)
            
            self.logger.info(f"📁 Validation results exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export validation results: {e}")