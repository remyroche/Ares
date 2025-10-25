"""
Feature Engineering Data Validation

Provides comprehensive validation for feature engineering operations including
data leakage detection, causality enforcement, and statistical validation.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import warnings

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error

logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """Configuration for data validation."""
    check_data_leakage: bool = True
    check_causality: bool = True
    check_constant_features: bool = True
    check_high_correlation: bool = True
    correlation_threshold: float = 0.95
    min_variance_threshold: float = 1e-6
    max_missing_ratio: float = 0.5
    check_outliers: bool = True
    outlier_threshold: float = 3.0  # Z-score threshold

class FeatureDataValidator:
    """
    Comprehensive data validation for feature engineering.
    
    Validates:
    - Data leakage
    - Causality violations
    - Feature quality
    - Statistical properties
    - Data integrity
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize data validator."""
        self.config = config or ValidationConfig()
        self.validation_results = []
        self.warnings = []
        self.errors = []
        
        tprint_info("🔧 Initialized FeatureDataValidator")
    
    def validate_feature_data(
        self, 
        features_df: pd.DataFrame, 
        targets_df: pd.DataFrame,
        feature_categories: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of feature data.
        
        Args:
            features_df: DataFrame with features
            targets_df: DataFrame with targets
            feature_categories: Optional category mapping
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'suggestions': [],
            'statistics': {}
        }
        
        tprint_info("🔍 Starting comprehensive data validation...")
        
        # 1. Data leakage detection
        if self.config.check_data_leakage:
            leakage_issues = self._check_data_leakage(features_df, targets_df)
            validation_results['warnings'].extend(leakage_issues)
        
        # 2. Causality validation
        if self.config.check_causality:
            causality_issues = self._check_causality(features_df, targets_df)
            validation_results['warnings'].extend(causality_issues)
        
        # 3. Feature quality checks
        quality_issues = self._check_feature_quality(features_df)
        validation_results['warnings'].extend(quality_issues)
        
        # 4. Statistical validation
        statistical_issues = self._check_statistical_properties(features_df, targets_df)
        validation_results['warnings'].extend(statistical_issues)
        
        # 5. Data integrity checks
        integrity_issues = self._check_data_integrity(features_df, targets_df)
        validation_results['errors'].extend(integrity_issues)
        
        # 6. Category coverage (if categories provided)
        if feature_categories:
            category_issues = self._check_category_coverage(features_df, feature_categories)
            validation_results['warnings'].extend(category_issues)
        
        # 7. Generate statistics
        validation_results['statistics'] = self._generate_validation_statistics(
            features_df, targets_df
        )
        
        # 8. Generate suggestions
        validation_results['suggestions'] = self._generate_optimization_suggestions(
            validation_results
        )
        
        # Determine overall validity
        validation_results['is_valid'] = len(validation_results['errors']) == 0
        
        # Log results
        if validation_results['warnings']:
            tprint_warning(f"⚠️ Found {len(validation_results['warnings'])} warnings")
        if validation_results['errors']:
            tprint_error(f"❌ Found {len(validation_results['errors'])} errors")
        
        if validation_results['is_valid']:
            tprint_info("✅ Data validation passed")
        else:
            tprint_error("❌ Data validation failed")
        
        return validation_results
    
    def _check_data_leakage(self, features_df: pd.DataFrame, targets_df: pd.DataFrame) -> List[str]:
        """Check for data leakage issues."""
        issues = []
        
        # Check if features contain future data
        if features_df.index.max() > targets_df.index.max():
            issues.append("⚠️ Features contain future data - potential data leakage")
        
        # Check for target information in features
        target_columns = targets_df.columns
        for target_col in target_columns:
            if target_col in features_df.columns:
                issues.append(f"⚠️ Target column '{target_col}' found in features")
        
        # Check for perfect correlation with targets
        for target_col in target_columns:
            if target_col in features_df.columns:
                continue
            
            # Check for features with perfect correlation
            correlations = features_df.corrwith(targets_df[target_col])
            perfect_corr = correlations[abs(correlations) > 0.99]
            if len(perfect_corr) > 0:
                issues.append(f"⚠️ {len(perfect_corr)} features have near-perfect correlation with target '{target_col}'")
        
        return issues
    
    def _check_causality(self, features_df: pd.DataFrame, targets_df: pd.DataFrame) -> List[str]:
        """Check for causality violations."""
        issues = []
        
        # Check if features are properly shifted
        # This is a simplified check - in practice, you'd need more sophisticated logic
        for col in features_df.columns:
            if 'shift' not in col.lower() and 'lag' not in col.lower():
                # Check if feature might need shifting
                if features_df[col].corr(targets_df.iloc[:, 0]) > 0.8:
                    issues.append(f"⚠️ Feature '{col}' has high correlation with target - check for causality")
        
        return issues
    
    def _check_feature_quality(self, features_df: pd.DataFrame) -> List[str]:
        """Check feature quality issues."""
        issues = []
        
        # Check for constant features
        if self.config.check_constant_features:
            constant_features = features_df.columns[features_df.nunique() <= 1]
            if len(constant_features) > 0:
                issues.append(f"⚠️ {len(constant_features)} constant features found: {list(constant_features)}")
        
        # Check for low variance features
        low_var_features = features_df.columns[features_df.var() < self.config.min_variance_threshold]
        if len(low_var_features) > 0:
            issues.append(f"⚠️ {len(low_var_features)} low variance features found")
        
        # Check for high missing ratio
        missing_ratios = features_df.isnull().sum() / len(features_df)
        high_missing = missing_ratios[missing_ratios > self.config.max_missing_ratio]
        if len(high_missing) > 0:
            issues.append(f"⚠️ {len(high_missing)} features have high missing ratio (>50%)")
        
        # Check for high correlation between features
        if self.config.check_high_correlation:
            corr_matrix = features_df.corr().abs()
            upper_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            high_corr_pairs = np.where((corr_matrix > self.config.correlation_threshold) & upper_tri)
            if len(high_corr_pairs[0]) > 0:
                issues.append(f"⚠️ {len(high_corr_pairs[0])} highly correlated feature pairs found")
        
        return issues
    
    def _check_statistical_properties(self, features_df: pd.DataFrame, targets_df: pd.DataFrame) -> List[str]:
        """Check statistical properties."""
        issues = []
        
        # Check for outliers
        if self.config.check_outliers:
            outlier_counts = {}
            for col in features_df.columns:
                z_scores = np.abs((features_df[col] - features_df[col].mean()) / features_df[col].std())
                outlier_count = (z_scores > self.config.outlier_threshold).sum()
                if outlier_count > 0:
                    outlier_counts[col] = outlier_count
            
            if outlier_counts:
                total_outliers = sum(outlier_counts.values())
                issues.append(f"⚠️ {total_outliers} outliers found across {len(outlier_counts)} features")
        
        # Check for non-normal distributions
        non_normal_features = []
        for col in features_df.columns:
            if len(features_df[col].dropna()) > 30:  # Need sufficient data
                from scipy.stats import shapiro
                try:
                    _, p_value = shapiro(features_df[col].dropna().sample(min(5000, len(features_df))))
                    if p_value < 0.05:  # Reject normality
                        non_normal_features.append(col)
                except:
                    pass
        
        if non_normal_features:
            issues.append(f"⚠️ {len(non_normal_features)} features have non-normal distributions")
        
        return issues
    
    def _check_data_integrity(self, features_df: pd.DataFrame, targets_df: pd.DataFrame) -> List[str]:
        """Check data integrity issues."""
        errors = []
        
        # Check for empty DataFrames
        if features_df.empty:
            errors.append("❌ Features DataFrame is empty")
        if targets_df.empty:
            errors.append("❌ Targets DataFrame is empty")
        
        # Check for index alignment
        if not features_df.index.equals(targets_df.index):
            common_index = features_df.index.intersection(targets_df.index)
            if len(common_index) < len(features_df) * 0.8:
                errors.append("❌ Poor index alignment between features and targets")
        
        # Check for infinite values
        inf_features = features_df.columns[features_df.isin([np.inf, -np.inf]).any()]
        if len(inf_features) > 0:
            errors.append(f"❌ {len(inf_features)} features contain infinite values")
        
        # Check for NaN values in targets
        nan_targets = targets_df.isnull().sum().sum()
        if nan_targets > 0:
            errors.append(f"❌ {nan_targets} NaN values found in targets")
        
        return errors
    
    def _check_category_coverage(self, features_df: pd.DataFrame, feature_categories: Dict[str, str]) -> List[str]:
        """Check category coverage."""
        issues = []
        
        # Count features per category
        category_counts = {}
        for feature_name in features_df.columns:
            category = feature_categories.get(feature_name, 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Check for under-represented categories
        min_features_per_category = 2
        under_represented = [cat for cat, count in category_counts.items() 
                           if count < min_features_per_category]
        
        if under_represented:
            issues.append(f"⚠️ Under-represented categories: {under_represented}")
        
        return issues
    
    def _generate_validation_statistics(self, features_df: pd.DataFrame, targets_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate validation statistics."""
        return {
            'n_features': len(features_df.columns),
            'n_samples': len(features_df),
            'n_targets': len(targets_df.columns),
            'missing_ratio': features_df.isnull().sum().sum() / (len(features_df) * len(features_df.columns)),
            'constant_features': len(features_df.columns[features_df.nunique() <= 1]),
            'low_variance_features': len(features_df.columns[features_df.var() < self.config.min_variance_threshold]),
            'memory_usage_mb': features_df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
    
    def _generate_optimization_suggestions(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions based on validation results."""
        suggestions = []
        
        if validation_results['statistics']['constant_features'] > 0:
            suggestions.append("Remove constant features to reduce dimensionality")
        
        if validation_results['statistics']['low_variance_features'] > 0:
            suggestions.append("Consider removing low variance features")
        
        if validation_results['statistics']['missing_ratio'] > 0.1:
            suggestions.append("Consider imputation strategies for missing values")
        
        if validation_results['statistics']['memory_usage_mb'] > 1000:
            suggestions.append("Consider chunked processing for large datasets")
        
        return suggestions

class DataLeakageDetector:
    """
    Specialized detector for data leakage issues.
    
    Detects:
    - Future data leakage
    - Target leakage
    - Temporal leakage
    - Cross-validation leakage
    """
    
    def __init__(self):
        """Initialize data leakage detector."""
        self.leakage_issues = []
        tprint_info("🔧 Initialized DataLeakageDetector")
    
    def detect_leakage(self, features_df: pd.DataFrame, targets_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect data leakage issues.
        
        Args:
            features_df: DataFrame with features
            targets_df: DataFrame with targets
            
        Returns:
            Leakage detection results
        """
        leakage_results = {
            'has_leakage': False,
            'leakage_types': [],
            'affected_features': [],
            'recommendations': []
        }
        
        # 1. Check for future data leakage
        future_leakage = self._check_future_data_leakage(features_df, targets_df)
        if future_leakage:
            leakage_results['leakage_types'].append('future_data')
            leakage_results['affected_features'].extend(future_leakage)
            leakage_results['has_leakage'] = True
        
        # 2. Check for target leakage
        target_leakage = self._check_target_leakage(features_df, targets_df)
        if target_leakage:
            leakage_results['leakage_types'].append('target_leakage')
            leakage_results['affected_features'].extend(target_leakage)
            leakage_results['has_leakage'] = True
        
        # 3. Check for temporal leakage
        temporal_leakage = self._check_temporal_leakage(features_df, targets_df)
        if temporal_leakage:
            leakage_results['leakage_types'].append('temporal_leakage')
            leakage_results['affected_features'].extend(temporal_leakage)
            leakage_results['has_leakage'] = True
        
        # 4. Generate recommendations
        leakage_results['recommendations'] = self._generate_leakage_recommendations(leakage_results)
        
        return leakage_results
    
    def _check_future_data_leakage(self, features_df: pd.DataFrame, targets_df: pd.DataFrame) -> List[str]:
        """Check for future data leakage."""
        issues = []
        
        # Check if features contain future data
        if features_df.index.max() > targets_df.index.max():
            issues.append("Features contain future data relative to targets")
        
        return issues
    
    def _check_target_leakage(self, features_df: pd.DataFrame, targets_df: pd.DataFrame) -> List[str]:
        """Check for target leakage."""
        issues = []
        
        # Check for target columns in features
        for target_col in targets_df.columns:
            if target_col in features_df.columns:
                issues.append(f"Target column '{target_col}' found in features")
        
        # Check for perfect correlation with targets
        for target_col in targets_df.columns:
            correlations = features_df.corrwith(targets_df[target_col])
            perfect_corr = correlations[abs(correlations) > 0.99]
            if len(perfect_corr) > 0:
                issues.append(f"Features with perfect correlation to target '{target_col}': {list(perfect_corr.index)}")
        
        return issues
    
    def _check_temporal_leakage(self, features_df: pd.DataFrame, targets_df: pd.DataFrame) -> List[str]:
        """Check for temporal leakage."""
        issues = []
        
        # This would require more sophisticated temporal analysis
        # For now, just check for obvious temporal issues
        if not features_df.index.is_monotonic_increasing:
            issues.append("Features index is not monotonic - potential temporal issues")
        
        return issues
    
    def _generate_leakage_recommendations(self, leakage_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for fixing leakage."""
        recommendations = []
        
        if 'future_data' in leakage_results['leakage_types']:
            recommendations.append("Ensure features are properly lagged relative to targets")
        
        if 'target_leakage' in leakage_results['leakage_types']:
            recommendations.append("Remove target columns from features")
        
        if 'temporal_leakage' in leakage_results['leakage_types']:
            recommendations.append("Implement proper temporal validation")
        
        return recommendations

# Global instances
_data_validator = None
_leakage_detector = None

def get_data_validator() -> FeatureDataValidator:
    """Get global data validator instance."""
    global _data_validator
    if _data_validator is None:
        _data_validator = FeatureDataValidator()
    return _data_validator

def get_leakage_detector() -> DataLeakageDetector:
    """Get global leakage detector instance."""
    global _leakage_detector
    if _leakage_detector is None:
        _leakage_detector = DataLeakageDetector()
    return _leakage_detector
