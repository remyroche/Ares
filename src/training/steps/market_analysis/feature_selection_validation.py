"""
Feature Selection Validation Utilities

This module provides validation utilities to ensure feature selection aligns with
intended use cases and maintains quality standards for regime clustering.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
import pandas as pd

# Import feature categorization system
from src.feature_generation.categories.regime_feature_categorization import (
    RegimeFeatureCategorizer,
    FeatureUseCase,
    FeatureCategory
)

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success

logger = logging.getLogger(__name__)


class FeatureSelectionValidator:
    """
    Validates feature selection quality and alignment with use cases.
    
    Ensures that:
    - Features are appropriate for the intended use case
    - Critical feature categories are adequately represented
    - Features meet quality standards (variance, correlation, etc.)
    - No circular dependencies in feature selection
    """
    
    def __init__(self):
        self.categorizer = RegimeFeatureCategorizer()
        self.validation_history = []
    
    def validate_feature_selection(
        self,
        selected_features: List[str],
        use_case: FeatureUseCase,
        features_df: Optional[pd.DataFrame] = None,
        expected_categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of feature selection.
        
        Args:
            selected_features: List of selected feature names
            use_case: Intended use case for the features
            features_df: Optional DataFrame with feature data for quality checks
            expected_categories: Optional list of expected category names
            
        Returns:
            Validation report dictionary
        """
        tprint_info(f"Validating feature selection for {use_case.value}")
        
        validation_report = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'recommendations': [],
            'use_case': use_case.value,
            'selected_count': len(selected_features),
            'validation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # 1. Validate feature use case alignment
        alignment_result = self._validate_use_case_alignment(selected_features, use_case)
        validation_report.update({
            'use_case_alignment': alignment_result
        })
        
        if alignment_result['invalid_count'] > 0:
            validation_report['valid'] = False
            validation_report['issues'].append(
                f"Found {alignment_result['invalid_count']} features not appropriate for {use_case.value}"
            )
        
        # 2. Validate category representation
        category_result = self._validate_category_representation(
            selected_features, use_case, expected_categories
        )
        validation_report.update({
            'category_representation': category_result
        })
        
        if not category_result['sufficient_representation']:
            validation_report['warnings'].append(
                f"Insufficient representation from critical categories: {category_result['underrepresented']}"
            )
        
        # 3. Validate feature quality (if data provided)
        if features_df is not None:
            quality_result = self._validate_feature_quality(selected_features, features_df)
            validation_report.update({
                'feature_quality': quality_result
            })
            
            if quality_result['quality_issues'] > 0:
                validation_report['warnings'].append(
                    f"Found {quality_result['quality_issues']} features with quality issues"
                )
        
        # 4. Generate recommendations
        recommendations = self._generate_recommendations(
            selected_features, use_case, validation_report
        )
        validation_report['recommendations'] = recommendations
        
        # Store validation history
        self.validation_history.append(validation_report)
        
        # Log results
        if validation_report['valid']:
            tprint_success(f"Feature selection validated successfully for {use_case.value}")
        else:
            tprint_error(f"Feature selection validation failed: {validation_report['issues']}")
        
        if validation_report['warnings']:
            for warning in validation_report['warnings']:
                tprint_warning(warning)
        
        return validation_report
    
    def _validate_use_case_alignment(
        self,
        selected_features: List[str],
        use_case: FeatureUseCase
    ) -> Dict[str, Any]:
        """Validate that features are appropriate for the use case."""
        valid_features, invalid_features = self.categorizer.validate_feature_usage(
            selected_features, use_case
        )
        
        return {
            'valid_features': valid_features,
            'invalid_features': invalid_features,
            'valid_count': len(valid_features),
            'invalid_count': len(invalid_features),
            'alignment_percentage': len(valid_features) / max(1, len(selected_features)) * 100
        }
    
    def _validate_category_representation(
        self,
        selected_features: List[str],
        use_case: FeatureUseCase,
        expected_categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Validate that critical feature categories are adequately represented."""
        
        # Get categories for use case
        categories = self.categorizer.get_categories_for_use_case(use_case)
        
        # Count features per category
        category_counts = {}
        for category in categories:
            count = sum(1 for f in selected_features if f in category.feature_names)
            category_counts[category.name] = {
                'count': count,
                'priority': category.priority,
                'required': category.name in (expected_categories or [])
            }
        
        # Define minimum thresholds
        min_features_high_priority = 5  # For priority >= 8
        min_features_medium_priority = 3  # For priority >= 5
        min_features_required = 3  # For explicitly required categories
        
        # Check representation
        underrepresented = []
        for cat_name, cat_info in category_counts.items():
            count = cat_info['count']
            priority = cat_info['priority']
            required = cat_info['required']
            
            if required and count < min_features_required:
                underrepresented.append(f"{cat_name} ({count}/{min_features_required})")
            elif priority >= 8 and count < min_features_high_priority:
                underrepresented.append(f"{cat_name} ({count}/{min_features_high_priority})")
            elif priority >= 5 and count < min_features_medium_priority:
                underrepresented.append(f"{cat_name} ({count}/{min_features_medium_priority})")
        
        return {
            'category_counts': category_counts,
            'underrepresented': underrepresented,
            'sufficient_representation': len(underrepresented) == 0,
            'total_categories': len(categories),
            'represented_categories': sum(1 for c in category_counts.values() if c['count'] > 0)
        }
    
    def _validate_feature_quality(
        self,
        selected_features: List[str],
        features_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Validate feature quality metrics."""
        
        quality_issues = 0
        low_variance_features = []
        high_correlation_features = []
        high_nan_features = []
        
        # Get selected feature data
        available_features = [f for f in selected_features if f in features_df.columns]
        if not available_features:
            return {
                'quality_issues': 0,
                'error': 'No selected features found in DataFrame'
            }
        
        feature_data = features_df[available_features]
        
        # 1. Check variance
        variances = feature_data.var()
        variance_threshold = 0.001  # Very low variance threshold
        low_variance = variances[variances < variance_threshold]
        if len(low_variance) > 0:
            quality_issues += len(low_variance)
            low_variance_features = low_variance.index.tolist()
        
        # 2. Check correlation
        if len(available_features) > 1:
            corr_matrix = feature_data.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            high_corr_pairs = []
            for column in upper_triangle.columns:
                correlated_features = upper_triangle.index[upper_triangle[column] > 0.95].tolist()
                if correlated_features:
                    high_corr_pairs.append((column, correlated_features))
            
            if high_corr_pairs:
                quality_issues += len(high_corr_pairs)
                high_correlation_features = [pair[0] for pair in high_corr_pairs]
        
        # 3. Check missing values
        nan_counts = feature_data.isnull().sum()
        nan_threshold = 0.1 * len(feature_data)  # More than 10% missing
        high_nan = nan_counts[nan_counts > nan_threshold]
        if len(high_nan) > 0:
            quality_issues += len(high_nan)
            high_nan_features = high_nan.index.tolist()
        
        return {
            'quality_issues': quality_issues,
            'low_variance_features': low_variance_features,
            'high_correlation_features': high_correlation_features,
            'high_nan_features': high_nan_features,
            'total_features_checked': len(available_features),
            'avg_variance': float(variances.mean()),
            'avg_correlation': float(corr_matrix.mean().mean()) if len(available_features) > 1 else 0.0
        }
    
    def _generate_recommendations(
        self,
        selected_features: List[str],
        use_case: FeatureUseCase,
        validation_report: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for improving feature selection."""
        recommendations = []
        
        # 1. Recommend additional features if count is low
        selected_count = len(selected_features)
        requirements = self.categorizer.get_feature_requirements(use_case)
        
        if selected_count < 40:
            recommendations.append(
                f"Consider increasing feature count to 40-80 (current: {selected_count})"
            )
        
        # 2. Recommend including priority features
        priority_features = self.categorizer.get_priority_features(use_case, 20)
        missing_priority = [f for f in priority_features if f not in selected_features]
        if missing_priority and len(missing_priority) > 5:
            recommendations.append(
                f"Consider including more priority features: {missing_priority[:5]}..."
            )
        
        # 3. Recommend addressing category gaps
        category_rep = validation_report.get('category_representation', {})
        if category_rep.get('underrepresented'):
            recommendations.append(
                f"Increase representation from: {', '.join(category_rep['underrepresented'])}"
            )
        
        # 4. Recommend addressing quality issues
        quality = validation_report.get('feature_quality', {})
        if quality.get('quality_issues', 0) > 0:
            if quality.get('low_variance_features'):
                recommendations.append(
                    f"Remove low-variance features: {quality['low_variance_features'][:3]}..."
                )
            if quality.get('high_correlation_features'):
                recommendations.append(
                    f"Remove highly correlated features: {quality['high_correlation_features'][:3]}..."
                )
        
        return recommendations
    
    def validate_circular_dependency(
        self,
        feature_selection_method: str,
        has_regime_labels: bool,
        clustering_stage: str
    ) -> Dict[str, Any]:
        """
        Validate that feature selection doesn't create circular dependency.
        
        Args:
            feature_selection_method: Method used for selection (e.g., 'treeshap', 'unsupervised')
            has_regime_labels: Whether regime labels are used
            clustering_stage: Stage of clustering ('pre', 'post', 'refinement')
            
        Returns:
            Validation result
        """
        result = {
            'has_circular_dependency': False,
            'dependency_type': None,
            'recommendation': None
        }
        
        # Check for circular dependency patterns
        if clustering_stage == 'pre' and has_regime_labels:
            result['has_circular_dependency'] = True
            result['dependency_type'] = 'Pre-clustering with labels'
            result['recommendation'] = 'Use unsupervised feature selection before initial clustering'
        
        elif feature_selection_method in ['treeshap', 'supervised'] and clustering_stage == 'pre':
            result['has_circular_dependency'] = True
            result['dependency_type'] = 'Supervised selection before clustering'
            result['recommendation'] = 'Use unsupervised methods (variance, correlation) for pre-clustering'
        
        if result['has_circular_dependency']:
            tprint_warning(f"Circular dependency detected: {result['dependency_type']}")
            tprint_info(f"Recommendation: {result['recommendation']}")
        
        return result
    
    def get_fallback_features(
        self,
        use_case: FeatureUseCase,
        max_features: int = 50
    ) -> List[str]:
        """
        Get fallback feature set based on categorization system.
        
        Args:
            use_case: Intended use case
            max_features: Maximum number of features
            
        Returns:
            List of fallback feature names
        """
        priority_features = self.categorizer.get_priority_features(use_case, max_features)
        
        tprint_info(f"Generated {len(priority_features)} fallback features for {use_case.value}")
        
        return priority_features
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validations performed."""
        if not self.validation_history:
            return {'total_validations': 0}
        
        return {
            'total_validations': len(self.validation_history),
            'successful_validations': sum(1 for v in self.validation_history if v['valid']),
            'failed_validations': sum(1 for v in self.validation_history if not v['valid']),
            'avg_features_selected': np.mean([v['selected_count'] for v in self.validation_history]),
            'use_cases': list(set(v['use_case'] for v in self.validation_history))
        }


def validate_regime_clustering_features(
    selected_features: List[str],
    features_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Convenience function to validate features for regime clustering.
    
    Args:
        selected_features: List of selected feature names
        features_df: Optional DataFrame with feature data
        
    Returns:
        Validation report
    """
    validator = FeatureSelectionValidator()
    
    return validator.validate_feature_selection(
        selected_features,
        FeatureUseCase.REGIME_CLUSTERING,
        features_df,
        expected_categories=['core_regime', 'structural_trend', 'clustering_only']
    )


def validate_hdbscan_features(
    selected_features: List[str],
    features_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Convenience function to validate features for HDBSCAN clustering.
    
    Args:
        selected_features: List of selected feature names
        features_df: Optional DataFrame with feature data
        
    Returns:
        Validation report
    """
    validator = FeatureSelectionValidator()
    
    return validator.validate_feature_selection(
        selected_features,
        FeatureUseCase.HDBSCAN_CLUSTERING,
        features_df,
        expected_categories=['clustering_only', 'core_regime', 'advanced_regime']
    )


__all__ = [
    'FeatureSelectionValidator',
    'validate_regime_clustering_features',
    'validate_hdbscan_features'
]
