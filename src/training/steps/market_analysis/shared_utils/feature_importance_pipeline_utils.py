"""
Feature Importance Pipeline Utilities

This module provides high-level utility functions for easy integration of
feature importance analysis into various pipeline components and workflows.

Key Utilities:
- Pipeline integration helpers
- Component configuration helpers
- Report enhancement utilities
- Quick analysis functions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

from .feature_importance_integration import (
    FeatureImportanceIntegrationManager, FeatureImportanceIntegrationConfig,
    integrate_feature_importance_with_clustering, enhance_regime_report_with_feature_importance
)

logger = logging.getLogger(__name__)


def create_feature_importance_config_for_pipeline(
    enable_pre_clustering: bool = True,
    enable_post_clustering: bool = True,
    enable_regime_characterization: bool = True,
    importance_methods: Optional[List[str]] = None,
    auto_integrate: bool = True
) -> FeatureImportanceIntegrationConfig:
    """
    Create a feature importance configuration optimized for pipeline integration.

    Args:
        enable_pre_clustering: Enable analysis before clustering
        enable_post_clustering: Enable analysis after clustering
        enable_regime_characterization: Enable regime-specific analysis
        importance_methods: List of importance calculation methods
        auto_integrate: Enable automatic integration with pipeline components

    Returns:
        Configured FeatureImportanceIntegrationConfig
    """
    if importance_methods is None:
        importance_methods = ["mutual_information", "f_classif", "variance"]

    return FeatureImportanceIntegrationConfig(
        enable_pre_clustering_analysis=enable_pre_clustering,
        enable_post_clustering_analysis=enable_post_clustering,
        enable_regime_characterization=enable_regime_characterization,
        importance_methods=importance_methods,
        auto_integrate_with_clustering=auto_integrate,
        auto_integrate_with_reporting=auto_integrate,
        include_detailed_analysis=True,
        save_feature_profiles=True,
        generate_interpretation=True
    )


def analyze_features_for_clustering(
    features: np.ndarray,
    feature_names: List[str],
    clusterer=None,
    config: Optional[FeatureImportanceIntegrationConfig] = None
) -> Dict[str, Any]:
    """
    Complete feature analysis pipeline for clustering workflows.

    This function provides a one-stop solution for:
    1. Pre-clustering feature validation
    2. Clustering with feature importance tracking
    3. Post-clustering regime analysis
    4. Enhanced clustering results

    Args:
        features: Feature matrix
        feature_names: Feature names
        clusterer: Clustering algorithm (optional)
        config: Feature importance configuration

    Returns:
        Comprehensive analysis results
    """
    if config is None:
        config = create_feature_importance_config_for_pipeline()

    try:
        # Initialize integration manager
        manager = FeatureImportanceIntegrationManager(config)

        # Pre-clustering analysis
        pre_analysis = {}
        if config.enable_pre_clustering_analysis:
            pre_analysis = manager.analyze_pre_clustering_features(features, feature_names)
            logger.info("✅ Pre-clustering feature analysis completed")

        # Clustering
        cluster_labels = None
        clustering_info = {}

        if clusterer is not None:
            try:
                cluster_labels = clusterer.fit_predict(features)
                clustering_info = {
                    'n_clusters': len(np.unique(cluster_labels)),
                    'clusterer_type': type(clusterer).__name__,
                    'n_samples': features.shape[0],
                    'n_features': features.shape[1]
                }
                logger.info("✅ Clustering completed")
            except Exception as e:
                logger.warning(f"⚠️ Clustering failed: {e}")
                cluster_labels = np.zeros(features.shape[0])  # Fallback

        # Post-clustering analysis
        post_analysis = {}
        if config.enable_post_clustering_analysis and cluster_labels is not None:
            post_analysis = manager.analyze_post_clustering_regimes(
                features, feature_names, cluster_labels, clustering_info
            )
            logger.info("✅ Post-clustering regime analysis completed")

        # Compile results
        results = {
            'pre_clustering_analysis': pre_analysis,
            'post_clustering_analysis': post_analysis,
            'clustering_results': {
                'cluster_labels': cluster_labels,
                'clustering_info': clustering_info
            },
            'feature_importance_manager': manager,
            'config_used': config,
            'summary': {
                'analysis_completed': True,
                'pre_clustering_enabled': config.enable_pre_clustering_analysis,
                'post_clustering_enabled': config.enable_post_clustering_analysis,
                'clustering_successful': cluster_labels is not None and clustering_info.get('n_clusters', 0) > 1
            }
        }

        logger.info("✅ Complete feature analysis pipeline completed")
        return results

    except Exception as e:
        logger.error(f"❌ Feature analysis pipeline failed: {e}")
        return {
            'error': str(e),
            'summary': {'analysis_completed': False}
        }


def enhance_pipeline_component_with_feature_importance(
    component_instance,
    method_name: str = "execute",
    config: Optional[FeatureImportanceIntegrationConfig] = None
) -> Any:
    """
    Enhance an existing pipeline component with feature importance analysis.

    This function wraps component methods to automatically inject feature importance
    analysis at appropriate stages.

    Args:
        component_instance: Pipeline component instance
        method_name: Name of method to enhance (default: 'execute')
        config: Feature importance configuration

    Returns:
        Enhanced component instance
    """
    if config is None:
        config = create_feature_importance_config_for_pipeline()

    # Create integration manager
    manager = FeatureImportanceIntegrationManager(config)

    # Store original method
    original_method = getattr(component_instance, method_name)

    def enhanced_method(*args, **kwargs):
        """Enhanced method with feature importance analysis."""
        try:
            # Call original method
            result = original_method(*args, **kwargs)

            # Enhance result with feature importance if applicable
            if (isinstance(result, dict) and
                config.auto_integrate_with_reporting):

                enhanced_result = enhance_regime_report_with_feature_importance(result, manager)

                # Add feature importance summary to result
                if manager.regime_characterizations:
                    enhanced_result['feature_importance_summary'] = {
                        'n_regime_analyses': len(manager.regime_characterizations),
                        'total_regimes_analyzed': len(manager.regime_characterizations),
                        'analysis_methods': config.importance_methods
                    }

                return enhanced_result

            return result

        except Exception as e:
            logger.warning(f"⚠️ Feature importance enhancement failed: {e}")
            return original_method(*args, **kwargs)

    # Replace method
    setattr(component_instance, method_name, enhanced_method)

    # Store manager for later access
    component_instance._feature_importance_manager = manager

    logger.info(f"✅ Component {type(component_instance).__name__} enhanced with feature importance analysis")
    return component_instance


def create_feature_importance_report_summary(
    analysis_results: Dict[str, Any],
    include_detailed_profiles: bool = False
) -> Dict[str, Any]:
    """
    Create a concise summary report of feature importance analysis.

    Args:
        analysis_results: Results from feature importance analysis
        include_detailed_profiles: Include detailed regime profiles

    Returns:
        Summary report dictionary
    """
    try:
        summary = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'analysis_type': 'feature_importance_summary',
            'key_insights': {},
            'regime_characteristics': {},
            'feature_rankings': {}
        }

        # Extract key insights
        post_analysis = analysis_results.get('post_clustering_analysis', {})

        if post_analysis:
            # Get most important features
            feature_ranking = post_analysis.get('feature_importance_ranking', [])
            if feature_ranking:
                summary['feature_rankings'] = {
                    'top_10_features': [name for name, _ in feature_ranking[:10]],
                    'bottom_10_features': [name for name, _ in feature_ranking[-10:]],
                    'total_features_analyzed': len(feature_ranking)
                }

            # Get regime profiles
            regime_profiles = post_analysis.get('regime_feature_profiles', {})
            if regime_profiles:
                for regime_id, profile in regime_profiles.items():
                    regime_summary = {
                        'dominant_features': profile.get('dominant_features', [])[:5],
                        'sample_count': profile.get('sample_count', 0),
                        'feature_variance_mean': np.mean(profile.get('feature_variance', [])),
                        'analysis_method': post_analysis.get('method_used', 'unknown')
                    }

                    if not include_detailed_profiles:
                        # Remove detailed arrays for summary
                        regime_summary.pop('feature_variance', None)

                    summary['regime_characteristics'][regime_id] = regime_summary

        # Generate key insights
        insights = []

        if summary['feature_rankings']:
            top_features = summary['feature_rankings']['top_10_features'][:3]
            insights.append(f"Most discriminative features: {', '.join(top_features)}")

        if summary['regime_characteristics']:
            n_regimes = len(summary['regime_characteristics'])
            insights.append(f"Analyzed {n_regimes} distinct regimes")

        summary['key_insights'] = {
            'main_findings': insights,
            'analysis_quality': 'high' if len(insights) >= 2 else 'moderate'
        }

        return summary

    except Exception as e:
        logger.warning(f"⚠️ Failed to create summary report: {e}")
        return {
            'error': str(e),
            'analysis_type': 'feature_importance_summary'
        }


def validate_feature_importance_integration(
    test_features: np.ndarray,
    test_labels: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate that feature importance integration is working correctly.

    Args:
        test_features: Test feature matrix
        test_labels: Test labels/regime assignments
        feature_names: Feature names (optional)

    Returns:
        Validation results
    """
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(test_features.shape[1])]

    try:
        # Create test configuration
        config = FeatureImportanceIntegrationConfig(
            importance_methods=["variance"],  # Use simple method for validation
            enable_pre_clustering_analysis=False,
            enable_post_clustering_analysis=True
        )

        # Initialize manager
        manager = FeatureImportanceIntegrationManager(config)

        # Test post-clustering analysis
        analysis_result = manager.analyze_post_clustering_regimes(
            test_features, feature_names, test_labels
        )

        # Validate results
        validation = {
            'integration_successful': True,
            'analysis_completed': analysis_result is not None and len(analysis_result) > 0,
            'regime_profiles_generated': len(manager.regime_characterizations) > 0,
            'feature_ranking_available': 'feature_importance_ranking' in analysis_result,
            'interpretation_generated': 'interpretation' in analysis_result
        }

        if analysis_result:
            validation['n_regimes_analyzed'] = len(manager.regime_characterizations)
            validation['n_features_analyzed'] = len(feature_names)
            validation['analysis_methods'] = config.importance_methods

        return validation

    except Exception as e:
        return {
            'integration_successful': False,
            'error': str(e),
            'analysis_completed': False
        }


# Quick integration functions for common use cases
def quick_regime_feature_analysis(
    features: np.ndarray,
    feature_names: List[str],
    regime_labels: np.ndarray,
    method: str = "mutual_information"
) -> Dict[str, Any]:
    """
    Quick feature importance analysis for regime discovery.

    Args:
        features: Feature matrix
        feature_names: Feature names
        regime_labels: Regime labels
        method: Analysis method

    Returns:
        Feature importance analysis results
    """
    try:
        from .balanced_feature_extractor import analyze_regime_feature_importance

        return analyze_regime_feature_importance(
            features=features,
            feature_names=feature_names,
            regime_labels=regime_labels,
            method=method
        )
    except Exception as e:
        logger.error(f"❌ Quick analysis failed: {e}")
        return {}


def extract_regime_insights_from_analysis(
    analysis_results: Dict[str, Any],
    top_k_features: int = 5
) -> Dict[str, Any]:
    """
    Extract actionable insights from feature importance analysis.

    Args:
        analysis_results: Feature importance analysis results
        top_k_features: Number of top features to include per regime

    Returns:
        Structured insights for decision making
    """
    try:
        insights = {
            'regime_discriminators': {},
            'regime_stability_indicators': {},
            'feature_redundancy_analysis': {},
            'actionable_recommendations': []
        }

        # Extract regime-specific insights
        regime_profiles = analysis_results.get('regime_feature_profiles', {})
        feature_ranking = analysis_results.get('feature_importance_ranking', [])

        if regime_profiles:
            for regime_id, profile in regime_profiles.items():
                dominant_features = profile.get('dominant_features', [])[:top_k_features]
                insights['regime_discriminators'][regime_id] = dominant_features

                # Stability indicators
                feature_variance = profile.get('feature_variance', [])
                if feature_variance:
                    insights['regime_stability_indicators'][regime_id] = {
                        'feature_variance_mean': np.mean(feature_variance),
                        'feature_variance_std': np.std(feature_variance),
                        'dominant_features': dominant_features
                    }

        # Feature redundancy analysis
        if feature_ranking:
            top_features = [name for name, _ in feature_ranking[:10]]
            insights['feature_redundancy_analysis'] = {
                'top_discriminative_features': top_features,
                'feature_coverage_score': len(set(top_features)) / len(feature_ranking)
            }

        # Generate recommendations
        if insights['regime_discriminators']:
            recommendations = [
                "Focus feature engineering on dominant regime discriminators",
                "Consider dimensionality reduction based on feature importance ranking",
                "Validate clustering quality using regime stability indicators"
            ]
            insights['actionable_recommendations'] = recommendations

        return insights

    except Exception as e:
        logger.warning(f"⚠️ Failed to extract insights: {e}")
        return {'error': str(e)}
